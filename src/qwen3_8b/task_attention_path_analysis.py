#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from tqdm.auto import tqdm

import sys

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from artifact_paths import ARTIFACT_ROOT, QWEN3_8B_PATH, QWEN3_8B_TRANSCODER_PATH  # noqa: E402

from toolcall_circuit.single_sample import load_hooked_qwen3  # noqa: E402


SEED = 42
MODEL_PATH = QWEN3_8B_PATH
TRANSCODER_DIR = QWEN3_8B_TRANSCODER_PATH
DATASET_ROOT = ARTIFACT_ROOT / "datasets" / "test"
DIFF_ROOT = ARTIFACT_ROOT / "results" / "8b_main" / "differential_mechanism" / "differential_features"
ATTN_OUTPUT_ROOT = ARTIFACT_ROOT / "results" / "8b_main" / "l29_attention_pattern"
PATH_OUTPUT_ROOT = ARTIFACT_ROOT / "results" / "8b_main" / "l25_to_l29_path"
TOOL_CALL_STR = "<tool_call>"
HEADS = (9, 11, 14)
ATTN_LAYER = 29
PATH_LAYER_DEFAULT = 25
TARGET_HEAD = 9
PROGRESSIVE_K = (10, 25, 50, 100, 200)
REGIONS = ("verb", "schema", "task_desc", "system", "special", "other")


@dataclass
class Sample:
    sample_id: str
    clean_path: Path
    corrupt_path: Path
    clean_text: str
    corrupt_text: str
    clean_tokens_cpu: torch.Tensor
    corrupt_tokens_cpu: torch.Tensor
    clean_offsets: list[tuple[int, int]]
    corrupt_offsets: list[tuple[int, int]]
    clean_region_masks: dict[str, torch.Tensor]
    corrupt_region_masks: dict[str, torch.Tensor]
    clean_verb: str
    corrupt_verb: str


@dataclass
class PairBatch:
    indices: list[int]
    clean_tokens_cpu: torch.Tensor
    corrupt_tokens_cpu: torch.Tensor
    token_len: int


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def clear_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    ensure_dir(path.parent)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, data: Dict[str, object]) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    ensure_dir(path.parent)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def load_manifest_rows(manifest_path: Path) -> list[dict]:
    rows: list[dict] = []
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def overlap(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    return a_start < b_end and b_start < a_end


def find_all_spans(text: str, needle: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    cursor = 0
    while True:
        idx = text.find(needle, cursor)
        if idx < 0:
            return spans
        spans.append((idx, idx + len(needle)))
        cursor = idx + len(needle)


def locate_user_span(text: str) -> tuple[int, int]:
    user_marker = "<|im_start|>user\n"
    user_start = text.find(user_marker)
    if user_start < 0:
        raise ValueError("Could not locate user block.")
    start = user_start + len(user_marker)
    end = text.find("<|im_end|>", start)
    if end < 0:
        raise ValueError("Could not find end of user block.")
    return start, end


def locate_system_span(text: str) -> tuple[int, int]:
    system_marker = "<|im_start|>system\n"
    start = text.find(system_marker)
    if start < 0:
        raise ValueError("Could not locate system block.")
    end = text.find("<|im_end|>", start)
    if end < 0:
        raise ValueError("Could not find end of system block.")
    return start + len(system_marker), end


def locate_schema_span(text: str) -> tuple[int, int] | None:
    start = text.find('{"type":"function"')
    if start < 0:
        return None
    end = text.find("</tools>", start)
    if end < 0:
        return None
    return start, end


def locate_verb_span(text: str, verb: str) -> tuple[int, int]:
    user_start, user_end = locate_user_span(text)
    first_line_end = text.find("\n", user_start, user_end)
    if first_line_end < 0:
        first_line_end = user_end
    first_line = text[user_start:first_line_end]
    match = re.search(rf"\b{re.escape(verb)}\b", first_line, flags=re.IGNORECASE)
    if match is None:
        raise ValueError(f"Could not locate verb {verb!r} in first user line: {first_line!r}")
    return user_start + match.start(), user_start + match.end()


def build_region_masks(
    text: str,
    offsets: Sequence[tuple[int, int]],
    special_ids: set[int],
    input_ids: Sequence[int],
    verb: str,
) -> dict[str, torch.Tensor]:
    n_tokens = len(offsets)
    labels = np.array(["other"] * n_tokens, dtype=object)
    system_span = locate_system_span(text)
    user_span = locate_user_span(text)
    verb_span = locate_verb_span(text, verb)
    schema_span = locate_schema_span(text)
    special_spans = find_all_spans(text, "<|im_start|>") + find_all_spans(text, "<|im_end|>")

    def assign(span: tuple[int, int] | None, label: str) -> None:
        if span is None:
            return
        start, end = span
        for idx, (tok_start, tok_end) in enumerate(offsets):
            if tok_start == tok_end:
                continue
            if overlap(start, end, tok_start, tok_end):
                labels[idx] = label

    assign(system_span, "system")
    assign(user_span, "task_desc")
    assign(schema_span, "schema")
    assign(verb_span, "verb")
    for span in special_spans:
        assign(span, "special")
    for idx, token_id in enumerate(input_ids):
        if int(token_id) in special_ids:
            labels[idx] = "special"

    return {region: torch.tensor(labels == region, dtype=torch.bool) for region in REGIONS}


def load_samples(
    dataset_root: Path,
    model,
    tokenizer,
    *,
    max_pairs: int,
) -> list[Sample]:
    clean_rows = load_manifest_rows(dataset_root / "clean" / "manifest.jsonl")
    corrupt_map = {
        Path(str(row["output_filename"])).stem: row for row in load_manifest_rows(dataset_root / "corrupt" / "manifest.jsonl")
    }
    samples: list[Sample] = []
    special_ids = set(getattr(tokenizer, "all_special_ids", []))

    for row in clean_rows:
        sample_id = Path(str(row["output_filename"])).stem
        corrupt_row = corrupt_map.get(sample_id)
        if corrupt_row is None:
            continue
        clean_path = dataset_root / "clean" / str(row["output_filename"])
        corrupt_path = dataset_root / "corrupt" / str(corrupt_row["output_filename"])
        clean_text = clean_path.read_text(encoding="utf-8")
        corrupt_text = corrupt_path.read_text(encoding="utf-8")

        clean_tokens_cpu = model.to_tokens(clean_text, prepend_bos=False).detach().cpu()
        corrupt_tokens_cpu = model.to_tokens(corrupt_text, prepend_bos=False).detach().cpu()
        if int(clean_tokens_cpu.shape[-1]) != int(corrupt_tokens_cpu.shape[-1]):
            continue

        clean_enc = tokenizer(clean_text, add_special_tokens=False, return_offsets_mapping=True)
        corrupt_enc = tokenizer(corrupt_text, add_special_tokens=False, return_offsets_mapping=True)
        clean_offsets = [(int(a), int(b)) for a, b in clean_enc["offset_mapping"]]
        corrupt_offsets = [(int(a), int(b)) for a, b in corrupt_enc["offset_mapping"]]
        if len(clean_offsets) != int(clean_tokens_cpu.shape[-1]) or len(corrupt_offsets) != int(corrupt_tokens_cpu.shape[-1]):
            raise RuntimeError(f"Tokenizer offsets length mismatch for {sample_id}")

        clean_verb = str(row["clean_candidate"])
        corrupt_verb = str(corrupt_row["assigned_candidate"])
        clean_region_masks = build_region_masks(clean_text, clean_offsets, special_ids, clean_enc["input_ids"], clean_verb)
        corrupt_region_masks = build_region_masks(corrupt_text, corrupt_offsets, special_ids, corrupt_enc["input_ids"], corrupt_verb)

        samples.append(
            Sample(
                sample_id=sample_id,
                clean_path=clean_path,
                corrupt_path=corrupt_path,
                clean_text=clean_text,
                corrupt_text=corrupt_text,
                clean_tokens_cpu=clean_tokens_cpu,
                corrupt_tokens_cpu=corrupt_tokens_cpu,
                clean_offsets=clean_offsets,
                corrupt_offsets=corrupt_offsets,
                clean_region_masks=clean_region_masks,
                corrupt_region_masks=corrupt_region_masks,
                clean_verb=clean_verb,
                corrupt_verb=corrupt_verb,
            )
        )
        if len(samples) >= max_pairs:
            break
    if len(samples) < max_pairs:
        raise RuntimeError(f"Only loaded {len(samples)} equal-length test pairs from {dataset_root}")
    return samples


def build_pair_batches(samples: Sequence[Sample], batch_size: int) -> list[PairBatch]:
    buckets: dict[int, list[tuple[int, Sample]]] = defaultdict(list)
    for idx, sample in enumerate(samples):
        buckets[int(sample.clean_tokens_cpu.shape[-1])].append((idx, sample))

    batches: list[PairBatch] = []
    for token_len in sorted(buckets):
        group = buckets[token_len]
        for start in range(0, len(group), batch_size):
            chunk = group[start : start + batch_size]
            batches.append(
                PairBatch(
                    indices=[idx for idx, _sample in chunk],
                    clean_tokens_cpu=torch.cat([sample.clean_tokens_cpu for _, sample in chunk], dim=0),
                    corrupt_tokens_cpu=torch.cat([sample.corrupt_tokens_cpu for _, sample in chunk], dim=0),
                    token_len=token_len,
                )
            )
    return batches


def paired_t_pvalue(clean_vals: np.ndarray, corrupt_vals: np.ndarray) -> float:
    if clean_vals.shape != corrupt_vals.shape:
        raise ValueError("Paired arrays must match shape.")
    diff = clean_vals - corrupt_vals
    n = diff.shape[0]
    if n <= 1:
        return 1.0
    mean = float(diff.mean())
    std = float(diff.std(ddof=1))
    if std == 0.0:
        return 1.0
    t_val = mean / (std / math.sqrt(n))
    try:
        from scipy.stats import t as student_t  # type: ignore

        return float(2.0 * student_t.sf(abs(t_val), df=n - 1))
    except Exception:
        # Normal approximation is enough for summary stars when scipy is unavailable.
        return float(2.0 * 0.5 * math.erfc(abs(t_val) / math.sqrt(2.0)))


def p_to_stars(p_value: float) -> str:
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return ""


def resolve_pattern_head_idx(requested_head: int, pattern_head_count: int, n_heads: int) -> int:
    if requested_head < pattern_head_count:
        return requested_head
    if pattern_head_count <= 0:
        raise ValueError(f"Invalid pattern head count {pattern_head_count}")
    if n_heads % pattern_head_count != 0:
        raise ValueError(f"Could not map head {requested_head} with n_heads={n_heads}, pattern_heads={pattern_head_count}")
    return requested_head // (n_heads // pattern_head_count)


def run_attention_analysis(
    model,
    tokenizer,
    samples: Sequence[Sample],
    pair_batches: Sequence[PairBatch],
    output_root: Path,
) -> dict[str, object]:
    ensure_dir(output_root / "heatmaps")
    n_heads = int(model.cfg.n_heads)
    hook_name = f"blocks.{ATTN_LAYER}.attn.hook_pattern"
    per_sample_rows: list[Dict[str, object]] = []
    pattern_shape: list[int] | None = None

    progress = tqdm(pair_batches, desc="Experiment A attention", dynamic_ncols=True)
    for batch in progress:
        for condition in ("clean", "corrupt"):
            tokens_cpu = batch.clean_tokens_cpu if condition == "clean" else batch.corrupt_tokens_cpu
            tokens = tokens_cpu.to(model.W_U.device)
            with torch.no_grad():
                _, cache = model.run_with_cache(tokens, names_filter=lambda name: name == hook_name)
            pattern_batch = cache[hook_name].detach().cpu().float()
            if pattern_shape is None:
                pattern_shape = [int(x) for x in pattern_batch.shape]
            for local_idx, sample_idx in enumerate(batch.indices):
                sample = samples[sample_idx]
                region_masks = sample.clean_region_masks if condition == "clean" else sample.corrupt_region_masks
                pattern = pattern_batch[local_idx]
                last_pos = int(pattern.shape[-2]) - 1
                pattern_head_count = int(pattern.shape[0])
                for head in HEADS:
                    pattern_head = resolve_pattern_head_idx(head, pattern_head_count, n_heads)
                    weights = pattern[pattern_head, last_pos, :]
                    row: Dict[str, object] = {
                        "sample_id": sample.sample_id,
                        "condition": condition,
                        "layer": ATTN_LAYER,
                        "head": head,
                        "pattern_head": pattern_head,
                        "seq_len": int(pattern.shape[-1]),
                        "verb_label": sample.clean_verb if condition == "clean" else sample.corrupt_verb,
                    }
                    for region in REGIONS:
                        row[f"{region}_attn"] = float(weights[region_masks[region]].sum().item())
                    per_sample_rows.append(row)
            del tokens, cache, pattern_batch
            clear_cuda()

    summary_rows: list[Dict[str, object]] = []
    for head in HEADS:
        clean_rows = [row for row in per_sample_rows if int(row["head"]) == head and row["condition"] == "clean"]
        corrupt_rows = [row for row in per_sample_rows if int(row["head"]) == head and row["condition"] == "corrupt"]
        for region in REGIONS:
            clean_vals = np.asarray([float(row[f"{region}_attn"]) for row in clean_rows], dtype=np.float64)
            corrupt_vals = np.asarray([float(row[f"{region}_attn"]) for row in corrupt_rows], dtype=np.float64)
            p_value = paired_t_pvalue(clean_vals, corrupt_vals)
            summary_rows.append(
                {
                    "layer": ATTN_LAYER,
                    "head": head,
                    "region": region,
                    "clean_mean": float(clean_vals.mean()),
                    "clean_std": float(clean_vals.std(ddof=1)),
                    "corrupt_mean": float(corrupt_vals.mean()),
                    "corrupt_std": float(corrupt_vals.std(ddof=1)),
                    "delta_corrupt_minus_clean": float(corrupt_vals.mean() - clean_vals.mean()),
                    "p_value": p_value,
                    "significance": p_to_stars(p_value),
                    "n_pairs": int(clean_vals.shape[0]),
                }
            )

    write_csv(output_root / "attention_by_region.csv", summary_rows)
    write_csv(output_root / "attention_by_region_per_sample.csv", per_sample_rows)

    for sample in samples[:5]:
        for condition in ("clean", "corrupt"):
            text = sample.clean_text if condition == "clean" else sample.corrupt_text
            tokens_cpu = sample.clean_tokens_cpu if condition == "clean" else sample.corrupt_tokens_cpu
            tokens = tokens_cpu.to(model.W_U.device)
            region_masks = sample.clean_region_masks if condition == "clean" else sample.corrupt_region_masks
            with torch.no_grad():
                _, cache = model.run_with_cache(tokens, names_filter=lambda name: name == hook_name)
            pattern = cache[hook_name][0].detach().cpu().float()
            token_strs = [tokenizer.decode([int(tok)]) for tok in tokens_cpu[0]]
            last_pos = int(pattern.shape[-2]) - 1
            head_labels = []
            matrix_rows = []
            for head in HEADS:
                head_idx = resolve_pattern_head_idx(head, int(pattern.shape[0]), n_heads)
                matrix_rows.append(pattern[head_idx, last_pos, :].numpy())
                head_labels.append(f"H{head}")

            fig, ax = plt.subplots(figsize=(max(12, len(token_strs) * 0.18), 3.8))
            image = ax.imshow(np.asarray(matrix_rows), aspect="auto", cmap="viridis")
            fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02, label="Attention")
            ax.set_yticks(np.arange(len(head_labels)), head_labels)
            ax.set_xticks(np.arange(len(token_strs)))
            pretty_tokens = [tok.replace("\n", "\\n")[:12] for tok in token_strs]
            ax.set_xticklabels(pretty_tokens, rotation=90, fontsize=7)
            ax.set_title(f"{sample.sample_id} {condition}")
            region_colors = {
                "verb": "#d62828",
                "schema": "#1d3557",
                "task_desc": "#2a9d8f",
                "system": "#8d99ae",
                "special": "#f4a261",
            }
            for region, color in region_colors.items():
                indices = torch.nonzero(region_masks[region], as_tuple=False).flatten().tolist()
                if indices:
                    ax.scatter(indices, [-0.45] * len(indices), s=8, color=color, marker="s", clip_on=False)
            plt.tight_layout()
            plt.savefig(output_root / "heatmaps" / f"{sample.sample_id}_{condition}.png", dpi=180, bbox_inches="tight")
            plt.close(fig)
            del tokens, cache, pattern
            clear_cuda()

    by_head = defaultdict(list)
    for row in summary_rows:
        by_head[int(row["head"])].append(row)
    lines = [
        "# L29 Attention Pattern",
        "",
        f"- 日期: 2026-04-13",
        f"- 样本: `datasets/test` 前 {len(samples)} 对 clean/corrupt",
        f"- 头: L29H9 / H11 / H14",
        "",
    ]
    for head in HEADS:
        lines.append(f"## L29H{head}")
        lines.append("")
        lines.append("| 区域 | clean mean±std | corrupt mean±std | Δ(corrupt-clean) | p | |")
        lines.append("|---|---:|---:|---:|---:|---|")
        rows = sorted(by_head[head], key=lambda item: abs(float(item["delta_corrupt_minus_clean"])), reverse=True)
        for row in rows:
            lines.append(
                f"| {row['region']} | "
                f"{float(row['clean_mean']):.4f} ± {float(row['clean_std']):.4f} | "
                f"{float(row['corrupt_mean']):.4f} ± {float(row['corrupt_std']):.4f} | "
                f"{float(row['delta_corrupt_minus_clean']):+.4f} | "
                f"{float(row['p_value']):.3g} | {row['significance']} |"
            )
        lines.append("")

    h9 = {str(row["region"]): row for row in by_head[9]}
    verb_delta = float(h9["verb"]["delta_corrupt_minus_clean"])
    schema_delta = float(h9["schema"]["delta_corrupt_minus_clean"])
    if verb_delta > 0 and schema_delta < 0:
        verdict = "L29H9 在 corrupt 下对 verb 的注意力上升、对 schema 的注意力下降，偏向 attention 分布被 verb 吸引。"
    else:
        verdict = "L29H9 的区域注意力差异不支持单纯的 verb 吸引，value/residual 内容变化仍然需要保留。"
    lines.extend(["## 结论", "", f"- {verdict}"])
    write_text(output_root / "summary.md", "\n".join(lines))

    return {
        "n_pairs": len(samples),
        "hook_pattern_shape": pattern_shape,
    }


def get_w_o_layer(model, layer: int) -> torch.Tensor:
    if hasattr(model, "W_O"):
        return model.W_O[layer]
    attn = model.blocks[layer].attn
    if not hasattr(attn, "W_O"):
        raise AttributeError("W_O not found.")
    return attn.W_O.view(int(model.cfg.n_heads), int(model.cfg.d_head), int(model.cfg.d_model))


def precompute_head_projection(model, layer: int, head: int, tool_token_id: int) -> torch.Tensor:
    wu_tool = model.W_U[:, tool_token_id].to(device=model.W_U.device, dtype=torch.float32)
    w_o = get_w_o_layer(model, layer).to(device=model.W_U.device, dtype=torch.float32)
    return torch.einsum("de,e->d", w_o[head], wu_tool)


def collect_l29h9_dla(
    model,
    pair_batches: Sequence[PairBatch],
    *,
    tool_token_id: int,
    hooks_factory=None,
) -> dict[str, torch.Tensor]:
    head_proj = precompute_head_projection(model, ATTN_LAYER, TARGET_HEAD, tool_token_id)
    hook_name = f"blocks.{ATTN_LAYER}.attn.hook_z"
    outputs = {
        "clean": torch.empty(sum(int(batch.clean_tokens_cpu.shape[0]) for batch in pair_batches), dtype=torch.float32),
        "corrupt": torch.empty(sum(int(batch.corrupt_tokens_cpu.shape[0]) for batch in pair_batches), dtype=torch.float32),
    }

    for condition in ("clean", "corrupt"):
        progress = tqdm(pair_batches, desc=f"DLA {condition}", dynamic_ncols=True)
        for batch in progress:
            tokens_cpu = batch.clean_tokens_cpu if condition == "clean" else batch.corrupt_tokens_cpu
            tokens = tokens_cpu.to(model.W_U.device)
            extra_hooks = hooks_factory(batch, condition) if hooks_factory is not None else []
            capture: dict[str, torch.Tensor] = {}

            def z_hook(value: torch.Tensor, hook):  # noqa: ANN001
                capture["z"] = value[:, -1, TARGET_HEAD, :].detach()
                return value

            hooks = list(extra_hooks) + [(hook_name, z_hook)]
            with torch.no_grad():
                _ = model.run_with_hooks(tokens, fwd_hooks=hooks)
            z = capture["z"].to(dtype=torch.float32)
            dla = torch.einsum("bd,d->b", z, head_proj).detach().cpu()
            outputs[condition][batch.indices] = dla
            del tokens, hooks, capture, z, dla
            clear_cuda()
    return outputs


def load_feature_ids(path: Path, category: str, k: int) -> list[int]:
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["category"] == category:
                rows.append(row)
    if category == "clean-selective":
        rows.sort(key=lambda row: float(row["delta"]), reverse=True)
    else:
        rows.sort(key=lambda row: float(row["delta"]))
    return [int(row["feature_idx"]) for row in rows[:k]]


def build_path_ablation_hooks_factory(
    *,
    path_layer: int,
    tc_weights: dict[str, torch.Tensor],
    feature_ids: Sequence[int],
):
    device_cache: dict[str, torch.Tensor] = {}
    state: dict[str, torch.Tensor] = {}
    feature_idx = torch.tensor(list(feature_ids), dtype=torch.long)

    def hooks_factory(_batch: PairBatch, _condition: str):
        if not device_cache:
            device = tc_weights["W_enc"].device
            idx = feature_idx.to(device)
            device_cache["W_enc"] = tc_weights["W_enc"][idx].to(dtype=torch.bfloat16)
            device_cache["b_enc"] = tc_weights["b_enc"][idx].to(dtype=torch.bfloat16)
            device_cache["W_dec"] = tc_weights["W_dec"][idx].to(dtype=torch.bfloat16)

        def in_hook(value: torch.Tensor, hook):  # noqa: ANN001
            state["mlp_in"] = value[:, -1, :].detach()
            return value

        def out_hook(value: torch.Tensor, hook):  # noqa: ANN001
            mlp_in = state.pop("mlp_in")
            acts = torch.relu(F.linear(mlp_in.to(dtype=torch.bfloat16), device_cache["W_enc"], device_cache["b_enc"]))
            contrib = acts @ device_cache["W_dec"]
            out = value.clone()
            out[:, -1, :] = out[:, -1, :] - contrib.to(dtype=out.dtype)
            return out

        return [
            (f"blocks.{path_layer}.hook_mlp_in", in_hook),
            (f"blocks.{path_layer}.hook_mlp_out", out_hook),
        ]

    return hooks_factory


def summarize_array(values: torch.Tensor) -> tuple[float, float]:
    arr = values.detach().cpu().numpy()
    return float(arr.mean()), float(arr.std(ddof=1))


def run_path_analysis(
    model,
    samples: Sequence[Sample],
    pair_batches: Sequence[PairBatch],
    *,
    path_layer: int,
    tool_token_id: int,
    output_root: Path,
    transcoder_dir: Path,
    diff_root: Path,
) -> dict[str, object]:
    ensure_dir(output_root)
    tc_weights_raw = load_file(str(transcoder_dir / f"layer_{path_layer}.safetensors"))
    tc_weights = {key: value.to(model.W_U.device) for key, value in tc_weights_raw.items()}
    diff_csv = diff_root / f"differential_features_L{path_layer}.csv"
    clean_top50 = load_feature_ids(diff_csv, "clean-selective", 50)
    corrupt_top50 = load_feature_ids(diff_csv, "corrupt-selective", 50)

    baseline = collect_l29h9_dla(model, pair_batches, tool_token_id=tool_token_id)
    clean_mean, clean_std = summarize_array(baseline["clean"])
    corrupt_mean, corrupt_std = summarize_array(baseline["corrupt"])

    rows: list[Dict[str, object]] = [
        {
            "condition": "baseline_clean",
            "sample_side": "clean",
            "feature_set": "none",
            "k": 0,
            "mean_dla": clean_mean,
            "std_dla": clean_std,
            "delta_vs_baseline": 0.0,
        },
        {
            "condition": "baseline_corrupt",
            "sample_side": "corrupt",
            "feature_set": "none",
            "k": 0,
            "mean_dla": corrupt_mean,
            "std_dla": corrupt_std,
            "delta_vs_baseline": 0.0,
        },
    ]

    condition_specs = [
        (f"ablate_L{path_layer}_clean_top50_on_clean", "clean", "clean-selective", clean_top50),
        (f"ablate_L{path_layer}_clean_top50_on_corrupt", "corrupt", "clean-selective", clean_top50),
        (f"ablate_L{path_layer}_corrupt_top50_on_corrupt", "corrupt", "corrupt-selective", corrupt_top50),
        (f"ablate_L{path_layer}_corrupt_top50_on_clean", "clean", "corrupt-selective", corrupt_top50),
    ]

    progressive_rows: list[Dict[str, object]] = []
    all_condition_outputs: dict[str, torch.Tensor] = {}
    for condition_name, side, feature_set, feature_ids in condition_specs:
        hook_factory = build_path_ablation_hooks_factory(path_layer=path_layer, tc_weights=tc_weights, feature_ids=feature_ids)
        result = collect_l29h9_dla(model, pair_batches, tool_token_id=tool_token_id, hooks_factory=hook_factory)
        values = result[side]
        all_condition_outputs[condition_name] = values
        baseline_values = baseline[side]
        rows.append(
            {
                "condition": condition_name,
                "sample_side": side,
                "feature_set": feature_set,
                "k": len(feature_ids),
                "mean_dla": float(values.mean().item()),
                "std_dla": float(values.std(unbiased=True).item()),
                "delta_vs_baseline": float((values - baseline_values).mean().item()),
            }
        )

    for feature_set, side, pool in (
        ("clean-selective", "clean", load_feature_ids(diff_csv, "clean-selective", max(PROGRESSIVE_K))),
        ("corrupt-selective", "corrupt", load_feature_ids(diff_csv, "corrupt-selective", max(PROGRESSIVE_K))),
    ):
        for k in PROGRESSIVE_K:
            hook_factory = build_path_ablation_hooks_factory(path_layer=path_layer, tc_weights=tc_weights, feature_ids=pool[:k])
            result = collect_l29h9_dla(model, pair_batches, tool_token_id=tool_token_id, hooks_factory=hook_factory)
            values = result[side]
            progressive_rows.append(
                {
                    "feature_set": feature_set,
                    "sample_side": side,
                    "k": k,
                    "mean_dla": float(values.mean().item()),
                    "std_dla": float(values.std(unbiased=True).item()),
                    "delta_vs_baseline": float((values - baseline[side]).mean().item()),
                }
            )

    write_csv(output_root / "ablation_results.csv", rows)
    write_csv(output_root / "progressive_k.csv", progressive_rows)

    verdict_parts: list[str] = []
    lookup = {str(row["condition"]): row for row in rows}
    clean_drop = float(lookup[f"ablate_L{path_layer}_clean_top50_on_clean"]["delta_vs_baseline"])
    corrupt_from_clean = float(lookup[f"ablate_L{path_layer}_clean_top50_on_corrupt"]["delta_vs_baseline"])
    corrupt_drop = float(lookup[f"ablate_L{path_layer}_corrupt_top50_on_corrupt"]["delta_vs_baseline"])
    clean_from_corrupt = float(lookup[f"ablate_L{path_layer}_corrupt_top50_on_clean"]["delta_vs_baseline"])
    if clean_drop < -0.05:
        verdict_parts.append(
            f"ablate clean-selective L{path_layer} features on clean 会显著拉低 L29H9 DLA，支持 L{path_layer} clean features 是上游。"
        )
    if abs(corrupt_from_clean) < abs(clean_drop):
        verdict_parts.append("同一组 clean-selective features 在 corrupt 上作用更弱，说明 corrupt 状态下这条 clean path 本来就弱。")
    if corrupt_drop > 0.05:
        verdict_parts.append(
            f"ablate corrupt-selective L{path_layer} features on corrupt 会抬高 L29H9 DLA，说明这组 feature 对 H9 带抑制作用。"
        )
    elif corrupt_drop < -0.05:
        verdict_parts.append(
            f"ablate corrupt-selective L{path_layer} features on corrupt 反而进一步压低 L29H9 DLA，说明它们可能也是 L29H9 的上游载体而不是纯抑制器。"
        )
    if not verdict_parts:
        verdict_parts.append(f"L{path_layer} feature ablation 对 L29H9 DLA 影响有限，暂时更像并行路径。")
    if abs(clean_from_corrupt) > 0.05:
        verdict_parts.append("corrupt-selective features 在 clean 上也有可测影响，说明该路径并非完全条件独占。")

    lines = [
        f"# L{path_layer} -> L29H9 Path Patching",
        "",
        f"- baseline clean DLA: {clean_mean:.4f} ± {clean_std:.4f}",
        f"- baseline corrupt DLA: {corrupt_mean:.4f} ± {corrupt_std:.4f}",
        f"- baseline delta (clean - corrupt): {clean_mean - corrupt_mean:+.4f}",
        "",
        "## Main Ablations",
        "",
        "| 条件 | mean DLA | std | Δ vs baseline |",
        "|---|---:|---:|---:|",
    ]
    for row in rows[2:]:
        lines.append(
            f"| {row['condition']} | {float(row['mean_dla']):.4f} | "
            f"{float(row['std_dla']):.4f} | {float(row['delta_vs_baseline']):+.4f} |"
        )
    lines.extend(["", "## 结论", ""])
    for part in verdict_parts:
        lines.append(f"- {part}")
    write_text(output_root / "summary.md", "\n".join(lines))

    return {
        "baseline_clean_mean": clean_mean,
        "baseline_corrupt_mean": corrupt_mean,
        "n_pairs": len(samples),
        "clean_top50_count": len(clean_top50),
        "corrupt_top50_count": len(corrupt_top50),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Task experiments for L29 attention pattern and L25->L29 path")
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH)
    parser.add_argument("--transcoder-dir", type=Path, default=TRANSCODER_DIR)
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--diff-root", type=Path, default=DIFF_ROOT)
    parser.add_argument("--attention-output-root", type=Path, default=ATTN_OUTPUT_ROOT)
    parser.add_argument("--path-output-root", type=Path, default=PATH_OUTPUT_ROOT)
    parser.add_argument("--mode", choices=("all", "attention_only", "path_only"), default="all")
    parser.add_argument("--path-layer", type=int, default=PATH_LAYER_DEFAULT)
    parser.add_argument("--max-pairs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    if args.mode in {"all", "attention_only"}:
        ensure_dir(args.attention_output_root)
    if args.mode in {"all", "path_only"}:
        ensure_dir(args.path_output_root)

    model, tokenizer = load_hooked_qwen3(str(args.model_path), device=args.device, dtype=torch.bfloat16)
    if hasattr(model, "set_use_hook_mlp_in"):
        model.set_use_hook_mlp_in(True)
    if hasattr(model, "cfg") and hasattr(model.cfg, "use_hook_mlp_in"):
        model.cfg.use_hook_mlp_in = True

    tool_token_ids = tokenizer.encode(TOOL_CALL_STR, add_special_tokens=False)
    if len(tool_token_ids) != 1:
        raise ValueError(f"{TOOL_CALL_STR!r} does not map to one token: {tool_token_ids}")
    tool_token_id = int(tool_token_ids[0])

    samples = load_samples(args.dataset_root, model, tokenizer, max_pairs=args.max_pairs)
    pair_batches = build_pair_batches(samples, args.batch_size)

    if args.mode in {"all", "attention_only"}:
        attn_meta = run_attention_analysis(model, tokenizer, samples, pair_batches, args.attention_output_root)
        write_json(
            args.attention_output_root / "metadata.json",
            {
                "seed": args.seed,
                "model_path": str(args.model_path),
                "dataset_root": str(args.dataset_root),
                "max_pairs": args.max_pairs,
                "batch_size": args.batch_size,
                **attn_meta,
            },
        )

    if args.mode in {"all", "path_only"}:
        path_meta = run_path_analysis(
            model,
            samples,
            pair_batches,
            path_layer=args.path_layer,
            tool_token_id=tool_token_id,
            output_root=args.path_output_root,
            transcoder_dir=args.transcoder_dir,
            diff_root=args.diff_root,
        )
        write_json(
            args.path_output_root / "metadata.json",
            {
                "seed": args.seed,
                "model_path": str(args.model_path),
                "transcoder_dir": str(args.transcoder_dir),
                "dataset_root": str(args.dataset_root),
                "diff_root": str(args.diff_root),
                "max_pairs": args.max_pairs,
                "batch_size": args.batch_size,
                "path_layer": args.path_layer,
                **path_meta,
            },
        )

    del model
    clear_cuda()


if __name__ == "__main__":
    main()
