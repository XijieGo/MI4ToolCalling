#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm.auto import tqdm

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from toolcall_circuit.single_sample import load_hooked_qwen3  # noqa: E402
from artifact_paths import ARTIFACT_ROOT, QWEN3_8B_PATH  # noqa: E402


SEED = 42
MODEL_PATH = QWEN3_8B_PATH
MANIFEST_PATH = ARTIFACT_ROOT / "results" / "8b_main" / "triplet_analysis" / "sample_manifest.csv"
DATASET_ROOT = ARTIFACT_ROOT / "datasets" / "train"
OUTPUT_ROOT = ARTIFACT_ROOT / "results" / "8b_main" / "attention_analysis"
TOOL_CALL_STR = "<tool_call>"
DEFAULT_LAYERS = tuple(range(25, 36))


@dataclass
class SamplePair:
    order: int
    sample_id: str
    clean_path: Path
    corrupt_path: Path
    clean_tokens_cpu: torch.Tensor
    corrupt_tokens_cpu: torch.Tensor
    token_len: int
    diff_positions: list[int]
    clean_candidate: str | None
    corrupt_candidate: str | None


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


def warn(message: str) -> None:
    print(f"[warning] {message}", flush=True)


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_condition_manifest_map(path: Path, condition: str) -> dict[str, dict]:
    manifest_path = path / condition / "manifest.jsonl"
    rows: dict[str, dict] = {}
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            filename = str(row.get("output_filename") or row.get("source_filename") or "")
            if not filename:
                continue
            rows[Path(filename).stem] = row
    if not rows:
        raise RuntimeError(f"No rows loaded from {manifest_path}")
    return rows


def load_sample_pairs(
    manifest_path: Path,
    model,
    clean_meta: dict[str, dict],
    corrupt_meta: dict[str, dict],
    *,
    max_pairs: int,
) -> list[SamplePair]:
    pairs: list[SamplePair] = []
    with manifest_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            clean_path = Path(row["clean_path"])
            corrupt_path = Path(row["corrupt_path"])
            clean_text = clean_path.read_text(encoding="utf-8")
            corrupt_text = corrupt_path.read_text(encoding="utf-8")
            clean_tokens_cpu = model.to_tokens(clean_text, prepend_bos=False).detach().cpu()
            corrupt_tokens_cpu = model.to_tokens(corrupt_text, prepend_bos=False).detach().cpu()
            clean_len = int(clean_tokens_cpu.shape[-1])
            corrupt_len = int(corrupt_tokens_cpu.shape[-1])
            if clean_len != corrupt_len:
                continue

            diff_mask = clean_tokens_cpu[0] != corrupt_tokens_cpu[0]
            diff_positions = torch.nonzero(diff_mask, as_tuple=False).flatten().tolist()
            if not diff_positions:
                warn(f"{row['sample_id']} has no token-level diff; skipping.")
                continue

            sample_id = str(row["sample_id"])
            clean_row = clean_meta.get(sample_id, {})
            corrupt_row = corrupt_meta.get(sample_id, {})
            pairs.append(
                SamplePair(
                    order=int(row["order"]),
                    sample_id=sample_id,
                    clean_path=clean_path,
                    corrupt_path=corrupt_path,
                    clean_tokens_cpu=clean_tokens_cpu,
                    corrupt_tokens_cpu=corrupt_tokens_cpu,
                    token_len=clean_len,
                    diff_positions=diff_positions,
                    clean_candidate=str(clean_row.get("clean_candidate")) if clean_row.get("clean_candidate") is not None else None,
                    corrupt_candidate=str(
                        corrupt_row.get("assigned_candidate")
                        or clean_row.get("corrupt_candidate")
                    )
                    if (corrupt_row.get("assigned_candidate") is not None or clean_row.get("corrupt_candidate") is not None)
                    else None,
                )
            )
            if len(pairs) >= max_pairs:
                break
    if len(pairs) < max_pairs:
        raise RuntimeError(f"Only loaded {len(pairs)} usable equal-length pairs from {manifest_path}")
    return pairs


def build_pair_batches(pairs: Sequence[SamplePair], batch_size: int) -> list[PairBatch]:
    buckets: dict[int, list[tuple[int, SamplePair]]] = defaultdict(list)
    for idx, pair in enumerate(pairs):
        buckets[pair.token_len].append((idx, pair))

    batches: list[PairBatch] = []
    for token_len in sorted(buckets):
        group = buckets[token_len]
        for start in range(0, len(group), batch_size):
            chunk = group[start : start + batch_size]
            indices = [idx for idx, _pair in chunk]
            clean_tokens_cpu = torch.cat([pair.clean_tokens_cpu for _, pair in chunk], dim=0)
            corrupt_tokens_cpu = torch.cat([pair.corrupt_tokens_cpu for _, pair in chunk], dim=0)
            batches.append(
                PairBatch(
                    indices=indices,
                    clean_tokens_cpu=clean_tokens_cpu,
                    corrupt_tokens_cpu=corrupt_tokens_cpu,
                    token_len=token_len,
                )
            )
    return batches


def tool_stats(logits: torch.Tensor, tool_token_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    last_logits = logits[:, -1, :]
    tool_logit = last_logits[:, tool_token_id].detach().cpu().float()
    top1 = last_logits.argmax(dim=-1).detach().cpu()
    return tool_logit, top1


def get_w_o_layer(model, layer: int) -> torch.Tensor:
    if hasattr(model, "W_O"):
        return model.W_O[layer]
    attn = model.blocks[layer].attn
    if not hasattr(attn, "W_O"):
        raise AttributeError("Could not locate W_O on model or attention block.")
    w_o = attn.W_O
    n_heads = int(model.cfg.n_heads)
    d_head = int(model.cfg.d_head)
    d_model = int(model.cfg.d_model)
    return w_o.view(n_heads, d_head, d_model)


def precompute_head_tool_projections(model, layers: Sequence[int], tool_token_id: int) -> dict[int, torch.Tensor]:
    wu_tool = model.W_U[:, tool_token_id].to(device=model.W_U.device, dtype=torch.float32)
    projections: dict[int, torch.Tensor] = {}
    for layer in layers:
        w_o = get_w_o_layer(model, layer).to(device=model.W_U.device, dtype=torch.float32)
        projections[layer] = torch.einsum("hde,e->hd", w_o, wu_tool)
    return projections


def analyze_dla(
    model,
    pair_batches: Sequence[PairBatch],
    *,
    layers: Sequence[int],
    tool_token_id: int,
) -> tuple[list[Dict[str, object]], list[Dict[str, object]], dict[str, object]]:
    n_heads = int(model.cfg.n_heads)
    head_tool_proj = precompute_head_tool_projections(model, layers, tool_token_id)
    hook_names = [f"blocks.{layer}.attn.hook_z" for layer in layers]

    sums = {
        "clean": {layer: torch.zeros(n_heads, dtype=torch.float64) for layer in layers},
        "corrupt": {layer: torch.zeros(n_heads, dtype=torch.float64) for layer in layers},
    }
    sums_sq = {
        "clean": {layer: torch.zeros(n_heads, dtype=torch.float64) for layer in layers},
        "corrupt": {layer: torch.zeros(n_heads, dtype=torch.float64) for layer in layers},
    }
    counts = {"clean": 0, "corrupt": 0}
    hook_z_shape: list[int] | None = None

    progress = tqdm(pair_batches, desc="Attention DLA", dynamic_ncols=True)
    for batch in progress:
        for condition in ("clean", "corrupt"):
            tokens_cpu = batch.clean_tokens_cpu if condition == "clean" else batch.corrupt_tokens_cpu
            tokens = tokens_cpu.to(model.W_U.device)
            with torch.no_grad():
                _, cache = model.run_with_cache(tokens, names_filter=lambda name: name in hook_names)
            if hook_z_shape is None:
                first_key = hook_names[0]
                hook_z_shape = [int(x) for x in cache[first_key].shape]
                print(f"[sanity] {first_key} shape: {tuple(hook_z_shape)}", flush=True)
            batch_size = int(tokens.shape[0])
            counts[condition] += batch_size
            for layer in layers:
                z_last = cache[f"blocks.{layer}.attn.hook_z"][:, -1, :, :].to(dtype=torch.float32)
                dla = torch.einsum("bhd,hd->bh", z_last, head_tool_proj[layer]).detach().cpu().double()
                sums[condition][layer] += dla.sum(dim=0)
                sums_sq[condition][layer] += (dla * dla).sum(dim=0)
            del tokens, cache
            clear_cuda()
        progress.set_postfix(last=batch.indices[-1], tok=batch.token_len)

    rows: list[Dict[str, object]] = []
    layer_rows: list[Dict[str, object]] = []
    for layer in layers:
        layer_entries: list[Dict[str, object]] = []
        for head in range(n_heads):
            clean_mean = float((sums["clean"][layer][head] / max(counts["clean"], 1)).item())
            corrupt_mean = float((sums["corrupt"][layer][head] / max(counts["corrupt"], 1)).item())
            clean_var = float(
                max(
                    (sums_sq["clean"][layer][head] / max(counts["clean"], 1)).item() - clean_mean * clean_mean,
                    0.0,
                )
            )
            corrupt_var = float(
                max(
                    (sums_sq["corrupt"][layer][head] / max(counts["corrupt"], 1)).item() - corrupt_mean * corrupt_mean,
                    0.0,
                )
            )
            row = {
                "layer": layer,
                "head": head,
                "mean_clean": clean_mean,
                "mean_corrupt": corrupt_mean,
                "delta": clean_mean - corrupt_mean,
                "abs_delta": abs(clean_mean - corrupt_mean),
                "std_clean": math.sqrt(clean_var),
                "std_corrupt": math.sqrt(corrupt_var),
                "n_clean": counts["clean"],
                "n_corrupt": counts["corrupt"],
            }
            rows.append(row)
            layer_entries.append(row)

        layer_entries_sorted = sorted(layer_entries, key=lambda item: abs(float(item["delta"])), reverse=True)
        pos_best = max(layer_entries, key=lambda item: float(item["delta"]))
        neg_best = min(layer_entries, key=lambda item: float(item["delta"]))
        layer_rows.append(
            {
                "layer": layer,
                "max_abs_delta": float(layer_entries_sorted[0]["abs_delta"]),
                "top_head_by_abs": int(layer_entries_sorted[0]["head"]),
                "top_head_delta": float(layer_entries_sorted[0]["delta"]),
                "top_positive_head": int(pos_best["head"]),
                "top_positive_delta": float(pos_best["delta"]),
                "top_negative_head": int(neg_best["head"]),
                "top_negative_delta": float(neg_best["delta"]),
                "mean_abs_delta": float(np.mean([float(item["abs_delta"]) for item in layer_entries])),
                "n_heads_abs_delta_gt_1": int(sum(float(item["abs_delta"]) > 1.0 for item in layer_entries)),
            }
        )

    rows.sort(key=lambda item: (float(item["abs_delta"]), -float(item["delta"])), reverse=True)
    layer_rows.sort(key=lambda item: int(item["layer"]))
    metadata = {
        "hook_z_shape": hook_z_shape,
        "n_clean": counts["clean"],
        "n_corrupt": counts["corrupt"],
    }
    return rows, layer_rows, metadata


def select_heads_for_attention(
    dla_rows: Sequence[Dict[str, object]],
    *,
    layers: Sequence[int],
    heads_per_layer: int,
    global_top_k: int,
) -> tuple[list[tuple[int, int]], list[Dict[str, object]]]:
    rows_by_layer: dict[int, list[Dict[str, object]]] = defaultdict(list)
    for row in dla_rows:
        rows_by_layer[int(row["layer"])].append(dict(row))

    selected: list[tuple[int, int]] = []
    selected_meta: list[Dict[str, object]] = []
    for layer in layers:
        layer_rows = sorted(rows_by_layer[layer], key=lambda item: float(item["abs_delta"]), reverse=True)[:heads_per_layer]
        for rank, row in enumerate(layer_rows, start=1):
            pair = (int(row["layer"]), int(row["head"]))
            if pair not in selected:
                selected.append(pair)
                selected_meta.append(
                    {
                        "selection": "per_layer",
                        "layer_rank": rank,
                        **row,
                    }
                )

    for global_rank, row in enumerate(sorted(dla_rows, key=lambda item: float(item["abs_delta"]), reverse=True)[:global_top_k], start=1):
        pair = (int(row["layer"]), int(row["head"]))
        if pair not in selected:
            selected.append(pair)
            selected_meta.append(
                {
                    "selection": "global",
                    "global_rank": global_rank,
                    **row,
                }
            )
    selected.sort()
    return selected, selected_meta


def resolve_pattern_head_idx(requested_head: int, pattern_head_count: int, n_heads: int) -> int:
    if requested_head < pattern_head_count:
        return requested_head
    if pattern_head_count <= 0:
        raise ValueError(f"Invalid pattern_head_count={pattern_head_count}")
    if n_heads % pattern_head_count == 0:
        group_size = n_heads // pattern_head_count
        return requested_head // group_size
    raise ValueError(
        f"Cannot map requested head {requested_head} to pattern head count {pattern_head_count} (n_heads={n_heads})"
    )


def analyze_verb_attention(
    model,
    pairs: Sequence[SamplePair],
    pair_batches: Sequence[PairBatch],
    *,
    selected_heads: Sequence[tuple[int, int]],
) -> tuple[list[Dict[str, object]], list[Dict[str, object]], dict[str, object]]:
    n_heads = int(model.cfg.n_heads)
    heads_by_layer: dict[int, list[int]] = defaultdict(list)
    for layer, head in selected_heads:
        heads_by_layer[layer].append(head)
    hook_names = [f"blocks.{layer}.attn.hook_pattern" for layer in sorted(heads_by_layer)]

    rows: list[Dict[str, object]] = []
    pattern_shape: list[int] | None = None
    progress = tqdm(pair_batches, desc="Verb attention", dynamic_ncols=True)
    for batch in progress:
        for condition in ("clean", "corrupt"):
            tokens_cpu = batch.clean_tokens_cpu if condition == "clean" else batch.corrupt_tokens_cpu
            tokens = tokens_cpu.to(model.W_U.device)
            with torch.no_grad():
                _, cache = model.run_with_cache(tokens, names_filter=lambda name: name in hook_names)
            if pattern_shape is None:
                first_key = hook_names[0]
                pattern_shape = [int(x) for x in cache[first_key].shape]
                print(f"[sanity] {first_key} shape: {tuple(pattern_shape)}", flush=True)
            for local_idx, pair_idx in enumerate(batch.indices):
                pair = pairs[pair_idx]
                diff_positions = pair.diff_positions
                diff_start = int(diff_positions[0])
                diff_end = int(diff_positions[-1])
                diff_count = int(len(diff_positions))
                for layer, requested_heads in heads_by_layer.items():
                    pattern = cache[f"blocks.{layer}.attn.hook_pattern"][local_idx].detach().cpu().float()
                    seq_len = int(pattern.shape[-1])
                    last_pos = seq_len - 1
                    pattern_head_count = int(pattern.shape[0])
                    for head in requested_heads:
                        pattern_head = resolve_pattern_head_idx(head, pattern_head_count, n_heads)
                        attn_to_verb = float(pattern[pattern_head, last_pos, diff_start].item())
                        attn_to_diff_span = float(pattern[pattern_head, last_pos, diff_positions].sum().item())
                        rows.append(
                            {
                                "sample_id": pair.sample_id,
                                "condition": condition,
                                "layer": layer,
                                "head": head,
                                "pattern_head": pattern_head,
                                "attn_to_verb": attn_to_verb,
                                "attn_to_diff_span": attn_to_diff_span,
                                "diff_start": diff_start,
                                "diff_end": diff_end,
                                "diff_count": diff_count,
                                "seq_len": seq_len,
                                "clean_candidate": pair.clean_candidate,
                                "corrupt_candidate": pair.corrupt_candidate,
                            }
                        )
            del tokens, cache
            clear_cuda()
        progress.set_postfix(last=batch.indices[-1], tok=batch.token_len)

    grouped: dict[tuple[int, int, str], list[float]] = defaultdict(list)
    grouped_span: dict[tuple[int, int, str], list[float]] = defaultdict(list)
    for row in rows:
        key = (int(row["layer"]), int(row["head"]), str(row["condition"]))
        grouped[key].append(float(row["attn_to_verb"]))
        grouped_span[key].append(float(row["attn_to_diff_span"]))

    summary_rows: list[Dict[str, object]] = []
    for layer, head in selected_heads:
        clean_vals = grouped[(layer, head, "clean")]
        corrupt_vals = grouped[(layer, head, "corrupt")]
        clean_span = grouped_span[(layer, head, "clean")]
        corrupt_span = grouped_span[(layer, head, "corrupt")]
        summary_rows.append(
            {
                "layer": layer,
                "head": head,
                "mean_clean_attn_to_verb": float(np.mean(clean_vals)),
                "mean_corrupt_attn_to_verb": float(np.mean(corrupt_vals)),
                "delta_attn_to_verb": float(np.mean(clean_vals) - np.mean(corrupt_vals)),
                "mean_clean_attn_to_diff_span": float(np.mean(clean_span)),
                "mean_corrupt_attn_to_diff_span": float(np.mean(corrupt_span)),
                "delta_attn_to_diff_span": float(np.mean(clean_span) - np.mean(corrupt_span)),
                "n_pairs": len(clean_vals),
            }
        )
    summary_rows.sort(key=lambda item: abs(float(item["delta_attn_to_verb"])), reverse=True)

    metadata = {
        "hook_pattern_shape": pattern_shape,
        "n_pairs": len(pairs),
        "selected_heads": [{"layer": layer, "head": head} for layer, head in selected_heads],
    }
    return rows, summary_rows, metadata


def make_multihead_zero_hook(batch_size: int, n_heads: int):
    def hook_fn(value: torch.Tensor, hook):  # noqa: ANN001
        out = value.clone()
        for head in range(n_heads):
            start = head * batch_size
            end = start + batch_size
            out[start:end, :, head, :] = 0
        return out

    return hook_fn


def analyze_head_ablation(
    model,
    pair_batches: Sequence[PairBatch],
    *,
    layers: Sequence[int],
    tool_token_id: int,
) -> tuple[list[Dict[str, object]], dict[str, object]]:
    n_heads = int(model.cfg.n_heads)
    accum = {
        layer: {
            "clean_delta_sum": torch.zeros(n_heads, dtype=torch.float64),
            "corrupt_delta_sum": torch.zeros(n_heads, dtype=torch.float64),
            "clean_flip_count": torch.zeros(n_heads, dtype=torch.int64),
            "corrupt_gain_count": torch.zeros(n_heads, dtype=torch.int64),
            "clean_count": 0,
            "corrupt_count": 0,
        }
        for layer in layers
    }
    clean_tool_top1 = 0
    corrupt_tool_top1 = 0
    clean_total = 0
    corrupt_total = 0

    progress = tqdm(pair_batches, desc="Head ablation", dynamic_ncols=True)
    for batch in progress:
        for condition in ("clean", "corrupt"):
            tokens_cpu = batch.clean_tokens_cpu if condition == "clean" else batch.corrupt_tokens_cpu
            tokens = tokens_cpu.to(model.W_U.device)
            with torch.no_grad():
                base_logits = model(tokens)
            base_tool_logit, base_top1 = tool_stats(base_logits, tool_token_id)
            batch_size = int(tokens.shape[0])
            if condition == "clean":
                clean_total += batch_size
                clean_tool_top1 += int((base_top1 == tool_token_id).sum().item())
            else:
                corrupt_total += batch_size
                corrupt_tool_top1 += int((base_top1 == tool_token_id).sum().item())

            repeated_tokens = tokens.repeat((n_heads, 1))
            for layer in layers:
                hook_name = f"blocks.{layer}.attn.hook_z"
                hook_fn = make_multihead_zero_hook(batch_size, n_heads)
                with torch.no_grad():
                    abl_logits = model.run_with_hooks(repeated_tokens, fwd_hooks=[(hook_name, hook_fn)])
                abl_tool_logit, abl_top1 = tool_stats(abl_logits, tool_token_id)
                abl_tool_logit = abl_tool_logit.view(n_heads, batch_size)
                abl_top1 = abl_top1.view(n_heads, batch_size)
                delta = abl_tool_logit.double() - base_tool_logit.unsqueeze(0).double()
                if condition == "clean":
                    flips = ((base_top1 == tool_token_id).unsqueeze(0) & (abl_top1 != tool_token_id)).to(torch.int64)
                    accum[layer]["clean_delta_sum"] += delta.sum(dim=1).cpu()
                    accum[layer]["clean_flip_count"] += flips.sum(dim=1).cpu()
                    accum[layer]["clean_count"] += batch_size
                else:
                    gains = ((base_top1 != tool_token_id).unsqueeze(0) & (abl_top1 == tool_token_id)).to(torch.int64)
                    accum[layer]["corrupt_delta_sum"] += delta.sum(dim=1).cpu()
                    accum[layer]["corrupt_gain_count"] += gains.sum(dim=1).cpu()
                    accum[layer]["corrupt_count"] += batch_size
                del abl_logits, abl_tool_logit, abl_top1
                clear_cuda()
            del tokens, base_logits, base_tool_logit, base_top1, repeated_tokens
            clear_cuda()
        progress.set_postfix(last=batch.indices[-1], tok=batch.token_len)

    rows: list[Dict[str, object]] = []
    for layer in layers:
        layer_accum = accum[layer]
        clean_count = max(int(layer_accum["clean_count"]), 1)
        corrupt_count = max(int(layer_accum["corrupt_count"]), 1)
        for head in range(n_heads):
            clean_delta_mean = float((layer_accum["clean_delta_sum"][head] / clean_count).item())
            corrupt_delta_mean = float((layer_accum["corrupt_delta_sum"][head] / corrupt_count).item())
            rows.append(
                {
                    "layer": layer,
                    "head": head,
                    "clean_flip_rate": float((layer_accum["clean_flip_count"][head].double() / clean_count).item()),
                    "corrupt_gain_rate": float((layer_accum["corrupt_gain_count"][head].double() / corrupt_count).item()),
                    "clean_logit_delta_mean": clean_delta_mean,
                    "corrupt_logit_delta_mean": corrupt_delta_mean,
                    "abs_clean_logit_delta_mean": abs(clean_delta_mean),
                }
            )
    rows.sort(
        key=lambda item: (
            float(item["clean_flip_rate"]),
            abs(float(item["clean_logit_delta_mean"])),
            -float(item["corrupt_gain_rate"]),
        ),
        reverse=True,
    )
    metadata = {
        "clean_tool_top1_rate": float(clean_tool_top1 / max(clean_total, 1)),
        "corrupt_tool_top1_rate": float(corrupt_tool_top1 / max(corrupt_total, 1)),
        "n_clean": clean_total,
        "n_corrupt": corrupt_total,
    }
    return rows, metadata


def build_matrix(rows: Sequence[Dict[str, object]], layers: Sequence[int], n_heads: int, value_key: str) -> np.ndarray:
    matrix = np.full((len(layers), n_heads), np.nan, dtype=np.float32)
    layer_to_row = {layer: idx for idx, layer in enumerate(layers)}
    for row in rows:
        layer = int(row["layer"])
        head = int(row["head"])
        if layer in layer_to_row:
            matrix[layer_to_row[layer], head] = float(row[value_key])
    return matrix


def plot_heatmap(
    matrix: np.ndarray,
    *,
    layers: Sequence[int],
    title: str,
    colorbar_label: str,
    path: Path,
    cmap: str,
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    plt.figure(figsize=(16, 5))
    image = plt.imshow(matrix, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar(image, label=colorbar_label)
    plt.xlabel("Head")
    plt.ylabel("Layer")
    plt.yticks(np.arange(len(layers)), [str(layer) for layer in layers])
    plt.title(title)
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()


def save_representative_attention_figures(
    model,
    pair: SamplePair,
    *,
    selected_heads: Sequence[tuple[int, int]],
    output_dir: Path,
) -> dict[str, object]:
    unique_layers = sorted({layer for layer, _head in selected_heads})
    hook_names = [f"blocks.{layer}.attn.hook_pattern" for layer in unique_layers]
    n_heads = int(model.cfg.n_heads)
    pattern_head_count: int | None = None

    for condition in ("clean", "corrupt"):
        tokens_cpu = pair.clean_tokens_cpu if condition == "clean" else pair.corrupt_tokens_cpu
        tokens = tokens_cpu.to(model.W_U.device)
        with torch.no_grad():
            _, cache = model.run_with_cache(tokens, names_filter=lambda name: name in hook_names)
        for layer, head in selected_heads:
            pattern = cache[f"blocks.{layer}.attn.hook_pattern"][0].detach().cpu().float()
            pattern_head_count = int(pattern.shape[0])
            seq_len = int(pattern.shape[-1])
            last_pos = seq_len - 1
            pattern_head = resolve_pattern_head_idx(head, pattern_head_count, n_heads)
            weights = pattern[pattern_head, last_pos, :].numpy()[None, :]
            plt.figure(figsize=(14, 2.4))
            plt.imshow(weights, aspect="auto", cmap="viridis")
            plt.colorbar(label="Attention")
            for diff_pos in pair.diff_positions:
                plt.axvline(diff_pos, color="#ff595e", linewidth=0.7, alpha=0.8)
            plt.yticks([0], [f"L{layer}H{head}"])
            plt.xlabel("Source position")
            plt.title(
                f"{pair.sample_id} {condition} last-token attention | "
                f"verb={pair.clean_candidate if condition == 'clean' else pair.corrupt_candidate}"
            )
            plt.tight_layout()
            figure_path = output_dir / f"attn_pattern_L{layer}H{head}_{condition}.png"
            plt.savefig(figure_path, dpi=220, bbox_inches="tight")
            plt.close()
        del tokens, cache
        clear_cuda()

    return {
        "sample_id": pair.sample_id,
        "pattern_head_count": pattern_head_count,
        "diff_positions": pair.diff_positions,
    }


def build_summary(
    *,
    layers: Sequence[int],
    dla_rows: Sequence[Dict[str, object]],
    layer_rows: Sequence[Dict[str, object]],
    verb_summary_rows: Sequence[Dict[str, object]],
    ablation_rows: Sequence[Dict[str, object]],
) -> str:
    global_top = list(sorted(dla_rows, key=lambda item: float(item["abs_delta"]), reverse=True)[:8])
    l29_top = [row for row in sorted((row for row in dla_rows if int(row["layer"]) == 29), key=lambda item: float(item["abs_delta"]), reverse=True)[:3]]
    verb_top = list(sorted(verb_summary_rows, key=lambda item: abs(float(item["delta_attn_to_verb"])), reverse=True)[:5])
    flip_top = list(sorted(ablation_rows, key=lambda item: float(item["clean_flip_rate"]), reverse=True)[:5])
    corrupt_gain = [row for row in ablation_rows if float(row["corrupt_gain_rate"]) > 0.05]

    lines = [
        "# L25-L35 Attention Head Analysis",
        "",
        "## DLA",
        "Top heads by |clean-corrupt DLA|:",
    ]
    for row in global_top:
        lines.append(
            f"- L{int(row['layer'])}H{int(row['head'])}: "
            f"delta={float(row['delta']):.3f} "
            f"(clean={float(row['mean_clean']):.3f}, corrupt={float(row['mean_corrupt']):.3f})"
        )

    lines.extend(
        [
            "",
            "L29 top-3 heads:",
        ]
    )
    for row in l29_top:
        lines.append(f"- H{int(row['head'])}: delta={float(row['delta']):.3f}")

    lines.extend(
        [
            "",
            "Per-layer max |delta|:",
        ]
    )
    for row in layer_rows:
        lines.append(
            f"- L{int(row['layer'])}: top_abs=L{int(row['layer'])}H{int(row['top_head_by_abs'])} "
            f"(delta={float(row['top_head_delta']):.3f}), "
            f"n(|delta|>1)={int(row['n_heads_abs_delta_gt_1'])}"
        )

    lines.extend(
        [
            "",
            "## Verb Attention",
            "Heads with the largest clean-corrupt verb-attention delta:",
        ]
    )
    for row in verb_top:
        lines.append(
            f"- L{int(row['layer'])}H{int(row['head'])}: "
            f"delta_attn_to_verb={float(row['delta_attn_to_verb']):.4f}, "
            f"clean={float(row['mean_clean_attn_to_verb']):.4f}, "
            f"corrupt={float(row['mean_corrupt_attn_to_verb']):.4f}"
        )

    lines.extend(
        [
            "",
            "## Causal Ablation",
            "Heads with the highest clean flip rate:",
        ]
    )
    for row in flip_top:
        lines.append(
            f"- L{int(row['layer'])}H{int(row['head'])}: "
            f"clean_flip_rate={float(row['clean_flip_rate']):.3f}, "
            f"corrupt_gain_rate={float(row['corrupt_gain_rate']):.3f}, "
            f"logit_delta=({float(row['clean_logit_delta_mean']):.3f}, {float(row['corrupt_logit_delta_mean']):.3f})"
        )

    lines.append("")
    if corrupt_gain:
        lines.append("Heads with corrupt_gain_rate > 0.05:")
        for row in sorted(corrupt_gain, key=lambda item: float(item["corrupt_gain_rate"]), reverse=True):
            lines.append(
                f"- L{int(row['layer'])}H{int(row['head'])}: corrupt_gain_rate={float(row['corrupt_gain_rate']):.3f}"
            )
    else:
        lines.append("No heads exceeded corrupt_gain_rate > 0.05 in L25-L35.")

    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Qwen3-8B L25-L35 attention head analysis")
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH)
    parser.add_argument("--manifest-path", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--layers", type=int, nargs="+", default=list(DEFAULT_LAYERS))
    parser.add_argument("--primary-layer", type=int, default=29)
    parser.add_argument("--max-pairs", type=int, default=200)
    parser.add_argument("--verb-pairs", type=int, default=50)
    parser.add_argument("--ablation-pairs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--verb-batch-size", type=int, default=2)
    parser.add_argument("--ablation-batch-size", type=int, default=2)
    parser.add_argument("--verb-heads-per-layer", type=int, default=5)
    parser.add_argument("--global-verb-heads", type=int, default=12)
    parser.add_argument("--figure-heads", type=int, default=6)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    layers = tuple(sorted(set(int(layer) for layer in args.layers)))
    if args.primary_layer not in layers:
        raise ValueError(f"--primary-layer {args.primary_layer} must be included in --layers")

    set_seed(args.seed)
    args.output_root.mkdir(parents=True, exist_ok=True)
    figures_dir = args.output_root / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    print(f"[setup] layers={layers}", flush=True)
    model, tokenizer = load_hooked_qwen3(str(args.model_path), device=args.device, dtype=torch.bfloat16)
    tool_token_ids = tokenizer.encode(TOOL_CALL_STR, add_special_tokens=False)
    if len(tool_token_ids) != 1:
        raise ValueError(f"{TOOL_CALL_STR!r} encoded to {tool_token_ids}, expected a single token.")
    tool_token_id = int(tool_token_ids[0])
    n_heads = int(model.cfg.n_heads)

    clean_meta = load_condition_manifest_map(args.dataset_root, "clean")
    corrupt_meta = load_condition_manifest_map(args.dataset_root, "corrupt")
    pairs = load_sample_pairs(
        args.manifest_path,
        model,
        clean_meta,
        corrupt_meta,
        max_pairs=args.max_pairs,
    )
    print(f"[setup] loaded {len(pairs)} pairs from {args.manifest_path}", flush=True)

    manifest_rows = [
        {
            "order": pair.order,
            "sample_id": pair.sample_id,
            "clean_path": str(pair.clean_path),
            "corrupt_path": str(pair.corrupt_path),
            "token_len": pair.token_len,
            "diff_start": pair.diff_positions[0],
            "diff_end": pair.diff_positions[-1],
            "diff_count": len(pair.diff_positions),
            "clean_candidate": pair.clean_candidate,
            "corrupt_candidate": pair.corrupt_candidate,
        }
        for pair in pairs
    ]
    write_csv(args.output_root / "sample_manifest.csv", manifest_rows)

    dla_batches = build_pair_batches(pairs, args.batch_size)
    print(f"[setup] DLA batches={len(dla_batches)} batch_size={args.batch_size}", flush=True)
    dla_rows, layer_rows, dla_meta = analyze_dla(
        model,
        dla_batches,
        layers=layers,
        tool_token_id=tool_token_id,
    )
    write_csv(args.output_root / "dla_per_head.csv", dla_rows)
    write_csv(args.output_root / "layer_comparison_l25_l35.csv", layer_rows)
    write_csv(
        args.output_root / "dla_l29_per_head.csv",
        [row for row in dla_rows if int(row["layer"]) == args.primary_layer],
    )

    selected_heads, selected_head_meta = select_heads_for_attention(
        dla_rows,
        layers=layers,
        heads_per_layer=args.verb_heads_per_layer,
        global_top_k=args.global_verb_heads,
    )
    write_csv(args.output_root / "selected_heads_for_attention.csv", selected_head_meta)

    verb_pairs = pairs[: args.verb_pairs]
    verb_batches = build_pair_batches(verb_pairs, args.verb_batch_size)
    print(f"[setup] Verb pairs={len(verb_pairs)} batches={len(verb_batches)} batch_size={args.verb_batch_size}", flush=True)
    verb_rows, verb_summary_rows, verb_meta = analyze_verb_attention(
        model,
        verb_pairs,
        verb_batches,
        selected_heads=selected_heads,
    )
    write_csv(args.output_root / "verb_attention_top_heads.csv", verb_rows)
    write_csv(args.output_root / "verb_attention_summary.csv", verb_summary_rows)
    write_csv(
        args.output_root / "verb_attention_l29_top_heads.csv",
        [row for row in verb_rows if int(row["layer"]) == args.primary_layer],
    )

    ablation_pairs = pairs[: args.ablation_pairs]
    ablation_batches = build_pair_batches(ablation_pairs, args.ablation_batch_size)
    print(
        f"[setup] Ablation pairs={len(ablation_pairs)} batches={len(ablation_batches)} "
        f"batch_size={args.ablation_batch_size}",
        flush=True,
    )
    ablation_rows, ablation_meta = analyze_head_ablation(
        model,
        ablation_batches,
        layers=layers,
        tool_token_id=tool_token_id,
    )
    write_csv(args.output_root / "ablation_per_head.csv", ablation_rows)
    write_csv(
        args.output_root / "ablation_l29_per_head.csv",
        [row for row in ablation_rows if int(row["layer"]) == args.primary_layer],
    )

    dla_matrix = build_matrix(dla_rows, layers, n_heads, "delta")
    plot_heatmap(
        dla_matrix,
        layers=layers,
        title="Attention Head DLA Delta (clean - corrupt)",
        colorbar_label="DLA delta",
        path=figures_dir / "dla_delta_heatmap.png",
        cmap="RdBu_r",
        vmin=float(np.nanpercentile(dla_matrix, 2)),
        vmax=float(np.nanpercentile(dla_matrix, 98)),
    )
    plot_heatmap(
        np.abs(dla_matrix),
        layers=layers,
        title="Attention Head |DLA Delta|",
        colorbar_label="|DLA delta|",
        path=figures_dir / "dla_abs_delta_heatmap.png",
        cmap="magma",
    )

    verb_matrix = build_matrix(
        [
            {
                "layer": row["layer"],
                "head": row["head"],
                "value": row["delta_attn_to_verb"],
            }
            for row in verb_summary_rows
        ],
        layers,
        n_heads,
        "value",
    )
    plot_heatmap(
        verb_matrix,
        layers=layers,
        title="Verb Attention Delta (clean - corrupt)",
        colorbar_label="attn_to_verb delta",
        path=figures_dir / "verb_attention_delta_heatmap.png",
        cmap="RdBu_r",
    )

    clean_flip_matrix = build_matrix(
        [
            {
                "layer": row["layer"],
                "head": row["head"],
                "value": row["clean_flip_rate"],
            }
            for row in ablation_rows
        ],
        layers,
        n_heads,
        "value",
    )
    plot_heatmap(
        clean_flip_matrix,
        layers=layers,
        title="Clean Flip Rate Under Single-Head Ablation",
        colorbar_label="clean flip rate",
        path=figures_dir / "ablation_clean_flip_heatmap.png",
        cmap="viridis",
        vmin=0.0,
        vmax=float(np.nanmax(clean_flip_matrix)),
    )

    clean_logit_delta_matrix = build_matrix(
        [
            {
                "layer": row["layer"],
                "head": row["head"],
                "value": row["clean_logit_delta_mean"],
            }
            for row in ablation_rows
        ],
        layers,
        n_heads,
        "value",
    )
    plot_heatmap(
        clean_logit_delta_matrix,
        layers=layers,
        title="Mean Clean Logit Delta Under Single-Head Ablation",
        colorbar_label="ablated - baseline <tool_call> logit",
        path=figures_dir / "ablation_clean_logit_delta_heatmap.png",
        cmap="RdBu_r",
    )

    figure_heads = [(int(row["layer"]), int(row["head"])) for row in sorted(dla_rows, key=lambda item: float(item["abs_delta"]), reverse=True)[: args.figure_heads]]
    figure_meta = save_representative_attention_figures(
        model,
        pairs[0],
        selected_heads=figure_heads,
        output_dir=figures_dir,
    )

    summary_text = build_summary(
        layers=layers,
        dla_rows=dla_rows,
        layer_rows=layer_rows,
        verb_summary_rows=verb_summary_rows,
        ablation_rows=ablation_rows,
    )
    (args.output_root / "summary.md").write_text(summary_text, encoding="utf-8")

    metadata = {
        "seed": args.seed,
        "model_path": str(args.model_path),
        "manifest_path": str(args.manifest_path),
        "dataset_root": str(args.dataset_root),
        "output_root": str(args.output_root),
        "tool_token_id": tool_token_id,
        "n_heads": n_heads,
        "n_layers": int(model.cfg.n_layers),
        "layers_analyzed": list(layers),
        "primary_layer": args.primary_layer,
        "n_pairs": len(pairs),
        "n_verb_pairs": len(verb_pairs),
        "n_ablation_pairs": len(ablation_pairs),
        "dla": dla_meta,
        "verb_attention": verb_meta,
        "ablation": ablation_meta,
        "representative_attention_figures": figure_meta,
        "selected_heads_for_attention": [{"layer": layer, "head": head} for layer, head in selected_heads],
        "figure_heads": [{"layer": layer, "head": head} for layer, head in figure_heads],
        "outputs": {
            "dla_per_head_csv": str(args.output_root / "dla_per_head.csv"),
            "layer_comparison_csv": str(args.output_root / "layer_comparison_l25_l35.csv"),
            "verb_attention_csv": str(args.output_root / "verb_attention_top_heads.csv"),
            "verb_attention_summary_csv": str(args.output_root / "verb_attention_summary.csv"),
            "ablation_per_head_csv": str(args.output_root / "ablation_per_head.csv"),
            "summary_md": str(args.output_root / "summary.md"),
        },
    }
    write_json(args.output_root / "metadata.json", metadata)

    del model
    clear_cuda()


if __name__ == "__main__":
    main()
