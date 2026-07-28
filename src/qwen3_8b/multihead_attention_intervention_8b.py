#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gc
import json
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

LEGACY_SRC = Path("./src")
if str(LEGACY_SRC) not in sys.path:
    sys.path.insert(0, str(LEGACY_SRC))

from toolcall_circuit.single_sample import load_hooked_qwen3  # noqa: E402


SEED = 42
MODEL_PATH = Path("./external/models/Qwen3-8B")
ATTN_ANALYSIS_ROOT = Path("./results/8B/attention_analysis")
MANIFEST_PATH = ATTN_ANALYSIS_ROOT / "sample_manifest.csv"
OUTPUT_ROOT = Path("./results/8B/attention_combo_intervention")
TOOL_CALL_STR = "<tool_call>"


@dataclass(frozen=True)
class HeadSpec:
    layer: int
    head: int

    @property
    def label(self) -> str:
        return f"L{self.layer}H{self.head}"


@dataclass(frozen=True)
class ComboSpec:
    name: str
    heads: tuple[HeadSpec, ...]
    rationale: str
    source: str

    @property
    def layers(self) -> tuple[int, ...]:
        return tuple(sorted({head.layer for head in self.heads}))

    @property
    def head_list(self) -> str:
        return "|".join(head.label for head in self.heads)


@dataclass
class SamplePair:
    order: int
    sample_id: str
    clean_path: Path
    corrupt_path: Path
    clean_tokens_cpu: torch.Tensor
    corrupt_tokens_cpu: torch.Tensor
    token_len: int


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


def write_json(path: Path, data: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def tool_stats(logits: torch.Tensor, tool_token_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    last_logits = logits[:, -1, :]
    tool_logit = last_logits[:, tool_token_id].detach().cpu().float()
    top1 = last_logits.argmax(dim=-1).detach().cpu()
    return tool_logit, top1


def load_sample_pairs(manifest_path: Path, model, *, max_pairs: int) -> list[SamplePair]:
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
            pairs.append(
                SamplePair(
                    order=int(row["order"]),
                    sample_id=str(row["sample_id"]),
                    clean_path=clean_path,
                    corrupt_path=corrupt_path,
                    clean_tokens_cpu=clean_tokens_cpu,
                    corrupt_tokens_cpu=corrupt_tokens_cpu,
                    token_len=clean_len,
                )
            )
            if len(pairs) >= max_pairs:
                break
    if len(pairs) < max_pairs:
        raise RuntimeError(f"Only found {len(pairs)} usable pairs in {manifest_path}")
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
            batches.append(
                PairBatch(
                    indices=[idx for idx, _pair in chunk],
                    clean_tokens_cpu=torch.cat([pair.clean_tokens_cpu for _, pair in chunk], dim=0),
                    corrupt_tokens_cpu=torch.cat([pair.corrupt_tokens_cpu for _, pair in chunk], dim=0),
                    token_len=token_len,
                )
            )
    return batches


def collect_cache_and_stats(
    model,
    tokens: torch.Tensor,
    hook_names: Sequence[str],
    tool_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    wanted = set(hook_names)
    with torch.no_grad():
        logits, cache = model.run_with_cache(tokens, names_filter=lambda name: name in wanted)
    tool_logit, top1 = tool_stats(logits, tool_token_id)
    cache_cpu = {name: cache[name].detach().cpu() for name in hook_names if name in cache}
    return tool_logit, top1, cache_cpu


def run_with_hooks_and_stats(
    model,
    tokens: torch.Tensor,
    fwd_hooks,
    tool_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        logits = model.run_with_hooks(tokens, fwd_hooks=fwd_hooks)
    return tool_stats(logits, tool_token_id)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def dedupe_heads(heads: Iterable[HeadSpec]) -> tuple[HeadSpec, ...]:
    out: list[HeadSpec] = []
    seen = set()
    for head in heads:
        key = (head.layer, head.head)
        if key not in seen:
            seen.add(key)
            out.append(head)
    return tuple(out)


def top_heads_from_rows(
    rows: Sequence[dict[str, str]],
    *,
    key: str,
    top_k: int,
    descending: bool,
    filters=None,
) -> tuple[HeadSpec, ...]:
    filtered = [row for row in rows if filters is None or filters(row)]
    ordered = sorted(filtered, key=lambda row: float(row[key]), reverse=descending)
    return tuple(HeadSpec(int(row["layer"]), int(row["head"])) for row in ordered[:top_k])


def build_default_combos(attn_root: Path) -> list[ComboSpec]:
    dla_rows = read_csv_rows(attn_root / "dla_per_head.csv")
    ablation_rows = read_csv_rows(attn_root / "ablation_per_head.csv")
    verb_rows = read_csv_rows(attn_root / "verb_attention_summary.csv")

    top_dla_positive = top_heads_from_rows(
        dla_rows,
        key="delta",
        top_k=6,
        descending=True,
        filters=lambda row: float(row["delta"]) > 0.0,
    )
    l29_positive = top_heads_from_rows(
        dla_rows,
        key="delta",
        top_k=3,
        descending=True,
        filters=lambda row: int(row["layer"]) == 29 and float(row["delta"]) > 0.0,
    )
    top_ablation = top_heads_from_rows(
        ablation_rows,
        key="clean_logit_delta_mean",
        top_k=5,
        descending=False,
    )
    top_verb_negative = top_heads_from_rows(
        verb_rows,
        key="delta_attn_to_verb",
        top_k=3,
        descending=False,
    )

    manual = {
        "single_L29H9": (HeadSpec(29, 9),),
        "single_L33H29": (HeadSpec(33, 29),),
        "single_L34H1": (HeadSpec(34, 1),),
        "single_L28H3": (HeadSpec(28, 3),),
        "l29_top3": l29_positive,
        "legacy_spine": (HeadSpec(28, 3), HeadSpec(29, 9), HeadSpec(33, 29)),
        "late_writer_triad": (HeadSpec(32, 3), HeadSpec(33, 29), HeadSpec(34, 1)),
        "top_dla_4": top_dla_positive[:4],
        "top_dla_6": top_dla_positive,
        "top_ablation_5": top_ablation,
        "verb_reader_3": top_verb_negative,
        "route_writer_bridge": (
            HeadSpec(28, 3),
            HeadSpec(29, 9),
            HeadSpec(29, 11),
            HeadSpec(32, 3),
            HeadSpec(33, 29),
            HeadSpec(34, 1),
        ),
    }

    rationale = {
        "single_L29H9": "Single-head patching baseline for the strongest L29 head.",
        "single_L33H29": "Single-head patching baseline for the strongest late writer head.",
        "single_L34H1": "Single-head patching baseline for the strongest L34 writer head.",
        "single_L28H3": "Single-head patching baseline for the legacy upstream route head.",
        "l29_top3": "L29-only bundle using the top positive DLA heads in L29.",
        "legacy_spine": "Legacy forward backbone reproduced by the new attention analysis.",
        "late_writer_triad": "Late writer block around L32-L34.",
        "top_dla_4": "Top-4 positive DLA heads across L25-L35.",
        "top_dla_6": "Top-6 positive DLA heads across L25-L35.",
        "top_ablation_5": "Top-5 heads by single-head clean logit drop.",
        "verb_reader_3": "Heads with the strongest corrupt-biased verb attention.",
        "route_writer_bridge": "Bridge from upstream routing into late writer heads.",
    }
    source = {
        "single_L29H9": "manual+attention_analysis",
        "single_L33H29": "manual+attention_analysis",
        "single_L34H1": "manual+attention_analysis",
        "single_L28H3": "manual+legacy_circuit",
        "l29_top3": "attention_analysis:dla_l29",
        "legacy_spine": "manual+legacy_circuit",
        "late_writer_triad": "manual+attention_analysis",
        "top_dla_4": "attention_analysis:dla_top4",
        "top_dla_6": "attention_analysis:dla_top6",
        "top_ablation_5": "attention_analysis:ablation_top5",
        "verb_reader_3": "attention_analysis:verb_top3_negative",
        "route_writer_bridge": "manual+attention_analysis",
    }

    combos: list[ComboSpec] = []
    for name, heads in manual.items():
        combos.append(
            ComboSpec(
                name=name,
                heads=dedupe_heads(heads),
                rationale=rationale[name],
                source=source[name],
            )
        )
    return combos


def combo_rows(combos: Sequence[ComboSpec]) -> list[dict[str, object]]:
    return [
        {
            "combo_name": combo.name,
            "n_heads": len(combo.heads),
            "head_list": combo.head_list,
            "layers": "|".join(str(layer) for layer in combo.layers),
            "rationale": combo.rationale,
            "source": combo.source,
        }
        for combo in combos
    ]


def heads_by_layer(combo: ComboSpec) -> dict[int, list[int]]:
    by_layer: dict[int, list[int]] = defaultdict(list)
    for head in combo.heads:
        by_layer[head.layer].append(head.head)
    return by_layer


def make_zero_hook(head_indices: Sequence[int]):
    head_indices = list(head_indices)

    def hook_fn(value: torch.Tensor, hook):  # noqa: ANN001
        out = value.clone()
        out[:, :, head_indices, :] = 0
        return out

    return hook_fn


def make_patch_hook(head_indices: Sequence[int], source_cpu: torch.Tensor):
    head_indices = list(head_indices)

    def hook_fn(value: torch.Tensor, hook):  # noqa: ANN001
        src = source_cpu.to(device=value.device, dtype=value.dtype)
        out = value.clone()
        out[:, :, head_indices, :] = src[:, :, head_indices, :]
        return out

    return hook_fn


def load_single_ablation_map(path: Path) -> dict[tuple[int, int], dict[str, float]]:
    out: dict[tuple[int, int], dict[str, float]] = {}
    for row in read_csv_rows(path):
        out[(int(row["layer"]), int(row["head"]))] = {
            "clean_logit_delta_mean": float(row["clean_logit_delta_mean"]),
            "corrupt_logit_delta_mean": float(row["corrupt_logit_delta_mean"]),
        }
    return out


def run_joint_interventions(
    model,
    pair_batches: Sequence[PairBatch],
    combos: Sequence[ComboSpec],
    *,
    tool_token_id: int,
    single_ablation_map: dict[tuple[int, int], dict[str, float]],
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    unique_hook_names = sorted({f"blocks.{head.layer}.attn.hook_z" for combo in combos for head in combo.heads})
    accum_ablation = {
        combo.name: {
            "clean_delta_sum": 0.0,
            "corrupt_delta_sum": 0.0,
            "clean_flip_count": 0,
            "corrupt_gain_count": 0,
            "clean_count": 0,
            "corrupt_count": 0,
        }
        for combo in combos
    }
    accum_patching = {
        combo.name: {
            "clean_to_corrupt_delta_sum": 0.0,
            "corrupt_to_clean_delta_sum": 0.0,
            "clean_to_corrupt_flip_count": 0,
            "corrupt_to_clean_flip_count": 0,
            "clean_count": 0,
            "corrupt_count": 0,
        }
        for combo in combos
    }
    hook_z_shape: list[int] | None = None
    clean_tool_top1 = 0
    corrupt_tool_top1 = 0
    clean_total = 0
    corrupt_total = 0

    progress = tqdm(pair_batches, desc="Joint head interventions", dynamic_ncols=True)
    for batch in progress:
        clean_tokens = batch.clean_tokens_cpu.to(model.W_U.device)
        corrupt_tokens = batch.corrupt_tokens_cpu.to(model.W_U.device)
        clean_logit, clean_top1, clean_cache = collect_cache_and_stats(model, clean_tokens, unique_hook_names, tool_token_id)
        corrupt_logit, corrupt_top1, corrupt_cache = collect_cache_and_stats(model, corrupt_tokens, unique_hook_names, tool_token_id)
        if hook_z_shape is None:
            first_key = unique_hook_names[0]
            hook_z_shape = [int(x) for x in clean_cache[first_key].shape]
            print(f"[sanity] {first_key} shape: {tuple(hook_z_shape)}", flush=True)

        batch_clean_count = int(clean_logit.numel())
        batch_corrupt_count = int(corrupt_logit.numel())
        clean_total += batch_clean_count
        corrupt_total += batch_corrupt_count
        clean_tool_top1 += int((clean_top1 == tool_token_id).sum().item())
        corrupt_tool_top1 += int((corrupt_top1 == tool_token_id).sum().item())

        for combo in combos:
            by_layer = heads_by_layer(combo)

            clean_zero_hooks = [
                (f"blocks.{layer}.attn.hook_z", make_zero_hook(heads))
                for layer, heads in sorted(by_layer.items())
            ]
            clean_abl_logit, clean_abl_top1 = run_with_hooks_and_stats(model, clean_tokens, clean_zero_hooks, tool_token_id)
            clean_delta = clean_abl_logit - clean_logit
            clean_flips = (clean_top1 == tool_token_id) & (clean_abl_top1 != tool_token_id)
            accum_ablation[combo.name]["clean_delta_sum"] += float(clean_delta.sum().item())
            accum_ablation[combo.name]["clean_flip_count"] += int(clean_flips.sum().item())
            accum_ablation[combo.name]["clean_count"] += batch_clean_count

            corrupt_zero_hooks = [
                (f"blocks.{layer}.attn.hook_z", make_zero_hook(heads))
                for layer, heads in sorted(by_layer.items())
            ]
            corrupt_abl_logit, corrupt_abl_top1 = run_with_hooks_and_stats(model, corrupt_tokens, corrupt_zero_hooks, tool_token_id)
            corrupt_delta = corrupt_abl_logit - corrupt_logit
            corrupt_gains = (corrupt_top1 != tool_token_id) & (corrupt_abl_top1 == tool_token_id)
            accum_ablation[combo.name]["corrupt_delta_sum"] += float(corrupt_delta.sum().item())
            accum_ablation[combo.name]["corrupt_gain_count"] += int(corrupt_gains.sum().item())
            accum_ablation[combo.name]["corrupt_count"] += batch_corrupt_count

            clean_patch_hooks = [
                (
                    f"blocks.{layer}.attn.hook_z",
                    make_patch_hook(heads, corrupt_cache[f"blocks.{layer}.attn.hook_z"]),
                )
                for layer, heads in sorted(by_layer.items())
            ]
            clean_patch_logit, clean_patch_top1 = run_with_hooks_and_stats(model, clean_tokens, clean_patch_hooks, tool_token_id)
            clean_patch_delta = clean_patch_logit - clean_logit
            clean_patch_flips = (clean_top1 == tool_token_id) & (clean_patch_top1 != tool_token_id)
            accum_patching[combo.name]["clean_to_corrupt_delta_sum"] += float(clean_patch_delta.sum().item())
            accum_patching[combo.name]["clean_to_corrupt_flip_count"] += int(clean_patch_flips.sum().item())
            accum_patching[combo.name]["clean_count"] += batch_clean_count

            corrupt_patch_hooks = [
                (
                    f"blocks.{layer}.attn.hook_z",
                    make_patch_hook(heads, clean_cache[f"blocks.{layer}.attn.hook_z"]),
                )
                for layer, heads in sorted(by_layer.items())
            ]
            corrupt_patch_logit, corrupt_patch_top1 = run_with_hooks_and_stats(model, corrupt_tokens, corrupt_patch_hooks, tool_token_id)
            corrupt_patch_delta = corrupt_patch_logit - corrupt_logit
            corrupt_patch_gains = (corrupt_top1 != tool_token_id) & (corrupt_patch_top1 == tool_token_id)
            accum_patching[combo.name]["corrupt_to_clean_delta_sum"] += float(corrupt_patch_delta.sum().item())
            accum_patching[combo.name]["corrupt_to_clean_flip_count"] += int(corrupt_patch_gains.sum().item())
            accum_patching[combo.name]["corrupt_count"] += batch_corrupt_count

        del clean_tokens, corrupt_tokens, clean_cache, corrupt_cache
        clear_cuda()
        progress.set_postfix(last=batch.indices[-1], tok=batch.token_len)

    ablation_rows: list[dict[str, object]] = []
    patching_rows: list[dict[str, object]] = []
    for combo in combos:
        combo_ab = accum_ablation[combo.name]
        combo_patch = accum_patching[combo.name]
        clean_count = max(combo_ab["clean_count"], 1)
        corrupt_count = max(combo_ab["corrupt_count"], 1)
        expected_clean = sum(single_ablation_map[(head.layer, head.head)]["clean_logit_delta_mean"] for head in combo.heads)
        expected_corrupt = sum(single_ablation_map[(head.layer, head.head)]["corrupt_logit_delta_mean"] for head in combo.heads)
        clean_mean = combo_ab["clean_delta_sum"] / clean_count
        corrupt_mean = combo_ab["corrupt_delta_sum"] / corrupt_count
        ablation_rows.append(
            {
                "combo_name": combo.name,
                "n_heads": len(combo.heads),
                "head_list": combo.head_list,
                "clean_flip_rate": combo_ab["clean_flip_count"] / clean_count,
                "corrupt_gain_rate": combo_ab["corrupt_gain_count"] / corrupt_count,
                "clean_logit_delta_mean": clean_mean,
                "corrupt_logit_delta_mean": corrupt_mean,
                "expected_clean_logit_delta_sum": expected_clean,
                "expected_corrupt_logit_delta_sum": expected_corrupt,
                "clean_synergy_vs_single_sum": clean_mean - expected_clean,
                "corrupt_synergy_vs_single_sum": corrupt_mean - expected_corrupt,
            }
        )

        patch_clean_count = max(combo_patch["clean_count"], 1)
        patch_corrupt_count = max(combo_patch["corrupt_count"], 1)
        patching_rows.append(
            {
                "combo_name": combo.name,
                "n_heads": len(combo.heads),
                "head_list": combo.head_list,
                "clean_to_corrupt_mean_logit_delta": combo_patch["clean_to_corrupt_delta_sum"] / patch_clean_count,
                "clean_to_corrupt_flip_rate": combo_patch["clean_to_corrupt_flip_count"] / patch_clean_count,
                "corrupt_to_clean_mean_logit_delta": combo_patch["corrupt_to_clean_delta_sum"] / patch_corrupt_count,
                "corrupt_to_clean_flip_rate": combo_patch["corrupt_to_clean_flip_count"] / patch_corrupt_count,
            }
        )

    ablation_rows.sort(key=lambda row: abs(float(row["clean_logit_delta_mean"])), reverse=True)
    patching_rows.sort(
        key=lambda row: max(
            abs(float(row["clean_to_corrupt_mean_logit_delta"])),
            abs(float(row["corrupt_to_clean_mean_logit_delta"])),
        ),
        reverse=True,
    )
    metadata = {
        "hook_z_shape": hook_z_shape,
        "clean_tool_top1_rate": clean_tool_top1 / max(clean_total, 1),
        "corrupt_tool_top1_rate": corrupt_tool_top1 / max(corrupt_total, 1),
        "n_clean": clean_total,
        "n_corrupt": corrupt_total,
    }
    return ablation_rows, patching_rows, metadata


def plot_joint_ablation(rows: Sequence[dict[str, object]], path: Path) -> None:
    names = [str(row["combo_name"]) for row in rows]
    clean_delta = np.asarray([float(row["clean_logit_delta_mean"]) for row in rows], dtype=np.float32)
    expected = np.asarray([float(row["expected_clean_logit_delta_sum"]) for row in rows], dtype=np.float32)
    x = np.arange(len(rows))
    width = 0.38

    plt.figure(figsize=(max(12, len(rows) * 0.7), 5.5))
    plt.bar(x - width / 2, clean_delta, width=width, label="observed", color="#4c78a8")
    plt.bar(x + width / 2, expected, width=width, label="sum(single-head)", color="#f58518")
    plt.axhline(0.0, color="#222222", linewidth=1.0)
    plt.xticks(x, names, rotation=35, ha="right")
    plt.ylabel("Mean clean logit delta")
    plt.title("Joint Head Ablation vs Sum of Single-Head Effects")
    plt.legend(frameon=False)
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()


def plot_cross_patching(rows: Sequence[dict[str, object]], path: Path) -> None:
    names = [str(row["combo_name"]) for row in rows]
    clean_to_corrupt = np.asarray([float(row["clean_to_corrupt_mean_logit_delta"]) for row in rows], dtype=np.float32)
    corrupt_to_clean = np.asarray([float(row["corrupt_to_clean_mean_logit_delta"]) for row in rows], dtype=np.float32)
    x = np.arange(len(rows))
    width = 0.38

    plt.figure(figsize=(max(12, len(rows) * 0.7), 5.5))
    plt.bar(x - width / 2, clean_to_corrupt, width=width, label="clean <- corrupt", color="#e45756")
    plt.bar(x + width / 2, corrupt_to_clean, width=width, label="corrupt <- clean", color="#72b7b2")
    plt.axhline(0.0, color="#222222", linewidth=1.0)
    plt.xticks(x, names, rotation=35, ha="right")
    plt.ylabel("Mean logit delta")
    plt.title("Cross-Condition Multi-Head Patching")
    plt.legend(frameon=False)
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()


def build_summary(ablation_rows: Sequence[dict[str, object]], patching_rows: Sequence[dict[str, object]]) -> str:
    top_ab = sorted(ablation_rows, key=lambda row: abs(float(row["clean_logit_delta_mean"])), reverse=True)[:6]
    top_patch = sorted(
        patching_rows,
        key=lambda row: max(
            abs(float(row["clean_to_corrupt_mean_logit_delta"])),
            abs(float(row["corrupt_to_clean_mean_logit_delta"])),
        ),
        reverse=True,
    )[:6]
    lines = [
        "# Multi-Head Joint Attention Interventions",
        "",
        "## Joint Ablation",
    ]
    for row in top_ab:
        lines.append(
            f"- {row['combo_name']}: clean_delta={float(row['clean_logit_delta_mean']):.3f}, "
            f"expected_single_sum={float(row['expected_clean_logit_delta_sum']):.3f}, "
            f"synergy={float(row['clean_synergy_vs_single_sum']):.3f}, "
            f"clean_flip_rate={float(row['clean_flip_rate']):.3f}"
        )
    lines.extend(["", "## Cross Patching"])
    for row in top_patch:
        lines.append(
            f"- {row['combo_name']}: clean<-corrupt={float(row['clean_to_corrupt_mean_logit_delta']):.3f} "
            f"(flip={float(row['clean_to_corrupt_flip_rate']):.3f}), "
            f"corrupt<-clean={float(row['corrupt_to_clean_mean_logit_delta']):.3f} "
            f"(flip={float(row['corrupt_to_clean_flip_rate']):.3f})"
        )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Qwen3-8B multi-head attention joint interventions")
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH)
    parser.add_argument("--attention-root", type=Path, default=ATTN_ANALYSIS_ROOT)
    parser.add_argument("--manifest-path", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--max-pairs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    args.output_root.mkdir(parents=True, exist_ok=True)
    figures_dir = args.output_root / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    combos = build_default_combos(args.attention_root)
    write_csv(args.output_root / "combo_definitions.csv", combo_rows(combos))

    model, tokenizer = load_hooked_qwen3(str(args.model_path), device=args.device, dtype=torch.bfloat16)
    tool_token_ids = tokenizer.encode(TOOL_CALL_STR, add_special_tokens=False)
    if len(tool_token_ids) != 1:
        raise ValueError(f"{TOOL_CALL_STR!r} encoded to {tool_token_ids}, expected one token.")
    tool_token_id = int(tool_token_ids[0])

    pairs = load_sample_pairs(args.manifest_path, model, max_pairs=args.max_pairs)
    pair_batches = build_pair_batches(pairs, args.batch_size)
    print(f"[setup] combos={len(combos)} pairs={len(pairs)} pair_batches={len(pair_batches)}", flush=True)

    single_ablation_map = load_single_ablation_map(args.attention_root / "ablation_per_head.csv")
    ablation_rows, patching_rows, metadata = run_joint_interventions(
        model,
        pair_batches,
        combos,
        tool_token_id=tool_token_id,
        single_ablation_map=single_ablation_map,
    )

    write_csv(args.output_root / "joint_ablation.csv", ablation_rows)
    write_csv(args.output_root / "cross_patching.csv", patching_rows)
    plot_joint_ablation(ablation_rows, figures_dir / "joint_ablation_clean_delta.png")
    plot_cross_patching(patching_rows, figures_dir / "cross_patching_logit_delta.png")

    summary = build_summary(ablation_rows, patching_rows)
    (args.output_root / "summary.md").write_text(summary, encoding="utf-8")

    write_json(
        args.output_root / "metadata.json",
        {
            "seed": args.seed,
            "model_path": str(args.model_path),
            "attention_root": str(args.attention_root),
            "manifest_path": str(args.manifest_path),
            "output_root": str(args.output_root),
            "tool_token_id": tool_token_id,
            "n_pairs": len(pairs),
            "batch_size": args.batch_size,
            "combos": [
                {
                    "combo_name": combo.name,
                    "head_list": combo.head_list,
                    "n_heads": len(combo.heads),
                    "rationale": combo.rationale,
                    "source": combo.source,
                }
                for combo in combos
            ],
            "intervention": metadata,
            "outputs": {
                "combo_definitions_csv": str(args.output_root / "combo_definitions.csv"),
                "joint_ablation_csv": str(args.output_root / "joint_ablation.csv"),
                "cross_patching_csv": str(args.output_root / "cross_patching.csv"),
                "summary_md": str(args.output_root / "summary.md"),
            },
        },
    )

    del model
    clear_cuda()


if __name__ == "__main__":
    main()
