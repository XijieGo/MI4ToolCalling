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
import torch.nn.functional as F
from safetensors.torch import load_file
from tqdm.auto import tqdm

LEGACY_SRC = Path("./src")
if str(LEGACY_SRC) not in sys.path:
    sys.path.insert(0, str(LEGACY_SRC))

from toolcall_circuit.single_sample import load_hooked_qwen3  # noqa: E402


SEED = 42
MODEL_PATH = Path("./external/models/Qwen3-8B")
TRANSCODER_PATH = Path("./external/transcoders/Qwen3-8B/layer_25.safetensors")
MANIFEST_PATH = Path("./results/8B/triplet_analysis/sample_manifest.csv")
OUTPUT_ROOT = Path("./results/8B/routing_validation")
TOOL_CALL_STR = "<tool_call>"


@dataclass(frozen=True)
class ComponentSpec:
    name: str
    kind: str
    layer: int
    head_idx: int | None
    role: str

    @property
    def hook_name(self) -> str:
        if self.kind == "head":
            return f"blocks.{self.layer}.attn.hook_z"
        return f"blocks.{self.layer}.hook_mlp_out"


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
    indices: List[int]
    clean_tokens_cpu: torch.Tensor
    corrupt_tokens_cpu: torch.Tensor
    token_len: int


PATCH_COMPONENTS: List[ComponentSpec] = [
    ComponentSpec("L24H30", "head", 24, 30, "routing hub"),
    ComponentSpec("L23H15", "head", 23, 15, "secondary hub"),
    ComponentSpec("MLP25", "mlp", 25, None, "router amplifier"),
    ComponentSpec("MLP27", "mlp", 27, None, "amplification stage"),
    ComponentSpec("MLP34", "mlp", 34, None, "writer control"),
]


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
    fieldnames: List[str] = []
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


def load_sample_pairs(
    manifest_path: Path,
    model,
    *,
    max_pairs: int,
) -> List[SamplePair]:
    pairs: List[SamplePair] = []
    with manifest_path.open(encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            clean_tokens = int(row["clean_tokens"])
            corrupt_tokens = int(row["corrupt_tokens"])
            if clean_tokens != corrupt_tokens:
                continue
            clean_path = Path(row["clean_path"])
            corrupt_path = Path(row["corrupt_path"])
            clean_text = clean_path.read_text(encoding="utf-8")
            corrupt_text = corrupt_path.read_text(encoding="utf-8")
            clean_tok_cpu = model.to_tokens(clean_text, prepend_bos=False).detach().cpu()
            corrupt_tok_cpu = model.to_tokens(corrupt_text, prepend_bos=False).detach().cpu()
            clean_len = int(clean_tok_cpu.shape[-1])
            corrupt_len = int(corrupt_tok_cpu.shape[-1])
            if clean_len != corrupt_len:
                continue
            pairs.append(
                SamplePair(
                    order=int(row["order"]),
                    sample_id=row["sample_id"],
                    clean_path=clean_path,
                    corrupt_path=corrupt_path,
                    clean_tokens_cpu=clean_tok_cpu,
                    corrupt_tokens_cpu=corrupt_tok_cpu,
                    token_len=clean_len,
                )
            )
            if len(pairs) >= max_pairs:
                break
    if not pairs:
        raise RuntimeError(f"No usable equal-length pairs found in {manifest_path}")
    return pairs


def build_pair_batches(pairs: Sequence[SamplePair], batch_size: int) -> List[PairBatch]:
    buckets: Dict[int, List[tuple[int, SamplePair]]] = defaultdict(list)
    for idx, pair in enumerate(pairs):
        buckets[pair.token_len].append((idx, pair))

    batches: List[PairBatch] = []
    for token_len in sorted(buckets.keys()):
        group = buckets[token_len]
        for start in range(0, len(group), batch_size):
            chunk = group[start : start + batch_size]
            indices = [idx for idx, _pair in chunk]
            clean_tokens = torch.cat([pair.clean_tokens_cpu for _, pair in chunk], dim=0)
            corrupt_tokens = torch.cat([pair.corrupt_tokens_cpu for _, pair in chunk], dim=0)
            batches.append(
                PairBatch(
                    indices=indices,
                    clean_tokens_cpu=clean_tokens,
                    corrupt_tokens_cpu=corrupt_tokens,
                    token_len=token_len,
                )
            )
    return batches


def tool_stats(logits: torch.Tensor, tool_token_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    last_logits = logits[:, -1, :]
    tool_logit = last_logits[:, tool_token_id].detach().cpu().float()
    top1 = last_logits.argmax(dim=-1).detach().cpu()
    return tool_logit, top1


def collect_cache_and_stats(
    model,
    tokens: torch.Tensor,
    hook_names: Sequence[str],
    tool_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
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


def make_patch_hook(component: ComponentSpec, source_cpu: torch.Tensor):
    if component.kind == "mlp":
        def hook_fn(value: torch.Tensor, hook):  # noqa: ANN001
            src = source_cpu.to(device=value.device, dtype=value.dtype)
            out = value.clone()
            out.copy_(src)
            return out

        return hook_fn

    head_idx = int(component.head_idx)

    def hook_fn(value: torch.Tensor, hook):  # noqa: ANN001
        src = source_cpu.to(device=value.device, dtype=value.dtype)
        out = value.clone()
        out[:, :, head_idx, :] = src[:, :, head_idx, :]
        return out

    return hook_fn


def zero_mlp_hook(value: torch.Tensor, hook):  # noqa: ANN001
    out = value.clone()
    out.zero_()
    return out


def run_patching_experiment(
    model,
    pair_batches: Sequence[PairBatch],
    *,
    tool_token_id: int,
    n_samples: int,
) -> tuple[List[Dict[str, object]], List[str], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, object]]:
    hook_names = [component.hook_name for component in PATCH_COMPONENTS]
    clean_baseline_logits = torch.empty(n_samples, dtype=torch.float32)
    corrupt_baseline_logits = torch.empty(n_samples, dtype=torch.float32)
    clean_baseline_top1 = torch.empty(n_samples, dtype=torch.long)
    corrupt_baseline_top1 = torch.empty(n_samples, dtype=torch.long)
    accumulators: Dict[str, Dict[str, Dict[str, float]]] = {
        component.name: {
            "clean2corrupt": {"delta_sum": 0.0, "flip_count": 0.0, "n_samples": 0.0},
            "corrupt2clean": {"delta_sum": 0.0, "flip_count": 0.0, "n_samples": 0.0},
        }
        for component in PATCH_COMPONENTS
    }
    hook_z_shape: Sequence[int] | None = None

    progress = tqdm(pair_batches, desc="Experiment A patching", dynamic_ncols=True)
    for batch in progress:
        device = model.W_U.device
        clean_tokens = batch.clean_tokens_cpu.to(device)
        corrupt_tokens = batch.corrupt_tokens_cpu.to(device)
        clean_logit, clean_top1, clean_cache = collect_cache_and_stats(model, clean_tokens, hook_names, tool_token_id)
        corrupt_logit, corrupt_top1, corrupt_cache = collect_cache_and_stats(model, corrupt_tokens, hook_names, tool_token_id)
        index_tensor = torch.tensor(batch.indices, dtype=torch.long)
        clean_baseline_logits[index_tensor] = clean_logit
        corrupt_baseline_logits[index_tensor] = corrupt_logit
        clean_baseline_top1[index_tensor] = clean_top1
        corrupt_baseline_top1[index_tensor] = corrupt_top1
        if hook_z_shape is None and "blocks.24.attn.hook_z" in clean_cache:
            hook_z_shape = tuple(int(x) for x in clean_cache["blocks.24.attn.hook_z"].shape)
            print(f"[sanity] blocks.24.attn.hook_z shape: {hook_z_shape}", flush=True)

        for component in PATCH_COMPONENTS:
            hook_name = component.hook_name
            if hook_name not in clean_cache or hook_name not in corrupt_cache:
                warn(f"Missing cache for {component.name} ({hook_name}); skipping.")
                continue

            clean_hook = [(hook_name, make_patch_hook(component, corrupt_cache[hook_name]))]
            clean_patched_logit, clean_patched_top1 = run_with_hooks_and_stats(
                model,
                clean_tokens,
                clean_hook,
                tool_token_id,
            )
            clean_delta = clean_patched_logit - clean_logit
            clean_flips = (clean_top1 == tool_token_id) & (clean_patched_top1 != tool_token_id)
            accumulators[component.name]["clean2corrupt"]["delta_sum"] += float(clean_delta.sum().item())
            accumulators[component.name]["clean2corrupt"]["flip_count"] += float(clean_flips.sum().item())
            accumulators[component.name]["clean2corrupt"]["n_samples"] += float(clean_delta.numel())

            corrupt_hook = [(hook_name, make_patch_hook(component, clean_cache[hook_name]))]
            corrupt_patched_logit, corrupt_patched_top1 = run_with_hooks_and_stats(
                model,
                corrupt_tokens,
                corrupt_hook,
                tool_token_id,
            )
            corrupt_delta = corrupt_patched_logit - corrupt_logit
            corrupt_flips = (corrupt_top1 != tool_token_id) & (corrupt_patched_top1 == tool_token_id)
            accumulators[component.name]["corrupt2clean"]["delta_sum"] += float(corrupt_delta.sum().item())
            accumulators[component.name]["corrupt2clean"]["flip_count"] += float(corrupt_flips.sum().item())
            accumulators[component.name]["corrupt2clean"]["n_samples"] += float(corrupt_delta.numel())

        progress.set_postfix(last=batch.indices[-1], tok=batch.token_len)
        del clean_tokens, corrupt_tokens, clean_cache, corrupt_cache
        clear_cuda()

    rows: List[Dict[str, object]] = []
    for component in PATCH_COMPONENTS:
        for direction in ("clean2corrupt", "corrupt2clean"):
            stats = accumulators[component.name][direction]
            count = int(stats["n_samples"])
            mean_delta = stats["delta_sum"] / max(count, 1)
            flip_count = int(stats["flip_count"])
            rows.append(
                {
                    "component": component.name,
                    "direction": direction,
                    "n_samples": count,
                    "mean_logit_delta": mean_delta,
                    "flip_rate": flip_count / max(count, 1),
                    "flip_count": flip_count,
                }
            )

    summary_lines = build_patching_summary(rows)
    metadata = {
        "hook_z_shape": list(hook_z_shape) if hook_z_shape is not None else None,
        "clean_tool_top1_rate": float((clean_baseline_top1 == tool_token_id).float().mean().item()),
        "corrupt_tool_top1_rate": float((corrupt_baseline_top1 == tool_token_id).float().mean().item()),
    }
    return (
        rows,
        summary_lines,
        clean_baseline_logits,
        clean_baseline_top1,
        corrupt_baseline_logits,
        corrupt_baseline_top1,
        metadata,
    )


def rows_by_component(rows: Sequence[Dict[str, object]]) -> Dict[str, Dict[str, Dict[str, object]]]:
    out: Dict[str, Dict[str, Dict[str, object]]] = {}
    for row in rows:
        out.setdefault(str(row["component"]), {})[str(row["direction"])] = dict(row)
    return out


def build_patching_summary(rows: Sequence[Dict[str, object]]) -> List[str]:
    by_component = rows_by_component(rows)
    control = by_component.get("MLP34", {})
    control_destroy = float(control.get("clean2corrupt", {}).get("flip_rate", 0.0))
    control_restore = float(control.get("corrupt2clean", {}).get("flip_rate", 0.0))
    lines = []
    for component in PATCH_COMPONENTS:
        clean_row = by_component[component.name]["clean2corrupt"]
        corrupt_row = by_component[component.name]["corrupt2clean"]
        clean_flip = float(clean_row["flip_rate"])
        corrupt_flip = float(corrupt_row["flip_rate"])
        clean_delta = float(clean_row["mean_logit_delta"])
        corrupt_delta = float(corrupt_row["mean_logit_delta"])
        if component.name in {"L24H30", "L23H15"}:
            supported = (
                clean_delta < 0.0
                and corrupt_delta > 0.0
                and clean_flip >= control_destroy
                and corrupt_flip >= control_restore
            )
            verdict = "支持路由枢纽假设" if supported else "不支持路由枢纽假设"
        elif component.name == "MLP25":
            supported = clean_delta < 0.0 and corrupt_delta > 0.0 and (clean_flip + corrupt_flip) >= (control_destroy + control_restore)
            verdict = "支持放大器假设" if supported else "不支持放大器假设"
        elif component.name == "MLP27":
            supported = clean_delta < 0.0 and corrupt_delta > 0.0
            verdict = "支持放大阶段判断" if supported else "不支持放大阶段判断"
        else:
            verdict = "对照组"
        lines.append(
            f"- {component.name}：破坏 flip_rate={clean_flip:.3f}，恢复 flip_rate={corrupt_flip:.3f}，"
            f"logit delta=({clean_delta:.3f}, {corrupt_delta:.3f})；结论：{verdict}。"
        )
    return lines


def run_ablation_sweep(
    model,
    pair_batches: Sequence[PairBatch],
    *,
    tool_token_id: int,
    clean_baseline_logits: torch.Tensor,
    clean_baseline_top1: torch.Tensor,
    corrupt_baseline_logits: torch.Tensor,
    corrupt_baseline_top1: torch.Tensor,
) -> List[Dict[str, object]]:
    n_layers = int(model.cfg.n_layers)
    rows: List[Dict[str, object]] = []
    clean_baseline_mean = float(clean_baseline_logits.mean().item())
    corrupt_baseline_mean = float(corrupt_baseline_logits.mean().item())

    progress = tqdm(range(n_layers), desc="Experiment B ablation", dynamic_ncols=True)
    for layer in progress:
        clean_ablated_sum = 0.0
        corrupt_ablated_sum = 0.0
        clean_flip_count = 0
        clean_count = 0
        corrupt_count = 0

        hook_name = f"blocks.{layer}.hook_mlp_out"
        for batch in pair_batches:
            device = model.W_U.device
            index_tensor = torch.tensor(batch.indices, dtype=torch.long)
            clean_tokens = batch.clean_tokens_cpu.to(device)
            corrupt_tokens = batch.corrupt_tokens_cpu.to(device)

            clean_logit, clean_top1 = run_with_hooks_and_stats(
                model,
                clean_tokens,
                [(hook_name, zero_mlp_hook)],
                tool_token_id,
            )
            corrupt_logit, _corrupt_top1 = run_with_hooks_and_stats(
                model,
                corrupt_tokens,
                [(hook_name, zero_mlp_hook)],
                tool_token_id,
            )

            clean_ablated_sum += float(clean_logit.sum().item())
            corrupt_ablated_sum += float(corrupt_logit.sum().item())
            clean_flip_count += int(((clean_baseline_top1[index_tensor] == tool_token_id) & (clean_top1 != tool_token_id)).sum().item())
            clean_count += int(clean_logit.numel())
            corrupt_count += int(corrupt_logit.numel())

            del clean_tokens, corrupt_tokens

        clean_ablated_mean = clean_ablated_sum / max(clean_count, 1)
        corrupt_ablated_mean = corrupt_ablated_sum / max(corrupt_count, 1)
        rows.append(
            {
                "layer": layer,
                "clean_baseline_mean": clean_baseline_mean,
                "clean_ablated_mean": clean_ablated_mean,
                "clean_logit_drop": clean_baseline_mean - clean_ablated_mean,
                "clean_flip_rate": clean_flip_count / max(clean_count, 1),
                "corrupt_baseline_mean": corrupt_baseline_mean,
                "corrupt_ablated_mean": corrupt_ablated_mean,
                "corrupt_logit_change": corrupt_ablated_mean - corrupt_baseline_mean,
            }
        )
        progress.set_postfix(layer=layer)
        clear_cuda()

    return rows


def plot_ablation_sweep(rows: Sequence[Dict[str, object]], path: Path) -> None:
    layers = np.asarray([int(row["layer"]) for row in rows], dtype=np.int64)
    clean_drop = np.asarray([float(row["clean_logit_drop"]) for row in rows], dtype=np.float32)
    clean_flip = np.asarray([float(row["clean_flip_rate"]) for row in rows], dtype=np.float32)
    highlights = {25: "MLP25", 27: "MLP27", 34: "MLP34", 35: "MLP35"}

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True, constrained_layout=True)
    colors = ["#4c78a8" if layer not in highlights else "#d95f02" for layer in layers]
    axes[0].bar(layers, clean_drop, color=colors, width=0.8)
    axes[0].set_ylabel("clean_logit_drop")
    axes[0].set_title("MLP Ablation Sweep")
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].plot(layers, clean_flip, color="#2f4b7c", marker="o", linewidth=2.0)
    axes[1].set_xlabel("Layer")
    axes[1].set_ylabel("clean_flip_rate")
    axes[1].grid(alpha=0.25)

    for ax in axes:
        for layer, label in highlights.items():
            ax.axvline(layer, color="#888888", linestyle=":", linewidth=1.0, alpha=0.7)
            ymax = ax.get_ylim()[1]
            ax.text(layer + 0.1, ymax * 0.93, label, rotation=90, va="top", ha="left", fontsize=9)

    axes[1].set_xticks(layers)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def resolve_mlp_input_hook(model, reference_tokens: torch.Tensor) -> str:
    candidates = ["blocks.25.hook_mlp_in", "blocks.25.mlp.hook_in"]
    if hasattr(model, "set_use_hook_mlp_in"):
        model.set_use_hook_mlp_in(True)
    if hasattr(model, "cfg") and hasattr(model.cfg, "use_hook_mlp_in"):
        model.cfg.use_hook_mlp_in = True

    for hook_name in candidates:
        with torch.no_grad():
            _, cache = model.run_with_cache(reference_tokens, names_filter=lambda name: name == hook_name)
        if hook_name in cache:
            print(f"[sanity] using MLP25 input hook: {hook_name}", flush=True)
            return hook_name

    fallback = "blocks.24.hook_resid_post"
    warn("MLP input hook unavailable; falling back to blocks.24.hook_resid_post")
    return fallback


def capture_activation(model, tokens: torch.Tensor, hook_name: str, *, final_only: bool) -> torch.Tensor:
    store: Dict[str, torch.Tensor] = {}

    def hook_fn(value: torch.Tensor, hook):  # noqa: ANN001
        store["value"] = value[:, -1, :].detach() if final_only else value.detach()
        return value

    with torch.no_grad():
        _ = model.run_with_hooks(tokens, fwd_hooks=[(hook_name, hook_fn)])
    if "value" not in store:
        raise RuntimeError(f"Hook {hook_name} did not fire.")
    return store["value"]


def analyze_mlp25_features(
    model,
    tokenizer,
    pair_batches: Sequence[PairBatch],
    *,
    hook_name: str,
    layer_path: Path,
    top_token_mining: bool,
) -> tuple[List[Dict[str, object]], Dict[str, object] | None, Dict[str, object], torch.Tensor]:
    tc_weights = load_file(str(layer_path))
    print(f"[sanity] transcoder keys: {sorted(tc_weights.keys())}", flush=True)
    W_enc = tc_weights["W_enc"].to(device=model.W_U.device, dtype=torch.bfloat16)
    b_enc = tc_weights["b_enc"].to(device=model.W_U.device, dtype=torch.bfloat16)
    n_features = int(W_enc.shape[0])

    clean_sum = torch.zeros(n_features, dtype=torch.float64)
    corrupt_sum = torch.zeros(n_features, dtype=torch.float64)
    clean_count = 0
    corrupt_count = 0

    progress = tqdm(pair_batches, desc="Experiment C feature delta", dynamic_ncols=True)
    for batch in progress:
        device = model.W_U.device
        clean_tokens = batch.clean_tokens_cpu.to(device)
        corrupt_tokens = batch.corrupt_tokens_cpu.to(device)

        clean_act = capture_activation(model, clean_tokens, hook_name, final_only=True)
        corrupt_act = capture_activation(model, corrupt_tokens, hook_name, final_only=True)

        clean_features = torch.relu(F.linear(clean_act.to(dtype=torch.bfloat16), W_enc, b_enc)).float().cpu()
        corrupt_features = torch.relu(F.linear(corrupt_act.to(dtype=torch.bfloat16), W_enc, b_enc)).float().cpu()

        clean_sum += clean_features.sum(dim=0, dtype=torch.float64)
        corrupt_sum += corrupt_features.sum(dim=0, dtype=torch.float64)
        clean_count += int(clean_features.shape[0])
        corrupt_count += int(corrupt_features.shape[0])

        del clean_tokens, corrupt_tokens, clean_act, corrupt_act, clean_features, corrupt_features
        clear_cuda()

    clean_mean = (clean_sum / max(clean_count, 1)).float()
    corrupt_mean = (corrupt_sum / max(corrupt_count, 1)).float()
    delta = clean_mean - corrupt_mean
    abs_delta = delta.abs()
    topk = min(200, n_features)
    values, indices = torch.topk(abs_delta, k=topk)

    feature_rows: List[Dict[str, object]] = []
    for rank, (feature_id, abs_value) in enumerate(zip(indices.tolist(), values.tolist()), start=1):
        feature_rows.append(
            {
                "feature_id": int(feature_id),
                "clean_mean": float(clean_mean[feature_id].item()),
                "corrupt_mean": float(corrupt_mean[feature_id].item()),
                "delta": float(delta[feature_id].item()),
                "abs_delta": float(abs_value),
                "rank": rank,
            }
        )

    top_tokens_payload = None
    if top_token_mining:
        top_feature_ids = [int(row["feature_id"]) for row in feature_rows[: min(50, len(feature_rows))]]
        top_tokens_payload = collect_top_tokens_for_features(
            model,
            tokenizer,
            pair_batches,
            hook_name=hook_name,
            W_enc=W_enc[top_feature_ids],
            b_enc=b_enc[top_feature_ids],
            feature_ids=top_feature_ids,
            delta=delta,
        )

    metadata = summarize_feature_distribution(delta)
    return feature_rows, top_tokens_payload, metadata, delta


def collect_top_tokens_for_features(
    model,
    tokenizer,
    pair_batches: Sequence[PairBatch],
    *,
    hook_name: str,
    W_enc: torch.Tensor,
    b_enc: torch.Tensor,
    feature_ids: Sequence[int],
    delta: torch.Tensor,
) -> Dict[str, object]:
    candidates: Dict[int, List[tuple[float, int]]] = {int(feature_id): [] for feature_id in feature_ids}
    progress = tqdm(pair_batches, desc="Experiment C top tokens", dynamic_ncols=True)

    for batch in progress:
        for condition in ("clean", "corrupt"):
            tokens_cpu = batch.clean_tokens_cpu if condition == "clean" else batch.corrupt_tokens_cpu
            tokens = tokens_cpu.to(model.W_U.device)
            acts = capture_activation(model, tokens, hook_name, final_only=False)
            feats = torch.relu(F.linear(acts.to(dtype=torch.bfloat16), W_enc, b_enc)).float()
            flat_feats = feats.reshape(-1, feats.shape[-1])
            flat_tokens = tokens.reshape(-1).detach().cpu()
            k = min(5, flat_feats.shape[0])
            top_vals, top_idx = torch.topk(flat_feats, k=k, dim=0)
            for col, feature_id in enumerate(feature_ids):
                for row in range(k):
                    score = float(top_vals[row, col].item())
                    token_id = int(flat_tokens[int(top_idx[row, col].item())].item())
                    candidates[int(feature_id)].append((score, token_id))
            del tokens, acts, feats, flat_feats
        clear_cuda()

    payload: Dict[str, object] = {}
    for feature_id in feature_ids:
        ranked = sorted(candidates[int(feature_id)], key=lambda item: item[0], reverse=True)
        tokens_unique: List[str] = []
        seen = set()
        for _score, token_id in ranked:
            token_text = tokenizer.decode([token_id])
            if token_text not in seen:
                seen.add(token_text)
                tokens_unique.append(token_text)
            if len(tokens_unique) >= 5:
                break
        payload[f"feature_{feature_id}"] = {
            "delta": float(delta[feature_id].item()),
            "top5_tokens": tokens_unique,
        }
    return payload


def summarize_feature_distribution(delta: torch.Tensor) -> Dict[str, object]:
    delta_cpu = delta.detach().cpu().float()
    sigma = float(delta_cpu.std(unbiased=True).item()) if delta_cpu.numel() > 1 else 0.0
    min_value = float(delta_cpu.min().item())
    max_value = float(delta_cpu.max().item())
    if math.isclose(min_value, max_value):
        max_value = min_value + 1e-6
    hist = torch.histc(delta_cpu, bins=120, min=min_value, max=max_value)
    edges = torch.linspace(min_value, max_value, steps=121)
    centers = 0.5 * (edges[:-1] + edges[1:])
    kernel = torch.ones(5, dtype=torch.float32) / 5.0
    smooth = torch.nn.functional.conv1d(
        hist.view(1, 1, -1),
        kernel.view(1, 1, -1),
        padding=2,
    ).view(-1)
    neg_mask = centers < 0
    pos_mask = centers > 0
    neg_peak = float(centers[neg_mask][torch.argmax(smooth[neg_mask])].item()) if bool(neg_mask.any()) else float("nan")
    pos_peak = float(centers[pos_mask][torch.argmax(smooth[pos_mask])].item()) if bool(pos_mask.any()) else float("nan")
    center_idx = int(torch.argmin(torch.abs(centers)).item())
    valley = float(smooth[center_idx].item())
    peak_floor = min(
        float(torch.max(smooth[neg_mask]).item()) if bool(neg_mask.any()) else 0.0,
        float(torch.max(smooth[pos_mask]).item()) if bool(pos_mask.any()) else 0.0,
    )
    bimodal = (
        math.isfinite(neg_peak)
        and math.isfinite(pos_peak)
        and sigma > 0
        and neg_peak < -0.25 * sigma
        and pos_peak > 0.25 * sigma
        and valley < 0.85 * peak_floor
    )
    return {
        "sigma": sigma,
        "neg_peak": neg_peak,
        "pos_peak": pos_peak,
        "bimodal": bool(bimodal),
    }


def plot_feature_hist(delta_values: torch.Tensor, metadata: Dict[str, object], path: Path) -> None:
    sigma = float(metadata["sigma"])
    plt.figure(figsize=(10, 5))
    plt.hist(delta_values.detach().cpu().tolist(), bins=120, color="#4c78a8", alpha=0.9)
    plt.axvline(0.0, color="#222222", linestyle="--", linewidth=1.3, label="delta=0")
    plt.axvline(+sigma, color="#d95f02", linestyle=":", linewidth=1.2, label="+1sigma")
    plt.axvline(-sigma, color="#d95f02", linestyle=":", linewidth=1.2, label="-1sigma")
    plt.xlabel("delta")
    plt.ylabel("feature count")
    plt.title("MLP25 feature delta histogram")
    plt.grid(axis="y", alpha=0.25)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()


def build_overall_summary(
    patch_rows: Sequence[Dict[str, object]],
    ablation_rows: Sequence[Dict[str, object]],
    feature_rows: Sequence[Dict[str, object]],
    feature_metadata: Dict[str, object],
    top_tokens_payload: Dict[str, object] | None,
) -> str:
    patch_map = rows_by_component(patch_rows)
    l24 = patch_map["L24H30"]
    mlp25_patch = patch_map["MLP25"]
    mlp34 = patch_map["MLP34"]

    l24_destroy = float(l24["clean2corrupt"]["flip_rate"])
    l24_restore = float(l24["corrupt2clean"]["flip_rate"])
    mlp25_destroy = float(mlp25_patch["clean2corrupt"]["flip_rate"])
    mlp25_restore = float(mlp25_patch["corrupt2clean"]["flip_rate"])
    mlp34_total = float(mlp34["clean2corrupt"]["flip_rate"]) + float(mlp34["corrupt2clean"]["flip_rate"])
    hub_supported = (
        float(l24["clean2corrupt"]["mean_logit_delta"]) < 0.0
        and float(l24["corrupt2clean"]["mean_logit_delta"]) > 0.0
        and (l24_destroy + l24_restore) > mlp34_total
    )

    top3 = sorted(ablation_rows, key=lambda row: float(row["clean_logit_drop"]), reverse=True)[:3]
    top3_layers = [int(row["layer"]) for row in top3]
    dla_match = top3_layers == [35, 34, 33]
    mlp25_rank = 1 + next(
        idx for idx, row in enumerate(sorted(ablation_rows, key=lambda row: float(row["clean_logit_drop"]), reverse=True))
        if int(row["layer"]) == 25
    )

    top_feature = feature_rows[0]
    top_token_text = "未提取"
    if top_tokens_payload is not None:
        top_token_text = ",".join(top_tokens_payload.get(f"feature_{int(top_feature['feature_id'])}", {}).get("top5_tokens", [])[:5]) or "未提取"

    summary = (
        f"1. A：L24H30 flip={l24_destroy:.2%}/{l24_restore:.2%}，MLP25 flip={mlp25_destroy:.2%}/{mlp25_restore:.2%}；"
        f"相对 MLP34，结论{'支持' if hub_supported else '不支持'}路由枢纽假设。"
        f"2. B：clean_logit_drop 前三层为 L{top3_layers[0]}/L{top3_layers[1]}/L{top3_layers[2]}，"
        f"与 DLA {'一致' if dla_match else '不一致'}；MLP25 排第 {mlp25_rank}。"
        f"3. C：delta 分布{'呈双峰' if bool(feature_metadata['bimodal']) else '未见明显双峰'}，"
        f"top-1 feature={int(top_feature['feature_id'])}(delta={float(top_feature['delta']):.3f})，top tokens={top_token_text}。"
    )
    return summary[:300]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="8B routing validation experiments")
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH)
    parser.add_argument("--transcoder-path", type=Path, default=TRANSCODER_PATH)
    parser.add_argument("--manifest-path", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--max-pairs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--top-token-mining", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    args.output_root.mkdir(parents=True, exist_ok=True)

    model, tokenizer = load_hooked_qwen3(str(args.model_path), device=args.device, dtype=torch.bfloat16)
    tool_token_ids = tokenizer.encode(TOOL_CALL_STR, add_special_tokens=False)
    if len(tool_token_ids) != 1:
        raise ValueError(f"{TOOL_CALL_STR!r} encoded to {tool_token_ids}, expected one token.")
    tool_token_id = int(tool_token_ids[0])

    pairs = load_sample_pairs(args.manifest_path, model, max_pairs=args.max_pairs)
    pair_batches = build_pair_batches(pairs, batch_size=args.batch_size)
    print(f"[setup] loaded {len(pairs)} usable pairs from {args.manifest_path}", flush=True)
    print(f"[setup] batch_size={args.batch_size}  pair_batches={len(pair_batches)}", flush=True)

    patch_rows, patching_summary, clean_baseline_logits, clean_baseline_top1, corrupt_baseline_logits, corrupt_baseline_top1, patch_meta = run_patching_experiment(
        model,
        pair_batches,
        tool_token_id=tool_token_id,
        n_samples=len(pairs),
    )
    write_csv(args.output_root / "patching_results.csv", patch_rows)
    (args.output_root / "patching_summary.md").write_text("\n".join(patching_summary) + "\n", encoding="utf-8")

    ablation_rows = run_ablation_sweep(
        model,
        pair_batches,
        tool_token_id=tool_token_id,
        clean_baseline_logits=clean_baseline_logits,
        clean_baseline_top1=clean_baseline_top1,
        corrupt_baseline_logits=corrupt_baseline_logits,
        corrupt_baseline_top1=corrupt_baseline_top1,
    )
    write_csv(args.output_root / "ablation_sweep.csv", ablation_rows)
    plot_ablation_sweep(ablation_rows, args.output_root / "ablation_sweep.png")

    reference_tokens = pairs[0].clean_tokens_cpu.to(model.W_U.device)
    hook_name = resolve_mlp_input_hook(model, reference_tokens)
    feature_rows, top_tokens_payload, feature_metadata, delta_values = analyze_mlp25_features(
        model,
        tokenizer,
        pair_batches,
        hook_name=hook_name,
        layer_path=args.transcoder_path,
        top_token_mining=args.top_token_mining,
    )
    write_csv(args.output_root / "mlp25_features.csv", feature_rows)
    if top_tokens_payload is not None:
        write_json(args.output_root / "mlp25_features_top50_tokens.json", top_tokens_payload)
    plot_feature_hist(delta_values, feature_metadata, args.output_root / "mlp25_feature_hist.png")

    summary_text = build_overall_summary(
        patch_rows,
        ablation_rows,
        feature_rows,
        feature_metadata,
        top_tokens_payload,
    )
    (args.output_root / "summary.md").write_text(summary_text + "\n", encoding="utf-8")

    metadata = {
        "seed": args.seed,
        "model_path": str(args.model_path),
        "transcoder_path": str(args.transcoder_path),
        "manifest_path": str(args.manifest_path),
        "output_root": str(args.output_root),
        "tool_token_id": tool_token_id,
        "n_pairs": len(pairs),
        "sample_ids": [pair.sample_id for pair in pairs],
        "patching": patch_meta,
        "feature_analysis": {
            "hook_name": hook_name,
            "top_token_mining": bool(args.top_token_mining),
            **feature_metadata,
        },
    }
    write_json(args.output_root / "metadata.json", metadata)

    del model
    clear_cuda()


if __name__ == "__main__":
    main()
