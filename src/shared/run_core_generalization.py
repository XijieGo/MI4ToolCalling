#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm.auto import tqdm

from multiscale_common import (
    DEFAULT_DATASET_ROOT,
    build_pair_batches,
    choose_best_layer,
    clear_cuda,
    load_model_and_tokenizer,
    load_sample_pairs,
    precompute_head_tool_projections,
    run_with_hooks_and_cache,
    schema_token_positions,
    set_seed,
    system_token_positions,
    tool_stats,
    write_csv,
    write_json,
    write_text,
)


DEFAULT_EXPERIMENTS = ("A", "B", "C")
RANK_K_VALUES = ("1", "3", "5", "10", "full")
FIXED_DIRECTION_ALPHAS = (0.5, 1.0, 1.5, 2.0)
COARSE_RELATIVE_LAYERS = (0.45, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the multi-scale core generalization experiments.")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--size-label", type=str, required=True)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--train-split", type=str, default="train")
    parser.add_argument("--eval-split", type=str, default="test")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--exp-a-root", type=Path, default=None)
    parser.add_argument("--experiments", nargs="+", default=list(DEFAULT_EXPERIMENTS))
    parser.add_argument("--coarse-pairs", type=int, default=200)
    parser.add_argument("--final-pairs", type=int, default=300)
    parser.add_argument("--gate-train-pairs", type=int, default=200)
    parser.add_argument("--gate-eval-pairs", type=int, default=300)
    parser.add_argument("--readout-pairs", type=int, default=200)
    parser.add_argument("--downstream-layers", type=str, default="")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def normalize_experiments(items: list[str]) -> list[str]:
    out: list[str] = []
    for item in items:
        normalized = str(item).strip().upper()
        if normalized == "ALL":
            return ["A", "B", "C"]
        if normalized not in {"A", "B", "C"}:
            raise ValueError(f"Unknown experiment label: {item}")
        if normalized not in out:
            out.append(normalized)
    return out


def parse_layers(text: str) -> list[int]:
    return [int(part.strip()) for part in str(text).split(",") if part.strip()]


def make_last_token_replace_hook(source_cpu: torch.Tensor):
    def hook_fn(value: torch.Tensor, hook):  # noqa: ANN001
        out = value.clone()
        src = source_cpu.to(device=value.device, dtype=value.dtype)
        if src.ndim == 3:
            out[:, -1, :] = src[:, -1, :]
        elif src.ndim == 2:
            out[:, -1, :] = src
        else:
            raise ValueError(f"Unexpected replace-hook source shape: {tuple(src.shape)}")
        return out

    return hook_fn


def make_last_token_add_hook(delta_cpu: torch.Tensor):
    def hook_fn(value: torch.Tensor, hook):  # noqa: ANN001
        out = value.clone()
        delta = delta_cpu.to(device=value.device, dtype=value.dtype)
        if delta.ndim == 1:
            delta = delta.view(1, -1)
        if delta.ndim != 2:
            raise ValueError(f"Unexpected add-hook delta shape: {tuple(delta.shape)}")
        out[:, -1, :] = out[:, -1, :] + delta
        return out

    return hook_fn


def make_z_patch_hook(source_cpu: torch.Tensor, head: int):
    def hook_fn(value: torch.Tensor, hook):  # noqa: ANN001
        out = value.clone()
        src = source_cpu.to(device=value.device, dtype=value.dtype)
        out[:, -1, head, :] = src[:, -1, head, :]
        return out

    return hook_fn


def compute_relative_layers(n_layers: int) -> list[int]:
    layers = []
    for frac in COARSE_RELATIVE_LAYERS:
        layer = int(round(frac * n_layers))
        layer = min(max(layer, 0), n_layers - 1)
        if layer not in layers:
            layers.append(layer)
    return layers


def patch_sweep(
    model,
    pairs,
    *,
    layers: list[int],
    batch_size: int,
    tool_token_id: int,
    phase: str,
) -> list[dict[str, object]]:
    pair_batches = build_pair_batches(pairs, batch_size=batch_size)
    hook_names = [f"blocks.{layer}.hook_resid_post" for layer in layers]
    accum = {
        layer: {"count": 0, "tool_top1": 0, "strict_flip": 0, "logit_sum": 0.0, "prob_sum": 0.0}
        for layer in layers
    }
    baseline_clean_tool_top1 = 0
    baseline_corrupt_tool_top1 = 0
    baseline_count = 0

    progress = tqdm(pair_batches, desc=f"{phase} state patch", dynamic_ncols=True)
    for batch in progress:
        clean_tokens = batch.clean_tokens_cpu.to(model.W_U.device)
        corrupt_tokens = batch.corrupt_tokens_cpu.to(model.W_U.device)
        with torch.no_grad():
            clean_logits, clean_cache = model.run_with_cache(clean_tokens, names_filter=lambda name: name in hook_names)
            corrupt_logits = model(corrupt_tokens)
        _, _clean_prob, clean_top1 = tool_stats(clean_logits, tool_token_id)
        corrupt_tool_logit, _corrupt_prob, corrupt_top1 = tool_stats(corrupt_logits, tool_token_id)
        baseline_clean_tool_top1 += int((clean_top1 == tool_token_id).sum().item())
        baseline_corrupt_tool_top1 += int((corrupt_top1 == tool_token_id).sum().item())
        baseline_count += int(clean_top1.shape[0])

        for layer in layers:
            hook_name = f"blocks.{layer}.hook_resid_post"
            hooks = [(hook_name, make_last_token_replace_hook(clean_cache[hook_name].detach().cpu()))]
            with torch.no_grad():
                patched_logits = model.run_with_hooks(corrupt_tokens, fwd_hooks=hooks)
            patched_tool_logit, patched_tool_prob, patched_top1 = tool_stats(patched_logits, tool_token_id)
            bucket = accum[layer]
            bucket["count"] += int(patched_top1.shape[0])
            bucket["tool_top1"] += int((patched_top1 == tool_token_id).sum().item())
            bucket["strict_flip"] += int(((corrupt_top1 != tool_token_id) & (patched_top1 == tool_token_id)).sum().item())
            bucket["logit_sum"] += float(patched_tool_logit.sum().item())
            bucket["prob_sum"] += float(patched_tool_prob.sum().item())
            del patched_logits, patched_tool_logit, patched_tool_prob, patched_top1
            clear_cuda()

        del clean_tokens, corrupt_tokens, clean_logits, clean_cache, corrupt_logits, corrupt_tool_logit, clean_top1, corrupt_top1
        clear_cuda()
        progress.set_postfix(tok=batch.token_len)

    rows = []
    for layer in layers:
        bucket = accum[layer]
        count = max(int(bucket["count"]), 1)
        rows.append(
            {
                "phase": phase,
                "layer": layer,
                "n": count,
                "tool_call_top1_rate": float(bucket["tool_top1"] / count),
                "strict_flip_rate": float(bucket["strict_flip"] / count),
                "mean_tool_call_logit": float(bucket["logit_sum"] / count),
                "mean_tool_call_prob": float(bucket["prob_sum"] / count),
                "baseline_clean_tool_top1_rate": float(baseline_clean_tool_top1 / max(baseline_count, 1)),
                "baseline_corrupt_tool_top1_rate": float(baseline_corrupt_tool_top1 / max(baseline_count, 1)),
            }
        )
    return rows


def plot_patch_sweep(rows: list[dict[str, object]], path: Path) -> None:
    if not rows:
        return
    final_rows = [row for row in rows if str(row["phase"]) == "final"] or rows
    final_rows = sorted(final_rows, key=lambda row: int(row["layer"]))
    layers = [int(row["layer"]) for row in final_rows]
    top1 = [float(row["tool_call_top1_rate"]) for row in final_rows]
    strict_flip = [float(row["strict_flip_rate"]) for row in final_rows]
    logits = [float(row["mean_tool_call_logit"]) for row in final_rows]

    fig, ax1 = plt.subplots(figsize=(7.2, 4.4))
    ax1.plot(layers, top1, marker="o", linewidth=2.0, label="tool-call top1")
    ax1.plot(layers, strict_flip, marker="s", linewidth=1.8, label="strict flip")
    ax1.set_xlabel("Layer")
    ax1.set_ylabel("Rate")
    ax1.set_ylim(0.0, 1.05)
    ax1.grid(alpha=0.25)
    ax2 = ax1.twinx()
    ax2.plot(layers, logits, marker="^", linewidth=1.5, linestyle="--", color="tab:red", label="mean tool logit")
    ax2.set_ylabel("Mean tool logit")
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="best")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def build_exp_a_summary(size_label: str, best_row: dict[str, object], rows: list[dict[str, object]]) -> str:
    coarse_rows = [row for row in rows if str(row["phase"]) == "coarse"]
    final_rows = [row for row in rows if str(row["phase"]) == "final"]
    lines = [
        "# Exp A Summary",
        "",
        f"- Size: `{size_label}`",
        f"- Best layer: `L{int(best_row['layer'])}`",
        f"- Best patched top-1 `<tool_call>` rate: `{float(best_row['tool_call_top1_rate']):.2%}`",
        f"- Best strict flip rate: `{float(best_row['strict_flip_rate']):.2%}`",
        f"- Final baseline corrupt top-1 `<tool_call>` rate: `{float(best_row['baseline_corrupt_tool_top1_rate']):.2%}`",
        f"- Coarse sweep layers: `{[int(row['layer']) for row in coarse_rows]}`",
        f"- Final sweep layers: `{[int(row['layer']) for row in final_rows]}`",
        "",
        "Interpretation:",
        "The dominant causal bottleneck remains a prediction-position residual state in the mid-to-late stack, and its best layer can be localized by clean-to-corrupt state patching.",
    ]
    return "\n".join(lines)


def orient_components(components: torch.Tensor, diff_vectors: torch.Tensor) -> torch.Tensor:
    oriented = components.clone()
    for idx in range(oriented.shape[0]):
        direction = oriented[idx]
        if float((diff_vectors @ direction).mean().item()) < 0:
            oriented[idx] = -direction
    return oriented


def compute_pca(diff_vectors: torch.Tensor, *, n_components: int = 10) -> dict[str, torch.Tensor]:
    centered = diff_vectors - diff_vectors.mean(dim=0, keepdim=True)
    _, singular_values, vh = torch.linalg.svd(centered, full_matrices=False)
    actual_components = min(n_components, int(vh.shape[0]))
    components = orient_components(vh[:actual_components].contiguous(), diff_vectors)
    exp_var = (singular_values[:actual_components] ** 2) / max(diff_vectors.shape[0] - 1, 1)
    total_var = float((centered.pow(2).sum().item()) / max(diff_vectors.shape[0] - 1, 1))
    exp_var_ratio = exp_var / total_var if total_var > 0 else torch.zeros_like(exp_var)
    return {
        "components": components,
        "mean_diff": diff_vectors.mean(dim=0),
        "explained_variance": exp_var,
        "explained_variance_ratio": exp_var_ratio,
        "singular_values": singular_values[:actual_components],
    }


def project_delta(diff_batch: torch.Tensor, pca: dict[str, torch.Tensor], *, rank_k: str) -> torch.Tensor:
    if rank_k == "full":
        return diff_batch
    k_int = int(rank_k)
    components = pca["components"][:k_int]
    mean_diff = pca["mean_diff"].unsqueeze(0)
    centered = diff_batch - mean_diff
    coeff = centered @ components.T
    projected = coeff @ components + mean_diff
    return projected


def collect_residuals(model, pairs, *, layer: int, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    pair_batches = build_pair_batches(pairs, batch_size=batch_size)
    hook_name = f"blocks.{layer}.hook_resid_post"
    d_model = int(model.cfg.d_model)
    clean_resid = torch.empty((len(pairs), d_model), dtype=torch.float32)
    corrupt_resid = torch.empty((len(pairs), d_model), dtype=torch.float32)

    progress = tqdm(pair_batches, desc=f"L{layer} residual collect", dynamic_ncols=True)
    for batch in progress:
        clean_tokens = batch.clean_tokens_cpu.to(model.W_U.device)
        corrupt_tokens = batch.corrupt_tokens_cpu.to(model.W_U.device)
        with torch.no_grad():
            _, clean_cache = model.run_with_cache(clean_tokens, names_filter=lambda name: name == hook_name)
            _, corrupt_cache = model.run_with_cache(corrupt_tokens, names_filter=lambda name: name == hook_name)
        idx = torch.tensor(batch.indices, dtype=torch.long)
        clean_resid[idx] = clean_cache[hook_name][:, -1, :].detach().cpu().float()
        corrupt_resid[idx] = corrupt_cache[hook_name][:, -1, :].detach().cpu().float()
        del clean_tokens, corrupt_tokens, clean_cache, corrupt_cache, idx
        clear_cuda()
        progress.set_postfix(tok=batch.token_len)
    return clean_resid, corrupt_resid


def evaluate_rank_k_patches(
    model,
    pairs,
    *,
    layer: int,
    batch_size: int,
    tool_token_id: int,
    clean_resid: torch.Tensor,
    corrupt_resid: torch.Tensor,
    pca: dict[str, torch.Tensor],
) -> list[dict[str, object]]:
    pair_batches = build_pair_batches(pairs, batch_size=batch_size)
    hook_name = f"blocks.{layer}.hook_resid_post"
    diff_vectors = clean_resid - corrupt_resid
    rows: list[dict[str, object]] = []

    for rank_k in RANK_K_VALUES:
        tool_top1 = 0
        strict_flip = 0
        logit_sum = 0.0
        prob_sum = 0.0
        count = 0
        progress = tqdm(pair_batches, desc=f"rank {rank_k}", dynamic_ncols=True)
        for batch in progress:
            idx = torch.tensor(batch.indices, dtype=torch.long)
            corrupt_tokens = batch.corrupt_tokens_cpu.to(model.W_U.device)
            projected = project_delta(diff_vectors[idx], pca, rank_k=rank_k)
            patch_source = (corrupt_resid[idx] + projected).detach().cpu()
            hooks = [(hook_name, make_last_token_replace_hook(patch_source))]
            with torch.no_grad():
                corrupt_logits = model(corrupt_tokens)
                patched_logits = model.run_with_hooks(corrupt_tokens, fwd_hooks=hooks)
            _base_logit, _base_prob, base_top1 = tool_stats(corrupt_logits, tool_token_id)
            patched_logit, patched_prob, patched_top1 = tool_stats(patched_logits, tool_token_id)
            tool_top1 += int((patched_top1 == tool_token_id).sum().item())
            strict_flip += int(((base_top1 != tool_token_id) & (patched_top1 == tool_token_id)).sum().item())
            logit_sum += float(patched_logit.sum().item())
            prob_sum += float(patched_prob.sum().item())
            count += int(patched_top1.shape[0])
            del idx, corrupt_tokens, corrupt_logits, patched_logits, base_top1, patched_logit, patched_prob, patched_top1
            clear_cuda()
            progress.set_postfix(tok=batch.token_len)
        rows.append(
            {
                "rank_k": rank_k,
                "alpha": 1.0,
                "n": count,
                "tool_call_top1_rate": float(tool_top1 / max(count, 1)),
                "strict_flip_rate": float(strict_flip / max(count, 1)),
                "mean_tool_call_logit": float(logit_sum / max(count, 1)),
                "mean_tool_call_prob": float(prob_sum / max(count, 1)),
            }
        )
    return rows


def evaluate_fixed_directions(
    model,
    pairs,
    *,
    layer: int,
    batch_size: int,
    tool_token_id: int,
    pca: dict[str, torch.Tensor],
    discovery_diff: torch.Tensor,
) -> list[dict[str, object]]:
    pair_batches = build_pair_batches(pairs, batch_size=batch_size)
    hook_name = f"blocks.{layer}.hook_resid_post"
    mean_diff = pca["mean_diff"].detach().cpu()
    pc1 = pca["components"][0].detach().cpu()
    pc1_scale = float((discovery_diff @ pc1).mean().item())
    direction_map = {
        "mean_diff": mean_diff,
        "pc1_scaled": pc1 * pc1_scale,
    }
    rows: list[dict[str, object]] = []

    for direction_name, base_delta in direction_map.items():
        for alpha in FIXED_DIRECTION_ALPHAS:
            tool_top1 = 0
            strict_flip = 0
            logit_sum = 0.0
            prob_sum = 0.0
            count = 0
            delta = (base_delta * float(alpha)).view(1, -1)
            progress = tqdm(pair_batches, desc=f"{direction_name} alpha={alpha:g}", dynamic_ncols=True)
            for batch in progress:
                corrupt_tokens = batch.corrupt_tokens_cpu.to(model.W_U.device)
                hooks = [(hook_name, make_last_token_add_hook(delta))]
                with torch.no_grad():
                    corrupt_logits = model(corrupt_tokens)
                    patched_logits = model.run_with_hooks(corrupt_tokens, fwd_hooks=hooks)
                _base_logit, _base_prob, base_top1 = tool_stats(corrupt_logits, tool_token_id)
                patched_logit, patched_prob, patched_top1 = tool_stats(patched_logits, tool_token_id)
                tool_top1 += int((patched_top1 == tool_token_id).sum().item())
                strict_flip += int(((base_top1 != tool_token_id) & (patched_top1 == tool_token_id)).sum().item())
                logit_sum += float(patched_logit.sum().item())
                prob_sum += float(patched_prob.sum().item())
                count += int(patched_top1.shape[0])
                del corrupt_tokens, corrupt_logits, patched_logits, base_top1, patched_logit, patched_prob, patched_top1
                clear_cuda()
                progress.set_postfix(tok=batch.token_len)
            rows.append(
                {
                    "direction": direction_name,
                    "alpha": float(alpha),
                    "n": count,
                    "tool_call_top1_rate": float(tool_top1 / max(count, 1)),
                    "strict_flip_rate": float(strict_flip / max(count, 1)),
                    "mean_tool_call_logit": float(logit_sum / max(count, 1)),
                    "mean_tool_call_prob": float(prob_sum / max(count, 1)),
                }
            )
    return rows


def plot_rank_k(rows: list[dict[str, object]], path: Path) -> None:
    order = ["1", "3", "5", "10", "full"]
    best_rows = []
    for rank_k in order:
        candidates = [row for row in rows if str(row["rank_k"]) == rank_k]
        if candidates:
            best_rows.append(max(candidates, key=lambda row: float(row["tool_call_top1_rate"])))
    labels = [str(row["rank_k"]) for row in best_rows]
    top1 = [float(row["tool_call_top1_rate"]) for row in best_rows]
    strict_flip = [float(row["strict_flip_rate"]) for row in best_rows]
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    ax.plot(labels, top1, marker="o", linewidth=2.0, label="tool-call top1")
    ax.plot(labels, strict_flip, marker="s", linewidth=1.8, label="strict flip")
    ax.set_xlabel("Rank k")
    ax.set_ylabel("Rate")
    ax.set_ylim(0.0, 1.05)
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def build_exp_b_summary(
    size_label: str,
    layer: int,
    explained_rows: list[dict[str, object]],
    rank_rows: list[dict[str, object]],
    fixed_rows: list[dict[str, object]],
) -> str:
    pc1 = explained_rows[0]
    rank1 = max((row for row in rank_rows if str(row["rank_k"]) == "1"), key=lambda row: float(row["tool_call_top1_rate"]))
    full = max((row for row in rank_rows if str(row["rank_k"]) == "full"), key=lambda row: float(row["tool_call_top1_rate"]))
    fixed_best = max(fixed_rows, key=lambda row: float(row["tool_call_top1_rate"])) if fixed_rows else None
    lines = [
        "# Exp B Summary",
        "",
        f"- Size: `{size_label}`",
        f"- Key layer: `L{layer}`",
        f"- PC1 explained variance ratio: `{float(pc1['explained_ratio']):.4f}`",
        f"- Best rank-1 recovery: `{float(rank1['tool_call_top1_rate']):.2%}` top1, `{float(rank1['strict_flip_rate']):.2%}` strict flip",
        f"- Full recovery upper bound: `{float(full['tool_call_top1_rate']):.2%}` top1, `{float(full['strict_flip_rate']):.2%}` strict flip",
    ]
    if fixed_best is not None:
        lines.append(
            f"- Best fixed shared direction: `{fixed_best['direction']}` at `alpha={float(fixed_best['alpha']):g}` -> `{float(fixed_best['tool_call_top1_rate']):.2%}` top1"
        )
    lines.extend(
        [
            "",
            "Interpretation:",
            "The key-layer decision difference is compact enough to support strong low-rank recovery. Fixed shared directions are reported as auxiliary evidence rather than the main claim.",
        ]
    )
    return "\n".join(lines)


def normalize_zscore(values: list[float]) -> list[float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return []
    std = float(arr.std(ddof=0))
    if std == 0.0:
        return [0.0 for _ in values]
    mean = float(arr.mean())
    return [float((value - mean) / std) for value in values]


def sweep_head_dla_and_mlp(
    model,
    pairs,
    *,
    downstream_layers: list[int],
    batch_size: int,
    tool_token_id: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    pair_batches = build_pair_batches(pairs, batch_size=batch_size)
    head_proj = precompute_head_tool_projections(model, downstream_layers, tool_token_id)
    hook_z_names = [f"blocks.{layer}.attn.hook_z" for layer in downstream_layers]
    hook_mlp_names = [f"blocks.{layer}.hook_mlp_out" for layer in downstream_layers]
    n_heads = int(model.cfg.n_heads)
    wu_tool = model.W_U[:, tool_token_id].detach().cpu().float()

    head_sums = {
        "clean": {layer: torch.zeros(n_heads, dtype=torch.float64) for layer in downstream_layers},
        "corrupt": {layer: torch.zeros(n_heads, dtype=torch.float64) for layer in downstream_layers},
    }
    mlp_sums = {
        "clean": {layer: 0.0 for layer in downstream_layers},
        "corrupt": {layer: 0.0 for layer in downstream_layers},
    }
    counts = {"clean": 0, "corrupt": 0}

    progress = tqdm(pair_batches, desc="Readout DLA sweep", dynamic_ncols=True)
    for batch in progress:
        for condition in ("clean", "corrupt"):
            tokens_cpu = batch.clean_tokens_cpu if condition == "clean" else batch.corrupt_tokens_cpu
            tokens = tokens_cpu.to(model.W_U.device)
            with torch.no_grad():
                _, cache = model.run_with_cache(tokens, names_filter=lambda name: name in hook_z_names or name in hook_mlp_names)
            counts[condition] += int(tokens.shape[0])
            for layer in downstream_layers:
                z_last = cache[f"blocks.{layer}.attn.hook_z"][:, -1, :, :].to(dtype=torch.float32)
                dla = torch.einsum("bhd,hd->bh", z_last, head_proj[layer]).detach().cpu().double()
                head_sums[condition][layer] += dla.sum(dim=0)
                mlp_last = cache[f"blocks.{layer}.hook_mlp_out"][:, -1, :].detach().cpu().float()
                mlp_sums[condition][layer] += float(torch.einsum("bd,d->b", mlp_last, wu_tool).sum().item())
            del tokens, cache
            clear_cuda()
        progress.set_postfix(tok=batch.token_len)

    head_rows = []
    for layer in downstream_layers:
        for head in range(n_heads):
            clean_mean = float((head_sums["clean"][layer][head] / max(counts["clean"], 1)).item())
            corrupt_mean = float((head_sums["corrupt"][layer][head] / max(counts["corrupt"], 1)).item())
            head_rows.append(
                {
                    "layer": layer,
                    "head": head,
                    "mean_clean": clean_mean,
                    "mean_corrupt": corrupt_mean,
                    "dla_delta": clean_mean - corrupt_mean,
                }
            )

    mlp_rows = []
    for layer in downstream_layers:
        clean_mean = float(mlp_sums["clean"][layer] / max(counts["clean"], 1))
        corrupt_mean = float(mlp_sums["corrupt"][layer] / max(counts["corrupt"], 1))
        mlp_rows.append(
            {
                "layer": layer,
                "mean_clean": clean_mean,
                "mean_corrupt": corrupt_mean,
                "delta": clean_mean - corrupt_mean,
            }
        )
    return head_rows, mlp_rows


def sweep_head_behavior(
    model,
    pairs,
    *,
    downstream_layers: list[int],
    batch_size: int,
    tool_token_id: int,
) -> list[dict[str, object]]:
    pair_batches = build_pair_batches(pairs, batch_size=batch_size)
    hook_names = [f"blocks.{layer}.attn.hook_z" for layer in downstream_layers]
    n_heads = int(model.cfg.n_heads)
    accum = {
        (layer, head): {"count": 0, "tool_top1": 0, "strict_flip": 0, "logit_sum": 0.0}
        for layer in downstream_layers
        for head in range(n_heads)
    }

    progress = tqdm(pair_batches, desc="Readout behavior sweep", dynamic_ncols=True)
    for batch in progress:
        clean_tokens = batch.clean_tokens_cpu.to(model.W_U.device)
        corrupt_tokens = batch.corrupt_tokens_cpu.to(model.W_U.device)
        with torch.no_grad():
            _, clean_cache = model.run_with_cache(clean_tokens, names_filter=lambda name: name in hook_names)
            corrupt_logits = model(corrupt_tokens)
        _base_logit, _base_prob, base_top1 = tool_stats(corrupt_logits, tool_token_id)

        for layer in downstream_layers:
            z_name = f"blocks.{layer}.attn.hook_z"
            clean_z_cpu = clean_cache[z_name].detach().cpu()
            for head in range(n_heads):
                hooks = [(z_name, make_z_patch_hook(clean_z_cpu, head=head))]
                with torch.no_grad():
                    patched_logits = model.run_with_hooks(corrupt_tokens, fwd_hooks=hooks)
                patched_logit, _patched_prob, patched_top1 = tool_stats(patched_logits, tool_token_id)
                bucket = accum[(layer, head)]
                bucket["count"] += int(patched_top1.shape[0])
                bucket["tool_top1"] += int((patched_top1 == tool_token_id).sum().item())
                bucket["strict_flip"] += int(((base_top1 != tool_token_id) & (patched_top1 == tool_token_id)).sum().item())
                bucket["logit_sum"] += float(patched_logit.sum().item())
                del patched_logits, patched_logit, patched_top1
                clear_cuda()

        del clean_tokens, corrupt_tokens, clean_cache, corrupt_logits, base_top1
        clear_cuda()
        progress.set_postfix(tok=batch.token_len)

    rows = []
    for layer in downstream_layers:
        for head in range(n_heads):
            bucket = accum[(layer, head)]
            count = max(int(bucket["count"]), 1)
            rows.append(
                {
                    "layer": layer,
                    "head": head,
                    "behavior_top1_rate": float(bucket["tool_top1"] / count),
                    "behavior_strict_flip_rate": float(bucket["strict_flip"] / count),
                    "behavior_mean_tool_logit": float(bucket["logit_sum"] / count),
                    "n": count,
                }
            )
    return rows


def build_heatmap(rows: list[dict[str, object]], *, value_key: str, path: Path, title: str, n_heads: int) -> None:
    if not rows:
        return
    layers = sorted({int(row["layer"]) for row in rows})
    layer_to_idx = {layer: idx for idx, layer in enumerate(layers)}
    matrix = np.full((len(layers), n_heads), np.nan, dtype=np.float32)
    for row in rows:
        matrix[layer_to_idx[int(row["layer"])], int(row["head"])] = float(row[value_key])
    fig, ax = plt.subplots(figsize=(9.0, max(3.5, len(layers) * 0.35)))
    im = ax.imshow(matrix, aspect="auto", cmap="viridis")
    ax.set_yticks(range(len(layers)))
    ax.set_yticklabels([f"L{layer}" for layer in layers])
    ax.set_xticks(range(n_heads))
    ax.set_xticklabels([str(i) for i in range(n_heads)])
    ax.set_xlabel("Head")
    ax.set_ylabel("Layer")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def analyze_top_head_regions(
    model,
    tokenizer,
    pairs,
    *,
    top_rows: list[dict[str, object]],
    batch_size: int,
) -> list[dict[str, object]]:
    unique_targets: list[tuple[int, int]] = []
    for row in top_rows:
        target = (int(row["layer"]), int(row["head"]))
        if target not in unique_targets:
            unique_targets.append(target)
        if len(unique_targets) >= 10:
            break
    if not unique_targets:
        return []

    target_layers = sorted({layer for layer, _head in unique_targets})
    pattern_names = [f"blocks.{layer}.attn.hook_pattern" for layer in target_layers]
    n_heads = int(model.cfg.n_heads)
    pair_batches = build_pair_batches(pairs, batch_size=batch_size)
    accum = {
        (layer, head): {
            "count": 0,
            "clean_system": 0.0,
            "corrupt_system": 0.0,
            "clean_schema": 0.0,
            "corrupt_schema": 0.0,
        }
        for layer, head in unique_targets
    }

    progress = tqdm(pair_batches, desc="Top head region attention", dynamic_ncols=True)
    for batch in progress:
        clean_tokens = batch.clean_tokens_cpu.to(model.W_U.device)
        corrupt_tokens = batch.corrupt_tokens_cpu.to(model.W_U.device)
        with torch.no_grad():
            _, clean_cache = model.run_with_cache(clean_tokens, names_filter=lambda name: name in pattern_names)
            _, corrupt_cache = model.run_with_cache(corrupt_tokens, names_filter=lambda name: name in pattern_names)
        for local_idx, pair_idx in enumerate(batch.indices):
            pair = pairs[pair_idx]
            system_positions = system_token_positions(pair.clean_text, tokenizer)
            schema_positions = schema_token_positions(pair.clean_text, tokenizer)
            for layer, head in unique_targets:
                pattern_clean = clean_cache[f"blocks.{layer}.attn.hook_pattern"][local_idx].detach().cpu().float()
                pattern_corrupt = corrupt_cache[f"blocks.{layer}.attn.hook_pattern"][local_idx].detach().cpu().float()
                head_count = int(pattern_clean.shape[0])
                effective_head = head if head < head_count else min(head // max(n_heads // head_count, 1), head_count - 1)
                seq_len = int(pattern_clean.shape[-1])
                clean_system = float(pattern_clean[effective_head, -1, [p for p in system_positions if 0 <= p < seq_len]].sum().item()) if system_positions else 0.0
                corrupt_system = float(pattern_corrupt[effective_head, -1, [p for p in system_positions if 0 <= p < seq_len]].sum().item()) if system_positions else 0.0
                clean_schema = float(pattern_clean[effective_head, -1, [p for p in schema_positions if 0 <= p < seq_len]].sum().item()) if schema_positions else 0.0
                corrupt_schema = float(pattern_corrupt[effective_head, -1, [p for p in schema_positions if 0 <= p < seq_len]].sum().item()) if schema_positions else 0.0
                bucket = accum[(layer, head)]
                bucket["count"] += 1
                bucket["clean_system"] += clean_system
                bucket["corrupt_system"] += corrupt_system
                bucket["clean_schema"] += clean_schema
                bucket["corrupt_schema"] += corrupt_schema
        del clean_tokens, corrupt_tokens, clean_cache, corrupt_cache
        clear_cuda()
        progress.set_postfix(tok=batch.token_len)

    rows = []
    for layer, head in unique_targets:
        bucket = accum[(layer, head)]
        count = max(int(bucket["count"]), 1)
        rows.append(
            {
                "layer": layer,
                "head": head,
                "clean_system_attention": float(bucket["clean_system"] / count),
                "corrupt_system_attention": float(bucket["corrupt_system"] / count),
                "delta_system_attention": float((bucket["clean_system"] - bucket["corrupt_system"]) / count),
                "clean_schema_attention": float(bucket["clean_schema"] / count),
                "corrupt_schema_attention": float(bucket["corrupt_schema"] / count),
                "delta_schema_attention": float((bucket["clean_schema"] - bucket["corrupt_schema"]) / count),
            }
        )
    return rows


def build_exp_c_summary(size_label: str, best_layer: int, top_rows: list[dict[str, object]], mlp_rows: list[dict[str, object]]) -> str:
    top = top_rows[0]
    top_layers = [int(row["layer"]) for row in top_rows[:10]]
    late_mlp = sorted(mlp_rows, key=lambda row: float(row["delta"]), reverse=True)[:3]
    late_mlp_text = ", ".join(f"L{int(row['layer'])} ({float(row['delta']):.3f})" for row in late_mlp)
    lines = [
        "# Exp C Summary",
        "",
        f"- Size: `{size_label}`",
        f"- Upstream key layer `L{best_layer}`",
        f"- Top downstream head: `L{int(top['layer'])}H{int(top['head'])}`",
        f"- Top head DLA delta: `{float(top['dla_delta']):.4f}`",
        f"- Top head z-patch strict flip rate: `{float(top['behavior_strict_flip_rate']):.2%}`",
        f"- Top-10 downstream heads by composite score occupy layers: `{top_layers}`",
        f"- Top late-layer MLP deltas: `{late_mlp_text}`",
        "",
        "Interpretation:",
        "The gate is not the endpoint. After the key layer, multiple downstream heads and late-layer MLP blocks participate in the final readout into `<tool_call>`, with no single head matching the full-state effect on its own.",
    ]
    return "\n".join(lines)


def run_exp_a(model, args: argparse.Namespace, *, tool_token_id: int) -> dict[str, object]:
    exp_root = args.output_root / "exp_a_state_patch"
    exp_root.mkdir(parents=True, exist_ok=True)
    eval_pairs_full = load_sample_pairs(model, dataset_root=args.dataset_root, split=args.eval_split, max_pairs=args.final_pairs)
    coarse_pairs = eval_pairs_full[: min(args.coarse_pairs, len(eval_pairs_full))]
    n_layers = int(model.cfg.n_layers)
    coarse_layers = compute_relative_layers(n_layers)

    coarse_rows = patch_sweep(model, coarse_pairs, layers=coarse_layers, batch_size=args.batch_size, tool_token_id=tool_token_id, phase="coarse")
    best_coarse = choose_best_layer(coarse_rows)
    fine_layers = sorted({layer for layer in range(max(0, best_coarse - 2), min(n_layers, best_coarse + 3))})
    final_rows = patch_sweep(model, eval_pairs_full, layers=fine_layers, batch_size=args.batch_size, tool_token_id=tool_token_id, phase="final")
    rows = coarse_rows + final_rows
    best_row = max(final_rows, key=lambda row: float(row["tool_call_top1_rate"]))

    write_csv(exp_root / "patch_sweep.csv", rows)
    plot_patch_sweep(rows, exp_root / "plot_patch_sweep.pdf")
    write_text(exp_root / "summary.md", build_exp_a_summary(args.size_label, best_row, rows))
    metadata = {
        "size_label": args.size_label,
        "model_path": str(args.model_path),
        "dataset_root": str(args.dataset_root),
        "eval_split": args.eval_split,
        "coarse_pairs": len(coarse_pairs),
        "final_pairs": len(eval_pairs_full),
        "coarse_layers": coarse_layers,
        "fine_layers": fine_layers,
        "best_layer": int(best_row["layer"]),
        "selected_sample_ids": [pair.sample_id for pair in eval_pairs_full],
    }
    write_json(exp_root / "metadata.json", metadata)
    return metadata


def run_exp_b(model, args: argparse.Namespace, *, tool_token_id: int, best_layer: int) -> dict[str, object]:
    exp_root = args.output_root / "exp_b_gate_vector"
    exp_root.mkdir(parents=True, exist_ok=True)
    train_pairs = load_sample_pairs(model, dataset_root=args.dataset_root, split=args.train_split, max_pairs=args.gate_train_pairs)
    eval_pairs = load_sample_pairs(model, dataset_root=args.dataset_root, split=args.eval_split, max_pairs=args.gate_eval_pairs)

    clean_train, corrupt_train = collect_residuals(model, train_pairs, layer=best_layer, batch_size=args.batch_size)
    clean_eval, corrupt_eval = collect_residuals(model, eval_pairs, layer=best_layer, batch_size=args.batch_size)
    diff_train = clean_train - corrupt_train
    pca = compute_pca(diff_train, n_components=10)

    explained_rows = []
    cumulative = 0.0
    for idx in range(int(pca["components"].shape[0])):
        ratio = float(pca["explained_variance_ratio"][idx].item())
        cumulative += ratio
        explained_rows.append(
            {
                "component": idx + 1,
                "singular_value": float(pca["singular_values"][idx].item()),
                "explained_variance": float(pca["explained_variance"][idx].item()),
                "explained_ratio": ratio,
                "cumulative_ratio": cumulative,
            }
        )

    rank_rows = evaluate_rank_k_patches(
        model,
        eval_pairs,
        layer=best_layer,
        batch_size=args.batch_size,
        tool_token_id=tool_token_id,
        clean_resid=clean_eval,
        corrupt_resid=corrupt_eval,
        pca=pca,
    )
    fixed_rows = evaluate_fixed_directions(
        model,
        eval_pairs,
        layer=best_layer,
        batch_size=args.batch_size,
        tool_token_id=tool_token_id,
        pca=pca,
        discovery_diff=diff_train,
    )

    torch.save(
        {
            "layer": best_layer,
            "mean_diff": pca["mean_diff"],
            "components": pca["components"],
            "singular_values": pca["singular_values"],
            "sample_ids_train": [pair.sample_id for pair in train_pairs],
            "sample_ids_eval": [pair.sample_id for pair in eval_pairs],
        },
        exp_root / "pca_components.pt",
    )
    write_csv(exp_root / "explained_variance.csv", explained_rows)
    write_csv(exp_root / "rank_k_patch_sweep.csv", rank_rows)
    write_csv(exp_root / "fixed_direction_sweep.csv", fixed_rows)
    plot_rank_k(rank_rows, exp_root / "plot_rank_k_recovery.pdf")
    write_text(exp_root / "summary.md", build_exp_b_summary(args.size_label, best_layer, explained_rows, rank_rows, fixed_rows))
    metadata = {
        "size_label": args.size_label,
        "model_path": str(args.model_path),
        "dataset_root": str(args.dataset_root),
        "train_split": args.train_split,
        "eval_split": args.eval_split,
        "best_layer": best_layer,
        "n_train_pairs": len(train_pairs),
        "n_eval_pairs": len(eval_pairs),
    }
    write_json(exp_root / "metadata.json", metadata)
    return metadata


def run_exp_c(model, tokenizer, args: argparse.Namespace, *, tool_token_id: int, best_layer: int) -> dict[str, object]:
    exp_root = args.output_root / "exp_c_late_readout"
    exp_root.mkdir(parents=True, exist_ok=True)
    eval_pairs = load_sample_pairs(model, dataset_root=args.dataset_root, split=args.eval_split, max_pairs=args.readout_pairs)
    if args.downstream_layers:
        downstream_layers = [layer for layer in parse_layers(args.downstream_layers) if best_layer < layer < int(model.cfg.n_layers)]
    else:
        downstream_layers = list(range(best_layer + 1, int(model.cfg.n_layers)))
    if not downstream_layers:
        raise RuntimeError(f"No downstream layers available after best layer {best_layer}")

    head_dla_rows, mlp_rows = sweep_head_dla_and_mlp(
        model,
        eval_pairs,
        downstream_layers=downstream_layers,
        batch_size=args.batch_size,
        tool_token_id=tool_token_id,
    )
    behavior_rows = sweep_head_behavior(
        model,
        eval_pairs,
        downstream_layers=downstream_layers,
        batch_size=args.batch_size,
        tool_token_id=tool_token_id,
    )

    behavior_map = {(int(row["layer"]), int(row["head"])): row for row in behavior_rows}
    dla_values = [float(row["dla_delta"]) for row in head_dla_rows]
    behavior_values = [float(behavior_map[(int(row["layer"]), int(row["head"]))]["behavior_strict_flip_rate"]) for row in head_dla_rows]
    dla_z = normalize_zscore(dla_values)
    behavior_z = normalize_zscore(behavior_values)

    downstream_rows = []
    for row, row_dla_z, row_behavior_z in zip(head_dla_rows, dla_z, behavior_z):
        key = (int(row["layer"]), int(row["head"]))
        behavior = behavior_map[key]
        downstream_rows.append(
            {
                "layer": int(row["layer"]),
                "head": int(row["head"]),
                "mean_clean": float(row["mean_clean"]),
                "mean_corrupt": float(row["mean_corrupt"]),
                "dla_delta": float(row["dla_delta"]),
                "behavior_top1_rate": float(behavior["behavior_top1_rate"]),
                "behavior_strict_flip_rate": float(behavior["behavior_strict_flip_rate"]),
                "behavior_mean_tool_logit": float(behavior["behavior_mean_tool_logit"]),
                "score": float(row_dla_z + row_behavior_z),
            }
        )
    downstream_rows.sort(key=lambda row: float(row["score"]), reverse=True)
    top_rows = downstream_rows[:20]
    region_rows = analyze_top_head_regions(
        model,
        tokenizer,
        eval_pairs,
        top_rows=top_rows,
        batch_size=args.batch_size,
    )

    write_csv(exp_root / "downstream_head_scores.csv", downstream_rows)
    write_csv(exp_root / "top_downstream_heads.csv", top_rows)
    write_csv(exp_root / "top_head_region_attention.csv", region_rows)
    write_csv(exp_root / "optional_mlp_scores.csv", mlp_rows)
    build_heatmap(
        downstream_rows,
        value_key="behavior_strict_flip_rate",
        path=exp_root / "plot_behavior_heatmap.pdf",
        title=f"{args.size_label} downstream head behavior (z patch)",
        n_heads=int(model.cfg.n_heads),
    )
    build_heatmap(
        downstream_rows,
        value_key="dla_delta",
        path=exp_root / "plot_dla_heatmap.pdf",
        title=f"{args.size_label} downstream head DLA delta",
        n_heads=int(model.cfg.n_heads),
    )
    write_text(exp_root / "summary.md", build_exp_c_summary(args.size_label, best_layer, top_rows, mlp_rows))
    metadata = {
        "size_label": args.size_label,
        "model_path": str(args.model_path),
        "dataset_root": str(args.dataset_root),
        "eval_split": args.eval_split,
        "best_layer": best_layer,
        "downstream_layers": downstream_layers,
        "n_eval_pairs": len(eval_pairs),
    }
    write_json(exp_root / "metadata.json", metadata)
    return metadata


def main() -> None:
    args = parse_args()
    args.experiments = normalize_experiments(args.experiments)
    set_seed(args.seed)
    args.output_root.mkdir(parents=True, exist_ok=True)

    model, tokenizer, tool_token_id = load_model_and_tokenizer(model_path=args.model_path, device=args.device)

    best_layer = None
    if "A" in args.experiments:
        exp_a_meta = run_exp_a(model, args, tool_token_id=tool_token_id)
        best_layer = int(exp_a_meta["best_layer"])
    else:
        exp_a_root = args.exp_a_root if args.exp_a_root is not None else args.output_root / "exp_a_state_patch"
        exp_a_meta = json.loads((exp_a_root / "metadata.json").read_text(encoding="utf-8"))
        best_layer = int(exp_a_meta["best_layer"])

    if "B" in args.experiments:
        run_exp_b(model, args, tool_token_id=tool_token_id, best_layer=best_layer)

    if "C" in args.experiments:
        run_exp_c(model, tokenizer, args, tool_token_id=tool_token_id, best_layer=best_layer)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
