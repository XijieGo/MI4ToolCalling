#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import torch
from tqdm.auto import tqdm

from phase7_l24_directionality_common import normalized_random_direction, tool_stats_with_margin
from phase8_common import configure_matplotlib, ensure_dir, percent
from task_attention_path_analysis import (
    MODEL_PATH,
    PairBatch,
    build_pair_batches,
    clear_cuda,
    load_samples,
    locate_user_span,
    set_seed,
    write_csv,
    write_json,
    write_text,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DISCOVERY_ROOT = PROJECT_ROOT / "datasets" / "train"
EVAL_ROOT = PROJECT_ROOT / "datasets" / "test"
OUTPUT_ROOT = PROJECT_ROOT / "results" / "8b_main" / "phase11_arditi_global_direction"
LEGACY_SRC = Path(__file__).resolve().parents[1]


if str(LEGACY_SRC) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(LEGACY_SRC))

from toolcall_circuit.single_sample import load_hooked_qwen3  # noqa: E402


TOOL_CALL_STR = "<tool_call>"


@dataclass(frozen=True)
class SideMetrics:
    tool_logit: torch.Tensor
    top1: torch.Tensor
    margin: torch.Tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Arditi-style generalized direction search for tool-call decisions in Qwen3-8B."
    )
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH)
    parser.add_argument("--discovery-root", type=Path, default=DISCOVERY_ROOT)
    parser.add_argument("--eval-root", type=Path, default=EVAL_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--discovery-total-pairs", type=int, default=160)
    parser.add_argument("--discovery-train-pairs", type=int, default=128)
    parser.add_argument("--validation-pairs", type=int, default=32)
    parser.add_argument("--heldout-pairs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--prune-last-layer-frac", type=float, default=0.20)
    parser.add_argument("--shortlist-size", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def get_tool_token_id(tokenizer) -> int:
    token_ids = tokenizer.encode(TOOL_CALL_STR, add_special_tokens=False)
    if len(token_ids) != 1:
        raise ValueError(f"{TOOL_CALL_STR!r} maps to unexpected token ids: {token_ids}")
    return int(token_ids[0])


def assistant_prefix_positions(sample_text: str, tokenizer) -> tuple[list[int], list[int], list[str], str]:
    user_start, user_end = locate_user_span(sample_text)
    suffix = sample_text[user_end:]
    token_ids = tokenizer.encode(suffix, add_special_tokens=False)
    if not token_ids:
        raise ValueError("Could not determine assistant-prefix tokens from sample.")
    rel_positions = list(range(-len(token_ids), 0))
    token_text = [tokenizer.decode([token_id]) for token_id in token_ids]
    return rel_positions, token_ids, token_text, suffix


def make_mean_capture(
    accumulator_cpu: torch.Tensor,
    *,
    layer_idx: int,
    rel_positions: Sequence[int],
    n_total: int,
):
    def hook_fn(value: torch.Tensor, hook):  # noqa: ANN001
        captured = value[:, rel_positions, :].detach().cpu().to(torch.float64)
        accumulator_cpu[:, layer_idx, :] += captured.sum(dim=0) / float(n_total)
        return value

    return hook_fn


def collect_mean_resid_pre(
    model,
    pair_batches: Sequence[PairBatch],
    *,
    side: str,
    rel_positions: Sequence[int],
) -> torch.Tensor:
    if side not in {"clean", "corrupt"}:
        raise ValueError(f"Unsupported side={side!r}")

    n_positions = len(rel_positions)
    n_layers = int(model.cfg.n_layers)
    d_model = int(model.cfg.d_model)
    n_total = sum(len(batch.indices) for batch in pair_batches)
    accumulator_cpu = torch.zeros((n_positions, n_layers, d_model), dtype=torch.float64)

    hooks = [
        (
            f"blocks.{layer}.hook_resid_pre",
            make_mean_capture(accumulator_cpu, layer_idx=layer, rel_positions=rel_positions, n_total=n_total),
        )
        for layer in range(n_layers)
    ]

    progress = tqdm(pair_batches, desc=f"Mean resid_pre {side}", dynamic_ncols=True)
    for batch in progress:
        tokens_cpu = batch.clean_tokens_cpu if side == "clean" else batch.corrupt_tokens_cpu
        with torch.no_grad():
            _ = model.run_with_hooks(tokens_cpu.to(model.W_U.device), fwd_hooks=hooks)
        clear_cuda()
    return accumulator_cpu.float()


def collect_resid_pre_examples(
    model,
    pair_batches: Sequence[PairBatch],
    *,
    side: str,
    rel_positions: Sequence[int],
    layer_limit: int,
) -> torch.Tensor:
    if side not in {"clean", "corrupt"}:
        raise ValueError(f"Unsupported side={side!r}")

    n_total = sum(len(batch.indices) for batch in pair_batches)
    d_model = int(model.cfg.d_model)
    storage_cpu = torch.empty((n_total, len(rel_positions), layer_limit, d_model), dtype=torch.bfloat16)

    progress = tqdm(pair_batches, desc=f"Capture resid_pre {side}", dynamic_ncols=True)
    for batch in progress:
        capture: dict[int, torch.Tensor] = {}

        def make_capture(layer_idx: int):
            def hook_fn(value: torch.Tensor, hook):  # noqa: ANN001
                capture[layer_idx] = value[:, rel_positions, :].detach().cpu().to(torch.bfloat16)
                return value

            return hook_fn

        hooks = [(f"blocks.{layer}.hook_resid_pre", make_capture(layer)) for layer in range(layer_limit)]
        tokens_cpu = batch.clean_tokens_cpu if side == "clean" else batch.corrupt_tokens_cpu
        with torch.no_grad():
            _ = model.run_with_hooks(tokens_cpu.to(model.W_U.device), fwd_hooks=hooks)
        for layer in range(layer_limit):
            storage_cpu[batch.indices, :, layer, :] = capture[layer]
        clear_cuda()
    return storage_cpu


def evaluate_plain(
    model,
    pair_batches: Sequence[PairBatch],
    *,
    side: str,
    tool_token_id: int,
) -> SideMetrics:
    if side not in {"clean", "corrupt"}:
        raise ValueError(f"Unsupported side={side!r}")

    n_total = sum(len(batch.indices) for batch in pair_batches)
    tool_logit = torch.empty(n_total, dtype=torch.float32)
    top1 = torch.empty(n_total, dtype=torch.long)
    margin = torch.empty(n_total, dtype=torch.float32)

    progress = tqdm(pair_batches, desc=f"Baseline {side}", dynamic_ncols=True)
    for batch in progress:
        tokens_cpu = batch.clean_tokens_cpu if side == "clean" else batch.corrupt_tokens_cpu
        with torch.no_grad():
            logits = model(tokens_cpu.to(model.W_U.device))
        batch_tool_logit, batch_top1, batch_margin = tool_stats_with_margin(logits, tool_token_id)
        tool_logit[batch.indices] = batch_tool_logit
        top1[batch.indices] = batch_top1
        margin[batch.indices] = batch_margin
        clear_cuda()

    return SideMetrics(tool_logit=tool_logit, top1=top1, margin=margin)


def make_direction_ablation_hook(direction_unit_cpu: torch.Tensor, *, rel_positions: Sequence[int] | None):
    def hook_fn(value: torch.Tensor, hook):  # noqa: ANN001
        out = value.clone()
        direction = direction_unit_cpu.to(device=out.device, dtype=out.dtype)
        positions = slice(None) if rel_positions is None else list(rel_positions)
        target = out[:, positions, :]
        coeff = torch.einsum("bsd,d->bs", target, direction)
        out[:, positions, :] = target - coeff.unsqueeze(-1) * direction.unsqueeze(0).unsqueeze(0)
        return out

    return hook_fn


def make_activation_add_hook(vector_cpu: torch.Tensor, *, rel_positions: Sequence[int] | None):
    def hook_fn(value: torch.Tensor, hook):  # noqa: ANN001
        out = value.clone()
        vector = vector_cpu.to(device=out.device, dtype=out.dtype)
        positions = slice(None) if rel_positions is None else list(rel_positions)
        out[:, positions, :] = out[:, positions, :] + vector.unsqueeze(0).unsqueeze(0)
        return out

    return hook_fn


def global_ablation_hooks(n_layers: int, direction_unit_cpu: torch.Tensor) -> list[tuple[str, object]]:
    hooks: list[tuple[str, object]] = []
    for layer in range(n_layers):
        hooks.append((f"blocks.{layer}.hook_resid_pre", make_direction_ablation_hook(direction_unit_cpu, rel_positions=None)))
        hooks.append((f"blocks.{layer}.hook_resid_mid", make_direction_ablation_hook(direction_unit_cpu, rel_positions=None)))
    hooks.append((f"blocks.{n_layers - 1}.hook_resid_post", make_direction_ablation_hook(direction_unit_cpu, rel_positions=None)))
    return hooks


def source_layer_add_all_position_hooks(layer: int, vector_cpu: torch.Tensor) -> list[tuple[str, object]]:
    return [(f"blocks.{layer}.hook_resid_pre", make_activation_add_hook(vector_cpu, rel_positions=None))]


def source_layer_add_last_position_hooks(layer: int, vector_cpu: torch.Tensor) -> list[tuple[str, object]]:
    return [(f"blocks.{layer}.hook_resid_pre", make_activation_add_hook(vector_cpu, rel_positions=[-1]))]


def source_layer_last_position_ablation_hooks(layer: int, direction_unit_cpu: torch.Tensor) -> list[tuple[str, object]]:
    return [(f"blocks.{layer}.hook_resid_pre", make_direction_ablation_hook(direction_unit_cpu, rel_positions=[-1]))]


def evaluate_with_hooks(
    model,
    pair_batches: Sequence[PairBatch],
    *,
    side: str,
    tool_token_id: int,
    hooks: Sequence[tuple[str, object]],
) -> SideMetrics:
    if side not in {"clean", "corrupt"}:
        raise ValueError(f"Unsupported side={side!r}")

    n_total = sum(len(batch.indices) for batch in pair_batches)
    tool_logit = torch.empty(n_total, dtype=torch.float32)
    top1 = torch.empty(n_total, dtype=torch.long)
    margin = torch.empty(n_total, dtype=torch.float32)

    progress = tqdm(pair_batches, desc=f"Eval {side} hooks", dynamic_ncols=True)
    for batch in progress:
        tokens_cpu = batch.clean_tokens_cpu if side == "clean" else batch.corrupt_tokens_cpu
        with torch.no_grad():
            logits = model.run_with_hooks(tokens_cpu.to(model.W_U.device), fwd_hooks=list(hooks))
        batch_tool_logit, batch_top1, batch_margin = tool_stats_with_margin(logits, tool_token_id)
        tool_logit[batch.indices] = batch_tool_logit
        top1[batch.indices] = batch_top1
        margin[batch.indices] = batch_margin
        clear_cuda()

    return SideMetrics(tool_logit=tool_logit, top1=top1, margin=margin)


def strict_drop_rate(baseline: SideMetrics, current: SideMetrics, *, tool_token_id: int) -> float:
    return float(((baseline.top1 == tool_token_id) & (current.top1 != tool_token_id)).float().mean().item())


def strict_flip_rate(baseline: SideMetrics, current: SideMetrics, *, tool_token_id: int) -> float:
    return float(((baseline.top1 != tool_token_id) & (current.top1 == tool_token_id)).float().mean().item())


def top1_rate(metrics: SideMetrics, *, tool_token_id: int) -> float:
    return float((metrics.top1 == tool_token_id).float().mean().item())


def harmonic_mean(a: float, b: float) -> float:
    if a <= 0.0 or b <= 0.0:
        return 0.0
    return float((2.0 * a * b) / (a + b))


def summarize_side(
    *,
    condition: str,
    side: str,
    baseline: SideMetrics,
    current: SideMetrics,
    tool_token_id: int,
) -> dict[str, object]:
    row: dict[str, object] = {
        "condition": condition,
        "side": side,
        "n_samples": int(current.top1.shape[0]),
        "tool_call_top1_rate": top1_rate(current, tool_token_id=tool_token_id),
        "mean_tool_logit": float(current.tool_logit.mean().item()),
        "delta_tool_logit_vs_baseline": float((current.tool_logit - baseline.tool_logit).mean().item()),
        "mean_margin": float(current.margin.mean().item()),
        "delta_margin_vs_baseline": float((current.margin - baseline.margin).mean().item()),
    }
    if side == "clean":
        row["strict_drop_rate"] = strict_drop_rate(baseline, current, tool_token_id=tool_token_id)
        row["strict_flip_rate"] = ""
    else:
        row["strict_flip_rate"] = strict_flip_rate(baseline, current, tool_token_id=tool_token_id)
        row["strict_drop_rate"] = ""
    return row


def plot_validation_heatmap(
    rows: list[dict[str, object]],
    *,
    rel_positions: Sequence[int],
    layer_limit: int,
    path: Path,
    score_key: str,
    title: str,
) -> None:
    configure_matplotlib()
    matrix = torch.full((len(rel_positions), layer_limit), float("nan"), dtype=torch.float32)
    for row in rows:
        matrix[int(row["position_index"]), int(row["layer"])] = float(row[score_key])

    fig, ax = plt.subplots(figsize=(12.0, 3.8))
    im = ax.imshow(matrix.numpy(), aspect="auto", origin="lower", interpolation="nearest")
    ax.set_xlabel("layer")
    ax.set_ylabel("relative position")
    ax.set_yticks(range(len(rel_positions)))
    ax.set_yticklabels([str(pos) for pos in rel_positions])
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    fig.tight_layout()
    ensure_dir(path.parent)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_heldout_bars(rows: list[dict[str, object]], path: Path) -> None:
    configure_matplotlib()
    labels = [str(row["condition"]) for row in rows]
    top1 = [float(row["tool_call_top1_rate"]) for row in rows]
    effects = [
        float(row["strict_drop_rate"]) if row["side"] == "clean" else float(row["strict_flip_rate"])
        for row in rows
    ]

    fig, axes = plt.subplots(1, 2, figsize=(14.0, 4.5))
    axes[0].bar(range(len(rows)), top1)
    axes[0].set_title("Held-out Top-1 Rate")
    axes[0].set_ylim(0.0, 1.02)
    axes[0].set_xticks(range(len(rows)))
    axes[0].set_xticklabels(labels, rotation=35, ha="right")

    axes[1].bar(range(len(rows)), effects)
    axes[1].set_title("Held-out Strict Effect Rate")
    axes[1].set_ylim(0.0, 1.02)
    axes[1].set_xticks(range(len(rows)))
    axes[1].set_xticklabels(labels, rotation=35, ha="right")

    for ax in axes:
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    ensure_dir(path.parent)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    ensure_dir(args.output_root)
    set_seed(args.seed)

    if args.discovery_train_pairs + args.validation_pairs > args.discovery_total_pairs:
        raise ValueError("discovery_train_pairs + validation_pairs must be <= discovery_total_pairs")

    model, tokenizer = load_hooked_qwen3(str(args.model_path), device=args.device, dtype=torch.bfloat16)
    tool_token_id = get_tool_token_id(tokenizer)
    n_layers = int(model.cfg.n_layers)

    discovery_samples = load_samples(args.discovery_root, model, tokenizer, max_pairs=args.discovery_total_pairs)
    train_samples = discovery_samples[: args.discovery_train_pairs]
    val_samples = discovery_samples[args.discovery_train_pairs : args.discovery_train_pairs + args.validation_pairs]
    heldout_samples = load_samples(args.eval_root, model, tokenizer, max_pairs=args.heldout_pairs)

    rel_positions, suffix_token_ids, suffix_token_text, suffix_text = assistant_prefix_positions(train_samples[0].clean_text, tokenizer)
    layer_limit = max(1, int(math.floor(n_layers * (1.0 - float(args.prune_last_layer_frac)))))

    split_info = {
        "model_path": str(args.model_path),
        "discovery_root": str(args.discovery_root),
        "eval_root": str(args.eval_root),
        "discovery_total_pairs": int(args.discovery_total_pairs),
        "discovery_train_pairs": int(args.discovery_train_pairs),
        "validation_pairs": int(args.validation_pairs),
        "heldout_pairs": int(args.heldout_pairs),
        "batch_size": int(args.batch_size),
        "shortlist_size": int(args.shortlist_size),
        "n_layers": n_layers,
        "layer_limit": layer_limit,
        "candidate_relative_positions": list(rel_positions),
        "suffix_text": suffix_text,
        "suffix_token_ids": list(suffix_token_ids),
        "suffix_token_text": list(suffix_token_text),
    }
    write_json(args.output_root / "split_info.json", split_info)

    train_batches = build_pair_batches(train_samples, args.batch_size)
    val_batches = build_pair_batches(val_samples, args.batch_size)
    heldout_batches = build_pair_batches(heldout_samples, args.batch_size)

    mean_clean = collect_mean_resid_pre(model, train_batches, side="clean", rel_positions=rel_positions)
    mean_corrupt = collect_mean_resid_pre(model, train_batches, side="corrupt", rel_positions=rel_positions)
    torch.save(
        {
            "mean_clean": mean_clean,
            "mean_corrupt": mean_corrupt,
            "relative_positions": list(rel_positions),
            "suffix_text": suffix_text,
            "suffix_token_ids": list(suffix_token_ids),
            "suffix_token_text": list(suffix_token_text),
        },
        args.output_root / "candidate_means.pt",
    )

    baseline_val_clean = evaluate_plain(model, val_batches, side="clean", tool_token_id=tool_token_id)
    baseline_val_corrupt = evaluate_plain(model, val_batches, side="corrupt", tool_token_id=tool_token_id)

    val_clean_examples = collect_resid_pre_examples(
        model,
        val_batches,
        side="clean",
        rel_positions=rel_positions,
        layer_limit=layer_limit,
    )
    val_corrupt_examples = collect_resid_pre_examples(
        model,
        val_batches,
        side="corrupt",
        rel_positions=rel_positions,
        layer_limit=layer_limit,
    )

    preselection_rows: list[dict[str, object]] = []
    for pos_idx in range(len(rel_positions)):
        for layer in range(layer_limit):
            vector = (mean_clean[pos_idx, layer] - mean_corrupt[pos_idx, layer]).float()
            vector_norm = float(vector.norm().item())
            if vector_norm <= 1e-12:
                continue
            direction_unit = vector / vector.norm().clamp_min(1e-12)
            clean_scores = torch.einsum("nd,d->n", val_clean_examples[:, pos_idx, layer, :].float(), direction_unit)
            corrupt_scores = torch.einsum("nd,d->n", val_corrupt_examples[:, pos_idx, layer, :].float(), direction_unit)
            expression_gap = float((clean_scores.mean() - corrupt_scores.mean()).item())
            pooled = torch.cat([clean_scores, corrupt_scores], dim=0)
            pooled_std = float(pooled.std(unbiased=False).item())
            expression_effect_size = expression_gap / max(pooled_std, 1e-6)
            preselection_rows.append(
                {
                    "layer": int(layer),
                    "position_index": int(pos_idx),
                    "relative_position": int(rel_positions[pos_idx]),
                    "position_token_id": int(suffix_token_ids[pos_idx]),
                    "position_token_text": suffix_token_text[pos_idx],
                    "vector_norm": vector_norm,
                    "expression_gap": expression_gap,
                    "expression_effect_size": expression_effect_size,
                }
            )

    preselection_rows.sort(
        key=lambda row: (
            float(row["expression_effect_size"]),
            float(row["expression_gap"]),
            float(row["vector_norm"]),
        ),
        reverse=True,
    )
    write_csv(args.output_root / "preselection_scores.csv", preselection_rows)
    plot_validation_heatmap(
        preselection_rows,
        rel_positions=rel_positions,
        layer_limit=layer_limit,
        path=args.output_root / "plot_preselection_heatmap.pdf",
        score_key="expression_effect_size",
        title="Validation Expression Effect Size",
    )

    shortlist = preselection_rows[: min(len(preselection_rows), int(args.shortlist_size))]
    write_csv(args.output_root / "preselection_topk.csv", shortlist)

    candidate_rows: list[dict[str, object]] = []
    progress = tqdm(shortlist, desc="Validate shortlist", dynamic_ncols=True)
    for short_row in progress:
        pos_idx = int(short_row["position_index"])
        layer = int(short_row["layer"])
        vector = (mean_clean[pos_idx, layer] - mean_corrupt[pos_idx, layer]).float()
        vector_norm = float(vector.norm().item())
        direction_unit = vector / vector.norm().clamp_min(1e-12)

        clean_ablate = evaluate_with_hooks(
            model,
            val_batches,
            side="clean",
            tool_token_id=tool_token_id,
            hooks=global_ablation_hooks(n_layers, direction_unit),
        )
        corrupt_add = evaluate_with_hooks(
            model,
            val_batches,
            side="corrupt",
            tool_token_id=tool_token_id,
            hooks=source_layer_add_all_position_hooks(layer, vector),
        )

        clean_drop = strict_drop_rate(baseline_val_clean, clean_ablate, tool_token_id=tool_token_id)
        corrupt_flip = strict_flip_rate(baseline_val_corrupt, corrupt_add, tool_token_id=tool_token_id)
        selection_score = harmonic_mean(clean_drop, corrupt_flip)
        candidate_rows.append(
            {
                "layer": int(layer),
                "position_index": int(pos_idx),
                "relative_position": int(rel_positions[pos_idx]),
                "position_token_id": int(suffix_token_ids[pos_idx]),
                "position_token_text": suffix_token_text[pos_idx],
                "vector_norm": vector_norm,
                "expression_gap": float(short_row["expression_gap"]),
                "expression_effect_size": float(short_row["expression_effect_size"]),
                "selection_score": selection_score,
                "validation_clean_drop_rate": clean_drop,
                "validation_clean_top1_rate": top1_rate(clean_ablate, tool_token_id=tool_token_id),
                "validation_clean_mean_tool_logit": float(clean_ablate.tool_logit.mean().item()),
                "validation_corrupt_flip_rate": corrupt_flip,
                "validation_corrupt_top1_rate": top1_rate(corrupt_add, tool_token_id=tool_token_id),
                "validation_corrupt_mean_tool_logit": float(corrupt_add.tool_logit.mean().item()),
            }
        )

    candidate_rows.sort(
        key=lambda row: (
            float(row["selection_score"]),
            float(row["validation_clean_drop_rate"]) + float(row["validation_corrupt_flip_rate"]),
            float(row["validation_corrupt_top1_rate"]),
        ),
        reverse=True,
    )
    write_csv(args.output_root / "candidate_scores.csv", candidate_rows)
    top_candidates = candidate_rows[: min(10, len(candidate_rows))]
    write_csv(args.output_root / "top_candidates.csv", top_candidates)

    if not candidate_rows:
        raise RuntimeError("No valid candidates were produced.")
    best = candidate_rows[0]
    best_pos_idx = int(best["position_index"])
    best_layer = int(best["layer"])
    best_vector = (mean_clean[best_pos_idx, best_layer] - mean_corrupt[best_pos_idx, best_layer]).float()
    best_direction_unit = best_vector / best_vector.norm().clamp_min(1e-12)
    random_direction = normalized_random_direction(best_vector.shape[0], seed=args.seed + 17)

    baseline_test_clean = evaluate_plain(model, heldout_batches, side="clean", tool_token_id=tool_token_id)
    baseline_test_corrupt = evaluate_plain(model, heldout_batches, side="corrupt", tool_token_id=tool_token_id)

    heldout_rows: list[dict[str, object]] = []
    heldout_rows.append(
        summarize_side(
            condition="baseline_clean",
            side="clean",
            baseline=baseline_test_clean,
            current=baseline_test_clean,
            tool_token_id=tool_token_id,
        )
    )
    heldout_rows.append(
        summarize_side(
            condition="baseline_corrupt",
            side="corrupt",
            baseline=baseline_test_corrupt,
            current=baseline_test_corrupt,
            tool_token_id=tool_token_id,
        )
    )

    clean_global_ablate = evaluate_with_hooks(
        model,
        heldout_batches,
        side="clean",
        tool_token_id=tool_token_id,
        hooks=global_ablation_hooks(n_layers, best_direction_unit),
    )
    heldout_rows.append(
        summarize_side(
            condition="clean_global_ablation_best",
            side="clean",
            baseline=baseline_test_clean,
            current=clean_global_ablate,
            tool_token_id=tool_token_id,
        )
    )

    clean_local_last_ablate = evaluate_with_hooks(
        model,
        heldout_batches,
        side="clean",
        tool_token_id=tool_token_id,
        hooks=source_layer_last_position_ablation_hooks(best_layer, best_direction_unit),
    )
    heldout_rows.append(
        summarize_side(
            condition="clean_local_last_ablation_best",
            side="clean",
            baseline=baseline_test_clean,
            current=clean_local_last_ablate,
            tool_token_id=tool_token_id,
        )
    )

    clean_global_ablate_random = evaluate_with_hooks(
        model,
        heldout_batches,
        side="clean",
        tool_token_id=tool_token_id,
        hooks=global_ablation_hooks(n_layers, random_direction),
    )
    heldout_rows.append(
        summarize_side(
            condition="clean_global_ablation_random",
            side="clean",
            baseline=baseline_test_clean,
            current=clean_global_ablate_random,
            tool_token_id=tool_token_id,
        )
    )

    corrupt_source_add_all = evaluate_with_hooks(
        model,
        heldout_batches,
        side="corrupt",
        tool_token_id=tool_token_id,
        hooks=source_layer_add_all_position_hooks(best_layer, best_vector),
    )
    heldout_rows.append(
        summarize_side(
            condition="corrupt_source_add_all_positions_best",
            side="corrupt",
            baseline=baseline_test_corrupt,
            current=corrupt_source_add_all,
            tool_token_id=tool_token_id,
        )
    )

    corrupt_source_add_last = evaluate_with_hooks(
        model,
        heldout_batches,
        side="corrupt",
        tool_token_id=tool_token_id,
        hooks=source_layer_add_last_position_hooks(best_layer, best_vector),
    )
    heldout_rows.append(
        summarize_side(
            condition="corrupt_source_add_last_position_best",
            side="corrupt",
            baseline=baseline_test_corrupt,
            current=corrupt_source_add_last,
            tool_token_id=tool_token_id,
        )
    )

    corrupt_source_add_random = evaluate_with_hooks(
        model,
        heldout_batches,
        side="corrupt",
        tool_token_id=tool_token_id,
        hooks=source_layer_add_all_position_hooks(best_layer, random_direction * best_vector.norm()),
    )
    heldout_rows.append(
        summarize_side(
            condition="corrupt_source_add_all_positions_random",
            side="corrupt",
            baseline=baseline_test_corrupt,
            current=corrupt_source_add_random,
            tool_token_id=tool_token_id,
        )
    )

    for row in heldout_rows:
        if "best" in str(row["condition"]):
            row["selected_layer"] = best_layer
            row["selected_relative_position"] = int(best["relative_position"])
            row["selected_position_token_text"] = best["position_token_text"]
            row["selected_vector_norm"] = float(best_vector.norm().item())
        else:
            row["selected_layer"] = ""
            row["selected_relative_position"] = ""
            row["selected_position_token_text"] = ""
            row["selected_vector_norm"] = ""

    write_csv(args.output_root / "heldout_results.csv", heldout_rows)
    plot_heldout_bars(heldout_rows, args.output_root / "plot_heldout_bars.pdf")

    best_clean_global = next(row for row in heldout_rows if row["condition"] == "clean_global_ablation_best")
    best_clean_local = next(row for row in heldout_rows if row["condition"] == "clean_local_last_ablation_best")
    best_corrupt_all = next(row for row in heldout_rows if row["condition"] == "corrupt_source_add_all_positions_best")
    best_corrupt_last = next(row for row in heldout_rows if row["condition"] == "corrupt_source_add_last_position_best")

    lines = [
        "# Phase 11 Arditi-Style Global Direction",
        "",
        "## Setup",
        "",
        f"- Discovery train / validation / held-out: `{args.discovery_train_pairs}` / `{args.validation_pairs}` / `{args.heldout_pairs}` pairs.",
        f"- Candidate source positions: `{rel_positions}`.",
        f"- Candidate source layers searched: `0..{layer_limit - 1}` (`prune_last_layer_frac={float(args.prune_last_layer_frac):.2f}`).",
        f"- Coarse-to-fine shortlist size: `{int(args.shortlist_size)}`.",
        f"- Best candidate: layer `{best_layer}`, relative position `{int(best['relative_position'])}`, token `{best['position_token_text']}`.",
        f"- Best preselection effect size: `{float(best['expression_effect_size']):.4f}`.",
        f"- Best validation score: `{float(best['selection_score']):.4f}` with clean drop `{percent(float(best['validation_clean_drop_rate']))}` and corrupt flip `{percent(float(best['validation_corrupt_flip_rate']))}`.",
        "",
        "## Held-out Main Results",
        "",
        f"- Global clean ablation: top-1 `{percent(float(best_clean_global['tool_call_top1_rate']))}`, strict drop `{percent(float(best_clean_global['strict_drop_rate']))}`, mean logit delta `{float(best_clean_global['delta_tool_logit_vs_baseline']):+.4f}`.",
        f"- Local clean ablation at selected layer/last position: top-1 `{percent(float(best_clean_local['tool_call_top1_rate']))}`, strict drop `{percent(float(best_clean_local['strict_drop_rate']))}`.",
        f"- Source-layer all-position addition on corrupt: top-1 `{percent(float(best_corrupt_all['tool_call_top1_rate']))}`, strict flip `{percent(float(best_corrupt_all['strict_flip_rate']))}`, mean logit delta `{float(best_corrupt_all['delta_tool_logit_vs_baseline']):+.4f}`.",
        f"- Source-layer last-position addition on corrupt: top-1 `{percent(float(best_corrupt_last['tool_call_top1_rate']))}`, strict flip `{percent(float(best_corrupt_last['strict_flip_rate']))}`.",
        "",
        "## Interpretation",
        "",
        "- This script is the closest analogue in this repo to the refusal paper's methodology: select a single difference-in-means vector from layer-position candidates, then test global directional ablation plus fixed-vector activation addition.",
        "- To keep the search tractable on this long-prompt dataset, candidate search uses a coarse-to-fine procedure: validation expression-gap shortlist first, then full causal validation on the shortlist.",
        "- The key question is whether the held-out global ablation is strong on the clean side and whether fixed all-position addition is strong on the corrupt side.",
        "- Compare `clean_global_ablation_best` vs `clean_local_last_ablation_best`, and `corrupt_source_add_all_positions_best` vs `corrupt_source_add_last_position_best`, to see whether the more generalized intervention actually buys anything.",
    ]
    write_text(args.output_root / "summary.md", "\n".join(lines))

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
