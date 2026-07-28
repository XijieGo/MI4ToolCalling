#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from phase8_common import (
    DEFAULT_PC_BUNDLE,
    EVAL_DATASET_ROOT,
    EXP_B_ROOT,
    L24_LAYER,
    MODEL_PATH,
    add_gate_marker,
    build_pair_batches,
    collect_pair_last_token_activations,
    configure_matplotlib,
    cosine_to_direction,
    ensure_dir,
    find_first_threshold_layer,
    largest_jump,
    load_gate_direction,
    load_model_and_tokenizer,
    load_samples,
    manifest_pair_count,
    projection,
    set_seed,
    write_csv,
    write_text,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 8 Exp B: trajectory onto the final L24 gate.")
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH)
    parser.add_argument("--dataset-root", type=Path, default=EVAL_DATASET_ROOT)
    parser.add_argument("--output-root", type=Path, default=EXP_B_ROOT)
    parser.add_argument("--pc-bundle", type=Path, default=DEFAULT_PC_BUNDLE)
    parser.add_argument("--max-pairs", type=int, default=manifest_pair_count(EVAL_DATASET_ROOT))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-layer", type=int, default=L24_LAYER)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def plot_trajectory(rows: list[dict[str, object]], path: Path) -> None:
    configure_matplotlib()
    layers = [int(row["layer"]) for row in rows]
    mean_proj = [float(row["mean_projection"]) for row in rows]
    std_proj = [float(row["std_projection"]) for row in rows]
    mean_cos = [float(row["mean_cosine"]) for row in rows]
    mean_norm = [float(row["mean_delta_norm"]) for row in rows]

    fig, axes = plt.subplots(2, 1, figsize=(10.5, 7.2), sharex=True)

    axes[0].plot(layers, mean_proj, color="#1b6ca8", marker="o", linewidth=2.2)
    axes[0].fill_between(
        layers,
        np.asarray(mean_proj) - np.asarray(std_proj),
        np.asarray(mean_proj) + np.asarray(std_proj),
        color="#1b6ca8",
        alpha=0.16,
        linewidth=0,
    )
    axes[0].axhline(0.0, color="black", linewidth=0.8, alpha=0.35)
    add_gate_marker(axes[0], gate_layer=L24_LAYER)
    axes[0].set_ylabel(r"mean $\langle \Delta^{(l)}, v_{gate} \rangle$")
    axes[0].set_title("Projection onto the Final L24 Gate")

    ax2 = axes[1]
    ax2b = ax2.twinx()
    line1 = ax2.plot(layers, mean_cos, color="#cc5803", marker="o", linewidth=2.0, label="mean cosine")
    line2 = ax2b.plot(layers, mean_norm, color="#2a9d8f", marker="s", linewidth=2.0, label="mean ||delta||")
    add_gate_marker(ax2, gate_layer=L24_LAYER)
    ax2.axhline(0.0, color="black", linewidth=0.8, alpha=0.35)
    ax2.set_ylabel("mean cosine")
    ax2b.set_ylabel("mean ||delta||")
    ax2.set_xlabel("layer (hook_resid_pre)")
    ax2.set_title("Alignment vs Magnitude")

    handles = line1 + line2
    labels = [line.get_label() for line in handles]
    ax2.legend(handles, labels, frameon=False, loc="upper left")
    fig.tight_layout()
    ensure_dir(path.parent)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    ensure_dir(args.output_root)
    set_seed(args.seed)

    model, tokenizer = load_model_and_tokenizer(model_path=args.model_path, device=args.device)
    gate_direction = load_gate_direction(args.pc_bundle)
    samples = load_samples(args.dataset_root, model, tokenizer, max_pairs=args.max_pairs)
    pair_batches = build_pair_batches(samples, args.batch_size)
    hook_names = [f"blocks.{layer}.hook_resid_pre" for layer in range(args.max_layer + 1)]

    clean = collect_pair_last_token_activations(
        model,
        pair_batches,
        hook_names=hook_names,
        side="clean",
        desc="Trajectory clean capture",
    )
    corrupt = collect_pair_last_token_activations(
        model,
        pair_batches,
        hook_names=hook_names,
        side="corrupt",
        desc="Trajectory corrupt capture",
    )

    per_sample_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for layer, hook_name in enumerate(hook_names):
        clean_resid = clean[hook_name]
        corrupt_resid = corrupt[hook_name]
        delta = clean_resid - corrupt_resid
        proj = projection(delta, gate_direction)
        cos = cosine_to_direction(delta, gate_direction)
        norm = delta.norm(dim=-1)
        clean_score = projection(clean_resid, gate_direction)
        corrupt_score = projection(corrupt_resid, gate_direction)

        for sample_idx, sample in enumerate(samples):
            per_sample_rows.append(
                {
                    "sample_id": sample.sample_id,
                    "layer": layer,
                    "projection": float(proj[sample_idx].item()),
                    "cosine": float(cos[sample_idx].item()),
                    "delta_norm": float(norm[sample_idx].item()),
                    "clean_gate_score": float(clean_score[sample_idx].item()),
                    "corrupt_gate_score": float(corrupt_score[sample_idx].item()),
                }
            )

        summary_rows.append(
            {
                "layer": layer,
                "mean_projection": float(proj.mean().item()),
                "std_projection": float(proj.std(unbiased=True).item()),
                "mean_cosine": float(cos.mean().item()),
                "std_cosine": float(cos.std(unbiased=True).item()),
                "mean_delta_norm": float(norm.mean().item()),
                "std_delta_norm": float(norm.std(unbiased=True).item()),
                "mean_clean_gate_score": float(clean_score.mean().item()),
                "mean_corrupt_gate_score": float(corrupt_score.mean().item()),
                "n_samples": len(samples),
            }
        )

    write_csv(args.output_root / "trajectory_per_sample.csv", per_sample_rows)
    write_csv(args.output_root / "trajectory_metrics.csv", summary_rows)
    plot_trajectory(summary_rows, args.output_root / "plot_gate_trajectory.pdf")

    projections = [float(row["mean_projection"]) for row in summary_rows]
    threshold_layer = find_first_threshold_layer(projections, threshold_ratio=0.1)
    jump = largest_jump(projections)
    final_projection = projections[-1]
    lines = [
        "# Phase 8 Exp B: Upstream Trajectory onto the Final L24 Gate",
        "",
        f"- Eval split: `{args.dataset_root}` with `{len(samples)}` equal-length clean/corrupt pairs.",
        f"- Gate direction: `PC1` from `{args.pc_bundle}`.",
        f"- Final L24 mean projection: `{final_projection:.4f}`.",
        f"- First layer above 10% of the final projection: `{threshold_layer}`." if threshold_layer is not None else "- First layer above 10% of the final projection: `n/a`.",
        (
            f"- Largest consecutive jump: `L{jump[0]} -> L{jump[0] + 1}` with `+{jump[1]:.4f}`."
            if jump is not None
            else "- Largest consecutive jump: `n/a`."
        ),
    ]
    if threshold_layer is not None and threshold_layer < L24_LAYER:
        lines.append("- Interpretation: projection starts building before L24, consistent with progressive gate formation.")
    if jump is not None and jump[0] >= max(L24_LAYER - 6, 0):
        lines.append("- Interpretation: the sharpest jump is late, suggesting the final gate is organized in the upper-mid stack.")
    write_text(args.output_root / "summary.md", "\n".join(lines))

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
