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
    EXP_C_ROOT,
    L24_LAYER,
    MODEL_PATH,
    build_pair_batches,
    collect_pair_last_token_activations,
    configure_matplotlib,
    ensure_dir,
    load_gate_direction,
    load_model_and_tokenizer,
    load_samples,
    manifest_pair_count,
    projection,
    set_seed,
    stage_abs_mass,
    write_csv,
    write_text,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 8 Exp C: layerwise gate component contributions.")
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH)
    parser.add_argument("--dataset-root", type=Path, default=EVAL_DATASET_ROOT)
    parser.add_argument("--output-root", type=Path, default=EXP_C_ROOT)
    parser.add_argument("--pc-bundle", type=Path, default=DEFAULT_PC_BUNDLE)
    parser.add_argument("--max-pairs", type=int, default=manifest_pair_count(EVAL_DATASET_ROOT))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-layer", type=int, default=L24_LAYER - 1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def plot_components(rows: list[dict[str, object]], path: Path) -> None:
    configure_matplotlib()
    layers = [int(row["layer"]) for row in rows]
    attn = [float(row["mean_attn_contrib"]) for row in rows]
    mlp = [float(row["mean_mlp_contrib"]) for row in rows]
    cum_attn = [float(row["cumulative_attn"]) for row in rows]
    cum_mlp = [float(row["cumulative_mlp"]) for row in rows]
    cum_total = [float(row["cumulative_total"]) for row in rows]

    fig, axes = plt.subplots(2, 1, figsize=(10.8, 7.5), sharex=True)
    width = 0.38
    axes[0].bar(np.asarray(layers) - width / 2, attn, width=width, color="#1b6ca8", label="attention")
    axes[0].bar(np.asarray(layers) + width / 2, mlp, width=width, color="#cc5803", label="MLP")
    axes[0].axhline(0.0, color="black", linewidth=0.8, alpha=0.35)
    axes[0].set_ylabel(r"mean signed contribution to $v_{gate}$")
    axes[0].set_title("Per-Layer Component Contributions into the L24 Gate")
    axes[0].legend(frameon=False)

    axes[1].plot(layers, cum_attn, color="#1b6ca8", marker="o", linewidth=2.0, label="cumulative attention")
    axes[1].plot(layers, cum_mlp, color="#cc5803", marker="s", linewidth=2.0, label="cumulative MLP")
    axes[1].plot(layers, cum_total, color="#2a9d8f", marker="^", linewidth=2.2, label="cumulative total")
    axes[1].axhline(0.0, color="black", linewidth=0.8, alpha=0.35)
    axes[1].set_xlabel("layer (block output written before L24)")
    axes[1].set_ylabel("cumulative mean contribution")
    axes[1].set_title("Cumulative Gate Formation")
    axes[1].legend(frameon=False, loc="upper left")

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

    pre_hooks = [f"blocks.{layer}.hook_resid_pre" for layer in range(args.max_layer + 1)]
    mid_hooks = [f"blocks.{layer}.hook_resid_mid" for layer in range(args.max_layer + 1)]
    post_hooks = [f"blocks.{layer}.hook_resid_post" for layer in range(args.max_layer + 1)]
    hook_names = pre_hooks + mid_hooks + post_hooks

    clean = collect_pair_last_token_activations(
        model,
        pair_batches,
        hook_names=hook_names,
        side="clean",
        desc="Component clean capture",
    )
    corrupt = collect_pair_last_token_activations(
        model,
        pair_batches,
        hook_names=hook_names,
        side="corrupt",
        desc="Component corrupt capture",
    )

    per_sample_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    cumulative_attn = 0.0
    cumulative_mlp = 0.0
    for layer in range(args.max_layer + 1):
        pre_name = f"blocks.{layer}.hook_resid_pre"
        mid_name = f"blocks.{layer}.hook_resid_mid"
        post_name = f"blocks.{layer}.hook_resid_post"

        attn_clean = clean[mid_name] - clean[pre_name]
        attn_corrupt = corrupt[mid_name] - corrupt[pre_name]
        mlp_clean = clean[post_name] - clean[mid_name]
        mlp_corrupt = corrupt[post_name] - corrupt[mid_name]

        delta_attn = attn_clean - attn_corrupt
        delta_mlp = mlp_clean - mlp_corrupt
        gate_attn = projection(delta_attn, gate_direction)
        gate_mlp = projection(delta_mlp, gate_direction)
        gate_total = gate_attn + gate_mlp

        for sample_idx, sample in enumerate(samples):
            per_sample_rows.append(
                {
                    "sample_id": sample.sample_id,
                    "layer": layer,
                    "attn_contrib": float(gate_attn[sample_idx].item()),
                    "mlp_contrib": float(gate_mlp[sample_idx].item()),
                    "total_contrib": float(gate_total[sample_idx].item()),
                }
            )

        mean_attn = float(gate_attn.mean().item())
        mean_mlp = float(gate_mlp.mean().item())
        cumulative_attn += mean_attn
        cumulative_mlp += mean_mlp
        summary_rows.append(
            {
                "layer": layer,
                "mean_attn_contrib": mean_attn,
                "std_attn_contrib": float(gate_attn.std(unbiased=True).item()),
                "mean_mlp_contrib": mean_mlp,
                "std_mlp_contrib": float(gate_mlp.std(unbiased=True).item()),
                "mean_total_contrib": float(gate_total.mean().item()),
                "std_total_contrib": float(gate_total.std(unbiased=True).item()),
                "cumulative_attn": cumulative_attn,
                "cumulative_mlp": cumulative_mlp,
                "cumulative_total": cumulative_attn + cumulative_mlp,
                "n_samples": len(samples),
            }
        )

    write_csv(args.output_root / "component_contributions_per_sample.csv", per_sample_rows)
    write_csv(args.output_root / "component_contributions.csv", summary_rows)
    plot_components(summary_rows, args.output_root / "plot_gate_component_contributions.pdf")

    attn_series = [float(row["mean_attn_contrib"]) for row in summary_rows]
    mlp_series = [float(row["mean_mlp_contrib"]) for row in summary_rows]
    attn_mass = sum(abs(x) for x in attn_series)
    mlp_mass = sum(abs(x) for x in mlp_series)
    early = range(0, min(8, len(summary_rows)))
    middle = range(8, min(16, len(summary_rows)))
    late = range(16, len(summary_rows))

    def dominant_label(values_a: list[float], values_b: list[float], stage: range) -> str:
        a_mass = stage_abs_mass(values_a, stage)
        b_mass = stage_abs_mass(values_b, stage)
        if np.isclose(a_mass, b_mass):
            return "mixed"
        return "attention" if a_mass > b_mass else "MLP"

    top_total = max(summary_rows, key=lambda row: abs(float(row["mean_total_contrib"])))
    lines = [
        "# Phase 8 Exp C: Per-Layer Attention / MLP Contribution into the Gate",
        "",
        f"- Eval split: `{args.dataset_root}` with `{len(samples)}` held-out pairs.",
        f"- Total absolute contribution mass: attention `{attn_mass:.4f}`, MLP `{mlp_mass:.4f}`.",
        f"- Strongest single-layer total contribution: `L{int(top_total['layer'])}` with `{float(top_total['mean_total_contrib']):+.4f}`.",
        f"- Early-stage dominance (L0-L7): `{dominant_label(attn_series, mlp_series, early)}`.",
        f"- Mid-stage dominance (L8-L15): `{dominant_label(attn_series, mlp_series, middle)}`.",
        f"- Late-stage dominance (L16-L{args.max_layer}): `{dominant_label(attn_series, mlp_series, late)}`.",
    ]
    write_text(args.output_root / "summary.md", "\n".join(lines))

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
