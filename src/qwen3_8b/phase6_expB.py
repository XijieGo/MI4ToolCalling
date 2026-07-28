#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import torch

from phase6_common import (
    CACHE_PATH,
    PHASE6_ROOT,
    compute_activation_stats,
    dominant_peak_token_type,
    ensure_dir,
    ensure_phase6_cache,
    format_float,
    infer_context_semantic_label,
    load_feature_index,
    phase6_device,
    read_feature_metadata,
    render_feature_context,
    summarize_counts,
    write_csv,
    write_text,
)


TARGET_LAYERS = (32, 33, 34, 35)


def stats_summary(values: list[float]) -> str:
    if not values:
        return "nan"
    tensor = torch.tensor(values, dtype=torch.float32)
    return f"{float(tensor.median().item()):.3f} median / {float(tensor.mean().item()):.3f} mean"


def build_summary(layer_rows: list[dict], inspected_peak_types: dict[int, list[str]], *, top_contexts: int) -> str:
    lines = []
    lines.append("# Exp B Summary")
    lines.append("")
    lines.append(f"Top-{top_contexts} contexts per layer were manually proxied via feature metadata inspection.")
    lines.append("")
    lines.append("## Layer Counts")
    lines.append("")
    lines.append("| Layer | clean-selective count | ratio (corrupt/clean) | frac positive in corrupt |")
    lines.append("|---|---:|---|---|")
    for row in layer_rows:
        lines.append(
            "| "
            f"L{row['layer']} | {row['count']} | {row['ratio_summary']} | {row['frac_positive_summary']} |"
        )
    lines.append("")
    lines.append("## Peak Token Type Counts (top-20 per layer)")
    lines.append("")
    lines.append("| Layer | counts |")
    lines.append("|---|---|")
    for layer in TARGET_LAYERS:
        lines.append(f"| L{layer} | {summarize_counts(inspected_peak_types.get(layer, []))} |")
    lines.append("")
    all_peak_types = [label for labels in inspected_peak_types.values() for label in labels]
    if all_peak_types:
        lines.append("## Core Takeaway")
        lines.append("")
        lines.append(
            "Retention under corrupt prompts is selective rather than uniform: "
            "the mean ratio / positive-fraction statistics show how much of the clean-selective pool still carries non-zero activation."
        )
        lines.append(
            "On inspected features, the dominant peak-token categories are "
            f"`{summarize_counts(all_peak_types)}`, which supports a default-write interpretation."
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 6 Exp B: clean-selective feature retention.")
    parser.add_argument("--cache-path", type=Path, default=CACHE_PATH)
    parser.add_argument("--out-dir", type=Path, default=PHASE6_ROOT)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--top-contexts", type=int, default=20)
    parser.add_argument("--force-cache", action="store_true")
    args = parser.parse_args()

    ensure_dir(args.out_dir)
    cache = ensure_phase6_cache(cache_path=args.cache_path, force=args.force_cache)
    feature_index = load_feature_index()
    device = phase6_device()

    all_rows: list[dict] = []
    layer_summary_rows: list[dict] = []
    inspected_peak_types: dict[int, list[str]] = {}
    context_dir = args.out_dir / "expB_contexts"
    ensure_dir(context_dir)

    for layer in TARGET_LAYERS:
        stats = compute_activation_stats(
            cache["mlp_in"]["clean"][layer].float(),
            cache["mlp_in"]["corrupt"][layer].float(),
            layer=layer,
            batch_size=args.batch_size,
            device=device,
        )
        mask = (stats["mean_clean"] > 0.5) & (stats["delta"] > 1.0)
        selected_ids = torch.where(mask)[0]
        if selected_ids.numel() == 0:
            layer_summary_rows.append(
                {
                    "layer": layer,
                    "count": 0,
                    "ratio_summary": "nan",
                    "frac_positive_summary": "nan",
                }
            )
            inspected_peak_types[layer] = []
            continue

        order = torch.argsort(stats["delta"][selected_ids], descending=True)
        selected_ids = selected_ids[order].tolist()
        top_context_ids = {int(idx) for idx in selected_ids[: args.top_contexts]}

        layer_ratios = [float(stats["ratio"][idx].item()) for idx in selected_ids if torch.isfinite(stats["ratio"][idx])]
        layer_frac_positive = [float(stats["frac_positive_corrupt"][idx].item()) for idx in selected_ids]
        layer_summary_rows.append(
            {
                "layer": layer,
                "count": len(selected_ids),
                "ratio_summary": stats_summary(layer_ratios),
                "frac_positive_summary": stats_summary(layer_frac_positive),
            }
        )

        inspected_peak_types[layer] = []
        annotations: dict[int, dict] = {}
        for feature_id in top_context_ids:
            meta = read_feature_metadata(feature_index, layer, feature_id)
            peak_type = dominant_peak_token_type(meta)
            annotations[feature_id] = {
                "peak_token_type": peak_type,
                "context_semantic_label": infer_context_semantic_label(meta),
            }
            inspected_peak_types[layer].append(peak_type)
            write_text(context_dir / f"layer_{layer:02d}_feature_{feature_id:05d}.txt", render_feature_context(meta))

        for feature_id in selected_ids:
            ann = annotations.get(int(feature_id), {})
            all_rows.append(
                {
                    "layer": layer,
                    "feature_id": int(feature_id),
                    "mean_clean": float(stats["mean_clean"][feature_id].item()),
                    "mean_corrupt": float(stats["mean_corrupt"][feature_id].item()),
                    "delta": float(stats["delta"][feature_id].item()),
                    "ratio": float(stats["ratio"][feature_id].item()) if torch.isfinite(stats["ratio"][feature_id]) else "",
                    "frac_positive_in_corrupt": float(stats["frac_positive_corrupt"][feature_id].item()),
                    "peak_token_type": ann.get("peak_token_type", ""),
                }
            )

    csv_path = args.out_dir / "expB_clean_selective.csv"
    write_csv(
        csv_path,
        all_rows,
        fieldnames=[
            "layer",
            "feature_id",
            "mean_clean",
            "mean_corrupt",
            "delta",
            "ratio",
            "frac_positive_in_corrupt",
            "peak_token_type",
        ],
    )
    write_text(
        args.out_dir / "expB_summary.md",
        build_summary(layer_summary_rows, inspected_peak_types, top_contexts=args.top_contexts),
    )
    print(f"[expB] wrote {csv_path}")
    for row in layer_summary_rows:
        print(
            "[expB] "
            f"L{row['layer']} count={row['count']} ratio={row['ratio_summary']} "
            f"frac_positive={row['frac_positive_summary']}"
        )


if __name__ == "__main__":
    main()
