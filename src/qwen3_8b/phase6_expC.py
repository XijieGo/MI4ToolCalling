#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from phase6_common import (
    CACHE_PATH,
    PHASE6_ROOT,
    compute_activation_stats,
    dominant_peak_token_type,
    ensure_dir,
    ensure_phase6_cache,
    infer_context_semantic_label,
    load_feature_index,
    logits_text,
    phase6_device,
    read_feature_metadata,
    render_feature_context,
    summarize_counts,
    write_csv,
    write_text,
)


TARGET_LAYERS = (25, 35)


def build_summary(top_rows: list[dict], inspected_peak_types: list[str], semantic_labels: list[str]) -> str:
    lines = []
    lines.append("# Exp C Summary")
    lines.append("")
    lines.append("Top-30 corrupt-selective features per layer were context-inspected.")
    lines.append("")
    lines.append("## Top Features")
    lines.append("")
    lines.append("| Rank | Layer | Feature | mean clean | mean corrupt | corrupt-clean | peak token type | semantic label |")
    lines.append("|---|---:|---:|---:|---:|---:|---|---|")
    for rank, row in enumerate(top_rows[:10], start=1):
        lines.append(
            "| "
            f"{rank} | L{row['layer']} | {row['feature_id']} | {row['mean_clean']:.4f} | "
            f"{row['mean_corrupt']:.4f} | {row['corrupt_minus_clean']:.4f} | "
            f"{row['peak_token_type']} | {row['context_semantic_label']} |"
        )
    lines.append("")
    if top_rows:
        lines.append("## Core Takeaway")
        lines.append("")
        lines.append(
            "The context-inspected corrupt-selective features cluster around "
            f"`{summarize_counts(inspected_peak_types)}` peak tokens and "
            f"`{summarize_counts(semantic_labels)}` semantics, not a clean suppressor family."
        )
        lines.append(
            "This matches the earlier causal ablation result: corrupt-selective features still write toward tool-call."
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 6 Exp C: corrupt-selective feature semantics.")
    parser.add_argument("--cache-path", type=Path, default=CACHE_PATH)
    parser.add_argument("--out-dir", type=Path, default=PHASE6_ROOT)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--top-contexts", type=int, default=30)
    parser.add_argument("--force-cache", action="store_true")
    args = parser.parse_args()

    ensure_dir(args.out_dir)
    cache = ensure_phase6_cache(cache_path=args.cache_path, force=args.force_cache)
    feature_index = load_feature_index()
    device = phase6_device()

    all_rows: list[dict] = []
    annotated_rows: list[dict] = []
    inspected_peak_types: list[str] = []
    semantic_labels: list[str] = []
    context_dir = args.out_dir / "expC_contexts"
    ensure_dir(context_dir)

    for layer in TARGET_LAYERS:
        stats = compute_activation_stats(
            cache["mlp_in"]["clean"][layer].float(),
            cache["mlp_in"]["corrupt"][layer].float(),
            layer=layer,
            batch_size=args.batch_size,
            device=device,
        )
        corrupt_minus_clean = stats["mean_corrupt"] - stats["mean_clean"]
        mask = (stats["mean_corrupt"] > 0.5) & (corrupt_minus_clean > 1.0)
        selected_ids = torch.where(mask)[0]
        if selected_ids.numel() == 0:
            continue

        order = torch.argsort(corrupt_minus_clean[selected_ids], descending=True)
        selected_ids = selected_ids[order].tolist()
        top_context_ids = {int(idx) for idx in selected_ids[: args.top_contexts]}

        annotations: dict[int, dict] = {}
        for feature_id in top_context_ids:
            meta = read_feature_metadata(feature_index, layer, feature_id)
            peak_type = dominant_peak_token_type(meta)
            semantic_label = infer_context_semantic_label(meta)
            top_logits = logits_text(meta.get("top_logits") or [], limit=5)
            annotations[feature_id] = {
                "peak_token_type": peak_type,
                "context_semantic_label": semantic_label,
                "top_logits": top_logits,
            }
            inspected_peak_types.append(peak_type)
            semantic_labels.append(semantic_label)
            write_text(context_dir / f"layer_{layer:02d}_feature_{feature_id:05d}.txt", render_feature_context(meta))

        for feature_id in selected_ids:
            ann = annotations.get(int(feature_id), {})
            row = {
                "layer": layer,
                "feature_id": int(feature_id),
                "mean_clean": float(stats["mean_clean"][feature_id].item()),
                "mean_corrupt": float(stats["mean_corrupt"][feature_id].item()),
                "delta": float(stats["delta"][feature_id].item()),
                "corrupt_minus_clean": float(corrupt_minus_clean[feature_id].item()),
                "top_logits": ann.get("top_logits", ""),
                "peak_token_type": ann.get("peak_token_type", ""),
                "context_semantic_label": ann.get("context_semantic_label", ""),
            }
            all_rows.append(row)
            if ann:
                annotated_rows.append(row)

    csv_path = args.out_dir / "expC_corrupt_selective.csv"
    write_csv(
        csv_path,
        all_rows,
        fieldnames=[
            "layer",
            "feature_id",
            "mean_clean",
            "mean_corrupt",
            "delta",
            "corrupt_minus_clean",
            "top_logits",
            "peak_token_type",
            "context_semantic_label",
        ],
    )
    annotated_rows.sort(key=lambda row: float(row["corrupt_minus_clean"]), reverse=True)
    write_text(args.out_dir / "expC_summary.md", build_summary(annotated_rows, inspected_peak_types, semantic_labels))
    print(f"[expC] wrote {csv_path}")
    print(f"[expC] inspected peak types: {summarize_counts(inspected_peak_types)}")
    print(f"[expC] semantic labels: {summarize_counts(semantic_labels)}")


if __name__ == "__main__":
    main()
