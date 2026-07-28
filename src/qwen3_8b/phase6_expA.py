#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import torch

from phase6_common import (
    CACHE_PATH,
    PHASE6_ROOT,
    clear_cuda,
    compute_ablation_diffs,
    dominant_peak_token_type,
    ensure_dir,
    ensure_phase6_cache,
    format_float,
    load_feature_index,
    logits_text,
    phase6_device,
    peak_token_examples,
    read_feature_metadata,
    render_feature_context,
    summarize_counts,
    write_csv,
    write_text,
)


def build_summary(rows: list[dict], *, sample_count: int, top_summary_k: int) -> str:
    top_rows = rows[:top_summary_k]
    peak_types = [str(row["peak_token_type"]) for row in top_rows]
    top_logits = [str(row["top5_logits"]) for row in top_rows[:10]]
    lines = []
    lines.append("# Exp A Summary")
    lines.append("")
    lines.append(f"- Samples per condition: `{sample_count}` clean + `{sample_count}` corrupt")
    lines.append("- Layer / head: `L33H29` -> `L33 MLP transcoder input`")
    lines.append(f"- Top-{top_summary_k} peak token type mix: `{summarize_counts(peak_types)}`")
    lines.append("")
    lines.append("## Top Features")
    lines.append("")
    lines.append("| Rank | Feature | clean diff | corrupt diff | delta | peak token type | top logits |")
    lines.append("|---|---:|---:|---:|---:|---|---|")
    for rank, row in enumerate(top_rows, start=1):
        lines.append(
            "| "
            f"{rank} | {row['feature_id']} | {format_float(row['clean_mean_diff'])} | "
            f"{format_float(row['corrupt_mean_diff'])} | {format_float(row['delta'])} | "
            f"{row['peak_token_type']} | {row['top5_logits']} |"
        )
    lines.append("")
    if top_rows:
        lines.append("## Core Takeaway")
        lines.append("")
        lines.append(
            "L33H29's clean-favoring feature write is concentrated in "
            f"`{summarize_counts(peak_types)}` contexts rather than a single verb-only feature."
        )
        lines.append(
            "The top logits most often look like: "
            f"`{' ; '.join(top_logits[:5])}`."
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 6 Exp A: L33H29 feature-level write analysis.")
    parser.add_argument("--cache-path", type=Path, default=CACHE_PATH)
    parser.add_argument("--out-dir", type=Path, default=PHASE6_ROOT)
    parser.add_argument("--sample-count", type=int, default=200)
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--summary-top-k", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--force-cache", action="store_true")
    args = parser.parse_args()

    ensure_dir(args.out_dir)
    cache = ensure_phase6_cache(cache_path=args.cache_path, force=args.force_cache)
    feature_index = load_feature_index()

    sample_count = min(args.sample_count, len(cache["sample_ids"]))
    clean_inputs = cache["mlp_in"]["clean"][33][:sample_count].float()
    corrupt_inputs = cache["mlp_in"]["corrupt"][33][:sample_count].float()
    clean_removed = cache["l33h29_out"]["clean"][:sample_count].float()
    corrupt_removed = cache["l33h29_out"]["corrupt"][:sample_count].float()

    device = phase6_device()
    clean_mean_diff = compute_ablation_diffs(
        clean_inputs,
        clean_removed,
        layer=33,
        batch_size=args.batch_size,
        device=device,
    )
    corrupt_mean_diff = compute_ablation_diffs(
        corrupt_inputs,
        corrupt_removed,
        layer=33,
        batch_size=args.batch_size,
        device=device,
    )
    delta = clean_mean_diff - corrupt_mean_diff
    top_ids = torch.argsort(delta, descending=True)[: args.top_k].tolist()

    rows: list[dict] = []
    context_dir = args.out_dir / "expA_contexts"
    ensure_dir(context_dir)
    peak_type_counter: Counter[str] = Counter()
    for feature_id in top_ids:
        meta = read_feature_metadata(feature_index, 33, int(feature_id))
        peak_token_type = dominant_peak_token_type(meta)
        peak_type_counter[peak_token_type] += 1
        row = {
            "feature_id": int(feature_id),
            "clean_mean_diff": float(clean_mean_diff[feature_id].item()),
            "corrupt_mean_diff": float(corrupt_mean_diff[feature_id].item()),
            "delta": float(delta[feature_id].item()),
            "peak_token_type": peak_token_type,
            "peak_token_examples": peak_token_examples(meta, limit=5),
            "top5_logits": logits_text(meta.get("top_logits") or [], limit=5),
        }
        rows.append(row)
        write_text(context_dir / f"feature_{int(feature_id):05d}.txt", render_feature_context(meta))

    csv_path = args.out_dir / "expA_l33h29_features.csv"
    write_csv(
        csv_path,
        rows,
        fieldnames=[
            "feature_id",
            "clean_mean_diff",
            "corrupt_mean_diff",
            "delta",
            "peak_token_type",
            "peak_token_examples",
            "top5_logits",
        ],
    )
    summary_text = build_summary(rows, sample_count=sample_count, top_summary_k=min(args.summary_top_k, len(rows)))
    write_text(args.out_dir / "expA_summary.md", summary_text)

    clear_cuda()
    print(f"[expA] wrote {csv_path}")
    print(f"[expA] top-{len(rows)} peak token types: {dict(peak_type_counter)}")


if __name__ == "__main__":
    main()
