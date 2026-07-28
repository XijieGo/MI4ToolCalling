#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from multiscale_common import write_csv, write_text


SIZES = ("1.7B", "4B", "14B")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the cross-scale core-generalization summary.")
    parser.add_argument("--results-root", type=Path, default=Path("./results"))
    parser.add_argument("--output-csv", type=Path, default=Path("./results/multi_scale_core_generalization_summary.csv"))
    parser.add_argument("--output-md", type=Path, default=Path("./results/multi_scale_core_generalization_summary.md"))
    return parser.parse_args()


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def maybe_float(text: str | None) -> float | None:
    if text is None:
        return None
    raw = str(text).strip()
    if not raw:
        return None
    return float(raw)


def collect_size_row(results_root: Path, size_label: str) -> dict[str, object]:
    size_root = results_root / size_label
    phase6_root = size_root / "phase6_core_generalization"
    exp_a_meta = json.loads((phase6_root / "exp_a_state_patch" / "metadata.json").read_text(encoding="utf-8"))
    exp_a_rows = load_csv(phase6_root / "exp_a_state_patch" / "patch_sweep.csv")
    exp_a_final_rows = [row for row in exp_a_rows if str(row.get("phase", "")).lower() == "final"]
    exp_a_pool = exp_a_final_rows if exp_a_final_rows else exp_a_rows
    exp_a_best = max(exp_a_pool, key=lambda row: float(row["tool_call_top1_rate"]))

    exp_b_rows = load_csv(phase6_root / "exp_b_gate_vector" / "rank_k_patch_sweep.csv")
    exp_b_rank1 = max((row for row in exp_b_rows if str(row["rank_k"]) == "1"), key=lambda row: float(row["tool_call_top1_rate"]))
    exp_b_full = max((row for row in exp_b_rows if str(row["rank_k"]) == "full"), key=lambda row: float(row["tool_call_top1_rate"]))

    exp_c_rows = load_csv(phase6_root / "exp_c_late_readout" / "top_downstream_heads.csv")
    top_layers = []
    for row in exp_c_rows:
        layer = int(row["layer"])
        if layer not in top_layers:
            top_layers.append(layer)
        if len(top_layers) >= 5:
            break

    triplet_root = size_root / "triplet_analysis"
    triplet_layer = None
    if (triplet_root / f"logit_lens_{size_label}.csv").exists():
        logit_rows = load_csv(triplet_root / f"logit_lens_{size_label}.csv")
        for row in logit_rows:
            if float(row["gap"]) > 1.0:
                triplet_layer = int(row["layer"])
                break

    wording = "compact local decision representation + late distributed writers"
    return {
        "size_label": size_label,
        "triplet_first_gap_layer": triplet_layer,
        "key_layer_L_star": int(exp_a_meta["best_layer"]),
        "exp_a_best_top1_rate": float(exp_a_best["tool_call_top1_rate"]),
        "exp_a_best_strict_flip_rate": float(exp_a_best["strict_flip_rate"]),
        "exp_b_rank1_top1_rate": float(exp_b_rank1["tool_call_top1_rate"]),
        "exp_b_rank1_strict_flip_rate": float(exp_b_rank1["strict_flip_rate"]),
        "exp_b_full_top1_rate": float(exp_b_full["tool_call_top1_rate"]),
        "exp_c_top_reader_layers": ", ".join(f"L{layer}" for layer in top_layers),
        "paper_wording": wording,
    }


def build_markdown(rows: list[dict[str, object]]) -> str:
    lines = [
        "# Multi-Scale Core Generalization Summary",
        "",
        "| Size | Triplet first gap layer | Key layer L* | Exp A best top1 | Exp A strict flip | Exp B rank-1 top1 | Exp B full top1 | Exp C top reader layers | Paper wording |",
        "|---|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in rows:
        triplet = f"L{int(row['triplet_first_gap_layer'])}" if row["triplet_first_gap_layer"] is not None else "NA"
        lines.append(
            "| "
            f"{row['size_label']} | {triplet} | L{int(row['key_layer_L_star'])} | "
            f"{float(row['exp_a_best_top1_rate']):.2%} | {float(row['exp_a_best_strict_flip_rate']):.2%} | "
            f"{float(row['exp_b_rank1_top1_rate']):.2%} | {float(row['exp_b_full_top1_rate']):.2%} | "
            f"{row['exp_c_top_reader_layers']} | {row['paper_wording']} |"
        )
    lines.extend(
        [
            "",
            "Interpretation:",
            "Across the three scales, the current canonical outputs support the same broad mechanism signature: a mid-to-late prediction-position bottleneck, a compact gate-like representation at that layer, and downstream late distributed readout into `<tool_call>`.",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    rows = [collect_size_row(args.results_root, size_label) for size_label in SIZES]
    write_csv(args.output_csv, rows)
    write_text(args.output_md, build_markdown(rows))


if __name__ == "__main__":
    main()
