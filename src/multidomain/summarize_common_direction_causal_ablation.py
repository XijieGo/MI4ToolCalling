#!/usr/bin/env python3
"""Validate and render one cross-model table for shared-direction ablations."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


MODELS = ("Qwen3-1.7B", "Qwen3-4B", "Qwen3-8B", "Qwen3-14B")
CONDITION_ORDER = (
    "native_full",
    "shared_target_norm_x1p0",
    "shared_target_norm_x1p5",
    "shared_target_norm_x2p0",
    "shared_target_norm_x3p0",
    "native_projection_on_shared",
    "native_residual_after_shared",
    "random_orthogonal_target_norm",
)
METRICS = (
    "sufficiency_normalized_logit_gap",
    "necessity_normalized_logit_gap",
    "add_strict_flip_rate",
    "remove_strict_drop_rate",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize and validate common-direction causal ablations.")
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--original-run-root", type=Path, required=True)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    campaign_root = args.campaign_root.resolve()
    original_root = args.original_run_root.resolve()
    table_rows: list[dict[str, Any]] = []
    validation: dict[str, Any] = {"models": {}, "status": "pass"}
    for model in MODELS:
        root = campaign_root / model
        completion = read_json(root / "completion.json")
        long_rows = read_csv(root / "causal_ablation_long.csv")
        aggregate_rows = read_csv(root / "aggregate_metrics.csv")
        geometry = read_json(root / "shared_direction_geometry.json")
        expected = 4 * len(CONDITION_ORDER)
        if completion.get("status") != "complete" or int(completion.get("observed_cells", -1)) != expected or len(long_rows) != expected:
            raise ValueError(f"{model}: incomplete experiment")
        observed = {row["condition"] for row in aggregate_rows}
        if observed != set(CONDITION_ORDER) or len(aggregate_rows) != len(CONDITION_ORDER):
            raise ValueError(f"{model}: aggregate condition mismatch")

        original = read_csv(original_root / model / "matrix_long.csv")
        native = {row["target_domain"]: row for row in long_rows if row["condition"] == "native_full"}
        original_diagonal = {
            row["target_domain"]: row
            for row in original
            if row["condition"] == "target_norm_aligned" and row["source_domain"] == row["target_domain"]
        }
        if set(native) != set(original_diagonal) or len(native) != 4:
            raise ValueError(f"{model}: native/original diagonal mismatch")
        max_abs_diff = max(
            abs(float(native[domain][metric]) - float(original_diagonal[domain][metric]))
            for domain in native
            for metric in METRICS
        )
        if max_abs_diff > 1e-6:
            raise ValueError(f"{model}: native rows do not reproduce original diagonal (max diff {max_abs_diff})")

        target_geometry = geometry["targets"]
        mean_projection_energy = sum(float(value["projection_energy_fraction"]) for value in target_geometry.values()) / len(target_geometry)
        for row in aggregate_rows:
            table_rows.append(
                {
                    "model": model,
                    "rank1_common_energy": float(geometry["rank1_explained_energy_ratio"]),
                    "mean_projection_energy": mean_projection_energy,
                    "treatment": row["condition"],
                    "vector_family": row["vector_family"],
                    "relative_norm": float(row["mean_effective_norm_over_target_native"]),
                    "sufficiency": float(row["mean_sufficiency"]),
                    "necessity": float(row["mean_necessity"]),
                    "strict_flip": float(row["mean_strict_flip_rate"]),
                    "strict_drop": float(row["mean_strict_drop_rate"]),
                }
            )
        validation["models"][model] = {
            "completion_cells": len(long_rows),
            "native_diagonal_max_abs_metric_diff": max_abs_diff,
            "rank1_common_energy": float(geometry["rank1_explained_energy_ratio"]),
            "mean_projection_energy": mean_projection_energy,
        }

    with (campaign_root / "shared_direction_causal_ablation_all_models.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table_rows[0]))
        writer.writeheader()
        writer.writerows(table_rows)
    lines = [
        "# Shared-direction causal ablation — all Qwen3 scales",
        "",
        "Each row averages the four target domains, each evaluated on 100 held-out pairs. The shared direction is the model-specific global uncentered-SVD rank-1 direction fitted from the four training-split native vectors. `projection` and `residual` retain literal native amplitudes (no renormalization).",
        "",
        "| model | rank-1 energy | proj. energy | treatment | relative norm | Suff. | Necc. | strict flip | strict drop |",
        "|---|---:|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in table_rows:
        lines.append(
            "| {model} | {rank:.1%} | {projection:.1%} | {treatment} | {norm:.3f} | {suff:.3f} | {necc:.3f} | {flip:.1%} | {drop:.1%} |".format(
                model=row["model"],
                rank=row["rank1_common_energy"],
                projection=row["mean_projection_energy"],
                treatment=row["treatment"],
                norm=row["relative_norm"],
                suff=row["sufficiency"],
                necc=row["necessity"],
                flip=row["strict_flip"],
                drop=row["strict_drop"],
            )
        )
    (campaign_root / "shared_direction_causal_ablation_all_models.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (campaign_root / "validation_report.json").write_text(json.dumps(validation, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(campaign_root / "shared_direction_causal_ablation_all_models.md")


if __name__ == "__main__":
    main()
