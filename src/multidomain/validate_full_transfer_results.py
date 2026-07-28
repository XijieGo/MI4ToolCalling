#!/usr/bin/env python3
"""Validate cardinality, tokenization, finiteness, and the Qwen3-8B anchor cells."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


DOMAINS = ("D1", "D3", "D4", "D5")
CONDITIONS = ("target_norm_aligned", "raw_1p5")
MODELS = ("Qwen3-1.7B", "Qwen3-4B", "Qwen3-8B", "Qwen3-14B")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate full Qwen3 transfer-matrix artifacts.")
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        result = json.load(handle)
    if not isinstance(result, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return result


def exact_alpha_row(path: Path, alpha: float) -> dict[str, str]:
    for row in load_csv(path):
        if abs(float(row["alpha"]) - alpha) < 1e-12:
            return row
    raise KeyError(f"No alpha={alpha} in {path}")


def assert_close(name: str, observed: float, expected: float, tolerance: float = 1e-9) -> None:
    if abs(observed - expected) > tolerance:
        raise AssertionError(f"{name}: {observed} != {expected} (tol={tolerance})")


def main() -> None:
    args = parse_args()
    run_root = args.run_root.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")
    checks: list[str] = []
    model_summaries: dict[str, Any] = {}
    for model in MODELS:
        root = run_root / model
        completion = load_json(root / "completion.json")
        if completion.get("status") != "complete" or int(completion.get("observed_cells", -1)) != 32:
            raise AssertionError(f"{model}: completion metadata invalid")
        rows = load_csv(root / "matrix_long.csv")
        expected_keys = {(source, target, condition) for source in DOMAINS for target in DOMAINS for condition in CONDITIONS}
        observed_keys = {(row["source_domain"], row["target_domain"], row["condition"]) for row in rows}
        if len(rows) != 32 or observed_keys != expected_keys:
            raise AssertionError(f"{model}: matrix is not a complete 4x4x2 grid")
        for row in rows:
            if int(row["n"]) != 100:
                raise AssertionError(f"{model}: non-100 held-out count in {row}")
            for key, value in row.items():
                if key in {"source_domain", "target_domain", "condition", "n"}:
                    continue
                if not math.isfinite(float(value)):
                    raise AssertionError(f"{model}: non-finite {key}")
        pair_validation = load_json(root / "pair_validation.json")
        for domain in DOMAINS:
            for split, expected_count in (("train", 400), ("test", 100)):
                payload = pair_validation.get(f"{domain}/{split}")
                if not isinstance(payload, dict) or int(payload.get("pair_count", -1)) != expected_count:
                    raise AssertionError(f"{model}: invalid pair validation for {domain}/{split}")
                if len(payload.get("unique_differing_token_positions", [])) != 1:
                    raise AssertionError(f"{model}: expected one shared differing-token position for {domain}/{split}")
        pca = load_json(root / "pca_common_direction.json")
        if not (0.0 <= float(pca["rank1_explained_energy_ratio"]) <= 1.0):
            raise AssertionError(f"{model}: invalid PCA rank-1 energy")
        model_summaries[model] = {
            "d_model": int(completion["model_d_model"]),
            "n_layers": int(completion["model_n_layers"]),
            "rank1_common_direction_energy": float(pca["rank1_explained_energy_ratio"]),
        }
        checks.append(f"{model}: complete 4x4x2 grid, finite metrics, 400/100 validated one-token pairs, valid PCA")

    # Anchor the batched implementation against the independently run Qwen3-8B
    # cells already completed before the matrix job.
    prior_root = Path("./results/runs/v4_d1_mu_transfer_qwen3_8b_20260724T173038Z")
    qwen8_rows = {
        (row["source_domain"], row["target_domain"], row["condition"]): row
        for row in load_csv(run_root / "Qwen3-8B" / "matrix_long.csv")
    }
    baseline_metrics: dict[str, dict[str, Any]] = {}
    for target in ("D3", "D4", "D5"):
        baseline_metrics[target] = load_json(prior_root / f"d1_to_{target.lower()}" / "logit_gap" / "logit_gap_metrics.json")
    anchor_cells = 0
    for target in ("D3", "D4", "D5"):
        baseline = baseline_metrics[target]
        gap = float(baseline["clean_mean_tool_logit"] - baseline["corrupt_mean_tool_logit"])
        for condition, alpha, root in (
            ("target_norm_aligned", 1.0, prior_root / f"aligned_d1_to_{target.lower()}_alpha_sweep"),
            ("raw_1p5", 1.5, prior_root / f"raw_d1_to_{target.lower()}_alpha_sweep"),
        ):
            row = qwen8_rows[("D1", target, condition)]
            add = exact_alpha_row(root / "alpha_sweep_add.csv", alpha)
            remove = exact_alpha_row(root / "alpha_sweep_remove.csv", alpha)
            expectations = {
                "add_tool_call_top1_rate": float(add["tool_call_top1_rate"]),
                "add_strict_flip_rate": float(add["strict_flip_rate"]),
                "add_mean_tool_call_logit": float(add["mean_tool_call_logit"]),
                "sufficiency_normalized_logit_gap": (float(add["mean_tool_call_logit"]) - float(baseline["corrupt_mean_tool_logit"])) / gap,
                "remove_remaining_tool_call_top1_rate": float(remove["remaining_tool_call_top1_rate"]),
                "remove_strict_drop_rate": float(remove["strict_drop_rate"]),
                "remove_mean_tool_call_logit": float(remove["mean_tool_call_logit"]),
                "necessity_normalized_logit_gap": (float(baseline["clean_mean_tool_logit"]) - float(remove["mean_tool_call_logit"])) / gap,
            }
            for metric, expected in expectations.items():
                assert_close(f"Qwen3-8B D1->{target} {condition} {metric}", float(row[metric]), expected)
            anchor_cells += 1
    checks.append(f"Qwen3-8B: {anchor_cells} independent D1→D3/D4/D5 condition cells exactly reproduce prior single-vector runs")

    report = {
        "status": "pass",
        "run_root": str(run_root),
        "checks": checks,
        "models": model_summaries,
        "anchor_cells_exactly_reproduced": anchor_cells,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
