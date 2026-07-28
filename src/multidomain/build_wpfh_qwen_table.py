#!/usr/bin/env python3
"""Audit completed Qwen-family transfer runs and render the wPFH table block.

This is a result-only postprocessor.  It refuses incomplete artifacts and
checks held-out cardinalities, norm matching, strict-rate denominators, and
the exact Qwen3 Code reproduction audit before rendering a Markdown table.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


MODELS = (
    ("qwen3_4b", "Qwen3-4B"),
    ("qwen3_8b", "Qwen3-8B"),
    ("qwen3_14b", "Qwen3-14B"),
    ("qwen35_4b", "Qwen3.5-4B"),
    ("qwen35_9b", "Qwen3.5-9B"),
)
RUN_DIRECTORY = {
    "qwen35_4b": Path("qwen35_4b") / "transfer",
    "qwen35_9b": Path("qwen35_9b") / "transfer",
}
TARGETS = (("D3", "Retrieval"), ("D4", "SQL"), ("D5", "Email"))
DIRECTIONS = ("Code", "Random")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"{path}: expected a JSON mapping")
    return value


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def pct(value: float) -> str:
    result = 100.0 * value
    rounded = round(result)
    if abs(result - rounded) < 1e-7:
        return str(rounded)
    return f"{result:.1f}".rstrip("0").rstrip(".")


def audit_run(root: Path, *, key: str) -> tuple[dict[tuple[str, str], dict[str, str]], dict[str, Any]]:
    completion = read_json(root / "completion.json")
    if completion.get("status") != "complete":
        raise ValueError(f"{key}: completion status is {completion.get('status')!r}")
    rows = read_rows(root / "matrix_long.csv")
    expected = {(target, direction) for target, _name in TARGETS for direction in DIRECTIONS}
    indexed: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        target = str(row.get("target_domain"))
        direction = str(row.get("direction"))
        if (target, direction) not in expected:
            continue
        if (target, direction) in indexed:
            raise ValueError(f"{key}: duplicate {target}/{direction} row")
        indexed[(target, direction)] = row
    missing = sorted(expected - set(indexed))
    if missing:
        raise ValueError(f"{key}: missing rows {missing}")

    audit: dict[str, Any] = {"completion": completion, "cells": {}}
    for target, direction in sorted(indexed):
        row = indexed[(target, direction)]
        n = int(row["n"])
        if n != 100:
            raise ValueError(f"{key}/{target}/{direction}: expected n=100, got {n}")
        clean_count = int(row["baseline_clean_tool_count"])
        corrupt_count = int(row["baseline_corrupt_non_tool_count"])
        if clean_count <= 0 or corrupt_count <= 0:
            raise ValueError(f"{key}/{target}/{direction}: empty strict-rate denominator")
        relative_norm = float(row["effective_norm_over_target_native"])
        if abs(relative_norm - 1.0) > 1e-5:
            raise ValueError(f"{key}/{target}/{direction}: norm mismatch {relative_norm}")
        flip_count = int(row["add_strict_flip_count"])
        drop_count = int(row["remove_strict_drop_count"])
        flip = float(row["add_strict_flip_rate"])
        drop = float(row["remove_strict_drop_rate"])
        if abs(flip - flip_count / corrupt_count) > 1e-10:
            raise ValueError(f"{key}/{target}/{direction}: inconsistent strict-flip rate")
        if abs(drop - drop_count / clean_count) > 1e-10:
            raise ValueError(f"{key}/{target}/{direction}: inconsistent strict-drop rate")
        audit["cells"][f"{target}/{direction}"] = {
            "strict_flip": {"count": flip_count, "denominator": corrupt_count, "rate": flip},
            "strict_drop": {"count": drop_count, "denominator": clean_count, "rate": drop},
            "effective_norm_over_target_native": relative_norm,
        }

    if key.startswith("qwen3_"):
        reproduction = read_json(root / "code_reproduction.json")
        for target, _name in TARGETS:
            for metric, delta in reproduction.get(target, {}).items():
                if abs(float(delta)) > 1e-8:
                    raise ValueError(f"{key}/{target}: Code reproduction differs for {metric}: {delta}")
        audit["code_reproduction"] = "exact"
    return indexed, audit


def main() -> None:
    args = parse_args()
    result_root = args.result_root.resolve()
    output_root = args.output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite {output_root}")
    result_by_model: dict[str, dict[tuple[str, str], dict[str, str]]] = {}
    audit: dict[str, Any] = {}
    for key, _label in MODELS:
        run_root = result_root / RUN_DIRECTORY.get(key, Path(key))
        result_by_model[key], audit[key] = audit_run(run_root, key=key)

    output_root.mkdir(parents=True, exist_ok=False)
    values: list[dict[str, Any]] = []
    for target, domain in TARGETS:
        for direction in DIRECTIONS:
            for key, label in MODELS:
                row = result_by_model[key][(target, direction)]
                values.append(
                    {
                        "domain": domain,
                        "direction": direction,
                        "model_key": key,
                        "model": label,
                        "strict_flip_count": int(row["add_strict_flip_count"]),
                        "strict_flip_denominator": int(row["baseline_corrupt_non_tool_count"]),
                        "strict_flip_rate": float(row["add_strict_flip_rate"]),
                        "strict_drop_count": int(row["remove_strict_drop_count"]),
                        "strict_drop_denominator": int(row["baseline_clean_tool_count"]),
                        "strict_drop_rate": float(row["remove_strict_drop_rate"]),
                        "cell": f"{pct(float(row['add_strict_flip_rate']))}/{pct(float(row['remove_strict_drop_rate']))}",
                    }
                )

    labels = [label for _key, label in MODELS]
    lines = [
        "# wPFH cross-domain table, Qwen-family candidate",
        "",
        "Cells are strict flip / strict drop (%). Exact count denominators are in `qwen_table_values.csv`.",
        "",
        "| Domain | Direction | " + " | ".join(labels) + " | Mistral | Granite |",
        "|---|---|" + "|".join(["---"] * (len(labels) + 2)) + "|",
    ]
    for target, domain in TARGETS:
        for direction in DIRECTIONS:
            cells = [
                next(item["cell"] for item in values if item["domain"] == domain and item["direction"] == direction and item["model_key"] == key)
                for key, _label in MODELS
            ]
            lines.append(f"| {domain} | {direction} | " + " | ".join(cells) + " | xx/xx | xx/xx |")
    lines.append("")
    lines.append("All five Qwen-family artifacts passed completion, held-out cardinality, norm-match, and strict-rate denominator audits. Qwen3 Code values additionally exactly reproduce their prior frozen matrices.")
    (output_root / "candidate_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    with (output_root / "qwen_table_values.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(values[0]))
        writer.writeheader()
        writer.writerows(values)
    (output_root / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "complete", "output_root": str(output_root), "cells": len(values)}))


if __name__ == "__main__":
    main()
