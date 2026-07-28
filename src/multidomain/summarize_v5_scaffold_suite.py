#!/usr/bin/env python3
"""Validate and render table-ready artifacts for one v5 scaffold suite."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


LADDER_ORDER = ("L0", "L1", "L2", "L3", "L4", "L5", "L6", "L7")
COMPONENT_ORDER = ("RTF", "-TF", "R-F", "RT-", "--F", "-T-", "R--", "---", "R_TLEN_F")
REBUTTAL_COMPONENTS = ("RTF", "R-F", "RT-", "--F", "R_TLEN_F")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request-ladder-root", type=Path, required=True)
    parser.add_argument("--component-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"{path}: expected a JSON object")
    return value


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def probability(value: float) -> str:
    if value == 0.0:
        return "0"
    if abs(value) < 1e-4:
        return f"{value:.2e}"
    return f"{value:.4f}"


def top1_cell(row: dict[str, str]) -> str:
    n = int(row["behavior_n"])
    rate = float(row["behavior_tool_call_top1_rate"])
    count = round(rate * n)
    require(abs(rate - count / n) < 1e-9, f"{row['ladder_level']}: non-integral top-1 rate")
    return f"{count}/{n} ({100.0 * rate:.1f}%)"


def validate_roots(ladder_root: Path, component_root: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    ladder_completion = read_json(ladder_root / "completion.json")
    component_completion = read_json(component_root / "completion.json")
    require(ladder_completion.get("status") == "complete", "Request-ladder run is incomplete")
    require(component_completion.get("status") == "complete", "Component run is incomplete")
    require(ladder_completion.get("dataset_cardinality") == {"heldout": 300}, "Unexpected request-ladder cardinality")
    require(component_completion.get("dataset_cardinality") == {"train": 200, "heldout": 300}, "Unexpected component cardinality")

    ladder_provenance = read_json(ladder_root / "dataset_provenance.json")
    component_provenance = read_json(component_root / "dataset_provenance.json")
    for provenance in (ladder_provenance, component_provenance):
        require(provenance.get("dataset_version") == "v5_model_specific_balanced", "Unexpected dataset version")
        require(provenance.get("model_key") == "qwen3_8b", "This report expects the Qwen3-8B v5 release")
        require(provenance.get("tool_call_token_id") == 151657, "Unexpected Qwen3-8B tool token")
    ladder_manifest_hash = ladder_provenance["release_files"]["manifest"]["sha256"]
    component_manifest_hash = component_provenance["release_files"]["manifest"]["sha256"]
    require(ladder_manifest_hash == component_manifest_hash, "Runs use different v5 manifests")
    require(ladder_completion.get("dataset_label") == component_completion.get("dataset_label"), "Runs use different labels")
    return ladder_provenance, component_provenance, {
        "dataset_label": ladder_completion["dataset_label"],
        "manifest_sha256": ladder_manifest_hash,
        "request_ladder_root": str(ladder_root),
        "component_root": str(component_root),
    }


def ladder_markdown(rows: list[dict[str, str]]) -> str:
    indexed = {row["ladder_level"]: row for row in rows}
    require(tuple(indexed) == LADDER_ORDER, f"Unexpected ladder rows: {tuple(indexed)}")
    lines = [
        "| Request | n | `<tool_call>` top-1 | Mean probability |",
        "|---|---:|---:|---:|",
    ]
    for level in LADDER_ORDER:
        row = indexed[level]
        lines.append(
            f"| {level}: {row['description']} | {row['behavior_n']} | {top1_cell(row)} | "
            f"{probability(float(row['behavior_mean_tool_call_prob']))} |"
        )
    return "\n".join(lines)


def component_markdown(
    derived_rows: list[dict[str, str]], behavior_rows: list[dict[str, str]], *, selected: tuple[str, ...]
) -> str:
    derived = {row["scaffold"]: row for row in derived_rows}
    behavior = {(row["scaffold"], row["request_type"]): row for row in behavior_rows}
    require(tuple(derived) == COMPONENT_ORDER, f"Unexpected component rows: {tuple(derived)}")
    lines = [
        "| Scaffold components | Neutral `p_call` | Analysis `p_call` | `P = neutral - analysis` |",
        "|---|---:|---:|---:|",
    ]
    for scaffold in selected:
        row = derived[scaffold]
        analysis = behavior[(scaffold, "analysis")]
        lines.append(
            f"| {row['scaffold_label']} | {probability(float(row['default_strength_p_call']))} | "
            f"{probability(float(analysis['behavior_mean_tool_call_prob']))} | "
            f"{probability(float(row['suppression_depth_p_call']))} |"
        )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    ladder_root = args.request_ladder_root.resolve()
    component_root = args.component_root.resolve()
    output_root = args.output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite existing directory: {output_root}")

    ladder_provenance, component_provenance, source = validate_roots(ladder_root, component_root)
    ladder_rows = read_csv(ladder_root / "ladder_summary.csv")
    derived_rows = read_csv(component_root / "derived_default_suppression.csv")
    behavior_rows = read_csv(component_root / "behavior_long.csv")
    require(tuple(row["ladder_level"] for row in ladder_rows) == LADDER_ORDER, "Unexpected ladder ordering")
    require(tuple(row["scaffold"] for row in derived_rows) == COMPONENT_ORDER, "Unexpected component ordering")

    output_root.mkdir(parents=True, exist_ok=False)
    summary = {
        "status": "complete",
        "source": source,
        "dataset_provenance": {
            "model_key": ladder_provenance["model_key"],
            "model_label": ladder_provenance["model_label"],
            "dataset_version": ladder_provenance["dataset_version"],
            "manifest_sha256": source["manifest_sha256"],
            "train_count": component_provenance["manifest_split_counts"]["train"],
            "heldout_count": ladder_provenance["manifest_split_counts"]["heldout"],
        },
        "ladder": {row["ladder_level"]: row for row in ladder_rows},
        "components": {row["scaffold"]: row for row in derived_rows},
    }
    write_json(output_root / "summary.json", summary)
    (output_root / "request_ladder_table.md").write_text(ladder_markdown(ladder_rows) + "\n", encoding="utf-8")
    (output_root / "scaffold_components_table.md").write_text(
        component_markdown(derived_rows, behavior_rows, selected=REBUTTAL_COMPONENTS) + "\n", encoding="utf-8"
    )
    (output_root / "scaffold_factorial_full_table.md").write_text(
        component_markdown(derived_rows, behavior_rows, selected=COMPONENT_ORDER) + "\n", encoding="utf-8"
    )
    (output_root / "README.md").write_text(
        "# Qwen3-8B v5 scaffold suite\n\n"
        "This directory is a result-only audit bundle for the v5 model-specific Qwen3-8B release. "
        "`request_ladder_table.md` keeps L0, L1, and L2 separate. "
        "`scaffold_components_table.md` reports both neutral and analysis probabilities, including near-zero values.\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "complete", "output_root": str(output_root), "manifest_sha256": source["manifest_sha256"]}))


if __name__ == "__main__":
    main()
