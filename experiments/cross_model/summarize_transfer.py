#!/usr/bin/env python3
"""Normalize the compact implicit-intent and tau2 evidence index."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from mi4tc.io import load_json, write_json  # noqa: E402


def find_condition(node: Any, name: str) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    if isinstance(node, dict):
        if node.get("condition") == name:
            found.append(node)
        for value in node.values():
            found.extend(find_condition(value, name))
    elif isinstance(node, list):
        for value in node:
            found.extend(find_condition(value, name))
    return found


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize verb-free and tau2 transfer results.")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    implicit = load_json(here / "implicit_intent/cross_model_summary.json")
    tau2 = load_json(here / "tau2_transfer/run_summary.json")
    implicit_rows = []
    for key, value in implicit.items():
        if not isinstance(value, dict):
            continue
        conditions = value.get("conditions", {})
        arm = conditions.get("mean_diff_a1.0", {})
        implicit_rows.append(
            {
                "model": value.get("model", key),
                "baseline_call_top1": value.get("baseline_tool_call_top1"),
                "removal_n": arm.get("n"),
                "removal_strict_drop": arm.get("strict_drop"),
                "removal_strict_drop_rate": arm.get("strict_drop_rate"),
            }
        )
    tau2_rows = []
    for key, value in tau2.get("models", {}).items():
        if not isinstance(value, dict):
            continue
        suppression = find_condition(value.get("results", {}).get("suppression", {}), "minus_mean_diff_alpha_1")
        induction = find_condition(value.get("results", {}).get("induction", {}), "plus_mean_diff_alpha_1")
        tau2_rows.append(
            {
                "model": value.get("model", key),
                "suppression_alpha1": suppression[0] if suppression else None,
                "induction_alpha1": induction[0] if induction else None,
            }
        )
    report = {
        "implicit_intent": implicit_rows,
        "tau2": tau2_rows,
        "scope": "Behavior-level transfer summary; raw tau2 trajectories and model weights are external.",
    }
    if args.output:
        write_json(args.output, report)
    else:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
