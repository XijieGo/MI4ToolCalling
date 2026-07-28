#!/usr/bin/env python3
"""
Collect a unified benchmark summary across baseline / RelP / EAP-IG / feature circuits runs.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, List

if __package__ in {None, ""}:
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


def read_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def safe_float(value: object, default: float = float("nan")) -> float:
    try:
        value = float(value)
    except Exception:
        return default
    return value if math.isfinite(value) else default


def structural_mix_purity(raw: str) -> float:
    try:
        obj = json.loads(raw)
    except Exception:
        return float("nan")
    vals = [float(v) for v in obj.values()]
    total = sum(vals)
    if total <= 0:
        return float("nan")
    return max(vals) / total


def load_baseline_like(method: str, run_root: Path) -> Dict[str, object]:
    signed = read_json(run_root / "signed_validate" / "signed_group_report.json")
    circuit = read_json(run_root / "final_signed_circuit" / "final_signed_circuit_summary.json")
    functional_rows = read_csv_rows(run_root / "functional_groups" / "functional_group_summary.csv")
    full_row = next((r for r in signed.get("summary_rows", []) if str(r["group"]) == "full_signed_circuit"), {})
    functional_purity = [
        structural_mix_purity(str(row.get("structural_mix", "")))
        for row in functional_rows
        if row.get("functional_group")
    ]
    semantic_coherence = sum(x for x in functional_purity if math.isfinite(x)) / max(
        1,
        sum(1 for x in functional_purity if math.isfinite(x)),
    )
    return {
        "method": method,
        "run_root": str(run_root),
        "faithfulness_median": 0.5
        * (
            safe_float(full_row.get("promote_suff_ratio_median"))
            + safe_float(full_row.get("suppress_suff_ratio_median"))
        ),
        "behavior_top1": 0.5
        * (
            safe_float(full_row.get("promote_tool_top1_rate"))
            + safe_float(full_row.get("suppress_no_tool_top1_rate"))
        ),
        "n_nodes": int(circuit.get("n_nodes", 0)),
        "semantic_coherence": semantic_coherence,
        "runtime_seconds": float("nan"),
        "notes": "semantic_coherence is a proxy based on structural-mix purity inside functional groups",
    }


def load_feature_run(method: str, run_root: Path) -> Dict[str, object]:
    report = read_json(run_root / "validation" / "feature_group_report.json")
    selected = read_csv_rows(run_root / "discovery" / "selected_features.csv")
    purity_rows = read_csv_rows(run_root / "group_purity.csv")
    rows = list(report.get("summary_rows", []))
    best_row = max(
        rows,
        key=lambda r: safe_float(r.get("promote_suff_ratio_median")) + safe_float(r.get("suppress_suff_ratio_median")),
    ) if rows else {}
    purity_vals = [safe_float(r.get("semantic_purity")) for r in purity_rows]
    semantic_coherence = sum(x for x in purity_vals if math.isfinite(x)) / max(
        1,
        sum(1 for x in purity_vals if math.isfinite(x)),
    )
    return {
        "method": method,
        "run_root": str(run_root),
        "faithfulness_median": 0.5
        * (
            safe_float(best_row.get("promote_suff_ratio_median"))
            + safe_float(best_row.get("suppress_suff_ratio_median"))
        ),
        "behavior_top1": 0.5
        * (
            safe_float(best_row.get("promote_tool_top1_rate"))
            + safe_float(best_row.get("suppress_no_tool_top1_rate"))
        ),
        "n_nodes": len(selected),
        "semantic_coherence": semantic_coherence,
        "runtime_seconds": float("nan"),
        "notes": f"best feature group = {best_row.get('group', '')}",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect a unified method benchmark summary.")
    parser.add_argument("--baseline-root", type=str, default="")
    parser.add_argument("--relp-root", type=str, default="")
    parser.add_argument("--eap-root", type=str, default="")
    parser.add_argument("--feature-root", type=str, default="")
    parser.add_argument("--output-root", type=str, required=True)
    args = parser.parse_args()

    out_root = Path(args.output_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    summary_rows: List[Dict[str, object]] = []
    if args.baseline_root:
        summary_rows.append(load_baseline_like("baseline_signed_kl", Path(args.baseline_root).resolve()))
    if args.relp_root:
        summary_rows.append(load_baseline_like("relp", Path(args.relp_root).resolve()))
    if args.eap_root:
        summary_rows.append(load_baseline_like("eap_ig", Path(args.eap_root).resolve()))
    if args.feature_root:
        summary_rows.append(load_feature_run("feature_circuits", Path(args.feature_root).resolve()))

    summary_rows.sort(key=lambda r: safe_float(r.get("faithfulness_median"), 0.0), reverse=True)

    with (out_root / "benchmark_summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()) if summary_rows else ["method"])
        writer.writeheader()
        writer.writerows(summary_rows)

    summary = {
        "summary_rows": summary_rows,
        "artifacts": {
            "summary_csv": str(out_root / "benchmark_summary.csv"),
            "summary_json": str(out_root / "benchmark_summary.json"),
        },
    }
    (out_root / "benchmark_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
