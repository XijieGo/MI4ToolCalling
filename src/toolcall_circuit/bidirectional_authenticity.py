#!/usr/bin/env python3
"""
Authenticity-style analysis for bidirectional causal patching results.

This script consumes `per_sample_cross_eval.json` and converts patch ratios into
more directly interpretable decision metrics:

- sign consistency: intervention moves the margin in the hypothesized direction
- boundary flip rate: intervention crosses the tool-call / no-tool decision boundary
- source dominance rate: patched state is closer to the source endpoint than the base
- endpoint authenticity: normalized closeness to the source endpoint in margin space

For a given sample with base margin m_base and source margin m_source, define:

  m_patched = m_base + ratio * gap
  gap = m_source - m_base

Then endpoint authenticity is

  A = clip(1 - |m_patched - m_source| / |m_source - m_base|, 0, 1)

This equals 1 when the intervention exactly reconstructs the source endpoint, 0
when it stays at the base endpoint, and remains bounded in [0, 1].
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


METRIC_RE = re.compile(r"^(promote|suppress)__(.+)__ratio$")


def finite(values: Iterable[float]) -> List[float]:
    return [float(v) for v in values if isinstance(v, (int, float)) and math.isfinite(float(v))]


def median(values: Iterable[float]) -> float:
    vals = finite(values)
    return float(np.median(vals)) if vals else float("nan")


def mean(values: Iterable[float]) -> float:
    vals = finite(values)
    return float(np.mean(vals)) if vals else float("nan")


def safe_rate(values: Iterable[bool]) -> float:
    vals = [1.0 if bool(v) else 0.0 for v in values]
    return float(np.mean(vals)) if vals else float("nan")


def clip01(x: float) -> float:
    return min(1.0, max(0.0, float(x)))


def write_csv(rows: List[Dict[str, object]], out_path: Path) -> None:
    if not rows:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze authenticity-style metrics for bidirectional causal patching.")
    parser.add_argument("--per-sample-cross-eval", type=str, required=True)
    parser.add_argument("--output-root", type=str, required=True)
    args = parser.parse_args()

    out_root = Path(args.output_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    per_sample = json.loads(Path(args.per_sample_cross_eval).resolve().read_text(encoding="utf-8"))
    if not isinstance(per_sample, list) or not per_sample:
        raise ValueError("per-sample cross eval file is empty or malformed")

    metric_keys = []
    for key in per_sample[0]:
        if METRIC_RE.match(key):
            metric_keys.append(key)
    metric_keys.sort()

    per_intervention_rows: List[Dict[str, object]] = []
    grouped: Dict[str, List[Dict[str, object]]] = {}

    for row in per_sample:
        sample_id = str(row["sample_id"])
        m_tool_toolcall = float(row["m_tool_toolcall"])
        m_tool_no_tool = float(row["m_tool_no_tool"])
        gap = float(row["gap"])
        if not math.isfinite(gap) or abs(gap) < 1e-8:
            continue

        for metric_key in metric_keys:
            if metric_key not in row:
                continue
            ratio = float(row[metric_key])
            match = METRIC_RE.match(metric_key)
            if not match:
                continue
            mode, group = match.groups()
            if mode == "promote":
                base_margin = m_tool_no_tool
                source_margin = m_tool_toolcall
                sign_ok = ratio > 0.0
                boundary_flip = (base_margin <= 0.0) and (base_margin + ratio * gap > 0.0)
            else:
                base_margin = m_tool_toolcall
                source_margin = m_tool_no_tool
                sign_ok = ratio < 0.0
                boundary_flip = (base_margin >= 0.0) and (base_margin + ratio * gap < 0.0)

            patched_margin = base_margin + ratio * gap
            endpoint_distance = abs(source_margin - base_margin)
            endpoint_authenticity = clip01(1.0 - abs(patched_margin - source_margin) / endpoint_distance)
            source_dominance = abs(patched_margin - source_margin) <= abs(patched_margin - base_margin)
            closer_to_source = abs(patched_margin - source_margin) < abs(source_margin - base_margin)
            signed_recovery = ratio if mode == "promote" else -ratio

            out = {
                "sample_id": sample_id,
                "mode": mode,
                "group": group,
                "ratio": ratio,
                "signed_recovery": signed_recovery,
                "base_margin": base_margin,
                "source_margin": source_margin,
                "patched_margin": patched_margin,
                "sign_consistent": sign_ok,
                "boundary_flip": boundary_flip,
                "closer_to_source": closer_to_source,
                "source_dominance": source_dominance,
                "endpoint_authenticity": endpoint_authenticity,
            }
            per_intervention_rows.append(out)
            grouped.setdefault(metric_key, []).append(out)

    per_group_rows: List[Dict[str, object]] = []
    for metric_key, rows in sorted(grouped.items()):
        mode, group = METRIC_RE.match(metric_key).groups()
        ratios = [float(r["ratio"]) for r in rows]
        signed_recoveries = [float(r["signed_recovery"]) for r in rows]
        patched_margins = [float(r["patched_margin"]) for r in rows]
        endpoint_authenticities = [float(r["endpoint_authenticity"]) for r in rows]
        per_group_rows.append(
            {
                "metric_key": metric_key,
                "mode": mode,
                "group": group,
                "n_samples": len(rows),
                "ratio_median": median(ratios),
                "ratio_mean": mean(ratios),
                "signed_recovery_median": median(signed_recoveries),
                "signed_recovery_mean": mean(signed_recoveries),
                "endpoint_authenticity_median": median(endpoint_authenticities),
                "endpoint_authenticity_mean": mean(endpoint_authenticities),
                "sign_consistency_rate": safe_rate(r["sign_consistent"] for r in rows),
                "boundary_flip_rate": safe_rate(r["boundary_flip"] for r in rows),
                "closer_to_source_rate": safe_rate(r["closer_to_source"] for r in rows),
                "source_dominance_rate": safe_rate(r["source_dominance"] for r in rows),
                "patched_margin_median": median(patched_margins),
                "recovery_ge_0_5_rate": safe_rate(float(r["signed_recovery"]) >= 0.5 for r in rows),
                "recovery_ge_0_9_rate": safe_rate(float(r["signed_recovery"]) >= 0.9 for r in rows),
            }
        )

    dual_rows: List[Dict[str, object]] = []
    by_group_mode = {(r["group"], r["mode"]): r for r in per_group_rows}
    groups = sorted({str(r["group"]) for r in per_group_rows})
    for group in groups:
        promote = by_group_mode.get((group, "promote"))
        suppress = by_group_mode.get((group, "suppress"))
        if not promote or not suppress:
            continue
        promote_strength = float(promote["signed_recovery_median"])
        suppress_strength = float(suppress["signed_recovery_median"])
        denom = max(promote_strength, suppress_strength, 1e-8)
        dual_rows.append(
            {
                "group": group,
                "promote_signed_recovery_median": promote_strength,
                "suppress_signed_recovery_median": suppress_strength,
                "promote_boundary_flip_rate": float(promote["boundary_flip_rate"]),
                "suppress_boundary_flip_rate": float(suppress["boundary_flip_rate"]),
                "promote_endpoint_authenticity_median": float(promote["endpoint_authenticity_median"]),
                "suppress_endpoint_authenticity_median": float(suppress["endpoint_authenticity_median"]),
                "duality_balance": min(promote_strength, suppress_strength) / denom,
            }
        )
    dual_rows.sort(key=lambda r: (r["duality_balance"], r["promote_signed_recovery_median"]), reverse=True)

    summary = {
        "n_samples": len({str(r["sample_id"]) for r in per_intervention_rows}),
        "n_metrics": len(per_group_rows),
        "artifacts": {
            "per_intervention_csv": str(out_root / "per_intervention_authenticity.csv"),
            "per_group_csv": str(out_root / "per_group_authenticity.csv"),
            "duality_csv": str(out_root / "group_duality.csv"),
            "summary_json": str(out_root / "authenticity_summary.json"),
        },
        "top_by_endpoint_authenticity": sorted(
            per_group_rows,
            key=lambda r: (
                float(r["endpoint_authenticity_median"]),
                float(r["boundary_flip_rate"]),
                float(r["sign_consistency_rate"]),
            ),
            reverse=True,
        )[:8],
        "top_by_boundary_flip": sorted(
            per_group_rows,
            key=lambda r: (
                float(r["boundary_flip_rate"]),
                float(r["endpoint_authenticity_median"]),
            ),
            reverse=True,
        )[:8],
        "group_duality": dual_rows,
    }

    write_csv(per_intervention_rows, out_root / "per_intervention_authenticity.csv")
    write_csv(per_group_rows, out_root / "per_group_authenticity.csv")
    write_csv(dual_rows, out_root / "group_duality.csv")
    (out_root / "authenticity_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
