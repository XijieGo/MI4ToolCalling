#!/usr/bin/env python3
"""
Stratify bidirectional results by dataset/language.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Dict, Iterable, List, Tuple

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


def finite(vals: Iterable[float]) -> List[float]:
    return [float(v) for v in vals if isinstance(v, (int, float)) and math.isfinite(float(v))]


def med(vals: Iterable[float]) -> float:
    xs = finite(vals)
    return float(median(xs)) if xs else float("nan")


def mean(vals: Iterable[float]) -> float:
    xs = finite(vals)
    return float(np.mean(xs)) if xs else float("nan")


def load_manifest_map(manifest_path: Path) -> Dict[str, Dict[str, str]]:
    out: Dict[str, Dict[str, str]] = {}
    if not manifest_path.exists():
        return out
    with manifest_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            fname = str(row.get("output_filename") or row.get("source_filename") or "")
            if not fname.endswith(".txt"):
                continue
            sample_id = Path(fname).stem
            out[sample_id] = {
                "dataset_name": str(row.get("dataset_name") or row.get("dataset") or ""),
                "language": str(row.get("language") or ""),
            }
    return out


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Stratify bidirectional results by dataset/language.")
    parser.add_argument("--clean-manifest", type=str, default="datasets/clean/manifest.jsonl")
    parser.add_argument("--per-sample-cross-eval", type=str, required=True)
    parser.add_argument("--per-sample-overlap-csv", type=str, required=True)
    parser.add_argument("--output-root", type=str, required=True)
    args = parser.parse_args()

    out_root = Path(args.output_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    meta = load_manifest_map(Path(args.clean_manifest).resolve())
    per_sample = json.loads(Path(args.per_sample_cross_eval).resolve().read_text(encoding="utf-8"))
    overlap_rows = read_csv(Path(args.per_sample_overlap_csv).resolve())

    # Build dataset key map.
    def key_for(sample_id: str) -> Tuple[str, str]:
        m = meta.get(sample_id, {})
        return (m.get("dataset_name", "") or "unknown", m.get("language", "") or "unknown")

    # Stratify cross-eval metrics.
    metric_keys = sorted({k for row in per_sample for k in row.keys() if k.endswith("__ratio")})
    groups: Dict[Tuple[str, str], Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for row in per_sample:
        sid = str(row.get("sample_id", ""))
        if not sid:
            continue
        k = key_for(sid)
        for mk in metric_keys:
            groups[k][mk].append(float(row.get(mk, float("nan"))))
        groups[k]["gap"].append(float(row.get("gap", float("nan"))))

    causal_out: List[Dict[str, object]] = []
    for (dataset, lang), vals_by_key in sorted(groups.items()):
        base = {
            "dataset_name": dataset,
            "language": lang,
            "n_samples": len(finite(vals_by_key.get("gap", []))),
            "gap_median": med(vals_by_key.get("gap", [])),
            "gap_mean": mean(vals_by_key.get("gap", [])),
        }
        for mk in metric_keys:
            base[f"{mk}__median"] = med(vals_by_key.get(mk, []))
            base[f"{mk}__mean"] = mean(vals_by_key.get(mk, []))
        causal_out.append(base)
    write_csv(out_root / "stratified_causal_metrics.csv", causal_out)

    # Stratify overlap.
    overlap_groups: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    for row in overlap_rows:
        sid = str(row.get("sample_id", ""))
        if not sid:
            continue
        k = key_for(sid)
        overlap_groups[k].append(float(row.get("node_jaccard", "nan")))
    overlap_out: List[Dict[str, object]] = []
    for (dataset, lang), vals in sorted(overlap_groups.items()):
        overlap_out.append(
            {
                "dataset_name": dataset,
                "language": lang,
                "n_samples": len(finite(vals)),
                "node_jaccard_median": med(vals),
                "node_jaccard_mean": mean(vals),
            }
        )
    write_csv(out_root / "stratified_overlap.csv", overlap_out)

    report = {
        "n_total_samples": len(per_sample),
        "n_total_overlap_rows": len(overlap_rows),
        "datasets": sorted({d for (d, _) in groups.keys()}),
        "languages": sorted({l for (_, l) in groups.keys()}),
        "artifacts": {
            "stratified_causal_metrics_csv": str(out_root / "stratified_causal_metrics.csv"),
            "stratified_overlap_csv": str(out_root / "stratified_overlap.csv"),
        },
    }
    (out_root / "stratified_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
