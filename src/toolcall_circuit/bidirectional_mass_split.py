#!/usr/bin/env python3
"""
Split selective-group support mass into shared-backbone overlap vs unique branch.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

if __package__ in {None, ""}:
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


def write_csv(rows: List[Dict[str, object]], out_path: Path) -> None:
    if not rows:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure support mass split between backbone overlap and unique branch.")
    parser.add_argument("--bidirectional-summary", type=str, required=True)
    parser.add_argument("--node-support-csv", type=str, required=True)
    parser.add_argument("--output-root", type=str, required=True)
    args = parser.parse_args()

    out_root = Path(args.output_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    bidi = json.loads(Path(args.bidirectional_summary).resolve().read_text(encoding="utf-8"))
    rows_by_node: Dict[str, Dict[str, str]] = {}
    with Path(args.node_support_csv).resolve().open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows_by_node[str(row["node"])] = row

    support = bidi.get("support_analysis", {})
    shared = set(map(str, support.get("shared_backbone_nodes", [])))
    group_specs = [
        ("forward_selective", set(map(str, support.get("forward_selective_nodes", []))), "forward_support"),
        ("reverse_selective", set(map(str, support.get("reverse_selective_nodes", []))), "reverse_support"),
    ]

    out_rows: List[Dict[str, object]] = []
    for group_name, nodes, support_key in group_specs:
        overlap = sorted(nodes & shared)
        unique = sorted(nodes - shared)
        total_mass = sum(float(rows_by_node[n][support_key]) for n in nodes if n in rows_by_node)
        overlap_mass = sum(float(rows_by_node[n][support_key]) for n in overlap if n in rows_by_node)
        unique_mass = sum(float(rows_by_node[n][support_key]) for n in unique if n in rows_by_node)
        out_rows.append(
            {
                "group": group_name,
                "support_key": support_key,
                "n_nodes_total": len(nodes),
                "n_nodes_overlap": len(overlap),
                "n_nodes_unique": len(unique),
                "total_mass": total_mass,
                "overlap_mass": overlap_mass,
                "unique_mass": unique_mass,
                "overlap_mass_fraction": overlap_mass / total_mass if total_mass else 0.0,
                "unique_mass_fraction": unique_mass / total_mass if total_mass else 0.0,
                "overlap_nodes": ",".join(overlap),
                "unique_nodes": ",".join(unique),
            }
        )

    summary = {
        "artifacts": {
            "csv": str(out_root / "selective_mass_split.csv"),
            "json": str(out_root / "selective_mass_split.json"),
        },
        "rows": out_rows,
    }
    write_csv(out_rows, out_root / "selective_mass_split.csv")
    (out_root / "selective_mass_split.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
