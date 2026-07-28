#!/usr/bin/env python3
"""
Summarize signed circuit edges at the semantic-group family level.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

if __package__ in {None, ""}:
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from toolcall_circuit.graph_utils import apply_plot_style
from toolcall_circuit.signed_circuit import EDGE_NO_TOOL, EDGE_SHARED, EDGE_TOOL, GROUP_META


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(rows: Sequence[Dict[str, object]], out_path: Path) -> None:
    if not rows:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def draw_family_graph(rows: Sequence[Dict[str, object]], out_path: Path) -> None:
    groups = [
        "no_tool_tail",
        "no_tool_bias_backbone",
        "symmetric_backbone",
        "tool_bias_backbone",
        "tool_tail",
    ]
    groups = [g for g in groups if g in {str(r["source_group"]) for r in rows} | {str(r["target_group"]) for r in rows}]
    pos = {
        "no_tool_tail": (-2.0, 0.0),
        "no_tool_bias_backbone": (-1.0, 1.0),
        "symmetric_backbone": (0.0, 2.0),
        "tool_bias_backbone": (1.0, 1.0),
        "tool_tail": (2.0, 0.0),
    }

    apply_plot_style()
    fig, ax = plt.subplots(figsize=(10.4, 7.2), constrained_layout=True)
    ax.axis("off")
    ax.set_title("Signed Circuit Family Graph")

    max_support = max(float(r["union_support_median"]) for r in rows) if rows else 1.0
    for row in rows:
        s = str(row["source_group"])
        t = str(row["target_group"])
        x1, y1 = pos[s]
        x2, y2 = pos[t]
        sign = str(row["sign"])
        color = EDGE_SHARED if sign == "shared" else (EDGE_TOOL if sign == "tool_bias" else EDGE_NO_TOOL)
        width = 1.3 + 4.0 * float(row["union_support_median"]) / max_support
        rad = 0.12 if s != t else 0.35
        arrow = FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle="->",
            mutation_scale=16,
            linewidth=width,
            color=color,
            alpha=0.88,
            connectionstyle=f"arc3,rad={rad}",
            zorder=1,
        )
        ax.add_patch(arrow)
        mx = (x1 + x2) / 2
        my = (y1 + y2) / 2 + (0.20 if sign == "tool_bias" else (-0.20 if sign == "no_tool_bias" else 0.0))
        ax.text(mx, my, str(row["n_edges"]), fontsize=9, ha="center", va="center", color="#1f1f1f")

    for g in groups:
        x, y = pos[g]
        meta = GROUP_META[g]
        ax.scatter([x], [y], s=1400, c=meta["color"], edgecolors="#2d2d2d", linewidths=1.5, zorder=3)
        ax.text(x, y - 0.42, meta["label"], ha="center", va="top", fontsize=11)

    ax.set_xlim(-3.0, 3.0)
    ax.set_ylim(-1.0, 3.0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize signed circuit edges by semantic family.")
    parser.add_argument("--signed-edges-csv", type=str, required=True)
    parser.add_argument("--output-root", type=str, required=True)
    args = parser.parse_args()

    out_root = Path(args.output_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    edge_rows = read_csv_rows(Path(args.signed_edges_csv).resolve())
    grouped: Dict[Tuple[str, str, str], List[Dict[str, str]]] = defaultdict(list)
    for row in edge_rows:
        if str(row["target"]) == "Residual Output: decision":
            continue
        key = (str(row["source_group"]), str(row["target_group"]), str(row["sign"]))
        grouped[key].append(row)

    summary_rows: List[Dict[str, object]] = []
    for (source_group, target_group, sign), rows in grouped.items():
        rows_sorted = sorted(rows, key=lambda r: float(r["union_support_max"]), reverse=True)
        summary_rows.append(
            {
                "source_group": source_group,
                "target_group": target_group,
                "sign": sign,
                "n_edges": len(rows),
                "union_support_median": float(
                    sorted(float(r["union_support_max"]) for r in rows)[len(rows) // 2]
                ),
                "shared_support_median": float(
                    sorted(float(r["shared_support_min"]) for r in rows)[len(rows) // 2]
                ),
                "direction_balance_median": float(
                    sorted(float(r["direction_balance"]) for r in rows)[len(rows) // 2]
                ),
                "top_edges": " | ".join(
                    f"{r['source']}->{r['target']} ({float(r['union_support_max']):.3f})" for r in rows_sorted[:4]
                ),
            }
        )
    summary_rows.sort(key=lambda r: (float(r["union_support_median"]), int(r["n_edges"])), reverse=True)

    draw_family_graph(summary_rows, out_root / "signed_family_graph.png")
    write_csv(summary_rows, out_root / "signed_family_summary.csv")
    summary = {
        "artifacts": {
            "summary_csv": str(out_root / "signed_family_summary.csv"),
            "graph_png": str(out_root / "signed_family_graph.png"),
            "summary_json": str(out_root / "signed_family_summary.json"),
        },
        "rows": summary_rows,
    }
    (out_root / "signed_family_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
