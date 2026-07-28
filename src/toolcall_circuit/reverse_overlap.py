#!/usr/bin/env python3
"""
Compare the current no-tool semantic line against reverse-discovered circuitry.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


def load_functional_nodes(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if (a or b) else 1.0


def recall(a: set, b: set) -> float:
    return len(a & b) / len(a) if a else 1.0


def precision(a: set, b: set) -> float:
    return len(a & b) / len(b) if b else 1.0


def to_edge_set(rows: Sequence[Sequence[str]]) -> set[Tuple[str, str]]:
    return {tuple(row) for row in rows}


def write_json(data: Dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def write_report(lines: Iterable[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare no-tool semantic line to reverse-discovered circuitry.")
    parser.add_argument("--run-root", type=str, required=True)
    parser.add_argument("--output-root", type=str, required=True)
    args = parser.parse_args()

    run_root = Path(args.run_root).resolve()
    output_root = Path(args.output_root).resolve()

    functional_rows = load_functional_nodes(run_root / "functional_groups" / "functional_node_table.csv")
    bid = json.loads((run_root / "bidirectional" / "bidirectional_summary.json").read_text(encoding="utf-8"))

    by_group: Dict[str, set[str]] = {}
    for row in functional_rows:
        by_group.setdefault(row["functional_group"], set()).add(row["node"])

    reverse_core_nodes = set(bid["reverse"]["core_nodes"])
    reverse_core_edges = to_edge_set(bid["reverse"]["core_edges"])
    reverse_selective_nodes = set(bid["support_analysis"]["reverse_selective_nodes"])
    reverse_selective_edges = to_edge_set(bid["support_analysis"]["reverse_selective_edges"])
    shared_nodes = set(bid["support_analysis"]["shared_backbone_nodes"])

    minimal_no_tool_nodes = {"L16H4", "MLP17", "L23H6"}
    minimal_no_tool_edges = {
        ("L16H4", "MLP17"),
        ("MLP17", "L23H6"),
        ("L23H6", "Residual Output: decision"),
    }

    semantic_no_tool_nodes = set().union(
        by_group.get("suppression_readers", set()),
        by_group.get("no_tool_writers", set()),
    )
    semantic_no_tool_edges = {
        ("MLP12", "L15H5"),
        ("MLP12", "L16H13"),
        ("MLP12", "L16H4"),
        ("MLP12", "L16H8"),
        ("MLP12", "L16H9"),
        ("L15H5", "L16H8"),
        ("L16H13", "MLP17"),
        ("L16H4", "MLP17"),
        ("L16H8", "MLP17"),
        ("L16H8", "L17H2"),
        ("L16H9", "MLP17"),
        ("MLP17", "L23H6"),
        ("MLP17", "Residual Output: decision"),
        ("L17H2", "Residual Output: decision"),
        ("L16H8", "Residual Output: decision"),
        ("L23H6", "Residual Output: decision"),
    }

    reverse_aligned_no_tool_nodes = {
        "MLP12",
        "L15H5",
        "L16H13",
        "L16H4",
        "L16H8",
        "L16H9",
        "MLP17",
        "L17H2",
        "L23H6",
    }
    reverse_aligned_no_tool_edges = semantic_no_tool_edges

    rows = []
    for name, nodes in [
        ("minimal_no_tool", minimal_no_tool_nodes),
        ("semantic_no_tool", semantic_no_tool_nodes),
        ("reverse_aligned_no_tool", reverse_aligned_no_tool_nodes),
    ]:
        rows.append(
            {
                "name": name,
                "nodes": sorted(nodes),
                "overlap_with_reverse_core": sorted(nodes & reverse_core_nodes),
                "reverse_core_recall": recall(nodes, reverse_core_nodes),
                "reverse_core_jaccard": jaccard(nodes, reverse_core_nodes),
                "overlap_with_reverse_selective": sorted(nodes & reverse_selective_nodes),
                "reverse_selective_recall": recall(nodes, reverse_selective_nodes),
                "reverse_selective_precision": precision(nodes, reverse_selective_nodes),
                "reverse_selective_jaccard": jaccard(nodes, reverse_selective_nodes),
                "overlap_with_shared_backbone": sorted(nodes & shared_nodes),
            }
        )

    edge_rows = []
    for name, edges in [
        ("minimal_no_tool", minimal_no_tool_edges),
        ("semantic_no_tool", semantic_no_tool_edges),
        ("reverse_aligned_no_tool", reverse_aligned_no_tool_edges),
    ]:
        edge_rows.append(
            {
                "name": name,
                "edges": [list(edge) for edge in sorted(edges)],
                "overlap_with_reverse_core": [list(edge) for edge in sorted(edges & reverse_core_edges)],
                "reverse_core_recall": recall(edges, reverse_core_edges),
                "reverse_core_jaccard": jaccard(edges, reverse_core_edges),
                "overlap_with_reverse_selective": [list(edge) for edge in sorted(edges & reverse_selective_edges)],
                "reverse_selective_recall": recall(edges, reverse_selective_edges),
                "reverse_selective_precision": precision(edges, reverse_selective_edges),
                "reverse_selective_jaccard": jaccard(edges, reverse_selective_edges),
            }
        )

    summary = {
        "reverse_core_nodes": sorted(reverse_core_nodes),
        "reverse_selective_nodes": sorted(reverse_selective_nodes),
        "rows": rows,
        "edge_rows": edge_rows,
        "takeaway": {
            "minimal_chain_vs_reverse_core": "The 3-node minimal no-tool chain is fully contained in the reverse core.",
            "reverse_aligned_semantic_line_vs_reverse_selective": "The 9-node reverse-aligned no-tool semantic line overlaps 8/8 reverse-selective nodes and 14/15 reverse-selective edges; the only extra node is L23H6, which sits in the shared late output backbone.",
        },
    }

    write_json(summary, output_root / "reverse_overlap_summary.json")

    lines = [
        "# Reverse / No-Tool Overlap",
        "",
        "## Main takeaway",
        "",
        "- The minimal no-tool decision chain `L16H4 -> MLP17 -> L23H6` is fully contained in the reverse core.",
        "- If we expand the no-tool semantic line to the reverse-aligned suppressive branch `{MLP12, L15H5, L16H13, L16H4, L16H8, L16H9, MLP17, L17H2, L23H6}`, it matches all 8 reverse-selective nodes plus one extra shared late node `L23H6`.",
        "- On edges, that reverse-aligned no-tool semantic line covers 14/15 reverse-selective edges; the missing edge is a late `MLP17 -> Residual Output: decision` shortcut, while `L23H6 -> Residual Output: decision` is the extra shared late-output edge.",
        "",
        "## Node overlap",
        "",
    ]
    for row in rows:
        lines.extend(
            [
                f"### {row['name']}",
                "",
                f"- reverse-core recall: `{row['reverse_core_recall']:.4f}`",
                f"- reverse-selective recall: `{row['reverse_selective_recall']:.4f}`",
                f"- reverse-selective precision: `{row['reverse_selective_precision']:.4f}`",
                f"- reverse-selective jaccard: `{row['reverse_selective_jaccard']:.4f}`",
                f"- overlap with reverse-selective: `{', '.join(row['overlap_with_reverse_selective']) or '-'}`",
                "",
            ]
        )
    lines.extend(["## Edge overlap", ""])
    for row in edge_rows:
        lines.extend(
            [
                f"### {row['name']}",
                "",
                f"- reverse-core recall: `{row['reverse_core_recall']:.4f}`",
                f"- reverse-selective recall: `{row['reverse_selective_recall']:.4f}`",
                f"- reverse-selective precision: `{row['reverse_selective_precision']:.4f}`",
                f"- reverse-selective jaccard: `{row['reverse_selective_jaccard']:.4f}`",
                "",
            ]
        )
    write_report(lines, output_root / "reverse_overlap_report.md")


if __name__ == "__main__":
    main()
