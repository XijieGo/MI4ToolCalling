#!/usr/bin/env python3
"""
Bidirectional analysis for tool-call promotion vs no-tool promotion circuits.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from toolcall_circuit import aggregate as base_aggregate
from toolcall_circuit.dataset import load_summary_records
from toolcall_circuit.graph_utils import draw_circuit_with_output
from toolcall_circuit.paths import RESULTS_ROOT

NORMALIZED_OUTPUT_NODE = "Residual Output: decision"
EPS = 1e-8


def normalize_edge(edge: Tuple[str, str]) -> Tuple[str, str]:
    src, dst = edge
    if dst.startswith("Residual Output:"):
        return (src, NORMALIZED_OUTPUT_NODE)
    return (src, dst)


def jaccard(a: Sequence[str], b: Sequence[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def finite_mean(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if isinstance(v, (int, float)) and math.isfinite(float(v))]
    return float(np.mean(vals)) if vals else float("nan")


def finite_median(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if isinstance(v, (int, float)) and math.isfinite(float(v))]
    return float(np.median(vals)) if vals else float("nan")


def load_weighted_supports(batch_root: Path, gap_min: float, ap_discount: float):
    records = base_aggregate.load_sample_records(batch_root, gap_min=gap_min, ap_discount=ap_discount)
    node_support, edge_support, node_cnt, edge_cnt, node_norm_scores, total_w = base_aggregate.aggregate_supports(records)
    normalized_edge_support = {normalize_edge(edge): support for edge, support in edge_support.items()}
    normalized_edge_cnt = {normalize_edge(edge): cnt for edge, cnt in edge_cnt.items()}
    merged_edge_support: Dict[Tuple[str, str], float] = {}
    merged_edge_cnt: Dict[Tuple[str, str], float] = {}
    for edge, support in normalized_edge_support.items():
        merged_edge_support[edge] = max(support, merged_edge_support.get(edge, 0.0))
    for edge, cnt in normalized_edge_cnt.items():
        merged_edge_cnt[edge] = max(cnt, merged_edge_cnt.get(edge, 0.0))
    return records, node_support, merged_edge_support, node_cnt, merged_edge_cnt, node_norm_scores, total_w


def build_support_rows(
    forward_node_support: Dict[str, float],
    reverse_node_support: Dict[str, float],
    forward_node_scores: Dict[str, List[float]],
    reverse_node_scores: Dict[str, List[float]],
) -> List[Dict[str, object]]:
    nodes = sorted(set(forward_node_support) | set(reverse_node_support), key=lambda n: (base_aggregate.node_layer(n), n))
    rows: List[Dict[str, object]] = []
    for node in nodes:
        fwd = float(forward_node_support.get(node, 0.0))
        rev = float(reverse_node_support.get(node, 0.0))
        total = fwd + rev
        rows.append(
            {
                "node": node,
                "layer": base_aggregate.node_layer(node),
                "forward_support": fwd,
                "reverse_support": rev,
                "shared_support_min": min(fwd, rev),
                "union_support_max": max(fwd, rev),
                "support_delta_reverse_minus_forward": rev - fwd,
                "direction_balance": (rev - fwd) / max(EPS, total),
                "forward_score_norm_median": finite_median(forward_node_scores.get(node, [])),
                "reverse_score_norm_median": finite_median(reverse_node_scores.get(node, [])),
            }
        )
    rows.sort(key=lambda row: (row["shared_support_min"], row["union_support_max"]), reverse=True)
    return rows


def build_edge_rows(
    forward_edge_support: Dict[Tuple[str, str], float],
    reverse_edge_support: Dict[Tuple[str, str], float],
) -> List[Dict[str, object]]:
    edges = sorted(set(forward_edge_support) | set(reverse_edge_support), key=lambda e: (e[0], e[1]))
    rows: List[Dict[str, object]] = []
    for edge in edges:
        fwd = float(forward_edge_support.get(edge, 0.0))
        rev = float(reverse_edge_support.get(edge, 0.0))
        total = fwd + rev
        rows.append(
            {
                "source": edge[0],
                "target": edge[1],
                "forward_support": fwd,
                "reverse_support": rev,
                "shared_support_min": min(fwd, rev),
                "union_support_max": max(fwd, rev),
                "support_delta_reverse_minus_forward": rev - fwd,
                "direction_balance": (rev - fwd) / max(EPS, total),
            }
        )
    rows.sort(key=lambda row: (row["shared_support_min"], row["union_support_max"]), reverse=True)
    return rows


def write_csv(rows: Sequence[Dict[str, object]], out_path: Path) -> None:
    if not rows:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_overlay_circuit(
    *,
    node_rows: Sequence[Dict[str, object]],
    edge_rows: Sequence[Dict[str, object]],
    mode: str,
    node_threshold: float,
    edge_threshold: float,
    max_nodes: int,
) -> Tuple[List[str], List[Tuple[str, str]]]:
    if mode == "shared":
        node_support = {str(r["node"]): float(r["shared_support_min"]) for r in node_rows}
        edge_support = {(str(r["source"]), str(r["target"])): float(r["shared_support_min"]) for r in edge_rows}
    elif mode == "union":
        node_support = {str(r["node"]): float(r["union_support_max"]) for r in node_rows}
        edge_support = {(str(r["source"]), str(r["target"])): float(r["union_support_max"]) for r in edge_rows}
    elif mode == "forward":
        node_support = {
            str(r["node"]): max(0.0, float(r["forward_support"]) - float(r["reverse_support"]))
            for r in node_rows
        }
        edge_support = {
            (str(r["source"]), str(r["target"])): max(0.0, float(r["forward_support"]) - float(r["reverse_support"]))
            for r in edge_rows
        }
    elif mode == "reverse":
        node_support = {
            str(r["node"]): max(0.0, float(r["reverse_support"]) - float(r["forward_support"]))
            for r in node_rows
        }
        edge_support = {
            (str(r["source"]), str(r["target"])): max(0.0, float(r["reverse_support"]) - float(r["forward_support"]))
            for r in edge_rows
        }
    else:
        raise ValueError(f"Unknown overlay mode: {mode}")

    base_aggregate.OUTPUT_NODE = NORMALIZED_OUTPUT_NODE
    nodes = base_aggregate.pick_consensus_nodes(
        node_support=node_support,
        node_threshold=node_threshold,
        min_nodes=min(8, max_nodes),
        max_nodes=max_nodes,
    )
    edges = base_aggregate.pick_consensus_edges(
        nodes=nodes,
        edge_support=edge_support,
        node_support=node_support,
        edge_threshold=edge_threshold,
        min_edges=max(8, len(nodes)),
    )
    return nodes, edges


def per_sample_overlap_rows(forward_root: Path, reverse_root: Path) -> List[Dict[str, object]]:
    fwd = {record.sample_id: record.summary for record in load_summary_records(forward_root)}
    rev = {record.sample_id: record.summary for record in load_summary_records(reverse_root)}
    rows: List[Dict[str, object]] = []
    for sample_id in sorted(set(fwd) & set(rev)):
        f_nodes = list(fwd[sample_id].get("detailed_nodes", []))
        r_nodes = list(rev[sample_id].get("detailed_nodes", []))
        inter = sorted(set(f_nodes) & set(r_nodes))
        union = sorted(set(f_nodes) | set(r_nodes))
        rows.append(
            {
                "sample_id": sample_id,
                "sample_rank": fwd[sample_id].get("sample_rank"),
                "forward_node_count": len(f_nodes),
                "reverse_node_count": len(r_nodes),
                "shared_node_count": len(inter),
                "union_node_count": len(union),
                "node_jaccard": jaccard(f_nodes, r_nodes),
                "forward_target": fwd[sample_id].get("target_token_str"),
                "reverse_target": rev[sample_id].get("target_token_str"),
            }
        )
    rows.sort(key=lambda row: ((row["sample_rank"] or 10**9), row["sample_id"]))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze tool-call circuits bidirectionally.")
    parser.add_argument("--forward-batch-root", type=str, default=str(RESULTS_ROOT / "11-21-37" / "batch"))
    parser.add_argument("--forward-aggregate-summary", type=str, default=str(RESULTS_ROOT / "11-21-37" / "aggregate" / "global_core_summary.json"))
    parser.add_argument("--reverse-batch-root", type=str, required=True)
    parser.add_argument("--reverse-aggregate-summary", type=str, required=True)
    parser.add_argument("--output-root", type=str, default=str(RESULTS_ROOT / "manual_run" / "bidirectional"))
    parser.add_argument("--gap-min", type=float, default=0.5)
    parser.add_argument("--ap-discount", type=float, default=0.7)
    parser.add_argument("--shared-node-th", type=float, default=0.20)
    parser.add_argument("--shared-edge-th", type=float, default=0.10)
    parser.add_argument("--direction-node-th", type=float, default=0.15)
    parser.add_argument("--direction-edge-th", type=float, default=0.08)
    args = parser.parse_args()

    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    forward_batch_root = Path(args.forward_batch_root).resolve()
    reverse_batch_root = Path(args.reverse_batch_root).resolve()
    forward_summary = json.loads(Path(args.forward_aggregate_summary).resolve().read_text(encoding="utf-8"))
    reverse_summary = json.loads(Path(args.reverse_aggregate_summary).resolve().read_text(encoding="utf-8"))

    (
        forward_records,
        forward_node_support,
        forward_edge_support,
        _forward_node_cnt,
        _forward_edge_cnt,
        forward_node_scores,
        _forward_total_w,
    ) = load_weighted_supports(forward_batch_root, gap_min=args.gap_min, ap_discount=args.ap_discount)
    (
        reverse_records,
        reverse_node_support,
        reverse_edge_support,
        _reverse_node_cnt,
        _reverse_edge_cnt,
        reverse_node_scores,
        _reverse_total_w,
    ) = load_weighted_supports(reverse_batch_root, gap_min=args.gap_min, ap_discount=args.ap_discount)

    node_rows = build_support_rows(forward_node_support, reverse_node_support, forward_node_scores, reverse_node_scores)
    edge_rows = build_edge_rows(forward_edge_support, reverse_edge_support)
    write_csv(node_rows, output_root / "node_bidirectional_support.csv")
    write_csv(edge_rows, output_root / "edge_bidirectional_support.csv")

    overlap_rows = per_sample_overlap_rows(forward_batch_root, reverse_batch_root)
    write_csv(overlap_rows, output_root / "per_sample_overlap.csv")

    shared_nodes, shared_edges = build_overlay_circuit(
        node_rows=node_rows,
        edge_rows=edge_rows,
        mode="shared",
        node_threshold=args.shared_node_th,
        edge_threshold=args.shared_edge_th,
        max_nodes=16,
    )
    union_nodes, union_edges = build_overlay_circuit(
        node_rows=node_rows,
        edge_rows=edge_rows,
        mode="union",
        node_threshold=max(args.shared_node_th, 0.25),
        edge_threshold=max(args.shared_edge_th, 0.12),
        max_nodes=24,
    )
    forward_nodes, forward_edges = build_overlay_circuit(
        node_rows=node_rows,
        edge_rows=edge_rows,
        mode="forward",
        node_threshold=args.direction_node_th,
        edge_threshold=args.direction_edge_th,
        max_nodes=14,
    )
    reverse_nodes, reverse_edges = build_overlay_circuit(
        node_rows=node_rows,
        edge_rows=edge_rows,
        mode="reverse",
        node_threshold=args.direction_node_th,
        edge_threshold=args.direction_edge_th,
        max_nodes=14,
    )

    draw_circuit_with_output(
        nodes=shared_nodes,
        edges=shared_edges,
        out_path=output_root / "shared_backbone.png",
        title="Shared Backbone Circuit",
        output_node=NORMALIZED_OUTPUT_NODE,
    )
    draw_circuit_with_output(
        nodes=union_nodes,
        edges=union_edges,
        out_path=output_root / "union_circuit.png",
        title="Union Circuit",
        output_node=NORMALIZED_OUTPUT_NODE,
    )
    draw_circuit_with_output(
        nodes=forward_nodes,
        edges=forward_edges,
        out_path=output_root / "forward_selective_circuit.png",
        title="Tool-call Selective Circuit",
        output_node=NORMALIZED_OUTPUT_NODE,
    )
    draw_circuit_with_output(
        nodes=reverse_nodes,
        edges=reverse_edges,
        out_path=output_root / "reverse_selective_circuit.png",
        title="No-tool Selective Circuit",
        output_node=NORMALIZED_OUTPUT_NODE,
    )

    reverse_target_counter = Counter()
    for record in reverse_records:
        reverse_target_counter[str(record.summary.get("target_token_str", ""))] += 1

    summary = {
        "forward": {
            "n_samples": len(forward_records),
            "core_nodes": forward_summary.get("core_nodes", []),
            "core_edges": forward_summary.get("core_edges", []),
        },
        "reverse": {
            "n_samples": len(reverse_records),
            "core_nodes": reverse_summary.get("core_nodes", []),
            "core_edges": reverse_summary.get("core_edges", []),
            "reverse_target_token_distribution": dict(reverse_target_counter.most_common()),
        },
        "set_analysis": {
            "core_node_jaccard": jaccard(forward_summary.get("core_nodes", []), reverse_summary.get("core_nodes", [])),
            "core_edge_jaccard": jaccard(
                [str(normalize_edge(tuple(edge))) for edge in forward_summary.get("core_edges", [])],
                [str(normalize_edge(tuple(edge))) for edge in reverse_summary.get("core_edges", [])],
            ),
            "core_node_intersection": sorted(set(forward_summary.get("core_nodes", [])) & set(reverse_summary.get("core_nodes", []))),
            "core_node_forward_only": sorted(set(forward_summary.get("core_nodes", [])) - set(reverse_summary.get("core_nodes", []))),
            "core_node_reverse_only": sorted(set(reverse_summary.get("core_nodes", [])) - set(forward_summary.get("core_nodes", []))),
        },
        "support_analysis": {
            "top_shared_nodes": [row["node"] for row in sorted(node_rows, key=lambda r: float(r["shared_support_min"]), reverse=True)[:12]],
            "top_forward_selective_nodes": [
                row["node"]
                for row in sorted(node_rows, key=lambda r: float(r["forward_support"]) - float(r["reverse_support"]), reverse=True)[:12]
            ],
            "top_reverse_selective_nodes": [
                row["node"]
                for row in sorted(node_rows, key=lambda r: float(r["reverse_support"]) - float(r["forward_support"]), reverse=True)[:12]
            ],
            "top_shared_edges": [
                [row["source"], row["target"]]
                for row in sorted(edge_rows, key=lambda r: float(r["shared_support_min"]), reverse=True)[:16]
            ],
            "top_forward_selective_edges": [
                [row["source"], row["target"]]
                for row in sorted(edge_rows, key=lambda r: float(r["forward_support"]) - float(r["reverse_support"]), reverse=True)[:16]
            ],
            "top_reverse_selective_edges": [
                [row["source"], row["target"]]
                for row in sorted(edge_rows, key=lambda r: float(r["reverse_support"]) - float(r["forward_support"]), reverse=True)[:16]
            ],
            "shared_backbone_nodes": shared_nodes,
            "shared_backbone_edges": shared_edges,
            "union_nodes": union_nodes,
            "union_edges": union_edges,
            "forward_selective_nodes": forward_nodes,
            "forward_selective_edges": forward_edges,
            "reverse_selective_nodes": reverse_nodes,
            "reverse_selective_edges": reverse_edges,
            "overlay_sizes": {
                "shared_nodes": len(shared_nodes),
                "shared_edges": len(shared_edges),
                "union_nodes": len(union_nodes),
                "union_edges": len(union_edges),
                "forward_selective_nodes": len(forward_nodes),
                "forward_selective_edges": len(forward_edges),
                "reverse_selective_nodes": len(reverse_nodes),
                "reverse_selective_edges": len(reverse_edges),
            },
        },
        "per_sample_overlap": {
            "n_shared_samples": len(overlap_rows),
            "node_jaccard_mean": finite_mean(row["node_jaccard"] for row in overlap_rows),
            "node_jaccard_median": finite_median(row["node_jaccard"] for row in overlap_rows),
            "shared_node_count_mean": finite_mean(row["shared_node_count"] for row in overlap_rows),
            "union_node_count_mean": finite_mean(row["union_node_count"] for row in overlap_rows),
            "forward_node_count_mean": finite_mean(row["forward_node_count"] for row in overlap_rows),
            "reverse_node_count_mean": finite_mean(row["reverse_node_count"] for row in overlap_rows),
        },
        "artifacts": {
            "node_bidirectional_support_csv": str(output_root / "node_bidirectional_support.csv"),
            "edge_bidirectional_support_csv": str(output_root / "edge_bidirectional_support.csv"),
            "per_sample_overlap_csv": str(output_root / "per_sample_overlap.csv"),
            "shared_backbone_png": str(output_root / "shared_backbone.png"),
            "union_circuit_png": str(output_root / "union_circuit.png"),
            "forward_selective_png": str(output_root / "forward_selective_circuit.png"),
            "reverse_selective_png": str(output_root / "reverse_selective_circuit.png"),
        },
    }
    (output_root / "bidirectional_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
