#!/usr/bin/env python3
"""
Track how the reverse core stabilizes as more reverse samples are added.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

if __package__ in {None, ""}:
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from toolcall_circuit import aggregate as base_aggregate
from toolcall_circuit.dataset import load_summary_records
from toolcall_circuit.paths import RESULTS_ROOT


def normalize_edge(edge: Sequence[str]) -> Tuple[str, str]:
    src, dst = str(edge[0]), str(edge[1])
    if dst.startswith("Residual Output:"):
        return src, "Residual Output: decision"
    return src, dst


def jaccard(a: Sequence[str], b: Sequence[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def parse_checkpoints(raw: str, max_n: int) -> List[int]:
    out: List[int] = []
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        out.append(int(chunk))
    out = [n for n in out if n > 0]
    if max_n not in out:
        out.append(max_n)
    return sorted(set(min(max_n, n) for n in out))


def load_prefix_records(input_root: Path, n: int, gap_min: float, ap_discount: float):
    records = base_aggregate.load_sample_records(input_root, gap_min=gap_min, ap_discount=ap_discount)
    return records[:n]


def main() -> None:
    parser = argparse.ArgumentParser(description="Track reverse-core stability as sample count grows.")
    parser.add_argument("--forward-aggregate-summary", type=str, default=str(RESULTS_ROOT / "11-21-37" / "aggregate" / "global_core_summary.json"))
    parser.add_argument("--reverse-batch-root", type=str, required=True)
    parser.add_argument("--output", type=str, default=str(RESULTS_ROOT / "manual_run" / "reverse_stability.json"))
    parser.add_argument("--checkpoints", type=str, default="25,50,100,200,400,800,1189")
    parser.add_argument("--gap-min", type=float, default=0.5)
    parser.add_argument("--ap-discount", type=float, default=0.7)
    parser.add_argument("--core-node-th", type=float, default=0.50)
    parser.add_argument("--core-edge-th", type=float, default=0.35)
    parser.add_argument("--min-nodes", type=int, default=8)
    parser.add_argument("--max-core-nodes", type=int, default=18)
    args = parser.parse_args()

    reverse_root = Path(args.reverse_batch_root).resolve()
    out_path = Path(args.output).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    forward_summary = json.loads(Path(args.forward_aggregate_summary).resolve().read_text(encoding="utf-8"))
    forward_core_nodes = list(forward_summary.get("core_nodes", []))
    forward_core_edges = [normalize_edge(edge) for edge in forward_summary.get("core_edges", [])]

    reverse_records_all = load_summary_records(reverse_root)
    if not reverse_records_all:
        raise ValueError(f"No reverse summaries found under {reverse_root}")

    checkpoints = parse_checkpoints(args.checkpoints, len(reverse_records_all))
    rows: List[Dict[str, object]] = []
    prev_nodes: List[str] = []
    prev_edges: List[Tuple[str, str]] = []

    for n in checkpoints:
        records = load_prefix_records(reverse_root, n, gap_min=args.gap_min, ap_discount=args.ap_discount)
        node_support, edge_support, _, _, _, _ = base_aggregate.aggregate_supports(records)
        base_aggregate.OUTPUT_NODE = "Residual Output: no_tool"
        core_nodes = base_aggregate.pick_consensus_nodes(
            node_support=node_support,
            node_threshold=args.core_node_th,
            min_nodes=args.min_nodes,
            max_nodes=args.max_core_nodes,
        )
        core_edges = base_aggregate.pick_consensus_edges(
            nodes=core_nodes,
            edge_support=edge_support,
            node_support=node_support,
            edge_threshold=args.core_edge_th,
            min_edges=max(10, len(core_nodes)),
        )
        norm_edges = [normalize_edge(edge) for edge in core_edges]
        row = {
            "n_reverse_samples": n,
            "core_node_count": len(core_nodes),
            "core_edge_count": len(core_edges),
            "node_jaccard_vs_forward": jaccard(core_nodes, forward_core_nodes),
            "edge_jaccard_vs_forward": jaccard([str(e) for e in norm_edges], [str(e) for e in forward_core_edges]),
            "node_jaccard_vs_prev": jaccard(core_nodes, prev_nodes) if prev_nodes else None,
            "edge_jaccard_vs_prev": jaccard([str(e) for e in norm_edges], [str(e) for e in prev_edges]) if prev_edges else None,
            "core_nodes": core_nodes,
            "core_edges": core_edges,
        }
        rows.append(row)
        prev_nodes = core_nodes
        prev_edges = norm_edges

    report = {
        "forward_core_nodes": forward_core_nodes,
        "forward_core_edges": forward_summary.get("core_edges", []),
        "checkpoints": rows,
    }
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    csv_path = out_path.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "n_reverse_samples",
                "core_node_count",
                "core_edge_count",
                "node_jaccard_vs_forward",
                "edge_jaccard_vs_forward",
                "node_jaccard_vs_prev",
                "edge_jaccard_vs_prev",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in writer.fieldnames})

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
