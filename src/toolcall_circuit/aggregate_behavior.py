#!/usr/bin/env python3
"""
Cross-sample aggregation for arbitrary behavior directions.

This wraps the existing aggregation logic but lets us choose a custom output-node label,
which is necessary for the reverse no-tool run.
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
from typing import Dict, List

import numpy as np
import torch

if __package__ in {None, ""}:
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from toolcall_circuit import aggregate as base_aggregate
from toolcall_circuit.graph_utils import draw_circuit_with_output
from toolcall_circuit.paths import MODEL_PATH_DEFAULT, RESULTS_ROOT
from toolcall_circuit.single_sample import node_layer


def _adaptive_threshold(
    values: List[float],
    *,
    quantile: float,
    lower: float,
    upper: float,
    fallback: float,
) -> float:
    finite = [float(v) for v in values if np.isfinite(float(v))]
    if not finite:
        return float(fallback)
    raw = float(np.quantile(np.asarray(finite, dtype=np.float32), quantile))
    return float(min(upper, max(lower, raw)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate cross-sample behavior circuits.")
    parser.add_argument("--input-root", type=str, default=str(RESULTS_ROOT / "manual_run" / "batch"))
    parser.add_argument("--output-root", type=str, default=str(RESULTS_ROOT / "manual_run" / "aggregate_behavior"))
    parser.add_argument("--model-path", type=str, default=str(MODEL_PATH_DEFAULT))
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output-node-label", type=str, default="Residual Output: <tool_call>")
    parser.add_argument("--summary-label", type=str, default="behavior")

    parser.add_argument("--gap-min", type=float, default=0.5)
    parser.add_argument("--ap-discount", type=float, default=0.7)

    parser.add_argument("--core-node-th", type=float, default=0.50)
    parser.add_argument("--core-edge-th", type=float, default=0.35)
    parser.add_argument("--relaxed-node-th", type=float, default=0.25)
    parser.add_argument("--relaxed-edge-th", type=float, default=0.15)
    parser.add_argument("--min-nodes", type=int, default=8)
    parser.add_argument("--max-core-nodes", type=int, default=18)
    parser.add_argument("--max-relaxed-nodes", type=int, default=26)

    parser.add_argument("--cluster-sim-th", type=float, default=0.45)
    parser.add_argument("--cluster-min-size", type=int, default=8)
    parser.add_argument("--cluster-max-plots", type=int, default=8)
    parser.add_argument("--adaptive-thresholds", action="store_true")

    parser.add_argument("--replay-random", type=int, default=2)
    parser.add_argument("--skip-replay", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    base_aggregate.OUTPUT_NODE = args.output_node_label

    input_root = Path(args.input_root).resolve()
    out_root = Path(args.output_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    records = base_aggregate.load_sample_records(
        root=input_root,
        gap_min=args.gap_min,
        ap_discount=args.ap_discount,
    )
    if not records:
        raise ValueError(f"No per-sample summaries found in {input_root}")

    node_support, edge_support, node_cnt, edge_cnt, node_norm_scores, total_w = base_aggregate.aggregate_supports(records)
    n_samples = len(records)
    inferred_n_layers = max((node_layer(n) for n in node_support), default=0) + 1

    core_node_th = args.core_node_th
    core_edge_th = args.core_edge_th
    relaxed_node_th = args.relaxed_node_th
    relaxed_edge_th = args.relaxed_edge_th
    min_nodes = args.min_nodes
    max_core_nodes = args.max_core_nodes
    max_relaxed_nodes = args.max_relaxed_nodes

    if args.adaptive_thresholds:
        node_vals = list(node_support.values())
        edge_vals = list(edge_support.values())
        core_node_th = _adaptive_threshold(node_vals, quantile=0.80, lower=0.14, upper=0.50, fallback=args.core_node_th)
        relaxed_node_th = min(core_node_th, _adaptive_threshold(node_vals, quantile=0.62, lower=0.08, upper=0.25, fallback=args.relaxed_node_th))
        core_edge_th = _adaptive_threshold(edge_vals, quantile=0.82, lower=0.10, upper=0.35, fallback=args.core_edge_th)
        relaxed_edge_th = min(core_edge_th, _adaptive_threshold(edge_vals, quantile=0.65, lower=0.05, upper=0.15, fallback=args.relaxed_edge_th))
        min_nodes = max(args.min_nodes, min(12, max(8, math.ceil(inferred_n_layers * 0.28))))
        max_core_nodes = max(args.max_core_nodes, min(24, max(14, math.ceil(inferred_n_layers * 0.50))))
        max_relaxed_nodes = max(args.max_relaxed_nodes, min(32, max(18, math.ceil(inferred_n_layers * 0.65))))

    base_aggregate.write_node_table(
        out_path=out_root / "node_support.csv",
        node_support=node_support,
        node_cnt=node_cnt,
        node_norm_scores=node_norm_scores,
        n_samples=n_samples,
    )
    base_aggregate.write_edge_table(
        out_path=out_root / "edge_support.csv",
        edge_support=edge_support,
        edge_cnt=edge_cnt,
        n_samples=n_samples,
    )

    sample_rows = [
        {
            "sample_id": r.sample_id,
            "sample_rank": r.sample_rank,
            "q_index": r.legacy_index,
            "filename": r.summary.get("filename"),
            "weight": r.weight,
            "gap": base_aggregate.safe_float(r.summary.get("gap"), float("nan")),
            "detailed_ratio_vs_gap": base_aggregate.safe_float(r.summary.get("detailed_ratio_vs_gap"), float("nan")),
            "necessity_ratio_vs_gap": base_aggregate.safe_float(r.summary.get("necessity_ratio_vs_gap"), float("nan")),
            "ap_mode": str(r.summary.get("ap_mode", "full")),
        }
        for r in records
    ]
    with (out_root / "sample_weights.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "sample_id",
                "sample_rank",
                "q_index",
                "filename",
                "weight",
                "gap",
                "detailed_ratio_vs_gap",
                "necessity_ratio_vs_gap",
                "ap_mode",
            ],
        )
        writer.writeheader()
        writer.writerows(sample_rows)

    core_nodes = base_aggregate.pick_consensus_nodes(
        node_support=node_support,
        node_threshold=core_node_th,
        min_nodes=min_nodes,
        max_nodes=max_core_nodes,
    )
    core_edges = base_aggregate.pick_consensus_edges(
        nodes=core_nodes,
        edge_support=edge_support,
        node_support=node_support,
        edge_threshold=core_edge_th,
        min_edges=max(10, len(core_nodes)),
    )
    draw_circuit_with_output(
        nodes=core_nodes,
        edges=core_edges,
        out_path=out_root / "final_circuit_global_core.png",
        title=f"Global Core Circuit ({args.summary_label})",
        output_node=args.output_node_label,
    )

    relaxed_nodes = base_aggregate.pick_consensus_nodes(
        node_support=node_support,
        node_threshold=relaxed_node_th,
        min_nodes=min_nodes + 2,
        max_nodes=max_relaxed_nodes,
    )
    relaxed_edges = base_aggregate.pick_consensus_edges(
        nodes=relaxed_nodes,
        edge_support=edge_support,
        node_support=node_support,
        edge_threshold=relaxed_edge_th,
        min_edges=max(14, len(relaxed_nodes)),
    )
    draw_circuit_with_output(
        nodes=relaxed_nodes,
        edges=relaxed_edges,
        out_path=out_root / "final_circuit_global_relaxed.png",
        title=f"Global Relaxed Circuit ({args.summary_label})",
        output_node=args.output_node_label,
    )

    assignment = base_aggregate.cluster_samples(records=records, sim_threshold=args.cluster_sim_th)
    base_aggregate.write_cluster_assignments(
        out_path=out_root / "cluster_assignments.csv",
        records=records,
        assignment=assignment,
    )

    cluster_to_samples: Dict[int, List[str]] = defaultdict(list)
    for sample_id, cluster_id in assignment.items():
        cluster_to_samples[cluster_id].append(sample_id)
    cluster_sorted = sorted(cluster_to_samples.items(), key=lambda x: len(x[1]), reverse=True)

    cluster_summaries = []
    plotted = 0
    for c_id, sample_ids in cluster_sorted:
        members = [r for r in records if r.sample_id in set(sample_ids)]
        if len(members) < args.cluster_min_size:
            continue
        ns, es, _, _, _, tw = base_aggregate.aggregate_supports(members)
        c_nodes = base_aggregate.pick_consensus_nodes(
            node_support=ns,
            node_threshold=max(0.10, core_node_th - 0.08),
            min_nodes=min(min_nodes, 8),
            max_nodes=14,
        )
        c_edges = base_aggregate.pick_consensus_edges(
            nodes=c_nodes,
            edge_support=es,
            node_support=ns,
            edge_threshold=max(0.05, core_edge_th - 0.08),
            min_edges=max(8, len(c_nodes)),
        )
        cluster_summaries.append(
            {
                "cluster_id": c_id,
                "size": len(members),
                "total_weight": tw,
                "nodes": c_nodes,
                "edges": c_edges,
            }
        )
        if plotted < args.cluster_max_plots:
            draw_circuit_with_output(
                nodes=c_nodes,
                edges=c_edges,
                out_path=out_root / f"final_circuit_cluster_{c_id:02d}.png",
                title=f"Cluster {c_id} Core Circuit ({args.summary_label}, n={len(members)})",
                output_node=args.output_node_label,
            )
            plotted += 1

    should_replay = not args.skip_replay
    if should_replay and args.device.startswith("cuda") and not torch.cuda.is_available():
        should_replay = False
        replay: Dict[str, object] = {
            "ran_samples": 0,
            "note": "Skipped replay because CUDA device is unavailable in current session.",
        }
    elif should_replay:
        replay = base_aggregate.replay_global_circuit(
            records=records,
            global_nodes=core_nodes,
            model_path=args.model_path,
            device=args.device,
            n_random=args.replay_random,
            seed=args.seed,
        )
    else:
        replay = {"ran_samples": 0, "note": "Replay skipped by --skip-replay."}

    (out_root / "global_core_replay.json").write_text(json.dumps(replay, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = {
        "input_root": str(input_root),
        "summary_label": args.summary_label,
        "output_node_label": args.output_node_label,
        "n_samples": n_samples,
        "total_weight": total_w,
        "weight_params": {"gap_min": args.gap_min, "ap_discount": args.ap_discount},
        "adaptive_thresholds": bool(args.adaptive_thresholds),
        "inferred_n_layers": inferred_n_layers,
        "core_thresholds": {"node": core_node_th, "edge": core_edge_th},
        "relaxed_thresholds": {"node": relaxed_node_th, "edge": relaxed_edge_th},
        "core_nodes": core_nodes,
        "core_edges": core_edges,
        "relaxed_nodes": relaxed_nodes,
        "relaxed_edges": relaxed_edges,
        "n_clusters": len(cluster_to_samples),
        "cluster_sizes": {str(k): len(v) for k, v in cluster_to_samples.items()},
        "cluster_summaries": cluster_summaries,
        "replay_summary": {
            "ran_samples": replay.get("ran_samples"),
            "global_suff_ratio_median": replay.get("global_suff_ratio_median"),
            "global_nec_ratio_median": replay.get("global_nec_ratio_median"),
            "random_suff_ratio_mean_median": replay.get("random_suff_ratio_mean_median"),
            "global_minus_random_median": replay.get("global_minus_random_median"),
        },
        "artifacts": {
            "node_support_csv": str(out_root / "node_support.csv"),
            "edge_support_csv": str(out_root / "edge_support.csv"),
            "sample_weights_csv": str(out_root / "sample_weights.csv"),
            "cluster_assignments_csv": str(out_root / "cluster_assignments.csv"),
            "global_core_png": str(out_root / "final_circuit_global_core.png"),
            "global_relaxed_png": str(out_root / "final_circuit_global_relaxed.png"),
            "global_replay_json": str(out_root / "global_core_replay.json"),
        },
    }
    (out_root / "global_core_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[done] aggregate outputs: {out_root}")
    print(f"[done] core nodes={len(core_nodes)} edges={len(core_edges)}")
    print(f"[done] replay samples={replay.get('ran_samples', 0)}")


if __name__ == "__main__":
    main()
