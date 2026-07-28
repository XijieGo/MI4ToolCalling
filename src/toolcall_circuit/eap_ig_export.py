#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from toolcall_circuit.graph_utils import DEFAULT_OUTPUT_NODE, INPUT_NODE

PROJECT_ROOT = Path(__file__).resolve().parents[4]
EAP_SRC = PROJECT_ROOT / "experiment" / "code" / "vendor" / "EAP-IG" / "src"

if str(EAP_SRC) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(EAP_SRC))

from eap.graph import Graph  # noqa: E402


def eap_node_to_local(node_name: str, *, output_node_label: str = DEFAULT_OUTPUT_NODE) -> str:
    if node_name == "input":
        return INPUT_NODE
    if node_name == "logits":
        return output_node_label
    if node_name.startswith("m"):
        return f"MLP{int(node_name[1:])}"
    if node_name.startswith("a"):
        layer, head = node_name.split(".")
        return f"L{int(layer[1:])}H{int(head[1:])}"
    raise ValueError(f"Unknown EAP node name: {node_name}")


def clone_graph(graph: Graph, *, with_scores: bool = True, with_in_graph: bool = True) -> Graph:
    out = Graph.from_model(dict(graph.cfg), node_scores=graph.nodes_scores is not None)
    if with_scores:
        out.scores[:] = graph.scores
        if graph.nodes_scores is not None and out.nodes_scores is not None:
            out.nodes_scores[:] = graph.nodes_scores
    if with_in_graph:
        out.in_graph[:] = graph.in_graph
        out.nodes_in_graph[:] = graph.nodes_in_graph
    return out


def complement_graph(graph: Graph) -> Graph:
    out = clone_graph(graph, with_scores=True, with_in_graph=False)
    out.in_graph[:] = graph.real_edge_mask & (~graph.in_graph)
    nodes_with_outgoing = out.in_graph.any(dim=1)
    nodes_with_ingoing = (out.in_graph.any(dim=0).float() @ out.forward_to_backward.T.float()) > 0
    nodes_with_ingoing[0] = True
    out.nodes_in_graph[:] = nodes_with_outgoing & nodes_with_ingoing
    out.prune()
    return out


def selected_node_names(graph: Graph, *, output_node_label: str = DEFAULT_OUTPUT_NODE) -> List[str]:
    names = []
    for node in graph.nodes.values():
        if node.name in {"input", "logits"}:
            continue
        if node.in_graph:
            names.append(eap_node_to_local(node.name, output_node_label=output_node_label))
    return sorted(set(names), key=lambda x: (int(x[3:]) if x.startswith("MLP") else int(x[1:].split("H")[0]), x))


def selected_raw_edge_rows(graph: Graph, *, output_node_label: str = DEFAULT_OUTPUT_NODE) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for edge in graph.edges.values():
        if not edge.in_graph:
            continue
        rows.append(
            {
                "edge_name": edge.name,
                "source_eap": edge.parent.name,
                "target_eap": edge.child.name,
                "qkv": edge.qkv or "",
                "source_local": eap_node_to_local(edge.parent.name, output_node_label=output_node_label),
                "target_local": eap_node_to_local(edge.child.name, output_node_label=output_node_label),
                "score": float(edge.score.item()),
            }
        )
    rows.sort(key=lambda row: float(row["score"]), reverse=True)
    return rows


def all_real_raw_edge_rows(graph: Graph, *, output_node_label: str = DEFAULT_OUTPUT_NODE) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for edge in graph.edges.values():
        if not bool(graph.real_edge_mask[edge.matrix_index]):
            continue
        rows.append(
            {
                "edge_name": edge.name,
                "source_eap": edge.parent.name,
                "target_eap": edge.child.name,
                "qkv": edge.qkv or "",
                "source_local": eap_node_to_local(edge.parent.name, output_node_label=output_node_label),
                "target_local": eap_node_to_local(edge.child.name, output_node_label=output_node_label),
                "score": float(edge.score.item()),
            }
        )
    rows.sort(key=lambda row: float(row["score"]), reverse=True)
    return rows


def top_raw_edge_rows(
    graph: Graph,
    *,
    output_node_label: str = DEFAULT_OUTPUT_NODE,
    topk: int = 256,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for edge in graph.edges.values():
        if not bool(graph.real_edge_mask[edge.matrix_index]):
            continue
        rows.append(
            {
                "edge_name": edge.name,
                "source_eap": edge.parent.name,
                "target_eap": edge.child.name,
                "qkv": edge.qkv or "",
                "source_local": eap_node_to_local(edge.parent.name, output_node_label=output_node_label),
                "target_local": eap_node_to_local(edge.child.name, output_node_label=output_node_label),
                "score": float(edge.score.item()),
            }
        )
    rows.sort(key=lambda row: float(row["score"]), reverse=True)
    return rows[: max(1, int(topk))]


def fold_selected_edges(raw_rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    buckets: Dict[Tuple[str, str], Dict[str, object]] = {}
    for row in raw_rows:
        key = (str(row["source_local"]), str(row["target_local"]))
        bucket = buckets.setdefault(
            key,
            {
                "source": key[0],
                "target": key[1],
                "qkv_channels": [],
                "qkv_count": 0,
                "score_sum": 0.0,
                "score_max": float("-inf"),
                "score_mean": 0.0,
                "raw_edge_count": 0,
                "raw_edges": [],
            },
        )
        qkv = str(row.get("qkv", ""))
        if qkv and qkv not in bucket["qkv_channels"]:
            bucket["qkv_channels"].append(qkv)
        score = float(row["score"])
        bucket["score_sum"] += score
        bucket["score_max"] = max(float(bucket["score_max"]), score)
        bucket["raw_edge_count"] += 1
        bucket["raw_edges"].append(str(row["edge_name"]))
    out = []
    for bucket in buckets.values():
        bucket["qkv_channels"] = sorted(bucket["qkv_channels"])
        bucket["qkv_count"] = len(bucket["qkv_channels"])
        bucket["score_mean"] = float(bucket["score_sum"]) / max(1, int(bucket["raw_edge_count"]))
        out.append(bucket)
    out.sort(key=lambda row: (float(row["score_max"]), float(row["score_sum"])), reverse=True)
    return out


def folded_edge_pairs(folded_rows: Sequence[Dict[str, object]]) -> List[Tuple[str, str]]:
    return [(str(row["source"]), str(row["target"])) for row in folded_rows]


def aggregate_node_scores(
    graph: Graph,
    *,
    output_node_label: str = DEFAULT_OUTPUT_NODE,
) -> List[Dict[str, object]]:
    stats: Dict[str, Dict[str, float]] = defaultdict(
        lambda: {
            "outgoing_pos_sum": 0.0,
            "incoming_pos_sum": 0.0,
            "incident_abs_sum": 0.0,
            "selected_edge_count": 0.0,
        }
    )

    for edge in graph.edges.values():
        if not bool(graph.real_edge_mask[edge.matrix_index]):
            continue
        score = float(edge.score.item())
        src = eap_node_to_local(edge.parent.name, output_node_label=output_node_label)
        dst = eap_node_to_local(edge.child.name, output_node_label=output_node_label)
        if src != INPUT_NODE:
            stats[src]["outgoing_pos_sum"] += max(0.0, score)
            stats[src]["incident_abs_sum"] += abs(score)
            if edge.in_graph:
                stats[src]["selected_edge_count"] += 1.0
        if dst != output_node_label:
            stats[dst]["incoming_pos_sum"] += max(0.0, score)
            stats[dst]["incident_abs_sum"] += abs(score)
            if edge.in_graph:
                stats[dst]["selected_edge_count"] += 1.0

    rows: List[Dict[str, object]] = []
    for node, node_stats in stats.items():
        outgoing = float(node_stats["outgoing_pos_sum"])
        incoming = float(node_stats["incoming_pos_sum"])
        incident_abs = float(node_stats["incident_abs_sum"])
        score = max(outgoing, incoming, 0.5 * (outgoing + incoming))
        rows.append(
            {
                "name": node,
                "score": score,
                "outgoing_pos_sum": outgoing,
                "incoming_pos_sum": incoming,
                "incident_abs_sum": incident_abs,
                "selected_edge_count": int(node_stats["selected_edge_count"]),
            }
        )
    rows.sort(key=lambda row: (float(row["score"]), float(row["incident_abs_sum"])), reverse=True)
    return rows


def write_csv(rows: Sequence[Dict[str, object]], out_path: Path) -> None:
    if not rows:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(data: Dict[str, object] | List[Dict[str, object]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def finite_float(value: object, default: float = float("nan")) -> float:
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return default


__all__ = [
    "aggregate_node_scores",
    "all_real_raw_edge_rows",
    "clone_graph",
    "complement_graph",
    "finite_float",
    "fold_selected_edges",
    "folded_edge_pairs",
    "selected_node_names",
    "selected_raw_edge_rows",
    "top_raw_edge_rows",
    "write_csv",
    "write_json",
]
