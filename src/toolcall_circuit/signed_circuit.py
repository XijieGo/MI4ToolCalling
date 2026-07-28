#!/usr/bin/env python3
"""
Build a final signed circuit view from bidirectional results.

The signed circuit is decomposed into:

- symmetric_backbone: shared backbone nodes not captured by directional selective sets
- tool_bias_backbone: shared-backbone nodes also selected by the tool-call direction
- no_tool_bias_backbone: shared-backbone nodes also selected by the no-tool direction
- tool_tail: forward-selective nodes outside the shared backbone
- no_tool_tail: reverse-selective nodes outside the shared backbone
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch

if __package__ in {None, ""}:
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from toolcall_circuit.graph_utils import INPUT_NODE, apply_plot_style, node_layer


GROUP_META = {
    "symmetric_backbone": {
        "label": "Symmetric Backbone",
        "color": "#d7d7d7",
        "description": "shared reversible backbone and output-integrating writers",
    },
    "tool_bias_backbone": {
        "label": "Tool-Bias Backbone",
        "color": "#d88c4a",
        "description": "shared backbone nodes tilted toward the tool-call endpoint",
    },
    "no_tool_bias_backbone": {
        "label": "No-Tool-Bias Backbone",
        "color": "#4e87b5",
        "description": "shared backbone nodes tilted toward the no-tool endpoint",
    },
    "tool_tail": {
        "label": "Tool Tail",
        "color": "#e6b27a",
        "description": "tool-mode auxiliary tail outside the shared backbone",
    },
    "no_tool_tail": {
        "label": "No-Tool Tail",
        "color": "#8ab6d6",
        "description": "no-tool auxiliary tail outside the shared backbone",
    },
}

EDGE_SHARED = "#7b7b7b"
EDGE_TOOL = "#b45f06"
EDGE_NO_TOOL = "#2f6f9f"


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


def load_head_delta_lookup(path: Path) -> Dict[str, Dict[str, float]]:
    rows = read_csv_rows(path)
    out: Dict[str, Dict[str, float]] = {}
    for row in rows:
        head = str(row["head"])
        out.setdefault(head, {})[str(row["set"])] = float(row["delta_median_no_minus_tool"])
    return out


def load_forward_role_lookup(path: Path) -> Dict[str, str]:
    if not path.exists():
        return {}
    rows = read_csv_rows(path)
    return {str(r["node"]): str(r.get("role", "")) for r in rows}


def derive_signed_groups(bidi_summary: Dict[str, object]) -> Dict[str, List[str]]:
    support = bidi_summary.get("support_analysis", {})
    shared = set(map(str, support.get("shared_backbone_nodes", [])))
    forward = set(map(str, support.get("forward_selective_nodes", [])))
    reverse = set(map(str, support.get("reverse_selective_nodes", [])))

    dual = sorted((shared & forward & reverse), key=lambda n: (node_layer(n), n))
    if dual:
        # In the current run this is empty, but keep the decomposition stable if it appears later.
        shared = shared - set(dual)
        forward = forward - set(dual)
        reverse = reverse - set(dual)

    groups = {
        "symmetric_backbone": sorted(shared - forward - reverse, key=lambda n: (node_layer(n), n)),
        "tool_bias_backbone": sorted((shared & forward) - reverse, key=lambda n: (node_layer(n), n)),
        "no_tool_bias_backbone": sorted((shared & reverse) - forward, key=lambda n: (node_layer(n), n)),
        "tool_tail": sorted(forward - shared, key=lambda n: (node_layer(n), n)),
        "no_tool_tail": sorted(reverse - shared, key=lambda n: (node_layer(n), n)),
    }
    if dual:
        groups["dual_bias_backbone"] = dual
    return {k: v for k, v in groups.items() if v}


def semantic_hint_for_node(
    node: str,
    group_key: str,
    head_delta_lookup: Dict[str, Dict[str, float]],
    forward_role_lookup: Dict[str, str],
) -> str:
    if node.startswith("MLP"):
        if group_key == "symmetric_backbone":
            return "shared writer MLP"
        if group_key == "tool_bias_backbone":
            return "tool-biased writer MLP"
        if group_key == "no_tool_bias_backbone":
            return "no-tool-biased writer MLP"
        if group_key == "tool_tail":
            return "tool-tail writer MLP"
        if group_key == "no_tool_tail":
            return "no-tool-tail writer MLP"
        return "writer MLP"

    deltas = head_delta_lookup.get(node, {})
    user = float(deltas.get("user_block", 0.0))
    tools = float(deltas.get("tools_block", 0.0))
    tags = float(deltas.get("tool_call_tags", 0.0))
    prefix = float(deltas.get("prefix_16", 0.0))

    forward_role = forward_role_lookup.get(node, "")
    if "Tool-Tag Reader" in forward_role:
        return "tool-tag reader"
    if "Query Reader" in forward_role:
        return "query reader"

    if user >= 0.05 and tools <= -0.02:
        return "user-content reader"
    if tags <= -0.1 or prefix >= 0.08:
        return "format/prefix router"
    if tools >= 0.04:
        return "schema reader"
    if group_key == "symmetric_backbone":
        return "shared router"
    if group_key == "tool_bias_backbone":
        return "tool-biased router"
    if group_key == "no_tool_bias_backbone":
        return "no-tool-biased router"
    if group_key == "tool_tail":
        return "tool-tail router"
    if group_key == "no_tool_tail":
        return "no-tool-tail router"
    return "router"


def edge_sign_from_balance(direction_balance: float, threshold: float = 0.08) -> str:
    if direction_balance <= -threshold:
        return "tool_bias"
    if direction_balance >= threshold:
        return "no_tool_bias"
    return "shared"


def draw_signed_circuit(
    *,
    nodes: Sequence[str],
    edges: Sequence[Dict[str, object]],
    node_group: Dict[str, str],
    out_path: Path,
    title: str,
    output_node: str = "Residual Output: decision",
) -> None:
    all_nodes = [INPUT_NODE] + list(nodes) + [output_node]
    min_l = min(node_layer(n) for n in nodes) if nodes else 0
    max_l = max(node_layer(n) for n in nodes) if nodes else 1
    layer_map = {INPUT_NODE: min_l - 2, output_node: max_l + 2}
    for n in nodes:
        layer_map[n] = node_layer(n)

    by_layer: Dict[int, List[str]] = {}
    for n in all_nodes:
        by_layer.setdefault(layer_map[n], []).append(n)

    pos: Dict[str, Tuple[float, float]] = {}
    group_order = {
        "no_tool_tail": -2,
        "no_tool_bias_backbone": -1,
        "symmetric_backbone": 0,
        "tool_bias_backbone": 1,
        "tool_tail": 2,
    }
    for layer in sorted(by_layer):
        group = sorted(
            by_layer[layer],
            key=lambda n: (
                0 if n in {INPUT_NODE, output_node} else group_order.get(node_group.get(n, ""), 0),
                0 if n.startswith("MLP") else 1,
                n,
            ),
        )
        k = len(group)
        span = 2.0
        xs = [0.0] if k == 1 else np.linspace(-span * (k - 1) / 2, span * (k - 1) / 2, k)
        for x, n in zip(xs, group):
            pos[n] = (float(x), float(layer))

    apply_plot_style()
    fig, ax = plt.subplots(figsize=(11.5, 12.0), constrained_layout=True)
    ax.set_title(title)
    ax.axis("off")

    for edge in edges:
        src = str(edge["source"])
        dst = str(edge["target"])
        x1, y1 = pos[src]
        x2, y2 = pos[dst]
        sign = str(edge["sign"])
        if sign == "tool_bias":
            color = EDGE_TOOL
        elif sign == "no_tool_bias":
            color = EDGE_NO_TOOL
        else:
            color = EDGE_SHARED
        rad = 0.08 if abs(x2 - x1) > 0.2 else 0.0
        width = 1.0 + 2.4 * float(edge.get("union_support_max", 0.0))
        arrow = FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle="->",
            mutation_scale=14,
            linewidth=width,
            color=color,
            alpha=0.9,
            connectionstyle=f"arc3,rad={rad}",
            zorder=1,
        )
        ax.add_patch(arrow)

    for n in all_nodes:
        x, y = pos[n]
        if n in {INPUT_NODE, output_node}:
            fc = "#cfe0f2"
            ec = "#cfe0f2"
            size = 260
        else:
            meta = GROUP_META[node_group[n]]
            fc = meta["color"]
            ec = "#2c2c2c"
            size = 240
        ax.scatter([x], [y], s=size, c=fc, edgecolors=ec, linewidths=1.5, zorder=3)
        ax.text(
            x + 0.30,
            y,
            n,
            va="center",
            ha="left",
            fontsize=11,
            zorder=4,
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.18, "alpha": 0.85},
        )

    legend_x = 5.8
    legend_y = max(y for _, y in pos.values()) + 0.2
    for i, key in enumerate(["tool_tail", "tool_bias_backbone", "symmetric_backbone", "no_tool_bias_backbone", "no_tool_tail"]):
        if key not in GROUP_META or key not in set(node_group.values()):
            continue
        meta = GROUP_META[key]
        y = legend_y - 0.55 * i
        ax.scatter([legend_x], [y], s=200, c=meta["color"], edgecolors="#2c2c2c", linewidths=1.4, zorder=5)
        ax.text(legend_x + 0.35, y, meta["label"], va="center", ha="left", fontsize=10)

    ys = [p[1] for p in pos.values()]
    ax.set_ylim(min(ys) - 1.0, max(ys) + 1.6)
    ax.set_xlim(-7.5, 9.2)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=280, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the final signed circuit view.")
    parser.add_argument("--bidirectional-summary", type=str, required=True)
    parser.add_argument("--node-support-csv", type=str, required=True)
    parser.add_argument("--edge-support-csv", type=str, required=True)
    parser.add_argument("--head-read-csv", type=str, required=True)
    parser.add_argument("--forward-node-roles-csv", type=str, default="")
    parser.add_argument("--output-root", type=str, required=True)
    args = parser.parse_args()

    out_root = Path(args.output_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    bidi = json.loads(Path(args.bidirectional_summary).resolve().read_text(encoding="utf-8"))
    groups = derive_signed_groups(bidi)
    final_nodes = sorted({n for nodes in groups.values() for n in nodes}, key=lambda n: (node_layer(n), n))

    node_group: Dict[str, str] = {}
    for g, members in groups.items():
        for n in members:
            node_group[n] = g

    head_delta_lookup = load_head_delta_lookup(Path(args.head_read_csv).resolve())
    forward_role_lookup = load_forward_role_lookup(Path(args.forward_node_roles_csv).resolve()) if args.forward_node_roles_csv else {}

    node_rows_src = {str(r["node"]): r for r in read_csv_rows(Path(args.node_support_csv).resolve())}
    node_rows: List[Dict[str, object]] = []
    for node in final_nodes:
        src = node_rows_src.get(node, {})
        group_key = node_group[node]
        node_rows.append(
            {
                "node": node,
                "layer": node_layer(node),
                "group_key": group_key,
                "group_label": GROUP_META[group_key]["label"],
                "group_description": GROUP_META[group_key]["description"],
                "semantic_hint": semantic_hint_for_node(node, group_key, head_delta_lookup, forward_role_lookup),
                "forward_support": float(src.get("forward_support", 0.0)),
                "reverse_support": float(src.get("reverse_support", 0.0)),
                "shared_support_min": float(src.get("shared_support_min", 0.0)),
                "direction_balance": float(src.get("direction_balance", 0.0)),
            }
        )

    support = bidi.get("support_analysis", {})
    union_edge_set = {
        (str(src), str(dst))
        for src, dst in support.get("union_edges", [])
    }

    edge_rows_src = read_csv_rows(Path(args.edge_support_csv).resolve())
    kept_edges: List[Dict[str, object]] = []
    for row in edge_rows_src:
        src = str(row["source"])
        dst = str(row["target"])
        if union_edge_set and (src, dst) not in union_edge_set:
            continue
        if dst == "Residual Output: decision":
            keep = src in final_nodes
        else:
            keep = src in final_nodes and dst in final_nodes
        if not keep:
            continue
        sign = edge_sign_from_balance(float(row["direction_balance"]))
        kept_edges.append(
            {
                "source": src,
                "target": dst,
                "source_group": node_group.get(src, "terminal"),
                "target_group": node_group.get(dst, "output"),
                "forward_support": float(row["forward_support"]),
                "reverse_support": float(row["reverse_support"]),
                "shared_support_min": float(row["shared_support_min"]),
                "union_support_max": float(row["union_support_max"]),
                "direction_balance": float(row["direction_balance"]),
                "sign": sign,
            }
        )

    draw_signed_circuit(
        nodes=final_nodes,
        edges=kept_edges,
        node_group=node_group,
        out_path=out_root / "final_signed_circuit.png",
        title="Final Signed Decision Circuit",
    )

    summary = {
        "groups": groups,
        "group_meta": GROUP_META,
        "n_nodes": len(final_nodes),
        "n_edges": len(kept_edges),
        "artifacts": {
            "nodes_csv": str(out_root / "final_signed_nodes.csv"),
            "edges_csv": str(out_root / "final_signed_edges.csv"),
            "summary_json": str(out_root / "final_signed_circuit_summary.json"),
            "figure_png": str(out_root / "final_signed_circuit.png"),
        },
    }

    write_csv(node_rows, out_root / "final_signed_nodes.csv")
    write_csv(kept_edges, out_root / "final_signed_edges.csv")
    (out_root / "final_signed_circuit_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
