#!/usr/bin/env python3
"""
Build functional semantic groups on top of the structural signed circuit.

The grouping combines:
1) read pattern evidence from attention destination sets;
2) write / causal evidence from node-level sufficiency and necessity;
3) structural tags as auxiliary provenance, not as the final explanation.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

if __package__ in {None, ""}:
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from toolcall_circuit.graph_utils import apply_plot_style


FUNCTIONAL_META = {
    "tool_schema_readers": {
        "label": "Tool-Schema Readers",
        "color": "#8c510a",
        "description": "heads that read tool schema or tool-call tag structure and favor the tool endpoint",
    },
    "user_query_readers": {
        "label": "User-Query Readers",
        "color": "#4d9221",
        "description": "heads that read user content while still promoting tool-use execution",
    },
    "suppression_readers": {
        "label": "Suppression Readers",
        "color": "#2166ac",
        "description": "heads that read user/prefix evidence and tilt the system toward no-tool mode",
    },
    "promotion_routers_mediators": {
        "label": "Promotion Routers",
        "color": "#d6604d",
        "description": "routing or mediation heads that transmit tool-promoting state without being primary readers",
    },
    "suppression_routers_mediators": {
        "label": "Suppression Routers",
        "color": "#4393c3",
        "description": "routing or mediation heads that transmit no-tool state without being primary readers",
    },
    "tool_call_writers": {
        "label": "Tool-Call Writers",
        "color": "#b2182b",
        "description": "writer nodes whose causal profile is strongest for the tool-call endpoint",
    },
    "no_tool_writers": {
        "label": "No-Tool Writers",
        "color": "#053061",
        "description": "writer nodes whose causal profile is strongest for the no-tool endpoint",
    },
    "arbitration_integrators": {
        "label": "Arbitration Integrators",
        "color": "#7f7f7f",
        "description": "shared late-stage nodes that integrate competing evidence and stabilize the decision boundary",
    },
}

FUNCTIONAL_ORDER = [
    "tool_schema_readers",
    "user_query_readers",
    "suppression_readers",
    "promotion_routers_mediators",
    "suppression_routers_mediators",
    "tool_call_writers",
    "no_tool_writers",
    "arbitration_integrators",
]

GRAPH_POS = {
    "tool_schema_readers": (-2.2, 1.8),
    "user_query_readers": (-0.8, 1.8),
    "suppression_readers": (0.8, 1.8),
    "promotion_routers_mediators": (-1.6, 0.6),
    "suppression_routers_mediators": (1.6, 0.6),
    "tool_call_writers": (-1.0, -0.8),
    "no_tool_writers": (1.0, -0.8),
    "arbitration_integrators": (0.0, 0.2),
}

EDGE_SIGN_COLORS = {
    "shared": "#7b7b7b",
    "tool_bias": "#b45f06",
    "no_tool_bias": "#2f6f9f",
}


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


def safe_float(value: object, default: float = 0.0) -> float:
    try:
        value = float(value)
    except Exception:
        return default
    return value if math.isfinite(value) else default


def median(values: Iterable[float]) -> float:
    vals = sorted(safe_float(v, float("nan")) for v in values if math.isfinite(safe_float(v, float("nan"))))
    if not vals:
        return float("nan")
    return vals[len(vals) // 2]


def load_head_read_metrics(path: Path) -> Dict[str, Dict[str, Dict[str, float]]]:
    metrics: Dict[str, Dict[str, Dict[str, float]]] = defaultdict(dict)
    for row in read_csv_rows(path):
        head = str(row["head"])
        set_name = str(row["set"])
        metrics[head][set_name] = {
            "tool_mass_median": safe_float(row.get("tool_mass_median"), 0.0),
            "no_tool_mass_median": safe_float(row.get("no_tool_mass_median"), 0.0),
            "delta_median_no_minus_tool": safe_float(row.get("delta_median_no_minus_tool"), 0.0),
        }
    return metrics


def build_read_feature_lookup(metrics: Dict[str, Dict[str, Dict[str, float]]], head: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for set_name, row in metrics.get(head, {}).items():
        out[f"{set_name}_tool_mass"] = safe_float(row.get("tool_mass_median"), 0.0)
        out[f"{set_name}_no_tool_mass"] = safe_float(row.get("no_tool_mass_median"), 0.0)
        out[f"{set_name}_delta"] = safe_float(row.get("delta_median_no_minus_tool"), 0.0)
    return out


def build_causal_lookup(path: Path) -> Dict[str, Dict[str, float]]:
    lookup: Dict[str, Dict[str, float]] = {}
    for row in read_csv_rows(path):
        node = str(row["node"])
        lookup[node] = {
            "promote_nec_drop": safe_float(row.get("promote_nec_drop_median"), 0.0),
            "suppress_nec_drop": safe_float(row.get("suppress_nec_drop_median"), 0.0),
            "promote_suff_ratio": safe_float(row.get("promote_suff_ratio_median"), 0.0),
            "suppress_suff_ratio": safe_float(row.get("suppress_suff_ratio_median"), 0.0),
            "promote_top1_rate": safe_float(row.get("promote_suff_top1_rate"), 0.0),
            "suppress_top1_rate": safe_float(row.get("suppress_suff_top1_rate"), 0.0),
        }
    return lookup


def promote_strength(causal: Dict[str, float]) -> float:
    return max(0.0, causal.get("promote_suff_ratio", 0.0)) + max(0.0, causal.get("promote_nec_drop", 0.0))


def suppress_strength(causal: Dict[str, float]) -> float:
    return max(0.0, causal.get("suppress_suff_ratio", 0.0)) + max(0.0, causal.get("suppress_nec_drop", 0.0))


def classify_head(node_row: Dict[str, str], read: Dict[str, float], causal: Dict[str, float]) -> tuple[str, str]:
    structural = str(node_row.get("group_key", ""))
    layer = int(node_row.get("layer", 0))
    direction_balance = safe_float(node_row.get("direction_balance"), 0.0)
    promote = promote_strength(causal)
    suppress = suppress_strength(causal)
    causal_delta = promote - suppress

    tool_tools = read.get("tools_block_tool_mass", 0.0)
    tool_tags = read.get("tool_call_tags_tool_mass", 0.0)
    tool_user = read.get("user_block_tool_mass", 0.0)
    no_tool_user = read.get("user_block_no_tool_mass", 0.0)
    delta_user = read.get("user_block_delta", 0.0)
    delta_tools = read.get("tools_block_delta", 0.0)
    delta_tags = read.get("tool_call_tags_delta", 0.0)
    delta_prefix = read.get("prefix_16_delta", 0.0)

    if structural == "symmetric_backbone" and abs(causal_delta) < 0.35 and layer >= 18:
        return "arbitration_integrators", f"late shared head with balanced causal profile ({causal_delta:+.2f})"
    schema_read = (tool_tools >= 0.10 or tool_tags >= 0.008) and (delta_tools <= -0.01 or delta_tags <= -0.002)
    user_read = tool_user >= 0.22 and tool_tools < 0.16

    if schema_read and (causal_delta > 0.05 or direction_balance <= 0.20):
        return "tool_schema_readers", f"reads tools/tags (tools={tool_tools:.2f}, tags={tool_tags:.3f})"
    if user_read and promote >= suppress - 0.05 and direction_balance <= 0.20 and delta_user < 0.08:
        return "user_query_readers", f"reads user content on tool endpoint (user={tool_user:.2f})"
    if delta_user >= 0.03 or delta_prefix >= 0.04 or direction_balance > 0.20 or (no_tool_user >= 0.30 and suppress >= promote - 0.05):
        return "suppression_readers", f"reads user/prefix more in no-tool mode (d_user={delta_user:+.2f}, d_prefix={delta_prefix:+.2f})"
    if causal_delta > 0.15 or structural in {"tool_bias_backbone", "tool_tail"}:
        return "promotion_routers_mediators", f"promote-heavy causal role ({causal_delta:+.2f})"
    if causal_delta < -0.15 or structural in {"no_tool_bias_backbone", "no_tool_tail"}:
        return "suppression_routers_mediators", f"suppress-heavy causal role ({causal_delta:+.2f})"
    return "arbitration_integrators", f"balanced router ({causal_delta:+.2f})"


def classify_mlp(node_row: Dict[str, str], causal: Dict[str, float]) -> tuple[str, str]:
    structural = str(node_row.get("group_key", ""))
    layer = int(node_row.get("layer", 0))
    promote = promote_strength(causal)
    suppress = suppress_strength(causal)
    causal_delta = promote - suppress

    if structural == "symmetric_backbone" and abs(causal_delta) < 0.35 and layer >= 16:
        return "arbitration_integrators", f"shared late writer with balanced causal role ({causal_delta:+.2f})"
    if causal_delta > 0.15 or structural in {"tool_bias_backbone", "tool_tail"}:
        return "tool_call_writers", f"tool-skewed writer ({causal_delta:+.2f})"
    if causal_delta < -0.15 or structural in {"no_tool_bias_backbone", "no_tool_tail"}:
        return "no_tool_writers", f"no-tool-skewed writer ({causal_delta:+.2f})"
    if promote >= suppress:
        return "tool_call_writers", f"writer defaults to tool side ({causal_delta:+.2f})"
    return "no_tool_writers", f"writer defaults to no-tool side ({causal_delta:+.2f})"


def functional_group_for_node(
    node_row: Dict[str, str],
    head_read_lookup: Dict[str, Dict[str, Dict[str, float]]],
    causal_lookup: Dict[str, Dict[str, float]],
) -> Dict[str, object]:
    node = str(node_row["node"])
    node_type = "mlp" if node.startswith("MLP") else "head"
    causal = causal_lookup.get(node, {})
    read = build_read_feature_lookup(head_read_lookup, node) if node_type == "head" else {}
    if node_type == "head":
        functional_group, evidence = classify_head(node_row, read, causal)
    else:
        functional_group, evidence = classify_mlp(node_row, causal)
    promote = promote_strength(causal)
    suppress = suppress_strength(causal)
    return {
        "node": node,
        "layer": int(node_row["layer"]),
        "node_type": node_type,
        "structural_group": str(node_row.get("group_key", "")),
        "functional_group": functional_group,
        "functional_label": FUNCTIONAL_META[functional_group]["label"],
        "evidence": evidence,
        "promote_strength": promote,
        "suppress_strength": suppress,
        "causal_delta": promote - suppress,
        "direction_balance": safe_float(node_row.get("direction_balance"), 0.0),
        "semantic_hint": str(node_row.get("semantic_hint", "")),
        "tools_block_tool_mass": read.get("tools_block_tool_mass", float("nan")),
        "tools_block_delta": read.get("tools_block_delta", float("nan")),
        "tool_call_tags_tool_mass": read.get("tool_call_tags_tool_mass", float("nan")),
        "tool_call_tags_delta": read.get("tool_call_tags_delta", float("nan")),
        "user_block_tool_mass": read.get("user_block_tool_mass", float("nan")),
        "user_block_no_tool_mass": read.get("user_block_no_tool_mass", float("nan")),
        "user_block_delta": read.get("user_block_delta", float("nan")),
        "prefix_16_delta": read.get("prefix_16_delta", float("nan")),
    }


def aggregate_group_edges(
    functional_rows: Sequence[Dict[str, object]],
    signed_edges: Sequence[Dict[str, str]],
) -> List[Dict[str, object]]:
    node_to_group = {str(row["node"]): str(row["functional_group"]) for row in functional_rows}
    buckets: Dict[Tuple[str, str, str], List[Dict[str, str]]] = defaultdict(list)
    for row in signed_edges:
        src = str(row["source"])
        dst = str(row["target"])
        if src not in node_to_group or dst not in node_to_group:
            continue
        key = (node_to_group[src], node_to_group[dst], str(row["sign"]))
        buckets[key].append(row)

    edge_rows: List[Dict[str, object]] = []
    for (src_group, dst_group, sign), rows in buckets.items():
        supports = [safe_float(r.get("union_support_max"), 0.0) for r in rows]
        edge_rows.append(
            {
                "source_group": src_group,
                "target_group": dst_group,
                "sign": sign,
                "n_edges": len(rows),
                "support_median": median(supports),
                "top_edges": " | ".join(
                    f"{r['source']}->{r['target']} ({safe_float(r.get('union_support_max'), 0.0):.3f})"
                    for r in sorted(rows, key=lambda x: safe_float(x.get("union_support_max"), 0.0), reverse=True)[:4]
                ),
            }
        )
    edge_rows.sort(key=lambda r: (safe_float(r["support_median"], 0.0), int(r["n_edges"])), reverse=True)
    return edge_rows


def draw_functional_graph(rows: Sequence[Dict[str, object]], out_path: Path) -> None:
    apply_plot_style()
    fig, ax = plt.subplots(figsize=(11.0, 7.5), constrained_layout=True)
    ax.axis("off")
    ax.set_title("Functional Semantic Groups")

    max_support = max(safe_float(r["support_median"], 0.0) for r in rows) if rows else 1.0
    for row in rows:
        src = str(row["source_group"])
        dst = str(row["target_group"])
        if src not in GRAPH_POS or dst not in GRAPH_POS:
            continue
        x1, y1 = GRAPH_POS[src]
        x2, y2 = GRAPH_POS[dst]
        color = EDGE_SIGN_COLORS.get(str(row["sign"]), "#7b7b7b")
        width = 1.2 + 4.2 * safe_float(row["support_median"], 0.0) / max(1e-6, max_support)
        rad = 0.14 if src != dst else 0.30
        arrow = FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle="->",
            mutation_scale=16,
            linewidth=width,
            color=color,
            alpha=0.86,
            connectionstyle=f"arc3,rad={rad}",
            zorder=1,
        )
        ax.add_patch(arrow)

    for key in FUNCTIONAL_ORDER:
        meta = FUNCTIONAL_META[key]
        x, y = GRAPH_POS[key]
        ax.scatter([x], [y], s=1500, c=meta["color"], edgecolors="#2d2d2d", linewidths=1.4, zorder=3)
        ax.text(x, y - 0.42, meta["label"], ha="center", va="top", fontsize=10)

    ax.set_xlim(-3.2, 3.2)
    ax.set_ylim(-1.6, 2.6)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build functional semantic groups for the signed circuit.")
    parser.add_argument("--signed-nodes-csv", type=str, required=True)
    parser.add_argument("--signed-edges-csv", type=str, required=True)
    parser.add_argument("--head-read-csv", type=str, required=True)
    parser.add_argument("--node-importance-csv", type=str, required=True)
    parser.add_argument("--output-root", type=str, required=True)
    args = parser.parse_args()

    out_root = Path(args.output_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    signed_nodes = read_csv_rows(Path(args.signed_nodes_csv).resolve())
    signed_edges = read_csv_rows(Path(args.signed_edges_csv).resolve())
    head_reads = load_head_read_metrics(Path(args.head_read_csv).resolve())
    causal_lookup = build_causal_lookup(Path(args.node_importance_csv).resolve())

    functional_rows = [
        functional_group_for_node(row, head_reads, causal_lookup)
        for row in signed_nodes
    ]
    functional_rows.sort(
        key=lambda r: (
            FUNCTIONAL_ORDER.index(str(r["functional_group"])),
            int(r["layer"]),
            str(r["node"]),
        )
    )

    groups: Dict[str, List[str]] = defaultdict(list)
    for row in functional_rows:
        groups[str(row["functional_group"])].append(str(row["node"]))

    summary_rows: List[Dict[str, object]] = []
    for key in FUNCTIONAL_ORDER:
        nodes = groups.get(key, [])
        if not nodes:
            continue
        members = [row for row in functional_rows if str(row["functional_group"]) == key]
        structural_counts = Counter(str(row["structural_group"]) for row in members)
        summary_rows.append(
            {
                "functional_group": key,
                "functional_label": FUNCTIONAL_META[key]["label"],
                "description": FUNCTIONAL_META[key]["description"],
                "n_nodes": len(nodes),
                "n_heads": sum(1 for row in members if row["node_type"] == "head"),
                "n_mlps": sum(1 for row in members if row["node_type"] == "mlp"),
                "nodes": ",".join(nodes),
                "representatives": ",".join(nodes[:5]),
                "promote_strength_median": median(row["promote_strength"] for row in members),
                "suppress_strength_median": median(row["suppress_strength"] for row in members),
                "direction_balance_median": median(row["direction_balance"] for row in members),
                "structural_mix": json.dumps(structural_counts, ensure_ascii=False, sort_keys=True),
            }
        )

    functional_edge_rows = aggregate_group_edges(functional_rows, signed_edges)
    draw_functional_graph(functional_edge_rows, out_root / "functional_group_graph.png")

    write_csv(functional_rows, out_root / "functional_node_table.csv")
    write_csv(summary_rows, out_root / "functional_group_summary.csv")
    write_csv(functional_edge_rows, out_root / "functional_group_edges.csv")

    summary = {
        "group_meta": FUNCTIONAL_META,
        "groups": {key: groups[key] for key in FUNCTIONAL_ORDER if key in groups},
        "artifacts": {
            "node_table_csv": str(out_root / "functional_node_table.csv"),
            "group_summary_csv": str(out_root / "functional_group_summary.csv"),
            "group_edges_csv": str(out_root / "functional_group_edges.csv"),
            "graph_png": str(out_root / "functional_group_graph.png"),
            "summary_json": str(out_root / "functional_group_summary.json"),
        },
        "summary_rows": summary_rows,
        "edge_rows": functional_edge_rows,
    }
    (out_root / "functional_group_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
