#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from toolcall_circuit.graph_utils import node_layer


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_json(payload: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def f(value: object) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def edge_summary(row: Dict[str, str]) -> Dict[str, Any]:
    return {
        "source": str(row["source"]),
        "target": str(row["target"]),
        "source_group": str(row.get("source_group", "")),
        "target_group": str(row.get("target_group", "")),
        "sign": str(row.get("sign", "")),
        "union_support_max": f(row.get("union_support_max")),
        "shared_support_min": f(row.get("shared_support_min")),
        "direction_balance": f(row.get("direction_balance")),
    }


def top_edges(
    rows: Sequence[Dict[str, str]],
    predicate: Callable[[Dict[str, str]], bool],
    *,
    limit: int,
) -> List[Dict[str, Any]]:
    picked = [edge_summary(r) for r in rows if predicate(r)]
    picked.sort(key=lambda r: (r["union_support_max"], -abs(r["direction_balance"])), reverse=True)
    return picked[:limit]


def sort_nodes(nodes: Iterable[str]) -> List[str]:
    return sorted({str(n) for n in nodes}, key=lambda n: (node_layer(n), n))


def node_groups(node_rows: Sequence[Dict[str, str]]) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for row in node_rows:
        out.setdefault(str(row["group_key"]), []).append(str(row["node"]))
    return {k: sort_nodes(v) for k, v in out.items()}


def node_lookup(node_rows: Sequence[Dict[str, str]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for row in node_rows:
        out[str(row["node"])] = {
            "layer": int(row["layer"]),
            "group_key": str(row["group_key"]),
            "semantic_hint": str(row.get("semantic_hint", "")),
            "forward_support": f(row.get("forward_support")),
            "reverse_support": f(row.get("reverse_support")),
            "shared_support_min": f(row.get("shared_support_min")),
            "direction_balance": f(row.get("direction_balance")),
        }
    return out


def row_lookup(rows: Sequence[Dict[str, Any]], key: str) -> Dict[str, Dict[str, Any]]:
    return {str(r[key]): dict(r) for r in rows}


def support_edge_dict(edge_pairs: Sequence[Sequence[str]]) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for pair in edge_pairs:
        if len(pair) != 2:
            continue
        out.append({"source": str(pair[0]), "target": str(pair[1])})
    return out


def module_registry(run_root: Path) -> Dict[str, Any]:
    bidirectional = read_json(run_root / "bidirectional" / "bidirectional_summary.json")
    signed = read_json(run_root / "final_signed_circuit" / "final_signed_circuit_summary.json")
    signed_validate = read_json(run_root / "signed_validate" / "signed_group_report.json")
    functional = read_json(run_root / "functional_groups" / "functional_group_summary.json")
    functional_validate = read_json(run_root / "functional_validate" / "functional_group_report.json")

    node_rows = read_csv_rows(run_root / "final_signed_circuit" / "final_signed_nodes.csv")
    edge_rows = read_csv_rows(run_root / "final_signed_circuit" / "final_signed_edges.csv")
    support = bidirectional["support_analysis"]
    signed_groups = node_groups(node_rows)
    node_meta = node_lookup(node_rows)
    signed_validate_rows = row_lookup(signed_validate.get("summary_rows", []), "group")
    functional_validate_rows = row_lookup(functional_validate.get("summary_rows", []), "group")
    functional_group_rows = row_lookup(functional.get("summary_rows", []), "functional_group")
    functional_groups = {str(k): sort_nodes(v) for k, v in functional.get("groups", {}).items()}

    arbitration = functional_groups.get("arbitration_integrators", [])
    tool_schema = functional_groups.get("tool_schema_readers", [])
    tool_writers = functional_groups.get("tool_call_writers", [])
    no_tool_writers = functional_groups.get("no_tool_writers", [])
    query_readers = functional_groups.get("user_query_readers", [])
    suppression_readers = functional_groups.get("suppression_readers", [])
    promotion_routers = functional_groups.get("promotion_routers_mediators", [])

    shared_spine = signed_groups.get("symmetric_backbone", [])
    tool_tail = signed_groups.get("tool_tail", [])
    no_tool_tail = signed_groups.get("no_tool_tail", [])

    instruction_ingress = sort_nodes(
        [
            n
            for n in query_readers + suppression_readers + promotion_routers
            if node_layer(n) <= 16
        ]
    )
    instruction_handoff = sort_nodes([n for n in shared_spine if 11 <= node_layer(n) <= 14])
    instruction_boundary_edges = support_edge_dict(
        [["L5H15", "L11H5"], ["L11H5", "L13H7"], ["L13H7", "L14H12"], ["MLP14", "L21H12"]]
    )

    decision_fork_tool = top_edges(
        edge_rows,
        lambda r: r["source_group"] == "symmetric_backbone"
        and r["target_group"] == "tool_tail"
        and r["sign"] == "tool_bias",
        limit=4,
    )
    decision_fork_no_tool = top_edges(
        edge_rows,
        lambda r: r["source_group"] == "symmetric_backbone"
        and r["target_group"] == "no_tool_tail"
        and r["sign"] == "no_tool_bias",
        limit=4,
    )

    construction_main_edges = support_edge_dict(
        [
            ["L22H7", "L23H14"],
            ["L23H14", "L24H6"],
            ["L24H6", "L26H0"],
            ["L26H0", "Residual Output: decision"],
        ]
    )
    construction_support_edges = top_edges(
        edge_rows,
        lambda r: r["source_group"] in {"symmetric_backbone", "tool_tail"}
        and r["target_group"] in {"symmetric_backbone", "tool_tail"}
        and r["sign"] == "tool_bias",
        limit=8,
    )
    construction_support_nodes = sort_nodes(
        [n for n in tool_schema + tool_writers if n not in {"L23H14", "L24H6", "L26H0"}]
    )

    suppression_main_edges = support_edge_dict(
        [
            ["L5H15", "L11H5"],
            ["L11H5", "L13H7"],
            ["L13H7", "L14H12"],
            ["L14H12", "L16H12"],
            ["L16H12", "MLP24"],
            ["MLP24", "L25H13"],
        ]
    )
    suppression_interference_edges = top_edges(
        edge_rows,
        lambda r: (
            (r["source_group"] == "symmetric_backbone" and r["target_group"] == "no_tool_tail")
            or (r["source_group"] == "no_tool_tail" and r["target_group"] in {"symmetric_backbone", "tool_tail"})
        )
        and r["sign"] == "no_tool_bias",
        limit=8,
    )

    registry = {
        "meta": {
            "run_root": str(run_root),
            "discovery_method": "eap_ig",
            "dataset_split": "train",
            "n_samples": int(bidirectional["forward"]["n_samples"]),
            "n_nodes": int(signed["n_nodes"]),
            "n_edges": int(signed["n_edges"]),
            "source_artifacts": {
                "bidirectional_summary": str(run_root / "bidirectional" / "bidirectional_summary.json"),
                "signed_summary": str(run_root / "final_signed_circuit" / "final_signed_circuit_summary.json"),
                "signed_validate": str(run_root / "signed_validate" / "signed_group_report.json"),
                "functional_groups": str(run_root / "functional_groups" / "functional_group_summary.json"),
                "functional_validate": str(run_root / "functional_validate" / "functional_group_report.json"),
            },
        },
        "signed_circuit": {
            "groups": signed_groups,
            "node_meta": node_meta,
            "final_signed_nodes_csv": str(run_root / "final_signed_circuit" / "final_signed_nodes.csv"),
            "final_signed_edges_csv": str(run_root / "final_signed_circuit" / "final_signed_edges.csv"),
            "signed_validate_rows": signed_validate_rows,
        },
        "support_sets": {
            "shared_backbone_nodes": support["shared_backbone_nodes"],
            "forward_selective_nodes": support["forward_selective_nodes"],
            "reverse_selective_nodes": support["reverse_selective_nodes"],
            "shared_backbone_edges": support["shared_backbone_edges"],
            "forward_selective_edges": support["forward_selective_edges"],
            "reverse_selective_edges": support["reverse_selective_edges"],
        },
        "functional_groups": {
            "groups": functional_groups,
            "summary_rows": functional_group_rows,
            "validation_rows": functional_validate_rows,
        },
        "instruction_integration": {
            "ingress_candidates": instruction_ingress,
            "handoff_nodes": instruction_handoff,
            "module_boundary_edges": instruction_boundary_edges,
            "evidence_groups": {
                "user_query_readers": query_readers,
                "suppression_readers": [n for n in suppression_readers if node_layer(n) <= 16],
                "promotion_routers_mediators": promotion_routers,
            },
            "validation_rows": {
                "user_query_readers": functional_validate_rows.get("user_query_readers", {}),
                "suppression_readers": functional_validate_rows.get("suppression_readers", {}),
                "promotion_routers_mediators": functional_validate_rows.get("promotion_routers_mediators", {}),
            },
        },
        "output_route_decision": {
            "decision_spine": shared_spine,
            "route_anchors": arbitration,
            "tool_fork_edges": decision_fork_tool,
            "no_tool_fork_edges": decision_fork_no_tool,
            "validation_rows": {
                "symmetric_backbone": signed_validate_rows.get("symmetric_backbone", {}),
                "arbitration_integrators": functional_validate_rows.get("arbitration_integrators", {}),
            },
        },
        "tool_call_construction": {
            "route_interface_nodes": arbitration,
            "main_chain_edges": construction_main_edges,
            "support_edges": construction_support_edges,
            "support_nodes": construction_support_nodes,
            "writer_nodes": sort_nodes([n for n in tool_writers if n.startswith("MLP")]),
            "evidence_groups": {
                "tool_schema_readers": tool_schema,
                "tool_call_writers": tool_writers,
            },
            "validation_rows": {
                "tool_tail": signed_validate_rows.get("tool_tail", {}),
                "tool_schema_readers": functional_validate_rows.get("tool_schema_readers", {}),
                "tool_call_writers": functional_validate_rows.get("tool_call_writers", {}),
            },
        },
        "tool_call_suppression": {
            "ingress_chain_nodes": sort_nodes(
                [n for n in no_tool_tail if node_layer(n) <= 16] + ["MLP24", "MLP25"]
            ),
            "main_chain_edges": suppression_main_edges,
            "interference_edges": suppression_interference_edges,
            "writer_nodes": no_tool_writers,
            "evidence_groups": {
                "suppression_readers": suppression_readers,
                "no_tool_writers": no_tool_writers,
            },
            "validation_rows": {
                "no_tool_tail": signed_validate_rows.get("no_tool_tail", {}),
                "suppression_readers": functional_validate_rows.get("suppression_readers", {}),
                "no_tool_writers": functional_validate_rows.get("no_tool_writers", {}),
            },
        },
    }
    return registry


def fmt(value: object, digits: int = 3) -> str:
    try:
        fv = float(value)
    except Exception:
        return str(value)
    if not math.isfinite(fv):
        return "nan"
    return f"{fv:.{digits}f}"


def build_markdown(registry: Dict[str, Any]) -> str:
    meta = registry["meta"]
    signed_rows = registry["signed_circuit"]["signed_validate_rows"]
    func_rows = registry["functional_groups"]["validation_rows"]
    lines: List[str] = []
    lines.append("# EAP-IG Module-Level Circuit Summary")
    lines.append("")
    lines.append("## 总览")
    lines.append("")
    lines.append(
        f"- Train discovery: `{meta['n_samples']}` 对样本，EAP-IG 双向全量发现。"
    )
    lines.append(
        f"- Final signed circuit: `{meta['n_nodes']}` 个节点，`{meta['n_edges']}` 条边。"
    )
    full_row = signed_rows["full_signed_circuit"]
    lines.append(
        f"- 全电路验证: promote suff `{fmt(full_row['promote_suff_ratio_median'])}`，"
        f"suppress suff `{fmt(full_row['suppress_suff_ratio_median'])}`，"
        f"promote top1 `{fmt(full_row['promote_tool_top1_rate'])}`，"
        f"suppress top1 `{fmt(full_row['suppress_no_tool_top1_rate'])}`。"
    )
    lines.append("")
    lines.append("## 结构组成")
    lines.append("")
    for key, nodes in registry["signed_circuit"]["groups"].items():
        row = signed_rows.get(key, {})
        lines.append(
            f"- `{key}`: `{len(nodes)}` 节点，"
            f"suff `{fmt(row.get('promote_suff_ratio_median'))}/{fmt(row.get('suppress_suff_ratio_median'))}`，"
            f"top1 `{fmt(row.get('promote_tool_top1_rate'))}/{fmt(row.get('suppress_no_tool_top1_rate'))}`。"
        )
    lines.append("")
    lines.append("## 模块电路")
    lines.append("")

    inst = registry["instruction_integration"]
    lines.append("### Instruction Integration")
    lines.append(
        f"- ingress candidates: `{', '.join(inst['ingress_candidates'])}`"
    )
    lines.append(
        f"- handoff nodes: `{', '.join(inst['handoff_nodes'])}`"
    )
    lines.append(
        f"- validation: query `{fmt(func_rows['user_query_readers']['promote_suff_ratio_median'])}/{fmt(func_rows['user_query_readers']['suppress_suff_ratio_median'])}`, "
        f"suppression readers `{fmt(func_rows['suppression_readers']['promote_suff_ratio_median'])}/{fmt(func_rows['suppression_readers']['suppress_suff_ratio_median'])}`."
    )
    lines.append("")

    decision = registry["output_route_decision"]
    lines.append("### Output-Route Decision")
    lines.append(
        f"- decision spine: `{', '.join(decision['decision_spine'])}`"
    )
    lines.append(
        f"- route anchors: `{', '.join(decision['route_anchors'])}`"
    )
    lines.append(
        f"- validation: shared spine `{fmt(signed_rows['symmetric_backbone']['promote_suff_ratio_median'])}/{fmt(signed_rows['symmetric_backbone']['suppress_suff_ratio_median'])}`, "
        f"arbitration integrators `{fmt(func_rows['arbitration_integrators']['promote_suff_ratio_median'])}/{fmt(func_rows['arbitration_integrators']['suppress_suff_ratio_median'])}`."
    )
    lines.append("")

    construction = registry["tool_call_construction"]
    lines.append("### Tool-Call Construction")
    lines.append(
        "- main chain: "
        + " -> ".join(f"{e['source']}->{e['target']}" for e in construction["main_chain_edges"])
    )
    lines.append(
        f"- support nodes: `{', '.join(construction['support_nodes'])}`"
    )
    lines.append(
        f"- validation: tool tail `{fmt(signed_rows['tool_tail']['promote_suff_ratio_median'])}/{fmt(signed_rows['tool_tail']['suppress_suff_ratio_median'])}`, "
        f"tool writers `{fmt(func_rows['tool_call_writers']['promote_suff_ratio_median'])}/{fmt(func_rows['tool_call_writers']['suppress_suff_ratio_median'])}`."
    )
    lines.append("")

    suppression = registry["tool_call_suppression"]
    suppression_edges_text = "; ".join(
        f"{e['source']}->{e['target']}" for e in suppression["interference_edges"][:5]
    )
    lines.append("### Tool-Call Suppression")
    lines.append(
        "- main chain: "
        + " -> ".join(f"{e['source']}->{e['target']}" for e in suppression["main_chain_edges"])
    )
    lines.append(
        f"- interference edges: `{suppression_edges_text}`"
    )
    lines.append(
        f"- validation: no-tool tail `{fmt(signed_rows['no_tool_tail']['promote_suff_ratio_median'])}/{fmt(signed_rows['no_tool_tail']['suppress_suff_ratio_median'])}`, "
        f"no-tool writers `{fmt(func_rows['no_tool_writers']['promote_suff_ratio_median'])}/{fmt(func_rows['no_tool_writers']['suppress_suff_ratio_median'])}`."
    )
    lines.append("")

    lines.append("## 关键文件")
    lines.append("")
    lines.append("- `final_signed_circuit/final_signed_nodes.csv`")
    lines.append("- `final_signed_circuit/final_signed_edges.csv`")
    lines.append("- `signed_validate/signed_group_report.json`")
    lines.append("- `functional_groups/functional_group_summary.json`")
    lines.append("- `functional_validate/functional_group_report.json`")
    lines.append("- `signed_layer_trajectory/signed_layer_trajectory_report.json`")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build module-level registry and summary for the EAP-IG train run.")
    parser.add_argument("--run-root", type=str, required=True)
    parser.add_argument("--registry-output", type=str, required=True)
    parser.add_argument("--summary-output", type=str, required=True)
    args = parser.parse_args()

    run_root = Path(args.run_root).resolve()
    registry = module_registry(run_root)
    write_json(registry, Path(args.registry_output).resolve())
    write_text(build_markdown(registry), Path(args.summary_output).resolve())

    print(
        json.dumps(
            {
                "registry_output": str(Path(args.registry_output).resolve()),
                "summary_output": str(Path(args.summary_output).resolve()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
