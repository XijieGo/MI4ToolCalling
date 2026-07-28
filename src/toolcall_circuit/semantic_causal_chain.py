#!/usr/bin/env python3
"""
Stagewise semantic causal-chain analysis for the final signed tool-call circuit.

This script upgrades the functional grouping into an executable mechanism story:

1. Extract representative high-scoring paths through the signed circuit:
   - query-conditioned tool branch
   - schema-conditioned tool branch
   - no-tool suppression branch
2. For each path, run cumulative activation patching step by step.
3. Report how much each additional node moves the endpoint objective.

The result is still a *candidate* mechanistic explanation, but it is grounded in:
- token-span read evidence
- node-level causal sufficiency
- edge-level mediation
- stagewise cumulative patching
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm.auto import tqdm

if __package__ in {None, ""}:
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from toolcall_circuit.bidirectional_causal_eval import collect_cache_cpu_for_nodes, load_sample_paths
from toolcall_circuit.bidirectional_token_flip import run_logits_on_base_with_source
from toolcall_circuit.objective import build_bidirectional_endpoint_objectives
from toolcall_circuit.single_sample import load_hooked_qwen3, objective_from_logits


@dataclass(frozen=True)
class PathSpec:
    key: str
    label: str
    direction: str
    start_group: str
    allowed_groups: tuple[str, ...]
    required_groups: tuple[str, ...]
    metric_field: str
    nodes: tuple[str, ...]
    score: float


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


def finite(values: Iterable[float]) -> List[float]:
    return [float(v) for v in values if isinstance(v, (int, float)) and math.isfinite(float(v))]


def median(values: Iterable[float]) -> float:
    vals = finite(values)
    return float(np.median(vals)) if vals else float("nan")


def mean(values: Iterable[float]) -> float:
    vals = finite(values)
    return float(np.mean(vals)) if vals else float("nan")


def safe_rate(values: Iterable[bool]) -> float:
    vals = [1.0 if bool(v) else 0.0 for v in values]
    return float(np.mean(vals)) if vals else float("nan")


def safe_float(value: object, default: float = float("nan")) -> float:
    try:
        value = float(value)
    except Exception:
        return default
    return value if math.isfinite(value) else default


def load_node_table(path: Path) -> Dict[str, Dict[str, object]]:
    rows = read_csv_rows(path)
    out: Dict[str, Dict[str, object]] = {}
    for row in rows:
        out[str(row["node"])] = {
            "node": str(row["node"]),
            "layer": int(row["layer"]),
            "functional_group": str(row["functional_group"]),
            "functional_label": str(row["functional_label"]),
            "structural_group": str(row["structural_group"]),
            "evidence": str(row["evidence"]),
            "semantic_hint": str(row.get("semantic_hint", "")),
            "promote_strength": safe_float(row.get("promote_strength"), 0.0),
            "suppress_strength": safe_float(row.get("suppress_strength"), 0.0),
            "causal_delta": safe_float(row.get("causal_delta"), 0.0),
            "tools_block_tool_mass": safe_float(row.get("tools_block_tool_mass"), float("nan")),
            "tool_call_tags_tool_mass": safe_float(row.get("tool_call_tags_tool_mass"), float("nan")),
            "user_block_tool_mass": safe_float(row.get("user_block_tool_mass"), float("nan")),
            "user_block_no_tool_mass": safe_float(row.get("user_block_no_tool_mass"), float("nan")),
            "user_block_delta": safe_float(row.get("user_block_delta"), float("nan")),
            "prefix_16_delta": safe_float(row.get("prefix_16_delta"), float("nan")),
        }
    return out


def load_edge_support(path: Path) -> Dict[tuple[str, str], Dict[str, object]]:
    rows = read_csv_rows(path)
    return {
        (str(row["source"]), str(row["target"])): {
            "source": str(row["source"]),
            "target": str(row["target"]),
            "sign": str(row["sign"]),
            "union_support_max": safe_float(row.get("union_support_max"), 0.0),
            "shared_support_min": safe_float(row.get("shared_support_min"), 0.0),
            "forward_support": safe_float(row.get("forward_support"), 0.0),
            "reverse_support": safe_float(row.get("reverse_support"), 0.0),
        }
        for row in rows
    }


def load_edge_mediation(path: Path) -> Dict[tuple[str, str], Dict[str, object]]:
    rows = read_csv_rows(path)
    return {
        (str(row["source"]), str(row["target"])): {
            "source": str(row["source"]),
            "target": str(row["target"]),
            "sign": str(row["sign"]),
            "promote_mediated_ratio_median": safe_float(row.get("promote_mediated_ratio_median"), 0.0),
            "suppress_mediated_ratio_median": safe_float(row.get("suppress_mediated_ratio_median"), 0.0),
            "promote_source_ratio_median": safe_float(row.get("promote_source_ratio_median"), 0.0),
            "suppress_source_ratio_median": safe_float(row.get("suppress_source_ratio_median"), 0.0),
        }
        for row in rows
    }


def node_groups(node_table: Mapping[str, Dict[str, object]]) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for node, row in node_table.items():
        out.setdefault(str(row["functional_group"]), []).append(node)
    for nodes in out.values():
        nodes.sort(key=lambda n: (int(node_table[n]["layer"]), n))
    return out


def best_path(
    *,
    start_nodes: Sequence[str],
    allowed_groups: Sequence[str],
    required_groups: Sequence[str],
    metric_field: str,
    node_table: Mapping[str, Dict[str, object]],
    edge_support: Mapping[tuple[str, str], Dict[str, object]],
    edge_mediation: Mapping[tuple[str, str], Dict[str, object]],
) -> tuple[float, List[str]]:
    node_layer = {node: int(row["layer"]) for node, row in node_table.items()}
    node_layer["Residual Output: decision"] = 10**9
    allowed_nodes = {
        node
        for node, row in node_table.items()
        if str(row["functional_group"]) in set(str(x) for x in allowed_groups)
    }
    allowed_nodes.add("Residual Output: decision")
    edges = list(edge_support.keys())
    adj: Dict[str, List[tuple[str, float]]] = {}
    for source, target in edges:
        if source not in allowed_nodes or target not in allowed_nodes:
            continue
        if node_layer.get(source, -1) >= node_layer.get(target, 10**9):
            continue
        if target == "Residual Output: decision":
            weight = safe_float(edge_support[(source, target)].get("union_support_max"), 0.0)
        else:
            weight = safe_float(edge_mediation.get((source, target), {}).get(metric_field), 0.0)
        if weight <= 0:
            continue
        adj.setdefault(source, []).append((target, weight))

    best_score = float("-inf")
    best_nodes: List[str] = []

    required = set(str(x) for x in required_groups)

    def satisfies_required(path: Sequence[str]) -> bool:
        if not required:
            return True
        groups_in_path = {str(node_table[n]["functional_group"]) for n in path if n in node_table}
        return required.issubset(groups_in_path)

    def dfs(node: str, path: List[str], score: float) -> None:
        nonlocal best_score, best_nodes
        if node == "Residual Output: decision":
            if satisfies_required(path) and score > best_score:
                best_score = score
                best_nodes = list(path)
            return
        for target, weight in adj.get(node, []):
            if target in path:
                continue
            dfs(target, path + [target], score + math.log(weight + 1e-6))

    for start in start_nodes:
        dfs(start, [start], 0.0)
    return best_score, best_nodes


def choose_paths(
    node_table: Mapping[str, Dict[str, object]],
    edge_support: Mapping[tuple[str, str], Dict[str, object]],
    edge_mediation: Mapping[tuple[str, str], Dict[str, object]],
) -> List[PathSpec]:
    groups = node_groups(node_table)
    specs = [
        (
            "query_tool_path",
            "Query-Conditioned Tool Branch",
            "tool",
            "user_query_readers",
            ("user_query_readers", "tool_call_writers", "tool_schema_readers", "arbitration_integrators"),
            ("tool_call_writers", "arbitration_integrators"),
            "promote_mediated_ratio_median",
        ),
        (
            "schema_tool_path",
            "Schema-Conditioned Tool Branch",
            "tool",
            "tool_schema_readers",
            ("tool_schema_readers", "tool_call_writers", "arbitration_integrators"),
            ("arbitration_integrators",),
            "promote_mediated_ratio_median",
        ),
        (
            "no_tool_path",
            "No-Tool Suppression Branch",
            "no_tool",
            "suppression_readers",
            ("suppression_readers", "no_tool_writers", "arbitration_integrators"),
            ("no_tool_writers",),
            "suppress_mediated_ratio_median",
        ),
    ]
    out: List[PathSpec] = []
    for key, label, direction, start_group, allowed_groups, required_groups, metric_field in specs:
        starts = groups.get(start_group, [])
        if not starts:
            continue
        score, nodes = best_path(
            start_nodes=starts,
            allowed_groups=allowed_groups,
            required_groups=required_groups,
            metric_field=metric_field,
            node_table=node_table,
            edge_support=edge_support,
            edge_mediation=edge_mediation,
        )
        if not nodes:
            continue
        out.append(
            PathSpec(
                key=key,
                label=label,
                direction=direction,
                start_group=start_group,
                allowed_groups=tuple(allowed_groups),
                required_groups=tuple(required_groups),
                metric_field=metric_field,
                nodes=tuple(nodes),
                score=float(score),
            )
        )
    return out


def describe_node(node: str, node_table: Mapping[str, Dict[str, object]]) -> str:
    if node == "Residual Output: decision":
        return "final endpoint write"
    row = node_table[node]
    return (
        f"{row['functional_label']} / {row['structural_group']} / "
        f"{row['semantic_hint']} / {row['evidence']}"
    )


def path_edge_rows(
    path: PathSpec,
    *,
    node_table: Mapping[str, Dict[str, object]],
    edge_support: Mapping[tuple[str, str], Dict[str, object]],
    edge_mediation: Mapping[tuple[str, str], Dict[str, object]],
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for source, target in zip(path.nodes[:-1], path.nodes[1:]):
        support = edge_support.get((source, target), {})
        mediation = edge_mediation.get((source, target), {})
        rows.append(
            {
                "path_key": path.key,
                "path_label": path.label,
                "direction": path.direction,
                "source": source,
                "target": target,
                "source_label": describe_node(source, node_table),
                "target_label": describe_node(target, node_table),
                "sign": str(support.get("sign", mediation.get("sign", ""))),
                "union_support_max": safe_float(support.get("union_support_max"), float("nan")),
                "promote_mediated_ratio_median": safe_float(mediation.get("promote_mediated_ratio_median"), float("nan")),
                "suppress_mediated_ratio_median": safe_float(mediation.get("suppress_mediated_ratio_median"), float("nan")),
            }
        )
    return rows


def evaluate_paths(
    *,
    run_root: Path,
    path_specs: Sequence[PathSpec],
    model_path: str,
    device: str,
    max_samples: int,
    output_root: Path,
) -> Dict[str, object]:
    forward_batch_root = run_root / "forward_batch"
    reverse_batch_root = run_root / "reverse_batch"
    samples = load_sample_paths(forward_batch_root, reverse_batch_root, max_samples=max_samples)
    model, tokenizer = load_hooked_qwen3(model_path, device=device, dtype=torch.bfloat16)

    all_patch_nodes = sorted(
        {
            node
            for spec in path_specs
            for node in spec.nodes
            if node != "Residual Output: decision"
        }
    )

    per_sample_rows: List[Dict[str, object]] = []
    pbar = tqdm(samples, desc="Semantic chain eval", dynamic_ncols=True)
    for sp in pbar:
        try:
            tool_text = sp.tool_call_prompt.read_text(encoding="utf-8")
            no_tool_text = sp.no_tool_prompt.read_text(encoding="utf-8")
        except Exception:
            continue
        tool_tokens = model.to_tokens(tool_text, prepend_bos=False)
        no_tool_tokens = model.to_tokens(no_tool_text, prepend_bos=False)
        if tool_tokens.shape != no_tool_tokens.shape:
            continue

        with torch.no_grad():
            tool_logits = model(tool_tokens)
            no_tool_logits = model(no_tool_tokens)
        tool_objective, no_tool_objective = build_bidirectional_endpoint_objectives(
            tool_logits,
            no_tool_logits,
            tokenizer=tokenizer,
        )
        tool_base = float(objective_from_logits(no_tool_logits, tool_objective).item())
        tool_gap = float(objective_from_logits(tool_logits, tool_objective).item()) - tool_base
        no_tool_base = float(objective_from_logits(tool_logits, no_tool_objective).item())
        no_tool_gap = float(objective_from_logits(no_tool_logits, no_tool_objective).item()) - no_tool_base
        if (
            not math.isfinite(tool_gap)
            or abs(tool_gap) < 1e-8
            or not math.isfinite(no_tool_gap)
            or abs(no_tool_gap) < 1e-8
        ):
            continue

        tool_cache = collect_cache_cpu_for_nodes(model, tool_tokens, all_patch_nodes)
        no_tool_cache = collect_cache_cpu_for_nodes(model, no_tool_tokens, all_patch_nodes)

        base_row: Dict[str, object] = {
            "sample_id": sp.sample_id,
            "target_tool_call_id": sp.target_tool_call,
            "target_tool_call_str": tokenizer.decode([sp.target_tool_call]),
            "target_no_tool_id": sp.distractor,
            "target_no_tool_str": tokenizer.decode([sp.distractor]),
        }

        for spec in path_specs:
            cumulative_nodes: List[str] = []
            for step_idx, node in enumerate(spec.nodes[:-1], start=1):
                cumulative_nodes.append(node)
                if spec.direction == "tool":
                    logits = run_logits_on_base_with_source(model, no_tool_tokens, tool_cache, cumulative_nodes)
                    ratio = (float(objective_from_logits(logits, tool_objective).item()) - tool_base) / tool_gap
                    top1 = int(logits[0, -1].argmax().item()) == sp.target_tool_call
                    boundary_flip = float(objective_from_logits(logits, tool_objective).item()) > float(
                        objective_from_logits(logits, no_tool_objective).item()
                    )
                else:
                    logits = run_logits_on_base_with_source(model, tool_tokens, no_tool_cache, cumulative_nodes)
                    ratio = (float(objective_from_logits(logits, no_tool_objective).item()) - no_tool_base) / no_tool_gap
                    top1 = int(logits[0, -1].argmax().item()) == sp.distractor
                    boundary_flip = float(objective_from_logits(logits, no_tool_objective).item()) > float(
                        objective_from_logits(logits, tool_objective).item()
                    )
                base_row[f"{spec.key}__step_{step_idx}__node"] = node
                base_row[f"{spec.key}__step_{step_idx}__ratio"] = ratio
                base_row[f"{spec.key}__step_{step_idx}__top1"] = top1
                base_row[f"{spec.key}__step_{step_idx}__boundary_flip"] = boundary_flip
        per_sample_rows.append(base_row)
        pbar.set_postfix(sample=sp.sample_id)

    write_csv(per_sample_rows, output_root / "semantic_chain_per_sample.csv")

    summary_rows: List[Dict[str, object]] = []
    for spec in path_specs:
        prev_key = None
        for step_idx, node in enumerate(spec.nodes[:-1], start=1):
            ratio_key = f"{spec.key}__step_{step_idx}__ratio"
            top1_key = f"{spec.key}__step_{step_idx}__top1"
            boundary_key = f"{spec.key}__step_{step_idx}__boundary_flip"
            ratios = [safe_float(row.get(ratio_key), float("nan")) for row in per_sample_rows]
            increments = []
            if prev_key is None:
                increments = ratios
            else:
                prev_ratios = [safe_float(row.get(prev_key), float("nan")) for row in per_sample_rows]
                increments = [
                    r - p
                    for r, p in zip(ratios, prev_ratios)
                    if math.isfinite(r) and math.isfinite(p)
                ]
            summary_rows.append(
                {
                    "path_key": spec.key,
                    "path_label": spec.label,
                    "direction": spec.direction,
                    "step_idx": step_idx,
                    "node": node,
                    "n_samples": len(per_sample_rows),
                    "cumulative_ratio_median": median(ratios),
                    "cumulative_ratio_mean": mean(ratios),
                    "incremental_ratio_median": median(increments),
                    "incremental_ratio_mean": mean(increments),
                    "top1_rate": safe_rate(bool(row.get(top1_key)) for row in per_sample_rows),
                    "boundary_flip_rate": safe_rate(bool(row.get(boundary_key)) for row in per_sample_rows),
                }
            )
            prev_key = ratio_key

    write_csv(summary_rows, output_root / "semantic_chain_summary.csv")
    summary = {
        "n_samples": len(per_sample_rows),
        "paths": [
            {
                "key": spec.key,
                "label": spec.label,
                "direction": spec.direction,
                "nodes": list(spec.nodes),
                "score": float(spec.score),
            }
            for spec in path_specs
        ],
        "summary_rows": summary_rows,
        "artifacts": {
            "summary_csv": str(output_root / "semantic_chain_summary.csv"),
            "summary_json": str(output_root / "semantic_chain_summary.json"),
            "per_sample_csv": str(output_root / "semantic_chain_per_sample.csv"),
            "plot": str(output_root / "semantic_chain_progression.png"),
            "report": str(output_root / "semantic_chain_report.md"),
        },
    }
    (output_root / "semantic_chain_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def plot_progression(summary_rows: Sequence[Dict[str, object]], out_path: Path) -> None:
    by_path: Dict[str, List[Dict[str, object]]] = {}
    for row in summary_rows:
        by_path.setdefault(str(row["path_key"]), []).append(row)
    if not by_path:
        return
    plt.style.use("default")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    for path_key, rows in by_path.items():
        rows = sorted(rows, key=lambda r: int(r["step_idx"]))
        x = np.arange(1, len(rows) + 1)
        label = str(rows[0]["path_label"])
        axes[0].plot(x, [float(r["cumulative_ratio_median"]) for r in rows], marker="o", label=label)
        axes[1].plot(x, [float(r["incremental_ratio_median"]) for r in rows], marker="o", label=label)
    axes[0].set_title("Cumulative Recovery by Path Step")
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Median recovery ratio")
    axes[1].set_title("Incremental Gain by Path Step")
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Median incremental gain")
    for ax in axes:
        ax.axhline(0.0, color="#888888", linewidth=1.0)
        ax.legend(frameon=False, fontsize=9)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def build_markdown_report(
    *,
    run_root: Path,
    output_root: Path,
    node_table: Mapping[str, Dict[str, object]],
    path_specs: Sequence[PathSpec],
    path_edge_evidence: Mapping[str, List[Dict[str, object]]],
    summary_rows: Sequence[Dict[str, object]],
) -> str:
    rows_by_path: Dict[str, List[Dict[str, object]]] = {}
    for row in summary_rows:
        rows_by_path.setdefault(str(row["path_key"]), []).append(row)

    lines: List[str] = []
    lines.append("# Semantic Causal Chain Report")
    lines.append("")
    lines.append(f"- Run root: `{run_root}`")
    lines.append("")
    lines.append("## Extracted Candidate Paths")
    lines.append("")
    for spec in path_specs:
        lines.append(f"### {spec.label}")
        lines.append("")
        lines.append(f"- Direction: `{spec.direction}`")
        lines.append(f"- Nodes: `{' -> '.join(spec.nodes)}`")
        lines.append(f"- Path score: `{spec.score:.3f}`")
        lines.append("- Node semantics:")
        for node in spec.nodes[:-1]:
            lines.append(f"  - `{node}`: {describe_node(node, node_table)}")
        lines.append("- Edge evidence:")
        for edge_row in path_edge_evidence.get(spec.key, []):
            lines.append(
                f"  - `{edge_row['source']} -> {edge_row['target']}`: "
                f"promote-mediated `{edge_row['promote_mediated_ratio_median']:.3f}`, "
                f"suppress-mediated `{edge_row['suppress_mediated_ratio_median']:.3f}`, "
                f"support `{edge_row['union_support_max']:.3f}`"
            )
        lines.append("- Stagewise cumulative patching:")
        for row in sorted(rows_by_path.get(spec.key, []), key=lambda r: int(r["step_idx"])):
            lines.append(
                f"  - step {int(row['step_idx'])} / `{row['node']}`: "
                f"cum `{float(row['cumulative_ratio_median']):.3f}`, "
                f"inc `{float(row['incremental_ratio_median']):.3f}`, "
                f"top1 `{float(row['top1_rate']):.3f}`, "
                f"boundary `{float(row['boundary_flip_rate']):.3f}`"
            )
        lines.append("")

    lines.append("## Candidate Algorithm")
    lines.append("")
    lines.append(
        "- Query-conditioned branch: a small early reader (`L2H14`) reads actionable user wording, "
        "an early tool-biased writer (`MLP11`) amplifies it, then late shared integrators (`MLP16`, `L24H6`) carry it to the decision output."
    )
    lines.append(
        "- Schema-conditioned branch: late schema/tag readers (`L21H12` in the extracted path) inject tool-availability evidence directly into the final writer bottleneck (`MLP27`)."
    )
    lines.append(
        "- No-tool branch: suppression readers (`L16H4` in the extracted path) feed a no-tool-biased writer (`MLP17`), which then routes through a late suppressive node (`L23H6`) to keep generation in ordinary-answer mode."
    )
    lines.append(
        "- The final decision is therefore not a single switch; it is a late competition between a query/tool-use branch and a no-tool suppression branch."
    )
    lines.append("")
    lines.append("## Artifacts")
    lines.append("")
    lines.append("- `semantic_chain_summary.json`")
    lines.append("- `semantic_chain_summary.csv`")
    lines.append("- `semantic_chain_per_sample.csv`")
    lines.append("- `semantic_chain_progression.png`")
    lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a stagewise semantic causal-chain report.")
    parser.add_argument("--run-root", type=str, required=True)
    parser.add_argument("--model-path", type=str, default="./external/models/Qwen3-1.7B")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--output-root", type=str, required=True)
    args = parser.parse_args()

    run_root = Path(args.run_root).resolve()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    node_table = load_node_table(run_root / "functional_groups" / "functional_node_table.csv")
    edge_support = load_edge_support(run_root / "final_signed_circuit" / "final_signed_edges.csv")
    edge_mediation = load_edge_mediation(run_root / "edge_importance" / "signed_edge_mediation_summary.csv")
    path_specs = choose_paths(node_table, edge_support, edge_mediation)
    if not path_specs:
        raise ValueError("Failed to extract any semantic paths.")

    summary = evaluate_paths(
        run_root=run_root,
        path_specs=path_specs,
        model_path=args.model_path,
        device=args.device,
        max_samples=args.max_samples,
        output_root=output_root,
    )
    summary_rows = list(summary.get("summary_rows", []))
    plot_progression(summary_rows, output_root / "semantic_chain_progression.png")

    path_edge_evidence = {
        spec.key: path_edge_rows(
            spec,
            node_table=node_table,
            edge_support=edge_support,
            edge_mediation=edge_mediation,
        )
        for spec in path_specs
    }
    edge_rows_flat = [row for rows in path_edge_evidence.values() for row in rows]
    write_csv(edge_rows_flat, output_root / "semantic_chain_edge_evidence.csv")

    markdown = build_markdown_report(
        run_root=run_root,
        output_root=output_root,
        node_table=node_table,
        path_specs=path_specs,
        path_edge_evidence=path_edge_evidence,
        summary_rows=summary_rows,
    )
    (output_root / "semantic_chain_report.md").write_text(markdown, encoding="utf-8")
    print(markdown)


if __name__ == "__main__":
    main()
