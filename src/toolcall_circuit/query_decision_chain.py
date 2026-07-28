#!/usr/bin/env python3
"""
Fixed-schema query decision-chain audit.

This analysis deliberately holds schema/protocol constant by comparing
`clean_full` against `corrupt_full`, which share the same system prompt.

The question it answers is narrower than the earlier factorized analysis:
with the tool schema/protocol already present, how does user-side information
reach the late writer path, and how does the competing no-tool chain suppress
that path?
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
from typing import Dict, Iterable, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm.auto import tqdm

if __package__ in {None, ""}:
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from toolcall_circuit.bidirectional_causal_eval import collect_cache_cpu_for_nodes, load_sample_paths
from toolcall_circuit.objective import build_bidirectional_endpoint_objectives
from toolcall_circuit.semantic_factorized_counterfactual import head_mass_summary, safe_float
from toolcall_circuit.signed_edge_importance import run_logits_with_assignments
from toolcall_circuit.single_sample import load_hooked_qwen3, objective_from_logits


QUERY_COMPONENTS = ["L20H5", "L21H1", "L21H12", "L24H6", "MLP27"]
SUPPRESS_COMPONENTS = ["L16H4", "MLP17", "L23H6"]
READ_HEADS = ["L20H5", "L21H1", "L21H12", "L16H4"]

QUERY_STEPS = [
    ("user_query_read", ["L20H5"]),
    ("query_to_l21h1", ["L20H5", "L21H1"]),
    ("query_to_l21h12", ["L20H5", "L21H12"]),
    ("query_to_both_l21", ["L20H5", "L21H1", "L21H12"]),
    ("query_to_late_router", ["L20H5", "L21H1", "L21H12", "L24H6"]),
    ("query_to_late_writer", ["L20H5", "L21H1", "L21H12", "L24H6", "MLP27"]),
]

SUPPRESS_STEPS = [
    ("no_tool_read", ["L16H4"]),
    ("no_tool_writer", ["L16H4", "MLP17"]),
    ("no_tool_late_relay", ["L16H4", "MLP17", "L23H6"]),
]

QUERY_EDGES = [
    ("L20H5", "L21H1"),
    ("L20H5", "L21H12"),
    ("L21H1", "MLP27"),
    ("L21H12", "MLP27"),
]

SUPPRESS_EDGES = [
    ("L16H4", "MLP17"),
    ("MLP17", "L23H6"),
    ("MLP17", "L20H5"),
    ("MLP17", "L21H1"),
    ("MLP17", "L21H12"),
]


def write_csv(rows: Sequence[Dict[str, object]], out_path: Path) -> None:
    if not rows:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def median(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if isinstance(v, (int, float)) and math.isfinite(float(v))]
    return float(np.median(vals)) if vals else float("nan")


def safe_rate(values: Iterable[bool]) -> float:
    vals = [1.0 if bool(v) else 0.0 for v in values]
    return float(np.mean(vals)) if vals else float("nan")


def plot_stepwise(summary_rows: Sequence[Dict[str, object]], out_path: Path) -> None:
    families = []
    for row in summary_rows:
        fam = str(row["family"])
        if fam not in families:
            families.append(fam)
    plt.style.use("default")
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8), constrained_layout=True)
    for family in families:
        rows = [r for r in summary_rows if str(r["family"]) == family]
        rows = sorted(rows, key=lambda r: int(r["step_idx"]))
        xs = np.arange(len(rows))
        axes[0].plot(xs, [float(r["decision_score_median"]) for r in rows], marker="o", label=family)
        axes[1].plot(xs, [float(r["top1_rate"]) for r in rows], marker="o", label=family)
    axes[0].axhline(0.0, color="#888888", linewidth=1.0)
    axes[0].set_title("Fixed-Schema Stepwise Decision Score")
    axes[0].set_ylabel("median decision score")
    axes[1].set_title("Fixed-Schema Stepwise Top-1 Rate")
    axes[1].set_ylabel("top-1 rate")
    for ax in axes:
        ax.set_xlabel("step index")
        ax.legend(frameon=False)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fixed-schema query decision-chain audit.")
    parser.add_argument("--run-root", type=str, required=True)
    parser.add_argument("--model-path", type=str, default="./external/models/Qwen3-1.7B")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--output-root", type=str, required=True)
    args = parser.parse_args()

    run_root = Path(args.run_root).resolve()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    all_nodes = sorted(set(QUERY_COMPONENTS + SUPPRESS_COMPONENTS))
    samples = load_sample_paths(run_root / "forward_batch", run_root / "reverse_batch", max_samples=args.max_samples)
    model, tokenizer = load_hooked_qwen3(args.model_path, device=args.device, dtype=torch.bfloat16)

    component_rows: List[Dict[str, object]] = []
    step_rows: List[Dict[str, object]] = []
    edge_rows: List[Dict[str, object]] = []
    read_rows: List[Dict[str, object]] = []

    pbar = tqdm(samples, desc="Query decision chain", dynamic_ncols=True)
    for sp in pbar:
        try:
            clean_text = sp.tool_call_prompt.read_text(encoding="utf-8")
            corrupt_text = sp.no_tool_prompt.read_text(encoding="utf-8")
        except Exception:
            continue

        clean_tokens = model.to_tokens(clean_text, prepend_bos=False)
        corrupt_tokens = model.to_tokens(corrupt_text, prepend_bos=False)
        with torch.no_grad():
            clean_logits = model(clean_tokens)
            corrupt_logits = model(corrupt_tokens)

        tool_objective, no_tool_objective = build_bidirectional_endpoint_objectives(
            clean_logits,
            corrupt_logits,
            tokenizer=tokenizer,
        )
        clean_tool = float(objective_from_logits(clean_logits, tool_objective).item())
        corrupt_tool = float(objective_from_logits(corrupt_logits, tool_objective).item())
        clean_no_tool = float(objective_from_logits(clean_logits, no_tool_objective).item())
        corrupt_no_tool = float(objective_from_logits(corrupt_logits, no_tool_objective).item())
        tool_gap = clean_tool - corrupt_tool
        no_tool_gap = corrupt_no_tool - clean_no_tool
        if (
            not math.isfinite(tool_gap)
            or abs(tool_gap) < 1e-8
            or not math.isfinite(no_tool_gap)
            or abs(no_tool_gap) < 1e-8
        ):
            continue

        clean_cache = collect_cache_cpu_for_nodes(model, clean_tokens, all_nodes)
        corrupt_cache = collect_cache_cpu_for_nodes(model, corrupt_tokens, all_nodes)

        for variant_name, text in [("clean_full", clean_text), ("corrupt_full", corrupt_text)]:
            masses = head_mass_summary(model, tokenizer, text, READ_HEADS)
            for head in READ_HEADS:
                for set_name in ["user_block", "tools_payload", "protocol_payload", "tool_call_tags"]:
                    read_rows.append(
                        {
                            "sample_id": sp.sample_id,
                            "variant": variant_name,
                            "component": head,
                            "set": set_name,
                            "mass": masses.get((head, set_name), float("nan")),
                        }
                    )

        for component in QUERY_COMPONENTS:
            patched_logits = run_logits_with_assignments(
                model,
                corrupt_tokens,
                clean_cache,
                corrupt_cache,
                [component],
                [],
            )
            patched_tool = float(objective_from_logits(patched_logits, tool_objective).item())
            patched_no_tool = float(objective_from_logits(patched_logits, no_tool_objective).item())
            component_rows.append(
                {
                    "sample_id": sp.sample_id,
                    "family": "query",
                    "component": component,
                    "base_variant": "corrupt_full",
                    "rescue_ratio": (patched_tool - corrupt_tool) / tool_gap,
                    "decision_score": patched_tool - patched_no_tool,
                    "top1_success": int(patched_logits[0, -1].argmax().item()) == sp.target_tool_call,
                    "boundary_flip": patched_tool > patched_no_tool,
                }
            )
        for component in SUPPRESS_COMPONENTS:
            patched_logits = run_logits_with_assignments(
                model,
                clean_tokens,
                corrupt_cache,
                clean_cache,
                [component],
                [],
            )
            patched_tool = float(objective_from_logits(patched_logits, tool_objective).item())
            patched_no_tool = float(objective_from_logits(patched_logits, no_tool_objective).item())
            component_rows.append(
                {
                    "sample_id": sp.sample_id,
                    "family": "suppress",
                    "component": component,
                    "base_variant": "clean_full",
                    "rescue_ratio": (patched_no_tool - clean_no_tool) / no_tool_gap,
                    "decision_score": patched_no_tool - patched_tool,
                    "top1_success": int(patched_logits[0, -1].argmax().item()) == sp.distractor,
                    "boundary_flip": patched_no_tool > patched_tool,
                }
            )

        for step_idx, (label, nodes) in enumerate(QUERY_STEPS, start=1):
            patched_logits = run_logits_with_assignments(
                model,
                corrupt_tokens,
                clean_cache,
                corrupt_cache,
                nodes,
                [],
            )
            patched_tool = float(objective_from_logits(patched_logits, tool_objective).item())
            patched_no_tool = float(objective_from_logits(patched_logits, no_tool_objective).item())
            step_rows.append(
                {
                    "sample_id": sp.sample_id,
                    "family": "query",
                    "step_idx": step_idx,
                    "step_label": label,
                    "nodes": "|".join(nodes),
                    "rescue_ratio": (patched_tool - corrupt_tool) / tool_gap,
                    "decision_score": patched_tool - patched_no_tool,
                    "top1_success": int(patched_logits[0, -1].argmax().item()) == sp.target_tool_call,
                    "boundary_flip": patched_tool > patched_no_tool,
                }
            )
        for step_idx, (label, nodes) in enumerate(SUPPRESS_STEPS, start=1):
            patched_logits = run_logits_with_assignments(
                model,
                clean_tokens,
                corrupt_cache,
                clean_cache,
                nodes,
                [],
            )
            patched_tool = float(objective_from_logits(patched_logits, tool_objective).item())
            patched_no_tool = float(objective_from_logits(patched_logits, no_tool_objective).item())
            step_rows.append(
                {
                    "sample_id": sp.sample_id,
                    "family": "suppress",
                    "step_idx": step_idx,
                    "step_label": label,
                    "nodes": "|".join(nodes),
                    "rescue_ratio": (patched_no_tool - clean_no_tool) / no_tool_gap,
                    "decision_score": patched_no_tool - patched_tool,
                    "top1_success": int(patched_logits[0, -1].argmax().item()) == sp.distractor,
                    "boundary_flip": patched_no_tool > patched_tool,
                }
            )

        for source, target in QUERY_EDGES:
            source_logits = run_logits_with_assignments(
                model,
                corrupt_tokens,
                clean_cache,
                corrupt_cache,
                [source],
                [],
            )
            blocked_logits = run_logits_with_assignments(
                model,
                corrupt_tokens,
                clean_cache,
                corrupt_cache,
                [source],
                [target],
            )
            source_ratio = (float(objective_from_logits(source_logits, tool_objective).item()) - corrupt_tool) / tool_gap
            blocked_ratio = (float(objective_from_logits(blocked_logits, tool_objective).item()) - corrupt_tool) / tool_gap
            edge_rows.append(
                {
                    "sample_id": sp.sample_id,
                    "family": "query",
                    "edge": f"{source}->{target}",
                    "source_ratio": source_ratio,
                    "blocked_ratio": blocked_ratio,
                    "mediated_ratio": source_ratio - blocked_ratio,
                }
            )
        for source, target in SUPPRESS_EDGES:
            source_logits = run_logits_with_assignments(
                model,
                clean_tokens,
                corrupt_cache,
                clean_cache,
                [source],
                [],
            )
            blocked_logits = run_logits_with_assignments(
                model,
                clean_tokens,
                corrupt_cache,
                clean_cache,
                [source],
                [target],
            )
            source_ratio = (float(objective_from_logits(source_logits, no_tool_objective).item()) - clean_no_tool) / no_tool_gap
            blocked_ratio = (float(objective_from_logits(blocked_logits, no_tool_objective).item()) - clean_no_tool) / no_tool_gap
            edge_rows.append(
                {
                    "sample_id": sp.sample_id,
                    "family": "suppress",
                    "edge": f"{source}->{target}",
                    "source_ratio": source_ratio,
                    "blocked_ratio": blocked_ratio,
                    "mediated_ratio": source_ratio - blocked_ratio,
                }
            )
        pbar.set_postfix(sample=sp.sample_id)

    write_csv(component_rows, output_root / "query_decision_component_per_sample.csv")
    write_csv(step_rows, output_root / "query_decision_stepwise_per_sample.csv")
    write_csv(edge_rows, output_root / "query_decision_edge_per_sample.csv")
    write_csv(read_rows, output_root / "query_decision_head_reads_per_sample.csv")

    component_summary: List[Dict[str, object]] = []
    for (family, component), rows in sorted(defaultdict(list, {
        (str(r["family"]), str(r["component"])): [x for x in component_rows if str(x["family"]) == str(r["family"]) and str(x["component"]) == str(r["component"])]
        for r in component_rows
    }).items()):
        component_summary.append(
            {
                "family": family,
                "component": component,
                "n_samples": len(rows),
                "rescue_ratio_median": median(safe_float(r["rescue_ratio"]) for r in rows),
                "decision_score_median": median(safe_float(r["decision_score"]) for r in rows),
                "top1_rate": safe_rate(bool(r["top1_success"]) for r in rows),
                "boundary_flip_rate": safe_rate(bool(r["boundary_flip"]) for r in rows),
            }
        )

    step_summary: List[Dict[str, object]] = []
    grouped_steps: Dict[tuple[str, int, str, str], List[Dict[str, object]]] = defaultdict(list)
    for row in step_rows:
        grouped_steps[(str(row["family"]), int(row["step_idx"]), str(row["step_label"]), str(row["nodes"]))].append(row)
    for (family, step_idx, step_label, nodes), rows in sorted(grouped_steps.items()):
        step_summary.append(
            {
                "family": family,
                "step_idx": step_idx,
                "step_label": step_label,
                "nodes": nodes,
                "n_samples": len(rows),
                "rescue_ratio_median": median(safe_float(r["rescue_ratio"]) for r in rows),
                "decision_score_median": median(safe_float(r["decision_score"]) for r in rows),
                "top1_rate": safe_rate(bool(r["top1_success"]) for r in rows),
                "boundary_flip_rate": safe_rate(bool(r["boundary_flip"]) for r in rows),
            }
        )

    edge_summary: List[Dict[str, object]] = []
    grouped_edges: Dict[tuple[str, str], List[Dict[str, object]]] = defaultdict(list)
    for row in edge_rows:
        grouped_edges[(str(row["family"]), str(row["edge"]))].append(row)
    for (family, edge), rows in sorted(grouped_edges.items()):
        edge_summary.append(
            {
                "family": family,
                "edge": edge,
                "n_samples": len(rows),
                "source_ratio_median": median(safe_float(r["source_ratio"]) for r in rows),
                "blocked_ratio_median": median(safe_float(r["blocked_ratio"]) for r in rows),
                "mediated_ratio_median": median(safe_float(r["mediated_ratio"]) for r in rows),
            }
        )

    read_summary: List[Dict[str, object]] = []
    grouped_reads: Dict[tuple[str, str, str], List[Dict[str, object]]] = defaultdict(list)
    for row in read_rows:
        grouped_reads[(str(row["variant"]), str(row["component"]), str(row["set"]))].append(row)
    for (variant, component, set_name), rows in sorted(grouped_reads.items()):
        read_summary.append(
            {
                "variant": variant,
                "component": component,
                "set": set_name,
                "mass_median": median(safe_float(r["mass"]) for r in rows),
            }
        )

    write_csv(component_summary, output_root / "query_decision_component_summary.csv")
    write_csv(step_summary, output_root / "query_decision_stepwise_summary.csv")
    write_csv(edge_summary, output_root / "query_decision_edge_summary.csv")
    write_csv(read_summary, output_root / "query_decision_head_reads_summary.csv")
    plot_stepwise(step_summary, output_root / "query_decision_stepwise.png")

    query_final = next((r for r in step_summary if r["family"] == "query" and r["step_label"] == "query_to_late_writer"), None)
    suppress_final = next((r for r in step_summary if r["family"] == "suppress" and r["step_label"] == "no_tool_late_relay"), None)
    l20 = next((r for r in component_summary if r["family"] == "query" and r["component"] == "L20H5"), None)
    l21h1 = next((r for r in component_summary if r["family"] == "query" and r["component"] == "L21H1"), None)
    l21h12 = next((r for r in component_summary if r["family"] == "query" and r["component"] == "L21H12"), None)
    mlp27 = next((r for r in component_summary if r["family"] == "query" and r["component"] == "MLP27"), None)
    mlp17 = next((r for r in component_summary if r["family"] == "suppress" and r["component"] == "MLP17"), None)
    edge_l20_l21h12 = next((r for r in edge_summary if r["edge"] == "L20H5->L21H12"), None)
    edge_l20_l21h1 = next((r for r in edge_summary if r["edge"] == "L20H5->L21H1"), None)
    edge_l21h12_mlp27 = next((r for r in edge_summary if r["edge"] == "L21H12->MLP27"), None)
    edge_mlp17_l20 = next((r for r in edge_summary if r["edge"] == "MLP17->L20H5"), None)

    summary = {
        "component_summary_rows": component_summary,
        "step_summary_rows": step_summary,
        "edge_summary_rows": edge_summary,
        "read_summary_rows": read_summary,
        "key_findings": {
            "query_l20h5_rescue_median": l20["rescue_ratio_median"] if l20 else float("nan"),
            "query_l21h1_rescue_median": l21h1["rescue_ratio_median"] if l21h1 else float("nan"),
            "query_l21h12_rescue_median": l21h12["rescue_ratio_median"] if l21h12 else float("nan"),
            "query_mlp27_rescue_median": mlp27["rescue_ratio_median"] if mlp27 else float("nan"),
            "query_chain_final_top1_rate": query_final["top1_rate"] if query_final else float("nan"),
            "suppress_mlp17_rescue_median": mlp17["rescue_ratio_median"] if mlp17 else float("nan"),
            "suppress_chain_final_top1_rate": suppress_final["top1_rate"] if suppress_final else float("nan"),
            "edge_l20h5_l21h12_mediated": edge_l20_l21h12["mediated_ratio_median"] if edge_l20_l21h12 else float("nan"),
            "edge_l20h5_l21h1_mediated": edge_l20_l21h1["mediated_ratio_median"] if edge_l20_l21h1 else float("nan"),
            "edge_l21h12_mlp27_mediated": edge_l21h12_mlp27["mediated_ratio_median"] if edge_l21h12_mlp27 else float("nan"),
            "edge_mlp17_l20h5_mediated": edge_mlp17_l20["mediated_ratio_median"] if edge_mlp17_l20 else float("nan"),
        },
        "artifacts": {
            "component_summary_csv": str(output_root / "query_decision_component_summary.csv"),
            "step_summary_csv": str(output_root / "query_decision_stepwise_summary.csv"),
            "edge_summary_csv": str(output_root / "query_decision_edge_summary.csv"),
            "read_summary_csv": str(output_root / "query_decision_head_reads_summary.csv"),
            "step_plot": str(output_root / "query_decision_stepwise.png"),
        },
    }
    (output_root / "query_decision_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = ["# Fixed-Schema Query Decision Chain", ""]
    lines.append("This report only compares `clean_full` vs `corrupt_full`, which keep the same schema/protocol and change the user-side prompt.")
    lines.append("")
    if l20 and l21h1 and l21h12 and mlp27:
        lines.append("## Query-Side Route")
        lines.append("")
        lines.append(
            f"- `L20H5` direct clean->corrupt rescue median `{float(l20['rescue_ratio_median']):.3f}`, "
            f"top1 `{float(l20['top1_rate']):.3f}`."
        )
        lines.append(
            f"- `L21H1` direct rescue median `{float(l21h1['rescue_ratio_median']):.3f}`; "
            f"`L21H12` direct rescue median `{float(l21h12['rescue_ratio_median']):.3f}`."
        )
        lines.append(
            f"- `MLP27` direct rescue median `{float(mlp27['rescue_ratio_median']):.3f}`, "
            f"top1 `{float(mlp27['top1_rate']):.3f}`."
        )
    if query_final:
        lines.append(
            f"- Cumulative query route `{query_final['nodes']}` reaches decision `{float(query_final['decision_score_median']):.3f}` "
            f"and tool-top1 `{float(query_final['top1_rate']):.3f}` on `corrupt_full`."
        )
    if edge_l20_l21h1 and edge_l20_l21h12 and edge_l21h12_mlp27:
        lines.append(
            f"- Path mediation: `L20H5->L21H1` `{float(edge_l20_l21h1['mediated_ratio_median']):.3f}`, "
            f"`L20H5->L21H12` `{float(edge_l20_l21h12['mediated_ratio_median']):.3f}`, "
            f"`L21H12->MLP27` `{float(edge_l21h12_mlp27['mediated_ratio_median']):.3f}`."
        )
    lines.append("")
    if mlp17 and suppress_final:
        lines.append("## Competing No-Tool Route")
        lines.append("")
        lines.append(
            f"- `MLP17` direct corrupt->clean no-tool rescue median `{float(mlp17['rescue_ratio_median']):.3f}`."
        )
        lines.append(
            f"- Cumulative suppress route `{suppress_final['nodes']}` reaches decision `{float(suppress_final['decision_score_median']):.3f}` "
            f"and no-tool top1 `{float(suppress_final['top1_rate']):.3f}` on `clean_full`."
        )
    if edge_mlp17_l20:
        lines.append(
            f"- `MLP17->L20H5` mediation `{float(edge_mlp17_l20['mediated_ratio_median']):.3f}`, consistent with the no-tool chain suppressing a user-to-tool ingress point."
        )
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append(
        "- Under fixed schema/protocol, the user-side difference is not expressed through an early isolated reader. "
        "Instead, it enters the late tool path at `L20H5`, is routed through `L21H1/L21H12`, and is then written out by `MLP27`."
    )
    lines.append(
        "- The competing no-tool route can push the same fixed-schema prompt back toward `no_tool` by writing through `MLP17` and suppressing downstream late tool ingress."
    )
    (output_root / "query_decision_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
