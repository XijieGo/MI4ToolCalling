#!/usr/bin/env python3
"""
Node-level necessity analysis inside the final signed circuit.
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
from toolcall_circuit.bidirectional_token_flip import run_logits_on_base_with_source
from toolcall_circuit.objective import build_bidirectional_endpoint_objectives
from toolcall_circuit.signed_circuit import derive_signed_groups
from toolcall_circuit.single_sample import load_hooked_qwen3, objective_from_logits


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


def plot_node_importance(rows: Sequence[Dict[str, object]], out_path: Path) -> None:
    labels = [str(r["node"]) for r in rows]
    mat = np.array(
        [
            [
                float(r["promote_nec_drop_median"]),
                float(r["suppress_nec_drop_median"]),
                float(r["promote_suff_ratio_median"]),
                float(r["suppress_suff_ratio_median"]),
            ]
            for r in rows
        ],
        dtype=float,
    )
    cols = ["promote nec", "suppress nec", "promote suff", "suppress suff"]
    finite_vals = mat[np.isfinite(mat)]
    vmax = max(float(np.percentile(np.abs(finite_vals), 98)), 1e-6) if finite_vals.size else 1.0
    plt.style.use("default")
    fig, ax = plt.subplots(figsize=(8.8, max(6.0, 0.44 * len(labels) + 1.8)), constrained_layout=True)
    im = ax.imshow(mat, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(np.arange(len(cols)))
    ax.set_xticklabels(cols, rotation=25, ha="right")
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_title("Signed Circuit Node Importance")
    cbar = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
    cbar.set_label("drop / ratio")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Node-level necessity analysis for the signed circuit.")
    parser.add_argument("--forward-batch-root", type=str, required=True)
    parser.add_argument("--bidirectional-summary", type=str, required=True)
    parser.add_argument("--signed-nodes-csv", type=str, required=True)
    parser.add_argument("--model-path", type=str, default="./external/models/Qwen3-1.7B")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max-samples", type=int, default=200)
    parser.add_argument("--output-root", type=str, required=True)
    args = parser.parse_args()

    out_root = Path(args.output_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    bidi = json.loads(Path(args.bidirectional_summary).resolve().read_text(encoding="utf-8"))
    groups = derive_signed_groups(bidi)
    final_nodes = sorted({n for members in groups.values() for n in members})
    node_rows = read_csv_rows(Path(args.signed_nodes_csv).resolve())
    node_meta = {str(r["node"]): r for r in node_rows}

    forward_batch_root = Path(args.forward_batch_root).resolve()
    reverse_batch_root = Path(args.bidirectional_summary).resolve().parent.parent / "reverse_batch"
    samples = load_sample_paths(forward_batch_root, reverse_batch_root, max_samples=args.max_samples)

    model, tokenizer = load_hooked_qwen3(args.model_path, device=args.device, dtype=torch.bfloat16)
    _ = tokenizer

    per_sample_rows: List[Dict[str, object]] = []
    pbar = tqdm(samples, desc="Signed node importance", dynamic_ncols=True)
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
        m_tool_tool = float(objective_from_logits(tool_logits, tool_objective).item())
        m_tool_no = float(objective_from_logits(no_tool_logits, tool_objective).item())
        gap_tool = m_tool_tool - m_tool_no
        m_no_tool_tool = float(objective_from_logits(tool_logits, no_tool_objective).item())
        m_no_tool_no = float(objective_from_logits(no_tool_logits, no_tool_objective).item())
        gap_no_tool = m_no_tool_no - m_no_tool_tool
        if (
            not math.isfinite(gap_tool)
            or abs(gap_tool) < 1e-8
            or not math.isfinite(gap_no_tool)
            or abs(gap_no_tool) < 1e-8
        ):
            continue

        tool_cache = collect_cache_cpu_for_nodes(model, tool_tokens, final_nodes)
        no_tool_cache = collect_cache_cpu_for_nodes(model, no_tool_tokens, final_nodes)

        full_promote_logits = run_logits_on_base_with_source(model, no_tool_tokens, tool_cache, final_nodes)
        full_promote_margin = float(objective_from_logits(full_promote_logits, tool_objective).item())
        full_promote_ratio = (full_promote_margin - m_tool_no) / gap_tool
        full_promote_top1 = int(full_promote_logits[0, -1].argmax().item()) == sp.target_tool_call

        full_suppress_logits = run_logits_on_base_with_source(model, tool_tokens, no_tool_cache, final_nodes)
        full_suppress_margin = float(objective_from_logits(full_suppress_logits, no_tool_objective).item())
        full_suppress_ratio = (full_suppress_margin - m_no_tool_tool) / gap_no_tool
        full_suppress_top1 = int(full_suppress_logits[0, -1].argmax().item()) == sp.distractor

        for node in final_nodes:
            node_suff_promote_logits = run_logits_on_base_with_source(model, no_tool_tokens, tool_cache, [node])
            node_suff_promote_margin = float(objective_from_logits(node_suff_promote_logits, tool_objective).item())
            node_suff_promote_ratio = (node_suff_promote_margin - m_tool_no) / gap_tool
            node_suff_promote_top1 = int(node_suff_promote_logits[0, -1].argmax().item()) == sp.target_tool_call

            node_suff_suppress_logits = run_logits_on_base_with_source(model, tool_tokens, no_tool_cache, [node])
            node_suff_suppress_margin = float(objective_from_logits(node_suff_suppress_logits, no_tool_objective).item())
            node_suff_suppress_ratio = (node_suff_suppress_margin - m_no_tool_tool) / gap_no_tool
            node_suff_suppress_top1 = int(node_suff_suppress_logits[0, -1].argmax().item()) == sp.distractor

            minus_nodes = [n for n in final_nodes if n != node]
            minus_promote_logits = run_logits_on_base_with_source(model, no_tool_tokens, tool_cache, minus_nodes)
            minus_promote_margin = float(objective_from_logits(minus_promote_logits, tool_objective).item())
            minus_promote_ratio = (minus_promote_margin - m_tool_no) / gap_tool
            minus_promote_top1 = int(minus_promote_logits[0, -1].argmax().item()) == sp.target_tool_call

            minus_suppress_logits = run_logits_on_base_with_source(model, tool_tokens, no_tool_cache, minus_nodes)
            minus_suppress_margin = float(objective_from_logits(minus_suppress_logits, no_tool_objective).item())
            minus_suppress_ratio = (minus_suppress_margin - m_no_tool_tool) / gap_no_tool
            minus_suppress_top1 = int(minus_suppress_logits[0, -1].argmax().item()) == sp.distractor

            meta = node_meta.get(node, {})
            per_sample_rows.append(
                {
                    "sample_id": sp.sample_id,
                    "node": node,
                    "group": str(meta.get("group_key", "")),
                    "group_label": str(meta.get("group_label", "")),
                    "semantic_hint": str(meta.get("semantic_hint", "")),
                    "promote_suff_ratio": node_suff_promote_ratio,
                    "suppress_suff_ratio": node_suff_suppress_ratio,
                    "promote_suff_top1": node_suff_promote_top1,
                    "suppress_suff_top1": node_suff_suppress_top1,
                    "promote_nec_drop": full_promote_ratio - minus_promote_ratio,
                    "suppress_nec_drop": minus_suppress_ratio - full_suppress_ratio,
                    "promote_nec_top1_drop": float(full_promote_top1) - float(minus_promote_top1),
                    "suppress_nec_top1_drop": float(full_suppress_top1) - float(minus_suppress_top1),
                }
            )

        pbar.set_postfix(sample=sp.sample_id)

    by_node: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in per_sample_rows:
        by_node[str(row["node"])].append(row)

    summary_rows: List[Dict[str, object]] = []
    for node in final_nodes:
        rows = by_node[node]
        meta = node_meta.get(node, {})
        summary_rows.append(
            {
                "node": node,
                "group": str(meta.get("group_key", "")),
                "group_label": str(meta.get("group_label", "")),
                "semantic_hint": str(meta.get("semantic_hint", "")),
                "n_samples": len(rows),
                "promote_nec_drop_median": median(r["promote_nec_drop"] for r in rows),
                "suppress_nec_drop_median": median(r["suppress_nec_drop"] for r in rows),
                "promote_nec_top1_drop_mean": mean(r["promote_nec_top1_drop"] for r in rows),
                "suppress_nec_top1_drop_mean": mean(r["suppress_nec_top1_drop"] for r in rows),
                "promote_suff_ratio_median": median(r["promote_suff_ratio"] for r in rows),
                "suppress_suff_ratio_median": median(r["suppress_suff_ratio"] for r in rows),
                "promote_suff_top1_rate": safe_rate(r["promote_suff_top1"] for r in rows),
                "suppress_suff_top1_rate": safe_rate(r["suppress_suff_top1"] for r in rows),
            }
        )

    summary_rows.sort(
        key=lambda r: (
            abs(float(r["promote_nec_drop_median"])) + abs(float(r["suppress_nec_drop_median"])),
            abs(float(r["promote_suff_ratio_median"])) + abs(float(r["suppress_suff_ratio_median"])),
        ),
        reverse=True,
    )

    write_csv(per_sample_rows, out_root / "signed_node_importance_per_sample.csv")
    write_csv(summary_rows, out_root / "signed_node_importance_summary.csv")
    plot_node_importance(summary_rows, out_root / "signed_node_importance_heatmap.png")

    summary = {
        "n_samples": len(samples),
        "artifacts": {
            "per_sample_csv": str(out_root / "signed_node_importance_per_sample.csv"),
            "summary_csv": str(out_root / "signed_node_importance_summary.csv"),
            "summary_json": str(out_root / "signed_node_importance_report.json"),
            "heatmap": str(out_root / "signed_node_importance_heatmap.png"),
        },
        "summary_rows": summary_rows,
    }
    (out_root / "signed_node_importance_report.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
