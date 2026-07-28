#!/usr/bin/env python3
"""
Composition analysis for the final signed circuit.

This script quantifies how the symmetric backbone changes once directional bias
groups are added back in. The goal is to turn the "shared backbone + bias"
story into direct causal comparisons.
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


def write_csv(rows: Sequence[Dict[str, object]], out_path: Path) -> None:
    if not rows:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_composition_summary(rows: Sequence[Dict[str, object]], out_path: Path) -> None:
    labels = [str(r["combo"]) for r in rows]
    mat = np.array(
        [
            [
                float(r["promote_ratio_median"]),
                float(r["suppress_ratio_median"]),
                float(r["promote_gain_vs_symmetric_median"]),
                float(r["suppress_gain_vs_symmetric_median"]),
            ]
            for r in rows
        ],
        dtype=float,
    )
    cols = ["promote ratio", "suppress ratio", "promote gain", "suppress gain"]
    finite_vals = mat[np.isfinite(mat)]
    vmax = max(float(np.percentile(np.abs(finite_vals), 98)), 1e-6) if finite_vals.size else 1.0
    plt.style.use("default")
    fig, ax = plt.subplots(figsize=(8.8, max(4.8, 0.75 * len(labels) + 1.8)), constrained_layout=True)
    im = ax.imshow(mat, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(np.arange(len(cols)))
    ax.set_xticklabels(cols, rotation=25, ha="right")
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_title("Signed Circuit Composition")
    cbar = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
    cbar.set_label("ratio / gain")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def make_combo_map(groups: Dict[str, List[str]]) -> Dict[str, List[str]]:
    symmetric = groups.get("symmetric_backbone", [])
    tool_bias = groups.get("tool_bias_backbone", [])
    no_tool_bias = groups.get("no_tool_bias_backbone", [])
    tool_tail = groups.get("tool_tail", [])
    no_tool_tail = groups.get("no_tool_tail", [])

    combos = {
        "symmetric_backbone": list(symmetric),
        "symmetric_plus_tool_bias": sorted(set(symmetric) | set(tool_bias)),
        "symmetric_plus_no_tool_bias": sorted(set(symmetric) | set(no_tool_bias)),
        "symmetric_plus_tool_mode": sorted(set(symmetric) | set(tool_bias) | set(tool_tail)),
        "symmetric_plus_no_tool_mode": sorted(set(symmetric) | set(no_tool_bias) | set(no_tool_tail)),
        "full_signed_circuit": sorted({n for members in groups.values() for n in members}),
    }
    return {k: v for k, v in combos.items() if v}


def main() -> None:
    parser = argparse.ArgumentParser(description="Composition analysis for the signed decision circuit.")
    parser.add_argument("--forward-batch-root", type=str, required=True)
    parser.add_argument("--bidirectional-summary", type=str, required=True)
    parser.add_argument("--model-path", type=str, default="./external/models/Qwen3-1.7B")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--output-root", type=str, required=True)
    args = parser.parse_args()

    out_root = Path(args.output_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    bidi = json.loads(Path(args.bidirectional_summary).resolve().read_text(encoding="utf-8"))
    groups = derive_signed_groups(bidi)
    combos = make_combo_map(groups)
    if "symmetric_backbone" not in combos:
        raise ValueError("Missing symmetric_backbone in signed groups.")

    forward_batch_root = Path(args.forward_batch_root).resolve()
    reverse_batch_root = Path(args.bidirectional_summary).resolve().parent.parent / "reverse_batch"
    samples = load_sample_paths(forward_batch_root, reverse_batch_root, max_samples=args.max_samples)

    all_nodes = sorted({n for members in combos.values() for n in members})
    model, tokenizer = load_hooked_qwen3(args.model_path, device=args.device, dtype=torch.bfloat16)
    _ = tokenizer

    per_sample_rows: List[Dict[str, object]] = []
    pbar = tqdm(samples, desc="Signed composition", dynamic_ncols=True)
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

        tool_cache = collect_cache_cpu_for_nodes(model, tool_tokens, all_nodes)
        no_tool_cache = collect_cache_cpu_for_nodes(model, no_tool_tokens, all_nodes)

        base_metrics: Dict[str, Dict[str, float]] = {}
        for combo_name, nodes in combos.items():
            promote_logits = run_logits_on_base_with_source(model, no_tool_tokens, tool_cache, nodes)
            promote_margin = float(objective_from_logits(promote_logits, tool_objective).item())
            promote_ratio = (promote_margin - m_tool_no) / gap_tool
            promote_top1 = int(promote_logits[0, -1].argmax().item()) == sp.target_tool_call

            suppress_logits = run_logits_on_base_with_source(model, tool_tokens, no_tool_cache, nodes)
            suppress_margin = float(objective_from_logits(suppress_logits, no_tool_objective).item())
            suppress_ratio = (suppress_margin - m_no_tool_tool) / gap_no_tool
            suppress_top1 = int(suppress_logits[0, -1].argmax().item()) == sp.distractor

            base_metrics[combo_name] = {
                "promote_ratio": promote_ratio,
                "suppress_ratio": suppress_ratio,
                "promote_top1": float(promote_top1),
                "suppress_top1": float(suppress_top1),
            }

        sym = base_metrics["symmetric_backbone"]
        for combo_name, metrics in base_metrics.items():
            per_sample_rows.append(
                {
                    "sample_id": sp.sample_id,
                    "combo": combo_name,
                    "n_nodes": len(combos[combo_name]),
                    "promote_ratio": metrics["promote_ratio"],
                    "suppress_ratio": metrics["suppress_ratio"],
                    "promote_tool_top1": bool(metrics["promote_top1"]),
                    "suppress_no_tool_top1": bool(metrics["suppress_top1"]),
                    "promote_gain_vs_symmetric": metrics["promote_ratio"] - sym["promote_ratio"],
                    "suppress_gain_vs_symmetric": metrics["suppress_ratio"] - sym["suppress_ratio"],
                    "promote_top1_gain_vs_symmetric": metrics["promote_top1"] - sym["promote_top1"],
                    "suppress_top1_gain_vs_symmetric": metrics["suppress_top1"] - sym["suppress_top1"],
                }
            )

        pbar.set_postfix(sample=sp.sample_id)

    by_combo: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in per_sample_rows:
        by_combo[str(row["combo"])].append(row)

    combo_order = [
        "symmetric_backbone",
        "symmetric_plus_tool_bias",
        "symmetric_plus_no_tool_bias",
        "symmetric_plus_tool_mode",
        "symmetric_plus_no_tool_mode",
        "full_signed_circuit",
    ]
    summary_rows: List[Dict[str, object]] = []
    for combo in combo_order:
        if combo not in by_combo:
            continue
        rows = by_combo[combo]
        summary_rows.append(
            {
                "combo": combo,
                "n_samples": len(rows),
                "n_nodes": int(rows[0]["n_nodes"]),
                "promote_ratio_median": median(r["promote_ratio"] for r in rows),
                "suppress_ratio_median": median(r["suppress_ratio"] for r in rows),
                "promote_tool_top1_rate": safe_rate(r["promote_tool_top1"] for r in rows),
                "suppress_no_tool_top1_rate": safe_rate(r["suppress_no_tool_top1"] for r in rows),
                "promote_gain_vs_symmetric_median": median(r["promote_gain_vs_symmetric"] for r in rows),
                "suppress_gain_vs_symmetric_median": median(r["suppress_gain_vs_symmetric"] for r in rows),
                "promote_top1_gain_vs_symmetric_mean": mean(r["promote_top1_gain_vs_symmetric"] for r in rows),
                "suppress_top1_gain_vs_symmetric_mean": mean(r["suppress_top1_gain_vs_symmetric"] for r in rows),
            }
        )

    write_csv(per_sample_rows, out_root / "signed_composition_per_sample.csv")
    write_csv(summary_rows, out_root / "signed_composition_summary.csv")
    plot_composition_summary(summary_rows, out_root / "signed_composition_heatmap.png")

    summary = {
        "n_samples": len(samples),
        "combos": {k: v for k, v in combos.items()},
        "artifacts": {
            "per_sample_csv": str(out_root / "signed_composition_per_sample.csv"),
            "summary_csv": str(out_root / "signed_composition_summary.csv"),
            "summary_json": str(out_root / "signed_composition_report.json"),
            "heatmap": str(out_root / "signed_composition_heatmap.png"),
        },
        "summary_rows": summary_rows,
    }
    (out_root / "signed_composition_report.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
