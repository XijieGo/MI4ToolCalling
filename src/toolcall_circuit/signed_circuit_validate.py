#!/usr/bin/env python3
"""
Bidirectional sufficiency/necessity validation for the final signed circuit.

For each group we evaluate:

1) group alone (sufficiency): patch only this group
2) full minus group (necessity inside the full signed circuit)

Both are evaluated in both directions:

- promote: tool-call source -> no-tool base
- suppress: no-tool source -> tool-call base
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
from toolcall_circuit.signed_circuit import GROUP_META, derive_signed_groups
from toolcall_circuit.single_sample import load_hooked_qwen3, objective_from_logits
from toolcall_circuit.bidirectional_token_flip import run_logits_on_base_with_source


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


def save_group_heatmap(rows: Sequence[Dict[str, object]], out_path: Path) -> None:
    group_order = [r["group"] for r in rows if r["group"] != "full_signed_circuit"]
    mat = np.array(
        [
            [
                float(r["promote_suff_ratio_median"]),
                float(r["suppress_suff_ratio_median"]),
                float(r["promote_nec_drop_median"]),
                float(r["suppress_nec_drop_median"]),
            ]
            for r in rows
            if r["group"] != "full_signed_circuit"
        ],
        dtype=float,
    )
    cols = ["promote suff", "suppress suff", "promote nec", "suppress nec"]
    finite_vals = mat[np.isfinite(mat)]
    vmax = max(float(np.percentile(np.abs(finite_vals), 98)), 1e-6) if finite_vals.size else 1.0
    plt.style.use("default")
    fig, ax = plt.subplots(figsize=(8.6, max(4.6, 0.7 * len(group_order) + 1.8)), constrained_layout=True)
    im = ax.imshow(mat, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(np.arange(len(cols)))
    ax.set_xticklabels(cols, rotation=25, ha="right")
    ax.set_yticks(np.arange(len(group_order)))
    ax.set_yticklabels(group_order)
    ax.set_title("Signed Circuit Group Validation")
    cbar = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
    cbar.set_label("ratio / drop")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the final signed circuit bidirectionally.")
    parser.add_argument("--forward-batch-root", type=str, required=True)
    parser.add_argument("--forward-aggregate-summary", type=str, required=True)
    parser.add_argument("--reverse-aggregate-summary", type=str, required=True)
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
    final_nodes = sorted({n for members in groups.values() for n in members})

    forward_batch_root = Path(args.forward_batch_root).resolve()
    reverse_batch_root = Path(args.bidirectional_summary).resolve().parent.parent / "reverse_batch"
    samples = load_sample_paths(forward_batch_root, reverse_batch_root, max_samples=args.max_samples)

    tool_nodes = sorted(final_nodes)
    no_tool_nodes = sorted(final_nodes)
    model, tokenizer = load_hooked_qwen3(args.model_path, device=args.device, dtype=torch.bfloat16)
    _ = tokenizer

    per_sample_rows: List[Dict[str, object]] = []
    pbar = tqdm(samples, desc="Signed circuit validate", dynamic_ncols=True)
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

        tool_cache = collect_cache_cpu_for_nodes(model, tool_tokens, tool_nodes)
        no_tool_cache = collect_cache_cpu_for_nodes(model, no_tool_tokens, no_tool_nodes)

        full_promote_logits = run_logits_on_base_with_source(model, no_tool_tokens, tool_cache, final_nodes)
        full_promote_margin = float(objective_from_logits(full_promote_logits, tool_objective).item())
        full_promote_ratio = (full_promote_margin - m_tool_no) / gap_tool
        full_promote_top1 = int(full_promote_logits[0, -1].argmax().item()) == sp.target_tool_call

        full_suppress_logits = run_logits_on_base_with_source(model, tool_tokens, no_tool_cache, final_nodes)
        full_suppress_margin = float(objective_from_logits(full_suppress_logits, no_tool_objective).item())
        full_suppress_ratio = (full_suppress_margin - m_no_tool_tool) / gap_no_tool
        full_suppress_top1 = int(full_suppress_logits[0, -1].argmax().item()) == sp.distractor

        base_row = {
            "sample_id": sp.sample_id,
            "group": "full_signed_circuit",
            "n_nodes": len(final_nodes),
            "promote_suff_ratio": full_promote_ratio,
            "suppress_suff_ratio": full_suppress_ratio,
            "promote_tool_top1": full_promote_top1,
            "suppress_no_tool_top1": full_suppress_top1,
            "promote_minus_ratio": float("nan"),
            "suppress_minus_ratio": float("nan"),
            "promote_nec_drop": 0.0,
            "suppress_nec_drop": 0.0,
            "promote_nec_top1_drop": 0.0,
            "suppress_nec_top1_drop": 0.0,
        }
        per_sample_rows.append(base_row)

        for group_name, members in groups.items():
            promote_logits = run_logits_on_base_with_source(model, no_tool_tokens, tool_cache, members)
            promote_margin = float(objective_from_logits(promote_logits, tool_objective).item())
            promote_ratio = (promote_margin - m_tool_no) / gap_tool
            promote_top1 = int(promote_logits[0, -1].argmax().item()) == sp.target_tool_call

            suppress_logits = run_logits_on_base_with_source(model, tool_tokens, no_tool_cache, members)
            suppress_margin = float(objective_from_logits(suppress_logits, no_tool_objective).item())
            suppress_ratio = (suppress_margin - m_no_tool_tool) / gap_no_tool
            suppress_top1 = int(suppress_logits[0, -1].argmax().item()) == sp.distractor

            minus_nodes = [n for n in final_nodes if n not in set(members)]
            if minus_nodes:
                minus_promote_logits = run_logits_on_base_with_source(model, no_tool_tokens, tool_cache, minus_nodes)
                minus_promote_margin = float(objective_from_logits(minus_promote_logits, tool_objective).item())
                minus_promote_ratio = (minus_promote_margin - m_tool_no) / gap_tool
                minus_promote_top1 = int(minus_promote_logits[0, -1].argmax().item()) == sp.target_tool_call

                minus_suppress_logits = run_logits_on_base_with_source(model, tool_tokens, no_tool_cache, minus_nodes)
                minus_suppress_margin = float(objective_from_logits(minus_suppress_logits, no_tool_objective).item())
                minus_suppress_ratio = (minus_suppress_margin - m_no_tool_tool) / gap_no_tool
                minus_suppress_top1 = int(minus_suppress_logits[0, -1].argmax().item()) == sp.distractor
            else:
                minus_promote_ratio = float("nan")
                minus_suppress_ratio = float("nan")
                minus_promote_top1 = False
                minus_suppress_top1 = False

            per_sample_rows.append(
                {
                    "sample_id": sp.sample_id,
                    "group": group_name,
                    "n_nodes": len(members),
                    "promote_suff_ratio": promote_ratio,
                    "suppress_suff_ratio": suppress_ratio,
                    "promote_tool_top1": promote_top1,
                    "suppress_no_tool_top1": suppress_top1,
                    "promote_minus_ratio": minus_promote_ratio,
                    "suppress_minus_ratio": minus_suppress_ratio,
                    "promote_nec_drop": full_promote_ratio - minus_promote_ratio if math.isfinite(minus_promote_ratio) else float("nan"),
                    "suppress_nec_drop": minus_suppress_ratio - full_suppress_ratio if math.isfinite(minus_suppress_ratio) else float("nan"),
                    "promote_nec_top1_drop": float(full_promote_top1) - float(minus_promote_top1),
                    "suppress_nec_top1_drop": float(full_suppress_top1) - float(minus_suppress_top1),
                }
            )

        pbar.set_postfix(sample=sp.sample_id)

    by_group: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in per_sample_rows:
        by_group[str(row["group"])].append(row)

    order = ["full_signed_circuit"] + [g for g in groups.keys() if g in by_group]
    summary_rows: List[Dict[str, object]] = []
    for group in order:
        rows = by_group[group]
        summary_rows.append(
            {
                "group": group,
                "group_label": GROUP_META.get(group, {}).get("label", group),
                "n_samples": len(rows),
                "n_nodes": int(rows[0]["n_nodes"]),
                "promote_suff_ratio_median": median(r["promote_suff_ratio"] for r in rows),
                "suppress_suff_ratio_median": median(r["suppress_suff_ratio"] for r in rows),
                "promote_tool_top1_rate": safe_rate(r["promote_tool_top1"] for r in rows),
                "suppress_no_tool_top1_rate": safe_rate(r["suppress_no_tool_top1"] for r in rows),
                "promote_nec_drop_median": median(r["promote_nec_drop"] for r in rows),
                "suppress_nec_drop_median": median(r["suppress_nec_drop"] for r in rows),
                "promote_nec_top1_drop_mean": mean(r["promote_nec_top1_drop"] for r in rows),
                "suppress_nec_top1_drop_mean": mean(r["suppress_nec_top1_drop"] for r in rows),
            }
        )

    save_group_heatmap(summary_rows, out_root / "signed_group_validation_heatmap.png")
    write_csv(per_sample_rows, out_root / "signed_group_per_sample.csv")
    write_csv(summary_rows, out_root / "signed_group_summary.csv")

    summary = {
        "groups": groups,
        "artifacts": {
            "per_sample_csv": str(out_root / "signed_group_per_sample.csv"),
            "summary_csv": str(out_root / "signed_group_summary.csv"),
            "summary_json": str(out_root / "signed_group_report.json"),
            "heatmap_png": str(out_root / "signed_group_validation_heatmap.png"),
        },
        "summary_rows": summary_rows,
    }
    (out_root / "signed_group_report.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
