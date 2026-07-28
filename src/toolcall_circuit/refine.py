#!/usr/bin/env python3
"""
Consensus-constrained per-sample refinement.

Given:
- per-sample circuit summaries
- global core nodes from aggregate summary

Produce:
- refined per-sample circuits: backbone + a few local nodes
- per-sample sufficiency/necessity re-evaluation
- consistency statistics (pairwise Jaccard)
"""

from __future__ import annotations

import os
import argparse
import csv
import gc
import json
import math
import random
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from tqdm.auto import tqdm

if __package__ in {None, ""}:
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from toolcall_circuit.dataset import load_summary_records
from toolcall_circuit.paths import MODEL_PATH_DEFAULT, RESULTS_ROOT
from toolcall_circuit.single_sample import (
    build_edges,
    collect_clean_cache_cpu,
    draw_circuit,
    evaluate_on_base_with_source,
    load_hooked_qwen3,
    node_layer,
)


def sort_nodes(nodes: Sequence[str]) -> List[str]:
    return sorted(set(nodes), key=lambda n: (node_layer(n), 0 if n.startswith("MLP") else 1, n))


def jaccard(a: Sequence[str], b: Sequence[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def pairwise_jaccard_mean(node_lists: Sequence[Sequence[str]]) -> Tuple[float, float]:
    vals = []
    n = len(node_lists)
    for i in range(n):
        for j in range(i + 1, n):
            vals.append(jaccard(node_lists[i], node_lists[j]))
    if not vals:
        return float("nan"), float("nan")
    vals = sorted(vals)
    return float(np.mean(vals)), float(np.median(vals))


def harmonic_mean(x: float, y: float, eps: float = 1e-8) -> float:
    if x <= 0 or y <= 0:
        return 0.0
    return 2.0 * x * y / max(eps, (x + y))


def build_score_lookup(summary: Dict[str, object], default_boost: float = 0.25) -> Dict[str, float]:
    score_lookup: Dict[str, float] = {}
    for d in summary.get("top_node_scores", []):
        name = str(d.get("name"))
        score = float(d.get("score", 0.0))
        score_lookup[name] = score
    # fallback for nodes possibly not in top list
    for n in summary.get("detailed_nodes", []):
        score_lookup.setdefault(str(n), default_boost)
    return score_lookup


def main() -> None:
    parser = argparse.ArgumentParser(description="Refine per-sample circuits with a global backbone.")
    parser.add_argument("--input-root", type=str, default=str(RESULTS_ROOT / "manual_run" / "batch"))
    parser.add_argument(
        "--aggregate-summary",
        type=str,
        default=str(RESULTS_ROOT / "manual_run" / "aggregate" / "global_core_summary.json"),
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=str(RESULTS_ROOT / "manual_run" / "refined_consistent"),
    )
    parser.add_argument("--model-path", type=str, default=str(MODEL_PATH_DEFAULT))
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max-local", type=int, default=4)
    parser.add_argument("--gap-min", type=float, default=0.5)
    parser.add_argument("--edge-max-parents", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    _ = rng  # reserved for future ablations

    input_root = Path(args.input_root).resolve()
    out_root = Path(args.output_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    agg_summary = json.loads(Path(args.aggregate_summary).resolve().read_text(encoding="utf-8"))
    backbone_nodes = sort_nodes(list(agg_summary.get("core_nodes", [])))
    if not backbone_nodes:
        raise ValueError("No core_nodes found in aggregate summary.")

    summary_records = load_summary_records(input_root)
    if not summary_records:
        raise ValueError(f"No sample dirs found under {input_root}")

    model, tokenizer = load_hooked_qwen3(args.model_path, device=args.device, dtype=torch.bfloat16)
    _ = tokenizer

    rows: List[Dict[str, object]] = []
    refined_node_lists = []
    orig_node_lists = []

    for summary_record in tqdm(summary_records, desc="Refine samples", dynamic_ncols=True):
        summary = summary_record.summary
        sample_id = summary_record.sample_id
        sample_rank = summary_record.sample_rank
        legacy_index = summary_record.legacy_index
        clean_prompt = Path(summary["clean_prompt"])
        corrupt_prompt = Path(summary["corrupt_prompt"])
        target = int(summary["target_token_id"])
        distractor = int(summary["distractor_token_id"])
        clean_obj = float(summary["clean_obj"])
        corrupt_obj = float(summary["corrupt_obj"])
        gap = float(summary["gap"])

        orig_nodes = sort_nodes(list(summary.get("detailed_nodes", [])))
        orig_node_lists.append(orig_nodes)

        if not clean_prompt.exists() or not corrupt_prompt.exists():
            continue

        clean_text = clean_prompt.read_text(encoding="utf-8")
        corrupt_text = corrupt_prompt.read_text(encoding="utf-8")
        clean_tokens = model.to_tokens(clean_text, prepend_bos=False)
        corrupt_tokens = model.to_tokens(corrupt_text, prepend_bos=False)
        if clean_tokens.shape != corrupt_tokens.shape:
            continue

        clean_cache = collect_clean_cache_cpu(model, clean_tokens)
        corrupt_cache = collect_clean_cache_cpu(model, corrupt_tokens)

        score_lookup = build_score_lookup(summary)
        for n in backbone_nodes:
            score_lookup.setdefault(n, 0.35)

        local_candidates = [
            str(d["name"])
            for d in summary.get("top_node_scores", [])
            if str(d["name"]) not in set(backbone_nodes)
        ]
        local_candidates = [n for n in local_candidates if score_lookup.get(n, 0.0) > 0]

        best = None
        for k in range(args.max_local + 1):
            chosen_local = local_candidates[:k]
            refined_nodes = sort_nodes(backbone_nodes + chosen_local)
            refined_edges = build_edges(
                refined_nodes,
                score_lookup=score_lookup,
                max_parents=args.edge_max_parents,
            )

            if abs(gap) <= 1e-8:
                suff_ratio = float("nan")
                nec_ratio = float("nan")
                score = float("-inf")
            else:
                suff_obj = evaluate_on_base_with_source(
                    model=model,
                    base_tokens=corrupt_tokens,
                    source_cache_cpu=clean_cache,
                    patch_nodes=refined_nodes,
                    target_token=target,
                    distractor_token=distractor,
                )
                suff_ratio = (suff_obj - corrupt_obj) / gap

                clean_with_refined_corrupt = evaluate_on_base_with_source(
                    model=model,
                    base_tokens=clean_tokens,
                    source_cache_cpu=corrupt_cache,
                    patch_nodes=refined_nodes,
                    target_token=target,
                    distractor_token=distractor,
                )
                nec_ratio = (clean_obj - clean_with_refined_corrupt) / gap
                score = harmonic_mean(max(0.0, suff_ratio), max(0.0, nec_ratio)) - 0.02 * k

            candidate = {
                "k_local": k,
                "local_nodes": chosen_local,
                "nodes": refined_nodes,
                "edges": refined_edges,
                "suff_ratio": suff_ratio,
                "nec_ratio": nec_ratio,
                "score": score,
            }
            if best is None or candidate["score"] > best["score"]:
                best = candidate

        assert best is not None
        refined_nodes = best["nodes"]
        refined_edges = best["edges"]
        refined_node_lists.append(refined_nodes)

        q_out = out_root / sample_id
        q_out.mkdir(parents=True, exist_ok=True)

        draw_circuit(
            nodes=refined_nodes,
            edges=refined_edges,
            out_path=q_out / "final_circuit_refined.png",
            title="Refined Consistent Circuit (Backbone + Local)",
        )

        refined_summary = {
            "sample_id": sample_id,
            "sample_rank": sample_rank,
            "q_index": legacy_index,
            "backbone_nodes": backbone_nodes,
            "orig_nodes": orig_nodes,
            "refined_nodes": refined_nodes,
            "refined_edges": refined_edges,
            "k_local": best["k_local"],
            "local_nodes": best["local_nodes"],
            "refined_suff_ratio_vs_gap": best["suff_ratio"],
            "refined_necessity_ratio_vs_gap": best["nec_ratio"],
            "orig_suff_ratio_vs_gap": summary.get("detailed_ratio_vs_gap"),
            "orig_necessity_ratio_vs_gap": summary.get("necessity_ratio_vs_gap"),
            "gap": gap,
            "ap_mode": summary.get("ap_mode", "full"),
        }
        (q_out / "summary_refined.json").write_text(
            json.dumps(refined_summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        rows.append(
            {
                "sample_id": sample_id,
                "sample_rank": sample_rank,
                "q_index": legacy_index,
                "gap": gap,
                "orig_suff_ratio": float(summary.get("detailed_ratio_vs_gap", float("nan"))),
                "orig_nec_ratio": float(summary.get("necessity_ratio_vs_gap", float("nan"))),
                "refined_suff_ratio": best["suff_ratio"],
                "refined_nec_ratio": best["nec_ratio"],
                "k_local": best["k_local"],
                "n_refined_nodes": len(refined_nodes),
                "n_orig_nodes": len(orig_nodes),
            }
        )

        model.reset_hooks()
        gc.collect()
        torch.cuda.empty_cache()

    # Write table
    with (out_root / "refined_summary_table.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "sample_id",
                "sample_rank",
                "q_index",
                "gap",
                "orig_suff_ratio",
                "orig_nec_ratio",
                "refined_suff_ratio",
                "refined_nec_ratio",
                "k_local",
                "n_refined_nodes",
                "n_orig_nodes",
            ],
        )
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda x: (x.get("sample_rank") or 10**9, x["sample_id"])))

    def finite(vals: Sequence[float]) -> List[float]:
        return [v for v in vals if isinstance(v, (int, float)) and math.isfinite(float(v))]

    orig_suff = finite([r["orig_suff_ratio"] for r in rows if r["gap"] > args.gap_min])
    orig_nec = finite([r["orig_nec_ratio"] for r in rows if r["gap"] > args.gap_min])
    ref_suff = finite([r["refined_suff_ratio"] for r in rows if r["gap"] > args.gap_min])
    ref_nec = finite([r["refined_nec_ratio"] for r in rows if r["gap"] > args.gap_min])

    # Consistency
    orig_j_mean, orig_j_median = pairwise_jaccard_mean(orig_node_lists)
    ref_j_mean, ref_j_median = pairwise_jaccard_mean(refined_node_lists)

    global_refined_nodes = sort_nodes(
        list(set(n for nodes in refined_node_lists for n in nodes))
    )
    # Build a global refined graph with simple support scoring by frequency.
    node_freq: Dict[str, float] = {}
    for n in global_refined_nodes:
        node_freq[n] = sum(1 for nodes in refined_node_lists if n in set(nodes)) / max(1, len(refined_node_lists))
    # synthetic edges based on sample edge frequency
    edge_freq: Dict[Tuple[str, str], float] = {}
    edge_count: Dict[Tuple[str, str], int] = {}
    for r in rows:
        p = out_root / str(r["sample_id"]) / "summary_refined.json"
        d = json.loads(p.read_text(encoding="utf-8"))
        for e in [tuple(x) for x in d["refined_edges"]]:
            edge_count[e] = edge_count.get(e, 0) + 1
    for e, c in edge_count.items():
        edge_freq[e] = c / max(1, len(rows))
    global_edges = build_edges(global_refined_nodes, score_lookup=node_freq, max_parents=2)
    draw_circuit(
        nodes=global_refined_nodes,
        edges=global_edges,
        out_path=out_root / "final_circuit_refined_global_union.png",
        title="Refined Global Union Circuit",
    )

    report = {
        "n_samples": len(rows),
        "backbone_nodes": backbone_nodes,
        "max_local": args.max_local,
        "gap_min_for_metrics": args.gap_min,
        "orig_metrics": {
            "suff_median": float(np.median(orig_suff)) if orig_suff else float("nan"),
            "nec_median": float(np.median(orig_nec)) if orig_nec else float("nan"),
            "suff_mean": float(np.mean(orig_suff)) if orig_suff else float("nan"),
            "nec_mean": float(np.mean(orig_nec)) if orig_nec else float("nan"),
        },
        "refined_metrics": {
            "suff_median": float(np.median(ref_suff)) if ref_suff else float("nan"),
            "nec_median": float(np.median(ref_nec)) if ref_nec else float("nan"),
            "suff_mean": float(np.mean(ref_suff)) if ref_suff else float("nan"),
            "nec_mean": float(np.mean(ref_nec)) if ref_nec else float("nan"),
        },
        "consistency": {
            "orig_pairwise_jaccard_mean": orig_j_mean,
            "orig_pairwise_jaccard_median": orig_j_median,
            "refined_pairwise_jaccard_mean": ref_j_mean,
            "refined_pairwise_jaccard_median": ref_j_median,
        },
        "artifacts": {
            "table_csv": str(out_root / "refined_summary_table.csv"),
            "global_union_png": str(out_root / "final_circuit_refined_global_union.png"),
        },
    }
    (out_root / "refined_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"[done] refined outputs: {out_root}")
    print(
        "[done] jaccard mean orig->refined: "
        f"{orig_j_mean:.3f} -> {ref_j_mean:.3f}"
    )
    print(
        "[done] suff median orig->refined: "
        f"{report['orig_metrics']['suff_median']:.3f} -> {report['refined_metrics']['suff_median']:.3f}"
    )


if __name__ == "__main__":
    main()
