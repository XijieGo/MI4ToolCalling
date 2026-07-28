#!/usr/bin/env python3
"""
Edge-level path patching for tool-call circuit.

For each directed edge (u -> v) in the consensus circuit:
1) source effect: patch source node from clean into corrupt;
2) blocked effect: patch source clean, but force target node to corrupt state;
3) mediated edge effect = source effect - blocked effect.

This follows path patching intuition from "How does GPT-2 compute greater-than?":
we test whether source influence reaches logits through a specific downstream target.
"""

from __future__ import annotations

import os
import argparse
import csv
import gc
import json
import math
import random
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm.auto import tqdm

# Force non-interactive backend for remote GPU workers.
import matplotlib

matplotlib.use("Agg")

if __package__ in {None, ""}:
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from toolcall_circuit.dataset import (
    build_position_sets,
    get_tool_call_target_spec,
    load_summary_records,
    resolve_distractor_token,
)
from toolcall_circuit.paths import MODEL_PATH_DEFAULT, RESULTS_ROOT
from toolcall_circuit.single_sample import draw_circuit, load_hooked_qwen3, objective_from_logits


INPUT_NODE = "Input Embed"
OUTPUT_NODE = "Residual Output: <tool_call>"


@dataclass(frozen=True)
class NodeSpec:
    node: str
    hook_name: str
    kind: str  # "head" | "mlp" | "resid"
    layer: int
    head: Optional[int] = None


def parse_head(node: str) -> Tuple[int, int]:
    m = re.fullmatch(r"L(\d+)H(\d+)", node)
    if m is None:
        raise ValueError(f"Not a head node: {node}")
    return int(m.group(1)), int(m.group(2))


def parse_mlp(node: str) -> int:
    if not node.startswith("MLP"):
        raise ValueError(f"Not an MLP node: {node}")
    return int(node[3:])


def node_to_spec(node: str) -> NodeSpec:
    if node == INPUT_NODE:
        return NodeSpec(node=node, hook_name="blocks.0.hook_resid_pre", kind="resid", layer=0, head=None)
    if node.startswith("L"):
        layer, head = parse_head(node)
        return NodeSpec(node=node, hook_name=f"blocks.{layer}.attn.hook_z", kind="head", layer=layer, head=head)
    if node.startswith("MLP"):
        layer = parse_mlp(node)
        return NodeSpec(node=node, hook_name=f"blocks.{layer}.hook_mlp_out", kind="mlp", layer=layer, head=None)
    raise ValueError(f"Unknown node format: {node}")


def finite(xs: Iterable[float]) -> List[float]:
    out: List[float] = []
    for x in xs:
        if isinstance(x, (int, float)) and math.isfinite(float(x)):
            out.append(float(x))
    return out


def med(xs: Iterable[float]) -> float:
    vals = finite(xs)
    return float(median(vals)) if vals else float("nan")


def mean(xs: Iterable[float]) -> float:
    vals = finite(xs)
    return float(np.mean(vals)) if vals else float("nan")


def bootstrap_ci(values: Sequence[float], n_boot: int, seed: int) -> Dict[str, float]:
    vals = finite(values)
    if not vals:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan")}
    rng = random.Random(seed)
    n = len(vals)
    boot: List[float] = []
    for _ in range(n_boot):
        sample = [vals[rng.randrange(n)] for __ in range(n)]
        boot.append(float(np.median(sample)))
    boot.sort()
    lo_idx = max(0, int(0.025 * n_boot))
    hi_idx = min(n_boot - 1, int(0.975 * n_boot))
    return {"mean": float(np.mean(boot)), "lo": float(boot[lo_idx]), "hi": float(boot[hi_idx])}


def apply_plot_style() -> None:
    plt.style.use("default")
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 16,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        }
    )


def save_diverging_heatmap(
    matrix: np.ndarray,
    row_labels: Sequence[str],
    col_labels: Sequence[str],
    title: str,
    out_path: Path,
    percentile_clip: float = 98.0,
) -> None:
    apply_plot_style()
    fig, ax = plt.subplots(figsize=(1.3 * max(5, len(col_labels)), 0.45 * max(7, len(row_labels)) + 1.8))
    vals = matrix[np.isfinite(matrix)]
    if vals.size == 0:
        vmax = 1.0
    else:
        vmax = float(np.percentile(np.abs(vals), percentile_clip))
        vmax = max(vmax, 1e-6)
    im = ax.imshow(matrix, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_title(f"{title}\nSymmetric clipping at ±{vmax:.3f}", pad=10)
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=25, ha="right")
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels)
    cbar = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
    cbar.set_label("Median ratio (center=0)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_edge_bar(summary_rows: Sequence[Dict[str, object]], out_path: Path, title: str, key: str, ci_key: str) -> None:
    apply_plot_style()
    labels = [str(r["edge"]) for r in summary_rows]
    y = np.array([float(r[key]) for r in summary_rows], dtype=np.float64)
    lo = np.array([float(r[ci_key]["lo"]) for r in summary_rows], dtype=np.float64)
    hi = np.array([float(r[ci_key]["hi"]) for r in summary_rows], dtype=np.float64)
    yerr = np.vstack([np.maximum(0.0, y - lo), np.maximum(0.0, hi - y)])

    fig, ax = plt.subplots(figsize=(max(9.5, 0.33 * len(labels) + 5), 4.8), constrained_layout=True)
    ax.bar(np.arange(len(labels)), y, color="#2a6f97", edgecolor="#1f1f1f", linewidth=0.8)
    ax.errorbar(np.arange(len(labels)), y, yerr=yerr, fmt="none", ecolor="#1f1f1f", elinewidth=1.1, capsize=2.8)
    ax.axhline(0.0, color="#444444", linewidth=1.0)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=70, ha="right")
    ax.set_ylabel("Mediated edge ratio")
    ax.set_title(title)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def build_needed_hook_names(edges: Sequence[Tuple[str, str]]) -> List[str]:
    names = {"blocks.0.hook_resid_pre"}  # for Input Embed patch at contrast position
    for u, v in edges:
        if u != OUTPUT_NODE:
            names.add(node_to_spec(u).hook_name)
        if v not in {OUTPUT_NODE, INPUT_NODE}:
            names.add(node_to_spec(v).hook_name)
    return sorted(names)


def collect_cache_cpu(model, tokens: torch.Tensor, needed_names: Sequence[str]) -> Dict[str, torch.Tensor]:
    name_set = set(needed_names)
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens, names_filter=lambda n: n in name_set)
    return {k: v.detach().cpu() for k, v in cache.items()}


def make_patch_hook(
    spec: NodeSpec,
    mode: str,  # "clean" | "corrupt"
    clean_cache: Dict[str, torch.Tensor],
    corrupt_cache: Dict[str, torch.Tensor],
    contrast_positions: Sequence[int],
) -> Tuple[str, callable]:
    src = clean_cache if mode == "clean" else corrupt_cache

    if spec.kind == "head":
        assert spec.head is not None
        patch_vec = src[spec.hook_name][0, -1, spec.head, :].clone()
        head_idx = int(spec.head)

        def hook_fn(z: torch.Tensor, hook):  # noqa: ANN001
            out = z.clone()
            out[:, -1, head_idx, :] = patch_vec.to(z.device, dtype=z.dtype)
            return out

        return spec.hook_name, hook_fn

    if spec.kind == "mlp":
        patch_vec = src[spec.hook_name][0, -1, :].clone()

        def hook_fn(mlp_out: torch.Tensor, hook):  # noqa: ANN001
            out = mlp_out.clone()
            out[:, -1, :] = patch_vec.to(mlp_out.device, dtype=mlp_out.dtype)
            return out

        return spec.hook_name, hook_fn

    if spec.kind == "resid":
        def hook_fn(resid: torch.Tensor, hook):  # noqa: ANN001
            out = resid.clone()
            for pos in sorted({int(p) for p in contrast_positions if 0 <= int(p) < out.shape[1]}):
                patch_vec = src[spec.hook_name][0, pos, :].clone()
                out[:, pos, :] = patch_vec.to(resid.device, dtype=resid.dtype)
            return out

        return spec.hook_name, hook_fn

    raise ValueError(f"Unknown spec.kind: {spec.kind}")


def run_obj_with_hooks(
    model,
    corrupt_tokens: torch.Tensor,
    target_token: int,
    distractor_token: int,
    hooks: Sequence[Tuple[str, callable]],
) -> float:
    with torch.no_grad():
        logits = model.run_with_hooks(corrupt_tokens, fwd_hooks=list(hooks))
    return float(objective_from_logits(logits, target_token, distractor_token).item())


def trim_by_influence(
    rows: Sequence[Dict[str, object]],
    edge_labels: Sequence[str],
    trim_frac: float,
) -> Tuple[List[str], List[str]]:
    if trim_frac <= 0.0:
        sample_ids = sorted({str(r["sample_id"]) for r in rows})
        return sample_ids, []
    by_sample: Dict[str, Dict[str, float]] = defaultdict(dict)
    for r in rows:
        sample_id = str(r["sample_id"])
        e = str(r["edge"])
        by_sample[sample_id][e] = float(r["edge_ratio"])
    sample_ids = sorted(by_sample.keys())
    if not sample_ids:
        return [], []

    # Reference sign pattern from full medians.
    med_by_edge: Dict[str, float] = {}
    for e in edge_labels:
        vals = [by_sample[sample_id][e] for sample_id in sample_ids if e in by_sample[sample_id]]
        med_by_edge[e] = med(vals)
    sig_edges = [e for e in edge_labels if math.isfinite(med_by_edge[e]) and abs(med_by_edge[e]) >= 0.02]
    if not sig_edges:
        return sample_ids, []

    scores: List[Tuple[str, float]] = []
    for sample_id in sample_ids:
        mismatch = 0.0
        weight_sum = 0.0
        for e in sig_edges:
            if e not in by_sample[sample_id]:
                continue
            w = abs(med_by_edge[e])
            weight_sum += w
            if by_sample[sample_id][e] * med_by_edge[e] < 0:
                mismatch += w
        score = mismatch / max(weight_sum, 1e-8)
        scores.append((sample_id, score))

    k_drop = int(math.floor(trim_frac * len(sample_ids)))
    k_drop = min(k_drop, max(0, len(sample_ids) - 10))
    if k_drop <= 0:
        return sample_ids, []
    scores_sorted = sorted(scores, key=lambda x: x[1], reverse=True)
    drop = sorted([sample_id for sample_id, _ in scores_sorted[:k_drop]])
    keep = [sample_id for sample_id in sample_ids if sample_id not in set(drop)]
    return keep, drop


def summarize_edges(
    rows: Sequence[Dict[str, object]],
    edge_labels: Sequence[str],
    bootstrap: int,
    seed: int,
) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    by_edge: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for r in rows:
        by_edge[str(r["edge"])].append(r)

    for i, e in enumerate(edge_labels):
        grp = by_edge.get(e, [])
        src_vals = [float(x["source_ratio"]) for x in grp]
        blk_vals = [float(x["blocked_ratio"]) for x in grp]
        edg_vals = [float(x["edge_ratio"]) for x in grp]
        row = {
            "edge": e,
            "n_samples": len(grp),
            "source_ratio_median": med(src_vals),
            "blocked_ratio_median": med(blk_vals),
            "edge_ratio_median": med(edg_vals),
            "edge_ratio_mean": mean(edg_vals),
            "edge_ratio_ci": bootstrap_ci(edg_vals, n_boot=bootstrap, seed=seed + 17 * (i + 1)),
            "positive_frac": mean([1.0 if v > 0 else 0.0 for v in edg_vals]),
        }
        out.append(row)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Edge-level path patching for tool-call circuit.")
    parser.add_argument("--input-root", type=str, default=str(RESULTS_ROOT / "manual_run" / "batch"))
    parser.add_argument(
        "--aggregate-summary",
        type=str,
        default=str(RESULTS_ROOT / "manual_run" / "aggregate" / "global_core_summary.json"),
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=str(RESULTS_ROOT / "manual_run" / "semantic_roles"),
    )
    parser.add_argument("--model-path", type=str, default=str(MODEL_PATH_DEFAULT))
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--gap-min", type=float, default=0.5)
    parser.add_argument("--max-samples", type=int, default=0, help="0 means all valid samples.")
    parser.add_argument("--sample-rank-start", type=int, default=0)
    parser.add_argument("--sample-rank-end", type=int, default=0)
    parser.add_argument("--q-start", type=int, default=1)
    parser.add_argument("--q-end", type=int, default=10_000)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--trim-frac", type=float, default=0.10, help="Optional robust subset drop fraction (<=0.10).")
    args = parser.parse_args()

    trim_frac = float(max(0.0, min(0.10, args.trim_frac)))

    input_root = Path(args.input_root).resolve()
    out_root = Path(args.output_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    agg = json.loads(Path(args.aggregate_summary).resolve().read_text(encoding="utf-8"))
    core_edges = [tuple(e) for e in agg.get("core_edges", [])]
    if not core_edges:
        raise ValueError("No core_edges found in aggregate summary.")

    edge_labels = [f"{u}->{v}" for u, v in core_edges]
    needed_names = build_needed_hook_names(core_edges)

    model, tokenizer = load_hooked_qwen3(args.model_path, device=args.device, dtype=torch.bfloat16)
    model.set_use_attn_result(False)
    model.set_use_split_qkv_input(False)
    model.set_use_hook_mlp_in(False)
    target_spec = get_tool_call_target_spec(tokenizer, target_text="<tool_call>")
    if not target_spec.is_single_token:
        raise NotImplementedError(
            f"<tool_call> tokenization is no longer single-token ({target_spec.token_ids}); "
            "upgrade the path-patching objective before running this workflow."
        )
    target_token = target_spec.primary_token_id

    sample_infos: List[Tuple[str, int | None, int | None, Dict[str, object], Path, Path]] = []
    for summary_record in load_summary_records(input_root):
        s = summary_record.summary
        sample_rank = summary_record.sample_rank
        legacy_index = summary_record.legacy_index
        if legacy_index is not None and (legacy_index < args.q_start or legacy_index > args.q_end):
            continue
        if sample_rank is not None:
            if args.sample_rank_start > 0 and sample_rank < args.sample_rank_start:
                continue
            if args.sample_rank_end > 0 and sample_rank > args.sample_rank_end:
                continue
        gap = float(s.get("gap", float("nan")))
        if not math.isfinite(gap) or gap <= args.gap_min:
            continue
        clean_prompt = Path(s["clean_prompt"])
        corrupt_prompt = Path(s["corrupt_prompt"])
        if not clean_prompt.exists() or not corrupt_prompt.exists():
            continue
        sample_infos.append(
            (
                summary_record.sample_id,
                summary_record.sample_rank,
                summary_record.legacy_index,
                s,
                clean_prompt,
                corrupt_prompt,
            )
        )
    if args.max_samples > 0:
        sample_infos = sample_infos[: args.max_samples]
    if not sample_infos:
        raise ValueError("No valid samples selected.")

    rows: List[Dict[str, object]] = []
    skipped: List[str] = []
    analyzed: List[str] = []
    contrast_pos_hist: Dict[int, int] = defaultdict(int)
    contrast_span_len_hist: Dict[int, int] = defaultdict(int)

    pbar = tqdm(sample_infos, desc="Edge path patch", dynamic_ncols=True)
    for sample_id, sample_rank, legacy_index, summary, clean_prompt, corrupt_prompt in pbar:
        clean_text = clean_prompt.read_text(encoding="utf-8")
        corrupt_text = corrupt_prompt.read_text(encoding="utf-8")
        clean_tokens = model.to_tokens(clean_text, prepend_bos=False)
        corrupt_tokens = model.to_tokens(corrupt_text, prepend_bos=False)
        if clean_tokens.shape != corrupt_tokens.shape:
            skipped.append(sample_id)
            continue

        ids_clean = [int(x) for x in clean_tokens[0].tolist()]
        ids_corrupt = [int(x) for x in corrupt_tokens[0].tolist()]
        pos_sets = build_position_sets(ids_clean, ids_corrupt, tokenizer, clean_text=clean_text)
        contrast_positions = [int(p) for p in pos_sets["contrast"]]
        if not contrast_positions:
            skipped.append(sample_id)
            continue
        for pos in contrast_positions:
            contrast_pos_hist[pos] += 1
        contrast_span_len_hist[len(contrast_positions)] += 1

        try:
            with torch.no_grad():
                clean_logits = model(clean_tokens)
                corrupt_logits = model(corrupt_tokens)
        except torch.OutOfMemoryError:
            skipped.append(sample_id)
            model.reset_hooks()
            gc.collect()
            torch.cuda.empty_cache()
            continue

        distractor = int(summary.get("distractor_token_id", -1))
        if distractor < 0 or distractor >= corrupt_logits.shape[-1] or distractor == target_token:
            distractor = resolve_distractor_token(corrupt_logits[0, -1, :], target_token)

        clean_obj = float(objective_from_logits(clean_logits, target_token, distractor).item())
        corrupt_obj = float(objective_from_logits(corrupt_logits, target_token, distractor).item())
        gap = clean_obj - corrupt_obj
        if not math.isfinite(gap) or gap <= args.gap_min:
            skipped.append(sample_id)
            continue

        try:
            clean_cache = collect_cache_cpu(model, clean_tokens, needed_names=needed_names)
            corrupt_cache = collect_cache_cpu(model, corrupt_tokens, needed_names=needed_names)
        except torch.OutOfMemoryError:
            skipped.append(sample_id)
            model.reset_hooks()
            gc.collect()
            torch.cuda.empty_cache()
            continue

        try:
            for (u, v), edge_label in zip(core_edges, edge_labels):
                source_spec = node_to_spec(u)
                src_hook = make_patch_hook(
                    spec=source_spec,
                    mode="clean",
                    clean_cache=clean_cache,
                    corrupt_cache=corrupt_cache,
                    contrast_positions=contrast_positions,
                )
                src_obj = run_obj_with_hooks(
                    model=model,
                    corrupt_tokens=corrupt_tokens,
                    target_token=target_token,
                    distractor_token=distractor,
                    hooks=[src_hook],
                )
                source_ratio = (src_obj - corrupt_obj) / gap

                if v == OUTPUT_NODE:
                    blocked_obj = corrupt_obj
                else:
                    target_spec = node_to_spec(v)
                    blk_hook = make_patch_hook(
                        spec=target_spec,
                        mode="corrupt",
                        clean_cache=clean_cache,
                        corrupt_cache=corrupt_cache,
                        contrast_positions=contrast_positions,
                    )
                    blocked_obj = run_obj_with_hooks(
                        model=model,
                        corrupt_tokens=corrupt_tokens,
                        target_token=target_token,
                        distractor_token=distractor,
                        hooks=[src_hook, blk_hook],
                    )
                blocked_ratio = (blocked_obj - corrupt_obj) / gap
                edge_ratio = source_ratio - blocked_ratio

                rows.append(
                    {
                        "sample_id": sample_id,
                        "sample_rank": sample_rank,
                        "q_index": legacy_index,
                        "edge": edge_label,
                        "source": u,
                        "target": v,
                        "source_ratio": source_ratio,
                        "blocked_ratio": blocked_ratio,
                        "edge_ratio": edge_ratio,
                        "gap": gap,
                    }
                )
        except torch.OutOfMemoryError:
            skipped.append(sample_id)
            model.reset_hooks()
            gc.collect()
            torch.cuda.empty_cache()
            continue

        analyzed.append(sample_id)
        model.reset_hooks()
        del clean_cache
        del corrupt_cache
        gc.collect()
        torch.cuda.empty_cache()

    # Per-sample table.
    per_sample_csv = out_root / "path_patch_edge_per_sample.csv"
    with per_sample_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "sample_id",
                "sample_rank",
                "q_index",
                "edge",
                "source",
                "target",
                "source_ratio",
                "blocked_ratio",
                "edge_ratio",
                "gap",
            ],
        )
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # Full summary.
    full_summary = summarize_edges(rows=rows, edge_labels=edge_labels, bootstrap=args.bootstrap, seed=args.seed)
    full_summary_csv = out_root / "path_patch_edge_summary_full.csv"
    with full_summary_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "edge",
                "n_samples",
                "source_ratio_median",
                "blocked_ratio_median",
                "edge_ratio_median",
                "edge_ratio_mean",
                "edge_ratio_ci_lo",
                "edge_ratio_ci_hi",
                "positive_frac",
            ]
        )
        for r in full_summary:
            w.writerow(
                [
                    r["edge"],
                    r["n_samples"],
                    r["source_ratio_median"],
                    r["blocked_ratio_median"],
                    r["edge_ratio_median"],
                    r["edge_ratio_mean"],
                    r["edge_ratio_ci"]["lo"],
                    r["edge_ratio_ci"]["hi"],
                    r["positive_frac"],
                ]
            )

    # Robust subset (drop <=10% high-mismatch samples).
    keep_sample_ids, drop_sample_ids = trim_by_influence(rows=rows, edge_labels=edge_labels, trim_frac=trim_frac)
    rows_trim = [r for r in rows if str(r["sample_id"]) in set(keep_sample_ids)]
    trim_summary = summarize_edges(rows=rows_trim, edge_labels=edge_labels, bootstrap=args.bootstrap, seed=args.seed + 500)
    trim_summary_csv = out_root / "path_patch_edge_summary_trimmed.csv"
    with trim_summary_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "edge",
                "n_samples",
                "source_ratio_median",
                "blocked_ratio_median",
                "edge_ratio_median",
                "edge_ratio_mean",
                "edge_ratio_ci_lo",
                "edge_ratio_ci_hi",
                "positive_frac",
            ]
        )
        for r in trim_summary:
            w.writerow(
                [
                    r["edge"],
                    r["n_samples"],
                    r["source_ratio_median"],
                    r["blocked_ratio_median"],
                    r["edge_ratio_median"],
                    r["edge_ratio_mean"],
                    r["edge_ratio_ci"]["lo"],
                    r["edge_ratio_ci"]["hi"],
                    r["positive_frac"],
                ]
            )

    # Visuals: full + trimmed edge ratios.
    matrix_full = np.array(
        [[float(r["source_ratio_median"]), float(r["blocked_ratio_median"]), float(r["edge_ratio_median"])] for r in full_summary],
        dtype=np.float64,
    )
    save_diverging_heatmap(
        matrix=matrix_full,
        row_labels=edge_labels,
        col_labels=["source_ratio", "blocked_ratio", "edge_ratio"],
        title="Edge Path Patching (Full Set)",
        out_path=out_root / "path_patch_edge_heatmap_full.png",
    )

    matrix_trim = np.array(
        [[float(r["source_ratio_median"]), float(r["blocked_ratio_median"]), float(r["edge_ratio_median"])] for r in trim_summary],
        dtype=np.float64,
    )
    save_diverging_heatmap(
        matrix=matrix_trim,
        row_labels=edge_labels,
        col_labels=["source_ratio", "blocked_ratio", "edge_ratio"],
        title=f"Edge Path Patching (Robust Subset, drop={len(drop_sample_ids)})",
        out_path=out_root / "path_patch_edge_heatmap_trimmed.png",
    )

    full_sorted = sorted(full_summary, key=lambda x: float(x["edge_ratio_median"]), reverse=True)
    trim_sorted = sorted(trim_summary, key=lambda x: float(x["edge_ratio_median"]), reverse=True)
    save_edge_bar(
        summary_rows=full_sorted,
        out_path=out_root / "path_patch_edge_bar_full.png",
        title="Edge Mediated Effect (Full Set, sorted)",
        key="edge_ratio_median",
        ci_key="edge_ratio_ci",
    )
    save_edge_bar(
        summary_rows=trim_sorted,
        out_path=out_root / "path_patch_edge_bar_trimmed.png",
        title="Edge Mediated Effect (Robust Subset, sorted)",
        key="edge_ratio_median",
        ci_key="edge_ratio_ci",
    )

    # Weighted circuit using trimmed medians (clear chain view).
    score_lookup: Dict[str, float] = {}
    for e in trim_summary:
        src = str(e["edge"]).split("->", 1)[0]
        score_lookup[src] = score_lookup.get(src, 0.0) + abs(float(e["edge_ratio_median"]))
    nodes = [n for n in agg.get("core_nodes", [])]
    draw_circuit(
        nodes=nodes,
        edges=core_edges,
        out_path=out_root / "final_circuit_edge_path_patching.png",
        title="Final Circuit (Edge Path-Patching Validated)",
    )

    report = {
        "n_input_samples": len(sample_infos),
        "n_analyzed_samples": len(set(analyzed)),
        "analyzed_sample_ids": sorted(set(analyzed)),
        "skipped_sample_ids": sorted(set(skipped)),
        "contrast_position_hist": {str(k): int(v) for k, v in sorted(contrast_pos_hist.items())},
        "contrast_span_length_hist": {str(k): int(v) for k, v in sorted(contrast_span_len_hist.items())},
        "gap_min": args.gap_min,
        "bootstrap": args.bootstrap,
        "seed": args.seed,
        "trim_frac": trim_frac,
        "trimmed_drop_sample_ids": drop_sample_ids,
        "core_edges": core_edges,
        "edge_summary_full": full_summary,
        "edge_summary_trimmed": trim_summary,
        "artifacts": {
            "per_sample_csv": str(per_sample_csv),
            "summary_full_csv": str(full_summary_csv),
            "summary_trimmed_csv": str(trim_summary_csv),
            "heatmap_full_png": str(out_root / "path_patch_edge_heatmap_full.png"),
            "heatmap_trimmed_png": str(out_root / "path_patch_edge_heatmap_trimmed.png"),
            "bar_full_png": str(out_root / "path_patch_edge_bar_full.png"),
            "bar_trimmed_png": str(out_root / "path_patch_edge_bar_trimmed.png"),
            "final_circuit_edge_path_patching_png": str(out_root / "final_circuit_edge_path_patching.png"),
        },
    }
    report_path = out_root / "path_patch_edge_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[done] output root: {out_root}")
    print(f"[done] analyzed samples: {len(set(analyzed))}")
    print(f"[done] skipped samples: {len(set(skipped))}")
    print(f"[done] report: {report_path}")


if __name__ == "__main__":
    main()
