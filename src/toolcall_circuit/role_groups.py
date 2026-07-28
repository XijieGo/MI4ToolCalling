#!/usr/bin/env python3
"""
Cross-sample causal validation at semantic-role level.

For each valid clean/corrupt pair:
1) evaluate full consensus circuit (sufficiency / necessity);
2) evaluate each semantic role group alone;
3) evaluate "full minus role group" to quantify role necessity inside the full circuit.

Outputs:
- per-sample CSV
- aggregated metrics with bootstrap CI
- role-group causal heatmap / bar figure
- detailed and coarse circuit figures
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
from pathlib import Path
from statistics import median
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import FancyArrowPatch
from tqdm.auto import tqdm

if __package__ in {None, ""}:
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from toolcall_circuit.dataset import get_tool_call_target_spec, load_summary_records, resolve_distractor_token
from toolcall_circuit.paths import MODEL_PATH_DEFAULT, RESULTS_ROOT
from toolcall_circuit.single_sample import (
    draw_circuit,
    evaluate_on_base_with_source,
    load_hooked_qwen3,
    objective_from_logits,
)


def parse_layer(node: str) -> int:
    if node.startswith("MLP"):
        return int(node[3:])
    m = re.fullmatch(r"L(\d+)H(\d+)", node)
    if m is None:
        raise ValueError(f"Unknown node name: {node}")
    return int(m.group(1))


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


def bootstrap_ci(
    values: Sequence[float],
    n_boot: int,
    seed: int,
    stat: str = "median",
) -> Dict[str, float]:
    vals = finite(values)
    if not vals:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan")}
    rng = random.Random(seed)
    n = len(vals)
    boot: List[float] = []
    for _ in range(n_boot):
        sample = [vals[rng.randrange(n)] for __ in range(n)]
        if stat == "mean":
            boot.append(float(np.mean(sample)))
        else:
            boot.append(float(np.median(sample)))
    boot.sort()
    lo_idx = max(0, int(0.025 * n_boot))
    hi_idx = min(n_boot - 1, int(0.975 * n_boot))
    return {
        "mean": float(np.mean(boot)),
        "lo": float(boot[lo_idx]),
        "hi": float(boot[hi_idx]),
    }


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
    fig, ax = plt.subplots(figsize=(1.25 * max(6, len(col_labels)), 0.75 * max(4, len(row_labels)) + 1.8))
    finite_vals = matrix[np.isfinite(matrix)]
    if finite_vals.size == 0:
        vmax = 1.0
    else:
        vmax = float(np.percentile(np.abs(finite_vals), percentile_clip))
        vmax = max(vmax, 1e-6)
    im = ax.imshow(matrix, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_title(f"{title}\nSymmetric clipping at ±{vmax:.3f}", pad=12)
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=28, ha="right")
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels)
    cbar = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
    cbar.set_label("Median effect (center=0)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_group_bar(
    groups: Sequence[str],
    values: Sequence[float],
    cis: Sequence[Dict[str, float]],
    title: str,
    ylabel: str,
    out_path: Path,
) -> None:
    apply_plot_style()
    x = np.arange(len(groups))
    y = np.array(values, dtype=np.float64)
    lo = np.array([float(c.get("lo", np.nan)) for c in cis], dtype=np.float64)
    hi = np.array([float(c.get("hi", np.nan)) for c in cis], dtype=np.float64)
    yerr = np.vstack([np.maximum(0.0, y - lo), np.maximum(0.0, hi - y)])

    fig, ax = plt.subplots(figsize=(max(8.2, 0.95 * len(groups)), 4.8), constrained_layout=True)
    ax.bar(x, y, color="#2a6f97", edgecolor="#1f1f1f", linewidth=0.8)
    ax.errorbar(x, y, yerr=yerr, fmt="none", ecolor="#1f1f1f", elinewidth=1.2, capsize=3)
    ax.axhline(0.0, color="#444444", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(groups, rotation=25, ha="right")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def collect_node_cache_cpu(model, tokens: torch.Tensor, nodes: Sequence[str]) -> Dict[str, torch.Tensor]:
    names = set()
    for node in nodes:
        if node.startswith("MLP"):
            names.add(f"blocks.{int(node[3:])}.hook_mlp_out")
        else:
            layer = parse_layer(node)
            names.add(f"blocks.{layer}.attn.hook_z")
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens, names_filter=lambda n: n in names)
    return {k: v.detach().cpu() for k, v in cache.items()}


def clean_group_name(name: str) -> str:
    return name.replace("_", " ").title()


def build_role_groups(core_nodes: Sequence[str], node_summary: Dict[str, Dict[str, object]]) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    primary: Dict[str, List[str]] = {
        "tool_tag_reader": [],
        "query_reader": [],
        "format_router": [],
        "primary_writer_mlp": [],
        "support_mlp": [],
        "aux_mlp": [],
    }

    for n in core_nodes:
        role = str(node_summary.get(n, {}).get("role", ""))
        if n.startswith("L"):
            if "Tool-Tag Reader" in role:
                primary["tool_tag_reader"].append(n)
            elif "Query Reader" in role:
                primary["query_reader"].append(n)
            elif "Format Router" in role:
                primary["format_router"].append(n)
        elif n.startswith("MLP"):
            if "Primary Writer MLP" in role:
                primary["primary_writer_mlp"].append(n)
            elif "Support" in role:
                primary["support_mlp"].append(n)
            elif "Aux" in role:
                primary["aux_mlp"].append(n)

    primary = {k: sorted(v, key=parse_layer) for k, v in primary.items() if v}
    heads_all = sorted([n for n in core_nodes if n.startswith("L")], key=parse_layer)
    mlps_all = sorted([n for n in core_nodes if n.startswith("MLP")], key=parse_layer)
    extended: Dict[str, List[str]] = dict(primary)
    if heads_all:
        extended["all_heads"] = heads_all
    if mlps_all:
        extended["all_mlps"] = mlps_all
    return primary, extended


def role_label_for_node(node: str, primary_groups: Dict[str, List[str]]) -> str:
    for g, members in primary_groups.items():
        if node in set(members):
            return g
    return "other"


def draw_coarse_circuit(
    core_nodes: Sequence[str],
    core_edges: Sequence[Tuple[str, str]],
    primary_groups: Dict[str, List[str]],
    out_path: Path,
    title: str,
) -> None:
    input_node = "Input Embed"
    output_node = "Residual Output: <tool_call>"
    node_to_group = {n: role_label_for_node(n, primary_groups) for n in core_nodes}

    # Keep only concrete primary groups in layer order.
    groups = sorted(
        [g for g in primary_groups.keys() if primary_groups[g]],
        key=lambda g: float(np.mean([parse_layer(n) for n in primary_groups[g]])),
    )
    all_nodes = [input_node] + groups + [output_node]

    y_pos: Dict[str, float] = {input_node: -1.5, output_node: 30.0}
    for g in groups:
        y_pos[g] = float(np.mean([parse_layer(n) for n in primary_groups[g]]))

    x_pos: Dict[str, float] = {n: 0.0 for n in all_nodes}

    group_edges: Dict[Tuple[str, str], int] = defaultdict(int)
    for s, t in core_edges:
        s = str(s)
        t = str(t)
        if s == input_node and t in core_nodes:
            gs = input_node
            gt = node_to_group.get(t, "other")
        elif s in core_nodes and t == output_node:
            gs = node_to_group.get(s, "other")
            gt = output_node
        elif s in core_nodes and t in core_nodes:
            gs = node_to_group.get(s, "other")
            gt = node_to_group.get(t, "other")
        else:
            continue
        if gs in set(all_nodes) and gt in set(all_nodes) and gs != gt:
            group_edges[(gs, gt)] += 1

    # Ensure every non-output node has out-degree.
    outdeg: Dict[str, int] = defaultdict(int)
    indeg: Dict[str, int] = defaultdict(int)
    for s, t in group_edges:
        outdeg[s] += 1
        indeg[t] += 1
    for g in groups:
        if indeg[g] == 0:
            group_edges[(input_node, g)] += 1
            outdeg[input_node] += 1
            indeg[g] += 1
    for g in groups:
        if outdeg[g] == 0:
            group_edges[(g, output_node)] += 1
            outdeg[g] += 1

    apply_plot_style()
    fig, ax = plt.subplots(figsize=(8.0, 10.5), constrained_layout=True)
    ax.set_title(title)
    ax.axis("off")

    max_w = max(group_edges.values()) if group_edges else 1
    for (s, t), w in sorted(group_edges.items(), key=lambda x: (y_pos[x[0][0]], y_pos[x[0][1]])):
        x1, y1 = x_pos[s], y_pos[s]
        x2, y2 = x_pos[t], y_pos[t]
        rad = 0.22 if (s, t) != (input_node, output_node) else 0.0
        arrow = FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle="->",
            mutation_scale=14,
            linewidth=1.2 + 2.0 * (w / max_w),
            color="#8a4f28",
            connectionstyle=f"arc3,rad={rad}",
            alpha=0.85,
            zorder=1,
        )
        ax.add_patch(arrow)

    for n in all_nodes:
        x, y = x_pos[n], y_pos[n]
        if n in {input_node, output_node}:
            fc = "#bcd3ea"
            txt = n
            fs = 12
            size = 260
        else:
            fc = "#f5ede7"
            members = ", ".join(primary_groups[n])
            txt = f"{clean_group_name(n)}\n[{members}]"
            fs = 10
            size = 240
        ax.scatter([x], [y], s=size, c=fc, edgecolors="#2e2e2e", linewidths=1.5, zorder=3)
        ax.text(
            x + 0.35,
            y,
            txt,
            va="center",
            ha="left",
            fontsize=fs,
            zorder=4,
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.2, "alpha": 0.86},
        )

    ys = list(y_pos.values())
    ax.set_ylim(min(ys) - 1.2, max(ys) + 1.5)
    ax.set_xlim(-2.2, 6.6)
    fig.savefig(out_path, dpi=230, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-sample semantic-role causal validation.")
    parser.add_argument("--input-root", type=str, default=str(RESULTS_ROOT / "manual_run" / "batch"))
    parser.add_argument(
        "--semantic-report",
        type=str,
        default=str(RESULTS_ROOT / "manual_run" / "semantic_roles" / "semantic_roles_report.json"),
    )
    parser.add_argument("--aggregate-summary", type=str, default="")
    parser.add_argument("--model-path", type=str, default=str(MODEL_PATH_DEFAULT))
    parser.add_argument("--output-root", type=str, default=str(RESULTS_ROOT / "manual_run" / "semantic_roles"))
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--gap-min", type=float, default=0.5)
    parser.add_argument("--max-samples", type=int, default=0, help="0 means all valid samples.")
    parser.add_argument("--sample-rank-start", type=int, default=0)
    parser.add_argument("--sample-rank-end", type=int, default=0)
    parser.add_argument("--q-start", type=int, default=1)
    parser.add_argument("--q-end", type=int, default=10_000)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    input_root = Path(args.input_root).resolve()
    out_root = Path(args.output_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    sem_report = json.loads(Path(args.semantic_report).resolve().read_text(encoding="utf-8"))
    core_nodes = list(sem_report.get("core_nodes", []))
    node_summary = sem_report.get("node_summary", {})
    if not core_nodes:
        raise ValueError("No core_nodes found in semantic report.")

    agg_candidates: List[Path] = []
    if args.aggregate_summary:
        agg_candidates.append(Path(args.aggregate_summary).resolve())
    agg_candidates.extend(
        [
            input_root.parent / f"{input_root.name}_aggregate" / "global_core_summary.json",
            input_root.parent / "aggregate" / "global_core_summary.json",
        ]
    )
    agg_summary = None
    for p in agg_candidates:
        if p.exists():
            agg_summary = json.loads(p.read_text(encoding="utf-8"))
            break
    if agg_summary is None:
        raise FileNotFoundError("Could not find aggregate global_core_summary.json.")
    core_edges = [tuple(e) for e in agg_summary.get("core_edges", [])]

    primary_groups, extended_groups = build_role_groups(core_nodes, node_summary)
    if not primary_groups:
        raise ValueError("No non-empty semantic role groups were built.")

    model, tokenizer = load_hooked_qwen3(args.model_path, device=args.device, dtype=torch.bfloat16)
    model.set_use_attn_result(False)
    model.set_use_split_qkv_input(False)
    model.set_use_hook_mlp_in(False)
    target_spec = get_tool_call_target_spec(tokenizer, target_text="<tool_call>")
    if not target_spec.is_single_token:
        raise NotImplementedError(
            f"<tool_call> tokenization is no longer single-token ({target_spec.token_ids}); "
            "upgrade the role-group objective before running this workflow."
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
        cp = Path(s["clean_prompt"])
        rp = Path(s["corrupt_prompt"])
        if not cp.exists() or not rp.exists():
            continue
        sample_infos.append((summary_record.sample_id, sample_rank, legacy_index, s, cp, rp))
    if args.max_samples > 0:
        sample_infos = sample_infos[: args.max_samples]
    if not sample_infos:
        raise ValueError("No valid samples selected.")

    rows: List[Dict[str, object]] = []
    skipped: List[str] = []
    analyzed: List[str] = []
    core_set = set(core_nodes)

    pbar = tqdm(sample_infos, desc="Role-group causal", dynamic_ncols=True)
    for sample_id, sample_rank, legacy_index, summary, clean_prompt, corrupt_prompt in pbar:
        clean_text = clean_prompt.read_text(encoding="utf-8")
        corrupt_text = corrupt_prompt.read_text(encoding="utf-8")
        clean_tokens = model.to_tokens(clean_text, prepend_bos=False)
        corrupt_tokens = model.to_tokens(corrupt_text, prepend_bos=False)
        if clean_tokens.shape != corrupt_tokens.shape:
            skipped.append(sample_id)
            continue

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
            clean_cache = collect_node_cache_cpu(model, clean_tokens, core_nodes)
            corrupt_cache = collect_node_cache_cpu(model, corrupt_tokens, core_nodes)
        except torch.OutOfMemoryError:
            skipped.append(sample_id)
            model.reset_hooks()
            gc.collect()
            torch.cuda.empty_cache()
            continue

        def suff_ratio(patch_nodes: Sequence[str]) -> float:
            obj = evaluate_on_base_with_source(
                model=model,
                base_tokens=corrupt_tokens,
                source_cache_cpu=clean_cache,
                patch_nodes=patch_nodes,
                target_token=target_token,
                distractor_token=distractor,
            )
            return (obj - corrupt_obj) / gap

        def nec_ratio(patch_nodes: Sequence[str]) -> float:
            obj = evaluate_on_base_with_source(
                model=model,
                base_tokens=clean_tokens,
                source_cache_cpu=corrupt_cache,
                patch_nodes=patch_nodes,
                target_token=target_token,
                distractor_token=distractor,
            )
            return (clean_obj - obj) / gap

        try:
            full_suff = suff_ratio(core_nodes)
            full_nec = nec_ratio(core_nodes)
        except torch.OutOfMemoryError:
            skipped.append(sample_id)
            model.reset_hooks()
            gc.collect()
            torch.cuda.empty_cache()
            continue

        rows.append(
            {
                "sample_id": sample_id,
                "sample_rank": sample_rank,
                "q_index": legacy_index,
                "group": "full_core",
                "n_nodes": len(core_nodes),
                "suff_ratio": full_suff,
                "nec_ratio": full_nec,
                "suff_minus_ratio": float("nan"),
                "nec_minus_ratio": float("nan"),
                "delta_full_suff_drop": 0.0,
                "delta_full_nec_drop": 0.0,
                "gap": gap,
            }
        )

        for g_name, members in extended_groups.items():
            if not members:
                continue
            member_set = set(members)
            minus_nodes = [n for n in core_nodes if n not in member_set]
            try:
                g_suff = suff_ratio(members)
                g_nec = nec_ratio(members)
                if minus_nodes:
                    g_suff_minus = suff_ratio(minus_nodes)
                    g_nec_minus = nec_ratio(minus_nodes)
                else:
                    g_suff_minus = float("nan")
                    g_nec_minus = float("nan")
            except torch.OutOfMemoryError:
                continue

            rows.append(
                {
                    "sample_id": sample_id,
                    "sample_rank": sample_rank,
                    "q_index": legacy_index,
                    "group": g_name,
                    "n_nodes": len(members),
                    "suff_ratio": g_suff,
                    "nec_ratio": g_nec,
                    "suff_minus_ratio": g_suff_minus,
                    "nec_minus_ratio": g_nec_minus,
                    "delta_full_suff_drop": full_suff - g_suff_minus if math.isfinite(g_suff_minus) else float("nan"),
                    "delta_full_nec_drop": full_nec - g_nec_minus if math.isfinite(g_nec_minus) else float("nan"),
                    "gap": gap,
                }
            )

        analyzed.append(sample_id)
        model.reset_hooks()
        del clean_cache
        del corrupt_cache
        gc.collect()
        torch.cuda.empty_cache()

    # Save per-sample table.
    per_sample_csv = out_root / "role_group_per_sample.csv"
    fieldnames = [
        "sample_id",
        "sample_rank",
        "q_index",
        "group",
        "n_nodes",
        "suff_ratio",
        "nec_ratio",
        "suff_minus_ratio",
        "nec_minus_ratio",
        "delta_full_suff_drop",
        "delta_full_nec_drop",
        "gap",
    ]
    with per_sample_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # Aggregate by group.
    by_group: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for r in rows:
        by_group[str(r["group"])].append(r)

    summary_rows: List[Dict[str, object]] = []
    group_order = [g for g in extended_groups.keys() if g in by_group]
    if "full_core" in by_group:
        group_order = ["full_core"] + group_order

    for i, g in enumerate(group_order):
        grp = by_group[g]
        suff_vals = [float(x["suff_ratio"]) for x in grp]
        nec_vals = [float(x["nec_ratio"]) for x in grp]
        suff_minus_vals = [float(x["suff_minus_ratio"]) for x in grp]
        nec_minus_vals = [float(x["nec_minus_ratio"]) for x in grp]
        drop_s_vals = [float(x["delta_full_suff_drop"]) for x in grp]
        drop_n_vals = [float(x["delta_full_nec_drop"]) for x in grp]
        summary_rows.append(
            {
                "group": g,
                "n_samples": len(grp),
                "n_nodes": int(grp[0]["n_nodes"]),
                "suff_median": med(suff_vals),
                "suff_mean": mean(suff_vals),
                "suff_median_ci": bootstrap_ci(suff_vals, n_boot=args.bootstrap, seed=args.seed + 7 + i),
                "nec_median": med(nec_vals),
                "nec_mean": mean(nec_vals),
                "nec_median_ci": bootstrap_ci(nec_vals, n_boot=args.bootstrap, seed=args.seed + 113 + i),
                "suff_minus_median": med(suff_minus_vals),
                "nec_minus_median": med(nec_minus_vals),
                "drop_full_suff_median": med(drop_s_vals),
                "drop_full_suff_ci": bootstrap_ci(drop_s_vals, n_boot=args.bootstrap, seed=args.seed + 233 + i),
                "drop_full_nec_median": med(drop_n_vals),
                "drop_full_nec_ci": bootstrap_ci(drop_n_vals, n_boot=args.bootstrap, seed=args.seed + 353 + i),
            }
        )

    # Aggregated CSV.
    summary_csv = out_root / "role_group_summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "group",
                "n_samples",
                "n_nodes",
                "suff_median",
                "suff_ci_lo",
                "suff_ci_hi",
                "nec_median",
                "nec_ci_lo",
                "nec_ci_hi",
                "drop_full_suff_median",
                "drop_full_suff_ci_lo",
                "drop_full_suff_ci_hi",
                "drop_full_nec_median",
                "drop_full_nec_ci_lo",
                "drop_full_nec_ci_hi",
            ]
        )
        for r in summary_rows:
            w.writerow(
                [
                    r["group"],
                    r["n_samples"],
                    r["n_nodes"],
                    r["suff_median"],
                    r["suff_median_ci"]["lo"],
                    r["suff_median_ci"]["hi"],
                    r["nec_median"],
                    r["nec_median_ci"]["lo"],
                    r["nec_median_ci"]["hi"],
                    r["drop_full_suff_median"],
                    r["drop_full_suff_ci"]["lo"],
                    r["drop_full_suff_ci"]["hi"],
                    r["drop_full_nec_median"],
                    r["drop_full_nec_ci"]["lo"],
                    r["drop_full_nec_ci"]["hi"],
                ]
            )

    # Plot bars for primary groups only.
    primary_order = [g for g in primary_groups.keys() if g in by_group]
    primary_rows = [next(r for r in summary_rows if r["group"] == g) for g in primary_order]
    labels = [clean_group_name(g) for g in primary_order]
    save_group_bar(
        groups=labels,
        values=[float(r["suff_median"]) for r in primary_rows],
        cis=[r["suff_median_ci"] for r in primary_rows],
        title="Role Group Sufficiency (Patch Group Alone)",
        ylabel="Recovery ratio vs gap",
        out_path=out_root / "role_group_sufficiency.png",
    )
    save_group_bar(
        groups=labels,
        values=[float(r["drop_full_nec_median"]) for r in primary_rows],
        cis=[r["drop_full_nec_ci"] for r in primary_rows],
        title="Role Group Necessity in Full Circuit (Drop in Necessity When Removed)",
        ylabel="Delta necessity ratio",
        out_path=out_root / "role_group_necessity_drop.png",
    )

    heat_cols = [
        "suff_median",
        "nec_median",
        "drop_full_suff_median",
        "drop_full_nec_median",
    ]
    heat_matrix = np.array(
        [
            [
                float(r["suff_median"]),
                float(r["nec_median"]),
                float(r["drop_full_suff_median"]),
                float(r["drop_full_nec_median"]),
            ]
            for r in primary_rows
        ],
        dtype=np.float64,
    )
    save_diverging_heatmap(
        matrix=heat_matrix,
        row_labels=labels,
        col_labels=heat_cols,
        title="Semantic Role Group Causal Metrics (Median)",
        out_path=out_root / "role_group_causal_heatmap.png",
    )

    # Circuit figures in required naming.
    draw_circuit(
        nodes=core_nodes,
        edges=core_edges,
        out_path=out_root / "final_circuit.png",
        title="Final Detailed Circuit (Consensus Core)",
    )
    draw_coarse_circuit(
        core_nodes=core_nodes,
        core_edges=core_edges,
        primary_groups=primary_groups,
        out_path=out_root / "final_circuit_coarse.png",
        title="Final Coarse Circuit (Semantic Role Groups)",
    )

    report = {
        "n_input_samples": len(sample_infos),
        "n_analyzed_samples": len(set(analyzed)),
        "analyzed_sample_ids": sorted(set(analyzed)),
        "skipped_sample_ids": sorted(set(skipped)),
        "gap_min": args.gap_min,
        "bootstrap": args.bootstrap,
        "seed": args.seed,
        "core_nodes": core_nodes,
        "core_edges": core_edges,
        "primary_groups": primary_groups,
        "extended_groups": extended_groups,
        "summary_rows": summary_rows,
        "artifacts": {
            "role_group_per_sample_csv": str(per_sample_csv),
            "role_group_summary_csv": str(summary_csv),
            "role_group_sufficiency_png": str(out_root / "role_group_sufficiency.png"),
            "role_group_necessity_drop_png": str(out_root / "role_group_necessity_drop.png"),
            "role_group_causal_heatmap_png": str(out_root / "role_group_causal_heatmap.png"),
            "final_circuit_png": str(out_root / "final_circuit.png"),
            "final_circuit_coarse_png": str(out_root / "final_circuit_coarse.png"),
        },
    }
    report_path = out_root / "role_group_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] output root: {out_root}")
    print(f"[done] analyzed samples: {len(set(analyzed))}")
    print(f"[done] report: {report_path}")


if __name__ == "__main__":
    main()
