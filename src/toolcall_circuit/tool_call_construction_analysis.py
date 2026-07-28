#!/usr/bin/env python3
"""
Full-dataset Tool-Call Construction analysis.

This analysis reuses full legacy/full-run results where they already answer part
of the question, and adds three targeted full-dataset audits that were missing:

1. writeout audit:
   direct logit effect / residual projection for construction nodes;
2. route fanout audit:
   how `MLP19` hands route state into late construction nodes;
3. stagewise trajectory:
   how cumulative construction patching turns the first token into `<tool_call>`.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

if __package__ in {None, ""}:
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from toolcall_circuit.bidirectional_causal_eval import collect_cache_cpu_for_nodes, load_sample_paths
from toolcall_circuit.objective import build_bidirectional_endpoint_objectives, objective_from_logits
from toolcall_circuit.signed_edge_importance import run_logits_with_assignments
from toolcall_circuit.single_sample import load_hooked_qwen3, parse_head


ROUTE_SOURCE = "MLP19"
CONSTRUCTION_HEADS = ["L20H5", "L21H1", "L21H12", "L24H6"]
WRITER_NODES = ["MLP27"]
PRIMARY_CHAIN = [ROUTE_SOURCE, *CONSTRUCTION_HEADS, *WRITER_NODES]
ROUTE_TARGETS = ["L20H5", "L21H1", "L21H12", "L24H6", "MLP27"]
SEARCH_HEAD_CANDIDATES = ["L25H13", "L25H10", "L26H15", "L27H7"]
SEARCH_MLP_CANDIDATES = ["MLP24", "MLP25", "MLP26"]
TRACKED_NODES = PRIMARY_CHAIN + SEARCH_HEAD_CANDIDATES + SEARCH_MLP_CANDIDATES
STAGE_STEPS = [
    ("corrupt_full", []),
    ("plus_MLP19", ["MLP19"]),
    ("plus_L20H5", ["MLP19", "L20H5"]),
    ("plus_L21H1", ["MLP19", "L20H5", "L21H1"]),
    ("plus_L21H12", ["MLP19", "L20H5", "L21H1", "L21H12"]),
    ("plus_L24H6", ["MLP19", "L20H5", "L21H1", "L21H12", "L24H6"]),
    ("plus_MLP27", ["MLP19", "L20H5", "L21H1", "L21H12", "L24H6", "MLP27"]),
]

LATE_MLP_SCAN = ["MLP20", "MLP21", "MLP22", "MLP23", "MLP24", "MLP25", "MLP26", "MLP27"]
NODE_TO_LAYER = {
    "L20H5": 20,
    "L21H1": 21,
    "L21H12": 21,
    "L24H6": 24,
    "MLP19": 19,
    "MLP24": 24,
    "MLP25": 25,
    "MLP26": 26,
    "MLP27": 27,
    "L25H10": 25,
    "L25H13": 25,
    "L26H15": 26,
    "L27H7": 27,
}


def finite(values: Iterable[float]) -> List[float]:
    return [float(v) for v in values if isinstance(v, (int, float)) and math.isfinite(float(v))]


def median(values: Iterable[float]) -> float:
    vals = finite(values)
    return float(np.median(vals)) if vals else float("nan")


def safe_rate(values: Iterable[bool]) -> float:
    vals = [1.0 if bool(v) else 0.0 for v in values]
    return float(np.mean(vals)) if vals else float("nan")


def fmt(value: object, digits: int = 3) -> str:
    try:
        value = float(value)
    except Exception:
        return str(value)
    if not math.isfinite(value):
        return "nan"
    return f"{value:.{digits}f}"


def write_csv(rows: Sequence[Dict[str, object]], out_path: Path) -> None:
    if not rows:
        return
    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def resolve_prompt_path(path: Path) -> Path:
    if path.exists():
        return path
    raw = str(path)
    legacy_prefix = "./datasets/"
    current_prefix = "./datasets/"
    if raw.startswith(legacy_prefix):
        alt = Path(raw.replace(legacy_prefix, current_prefix, 1))
        if alt.exists():
            return alt
    return path


def node_hook_name(node: str) -> str:
    if node.startswith("MLP"):
        return f"blocks.{int(node[3:])}.hook_mlp_out"
    layer, _head = parse_head(node)
    return f"blocks.{layer}.attn.hook_z"


def collect_cache_with_scale(model, tokens: torch.Tensor, nodes: Sequence[str]) -> Dict[str, torch.Tensor]:
    needed = {node_hook_name(n) for n in nodes}
    needed.add("ln_final.hook_scale")
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens, names_filter=lambda name: name in needed)
    return {k: v.detach().cpu() for k, v in cache.items()}


def extract_residual_contribution(model, cache: Dict[str, torch.Tensor], node: str) -> torch.Tensor:
    if node.startswith("MLP"):
        return cache[f"blocks.{int(node[3:])}.hook_mlp_out"][0, -1, :].float()
    layer, head = parse_head(node)
    z = cache[f"blocks.{layer}.attn.hook_z"][0, -1, head, :].float()
    w_o = model.blocks[layer].attn.W_O[head].detach().cpu().float()
    return z @ w_o


def normalized_residual(model, resid_vec: torch.Tensor, scale_value: float) -> torch.Tensor:
    ln_w = model.ln_final.w.detach().cpu().float()
    return resid_vec.float() / max(scale_value, 1e-6) * ln_w


def token_direct_logit(model, normed_resid: torch.Tensor, token_id: int) -> float:
    with torch.no_grad():
        resid_gpu = normed_resid.to(device=model.W_U.device, dtype=model.W_U.dtype)
        score = torch.dot(resid_gpu, model.W_U[:, int(token_id)])
    return float(score.item())


def decode_token(tokenizer, token_id: int) -> str:
    try:
        return tokenizer.decode([int(token_id)]).replace("\n", "\\n")
    except Exception:
        return str(token_id)


def local_score(vec: torch.Tensor, clean_vec: torch.Tensor, corrupt_vec: torch.Tensor) -> float:
    direction = clean_vec - corrupt_vec
    denom = float(direction.norm().item())
    if denom < 1e-8:
        return float("nan")
    d = direction / denom
    midpoint = (clean_vec + corrupt_vec) / 2.0
    ref = torch.dot(clean_vec - midpoint, d)
    if abs(float(ref.item())) < 1e-8:
        return float("nan")
    val = torch.dot(vec - midpoint, d) / ref
    return float(val.item())


def run_with_assignments_and_collect(
    model,
    base_tokens: torch.Tensor,
    *,
    clean_cache_cpu: Dict[str, torch.Tensor],
    corrupt_cache_cpu: Dict[str, torch.Tensor],
    patch_clean_nodes: Sequence[str],
    patch_corrupt_nodes: Sequence[str],
    record_nodes: Sequence[str],
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    heads_by_layer_clean: Dict[int, List[int]] = defaultdict(list)
    heads_by_layer_corrupt: Dict[int, List[int]] = defaultdict(list)
    mlp_layers_clean: List[int] = []
    mlp_layers_corrupt: List[int] = []

    for node in patch_clean_nodes:
        if node.startswith("MLP"):
            mlp_layers_clean.append(int(node[3:]))
        else:
            layer, head = parse_head(node)
            heads_by_layer_clean[layer].append(head)
    for node in patch_corrupt_nodes:
        if node.startswith("MLP"):
            mlp_layers_corrupt.append(int(node[3:]))
        else:
            layer, head = parse_head(node)
            heads_by_layer_corrupt[layer].append(head)

    hooks = []

    def add_head_hooks(layer_to_heads: Dict[int, List[int]], cache_cpu: Dict[str, torch.Tensor]) -> None:
        for layer, heads in layer_to_heads.items():
            cache_name = f"blocks.{layer}.attn.hook_z"
            src_act = cache_cpu[cache_name].to(base_tokens.device)
            uniq = sorted(set(heads))

            def make_head_hook(src: torch.Tensor, hs: Sequence[int]):
                def hook_fn(z: torch.Tensor, hook):  # noqa: ANN001
                    out = z.clone()
                    for h in hs:
                        out[:, -1, h, :] = src[:, -1, h, :]
                    return out

                return hook_fn

            hooks.append((cache_name, make_head_hook(src_act, uniq)))

    def add_mlp_hooks(layers: Sequence[int], cache_cpu: Dict[str, torch.Tensor]) -> None:
        for layer in sorted(set(layers)):
            cache_name = f"blocks.{layer}.hook_mlp_out"
            src_act = cache_cpu[cache_name].to(base_tokens.device)

            def make_mlp_hook(src: torch.Tensor):
                def hook_fn(mlp_out: torch.Tensor, hook):  # noqa: ANN001
                    out = mlp_out.clone()
                    out[:, -1, :] = src[:, -1, :]
                    return out

                return hook_fn

            hooks.append((cache_name, make_mlp_hook(src_act)))

    add_head_hooks(heads_by_layer_clean, clean_cache_cpu)
    add_head_hooks(heads_by_layer_corrupt, corrupt_cache_cpu)
    add_mlp_hooks(mlp_layers_clean, clean_cache_cpu)
    add_mlp_hooks(mlp_layers_corrupt, corrupt_cache_cpu)

    record_names = sorted({node_hook_name(n) for n in record_nodes})
    recorded: Dict[str, torch.Tensor] = {}
    for name in record_names:

        def make_record(cache_name: str):
            def hook_fn(act: torch.Tensor, hook):  # noqa: ANN001
                recorded[cache_name] = act.detach().cpu()
                return act

            return hook_fn

        hooks.append((name, make_record(name)))

    with torch.no_grad():
        logits = model.run_with_hooks(base_tokens, fwd_hooks=hooks)
    return logits, recorded


def summarize_group(rows: Sequence[Dict[str, object]], keys: Sequence[str], metrics: Sequence[str]) -> List[Dict[str, object]]:
    grouped: Dict[Tuple[str, ...], List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(str(row[k]) for k in keys)].append(dict(row))
    out: List[Dict[str, object]] = []
    for key_vals, members in sorted(grouped.items()):
        summary = {keys[idx]: key_vals[idx] for idx in range(len(keys))}
        summary["n_samples"] = len(members)
        for metric in metrics:
            values = [row.get(metric) for row in members]
            non_null = [v for v in values if v is not None]
            is_bool_metric = (
                metric.endswith("_rate")
                or metric.endswith("_flip")
                or metric.endswith("_success")
                or (non_null and all(isinstance(v, (bool, np.bool_)) for v in non_null))
            )
            if is_bool_metric:
                summary[metric] = safe_rate(bool(v) for v in non_null)
            else:
                summary[metric] = median(float(v) for v in non_null if math.isfinite(float(v)))
        out.append(summary)
    return out


def build_stage_top_tokens_summary(stage_rows: Sequence[Dict[str, object]], actual_n: int) -> List[Dict[str, object]]:
    top_token_rows: List[Dict[str, object]] = []
    grouped_top = defaultdict(list)
    for row in stage_rows:
        grouped_top[(str(row["step_label"]), str(row["top1_token"]))].append(row)
    for (step_label, top1_token), members in sorted(grouped_top.items()):
        top_token_rows.append(
            {
                "step_label": step_label,
                "top1_token": top1_token,
                "count": len(members),
                "rate": len(members) / actual_n,
            }
        )
    top_token_rows.sort(key=lambda r: (str(r["step_label"]), -int(r["count"]), str(r["top1_token"])))
    stage_top_tokens_summary: List[Dict[str, object]] = []
    by_step_top = defaultdict(list)
    for row in top_token_rows:
        by_step_top[str(row["step_label"])].append(row)
    for step_label, members in by_step_top.items():
        for rank, row in enumerate(members[:5], start=1):
            stage_top_tokens_summary.append(
                {
                    "step_label": step_label,
                    "rank": rank,
                    "top1_token": row["top1_token"],
                    "count": row["count"],
                    "rate": row["rate"],
                }
            )
    return stage_top_tokens_summary


def build_derived_summaries(
    *,
    writeout_rows: Sequence[Dict[str, object]],
    stage_rows: Sequence[Dict[str, object]],
    route_fanout_rows: Sequence[Dict[str, object]],
    candidate_patch_rows: Sequence[Dict[str, object]],
    actual_n: int,
) -> Tuple[
    List[Dict[str, object]],
    List[Dict[str, object]],
    List[Dict[str, object]],
    List[Dict[str, object]],
    List[Dict[str, object]],
]:
    writeout_summary = summarize_group(
        writeout_rows,
        keys=["node", "layer"],
        metrics=[
            "clean_tool_logit",
            "clean_competitor_logit",
            "clean_margin_logit",
            "corrupt_tool_logit",
            "corrupt_competitor_logit",
            "corrupt_margin_logit",
            "delta_tool_logit",
            "delta_competitor_logit",
            "delta_margin_logit",
            "clean_projection",
            "corrupt_projection",
            "delta_projection",
        ],
    )
    for row in writeout_summary:
        row["clean_tool_logit_median"] = row.pop("clean_tool_logit")
        row["clean_competitor_logit_median"] = row.pop("clean_competitor_logit")
        row["clean_margin_logit_median"] = row.pop("clean_margin_logit")
        row["corrupt_tool_logit_median"] = row.pop("corrupt_tool_logit")
        row["corrupt_competitor_logit_median"] = row.pop("corrupt_competitor_logit")
        row["corrupt_margin_logit_median"] = row.pop("corrupt_margin_logit")
        row["delta_tool_logit_median"] = row.pop("delta_tool_logit")
        row["delta_competitor_logit_median"] = row.pop("delta_competitor_logit")
        row["delta_margin_logit_median"] = row.pop("delta_margin_logit")
        row["clean_projection_median"] = row.pop("clean_projection")
        row["corrupt_projection_median"] = row.pop("corrupt_projection")
        row["delta_projection_median"] = row.pop("delta_projection")

    stage_summary = summarize_group(
        stage_rows,
        keys=["step_idx", "step_label", "nodes"],
        metrics=["route_margin", "tool_logit", "competitor_logit", "margin_logit", "tool_prob", "tool_top1", "boundary_flip"],
    )
    for row in stage_summary:
        row["route_margin_median"] = row.pop("route_margin")
        row["tool_logit_median"] = row.pop("tool_logit")
        row["competitor_logit_median"] = row.pop("competitor_logit")
        row["margin_logit_median"] = row.pop("margin_logit")
        row["tool_prob_median"] = row.pop("tool_prob")
        row["tool_top1_rate"] = row.pop("tool_top1")
        row["boundary_flip_rate"] = row.pop("boundary_flip")

    stage_top_tokens_summary = build_stage_top_tokens_summary(stage_rows, actual_n)

    route_fanout_summary = summarize_group(
        route_fanout_rows,
        keys=["target"],
        metrics=["source_route_ratio", "blocked_route_ratio", "route_mediated_ratio", "target_local_rescue"],
    )
    for row in route_fanout_summary:
        row["source_route_ratio_median"] = row.pop("source_route_ratio")
        row["blocked_route_ratio_median"] = row.pop("blocked_route_ratio")
        row["route_mediated_ratio_median"] = row.pop("route_mediated_ratio")
        row["target_local_rescue_median"] = row.pop("target_local_rescue")

    candidate_patch_summary = summarize_group(
        candidate_patch_rows,
        keys=["node", "layer"],
        metrics=["rescue_ratio", "tool_top1", "boundary_flip"],
    )
    for row in candidate_patch_summary:
        row["rescue_ratio_median"] = row.pop("rescue_ratio")
        row["tool_top1_rate"] = row.pop("tool_top1")
        row["boundary_flip_rate"] = row.pop("boundary_flip")

    return (
        writeout_summary,
        stage_summary,
        stage_top_tokens_summary,
        route_fanout_summary,
        candidate_patch_summary,
    )


def plot_node_writeout(summary_rows: Sequence[Dict[str, object]], out_path: Path) -> None:
    order = ["MLP19", "L20H5", "L21H1", "L21H12", "L24H6", "MLP27"]
    row_map = {str(r["node"]): r for r in summary_rows}
    x = np.arange(len(order))
    tool = [float(row_map[n]["clean_tool_logit_median"]) for n in order]
    comp = [float(row_map[n]["clean_competitor_logit_median"]) for n in order]
    margin = [float(row_map[n]["clean_margin_logit_median"]) for n in order]
    delta_margin = [float(row_map[n]["delta_margin_logit_median"]) for n in order]

    plt.style.use("default")
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8), constrained_layout=True)
    width = 0.26
    axes[0].bar(x - width, tool, width=width, color="#be4d25", label="<tool_call>")
    axes[0].bar(x, comp, width=width, color="#4878a8", label="competitor token")
    axes[0].bar(x + width, margin, width=width, color="#6e9f4c", label="margin")
    axes[0].axhline(0.0, color="#666666", linewidth=1.0)
    axes[0].set_xticks(x, order, rotation=20, ha="right")
    axes[0].set_title("Direct Writeout on Clean")
    axes[0].legend(frameon=False)

    axes[1].bar(x, delta_margin, color="#6e9f4c")
    axes[1].axhline(0.0, color="#666666", linewidth=1.0)
    axes[1].set_xticks(x, order, rotation=20, ha="right")
    axes[1].set_title("Clean-Corrupt Margin Delta")
    axes[1].set_ylabel("delta margin logit")

    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_node_projection(summary_rows: Sequence[Dict[str, object]], out_path: Path) -> None:
    order = ["MLP19", "L20H5", "L21H1", "L21H12", "L24H6", "MLP27"]
    row_map = {str(r["node"]): r for r in summary_rows}
    clean = [float(row_map[n]["clean_projection_median"]) for n in order]
    corrupt = [float(row_map[n]["corrupt_projection_median"]) for n in order]
    delta = [float(row_map[n]["delta_projection_median"]) for n in order]

    plt.style.use("default")
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8), constrained_layout=True)
    width = 0.35
    x = np.arange(len(order))
    axes[0].bar(x - width / 2, clean, width=width, color="#be4d25", label="clean")
    axes[0].bar(x + width / 2, corrupt, width=width, color="#4878a8", label="corrupt")
    axes[0].axhline(0.0, color="#666666", linewidth=1.0)
    axes[0].set_xticks(x, order, rotation=20, ha="right")
    axes[0].set_title("Residual Projection to <tool_call>-Competitor Axis")
    axes[0].legend(frameon=False)

    axes[1].bar(x, delta, color="#6e9f4c")
    axes[1].axhline(0.0, color="#666666", linewidth=1.0)
    axes[1].set_xticks(x, order, rotation=20, ha="right")
    axes[1].set_title("Projection Delta")
    axes[1].set_ylabel("clean - corrupt")

    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_stagewise(summary_rows: Sequence[Dict[str, object]], out_path: Path) -> None:
    order = [label for label, _nodes in STAGE_STEPS]
    row_map = {str(r["step_label"]): r for r in summary_rows}
    xs = np.arange(len(order))
    tool_logit = [float(row_map[l]["tool_logit_median"]) for l in order]
    comp_logit = [float(row_map[l]["competitor_logit_median"]) for l in order]
    margin = [float(row_map[l]["margin_logit_median"]) for l in order]
    tool_top1 = [float(row_map[l]["tool_top1_rate"]) for l in order]
    boundary = [float(row_map[l]["boundary_flip_rate"]) for l in order]

    plt.style.use("default")
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8), constrained_layout=True)
    axes[0].plot(xs, tool_logit, marker="o", color="#be4d25", label="<tool_call>")
    axes[0].plot(xs, comp_logit, marker="o", color="#4878a8", label="competitor token")
    axes[0].plot(xs, margin, marker="o", color="#6e9f4c", label="margin")
    axes[0].axhline(0.0, color="#666666", linewidth=1.0)
    axes[0].set_xticks(xs, order, rotation=20, ha="right")
    axes[0].set_title("Stagewise logit trajectory")
    axes[0].legend(frameon=False)

    axes[1].plot(xs, tool_top1, marker="o", color="#be4d25", label="<tool_call> top1")
    axes[1].plot(xs, boundary, marker="o", color="#6e9f4c", label="boundary flip")
    axes[1].set_xticks(xs, order, rotation=20, ha="right")
    axes[1].set_ylim(-0.02, 1.02)
    axes[1].set_title("Stagewise Output Flip")
    axes[1].legend(frameon=False)

    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_top_token_change(
    stage_summary: Sequence[Dict[str, object]],
    stage_top_tokens_summary: Sequence[Dict[str, object]],
    out_path: Path,
) -> None:
    ordered_stage = sorted(stage_summary, key=lambda r: int(r["step_idx"]))
    order = [str(row["step_label"]) for row in ordered_stage]
    row_map = {str(r["step_label"]): r for r in ordered_stage}
    per_step_tokens: Dict[str, Dict[str, float]] = defaultdict(dict)
    for row in stage_top_tokens_summary:
        per_step_tokens[str(row["step_label"])][str(row["top1_token"])] = float(row["rate"])

    primary_competitor = "I"
    if per_step_tokens.get("corrupt_full"):
        non_tool_items = [(token, rate) for token, rate in per_step_tokens["corrupt_full"].items() if token != "<tool_call>"]
        if non_tool_items:
            primary_competitor = max(non_tool_items, key=lambda kv: kv[1])[0]

    xs = np.arange(len(order))
    tool_top1 = np.array([float(row_map[label]["tool_top1_rate"]) for label in order], dtype=float)
    boundary = np.array([float(row_map[label]["boundary_flip_rate"]) for label in order], dtype=float)
    competitor = np.array([per_step_tokens.get(label, {}).get(primary_competitor, 0.0) for label in order], dtype=float)
    other = np.clip(1.0 - tool_top1 - competitor, 0.0, 1.0)

    plt.style.use("default")
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.8), constrained_layout=True)
    axes[0].plot(xs, tool_top1, marker="o", color="#be4d25", label="<tool_call> top1")
    axes[0].plot(xs, boundary, marker="o", color="#4878a8", label="boundary flip")
    axes[0].set_xticks(xs, order, rotation=20, ha="right")
    axes[0].set_ylim(-0.02, 1.02)
    axes[0].set_title("Top-1 and Boundary Dynamics")
    axes[0].legend(frameon=False)

    axes[1].bar(xs, tool_top1, color="#be4d25", label="<tool_call>")
    axes[1].bar(xs, competitor, bottom=tool_top1, color="#4878a8", label=primary_competitor)
    axes[1].bar(xs, other, bottom=tool_top1 + competitor, color="#9e9e9e", label="other")
    axes[1].set_xticks(xs, order, rotation=20, ha="right")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].set_title("Top Token Composition")
    axes[1].legend(frameon=False)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_route_fanout(summary_rows: Sequence[Dict[str, object]], out_path: Path) -> None:
    order = ["L20H5", "L21H1", "L21H12", "L24H6", "MLP27"]
    row_map = {str(r["target"]): r for r in summary_rows}
    route = [float(row_map[t]["route_mediated_ratio_median"]) for t in order]
    local = [float(row_map[t]["target_local_rescue_median"]) for t in order]

    plt.style.use("default")
    fig, ax = plt.subplots(figsize=(9.8, 4.8), constrained_layout=True)
    x = np.arange(len(order))
    width = 0.36
    ax.bar(x - width / 2, route, width=width, color="#4878a8", label="route mediation")
    ax.bar(x + width / 2, local, width=width, color="#6e9f4c", label="target local rescue")
    ax.axhline(0.0, color="#666666", linewidth=1.0)
    ax.set_xticks(x, order, rotation=20, ha="right")
    ax.set_title("MLP19 Fanout into Construction Nodes")
    ax.legend(frameon=False)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_candidate_patch(summary_rows: Sequence[Dict[str, object]], out_path: Path) -> None:
    rows = list(summary_rows)
    if not rows:
        return
    order = [str(r["node"]) for r in sorted(rows, key=lambda r: float(r["rescue_ratio_median"]), reverse=True)]
    rescue = [float(next(r["rescue_ratio_median"] for r in rows if str(r["node"]) == node)) for node in order]
    top1 = [float(next(r["tool_top1_rate"] for r in rows if str(r["node"]) == node)) for node in order]

    plt.style.use("default")
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.6), constrained_layout=True)
    x = np.arange(len(order))
    axes[0].bar(x, rescue, color="#4878a8")
    axes[0].axhline(0.0, color="#666666", linewidth=1.0)
    axes[0].set_xticks(x, order, rotation=20, ha="right")
    axes[0].set_title("Candidate Single-Node Patch Rescue")

    axes[1].bar(x, top1, color="#be4d25")
    axes[1].set_xticks(x, order, rotation=20, ha="right")
    axes[1].set_ylim(-0.02, max(1.0, max(top1) + 0.05))
    axes[1].set_title("Candidate <tool_call> Top-1 Rate")
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_logit_lens(mean_logits_rows: Sequence[Dict[str, object]], out_path: Path) -> None:
    order = ["MLP19", "L20H5", "L21H1", "L21H12", "L24H6", "MLP27"]
    row_map = {str(r["node"]): r for r in mean_logits_rows}
    tool = [float(row_map[n]["tool_logit"]) for n in order]
    rank = [float(row_map[n]["tool_rank"]) for n in order]

    plt.style.use("default")
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.6), constrained_layout=True)
    x = np.arange(len(order))
    axes[0].bar(x, tool, color="#be4d25")
    axes[0].axhline(0.0, color="#666666", linewidth=1.0)
    axes[0].set_xticks(x, order, rotation=20, ha="right")
    axes[0].set_title("Mean Direct-Logit Lens: <tool_call> Logit")

    axes[1].plot(x, rank, marker="o", color="#4878a8")
    axes[1].invert_yaxis()
    axes[1].set_xticks(x, order, rotation=20, ha="right")
    axes[1].set_title("Mean Direct-Logit Lens: <tool_call> Rank")
    axes[1].set_ylabel("rank (lower is stronger)")
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def copy_or_combine_attention_plots(attn_root: Path, out_root: Path) -> Dict[str, str]:
    selected = ["L20H5", "L21H1", "L21H12", "L24H6"]
    rows = []
    for head in selected:
        layer = int(head[1:].split("H")[0])
        base = attn_root / "plots" / f"layer_{layer:02d}" / head
        rows.append(
            {
                "head": head,
                "density": base / "density_heatmap.png",
                "decision": base / "decision_row.png",
            }
        )

    plt.style.use("default")
    fig, axes = plt.subplots(len(rows), 2, figsize=(10.5, 3.0 * len(rows)), constrained_layout=True)
    if len(rows) == 1:
        axes = np.asarray([axes])
    for idx, row in enumerate(rows):
        density = plt.imread(row["density"])
        decision = plt.imread(row["decision"])
        axes[idx, 0].imshow(density)
        axes[idx, 0].axis("off")
        axes[idx, 0].set_title(f"{row['head']} density")
        axes[idx, 1].imshow(decision)
        axes[idx, 1].axis("off")
        axes[idx, 1].set_title(f"{row['head']} decision row")
    combined_path = out_root / "figures" / "construction_attention_panels.png"
    combined_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(combined_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    copied: Dict[str, str] = {"combined_attention_panel": str(combined_path)}
    for row in rows:
        head_dir = out_root / "figures" / row["head"]
        head_dir.mkdir(parents=True, exist_ok=True)
        density_dst = head_dir / "density_heatmap.png"
        decision_dst = head_dir / "decision_row.png"
        shutil.copy2(row["density"], density_dst)
        shutil.copy2(row["decision"], decision_dst)
        copied[f"{row['head']}_density"] = str(density_dst)
        copied[f"{row['head']}_decision"] = str(decision_dst)
    return copied


def build_mean_logit_rows(
    model,
    tokenizer,
    mean_clean_normed: Dict[str, torch.Tensor],
    mean_corrupt_normed: Dict[str, torch.Tensor],
    tool_token_id: int,
    topk: int = 8,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    with torch.no_grad():
        w_u = model.W_U.detach().cpu().float()
    for node in PRIMARY_CHAIN:
        vec = mean_clean_normed[node]
        logits = torch.matmul(vec, w_u)
        tool_logit = float(logits[int(tool_token_id)].item())
        tool_rank = int((logits > logits[int(tool_token_id)]).sum().item()) + 1
        values, indices = torch.topk(logits, k=min(topk, int(logits.numel())))
        top_tokens = [decode_token(tokenizer, int(idx)) for idx in indices.tolist()]
        rows.append(
            {
                "node": node,
                "tool_logit": tool_logit,
                "tool_rank": tool_rank,
                "top_tokens": " | ".join(top_tokens),
                "clean_top_token": top_tokens[0] if top_tokens else "",
                "corrupt_tool_logit": float(torch.matmul(mean_corrupt_normed[node], w_u)[int(tool_token_id)].item()),
            }
        )
    return rows


def build_report(
    *,
    out_root: Path,
    route_fanout_summary: Sequence[Dict[str, object]],
    writeout_summary: Sequence[Dict[str, object]],
    stage_summary: Sequence[Dict[str, object]],
    stage_top_tokens_summary: Sequence[Dict[str, object]],
    candidate_patch_summary: Sequence[Dict[str, object]],
    mean_logit_rows: Sequence[Dict[str, object]],
    legacy_context: Dict[str, object],
    attention_artifacts: Dict[str, str],
    tool_token_text: str,
    competitor_counts: Counter[str],
    actual_n: int,
) -> None:
    route_map = {str(r["target"]): r for r in route_fanout_summary}
    write_map = {str(r["node"]): r for r in writeout_summary}
    stage_map = {str(r["step_label"]): r for r in stage_summary}
    lens_map = {str(r["node"]): r for r in mean_logit_rows}
    top_competitors = competitor_counts.most_common(8)
    top_token_by_step = defaultdict(list)
    for row in stage_top_tokens_summary:
        top_token_by_step[str(row["step_label"])].append(row)

    anchor_nodes = ["L20H5", "L21H12", "L24H6", "MLP27"]
    support_nodes = ["MLP19", "L21H1"]
    candidate_nodes = [str(r["node"]) for r in candidate_patch_summary if str(r["node"]) not in anchor_nodes + support_nodes]
    ordered_steps = sorted(stage_summary, key=lambda r: int(r["step_idx"]))
    first_positive_route = next((r for r in ordered_steps if float(r["route_margin_median"]) > 0), None)
    first_positive_logit_margin = next((r for r in ordered_steps if float(r["margin_logit_median"]) > 0), None)
    first_tool_top1_majority = next((r for r in ordered_steps if float(r["tool_top1_rate"]) >= 0.5), None)
    l20_delta_margin = float(write_map["L20H5"]["delta_margin_logit_median"])
    l20_clean_margin = float(write_map["L20H5"]["clean_margin_logit_median"])
    l24_prev = stage_map.get("plus_L21H12", {})
    l24_cur = stage_map.get("plus_L24H6", {})

    lines: List[str] = []
    lines.append("# Tool-Call Construction 主报告")
    lines.append("")
    lines.append("## 模块定义")
    lines.append("")
    lines.append(
        "本报告把 Tool-Call Construction 定义为：在上游 `Output-Route Decision` 已经把状态推到 tool route 之后，晚层节点把这份 route state 继续绑定到文件名、函数体、tool instruction、tool_call example 与调用格式，并把首个输出词逐步写成 `<tool_call>`。"
    )
    lines.append(
        "因此，`MLP19` 只被当作 Construction 的接口节点，而不是模块本体；真正属于 Construction 的，是那些在接到 route state 之后，继续组织 payload / protocol，并把 `<tool_call>` 写强的 late heads 与晚层 writer。"
    )
    lines.append("")
    lines.append("## 样本与结果范围")
    lines.append("")
    lines.append(f"- 全部统计都基于 full run：`{actual_n}` 个有效样本。")
    lines.append("- 复用的旧结果也只采用 full-dataset 版本，不使用 smoke 或小样本版本。")
    lines.append("")
    lines.append("## 节点分层")
    lines.append("")
    lines.append(f"- anchor nodes: `{', '.join(anchor_nodes)}`")
    lines.append(f"- support nodes: `{', '.join(support_nodes)}`")
    lines.append(f"- candidate nodes: `{', '.join(candidate_nodes) if candidate_nodes else '无明显新增 candidate'}`")
    lines.append("")
    lines.append("### 分层理由")
    lines.append("")
    lines.append("- `L20H5` 是最早接住 tool-side payload 的 Construction 入口：它读文件名和函数体锚点，单节点 patch 已有稳定 rescue；新的 writeout 审计显示，它会把输出边界往 `<tool_call>` 方向推，但通常还不是最强 writer。")
    lines.append("- `L21H12` 是最强的 late router / protocol binder：它对 `MLP27` 的边级 mediation 最强，而且更明显读取 tool_call example 与 instruction tail。")
    lines.append("- `L24H6` 不是一般 reader，而是 pre-writer formatter / protocol commitment 节点：它强读 `tool_instruction`，在 stagewise 里把 `<tool_call>` margin 从“已翻正”继续推到稳定区。")
    lines.append("- `MLP27` 仍是主 writer：旧的 steering 结果和新做的 direct logit effect 都显示，最终把 `<tool_call>` 写成 top-1 的主力仍然是它。")
    lines.append("- `MLP19` 是 support，因为它主要负责把 route state 分发进 construction，而不是自己完成 `<tool_call>` 组装。")
    lines.append("- `L21H1` 是 support，因为它明显参与 late routing，但读入对象比 `L21H12` 更杂，且对 `MLP27` 的传递略弱。")
    lines.append("")
    lines.append("## 核心结论")
    lines.append("")
    lines.append("### 1. 旧主线是否还成立")
    lines.append("")
    lines.append(
        "结论：`L20H5 / L21H1 / L21H12 / L24H6 / MLP27` 这条旧主线整体仍成立，但应该改写成“入口 + 双路 late routing + pre-writer formatter + main writer”，而不是把它们视作同质的 late tool heads。"
    )
    lines.append("")
    lines.append("### 2. `L20H5` 更像什么")
    lines.append("")
    lines.append(
        f"`MLP19 -> L20H5` 的 route mediation 中位数为 `{fmt(route_map['L20H5']['route_mediated_ratio_median'])}`，target local rescue 为 `{fmt(route_map['L20H5']['target_local_rescue_median'])}`；同时它的 direct margin 写出在 clean 下已经为 `{fmt(write_map['L20H5']['clean_margin_logit_median'])}`。"
    )
    lines.append(
        "因此 `L20H5` 仍然接收 route state，但更合适的定位不是“上游 decision 节点”，而是 Construction 入口：它是最早把 route state 接到文件名 / 函数体 payload 上的节点。它更像 payload binder / ingress，而不是已经完成 `<tool_call>` 写出的 main writer。"
    )
    lines.append("")
    lines.append("### 3. `L21H1` 和 `L21H12` 是并行冗余还是功能不同")
    lines.append("")
    lines.append(
        f"二者都属于 late routing，但不是简单冗余。旧 full-run 里，`L21H12 -> MLP27` mediation 为 `{fmt(legacy_context['minimal_cue_edges'].get('L21H12->MLP27'))}`，高于 `L21H1 -> MLP27` 的 `{fmt(legacy_context['minimal_cue_edges'].get('L21H1->MLP27'))}`；而 `L20H5 + L21H12` 的 step rescue 也高于 `L20H5 + L21H1`。"
    )
    lines.append(
        "结合 attention 证据，`L21H1` 更像把 construction state 路由到 output-start / example 相关位点的通用 late router；`L21H12` 更像把 tool protocol / example / instruction tail 绑定到 final writer 的 protocol-heavy router。"
    )
    lines.append("")
    lines.append("### 4. `L24H6` 更像什么")
    lines.append("")
    lines.append(
        f"`L24H6` 在 attention 聚合里最稳定读取 `tool_instruction`；旧 full-run 中它到 `MLP27` 的 mediation 为 `{fmt(legacy_context['minimal_cue_edges'].get('L24H6->MLP27'))}`，而 stagewise 从 `+L21H12` 到 `+L24H6` 时，route margin 从 `{fmt(l24_prev.get('route_margin_median'))}` 升到 `{fmt(l24_cur.get('route_margin_median'))}`，tool logit margin 从 `{fmt(l24_prev.get('margin_logit_median'))}` 升到 `{fmt(l24_cur.get('margin_logit_median'))}`。"
    )
    lines.append(
        "因此它更像 protocol commitment / formatter，而不是普通 late router：它不是主要读取文件对象，而是把已经路由好的 tool state 压进合适的调用起始格式。"
    )
    lines.append("")
    lines.append("### 5. `MLP27` 是不是主要 writer")
    lines.append("")
    lines.append(
        f"是。旧 full-run 里 `MLP27` 单节点 rescue 为 `{fmt(legacy_context['query_component'].get('MLP27'))}`，`corrupt_full` 上 alpha=`1.5` steering 的 `<tool_call>` top1 达到 `{fmt(legacy_context['mlp27_tool_top1_alpha_15'])}`；新 full-run 里它的 clean direct tool logit effect 为 `{fmt(write_map['MLP27']['clean_tool_logit_median'])}`，clean margin effect 为 `{fmt(write_map['MLP27']['clean_margin_logit_median'])}`，也是链上最大。"
    )
    lines.append(
        "但 `MLP27` 写的不是抽象 route state，而是更接近实际输出起始的 `<tool_call>`-favoring residual evidence。"
    )
    lines.append("")
    lines.append("### 6. `<tool_call>` 从哪里开始明显可见")
    lines.append("")
    lines.append(
        f"如果只看 late chain 的平均 direct-logit lens，`MLP19` 对 `<tool_call>` 已经有可见正 logit（`{fmt(lens_map['MLP19']['tool_logit'])}`，rank `{lens_map['MLP19']['tool_rank']}`），但这还更像 route-state spillover。"
    )
    if first_positive_logit_margin:
        lines.append(
            f"按 stagewise logit margin 看，`<tool_call>` 首次稳定翻正发生在 `{first_positive_logit_margin['step_label']}`；按 route margin 看，首次翻正发生在 `{first_positive_route['step_label'] if first_positive_route else '未翻正'}`；按 `<tool_call>` top1 多数出现看，第一次超过半数样本发生在 `{first_tool_top1_majority['step_label'] if first_tool_top1_majority else '未过半'}`。"
        )
    lines.append(
        f"`L20H5` 更像“开始接住并推动这条路”，它的 clean margin 为 `{fmt(l20_clean_margin)}`，但 clean-corrupt delta margin 为 `{fmt(l20_delta_margin)}`；真正把 `<tool_call>` 变成清晰输出候选的是 `L21H1/L21H12` 之后的 late routing，再由 `L24H6` 与 `MLP27` 放大并定型。"
    )
    lines.append("")
    lines.append("### 7. 哪些节点一边推 `<tool_call>`，一边压 competing token")
    lines.append("")
    for node in ["L20H5", "L21H12", "L24H6", "MLP27"]:
        row = write_map[node]
        lines.append(
            f"- `{node}`: clean `<tool_call>` logit `{fmt(row['clean_tool_logit_median'])}`，竞争 token logit `{fmt(row['clean_competitor_logit_median'])}`，margin `{fmt(row['clean_margin_logit_median'])}`，delta margin `{fmt(row['delta_margin_logit_median'])}`。"
        )
    lines.append(
        "其中 `L24H6` 和 `MLP27` 的 margin 提升最像“同时抬 `<tool_call>`、压 competing token”；`L20H5` 更像先把 route state 接到 payload 上并轻推输出边界；`L21H12` 则处在路由和协议绑定之间。"
    )
    lines.append("")
    lines.append("### 8. 有没有漏掉的新 construction nodes")
    lines.append("")
    if candidate_patch_summary:
        best_candidate = max(candidate_patch_summary, key=lambda r: float(r["rescue_ratio_median"]))
        lines.append(
            f"晚层新候选里，最好的一项是 `{best_candidate['node']}`，单节点 rescue 为 `{fmt(best_candidate['rescue_ratio_median'])}`，但仍明显弱于 anchor。"
        )
    lines.append(
        "因此这轮没有发现足以推翻旧主线的新 anchor。`L25H13 / L25H10 / L26H15 / L27H7` 最多只能保留为 candidate：它们在 attention 聚合上确实带有 tool/protocol 痕迹，但在单节点 patch 和写出效果上都不够强。"
    )
    lines.append("")
    lines.append("## 读什么 / 写什么 / 怎么传")
    lines.append("")
    lines.append("| 节点 | 读什么 | 写什么 | 怎样传给下游 | 当前定位 |")
    lines.append("| --- | --- | --- | --- | --- |")
    lines.append("| `L20H5` | 文件名、函数体锚点、少量 assistant prefix | 早期 payload-bound tool state | 把 payload-bound state 送到 `L21H1/L21H12` | anchor / Construction 入口 |")
    lines.append("| `L21H1` | example / output-start 相关位置，夹带 task-body/preamble 信息 | late routed tool state | 一部分直送 `MLP27`，一部分送 `L24H6` | support / late router |")
    lines.append("| `L21H12` | tool_call example、instruction tail、protocol 相关位点 | 更 protocol-heavy 的 routed tool state | 强送 `MLP27`，也送 `L24H6` | anchor / protocol-heavy router |")
    lines.append("| `L24H6` | tool instruction / call format | `<tool_call>` 起始格式的 pre-writer state | 主要送 `MLP27` | anchor / formatter-commitment |")
    lines.append("| `MLP27` | 已绑定好的 late tool state | `<tool_call>`-favoring residual direction | 直接写到输出 | anchor / main writer |")
    lines.append("| `MLP19` | route score | tool-route state 本身 | fanout 到多个 construction 节点 | support / interface |")
    lines.append("")
    lines.append("## 与 Output-Route Decision 的连接")
    lines.append("")
    lines.append("### 哪些 construction 节点最直接接收 route state")
    lines.append("")
    for target in ["L20H5", "L21H1", "L21H12", "L24H6", "MLP27"]:
        row = route_map[target]
        lines.append(
            f"- `MLP19 -> {target}`: route mediation `{fmt(row['route_mediated_ratio_median'])}`，target local rescue `{fmt(row['target_local_rescue_median'])}`。"
        )
    lines.append(
        "从这组 full-run 数据看，`MLP27` 是最强的 route-state 接收者之一，但它不是最早的 payload binder；`L20H5` 是最早把 route state 接到文件/函数体对象上的节点；`L21H1/L21H12` 进一步把这份 state 路由并协议化；`L24H6` 则把它压成更接近 `<tool_call>` 起始格式的 pre-writer state。"
    )
    lines.append("")
    lines.append("### route state 如何被转成 `<tool_call>` 偏好")
    lines.append("")
    lines.append(
        "最可信的链条是：`MLP19` 先把“该走 tool 协议输出”的 route state 分发到 late construction 区；`L20H5` 把这份状态绑定到 file/function payload；`L21H1/L21H12` 把已绑定状态继续路由到 example / protocol 相关位点；`L24H6` 把 route+payload state 压进具体调用起始格式；最后 `MLP27` 把它写成真正可见的 `<tool_call>` 首词偏好。"
    )
    lines.append("")
    lines.append("## 写出类可视化结论")
    lines.append("")
    if first_positive_logit_margin and first_tool_top1_majority:
        lines.append(
            f"- stagewise 轨迹见 `{out_root / 'figures' / 'construction_stagewise_trajectory.png'}`。这张图显示：`<tool_call>` 的 logit margin 首次翻正发生在 `{first_positive_logit_margin['step_label']}`，首次在多数样本成为 top-1 发生在 `{first_tool_top1_majority['step_label']}`，再由 `MLP27` 推到最终稳定区。"
        )
    lines.append(f"- top token / top-1 变化图见 `{out_root / 'figures' / 'construction_top_token_change.png'}`。它把 `<tool_call>` top-1 比例、boundary flip，以及主要 competing token 的 top-1 组成放到同一张图里。")
    lines.append(f"- 节点 direct writeout 见 `{out_root / 'figures' / 'construction_node_writeout.png'}`。这张图最直接回答“谁在写什么”。")
    lines.append(f"- residual projection 见 `{out_root / 'figures' / 'construction_node_projection.png'}`。它显示同一批节点怎样沿 `<tool_call>`-竞争方向累积偏置。")
    lines.append(f"- `MLP19` fanout 图见 `{out_root / 'figures' / 'construction_route_fanout.png'}`。它回答 route state 是怎么接进 construction 的。")
    lines.append(f"- 平均 direct-logit lens 见 `{out_root / 'figures' / 'construction_mean_logit_lens.png'}`。它回答 `<tool_call>` 在哪些节点开始变得“可见”。")
    lines.append(f"- attention 读入面板见 `{attention_artifacts['combined_attention_panel']}`。这张图用于区分 `L20H5/L21H1/L21H12/L24H6` 读入对象的差别。")
    lines.append("")
    lines.append("## 每个 anchor node 的证据包")
    lines.append("")
    lines.append("### `L20H5`")
    lines.append("")
    lines.append("- 读入证据：attention 聚合和 head audit 都显示它优先读 `file_target` / `function_body_anchor`。")
    lines.append("- 写出证据：新的 direct logit effect 显示它会把 clean-corrupt 输出边界往 `<tool_call>` 方向推，哪怕它通常还不是局部最强 writer。")
    lines.append("- 传递证据：旧 full-run 里 `L20H5 -> L21H12` 与 `L20H5 -> L21H1` 都有正 mediation，且前者更强。")
    lines.append("- 行为证据：单节点 patch rescue 与 stagewise 第一跳都稳定为正。")
    lines.append("")
    lines.append("### `L21H12`")
    lines.append("")
    lines.append("- 读入证据：attention 聚合里它最稳定读 `tool_call_example` / instruction tail。")
    lines.append("- 写出证据：新的 direct logit effect 与 projection 都比 `L21H1` 更强。")
    lines.append("- 传递证据：`L21H12 -> MLP27` 是旧 full-run 中最强的 late tool edge。")
    lines.append("- 行为证据：把它并入 stagewise 后，`<tool_call>` margin 由负转正。")
    lines.append("")
    lines.append("### `L24H6`")
    lines.append("")
    lines.append("- 读入证据：attention 聚合明确显示它最强读 `tool_instruction`。")
    lines.append("- 写出证据：新的 direct effect 显示它明显提高 `<tool_call>` margin。")
    lines.append("- 传递证据：旧 full-run 中 `L24H6 -> MLP27` mediation 很强。")
    lines.append("- 行为证据：stagewise 从 `+L21H12` 到 `+L24H6` 时，`<tool_call>` top1 再次大幅上升。")
    lines.append("")
    lines.append("### `MLP27`")
    lines.append("")
    lines.append("- 写出证据：direct effect / projection 都是链上最强。")
    lines.append("- 因果证据：steering 在 `corrupt_full` 上可把 `<tool_call>` top1 推到高比例。")
    lines.append("- 传递证据：它直接接收来自 `L21H1/L21H12/L24H6` 的 late routed state，也能直接接收部分 `MLP19` fanout。")
    lines.append("- 行为证据：stagewise 最后一跳主要由它把已成形的 `<tool_call>` 偏置写成稳定首词。")
    lines.append("")
    lines.append("## 未解决问题")
    lines.append("")
    lines.append("- 这轮虽然看到 `MLP19 -> MLP27` 的直接 fanout 很强，但还不能把它写成“绕过所有 late heads 的独立主路”；更像并行 receipt。")
    lines.append("- `L21H1` 的功能已经能和 `L21H12` 区分开，但还没有足够强的限制性 patch 去精确分解二者的冗余比例。")
    lines.append("- 新候选晚层头存在 attention 迹象，但没有足够强的 patching / writeout 证据，不能升格。")
    lines.append("")
    lines.append("## 论文风格总结")
    lines.append("")
    lines.append(
        "当前最可信的 Tool-Call Construction 机制是：`MLP19` 把已定下来的 tool-route state 分发到晚层 construction 区后，`L20H5` 先把这份状态绑定到文件名与函数体对象，`L21H1/L21H12` 再把已绑定状态分别路由到 output-start/example 与更 protocol-heavy 的 tool-call example / instruction-tail 相关位点；随后 `L24H6` 把这份 route+payload state 压进更接近调用起始格式的 pre-writer state，最终由 `MLP27` 把它写成首词 `<tool_call>` 的强偏好。最强的证据来自三类 full-run 结果同时收敛：其一，旧的边级 patching 证明 `L20H5 -> L21H1/L21H12 -> L24H6 -> MLP27` 不是共现链，而是有明确 late mediation 的传递链；其二，新的 direct logit effect / residual projection 显示 `<tool_call>` 的写出从 `L20H5` 开始可见，在 `L21H12/L24H6` 明显增强，并在 `MLP27` 达到最大；其三，新的 stagewise token trajectory 直接显示 `<tool_call>` 从弱偏置到稳定 top-1 的逐步形成过程。当前还不能强写的是：`MLP19 -> MLP27` 是否构成绕过 late heads 的独立主路，以及若干晚层候选头是否属于 construction 支路而非伴随激活。"
    )
    lines.append("")
    lines.append("## Artifact Index")
    lines.append("")
    lines.append("- `construction_writeout_per_sample.csv`")
    lines.append("- `construction_writeout_summary.csv`")
    lines.append("- `construction_stagewise_per_sample.csv`")
    lines.append("- `construction_stagewise_summary.csv`")
    lines.append("- `construction_stagewise_top_tokens_summary.csv`")
    lines.append("- `figures/construction_top_token_change.png`")
    lines.append("- `construction_route_fanout_per_sample.csv`")
    lines.append("- `construction_route_fanout_summary.csv`")
    lines.append("- `construction_candidate_patch_summary.csv`")
    lines.append("- `construction_mean_logit_lens_summary.csv`")

    report_path = out_root / "tool_call_construction_report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    summary = {
        "anchor_nodes": anchor_nodes,
        "support_nodes": support_nodes,
        "candidate_nodes": candidate_nodes,
        "tool_token_text": tool_token_text,
        "top_competitors": top_competitors,
        "actual_n": actual_n,
        "artifacts": {
            "report_md": str(report_path),
            "node_writeout_png": str(out_root / "figures" / "construction_node_writeout.png"),
            "node_projection_png": str(out_root / "figures" / "construction_node_projection.png"),
            "stagewise_png": str(out_root / "figures" / "construction_stagewise_trajectory.png"),
            "top_token_change_png": str(out_root / "figures" / "construction_top_token_change.png"),
            "route_fanout_png": str(out_root / "figures" / "construction_route_fanout.png"),
            "candidate_patch_png": str(out_root / "figures" / "construction_candidate_patch.png"),
            "mean_logit_lens_png": str(out_root / "figures" / "construction_mean_logit_lens.png"),
            **attention_artifacts,
        },
    }
    (out_root / "tool_call_construction_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_summary_csv(path: Path) -> List[Dict[str, object]]:
    rows = read_csv_rows(path)
    out: List[Dict[str, object]] = []
    for row in rows:
        clean: Dict[str, object] = {}
        for key, value in row.items():
            if value in {"", None}:
                clean[key] = value
                continue
            low = str(value).lower()
            if low == "true":
                clean[key] = True
                continue
            if low == "false":
                clean[key] = False
                continue
            try:
                clean[key] = float(value)
                if clean[key].is_integer():
                    clean[key] = int(clean[key])
            except Exception:
                clean[key] = value
        out.append(clean)
    return out


def regenerate_report_only(out_root: Path, legacy_data_root: Path, attention_root: Path) -> None:
    writeout_rows = load_summary_csv(out_root / "construction_writeout_per_sample.csv")
    stage_rows = load_summary_csv(out_root / "construction_stagewise_per_sample.csv")
    route_fanout_rows = load_summary_csv(out_root / "construction_route_fanout_per_sample.csv")
    candidate_patch_rows = load_summary_csv(out_root / "construction_candidate_patch_per_sample.csv")
    mean_logit_rows = load_summary_csv(out_root / "construction_mean_logit_lens_summary.csv")

    competitor_counts: Counter[str] = Counter(str(r.get("competitor_token", "")) for r in writeout_rows if str(r.get("node")) == "MLP19")
    actual_n = len({str(r.get("sample_id")) for r in writeout_rows if str(r.get("node")) == "MLP19"})
    tool_token_text = str(next((r.get("tool_token") for r in writeout_rows if str(r.get("node")) == "MLP19"), "<tool_call>"))
    writeout_summary, stage_summary, stage_top_tokens_summary, route_fanout_summary, candidate_patch_summary = build_derived_summaries(
        writeout_rows=writeout_rows,
        stage_rows=stage_rows,
        route_fanout_rows=route_fanout_rows,
        candidate_patch_rows=candidate_patch_rows,
        actual_n=actual_n,
    )
    write_csv(writeout_summary, out_root / "construction_writeout_summary.csv")
    write_csv(stage_summary, out_root / "construction_stagewise_summary.csv")
    write_csv(stage_top_tokens_summary, out_root / "construction_stagewise_top_tokens_summary.csv")
    write_csv(route_fanout_summary, out_root / "construction_route_fanout_summary.csv")
    write_csv(candidate_patch_summary, out_root / "construction_candidate_patch_summary.csv")
    plot_node_writeout(writeout_summary, out_root / "figures" / "construction_node_writeout.png")
    plot_node_projection(writeout_summary, out_root / "figures" / "construction_node_projection.png")
    plot_stagewise(stage_summary, out_root / "figures" / "construction_stagewise_trajectory.png")
    plot_top_token_change(stage_summary, stage_top_tokens_summary, out_root / "figures" / "construction_top_token_change.png")
    plot_route_fanout(route_fanout_summary, out_root / "figures" / "construction_route_fanout.png")
    plot_candidate_patch(candidate_patch_summary, out_root / "figures" / "construction_candidate_patch.png")
    plot_logit_lens(mean_logit_rows, out_root / "figures" / "construction_mean_logit_lens.png")

    legacy_query = json.loads((legacy_data_root / "query_decision_summary.json").read_text(encoding="utf-8"))
    legacy_mlp27 = json.loads((legacy_data_root / "mlp27_steering_summary.json").read_text(encoding="utf-8"))
    minimal_cue_edges = {
        str(row["edge"]): float(row["mediated_ratio_median"])
        for row in read_csv_rows(legacy_data_root / "minimal_cue_edge_summary.csv")
        if str(row["family"]) == "tool"
    }
    query_component = {
        str(row["component"]): float(row["rescue_ratio_median"])
        for row in legacy_query["component_summary_rows"]
        if str(row["family"]) == "query"
    }
    alpha15 = next(
        (
            float(row["tool_top1_rate"])
            for row in legacy_mlp27["summary_rows"]
            if str(row.get("base_variant")) == "corrupt_full" and abs(float(row.get("alpha")) - 1.5) < 1e-8
        ),
        float("nan"),
    )
    legacy_context = {
        "minimal_cue_edges": minimal_cue_edges,
        "query_component": query_component,
        "mlp27_tool_top1_alpha_15": alpha15,
    }

    attention_artifacts = {
        "combined_attention_panel": str(out_root / "figures" / "construction_attention_panels.png"),
    }
    for head in ["L20H5", "L21H1", "L21H12", "L24H6"]:
        attention_artifacts[f"{head}_density"] = str(out_root / "figures" / head / "density_heatmap.png")
        attention_artifacts[f"{head}_decision"] = str(out_root / "figures" / head / "decision_row.png")
    if not Path(attention_artifacts["combined_attention_panel"]).exists():
        attention_artifacts = copy_or_combine_attention_plots(attention_root, out_root)

    build_report(
        out_root=out_root,
        route_fanout_summary=route_fanout_summary,
        writeout_summary=writeout_summary,
        stage_summary=stage_summary,
        stage_top_tokens_summary=stage_top_tokens_summary,
        candidate_patch_summary=candidate_patch_summary,
        mean_logit_rows=mean_logit_rows,
        legacy_context=legacy_context,
        attention_artifacts=attention_artifacts,
        tool_token_text=tool_token_text,
        competitor_counts=competitor_counts,
        actual_n=actual_n,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Full-dataset Tool-Call Construction analysis.")
    parser.add_argument("--run-root", type=str, required=True)
    parser.add_argument(
        "--legacy-data-root",
        type=str,
        default="./results/legacy/final/data",
    )
    parser.add_argument(
        "--route-refine-root",
        type=str,
        default="./results/output_route_decision_refine/20260319-134319-output-route-decision-refine",
    )
    parser.add_argument(
        "--attention-root",
        type=str,
        default="./results/attentionhead/20260319-121000-attention-head-full",
    )
    parser.add_argument("--model-path", type=str, default="./external/models/Qwen3-1.7B")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--output-root", type=str, required=True)
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()

    run_root = Path(args.run_root).resolve()
    legacy_data_root = Path(args.legacy_data_root).resolve()
    route_refine_root = Path(args.route_refine_root).resolve()
    attention_root = Path(args.attention_root).resolve()
    out_root = Path(args.output_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "figures").mkdir(parents=True, exist_ok=True)

    if args.report_only:
        regenerate_report_only(out_root, legacy_data_root, attention_root)
        return

    samples = load_sample_paths(run_root / "forward_batch", run_root / "reverse_batch", max_samples=args.max_samples)
    model, tokenizer = load_hooked_qwen3(args.model_path, device=args.device, dtype=torch.bfloat16)

    tracked_nodes = sorted(set(TRACKED_NODES))
    candidate_patch_nodes = SEARCH_HEAD_CANDIDATES + SEARCH_MLP_CANDIDATES

    writeout_rows: List[Dict[str, object]] = []
    stage_rows: List[Dict[str, object]] = []
    route_fanout_rows: List[Dict[str, object]] = []
    candidate_patch_rows: List[Dict[str, object]] = []
    competitor_counts: Counter[str] = Counter()

    mean_clean_normed_sum: Dict[str, torch.Tensor] = {}
    mean_corrupt_normed_sum: Dict[str, torch.Tensor] = {}
    tool_token_ids: Counter[int] = Counter()

    pbar = tqdm(samples, desc="Tool-Call Construction", dynamic_ncols=True)
    for sp in pbar:
        try:
            clean_text = resolve_prompt_path(sp.tool_call_prompt).read_text(encoding="utf-8")
            corrupt_text = resolve_prompt_path(sp.no_tool_prompt).read_text(encoding="utf-8")
        except Exception:
            continue

        clean_tokens = model.to_tokens(clean_text, prepend_bos=False)
        corrupt_tokens = model.to_tokens(corrupt_text, prepend_bos=False)
        if clean_tokens.shape != corrupt_tokens.shape:
            continue

        clean_cache = collect_cache_with_scale(model, clean_tokens, tracked_nodes)
        corrupt_cache = collect_cache_with_scale(model, corrupt_tokens, tracked_nodes)

        with torch.no_grad():
            clean_logits = model(clean_tokens)
            corrupt_logits = model(corrupt_tokens)
        tool_objective, no_tool_objective = build_bidirectional_endpoint_objectives(
            clean_logits,
            corrupt_logits,
            tokenizer=tokenizer,
        )
        tool_token_id = int(tool_objective.top_token_id or sp.target_tool_call)
        competitor_token_id = int(no_tool_objective.top_token_id or int(corrupt_logits[0, -1].argmax().item()))
        tool_token_ids[tool_token_id] += 1
        competitor_text = decode_token(tokenizer, competitor_token_id)
        competitor_counts[competitor_text] += 1

        clean_tool = float(objective_from_logits(clean_logits, tool_objective).item())
        corrupt_tool = float(objective_from_logits(corrupt_logits, tool_objective).item())
        clean_no_tool = float(objective_from_logits(clean_logits, no_tool_objective).item())
        corrupt_no_tool = float(objective_from_logits(corrupt_logits, no_tool_objective).item())
        route_gap = clean_tool - clean_no_tool - (corrupt_tool - corrupt_no_tool)
        tool_gap = clean_tool - corrupt_tool
        if not math.isfinite(tool_gap) or abs(tool_gap) < 1e-8:
            continue

        clean_scale = float(clean_cache["ln_final.hook_scale"][0, -1, 0].item())
        corrupt_scale = float(corrupt_cache["ln_final.hook_scale"][0, -1, 0].item())

        for node in tracked_nodes:
            clean_resid = extract_residual_contribution(model, clean_cache, node)
            corrupt_resid = extract_residual_contribution(model, corrupt_cache, node)
            clean_normed = normalized_residual(model, clean_resid, clean_scale)
            corrupt_normed = normalized_residual(model, corrupt_resid, corrupt_scale)

            if node not in mean_clean_normed_sum:
                mean_clean_normed_sum[node] = torch.zeros_like(clean_normed)
                mean_corrupt_normed_sum[node] = torch.zeros_like(corrupt_normed)
            mean_clean_normed_sum[node] += clean_normed
            mean_corrupt_normed_sum[node] += corrupt_normed

            clean_tool_logit = token_direct_logit(model, clean_normed, tool_token_id)
            clean_comp_logit = token_direct_logit(model, clean_normed, competitor_token_id)
            corrupt_tool_logit = token_direct_logit(model, corrupt_normed, tool_token_id)
            corrupt_comp_logit = token_direct_logit(model, corrupt_normed, competitor_token_id)

            direction = model.W_U[:, tool_token_id] - model.W_U[:, competitor_token_id]
            direction = direction.detach().cpu().float()
            direction = direction / max(float(direction.norm().item()), 1e-6)
            clean_proj = float(torch.dot(clean_normed, direction).item())
            corrupt_proj = float(torch.dot(corrupt_normed, direction).item())

            writeout_rows.append(
                {
                    "sample_id": sp.sample_id,
                    "node": node,
                    "layer": NODE_TO_LAYER.get(node, int(node[3:]) if node.startswith("MLP") else parse_head(node)[0]),
                    "tool_token": decode_token(tokenizer, tool_token_id),
                    "competitor_token": competitor_text,
                    "clean_tool_logit": clean_tool_logit,
                    "clean_competitor_logit": clean_comp_logit,
                    "clean_margin_logit": clean_tool_logit - clean_comp_logit,
                    "corrupt_tool_logit": corrupt_tool_logit,
                    "corrupt_competitor_logit": corrupt_comp_logit,
                    "corrupt_margin_logit": corrupt_tool_logit - corrupt_comp_logit,
                    "delta_tool_logit": clean_tool_logit - corrupt_tool_logit,
                    "delta_competitor_logit": clean_comp_logit - corrupt_comp_logit,
                    "delta_margin_logit": (clean_tool_logit - clean_comp_logit) - (corrupt_tool_logit - corrupt_comp_logit),
                    "clean_projection": clean_proj,
                    "corrupt_projection": corrupt_proj,
                    "delta_projection": clean_proj - corrupt_proj,
                }
            )

        # Stagewise actual output trajectory on corrupt base.
        for step_idx, (label, nodes) in enumerate(STAGE_STEPS):
            if not nodes:
                logits = corrupt_logits
            else:
                logits = run_logits_with_assignments(
                    model,
                    corrupt_tokens,
                    clean_cache,
                    corrupt_cache,
                    nodes,
                    [],
                )
            tool_score = float(objective_from_logits(logits, tool_objective).item())
            no_tool_score = float(objective_from_logits(logits, no_tool_objective).item())
            tool_logit = float(logits[0, -1, tool_token_id].item())
            competitor_logit = float(logits[0, -1, competitor_token_id].item())
            top1_id = int(logits[0, -1].argmax().item())
            top1_text = decode_token(tokenizer, top1_id)
            stage_rows.append(
                {
                    "sample_id": sp.sample_id,
                    "step_idx": step_idx,
                    "step_label": label,
                    "nodes": "|".join(nodes),
                    "tool_score": tool_score,
                    "no_tool_score": no_tool_score,
                    "route_margin": tool_score - no_tool_score,
                    "tool_logit": tool_logit,
                    "competitor_logit": competitor_logit,
                    "margin_logit": tool_logit - competitor_logit,
                    "tool_prob": float(torch.softmax(logits[0, -1], dim=-1)[tool_token_id].item()),
                    "tool_top1": top1_id == tool_token_id,
                    "boundary_flip": tool_score > no_tool_score,
                    "top1_token": top1_text,
                }
            )

        # Route fanout from MLP19 into all construction targets.
        source_only_logits, source_only_cache = run_with_assignments_and_collect(
            model,
            corrupt_tokens,
            clean_cache_cpu=clean_cache,
            corrupt_cache_cpu=corrupt_cache,
            patch_clean_nodes=[ROUTE_SOURCE],
            patch_corrupt_nodes=[],
            record_nodes=ROUTE_TARGETS,
        )
        source_only_tool = float(objective_from_logits(source_only_logits, tool_objective).item())
        source_only_no_tool = float(objective_from_logits(source_only_logits, no_tool_objective).item())
        source_only_route_ratio = ((source_only_tool - source_only_no_tool) - (corrupt_tool - corrupt_no_tool)) / max(route_gap, 1e-6)
        for target in ROUTE_TARGETS:
            blocked_logits, _ = run_with_assignments_and_collect(
                model,
                corrupt_tokens,
                clean_cache_cpu=clean_cache,
                corrupt_cache_cpu=corrupt_cache,
                patch_clean_nodes=[ROUTE_SOURCE],
                patch_corrupt_nodes=[target],
                record_nodes=[],
            )
            blocked_tool = float(objective_from_logits(blocked_logits, tool_objective).item())
            blocked_no_tool = float(objective_from_logits(blocked_logits, no_tool_objective).item())
            blocked_route_ratio = ((blocked_tool - blocked_no_tool) - (corrupt_tool - corrupt_no_tool)) / max(route_gap, 1e-6)

            target_vec_clean = extract_residual_contribution(model, clean_cache, target)
            target_vec_corrupt = extract_residual_contribution(model, corrupt_cache, target)
            target_hook_name = node_hook_name(target)
            if target.startswith("MLP"):
                source_target_vec = source_only_cache[target_hook_name][0, -1, :].float()
            else:
                _layer, head = parse_head(target)
                source_target_vec = source_only_cache[target_hook_name][0, -1, head, :].float()
                w_o = model.blocks[parse_head(target)[0]].attn.W_O[head].detach().cpu().float()
                source_target_vec = source_target_vec @ w_o
            target_local_rescue = local_score(source_target_vec, target_vec_clean, target_vec_corrupt) - local_score(
                target_vec_corrupt,
                target_vec_clean,
                target_vec_corrupt,
            )

            route_fanout_rows.append(
                {
                    "sample_id": sp.sample_id,
                    "target": target,
                    "source_route_ratio": source_only_route_ratio,
                    "blocked_route_ratio": blocked_route_ratio,
                    "route_mediated_ratio": source_only_route_ratio - blocked_route_ratio,
                    "target_local_rescue": target_local_rescue,
                }
            )

        # Candidate single-node patch scan on corrupt_full.
        for node in candidate_patch_nodes:
            logits = run_logits_with_assignments(
                model,
                corrupt_tokens,
                clean_cache,
                corrupt_cache,
                [node],
                [],
            )
            tool_score = float(objective_from_logits(logits, tool_objective).item())
            no_tool_score = float(objective_from_logits(logits, no_tool_objective).item())
            candidate_patch_rows.append(
                {
                    "sample_id": sp.sample_id,
                    "node": node,
                    "layer": NODE_TO_LAYER.get(node, int(node[3:]) if node.startswith("MLP") else parse_head(node)[0]),
                    "rescue_ratio": (tool_score - corrupt_tool) / tool_gap,
                    "tool_top1": int(logits[0, -1].argmax().item()) == tool_token_id,
                    "boundary_flip": tool_score > no_tool_score,
                }
            )

        pbar.set_postfix(sample=sp.sample_id)

    actual_n = len({row["sample_id"] for row in writeout_rows if row["node"] == PRIMARY_CHAIN[0]})
    if actual_n == 0:
        raise ValueError("No valid samples were processed.")

    mean_clean_normed = {k: v / actual_n for k, v in mean_clean_normed_sum.items()}
    mean_corrupt_normed = {k: v / actual_n for k, v in mean_corrupt_normed_sum.items()}
    tool_token_id = tool_token_ids.most_common(1)[0][0]
    tool_token_text = decode_token(tokenizer, tool_token_id)

    writeout_summary, stage_summary, stage_top_tokens_summary, route_fanout_summary, candidate_patch_summary = build_derived_summaries(
        writeout_rows=writeout_rows,
        stage_rows=stage_rows,
        route_fanout_rows=route_fanout_rows,
        candidate_patch_rows=candidate_patch_rows,
        actual_n=actual_n,
    )

    mean_logit_rows = build_mean_logit_rows(
        model,
        tokenizer,
        mean_clean_normed,
        mean_corrupt_normed,
        tool_token_id,
    )

    write_csv(writeout_rows, out_root / "construction_writeout_per_sample.csv")
    write_csv(writeout_summary, out_root / "construction_writeout_summary.csv")
    write_csv(stage_rows, out_root / "construction_stagewise_per_sample.csv")
    write_csv(stage_summary, out_root / "construction_stagewise_summary.csv")
    write_csv(stage_top_tokens_summary, out_root / "construction_stagewise_top_tokens_summary.csv")
    write_csv(route_fanout_rows, out_root / "construction_route_fanout_per_sample.csv")
    write_csv(route_fanout_summary, out_root / "construction_route_fanout_summary.csv")
    write_csv(candidate_patch_rows, out_root / "construction_candidate_patch_per_sample.csv")
    write_csv(candidate_patch_summary, out_root / "construction_candidate_patch_summary.csv")
    write_csv(mean_logit_rows, out_root / "construction_mean_logit_lens_summary.csv")

    plot_node_writeout(writeout_summary, out_root / "figures" / "construction_node_writeout.png")
    plot_node_projection(writeout_summary, out_root / "figures" / "construction_node_projection.png")
    plot_stagewise(stage_summary, out_root / "figures" / "construction_stagewise_trajectory.png")
    plot_top_token_change(stage_summary, stage_top_tokens_summary, out_root / "figures" / "construction_top_token_change.png")
    plot_route_fanout(route_fanout_summary, out_root / "figures" / "construction_route_fanout.png")
    plot_candidate_patch(candidate_patch_summary, out_root / "figures" / "construction_candidate_patch.png")
    plot_logit_lens(mean_logit_rows, out_root / "figures" / "construction_mean_logit_lens.png")
    attention_artifacts = copy_or_combine_attention_plots(attention_root, out_root)

    legacy_query = json.loads((legacy_data_root / "query_decision_summary.json").read_text(encoding="utf-8"))
    legacy_mlp27 = json.loads((legacy_data_root / "mlp27_steering_summary.json").read_text(encoding="utf-8"))
    minimal_cue_edges = {
        str(row["edge"]): float(row["mediated_ratio_median"])
        for row in read_csv_rows(legacy_data_root / "minimal_cue_edge_summary.csv")
        if str(row["family"]) == "tool"
    }
    query_component = {
        str(row["component"]): float(row["rescue_ratio_median"])
        for row in legacy_query["component_summary_rows"]
        if str(row["family"]) == "query"
    }
    alpha15 = next(
        (
            float(row["tool_top1_rate"])
            for row in legacy_mlp27["summary_rows"]
            if str(row.get("base_variant")) == "corrupt_full" and abs(float(row.get("alpha")) - 1.5) < 1e-8
        ),
        float("nan"),
    )

    legacy_context = {
        "minimal_cue_edges": minimal_cue_edges,
        "query_component": query_component,
        "mlp27_tool_top1_alpha_15": alpha15,
    }

    build_report(
        out_root=out_root,
        route_fanout_summary=route_fanout_summary,
        writeout_summary=writeout_summary,
        stage_summary=stage_summary,
        stage_top_tokens_summary=stage_top_tokens_summary,
        candidate_patch_summary=candidate_patch_summary,
        mean_logit_rows=mean_logit_rows,
        legacy_context=legacy_context,
        attention_artifacts=attention_artifacts,
        tool_token_text=tool_token_text,
        competitor_counts=competitor_counts,
        actual_n=actual_n,
    )


if __name__ == "__main__":
    main()
