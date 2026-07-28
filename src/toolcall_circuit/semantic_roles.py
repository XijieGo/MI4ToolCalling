#!/usr/bin/env python3
"""
Semantic role analysis for the aggregated tool-call circuit.

This script quantifies for core heads/MLPs:
1) read-side behavior: where attention heads read from;
2) causal read-side evidence: patching only selected source positions;
3) write-side behavior: which logits are pushed by each node.

Outputs include CSV/JSON reports, heatmaps, and a semantic-annotated circuit.
"""

from __future__ import annotations

import os
import argparse
import csv
import gc
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm.auto import tqdm

if __package__ in {None, ""}:
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from toolcall_circuit.dataset import (
    build_position_sets as build_dynamic_position_sets,
    get_tool_call_target_spec,
    load_summary_records,
    resolve_distractor_token,
)
from toolcall_circuit.paths import MODEL_PATH_DEFAULT, RESULTS_ROOT
from toolcall_circuit.single_sample import draw_circuit, load_hooked_qwen3, objective_from_logits


def parse_head(node: str) -> Tuple[int, int]:
    # Node format: L{layer}H{head}
    body = node[1:]
    layer_s, head_s = body.split("H")
    return int(layer_s), int(head_s)


def parse_mlp(node: str) -> int:
    # Node format: MLP{layer}
    return int(node[3:])


def safe_ratio(num: float, den: float) -> float:
    if abs(den) < 1e-8:
        return float("nan")
    return num / den


def finite_vals(xs: Iterable[float]) -> List[float]:
    out: List[float] = []
    for x in xs:
        if isinstance(x, (int, float)) and math.isfinite(float(x)):
            out.append(float(x))
    return out


def med(xs: Iterable[float]) -> float:
    vals = finite_vals(xs)
    return float(median(vals)) if vals else float("nan")


def mean(xs: Iterable[float]) -> float:
    vals = finite_vals(xs)
    return float(np.mean(vals)) if vals else float("nan")


def topk_indices(values: torch.Tensor, k: int) -> List[int]:
    if values.numel() == 0:
        return []
    k = min(k, int(values.numel()))
    _, idx = torch.topk(values, k)
    return [int(i) for i in idx.tolist()]


def decode_pos_token(tokenizer, ids: Sequence[int], pos: int) -> str:
    if pos < 0 or pos >= len(ids):
        return ""
    t = tokenizer.decode([int(ids[pos])])
    return t.replace("\n", "\\n")


def tokenize_id(tokenizer, text: str) -> int:
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if not ids:
        raise ValueError(f"Text tokenized to empty sequence: {text}")
    return int(ids[0])


def collect_mlp_cache(model, tokens: torch.Tensor, mlp_layers: Sequence[int]) -> Dict[str, torch.Tensor]:
    names = {f"blocks.{l}.hook_mlp_out" for l in sorted(set(mlp_layers))}
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens, names_filter=lambda n: n in names)
    return {k: v.detach().cpu() for k, v in cache.items()}


def collect_head_layer_cache(model, tokens: torch.Tensor, layer: int) -> Dict[str, torch.Tensor]:
    names = {
        f"blocks.{layer}.attn.hook_pattern",
        f"blocks.{layer}.attn.hook_v",
        f"blocks.{layer}.attn.hook_z",
    }
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens, names_filter=lambda n: n in names)
    return {k: v.detach().cpu() for k, v in cache.items()}


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
    fig, ax = plt.subplots(figsize=(1.2 * max(6, len(col_labels)), 0.7 * max(4, len(row_labels)) + 2))
    finite = matrix[np.isfinite(matrix)]
    if finite.size == 0:
        vmax = 1.0
    else:
        vmax = float(np.percentile(np.abs(finite), percentile_clip))
        vmax = max(vmax, 1e-6)
    im = ax.imshow(matrix, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_title(title, pad=12)
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels)
    cbar = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
    cbar.set_label("Value (center=0)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_bar(values: Dict[str, float], title: str, ylabel: str, out_path: Path) -> None:
    apply_plot_style()
    keys = list(values.keys())
    vals = [values[k] for k in keys]
    fig, ax = plt.subplots(figsize=(max(8, 0.9 * len(keys)), 4.8))
    ax.bar(np.arange(len(keys)), vals, color="#2a6f97")
    ax.axhline(0.0, color="#444444", linewidth=1.0)
    ax.set_xticks(np.arange(len(keys)))
    ax.set_xticklabels(keys, rotation=35, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def role_from_metrics(head_metrics: Dict[str, float]) -> str:
    read_tool_call = head_metrics.get("tool_call_tags_ratio_median", float("nan"))
    read_tools = head_metrics.get("tools_block_ratio_median", float("nan"))
    read_contrast = head_metrics.get("contrast_ratio_median", float("nan"))
    read_user = head_metrics.get("user_block_ratio_median", float("nan"))
    attn_tool = head_metrics.get("attn_tool_call_tags_clean_mean", float("nan"))
    attn_contrast = head_metrics.get("attn_contrast_clean_mean", float("nan"))
    target_delta = head_metrics.get("target_logit_delta_median", float("nan"))

    top1_tokens = head_metrics.get("top1_token_counts", {}) if isinstance(head_metrics.get("top1_token_counts"), dict) else {}
    top1_token = ""
    top1_count = 0
    for t, c in top1_tokens.items():
        if int(c) > top1_count:
            top1_count = int(c)
            top1_token = str(t)

    read_role = "Format Router"
    if "<tool_call>" in top1_token or (
        math.isfinite(read_tool_call) and read_tool_call >= 0.15 and math.isfinite(attn_tool) and attn_tool >= 0.08
    ):
        read_role = "Tool-Tag Reader"
    elif math.isfinite(read_user) and read_user >= 0.15:
        read_role = "Query Reader"
    elif math.isfinite(read_tools) and read_tools >= 0.12:
        read_role = "Schema Reader"
    elif math.isfinite(read_contrast) and read_contrast >= 0.05:
        read_role = "Instruction Gate"
    elif math.isfinite(attn_contrast) and attn_contrast >= 0.01:
        read_role = "Instruction Gate"

    distract_delta = head_metrics.get("distractor_logit_delta_median", float("nan"))
    write_role = "Weak Writer"
    if math.isfinite(distract_delta) and distract_delta <= -1.0:
        write_role = "Distractor Suppressor"
    if math.isfinite(target_delta) and target_delta >= 0.8:
        write_role = "Target Booster"
    if (
        math.isfinite(target_delta)
        and target_delta >= 0.8
        and math.isfinite(distract_delta)
        and distract_delta <= -1.0
    ):
        write_role = "Target+Suppressor"

    return f"{read_role} / {write_role}"


def mlp_role_from_metrics(mlp_metrics: Dict[str, float]) -> str:
    ratio = mlp_metrics.get("full_ratio_median", float("nan"))
    target_delta = mlp_metrics.get("target_logit_delta_median", float("nan"))
    distract_delta = mlp_metrics.get("distractor_logit_delta_median", float("nan"))
    if math.isfinite(ratio) and ratio >= 0.30 and math.isfinite(target_delta) and target_delta >= 0.8:
        return "Primary Writer MLP"
    if math.isfinite(ratio) and ratio >= 0.18:
        if math.isfinite(distract_delta) and distract_delta <= -0.8:
            return "Support Suppressor MLP"
        return "Support Writer MLP"
    if math.isfinite(ratio) and ratio >= 0.05:
        return "Aux MLP"
    return "Weak MLP"


def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic role analysis for tool-call circuit.")
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
    parser.add_argument("--max-samples", type=int, default=0, help="0 means use all valid samples.")
    parser.add_argument("--sample-rank-start", type=int, default=0)
    parser.add_argument("--sample-rank-end", type=int, default=0)
    parser.add_argument("--q-start", type=int, default=1)
    parser.add_argument("--q-end", type=int, default=10_000)
    parser.add_argument(
        "--recompute-baseline",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Recompute clean/corrupt objectives with the current run configuration.",
    )
    args = parser.parse_args()

    input_root = Path(args.input_root).resolve()
    out_root = Path(args.output_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    agg = json.loads(Path(args.aggregate_summary).resolve().read_text(encoding="utf-8"))
    core_nodes = list(agg.get("core_nodes", []))
    core_edges = [tuple(e) for e in agg.get("core_edges", [])]
    if not core_nodes:
        raise ValueError("No core_nodes found in aggregate summary.")

    head_nodes = sorted([n for n in core_nodes if n.startswith("L")], key=lambda x: (parse_head(x)[0], parse_head(x)[1]))
    mlp_nodes = sorted([n for n in core_nodes if n.startswith("MLP")], key=parse_mlp)
    if not head_nodes and not mlp_nodes:
        raise ValueError("No head/MLP nodes in core_nodes.")

    head_layers = [parse_head(n)[0] for n in head_nodes]
    mlp_layers = [parse_mlp(n) for n in mlp_nodes]
    head_nodes_by_layer: Dict[int, List[str]] = defaultdict(list)
    for n in head_nodes:
        l, _ = parse_head(n)
        head_nodes_by_layer[l].append(n)

    model, tokenizer = load_hooked_qwen3(args.model_path, device=args.device, dtype=torch.bfloat16)
    # Reduce activation memory footprint for long batched analysis runs.
    model.set_use_attn_result(False)
    model.set_use_split_qkv_input(False)
    model.set_use_hook_mlp_in(False)
    target_spec = get_tool_call_target_spec(tokenizer, target_text="<tool_call>")
    if not target_spec.is_single_token:
        raise NotImplementedError(
            f"<tool_call> tokenization is no longer single-token ({target_spec.token_ids}); "
            "upgrade the semantic-role objective before running this workflow."
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
    node_metric_lists: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    node_top_tokens: Dict[str, Counter[str]] = defaultdict(Counter)
    head_top1_category_counter: Dict[str, Counter[str]] = defaultdict(Counter)
    head_top1_token_counter: Dict[str, Counter[str]] = defaultdict(Counter)
    skipped_oom: List[str] = []
    analyzed_sample_ids: List[str] = []

    pbar = tqdm(sample_infos, desc="Semantic roles", dynamic_ncols=True)
    for sample_id, sample_rank, legacy_index, summary, clean_prompt, corrupt_prompt in pbar:
        clean_text = clean_prompt.read_text(encoding="utf-8")
        corrupt_text = corrupt_prompt.read_text(encoding="utf-8")
        clean_tokens = model.to_tokens(clean_text, prepend_bos=False)
        corrupt_tokens = model.to_tokens(corrupt_text, prepend_bos=False)
        if clean_tokens.shape != corrupt_tokens.shape:
            continue

        ids_clean = [int(x) for x in clean_tokens[0].tolist()]
        ids_corrupt = [int(x) for x in corrupt_tokens[0].tolist()]
        pos_sets = build_dynamic_position_sets(ids_clean, ids_corrupt, tokenizer, clean_text=clean_text)

        clean_obj = float(summary.get("clean_obj", float("nan")))
        corrupt_obj = float(summary.get("corrupt_obj", float("nan")))
        gap = float(summary.get("gap", clean_obj - corrupt_obj))
        if not (math.isfinite(clean_obj) and math.isfinite(corrupt_obj) and math.isfinite(gap)):
            continue
        if abs(gap) < 1e-8:
            continue
        try:
            with torch.no_grad():
                clean_logits = model(clean_tokens)
                corrupt_logits = model(corrupt_tokens)
        except torch.OutOfMemoryError:
            skipped_oom.append(sample_id)
            model.reset_hooks()
            gc.collect()
            torch.cuda.empty_cache()
            continue
        distractor = int(summary.get("distractor_token_id", -1))
        if distractor < 0 or distractor >= corrupt_logits.shape[-1] or distractor == target_token:
            distractor = resolve_distractor_token(corrupt_logits[0, -1, :], target_token)

        if args.recompute_baseline:
            clean_obj = float(objective_from_logits(clean_logits, target_token, distractor).item())
            corrupt_obj = float(objective_from_logits(corrupt_logits, target_token, distractor).item())
            gap = clean_obj - corrupt_obj
            if not math.isfinite(gap) or abs(gap) < 1e-8:
                continue
            if gap <= args.gap_min:
                continue

        try:
            mlp_cache_clean = collect_mlp_cache(model, clean_tokens, mlp_layers=mlp_layers)
        except torch.OutOfMemoryError:
            skipped_oom.append(sample_id)
            model.reset_hooks()
            gc.collect()
            torch.cuda.empty_cache()
            continue

        corrupt_last_logits = corrupt_logits[0, -1, :].float()

        # Analyze heads layer by layer to avoid OOM on long sequences.
        head_failed = False
        for layer in sorted(head_nodes_by_layer.keys()):
            try:
                head_cache_clean = collect_head_layer_cache(model, clean_tokens, layer)
                head_cache_corrupt = collect_head_layer_cache(model, corrupt_tokens, layer)
            except torch.OutOfMemoryError:
                head_failed = True
                break

            pat_key = f"blocks.{layer}.attn.hook_pattern"
            v_key = f"blocks.{layer}.attn.hook_v"
            z_key = f"blocks.{layer}.attn.hook_z"
            v_clean_all_layer = head_cache_clean[v_key][0].float().to(corrupt_tokens.device)  # [pos, kv_heads, d_head]
            v_corrupt_all_layer = head_cache_corrupt[v_key][0].float().to(corrupt_tokens.device)
            q_heads = int(model.cfg.n_heads)
            kv_heads = int(v_clean_all_layer.shape[1])
            group = max(1, q_heads // kv_heads)

            for node in head_nodes_by_layer[layer]:
                _, head = parse_head(node)
                pat_clean = head_cache_clean[pat_key][0, head, -1, :].float().to(corrupt_tokens.device)
                pat_corrupt = head_cache_corrupt[pat_key][0, head, -1, :].float().to(corrupt_tokens.device)

                kv_index = head // group
                v_clean = v_clean_all_layer[:, kv_index, :]
                v_corrupt = v_corrupt_all_layer[:, kv_index, :]

                z_corrupt = head_cache_corrupt[z_key][0, -1, head, :].float().to(corrupt_tokens.device)
                delta_by_pos = pat_clean[:, None] * v_clean - pat_corrupt[:, None] * v_corrupt
                seq_len = int(delta_by_pos.shape[0])

                def run_subset(subset_positions: Sequence[int]) -> Tuple[float, torch.Tensor]:
                    mask = torch.zeros(seq_len, device=corrupt_tokens.device)
                    for p in subset_positions:
                        if 0 <= int(p) < seq_len:
                            mask[int(p)] = 1.0
                    target_z = z_corrupt + (delta_by_pos * mask[:, None]).sum(dim=0)
                    target_z = target_z.to(dtype=corrupt_tokens.dtype)

                    hook_name = f"blocks.{layer}.attn.hook_z"

                    def hook_fn(z: torch.Tensor, hook):  # noqa: ANN001
                        z = z.clone()
                        z[:, -1, head, :] = target_z
                        return z

                    try:
                        with torch.no_grad():
                            logits = model.run_with_hooks(corrupt_tokens, fwd_hooks=[(hook_name, hook_fn)])
                    except torch.OutOfMemoryError:
                        gc.collect()
                        torch.cuda.empty_cache()
                        nan_logits = torch.full_like(corrupt_logits[0, -1, :].float(), float("nan"))
                        return float("nan"), nan_logits
                    obj = float(objective_from_logits(logits, target_token, distractor).item())
                    return obj, logits[0, -1, :].float()

                top1 = topk_indices(pat_clean, 1)
                top3 = topk_indices(pat_clean, 3)

                subset_map = {
                    "full": list(range(seq_len)),
                    "top1": top1,
                    "top3": top3,
                    "contrast": pos_sets["contrast"],
                    "tool_call_tags": pos_sets["tool_call_tags"],
                    "tools_block": pos_sets["tools_block"],
                    "user_block": pos_sets["user_block"],
                }

                subset_ratio: Dict[str, float] = {}
                full_logits_last = None
                for name, pos in subset_map.items():
                    patched_obj, patched_last_logits = run_subset(pos)
                    ratio = safe_ratio(patched_obj - corrupt_obj, gap)
                    subset_ratio[name] = ratio
                    if name == "full":
                        full_logits_last = patched_last_logits
                if full_logits_last is None:
                    continue
                if not math.isfinite(float(subset_ratio.get("full", float("nan")))):
                    continue

                attn_mass_clean = {
                    "contrast": float(pat_clean[pos_sets["contrast"]].sum().item()) if pos_sets["contrast"] else 0.0,
                    "tool_call_tags": float(pat_clean[pos_sets["tool_call_tags"]].sum().item()) if pos_sets["tool_call_tags"] else 0.0,
                    "tools_block": float(pat_clean[pos_sets["tools_block"]].sum().item()) if pos_sets["tools_block"] else 0.0,
                    "user_block": float(pat_clean[pos_sets["user_block"]].sum().item()) if pos_sets["user_block"] else 0.0,
                    "recent_32": float(pat_clean[pos_sets["recent_32"]].sum().item()) if pos_sets["recent_32"] else 0.0,
                    "prefix_16": float(pat_clean[pos_sets["prefix_16"]].sum().item()) if pos_sets["prefix_16"] else 0.0,
                }
                attn_mass_corrupt = {
                    "contrast": float(pat_corrupt[pos_sets["contrast"]].sum().item()) if pos_sets["contrast"] else 0.0,
                    "tool_call_tags": float(pat_corrupt[pos_sets["tool_call_tags"]].sum().item()) if pos_sets["tool_call_tags"] else 0.0,
                    "tools_block": float(pat_corrupt[pos_sets["tools_block"]].sum().item()) if pos_sets["tools_block"] else 0.0,
                    "user_block": float(pat_corrupt[pos_sets["user_block"]].sum().item()) if pos_sets["user_block"] else 0.0,
                    "recent_32": float(pat_corrupt[pos_sets["recent_32"]].sum().item()) if pos_sets["recent_32"] else 0.0,
                    "prefix_16": float(pat_corrupt[pos_sets["prefix_16"]].sum().item()) if pos_sets["prefix_16"] else 0.0,
                }

                logits_delta = full_logits_last - corrupt_last_logits
                delta_target = float(logits_delta[target_token].item())
                delta_distractor = float(logits_delta[distractor].item())
                top_pos_token_ids = topk_indices(logits_delta, 5)
                top_pos_tokens = [tokenizer.decode([tid]).replace("\n", "\\n") for tid in top_pos_token_ids]
                for t in top_pos_tokens:
                    node_top_tokens[node][t] += 1

                top1_pos = top1[0] if top1 else -1
                top1_token = decode_pos_token(tokenizer, ids_clean, top1_pos) if top1_pos >= 0 else ""
                if top1_pos in set(pos_sets["contrast"]):
                    top1_cat = "contrast"
                elif top1_pos in set(pos_sets["tool_call_tags"]):
                    top1_cat = "tool_call_tags"
                elif top1_pos in set(pos_sets["tools_block"]):
                    top1_cat = "tools_block"
                elif top1_pos in set(pos_sets["user_block"]):
                    top1_cat = "user_block"
                else:
                    top1_cat = "other"
                head_top1_category_counter[node][top1_cat] += 1
                if top1_token:
                    head_top1_token_counter[node][top1_token] += 1

                row = {
                    "sample_id": sample_id,
                    "sample_rank": sample_rank,
                    "q_index": legacy_index,
                    "node": node,
                    "node_type": "head",
                    "gap": gap,
                    "full_ratio": subset_ratio["full"],
                    "top1_ratio": subset_ratio["top1"],
                    "top3_ratio": subset_ratio["top3"],
                    "contrast_ratio": subset_ratio["contrast"],
                    "tool_call_tags_ratio": subset_ratio["tool_call_tags"],
                    "tools_block_ratio": subset_ratio["tools_block"],
                    "user_block_ratio": subset_ratio["user_block"],
                    "attn_contrast_clean": attn_mass_clean["contrast"],
                    "attn_tool_call_tags_clean": attn_mass_clean["tool_call_tags"],
                    "attn_tools_block_clean": attn_mass_clean["tools_block"],
                    "attn_user_block_clean": attn_mass_clean["user_block"],
                    "attn_recent_32_clean": attn_mass_clean["recent_32"],
                    "attn_prefix_16_clean": attn_mass_clean["prefix_16"],
                    "attn_contrast_corrupt": attn_mass_corrupt["contrast"],
                    "attn_tool_call_tags_corrupt": attn_mass_corrupt["tool_call_tags"],
                    "attn_tools_block_corrupt": attn_mass_corrupt["tools_block"],
                    "attn_user_block_corrupt": attn_mass_corrupt["user_block"],
                    "attn_recent_32_corrupt": attn_mass_corrupt["recent_32"],
                    "attn_prefix_16_corrupt": attn_mass_corrupt["prefix_16"],
                    "top1_pos": top1_pos,
                    "top1_token": top1_token,
                    "top1_category": top1_cat,
                    "target_logit_delta": delta_target,
                    "distractor_logit_delta": delta_distractor,
                    "top_positive_tokens": " | ".join(top_pos_tokens),
                }
                rows.append(row)
                for k, v in row.items():
                    if isinstance(v, (int, float)):
                        node_metric_lists[node][k].append(float(v))

            del head_cache_clean
            del head_cache_corrupt
            gc.collect()
            torch.cuda.empty_cache()

        if head_failed:
            skipped_oom.append(sample_id)
            model.reset_hooks()
            gc.collect()
            torch.cuda.empty_cache()
            continue

        # Analyze MLPs
        for node in mlp_nodes:
            layer = parse_mlp(node)
            mlp_key = f"blocks.{layer}.hook_mlp_out"
            clean_vec = mlp_cache_clean[mlp_key][0, -1, :].to(corrupt_tokens.device)

            def hook_fn(mlp_out: torch.Tensor, hook):  # noqa: ANN001
                mlp_out = mlp_out.clone()
                mlp_out[:, -1, :] = clean_vec.to(dtype=mlp_out.dtype)
                return mlp_out

            with torch.no_grad():
                logits = model.run_with_hooks(corrupt_tokens, fwd_hooks=[(mlp_key, hook_fn)])
            obj = float(objective_from_logits(logits, target_token, distractor).item())
            ratio = safe_ratio(obj - corrupt_obj, gap)

            last_logits = logits[0, -1, :].float()
            logits_delta = last_logits - corrupt_last_logits
            delta_target = float(logits_delta[target_token].item())
            delta_distractor = float(logits_delta[distractor].item())
            top_pos_token_ids = topk_indices(logits_delta, 5)
            top_pos_tokens = [tokenizer.decode([tid]).replace("\n", "\\n") for tid in top_pos_token_ids]
            for t in top_pos_tokens:
                node_top_tokens[node][t] += 1

            row = {
                "sample_id": sample_id,
                "sample_rank": sample_rank,
                "q_index": legacy_index,
                "node": node,
                "node_type": "mlp",
                "gap": gap,
                "full_ratio": ratio,
                "top1_ratio": float("nan"),
                "top3_ratio": float("nan"),
                "contrast_ratio": float("nan"),
                "tool_call_tags_ratio": float("nan"),
                "tools_block_ratio": float("nan"),
                "user_block_ratio": float("nan"),
                "attn_contrast_clean": float("nan"),
                "attn_tool_call_tags_clean": float("nan"),
                "attn_tools_block_clean": float("nan"),
                "attn_user_block_clean": float("nan"),
                "attn_recent_32_clean": float("nan"),
                "attn_prefix_16_clean": float("nan"),
                "attn_contrast_corrupt": float("nan"),
                "attn_tool_call_tags_corrupt": float("nan"),
                "attn_tools_block_corrupt": float("nan"),
                "attn_user_block_corrupt": float("nan"),
                "attn_recent_32_corrupt": float("nan"),
                "attn_prefix_16_corrupt": float("nan"),
                "top1_pos": -1,
                "top1_token": "",
                "top1_category": "",
                "target_logit_delta": delta_target,
                "distractor_logit_delta": delta_distractor,
                "top_positive_tokens": " | ".join(top_pos_tokens),
            }
            rows.append(row)
            for k, v in row.items():
                if isinstance(v, (int, float)):
                    node_metric_lists[node][k].append(float(v))

        analyzed_sample_ids.append(sample_id)
        model.reset_hooks()
        del mlp_cache_clean
        gc.collect()
        torch.cuda.empty_cache()

    # Save per-sample rows
    csv_path = out_root / "semantic_node_metrics.csv"
    fieldnames = [
        "sample_id",
        "sample_rank",
        "q_index",
        "node",
        "node_type",
        "gap",
        "full_ratio",
        "top1_ratio",
        "top3_ratio",
        "contrast_ratio",
        "tool_call_tags_ratio",
        "tools_block_ratio",
        "user_block_ratio",
        "attn_contrast_clean",
        "attn_tool_call_tags_clean",
        "attn_tools_block_clean",
        "attn_user_block_clean",
        "attn_recent_32_clean",
        "attn_prefix_16_clean",
        "attn_contrast_corrupt",
        "attn_tool_call_tags_corrupt",
        "attn_tools_block_corrupt",
        "attn_user_block_corrupt",
        "attn_recent_32_corrupt",
        "attn_prefix_16_corrupt",
        "top1_pos",
        "top1_token",
        "top1_category",
        "target_logit_delta",
        "distractor_logit_delta",
        "top_positive_tokens",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # Aggregate
    node_summary: Dict[str, Dict[str, object]] = {}
    role_labels: Dict[str, str] = {}
    for node in core_nodes:
        m = node_metric_lists[node]
        summary = {
            "full_ratio_median": med(m.get("full_ratio", [])),
            "full_ratio_mean": mean(m.get("full_ratio", [])),
            "top1_ratio_median": med(m.get("top1_ratio", [])),
            "top3_ratio_median": med(m.get("top3_ratio", [])),
            "contrast_ratio_median": med(m.get("contrast_ratio", [])),
            "tool_call_tags_ratio_median": med(m.get("tool_call_tags_ratio", [])),
            "tools_block_ratio_median": med(m.get("tools_block_ratio", [])),
            "user_block_ratio_median": med(m.get("user_block_ratio", [])),
            "attn_contrast_clean_mean": mean(m.get("attn_contrast_clean", [])),
            "attn_tool_call_tags_clean_mean": mean(m.get("attn_tool_call_tags_clean", [])),
            "attn_tools_block_clean_mean": mean(m.get("attn_tools_block_clean", [])),
            "attn_user_block_clean_mean": mean(m.get("attn_user_block_clean", [])),
            "attn_contrast_delta_mean": mean(
                np.array(finite_vals(m.get("attn_contrast_clean", [])))
                - np.array(finite_vals(m.get("attn_contrast_corrupt", [])))
                if finite_vals(m.get("attn_contrast_clean", [])) and finite_vals(m.get("attn_contrast_corrupt", []))
                else []
            ),
            "attn_tool_call_tags_delta_mean": mean(
                np.array(finite_vals(m.get("attn_tool_call_tags_clean", [])))
                - np.array(finite_vals(m.get("attn_tool_call_tags_corrupt", [])))
                if finite_vals(m.get("attn_tool_call_tags_clean", []))
                and finite_vals(m.get("attn_tool_call_tags_corrupt", []))
                else []
            ),
            "target_logit_delta_median": med(m.get("target_logit_delta", [])),
            "distractor_logit_delta_median": med(m.get("distractor_logit_delta", [])),
            "top_positive_tokens": [t for t, _ in node_top_tokens[node].most_common(8)],
            "top_positive_token_counts": dict(node_top_tokens[node].most_common(8)),
        }

        if node.startswith("L"):
            summary["top1_category_counts"] = dict(head_top1_category_counter[node])
            summary["top1_token_counts"] = dict(head_top1_token_counter[node].most_common(8))
            role = role_from_metrics(summary)  # type: ignore[arg-type]
        else:
            role = mlp_role_from_metrics(summary)  # type: ignore[arg-type]
        summary["role"] = role
        role_labels[node] = role
        node_summary[node] = summary

    # Heatmaps
    head_rows = head_nodes
    read_cols = ["full_ratio", "top1_ratio", "top3_ratio", "contrast_ratio", "tool_call_tags_ratio", "tools_block_ratio", "user_block_ratio"]
    read_matrix = np.array(
        [
            [
                float(node_summary[h].get("full_ratio_median", float("nan"))),
                float(node_summary[h].get("top1_ratio_median", float("nan"))),
                float(node_summary[h].get("top3_ratio_median", float("nan"))),
                float(node_summary[h].get("contrast_ratio_median", float("nan"))),
                float(node_summary[h].get("tool_call_tags_ratio_median", float("nan"))),
                float(node_summary[h].get("tools_block_ratio_median", float("nan"))),
                float(node_summary[h].get("user_block_ratio_median", float("nan"))),
            ]
            for h in head_rows
        ],
        dtype=np.float64,
    )
    save_diverging_heatmap(
        matrix=read_matrix,
        row_labels=head_rows,
        col_labels=read_cols,
        title="Head Read-Side Causal Ratios (Median, Patched from Clean)",
        out_path=out_root / "semantic_read_causal_heatmap.png",
    )

    attn_cols = ["contrast", "tool_call_tags", "tools_block", "user_block"]
    attn_matrix = np.array(
        [
            [
                float(node_summary[h].get("attn_contrast_delta_mean", float("nan"))),
                float(node_summary[h].get("attn_tool_call_tags_delta_mean", float("nan"))),
                float(node_summary[h].get("attn_tools_block_clean_mean", float("nan")))
                - float(mean(node_metric_lists[h].get("attn_tools_block_corrupt", []))),
                float(node_summary[h].get("attn_user_block_clean_mean", float("nan")))
                - float(mean(node_metric_lists[h].get("attn_user_block_corrupt", []))),
            ]
            for h in head_rows
        ],
        dtype=np.float64,
    )
    save_diverging_heatmap(
        matrix=attn_matrix,
        row_labels=head_rows,
        col_labels=attn_cols,
        title="Attention Mass Delta (Clean - Corrupt) at Final Query",
        out_path=out_root / "semantic_attention_delta_heatmap.png",
    )

    target_bars = {n: float(node_summary[n].get("target_logit_delta_median", float("nan"))) for n in core_nodes}
    save_bar(
        values=target_bars,
        title="Node Write-Side Effect: Delta Logit(<tool_call>) Median",
        ylabel="Delta logit",
        out_path=out_root / "semantic_write_target_delta.png",
    )

    # Keep node ids unchanged for draw_circuit layout; roles are written to a companion CSV.
    draw_circuit(
        nodes=core_nodes,
        edges=core_edges,
        out_path=out_root / "final_circuit_semantic.png",
        title="Semantic-Role Circuit (See node_roles.csv for role labels)",
    )

    role_csv = out_root / "node_roles.csv"
    with role_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["node", "role", "full_ratio_median", "target_logit_delta_median"])
        for n in core_nodes:
            w.writerow(
                [
                    n,
                    role_labels.get(n, ""),
                    node_summary[n].get("full_ratio_median"),
                    node_summary[n].get("target_logit_delta_median"),
                ]
            )

    report = {
        "n_samples": len(sample_infos),
        "n_analyzed_samples": len(set(analyzed_sample_ids)),
        "analyzed_sample_ids": sorted(set(analyzed_sample_ids)),
        "gap_min": args.gap_min,
        "recompute_baseline": args.recompute_baseline,
        "skipped_oom_sample_ids": sorted(set(skipped_oom)),
        "core_nodes": core_nodes,
        "head_nodes": head_nodes,
        "mlp_nodes": mlp_nodes,
        "node_summary": node_summary,
        "artifacts": {
            "semantic_node_metrics_csv": str(csv_path),
            "semantic_read_causal_heatmap": str(out_root / "semantic_read_causal_heatmap.png"),
            "semantic_attention_delta_heatmap": str(out_root / "semantic_attention_delta_heatmap.png"),
            "semantic_write_target_delta": str(out_root / "semantic_write_target_delta.png"),
            "final_circuit_semantic": str(out_root / "final_circuit_semantic.png"),
            "node_roles_csv": str(role_csv),
        },
    }
    (out_root / "semantic_roles_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"[done] output root: {out_root}")
    print(f"[done] samples: {len(sample_infos)}")
    print(f"[done] report: {out_root / 'semantic_roles_report.json'}")


if __name__ == "__main__":
    main()
