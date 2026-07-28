#!/usr/bin/env python3
"""
Causal tracing for the clean/corrupt contrast token set.

For each sample, patch one residual state subset (layer l, positions P) from clean into corrupt,
and measure objective recovery. Aggregates median recovery curves across samples.
"""

from __future__ import annotations

import os
import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from statistics import median
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm.auto import tqdm

if __package__ in {None, ""}:
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from toolcall_circuit.dataset import (
    build_position_sets,
    get_tool_call_target_spec,
    load_summary_records,
    resolve_distractor_token,
)
from toolcall_circuit.paths import MODEL_PATH_DEFAULT, RESULTS_ROOT
from toolcall_circuit.single_sample import load_hooked_qwen3, objective_from_logits


def med(xs: Sequence[float]) -> float:
    vals = [float(x) for x in xs if isinstance(x, (int, float)) and math.isfinite(float(x))]
    return float(median(vals)) if vals else float("nan")


def mean(xs: Sequence[float]) -> float:
    vals = [float(x) for x in xs if isinstance(x, (int, float)) and math.isfinite(float(x))]
    return float(np.mean(vals)) if vals else float("nan")


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Trace contrast-token-set causal effect across layers.")
    parser.add_argument("--input-root", type=str, default=str(RESULTS_ROOT / "manual_run" / "batch"))
    parser.add_argument("--output-root", type=str, default=str(RESULTS_ROOT / "manual_run" / "semantic_roles"))
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

    model, tokenizer = load_hooked_qwen3(args.model_path, device=args.device, dtype=torch.bfloat16)
    model.set_use_attn_result(False)
    model.set_use_split_qkv_input(False)
    model.set_use_hook_mlp_in(False)
    n_layers = int(model.cfg.n_layers)

    target_spec = get_tool_call_target_spec(tokenizer, target_text="<tool_call>")
    if not target_spec.is_single_token:
        raise NotImplementedError(
            f"<tool_call> tokenization is no longer single-token ({target_spec.token_ids}); "
            "upgrade the tracing objective before running this workflow."
        )
    target_token = target_spec.primary_token_id
    prefix_positions = [0]

    sample_infos: List[Tuple[str, int | None, int | None, Dict[str, object], Path, Path]] = []
    for summary_record in load_summary_records(input_root):
        summary = summary_record.summary
        sample_rank = summary_record.sample_rank
        legacy_index = summary_record.legacy_index
        if legacy_index is not None and (legacy_index < args.q_start or legacy_index > args.q_end):
            continue
        if sample_rank is not None:
            if args.sample_rank_start > 0 and sample_rank < args.sample_rank_start:
                continue
            if args.sample_rank_end > 0 and sample_rank > args.sample_rank_end:
                continue
        gap = float(summary.get("gap", float("nan")))
        if not math.isfinite(gap) or gap <= args.gap_min:
            continue
        clean_prompt = Path(summary["clean_prompt"])
        corrupt_prompt = Path(summary["corrupt_prompt"])
        if not clean_prompt.exists() or not corrupt_prompt.exists():
            continue
        sample_infos.append(
            (
                summary_record.sample_id,
                summary_record.sample_rank,
                summary_record.legacy_index,
                summary,
                clean_prompt,
                corrupt_prompt,
            )
        )
    if args.max_samples > 0:
        sample_infos = sample_infos[: args.max_samples]
    if not sample_infos:
        raise ValueError("No valid samples selected.")

    layer_vals_contrast: Dict[int, List[float]] = {l: [] for l in range(n_layers)}
    layer_vals_toolcall: Dict[int, List[float]] = {l: [] for l in range(n_layers)}
    layer_vals_prefix: Dict[int, List[float]] = {l: [] for l in range(n_layers)}
    skipped: List[str] = []
    analyzed: List[str] = []
    contrast_pos_counter: Counter[int] = Counter()
    contrast_span_len_counter: Counter[int] = Counter()
    tool_call_pos_counter: Counter[int] = Counter()

    pbar = tqdm(sample_infos, desc="Contrast trace", dynamic_ncols=True)
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
        tool_call_positions = [int(p) for p in pos_sets["tool_call_tags"]]
        if not contrast_positions:
            skipped.append(sample_id)
            continue

        clean_obj = float(summary.get("clean_obj", float("nan")))
        corrupt_obj = float(summary.get("corrupt_obj", float("nan")))
        gap = float(summary.get("gap", clean_obj - corrupt_obj))
        if not (math.isfinite(clean_obj) and math.isfinite(corrupt_obj) and math.isfinite(gap)) or abs(gap) < 1e-8:
            skipped.append(sample_id)
            continue

        try:
            with torch.no_grad():
                clean_logits = model(clean_tokens)
                corrupt_logits = model(corrupt_tokens)
            distractor = int(summary.get("distractor_token_id", -1))
            if distractor < 0 or distractor >= corrupt_logits.shape[-1] or distractor == target_token:
                distractor = resolve_distractor_token(corrupt_logits[0, -1, :], target_token)

            if args.recompute_baseline:
                clean_obj = float(objective_from_logits(clean_logits, target_token, distractor).item())
                corrupt_obj = float(objective_from_logits(corrupt_logits, target_token, distractor).item())
                gap = clean_obj - corrupt_obj
                if not math.isfinite(gap) or abs(gap) < 1e-8 or gap <= args.gap_min:
                    skipped.append(sample_id)
                    continue
        except torch.OutOfMemoryError:
            skipped.append(sample_id)
            model.reset_hooks()
            torch.cuda.empty_cache()
            continue

        for pos in contrast_positions:
            contrast_pos_counter[pos] += 1
        contrast_span_len_counter[len(contrast_positions)] += 1
        for pos in tool_call_positions:
            tool_call_pos_counter[pos] += 1

        for layer in range(n_layers):
            try:
                hook_name = f"blocks.{layer}.hook_resid_pre"
                with torch.no_grad():
                    _, clean_cache = model.run_with_cache(clean_tokens, names_filter=lambda n, t=hook_name: n == t)
                clean_layer = clean_cache[hook_name][0].detach().cpu()
                del clean_cache

                def patch_positions(positions: Sequence[int]) -> float:
                    if not positions:
                        return float("nan")
                    patch_positions_local = sorted({int(pos) for pos in positions if 0 <= int(pos) < clean_layer.shape[0]})
                    if not patch_positions_local:
                        return float("nan")

                    def hook_fn(resid: torch.Tensor, hook):  # noqa: ANN001
                        out = resid.clone()
                        for pos in patch_positions_local:
                            clean_vec = clean_layer[pos, :].to(out.device, dtype=out.dtype)
                            out[:, pos, :] = clean_vec
                        return out

                    with torch.no_grad():
                        logits = model.run_with_hooks(corrupt_tokens, fwd_hooks=[(hook_name, hook_fn)])
                    obj = float(objective_from_logits(logits, target_token, distractor).item())
                    return (obj - corrupt_obj) / gap

                layer_vals_contrast[layer].append(patch_positions(contrast_positions))
                layer_vals_toolcall[layer].append(patch_positions(tool_call_positions))
                layer_vals_prefix[layer].append(patch_positions(prefix_positions))
                del clean_layer
            except torch.OutOfMemoryError:
                break

        analyzed.append(sample_id)
        model.reset_hooks()
        torch.cuda.empty_cache()

    layers = list(range(n_layers))
    contrast_med = [med(layer_vals_contrast[l]) for l in layers]
    toolcall_med = [med(layer_vals_toolcall[l]) for l in layers]
    prefix_med = [med(layer_vals_prefix[l]) for l in layers]
    contrast_mean = [mean(layer_vals_contrast[l]) for l in layers]

    apply_plot_style()
    fig, ax = plt.subplots(figsize=(11.5, 5.6), constrained_layout=True)
    ax.plot(layers, contrast_med, label="Patch contrast token set", color="#1f77b4", linewidth=2.2)
    ax.plot(layers, toolcall_med, label="Patch <tool_call> tag set", color="#d62728", linewidth=1.8, alpha=0.85)
    ax.plot(layers, prefix_med, label="Patch prefix token (pos 0)", color="#2ca02c", linewidth=1.8, alpha=0.85)
    ax.axhline(0.0, color="#444444", linewidth=1.0)
    ax.set_xlabel("Layer (hook_resid_pre)")
    ax.set_ylabel("Recovery ratio vs gap")
    ax.set_title("Causal Trace by Position Set: Recovery Curve Across Layers")
    ax.legend(loc="best")
    fig.savefig(out_root / "contrast_token_trace.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    report = {
        "n_samples": len(sample_infos),
        "n_analyzed_samples": len(set(analyzed)),
        "analyzed_sample_ids": sorted(set(analyzed)),
        "skipped_sample_ids": sorted(set(skipped)),
        "contrast_position_hist": {str(k): int(v) for k, v in sorted(contrast_pos_counter.items())},
        "contrast_span_length_hist": {str(k): int(v) for k, v in sorted(contrast_span_len_counter.items())},
        "tool_call_tag_position_hist": {str(k): int(v) for k, v in sorted(tool_call_pos_counter.items())},
        "prefix_positions": prefix_positions,
        "layers": layers,
        "contrast_recovery_median": contrast_med,
        "contrast_recovery_mean": contrast_mean,
        "toolcall_tag_recovery_median": toolcall_med,
        "prefix_recovery_median": prefix_med,
        "artifacts": {
            "trace_png": str(out_root / "contrast_token_trace.png"),
        },
    }
    (out_root / "contrast_token_trace_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[done] wrote {out_root / 'contrast_token_trace_report.json'}")


if __name__ == "__main__":
    main()
