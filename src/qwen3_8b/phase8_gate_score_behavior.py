#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from phase4_reviewer_strengthening import (
    SinglePromptItem,
    build_single_batches,
    make_length_matched_ids,
    token_span_from_char_span,
)
from phase8_common import (
    DEFAULT_PC_BUNDLE,
    EVAL_DATASET_ROOT,
    EXP_D_ROOT,
    L24_LAYER,
    MODEL_PATH,
    configure_matplotlib,
    ensure_dir,
    get_tool_token_id,
    load_gate_direction,
    load_model_and_tokenizer,
    load_samples,
    make_last_token_vector_capture,
    manifest_pair_count,
    projection,
    safe_corrcoef,
    set_seed,
    tool_stats,
    write_csv,
    write_text,
)
from task_attention_path_analysis import locate_schema_span


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 8 Exp D: gate score vs behavior / condition.")
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH)
    parser.add_argument("--dataset-root", type=Path, default=EVAL_DATASET_ROOT)
    parser.add_argument("--pc-bundle", type=Path, default=DEFAULT_PC_BUNDLE)
    parser.add_argument("--output-root", type=Path, default=EXP_D_ROOT)
    parser.add_argument("--max-pairs", type=int, default=manifest_pair_count(EVAL_DATASET_ROOT))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def build_items(samples, tokenizer) -> list[SinglePromptItem]:
    first_sample = samples[0]
    schema_span = locate_schema_span(first_sample.clean_text)
    if schema_span is None:
        raise ValueError("Could not locate schema span in sample prompt.")
    schema_start, schema_end = token_span_from_char_span(first_sample.clean_offsets, schema_span)
    target_schema_len = schema_end - schema_start
    original_schema_ids = first_sample.clean_tokens_cpu[0, schema_start:schema_end].clone().long()

    replacement_ids = {
        "original_schema": original_schema_ids,
        "schema_removed": make_length_matched_ids(
            tokenizer,
            prefix_text="[schema removed]",
            suffix_text="",
            filler_text=" neutral",
            target_len=target_schema_len,
        ),
        "unrelated_schema": make_length_matched_ids(
            tokenizer,
            prefix_text='{"type":"function","function":{"name":"check_weather","description":"Weather',
            suffix_text='","parameters":{"type":"object","properties":{"city":{"type":"string"},"date":{"type":"string"}},"required":["city","date"]}}}',
            filler_text=" weather",
            target_len=target_schema_len,
        ),
        "random_text": make_length_matched_ids(
            tokenizer,
            prefix_text="random text",
            suffix_text="",
            filler_text=" lorem",
            target_len=target_schema_len,
        ),
    }

    requested = {
        ("clean", "original_schema", "clean_original_schema"),
        ("clean", "schema_removed", "clean_schema_removed"),
        ("clean", "random_text", "clean_random_text"),
        ("clean", "unrelated_schema", "clean_unrelated_schema"),
        ("corrupt", "original_schema", "corrupt_original_schema"),
        ("corrupt", "schema_removed", "corrupt_schema_removed"),
    }

    items: list[SinglePromptItem] = []
    for side, replacement_key, label in requested:
        for sample in samples:
            tokens = sample.clean_tokens_cpu[0].clone().long() if side == "clean" else sample.corrupt_tokens_cpu[0].clone().long()
            offsets = sample.clean_offsets if side == "clean" else sample.corrupt_offsets
            text = sample.clean_text if side == "clean" else sample.corrupt_text
            span = locate_schema_span(text)
            if span is None:
                raise ValueError(f"Missing schema span for {sample.sample_id}")
            start, end = token_span_from_char_span(offsets, span)
            modified = torch.cat([tokens[:start], replacement_ids[replacement_key], tokens[end:]], dim=0)
            if int(modified.shape[0]) != int(tokens.shape[0]):
                raise RuntimeError(f"Length mismatch for {label} on {sample.sample_id}")
            items.append(
                SinglePromptItem(
                    sample_id=sample.sample_id,
                    condition=label,
                    verb_condition=side,
                    token_len=int(modified.shape[0]),
                    tokens_cpu=modified,
                    system_mask=(sample.clean_region_masks["system"] if side == "clean" else sample.corrupt_region_masks["system"]),
                )
            )
    return items


def condition_family(label: str) -> str:
    if label == "clean_original_schema":
        return "clean_baseline"
    if label == "corrupt_original_schema":
        return "corrupt_baseline"
    if "removed" in label:
        return "schema_removed"
    if "random" in label:
        return "random_text"
    return "unrelated_schema"


def short_label(label: str) -> str:
    mapping = {
        "clean_original_schema": "clean+orig",
        "corrupt_original_schema": "corrupt+orig",
        "clean_schema_removed": "clean-removed",
        "corrupt_schema_removed": "corrupt-removed",
        "clean_random_text": "clean-random",
        "clean_unrelated_schema": "clean-unrel",
    }
    return mapping.get(label, label)


def plot_behavior(rows: list[dict[str, object]], path: Path) -> None:
    configure_matplotlib()
    family_colors = {
        "clean_baseline": "#1b6ca8",
        "corrupt_baseline": "#cc5803",
        "schema_removed": "#6c757d",
        "random_text": "#8d6a9f",
        "unrelated_schema": "#2a9d8f",
    }

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.0))
    for row in rows:
        family = str(row["condition_family"])
        label = short_label(str(row["condition"]))
        x = float(row["mean_gate_score"])
        y_rate = float(row["tool_call_rate"])
        y_logit = float(row["mean_tool_logit"])
        color = family_colors[family]
        axes[0].scatter([x], [y_rate], color=color, s=70)
        axes[1].scatter([x], [y_logit], color=color, s=70)
        axes[0].annotate(label, (x, y_rate), xytext=(6, 6), textcoords="offset points", fontsize=9)
        axes[1].annotate(label, (x, y_logit), xytext=(6, 6), textcoords="offset points", fontsize=9)

    axes[0].set_xlabel("mean L24 gate score")
    axes[0].set_ylabel("tool-call rate")
    axes[0].set_title("Gate Score vs Tool-Call Rate")
    axes[0].set_ylim(-0.02, 1.02)

    axes[1].set_xlabel("mean L24 gate score")
    axes[1].set_ylabel("mean <tool_call> logit")
    axes[1].set_title("Gate Score vs Tool Logit")

    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=color, markersize=8, label=family)
        for family, color in family_colors.items()
    ]
    axes[1].legend(handles=handles, frameon=False, loc="best")
    fig.tight_layout()
    ensure_dir(path.parent)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    ensure_dir(args.output_root)
    set_seed(args.seed)

    model, tokenizer = load_model_and_tokenizer(model_path=args.model_path, device=args.device)
    tool_token_id = get_tool_token_id(tokenizer)
    gate_direction = load_gate_direction(args.pc_bundle)
    samples = load_samples(args.dataset_root, model, tokenizer, max_pairs=args.max_pairs)
    items = build_items(samples, tokenizer)
    batches = build_single_batches(items, args.batch_size)
    resid_hook_name = f"blocks.{L24_LAYER}.hook_resid_pre"

    per_sample_rows: list[dict[str, object]] = []
    for batch in batches:
        capture: dict[str, torch.Tensor] = {}
        with torch.no_grad():
            logits = model.run_with_hooks(
                batch.tokens_cpu.to(model.W_U.device),
                fwd_hooks=[(resid_hook_name, make_last_token_vector_capture(capture, "resid"))],
            )
        tool_logit, top1 = tool_stats(logits, tool_token_id)
        gate_scores = projection(capture["resid"], gate_direction)
        for local_idx, item in enumerate(batch.items):
            per_sample_rows.append(
                {
                    "sample_id": item.sample_id,
                    "condition": item.condition,
                    "condition_family": condition_family(str(item.condition)),
                    "gate_score": float(gate_scores[local_idx].item()),
                    "tool_logit": float(tool_logit[local_idx].item()),
                    "is_tool_call_top1": int(top1[local_idx].item() == tool_token_id),
                }
            )

    summary_rows: list[dict[str, object]] = []
    for condition in sorted({str(row["condition"]) for row in per_sample_rows}):
        group = [row for row in per_sample_rows if str(row["condition"]) == condition]
        gate_scores = np.asarray([float(row["gate_score"]) for row in group], dtype=np.float64)
        tool_logits = np.asarray([float(row["tool_logit"]) for row in group], dtype=np.float64)
        tool_calls = np.asarray([float(row["is_tool_call_top1"]) for row in group], dtype=np.float64)
        summary_rows.append(
            {
                "condition": condition,
                "condition_family": str(group[0]["condition_family"]),
                "mean_gate_score": float(gate_scores.mean()),
                "std_gate_score": float(gate_scores.std(ddof=1)),
                "mean_tool_logit": float(tool_logits.mean()),
                "std_tool_logit": float(tool_logits.std(ddof=1)),
                "tool_call_rate": float(tool_calls.mean()),
                "n_samples": len(group),
            }
        )

    summary_rows.sort(key=lambda row: float(row["mean_gate_score"]))
    write_csv(args.output_root / "gate_behavior_per_sample.csv", per_sample_rows)
    write_csv(args.output_root / "gate_behavior_table.csv", summary_rows)
    plot_behavior(summary_rows, args.output_root / "plot_gate_score_vs_behavior.pdf")

    x = np.asarray([float(row["mean_gate_score"]) for row in summary_rows], dtype=np.float64)
    y_rate = np.asarray([float(row["tool_call_rate"]) for row in summary_rows], dtype=np.float64)
    y_logit = np.asarray([float(row["mean_tool_logit"]) for row in summary_rows], dtype=np.float64)
    corr_rate = safe_corrcoef(x, y_rate)
    corr_logit = safe_corrcoef(x, y_logit)

    lines = [
        "# Phase 8 Exp D: Gate Score vs Behavior / Condition",
        "",
        f"- Eval split: `{args.dataset_root}` with `{len(samples)}` prompt pairs.",
        f"- Pearson(mean gate score, tool-call rate): `{corr_rate:.4f}` across `{len(summary_rows)}` condition means.",
        f"- Pearson(mean gate score, tool logit): `{corr_logit:.4f}` across `{len(summary_rows)}` condition means.",
        "",
        "| condition | mean_gate_score | mean_tool_logit | tool_call_rate |",
        "|---|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['condition']} | {float(row['mean_gate_score']):.4f} | "
            f"{float(row['mean_tool_logit']):.4f} | {float(row['tool_call_rate']):.2%} |"
        )
    write_text(args.output_root / "summary.md", "\n".join(lines))

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
