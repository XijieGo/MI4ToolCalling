#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

from task_attention_path_analysis import (
    DATASET_ROOT,
    MODEL_PATH,
    ensure_dir,
    load_samples,
    locate_schema_span,
    set_seed,
    write_csv,
    write_text,
)

from phase4_reviewer_strengthening import (
    L24_LAYER,
    SEED,
    TOOL_CALL_STR,
    SinglePromptItem,
    build_single_batches,
    make_length_matched_ids,
    make_resid_last_capture,
    token_span_from_char_span,
    tool_stats,
)

import sys

LEGACY_SRC = Path("./src")
if str(LEGACY_SRC) not in sys.path:
    sys.path.insert(0, str(LEGACY_SRC))

from toolcall_circuit.single_sample import load_hooked_qwen3  # noqa: E402


PHASE5_ROOT = Path("./results/8B/phase5_neurips/exp_e_schema_verb_state")
PHASE5_B_ROOT = Path("./results/8B/phase5_neurips/exp_b_gate_direction/pca_components.pt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 5 experiment E: schema-verb interaction in PC1 gate state")
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH)
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--pc1-path", type=Path, default=PHASE5_B_ROOT)
    parser.add_argument("--output-root", type=Path, default=PHASE5_ROOT)
    parser.add_argument("--max-pairs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=SEED)
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
        "unrelated_valid_schema_length_matched": make_length_matched_ids(
            tokenizer,
            prefix_text='{"type":"function","function":{"name":"check_weather","description":"Weather',
            suffix_text='","parameters":{"type":"object","properties":{"city":{"type":"string"},"date":{"type":"string"}},"required":["city","date"]}}}',
            filler_text=" weather",
            target_len=target_schema_len,
        ),
        "random_text_length_matched": make_length_matched_ids(
            tokenizer,
            prefix_text="random text",
            suffix_text="",
            filler_text=" lorem",
            target_len=target_schema_len,
        ),
    }
    requested_conditions = {
        ("clean_action_side", "original_schema"),
        ("clean_action_side", "schema_removed"),
        ("clean_action_side", "random_text_length_matched"),
        ("clean_action_side", "unrelated_valid_schema_length_matched"),
        ("corrupt_analysis_side", "original_schema"),
        ("corrupt_analysis_side", "schema_removed"),
    }

    items: list[SinglePromptItem] = []
    for side, verb_condition in (("clean", "clean_action_side"), ("corrupt", "corrupt_analysis_side")):
        for sample in samples:
            tokens = sample.clean_tokens_cpu[0].clone().long() if side == "clean" else sample.corrupt_tokens_cpu[0].clone().long()
            offsets = sample.clean_offsets if side == "clean" else sample.corrupt_offsets
            text = sample.clean_text if side == "clean" else sample.corrupt_text
            span = locate_schema_span(text)
            if span is None:
                raise ValueError("Missing schema span in prompt.")
            start, end = token_span_from_char_span(offsets, span)
            system_mask = sample.clean_region_masks["system"] if side == "clean" else sample.corrupt_region_masks["system"]
            for schema_type, ids in replacement_ids.items():
                if (verb_condition, schema_type) not in requested_conditions:
                    continue
                modified = torch.cat([tokens[:start], ids, tokens[end:]], dim=0)
                if int(modified.shape[0]) != int(tokens.shape[0]):
                    raise RuntimeError(f"Length mismatch for {verb_condition} / {schema_type} on {sample.sample_id}")
                items.append(
                    SinglePromptItem(
                        sample_id=sample.sample_id,
                        condition=f"{verb_condition}__{schema_type}",
                        verb_condition=verb_condition,
                        token_len=int(modified.shape[0]),
                        tokens_cpu=modified,
                        system_mask=system_mask,
                    )
                )
    return items


def main() -> None:
    args = parse_args()
    ensure_dir(args.output_root)
    set_seed(args.seed)

    pc_bundle = torch.load(args.pc1_path, map_location="cpu")
    pc1 = pc_bundle["components"][0].float()

    model, tokenizer = load_hooked_qwen3(str(args.model_path), device=args.device, dtype=torch.bfloat16)
    tool_token_ids = tokenizer.encode(TOOL_CALL_STR, add_special_tokens=False)
    if len(tool_token_ids) != 1:
        raise ValueError(f"{TOOL_CALL_STR!r} maps to unexpected token ids: {tool_token_ids}")
    tool_token_id = int(tool_token_ids[0])

    samples = load_samples(args.dataset_root, model, tokenizer, max_pairs=args.max_pairs)
    items = build_items(samples, tokenizer)
    batches = build_single_batches(items, args.batch_size)
    resid_hook_name = f"blocks.{L24_LAYER}.hook_resid_pre"

    per_sample_rows: list[dict[str, object]] = []
    progress = tqdm(batches, desc="Exp E score PC1", dynamic_ncols=True)
    for batch in progress:
        tokens = batch.tokens_cpu.to(model.W_U.device)
        capture: dict[str, torch.Tensor] = {}
        with torch.no_grad():
            logits = model.run_with_hooks(tokens, fwd_hooks=[(resid_hook_name, make_resid_last_capture(capture, "resid"))])
        tool_logit, top1 = tool_stats(logits, tool_token_id)
        scores = torch.mv(capture["resid"], pc1)
        for local_idx, item in enumerate(batch.items):
            verb_condition, schema_type = str(item.condition).split("__", maxsplit=1)
            per_sample_rows.append(
                {
                    "sample_id": item.sample_id,
                    "condition": item.condition,
                    "verb_condition": verb_condition,
                    "schema_type": schema_type,
                    "pc1_score": float(scores[local_idx].item()),
                    "tool_logit": float(tool_logit[local_idx].item()),
                    "is_tool_call_top1": int(top1[local_idx].item() == tool_token_id),
                }
            )

    write_csv(args.output_root / "gate_signal_per_sample.csv", per_sample_rows)

    grouped_rows: list[dict[str, object]] = []
    for condition in sorted({str(row["condition"]) for row in per_sample_rows}):
        group = [row for row in per_sample_rows if str(row["condition"]) == condition]
        scores = np.asarray([float(row["pc1_score"]) for row in group], dtype=np.float64)
        logits = np.asarray([float(row["tool_logit"]) for row in group], dtype=np.float64)
        top1 = np.asarray([float(row["is_tool_call_top1"]) for row in group], dtype=np.float64)
        grouped_rows.append(
            {
                "condition": condition,
                "verb_condition": str(group[0]["verb_condition"]),
                "schema_type": str(group[0]["schema_type"]),
                "mean_pc1_score": float(scores.mean()),
                "std_pc1_score": float(scores.std(ddof=1)),
                "mean_tool_logit": float(logits.mean()),
                "tool_call_rate": float(top1.mean()),
                "n_samples": int(len(group)),
            }
        )
    grouped_rows.sort(key=lambda row: float(row["mean_pc1_score"]), reverse=True)
    write_csv(args.output_root / "gate_signal_by_condition.csv", grouped_rows)

    mean_scores = np.asarray([float(row["mean_pc1_score"]) for row in grouped_rows], dtype=np.float64)
    tool_rates = np.asarray([float(row["tool_call_rate"]) for row in grouped_rows], dtype=np.float64)
    pearson = float(np.corrcoef(mean_scores, tool_rates)[0, 1]) if len(grouped_rows) >= 2 else float("nan")

    lines = [
        "# Experiment E: Schema-Verb State Mechanism",
        "",
        f"- 模型: `{args.model_path}`",
        f"- 样本: `datasets/test` 前 `{len(samples)}` 对",
        f"- PC1 来源: `{args.pc1_path}`",
        "",
        "## Condition Means",
        "",
        "| condition | mean_pc1_score | mean_tool_logit | tool_call_rate |",
        "|---|---:|---:|---:|",
    ]
    for row in grouped_rows:
        lines.append(
            f"| {row['condition']} | {float(row['mean_pc1_score']):.4f} | "
            f"{float(row['mean_tool_logit']):.4f} | {float(row['tool_call_rate']):.2%} |"
        )
    lines.extend(
        [
            "",
            "## Correlation",
            "",
            f"- Pearson(mean PC1 score, tool-call rate) = `{pearson:.4f}` across `{len(grouped_rows)}` schema-verb conditions.",
        ]
    )
    write_text(args.output_root / "summary.md", "\n".join(lines))

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
