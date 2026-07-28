#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

from task_attention_path_analysis import (
    DATASET_ROOT,
    MODEL_PATH,
    REGIONS,
    build_pair_batches,
    clear_cuda,
    ensure_dir,
    load_samples,
    set_seed,
    write_csv,
    write_text,
)

from phase4_reviewer_strengthening import (
    L24_LAYER,
    SEED,
    TOOL_CALL_STR,
    make_resid_full_capture,
    make_resid_position_replace_hook,
    make_z_zero_hook,
    tool_stats,
)

import sys

LEGACY_SRC = Path("./src")
if str(LEGACY_SRC) not in sys.path:
    sys.path.insert(0, str(LEGACY_SRC))

from toolcall_circuit.single_sample import load_hooked_qwen3  # noqa: E402


PHASE5_ROOT = Path("./results/8B/phase5_neurips/exp_d_cross_position")
CANDIDATE_LAYERS = tuple(range(28, 36))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 5 experiment D: cross-position path localization")
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH)
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--output-root", type=Path, default=PHASE5_ROOT)
    parser.add_argument("--max-pairs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--candidate-layers", type=int, nargs="+", default=list(CANDIDATE_LAYERS))
    parser.add_argument("--candidate-threshold", type=float, default=0.01)
    parser.add_argument("--top-candidates", type=int, default=10)
    return parser.parse_args()


def make_pattern_all_heads_lastrow_capture(capture: dict[str, torch.Tensor], key: str):
    def hook_fn(value: torch.Tensor, hook):  # noqa: ANN001
        capture[key] = value[:, :, -1, :].detach().cpu().float()
        return value

    return hook_fn


def make_multihead_z_zero_hook(heads: list[int]):
    heads = list(sorted(set(heads)))

    def hook_fn(value: torch.Tensor, hook):  # noqa: ANN001
        out = value.clone()
        out[:, -1, heads, :] = 0
        return out

    return hook_fn


def aggregate_head_metric(
    weights: torch.Tensor,
    sample,
    *,
    side: str,
) -> dict[str, float]:
    masks = sample.clean_region_masks if side == "clean" else sample.corrupt_region_masks
    out = {
        f"{region}_attn": float(weights[masks[region]].sum().item())
        for region in REGIONS
    }
    out["non_last_attn"] = float(weights[:-1].sum().item())
    out["self_attn"] = float(weights[-1].item())
    return out


def select_candidates(
    head_rows: list[dict[str, object]],
    *,
    threshold: float,
    top_k: int,
) -> list[tuple[int, int]]:
    filtered = [
        row
        for row in head_rows
        if float(row["delta_non_last_patch_vs_corrupt"]) >= threshold
    ]
    pool = filtered if filtered else sorted(head_rows, key=lambda row: float(row["delta_non_last_patch_vs_corrupt"]), reverse=True)
    top = pool[:top_k]
    return [(int(row["layer"]), int(row["head"])) for row in top]


def main() -> None:
    args = parse_args()
    ensure_dir(args.output_root)
    set_seed(args.seed)

    model, tokenizer = load_hooked_qwen3(str(args.model_path), device=args.device, dtype=torch.bfloat16)
    tool_token_ids = tokenizer.encode(TOOL_CALL_STR, add_special_tokens=False)
    if len(tool_token_ids) != 1:
        raise ValueError(f"{TOOL_CALL_STR!r} maps to unexpected token ids: {tool_token_ids}")
    tool_token_id = int(tool_token_ids[0])

    samples = load_samples(args.dataset_root, model, tokenizer, max_pairs=args.max_pairs)
    pair_batches = build_pair_batches(samples, args.batch_size)
    resid_hook_name = f"blocks.{L24_LAYER}.hook_resid_pre"
    pattern_hook_names = {layer: f"blocks.{layer}.attn.hook_pattern" for layer in args.candidate_layers}

    head_metrics: dict[tuple[int, int, str], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    baseline_corrupt_top1 = torch.empty(len(samples), dtype=torch.long)
    baseline_all_except_top1 = torch.empty(len(samples), dtype=torch.long)
    baseline_all_except_logit = torch.empty(len(samples), dtype=torch.float32)
    clean_resid_full_by_sample: list[torch.Tensor | None] = [None for _ in samples]

    progress = tqdm(pair_batches, desc="Exp D scan heads", dynamic_ncols=True)
    for batch in progress:
        clean_tokens = batch.clean_tokens_cpu.to(model.W_U.device)
        corrupt_tokens = batch.corrupt_tokens_cpu.to(model.W_U.device)
        clean_capture: dict[str, torch.Tensor] = {}
        corrupt_capture: dict[str, torch.Tensor] = {}
        patched_capture: dict[str, torch.Tensor] = {}
        clean_hooks = [(resid_hook_name, make_resid_full_capture(clean_capture, "resid"))]
        clean_hooks.extend(
            (pattern_hook_names[layer], make_pattern_all_heads_lastrow_capture(clean_capture, f"pattern_{layer}"))
            for layer in args.candidate_layers
        )
        corrupt_hooks = [
            (pattern_hook_names[layer], make_pattern_all_heads_lastrow_capture(corrupt_capture, f"pattern_{layer}"))
            for layer in args.candidate_layers
        ]
        with torch.no_grad():
            _ = model.run_with_hooks(clean_tokens, fwd_hooks=clean_hooks)
            corrupt_logits = model.run_with_hooks(corrupt_tokens, fwd_hooks=corrupt_hooks)
        corrupt_tool_logit, corrupt_top1 = tool_stats(corrupt_logits, tool_token_id)
        baseline_corrupt_top1[batch.indices] = corrupt_top1

        masks = [
            torch.cat([torch.ones(batch.token_len - 1, dtype=torch.bool), torch.zeros(1, dtype=torch.bool)])
            for _ in batch.indices
        ]
        patched_hooks = [(resid_hook_name, make_resid_position_replace_hook(clean_capture["resid"], masks))]
        patched_hooks.extend(
            (pattern_hook_names[layer], make_pattern_all_heads_lastrow_capture(patched_capture, f"pattern_{layer}"))
            for layer in args.candidate_layers
        )
        with torch.no_grad():
            patched_logits = model.run_with_hooks(corrupt_tokens, fwd_hooks=patched_hooks)
        patched_tool_logit, patched_top1 = tool_stats(patched_logits, tool_token_id)
        baseline_all_except_top1[batch.indices] = patched_top1
        baseline_all_except_logit[batch.indices] = patched_tool_logit

        for local_idx, sample_idx in enumerate(batch.indices):
            sample = samples[sample_idx]
            clean_resid_full_by_sample[sample_idx] = clean_capture["resid"][local_idx].clone()
            for layer in args.candidate_layers:
                clean_lastrow = clean_capture[f"pattern_{layer}"][local_idx]
                corrupt_lastrow = corrupt_capture[f"pattern_{layer}"][local_idx]
                patched_lastrow = patched_capture[f"pattern_{layer}"][local_idx]
                for head in range(clean_lastrow.shape[0]):
                    key_clean = (layer, head, "clean")
                    key_corrupt = (layer, head, "corrupt")
                    key_patched = (layer, head, "all_except_last_patch")
                    for metric_name, metric_value in aggregate_head_metric(clean_lastrow[head], sample, side="clean").items():
                        head_metrics[key_clean][metric_name].append(metric_value)
                    for metric_name, metric_value in aggregate_head_metric(corrupt_lastrow[head], sample, side="corrupt").items():
                        head_metrics[key_corrupt][metric_name].append(metric_value)
                    for metric_name, metric_value in aggregate_head_metric(patched_lastrow[head], sample, side="corrupt").items():
                        head_metrics[key_patched][metric_name].append(metric_value)
        clear_cuda()

    head_rows: list[dict[str, object]] = []
    for layer in args.candidate_layers:
        for head in range(int(model.cfg.n_heads)):
            clean = head_metrics[(layer, head, "clean")]
            corrupt = head_metrics[(layer, head, "corrupt")]
            patched = head_metrics[(layer, head, "all_except_last_patch")]
            if not clean:
                continue
            row = {
                "layer": layer,
                "head": head,
                "clean_verb_attn": float(np.mean(clean["verb_attn"])),
                "corrupt_verb_attn": float(np.mean(corrupt["verb_attn"])),
                "patched_verb_attn": float(np.mean(patched["verb_attn"])),
                "clean_task_desc_attn": float(np.mean(clean["task_desc_attn"])),
                "corrupt_task_desc_attn": float(np.mean(corrupt["task_desc_attn"])),
                "patched_task_desc_attn": float(np.mean(patched["task_desc_attn"])),
                "clean_non_last_attn": float(np.mean(clean["non_last_attn"])),
                "corrupt_non_last_attn": float(np.mean(corrupt["non_last_attn"])),
                "patched_non_last_attn": float(np.mean(patched["non_last_attn"])),
                "clean_self_attn": float(np.mean(clean["self_attn"])),
                "corrupt_self_attn": float(np.mean(corrupt["self_attn"])),
                "patched_self_attn": float(np.mean(patched["self_attn"])),
            }
            row["delta_non_last_clean_vs_corrupt"] = row["clean_non_last_attn"] - row["corrupt_non_last_attn"]
            row["delta_non_last_patch_vs_corrupt"] = row["patched_non_last_attn"] - row["corrupt_non_last_attn"]
            row["delta_verb_clean_vs_corrupt"] = row["clean_verb_attn"] - row["corrupt_verb_attn"]
            row["delta_verb_patch_vs_corrupt"] = row["patched_verb_attn"] - row["corrupt_verb_attn"]
            row["delta_task_desc_patch_vs_corrupt"] = row["patched_task_desc_attn"] - row["corrupt_task_desc_attn"]
            head_rows.append(row)

    head_rows.sort(key=lambda row: float(row["delta_non_last_patch_vs_corrupt"]), reverse=True)
    write_csv(args.output_root / "head_attn_change.csv", head_rows)

    candidates = select_candidates(head_rows, threshold=args.candidate_threshold, top_k=args.top_candidates)
    ablation_specs: list[tuple[str, list[tuple[int, int]]]] = [("none", [])]
    ablation_specs.extend((f"L{layer}H{head}", [(layer, head)]) for layer, head in candidates)
    if len(candidates) >= 3:
        ablation_specs.append(("joint_top3", candidates[:3]))
    if len(candidates) >= 5:
        ablation_specs.append(("joint_top5", candidates[:5]))

    ablation_rows: list[dict[str, object]] = [
        {
            "condition": "all_except_last_patch",
            "ablated_head": "none",
            "tool_call_top1_rate": float((baseline_all_except_top1 == tool_token_id).float().mean().item()),
            "flip_rate_from_corrupt": float(
                (((baseline_corrupt_top1 != tool_token_id) & (baseline_all_except_top1 == tool_token_id)).float().mean().item())
            ),
            "mean_tool_logit": float(baseline_all_except_logit.mean().item()),
            "logit_delta_vs_all_except_last": 0.0,
        }
    ]

    progress = tqdm(ablation_specs[1:], desc="Exp D ablations", dynamic_ncols=True)
    for label, targets in progress:
        top1_buffer = torch.empty(len(samples), dtype=torch.long)
        logit_buffer = torch.empty(len(samples), dtype=torch.float32)
        for batch in pair_batches:
            corrupt_tokens = batch.corrupt_tokens_cpu.to(model.W_U.device)
            clean_resid_batch = torch.stack(
                [clean_resid_full_by_sample[idx] for idx in batch.indices], dim=0  # type: ignore[list-item]
            )
            masks = [
                torch.cat([torch.ones(batch.token_len - 1, dtype=torch.bool), torch.zeros(1, dtype=torch.bool)])
                for _ in batch.indices
            ]
            hooks = [(resid_hook_name, make_resid_position_replace_hook(clean_resid_batch, masks))]
            heads_by_layer: dict[int, list[int]] = defaultdict(list)
            for layer, head in targets:
                heads_by_layer[layer].append(head)
            for layer, heads in heads_by_layer.items():
                hook_name = f"blocks.{layer}.attn.hook_z"
                if len(heads) == 1:
                    hooks.append((hook_name, make_z_zero_hook(heads[0])))
                else:
                    hooks.append((hook_name, make_multihead_z_zero_hook(heads)))
            with torch.no_grad():
                logits = model.run_with_hooks(corrupt_tokens, fwd_hooks=hooks)
            tool_logit, top1 = tool_stats(logits, tool_token_id)
            top1_buffer[batch.indices] = top1
            logit_buffer[batch.indices] = tool_logit
            clear_cuda()
        ablation_rows.append(
            {
                "condition": "all_except_last_patch_plus_ablation",
                "ablated_head": label,
                "tool_call_top1_rate": float((top1_buffer == tool_token_id).float().mean().item()),
                "flip_rate_from_corrupt": float(
                    (((baseline_corrupt_top1 != tool_token_id) & (top1_buffer == tool_token_id)).float().mean().item())
                ),
                "mean_tool_logit": float(logit_buffer.mean().item()),
                "logit_delta_vs_all_except_last": float((logit_buffer - baseline_all_except_logit).mean().item()),
            }
        )

    write_csv(args.output_root / "ablation_results.csv", ablation_rows)

    candidate_lines = [
        f"- `L{int(row['layer'])}H{int(row['head'])}`: patched non-last delta `{float(row['delta_non_last_patch_vs_corrupt']):+.4f}`, "
        f"patched verb delta `{float(row['delta_verb_patch_vs_corrupt']):+.4f}`, patched task-desc delta `{float(row['delta_task_desc_patch_vs_corrupt']):+.4f}`"
        for row in head_rows[: min(8, len(head_rows))]
    ]
    best_drop = min(
        (row for row in ablation_rows if row["ablated_head"] != "none"),
        key=lambda row: float(row["tool_call_top1_rate"]),
        default=None,
    )
    baseline_row = ablation_rows[0]
    lines = [
        "# Experiment D: Cross-Position Path",
        "",
        f"- 模型: `{args.model_path}`",
        f"- 样本: `datasets/test` 前 `{len(samples)}` 对",
        f"- 目标条件: `all_except_last_patch` at `L{L24_LAYER}`",
        "",
        "## Candidate Heads",
        "",
        *candidate_lines,
        "",
        "## Ablation Summary",
        "",
        f"- baseline `all_except_last_patch`: top1 `{float(baseline_row['tool_call_top1_rate']):.2%}`, strict flip `{float(baseline_row['flip_rate_from_corrupt']):.2%}`, mean logit `{float(baseline_row['mean_tool_logit']):.4f}`",
    ]
    if best_drop is None:
        lines.append("- 没有可报告的候选 head ablation 结果。")
    else:
        baseline_top1 = float(baseline_row["tool_call_top1_rate"])
        best_top1 = float(best_drop["tool_call_top1_rate"])
        drop = baseline_top1 - best_top1
        lines.append(
            f"- best ablation `{best_drop['ablated_head']}`: top1 `{best_top1:.2%}`, "
            f"drop `{drop:.2%}`, logit delta `{float(best_drop['logit_delta_vs_all_except_last']):+.4f}`"
        )
        if drop < 0.05:
            lines.append("- 结论：没有发现单一强中介 head；cross-position channel 仍更像分布式 open question。")
        else:
            lines.append(f"- 结论：`{best_drop['ablated_head']}` 是当前最强的 cross-position candidate。")
    write_text(args.output_root / "summary.md", "\n".join(lines))

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
