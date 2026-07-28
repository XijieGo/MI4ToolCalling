#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import torch
from tqdm.auto import tqdm

from task_attention_path_analysis import (
    ATTN_LAYER,
    DATASET_ROOT,
    MODEL_PATH,
    REGIONS,
    TARGET_HEAD,
    Sample,
    build_pair_batches,
    clear_cuda,
    ensure_dir,
    load_samples,
    paired_t_pvalue,
    p_to_stars,
    resolve_pattern_head_idx,
    set_seed,
    write_csv,
    write_json,
    write_text,
)

import sys

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from toolcall_circuit.single_sample import load_hooked_qwen3  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SEED = 42
TOOL_CALL_STR = "<tool_call>"
QUERY_SHIFT_OUTPUT_ROOT = PROJECT_ROOT / "results" / "8b_main" / "query_shift_source"
QK_OUTPUT_ROOT = PROJECT_ROOT / "results" / "8b_main" / "l29_qk_analysis"
SUFFICIENT_OUTPUT_ROOT = PROJECT_ROOT / "results" / "8b_main" / "sufficient_condition"
PATCH_LAYERS = (1, 5, 10, 15, 20, 21, 24, 25, 26, 27, 28)
QK_PAIRS = 50
QUERY_RECOVERY_THRESHOLD = 0.8
ROUTE_WRITER_BRIDGE = (
    (28, 3),
    (29, 9),
    (29, 11),
    (32, 3),
    (33, 29),
    (34, 1),
)


def tool_stats(logits: torch.Tensor, tool_token_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    last_logits = logits[:, -1, :]
    return (
        last_logits[:, tool_token_id].detach().cpu().float(),
        last_logits.argmax(dim=-1).detach().cpu(),
    )


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float()
    b = b.float()
    denom = float(a.norm().item() * b.norm().item())
    if denom == 0.0:
        return float("nan")
    return float(torch.dot(a, b).item() / denom)


def make_pattern_capture(capture: dict[str, torch.Tensor], key: str):
    def hook_fn(value: torch.Tensor, hook):  # noqa: ANN001
        capture[key] = value.detach().cpu().float()
        return value

    return hook_fn


def make_q_capture(capture: dict[str, torch.Tensor], key: str):
    def hook_fn(value: torch.Tensor, hook):  # noqa: ANN001
        capture[key] = value.detach().cpu().float()
        return value

    return hook_fn


def make_k_capture(capture: dict[str, torch.Tensor], key: str):
    def hook_fn(value: torch.Tensor, hook):  # noqa: ANN001
        capture[key] = value.detach().cpu().float()
        return value

    return hook_fn


def make_last_token_capture(capture: dict[int, torch.Tensor], layer: int):
    def hook_fn(value: torch.Tensor, hook):  # noqa: ANN001
        capture[layer] = value[:, -1, :].detach().cpu().float()
        return value

    return hook_fn


def make_last_token_patch_hook(source_last_cpu: torch.Tensor):
    def hook_fn(value: torch.Tensor, hook):  # noqa: ANN001
        out = value.clone()
        src = source_last_cpu.to(device=value.device, dtype=value.dtype)
        out[:, -1, :] = src
        return out

    return hook_fn


def make_head_patch_hook(head_indices: Sequence[int], source_cpu: torch.Tensor):
    head_indices = list(head_indices)

    def hook_fn(value: torch.Tensor, hook):  # noqa: ANN001
        out = value.clone()
        src = source_cpu.to(device=value.device, dtype=value.dtype)
        out[:, :, head_indices, :] = src[:, :, head_indices, :]
        return out

    return hook_fn


def aggregate_h9_attn(
    pattern_batch: torch.Tensor,
    samples: Sequence[Sample],
    batch_indices: Sequence[int],
    *,
    condition: str,
    model_n_heads: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    pattern_head = resolve_pattern_head_idx(TARGET_HEAD, int(pattern_batch.shape[1]), model_n_heads)
    for local_idx, sample_idx in enumerate(batch_indices):
        sample = samples[sample_idx]
        region_masks = sample.clean_region_masks if condition == "clean" else sample.corrupt_region_masks
        weights = pattern_batch[local_idx, pattern_head, -1, :]
        row: dict[str, object] = {
            "sample_id": sample.sample_id,
            "condition": condition,
        }
        for region in REGIONS:
            row[f"{region}_attn"] = float(weights[region_masks[region]].sum().item())
        rows.append(row)
    return rows


def run_query_shift_sweep(
    model,
    samples: Sequence[Sample],
    *,
    batch_size: int,
    tool_token_id: int,
    output_root: Path,
    recovery_threshold: float,
) -> dict[str, object]:
    ensure_dir(output_root)
    pair_batches = build_pair_batches(samples, batch_size)
    pattern_hook_name = f"blocks.{ATTN_LAYER}.attn.hook_pattern"
    model_n_heads = int(model.cfg.n_heads)

    baseline_clean_top1 = torch.empty(len(samples), dtype=torch.long)
    baseline_corrupt_top1 = torch.empty(len(samples), dtype=torch.long)
    baseline_clean_rows: list[dict[str, object]] = []
    baseline_corrupt_rows: list[dict[str, object]] = []
    patched_rows: list[dict[str, object]] = []

    summary_accum: dict[object, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    progress = tqdm(pair_batches, desc="Experiment C query-shift sweep", dynamic_ncols=True)
    for batch in progress:
        clean_tokens = batch.clean_tokens_cpu.to(model.W_U.device)
        corrupt_tokens = batch.corrupt_tokens_cpu.to(model.W_U.device)

        clean_capture: dict[str, torch.Tensor] = {}
        clean_resid: dict[int, torch.Tensor] = {}
        clean_hooks = [(pattern_hook_name, make_pattern_capture(clean_capture, "pattern"))]
        clean_hooks.extend((f"blocks.{layer}.hook_resid_pre", make_last_token_capture(clean_resid, layer)) for layer in PATCH_LAYERS)
        with torch.no_grad():
            clean_logits = model.run_with_hooks(clean_tokens, fwd_hooks=clean_hooks)
        clean_tool_logit, clean_top1 = tool_stats(clean_logits, tool_token_id)
        baseline_clean_top1[batch.indices] = clean_top1
        clean_pattern = clean_capture["pattern"]

        corrupt_capture: dict[str, torch.Tensor] = {}
        with torch.no_grad():
            corrupt_logits = model.run_with_hooks(
                corrupt_tokens,
                fwd_hooks=[(pattern_hook_name, make_pattern_capture(corrupt_capture, "pattern"))],
            )
        corrupt_tool_logit, corrupt_top1 = tool_stats(corrupt_logits, tool_token_id)
        baseline_corrupt_top1[batch.indices] = corrupt_top1
        corrupt_pattern = corrupt_capture["pattern"]

        clean_rows = aggregate_h9_attn(clean_pattern, samples, batch.indices, condition="clean", model_n_heads=model_n_heads)
        corrupt_rows = aggregate_h9_attn(corrupt_pattern, samples, batch.indices, condition="corrupt", model_n_heads=model_n_heads)

        for local_idx, sample_idx in enumerate(batch.indices):
            clean_row = clean_rows[local_idx]
            corrupt_row = corrupt_rows[local_idx]
            clean_row["patch_layer"] = "baseline_clean"
            clean_row["tool_logit"] = float(clean_tool_logit[local_idx].item())
            clean_row["is_tool_call_top1"] = int(clean_top1[local_idx].item() == tool_token_id)
            clean_row["flip_from_corrupt"] = 0
            baseline_clean_rows.append(clean_row)
            for region in ("system", "special", "verb"):
                summary_accum["baseline_clean"][region].append(float(clean_row[f"{region}_attn"]))
            summary_accum["baseline_clean"]["tool_logit"].append(float(clean_row["tool_logit"]))
            summary_accum["baseline_clean"]["top1"].append(float(clean_row["is_tool_call_top1"]))

            corrupt_row["patch_layer"] = "baseline_corrupt"
            corrupt_row["tool_logit"] = float(corrupt_tool_logit[local_idx].item())
            corrupt_row["is_tool_call_top1"] = int(corrupt_top1[local_idx].item() == tool_token_id)
            corrupt_row["flip_from_corrupt"] = 0
            baseline_corrupt_rows.append(corrupt_row)
            for region in ("system", "special", "verb"):
                summary_accum["baseline_corrupt"][region].append(float(corrupt_row[f"{region}_attn"]))
            summary_accum["baseline_corrupt"]["tool_logit"].append(float(corrupt_row["tool_logit"]))
            summary_accum["baseline_corrupt"]["top1"].append(float(corrupt_row["is_tool_call_top1"]))

        for patch_layer in PATCH_LAYERS:
            patch_capture: dict[str, torch.Tensor] = {}
            resid_hook_name = f"blocks.{patch_layer}.hook_resid_pre"
            hooks = [
                (resid_hook_name, make_last_token_patch_hook(clean_resid[patch_layer])),
                (pattern_hook_name, make_pattern_capture(patch_capture, "pattern")),
            ]
            with torch.no_grad():
                patched_logits = model.run_with_hooks(corrupt_tokens, fwd_hooks=hooks)
            patched_tool_logit, patched_top1 = tool_stats(patched_logits, tool_token_id)
            patch_pattern = patch_capture["pattern"]
            patch_attn_rows = aggregate_h9_attn(
                patch_pattern,
                samples,
                batch.indices,
                condition="corrupt",
                model_n_heads=model_n_heads,
            )

            for local_idx, sample_idx in enumerate(batch.indices):
                row = patch_attn_rows[local_idx]
                row["patch_layer"] = patch_layer
                row["tool_logit"] = float(patched_tool_logit[local_idx].item())
                row["is_tool_call_top1"] = int(patched_top1[local_idx].item() == tool_token_id)
                row["flip_from_corrupt"] = int(
                    int(corrupt_top1[local_idx].item()) != tool_token_id and int(patched_top1[local_idx].item()) == tool_token_id
                )
                patched_rows.append(row)
                for region in ("system", "special", "verb"):
                    summary_accum[patch_layer][region].append(float(row[f"{region}_attn"]))
                summary_accum[patch_layer]["tool_logit"].append(float(row["tool_logit"]))
                summary_accum[patch_layer]["top1"].append(float(row["is_tool_call_top1"]))
                summary_accum[patch_layer]["flip"].append(float(row["flip_from_corrupt"]))

        del clean_tokens, corrupt_tokens, clean_logits, corrupt_logits, clean_pattern, corrupt_pattern
        clear_cuda()

    per_sample_rows = baseline_clean_rows + baseline_corrupt_rows + patched_rows
    write_csv(output_root / "patch_sweep_per_sample.csv", per_sample_rows)

    summary_rows: list[dict[str, object]] = []
    for patch_layer in ["baseline_clean", "baseline_corrupt", *PATCH_LAYERS]:
        entry = summary_accum[patch_layer]
        top1 = np.asarray(entry["top1"], dtype=np.float64)
        tool_logit = np.asarray(entry["tool_logit"], dtype=np.float64)
        system = np.asarray(entry["system"], dtype=np.float64)
        special = np.asarray(entry["special"], dtype=np.float64)
        verb = np.asarray(entry["verb"], dtype=np.float64)
        row = {
            "patch_layer": patch_layer,
            "mean_system_attn_h9": float(system.mean()),
            "std_system_attn_h9": float(system.std(ddof=1)),
            "mean_special_attn_h9": float(special.mean()),
            "std_special_attn_h9": float(special.std(ddof=1)),
            "mean_verb_attn_h9": float(verb.mean()),
            "std_verb_attn_h9": float(verb.std(ddof=1)),
            "mean_tool_logit": float(tool_logit.mean()),
            "std_tool_logit": float(tool_logit.std(ddof=1)),
            "tool_call_top1_rate": float(top1.mean()),
            "n_pairs": int(len(system)),
        }
        if patch_layer in {"baseline_clean", "baseline_corrupt"}:
            row["tool_call_flip_rate"] = 0.0
        else:
            flips = np.asarray(entry["flip"], dtype=np.float64)
            row["tool_call_flip_rate"] = float(flips.mean())
        summary_rows.append(row)

    write_csv(output_root / "patch_sweep.csv", summary_rows)

    patched_numeric_rows = [row for row in summary_rows if isinstance(row["patch_layer"], int)]
    recovered = [row for row in patched_numeric_rows if float(row["mean_system_attn_h9"]) >= recovery_threshold]
    if recovered:
        key_layer = int(sorted(recovered, key=lambda row: int(row["patch_layer"]))[0]["patch_layer"])
        key_reason = f"首个使 mean system attention 恢复到 >= {recovery_threshold:.2f} 的层"
    else:
        best_row = max(patched_numeric_rows, key=lambda row: float(row["mean_system_attn_h9"]))
        key_layer = int(best_row["patch_layer"])
        key_reason = "没有层达到阈值，取 mean system attention 最大的层"

    baseline_clean_row = next(row for row in summary_rows if row["patch_layer"] == "baseline_clean")
    baseline_corrupt_row = next(row for row in summary_rows if row["patch_layer"] == "baseline_corrupt")
    key_row = next(row for row in summary_rows if row["patch_layer"] == key_layer)
    lines = [
        "# Query Shift Source Sweep",
        "",
        f"- 样本: `datasets/test` 前 {len(samples)} 对",
        f"- patch 位置: `blocks.L.hook_resid_pre` 的 prediction position（最后一个 token）",
        f"- 关键层 L*: `{key_layer}`",
        f"- 选择规则: {key_reason}",
        "",
        "## Baselines",
        "",
        f"- clean: system attn `{float(baseline_clean_row['mean_system_attn_h9']):.4f}`, tool logit `{float(baseline_clean_row['mean_tool_logit']):.4f}`, top1 `{float(baseline_clean_row['tool_call_top1_rate']):.2%}`",
        f"- corrupt: system attn `{float(baseline_corrupt_row['mean_system_attn_h9']):.4f}`, tool logit `{float(baseline_corrupt_row['mean_tool_logit']):.4f}`, top1 `{float(baseline_corrupt_row['tool_call_top1_rate']):.2%}`",
        "",
        "## Sweep",
        "",
        "| patch_layer | system attn | special attn | verb attn | tool logit | flip rate |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in patched_numeric_rows:
        lines.append(
            f"| {row['patch_layer']} | {float(row['mean_system_attn_h9']):.4f} | "
            f"{float(row['mean_special_attn_h9']):.4f} | {float(row['mean_verb_attn_h9']):.4f} | "
            f"{float(row['mean_tool_logit']):.4f} | {float(row['tool_call_flip_rate']):.2%} |"
        )
    lines.extend(
        [
            "",
            "## 结论",
            "",
            f"- L* = `{key_layer}`，对应的 mean system attention 为 `{float(key_row['mean_system_attn_h9']):.4f}`，相对 corrupt baseline 回升 `{float(key_row['mean_system_attn_h9']) - float(baseline_corrupt_row['mean_system_attn_h9']):+.4f}`。",
            f"- 该层的 tool logit 为 `{float(key_row['mean_tool_logit']):.4f}`，flip rate 为 `{float(key_row['tool_call_flip_rate']):.2%}`。",
        ]
    )
    write_text(output_root / "summary.md", "\n".join(lines))

    return {
        "key_layer": key_layer,
        "key_reason": key_reason,
        "baseline_clean_system_attn": float(baseline_clean_row["mean_system_attn_h9"]),
        "baseline_corrupt_system_attn": float(baseline_corrupt_row["mean_system_attn_h9"]),
    }


def run_qk_analysis(
    model,
    samples: Sequence[Sample],
    *,
    batch_size: int,
    output_root: Path,
) -> dict[str, object]:
    ensure_dir(output_root)
    subset = list(samples)
    pair_batches = build_pair_batches(subset, batch_size)
    q_hook_name = f"blocks.{ATTN_LAYER}.attn.hook_q"
    k_hook_name = f"blocks.{ATTN_LAYER}.attn.hook_k"
    d_head = int(model.cfg.d_head)
    q_scale = math.sqrt(d_head)

    region_rows: list[dict[str, object]] = []
    similarity_rows: list[dict[str, object]] = []

    progress = tqdm(pair_batches, desc="Experiment D QK analysis", dynamic_ncols=True)
    for batch in progress:
        clean_capture: dict[str, torch.Tensor] = {}
        corrupt_capture: dict[str, torch.Tensor] = {}
        clean_tokens = batch.clean_tokens_cpu.to(model.W_U.device)
        corrupt_tokens = batch.corrupt_tokens_cpu.to(model.W_U.device)
        with torch.no_grad():
            _ = model.run_with_hooks(
                clean_tokens,
                fwd_hooks=[
                    (q_hook_name, make_q_capture(clean_capture, "q")),
                    (k_hook_name, make_k_capture(clean_capture, "k")),
                ],
            )
            _ = model.run_with_hooks(
                corrupt_tokens,
                fwd_hooks=[
                    (q_hook_name, make_q_capture(corrupt_capture, "q")),
                    (k_hook_name, make_k_capture(corrupt_capture, "k")),
                ],
            )
        clean_q = clean_capture["q"]
        clean_k = clean_capture["k"]
        corrupt_q = corrupt_capture["q"]
        corrupt_k = corrupt_capture["k"]

        q_heads = int(clean_q.shape[2])
        kv_heads = int(clean_k.shape[2])
        group = max(1, q_heads // kv_heads)
        kv_index = TARGET_HEAD // group

        for local_idx, sample_idx in enumerate(batch.indices):
            sample = subset[sample_idx]
            clean_query = clean_q[local_idx, -1, TARGET_HEAD, :]
            corrupt_query = corrupt_q[local_idx, -1, TARGET_HEAD, :]
            clean_keys = clean_k[local_idx, :, kv_index, :]
            corrupt_keys = corrupt_k[local_idx, :, kv_index, :]

            clean_scores = torch.matmul(clean_keys, clean_query) / q_scale
            corrupt_scores = torch.matmul(corrupt_keys, corrupt_query) / q_scale

            for region in REGIONS:
                clean_mask = sample.clean_region_masks[region]
                corrupt_mask = sample.corrupt_region_masks[region]
                if int(clean_mask.sum().item()) > 0:
                    clean_score = float(clean_scores[clean_mask].mean().item())
                    clean_key_mean = clean_keys[clean_mask].mean(dim=0)
                else:
                    clean_score = float("nan")
                    clean_key_mean = None
                if int(corrupt_mask.sum().item()) > 0:
                    corrupt_score = float(corrupt_scores[corrupt_mask].mean().item())
                    corrupt_key_mean = corrupt_keys[corrupt_mask].mean(dim=0)
                else:
                    corrupt_score = float("nan")
                    corrupt_key_mean = None
                region_rows.append(
                    {
                        "sample_id": sample.sample_id,
                        "region": region,
                        "clean_score": clean_score,
                        "corrupt_score": corrupt_score,
                    }
                )
                if region == "system" and clean_key_mean is not None and corrupt_key_mean is not None:
                    similarity_rows.append(
                        {
                            "sample_id": sample.sample_id,
                            "metric": "clean_query_vs_corrupt_query",
                            "value": cosine_similarity(clean_query, corrupt_query),
                        }
                    )
                    similarity_rows.append(
                        {
                            "sample_id": sample.sample_id,
                            "metric": "clean_query_vs_clean_system_key",
                            "value": cosine_similarity(clean_query, clean_key_mean),
                        }
                    )
                    similarity_rows.append(
                        {
                            "sample_id": sample.sample_id,
                            "metric": "corrupt_query_vs_corrupt_system_key",
                            "value": cosine_similarity(corrupt_query, corrupt_key_mean),
                        }
                    )
                    similarity_rows.append(
                        {
                            "sample_id": sample.sample_id,
                            "metric": "clean_system_key_vs_corrupt_system_key",
                            "value": cosine_similarity(clean_key_mean, corrupt_key_mean),
                        }
                    )
                if region == "special" and clean_key_mean is not None and corrupt_key_mean is not None:
                    similarity_rows.append(
                        {
                            "sample_id": sample.sample_id,
                            "metric": "clean_query_vs_clean_special_key",
                            "value": cosine_similarity(clean_query, clean_key_mean),
                        }
                    )
                    similarity_rows.append(
                        {
                            "sample_id": sample.sample_id,
                            "metric": "corrupt_query_vs_corrupt_special_key",
                            "value": cosine_similarity(corrupt_query, corrupt_key_mean),
                        }
                    )
                    similarity_rows.append(
                        {
                            "sample_id": sample.sample_id,
                            "metric": "clean_special_key_vs_corrupt_special_key",
                            "value": cosine_similarity(clean_key_mean, corrupt_key_mean),
                        }
                    )

        del clean_tokens, corrupt_tokens, clean_q, clean_k, corrupt_q, corrupt_k
        clear_cuda()

    write_csv(output_root / "qk_scores_per_sample.csv", region_rows)
    write_csv(output_root / "query_similarity_per_sample.csv", similarity_rows)

    summary_region_rows: list[dict[str, object]] = []
    for region in REGIONS:
        clean_vals = np.asarray([float(row["clean_score"]) for row in region_rows if row["region"] == region], dtype=np.float64)
        corrupt_vals = np.asarray([float(row["corrupt_score"]) for row in region_rows if row["region"] == region], dtype=np.float64)
        p_value = paired_t_pvalue(clean_vals, corrupt_vals)
        summary_region_rows.append(
            {
                "region": region,
                "clean_mean_score": float(np.nanmean(clean_vals)),
                "clean_std_score": float(np.nanstd(clean_vals, ddof=1)),
                "corrupt_mean_score": float(np.nanmean(corrupt_vals)),
                "corrupt_std_score": float(np.nanstd(corrupt_vals, ddof=1)),
                "delta": float(np.nanmean(corrupt_vals) - np.nanmean(clean_vals)),
                "p_value": p_value,
                "significance": p_to_stars(p_value),
            }
        )
    write_csv(output_root / "qk_scores_by_region.csv", summary_region_rows)

    similarity_summary_rows: list[dict[str, object]] = []
    metrics = sorted({str(row["metric"]) for row in similarity_rows})
    for metric in metrics:
        vals = np.asarray([float(row["value"]) for row in similarity_rows if row["metric"] == metric], dtype=np.float64)
        similarity_summary_rows.append(
            {
                "metric": metric,
                "mean": float(np.nanmean(vals)),
                "std": float(np.nanstd(vals, ddof=1)),
                "n_samples": int(np.isfinite(vals).sum()),
            }
        )
    write_csv(output_root / "query_similarity.csv", similarity_summary_rows)

    system_row = next(row for row in summary_region_rows if row["region"] == "system")
    special_row = next(row for row in summary_region_rows if row["region"] == "special")
    sim_lookup = {str(row["metric"]): row for row in similarity_summary_rows}
    lines = [
        "# L29H9 QK Analysis",
        "",
        f"- 样本: `datasets/test` 前 {len(subset)} 对",
        f"- 使用 `hook_q` / `hook_k`（post-projection）直接计算 raw QK score",
        "",
        "## Region Scores",
        "",
        "| region | clean | corrupt | delta | p |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in summary_region_rows:
        lines.append(
            f"| {row['region']} | {float(row['clean_mean_score']):.4f} | {float(row['corrupt_mean_score']):.4f} | "
            f"{float(row['delta']):+.4f} | {float(row['p_value']):.3g} {row['significance']} |"
        )
    lines.extend(
        [
            "",
            "## Cosine Similarities",
            "",
        ]
    )
    for row in similarity_summary_rows:
        lines.append(f"- {row['metric']}: {float(row['mean']):.4f} ± {float(row['std']):.4f}")
    lines.extend(
        [
            "",
            "## 结论",
            "",
            f"- system region 的 raw QK score 从 `{float(system_row['clean_mean_score']):.4f}` 变到 `{float(system_row['corrupt_mean_score']):.4f}`；special region 从 `{float(special_row['clean_mean_score']):.4f}` 变到 `{float(special_row['corrupt_mean_score']):.4f}`。",
            f"- clean/corrupt query cosine 为 `{float(sim_lookup['clean_query_vs_corrupt_query']['mean']):.4f}`；clean system key vs corrupt system key cosine 为 `{float(sim_lookup['clean_system_key_vs_corrupt_system_key']['mean']):.4f}`。",
            "- 如果 system key 余弦接近 1，而 query 余弦明显更低，就支持“主要是 query 偏移，不是 system key 自身变化”。",
        ]
    )
    write_text(output_root / "summary.md", "\n".join(lines))

    return {
        "n_pairs": len(subset),
        "system_qk_clean": float(system_row["clean_mean_score"]),
        "system_qk_corrupt": float(system_row["corrupt_mean_score"]),
        "clean_query_vs_corrupt_query": float(sim_lookup["clean_query_vs_corrupt_query"]["mean"]),
    }


def run_sufficient_condition(
    model,
    samples: Sequence[Sample],
    *,
    batch_size: int,
    tool_token_id: int,
    output_root: Path,
    key_layer: int,
) -> dict[str, object]:
    ensure_dir(output_root)
    pair_batches = build_pair_batches(samples, batch_size)
    route_heads_by_layer: dict[int, list[int]] = defaultdict(list)
    for layer, head in ROUTE_WRITER_BRIDGE:
        route_heads_by_layer[layer].append(head)

    required_resid_layers = sorted({25, key_layer})
    baseline_corrupt_logit = torch.empty(len(samples), dtype=torch.float32)
    baseline_corrupt_top1 = torch.empty(len(samples), dtype=torch.long)
    deltas_by_condition: dict[str, list[float]] = defaultdict(list)
    flips_by_condition: dict[str, list[int]] = defaultdict(list)
    top1_by_condition: dict[str, list[int]] = defaultdict(list)

    condition_aliases: list[tuple[str, tuple[bool, int | None]]] = [
        ("baseline", (False, None)),
        ("heads_only", (True, None)),
        ("resid_only_L25", (False, 25)),
        ("heads_plus_resid", (True, 25)),
        (f"resid_only_L{key_layer}", (False, key_layer)),
        (f"heads_plus_resid_L{key_layer}", (True, key_layer)),
    ]

    progress = tqdm(pair_batches, desc="Experiment E sufficient condition", dynamic_ncols=True)
    for batch in progress:
        clean_tokens = batch.clean_tokens_cpu.to(model.W_U.device)
        corrupt_tokens = batch.corrupt_tokens_cpu.to(model.W_U.device)

        clean_capture: dict[str, torch.Tensor] = {}
        clean_hooks = []
        for layer in sorted(route_heads_by_layer):
            hook_name = f"blocks.{layer}.attn.hook_z"
            clean_hooks.append((hook_name, make_pattern_capture(clean_capture, hook_name)))
        for layer in required_resid_layers:
            hook_name = f"blocks.{layer}.hook_resid_pre"
            clean_hooks.append((hook_name, make_pattern_capture(clean_capture, hook_name)))
        with torch.no_grad():
            _ = model.run_with_hooks(clean_tokens, fwd_hooks=clean_hooks)
            corrupt_logits = model(corrupt_tokens)
        corrupt_tool_logit, corrupt_top1 = tool_stats(corrupt_logits, tool_token_id)
        baseline_corrupt_logit[batch.indices] = corrupt_tool_logit
        baseline_corrupt_top1[batch.indices] = corrupt_top1

        local_results: dict[tuple[bool, int | None], tuple[torch.Tensor, torch.Tensor]] = {}
        for _name, config in condition_aliases:
            if config in local_results:
                continue
            use_heads, resid_layer = config
            if not use_heads and resid_layer is None:
                local_results[config] = (corrupt_tool_logit, corrupt_top1)
                continue
            hooks = []
            if use_heads:
                for layer, heads in sorted(route_heads_by_layer.items()):
                    hook_name = f"blocks.{layer}.attn.hook_z"
                    hooks.append((hook_name, make_head_patch_hook(heads, clean_capture[hook_name])))
            if resid_layer is not None:
                hook_name = f"blocks.{resid_layer}.hook_resid_pre"
                source_last = clean_capture[hook_name][:, -1, :]
                hooks.append((hook_name, make_last_token_patch_hook(source_last)))
            with torch.no_grad():
                logits = model.run_with_hooks(corrupt_tokens, fwd_hooks=hooks)
            local_results[config] = tool_stats(logits, tool_token_id)

        for name, config in condition_aliases:
            patched_logit, patched_top1 = local_results[config]
            delta = patched_logit - corrupt_tool_logit
            flips = (corrupt_top1 != tool_token_id) & (patched_top1 == tool_token_id)
            deltas_by_condition[name].extend(delta.tolist())
            flips_by_condition[name].extend(flips.int().tolist())
            top1_by_condition[name].extend((patched_top1 == tool_token_id).int().tolist())

        del clean_tokens, corrupt_tokens, corrupt_logits
        clear_cuda()

    result_rows: list[dict[str, object]] = []
    for name, _config in condition_aliases:
        deltas = np.asarray(deltas_by_condition[name], dtype=np.float64)
        flips = np.asarray(flips_by_condition[name], dtype=np.float64)
        top1 = np.asarray(top1_by_condition[name], dtype=np.float64)
        result_rows.append(
            {
                "condition": name,
                "flip_rate": float(flips.mean()),
                "tool_call_top1_rate": float(top1.mean()),
                "mean_logit_delta": float(deltas.mean()),
                "std_logit_delta": float(deltas.std(ddof=1)),
                "n_pairs": int(deltas.shape[0]),
            }
        )
    write_csv(output_root / "intervention_results.csv", result_rows)

    lookup = {str(row["condition"]): row for row in result_rows}
    heads_plus_name = "heads_plus_resid" if key_layer == 25 else f"heads_plus_resid_L{key_layer}"
    lines = [
        "# Sufficient Condition Interventions",
        "",
        f"- 样本: `datasets/test` 前 {len(samples)} 对 corrupt prompts",
        f"- route_writer_bridge: `{', '.join(f'L{layer}H{head}' for layer, head in ROUTE_WRITER_BRIDGE)}`",
        f"- 关键 residual 层 L*: `{key_layer}`",
        "",
        "| condition | flip_rate | top1_rate | mean logit delta | std |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in result_rows:
        lines.append(
            f"| {row['condition']} | {float(row['flip_rate']):.2%} | {float(row['tool_call_top1_rate']):.2%} | "
            f"{float(row['mean_logit_delta']):+.4f} | {float(row['std_logit_delta']):.4f} |"
        )
    lines.extend(
        [
            "",
            "## 结论",
            "",
            f"- `heads_only` 的 flip rate 为 `{float(lookup['heads_only']['flip_rate']):.2%}`，`resid_only_L25` 为 `{float(lookup['resid_only_L25']['flip_rate']):.2%}`。",
            f"- `heads_plus_resid`（L25）flip rate 为 `{float(lookup['heads_plus_resid']['flip_rate']):.2%}`。",
            f"- 基于实验 C 的关键层 `{key_layer}`，`{heads_plus_name}` 的 flip rate 为 `{float(lookup[heads_plus_name]['flip_rate']):.2%}`。",
        ]
    )
    write_text(output_root / "summary.md", "\n".join(lines))

    return {
        "key_layer": key_layer,
        "heads_only_flip_rate": float(lookup["heads_only"]["flip_rate"]),
        "heads_plus_resid_flip_rate": float(lookup["heads_plus_resid"]["flip_rate"]),
        f"heads_plus_resid_L{key_layer}_flip_rate": float(lookup[heads_plus_name]["flip_rate"]),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 3 query-shift supplemental experiments")
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH)
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--query-shift-output-root", type=Path, default=QUERY_SHIFT_OUTPUT_ROOT)
    parser.add_argument("--qk-output-root", type=Path, default=QK_OUTPUT_ROOT)
    parser.add_argument("--sufficient-output-root", type=Path, default=SUFFICIENT_OUTPUT_ROOT)
    parser.add_argument("--mode", choices=("all", "query_shift", "qk", "sufficient"), default="all")
    parser.add_argument("--max-pairs", type=int, default=100)
    parser.add_argument("--qk-pairs", type=int, default=QK_PAIRS)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--recovery-threshold", type=float, default=QUERY_RECOVERY_THRESHOLD)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    model, tokenizer = load_hooked_qwen3(str(args.model_path), device=args.device, dtype=torch.bfloat16)
    tool_token_ids = tokenizer.encode(TOOL_CALL_STR, add_special_tokens=False)
    if len(tool_token_ids) != 1:
        raise ValueError(f"{TOOL_CALL_STR!r} does not map to one token: {tool_token_ids}")
    tool_token_id = int(tool_token_ids[0])

    samples = load_samples(args.dataset_root, model, tokenizer, max_pairs=args.max_pairs)

    query_meta: dict[str, object] = {}
    if args.mode in {"all", "query_shift"}:
        query_meta = run_query_shift_sweep(
            model,
            samples,
            batch_size=args.batch_size,
            tool_token_id=tool_token_id,
            output_root=args.query_shift_output_root,
            recovery_threshold=args.recovery_threshold,
        )
        write_json(
            args.query_shift_output_root / "metadata.json",
            {
                "seed": args.seed,
                "model_path": str(args.model_path),
                "dataset_root": str(args.dataset_root),
                "max_pairs": args.max_pairs,
                "batch_size": args.batch_size,
                **query_meta,
            },
        )

    qk_pairs = min(args.qk_pairs, len(samples))
    qk_meta: dict[str, object] = {}
    if args.mode in {"all", "qk"}:
        qk_meta = run_qk_analysis(
            model,
            samples[:qk_pairs],
            batch_size=args.batch_size,
            output_root=args.qk_output_root,
        )
        write_json(
            args.qk_output_root / "metadata.json",
            {
                "seed": args.seed,
                "model_path": str(args.model_path),
                "dataset_root": str(args.dataset_root),
                "qk_pairs": qk_pairs,
                "batch_size": args.batch_size,
                **qk_meta,
            },
        )

    if args.mode in {"all", "sufficient"}:
        key_layer = int(query_meta.get("key_layer", 25))
        sufficient_meta = run_sufficient_condition(
            model,
            samples,
            batch_size=args.batch_size,
            tool_token_id=tool_token_id,
            output_root=args.sufficient_output_root,
            key_layer=key_layer,
        )
        write_json(
            args.sufficient_output_root / "metadata.json",
            {
                "seed": args.seed,
                "model_path": str(args.model_path),
                "dataset_root": str(args.dataset_root),
                "max_pairs": args.max_pairs,
                "batch_size": args.batch_size,
                **sufficient_meta,
            },
        )

    del model
    clear_cuda()


if __name__ == "__main__":
    main()
