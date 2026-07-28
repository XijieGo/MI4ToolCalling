#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from tqdm.auto import tqdm

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from toolcall_circuit.single_sample import load_hooked_qwen3  # noqa: E402

from task_attention_path_analysis import (  # noqa: E402
    ATTN_LAYER,
    DATASET_ROOT,
    DIFF_ROOT,
    MODEL_PATH,
    REGIONS,
    TRANSCODER_DIR,
    Sample,
    build_pair_batches,
    clear_cuda,
    ensure_dir,
    load_samples,
    locate_schema_span,
    overlap,
    p_to_stars,
    resolve_pattern_head_idx,
    set_seed,
    write_csv,
    write_json,
    write_text,
)


SEED = 42
TOOL_CALL_STR = "<tool_call>"
PHASE4_ROOT = Path(__file__).resolve().parents[2] / "results" / "8b_main" / "phase4_reviewer"

H9_LAYER = 29
H9_HEAD = 9
RANDOM_CONTROL_HEAD = 0
L24_LAYER = 24
LATE_LAYER = 33
LATE_HEAD = 29
LOW_RANK_KS = (1, 5, 10, 50)
SUPPRESSOR_LAYERS = (21, 22, 23, 24)
SUPPRESSOR_KS = (10, 25, 50, 100)


@dataclass
class SinglePromptItem:
    sample_id: str
    condition: str
    verb_condition: str
    token_len: int
    tokens_cpu: torch.Tensor
    system_mask: torch.Tensor


@dataclass
class SingleBatch:
    items: list[SinglePromptItem]
    tokens_cpu: torch.Tensor
    token_len: int


def tool_stats(logits: torch.Tensor, tool_token_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    last_logits = logits[:, -1, :]
    return (
        last_logits[:, tool_token_id].detach().cpu().float(),
        last_logits.argmax(dim=-1).detach().cpu(),
    )


def get_w_o_layer(model, layer: int) -> torch.Tensor:
    if hasattr(model, "W_O"):
        return model.W_O[layer]
    attn = model.blocks[layer].attn
    if not hasattr(attn, "W_O"):
        raise AttributeError("W_O not found on model.")
    return attn.W_O.view(int(model.cfg.n_heads), int(model.cfg.d_head), int(model.cfg.d_model))


def precompute_head_projection(model, layer: int, head: int, tool_token_id: int) -> torch.Tensor:
    wu_tool = model.W_U[:, tool_token_id].to(device=model.W_U.device, dtype=torch.float32)
    w_o = get_w_o_layer(model, layer).to(device=model.W_U.device, dtype=torch.float32)
    return torch.einsum("de,e->d", w_o[head], wu_tool)


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float()
    b = b.float()
    denom = float(a.norm().item() * b.norm().item())
    if denom == 0.0:
        return float("nan")
    return float(torch.dot(a, b).item() / denom)


def bootstrap_ci_mean(values: np.ndarray, *, n_boot: int, seed: int) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, values.size, size=(n_boot, values.size))
    means = values[idx].mean(axis=1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    return float(lo), float(hi)


def bootstrap_ci_delta(deltas: np.ndarray, *, n_boot: int, seed: int) -> tuple[float, float]:
    deltas = np.asarray(deltas, dtype=np.float64)
    if deltas.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, deltas.size, size=(n_boot, deltas.size))
    means = deltas[idx].mean(axis=1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    return float(lo), float(hi)


def paired_permutation_pvalue(deltas: np.ndarray, *, n_perm: int, seed: int) -> float:
    deltas = np.asarray(deltas, dtype=np.float64)
    if deltas.size == 0:
        return float("nan")
    observed = abs(float(deltas.mean()))
    if observed == 0.0:
        return 1.0
    rng = np.random.default_rng(seed)
    signs = rng.choice(np.array([-1.0, 1.0], dtype=np.float64), size=(n_perm, deltas.size))
    permuted = np.abs((signs * deltas[None, :]).mean(axis=1))
    p = (1.0 + float((permuted >= observed).sum())) / (n_perm + 1.0)
    return float(p)


def cohen_dz(deltas: np.ndarray) -> float:
    deltas = np.asarray(deltas, dtype=np.float64)
    if deltas.size <= 1:
        return float("nan")
    std = float(deltas.std(ddof=1))
    if std == 0.0:
        return float("nan")
    return float(deltas.mean() / std)


def make_head_q_capture(capture: dict[str, torch.Tensor], key: str, head: int):
    def hook_fn(value: torch.Tensor, hook):  # noqa: ANN001
        capture[key] = value[:, -1, head, :].detach().cpu().float()
        return value

    return hook_fn


def make_head_z_capture(capture: dict[str, torch.Tensor], key: str, head: int):
    def hook_fn(value: torch.Tensor, hook):  # noqa: ANN001
        capture[key] = value[:, -1, head, :].detach().cpu().float()
        return value

    return hook_fn


def make_pattern_lastrow_capture(capture: dict[str, torch.Tensor], key: str, head: int, model_n_heads: int):
    def hook_fn(value: torch.Tensor, hook):  # noqa: ANN001
        head_idx = resolve_pattern_head_idx(head, int(value.shape[1]), model_n_heads)
        capture[key] = value[:, head_idx, -1, :].detach().cpu().float()
        return value

    return hook_fn


def make_resid_last_capture(capture: dict[str, torch.Tensor], key: str):
    def hook_fn(value: torch.Tensor, hook):  # noqa: ANN001
        capture[key] = value[:, -1, :].detach().cpu().float()
        return value

    return hook_fn


def make_resid_full_capture(capture: dict[str, torch.Tensor], key: str):
    def hook_fn(value: torch.Tensor, hook):  # noqa: ANN001
        capture[key] = value.detach().cpu().to(torch.bfloat16)
        return value

    return hook_fn


def make_k_capture(capture: dict[str, torch.Tensor], key: str):
    def hook_fn(value: torch.Tensor, hook):  # noqa: ANN001
        capture[key] = value.detach().cpu().float()
        return value

    return hook_fn


def make_q_replace_hook(head: int, source_cpu: torch.Tensor):
    def hook_fn(value: torch.Tensor, hook):  # noqa: ANN001
        out = value.clone()
        src = source_cpu.to(device=value.device, dtype=value.dtype)
        out[:, -1, head, :] = src
        return out

    return hook_fn


def make_q_add_hook(head: int, delta_cpu: torch.Tensor):
    def hook_fn(value: torch.Tensor, hook):  # noqa: ANN001
        out = value.clone()
        delta = delta_cpu.to(device=value.device, dtype=value.dtype)
        out[:, -1, head, :] = out[:, -1, head, :] + delta
        return out

    return hook_fn


def make_pattern_replace_hook(head: int, source_cpu: torch.Tensor, model_n_heads: int):
    def hook_fn(value: torch.Tensor, hook):  # noqa: ANN001
        out = value.clone()
        src = source_cpu.to(device=value.device, dtype=value.dtype)
        head_idx = resolve_pattern_head_idx(head, int(out.shape[1]), model_n_heads)
        out[:, head_idx, -1, :] = src
        return out

    return hook_fn


def make_z_replace_hook(head: int, source_cpu: torch.Tensor):
    def hook_fn(value: torch.Tensor, hook):  # noqa: ANN001
        out = value.clone()
        src = source_cpu.to(device=value.device, dtype=value.dtype)
        out[:, -1, head, :] = src
        return out

    return hook_fn


def make_z_zero_hook(head: int):
    def hook_fn(value: torch.Tensor, hook):  # noqa: ANN001
        out = value.clone()
        out[:, -1, head, :] = 0
        return out

    return hook_fn


def make_last_token_resid_add_hook(delta_cpu: torch.Tensor):
    def hook_fn(value: torch.Tensor, hook):  # noqa: ANN001
        out = value.clone()
        delta = delta_cpu.to(device=value.device, dtype=value.dtype)
        out[:, -1, :] = out[:, -1, :] + delta
        return out

    return hook_fn


def make_resid_position_replace_hook(source_cpu: torch.Tensor, position_masks: Sequence[torch.Tensor]):
    masks = [mask.clone().bool() for mask in position_masks]

    def hook_fn(value: torch.Tensor, hook):  # noqa: ANN001
        out = value.clone()
        src = source_cpu.to(device=value.device, dtype=value.dtype)
        for batch_idx, mask_cpu in enumerate(masks):
            mask = mask_cpu.to(device=value.device)
            out[batch_idx, mask, :] = src[batch_idx, mask, :]
        return out

    return hook_fn


def region_attention_from_lastrow(lastrow: torch.Tensor, region_masks: dict[str, torch.Tensor]) -> dict[str, float]:
    out: dict[str, float] = {}
    for region in REGIONS:
        out[f"{region}_attn"] = float(lastrow[region_masks[region]].sum().item())
    return out


def append_basic_rows(
    rows: list[dict[str, object]],
    *,
    condition: str,
    samples: Sequence[Sample],
    batch_indices: Sequence[int],
    tool_logits: torch.Tensor,
    top1: torch.Tensor,
    h9_lastrow: torch.Tensor,
    side: str,
    extra_by_local: dict[str, torch.Tensor] | None = None,
) -> None:
    for local_idx, sample_idx in enumerate(batch_indices):
        sample = samples[sample_idx]
        region_masks = sample.clean_region_masks if side == "clean" else sample.corrupt_region_masks
        row: dict[str, object] = {
            "sample_id": sample.sample_id,
            "condition": condition,
            "tool_logit": float(tool_logits[local_idx].item()),
            "is_tool_call_top1": int(top1[local_idx].item()),
        }
        row.update(region_attention_from_lastrow(h9_lastrow[local_idx], region_masks))
        row["system_attn_h9"] = row["system_attn"]
        row["special_attn_h9"] = row["special_attn"]
        row["verb_attn_h9"] = row["verb_attn"]
        if extra_by_local is not None:
            for key, values in extra_by_local.items():
                row[key] = float(values[local_idx].item()) if torch.is_tensor(values) else float(values[local_idx])
        rows.append(row)


def summarize_rows_by_condition(
    rows: Sequence[dict[str, object]],
    *,
    metric_names: Sequence[str],
    baseline_condition: str | None,
    order: Sequence[str] | None,
    bootstrap_samples: int,
    seed: int,
) -> list[dict[str, object]]:
    grouped: dict[str, dict[str, dict[str, object]]] = defaultdict(dict)
    for row in rows:
        grouped[str(row["condition"])][str(row["sample_id"])] = dict(row)

    conditions = list(order) if order is not None else sorted(grouped)
    baseline_group = grouped.get(baseline_condition or "", {})
    summary_rows: list[dict[str, object]] = []

    for cond_idx, condition in enumerate(conditions):
        group = grouped.get(condition)
        if not group:
            continue
        sample_ids = sorted(group)
        top1_vals = np.asarray([float(group[sid]["is_tool_call_top1"]) for sid in sample_ids], dtype=np.float64)
        top1_lo, top1_hi = bootstrap_ci_mean(top1_vals, n_boot=bootstrap_samples, seed=seed + cond_idx * 1000 + 1)
        row: dict[str, object] = {
            "condition": condition,
            "n_samples": len(sample_ids),
            "tool_call_top1_rate": float(top1_vals.mean()),
            "tool_call_top1_rate_ci_low": top1_lo,
            "tool_call_top1_rate_ci_high": top1_hi,
        }
        for metric_idx, metric in enumerate(metric_names):
            vals = np.asarray([float(group[sid][metric]) for sid in sample_ids], dtype=np.float64)
            lo, hi = bootstrap_ci_mean(vals, n_boot=bootstrap_samples, seed=seed + cond_idx * 1000 + 100 + metric_idx)
            row[f"mean_{metric}"] = float(vals.mean())
            row[f"{metric}_ci_low"] = lo
            row[f"{metric}_ci_high"] = hi
            if baseline_condition is not None and condition != baseline_condition and baseline_group:
                common_ids = [sid for sid in sample_ids if sid in baseline_group]
                if common_ids:
                    deltas = np.asarray(
                        [float(group[sid][metric]) - float(baseline_group[sid][metric]) for sid in common_ids],
                        dtype=np.float64,
                    )
                    d_lo, d_hi = bootstrap_ci_delta(
                        deltas,
                        n_boot=bootstrap_samples,
                        seed=seed + cond_idx * 1000 + 200 + metric_idx,
                    )
                    row[f"delta_{metric}_vs_{baseline_condition}"] = float(deltas.mean())
                    row[f"delta_{metric}_vs_{baseline_condition}_ci_low"] = d_lo
                    row[f"delta_{metric}_vs_{baseline_condition}_ci_high"] = d_hi
        summary_rows.append(row)
    return summary_rows


def load_top_feature_ids(path: Path, category: str, k: int) -> list[int]:
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["category"] == category:
                rows.append(row)
    if category == "clean-selective":
        rows.sort(key=lambda row: float(row["delta"]), reverse=True)
    else:
        rows.sort(key=lambda row: float(row["delta"]))
    return [int(row["feature_idx"]) for row in rows[:k]]


def build_feature_ablation_hooks(
    *,
    tc_weights: dict[str, torch.Tensor],
    feature_ids: Sequence[int],
):
    idx = torch.tensor(list(feature_ids), dtype=torch.long)
    device_cache: dict[str, torch.Tensor] = {}
    state: dict[str, torch.Tensor] = {}

    def hooks():
        if not device_cache:
            device = tc_weights["W_enc"].device
            device_idx = idx.to(device)
            device_cache["W_enc"] = tc_weights["W_enc"][device_idx].to(dtype=torch.bfloat16)
            device_cache["b_enc"] = tc_weights["b_enc"][device_idx].to(dtype=torch.bfloat16)
            device_cache["W_dec"] = tc_weights["W_dec"][device_idx].to(dtype=torch.bfloat16)

        def in_hook(value: torch.Tensor, hook):  # noqa: ANN001
            state["mlp_in"] = value[:, -1, :].detach()
            return value

        def out_hook(value: torch.Tensor, hook):  # noqa: ANN001
            mlp_in = state.pop("mlp_in")
            acts = torch.relu(F.linear(mlp_in.to(dtype=torch.bfloat16), device_cache["W_enc"], device_cache["b_enc"]))
            contrib = acts @ device_cache["W_dec"]
            out = value.clone()
            out[:, -1, :] = out[:, -1, :] - contrib.to(dtype=out.dtype)
            return out

        return in_hook, out_hook

    return hooks


def token_span_from_char_span(
    offsets: Sequence[tuple[int, int]],
    char_span: tuple[int, int],
) -> tuple[int, int]:
    start_char, end_char = char_span
    token_indices = [
        idx
        for idx, (tok_start, tok_end) in enumerate(offsets)
        if tok_start != tok_end and overlap(start_char, end_char, tok_start, tok_end)
    ]
    if not token_indices:
        raise ValueError("Failed to map char span to token span.")
    return min(token_indices), max(token_indices) + 1


def build_single_batches(items: Sequence[SinglePromptItem], batch_size: int) -> list[SingleBatch]:
    buckets: dict[int, list[SinglePromptItem]] = defaultdict(list)
    for item in items:
        buckets[item.token_len].append(item)

    batches: list[SingleBatch] = []
    for token_len in sorted(buckets):
        group = buckets[token_len]
        for start in range(0, len(group), batch_size):
            chunk = group[start : start + batch_size]
            batches.append(
                SingleBatch(
                    items=list(chunk),
                    tokens_cpu=torch.stack([item.tokens_cpu for item in chunk], dim=0),
                    token_len=token_len,
                )
            )
    return batches


def make_length_matched_ids(
    tokenizer,
    *,
    prefix_text: str,
    suffix_text: str,
    filler_text: str,
    target_len: int,
) -> torch.Tensor:
    prefix_ids = tokenizer.encode(prefix_text, add_special_tokens=False)
    suffix_ids = tokenizer.encode(suffix_text, add_special_tokens=False)
    filler_ids = tokenizer.encode(filler_text, add_special_tokens=False)
    if not filler_ids:
        raise ValueError("filler_text must encode to at least one token.")
    needed = target_len - len(prefix_ids) - len(suffix_ids)
    if needed < 0:
        raise ValueError("Base schema template exceeds target token length.")
    repeated = (filler_ids * math.ceil(needed / len(filler_ids)))[:needed]
    out = torch.tensor(prefix_ids + repeated + suffix_ids, dtype=torch.long)
    if int(out.shape[0]) != target_len:
        raise RuntimeError("Failed to build length-matched token ids.")
    return out


def run_head_attention_qk_analysis(
    model,
    samples: Sequence[Sample],
    *,
    batch_size: int,
    layer: int,
    head: int,
    output_root: Path,
    title: str,
) -> dict[str, object]:
    ensure_dir(output_root)
    pair_batches = build_pair_batches(samples, batch_size)
    n_heads = int(model.cfg.n_heads)
    d_head = int(model.cfg.d_head)
    q_scale = math.sqrt(d_head)
    pattern_hook_name = f"blocks.{layer}.attn.hook_pattern"
    q_hook_name = f"blocks.{layer}.attn.hook_q"
    k_hook_name = f"blocks.{layer}.attn.hook_k"

    attn_rows: list[dict[str, object]] = []
    qk_rows: list[dict[str, object]] = []
    sim_rows: list[dict[str, object]] = []

    progress = tqdm(pair_batches, desc=f"{title} attention+qk", dynamic_ncols=True)
    for batch in progress:
        clean_capture: dict[str, torch.Tensor] = {}
        corrupt_capture: dict[str, torch.Tensor] = {}
        clean_tokens = batch.clean_tokens_cpu.to(model.W_U.device)
        corrupt_tokens = batch.corrupt_tokens_cpu.to(model.W_U.device)

        clean_hooks = [
            (pattern_hook_name, make_pattern_lastrow_capture(clean_capture, "pattern", head, n_heads)),
            (q_hook_name, make_q_capture(clean_capture, "q")),
            (k_hook_name, make_k_capture(clean_capture, "k")),
        ]
        corrupt_hooks = [
            (pattern_hook_name, make_pattern_lastrow_capture(corrupt_capture, "pattern", head, n_heads)),
            (q_hook_name, make_q_capture(corrupt_capture, "q")),
            (k_hook_name, make_k_capture(corrupt_capture, "k")),
        ]
        with torch.no_grad():
            _ = model.run_with_hooks(clean_tokens, fwd_hooks=clean_hooks)
            _ = model.run_with_hooks(corrupt_tokens, fwd_hooks=corrupt_hooks)

        clean_patterns = clean_capture["pattern"]
        corrupt_patterns = corrupt_capture["pattern"]
        clean_q = clean_capture["q"]
        corrupt_q = corrupt_capture["q"]
        clean_k = clean_capture["k"]
        corrupt_k = corrupt_capture["k"]
        q_heads = int(clean_q.shape[2])
        kv_heads = int(clean_k.shape[2])
        group = max(1, q_heads // kv_heads)
        kv_index = head // group

        for local_idx, sample_idx in enumerate(batch.indices):
            sample = samples[sample_idx]
            clean_region_masks = sample.clean_region_masks
            corrupt_region_masks = sample.corrupt_region_masks

            clean_lastrow = clean_patterns[local_idx]
            corrupt_lastrow = corrupt_patterns[local_idx]
            clean_row = {"sample_id": sample.sample_id, "condition": "clean"}
            corrupt_row = {"sample_id": sample.sample_id, "condition": "corrupt"}
            clean_row.update(region_attention_from_lastrow(clean_lastrow, clean_region_masks))
            corrupt_row.update(region_attention_from_lastrow(corrupt_lastrow, corrupt_region_masks))
            attn_rows.extend([clean_row, corrupt_row])

            clean_query = clean_q[local_idx, -1, head, :]
            corrupt_query = corrupt_q[local_idx, -1, head, :]
            clean_keys = clean_k[local_idx, :, kv_index, :]
            corrupt_keys = corrupt_k[local_idx, :, kv_index, :]
            clean_scores = torch.matmul(clean_keys, clean_query) / q_scale
            corrupt_scores = torch.matmul(corrupt_keys, corrupt_query) / q_scale

            for region in REGIONS:
                clean_mask = clean_region_masks[region]
                corrupt_mask = corrupt_region_masks[region]
                qk_rows.append(
                    {
                        "sample_id": sample.sample_id,
                        "region": region,
                        "clean_score": float(clean_scores[clean_mask].mean().item()) if int(clean_mask.sum().item()) else float("nan"),
                        "corrupt_score": float(corrupt_scores[corrupt_mask].mean().item()) if int(corrupt_mask.sum().item()) else float("nan"),
                    }
                )

            clean_system_key = clean_keys[clean_region_masks["system"]].mean(dim=0)
            corrupt_system_key = corrupt_keys[corrupt_region_masks["system"]].mean(dim=0)
            clean_special_key = clean_keys[clean_region_masks["special"]].mean(dim=0)
            corrupt_special_key = corrupt_keys[corrupt_region_masks["special"]].mean(dim=0)
            sim_rows.extend(
                [
                    {
                        "sample_id": sample.sample_id,
                        "metric": "clean_query_vs_corrupt_query",
                        "value": cosine_similarity(clean_query, corrupt_query),
                    },
                    {
                        "sample_id": sample.sample_id,
                        "metric": "clean_query_vs_clean_system_key",
                        "value": cosine_similarity(clean_query, clean_system_key),
                    },
                    {
                        "sample_id": sample.sample_id,
                        "metric": "corrupt_query_vs_corrupt_system_key",
                        "value": cosine_similarity(corrupt_query, corrupt_system_key),
                    },
                    {
                        "sample_id": sample.sample_id,
                        "metric": "clean_system_key_vs_corrupt_system_key",
                        "value": cosine_similarity(clean_system_key, corrupt_system_key),
                    },
                    {
                        "sample_id": sample.sample_id,
                        "metric": "clean_query_vs_clean_special_key",
                        "value": cosine_similarity(clean_query, clean_special_key),
                    },
                    {
                        "sample_id": sample.sample_id,
                        "metric": "corrupt_query_vs_corrupt_special_key",
                        "value": cosine_similarity(corrupt_query, corrupt_special_key),
                    },
                    {
                        "sample_id": sample.sample_id,
                        "metric": "clean_special_key_vs_corrupt_special_key",
                        "value": cosine_similarity(clean_special_key, corrupt_special_key),
                    },
                ]
            )

        clear_cuda()

    attn_summary_rows: list[dict[str, object]] = []
    for region in REGIONS:
        clean_vals = np.asarray(
            [float(row[f"{region}_attn"]) for row in attn_rows if row["condition"] == "clean"],
            dtype=np.float64,
        )
        corrupt_vals = np.asarray(
            [float(row[f"{region}_attn"]) for row in attn_rows if row["condition"] == "corrupt"],
            dtype=np.float64,
        )
        diff = clean_vals - corrupt_vals
        p_value = paired_permutation_pvalue(diff, n_perm=5000, seed=SEED + 7 + len(attn_summary_rows))
        attn_summary_rows.append(
            {
                "region": region,
                "clean_mean": float(clean_vals.mean()),
                "clean_std": float(clean_vals.std(ddof=1)),
                "corrupt_mean": float(corrupt_vals.mean()),
                "corrupt_std": float(corrupt_vals.std(ddof=1)),
                "delta_corrupt_minus_clean": float(corrupt_vals.mean() - clean_vals.mean()),
                "p_value": p_value,
                "significance": p_to_stars(p_value),
            }
        )

    qk_summary_rows: list[dict[str, object]] = []
    for region in REGIONS:
        clean_vals = np.asarray([float(row["clean_score"]) for row in qk_rows if row["region"] == region], dtype=np.float64)
        corrupt_vals = np.asarray([float(row["corrupt_score"]) for row in qk_rows if row["region"] == region], dtype=np.float64)
        diff = clean_vals - corrupt_vals
        p_value = paired_permutation_pvalue(diff, n_perm=5000, seed=SEED + 101 + len(qk_summary_rows))
        qk_summary_rows.append(
            {
                "region": region,
                "clean_mean_score": float(np.nanmean(clean_vals)),
                "clean_std_score": float(np.nanstd(clean_vals, ddof=1)),
                "corrupt_mean_score": float(np.nanmean(corrupt_vals)),
                "corrupt_std_score": float(np.nanstd(corrupt_vals, ddof=1)),
                "delta_corrupt_minus_clean": float(np.nanmean(corrupt_vals) - np.nanmean(clean_vals)),
                "p_value": p_value,
                "significance": p_to_stars(p_value),
            }
        )

    sim_summary_rows: list[dict[str, object]] = []
    for metric in sorted({str(row["metric"]) for row in sim_rows}):
        vals = np.asarray([float(row["value"]) for row in sim_rows if row["metric"] == metric], dtype=np.float64)
        sim_summary_rows.append(
            {
                "metric": metric,
                "mean": float(np.nanmean(vals)),
                "std": float(np.nanstd(vals, ddof=1)),
                "n_samples": int(np.isfinite(vals).sum()),
            }
        )

    write_csv(output_root / "l33h29_region_attention.csv", attn_summary_rows)
    write_csv(output_root / "l33h29_qk_scores.csv", qk_summary_rows)
    write_csv(output_root / "l33h29_qk_scores_per_sample.csv", qk_rows)
    write_csv(output_root / "l33h29_similarity.csv", sim_summary_rows)

    sim_lookup = {str(row["metric"]): row for row in sim_summary_rows}
    lines = [
        f"# {title}",
        "",
        f"- 样本: `datasets/test` 全 {len(samples)} 对",
        f"- 目标头: `L{layer}H{head}`",
        "",
        "## Attention by Region",
        "",
        "| region | clean mean±std | corrupt mean±std | Δ(corrupt-clean) | p |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in attn_summary_rows:
        lines.append(
            f"| {row['region']} | {float(row['clean_mean']):.4f} ± {float(row['clean_std']):.4f} | "
            f"{float(row['corrupt_mean']):.4f} ± {float(row['corrupt_std']):.4f} | "
            f"{float(row['delta_corrupt_minus_clean']):+.4f} | {float(row['p_value']):.3g} {row['significance']} |"
        )
    lines.extend(
        [
            "",
            "## Raw QK by Region",
            "",
            "| region | clean | corrupt | Δ(corrupt-clean) | p |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in qk_summary_rows:
        lines.append(
            f"| {row['region']} | {float(row['clean_mean_score']):.4f} | {float(row['corrupt_mean_score']):.4f} | "
            f"{float(row['delta_corrupt_minus_clean']):+.4f} | {float(row['p_value']):.3g} {row['significance']} |"
        )
    lines.extend(["", "## Cosine", ""])
    for row in sim_summary_rows:
        lines.append(f"- {row['metric']}: {float(row['mean']):.4f} ± {float(row['std']):.4f}")
    lines.extend(
        [
            "",
            "## 判读",
            "",
            f"- clean/corrupt query cosine 为 `{float(sim_lookup['clean_query_vs_corrupt_query']['mean']):.4f}`。",
            f"- system key cosine 为 `{float(sim_lookup['clean_system_key_vs_corrupt_system_key']['mean']):.4f}`。",
        ]
    )
    write_text(output_root / "summary.md", "\n".join(lines))
    return {
        "n_pairs": len(samples),
        "clean_query_vs_corrupt_query": float(sim_lookup["clean_query_vs_corrupt_query"]["mean"]),
        "clean_system_key_vs_corrupt_system_key": float(sim_lookup["clean_system_key_vs_corrupt_system_key"]["mean"]),
    }


def run_experiment_h(
    model,
    samples: Sequence[Sample],
    *,
    batch_size: int,
    tool_token_id: int,
    output_root: Path,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, object]:
    ensure_dir(output_root)
    pair_batches = build_pair_batches(samples, batch_size)
    n_heads = int(model.cfg.n_heads)
    q_hook_name = f"blocks.{H9_LAYER}.attn.hook_q"
    pattern_hook_name = f"blocks.{H9_LAYER}.attn.hook_pattern"
    resid_hook_name = f"blocks.{H9_LAYER}.hook_resid_pre"

    clean_q_h9 = torch.empty((len(samples), int(model.cfg.d_head)), dtype=torch.float32)
    clean_q_ctrl = torch.empty((len(samples), int(model.cfg.d_head)), dtype=torch.float32)
    corrupt_q_h9 = torch.empty((len(samples), int(model.cfg.d_head)), dtype=torch.float32)
    resid_delta = torch.empty((len(samples), int(model.cfg.d_model)), dtype=torch.float32)

    per_sample_rows: list[dict[str, object]] = []
    condition_order = [
        "baseline_clean",
        "baseline_corrupt",
        "h9_q_replace_exact",
        "h9_q_add_mean_delta",
        "random_head_q_replace_control",
        "resid_low_rank_add_1",
        "resid_low_rank_add_5",
        "resid_low_rank_add_10",
        "resid_low_rank_add_50",
        "resid_low_rank_add_full",
    ]

    progress = tqdm(pair_batches, desc="Experiment H baselines", dynamic_ncols=True)
    for batch in progress:
        clean_capture: dict[str, torch.Tensor] = {}
        corrupt_capture: dict[str, torch.Tensor] = {}
        clean_tokens = batch.clean_tokens_cpu.to(model.W_U.device)
        corrupt_tokens = batch.corrupt_tokens_cpu.to(model.W_U.device)
        clean_hooks = [
            (q_hook_name, make_head_q_capture(clean_capture, "q_h9", H9_HEAD)),
            (q_hook_name, make_head_q_capture(clean_capture, "q_ctrl", RANDOM_CONTROL_HEAD)),
            (pattern_hook_name, make_pattern_lastrow_capture(clean_capture, "pattern", H9_HEAD, n_heads)),
            (resid_hook_name, make_resid_last_capture(clean_capture, "resid",)),
        ]
        corrupt_hooks = [
            (q_hook_name, make_head_q_capture(corrupt_capture, "q_h9", H9_HEAD)),
            (pattern_hook_name, make_pattern_lastrow_capture(corrupt_capture, "pattern", H9_HEAD, n_heads)),
            (resid_hook_name, make_resid_last_capture(corrupt_capture, "resid",)),
        ]
        with torch.no_grad():
            clean_logits = model.run_with_hooks(clean_tokens, fwd_hooks=clean_hooks)
            corrupt_logits = model.run_with_hooks(corrupt_tokens, fwd_hooks=corrupt_hooks)
        clean_tool_logit, clean_top1 = tool_stats(clean_logits, tool_token_id)
        corrupt_tool_logit, corrupt_top1 = tool_stats(corrupt_logits, tool_token_id)

        clean_q_h9[batch.indices] = clean_capture["q_h9"]
        clean_q_ctrl[batch.indices] = clean_capture["q_ctrl"]
        corrupt_q_h9[batch.indices] = corrupt_capture["q_h9"]
        resid_delta[batch.indices] = clean_capture["resid"] - corrupt_capture["resid"]

        append_basic_rows(
            per_sample_rows,
            condition="baseline_clean",
            samples=samples,
            batch_indices=batch.indices,
            tool_logits=clean_tool_logit,
            top1=(clean_top1 == tool_token_id).int(),
            h9_lastrow=clean_capture["pattern"],
            side="clean",
        )
        append_basic_rows(
            per_sample_rows,
            condition="baseline_corrupt",
            samples=samples,
            batch_indices=batch.indices,
            tool_logits=corrupt_tool_logit,
            top1=(corrupt_top1 == tool_token_id).int(),
            h9_lastrow=corrupt_capture["pattern"],
            side="corrupt",
        )
        clear_cuda()

    mean_q_delta = (clean_q_h9 - corrupt_q_h9).mean(dim=0)
    resid_np = resid_delta.numpy()
    resid_mean = resid_np.mean(axis=0, keepdims=True)
    resid_mean_t = torch.from_numpy(resid_mean).float()
    centered = resid_np - resid_mean
    centered_t = torch.from_numpy(centered)
    basis = torch.linalg.svd(centered_t, full_matrices=False).Vh[: max(LOW_RANK_KS)].T.contiguous()

    progress = tqdm(pair_batches, desc="Experiment H interventions", dynamic_ncols=True)
    for batch in progress:
        corrupt_tokens = batch.corrupt_tokens_cpu.to(model.W_U.device)
        sample_delta = resid_delta[batch.indices]
        centered_delta = sample_delta - resid_mean_t
        projected_by_k: dict[int | str, torch.Tensor] = {"full": sample_delta}
        for k in LOW_RANK_KS:
            basis_k = basis[:, :k]
            projected = centered_delta @ basis_k @ basis_k.T + resid_mean_t
            projected_by_k[k] = projected

        condition_hooks = {
            "h9_q_replace_exact": [(q_hook_name, make_q_replace_hook(H9_HEAD, clean_q_h9[batch.indices]))],
            "h9_q_add_mean_delta": [(q_hook_name, make_q_add_hook(H9_HEAD, mean_q_delta))],
            "random_head_q_replace_control": [(q_hook_name, make_q_replace_hook(RANDOM_CONTROL_HEAD, clean_q_ctrl[batch.indices]))],
            "resid_low_rank_add_1": [(resid_hook_name, make_last_token_resid_add_hook(projected_by_k[1]))],
            "resid_low_rank_add_5": [(resid_hook_name, make_last_token_resid_add_hook(projected_by_k[5]))],
            "resid_low_rank_add_10": [(resid_hook_name, make_last_token_resid_add_hook(projected_by_k[10]))],
            "resid_low_rank_add_50": [(resid_hook_name, make_last_token_resid_add_hook(projected_by_k[50]))],
            "resid_low_rank_add_full": [(resid_hook_name, make_last_token_resid_add_hook(projected_by_k["full"]))],
        }

        for condition, hooks in condition_hooks.items():
            capture: dict[str, torch.Tensor] = {}
            fwd_hooks = list(hooks) + [(pattern_hook_name, make_pattern_lastrow_capture(capture, "pattern", H9_HEAD, n_heads))]
            with torch.no_grad():
                logits = model.run_with_hooks(corrupt_tokens, fwd_hooks=fwd_hooks)
            tool_logit, top1 = tool_stats(logits, tool_token_id)
            append_basic_rows(
                per_sample_rows,
                condition=condition,
                samples=samples,
                batch_indices=batch.indices,
                tool_logits=tool_logit,
                top1=(top1 == tool_token_id).int(),
                h9_lastrow=capture["pattern"],
                side="corrupt",
            )
        clear_cuda()

    write_csv(output_root / "query_specific_per_sample.csv", per_sample_rows)
    summary_rows = summarize_rows_by_condition(
        per_sample_rows,
        metric_names=("tool_logit", "system_attn_h9", "special_attn_h9"),
        baseline_condition="baseline_corrupt",
        order=condition_order,
        bootstrap_samples=bootstrap_samples,
        seed=seed,
    )
    write_csv(output_root / "query_specific_summary.csv", summary_rows)

    lookup = {str(row["condition"]): row for row in summary_rows}
    lines = [
        "# Experiment H: Query-Specific Intervention",
        "",
        f"- 样本: `datasets/test` 全 {len(samples)} 对",
        f"- 关键头: `L29H9`；负对照头: `L29H{RANDOM_CONTROL_HEAD}`",
        "",
        "| condition | flip_rate | mean_tool_logit | mean_system_attn_h9 | mean_special_attn_h9 |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['condition']} | {float(row['tool_call_top1_rate']):.2%} | "
            f"{float(row['mean_tool_logit']):.4f} | {float(row['mean_system_attn_h9']):.4f} | "
            f"{float(row['mean_special_attn_h9']):.4f} |"
        )
    lines.extend(
        [
            "",
            "## 判读",
            "",
            f"- `h9_q_replace_exact` 的 tool-call top-1 率为 `{float(lookup['h9_q_replace_exact']['tool_call_top1_rate']):.2%}`，system attention 为 `{float(lookup['h9_q_replace_exact']['mean_system_attn_h9']):.4f}`。",
            f"- `h9_q_add_mean_delta` 的恢复幅度为 `{float(lookup['h9_q_add_mean_delta']['delta_tool_logit_vs_baseline_corrupt']):+.4f}` logit。",
            f"- `random_head_q_replace_control` 的 tool-call top-1 率为 `{float(lookup['random_head_q_replace_control']['tool_call_top1_rate']):.2%}`。",
            f"- `resid_low_rank_add_full` 的 tool-call top-1 率为 `{float(lookup['resid_low_rank_add_full']['tool_call_top1_rate']):.2%}`，可作为 whole-state 上界。",
        ]
    )
    write_text(output_root / "summary.md", "\n".join(lines))
    return {"summary": summary_rows}


def make_q_capture(capture: dict[str, torch.Tensor], key: str):
    def hook_fn(value: torch.Tensor, hook):  # noqa: ANN001
        capture[key] = value.detach().cpu().float()
        return value

    return hook_fn


def run_experiment_i(
    model,
    samples: Sequence[Sample],
    *,
    batch_size: int,
    tool_token_id: int,
    output_root: Path,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, object]:
    ensure_dir(output_root)
    pair_batches = build_pair_batches(samples, batch_size)
    n_heads = int(model.cfg.n_heads)
    h9_pattern_hook = f"blocks.{H9_LAYER}.attn.hook_pattern"
    h9_z_hook = f"blocks.{H9_LAYER}.attn.hook_z"
    late_pattern_hook = f"blocks.{LATE_LAYER}.attn.hook_pattern"
    late_z_hook = f"blocks.{LATE_LAYER}.attn.hook_z"
    late_proj = precompute_head_projection(model, LATE_LAYER, LATE_HEAD, tool_token_id).cpu()

    mean_pattern_sum: dict[int, torch.Tensor] = {}
    mean_pattern_count: dict[int, int] = defaultdict(int)
    prepass = tqdm(pair_batches, desc="Experiment I clean mean pattern", dynamic_ncols=True)
    for batch in prepass:
        clean_tokens = batch.clean_tokens_cpu.to(model.W_U.device)
        capture: dict[str, torch.Tensor] = {}
        with torch.no_grad():
            _ = model.run_with_hooks(
                clean_tokens,
                fwd_hooks=[(h9_pattern_hook, make_pattern_lastrow_capture(capture, "pattern", H9_HEAD, n_heads))],
            )
        lastrows = capture["pattern"]
        if batch.token_len not in mean_pattern_sum:
            mean_pattern_sum[batch.token_len] = lastrows.sum(dim=0)
        else:
            mean_pattern_sum[batch.token_len] += lastrows.sum(dim=0)
        mean_pattern_count[batch.token_len] += int(lastrows.shape[0])
        clear_cuda()
    mean_pattern_by_len = {length: total / mean_pattern_count[length] for length, total in mean_pattern_sum.items()}

    per_sample_rows: list[dict[str, object]] = []
    dependency_rows: list[dict[str, object]] = []
    condition_order = [
        "baseline_clean",
        "baseline_corrupt",
        "h9_pattern_replace_exact",
        "h9_pattern_replace_mean_clean",
        "h9_z_replace_exact",
    ]

    progress = tqdm(pair_batches, desc="Experiment I forcing", dynamic_ncols=True)
    for batch in progress:
        clean_tokens = batch.clean_tokens_cpu.to(model.W_U.device)
        corrupt_tokens = batch.corrupt_tokens_cpu.to(model.W_U.device)
        clean_capture: dict[str, torch.Tensor] = {}
        corrupt_capture: dict[str, torch.Tensor] = {}

        clean_hooks = [
            (h9_pattern_hook, make_pattern_lastrow_capture(clean_capture, "h9_pattern", H9_HEAD, n_heads)),
            (h9_z_hook, make_head_z_capture(clean_capture, "h9_z", H9_HEAD)),
            (late_pattern_hook, make_pattern_lastrow_capture(clean_capture, "late_pattern", LATE_HEAD, n_heads)),
            (late_z_hook, make_head_z_capture(clean_capture, "late_z", LATE_HEAD)),
        ]
        corrupt_hooks = [
            (h9_pattern_hook, make_pattern_lastrow_capture(corrupt_capture, "h9_pattern", H9_HEAD, n_heads)),
            (late_pattern_hook, make_pattern_lastrow_capture(corrupt_capture, "late_pattern", LATE_HEAD, n_heads)),
            (late_z_hook, make_head_z_capture(corrupt_capture, "late_z", LATE_HEAD)),
        ]
        with torch.no_grad():
            clean_logits = model.run_with_hooks(clean_tokens, fwd_hooks=clean_hooks)
            corrupt_logits = model.run_with_hooks(corrupt_tokens, fwd_hooks=corrupt_hooks)
        clean_tool_logit, clean_top1 = tool_stats(clean_logits, tool_token_id)
        corrupt_tool_logit, corrupt_top1 = tool_stats(corrupt_logits, tool_token_id)
        clean_late_dla = torch.einsum("bd,d->b", clean_capture["late_z"], late_proj)
        corrupt_late_dla = torch.einsum("bd,d->b", corrupt_capture["late_z"], late_proj)

        append_basic_rows(
            per_sample_rows,
            condition="baseline_clean",
            samples=samples,
            batch_indices=batch.indices,
            tool_logits=clean_tool_logit,
            top1=(clean_top1 == tool_token_id).int(),
            h9_lastrow=clean_capture["h9_pattern"],
            side="clean",
            extra_by_local={
                "l33h29_system_attn": torch.tensor(
                    [region_attention_from_lastrow(clean_capture["late_pattern"][i], samples[idx].clean_region_masks)["system_attn"] for i, idx in enumerate(batch.indices)],
                    dtype=torch.float32,
                ),
                "l33h29_special_attn": torch.tensor(
                    [region_attention_from_lastrow(clean_capture["late_pattern"][i], samples[idx].clean_region_masks)["special_attn"] for i, idx in enumerate(batch.indices)],
                    dtype=torch.float32,
                ),
                "l33h29_dla": clean_late_dla,
            },
        )
        append_basic_rows(
            per_sample_rows,
            condition="baseline_corrupt",
            samples=samples,
            batch_indices=batch.indices,
            tool_logits=corrupt_tool_logit,
            top1=(corrupt_top1 == tool_token_id).int(),
            h9_lastrow=corrupt_capture["h9_pattern"],
            side="corrupt",
            extra_by_local={
                "l33h29_system_attn": torch.tensor(
                    [region_attention_from_lastrow(corrupt_capture["late_pattern"][i], samples[idx].corrupt_region_masks)["system_attn"] for i, idx in enumerate(batch.indices)],
                    dtype=torch.float32,
                ),
                "l33h29_special_attn": torch.tensor(
                    [region_attention_from_lastrow(corrupt_capture["late_pattern"][i], samples[idx].corrupt_region_masks)["special_attn"] for i, idx in enumerate(batch.indices)],
                    dtype=torch.float32,
                ),
                "l33h29_dla": corrupt_late_dla,
            },
        )

        condition_hooks = {
            "h9_pattern_replace_exact": [
                (h9_pattern_hook, make_pattern_replace_hook(H9_HEAD, clean_capture["h9_pattern"], n_heads)),
            ],
            "h9_pattern_replace_mean_clean": [
                (h9_pattern_hook, make_pattern_replace_hook(H9_HEAD, mean_pattern_by_len[batch.token_len].unsqueeze(0).repeat(len(batch.indices), 1), n_heads)),
            ],
            "h9_z_replace_exact": [
                (h9_z_hook, make_z_replace_hook(H9_HEAD, clean_capture["h9_z"])),
            ],
        }

        for condition, hooks in condition_hooks.items():
            capture: dict[str, torch.Tensor] = {}
            fwd_hooks = list(hooks) + [
                (h9_pattern_hook, make_pattern_lastrow_capture(capture, "h9_pattern", H9_HEAD, n_heads)),
                (late_pattern_hook, make_pattern_lastrow_capture(capture, "late_pattern", LATE_HEAD, n_heads)),
                (late_z_hook, make_head_z_capture(capture, "late_z", LATE_HEAD)),
            ]
            with torch.no_grad():
                logits = model.run_with_hooks(corrupt_tokens, fwd_hooks=fwd_hooks)
            tool_logit, top1 = tool_stats(logits, tool_token_id)
            late_dla = torch.einsum("bd,d->b", capture["late_z"], late_proj)
            extra = {
                "l33h29_system_attn": torch.tensor(
                    [region_attention_from_lastrow(capture["late_pattern"][i], samples[idx].corrupt_region_masks)["system_attn"] for i, idx in enumerate(batch.indices)],
                    dtype=torch.float32,
                ),
                "l33h29_special_attn": torch.tensor(
                    [region_attention_from_lastrow(capture["late_pattern"][i], samples[idx].corrupt_region_masks)["special_attn"] for i, idx in enumerate(batch.indices)],
                    dtype=torch.float32,
                ),
                "l33h29_dla": late_dla,
            }
            append_basic_rows(
                per_sample_rows,
                condition=condition,
                samples=samples,
                batch_indices=batch.indices,
                tool_logits=tool_logit,
                top1=(top1 == tool_token_id).int(),
                h9_lastrow=capture["h9_pattern"],
                side="corrupt",
                extra_by_local=extra,
            )
        clear_cuda()

    write_csv(output_root / "forcing_per_sample.csv", per_sample_rows)
    summary_rows = summarize_rows_by_condition(
        per_sample_rows,
        metric_names=("tool_logit", "system_attn_h9", "special_attn_h9", "l33h29_system_attn", "l33h29_dla"),
        baseline_condition="baseline_corrupt",
        order=condition_order,
        bootstrap_samples=bootstrap_samples,
        seed=seed,
    )
    write_csv(output_root / "forcing_summary.csv", summary_rows)

    for row in summary_rows:
        dependency_rows.append(
            {
                "condition": row["condition"],
                "mean_l33h29_system_attn": row.get("mean_l33h29_system_attn"),
                "mean_l33h29_dla": row.get("mean_l33h29_dla"),
                "mean_tool_logit": row.get("mean_tool_logit"),
                "tool_call_top1_rate": row.get("tool_call_top1_rate"),
            }
        )
    write_csv(output_root / "forcing_l33_dependency.csv", dependency_rows)

    lookup = {str(row["condition"]): row for row in summary_rows}
    lines = [
        "# Experiment I: H9 Attention Forcing",
        "",
        f"- 样本: `datasets/test` 全 {len(samples)} 对",
        "",
        "| condition | flip_rate | mean_tool_logit | mean_system_attn_h9 | mean_l33h29_dla |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['condition']} | {float(row['tool_call_top1_rate']):.2%} | "
            f"{float(row['mean_tool_logit']):.4f} | {float(row['mean_system_attn_h9']):.4f} | "
            f"{float(row['mean_l33h29_dla']):.4f} |"
        )
    lines.extend(
        [
            "",
            "## 判读",
            "",
            f"- `h9_pattern_replace_exact` 的 tool-call top-1 率为 `{float(lookup['h9_pattern_replace_exact']['tool_call_top1_rate']):.2%}`。",
            f"- `h9_pattern_replace_mean_clean` 的 tool-call top-1 率为 `{float(lookup['h9_pattern_replace_mean_clean']['tool_call_top1_rate']):.2%}`。",
            f"- `h9_z_replace_exact` 的 tool-call top-1 率为 `{float(lookup['h9_z_replace_exact']['tool_call_top1_rate']):.2%}`。",
        ]
    )
    write_text(output_root / "summary.md", "\n".join(lines))
    return {"summary": summary_rows}


def run_experiment_j(
    model,
    samples: Sequence[Sample],
    *,
    batch_size: int,
    tool_token_id: int,
    output_root: Path,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, object]:
    ensure_dir(output_root)
    attention_qk_root = output_root / "attention_qk"
    head_meta = run_head_attention_qk_analysis(
        model,
        samples,
        batch_size=batch_size,
        layer=LATE_LAYER,
        head=LATE_HEAD,
        output_root=attention_qk_root,
        title="Experiment J1: L33H29 Attention + QK",
    )

    pair_batches = build_pair_batches(samples, batch_size)
    n_heads = int(model.cfg.n_heads)
    h9_q_hook = f"blocks.{H9_LAYER}.attn.hook_q"
    h9_pattern_hook = f"blocks.{H9_LAYER}.attn.hook_pattern"
    h9_z_hook = f"blocks.{H9_LAYER}.attn.hook_z"
    l24_resid_hook = f"blocks.{L24_LAYER}.hook_resid_pre"
    late_pattern_hook = f"blocks.{LATE_LAYER}.attn.hook_pattern"
    late_z_hook = f"blocks.{LATE_LAYER}.attn.hook_z"
    late_proj = precompute_head_projection(model, LATE_LAYER, LATE_HEAD, tool_token_id).cpu()

    per_sample_rows: list[dict[str, object]] = []
    condition_order = [
        "baseline_clean",
        "baseline_corrupt",
        "corrupt_h9_q_replace_exact",
        "corrupt_h9_pattern_replace_exact",
        "corrupt_l24_last_token_patch",
        "clean_h9_z_zero_ablation",
    ]

    progress = tqdm(pair_batches, desc="Experiment J dependency", dynamic_ncols=True)
    for batch in progress:
        clean_tokens = batch.clean_tokens_cpu.to(model.W_U.device)
        corrupt_tokens = batch.corrupt_tokens_cpu.to(model.W_U.device)
        clean_capture: dict[str, torch.Tensor] = {}
        corrupt_capture: dict[str, torch.Tensor] = {}

        clean_hooks = [
            (h9_q_hook, make_head_q_capture(clean_capture, "h9_q", H9_HEAD)),
            (h9_pattern_hook, make_pattern_lastrow_capture(clean_capture, "h9_pattern", H9_HEAD, n_heads)),
            (l24_resid_hook, make_resid_last_capture(clean_capture, "l24_last")),
            (late_pattern_hook, make_pattern_lastrow_capture(clean_capture, "late_pattern", LATE_HEAD, n_heads)),
            (late_z_hook, make_head_z_capture(clean_capture, "late_z", LATE_HEAD)),
        ]
        corrupt_hooks = [
            (h9_pattern_hook, make_pattern_lastrow_capture(corrupt_capture, "h9_pattern", H9_HEAD, n_heads)),
            (l24_resid_hook, make_resid_last_capture(corrupt_capture, "l24_last")),
            (late_pattern_hook, make_pattern_lastrow_capture(corrupt_capture, "late_pattern", LATE_HEAD, n_heads)),
            (late_z_hook, make_head_z_capture(corrupt_capture, "late_z", LATE_HEAD)),
        ]
        with torch.no_grad():
            clean_logits = model.run_with_hooks(clean_tokens, fwd_hooks=clean_hooks)
            corrupt_logits = model.run_with_hooks(corrupt_tokens, fwd_hooks=corrupt_hooks)
        clean_tool_logit, clean_top1 = tool_stats(clean_logits, tool_token_id)
        corrupt_tool_logit, corrupt_top1 = tool_stats(corrupt_logits, tool_token_id)
        clean_late_dla = torch.einsum("bd,d->b", clean_capture["late_z"], late_proj)
        corrupt_late_dla = torch.einsum("bd,d->b", corrupt_capture["late_z"], late_proj)

        append_basic_rows(
            per_sample_rows,
            condition="baseline_clean",
            samples=samples,
            batch_indices=batch.indices,
            tool_logits=clean_tool_logit,
            top1=(clean_top1 == tool_token_id).int(),
            h9_lastrow=clean_capture["h9_pattern"],
            side="clean",
            extra_by_local={
                "l33h29_system_attn": torch.tensor(
                    [region_attention_from_lastrow(clean_capture["late_pattern"][i], samples[idx].clean_region_masks)["system_attn"] for i, idx in enumerate(batch.indices)],
                    dtype=torch.float32,
                ),
                "l33h29_special_attn": torch.tensor(
                    [region_attention_from_lastrow(clean_capture["late_pattern"][i], samples[idx].clean_region_masks)["special_attn"] for i, idx in enumerate(batch.indices)],
                    dtype=torch.float32,
                ),
                "l33h29_dla": clean_late_dla,
            },
        )
        append_basic_rows(
            per_sample_rows,
            condition="baseline_corrupt",
            samples=samples,
            batch_indices=batch.indices,
            tool_logits=corrupt_tool_logit,
            top1=(corrupt_top1 == tool_token_id).int(),
            h9_lastrow=corrupt_capture["h9_pattern"],
            side="corrupt",
            extra_by_local={
                "l33h29_system_attn": torch.tensor(
                    [region_attention_from_lastrow(corrupt_capture["late_pattern"][i], samples[idx].corrupt_region_masks)["system_attn"] for i, idx in enumerate(batch.indices)],
                    dtype=torch.float32,
                ),
                "l33h29_special_attn": torch.tensor(
                    [region_attention_from_lastrow(corrupt_capture["late_pattern"][i], samples[idx].corrupt_region_masks)["special_attn"] for i, idx in enumerate(batch.indices)],
                    dtype=torch.float32,
                ),
                "l33h29_dla": corrupt_late_dla,
            },
        )

        condition_hooks = {
            "corrupt_h9_q_replace_exact": [(h9_q_hook, make_q_replace_hook(H9_HEAD, clean_capture["h9_q"]))],
            "corrupt_h9_pattern_replace_exact": [(h9_pattern_hook, make_pattern_replace_hook(H9_HEAD, clean_capture["h9_pattern"], n_heads))],
            "corrupt_l24_last_token_patch": [
                (l24_resid_hook, make_last_token_resid_add_hook(clean_capture["l24_last"] - corrupt_capture["l24_last"]))
            ],
        }

        for condition, hooks in condition_hooks.items():
            capture: dict[str, torch.Tensor] = {}
            fwd_hooks = list(hooks) + [
                (h9_pattern_hook, make_pattern_lastrow_capture(capture, "h9_pattern", H9_HEAD, n_heads)),
                (late_pattern_hook, make_pattern_lastrow_capture(capture, "late_pattern", LATE_HEAD, n_heads)),
                (late_z_hook, make_head_z_capture(capture, "late_z", LATE_HEAD)),
            ]
            with torch.no_grad():
                logits = model.run_with_hooks(corrupt_tokens, fwd_hooks=fwd_hooks)
            tool_logit, top1 = tool_stats(logits, tool_token_id)
            late_dla = torch.einsum("bd,d->b", capture["late_z"], late_proj)
            append_basic_rows(
                per_sample_rows,
                condition=condition,
                samples=samples,
                batch_indices=batch.indices,
                tool_logits=tool_logit,
                top1=(top1 == tool_token_id).int(),
                h9_lastrow=capture["h9_pattern"],
                side="corrupt",
                extra_by_local={
                    "l33h29_system_attn": torch.tensor(
                        [region_attention_from_lastrow(capture["late_pattern"][i], samples[idx].corrupt_region_masks)["system_attn"] for i, idx in enumerate(batch.indices)],
                        dtype=torch.float32,
                    ),
                    "l33h29_special_attn": torch.tensor(
                        [region_attention_from_lastrow(capture["late_pattern"][i], samples[idx].corrupt_region_masks)["special_attn"] for i, idx in enumerate(batch.indices)],
                        dtype=torch.float32,
                    ),
                    "l33h29_dla": late_dla,
                },
            )

        zero_capture: dict[str, torch.Tensor] = {}
        clean_zero_hooks = [
            (h9_z_hook, make_z_zero_hook(H9_HEAD)),
            (h9_pattern_hook, make_pattern_lastrow_capture(zero_capture, "h9_pattern", H9_HEAD, n_heads)),
            (late_pattern_hook, make_pattern_lastrow_capture(zero_capture, "late_pattern", LATE_HEAD, n_heads)),
            (late_z_hook, make_head_z_capture(zero_capture, "late_z", LATE_HEAD)),
        ]
        with torch.no_grad():
            clean_zero_logits = model.run_with_hooks(clean_tokens, fwd_hooks=clean_zero_hooks)
        clean_zero_tool_logit, clean_zero_top1 = tool_stats(clean_zero_logits, tool_token_id)
        clean_zero_late_dla = torch.einsum("bd,d->b", zero_capture["late_z"], late_proj)
        append_basic_rows(
            per_sample_rows,
            condition="clean_h9_z_zero_ablation",
            samples=samples,
            batch_indices=batch.indices,
            tool_logits=clean_zero_tool_logit,
            top1=(clean_zero_top1 == tool_token_id).int(),
            h9_lastrow=zero_capture["h9_pattern"],
            side="clean",
            extra_by_local={
                "l33h29_system_attn": torch.tensor(
                    [region_attention_from_lastrow(zero_capture["late_pattern"][i], samples[idx].clean_region_masks)["system_attn"] for i, idx in enumerate(batch.indices)],
                    dtype=torch.float32,
                ),
                "l33h29_special_attn": torch.tensor(
                    [region_attention_from_lastrow(zero_capture["late_pattern"][i], samples[idx].clean_region_masks)["special_attn"] for i, idx in enumerate(batch.indices)],
                    dtype=torch.float32,
                ),
                "l33h29_dla": clean_zero_late_dla,
            },
        )
        clear_cuda()

    write_csv(output_root / "l33h29_dependency_per_sample.csv", per_sample_rows)
    summary_rows = summarize_rows_by_condition(
        per_sample_rows,
        metric_names=("tool_logit", "system_attn_h9", "l33h29_system_attn", "l33h29_dla"),
        baseline_condition="baseline_corrupt",
        order=condition_order,
        bootstrap_samples=bootstrap_samples,
        seed=seed,
    )
    write_csv(output_root / "l33h29_dependency_summary.csv", summary_rows)

    lookup = {str(row["condition"]): row for row in summary_rows}
    lines = [
        "# Experiment J2: L33H29 Dependency on H9 and L24 State",
        "",
        "| condition | top1_rate | mean_tool_logit | mean_l33h29_system_attn | mean_l33h29_dla |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['condition']} | {float(row['tool_call_top1_rate']):.2%} | {float(row['mean_tool_logit']):.4f} | "
            f"{float(row['mean_l33h29_system_attn']):.4f} | {float(row['mean_l33h29_dla']):.4f} |"
        )
    lines.extend(
        [
            "",
            "## 判读",
            "",
            f"- `corrupt_h9_q_replace_exact` 的 L33H29 DLA 为 `{float(lookup['corrupt_h9_q_replace_exact']['mean_l33h29_dla']):.4f}`。",
            f"- `corrupt_h9_pattern_replace_exact` 的 L33H29 DLA 为 `{float(lookup['corrupt_h9_pattern_replace_exact']['mean_l33h29_dla']):.4f}`。",
            f"- `corrupt_l24_last_token_patch` 的 L33H29 DLA 为 `{float(lookup['corrupt_l24_last_token_patch']['mean_l33h29_dla']):.4f}`。",
            f"- `clean_h9_z_zero_ablation` 的 L33H29 DLA 为 `{float(lookup['clean_h9_z_zero_ablation']['mean_l33h29_dla']):.4f}`。",
        ]
    )
    write_text(output_root / "summary.md", "\n".join(lines))
    return {"attention_qk": head_meta, "dependency": summary_rows}


def run_experiment_k(
    model,
    samples: Sequence[Sample],
    *,
    batch_size: int,
    tool_token_id: int,
    output_root: Path,
    bootstrap_samples: int,
    permutation_samples: int,
    seed: int,
) -> dict[str, object]:
    ensure_dir(output_root)
    pair_batches = build_pair_batches(samples, batch_size)
    pattern_hook_name = f"blocks.{H9_LAYER}.attn.hook_pattern"
    z_hook_name = f"blocks.{H9_LAYER}.attn.hook_z"
    head_proj = precompute_head_projection(model, H9_LAYER, H9_HEAD, tool_token_id).cpu()
    n_heads = int(model.cfg.n_heads)

    baseline_rows: list[dict[str, object]] = []
    baseline_vectors: dict[str, np.ndarray] = {}
    progress = tqdm(pair_batches, desc="Experiment K baseline", dynamic_ncols=True)
    baseline_tool = np.empty(len(samples), dtype=np.float64)
    baseline_top1 = np.empty(len(samples), dtype=np.float64)
    baseline_system = np.empty(len(samples), dtype=np.float64)
    baseline_dla = np.empty(len(samples), dtype=np.float64)
    for batch in progress:
        corrupt_tokens = batch.corrupt_tokens_cpu.to(model.W_U.device)
        capture: dict[str, torch.Tensor] = {}
        with torch.no_grad():
            logits = model.run_with_hooks(
                corrupt_tokens,
                fwd_hooks=[
                    (pattern_hook_name, make_pattern_lastrow_capture(capture, "pattern", H9_HEAD, n_heads)),
                    (z_hook_name, make_head_z_capture(capture, "z", H9_HEAD)),
                ],
            )
        tool_logit, top1 = tool_stats(logits, tool_token_id)
        dla = torch.einsum("bd,d->b", capture["z"], head_proj)
        append_basic_rows(
            baseline_rows,
            condition="baseline_corrupt",
            samples=samples,
            batch_indices=batch.indices,
            tool_logits=tool_logit,
            top1=(top1 == tool_token_id).int(),
            h9_lastrow=capture["pattern"],
            side="corrupt",
            extra_by_local={"dla_h9": dla},
        )
        for local_idx, sample_idx in enumerate(batch.indices):
            baseline_tool[sample_idx] = float(tool_logit[local_idx].item())
            baseline_top1[sample_idx] = float((top1[local_idx] == tool_token_id).item())
            baseline_system[sample_idx] = float(region_attention_from_lastrow(capture["pattern"][local_idx], samples[sample_idx].corrupt_region_masks)["system_attn"])
            baseline_dla[sample_idx] = float(dla[local_idx].item())
        clear_cuda()
    baseline_vectors = {
        "tool_logit": baseline_tool,
        "tool_call_top1": baseline_top1,
        "system_attn_h9": baseline_system,
        "dla_h9": baseline_dla,
    }

    comparison_rows: list[dict[str, object]] = []
    bootstrap_rows: list[dict[str, object]] = []
    permutation_rows: list[dict[str, object]] = []

    for layer in SUPPRESSOR_LAYERS:
        tc_weights_raw = load_file(str(TRANSCODER_DIR / f"layer_{layer}.safetensors"))
        tc_weights = {key: value.to(model.W_U.device) for key, value in tc_weights_raw.items()}
        diff_csv = DIFF_ROOT / f"differential_features_L{layer}.csv"
        corrupt_pool = load_top_feature_ids(diff_csv, "corrupt-selective", max(SUPPRESSOR_KS))
        layer_progress = tqdm(SUPPRESSOR_KS, desc=f"Experiment K L{layer}", dynamic_ncols=True)
        for k in layer_progress:
            hook_builder = build_feature_ablation_hooks(tc_weights=tc_weights, feature_ids=corrupt_pool[:k])
            out_tool = np.empty(len(samples), dtype=np.float64)
            out_top1 = np.empty(len(samples), dtype=np.float64)
            out_system = np.empty(len(samples), dtype=np.float64)
            out_dla = np.empty(len(samples), dtype=np.float64)
            for batch in pair_batches:
                corrupt_tokens = batch.corrupt_tokens_cpu.to(model.W_U.device)
                capture: dict[str, torch.Tensor] = {}
                in_hook, out_hook = hook_builder()
                hooks = [
                    (f"blocks.{layer}.hook_mlp_in", in_hook),
                    (f"blocks.{layer}.hook_mlp_out", out_hook),
                    (pattern_hook_name, make_pattern_lastrow_capture(capture, "pattern", H9_HEAD, n_heads)),
                    (z_hook_name, make_head_z_capture(capture, "z", H9_HEAD)),
                ]
                with torch.no_grad():
                    logits = model.run_with_hooks(corrupt_tokens, fwd_hooks=hooks)
                tool_logit, top1 = tool_stats(logits, tool_token_id)
                dla = torch.einsum("bd,d->b", capture["z"], head_proj)
                for local_idx, sample_idx in enumerate(batch.indices):
                    out_tool[sample_idx] = float(tool_logit[local_idx].item())
                    out_top1[sample_idx] = float((top1[local_idx] == tool_token_id).item())
                    out_system[sample_idx] = float(
                        region_attention_from_lastrow(capture["pattern"][local_idx], samples[sample_idx].corrupt_region_masks)["system_attn"]
                    )
                    out_dla[sample_idx] = float(dla[local_idx].item())
                clear_cuda()

            tool_delta = out_tool - baseline_vectors["tool_logit"]
            system_delta = out_system - baseline_vectors["system_attn_h9"]
            dla_delta = out_dla - baseline_vectors["dla_h9"]
            comparison_rows.append(
                {
                    "layer": layer,
                    "k": k,
                    "tool_call_top1_rate": float(out_top1.mean()),
                    "mean_tool_logit": float(out_tool.mean()),
                    "mean_system_attn_h9": float(out_system.mean()),
                    "mean_dla_h9": float(out_dla.mean()),
                    "mean_tool_logit_delta": float(tool_delta.mean()),
                    "mean_system_attn_h9_delta": float(system_delta.mean()),
                    "mean_dla_h9_delta": float(dla_delta.mean()),
                }
            )
            for metric_name, deltas in (
                ("tool_logit_delta", tool_delta),
                ("system_attn_h9_delta", system_delta),
                ("dla_h9_delta", dla_delta),
            ):
                ci_low, ci_high = bootstrap_ci_delta(deltas, n_boot=bootstrap_samples, seed=seed + layer * 100 + k + len(metric_name))
                bootstrap_rows.append(
                    {
                        "layer": layer,
                        "k": k,
                        "metric": metric_name,
                        "mean_delta": float(deltas.mean()),
                        "ci_low": ci_low,
                        "ci_high": ci_high,
                    }
                )
                p_value = paired_permutation_pvalue(deltas, n_perm=permutation_samples, seed=seed + 5000 + layer * 100 + k + len(metric_name))
                permutation_rows.append(
                    {
                        "layer": layer,
                        "k": k,
                        "metric": metric_name,
                        "mean_delta": float(deltas.mean()),
                        "p_value": p_value,
                        "effect_size_dz": cohen_dz(deltas),
                    }
                )
        del tc_weights
        clear_cuda()

    write_csv(output_root / "layer_comparison.csv", comparison_rows)
    write_csv(output_root / "bootstrap_ci.csv", bootstrap_rows)
    write_csv(output_root / "permutation_stats.csv", permutation_rows)

    dla_lookup = {(int(row["layer"]), int(row["k"])): row for row in comparison_rows}
    perm_lookup = {(int(row["layer"]), int(row["k"]), str(row["metric"])): row for row in permutation_rows}
    lines = [
        "# Experiment K: L21 Suppressor Significance",
        "",
        f"- 样本: `datasets/test` 全 {len(samples)} 对",
        "",
        "| layer | k | mean_dla_delta | mean_tool_logit_delta | mean_system_attn_delta |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in comparison_rows:
        lines.append(
            f"| L{row['layer']} | {row['k']} | {float(row['mean_dla_h9_delta']):+.4f} | "
            f"{float(row['mean_tool_logit_delta']):+.4f} | {float(row['mean_system_attn_h9_delta']):+.4f} |"
        )
    lines.extend(["", "## 判读", ""])
    focus = dla_lookup[(21, 50)]
    focus_perm = perm_lookup[(21, 50, "dla_h9_delta")]
    lines.append(
        f"- L21 k=50 的 DLA delta 为 `{float(focus['mean_dla_h9_delta']):+.4f}`，permutation p=`{float(focus_perm['p_value']):.3g}`。"
    )
    write_text(output_root / "summary.md", "\n".join(lines))
    return {"comparison": comparison_rows}


def run_experiment_l(
    model,
    tokenizer,
    samples: Sequence[Sample],
    *,
    batch_size: int,
    tool_token_id: int,
    output_root: Path,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, object]:
    ensure_dir(output_root)
    n_heads = int(model.cfg.n_heads)
    pattern_hook_name = f"blocks.{H9_LAYER}.attn.hook_pattern"

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
        "format_preserved_semantics_broken": make_length_matched_ids(
            tokenizer,
            prefix_text='{"type":"function","function":{"name":"zz","description":"broken',
            suffix_text='","parameters":{"type":"object","properties":{"aa":{"type":"string"},"bb":{"type":"string"}},"required":["aa","bb"]}}}',
            filler_text=" broken",
            target_len=target_schema_len,
        ),
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
            for condition, ids in replacement_ids.items():
                modified = torch.cat([tokens[:start], ids, tokens[end:]], dim=0)
                if int(modified.shape[0]) != int(tokens.shape[0]):
                    raise RuntimeError(f"Length mismatch for {condition} on {sample.sample_id}")
                items.append(
                    SinglePromptItem(
                        sample_id=sample.sample_id,
                        condition=condition,
                        verb_condition=verb_condition,
                        token_len=int(modified.shape[0]),
                        tokens_cpu=modified,
                        system_mask=system_mask,
                    )
                )

    batches = build_single_batches(items, batch_size)
    per_sample_rows: list[dict[str, object]] = []
    progress = tqdm(batches, desc="Experiment L schema control", dynamic_ncols=True)
    for batch in progress:
        tokens = batch.tokens_cpu.to(model.W_U.device)
        capture: dict[str, torch.Tensor] = {}
        with torch.no_grad():
            logits = model.run_with_hooks(tokens, fwd_hooks=[(pattern_hook_name, make_pattern_lastrow_capture(capture, "pattern", H9_HEAD, n_heads))])
        tool_logit, top1 = tool_stats(logits, tool_token_id)
        for local_idx, item in enumerate(batch.items):
            system_attn = float(capture["pattern"][local_idx][item.system_mask].sum().item())
            per_sample_rows.append(
                {
                    "sample_id": item.sample_id,
                    "condition": item.condition,
                    "verb_condition": item.verb_condition,
                    "tool_logit": float(tool_logit[local_idx].item()),
                    "is_tool_call_top1": int(top1[local_idx].item() == tool_token_id),
                    "system_attn_h9": system_attn,
                }
            )
        clear_cuda()

    write_csv(output_root / "schema_control_per_sample.csv", per_sample_rows)
    grouped_rows: list[dict[str, object]] = []
    for verb_condition in ("clean_action_side", "corrupt_analysis_side"):
        side_rows = [row for row in per_sample_rows if row["verb_condition"] == verb_condition]
        summary = summarize_rows_by_condition(
            side_rows,
            metric_names=("tool_logit", "system_attn_h9"),
            baseline_condition="original_schema",
            order=[
                "original_schema",
                "schema_removed",
                "unrelated_valid_schema_length_matched",
                "random_text_length_matched",
                "format_preserved_semantics_broken",
            ],
            bootstrap_samples=bootstrap_samples,
            seed=seed + (0 if verb_condition == "clean_action_side" else 10_000),
        )
        for row in summary:
            row["verb_condition"] = verb_condition
            grouped_rows.append(row)
    write_csv(output_root / "schema_control_summary.csv", grouped_rows)

    lines = [
        "# Experiment L: Schema Replacement Control",
        "",
        f"- 样本: `datasets/test` 全 {len(samples)} 对",
        "",
        "| verb_condition | condition | top1_rate | mean_tool_logit | mean_system_attn_h9 |",
        "|---|---|---:|---:|---:|",
    ]
    for row in grouped_rows:
        lines.append(
            f"| {row['verb_condition']} | {row['condition']} | {float(row['tool_call_top1_rate']):.2%} | "
            f"{float(row['mean_tool_logit']):.4f} | {float(row['mean_system_attn_h9']):.4f} |"
        )
    write_text(output_root / "summary.md", "\n".join(lines))
    return {"summary": grouped_rows}


def run_experiment_m(
    model,
    samples: Sequence[Sample],
    *,
    batch_size: int,
    tool_token_id: int,
    output_root: Path,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, object]:
    ensure_dir(output_root)
    pair_batches = build_pair_batches(samples, batch_size)
    n_heads = int(model.cfg.n_heads)
    resid_hook_name = f"blocks.{L24_LAYER}.hook_resid_pre"
    pattern_hook_name = f"blocks.{H9_LAYER}.attn.hook_pattern"

    per_sample_rows: list[dict[str, object]] = []
    condition_order = [
        "baseline_clean",
        "baseline_corrupt",
        "last_token_only_patch",
        "all_tokens_patch",
        "all_except_last_patch",
        "verb_token_only_patch",
        "system_region_patch",
    ]

    progress = tqdm(pair_batches, desc="Experiment M cross-position", dynamic_ncols=True)
    for batch in progress:
        clean_tokens = batch.clean_tokens_cpu.to(model.W_U.device)
        corrupt_tokens = batch.corrupt_tokens_cpu.to(model.W_U.device)
        clean_capture: dict[str, torch.Tensor] = {}
        corrupt_capture: dict[str, torch.Tensor] = {}
        with torch.no_grad():
            clean_logits = model.run_with_hooks(
                clean_tokens,
                fwd_hooks=[
                    (resid_hook_name, make_resid_full_capture(clean_capture, "resid")),
                    (pattern_hook_name, make_pattern_lastrow_capture(clean_capture, "pattern", H9_HEAD, n_heads)),
                ],
            )
            corrupt_logits = model.run_with_hooks(
                corrupt_tokens,
                fwd_hooks=[
                    (resid_hook_name, make_resid_full_capture(corrupt_capture, "resid")),
                    (pattern_hook_name, make_pattern_lastrow_capture(corrupt_capture, "pattern", H9_HEAD, n_heads)),
                ],
            )
        clean_tool_logit, clean_top1 = tool_stats(clean_logits, tool_token_id)
        corrupt_tool_logit, corrupt_top1 = tool_stats(corrupt_logits, tool_token_id)
        append_basic_rows(
            per_sample_rows,
            condition="baseline_clean",
            samples=samples,
            batch_indices=batch.indices,
            tool_logits=clean_tool_logit,
            top1=(clean_top1 == tool_token_id).int(),
            h9_lastrow=clean_capture["pattern"],
            side="clean",
        )
        append_basic_rows(
            per_sample_rows,
            condition="baseline_corrupt",
            samples=samples,
            batch_indices=batch.indices,
            tool_logits=corrupt_tool_logit,
            top1=(corrupt_top1 == tool_token_id).int(),
            h9_lastrow=corrupt_capture["pattern"],
            side="corrupt",
        )

        clean_resid = clean_capture["resid"]
        masks_by_condition = {
            "last_token_only_patch": [torch.nn.functional.pad(torch.tensor([True]), (batch.token_len - 1, 0), value=False) for _ in batch.indices],
            "all_tokens_patch": [torch.ones(batch.token_len, dtype=torch.bool) for _ in batch.indices],
            "all_except_last_patch": [torch.cat([torch.ones(batch.token_len - 1, dtype=torch.bool), torch.zeros(1, dtype=torch.bool)]) for _ in batch.indices],
            "verb_token_only_patch": [samples[idx].corrupt_region_masks["verb"] for idx in batch.indices],
            "system_region_patch": [samples[idx].corrupt_region_masks["system"] for idx in batch.indices],
        }
        for condition, masks in masks_by_condition.items():
            capture: dict[str, torch.Tensor] = {}
            hooks = [
                (resid_hook_name, make_resid_position_replace_hook(clean_resid, masks)),
                (pattern_hook_name, make_pattern_lastrow_capture(capture, "pattern", H9_HEAD, n_heads)),
            ]
            with torch.no_grad():
                logits = model.run_with_hooks(corrupt_tokens, fwd_hooks=hooks)
            tool_logit, top1 = tool_stats(logits, tool_token_id)
            append_basic_rows(
                per_sample_rows,
                condition=condition,
                samples=samples,
                batch_indices=batch.indices,
                tool_logits=tool_logit,
                top1=(top1 == tool_token_id).int(),
                h9_lastrow=capture["pattern"],
                side="corrupt",
            )
        clear_cuda()

    write_csv(output_root / "cross_position_per_sample.csv", per_sample_rows)
    summary_rows = summarize_rows_by_condition(
        per_sample_rows,
        metric_names=("tool_logit", "system_attn_h9"),
        baseline_condition="baseline_corrupt",
        order=condition_order,
        bootstrap_samples=bootstrap_samples,
        seed=seed,
    )
    write_csv(output_root / "cross_position_summary.csv", summary_rows)

    lines = [
        "# Experiment M: Cross-Position / KV Control",
        "",
        "| condition | top1_rate | mean_tool_logit | mean_system_attn_h9 |",
        "|---|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['condition']} | {float(row['tool_call_top1_rate']):.2%} | "
            f"{float(row['mean_tool_logit']):.4f} | {float(row['mean_system_attn_h9']):.4f} |"
        )
    write_text(output_root / "summary.md", "\n".join(lines))
    return {"summary": summary_rows}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 4 reviewer-strengthening experiments for Qwen3-8B")
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH)
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--output-root", type=Path, default=PHASE4_ROOT)
    parser.add_argument("--max-pairs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--permutation-samples", type=int, default=5000)
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=["all"],
        choices=["all", "H", "I", "J", "K", "L", "M"],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    experiments = list(args.experiments)
    if "all" in experiments:
        experiments = ["H", "I", "J", "K", "L", "M"]

    model, tokenizer = load_hooked_qwen3(str(args.model_path), device=args.device, dtype=torch.bfloat16)
    if hasattr(model, "set_use_hook_mlp_in"):
        model.set_use_hook_mlp_in(True)
    if hasattr(model, "cfg") and hasattr(model.cfg, "use_hook_mlp_in"):
        model.cfg.use_hook_mlp_in = True

    tool_token_ids = tokenizer.encode(TOOL_CALL_STR, add_special_tokens=False)
    if len(tool_token_ids) != 1:
        raise ValueError(f"{TOOL_CALL_STR!r} maps to unexpected token ids: {tool_token_ids}")
    tool_token_id = int(tool_token_ids[0])

    samples = load_samples(args.dataset_root, model, tokenizer, max_pairs=args.max_pairs)

    root_meta: dict[str, object] = {
        "seed": args.seed,
        "model_path": str(args.model_path),
        "dataset_root": str(args.dataset_root),
        "max_pairs": len(samples),
        "batch_size": args.batch_size,
        "experiments": experiments,
    }

    if "H" in experiments:
        meta = run_experiment_h(
            model,
            samples,
            batch_size=args.batch_size,
            tool_token_id=tool_token_id,
            output_root=args.output_root / "exp_h_query_specific",
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed,
        )
        root_meta["H"] = meta

    if "I" in experiments:
        meta = run_experiment_i(
            model,
            samples,
            batch_size=args.batch_size,
            tool_token_id=tool_token_id,
            output_root=args.output_root / "exp_i_h9_forcing",
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed + 100,
        )
        root_meta["I"] = meta

    if "J" in experiments:
        meta = run_experiment_j(
            model,
            samples,
            batch_size=args.batch_size,
            tool_token_id=tool_token_id,
            output_root=args.output_root / "exp_j_l33h29_mechanism",
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed + 200,
        )
        root_meta["J"] = meta

    if "K" in experiments:
        meta = run_experiment_k(
            model,
            samples,
            batch_size=args.batch_size,
            tool_token_id=tool_token_id,
            output_root=args.output_root / "exp_k_l21_significance",
            bootstrap_samples=args.bootstrap_samples,
            permutation_samples=args.permutation_samples,
            seed=args.seed + 300,
        )
        root_meta["K"] = meta

    if "L" in experiments:
        meta = run_experiment_l(
            model,
            tokenizer,
            samples,
            batch_size=args.batch_size,
            tool_token_id=tool_token_id,
            output_root=args.output_root / "exp_l_schema_control",
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed + 400,
        )
        root_meta["L"] = meta

    if "M" in experiments:
        meta = run_experiment_m(
            model,
            samples,
            batch_size=args.batch_size,
            tool_token_id=tool_token_id,
            output_root=args.output_root / "exp_m_cross_position_control",
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed + 500,
        )
        root_meta["M"] = meta

    write_json(args.output_root / "metadata.json", root_meta)
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
