#!/usr/bin/env python3
from __future__ import annotations

import csv
import gc
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import torch
from tqdm.auto import tqdm

from phase4_reviewer_strengthening import (
    H9_HEAD,
    H9_LAYER,
    LATE_HEAD,
    LATE_LAYER,
    TOOL_CALL_STR,
    make_head_z_capture,
    make_pattern_lastrow_capture,
    make_resid_last_capture,
    precompute_head_projection,
    region_attention_from_lastrow,
)
from task_attention_path_analysis import (
    DATASET_ROOT,
    MODEL_PATH,
    PairBatch,
    Sample,
    build_pair_batches,
    clear_cuda,
    ensure_dir,
    load_samples,
    set_seed,
    write_csv,
    write_text,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DISCOVERY_DATASET_ROOT = PROJECT_ROOT / "datasets" / "train"
EVAL_DATASET_ROOT = PROJECT_ROOT / "datasets" / "test"
PHASE7_ROOT = PROJECT_ROOT / "results" / "8b_main" / "phase7_l24_directionality"
EXP_A_ROOT = PHASE7_ROOT / "exp_a_fixed_direction"
EXP_B_ROOT = PHASE7_ROOT / "exp_b_directional_necessity"

PATCH_LAYER = 24
DEFAULT_N_COMPONENTS = 10
DEFAULT_ALPHA_SWEEP_A = (0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0)
DEFAULT_ALPHA_SWEEP_B = (0.25, 0.5, 0.75, 1.0, 1.25, 1.5)


@dataclass(frozen=True)
class PairBaselineCache:
    samples: list[Sample]
    pair_batches: list[PairBatch]
    clean_resid: torch.Tensor
    corrupt_resid: torch.Tensor
    baseline_clean: dict[str, torch.Tensor]
    baseline_corrupt: dict[str, torch.Tensor]


def manifest_pair_count(dataset_root: Path) -> int:
    manifest_path = dataset_root / "clean" / "manifest.jsonl"
    with manifest_path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def orient_components(components: torch.Tensor, diff_vectors: torch.Tensor) -> torch.Tensor:
    oriented = components.clone()
    for idx in range(oriented.shape[0]):
        direction = oriented[idx]
        if float((diff_vectors @ direction).mean().item()) < 0:
            oriented[idx] = -direction
    return oriented


def compute_pca(diff_vectors: torch.Tensor, *, n_components: int) -> dict[str, torch.Tensor]:
    centered = diff_vectors - diff_vectors.mean(dim=0, keepdim=True)
    _, singular_values, vh = torch.linalg.svd(centered, full_matrices=False)
    actual_components = min(n_components, int(vh.shape[0]))
    components = orient_components(vh[:actual_components].contiguous(), diff_vectors)
    exp_var = (singular_values[:actual_components] ** 2) / max(diff_vectors.shape[0] - 1, 1)
    total_var = float((centered.pow(2).sum().item()) / max(diff_vectors.shape[0] - 1, 1))
    exp_var_ratio = exp_var / total_var if total_var > 0 else torch.zeros_like(exp_var)
    return {
        "components": components,
        "mean_diff": diff_vectors.mean(dim=0),
        "explained_variance": exp_var,
        "explained_variance_ratio": exp_var_ratio,
        "centered": centered,
    }


def project_delta(diff_batch: torch.Tensor, pca_bundle: dict[str, torch.Tensor], *, k: int) -> torch.Tensor:
    components = pca_bundle["components"][:k]
    mean_diff = pca_bundle["mean_diff"].unsqueeze(0)
    centered = diff_batch - mean_diff
    coeff = centered @ components.T
    return coeff @ components + mean_diff


def make_position_resid_add_hook(delta_cpu: torch.Tensor, position_mode: str):
    if position_mode not in {"last", "all", "all_except_last"}:
        raise ValueError(f"Unsupported position_mode={position_mode!r}")

    def hook_fn(value: torch.Tensor, hook):  # noqa: ANN001
        out = value.clone()
        delta = delta_cpu.to(device=value.device, dtype=value.dtype)
        if position_mode == "last":
            out[:, -1, :] = out[:, -1, :] + delta
        elif position_mode == "all":
            out = out + delta.unsqueeze(1)
        else:
            if int(out.shape[1]) > 1:
                out[:, :-1, :] = out[:, :-1, :] + delta.unsqueeze(1)
        return out

    return hook_fn


def tool_stats_with_margin(logits: torch.Tensor, tool_token_id: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    last_logits = logits[:, -1, :].detach()
    tool_logit = last_logits[:, tool_token_id].detach().cpu().float()
    top_vals, top_idx = torch.topk(last_logits, k=2, dim=-1)
    top1 = top_idx[:, 0].detach().cpu()
    competitor = torch.where(
        top_idx[:, 0] == tool_token_id,
        top_vals[:, 1],
        top_vals[:, 0],
    )
    margin = (last_logits[:, tool_token_id] - competitor).detach().cpu().float()
    return tool_logit, top1, margin


def direction_scores(resid: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
    return torch.mv(resid.float(), direction.float())


def normalized_random_direction(dim: int, *, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    direction = torch.randn(dim, generator=generator, dtype=torch.float32)
    return direction / direction.norm().clamp_min(1e-12)


def _empty_metric_dict(n_samples: int) -> dict[str, torch.Tensor]:
    return {
        "tool_logit": torch.empty(n_samples, dtype=torch.float32),
        "top1": torch.empty(n_samples, dtype=torch.long),
        "margin": torch.empty(n_samples, dtype=torch.float32),
        "system_attn_h9": torch.empty(n_samples, dtype=torch.float32),
        "l33h29_system_attn": torch.empty(n_samples, dtype=torch.float32),
        "l33h29_dla": torch.empty(n_samples, dtype=torch.float32),
    }


def _system_attn_tensor(
    lastrows: torch.Tensor,
    samples: Sequence[Sample],
    batch_indices: Sequence[int],
    *,
    eval_side: str,
) -> torch.Tensor:
    values = []
    for local_idx, sample_idx in enumerate(batch_indices):
        sample = samples[sample_idx]
        masks = sample.clean_region_masks if eval_side == "clean" else sample.corrupt_region_masks
        values.append(region_attention_from_lastrow(lastrows[local_idx], masks)["system_attn"])
    return torch.tensor(values, dtype=torch.float32)


def collect_pair_last_residuals(
    model,
    pair_batches: Sequence[PairBatch],
    *,
    layer: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    hook_name = f"blocks.{layer}.hook_resid_pre"
    n_samples = sum(len(batch.indices) for batch in pair_batches)
    d_model = int(model.cfg.d_model)
    clean_resid = torch.empty((n_samples, d_model), dtype=torch.float32)
    corrupt_resid = torch.empty((n_samples, d_model), dtype=torch.float32)

    progress = tqdm(pair_batches, desc=f"Capture L{layer} residuals", dynamic_ncols=True)
    for batch in progress:
        clean_tokens = batch.clean_tokens_cpu.to(model.W_U.device)
        corrupt_tokens = batch.corrupt_tokens_cpu.to(model.W_U.device)
        clean_capture: dict[str, torch.Tensor] = {}
        corrupt_capture: dict[str, torch.Tensor] = {}
        with torch.no_grad():
            _ = model.run_with_hooks(clean_tokens, fwd_hooks=[(hook_name, make_resid_last_capture(clean_capture, "resid"))])
            _ = model.run_with_hooks(corrupt_tokens, fwd_hooks=[(hook_name, make_resid_last_capture(corrupt_capture, "resid"))])
        clean_resid[batch.indices] = clean_capture["resid"]
        corrupt_resid[batch.indices] = corrupt_capture["resid"]
        clear_cuda()
    return clean_resid, corrupt_resid


def load_or_collect_pair_baseline(
    model,
    tokenizer,
    *,
    dataset_root: Path,
    max_pairs: int,
    batch_size: int,
    patch_layer: int,
    cache_path: Path | None = None,
) -> PairBaselineCache:
    samples = load_samples(dataset_root, model, tokenizer, max_pairs=max_pairs)
    pair_batches = build_pair_batches(samples, batch_size)
    sample_ids = [sample.sample_id for sample in samples]

    if cache_path is not None and cache_path.exists():
        payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        if list(payload.get("sample_ids", [])) == sample_ids and int(payload.get("patch_layer", -1)) == int(patch_layer):
            return PairBaselineCache(
                samples=samples,
                pair_batches=pair_batches,
                clean_resid=payload["clean_resid"].float(),
                corrupt_resid=payload["corrupt_resid"].float(),
                baseline_clean={key: value.clone() for key, value in payload["baseline_clean"].items()},
                baseline_corrupt={key: value.clone() for key, value in payload["baseline_corrupt"].items()},
            )

    n_samples = len(samples)
    d_model = int(model.cfg.d_model)
    clean_resid = torch.empty((n_samples, d_model), dtype=torch.float32)
    corrupt_resid = torch.empty((n_samples, d_model), dtype=torch.float32)
    baseline_clean = _empty_metric_dict(n_samples)
    baseline_corrupt = _empty_metric_dict(n_samples)

    resid_hook_name = f"blocks.{patch_layer}.hook_resid_pre"
    h9_pattern_hook = f"blocks.{H9_LAYER}.attn.hook_pattern"
    late_pattern_hook = f"blocks.{LATE_LAYER}.attn.hook_pattern"
    late_z_hook = f"blocks.{LATE_LAYER}.attn.hook_z"
    n_heads = int(model.cfg.n_heads)
    tool_token_ids = tokenizer.encode(TOOL_CALL_STR, add_special_tokens=False)
    if len(tool_token_ids) != 1:
        raise ValueError(f"{TOOL_CALL_STR!r} is not a single token: {tool_token_ids}")
    tool_token_id = int(tool_token_ids[0])
    late_proj = precompute_head_projection(model, LATE_LAYER, LATE_HEAD, tool_token_id).cpu()

    progress = tqdm(pair_batches, desc="Held-out baseline cache", dynamic_ncols=True)
    for batch in progress:
        clean_tokens = batch.clean_tokens_cpu.to(model.W_U.device)
        corrupt_tokens = batch.corrupt_tokens_cpu.to(model.W_U.device)
        clean_capture: dict[str, torch.Tensor] = {}
        corrupt_capture: dict[str, torch.Tensor] = {}

        clean_hooks = [
            (resid_hook_name, make_resid_last_capture(clean_capture, "resid")),
            (h9_pattern_hook, make_pattern_lastrow_capture(clean_capture, "h9_pattern", H9_HEAD, n_heads)),
            (late_pattern_hook, make_pattern_lastrow_capture(clean_capture, "late_pattern", LATE_HEAD, n_heads)),
            (late_z_hook, make_head_z_capture(clean_capture, "late_z", LATE_HEAD)),
        ]
        corrupt_hooks = [
            (resid_hook_name, make_resid_last_capture(corrupt_capture, "resid")),
            (h9_pattern_hook, make_pattern_lastrow_capture(corrupt_capture, "h9_pattern", H9_HEAD, n_heads)),
            (late_pattern_hook, make_pattern_lastrow_capture(corrupt_capture, "late_pattern", LATE_HEAD, n_heads)),
            (late_z_hook, make_head_z_capture(corrupt_capture, "late_z", LATE_HEAD)),
        ]
        with torch.no_grad():
            clean_logits = model.run_with_hooks(clean_tokens, fwd_hooks=clean_hooks)
            corrupt_logits = model.run_with_hooks(corrupt_tokens, fwd_hooks=corrupt_hooks)

        clean_resid[batch.indices] = clean_capture["resid"]
        corrupt_resid[batch.indices] = corrupt_capture["resid"]

        clean_tool_logit, clean_top1, clean_margin = tool_stats_with_margin(clean_logits, tool_token_id)
        corrupt_tool_logit, corrupt_top1, corrupt_margin = tool_stats_with_margin(corrupt_logits, tool_token_id)
        clean_late_dla = torch.einsum("bd,d->b", clean_capture["late_z"], late_proj)
        corrupt_late_dla = torch.einsum("bd,d->b", corrupt_capture["late_z"], late_proj)

        baseline_clean["tool_logit"][batch.indices] = clean_tool_logit
        baseline_clean["top1"][batch.indices] = clean_top1
        baseline_clean["margin"][batch.indices] = clean_margin
        baseline_clean["system_attn_h9"][batch.indices] = _system_attn_tensor(
            clean_capture["h9_pattern"],
            samples,
            batch.indices,
            eval_side="clean",
        )
        baseline_clean["l33h29_system_attn"][batch.indices] = _system_attn_tensor(
            clean_capture["late_pattern"],
            samples,
            batch.indices,
            eval_side="clean",
        )
        baseline_clean["l33h29_dla"][batch.indices] = clean_late_dla.float()

        baseline_corrupt["tool_logit"][batch.indices] = corrupt_tool_logit
        baseline_corrupt["top1"][batch.indices] = corrupt_top1
        baseline_corrupt["margin"][batch.indices] = corrupt_margin
        baseline_corrupt["system_attn_h9"][batch.indices] = _system_attn_tensor(
            corrupt_capture["h9_pattern"],
            samples,
            batch.indices,
            eval_side="corrupt",
        )
        baseline_corrupt["l33h29_system_attn"][batch.indices] = _system_attn_tensor(
            corrupt_capture["late_pattern"],
            samples,
            batch.indices,
            eval_side="corrupt",
        )
        baseline_corrupt["l33h29_dla"][batch.indices] = corrupt_late_dla.float()
        clear_cuda()

    if cache_path is not None:
        ensure_dir(cache_path.parent)
        torch.save(
            {
                "sample_ids": sample_ids,
                "patch_layer": int(patch_layer),
                "clean_resid": clean_resid,
                "corrupt_resid": corrupt_resid,
                "baseline_clean": baseline_clean,
                "baseline_corrupt": baseline_corrupt,
            },
            cache_path,
        )

    return PairBaselineCache(
        samples=samples,
        pair_batches=pair_batches,
        clean_resid=clean_resid,
        corrupt_resid=corrupt_resid,
        baseline_clean=baseline_clean,
        baseline_corrupt=baseline_corrupt,
    )


def evaluate_intervention(
    model,
    baseline_cache: PairBaselineCache,
    *,
    tool_token_id: int,
    eval_side: str,
    layer: int,
    position_mode: str,
    delta_resolver: Callable[[Sequence[int]], torch.Tensor],
    direction: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    if eval_side not in {"clean", "corrupt"}:
        raise ValueError(f"Unsupported eval_side={eval_side!r}")

    n_samples = len(baseline_cache.samples)
    outputs = _empty_metric_dict(n_samples)
    outputs["direction_score"] = torch.full((n_samples,), float("nan"), dtype=torch.float32)

    hook_name = f"blocks.{layer}.hook_resid_pre"
    h9_pattern_hook = f"blocks.{H9_LAYER}.attn.hook_pattern"
    late_pattern_hook = f"blocks.{LATE_LAYER}.attn.hook_pattern"
    late_z_hook = f"blocks.{LATE_LAYER}.attn.hook_z"
    n_heads = int(model.cfg.n_heads)
    late_proj = precompute_head_projection(model, LATE_LAYER, LATE_HEAD, tool_token_id).cpu()

    progress = tqdm(
        baseline_cache.pair_batches,
        desc=f"Eval {eval_side} L{layer} {position_mode}",
        dynamic_ncols=True,
    )
    for batch in progress:
        tokens_cpu = batch.clean_tokens_cpu if eval_side == "clean" else batch.corrupt_tokens_cpu
        tokens = tokens_cpu.to(model.W_U.device)
        capture: dict[str, torch.Tensor] = {}
        delta_batch = delta_resolver(batch.indices).float()
        hooks = [
            (hook_name, make_position_resid_add_hook(delta_batch, position_mode)),
            (h9_pattern_hook, make_pattern_lastrow_capture(capture, "h9_pattern", H9_HEAD, n_heads)),
            (late_pattern_hook, make_pattern_lastrow_capture(capture, "late_pattern", LATE_HEAD, n_heads)),
            (late_z_hook, make_head_z_capture(capture, "late_z", LATE_HEAD)),
        ]
        with torch.no_grad():
            logits = model.run_with_hooks(tokens, fwd_hooks=hooks)

        tool_logit, top1, margin = tool_stats_with_margin(logits, tool_token_id)
        late_dla = torch.einsum("bd,d->b", capture["late_z"], late_proj)
        outputs["tool_logit"][batch.indices] = tool_logit
        outputs["top1"][batch.indices] = top1
        outputs["margin"][batch.indices] = margin
        outputs["system_attn_h9"][batch.indices] = _system_attn_tensor(
            capture["h9_pattern"],
            baseline_cache.samples,
            batch.indices,
            eval_side=eval_side,
        )
        outputs["l33h29_system_attn"][batch.indices] = _system_attn_tensor(
            capture["late_pattern"],
            baseline_cache.samples,
            batch.indices,
            eval_side=eval_side,
        )
        outputs["l33h29_dla"][batch.indices] = late_dla.float()

        if direction is not None and int(layer) == int(PATCH_LAYER):
            base_resid = baseline_cache.clean_resid[batch.indices] if eval_side == "clean" else baseline_cache.corrupt_resid[batch.indices]
            if position_mode in {"last", "all"}:
                outputs["direction_score"][batch.indices] = direction_scores(base_resid + delta_batch, direction)
            else:
                outputs["direction_score"][batch.indices] = direction_scores(base_resid, direction)
        clear_cuda()
    return outputs


def baseline_summary_row(
    *,
    condition: str,
    condition_family: str,
    eval_side: str,
    baseline_metrics: dict[str, torch.Tensor],
    tool_token_id: int,
    direction_score: torch.Tensor | None = None,
    layer: int = PATCH_LAYER,
    position_mode: str = "last",
    alpha: float | None = None,
    direction_name: str = "pc1",
) -> dict[str, object]:
    outputs = {key: value for key, value in baseline_metrics.items()}
    outputs["direction_score"] = (
        direction_score.clone().float()
        if direction_score is not None
        else torch.full((baseline_metrics["top1"].shape[0],), float("nan"), dtype=torch.float32)
    )
    row = summarize_condition_row(
        condition=condition,
        condition_family=condition_family,
        eval_side=eval_side,
        outputs=outputs,
        baseline_metrics=baseline_metrics,
        tool_token_id=tool_token_id,
        layer=layer,
        position_mode=position_mode,
        alpha=alpha,
        direction_name=direction_name,
    )
    return row


def nanmean_tensor(values: torch.Tensor) -> float:
    finite = values[torch.isfinite(values)]
    if finite.numel() == 0:
        return float("nan")
    return float(finite.float().mean().item())


def summarize_condition_row(
    *,
    condition: str,
    condition_family: str,
    eval_side: str,
    outputs: dict[str, torch.Tensor],
    baseline_metrics: dict[str, torch.Tensor],
    tool_token_id: int,
    layer: int,
    position_mode: str,
    alpha: float | None,
    direction_name: str,
) -> dict[str, object]:
    top1_is_tool = outputs["top1"] == tool_token_id
    baseline_is_tool = baseline_metrics["top1"] == tool_token_id
    row: dict[str, object] = {
        "condition": condition,
        "condition_family": condition_family,
        "eval_side": eval_side,
        "direction_name": direction_name,
        "layer": int(layer),
        "position_mode": position_mode,
        "alpha": "" if alpha is None else float(alpha),
        "n_samples": int(outputs["top1"].shape[0]),
        "tool_call_top1_rate": float(top1_is_tool.float().mean().item()),
        "mean_tool_logit": float(outputs["tool_logit"].mean().item()),
        "mean_margin": float(outputs["margin"].mean().item()),
        "mean_system_attn_h9": float(outputs["system_attn_h9"].mean().item()),
        "mean_l33h29_system_attn": float(outputs["l33h29_system_attn"].mean().item()),
        "mean_l33h29_dla": float(outputs["l33h29_dla"].mean().item()),
        "mean_direction_score": nanmean_tensor(outputs["direction_score"]),
        "delta_tool_logit_vs_baseline": float((outputs["tool_logit"] - baseline_metrics["tool_logit"]).mean().item()),
        "delta_margin_vs_baseline": float((outputs["margin"] - baseline_metrics["margin"]).mean().item()),
        "delta_system_attn_h9_vs_baseline": float(
            (outputs["system_attn_h9"] - baseline_metrics["system_attn_h9"]).mean().item()
        ),
        "delta_l33h29_system_attn_vs_baseline": float(
            (outputs["l33h29_system_attn"] - baseline_metrics["l33h29_system_attn"]).mean().item()
        ),
        "delta_l33h29_dla_vs_baseline": float((outputs["l33h29_dla"] - baseline_metrics["l33h29_dla"]).mean().item()),
    }
    if eval_side == "corrupt":
        row["strict_flip_rate"] = float(((~baseline_is_tool) & top1_is_tool).float().mean().item())
        row["strict_drop_rate"] = ""
    else:
        row["strict_flip_rate"] = ""
        row["strict_drop_rate"] = float((baseline_is_tool & (~top1_is_tool)).float().mean().item())
    return row


def best_row(
    rows: Sequence[dict[str, object]],
    *,
    condition_family: str,
    metric: str,
    maximize: bool = True,
) -> dict[str, object]:
    filtered = [row for row in rows if str(row["condition_family"]) == condition_family]
    if not filtered:
        raise ValueError(f"No rows found for condition_family={condition_family!r}")
    return max(filtered, key=lambda row: float(row[metric])) if maximize else min(filtered, key=lambda row: float(row[metric]))


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def pick_headline_metrics(row: dict[str, object]) -> tuple[float, float]:
    top1 = float(row["tool_call_top1_rate"])
    if row.get("strict_flip_rate", "") not in {"", None}:
        effect = float(row["strict_flip_rate"])
    else:
        effect = float(row["strict_drop_rate"])
    return top1, effect


def close_figures() -> None:
    gc.collect()
