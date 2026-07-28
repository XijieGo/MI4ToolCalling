#!/usr/bin/env python3
"""
ACDC-inspired circuit mining for Qwen3-1.7B tool-call behavior on one sample.

This entry point now supports both:
- the new dataset-root layout (`datasets/clean` + `datasets/corrupt`);
- the legacy pair-style layout from the reference project.
"""

from __future__ import annotations

import os
import argparse
import gc
import importlib.util
import inspect
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import FancyArrowPatch
from tqdm.auto import tqdm


_ORIGINAL_FIND_SPEC = importlib.util.find_spec


def _patched_find_spec(name: str, package: str | None = None):
    # Some environments ship a broken sklearn/scipy combo that makes
    # `transformers` fail at import time when it probes candidate generators.
    # We do not use sklearn anywhere in this codepath, so disable that optional
    # backend unless the caller explicitly opts back in.
    if name == "sklearn" and os.environ.get("TOOLCALL_ENABLE_TRANSFORMERS_SKLEARN", "0") != "1":
        return None
    return _ORIGINAL_FIND_SPEC(name, package)


importlib.util.find_spec = _patched_find_spec
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers import logging as hf_logging
    from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
finally:
    importlib.util.find_spec = _ORIGINAL_FIND_SPEC

from transformer_lens import HookedTransformer
import transformer_lens.loading_from_pretrained as tl_loading

if __package__ in {None, ""}:
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from toolcall_circuit.dataset import (
    ToolCallSample,
    build_position_sets,
    decode_token_at,
    get_tool_call_target_spec,
    load_toolcall_samples,
    resolve_distractor_token,
    select_samples,
)
from toolcall_circuit.objective import (
    DistributionObjective,
    build_distribution_objective,
    objective_from_logits,
    objective_vector_from_logits,
    summarize_endpoint_pair,
)
from toolcall_circuit.paths import DATASETS_ROOT, MODEL_PATH_DEFAULT, RESULTS_ROOT


# -----------------------------
# Model + token helpers
# -----------------------------


def patch_qwen3_rope_theta() -> None:
    """Backfill `rope_theta` only for older Qwen3 configs that don't expose it."""
    if "rope_theta" in inspect.signature(Qwen3Config.__init__).parameters:
        return

    def _get_rope_theta(self) -> float:
        stored = getattr(self, "_acdc_rope_theta", None)
        if stored is not None:
            return stored
        return (
            (getattr(self, "rope_scaling", None) or {}).get("rope_theta")
            or (getattr(self, "rope_parameters", None) or {}).get("rope_theta")
            or 1_000_000
        )

    def _set_rope_theta(self, value: float) -> None:
        self._acdc_rope_theta = value

    Qwen3Config.rope_theta = property(_get_rope_theta, _set_rope_theta)  # type: ignore[attr-defined]


def load_hooked_qwen3(model_path: str, device: str, dtype: torch.dtype) -> Tuple[HookedTransformer, AutoTokenizer]:
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    hf_logging.set_verbosity_error()
    try:
        hf_logging.disable_progress_bar()
    except Exception:
        pass

    patch_qwen3_rope_theta()

    hf_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map="cpu",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.bos_token is None:
        tokenizer.bos_token = "<|endoftext|>"
    tokenizer.add_bos_token = True

    if Path(model_path).exists():
        cfg = tl_loading.get_pretrained_model_config(
            model_path,
            hf_cfg=hf_model.config.to_dict(),
            fold_ln=False,
            device=device,
            n_devices=1,
            dtype=dtype,
            trust_remote_code=True,
        )
        state_dict = tl_loading.get_pretrained_state_dict(
            model_path,
            cfg,
            hf_model=hf_model,
            dtype=dtype,
            trust_remote_code=True,
        )
        model = HookedTransformer(
            cfg,
            tokenizer,
            move_to_device=False,
            default_padding_side="right",
        )
        model.load_and_process_state_dict(
            state_dict,
            fold_ln=False,
            center_writing_weights=False,
            center_unembed=False,
        )
        model.move_model_modules_to_device()
    else:
        model = HookedTransformer.from_pretrained(
            model_path,
            hf_model=hf_model,
            tokenizer=tokenizer,
            device=device,
            dtype=dtype,
            fold_ln=False,
            center_writing_weights=False,
            center_unembed=False,
            trust_remote_code=True,
        )
    # `move_model_modules_to_device()` is not always sufficient in this environment;
    # force the final TransformerLens module onto the requested runtime device.
    model.to(device, print_details=False)
    # These workflows only read `hook_z` and `hook_mlp_out`; the heavier
    # attention-result and split-qkv hooks materially increase activation
    # memory for long prompts without improving the current analyses.
    model.set_use_attn_result(False)
    model.set_use_split_qkv_input(False)
    model.set_use_hook_mlp_in(False)
    model.eval()

    # Free HF model copy ASAP.
    del hf_model
    gc.collect()
    torch.cuda.empty_cache()

    return model, tokenizer


def node_layer(node_name: str) -> int:
    if node_name.startswith("MLP"):
        return int(node_name[3:])
    m = re.fullmatch(r"L(\d+)H(\d+)", node_name)
    if m:
        return int(m.group(1))
    raise ValueError(f"Unknown node name format: {node_name}")


def parse_head(node_name: str) -> Tuple[int, int]:
    m = re.fullmatch(r"L(\d+)H(\d+)", node_name)
    if m is None:
        raise ValueError(f"Not a head node: {node_name}")
    return int(m.group(1)), int(m.group(2))


def normalize_head_score_mode(raw: str) -> str:
    mode = (raw or "").strip().lower()
    aliases = {
        "ap": "ap",
        "ap_proxy": "ap",
        "exact": "exact_patch",
        "exact_patch": "exact_patch",
    }
    if mode not in aliases:
        raise ValueError(f"Unknown head score mode: {raw}")
    return aliases[mode]


# -----------------------------
# Patching evaluation
# -----------------------------


def collect_clean_cache_cpu(
    model: HookedTransformer, clean_tokens: torch.Tensor
) -> Dict[str, torch.Tensor]:
    names_filter = lambda n: n.endswith("attn.hook_z") or n.endswith("hook_mlp_out")
    with torch.no_grad():
        _, cache_gpu = model.run_with_cache(clean_tokens, names_filter=names_filter)
    cache_cpu = {k: v.detach().cpu() for k, v in cache_gpu.items()}
    del cache_gpu
    torch.cuda.empty_cache()
    return cache_cpu


def evaluate_on_base_with_source(
    model: HookedTransformer,
    base_tokens: torch.Tensor,
    source_cache_cpu: Dict[str, torch.Tensor],
    patch_nodes: Sequence[str],
    target_token: int | DistributionObjective,
    distractor_token: int | None = None,
) -> float:
    heads_by_layer: Dict[int, List[int]] = {}
    mlp_layers: List[int] = []

    for node in patch_nodes:
        if node.startswith("MLP"):
            mlp_layers.append(int(node[3:]))
        else:
            layer, head = parse_head(node)
            heads_by_layer.setdefault(layer, []).append(head)

    hooks = []

    for layer, heads in heads_by_layer.items():
        cache_name = f"blocks.{layer}.attn.hook_z"
        clean_act = source_cache_cpu[cache_name].to(base_tokens.device)
        heads = sorted(set(heads))

        def make_head_hook(src: torch.Tensor, hs: Sequence[int]):
            def hook_fn(z: torch.Tensor, hook):  # noqa: ANN001
                z = z.clone()
                for h in hs:
                    # Patch only the next-token prediction position to avoid full-sequence leakage.
                    z[:, -1, h, :] = src[:, -1, h, :]
                return z

            return hook_fn

        hooks.append((cache_name, make_head_hook(clean_act, heads)))

    for layer in sorted(set(mlp_layers)):
        cache_name = f"blocks.{layer}.hook_mlp_out"
        clean_act = source_cache_cpu[cache_name].to(base_tokens.device)

        def make_mlp_hook(src: torch.Tensor):
            def hook_fn(mlp_out: torch.Tensor, hook):  # noqa: ANN001
                mlp_out = mlp_out.clone()
                mlp_out[:, -1, :] = src[:, -1, :]
                return mlp_out

            return hook_fn

        hooks.append((cache_name, make_mlp_hook(clean_act)))

    with torch.no_grad():
        logits = model.run_with_hooks(base_tokens, fwd_hooks=hooks)
    return float(objective_from_logits(logits, target_token, distractor_token).item())


def compute_exact_patch_head_gain(
    model: HookedTransformer,
    corrupt_tokens: torch.Tensor,
    clean_cache_cpu: Dict[str, torch.Tensor],
    target_token: int | DistributionObjective,
    distractor_token: int | None,
    corrupt_obj: float,
    head_batch_size: int = 0,
    candidate_heads_by_layer: Optional[Dict[int, Sequence[int]]] = None,
    fallback_scores: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    if fallback_scores is not None:
        scores = fallback_scores.detach().float().cpu().clone()
    else:
        scores = torch.zeros(n_layers, n_heads, dtype=torch.float32)

    if candidate_heads_by_layer is None:
        layer_sequence = list(range(n_layers))
    else:
        layer_sequence = [layer for layer in range(n_layers) if candidate_heads_by_layer.get(layer)]

    for layer in tqdm(layer_sequence, desc="Exact head patch", leave=False):
        cache_name = f"blocks.{layer}.attn.hook_z"
        clean_act = clean_cache_cpu[cache_name].to(corrupt_tokens.device)
        layer_heads = (
            list(range(n_heads))
            if candidate_heads_by_layer is None
            else [int(head) for head in candidate_heads_by_layer.get(layer, ())]
        )
        if not layer_heads:
            continue
        start = 0
        cur_batch_size = len(layer_heads) if head_batch_size <= 0 else max(1, min(len(layer_heads), int(head_batch_size)))
        while start < len(layer_heads):
            batch_heads = layer_heads[start : min(start + cur_batch_size, len(layer_heads))]
            batch_tokens = corrupt_tokens.repeat(len(batch_heads), 1)

            def make_head_hook(src: torch.Tensor, heads: Sequence[int]):
                def hook_fn(z: torch.Tensor, hook):  # noqa: ANN001
                    out = z.clone()
                    for batch_idx, head in enumerate(heads):
                        out[batch_idx, -1, head, :] = src[0, -1, head, :].to(out.dtype)
                    return out

                return hook_fn

            try:
                with torch.no_grad():
                    logits = model.run_with_hooks(batch_tokens, fwd_hooks=[(cache_name, make_head_hook(clean_act, batch_heads))])
                obj = objective_vector_from_logits(logits, target_token, distractor_token)
                for batch_idx, head in enumerate(batch_heads):
                    scores[layer, head] = float(obj[batch_idx].item()) - corrupt_obj
                start += len(batch_heads)
            except torch.OutOfMemoryError:
                gc.collect()
                torch.cuda.empty_cache()
                if cur_batch_size <= 1:
                    raise
                cur_batch_size = max(1, cur_batch_size // 2)

    return scores


def compute_ct_head_gain(
    model: HookedTransformer,
    corrupt_tokens: torch.Tensor,
    clean_cache_cpu: Dict[str, torch.Tensor],
    target_token: int | DistributionObjective,
    distractor_token: int | None,
    corrupt_obj: float,
    head_batch_size: int = 0,
    candidate_heads_by_layer: Optional[Dict[int, Sequence[int]]] = None,
    fallback_scores: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Backward-compatible alias for exact clean->corrupt head patch scoring."""
    return compute_exact_patch_head_gain(
        model=model,
        corrupt_tokens=corrupt_tokens,
        clean_cache_cpu=clean_cache_cpu,
        target_token=target_token,
        distractor_token=distractor_token,
        corrupt_obj=corrupt_obj,
        head_batch_size=head_batch_size,
        candidate_heads_by_layer=candidate_heads_by_layer,
        fallback_scores=fallback_scores,
    )


def select_ct_candidate_heads(
    ap_head: torch.Tensor,
    *,
    ap_top_per_layer: int,
    ap_top_global: int,
) -> Optional[Dict[int, List[int]]]:
    if ap_top_per_layer <= 0 and ap_top_global <= 0:
        return None

    n_layers, n_heads = ap_head.shape
    chosen: set[Tuple[int, int]] = set()

    if ap_top_per_layer > 0:
        for layer in range(n_layers):
            layer_rank = [
                (float(ap_head[layer, head].item()), head)
                for head in range(n_heads)
                if float(ap_head[layer, head].item()) > 0.0
            ]
            layer_rank.sort(key=lambda x: x[0], reverse=True)
            for _, head in layer_rank[:ap_top_per_layer]:
                chosen.add((layer, head))

    if ap_top_global > 0:
        global_rank = [
            (float(ap_head[layer, head].item()), layer, head)
            for layer in range(n_layers)
            for head in range(n_heads)
            if float(ap_head[layer, head].item()) > 0.0
        ]
        global_rank.sort(key=lambda x: x[0], reverse=True)
        for _, layer, head in global_rank[:ap_top_global]:
            chosen.add((layer, head))

    if not chosen:
        return None

    by_layer: Dict[int, List[int]] = {}
    for layer, head in sorted(chosen):
        by_layer.setdefault(layer, []).append(head)
    return by_layer


def compute_mlp_gain(
    model: HookedTransformer,
    corrupt_tokens: torch.Tensor,
    clean_cache_cpu: Dict[str, torch.Tensor],
    target_token: int | DistributionObjective,
    distractor_token: int | None,
    corrupt_obj: float,
    layer_batch_size: int = 0,
) -> torch.Tensor:
    n_layers = model.cfg.n_layers
    gains = torch.zeros(n_layers, dtype=torch.float32)
    start = 0
    cur_batch_size = n_layers if layer_batch_size <= 0 else max(1, min(n_layers, int(layer_batch_size)))

    with tqdm(total=n_layers, desc="Exact MLP patch", leave=False) as pbar:
        while start < n_layers:
            batch_layers = list(range(start, min(start + cur_batch_size, n_layers)))
            batch_tokens = corrupt_tokens.repeat(len(batch_layers), 1)
            hooks = []

            for batch_idx, layer in enumerate(batch_layers):
                cache_name = f"blocks.{layer}.hook_mlp_out"
                clean_act = clean_cache_cpu[cache_name].to(corrupt_tokens.device)

                def make_hook(src: torch.Tensor, row_idx: int):
                    def hook_fn(mlp_out: torch.Tensor, hook):  # noqa: ANN001
                        out = mlp_out.clone()
                        out[row_idx, -1, :] = src[0, -1, :].to(out.dtype)
                        return out

                    return hook_fn

                hooks.append((cache_name, make_hook(clean_act, batch_idx)))

            try:
                with torch.no_grad():
                    logits = model.run_with_hooks(batch_tokens, fwd_hooks=hooks)
                obj = objective_vector_from_logits(logits, target_token, distractor_token)
                for batch_idx, layer in enumerate(batch_layers):
                    gains[layer] = float(obj[batch_idx].item()) - corrupt_obj
                start += len(batch_layers)
                pbar.update(len(batch_layers))
            except torch.OutOfMemoryError:
                gc.collect()
                torch.cuda.empty_cache()
                if cur_batch_size <= 1:
                    raise
                cur_batch_size = max(1, cur_batch_size // 2)

    return gains


def compute_ap_head_gain(
    model: HookedTransformer,
    corrupt_tokens: torch.Tensor,
    clean_cache_cpu: Dict[str, torch.Tensor],
    target_token: int | DistributionObjective,
    distractor_token: int | None,
) -> torch.Tensor:
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    z_store: Dict[str, torch.Tensor] = {}

    def z_hook(z: torch.Tensor, hook):  # noqa: ANN001
        # Some callers leave global grad disabled; make the hooked tensor explicit.
        if not z.requires_grad:
            z = z.detach().requires_grad_(True)
        z.retain_grad()
        z_store[hook.name] = z
        return z

    model.reset_hooks()
    model.add_hook(lambda n: n.endswith("attn.hook_z"), z_hook)

    with torch.enable_grad():
        model.zero_grad(set_to_none=True)
        logits = model(corrupt_tokens)
        obj = objective_from_logits(logits, target_token, distractor_token)
        obj.backward()

    ap = torch.zeros(n_layers, n_heads, dtype=torch.float32)
    for layer in range(n_layers):
        name = f"blocks.{layer}.attn.hook_z"
        z = z_store[name]
        grad = z.grad
        delta = clean_cache_cpu[name].to(z.device) - z.detach()
        # Position-local AP at the prediction position.
        ap[layer] = (grad[:, -1, :, :] * delta[:, -1, :, :]).sum(dim=(0, 2)).float().cpu()

    model.reset_hooks()
    torch.cuda.empty_cache()
    return ap


def compute_ap_head_gain_lowmem(
    model: HookedTransformer,
    corrupt_tokens: torch.Tensor,
    clean_cache_cpu: Dict[str, torch.Tensor],
    target_token: int | DistributionObjective,
    distractor_token: int | None,
) -> torch.Tensor:
    """
    Low-memory fallback for AP.
    Computes gradient*delta per layer by running backward once per layer.
    """
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    ap = torch.zeros(n_layers, n_heads, dtype=torch.float32)

    for layer in tqdm(range(n_layers), desc="AP lowmem", leave=False):
        name = f"blocks.{layer}.attn.hook_z"
        z_store: Dict[str, torch.Tensor] = {}

        def z_hook(z: torch.Tensor, hook):  # noqa: ANN001
            if not z.requires_grad:
                z = z.detach().requires_grad_(True)
            z.retain_grad()
            z_store[hook.name] = z
            return z

        model.reset_hooks()
        try:
            model.add_hook(lambda n, target=name: n == target, z_hook)
            with torch.enable_grad():
                model.zero_grad(set_to_none=True)
                logits = model(corrupt_tokens)
                obj = objective_from_logits(logits, target_token, distractor_token)
                obj.backward()

            z = z_store[name]
            grad = z.grad
            delta = clean_cache_cpu[name].to(z.device) - z.detach()
            ap[layer] = (grad[:, -1, :, :] * delta[:, -1, :, :]).sum(dim=(0, 2)).float().cpu()
        finally:
            model.reset_hooks()
            gc.collect()
            torch.cuda.empty_cache()

    return ap


# -----------------------------
# Node/edge selection
# -----------------------------


@dataclass
class NodeScore:
    name: str
    score: float
    exact_patch: float
    ap: float


def _layer_bins(n_layers: int, n_bins: int = 4) -> List[Tuple[int, int]]:
    if n_layers <= 0:
        return [(0, 0)]
    boundaries = np.linspace(0, n_layers, num=max(2, n_bins + 1), dtype=int)
    bins: List[Tuple[int, int]] = []
    for idx in range(len(boundaries) - 1):
        lo = int(boundaries[idx])
        hi = int(boundaries[idx + 1]) - 1
        if hi < lo:
            hi = lo
        bins.append((lo, hi))
    if bins:
        last_lo, _ = bins[-1]
        bins[-1] = (last_lo, n_layers - 1)
    return bins


def _positive_quantile(values: Sequence[float], q: float, *, floor: float) -> float:
    pos = [float(v) for v in values if math.isfinite(float(v)) and float(v) > 0.0]
    if not pos:
        return float(floor)
    return max(float(floor), float(np.quantile(np.asarray(pos, dtype=np.float32), q)))


def pick_nodes(
    head_score: torch.Tensor,
    ap_head: torch.Tensor,
    mlp_gain: torch.Tensor,
    *,
    head_score_mode: str,
) -> Tuple[List[str], List[str], List[NodeScore]]:
    n_layers, n_heads = head_score.shape
    combo = 0.55 * head_score + 0.45 * ap_head

    head_rank: List[Tuple[int, int, float]] = []
    for l in range(n_layers):
        for h in range(n_heads):
            head_rank.append((l, h, float(combo[l, h].item())))
    head_rank.sort(key=lambda x: x[2], reverse=True)

    mlp_rank: List[Tuple[int, float]] = [(l, float(mlp_gain[l].item())) for l in range(n_layers)]
    mlp_rank.sort(key=lambda x: x[1], reverse=True)

    bins = _layer_bins(n_layers, n_bins=4)
    head_scores_only = [score for (_, _, score) in head_rank]
    mlp_scores_only = [score for (_, score) in mlp_rank]
    head_cut = _positive_quantile(head_scores_only, 0.82, floor=0.08)
    mlp_cut = _positive_quantile(mlp_scores_only, 0.75, floor=0.05)
    head_cap = max(10, min(16, math.ceil(n_layers * 0.30)))
    mlp_cap = max(6, min(10, math.ceil(n_layers * 0.22)))
    min_heads = max(6, min(head_cap, len(bins) * 2))
    min_mlps = max(4, min(mlp_cap, len(bins)))

    chosen_heads: List[Tuple[int, int]] = []
    for lo, hi in bins:
        bin_candidates = [(l, h, s) for (l, h, s) in head_rank if lo <= l <= hi and s >= head_cut]
        for l, h, _ in bin_candidates[:2]:
            if (l, h) not in chosen_heads:
                chosen_heads.append((l, h))
    for l, h, s in head_rank:
        if s < head_cut:
            continue
        if (l, h) not in chosen_heads:
            chosen_heads.append((l, h))
        if len(chosen_heads) >= head_cap:
            break
    if len(chosen_heads) < min_heads:
        for l, h, _ in head_rank:
            if (l, h) not in chosen_heads:
                chosen_heads.append((l, h))
            if len(chosen_heads) >= min_heads:
                break

    chosen_mlps: List[int] = []
    for lo, hi in bins:
        cand = next((l for (l, s) in mlp_rank if lo <= l <= hi and s >= mlp_cut), None)
        if cand is not None and cand not in chosen_mlps:
            chosen_mlps.append(cand)
    for l, s in mlp_rank:
        if s < mlp_cut:
            continue
        if l not in chosen_mlps:
            chosen_mlps.append(l)
        if len(chosen_mlps) >= mlp_cap:
            break
    if len(chosen_mlps) < min_mlps:
        for l, _ in mlp_rank:
            if l not in chosen_mlps:
                chosen_mlps.append(l)
            if len(chosen_mlps) >= min_mlps:
                break

    detailed_nodes = [f"MLP{l}" for l in chosen_mlps] + [f"L{l}H{h}" for l, h in chosen_heads]
    detailed_nodes = sorted(set(detailed_nodes), key=lambda n: (node_layer(n), 0 if n.startswith("MLP") else 1, n))

    score_table: List[NodeScore] = []
    for name in detailed_nodes:
        if name.startswith("MLP"):
            l = int(name[3:])
            score_table.append(
                NodeScore(
                    name=name,
                    score=float(mlp_gain[l].item()),
                    exact_patch=float(mlp_gain[l].item()),
                    ap=0.0,
                )
            )
        else:
            l, h = parse_head(name)
            score_table.append(
                NodeScore(
                    name=name,
                    score=float(combo[l, h].item()),
                    exact_patch=(
                        float(head_score[l, h].item())
                        if normalize_head_score_mode(head_score_mode) == "exact_patch"
                        else float("nan")
                    ),
                    ap=float(ap_head[l, h].item()),
                )
            )
    score_table.sort(key=lambda x: x.score, reverse=True)

    # Keep a compact but branched rough circuit.
    score_lookup = {s.name: s.score for s in score_table}
    rough_set = set()
    must_keep = set()
    if detailed_nodes:
        rough_set.add(detailed_nodes[0])
        rough_set.add(detailed_nodes[-1])
        must_keep.add(detailed_nodes[0])
        must_keep.add(detailed_nodes[-1])
        early_layer = min(node_layer(n) for n in detailed_nodes)
        early_nodes = [n for n in detailed_nodes if node_layer(n) == early_layer]
        early_nodes = sorted(early_nodes, key=lambda n: score_lookup.get(n, 0.0), reverse=True)
        rough_set.update(early_nodes[:2])
        must_keep.update(early_nodes[:2])

    top_heads = [s.name for s in score_table if s.name.startswith("L")]
    top_mlps = [s.name for s in score_table if s.name.startswith("MLP")]
    rough_set.update(top_heads[:3])
    rough_set.update(top_mlps[:3])

    for lo, hi in bins:
        bin_nodes = [n for n in detailed_nodes if lo <= node_layer(n) <= hi]
        if bin_nodes:
            best = max(bin_nodes, key=lambda n: score_lookup.get(n, 0.0))
            rough_set.add(best)

    # Prefer at least one duplicated-layer pair (visible branch), then top up by score.
    for l in sorted({node_layer(n) for n in detailed_nodes}):
        same_layer = [n for n in detailed_nodes if node_layer(n) == l]
        if len(same_layer) >= 2:
            same_layer = sorted(same_layer, key=lambda n: score_lookup.get(n, 0.0), reverse=True)
            rough_set.update(same_layer[:2])
            break

    for s in score_table:
        if len(rough_set) >= 8:
            break
        rough_set.add(s.name)

    rough_nodes = sorted(rough_set, key=lambda n: (node_layer(n), 0 if n.startswith("MLP") else 1, n))
    if len(rough_nodes) > 8:
        keep = set(must_keep)
        if len(keep) > 8:
            mandatory_scored = sorted(keep, key=lambda n: score_lookup.get(n, 0.0), reverse=True)
            keep = set(mandatory_scored[:8])
        for s in score_table:
            if len(keep) >= 8:
                break
            if s.name in rough_set:
                keep.add(s.name)
        rough_nodes = sorted(keep, key=lambda n: (node_layer(n), 0 if n.startswith("MLP") else 1, n))

    return detailed_nodes, rough_nodes, score_table


def build_edges(nodes: Sequence[str], score_lookup: Dict[str, float], max_parents: int = 2) -> List[Tuple[str, str]]:
    input_node = "Input Embed"
    output_node = "Residual Output: <tool_call>"

    sorted_nodes = sorted(nodes, key=lambda n: (node_layer(n), 0 if n.startswith("MLP") else 1, n))
    if not sorted_nodes:
        return [(input_node, output_node)]

    edges: List[Tuple[str, str]] = []

    # Input fans into 1-2 earliest anchors.
    earliest_layer = min(node_layer(n) for n in sorted_nodes)
    earliest_nodes = [n for n in sorted_nodes if node_layer(n) == earliest_layer]
    earliest_nodes = sorted(earliest_nodes, key=lambda n: score_lookup.get(n, 0.0), reverse=True)
    for n in earliest_nodes[:2]:
        edges.append((input_node, n))
    if len(earliest_nodes) == 1 and len(sorted_nodes) > 1:
        edges.append((input_node, sorted_nodes[1]))

    # Multi-parent DAG edges based on score + distance decay.
    for idx, target in enumerate(sorted_nodes):
        t_layer = node_layer(target)
        candidates = [n for n in sorted_nodes[:idx] if node_layer(n) < t_layer]
        if not candidates:
            continue

        ranked: List[Tuple[float, str]] = []
        for src in candidates:
            gap = max(1, t_layer - node_layer(src))
            type_bonus = 0.18 if (src.startswith("MLP") != target.startswith("MLP")) else 0.0
            rank_score = score_lookup.get(src, 0.0) / (1.0 + 0.35 * gap) + type_bonus
            ranked.append((rank_score, src))
        ranked.sort(key=lambda x: x[0], reverse=True)
        for _, src in ranked[:max_parents]:
            edges.append((src, target))

    # Multiple late nodes feed output.
    latest_sorted = sorted(
        sorted_nodes,
        key=lambda n: (node_layer(n), score_lookup.get(n, 0.0)),
        reverse=True,
    )
    output_parents: List[str] = []
    seen_layers = set()
    for n in latest_sorted:
        l = node_layer(n)
        if l in seen_layers and len(output_parents) >= 1:
            continue
        output_parents.append(n)
        seen_layers.add(l)
        if len(output_parents) >= 3:
            break
    for n in output_parents:
        edges.append((n, output_node))

    def dedup_edges(edge_list: Sequence[Tuple[str, str]]) -> List[Tuple[str, str]]:
        out: List[Tuple[str, str]] = []
        seen = set()
        for e in edge_list:
            if e not in seen:
                out.append(e)
                seen.add(e)
        return out

    edges = dedup_edges(edges)

    # Connectivity repair from Input.
    adjacency: Dict[str, List[str]] = {}
    for s, t in edges:
        adjacency.setdefault(s, []).append(t)

    reached = {input_node}
    queue = [input_node]
    while queue:
        cur = queue.pop(0)
        for nxt in adjacency.get(cur, []):
            if nxt not in reached:
                reached.add(nxt)
                queue.append(nxt)

    for n in sorted_nodes:
        if n in reached:
            continue
        preds = [p for p in sorted_nodes if node_layer(p) < node_layer(n) and p in reached]
        if preds:
            best_pred = max(preds, key=lambda x: score_lookup.get(x, 0.0))
            edges.append((best_pred, n))
        else:
            edges.append((input_node, n))

    edges = dedup_edges(edges)

    # Enforce out-degree for all non-output nodes.
    outdeg: Dict[str, int] = {}
    for s, _ in edges:
        outdeg[s] = outdeg.get(s, 0) + 1

    for n in [input_node] + sorted_nodes:
        if outdeg.get(n, 0) > 0:
            continue
        later = [m for m in sorted_nodes if node_layer(m) > node_layer(n)]
        if later:
            target = max(
                later,
                key=lambda m: score_lookup.get(m, 0.0) - 0.18 * (node_layer(m) - node_layer(n)),
            )
            edges.append((n, target))
        else:
            edges.append((n, output_node))

    return dedup_edges(edges)


def assert_no_dead_ends(nodes: Sequence[str], edges: Sequence[Tuple[str, str]]) -> None:
    input_node = "Input Embed"
    output_node = "Residual Output: <tool_call>"
    outdeg: Dict[str, int] = {}
    for s, _ in edges:
        outdeg[s] = outdeg.get(s, 0) + 1
    dead_nodes = [n for n in [input_node] + list(nodes) if n != output_node and outdeg.get(n, 0) == 0]
    if dead_nodes:
        raise RuntimeError(f"Graph contains dead-end non-output nodes: {dead_nodes}")


# -----------------------------
# Plotting
# -----------------------------


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


def plot_head_heatmap(data: np.ndarray, title: str, out_path: Path) -> None:
    apply_plot_style()

    clip = float(np.percentile(np.abs(data), 98))
    clip = max(clip, 1e-6)

    fig, ax = plt.subplots(figsize=(10.5, 6.2), constrained_layout=True)
    im = ax.imshow(data, cmap="RdBu_r", vmin=-clip, vmax=clip, aspect="auto", origin="lower")
    ax.set_title(f"{title}\nSymmetric clipping at ±{clip:.3f}")
    ax.set_xlabel("Head")
    ax.set_ylabel("Layer")
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label("Contribution (positive = supports the endpoint objective)")
    fig.savefig(out_path, dpi=260, bbox_inches="tight")
    plt.close(fig)


def plot_probe(
    out_path: Path,
    head_name: str,
    corrupt_obj: float,
    corrupt_with_head: float,
    clean_obj: float,
    clean_with_head_corrupt: float,
) -> None:
    labels = [
        "Corrupt baseline",
        f"Corrupt + {head_name}(clean)",
        "Clean baseline",
        f"Clean + {head_name}(corrupt)",
    ]
    vals = [corrupt_obj, corrupt_with_head, clean_obj, clean_with_head_corrupt]
    colors = ["#b35d5d", "#4f81bd", "#5a9f6f", "#c08a4f"]

    apply_plot_style()
    plt.rcParams.update({"axes.titlesize": 15})

    fig, ax = plt.subplots(figsize=(9.6, 5.4), constrained_layout=True)
    bars = ax.bar(np.arange(len(vals)), vals, color=colors, edgecolor="black", linewidth=1.1)
    ax.axhline(0.0, color="black", linewidth=1.0)
    ax.set_title(f"Component Probe: {head_name}")
    ax.set_ylabel("Endpoint score")
    ax.set_xticks(np.arange(len(vals)))
    ax.set_xticklabels(labels, rotation=14, ha="right")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + (0.08 if v >= 0 else -0.08), f"{v:.2f}", ha="center", va="bottom" if v >= 0 else "top")
    fig.savefig(out_path, dpi=260, bbox_inches="tight")
    plt.close(fig)


def draw_circuit(
    nodes: Sequence[str],
    edges: Sequence[Tuple[str, str]],
    out_path: Path,
    title: str,
) -> None:
    input_node = "Input Embed"
    output_node = "Residual Output: <tool_call>"
    all_nodes = [input_node] + list(nodes) + [output_node]

    # Layer map for vertical layout (input at bottom, output at top).
    if nodes:
        min_l = min(node_layer(n) for n in nodes)
        max_l = max(node_layer(n) for n in nodes)
    else:
        min_l = 0
        max_l = 1
    layer_map = {input_node: min_l - 2, output_node: max_l + 2}
    for n in nodes:
        layer_map[n] = node_layer(n)

    by_layer: Dict[int, List[str]] = {}
    for n in all_nodes:
        by_layer.setdefault(layer_map[n], []).append(n)

    pos: Dict[str, Tuple[float, float]] = {}
    for layer in sorted(by_layer):
        group = sorted(by_layer[layer], key=lambda n: (0 if n.startswith("MLP") else 1, n))
        k = len(group)
        span = 2.2
        xs = [0.0] if k == 1 else np.linspace(-span * (k - 1) / 2, span * (k - 1) / 2, k)
        for x, n in zip(xs, group):
            pos[n] = (float(x), float(layer))

    apply_plot_style()
    fig, ax = plt.subplots(figsize=(8.6, 11.0), constrained_layout=True)
    ax.set_title(title)
    ax.axis("off")

    # Edges
    edge_color = "#8a4f28"
    for s, t in edges:
        x1, y1 = pos[s]
        x2, y2 = pos[t]
        rad = 0.09 if abs(x2 - x1) > 0.2 else 0.0
        arrow = FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle="->",
            mutation_scale=14,
            linewidth=1.7,
            color=edge_color,
            connectionstyle=f"arc3,rad={rad}",
            zorder=1,
        )
        ax.add_patch(arrow)

    # Nodes
    for n in all_nodes:
        x, y = pos[n]
        if n in {input_node, output_node}:
            fc = "#bcd3ea"
            ec = "#bcd3ea"
            size = 240
        else:
            fc = "#f5ede7"
            ec = "#2e2e2e"
            size = 200
        ax.scatter([x], [y], s=size, c=fc, edgecolors=ec, linewidths=1.6, zorder=3)

        if n == input_node:
            text = "Input Embed"
        elif n == output_node:
            text = "Residual Output: <tool_call>"
        else:
            text = n
        ax.text(
            x + 0.34,
            y,
            text,
            va="center",
            ha="left",
            fontsize=11 if n not in {input_node, output_node} else 13,
            zorder=4,
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.2, "alpha": 0.85},
        )

    ys = [p[1] for p in pos.values()]
    ax.set_ylim(min(ys) - 1.0, max(ys) + 1.3)
    ax.set_xlim(-6.8, 7.8)
    fig.savefig(out_path, dpi=280, bbox_inches="tight")
    plt.close(fig)


def resolve_single_sample(
    *,
    dataset_root: str,
    pair_dir: str,
    sample_id: str,
    sample_rank: int,
    q_index: int,
) -> ToolCallSample:
    use_pair = q_index > 0 and not sample_id and sample_rank <= 0
    if use_pair:
        samples = load_toolcall_samples(pair_dir=Path(pair_dir))
        chosen = select_samples(samples, legacy_indices=[q_index])
        if not chosen:
            raise ValueError(f"Could not resolve q_index={q_index} under {pair_dir}")
        return chosen[0]

    samples = load_toolcall_samples(dataset_root=Path(dataset_root))
    chosen = select_samples(
        samples,
        sample_ids=[sample_id] if sample_id else (),
        sample_rank_min=sample_rank if sample_rank > 0 else 1,
        sample_rank_max=sample_rank if sample_rank > 0 else 1,
    )
    if not chosen:
        sample_ref = sample_id or f"sample_rank={sample_rank if sample_rank > 0 else 1}"
        raise ValueError(f"Could not resolve {sample_ref} under {dataset_root}")
    return chosen[0]


def run_one_sample(
    *,
    sample: ToolCallSample,
    out_dir: Path,
    model: HookedTransformer,
    tokenizer,
    model_path: str,
    head_score_mode: str = "ap",
    skip_plots: bool = False,
) -> Dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)

    clean_text = sample.clean_path.read_text(encoding="utf-8")
    corrupt_text = sample.corrupt_path.read_text(encoding="utf-8")
    clean_tokens = model.to_tokens(clean_text, prepend_bos=False)
    corrupt_tokens = model.to_tokens(corrupt_text, prepend_bos=False)
    if clean_tokens.shape != corrupt_tokens.shape:
        raise ValueError(f"Clean/corrupt token shapes differ: {clean_tokens.shape} vs {corrupt_tokens.shape}")

    ids_clean = [int(x) for x in clean_tokens[0].tolist()]
    ids_corrupt = [int(x) for x in corrupt_tokens[0].tolist()]
    pos_sets = build_position_sets(ids_clean, ids_corrupt, tokenizer, clean_text=clean_text)

    with torch.no_grad():
        clean_logits = model(clean_tokens)
        corrupt_logits = model(corrupt_tokens)

    target_spec = get_tool_call_target_spec(tokenizer, target_text="<tool_call>")
    if not target_spec.is_single_token:
        raise NotImplementedError(
            f"<tool_call> tokenization is no longer single-token ({target_spec.token_ids}); "
            "upgrade the objective to sequence-level scoring before running this workflow."
        )
    target_token = target_spec.primary_token_id
    distractor_token = resolve_distractor_token(corrupt_logits[0, -1, :], target_token)
    endpoint_temperature = 1.0
    endpoint_masked_token_ids: tuple[int, ...] = ()
    endpoint_objective = build_distribution_objective(
        clean_logits,
        endpoint_label="tool_call",
        tokenizer=tokenizer,
        temperature=endpoint_temperature,
        masked_token_ids=endpoint_masked_token_ids,
    )
    endpoint_summary = summarize_endpoint_pair(
        tool_logits=clean_logits,
        no_tool_logits=corrupt_logits,
        tokenizer=tokenizer,
        temperature=endpoint_temperature,
        masked_token_ids=endpoint_masked_token_ids,
        topk=12,
    )

    clean_obj = float(objective_from_logits(clean_logits, endpoint_objective).item())
    corrupt_obj = float(objective_from_logits(corrupt_logits, endpoint_objective).item())
    gap = clean_obj - corrupt_obj
    if not math.isfinite(gap) or abs(gap) <= 1e-8:
        raise ValueError(f"Degenerate tool-call endpoint gap for {sample.sample_id}: {gap}")

    clean_cache_cpu = collect_clean_cache_cpu(model, clean_tokens)

    ap_mode = "full"
    try:
        ap_head = compute_ap_head_gain(
            model=model,
            corrupt_tokens=corrupt_tokens,
            clean_cache_cpu=clean_cache_cpu,
            target_token=endpoint_objective,
            distractor_token=None,
        )
    except torch.OutOfMemoryError:
        gc.collect()
        torch.cuda.empty_cache()
        ap_mode = "lowmem"
        try:
            ap_head = compute_ap_head_gain_lowmem(
                model=model,
                corrupt_tokens=corrupt_tokens,
                clean_cache_cpu=clean_cache_cpu,
                target_token=endpoint_objective,
                distractor_token=None,
            )
        except torch.OutOfMemoryError:
            gc.collect()
            torch.cuda.empty_cache()
            ap_mode = "zero_fallback"
            ap_head = torch.zeros(model.cfg.n_layers, model.cfg.n_heads, dtype=torch.float32)
    model.reset_hooks()

    requested_head_score_mode = normalize_head_score_mode(head_score_mode)

    head_ap_top_per_layer = max(
        0,
        int(
            os.environ.get(
                "TOOLCALL_HEAD_AP_PER_LAYER",
                os.environ.get("TOOLCALL_CT_AP_PER_LAYER", os.environ.get("ACDC_CT_AP_PER_LAYER", "0")),
            )
        ),
    )
    head_ap_top_global = max(
        0,
        int(
            os.environ.get(
                "TOOLCALL_HEAD_AP_TOP_GLOBAL",
                os.environ.get("TOOLCALL_CT_AP_TOP_GLOBAL", os.environ.get("ACDC_CT_AP_TOP_GLOBAL", "0")),
            )
        ),
    )
    head_candidate_heads = None
    actual_head_score_mode = "exact_patch"
    if ap_mode != "zero_fallback":
        head_candidate_heads = select_ct_candidate_heads(
            ap_head,
            ap_top_per_layer=head_ap_top_per_layer,
            ap_top_global=head_ap_top_global,
        )
        if head_candidate_heads:
            actual_head_score_mode = "ap_pruned_exact_patch"

    if requested_head_score_mode == "ap":
        head_score = ap_head.detach().float().cpu().clone()
        actual_head_score_mode = "ap"
    else:
        head_score = compute_exact_patch_head_gain(
            model=model,
            corrupt_tokens=corrupt_tokens,
            clean_cache_cpu=clean_cache_cpu,
            target_token=endpoint_objective,
            distractor_token=None,
            corrupt_obj=corrupt_obj,
            candidate_heads_by_layer=head_candidate_heads,
            fallback_scores=ap_head if head_candidate_heads else None,
        )

    mlp_gain = compute_mlp_gain(
        model=model,
        corrupt_tokens=corrupt_tokens,
        clean_cache_cpu=clean_cache_cpu,
        target_token=endpoint_objective,
        distractor_token=None,
        corrupt_obj=corrupt_obj,
    )

    detailed_nodes, rough_nodes, score_table = pick_nodes(
        head_score,
        ap_head,
        mlp_gain,
        head_score_mode=requested_head_score_mode,
    )
    score_lookup = {s.name: s.score for s in score_table}
    detailed_edges = build_edges(detailed_nodes, score_lookup=score_lookup, max_parents=2)
    rough_edges = build_edges(rough_nodes, score_lookup=score_lookup, max_parents=2)
    assert_no_dead_ends(detailed_nodes, detailed_edges)
    assert_no_dead_ends(rough_nodes, rough_edges)

    detailed_obj = evaluate_on_base_with_source(
        model=model,
        base_tokens=corrupt_tokens,
        source_cache_cpu=clean_cache_cpu,
        patch_nodes=detailed_nodes,
        target_token=endpoint_objective,
        distractor_token=None,
    )
    detailed_ratio = (detailed_obj - corrupt_obj) / gap if abs(gap) > 1e-8 else float("nan")

    rough_obj = evaluate_on_base_with_source(
        model=model,
        base_tokens=corrupt_tokens,
        source_cache_cpu=clean_cache_cpu,
        patch_nodes=rough_nodes,
        target_token=endpoint_objective,
        distractor_token=None,
    )
    rough_ratio = (rough_obj - corrupt_obj) / gap if abs(gap) > 1e-8 else float("nan")

    corrupt_cache_cpu = collect_clean_cache_cpu(model, corrupt_tokens)
    clean_with_detailed_corrupted = evaluate_on_base_with_source(
        model=model,
        base_tokens=clean_tokens,
        source_cache_cpu=corrupt_cache_cpu,
        patch_nodes=detailed_nodes,
        target_token=endpoint_objective,
        distractor_token=None,
    )
    necessity_drop = clean_obj - clean_with_detailed_corrupted
    necessity_ratio = necessity_drop / gap if abs(gap) > 1e-8 else float("nan")

    clean_with_rough_corrupted = evaluate_on_base_with_source(
        model=model,
        base_tokens=clean_tokens,
        source_cache_cpu=corrupt_cache_cpu,
        patch_nodes=rough_nodes,
        target_token=endpoint_objective,
        distractor_token=None,
    )
    rough_necessity_drop = clean_obj - clean_with_rough_corrupted
    rough_necessity_ratio = rough_necessity_drop / gap if abs(gap) > 1e-8 else float("nan")

    head_scores = [s for s in score_table if s.name.startswith("L")]
    probe_head = head_scores[0].name if head_scores else "L0H0"

    corrupt_with_probe = evaluate_on_base_with_source(
        model=model,
        base_tokens=corrupt_tokens,
        source_cache_cpu=clean_cache_cpu,
        patch_nodes=[probe_head],
        target_token=endpoint_objective,
        distractor_token=None,
    )
    clean_with_probe_corrupt = evaluate_on_base_with_source(
        model=model,
        base_tokens=clean_tokens,
        source_cache_cpu=corrupt_cache_cpu,
        patch_nodes=[probe_head],
        target_token=endpoint_objective,
        distractor_token=None,
    )
    if not skip_plots:
        plot_head_heatmap(
            ap_head.numpy(),
            "Attribution Patching Head Heatmap (AP)",
            out_dir / "ap_head_heatmap.png",
        )
        plot_head_heatmap(
            head_score.numpy(),
            (
                "Head Score Heatmap (AP)"
                if actual_head_score_mode == "ap"
                else "Head Score Heatmap (Exact clean-to-corrupt patch)"
            ),
            out_dir / "head_score_heatmap.png",
        )
        plot_probe(
            out_path=out_dir / f"{probe_head}_probe.png",
            head_name=probe_head,
            corrupt_obj=corrupt_obj,
            corrupt_with_head=corrupt_with_probe,
            clean_obj=clean_obj,
            clean_with_head_corrupt=clean_with_probe_corrupt,
        )
        draw_circuit(
            nodes=detailed_nodes,
            edges=detailed_edges,
            out_path=out_dir / "final_circuit_detailed.png",
            title="Detailed Circuit (Tool-call Decision)",
        )
        draw_circuit(
            nodes=rough_nodes,
            edges=rough_edges,
            out_path=out_dir / "final_circuit.png",
            title="Simplified Circuit (Tool-call Decision)",
        )

    contrast_token_details = [
        {
            "position": int(pos),
            "clean_token": decode_token_at(tokenizer, ids_clean, int(pos)),
            "corrupt_token": decode_token_at(tokenizer, ids_corrupt, int(pos)),
        }
        for pos in pos_sets["contrast"]
    ]

    summary: Dict[str, object] = {
        "sample_id": sample.sample_id,
        "sample_rank": sample.sample_rank,
        "filename": sample.filename,
        "source_kind": sample.source_kind,
        "q_index": sample.legacy_index,
        "clean_prompt": str(sample.clean_path),
        "corrupt_prompt": str(sample.corrupt_path),
        "model_path": model_path,
        "clean_prompt_token_length": int(clean_tokens.shape[1]),
        "corrupt_prompt_token_length": int(corrupt_tokens.shape[1]),
        "token_lengths_aligned": bool(clean_tokens.shape == corrupt_tokens.shape),
        "target_tokenization": target_spec.to_dict(tokenizer),
        "target_token_id": target_token,
        "target_token_str": tokenizer.decode([target_token]),
        "distractor_token_id": distractor_token,
        "distractor_token_str": tokenizer.decode([distractor_token]),
        "ap_mode": ap_mode,
        "head_score_mode": actual_head_score_mode,
        "head_score_mode_requested": requested_head_score_mode,
        "objective_mode": "negative_kl_to_clean_endpoint",
        "objective_endpoint": "tool_call",
        "objective_temperature": endpoint_temperature,
        "objective_masked_token_ids": list(endpoint_masked_token_ids),
        "head_candidate_ap_top_per_layer": head_ap_top_per_layer,
        "head_candidate_ap_top_global": head_ap_top_global,
        "head_candidate_count": sum(len(v) for v in head_candidate_heads.values()) if head_candidate_heads else 0,
        "clean_obj": clean_obj,
        "corrupt_obj": corrupt_obj,
        "gap": gap,
        "clean_kl_to_endpoint": -clean_obj,
        "corrupt_kl_to_endpoint": -corrupt_obj,
        "endpoint_js_divergence": endpoint_summary["endpoint_js_divergence"],
        "detailed_obj": detailed_obj,
        "detailed_ratio_vs_gap": detailed_ratio,
        "detailed_kl_recovery_ratio": detailed_ratio,
        "rough_obj": rough_obj,
        "rough_ratio_vs_gap": rough_ratio,
        "rough_kl_recovery_ratio": rough_ratio,
        "clean_with_detailed_corrupted": clean_with_detailed_corrupted,
        "necessity_drop": necessity_drop,
        "necessity_ratio_vs_gap": necessity_ratio,
        "necessity_kl_drop_ratio": necessity_ratio,
        "clean_with_rough_corrupted": clean_with_rough_corrupted,
        "rough_necessity_drop": rough_necessity_drop,
        "rough_necessity_ratio_vs_gap": rough_necessity_ratio,
        "rough_necessity_kl_drop_ratio": rough_necessity_ratio,
        "probe_head": probe_head,
        "probe_corrupt_with_head": corrupt_with_probe,
        "probe_clean_with_head_corrupt": clean_with_probe_corrupt,
        "detailed_nodes": detailed_nodes,
        "detailed_edges": detailed_edges,
        "rough_nodes": rough_nodes,
        "rough_edges": rough_edges,
        "contrast_positions": list(pos_sets["contrast"]),
        "contrast_spans": list(pos_sets["contrast_spans"]),
        "contrast_token_details": contrast_token_details,
        "tool_call_open_positions": list(pos_sets["tool_call_open"]),
        "tool_call_close_positions": list(pos_sets["tool_call_close"]),
        "tool_call_tag_positions": list(pos_sets["tool_call_tags"]),
        "tools_block_positions": list(pos_sets["tools_block"]),
        "user_block_positions": list(pos_sets["user_block"]),
        "sample_catalog_record": sample.catalog_record(),
        "endpoint_distribution_summary": endpoint_summary,
        "top_node_scores": [
            {"name": s.name, "score": s.score, "exact_patch": s.exact_patch, "ap": s.ap}
            for s in score_table
        ],
        "artifacts": {
            "ap_head_heatmap": str(out_dir / "ap_head_heatmap.png"),
            "head_score_heatmap": str(out_dir / "head_score_heatmap.png"),
            "probe": str(out_dir / f"{probe_head}_probe.png"),
            "final_circuit_detailed": str(out_dir / "final_circuit_detailed.png"),
            "final_circuit": str(out_dir / "final_circuit.png"),
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


# -----------------------------
# Main
# -----------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Mine a tool-call circuit on Qwen3 with ACDC-inspired patching.")
    parser.add_argument("--dataset-root", type=str, default=str(DATASETS_ROOT))
    parser.add_argument("--sample-id", type=str, default="")
    parser.add_argument("--sample-rank", type=int, default=0)
    parser.add_argument("--q-index", type=int, default=0, help="Legacy pair-style sample index.")
    parser.add_argument("--pair-dir", type=str, default="", help="Legacy pair directory; required only with --q-index.")
    parser.add_argument("--model-path", type=str, default=str(MODEL_PATH_DEFAULT))
    parser.add_argument("--out-dir", type=str, default=str(RESULTS_ROOT / "manual_run" / "single_sample"))
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--head-score-mode", choices=["ap", "exact_patch"], default="ap")
    parser.add_argument("--ct-head-mode", dest="legacy_head_score_mode", choices=["exact", "ap_proxy"], default=None, help=argparse.SUPPRESS)
    parser.add_argument("--skip-plots", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    sample = resolve_single_sample(
        dataset_root=args.dataset_root,
        pair_dir=args.pair_dir,
        sample_id=args.sample_id,
        sample_rank=args.sample_rank,
        q_index=args.q_index,
    )
    if out_dir.name == "single_sample":
        out_dir = out_dir / sample.sample_id
        out_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer = load_hooked_qwen3(args.model_path, device=args.device, dtype=torch.bfloat16)
    head_score_mode = args.head_score_mode
    if args.legacy_head_score_mode is not None:
        head_score_mode = normalize_head_score_mode(args.legacy_head_score_mode)
    summary = run_one_sample(
        sample=sample,
        out_dir=out_dir,
        model=model,
        tokenizer=tokenizer,
        model_path=args.model_path,
        head_score_mode=head_score_mode,
        skip_plots=args.skip_plots,
    )

    print(f"[done] outputs written to: {out_dir}")
    print(f"[done] sample: {summary['sample_id']}")
    print(f"[done] probe component: {summary['probe_head']}")
    print(f"[done] detailed sufficiency ratio: {summary['detailed_ratio_vs_gap']:.3f}")
    print(f"[done] detailed necessity ratio: {summary['necessity_ratio_vs_gap']:.3f}")
    print(f"[done] rough sufficiency ratio: {summary['rough_ratio_vs_gap']:.3f}")
    print(f"[done] rough necessity ratio: {summary['rough_necessity_ratio_vs_gap']:.3f}")


if __name__ == "__main__":
    main()
