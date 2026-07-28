#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gc
import json
import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from tqdm.auto import tqdm

LEGACY_SRC = Path("./src")
if str(LEGACY_SRC) not in sys.path:
    sys.path.insert(0, str(LEGACY_SRC))

from toolcall_circuit.single_sample import load_hooked_qwen3  # noqa: E402


SEED = 42
MODEL_PATH = Path("./external/models/Qwen3-8B")
ATTENTION_ROOT = Path("./results/8B/attention_analysis")
DIFF_ROOT = Path("./results/8B/differential_mechanism")
TRANSCODER_PATH = Path("./external/transcoders/Qwen3-8B/layer_29.safetensors")
MANIFEST_PATH = ATTENTION_ROOT / "sample_manifest.csv"
OUTPUT_ROOT = Path("./results/8B/l29_deep_analysis")
TOOL_CALL_STR = "<tool_call>"
LAYER = 29


@dataclass
class SamplePair:
    order: int
    sample_id: str
    clean_path: Path
    corrupt_path: Path
    clean_tokens_cpu: torch.Tensor
    corrupt_tokens_cpu: torch.Tensor
    token_len: int


@dataclass
class PairBatch:
    indices: list[int]
    clean_tokens_cpu: torch.Tensor
    corrupt_tokens_cpu: torch.Tensor
    token_len: int


@dataclass(frozen=True)
class FeatureInfo:
    feature_idx: int
    category: str
    mean_clean: float
    mean_corrupt: float
    delta: float
    auc_effective: float
    tool_call_projection: float
    clean_contrib: float
    corrupt_contrib: float
    delta_contrib: float


@dataclass
class FeatureSet:
    name: str
    feature_ids: tuple[int, ...]
    clean_values: torch.Tensor
    corrupt_values: torch.Tensor
    W_enc: torch.Tensor
    b_enc: torch.Tensor
    W_dec: torch.Tensor
    rationale: str


@dataclass(frozen=True)
class InterventionSpec:
    name: str
    attn_heads: tuple[int, ...]
    mlp_full: bool
    feature_key: str | None
    rationale: str

    @property
    def head_list(self) -> str:
        if not self.attn_heads:
            return ""
        return "|".join(f"L29H{head}" for head in self.attn_heads)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def clear_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, data: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def tool_stats(logits: torch.Tensor, tool_token_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    last_logits = logits[:, -1, :]
    tool_logit = last_logits[:, tool_token_id].detach().cpu().float()
    top1 = last_logits.argmax(dim=-1).detach().cpu()
    return tool_logit, top1


def load_sample_pairs(manifest_path: Path, model, *, max_pairs: int) -> list[SamplePair]:
    pairs: list[SamplePair] = []
    with manifest_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            clean_path = Path(row["clean_path"])
            corrupt_path = Path(row["corrupt_path"])
            clean_text = clean_path.read_text(encoding="utf-8")
            corrupt_text = corrupt_path.read_text(encoding="utf-8")
            clean_tokens_cpu = model.to_tokens(clean_text, prepend_bos=False).detach().cpu()
            corrupt_tokens_cpu = model.to_tokens(corrupt_text, prepend_bos=False).detach().cpu()
            clean_len = int(clean_tokens_cpu.shape[-1])
            corrupt_len = int(corrupt_tokens_cpu.shape[-1])
            if clean_len != corrupt_len:
                continue
            pairs.append(
                SamplePair(
                    order=int(row["order"]),
                    sample_id=str(row["sample_id"]),
                    clean_path=clean_path,
                    corrupt_path=corrupt_path,
                    clean_tokens_cpu=clean_tokens_cpu,
                    corrupt_tokens_cpu=corrupt_tokens_cpu,
                    token_len=clean_len,
                )
            )
            if len(pairs) >= max_pairs:
                break
    if len(pairs) < max_pairs:
        raise RuntimeError(f"Only found {len(pairs)} usable pairs in {manifest_path}")
    return pairs


def build_pair_batches(pairs: Sequence[SamplePair], batch_size: int) -> list[PairBatch]:
    buckets: dict[int, list[tuple[int, SamplePair]]] = defaultdict(list)
    for idx, pair in enumerate(pairs):
        buckets[pair.token_len].append((idx, pair))

    batches: list[PairBatch] = []
    for token_len in sorted(buckets):
        group = buckets[token_len]
        for start in range(0, len(group), batch_size):
            chunk = group[start : start + batch_size]
            batches.append(
                PairBatch(
                    indices=[idx for idx, _pair in chunk],
                    clean_tokens_cpu=torch.cat([pair.clean_tokens_cpu for _, pair in chunk], dim=0),
                    corrupt_tokens_cpu=torch.cat([pair.corrupt_tokens_cpu for _, pair in chunk], dim=0),
                    token_len=token_len,
                )
            )
    return batches


def collect_baseline_states(
    model,
    pair_batches: Sequence[PairBatch],
    *,
    tool_token_id: int,
) -> tuple[dict[str, torch.Tensor], dict[str, object]]:
    if hasattr(model, "set_use_hook_mlp_in"):
        model.set_use_hook_mlp_in(True)
    if hasattr(model, "cfg") and hasattr(model.cfg, "use_hook_mlp_in"):
        model.cfg.use_hook_mlp_in = True

    n_samples = sum(len(batch.indices) for batch in pair_batches)
    d_model = int(model.cfg.d_model)
    clean_logits = torch.empty(n_samples, dtype=torch.float32)
    corrupt_logits = torch.empty(n_samples, dtype=torch.float32)
    clean_top1 = torch.empty(n_samples, dtype=torch.long)
    corrupt_top1 = torch.empty(n_samples, dtype=torch.long)
    clean_mlp_in = torch.empty((n_samples, d_model), dtype=torch.bfloat16)
    corrupt_mlp_in = torch.empty((n_samples, d_model), dtype=torch.bfloat16)
    clean_mlp_out = torch.empty((n_samples, d_model), dtype=torch.float32)
    corrupt_mlp_out = torch.empty((n_samples, d_model), dtype=torch.float32)
    hook_names = {f"blocks.{LAYER}.hook_mlp_in", f"blocks.{LAYER}.hook_mlp_out"}
    shape_info: dict[str, list[int]] = {}

    progress = tqdm(pair_batches, desc="Collecting L29 baseline", dynamic_ncols=True)
    for batch in progress:
        for condition in ("clean", "corrupt"):
            tokens_cpu = batch.clean_tokens_cpu if condition == "clean" else batch.corrupt_tokens_cpu
            tokens = tokens_cpu.to(model.W_U.device)
            with torch.no_grad():
                logits, cache = model.run_with_cache(tokens, names_filter=lambda name: name in hook_names)
            batch_logits, batch_top1 = tool_stats(logits, tool_token_id)
            hook_in = cache[f"blocks.{LAYER}.hook_mlp_in"][:, -1, :].detach().cpu().to(torch.bfloat16)
            hook_out = cache[f"blocks.{LAYER}.hook_mlp_out"][:, -1, :].detach().cpu().float()
            if not shape_info:
                shape_info["hook_mlp_in"] = [int(x) for x in cache[f"blocks.{LAYER}.hook_mlp_in"].shape]
                shape_info["hook_mlp_out"] = [int(x) for x in cache[f"blocks.{LAYER}.hook_mlp_out"].shape]
            idx = torch.tensor(batch.indices, dtype=torch.long)
            if condition == "clean":
                clean_logits[idx] = batch_logits
                clean_top1[idx] = batch_top1
                clean_mlp_in[idx] = hook_in
                clean_mlp_out[idx] = hook_out
            else:
                corrupt_logits[idx] = batch_logits
                corrupt_top1[idx] = batch_top1
                corrupt_mlp_in[idx] = hook_in
                corrupt_mlp_out[idx] = hook_out
            del tokens, logits, cache
            clear_cuda()
        progress.set_postfix(last=batch.indices[-1], tok=batch.token_len)

    baseline = {
        "clean_tool_logit": clean_logits,
        "corrupt_tool_logit": corrupt_logits,
        "clean_top1": clean_top1,
        "corrupt_top1": corrupt_top1,
        "clean_mlp_in": clean_mlp_in,
        "corrupt_mlp_in": corrupt_mlp_in,
        "clean_mlp_out": clean_mlp_out,
        "corrupt_mlp_out": corrupt_mlp_out,
    }
    metadata = {
        "hook_mlp_in_shape": shape_info.get("hook_mlp_in"),
        "hook_mlp_out_shape": shape_info.get("hook_mlp_out"),
        "clean_tool_top1_rate": float((clean_top1 == tool_token_id).float().mean().item()),
        "corrupt_tool_top1_rate": float((corrupt_top1 == tool_token_id).float().mean().item()),
    }
    return baseline, metadata


def compute_feature_info(
    diff_rows: Sequence[dict[str, str]],
    baseline: dict[str, torch.Tensor],
    transcoder_weights: dict[str, torch.Tensor],
    tool_writer: torch.Tensor,
) -> list[FeatureInfo]:
    W_dec = transcoder_weights["W_dec"].float()
    out: list[FeatureInfo] = []
    for row in diff_rows:
        if row["category"] == "none":
            continue
        idx = int(row["feature_idx"])
        proj = float(torch.dot(W_dec[idx], tool_writer).item())
        mean_clean = float(row["mean_clean"])
        mean_corrupt = float(row["mean_corrupt"])
        out.append(
            FeatureInfo(
                feature_idx=idx,
                category=row["category"],
                mean_clean=mean_clean,
                mean_corrupt=mean_corrupt,
                delta=float(row["delta"]),
                auc_effective=float(row["auc_effective"]),
                tool_call_projection=proj,
                clean_contrib=mean_clean * proj,
                corrupt_contrib=mean_corrupt * proj,
                delta_contrib=(mean_clean - mean_corrupt) * proj,
            )
        )
    return out


def build_feature_sets(
    baseline: dict[str, torch.Tensor],
    transcoder_weights: dict[str, torch.Tensor],
    feature_info: Sequence[FeatureInfo],
) -> tuple[dict[str, FeatureSet], list[dict[str, object]]]:
    W_enc = transcoder_weights["W_enc"].detach().cpu()
    b_enc = transcoder_weights["b_enc"].detach().cpu()
    W_dec = transcoder_weights["W_dec"].detach().cpu()

    clean_inputs = baseline["clean_mlp_in"].float()
    corrupt_inputs = baseline["corrupt_mlp_in"].float()

    clean_writer = [info for info in feature_info if info.delta > 0 and info.tool_call_projection > 0]
    clean_writer.sort(key=lambda info: info.delta_contrib, reverse=True)
    anti_tool = [info for info in feature_info if info.delta < 0 and info.tool_call_projection < 0]
    anti_tool.sort(key=lambda info: abs(info.delta_contrib), reverse=True)

    set_specs: list[tuple[str, list[FeatureInfo], str]] = [
        ("clean_writer_top2", clean_writer[:2], "Top-2 L29 MLP clean-writer features by delta*projection."),
        ("clean_writer_top4", clean_writer[:4], "Top-4 L29 MLP clean-writer features by delta*projection."),
        ("clean_writer_all", clean_writer, "All positive-projection clean-biased L29 features."),
        ("anti_tool_all", anti_tool, "All negative-projection anti-tool L29 features."),
    ]

    feature_sets: dict[str, FeatureSet] = {}
    rows: list[dict[str, object]] = []
    for name, infos, rationale in set_specs:
        ids = [info.feature_idx for info in infos]
        if not ids:
            continue
        idx = torch.tensor(ids, dtype=torch.long)
        W_enc_sel = W_enc[idx].contiguous().to(torch.bfloat16)
        b_enc_sel = b_enc[idx].contiguous().to(torch.bfloat16)
        W_dec_sel = W_dec[idx].contiguous().to(torch.bfloat16)
        clean_values = torch.relu(F.linear(clean_inputs.to(torch.bfloat16), W_enc_sel, b_enc_sel)).float().cpu()
        corrupt_values = torch.relu(F.linear(corrupt_inputs.to(torch.bfloat16), W_enc_sel, b_enc_sel)).float().cpu()
        feature_sets[name] = FeatureSet(
            name=name,
            feature_ids=tuple(ids),
            clean_values=clean_values,
            corrupt_values=corrupt_values,
            W_enc=W_enc_sel,
            b_enc=b_enc_sel,
            W_dec=W_dec_sel,
            rationale=rationale,
        )
        rows.append(
            {
                "feature_set": name,
                "n_features": len(ids),
                "feature_ids": "|".join(str(feature_id) for feature_id in ids),
                "rationale": rationale,
                "mean_clean_feature_sum": float(clean_values.sum(dim=1).mean().item()),
                "mean_corrupt_feature_sum": float(corrupt_values.sum(dim=1).mean().item()),
            }
        )
    return feature_sets, rows


def build_intervention_specs(feature_sets: dict[str, FeatureSet], n_heads: int) -> list[InterventionSpec]:
    specs: list[InterventionSpec] = [
        InterventionSpec("attn_H9", (9,), False, None, "Single strongest L29 attention head."),
        InterventionSpec("attn_top3", (9, 11, 14), False, None, "Top-3 positive-DLA heads in L29."),
        InterventionSpec("attn_full", tuple(range(n_heads)), False, None, "All L29 attention heads."),
        InterventionSpec("mlp_full", (), True, None, "Whole L29 MLP output."),
        InterventionSpec("attn_top3_plus_mlp_full", (9, 11, 14), True, None, "Top-3 L29 attention heads plus full L29 MLP."),
        InterventionSpec("attn_full_plus_mlp_full", tuple(range(n_heads)), True, None, "Entire L29 layer output (attention + MLP)."),
    ]
    if "clean_writer_top4" in feature_sets:
        specs.append(InterventionSpec("feat_clean_writer_top4", (), False, "clean_writer_top4", "Top-4 clean-writer features in L29 MLP."))
        specs.append(
            InterventionSpec(
                "attn_top3_plus_feat_clean_writer_top4",
                (9, 11, 14),
                False,
                "clean_writer_top4",
                "Top-3 L29 attention heads plus top-4 clean-writer L29 MLP features.",
            )
        )
    if "clean_writer_all" in feature_sets:
        specs.append(InterventionSpec("feat_clean_writer_all", (), False, "clean_writer_all", "All clean-writer features in L29 MLP."))
    if "anti_tool_all" in feature_sets:
        specs.append(InterventionSpec("feat_anti_tool_all", (), False, "anti_tool_all", "All anti-tool features in L29 MLP."))
    return specs


def make_zero_attn_hook(heads: Sequence[int]):
    heads = list(heads)

    def hook_fn(value: torch.Tensor, hook):  # noqa: ANN001
        out = value.clone()
        out[:, :, heads, :] = 0
        return out

    return hook_fn


def make_patch_attn_hook(heads: Sequence[int], source_cpu: torch.Tensor):
    heads = list(heads)

    def hook_fn(value: torch.Tensor, hook):  # noqa: ANN001
        src = source_cpu.to(device=value.device, dtype=value.dtype)
        out = value.clone()
        out[:, :, heads, :] = src[:, :, heads, :]
        return out

    return hook_fn


def zero_mlp_hook(value: torch.Tensor, hook):  # noqa: ANN001
    out = value.clone()
    out.zero_()
    return out


def make_patch_mlp_hook(source_cpu: torch.Tensor):
    def hook_fn(value: torch.Tensor, hook):  # noqa: ANN001
        src = source_cpu.to(device=value.device, dtype=value.dtype)
        out = value.clone()
        out.copy_(src)
        return out

    return hook_fn


def make_feature_hooks(feature_set: FeatureSet, batch_indices: Sequence[int], *, mode: str, source_side: str | None):
    state: dict[str, torch.Tensor] = {}
    source_values = None
    if mode == "inject":
        if source_side == "clean":
            source_values = feature_set.clean_values[list(batch_indices)]
        elif source_side == "corrupt":
            source_values = feature_set.corrupt_values[list(batch_indices)]
        else:
            raise ValueError("Feature injection requires source_side.")

    def in_hook(value: torch.Tensor, hook):  # noqa: ANN001
        state["mlp_in"] = value[:, -1, :].detach()
        return value

    def out_hook(value: torch.Tensor, hook):  # noqa: ANN001
        mlp_in = state.pop("mlp_in")
        W_enc = feature_set.W_enc.to(device=value.device, dtype=torch.bfloat16)
        b_enc = feature_set.b_enc.to(device=value.device, dtype=torch.bfloat16)
        W_dec = feature_set.W_dec.to(device=value.device, dtype=torch.bfloat16)
        acts = torch.relu(F.linear(mlp_in.to(dtype=torch.bfloat16), W_enc, b_enc))
        current_contrib = acts @ W_dec
        out = value.clone()
        if mode == "ablate":
            out[:, -1, :] = out[:, -1, :] - current_contrib.to(dtype=out.dtype)
        else:
            if source_values is None:
                raise RuntimeError("Feature injection missing source values.")
            target_contrib = source_values.to(device=value.device, dtype=torch.bfloat16) @ W_dec
            out[:, -1, :] = out[:, -1, :] - current_contrib.to(dtype=out.dtype) + target_contrib.to(dtype=out.dtype)
        return out

    return [(f"blocks.{LAYER}.hook_mlp_in", in_hook), (f"blocks.{LAYER}.hook_mlp_out", out_hook)]


def collect_patch_cache(model, tokens: torch.Tensor, tool_token_id: int) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    hook_names = {f"blocks.{LAYER}.attn.hook_z", f"blocks.{LAYER}.hook_mlp_out"}
    with torch.no_grad():
        logits, cache = model.run_with_cache(tokens, names_filter=lambda name: name in hook_names)
    tool_logit, top1 = tool_stats(logits, tool_token_id)
    cache_cpu = {name: cache[name].detach().cpu() for name in hook_names}
    return tool_logit, top1, cache_cpu


def run_intervention_suite(
    model,
    pair_batches: Sequence[PairBatch],
    specs: Sequence[InterventionSpec],
    feature_sets: dict[str, FeatureSet],
    *,
    tool_token_id: int,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    results = {
        spec.name: {
            "clean_ablation_delta_sum": 0.0,
            "corrupt_ablation_delta_sum": 0.0,
            "clean_patch_delta_sum": 0.0,
            "corrupt_patch_delta_sum": 0.0,
            "clean_ablation_flip_count": 0,
            "corrupt_ablation_gain_count": 0,
            "clean_patch_flip_count": 0,
            "corrupt_patch_gain_count": 0,
            "clean_count": 0,
            "corrupt_count": 0,
        }
        for spec in specs
    }
    hook_shapes: dict[str, list[int]] = {}

    progress = tqdm(pair_batches, desc="L29 interventions", dynamic_ncols=True)
    for batch in progress:
        clean_tokens = batch.clean_tokens_cpu.to(model.W_U.device)
        corrupt_tokens = batch.corrupt_tokens_cpu.to(model.W_U.device)
        clean_logit, clean_top1, clean_cache = collect_patch_cache(model, clean_tokens, tool_token_id)
        corrupt_logit, corrupt_top1, corrupt_cache = collect_patch_cache(model, corrupt_tokens, tool_token_id)
        if not hook_shapes:
            for key, value in clean_cache.items():
                hook_shapes[key] = [int(x) for x in value.shape]

        clean_count = int(clean_logit.numel())
        corrupt_count = int(corrupt_logit.numel())
        for spec in specs:
            hooks = []
            if spec.attn_heads:
                hooks.append((f"blocks.{LAYER}.attn.hook_z", make_zero_attn_hook(spec.attn_heads)))
            if spec.mlp_full:
                hooks.append((f"blocks.{LAYER}.hook_mlp_out", zero_mlp_hook))
            if spec.feature_key is not None:
                hooks.extend(make_feature_hooks(feature_sets[spec.feature_key], batch.indices, mode="ablate", source_side=None))

            clean_ab_logits, clean_ab_top1 = run_with_hooks_and_stats(model, clean_tokens, hooks, tool_token_id)
            corrupt_ab_logits, corrupt_ab_top1 = run_with_hooks_and_stats(model, corrupt_tokens, hooks, tool_token_id)

            clean_ab_delta = clean_ab_logits - clean_logit
            corrupt_ab_delta = corrupt_ab_logits - corrupt_logit
            results[spec.name]["clean_ablation_delta_sum"] += float(clean_ab_delta.sum().item())
            results[spec.name]["corrupt_ablation_delta_sum"] += float(corrupt_ab_delta.sum().item())
            results[spec.name]["clean_ablation_flip_count"] += int(((clean_top1 == tool_token_id) & (clean_ab_top1 != tool_token_id)).sum().item())
            results[spec.name]["corrupt_ablation_gain_count"] += int(((corrupt_top1 != tool_token_id) & (corrupt_ab_top1 == tool_token_id)).sum().item())
            results[spec.name]["clean_count"] += clean_count
            results[spec.name]["corrupt_count"] += corrupt_count

            clean_patch_hooks = []
            corrupt_patch_hooks = []
            if spec.attn_heads:
                clean_patch_hooks.append((f"blocks.{LAYER}.attn.hook_z", make_patch_attn_hook(spec.attn_heads, corrupt_cache[f"blocks.{LAYER}.attn.hook_z"])))
                corrupt_patch_hooks.append((f"blocks.{LAYER}.attn.hook_z", make_patch_attn_hook(spec.attn_heads, clean_cache[f"blocks.{LAYER}.attn.hook_z"])))
            if spec.mlp_full:
                clean_patch_hooks.append((f"blocks.{LAYER}.hook_mlp_out", make_patch_mlp_hook(corrupt_cache[f"blocks.{LAYER}.hook_mlp_out"])))
                corrupt_patch_hooks.append((f"blocks.{LAYER}.hook_mlp_out", make_patch_mlp_hook(clean_cache[f"blocks.{LAYER}.hook_mlp_out"])))
            if spec.feature_key is not None:
                clean_patch_hooks.extend(make_feature_hooks(feature_sets[spec.feature_key], batch.indices, mode="inject", source_side="corrupt"))
                corrupt_patch_hooks.extend(make_feature_hooks(feature_sets[spec.feature_key], batch.indices, mode="inject", source_side="clean"))

            clean_patch_logits, clean_patch_top1 = run_with_hooks_and_stats(model, clean_tokens, clean_patch_hooks, tool_token_id)
            corrupt_patch_logits, corrupt_patch_top1 = run_with_hooks_and_stats(model, corrupt_tokens, corrupt_patch_hooks, tool_token_id)
            clean_patch_delta = clean_patch_logits - clean_logit
            corrupt_patch_delta = corrupt_patch_logits - corrupt_logit
            results[spec.name]["clean_patch_delta_sum"] += float(clean_patch_delta.sum().item())
            results[spec.name]["corrupt_patch_delta_sum"] += float(corrupt_patch_delta.sum().item())
            results[spec.name]["clean_patch_flip_count"] += int(((clean_top1 == tool_token_id) & (clean_patch_top1 != tool_token_id)).sum().item())
            results[spec.name]["corrupt_patch_gain_count"] += int(((corrupt_top1 != tool_token_id) & (corrupt_patch_top1 == tool_token_id)).sum().item())

            clear_cuda()

        del clean_tokens, corrupt_tokens, clean_cache, corrupt_cache
        clear_cuda()
        progress.set_postfix(last=batch.indices[-1], tok=batch.token_len)

    rows: list[dict[str, object]] = []
    for spec in specs:
        acc = results[spec.name]
        clean_count = max(acc["clean_count"], 1)
        corrupt_count = max(acc["corrupt_count"], 1)
        rows.append(
            {
                "name": spec.name,
                "attn_heads": spec.head_list,
                "mlp_full": spec.mlp_full,
                "feature_key": spec.feature_key or "",
                "clean_ablation_flip_rate": acc["clean_ablation_flip_count"] / clean_count,
                "corrupt_ablation_gain_rate": acc["corrupt_ablation_gain_count"] / corrupt_count,
                "clean_ablation_logit_delta_mean": acc["clean_ablation_delta_sum"] / clean_count,
                "corrupt_ablation_logit_delta_mean": acc["corrupt_ablation_delta_sum"] / corrupt_count,
                "clean_patch_flip_rate": acc["clean_patch_flip_count"] / clean_count,
                "corrupt_patch_gain_rate": acc["corrupt_patch_gain_count"] / corrupt_count,
                "clean_patch_logit_delta_mean": acc["clean_patch_delta_sum"] / clean_count,
                "corrupt_patch_logit_delta_mean": acc["corrupt_patch_delta_sum"] / corrupt_count,
                "rationale": spec.rationale,
            }
        )
    rows.sort(
        key=lambda row: (
            float(row["clean_ablation_flip_rate"]),
            abs(float(row["clean_ablation_logit_delta_mean"])),
            float(row["corrupt_patch_gain_rate"]),
            abs(float(row["corrupt_patch_logit_delta_mean"])),
        ),
        reverse=True,
    )
    metadata = {"hook_shapes": hook_shapes}
    return rows, metadata


def run_with_hooks_and_stats(model, tokens: torch.Tensor, hooks, tool_token_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        logits = model.run_with_hooks(tokens, fwd_hooks=hooks)
    return tool_stats(logits, tool_token_id)


def plot_intervention_bars(rows: Sequence[dict[str, object]], path: Path) -> None:
    names = [str(row["name"]) for row in rows]
    clean_delta = np.asarray([float(row["clean_ablation_logit_delta_mean"]) for row in rows], dtype=np.float32)
    clean_flip = np.asarray([float(row["clean_ablation_flip_rate"]) for row in rows], dtype=np.float32)
    x = np.arange(len(rows))

    fig, axes = plt.subplots(2, 1, figsize=(max(12, len(rows) * 0.8), 8), constrained_layout=True)
    axes[0].bar(x, clean_delta, color="#4c78a8")
    axes[0].axhline(0.0, color="#222222", linewidth=1.0)
    axes[0].set_ylabel("clean ablation delta")
    axes[0].set_title("L29 Intervention Effects")
    axes[1].bar(x, clean_flip, color="#e45756")
    axes[1].set_ylabel("clean ablation flip rate")
    axes[1].set_ylim(0.0, 1.05)
    axes[1].set_xticks(x, names, rotation=35, ha="right")
    for ax in axes:
        ax.grid(axis="y", alpha=0.25)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def build_summary(
    attention_rows: Sequence[dict[str, str]],
    mlp_component_row: dict[str, object],
    feature_rows: Sequence[dict[str, object]],
    intervention_rows: Sequence[dict[str, object]],
) -> str:
    l29_attn = sorted(attention_rows, key=lambda row: abs(float(row["delta"])), reverse=True)
    top_interventions = intervention_rows[:8]
    lines = [
        "# L29 Deep Analysis",
        "",
        "## Attention",
        f"- L29 attention total delta = {sum(float(row['delta']) for row in attention_rows):.3f}",
        f"- top heads: H{l29_attn[0]['head']} ({float(l29_attn[0]['delta']):.3f}), "
        f"H{l29_attn[1]['head']} ({float(l29_attn[1]['delta']):.3f}), "
        f"H{l29_attn[2]['head']} ({float(l29_attn[2]['delta']):.3f})",
        "",
        "## MLP",
        f"- L29 MLP whole-layer delta = {float(mlp_component_row['delta']):.3f} "
        f"(clean={float(mlp_component_row['mean_clean']):.3f}, corrupt={float(mlp_component_row['mean_corrupt']):.3f})",
    ]
    if feature_rows:
        lines.append("- top feature rows:")
        for row in feature_rows[:8]:
            lines.append(
                f"  - F{row['feature_idx']} {row['category']} "
                f"delta={float(row['delta']):.3f} proj={float(row['tool_call_projection']):.3f} "
                f"delta_contrib={float(row['delta_contrib']):.3f}"
            )
    lines.extend(["", "## Causal Interventions"])
    for row in top_interventions:
        lines.append(
            f"- {row['name']}: clean_ablate_flip={float(row['clean_ablation_flip_rate']):.3f}, "
            f"clean_ablate_delta={float(row['clean_ablation_logit_delta_mean']):.3f}, "
            f"corrupt_patch_gain={float(row['corrupt_patch_gain_rate']):.3f}, "
            f"corrupt_patch_delta={float(row['corrupt_patch_logit_delta_mean']):.3f}"
        )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deep dive on Qwen3-8B layer 29")
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH)
    parser.add_argument("--manifest-path", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--attention-root", type=Path, default=ATTENTION_ROOT)
    parser.add_argument("--diff-root", type=Path, default=DIFF_ROOT)
    parser.add_argument("--transcoder-path", type=Path, default=TRANSCODER_PATH)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--max-pairs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    args.output_root.mkdir(parents=True, exist_ok=True)
    figures_dir = args.output_root / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer = load_hooked_qwen3(str(args.model_path), device=args.device, dtype=torch.bfloat16)
    tool_token_ids = tokenizer.encode(TOOL_CALL_STR, add_special_tokens=False)
    if len(tool_token_ids) != 1:
        raise ValueError(f"{TOOL_CALL_STR!r} encoded to {tool_token_ids}, expected one token.")
    tool_token_id = int(tool_token_ids[0])

    pairs = load_sample_pairs(args.manifest_path, model, max_pairs=args.max_pairs)
    pair_batches = build_pair_batches(pairs, args.batch_size)
    baseline, baseline_meta = collect_baseline_states(model, pair_batches, tool_token_id=tool_token_id)

    tool_writer = model.W_U[:, tool_token_id].detach().cpu().float()
    diff_rows = read_csv_rows(args.diff_root / "differential_features" / "differential_features_L29.csv")
    transcoder_weights = load_file(str(args.transcoder_path))
    feature_info = compute_feature_info(diff_rows, baseline, transcoder_weights, tool_writer)
    feature_info_rows = [
        {
            "feature_idx": info.feature_idx,
            "category": info.category,
            "mean_clean": info.mean_clean,
            "mean_corrupt": info.mean_corrupt,
            "delta": info.delta,
            "auc_effective": info.auc_effective,
            "tool_call_projection": info.tool_call_projection,
            "clean_contrib": info.clean_contrib,
            "corrupt_contrib": info.corrupt_contrib,
            "delta_contrib": info.delta_contrib,
        }
        for info in sorted(feature_info, key=lambda item: abs(item.delta_contrib), reverse=True)
    ]
    write_csv(args.output_root / "l29_feature_info.csv", feature_info_rows)

    feature_sets, feature_set_rows = build_feature_sets(baseline, transcoder_weights, feature_info)
    write_csv(args.output_root / "l29_feature_sets.csv", feature_set_rows)

    mlp_clean_mean = float((baseline["clean_mlp_out"] @ tool_writer).mean().item())
    mlp_corrupt_mean = float((baseline["corrupt_mlp_out"] @ tool_writer).mean().item())
    mlp_component_row = {
        "component": "L29_MLP_full",
        "mean_clean": mlp_clean_mean,
        "mean_corrupt": mlp_corrupt_mean,
        "delta": mlp_clean_mean - mlp_corrupt_mean,
    }
    write_csv(args.output_root / "l29_mlp_component.csv", [mlp_component_row])

    attention_rows = read_csv_rows(args.attention_root / "dla_l29_per_head.csv")
    write_csv(
        args.output_root / "l29_attention_from_global.csv",
        [{k: row[k] for k in row.keys()} for row in attention_rows],
    )

    specs = build_intervention_specs(feature_sets, int(model.cfg.n_heads))
    write_csv(
        args.output_root / "l29_intervention_specs.csv",
        [
            {
                "name": spec.name,
                "attn_heads": spec.head_list,
                "mlp_full": spec.mlp_full,
                "feature_key": spec.feature_key or "",
                "rationale": spec.rationale,
            }
            for spec in specs
        ],
    )
    intervention_rows, intervention_meta = run_intervention_suite(
        model,
        pair_batches,
        specs,
        feature_sets,
        tool_token_id=tool_token_id,
    )
    write_csv(args.output_root / "l29_interventions.csv", intervention_rows)
    plot_intervention_bars(intervention_rows, figures_dir / "l29_interventions.png")

    summary = build_summary(attention_rows, mlp_component_row, feature_info_rows, intervention_rows)
    (args.output_root / "summary.md").write_text(summary, encoding="utf-8")

    write_json(
        args.output_root / "metadata.json",
        {
            "seed": args.seed,
            "model_path": str(args.model_path),
            "manifest_path": str(args.manifest_path),
            "attention_root": str(args.attention_root),
            "diff_root": str(args.diff_root),
            "transcoder_path": str(args.transcoder_path),
            "output_root": str(args.output_root),
            "tool_token_id": tool_token_id,
            "n_pairs": len(pairs),
            "batch_size": args.batch_size,
            "baseline": baseline_meta,
            "interventions": intervention_meta,
            "feature_sets": feature_set_rows,
        },
    )

    del model
    clear_cuda()


if __name__ == "__main__":
    main()
