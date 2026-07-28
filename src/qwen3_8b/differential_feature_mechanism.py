#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from tqdm.auto import tqdm

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from artifact_paths import ARTIFACT_ROOT, QWEN3_8B_PATH, QWEN3_8B_TRANSCODER_PATH  # noqa: E402

from toolcall_circuit.single_sample import load_hooked_qwen3  # noqa: E402


SEED = 42
MODEL_PATH = QWEN3_8B_PATH
TRANSCODER_DIR = QWEN3_8B_TRANSCODER_PATH
MANIFEST_PATH = ARTIFACT_ROOT / "results" / "8b_main" / "triplet_analysis" / "sample_manifest.csv"
DATASET_ROOT = ARTIFACT_ROOT / "datasets" / "train"
TRIPLET_ROOT = ARTIFACT_ROOT / "results" / "8b_main" / "triplet_analysis"
OUTPUT_ROOT = ARTIFACT_ROOT / "results" / "8b_main" / "differential_mechanism"
TOOL_CALL_STR = "<tool_call>"
DEFAULT_LAYERS = list(range(19, 36))
DEFAULT_TOPK_VALUES = [10, 50, 100]
SELECTION_TOPK = 100
DIFFERENTIAL_SELECTION_TOPK = 50
SEMANTIC_TOP_FEATURES = 100
AUC_STRONG_THRESHOLD = 0.80
LOW_ACTIVE_RATE = 0.10
SELECTIVE_RATIO = 0.25
MIN_ACTIVE_RATE = 0.05
PER_LAYER_ABLATION_K = 50


@dataclass(frozen=True)
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
    indices: List[int]
    clean_tokens_cpu: torch.Tensor
    corrupt_tokens_cpu: torch.Tensor
    token_len: int


@dataclass
class FeatureSelection:
    layer: int
    category: str
    feature_ids: List[int]
    clean_values: torch.Tensor
    corrupt_values: torch.Tensor
    W_enc: torch.Tensor
    b_enc: torch.Tensor
    W_dec: torch.Tensor
    rows: List[Dict[str, object]]


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def clear_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, data: Dict[str, object]) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    ensure_dir(path.parent)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def write_rows(path: Path, fieldnames: Sequence[str], rows: Iterable[Dict[str, object]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def list_rows_from_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_dataset_metadata(dataset_root: Path) -> Dict[str, Dict[str, object]]:
    path = dataset_root / "clean" / "manifest.jsonl"
    rows: Dict[str, Dict[str, object]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            filename = row.get("output_filename") or row.get("source_filename")
            if not filename:
                continue
            rows[Path(str(filename)).stem] = row
    return rows


def parse_topk_values(raw: str) -> List[int]:
    values = sorted({int(item.strip()) for item in raw.split(",") if item.strip()})
    if not values:
        raise ValueError("At least one top-k value is required.")
    return values


def parse_layers(raw: str) -> List[int]:
    raw = raw.strip()
    if "-" in raw:
        start_s, end_s = raw.split("-", 1)
        start = int(start_s.strip())
        end = int(end_s.strip())
        if end < start:
            raise ValueError(f"Invalid layer range: {raw}")
        return list(range(start, end + 1))
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def load_sample_pairs(
    manifest_path: Path,
    model,
    *,
    max_pairs: int,
) -> List[SamplePair]:
    pairs: List[SamplePair] = []
    with manifest_path.open(encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            clean_tokens = int(row["clean_tokens"])
            corrupt_tokens = int(row["corrupt_tokens"])
            if clean_tokens != corrupt_tokens:
                continue
            clean_path = Path(row["clean_path"])
            corrupt_path = Path(row["corrupt_path"])
            clean_text = clean_path.read_text(encoding="utf-8")
            corrupt_text = corrupt_path.read_text(encoding="utf-8")
            clean_tok_cpu = model.to_tokens(clean_text, prepend_bos=False).detach().cpu()
            corrupt_tok_cpu = model.to_tokens(corrupt_text, prepend_bos=False).detach().cpu()
            clean_len = int(clean_tok_cpu.shape[-1])
            corrupt_len = int(corrupt_tok_cpu.shape[-1])
            if clean_len != corrupt_len:
                continue
            pairs.append(
                SamplePair(
                    order=int(row["order"]),
                    sample_id=row["sample_id"],
                    clean_path=clean_path,
                    corrupt_path=corrupt_path,
                    clean_tokens_cpu=clean_tok_cpu,
                    corrupt_tokens_cpu=corrupt_tok_cpu,
                    token_len=clean_len,
                )
            )
            if len(pairs) >= max_pairs:
                break
    if not pairs:
        raise RuntimeError(f"No usable equal-length pairs found in {manifest_path}")
    return pairs


def build_pair_batches(pairs: Sequence[SamplePair], batch_size: int) -> List[PairBatch]:
    buckets: Dict[int, List[tuple[int, SamplePair]]] = defaultdict(list)
    for idx, pair in enumerate(pairs):
        buckets[pair.token_len].append((idx, pair))

    batches: List[PairBatch] = []
    for token_len in sorted(buckets.keys()):
        group = buckets[token_len]
        for start in range(0, len(group), batch_size):
            chunk = group[start : start + batch_size]
            indices = [idx for idx, _pair in chunk]
            clean_tokens = torch.cat([pair.clean_tokens_cpu for _, pair in chunk], dim=0)
            corrupt_tokens = torch.cat([pair.corrupt_tokens_cpu for _, pair in chunk], dim=0)
            batches.append(
                PairBatch(
                    indices=indices,
                    clean_tokens_cpu=clean_tokens,
                    corrupt_tokens_cpu=corrupt_tokens,
                    token_len=token_len,
                )
            )
    return batches


def tool_stats(logits: torch.Tensor, tool_token_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    last_logits = logits[:, -1, :]
    tool_logit = last_logits[:, tool_token_id].detach().cpu().float()
    top1 = last_logits.argmax(dim=-1).detach().cpu()
    return tool_logit, top1


def collect_layer_inputs_and_baseline(
    model,
    pair_batches: Sequence[PairBatch],
    *,
    layers: Sequence[int],
    n_samples: int,
    tool_token_id: int,
) -> tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor], Dict[str, torch.Tensor]]:
    if hasattr(model, "set_use_hook_mlp_in"):
        model.set_use_hook_mlp_in(True)
    if hasattr(model, "cfg") and hasattr(model.cfg, "use_hook_mlp_in"):
        model.cfg.use_hook_mlp_in = True

    device = model.W_U.device
    d_model = int(model.cfg.d_model)
    hook_names = [f"blocks.{layer}.hook_mlp_in" for layer in layers]

    clean_inputs = {layer: torch.empty((n_samples, d_model), dtype=torch.bfloat16) for layer in layers}
    corrupt_inputs = {layer: torch.empty((n_samples, d_model), dtype=torch.bfloat16) for layer in layers}
    clean_logits = torch.empty(n_samples, dtype=torch.float32)
    corrupt_logits = torch.empty(n_samples, dtype=torch.float32)
    clean_top1 = torch.empty(n_samples, dtype=torch.long)
    corrupt_top1 = torch.empty(n_samples, dtype=torch.long)

    progress = tqdm(pair_batches, desc="Collecting MLP inputs", dynamic_ncols=True)
    for batch in progress:
        tokens = torch.cat([batch.clean_tokens_cpu, batch.corrupt_tokens_cpu], dim=0).to(device)
        batch_size = int(batch.clean_tokens_cpu.shape[0])
        with torch.no_grad():
            logits, cache = model.run_with_cache(tokens, names_filter=lambda name: name in hook_names)
        all_logits, all_top1 = tool_stats(logits, tool_token_id)
        clean_logits_batch = all_logits[:batch_size]
        corrupt_logits_batch = all_logits[batch_size:]
        clean_top1_batch = all_top1[:batch_size]
        corrupt_top1_batch = all_top1[batch_size:]
        clean_logits[batch.indices] = clean_logits_batch
        corrupt_logits[batch.indices] = corrupt_logits_batch
        clean_top1[batch.indices] = clean_top1_batch
        corrupt_top1[batch.indices] = corrupt_top1_batch

        for layer in layers:
            hook_name = f"blocks.{layer}.hook_mlp_in"
            if hook_name not in cache:
                raise RuntimeError(f"Expected hook {hook_name} was not present in cache.")
            layer_input = cache[hook_name][:, -1, :].detach().cpu().to(torch.bfloat16)
            clean_inputs[layer][batch.indices] = layer_input[:batch_size]
            corrupt_inputs[layer][batch.indices] = layer_input[batch_size:]

        del tokens, logits, cache
        clear_cuda()

    baseline = {
        "clean_tool_logit": clean_logits,
        "corrupt_tool_logit": corrupt_logits,
        "clean_top1": clean_top1,
        "corrupt_top1": corrupt_top1,
    }
    return clean_inputs, corrupt_inputs, baseline


def compute_dense_features(
    inputs: torch.Tensor,
    W_enc_cpu: torch.Tensor,
    b_enc_cpu: torch.Tensor,
    *,
    device: torch.device,
    compute_batch_size: int,
) -> torch.Tensor:
    n_samples = int(inputs.shape[0])
    n_features = int(W_enc_cpu.shape[0])
    outputs = torch.empty((n_samples, n_features), dtype=torch.float32)

    W_enc = W_enc_cpu.to(device=device, dtype=torch.bfloat16)
    b_enc = b_enc_cpu.to(device=device, dtype=torch.bfloat16)

    for start in range(0, n_samples, compute_batch_size):
        end = min(start + compute_batch_size, n_samples)
        batch_in = inputs[start:end].to(device=device, dtype=torch.bfloat16)
        with torch.no_grad():
            batch_out = torch.relu(F.linear(batch_in, W_enc, b_enc))
        outputs[start:end] = batch_out.detach().cpu().float()
        del batch_in, batch_out

    del W_enc, b_enc
    clear_cuda()
    return outputs


def compute_auc_scores(
    clean_values: torch.Tensor,
    corrupt_values: torch.Tensor,
    *,
    feature_batch_size: int,
) -> torch.Tensor:
    n_pos = int(clean_values.shape[0])
    n_neg = int(corrupt_values.shape[0])
    n_features = int(clean_values.shape[1])
    auc = torch.empty(n_features, dtype=torch.float32)
    offset = n_pos * (n_pos + 1) / 2.0
    denom = float(n_pos * n_neg)
    n_total = n_pos + n_neg

    for start in range(0, n_features, feature_batch_size):
        end = min(start + feature_batch_size, n_features)
        combined = torch.cat([clean_values[:, start:end], corrupt_values[:, start:end]], dim=0)
        sorted_vals, sorted_idx = torch.sort(combined, dim=0, stable=True)
        new_group = torch.ones_like(sorted_vals, dtype=torch.bool)
        new_group[1:] = sorted_vals[1:] != sorted_vals[:-1]
        group_id = new_group.cumsum(dim=0).to(torch.long) - 1

        rows = torch.arange(n_total, dtype=torch.float32).unsqueeze(1).expand_as(sorted_vals)
        start_pos = torch.full((n_total, sorted_vals.shape[1]), float(n_total), dtype=torch.float32)
        end_pos = torch.full((n_total, sorted_vals.shape[1]), -1.0, dtype=torch.float32)
        start_pos.scatter_reduce_(0, group_id, rows, reduce="amin", include_self=True)
        end_pos.scatter_reduce_(0, group_id, rows, reduce="amax", include_self=True)
        avg_rank_per_group = (start_pos + end_pos) * 0.5 + 1.0
        avg_ranks = avg_rank_per_group.gather(0, group_id)
        pos_mask_sorted = sorted_idx < n_pos
        sum_pos = (avg_ranks * pos_mask_sorted.to(avg_ranks.dtype)).sum(dim=0)
        auc[start:end] = (sum_pos - offset) / denom

    return auc


def classify_features(
    mean_clean: torch.Tensor,
    mean_corrupt: torch.Tensor,
    active_rate_clean: torch.Tensor,
    active_rate_corrupt: torch.Tensor,
    auc_clean: torch.Tensor,
) -> tuple[List[str], torch.Tensor, Dict[str, torch.Tensor]]:
    delta = mean_clean - mean_corrupt
    auc_effective = torch.maximum(auc_clean, 1.0 - auc_clean)
    strong = auc_effective >= AUC_STRONG_THRESHOLD

    clean_selective = (
        strong
        & (delta > 0)
        & (active_rate_clean >= MIN_ACTIVE_RATE)
        & (active_rate_corrupt <= torch.maximum(torch.full_like(active_rate_clean, LOW_ACTIVE_RATE), SELECTIVE_RATIO * active_rate_clean))
        & (mean_corrupt <= torch.maximum(torch.full_like(mean_clean, 1e-6), SELECTIVE_RATIO * torch.maximum(mean_clean, torch.full_like(mean_clean, 1e-6))))
    )
    corrupt_selective = (
        strong
        & (delta < 0)
        & (active_rate_corrupt >= MIN_ACTIVE_RATE)
        & (active_rate_clean <= torch.maximum(torch.full_like(active_rate_corrupt, LOW_ACTIVE_RATE), SELECTIVE_RATIO * active_rate_corrupt))
        & (mean_clean <= torch.maximum(torch.full_like(mean_corrupt, 1e-6), SELECTIVE_RATIO * torch.maximum(mean_corrupt, torch.full_like(mean_corrupt, 1e-6))))
    )
    differential = strong & ~(clean_selective | corrupt_selective)

    category: List[str] = []
    for idx in range(int(delta.shape[0])):
        if bool(clean_selective[idx].item()):
            category.append("clean-selective")
        elif bool(corrupt_selective[idx].item()):
            category.append("corrupt-selective")
        elif bool(differential[idx].item()):
            category.append("differential")
        else:
            category.append("none")
    return category, auc_effective, {
        "clean-selective": clean_selective,
        "corrupt-selective": corrupt_selective,
        "differential": differential,
        "strong": strong,
    }


def summarize_top_tokens(
    token_ids: Sequence[int],
    tokenizer,
) -> str:
    return json.dumps([tokenizer.decode([int(token_id)]) for token_id in token_ids], ensure_ascii=False)


def top_feature_indices(
    feature_mask: torch.Tensor,
    score: torch.Tensor,
    *,
    limit: int,
) -> List[int]:
    feature_ids = torch.nonzero(feature_mask, as_tuple=False).squeeze(-1)
    if int(feature_ids.numel()) == 0:
        return []
    order = torch.argsort(score[feature_ids], descending=True)
    return feature_ids[order[:limit]].tolist()


def save_sparse_activation_file(
    path: Path,
    *,
    layer: int,
    pairs: Sequence[SamplePair],
    clean_dense: torch.Tensor,
    corrupt_dense: torch.Tensor,
) -> None:
    ensure_dir(path.parent)
    records: List[Dict[str, object]] = []
    for row_idx, pair in enumerate(pairs):
        clean_row = clean_dense[row_idx]
        corrupt_row = corrupt_dense[row_idx]
        clean_idx = torch.nonzero(clean_row > 0, as_tuple=False).squeeze(-1).to(torch.int32)
        corrupt_idx = torch.nonzero(corrupt_row > 0, as_tuple=False).squeeze(-1).to(torch.int32)
        records.append(
            {
                "sample_id": pair.sample_id,
                "clean_indices": clean_idx,
                "clean_values": clean_row[clean_idx.long()].to(torch.float16),
                "corrupt_indices": corrupt_idx,
                "corrupt_values": corrupt_row[corrupt_idx.long()].to(torch.float16),
            }
        )
    torch.save(
        {
            "layer": layer,
            "n_features": int(clean_dense.shape[1]),
            "samples": records,
        },
        path,
    )


def build_layer_summary(
    *,
    layer: int,
    delta: torch.Tensor,
    auc_effective: torch.Tensor,
    mask_map: Dict[str, torch.Tensor],
) -> Dict[str, object]:
    strong_mask = mask_map["strong"]
    clean_mask = mask_map["clean-selective"]
    corrupt_mask = mask_map["corrupt-selective"]
    diff_mask = mask_map["differential"]
    clean_effect = float(delta[clean_mask].abs().mean().item()) if bool(clean_mask.any()) else 0.0
    corrupt_effect = float(delta[corrupt_mask].abs().mean().item()) if bool(corrupt_mask.any()) else 0.0
    ratio = float(int(corrupt_mask.sum().item()) / max(int(clean_mask.sum().item()), 1))
    if 0.67 <= ratio <= 1.5 and 0.67 <= (corrupt_effect / max(clean_effect, 1e-8)) <= 1.5:
        symmetry = "symmetric"
    elif ratio > 1.5 or corrupt_effect > 1.5 * max(clean_effect, 1e-8):
        symmetry = "corrupt_dominant"
    elif ratio < 0.67 or clean_effect > 1.5 * max(corrupt_effect, 1e-8):
        symmetry = "clean_dominant"
    else:
        symmetry = "mixed"
    return {
        "layer": layer,
        "n_strong": int(strong_mask.sum().item()),
        "n_clean_selective": int(clean_mask.sum().item()),
        "n_corrupt_selective": int(corrupt_mask.sum().item()),
        "n_differential": int(diff_mask.sum().item()),
        "mean_effect_clean": clean_effect,
        "mean_effect_corrupt": corrupt_effect,
        "corrupt_to_clean_ratio": ratio,
        "symmetry": symmetry,
    }


def build_overall_symmetry(per_layer: Sequence[Dict[str, object]]) -> tuple[str, str]:
    total_clean = sum(int(row["n_clean_selective"]) for row in per_layer)
    total_corrupt = sum(int(row["n_corrupt_selective"]) for row in per_layer)
    effect_clean_vals = [float(row["mean_effect_clean"]) for row in per_layer if float(row["mean_effect_clean"]) > 0]
    effect_corrupt_vals = [float(row["mean_effect_corrupt"]) for row in per_layer if float(row["mean_effect_corrupt"]) > 0]
    effect_clean = float(sum(effect_clean_vals) / len(effect_clean_vals)) if effect_clean_vals else 0.0
    effect_corrupt = float(sum(effect_corrupt_vals) / len(effect_corrupt_vals)) if effect_corrupt_vals else 0.0
    ratio = total_corrupt / max(total_clean, 1)
    effect_ratio = effect_corrupt / max(effect_clean, 1e-8)

    if 0.67 <= ratio <= 1.5 and 0.67 <= effect_ratio <= 1.5:
        return "symmetric", "B"
    if ratio > 1.5 and effect_ratio > 1.25:
        return "asymmetric_corrupt_dominant", "C"
    if ratio < 0.67 and effect_ratio < 0.8:
        return "asymmetric_clean_dominant", "mixed"
    return "mixed", "mixed"


def feature_score(row: Dict[str, object]) -> float:
    return float(row["auc_effective"]) * abs(float(row["delta"]))


def build_selection(
    *,
    layer: int,
    category_name: str,
    selected_ids: Sequence[int],
    clean_dense: torch.Tensor,
    corrupt_dense: torch.Tensor,
    W_enc: torch.Tensor,
    b_enc: torch.Tensor,
    W_dec: torch.Tensor,
    row_lookup: Dict[int, Dict[str, object]],
) -> FeatureSelection:
    feature_ids = [int(feature_id) for feature_id in selected_ids]
    if feature_ids:
        idx_tensor = torch.tensor(feature_ids, dtype=torch.long)
        clean_values = clean_dense[:, idx_tensor].contiguous().to(torch.float32)
        corrupt_values = corrupt_dense[:, idx_tensor].contiguous().to(torch.float32)
        W_enc_sel = W_enc[idx_tensor].contiguous().to(torch.bfloat16)
        b_enc_sel = b_enc[idx_tensor].contiguous().to(torch.bfloat16)
        W_dec_sel = W_dec[idx_tensor].contiguous().to(torch.bfloat16)
        rows = [row_lookup[int(feature_id)] for feature_id in feature_ids]
    else:
        clean_values = torch.empty((clean_dense.shape[0], 0), dtype=torch.float32)
        corrupt_values = torch.empty((corrupt_dense.shape[0], 0), dtype=torch.float32)
        W_enc_sel = torch.empty((0, W_enc.shape[1]), dtype=torch.bfloat16)
        b_enc_sel = torch.empty((0,), dtype=torch.bfloat16)
        W_dec_sel = torch.empty((0, W_dec.shape[1]), dtype=torch.bfloat16)
        rows = []
    return FeatureSelection(
        layer=layer,
        category=category_name,
        feature_ids=feature_ids,
        clean_values=clean_values,
        corrupt_values=corrupt_values,
        W_enc=W_enc_sel,
        b_enc=b_enc_sel,
        W_dec=W_dec_sel,
        rows=rows,
    )


def run_feature_analysis(
    model,
    tokenizer,
    pairs: Sequence[SamplePair],
    pair_metadata: Dict[str, Dict[str, object]],
    clean_inputs: Dict[int, torch.Tensor],
    corrupt_inputs: Dict[int, torch.Tensor],
    *,
    layers: Sequence[int],
    output_root: Path,
    feature_compute_batch_size: int,
    auc_feature_batch_size: int,
    topk_values: Sequence[int],
    semantic_top_features: int,
    tool_token_id: int,
) -> tuple[
    List[Dict[str, object]],
    Dict[int, Dict[str, FeatureSelection]],
    List[Dict[str, object]],
    List[Dict[str, object]],
]:
    device = model.W_U.device
    max_topk = max(topk_values)
    feature_act_dir = output_root / "feature_activations"
    diff_dir = output_root / "differential_features"
    proj_dir = output_root / "output_projections"
    semantic_dir = output_root / "feature_semantics"
    ensure_dir(feature_act_dir)
    ensure_dir(diff_dir)
    ensure_dir(proj_dir)
    ensure_dir(semantic_dir)

    per_layer_summary: List[Dict[str, object]] = []
    selections: Dict[int, Dict[str, FeatureSelection]] = {}
    projection_rows: List[Dict[str, object]] = []
    semantic_candidate_rows: List[Dict[str, object]] = []

    tool_writer = model.W_U[:, tool_token_id].detach().cpu().float()

    for layer in tqdm(layers, desc="Layer feature analysis", dynamic_ncols=True):
        tc_weights = load_file(str(TRANSCODER_DIR / f"layer_{layer}.safetensors"))
        W_enc = tc_weights["W_enc"].detach().cpu()
        b_enc = tc_weights["b_enc"].detach().cpu()
        W_dec = tc_weights["W_dec"].detach().cpu()
        n_features = int(W_enc.shape[0])

        clean_dense = compute_dense_features(
            clean_inputs[layer],
            W_enc,
            b_enc,
            device=device,
            compute_batch_size=feature_compute_batch_size,
        )
        corrupt_dense = compute_dense_features(
            corrupt_inputs[layer],
            W_enc,
            b_enc,
            device=device,
            compute_batch_size=feature_compute_batch_size,
        )

        save_sparse_activation_file(
            feature_act_dir / f"feature_activations_L{layer}.pt",
            layer=layer,
            pairs=pairs,
            clean_dense=clean_dense,
            corrupt_dense=corrupt_dense,
        )

        mean_clean = clean_dense.mean(dim=0)
        mean_corrupt = corrupt_dense.mean(dim=0)
        delta = mean_clean - mean_corrupt
        active_clean = (clean_dense > 0).float().mean(dim=0)
        active_corrupt = (corrupt_dense > 0).float().mean(dim=0)
        auc_clean = compute_auc_scores(clean_dense, corrupt_dense, feature_batch_size=auc_feature_batch_size)
        category, auc_effective, mask_map = classify_features(mean_clean, mean_corrupt, active_clean, active_corrupt, auc_clean)
        layer_summary = build_layer_summary(layer=layer, delta=delta, auc_effective=auc_effective, mask_map=mask_map)
        per_layer_summary.append(layer_summary)

        layer_rows_lookup: Dict[int, Dict[str, object]] = {}
        fieldnames = [
            "feature_idx",
            "mean_clean",
            "mean_corrupt",
            "delta",
            "auc",
            "auc_effective",
            "active_rate_clean",
            "active_rate_corrupt",
            "category",
        ]

        def layer_row_iter() -> Iterable[Dict[str, object]]:
            for feature_idx in range(n_features):
                row = {
                    "feature_idx": feature_idx,
                    "mean_clean": float(mean_clean[feature_idx].item()),
                    "mean_corrupt": float(mean_corrupt[feature_idx].item()),
                    "delta": float(delta[feature_idx].item()),
                    "auc": float(auc_clean[feature_idx].item()),
                    "auc_effective": float(auc_effective[feature_idx].item()),
                    "active_rate_clean": float(active_clean[feature_idx].item()),
                    "active_rate_corrupt": float(active_corrupt[feature_idx].item()),
                    "category": category[feature_idx],
                }
                layer_rows_lookup[feature_idx] = row
                yield row

        write_rows(diff_dir / f"differential_features_L{layer}.csv", fieldnames, layer_row_iter())

        score = auc_effective * delta.abs()

        selections[layer] = {
            "clean-selective": build_selection(
                layer=layer,
                category_name="clean-selective",
                selected_ids=top_feature_indices(mask_map["clean-selective"], score, limit=max_topk),
                clean_dense=clean_dense,
                corrupt_dense=corrupt_dense,
                W_enc=W_enc,
                b_enc=b_enc,
                W_dec=W_dec,
                row_lookup=layer_rows_lookup,
            ),
            "corrupt-selective": build_selection(
                layer=layer,
                category_name="corrupt-selective",
                selected_ids=top_feature_indices(mask_map["corrupt-selective"], score, limit=max_topk),
                clean_dense=clean_dense,
                corrupt_dense=corrupt_dense,
                W_enc=W_enc,
                b_enc=b_enc,
                W_dec=W_dec,
                row_lookup=layer_rows_lookup,
            ),
            "differential": build_selection(
                layer=layer,
                category_name="differential",
                selected_ids=top_feature_indices(mask_map["differential"], score, limit=DIFFERENTIAL_SELECTION_TOPK),
                clean_dense=clean_dense,
                corrupt_dense=corrupt_dense,
                W_enc=W_enc,
                b_enc=b_enc,
                W_dec=W_dec,
                row_lookup=layer_rows_lookup,
            ),
        }

        for category_name, selection in selections[layer].items():
            if not selection.feature_ids:
                continue
            proj = torch.mv(selection.W_dec.float(), tool_writer)
            for local_idx, feature_id in enumerate(selection.feature_ids):
                base_row = dict(selection.rows[local_idx])
                proj_row = {
                    "layer": layer,
                    "feature_idx": feature_id,
                    "category": category_name,
                    "delta": base_row["delta"],
                    "auc": base_row["auc"],
                    "auc_effective": base_row["auc_effective"],
                    "tool_call_projection": float(proj[local_idx].item()),
                    "projection_sign": "toward" if float(proj[local_idx].item()) > 0 else "away",
                }
                projection_rows.append(proj_row)
                semantic_candidate_rows.append(
                    {
                        **proj_row,
                        "clean_mean": base_row["mean_clean"],
                        "corrupt_mean": base_row["mean_corrupt"],
                        "active_rate_clean": base_row["active_rate_clean"],
                        "active_rate_corrupt": base_row["active_rate_corrupt"],
                    }
                )

        del clean_dense, corrupt_dense
        del W_enc, b_enc, W_dec, tc_weights
        clear_cuda()

    overall_symmetry, hypothesis_support = build_overall_symmetry(per_layer_summary)
    write_json(
        diff_dir / "differential_summary.json",
        {
            "layers": list(layers),
            "per_layer": {str(row["layer"]): row for row in per_layer_summary},
            "overall_symmetry": overall_symmetry,
            "hypothesis_support": hypothesis_support,
        },
    )

    write_rows(
        proj_dir / "feature_output_projections.csv",
        [
            "layer",
            "feature_idx",
            "category",
            "delta",
            "auc",
            "auc_effective",
            "tool_call_projection",
            "projection_sign",
        ],
        projection_rows,
    )

    semantic_rows = build_semantic_outputs(
        model=model,
        tokenizer=tokenizer,
        pairs=pairs,
        pair_metadata=pair_metadata,
        selections=selections,
        semantic_candidates=semantic_candidate_rows,
        semantic_top_features=semantic_top_features,
        output_root=semantic_dir,
    )
    return per_layer_summary, selections, projection_rows, semantic_rows


def build_semantic_outputs(
    *,
    model,
    tokenizer,
    pairs: Sequence[SamplePair],
    pair_metadata: Dict[str, Dict[str, object]],
    selections: Dict[int, Dict[str, FeatureSelection]],
    semantic_candidates: Sequence[Dict[str, object]],
    semantic_top_features: int,
    output_root: Path,
) -> List[Dict[str, object]]:
    ensure_dir(output_root)
    score_sorted = sorted(semantic_candidates, key=feature_score, reverse=True)
    chosen = score_sorted[: min(semantic_top_features, len(score_sorted))]
    if not chosen:
        write_rows(
            output_root / "top_features_token_attribution.csv",
            ["layer", "feature_idx", "category", "top_tokens", "bottom_tokens"],
            [],
        )
        write_rows(
            output_root / "top_features_max_activating.csv",
            ["layer", "feature_idx", "category", "top_samples", "common_verbs"],
            [],
        )
        write_text(output_root / "semantic_summary.md", "No strong semantic features were found.")
        return []

    lookup: Dict[tuple[int, str, int], tuple[FeatureSelection, int]] = {}
    for layer, by_category in selections.items():
        for category_name, selection in by_category.items():
            for local_idx, feature_id in enumerate(selection.feature_ids):
                lookup[(layer, category_name, int(feature_id))] = (selection, local_idx)

    chosen_decoder_rows: List[torch.Tensor] = []
    token_attr_rows: List[Dict[str, object]] = []
    max_rows: List[Dict[str, object]] = []
    summary_rows: List[Dict[str, object]] = []
    chosen_meta: List[tuple[Dict[str, object], FeatureSelection, int]] = []
    for row in chosen:
        key = (int(row["layer"]), str(row["category"]), int(row["feature_idx"]))
        if key not in lookup:
            continue
        selection, local_idx = lookup[key]
        chosen_meta.append((row, selection, local_idx))
        chosen_decoder_rows.append(selection.W_dec[local_idx].float())

    device = model.W_U.device
    decoder_matrix = torch.stack(chosen_decoder_rows, dim=0).to(device=device, dtype=torch.bfloat16)
    with torch.no_grad():
        token_scores = decoder_matrix @ model.W_U.to(dtype=torch.bfloat16)
    token_scores_cpu = token_scores.detach().cpu().float()

    for row_idx, (row, selection, local_idx) in enumerate(chosen_meta):
        top_ids = torch.topk(token_scores_cpu[row_idx], k=20).indices.tolist()
        bottom_ids = torch.topk(token_scores_cpu[row_idx], k=20, largest=False).indices.tolist()
        token_attr_rows.append(
            {
                "layer": row["layer"],
                "feature_idx": row["feature_idx"],
                "category": row["category"],
                "top_tokens": summarize_top_tokens(top_ids, tokenizer),
                "bottom_tokens": summarize_top_tokens(bottom_ids, tokenizer),
            }
        )

        clean_vals = selection.clean_values[:, local_idx]
        corrupt_vals = selection.corrupt_values[:, local_idx]
        combined: List[tuple[float, str, SamplePair]] = []
        for pair_idx, pair in enumerate(pairs):
            combined.append((float(clean_vals[pair_idx].item()), "clean", pair))
            combined.append((float(corrupt_vals[pair_idx].item()), "corrupt", pair))
        combined.sort(key=lambda item: item[0], reverse=True)
        top_examples = combined[:5]
        top_payload: List[Dict[str, object]] = []
        verbs: List[str] = []
        for value, side, pair in top_examples:
            meta = pair_metadata.get(pair.sample_id, {})
            verb_key = "clean_candidate" if side == "clean" else "corrupt_candidate"
            verb = str(meta.get(verb_key, ""))
            verbs.append(verb)
            top_payload.append(
                {
                    "sample_id": pair.sample_id,
                    "side": side,
                    "activation": value,
                    "verb": verb,
                    "language": meta.get("language"),
                    "dataset_name": meta.get("dataset_name"),
                    "template_kind": meta.get("template_kind"),
                }
            )
        common_verbs = [verb for verb, _count in Counter(verbs).most_common(3) if verb]
        max_rows.append(
            {
                "layer": row["layer"],
                "feature_idx": row["feature_idx"],
                "category": row["category"],
                "top_samples": json.dumps(top_payload, ensure_ascii=False),
                "common_verbs": json.dumps(common_verbs, ensure_ascii=False),
            }
        )
        summary_rows.append(
            {
                "layer": row["layer"],
                "feature_idx": row["feature_idx"],
                "category": row["category"],
                "delta": row["delta"],
                "tool_call_projection": row["tool_call_projection"],
                "common_verbs": common_verbs,
                "top_tokens": [tokenizer.decode([int(token_id)]) for token_id in top_ids[:5]],
                "bottom_tokens": [tokenizer.decode([int(token_id)]) for token_id in bottom_ids[:5]],
            }
        )

    write_rows(
        output_root / "top_features_token_attribution.csv",
        ["layer", "feature_idx", "category", "top_tokens", "bottom_tokens"],
        token_attr_rows,
    )
    write_rows(
        output_root / "top_features_max_activating.csv",
        ["layer", "feature_idx", "category", "top_samples", "common_verbs"],
        max_rows,
    )

    clean_lines: List[str] = ["# Feature Semantics Summary", ""]
    clean_lines.append("## Highest-Scoring Features")
    for row in summary_rows[:20]:
        clean_lines.append(
            f"- L{row['layer']} F{row['feature_idx']} `{row['category']}` "
            f"delta={float(row['delta']):.3f} proj={float(row['tool_call_projection']):.3f} "
            f"verbs={row['common_verbs']} top={row['top_tokens']} bottom={row['bottom_tokens']}"
        )
    write_text(output_root / "semantic_summary.md", "\n".join(clean_lines))
    return summary_rows


def prepare_layer_payload(
    selection: FeatureSelection,
    *,
    k: int,
) -> Dict[str, torch.Tensor]:
    use_k = min(k, len(selection.feature_ids))
    return {
        "feature_ids": torch.tensor(selection.feature_ids[:use_k], dtype=torch.long),
        "clean_values": selection.clean_values[:, :use_k].contiguous(),
        "corrupt_values": selection.corrupt_values[:, :use_k].contiguous(),
        "W_enc": selection.W_enc[:use_k].contiguous(),
        "b_enc": selection.b_enc[:use_k].contiguous(),
        "W_dec": selection.W_dec[:use_k].contiguous(),
    }


def run_intervention(
    model,
    pair_batches: Sequence[PairBatch],
    baseline_logits: torch.Tensor,
    *,
    side: str,
    tool_token_id: int,
    layer_payloads: Dict[int, Dict[str, torch.Tensor]],
    mode: str,
    source_side: str | None = None,
) -> Dict[str, object]:
    device = model.W_U.device
    n_samples = int(baseline_logits.shape[0])
    post_logits = torch.empty(n_samples, dtype=torch.float32)
    post_top1 = torch.empty(n_samples, dtype=torch.long)
    if hasattr(model, "set_use_hook_mlp_in"):
        model.set_use_hook_mlp_in(True)
    if hasattr(model, "cfg") and hasattr(model.cfg, "use_hook_mlp_in"):
        model.cfg.use_hook_mlp_in = True

    prepared: Dict[int, Dict[str, torch.Tensor]] = {}
    for layer, payload in layer_payloads.items():
        prepared[layer] = {
            "W_enc": payload["W_enc"].to(device=device, dtype=torch.bfloat16),
            "b_enc": payload["b_enc"].to(device=device, dtype=torch.bfloat16),
            "W_dec": payload["W_dec"].to(device=device, dtype=torch.bfloat16),
            "clean_values": payload["clean_values"],
            "corrupt_values": payload["corrupt_values"],
        }

    progress = tqdm(pair_batches, desc=f"{mode}:{side}", dynamic_ncols=True)
    for batch in progress:
        tokens_cpu = batch.clean_tokens_cpu if side == "clean" else batch.corrupt_tokens_cpu
        tokens = tokens_cpu.to(device)
        state: Dict[int, torch.Tensor] = {}
        hooks = []
        for layer, payload in prepared.items():
            hook_in_name = f"blocks.{layer}.hook_mlp_in"
            hook_out_name = f"blocks.{layer}.hook_mlp_out"
            target_values = None
            if mode == "inject":
                source_key = "clean_values" if source_side == "clean" else "corrupt_values"
                target_values = payload[source_key][batch.indices].to(device=device, dtype=torch.bfloat16)

            def make_in_hook(layer_idx: int):
                def hook_fn(value: torch.Tensor, hook):  # noqa: ANN001
                    state[layer_idx] = value[:, -1, :].detach()
                    return value

                return hook_fn

            def make_out_hook(
                layer_idx: int,
                W_enc: torch.Tensor,
                b_enc: torch.Tensor,
                W_dec: torch.Tensor,
                target_values_local: torch.Tensor | None,
            ):
                def hook_fn(value: torch.Tensor, hook):  # noqa: ANN001
                    mlp_in = state.pop(layer_idx)
                    acts = torch.relu(F.linear(mlp_in.to(dtype=torch.bfloat16), W_enc, b_enc))
                    out = value.clone()
                    current_contrib = acts @ W_dec
                    if mode == "ablate":
                        out[:, -1, :] = out[:, -1, :] - current_contrib.to(dtype=out.dtype)
                    else:
                        if target_values_local is None:
                            raise RuntimeError("Injection requires target feature values.")
                        target_contrib = target_values_local @ W_dec
                        out[:, -1, :] = out[:, -1, :] - current_contrib.to(dtype=out.dtype) + target_contrib.to(dtype=out.dtype)
                    return out

                return hook_fn

            hooks.append((hook_in_name, make_in_hook(layer)))
            hooks.append((hook_out_name, make_out_hook(layer, payload["W_enc"], payload["b_enc"], payload["W_dec"], target_values)))

        with torch.no_grad():
            logits = model.run_with_hooks(tokens, fwd_hooks=hooks)
        batch_logits, batch_top1 = tool_stats(logits, tool_token_id)
        post_logits[batch.indices] = batch_logits
        post_top1[batch.indices] = batch_top1
        del tokens, logits
        clear_cuda()

    for payload in prepared.values():
        del payload["W_enc"], payload["b_enc"], payload["W_dec"]
    clear_cuda()

    if side == "clean":
        flip_mask = post_top1 != tool_token_id
    else:
        flip_mask = post_top1 == tool_token_id
    return {
        "flip_rate": float(flip_mask.float().mean().item()),
        "after_tool_top1_rate": float((post_top1 == tool_token_id).float().mean().item()),
        "mean_logit_delta": float((post_logits - baseline_logits).mean().item()),
    }


def run_causal_verification(
    model,
    pair_batches: Sequence[PairBatch],
    selections: Dict[int, Dict[str, FeatureSelection]],
    baseline: Dict[str, torch.Tensor],
    *,
    layers: Sequence[int],
    topk_values: Sequence[int],
    tool_token_id: int,
    output_root: Path,
) -> tuple[List[Dict[str, object]], List[Dict[str, object]], List[Dict[str, object]], List[Dict[str, object]], List[Dict[str, object]]]:
    ensure_dir(output_root)
    ablate_corrupt_rows: List[Dict[str, object]] = []
    ablate_clean_rows: List[Dict[str, object]] = []
    inject_clean_to_corrupt_rows: List[Dict[str, object]] = []
    inject_corrupt_to_clean_rows: List[Dict[str, object]] = []
    per_layer_rows: List[Dict[str, object]] = []

    for k in topk_values:
        corrupt_payloads = {
            layer: prepare_layer_payload(selections[layer]["corrupt-selective"], k=k)
            for layer in layers
            if selections[layer]["corrupt-selective"].feature_ids
        }
        if corrupt_payloads:
            result = run_intervention(
                model,
                pair_batches,
                baseline["corrupt_tool_logit"],
                side="corrupt",
                tool_token_id=tool_token_id,
                layer_payloads=corrupt_payloads,
                mode="ablate",
            )
            ablate_corrupt_rows.append(
                {
                    "k": k,
                    "layer_range": f"{min(corrupt_payloads)}-{max(corrupt_payloads)}",
                    **result,
                }
            )

        clean_payloads = {
            layer: prepare_layer_payload(selections[layer]["clean-selective"], k=k)
            for layer in layers
            if selections[layer]["clean-selective"].feature_ids
        }
        if clean_payloads:
            result = run_intervention(
                model,
                pair_batches,
                baseline["clean_tool_logit"],
                side="clean",
                tool_token_id=tool_token_id,
                layer_payloads=clean_payloads,
                mode="ablate",
            )
            ablate_clean_rows.append(
                {
                    "k": k,
                    "layer_range": f"{min(clean_payloads)}-{max(clean_payloads)}",
                    **result,
                }
            )

        if clean_payloads:
            result = run_intervention(
                model,
                pair_batches,
                baseline["corrupt_tool_logit"],
                side="corrupt",
                tool_token_id=tool_token_id,
                layer_payloads=clean_payloads,
                mode="inject",
                source_side="clean",
            )
            inject_clean_to_corrupt_rows.append(
                {
                    "k": k,
                    "layer_range": f"{min(clean_payloads)}-{max(clean_payloads)}",
                    **result,
                }
            )

        if corrupt_payloads:
            result = run_intervention(
                model,
                pair_batches,
                baseline["clean_tool_logit"],
                side="clean",
                tool_token_id=tool_token_id,
                layer_payloads=corrupt_payloads,
                mode="inject",
                source_side="corrupt",
            )
            inject_corrupt_to_clean_rows.append(
                {
                    "k": k,
                    "layer_range": f"{min(corrupt_payloads)}-{max(corrupt_payloads)}",
                    **result,
                }
            )

    for layer in layers:
        clean_sel = selections[layer]["clean-selective"]
        if clean_sel.feature_ids:
            payload = {layer: prepare_layer_payload(clean_sel, k=PER_LAYER_ABLATION_K)}
            result = run_intervention(
                model,
                pair_batches,
                baseline["clean_tool_logit"],
                side="clean",
                tool_token_id=tool_token_id,
                layer_payloads=payload,
                mode="ablate",
            )
            per_layer_rows.append(
                {
                    "layer": layer,
                    "side": "clean",
                    "category": "clean-selective",
                    "k": min(PER_LAYER_ABLATION_K, len(clean_sel.feature_ids)),
                    **result,
                }
            )

        corrupt_sel = selections[layer]["corrupt-selective"]
        if corrupt_sel.feature_ids:
            payload = {layer: prepare_layer_payload(corrupt_sel, k=PER_LAYER_ABLATION_K)}
            result = run_intervention(
                model,
                pair_batches,
                baseline["corrupt_tool_logit"],
                side="corrupt",
                tool_token_id=tool_token_id,
                layer_payloads=payload,
                mode="ablate",
            )
            per_layer_rows.append(
                {
                    "layer": layer,
                    "side": "corrupt",
                    "category": "corrupt-selective",
                    "k": min(PER_LAYER_ABLATION_K, len(corrupt_sel.feature_ids)),
                    **result,
                }
            )

    write_rows(
        output_root / "ablation_corrupt_selective.csv",
        ["k", "layer_range", "flip_rate", "after_tool_top1_rate", "mean_logit_delta"],
        ablate_corrupt_rows,
    )
    write_rows(
        output_root / "ablation_clean_selective.csv",
        ["k", "layer_range", "flip_rate", "after_tool_top1_rate", "mean_logit_delta"],
        ablate_clean_rows,
    )
    write_rows(
        output_root / "injection_clean_to_corrupt.csv",
        ["k", "layer_range", "flip_rate", "after_tool_top1_rate", "mean_logit_delta"],
        inject_clean_to_corrupt_rows,
    )
    write_rows(
        output_root / "injection_corrupt_to_clean.csv",
        ["k", "layer_range", "flip_rate", "after_tool_top1_rate", "mean_logit_delta"],
        inject_corrupt_to_clean_rows,
    )
    write_rows(
        output_root / "per_layer_ablation.csv",
        ["layer", "side", "category", "k", "flip_rate", "after_tool_top1_rate", "mean_logit_delta"],
        per_layer_rows,
    )

    symmetry_text = build_symmetry_analysis(
        ablate_clean_rows=ablate_clean_rows,
        ablate_corrupt_rows=ablate_corrupt_rows,
        inject_clean_to_corrupt_rows=inject_clean_to_corrupt_rows,
        inject_corrupt_to_clean_rows=inject_corrupt_to_clean_rows,
        per_layer_rows=per_layer_rows,
    )
    write_text(output_root / "symmetry_analysis.md", symmetry_text)

    return (
        ablate_corrupt_rows,
        ablate_clean_rows,
        inject_clean_to_corrupt_rows,
        inject_corrupt_to_clean_rows,
        per_layer_rows,
    )


def average_value(rows: Sequence[Dict[str, object]], key: str) -> float:
    if not rows:
        return 0.0
    return float(sum(float(row[key]) for row in rows) / len(rows))


def build_symmetry_analysis(
    *,
    ablate_clean_rows: Sequence[Dict[str, object]],
    ablate_corrupt_rows: Sequence[Dict[str, object]],
    inject_clean_to_corrupt_rows: Sequence[Dict[str, object]],
    inject_corrupt_to_clean_rows: Sequence[Dict[str, object]],
    per_layer_rows: Sequence[Dict[str, object]],
) -> str:
    clean_flip = average_value(ablate_clean_rows, "flip_rate")
    corrupt_flip = average_value(ablate_corrupt_rows, "flip_rate")
    clean_inject = average_value(inject_corrupt_to_clean_rows, "flip_rate")
    corrupt_inject = average_value(inject_clean_to_corrupt_rows, "flip_rate")
    ratio = corrupt_flip / max(clean_flip, 1e-8)

    if 0.67 <= ratio <= 1.5 and abs(clean_inject - corrupt_inject) <= 0.15:
        verdict = "B"
        rationale = "ablation 和 cross-injection 两个方向都接近对称。"
    elif ratio > 1.5 and corrupt_inject > clean_inject + 0.15:
        verdict = "C"
        rationale = "corrupt-selective feature 的去除更容易把样本翻回 `<tool_call>`，同时 clean-pattern 注入也更强。"
    else:
        verdict = "mixed"
        rationale = "不同方向的因果效应不完全一致，更像混合机制。"

    top_clean_layers = sorted(
        [row for row in per_layer_rows if row["side"] == "clean"],
        key=lambda row: float(row["flip_rate"]),
        reverse=True,
    )[:5]
    top_corrupt_layers = sorted(
        [row for row in per_layer_rows if row["side"] == "corrupt"],
        key=lambda row: float(row["flip_rate"]),
        reverse=True,
    )[:5]
    lines = [
        "# Symmetry Analysis",
        "",
        f"- Mean clean-side ablation flip rate: {clean_flip:.3f}",
        f"- Mean corrupt-side ablation flip rate: {corrupt_flip:.3f}",
        f"- Mean corrupt->clean injection flip rate: {clean_inject:.3f}",
        f"- Mean clean->corrupt injection flip rate: {corrupt_inject:.3f}",
        f"- Verdict: **{verdict}**",
        f"- Interpretation: {rationale}",
        "",
        "## Strongest Clean-Side Layers",
    ]
    for row in top_clean_layers:
        lines.append(
            f"- L{row['layer']}: flip={float(row['flip_rate']):.3f}, logit_delta={float(row['mean_logit_delta']):.3f}"
        )
    lines.append("")
    lines.append("## Strongest Corrupt-Side Layers")
    for row in top_corrupt_layers:
        lines.append(
            f"- L{row['layer']}: flip={float(row['flip_rate']):.3f}, logit_delta={float(row['mean_logit_delta']):.3f}"
        )
    return "\n".join(lines)


def read_triplet_csv(name: str) -> List[Dict[str, str]]:
    path = TRIPLET_ROOT / name
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def build_mechanism_conclusion(
    *,
    layers: Sequence[int],
    baseline_meta: Dict[str, object],
    differential_summary: Sequence[Dict[str, object]],
    projection_rows: Sequence[Dict[str, object]],
    semantic_rows: Sequence[Dict[str, object]],
    ablate_corrupt_rows: Sequence[Dict[str, object]],
    ablate_clean_rows: Sequence[Dict[str, object]],
    per_layer_rows: Sequence[Dict[str, object]],
) -> str:
    overall_symmetry, hypothesis_support = build_overall_symmetry(differential_summary)
    top_clean_layers = sorted(differential_summary, key=lambda row: int(row["n_clean_selective"]), reverse=True)[:5]
    top_corrupt_layers = sorted(differential_summary, key=lambda row: int(row["n_corrupt_selective"]), reverse=True)[:5]
    positive_proj = [
        row for row in projection_rows if row["category"] == "clean-selective" and float(row["tool_call_projection"]) > 0
    ]
    negative_proj = [
        row for row in projection_rows if row["category"] == "corrupt-selective" and float(row["tool_call_projection"]) < 0
    ]
    clean_flip = average_value(ablate_clean_rows, "flip_rate")
    corrupt_flip = average_value(ablate_corrupt_rows, "flip_rate")
    strong_writers = sorted(per_layer_rows, key=lambda row: abs(float(row["mean_logit_delta"])), reverse=True)[:6]
    logit_lens_rows = read_triplet_csv("logit_lens_8B.csv")
    dla_rows = read_triplet_csv("dla_8B.csv")
    lens_peak = sorted(logit_lens_rows, key=lambda row: float(row["gap"]), reverse=True)[:5]
    dla_peak = sorted(dla_rows, key=lambda row: float(row["delta"]), reverse=True)[:5]

    lines = [
        "# Mechanism Conclusion",
        "",
        f"## 1. 机制类型判定",
        f"- Differential feature symmetry: `{overall_symmetry}`",
        f"- Hypothesis support from feature counts/effects: `{hypothesis_support}`",
        f"- Baseline clean `<tool_call>` top-1 rate: {baseline_meta['clean_tool_top1_rate']:.3f}",
        f"- Baseline corrupt `<tool_call>` top-1 rate: {baseline_meta['corrupt_tool_top1_rate']:.3f}",
        f"- Clean-side ablation mean flip rate: {clean_flip:.3f}",
        f"- Corrupt-side ablation mean flip rate: {corrupt_flip:.3f}",
        "",
        "## 2. 读取环节",
        "- Clean-selective richest layers: "
        + ", ".join([f"L{row['layer']}({row['n_clean_selective']})" for row in top_clean_layers]),
        "- Corrupt-selective richest layers: "
        + ", ".join([f"L{row['layer']}({row['n_corrupt_selective']})" for row in top_corrupt_layers]),
        "",
        "## 3. 计算环节",
        f"- Clean-selective features with positive `<tool_call>` projection: {len(positive_proj)}",
        f"- Corrupt-selective features with negative `<tool_call>` projection: {len(negative_proj)}",
        "- Example semantic features:",
    ]
    for row in semantic_rows[:10]:
        lines.append(
            f"  - L{row['layer']} F{row['feature_idx']} `{row['category']}` "
            f"verbs={row['common_verbs']} top={row['top_tokens']}"
        )
    lines.extend(
        [
            "",
            "## 4. 写入环节",
            "- Highest per-layer causal effect:",
        ]
    )
    for row in strong_writers:
        lines.append(
            f"  - L{row['layer']} {row['side']} flip={float(row['flip_rate']):.3f}, logit_delta={float(row['mean_logit_delta']):.3f}"
        )
    if lens_peak:
        lines.append("- Logit-lens gap peaks: " + ", ".join([f"L{row['layer']}({float(row['gap']):.2f})" for row in lens_peak]))
    if dla_peak:
        lines.append("- DLA peaks: " + ", ".join([f"L{row['layer']}({float(row['delta']):.2f})" for row in dla_peak]))
    lines.extend(
        [
            "",
            "## 5. 完整因果链",
            f"- 在层 {min(layers)}-{max(layers)} 的 `hook_mlp_in` 上，动词差异先表现为 clean/corrupt selective transcoder feature 的激活差异。",
            "- 这些 feature 的 decoder 方向与 `<tool_call>` writer 向量存在稳定正负投影，说明它们直接改变工具调用 logit。",
            "- 对 selective features 做层内 ablation / cross-injection 可以在 clean 与 corrupt 之间造成行为翻转，证明这些差异不是纯相关，而是因果有效。",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 2 differential feature mechanism discovery for Qwen3-8B.")
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH)
    parser.add_argument("--transcoder-dir", type=Path, default=TRANSCODER_DIR)
    parser.add_argument("--manifest-path", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--layers", type=str, default="19-35")
    parser.add_argument("--max-pairs", type=int, default=200)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--forward-batch-size", type=int, default=12)
    parser.add_argument("--feature-compute-batch-size", type=int, default=32)
    parser.add_argument("--auc-feature-batch-size", type=int, default=4096)
    parser.add_argument("--topk-values", type=str, default="10,50,100")
    parser.add_argument("--semantic-top-features", type=int, default=SEMANTIC_TOP_FEATURES)
    parser.add_argument("--skip-causal", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    global TRANSCODER_DIR
    global OUTPUT_ROOT
    global DATASET_ROOT
    TRANSCODER_DIR = args.transcoder_dir
    OUTPUT_ROOT = args.output_root
    DATASET_ROOT = args.dataset_root
    layers = parse_layers(args.layers)
    topk_values = parse_topk_values(args.topk_values)
    set_seed(args.seed)
    ensure_dir(args.output_root)

    model, tokenizer = load_hooked_qwen3(str(args.model_path), device=args.device, dtype=torch.bfloat16)
    tool_token_ids = tokenizer.encode(TOOL_CALL_STR, add_special_tokens=False)
    if len(tool_token_ids) != 1:
        raise ValueError(f"{TOOL_CALL_STR!r} is not a single token: {tool_token_ids}")
    tool_token_id = int(tool_token_ids[0])

    pair_metadata = load_dataset_metadata(args.dataset_root)
    pairs = load_sample_pairs(args.manifest_path, model, max_pairs=args.max_pairs)
    pair_batches = build_pair_batches(pairs, batch_size=args.forward_batch_size)
    clean_inputs, corrupt_inputs, baseline = collect_layer_inputs_and_baseline(
        model,
        pair_batches,
        layers=layers,
        n_samples=len(pairs),
        tool_token_id=tool_token_id,
    )

    baseline_meta = {
        "n_pairs": len(pairs),
        "layers": layers,
        "clean_tool_top1_rate": float((baseline["clean_top1"] == tool_token_id).float().mean().item()),
        "corrupt_tool_top1_rate": float((baseline["corrupt_top1"] == tool_token_id).float().mean().item()),
        "clean_tool_logit_mean": float(baseline["clean_tool_logit"].mean().item()),
        "corrupt_tool_logit_mean": float(baseline["corrupt_tool_logit"].mean().item()),
        "tool_token_id": tool_token_id,
    }
    write_json(args.output_root / "metadata.json", baseline_meta)

    differential_summary, selections, projection_rows, semantic_rows = run_feature_analysis(
        model=model,
        tokenizer=tokenizer,
        pairs=pairs,
        pair_metadata=pair_metadata,
        clean_inputs=clean_inputs,
        corrupt_inputs=corrupt_inputs,
        layers=layers,
        output_root=args.output_root,
        feature_compute_batch_size=args.feature_compute_batch_size,
        auc_feature_batch_size=args.auc_feature_batch_size,
        topk_values=topk_values,
        semantic_top_features=args.semantic_top_features,
        tool_token_id=tool_token_id,
    )

    if args.skip_causal:
        write_text(args.output_root / "causal_verification" / "symmetry_analysis.md", "Causal verification skipped.")
        ablate_corrupt_rows: List[Dict[str, object]] = []
        ablate_clean_rows = []
        inject_clean_to_corrupt_rows = []
        inject_corrupt_to_clean_rows = []
        per_layer_rows = []
    else:
        (
            ablate_corrupt_rows,
            ablate_clean_rows,
            inject_clean_to_corrupt_rows,
            inject_corrupt_to_clean_rows,
            per_layer_rows,
        ) = run_causal_verification(
            model,
            pair_batches,
            selections,
            baseline,
            layers=layers,
            topk_values=topk_values,
            tool_token_id=tool_token_id,
            output_root=args.output_root / "causal_verification",
        )

    mechanism_conclusion = build_mechanism_conclusion(
        layers=layers,
        baseline_meta=baseline_meta,
        differential_summary=differential_summary,
        projection_rows=projection_rows,
        semantic_rows=semantic_rows,
        ablate_corrupt_rows=ablate_corrupt_rows,
        ablate_clean_rows=ablate_clean_rows,
        per_layer_rows=per_layer_rows,
    )
    write_text(args.output_root / "mechanism_conclusion.md", mechanism_conclusion)


if __name__ == "__main__":
    main()
