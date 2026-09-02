#!/usr/bin/env python3
"""Qwen3.5-4B feature-level suppression analysis.

The run is deliberately split by the published v5 manifest:

* the 200 ``train`` pairs fit the L29 clean-minus-corrupt direction and select
  features;
* the 300 ``heldout`` pairs are used only for validation and causal feature
  swaps.

The two available Qwen3.5 Transcoders are checkpoints for L28 and L29.  Their
MLP input is captured at the HF ``layer.mlp`` pre-hook, and their decoder rows
are projected onto the train-fitted L29 direction.  The causal screen swaps
the selected Transcoder contribution into the actual MLP output; it does not
replace the whole MLP with the Transcoder reconstruction.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import importlib.util
import json
import math
import os
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from path_defaults import QWEN35_4B_PATH, QWEN35_4B_TRANSCODER_PATH, REBUTTAL_ROOT, V5_DATASET_ROOT


# The local Qwen3.5 Transformers build probes sklearn while importing the
# model.  The existing Qwen3.5 runners use this same small compatibility shim.
_ORIGINAL_FIND_SPEC = importlib.util.find_spec


def _patched_find_spec(name: str, package: str | None = None):
    if name == "sklearn" and os.environ.get("MECH_ENABLE_TRANSFORMERS_SKLEARN", "0") != "1":
        return None
    return _ORIGINAL_FIND_SPEC(name, package)


importlib.util.find_spec = _patched_find_spec
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
finally:
    importlib.util.find_spec = _ORIGINAL_FIND_SPEC


DEFAULT_MODEL_PATH = QWEN35_4B_PATH
DEFAULT_DATASET_ROOT = V5_DATASET_ROOT / "qwen35_4b"
DEFAULT_TC_ROOT = QWEN35_4B_TRANSCODER_PATH
DEFAULT_OUTPUT_ROOT = REBUTTAL_ROOT / "09_qwen35_transcoder_feature_analysis" / "qwen35_4b"
TOOL_CALL_TEXT = "<tool_call>"


@dataclass(frozen=True)
class Pair:
    index: int
    sample_id: str
    split: str
    clean_verb: str
    corrupt_verb: str
    clean_text: str
    corrupt_text: str
    clean_tokens: torch.Tensor
    corrupt_tokens: torch.Tensor

    @property
    def token_len(self) -> int:
        return int(self.clean_tokens.shape[0])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--transcoder-root", type=Path, default=DEFAULT_TC_ROOT)
    parser.add_argument("--layer28-checkpoint", type=Path, default=None)
    parser.add_argument("--layer29-checkpoint", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--decision-layer", type=int, default=29)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--feature-batch-size", type=int, default=16)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--causal-k", type=int, default=5)
    parser.add_argument("--consistency-threshold", type=float, default=None)
    parser.add_argument("--random-consistency-threshold", type=float, default=None)
    parser.add_argument("--token-top-k", type=int, default=8)
    parser.add_argument("--example-top-k", type=int, default=6)
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=20260802)
    parser.add_argument("--skip-causal", action="store_true")
    parser.add_argument("--skip-feature-semantics", action="store_true")
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_text(path: Path, text: str) -> None:
    ensure_dir(path.parent)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def clear_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def get_dtype(name: str) -> torch.dtype:
    return getattr(torch, name)


def resolve_text_model(model) -> Any:
    for path in ("model.language_model", "model", "base_model.model", "base_model", "language_model", "transformer"):
        obj = model
        try:
            for part in path.split("."):
                obj = getattr(obj, part)
        except AttributeError:
            continue
        if hasattr(obj, "layers"):
            return obj
        if hasattr(obj, "model") and hasattr(obj.model, "layers"):
            return obj.model
    raise RuntimeError("Could not locate the Qwen3.5 text decoder stack")


def load_model(model_path: Path, dtype_name: str, device_name: str):
    dtype = get_dtype(dtype_name)
    if device_name.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        device_map: str | dict[str, int] | None = {"": 0}
    else:
        device_map = None
    kwargs: dict[str, Any] = {
        "dtype": dtype,
        "low_cpu_mem_usage": True,
        "trust_remote_code": True,
    }
    if device_map is not None:
        kwargs["device_map"] = device_map
    model = AutoModelForCausalLM.from_pretrained(str(model_path), **kwargs)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return model, tokenizer


def model_device(model) -> torch.device:
    return next(model.parameters()).device


def load_pairs(
    dataset_root: Path,
    tokenizer,
    split: str,
    *,
    token_length_policy: str = "error",
) -> list[Pair]:
    if token_length_policy not in {"error", "skip", "allow"}:
        raise ValueError(f"Unknown token_length_policy: {token_length_policy!r}")
    manifest_path = dataset_root / "manifest.jsonl"
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    rows: list[dict[str, Any]] = []
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                if str(row.get("split")) == split:
                    rows.append(row)
    pairs: list[Pair] = []
    for row in rows:
        clean_rel = str(row.get("clean_relpath") or row.get("clean_filename") or "")
        corrupt_rel = str(row.get("corrupt_relpath") or row.get("corrupt_filename") or "")
        if not clean_rel or not corrupt_rel:
            raise ValueError(f"Manifest row has no clean/corrupt path: {row}")
        clean_path = dataset_root / clean_rel
        corrupt_path = dataset_root / corrupt_rel
        clean_text = clean_path.read_text(encoding="utf-8")
        corrupt_text = corrupt_path.read_text(encoding="utf-8")
        clean_ids = torch.tensor(tokenizer(clean_text, add_special_tokens=False)["input_ids"], dtype=torch.long)
        corrupt_ids = torch.tensor(tokenizer(corrupt_text, add_special_tokens=False)["input_ids"], dtype=torch.long)
        if clean_ids.numel() != corrupt_ids.numel():
            if token_length_policy == "skip":
                continue
            if token_length_policy == "error":
                raise ValueError(
                    f"Token-length mismatch for {row['sample_id']}: "
                    f"{clean_ids.numel()} vs {corrupt_ids.numel()}"
                )
        pairs.append(
            Pair(
                index=len(pairs),
                sample_id=str(row["sample_id"]),
                split=split,
                clean_verb=str(row.get("clean_candidate") or row.get("clean_verb") or ""),
                corrupt_verb=str(row.get("corrupt_candidate") or row.get("corrupt_verb") or ""),
                clean_text=clean_text,
                corrupt_text=corrupt_text,
                clean_tokens=clean_ids,
                corrupt_tokens=corrupt_ids,
            )
        )
    if not pairs:
        raise RuntimeError(f"No {split} pairs found under {dataset_root}")
    return pairs


def iter_batches(pairs: Sequence[Pair], batch_size: int, pad_token_id: int):
    ordered = sorted(pairs, key=lambda item: (item.token_len, item.index))
    for start in range(0, len(ordered), max(batch_size, 1)):
        batch = ordered[start : start + max(batch_size, 1)]
        max_len = max(item.token_len for item in batch)

        def pad(tokens: torch.Tensor) -> torch.Tensor:
            padding = max_len - int(tokens.shape[0])
            if padding <= 0:
                return tokens
            return F.pad(tokens, (padding, 0), value=pad_token_id)

        clean = torch.stack([pad(item.clean_tokens) for item in batch], dim=0)
        corrupt = torch.stack([pad(item.corrupt_tokens) for item in batch], dim=0)
        attention_mask = torch.stack(
            [F.pad(torch.ones(item.token_len, dtype=torch.long), (padding := max_len - item.token_len, 0), value=0) for item in batch],
            dim=0,
        )
        yield batch, clean, corrupt, attention_mask


def tensor_output(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (tuple, list)) and value and isinstance(value[0], torch.Tensor):
        return value[0]
    raise TypeError(f"Unsupported module output: {type(value).__name__}")


def replace_tensor_output(original: Any, replacement: torch.Tensor) -> Any:
    if isinstance(original, torch.Tensor):
        return replacement
    if isinstance(original, tuple):
        return (replacement, *original[1:])
    if isinstance(original, list):
        return [replacement, *original[1:]]
    raise TypeError(f"Unsupported module output: {type(original).__name__}")


def last_logits(outputs) -> torch.Tensor:
    logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
    return logits[:, -1, :].detach().float().cpu()


def metric_tensors(logits: torch.Tensor, tool_id: int) -> dict[str, torch.Tensor]:
    tool_logit = logits[:, tool_id]
    non_tool_logits = logits.clone()
    non_tool_logits[:, tool_id] = -torch.inf
    max_non_tool = non_tool_logits.max(dim=-1).values
    top1 = logits.argmax(dim=-1)
    tool_prob = torch.softmax(logits, dim=-1)[:, tool_id]
    rank = 1 + (logits > tool_logit[:, None]).sum(dim=-1)
    return {
        "tool_logit": tool_logit,
        "tool_prob": tool_prob,
        "top1_margin": tool_logit - max_non_tool,
        "log_odds": tool_logit - torch.logsumexp(non_tool_logits, dim=-1),
        "top1": top1,
        "rank": rank,
    }


def register_capture_hooks(model_layers, capture_layers: Sequence[int], decision_layer: int, holders: dict[str, Any]):
    handles = []
    for layer_id in capture_layers:
        mlp = model_layers[layer_id].mlp

        def make_pre_hook(saved_layer: int):
            def hook(_module, inputs):
                if not inputs or not isinstance(inputs[0], torch.Tensor):
                    raise TypeError(f"L{saved_layer} MLP pre-hook did not receive a tensor")
                holders["mlp"][saved_layer] = inputs[0][:, -1, :].detach().float().cpu()

            return hook

        handles.append(mlp.register_forward_pre_hook(make_pre_hook(layer_id)))

    def decision_hook(_module, _inputs, output):
        holders["decision"] = tensor_output(output)[:, -1, :].detach().float().cpu()

    handles.append(model_layers[decision_layer].register_forward_hook(decision_hook))
    return handles


def capture_dataset(
    model,
    tokenizer,
    pairs: Sequence[Pair],
    *,
    capture_layers: Sequence[int],
    decision_layer: int,
    tool_id: int,
    batch_size: int,
    label: str,
) -> dict[str, Any]:
    layers = resolve_text_model(model).layers
    device = model_device(model)
    inputs = {side: {layer: [] for layer in capture_layers} for side in ("clean", "corrupt")}
    states = {side: [] for side in ("clean", "corrupt")}
    metric_parts = {side: defaultdict(list) for side in ("clean", "corrupt")}
    metadata: list[dict[str, Any]] = []
    pad_token_id = int(tokenizer.pad_token_id)
    batches = list(iter_batches(pairs, batch_size, pad_token_id))
    for batch, clean_cpu, corrupt_cpu, attention_mask_cpu in tqdm(batches, desc=f"Capture {label}", dynamic_ncols=True):
        tokens = torch.cat([clean_cpu, corrupt_cpu], dim=0).to(device)
        attention_mask = torch.cat([attention_mask_cpu, attention_mask_cpu], dim=0).to(device)
        holders: dict[str, Any] = {"mlp": {}, "decision": None}
        handles = register_capture_hooks(layers, capture_layers, decision_layer, holders)
        try:
            with torch.inference_mode():
                outputs = model(
                    input_ids=tokens,
                    attention_mask=attention_mask,
                    use_cache=False,
                    logits_to_keep=1,
                    return_dict=True,
                )
            logits = last_logits(outputs)
        finally:
            for handle in handles:
                handle.remove()
        n = len(batch)
        if holders["decision"] is None or set(holders["mlp"]) != set(capture_layers):
            raise RuntimeError(f"{label}: capture hook did not fire for all requested layers")
        for side, offset in (("clean", 0), ("corrupt", n)):
            side_slice = slice(offset, offset + n)
            states[side].append(holders["decision"][side_slice])
            metrics = metric_tensors(logits[side_slice], tool_id)
            for key, value in metrics.items():
                metric_parts[side][key].append(value)
            for layer_id in capture_layers:
                inputs[side][layer_id].append(holders["mlp"][layer_id][side_slice])
        for item in batch:
            metadata.append(
                {
                    "index": int(item.index),
                    "sample_id": item.sample_id,
                    "split": item.split,
                    "clean_verb": item.clean_verb,
                    "corrupt_verb": item.corrupt_verb,
                }
            )
        del tokens, attention_mask, outputs, logits, holders
        clear_cuda()

    # Batches are length-grouped, so restore every array to manifest order.
    order = [int(row["index"]) for row in metadata]
    inverse = {index: position for position, index in enumerate(order)}

    def restore(rows: list[torch.Tensor]) -> torch.Tensor:
        joined = torch.cat(rows, dim=0)
        return joined[torch.tensor([inverse[i] for i in range(len(pairs))], dtype=torch.long)].contiguous()

    result_inputs = {side: {layer: restore(inputs[side][layer]) for layer in capture_layers} for side in inputs}
    result_states = {side: restore(states[side]) for side in states}
    result_metrics = {
        side: {key: restore(parts) for key, parts in metric_parts[side].items()} for side in metric_parts
    }
    result_metadata = sorted(metadata, key=lambda row: int(row["index"]))
    return {
        "inputs": result_inputs,
        "states": result_states,
        "metrics": result_metrics,
        "metadata": result_metadata,
    }


def load_transcoder(path: Path, expected_layer: int) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    actual_layer = int(payload.get("layer", -1))
    if actual_layer != expected_layer:
        raise ValueError(f"{path}: checkpoint records L{actual_layer}, expected L{expected_layer}")
    tc = payload["transcoder"]
    weights = {
        "W_enc": tc["W_enc"].detach().cpu().contiguous(),
        "b_enc": tc["b_enc"].detach().cpu().contiguous(),
        "W_dec": tc["W_dec"].detach().cpu().float().contiguous(),
        "b_dec": tc["b_dec"].detach().cpu().float().contiguous(),
        "checkpoint_step": int(payload.get("step", -1)),
        "d_model": int(payload.get("d_model", tc["W_dec"].shape[-1])),
        "d_feature": int(payload.get("d_feature", tc["W_dec"].shape[0])),
    }
    del payload, tc
    gc.collect()
    return weights


def collect_feature_acts(
    inputs: torch.Tensor,
    W_enc: torch.Tensor,
    b_enc: torch.Tensor,
    *,
    device: torch.device,
    batch_size: int,
    selected_ids: torch.Tensor | None = None,
    return_acts: bool = False,
) -> dict[str, torch.Tensor]:
    W = W_enc.to(device=device, dtype=torch.bfloat16)
    b = b_enc.to(device=device, dtype=torch.bfloat16)
    n = int(inputs.shape[0])
    d_feature = int(W.shape[0])
    sum_act = torch.zeros(d_feature, dtype=torch.float64)
    active = torch.zeros(d_feature, dtype=torch.int64)
    selected_parts: list[torch.Tensor] = []
    act_parts: list[torch.Tensor] = []
    for start in range(0, n, max(batch_size, 1)):
        end = min(start + max(batch_size, 1), n)
        batch = inputs[start:end].to(device=device, dtype=torch.bfloat16)
        with torch.inference_mode():
            acts = F.relu(F.linear(batch, W, b))
        acts_float = acts.float()
        sum_act += acts_float.sum(dim=0).cpu().double()
        active += (acts > 0).sum(dim=0).cpu().long()
        if selected_ids is not None:
            selected_parts.append(acts_float.index_select(1, selected_ids.to(device)).cpu())
        if return_acts:
            act_parts.append(acts_float.cpu())
        del batch, acts, acts_float
    del W, b
    clear_cuda()
    result = {
        "mean": (sum_act / max(n, 1)).float(),
        "active_rate": (active.float() / max(n, 1)),
    }
    if selected_ids is not None:
        result["selected"] = torch.cat(selected_parts, dim=0) if selected_parts else torch.empty((0, selected_ids.numel()))
    if return_acts:
        result["acts"] = torch.cat(act_parts, dim=0) if act_parts else torch.empty((0, d_feature))
    return result


def metric_summary(metrics: dict[str, torch.Tensor], tool_id: int) -> dict[str, Any]:
    top1 = metrics["top1"] == tool_id
    return {
        "n": int(top1.numel()),
        "mean_tool_logit": float(metrics["tool_logit"].mean().item()),
        "mean_tool_probability": float(metrics["tool_prob"].mean().item()),
        "mean_top1_margin": float(metrics["top1_margin"].mean().item()),
        "mean_log_odds": float(metrics["log_odds"].mean().item()),
        "tool_call_top1_rate": float(top1.float().mean().item()),
        "median_tool_rank": float(metrics["rank"].median().item()),
        "top10_rate": float((metrics["rank"] <= 10).float().mean().item()),
    }


def quadrant_summary(delta: torch.Tensor, beta: torch.Tensor, kappa: torch.Tensor) -> dict[str, float | int | None]:
    masks = {
        "suppressor": (delta < 0) & (beta < 0),
        "driver": (delta > 0) & (beta > 0),
        "clean_higher_away": (delta > 0) & (beta < 0),
        "corrupt_higher_toward": (delta < 0) & (beta > 0),
    }
    values: dict[str, float | int | None] = {
        "G_kappa": float(kappa.sum().item()),
        "n_features": int(delta.numel()),
    }
    for name, mask in masks.items():
        values[f"{name}_mass"] = float(kappa[mask].abs().sum().item())
        values[f"{name}_signed_kappa"] = float(kappa[mask].sum().item())
        values[f"n_{name}"] = int(mask.sum().item())
    s = float(values["suppressor_mass"])
    e = float(values["driver_mass"])
    values["suppressor_driver_ratio"] = s / e if e else None
    values["S_minus_E"] = s - e
    values["quadrant_reconstruction"] = s + e - float(values["clean_higher_away_mass"]) - float(values["corrupt_higher_toward_mass"])
    values["quadrant_reconstruction_residual"] = float(values["G_kappa"]) - float(values["quadrant_reconstruction"])
    return values


def select_top_rows(
    layer: int,
    train_stats: dict[str, torch.Tensor],
    heldout_stats: dict[str, torch.Tensor],
    category: str,
    limit: int,
    consistency_threshold: float | None = None,
) -> list[dict[str, Any]]:
    delta = train_stats["delta"]
    beta = train_stats["beta"]
    if category == "suppressor":
        mask = (delta < 0) & (beta < 0)
    elif category == "driver":
        mask = (delta > 0) & (beta > 0)
    elif category == "clean_higher_away":
        mask = (delta > 0) & (beta < 0)
    elif category == "corrupt_higher_toward":
        mask = (delta < 0) & (beta > 0)
    else:
        raise ValueError(category)
    if consistency_threshold is not None and category in {"suppressor", "driver"}:
        if "consistency" not in train_stats:
            raise KeyError("Train pair consistency was not computed")
        mask = mask & (train_stats["consistency"] >= float(consistency_threshold))
    ids = torch.nonzero(mask, as_tuple=False).flatten()
    order = torch.argsort(train_stats["kappa"][ids].abs(), descending=True)
    rows: list[dict[str, Any]] = []
    for feature_id in ids[order[:limit]].tolist():
        rows.append(
            {
                "layer": int(layer),
                "feature_idx": int(feature_id),
                "category": category,
                "train_delta_activation": float(train_stats["delta"][feature_id].item()),
                "heldout_delta_activation": float(heldout_stats["delta"][feature_id].item()),
                "beta_mu": float(train_stats["beta"][feature_id].item()),
                "train_kappa": float(train_stats["kappa"][feature_id].item()),
                "heldout_kappa": float(heldout_stats["kappa"][feature_id].item()),
                "train_abs_kappa": float(abs(train_stats["kappa"][feature_id].item())),
                "heldout_abs_kappa": float(abs(heldout_stats["kappa"][feature_id].item())),
                "train_active_rate_clean": float(train_stats["clean_active_rate"][feature_id].item()),
                "train_active_rate_corrupt": float(train_stats["corrupt_active_rate"][feature_id].item()),
                "heldout_active_rate_clean": float(heldout_stats["clean_active_rate"][feature_id].item()),
                "heldout_active_rate_corrupt": float(heldout_stats["corrupt_active_rate"][feature_id].item()),
                "train_consistency": float(train_stats["consistency"][feature_id].item()) if "consistency" in train_stats else None,
            }
        )
    return rows


def build_feature_lookup(stats_by_layer: dict[int, dict[str, torch.Tensor]], layer: int, feature_id: int, category: str) -> dict[str, Any]:
    stats = stats_by_layer[layer]
    return {
        "layer": int(layer),
        "feature_idx": int(feature_id),
        "category": category,
        "train_delta_activation": float(stats["delta"][feature_id].item()),
        "beta_mu": float(stats["beta"][feature_id].item()),
        "train_kappa": float(stats["kappa"][feature_id].item()),
        "train_abs_kappa": float(abs(stats["kappa"][feature_id].item())),
        "train_consistency": float(stats["consistency"][feature_id].item()) if "consistency" in stats else None,
    }


def decoder_token_rows(
    rows: Sequence[dict[str, Any]],
    weights_by_layer: dict[int, dict[str, Any]],
    model,
    tokenizer,
    *,
    tool_id: int,
    token_top_k: int,
) -> list[dict[str, Any]]:
    output_embeddings = model.get_output_embeddings().weight.detach()
    device = output_embeddings.device
    result: list[dict[str, Any]] = []
    for start in range(0, len(rows), 32):
        chunk = rows[start : start + 32]
        for row in chunk:
            decoder = weights_by_layer[int(row["layer"])]
            vector = decoder["W_dec"][int(row["feature_idx"])].to(device=device, dtype=output_embeddings.dtype)
            scores = torch.mv(output_embeddings, vector).float()
            top_ids = torch.topk(scores, k=token_top_k).indices.tolist()
            bottom_ids = torch.topk(scores, k=token_top_k, largest=False).indices.tolist()
            result.append(
                {
                    **row,
                    "tool_call_projection": float(scores[tool_id].item()),
                    "top_tokens": json.dumps(
                        [tokenizer.decode([int(token)], clean_up_tokenization_spaces=False) for token in top_ids],
                        ensure_ascii=False,
                    ),
                    "bottom_tokens": json.dumps(
                        [tokenizer.decode([int(token)], clean_up_tokenization_spaces=False) for token in bottom_ids],
                        ensure_ascii=False,
                    ),
                }
            )
        del chunk
    return result


def add_activation_diagnostics(
    rows: list[dict[str, Any]],
    selected_ids_by_layer: dict[int, list[int]],
    train_selected: dict[int, torch.Tensor],
    heldout_selected: dict[int, torch.Tensor],
    train_pairs: Sequence[Pair],
    heldout_pairs: Sequence[Pair],
) -> None:
    lookup_by_layer = {layer: {feature_id: column for column, feature_id in enumerate(ids)} for layer, ids in selected_ids_by_layer.items()}
    pair_info = {
        "train": train_pairs,
        "heldout": heldout_pairs,
    }
    acts_by_split = {
        "train": train_selected,
        "heldout": heldout_selected,
    }
    row_lookup = {(int(row["layer"]), int(row["feature_idx"])): row for row in rows}
    for layer, ids in selected_ids_by_layer.items():
        for feature_id in ids:
            row = row_lookup.get((layer, feature_id))
            if row is None:
                continue
            col = lookup_by_layer[layer][feature_id]
            examples: list[tuple[float, str, str, str]] = []
            train_values = train_selected[layer][:, col]
            heldout_values = heldout_selected[layer][:, col]
            beta = float(row["beta_mu"])
            category = str(row["category"])
            for split, values in (("train", train_values), ("heldout", heldout_values)):
                pairs = pair_info[split]
                for index, value in enumerate(values.tolist()):
                    pair = pairs[index]
                    examples.append((float(value), split, "clean", pair.clean_verb))
                    examples.append((float(value), split, "corrupt", pair.corrupt_verb))
            examples.sort(key=lambda item: item[0], reverse=True)
            row["top_activation_examples"] = json.dumps(
                [
                    {"activation": value, "split": split, "side": side, "verb": verb}
                    for value, split, side, verb in examples[:6]
                ],
                ensure_ascii=False,
            )
            # Pair-level sign consistency is evaluated with the train-fitted
            # category sign; it is a diagnostic, not a selection criterion.
            for split, values in (("train", train_values), ("heldout", heldout_values)):
                pairs = pair_info[split]
                # The selected matrix stores clean and corrupt columns next to
                # each other only when generated by caller; this branch is
                # filled below when the matrices are provided as a 2-column
                # concatenation.
                _ = pairs, beta, category, values


def group_rows(rows: Sequence[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["layer"])].append(row)
    return dict(grouped)


def make_delta_vectors(
    group: Sequence[dict[str, Any]],
    *,
    selected_acts: dict[int, dict[str, torch.Tensor]],
    weights_by_layer: dict[int, dict[str, Any]],
    indices: Sequence[int],
    direction: str,
) -> dict[int, torch.Tensor]:
    if direction not in {"to_clean", "to_corrupt"}:
        raise ValueError(direction)
    grouped = group_rows(group)
    output: dict[int, torch.Tensor] = {}
    index_tensor = torch.tensor(list(indices), dtype=torch.long)
    for layer, layer_rows in grouped.items():
        ids = [int(row["feature_idx"]) for row in layer_rows]
        layer_acts = selected_acts[layer]
        available_ids = [int(feature_id) for feature_id in layer_acts["ids"].tolist()]
        missing = [feature_id for feature_id in ids if feature_id not in available_ids]
        if missing:
            raise KeyError(f"Selected activation cache is missing layer {layer} features {missing}")
        positions = torch.tensor([available_ids.index(feature_id) for feature_id in ids], dtype=torch.long)
        clean = layer_acts["clean"].index_select(0, index_tensor).index_select(1, positions)
        corrupt = layer_acts["corrupt"].index_select(0, index_tensor).index_select(1, positions)
        source_to_target = clean - corrupt if direction == "to_clean" else corrupt - clean
        decoder = weights_by_layer[layer]["W_dec"].index_select(0, torch.tensor(ids, dtype=torch.long))
        output[layer] = source_to_target.double().matmul(decoder.double()).float()
    return output


def make_delta_cache(
    group: Sequence[dict[str, Any]],
    *,
    selected_acts: dict[int, dict[str, torch.Tensor]],
    weights_by_layer: dict[int, dict[str, Any]],
) -> dict[int, torch.Tensor]:
    """Precompute the clean-minus-corrupt decoder contribution for all pairs."""

    grouped = group_rows(group)
    output: dict[int, torch.Tensor] = {}
    for layer, layer_rows in grouped.items():
        ids = [int(row["feature_idx"]) for row in layer_rows]
        layer_acts = selected_acts[layer]
        available_ids = [int(feature_id) for feature_id in layer_acts["ids"].tolist()]
        missing = [feature_id for feature_id in ids if feature_id not in available_ids]
        if missing:
            raise KeyError(f"Selected activation cache is missing layer {layer} features {missing}")
        positions = torch.tensor([available_ids.index(feature_id) for feature_id in ids], dtype=torch.long)
        clean = layer_acts["clean"].index_select(1, positions).float()
        corrupt = layer_acts["corrupt"].index_select(1, positions).float()
        decoder = weights_by_layer[layer]["W_dec"].index_select(0, torch.tensor(ids, dtype=torch.long)).float()
        output[layer] = (clean - corrupt).matmul(decoder)
    return output


def baseline_row_for_index(metrics: dict[str, torch.Tensor], index: int, tool_id: int) -> dict[str, float | int]:
    return {
        "tool_logit": float(metrics["tool_logit"][index].item()),
        "tool_probability": float(metrics["tool_prob"][index].item()),
        "top1_margin": float(metrics["top1_margin"][index].item()),
        "log_odds": float(metrics["log_odds"][index].item()),
        "tool_rank": int(metrics["rank"][index].item()),
        "is_tool_top1": int(metrics["top1"][index].item() == tool_id),
    }


def run_intervention(
    model,
    tokenizer,
    pairs: Sequence[Pair],
    *,
    group_name: str,
    group: Sequence[dict[str, Any]],
    side: str,
    direction: str,
    delta_cache: dict[int, torch.Tensor],
    selected_acts: dict[int, dict[str, torch.Tensor]],
    weights_by_layer: dict[int, dict[str, Any]],
    baseline_metrics: dict[str, torch.Tensor],
    baseline_states: dict[str, torch.Tensor],
    u: torch.Tensor,
    layers,
    decision_layer: int,
    tool_id: int,
    batch_size: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    device = model_device(model)
    total_rows: list[dict[str, Any]] = []
    pad_token_id = int(tokenizer.pad_token_id)
    for batch, clean_cpu, corrupt_cpu, attention_mask_cpu in tqdm(iter_batches(pairs, batch_size, pad_token_id), desc=f"Causal {group_name}/{side}", dynamic_ncols=True, leave=False):
        indices = [item.index for item in batch]
        index_tensor = torch.tensor(indices, dtype=torch.long)
        sign = 1.0 if direction == "to_clean" else -1.0
        deltas = {layer: sign * values.index_select(0, index_tensor) for layer, values in delta_cache.items()}
        tokens = (clean_cpu if side == "clean" else corrupt_cpu).to(device)
        attention_mask = attention_mask_cpu.to(device)
        handles = []
        for layer, delta_cpu in deltas.items():
            def make_mlp_hook(delta: torch.Tensor):
                def hook(_module, _inputs, output):
                    value = tensor_output(output).clone()
                    value[:, -1, :] = value[:, -1, :] + delta.to(device=value.device, dtype=value.dtype)
                    return replace_tensor_output(output, value)

                return hook

            handles.append(layers[layer].mlp.register_forward_hook(make_mlp_hook(delta_cpu)))
        gate_holder: dict[str, torch.Tensor] = {}

        def gate_hook(_module, _inputs, output):
            gate_holder["state"] = tensor_output(output)[:, -1, :].detach().float().cpu()

        handles.append(layers[decision_layer].register_forward_hook(gate_hook))
        try:
            with torch.inference_mode():
                outputs = model(input_ids=tokens, attention_mask=attention_mask, use_cache=False, logits_to_keep=1, return_dict=True)
            logits = last_logits(outputs)
        finally:
            for handle in handles:
                handle.remove()
        if "state" not in gate_holder:
            raise RuntimeError("Decision-layer hook did not fire during intervention")
        metrics = metric_tensors(logits, tool_id)
        gate_scores = gate_holder["state"].matmul(u.float())
        base_gate = baseline_states[side].index_select(0, torch.tensor(indices)).matmul(u.float())
        for local, item in enumerate(batch):
            base = baseline_row_for_index(baseline_metrics[side], item.index, tool_id)
            new_top1 = int(metrics["top1"][local].item() == tool_id)
            new_rank = int(metrics["rank"][local].item())
            total_rows.append(
                {
                    "group": group_name,
                    "side": side,
                    "direction": direction,
                    "sample_id": item.sample_id,
                    "pair_index": int(item.index),
                    "intervened_tool_logit": float(metrics["tool_logit"][local].item()),
                    "intervened_tool_probability": float(metrics["tool_prob"][local].item()),
                    "intervened_top1_margin": float(metrics["top1_margin"][local].item()),
                    "intervened_log_odds": float(metrics["log_odds"][local].item()),
                    "intervened_tool_rank": new_rank,
                    "intervened_is_tool_top1": new_top1,
                    "tool_logit_delta": float(metrics["tool_logit"][local].item()) - base["tool_logit"],
                    "tool_probability_delta": float(metrics["tool_prob"][local].item()) - base["tool_probability"],
                    "top1_margin_delta": float(metrics["top1_margin"][local].item()) - base["top1_margin"],
                    "log_odds_delta": float(metrics["log_odds"][local].item()) - base["log_odds"],
                    "gate_score": float(gate_scores[local].item()),
                    "gate_score_delta": float(gate_scores[local].item() - base_gate[local].item()),
                    "baseline_tool_logit": base["tool_logit"],
                    "baseline_tool_probability": base["tool_probability"],
                    "baseline_tool_rank": base["tool_rank"],
                    "baseline_is_tool_top1": base["is_tool_top1"],
                    "strict_recovery": int(side == "corrupt" and not base["is_tool_top1"] and new_top1),
                    "strict_drop": int(side == "clean" and bool(base["is_tool_top1"]) and not new_top1),
                }
            )
        del tokens, outputs, logits, metrics, gate_holder
        clear_cuda()
    metric_keys = [
        "intervened_tool_logit",
        "intervened_tool_probability",
        "gate_score",
        "tool_logit_delta",
        "tool_probability_delta",
        "gate_score_delta",
    ]
    n = max(len(total_rows), 1)
    summary: dict[str, Any] = {
        "group": group_name,
        "side": side,
        "direction": direction,
        "n": len(total_rows),
        "mean_intervened_tool_logit": sum(float(row["intervened_tool_logit"]) for row in total_rows) / n,
        "mean_intervened_tool_probability": sum(float(row["intervened_tool_probability"]) for row in total_rows) / n,
        "tool_call_top1_rate": sum(int(row["intervened_is_tool_top1"]) for row in total_rows) / n,
        "median_tool_rank": float(torch.tensor([int(row["intervened_tool_rank"]) for row in total_rows]).median().item()) if total_rows else math.nan,
        "top10_rate": sum(int(int(row["intervened_tool_rank"]) <= 10) for row in total_rows) / n,
        "mean_tool_logit_delta": sum(float(row["tool_logit_delta"]) for row in total_rows) / n,
        "mean_tool_probability_delta": sum(float(row["tool_probability_delta"]) for row in total_rows) / n,
        "mean_top1_margin_delta": sum(float(row["top1_margin_delta"]) for row in total_rows) / n,
        "mean_log_odds_delta": sum(float(row["log_odds_delta"]) for row in total_rows) / n,
        "mean_gate_score": sum(float(row["gate_score"]) for row in total_rows) / n,
        "mean_gate_score_delta": sum(float(row["gate_score_delta"]) for row in total_rows) / n,
        "strict_recovery_rate": sum(int(row["strict_recovery"]) for row in total_rows) / n,
        "strict_drop_rate": sum(int(row["strict_drop"]) for row in total_rows) / n,
        "feature_count": len(group),
        "layers": dict(Counter(int(row["layer"]) for row in group)),
    }
    _ = metric_keys
    return summary, total_rows


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    ensure_dir(args.output_root)

    layer_ids = [28, 29]
    checkpoint_paths = {
        28: args.layer28_checkpoint or (args.transcoder_root / "checkpoint_step_0070000.pt"),
        29: args.layer29_checkpoint or (args.transcoder_root / "checkpoint_step_0065000.pt"),
    }
    model, tokenizer = load_model(args.model_path, args.dtype, args.device)
    tool_ids = tokenizer.encode(TOOL_CALL_TEXT, add_special_tokens=False)
    if len(tool_ids) != 1:
        raise ValueError(f"{TOOL_CALL_TEXT!r} is not a single token: {tool_ids}")
    tool_id = int(tool_ids[0])
    text_model = resolve_text_model(model)
    layers = text_model.layers
    if args.decision_layer < 0 or args.decision_layer >= len(layers):
        raise ValueError(f"Invalid decision layer L{args.decision_layer} for {len(layers)} layers")

    train_pairs = load_pairs(args.dataset_root, tokenizer, "train")
    heldout_pairs = load_pairs(args.dataset_root, tokenizer, "heldout")
    if len(train_pairs) != 200 or len(heldout_pairs) != 300:
        raise ValueError(f"Expected 200/300 split, got {len(train_pairs)}/{len(heldout_pairs)}")

    train = capture_dataset(
        model,
        tokenizer,
        train_pairs,
        capture_layers=layer_ids,
        decision_layer=args.decision_layer,
        tool_id=tool_id,
        batch_size=args.batch_size,
        label="train-200",
    )
    heldout = capture_dataset(
        model,
        tokenizer,
        heldout_pairs,
        capture_layers=layer_ids,
        decision_layer=args.decision_layer,
        tool_id=tool_id,
        batch_size=args.batch_size,
        label="heldout-300",
    )

    cache_payload = {
        "train_inputs": train["inputs"],
        "train_states": train["states"],
        "heldout_inputs": heldout["inputs"],
        "heldout_states": heldout["states"],
        "decision_layer": args.decision_layer,
        "layers": layer_ids,
    }
    torch.save(cache_payload, args.output_root / "activation_cache.pt")
    write_json(
        args.output_root / "baseline_summary.json",
        {
            "train": {side: metric_summary(train["metrics"][side], tool_id) for side in ("clean", "corrupt")},
            "heldout": {side: metric_summary(heldout["metrics"][side], tool_id) for side in ("clean", "corrupt")},
        },
    )
    write_csv(
        args.output_root / "pair_metadata.csv",
        train["metadata"] + heldout["metadata"],
    )

    mu_delta = train["states"]["clean"].mean(dim=0) - train["states"]["corrupt"].mean(dim=0)
    mu_norm = float(mu_delta.norm().item())
    if mu_norm <= 0:
        raise RuntimeError("L29 clean-minus-corrupt direction has zero norm")
    u = mu_delta / mu_norm
    torch.save({"mu_delta": mu_delta, "u": u, "layer": args.decision_layer}, args.output_root / "train_mu_delta_L29.pt")

    weights_by_layer: dict[int, dict[str, Any]] = {}
    for layer_id in layer_ids:
        print(f"Loading Transcoder checkpoint for L{layer_id}: {checkpoint_paths[layer_id]}", flush=True)
        weights_by_layer[layer_id] = load_transcoder(checkpoint_paths[layer_id], layer_id)
        if weights_by_layer[layer_id]["d_model"] != int(train["inputs"]["clean"][layer_id].shape[-1]):
            raise ValueError(f"L{layer_id}: Transcoder/model dimension mismatch")

    train_stats_by_layer: dict[int, dict[str, torch.Tensor]] = {}
    heldout_stats_by_layer: dict[int, dict[str, torch.Tensor]] = {}
    need_train_consistency = args.consistency_threshold is not None or args.random_consistency_threshold is not None
    for layer_id in layer_ids:
        weights = weights_by_layer[layer_id]
        train_clean = collect_feature_acts(
            train["inputs"]["clean"][layer_id],
            weights["W_enc"],
            weights["b_enc"],
            device=model_device(model),
            batch_size=args.feature_batch_size,
            return_acts=need_train_consistency,
        )
        train_corrupt = collect_feature_acts(
            train["inputs"]["corrupt"][layer_id],
            weights["W_enc"],
            weights["b_enc"],
            device=model_device(model),
            batch_size=args.feature_batch_size,
            return_acts=need_train_consistency,
        )
        heldout_clean = collect_feature_acts(
            heldout["inputs"]["clean"][layer_id],
            weights["W_enc"],
            weights["b_enc"],
            device=model_device(model),
            batch_size=args.feature_batch_size,
        )
        heldout_corrupt = collect_feature_acts(
            heldout["inputs"]["corrupt"][layer_id],
            weights["W_enc"],
            weights["b_enc"],
            device=model_device(model),
            batch_size=args.feature_batch_size,
        )
        beta = weights["W_dec"].matmul(u.float())
        train_delta = train_clean["mean"] - train_corrupt["mean"]
        heldout_delta = heldout_clean["mean"] - heldout_corrupt["mean"]
        train_stats = {
            "delta": train_delta,
            "beta": beta,
            "kappa": train_delta * beta,
            "clean_active_rate": train_clean["active_rate"],
            "corrupt_active_rate": train_corrupt["active_rate"],
        }
        if need_train_consistency:
            pair_contribution = (train_clean["acts"] - train_corrupt["acts"]).double() * beta.double().unsqueeze(0)
            train_stats["consistency"] = (pair_contribution > 0.0).float().mean(dim=0)
            del pair_contribution
        train_stats_by_layer[layer_id] = train_stats
        heldout_stats_by_layer[layer_id] = {
            "delta": heldout_delta,
            "beta": beta,
            "kappa": heldout_delta * beta,
            "clean_active_rate": heldout_clean["active_rate"],
            "corrupt_active_rate": heldout_corrupt["active_rate"],
        }
        for split_name, stats in (("train", train_stats_by_layer[layer_id]), ("heldout", heldout_stats_by_layer[layer_id])):
            write_json(
                args.output_root / f"L{layer_id}" / f"{split_name}_quadrants.json",
                quadrant_summary(stats["delta"], stats["beta"], stats["kappa"]),
            )
        del train_clean, train_corrupt, heldout_clean, heldout_corrupt
        clear_cuda()

    all_summary_rows: list[dict[str, Any]] = []
    for layer_id in layer_ids:
        for split_name, stats in (("train", train_stats_by_layer[layer_id]), ("heldout", heldout_stats_by_layer[layer_id])):
            row = {"layer": layer_id, "split": split_name, **quadrant_summary(stats["delta"], stats["beta"], stats["kappa"])}
            all_summary_rows.append(row)
    write_csv(args.output_root / "quadrant_summary.csv", all_summary_rows)

    top_rows: list[dict[str, Any]] = []
    categories = ("suppressor", "driver", "clean_higher_away", "corrupt_higher_toward")
    for layer_id in layer_ids:
        for category in categories:
            top_rows.extend(
                select_top_rows(
                    layer_id,
                    train_stats_by_layer[layer_id],
                    heldout_stats_by_layer[layer_id],
                    category,
                    args.top_k,
                    args.consistency_threshold,
                )
            )

    global_suppressors = sorted(
        [row for row in top_rows if row["category"] == "suppressor"],
        key=lambda row: (-float(row["train_abs_kappa"]), int(row["layer"]), int(row["feature_idx"])),
    )
    global_drivers = sorted(
        [row for row in top_rows if row["category"] == "driver"],
        key=lambda row: (-float(row["train_abs_kappa"]), int(row["layer"]), int(row["feature_idx"])),
    )
    if len(global_suppressors) < args.causal_k or len(global_drivers) < args.causal_k:
        raise RuntimeError(
            f"Not enough selected train-consistent features for causal-k={args.causal_k}: "
            f"suppressors={len(global_suppressors)}, drivers={len(global_drivers)}"
        )
    causal_suppressors = global_suppressors[: args.causal_k]
    causal_drivers = global_drivers[: args.causal_k]
    reserved = {(int(row["layer"]), int(row["feature_idx"])) for row in causal_suppressors + causal_drivers}
    rng = random.Random(args.seed)
    random_rows: list[dict[str, Any]] = []
    s_counts = Counter(int(row["layer"]) for row in causal_suppressors)
    for layer_id, count in sorted(s_counts.items()):
        d_feature = int(weights_by_layer[layer_id]["d_feature"])
        if args.random_consistency_threshold is not None:
            stats = train_stats_by_layer[layer_id]
            stable_mask = (
                (stats["delta"] < 0)
                & (stats["beta"] < 0)
                & (stats["consistency"] >= float(args.random_consistency_threshold))
            )
            stable_ids = torch.nonzero(stable_mask, as_tuple=False).flatten().tolist()
        else:
            stable_ids = list(range(d_feature))
        pool = [feature_id for feature_id in stable_ids if (layer_id, feature_id) not in reserved]
        if len(pool) < count:
            raise RuntimeError(f"L{layer_id}: only {len(pool)} stable random candidates for {count} controls")
        chosen = rng.sample(pool, count)
        for feature_id in chosen:
            random_rows.append(build_feature_lookup(train_stats_by_layer, layer_id, feature_id, "random_layer_matched"))

    selected_ids_by_layer: dict[int, list[int]] = {layer: [] for layer in layer_ids}
    selected_rows_for_cache = (causal_suppressors + causal_drivers + random_rows) if args.skip_feature_semantics else (top_rows + random_rows)
    for row in selected_rows_for_cache:
        selected_ids_by_layer[int(row["layer"])].append(int(row["feature_idx"]))
    for layer_id in selected_ids_by_layer:
        selected_ids_by_layer[layer_id] = sorted(set(selected_ids_by_layer[layer_id]))

    # Recompute only the small selected columns for causal swaps and semantic
    # examples.  Full feature matrices are never retained.
    selected_acts: dict[str, dict[int, dict[str, torch.Tensor]]] = {"train": {}, "heldout": {}}
    for layer_id in layer_ids:
        ids = torch.tensor(selected_ids_by_layer[layer_id], dtype=torch.long)
        weights = weights_by_layer[layer_id]
        for split_name, payload in (("train", train), ("heldout", heldout)):
            clean = collect_feature_acts(
                payload["inputs"]["clean"][layer_id],
                weights["W_enc"],
                weights["b_enc"],
                device=model_device(model),
                batch_size=args.feature_batch_size,
                selected_ids=ids,
            )["selected"]
            corrupt = collect_feature_acts(
                payload["inputs"]["corrupt"][layer_id],
                weights["W_enc"],
                weights["b_enc"],
                device=model_device(model),
                batch_size=args.feature_batch_size,
                selected_ids=ids,
            )["selected"]
            selected_acts[split_name][layer_id] = {"ids": ids, "clean": clean, "corrupt": corrupt}

    # Add pair-level consistency and decoder/token semantics to the selected
    # top rows.  The random control is intentionally not semantically labeled.
    if args.skip_feature_semantics:
        top_rows_with_tokens = top_rows
    else:
        top_rows_with_tokens = decoder_token_rows(
            top_rows,
            weights_by_layer,
            model,
            tokenizer,
            tool_id=tool_id,
            token_top_k=args.token_top_k,
        )
        for row in top_rows_with_tokens:
            layer_id = int(row["layer"])
            feature_id = int(row["feature_idx"])
            column = selected_ids_by_layer[layer_id].index(feature_id)
            beta = float(row["beta_mu"])
            expected_positive = torch.tensor(float(row["train_delta_activation"]) * beta > 0).item()
            for split_name, pairs in (("train", train_pairs), ("heldout", heldout_pairs)):
                clean = selected_acts[split_name][layer_id]["clean"][:, column]
                corrupt = selected_acts[split_name][layer_id]["corrupt"][:, column]
                contribution = (clean - corrupt) * beta
                row[f"{split_name}_pair_sign_consistency"] = float((contribution > 0).float().mean().item())
                examples: list[dict[str, Any]] = []
                for side, values in (("clean", clean), ("corrupt", corrupt)):
                    order = torch.argsort(values, descending=True)[: args.example_top_k]
                    for index in order.tolist():
                        pair = pairs[index]
                        examples.append(
                            {
                                "activation": float(values[index].item()),
                                "split": split_name,
                                "side": side,
                                "sample_id": pair.sample_id,
                                "verb": pair.clean_verb if side == "clean" else pair.corrupt_verb,
                            }
                        )
                examples.sort(key=lambda item: float(item["activation"]), reverse=True)
                row[f"{split_name}_max_activating_examples"] = json.dumps(examples[: args.example_top_k], ensure_ascii=False)
                _ = expected_positive

    write_csv(args.output_root / "top_features.csv", top_rows_with_tokens)
    write_csv(args.output_root / "causal_feature_selection.csv", causal_suppressors + causal_drivers + random_rows)

    intervention_summaries: list[dict[str, Any]] = []
    intervention_rows: list[dict[str, Any]] = []
    if not args.skip_causal:
        groups = {
            "suppressor_top": causal_suppressors,
            "driver_top": causal_drivers,
            "random_layer_matched": random_rows,
        }
        delta_caches = {
            group_name: make_delta_cache(
                group,
                selected_acts={layer: selected_acts["heldout"][layer] for layer in layer_ids},
                weights_by_layer=weights_by_layer,
            )
            for group_name, group in groups.items()
            if group
        }
        for group_name, group in groups.items():
            if not group:
                continue
            for side, direction in (("clean", "to_corrupt"), ("corrupt", "to_clean")):
                summary, rows = run_intervention(
                    model,
                    tokenizer,
                    heldout_pairs,
                    group_name=group_name,
                    group=group,
                    side=side,
                    direction=direction,
                    delta_cache=delta_caches[group_name],
                    selected_acts={layer: selected_acts["heldout"][layer] for layer in layer_ids},
                    weights_by_layer=weights_by_layer,
                    baseline_metrics=heldout["metrics"],
                    baseline_states=heldout["states"],
                    u=u,
                    layers=layers,
                    decision_layer=args.decision_layer,
                    tool_id=tool_id,
                    batch_size=args.batch_size,
                )
                intervention_summaries.append(summary)
                intervention_rows.extend(rows)
        write_csv(args.output_root / "causal_intervention_summary.csv", intervention_summaries)
        write_csv(args.output_root / "causal_intervention_per_sample.csv", intervention_rows)

    baseline_payload = {
        "train": {side: metric_summary(train["metrics"][side], tool_id) for side in ("clean", "corrupt")},
        "heldout": {side: metric_summary(heldout["metrics"][side], tool_id) for side in ("clean", "corrupt")},
    }
    run_config = {
        "model_path": str(args.model_path),
        "dataset_root": str(args.dataset_root),
        "manifest": str(args.dataset_root / "manifest.jsonl"),
        "manifest_sha256": sha256_file(args.dataset_root / "manifest.jsonl"),
        "transcoder_checkpoints": {str(layer): str(path) for layer, path in checkpoint_paths.items()},
        "transcoder_steps": {str(layer): weights_by_layer[layer]["checkpoint_step"] for layer in layer_ids},
        "layers_analyzed": layer_ids,
        "decision_layer": args.decision_layer,
        "train_pairs": len(train_pairs),
        "heldout_pairs": len(heldout_pairs),
        "tool_call_token_id": tool_id,
        "tool_call_token_text": tokenizer.decode([tool_id], clean_up_tokenization_spaces=False),
        "dtype": args.dtype,
        "batch_size": args.batch_size,
        "feature_batch_size": args.feature_batch_size,
        "top_k_per_category_per_layer": args.top_k,
        "causal_k": args.causal_k,
        "consistency_threshold": args.consistency_threshold,
        "random_consistency_threshold": args.random_consistency_threshold,
        "skip_feature_semantics": args.skip_feature_semantics,
        "seed": args.seed,
        "vector_fit": "mean(final decoder-block state clean - corrupt) at L29 over the 200 train pairs",
        "feature_activation": "relu(F.linear(MLP pre-hook input, W_enc, b_enc))",
        "kappa": "(mean_clean_activation - mean_corrupt_activation) * (W_dec @ unit_mu_delta)",
        "causal_intervention": "add selected decoder contribution difference to the actual HF MLP output at the final prompt position",
        "heldout_not_used_for_selection": True,
    }
    write_json(args.output_root / "run_config.json", run_config)
    write_json(args.output_root / "baseline_summary.json", baseline_payload)

    heldout_rows = [row for row in all_summary_rows if row["split"] == "heldout"]
    md: list[str] = [
        "# Qwen3.5-4B Transcoder feature analysis",
        "",
        "The 200 train pairs fit the L29 clean-minus-corrupt direction and select features; the 300 held-out pairs are validation only.",
        "",
        f"- Transcoders: L28 checkpoint step `{weights_by_layer[28]['checkpoint_step']}`, L29 checkpoint step `{weights_by_layer[29]['checkpoint_step']}`.",
        f"- Fitted direction: L{args.decision_layer} decoder-block output; norm `{mu_norm:.4f}`.",
        f"- Tool token: `{TOOL_CALL_TEXT}` (ID `{tool_id}`).",
        "",
        "## Held-out quadrant summary",
        "",
        "| layer | suppressor mass | driver mass | S/E | S-E | quadrant residual |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in heldout_rows:
        md.append(
            f"| L{row['layer']} | {float(row['suppressor_mass']):.4f} | {float(row['driver_mass']):.4f} | "
            f"{float(row['suppressor_driver_ratio']):.3f} | {float(row['S_minus_E']):+.4f} | "
            f"{float(row['quadrant_reconstruction_residual']):+.2e} |"
        )
    md.extend(["", "## Interpretation", ""])
    if all(float(row["suppressor_driver_ratio"] or 0.0) > 1.0 for row in heldout_rows):
        md.append("Both analyzed layers have more vector-aligned corrupt-higher suppressor mass than clean-higher driver mass on held-out prompts.")
    else:
        md.append("The suppressor-over-driver pattern is not uniform across the two analyzed layers; the layer-level table should be read rather than collapsed into a blanket replication claim.")
    if intervention_summaries:
        md.extend(["", "## Held-out causal feature swaps", "", "| group | side | direction | mean gate Δ | mean margin Δ | mean log-odds Δ | mean tool-logit Δ | top-1 | recovery | drop |", "|---|---|---|---:|---:|---:|---:|---:|---:|---:|"])
        for row in intervention_summaries:
            md.append(
                f"| {row['group']} | {row['side']} | {row['direction']} | {float(row['mean_gate_score_delta']):+.4f} | "
                f"{float(row['mean_top1_margin_delta']):+.4f} | {float(row['mean_log_odds_delta']):+.4f} | "
                f"{float(row['mean_tool_logit_delta']):+.4f} | {100.0*float(row['tool_call_top1_rate']):.1f}% | "
                f"{100.0*float(row['strict_recovery_rate']):.1f}% | {100.0*float(row['strict_drop_rate']):.1f}% |"
            )
        md.append("")
        md.append("The causal table is the direct replication check: suppressor swaps should move clean prompts downward and corrupt prompts upward if the Qwen3 suppression account transfers.")
    write_text(args.output_root / "summary.md", "\n".join(md))

    print(json.dumps({"output_root": str(args.output_root), "baseline": baseline_payload, "heldout_quadrants": heldout_rows, "causal": intervention_summaries}, indent=2))


if __name__ == "__main__":
    main()
