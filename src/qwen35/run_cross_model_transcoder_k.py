#!/usr/bin/env python3
"""Compute paper-style Transcoder K_c/K_e across the available model weights.

For each model, the 200 train pairs fit the clean-minus-corrupt direction at
the historically frozen reference layer.  The 300 held-out pairs then provide
the descriptive feature decomposition.  Two related estimands are reported:

1. paper K: sum |kappa| over every feature grouped by activation side;
2. Table-3-style top-k: aligned suppressor/driver masks, top-k per layer on
   train, then a global top-k across the available Transcoder layers, with a
   frozen held-out validation of those train-selected features.

The second quantity is deliberately kept separate from the first.  The
available Transcoder layers are recorded explicitly because they are not the
same four-layer windows for all three model families.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

import run_transcoder_feature_analysis as base
from path_defaults import CROSS_MODEL_SPECS, REBUTTAL_ROOT


OUTPUT_ROOT = REBUTTAL_ROOT / "10_cross_model_transcoder_k"
MODEL_SPECS = CROSS_MODEL_SPECS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-key", choices=tuple(MODEL_SPECS), required=True)
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--transcoder-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--reference-layer", type=int, default=None)
    parser.add_argument("--layers", type=str, default=None)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument(
        "--alignment-policy",
        choices=("strict", "allow-unequal"),
        default="strict",
        help="strict keeps only equal-length clean/corrupt pairs; allow-unequal keeps all pairs and uses their final positions",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--feature-batch-size", type=int, default=16)
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def unaligned_batches(pairs, batch_size: int, pad_token_id: int):
    """Yield common-width batches while preserving separate clean/corrupt masks.

    The paper protocol is strict token alignment.  This fallback is only for a
    diagnostic run on model-specific data that contains a deliberately
    recorded tokenizer-length mismatch.  Both sides are left padded to a
    common width, so the hook at ``[:, -1, :]`` still reads each side's final
    real token rather than a pad token.
    """
    ordered = sorted(
        pairs,
        key=lambda item: (
            max(int(item.clean_tokens.numel()), int(item.corrupt_tokens.numel())),
            item.index,
        ),
    )
    for start in range(0, len(ordered), max(batch_size, 1)):
        batch = ordered[start : start + max(batch_size, 1)]
        max_len = max(
            max(int(item.clean_tokens.numel()), int(item.corrupt_tokens.numel()))
            for item in batch
        )

        def pad(tokens: torch.Tensor) -> torch.Tensor:
            padding = max_len - int(tokens.shape[0])
            return F.pad(tokens, (padding, 0), value=pad_token_id) if padding else tokens

        clean = torch.stack([pad(item.clean_tokens) for item in batch], dim=0)
        corrupt = torch.stack([pad(item.corrupt_tokens) for item in batch], dim=0)
        clean_mask = torch.stack(
            [
                F.pad(
                    torch.ones(int(item.clean_tokens.numel()), dtype=torch.long),
                    (max_len - int(item.clean_tokens.numel()), 0),
                    value=0,
                )
                for item in batch
            ],
            dim=0,
        )
        corrupt_mask = torch.stack(
            [
                F.pad(
                    torch.ones(int(item.corrupt_tokens.numel()), dtype=torch.long),
                    (max_len - int(item.corrupt_tokens.numel()), 0),
                    value=0,
                )
                for item in batch
            ],
            dim=0,
        )
        yield batch, clean, corrupt, clean_mask, corrupt_mask


def capture_dataset_allow_unequal(
    model,
    tokenizer,
    pairs,
    *,
    capture_layers: list[int],
    decision_layer: int,
    tool_id: int,
    batch_size: int,
    label: str,
) -> dict[str, Any]:
    """Capture final-position states for the explicitly non-strict diagnostic."""
    layers = base.resolve_text_model(model).layers
    device = base.model_device(model)
    inputs = {side: {layer: [] for layer in capture_layers} for side in ("clean", "corrupt")}
    states = {side: [] for side in ("clean", "corrupt")}
    metric_parts = {side: {} for side in ("clean", "corrupt")}
    for side in metric_parts:
        metric_parts[side] = {key: [] for key in ("tool_logit", "tool_prob", "top1_margin", "log_odds", "top1", "rank")}
    metadata: list[dict[str, Any]] = []
    pad_token_id = int(tokenizer.pad_token_id)
    batches = list(unaligned_batches(pairs, batch_size, pad_token_id))
    for batch, clean_cpu, corrupt_cpu, clean_mask_cpu, corrupt_mask_cpu in tqdm(
        batches, desc=f"Capture {label}", dynamic_ncols=True
    ):
        tokens = torch.cat([clean_cpu, corrupt_cpu], dim=0).to(device)
        attention_mask = torch.cat([clean_mask_cpu, corrupt_mask_cpu], dim=0).to(device)
        holders: dict[str, Any] = {"mlp": {}, "decision": None}
        handles = base.register_capture_hooks(layers, capture_layers, decision_layer, holders)
        try:
            with torch.inference_mode():
                outputs = model(
                    input_ids=tokens,
                    attention_mask=attention_mask,
                    use_cache=False,
                    logits_to_keep=1,
                    return_dict=True,
                )
            logits = base.last_logits(outputs)
        finally:
            for handle in handles:
                handle.remove()
        n = len(batch)
        if holders["decision"] is None or set(holders["mlp"]) != set(capture_layers):
            raise RuntimeError(f"{label}: capture hook did not fire for all requested layers")
        for side, offset in (("clean", 0), ("corrupt", n)):
            side_slice = slice(offset, offset + n)
            states[side].append(holders["decision"][side_slice])
            metrics = base.metric_tensors(logits[side_slice], tool_id)
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
        base.clear_cuda()

    order = [int(row["index"]) for row in metadata]
    inverse = {index: position for position, index in enumerate(order)}

    def restore(rows: list[torch.Tensor]) -> torch.Tensor:
        joined = torch.cat(rows, dim=0)
        return joined[torch.tensor([inverse[i] for i in range(len(pairs))], dtype=torch.long)].contiguous()

    return {
        "inputs": {side: {layer: restore(inputs[side][layer]) for layer in capture_layers} for side in inputs},
        "states": {side: restore(states[side]) for side in states},
        "metrics": {side: {key: restore(parts) for key, parts in metric_parts[side].items()} for side in metric_parts},
        "metadata": sorted(metadata, key=lambda row: int(row["index"])),
    }


def parse_layers(raw: str) -> list[int]:
    layers = sorted({int(part.strip()) for part in raw.split(",") if part.strip()})
    if not layers or any(layer < 0 for layer in layers):
        raise ValueError(f"Invalid layer list: {raw!r}")
    return layers


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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


def discover_layers(root: Path) -> list[int]:
    layers: set[int] = set()
    for path in root.glob("layer*"):
        if path.is_dir():
            match = re.fullmatch(r"layer(\d+)", path.name)
            if match:
                layers.add(int(match.group(1)))
    if not layers:
        raise FileNotFoundError(f"No layer<number> directories under {root}")
    return sorted(layers)


def resolve_checkpoint(root: Path, layer: int) -> Path:
    layer_dir = root / f"layer{layer}"
    candidates = sorted(layer_dir.glob("checkpoint_step_*.pt"))
    if candidates:
        return candidates[-1]
    candidates = sorted(root.glob("checkpoint_step_*.pt"))
    for candidate in candidates:
        payload = torch.load(candidate, map_location="cpu", weights_only=False)
        actual = int(payload.get("layer", -1))
        del payload
        if actual == layer:
            return candidate
    raise FileNotFoundError(f"No checkpoint found for L{layer} under {root}")


def load_marker(dataset_root: Path) -> str:
    summary = json.loads((dataset_root / "summary.json").read_text(encoding="utf-8"))
    marker = str(summary.get("tool_call_marker", ""))
    if not marker:
        raise KeyError(f"{dataset_root}/summary.json has no tool_call_marker")
    return marker


def paper_mass(kappa: torch.Tensor, delta: torch.Tensor, side: str) -> float:
    if side == "corrupt":
        mask = delta < 0.0
    elif side == "clean":
        mask = delta > 0.0
    else:
        raise ValueError(side)
    return float(kappa[mask].abs().sum().item())


def aligned_mass(kappa: torch.Tensor, delta: torch.Tensor, beta: torch.Tensor, category: str) -> float:
    if category == "corrupt":
        mask = (delta < 0.0) & (beta < 0.0)
    elif category == "clean":
        mask = (delta > 0.0) & (beta > 0.0)
    else:
        raise ValueError(category)
    return float(kappa[mask].abs().sum().item())


def collect_stats(
    inputs: dict[str, dict[int, torch.Tensor]],
    layer: int,
    weights: dict[str, Any],
    direction: torch.Tensor,
    *,
    device: torch.device,
    batch_size: int,
) -> dict[str, torch.Tensor]:
    clean = base.collect_feature_acts(
        inputs["clean"][layer],
        weights["W_enc"],
        weights["b_enc"],
        device=device,
        batch_size=batch_size,
    )
    corrupt = base.collect_feature_acts(
        inputs["corrupt"][layer],
        weights["W_enc"],
        weights["b_enc"],
        device=device,
        batch_size=batch_size,
    )
    beta = weights["W_dec"].matmul(direction.float())
    delta = clean["mean"] - corrupt["mean"]
    return {
        "delta": delta,
        "beta": beta,
        "kappa": delta * beta,
        "active_rate_clean": clean["active_rate"],
        "active_rate_corrupt": corrupt["active_rate"],
    }


def paper_row(layer: int, split: str, stats: dict[str, torch.Tensor]) -> dict[str, Any]:
    delta, beta, kappa = stats["delta"], stats["beta"], stats["kappa"]
    corrupt_mask = delta < 0.0
    clean_mask = delta > 0.0
    suppressor_mask = corrupt_mask & (beta < 0.0)
    driver_mask = clean_mask & (beta > 0.0)
    counter_corrupt = corrupt_mask & (beta > 0.0)
    counter_clean = clean_mask & (beta < 0.0)
    k_corrupt = float(kappa[corrupt_mask].abs().sum().item())
    k_clean = float(kappa[clean_mask].abs().sum().item())
    return {
        "layer": layer,
        "split": split,
        "n_features": int(delta.numel()),
        "n_corrupt_higher": int(corrupt_mask.sum().item()),
        "n_clean_higher": int(clean_mask.sum().item()),
        "K_corrupt": k_corrupt,
        "K_clean": k_clean,
        "K_corrupt_clean_ratio": k_corrupt / k_clean if k_clean else None,
        "G_kappa": float(kappa.sum().item()),
        "G_abs_kappa": float(kappa.abs().sum().item()),
        "suppressor_aligned_mass": float(kappa[suppressor_mask].abs().sum().item()),
        "driver_aligned_mass": float(kappa[driver_mask].abs().sum().item()),
        "aligned_S_over_E": (
            float(kappa[suppressor_mask].abs().sum().item())
            / float(kappa[driver_mask].abs().sum().item())
            if driver_mask.any()
            else None
        ),
        "n_suppressor_aligned": int(suppressor_mask.sum().item()),
        "n_driver_aligned": int(driver_mask.sum().item()),
        "counter_corrupt_toward_mass": float(kappa[counter_corrupt].abs().sum().item()),
        "counter_clean_away_mass": float(kappa[counter_clean].abs().sum().item()),
    }


def train_selected_topk(
    train_stats: dict[int, dict[str, torch.Tensor]],
    heldout_stats: dict[int, dict[str, torch.Tensor]],
    layers: list[int],
    *,
    category: str,
    top_k: int,
) -> tuple[list[dict[str, Any]], int]:
    per_layer: list[dict[str, Any]] = []
    for layer in layers:
        stats = train_stats[layer]
        if category == "corrupt":
            mask = (stats["delta"] < 0.0) & (stats["beta"] < 0.0)
        elif category == "clean":
            mask = (stats["delta"] > 0.0) & (stats["beta"] > 0.0)
        else:
            raise ValueError(category)
        ids = torch.nonzero(mask, as_tuple=False).flatten()
        order = torch.argsort(stats["kappa"][ids].abs(), descending=True)[:top_k]
        for feature_idx in ids[order].tolist():
            held = heldout_stats[layer]
            per_layer.append(
                {
                    "layer": int(layer),
                    "feature_idx": int(feature_idx),
                    "category": category,
                    "train_kappa": float(stats["kappa"][feature_idx].item()),
                    "train_abs_kappa": float(stats["kappa"][feature_idx].abs().item()),
                    "heldout_kappa": float(held["kappa"][feature_idx].item()),
                    "heldout_abs_kappa": float(held["kappa"][feature_idx].abs().item()),
                    "train_delta": float(stats["delta"][feature_idx].item()),
                    "heldout_delta": float(held["delta"][feature_idx].item()),
                    "beta": float(stats["beta"][feature_idx].item()),
                    "train_active_rate_clean": float(stats["active_rate_clean"][feature_idx].item()),
                    "train_active_rate_corrupt": float(stats["active_rate_corrupt"][feature_idx].item()),
                    "heldout_active_rate_clean": float(held["active_rate_clean"][feature_idx].item()),
                    "heldout_active_rate_corrupt": float(held["active_rate_corrupt"][feature_idx].item()),
                }
            )
    per_layer.sort(key=lambda row: (-row["train_abs_kappa"], row["layer"], row["feature_idx"]))
    return per_layer[:top_k], len(per_layer)


def topk_summary(
    train_stats: dict[int, dict[str, torch.Tensor]],
    heldout_stats: dict[int, dict[str, torch.Tensor]],
    layers: list[int],
    *,
    top_k: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    suppressors, suppressor_pool = train_selected_topk(
        train_stats, heldout_stats, layers, category="corrupt", top_k=top_k
    )
    drivers, driver_pool = train_selected_topk(
        train_stats, heldout_stats, layers, category="clean", top_k=top_k
    )
    complete = len(suppressors) == top_k and len(drivers) == top_k
    payload: dict[str, Any] = {
        "top_k": top_k,
        "selection": "train aligned mask; top-k abs(kappa) per layer; global top-k per category",
        "status": "complete" if complete else "insufficient_candidates",
        "suppressor_pool_count": suppressor_pool,
        "driver_pool_count": driver_pool,
        "suppressor_selected_count": len(suppressors),
        "driver_selected_count": len(drivers),
    }
    selected_rows = suppressors + drivers
    if not complete:
        return payload, selected_rows
    for prefix, rows in (("corrupt", suppressors), ("clean", drivers)):
        payload[f"{prefix}_train_abs_kappa"] = float(sum(row["train_abs_kappa"] for row in rows))
        payload[f"{prefix}_heldout_abs_kappa"] = float(sum(row["heldout_abs_kappa"] for row in rows))
        payload[f"{prefix}_heldout_recomputed_abs_kappa"] = None
    payload["train_ratio_corrupt_over_clean"] = (
        payload["corrupt_train_abs_kappa"] / payload["clean_train_abs_kappa"]
        if payload["clean_train_abs_kappa"]
        else None
    )
    payload["heldout_frozen_ratio_corrupt_over_clean"] = (
        payload["corrupt_heldout_abs_kappa"] / payload["clean_heldout_abs_kappa"]
        if payload["clean_heldout_abs_kappa"]
        else None
    )
    payload["heldout_frozen_delta_corrupt_minus_clean"] = (
        payload["corrupt_heldout_abs_kappa"] - payload["clean_heldout_abs_kappa"]
    )

    # A secondary heldout re-ranking is diagnostic only; it is not the
    # train-selected Table-3 estimate.
    for category, prefix in (("corrupt", "corrupt"), ("clean", "clean")):
        candidates: list[dict[str, Any]] = []
        for layer in layers:
            stats = heldout_stats[layer]
            if category == "corrupt":
                mask = (stats["delta"] < 0.0) & (stats["beta"] < 0.0)
            else:
                mask = (stats["delta"] > 0.0) & (stats["beta"] > 0.0)
            ids = torch.nonzero(mask, as_tuple=False).flatten()
            order = torch.argsort(stats["kappa"][ids].abs(), descending=True)[:top_k]
            candidates.extend(
                {"layer": layer, "feature_idx": int(feature_idx), "abs_kappa": float(stats["kappa"][feature_idx].abs().item())}
                for feature_idx in ids[order].tolist()
            )
        candidates.sort(key=lambda row: (-row["abs_kappa"], row["layer"], row["feature_idx"]))
        payload[f"{prefix}_heldout_recomputed_abs_kappa"] = float(
            sum(row["abs_kappa"] for row in candidates[:top_k])
        )
    payload["heldout_recomputed_ratio_corrupt_over_clean"] = (
        payload["corrupt_heldout_recomputed_abs_kappa"] / payload["clean_heldout_recomputed_abs_kappa"]
        if payload["clean_heldout_recomputed_abs_kappa"]
        else None
    )
    return payload, selected_rows


def main() -> None:
    args = parse_args()
    spec = MODEL_SPECS[args.model_key]
    model_path = (args.model_path or spec["model_path"]).resolve()
    dataset_root = (args.dataset_root or spec["dataset_root"]).resolve()
    transcoder_root = (args.transcoder_root or spec["transcoder_root"]).resolve()
    output_root = (args.output_root or OUTPUT_ROOT / args.model_key).resolve()
    reference_layer = int(args.reference_layer if args.reference_layer is not None else spec["reference_layer"])
    layers = parse_layers(args.layers) if args.layers else discover_layers(transcoder_root)
    if args.top_k <= 0:
        raise ValueError("--top-k must be positive")
    if reference_layer in layers:
        reference_note = "reference layer is included in the available Transcoder coverage"
    else:
        reference_note = "reference layer is outside the available Transcoder coverage"

    marker = load_marker(dataset_root)
    model, tokenizer = base.load_model(model_path, args.dtype, args.device)
    token_ids = tokenizer.encode(marker, add_special_tokens=False)
    if len(token_ids) != 1:
        raise ValueError(f"{args.model_key}: marker {marker!r} is not one token: {token_ids}")
    tool_id = int(token_ids[0])
    text_model = base.resolve_text_model(model)
    n_layers = len(text_model.layers)
    if reference_layer < 0 or reference_layer >= n_layers:
        raise ValueError(f"Reference L{reference_layer} invalid for {n_layers} model layers")
    if any(layer < 0 or layer >= n_layers for layer in layers):
        raise ValueError(f"Transcoder layers {layers} exceed model depth {n_layers}")
    if any(not (transcoder_root / f"layer{layer}").exists() for layer in layers):
        raise FileNotFoundError(f"Requested layers are not represented by layer directories under {transcoder_root}")

    # Always audit the full model-specific split first.  The strict policy then
    # drops only pairs that violate the paper's token-position alignment
    # assumption; the diagnostic policy keeps them and captures each side's
    # final real token.
    all_train_pairs = base.load_pairs(
        dataset_root, tokenizer, "train", token_length_policy="allow"
    )
    all_heldout_pairs = base.load_pairs(
        dataset_root, tokenizer, "heldout", token_length_policy="allow"
    )
    train_mismatches = sum(
        int(item.clean_tokens.numel() != item.corrupt_tokens.numel()) for item in all_train_pairs
    )
    heldout_mismatches = sum(
        int(item.clean_tokens.numel() != item.corrupt_tokens.numel()) for item in all_heldout_pairs
    )
    if args.alignment_policy == "strict":
        train_pairs = base.load_pairs(
            dataset_root, tokenizer, "train", token_length_policy="skip"
        )
        heldout_pairs = base.load_pairs(
            dataset_root, tokenizer, "heldout", token_length_policy="skip"
        )
    else:
        train_pairs = all_train_pairs
        heldout_pairs = all_heldout_pairs
    if not train_pairs or not heldout_pairs:
        raise ValueError(f"No usable pairs after alignment policy {args.alignment_policy!r}")

    capture_layers = sorted(set(layers))
    capture_fn = (
        capture_dataset_allow_unequal
        if args.alignment_policy == "allow-unequal"
        else base.capture_dataset
    )
    train = capture_fn(
        model,
        tokenizer,
        train_pairs,
        capture_layers=capture_layers,
        decision_layer=reference_layer,
        tool_id=tool_id,
        batch_size=args.batch_size,
        label=f"{args.model_key} train-{len(train_pairs)}",
    )
    heldout = capture_fn(
        model,
        tokenizer,
        heldout_pairs,
        capture_layers=capture_layers,
        decision_layer=reference_layer,
        tool_id=tool_id,
        batch_size=args.batch_size,
        label=f"{args.model_key} heldout-{len(heldout_pairs)}",
    )
    direction_raw = train["states"]["clean"].mean(dim=0) - train["states"]["corrupt"].mean(dim=0)
    direction_norm = float(direction_raw.norm().item())
    if direction_norm <= 0:
        raise RuntimeError("Train clean-minus-corrupt direction has zero norm")
    direction = direction_raw / direction_norm
    device = torch.device(args.device)

    checkpoint_map = {str(layer): str(resolve_checkpoint(transcoder_root, layer)) for layer in layers}
    train_stats: dict[int, dict[str, torch.Tensor]] = {}
    heldout_stats: dict[int, dict[str, torch.Tensor]] = {}
    paper_rows: list[dict[str, Any]] = []
    for layer in tqdm(layers, desc=f"Scoring {args.model_key} Transcoders", dynamic_ncols=True):
        checkpoint = Path(checkpoint_map[str(layer)])
        weights = base.load_transcoder(checkpoint, layer)
        train_stats[layer] = collect_stats(
            train["inputs"], layer, weights, direction, device=device, batch_size=args.feature_batch_size
        )
        heldout_stats[layer] = collect_stats(
            heldout["inputs"], layer, weights, direction, device=device, batch_size=args.feature_batch_size
        )
        paper_rows.append(paper_row(layer, "train", train_stats[layer]))
        paper_rows.append(paper_row(layer, "heldout", heldout_stats[layer]))
        del weights
        base.clear_cuda()

    train_paper = {
        key: float(sum(row[key] for row in paper_rows if row["split"] == "train"))
        for key in ("K_corrupt", "K_clean", "G_kappa", "G_abs_kappa", "suppressor_aligned_mass", "driver_aligned_mass")
    }
    heldout_paper = {
        key: float(sum(row[key] for row in paper_rows if row["split"] == "heldout"))
        for key in ("K_corrupt", "K_clean", "G_kappa", "G_abs_kappa", "suppressor_aligned_mass", "driver_aligned_mass")
    }
    for aggregate in (train_paper, heldout_paper):
        aggregate["K_corrupt_clean_ratio"] = aggregate["K_corrupt"] / aggregate["K_clean"] if aggregate["K_clean"] else None
        aggregate["aligned_S_over_E"] = (
            aggregate["suppressor_aligned_mass"] / aggregate["driver_aligned_mass"]
            if aggregate["driver_aligned_mass"]
            else None
        )
    topk, selected_rows = topk_summary(train_stats, heldout_stats, layers, top_k=args.top_k)

    dataset_summary = json.loads((dataset_root / "summary.json").read_text(encoding="utf-8"))
    run_config = {
        "experiment": "cross_model_paper_transcoder_K",
        "model_key": args.model_key,
        "model_label": spec["label"],
        "model_path": str(model_path),
        "dataset_root": str(dataset_root),
        "dataset_version": dataset_summary.get("dataset_version"),
        "transcoder_root": str(transcoder_root),
        "transcoder_checkpoints": checkpoint_map,
        "model_layer_count": n_layers,
        "reference_layer": reference_layer,
        "available_transcoder_layers": layers,
        "available_layer_relation": reference_note,
        "alignment_policy": args.alignment_policy,
        "full_train_pairs_before_alignment": len(all_train_pairs),
        "full_heldout_pairs_before_alignment": len(all_heldout_pairs),
        "train_token_length_mismatch_count": train_mismatches,
        "heldout_token_length_mismatch_count": heldout_mismatches,
        "train_pairs": len(train_pairs),
        "heldout_pairs": len(heldout_pairs),
        "tool_call_marker": marker,
        "tool_call_token_id": tool_id,
        "direction": f"unit mean(clean - corrupt) reference-layer output over train-{len(train_pairs)}",
        "paper_K_definition": "sum(abs(kappa) by activation side over all features; no top-k or C filter)",
        "table3_topk_definition": "aligned mask; top-k abs(train kappa) per layer; global top-k per category; heldout frozen validation",
        "top_k": args.top_k,
        "dtype": args.dtype,
        "batch_size": args.batch_size,
        "feature_batch_size": args.feature_batch_size,
        "direction_norm_before_unit": direction_norm,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    write_json(output_root / "run_config.json", run_config)
    write_json(
        output_root / "summary.json",
        {
            **run_config,
            "train_behavior": {
                "clean": base.metric_summary(train["metrics"]["clean"], tool_id),
                "corrupt": base.metric_summary(train["metrics"]["corrupt"], tool_id),
            },
            "heldout_behavior": {
                "clean": base.metric_summary(heldout["metrics"]["clean"], tool_id),
                "corrupt": base.metric_summary(heldout["metrics"]["corrupt"], tool_id),
            },
            "paper_K_train": train_paper,
            "paper_K_heldout": heldout_paper,
            "table3_topk": topk,
            "paper_rows": paper_rows,
        },
    )
    write_csv(output_root / "paper_K_by_layer.csv", paper_rows)
    write_csv(output_root / "table3_topk_selected_features.csv", selected_rows)
    torch.save(
        {
            "train_inputs": train["inputs"],
            "train_states": train["states"],
            "heldout_inputs": heldout["inputs"],
            "heldout_states": heldout["states"],
            "reference_layer": reference_layer,
            "layers": layers,
        },
        output_root / "activation_cache.pt",
    )

    def ratio_text(payload: dict[str, Any], key: str) -> str:
        ratio = payload.get(key)
        return "—" if ratio is None else f"{float(ratio):.3f}"

    md = [
        f"# {spec['label']}: Transcoder K_c/K_e",
        "",
        f"Train-{len(train_pairs)} fits the direction and selects Table-3-style top-{args.top_k} features; heldout-{len(heldout_pairs)} validates the frozen selection.",
        f"Available Transcoder layers: `{layers}`. Historical reference layer: `L{reference_layer}`; {reference_note}.",
        "",
        "## Paper-style all-feature K",
        "",
        "| split | K_corrupt | K_clean | Kc/Kclean | aligned S | aligned E | S/E |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label, payload in (("train", train_paper), ("heldout", heldout_paper)):
        md.append(
            f"| {label} | {payload['K_corrupt']:.5f} | {payload['K_clean']:.5f} | "
            f"{ratio_text(payload, 'K_corrupt_clean_ratio')} | {payload['suppressor_aligned_mass']:.5f} | "
            f"{payload['driver_aligned_mass']:.5f} | {ratio_text(payload, 'aligned_S_over_E')} |"
        )
    md.extend(
        [
            "",
            "## Table-3-style aligned global top-k",
            "",
            "| k | status | train Kc/Ke | heldout frozen Kc/Ke | heldout ratio | heldout Kc−Ke | heldout re-ranked ratio |",
            "|---:|---|---:|---:|---:|---:|---:|",
            f"| {args.top_k} | {topk['status']} | "
            + (
                f"{topk['corrupt_train_abs_kappa']:.5f}/{topk['clean_train_abs_kappa']:.5f} | "
                f"{topk['corrupt_heldout_abs_kappa']:.5f}/{topk['clean_heldout_abs_kappa']:.5f} | "
                f"{ratio_text(topk, 'heldout_frozen_ratio_corrupt_over_clean')} | "
                f"{topk['heldout_frozen_delta_corrupt_minus_clean']:+.5f} | "
                f"{ratio_text(topk, 'heldout_recomputed_ratio_corrupt_over_clean')} |"
                if topk["status"] == "complete"
                else "— | — | — | — | — |"
            ),
            "",
            "The all-feature K and aligned top-k estimates are different estimands; neither has a consistency/frequency filter in this primary run.",
        ]
    )
    (output_root / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps({"status": "complete", "model_key": args.model_key, "output_root": str(output_root)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
