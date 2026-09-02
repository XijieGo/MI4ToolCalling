#!/usr/bin/env python3
"""Bidirectional S/E causal swaps on model-specific Transcoder windows.

This runner implements the causal table proposed in the rebuttal draft:

* fit the clean-minus-corrupt direction on the 200-pair train split;
* decompose each formation-window MLP output with its Transcoder;
* select equal-budget aligned suppressor (S) and execution/driver (E)
  features by train |kappa| ranking, frozen before held-out evaluation;
* on held-out clean prompts, replace selected activations with their paired
  corrupt values (S->clean / E->clean);
* on held-out corrupt prompts, replace selected activations with their paired
  clean values (S->corrupt / E->corrupt).

The Transcoder contribution is added to the actual HF MLP output.  The whole
MLP is never replaced by its reconstruction.  Thus each intervention is a
local feature-level activation swap with the exact decoder write used by the
Transcoder decomposition.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

import run_cross_model_transcoder_k as cross
import run_transcoder_feature_analysis as base
from path_defaults import REBUTTAL_ROOT, SE_DOMINANCE_SPECS


DEFAULT_OUTPUT_ROOT = REBUTTAL_ROOT / "14_transcoder_se_dominance_v2"
# Mistral is intentionally absent: the release has no matching Transcoder.
MODEL_SPECS = SE_DOMINANCE_SPECS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-keys",
        type=str,
        default=",".join(MODEL_SPECS),
        help="Comma-separated model keys; Mistral is intentionally absent because no Transcoder is available.",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--budget", type=int, default=20, help="Equal total number of S and E features.")
    parser.add_argument(
        "--per-layer-top-k",
        type=int,
        default=20,
        help="Train-side aligned candidates retained per layer before global budget ranking.",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--feature-batch-size", type=int, default=16)
    parser.add_argument("--alignment-policy", choices=("allow-unequal", "strict"), default="allow-unequal")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=20260803)
    return parser.parse_args()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
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


def resolve_checkpoint(root: Path, layer: int) -> Path:
    # Qwen3 dictionaries are the public safetensors files layer_<n>.safetensors.
    for candidate in (root / f"layer_{layer}.safetensors", root / f"layer{layer}.safetensors"):
        if candidate.exists():
            return candidate
    # Qwen3.5/Granite dictionaries are training checkpoints under layer<n>/.
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
    raise FileNotFoundError(f"No Transcoder checkpoint for L{layer} under {root}")


def load_transcoder_any(path: Path, expected_layer: int) -> dict[str, Any]:
    if path.suffix == ".safetensors":
        from safetensors import safe_open

        with safe_open(str(path), framework="pt", device="cpu") as handle:
            keys = set(handle.keys())
            required = {"W_enc", "b_enc", "W_dec", "b_dec"}
            missing = required - keys
            if missing:
                raise KeyError(f"{path}: missing Transcoder tensors {sorted(missing)}")
            weights = {key: handle.get_tensor(key).contiguous() for key in required}
        weights["W_dec"] = weights["W_dec"].float().contiguous()
        weights["b_dec"] = weights["b_dec"].float().contiguous()
        weights["checkpoint_step"] = -1
        weights["layer"] = expected_layer
        weights["d_model"] = int(weights["W_dec"].shape[-1])
        weights["d_feature"] = int(weights["W_dec"].shape[0])
        return weights
    weights = base.load_transcoder(path, expected_layer)
    weights["layer"] = expected_layer
    return weights


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
        inputs["clean"][layer], weights["W_enc"], weights["b_enc"], device=device, batch_size=batch_size
    )
    corrupt = base.collect_feature_acts(
        inputs["corrupt"][layer], weights["W_enc"], weights["b_enc"], device=device, batch_size=batch_size
    )
    beta = weights["W_dec"].matmul(direction.float())
    delta = clean["mean"] - corrupt["mean"]
    return {
        "delta": delta,
        "beta": beta,
        "kappa": delta * beta,
        "clean_active_rate": clean["active_rate"],
        "corrupt_active_rate": corrupt["active_rate"],
    }


def aligned_mask(stats: dict[str, torch.Tensor], category: str) -> torch.Tensor:
    if category == "suppressor":
        return (stats["delta"] < 0.0) & (stats["beta"] < 0.0)
    if category == "driver":
        return (stats["delta"] > 0.0) & (stats["beta"] > 0.0)
    raise ValueError(category)


def select_global_group(
    train_stats: dict[int, dict[str, torch.Tensor]],
    heldout_stats: dict[int, dict[str, torch.Tensor]],
    layers: Sequence[int],
    *,
    category: str,
    per_layer_top_k: int,
    budget: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for layer in layers:
        stats = train_stats[layer]
        held = heldout_stats[layer]
        ids = torch.nonzero(aligned_mask(stats, category), as_tuple=False).flatten()
        order = torch.argsort(stats["kappa"][ids].abs(), descending=True)[:per_layer_top_k]
        for feature_idx in ids[order].tolist():
            rows.append(
                {
                    "layer": int(layer),
                    "feature_idx": int(feature_idx),
                    "category": category,
                    "train_delta": float(stats["delta"][feature_idx].item()),
                    "heldout_delta": float(held["delta"][feature_idx].item()),
                    "beta_mu": float(stats["beta"][feature_idx].item()),
                    "train_kappa": float(stats["kappa"][feature_idx].item()),
                    "heldout_kappa": float(held["kappa"][feature_idx].item()),
                    "train_abs_kappa": float(stats["kappa"][feature_idx].abs().item()),
                    "heldout_abs_kappa": float(held["kappa"][feature_idx].abs().item()),
                    "train_active_rate_clean": float(stats["clean_active_rate"][feature_idx].item()),
                    "train_active_rate_corrupt": float(stats["corrupt_active_rate"][feature_idx].item()),
                    "heldout_active_rate_clean": float(held["clean_active_rate"][feature_idx].item()),
                    "heldout_active_rate_corrupt": float(held["corrupt_active_rate"][feature_idx].item()),
                }
            )
    rows.sort(key=lambda row: (-row["train_abs_kappa"], row["layer"], row["feature_idx"]))
    return rows[:budget]


def selected_activation_cache(
    inputs: dict[str, dict[int, torch.Tensor]],
    groups: Sequence[dict[str, Any]],
    checkpoint_map: dict[str, str],
    *,
    device: torch.device,
    batch_size: int,
) -> tuple[dict[int, dict[str, torch.Tensor]], dict[int, torch.Tensor]]:
    ids_by_layer: dict[int, list[int]] = {}
    for row in groups:
        ids_by_layer.setdefault(int(row["layer"]), []).append(int(row["feature_idx"]))
    acts: dict[int, dict[str, torch.Tensor]] = {}
    decoders: dict[int, torch.Tensor] = {}
    for layer, ids_list in ids_by_layer.items():
        ids = torch.tensor(sorted(set(ids_list)), dtype=torch.long)
        if ids.numel() == 0:
            continue
        weights = load_transcoder_any(Path(checkpoint_map[str(layer)]), layer)
        if int(weights["d_model"]) != int(inputs["clean"][layer].shape[-1]):
            raise ValueError(f"L{layer}: Transcoder/model dimension mismatch while building selected cache")
        clean = base.collect_feature_acts(
            inputs["clean"][layer], weights["W_enc"], weights["b_enc"],
            device=device, batch_size=batch_size, selected_ids=ids,
        )["selected"]
        corrupt = base.collect_feature_acts(
            inputs["corrupt"][layer], weights["W_enc"], weights["b_enc"],
            device=device, batch_size=batch_size, selected_ids=ids,
        )["selected"]
        acts[layer] = {"ids": ids, "clean": clean, "corrupt": corrupt}
        decoders[layer] = weights["W_dec"].index_select(0, ids).float().contiguous()
        del weights
        base.clear_cuda()
    return acts, decoders


def make_delta_cache(
    group: Sequence[dict[str, Any]],
    selected_acts: dict[int, dict[str, torch.Tensor]],
    decoders: dict[int, torch.Tensor],
) -> dict[int, torch.Tensor]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in group:
        grouped.setdefault(int(row["layer"]), []).append(row)
    result: dict[int, torch.Tensor] = {}
    for layer, rows in grouped.items():
        ids = [int(row["feature_idx"]) for row in rows]
        available = [int(x) for x in selected_acts[layer]["ids"].tolist()]
        positions = torch.tensor([available.index(feature_id) for feature_id in ids], dtype=torch.long)
        clean = selected_acts[layer]["clean"].index_select(1, positions).float()
        corrupt = selected_acts[layer]["corrupt"].index_select(1, positions).float()
        decoder_positions = torch.tensor([available.index(feature_id) for feature_id in ids], dtype=torch.long)
        decoder = decoders[layer].index_select(0, decoder_positions).float()
        # Positive cache is clean-minus-corrupt.  The clean-side intervention
        # uses its negative; the corrupt-side intervention uses its positive.
        result[layer] = (clean - corrupt).matmul(decoder)
    return result


def baseline_row(metrics: dict[str, torch.Tensor], index: int, tool_id: int) -> dict[str, float | int]:
    return {
        "tool_logit": float(metrics["tool_logit"][index].item()),
        "tool_probability": float(metrics["tool_prob"][index].item()),
        "margin": float(metrics["top1_margin"][index].item()),
        "log_odds": float(metrics["log_odds"][index].item()),
        "rank": int(metrics["rank"][index].item()),
        "is_tool_top1": int(metrics["top1"][index].item() == tool_id),
    }


def run_swap(
    model,
    tokenizer,
    pairs,
    *,
    group_name: str,
    group: Sequence[dict[str, Any]],
    side: str,
    selected_acts: dict[int, dict[str, torch.Tensor]],
    decoders: dict[int, torch.Tensor],
    baseline_metrics: dict[str, torch.Tensor],
    baseline_states: dict[str, torch.Tensor],
    direction: torch.Tensor,
    decision_layer: int,
    tool_id: int,
    batch_size: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if side not in {"clean", "corrupt"}:
        raise ValueError(side)
    model_layers = base.resolve_text_model(model).layers
    device = base.model_device(model)
    delta_cache = make_delta_cache(group, selected_acts, decoders)
    pad_token_id = int(tokenizer.pad_token_id)
    all_rows: list[dict[str, Any]] = []
    transfer = "corrupt_to_clean" if side == "clean" else "clean_to_corrupt"
    for batch, clean_cpu, corrupt_cpu, clean_mask_cpu, corrupt_mask_cpu in tqdm(
        cross.unaligned_batches(pairs, batch_size, pad_token_id),
        desc=f"Swap {group_name}/{transfer}",
        dynamic_ncols=True,
        leave=False,
    ):
        indices = [int(item.index) for item in batch]
        index_tensor = torch.tensor(indices, dtype=torch.long)
        sign = -1.0 if side == "clean" else 1.0
        deltas = {layer: sign * values.index_select(0, index_tensor) for layer, values in delta_cache.items()}
        tokens = (clean_cpu if side == "clean" else corrupt_cpu).to(device)
        attention_mask = (clean_mask_cpu if side == "clean" else corrupt_mask_cpu).to(device)
        gate_holder: dict[str, torch.Tensor] = {}
        handles = []
        for layer, delta_cpu in deltas.items():
            def make_hook(delta: torch.Tensor):
                def hook(_module, _inputs, output):
                    value = base.tensor_output(output).clone()
                    value[:, -1, :] = value[:, -1, :] + delta.to(device=value.device, dtype=value.dtype)
                    return base.replace_tensor_output(output, value)

                return hook

            handles.append(model_layers[layer].mlp.register_forward_hook(make_hook(delta_cpu)))

        def decision_hook(_module, _inputs, output):
            gate_holder["state"] = base.tensor_output(output)[:, -1, :].detach().float().cpu()

        handles.append(model_layers[decision_layer].register_forward_hook(decision_hook))
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
        if "state" not in gate_holder:
            raise RuntimeError(f"{group_name}/{transfer}: decision hook did not fire")
        metrics = base.metric_tensors(logits, tool_id)
        gate = gate_holder["state"].matmul(direction.float())
        base_gate = baseline_states[side].index_select(0, index_tensor).matmul(direction.float())
        for local, item in enumerate(batch):
            old = baseline_row(baseline_metrics[side], item.index, tool_id)
            new_top1 = int(metrics["top1"][local].item() == tool_id)
            new_margin = float(metrics["top1_margin"][local].item())
            target_side = "corrupt" if side == "clean" else "clean"
            source_tool = old["tool_logit"]
            target_tool = float(baseline_metrics[target_side]["tool_logit"][item.index].item())
            source_margin = old["margin"]
            target_margin = float(baseline_metrics[target_side]["top1_margin"][item.index].item())
            intervention_tool = float(metrics["tool_logit"][local].item())
            midpoint = 0.5 * (source_tool + target_tool)
            all_rows.append(
                {
                    "group": group_name,
                    "side": side,
                    "transfer": transfer,
                    "sample_id": item.sample_id,
                    "pair_index": int(item.index),
                    "intervened_tool_logit": float(metrics["tool_logit"][local].item()),
                    "intervened_margin": new_margin,
                    "intervened_tool_probability": float(metrics["tool_prob"][local].item()),
                    "intervened_rank": int(metrics["rank"][local].item()),
                    "intervened_is_tool_top1": new_top1,
                    "tool_logit_delta": float(metrics["tool_logit"][local].item()) - old["tool_logit"],
                    "margin_delta": new_margin - old["margin"],
                    "log_odds_delta": float(metrics["log_odds"][local].item()) - old["log_odds"],
                    "gate_delta": float(gate[local].item() - base_gate[local].item()),
                    "baseline_tool_logit": old["tool_logit"],
                    "baseline_margin": old["margin"],
                    "baseline_rank": old["rank"],
                    "baseline_is_tool_top1": old["is_tool_top1"],
                    "paired_target_tool_logit": target_tool,
                    "paired_target_margin": target_margin,
                    "paired_tool_logit_midpoint": midpoint,
                    "tool_logit_gap_progress": (
                        (intervention_tool - source_tool) / (target_tool - source_tool)
                        if abs(target_tool - source_tool) > 1e-6
                        else None
                    ),
                    "raw_tool_logit_midpoint_flip": int(
                        intervention_tool <= midpoint if side == "clean" else intervention_tool >= midpoint
                    ),
                    "raw_tool_logit_reached_paired_target": int(
                        intervention_tool <= target_tool if side == "clean" else intervention_tool >= target_tool
                    ),
                    "margin_boundary_flip": int(
                        source_margin >= 0.0 > new_margin
                        if side == "clean"
                        else source_margin <= 0.0 < new_margin
                    ),
                    "top1_flip_same_direction": int(
                        (side == "corrupt" and not old["is_tool_top1"] and new_top1)
                        or (side == "clean" and old["is_tool_top1"] and not new_top1)
                    ),
                }
            )
        del tokens, attention_mask, outputs, logits, metrics, gate_holder
        base.clear_cuda()

    n = max(len(all_rows), 1)
    summary = {
        "group": group_name,
        "side": side,
        "transfer": transfer,
        "n": len(all_rows),
        "feature_count": len(group),
        "layers": dict(Counter(int(row["layer"]) for row in group)),
        "mean_delta_m": sum(float(row["margin_delta"]) for row in all_rows) / n,
        "mean_tool_logit_delta": sum(float(row["tool_logit_delta"]) for row in all_rows) / n,
        "mean_gate_delta": sum(float(row["gate_delta"]) for row in all_rows) / n,
        "top1_flip_rate": sum(int(row["top1_flip_same_direction"]) for row in all_rows) / n,
        "margin_boundary_flip_rate": sum(int(row["margin_boundary_flip"]) for row in all_rows) / n,
        "raw_tool_logit_midpoint_flip_rate": sum(
            int(row["raw_tool_logit_midpoint_flip"]) for row in all_rows
        ) / n,
        "raw_tool_logit_reached_paired_target_rate": sum(
            int(row["raw_tool_logit_reached_paired_target"]) for row in all_rows
        ) / n,
        "intervened_tool_top1_rate": sum(int(row["intervened_is_tool_top1"]) for row in all_rows) / n,
    }
    return summary, all_rows


def summarize_s_over_e(summaries: Sequence[dict[str, Any]]) -> dict[str, Any]:
    lookup = {(row["group"], row["side"]): row for row in summaries}
    required = [("S_top20", "clean"), ("S_top20", "corrupt"), ("E_top20", "clean"), ("E_top20", "corrupt")]
    if any(key not in lookup for key in required):
        raise KeyError(f"Missing S/E summary cells: {required}")
    s_clean = float(lookup[("S_top20", "clean")]["mean_delta_m"])
    s_corrupt = float(lookup[("S_top20", "corrupt")]["mean_delta_m"])
    e_clean = float(lookup[("E_top20", "clean")]["mean_delta_m"])
    e_corrupt = float(lookup[("E_top20", "corrupt")]["mean_delta_m"])
    denominator = abs(e_clean) + abs(e_corrupt)
    return {
        "S_to_clean_delta_m": s_clean,
        "E_to_clean_delta_m": e_clean,
        "S_to_corrupt_delta_m": s_corrupt,
        "E_to_corrupt_delta_m": e_corrupt,
        "S_over_E": (abs(s_clean) + abs(s_corrupt)) / denominator if denominator else None,
    }


def run_model(args: argparse.Namespace, model_key: str) -> dict[str, Any]:
    spec = MODEL_SPECS[model_key]
    output_root = (args.output_root / model_key).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    model_path = Path(spec["model_path"])
    dataset_root = Path(spec["dataset_root"])
    transcoder_root = Path(spec["transcoder_root"])
    layers = list(spec["layers"])
    decision_layer = int(spec["decision_layer"])

    model, tokenizer = base.load_model(model_path, args.dtype, args.device)
    marker = cross.load_marker(dataset_root)
    token_ids = tokenizer.encode(marker, add_special_tokens=False)
    if len(token_ids) != 1:
        raise ValueError(f"{model_key}: marker {marker!r} is not a single token: {token_ids}")
    tool_id = int(token_ids[0])
    model_layers = base.resolve_text_model(model).layers
    if decision_layer < 0 or decision_layer >= len(model_layers):
        raise ValueError(f"{model_key}: invalid decision layer L{decision_layer} for {len(model_layers)} layers")
    if any(layer < 0 or layer >= len(model_layers) for layer in layers):
        raise ValueError(f"{model_key}: invalid Transcoder layers {layers}")

    # The base loader uses "allow" / "skip"; normalize the CLI spelling.
    if args.alignment_policy == "allow-unequal":
        train_pairs = base.load_pairs(dataset_root, tokenizer, "train", token_length_policy="allow")
        heldout_pairs = base.load_pairs(dataset_root, tokenizer, "heldout", token_length_policy="allow")
        capture_fn = cross.capture_dataset_allow_unequal
    else:
        train_pairs = base.load_pairs(dataset_root, tokenizer, "train", token_length_policy="skip")
        heldout_pairs = base.load_pairs(dataset_root, tokenizer, "heldout", token_length_policy="skip")
        capture_fn = base.capture_dataset
    if len(train_pairs) != 200 or len(heldout_pairs) != 300:
        raise ValueError(f"{model_key}: expected 200/300 pairs under allow-unequal protocol, got {len(train_pairs)}/{len(heldout_pairs)}")

    train = capture_fn(
        model, tokenizer, train_pairs, capture_layers=layers, decision_layer=decision_layer,
        tool_id=tool_id, batch_size=args.batch_size, label=f"{model_key} train-{len(train_pairs)}",
    )
    heldout = capture_fn(
        model, tokenizer, heldout_pairs, capture_layers=layers, decision_layer=decision_layer,
        tool_id=tool_id, batch_size=args.batch_size, label=f"{model_key} heldout-{len(heldout_pairs)}",
    )
    direction_raw = train["states"]["clean"].mean(dim=0) - train["states"]["corrupt"].mean(dim=0)
    direction_norm = float(direction_raw.norm().item())
    if direction_norm <= 0:
        raise RuntimeError(f"{model_key}: zero train clean-minus-corrupt direction")
    direction = direction_raw / direction_norm

    checkpoint_map = {str(layer): str(resolve_checkpoint(transcoder_root, layer)) for layer in layers}
    train_stats: dict[int, dict[str, torch.Tensor]] = {}
    heldout_stats: dict[int, dict[str, torch.Tensor]] = {}
    for layer in tqdm(layers, desc=f"Score {model_key} Transcoders", dynamic_ncols=True):
        weights = load_transcoder_any(Path(checkpoint_map[str(layer)]), layer)
        if int(weights["d_model"]) != int(train["inputs"]["clean"][layer].shape[-1]):
            raise ValueError(f"{model_key} L{layer}: Transcoder/model dimension mismatch")
        train_stats[layer] = collect_stats(
            train["inputs"], layer, weights, direction, device=base.model_device(model), batch_size=args.feature_batch_size,
        )
        heldout_stats[layer] = collect_stats(
            heldout["inputs"], layer, weights, direction, device=base.model_device(model), batch_size=args.feature_batch_size,
        )
        del weights
        base.clear_cuda()

    suppressors = select_global_group(
        train_stats, heldout_stats, layers, category="suppressor",
        per_layer_top_k=args.per_layer_top_k, budget=args.budget,
    )
    drivers = select_global_group(
        train_stats, heldout_stats, layers, category="driver",
        per_layer_top_k=args.per_layer_top_k, budget=args.budget,
    )
    if len(suppressors) != args.budget or len(drivers) != args.budget:
        raise RuntimeError(f"{model_key}: insufficient S/E candidates: {len(suppressors)}/{len(drivers)}")

    all_selected = suppressors + drivers
    acts, decoders = selected_activation_cache(
        heldout["inputs"], all_selected, checkpoint_map,
        device=base.model_device(model), batch_size=args.feature_batch_size,
    )
    summaries: list[dict[str, Any]] = []
    per_sample: list[dict[str, Any]] = []
    for group_name, group in (("S_top20", suppressors), ("E_top20", drivers)):
        for side in ("clean", "corrupt"):
            summary, rows = run_swap(
                model, tokenizer, heldout_pairs, group_name=group_name, group=group, side=side,
                selected_acts=acts, decoders=decoders, baseline_metrics=heldout["metrics"],
                baseline_states=heldout["states"], direction=direction, decision_layer=decision_layer,
                tool_id=tool_id, batch_size=args.batch_size,
            )
            summaries.append(summary)
            per_sample.extend(rows)

    se = summarize_s_over_e(summaries)
    baseline = {
        "train": {side: base.metric_summary(train["metrics"][side], tool_id) for side in ("clean", "corrupt")},
        "heldout": {side: base.metric_summary(heldout["metrics"][side], tool_id) for side in ("clean", "corrupt")},
    }
    selected_rows = suppressors + drivers
    write_csv(output_root / "selected_SE_features.csv", selected_rows)
    write_csv(output_root / "causal_SE_summary.csv", summaries)
    write_csv(output_root / "causal_SE_per_sample.csv", per_sample)
    write_json(output_root / "baseline_summary.json", baseline)
    run_config = {
        "experiment": "transcoder_SE_dominance",
        "model_key": model_key,
        "model_label": spec["label"],
        "model_path": str(model_path),
        "dataset_root": str(dataset_root),
        "transcoder_root": str(transcoder_root),
        "transcoder_checkpoints": checkpoint_map,
        "formation_layers": layers,
        "decision_layer": decision_layer,
        "train_pairs": len(train_pairs),
        "heldout_pairs": len(heldout_pairs),
        "alignment_policy": args.alignment_policy,
        "selection": "train aligned S=(delta<0,beta<0), E=(delta>0,beta>0); per-layer top |kappa| then global equal budget",
        "budget": args.budget,
        "per_layer_top_k": args.per_layer_top_k,
        "intervention": "heldout paired activation swap into actual HF MLP output",
        "S_to_clean": "clean prompt selected z <- paired corrupt z",
        "S_to_corrupt": "corrupt prompt selected z <- paired clean z",
        "direction": "unit mean(clean - corrupt) decision-layer output on train",
        "seed": args.seed,
    }
    write_json(output_root / "run_config.json", run_config)
    write_json(output_root / "S_over_E.json", {**se, "model_key": model_key, "model_label": spec["label"]})
    summary_text = [
        f"# {spec['label']}: Transcoder S/E causal dominance",
        "",
        f"Formation window: {','.join(f'L{x}' for x in layers)}; decision layer: L{decision_layer}; held-out pairs: {len(heldout_pairs)}.",
        "S is selected from corrupt-higher / write-away aligned features; E is clean-higher / write-toward aligned.",
        "",
        "| group | target side | transfer | mean Δm | mean Δ tool logit | mean Δ gate | top-1 flip | margin flip | logit-midpoint flip |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        summary_text.append(
            f"| {row['group']} | {row['side']} | {row['transfer']} | {row['mean_delta_m']:+.6f} | {row['mean_tool_logit_delta']:+.6f} | {row['mean_gate_delta']:+.6f} | {row['top1_flip_rate']:.2%} | {row['margin_boundary_flip_rate']:.2%} | {row['raw_tool_logit_midpoint_flip_rate']:.2%} |"
        )
    summary_text += [
        "",
        f"S:E = ({abs(se['S_to_clean_delta_m']):.6f} + {abs(se['S_to_corrupt_delta_m']):.6f}) / ({abs(se['E_to_clean_delta_m']):.6f} + {abs(se['E_to_corrupt_delta_m']):.6f}) = {se['S_over_E']:.6f}",
        "",
        "All feature choices are frozen from train; the four causal cells use held-out paired activation values.",
    ]
    (output_root / "summary.md").write_text("\n".join(summary_text) + "\n", encoding="utf-8")

    del acts, decoders, train_stats, heldout_stats, train, heldout, model, tokenizer
    base.clear_cuda()
    gc.collect()
    return {"model_key": model_key, "model_label": spec["label"], **se, "output_root": str(output_root)}


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    requested = [key.strip() for key in args.model_keys.split(",") if key.strip()]
    unknown = sorted(set(requested) - set(MODEL_SPECS))
    if unknown:
        raise ValueError(f"Unknown model keys: {unknown}")
    results: list[dict[str, Any]] = []
    for model_key in requested:
        results.append(run_model(args, model_key))
    write_json(args.output_root / "summary.json", results)
    write_csv(args.output_root / "summary.csv", results)
    lines = [
        "# Transcoder S/E dominance across available models",
        "",
        "| Model | S→clean Δm | E→clean Δm | S→corrupt Δm | E→corrupt Δm | S:E |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in results:
        lines.append(
            f"| {row['model_label']} | {row['S_to_clean_delta_m']:+.3f} | {row['E_to_clean_delta_m']:+.3f} | {row['S_to_corrupt_delta_m']:+.3f} | {row['E_to_corrupt_delta_m']:+.3f} | {row['S_over_E']:.3f} |"
        )
    lines += ["", "Mistral-Small-3.2-24B is not included because no matching Transcoder checkpoint is part of this audit."]
    (args.output_root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
