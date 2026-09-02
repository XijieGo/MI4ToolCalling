#!/usr/bin/env python3
"""Causal zero-ablation tests for the selected per-layer Transcoder top-20s.

For each target model/layer, features are selected on the train split using
the same aligned masks as the K audit:

* suppressor: delta_activation < 0 and beta_mu < 0;
* driver: delta_activation > 0 and beta_mu > 0.

The held-out intervention subtracts the selected Transcoder contribution from
the actual HF MLP output at the final position.  Suppressors are tested on
corrupt prompts (corrupt -> clean), drivers on clean prompts (clean ->
corrupt), and a layer-matched random group is a control.  In addition to
top-1 changes, the output records raw tool-call logit movement,
tool-vs-best-non-tool margin crossing, and paired logit-gap progress.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import random
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

import run_cross_model_transcoder_k as cross
import run_transcoder_feature_analysis as base
from path_defaults import REBUTTAL_ROOT, activation_cache_root


MODEL_TARGETS = {
    "granite_3p3_8b": 31,
    "qwen35_4b": 27,
    "qwen35_9b": 28,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-keys", type=str, default=",".join(MODEL_TARGETS))
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REBUTTAL_ROOT / "11_selected_layer_transcoder_causal_ablation",
    )
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--feature-batch-size", type=int, default=16)
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=20260802)
    return parser.parse_args()


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


def parse_model_keys(raw: str) -> list[str]:
    keys = [item.strip() for item in raw.split(",") if item.strip()]
    unknown = [key for key in keys if key not in MODEL_TARGETS]
    if unknown:
        raise ValueError(f"Unknown model keys: {unknown}")
    if not keys:
        raise ValueError("No model keys selected")
    return keys


def selected_feature_rows(
    target_layer: int,
    train_stats: dict[str, torch.Tensor],
    heldout_stats: dict[str, torch.Tensor],
    *,
    top_k: int,
    seed: int,
) -> tuple[dict[str, list[dict[str, Any]]], list[int]]:
    delta, beta, kappa = train_stats["delta"], train_stats["beta"], train_stats["kappa"]
    groups: dict[str, list[dict[str, Any]]] = {}
    masks = {
        "suppressor_top20": (delta < 0) & (beta < 0),
        "driver_top20": (delta > 0) & (beta > 0),
    }
    reserved: set[int] = set()
    for group_name, mask in masks.items():
        ids = torch.nonzero(mask, as_tuple=False).flatten()
        order = torch.argsort(kappa[ids].abs(), descending=True)[:top_k]
        rows: list[dict[str, Any]] = []
        for feature_id in ids[order].tolist():
            reserved.add(int(feature_id))
            rows.append(
                {
                    "layer": int(target_layer),
                    "feature_idx": int(feature_id),
                    "category": "suppressor" if group_name.startswith("suppressor") else "driver",
                    "selection_group": group_name,
                    "train_delta": float(delta[feature_id].item()),
                    "heldout_delta": float(heldout_stats["delta"][feature_id].item()),
                    "beta_mu": float(beta[feature_id].item()),
                    "train_kappa": float(kappa[feature_id].item()),
                    "heldout_kappa": float(heldout_stats["kappa"][feature_id].item()),
                    "train_abs_kappa": float(kappa[feature_id].abs().item()),
                    "heldout_abs_kappa": float(heldout_stats["kappa"][feature_id].abs().item()),
                    "train_active_rate_clean": float(train_stats["clean_active_rate"][feature_id].item()),
                    "train_active_rate_corrupt": float(train_stats["corrupt_active_rate"][feature_id].item()),
                    "heldout_active_rate_clean": float(heldout_stats["clean_active_rate"][feature_id].item()),
                    "heldout_active_rate_corrupt": float(heldout_stats["corrupt_active_rate"][feature_id].item()),
                }
            )
        if len(rows) != top_k:
            raise RuntimeError(f"{group_name}: only {len(rows)} aligned features, expected {top_k}")
        groups[group_name] = rows

    rng = random.Random(seed)
    d_feature = int(kappa.numel())
    pool = [feature_id for feature_id in range(d_feature) if feature_id not in reserved]
    random_ids = rng.sample(pool, top_k)
    groups["random_layer_matched"] = [
        {
            "layer": int(target_layer),
            "feature_idx": int(feature_id),
            "category": "random_layer_matched",
            "selection_group": "random_layer_matched",
        }
        for feature_id in random_ids
    ]
    return groups, sorted(reserved | set(random_ids))


def add_logit_level_metrics(
    rows: list[dict[str, Any]],
    baseline_metrics: dict[str, torch.Tensor],
    tool_id: int,
) -> None:
    """Add raw-logit and margin-crossing metrics to base intervention rows."""
    for row in rows:
        index = int(row["pair_index"])
        side = str(row["side"])
        target_side = "clean" if side == "corrupt" else "corrupt"
        source_tool = float(baseline_metrics[side]["tool_logit"][index].item())
        target_tool = float(baseline_metrics[target_side]["tool_logit"][index].item())
        source_margin = float(baseline_metrics[side]["top1_margin"][index].item())
        target_margin = float(baseline_metrics[target_side]["top1_margin"][index].item())
        intervention_tool = float(row["intervened_tool_logit"])
        intervention_margin = float(row["intervened_top1_margin"])
        midpoint = 0.5 * (source_tool + target_tool)
        denominator = target_tool - source_tool
        row.update(
            {
                "baseline_best_non_tool_logit": source_tool - source_margin,
                "intervened_best_non_tool_logit": intervention_tool - intervention_margin,
                "paired_target_tool_logit": target_tool,
                "paired_target_margin": target_margin,
                "paired_tool_logit_midpoint": midpoint,
                "tool_logit_gap_progress": (
                    (intervention_tool - source_tool) / denominator if abs(denominator) > 1e-6 else None
                ),
                # This is the logit-space analogue of top-1 recovery/drop:
                # the tool-call logit must cross the pair's clean/corrupt
                # midpoint, independently of which token is argmax.
                "raw_tool_logit_midpoint_flip": int(
                    intervention_tool >= midpoint if target_side == "clean" else intervention_tool <= midpoint
                ),
                "raw_tool_logit_reached_paired_target": int(
                    intervention_tool >= target_tool if target_side == "clean" else intervention_tool <= target_tool
                ),
                "margin_boundary_flip": int(
                    source_margin <= 0.0 < intervention_margin
                    if target_side == "clean"
                    else source_margin >= 0.0 > intervention_margin
                ),
                "top1_flip_same_direction": int(
                    not bool(row["baseline_is_tool_top1"]) and bool(row["intervened_is_tool_top1"])
                    if target_side == "clean"
                    else bool(row["baseline_is_tool_top1"]) and not bool(row["intervened_is_tool_top1"])
                ),
            }
        )


def intervention_summary(rows: list[dict[str, Any]], feature_count: int) -> dict[str, Any]:
    n = max(len(rows), 1)
    mean = lambda key: sum(float(row[key]) for row in rows) / n
    return {
        "group": rows[0]["group"] if rows else None,
        "side": rows[0]["side"] if rows else None,
        "direction": rows[0]["direction"] if rows else None,
        "n": len(rows),
        "feature_count": int(feature_count),
        "mean_intervened_tool_logit": mean("intervened_tool_logit"),
        "mean_tool_logit_delta": mean("tool_logit_delta"),
        "mean_intervened_best_non_tool_logit": mean("intervened_best_non_tool_logit"),
        "mean_top1_margin_delta": mean("top1_margin_delta"),
        "mean_log_odds_delta": mean("log_odds_delta"),
        "mean_tool_logit_gap_progress": mean("tool_logit_gap_progress"),
        "intervened_tool_top1_rate": sum(int(row["intervened_is_tool_top1"]) for row in rows) / n,
        "top1_flip_rate": sum(int(row["top1_flip_same_direction"]) for row in rows) / n,
        "margin_boundary_flip_rate": sum(int(row["margin_boundary_flip"]) for row in rows) / n,
        "raw_tool_logit_midpoint_flip_rate": sum(int(row["raw_tool_logit_midpoint_flip"]) for row in rows) / n,
        "raw_tool_logit_reached_paired_target_rate": sum(
            int(row["raw_tool_logit_reached_paired_target"]) for row in rows
        )
        / n,
        "mean_gate_score_delta": mean("gate_score_delta"),
    }


def run_ablation_intervention(
    model,
    tokenizer,
    pairs,
    *,
    group_name: str,
    group: list[dict[str, Any]],
    side: str,
    target_layer: int,
    decision_layer: int,
    weights: dict[str, Any],
    baseline_metrics: dict[str, dict[str, torch.Tensor]],
    baseline_states: dict[str, torch.Tensor],
    direction: torch.Tensor,
    layers,
    tool_id: int,
    batch_size: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Subtract the selected features' current decoder contribution."""
    feature_ids = torch.tensor([int(row["feature_idx"]) for row in group], dtype=torch.long)
    device = base.model_device(model)
    rows: list[dict[str, Any]] = []
    pad_token_id = int(tokenizer.pad_token_id)
    for batch, clean_cpu, corrupt_cpu, attention_mask_cpu in tqdm(
        base.iter_batches(pairs, batch_size, pad_token_id),
        desc=f"Ablation {group_name}/{side}",
        dynamic_ncols=True,
        leave=False,
    ):
        indices = [item.index for item in batch]
        tokens = (clean_cpu if side == "clean" else corrupt_cpu).to(device)
        attention_mask = attention_mask_cpu.to(device)
        state: dict[str, torch.Tensor] = {}

        def capture_pre(_module, inputs):
            if not inputs or not isinstance(inputs[0], torch.Tensor):
                raise TypeError(f"L{target_layer}: MLP pre-hook did not receive hidden states")
            state["x"] = inputs[0][:, -1, :].detach()

        def ablate_post(_module, _inputs, output):
            if "x" not in state:
                raise RuntimeError(f"L{target_layer}: MLP post-hook has no captured input")
            current = base.tensor_output(output)
            x = state.pop("x")
            enc = weights["W_enc"].index_select(0, feature_ids).to(device=current.device, dtype=torch.bfloat16)
            bias = weights["b_enc"].index_select(0, feature_ids).to(device=current.device, dtype=torch.bfloat16)
            dec = weights["W_dec"].index_select(0, feature_ids).to(device=current.device, dtype=torch.bfloat16)
            acts = F.relu(F.linear(x.to(dtype=torch.bfloat16), enc, bias))
            contribution = acts.float().matmul(dec.float())
            edited = current.clone()
            edited[:, -1, :] = edited[:, -1, :] - contribution.to(dtype=edited.dtype)
            return base.replace_tensor_output(output, edited)

        gate_holder: dict[str, torch.Tensor] = {}

        def decision_hook(_module, _inputs, output):
            gate_holder["state"] = base.tensor_output(output)[:, -1, :].detach().float().cpu()

        handles = [
            layers[target_layer].mlp.register_forward_pre_hook(capture_pre),
            layers[target_layer].mlp.register_forward_hook(ablate_post),
            layers[decision_layer].register_forward_hook(decision_hook),
        ]
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
            raise RuntimeError(f"{group_name}: decision-layer hook did not fire")
        metrics = base.metric_tensors(logits, tool_id)
        gate = gate_holder["state"].matmul(direction.float())
        base_gate = baseline_states[side].index_select(0, torch.tensor(indices)).matmul(direction.float())
        for local, item in enumerate(batch):
            base_row = {
                "tool_logit": float(baseline_metrics[side]["tool_logit"][item.index].item()),
                "tool_probability": float(baseline_metrics[side]["tool_prob"][item.index].item()),
                "top1_margin": float(baseline_metrics[side]["top1_margin"][item.index].item()),
                "log_odds": float(baseline_metrics[side]["log_odds"][item.index].item()),
                "tool_rank": int(baseline_metrics[side]["rank"][item.index].item()),
                "is_tool_top1": int(baseline_metrics[side]["top1"][item.index].item() == tool_id),
            }
            rows.append(
                {
                    "group": group_name,
                    "side": side,
                    "direction": "ablate",
                    "sample_id": item.sample_id,
                    "pair_index": int(item.index),
                    "intervened_tool_logit": float(metrics["tool_logit"][local].item()),
                    "intervened_tool_probability": float(metrics["tool_prob"][local].item()),
                    "intervened_top1_margin": float(metrics["top1_margin"][local].item()),
                    "intervened_log_odds": float(metrics["log_odds"][local].item()),
                    "intervened_tool_rank": int(metrics["rank"][local].item()),
                    "intervened_is_tool_top1": int(metrics["top1"][local].item() == tool_id),
                    "tool_logit_delta": float(metrics["tool_logit"][local].item()) - base_row["tool_logit"],
                    "tool_probability_delta": float(metrics["tool_prob"][local].item()) - base_row["tool_probability"],
                    "top1_margin_delta": float(metrics["top1_margin"][local].item()) - base_row["top1_margin"],
                    "log_odds_delta": float(metrics["log_odds"][local].item()) - base_row["log_odds"],
                    "gate_score": float(gate[local].item()),
                    "gate_score_delta": float(gate[local].item() - base_gate[local].item()),
                    "baseline_tool_logit": base_row["tool_logit"],
                    "baseline_tool_probability": base_row["tool_probability"],
                    "baseline_tool_rank": base_row["tool_rank"],
                    "baseline_is_tool_top1": base_row["is_tool_top1"],
                }
            )
        del tokens, attention_mask, outputs, logits, metrics, gate_holder
        base.clear_cuda()
    add_logit_level_metrics(rows, baseline_metrics, tool_id)
    return intervention_summary(rows, len(group)), rows


def run_model(args: argparse.Namespace, model_key: str) -> dict[str, Any]:
    spec = cross.MODEL_SPECS[model_key]
    target_layer = MODEL_TARGETS[model_key]
    output_root = (args.output_root / model_key).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    dataset_root = Path(spec["dataset_root"])
    transcoder_root = Path(spec["transcoder_root"])
    reference_layer = int(spec["reference_layer"])
    cache_root = activation_cache_root(model_key)
    cache = torch.load(cache_root / "activation_cache.pt", map_location="cpu", weights_only=False)
    direction_raw = cache["train_states"]["clean"].float().mean(dim=0) - cache["train_states"]["corrupt"].float().mean(dim=0)
    direction = direction_raw / direction_raw.norm()

    model, tokenizer = base.load_model(Path(spec["model_path"]), args.dtype, args.device)
    marker = cross.load_marker(dataset_root)
    marker_ids = tokenizer.encode(marker, add_special_tokens=False)
    if len(marker_ids) != 1:
        raise ValueError(f"{model_key}: marker {marker!r} is not one token: {marker_ids}")
    tool_id = int(marker_ids[0])
    text_model = base.resolve_text_model(model)
    layers = text_model.layers

    all_train = base.load_pairs(dataset_root, tokenizer, "train", token_length_policy="allow")
    all_heldout = base.load_pairs(dataset_root, tokenizer, "heldout", token_length_policy="allow")
    train_pairs = base.load_pairs(dataset_root, tokenizer, "train", token_length_policy="skip")
    heldout_pairs = base.load_pairs(dataset_root, tokenizer, "heldout", token_length_policy="skip")
    if len(train_pairs) != int(cache["train_states"]["clean"].shape[0]) or len(heldout_pairs) != int(cache["heldout_states"]["clean"].shape[0]):
        raise RuntimeError(f"{model_key}: pair/cache mismatch after strict alignment")

    checkpoint = cross.resolve_checkpoint(transcoder_root, target_layer)
    weights = base.load_transcoder(checkpoint, target_layer)
    device = torch.device(args.device)

    def feature_stats(inputs: dict[str, dict[int, torch.Tensor]]) -> dict[str, torch.Tensor]:
        clean = base.collect_feature_acts(inputs["clean"][target_layer], weights["W_enc"], weights["b_enc"], device=device, batch_size=args.feature_batch_size)
        corrupt = base.collect_feature_acts(inputs["corrupt"][target_layer], weights["W_enc"], weights["b_enc"], device=device, batch_size=args.feature_batch_size)
        delta = clean["mean"] - corrupt["mean"]
        beta = weights["W_dec"].matmul(direction)
        return {
            "delta": delta,
            "beta": beta,
            "kappa": delta * beta,
            "clean_active_rate": clean["active_rate"],
            "corrupt_active_rate": corrupt["active_rate"],
        }

    train_stats = feature_stats(cache["train_inputs"])
    heldout_stats = feature_stats(cache["heldout_inputs"])
    groups, selected_ids = selected_feature_rows(
        target_layer, train_stats, heldout_stats, top_k=args.top_k, seed=args.seed
    )
    selected_tensor = torch.tensor(selected_ids, dtype=torch.long)
    heldout_clean = base.collect_feature_acts(
        cache["heldout_inputs"]["clean"][target_layer], weights["W_enc"], weights["b_enc"],
        device=device, batch_size=args.feature_batch_size, selected_ids=selected_tensor,
    )["selected"]
    heldout_corrupt = base.collect_feature_acts(
        cache["heldout_inputs"]["corrupt"][target_layer], weights["W_enc"], weights["b_enc"],
        device=device, batch_size=args.feature_batch_size, selected_ids=selected_tensor,
    )["selected"]
    selected_acts = {
        target_layer: {"ids": selected_tensor, "clean": heldout_clean, "corrupt": heldout_corrupt}
    }

    baseline = base.capture_dataset(
        model, tokenizer, heldout_pairs, capture_layers=[target_layer], decision_layer=reference_layer,
        tool_id=tool_id, batch_size=args.batch_size, label=f"{model_key} heldout baseline",
    )
    baseline_metrics = baseline["metrics"]
    baseline_states = baseline["states"]
    write_json(
        output_root / "baseline_summary.json",
        {side: base.metric_summary(baseline_metrics[side], tool_id) for side in ("clean", "corrupt")},
    )
    selected_rows = [row for group in groups.values() for row in group]
    write_csv(output_root / "selected_features.csv", selected_rows)

    intervention_summaries: list[dict[str, Any]] = []
    intervention_rows: list[dict[str, Any]] = []
    run_specs = [
        ("suppressor_top20", "corrupt"),
        ("driver_top20", "clean"),
        ("random_layer_matched", "corrupt"),
    ]
    for group_name, side in run_specs:
        group = groups[group_name]
        summary, rows = run_ablation_intervention(
            model,
            tokenizer,
            heldout_pairs,
            group_name=group_name,
            group=group,
            side=side,
            target_layer=target_layer,
            decision_layer=reference_layer,
            weights=weights,
            baseline_metrics=baseline_metrics,
            baseline_states=baseline_states,
            direction=direction,
            layers=layers,
            tool_id=tool_id,
            batch_size=args.batch_size,
        )
        intervention_summaries.append(summary)
        intervention_rows.extend(rows)

    config = {
        "experiment": "selected_layer_transcoder_causal_logit_audit",
        "model_key": model_key,
        "model_label": spec["label"],
        "model_path": str(spec["model_path"]),
        "dataset_root": str(dataset_root),
        "transcoder_checkpoint": str(checkpoint),
        "target_layer": target_layer,
        "reference_layer": reference_layer,
        "train_pairs": len(train_pairs),
        "heldout_pairs": len(heldout_pairs),
        "full_train_pairs_before_alignment": len(all_train),
        "full_heldout_pairs_before_alignment": len(all_heldout),
        "train_token_length_mismatch_count": len(all_train) - len(train_pairs),
        "heldout_token_length_mismatch_count": len(all_heldout) - len(heldout_pairs),
        "top_k": args.top_k,
        "feature_selection": "train aligned mask; top-k abs(kappa) within target layer",
        "intervention": "heldout zero-ablation: subtract selected Transcoder contribution from actual HF MLP output final position",
        "logit_boundary": "tool_logit - best_non_tool_logit crosses from <=0 to >0 for rescue, or >=0 to <0 for drop",
        "raw_logit_flip": "tool-call logit crosses paired clean/corrupt midpoint",
        "dtype": args.dtype,
        "seed": args.seed,
    }
    write_json(output_root / "run_config.json", config)
    write_json(output_root / "causal_intervention_summary.json", intervention_summaries)
    write_csv(output_root / "causal_intervention_summary.csv", intervention_summaries)
    write_csv(output_root / "causal_intervention_per_sample.csv", intervention_rows)

    md = [
        f"# {spec['label']} L{target_layer} selected Transcoder top-{args.top_k} causal audit",
        "",
        f"Heldout pairs: {len(heldout_pairs)}; reference layer: L{reference_layer}; checkpoint: `{checkpoint}`.",
        "The raw-logit midpoint flip is separate from top-1: it asks whether the intervened tool-call logit crosses the paired clean/corrupt tool-logit midpoint.",
        "",
        "| group | side | direction | mean tool-logit Δ | mean margin Δ | gap progress | top-1 flip | margin flip | raw-logit midpoint flip | reached paired target logit |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in intervention_summaries:
        md.append(
            f"| {row['group']} | {row['side']} | {row['direction']} | {row['mean_tool_logit_delta']:+.4f} | "
            f"{row['mean_top1_margin_delta']:+.4f} | {row['mean_tool_logit_gap_progress']:+.3f} | "
            f"{100*row['top1_flip_rate']:.1f}% | {100*row['margin_boundary_flip_rate']:.1f}% | "
            f"{100*row['raw_tool_logit_midpoint_flip_rate']:.1f}% | "
            f"{100*row['raw_tool_logit_reached_paired_target_rate']:.1f}% |"
        )
    (output_root / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    del model, tokenizer, weights, cache, baseline
    gc.collect()
    base.clear_cuda()
    return {"model_key": model_key, "output_root": str(output_root), "summaries": intervention_summaries}


def main() -> None:
    args = parse_args()
    if args.top_k <= 0:
        raise ValueError("--top-k must be positive")
    results = []
    for model_key in parse_model_keys(args.model_keys):
        results.append(run_model(args, model_key))
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
