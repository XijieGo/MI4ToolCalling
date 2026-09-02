#!/usr/bin/env python3
"""Sweep causal effects of train-selected aligned Transcoder suppressors.

This is the old-paper-style follow-up to the single top-20 audit.  At each
model's selected target layer, the train split ranks features satisfying

    clean-minus-corrupt activation delta < 0 and decoder beta < 0.

The held-out corrupt prompts are then intervened on with k=1,5,10,20 using
either zero-ablation or paired activation replacement (corrupt activation ->
clean activation).  A layer-matched random feature group is an ablation
control.  The output keeps gate movement separate from behavior: tool-call
raw-logit movement, paired-logit midpoint crossing, margin crossing, and
top-1 recovery are all recorded.
"""

from __future__ import annotations

import argparse
import gc
import json
import random
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

import run_cross_model_transcoder_k as cross
import run_selected_layer_transcoder_causal as single
import run_transcoder_feature_analysis as base
from path_defaults import REBUTTAL_ROOT, activation_cache_root


MODEL_TARGETS = dict(single.MODEL_TARGETS)
K_VALUES = (1, 5, 10, 20)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-keys", type=str, default=",".join(MODEL_TARGETS))
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REBUTTAL_ROOT / "12_selected_layer_causal_sweep",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--feature-batch-size", type=int, default=16)
    parser.add_argument(
        "--consistency-threshold",
        type=float,
        default=None,
        help="Optional train-pair contribution consistency threshold for selected groups (e.g. 0.80).",
    )
    parser.add_argument(
        "--random-consistency-threshold",
        type=float,
        default=None,
        help="Optional consistency threshold for the layer-matched random suppressor control.",
    )
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=20260802)
    return parser.parse_args()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    single.write_csv(path, rows)


def make_feature_stats(
    inputs: dict[str, torch.Tensor],
    weights: dict[str, Any],
    direction: torch.Tensor,
    *,
    device: torch.device,
    batch_size: int,
    return_consistency: bool = False,
) -> dict[str, torch.Tensor]:
    clean = base.collect_feature_acts(
        inputs["clean"],
        weights["W_enc"],
        weights["b_enc"],
        device=device,
        batch_size=batch_size,
        return_acts=return_consistency,
    )
    corrupt = base.collect_feature_acts(
        inputs["corrupt"],
        weights["W_enc"],
        weights["b_enc"],
        device=device,
        batch_size=batch_size,
        return_acts=return_consistency,
    )
    delta = clean["mean"] - corrupt["mean"]
    beta = weights["W_dec"].matmul(direction)
    result = {
        "delta": delta,
        "beta": beta,
        "kappa": delta * beta,
        "clean_active_rate": clean["active_rate"],
        "corrupt_active_rate": corrupt["active_rate"],
    }
    if return_consistency:
        pair_delta = clean["acts"] - corrupt["acts"]
        result["consistency"] = (pair_delta * beta.unsqueeze(0) > 0.0).float().mean(dim=0)
    return result


def make_rows(
    layer: int,
    feature_ids: list[int],
    category: str,
    selection_group: str,
    train_stats: dict[str, torch.Tensor],
    heldout_stats: dict[str, torch.Tensor],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for feature_id in feature_ids:
        rows.append(
            {
                "layer": int(layer),
                "feature_idx": int(feature_id),
                "category": category,
                "selection_group": selection_group,
                "train_delta": float(train_stats["delta"][feature_id].item()),
                "heldout_delta": float(heldout_stats["delta"][feature_id].item()),
                "beta_mu": float(train_stats["beta"][feature_id].item()),
                "train_kappa": float(train_stats["kappa"][feature_id].item()),
                "heldout_kappa": float(heldout_stats["kappa"][feature_id].item()),
                "train_abs_kappa": float(train_stats["kappa"][feature_id].abs().item()),
                "heldout_abs_kappa": float(heldout_stats["kappa"][feature_id].abs().item()),
                "train_active_rate_clean": float(train_stats["clean_active_rate"][feature_id].item()),
                "train_active_rate_corrupt": float(train_stats["corrupt_active_rate"][feature_id].item()),
                "heldout_active_rate_clean": float(heldout_stats["clean_active_rate"][feature_id].item()),
                "heldout_active_rate_corrupt": float(heldout_stats["corrupt_active_rate"][feature_id].item()),
                "train_consistency": (
                    float(train_stats["consistency"][feature_id].item())
                    if "consistency" in train_stats
                    else None
                ),
            }
        )
    return rows


def select_sweep_groups(
    layer: int,
    train_stats: dict[str, torch.Tensor],
    heldout_stats: dict[str, torch.Tensor],
    *,
    seed: int,
    consistency_threshold: float | None = None,
    random_consistency_threshold: float | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Select a frozen top-20 suppressor list and a matched random list."""
    delta, beta, kappa = train_stats["delta"], train_stats["beta"], train_stats["kappa"]
    suppressor_mask = (delta < 0) & (beta < 0)
    if consistency_threshold is not None:
        if "consistency" not in train_stats:
            raise RuntimeError("Consistency threshold requested without pair-level train statistics")
        suppressor_mask &= train_stats["consistency"] >= float(consistency_threshold)
    suppressor_ids = torch.nonzero(suppressor_mask, as_tuple=False).flatten()
    suppressor_order = torch.argsort(kappa[suppressor_ids].abs(), descending=True)[:20]
    suppressor_ids = [int(x) for x in suppressor_ids[suppressor_order].tolist()]
    if len(suppressor_ids) != 20:
        raise RuntimeError(f"L{layer}: only {len(suppressor_ids)} aligned suppressors")

    driver_mask = (delta > 0) & (beta > 0)
    if consistency_threshold is not None:
        driver_mask &= train_stats["consistency"] >= float(consistency_threshold)
    driver_ids_tensor = torch.nonzero(driver_mask, as_tuple=False).flatten()
    driver_order = torch.argsort(kappa[driver_ids_tensor].abs(), descending=True)[:20]
    driver_ids = [int(x) for x in driver_ids_tensor[driver_order].tolist()]
    if len(driver_ids) != 20:
        raise RuntimeError(f"L{layer}: only {len(driver_ids)} aligned drivers")

    reserved = set(suppressor_ids) | set(driver_ids)
    if random_consistency_threshold is not None:
        if "consistency" not in train_stats:
            raise RuntimeError("Random consistency threshold requested without pair-level train statistics")
        random_mask = (delta < 0) & (beta < 0) & (
            train_stats["consistency"] >= float(random_consistency_threshold)
        )
        pool = [
            int(feature_id)
            for feature_id in torch.nonzero(random_mask, as_tuple=False).flatten().tolist()
            if int(feature_id) not in reserved
        ]
    else:
        pool = [feature_id for feature_id in range(int(kappa.numel())) if feature_id not in reserved]
    if len(pool) < 20:
        raise RuntimeError(
            f"L{layer}: only {len(pool)} random-control candidates after the requested consistency filter"
        )
    random_ids = random.Random(seed).sample(pool, 20)
    return {
        "suppressor_top20": make_rows(
            layer, suppressor_ids, "suppressor", "suppressor_top20", train_stats, heldout_stats
        ),
        "driver_top20": make_rows(layer, driver_ids, "driver", "driver_top20", train_stats, heldout_stats),
        "random_layer_matched": make_rows(
            layer, random_ids, "random_layer_matched", "random_layer_matched", train_stats, heldout_stats
        ),
    }


def run_feature_intervention(
    model,
    tokenizer,
    pairs,
    *,
    group_name: str,
    group: list[dict[str, Any]],
    side: str,
    mode: str,
    k: int,
    target_layer: int,
    decision_layer: int,
    weights: dict[str, Any],
    selected_acts: dict[int, dict[str, torch.Tensor]],
    baseline_metrics: dict[str, dict[str, torch.Tensor]],
    baseline_states: dict[str, torch.Tensor],
    direction: torch.Tensor,
    layers,
    tool_id: int,
    batch_size: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if mode not in {"ablate", "swap"}:
        raise ValueError(mode)
    group = group[:k]
    feature_ids = torch.tensor([int(row["feature_idx"]) for row in group], dtype=torch.long)
    available_ids = [int(x) for x in selected_acts[target_layer]["ids"].tolist()]
    positions = torch.tensor([available_ids.index(int(x)) for x in feature_ids.tolist()], dtype=torch.long)
    device = base.model_device(model)
    rows: list[dict[str, Any]] = []
    pad_token_id = int(tokenizer.pad_token_id)

    for batch, clean_cpu, corrupt_cpu, attention_mask_cpu in tqdm(
        base.iter_batches(pairs, batch_size, pad_token_id),
        desc=f"{mode} {group_name}/{side}/k={k}",
        dynamic_ncols=True,
        leave=False,
    ):
        indices = [item.index for item in batch]
        index_tensor = torch.tensor(indices, dtype=torch.long)
        tokens = (clean_cpu if side == "clean" else corrupt_cpu).to(device)
        attention_mask = attention_mask_cpu.to(device)
        state: dict[str, torch.Tensor] = {}

        def capture_pre(_module, inputs):
            if not inputs or not isinstance(inputs[0], torch.Tensor):
                raise TypeError(f"L{target_layer}: MLP pre-hook did not receive hidden states")
            state["x"] = inputs[0][:, -1, :].detach()

        def edit_post(_module, _inputs, output):
            if "x" not in state:
                raise RuntimeError(f"L{target_layer}: MLP post-hook has no captured input")
            current = base.tensor_output(output)
            x = state.pop("x")
            enc = weights["W_enc"].index_select(0, feature_ids).to(device=current.device, dtype=torch.bfloat16)
            bias = weights["b_enc"].index_select(0, feature_ids).to(device=current.device, dtype=torch.bfloat16)
            dec = weights["W_dec"].index_select(0, feature_ids).to(device=current.device, dtype=torch.bfloat16)
            acts = F.relu(F.linear(x.to(dtype=torch.bfloat16), enc, bias))
            if mode == "ablate":
                contribution = acts.float().matmul(dec.float())
            else:
                target_side = "clean" if side == "corrupt" else "corrupt"
                target_values = selected_acts[target_layer][target_side].index_select(0, index_tensor).index_select(1, positions)
                contribution = (target_values.to(device=current.device, dtype=torch.float32) - acts.float()).matmul(dec.float())
            edited = current.clone()
            if mode == "ablate":
                edited[:, -1, :] = edited[:, -1, :] - contribution.to(dtype=edited.dtype)
            else:
                edited[:, -1, :] = edited[:, -1, :] + contribution.to(dtype=edited.dtype)
            return base.replace_tensor_output(output, edited)

        gate_holder: dict[str, torch.Tensor] = {}

        def decision_hook(_module, _inputs, output):
            gate_holder["state"] = base.tensor_output(output)[:, -1, :].detach().float().cpu()

        handles = [
            layers[target_layer].mlp.register_forward_pre_hook(capture_pre),
            layers[target_layer].mlp.register_forward_hook(edit_post),
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
        base_gate = baseline_states[side].index_select(0, index_tensor).matmul(direction.float())
        for local, item in enumerate(batch):
            baseline_tool_top1 = int(baseline_metrics[side]["top1"][item.index].item() == tool_id)
            baseline_tool_logit = float(baseline_metrics[side]["tool_logit"][item.index].item())
            baseline_tool_prob = float(baseline_metrics[side]["tool_prob"][item.index].item())
            baseline_margin = float(baseline_metrics[side]["top1_margin"][item.index].item())
            baseline_log_odds = float(baseline_metrics[side]["log_odds"][item.index].item())
            intervened_tool_top1 = int(metrics["top1"][local].item() == tool_id)
            rows.append(
                {
                    "group": group_name,
                    "side": side,
                    "direction": "to_clean" if side == "corrupt" else "to_corrupt",
                    "mode": mode,
                    "k": int(k),
                    "sample_id": item.sample_id,
                    "pair_index": int(item.index),
                    "intervened_tool_logit": float(metrics["tool_logit"][local].item()),
                    "intervened_tool_probability": float(metrics["tool_prob"][local].item()),
                    "intervened_top1_margin": float(metrics["top1_margin"][local].item()),
                    "intervened_log_odds": float(metrics["log_odds"][local].item()),
                    "intervened_tool_rank": int(metrics["rank"][local].item()),
                    "intervened_is_tool_top1": intervened_tool_top1,
                    "tool_logit_delta": float(metrics["tool_logit"][local].item()) - baseline_tool_logit,
                    "tool_probability_delta": float(metrics["tool_prob"][local].item()) - baseline_tool_prob,
                    "top1_margin_delta": float(metrics["top1_margin"][local].item()) - baseline_margin,
                    "log_odds_delta": float(metrics["log_odds"][local].item()) - baseline_log_odds,
                    "gate_score": float(gate[local].item()),
                    "gate_score_delta": float(gate[local].item() - base_gate[local].item()),
                    "baseline_tool_logit": baseline_tool_logit,
                    "baseline_tool_probability": baseline_tool_prob,
                    "baseline_tool_rank": int(baseline_metrics[side]["rank"][item.index].item()),
                    "baseline_is_tool_top1": baseline_tool_top1,
                }
            )
        del tokens, attention_mask, outputs, logits, metrics, gate_holder
        base.clear_cuda()

    single.add_logit_level_metrics(rows, baseline_metrics, tool_id)
    summary = single.intervention_summary(rows, len(group))
    summary.update({"mode": mode, "k": int(k)})
    return summary, rows


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
    layers = base.resolve_text_model(model).layers
    all_train = base.load_pairs(dataset_root, tokenizer, "train", token_length_policy="allow")
    all_heldout = base.load_pairs(dataset_root, tokenizer, "heldout", token_length_policy="allow")
    train_pairs = base.load_pairs(dataset_root, tokenizer, "train", token_length_policy="skip")
    heldout_pairs = base.load_pairs(dataset_root, tokenizer, "heldout", token_length_policy="skip")
    if len(train_pairs) != int(cache["train_states"]["clean"].shape[0]) or len(heldout_pairs) != int(cache["heldout_states"]["clean"].shape[0]):
        raise RuntimeError(f"{model_key}: pair/cache mismatch after strict alignment")

    checkpoint = cross.resolve_checkpoint(transcoder_root, target_layer)
    weights = base.load_transcoder(checkpoint, target_layer)
    device = torch.device(args.device)
    train_inputs = {
        "clean": cache["train_inputs"]["clean"][target_layer],
        "corrupt": cache["train_inputs"]["corrupt"][target_layer],
    }
    heldout_inputs = {
        "clean": cache["heldout_inputs"]["clean"][target_layer],
        "corrupt": cache["heldout_inputs"]["corrupt"][target_layer],
    }
    need_consistency = args.consistency_threshold is not None or args.random_consistency_threshold is not None
    train_stats = make_feature_stats(
        train_inputs,
        weights,
        direction,
        device=device,
        batch_size=args.feature_batch_size,
        return_consistency=need_consistency,
    )
    heldout_stats = make_feature_stats(heldout_inputs, weights, direction, device=device, batch_size=args.feature_batch_size)
    groups = select_sweep_groups(
        target_layer,
        train_stats,
        heldout_stats,
        seed=args.seed,
        consistency_threshold=args.consistency_threshold,
        random_consistency_threshold=args.random_consistency_threshold,
    )
    selected_ids = sorted({int(row["feature_idx"]) for group in groups.values() for row in group})
    selected_tensor = torch.tensor(selected_ids, dtype=torch.long)
    selected_acts = {
        target_layer: {
            "ids": selected_tensor,
            "clean": base.collect_feature_acts(
                cache["heldout_inputs"]["clean"][target_layer], weights["W_enc"], weights["b_enc"],
                device=device, batch_size=args.feature_batch_size, selected_ids=selected_tensor,
            )["selected"],
            "corrupt": base.collect_feature_acts(
                cache["heldout_inputs"]["corrupt"][target_layer], weights["W_enc"], weights["b_enc"],
                device=device, batch_size=args.feature_batch_size, selected_ids=selected_tensor,
            )["selected"],
        }
    }

    baseline = base.capture_dataset(
        model, tokenizer, heldout_pairs, capture_layers=[target_layer], decision_layer=reference_layer,
        tool_id=tool_id, batch_size=args.batch_size, label=f"{model_key} heldout baseline",
    )
    baseline_metrics, baseline_states = baseline["metrics"], baseline["states"]
    write_json(output_root / "baseline_summary.json", {side: base.metric_summary(baseline_metrics[side], tool_id) for side in ("clean", "corrupt")})
    write_csv(output_root / "selected_features.csv", [row for group in groups.values() for row in group])

    summaries: list[dict[str, Any]] = []
    all_rows: list[dict[str, Any]] = []
    # The paper-style primary direction is corrupt -> clean.  The driver group
    # is retained as a secondary symmetric check on clean prompts.
    run_groups = [("suppressor_top20", "corrupt", True), ("driver_top20", "clean", True), ("random_layer_matched", "corrupt", False)]
    for group_name, side, include_swap in run_groups:
        for k in K_VALUES:
            modes = ("ablate", "swap") if include_swap else ("ablate",)
            for mode in modes:
                summary, rows = run_feature_intervention(
                    model, tokenizer, heldout_pairs, group_name=group_name, group=groups[group_name],
                    side=side, mode=mode, k=k, target_layer=target_layer, decision_layer=reference_layer,
                    weights=weights, selected_acts=selected_acts, baseline_metrics=baseline_metrics,
                    baseline_states=baseline_states, direction=direction, layers=layers, tool_id=tool_id,
                    batch_size=args.batch_size,
                )
                summaries.append(summary)
                all_rows.extend(rows)

    config = {
        "experiment": "selected_layer_transcoder_causal_k_sweep",
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
        "k_values": list(K_VALUES),
        "feature_selection": (
            "train-frozen aligned suppressor delta<0,beta<0 and driver delta>0,beta>0; "
            "descending abs(kappa) within target layer"
        ),
        "consistency_threshold": args.consistency_threshold,
        "random_consistency_threshold": args.random_consistency_threshold,
        "primary_intervention": "heldout corrupt->clean zero-ablation and paired feature activation replacement",
        "logit_boundary": "tool_logit - best_non_tool_logit crosses 0",
        "raw_logit_flip": "tool-call logit crosses paired clean/corrupt midpoint",
        "dtype": args.dtype,
        "seed": args.seed,
    }
    write_json(output_root / "run_config.json", config)
    write_json(output_root / "causal_sweep_summary.json", summaries)
    write_csv(output_root / "causal_sweep_summary.csv", summaries)
    write_csv(output_root / "causal_sweep_per_sample.csv", all_rows)

    lines = [
        f"# {spec['label']} L{target_layer} causal k sweep",
        "",
        f"Heldout pairs: {len(heldout_pairs)}; reference layer: L{reference_layer}; checkpoint: `{checkpoint}`.",
        "Suppressor groups are selected on train with delta<0 and beta<0, then frozen on heldout.",
        "Raw-logit midpoint flip is separate from top-1 and margin-boundary flip.",
        "",
        "| group | k | mode | side | mean Δ tool logit | mean Δ margin | gap progress | gate Δ | top-1 flip | raw-logit midpoint | margin flip |",
        "|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            f"| {row['group']} | {row['k']} | {row['mode']} | {row['side']} | {row['mean_tool_logit_delta']:+.4f} | "
            f"{row['mean_top1_margin_delta']:+.4f} | {row['mean_tool_logit_gap_progress']:+.3f} | {row['mean_gate_score_delta']:+.3f} | "
            f"{100*row['top1_flip_rate']:.1f}% | {100*row['raw_tool_logit_midpoint_flip_rate']:.1f}% | {100*row['margin_boundary_flip_rate']:.1f}% |"
        )
    (output_root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    del model, tokenizer, weights, cache, baseline
    gc.collect()
    base.clear_cuda()
    return {"model_key": model_key, "output_root": str(output_root), "summaries": summaries}


def main() -> None:
    args = parse_args()
    results = [run_model(args, key) for key in single.parse_model_keys(args.model_keys)]
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
