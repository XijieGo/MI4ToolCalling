#!/usr/bin/env python3
"""Paper-faithful Qwen3.5 Transcoder decomposition and causal ablation.

This runner keeps two quantities separate:

* the paper's descriptive ``K_corrupt``/``K_clean`` totals, which group every
  feature by the side on which it has the larger mean activation; and
* the aligned quadrants used to identify a corrupt-side suppressor
  (clean-minus-corrupt activation < 0, decoder projection < 0) or a clean-side
  driver (both signs > 0).

Its ``K`` columns follow the activation-side definition in the revised paper
prose.  The optional ``top-20`` columns are per-layer activation-side
diagnostics; they are not the historical cross-scale Table 3 statistic, which
uses aligned masks and a global top-20 over its formation window.  That latter
protocol is audited separately by ``audit_table3_k_protocol.py``.

The 200 train pairs fit the L29 clean-minus-corrupt direction and select the
top aligned features.  The 300 held-out pairs provide the paper-style K
summary, feature diagnostics, and causal ablations.  The causal experiment
ablates the selected Transcoder decoder contributions from the actual MLP
output on held-out corrupt prompts, matching the paper's top-feature ablation
rather than a native-unit E/S swap protocol.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

import run_transcoder_feature_analysis as base
from path_defaults import REBUTTAL_ROOT


DEFAULT_OUTPUT_ROOT = REBUTTAL_ROOT / "09_qwen35_transcoder_feature_analysis" / "qwen35_4b_paper_style"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=base.DEFAULT_MODEL_PATH)
    parser.add_argument("--dataset-root", type=Path, default=base.DEFAULT_DATASET_ROOT)
    parser.add_argument("--transcoder-root", type=Path, default=base.DEFAULT_TC_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--decision-layer", type=int, default=29)
    parser.add_argument("--layers", type=str, default="28,29")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--paper-top-k", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--feature-batch-size", type=int, default=16)
    parser.add_argument("--token-top-k", type=int, default=8)
    parser.add_argument("--example-top-k", type=int, default=6)
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=20260802)
    parser.add_argument("--skip-causal", action="store_true")
    return parser.parse_args()


def parse_layers(raw: str) -> list[int]:
    layers = sorted({int(part.strip()) for part in raw.split(",") if part.strip()})
    if not layers or any(layer < 0 for layer in layers):
        raise ValueError(f"Invalid layer list: {raw!r}")
    return layers


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    base.ensure_dir(path.parent)
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


def feature_stats(
    payload: dict[str, Any],
    layer: int,
    weights: dict[str, Any],
    u: torch.Tensor,
    *,
    device: torch.device,
    batch_size: int,
) -> dict[str, torch.Tensor]:
    clean = base.collect_feature_acts(
        payload["inputs"]["clean"][layer],
        weights["W_enc"],
        weights["b_enc"],
        device=device,
        batch_size=batch_size,
    )
    corrupt = base.collect_feature_acts(
        payload["inputs"]["corrupt"][layer],
        weights["W_enc"],
        weights["b_enc"],
        device=device,
        batch_size=batch_size,
    )
    beta = weights["W_dec"].matmul(u.float())
    delta = clean["mean"] - corrupt["mean"]
    return {
        "delta": delta,
        "beta": beta,
        "kappa": delta * beta,
        "clean_active_rate": clean["active_rate"],
        "corrupt_active_rate": corrupt["active_rate"],
    }


def side_mask(delta: torch.Tensor, side: str) -> torch.Tensor:
    if side == "corrupt":
        return delta < 0
    if side == "clean":
        return delta > 0
    raise ValueError(side)


def mass_for_mask(kappa: torch.Tensor, mask: torch.Tensor, top_k: int | None = None) -> float:
    ids = torch.nonzero(mask, as_tuple=False).flatten()
    if top_k is not None:
        ids = ids[torch.argsort(kappa[ids].abs(), descending=True)[:top_k]]
    return float(kappa[ids].abs().sum().item()) if ids.numel() else 0.0


def paper_layer_summary(stats: dict[str, torch.Tensor], paper_top_k: int) -> dict[str, Any]:
    delta, beta, kappa = stats["delta"], stats["beta"], stats["kappa"]
    corrupt_mask = side_mask(delta, "corrupt")
    clean_mask = side_mask(delta, "clean")
    suppressor_mask = corrupt_mask & (beta < 0)
    driver_mask = clean_mask & (beta > 0)
    counter_corrupt_toward = corrupt_mask & (beta > 0)
    counter_clean_away = clean_mask & (beta < 0)
    values: dict[str, Any] = {
        "G_kappa": float(kappa.sum().item()),
        "G_abs_kappa": float(kappa.abs().sum().item()),
        "n_features": int(delta.numel()),
        "n_corrupt_higher": int(corrupt_mask.sum().item()),
        "n_clean_higher": int(clean_mask.sum().item()),
        "K_corrupt": mass_for_mask(kappa, corrupt_mask),
        "K_clean": mass_for_mask(kappa, clean_mask),
        "K_corrupt_clean_ratio": None,
        "K_corrupt_topk": mass_for_mask(kappa, corrupt_mask, paper_top_k),
        "K_clean_topk": mass_for_mask(kappa, clean_mask, paper_top_k),
        "K_corrupt_topk_clean_ratio": None,
        "suppressor_aligned_mass": mass_for_mask(kappa, suppressor_mask),
        "driver_aligned_mass": mass_for_mask(kappa, driver_mask),
        "counter_corrupt_toward_mass": mass_for_mask(kappa, counter_corrupt_toward),
        "counter_clean_away_mass": mass_for_mask(kappa, counter_clean_away),
        "n_suppressor_aligned": int(suppressor_mask.sum().item()),
        "n_driver_aligned": int(driver_mask.sum().item()),
        "n_counter_corrupt_toward": int(counter_corrupt_toward.sum().item()),
        "n_counter_clean_away": int(counter_clean_away.sum().item()),
    }
    if values["K_clean"]:
        values["K_corrupt_clean_ratio"] = values["K_corrupt"] / values["K_clean"]
    if values["K_clean_topk"]:
        values["K_corrupt_topk_clean_ratio"] = values["K_corrupt_topk"] / values["K_clean_topk"]
    values["aligned_S_over_E"] = (
        values["suppressor_aligned_mass"] / values["driver_aligned_mass"]
        if values["driver_aligned_mass"]
        else None
    )
    values["side_mass_residual"] = (
        values["K_corrupt"]
        + values["K_clean"]
        - values["G_abs_kappa"]
    )
    return values


def aligned_mask(stats: dict[str, torch.Tensor], category: str) -> torch.Tensor:
    delta, beta = stats["delta"], stats["beta"]
    if category == "suppressor":
        return (delta < 0) & (beta < 0)
    if category == "driver":
        return (delta > 0) & (beta > 0)
    raise ValueError(category)


def feature_row(
    layer: int,
    feature_id: int,
    category: str,
    train_stats: dict[str, torch.Tensor],
    heldout_stats: dict[str, torch.Tensor],
) -> dict[str, Any]:
    return {
        "layer": int(layer),
        "feature_idx": int(feature_id),
        "category": category,
        "train_delta_activation": float(train_stats["delta"][feature_id].item()),
        "heldout_delta_activation": float(heldout_stats["delta"][feature_id].item()),
        "beta_mu": float(train_stats["beta"][feature_id].item()),
        "train_kappa": float(train_stats["kappa"][feature_id].item()),
        "heldout_kappa": float(heldout_stats["kappa"][feature_id].item()),
        "train_abs_kappa": float(train_stats["kappa"][feature_id].abs().item()),
        "heldout_abs_kappa": float(heldout_stats["kappa"][feature_id].abs().item()),
        "train_active_rate_clean": float(train_stats["clean_active_rate"][feature_id].item()),
        "train_active_rate_corrupt": float(train_stats["corrupt_active_rate"][feature_id].item()),
        "heldout_active_rate_clean": float(heldout_stats["clean_active_rate"][feature_id].item()),
        "heldout_active_rate_corrupt": float(heldout_stats["corrupt_active_rate"][feature_id].item()),
    }


def top_aligned_rows(
    layer: int,
    train_stats: dict[str, torch.Tensor],
    heldout_stats: dict[str, torch.Tensor],
    category: str,
    limit: int,
) -> list[dict[str, Any]]:
    ids = torch.nonzero(aligned_mask(train_stats, category), as_tuple=False).flatten()
    order = torch.argsort(train_stats["kappa"][ids].abs(), descending=True)[:limit]
    return [
        feature_row(layer, int(feature_id), category, train_stats, heldout_stats)
        for feature_id in ids[order].tolist()
    ]


def build_lookup(
    layer: int,
    feature_id: int,
    category: str,
    train_stats: dict[str, torch.Tensor],
    heldout_stats: dict[str, torch.Tensor],
) -> dict[str, Any]:
    return feature_row(layer, feature_id, category, train_stats, heldout_stats)


def add_selected_diagnostics(
    rows: list[dict[str, Any]],
    selected_acts: dict[str, dict[int, dict[str, torch.Tensor]]],
    train_pairs: Sequence[base.Pair],
    heldout_pairs: Sequence[base.Pair],
    example_top_k: int,
) -> None:
    by_layer = {
        layer: {int(feature_id): column for column, feature_id in enumerate(payload["ids"].tolist())}
        for layer, payload in selected_acts["train"].items()
    }
    for row in rows:
        layer = int(row["layer"])
        feature_id = int(row["feature_idx"])
        column = by_layer[layer][feature_id]
        beta = float(row["beta_mu"])
        for split_name, pairs in (("train", train_pairs), ("heldout", heldout_pairs)):
            clean = selected_acts[split_name][layer]["clean"][:, column]
            corrupt = selected_acts[split_name][layer]["corrupt"][:, column]
            contribution = (clean - corrupt) * beta
            row[f"{split_name}_pair_sign_consistency"] = float((contribution > 0).float().mean().item())
            examples: list[dict[str, Any]] = []
            for side, values in (("clean", clean), ("corrupt", corrupt)):
                order = torch.argsort(values, descending=True)[:example_top_k]
                for index in order.tolist():
                    pair = pairs[index]
                    examples.append(
                        {
                            "activation": float(values[index].item()),
                            "side": side,
                            "sample_id": pair.sample_id,
                            "verb": pair.clean_verb if side == "clean" else pair.corrupt_verb,
                        }
                    )
            examples.sort(key=lambda item: float(item["activation"]), reverse=True)
            row[f"{split_name}_max_activating_examples"] = json.dumps(examples[:example_top_k], ensure_ascii=False)


def baseline_row(metrics: dict[str, torch.Tensor], index: int, tool_id: int) -> dict[str, Any]:
    return {
        "tool_logit": float(metrics["tool_logit"][index].item()),
        "tool_probability": float(metrics["tool_prob"][index].item()),
        "top1_margin": float(metrics["top1_margin"][index].item()),
        "log_odds": float(metrics["log_odds"][index].item()),
        "tool_rank": int(metrics["rank"][index].item()),
        "tool_top1": int(metrics["top1"][index].item() == tool_id),
    }


def run_feature_ablation(
    model,
    tokenizer,
    pairs: Sequence[base.Pair],
    *,
    group_name: str,
    group: Sequence[dict[str, Any]],
    weights_by_layer: dict[int, dict[str, Any]],
    baseline_metrics: dict[str, torch.Tensor],
    baseline_states: dict[str, torch.Tensor],
    u: torch.Tensor,
    layers,
    decision_layer: int,
    tool_id: int,
    batch_size: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not group:
        raise ValueError(f"{group_name}: empty feature group")
    grouped: dict[int, list[int]] = defaultdict(list)
    for row in group:
        grouped[int(row["layer"])].append(int(row["feature_idx"]))
    device = base.model_device(model)
    rows: list[dict[str, Any]] = []
    pad_token_id = int(tokenizer.pad_token_id)
    for batch, _clean_cpu, corrupt_cpu, attention_mask_cpu in tqdm(
        base.iter_batches(pairs, batch_size, pad_token_id),
        desc=f"Paper ablation {group_name}",
        dynamic_ncols=True,
        leave=False,
    ):
        indices = [item.index for item in batch]
        tokens = corrupt_cpu.to(device)
        attention_mask = attention_mask_cpu.to(device)
        holders: dict[int, torch.Tensor] = {}
        handles = []
        for layer, feature_ids in grouped.items():
            weights = weights_by_layer[layer]
            ids = torch.tensor(feature_ids, dtype=torch.long)
            enc_cpu = weights["W_enc"].index_select(0, ids).contiguous()
            bias_cpu = weights["b_enc"].index_select(0, ids).contiguous()
            dec_cpu = weights["W_dec"].index_select(0, ids).contiguous()
            state: dict[str, torch.Tensor] = {}

            def make_pre_hook(saved_layer: int, saved_state: dict[str, torch.Tensor]):
                def hook(_module, inputs):
                    if not inputs or not isinstance(inputs[0], torch.Tensor):
                        raise TypeError(f"L{saved_layer}: MLP pre-hook did not receive hidden states")
                    saved_state["x"] = inputs[0][:, -1, :].detach()

                return hook

            def make_post_hook(
                saved_layer: int,
                saved_state: dict[str, torch.Tensor],
                enc: torch.Tensor,
                bias: torch.Tensor,
                dec: torch.Tensor,
            ):
                def hook(_module, _inputs, output):
                    if "x" not in saved_state:
                        raise RuntimeError(f"L{saved_layer}: MLP post-hook has no captured input")
                    current = base.tensor_output(output)
                    x = saved_state.pop("x")
                    enc_dev = enc.to(device=current.device, dtype=torch.bfloat16)
                    bias_dev = bias.to(device=current.device, dtype=torch.bfloat16)
                    dec_dev = dec.to(device=current.device, dtype=torch.bfloat16)
                    acts = F.relu(F.linear(x.to(dtype=torch.bfloat16), enc_dev, bias_dev))
                    contribution = acts.float().matmul(dec_dev.float())
                    edited = current.clone()
                    edited[:, -1, :] = edited[:, -1, :] - contribution.to(dtype=edited.dtype)
                    return base.replace_tensor_output(output, edited)

                return hook

            handles.append(layers[layer].mlp.register_forward_pre_hook(make_pre_hook(layer, state)))
            handles.append(layers[layer].mlp.register_forward_hook(make_post_hook(layer, state, enc_cpu, bias_cpu, dec_cpu)))

        gate_holder: dict[str, torch.Tensor] = {}

        def decision_hook(_module, _inputs, output):
            gate_holder["state"] = base.tensor_output(output)[:, -1, :].detach().float().cpu()

        handles.append(layers[decision_layer].register_forward_hook(decision_hook))
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
        gate = gate_holder["state"].matmul(u.float())
        base_gate = baseline_states["corrupt"].index_select(0, torch.tensor(indices)).matmul(u.float())
        for local, item in enumerate(batch):
            base_row = baseline_row(baseline_metrics["corrupt"], item.index, tool_id)
            new_top1 = int(metrics["top1"][local].item() == tool_id)
            rows.append(
                {
                    "group": group_name,
                    "sample_id": item.sample_id,
                    "pair_index": int(item.index),
                    "intervened_tool_logit": float(metrics["tool_logit"][local].item()),
                    "intervened_tool_probability": float(metrics["tool_prob"][local].item()),
                    "intervened_top1_margin": float(metrics["top1_margin"][local].item()),
                    "intervened_log_odds": float(metrics["log_odds"][local].item()),
                    "intervened_tool_rank": int(metrics["rank"][local].item()),
                    "intervened_tool_top1": new_top1,
                    "tool_logit_delta": float(metrics["tool_logit"][local].item()) - base_row["tool_logit"],
                    "tool_probability_delta": float(metrics["tool_prob"][local].item()) - base_row["tool_probability"],
                    "top1_margin_delta": float(metrics["top1_margin"][local].item()) - base_row["top1_margin"],
                    "log_odds_delta": float(metrics["log_odds"][local].item()) - base_row["log_odds"],
                    "gate_score": float(gate[local].item()),
                    "gate_score_delta": float(gate[local].item() - base_gate[local].item()),
                    "baseline_tool_logit": base_row["tool_logit"],
                    "baseline_tool_top1": base_row["tool_top1"],
                    "strict_recovery": int(not base_row["tool_top1"] and new_top1),
                }
            )
        del tokens, attention_mask, outputs, logits, metrics, gate_holder
        base.clear_cuda()

    n = max(len(rows), 1)
    clean_gate = baseline_states["clean"].matmul(u.float())
    corrupt_gate = baseline_states["corrupt"].matmul(u.float())
    progress: list[float] = []
    for row in rows:
        index = int(row["pair_index"])
        denominator = float(clean_gate[index].item() - corrupt_gate[index].item())
        if abs(denominator) > 1e-6:
            progress.append(float(row["gate_score_delta"]) / denominator)
    summary = {
        "group": group_name,
        "n": len(rows),
        "feature_count": len(group),
        "layers": dict(Counter(int(row["layer"]) for row in group)),
        "mean_gate_score_delta": sum(float(row["gate_score_delta"]) for row in rows) / n,
        "mean_gate_progress_toward_clean": sum(progress) / len(progress) if progress else None,
        "mean_tool_logit_delta": sum(float(row["tool_logit_delta"]) for row in rows) / n,
        "mean_tool_probability_delta": sum(float(row["tool_probability_delta"]) for row in rows) / n,
        "mean_top1_margin_delta": sum(float(row["top1_margin_delta"]) for row in rows) / n,
        "mean_log_odds_delta": sum(float(row["log_odds_delta"]) for row in rows) / n,
        "intervened_tool_top1_rate": sum(int(row["intervened_tool_top1"]) for row in rows) / n,
        "strict_recovery_rate": sum(int(row["strict_recovery"]) for row in rows) / n,
    }
    return summary, rows


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    layers_to_analyze = parse_layers(args.layers)
    if args.top_k <= 0 or args.paper_top_k <= 0:
        raise ValueError("top-k values must be positive")
    base.ensure_dir(args.output_root)

    checkpoint_paths = {
        28: args.transcoder_root / "checkpoint_step_0070000.pt",
        29: args.transcoder_root / "checkpoint_step_0065000.pt",
    }
    missing = [layer for layer in layers_to_analyze if layer not in checkpoint_paths]
    if missing:
        raise ValueError(f"No default checkpoint mapping for layers {missing}")

    model, tokenizer = base.load_model(args.model_path, args.dtype, args.device)
    tool_ids = tokenizer.encode(base.TOOL_CALL_TEXT, add_special_tokens=False)
    if len(tool_ids) != 1:
        raise ValueError(f"{base.TOOL_CALL_TEXT!r} is not a single token: {tool_ids}")
    tool_id = int(tool_ids[0])
    text_model = base.resolve_text_model(model)
    layers = text_model.layers
    if args.decision_layer < 0 or args.decision_layer >= len(layers):
        raise ValueError(f"Invalid decision layer L{args.decision_layer}")
    if any(layer >= len(layers) for layer in layers_to_analyze):
        raise ValueError(f"Invalid analyzed layer in {layers_to_analyze}")

    train_pairs = base.load_pairs(args.dataset_root, tokenizer, "train")
    heldout_pairs = base.load_pairs(args.dataset_root, tokenizer, "heldout")
    if len(train_pairs) != 200 or len(heldout_pairs) != 300:
        raise ValueError(f"Expected 200/300 pairs, got {len(train_pairs)}/{len(heldout_pairs)}")

    train = base.capture_dataset(
        model,
        tokenizer,
        train_pairs,
        capture_layers=layers_to_analyze,
        decision_layer=args.decision_layer,
        tool_id=tool_id,
        batch_size=args.batch_size,
        label="train-200",
    )
    heldout = base.capture_dataset(
        model,
        tokenizer,
        heldout_pairs,
        capture_layers=layers_to_analyze,
        decision_layer=args.decision_layer,
        tool_id=tool_id,
        batch_size=args.batch_size,
        label="heldout-300",
    )

    mu_delta = train["states"]["clean"].mean(dim=0) - train["states"]["corrupt"].mean(dim=0)
    mu_norm = float(mu_delta.norm().item())
    if mu_norm <= 0:
        raise RuntimeError("Train L29 clean-minus-corrupt direction has zero norm")
    u = mu_delta / mu_norm

    weights_by_layer: dict[int, dict[str, Any]] = {}
    for layer in layers_to_analyze:
        weights_by_layer[layer] = base.load_transcoder(checkpoint_paths[layer], layer)

    train_stats_by_layer: dict[int, dict[str, torch.Tensor]] = {}
    heldout_stats_by_layer: dict[int, dict[str, torch.Tensor]] = {}
    paper_rows: list[dict[str, Any]] = []
    for layer in layers_to_analyze:
        train_stats = feature_stats(
            train,
            layer,
            weights_by_layer[layer],
            u,
            device=base.model_device(model),
            batch_size=args.feature_batch_size,
        )
        heldout_stats = feature_stats(
            heldout,
            layer,
            weights_by_layer[layer],
            u,
            device=base.model_device(model),
            batch_size=args.feature_batch_size,
        )
        train_stats_by_layer[layer] = train_stats
        heldout_stats_by_layer[layer] = heldout_stats
        for split_name, stats in (("train", train_stats), ("heldout", heldout_stats)):
            paper_rows.append(
                {
                    "layer": layer,
                    "split": split_name,
                    **paper_layer_summary(stats, args.paper_top_k),
                }
            )
        base.clear_cuda()

    suppressors: list[dict[str, Any]] = []
    drivers: list[dict[str, Any]] = []
    for layer in layers_to_analyze:
        suppressors.extend(top_aligned_rows(layer, train_stats_by_layer[layer], heldout_stats_by_layer[layer], "suppressor", args.top_k))
        drivers.extend(top_aligned_rows(layer, train_stats_by_layer[layer], heldout_stats_by_layer[layer], "driver", args.top_k))
    suppressors.sort(key=lambda row: (-float(row["train_abs_kappa"]), int(row["layer"]), int(row["feature_idx"])))
    drivers.sort(key=lambda row: (-float(row["train_abs_kappa"]), int(row["layer"]), int(row["feature_idx"])))
    # Select the top feature family globally over the analyzed layer window.
    suppressors = suppressors[: args.top_k]
    drivers = drivers[: args.top_k]

    reserved = {(int(row["layer"]), int(row["feature_idx"])) for row in suppressors + drivers}
    random_rows: list[dict[str, Any]] = []
    wanted = Counter(int(row["layer"]) for row in suppressors)
    rng = random.Random(args.seed)
    for layer, count in sorted(wanted.items()):
        d_feature = int(weights_by_layer[layer]["d_feature"])
        pool = [feature_id for feature_id in range(d_feature) if (layer, feature_id) not in reserved]
        chosen = rng.sample(pool, count)
        random_rows.extend(
            build_lookup(layer, feature_id, "random_layer_matched", train_stats_by_layer[layer], heldout_stats_by_layer[layer])
            for feature_id in chosen
        )

    top_rows = [*suppressors, *drivers]
    selected_ids_by_layer: dict[int, list[int]] = {layer: [] for layer in layers_to_analyze}
    for row in [*top_rows, *random_rows]:
        selected_ids_by_layer[int(row["layer"])].append(int(row["feature_idx"]))
    for layer in selected_ids_by_layer:
        selected_ids_by_layer[layer] = sorted(set(selected_ids_by_layer[layer]))

    selected_acts: dict[str, dict[int, dict[str, torch.Tensor]]] = {"train": {}, "heldout": {}}
    for layer in layers_to_analyze:
        ids = torch.tensor(selected_ids_by_layer[layer], dtype=torch.long)
        for split_name, payload in (("train", train), ("heldout", heldout)):
            clean = base.collect_feature_acts(
                payload["inputs"]["clean"][layer],
                weights_by_layer[layer]["W_enc"],
                weights_by_layer[layer]["b_enc"],
                device=base.model_device(model),
                batch_size=args.feature_batch_size,
                selected_ids=ids,
            )["selected"]
            corrupt = base.collect_feature_acts(
                payload["inputs"]["corrupt"][layer],
                weights_by_layer[layer]["W_enc"],
                weights_by_layer[layer]["b_enc"],
                device=base.model_device(model),
                batch_size=args.feature_batch_size,
                selected_ids=ids,
            )["selected"]
            selected_acts[split_name][layer] = {"ids": ids, "clean": clean, "corrupt": corrupt}

    top_rows = base.decoder_token_rows(
        top_rows,
        weights_by_layer,
        model,
        tokenizer,
        tool_id=tool_id,
        token_top_k=args.token_top_k,
    )
    add_selected_diagnostics(top_rows, selected_acts, train_pairs, heldout_pairs, args.example_top_k)
    write_csv(args.output_root / "paper_top_features.csv", top_rows)
    write_csv(args.output_root / "random_feature_selection.csv", random_rows)
    write_csv(args.output_root / "paper_k_summary.csv", paper_rows)

    causal_summaries: list[dict[str, Any]] = []
    causal_rows: list[dict[str, Any]] = []
    groups = {
        "paper_suppressor_top": suppressors,
        "aligned_driver_top_control": drivers,
        "random_layer_matched_control": random_rows,
    }
    if not args.skip_causal:
        for group_name, group in groups.items():
            summary, rows = run_feature_ablation(
                model,
                tokenizer,
                heldout_pairs,
                group_name=group_name,
                group=group,
                weights_by_layer=weights_by_layer,
                baseline_metrics=heldout["metrics"],
                baseline_states=heldout["states"],
                u=u,
                layers=layers,
                decision_layer=args.decision_layer,
                tool_id=tool_id,
                batch_size=args.batch_size,
            )
            causal_summaries.append(summary)
            causal_rows.extend(rows)
        write_csv(args.output_root / "paper_causal_summary.csv", causal_summaries)
        write_csv(args.output_root / "paper_causal_per_sample.csv", causal_rows)

    baseline_payload = {
        "train": {side: base.metric_summary(train["metrics"][side], tool_id) for side in ("clean", "corrupt")},
        "heldout": {side: base.metric_summary(heldout["metrics"][side], tool_id) for side in ("clean", "corrupt")},
    }
    run_config = {
        "experiment": "paper_style_transcoder_decomposition_qwen35",
        "model_path": str(args.model_path),
        "dataset_root": str(args.dataset_root),
        "manifest_sha256": base.sha256_file(args.dataset_root / "manifest.jsonl"),
        "transcoder_checkpoints": {str(layer): str(checkpoint_paths[layer]) for layer in layers_to_analyze},
        "transcoder_steps": {str(layer): weights_by_layer[layer]["checkpoint_step"] for layer in layers_to_analyze},
        "layers_analyzed": layers_to_analyze,
        "decision_layer": args.decision_layer,
        "train_pairs": len(train_pairs),
        "heldout_pairs": len(heldout_pairs),
        "vector_fit": "mean L29 decoder-block output clean-minus-corrupt over train-200",
        "feature_activation": "relu(F.linear(MLP pre-hook input, W_enc, b_enc))",
        "paper_kappa": "(mean_clean_activation - mean_corrupt_activation) * (W_dec @ unit_train_mu_delta)",
        "paper_K_corrupt": "sum(abs(kappa) for delta_activation < 0), including both beta signs",
        "paper_K_clean": "sum(abs(kappa) for delta_activation > 0), including both beta signs",
        "paper_topk_scope": "top-k within each layer and activation-side mask; not historical Table 3 global aligned top-k",
        "historical_table3_audit": "audit_table3_k_protocol.py",
        "aligned_suppressor": "delta_activation < 0 and beta_mu < 0",
        "aligned_driver": "delta_activation > 0 and beta_mu > 0",
        "top_k_aligned_selected_on_train": args.top_k,
        "heldout_not_used_for_selection": True,
        "causal_protocol": "ablate selected decoder contributions from actual MLP output on heldout corrupt prompts",
        "causal_groups": list(groups),
        "dtype": args.dtype,
        "seed": args.seed,
    }
    base.write_json(args.output_root / "run_config.json", run_config)
    base.write_json(args.output_root / "baseline_summary.json", baseline_payload)
    torch.save(
        {
            "train_inputs": train["inputs"],
            "train_states": train["states"],
            "heldout_inputs": heldout["inputs"],
            "heldout_states": heldout["states"],
            "decision_layer": args.decision_layer,
            "layers": layers_to_analyze,
        },
        args.output_root / "activation_cache.pt",
    )

    md = [
        "# Qwen3.5-4B paper-style Transcoder analysis",
        "",
        "Train-200 fits the direction and selects features; heldout-300 is used for validation and causal ablation.",
        "",
        "The paper-style K totals group all features by activation side only. The aligned S/E quadrants are reported separately and are not substituted for K.",
        "The displayed K top-20 columns are per-layer activation-side diagnostics; historical Table 3 used aligned global top-20 and is reported by audit_table3_k_protocol.py.",
        "",
        "## Held-out paper K summary",
        "",
        "| layer | K_corrupt | K_clean | Kc/Kclean | Kc top-20 | Kclean top-20 | aligned S | aligned E |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in paper_rows:
        if row["split"] != "heldout":
            continue
        md.append(
            f"| L{row['layer']} | {float(row['K_corrupt']):.4f} | {float(row['K_clean']):.4f} | "
            f"{float(row['K_corrupt_clean_ratio']):.3f} | {float(row['K_corrupt_topk']):.4f} | "
            f"{float(row['K_clean_topk']):.4f} | {float(row['suppressor_aligned_mass']):.4f} | "
            f"{float(row['driver_aligned_mass']):.4f} |"
        )
    if causal_summaries:
        md.extend(
            [
                "",
                "## Held-out corrupt-prompt feature ablation",
                "",
                "| group | features | mean gate Δ | mean margin Δ | mean log-odds Δ | mean tool-logit Δ | recovery |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in causal_summaries:
            md.append(
                f"| {row['group']} | {row['feature_count']} | {float(row['mean_gate_score_delta']):+.4f} | "
                f"{float(row['mean_top1_margin_delta']):+.4f} | {float(row['mean_log_odds_delta']):+.4f} | "
                f"{float(row['mean_tool_logit_delta']):+.4f} | {100.0 * float(row['strict_recovery_rate']):.2f}% |"
            )
    base.write_text(args.output_root / "summary.md", "\n".join(md))
    print(
        json.dumps(
            {
                "output_root": str(args.output_root),
                "heldout_paper_K": [row for row in paper_rows if row["split"] == "heldout"],
                "causal": causal_summaries,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
