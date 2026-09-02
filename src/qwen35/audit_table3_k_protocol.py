#!/usr/bin/env python3
"""Audit the historical cross-scale Transcoder K protocol on Qwen3.5.

The historical cross-scale table is not the same statistic as the paper's
activation-side K summary.  It uses the train split, keeps only the two
aligned sign quadrants, preselects the largest 20 rows per layer, and then
keeps the global top 20 rows per quadrant over the formation window.  This
script applies that protocol to the cached Qwen3.5 train-200/heldout-300
inputs, while also reporting frozen-selection held-out values.

Qwen3.5 currently has only L28 and L29 Transcoder checkpoints.  Therefore the
default L28/L29 run is an available-checkpoint audit, not the exact four-layer
pre-commit formation window used by the Qwen3-8B cross-scale row.  Use
``--layers 28`` for the strict pre-decision subset when L29 is the decision
layer.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import torch
from tqdm.auto import tqdm

import run_transcoder_feature_analysis as base
from path_defaults import QWEN35_4B_TRANSCODER_PATH, REBUTTAL_ROOT


DEFAULT_CACHE = (
    REBUTTAL_ROOT
    / "rebuttal"
    / "09_qwen35_transcoder_feature_analysis"
    / "qwen35_4b_paper_style"
    / "activation_cache.pt"
)
DEFAULT_TRANSCODER_ROOT = QWEN35_4B_TRANSCODER_PATH
DEFAULT_OUTPUT_ROOT = (
    REBUTTAL_ROOT
    / "rebuttal"
    / "09_qwen35_transcoder_feature_analysis"
    / "qwen35_4b_table3_top20_audit"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--transcoder-root", type=Path, default=DEFAULT_TRANSCODER_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--layers", type=str, default="28,29")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--feature-batch-size", type=int, default=16)
    return parser.parse_args()


def parse_layers(raw: str) -> list[int]:
    layers = sorted({int(part.strip()) for part in raw.split(",") if part.strip()})
    if not layers or any(layer < 0 for layer in layers):
        raise ValueError(f"Invalid layer list: {raw!r}")
    return layers


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


def feature_stats(
    inputs: dict[str, dict[int, torch.Tensor]],
    layer: int,
    weights: dict[str, torch.Tensor],
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
    delta = clean["mean"] - corrupt["mean"]
    beta = weights["W_dec"].matmul(direction)
    kappa = delta * beta
    return {
        "delta": delta,
        "beta": beta,
        "kappa": kappa,
    }


def aligned_mask(stats: dict[str, torch.Tensor], category: str) -> torch.Tensor:
    delta = stats["delta"]
    beta = stats["beta"]
    if category == "corrupt":
        return (delta < 0) & (beta < 0)
    if category == "clean":
        return (delta > 0) & (beta > 0)
    raise ValueError(category)


def side_mask(stats: dict[str, torch.Tensor], category: str) -> torch.Tensor:
    delta = stats["delta"]
    if category == "corrupt":
        return delta < 0
    if category == "clean":
        return delta > 0
    raise ValueError(category)


def mass(stats: dict[str, torch.Tensor], mask: torch.Tensor) -> float:
    values = stats["kappa"][mask]
    return float(values.abs().sum().item()) if values.numel() else 0.0


def candidate_rows(
    layer: int,
    stats: dict[str, torch.Tensor],
    category: str,
    limit: int,
    *,
    selection_split: str,
    validation_stats: dict[str, torch.Tensor] | None,
) -> list[dict[str, Any]]:
    ids = torch.nonzero(aligned_mask(stats, category), as_tuple=False).flatten()
    order = torch.argsort(stats["kappa"][ids].abs(), descending=True)[:limit]
    rows: list[dict[str, Any]] = []
    for feature_id in ids[order].tolist():
        row: dict[str, Any] = {
            "category": category,
            "layer": int(layer),
            "feature_idx": int(feature_id),
            "selection_split": selection_split,
            "selection_delta": float(stats["delta"][feature_id].item()),
            "beta_mu": float(stats["beta"][feature_id].item()),
            "selection_kappa": float(stats["kappa"][feature_id].item()),
            "selection_abs_kappa": float(stats["kappa"][feature_id].abs().item()),
        }
        if validation_stats is not None:
            row.update(
                {
                    "heldout_delta": float(validation_stats["delta"][feature_id].item()),
                    "heldout_kappa": float(validation_stats["kappa"][feature_id].item()),
                    "heldout_abs_kappa": float(validation_stats["kappa"][feature_id].abs().item()),
                    "heldout_still_aligned": bool(aligned_mask(validation_stats, category)[feature_id].item()),
                }
            )
        rows.append(row)
    return rows


def select_global_top(rows: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
    rows = sorted(
        rows,
        key=lambda row: (
            -float(row["selection_abs_kappa"]),
            int(row["layer"]),
            int(row["feature_idx"]),
        ),
    )[:top_k]
    for rank, row in enumerate(rows, start=1):
        row["global_selection_rank"] = rank
    return rows


def selected_sum(rows: list[dict[str, Any]], field: str) -> float:
    return float(sum(float(row[field]) for row in rows))


def selected_aligned_sum(rows: list[dict[str, Any]]) -> float:
    return float(
        sum(float(row["heldout_abs_kappa"]) for row in rows if bool(row["heldout_still_aligned"])))


def split_summary(
    stats_by_layer: dict[int, dict[str, torch.Tensor]],
    layers: list[int],
    *,
    split: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for layer in layers:
        stats = stats_by_layer[layer]
        rows.append(
            {
                "split": split,
                "layer": layer,
                "n_features": int(stats["delta"].numel()),
                "all_side_K_corrupt": mass(stats, side_mask(stats, "corrupt")),
                "all_side_K_clean": mass(stats, side_mask(stats, "clean")),
                "aligned_S_corrupt": mass(stats, aligned_mask(stats, "corrupt")),
                "aligned_E_clean": mass(stats, aligned_mask(stats, "clean")),
                "n_aligned_corrupt": int(aligned_mask(stats, "corrupt").sum().item()),
                "n_aligned_clean": int(aligned_mask(stats, "clean").sum().item()),
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    if args.top_k <= 0:
        raise ValueError("--top-k must be positive")
    layers = parse_layers(args.layers)
    cache = torch.load(args.cache, map_location="cpu", weights_only=False)
    train_states = cache["train_states"]
    direction_raw = train_states["clean"].float().mean(dim=0) - train_states["corrupt"].float().mean(dim=0)
    direction_norm = float(direction_raw.norm().item())
    if direction_norm <= 0:
        raise RuntimeError("The cached train direction has zero norm")
    direction = direction_raw / direction_norm
    device = torch.device(args.device)

    train_stats_by_layer: dict[int, dict[str, torch.Tensor]] = {}
    heldout_stats_by_layer: dict[int, dict[str, torch.Tensor]] = {}
    for layer in tqdm(layers, desc="Scoring cached train/heldout inputs", dynamic_ncols=True):
        checkpoint = args.transcoder_root / {
            28: "checkpoint_step_0070000.pt",
            29: "checkpoint_step_0065000.pt",
        }.get(layer, f"checkpoint_layer_{layer}.pt")
        weights = base.load_transcoder(checkpoint, layer)
        train_stats_by_layer[layer] = feature_stats(
            cache["train_inputs"],
            layer,
            weights,
            direction,
            device=device,
            batch_size=args.feature_batch_size,
        )
        heldout_stats_by_layer[layer] = feature_stats(
            cache["heldout_inputs"],
            layer,
            weights,
            direction,
            device=device,
            batch_size=args.feature_batch_size,
        )
        del weights
        base.clear_cuda()

    train_candidates: dict[str, list[dict[str, Any]]] = {}
    heldout_recomputed: dict[str, list[dict[str, Any]]] = {}
    selected_rows: list[dict[str, Any]] = []
    for category in ("corrupt", "clean"):
        train_pool: list[dict[str, Any]] = []
        heldout_pool: list[dict[str, Any]] = []
        for layer in layers:
            train_pool.extend(
                candidate_rows(
                    layer,
                    train_stats_by_layer[layer],
                    category,
                    args.top_k,
                    selection_split="train",
                    validation_stats=heldout_stats_by_layer[layer],
                )
            )
            heldout_pool.extend(
                candidate_rows(
                    layer,
                    heldout_stats_by_layer[layer],
                    category,
                    args.top_k,
                    selection_split="heldout_recomputed",
                    validation_stats=None,
                )
            )
        selected = select_global_top(train_pool, args.top_k)
        recomputed = select_global_top(heldout_pool, args.top_k)
        train_candidates[category] = selected
        heldout_recomputed[category] = recomputed
        selected_rows.extend(selected)

    write_csv(args.output_root / "table3_top20_frozen_train_selection.csv", selected_rows)
    write_csv(
        args.output_root / "table3_top20_heldout_recomputed.csv",
        [row for category in ("corrupt", "clean") for row in heldout_recomputed[category]],
    )
    split_rows = split_summary(train_stats_by_layer, layers, split="train") + split_summary(
        heldout_stats_by_layer, layers, split="heldout"
    )
    write_csv(args.output_root / "per_layer_protocol_audit.csv", split_rows)

    summary: dict[str, Any] = {
        "experiment": "qwen35_table3_top20_protocol_audit",
        "cache": str(args.cache),
        "transcoder_root": str(args.transcoder_root),
        "layers": layers,
        "window_note": (
            "Available-checkpoint audit over L28/L29; not the exact four-layer pre-commit window. "
            "For the strict pre-decision subset with L29 as decision layer, use --layers 28."
            if 29 in layers
            else "Strict pre-decision subset using the available L28 checkpoint."
        ),
        "top_k": args.top_k,
        "selection_split": "train",
        "selection_pairs": int(train_states["clean"].shape[0]),
        "validation_split": "heldout",
        "validation_pairs": int(cache["heldout_states"]["clean"].shape[0]),
        "direction": "unit mean(clean - corrupt) of cached train L29 decoder-block states",
        "direction_norm_before_unit": direction_norm,
        "candidate_mask": {
            "corrupt": "delta_activation < 0 and beta_mu < 0",
            "clean": "delta_activation > 0 and beta_mu > 0",
        },
        "selection": "top-20 abs(kappa) per layer prefilter, then global top-20 per aligned category",
        "train_top20": {},
        "heldout_recomputed_top20": {},
        "per_layer": split_rows,
    }
    for category, label in (("corrupt", "K_corrupt"), ("clean", "K_clean")):
        selected = train_candidates[category]
        recomputed = heldout_recomputed[category]
        summary["train_top20"][label] = selected_sum(selected, "selection_abs_kappa")
        summary["heldout_recomputed_top20"][label] = selected_sum(recomputed, "selection_abs_kappa")
        summary["train_top20"][f"{label}_n"] = len(selected)
        summary["heldout_recomputed_top20"][f"{label}_n"] = len(recomputed)
        summary["train_top20"][f"{label}_layers"] = {
            str(layer): sum(1 for row in selected if int(row["layer"]) == layer)
            for layer in layers
        }
        summary["heldout_recomputed_top20"][f"{label}_layers"] = {
            str(layer): sum(1 for row in recomputed if int(row["layer"]) == layer)
            for layer in layers
        }
        summary["train_top20"][f"{label}_frozen_heldout_abs"] = selected_sum(selected, "heldout_abs_kappa")
        summary["train_top20"][f"{label}_frozen_heldout_aligned_abs"] = selected_aligned_sum(selected)

    summary["train_top20"]["ratio"] = (
        summary["train_top20"]["K_corrupt"] / summary["train_top20"]["K_clean"]
        if summary["train_top20"]["K_clean"]
        else None
    )
    summary["heldout_recomputed_top20"]["ratio"] = (
        summary["heldout_recomputed_top20"]["K_corrupt"]
        / summary["heldout_recomputed_top20"]["K_clean"]
        if summary["heldout_recomputed_top20"]["K_clean"]
        else None
    )

    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    train = summary["train_top20"]
    heldout = summary["heldout_recomputed_top20"]
    md = [
        "# Qwen3.5-4B historical Table 3 K-protocol audit",
        "",
        "This is the historical cross-scale protocol, not the paper's all-feature activation-side K summary.",
        "",
        f"- Selection: train-{summary['selection_pairs']} pairs; validation: heldout-{summary['validation_pairs']} pairs",
        f"- Layers: `{layers}`",
        f"- Window note: {summary['window_note']}",
        f"- Candidate masks: corrupt `{summary['candidate_mask']['corrupt']}`; clean `{summary['candidate_mask']['clean']}`",
        f"- Selection: `{summary['selection']}`",
        "",
        "| evaluation | K_corrupt | K_clean | ratio |",
        "|---|---:|---:|---:|",
        f"| train selection | {train['K_corrupt']:.6f} | {train['K_clean']:.6f} | {train['ratio']:.4f} |",
        f"| heldout recomputed top-20 | {heldout['K_corrupt']:.6f} | {heldout['K_clean']:.6f} | {heldout['ratio']:.4f} |",
        "",
        "Frozen train-selected features evaluated on heldout:",
        "",
        "| category | train-selected abs mass on heldout | still-aligned heldout mass |",
        "|---|---:|---:|",
        f"| corrupt | {train['K_corrupt_frozen_heldout_abs']:.6f} | {train['K_corrupt_frozen_heldout_aligned_abs']:.6f} |",
        f"| clean | {train['K_clean_frozen_heldout_abs']:.6f} | {train['K_clean_frozen_heldout_aligned_abs']:.6f} |",
    ]
    (args.output_root / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
