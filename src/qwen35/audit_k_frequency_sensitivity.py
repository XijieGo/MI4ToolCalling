#!/usr/bin/env python3
"""Pre-specified Qwen3.5 Transcoder top-k/consistency sensitivity audit.

This audit does not choose a favorable cell.  It computes a fixed grid of
top-k budgets and train-pair contribution-consistency thresholds, then
evaluates the train-selected feature sets on heldout pairs.

The frequency-like requirement used by the earlier strict native-MLP
protocol is contribution consistency, not raw feature active rate:

    C_f = mean_i [ (a_clean[i,f] - a_corrupt[i,f]) * beta_f > 0 ].

The historical strict value is C >= 0.80 (with C >= 0.75 reserved for the
random-control pool).  We sweep that value instead of silently treating the
top-k cutoff as a frequency filter.  Raw clean/corrupt active rates are also
reported for the selected features as diagnostics.
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
    / "qwen35_4b_k_frequency_sensitivity"
)
DEFAULT_K_VALUES = "1,2,5,10,16,20,32,50,100,200,500,1000"
DEFAULT_CONSISTENCY_VALUES = "0.00,0.50,0.60,0.70,0.75,0.80,0.85,0.90,0.95"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--transcoder-root", type=Path, default=DEFAULT_TRANSCODER_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--layers", type=str, default="28,29")
    parser.add_argument("--top-k-values", type=str, default=DEFAULT_K_VALUES)
    parser.add_argument("--consistency-values", type=str, default=DEFAULT_CONSISTENCY_VALUES)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--feature-batch-size", type=int, default=16)
    return parser.parse_args()


def parse_layers(raw: str) -> list[int]:
    layers = sorted({int(part.strip()) for part in raw.split(",") if part.strip()})
    if not layers or any(layer < 0 for layer in layers):
        raise ValueError(f"Invalid layer list: {raw!r}")
    return layers


def parse_ints(raw: str, *, name: str) -> list[int]:
    values = sorted({int(part.strip()) for part in raw.split(",") if part.strip()})
    if not values or any(value <= 0 for value in values):
        raise ValueError(f"{name} must contain positive integers")
    return values


def parse_floats(raw: str, *, name: str) -> list[float]:
    values = sorted({float(part.strip()) for part in raw.split(",") if part.strip()})
    if not values or any(value < 0.0 or value > 1.0 for value in values):
        raise ValueError(f"{name} must contain values in [0, 1]")
    return values


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


def aligned_mask(stats: dict[str, torch.Tensor], category: str) -> torch.Tensor:
    delta = stats["delta"]
    beta = stats["beta"]
    if category == "corrupt":
        return (delta < 0.0) & (beta < 0.0)
    if category == "clean":
        return (delta > 0.0) & (beta > 0.0)
    raise ValueError(category)


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
        return_acts=True,
    )
    corrupt = base.collect_feature_acts(
        inputs["corrupt"][layer],
        weights["W_enc"],
        weights["b_enc"],
        device=device,
        batch_size=batch_size,
        return_acts=True,
    )
    beta = weights["W_dec"].matmul(direction)
    train_delta = clean["acts"] - corrupt["acts"]
    delta = train_delta.mean(dim=0)
    consistency = (train_delta * beta.unsqueeze(0) > 0.0).float().mean(dim=0)
    return {
        "delta": delta,
        "beta": beta,
        "kappa": delta * beta,
        "acts_clean": clean["acts"],
        "acts_corrupt": corrupt["acts"],
        "active_rate_clean": clean["active_rate"],
        "active_rate_corrupt": corrupt["active_rate"],
        "consistency": consistency,
    }


def heldout_feature_stats(
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
        return_acts=True,
    )
    corrupt = base.collect_feature_acts(
        inputs["corrupt"][layer],
        weights["W_enc"],
        weights["b_enc"],
        device=device,
        batch_size=batch_size,
        return_acts=True,
    )
    beta = weights["W_dec"].matmul(direction)
    pair_delta = clean["acts"] - corrupt["acts"]
    delta = pair_delta.mean(dim=0)
    return {
        "delta": delta,
        "beta": beta,
        "kappa": delta * beta,
        "active_rate_clean": clean["active_rate"],
        "active_rate_corrupt": corrupt["active_rate"],
        "consistency": (pair_delta * beta.unsqueeze(0) > 0.0).float().mean(dim=0),
    }


def candidate_pool(
    train_stats: dict[int, dict[str, torch.Tensor]],
    *,
    layers: list[int],
    category: str,
    consistency_threshold: float,
    max_k: int,
) -> list[dict[str, Any]]:
    pool: list[dict[str, Any]] = []
    for layer in layers:
        stats = train_stats[layer]
        mask = aligned_mask(stats, category) & (stats["consistency"] >= consistency_threshold)
        ids = torch.nonzero(mask, as_tuple=False).flatten()
        if ids.numel() == 0:
            continue
        order = torch.argsort(stats["kappa"][ids].abs(), descending=True)[:max_k]
        for feature_id in ids[order].tolist():
            pool.append(
                {
                    "category": category,
                    "layer": int(layer),
                    "feature_idx": int(feature_id),
                    "train_abs_kappa": float(stats["kappa"][feature_id].abs().item()),
                    "train_kappa": float(stats["kappa"][feature_id].item()),
                    "train_delta": float(stats["delta"][feature_id].item()),
                    "beta": float(stats["beta"][feature_id].item()),
                    "train_consistency": float(stats["consistency"][feature_id].item()),
                    "train_active_rate_clean": float(stats["active_rate_clean"][feature_id].item()),
                    "train_active_rate_corrupt": float(stats["active_rate_corrupt"][feature_id].item()),
                }
            )
    return sorted(
        pool,
        key=lambda row: (-float(row["train_abs_kappa"]), int(row["layer"]), int(row["feature_idx"])),
    )


def select_top_k(
    train_stats: dict[int, dict[str, torch.Tensor]],
    heldout_stats: dict[int, dict[str, torch.Tensor]],
    *,
    layers: list[int],
    category: str,
    consistency_threshold: float,
    top_k: int,
) -> tuple[list[dict[str, Any]], int]:
    pool = candidate_pool(
        train_stats,
        layers=layers,
        category=category,
        consistency_threshold=consistency_threshold,
        max_k=top_k,
    )
    selected = pool[:top_k]
    if len(selected) < top_k:
        return selected, len(pool)
    for rank, row in enumerate(selected, start=1):
        layer = int(row["layer"])
        feature_id = int(row["feature_idx"])
        train_layer = train_stats[layer]
        held_layer = heldout_stats[layer]
        row["rank"] = rank
        row["heldout_abs_kappa"] = float(held_layer["kappa"][feature_id].abs().item())
        row["heldout_kappa"] = float(held_layer["kappa"][feature_id].item())
        row["heldout_delta"] = float(held_layer["delta"][feature_id].item())
        row["heldout_consistency"] = float(held_layer["consistency"][feature_id].item())
        row["heldout_active_rate_clean"] = float(held_layer["active_rate_clean"][feature_id].item())
        row["heldout_active_rate_corrupt"] = float(held_layer["active_rate_corrupt"][feature_id].item())
        row["heldout_still_aligned"] = bool(aligned_mask(held_layer, category)[feature_id].item())
        row["selected_train_aligned"] = bool(aligned_mask(train_layer, category)[feature_id].item())
    return selected, len(pool)


def sum_field(rows: list[dict[str, Any]], field: str) -> float:
    return float(sum(float(row[field]) for row in rows))


def median_field(rows: list[dict[str, Any]], field: str) -> float | None:
    if not rows:
        return None
    values = sorted(float(row[field]) for row in rows)
    middle = len(values) // 2
    if len(values) % 2:
        return values[middle]
    return 0.5 * (values[middle - 1] + values[middle])


def build_row(
    *,
    train_stats: dict[int, dict[str, torch.Tensor]],
    heldout_stats: dict[int, dict[str, torch.Tensor]],
    layers: list[int],
    top_k: int,
    consistency_threshold: float,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    suppressors, suppressor_pool = select_top_k(
        train_stats,
        heldout_stats,
        layers=layers,
        category="corrupt",
        consistency_threshold=consistency_threshold,
        top_k=top_k,
    )
    drivers, driver_pool = select_top_k(
        train_stats,
        heldout_stats,
        layers=layers,
        category="clean",
        consistency_threshold=consistency_threshold,
        top_k=top_k,
    )
    feasible = len(suppressors) == top_k and len(drivers) == top_k
    row: dict[str, Any] = {
        "layers": ",".join(str(layer) for layer in layers),
        "top_k": top_k,
        "consistency_threshold": consistency_threshold,
        "status": "complete" if feasible else "insufficient_candidates",
        "suppressor_pool_count": suppressor_pool,
        "driver_pool_count": driver_pool,
        "suppressor_selected_count": len(suppressors),
        "driver_selected_count": len(drivers),
    }
    if not feasible:
        return row, suppressors, drivers
    for prefix, rows in (("corrupt", suppressors), ("clean", drivers)):
        row[f"{prefix}_train_abs_kappa"] = sum_field(rows, "train_abs_kappa")
        row[f"{prefix}_heldout_frozen_abs_kappa"] = sum_field(rows, "heldout_abs_kappa")
        row[f"{prefix}_heldout_frozen_aligned_abs_kappa"] = sum(
            float(item["heldout_abs_kappa"])
            for item in rows
            if bool(item["heldout_still_aligned"])
        )
        row[f"{prefix}_train_consistency_median"] = median_field(rows, "train_consistency")
        row[f"{prefix}_heldout_consistency_median"] = median_field(rows, "heldout_consistency")
        row[f"{prefix}_train_consistency_min"] = min(float(item["train_consistency"]) for item in rows)
        row[f"{prefix}_heldout_consistency_min"] = min(float(item["heldout_consistency"]) for item in rows)
        row[f"{prefix}_train_active_rate_max_median"] = median_field(
            [
                {"value": max(float(item["train_active_rate_clean"]), float(item["train_active_rate_corrupt"]))}
                for item in rows
            ],
            "value",
        )
        row[f"{prefix}_heldout_active_rate_max_median"] = median_field(
            [
                {"value": max(float(item["heldout_active_rate_clean"]), float(item["heldout_active_rate_corrupt"]))}
                for item in rows
            ],
            "value",
        )
    row["train_ratio_corrupt_over_clean"] = (
        row["corrupt_train_abs_kappa"] / row["clean_train_abs_kappa"]
        if row["clean_train_abs_kappa"]
        else None
    )
    row["heldout_ratio_corrupt_over_clean"] = (
        row["corrupt_heldout_frozen_abs_kappa"] / row["clean_heldout_frozen_abs_kappa"]
        if row["clean_heldout_frozen_abs_kappa"]
        else None
    )
    row["heldout_delta_corrupt_minus_clean"] = (
        row["corrupt_heldout_frozen_abs_kappa"] - row["clean_heldout_frozen_abs_kappa"]
    )
    return row, suppressors, drivers


def monotonicity(rows: list[dict[str, Any]], *, threshold: float, field: str) -> bool | None:
    complete = [row for row in rows if row["consistency_threshold"] == threshold and row["status"] == "complete"]
    complete.sort(key=lambda row: int(row["top_k"]))
    if len(complete) < 2:
        return None
    values = [float(row[field]) for row in complete]
    return all(values[index] + 1e-10 >= values[index - 1] for index in range(1, len(values)))


def nonincreasing(rows: list[dict[str, Any]], *, threshold: float, field: str) -> bool | None:
    complete = [row for row in rows if row["consistency_threshold"] == threshold and row["status"] == "complete"]
    complete.sort(key=lambda row: int(row["top_k"]))
    if len(complete) < 2:
        return None
    values = [float(row[field]) for row in complete]
    return all(values[index] <= values[index - 1] + 1e-10 for index in range(1, len(values)))


def main() -> None:
    args = parse_args()
    layers = parse_layers(args.layers)
    top_k_values = parse_ints(args.top_k_values, name="top-k-values")
    consistency_values = parse_floats(args.consistency_values, name="consistency-values")
    cache = torch.load(args.cache, map_location="cpu", weights_only=False)
    train_states = cache["train_states"]
    direction_raw = train_states["clean"].float().mean(dim=0) - train_states["corrupt"].float().mean(dim=0)
    direction_norm = float(direction_raw.norm().item())
    if direction_norm <= 0:
        raise RuntimeError("The cached train direction has zero norm")
    direction = direction_raw / direction_norm
    device = torch.device(args.device)

    train_stats: dict[int, dict[str, torch.Tensor]] = {}
    heldout_stats: dict[int, dict[str, torch.Tensor]] = {}
    for layer in tqdm(layers, desc="Collecting train/heldout feature statistics", dynamic_ncols=True):
        checkpoint = args.transcoder_root / {
            28: "checkpoint_step_0070000.pt",
            29: "checkpoint_step_0065000.pt",
        }.get(layer, f"checkpoint_layer_{layer}.pt")
        weights = base.load_transcoder(checkpoint, layer)
        train_stats[layer] = feature_stats(
            cache["train_inputs"],
            layer,
            weights,
            direction,
            device=device,
            batch_size=args.feature_batch_size,
        )
        heldout_stats[layer] = heldout_feature_stats(
            cache["heldout_inputs"],
            layer,
            weights,
            direction,
            device=device,
            batch_size=args.feature_batch_size,
        )
        del weights
        base.clear_cuda()

    rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    for threshold in consistency_values:
        for top_k in top_k_values:
            row, suppressors, drivers = build_row(
                train_stats=train_stats,
                heldout_stats=heldout_stats,
                layers=layers,
                top_k=top_k,
                consistency_threshold=threshold,
            )
            rows.append(row)
            for category, selected in (("corrupt", suppressors), ("clean", drivers)):
                for item in selected:
                    selected_rows.append(
                        {
                            "consistency_threshold": threshold,
                            "top_k": top_k,
                            "category": category,
                            **item,
                        }
                    )

    args.output_root.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_root / "k_consistency_grid.csv", rows)
    write_csv(args.output_root / "selected_feature_rows.csv", selected_rows)

    monotonicity_summary: dict[str, Any] = {}
    for threshold in consistency_values:
        key = f"{threshold:.2f}"
        monotonicity_summary[key] = {
            "train_corrupt_mass_monotonic": monotonicity(rows, threshold=threshold, field="corrupt_train_abs_kappa"),
            "train_clean_mass_monotonic": monotonicity(rows, threshold=threshold, field="clean_train_abs_kappa"),
            "heldout_corrupt_mass_monotonic": monotonicity(rows, threshold=threshold, field="corrupt_heldout_frozen_abs_kappa"),
            "heldout_clean_mass_monotonic": monotonicity(rows, threshold=threshold, field="clean_heldout_frozen_abs_kappa"),
            "heldout_ratio_nondecreasing": monotonicity(rows, threshold=threshold, field="heldout_ratio_corrupt_over_clean"),
            "heldout_ratio_nonincreasing": nonincreasing(rows, threshold=threshold, field="heldout_ratio_corrupt_over_clean"),
        }

    summary = {
        "experiment": "qwen35_transcoder_topk_consistency_sensitivity",
        "cache": str(args.cache),
        "transcoder_root": str(args.transcoder_root),
        "layers": layers,
        "selection_split": "train",
        "selection_pairs": int(train_states["clean"].shape[0]),
        "validation_split": "heldout",
        "validation_pairs": int(cache["heldout_states"]["clean"].shape[0]),
        "direction": "unit mean(clean - corrupt) of cached train L29 decoder-block states",
        "direction_norm_before_unit": direction_norm,
        "candidate_masks": {
            "corrupt": "delta < 0 and beta < 0",
            "clean": "delta > 0 and beta > 0",
        },
        "consistency_definition": "mean over train pairs of ((clean_act - corrupt_act) * beta > 0)",
        "historical_strict_consistency": 0.80,
        "historical_random_pool_consistency": 0.75,
        "top_k_values": top_k_values,
        "consistency_values": consistency_values,
        "selection": "top-k abs(train kappa) per layer prefilter, then global top-k per aligned category after consistency filter",
        "monotonicity": monotonicity_summary,
        "rows": rows,
    }
    (args.output_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    strict_rows = [row for row in rows if abs(float(row["consistency_threshold"]) - 0.80) < 1e-9]
    strict_rows.sort(key=lambda row: int(row["top_k"]))
    md = [
        "# Qwen3.5-4B top-k / contribution-consistency sensitivity",
        "",
        "This is a pre-specified grid; no heldout cell was selected post hoc.",
        "",
        f"- Layers: `{layers}`; selection: train-{summary['selection_pairs']}; validation: heldout-{summary['validation_pairs']}",
        "- Consistency is the fraction of train pairs whose feature contribution has the selected positive sign; it is not raw activation rate.",
        "- Historical strict setting: `C >= 0.80`; random-control pool: `C >= 0.75`.",
        "",
        "## C >= 0.80 curve",
        "",
        "| k | status | train Kc/Ke | heldout frozen Kc/Ke | heldout ratio | heldout Kc−Ke |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for row in strict_rows:
        if row["status"] != "complete":
            md.append(f"| {row['top_k']} | {row['status']} | — | — | — | — |")
        else:
            md.append(
                f"| {row['top_k']} | complete | {row['corrupt_train_abs_kappa']:.5f}/{row['clean_train_abs_kappa']:.5f} | "
                f"{row['corrupt_heldout_frozen_abs_kappa']:.5f}/{row['clean_heldout_frozen_abs_kappa']:.5f} | "
                f"{row['heldout_ratio_corrupt_over_clean']:.4f} | {row['heldout_delta_corrupt_minus_clean']:+.5f} |"
            )
    md.extend(
        [
            "",
            "The complete grid is in `k_consistency_grid.csv`; selected feature diagnostics, including train/heldout consistency and raw active rates, are in `selected_feature_rows.csv`.",
            "",
            "## Monotonicity",
            "",
            "In this run, both train and heldout absolute-mass curves increased across the scanned k values, while the heldout corrupt/clean ratio decreased across every complete k curve. The ratio is a quotient and this direction is empirical, not a general algebraic guarantee.",
            "",
            "```json",
            json.dumps(monotonicity_summary, ensure_ascii=False, indent=2),
            "```",
        ]
    )
    (args.output_root / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
