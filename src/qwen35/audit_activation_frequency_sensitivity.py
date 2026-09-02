#!/usr/bin/env python3
"""Audit the legacy activation-frequency/selectivity rule separately.

The older Qwen3-8B differential-feature classifier used a different notion
of "frequency" from the native E/S consistency rule: the relevant side had to
be active on at least 5% of pairs, while the opposite side had to satisfy
``active_other <= max(0.10, 0.25 * active_relevant)``.  It also imposed the
same 0.25 selectivity rule on mean activations.

This script applies the activation-frequency/selectivity part as a separately
labelled robustness audit of the Qwen3.5 Transcoder top-k statistic.  The old
classifier also used an AUC threshold; we deliberately do not silently mix
that classifier with the current aligned-quadrant/C-consistency statistic.
The audit therefore keeps the current C threshold and adds only the old
frequency/selectivity constraints.  It does not alter the primary
paper-style K definition or the historical Table 3 statistic.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import torch
from tqdm.auto import tqdm

import audit_k_frequency_sensitivity as sensitivity
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
    / "qwen35_4b_activation_frequency_sensitivity"
)
DEFAULT_K_VALUES = "1,2,5,10,16,20,32,50,100"
DEFAULT_CONSISTENCY_VALUES = "0.00,0.75,0.80"
DEFAULT_FREQUENCY_FLOORS = "0.00,0.01,0.05,0.10"
DEFAULT_INACTIVE_CAPS = "0.05,0.10,0.20,1.00"
DEFAULT_SELECTIVITY_RATIOS = "0.10,0.25,0.50,1.00"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--transcoder-root", type=Path, default=DEFAULT_TRANSCODER_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--layers", type=str, default="28,29")
    parser.add_argument("--top-k-values", type=str, default=DEFAULT_K_VALUES)
    parser.add_argument("--consistency-values", type=str, default=DEFAULT_CONSISTENCY_VALUES)
    parser.add_argument("--frequency-floors", type=str, default=DEFAULT_FREQUENCY_FLOORS)
    parser.add_argument("--inactive-caps", type=str, default=DEFAULT_INACTIVE_CAPS)
    parser.add_argument("--selectivity-ratios", type=str, default=DEFAULT_SELECTIVITY_RATIOS)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--feature-batch-size", type=int, default=16)
    return parser.parse_args()


def parse_floats(raw: str, *, name: str, positive: bool = False) -> list[float]:
    values = sorted({float(part.strip()) for part in raw.split(",") if part.strip()})
    lower = 0.0 if not positive else 1e-12
    if not values or any(value < lower or value > 1.0 for value in values):
        raise ValueError(f"{name} must contain values in [{lower}, 1]")
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


def frequency_mask(
    stats: dict[str, torch.Tensor],
    category: str,
    *,
    floor: float,
    inactive_cap: float,
    selectivity_ratio: float,
    include_mean_rule: bool,
) -> torch.Tensor:
    active_clean = stats["active_rate_clean"]
    active_corrupt = stats["active_rate_corrupt"]
    mean_clean = stats["acts_clean"].mean(dim=0)
    mean_corrupt = stats["acts_corrupt"].mean(dim=0)
    if category == "corrupt":
        relevant_active, other_active = active_corrupt, active_clean
        relevant_mean, other_mean = mean_corrupt, mean_clean
    elif category == "clean":
        relevant_active, other_active = active_clean, active_corrupt
        relevant_mean, other_mean = mean_clean, mean_corrupt
    else:
        raise ValueError(category)

    mask = relevant_active >= floor
    mask &= other_active <= torch.maximum(
        torch.full_like(relevant_active, inactive_cap),
        selectivity_ratio * relevant_active,
    )
    if include_mean_rule:
        epsilon = torch.full_like(relevant_mean, 1e-6)
        mask &= other_mean <= torch.maximum(epsilon, selectivity_ratio * torch.maximum(relevant_mean, epsilon))
    return mask


def candidate_pool(
    train_stats: dict[int, dict[str, torch.Tensor]],
    *,
    layers: list[int],
    category: str,
    consistency_threshold: float,
    top_k: int,
    floor: float,
    inactive_cap: float,
    selectivity_ratio: float,
    include_mean_rule: bool,
) -> list[dict[str, Any]]:
    pool: list[dict[str, Any]] = []
    for layer in layers:
        stats = train_stats[layer]
        mask = sensitivity.aligned_mask(stats, category)
        mask &= stats["consistency"] >= consistency_threshold
        mask &= frequency_mask(
            stats,
            category,
            floor=floor,
            inactive_cap=inactive_cap,
            selectivity_ratio=selectivity_ratio,
            include_mean_rule=include_mean_rule,
        )
        ids = torch.nonzero(mask, as_tuple=False).flatten()
        if ids.numel() == 0:
            continue
        order = torch.argsort(stats["kappa"][ids].abs(), descending=True)[:top_k]
        for feature_id in ids[order].tolist():
            pool.append(
                {
                    "category": category,
                    "layer": int(layer),
                    "feature_idx": int(feature_id),
                    "train_abs_kappa": float(stats["kappa"][feature_id].abs().item()),
                    "train_consistency": float(stats["consistency"][feature_id].item()),
                    "train_active_rate_clean": float(stats["active_rate_clean"][feature_id].item()),
                    "train_active_rate_corrupt": float(stats["active_rate_corrupt"][feature_id].item()),
                }
            )
    return sorted(
        pool,
        key=lambda row: (-float(row["train_abs_kappa"]), int(row["layer"]), int(row["feature_idx"])),
    )


def select(
    train_stats: dict[int, dict[str, torch.Tensor]],
    heldout_stats: dict[int, dict[str, torch.Tensor]],
    *,
    layers: list[int],
    category: str,
    consistency_threshold: float,
    top_k: int,
    floor: float,
    inactive_cap: float,
    selectivity_ratio: float,
    include_mean_rule: bool,
) -> tuple[list[dict[str, Any]], int]:
    pool = candidate_pool(
        train_stats,
        layers=layers,
        category=category,
        consistency_threshold=consistency_threshold,
        top_k=top_k,
        floor=floor,
        inactive_cap=inactive_cap,
        selectivity_ratio=selectivity_ratio,
        include_mean_rule=include_mean_rule,
    )
    selected = pool[:top_k]
    for row in selected:
        layer, feature_id = int(row["layer"]), int(row["feature_idx"])
        stats = heldout_stats[layer]
        row["heldout_abs_kappa"] = float(stats["kappa"][feature_id].abs().item())
        row["heldout_active_rate_clean"] = float(stats["active_rate_clean"][feature_id].item())
        row["heldout_active_rate_corrupt"] = float(stats["active_rate_corrupt"][feature_id].item())
        row["heldout_still_aligned"] = bool(sensitivity.aligned_mask(stats, category)[feature_id].item())
    return selected, len(pool)


def median(values: list[float]) -> float | None:
    if not values:
        return None
    values = sorted(values)
    middle = len(values) // 2
    if len(values) % 2:
        return values[middle]
    return 0.5 * (values[middle - 1] + values[middle])


def build_row(
    train_stats: dict[int, dict[str, torch.Tensor]],
    heldout_stats: dict[int, dict[str, torch.Tensor]],
    *,
    layers: list[int],
    top_k: int,
    consistency_threshold: float,
    floor: float,
    inactive_cap: float,
    selectivity_ratio: float,
    include_mean_rule: bool,
) -> dict[str, Any]:
    suppressors, suppressor_pool = select(
        train_stats,
        heldout_stats,
        layers=layers,
        category="corrupt",
        consistency_threshold=consistency_threshold,
        top_k=top_k,
        floor=floor,
        inactive_cap=inactive_cap,
        selectivity_ratio=selectivity_ratio,
        include_mean_rule=include_mean_rule,
    )
    drivers, driver_pool = select(
        train_stats,
        heldout_stats,
        layers=layers,
        category="clean",
        consistency_threshold=consistency_threshold,
        top_k=top_k,
        floor=floor,
        inactive_cap=inactive_cap,
        selectivity_ratio=selectivity_ratio,
        include_mean_rule=include_mean_rule,
    )
    complete = len(suppressors) == top_k and len(drivers) == top_k
    row: dict[str, Any] = {
        "layers": ",".join(str(layer) for layer in layers),
        "frequency_rule": "active_plus_mean" if include_mean_rule else "active_only",
        "consistency_threshold": consistency_threshold,
        "frequency_floor": floor,
        "inactive_rate_cap": inactive_cap,
        "selectivity_ratio": selectivity_ratio,
        "top_k": top_k,
        "status": "complete" if complete else "insufficient_candidates",
        "suppressor_pool_count": suppressor_pool,
        "driver_pool_count": driver_pool,
        "suppressor_selected_count": len(suppressors),
        "driver_selected_count": len(drivers),
    }
    if not complete:
        return row
    for prefix, selected, category in (
        ("corrupt", suppressors, "corrupt"),
        ("clean", drivers, "clean"),
    ):
        row[f"{prefix}_train_abs_kappa"] = sum(float(item["train_abs_kappa"]) for item in selected)
        row[f"{prefix}_heldout_abs_kappa"] = sum(float(item["heldout_abs_kappa"]) for item in selected)
        row[f"{prefix}_train_consistency_median"] = median([float(item["train_consistency"]) for item in selected])
        relevant_train = [
            float(item["train_active_rate_corrupt"] if category == "corrupt" else item["train_active_rate_clean"])
            for item in selected
        ]
        relevant_heldout = [
            float(item["heldout_active_rate_corrupt"] if category == "corrupt" else item["heldout_active_rate_clean"])
            for item in selected
        ]
        row[f"{prefix}_train_relevant_active_min"] = min(relevant_train)
        row[f"{prefix}_train_relevant_active_median"] = median(relevant_train)
        row[f"{prefix}_heldout_relevant_active_min"] = min(relevant_heldout)
        row[f"{prefix}_heldout_relevant_active_median"] = median(relevant_heldout)
    row["train_ratio_corrupt_over_clean"] = row["corrupt_train_abs_kappa"] / row["clean_train_abs_kappa"]
    row["heldout_ratio_corrupt_over_clean"] = row["corrupt_heldout_abs_kappa"] / row["clean_heldout_abs_kappa"]
    row["heldout_delta_corrupt_minus_clean"] = row["corrupt_heldout_abs_kappa"] - row["clean_heldout_abs_kappa"]
    return row


def main() -> None:
    args = parse_args()
    layers = sensitivity.parse_layers(args.layers)
    top_k_values = sensitivity.parse_ints(args.top_k_values, name="top-k-values")
    consistency_values = parse_floats(args.consistency_values, name="consistency-values")
    floors = parse_floats(args.frequency_floors, name="frequency-floors")
    caps = parse_floats(args.inactive_caps, name="inactive-caps")
    ratios = parse_floats(args.selectivity_ratios, name="selectivity-ratios")

    cache = torch.load(args.cache, map_location="cpu", weights_only=False)
    train_states = cache["train_states"]
    direction_raw = train_states["clean"].float().mean(dim=0) - train_states["corrupt"].float().mean(dim=0)
    direction_norm = float(direction_raw.norm().item())
    direction = direction_raw / max(direction_norm, 1e-12)
    device = torch.device(args.device)
    train_stats: dict[int, dict[str, torch.Tensor]] = {}
    heldout_stats: dict[int, dict[str, torch.Tensor]] = {}
    for layer in tqdm(layers, desc="Collecting train/heldout feature statistics", dynamic_ncols=True):
        checkpoint = args.transcoder_root / {
            28: "checkpoint_step_0070000.pt",
            29: "checkpoint_step_0065000.pt",
        }.get(layer, f"checkpoint_layer_{layer}.pt")
        weights = base.load_transcoder(checkpoint, layer)
        train_stats[layer] = sensitivity.feature_stats(
            cache["train_inputs"], layer, weights, direction, device=device, batch_size=args.feature_batch_size
        )
        heldout_stats[layer] = sensitivity.heldout_feature_stats(
            cache["heldout_inputs"], layer, weights, direction, device=device, batch_size=args.feature_batch_size
        )
        del weights
        base.clear_cuda()

    rows: list[dict[str, Any]] = []
    for include_mean_rule in (False, True):
        for floor in floors:
            for cap in caps:
                for ratio in ratios:
                    for threshold in consistency_values:
                        for top_k in top_k_values:
                            rows.append(
                                build_row(
                                    train_stats,
                                    heldout_stats,
                                    layers=layers,
                                    top_k=top_k,
                                    consistency_threshold=threshold,
                                    floor=floor,
                                    inactive_cap=cap,
                                    selectivity_ratio=ratio,
                                    include_mean_rule=include_mean_rule,
                                )
                            )

    args.output_root.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_root / "activation_frequency_grid.csv", rows)
    summary = {
        "experiment": "qwen35_transcoder_activation_frequency_sensitivity",
        "cache": str(args.cache),
        "transcoder_root": str(args.transcoder_root),
        "layers": layers,
        "selection_split": "train",
        "validation_split": "heldout",
        "direction": "unit mean(clean - corrupt) of cached train L29 decoder-block states",
        "legacy_frequency_constraints": {
            "minimum_relevant_active_rate": 0.05,
            "inactive_rate_cap": 0.10,
            "selectivity_ratio": 0.25,
            "mean_activation_rule": True,
        },
        "not_combined_from_legacy_classifier": "AUC >= 0.80; current audit keeps contribution consistency C instead",
        "top_k_values": top_k_values,
        "consistency_values": consistency_values,
        "frequency_floors": floors,
        "inactive_rate_caps": caps,
        "selectivity_ratios": ratios,
        "rows": rows,
    }
    (args.output_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    historical = [
        row
        for row in rows
        if row["frequency_rule"] == "active_plus_mean"
        and abs(float(row["consistency_threshold"]) - 0.80) < 1e-9
        and abs(float(row["frequency_floor"]) - 0.05) < 1e-9
        and abs(float(row["inactive_rate_cap"]) - 0.10) < 1e-9
        and abs(float(row["selectivity_ratio"]) - 0.25) < 1e-9
    ]
    historical.sort(key=lambda row: int(row["top_k"]))
    md = [
        "# Qwen3.5-4B activation-frequency/selectivity sensitivity",
        "",
        "This is a separate robustness audit; it is not substituted for the paper K or historical Table 3 definition.",
        "",
        f"- Layers: `{layers}`; selection: train-{int(cache['train_states']['clean'].shape[0])}; validation: heldout-{int(cache['heldout_states']['clean'].shape[0])}",
        "- Legacy rule: relevant active rate >= 0.05; opposite active rate <= max(0.10, 0.25 x relevant); the active-plus-mean mode applies the analogous mean-activation rule.",
        "- All cells were evaluated on a fixed grid; no heldout cell was selected post hoc.",
        "",
        "## Legacy frequency/selectivity constraints + current C >= 0.80",
        "",
        "| k | mode/status | train Kc/Ke | heldout frozen Kc/Ke | heldout ratio | heldout Kc−Ke | S/E train relevant active min |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in historical:
        if row["status"] != "complete":
            md.append(f"| {row['top_k']} | {row['frequency_rule']}/{row['status']} | — | — | — | — | — |")
        else:
            md.append(
                f"| {row['top_k']} | {row['frequency_rule']}/complete | "
                f"{row['corrupt_train_abs_kappa']:.5f}/{row['clean_train_abs_kappa']:.5f} | "
                f"{row['corrupt_heldout_abs_kappa']:.5f}/{row['clean_heldout_abs_kappa']:.5f} | "
                f"{row['heldout_ratio_corrupt_over_clean']:.4f} | {row['heldout_delta_corrupt_minus_clean']:+.5f} | "
                f"{row['corrupt_train_relevant_active_min']:.3f}/{row['clean_train_relevant_active_min']:.3f} |"
            )
    md.extend(
        [
            "",
            "The full frequency/consistency grid is in `activation_frequency_grid.csv`; `summary.json` records every parameter cell.",
        ]
    )
    (args.output_root / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps({"output_root": str(args.output_root), "rows": len(rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
