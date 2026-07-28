#!/usr/bin/env python3
"""Causally test a shared cross-domain direction at a fixed intervention site.

For a completed model-specific 4x4 run, this script reuses its exact native
mean-difference vectors and global uncentered-SVD direction.  On independent
held-out pairs it compares, per target domain:

* its native vector;
* the shared unit direction at the target-native L2 norm and 1.5/2/3 times it;
* the literal projection of the native vector onto the shared direction;
* the residual after that projection; and
* an equal-norm random direction orthogonal to the shared direction.

The "projection" treatment is never renormalized.  Thus subtracting it from a
clean activation is exactly the requested necessity intervention: it leaves
the native residual in place.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch
from tqdm.auto import tqdm


THIS_DIR = Path(__file__).resolve().parent
SHARED_DIR = THIS_DIR.parent / "shared"
if str(SHARED_DIR) not in sys.path:
    sys.path.insert(0, str(SHARED_DIR))

from multiscale_common import (  # noqa: E402
    build_pair_batches,
    clear_cuda,
    ensure_dir,
    load_model_and_tokenizer,
    load_sample_pairs,
    set_seed,
    tool_stats,
    write_csv,
    write_json,
)
from run_cross_scale_fix_gate_and_patch import hook_name, make_last_token_add_hook  # noqa: E402
from run_full_transfer_matrix import DEFAULT_DOMAINS, validate_pairs  # noqa: E402


SHARED_MULTIPLIERS: tuple[tuple[str, float], ...] = (
    ("shared_target_norm_x1p0", 1.0),
    ("shared_target_norm_x1p5", 1.5),
    ("shared_target_norm_x2p0", 2.0),
    ("shared_target_norm_x3p0", 3.0),
)
CONDITION_ORDER = (
    "native_full",
    *(name for name, _ in SHARED_MULTIPLIERS),
    "native_projection_on_shared",
    "native_residual_after_shared",
    "random_orthogonal_target_norm",
)


@dataclass(frozen=True)
class Treatment:
    condition: str
    vector: torch.Tensor
    vector_family: str
    multiplier: float | None
    target_domain: str
    target_native_l2_norm: float
    projection_coefficient: float | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Causal ablation of a global shared SVD direction.")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--model-label", type=str, required=True)
    parser.add_argument("--dataset-view-root", type=Path, required=True)
    parser.add_argument(
        "--native-vector-root",
        type=Path,
        required=True,
        help="Completed model root containing native_vectors/ and pca_common_direction_L24_pre.pt.",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--target-domains", nargs="+", choices=DEFAULT_DOMAINS, default=list(DEFAULT_DOMAINS))
    parser.add_argument("--layer", type=int, default=24)
    parser.add_argument("--hook-kind", choices=("pre",), default="pre")
    parser.add_argument("--eval-pairs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def unique_domains(domains: Iterable[str]) -> tuple[str, ...]:
    result = tuple(str(domain).upper() for domain in domains)
    if not result or len(set(result)) != len(result):
        raise ValueError(f"Invalid target domains: {result}")
    if set(result) - set(DEFAULT_DOMAINS):
        raise ValueError(f"Unsupported target domains: {result}")
    return result


def load_vector_bundle(
    root: Path, *, layer: int, d_model: int
) -> tuple[dict[str, torch.Tensor], torch.Tensor, dict[str, Any]]:
    vectors: dict[str, torch.Tensor] = {}
    vector_dir = root / "native_vectors"
    for domain in DEFAULT_DOMAINS:
        path = vector_dir / f"{domain}_L{layer}_pre.pt"
        if not path.is_file():
            raise FileNotFoundError(f"Missing native vector: {path}")
        bundle = torch.load(path, map_location="cpu", weights_only=False)
        vector = bundle["mean_diff"].detach().cpu().float().view(-1)
        if vector.numel() != d_model or not bool(torch.isfinite(vector).all()) or float(vector.norm().item()) <= 0.0:
            raise ValueError(f"Invalid native vector for {domain}: {path}")
        vectors[domain] = vector.contiguous()

    pca_path = root / f"pca_common_direction_L{layer}_pre.pt"
    if not pca_path.is_file():
        raise FileNotFoundError(f"Missing stored common direction: {pca_path}")
    pca_tensors = torch.load(pca_path, map_location="cpu", weights_only=False)
    common = pca_tensors["common_direction_unit"].detach().cpu().float().view(-1)
    if common.numel() != d_model or not bool(torch.isfinite(common).all()) or float(common.norm().item()) <= 0.0:
        raise ValueError(f"Invalid stored common direction: {pca_path}")
    common = (common / common.norm()).contiguous()

    summary_path = root / "pca_common_direction.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"Missing PCA summary: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    return vectors, common, summary


def make_random_orthogonal(common: torch.Tensor, *, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    candidate = torch.randn(common.shape, generator=generator, dtype=torch.float32)
    candidate = candidate - torch.dot(candidate, common) * common
    if float(candidate.norm().item()) <= 0.0:
        raise ValueError("Degenerate random control")
    return (candidate / candidate.norm()).contiguous()


def treatments_for_target(
    *,
    native: torch.Tensor,
    common: torch.Tensor,
    random_unit: torch.Tensor,
    target_domain: str,
) -> tuple[list[Treatment], dict[str, float]]:
    target_norm = float(native.norm().item())
    projection_coefficient = float(torch.dot(native, common).item())
    projection = (projection_coefficient * common).contiguous()
    residual = (native - projection).contiguous()
    if float(residual.norm().item()) <= 0.0:
        raise ValueError(f"{target_domain}: residual is unexpectedly zero")
    treatments: list[Treatment] = [
        Treatment(
            condition="native_full",
            vector=native,
            vector_family="native",
            multiplier=1.0,
            target_domain=target_domain,
            target_native_l2_norm=target_norm,
            projection_coefficient=projection_coefficient,
        )
    ]
    for condition, multiplier in SHARED_MULTIPLIERS:
        treatments.append(
            Treatment(
                condition=condition,
                vector=(common * (target_norm * multiplier)).contiguous(),
                vector_family="global_shared_direction",
                multiplier=multiplier,
                target_domain=target_domain,
                target_native_l2_norm=target_norm,
                projection_coefficient=projection_coefficient,
            )
        )
    treatments.extend(
        [
            Treatment(
                condition="native_projection_on_shared",
                vector=projection,
                vector_family="literal_projection",
                multiplier=None,
                target_domain=target_domain,
                target_native_l2_norm=target_norm,
                projection_coefficient=projection_coefficient,
            ),
            Treatment(
                condition="native_residual_after_shared",
                vector=residual,
                vector_family="literal_residual",
                multiplier=None,
                target_domain=target_domain,
                target_native_l2_norm=target_norm,
                projection_coefficient=projection_coefficient,
            ),
            Treatment(
                condition="random_orthogonal_target_norm",
                vector=(random_unit * target_norm).contiguous(),
                vector_family="random_orthogonal_control",
                multiplier=1.0,
                target_domain=target_domain,
                target_native_l2_norm=target_norm,
                projection_coefficient=projection_coefficient,
            ),
        ]
    )
    geometry = {
        "target_native_l2_norm": target_norm,
        "native_cosine_to_common": projection_coefficient / target_norm,
        "projection_l2_norm": float(projection.norm().item()),
        "residual_l2_norm": float(residual.norm().item()),
        "projection_energy_fraction": float(projection.square().sum().item() / native.square().sum().item()),
        "residual_energy_fraction": float(residual.square().sum().item() / native.square().sum().item()),
        "random_cosine_to_common": float(torch.dot(random_unit, common).item()),
        "random_cosine_to_native": float(torch.dot(random_unit, native / native.norm()).item()),
    }
    return treatments, geometry


def evaluate_target(
    model,
    pairs: Sequence[Any],
    *,
    treatments: Sequence[Treatment],
    layer: int,
    batch_size: int,
    tool_token_id: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if len(treatments) != len(CONDITION_ORDER):
        raise ValueError(f"Expected {len(CONDITION_ORDER)} treatments, got {len(treatments)}")
    target_domain = treatments[0].target_domain
    accum: dict[str, dict[str, float]] = {treatment.condition: defaultdict(float) for treatment in treatments}
    clean_logit_sum = corrupt_logit_sum = 0.0
    clean_prob_sum = corrupt_prob_sum = 0.0
    clean_tool_top1 = corrupt_tool_top1 = 0
    clean_tool_count = corrupt_non_tool_count = 0
    count = 0
    name = hook_name(layer, "pre")

    for batch in tqdm(
        build_pair_batches(pairs, batch_size=batch_size),
        desc=f"{target_domain}: shared-direction ablation",
        dynamic_ncols=True,
    ):
        clean_tokens = batch.clean_tokens_cpu.to(model.W_U.device)
        corrupt_tokens = batch.corrupt_tokens_cpu.to(model.W_U.device)
        with torch.no_grad():
            clean_logits = model(clean_tokens)
            corrupt_logits = model(corrupt_tokens)
        clean_logit, clean_prob, clean_top1 = tool_stats(clean_logits, tool_token_id)
        corrupt_logit, corrupt_prob, corrupt_top1 = tool_stats(corrupt_logits, tool_token_id)
        clean_is_tool = clean_top1 == tool_token_id
        corrupt_is_tool = corrupt_top1 == tool_token_id
        batch_count = int(clean_top1.shape[0])
        count += batch_count
        clean_logit_sum += float(clean_logit.sum().item())
        corrupt_logit_sum += float(corrupt_logit.sum().item())
        clean_prob_sum += float(clean_prob.sum().item())
        corrupt_prob_sum += float(corrupt_prob.sum().item())
        clean_tool_top1 += int(clean_is_tool.sum().item())
        corrupt_tool_top1 += int(corrupt_is_tool.sum().item())
        clean_tool_count += int(clean_is_tool.sum().item())
        corrupt_non_tool_count += int((~corrupt_is_tool).sum().item())

        for treatment in treatments:
            delta = treatment.vector.view(1, -1)
            with torch.no_grad():
                add_logits = model.run_with_hooks(
                    corrupt_tokens, fwd_hooks=[(name, make_last_token_add_hook(delta))]
                )
                remove_logits = model.run_with_hooks(
                    clean_tokens, fwd_hooks=[(name, make_last_token_add_hook(-delta))]
                )
            add_logit, add_prob, add_top1 = tool_stats(add_logits, tool_token_id)
            remove_logit, remove_prob, remove_top1 = tool_stats(remove_logits, tool_token_id)
            bucket = accum[treatment.condition]
            bucket["add_tool_top1"] += float((add_top1 == tool_token_id).sum().item())
            bucket["add_strict_flip"] += float(((~corrupt_is_tool) & (add_top1 == tool_token_id)).sum().item())
            bucket["add_logit_sum"] += float(add_logit.sum().item())
            bucket["add_prob_sum"] += float(add_prob.sum().item())
            bucket["remove_tool_top1"] += float((remove_top1 == tool_token_id).sum().item())
            bucket["remove_strict_drop"] += float((clean_is_tool & (remove_top1 != tool_token_id)).sum().item())
            bucket["remove_logit_sum"] += float(remove_logit.sum().item())
            bucket["remove_prob_sum"] += float(remove_prob.sum().item())
            del add_logits, remove_logits, add_logit, add_prob, add_top1, remove_logit, remove_prob, remove_top1
            clear_cuda()

        del clean_tokens, corrupt_tokens, clean_logits, corrupt_logits
        del clean_logit, clean_prob, clean_top1, corrupt_logit, corrupt_prob, corrupt_top1, clean_is_tool, corrupt_is_tool
        clear_cuda()

    if count != len(pairs):
        raise AssertionError(f"{target_domain}: processed {count} examples, expected {len(pairs)}")
    baseline = {
        "target_domain": target_domain,
        "n": count,
        "clean_mean_tool_logit": clean_logit_sum / count,
        "corrupt_mean_tool_logit": corrupt_logit_sum / count,
        "clean_mean_tool_prob": clean_prob_sum / count,
        "corrupt_mean_tool_prob": corrupt_prob_sum / count,
        "clean_top1_rate": clean_tool_top1 / count,
        "corrupt_top1_rate": corrupt_tool_top1 / count,
        "clean_tool_count": clean_tool_count,
        "corrupt_non_tool_count": corrupt_non_tool_count,
    }
    logit_gap = float(baseline["clean_mean_tool_logit"] - baseline["corrupt_mean_tool_logit"])
    if logit_gap <= 0.0:
        raise ValueError(f"{target_domain}: non-positive clean-corrupt logit gap ({logit_gap})")

    rows: list[dict[str, Any]] = []
    by_condition = {treatment.condition: treatment for treatment in treatments}
    for condition in CONDITION_ORDER:
        treatment = by_condition[condition]
        bucket = accum[condition]
        add_logit = bucket["add_logit_sum"] / count
        remove_logit = bucket["remove_logit_sum"] / count
        effective_norm = float(treatment.vector.norm().item())
        rows.append(
            {
                "target_domain": target_domain,
                "condition": condition,
                "vector_family": treatment.vector_family,
                "multiplier_from_target_norm": treatment.multiplier,
                "n": count,
                "target_native_l2_norm": treatment.target_native_l2_norm,
                "projection_coefficient": treatment.projection_coefficient,
                "effective_l2_norm": effective_norm,
                "effective_norm_over_target_native": effective_norm / treatment.target_native_l2_norm,
                "baseline_clean_mean_tool_logit": baseline["clean_mean_tool_logit"],
                "baseline_corrupt_mean_tool_logit": baseline["corrupt_mean_tool_logit"],
                "baseline_clean_top1_rate": baseline["clean_top1_rate"],
                "baseline_corrupt_top1_rate": baseline["corrupt_top1_rate"],
                "add_tool_call_top1_rate": bucket["add_tool_top1"] / count,
                "add_strict_flip_rate": bucket["add_strict_flip"] / max(corrupt_non_tool_count, 1),
                "add_mean_tool_call_logit": add_logit,
                "add_mean_tool_call_prob": bucket["add_prob_sum"] / count,
                "sufficiency_normalized_logit_gap": (add_logit - baseline["corrupt_mean_tool_logit"]) / logit_gap,
                "remove_remaining_tool_call_top1_rate": bucket["remove_tool_top1"] / count,
                "remove_strict_drop_rate": bucket["remove_strict_drop"] / max(clean_tool_count, 1),
                "remove_mean_tool_call_logit": remove_logit,
                "remove_mean_tool_call_prob": bucket["remove_prob_sum"] / count,
                "necessity_normalized_logit_gap": (baseline["clean_mean_tool_logit"] - remove_logit) / logit_gap,
            }
        )
    return baseline, rows


def aggregate_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for condition in CONDITION_ORDER:
        subset = [row for row in rows if row["condition"] == condition]
        if not subset:
            continue
        output.append(
            {
                "condition": condition,
                "vector_family": subset[0]["vector_family"],
                "multiplier_from_target_norm": subset[0]["multiplier_from_target_norm"],
                "target_count": len(subset),
                "mean_effective_norm_over_target_native": sum(float(row["effective_norm_over_target_native"]) for row in subset) / len(subset),
                "mean_sufficiency": sum(float(row["sufficiency_normalized_logit_gap"]) for row in subset) / len(subset),
                "mean_necessity": sum(float(row["necessity_normalized_logit_gap"]) for row in subset) / len(subset),
                "mean_strict_flip_rate": sum(float(row["add_strict_flip_rate"]) for row in subset) / len(subset),
                "mean_strict_drop_rate": sum(float(row["remove_strict_drop_rate"]) for row in subset) / len(subset),
            }
        )
    return output


def write_summary_markdown(path: Path, *, model_label: str, aggregates: Sequence[dict[str, Any]]) -> None:
    lines = [
        f"# {model_label}: shared-direction causal ablation",
        "",
        "All rows average one held-out evaluation over each target domain. `projection` and `residual` retain their native amplitudes; they are not renormalized.",
        "",
        "| treatment | family | relative norm | Suff. | Necc. | strict flip | strict drop |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in aggregates:
        multiplier = row["multiplier_from_target_norm"]
        relative_norm = f"{float(row['mean_effective_norm_over_target_native']):.3f}"
        if multiplier is not None and row["vector_family"] == "global_shared_direction":
            relative_norm = f"{float(multiplier):.1f}×"
        lines.append(
            "| {condition} | {vector_family} | {norm} | {suff:.3f} | {necc:.3f} | {flip:.1%} | {drop:.1%} |".format(
                condition=row["condition"],
                vector_family=row["vector_family"],
                norm=relative_norm,
                suff=float(row["mean_sufficiency"]),
                necc=float(row["mean_necessity"]),
                flip=float(row["mean_strict_flip_rate"]),
                drop=float(row["mean_strict_drop_rate"]),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    targets = unique_domains(args.target_domains)
    output_root = args.output_root.resolve()
    dataset_root = args.dataset_view_root.resolve()
    native_root = args.native_vector_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite output root: {output_root}")
    if args.layer != 24:
        raise ValueError("This registered experiment intentionally uses the existing L24 direction.")
    for domain in targets:
        if not (dataset_root / domain).is_dir():
            raise FileNotFoundError(f"Missing dataset view for {domain}: {dataset_root / domain}")

    set_seed(args.seed)
    ensure_dir(output_root)
    write_json(
        output_root / "run_config.json",
        {
            "model_label": args.model_label,
            "model_path": str(args.model_path.resolve()),
            "dataset_view_root": str(dataset_root),
            "native_vector_root": str(native_root),
            "native_vector_provenance": "exact native vectors and global uncentered-SVD direction reused from completed full matrix run",
            "domains_used_to_fit_shared_direction": list(DEFAULT_DOMAINS),
            "target_domains": list(targets),
            "layer": args.layer,
            "hook_kind": args.hook_kind,
            "eval_pairs": args.eval_pairs,
            "batch_size": args.batch_size,
            "conditions": list(CONDITION_ORDER),
            "shared_direction_calibration": "target-native vector L2 norm; shared direction then multiplied by 1.0, 1.5, 2.0, or 3.0",
            "projection_necessity_protocol": "subtract literal P_u(v_d) from clean activations; no projection/residual renormalization",
            "random_control": "one seeded equal-norm random unit vector explicitly orthogonal to the global common direction",
        },
    )

    model, _tokenizer, tool_token_id = load_model_and_tokenizer(model_path=args.model_path.resolve(), device=args.device)
    try:
        if args.layer < 0 or args.layer >= int(model.cfg.n_layers):
            raise ValueError(f"Invalid layer L{args.layer} for {args.model_label}")
        vectors, common, pca_summary = load_vector_bundle(native_root, layer=args.layer, d_model=int(model.cfg.d_model))
        random_unit = make_random_orthogonal(common, seed=args.seed + 9973)
        geometry: dict[str, Any] = {
            "rank1_explained_energy_ratio": pca_summary["rank1_explained_energy_ratio"],
            "common_direction_method": pca_summary["common_direction_method"],
            "targets": {},
        }
        validation: dict[str, dict[str, Any]] = {}
        baselines: list[dict[str, Any]] = []
        all_rows: list[dict[str, Any]] = []
        for target_domain in targets:
            treatments, target_geometry = treatments_for_target(
                native=vectors[target_domain],
                common=common,
                random_unit=random_unit,
                target_domain=target_domain,
            )
            geometry["targets"][target_domain] = target_geometry
            pairs = load_sample_pairs(
                model,
                dataset_root=dataset_root / target_domain,
                split="test",
                max_pairs=args.eval_pairs,
            )
            validation[f"{target_domain}/test"] = validate_pairs(
                pairs, domain=target_domain, split="test", expected_count=args.eval_pairs
            )
            baseline, target_rows = evaluate_target(
                model,
                pairs,
                treatments=treatments,
                layer=args.layer,
                batch_size=args.batch_size,
                tool_token_id=tool_token_id,
            )
            baselines.append(baseline)
            all_rows.extend(target_rows)
            write_csv(output_root / "causal_ablation_long.partial.csv", all_rows)
            write_csv(output_root / "target_baselines.partial.csv", baselines)
            del pairs
            clear_cuda()

        expected_cells = len(targets) * len(CONDITION_ORDER)
        if len(all_rows) != expected_cells:
            raise AssertionError(f"Expected {expected_cells} cells, got {len(all_rows)}")
        aggregates = aggregate_rows(all_rows)
        write_csv(output_root / "causal_ablation_long.csv", all_rows)
        write_csv(output_root / "aggregate_metrics.csv", aggregates)
        write_csv(output_root / "target_baselines.csv", baselines)
        write_json(output_root / "shared_direction_geometry.json", geometry)
        write_json(output_root / "pair_validation.json", validation)
        write_summary_markdown(output_root / "shared_direction_causal_table.md", model_label=args.model_label, aggregates=aggregates)
        write_json(
            output_root / "completion.json",
            {
                "status": "complete",
                "expected_cells": expected_cells,
                "observed_cells": len(all_rows),
                "tool_token_id": tool_token_id,
                "layer": args.layer,
                "hook_kind": args.hook_kind,
                "model_d_model": int(model.cfg.d_model),
                "model_n_layers": int(model.cfg.n_layers),
            },
        )
        print(json.dumps({"status": "complete", "output_root": str(output_root), "aggregates": aggregates}, ensure_ascii=False))
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
