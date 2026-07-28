#!/usr/bin/env python3
"""Evaluate Qwen3-14B raw-vector transfer at 2x and 3x only.

This is a provenance-preserving extension of the completed L24-pre 4x4
matrix run.  It reuses the exact four native vectors already extracted from
the original 400-pair training splits, then evaluates only the new raw-source
multipliers (2.0 and 3.0) on the same 100 held-out pairs per target domain.
It deliberately does *not* re-run the aligned or 1.5x conditions.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

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


RAW_CONDITIONS: tuple[tuple[str, float], ...] = (("raw_2p0", 2.0), ("raw_3p0", 3.0))


@dataclass(frozen=True)
class Treatment:
    source_domain: str
    target_domain: str
    condition: str
    alpha: float
    vector: torch.Tensor
    source_l2_norm: float
    target_l2_norm: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Qwen3-14B 2x/3x raw-vector 4x4 extension.")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--dataset-view-root", type=Path, required=True)
    parser.add_argument("--native-vector-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=24)
    parser.add_argument("--hook-kind", choices=("pre",), default="pre")
    parser.add_argument("--eval-pairs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_native_vectors(root: Path, *, layer: int, d_model: int) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
    vectors: dict[str, torch.Tensor] = {}
    norms: dict[str, float] = {}
    vector_dir = root / "native_vectors"
    for domain in DEFAULT_DOMAINS:
        path = vector_dir / f"{domain}_L{layer}_pre.pt"
        if not path.is_file():
            raise FileNotFoundError(f"Missing native vector: {path}")
        bundle = torch.load(path, map_location="cpu", weights_only=False)
        vector = bundle["mean_diff"].detach().cpu().float().view(-1)
        if vector.numel() != d_model:
            raise ValueError(f"{domain}: expected d_model={d_model}, got {vector.numel()}")
        if not bool(torch.isfinite(vector).all()):
            raise ValueError(f"{domain}: native vector has non-finite values")
        norm = float(vector.norm().item())
        if norm <= 0.0:
            raise ValueError(f"{domain}: native vector has zero norm")
        vectors[domain] = vector.contiguous()
        norms[domain] = norm
    return vectors, norms


def make_treatments(
    vectors: dict[str, torch.Tensor], *, target_domain: str, norms: dict[str, float]
) -> list[Treatment]:
    return [
        Treatment(
            source_domain=source_domain,
            target_domain=target_domain,
            condition=condition,
            alpha=alpha,
            vector=(vectors[source_domain] * alpha).contiguous(),
            source_l2_norm=norms[source_domain],
            target_l2_norm=norms[target_domain],
        )
        for source_domain in DEFAULT_DOMAINS
        for condition, alpha in RAW_CONDITIONS
    ]


def evaluate_target(
    model,
    pairs: Sequence[Any],
    *,
    target_domain: str,
    vectors: dict[str, torch.Tensor],
    norms: dict[str, float],
    layer: int,
    batch_size: int,
    tool_token_id: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    treatments = make_treatments(vectors, target_domain=target_domain, norms=norms)
    accum: dict[tuple[str, str], dict[str, float]] = {
        (item.source_domain, item.condition): defaultdict(float) for item in treatments
    }
    clean_logit_sum = corrupt_logit_sum = 0.0
    clean_prob_sum = corrupt_prob_sum = 0.0
    clean_tool_top1 = corrupt_tool_top1 = 0
    clean_tool_count = corrupt_non_tool_count = 0
    count = 0
    name = hook_name(layer, "pre")

    for batch in tqdm(build_pair_batches(pairs, batch_size=batch_size), desc=f"{target_domain}: 14B raw 2x/3x", dynamic_ncols=True):
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
            bucket = accum[(treatment.source_domain, treatment.condition)]
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
        raise ValueError(f"{target_domain}: non-positive clean-corrupt logit gap: {logit_gap}")

    by_treatment = {(item.source_domain, item.condition): item for item in treatments}
    rows: list[dict[str, Any]] = []
    for source_domain in DEFAULT_DOMAINS:
        for condition, _alpha in RAW_CONDITIONS:
            treatment = by_treatment[(source_domain, condition)]
            bucket = accum[(source_domain, condition)]
            add_logit = bucket["add_logit_sum"] / count
            remove_logit = bucket["remove_logit_sum"] / count
            effective_norm = float(treatment.vector.norm().item())
            rows.append(
                {
                    "source_domain": source_domain,
                    "target_domain": target_domain,
                    "condition": treatment.condition,
                    "n": count,
                    "source_l2_norm": treatment.source_l2_norm,
                    "target_native_l2_norm": treatment.target_l2_norm,
                    "applied_scale_from_source": treatment.alpha,
                    "effective_l2_norm": effective_norm,
                    "effective_norm_over_target_native": effective_norm / treatment.target_l2_norm,
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


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for condition, alpha in RAW_CONDITIONS:
        for scope, subset in (
            ("all", [row for row in rows if row["condition"] == condition]),
            ("off_diagonal", [row for row in rows if row["condition"] == condition and row["source_domain"] != row["target_domain"]]),
        ):
            output.append(
                {
                    "condition": condition,
                    "alpha": alpha,
                    "scope": scope,
                    "cells": len(subset),
                    "mean_sufficiency": sum(float(row["sufficiency_normalized_logit_gap"]) for row in subset) / len(subset),
                    "mean_necessity": sum(float(row["necessity_normalized_logit_gap"]) for row in subset) / len(subset),
                    "mean_strict_flip_rate": sum(float(row["add_strict_flip_rate"]) for row in subset) / len(subset),
                    "mean_strict_drop_rate": sum(float(row["remove_strict_drop_rate"]) for row in subset) / len(subset),
                }
            )
    return output


def main() -> None:
    args = parse_args()
    output_root = args.output_root.resolve()
    dataset_root = args.dataset_view_root.resolve()
    native_vector_root = args.native_vector_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite: {output_root}")
    for domain in DEFAULT_DOMAINS:
        if not (dataset_root / domain).is_dir():
            raise FileNotFoundError(f"Missing domain data: {dataset_root / domain}")
    if args.layer != 24:
        raise ValueError("This registered extension is intentionally fixed to the original L24 experiment.")

    set_seed(args.seed)
    ensure_dir(output_root)
    write_json(
        output_root / "run_config.json",
        {
            "model_label": "Qwen3-14B",
            "model_path": str(args.model_path.resolve()),
            "dataset_view_root": str(dataset_root),
            "native_vector_root": str(native_vector_root),
            "native_vector_provenance": "exact L24-pre native vectors reused from the completed 4x4 run",
            "domains": list(DEFAULT_DOMAINS),
            "layer": args.layer,
            "hook_kind": args.hook_kind,
            "eval_pairs": args.eval_pairs,
            "batch_size": args.batch_size,
            "conditions": [name for name, _alpha in RAW_CONDITIONS],
            "condition_definitions": {name: f"unmodified source native vector multiplied by alpha={alpha}" for name, alpha in RAW_CONDITIONS},
        },
    )

    model, _tokenizer, tool_token_id = load_model_and_tokenizer(model_path=args.model_path.resolve(), device=args.device)
    try:
        if int(model.cfg.n_layers) <= args.layer:
            raise ValueError(f"L{args.layer} is invalid for this model")
        vectors, norms = load_native_vectors(native_vector_root, layer=args.layer, d_model=int(model.cfg.d_model))
        validation: dict[str, dict[str, Any]] = {}
        baselines: list[dict[str, Any]] = []
        rows: list[dict[str, Any]] = []
        for target_domain in DEFAULT_DOMAINS:
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
                target_domain=target_domain,
                vectors=vectors,
                norms=norms,
                layer=args.layer,
                batch_size=args.batch_size,
                tool_token_id=tool_token_id,
            )
            baselines.append(baseline)
            rows.extend(target_rows)
            write_csv(output_root / "matrix_long.partial.csv", rows)
            write_csv(output_root / "target_baselines.partial.csv", baselines)
            del pairs
            clear_cuda()

        expected_cells = len(DEFAULT_DOMAINS) * len(DEFAULT_DOMAINS) * len(RAW_CONDITIONS)
        if len(rows) != expected_cells:
            raise AssertionError(f"Expected {expected_cells} cells, got {len(rows)}")
        summary = summarize(rows)
        write_csv(output_root / "matrix_long.csv", rows)
        write_csv(output_root / "target_baselines.csv", baselines)
        write_csv(output_root / "aggregate_metrics.csv", summary)
        write_json(output_root / "pair_validation.json", validation)
        write_json(output_root / "native_vector_norms.json", norms)
        write_json(
            output_root / "completion.json",
            {
                "status": "complete",
                "expected_cells": expected_cells,
                "observed_cells": len(rows),
                "tool_token_id": tool_token_id,
                "layer": args.layer,
                "hook_kind": args.hook_kind,
                "model_d_model": int(model.cfg.d_model),
                "model_n_layers": int(model.cfg.n_layers),
            },
        )
        print(json.dumps({"status": "complete", "output_root": str(output_root), "summary": summary}, ensure_ascii=False))
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
