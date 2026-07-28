#!/usr/bin/env python3
"""Reproduce the frozen D1 Coding transfer and add its random control.

This is intentionally a narrow companion to :mod:`run_full_transfer_matrix`.
It reads the already-completed Qwen3 matrix's frozen D1/D3/D4/D5 vectors,
re-evaluates the three D1 -> target Code cells as an integrity check, and
evaluates one seeded, equal-norm random direction on the same held-out pairs.

The output uses the same strict-flip / strict-drop definitions as the matrix:
the denominator is the target-domain corrupt non-tool / clean tool subset,
respectively.  It never modifies the frozen matrix or dataset view.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import torch
from tqdm.auto import tqdm

from run_full_transfer_matrix import (
    build_pair_batches,
    clear_cuda,
    hook_name,
    load_model_and_tokenizer,
    load_sample_pairs,
    make_last_token_add_hook,
    tool_stats,
    validate_pairs,
    write_csv,
    write_json,
)


TARGET_DOMAINS = ("D3", "D4", "D5")
TABLE_DOMAIN = {"D3": "Retrieval", "D4": "SQL", "D5": "Email"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--dataset-view-root", type=Path, required=True)
    parser.add_argument("--prior-matrix-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--targets", nargs="+", choices=TARGET_DOMAINS, default=list(TARGET_DOMAINS))
    parser.add_argument("--layer", type=int, default=24)
    parser.add_argument("--hook-kind", choices=("pre", "post"), default="pre")
    parser.add_argument("--eval-pairs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--random-seed", type=int, default=20260727)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def load_vector(path: Path, *, expected_dim: int, expected_layer: int, expected_hook_kind: str) -> torch.Tensor:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "mean_diff" not in payload:
        raise ValueError(f"{path}: expected a vector bundle with mean_diff")
    vector = torch.as_tensor(payload["mean_diff"], dtype=torch.float32).reshape(-1).contiguous()
    if vector.numel() != int(expected_dim):
        raise ValueError(f"{path}: vector dim {vector.numel()} != model dim {expected_dim}")
    if not bool(torch.isfinite(vector).all()) or float(vector.norm().item()) <= 0.0:
        raise ValueError(f"{path}: invalid vector")
    inferred_hook = str(payload.get("hook_kind", "pre" if payload.get("patch_layer") is not None else "post"))
    bundle_layer = payload.get("patch_layer") if inferred_hook == "pre" else payload.get("layer")
    if inferred_hook != expected_hook_kind or int(bundle_layer) != int(expected_layer):
        raise ValueError(
            f"{path}: expected L{expected_layer} {expected_hook_kind}, got L{bundle_layer} {inferred_hook}"
        )
    return vector


def read_prior_code_rows(path: Path) -> dict[str, dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows: dict[str, dict[str, str]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if (
                row.get("source_domain") == "D1"
                and row.get("target_domain") in TARGET_DOMAINS
                and row.get("condition") == "target_norm_aligned"
            ):
                rows[str(row["target_domain"])] = row
    missing = [domain for domain in TARGET_DOMAINS if domain not in rows]
    if missing:
        raise ValueError(f"Prior matrix has no D1 target-norm Code cells for {missing}")
    return rows


def seeded_unit_vector(dim: int, *, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    vector = torch.randn((int(dim),), generator=generator, dtype=torch.float32)
    if float(vector.norm().item()) <= 0.0:
        raise RuntimeError("Degenerate random direction")
    return (vector / vector.norm()).contiguous()


def evaluate_target(
    model: Any,
    pairs: Sequence[Any],
    *,
    target_domain: str,
    code_vector: torch.Tensor,
    random_vector: torch.Tensor,
    source_norm: float,
    target_norm: float,
    layer: int,
    hook_kind: str,
    batch_size: int,
    tool_token_id: int,
    random_seed: int,
    random_cosine_to_code: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    treatments = {
        "Code": code_vector.contiguous(),
        "Random": random_vector.contiguous(),
    }
    name = hook_name(layer, hook_kind)
    accum: dict[str, dict[str, float]] = {name: defaultdict(float) for name in treatments}
    clean_logit_sum = corrupt_logit_sum = 0.0
    clean_prob_sum = corrupt_prob_sum = 0.0
    clean_tool_top1 = corrupt_tool_top1 = 0
    clean_tool_count = corrupt_non_tool_count = 0
    count = 0

    for batch in tqdm(
        build_pair_batches(pairs, batch_size=batch_size),
        desc=f"{target_domain}: Code + Random control",
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

        for direction, vector in treatments.items():
            delta = vector.view(1, -1)
            with torch.no_grad():
                add_logits = model.run_with_hooks(
                    corrupt_tokens, fwd_hooks=[(name, make_last_token_add_hook(delta))]
                )
                remove_logits = model.run_with_hooks(
                    clean_tokens, fwd_hooks=[(name, make_last_token_add_hook(-delta))]
                )
            add_logit, add_prob, add_top1 = tool_stats(add_logits, tool_token_id)
            remove_logit, remove_prob, remove_top1 = tool_stats(remove_logits, tool_token_id)
            bucket = accum[direction]
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
        raise AssertionError(f"{target_domain}: processed {count}, expected {len(pairs)}")
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
    gap = float(baseline["clean_mean_tool_logit"] - baseline["corrupt_mean_tool_logit"])
    if gap <= 0.0:
        raise ValueError(f"{target_domain}: non-positive clean/corrupt logit gap: {gap}")

    rows: list[dict[str, Any]] = []
    for direction, vector in treatments.items():
        bucket = accum[direction]
        add_logit = bucket["add_logit_sum"] / count
        remove_logit = bucket["remove_logit_sum"] / count
        effective_norm = float(vector.norm().item())
        rows.append(
            {
                "source_domain": "D1",
                "target_domain": target_domain,
                "table_domain": TABLE_DOMAIN[target_domain],
                "direction": direction,
                "condition": "target_norm_aligned" if direction == "Code" else "random_target_norm",
                "n": count,
                "source_l2_norm": source_norm if direction == "Code" else 1.0,
                "target_native_l2_norm": target_norm,
                "applied_scale_from_source": target_norm / source_norm if direction == "Code" else target_norm,
                "effective_l2_norm": effective_norm,
                "effective_norm_over_target_native": effective_norm / target_norm,
                "random_seed": int(random_seed) if direction == "Random" else None,
                "random_cosine_to_code": random_cosine_to_code if direction == "Random" else None,
                "baseline_clean_mean_tool_logit": baseline["clean_mean_tool_logit"],
                "baseline_corrupt_mean_tool_logit": baseline["corrupt_mean_tool_logit"],
                "baseline_clean_top1_rate": baseline["clean_top1_rate"],
                "baseline_corrupt_top1_rate": baseline["corrupt_top1_rate"],
                "baseline_clean_tool_count": clean_tool_count,
                "baseline_corrupt_non_tool_count": corrupt_non_tool_count,
                "add_tool_call_top1_rate": bucket["add_tool_top1"] / count,
                "add_strict_flip_count": int(bucket["add_strict_flip"]),
                "add_strict_flip_rate": bucket["add_strict_flip"] / max(corrupt_non_tool_count, 1),
                "add_mean_tool_call_logit": add_logit,
                "add_mean_tool_call_prob": bucket["add_prob_sum"] / count,
                "sufficiency_normalized_logit_gap": (add_logit - baseline["corrupt_mean_tool_logit"]) / gap,
                "remove_remaining_tool_call_top1_rate": bucket["remove_tool_top1"] / count,
                "remove_strict_drop_count": int(bucket["remove_strict_drop"]),
                "remove_strict_drop_rate": bucket["remove_strict_drop"] / max(clean_tool_count, 1),
                "remove_mean_tool_call_logit": remove_logit,
                "remove_mean_tool_call_prob": bucket["remove_prob_sum"] / count,
                "necessity_normalized_logit_gap": (baseline["clean_mean_tool_logit"] - remove_logit) / gap,
            }
        )
    return baseline, rows


def validate_code_reproduction(observed: dict[str, Any], expected: dict[str, str], *, target_domain: str) -> dict[str, float]:
    comparisons = {
        "baseline_clean_top1_rate": float(expected["baseline_clean_top1_rate"]),
        "baseline_corrupt_top1_rate": float(expected["baseline_corrupt_top1_rate"]),
        "add_strict_flip_rate": float(expected["add_strict_flip_rate"]),
        "remove_strict_drop_rate": float(expected["remove_strict_drop_rate"]),
    }
    deltas: dict[str, float] = {}
    for key, expected_value in comparisons.items():
        delta = abs(float(observed[key]) - expected_value)
        deltas[key] = delta
        if delta > 1e-8:
            raise AssertionError(
                f"{target_domain}: re-evaluated Code {key}={observed[key]} differs from frozen matrix {expected_value}"
            )
    return deltas


def write_summary(path: Path, *, model_label: str, rows: Sequence[dict[str, Any]]) -> None:
    lines = [
        f"# {model_label}: frozen Coding transfer with Random control",
        "",
        "Code re-evaluates the frozen D1 vector from the prior target-norm matrix. Random is one seeded Gaussian unit direction, rescaled separately to each target native norm.",
        "",
        "| domain | direction | strict flip / strict drop | eligible corrupt / clean |",
        "|---|---|---:|---:|",
    ]
    for domain in TARGET_DOMAINS:
        for direction in ("Code", "Random"):
            row = next(item for item in rows if item["target_domain"] == domain and item["direction"] == direction)
            lines.append(
                f"| {row['table_domain']} | {direction} | {100 * float(row['add_strict_flip_rate']):.0f}/{100 * float(row['remove_strict_drop_rate']):.0f} | {row['baseline_corrupt_non_tool_count']}/{row['baseline_clean_tool_count']} |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    targets = tuple(dict.fromkeys(args.targets))
    output_root = args.output_root.resolve()
    prior_root = args.prior_matrix_root.resolve()
    dataset_view_root = args.dataset_view_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite existing output root: {output_root}")
    if not dataset_view_root.is_dir() or not prior_root.is_dir():
        raise FileNotFoundError("dataset view or prior matrix root is missing")
    prior_rows = read_prior_code_rows(prior_root / "matrix_long.csv")
    output_root.mkdir(parents=True, exist_ok=False)

    model = None
    try:
        model, _tokenizer, tool_token_id = load_model_and_tokenizer(
            model_path=args.model_path.resolve(), device=args.device
        )
        if args.layer < 0 or args.layer >= int(model.cfg.n_layers):
            raise ValueError(f"L{args.layer} is invalid for {args.model_label}")
        d_model = int(model.cfg.d_model)
        d1_vector = load_vector(
            prior_root / "native_vectors" / f"D1_L{args.layer}_{args.hook_kind}.pt",
            expected_dim=d_model,
            expected_layer=args.layer,
            expected_hook_kind=args.hook_kind,
        )
        d1_norm = float(d1_vector.norm().item())
        random_unit = seeded_unit_vector(d_model, seed=args.random_seed)
        code_unit = d1_vector / d1_vector.norm().clamp_min(1e-12)
        random_cosine = float(torch.dot(random_unit, code_unit).item())

        write_json(
            output_root / "run_config.json",
            {
                "model_label": args.model_label,
                "model_path": str(args.model_path.resolve()),
                "dataset_view_root": str(dataset_view_root),
                "prior_matrix_root": str(prior_root),
                "source_vector": str(prior_root / "native_vectors" / f"D1_L{args.layer}_{args.hook_kind}.pt"),
                "targets": list(targets),
                "layer": args.layer,
                "hook_kind": args.hook_kind,
                "eval_pairs": args.eval_pairs,
                "random_control": {
                    "kind": "one seeded Gaussian unit vector per model",
                    "seed": args.random_seed,
                    "cosine_to_frozen_coding_direction": random_cosine,
                    "per_target_scaling": "target native-vector L2 norm",
                },
            },
        )
        torch.save(
            {"random_unit": random_unit, "seed": args.random_seed, "cosine_to_code": random_cosine},
            output_root / "random_direction.pt",
        )

        validations: dict[str, Any] = {}
        target_vectors: dict[str, float] = {}
        baseline_rows: list[dict[str, Any]] = []
        result_rows: list[dict[str, Any]] = []
        reproduction: dict[str, Any] = {}
        for target in targets:
            target_vector = load_vector(
                prior_root / "native_vectors" / f"{target}_L{args.layer}_{args.hook_kind}.pt",
                expected_dim=d_model,
                expected_layer=args.layer,
                expected_hook_kind=args.hook_kind,
            )
            target_norm = float(target_vector.norm().item())
            target_vectors[target] = target_norm
            pairs = load_sample_pairs(
                model,
                dataset_root=dataset_view_root / target,
                split="test",
                max_pairs=args.eval_pairs,
            )
            validations[f"{target}/test"] = validate_pairs(
                pairs, domain=target, split="test", expected_count=args.eval_pairs
            )
            baseline, rows = evaluate_target(
                model,
                pairs,
                target_domain=target,
                code_vector=(code_unit * target_norm),
                random_vector=(random_unit * target_norm),
                source_norm=d1_norm,
                target_norm=target_norm,
                layer=args.layer,
                hook_kind=args.hook_kind,
                batch_size=args.batch_size,
                tool_token_id=tool_token_id,
                random_seed=args.random_seed,
                random_cosine_to_code=random_cosine,
            )
            code_row = next(row for row in rows if row["direction"] == "Code")
            reproduction[target] = validate_code_reproduction(code_row, prior_rows[target], target_domain=target)
            baseline_rows.append(baseline)
            result_rows.extend(rows)
            write_csv(output_root / "matrix_long.partial.csv", result_rows)
            write_csv(output_root / "target_baselines.partial.csv", baseline_rows)
            del pairs
            clear_cuda()

        expected_rows = len(targets) * 2
        if len(result_rows) != expected_rows:
            raise AssertionError(f"Expected {expected_rows} result rows, found {len(result_rows)}")
        write_csv(output_root / "matrix_long.csv", result_rows)
        write_csv(output_root / "target_baselines.csv", baseline_rows)
        write_json(output_root / "pair_validation.json", validations)
        write_json(
            output_root / "vector_metadata.json",
            {
                "source_domain": "D1",
                "source_l2_norm": d1_norm,
                "target_native_l2_norm": target_vectors,
                "random_seed": args.random_seed,
                "random_cosine_to_code": random_cosine,
            },
        )
        write_json(output_root / "code_reproduction.json", reproduction)
        write_summary(output_root / "summary.md", model_label=args.model_label, rows=result_rows)
        write_json(
            output_root / "completion.json",
            {
                "status": "complete",
                "expected_rows": expected_rows,
                "observed_rows": len(result_rows),
                "tool_token_id": tool_token_id,
                "layer": args.layer,
                "hook_kind": args.hook_kind,
                "model_d_model": d_model,
                "code_reproduction": "exact strict-rate match to the frozen matrix",
            },
        )
        print(json.dumps({"status": "complete", "output_root": str(output_root), "rows": len(result_rows)}))
    finally:
        if model is not None:
            del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
