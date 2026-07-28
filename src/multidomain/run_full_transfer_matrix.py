#!/usr/bin/env python3
"""Run a full 4x4 cross-domain causal-vector transfer matrix for one model.

The implementation deliberately reuses the submitted experiment's mechanism:
the L24 pre-residual mean clean-minus-corrupt vector, injected at the final
prompt token.  Unlike invoking the one-vector runner 32 times, it loads a
model once, builds all native vectors, then evaluates every source/target
pair with both requested magnitude treatments in a single pass over each
target's held-out pairs.

Conditions
----------
``target_norm_aligned``
    Preserve the source direction and set its L2 norm to the target domain's
    native-vector L2 norm.
``raw_1p5``
    Use the source domain's native vector multiplied by 1.5.

For every directed cell the output contains both injection (sufficiency) and
removal (necessity) measurements.  ``suff`` and ``necc`` use the paper's
target-specific clean--corrupt tool-call logit-gap normalization.
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
from run_cross_scale_fix_gate_and_patch import (  # noqa: E402
    collect_residuals_at_hook,
    compute_pca,
    hook_name,
    make_last_token_add_hook,
)


DEFAULT_DOMAINS = ("D1", "D3", "D4", "D5")
CONDITIONS = ("target_norm_aligned", "raw_1p5")


@dataclass(frozen=True)
class VectorTreatment:
    source_domain: str
    target_domain: str
    condition: str
    vector: torch.Tensor
    source_l2_norm: float
    target_l2_norm: float
    applied_scale_from_source: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one Qwen3 model's full bidirectional 4x4 transfer matrix.")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--model-label", type=str, required=True)
    parser.add_argument("--dataset-view-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--domains", nargs="+", choices=DEFAULT_DOMAINS, default=list(DEFAULT_DOMAINS))
    parser.add_argument("--source-domains", nargs="+", choices=DEFAULT_DOMAINS, default=None)
    parser.add_argument("--target-domains", nargs="+", choices=DEFAULT_DOMAINS, default=None)
    parser.add_argument("--layer", type=int, default=24)
    parser.add_argument("--hook-kind", choices=("pre", "post"), default="pre")
    parser.add_argument("--train-pairs", type=int, default=400)
    parser.add_argument("--eval-pairs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def unique_domains(values: Iterable[str], *, label: str) -> tuple[str, ...]:
    output = tuple(str(value).upper() for value in values)
    if len(set(output)) != len(output):
        raise ValueError(f"Duplicate value in {label}: {output}")
    invalid = sorted(set(output) - set(DEFAULT_DOMAINS))
    if invalid:
        raise ValueError(f"Unsupported domain in {label}: {invalid}")
    if not output:
        raise ValueError(f"{label} must not be empty")
    return output


def validate_pairs(pairs: Sequence[Any], *, domain: str, split: str, expected_count: int) -> dict[str, Any]:
    if len(pairs) != int(expected_count):
        raise ValueError(f"{domain}/{split}: expected {expected_count} usable pairs, found {len(pairs)}")
    differing_positions: set[int] = set()
    for pair in pairs:
        clean = pair.clean_tokens_cpu
        corrupt = pair.corrupt_tokens_cpu
        if clean.shape != corrupt.shape:
            raise ValueError(f"{domain}/{split}/{pair.sample_id}: token shapes differ")
        differences = (clean != corrupt).nonzero(as_tuple=False)
        if int(differences.shape[0]) != 1:
            raise ValueError(
                f"{domain}/{split}/{pair.sample_id}: expected exactly one differing token, found {differences.shape[0]}"
            )
        differing_positions.add(int(differences[0, -1].item()))
    return {
        "pair_count": len(pairs),
        "token_length_min": min(int(pair.token_len) for pair in pairs),
        "token_length_max": max(int(pair.token_len) for pair in pairs),
        "unique_differing_token_positions": sorted(differing_positions),
    }


def collect_native_vectors(
    model,
    *,
    dataset_view_root: Path,
    vector_domains: Sequence[str],
    layer: int,
    hook_kind: str,
    train_pairs: int,
    batch_size: int,
    output_root: Path,
) -> tuple[dict[str, torch.Tensor], dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    vectors: dict[str, torch.Tensor] = {}
    bundles: dict[str, dict[str, Any]] = {}
    validation: dict[str, dict[str, Any]] = {}
    native_root = output_root / "native_vectors"
    ensure_dir(native_root)
    for domain in vector_domains:
        pairs = load_sample_pairs(
            model,
            dataset_root=dataset_view_root / domain,
            split="train",
            max_pairs=train_pairs,
        )
        validation[f"{domain}/train"] = validate_pairs(
            pairs, domain=domain, split="train", expected_count=train_pairs
        )
        clean_resid, corrupt_resid = collect_residuals_at_hook(
            model,
            pairs,
            layer=layer,
            hook_kind=hook_kind,
            batch_size=batch_size,
        )
        diff = clean_resid - corrupt_resid
        pca = compute_pca(diff, n_components=10)
        mean_diff = pca["mean_diff"].detach().cpu().float().view(-1)
        if mean_diff.numel() != int(model.cfg.d_model):
            raise ValueError(f"{domain}: unexpected vector dimension {mean_diff.numel()}")
        if not bool(torch.isfinite(mean_diff).all()) or float(mean_diff.norm().item()) <= 0.0:
            raise ValueError(f"{domain}: invalid native vector")
        bundle: dict[str, Any] = {
            "mean_diff": mean_diff,
            "components": pca["components"].detach().cpu(),
            "explained_variance": pca["explained_variance"].detach().cpu(),
            "explained_variance_ratio": pca["explained_variance_ratio"].detach().cpu(),
            "singular_values": pca["singular_values"].detach().cpu(),
            "sample_ids_train": [pair.sample_id for pair in pairs],
            "patch_layer": int(layer) if hook_kind == "pre" else None,
            "layer": int(layer) if hook_kind == "post" else None,
            "hook_kind": hook_kind,
            "domain": domain,
            "model_d_model": int(model.cfg.d_model),
        }
        torch.save(bundle, native_root / f"{domain}_L{layer}_{hook_kind}.pt")
        vectors[domain] = mean_diff
        bundles[domain] = {
            "path": str(native_root / f"{domain}_L{layer}_{hook_kind}.pt"),
            "l2_norm": float(mean_diff.norm().item()),
            "rms": float(mean_diff.square().mean().sqrt().item()),
            "train_pair_count": len(pairs),
            "pca_explained_variance_ratio": [float(value) for value in pca["explained_variance_ratio"].tolist()],
        }
        del clean_resid, corrupt_resid, diff, pca, pairs
        clear_cuda()
    return vectors, bundles, validation


def build_treatments(
    vectors: dict[str, torch.Tensor], *, source_domains: Sequence[str], target_domain: str
) -> list[VectorTreatment]:
    target_norm = float(vectors[target_domain].norm().item())
    treatments: list[VectorTreatment] = []
    for source_domain in source_domains:
        source = vectors[source_domain].detach().cpu().float().view(-1)
        source_norm = float(source.norm().item())
        if source_norm <= 0.0:
            raise ValueError(f"{source_domain}: source vector has zero norm")
        treatments.append(
            VectorTreatment(
                source_domain=source_domain,
                target_domain=target_domain,
                condition="target_norm_aligned",
                vector=(source * (target_norm / source_norm)).contiguous(),
                source_l2_norm=source_norm,
                target_l2_norm=target_norm,
                applied_scale_from_source=target_norm / source_norm,
            )
        )
        treatments.append(
            VectorTreatment(
                source_domain=source_domain,
                target_domain=target_domain,
                condition="raw_1p5",
                vector=(source * 1.5).contiguous(),
                source_l2_norm=source_norm,
                target_l2_norm=target_norm,
                applied_scale_from_source=1.5,
            )
        )
    return treatments


def evaluate_target_matrix(
    model,
    pairs: Sequence[Any],
    *,
    target_domain: str,
    vectors: dict[str, torch.Tensor],
    source_domains: Sequence[str],
    layer: int,
    hook_kind: str,
    batch_size: int,
    tool_token_id: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    treatments = build_treatments(vectors, source_domains=source_domains, target_domain=target_domain)
    batches = build_pair_batches(pairs, batch_size=batch_size)
    name = hook_name(layer, hook_kind)
    accum: dict[tuple[str, str], dict[str, float]] = {
        (item.source_domain, item.condition): defaultdict(float) for item in treatments
    }
    clean_logit_sum = 0.0
    corrupt_logit_sum = 0.0
    clean_prob_sum = 0.0
    corrupt_prob_sum = 0.0
    clean_tool_top1 = 0
    corrupt_tool_top1 = 0
    clean_tool_count = 0
    corrupt_non_tool_count = 0
    count = 0

    progress = tqdm(batches, desc=f"{target_domain}: full matrix L{layer} {hook_kind}", dynamic_ncols=True)
    for batch in progress:
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
        clean_logit_sum += float(clean_logit.sum().item())
        corrupt_logit_sum += float(corrupt_logit.sum().item())
        clean_prob_sum += float(clean_prob.sum().item())
        corrupt_prob_sum += float(corrupt_prob.sum().item())
        clean_tool_top1 += int(clean_is_tool.sum().item())
        corrupt_tool_top1 += int(corrupt_is_tool.sum().item())
        clean_tool_count += int(clean_is_tool.sum().item())
        corrupt_non_tool_count += int((~corrupt_is_tool).sum().item())
        count += batch_count

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
            key = (treatment.source_domain, treatment.condition)
            bucket = accum[key]
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

        del (
            clean_tokens,
            corrupt_tokens,
            clean_logits,
            corrupt_logits,
            clean_logit,
            clean_prob,
            clean_top1,
            corrupt_logit,
            corrupt_prob,
            corrupt_top1,
            clean_is_tool,
            corrupt_is_tool,
        )
        clear_cuda()
        progress.set_postfix(tok=batch.token_len)

    if count != len(pairs):
        raise AssertionError(f"{target_domain}: processed {count} instead of {len(pairs)} pairs")
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
        raise ValueError(f"{target_domain}: non-positive clean--corrupt tool-logit gap ({logit_gap})")

    rows: list[dict[str, Any]] = []
    by_treatment = {(item.source_domain, item.condition): item for item in treatments}
    for source_domain in source_domains:
        for condition in CONDITIONS:
            treatment = by_treatment[(source_domain, condition)]
            bucket = accum[(source_domain, condition)]
            add_logit = bucket["add_logit_sum"] / count
            remove_logit = bucket["remove_logit_sum"] / count
            vector_norm = float(treatment.vector.norm().item())
            rows.append(
                {
                    "source_domain": source_domain,
                    "target_domain": target_domain,
                    "condition": condition,
                    "n": count,
                    "source_l2_norm": treatment.source_l2_norm,
                    "target_native_l2_norm": treatment.target_l2_norm,
                    "applied_scale_from_source": treatment.applied_scale_from_source,
                    "effective_l2_norm": vector_norm,
                    "effective_norm_over_target_native": vector_norm / treatment.target_l2_norm,
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


def pairwise_cosine(vectors: dict[str, torch.Tensor], domains: Sequence[str]) -> dict[str, dict[str, float]]:
    units = {domain: vectors[domain].float() / vectors[domain].float().norm().clamp_min(1e-12) for domain in domains}
    return {
        source: {target: float(torch.dot(units[source], units[target]).item()) for target in domains}
        for source in domains
    }


def common_direction_pca(vectors: dict[str, torch.Tensor], domains: Sequence[str]) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    raw = torch.stack([vectors[domain].detach().cpu().float() for domain in domains], dim=0)
    norms = raw.norm(dim=1, keepdim=True).clamp_min(1e-12)
    unit = raw / norms
    # Uncentered SVD answers the common-direction question directly: how much
    # of total unit-vector energy lies in one shared rank-1 direction.
    _u_uncentered, singular_uncentered, vh_uncentered = torch.linalg.svd(unit, full_matrices=False)
    energy = singular_uncentered.square()
    rank1_energy_ratio = float((energy[0] / energy.sum()).item())
    common_direction = vh_uncentered[0].contiguous()
    if float((unit @ common_direction).mean().item()) < 0.0:
        common_direction = -common_direction

    centered = unit - unit.mean(dim=0, keepdim=True)
    _u_centered, singular_centered, vh_centered = torch.linalg.svd(centered, full_matrices=False)
    centered_energy = singular_centered.square()
    if float(centered_energy.sum().item()) > 0.0:
        centered_ratio = (centered_energy / centered_energy.sum()).tolist()
    else:
        centered_ratio = [0.0 for _ in range(int(singular_centered.numel()))]
    cosine_to_common = {domain: float(torch.dot(unit[idx], common_direction).item()) for idx, domain in enumerate(domains)}
    summary = {
        "domains": list(domains),
        "input": "L2-normalized native mean-difference vectors at the shared intervention layer",
        "common_direction_method": "uncentered SVD / rank-1 energy on unit vectors",
        "rank1_explained_energy_ratio": rank1_energy_ratio,
        "uncentered_singular_values": [float(value) for value in singular_uncentered.tolist()],
        "centered_pca_explained_variance_ratio": [float(value) for value in centered_ratio],
        "cosine_to_common_direction": cosine_to_common,
        "pairwise_cosine": pairwise_cosine(vectors, domains),
        "native_l2_norm": {domain: float(vectors[domain].norm().item()) for domain in domains},
    }
    tensors = {
        "common_direction_unit": common_direction,
        "domain_unit_vectors": unit,
        "centered_components": vh_centered,
        "uncentered_components": vh_uncentered,
    }
    return summary, tensors


def main() -> None:
    args = parse_args()
    domains = unique_domains(args.domains, label="domains")
    source_domains = unique_domains(args.source_domains or domains, label="source-domains")
    target_domains = unique_domains(args.target_domains or domains, label="target-domains")
    # A full matrix naturally has the same domain names in sources and
    # targets.  Keep their first-seen order while taking the required union.
    vector_domains = tuple(dict.fromkeys((*source_domains, *target_domains)))
    if not set(vector_domains).issubset(set(domains)):
        raise ValueError("source-domains and target-domains must be subsets of --domains")
    output_root = args.output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite existing output root: {output_root}")
    dataset_view_root = args.dataset_view_root.resolve()
    if not dataset_view_root.is_dir():
        raise FileNotFoundError(f"Dataset view root not found: {dataset_view_root}")
    for domain in vector_domains:
        if not (dataset_view_root / domain).is_dir():
            raise FileNotFoundError(f"Missing domain view: {dataset_view_root / domain}")

    set_seed(args.seed)
    output_root.mkdir(parents=True, exist_ok=False)
    write_json(
        output_root / "run_config.json",
        {
            "model_label": args.model_label,
            "model_path": str(args.model_path.resolve()),
            "dataset_view_root": str(dataset_view_root),
            "domains": list(domains),
            "source_domains": list(source_domains),
            "target_domains": list(target_domains),
            "vector_domains": list(vector_domains),
            "layer": args.layer,
            "hook_kind": args.hook_kind,
            "train_pairs": args.train_pairs,
            "eval_pairs": args.eval_pairs,
            "batch_size": args.batch_size,
            "conditions": list(CONDITIONS),
            "condition_definitions": {
                "target_norm_aligned": "source direction rescaled to target native L2 norm; alpha=1",
                "raw_1p5": "unmodified source native vector multiplied by alpha=1.5",
            },
        },
    )

    model, _tokenizer, tool_token_id = load_model_and_tokenizer(model_path=args.model_path.resolve(), device=args.device)
    if args.layer < 0 or args.layer >= int(model.cfg.n_layers):
        raise ValueError(f"L{args.layer} is invalid for {args.model_label} ({model.cfg.n_layers} layers)")
    validation: dict[str, dict[str, Any]] = {}
    try:
        vectors, native_summary, train_validation = collect_native_vectors(
            model,
            dataset_view_root=dataset_view_root,
            vector_domains=vector_domains,
            layer=args.layer,
            hook_kind=args.hook_kind,
            train_pairs=args.train_pairs,
            batch_size=args.batch_size,
            output_root=output_root,
        )
        validation.update(train_validation)

        baseline_rows: list[dict[str, Any]] = []
        matrix_rows: list[dict[str, Any]] = []
        for target_domain in target_domains:
            pairs = load_sample_pairs(
                model,
                dataset_root=dataset_view_root / target_domain,
                split="test",
                max_pairs=args.eval_pairs,
            )
            validation[f"{target_domain}/test"] = validate_pairs(
                pairs, domain=target_domain, split="test", expected_count=args.eval_pairs
            )
            baseline, rows = evaluate_target_matrix(
                model,
                pairs,
                target_domain=target_domain,
                vectors=vectors,
                source_domains=source_domains,
                layer=args.layer,
                hook_kind=args.hook_kind,
                batch_size=args.batch_size,
                tool_token_id=tool_token_id,
            )
            baseline_rows.append(baseline)
            matrix_rows.extend(rows)
            write_csv(output_root / "matrix_long.partial.csv", matrix_rows)
            write_csv(output_root / "target_baselines.partial.csv", baseline_rows)
            del pairs
            clear_cuda()

        expected_cells = len(source_domains) * len(target_domains) * len(CONDITIONS)
        if len(matrix_rows) != expected_cells:
            raise AssertionError(f"Expected {expected_cells} cells, found {len(matrix_rows)}")
        pca_summary, pca_tensors = common_direction_pca(vectors, vector_domains)
        torch.save({"domains": list(vector_domains), **pca_tensors}, output_root / "pca_common_direction_L24_pre.pt")
        write_csv(output_root / "matrix_long.csv", matrix_rows)
        write_csv(output_root / "target_baselines.csv", baseline_rows)
        write_json(output_root / "native_vectors.json", native_summary)
        write_json(output_root / "pair_validation.json", validation)
        write_json(output_root / "pca_common_direction.json", pca_summary)
        write_json(
            output_root / "completion.json",
            {
                "status": "complete",
                "expected_cells": expected_cells,
                "observed_cells": len(matrix_rows),
                "tool_token_id": tool_token_id,
                "layer": args.layer,
                "hook_kind": args.hook_kind,
                "model_d_model": int(model.cfg.d_model),
                "model_n_layers": int(model.cfg.n_layers),
            },
        )
        print(json.dumps({"status": "complete", "output_root": str(output_root), "cells": len(matrix_rows)}, ensure_ascii=False))
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
