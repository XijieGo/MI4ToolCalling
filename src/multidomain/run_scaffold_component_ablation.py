#!/usr/bin/env python3
"""Run rebuttal §3: R/T/F scaffold-component ablation on one fixed release.

The required design is the full 2^3 role-instructions (R), tool-schema (T),
and format-template (F) factorial crossed with execution, neutral, and
analysis requests.  A ninth, supplemental R/T-length-matched-neutral/F cell
tests whether deleting the schema merely changes prompt length. Every vector
is estimated on the fixed train split; every behavior and causal intervention
is evaluated on the fixed held-out split.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch


THIS_DIR = Path(__file__).resolve().parent
SHARED_DIR = THIS_DIR.parent / "shared"
if str(SHARED_DIR) not in sys.path:
    sys.path.insert(0, str(SHARED_DIR))

from multiscale_common import clear_cuda, load_model_and_tokenizer, set_seed, write_csv, write_json  # noqa: E402
from scaffold_ablation_common import (  # noqa: E402
    PromptItem,
    balanced_neutral_assignment,
    build_length_matched_neutral_block,
    collect_variant_vector,
    cosine_similarity,
    evaluate_frozen_vector_causal,
    evaluate_prompt_items,
    flatten_metrics,
    is_v5_model_specific_dataset,
    load_fixed_pairs,
    parse_prompt_parts,
    render_pair_for_scaffold,
    replace_leading_user_verb,
    validate_fixed_pairs,
    validate_invariant_scaffold,
    validate_neutral_tokenization,
    validate_rendered_pairs,
    validate_v5_model_compatibility,
    validate_v5_requested_split_sizes,
    v5_dataset_provenance,
)


PROJECT_ROOT = THIS_DIR.parents[1]
DEFAULT_DATASET_VIEW_ROOT = PROJECT_ROOT / "datasets" / "v4_multidomain_balanced"
DEFAULT_MODEL_PATH = Path(os.environ.get("QWEN3_8B_PATH", PROJECT_ROOT / "external" / "models" / "Qwen3-8B")).expanduser()
DOMAIN_CHOICES = ("D1", "D3", "D4", "D5")


@dataclass(frozen=True)
class ScaffoldSpec:
    key: str
    label: str
    include_role: bool
    include_tool_schema: bool
    include_format: bool
    length_matched_control: bool = False

    @property
    def tool_state(self) -> str:
        if self.length_matched_control:
            return "length_matched_neutral_text"
        return "present" if self.include_tool_schema else "removed"


CORE_SPECS = (
    ScaffoldSpec("RTF", "R + T + F (full scaffold)", True, True, True),
    ScaffoldSpec("-TF", "T + F (no role instructions)", False, True, True),
    ScaffoldSpec("R-F", "R + F (no tool schema)", True, False, True),
    ScaffoldSpec("RT-", "R + T (no format template)", True, True, False),
    ScaffoldSpec("--F", "F only", False, False, True),
    ScaffoldSpec("-T-", "T only", False, True, False),
    ScaffoldSpec("R--", "R only", True, False, False),
    ScaffoldSpec("---", "empty system scaffold", False, False, False),
)
LENGTH_MATCHED_SPEC = ScaffoldSpec(
    "R_TLEN_F",
    "R + length-matched neutral text + F",
    True,
    False,
    True,
    length_matched_control=True,
)
ALL_SPECS = CORE_SPECS + (LENGTH_MATCHED_SPEC,)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", choices=DOMAIN_CHOICES, help="Legacy v4 domain under --dataset-view-root.")
    parser.add_argument("--dataset-view-root", type=Path, default=DEFAULT_DATASET_VIEW_ROOT)
    parser.add_argument("--dataset-root", type=Path, help="Direct paired dataset root, including a v5 model directory.")
    parser.add_argument("--dataset-label", type=str, help="Result label for a direct dataset root.")
    parser.add_argument("--eval-split", choices=("test", "heldout"), help="Defaults to heldout for v5 and test otherwise.")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--model-label", type=str, default="Qwen3-8B")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=24)
    parser.add_argument("--hook-kind", choices=("pre", "post"), default="pre")
    parser.add_argument("--train-pairs", type=int, default=400)
    parser.add_argument("--eval-pairs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def resolve_dataset_input(args: argparse.Namespace) -> tuple[Path, str, str, bool]:
    if args.dataset_root is not None and args.domain is not None:
        raise ValueError("Specify either --dataset-root or --domain, not both")
    if args.dataset_root is None and args.domain is None:
        raise ValueError("Specify --dataset-root for a direct release or --domain for a legacy v4 view")
    dataset_root = args.dataset_root.resolve() if args.dataset_root is not None else (args.dataset_view_root / args.domain).resolve()
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")
    is_v5 = is_v5_model_specific_dataset(dataset_root)
    eval_split = args.eval_split or ("heldout" if is_v5 else "test")
    dataset_label = args.dataset_label or args.domain or dataset_root.name
    return dataset_root, dataset_label, eval_split, is_v5


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def scaffold_metadata(spec: ScaffoldSpec) -> dict[str, Any]:
    return {
        "scaffold": spec.key,
        "scaffold_label": spec.label,
        "role_instructions": spec.include_role,
        "tool_schema": spec.tool_state,
        "format_template": spec.include_format,
        "length_matched_control": spec.length_matched_control,
    }


def render_spec_pairs(model, pairs, spec: ScaffoldSpec, *, neutral_block: str | None) -> list:
    rendered = []
    for pair in pairs:
        clean_parts = parse_prompt_parts(pair.clean_text)
        corrupt_parts = parse_prompt_parts(pair.corrupt_text)
        rendered.append(
            render_pair_for_scaffold(
                model,
                pair,
                clean_parts=clean_parts,
                corrupt_parts=corrupt_parts,
                include_role=spec.include_role,
                include_tool_schema=spec.include_tool_schema,
                include_format=spec.include_format,
                tool_schema_override=neutral_block if spec.length_matched_control else None,
            )
        )
    return rendered


def validate_length_matched_control(original_pairs, rendered_pairs, *, label: str) -> dict[str, Any]:
    if len(original_pairs) != len(rendered_pairs):
        raise ValueError(f"{label}: source and rendered cardinalities differ")
    length_deltas: set[int] = set()
    for source, rendered in zip(original_pairs, rendered_pairs):
        clean_delta = int(rendered.clean_tokens_cpu.shape[-1] - source.clean_tokens_cpu.shape[-1])
        corrupt_delta = int(rendered.corrupt_tokens_cpu.shape[-1] - source.corrupt_tokens_cpu.shape[-1])
        length_deltas.update((clean_delta, corrupt_delta))
        if clean_delta != 0 or corrupt_delta != 0:
            raise ValueError(f"{label}/{source.sample_id}: length-matched control changed full prompt length")
    return {"full_prompt_token_length_deltas": sorted(length_deltas), "all_exact": length_deltas == {0}}


def build_neutral_items(model, pairs, spec: ScaffoldSpec, *, neutral_block: str | None) -> list[PromptItem]:
    assignment = balanced_neutral_assignment(pairs)
    items: list[PromptItem] = []
    for pair in pairs:
        parts = parse_prompt_parts(pair.clean_text)
        neutral_verb = assignment[pair.sample_id]
        user_content = replace_leading_user_verb(parts.user_content, neutral_verb)
        prompt = parts.render(
            user_content=user_content,
            include_role=spec.include_role,
            include_tool_schema=spec.include_tool_schema,
            include_format=spec.include_format,
            tool_schema_override=neutral_block if spec.length_matched_control else None,
        )
        tokens = model.to_tokens(prompt, prepend_bos=False).detach().cpu()
        items.append(
            PromptItem(
                sample_id=pair.sample_id,
                prompt=prompt,
                tokens_cpu=tokens,
                token_len=int(tokens.shape[-1]),
                metadata={
                    **scaffold_metadata(spec),
                    "condition": "neutral",
                    "request_type": "neutral_verb_task",
                    "neutral_verb": neutral_verb,
                    "prompt_sha256": sha256_text(prompt),
                },
            )
        )
    return items


def save_vector_bundle(
    path: Path,
    *,
    vector: torch.Tensor,
    pca: dict[str, torch.Tensor],
    domain: str,
    spec: ScaffoldSpec,
    sample_ids: Sequence[str],
    layer: int,
    hook_kind: str,
    model_d_model: int,
) -> None:
    torch.save(
        {
            "mean_diff": vector,
            "components": pca["components"].detach().cpu(),
            "explained_variance": pca["explained_variance"].detach().cpu(),
            "explained_variance_ratio": pca["explained_variance_ratio"].detach().cpu(),
            "singular_values": pca["singular_values"].detach().cpu(),
            "domain": domain,
            "scaffold": spec.key,
            "scaffold_label": spec.label,
            "train_sample_ids": list(sample_ids),
            "layer": layer,
            "hook_kind": hook_kind,
            "model_d_model": model_d_model,
        },
        path,
    )


def plot_core_results(behavior_rows: list[dict[str, Any]], derived_rows: list[dict[str, Any]], path: Path) -> None:
    core_keys = [spec.key for spec in CORE_SPECS]
    request_order = ("execution", "neutral", "analysis")
    matrix = np.array(
        [
            [
                next(
                    float(row["behavior_mean_tool_call_prob"])
                    for row in behavior_rows
                    if row["scaffold"] == scaffold and row["request_type"] == request
                )
                for request in request_order
            ]
            for scaffold in core_keys
        ]
    )
    default_strength = [
        next(float(row["default_strength_p_call"]) for row in derived_rows if row["scaffold"] == key)
        for key in core_keys
    ]
    suppression_depth = [
        next(float(row["suppression_depth_p_call"]) for row in derived_rows if row["scaffold"] == key)
        for key in core_keys
    ]
    fig, (heat_ax, bar_ax) = plt.subplots(1, 2, figsize=(12.5, 5.2), gridspec_kw={"width_ratios": [1.15, 1.0]})
    image = heat_ax.imshow(matrix, aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
    heat_ax.set_xticks(range(len(request_order)), request_order)
    heat_ax.set_yticks(range(len(core_keys)), core_keys)
    heat_ax.set_xlabel("request type")
    heat_ax.set_ylabel("scaffold components")
    heat_ax.set_title("p(<tool_call>) across the R/T/F factorial")
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            value = matrix[row_index, column_index]
            heat_ax.text(
                column_index,
                row_index,
                f"{value:.2g}",
                ha="center",
                va="center",
                color="white" if value < 0.45 else "black",
                fontsize=8,
            )
    fig.colorbar(image, ax=heat_ax, fraction=0.046, pad=0.04)

    x_values = np.arange(len(core_keys))
    width = 0.38
    bar_ax.bar(x_values - width / 2, default_strength, width, label="D(S): neutral p_call", color="#4c78a8")
    bar_ax.bar(x_values + width / 2, suppression_depth, width, label="P(S): neutral − analysis", color="#e45756")
    bar_ax.set_xticks(x_values, core_keys, rotation=35, ha="right")
    bar_ax.set_ylim(min(-0.05, min(suppression_depth, default=0.0) - 0.05), 1.05)
    bar_ax.set_ylabel("probability difference")
    bar_ax.set_title("Default strength and suppression depth")
    bar_ax.grid(axis="y", alpha=0.22)
    bar_ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_root = args.output_root.resolve()
    dataset_root, dataset_label, eval_split, is_v5 = resolve_dataset_input(args)
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite existing result directory: {output_root}")
    if int(args.train_pairs) <= 0 or int(args.eval_pairs) <= 0:
        raise ValueError("--train-pairs and --eval-pairs must both be positive")
    provenance = v5_dataset_provenance(dataset_root, splits=("train", eval_split)) if is_v5 else None
    if provenance is not None:
        validate_v5_requested_split_sizes(
            provenance, requested_counts={"train": args.train_pairs, eval_split: args.eval_pairs}
        )
    output_root.mkdir(parents=True, exist_ok=False)
    write_json(
        output_root / "run_config.json",
        {
            "experiment": "scaffold_component_factorial",
            "dataset_label": dataset_label,
            "dataset_root": str(dataset_root),
            "train_split": "train",
            "eval_split": eval_split,
            "model_path": str(args.model_path.resolve()),
            "model_label": args.model_label,
            "layer": args.layer,
            "hook_kind": args.hook_kind,
            "train_pairs": args.train_pairs,
            "eval_pairs": args.eval_pairs,
            "batch_size": args.batch_size,
            "seed": args.seed,
            "core_scaffolds": [spec.key for spec in CORE_SPECS],
            "supplemental_length_matched_control": LENGTH_MATCHED_SPEC.key,
            "request_types": ["execution", "neutral", "analysis"],
            "membership_rule": "Fixed released split; no scaffold-variant behavioral re-screening or re-selection.",
            "frozen_vector": "RTF native mean(clean L24 pre residual − analysis L24 pre residual) on the fixed train split.",
            "alternate_templates": "Not part of the required 2^3 core; intentionally deferred until the main factorial is complete.",
        },
    )
    if provenance is not None:
        write_json(output_root / "dataset_provenance.json", provenance)

    set_seed(args.seed)
    model, tokenizer, tool_token_id = load_model_and_tokenizer(model_path=args.model_path.resolve(), device=args.device)
    try:
        if args.layer < 0 or args.layer >= int(model.cfg.n_layers):
            raise ValueError(f"L{args.layer} is invalid for this model ({model.cfg.n_layers} layers)")
        if provenance is not None:
            validate_v5_model_compatibility(
                provenance, model_path=args.model_path, model_label=args.model_label, tool_token_id=tool_token_id
            )
        train_pairs = load_fixed_pairs(model, dataset_root=dataset_root, split="train", max_pairs=args.train_pairs)
        test_pairs = load_fixed_pairs(model, dataset_root=dataset_root, split=eval_split, max_pairs=args.eval_pairs)
        fixed_validation = {
            "train": validate_fixed_pairs(train_pairs, expected_count=args.train_pairs, label=f"{dataset_label}/train"),
            eval_split: validate_fixed_pairs(test_pairs, expected_count=args.eval_pairs, label=f"{dataset_label}/{eval_split}"),
        }
        reference_parts = validate_invariant_scaffold(train_pairs + test_pairs, label=f"{dataset_label}/all_fixed")
        neutral_protocol = validate_neutral_tokenization(tokenizer)
        length_control = build_length_matched_neutral_block(model, reference_parts)
        neutral_block = str(length_control["block"])
        write_json(
            output_root / "scaffold_components.json",
            {
                "role_instructions": reference_parts.role_instructions,
                "tool_schema_block": reference_parts.tool_schema_block,
                "format_template": reference_parts.format_template,
                "role_instructions_sha256": sha256_text(reference_parts.role_instructions),
                "tool_schema_block_sha256": sha256_text(reference_parts.tool_schema_block),
                "format_template_sha256": sha256_text(reference_parts.format_template),
                "component_definition": {
                    "R": "All system text before <tools>, excluding the system-role wrapper.",
                    "T": "The complete <tools> schema block, including its trailing separators.",
                    "F": "The complete function-call output-format instruction and literal <tool_call> example.",
                    "system_role_wrapper": "Retained in all 2^3 cells so the model receives a valid native chat prompt; L7 of the separate request ladder removes it.",
                },
                "length_matched_control": length_control,
            },
        )

        pair_validation: dict[str, Any] = {"fixed": fixed_validation}
        vector_root = output_root / "native_vectors"
        vector_root.mkdir(parents=True, exist_ok=True)
        vectors: dict[str, torch.Tensor] = {}
        vector_summaries: dict[str, dict[str, Any]] = {}
        for spec in ALL_SPECS:
            rendered_train = render_spec_pairs(model, train_pairs, spec, neutral_block=neutral_block)
            pair_validation[f"{spec.key}/train"] = validate_rendered_pairs(
                rendered_train, expected_count=args.train_pairs, label=f"{dataset_label}/{spec.key}/train"
            )
            if spec.length_matched_control:
                pair_validation[f"{spec.key}/train_length_control"] = validate_length_matched_control(
                    train_pairs, rendered_train, label=f"{dataset_label}/{spec.key}/train"
                )
            if spec.key == "RTF" and any(
                rendered.clean_text != source.clean_text or rendered.corrupt_text != source.corrupt_text
                for rendered, source in zip(rendered_train, train_pairs)
            ):
                raise AssertionError("RTF did not reconstruct the original fixed train prompts byte-for-byte")
            vector, pca = collect_variant_vector(
                model,
                rendered_train,
                layer=args.layer,
                hook_kind=args.hook_kind,
                batch_size=args.batch_size,
            )
            bundle_path = vector_root / f"{spec.key}_L{args.layer}_{args.hook_kind}.pt"
            save_vector_bundle(
                bundle_path,
                vector=vector,
                pca=pca,
                domain=dataset_label,
                spec=spec,
                sample_ids=[pair.sample_id for pair in rendered_train],
                layer=args.layer,
                hook_kind=args.hook_kind,
                model_d_model=int(model.cfg.d_model),
            )
            vectors[spec.key] = vector
            vector_summaries[spec.key] = {
                **scaffold_metadata(spec),
                "bundle": str(bundle_path),
                "l2_norm": float(vector.norm().item()),
                "rms": float(vector.square().mean().sqrt().item()),
                "train_pair_count": len(rendered_train),
                "pca_explained_variance_ratio": [
                    float(value) for value in pca["explained_variance_ratio"].detach().cpu().tolist()
                ],
            }
            del rendered_train, vector, pca
            clear_cuda()

        frozen_rtf = vectors["RTF"].detach().cpu().float().contiguous()
        for spec in ALL_SPECS:
            vector_summaries[spec.key]["cosine_to_frozen_RTF"] = cosine_similarity(vectors[spec.key], frozen_rtf)
            vector_summaries[spec.key]["l2_norm_over_RTF"] = float(vectors[spec.key].norm().item() / frozen_rtf.norm().item())
        write_json(output_root / "native_vectors.json", vector_summaries)

        behavior_rows: list[dict[str, Any]] = []
        derived_rows: list[dict[str, Any]] = []
        intervention_rows: list[dict[str, Any]] = []
        sample_rows: list[dict[str, Any]] = []
        detailed_results: dict[str, Any] = {}
        for spec in ALL_SPECS:
            rendered_test = render_spec_pairs(model, test_pairs, spec, neutral_block=neutral_block)
            pair_validation[f"{spec.key}/{eval_split}"] = validate_rendered_pairs(
                rendered_test, expected_count=args.eval_pairs, label=f"{dataset_label}/{spec.key}/{eval_split}"
            )
            if spec.length_matched_control:
                pair_validation[f"{spec.key}/{eval_split}_length_control"] = validate_length_matched_control(
                    test_pairs, rendered_test, label=f"{dataset_label}/{spec.key}/{eval_split}"
                )
            if spec.key == "RTF" and any(
                rendered.clean_text != source.clean_text or rendered.corrupt_text != source.corrupt_text
                for rendered, source in zip(rendered_test, test_pairs)
            ):
                raise AssertionError("RTF did not reconstruct the original fixed test prompts byte-for-byte")
            causal, causal_rows = evaluate_frozen_vector_causal(
                model,
                rendered_test,
                layer=args.layer,
                hook_kind=args.hook_kind,
                frozen_vector=frozen_rtf,
                batch_size=args.batch_size,
                tool_token_id=tool_token_id,
                scaffold_label=f"{dataset_label}/{spec.key}",
            )
            neutral_items = build_neutral_items(model, test_pairs, spec, neutral_block=neutral_block)
            neutral_metrics_map, neutral_rows = evaluate_prompt_items(
                model,
                neutral_items,
                batch_size=args.batch_size,
                tool_token_id=tool_token_id,
                progress_label=f"{dataset_label}/{spec.key}: neutral requests",
            )
            neutral = neutral_metrics_map["neutral"]
            execution = causal["baseline"]["execution"]
            analysis = causal["baseline"]["analysis"]
            intervention = causal["frozen_original_intervention"]
            metadata = {"dataset_label": dataset_label, **scaffold_metadata(spec)}
            for request_type, metrics in (("execution", execution), ("neutral", neutral), ("analysis", analysis)):
                behavior_rows.append({**metadata, "request_type": request_type, **flatten_metrics("behavior", metrics)})
            derived_rows.append(
                {
                    **metadata,
                    "default_strength_p_call": neutral["mean_tool_call_prob"],
                    "suppression_depth_p_call": neutral["mean_tool_call_prob"] - analysis["mean_tool_call_prob"],
                    "execution_minus_analysis_p_call": execution["mean_tool_call_prob"] - analysis["mean_tool_call_prob"],
                    "neutral_minus_execution_p_call": neutral["mean_tool_call_prob"] - execution["mean_tool_call_prob"],
                }
            )
            intervention_rows.append(
                {
                    **metadata,
                    "cosine_to_frozen_RTF": vector_summaries[spec.key]["cosine_to_frozen_RTF"],
                    "native_l2_norm": vector_summaries[spec.key]["l2_norm"],
                    "native_l2_norm_over_RTF": vector_summaries[spec.key]["l2_norm_over_RTF"],
                    "baseline_execution_top1_rate": execution["tool_call_top1_rate"],
                    "baseline_analysis_top1_rate": analysis["tool_call_top1_rate"],
                    "baseline_execution_mean_tool_prob": execution["mean_tool_call_prob"],
                    "baseline_analysis_mean_tool_prob": analysis["mean_tool_call_prob"],
                    "condition_logit_gap_execution_minus_analysis": intervention[
                        "condition_logit_gap_execution_minus_analysis"
                    ],
                    "normalization_valid": intervention["normalization_valid"],
                    "frozen_RTF_add_top1_rate": intervention["add_to_analysis"]["tool_call_top1_rate"],
                    "frozen_RTF_add_mean_tool_prob": intervention["add_to_analysis"]["mean_tool_call_prob"],
                    "frozen_RTF_add_strict_flip_rate": intervention["add_strict_flip_rate"],
                    "frozen_RTF_sufficiency": intervention["sufficiency_normalized_logit_gap"],
                    "frozen_RTF_remove_remaining_top1_rate": intervention["remove_from_execution"]["tool_call_top1_rate"],
                    "frozen_RTF_remove_mean_tool_prob": intervention["remove_from_execution"]["mean_tool_call_prob"],
                    "frozen_RTF_remove_strict_drop_rate": intervention["remove_strict_drop_rate"],
                    "frozen_RTF_necessity": intervention["necessity_normalized_logit_gap"],
                }
            )
            for row in causal_rows:
                row.update(metadata)
            for row in neutral_rows:
                row["dataset_label"] = dataset_label
            sample_rows.extend(causal_rows)
            sample_rows.extend(neutral_rows)
            detailed_results[spec.key] = {
                **metadata,
                "baseline": {"execution": execution, "neutral": neutral, "analysis": analysis},
                "derived": derived_rows[-1],
                "frozen_original_intervention": intervention,
            }
            write_csv(output_root / "behavior_long.partial.csv", behavior_rows)
            write_csv(output_root / "derived_default_suppression.partial.csv", derived_rows)
            write_csv(output_root / "intervention_long.partial.csv", intervention_rows)
            write_csv(output_root / "sample_metrics.partial.csv", sample_rows)
            del rendered_test, neutral_items, causal_rows, neutral_rows
            clear_cuda()

        pre_registered_test = {
            "prediction": "F carries the literal token-level prior; T carries task-affordance relevance; R is weakest.",
            "falsifier": "If removing any component leaves D(S)=p_call(neutral) unchanged, the default is not installed by that component.",
            "primary_quantities": {
                "default_strength": "D(S)=p_call(neutral)",
                "suppression_depth": "P(S)=p_call(neutral)-p_call(analysis)",
            },
        }
        write_json(output_root / "pair_validation.json", pair_validation)
        write_json(output_root / "neutral_verb_protocol.json", neutral_protocol)
        write_json(output_root / "scaffold_component_results.json", detailed_results)
        write_json(output_root / "pre_registered_interpretation.json", pre_registered_test)
        write_csv(output_root / "behavior_long.csv", behavior_rows)
        write_csv(output_root / "derived_default_suppression.csv", derived_rows)
        write_csv(output_root / "intervention_long.csv", intervention_rows)
        write_csv(output_root / "sample_metrics.csv", sample_rows)
        write_json(
            output_root / "completion.json",
            {
                "status": "complete",
                "dataset_label": dataset_label,
                "dataset_cardinality": {"train": len(train_pairs), eval_split: len(test_pairs)},
                "tool_token_id": tool_token_id,
                "layer": args.layer,
                "hook_kind": args.hook_kind,
                "core_scaffolds": [spec.key for spec in CORE_SPECS],
                "supplemental_length_matched_control": LENGTH_MATCHED_SPEC.key,
            },
        )
        # Preserve numerical artifacts before optional plotting; the run is
        # auditable even if a headless Matplotlib backend fails afterward.
        plot_core_results(behavior_rows, derived_rows, output_root / "scaffold_factorial.pdf")
        print(json.dumps({"status": "complete", "output_root": str(output_root), "dataset_label": dataset_label}, ensure_ascii=False))
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
