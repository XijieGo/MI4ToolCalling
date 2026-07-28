#!/usr/bin/env python3
"""Validate the tool-identity ablation in v4 D3--D5 after the D1 gate passes.

This runner deliberately reuses the D1 measurement protocol, not its code-tool
schemas.  Each domain receives an analogous set of V0--V6 schema operations:
an alternate matched affordance, opaque name/description, unchanged name with
unrelated description, a domain-specific affordance reversal, a mismatched
weather tool, and a two-tool condition.  It does not re-screen any variant.

The D1 Write/Review x write_file/submit_review grid remains the primary
affordance-reversal visualization.  Here V4 is a domain-appropriate
validation probe; it is intentionally not pooled as though ``ignore``, SQL
review, and email rating were the same semantic action.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import torch


THIS_DIR = Path(__file__).resolve().parent
SHARED_DIR = THIS_DIR.parent / "shared"
if str(SHARED_DIR) not in sys.path:
    sys.path.insert(0, str(SHARED_DIR))

from multiscale_common import (  # noqa: E402
    clear_cuda,
    load_model_and_tokenizer,
    load_sample_pairs,
    set_seed,
    write_csv,
    write_json,
)
from run_tool_identity_ablation import (  # noqa: E402
    DEFAULT_MODEL_PATH,
    VARIANT_ORDER,
    RenderedPair,
    VariantSpec,
    collect_variant_vector,
    cosine_similarity,
    evaluate_variant,
    extract_schema,
    flatten_metrics,
    render_pairs_for_variant,
    validate_pairs,
)


PROJECT_ROOT = THIS_DIR.parents[1]
DEFAULT_DATASET_VIEW_ROOT = PROJECT_ROOT / "datasets" / "v4_multidomain_balanced"
DEFAULT_DOMAINS = ("D3", "D4", "D5")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate v4 D3--D5 tool-identity ablations after the D1 gate.")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--model-label", type=str, default="Qwen3-8B")
    parser.add_argument("--dataset-view-root", type=Path, default=DEFAULT_DATASET_VIEW_ROOT)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--domains", nargs="+", choices=DEFAULT_DOMAINS, default=list(DEFAULT_DOMAINS))
    parser.add_argument("--layer", type=int, default=24)
    parser.add_argument("--hook-kind", choices=("pre", "post"), default="pre")
    parser.add_argument("--train-pairs", type=int, default=400)
    parser.add_argument("--eval-pairs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--variants", nargs="+", choices=VARIANT_ORDER, default=list(VARIANT_ORDER))
    return parser.parse_args()


def compact_function_schema(name: str, description: str, parameters: dict[str, Any]) -> str:
    payload = {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": parameters,
        },
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def object_parameters(properties: dict[str, dict[str, str]], required: Sequence[str]) -> dict[str, Any]:
    return {"type": "object", "properties": properties, "required": list(required)}


def domain_specs(domain: str, original_schema: str) -> dict[str, VariantSpec]:
    """Build V0--V6 schemas while retaining each real domain's input contract."""

    parsed = json.loads(original_schema)
    function = parsed.get("function")
    if not isinstance(function, dict):
        raise ValueError(f"{domain}: expected one function schema")
    original_name = str(function.get("name", ""))
    parameters = function.get("parameters")
    if not original_name or not isinstance(parameters, dict):
        raise ValueError(f"{domain}: schema lacks name or parameters")

    config: dict[str, dict[str, Any]] = {
        "D3": {
            "alternate_name": "browse_web",
            "alternate_description": "Browse an online source.",
            "reversal_name": "mark_ignored",
            "reversal_description": "Mark a factual claim as ignored.",
            "reversal_parameters": object_parameters(
                {"claim": {"type": "string"}, "reason": {"type": "string"}}, ["claim"]
            ),
            "secondary_name": "explain_claim",
            "secondary_description": "Explain a factual claim.",
            "secondary_parameters": object_parameters(
                {"claim": {"type": "string"}, "explanation": {"type": "string"}}, ["claim"]
            ),
        },
        "D4": {
            "alternate_name": "database_action",
            "alternate_description": "Operate on a database statement.",
            "reversal_name": "submit_review",
            "reversal_description": "Submit a review of a SQL query.",
            "reversal_parameters": object_parameters(
                {"database": {"type": "string"}, "comments": {"type": "string"}}, ["comments"]
            ),
            "secondary_name": "explain_query",
            "secondary_description": "Explain a SQL query.",
            "secondary_parameters": object_parameters(
                {"query": {"type": "string"}, "explanation": {"type": "string"}}, ["query"]
            ),
        },
        "D5": {
            "alternate_name": "deliver_message",
            "alternate_description": "Deliver an email message.",
            "reversal_name": "submit_rating",
            "reversal_description": "Submit a rating for an email message.",
            "reversal_parameters": object_parameters(
                {"recipient": {"type": "string"}, "score": {"type": "integer"}, "comments": {"type": "string"}},
                ["score"],
            ),
            "secondary_name": "review_email",
            "secondary_description": "Review an email message.",
            "secondary_parameters": object_parameters(
                {"message": {"type": "string"}, "comments": {"type": "string"}}, ["message"]
            ),
        },
    }[domain]
    weather_parameters = object_parameters({"location": {"type": "string"}}, ["location"])
    opaque_schema = compact_function_schema("f1", "", parameters)
    unrelated_schema = compact_function_schema(original_name, "Get the current weather forecast.", parameters)
    alternate_schema = compact_function_schema(
        str(config["alternate_name"]), str(config["alternate_description"]), parameters
    )
    reversal_schema = compact_function_schema(
        str(config["reversal_name"]), str(config["reversal_description"]), dict(config["reversal_parameters"])
    )
    weather_schema = compact_function_schema("get_weather", "Get weather.", weather_parameters)
    secondary_schema = compact_function_schema(
        str(config["secondary_name"]), str(config["secondary_description"]), dict(config["secondary_parameters"])
    )
    return {
        "V0": VariantSpec("V0", f"original {original_name}", "Original v4 schema, preserved byte-for-byte.", None),
        "V1": VariantSpec(
            "V1",
            str(config["alternate_name"]),
            "Same domain affordance under an alternate name and description; original parameters are retained.",
            alternate_schema,
        ),
        "V2": VariantSpec(
            "V2",
            "f1 / empty description",
            "Opaque tool name and empty description; original argument structure is retained.",
            opaque_schema,
        ),
        "V3": VariantSpec(
            "V3",
            f"{original_name} / unrelated description",
            "Original name and parameters with unrelated weather description.",
            unrelated_schema,
        ),
        "V4": VariantSpec(
            "V4",
            str(config["reversal_name"]),
            "Domain-specific affordance reversal matched to a no-call-side action, reported separately from D1's 2x2 grid.",
            reversal_schema,
        ),
        "V5": VariantSpec("V5", "get_weather", "A deliberately mismatched weather tool.", weather_schema),
        "V6": VariantSpec(
            "V6",
            f"{original_name} + {config['secondary_name']}",
            "Original action tool plus a domain-specific explanatory/review tool.",
            original_schema + "\n" + secondary_schema,
        ),
    }


def load_fixed_domain_pairs(model, *, dataset_root: Path, split: str, max_pairs: int) -> list[Any]:
    """Load either of the two v4 manifest layouts without changing membership.

    D1 has per-condition ``manifest.jsonl`` files.  The newer D3--D5 release
    stores the same rendered prompt pairs in one root-level
    ``selected_pairs.jsonl`` with an explicit split field.  Both are immutable
    source artifacts; this adapter simply reads the existing representation.
    """

    clean_manifest = dataset_root / split / "clean" / "manifest.jsonl"
    if clean_manifest.exists():
        return load_sample_pairs(model, dataset_root=dataset_root, split=split, max_pairs=max_pairs)
    selected_path = dataset_root / "selected_pairs.jsonl"
    if not selected_path.exists():
        raise FileNotFoundError(
            f"{dataset_root}: neither {clean_manifest.relative_to(dataset_root)} nor selected_pairs.jsonl exists"
        )
    pairs: list[RenderedPair] = []
    with selected_path.open("r", encoding="utf-8") as handle:
        for row in (json.loads(line) for line in handle if line.strip()):
            if str(row.get("split")) != split:
                continue
            clean_text = str(row.get("clean_prompt", ""))
            corrupt_text = str(row.get("corrupt_prompt", ""))
            if not clean_text or not corrupt_text:
                raise ValueError(f"{dataset_root}: selected pair lacks rendered prompts")
            clean_tokens = model.to_tokens(clean_text, prepend_bos=False).detach().cpu()
            corrupt_tokens = model.to_tokens(corrupt_text, prepend_bos=False).detach().cpu()
            if int(clean_tokens.shape[-1]) != int(corrupt_tokens.shape[-1]):
                raise ValueError(f"{dataset_root}/{row.get('candidate_id')}: non-aligned rendered pair")
            pairs.append(
                RenderedPair(
                    order=len(pairs) + 1,
                    sample_id=str(row.get("candidate_id") or row.get("source_id") or len(pairs) + 1),
                    clean_text=clean_text,
                    corrupt_text=corrupt_text,
                    clean_tokens_cpu=clean_tokens,
                    corrupt_tokens_cpu=corrupt_tokens,
                    token_len=int(clean_tokens.shape[-1]),
                    clean_candidate=str(row.get("clean_verb")) if row.get("clean_verb") is not None else None,
                    corrupt_candidate=str(row.get("corrupt_verb")) if row.get("corrupt_verb") is not None else None,
                )
            )
            if len(pairs) >= int(max_pairs):
                break
    if not pairs:
        raise ValueError(f"{dataset_root}: no {split} pairs in selected_pairs.jsonl")
    return pairs


def run_domain(
    model,
    *,
    domain: str,
    dataset_root: Path,
    output_root: Path,
    variants: Sequence[str],
    layer: int,
    hook_kind: str,
    train_pairs: int,
    eval_pairs: int,
    batch_size: int,
    tool_token_id: int,
) -> dict[str, Any]:
    domain_root = output_root / domain
    domain_root.mkdir(parents=True, exist_ok=False)
    base_train = load_fixed_domain_pairs(model, dataset_root=dataset_root, split="train", max_pairs=train_pairs)
    base_test = load_fixed_domain_pairs(model, dataset_root=dataset_root, split="test", max_pairs=eval_pairs)
    if len(base_train) != int(train_pairs) or len(base_test) != int(eval_pairs):
        raise ValueError(f"{domain}: dataset did not yield requested fixed 400/100 cardinalities")
    original_schema = extract_schema(base_train[0].clean_text)
    for pair in (*base_train, *base_test):
        if extract_schema(pair.clean_text) != original_schema or extract_schema(pair.corrupt_text) != original_schema:
            raise ValueError(f"{domain}/{pair.sample_id}: original schema is not invariant")
    specs = domain_specs(domain, original_schema)
    write_json(
        domain_root / "variants.json",
        {
            key: {
                "label": specs[key].label,
                "description": specs[key].description,
                "schema": specs[key].schema,
            }
            for key in variants
        },
    )

    pair_validation: dict[str, Any] = {}
    vectors: dict[str, torch.Tensor] = {}
    vector_summaries: dict[str, dict[str, Any]] = {}
    vector_root = domain_root / "native_vectors"
    vector_root.mkdir(parents=True, exist_ok=True)
    for key in variants:
        rendered_train = render_pairs_for_variant(model, base_train, variant=specs[key])
        pair_validation[f"{key}/train"] = validate_pairs(
            rendered_train, expected_count=train_pairs, label=f"{domain}/{key}/train"
        )
        vector, pca = collect_variant_vector(
            model,
            rendered_train,
            layer=layer,
            hook_kind=hook_kind,
            batch_size=batch_size,
        )
        bundle_path = vector_root / f"{key}_L{layer}_{hook_kind}.pt"
        torch.save(
            {
                "mean_diff": vector,
                "components": pca["components"].detach().cpu(),
                "explained_variance": pca["explained_variance"].detach().cpu(),
                "explained_variance_ratio": pca["explained_variance_ratio"].detach().cpu(),
                "singular_values": pca["singular_values"].detach().cpu(),
                "domain": domain,
                "variant": key,
                "train_sample_ids": [pair.sample_id for pair in rendered_train],
                "layer": layer,
                "hook_kind": hook_kind,
                "model_d_model": int(model.cfg.d_model),
            },
            bundle_path,
        )
        vectors[key] = vector
        vector_summaries[key] = {
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

    frozen_v0 = vectors["V0"].detach().cpu().float().contiguous()
    for key, vector in vectors.items():
        vector_summaries[key]["cosine_to_frozen_v0"] = cosine_similarity(vector, frozen_v0)
        vector_summaries[key]["l2_norm_over_v0"] = float(vector.norm().item() / frozen_v0.norm().item())
    write_json(domain_root / "native_vectors.json", vector_summaries)

    behavior_rows: list[dict[str, Any]] = []
    intervention_rows: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []
    detailed_results: dict[str, Any] = {}
    for key in variants:
        rendered_test = render_pairs_for_variant(model, base_test, variant=specs[key])
        pair_validation[f"{key}/test"] = validate_pairs(
            rendered_test, expected_count=eval_pairs, label=f"{domain}/{key}/test"
        )
        result = evaluate_variant(
            model,
            rendered_test,
            variant=key,
            layer=layer,
            hook_kind=hook_kind,
            frozen_v0=frozen_v0,
            batch_size=batch_size,
            tool_token_id=tool_token_id,
            sample_rows=sample_rows,
        )
        detailed_results[key] = result
        clean = result["baseline"]["clean"]
        corrupt = result["baseline"]["corrupt"]
        intervention = result["frozen_v0_intervention"]
        behavior_rows.extend(
            [
                {"domain": domain, "variant": key, "side": "clean", **flatten_metrics("behavior", clean)},
                {"domain": domain, "variant": key, "side": "corrupt", **flatten_metrics("behavior", corrupt)},
            ]
        )
        intervention_rows.append(
            {
                "domain": domain,
                "variant": key,
                "cosine_to_frozen_v0": vector_summaries[key]["cosine_to_frozen_v0"],
                "native_l2_norm": vector_summaries[key]["l2_norm"],
                "native_l2_norm_over_v0": vector_summaries[key]["l2_norm_over_v0"],
                "baseline_clean_top1_rate": clean["tool_call_top1_rate"],
                "baseline_corrupt_top1_rate": corrupt["tool_call_top1_rate"],
                "baseline_clean_mean_tool_prob": clean["mean_tool_call_prob"],
                "baseline_corrupt_mean_tool_prob": corrupt["mean_tool_call_prob"],
                "condition_logit_gap_clean_minus_corrupt": intervention["condition_logit_gap_clean_minus_corrupt"],
                "normalization_valid": intervention["normalization_valid"],
                "frozen_v0_add_top1_rate": intervention["add_to_corrupt"]["tool_call_top1_rate"],
                "frozen_v0_add_mean_tool_prob": intervention["add_to_corrupt"]["mean_tool_call_prob"],
                "frozen_v0_add_strict_flip_rate": intervention["add_strict_flip_rate"],
                "frozen_v0_sufficiency": intervention["sufficiency_normalized_logit_gap"],
                "frozen_v0_remove_remaining_top1_rate": intervention["remove_from_clean"]["tool_call_top1_rate"],
                "frozen_v0_remove_mean_tool_prob": intervention["remove_from_clean"]["mean_tool_call_prob"],
                "frozen_v0_remove_strict_drop_rate": intervention["remove_strict_drop_rate"],
                "frozen_v0_necessity": intervention["necessity_normalized_logit_gap"],
            }
        )
        write_csv(domain_root / "behavior_long.partial.csv", behavior_rows)
        write_csv(domain_root / "intervention_long.partial.csv", intervention_rows)
        write_csv(domain_root / "sample_metrics.partial.csv", sample_rows)
        del rendered_test
        clear_cuda()

    write_json(domain_root / "pair_validation.json", pair_validation)
    write_json(domain_root / "tool_identity_results.json", detailed_results)
    write_csv(domain_root / "behavior_long.csv", behavior_rows)
    write_csv(domain_root / "intervention_long.csv", intervention_rows)
    write_csv(domain_root / "sample_metrics.csv", sample_rows)
    summary = {
        "domain": domain,
        "dataset_cardinality": {"train": len(base_train), "test": len(base_test)},
        "variants": list(variants),
        "observed": {
            row["variant"]: {
                "cosine_to_frozen_v0": row["cosine_to_frozen_v0"],
                "frozen_v0_sufficiency": row["frozen_v0_sufficiency"],
                "frozen_v0_necessity": row["frozen_v0_necessity"],
                "normalization_valid": row["normalization_valid"],
            }
            for row in intervention_rows
        },
    }
    write_json(domain_root / "domain_validation_summary.json", summary)
    write_json(
        domain_root / "completion.json",
        {
            "status": "complete",
            "domain": domain,
            "dataset_cardinality": {"train": len(base_train), "test": len(base_test)},
            "variants": list(variants),
            "layer": layer,
            "hook_kind": hook_kind,
            "tool_token_id": tool_token_id,
        },
    )
    del base_train, base_test, vectors, frozen_v0
    clear_cuda()
    return summary


def main() -> None:
    args = parse_args()
    domains = tuple(args.domains)
    variants = tuple(args.variants)
    if "V0" not in variants:
        raise ValueError("V0 is required because every intervention freezes its domain's original vector")
    if len(set(domains)) != len(domains) or len(set(variants)) != len(variants):
        raise ValueError("Duplicate domains or variants are not allowed")
    output_root = args.output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite existing output root: {output_root}")
    dataset_view_root = args.dataset_view_root.resolve()
    for domain in domains:
        if not (dataset_view_root / domain).is_dir():
            raise FileNotFoundError(f"Missing dataset view for {domain}: {dataset_view_root / domain}")
    output_root.mkdir(parents=True, exist_ok=False)
    write_json(
        output_root / "run_config.json",
        {
            "experiment": "v4_tool_identity_ablation_remaining_domains",
            "dataset_view_root": str(dataset_view_root),
            "domains": list(domains),
            "model_path": str(args.model_path.resolve()),
            "model_label": args.model_label,
            "layer": args.layer,
            "hook_kind": args.hook_kind,
            "train_pairs": args.train_pairs,
            "eval_pairs": args.eval_pairs,
            "batch_size": args.batch_size,
            "seed": args.seed,
            "variants": list(variants),
            "selection_rule": "No schema-variant behavioral re-screening or re-selection.",
            "scope_note": "D1 is the primary Write/Review affordance-reversal grid; D3--D5 validate the fixed-vector tool-identity result with domain-specific schemas.",
        },
    )
    set_seed(args.seed)
    model, _tokenizer, tool_token_id = load_model_and_tokenizer(
        model_path=args.model_path.resolve(), device=args.device
    )
    if args.layer < 0 or args.layer >= int(model.cfg.n_layers):
        raise ValueError(f"L{args.layer} is invalid for this model ({model.cfg.n_layers} layers)")
    try:
        summaries = []
        for domain in domains:
            summaries.append(
                run_domain(
                    model,
                    domain=domain,
                    dataset_root=dataset_view_root / domain,
                    output_root=output_root,
                    variants=variants,
                    layer=args.layer,
                    hook_kind=args.hook_kind,
                    train_pairs=args.train_pairs,
                    eval_pairs=args.eval_pairs,
                    batch_size=args.batch_size,
                    tool_token_id=tool_token_id,
                )
            )
            write_json(output_root / "domain_summaries.partial.json", {row["domain"]: row for row in summaries})
        write_json(output_root / "domain_summaries.json", {row["domain"]: row for row in summaries})
        write_json(
            output_root / "completion.json",
            {
                "status": "complete",
                "domains": list(domains),
                "variants": list(variants),
                "layer": args.layer,
                "hook_kind": args.hook_kind,
                "tool_token_id": tool_token_id,
            },
        )
        print(json.dumps({"status": "complete", "output_root": str(output_root), "domains": list(domains)}, ensure_ascii=False))
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
