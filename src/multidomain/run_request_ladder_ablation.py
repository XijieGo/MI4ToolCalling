#!/usr/bin/env python3
"""Run rebuttal §1 (Request Ladder) on one fixed released dataset split.

Only the user turn changes for L0--L6.  L7 removes the complete system
scaffold while retaining Qwen's native user/assistant chat boundary, so it is
the requested floor control rather than an invalid raw-text prompt.  The
script reports all four pre-registered first-token metrics and explicitly
marks L0--L2 as one unique literal prompt each; it does not treat 100 copies
of ``Hello`` as independent evidence.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch


THIS_DIR = Path(__file__).resolve().parent
SHARED_DIR = THIS_DIR.parent / "shared"
if str(SHARED_DIR) not in sys.path:
    sys.path.insert(0, str(SHARED_DIR))

from multiscale_common import clear_cuda, load_model_and_tokenizer, set_seed, write_csv, write_json  # noqa: E402
from scaffold_ablation_common import (  # noqa: E402
    PromptItem,
    balanced_neutral_assignment,
    evaluate_prompt_items,
    flatten_metrics,
    is_v5_model_specific_dataset,
    load_fixed_pairs,
    parse_prompt_parts,
    replace_leading_user_verb,
    split_instruction_and_body,
    validate_fixed_pairs,
    validate_invariant_scaffold,
    validate_neutral_tokenization,
    validate_v5_model_compatibility,
    validate_v5_requested_split_sizes,
    v5_dataset_provenance,
)


PROJECT_ROOT = THIS_DIR.parents[1]
DEFAULT_DATASET_VIEW_ROOT = PROJECT_ROOT / "datasets" / "v4_multidomain_balanced"
DEFAULT_MODEL_PATH = Path(os.environ.get("QWEN3_8B_PATH", PROJECT_ROOT / "external" / "models" / "Qwen3-8B")).expanduser()
DOMAIN_CHOICES = ("D1", "D3", "D4", "D5")
LADDER_ORDER = ("L0", "L1", "L2", "L3", "L4", "L5", "L6", "L7")
LADDER_LABELS = {
    "L0": "empty user turn",
    "L1": "Hello",
    "L2": "unrelated factual question",
    "L3": "task body only",
    "L4": "predeclared neutral verb + body",
    "L5": "analysis verb + body",
    "L6": "execution verb + body",
    "L7": "no system scaffold + execution + body",
}


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
    parser.add_argument("--eval-pairs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
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


def make_item(
    model,
    *,
    sample_id: str,
    prompt: str,
    level: str,
    request_type: str,
    neutral_verb: str | None = None,
) -> PromptItem:
    tokens = model.to_tokens(prompt, prepend_bos=False).detach().cpu()
    return PromptItem(
        sample_id=sample_id,
        prompt=prompt,
        tokens_cpu=tokens,
        token_len=int(tokens.shape[-1]),
        metadata={
            "condition": level,
            "request_type": request_type,
            "neutral_verb": neutral_verb or "",
            "prompt_sha256": sha256_text(prompt),
        },
    )


def build_ladder_items(model, pairs, reference_parts) -> list[PromptItem]:
    """Create L0--L7 exactly once for literal prompts and once per body otherwise."""

    items: list[PromptItem] = []
    literal_levels = {
        "L0": ("", "literal_baseline"),
        "L1": ("Hello", "neutral_greeting"),
        "L2": ("What is the capital of France?", "unrelated_question"),
    }
    for level, (user_content, request_type) in literal_levels.items():
        items.append(
            make_item(
                model,
                sample_id=f"{level}_literal_singleton",
                prompt=reference_parts.render(
                    user_content=user_content,
                    include_role=True,
                    include_tool_schema=True,
                    include_format=True,
                ),
                level=level,
                request_type=request_type,
            )
        )

    neutral_assignment = balanced_neutral_assignment(pairs)
    for pair in pairs:
        clean_parts = parse_prompt_parts(pair.clean_text)
        corrupt_parts = parse_prompt_parts(pair.corrupt_text)
        _instruction, body = split_instruction_and_body(clean_parts.user_content)
        neutral_verb = neutral_assignment[pair.sample_id]
        neutral_content = replace_leading_user_verb(clean_parts.user_content, neutral_verb)
        variants = {
            "L3": (body, "verb_free_task", None, clean_parts.render(
                user_content=body,
                include_role=True,
                include_tool_schema=True,
                include_format=True,
            )),
            "L4": (neutral_content, "neutral_verb_task", neutral_verb, clean_parts.render(
                user_content=neutral_content,
                include_role=True,
                include_tool_schema=True,
                include_format=True,
            )),
            "L5": (corrupt_parts.user_content, "analysis_task", None, corrupt_parts.render(
                user_content=corrupt_parts.user_content,
                include_role=True,
                include_tool_schema=True,
                include_format=True,
            )),
            "L6": (clean_parts.user_content, "execution_task", None, clean_parts.render(
                user_content=clean_parts.user_content,
                include_role=True,
                include_tool_schema=True,
                include_format=True,
            )),
            "L7": (clean_parts.user_content, "execution_without_scaffold", None, clean_parts.render_without_system(
                user_content=clean_parts.user_content
            )),
        }
        for level, (_content, request_type, maybe_neutral, prompt) in variants.items():
            items.append(
                make_item(
                    model,
                    sample_id=pair.sample_id,
                    prompt=prompt,
                    level=level,
                    request_type=request_type,
                    neutral_verb=maybe_neutral,
                )
            )
    return items


def write_plot(sample_rows: list[dict[str, Any]], summary_rows: list[dict[str, Any]], path: Path) -> None:
    """Create the plan's logit/probability scatter with ladder-order means."""

    colors = {
        "L0": "#666666",
        "L1": "#888888",
        "L2": "#aaaaaa",
        "L3": "#4c78a8",
        "L4": "#72b7b2",
        "L5": "#e45756",
        "L6": "#54a24b",
        "L7": "#b279a2",
    }
    fig, (scatter_ax, bar_ax) = plt.subplots(1, 2, figsize=(12.0, 4.6), gridspec_kw={"width_ratios": [1.35, 1.0]})
    for level in LADDER_ORDER:
        rows = [row for row in sample_rows if row["condition"] == level]
        scatter_ax.scatter(
            [float(row["tool_call_logit"]) for row in rows],
            [float(row["tool_call_prob"]) for row in rows],
            s=24 if len(rows) == 1 else 12,
            alpha=0.62,
            color=colors[level],
            label=level,
        )
    mean_x = [next(float(row["behavior_mean_tool_call_logit"]) for row in summary_rows if row["ladder_level"] == level) for level in LADDER_ORDER]
    mean_y = [next(float(row["behavior_mean_tool_call_prob"]) for row in summary_rows if row["ladder_level"] == level) for level in LADDER_ORDER]
    scatter_ax.plot(mean_x, mean_y, color="black", linewidth=1.1, alpha=0.8, zorder=1)
    for level, x_value, y_value in zip(LADDER_ORDER, mean_x, mean_y):
        scatter_ax.annotate(level, (x_value, y_value), xytext=(4, 4), textcoords="offset points", fontsize=8)
    scatter_ax.set_xlabel("<tool_call> logit")
    scatter_ax.set_ylabel("p(<tool_call>)")
    scatter_ax.set_title("Request ladder: point-level behavior")
    scatter_ax.grid(alpha=0.22)
    scatter_ax.legend(ncol=2, fontsize=8, frameon=False)

    bar_ax.bar(
        range(len(LADDER_ORDER)),
        mean_y,
        color=[colors[level] for level in LADDER_ORDER],
        width=0.74,
    )
    bar_ax.set_xticks(range(len(LADDER_ORDER)), LADDER_ORDER)
    bar_ax.set_ylim(0.0, 1.03)
    bar_ax.set_ylabel("mean p(<tool_call>)")
    bar_ax.set_title("Primary probability metric")
    bar_ax.grid(axis="y", alpha=0.22)
    for index, value in enumerate(mean_y):
        bar_ax.text(index, min(value + 0.025, 1.005), f"{value:.2g}", ha="center", va="bottom", fontsize=7)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def ladder_interpretation(summary_rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_level = {str(row["ladder_level"]): row for row in summary_rows}
    p = {level: float(by_level[level]["behavior_mean_tool_call_prob"]) for level in LADDER_ORDER}
    l0_l1_top1 = all(float(by_level[level]["behavior_tool_call_top1_rate"]) == 1.0 for level in ("L0", "L1"))
    distance = {
        level: {
            "to_execution_L6": abs(p[level] - p["L6"]),
            "to_analysis_L5": abs(p[level] - p["L5"]),
        }
        for level in ("L3", "L4")
    }
    same_side = all(values["to_execution_L6"] <= values["to_analysis_L5"] for values in distance.values())
    if l0_l1_top1:
        preregistered_case = "A_literal_default"
    elif same_side:
        preregistered_case = "B_task_conditioned_default"
    else:
        preregistered_case = "C_default_falsified_by_L3_or_L4"
    return {
        "pre_registered_decision_rule": (
            "A if both L0 and L1 predict <tool_call> top-1; otherwise B when both L3 and L4 are closer in "
            "mean p_call to L6 than to L5; otherwise C. This is an interpretation aid, not a significance test."
        ),
        "observed_case": preregistered_case,
        "literal_baseline_top1": {level: by_level[level]["behavior_tool_call_top1_rate"] for level in ("L0", "L1")},
        "mean_probability": p,
        "L3_L4_distance_to_execution_vs_analysis": distance,
    }


def main() -> None:
    args = parse_args()
    output_root = args.output_root.resolve()
    dataset_root, dataset_label, eval_split, is_v5 = resolve_dataset_input(args)
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite existing result directory: {output_root}")
    if int(args.eval_pairs) <= 0:
        raise ValueError("--eval-pairs must be positive")
    provenance = v5_dataset_provenance(dataset_root, splits=(eval_split,)) if is_v5 else None
    if provenance is not None:
        validate_v5_requested_split_sizes(provenance, requested_counts={eval_split: args.eval_pairs})
    output_root.mkdir(parents=True, exist_ok=False)
    write_json(
        output_root / "run_config.json",
        {
            "experiment": "request_ladder",
            "dataset_label": dataset_label,
            "dataset_root": str(dataset_root),
            "eval_split": eval_split,
            "model_path": str(args.model_path.resolve()),
            "model_label": args.model_label,
            "eval_pairs": args.eval_pairs,
            "batch_size": args.batch_size,
            "seed": args.seed,
            "levels": list(LADDER_ORDER),
            "membership_rule": "Fixed released split; no scaffold-variant re-screening or behavioral re-selection.",
            "literal_prompt_rule": "L0/L1/L2 are each evaluated once because their user turn contains no task body; results are never pseudo-replicated.",
            "L7_definition": "System turn entirely removed; native Qwen user/assistant boundary retained.",
        },
    )
    if provenance is not None:
        write_json(output_root / "dataset_provenance.json", provenance)

    set_seed(args.seed)
    model, tokenizer, tool_token_id = load_model_and_tokenizer(model_path=args.model_path.resolve(), device=args.device)
    try:
        if provenance is not None:
            validate_v5_model_compatibility(
                provenance, model_path=args.model_path, model_label=args.model_label, tool_token_id=tool_token_id
            )
        test_pairs = load_fixed_pairs(model, dataset_root=dataset_root, split=eval_split, max_pairs=args.eval_pairs)
        pair_validation = validate_fixed_pairs(test_pairs, expected_count=args.eval_pairs, label=f"{dataset_label}/{eval_split}")
        reference_parts = validate_invariant_scaffold(test_pairs, label=f"{dataset_label}/{eval_split}")
        neutral_protocol = validate_neutral_tokenization(tokenizer)
        items = build_ladder_items(model, test_pairs, reference_parts)

        expected_counts = {"L0": 1, "L1": 1, "L2": 1, "L3": args.eval_pairs, "L4": args.eval_pairs, "L5": args.eval_pairs, "L6": args.eval_pairs, "L7": args.eval_pairs}
        observed_counts = {level: sum(item.metadata["condition"] == level for item in items) for level in LADDER_ORDER}
        if observed_counts != expected_counts:
            raise AssertionError(f"Unexpected ladder prompt counts: {observed_counts}")
        metrics, sample_rows = evaluate_prompt_items(
            model,
            items,
            batch_size=args.batch_size,
            tool_token_id=tool_token_id,
            progress_label=f"{dataset_label}: request ladder",
        )
        summary_rows: list[dict[str, Any]] = []
        for level in LADDER_ORDER:
            level_rows = [row for row in sample_rows if row["condition"] == level]
            summary_rows.append(
                {
                    "dataset_label": dataset_label,
                    "ladder_level": level,
                    "description": LADDER_LABELS[level],
                    "request_type": level_rows[0]["request_type"],
                    "unique_prompt_count": len({row["prompt_sha256"] for row in level_rows}),
                    **flatten_metrics("behavior", metrics[level]),
                }
            )
        interpretation = ladder_interpretation(summary_rows)

        component_hashes = {
            "role_instructions_sha256": sha256_text(reference_parts.role_instructions),
            "tool_schema_block_sha256": sha256_text(reference_parts.tool_schema_block),
            "format_template_sha256": sha256_text(reference_parts.format_template),
            "assistant_suffix_sha256": sha256_text(reference_parts.assistant_suffix),
        }
        write_json(output_root / "pair_validation.json", {eval_split: pair_validation, "scaffold_component_hashes": component_hashes})
        write_json(output_root / "neutral_verb_protocol.json", neutral_protocol)
        write_csv(output_root / "sample_metrics.csv", sample_rows)
        write_csv(output_root / "ladder_summary.csv", summary_rows)
        write_json(output_root / "ladder_interpretation.json", interpretation)
        write_json(
            output_root / "completion.json",
            {
                "status": "complete",
                "dataset_label": dataset_label,
                "dataset_cardinality": {eval_split: len(test_pairs)},
                "tool_token_id": tool_token_id,
                "levels": list(LADDER_ORDER),
                "interpretation_case": interpretation["observed_case"],
            },
        )
        # Compute artifacts are committed before optional rendering so an
        # environment-specific Matplotlib issue cannot make a completed GPU
        # measurement look incomplete.
        write_plot(sample_rows, summary_rows, output_root / "request_ladder.pdf")
        print(json.dumps({"status": "complete", "output_root": str(output_root), "dataset_label": dataset_label}, ensure_ascii=False))
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
