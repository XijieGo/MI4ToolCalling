#!/usr/bin/env python3
"""Run or rerun only the frozen-vector τ² phase for a completed Qwen3.5 calibration.

This deliberately refuses to estimate a layer or vector.  It loads the
coding-only bundle produced by ``run_tau2_qwen35.py`` and is useful when a
long-context τ² screen needs a different safe context cap or batch size.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import run_tau2_qwen35 as qwen  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=sorted(qwen.SPECS), required=True)
    parser.add_argument("--output-root", type=Path, default=qwen.DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--raw-root", type=Path, default=qwen.RAW_ROOT)
    parser.add_argument("--prepared-root", type=Path, default=qwen.PREPARED_ROOT)
    parser.add_argument("--tau-batch-size", type=int, default=4)
    parser.add_argument("--max-context-tokens", type=int, default=16384)
    parser.add_argument("--tool-screen-limit", type=int, default=400)
    parser.add_argument("--induction-screen-limit", type=int, default=0)
    parser.add_argument("--final-count", type=int, default=50)
    parser.add_argument("--max-baseline-tool-probability", type=float, default=0.05)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--seed", type=int, default=qwen.SEED)
    parser.add_argument(
        "--reuse-tool-selection",
        action="store_true",
        help="Reuse a previously completed model-specific selected_tool_prefixes.jsonl; never recomputes it.",
    )
    parser.add_argument("--skip-generation", action="store_true")
    return parser.parse_args()


def load_bundle(path: Path) -> dict[str, Any]:
    try:
        bundle = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        bundle = torch.load(path, map_location="cpu")
    if not isinstance(bundle, dict):
        raise TypeError(f"Unexpected vector bundle at {path}")
    return bundle


def main() -> None:
    args = parse_args()
    if args.tau_batch_size < 1 or args.final_count < 1:
        raise ValueError("Batch size and final count must be positive")
    spec = qwen.SPECS[args.model]
    output_root = args.output_root.resolve() / spec.key
    bundle_path = output_root / "coding_vector_bundle.pt"
    summary_path = output_root / "coding_vector_summary.json"
    if not bundle_path.exists() or not summary_path.exists():
        raise FileNotFoundError("No completed coding-only calibration bundle; run run_tau2_qwen35.py first")
    bundle = load_bundle(bundle_path)
    if str(bundle.get("model_path")) != str(spec.model_path):
        raise RuntimeError("Frozen vector bundle was created for a different model path")
    vector = bundle.get("mean_diff")
    random_unit = bundle.get("random_direction_unit")
    layer = int(bundle.get("layer"))
    if not isinstance(vector, torch.Tensor) or not isinstance(random_unit, torch.Tensor):
        raise TypeError("Frozen vector bundle has no tensor directions")
    vector = vector.detach().float().cpu().contiguous()
    random_unit = random_unit.detach().float().cpu().contiguous()
    coding_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    raw_root = args.raw_root.resolve()
    prepared_root = args.prepared_root.resolve()
    resources = qwen.read_resources(raw_root)

    model: Any | None = None
    tokenizer: Any | None = None
    try:
        model, tokenizer, device = qwen.load_model_and_tokenizer(spec)
        tool_token_id, token_info = qwen.get_tool_token_id(tokenizer)
        layer_count = len(qwen.resolve_qwen_layers(model))
        if layer < 0 or layer >= layer_count:
            raise RuntimeError(f"Frozen layer L{layer} is invalid for {spec.display_name}")
        qwen.common.write_json(
            output_root / "tau2_resume_config.json",
            {
                "started_unix": time.time(),
                "model": spec.display_name,
                "frozen_coding_bundle": str(bundle_path),
                "frozen_layer": layer,
                "frozen_vector_norm": float(vector.norm().item()),
                "tool_token": token_info,
                "arguments": vars(args),
            },
        )

        baseline = qwen.common.Condition("baseline_no_hook_alpha_0", "baseline", 0.0, None, "no intervention")
        tool_source = qwen.common.read_jsonl(prepared_root / "screen_pool.jsonl")
        selected_tool_path = output_root / "selected_tool_prefixes.jsonl"
        if args.reuse_tool_selection:
            if not selected_tool_path.exists():
                raise FileNotFoundError(f"Cannot reuse missing tool selection: {selected_tool_path}")
            selected_tool_manifest = qwen.common.read_jsonl(selected_tool_path)
            if len(selected_tool_manifest) != args.final_count:
                raise RuntimeError(
                    f"Frozen tool selection has {len(selected_tool_manifest)} rows; expected {args.final_count}"
                )
            prior_screen = output_root / "tool_baseline_screen.jsonl"
            tool_screen_count = len(qwen.common.read_jsonl(prior_screen)) if prior_screen.exists() else None
            tool_stats = {"reused_frozen_model_specific_selection": True}
        else:
            tool_input = qwen.common.balanced_subset_tool(
                tool_source, limit=args.tool_screen_limit, seed=args.seed
            )
            tool_prepared, tool_rejected = qwen.prepare_screen_pool(
                tool_input,
                tokenizer=tokenizer,
                resources=resources,
                max_context_tokens=args.max_context_tokens,
            )
            qwen.common.write_json(
                output_root / "tool_render_audit.json",
                {
                    "source_rows": len(tool_source),
                    "screen_input_rows": len(tool_input),
                    "rendered_rows": len(tool_prepared),
                    "rejections": tool_rejected,
                },
            )
            tool_screen, tool_stats = qwen.common.evaluate_condition(
                tool_prepared,
                model=model,
                tokenizer=tokenizer,
                device=device,
                tool_token_id=tool_token_id,
                layer=layer,
                condition=baseline,
                batch_size=args.tau_batch_size,
            )
            qwen.common.write_jsonl(output_root / "tool_baseline_screen.jsonl", tool_screen)
            tool_screen_count = len(tool_screen)
            selected_tool_manifest = qwen.common.select_tool_rows(
                tool_screen, final_count=args.final_count, seed=args.seed
            )
            qwen.common.write_jsonl(selected_tool_path, selected_tool_manifest)
        tool_by_id = {str(row["candidate_id"]): row for row in tool_source}
        tool_selected, rejected = qwen.prepare_screen_pool(
            [tool_by_id[str(row["candidate_id"])] for row in selected_tool_manifest],
            tokenizer=tokenizer,
            resources=resources,
            max_context_tokens=args.max_context_tokens,
        )
        if rejected or len(tool_selected) != args.final_count:
            raise RuntimeError("Selected tool prefixes no longer render")

        induction_source = qwen.common.read_jsonl(
            prepared_root / "text_reply_induction_terminal" / "screen_pool.jsonl"
        )
        induction_input = qwen.common.balanced_subset_induction(
            induction_source, limit=args.induction_screen_limit, seed=args.seed
        )
        induction_prepared, induction_rejected = qwen.prepare_screen_pool(
            induction_input,
            tokenizer=tokenizer,
            resources=resources,
            max_context_tokens=args.max_context_tokens,
        )
        qwen.common.write_json(
            output_root / "induction_render_audit.json",
            {
                "source_rows": len(induction_source),
                "screen_input_rows": len(induction_input),
                "rendered_rows": len(induction_prepared),
                "rejections": induction_rejected,
            },
        )
        induction_screen, induction_stats = qwen.common.evaluate_condition(
            induction_prepared,
            model=model,
            tokenizer=tokenizer,
            device=device,
            tool_token_id=tool_token_id,
            layer=layer,
            condition=baseline,
            batch_size=args.tau_batch_size,
        )
        qwen.common.write_jsonl(output_root / "induction_baseline_screen.jsonl", induction_screen)
        selected_induction_manifest = qwen.common.select_induction_rows(
            induction_screen,
            final_count=args.final_count,
            max_probability=args.max_baseline_tool_probability,
            seed=args.seed,
        )
        qwen.common.write_jsonl(output_root / "selected_induction_prefixes.jsonl", selected_induction_manifest)
        induction_by_id = {str(row["candidate_id"]): row for row in induction_source}
        induction_selected, rejected = qwen.prepare_screen_pool(
            [induction_by_id[str(row["candidate_id"])] for row in selected_induction_manifest],
            tokenizer=tokenizer,
            resources=resources,
            max_context_tokens=args.max_context_tokens,
        )
        if rejected or len(induction_selected) != args.final_count:
            raise RuntimeError("Selected induction prefixes no longer render")

        suppression_rows, suppression, suppression_stats, _ = qwen.common.run_intervention_suite(
            tool_selected,
            kind="suppression",
            model=model,
            tokenizer=tokenizer,
            device=device,
            tool_token_id=tool_token_id,
            layer=layer,
            vector=vector,
            random_unit=random_unit,
            batch_size=args.tau_batch_size,
        )
        induction_rows, induction, induction_hook_stats, by_condition = qwen.common.run_intervention_suite(
            induction_selected,
            kind="induction",
            model=model,
            tokenizer=tokenizer,
            device=device,
            tool_token_id=tool_token_id,
            layer=layer,
            vector=vector,
            random_unit=random_unit,
            batch_size=args.tau_batch_size,
        )
        qwen.common.write_jsonl(output_root / "suppression_per_sample.jsonl", suppression_rows)
        qwen.common.write_jsonl(output_root / "induction_per_sample.jsonl", induction_rows)
        qwen.common.write_csv(output_root / "suppression_summary.csv", suppression)
        qwen.common.write_csv(output_root / "induction_summary.csv", induction)
        qwen.common.write_json(output_root / "suppression_summary.json", {"conditions": suppression, "hook_stats": suppression_stats})
        qwen.common.write_json(output_root / "induction_summary.json", {"conditions": induction, "hook_stats": induction_hook_stats})

        generation_rows: list[dict[str, Any]] = []
        generation_stats = {"hook_calls": 0, "modified_calls": 0}
        if not args.skip_generation:
            generation_rows, generation_stats = qwen.greedy_induction_audit(
                induction_selected,
                model=model,
                tokenizer=tokenizer,
                device=device,
                tool_token_id=tool_token_id,
                layer=layer,
                vector=vector,
                resources=resources,
                max_new_tokens=args.max_new_tokens,
                direct_rows={str(row["candidate_id"]): row for row in by_condition["plus_mean_diff_alpha_1"]},
            )
        qwen.common.write_jsonl(output_root / "induction_alpha1_generations.jsonl", generation_rows)
        result = {
            "model": spec.display_name,
            "coding_vector": coding_summary,
            "screening": {
                "tool": {"screen_stats": tool_stats, "selected": len(selected_tool_manifest), "screened": tool_screen_count},
                "induction": {"screen_stats": induction_stats, "selected": len(selected_induction_manifest), "screened": len(induction_screen)},
            },
            "suppression": suppression,
            "induction": induction,
            "induction_alpha1_generation": {
                "n": len(generation_rows),
                "first_tool_call": sum(bool(row["first_generated_token_is_tool_call"]) for row in generation_rows),
                "well_formed_tool_call": sum(bool(row["well_formed_tool_call"]) for row in generation_rows),
                "direct_forward_match": sum(bool(row["first_token_matches_direct_forward"]) for row in generation_rows),
                "hook_stats": generation_stats,
            },
            "completed_unix": time.time(),
        }
        qwen.common.write_json(output_root / "final_result.json", result)
        print(json.dumps({"completed": True, "model": spec.key, "result": result}, ensure_ascii=False), flush=True)
    finally:
        if model is not None:
            del model
        if tokenizer is not None:
            del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
