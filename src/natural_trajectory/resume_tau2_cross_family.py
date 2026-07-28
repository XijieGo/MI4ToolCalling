#!/usr/bin/env python3
"""Resume a τ² cross-family run after model-specific baseline screening.

This utility deliberately never estimates a direction from Telecom data.  It
loads the frozen coding-only vector bundle produced by
``run_tau2_cross_family.py`` and reuses the model's already-written baseline
screens.  Its intended use is a recoverable eligibility adjustment (for
example, a model has fewer than 50 non-tool prefixes below an initially too
strict probability cutoff) without re-fitting on τ² or re-running expensive
long-context baseline forwards.
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path
from typing import Any

import torch

import run_tau2_cross_family as core


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=sorted(core.SPECS), required=True)
    parser.add_argument("--output-root", type=Path, default=core.DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--raw-root", type=Path, default=core.RAW_ROOT)
    parser.add_argument("--prepared-root", type=Path, default=core.PREPARED_ROOT)
    parser.add_argument("--final-count", type=int, default=50)
    parser.add_argument("--tau-batch-size", type=int, default=1)
    parser.add_argument("--max-context-tokens", type=int, default=16384)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--max-baseline-tool-probability", type=float, default=0.10)
    parser.add_argument("--skip-generation", action="store_true")
    return parser.parse_args()


def read_bundle(path: Path) -> dict[str, Any]:
    try:
        bundle = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        bundle = torch.load(path, map_location="cpu")
    if not isinstance(bundle, dict) or "mean_diff" not in bundle or "random_direction_unit" not in bundle:
        raise RuntimeError(f"Invalid coding vector bundle: {path}")
    return bundle


def main() -> None:
    args = parse_args()
    if args.final_count < 1 or args.tau_batch_size < 1:
        raise ValueError("--final-count and --tau-batch-size must be positive")
    if not 0.0 <= args.max_baseline_tool_probability <= 1.0:
        raise ValueError("--max-baseline-tool-probability must be in [0, 1]")
    spec = core.SPECS[args.model]
    output = args.output_root.resolve() / spec.key
    raw_root = args.raw_root.resolve()
    prepared_root = args.prepared_root.resolve()
    bundle_path = output / "coding_vector_bundle.pt"
    coding_summary_path = output / "coding_vector_summary.json"
    tool_screen_path = output / "tool_baseline_screen.jsonl"
    selected_tool_path = output / "selected_tool_prefixes.jsonl"
    induction_screen_path = output / "induction_baseline_screen.jsonl"
    for path in (bundle_path, coding_summary_path, tool_screen_path, selected_tool_path, induction_screen_path):
        if not path.exists():
            raise FileNotFoundError(f"Cannot resume without prior screening artifact: {path}")

    bundle = read_bundle(bundle_path)
    coding_summary = json.loads(coding_summary_path.read_text(encoding="utf-8"))
    vector = bundle["mean_diff"].detach().float().cpu().contiguous()
    random_unit = bundle["random_direction_unit"].detach().float().cpu().contiguous()
    if vector.ndim != 1 or random_unit.shape != vector.shape:
        raise RuntimeError("Coding vector bundle has incompatible tensor shapes")
    layer = int(bundle["layer"])

    model, tokenizer, device = core.load_tokenizer_and_model(spec)
    try:
        tool_token_id, token_info = core.resolve_tool_token_id(tokenizer, spec)
        saved_token = bundle.get("tool_token", {})
        if int(saved_token.get("tool_token_id_via_convert", tool_token_id)) != tool_token_id:
            raise RuntimeError("Current tokenizer's semantic tool-call token differs from the frozen coding bundle")
        if layer >= len(core.resolve_layers(model)):
            raise RuntimeError(f"L{layer} is absent from {spec.display_name}")

        tool_source = core.read_jsonl(prepared_root / "screen_pool.jsonl")
        tool_by_id = {str(row["candidate_id"]): row for row in tool_source}
        selected_tool_manifest = core.read_jsonl(selected_tool_path)
        selected_tool_rows = [tool_by_id[str(row["candidate_id"])] for row in selected_tool_manifest]
        tool_selected, tool_rejected = core.prepare_screen_pool(
            selected_tool_rows,
            tokenizer=tokenizer,
            spec=spec,
            raw_root=raw_root,
            max_context_tokens=args.max_context_tokens,
        )
        if tool_rejected or len(tool_selected) != args.final_count:
            raise RuntimeError("Prior selected tool prefixes no longer render under the current native template")

        induction_source = core.read_jsonl(prepared_root / "text_reply_induction_terminal" / "screen_pool.jsonl")
        induction_by_id = {str(row["candidate_id"]): row for row in induction_source}
        induction_screen = core.read_jsonl(induction_screen_path)
        selected_induction_manifest = core.select_induction_rows(
            induction_screen,
            final_count=args.final_count,
            max_probability=args.max_baseline_tool_probability,
            seed=20260726,
        )
        core.write_jsonl(output / "selected_induction_prefixes.jsonl", selected_induction_manifest)
        selected_induction_rows = [induction_by_id[str(row["candidate_id"])] for row in selected_induction_manifest]
        induction_selected, induction_rejected = core.prepare_screen_pool(
            selected_induction_rows,
            tokenizer=tokenizer,
            spec=spec,
            raw_root=raw_root,
            max_context_tokens=args.max_context_tokens,
        )
        if induction_rejected or len(induction_selected) != args.final_count:
            raise RuntimeError("Selected text-reply prefixes no longer render under the current native template")

        resume_metadata = {
            "resumed_unix": time.time(),
            "model": spec.display_name,
            "frozen_vector_bundle": str(bundle_path),
            "vector_source": "coding clean/corrupt pairs only; no Telecom fitting",
            "layer": layer,
            "tool_token": token_info,
            "reused_tool_baseline_screen": str(tool_screen_path),
            "reused_induction_baseline_screen": str(induction_screen_path),
            "induction_selection": {
                "n_screened": len(induction_screen),
                "max_baseline_tool_probability": args.max_baseline_tool_probability,
                "selected": len(selected_induction_manifest),
            },
        }
        core.write_json(output / "resume_metadata.json", resume_metadata)

        tool_rows, tool_summaries, tool_hook_stats, _tool_by_condition = core.run_intervention_suite(
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
        induction_rows, induction_summaries, induction_hook_stats, induction_by_condition = core.run_intervention_suite(
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
        core.write_jsonl(output / "suppression_per_sample.jsonl", tool_rows)
        core.write_jsonl(output / "induction_per_sample.jsonl", induction_rows)
        core.write_csv(output / "suppression_summary.csv", tool_summaries)
        core.write_csv(output / "induction_summary.csv", induction_summaries)
        core.write_json(output / "suppression_summary.json", {"conditions": tool_summaries, "hook_stats": tool_hook_stats})
        core.write_json(output / "induction_summary.json", {"conditions": induction_summaries, "hook_stats": induction_hook_stats})

        greedy_rows: list[dict[str, Any]] = []
        greedy_stats = {"hook_calls": 0, "modified_calls": 0}
        if not args.skip_generation:
            greedy_rows, greedy_stats = core.greedy_induction_audit(
                induction_selected,
                model=model,
                tokenizer=tokenizer,
                device=device,
                tool_token_id=tool_token_id,
                layer=layer,
                vector=vector,
                spec=spec,
                raw_root=raw_root,
                max_new_tokens=args.max_new_tokens,
                direct_rows={row["candidate_id"]: row for row in induction_by_condition["plus_mean_diff_alpha_1"]},
            )
        core.write_jsonl(output / "induction_alpha1_generations.jsonl", greedy_rows)
        result = {
            "model": spec.display_name,
            "coding_vector": coding_summary,
            "resume": resume_metadata,
            "screening": {
                "tool": {"selected": len(selected_tool_manifest), "screened": len(core.read_jsonl(tool_screen_path)), "reused": True},
                "induction": {"selected": len(selected_induction_manifest), "screened": len(induction_screen), "reused": True},
            },
            "suppression": tool_summaries,
            "induction": induction_summaries,
            "induction_alpha1_generation": {
                "n": len(greedy_rows),
                "first_tool_call": sum(bool(row["first_generated_token_is_tool_call"]) for row in greedy_rows),
                "well_formed_tool_call": sum(bool(row["well_formed_tool_call"]) for row in greedy_rows),
                "direct_forward_match": sum(bool(row["first_token_matches_direct_forward"]) for row in greedy_rows),
                "hook_stats": greedy_stats,
            },
            "completed_unix": time.time(),
        }
        core.write_json(output / "final_result.json", result)
        print(json.dumps({"completed": True, "model": spec.key, "result": result}, ensure_ascii=False), flush=True)
    finally:
        del model
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
