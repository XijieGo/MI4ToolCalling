#!/usr/bin/env python3
"""Polarity-aware causal audit for V4 affordance-reversal conditions.

V4 deliberately can make the original *corrupt* request the tool-matched
one.  In that case a raw clean-minus-corrupt vector changes sign by design,
and applying the usual add-to-corrupt / remove-from-clean intervention is not
the causal question in the ablation plan.  This audit reports both possible
call/no-call orientations for every domain, alongside the raw and
call-oriented vector cosine, without selecting an orientation after seeing an
intervention result.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
from tqdm.auto import tqdm


THIS_DIR = Path(__file__).resolve().parent
SHARED_DIR = THIS_DIR.parent / "shared"
if str(SHARED_DIR) not in sys.path:
    sys.path.insert(0, str(SHARED_DIR))

from multiscale_common import build_pair_batches, clear_cuda, load_model_and_tokenizer, set_seed, write_csv, write_json  # noqa: E402
from run_cross_scale_fix_gate_and_patch import hook_name, make_last_token_add_hook  # noqa: E402
from run_tool_identity_ablation import (  # noqa: E402
    DEFAULT_MODEL_PATH,
    MetricAccumulator,
    extract_schema,
    render_pairs_for_variant,
    token_stats,
    validate_pairs,
)
from run_tool_identity_ablation_remaining_domains import domain_specs, load_fixed_domain_pairs  # noqa: E402


DEFAULT_DOMAINS = ("D3", "D4", "D5")


@dataclass
class Baseline:
    clean: dict[str, Any]
    corrupt: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit V4 tool-identity affordance reversal with both polarities.")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--dataset-view-root", type=Path, required=True)
    parser.add_argument("--source-run-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--domains", nargs="+", choices=DEFAULT_DOMAINS, default=list(DEFAULT_DOMAINS))
    parser.add_argument("--layer", type=int, default=24)
    parser.add_argument("--hook-kind", choices=("pre", "post"), default="pre")
    parser.add_argument("--eval-pairs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def bundle_vector(path: Path) -> torch.Tensor:
    bundle = torch.load(path, map_location="cpu", weights_only=False)
    vector = bundle.get("mean_diff")
    if not isinstance(vector, torch.Tensor):
        vector = torch.as_tensor(vector)
    vector = vector.detach().cpu().float().view(-1).contiguous()
    if vector.numel() == 0 or not bool(torch.isfinite(vector).all()):
        raise ValueError(f"Invalid mean_diff in {path}")
    return vector


def cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    left = left.float() / left.float().norm().clamp_min(1e-12)
    right = right.float() / right.float().norm().clamp_min(1e-12)
    return float(torch.dot(left, right).clamp(-1.0, 1.0).item())


def baseline_metrics(model, pairs: Sequence[Any], *, batch_size: int, tool_token_id: int) -> Baseline:
    clean_acc = MetricAccumulator()
    corrupt_acc = MetricAccumulator()
    for batch in tqdm(build_pair_batches(pairs, batch_size), desc="V4 baseline", dynamic_ncols=True):
        clean_tokens = batch.clean_tokens_cpu.to(model.W_U.device)
        corrupt_tokens = batch.corrupt_tokens_cpu.to(model.W_U.device)
        with torch.no_grad():
            clean_logits = model(clean_tokens)
            corrupt_logits = model(corrupt_tokens)
        clean_acc.update(token_stats(clean_logits, tool_token_id), tool_token_id=tool_token_id)
        corrupt_acc.update(token_stats(corrupt_logits, tool_token_id), tool_token_id=tool_token_id)
        del clean_tokens, corrupt_tokens, clean_logits, corrupt_logits
    return Baseline(clean=clean_acc.summary(), corrupt=corrupt_acc.summary())


def evaluate_orientation(
    model,
    pairs: Sequence[Any],
    *,
    baseline: Baseline,
    call_side: str,
    frozen_v0: torch.Tensor,
    layer: int,
    hook_kind: str,
    batch_size: int,
    tool_token_id: int,
) -> dict[str, Any]:
    """Add V0 to the stipulated no-call side; remove it from the call side."""

    if call_side not in {"clean", "corrupt"}:
        raise ValueError(f"Unknown call side {call_side}")
    no_call_side = "corrupt" if call_side == "clean" else "clean"
    add_acc = MetricAccumulator()
    remove_acc = MetricAccumulator()
    add_strict = 0
    remove_strict = 0
    no_call_base_count = 0
    call_base_count = 0
    name = hook_name(layer, hook_kind)
    add_hook = (name, make_last_token_add_hook(frozen_v0))
    remove_hook = (name, make_last_token_add_hook(-frozen_v0))
    for batch in tqdm(build_pair_batches(pairs, batch_size), desc=f"V4 {call_side}-call audit", dynamic_ncols=True):
        call_tokens_cpu = batch.clean_tokens_cpu if call_side == "clean" else batch.corrupt_tokens_cpu
        no_call_tokens_cpu = batch.corrupt_tokens_cpu if no_call_side == "corrupt" else batch.clean_tokens_cpu
        call_tokens = call_tokens_cpu.to(model.W_U.device)
        no_call_tokens = no_call_tokens_cpu.to(model.W_U.device)
        with torch.no_grad():
            call_logits = model(call_tokens)
            no_call_logits = model(no_call_tokens)
            added_logits = model.run_with_hooks(no_call_tokens, fwd_hooks=[add_hook])
            removed_logits = model.run_with_hooks(call_tokens, fwd_hooks=[remove_hook])
        call_stats = token_stats(call_logits, tool_token_id)
        no_call_stats = token_stats(no_call_logits, tool_token_id)
        add_stats = token_stats(added_logits, tool_token_id)
        remove_stats = token_stats(removed_logits, tool_token_id)
        add_acc.update(add_stats, tool_token_id=tool_token_id)
        remove_acc.update(remove_stats, tool_token_id=tool_token_id)
        call_is_tool = call_stats["top1"] == tool_token_id
        no_call_is_tool = no_call_stats["top1"] == tool_token_id
        add_is_tool = add_stats["top1"] == tool_token_id
        remove_is_tool = remove_stats["top1"] == tool_token_id
        add_strict += int(((~no_call_is_tool) & add_is_tool).sum().item())
        remove_strict += int((call_is_tool & (~remove_is_tool)).sum().item())
        no_call_base_count += int((~no_call_is_tool).sum().item())
        call_base_count += int(call_is_tool.sum().item())
        del (
            call_tokens,
            no_call_tokens,
            call_logits,
            no_call_logits,
            added_logits,
            removed_logits,
            call_stats,
            no_call_stats,
            add_stats,
            remove_stats,
            call_is_tool,
            no_call_is_tool,
            add_is_tool,
            remove_is_tool,
        )

    call_metrics = baseline.clean if call_side == "clean" else baseline.corrupt
    no_call_metrics = baseline.corrupt if no_call_side == "corrupt" else baseline.clean
    added = add_acc.summary()
    removed = remove_acc.summary()
    gap = float(call_metrics["mean_tool_call_logit"] - no_call_metrics["mean_tool_call_logit"])
    valid = gap > 1e-8
    return {
        "call_side": call_side,
        "no_call_side": no_call_side,
        "call_minus_no_call_logit_gap": gap,
        "normalization_valid": valid,
        "add_to_no_call": added,
        "remove_from_call": removed,
        "add_strict_flip_rate": add_strict / max(no_call_base_count, 1),
        "remove_strict_drop_rate": remove_strict / max(call_base_count, 1),
        "sufficiency": (
            (float(added["mean_tool_call_logit"]) - float(no_call_metrics["mean_tool_call_logit"])) / gap
            if valid
            else None
        ),
        "necessity": (
            (float(call_metrics["mean_tool_call_logit"]) - float(removed["mean_tool_call_logit"])) / gap
            if valid
            else None
        ),
    }


def main() -> None:
    args = parse_args()
    domains = tuple(args.domains)
    if len(set(domains)) != len(domains):
        raise ValueError("Duplicate domains are not allowed")
    source_root = args.source_run_root.resolve()
    dataset_root = args.dataset_view_root.resolve()
    output_root = args.output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite existing output root: {output_root}")
    if not (source_root / "completion.json").exists():
        raise FileNotFoundError(f"Source V0--V6 run is incomplete: {source_root}")
    output_root.mkdir(parents=True, exist_ok=False)
    write_json(
        output_root / "run_config.json",
        {
            "experiment": "v4_tool_identity_v4_polarity_audit",
            "source_run_root": str(source_root),
            "dataset_view_root": str(dataset_root),
            "domains": list(domains),
            "layer": args.layer,
            "hook_kind": args.hook_kind,
            "eval_pairs": args.eval_pairs,
            "rule": "Report both clean-call and corrupt-call orientations; no post-intervention orientation selection.",
        },
    )
    set_seed(args.seed)
    model, _tokenizer, tool_token_id = load_model_and_tokenizer(model_path=args.model_path.resolve(), device=args.device)
    try:
        all_rows: list[dict[str, Any]] = []
        summaries: dict[str, Any] = {}
        for domain in domains:
            root = dataset_root / domain
            base_pairs = load_fixed_domain_pairs(model, dataset_root=root, split="test", max_pairs=args.eval_pairs)
            original_schema = extract_schema(base_pairs[0].clean_text)
            specs = domain_specs(domain, original_schema)
            pairs = render_pairs_for_variant(model, base_pairs, variant=specs["V4"])
            validation = validate_pairs(pairs, expected_count=args.eval_pairs, label=f"{domain}/V4/test")
            baseline = baseline_metrics(model, pairs, batch_size=args.batch_size, tool_token_id=tool_token_id)
            v0 = bundle_vector(source_root / domain / "native_vectors" / f"V0_L{args.layer}_{args.hook_kind}.pt")
            raw_v4 = bundle_vector(source_root / domain / "native_vectors" / f"V4_L{args.layer}_{args.hook_kind}.pt")
            orientations: dict[str, Any] = {}
            for call_side in ("clean", "corrupt"):
                result = evaluate_orientation(
                    model,
                    pairs,
                    baseline=baseline,
                    call_side=call_side,
                    frozen_v0=v0,
                    layer=args.layer,
                    hook_kind=args.hook_kind,
                    batch_size=args.batch_size,
                    tool_token_id=tool_token_id,
                )
                oriented = raw_v4 if call_side == "clean" else -raw_v4
                result["raw_clean_minus_corrupt_cosine_to_v0"] = cosine(raw_v4, v0)
                result["call_oriented_cosine_to_v0"] = cosine(oriented, v0)
                orientations[call_side] = result
                all_rows.append(
                    {
                        "domain": domain,
                        "call_side_assumption": call_side,
                        "no_call_side": result["no_call_side"],
                        "raw_clean_minus_corrupt_cosine_to_v0": result["raw_clean_minus_corrupt_cosine_to_v0"],
                        "call_oriented_cosine_to_v0": result["call_oriented_cosine_to_v0"],
                        "call_minus_no_call_logit_gap": result["call_minus_no_call_logit_gap"],
                        "normalization_valid": result["normalization_valid"],
                        "frozen_v0_add_top1_rate": result["add_to_no_call"]["tool_call_top1_rate"],
                        "frozen_v0_add_strict_flip_rate": result["add_strict_flip_rate"],
                        "frozen_v0_sufficiency": result["sufficiency"],
                        "frozen_v0_remove_remaining_top1_rate": result["remove_from_call"]["tool_call_top1_rate"],
                        "frozen_v0_remove_strict_drop_rate": result["remove_strict_drop_rate"],
                        "frozen_v0_necessity": result["necessity"],
                    }
                )
            preferred = "clean" if baseline.clean["mean_tool_call_prob"] >= baseline.corrupt["mean_tool_call_prob"] else "corrupt"
            summaries[domain] = {
                "pair_validation": validation,
                "baseline": {"clean": baseline.clean, "corrupt": baseline.corrupt},
                "preferred_behavioral_call_side_by_mean_probability": preferred,
                "orientations": orientations,
            }
            write_json(output_root / f"{domain}_V4_polarity.json", summaries[domain])
            write_csv(output_root / "V4_polarity_audit.partial.csv", all_rows)
            del base_pairs, pairs, v0, raw_v4
            clear_cuda()
        write_json(output_root / "V4_polarity_summary.json", summaries)
        write_csv(output_root / "V4_polarity_audit.csv", all_rows)
        write_json(output_root / "completion.json", {"status": "complete", "domains": list(domains), "tool_token_id": tool_token_id})
        print(json.dumps({"status": "complete", "output_root": str(output_root)}, ensure_ascii=False))
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
