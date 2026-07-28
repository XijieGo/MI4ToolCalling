#!/usr/bin/env python3
"""Refit adjacent-layer tool-call vectors on the Qwen3-8B v5 release.

Reviewer LxEy asked whether the mean clean-minus-corrupt vector that is
effective at layer 24 also works when it is independently refit at nearby
layers.  This runner keeps that comparison fully held out:

* it fits one raw mean-difference vector at each requested decoder-block
  *input* (the original ``resid_pre`` convention) using all 200 v5 train
  pairs;
* it evaluates each frozen vector on all 300 v5 held-out pairs, adding it to
  corrupt prompts and subtracting it from clean prompts at the same layer; and
* it repeats both interventions with a seeded equal-norm random direction at
  every layer.

The runner consumes only native V0 prompts from the immutable v5 release.  It
never re-screens or modifies data, audits the release's first-token behavior
before reporting a causal result, and refuses to overwrite an existing output
directory.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "datasets" / "v5_model_specific_balanced" / "qwen3_8b"
DEPENDENCY_RUNNER_PATH = PROJECT_ROOT / "src" / "rebuttal" / "run_cjrq_v5_tool_identity.py"
MODEL_KEY = "qwen3_8b"
DEFAULT_LAYERS = (22, 23, 24, 25)


def load_dependency_runner():
    spec = importlib.util.spec_from_file_location("cjrq_v5_runner_for_lxey_neighbor", DEPENDENCY_RUNNER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import v5 native-input helper: {DEPENDENCY_RUNNER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


RUNNER = load_dependency_runner()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--layers", type=int, nargs="+", default=list(DEFAULT_LAYERS))
    parser.add_argument("--train-pairs", type=int, default=200)
    parser.add_argument("--heldout-pairs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--attn-implementation", default="")
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument(
        "--allow-subset",
        action="store_true",
        help="Permit manifest-order prefixes for a smoke test; canonical v5 uses 200 train / 300 held-out.",
    )
    return parser.parse_args()


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def rate(rows: Sequence[dict[str, Any]], field: str) -> float:
    if not rows:
        raise ValueError("Cannot summarize zero rows")
    return sum(float(row[field]) for row in rows) / len(rows)


def cosine_similarity(left: torch.Tensor, right: torch.Tensor) -> float:
    left_unit = left.float() / left.float().norm().clamp_min(1e-12)
    right_unit = right.float() / right.float().norm().clamp_min(1e-12)
    return float(torch.dot(left_unit, right_unit).clamp(-1.0, 1.0).item())


def make_random_equal_norm(vector: torch.Tensor, *, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    unit = torch.randn(vector.shape, generator=generator, dtype=torch.float32)
    unit = unit / unit.norm().clamp_min(1e-12)
    return unit * vector.norm().float(), unit


def build_v0_items(
    records: Sequence[Any], *, family: str, tokenizer: Any
) -> tuple[dict[str, list[Any]], list[dict[str, Any]]]:
    """Render immutable V0 inputs and retain prompt provenance for both sides."""

    items = {"clean": [], "corrupt": []}
    provenance: list[dict[str, Any]] = []
    for record in records:
        for side in ("clean", "corrupt"):
            source = record.clean if side == "clean" else record.corrupt
            item = RUNNER.render_prompt_item(
                source,
                sample_id=str(record.sample_id),
                split=str(record.split),
                side=side,
                variant="V0",
                family=family,
                tokenizer=tokenizer,
            )
            items[side].append(item)
            provenance.append(
                {
                    "sample_id": str(item.sample_id),
                    "split": str(item.split),
                    "side": side,
                    "source_relpath": str(item.relpath),
                    "source_prompt_sha256": str(item.source_prompt_sha256),
                    "rendered_prompt_sha256": str(item.rendered_prompt_sha256),
                    "input_ids_sha256": str(item.input_ids_sha256),
                    "input_token_count": len(item.input_ids),
                    "render_method": str(item.render_method),
                    "v0_template_roundtrip": item.v0_template_roundtrip,
                }
            )
    return items, provenance


def hidden_from_pre_hook(args: tuple[Any, ...], kwargs: dict[str, Any]) -> torch.Tensor:
    hidden = args[0] if args else kwargs.get("hidden_states")
    if not isinstance(hidden, torch.Tensor) or hidden.ndim != 3:
        raise TypeError(f"Expected [batch, sequence, hidden] block input, got {type(hidden).__name__}")
    return hidden


def capture_pre_states(
    model: Any,
    tokenizer: Any,
    items: Sequence[Any],
    *,
    layers: Sequence[torch.nn.Module],
    layer_ids: Sequence[int],
    batch_size: int,
    progress_label: str,
) -> dict[int, torch.Tensor]:
    """Capture final-token decoder-block inputs for several layers in one pass."""

    if not items:
        raise ValueError("Cannot capture states for zero prompts")
    if not layer_ids:
        raise ValueError("No layer IDs requested")
    device = RUNNER.model_device(model)
    captures: dict[int, list[torch.Tensor]] = {int(layer_id): [] for layer_id in layer_ids}
    total = math.ceil(len(items) / batch_size)
    for batch_index, batch in enumerate(RUNNER.chunks(items, batch_size), start=1):
        holders: dict[int, torch.Tensor] = {}
        handles = []
        for requested_layer in layer_ids:
            layer_id = int(requested_layer)

            def capture_hook(_module, args, kwargs, *, captured_layer=layer_id):
                holders[captured_layer] = hidden_from_pre_hook(args, kwargs)[:, -1, :].detach().cpu().float()

            handles.append(layers[layer_id].register_forward_pre_hook(capture_hook, with_kwargs=True))
        input_ids = attention_mask = outputs = None
        try:
            input_ids, attention_mask = RUNNER.batch_inputs(
                batch,
                pad_token_id=int(tokenizer.pad_token_id),
                device=device,
            )
            with torch.inference_mode():
                outputs = RUNNER.forward_model(model, input_ids=input_ids, attention_mask=attention_mask)
        finally:
            for handle in handles:
                handle.remove()
        if set(holders) != set(captures):
            raise RuntimeError(f"{progress_label}: one or more block-input hooks did not fire")
        for layer_id, state in holders.items():
            captures[layer_id].append(state)
        del input_ids, attention_mask, outputs
        if batch_index == total or batch_index % max(total // 8, 1) == 0:
            print(f"{progress_label}: {batch_index}/{total} batches", flush=True)
    RUNNER.clear_cuda()
    return {layer_id: torch.cat(values, dim=0).contiguous() for layer_id, values in captures.items()}


def metrics_from_logits(logits: torch.Tensor, *, tool_id: int) -> dict[str, torch.Tensor]:
    target = logits[:, int(tool_id)]
    ranks = (logits > target.unsqueeze(-1)).sum(dim=-1) + 1
    ties = (logits == target.unsqueeze(-1)).sum(dim=-1)
    top1 = logits.argmax(dim=-1)
    probability = torch.exp(target - torch.logsumexp(logits, dim=-1))
    non_tool = logits.clone()
    non_tool[:, int(tool_id)] = -torch.inf
    margin = target - non_tool.max(dim=-1).values
    return {
        "tool_call_logit": target.detach().cpu(),
        "tool_call_probability": probability.detach().cpu(),
        "tool_call_rank": ranks.detach().cpu(),
        "target_logit_tie_count": ties.detach().cpu(),
        "top1_token_id": top1.detach().cpu(),
        "margin_vs_best_non_tool": margin.detach().cpu(),
    }


def evaluate_condition(
    model: Any,
    tokenizer: Any,
    items: Sequence[Any],
    *,
    tool_id: int,
    batch_size: int,
    condition: str,
    layer_id: int | None,
    layer: torch.nn.Module | None,
    delta_cpu: torch.Tensor | None,
    sign: float | None,
    progress_label: str,
) -> list[dict[str, Any]]:
    """Evaluate a V0 baseline or a final-token block-input intervention."""

    if (layer_id is None) != (layer is None):
        raise ValueError("layer_id and layer must either both be present or both be absent")
    if layer is None and (delta_cpu is not None or sign is not None):
        raise ValueError("A baseline condition cannot include a delta")
    if layer is not None and (delta_cpu is None or sign not in {-1.0, 1.0}):
        raise ValueError("An intervention requires a vector and a +/-1 sign")
    if not items:
        raise ValueError("Cannot evaluate zero prompts")

    device = RUNNER.model_device(model)
    vector_gpu = delta_cpu.to(device=device) if delta_cpu is not None else None
    rows: list[dict[str, Any]] = []
    total = math.ceil(len(items) / batch_size)
    for batch_index, batch in enumerate(RUNNER.chunks(items, batch_size), start=1):
        handle = None
        if layer is not None:

            def intervention_hook(_module, args, kwargs):
                hidden = hidden_from_pre_hook(args, kwargs)
                delta = vector_gpu.to(device=hidden.device, dtype=hidden.dtype)
                edited = hidden.clone()
                edited[:, -1, :] = edited[:, -1, :] + float(sign) * delta
                if args:
                    return (edited, *args[1:]), kwargs
                updated_kwargs = dict(kwargs)
                updated_kwargs["hidden_states"] = edited
                return args, updated_kwargs

            handle = layer.register_forward_pre_hook(intervention_hook, with_kwargs=True)
        input_ids = attention_mask = outputs = logits = stats = None
        try:
            input_ids, attention_mask = RUNNER.batch_inputs(
                batch,
                pad_token_id=int(tokenizer.pad_token_id),
                device=device,
            )
            with torch.inference_mode():
                outputs = RUNNER.forward_model(model, input_ids=input_ids, attention_mask=attention_mask)
                logits = RUNNER.final_logits(outputs)
            stats = metrics_from_logits(logits, tool_id=tool_id)
        finally:
            if handle is not None:
                handle.remove()
        for index, item in enumerate(batch):
            rank = int(stats["tool_call_rank"][index].item())
            tie_count = int(stats["target_logit_tie_count"][index].item())
            top1_token_id = int(stats["top1_token_id"][index].item())
            rows.append(
                {
                    "sample_id": str(item.sample_id),
                    "split": str(item.split),
                    "side": str(item.side),
                    "condition": condition,
                    "intervention_layer": int(layer_id) if layer_id is not None else None,
                    "tool_call_token_id": int(tool_id),
                    "tool_call_logit": float(stats["tool_call_logit"][index].item()),
                    "tool_call_probability": float(stats["tool_call_probability"][index].item()),
                    "tool_call_rank": rank,
                    "target_logit_tie_count": tie_count,
                    "tool_call_top1_argmax": bool(top1_token_id == int(tool_id)),
                    "tool_call_top1_strict": bool(rank == 1 and tie_count == 1),
                    "top1_token_id": top1_token_id,
                    "margin_vs_best_non_tool": float(stats["margin_vs_best_non_tool"][index].item()),
                    "input_ids_sha256": str(item.input_ids_sha256),
                    "source_prompt_sha256": str(item.source_prompt_sha256),
                }
            )
        del input_ids, attention_mask, outputs, logits, stats
        if batch_index == total or batch_index % max(total // 8, 1) == 0:
            print(f"{progress_label}: {batch_index}/{total} batches", flush=True)
    RUNNER.clear_cuda()
    return rows


def audit_v0_baseline(rows: Sequence[dict[str, Any]], records: Sequence[Any], *, side: str) -> dict[str, Any]:
    expected = {str(record.sample_id): bool(record.manifest[f"{side}_is_tool_top1"]) for record in records}
    observed = {str(row["sample_id"]): bool(row["tool_call_top1_argmax"]) for row in rows}
    if set(observed) != set(expected):
        raise RuntimeError(f"{side}: V0 sample IDs do not match the requested v5 split")
    mismatches = [
        {"sample_id": sample_id, "manifest": expected[sample_id], "observed": observed[sample_id]}
        for sample_id in expected
        if expected[sample_id] != observed[sample_id]
    ]
    result = {
        "side": side,
        "n": len(expected),
        "manifest_tool_top1_count": sum(expected.values()),
        "rerun_tool_top1_count": sum(observed.values()),
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
        "status": "exact_match" if not mismatches else "failed",
    }
    if mismatches:
        raise RuntimeError(f"{side}: V0 behavior replay disagrees with v5 for {len(mismatches)} items")
    return result


def summarize_rows(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("Cannot summarize zero rows")
    probabilities = sorted(float(row["tool_call_probability"]) for row in rows)
    middle = len(probabilities) // 2
    median_probability = (
        probabilities[middle]
        if len(probabilities) % 2
        else (probabilities[middle - 1] + probabilities[middle]) / 2.0
    )
    return {
        "n": len(rows),
        "tool_call_top1_count": sum(bool(row["tool_call_top1_strict"]) for row in rows),
        "tool_call_top1_rate": rate(rows, "tool_call_top1_strict"),
        "argmax_tool_call_top1_count": sum(bool(row["tool_call_top1_argmax"]) for row in rows),
        "mean_tool_call_logit": rate(rows, "tool_call_logit"),
        "mean_tool_call_probability": rate(rows, "tool_call_probability"),
        "median_tool_call_probability": median_probability,
        "mean_tool_call_rank": rate(rows, "tool_call_rank"),
        "tool_call_top3_rate": sum(int(row["tool_call_rank"]) <= 3 for row in rows) / len(rows),
        "target_logit_tie_count": sum(int(row["target_logit_tie_count"]) > 1 for row in rows),
    }


def paired_effect(
    baseline: Sequence[dict[str, Any]],
    intervened: Sequence[dict[str, Any]],
    *,
    direction: str,
) -> dict[str, Any]:
    """Summarize strict flips/drops against the matching V0 baseline."""

    if direction not in {"flip", "drop"}:
        raise ValueError(f"Unknown paired effect direction {direction!r}")
    base_by_id = {str(row["sample_id"]): row for row in baseline}
    edited_by_id = {str(row["sample_id"]): row for row in intervened}
    if set(base_by_id) != set(edited_by_id):
        raise RuntimeError("Baseline and intervention do not cover the same held-out sample IDs")
    if direction == "flip":
        eligible = [sample_id for sample_id, row in base_by_id.items() if not bool(row["tool_call_top1_strict"])]
        successes = [sample_id for sample_id in eligible if bool(edited_by_id[sample_id]["tool_call_top1_strict"])]
    else:
        eligible = [sample_id for sample_id, row in base_by_id.items() if bool(row["tool_call_top1_strict"])]
        successes = [sample_id for sample_id in eligible if not bool(edited_by_id[sample_id]["tool_call_top1_strict"])]
    baseline_summary = summarize_rows(baseline)
    intervened_summary = summarize_rows(intervened)
    return {
        "effect": direction,
        "eligible_count": len(eligible),
        "success_count": len(successes),
        "success_rate": len(successes) / len(eligible) if eligible else None,
        "before": baseline_summary,
        "after": intervened_summary,
        "mean_tool_logit_change": (
            float(intervened_summary["mean_tool_call_logit"]) - float(baseline_summary["mean_tool_call_logit"])
        ),
    }


def format_rate(value: float | None) -> str:
    return "--" if value is None else f"{100.0 * value:.1f}%"


def render_report(results: dict[str, Any]) -> str:
    lines = [
        "# v5 adjacent-layer mean-difference sweep",
        "",
        f"Each raw clean-minus-corrupt vector is fit on {int(results['n_train_pairs'])} Qwen3-8B v5 training pairs at the decoder-block input (`resid_pre`) of that layer. Every intervention uses the same frozen vector on {int(results['n_heldout_pairs'])} disjoint held-out pairs.",
        "",
        "|Layer|$\\|\\mu_\\Delta\\|$|Corrupt +$\\mu_\\Delta$: top-1|strict flips|Corrupt +random: top-1|Clean -$\\mu_\\Delta$: top-1|strict drops|Clean -random: top-1|",
        "|-:|-:|-:|-:|-:|-:|-:|-:|",
    ]
    for layer_key in sorted(results["layers"], key=int):
        row = results["layers"][layer_key]
        add_mu = row["corrupt_add_mu"]
        add_random = row["corrupt_add_random"]
        remove_mu = row["clean_subtract_mu"]
        remove_random = row["clean_subtract_random"]
        lines.append(
            "|"
            + "|".join(
                (
                    f"L{layer_key}",
                    f"{float(row['vector']['l2_norm']):.3f}",
                    format_rate(float(add_mu["after"]["tool_call_top1_rate"])),
                    f"{int(add_mu['success_count'])}/{int(add_mu['eligible_count'])}",
                    format_rate(float(add_random["after"]["tool_call_top1_rate"])),
                    format_rate(float(remove_mu["after"]["tool_call_top1_rate"])),
                    f"{int(remove_mu['success_count'])}/{int(remove_mu['eligible_count'])}",
                    format_rate(float(remove_random["after"]["tool_call_top1_rate"])),
                )
            )
            + "|"
        )
    lines.extend(
        [
            "",
            "`top-1` is strict: the tool-call logit must be uniquely rank 1. Random directions are independently seeded unit vectors rescaled to the corresponding $\\|\\mu_\\Delta\\|$.",
        ]
    )
    return "\n".join(lines) + "\n"


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_root = args.output_root.resolve()
    dataset_root = args.dataset_root.resolve()
    spec = RUNNER.MODEL_SPECS[MODEL_KEY]
    family = str(spec["family"])
    expected_model_path = Path(str(spec["model_path"])).resolve()
    model_path = (args.model_path or expected_model_path).resolve()
    requested_layers = tuple(int(layer) for layer in args.layers)

    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite existing result directory: {output_root}")
    if not dataset_root.is_dir():
        raise FileNotFoundError(dataset_root)
    if model_path != expected_model_path:
        raise ValueError(f"v5 {MODEL_KEY} requires model path {expected_model_path}, received {model_path}")
    if not requested_layers or len(requested_layers) != len(set(requested_layers)):
        raise ValueError("--layers must contain one or more distinct layer IDs")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if not args.allow_subset and (int(args.train_pairs), int(args.heldout_pairs)) != (200, 300):
        raise ValueError("Canonical v5 adjacent-layer run requires --train-pairs 200 --heldout-pairs 300")

    summary, records = RUNNER.load_pairs(dataset_root, model_key=MODEL_KEY, family=family)
    if Path(str(summary.get("model_path"))).resolve() != expected_model_path:
        raise RuntimeError("v5 Qwen3-8B summary model path does not match the required model")
    train_records = [record for record in records if str(record.split) == "train"]
    heldout_records = [record for record in records if str(record.split) == "heldout"]
    if not 0 < int(args.train_pairs) <= len(train_records):
        raise ValueError(f"--train-pairs must be in 1..{len(train_records)}")
    if not 0 < int(args.heldout_pairs) <= len(heldout_records):
        raise ValueError(f"--heldout-pairs must be in 1..{len(heldout_records)}")
    train_records = train_records[: int(args.train_pairs)]
    heldout_records = heldout_records[: int(args.heldout_pairs)]
    manifest_before = RUNNER.sha256_file(dataset_root / "manifest.jsonl")
    summary_before = RUNNER.sha256_file(dataset_root / "summary.json")

    output_root.mkdir(parents=True, exist_ok=False)
    RUNNER.write_json(
        output_root / "run_config.json",
        {
            "experiment": "reviewer_lxey_v5_neighbor_layers",
            "created_at": now_utc(),
            "model_key": MODEL_KEY,
            "model_label": str(spec["label"]),
            "model_path": str(model_path),
            "dataset_root": str(dataset_root),
            "dataset_version": str(summary["dataset_version"]),
            "layers": list(requested_layers),
            "train_pairs": len(train_records),
            "heldout_pairs": len(heldout_records),
            "partition_rule": "all v5 train / all v5 held-out in canonical mode; manifest-order prefixes only under --allow-subset",
            "vector_protocol": "mean(clean block-input state - corrupt block-input state) at final non-padding prompt token",
            "intervention_protocol": "add each frozen vector to corrupt prompts and subtract it from clean prompts at the same decoder-block input / final prompt token",
            "control_protocol": "one independently seeded random unit vector per layer, scaled to that layer's raw mean-difference norm; both add and subtract arms",
            "top1_definition": "tool logit is uniquely rank 1; exact ties are losses",
            "v0_input_rule": "native release text encoded with add_special_tokens=False for Qwen3; no prompt mutations",
            "batch_size": int(args.batch_size),
            "dtype": str(args.dtype),
            "attn_implementation": str(args.attn_implementation) or None,
            "seed": int(args.seed),
            "allow_subset": bool(args.allow_subset),
            "script_path": str(Path(__file__).resolve()),
            "script_sha256": RUNNER.sha256_file(Path(__file__).resolve()),
            "native_input_helper_path": str(DEPENDENCY_RUNNER_PATH),
            "native_input_helper_sha256": RUNNER.sha256_file(DEPENDENCY_RUNNER_PATH),
            "git": RUNNER.git_provenance(),
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "gpu_before": RUNNER.gpu_snapshot(),
        },
    )
    RUNNER.write_json(
        output_root / "dataset_provenance.json",
        {
            "dataset_version": str(summary["dataset_version"]),
            "dataset_root": str(dataset_root),
            "summary_sha256": summary_before,
            "manifest_sha256": manifest_before,
            "train_sample_ids": [str(record.sample_id) for record in train_records],
            "heldout_sample_ids": [str(record.sample_id) for record in heldout_records],
            "disjoint": not set(record.sample_id for record in train_records).intersection(
                record.sample_id for record in heldout_records
            ),
            "train_prompt_hashes": [
                {
                    "sample_id": str(record.sample_id),
                    "clean_prompt_sha256": str(record.clean.prompt_sha256),
                    "corrupt_prompt_sha256": str(record.corrupt.prompt_sha256),
                }
                for record in train_records
            ],
            "heldout_prompt_hashes": [
                {
                    "sample_id": str(record.sample_id),
                    "clean_prompt_sha256": str(record.clean.prompt_sha256),
                    "corrupt_prompt_sha256": str(record.corrupt.prompt_sha256),
                }
                for record in heldout_records
            ],
        },
    )

    tokenizer = None
    model = None
    try:
        torch.manual_seed(int(args.seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(args.seed))
            torch.cuda.reset_peak_memory_stats()
        loader_args = argparse.Namespace(
            model_path=model_path,
            dtype=args.dtype,
            attn_implementation=args.attn_implementation,
        )
        print(json.dumps({"event": "load_model", "model": spec["label"], "batch_size": args.batch_size}), flush=True)
        tokenizer, model, loaded_model_path = RUNNER.load_tokenizer_and_model(spec, loader_args)
        if Path(loaded_model_path).resolve() != model_path:
            raise RuntimeError("Model loader returned an unexpected path")
        tool_id, tool_info = RUNNER.tool_token_id(
            tokenizer,
            marker=str(summary["tool_call_marker"]),
            family=family,
        )
        if int(tool_id) != int(summary["tool_call_token_id"]):
            raise RuntimeError("Tokenizer's tool-call token ID differs from the v5 release")
        RUNNER.write_json(output_root / "tool_token.json", tool_info)

        train_items, train_provenance = build_v0_items(train_records, family=family, tokenizer=tokenizer)
        heldout_items, heldout_provenance = build_v0_items(heldout_records, family=family, tokenizer=tokenizer)
        for row in train_provenance:
            row["partition"] = "vector_fit_train"
        for row in heldout_provenance:
            row["partition"] = "heldout"
        RUNNER.write_jsonl(output_root / "input_provenance.jsonl", train_provenance + heldout_provenance)

        decoder_layers = RUNNER.get_layers(model)
        invalid_layers = [layer for layer in requested_layers if layer < 0 or layer >= len(decoder_layers)]
        if invalid_layers:
            raise ValueError(f"Requested layers outside model's 0..{len(decoder_layers) - 1}: {invalid_layers}")

        train_clean_states = capture_pre_states(
            model,
            tokenizer,
            train_items["clean"],
            layers=decoder_layers,
            layer_ids=requested_layers,
            batch_size=int(args.batch_size),
            progress_label="v5 train clean block-input capture",
        )
        train_corrupt_states = capture_pre_states(
            model,
            tokenizer,
            train_items["corrupt"],
            layers=decoder_layers,
            layer_ids=requested_layers,
            batch_size=int(args.batch_size),
            progress_label="v5 train corrupt block-input capture",
        )

        vectors: dict[int, torch.Tensor] = {}
        random_vectors: dict[int, torch.Tensor] = {}
        vector_summaries: dict[int, dict[str, Any]] = {}
        vector_root = output_root / "vectors"
        vector_root.mkdir(parents=True, exist_ok=False)
        for layer_id in requested_layers:
            differences = (train_clean_states[layer_id] - train_corrupt_states[layer_id]).contiguous().float()
            vector = differences.mean(dim=0).contiguous().float()
            if not bool(torch.isfinite(vector).all()) or float(vector.norm().item()) <= 0.0:
                raise RuntimeError(f"L{layer_id}: invalid v5 mean-difference vector")
            random_vector, random_unit = make_random_equal_norm(vector, seed=int(args.seed) + 10_000 + layer_id)
            vectors[layer_id] = vector
            random_vectors[layer_id] = random_vector
            vector_summaries[layer_id] = {
                "layer": layer_id,
                "hook_location": "decoder-block input (resid_pre)",
                "n_train_pairs": len(train_records),
                "hidden_size": int(vector.numel()),
                "l2_norm": float(vector.norm().item()),
                "rms": float(vector.square().mean().sqrt().item()),
                "mean_pair_delta_l2_norm": float(differences.norm(dim=1).mean().item()),
                "random_seed": int(args.seed) + 10_000 + layer_id,
                "random_l2_norm": float(random_vector.norm().item()),
                "train_sample_ids": [str(record.sample_id) for record in train_records],
            }
            torch.save(
                {
                    "mean_diff": vector,
                    "random_equal_norm": random_vector,
                    "random_unit": random_unit,
                    **vector_summaries[layer_id],
                },
                vector_root / f"L{layer_id}_mean_clean_minus_corrupt.pt",
            )
            del differences
        reference = vectors[24] if 24 in vectors else vectors[requested_layers[0]]
        for layer_id, vector_summary in vector_summaries.items():
            vector_summary["cosine_to_L24" if 24 in vectors else "cosine_to_reference_layer"] = cosine_similarity(
                vectors[layer_id], reference
            )
        RUNNER.write_json(output_root / "vector_summaries.json", {str(key): value for key, value in vector_summaries.items()})
        del train_clean_states, train_corrupt_states
        RUNNER.clear_cuda()

        clean_baseline = evaluate_condition(
            model,
            tokenizer,
            heldout_items["clean"],
            tool_id=int(tool_id),
            batch_size=int(args.batch_size),
            condition="baseline_clean",
            layer_id=None,
            layer=None,
            delta_cpu=None,
            sign=None,
            progress_label="v5 held-out clean baseline",
        )
        corrupt_baseline = evaluate_condition(
            model,
            tokenizer,
            heldout_items["corrupt"],
            tool_id=int(tool_id),
            batch_size=int(args.batch_size),
            condition="baseline_corrupt",
            layer_id=None,
            layer=None,
            delta_cpu=None,
            sign=None,
            progress_label="v5 held-out corrupt baseline",
        )
        baseline_audit = {
            "clean": audit_v0_baseline(clean_baseline, heldout_records, side="clean"),
            "corrupt": audit_v0_baseline(corrupt_baseline, heldout_records, side="corrupt"),
        }
        RUNNER.write_json(output_root / "baseline_screening_replay.json", baseline_audit)

        all_rows: list[dict[str, Any]] = clean_baseline + corrupt_baseline
        result_layers: dict[str, Any] = {}
        for layer_id in requested_layers:
            layer = decoder_layers[layer_id]
            print(f"[L{layer_id}] evaluating v5 mean-difference and equal-norm random controls", flush=True)
            corrupt_add_mu = evaluate_condition(
                model,
                tokenizer,
                heldout_items["corrupt"],
                tool_id=int(tool_id),
                batch_size=int(args.batch_size),
                condition="corrupt_add_mu_delta",
                layer_id=layer_id,
                layer=layer,
                delta_cpu=vectors[layer_id],
                sign=1.0,
                progress_label=f"L{layer_id} held-out corrupt +mu",
            )
            corrupt_add_random = evaluate_condition(
                model,
                tokenizer,
                heldout_items["corrupt"],
                tool_id=int(tool_id),
                batch_size=int(args.batch_size),
                condition="corrupt_add_random_equal_norm",
                layer_id=layer_id,
                layer=layer,
                delta_cpu=random_vectors[layer_id],
                sign=1.0,
                progress_label=f"L{layer_id} held-out corrupt +random",
            )
            clean_subtract_mu = evaluate_condition(
                model,
                tokenizer,
                heldout_items["clean"],
                tool_id=int(tool_id),
                batch_size=int(args.batch_size),
                condition="clean_subtract_mu_delta",
                layer_id=layer_id,
                layer=layer,
                delta_cpu=vectors[layer_id],
                sign=-1.0,
                progress_label=f"L{layer_id} held-out clean -mu",
            )
            clean_subtract_random = evaluate_condition(
                model,
                tokenizer,
                heldout_items["clean"],
                tool_id=int(tool_id),
                batch_size=int(args.batch_size),
                condition="clean_subtract_random_equal_norm",
                layer_id=layer_id,
                layer=layer,
                delta_cpu=random_vectors[layer_id],
                sign=-1.0,
                progress_label=f"L{layer_id} held-out clean -random",
            )
            all_rows.extend(corrupt_add_mu + corrupt_add_random + clean_subtract_mu + clean_subtract_random)
            result_layers[str(layer_id)] = {
                "vector": vector_summaries[layer_id],
                "corrupt_add_mu": paired_effect(corrupt_baseline, corrupt_add_mu, direction="flip"),
                "corrupt_add_random": paired_effect(corrupt_baseline, corrupt_add_random, direction="flip"),
                "clean_subtract_mu": paired_effect(clean_baseline, clean_subtract_mu, direction="drop"),
                "clean_subtract_random": paired_effect(clean_baseline, clean_subtract_random, direction="drop"),
            }

        result = {
            "experiment": "reviewer_lxey_v5_neighbor_layers",
            "model_key": MODEL_KEY,
            "model_label": str(spec["label"]),
            "dataset_version": str(summary["dataset_version"]),
            "n_train_pairs": len(train_records),
            "n_heldout_pairs": len(heldout_records),
            "layers": result_layers,
            "baseline": {
                "clean": summarize_rows(clean_baseline),
                "corrupt": summarize_rows(corrupt_baseline),
            },
            "baseline_screening_replay": baseline_audit,
            "gpu_after": RUNNER.gpu_snapshot(),
        }
        RUNNER.write_csv(output_root / "per_sample_metrics.csv", all_rows)
        RUNNER.write_json(output_root / "summary.json", result)
        report = render_report(result)
        (output_root / "summary.md").write_text(report, encoding="utf-8")

        manifest_after = RUNNER.sha256_file(dataset_root / "manifest.jsonl")
        summary_after = RUNNER.sha256_file(dataset_root / "summary.json")
        if manifest_before != manifest_after or summary_before != summary_after:
            raise RuntimeError("The v5 release metadata changed while the experiment was running")
        RUNNER.write_json(
            output_root / "completion.json",
            {
                "status": "complete",
                "completed_at": now_utc(),
                "n_train_pairs": len(train_records),
                "n_heldout_pairs": len(heldout_records),
                "layers": list(requested_layers),
                "baseline_replay_status": {side: audit["status"] for side, audit in baseline_audit.items()},
                "summary": str(output_root / "summary.json"),
                "report": str(output_root / "summary.md"),
            },
        )
        return result
    finally:
        del model
        RUNNER.clear_cuda()


def main() -> None:
    args = parse_args()
    try:
        result = run(args)
    except Exception as exc:
        output_root = args.output_root.resolve()
        if output_root.exists():
            RUNNER.write_json(
                output_root / "failure.json",
                {
                    "status": "failed",
                    "failed_at": now_utc(),
                    "error": repr(exc),
                    "traceback": traceback.format_exc(),
                },
            )
        raise
    print(render_report(result), flush=True)
    print(json.dumps({"status": "complete", "output_root": str(args.output_root.resolve())}), flush=True)


if __name__ == "__main__":
    main()
