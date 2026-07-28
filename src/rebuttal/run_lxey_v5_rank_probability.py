#!/usr/bin/env python3
"""Measure first-token tool-call rank and probability on v5 held-out prompts.

This is the rank/probability analysis requested by Reviewer LxEy, kept
separate from the schema-identity intervention runner.  It replays the native
V0 corrupt prompt for every model-specific v5 held-out pair and records the
full per-sample rank/probability distribution.  Mistral consumes its stored
native IDs; text-model prompts are encoded directly from their release files.

The runner deliberately records enough provenance to distinguish a fresh v5
measurement from the historical v2 values previously drafted in the rebuttal.
It never changes a dataset file and refuses to overwrite an existing run root.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "datasets" / "v5_model_specific_balanced"
DEPENDENCY_RUNNER_PATH = PROJECT_ROOT / "src" / "rebuttal" / "run_cjrq_v5_tool_identity.py"
MODEL_ORDER = (
    "qwen3_4b",
    "qwen3_8b",
    "qwen3_14b",
    "qwen35_4b",
    "qwen35_9b",
    "mistral_3p2_24b",
    "granite_3p3_8b",
)
TABLE_LABELS = {
    "qwen3_4b": "Qwen3-4B",
    "qwen3_8b": "Qwen3-8B",
    "qwen3_14b": "Qwen3-14B",
    "qwen35_4b": "Qwen3.5-4B",
    "qwen35_9b": "Qwen3.5-9B",
    "mistral_3p2_24b": "Mistral",
    "granite_3p3_8b": "Granite",
}
PROBABILITY_TOLERANCE = 5e-5


def load_dependency_runner():
    spec = importlib.util.spec_from_file_location("cjrq_v5_runner_for_lxey_rank", DEPENDENCY_RUNNER_PATH)
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
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT, type=Path)
    parser.add_argument("--models", nargs="+", choices=MODEL_ORDER, default=list(MODEL_ORDER))
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--attn-implementation", default="")
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--heldout-pairs", type=int, default=300)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue an interrupted run root, skipping only models with a complete local completion record.",
    )
    parser.add_argument(
        "--allow-subset",
        action="store_true",
        help="Permit a non-canonical prefix of held-out pairs for a smoke test.",
    )
    return parser.parse_args()


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def median(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("Cannot compute a median of zero values")
    ordered = sorted(float(value) for value in values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def evaluate_items(
    model,
    tokenizer,
    items: Sequence[Any],
    *,
    tool_id: int,
    batch_size: int,
    progress_label: str,
) -> list[dict[str, Any]]:
    """Replay V0 prompts and retain tie-aware rank statistics."""

    if not items:
        raise ValueError("Cannot evaluate zero prompt items")
    device = RUNNER.model_device(model)
    rows: list[dict[str, Any]] = []
    total = math.ceil(len(items) / batch_size)
    for batch_number, batch in enumerate(RUNNER.chunks(items, batch_size), start=1):
        input_ids, attention_mask = RUNNER.batch_inputs(
            batch,
            pad_token_id=int(tokenizer.pad_token_id),
            device=device,
        )
        with torch.inference_mode():
            outputs = RUNNER.forward_model(model, input_ids=input_ids, attention_mask=attention_mask)
            logits = RUNNER.final_logits(outputs)
        target = logits[:, int(tool_id)]
        ranks = (logits > target.unsqueeze(-1)).sum(dim=-1) + 1
        target_tie_counts = (logits == target.unsqueeze(-1)).sum(dim=-1)
        top1 = logits.argmax(dim=-1)
        probabilities = torch.exp(target - torch.logsumexp(logits, dim=-1))
        for index, item in enumerate(batch):
            rank = int(ranks[index].item())
            tie_count = int(target_tie_counts[index].item())
            argmax_top1 = int(top1[index].item()) == int(tool_id)
            rows.append(
                {
                    "sample_id": str(item.sample_id),
                    "split": str(item.split),
                    "side": str(item.side),
                    "tool_call_token_id": int(tool_id),
                    "tool_call_logit": float(target[index].item()),
                    "tool_call_probability": float(probabilities[index].item()),
                    "tool_call_rank": rank,
                    "target_logit_tie_count": tie_count,
                    "tool_call_top1_argmax": argmax_top1,
                    "tool_call_top1_strict": bool(rank == 1 and tie_count == 1),
                    "top1_token_id": int(top1[index].item()),
                    "input_ids_sha256": str(item.input_ids_sha256),
                    "source_prompt_sha256": str(item.source_prompt_sha256),
                    "render_method": str(item.render_method),
                    "v0_template_roundtrip": item.v0_template_roundtrip,
                }
            )
        del input_ids, attention_mask, outputs, logits, target, ranks, target_tie_counts, top1, probabilities
        if batch_number == total or batch_number % max(total // 8, 1) == 0:
            print(f"{progress_label}: {batch_number}/{total} batches", flush=True)
    RUNNER.clear_cuda()
    return rows


def rank_bins(rows: Sequence[dict[str, Any]]) -> dict[str, int]:
    result = {str(rank): 0 for rank in range(1, 11)}
    result.update({"11-20": 0, "21-50": 0, "51-100": 0, "101+": 0})
    for row in rows:
        rank = int(row["tool_call_rank"])
        if rank <= 10:
            result[str(rank)] += 1
        elif rank <= 20:
            result["11-20"] += 1
        elif rank <= 50:
            result["21-50"] += 1
        elif rank <= 100:
            result["51-100"] += 1
        else:
            result["101+"] += 1
    return result


def summarize_rows(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("Cannot summarize zero rows")
    probabilities = [float(row["tool_call_probability"]) for row in rows]
    exact_histogram = Counter(int(row["tool_call_rank"]) for row in rows)
    return {
        "n": len(rows),
        "top1_count": sum(bool(row["tool_call_top1_strict"]) for row in rows),
        "top1_rate": sum(bool(row["tool_call_top1_strict"]) for row in rows) / len(rows),
        "top3_count": sum(int(row["tool_call_rank"]) <= 3 for row in rows),
        "top3_rate": sum(int(row["tool_call_rank"]) <= 3 for row in rows) / len(rows),
        "top10_count": sum(int(row["tool_call_rank"]) <= 10 for row in rows),
        "top10_rate": sum(int(row["tool_call_rank"]) <= 10 for row in rows) / len(rows),
        "median_probability": median(probabilities),
        "mean_probability": sum(probabilities) / len(probabilities),
        "rank_2_count": sum(int(row["tool_call_rank"]) == 2 for row in rows),
        "probability_at_least_0p30_count": sum(probability >= 0.30 for probability in probabilities),
        "target_logit_tie_count": sum(int(row["target_logit_tie_count"]) > 1 for row in rows),
        "rank_bins": rank_bins(rows),
        "exact_rank_histogram": {str(rank): count for rank, count in sorted(exact_histogram.items())},
    }


def verify_replay(
    rows: Sequence[dict[str, Any]],
    records: Sequence[Any],
) -> dict[str, Any]:
    expected = {str(record.sample_id): record.manifest for record in records}
    observed = {str(row["sample_id"]): row for row in rows}
    if set(observed) != set(expected):
        missing = sorted(set(expected) - set(observed))
        unexpected = sorted(set(observed) - set(expected))
        raise RuntimeError(f"Held-out membership mismatch: missing={missing[:5]}, unexpected={unexpected[:5]}")
    argmax_mismatches: list[dict[str, Any]] = []
    rank_mismatches: list[dict[str, Any]] = []
    probability_mismatches: list[dict[str, Any]] = []
    maximum_probability_delta = 0.0
    for sample_id, manifest in expected.items():
        row = observed[sample_id]
        expected_top1 = bool(manifest["corrupt_is_tool_top1"])
        expected_rank = int(manifest["corrupt_tool_rank"])
        expected_probability = float(manifest["corrupt_tool_probability"])
        probability_delta = abs(float(row["tool_call_probability"]) - expected_probability)
        maximum_probability_delta = max(maximum_probability_delta, probability_delta)
        if bool(row["tool_call_top1_argmax"]) != expected_top1:
            argmax_mismatches.append(
                {
                    "sample_id": sample_id,
                    "manifest": expected_top1,
                    "observed": bool(row["tool_call_top1_argmax"]),
                    "observed_top1_token_id": int(row["top1_token_id"]),
                }
            )
        if int(row["tool_call_rank"]) != expected_rank:
            rank_mismatches.append(
                {
                    "sample_id": sample_id,
                    "manifest": expected_rank,
                    "observed": int(row["tool_call_rank"]),
                }
            )
        if probability_delta > PROBABILITY_TOLERANCE:
            probability_mismatches.append(
                {
                    "sample_id": sample_id,
                    "manifest": expected_probability,
                    "observed": float(row["tool_call_probability"]),
                    "absolute_delta": probability_delta,
                }
            )
        row["manifest_tool_call_rank"] = expected_rank
        row["manifest_tool_call_probability"] = expected_probability
        row["manifest_tool_call_top1_argmax"] = expected_top1
        row["probability_absolute_delta_vs_manifest"] = probability_delta
    behavior_exact = not argmax_mismatches
    result = {
        "n": len(rows),
        "argmax_top1_mismatch_count": len(argmax_mismatches),
        "rank_mismatch_count": len(rank_mismatches),
        "probability_mismatch_count": len(probability_mismatches),
        "probability_tolerance": PROBABILITY_TOLERANCE,
        "maximum_probability_absolute_delta": maximum_probability_delta,
        "argmax_top1_mismatches": argmax_mismatches,
        "rank_mismatches": rank_mismatches,
        "probability_mismatches": probability_mismatches,
    }
    # The v5 release's explicit acceptance/replay contract is the V0 first
    # token behavior.  Construction-time rank/probability records remain
    # useful audit references, but they are not a release invariant across
    # inference environments.  The fresh, fixed analysis layout below is the
    # source of the reported rank and probability values.
    result["behavior_status"] = "exact_match" if behavior_exact else "failed"
    result["rank_vs_construction_status"] = "exact_match" if not rank_mismatches else "differs_from_construction"
    result["probability_vs_construction_status"] = (
        "within_tolerance" if not probability_mismatches else "environment_sensitive"
    )
    result["status"] = "complete" if behavior_exact else "failed"
    return result


def heldout_items(records: Sequence[Any], *, family: str, tokenizer: Any) -> tuple[list[Any], list[dict[str, Any]]]:
    items: list[Any] = []
    provenance: list[dict[str, Any]] = []
    for record in records:
        item = RUNNER.render_prompt_item(
            record.corrupt,
            sample_id=str(record.sample_id),
            split=str(record.split),
            side="corrupt",
            variant="V0",
            family=family,
            tokenizer=tokenizer,
        )
        items.append(item)
        provenance.append(
            {
                "sample_id": str(item.sample_id),
                "split": str(item.split),
                "side": "corrupt",
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


def run_model(args: argparse.Namespace, *, output_root: Path, model_key: str, dataset_root: Path) -> dict[str, Any]:
    model_root = output_root / model_key
    if model_root.exists():
        if any(model_root.iterdir()):
            raise FileExistsError(f"{model_key}: refusing to overwrite nonempty result directory {model_root}")
    else:
        model_root.mkdir(parents=False, exist_ok=False)
    spec = RUNNER.MODEL_SPECS[model_key]
    family = str(spec["family"])
    model_dataset_root = dataset_root / model_key
    summary, all_records = RUNNER.load_pairs(model_dataset_root, model_key=model_key, family=family)
    expected_model_path = Path(str(spec["model_path"])).resolve()
    if Path(str(summary.get("model_path"))).resolve() != expected_model_path:
        raise RuntimeError(f"{model_key}: v5 summary model path disagrees with model specification")
    selected_records = [record for record in all_records if str(record.split) == "heldout"]
    if args.heldout_pairs <= 0 or args.heldout_pairs > len(selected_records):
        raise ValueError(f"{model_key}: --heldout-pairs must be in 1..{len(selected_records)}")
    selected_records = selected_records[: int(args.heldout_pairs)]
    if not args.allow_subset and len(selected_records) != 300:
        raise ValueError("Canonical v5 analysis requires all 300 held-out pairs")
    manifest_before = RUNNER.sha256_file(model_dataset_root / "manifest.jsonl")
    summary_before = RUNNER.sha256_file(model_dataset_root / "summary.json")
    run_config = {
        "experiment": "reviewer_lxey_v5_rank_probability",
        "created_at": now_utc(),
        "model_key": model_key,
        "model_label": str(spec["label"]),
        "v5_summary_model_label": str(summary["model_label"]),
        "model_path": str(expected_model_path),
        "dataset_root": str(model_dataset_root),
        "dataset_version": str(summary["dataset_version"]),
        "side": "corrupt",
        "split": "heldout",
        "heldout_pairs": len(selected_records),
        "heldout_selection": "full manifest heldout split" if len(selected_records) == 300 else "manifest-order prefix for smoke test",
        "batch_size": int(spec["batch_size"]),
        "dtype": str(args.dtype),
        "attn_implementation": str(args.attn_implementation) or None,
        "seed": int(args.seed),
        "top1_definition": "tool logit is strictly greater than every other vocabulary logit; exact ties are losses",
        "rank_definition": "1 + number of vocabulary logits strictly greater than the tool-call logit",
        "probability_definition": "softmax probability of the native first-token tool-call marker",
        "mistral_rule": "V0 uses stored native input_ids and verifies chat-template roundtrip; it never decodes and re-encodes the prompt",
        "replay_rule": "V5 V0 argmax behavior is the hard release check. Construction-time rank/probability are retained as audit references; this fixed-environment canonical replay supplies the reported rank/probability values.",
        "script_path": str(Path(__file__).resolve()),
        "script_sha256": RUNNER.sha256_file(Path(__file__).resolve()),
        "native_input_helper_path": str(DEPENDENCY_RUNNER_PATH),
        "native_input_helper_sha256": RUNNER.sha256_file(DEPENDENCY_RUNNER_PATH),
        "git": RUNNER.git_provenance(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu_before": RUNNER.gpu_snapshot(),
    }
    RUNNER.write_json(model_root / "run_config.json", run_config)
    provenance = {
        "dataset_version": str(summary["dataset_version"]),
        "model_key": model_key,
        "model_label": str(summary["model_label"]),
        "model_path": str(summary["model_path"]),
        "manifest_sha256": manifest_before,
        "summary_sha256": summary_before,
        "heldout_sample_ids": [str(record.sample_id) for record in selected_records],
        "heldout_prompt_hashes": [
            {
                "sample_id": str(record.sample_id),
                "corrupt_relpath": str(record.corrupt.relpath),
                "corrupt_prompt_sha256": str(record.corrupt.prompt_sha256),
            }
            for record in selected_records
        ],
    }
    RUNNER.write_json(model_root / "dataset_provenance.json", provenance)
    tokenizer = None
    model = None
    try:
        torch.manual_seed(int(args.seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(args.seed))
        loader_args = argparse.Namespace(model_path=None, dtype=args.dtype, attn_implementation=args.attn_implementation)
        tokenizer, model, loaded_model_path = RUNNER.load_tokenizer_and_model(spec, loader_args)
        if Path(loaded_model_path).resolve() != expected_model_path:
            raise RuntimeError(f"{model_key}: loader returned an unexpected model path")
        tool_id, tool_info = RUNNER.tool_token_id(
            tokenizer,
            marker=str(summary["tool_call_marker"]),
            family=family,
        )
        if int(tool_id) != int(summary["tool_call_token_id"]):
            raise RuntimeError(f"{model_key}: native tool marker ID differs from v5 summary")
        if any(int(record.manifest["tool_call_token_id"]) != int(tool_id) for record in selected_records):
            raise RuntimeError(f"{model_key}: held-out manifest contains a different tool marker ID")
        items, input_provenance = heldout_items(selected_records, family=family, tokenizer=tokenizer)
        RUNNER.write_jsonl(model_root / "input_provenance.jsonl", input_provenance)
        rows = evaluate_items(
            model,
            tokenizer,
            items,
            tool_id=int(tool_id),
            batch_size=int(spec["batch_size"]),
            progress_label=f"{model_key} v5 corrupt heldout",
        )
        replay = verify_replay(rows, selected_records)
        RUNNER.write_json(model_root / "baseline_screening_replay.json", replay)
        RUNNER.write_csv(model_root / "sample_metrics.csv", rows)
        metrics = summarize_rows(rows)
        result = {
            "model_key": model_key,
            "model_label": TABLE_LABELS[model_key],
            "tool_call": tool_info,
            "metrics": metrics,
            "baseline_screening_replay": replay,
            "gpu_after": RUNNER.gpu_snapshot(),
        }
        RUNNER.write_json(model_root / "summary.json", result)
        manifest_after = RUNNER.sha256_file(model_dataset_root / "manifest.jsonl")
        summary_after = RUNNER.sha256_file(model_dataset_root / "summary.json")
        if manifest_after != manifest_before or summary_after != summary_before:
            raise RuntimeError(f"{model_key}: v5 release metadata changed while the analysis was running")
        if replay["behavior_status"] != "exact_match":
            raise RuntimeError(
                f"{model_key}: fresh replay differs from v5 screening record "
                f"(argmax={replay['argmax_top1_mismatch_count']})"
            )
        completion = {
            "status": "complete",
            "completed_at": now_utc(),
            "model_key": model_key,
            "n_heldout": len(rows),
            "manifest_sha256": manifest_before,
            "summary_sha256": summary_before,
            "replay_status": replay["status"],
            "behavior_replay_status": replay["behavior_status"],
            "rank_vs_construction_status": replay["rank_vs_construction_status"],
            "probability_vs_construction_status": replay["probability_vs_construction_status"],
        }
        RUNNER.write_json(model_root / "completion.json", completion)
        return result
    finally:
        del model
        RUNNER.clear_cuda()


def format_rate(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def format_probability(value: float) -> str:
    rendered = f"{value:.3g}"
    return rendered[1:] if rendered.startswith("0.") else rendered


def render_table(results: dict[str, dict[str, Any]], models: Sequence[str]) -> str:
    headers = [TABLE_LABELS[model_key] for model_key in models]
    rows = (
        ("Top-1 rate", "top1_rate", format_rate),
        ("Top-3 rate", "top3_rate", format_rate),
        ("Top-10 rate", "top10_rate", format_rate),
        ("Median probability", "median_probability", format_probability),
        ("Mean probability", "mean_probability", format_probability),
    )
    lines = [
        "|Metric|" + "|".join(headers) + "|",
        "|-|" + "|".join("-" for _ in headers) + "|",
    ]
    for label, field, formatter in rows:
        cells = [formatter(float(results[model_key]["metrics"][field])) for model_key in models]
        lines.append("|" + label + "|" + "|".join(cells) + "|")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    output_root = args.output_root.resolve()
    dataset_root = args.dataset_root.resolve()
    models = tuple(args.models)
    if len(models) != len(set(models)):
        raise ValueError("--models must not contain duplicates")
    if output_root.exists() and not args.resume:
        raise FileExistsError(f"Refusing to overwrite existing result directory: {output_root}")
    if not dataset_root.is_dir():
        raise FileNotFoundError(dataset_root)
    if not args.allow_subset and int(args.heldout_pairs) != 300:
        raise ValueError("Canonical v5 analysis requires --heldout-pairs 300")
    if not output_root.exists():
        output_root.mkdir(parents=True, exist_ok=False)
    root_config = {
        "experiment": "reviewer_lxey_v5_rank_probability",
        "created_at": now_utc(),
        "dataset_root": str(dataset_root),
        "models": list(models),
        "dtype": str(args.dtype),
        "heldout_pairs": int(args.heldout_pairs),
        "allow_subset": bool(args.allow_subset),
        "git": RUNNER.git_provenance(),
        "script_path": str(Path(__file__).resolve()),
        "script_sha256": RUNNER.sha256_file(Path(__file__).resolve()),
        "native_input_helper_path": str(DEPENDENCY_RUNNER_PATH),
        "native_input_helper_sha256": RUNNER.sha256_file(DEPENDENCY_RUNNER_PATH),
    }
    if args.resume:
        existing_config_path = output_root / "run_config.json"
        existing_status_path = output_root / "run_status.json"
        if not existing_config_path.is_file() or not existing_status_path.is_file():
            raise FileNotFoundError("--resume requires an existing run_config.json and run_status.json")
        existing_config = RUNNER.read_json(existing_config_path)
        if existing_config.get("experiment") != "reviewer_lxey_v5_rank_probability":
            raise ValueError("--resume target is not an LxEy v5 rank/probability run")
        status = RUNNER.read_json(existing_status_path)
        resume_path = output_root / "resume_provenance.json"
        if resume_path.exists():
            raise FileExistsError(f"Refusing to overwrite existing resume record: {resume_path}")
        RUNNER.write_json(
            resume_path,
            {
                "resumed_at": now_utc(),
                "reason": "continue after pre-forward Mistral display-label validation failure",
                "current_script_path": str(Path(__file__).resolve()),
                "current_script_sha256": RUNNER.sha256_file(Path(__file__).resolve()),
                "prior_script_sha256": existing_config.get("script_sha256"),
                "requested_models": list(models),
            },
        )
    else:
        RUNNER.write_json(output_root / "run_config.json", root_config)
        status = {"status": "running", "models": {}}
        RUNNER.write_json(output_root / "run_status.json", status)
    status["status"] = "running"
    RUNNER.write_json(output_root / "run_status.json", status)
    results: dict[str, dict[str, Any]] = {}
    for model_key in models:
        model_root = output_root / model_key
        completion_path = model_root / "completion.json"
        if args.resume and completion_path.is_file():
            completion = RUNNER.read_json(completion_path)
            summary_path = model_root / "summary.json"
            if completion.get("status") != "complete" or not summary_path.is_file():
                raise RuntimeError(f"{model_key}: resume found an invalid completion record")
            results[model_key] = RUNNER.read_json(summary_path)
            status["models"][model_key] = {"status": "complete", "n_heldout": results[model_key]["metrics"]["n"]}
            RUNNER.write_json(output_root / "run_status.json", status)
            print(f"[{model_key}] already complete; preserving existing result", flush=True)
            continue
        print(f"[{model_key}] starting v5 held-out rank/probability replay", flush=True)
        try:
            result = run_model(args, output_root=output_root, model_key=model_key, dataset_root=dataset_root)
        except Exception as exc:
            status["models"][model_key] = {
                "status": "failed",
                "error": repr(exc),
                "traceback": traceback.format_exc(),
            }
            status["status"] = "failed"
            RUNNER.write_json(output_root / "run_status.json", status)
            raise
        results[model_key] = result
        status["models"][model_key] = {"status": "complete", "n_heldout": result["metrics"]["n"]}
        RUNNER.write_json(output_root / "run_status.json", status)
        print(f"[{model_key}] complete", flush=True)
    table = render_table(results, models)
    (output_root / "rebuttal_table.md").write_text(table, encoding="utf-8")
    table_rows = []
    for model_key in models:
        metrics = results[model_key]["metrics"]
        table_rows.append(
            {
                "model_key": model_key,
                "model_label": TABLE_LABELS[model_key],
                "n": metrics["n"],
                "top1_count": metrics["top1_count"],
                "top1_rate": metrics["top1_rate"],
                "top3_count": metrics["top3_count"],
                "top3_rate": metrics["top3_rate"],
                "top10_count": metrics["top10_count"],
                "top10_rate": metrics["top10_rate"],
                "median_probability": metrics["median_probability"],
                "mean_probability": metrics["mean_probability"],
            }
        )
    RUNNER.write_csv(output_root / "table_metrics.csv", table_rows)
    RUNNER.write_json(output_root / "table_metrics.json", {key: value["metrics"] for key, value in results.items()})
    status["status"] = "complete"
    status["completed_at"] = now_utc()
    RUNNER.write_json(output_root / "run_status.json", status)
    RUNNER.write_json(
        output_root / "completion.json",
        {
            "status": "complete",
            "completed_at": now_utc(),
            "models": list(models),
            "heldout_pairs_per_model": int(args.heldout_pairs),
            "rebuttal_table": str(output_root / "rebuttal_table.md"),
        },
    )
    print(table, flush=True)
    print(json.dumps({"status": "complete", "output_root": str(output_root)}), flush=True)


if __name__ == "__main__":
    main()
