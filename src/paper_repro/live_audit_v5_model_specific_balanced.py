#!/usr/bin/env python3
"""Replay every v5 V0 prompt on its target model before releasing the data.

The release builders screen candidates in batches.  A near-tie can, in
principle, change its argmax after a loader or batch-shape change.  This audit
uses the exact V0 rendering and model-loading code used by the CJrQ rerun,
then checks every clean and corrupt prompt both in the canonical experiment
batch size and as a singleton.  A model is accepted only when every replay
has the intended first-token behavior in both layouts.
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "datasets" / "v5_model_specific_balanced"
DEFAULT_RESULTS_ROOT = PROJECT_ROOT / "results" / "rebuild_checks"
RUNNER_PATH = PROJECT_ROOT / "src" / "rebuttal" / "run_cjrq_v5_tool_identity.py"


def load_runner():
    spec = importlib.util.spec_from_file_location("cjrq_v5_runner_for_live_audit", RUNNER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import runner from {RUNNER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


RUNNER = load_runner()
MODEL_KEYS = tuple(RUNNER.MODEL_SPECS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--models", nargs="+", choices=MODEL_KEYS, default=list(MODEL_KEYS))
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument(
        "--output",
        type=Path,
        help="Audit JSON path. Defaults to a timestamped file under results/rebuild_checks/.",
    )
    return parser.parse_args()


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def clear_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def manifest_expectations(records: Sequence[Any], side: str) -> dict[str, dict[str, Any]]:
    field = f"{side}_is_tool_top1"
    margin_field = f"{side}_margin_vs_best_non_tool"
    return {
        str(record.sample_id): {
            "expected_tool_top1": bool(record.manifest[field]),
            "manifest_margin_vs_best_non_tool": float(record.manifest[margin_field]),
            "split": str(record.split),
        }
        for record in records
    }


def audit_layout(
    model: Any,
    tokenizer: Any,
    items: Sequence[Any],
    *,
    expected: dict[str, dict[str, Any]],
    tool_id: int,
    batch_size: int,
    model_key: str,
    side: str,
) -> dict[str, Any]:
    rows = RUNNER.evaluate_condition(
        model,
        items,
        tokenizer=tokenizer,
        layer=None,
        mode=None,
        value_cpu=None,
        tool_id=tool_id,
        batch_size=batch_size,
        condition=f"live_audit_{side}_batch_{batch_size}",
        layer_id=-1,
        progress_label=f"{model_key} live audit {side} batch={batch_size}",
    )
    observed = {str(row["sample_id"]): row for row in rows}
    if set(observed) != set(expected):
        raise AssertionError(f"{model_key}/{side}/batch={batch_size}: sample IDs do not match manifest")
    mismatches: list[dict[str, Any]] = []
    for sample_id in sorted(expected):
        row = observed[sample_id]
        expectation = expected[sample_id]
        actual = bool(row["tool_call_top1"])
        if actual == expectation["expected_tool_top1"]:
            continue
        mismatches.append(
            {
                "sample_id": sample_id,
                "split": expectation["split"],
                "expected_tool_top1": expectation["expected_tool_top1"],
                "observed_tool_top1": actual,
                "manifest_margin_vs_best_non_tool": expectation["manifest_margin_vs_best_non_tool"],
                "observed_tool_logit": float(row["tool_call_logit"]),
                "observed_tool_rank": int(row["tool_call_rank"]),
                "observed_top1_token_id": int(row["top1_token_id"]),
            }
        )
    return {
        "batch_size": batch_size,
        "n": len(rows),
        "tool_top1_count": sum(bool(row["tool_call_top1"]) for row in rows),
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
        "status": "pass" if not mismatches else "fail",
    }


def audit_model(dataset_root: Path, model_key: str, *, dtype: str) -> dict[str, Any]:
    model_spec = RUNNER.MODEL_SPECS[model_key]
    model_dataset_root = dataset_root / model_key
    summary, records = RUNNER.load_pairs(
        model_dataset_root,
        model_key=model_key,
        family=str(model_spec["family"]),
    )
    loader_args = argparse.Namespace(model_path=None, dtype=dtype, attn_implementation="")
    tokenizer = None
    model = None
    try:
        torch.manual_seed(20260728)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(20260728)
        tokenizer, model, loaded_model_path = RUNNER.load_tokenizer_and_model(model_spec, loader_args)
        tool_id, tool_info = RUNNER.tool_token_id(
            tokenizer,
            marker=str(summary["tool_call_marker"]),
            family=str(model_spec["family"]),
        )
        if int(tool_id) != int(summary["tool_call_token_id"]):
            raise RuntimeError(f"{model_key}: tool marker token ID differs from manifest")
        items, _provenance = RUNNER.build_variant_items(
            records,
            variant="V0",
            family=str(model_spec["family"]),
            tokenizer=tokenizer,
            catalog={},
        )
        canonical_batch_size = int(model_spec["batch_size"])
        layouts = (canonical_batch_size, 1)
        sides: dict[str, Any] = {}
        for side in ("clean", "corrupt"):
            expected = manifest_expectations(records, side)
            sides[side] = [
                audit_layout(
                    model,
                    tokenizer,
                    items[side],
                    expected=expected,
                    tool_id=tool_id,
                    batch_size=batch_size,
                    model_key=model_key,
                    side=side,
                )
                for batch_size in layouts
            ]
        mismatches = sum(layout["mismatch_count"] for values in sides.values() for layout in values)
        return {
            "model_key": model_key,
            "model_label": str(model_spec["label"]),
            "model_path": str(loaded_model_path),
            "dataset_root": str(model_dataset_root),
            "tool_call": tool_info,
            "n_pairs": len(records),
            "layouts": {"canonical_batch_size": canonical_batch_size, "singleton_batch_size": 1},
            "sides": sides,
            "mismatch_count": mismatches,
            "status": "pass" if mismatches == 0 else "fail",
        }
    finally:
        del model
        clear_cuda()


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    output = args.output.resolve() if args.output else DEFAULT_RESULTS_ROOT / f"v5_live_audit_{utc_stamp()}.json"
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing audit: {output}")
    report: dict[str, Any] = {
        "audit": "v5_live_behavior_replay",
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "dataset_root": str(dataset_root),
        "dtype": args.dtype,
        "models": {},
    }
    for model_key in args.models:
        print(f"[{model_key}] loading and replaying all 500 pairs", flush=True)
        result = audit_model(dataset_root, model_key, dtype=str(args.dtype))
        report["models"][model_key] = result
        write_json(output, report)
        print(f"[{model_key}] {result['status']}: {result['mismatch_count']} mismatches", flush=True)
    failed = [key for key, value in report["models"].items() if value["status"] != "pass"]
    report["status"] = "pass" if not failed else "fail"
    report["failed_models"] = failed
    write_json(output, report)
    print(json.dumps({"status": report["status"], "output": str(output), "failed_models": failed}), flush=True)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
