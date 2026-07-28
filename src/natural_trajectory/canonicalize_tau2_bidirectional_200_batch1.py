#!/usr/bin/env python3
"""Revalidate and reselect τ² 200+200 collections with batch-size-one baselines.

The source collections are built from target-model native baseline decisions.
For long left-padded histories, a small number of near-tie logits can depend on
the surrounding batch shape.  This utility makes the intervention population
canonical: every candidate is re-rendered with the same native template and
evaluated one trajectory at a time, then the original balanced, task-disjoint
selection rule is applied again.  It never estimates a vector, selects a
layer, uses a source reward, or looks at intervention outcomes.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import torch


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import build_tau2_bidirectional_200 as collection  # noqa: E402


SEED = 20260726
TELECOM_SOURCE_ROOT = (
    PROJECT_ROOT
    / "datasets"
    / "external"
    / "tau2_telecom_qwen35_9b"
    / "collections"
    / "bidirectional_200_per_model_20260726"
)
RETAIL_SOURCE_ROOT = (
    PROJECT_ROOT
    / "datasets"
    / "external"
    / "tau2_retail_qwen35_9b"
    / "collections"
    / "bidirectional_200_per_model_20260726"
)
TELECOM_OUTPUT_ROOT = TELECOM_SOURCE_ROOT.with_name(TELECOM_SOURCE_ROOT.name + "_canonical_batch1")
RETAIL_OUTPUT_ROOT = RETAIL_SOURCE_ROOT.with_name(RETAIL_SOURCE_ROOT.name + "_canonical_batch1")
TELECOM_PREPARED_ROOT = PROJECT_ROOT / "datasets" / "external" / "tau2_telecom_qwen35_9b" / "prepared"
RETAIL_PREPARED_ROOT = PROJECT_ROOT / "datasets" / "external" / "tau2_retail_qwen35_9b" / "prepared"
TELECOM_RAW_ROOT = PROJECT_ROOT / "datasets" / "external" / "tau2_telecom_qwen35_9b" / "raw"
RETAIL_RAW_ROOT = PROJECT_ROOT / "datasets" / "external" / "tau2_retail_qwen35_9b" / "raw"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", default="all", help="Comma-separated model keys or 'all'.")
    parser.add_argument("--max-context-tokens", type=int, default=30_000)
    parser.add_argument("--render-chunk-size", type=int, default=25)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_models(raw: str) -> list[collection.TargetSpec]:
    if raw.strip() == "all":
        return [collection.SPECS[key] for key in collection.SPECS]
    keys = [value.strip() for value in raw.split(",") if value.strip()]
    unknown = [key for key in keys if key not in collection.SPECS]
    if unknown:
        raise ValueError(f"Unknown model keys: {unknown}; choices are {sorted(collection.SPECS)}")
    return [collection.SPECS[key] for key in keys]


def roots_for(spec: collection.TargetSpec) -> tuple[Path, Path, Path, Path, str]:
    if spec.key.startswith("qwen"):
        return (
            TELECOM_SOURCE_ROOT,
            TELECOM_OUTPUT_ROOT,
            TELECOM_PREPARED_ROOT,
            TELECOM_RAW_ROOT,
            "telecom",
        )
    return RETAIL_SOURCE_ROOT, RETAIL_OUTPUT_ROOT, RETAIL_PREPARED_ROOT, RETAIL_RAW_ROOT, "retail"


def old_screen_rows(model_root: Path, arm: str) -> list[dict[str, Any]]:
    """Use exactly the candidate pools that were eligible for the published 200 set."""

    merged: dict[str, dict[str, Any]] = {}
    for tier in ("primary", "closure", "broad"):
        path = collection.screen_file(model_root, arm, tier)
        if not path.exists():
            continue
        for row in collection.read_jsonl(path):
            candidate_id = str(row["candidate_id"])
            if candidate_id in merged:
                raise RuntimeError(f"Duplicate source-screen candidate {candidate_id}")
            merged[candidate_id] = row
    if not merged:
        raise RuntimeError(f"No original {arm} screen records in {model_root}")
    return list(merged.values())


def candidates_from_original_screen(
    *,
    old_model_root: Path,
    prepared_root: Path,
    arm: str,
) -> list[dict[str, Any]]:
    source_map = collection.raw_source_map(prepared_root, arm)
    screen_rows = old_screen_rows(old_model_root, arm)
    rows: list[dict[str, Any]] = []
    for screen in screen_rows:
        candidate_id = str(screen["candidate_id"])
        source = source_map.get(candidate_id)
        if source is None:
            raise KeyError(f"Original {arm} candidate missing from source map: {candidate_id}")
        rows.append(source)
    if len({str(row["candidate_id"]) for row in rows}) != len(rows):
        raise RuntimeError(f"Duplicate raw source candidate in {arm}")
    return rows


def load_cached(path: Path, expected_ids: set[str]) -> list[dict[str, Any]] | None:
    if not path.exists():
        return None
    rows = collection.read_jsonl(path)
    if len(rows) != len(expected_ids) or {str(row.get("candidate_id")) for row in rows} != expected_ids:
        return None
    return rows


def screen_with_checkpoints(
    *,
    rows: Sequence[dict[str, Any]],
    path: Path,
    model: Any,
    tokenizer: Any,
    device: torch.device,
    spec: collection.TargetSpec,
    resources: collection.TauResources,
    tool_token_id: int,
    max_context_tokens: int,
    chunk_size: int,
    resume: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Persist every small batch-one screen chunk so GPU interruptions are cheap."""

    expected_ids = {str(row["candidate_id"]) for row in rows}
    existing = collection.read_jsonl(path) if resume and path.exists() else []
    existing_ids = {str(row["candidate_id"]) for row in existing}
    if len(existing_ids) != len(existing) or not existing_ids <= expected_ids:
        raise RuntimeError(f"Invalid partial canonical checkpoint: {path}")
    pending = [row for row in rows if str(row["candidate_id"]) not in existing_ids]
    rejected_all: list[dict[str, Any]] = []
    combined = list(existing)
    for start in range(0, len(pending), chunk_size):
        chunk = pending[start : start + chunk_size]
        result_rows, rejected = collection.screen_rows(
            chunk,
            model=model,
            tokenizer=tokenizer,
            device=device,
            spec=spec,
            resources=resources,
            tool_token_id=tool_token_id,
            max_context_tokens=max_context_tokens,
            chunk_size=chunk_size,
            source_tier="canonical_batch1",
        )
        if rejected or len(result_rows) != len(chunk):
            rejected_all.extend(rejected)
            raise RuntimeError(f"Canonical screen rejected {len(rejected)} rows in {path}")
        combined.extend(result_rows)
        if len({str(row["candidate_id"]) for row in combined}) != len(combined):
            raise RuntimeError(f"Duplicate canonical result while writing {path}")
        collection.write_jsonl(path, combined)
        print(
            json.dumps(
                {"canonical_checkpoint": path.name, "screened": len(combined), "total": len(rows)},
                ensure_ascii=False,
            ),
            flush=True,
        )
    if len(combined) != len(rows):
        raise RuntimeError(f"Canonical screen incomplete for {path}: {len(combined)}/{len(rows)}")
    return combined, rejected_all


def select_fresh_sets(
    *,
    tool_screen: Sequence[dict[str, Any]],
    direct_screen: Sequence[dict[str, Any]],
    spec: collection.TargetSpec,
    final_count: int,
    seed: int,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    legacy_ids, legacy_counts = collection.load_legacy_task_ids(spec)
    tool_eligible = [row for row in tool_screen if bool(row["baseline_is_tool_call_top1"])]
    direct_eligible = [
        row
        for row in direct_screen
        if not bool(row["baseline_is_tool_call_top1"])
        and float(row["baseline_tool_probability"]) <= 0.05
    ]
    available = {
        "tool": collection.unique_task_count([row for row in tool_eligible if str(row["task_id"]) not in legacy_ids]),
        "direct": collection.unique_task_count([row for row in direct_eligible if str(row["task_id"]) not in legacy_ids]),
    }
    order = ("tool", "direct") if available["tool"] <= available["direct"] else ("direct", "tool")
    eligible = {"tool": tool_eligible, "direct": direct_eligible}
    selected: dict[str, list[dict[str, Any]]] = {}
    forbidden = set(legacy_ids)
    for arm in order:
        chosen = collection.choose_balanced(
            eligible[arm],
            arm=arm,
            count=final_count,
            forbidden_task_ids=forbidden,
            seed=seed,
            model_key=spec.key,
        )
        selected[arm] = chosen
        forbidden.update(str(row["task_id"]) for row in chosen)
    if any(len(selected.get(arm, [])) != final_count for arm in ("tool", "direct")):
        raise RuntimeError(
            f"Batch-1 candidate pool cannot form {final_count}+{final_count}: "
            f"tool={len(selected.get('tool', []))}, direct={len(selected.get('direct', []))}, available={available}"
        )
    tool_ids = {str(row["task_id"]) for row in selected["tool"]}
    direct_ids = {str(row["task_id"]) for row in selected["direct"]}
    if tool_ids & direct_ids or (tool_ids | direct_ids) & legacy_ids:
        raise RuntimeError("Canonical selection violates task disjointness")
    summary = {
        "target_per_arm": final_count,
        "max_non_tool_probability": 0.05,
        "legacy_50_task_ids_excluded": len(legacy_ids),
        "legacy_selection_task_counts": legacy_counts,
        "screen_records": {"tool": len(tool_screen), "direct": len(direct_screen)},
        "eligible_unique_tasks_after_legacy_exclusion": available,
        "selection_priority": list(order),
        "selected": {arm: len(selected[arm]) for arm in ("tool", "direct")},
        "task_overlap_between_arms": 0,
    }
    return selected, summary


def run_model(args: argparse.Namespace, spec: collection.TargetSpec) -> dict[str, Any]:
    source_root, output_root, prepared_root, raw_root, source_domain = roots_for(spec)
    old_model_root = source_root / spec.key
    model_root = output_root / spec.key
    old_summary_path = old_model_root / "selection_summary.json"
    if not old_summary_path.exists() or not bool(json.loads(old_summary_path.read_text()).get("complete")):
        raise RuntimeError(f"Original selection is not complete: {old_summary_path}")
    raw_rows = {
        "tool": candidates_from_original_screen(old_model_root=old_model_root, prepared_root=prepared_root, arm="tool"),
        "direct": candidates_from_original_screen(old_model_root=old_model_root, prepared_root=prepared_root, arm="direct"),
    }
    model_root.mkdir(parents=True, exist_ok=True)
    screen_paths = {
        arm: model_root / f"{arm}_baseline_screen_canonical_batch1.jsonl" for arm in ("tool", "direct")
    }
    resources = collection.read_resources(raw_root)
    model = None
    tokenizer = None
    try:
        model, tokenizer, device = collection.load_model_and_tokenizer(spec)
        tool_token_id, token_audit = collection.resolve_tool_token(tokenizer, spec)
        batch1_spec = dataclasses.replace(spec, batch_size=1, max_batch_tokens=int(args.max_context_tokens))
        fresh_screen: dict[str, list[dict[str, Any]]] = {}
        rejections: dict[str, list[dict[str, Any]]] = {}
        for arm in ("tool", "direct"):
            expected_ids = {str(row["candidate_id"]) for row in raw_rows[arm]}
            cached = load_cached(screen_paths[arm], expected_ids) if args.resume else None
            if cached is not None:
                print(json.dumps({"model": spec.key, "arm": arm, "canonical_batch1": "cached", "n": len(cached)}), flush=True)
                fresh_screen[arm] = cached
                rejections[arm] = []
                continue
            if screen_paths[arm].exists() and not args.resume:
                raise FileExistsError(
                    f"Canonical screen already exists at {screen_paths[arm]}; pass --resume to continue/use it"
                )
            print(json.dumps({"model": spec.key, "arm": arm, "canonical_batch1": "starting", "n": len(raw_rows[arm])}), flush=True)
            rows, rejected = screen_with_checkpoints(
                rows=raw_rows[arm],
                path=screen_paths[arm],
                model=model,
                tokenizer=tokenizer,
                device=device,
                spec=batch1_spec,
                resources=resources,
                tool_token_id=tool_token_id,
                max_context_tokens=int(args.max_context_tokens),
                chunk_size=int(args.render_chunk_size),
                resume=bool(args.resume),
            )
            fresh_screen[arm] = rows
            rejections[arm] = rejected
        selected, summary = select_fresh_sets(
            tool_screen=fresh_screen["tool"],
            direct_screen=fresh_screen["direct"],
            spec=spec,
            final_count=200,
            seed=int(args.seed),
        )
        source_maps = {arm: {str(row["candidate_id"]): row for row in raw_rows[arm]} for arm in ("tool", "direct")}
        for arm in ("tool", "direct"):
            output_rows: list[dict[str, Any]] = []
            for screen in selected[arm]:
                source = source_maps[arm][str(screen["candidate_id"])]
                merged = dict(source)
                merged["tau2_200_baseline"] = screen
                output_rows.append(merged)
            collection.write_jsonl(model_root / f"selected_{'tool' if arm == 'tool' else 'direct'}_200.jsonl", output_rows)
        old_selected = {
            arm: {str(row["candidate_id"]) for row in collection.read_jsonl(old_model_root / f"selected_{'tool' if arm == 'tool' else 'direct'}_200.jsonl")}
            for arm in ("tool", "direct")
        }
        summary.update(
            {
                "collection_name": f"tau2_{source_domain}_bidirectional_200_canonical_batch1_20260727",
                "model": spec.display_name,
                "model_key": spec.key,
                "source_domain": source_domain,
                "complete": True,
                "canonical_baseline": {
                    "batch_size": 1,
                    "max_context_tokens": int(args.max_context_tokens),
                    "native_tool_decision": token_audit,
                    "rule": "one trajectory per forward; no batch-shape dependence in final labels",
                    "source_collection": str(old_model_root),
                    "source_collection_summary_sha256": sha256_file(old_summary_path),
                },
                "old_selection_overlap": {
                    arm: len(old_selected[arm] & {str(row["candidate_id"]) for row in selected[arm]})
                    for arm in ("tool", "direct")
                },
                "render_rejections": {arm: len(rejections[arm]) for arm in ("tool", "direct")},
                "files": {
                    "tool": str(model_root / "selected_tool_200.jsonl"),
                    "direct": str(model_root / "selected_direct_200.jsonl"),
                },
            }
        )
        collection.write_json(model_root / "selection_summary.json", summary)
        collection.write_json(
            model_root / "canonicalization_config.json",
            {
                "model": spec.display_name,
                "source_domain": source_domain,
                "old_model_root": str(old_model_root),
                "old_screen_files": {
                    arm: [str(collection.screen_file(old_model_root, arm, tier)) for tier in ("primary", "closure", "broad") if collection.screen_file(old_model_root, arm, tier).exists()]
                    for arm in ("tool", "direct")
                },
                "fresh_screen_files": {arm: str(screen_paths[arm]) for arm in ("tool", "direct")},
                "raw_resources": {
                    "system_prompt_sha256": sha256_file(raw_root / "tau2_system_prompt.txt"),
                    "tool_schemas_sha256": sha256_file(raw_root / "tau2_tool_schemas.json"),
                },
                "batch_size": 1,
                "selection_success_or_reward_used": False,
                "completed_unix": time.time(),
            },
        )
        print(json.dumps({"completed": True, "model": spec.key, "output_root": str(model_root)}, ensure_ascii=False), flush=True)
        return summary
    finally:
        if model is not None:
            del model
        if tokenizer is not None:
            del tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> None:
    args = parse_args()
    if args.max_context_tokens < 1 or args.render_chunk_size < 1:
        raise ValueError("Context and render chunk sizes must be positive")
    results = {spec.key: run_model(args, spec) for spec in parse_models(args.models)}
    collection.write_json(
        TELECOM_OUTPUT_ROOT.parent / "canonical_batch1_run_summary.json",
        {"results": results, "completed_unix": time.time()},
    )


if __name__ == "__main__":
    main()
