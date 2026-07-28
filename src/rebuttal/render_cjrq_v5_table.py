#!/usr/bin/env python3
"""Audit seven completed CJrQ v5 model runs and render the rebuttal table."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[2]
V5_ROOT = PROJECT_ROOT / "datasets" / "v5_model_specific_balanced"

MODELS = (
    ("qwen3_4b", "Qwen3-4B"),
    ("qwen3_8b", "Qwen3-8B"),
    ("qwen3_14b", "Qwen3-14B"),
    ("qwen35_4b", "Qwen3.5-4B"),
    ("qwen35_9b", "Qwen3.5-9B"),
    ("mistral_3p2_24b", "Mistral"),
    ("granite_3p3_8b", "Granite"),
)
VARIANTS = (("V1", "Renamed"), ("V2", "Removed"), ("V5", "Mismatched"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, type=Path)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path}: expected JSON object")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def as_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes"}


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def manifest_membership(model_key: str) -> tuple[list[str], list[str], str]:
    root = V5_ROOT / model_key
    manifest = root / "manifest.jsonl"
    rows = read_jsonl(manifest)
    train = [str(row["sample_id"]) for row in rows if str(row["split"]) == "train"]
    heldout = [str(row["sample_id"]) for row in rows if str(row["split"]) == "heldout"]
    if len(train) != 200 or len(set(train)) != 200 or len(heldout) != 300 or len(set(heldout)) != 300:
        raise ValueError(f"{model_key}: current v5 manifest does not contain the required 200/300 unique split IDs")
    return train, heldout, sha256_file(manifest)


def audit_model(run_root: Path, model_key: str, label: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model_root = run_root / model_key
    required = (
        "completion.json",
        "run_config.json",
        "dataset_provenance.json",
        "membership.json",
        "input_provenance.jsonl",
        "sample_metrics.csv",
        "table_metrics.json",
        "baseline_screening_replay.json",
    )
    missing = [name for name in required if not (model_root / name).is_file()]
    if missing:
        raise FileNotFoundError(f"{model_key}: incomplete run; missing {missing}")
    completion = read_json(model_root / "completion.json")
    config = read_json(model_root / "run_config.json")
    provenance = read_json(model_root / "dataset_provenance.json")
    membership = read_json(model_root / "membership.json")
    table = read_json(model_root / "table_metrics.json")
    replay = read_json(model_root / "baseline_screening_replay.json")
    if completion.get("status") != "complete":
        raise ValueError(f"{model_key}: completion status is not complete")
    if config.get("experiment") != "reviewer_cjrq_v5_tool_identity":
        raise ValueError(f"{model_key}: wrong experiment type")
    if provenance.get("dataset_version") != "v5_model_specific_balanced":
        raise ValueError(f"{model_key}: not a v5 provenance record")
    if int(completion.get("n_layer_selection_train", -1)) != 64:
        raise ValueError(f"{model_key}: expected 64 selection pairs")
    if int(completion.get("n_vector_fit_train", -1)) != 136:
        raise ValueError(f"{model_key}: expected 136 vector-fit pairs")
    if int(completion.get("n_heldout", -1)) != 300:
        raise ValueError(f"{model_key}: expected all 300 heldout pairs")
    train_ids, heldout_ids, current_manifest_sha = manifest_membership(model_key)
    if membership.get("heldout_sample_ids") != heldout_ids:
        raise ValueError(f"{model_key}: result heldout membership differs from current v5 manifest")
    if membership.get("layer_selection_train_sample_ids") != train_ids[:64]:
        raise ValueError(f"{model_key}: layer-selection membership differs from v5 train prefix")
    if membership.get("vector_fit_train_sample_ids") != train_ids[64:200]:
        raise ValueError(f"{model_key}: vector-fit membership differs from remaining v5 train split")
    partitions = (
        membership.get("layer_selection_train_sample_ids", []),
        membership.get("vector_fit_train_sample_ids", []),
        membership.get("heldout_sample_ids", []),
    )
    if [len(values) for values in partitions] != [64, 136, 300]:
        raise ValueError(f"{model_key}: invalid partition cardinalities")
    all_ids = [sample_id for values in partitions for sample_id in values]
    if len(all_ids) != len(set(all_ids)):
        raise ValueError(f"{model_key}: partitions overlap")
    recorded_manifest_sha = provenance["release_files"]["manifest"]["sha256"]
    if recorded_manifest_sha != current_manifest_sha:
        raise ValueError(f"{model_key}: v5 manifest hash changed after model run")
    for side in ("clean", "corrupt"):
        side_audit = replay.get(side, {})
        mismatches = side_audit.get("mismatches", [])
        if int(side_audit.get("mismatch_count", -1)) != len(mismatches):
            raise ValueError(f"{model_key}/{side}: baseline replay mismatch count is inconsistent")
        if mismatches:
            raise ValueError(f"{model_key}/{side}: V0 baseline replay has {len(mismatches)} mismatches")
        if side_audit.get("status") != "exact_match":
            raise ValueError(f"{model_key}/{side}: V0 baseline replay is not an exact match")

    input_rows = read_jsonl(model_root / "input_provenance.jsonl")
    if len(input_rows) != 4000:
        raise ValueError(f"{model_key}: expected 4000 rendered input provenance rows, found {len(input_rows)}")
    mistral_native_input_rule_verified = False
    if model_key == "mistral_3p2_24b":
        v0 = [row for row in input_rows if row["variant"] == "V0"]
        edited = [row for row in input_rows if row["variant"] != "V0"]
        if not all(row["render_method"] == "stored_native_input_ids" and row["v0_template_roundtrip"] is True for row in v0):
            raise ValueError("Mistral V0 did not use and validate stored native input_ids")
        if not all(row["render_method"].startswith("apply_chat_template(tokenize=True") for row in edited):
            raise ValueError("Mistral edited schemas were not rendered natively")
        mistral_native_input_rule_verified = True

    with (model_root / "sample_metrics.csv").open(encoding="utf-8", newline="") as handle:
        metrics = list(csv.DictReader(handle))
    if len(metrics) != 4800:
        raise ValueError(f"{model_key}: expected 4800 heldout sample metrics rows, found {len(metrics)}")
    long_rows: list[dict[str, Any]] = []
    for variant, manipulation in VARIANTS:
        subset = [row for row in metrics if row["variant"] == variant]
        by_condition = {
            condition: [row for row in subset if row["condition"] == condition]
            for condition in (
                "baseline_clean",
                "baseline_corrupt",
                "frozen_v0_add_to_corrupt",
                "frozen_v0_subtract_from_clean",
            )
        }
        if any(len(rows) != 300 for rows in by_condition.values()):
            raise ValueError(f"{model_key}/{variant}: incomplete heldout condition rows")
        clean = {row["sample_id"]: as_bool(row["tool_call_top1"]) for row in by_condition["baseline_clean"]}
        corrupt = {row["sample_id"]: as_bool(row["tool_call_top1"]) for row in by_condition["baseline_corrupt"]}
        added = {row["sample_id"]: as_bool(row["tool_call_top1"]) for row in by_condition["frozen_v0_add_to_corrupt"]}
        removed = {row["sample_id"]: as_bool(row["tool_call_top1"]) for row in by_condition["frozen_v0_subtract_from_clean"]}
        if any(set(values) != set(heldout_ids) for values in (clean, corrupt, added, removed)):
            raise ValueError(f"{model_key}/{variant}: condition membership is not the full heldout split")
        flip_denominator = sum(not corrupt[sample_id] for sample_id in heldout_ids)
        drop_denominator = sum(clean[sample_id] for sample_id in heldout_ids)
        flip_count = sum(not corrupt[sample_id] and added[sample_id] for sample_id in heldout_ids)
        drop_count = sum(clean[sample_id] and not removed[sample_id] for sample_id in heldout_ids)
        cell = table.get(variant)
        if not isinstance(cell, dict):
            raise ValueError(f"{model_key}: missing {variant} table metrics")
        expected_numbers = {
            "strict_flip_count": flip_count,
            "strict_flip_denominator": flip_denominator,
            "strict_drop_count": drop_count,
            "strict_drop_denominator": drop_denominator,
        }
        for field, value in expected_numbers.items():
            if int(cell.get(field, -1)) != value:
                raise ValueError(f"{model_key}/{variant}: table {field} does not match per-sample audit")
        for rate_field, numerator, denominator in (
            ("strict_flip", flip_count, flip_denominator),
            ("strict_drop", drop_count, drop_denominator),
        ):
            reported = cell.get(rate_field)
            if denominator == 0:
                if reported is not None:
                    raise ValueError(f"{model_key}/{variant}: {rate_field} has a rate despite a zero denominator")
            elif reported is None or abs(float(reported) - numerator / denominator) > 1e-12:
                raise ValueError(f"{model_key}/{variant}: {rate_field} does not match its audited numerator/denominator")
        long_rows.append(
            {
                "model_key": model_key,
                "model_label": label,
                "variant": variant,
                "manipulation": manipulation,
                "cosine": float(cell["cosine"]),
                "strict_flip": None if cell["strict_flip"] is None else float(cell["strict_flip"]),
                "strict_flip_count": flip_count,
                "strict_flip_denominator": flip_denominator,
                "strict_drop": None if cell["strict_drop"] is None else float(cell["strict_drop"]),
                "strict_drop_count": drop_count,
                "strict_drop_denominator": drop_denominator,
            }
        )
    audit = {
        "model_key": model_key,
        "model_label": label,
        "model_root": str(model_root),
        "heldout_membership_verified": True,
        "heldout_count": len(heldout_ids),
        "manifest_sha256_verified": current_manifest_sha,
        "input_provenance_count": len(input_rows),
        "sample_metric_count": len(metrics),
        "baseline_replay_verified": True,
        "mistral_native_input_rule_verified": mistral_native_input_rule_verified if model_key == "mistral_3p2_24b" else None,
    }
    return audit, long_rows


def format_rate(value: float | None) -> str:
    return "NA" if value is None else f"{value:.3f}"


def format_cell(row: dict[str, Any]) -> str:
    return f"{row['cosine']:.3f}/{format_rate(row['strict_flip'])}/{format_rate(row['strict_drop'])}"


def main() -> None:
    args = parse_args()
    run_root = args.run_root.resolve()
    if not run_root.is_dir():
        raise FileNotFoundError(run_root)
    outputs = ("rebuttal_table.md", "table_metrics.csv", "table_audit.json", "completion.json")
    existing = [name for name in outputs if (run_root / name).exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite table outputs: {existing}")
    audits: list[dict[str, Any]] = []
    long_rows: list[dict[str, Any]] = []
    for model_key, label in MODELS:
        audit, rows = audit_model(run_root, model_key, label)
        audits.append(audit)
        long_rows.extend(rows)
    by_key = {(row["manipulation"], row["model_key"]): row for row in long_rows}
    header = "| Manipulation | " + " | ".join(label for _key, label in MODELS) + " |"
    divider = "|---|" + "|".join("---" for _key, _label in MODELS) + "|"
    table_rows = [header, divider]
    for _variant, manipulation in VARIANTS:
        cells = [format_cell(by_key[(manipulation, model_key)]) for model_key, _label in MODELS]
        table_rows.append("| " + manipulation + " | " + " | ".join(cells) + " |")
    table_rows.extend(
        (
            "",
            "Each cell is cosine / strict flip / strict drop. Directions are fit on 136 disjoint training pairs after a 64-pair train-only layer selection; every metric uses all 300 v5 held-out pairs.",
            "Strict flip is conditioned on each variant's baseline-corrupt non-tool cases; strict drop is conditioned on its baseline-clean tool cases. `NA` means that condition had a zero denominator; exact counts and denominators are in `table_metrics.csv`.",
        )
    )
    (run_root / "rebuttal_table.md").write_text("\n".join(table_rows) + "\n", encoding="utf-8")
    write_csv(run_root / "table_metrics.csv", long_rows)
    write_json(
        run_root / "table_audit.json",
        {
            "status": "complete",
            "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "run_root": str(run_root),
            "model_audits": audits,
            "checked_models": [key for key, _label in MODELS],
            "checked_variants": [variant for variant, _label in VARIANTS],
            "table_source": "per-sample heldout metrics, independently recomputed strict numerators and denominators",
        },
    )
    write_json(
        run_root / "completion.json",
        {
            "status": "complete",
            "experiment": "reviewer_cjrq_v5_tool_identity_all_models",
            "completed_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "models": [key for key, _label in MODELS],
            "table_path": str(run_root / "rebuttal_table.md"),
            "metrics_path": str(run_root / "table_metrics.csv"),
            "audit_path": str(run_root / "table_audit.json"),
        },
    )
    print("\n".join(table_rows), flush=True)


if __name__ == "__main__":
    main()
