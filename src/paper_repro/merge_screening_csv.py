#!/usr/bin/env python3
"""Merge an additional model's screen into an existing candidate CSV.

The historical v1 CSV already contains Qwen3-1.7B/4B/8B results.  This helper
adds a newly run model (for example Qwen3-14B) without re-running the older
models.  It performs a strict key and scaffold check before writing, then
recomputes ``all_models_valid`` across every model column.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable

from .dataset import write_json


KEY_FIELDS = ("sample_id", "filename", "pool", "candidate")
MODEL_SUFFIX = "_is_tool_call_top1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-csv", type=Path, required=True, help="Existing multi-model screening CSV")
    parser.add_argument(
        "--additional-csv",
        action="append",
        type=Path,
        required=True,
        help="One newly screened model CSV; repeat to add more models",
    )
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--metadata-output", type=Path, default=None)
    return parser.parse_args()


def read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.resolve().open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fields = list(reader.fieldnames or [])
    if not rows:
        raise ValueError(f"Screening CSV contains no rows: {path}")
    missing = [field for field in KEY_FIELDS if field not in fields]
    if missing:
        raise ValueError(f"{path} is missing key fields: {missing}")
    return rows, fields


def key(row: dict[str, str]) -> tuple[str, str, str, str]:
    return tuple(row[field] for field in KEY_FIELDS)  # type: ignore[return-value]


def model_names(fields: Iterable[str]) -> list[str]:
    return sorted(
        field[: -len(MODEL_SUFFIX)]
        for field in fields
        if field.endswith(MODEL_SUFFIX)
    )


def main() -> None:
    args = parse_args()
    base_rows, base_fields = read_csv(args.base_csv)
    base_rows_before_intersection = len(base_rows)
    base_by_key = {key(row): row for row in base_rows}
    if len(base_by_key) != len(base_rows):
        raise ValueError("Base CSV contains duplicate screening keys")

    base_models = model_names(base_fields)
    additions: list[
        tuple[Path, dict[tuple[str, str, str, str], dict[str, str]], list[str], list[str]]
    ] = []
    seen_models = set(base_models)
    merged_fields = list(base_fields)
    if "all_models_valid" in merged_fields:
        merged_fields.remove("all_models_valid")

    for path in args.additional_csv:
        rows, fields = read_csv(path)
        by_key = {key(row): row for row in rows}
        if len(by_key) != len(rows):
            raise ValueError(f"{path} contains duplicate screening keys")
        extra = sorted(set(by_key) - set(base_by_key))[:3]
        if extra:
            raise ValueError(f"Screening key mismatch for {path}: additional rows absent from base={extra}")
        # The historical v1 candidate CSV contains 11 stale source IDs that
        # are not present in the archived base-prompt directory.  A fresh
        # model screen covers the real 1,711-row source pool, so intersect the
        # keys and record the dropped legacy rows rather than silently pairing
        # a model result with a missing scaffold.
        common_keys = set(base_by_key).intersection(by_key)
        base_rows = [row for row in base_rows if key(row) in common_keys]
        base_by_key = {key(row): row for row in base_rows}
        added_models = model_names(fields)
        if not added_models:
            raise ValueError(f"No *_is_tool_call_top1 model columns found in {path}")
        overlap = sorted(seen_models.intersection(added_models))
        if overlap:
            raise ValueError(f"Model columns already present: {overlap}")
        for field in KEY_FIELDS + ("language", "dataset_name", "rendered_first_user_line"):
            if field not in fields or field not in base_fields:
                continue
            for row in base_rows:
                if row[field] != by_key[key(row)][field]:
                    raise ValueError(f"Scaffold metadata mismatch for {field} in {path} at {key(row)}")
        for field in fields:
            if field in KEY_FIELDS or field in {"language", "dataset_name", "rendered_first_user_line", "all_models_valid"}:
                continue
            if field in merged_fields:
                raise ValueError(f"Duplicate non-key field {field!r} in {path}")
            merged_fields.append(field)
        additions.append((path, by_key, fields, added_models))
        seen_models.update(added_models)

    merged_fields.append("all_models_valid")
    output_rows: list[dict[str, str]] = []
    for base_row in base_rows:
        merged = dict(base_row)
        merged.pop("all_models_valid", None)
        for _path, by_key, fields, _models in additions:
            extra_row = by_key[key(base_row)]
            merged.update(
                {
                    field: extra_row[field]
                    for field in fields
                    if field not in KEY_FIELDS and field != "all_models_valid"
                }
            )
        checks: list[bool] = []
        expected = base_row["pool"].strip().lower() == "clean"
        for model in sorted(seen_models):
            field = f"{model}_is_tool_call_top1"
            value = merged.get(field, "").strip().lower() == "true"
            checks.append(value == expected)
        merged["all_models_valid"] = str(all(checks))
        output_rows.append(merged)

    args.output_csv.resolve().parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.resolve().open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=merged_fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(output_rows)

    metadata_path = args.metadata_output or args.output_csv.with_suffix(args.output_csv.suffix + ".metadata.json")
    write_json(
        metadata_path.resolve(),
        {
            "base_csv": str(args.base_csv.resolve()),
            "additional_csvs": [str(path.resolve()) for path in args.additional_csv],
            "output_csv": str(args.output_csv.resolve()),
            "models": sorted(seen_models),
            "n_base_rows_before_intersection": base_rows_before_intersection,
            "n_rows": len(output_rows),
            "n_base_rows_dropped_for_key_intersection": base_rows_before_intersection - len(output_rows),
            "n_all_models_valid": sum(row["all_models_valid"] == "True" for row in output_rows),
        },
    )
    print(
        f"Merged {len(output_rows)} rows; models={','.join(sorted(seen_models))}; "
        f"all_models_valid={sum(row['all_models_valid'] == 'True' for row in output_rows)}"
    )


if __name__ == "__main__":
    main()
