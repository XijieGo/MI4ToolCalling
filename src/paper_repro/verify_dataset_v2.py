#!/usr/bin/env python3
"""Validate the active v2 dataset against its frozen provenance definition."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from .dataset import iter_jsonl, read_csv, replace_instruction_verb, sha256_file, write_json
from .paths import DATASETS_ROOT, PROVENANCE_ROOT, RESULTS_ROOT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify v2_1500 prompt/data provenance and split integrity.")
    parser.add_argument("--dataset-root", type=Path, default=DATASETS_ROOT)
    parser.add_argument("--provenance-root", type=Path, default=PROVENANCE_ROOT)
    parser.add_argument("--report-path", type=Path, default=RESULTS_ROOT / "rebuild_checks" / "v2_1500_verification.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    provenance_root = args.provenance_root.resolve()
    v1_source = provenance_root / "v1_1711" / "base_pairs"
    v2_root = provenance_root / "v2_1500"
    selection_rows = read_csv(v2_root / "selection_manifest.csv")
    split_rows = read_csv(v2_root / "split_manifest.csv")
    split_by_filename = {row["filename"]: row["split"] for row in split_rows}

    errors: list[str] = []
    expected_files: dict[tuple[str, str], set[str]] = {(split, side): set() for split in ("train", "test") for side in ("clean", "corrupt")}
    expected_content: dict[tuple[str, str, str], str] = {}
    for row in selection_rows:
        filename = row["filename"]
        split = split_by_filename.get(filename)
        if split not in {"train", "test"}:
            errors.append(f"No valid split for {filename}")
            continue
        source = v1_source / "clean" / filename
        if not source.exists():
            errors.append(f"Missing v1 source prompt {source}")
            continue
        scaffold = source.read_text(encoding="utf-8")
        for side in ("clean", "corrupt"):
            expected_files[(split, side)].add(filename)
            expected_content[(split, side, filename)] = replace_instruction_verb(scaffold, row[f"{side}_candidate"])[0]

    observed_counts: dict[str, int] = {}
    manifest_counts: dict[str, int] = {}
    content_mismatches: list[str] = []
    for split in ("train", "test"):
        for side in ("clean", "corrupt"):
            directory = dataset_root / split / side
            actual = {path.name for path in directory.glob("*.txt")}
            expected = expected_files[(split, side)]
            observed_counts[f"{split}_{side}"] = len(actual)
            if actual != expected:
                errors.append(
                    f"{split}/{side} membership mismatch: missing={sorted(expected - actual)[:3]}, extra={sorted(actual - expected)[:3]}"
                )
            for filename in sorted(actual & expected):
                actual_text = (directory / filename).read_text(encoding="utf-8")
                if actual_text != expected_content[(split, side, filename)]:
                    content_mismatches.append(f"{split}/{side}/{filename}")
            manifest_path = directory / "manifest.jsonl"
            if not manifest_path.exists():
                errors.append(f"Missing manifest: {manifest_path}")
            else:
                manifest_names = {
                    str(row.get("output_filename") or row.get("source_filename") or "") for row in iter_jsonl(manifest_path)
                }
                manifest_counts[f"{split}_{side}"] = len(manifest_names)
                if manifest_names != expected:
                    errors.append(f"{split}/{side} manifest membership does not match prompt membership")

    if content_mismatches:
        errors.append(f"Prompt content mismatch in {len(content_mismatches)} file(s)")
    split_counts = Counter(row["split"] for row in split_rows)
    report = {
        "dataset_version": "v2_1500",
        "dataset_root": str(dataset_root),
        "selection_pairs": len(selection_rows),
        "split_counts": dict(sorted(split_counts.items())),
        "observed_prompt_counts": observed_counts,
        "manifest_counts": manifest_counts,
        "content_mismatch_count": len(content_mismatches),
        "content_mismatch_examples": content_mismatches[:20],
        "errors": errors,
        "ok": not errors,
        "active_manifest_sha256": {
            f"{split}_{side}": sha256_file(dataset_root / split / side / "manifest.jsonl")
            for split in ("train", "test")
            for side in ("clean", "corrupt")
        },
    }
    write_json(args.report_path.resolve(), report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

