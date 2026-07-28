#!/usr/bin/env python3
"""Rebuild the frozen 1,500-pair v2 dataset without touching the active copy.

This is an exact *replay* of the submitted v2 data definition.  It intentionally
uses a frozen selection manifest, rather than silently recomputing membership
from the legacy 1,711-pair data.  A new behavioral screening run can be made
with :mod:`paper_repro.screen_candidates`; it should create a new dataset
version, not mutate v2 in place.
"""
from __future__ import annotations

import argparse
import shutil
from collections import Counter
from pathlib import Path

from .dataset import dataset_file_map, read_csv, replace_instruction_verb, sha256_file, write_json
from .paths import DATASETS_ROOT, PROVENANCE_ROOT, REPO_ROOT, RESULTS_ROOT


V1_SOURCE_ROOT = PROVENANCE_ROOT / "v1_1711" / "base_pairs"
V2_PROVENANCE_ROOT = PROVENANCE_ROOT / "v2_1500"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay the frozen 1,500-pair v2 dataset from archived provenance.")
    parser.add_argument("--source-root", type=Path, default=V1_SOURCE_ROOT)
    parser.add_argument("--selection-manifest", type=Path, default=V2_PROVENANCE_ROOT / "selection_manifest.csv")
    parser.add_argument("--split-manifest", type=Path, default=V2_PROVENANCE_ROOT / "split_manifest.csv")
    parser.add_argument("--frozen-manifest-root", type=Path, default=V2_PROVENANCE_ROOT / "manifests")
    parser.add_argument("--metadata-root", type=Path, default=V2_PROVENANCE_ROOT)
    parser.add_argument("--output-root", type=Path, default=RESULTS_ROOT / "rebuild_checks" / "v2_1500_rebuild")
    parser.add_argument("--verify-against", type=Path, default=DATASETS_ROOT)
    parser.add_argument("--overwrite", action="store_true", help="Replace this explicit output directory if it already exists.")
    return parser.parse_args()


def ensure_safe_output(output_root: Path, *, overwrite: bool) -> None:
    resolved = output_root.resolve()
    forbidden = {REPO_ROOT.resolve(), DATASETS_ROOT.resolve(), PROVENANCE_ROOT.resolve(), RESULTS_ROOT.resolve()}
    if resolved in forbidden:
        raise ValueError(f"Refusing to use broad repository directory as output: {resolved}")
    if output_root.exists():
        if not overwrite:
            raise FileExistsError(f"Output already exists: {output_root}; pass --overwrite for this exact directory.")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=False)


def copy_frozen_metadata(metadata_root: Path, output_root: Path) -> None:
    for name in [
        "merge_summary.json",
        "split_summary.json",
        "clean_assignment_audit.csv",
        "corrupt_assignment_audit.csv",
        "dataset_assignment_audit.csv",
        "dropped_samples.csv",
    ]:
        source = metadata_root / name
        if source.exists():
            shutil.copy2(source, output_root / name)
    for split in ("train", "test"):
        source = metadata_root / "split_metadata" / split / "merge_summary.json"
        if source.exists():
            target = output_root / split / "merge_summary.json"
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)


def copy_frozen_manifests(manifest_root: Path, output_root: Path) -> None:
    for split in ("train", "test"):
        for side in ("clean", "corrupt"):
            source = manifest_root / f"{split}_{side}.jsonl"
            target = output_root / split / side / "manifest.jsonl"
            if not source.exists():
                raise FileNotFoundError(f"Missing frozen manifest: {source}")
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)


def compare_trees(rebuilt_root: Path, reference_root: Path) -> dict[str, object]:
    rebuilt = dataset_file_map(rebuilt_root)
    reference = dataset_file_map(reference_root)
    only_rebuilt = sorted(set(rebuilt) - set(reference))
    only_reference = sorted(set(reference) - set(rebuilt))
    changed = [
        relative
        for relative in sorted(set(rebuilt) & set(reference))
        if sha256_file(rebuilt[relative]) != sha256_file(reference[relative])
    ]
    return {
        "exact_match": not only_rebuilt and not only_reference and not changed,
        "rebuilt_file_count": len(rebuilt),
        "reference_file_count": len(reference),
        "only_in_rebuild": only_rebuilt,
        "only_in_reference": only_reference,
        "content_mismatch": changed,
    }


def main() -> None:
    args = parse_args()
    source_root = args.source_root.resolve()
    selection_rows = read_csv(args.selection_manifest.resolve())
    split_rows = read_csv(args.split_manifest.resolve())
    split_by_filename = {row["filename"]: row["split"] for row in split_rows}
    if len(split_by_filename) != len(split_rows):
        raise ValueError("Split manifest contains duplicate filenames")
    selection_names = {row["filename"] for row in selection_rows}
    if selection_names != set(split_by_filename):
        missing_split = sorted(selection_names - set(split_by_filename))
        extra_split = sorted(set(split_by_filename) - selection_names)
        raise ValueError(f"Selection/split mismatch: missing_split={missing_split[:3]}, extra_split={extra_split[:3]}")

    ensure_safe_output(args.output_root, overwrite=args.overwrite)
    output_root = args.output_root.resolve()
    counts: Counter[str] = Counter()
    rendered_line_examples: list[dict[str, str]] = []

    for row in selection_rows:
        filename = row["filename"]
        split = split_by_filename[filename]
        if split not in {"train", "test"}:
            raise ValueError(f"Unexpected split {split!r} for {filename}")
        source_clean = source_root / "clean" / filename
        if not source_clean.exists():
            raise FileNotFoundError(f"Missing legacy source prompt: {source_clean}")
        scaffold = source_clean.read_text(encoding="utf-8")
        for side in ("clean", "corrupt"):
            prompt, rendered_line = replace_instruction_verb(scaffold, row[f"{side}_candidate"])
            destination = output_root / split / side / filename
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(prompt, encoding="utf-8")
            counts[f"{split}_{side}"] += 1
            if len(rendered_line_examples) < 3 and side == "clean":
                rendered_line_examples.append({"filename": filename, "line": rendered_line})

    copy_frozen_manifests(args.frozen_manifest_root.resolve(), output_root)
    copy_frozen_metadata(args.metadata_root.resolve(), output_root)
    comparison = compare_trees(output_root, args.verify_against.resolve())
    report = {
        "dataset_version": "v2_1500",
        "rebuild_mode": "frozen-selection replay",
        "source_root": str(source_root),
        "selection_manifest": str(args.selection_manifest.resolve()),
        "split_manifest": str(args.split_manifest.resolve()),
        "output_root": str(output_root),
        "verify_against": str(args.verify_against.resolve()),
        "pair_count": len(selection_rows),
        "counts": dict(sorted(counts.items())),
        "rendered_line_examples": rendered_line_examples,
        "comparison": comparison,
    }
    write_json(output_root / "rebuild_report.json", report)
    print(f"Rebuilt {len(selection_rows)} v2 pairs at {output_root}")
    print(f"Exact match with active v2 dataset: {comparison['exact_match']}")
    if not comparison["exact_match"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
