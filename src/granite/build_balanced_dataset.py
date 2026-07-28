#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_ROOT = PROJECT_ROOT / "results" / "granite-3.3-8b-instruct"
DEFAULT_BEHAVIOR_ROOT = DEFAULT_RESULTS_ROOT / "behavior_scan"
DEFAULT_CONVERTED_ROOT = DEFAULT_RESULTS_ROOT / "converted_dataset"
DEFAULT_OUTPUT_ROOT = DEFAULT_RESULTS_ROOT / "dataset"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a behavior-valid, candidate-balanced Granite dataset from evaluated pairs.")
    parser.add_argument("--behavior-root", type=Path, default=DEFAULT_BEHAVIOR_ROOT)
    parser.add_argument("--converted-root", type=Path, default=DEFAULT_CONVERTED_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--target-pairs",
        type=int,
        default=0,
        help="Exact requested size; 0 selects the largest feasible balanced subset.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace only this explicit output directory.")
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def prepare_output_root(output_root: Path, *, overwrite: bool) -> Path:
    resolved = output_root.resolve()
    forbidden = {PROJECT_ROOT.resolve(), (PROJECT_ROOT / "datasets").resolve(), (PROJECT_ROOT / "results").resolve()}
    if resolved in forbidden:
        raise ValueError(f"Refusing broad output directory: {resolved}")
    if resolved.exists() and any(resolved.iterdir()):
        if not overwrite:
            raise FileExistsError(f"Output already exists: {resolved}; pass --overwrite for this exact directory.")
        shutil.rmtree(resolved)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def write_json(path: Path, payload: dict) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    ensure_dir(path.parent)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def int_partition(total: int, parts: list[str]) -> dict[str, int]:
    base = total // len(parts)
    remainder = total % len(parts)
    targets: dict[str, int] = {}
    for index, name in enumerate(parts):
        targets[name] = base + (1 if index < remainder else 0)
    return targets


def select_balanced_rows(
    *,
    buckets: dict[tuple[str, str], list[dict[str, str]]],
    stable_clean_candidates: list[str],
    corrupt_candidates: list[str],
    row_targets: dict[str, int],
    col_targets: dict[str, int],
    rng: random.Random,
) -> tuple[list[dict[str, str]], list[dict[str, object]]]:
    remaining_clean = dict(row_targets)
    selected_rows: list[dict[str, str]] = []
    selection_matrix_rows: list[dict[str, object]] = []

    for corrupt_candidate in corrupt_candidates:
        need = col_targets[corrupt_candidate]
        available_by_clean = {
            clean_candidate: list(buckets[(clean_candidate, corrupt_candidate)])
            for clean_candidate in stable_clean_candidates
        }
        for rows in available_by_clean.values():
            rng.shuffle(rows)

        chosen_per_clean: dict[str, int] = {clean_candidate: 0 for clean_candidate in stable_clean_candidates}
        while need > 0:
            eligible = [
                clean_candidate
                for clean_candidate in stable_clean_candidates
                if remaining_clean[clean_candidate] > 0
                and available_by_clean[clean_candidate]
            ]
            if not eligible:
                raise RuntimeError(f"Could not fulfill corrupt candidate quota for {corrupt_candidate}.")

            eligible.sort(
                key=lambda clean_candidate: (
                    -remaining_clean[clean_candidate],
                    -len(available_by_clean[clean_candidate]),
                    clean_candidate,
                )
            )
            clean_candidate = eligible[0]
            row = available_by_clean[clean_candidate].pop()
            selected_rows.append(row)
            chosen_per_clean[clean_candidate] += 1
            remaining_clean[clean_candidate] -= 1
            need -= 1

        for clean_candidate in stable_clean_candidates:
            selection_matrix_rows.append(
                {
                    "clean_candidate": clean_candidate,
                    "corrupt_candidate": corrupt_candidate,
                    "selected_count": chosen_per_clean[clean_candidate],
                    "available_count": len(buckets[(clean_candidate, corrupt_candidate)]),
                }
            )

    if any(value != 0 for value in remaining_clean.values()):
        raise RuntimeError(f"Clean quotas not fully consumed: {remaining_clean}")
    return selected_rows, selection_matrix_rows


def select_for_target(
    *,
    target_pairs: int,
    buckets: dict[tuple[str, str], list[dict[str, str]]],
    stable_clean_candidates: list[str],
    corrupt_candidates: list[str],
    seed: int,
) -> tuple[list[dict[str, str]], list[dict[str, object]], dict[str, int], dict[str, int]]:
    if target_pairs <= 0:
        raise ValueError("target_pairs must be positive inside select_for_target")
    if target_pairs % len(corrupt_candidates) != 0:
        raise ValueError("target_pairs must be divisible by the number of corrupt candidates for exact balancing.")
    row_targets = int_partition(target_pairs, stable_clean_candidates)
    col_targets = {name: target_pairs // len(corrupt_candidates) for name in corrupt_candidates}
    selected_rows, matrix_rows = select_balanced_rows(
        buckets=buckets,
        stable_clean_candidates=stable_clean_candidates,
        corrupt_candidates=corrupt_candidates,
        row_targets=row_targets,
        col_targets=col_targets,
        rng=random.Random(seed),
    )
    if len(selected_rows) != target_pairs:
        raise RuntimeError(f"Selected {len(selected_rows)} rows, expected {target_pairs}.")
    return selected_rows, matrix_rows, row_targets, col_targets


def summarize_selected(rows: list[dict[str, str]]) -> dict[str, object]:
    def rate(key: str) -> float:
        return float(sum(int(row[key] == "True") for row in rows) / len(rows)) if rows else 0.0

    clean_counter = Counter(row["clean_candidate"] for row in rows)
    corrupt_counter = Counter(row["corrupt_candidate"] for row in rows)
    language_counter = Counter(row["language"] for row in rows)
    dataset_counter = Counter(row["dataset_name"] for row in rows)
    quadrant_counter = Counter()
    for row in rows:
        c = row["clean_is_tool_call_top1"] == "True"
        x = row["corrupt_is_tool_call_top1"] == "True"
        if c and x:
            quadrant_counter["both_tool"] += 1
        elif c and not x:
            quadrant_counter["clean_only"] += 1
        elif (not c) and x:
            quadrant_counter["corrupt_only"] += 1
        else:
            quadrant_counter["neither"] += 1
    return {
        "n_pairs": len(rows),
        "clean_tool_call_top1_rate": rate("clean_is_tool_call_top1"),
        "corrupt_tool_call_top1_rate": rate("corrupt_is_tool_call_top1"),
        "clean_candidate_counts": dict(clean_counter),
        "corrupt_candidate_counts": dict(corrupt_counter),
        "language_counts": dict(language_counter),
        "dataset_counts": dict(dataset_counter),
        "quadrant_counts": dict(quadrant_counter),
    }


def build_summary_md(summary: dict[str, object], selection_matrix_rows: list[dict[str, object]]) -> str:
    lines = [
        "# Granite Behavior-Validated Balanced Dataset",
        "",
        f"- n_pairs: {summary['n_pairs']}",
        f"- clean tool-call rate: {summary['clean_tool_call_top1_rate']:.4f}",
        f"- corrupt tool-call rate: {summary['corrupt_tool_call_top1_rate']:.4f}",
        f"- clean minus corrupt gap: {summary['clean_tool_call_top1_rate'] - summary['corrupt_tool_call_top1_rate']:.4f}",
        f"- language_counts: {summary['language_counts']}",
        f"- clean_candidate_counts: {summary['clean_candidate_counts']}",
        f"- corrupt_candidate_counts: {summary['corrupt_candidate_counts']}",
        f"- quadrant_counts: {summary['quadrant_counts']}",
        "",
        "## Selection matrix",
        "",
        "| clean | corrupt | selected |",
        "| --- | --- | ---: |",
    ]
    for row in selection_matrix_rows:
        lines.append(f"| {row['clean_candidate']} | {row['corrupt_candidate']} | {row['selected_count']} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    if args.target_pairs < 0:
        raise ValueError("--target-pairs must be non-negative")

    pair_rows = read_csv(args.behavior_root / "pair_decisions.csv")
    canonical_rows = read_jsonl(args.converted_root / "canonical_pairs.jsonl")
    canonical_by_sample_id = {str(row["sample_id"]): row for row in canonical_rows}

    stable_clean_candidates = ["add", "save", "write"]
    corrupt_candidates = ["discuss", "explore", "inspect", "review", "study"]

    filtered_rows: list[dict[str, str]] = []
    for row in pair_rows:
        clean_top1 = row["clean_is_tool_call_top1"] == "True"
        corrupt_top1 = row["corrupt_is_tool_call_top1"] == "True"
        if not clean_top1 or corrupt_top1:
            continue
        if row["clean_candidate"] not in stable_clean_candidates:
            continue
        if row["corrupt_candidate"] not in corrupt_candidates:
            continue
        filtered_rows.append(row)

    buckets: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in filtered_rows:
        key = (row["clean_candidate"], row["corrupt_candidate"])
        buckets[key].append(row)

    if args.target_pairs:
        if len(filtered_rows) < args.target_pairs:
            raise RuntimeError(
                f"Only {len(filtered_rows)} clean-only stable pairs available, fewer than target {args.target_pairs}."
            )
        selected_rows, selection_matrix_rows, row_targets, col_targets = select_for_target(
            target_pairs=args.target_pairs,
            buckets=buckets,
            stable_clean_candidates=stable_clean_candidates,
            corrupt_candidates=corrupt_candidates,
            seed=args.seed,
        )
        selected_target = args.target_pairs
    else:
        selected_rows = []
        selection_matrix_rows = []
        row_targets = {}
        col_targets = {}
        max_target = len(filtered_rows) - (len(filtered_rows) % len(corrupt_candidates))
        for target in range(max_target, 0, -len(corrupt_candidates)):
            try:
                selected_rows, selection_matrix_rows, row_targets, col_targets = select_for_target(
                    target_pairs=target,
                    buckets=buckets,
                    stable_clean_candidates=stable_clean_candidates,
                    corrupt_candidates=corrupt_candidates,
                    seed=args.seed,
                )
            except RuntimeError:
                continue
            selected_target = target
            break
        else:
            raise RuntimeError("No behavior-valid Granite subset satisfies the requested candidate balance.")

    selected_rows = sorted(selected_rows, key=lambda row: row["sample_id"])

    args.output_root = prepare_output_root(args.output_root, overwrite=args.overwrite)
    clean_root = args.output_root / "clean"
    corrupt_root = args.output_root / "corrupt"
    ensure_dir(clean_root)
    ensure_dir(corrupt_root)

    manifest_rows: list[dict[str, object]] = []
    canonical_selected_rows: list[dict] = []
    for dataset_index, row in enumerate(selected_rows, start=1):
        sample_id = row["sample_id"]
        canonical = canonical_by_sample_id[sample_id]
        clean_src = Path(row["clean_prompt_path"])
        corrupt_src = Path(row["corrupt_prompt_path"])
        clean_dst = clean_root / clean_src.name
        corrupt_dst = corrupt_root / corrupt_src.name
        shutil.copyfile(clean_src, clean_dst)
        shutil.copyfile(corrupt_src, corrupt_dst)
        manifest_rows.append(
            {
                "dataset_index": dataset_index,
                "sample_id": sample_id,
                "split": row["split"],
                "dataset_name": row["dataset_name"],
                "language": row["language"],
                "template_kind": row["template_kind"],
                "clean_candidate": row["clean_candidate"],
                "corrupt_candidate": row["corrupt_candidate"],
                "clean_prompt_path": str(clean_dst.resolve()),
                "corrupt_prompt_path": str(corrupt_dst.resolve()),
                "source_clean_prompt_path": row["clean_prompt_path"],
                "source_corrupt_prompt_path": row["corrupt_prompt_path"],
                "clean_tool_token_prob": float(row["clean_tool_token_prob"]),
                "corrupt_tool_token_prob": float(row["corrupt_tool_token_prob"]),
                "clean_top1_token_text": row["clean_top1_token_text"],
                "corrupt_top1_token_text": row["corrupt_top1_token_text"],
                "clean_is_tool_call_top1": row["clean_is_tool_call_top1"] == "True",
                "corrupt_is_tool_call_top1": row["corrupt_is_tool_call_top1"] == "True",
                "selection_seed": args.seed,
                "edit_applied": False,
            }
        )
        canonical_selected_rows.append(canonical)

    summary = summarize_selected(selected_rows)
    summary.update(
        {
            "seed": args.seed,
            "target_pairs_requested": args.target_pairs,
            "target_pairs": selected_target,
            "selection_policy": {
                "pair_filter": "clean_is_tool_call_top1 == True and corrupt_is_tool_call_top1 == False",
                "clean_candidates": stable_clean_candidates,
                "corrupt_candidates": corrupt_candidates,
                "row_targets": row_targets,
                "col_targets": col_targets,
                "edit_applied_count": 0,
            },
        }
    )

    write_csv(args.output_root / "selected_pair_decisions.csv", selected_rows)
    write_csv(args.output_root / "selection_matrix.csv", selection_matrix_rows)
    write_jsonl(args.output_root / "manifest.jsonl", manifest_rows)
    write_jsonl(args.output_root / "canonical_pairs.jsonl", canonical_selected_rows)
    write_json(args.output_root / "selection_summary.json", summary)
    (args.output_root / "selection_summary.md").write_text(
        build_summary_md(summary, selection_matrix_rows),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "n_pairs": summary["n_pairs"],
                "clean_tool_call_top1_rate": summary["clean_tool_call_top1_rate"],
                "corrupt_tool_call_top1_rate": summary["corrupt_tool_call_top1_rate"],
                "clean_candidate_counts": summary["clean_candidate_counts"],
                "corrupt_candidate_counts": summary["corrupt_candidate_counts"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
