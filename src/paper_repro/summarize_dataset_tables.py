#!/usr/bin/env python3
"""Generate fresh, data-side paper tables from manifests and raw screens.

The submitted Appendix A contains two different kinds of numbers: candidate
verb behavior rates, and the final paired-data composition.  This tool keeps
them separate.  It never reads a selected external-model subset and never
parses values back out of the manuscript.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable


MODEL_ORDER = {
    "Qwen3-0.6B": 0,
    "Qwen3-1.7B": 1,
    "Qwen3-4B": 2,
    "Qwen3-8B": 3,
    "Qwen3-14B": 4,
}
POOL_ORDER = {"clean": 0, "corrupt": 1}
SOURCE_ORDER = {"apps": 0, "codecontests": 1, "humaneval": 2, "mbpp": 3}
APPENDIX_EXECUTION_VERBS = {"add", "build", "complete", "create", "generate", "implement", "modify", "save", "update", "write"}
APPENDIX_ANALYSIS_VERBS = {"analyze", "assess", "compare", "discuss", "examine", "explore", "inspect", "review", "summarize", "study"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Write data-composition and Appendix-A screening tables from one explicit dataset/run."
    )
    parser.add_argument("--dataset-root", type=Path, required=True, help="Frozen v2 or a freshly built paired-data root.")
    parser.add_argument(
        "--screen-csv",
        type=Path,
        default=None,
        help="Optional raw --candidate-profile appendix_audit CSV. When supplied, writes candidate-rate tables too.",
    )
    parser.add_argument(
        "--strict-appendix-screen",
        action="store_true",
        help="Require the submitted Appendix-A 10 execution and 10 analysis candidates, rather than accepting a smaller diagnostic screen.",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No rows in {path}")
    return rows


def write_csv(path: Path, rows: Iterable[dict[str, object]], fieldnames: list[str] | None = None) -> None:
    materialized = list(rows)
    if not materialized:
        if fieldnames is None:
            raise ValueError(f"Cannot infer CSV fields for empty output {path}")
        path.write_text(",".join(fieldnames) + "\n", encoding="utf-8")
        return
    fields = fieldnames or list(materialized[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(materialized)


def format_source(source: str) -> str:
    return {
        "apps": "APPS",
        "codecontests": "CodeContests",
        "humaneval": "HumanEval",
        "mbpp": "MBPP",
    }.get(source.lower(), source)


def assignment_path(dataset_root: Path) -> Path:
    for candidate in (dataset_root / "dataset_assignment_audit.csv", dataset_root / "selection_manifest.csv"):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"Could not find an assignment manifest below {dataset_root}; expected dataset_assignment_audit.csv or selection_manifest.csv"
    )


def split_path(dataset_root: Path) -> Path:
    candidates = (
        dataset_root / "split_manifest.csv",
        dataset_root / "provenance" / "v2_1500" / "split_manifest.csv",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Could not find a split manifest below {dataset_root}")


def load_assignments(dataset_root: Path) -> tuple[list[dict[str, str]], Path, Path]:
    assignments_file = assignment_path(dataset_root)
    splits_file = split_path(dataset_root)
    assignments = read_csv(assignments_file)
    splits = read_csv(splits_file)
    required = {"filename", "language", "dataset_name", "clean_candidate", "corrupt_candidate"}
    missing = required - set(assignments[0])
    if missing:
        raise ValueError(f"{assignments_file} lacks required assignment columns: {sorted(missing)}")
    split_by_filename: dict[str, str] = {}
    for row in splits:
        filename = row.get("filename", "")
        split = row.get("split", "")
        if not filename or split not in {"train", "test"}:
            raise ValueError(f"Malformed split row in {splits_file}: {row}")
        if filename in split_by_filename:
            raise ValueError(f"Duplicate split assignment for {filename}")
        split_by_filename[filename] = split
    normalized: list[dict[str, str]] = []
    for row in assignments:
        filename = row["filename"]
        split = split_by_filename.get(filename)
        if split is None:
            raise ValueError(f"Assignment file references a filename absent from split manifest: {filename}")
        normalized.append({**row, "split": split})
    if len(split_by_filename) != len(normalized):
        missing = sorted(set(split_by_filename) - {row["filename"] for row in normalized})
        raise ValueError(f"Split manifest has {len(missing)} unassigned filename(s), e.g. {missing[:3]}")
    return normalized, assignments_file, splits_file


def composition_rows(assignments: list[dict[str, str]]) -> list[dict[str, object]]:
    counts: Counter[tuple[str, str, str]] = Counter()
    totals: Counter[str] = Counter()
    for row in assignments:
        split = row["split"]
        source = row["dataset_name"].lower()
        language = row["language"].lower()
        counts[(split, source, language)] += 1
        totals[split] += 1
    rows: list[dict[str, object]] = []
    for split in ("train", "test"):
        rows.append({"split": split, "source": "all", "language": "all", "pairs": totals[split]})
        for (row_split, source, language), count in sorted(
            counts.items(), key=lambda item: (SOURCE_ORDER.get(item[0][1], 99), item[0][1], item[0][2])
        ):
            if row_split == split:
                rows.append({"split": split, "source": source, "language": language, "pairs": count})
    return rows


def verb_rows(assignments: list[dict[str, str]]) -> list[dict[str, object]]:
    total: Counter[tuple[str, str]] = Counter()
    by_split: Counter[tuple[str, str, str]] = Counter()
    for row in assignments:
        split = row["split"]
        for pool, field in (("clean", "clean_candidate"), ("corrupt", "corrupt_candidate")):
            candidate = row[field].lower()
            total[(pool, candidate)] += 1
            by_split[(pool, candidate, split)] += 1
    rows: list[dict[str, object]] = []
    for pool in ("clean", "corrupt"):
        candidates = sorted(candidate for item_pool, candidate in total if item_pool == pool)
        for candidate in candidates:
            rows.append(
                {
                    "pool": pool,
                    "candidate": candidate,
                    "total_pairs": total[(pool, candidate)],
                    "train_pairs": by_split[(pool, candidate, "train")],
                    "test_pairs": by_split[(pool, candidate, "test")],
                }
            )
    return rows


def bool_value(raw: object) -> bool:
    return str(raw).strip().lower() in {"1", "true", "yes"}


def screening_rows(path: Path) -> tuple[list[dict[str, object]], dict[str, list[str]]]:
    rows = read_csv(path)
    required = {"pool", "candidate"}
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"{path} lacks required screening columns: {sorted(missing)}")
    models = sorted(
        (field[: -len("_is_tool_call_top1")] for field in rows[0] if field.endswith("_is_tool_call_top1")),
        key=lambda model: (MODEL_ORDER.get(model, 99), model),
    )
    if not models:
        raise ValueError(f"No *_is_tool_call_top1 columns in {path}")
    candidates: dict[str, list[str]] = {
        pool: sorted({row["candidate"].lower() for row in rows if row["pool"].lower() == pool})
        for pool in ("clean", "corrupt")
    }
    output: list[dict[str, object]] = []
    for model in models:
        field = f"{model}_is_tool_call_top1"
        for pool in ("clean", "corrupt"):
            for candidate in candidates[pool]:
                values = [bool_value(row[field]) for row in rows if row["pool"].lower() == pool and row["candidate"].lower() == candidate]
                if not values:
                    raise ValueError(f"No rows for {model}/{pool}/{candidate} in {path}")
                output.append(
                    {
                        "model": model,
                        "pool": pool,
                        "candidate": candidate,
                        "n_prompts": len(values),
                        "tool_call_top1_count": sum(values),
                        "tool_call_top1_rate": sum(values) / len(values),
                    }
                )
    return output, candidates


def screening_wide_rows(rows: list[dict[str, object]], pool: str, candidates: list[str]) -> list[dict[str, object]]:
    lookup = {(str(row["model"]), str(row["pool"]), str(row["candidate"])): row for row in rows}
    models = sorted({str(row["model"]) for row in rows}, key=lambda model: (MODEL_ORDER.get(model, 99), model))
    return [
        {
            "model": model,
            **{
                candidate: round(100.0 * float(lookup[(model, pool, candidate)]["tool_call_top1_rate"]), 6)
                for candidate in candidates
            },
        }
        for model in models
    ]


def markdown_table(headers: list[str], rows: Iterable[Iterable[object]]) -> list[str]:
    materialized = [[str(value) for value in row] for row in rows]
    return [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
        *("| " + " | ".join(row) + " |" for row in materialized),
    ]


def latex_rate_rows(rows: list[dict[str, object]], candidates: list[str]) -> str:
    return "\n".join(
        " & ".join([str(row["model"]), *(f"{float(row[candidate]):.1f}" for candidate in candidates)]) + r" \\"
        for row in rows
    ) + "\n"


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    assignments, assignments_file, splits_file = load_assignments(dataset_root)

    composition = composition_rows(assignments)
    verbs = verb_rows(assignments)
    write_csv(output_root / "dataset_split_composition.csv", composition)
    write_csv(output_root / "dataset_verb_distribution.csv", verbs)

    summary: dict[str, object] = {
        "dataset_root": str(dataset_root),
        "assignment_manifest": str(assignments_file),
        "split_manifest": str(splits_file),
        "n_pairs": len(assignments),
        "split_counts": dict(sorted(Counter(row["split"] for row in assignments).items())),
        "source_counts": dict(sorted(Counter(row["dataset_name"].lower() for row in assignments).items())),
        "language_counts": dict(sorted(Counter(row["language"].lower() for row in assignments).items())),
    }

    markdown = ["# Dataset-side regenerated tables", "", "## Split composition", ""]
    markdown.extend(
        markdown_table(
            ["Split", "Source", "Language", "Pairs"],
            ((row["split"], format_source(str(row["source"])), row["language"], row["pairs"]) for row in composition),
        )
    )
    markdown.extend(["", "## Verb distribution", ""])
    markdown.extend(
        markdown_table(
            ["Pool", "Verb", "Total", "Train", "Test"],
            ((row["pool"], row["candidate"], row["total_pairs"], row["train_pairs"], row["test_pairs"]) for row in verbs),
        )
    )

    if args.screen_csv is not None:
        screen_path = args.screen_csv.expanduser().resolve()
        rates, candidates = screening_rows(screen_path)
        if args.strict_appendix_screen:
            if set(candidates["clean"]) != APPENDIX_EXECUTION_VERBS or set(candidates["corrupt"]) != APPENDIX_ANALYSIS_VERBS:
                raise ValueError(
                    "--strict-appendix-screen requires the exact 10+10 Appendix-A candidate protocol; "
                    f"got clean={candidates['clean']} corrupt={candidates['corrupt']}"
                )
        write_csv(output_root / "appendix_candidate_screen_rates.csv", rates)
        summary["screen_csv"] = str(screen_path)
        summary["screen_candidate_pools"] = candidates
        for pool, stem in (("clean", "appendix_execution_verb_screening"), ("corrupt", "appendix_analysis_verb_screening")):
            wide = screening_wide_rows(rates, pool, candidates[pool])
            write_csv(output_root / f"{stem}.csv", wide)
            (output_root / f"{stem}.tex").write_text(latex_rate_rows(wide, candidates[pool]), encoding="utf-8")
            markdown.extend(["", f"## {pool.title()} candidate screen", ""])
            markdown.extend(
                markdown_table(
                    ["Model", *candidates[pool]],
                    ([row["model"], *(f"{float(row[candidate]):.1f}%" for candidate in candidates[pool])] for row in wide),
                )
            )

    (output_root / "dataset_table_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_root / "dataset_tables.md").write_text("\n".join(markdown).rstrip() + "\n", encoding="utf-8")
    print(json.dumps({"output_root": str(output_root), "n_pairs": len(assignments), "screening": args.screen_csv is not None}, ensure_ascii=False))


if __name__ == "__main__":
    main()
