#!/usr/bin/env python3
"""Freeze a fresh Mistral-native subset only after behavioral screening.

The old Mistral workflow selected a source pool and then ran a behavior scan,
but downstream mechanism scripts could still consume the unfiltered source
pool.  This selector makes the behavior criterion an explicit data boundary:
only ``clean=<tool call>`` and ``corrupt!=<tool call>`` pairs are copied into
the downstream dataset.  It writes a portable manifest with current-run paths.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select balanced behavior-valid Mistral native pairs.")
    parser.add_argument("--converted-root", type=Path, required=True)
    parser.add_argument("--behavior-csv", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--target-pairs",
        type=int,
        default=0,
        help="Exact balanced target. 0 selects the largest complete clean×corrupt verb grid.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace only this explicit output directory.")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def as_bool(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def as_float(value: object) -> float:
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return float("nan")


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def resolve_current_prompt_path(raw: object, *, converted_root: Path, side: str, sample_id: str) -> Path:
    value = Path(str(raw or ""))
    candidates: list[Path] = []
    if value.name:
        candidates.extend((converted_root / side / value.name, converted_root / value.name))
    candidates.append(converted_root / side / f"{sample_id}.txt")
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        f"Could not resolve current {side} prompt for sample={sample_id!r} under {converted_root}; "
        f"tried {[str(item) for item in candidates]}"
    )


def prepare_output_root(output_root: Path, *, overwrite: bool) -> Path:
    resolved = output_root.resolve()
    forbidden = {PROJECT_ROOT.resolve(), (PROJECT_ROOT / "datasets").resolve(), (PROJECT_ROOT / "results").resolve()}
    if resolved in forbidden:
        raise ValueError(f"Refusing broad output directory: {resolved}")
    if resolved.exists():
        if not overwrite:
            raise FileExistsError(f"Output already exists: {resolved}; pass --overwrite for this exact directory.")
        shutil.rmtree(resolved)
    (resolved / "clean").mkdir(parents=True, exist_ok=False)
    (resolved / "corrupt").mkdir(parents=True, exist_ok=False)
    return resolved


def main() -> None:
    args = parse_args()
    if args.target_pairs < 0:
        raise ValueError("--target-pairs must be non-negative")
    converted_root = args.converted_root.resolve()
    manifest_path = converted_root / "manifest.jsonl"
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    manifest_rows = read_jsonl(manifest_path)
    manifest_by_id = {str(row["sample_id"]): row for row in manifest_rows}
    if len(manifest_by_id) != len(manifest_rows):
        raise ValueError(f"Duplicate sample_id values in {manifest_path}")

    behavior_rows = read_csv(args.behavior_csv.resolve())
    valid_by_combo: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    missing_manifest_ids: list[str] = []
    for behavior in behavior_rows:
        sample_id = str(behavior.get("sample_id") or "").strip()
        if not sample_id:
            continue
        manifest = manifest_by_id.get(sample_id)
        if manifest is None:
            missing_manifest_ids.append(sample_id)
            continue
        if not as_bool(behavior.get("clean_is_tool_call_top1")):
            continue
        if as_bool(behavior.get("corrupt_is_tool_call_top1")):
            continue
        clean_candidate = str(behavior.get("clean_candidate") or manifest.get("clean_candidate") or "")
        corrupt_candidate = str(behavior.get("corrupt_candidate") or manifest.get("corrupt_candidate") or "")
        if not clean_candidate or not corrupt_candidate:
            raise ValueError(f"Missing verb labels for behavior-valid sample {sample_id}")
        clean_path = resolve_current_prompt_path(
            manifest.get("clean_prompt_path"), converted_root=converted_root, side="clean", sample_id=sample_id
        )
        corrupt_path = resolve_current_prompt_path(
            manifest.get("corrupt_prompt_path"), converted_root=converted_root, side="corrupt", sample_id=sample_id
        )
        margin = as_float(behavior.get("clean_tool_token_prob")) - as_float(behavior.get("corrupt_tool_token_prob"))
        valid_by_combo[(clean_candidate, corrupt_candidate)].append(
            {
                "sample_id": sample_id,
                "split": str(behavior.get("split") or manifest.get("split") or "all"),
                "language": str(behavior.get("language") or manifest.get("language") or "unknown"),
                "dataset_name": str(manifest.get("dataset_name") or "unknown"),
                "clean_candidate": clean_candidate,
                "corrupt_candidate": corrupt_candidate,
                "clean_path": clean_path,
                "corrupt_path": corrupt_path,
                "clean_tool_token_prob": as_float(behavior.get("clean_tool_token_prob")),
                "corrupt_tool_token_prob": as_float(behavior.get("corrupt_tool_token_prob")),
                "clean_top1_token_text": str(behavior.get("clean_top1_token_text") or ""),
                "corrupt_top1_token_text": str(behavior.get("corrupt_top1_token_text") or ""),
                "score_margin": margin,
            }
        )
    if missing_manifest_ids:
        raise ValueError(
            f"Behavior file contains {len(missing_manifest_ids)} IDs absent from the current converted manifest; "
            f"first={missing_manifest_ids[:5]}"
        )
    if not valid_by_combo:
        raise RuntimeError("No behavior-valid Mistral pairs were found")

    combos = sorted(valid_by_combo)
    for combo in combos:
        valid_by_combo[combo].sort(
            key=lambda row: (-float(row["score_margin"]), -float(row["clean_tool_token_prob"]), str(row["sample_id"]))
        )
    if args.target_pairs:
        if args.target_pairs % len(combos) != 0:
            raise ValueError(f"target_pairs={args.target_pairs} is not divisible by {len(combos)} observed verb combinations")
        per_combo = args.target_pairs // len(combos)
    else:
        per_combo = min(len(rows) for rows in valid_by_combo.values())
    if per_combo <= 0:
        raise RuntimeError("At least one observed verb combination has no behavior-valid pairs")
    shortfalls = {f"{clean}|{corrupt}": len(rows) for (clean, corrupt), rows in valid_by_combo.items() if len(rows) < per_combo}
    if shortfalls:
        raise RuntimeError(f"Cannot satisfy balanced selection of {per_combo} per verb pair: {shortfalls}")

    selected = [row for combo in combos for row in valid_by_combo[combo][:per_combo]]
    selected.sort(key=lambda row: (str(row["clean_candidate"]), str(row["corrupt_candidate"]), str(row["sample_id"])))
    output_root = prepare_output_root(args.output_root, overwrite=args.overwrite)
    manifest_output: list[dict[str, object]] = []
    for pair_id, row in enumerate(selected, start=1):
        clean_source = Path(row["clean_path"])
        corrupt_source = Path(row["corrupt_path"])
        clean_destination = output_root / "clean" / f"clean_{pair_id}.txt"
        corrupt_destination = output_root / "corrupt" / f"corrupt_{pair_id}.txt"
        shutil.copyfile(clean_source, clean_destination)
        shutil.copyfile(corrupt_source, corrupt_destination)
        manifest_output.append(
            {
                "pair_id": pair_id,
                "sample_id": row["sample_id"],
                "split": row["split"],
                "language": row["language"],
                "dataset_name": row["dataset_name"],
                "clean_candidate": row["clean_candidate"],
                "corrupt_candidate": row["corrupt_candidate"],
                "clean_path": str(clean_destination.resolve()),
                "corrupt_path": str(corrupt_destination.resolve()),
                # Mistral's mechanism loader uses these native-renderer names;
                # the generic triplet loader also accepts the shorter aliases.
                "clean_prompt_path": str(clean_destination.resolve()),
                "corrupt_prompt_path": str(corrupt_destination.resolve()),
                "source_clean_prompt_path": str(clean_source),
                "source_corrupt_prompt_path": str(corrupt_source),
                "clean_tool_token_prob": row["clean_tool_token_prob"],
                "corrupt_tool_token_prob": row["corrupt_tool_token_prob"],
                "clean_top1_token_text": row["clean_top1_token_text"],
                "corrupt_top1_token_text": row["corrupt_top1_token_text"],
                "score_margin": row["score_margin"],
                "selection_rank_within_combo": next(
                    index + 1 for index, candidate in enumerate(valid_by_combo[(row["clean_candidate"], row["corrupt_candidate"])])
                    if candidate["sample_id"] == row["sample_id"]
                ),
                "selection_rule": "clean_is_tool_call_top1 and not corrupt_is_tool_call_top1; balanced by clean/corrupt verb pair",
            }
        )

    matrix_rows = [
        {
            "clean_candidate": clean,
            "corrupt_candidate": corrupt,
            "behavior_valid_available": len(valid_by_combo[(clean, corrupt)]),
            "selected": per_combo,
        }
        for clean, corrupt in combos
    ]
    write_jsonl(output_root / "manifest.jsonl", manifest_output)
    write_csv(output_root / "manifest.csv", manifest_output)
    write_csv(output_root / "selection_matrix.csv", matrix_rows)
    write_csv(output_root / "selected_pair_decisions.csv", manifest_output)
    summary = {
        "converted_root": str(converted_root),
        "behavior_csv": str(args.behavior_csv.resolve()),
        "target_pairs_requested": args.target_pairs,
        "selected_pairs": len(manifest_output),
        "verb_combinations": len(combos),
        "per_combo_selected": per_combo,
        "clean_candidate_counts": dict(sorted(Counter(str(row["clean_candidate"]) for row in manifest_output).items())),
        "corrupt_candidate_counts": dict(sorted(Counter(str(row["corrupt_candidate"]) for row in manifest_output).items())),
        "language_counts": dict(sorted(Counter(str(row["language"]) for row in manifest_output).items())),
        "split_counts": dict(sorted(Counter(str(row["split"]) for row in manifest_output).items())),
        "selection_rule": "strict paired first-token behavior filter followed by exact clean×corrupt verb balancing",
    }
    (output_root / "selection_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_root / "selection_summary.md").write_text(
        "\n".join(
            [
                "# Fresh Mistral Behavior-Validated Subset",
                "",
                f"- Selected pairs: `{len(manifest_output)}`",
                f"- Clean×corrupt verb cells: `{len(combos)}`",
                f"- Pairs per cell: `{per_combo}`",
                f"- Language counts: `{summary['language_counts']}`",
                f"- Selection rule: {summary['selection_rule']}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
