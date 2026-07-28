#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
USER_MARKER = "<|im_start|>user\n"

CLEAN_VERBS = ("save", "write", "add", "complete")
CORRUPT_VERBS = ("discuss", "explore", "inspect", "review", "study")


@dataclass(frozen=True)
class PromptTemplate:
    sample_id: str
    source_filename: str
    dataset_name: str
    language: str
    original_clean_candidate: str
    original_corrupt_candidate: str
    prompt_prefix: str
    prompt_suffix: str

    def render(self, verb: str) -> str:
        return self.prompt_prefix + verb.capitalize() + self.prompt_suffix


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a 500-pair VIBE-era Mistral dataset from current positive cues.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=PROJECT_ROOT / "datasets",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "results" / "runs" / "mistral_3p2_24b" / "source_pairs",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--target-pairs",
        type=int,
        default=500,
        help="Balanced output size; it must be divisible by the 4×5 verb grid.",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def extract_template(row: dict[str, Any], raw_text: str) -> PromptTemplate:
    marker_pos = raw_text.find(USER_MARKER)
    if marker_pos < 0:
        raise ValueError(f"{row['output_filename']}: missing user marker")
    user_start = marker_pos + len(USER_MARKER)
    tail = raw_text[user_start:]
    if not tail:
        raise ValueError(f"{row['output_filename']}: empty user content")
    verb_end = 0
    while verb_end < len(tail) and not tail[verb_end].isspace():
        verb_end += 1
    if verb_end <= 0:
        raise ValueError(f"{row['output_filename']}: failed to parse first verb")
    return PromptTemplate(
        sample_id=Path(str(row["output_filename"])).stem,
        source_filename=str(row["output_filename"]),
        dataset_name=str(row.get("dataset_name") or row.get("dataset") or ""),
        language=str(row["language"]),
        original_clean_candidate=str(row["clean_candidate"]),
        original_corrupt_candidate=str(row["corrupt_candidate"]),
        prompt_prefix=raw_text[:user_start],
        prompt_suffix=tail[verb_end:],
    )


def source_clean_rows(dataset_root: Path) -> list[tuple[dict[str, Any], Path]]:
    """Read either the old flat layout or the frozen v2 train/test layout.

    The historical Mistral selection script expected ``datasets/clean``.  The
    canonical v2 dataset intentionally keeps its split at
    ``datasets/{train,test}/clean``.  Supporting both layouts lets the same
    deterministic selection protocol be replayed without copying or mutating
    the active v2 data tree.
    """

    flat_manifest = dataset_root / "clean" / "manifest.jsonl"
    if flat_manifest.exists():
        return [
            (row, dataset_root / "clean" / str(row["output_filename"]))
            for row in read_jsonl(flat_manifest)
        ]

    rows: list[tuple[dict[str, Any], Path]] = []
    for split in ("train", "test"):
        manifest_path = dataset_root / split / "clean" / "manifest.jsonl"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Expected either {flat_manifest} or split manifest {manifest_path}."
            )
        for row in read_jsonl(manifest_path):
            copied = dict(row)
            copied["source_split"] = split
            rows.append((copied, dataset_root / split / "clean" / str(row["output_filename"])))
    return rows


def select_python_templates(dataset_root: Path, seed: int, *, target_pairs: int) -> list[PromptTemplate]:
    if target_pairs <= 0:
        raise ValueError("target_pairs must be positive")
    source_rows = source_clean_rows(dataset_root)
    python_rows = [item for item in source_rows if str(item[0]["language"]) == "python"]
    by_dataset: dict[str, list[tuple[dict[str, Any], Path]]] = defaultdict(list)
    for row, path in python_rows:
        by_dataset[str(row.get("dataset_name") or row.get("dataset") or "")].append((row, path))

    rng = random.Random(seed)
    selected_rows: list[tuple[dict[str, Any], Path]] = []

    # Keep all smaller Python pools and sample the remainder from APPS to maximize diversity.
    for dataset_name in ("humaneval", "mbpp"):
        rows = list(by_dataset[dataset_name])
        rows.sort(key=lambda item: str(item[0]["output_filename"]))
        selected_rows.extend(rows)

    apps_rows = list(by_dataset["apps"])
    rng.shuffle(apps_rows)
    remaining = target_pairs - len(selected_rows)
    if remaining < 0:
        raise RuntimeError(
            f"The fixed HumanEval/MBPP pools already contain {len(selected_rows)} rows, "
            f"which exceeds target_pairs={target_pairs}."
        )
    if len(apps_rows) < remaining:
        raise RuntimeError(
            f"The v2 Python pool supplies only {len(selected_rows) + len(apps_rows)} rows for "
            f"the historical Mistral selection policy, fewer than target_pairs={target_pairs}."
        )
    selected_rows.extend(apps_rows[:remaining])
    selected_rows.sort(
        key=lambda item: (str(item[0].get("dataset_name") or ""), str(item[0]["output_filename"]))
    )

    templates: list[PromptTemplate] = []
    for row, clean_path in selected_rows:
        raw_text = clean_path.read_text(encoding="utf-8")
        templates.append(extract_template(row, raw_text))
    if len(templates) != target_pairs:
        raise RuntimeError(f"Expected {target_pairs} selected templates, got {len(templates)}")
    return templates


def write_dataset(output_root: Path, templates: list[PromptTemplate], *, target_pairs: int) -> None:
    ensure_dir(output_root)

    combo_order = [(clean_verb, corrupt_verb) for clean_verb in CLEAN_VERBS for corrupt_verb in CORRUPT_VERBS]
    if target_pairs % len(combo_order) != 0:
        raise ValueError(
            f"target_pairs={target_pairs} is not divisible by the {len(combo_order)}-cell verb grid"
        )
    per_combo = target_pairs // len(combo_order)
    assigned_pairs = []
    for combo_idx, combo in enumerate(combo_order):
        start = combo_idx * per_combo
        stop = start + per_combo
        for template in templates[start:stop]:
            assigned_pairs.append((template, combo[0], combo[1], combo_idx))

    if len(assigned_pairs) != target_pairs:
        raise RuntimeError(f"Expected {target_pairs} assigned pairs, got {len(assigned_pairs)}")

    manifest_rows: list[dict[str, Any]] = []
    for pair_idx, (template, clean_verb, corrupt_verb, combo_idx) in enumerate(assigned_pairs, start=1):
        clean_path = output_root / f"clean_{pair_idx}.txt"
        corrupt_path = output_root / f"corrupt_{pair_idx}.txt"
        clean_text = template.render(clean_verb)
        corrupt_text = template.render(corrupt_verb)
        clean_path.write_text(clean_text, encoding="utf-8")
        corrupt_path.write_text(corrupt_text, encoding="utf-8")
        manifest_rows.append(
            {
                "pair_id": pair_idx,
                "clean_filename": clean_path.name,
                "corrupt_filename": corrupt_path.name,
                "source_sample_id": template.sample_id,
                "source_filename": template.source_filename,
                "dataset_name": template.dataset_name,
                "language": template.language,
                "original_clean_candidate": template.original_clean_candidate,
                "original_corrupt_candidate": template.original_corrupt_candidate,
                "assigned_clean_candidate": clean_verb,
                "assigned_corrupt_candidate": corrupt_verb,
                "combo_index": combo_idx,
            }
        )

    with (output_root / "pair_manifest.jsonl").open("w", encoding="utf-8") as handle:
        for row in manifest_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "n_pairs": len(manifest_rows),
        "target_pairs": target_pairs,
        "selection_rule": (
            "Use only python samples because current Mistral+VIBE partial results show the strongest clean/corrupt separation there. "
            "Exclude build because current partial evidence is negative. Reassign clean verbs uniformly over save/write/add/complete "
            "and corrupt verbs uniformly over discuss/explore/inspect/review/study."
        ),
        "clean_verbs": list(CLEAN_VERBS),
        "corrupt_verbs": list(CORRUPT_VERBS),
        "clean_counts": dict(Counter(row["assigned_clean_candidate"] for row in manifest_rows)),
        "corrupt_counts": dict(Counter(row["assigned_corrupt_candidate"] for row in manifest_rows)),
        "combo_counts": dict(
            Counter(
                f"{row['assigned_clean_candidate']}__{row['assigned_corrupt_candidate']}"
                for row in manifest_rows
            )
        ),
        "dataset_counts": dict(Counter(row["dataset_name"] for row in manifest_rows)),
        "language_counts": dict(Counter(row["language"] for row in manifest_rows)),
    }
    (output_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    templates = select_python_templates(args.dataset_root, args.seed, target_pairs=args.target_pairs)
    write_dataset(args.output_root, templates, target_pairs=args.target_pairs)


if __name__ == "__main__":
    main()
