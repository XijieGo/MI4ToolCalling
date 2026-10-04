#!/usr/bin/env python3
"""Build the canonical Qwen3-8B input collection for the clean rerun.

The script deliberately imports only screened *inputs*.  It does not copy a
historical vector, baseline prediction, intervention result, or feature
selection.  The pair split is re-created from all 500 model-specific pairs
with fixed verb-stratified quotas: 300 fit pairs and 200 held-out pairs.
"""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_PAIR_ROOT = REPO_ROOT / "datasets/qwen3_8b/native"
SOURCE_VERB_FREE = REPO_ROOT / "datasets/qwen3_8b/verb_free/implicit_intent.jsonl"
SOURCE_TAU2_ROOT = (
    REPO_ROOT.parent
    / "MI4ToolCalling/datasets/external/tau2_telecom_qwen35_9b/collections"
    / "bidirectional_200_per_model_20260726_canonical_batch1/qwen3_8b"
)
TARGET_ROOT = REPO_ROOT / "datasets/qwen3_8b"

SPLIT_SEED = "qwen3_8b_clean_rerun_pair_split"
EXECUTION_VERBS = ("add", "build", "complete", "save", "write")
ANALYSIS_VERBS = ("discuss", "explore", "review", "study")

# Each row sums to 40 and each column sums to 50.  These are the closest
# integer 40% allocations of the 20 screened verb-pair strata.
HELDOUT_QUOTAS: dict[tuple[str, str], int] = {
    ("add", "discuss"): 10,
    ("add", "explore"): 10,
    ("add", "review"): 9,
    ("add", "study"): 11,
    ("build", "discuss"): 9,
    ("build", "explore"): 11,
    ("build", "review"): 10,
    ("build", "study"): 10,
    ("complete", "discuss"): 9,
    ("complete", "explore"): 10,
    ("complete", "review"): 12,
    ("complete", "study"): 9,
    ("save", "discuss"): 11,
    ("save", "explore"): 10,
    ("save", "review"): 10,
    ("save", "study"): 9,
    ("write", "discuss"): 11,
    ("write", "explore"): 9,
    ("write", "review"): 9,
    ("write", "study"): 11,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_rank(row: dict[str, Any]) -> str:
    source_id = str(row["source_sample_id"])
    return hashlib.sha256(f"{SPLIT_SEED}:{source_id}".encode("utf-8")).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def check_quotas(rows: list[dict[str, Any]]) -> dict[tuple[str, str], list[dict[str, Any]]]:
    if len(rows) != 500:
        raise ValueError(f"Expected 500 source pairs, found {len(rows)}")
    if len({row["source_sample_id"] for row in rows}) != len(rows):
        raise ValueError("Source pair IDs are not unique")

    buckets: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (str(row["clean_candidate"]), str(row["corrupt_candidate"]))
        buckets[key].append(row)
    if set(buckets) != set(HELDOUT_QUOTAS):
        raise ValueError(f"Unexpected verb-pair strata: {sorted(buckets)}")
    for key, quota in HELDOUT_QUOTAS.items():
        if len(buckets[key]) < quota:
            raise ValueError(f"{key}: quota {quota} exceeds {len(buckets[key])} candidates")
    return buckets


def copy_pair_files(row: dict[str, Any], split: str, target_pair_root: Path) -> dict[str, str]:
    source_id = str(row["source_sample_id"])
    filename = f"{source_id}.txt"
    result: dict[str, str] = {}
    for condition, source_key in (("clean", "clean_relpath"), ("corrupt", "corrupt_relpath")):
        source = SOURCE_PAIR_ROOT / str(row[source_key])
        target = target_pair_root / split / condition / filename
        if not source.is_file():
            raise FileNotFoundError(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        result[condition] = str(target.relative_to(target_pair_root))
    return result


def pair_record(row: dict[str, Any], split: str, paths: dict[str, str]) -> dict[str, Any]:
    return {
        "sample_id": f"qwen3_8b_{row['source_sample_id']}",
        "source_sample_id": row["source_sample_id"],
        "split": split,
        "clean_verb": row["clean_candidate"],
        "corrupt_verb": row["corrupt_candidate"],
        "source_dataset": row["source_dataset"],
        "source_language": row["source_language"],
        "clean_relpath": paths["clean"],
        "corrupt_relpath": paths["corrupt"],
        "clean_prompt_sha256": row["clean_prompt_sha256"],
        "corrupt_prompt_sha256": row["corrupt_prompt_sha256"],
    }


def build_pairs() -> dict[str, Any]:
    rows = read_jsonl(SOURCE_PAIR_ROOT / "manifest.jsonl")
    buckets = check_quotas(rows)
    target_pair_root = TARGET_ROOT / "pair"
    outputs: list[dict[str, Any]] = []
    balance_rows: list[dict[str, Any]] = []
    for clean_verb in EXECUTION_VERBS:
        for corrupt_verb in ANALYSIS_VERBS:
            key = (clean_verb, corrupt_verb)
            ordered = sorted(buckets[key], key=stable_rank)
            heldout_n = HELDOUT_QUOTAS[key]
            split_rows = {
                "heldout": ordered[:heldout_n],
                "train": ordered[heldout_n:],
            }
            balance_rows.append(
                {
                    "clean_verb": clean_verb,
                    "corrupt_verb": corrupt_verb,
                    "all_pairs": len(ordered),
                    "train_pairs": len(split_rows["train"]),
                    "heldout_pairs": len(split_rows["heldout"]),
                }
            )
            for split, members in split_rows.items():
                for row in members:
                    paths = copy_pair_files(row, split, target_pair_root)
                    outputs.append(pair_record(row, split, paths))

    outputs.sort(key=lambda row: (row["split"], row["sample_id"]))
    write_jsonl(target_pair_root / "pairs.jsonl", outputs)
    with (target_pair_root / "balance.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(balance_rows[0]))
        writer.writeheader()
        writer.writerows(balance_rows)

    split_counts = Counter(row["split"] for row in outputs)
    clean_counts = {
        split: Counter(row["clean_verb"] for row in outputs if row["split"] == split)
        for split in ("train", "heldout")
    }
    corrupt_counts = {
        split: Counter(row["corrupt_verb"] for row in outputs if row["split"] == split)
        for split in ("train", "heldout")
    }
    if split_counts != {"train": 300, "heldout": 200}:
        raise ValueError(f"Unexpected split counts: {dict(split_counts)}")
    if set(clean_counts["train"].values()) != {60} or set(clean_counts["heldout"].values()) != {40}:
        raise ValueError(f"Unexpected clean-verb balance: {clean_counts}")
    if set(corrupt_counts["train"].values()) != {75} or set(corrupt_counts["heldout"].values()) != {50}:
        raise ValueError(f"Unexpected corrupt-verb balance: {corrupt_counts}")
    return {
        "pairs_jsonl_sha256": sha256_file(target_pair_root / "pairs.jsonl"),
        "balance_csv_sha256": sha256_file(target_pair_root / "balance.csv"),
        "split_counts": dict(split_counts),
        "clean_verb_counts": {split: dict(counts) for split, counts in clean_counts.items()},
        "corrupt_verb_counts": {split: dict(counts) for split, counts in corrupt_counts.items()},
    }


def strip_verb_free_annotations(row: dict[str, Any]) -> dict[str, Any]:
    keep = ("item_id", "domain", "pattern", "source_id", "text", "prompt")
    return {key: row[key] for key in keep if key in row}


def build_verb_free() -> dict[str, Any]:
    rows = [strip_verb_free_annotations(row) for row in read_jsonl(SOURCE_VERB_FREE)]
    if len(rows) != 600:
        raise ValueError(f"Expected 600 verb-free requests, found {len(rows)}")
    target = TARGET_ROOT / "verb_free"
    target.mkdir(parents=True, exist_ok=True)
    write_jsonl(target / "requests.jsonl", rows)
    payload = {
        "schema_version": 1,
        "model_key": "qwen3_8b",
        "n_requests": len(rows),
        "source_sha256": sha256_file(SOURCE_VERB_FREE),
        "output_sha256": sha256_file(target / "requests.jsonl"),
        "selection_rule": "Run the current model baseline first; freeze the native-call arm only after that run.",
    }
    (target / "manifest.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload


def strip_tau2_baseline(row: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if key != "tau2_200_baseline"}


def build_tau2() -> dict[str, Any]:
    target = TARGET_ROOT / "tau2_bench"
    target.mkdir(parents=True, exist_ok=True)
    source_files = {
        "native-call-200.jsonl": SOURCE_TAU2_ROOT / "selected_tool_200.jsonl",
        "native-text-200.jsonl": SOURCE_TAU2_ROOT / "selected_direct_200.jsonl",
    }
    records: dict[str, Any] = {}
    for target_name, source in source_files.items():
        rows = [strip_tau2_baseline(row) for row in read_jsonl(source)]
        if len(rows) != 200:
            raise ValueError(f"Expected 200 rows in {source}, found {len(rows)}")
        destination = target / target_name
        write_jsonl(destination, rows)
        records[target_name] = {
            "n": len(rows),
            "source_sha256": sha256_file(source),
            "output_sha256": sha256_file(destination),
        }
    payload = {
        "schema_version": 1,
        "model_key": "qwen3_8b",
        "source_domain": "tau2_telecom",
        "candidate_sets": records,
        "selection_rule": "Re-run the current model baseline on both frozen candidate sets before intervention; do not estimate the direction on tau2.",
    }
    (target / "manifest.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload


def main() -> int:
    # native/ and verb_free/implicit_intent.jsonl live inside TARGET_ROOT and are
    # inputs. Deleting the model directory to get past this check deletes them.
    if TARGET_ROOT.exists():
        raise FileExistsError(f"Refusing to overwrite existing target: {TARGET_ROOT}")
    pair = build_pairs()
    verb_free = build_verb_free()
    tau2 = build_tau2()
    manifest = {
        "schema_version": 1,
        "model_key": "qwen3_8b",
        "model_label": "Qwen3-8B",
        "pair": pair,
        "verb_free": verb_free,
        "tau2": tau2,
        "pair_split_seed": SPLIT_SEED,
        "pair_heldout_quotas": [
            {"clean_verb": clean, "corrupt_verb": corrupt, "heldout_pairs": HELDOUT_QUOTAS[(clean, corrupt)]}
            for clean in EXECUTION_VERBS
            for corrupt in ANALYSIS_VERBS
        ],
    }
    (TARGET_ROOT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Built {TARGET_ROOT}")
    print(json.dumps({"pair": pair, "verb_free": verb_free["n_requests"], "tau2": {key: value["n"] for key, value in tau2["candidate_sets"].items()}}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
