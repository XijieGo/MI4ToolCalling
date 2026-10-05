#!/usr/bin/env python3
"""Check that the shared loaders can read the datasets in this repo."""

from __future__ import annotations

import csv
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from mi4tc.io import iter_jsonl, load_json, sha256_file  # noqa: E402
from mi4tc.pairs import load_model_native_pairs, validate_pairs  # noqa: E402


ERRORS: list[str] = []


def fail(message: str) -> None:
    ERRORS.append(message)


def dataset_child(root: Path, relative: object, label: str) -> Path | None:
    candidate = Path(str(relative))
    if candidate.is_absolute() or ".." in candidate.parts:
        fail(f"non-portable {label}: {relative}")
        return None
    return root / candidate


def check_controlled_pairs() -> None:
    root = REPO_ROOT / "datasets/qwen3_8b/controlled"
    try:
        report = validate_pairs(root)
        if report["splits"]["train"]["pairs"] != 1200 or report["splits"]["test"]["pairs"] != 300:
            fail(f"unexpected controlled-pair counts: {report}")
    except Exception as exc:
        fail(f"controlled-pair validation failed: {exc}")
        return

    dataset = load_json(root / "dataset.json")
    for relative, expected in dataset["frozen_files_sha256"].items():
        path = root / relative
        if not path.is_file():
            fail(f"controlled-pair hash target missing: {relative}")
        elif sha256_file(path) != expected:
            fail(f"controlled-pair hash mismatch: {relative}")


def check_model_native_pairs() -> None:
    expected = {
        "qwen3_4b": REPO_ROOT / "datasets/qwen3_4b/pair",
        "qwen3_8b": REPO_ROOT / "datasets/qwen3_8b/native",
        "qwen3_14b": REPO_ROOT / "datasets/qwen3_14b/pair",
        "qwen35_4b": REPO_ROOT / "datasets/qwen35_4b/pair",
        "qwen35_9b": REPO_ROOT / "datasets/qwen35_9b/pair",
        "mistral_3p2_24b": REPO_ROOT / "datasets/mistral_3p2_24b/pair",
        "granite_3p3_8b": REPO_ROOT / "datasets/granite_3p3_8b/pair",
    }
    for model_key, model_dir in expected.items():
        try:
            collection = load_model_native_pairs(model_dir)
            if collection.model_key != model_key:
                fail(f"{model_key}: loaded model_key {collection.model_key}")
            if len(collection.train) != 200 or len(collection.heldout) != 300:
                fail(
                    f"{model_key}: unexpected split counts "
                    f"train={len(collection.train)} heldout={len(collection.heldout)}"
                )
        except Exception as exc:
            fail(f"{model_key}: native pair validation failed: {exc}")


def check_qwen3_8b_rerun() -> None:
    root = REPO_ROOT / "datasets/qwen3_8b"
    pair_root = root / "pair"
    pairs_path = pair_root / "pairs.jsonl"
    balance_path = pair_root / "balance.csv"
    manifest_path = root / "manifest.json"
    for path in (pairs_path, balance_path, manifest_path):
        if not path.is_file():
            fail(f"missing Qwen3-8B rerun input: {path.relative_to(REPO_ROOT)}")
            return
    rows = list(iter_jsonl(pairs_path))
    if len(rows) != 500:
        fail(f"Qwen3-8B rerun pairs: expected 500 rows, found {len(rows)}")
        return
    counts = {"train": 0, "heldout": 0}
    clean_by_split: dict[str, dict[str, int]] = {"train": {}, "heldout": {}}
    corrupt_by_split: dict[str, dict[str, int]] = {"train": {}, "heldout": {}}
    for row in rows:
        split = str(row.get("split"))
        if split not in counts:
            fail(f"Qwen3-8B rerun pairs: invalid split {split!r}")
            continue
        counts[split] += 1
        clean = str(row.get("clean_verb"))
        corrupt = str(row.get("corrupt_verb"))
        clean_by_split[split][clean] = clean_by_split[split].get(clean, 0) + 1
        corrupt_by_split[split][corrupt] = corrupt_by_split[split].get(corrupt, 0) + 1
        for key in ("clean_relpath", "corrupt_relpath"):
            relative = row.get(key)
            path = dataset_child(pair_root, relative, f"Qwen3-8B rerun {key}")
            if path is None or not path.is_file():
                fail(f"Qwen3-8B rerun pairs: missing {relative}")
                continue
            expected = row.get("clean_prompt_sha256" if key == "clean_relpath" else "corrupt_prompt_sha256")
            if expected and sha256_file(path) != expected:
                fail(f"Qwen3-8B rerun pairs: hash mismatch for {relative}")
    if counts != {"train": 300, "heldout": 200}:
        fail(f"Qwen3-8B rerun pairs: unexpected split counts {counts}")
    if sorted(clean_by_split["train"].values()) != [60] * 5 or sorted(clean_by_split["heldout"].values()) != [20, 20, 53, 53, 54]:
        fail(f"Qwen3-8B rerun pairs: clean-verb balance is {clean_by_split}")
    if sorted(corrupt_by_split["train"].values()) != [75] * 4 or sorted(corrupt_by_split["heldout"].values()) != [40] * 5:
        fail(f"Qwen3-8B rerun pairs: corrupt-verb balance is {corrupt_by_split}")
    with balance_path.open(encoding="utf-8", newline="") as handle:
        balance = list(csv.DictReader(handle))
    if len(balance) != 25 or sum(int(row["heldout_pairs"]) for row in balance) != 200:
        fail("Qwen3-8B rerun pairs: invalid verb-stratum balance table")
    root_manifest = load_json(manifest_path)
    pair_manifest = root_manifest.get("pair", {})
    if pair_manifest.get("pairs_jsonl_sha256") != sha256_file(pairs_path):
        fail("Qwen3-8B rerun pairs: manifest hash mismatch for pairs.jsonl")
    if pair_manifest.get("balance_csv_sha256") != sha256_file(balance_path):
        fail("Qwen3-8B rerun pairs: manifest hash mismatch for balance.csv")

    verb_free = root / "verb_free/requests.jsonl"
    if not verb_free.is_file() or len(list(iter_jsonl(verb_free))) != 600:
        fail("Qwen3-8B verb-free input must contain 600 requests")
    elif any(any(key.startswith("baseline_") for key in row) for row in iter_jsonl(verb_free)):
        fail("Qwen3-8B verb-free input contains historical baseline outputs")
    for filename in ("native-call-200.jsonl", "native-text-200.jsonl"):
        path = root / "tau2_bench" / filename
        if not path.is_file() or len(list(iter_jsonl(path))) != 200:
            fail(f"Qwen3-8B tau2 input {filename} must contain 200 requests")
        elif any("tau2_200_baseline" in row for row in iter_jsonl(path)):
            fail(f"Qwen3-8B tau2 input {filename} contains historical baseline outputs")


def check_multi_domain() -> None:
    root = REPO_ROOT / "datasets/qwen3_8b/multi_domain"
    expected_domains = {"code", "retrieval", "operations", "communication"}
    actual_domains = {path.name for path in root.iterdir() if path.is_dir()}
    if actual_domains != expected_domains:
        fail(f"unexpected multi-domain directories: {sorted(actual_domains)}")
        return

    code = root / "code"
    for split, expected_count in (("train", 400), ("test", 100)):
        clean = list(iter_jsonl(code / split / "clean" / "manifest.jsonl"))
        corrupt = list(iter_jsonl(code / split / "corrupt" / "manifest.jsonl"))
        if len(clean) != expected_count or len(corrupt) != expected_count:
            fail(f"multi-domain code {split}: expected {expected_count} clean/corrupt rows")
            continue
        for clean_row, corrupt_row in zip(clean, corrupt):
            clean_name = clean_row.get("output_filename") or clean_row.get("filename")
            corrupt_name = corrupt_row.get("output_filename") or corrupt_row.get("filename")
            if not clean_name or clean_name != corrupt_name:
                fail(f"multi-domain code {split}: clean/corrupt manifest pairing mismatch")
                continue
            for condition in ("clean", "corrupt"):
                path = dataset_child(code / split / condition, clean_name, "multi-domain code filename")
                if path is not None and not path.is_file():
                    fail(f"multi-domain code {split}: missing {condition}/{clean_name}")

    for domain in sorted(expected_domains - {"code"}):
        selected = root / domain / "selected_pairs.jsonl"
        rows = list(iter_jsonl(selected))
        if len(rows) != 500:
            fail(f"multi-domain {domain}: expected 500 selected rows, found {len(rows)}")


def main() -> int:
    check_controlled_pairs()
    check_multi_domain()
    check_model_native_pairs()
    check_qwen3_8b_rerun()
    if ERRORS:
        print("Input check failed:")
        for error in ERRORS:
            print(f"- {error}")
        return 1
    print("Input check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
