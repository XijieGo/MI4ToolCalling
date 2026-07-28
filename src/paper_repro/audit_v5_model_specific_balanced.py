#!/usr/bin/env python3
"""Independently validate the complete v5 model-specific dataset release."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = PROJECT_ROOT / "datasets" / "v5_model_specific_balanced"
EXPECTED_MODELS = (
    "qwen3_4b",
    "qwen3_8b",
    "qwen3_14b",
    "qwen35_4b",
    "qwen35_9b",
    "mistral_3p2_24b",
    "granite_3p3_8b",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_ids(ids: list[int]) -> str:
    payload = json.dumps(ids, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def balanced(counts: Counter[str]) -> bool:
    values = list(counts.values())
    return bool(values) and max(values) - min(values) <= 1


def audit_model(root: Path, key: str) -> dict[str, Any]:
    directory = root / key
    summary_path = directory / "summary.json"
    manifest_path = directory / "manifest.jsonl"
    if not summary_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(f"{key}: missing summary or manifest")
    summary = read_json(summary_path)
    rows = read_jsonl(manifest_path)
    if summary.get("model_key") != key:
        raise AssertionError(f"{key}: summary model_key mismatch")
    if len(rows) != 500 or summary.get("n_pairs") != 500:
        raise AssertionError(f"{key}: expected 500 pairs")
    split_counts = Counter(str(row.get("split")) for row in rows)
    if split_counts != Counter({"train": 200, "heldout": 300}):
        raise AssertionError(f"{key}: invalid split sizes {dict(split_counts)}")
    sample_ids = [str(row.get("sample_id")) for row in rows]
    source_ids = [str(row.get("source_sample_id")) for row in rows]
    if len(sample_ids) != len(set(sample_ids)):
        raise AssertionError(f"{key}: duplicate v5 sample IDs")
    if len(source_ids) != len(set(source_ids)):
        raise AssertionError(f"{key}: a source state appears in both selections")
    if key in {"qwen35_4b", "mistral_3p2_24b"}:
        task_ids = [str(row.get("source_task_id")) for row in rows]
        if len(task_ids) != len(set(task_ids)):
            raise AssertionError(f"{key}: a TAU2 task appears in both selections")
    split_report: dict[str, Any] = {}
    for split, expected in (("train", 200), ("heldout", 300)):
        selected = [row for row in rows if row["split"] == split]
        if len(selected) != expected:
            raise AssertionError(f"{key}/{split}: size mismatch")
        clean_counts = Counter(str(row["clean_candidate"]) for row in selected)
        corrupt_counts = Counter(str(row["corrupt_candidate"]) for row in selected)
        if not balanced(clean_counts) or not balanced(corrupt_counts):
            raise AssertionError(
                f"{key}/{split}: unbalanced verbs clean={dict(clean_counts)} corrupt={dict(corrupt_counts)}"
            )
        if not all(bool(row.get("clean_is_tool_top1")) and not bool(row.get("corrupt_is_tool_top1")) for row in selected):
            raise AssertionError(f"{key}/{split}: contains a behavior-invalid pair")
        split_report[split] = {
            "n_pairs": len(selected),
            "clean_verb_counts": dict(sorted(clean_counts.items())),
            "corrupt_verb_counts": dict(sorted(corrupt_counts.items())),
        }
    for row in rows:
        clean_path = directory / str(row["clean_relpath"])
        corrupt_path = directory / str(row["corrupt_relpath"])
        if not clean_path.is_file() or not corrupt_path.is_file():
            raise FileNotFoundError(f"{key}: missing prompt file for {row['sample_id']}")
        if row.get("prompt_format") == "mistral_native_input_ids":
            clean_payload = read_json(clean_path)
            corrupt_payload = read_json(corrupt_path)
            clean_ids = clean_payload.get("input_ids")
            corrupt_ids = corrupt_payload.get("input_ids")
            if not isinstance(clean_ids, list) or not isinstance(corrupt_ids, list):
                raise AssertionError(f"{key}: native-ID payload missing IDs for {row['sample_id']}")
            if sha256_ids(clean_ids) != str(row["clean_prompt_sha256"]):
                raise AssertionError(f"{key}: clean native-ID hash mismatch for {row['sample_id']}")
            if sha256_ids(corrupt_ids) != str(row["corrupt_prompt_sha256"]):
                raise AssertionError(f"{key}: corrupt native-ID hash mismatch for {row['sample_id']}")
            if clean_ids == corrupt_ids:
                raise AssertionError(f"{key}: identical clean/corrupt native IDs for {row['sample_id']}")
        else:
            if sha256_file(clean_path) != str(row["clean_prompt_sha256"]):
                raise AssertionError(f"{key}: clean prompt hash mismatch for {row['sample_id']}")
            if sha256_file(corrupt_path) != str(row["corrupt_prompt_sha256"]):
                raise AssertionError(f"{key}: corrupt prompt hash mismatch for {row['sample_id']}")
            if clean_path.read_bytes() == corrupt_path.read_bytes():
                raise AssertionError(f"{key}: identical clean/corrupt prompt for {row['sample_id']}")
    return {
        "model_label": summary.get("model_label"),
        "n_pairs": len(rows),
        "n_train": split_counts["train"],
        "n_heldout": split_counts["heldout"],
        "selected_clean_verbs": summary.get("selected_clean_verbs"),
        "selected_corrupt_verbs": summary.get("selected_corrupt_verbs"),
        "splits": split_report,
        "behavior_rule_verified_from_manifest": "clean top-1 tool call; corrupt not top-1 tool call",
        "prompt_hashes_verified": True,
        "source_states_unique": True,
    }


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    report = {
        "dataset_version": "v5_model_specific_balanced",
        "root": str(root),
        "models": {key: audit_model(root, key) for key in EXPECTED_MODELS},
        "status": "passed",
    }
    output = args.output.resolve() if args.output else root / "audit.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "passed", "models": list(report["models"])}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
