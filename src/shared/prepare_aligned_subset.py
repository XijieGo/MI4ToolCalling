#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from transformers import AutoTokenizer


def load_manifest_rows(manifest_path: Path) -> Dict[str, Dict[str, object]]:
    rows: Dict[str, Dict[str, object]] = {}
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            filename = str(row.get("output_filename") or row.get("source_filename") or "")
            if filename:
                rows[filename] = row
    return rows


def configure_tokenizer(model_path: str):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.bos_token is None:
        tokenizer.bos_token = "<|endoftext|>"
    tokenizer.add_bos_token = True
    return tokenizer


def token_length(tokenizer, text: str) -> int:
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def reset_output_root(output_root: Path) -> None:
    if output_root.exists():
        shutil.rmtree(output_root)
    (output_root / "clean").mkdir(parents=True, exist_ok=True)
    (output_root / "corrupt").mkdir(parents=True, exist_ok=True)


def write_manifest(rows: Iterable[Dict[str, object]], out_path: Path) -> None:
    with out_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def symlink_file(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    dst.symlink_to(src)


def analyze_split(
    split_root: Path,
    *,
    tokenizers: List[Tuple[str, object]],
    output_root: Path,
    mismatch_examples_limit: int,
) -> Dict[str, object]:
    clean_root = split_root / "clean"
    corrupt_root = split_root / "corrupt"
    if not clean_root.exists() or not corrupt_root.exists():
        raise FileNotFoundError(f"Expected clean/ and corrupt/ under {split_root}")

    clean_manifest = load_manifest_rows(clean_root / "manifest.jsonl")
    corrupt_manifest = load_manifest_rows(corrupt_root / "manifest.jsonl")
    filenames = sorted(set(clean_manifest) & set(corrupt_manifest))
    if not filenames:
        raise ValueError(f"No shared clean/corrupt filenames found under {split_root}")

    reset_output_root(output_root)

    kept_rows_clean: List[Dict[str, object]] = []
    kept_rows_corrupt: List[Dict[str, object]] = []
    kept_sample_ids: List[str] = []
    mismatch_examples: List[Dict[str, object]] = []

    for filename in filenames:
        clean_path = clean_root / filename
        corrupt_path = corrupt_root / filename
        clean_text = clean_path.read_text(encoding="utf-8")
        corrupt_text = corrupt_path.read_text(encoding="utf-8")

        per_model_lengths: Dict[str, Dict[str, int]] = {}
        aligned = True
        for model_name, tokenizer in tokenizers:
            clean_len = token_length(tokenizer, clean_text)
            corrupt_len = token_length(tokenizer, corrupt_text)
            per_model_lengths[model_name] = {
                "clean_len": clean_len,
                "corrupt_len": corrupt_len,
            }
            if clean_len != corrupt_len:
                aligned = False

        if not aligned:
            if len(mismatch_examples) < mismatch_examples_limit:
                mismatch_examples.append(
                    {
                        "filename": filename,
                        "sample_id": Path(filename).stem,
                        "lengths": per_model_lengths,
                    }
                )
            continue

        symlink_file(clean_path.resolve(), output_root / "clean" / filename)
        symlink_file(corrupt_path.resolve(), output_root / "corrupt" / filename)
        kept_rows_clean.append(clean_manifest[filename])
        kept_rows_corrupt.append(corrupt_manifest[filename])
        kept_sample_ids.append(Path(filename).stem)

    write_manifest(kept_rows_clean, output_root / "clean" / "manifest.jsonl")
    write_manifest(kept_rows_corrupt, output_root / "corrupt" / "manifest.jsonl")
    (output_root / "sample_ids.json").write_text(
        json.dumps(kept_sample_ids, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_root / "sample_ids.txt").write_text("\n".join(kept_sample_ids) + "\n", encoding="utf-8")

    clean_candidate_counts: Dict[str, int] = {}
    corrupt_candidate_counts: Dict[str, int] = {}
    language_counts: Dict[str, int] = {}
    for row in kept_rows_clean:
        clean_candidate = str(row.get("clean_candidate") or "")
        if clean_candidate:
            clean_candidate_counts[clean_candidate] = clean_candidate_counts.get(clean_candidate, 0) + 1
        language = str(row.get("language") or "")
        if language:
            language_counts[language] = language_counts.get(language, 0) + 1
    for row in kept_rows_corrupt:
        corrupt_candidate = str(row.get("assigned_candidate") or row.get("corrupt_candidate") or "")
        if corrupt_candidate:
            corrupt_candidate_counts[corrupt_candidate] = corrupt_candidate_counts.get(corrupt_candidate, 0) + 1

    return {
        "source_root": str(split_root),
        "output_root": str(output_root),
        "n_total": len(filenames),
        "n_aligned": len(kept_sample_ids),
        "aligned_ratio": (len(kept_sample_ids) / len(filenames)) if filenames else 0.0,
        "sample_ids_path": str(output_root / "sample_ids.json"),
        "language_counts": dict(sorted(language_counts.items())),
        "clean_candidate_counts": dict(sorted(clean_candidate_counts.items())),
        "corrupt_candidate_counts": dict(sorted(corrupt_candidate_counts.items())),
        "tokenizer_models": [model_name for model_name, _ in tokenizers],
        "mismatch_examples": mismatch_examples,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build token-aligned dataset subsets shared across multiple Qwen3 models.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("./datasets"),
        help="Root containing train/ and test/ splits.",
    )
    parser.add_argument(
        "--model-path",
        action="append",
        dest="model_paths",
        default=[],
        help="Model path to validate alignment against. May be passed multiple times.",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="aligned_shared_qwen3",
        help="Will create <dataset-root>/<output-prefix>/<split>/clean and corrupt.",
    )
    parser.add_argument("--mismatch-examples-limit", type=int, default=12)
    args = parser.parse_args()

    model_paths = args.model_paths or [
        "./external/models/Qwen3-1.7B",
        "./external/models/Qwen3-4B",
        "./external/models/Qwen3-8B",
    ]
    tokenizers = [(Path(model_path).name, configure_tokenizer(model_path)) for model_path in model_paths]

    dataset_root = args.dataset_root.resolve()
    output_base = dataset_root / args.output_prefix
    output_base.mkdir(parents=True, exist_ok=True)

    split_summaries: Dict[str, object] = {}
    for split_name in ["train", "test"]:
        split_root = dataset_root / split_name
        output_root = output_base / split_name
        split_summaries[split_name] = analyze_split(
            split_root,
            tokenizers=tokenizers,
            output_root=output_root,
            mismatch_examples_limit=args.mismatch_examples_limit,
        )

    summary = {
        "dataset_root": str(dataset_root),
        "output_prefix": args.output_prefix,
        "output_base": str(output_base),
        "model_paths": model_paths,
        "splits": split_summaries,
    }
    (output_base / "alignment_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
