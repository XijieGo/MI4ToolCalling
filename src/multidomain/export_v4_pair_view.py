#!/usr/bin/env python3
"""Create a read-only legacy-layout view of the released v4 paired prompts.

The legacy Qwen3 mechanism runners consume
``<root>/<split>/{clean,corrupt}/manifest.jsonl``.  D1 in v4 already has
that layout, whereas D3--D5 record their pair metadata in
``selected_pairs.jsonl`` and ``selection_manifest.csv``.  This script creates
only manifests and relative symlinks in a caller-specified *new* output
directory; the frozen v4 data are never modified or copied.

Every exported pair is checked with the supplied tokenizer to have equal
length and exactly one differing input token.  Consequently the resulting
view can be passed directly to the existing residual-vector and intervention
runners without changing their causal logic.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from transformers import AutoTokenizer


DOMAINS = ("D1", "D3", "D4", "D5")


@dataclass(frozen=True)
class PairRecord:
    domain: str
    split: str
    filename: str
    candidate_id: str
    clean_verb: str
    corrupt_verb: str
    clean_source: Path
    corrupt_source: Path
    clean_text: str
    corrupt_text: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a non-mutating manifest/symlink view of v4 multi-domain paired prompts."
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--tokenizer-path", type=Path, required=True)
    parser.add_argument("--domains", nargs="+", choices=DOMAINS, default=list(DOMAINS))
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def v4_d1_pairs(domain_root: Path, split: str) -> list[PairRecord]:
    clean_root = domain_root / split / "clean"
    corrupt_root = domain_root / split / "corrupt"
    clean_rows = read_jsonl(clean_root / "manifest.jsonl")
    corrupt_rows = {
        str(row["output_filename"]): row
        for row in read_jsonl(corrupt_root / "manifest.jsonl")
    }
    records: list[PairRecord] = []
    for clean_row in clean_rows:
        filename = str(clean_row["output_filename"])
        corrupt_row = corrupt_rows.get(filename)
        if corrupt_row is None:
            raise ValueError(f"D1 {split}: missing corrupt manifest row for {filename}")
        clean_path = clean_root / filename
        corrupt_path = corrupt_root / filename
        records.append(
            PairRecord(
                domain="D1",
                split=split,
                filename=filename,
                candidate_id=str(clean_row.get("sample_id", Path(filename).stem)),
                clean_verb=str(clean_row["clean_candidate"]),
                corrupt_verb=str(corrupt_row["corrupt_candidate"]),
                clean_source=clean_path,
                corrupt_source=corrupt_path,
                clean_text=clean_path.read_text(encoding="utf-8"),
                corrupt_text=corrupt_path.read_text(encoding="utf-8"),
            )
        )
    return records


def v4_selected_pair_records(domain: str, domain_root: Path, split: str) -> list[PairRecord]:
    manifest_rows = read_csv_rows(domain_root / "selection_manifest.csv")
    filename_by_candidate = {str(row["candidate_id"]): str(row["filename"]) for row in manifest_rows}
    records: list[PairRecord] = []
    for row in read_jsonl(domain_root / "selected_pairs.jsonl"):
        if str(row["split"]) != split:
            continue
        candidate_id = str(row["candidate_id"])
        filename = filename_by_candidate.get(candidate_id)
        if filename is None:
            raise ValueError(f"{domain} {split}: no filename for candidate {candidate_id}")
        clean_path = domain_root / split / "clean" / filename
        corrupt_path = domain_root / split / "corrupt" / filename
        clean_text = clean_path.read_text(encoding="utf-8")
        corrupt_text = corrupt_path.read_text(encoding="utf-8")
        if clean_text != str(row["clean_prompt"]) or corrupt_text != str(row["corrupt_prompt"]):
            raise ValueError(f"{domain} {split}: rendered prompt differs from selected_pairs.jsonl for {candidate_id}")
        records.append(
            PairRecord(
                domain=domain,
                split=split,
                filename=filename,
                candidate_id=candidate_id,
                clean_verb=str(row["clean_verb"]),
                corrupt_verb=str(row["corrupt_verb"]),
                clean_source=clean_path,
                corrupt_source=corrupt_path,
                clean_text=clean_text,
                corrupt_text=corrupt_text,
            )
        )
    return records


def domain_pairs(dataset_root: Path, domain: str, split: str) -> list[PairRecord]:
    domain_root = dataset_root / domain
    if domain == "D1":
        return v4_d1_pairs(domain_root, split)
    return v4_selected_pair_records(domain, domain_root, split)


def token_ids(tokenizer: Any, text: str) -> list[int]:
    encoded = tokenizer(text, add_special_tokens=False)
    return [int(token_id) for token_id in encoded["input_ids"]]


def validate_pair(record: PairRecord, tokenizer: Any) -> dict[str, int]:
    if not record.clean_source.is_file() or not record.corrupt_source.is_file():
        raise FileNotFoundError(f"Missing prompt file for {record.domain}/{record.split}/{record.filename}")
    clean_ids = token_ids(tokenizer, record.clean_text)
    corrupt_ids = token_ids(tokenizer, record.corrupt_text)
    if len(clean_ids) != len(corrupt_ids):
        raise ValueError(
            f"{record.domain}/{record.split}/{record.filename}: token length differs "
            f"({len(clean_ids)} vs {len(corrupt_ids)})"
        )
    differing = [idx for idx, (clean_id, corrupt_id) in enumerate(zip(clean_ids, corrupt_ids)) if clean_id != corrupt_id]
    if len(differing) != 1:
        raise ValueError(
            f"{record.domain}/{record.split}/{record.filename}: expected one differing token, found {len(differing)}"
        )
    return {"token_length": len(clean_ids), "differing_token_position": differing[0]}


def relative_symlink(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or os.path.lexists(destination):
        raise FileExistsError(f"Refusing to replace existing output: {destination}")
    destination.symlink_to(os.path.relpath(source, start=destination.parent))


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def export_split(output_domain_root: Path, records: list[PairRecord], tokenizer: Any) -> dict[str, Any]:
    if not records:
        raise ValueError(f"No records supplied for {output_domain_root}")
    split = records[0].split
    clean_rows: list[dict[str, Any]] = []
    corrupt_rows: list[dict[str, Any]] = []
    lengths: list[int] = []
    positions: list[int] = []
    for record in sorted(records, key=lambda item: item.filename):
        check = validate_pair(record, tokenizer)
        lengths.append(check["token_length"])
        positions.append(check["differing_token_position"])
        relative_symlink(record.clean_source, output_domain_root / split / "clean" / record.filename)
        relative_symlink(record.corrupt_source, output_domain_root / split / "corrupt" / record.filename)
        common = {
            "output_filename": record.filename,
            "source_filename": record.filename,
            "sample_id": Path(record.filename).stem,
            "candidate_id": record.candidate_id,
            "domain": record.domain,
            "split": split,
            "token_length": check["token_length"],
            "differing_token_position": check["differing_token_position"],
        }
        clean_rows.append(
            {
                **common,
                "clean_candidate": record.clean_verb,
                "corrupt_candidate": record.corrupt_verb,
                "prompt_sha256": sha256_text(record.clean_text),
            }
        )
        corrupt_rows.append(
            {
                **common,
                "clean_candidate": record.clean_verb,
                "corrupt_candidate": record.corrupt_verb,
                "prompt_sha256": sha256_text(record.corrupt_text),
            }
        )
    clean_manifest = output_domain_root / split / "clean" / "manifest.jsonl"
    corrupt_manifest = output_domain_root / split / "corrupt" / "manifest.jsonl"
    write_jsonl(clean_manifest, clean_rows)
    write_jsonl(corrupt_manifest, corrupt_rows)
    return {
        "split": split,
        "pair_count": len(records),
        "token_length_min": min(lengths),
        "token_length_max": max(lengths),
        "unique_differing_token_positions": sorted(set(positions)),
        "clean_manifest": str(clean_manifest),
        "corrupt_manifest": str(corrupt_manifest),
    }


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    output_root = args.output_root.resolve()
    if output_root.exists() or os.path.lexists(output_root):
        raise FileExistsError(f"Output root already exists: {output_root}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)
    output_root.mkdir(parents=True, exist_ok=False)
    metadata: dict[str, Any] = {
        "source_dataset_root": str(dataset_root),
        "tokenizer_path": str(args.tokenizer_path.resolve()),
        "domains": {},
        "non_mutating": True,
        "prompt_files": "relative symlinks to frozen v4 files",
    }
    for domain in args.domains:
        domain_metadata = {
            split: export_split(output_root / domain, domain_pairs(dataset_root, domain, split), tokenizer)
            for split in ("train", "test")
        }
        metadata["domains"][domain] = domain_metadata
        print(
            f"{domain}: train={domain_metadata['train']['pair_count']} "
            f"test={domain_metadata['test']['pair_count']} "
            f"tokens={domain_metadata['train']['token_length_min']}..{domain_metadata['train']['token_length_max']}",
            flush=True,
        )
    (output_root / "view_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
