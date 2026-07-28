from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "results" / "section6_generalization" / "devstral_2_24b" / "datasets"


@dataclass(frozen=True)
class PairExample:
    order: int
    pair_id: int
    sample_id: str
    split: str
    language: str
    clean_path: Path
    corrupt_path: Path
    clean_text: str
    corrupt_text: str
    clean_input_ids_cpu: torch.Tensor
    corrupt_input_ids_cpu: torch.Tensor
    clean_len: int
    corrupt_len: int


@dataclass(frozen=True)
class PairBatch:
    indices: list[int]
    examples: list[PairExample]
    max_len: int


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_manifest_rows(dataset_root: Path, split: str | None = None) -> list[dict[str, Any]]:
    manifest_path = dataset_root / "manifest.jsonl"
    rows = read_jsonl(manifest_path)
    if split is not None and split != "all":
        rows = [row for row in rows if str(row.get("split")) == split]
    rows.sort(key=lambda row: (int(row.get("pair_id", 10**9)), str(row.get("sample_id", ""))))
    return rows


def resolve_prompt_path(raw_path: str, dataset_root: Path, side: str, pair_id: int) -> Path:
    path = Path(raw_path)
    if path.exists():
        return path
    name = path.name or f"{side}_{pair_id}.txt"
    candidates = [
        dataset_root / side / name,
        dataset_root / side / f"{side}_{pair_id}.txt",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[-1]


def load_pairs(
    tokenizer,
    *,
    dataset_root: Path = DEFAULT_DATASET_ROOT,
    split: str | None = None,
    max_pairs: int = 0,
) -> list[PairExample]:
    rows = load_manifest_rows(dataset_root, split)
    pairs: list[PairExample] = []
    for order, row in enumerate(rows, start=1):
        pair_id = int(row.get("pair_id", order))
        clean_path = resolve_prompt_path(str(row["clean_path"]), dataset_root, "clean", pair_id)
        corrupt_path = resolve_prompt_path(str(row["corrupt_path"]), dataset_root, "corrupt", pair_id)
        if not clean_path.exists():
            raise FileNotFoundError(clean_path)
        if not corrupt_path.exists():
            raise FileNotFoundError(corrupt_path)

        clean_text = clean_path.read_text(encoding="utf-8")
        corrupt_text = corrupt_path.read_text(encoding="utf-8")
        clean_ids = tokenizer(clean_text, add_special_tokens=False, return_tensors="pt")["input_ids"][0].detach().cpu()
        corrupt_ids = tokenizer(corrupt_text, add_special_tokens=False, return_tensors="pt")["input_ids"][0].detach().cpu()
        pair = PairExample(
            order=order,
            pair_id=pair_id,
            sample_id=str(row["sample_id"]),
            split=str(row["split"]),
            language=str(row["language"]),
            clean_path=clean_path,
            corrupt_path=corrupt_path,
            clean_text=clean_text,
            corrupt_text=corrupt_text,
            clean_input_ids_cpu=clean_ids,
            corrupt_input_ids_cpu=corrupt_ids,
            clean_len=int(clean_ids.numel()),
            corrupt_len=int(corrupt_ids.numel()),
        )
        pairs.append(pair)
        if max_pairs > 0 and len(pairs) >= max_pairs:
            break
    if not pairs:
        raise RuntimeError(f"No pairs found in {dataset_root}")
    return pairs


def build_pair_batches(pairs: Sequence[PairExample], batch_size: int) -> list[PairBatch]:
    ordered = sorted(
        enumerate(pairs),
        key=lambda item: (max(item[1].clean_len, item[1].corrupt_len), item[1].pair_id, item[1].sample_id),
    )
    batches: list[PairBatch] = []
    step = max(int(batch_size), 1)
    for start in range(0, len(ordered), step):
        chunk = ordered[start : start + step]
        batches.append(
            PairBatch(
                indices=[idx for idx, _pair in chunk],
                examples=[pair for _idx, pair in chunk],
                max_len=max(max(pair.clean_len, pair.corrupt_len) for _idx, pair in chunk),
            )
        )
    return batches


def collate_pair_side(tokenizer, examples: Sequence[PairExample], side: str) -> dict[str, torch.Tensor]:
    if side not in {"clean", "corrupt"}:
        raise ValueError(f"Unknown side: {side}")
    ids = [
        example.clean_input_ids_cpu if side == "clean" else example.corrupt_input_ids_cpu
        for example in examples
    ]
    batch = tokenizer.pad(
        {
            "input_ids": ids,
            "attention_mask": [torch.ones_like(item, dtype=torch.long) for item in ids],
        },
        padding=True,
        return_tensors="pt",
    )
    return batch


def collate_pair_batch(tokenizer, batch: PairBatch) -> dict[str, torch.Tensor]:
    clean = collate_pair_side(tokenizer, batch.examples, "clean")
    corrupt = collate_pair_side(tokenizer, batch.examples, "corrupt")
    return {
        "clean_input_ids": clean["input_ids"],
        "clean_attention_mask": clean["attention_mask"],
        "corrupt_input_ids": corrupt["input_ids"],
        "corrupt_attention_mask": corrupt["attention_mask"],
    }
