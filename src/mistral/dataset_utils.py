from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW_DATASET_ROOT = PROJECT_ROOT / "results" / "section6_generalization" / "mistral_3p2_24b" / "datasets"
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "results" / "section6_generalization" / "mistral_3p2_24b" / "datasets"
DEFAULT_METADATA_ROOT = PROJECT_ROOT / "results" / "section6_generalization" / "mistral_3p2_24b" / "converted_dataset"
PAIR_NAME_RE = re.compile(r"^(clean|corrupt)_(\d+)\.txt$")


@dataclass(frozen=True)
class PairExample:
    order: int
    pair_id: int
    sample_id: str
    split: str
    language: str
    dataset_name: str
    clean_candidate: str
    corrupt_candidate: str
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


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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


def _load_pair_manifest(dataset_root: Path) -> dict[int, dict[str, Any]]:
    manifest_path = dataset_root / "pair_manifest.jsonl"
    if not manifest_path.exists():
        return {}
    rows = read_jsonl(manifest_path)
    output: dict[int, dict[str, Any]] = {}
    for row in rows:
        output[int(row["pair_id"])] = row
    return output


def _resolve_prompt_path(path_text: str, dataset_root: Path) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    candidate = PROJECT_ROOT / path
    if candidate.exists():
        return candidate
    candidate = dataset_root / path
    if candidate.exists():
        return candidate
    return path


def _pair_id_from_manifest_row(row: dict[str, Any], order: int) -> int:
    sample_id = str(row.get("sample_id", ""))
    match = re.fullmatch(r"pair_(\d+)", sample_id)
    if match is not None:
        return int(match.group(1))
    clean_path = str(row.get("clean_prompt_path", ""))
    match = re.search(r"clean_(\d+)\.txt", clean_path)
    if match is not None:
        return int(match.group(1))
    return int(order)


def _load_manifest_pairs(
    tokenizer,
    *,
    dataset_root: Path,
    split: str | None = None,
    max_pairs: int = 0,
) -> list[PairExample]:
    canonical_path = dataset_root / "canonical_pairs.jsonl"
    verification_path = dataset_root / "tokenizer_verification.json"
    canonical_by_sample: dict[str, dict[str, Any]] = {}
    system_prompt = None
    if canonical_path.exists():
        canonical_rows = read_jsonl(canonical_path)
        canonical_by_sample = {str(row["sample_id"]): row for row in canonical_rows}
    if verification_path.exists():
        verification = read_json(verification_path)
        system_prompt = verification.get("system_prompt_preview")
    rows = read_jsonl(dataset_root / "manifest.jsonl")
    pairs: list[PairExample] = []
    for order, row in enumerate(rows, start=1):
        pair_split = str(row.get("split", "all"))
        if split is not None and split != "all" and pair_split != split:
            continue
        clean_path = _resolve_prompt_path(str(row["clean_prompt_path"]), dataset_root)
        corrupt_path = _resolve_prompt_path(str(row["corrupt_prompt_path"]), dataset_root)
        if not clean_path.exists():
            raise FileNotFoundError(clean_path)
        if not corrupt_path.exists():
            raise FileNotFoundError(corrupt_path)
        pair_id = _pair_id_from_manifest_row(row, order)
        sample_id = str(row.get("sample_id", f"pair_{pair_id}"))
        canonical = canonical_by_sample.get(sample_id)
        if canonical is not None and system_prompt is not None:
            clean_messages = [
                {"role": "system", "content": str(system_prompt)},
                {"role": "user", "content": str(canonical["user_content_clean"])},
            ]
            corrupt_messages = [
                {"role": "system", "content": str(system_prompt)},
                {"role": "user", "content": str(canonical["user_content_corrupt"])},
            ]
            tools = canonical["tools_schema"]
            clean_enc = tokenizer.apply_chat_template(
                clean_messages,
                tools=tools,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
            corrupt_enc = tokenizer.apply_chat_template(
                corrupt_messages,
                tools=tools,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
            clean_ids = clean_enc["input_ids"][0].detach().cpu()
            corrupt_ids = corrupt_enc["input_ids"][0].detach().cpu()
            clean_text = tokenizer.decode(clean_ids.tolist(), clean_up_tokenization_spaces=False)
            corrupt_text = tokenizer.decode(corrupt_ids.tolist(), clean_up_tokenization_spaces=False)
        else:
            clean_text = clean_path.read_text(encoding="utf-8")
            corrupt_text = corrupt_path.read_text(encoding="utf-8")
            clean_ids = tokenizer(clean_text, add_special_tokens=False, return_tensors="pt")["input_ids"][0].detach().cpu()
            corrupt_ids = tokenizer(corrupt_text, add_special_tokens=False, return_tensors="pt")["input_ids"][0].detach().cpu()
        example = PairExample(
            order=order,
            pair_id=pair_id,
            sample_id=sample_id,
            split=pair_split,
            language=str(row.get("language", "unknown")),
            dataset_name=str(row.get("dataset_name", "manifest_pairs")),
            clean_candidate=str(row.get("clean_candidate", "unknown")),
            corrupt_candidate=str(row.get("corrupt_candidate", "unknown")),
            clean_path=clean_path,
            corrupt_path=corrupt_path,
            clean_text=clean_text,
            corrupt_text=corrupt_text,
            clean_input_ids_cpu=clean_ids,
            corrupt_input_ids_cpu=corrupt_ids,
            clean_len=int(clean_ids.numel()),
            corrupt_len=int(corrupt_ids.numel()),
        )
        pairs.append(example)
        if max_pairs > 0 and len(pairs) >= max_pairs:
            break
    if not pairs:
        raise RuntimeError(f"No manifest pairs found in {dataset_root} for split={split!r}")
    return pairs


def _load_split_metadata(metadata_root: Path) -> dict[str, dict[str, Any]]:
    manifest_path = metadata_root / "manifest.jsonl"
    if not manifest_path.exists():
        return {}
    rows = read_jsonl(manifest_path)
    return {str(row["sample_id"]): row for row in rows}


def _discover_flat_pairs(dataset_root: Path) -> tuple[dict[int, Path], dict[int, Path]]:
    clean_paths: dict[int, Path] = {}
    corrupt_paths: dict[int, Path] = {}
    for path in sorted(dataset_root.glob("*.txt")):
        match = PAIR_NAME_RE.fullmatch(path.name)
        if match is None:
            continue
        side = match.group(1)
        pair_id = int(match.group(2))
        if side == "clean":
            clean_paths[pair_id] = path
        else:
            corrupt_paths[pair_id] = path
    if len(clean_paths) != len(corrupt_paths):
        raise RuntimeError(
            f"Flat pair count mismatch under {dataset_root}: clean={len(clean_paths)} corrupt={len(corrupt_paths)}"
        )
    if set(clean_paths) != set(corrupt_paths):
        missing_in_corrupt = sorted(set(clean_paths) - set(corrupt_paths))
        missing_in_clean = sorted(set(corrupt_paths) - set(clean_paths))
        raise RuntimeError(
            "Flat pair id mismatch under "
            f"{dataset_root}: missing_in_corrupt={missing_in_corrupt[:20]} missing_in_clean={missing_in_clean[:20]}"
        )
    if not clean_paths:
        raise RuntimeError(f"No clean/corrupt flat pairs found under {dataset_root}")
    return clean_paths, corrupt_paths


def _resolve_split_and_sample_id(
    pair_id: int,
    pair_manifest_row: dict[str, Any] | None,
    split_manifest_by_sample: dict[str, dict[str, Any]],
) -> tuple[str, str]:
    if pair_manifest_row is None:
        return "all", f"pair_{pair_id}"
    source_sample_id = str(pair_manifest_row.get("source_sample_id", f"pair_{pair_id}"))
    if source_sample_id in split_manifest_by_sample:
        return str(split_manifest_by_sample[source_sample_id].get("split", "all")), source_sample_id
    return "custom500", source_sample_id


def load_pairs(
    tokenizer,
    *,
    dataset_root: Path = DEFAULT_DATASET_ROOT,
    metadata_root: Path = DEFAULT_METADATA_ROOT,
    split: str | None = None,
    max_pairs: int = 0,
) -> list[PairExample]:
    manifest_path = dataset_root / "manifest.jsonl"
    clean_dir = dataset_root / "clean"
    corrupt_dir = dataset_root / "corrupt"
    if manifest_path.exists() and clean_dir.exists() and corrupt_dir.exists():
        return _load_manifest_pairs(
            tokenizer,
            dataset_root=dataset_root,
            split=split,
            max_pairs=max_pairs,
        )
    pair_manifest = _load_pair_manifest(dataset_root)
    split_manifest_by_sample = _load_split_metadata(metadata_root)
    clean_paths, corrupt_paths = _discover_flat_pairs(dataset_root)
    pairs: list[PairExample] = []
    for order, pair_id in enumerate(sorted(clean_paths), start=1):
        row = pair_manifest.get(pair_id)
        pair_split, sample_id = _resolve_split_and_sample_id(pair_id, row, split_manifest_by_sample)
        if split is not None and split != "all" and pair_split != split:
            continue
        clean_path = clean_paths[pair_id]
        corrupt_path = corrupt_paths[pair_id]
        clean_text = clean_path.read_text(encoding="utf-8")
        corrupt_text = corrupt_path.read_text(encoding="utf-8")
        clean_ids = tokenizer(clean_text, add_special_tokens=False, return_tensors="pt")["input_ids"][0].detach().cpu()
        corrupt_ids = tokenizer(corrupt_text, add_special_tokens=False, return_tensors="pt")["input_ids"][0].detach().cpu()
        example = PairExample(
            order=order,
            pair_id=pair_id,
            sample_id=sample_id,
            split=pair_split,
            language=str((row or {}).get("language", "unknown")),
            dataset_name=str((row or {}).get("dataset_name", "flat_pairs")),
            clean_candidate=str((row or {}).get("assigned_clean_candidate", (row or {}).get("clean_candidate", "unknown"))),
            corrupt_candidate=str((row or {}).get("assigned_corrupt_candidate", (row or {}).get("corrupt_candidate", "unknown"))),
            clean_path=clean_path,
            corrupt_path=corrupt_path,
            clean_text=clean_text,
            corrupt_text=corrupt_text,
            clean_input_ids_cpu=clean_ids,
            corrupt_input_ids_cpu=corrupt_ids,
            clean_len=int(clean_ids.numel()),
            corrupt_len=int(corrupt_ids.numel()),
        )
        pairs.append(example)
        if max_pairs > 0 and len(pairs) >= max_pairs:
            break
    if not pairs:
        raise RuntimeError(f"No pairs found in {dataset_root} for split={split!r}")
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
