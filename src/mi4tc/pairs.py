"""Frozen paired-prompt loading and construction invariants."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from .io import iter_jsonl


@dataclass(frozen=True)
class Pair:
    sample_id: str
    split: str
    clean_path: Path
    corrupt_path: Path
    clean_text: str
    corrupt_text: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class NativePair:
    """One frozen model-native pair without reconstructing its prompt format.

    Text-native families retain their exact rendered prompt text.  Mistral
    pairs retain the token IDs emitted by its native chat renderer, because
    decoding and re-encoding those IDs loses special control tokens.
    """

    sample_id: str
    split: str
    clean_path: Path
    corrupt_path: Path
    prompt_format: str
    clean_text: str | None
    corrupt_text: str | None
    clean_input_ids: tuple[int, ...] | None
    corrupt_input_ids: tuple[int, ...] | None
    metadata: dict[str, Any]


@dataclass(frozen=True)
class ModelNativePairs:
    """Validated selected-pair collection for one target model."""

    root: Path
    model_key: str
    model_label: str
    tool_call_marker: str
    tool_call_token_id: int
    prompt_format: str
    summary: dict[str, Any]
    train: tuple[NativePair, ...]
    heldout: tuple[NativePair, ...]

    def split_pairs(self, split: str, *, max_pairs: int = 0) -> list[NativePair]:
        if split == "train":
            pairs = self.train
        elif split == "heldout":
            pairs = self.heldout
        else:
            raise ValueError(f"Unknown model-native split: {split!r}")
        if max_pairs < 0:
            raise ValueError("max_pairs must be non-negative")
        return list(pairs[:max_pairs] if max_pairs else pairs)


def _manifest_path(dataset_root: Path, split: str, condition: str) -> Path:
    return dataset_root / split / condition / "manifest.jsonl"


def _row_filename(row: dict[str, Any]) -> str:
    filename = row.get("output_filename") or row.get("filename")
    if not filename:
        raise ValueError(f"Manifest row has no filename: {row}")
    name = str(filename)
    relative = Path(name)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Manifest filename escapes the dataset root: {name!r}")
    return name


def load_pairs(dataset_root: Path, split: str = "test", *, max_pairs: int = 0) -> list[Pair]:
    """Load matched clean/corrupt text files from the portable paired layout."""

    clean_manifest = _manifest_path(dataset_root, split, "clean")
    corrupt_manifest = _manifest_path(dataset_root, split, "corrupt")
    clean_rows = list(iter_jsonl(clean_manifest))
    corrupt_rows = list(iter_jsonl(corrupt_manifest))
    if len(clean_rows) != len(corrupt_rows):
        raise ValueError(f"Manifest lengths differ for {split}: {len(clean_rows)} vs {len(corrupt_rows)}")

    pairs: list[Pair] = []
    for clean_row, corrupt_row in zip(clean_rows, corrupt_rows):
        clean_name = _row_filename(clean_row)
        corrupt_name = _row_filename(corrupt_row)
        if clean_name != corrupt_name:
            raise ValueError(f"Manifest order/name mismatch: {clean_name!r} vs {corrupt_name!r}")
        clean_path = dataset_root / split / "clean" / clean_name
        corrupt_path = dataset_root / split / "corrupt" / corrupt_name
        if not clean_path.is_file() or not corrupt_path.is_file():
            raise FileNotFoundError(f"Missing pair files for {clean_name}")
        metadata = dict(clean_row)
        metadata["corrupt_manifest"] = corrupt_row
        pairs.append(
            Pair(
                sample_id=Path(clean_name).stem,
                split=split,
                clean_path=clean_path,
                corrupt_path=corrupt_path,
                clean_text=clean_path.read_text(encoding="utf-8"),
                corrupt_text=corrupt_path.read_text(encoding="utf-8"),
                metadata=metadata,
            )
        )
        if max_pairs and len(pairs) >= max_pairs:
            break
    return pairs


def sha256_input_ids(ids: Sequence[int]) -> str:
    """Hash native IDs using the compact representation used by the release."""

    payload = json.dumps([int(value) for value in ids], separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def _dataset_path(root: Path, relative: object, *, label: str) -> Path:
    path = Path(str(relative))
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{label} escapes its dataset root: {relative!r}")
    resolved = root / path
    if not resolved.is_file():
        raise FileNotFoundError(f"Missing {label}: {resolved}")
    return resolved


def _require_sha256(value: object, *, label: str) -> str:
    digest = str(value or "")
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return digest


def _native_prompt_format(summary: dict[str, Any], row: dict[str, Any]) -> str:
    declared = summary.get("prompt_format")
    if declared is not None:
        return str(declared)
    clean_relpath = Path(str(row.get("clean_relpath") or ""))
    return "mistral_native_input_ids" if clean_relpath.suffix == ".json" else "native_text"


def _read_native_side(
    path: Path,
    *,
    prompt_format: str,
    expected_hash: str,
    sample_id: str,
    side: str,
) -> tuple[str | None, tuple[int, ...] | None]:
    label = f"{sample_id}/{side}"
    if prompt_format == "native_text":
        if path.suffix != ".txt":
            raise ValueError(f"{label}: native_text prompt must be a .txt file")
        text = path.read_text(encoding="utf-8")
        actual_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if actual_hash != expected_hash:
            raise ValueError(f"{label}: rendered prompt hash does not match the manifest")
        return text, None

    if prompt_format == "mistral_native_input_ids":
        if path.suffix != ".json":
            raise ValueError(f"{label}: Mistral native prompt must be a .json file")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"{label}: native prompt JSON must contain an object")
        if payload.get("prompt_format") != "mistral_native_input_ids":
            raise ValueError(f"{label}: unexpected native prompt format")
        raw_ids = payload.get("input_ids")
        if not isinstance(raw_ids, list) or not raw_ids:
            raise ValueError(f"{label}: native prompt has no input_ids")
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in raw_ids):
            raise ValueError(f"{label}: input_ids must be non-negative integers")
        ids = tuple(int(value) for value in raw_ids)
        actual_hash = sha256_input_ids(ids)
        if actual_hash != expected_hash:
            raise ValueError(f"{label}: stored input_ids hash does not match the manifest")
        declared_hash = _require_sha256(payload.get("input_ids_sha256"), label=f"{label} input_ids_sha256")
        if declared_hash != actual_hash:
            raise ValueError(f"{label}: embedded input_ids hash does not match stored IDs")
        return None, ids

    raise ValueError(f"{label}: unsupported model-native prompt format {prompt_format!r}")


def load_model_native_pairs(dataset_root: Path) -> ModelNativePairs:
    """Load all 500 frozen selected pairs for one model with format checks.

    This reads repository-relative prompt files only.  It never follows the
    historical source paths retained as descriptive manifest provenance.
    """

    summary_path = dataset_root / "summary.json"
    manifest_path = dataset_root / "manifest.jsonl"
    if not summary_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(f"{dataset_root}: expected summary.json and manifest.jsonl")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if not isinstance(summary, dict):
        raise ValueError(f"{summary_path}: expected an object")
    model_key = str(summary.get("model_key") or "")
    # Native collections live at datasets/<model>/pair, except Qwen3-8B, whose
    # pair/ directory is the clean rerun and whose screened source is native/.
    if model_key not in {dataset_root.name, dataset_root.parent.name}:
        raise ValueError(f"{dataset_root}: summary model_key does not match its model directory")
    model_label = str(summary.get("model_label") or "")
    marker = str(summary.get("tool_call_marker") or "")
    token_id = summary.get("tool_call_token_id")
    if not model_label or not marker or isinstance(token_id, bool) or not isinstance(token_id, int) or token_id < 0:
        raise ValueError(f"{dataset_root}: incomplete native model metadata")

    rows = list(iter_jsonl(manifest_path))
    expected_total = summary.get("n_pairs")
    if expected_total != 500 or len(rows) != 500:
        raise ValueError(f"{dataset_root}: expected exactly 500 selected pairs")
    expected_by_split = {"train": summary.get("n_train"), "heldout": summary.get("n_heldout")}
    if expected_by_split != {"train": 200, "heldout": 300}:
        raise ValueError(f"{dataset_root}: expected a 200-train / 300-heldout split")

    grouped: dict[str, list[NativePair]] = {"train": [], "heldout": []}
    seen_ids: set[str] = set()
    observed_formats: set[str] = set()
    for row in rows:
        sample_id = str(row.get("sample_id") or "")
        split = str(row.get("split") or "")
        if not sample_id or sample_id in seen_ids:
            raise ValueError(f"{dataset_root}: missing or duplicate sample_id {sample_id!r}")
        if split not in grouped:
            raise ValueError(f"{dataset_root}: unknown split {split!r} for {sample_id}")
        seen_ids.add(sample_id)
        row_marker = str(row.get("tool_call_marker") or "")
        row_token_id = row.get("tool_call_token_id")
        if row_marker != marker or row_token_id != token_id:
            raise ValueError(f"{dataset_root}: marker metadata mismatch for {sample_id}")
        prompt_format = _native_prompt_format(summary, row)
        observed_formats.add(prompt_format)
        clean_path = _dataset_path(dataset_root, row.get("clean_relpath"), label=f"{sample_id} clean prompt")
        corrupt_path = _dataset_path(dataset_root, row.get("corrupt_relpath"), label=f"{sample_id} corrupt prompt")
        clean_hash = _require_sha256(row.get("clean_prompt_sha256"), label=f"{sample_id} clean_prompt_sha256")
        corrupt_hash = _require_sha256(row.get("corrupt_prompt_sha256"), label=f"{sample_id} corrupt_prompt_sha256")
        clean_text, clean_ids = _read_native_side(
            clean_path,
            prompt_format=prompt_format,
            expected_hash=clean_hash,
            sample_id=sample_id,
            side="clean",
        )
        corrupt_text, corrupt_ids = _read_native_side(
            corrupt_path,
            prompt_format=prompt_format,
            expected_hash=corrupt_hash,
            sample_id=sample_id,
            side="corrupt",
        )
        grouped[split].append(
            NativePair(
                sample_id=sample_id,
                split=split,
                clean_path=clean_path,
                corrupt_path=corrupt_path,
                prompt_format=prompt_format,
                clean_text=clean_text,
                corrupt_text=corrupt_text,
                clean_input_ids=clean_ids,
                corrupt_input_ids=corrupt_ids,
                metadata=dict(row),
            )
        )
    if len(observed_formats) != 1:
        raise ValueError(f"{dataset_root}: mixed native prompt formats are not supported")
    if {split: len(pairs) for split, pairs in grouped.items()} != expected_by_split:
        raise ValueError(f"{dataset_root}: unexpected selected-pair split cardinalities")
    prompt_format = observed_formats.pop()
    return ModelNativePairs(
        root=dataset_root,
        model_key=model_key,
        model_label=model_label,
        tool_call_marker=marker,
        tool_call_token_id=int(token_id),
        prompt_format=prompt_format,
        summary=summary,
        train=tuple(grouped["train"]),
        heldout=tuple(grouped["heldout"]),
    )


def native_pair_token_ids(pair: NativePair, tokenizer: Any | None = None) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Return the exact model input IDs for a native pair.

    Mistral IDs are returned verbatim.  Text-native families are encoded with
    no added special tokens because their prompt files already contain the
    complete native chat rendering.
    """

    if pair.clean_input_ids is not None and pair.corrupt_input_ids is not None:
        return pair.clean_input_ids, pair.corrupt_input_ids
    if pair.clean_text is None or pair.corrupt_text is None:
        raise ValueError(f"{pair.sample_id}: incomplete native prompt payload")
    if tokenizer is None:
        raise ValueError(f"{pair.sample_id}: a tokenizer is required for native text prompts")
    return tuple(token_ids(tokenizer, pair.clean_text)), tuple(token_ids(tokenizer, pair.corrupt_text))


def changed_token_positions_from_ids(clean_ids: Sequence[int], corrupt_ids: Sequence[int]) -> list[int]:
    """Return changed positions for an already tokenized pair."""

    if len(clean_ids) != len(corrupt_ids):
        raise ValueError(f"Token lengths differ ({len(clean_ids)} vs {len(corrupt_ids)})")
    return [index for index, (clean, corrupt) in enumerate(zip(clean_ids, corrupt_ids)) if int(clean) != int(corrupt)]


def changed_char_span(clean_text: str, corrupt_text: str) -> tuple[int, int, int, int]:
    """Return the single replacement span in a pair, or raise if edits are complex."""

    prefix = 0
    limit = min(len(clean_text), len(corrupt_text))
    while prefix < limit and clean_text[prefix] == corrupt_text[prefix]:
        prefix += 1
    clean_suffix = len(clean_text)
    corrupt_suffix = len(corrupt_text)
    while clean_suffix > prefix and corrupt_suffix > prefix and clean_text[clean_suffix - 1] == corrupt_text[corrupt_suffix - 1]:
        clean_suffix -= 1
        corrupt_suffix -= 1
    if clean_text[:prefix] != corrupt_text[:prefix] or clean_text[clean_suffix:] != corrupt_text[corrupt_suffix:]:
        raise ValueError("Could not isolate a contiguous replacement")
    if prefix == clean_suffix and prefix == corrupt_suffix:
        raise ValueError("Pair contains no changed characters")
    return prefix, clean_suffix, prefix, corrupt_suffix


def validate_pairs(dataset_root: Path, *, expected: dict[str, int] | None = None) -> dict[str, Any]:
    expected = expected or {"train": 1200, "test": 300}
    report: dict[str, Any] = {"dataset_root": str(dataset_root), "splits": {}}
    for split, expected_count in expected.items():
        pairs = load_pairs(dataset_root, split)
        changed = 0
        for pair in pairs:
            changed_char_span(pair.clean_text, pair.corrupt_text)
            changed += 1
        if len(pairs) != expected_count:
            raise ValueError(f"Expected {expected_count} {split} pairs, found {len(pairs)}")
        report["splits"][split] = {"pairs": len(pairs), "single_replacement_pairs": changed}
    return report


def token_ids(tokenizer: Any, text: str) -> list[int]:
    encoded = tokenizer(text, add_special_tokens=False)
    if isinstance(encoded, Mapping):
        ids = encoded["input_ids"]
    elif hasattr(encoded, "input_ids"):
        ids = encoded.input_ids
    else:
        ids = encoded
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return [int(item) for item in ids]


def changed_token_positions(tokenizer: Any, pair: Pair) -> list[int]:
    clean_ids = token_ids(tokenizer, pair.clean_text)
    corrupt_ids = token_ids(tokenizer, pair.corrupt_text)
    if len(clean_ids) != len(corrupt_ids):
        raise ValueError(f"{pair.sample_id}: token lengths differ ({len(clean_ids)} vs {len(corrupt_ids)})")
    return [idx for idx, (clean, corrupt) in enumerate(zip(clean_ids, corrupt_ids)) if clean != corrupt]
