#!/usr/bin/env python3
"""
Shared dataset and summary helpers for tool-call circuit experiments.

Supports both:
- the new dataset-root layout with `clean/` and `corrupt/` text files plus manifests;
- the legacy pair-style layout used by the reference project.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch


@dataclass(frozen=True)
class ToolCallTargetSpec:
    text: str
    token_ids: Tuple[int, ...]

    @property
    def is_single_token(self) -> bool:
        return len(self.token_ids) == 1

    @property
    def primary_token_id(self) -> int:
        if not self.is_single_token:
            raise ValueError(f"Target text {self.text!r} is not a single token: {self.token_ids}")
        return int(self.token_ids[0])

    def to_dict(self, tokenizer) -> Dict[str, object]:
        return {
            "text": self.text,
            "token_ids": list(self.token_ids),
            "tokens": tokenizer.convert_ids_to_tokens(list(self.token_ids)),
            "length": len(self.token_ids),
            "is_single_token": self.is_single_token,
        }


@dataclass(frozen=True)
class ToolCallSample:
    sample_id: str
    sample_rank: int
    filename: str
    clean_path: Path
    corrupt_path: Path
    source_kind: str
    legacy_index: Optional[int] = None
    clean_manifest: Optional[Dict[str, object]] = None
    corrupt_manifest: Optional[Dict[str, object]] = None

    @property
    def stem(self) -> str:
        return self.clean_path.stem

    def catalog_record(self) -> Dict[str, object]:
        clean_manifest = dict(self.clean_manifest or {})
        corrupt_manifest = dict(self.corrupt_manifest or {})
        return {
            "sample_id": self.sample_id,
            "sample_rank": self.sample_rank,
            "filename": self.filename,
            "clean_path": str(self.clean_path),
            "corrupt_path": str(self.corrupt_path),
            "source_kind": self.source_kind,
            "legacy_index": self.legacy_index,
            "clean_candidate": clean_manifest.get("clean_candidate"),
            "clean_candidate_type": clean_manifest.get("clean_candidate_type"),
            "corrupt_candidate": clean_manifest.get("corrupt_candidate")
            or corrupt_manifest.get("assigned_candidate"),
            "corrupt_candidate_type": clean_manifest.get("corrupt_candidate_type")
            or corrupt_manifest.get("assigned_candidate_type"),
            "dataset_name": clean_manifest.get("dataset_name")
            or clean_manifest.get("dataset")
            or corrupt_manifest.get("dataset_name")
            or corrupt_manifest.get("dataset"),
            "language": clean_manifest.get("language") or corrupt_manifest.get("language"),
            "global_row_id": clean_manifest.get("global_row_id") or corrupt_manifest.get("global_row_id"),
            "clean_prompt_token_length": clean_manifest.get("clean_prompt_token_length"),
            "corrupt_prompt_token_length": clean_manifest.get("corrupt_prompt_token_length"),
            "clean_manifest": clean_manifest,
            "corrupt_manifest": corrupt_manifest,
        }


@dataclass(frozen=True)
class SummaryRecord:
    sample_id: str
    sample_rank: Optional[int]
    legacy_index: Optional[int]
    summary_path: Path
    summary: Dict[str, object]


def _load_manifest_by_filename(manifest_path: Path) -> Dict[str, Dict[str, object]]:
    if not manifest_path.exists():
        return {}
    records: Dict[str, Dict[str, object]] = {}
    with manifest_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            filename = str(row.get("output_filename") or row.get("source_filename") or "")
            if not filename:
                continue
            records[filename] = row
    return records


def _pair_sort_key(filename: str) -> Tuple[int, str]:
    stem = Path(filename).stem
    prefix_digits = ""
    for ch in reversed(stem):
        if ch.isdigit():
            prefix_digits = ch + prefix_digits
        else:
            break
    numeric = int(prefix_digits) if prefix_digits else 10**9
    return numeric, stem


def load_dataset_samples(dataset_root: Path) -> List[ToolCallSample]:
    dataset_root = dataset_root.resolve()
    clean_dir = dataset_root / "clean"
    corrupt_dir = dataset_root / "corrupt"
    if not clean_dir.exists() or not corrupt_dir.exists():
        raise FileNotFoundError(f"Expected clean/ and corrupt/ under {dataset_root}")

    clean_txt = {p.name: p for p in clean_dir.glob("*.txt")}
    corrupt_txt = {p.name: p for p in corrupt_dir.glob("*.txt")}
    shared = sorted(set(clean_txt) & set(corrupt_txt), key=_pair_sort_key)
    missing_clean = sorted(set(corrupt_txt) - set(clean_txt))
    missing_corrupt = sorted(set(clean_txt) - set(corrupt_txt))
    if missing_clean or missing_corrupt:
        raise ValueError(
            f"Mismatched clean/corrupt files under {dataset_root}: "
            f"missing_clean={len(missing_clean)} missing_corrupt={len(missing_corrupt)}"
        )

    clean_manifest = _load_manifest_by_filename(clean_dir / "manifest.jsonl")
    corrupt_manifest = _load_manifest_by_filename(corrupt_dir / "manifest.jsonl")

    samples: List[ToolCallSample] = []
    for rank, filename in enumerate(shared, start=1):
        stem = Path(filename).stem
        samples.append(
            ToolCallSample(
                sample_id=stem,
                sample_rank=rank,
                filename=filename,
                clean_path=clean_txt[filename].resolve(),
                corrupt_path=corrupt_txt[filename].resolve(),
                source_kind="dataset_root",
                legacy_index=None,
                clean_manifest=clean_manifest.get(filename, {}),
                corrupt_manifest=corrupt_manifest.get(filename, {}),
            )
        )
    return samples


def load_pair_samples(pair_dir: Path) -> List[ToolCallSample]:
    pair_dir = pair_dir.resolve()
    clean_by_q: Dict[int, Path] = {}
    corrupt_by_q: Dict[int, Path] = {}
    for p in pair_dir.glob("prompt-clean-q*.txt"):
        q = int(p.stem.split("-q")[-1])
        clean_by_q[q] = p
    for p in pair_dir.glob("prompt-corrupted-q*.txt"):
        q = int(p.stem.split("-q")[-1])
        corrupt_by_q[q] = p
    shared = sorted(set(clean_by_q) & set(corrupt_by_q))
    samples: List[ToolCallSample] = []
    for rank, q in enumerate(shared, start=1):
        samples.append(
            ToolCallSample(
                sample_id=f"q{q:03d}",
                sample_rank=rank,
                filename=f"q{q:03d}",
                clean_path=clean_by_q[q].resolve(),
                corrupt_path=corrupt_by_q[q].resolve(),
                source_kind="pair_dir",
                legacy_index=q,
                clean_manifest={},
                corrupt_manifest={},
            )
        )
    return samples


def load_toolcall_samples(
    dataset_root: Optional[Path] = None,
    pair_dir: Optional[Path] = None,
) -> List[ToolCallSample]:
    if dataset_root is not None:
        return load_dataset_samples(Path(dataset_root))
    if pair_dir is not None:
        return load_pair_samples(Path(pair_dir))
    raise ValueError("Either dataset_root or pair_dir must be provided.")


def parse_legacy_index_list(raw: str) -> List[int]:
    out: List[int] = []
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            lo_s, hi_s = chunk.split("-", 1)
            lo, hi = int(lo_s), int(hi_s)
            if lo > hi:
                lo, hi = hi, lo
            out.extend(range(lo, hi + 1))
        else:
            out.append(int(chunk))
    return sorted(set(out))


def parse_sample_id_list(raw: str) -> List[str]:
    out: List[str] = []
    seen = set()
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk or chunk in seen:
            continue
        out.append(chunk)
        seen.add(chunk)
    return out


def select_samples(
    samples: Sequence[ToolCallSample],
    *,
    sample_ids: Sequence[str] = (),
    sample_rank_min: int = 1,
    sample_rank_max: int = 0,
    legacy_indices: Sequence[int] = (),
    max_samples: int = 0,
) -> List[ToolCallSample]:
    if sample_ids:
        by_id = {s.sample_id: s for s in samples}
        selected = [by_id[sid] for sid in sample_ids if sid in by_id]
    elif legacy_indices:
        legacy_set = set(int(x) for x in legacy_indices)
        selected = [s for s in samples if s.legacy_index in legacy_set]
    else:
        selected = []
        for sample in samples:
            if sample.sample_rank < sample_rank_min:
                continue
            if sample_rank_max > 0 and sample.sample_rank > sample_rank_max:
                continue
            selected.append(sample)
    if max_samples > 0:
        selected = selected[:max_samples]
    return selected


def save_sample_catalog(samples: Sequence[ToolCallSample], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample.catalog_record(), ensure_ascii=False) + "\n")


def get_tool_call_target_spec(tokenizer, target_text: str = "<tool_call>") -> ToolCallTargetSpec:
    ids = tokenizer(target_text, add_special_tokens=False)["input_ids"]
    if not ids:
        raise ValueError(f"Tokenizer produced an empty sequence for {target_text!r}")
    return ToolCallTargetSpec(text=target_text, token_ids=tuple(int(x) for x in ids))


def resolve_distractor_token(corrupt_last_logits: torch.Tensor, target_token_id: int) -> int:
    topk = torch.topk(corrupt_last_logits, k=min(5, int(corrupt_last_logits.shape[-1]))).indices.tolist()
    for idx in topk:
        if int(idx) != int(target_token_id):
            return int(idx)
    return int(torch.argmax(corrupt_last_logits).item())


def find_diff_positions(ids_a: Sequence[int], ids_b: Sequence[int]) -> List[int]:
    return [i for i, (a, b) in enumerate(zip(ids_a, ids_b)) if int(a) != int(b)]


def group_contiguous_positions(positions: Sequence[int]) -> List[List[int]]:
    ordered = sorted({int(p) for p in positions})
    if not ordered:
        return []
    groups: List[List[int]] = [[ordered[0]]]
    for pos in ordered[1:]:
        if pos == groups[-1][-1] + 1:
            groups[-1].append(pos)
        else:
            groups.append([pos])
    return groups


def find_subsequence_positions(seq: Sequence[int], sub: Sequence[int]) -> List[int]:
    if not sub or len(sub) > len(seq):
        return []
    out: List[int] = []
    width = len(sub)
    target = [int(x) for x in sub]
    values = [int(x) for x in seq]
    for start in range(len(values) - width + 1):
        if values[start : start + width] == target:
            out.extend(range(start, start + width))
    return out


def find_last_subsequence_start(seq: Sequence[int], sub: Sequence[int]) -> int:
    if not sub or len(sub) > len(seq):
        return -1
    width = len(sub)
    target = [int(x) for x in sub]
    values = [int(x) for x in seq]
    for start in range(len(values) - width, -1, -1):
        if values[start : start + width] == target:
            return start
    return -1


def find_next_subsequence_start(seq: Sequence[int], sub: Sequence[int], start: int) -> int:
    if not sub or len(sub) > len(seq):
        return -1
    width = len(sub)
    target = [int(x) for x in sub]
    values = [int(x) for x in seq]
    for idx in range(max(0, int(start)), len(values) - width + 1):
        if values[idx : idx + width] == target:
            return idx
    return -1


def _token_offsets(text: str, tokenizer) -> List[Tuple[int, int]]:
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    return [(int(start), int(end)) for start, end in enc["offset_mapping"]]


def find_text_occurrence_token_spans(text: str, needle: str, tokenizer) -> List[List[int]]:
    if not needle:
        return []
    offsets = _token_offsets(text, tokenizer)
    spans: List[List[int]] = []
    search_start = 0
    while True:
        char_start = text.find(needle, search_start)
        if char_start < 0:
            break
        char_end = char_start + len(needle)
        positions = [idx for idx, (tok_start, tok_end) in enumerate(offsets) if tok_start < char_end and tok_end > char_start]
        if positions:
            spans.append(positions)
        search_start = char_start + 1
    return spans


def build_position_sets(
    ids_clean: Sequence[int],
    ids_corrupt: Sequence[int],
    tokenizer,
    clean_text: Optional[str] = None,
) -> Dict[str, List[int]]:
    seq_len = min(len(ids_clean), len(ids_corrupt))
    diff = find_diff_positions(ids_clean[:seq_len], ids_corrupt[:seq_len])
    contrast_spans = group_contiguous_positions(diff)

    if clean_text is not None:
        open_tool_tag_spans = find_text_occurrence_token_spans(clean_text, "<tool_call>", tokenizer)
        close_tool_tag_spans = find_text_occurrence_token_spans(clean_text, "</tool_call>", tokenizer)
        open_tool_tag_positions = [pos for span in open_tool_tag_spans for pos in span]
        close_tool_tag_positions = [pos for span in close_tool_tag_spans for pos in span]
    else:
        tok_tool_call = tokenizer("<tool_call>", add_special_tokens=False)["input_ids"]
        tok_end_tool_call = tokenizer("</tool_call>", add_special_tokens=False)["input_ids"]
        open_tool_tag_positions = find_subsequence_positions(ids_clean, tok_tool_call)
        close_tool_tag_positions = find_subsequence_positions(ids_clean, tok_end_tool_call)
    tool_call_tags = sorted(set(open_tool_tag_positions + close_tool_tag_positions))

    if clean_text is not None:
        tools_open_spans = find_text_occurrence_token_spans(clean_text, "<tools>", tokenizer)
        tools_close_spans = find_text_occurrence_token_spans(clean_text, "</tools>", tokenizer)
        tools_open_pos = [pos for span in tools_open_spans for pos in span]
        tools_close_pos = [pos for span in tools_close_spans for pos in span]
    else:
        tok_tools_open = tokenizer("<tools>", add_special_tokens=False)["input_ids"]
        tok_tools_close = tokenizer("</tools>", add_special_tokens=False)["input_ids"]
        tools_open_pos = find_subsequence_positions(ids_clean, tok_tools_open)
        tools_close_pos = find_subsequence_positions(ids_clean, tok_tools_close)
    if tools_open_pos and tools_close_pos:
        tools_block = list(range(min(tools_open_pos), max(tools_close_pos) + 1))
    else:
        tools_block = []

    if clean_text is not None:
        user_spans = find_text_occurrence_token_spans(clean_text, "<|im_start|>user", tokenizer)
        im_end_spans = find_text_occurrence_token_spans(clean_text, "<|im_end|>", tokenizer)
        user_span = user_spans[-1] if user_spans else []
        later_im_end_spans = [span for span in im_end_spans if span and user_span and span[0] >= user_span[0]]
        if user_span and later_im_end_spans:
            user_block = list(range(user_span[0], later_im_end_spans[0][-1] + 1))
        elif user_span:
            user_block = list(range(user_span[0], len(ids_clean)))
        else:
            user_block = []
    else:
        tok_user = tokenizer("<|im_start|>user", add_special_tokens=False)["input_ids"]
        tok_im_end = tokenizer("<|im_end|>", add_special_tokens=False)["input_ids"]
        user_start = find_last_subsequence_start(ids_clean, tok_user)
        if user_start >= 0 and tok_im_end:
            user_end = find_next_subsequence_start(ids_clean, tok_im_end, user_start + len(tok_user))
            if user_end >= 0:
                user_block = list(range(user_start, user_end + len(tok_im_end)))
            else:
                user_block = list(range(user_start, len(ids_clean)))
        else:
            user_block = []

    recent_hi = max(0, len(ids_clean) - 1)
    recent_lo = max(0, recent_hi - 32)
    recent_32 = list(range(recent_lo, recent_hi))
    prefix_16 = list(range(min(16, len(ids_clean))))

    return {
        "contrast": diff,
        "contrast_spans": [list(span) for span in contrast_spans],
        "tool_call_open": sorted(set(open_tool_tag_positions)),
        "tool_call_close": sorted(set(close_tool_tag_positions)),
        "tool_call_tags": tool_call_tags,
        "tools_block": tools_block,
        "user_block": user_block,
        "recent_32": recent_32,
        "prefix_16": prefix_16,
    }


def decode_token_at(tokenizer, ids: Sequence[int], pos: int) -> str:
    if pos < 0 or pos >= len(ids):
        return ""
    return tokenizer.decode([int(ids[pos])]).replace("\n", "\\n")


def summary_sort_key(summary: Dict[str, object], fallback_id: str) -> Tuple[int, int, str]:
    rank = summary.get("sample_rank")
    if isinstance(rank, int):
        return (0, int(rank), fallback_id)
    if isinstance(rank, float) and math.isfinite(rank):
        return (0, int(rank), fallback_id)
    legacy = summary.get("q_index")
    if isinstance(legacy, int):
        return (1, int(legacy), fallback_id)
    if isinstance(legacy, float) and math.isfinite(legacy):
        return (1, int(legacy), fallback_id)
    return (2, 0, fallback_id)


def load_summary_records(root: Path, summary_name: str = "summary.json") -> List[SummaryRecord]:
    root = root.resolve()
    records: List[SummaryRecord] = []
    for summary_path in root.glob(f"*/{summary_name}"):
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        sample_id = str(summary.get("sample_id") or summary_path.parent.name)
        sample_rank_raw = summary.get("sample_rank")
        sample_rank: Optional[int]
        if isinstance(sample_rank_raw, (int, float)) and math.isfinite(float(sample_rank_raw)):
            sample_rank = int(sample_rank_raw)
        else:
            sample_rank = None
        legacy_raw = summary.get("q_index")
        legacy_index: Optional[int]
        if isinstance(legacy_raw, (int, float)) and math.isfinite(float(legacy_raw)):
            legacy_index = int(legacy_raw)
        else:
            legacy_index = None
        records.append(
            SummaryRecord(
                sample_id=sample_id,
                sample_rank=sample_rank,
                legacy_index=legacy_index,
                summary_path=summary_path.resolve(),
                summary=summary,
            )
        )
    records.sort(key=lambda r: summary_sort_key(r.summary, r.sample_id))
    return records


def jsonable_sample_identity(summary: Dict[str, object], fallback_id: str) -> Dict[str, object]:
    return {
        "sample_id": str(summary.get("sample_id") or fallback_id),
        "sample_rank": summary.get("sample_rank"),
        "legacy_index": summary.get("q_index"),
        "filename": summary.get("filename"),
    }
