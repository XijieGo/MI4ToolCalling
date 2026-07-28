#!/usr/bin/env python3
from __future__ import annotations

import csv
import gc
import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Sequence

import numpy as np
import torch

import sys

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from toolcall_circuit.single_sample import load_hooked_qwen3  # noqa: E402


SEED = 42
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "datasets"
TOOL_CALL_STR = "<tool_call>"
SYSTEM_START = "<|im_start|>system\n"
SYSTEM_END = "<|im_end|>"


@dataclass
class SamplePair:
    order: int
    sample_id: str
    clean_path: Path
    corrupt_path: Path
    clean_text: str
    corrupt_text: str
    clean_tokens_cpu: torch.Tensor
    corrupt_tokens_cpu: torch.Tensor
    token_len: int
    clean_candidate: str | None
    corrupt_candidate: str | None


@dataclass
class PairBatch:
    indices: list[int]
    clean_tokens_cpu: torch.Tensor
    corrupt_tokens_cpu: torch.Tensor
    token_len: int


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def clear_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    ensure_dir(path.parent)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Dict[str, object]) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    ensure_dir(path.parent)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def load_model_and_tokenizer(
    *,
    model_path: Path,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    tool_call_str: str = TOOL_CALL_STR,
):
    model, tokenizer = load_hooked_qwen3(str(model_path), device=device, dtype=dtype)
    tool_token_ids = tokenizer.encode(tool_call_str, add_special_tokens=False)
    if len(tool_token_ids) != 1:
        raise ValueError(f"{tool_call_str!r} is not a single token: {tool_token_ids}")
    return model, tokenizer, int(tool_token_ids[0])


def load_manifest_rows(dataset_root: Path, split: str, condition: str) -> list[dict]:
    manifest_path = dataset_root / split / condition / "manifest.jsonl"
    rows: list[dict] = []
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        raise RuntimeError(f"No rows found in {manifest_path}")
    return rows


def load_sample_pairs(
    model,
    *,
    dataset_root: Path = DEFAULT_DATASET_ROOT,
    split: str = "test",
    max_pairs: int = 0,
) -> list[SamplePair]:
    clean_root = dataset_root / split / "clean"
    corrupt_root = dataset_root / split / "corrupt"
    clean_rows = load_manifest_rows(dataset_root, split, "clean")

    pairs: list[SamplePair] = []
    for order, row in enumerate(clean_rows, start=1):
        filename = str(row.get("output_filename") or row.get("source_filename") or "")
        if not filename:
            continue
        sample_id = Path(filename).stem
        clean_path = clean_root / filename
        corrupt_path = corrupt_root / filename
        if not clean_path.exists() or not corrupt_path.exists():
            continue
        clean_text = clean_path.read_text(encoding="utf-8")
        corrupt_text = corrupt_path.read_text(encoding="utf-8")
        clean_tokens_cpu = model.to_tokens(clean_text, prepend_bos=False).detach().cpu()
        corrupt_tokens_cpu = model.to_tokens(corrupt_text, prepend_bos=False).detach().cpu()
        clean_len = int(clean_tokens_cpu.shape[-1])
        corrupt_len = int(corrupt_tokens_cpu.shape[-1])
        if clean_len != corrupt_len:
            continue
        pairs.append(
            SamplePair(
                order=order,
                sample_id=sample_id,
                clean_path=clean_path,
                corrupt_path=corrupt_path,
                clean_text=clean_text,
                corrupt_text=corrupt_text,
                clean_tokens_cpu=clean_tokens_cpu,
                corrupt_tokens_cpu=corrupt_tokens_cpu,
                token_len=clean_len,
                clean_candidate=str(row.get("clean_candidate")) if row.get("clean_candidate") is not None else None,
                corrupt_candidate=str(row.get("corrupt_candidate")) if row.get("corrupt_candidate") is not None else None,
            )
        )
        if max_pairs > 0 and len(pairs) >= max_pairs:
            break
    if not pairs:
        raise RuntimeError(f"No usable equal-length pairs found in {dataset_root / split}")
    return pairs


def build_pair_batches(pairs: Sequence[SamplePair], batch_size: int) -> list[PairBatch]:
    buckets: dict[int, list[tuple[int, SamplePair]]] = defaultdict(list)
    for idx, pair in enumerate(pairs):
        buckets[pair.token_len].append((idx, pair))

    batches: list[PairBatch] = []
    for token_len in sorted(buckets):
        group = buckets[token_len]
        for start in range(0, len(group), max(int(batch_size), 1)):
            chunk = group[start : start + max(int(batch_size), 1)]
            indices = [idx for idx, _pair in chunk]
            clean_tokens_cpu = torch.cat([pair.clean_tokens_cpu for _idx, pair in chunk], dim=0)
            corrupt_tokens_cpu = torch.cat([pair.corrupt_tokens_cpu for _idx, pair in chunk], dim=0)
            batches.append(
                PairBatch(
                    indices=indices,
                    clean_tokens_cpu=clean_tokens_cpu,
                    corrupt_tokens_cpu=corrupt_tokens_cpu,
                    token_len=token_len,
                )
            )
    return batches


def tool_stats(logits: torch.Tensor, tool_token_id: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    last_logits = logits[:, -1, :].float()
    tool_logit = last_logits[:, tool_token_id].detach().cpu()
    tool_prob = torch.softmax(last_logits, dim=-1)[:, tool_token_id].detach().cpu()
    top1 = last_logits.argmax(dim=-1).detach().cpu()
    return tool_logit, tool_prob, top1


def decode_token(tokenizer, token_id: int) -> str:
    return tokenizer.decode([int(token_id)], clean_up_tokenization_spaces=False)


def choose_best_layer(rows: Sequence[Dict[str, object]], *, tolerance: float = 0.02) -> int:
    if not rows:
        raise ValueError("No rows supplied")
    rows_sorted = sorted(rows, key=lambda row: int(row["layer"]))
    max_rate = max(float(row["tool_call_top1_rate"]) for row in rows_sorted)
    candidates = [
        row
        for row in rows_sorted
        if float(row["tool_call_top1_rate"]) >= max_rate - float(tolerance)
    ]
    if candidates:
        return int(candidates[0]["layer"])
    return int(max(rows_sorted, key=lambda row: float(row["tool_call_top1_rate"]))["layer"])


def get_w_o_layer(model, layer: int) -> torch.Tensor:
    if hasattr(model, "W_O"):
        return model.W_O[layer]
    attn = model.blocks[layer].attn
    if not hasattr(attn, "W_O"):
        raise AttributeError("Could not locate W_O on the model.")
    w_o = attn.W_O
    n_heads = int(model.cfg.n_heads)
    d_head = int(model.cfg.d_head)
    d_model = int(model.cfg.d_model)
    return w_o.view(n_heads, d_head, d_model)


def precompute_head_tool_projections(model, layers: Sequence[int], tool_token_id: int) -> dict[int, torch.Tensor]:
    wu_tool = model.W_U[:, tool_token_id].to(device=model.W_U.device, dtype=torch.float32)
    projections: dict[int, torch.Tensor] = {}
    for layer in layers:
        w_o = get_w_o_layer(model, layer).to(device=model.W_U.device, dtype=torch.float32)
        projections[layer] = torch.einsum("hde,e->hd", w_o, wu_tool)
    return projections


def resolve_pattern_head_idx(requested_head: int, pattern_head_count: int, n_heads: int) -> int:
    if requested_head < pattern_head_count:
        return requested_head
    if pattern_head_count <= 0:
        raise ValueError(f"Invalid pattern_head_count={pattern_head_count}")
    if n_heads % pattern_head_count == 0:
        group_size = n_heads // pattern_head_count
        return requested_head // group_size
    raise ValueError(
        f"Cannot map requested head {requested_head} to pattern head count {pattern_head_count} (n_heads={n_heads})"
    )


def token_positions_for_char_span(text: str, start: int, end: int, tokenizer) -> list[int]:
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    positions: list[int] = []
    for idx, (tok_start, tok_end) in enumerate(encoded["offset_mapping"]):
        if int(tok_start) < int(end) and int(tok_end) > int(start):
            positions.append(int(idx))
    return positions


def system_token_positions(text: str, tokenizer) -> list[int]:
    start = text.find(SYSTEM_START)
    if start < 0:
        return []
    content_start = start + len(SYSTEM_START)
    end = text.find(SYSTEM_END, content_start)
    if end < 0:
        return []
    return token_positions_for_char_span(text, content_start, end, tokenizer)


def locate_schema_span(text: str) -> tuple[int, int] | None:
    start = text.find('{"type":"function"')
    if start < 0:
        return None
    end = text.find("</tools>", start)
    if end < 0:
        return None
    return start, end


def schema_token_positions(text: str, tokenizer) -> list[int]:
    span = locate_schema_span(text)
    if span is None:
        return []
    return token_positions_for_char_span(text, span[0], span[1], tokenizer)


def run_with_hooks_and_cache(model, tokens: torch.Tensor, hook_names: Iterable[str], fwd_hooks):
    wanted = set(hook_names)
    with model.hooks(fwd_hooks=fwd_hooks):
        with torch.no_grad():
            logits, cache = model.run_with_cache(tokens, names_filter=lambda name: name in wanted)
    return logits, cache
