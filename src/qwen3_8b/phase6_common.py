#!/usr/bin/env python3
from __future__ import annotations

import csv
import gc
import gzip
import json
import math
import os
import re
import struct
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from tqdm.auto import tqdm

LEGACY_SRC = Path(__file__).resolve().parents[1]
if str(LEGACY_SRC) not in sys.path:
    sys.path.insert(0, str(LEGACY_SRC))

from toolcall_circuit.single_sample import load_hooked_qwen3  # noqa: E402
from artifact_paths import ARTIFACT_ROOT, QWEN3_8B_PATH, QWEN3_8B_TRANSCODER_PATH  # noqa: E402


PROJECT_ROOT = ARTIFACT_ROOT
# The canonical runner sets these explicit variables for every fresh run.
# Fall back to artifact_paths only for a user intentionally invoking a legacy
# script directly.
MODEL_PATH = Path(os.environ.get("MI4_PHASE6_MODEL_PATH", str(QWEN3_8B_PATH)))
TRANSCODER_DIR = Path(os.environ.get("MI4_PHASE6_TRANSCODER_PATH", str(QWEN3_8B_TRANSCODER_PATH)))
DATASET_ROOT = Path(os.environ.get("MI4_PHASE6_DATASET_ROOT", str(PROJECT_ROOT / "datasets" / "test")))
PHASE6_ROOT = Path(os.environ.get("MI4_PHASE6_ROOT", str(PROJECT_ROOT / "results" / "8b_main" / "phase6")))
RUNTIME_DEVICE = os.environ.get("MI4_PHASE6_DEVICE", "cuda")
CACHE_PATH = PHASE6_ROOT / "cache" / "test_activation_cache.pt"
FEATURE_INDEX_PATH = TRANSCODER_DIR / "features" / "index.json.gz"
TOOL_CALL_TOKEN = "<tool_call>"
DEFAULT_CACHE_LAYERS = (25, 32, 33, 34, 35)
L33_HEAD = 29
DTYPE = torch.bfloat16

ACTION_VERBS = {
    "add",
    "build",
    "complete",
    "create",
    "implement",
    "make",
    "process",
    "save",
    "write",
}
ANALYSIS_VERBS = {
    "analyze",
    "analyse",
    "benchmark",
    "clarify",
    "detail",
    "discuss",
    "evaluate",
    "explore",
    "inspect",
    "parse",
    "review",
    "study",
}
SCHEMA_KEYWORDS = {
    "argument",
    "arguments",
    "assistant",
    "description",
    "function",
    "functions",
    "json",
    "name",
    "parameter",
    "parameters",
    "properties",
    "property",
    "required",
    "schema",
    "system",
    "tool",
    "tool_call",
    "tool_calls",
    "tools",
    "type",
    "user",
    "xml",
}
CODE_KEYWORDS = {
    "break",
    "case",
    "class",
    "const",
    "continue",
    "def",
    "else",
    "enum",
    "for",
    "function",
    "if",
    "import",
    "int",
    "let",
    "main",
    "public",
    "private",
    "protected",
    "return",
    "self",
    "solve",
    "static",
    "struct",
    "template",
    "this",
    "var",
    "void",
    "while",
}
CODE_PUNCT = {
    "(",
    ")",
    "{",
    "}",
    "[",
    "]",
    ",",
    ".",
    ":",
    ";",
    "=",
    "+",
    "-",
    "*",
    "/",
    "<",
    ">",
    "::",
    "->",
}
IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
SNAKE_RE = re.compile(r"^[a-z_][a-z0-9_]*$")


@dataclass(frozen=True)
class SamplePair:
    index: int
    sample_id: str
    clean_path: Path
    corrupt_path: Path
    clean_candidate: str
    corrupt_candidate: str
    clean_tokens_cpu: torch.Tensor
    corrupt_tokens_cpu: torch.Tensor
    token_len: int


@dataclass(frozen=True)
class PairBatch:
    indices: List[int]
    clean_tokens_cpu: torch.Tensor
    corrupt_tokens_cpu: torch.Tensor
    token_len: int


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def clear_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def phase6_device() -> str:
    """Honor the canonical runner's device choice with a safe CPU fallback."""

    if RUNTIME_DEVICE.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return RUNTIME_DEVICE


def write_csv(path: Path, rows: Sequence[Dict[str, object]], fieldnames: Sequence[str] | None = None) -> None:
    ensure_dir(path.parent)
    if fieldnames is None:
        seen: List[str] = []
        used = set()
        for row in rows:
            for key in row.keys():
                if key not in used:
                    used.add(key)
                    seen.append(key)
        fieldnames = seen
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_text(path: Path, text: str) -> None:
    ensure_dir(path.parent)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def format_float(value: float, digits: int = 4) -> str:
    if value is None or not math.isfinite(float(value)):
        return "nan"
    return f"{float(value):.{digits}f}"


def read_manifest_rows(condition: str) -> List[dict]:
    manifest_path = DATASET_ROOT / condition / "manifest.jsonl"
    rows: List[dict] = []
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def tool_call_token_id(model) -> int:
    token_ids = model.tokenizer.encode(TOOL_CALL_TOKEN, add_special_tokens=False)
    if len(token_ids) != 1:
        raise ValueError(f"{TOOL_CALL_TOKEN!r} is not a single token: {token_ids}")
    return int(token_ids[0])


def build_sample_pairs(model, *, max_pairs: int | None = None) -> List[SamplePair]:
    clean_rows = read_manifest_rows("clean")
    pairs: List[SamplePair] = []
    for idx, row in enumerate(clean_rows):
        sample_id = Path(str(row["output_filename"])).stem
        clean_path = DATASET_ROOT / "clean" / f"{sample_id}.txt"
        corrupt_path = DATASET_ROOT / "corrupt" / f"{sample_id}.txt"
        if not clean_path.exists() or not corrupt_path.exists():
            continue

        clean_text = clean_path.read_text(encoding="utf-8")
        corrupt_text = corrupt_path.read_text(encoding="utf-8")
        clean_tokens = model.to_tokens(clean_text, prepend_bos=False).detach().cpu()
        corrupt_tokens = model.to_tokens(corrupt_text, prepend_bos=False).detach().cpu()
        if clean_tokens.shape[-1] != corrupt_tokens.shape[-1]:
            continue

        pairs.append(
            SamplePair(
                index=len(pairs),
                sample_id=sample_id,
                clean_path=clean_path,
                corrupt_path=corrupt_path,
                clean_candidate=str(row.get("clean_candidate") or ""),
                corrupt_candidate=str(row.get("corrupt_candidate") or ""),
                clean_tokens_cpu=clean_tokens,
                corrupt_tokens_cpu=corrupt_tokens,
                token_len=int(clean_tokens.shape[-1]),
            )
        )
        if max_pairs is not None and len(pairs) >= max_pairs:
            break

    if not pairs:
        raise RuntimeError("No aligned clean/corrupt test pairs were found.")
    return pairs


def build_pair_batches(pairs: Sequence[SamplePair]) -> List[PairBatch]:
    buckets: Dict[int, List[SamplePair]] = defaultdict(list)
    for pair in pairs:
        buckets[pair.token_len].append(pair)

    batches: List[PairBatch] = []
    for token_len in sorted(buckets.keys()):
        chunk = buckets[token_len]
        batches.append(
            PairBatch(
                indices=[pair.index for pair in chunk],
                clean_tokens_cpu=torch.cat([pair.clean_tokens_cpu for pair in chunk], dim=0),
                corrupt_tokens_cpu=torch.cat([pair.corrupt_tokens_cpu for pair in chunk], dim=0),
                token_len=token_len,
            )
        )
    return batches


def ensure_phase6_cache(
    *,
    cache_path: Path = CACHE_PATH,
    layers: Sequence[int] = DEFAULT_CACHE_LAYERS,
    max_pairs: int | None = None,
    force: bool = False,
) -> dict:
    if cache_path.exists() and not force:
        return torch.load(cache_path, map_location="cpu")

    ensure_dir(cache_path.parent)
    device = phase6_device()
    model, _tokenizer = load_hooked_qwen3(str(MODEL_PATH), device, DTYPE)
    model.set_use_hook_mlp_in(True)
    if hasattr(model.cfg, "use_hook_mlp_in"):
        model.cfg.use_hook_mlp_in = True

    pairs = build_sample_pairs(model, max_pairs=max_pairs)
    batches = build_pair_batches(pairs)
    n_samples = len(pairs)
    d_model = int(model.cfg.d_model)
    tool_token = tool_call_token_id(model)

    mlp_in_clean = {int(layer): torch.empty((n_samples, d_model), dtype=DTYPE) for layer in layers}
    mlp_in_corrupt = {int(layer): torch.empty((n_samples, d_model), dtype=DTYPE) for layer in layers}
    l33h29_out_clean = torch.empty((n_samples, d_model), dtype=DTYPE)
    l33h29_out_corrupt = torch.empty((n_samples, d_model), dtype=DTYPE)
    clean_tool_logit = torch.empty(n_samples, dtype=torch.float32)
    corrupt_tool_logit = torch.empty(n_samples, dtype=torch.float32)
    clean_top1 = torch.empty(n_samples, dtype=torch.long)
    corrupt_top1 = torch.empty(n_samples, dtype=torch.long)

    hook_names = {f"blocks.{int(layer)}.hook_mlp_in" for layer in layers}
    hook_names.add("blocks.33.attn.hook_z")
    W_O = model.blocks[33].attn.W_O[L33_HEAD].detach()

    progress = tqdm(batches, desc="Collect phase6 cache", dynamic_ncols=True)
    for batch in progress:
        tokens = torch.cat([batch.clean_tokens_cpu, batch.corrupt_tokens_cpu], dim=0).to(model.cfg.device)
        clean_batch_size = int(batch.clean_tokens_cpu.shape[0])

        with torch.no_grad():
            logits, cache = model.run_with_cache(tokens, names_filter=lambda name: name in hook_names)

        last_logits = logits[:, -1, :]
        tool_logit = last_logits[:, tool_token].detach().cpu().float()
        top1 = last_logits.argmax(dim=-1).detach().cpu().long()
        clean_tool_logit[batch.indices] = tool_logit[:clean_batch_size]
        corrupt_tool_logit[batch.indices] = tool_logit[clean_batch_size:]
        clean_top1[batch.indices] = top1[:clean_batch_size]
        corrupt_top1[batch.indices] = top1[clean_batch_size:]

        for layer in layers:
            hook_name = f"blocks.{int(layer)}.hook_mlp_in"
            hidden = cache[hook_name][:, -1, :].detach().cpu().to(dtype=DTYPE)
            mlp_in_clean[int(layer)][batch.indices] = hidden[:clean_batch_size]
            mlp_in_corrupt[int(layer)][batch.indices] = hidden[clean_batch_size:]

        head_z = cache["blocks.33.attn.hook_z"][:, -1, L33_HEAD, :].detach()
        head_out = torch.einsum("bd,dm->bm", head_z, W_O).detach().cpu().to(dtype=DTYPE)
        l33h29_out_clean[batch.indices] = head_out[:clean_batch_size]
        l33h29_out_corrupt[batch.indices] = head_out[clean_batch_size:]

        del tokens, logits, cache, last_logits, tool_logit, top1, head_z, head_out
        clear_cuda()

    cache_obj = {
        "provenance": {
            "model_path": str(MODEL_PATH),
            "transcoder_path": str(TRANSCODER_DIR),
            "dataset_root": str(DATASET_ROOT),
            "device": device,
        },
        "sample_ids": [pair.sample_id for pair in pairs],
        "clean_candidates": [pair.clean_candidate for pair in pairs],
        "corrupt_candidates": [pair.corrupt_candidate for pair in pairs],
        "token_lengths": torch.tensor([pair.token_len for pair in pairs], dtype=torch.int32),
        "layers": [int(layer) for layer in layers],
        "mlp_in": {
            "clean": {int(layer): tensor for layer, tensor in mlp_in_clean.items()},
            "corrupt": {int(layer): tensor for layer, tensor in mlp_in_corrupt.items()},
        },
        "l33h29_out": {
            "clean": l33h29_out_clean,
            "corrupt": l33h29_out_corrupt,
        },
        "tool_logit": {
            "clean": clean_tool_logit,
            "corrupt": corrupt_tool_logit,
        },
        "top1": {
            "clean": clean_top1,
            "corrupt": corrupt_top1,
        },
        "tool_token_id": tool_token,
    }
    torch.save(cache_obj, cache_path)

    del model
    clear_cuda()
    return cache_obj


def load_encoder_params(layer: int, *, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    tensors = load_file(str(TRANSCODER_DIR / f"layer_{layer}.safetensors"), device="cpu")
    W_enc = tensors["W_enc"].to(device=device, dtype=DTYPE)
    b_enc = tensors["b_enc"].to(device=device, dtype=DTYPE)
    return W_enc, b_enc


def compute_activation_stats(
    inputs_clean: torch.Tensor,
    inputs_corrupt: torch.Tensor,
    *,
    layer: int,
    batch_size: int = 32,
    device: str | torch.device = "cuda",
) -> dict:
    device = torch.device(device)
    W_enc, b_enc = load_encoder_params(layer, device=device)
    n_features = int(W_enc.shape[0])
    sum_clean = torch.zeros(n_features, dtype=torch.float64)
    sum_corrupt = torch.zeros(n_features, dtype=torch.float64)
    pos_clean = torch.zeros(n_features, dtype=torch.float64)
    pos_corrupt = torch.zeros(n_features, dtype=torch.float64)

    n_samples = int(inputs_clean.shape[0])
    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        batch_clean = inputs_clean[start:end].to(device=device, dtype=DTYPE)
        batch_corrupt = inputs_corrupt[start:end].to(device=device, dtype=DTYPE)
        with torch.no_grad():
            acts_clean = torch.relu(F.linear(batch_clean, W_enc, b_enc))
            acts_corrupt = torch.relu(F.linear(batch_corrupt, W_enc, b_enc))
        sum_clean += acts_clean.sum(dim=0).detach().cpu().double()
        sum_corrupt += acts_corrupt.sum(dim=0).detach().cpu().double()
        pos_clean += (acts_clean > 0).sum(dim=0).detach().cpu().double()
        pos_corrupt += (acts_corrupt > 0).sum(dim=0).detach().cpu().double()
        del batch_clean, batch_corrupt, acts_clean, acts_corrupt
        clear_cuda()

    mean_clean = (sum_clean / max(n_samples, 1)).to(dtype=torch.float32)
    mean_corrupt = (sum_corrupt / max(n_samples, 1)).to(dtype=torch.float32)
    frac_positive_clean = (pos_clean / max(n_samples, 1)).to(dtype=torch.float32)
    frac_positive_corrupt = (pos_corrupt / max(n_samples, 1)).to(dtype=torch.float32)
    delta = mean_clean - mean_corrupt
    ratio = torch.where(mean_clean.abs() > 1e-8, mean_corrupt / mean_clean, torch.full_like(mean_clean, float("nan")))

    del W_enc, b_enc, sum_clean, sum_corrupt, pos_clean, pos_corrupt
    clear_cuda()
    return {
        "mean_clean": mean_clean,
        "mean_corrupt": mean_corrupt,
        "delta": delta,
        "ratio": ratio,
        "frac_positive_clean": frac_positive_clean,
        "frac_positive_corrupt": frac_positive_corrupt,
    }


def compute_ablation_diffs(
    inputs_full: torch.Tensor,
    removed_component: torch.Tensor,
    *,
    layer: int,
    batch_size: int = 32,
    device: str | torch.device = "cuda",
) -> torch.Tensor:
    device = torch.device(device)
    W_enc, b_enc = load_encoder_params(layer, device=device)
    n_features = int(W_enc.shape[0])
    sum_diff = torch.zeros(n_features, dtype=torch.float64)
    n_samples = int(inputs_full.shape[0])

    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        batch_full = inputs_full[start:end].to(device=device, dtype=DTYPE)
        batch_ablated = (inputs_full[start:end] - removed_component[start:end]).to(device=device, dtype=DTYPE)
        with torch.no_grad():
            acts_full = torch.relu(F.linear(batch_full, W_enc, b_enc))
            acts_ablated = torch.relu(F.linear(batch_ablated, W_enc, b_enc))
        sum_diff += (acts_full - acts_ablated).sum(dim=0).detach().cpu().double()
        del batch_full, batch_ablated, acts_full, acts_ablated
        clear_cuda()

    mean_diff = (sum_diff / max(n_samples, 1)).to(dtype=torch.float32)
    del W_enc, b_enc, sum_diff
    clear_cuda()
    return mean_diff


def load_feature_index() -> dict:
    with gzip.open(FEATURE_INDEX_PATH, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def read_feature_metadata(feature_index: dict, layer: int, feature_id: int) -> dict:
    entry = feature_index[str(layer)]
    feature_path = TRANSCODER_DIR / "features" / str(entry["filename"])
    offset = int(entry["offsets"][feature_id])
    with feature_path.open("rb") as handle:
        handle.seek(offset)
        payload_size = struct.unpack("<I", handle.read(4))[0]
        payload = handle.read(payload_size)
    return json.loads(gzip.decompress(payload).decode("utf-8"))


def normalize_token(token: object) -> str:
    text = str(token)
    text = text.replace("Ġ", " ").replace("▁", " ").replace("Ċ", "\n")
    text = text.replace("\\n", "\n")
    return text


def canonical_token(token: object) -> str:
    text = normalize_token(token).strip().lower()
    text = text.strip("`'\"")
    return text


def infer_peak_token_type(tokens: Sequence[object], peak_index: int) -> str:
    peak = canonical_token(tokens[peak_index]) if tokens else ""
    window_tokens = [canonical_token(tok) for tok in tokens[max(0, peak_index - 3) : peak_index + 4]]
    window_text = " ".join(tok for tok in window_tokens if tok)

    if peak in ACTION_VERBS:
        return "action_verb"
    if peak in ANALYSIS_VERBS:
        return "analysis_verb"
    if peak in SCHEMA_KEYWORDS or "tool_call" in peak or "tool" in peak:
        return "schema_keyword"
    if any(word in window_text for word in ("tool", "schema", "parameter", "function", "json", "xml")):
        return "schema_keyword"
    if peak in CODE_KEYWORDS or peak in CODE_PUNCT:
        return "code_token"
    if IDENTIFIER_RE.fullmatch(peak or "") and (peak in CODE_KEYWORDS or SNAKE_RE.fullmatch(peak or "") or peak.endswith("()")):
        return "code_token"
    if re.fullmatch(r"[()\[\]{}:;,.=+\-*/<>]+", peak or ""):
        return "code_token"
    if IDENTIFIER_RE.fullmatch(peak or "") and any(tok in CODE_PUNCT or tok in CODE_KEYWORDS for tok in window_tokens):
        return "code_token"
    return "other"


def infer_context_semantic_label(meta: dict, *, max_examples: int = 5) -> str:
    peak_types = []
    peak_tokens = []
    for example in top_quantile_examples(meta, limit=max_examples):
        tokens = example.get("tokens") or []
        peak_index = int(example.get("train_token_ind") or 0)
        peak_types.append(infer_peak_token_type(tokens, peak_index))
        peak_tokens.append(canonical_token(tokens[peak_index]) if tokens else "")

    counts = Counter(peak_types)
    dominant = counts.most_common(1)[0][0] if counts else "other"
    if dominant == "schema_keyword":
        return "tool_schema"
    if dominant == "action_verb":
        return "action_instruction"
    if dominant == "analysis_verb":
        return "analysis_instruction"
    if dominant == "code_token":
        if any(tok in {"def", "class", "return", "solve", "main"} for tok in peak_tokens):
            return "code_structure"
        return "code_syntax"
    return "mixed_or_other"


def top_quantile_examples(meta: dict, *, limit: int = 5) -> List[dict]:
    quantiles = meta.get("examples_quantiles") or []
    target = None
    for quantile in quantiles:
        if str(quantile.get("quantile_name") or "").lower() == "top":
            target = quantile
            break
    if target is None and quantiles:
        target = quantiles[0]
    if not target:
        return []
    return list((target.get("examples") or [])[:limit])


def peak_token_examples(meta: dict, *, limit: int = 5) -> str:
    peaks: List[str] = []
    for example in top_quantile_examples(meta, limit=limit):
        tokens = example.get("tokens") or []
        peak_index = int(example.get("train_token_ind") or 0)
        if not tokens:
            continue
        peak_text = normalize_token(tokens[peak_index]).replace("\n", "\\n").strip()
        peaks.append(peak_text or "<blank>")
    return " | ".join(peaks)


def dominant_peak_token_type(meta: dict, *, limit: int = 5) -> str:
    peak_types = []
    for example in top_quantile_examples(meta, limit=limit):
        tokens = example.get("tokens") or []
        peak_index = int(example.get("train_token_ind") or 0)
        if not tokens:
            continue
        peak_types.append(infer_peak_token_type(tokens, peak_index))
    if not peak_types:
        return "unknown"
    return Counter(peak_types).most_common(1)[0][0]


def logits_text(items: Iterable[object], *, limit: int = 5) -> str:
    return ", ".join(str(item) for item in list(items)[:limit])


def render_feature_context(meta: dict) -> str:
    lines = []
    lines.append(f"feature_id: {meta.get('index')}")
    lines.append(f"activation_frequency: {meta.get('activation_frequency')}")
    lines.append(f"top_logits: {logits_text(meta.get('top_logits') or [], limit=10)}")
    lines.append(f"bottom_logits: {logits_text(meta.get('bottom_logits') or [], limit=10)}")
    lines.append("")
    lines.append(f"dominant_peak_token_type: {dominant_peak_token_type(meta)}")
    lines.append(f"context_semantic_label: {infer_context_semantic_label(meta)}")
    lines.append("")
    lines.append("Semantic note:")
    lines.append("")

    for example_idx, example in enumerate(top_quantile_examples(meta, limit=5), start=1):
        tokens = list(example.get("tokens") or [])
        acts = list(example.get("tokens_acts_list") or [])
        peak_index = int(example.get("train_token_ind") or 0)
        joined = []
        for idx, token in enumerate(tokens):
            text = normalize_token(token).replace("\n", "\\n")
            if idx == peak_index:
                joined.append(f"<<{text}>>")
            else:
                joined.append(text)
        lines.append(f"Example {example_idx}:")
        lines.append(f"context: {''.join(joined)}")
        lines.append(f"peak_token_type: {infer_peak_token_type(tokens, peak_index)}")
        lines.append("token_acts:")
        for idx, token in enumerate(tokens):
            text = normalize_token(token).replace("\n", "\\n")
            act = acts[idx] if idx < len(acts) else float('nan')
            marker = " <PEAK>" if idx == peak_index else ""
            lines.append(f"  [{idx:03d}] {text!r:>24}  act={float(act):8.4f}{marker}")
        lines.append("")
    return "\n".join(lines).rstrip()


def summarize_counts(labels: Iterable[str]) -> str:
    counts = Counter(labels)
    if not counts:
        return "none"
    return ", ".join(f"{label}={count}" for label, count in counts.most_common())
