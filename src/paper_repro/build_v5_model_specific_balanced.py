#!/usr/bin/env python3
"""Build behavior-screened, model-specific v5 prompt-pair datasets.

Each output dataset contains 500 paired prompts: 200 train pairs and 300
held-out pairs.  Clean and corrupt verbs are balanced independently within
both splits.  A pair is admitted only when the target model predicts its
native tool-call marker as the clean prompt's first token and predicts a
different first token for the corrupt prompt.

The builder deliberately writes only below a new v5 output root.  It never
modifies frozen v2/v3/v4 data or historical model-specific selections.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import itertools
import json
import math
import os
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import networkx as nx
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Mistral3ForConditionalGeneration


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "datasets" / "v5_model_specific_balanced"
MODEL_ROOT = Path(os.environ.get("MODEL_ROOT", PROJECT_ROOT / "external" / "models")).expanduser()
RELEASE_SOURCE_POOLS = PROJECT_ROOT / "artifacts" / "01_model_specific_500" / "data" / "source_pools"

EXECUTION_VERBS = ("add", "build", "complete", "save", "write")
ANALYSIS_VERBS = ("discuss", "explore", "inspect", "review", "study")


@dataclass(frozen=True)
class ModelSpec:
    key: str
    label: str
    model_path: Path
    source_root: Path
    tool_marker: str
    loader: str
    clean_verbs: tuple[str, ...]
    corrupt_verbs: tuple[str, ...]
    candidates_per_cell: int
    batch_size: int


MODEL_SPECS: dict[str, ModelSpec] = {
    "qwen3_4b": ModelSpec(
        key="qwen3_4b",
        label="Qwen3-4B",
        model_path=Path(os.environ.get("QWEN3_4B_PATH", MODEL_ROOT / "Qwen3-4B")).expanduser(),
        source_root=PROJECT_ROOT / "datasets" / "provenance" / "v1_1711" / "base_pairs",
        tool_marker="<tool_call>",
        loader="auto",
        clean_verbs=EXECUTION_VERBS,
        corrupt_verbs=ANALYSIS_VERBS,
        candidates_per_cell=60,
        batch_size=16,
    ),
    "qwen3_8b": ModelSpec(
        key="qwen3_8b",
        label="Qwen3-8B",
        model_path=Path(os.environ.get("QWEN3_8B_PATH", MODEL_ROOT / "Qwen3-8B")).expanduser(),
        source_root=PROJECT_ROOT / "datasets" / "provenance" / "v1_1711" / "base_pairs",
        tool_marker="<tool_call>",
        loader="auto",
        clean_verbs=EXECUTION_VERBS,
        corrupt_verbs=ANALYSIS_VERBS,
        candidates_per_cell=60,
        batch_size=12,
    ),
    "qwen3_14b": ModelSpec(
        key="qwen3_14b",
        label="Qwen3-14B",
        model_path=Path(os.environ.get("QWEN3_14B_PATH", MODEL_ROOT / "Qwen3-14B")).expanduser(),
        source_root=PROJECT_ROOT / "datasets" / "provenance" / "v1_1711" / "base_pairs",
        tool_marker="<tool_call>",
        loader="auto",
        clean_verbs=EXECUTION_VERBS,
        corrupt_verbs=ANALYSIS_VERBS,
        candidates_per_cell=60,
        batch_size=8,
    ),
    "qwen35_4b": ModelSpec(
        key="qwen35_4b",
        label="Qwen3.5-4B",
        model_path=Path(os.environ.get("QWEN35_4B_PATH", MODEL_ROOT / "Qwen3.5-4B")).expanduser(),
        source_root=RELEASE_SOURCE_POOLS / "qwen35_9b_converted_dataset",
        tool_marker="<tool_call>",
        loader="auto",
        clean_verbs=EXECUTION_VERBS,
        corrupt_verbs=ANALYSIS_VERBS,
        candidates_per_cell=60,
        batch_size=16,
    ),
    "qwen35_9b": ModelSpec(
        key="qwen35_9b",
        label="Qwen3.5-9B",
        model_path=Path(os.environ.get("QWEN35_9B_PATH", MODEL_ROOT / "Qwen3.5-9B")).expanduser(),
        source_root=RELEASE_SOURCE_POOLS / "qwen35_9b_converted_dataset",
        tool_marker="<tool_call>",
        loader="auto",
        clean_verbs=EXECUTION_VERBS,
        corrupt_verbs=ANALYSIS_VERBS,
        candidates_per_cell=60,
        batch_size=12,
    ),
    "mistral_3p2_24b": ModelSpec(
        key="mistral_3p2_24b",
        label="Mistral-Small-3.2-24B-Instruct-2506",
        model_path=Path(os.environ.get("MISTRAL_3P2_24B_PATH", MODEL_ROOT / "Mistral-Small-3.2-24B-Instruct-2506")).expanduser(),
        source_root=PROJECT_ROOT / "datasets" / "external" / "tau2_telecom_qwen35_9b",
        tool_marker="[TOOL_CALLS]",
        loader="mistral",
        # The native Mistral template's stable construction pool excludes build.
        clean_verbs=("add", "complete", "save", "write"),
        corrupt_verbs=ANALYSIS_VERBS,
        candidates_per_cell=60,
        batch_size=4,
    ),
    "granite_3p3_8b": ModelSpec(
        key="granite_3p3_8b",
        label="Granite-3.3-8B-Instruct",
        model_path=Path(os.environ.get("GRANITE_3P3_8B_PATH", MODEL_ROOT / "granite-3.3-8b-instruct")).expanduser(),
        source_root=RELEASE_SOURCE_POOLS / "granite_3p3_8b_converted_dataset",
        tool_marker="<|tool_call|>",
        loader="auto",
        # Existing Granite screening identifies these as the reliable call verbs.
        clean_verbs=("add", "save", "write"),
        corrupt_verbs=ANALYSIS_VERBS,
        candidates_per_cell=100,
        batch_size=12,
    ),
}

# Mistral uses native chat-template token IDs and therefore has a dedicated
# builder (`paper_repro.build_v5_mistral_tau2`) rather than this text-prompt
# generic path.
DEFAULT_GENERIC_MODELS = tuple(key for key in MODEL_SPECS if key != "mistral_3p2_24b")


@dataclass(frozen=True)
class PromptTemplate:
    source_sample_id: str
    filename: str
    dataset_name: str
    language: str
    source_split: str
    clean_verb: str
    corrupt_verb: str
    clean_text: str
    corrupt_text: str
    clean_path: Path
    corrupt_path: Path
    edit_start: int
    edit_end: int


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    source: PromptTemplate
    clean_verb: str
    corrupt_verb: str
    clean_text: str
    corrupt_text: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--models", nargs="+", choices=tuple(MODEL_SPECS), default=list(DEFAULT_GENERIC_MODELS))
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--candidate-multiplier", type=float, default=1.0)
    parser.add_argument(
        "--min-clean-margin",
        type=float,
        default=0.0,
        help="Require the screened clean tool-marker logit to exceed the best non-tool logit by this amount.",
    )
    parser.add_argument(
        "--min-corrupt-non-tool-margin",
        type=float,
        default=0.0,
        help="Require the screened corrupt best non-tool logit to exceed the tool-marker logit by this amount.",
    )
    parser.add_argument("--resume", action="store_true", help="Add only missing model directories to an existing v5 root.")
    parser.add_argument("--skip-model-load", action="store_true", help="Only validate sources; do not create a dataset.")
    return parser.parse_args()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def resolve_prompt_path(root: Path, row: dict[str, Any], side: str) -> Path:
    key = f"{side}_prompt_path"
    raw_value = str(row.get(key) or "")
    filename = Path(raw_value).name
    if not filename:
        filename = str(row.get("output_filename") or row.get("filename") or "")
    candidates: list[Path] = [root / side / filename]
    if raw_value:
        raw = Path(raw_value)
        candidates.append(raw)
        if not raw.is_absolute():
            candidates.append(root / raw)
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    attempted = "\n  ".join(str(item) for item in candidates)
    raise FileNotFoundError(f"Could not resolve {side} prompt for {row.get('sample_id')}:\n  {attempted}")


def longest_edit(clean_text: str, corrupt_text: str, clean_verb: str, corrupt_verb: str) -> tuple[int, int]:
    start = 0
    limit = min(len(clean_text), len(corrupt_text))
    while start < limit and clean_text[start] == corrupt_text[start]:
        start += 1
    # The words themselves can share a prefix (``Save`` / ``Study``).  Walk
    # back to the first character of that lexical word before using manifest
    # word lengths to delimit the edit.
    while start > 0 and clean_text[start - 1].isalpha() and corrupt_text[start - 1].isalpha():
        start -= 1
    # Do not derive the right edge from a character-level common suffix: e.g.
    # ``Write`` and ``Explore`` share a terminal ``e`` even though that letter
    # belongs to the edited word.  The manifest supplies the expected words.
    clean_end = start + len(clean_verb)
    corrupt_end = start + len(corrupt_verb)
    clean_span = clean_text[start:clean_end]
    corrupt_span = corrupt_text[start:corrupt_end]
    if clean_span.lower() != clean_verb.lower() or corrupt_span.lower() != corrupt_verb.lower():
        raise ValueError(
            "Expected a one-word clean/corrupt edit, found "
            f"{clean_span!r} -> {corrupt_span!r}; expected {clean_verb!r} -> {corrupt_verb!r}"
        )
    if clean_text[clean_end:] != corrupt_text[corrupt_end:]:
        raise ValueError("Clean/corrupt prompts differ outside the declared verb span")
    return start, clean_end


def load_templates(spec: ModelSpec) -> list[PromptTemplate]:
    manifest_path = spec.source_root / "clean" / "manifest.jsonl"
    if not manifest_path.is_file():
        manifest_path = spec.source_root / "manifest.jsonl"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing source manifest for {spec.key}: {manifest_path}")

    templates: list[PromptTemplate] = []
    failures: list[str] = []
    for row in read_jsonl(manifest_path):
        clean_verb = str(row.get("clean_candidate") or row.get("assigned_clean_candidate") or "").lower()
        corrupt_verb = str(row.get("corrupt_candidate") or row.get("assigned_corrupt_candidate") or "").lower()
        if not clean_verb or not corrupt_verb:
            failures.append(f"{row.get('sample_id')}: missing verb fields")
            continue
        try:
            clean_path = resolve_prompt_path(spec.source_root, row, "clean")
            corrupt_path = resolve_prompt_path(spec.source_root, row, "corrupt")
            clean_text = clean_path.read_text(encoding="utf-8")
            corrupt_text = corrupt_path.read_text(encoding="utf-8")
            edit_start, edit_end = longest_edit(clean_text, corrupt_text, clean_verb, corrupt_verb)
        except (FileNotFoundError, ValueError) as exc:
            failures.append(f"{row.get('sample_id')}: {exc}")
            continue
        source_sample_id = str(row.get("sample_id") or Path(clean_path).stem)
        filename = str(row.get("output_filename") or row.get("filename") or clean_path.name)
        templates.append(
            PromptTemplate(
                source_sample_id=source_sample_id,
                filename=filename,
                dataset_name=str(row.get("dataset_name") or row.get("dataset") or "unknown"),
                language=str(row.get("language") or "unknown"),
                source_split=str(row.get("split") or "source_unspecified"),
                clean_verb=clean_verb,
                corrupt_verb=corrupt_verb,
                clean_text=clean_text,
                corrupt_text=corrupt_text,
                clean_path=clean_path,
                corrupt_path=corrupt_path,
                edit_start=edit_start,
                edit_end=edit_end,
            )
        )
    if failures:
        preview = "\n".join(failures[:5])
        print(f"[{spec.key}] skipped {len(failures)} malformed source pairs:\n{preview}", flush=True)
    if not templates:
        raise RuntimeError(f"No usable prompt templates for {spec.key}")
    if len({item.source_sample_id for item in templates}) != len(templates):
        raise RuntimeError(f"Source manifest for {spec.key} contains duplicate sample IDs")
    return templates


def render_verb(template: PromptTemplate, verb: str) -> str:
    original = template.clean_text[template.edit_start : template.edit_end]
    rendered = verb[:1].upper() + verb[1:] if original[:1].isupper() else verb
    return template.clean_text[: template.edit_start] + rendered + template.clean_text[template.edit_end :]


def build_candidates(
    templates: Sequence[PromptTemplate],
    *,
    clean_verbs: Sequence[str],
    corrupt_verbs: Sequence[str],
    candidates_per_cell: int,
    seed: int,
) -> list[Candidate]:
    cells = [(clean, corrupt) for clean in clean_verbs for corrupt in corrupt_verbs]
    needed = len(cells) * candidates_per_cell
    if needed > len(templates):
        raise ValueError(
            f"Need {needed} unique source templates for {len(cells)} cells, only {len(templates)} available"
        )
    shuffled = list(templates)
    random.Random(seed).shuffle(shuffled)
    candidates: list[Candidate] = []
    cursor = 0
    for slot in range(candidates_per_cell):
        for clean_verb, corrupt_verb in cells:
            source = shuffled[cursor]
            cursor += 1
            clean_text = render_verb(source, clean_verb)
            corrupt_text = render_verb(source, corrupt_verb)
            candidate_id = f"{source.source_sample_id}__{clean_verb}__{corrupt_verb}"
            candidates.append(
                Candidate(
                    candidate_id=candidate_id,
                    source=source,
                    clean_verb=clean_verb,
                    corrupt_verb=corrupt_verb,
                    clean_text=clean_text,
                    corrupt_text=corrupt_text,
                )
            )
    if len({item.source.source_sample_id for item in candidates}) != len(candidates):
        raise AssertionError("Candidate construction reused a source body")
    return candidates


def model_input_device(model) -> torch.device:
    try:
        return model.get_input_embeddings().weight.device
    except Exception:
        return next(model.parameters()).device


def load_model_and_tokenizer(spec: ModelSpec, dtype_name: str):
    if not spec.model_path.is_dir():
        raise FileNotFoundError(f"Model path missing for {spec.key}: {spec.model_path}")
    dtype = getattr(torch, dtype_name)
    tokenizer = AutoTokenizer.from_pretrained(str(spec.model_path), trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError(f"Tokenizer for {spec.key} has neither pad nor eos token")
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    if spec.loader == "mistral":
        model = Mistral3ForConditionalGeneration.from_pretrained(
            str(spec.model_path),
            dtype=dtype,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            str(spec.model_path),
            torch_dtype=dtype,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
    model.eval()
    marker_ids = tokenizer.encode(spec.tool_marker, add_special_tokens=False)
    if len(marker_ids) != 1:
        converted = tokenizer.convert_tokens_to_ids(spec.tool_marker)
        if converted is None or converted < 0 or converted == tokenizer.unk_token_id:
            raise ValueError(f"{spec.key}: tool marker {spec.tool_marker!r} is not one token: {marker_ids}")
        marker_ids = [int(converted)]
    return model, tokenizer, int(marker_ids[0])


def forward_logits(model, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    kwargs = {"input_ids": input_ids, "attention_mask": attention_mask, "use_cache": False, "return_dict": True}
    try:
        outputs = model(**kwargs, logits_to_keep=1)
    except TypeError:
        outputs = model(**kwargs)
    logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
    if logits.ndim != 3:
        raise RuntimeError(f"Expected [batch, sequence, vocab] logits, received {tuple(logits.shape)}")
    return logits[:, -1, :].float()


def evaluate_texts(
    model,
    tokenizer,
    texts: Sequence[str],
    *,
    tool_token_id: int,
    batch_size: int,
    progress_label: str,
) -> list[dict[str, Any]]:
    device = model_input_device(model)
    results: list[dict[str, Any]] = []
    total_batches = math.ceil(len(texts) / batch_size)
    for batch_index, start in enumerate(range(0, len(texts), batch_size), start=1):
        batch_texts = list(texts[start : start + batch_size])
        batch = tokenizer(batch_texts, add_special_tokens=False, padding=True, return_tensors="pt")
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        with torch.inference_mode():
            logits = forward_logits(model, input_ids, attention_mask)
        target = logits[:, tool_token_id]
        top1 = logits.argmax(dim=-1)
        ranks = (logits > target.unsqueeze(-1)).sum(dim=-1) + 1
        non_target = logits.clone()
        non_target[:, tool_token_id] = -torch.inf
        best_non_target = non_target.max(dim=-1).values
        probs = torch.softmax(logits, dim=-1)[:, tool_token_id]
        for offset in range(logits.shape[0]):
            token_id = int(top1[offset].item())
            results.append(
                {
                    "tool_logit": float(target[offset].item()),
                    "tool_probability": float(probs[offset].item()),
                    "tool_rank": int(ranks[offset].item()),
                    "top1_token_id": token_id,
                    "top1_token_text": tokenizer.decode([token_id], clean_up_tokenization_spaces=False),
                    "is_tool_top1": bool(token_id == tool_token_id),
                    "margin_vs_best_non_tool": float((target[offset] - best_non_target[offset]).item()),
                }
            )
        del batch, input_ids, attention_mask, logits, target, top1, ranks, non_target, best_non_target, probs
        if batch_index == total_batches or batch_index % max(total_batches // 10, 1) == 0:
            print(f"{progress_label}: {batch_index}/{total_batches} batches", flush=True)
    return results


def evaluate_candidates(
    model,
    tokenizer,
    candidates: Sequence[Candidate],
    *,
    tool_token_id: int,
    batch_size: int,
) -> list[dict[str, Any]]:
    clean = evaluate_texts(
        model,
        tokenizer,
        [item.clean_text for item in candidates],
        tool_token_id=tool_token_id,
        batch_size=batch_size,
        progress_label="clean screening",
    )
    corrupt = evaluate_texts(
        model,
        tokenizer,
        [item.corrupt_text for item in candidates],
        tool_token_id=tool_token_id,
        batch_size=batch_size,
        progress_label="corrupt screening",
    )
    rows: list[dict[str, Any]] = []
    for item, clean_row, corrupt_row in zip(candidates, clean, corrupt, strict=True):
        behavior_valid = bool(clean_row["is_tool_top1"]) and not bool(corrupt_row["is_tool_top1"])
        quality = float(clean_row["margin_vs_best_non_tool"]) - float(corrupt_row["margin_vs_best_non_tool"])
        rows.append(
            {
                "candidate_id": item.candidate_id,
                "source_sample_id": item.source.source_sample_id,
                "source_filename": item.source.filename,
                "source_dataset": item.source.dataset_name,
                "source_language": item.source.language,
                "source_split": item.source.source_split,
                "clean_verb": item.clean_verb,
                "corrupt_verb": item.corrupt_verb,
                "clean_prompt_sha256": sha256_text(item.clean_text),
                "corrupt_prompt_sha256": sha256_text(item.corrupt_text),
                "clean_tool_logit": clean_row["tool_logit"],
                "clean_tool_probability": clean_row["tool_probability"],
                "clean_tool_rank": clean_row["tool_rank"],
                "clean_top1_token_id": clean_row["top1_token_id"],
                "clean_top1_token_text": clean_row["top1_token_text"],
                "clean_is_tool_top1": clean_row["is_tool_top1"],
                "clean_margin_vs_best_non_tool": clean_row["margin_vs_best_non_tool"],
                "corrupt_tool_logit": corrupt_row["tool_logit"],
                "corrupt_tool_probability": corrupt_row["tool_probability"],
                "corrupt_tool_rank": corrupt_row["tool_rank"],
                "corrupt_top1_token_id": corrupt_row["top1_token_id"],
                "corrupt_top1_token_text": corrupt_row["top1_token_text"],
                "corrupt_is_tool_top1": corrupt_row["is_tool_top1"],
                "corrupt_margin_vs_best_non_tool": corrupt_row["margin_vs_best_non_tool"],
                "behavior_valid": behavior_valid,
                "quality": quality,
            }
        )
    return rows


def balanced_quotas(total: int, names: Sequence[str]) -> dict[str, int]:
    if not names:
        raise ValueError("Cannot balance an empty vocabulary")
    base, remainder = divmod(total, len(names))
    return {name: base + int(index < remainder) for index, name in enumerate(names)}


def min_cost_balanced_selection(
    valid_rows: Sequence[dict[str, Any]],
    *,
    clean_verbs: Sequence[str],
    corrupt_verbs: Sequence[str],
    total: int,
) -> list[dict[str, Any]]:
    clean_quota = balanced_quotas(total, clean_verbs)
    corrupt_quota = balanced_quotas(total, corrupt_verbs)
    graph = nx.DiGraph()
    source = "source"
    sink = "sink"
    for clean, quota in clean_quota.items():
        graph.add_edge(source, f"clean::{clean}", capacity=quota, weight=0)
    for corrupt, quota in corrupt_quota.items():
        graph.add_edge(f"corrupt::{corrupt}", sink, capacity=quota, weight=0)
    by_id: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(valid_rows):
        clean = str(row["clean_verb"])
        corrupt = str(row["corrupt_verb"])
        if clean not in clean_quota or corrupt not in corrupt_quota:
            continue
        node = f"candidate::{index}"
        quality_cost = -int(round(float(row["quality"]) * 1000.0))
        graph.add_edge(f"clean::{clean}", node, capacity=1, weight=quality_cost)
        graph.add_edge(node, f"corrupt::{corrupt}", capacity=1, weight=0)
        by_id[node] = row
    flow = nx.max_flow_min_cost(graph, source, sink)
    value = sum(int(flow[source].get(f"clean::{clean}", 0)) for clean in clean_verbs)
    if value != total:
        available = Counter((str(row["clean_verb"]), str(row["corrupt_verb"])) for row in valid_rows)
        raise RuntimeError(
            f"Only {value}/{total} valid pairs satisfy the requested verb balance. Available valid cells: {dict(sorted(available.items()))}"
        )
    selected: list[dict[str, Any]] = []
    for node, row in by_id.items():
        if int(flow.get(node, {}).get(f"corrupt::{row['corrupt_verb']}", 0)) == 1:
            selected.append(row)
    if len(selected) != total:
        raise AssertionError(f"Flow reported {value} but selected {len(selected)} rows")
    if Counter(str(row["clean_verb"]) for row in selected) != Counter(clean_quota):
        raise AssertionError("Clean verb quota mismatch")
    if Counter(str(row["corrupt_verb"]) for row in selected) != Counter(corrupt_quota):
        raise AssertionError("Corrupt verb quota mismatch")
    return selected


def choose_largest_feasible_vocabulary(
    valid_rows: Sequence[dict[str, Any]],
    *,
    preferred_clean_verbs: Sequence[str],
    preferred_corrupt_verbs: Sequence[str],
    total: int,
) -> tuple[list[dict[str, Any]], tuple[str, ...], tuple[str, ...]]:
    """Keep the largest balanced verb inventory that can remain behavior-pure.

    A model can make one otherwise sensible execution verb unreliable.  The
    correct response is to omit that verb for this model's v5 dataset, never to
    admit a clean/non-call or corrupt/call example merely to fill a cell.
    """

    successes: list[tuple[tuple[int, int, int, float], list[dict[str, Any]], tuple[str, ...], tuple[str, ...]]] = []
    for n_clean in range(len(preferred_clean_verbs), 1, -1):
        for clean_subset in itertools.combinations(preferred_clean_verbs, n_clean):
            for n_corrupt in range(len(preferred_corrupt_verbs), 1, -1):
                for corrupt_subset in itertools.combinations(preferred_corrupt_verbs, n_corrupt):
                    try:
                        selected = min_cost_balanced_selection(
                            valid_rows,
                            clean_verbs=clean_subset,
                            corrupt_verbs=corrupt_subset,
                            total=total,
                        )
                    except RuntimeError:
                        continue
                    score = sum(float(row["quality"]) for row in selected)
                    # Maximize the usable vocabulary first, then favor keeping
                    # both sides broad and finally prefer stronger examples.
                    rank = (len(clean_subset) * len(corrupt_subset), len(corrupt_subset), len(clean_subset), score)
                    successes.append((rank, selected, clean_subset, corrupt_subset))
        if successes:
            break
    if not successes:
        raise RuntimeError("No behavior-pure, balanced 500-pair vocabulary can be formed from the screened candidates")
    _, selected, clean_verbs, corrupt_verbs = max(successes, key=lambda item: item[0])
    return selected, clean_verbs, corrupt_verbs


def allocate_train_per_cell(
    selected: Sequence[dict[str, Any]],
    *,
    clean_verbs: Sequence[str],
    corrupt_verbs: Sequence[str],
    train_total: int,
) -> dict[tuple[str, str], int]:
    clean_quota = balanced_quotas(train_total, clean_verbs)
    corrupt_quota = balanced_quotas(train_total, corrupt_verbs)
    available = Counter((str(row["clean_verb"]), str(row["corrupt_verb"])) for row in selected)
    graph = nx.DiGraph()
    source = "source"
    sink = "sink"
    for clean, quota in clean_quota.items():
        graph.add_edge(source, f"clean::{clean}", capacity=quota)
    for corrupt, quota in corrupt_quota.items():
        graph.add_edge(f"corrupt::{corrupt}", sink, capacity=quota)
    for clean in clean_verbs:
        for corrupt in corrupt_verbs:
            graph.add_edge(f"clean::{clean}", f"corrupt::{corrupt}", capacity=available[(clean, corrupt)])
    flow_value, flow = nx.maximum_flow(graph, source, sink)
    if flow_value != train_total:
        raise RuntimeError(f"Could not allocate an exactly balanced {train_total}-pair train split")
    allocation = {
        (clean, corrupt): int(flow[f"clean::{clean}"].get(f"corrupt::{corrupt}", 0))
        for clean in clean_verbs
        for corrupt in corrupt_verbs
    }
    return allocation


def split_selected(
    selected: Sequence[dict[str, Any]],
    *,
    clean_verbs: Sequence[str],
    corrupt_verbs: Sequence[str],
    seed: int,
) -> list[dict[str, Any]]:
    per_cell: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in selected:
        per_cell[(str(row["clean_verb"]), str(row["corrupt_verb"]))].append(dict(row))
    train_allocation = allocate_train_per_cell(
        selected,
        clean_verbs=clean_verbs,
        corrupt_verbs=corrupt_verbs,
        train_total=200,
    )
    output: list[dict[str, Any]] = []
    randomizer = random.Random(seed)
    for cell, rows in sorted(per_cell.items()):
        rows.sort(key=lambda row: (-float(row["quality"]), str(row["candidate_id"])))
        # Randomize only exact-quality ties, keeping stronger examples preferred.
        grouped: dict[float, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            grouped[float(row["quality"])].append(row)
        ordered: list[dict[str, Any]] = []
        for quality in sorted(grouped, reverse=True):
            tie = grouped[quality]
            randomizer.shuffle(tie)
            ordered.extend(tie)
        train_count = train_allocation[cell]
        for index, row in enumerate(ordered):
            row["split"] = "train" if index < train_count else "heldout"
            output.append(row)
    if len(output) != 500:
        raise AssertionError(f"Expected 500 selected rows, found {len(output)}")
    if sum(row["split"] == "train" for row in output) != 200:
        raise AssertionError("Train size mismatch")
    if sum(row["split"] == "heldout" for row in output) != 300:
        raise AssertionError("Held-out size mismatch")
    for split, expected in (("train", 200), ("heldout", 300)):
        rows = [row for row in output if row["split"] == split]
        if Counter(str(row["clean_verb"]) for row in rows) != Counter(balanced_quotas(expected, clean_verbs)):
            raise AssertionError(f"{split} clean balance mismatch")
        if Counter(str(row["corrupt_verb"]) for row in rows) != Counter(balanced_quotas(expected, corrupt_verbs)):
            raise AssertionError(f"{split} corrupt balance mismatch")
    return output


def summary_for_rows(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return {
        "n_pairs": len(rows),
        "clean_tool_call_top1_rate": sum(bool(row["clean_is_tool_top1"]) for row in rows) / max(len(rows), 1),
        "corrupt_tool_call_top1_rate": sum(bool(row["corrupt_is_tool_top1"]) for row in rows) / max(len(rows), 1),
        "mean_clean_tool_probability": sum(float(row["clean_tool_probability"]) for row in rows) / max(len(rows), 1),
        "mean_corrupt_tool_probability": sum(float(row["corrupt_tool_probability"]) for row in rows) / max(len(rows), 1),
        "minimum_clean_margin": min(float(row["clean_margin_vs_best_non_tool"]) for row in rows),
        "minimum_corrupt_non_tool_margin": min(-float(row["corrupt_margin_vs_best_non_tool"]) for row in rows),
        "clean_verb_counts": dict(sorted(Counter(str(row["clean_verb"]) for row in rows).items())),
        "corrupt_verb_counts": dict(sorted(Counter(str(row["corrupt_verb"]) for row in rows).items())),
        "verb_pair_counts": {
            f"{clean}__{corrupt}": count
            for (clean, corrupt), count in sorted(Counter((str(row["clean_verb"]), str(row["corrupt_verb"])) for row in rows).items())
        },
        "source_dataset_counts": dict(sorted(Counter(str(row["source_dataset"]) for row in rows).items())),
        "source_language_counts": dict(sorted(Counter(str(row["source_language"]) for row in rows).items())),
    }


def write_dataset(
    output_root: Path,
    *,
    spec: ModelSpec,
    tool_token_id: int,
    selected: Sequence[dict[str, Any]],
    candidate_map: dict[str, Candidate],
    seed: int,
    candidates_per_cell: int,
    selected_clean_verbs: Sequence[str],
    selected_corrupt_verbs: Sequence[str],
    min_clean_margin: float,
    min_corrupt_non_tool_margin: float,
) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=False)
    manifest_rows: list[dict[str, Any]] = []
    for split in ("train", "heldout"):
        (output_root / split / "clean").mkdir(parents=True)
        (output_root / split / "corrupt").mkdir(parents=True)
    for index, row in enumerate(sorted(selected, key=lambda item: (str(item["split"]), str(item["clean_verb"]), str(item["corrupt_verb"]), str(item["candidate_id"]))), start=1):
        candidate = candidate_map[str(row["candidate_id"])]
        split = str(row["split"])
        filename = f"{index:03d}_{candidate.source.source_sample_id}.txt"
        clean_path = output_root / split / "clean" / filename
        corrupt_path = output_root / split / "corrupt" / filename
        clean_path.write_text(candidate.clean_text, encoding="utf-8")
        corrupt_path.write_text(candidate.corrupt_text, encoding="utf-8")
        if clean_path.read_text(encoding="utf-8") != candidate.clean_text or corrupt_path.read_text(encoding="utf-8") != candidate.corrupt_text:
            raise RuntimeError(f"Write verification failed for {filename}")
        manifest_rows.append(
            {
                "sample_id": f"v5_{spec.key}_{index:03d}",
                "split": split,
                "filename": filename,
                "clean_relpath": str(clean_path.relative_to(output_root)),
                "corrupt_relpath": str(corrupt_path.relative_to(output_root)),
                "source_sample_id": candidate.source.source_sample_id,
                "source_filename": candidate.source.filename,
                "source_dataset": candidate.source.dataset_name,
                "source_language": candidate.source.language,
                "source_split": candidate.source.source_split,
                "source_clean_path": str(candidate.source.clean_path),
                "source_corrupt_path": str(candidate.source.corrupt_path),
                "clean_candidate": candidate.clean_verb,
                "corrupt_candidate": candidate.corrupt_verb,
                "tool_call_marker": spec.tool_marker,
                "tool_call_token_id": tool_token_id,
                "clean_prompt_sha256": row["clean_prompt_sha256"],
                "corrupt_prompt_sha256": row["corrupt_prompt_sha256"],
                "clean_tool_logit": row["clean_tool_logit"],
                "clean_tool_probability": row["clean_tool_probability"],
                "clean_tool_rank": row["clean_tool_rank"],
                "clean_top1_token_id": row["clean_top1_token_id"],
                "clean_top1_token_text": row["clean_top1_token_text"],
                "clean_is_tool_top1": row["clean_is_tool_top1"],
                "clean_margin_vs_best_non_tool": row["clean_margin_vs_best_non_tool"],
                "corrupt_tool_logit": row["corrupt_tool_logit"],
                "corrupt_tool_probability": row["corrupt_tool_probability"],
                "corrupt_tool_rank": row["corrupt_tool_rank"],
                "corrupt_top1_token_id": row["corrupt_top1_token_id"],
                "corrupt_top1_token_text": row["corrupt_top1_token_text"],
                "corrupt_is_tool_top1": row["corrupt_is_tool_top1"],
                "corrupt_margin_vs_best_non_tool": row["corrupt_margin_vs_best_non_tool"],
                "selection_quality": row["quality"],
            }
        )
    write_jsonl(output_root / "manifest.jsonl", manifest_rows)
    split_summaries = {split: summary_for_rows([row for row in selected if row["split"] == split]) for split in ("train", "heldout")}
    all_summary = summary_for_rows(selected)
    for split, summary in split_summaries.items():
        if summary["clean_tool_call_top1_rate"] != 1.0 or summary["corrupt_tool_call_top1_rate"] != 0.0:
            raise AssertionError(f"{spec.key}/{split} contains a behavior-invalid selected row")
    summary = {
        "schema_version": 1,
        "dataset_version": "v5_model_specific_balanced",
        "model_key": spec.key,
        "model_label": spec.label,
        "model_path": str(spec.model_path),
        "source_root": str(spec.source_root),
        "tool_call_marker": spec.tool_marker,
        "tool_call_token_id": tool_token_id,
        "seed": seed,
        "candidate_per_verb_cell": candidates_per_cell,
        "selected_clean_verbs": list(selected_clean_verbs),
        "selected_corrupt_verbs": list(selected_corrupt_verbs),
        "selection_rule": "Target-model first-token clean-call/corrupt-non-call screen with required safety margins, followed by maximum-quality exact marginal verb balancing.",
        "screening_margin_requirements": {
            "minimum_clean_tool_minus_best_non_tool": min_clean_margin,
            "minimum_corrupt_best_non_tool_minus_tool": min_corrupt_non_tool_margin,
        },
        "n_pairs": len(selected),
        "n_train": len([row for row in selected if row["split"] == "train"]),
        "n_heldout": len([row for row in selected if row["split"] == "heldout"]),
        "all": all_summary,
        "splits": split_summaries,
    }
    write_json(output_root / "summary.json", summary)
    readme = "\n".join(
        [
            f"# {spec.label} v5 dataset",
            "",
            "This dataset is target-model-specific and behavior-screened.",
            "",
            f"- Pairs: `{summary['n_pairs']}` (`{summary['n_train']}` train, `{summary['n_heldout']}` held-out).",
            f"- Native first-token call marker: `{spec.tool_marker}` (ID `{tool_token_id}`).",
            "- Admission rule: clean marker is top-1 and corrupt marker is not top-1.",
            "- Verb counts are balanced independently on clean and corrupt sides in every split.",
            "- `manifest.jsonl` contains prompt hashes and the complete screening metrics for every selected pair.",
            "",
        ]
    )
    (output_root / "README.md").write_text(readme, encoding="utf-8")
    return summary


def release_model(model) -> None:
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def build_one(
    spec: ModelSpec,
    *,
    output_root: Path,
    seed: int,
    dtype_name: str,
    candidate_multiplier: float,
    min_clean_margin: float,
    min_corrupt_non_tool_margin: float,
    skip_model_load: bool,
) -> None:
    model_root = output_root / spec.key
    if model_root.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {model_root}")
    templates = load_templates(spec)
    candidate_count = max(20, int(round(spec.candidates_per_cell * candidate_multiplier)))
    candidates = build_candidates(
        templates,
        clean_verbs=spec.clean_verbs,
        corrupt_verbs=spec.corrupt_verbs,
        candidates_per_cell=candidate_count,
        seed=seed + sum(ord(char) for char in spec.key),
    )
    if skip_model_load:
        print(f"[{spec.key}] source validation passed: {len(templates)} templates, {len(candidates)} candidates", flush=True)
        return
    print(f"[{spec.key}] loading {spec.label}", flush=True)
    model, tokenizer, tool_token_id = load_model_and_tokenizer(spec, dtype_name)
    try:
        screening = evaluate_candidates(
            model,
            tokenizer,
            candidates,
            tool_token_id=tool_token_id,
            batch_size=spec.batch_size,
        )
    finally:
        release_model(model)
    valid = [
        row
        for row in screening
        if bool(row["behavior_valid"])
        and float(row["clean_margin_vs_best_non_tool"]) > min_clean_margin
        and -float(row["corrupt_margin_vs_best_non_tool"]) > min_corrupt_non_tool_margin
    ]
    print(
        f"[{spec.key}] valid target behavior with clean>{min_clean_margin:g}, "
        f"corrupt-non-tool>{min_corrupt_non_tool_margin:g}: {len(valid)}/{len(screening)} candidates",
        flush=True,
    )
    selected, selected_clean_verbs, selected_corrupt_verbs = choose_largest_feasible_vocabulary(
        valid,
        preferred_clean_verbs=spec.clean_verbs,
        preferred_corrupt_verbs=spec.corrupt_verbs,
        total=500,
    )
    print(
        f"[{spec.key}] selected stable vocabulary: clean={list(selected_clean_verbs)}, "
        f"corrupt={list(selected_corrupt_verbs)}",
        flush=True,
    )
    selected = split_selected(
        selected,
        clean_verbs=selected_clean_verbs,
        corrupt_verbs=selected_corrupt_verbs,
        seed=seed + 1,
    )
    candidate_map = {item.candidate_id: item for item in candidates}
    summary = write_dataset(
        model_root,
        spec=spec,
        tool_token_id=tool_token_id,
        selected=selected,
        candidate_map=candidate_map,
        seed=seed,
        candidates_per_cell=candidate_count,
        selected_clean_verbs=selected_clean_verbs,
        selected_corrupt_verbs=selected_corrupt_verbs,
        min_clean_margin=min_clean_margin,
        min_corrupt_non_tool_margin=min_corrupt_non_tool_margin,
    )
    write_csv(model_root / "candidate_screening.csv", screening)
    print(
        f"[{spec.key}] complete: {summary['n_train']} train / {summary['n_heldout']} held-out, "
        f"clean={summary['all']['clean_tool_call_top1_rate']:.1%}, "
        f"corrupt={summary['all']['corrupt_tool_call_top1_rate']:.1%}",
        flush=True,
    )


def main() -> None:
    args = parse_args()
    if "mistral_3p2_24b" in args.models:
        raise SystemExit(
            "Mistral uses native chat-template input IDs. Run "
            "`python -m paper_repro.build_v5_mistral_tau2` instead of the generic builder."
        )
    if args.min_clean_margin < 0.0 or args.min_corrupt_non_tool_margin < 0.0:
        raise ValueError("Margin requirements must be non-negative")
    output_root = args.output_root.resolve()
    if output_root.exists() and not args.resume:
        raise FileExistsError(f"Output root already exists: {output_root}")
    if output_root == PROJECT_ROOT or output_root == PROJECT_ROOT / "datasets":
        raise ValueError("Output root must be a new child directory, not the repository or datasets root")
    output_root.mkdir(parents=True, exist_ok=bool(args.resume))
    try:
        for model_key in args.models:
            build_one(
                MODEL_SPECS[model_key],
                output_root=output_root,
                seed=int(args.seed),
                dtype_name=str(args.dtype),
                candidate_multiplier=float(args.candidate_multiplier),
                min_clean_margin=float(args.min_clean_margin),
                min_corrupt_non_tool_margin=float(args.min_corrupt_non_tool_margin),
                skip_model_load=bool(args.skip_model_load),
            )
    except Exception:
        # Preserve partial completed model directories for inspection, but do not
        # leave an ambiguous top-level success marker.
        raise
    write_json(
        output_root / "completion.json",
        {
            "status": "complete",
            "dataset_version": "v5_model_specific_balanced",
            "models": list(args.models),
            "seed": int(args.seed),
            "screening_margin_requirements": {
                "minimum_clean_tool_minus_best_non_tool": float(args.min_clean_margin),
                "minimum_corrupt_best_non_tool_minus_tool": float(args.min_corrupt_non_tool_margin),
            },
        },
    )


if __name__ == "__main__":
    main()
