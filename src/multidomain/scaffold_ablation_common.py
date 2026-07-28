#!/usr/bin/env python3
"""Shared, auditable helpers for the rebuttal scaffold ablations.

This module centralizes the non-model operations that otherwise make an
ablation easy to get subtly wrong: parsing the exact three original
system-prompt components, retaining the native Qwen chat boundary, loading
released pair manifests, and reporting full-distribution first-token metrics.

No helper in this file performs behavioral selection.  In particular,
``neutral_verb_for`` is a deterministic assignment over already-fixed sample
IDs and never observes a model logit or probability.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch
from tqdm.auto import tqdm

from multiscale_common import build_pair_batches, clear_cuda, load_sample_pairs
from run_cross_scale_fix_gate_and_patch import (
    collect_residuals_at_hook,
    compute_pca,
    hook_name,
    make_last_token_add_hook,
)


SYSTEM_OPEN = "<|im_start|>system\n"
TOOLS_OPEN = "<tools>\n"
FORMAT_OPEN = "For each function call, return a json object with function name and arguments within "
SYSTEM_TO_USER = "<|im_end|>\n<|im_start|>user\n"
USER_OPEN = "<|im_start|>user\n"
ASSISTANT_BOUNDARY = "<|im_end|>\n<|im_start|>assistant\n"

# These are fixed *before* any scaffold-ablation forward pass.  They are the
# five examples listed in the pre-registered plan and are assigned in a
# balanced round robin by stable sample-ID order.
NEUTRAL_VERBS = ("Consider", "Handle", "Take", "Use", "Process")
V5_DATASET_VERSION = "v5_model_specific_balanced"


@dataclass(frozen=True)
class FixedPair:
    """One immutable v4 clean/corrupt pair, normalized across file layouts."""

    order: int
    sample_id: str
    clean_text: str
    corrupt_text: str
    clean_tokens_cpu: torch.Tensor
    corrupt_tokens_cpu: torch.Tensor
    token_len: int
    clean_candidate: str | None
    corrupt_candidate: str | None


@dataclass(frozen=True)
class PromptParts:
    """Exact byte ranges of the original Qwen system scaffold and user turn."""

    system_open: str
    role_instructions: str
    tool_schema_block: str
    format_template: str
    system_to_user: str
    user_content: str
    assistant_suffix: str

    def original(self) -> str:
        return (
            self.system_open
            + self.role_instructions
            + self.tool_schema_block
            + self.format_template
            + self.system_to_user
            + self.user_content
            + self.assistant_suffix
        )

    def render(
        self,
        *,
        user_content: str,
        include_role: bool,
        include_tool_schema: bool,
        include_format: bool,
        tool_schema_override: str | None = None,
    ) -> str:
        if tool_schema_override is not None and include_tool_schema:
            raise ValueError("A tool-schema override is only valid when the original T component is absent")
        tool_block = (
            self.tool_schema_block
            if include_tool_schema
            else (tool_schema_override if tool_schema_override is not None else "")
        )
        return (
            self.system_open
            + (self.role_instructions if include_role else "")
            + tool_block
            + (self.format_template if include_format else "")
            + self.system_to_user
            + user_content
            + self.assistant_suffix
        )

    def render_without_system(self, *, user_content: str) -> str:
        """Keep only Qwen's mandatory user/assistant chat boundary for L7."""

        return USER_OPEN + user_content + self.assistant_suffix


@dataclass
class PromptItem:
    sample_id: str
    prompt: str
    tokens_cpu: torch.Tensor
    token_len: int
    metadata: dict[str, Any]


@dataclass
class RenderedPair:
    order: int
    sample_id: str
    clean_text: str
    corrupt_text: str
    clean_tokens_cpu: torch.Tensor
    corrupt_tokens_cpu: torch.Tensor
    token_len: int
    clean_candidate: str | None
    corrupt_candidate: str | None


@dataclass
class MetricAccumulator:
    n: int = 0
    tool_logit_sum: float = 0.0
    tool_prob_sum: float = 0.0
    top1_count: int = 0
    ranks: list[int] = field(default_factory=list)

    def update(self, stats: dict[str, torch.Tensor], *, tool_token_id: int) -> None:
        self.n += int(stats["tool_logit"].shape[0])
        self.tool_logit_sum += float(stats["tool_logit"].sum().item())
        self.tool_prob_sum += float(stats["tool_prob"].sum().item())
        self.top1_count += int((stats["top1"] == tool_token_id).sum().item())
        self.ranks.extend(int(value) for value in stats["tool_rank"].tolist())

    def summary(self) -> dict[str, Any]:
        if self.n <= 0:
            raise ValueError("Cannot summarize zero prompts")
        ranks = sorted(self.ranks)
        mid = len(ranks) // 2
        median = float(ranks[mid]) if len(ranks) % 2 else (ranks[mid - 1] + ranks[mid]) / 2.0
        return {
            "n": self.n,
            "mean_tool_call_logit": self.tool_logit_sum / self.n,
            "mean_tool_call_prob": self.tool_prob_sum / self.n,
            "tool_call_top1_rate": self.top1_count / self.n,
            "mean_tool_call_rank": sum(ranks) / self.n,
            "median_tool_call_rank": median,
            "tool_call_top3_rate": sum(rank <= 3 for rank in ranks) / self.n,
            "rank_buckets": {
                "rank_1": sum(rank == 1 for rank in ranks),
                "rank_2": sum(rank == 2 for rank in ranks),
                "rank_3": sum(rank == 3 for rank in ranks),
                "rank_4_to_10": sum(4 <= rank <= 10 for rank in ranks),
                "rank_above_10": sum(rank > 10 for rank in ranks),
            },
        }


def parse_prompt_parts(prompt: str) -> PromptParts:
    """Split an original v4 prompt into the pre-registered R/T/F components."""

    if not prompt.startswith(SYSTEM_OPEN):
        raise ValueError("Expected the original Qwen system turn")
    tools_start = prompt.find(TOOLS_OPEN, len(SYSTEM_OPEN))
    if tools_start < 0:
        raise ValueError("Could not find original <tools> block")
    format_start = prompt.find(FORMAT_OPEN, tools_start)
    if format_start < 0:
        raise ValueError("Could not find original function-call format template")
    system_to_user_start = prompt.find(SYSTEM_TO_USER, format_start)
    if system_to_user_start < 0:
        raise ValueError("Could not find original system-to-user boundary")
    user_content_start = system_to_user_start + len(SYSTEM_TO_USER)
    assistant_start = prompt.find(ASSISTANT_BOUNDARY, user_content_start)
    if assistant_start < 0:
        raise ValueError("Could not find original assistant boundary")
    parts = PromptParts(
        system_open=SYSTEM_OPEN,
        role_instructions=prompt[len(SYSTEM_OPEN) : tools_start],
        tool_schema_block=prompt[tools_start:format_start],
        format_template=prompt[format_start:system_to_user_start],
        system_to_user=SYSTEM_TO_USER,
        user_content=prompt[user_content_start:assistant_start],
        assistant_suffix=prompt[assistant_start:],
    )
    if parts.original() != prompt:
        raise AssertionError("Prompt component parser did not reconstruct the source prompt byte-for-byte")
    return parts


def parse_user_content(prompt: str) -> str:
    return parse_prompt_parts(prompt).user_content


def split_instruction_and_body(user_content: str) -> tuple[str, str]:
    line_end = user_content.find("\n")
    if line_end < 0:
        raise ValueError("Expected an instruction line followed by a task body")
    return user_content[:line_end], user_content[line_end + 1 :]


def replace_leading_user_verb(user_content: str, verb: str) -> str:
    """Replace only the first word of an already-rendered user instruction."""

    normalized = verb.strip()
    if not normalized or any(char.isspace() for char in normalized):
        raise ValueError(f"Expected one non-empty verb, got {verb!r}")
    space = user_content.find(" ")
    line_end = user_content.find("\n")
    if space < 0 or (line_end >= 0 and space > line_end):
        raise ValueError(f"Could not identify a leading verb in {user_content[:120]!r}")
    return normalized + user_content[space:]


def neutral_verb_for(sample_id: str, ordered_sample_ids: Sequence[str]) -> str:
    """Assign the fixed neutral vocabulary in balanced, stable sample-ID order."""

    try:
        index = list(ordered_sample_ids).index(sample_id)
    except ValueError as exc:
        raise KeyError(f"Unknown fixed sample ID: {sample_id}") from exc
    return NEUTRAL_VERBS[index % len(NEUTRAL_VERBS)]


def balanced_neutral_assignment(pairs: Sequence[FixedPair]) -> dict[str, str]:
    ordered_ids = sorted(pair.sample_id for pair in pairs)
    if len(set(ordered_ids)) != len(ordered_ids):
        raise ValueError("Fixed split contains duplicate sample IDs")
    return {sample_id: neutral_verb_for(sample_id, ordered_ids) for sample_id in ordered_ids}


def validate_neutral_tokenization(tokenizer) -> dict[str, Any]:
    """Verify the predeclared neutral vocabulary is one Qwen token per word."""

    rows: list[dict[str, Any]] = []
    for verb in NEUTRAL_VERBS:
        ids = tokenizer.encode(verb, add_special_tokens=False)
        rows.append({"verb": verb, "token_ids": [int(value) for value in ids], "token_count": len(ids)})
    invalid = [row for row in rows if int(row["token_count"]) != 1]
    if invalid:
        raise ValueError(f"Predeclared neutral verbs are not all single tokens: {invalid}")
    return {
        "verbs": list(NEUTRAL_VERBS),
        "assignment": "stable sample-ID order, round robin; no behavior-based selection",
        "tokenization": rows,
        "semantic_protocol": (
            "Predeclared independent semantic annotation: no produce/modify or explain/evaluate meaning; "
            "the vocabulary was fixed before inspecting p_call."
        ),
    }


def _coerce_pair(source: Any, order: int) -> FixedPair:
    return FixedPair(
        order=int(order),
        sample_id=str(source.sample_id),
        clean_text=str(source.clean_text),
        corrupt_text=str(source.corrupt_text),
        clean_tokens_cpu=source.clean_tokens_cpu.detach().cpu(),
        corrupt_tokens_cpu=source.corrupt_tokens_cpu.detach().cpu(),
        token_len=int(source.token_len),
        clean_candidate=str(source.clean_candidate) if source.clean_candidate is not None else None,
        corrupt_candidate=str(source.corrupt_candidate) if source.corrupt_candidate is not None else None,
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"{path}: expected a JSON object")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"{path}:{line_number}: expected a JSON object")
            rows.append(value)
    return rows


def is_v5_model_specific_dataset(dataset_root: Path) -> bool:
    summary_path = dataset_root.resolve() / "summary.json"
    if not summary_path.is_file():
        return False
    return _read_json(summary_path).get("dataset_version") == V5_DATASET_VERSION


def _v5_split_name(split: str) -> str:
    return "heldout" if split == "test" else split


def _v5_manifest_rows(dataset_root: Path) -> list[dict[str, Any]]:
    manifest_path = dataset_root / "manifest.jsonl"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing v5 manifest: {manifest_path}")
    rows = _read_jsonl(manifest_path)
    if not rows:
        raise ValueError(f"{manifest_path}: empty manifest")
    required = {
        "sample_id",
        "split",
        "clean_relpath",
        "corrupt_relpath",
        "clean_prompt_sha256",
        "corrupt_prompt_sha256",
        "clean_candidate",
        "corrupt_candidate",
        "tool_call_marker",
        "tool_call_token_id",
    }
    for index, row in enumerate(rows, start=1):
        missing = sorted(required - set(row))
        if missing:
            raise ValueError(f"{manifest_path}:{index}: manifest is missing required fields {missing}")
    return rows


def v5_dataset_provenance(dataset_root: Path, *, splits: Sequence[str]) -> dict[str, Any]:
    """Return immutable v5 release metadata for result-side provenance."""

    dataset_root = dataset_root.resolve()
    summary_path = dataset_root / "summary.json"
    if not is_v5_model_specific_dataset(dataset_root):
        raise ValueError(f"{dataset_root} is not a {V5_DATASET_VERSION} dataset")
    summary = _read_json(summary_path)
    rows = _v5_manifest_rows(dataset_root)
    requested_splits = tuple(dict.fromkeys(_v5_split_name(split) for split in splits))
    split_rows: dict[str, list[dict[str, Any]]] = {}
    for split in requested_splits:
        selected = [row for row in rows if str(row["split"]) == split]
        if not selected:
            raise ValueError(f"{dataset_root}: no manifest rows for split={split!r}")
        split_rows[split] = selected

    file_paths = {
        "release_readme": dataset_root.parent / "README.md",
        "dataset_readme": dataset_root / "README.md",
        "summary": summary_path,
        "manifest": dataset_root / "manifest.jsonl",
    }
    for label, path in file_paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"{dataset_root}: missing {label} file {path}")

    selected_manifest_rows = {
        split: [
            {
                key: row[key]
                for key in (
                    "sample_id",
                    "split",
                    "filename",
                    "clean_relpath",
                    "corrupt_relpath",
                    "source_sample_id",
                    "source_dataset",
                    "source_language",
                    "clean_candidate",
                    "corrupt_candidate",
                    "clean_prompt_sha256",
                    "corrupt_prompt_sha256",
                    "tool_call_marker",
                    "tool_call_token_id",
                )
                if key in row
            }
            for row in split_rows[split]
        ]
        for split in requested_splits
    }
    return {
        "dataset_version": summary["dataset_version"],
        "dataset_root": str(dataset_root),
        "model_key": summary.get("model_key"),
        "model_label": summary.get("model_label"),
        "model_path": summary.get("model_path"),
        "tool_call_marker": summary.get("tool_call_marker"),
        "tool_call_token_id": summary.get("tool_call_token_id"),
        "release_files": {
            label: {"path": str(path), "sha256": sha256_file(path)} for label, path in file_paths.items()
        },
        "declared_split_counts": {name: details.get("n_pairs") for name, details in summary.get("splits", {}).items()},
        "manifest_split_counts": {split: len(split_rows[split]) for split in requested_splits},
        "selected_manifest_rows": selected_manifest_rows,
    }


def validate_v5_model_compatibility(
    provenance: dict[str, Any], *, model_path: Path, model_label: str, tool_token_id: int
) -> None:
    """Reject accidental use of a v5 split with a non-matching model/checkpoint."""

    expected_path = Path(str(provenance["model_path"])).resolve()
    actual_path = model_path.resolve()
    if actual_path != expected_path:
        raise ValueError(f"v5 dataset expects model_path={expected_path}, received {actual_path}")
    expected_label = str(provenance["model_label"])
    if model_label != expected_label:
        raise ValueError(f"v5 dataset expects model_label={expected_label!r}, received {model_label!r}")
    expected_token_id = int(provenance["tool_call_token_id"])
    if int(tool_token_id) != expected_token_id:
        raise ValueError(f"v5 dataset expects tool token {expected_token_id}, received {tool_token_id}")


def validate_v5_requested_split_sizes(provenance: dict[str, Any], *, requested_counts: dict[str, int]) -> None:
    """Require a released v5 run to consume each requested split in full."""

    declared_counts = provenance["declared_split_counts"]
    for split, requested_count in requested_counts.items():
        declared_count = declared_counts.get(_v5_split_name(split))
        if declared_count is None:
            raise ValueError(f"v5 release does not declare split={split!r}")
        if int(requested_count) != int(declared_count):
            raise ValueError(
                f"v5 split {split!r} contains {declared_count} pairs; requested {requested_count}. "
                "Use the complete released split for this rerun."
            )


def _load_v5_pairs(model, *, dataset_root: Path, split: str, max_pairs: int) -> list[FixedPair]:
    summary = _read_json(dataset_root / "summary.json")
    if str(summary.get("model_key")) == "mistral_3p2_24b":
        raise ValueError(
            "The v5 Mistral split requires stored native input_ids or apply_chat_template(tokenize=True); "
            "this Qwen text-tokenization loader intentionally refuses it."
        )
    selected_split = _v5_split_name(split)
    rows = [row for row in _v5_manifest_rows(dataset_root) if str(row["split"]) == selected_split]
    if not rows:
        raise ValueError(f"{dataset_root}: no rows for split={selected_split!r}")
    if max_pairs > 0:
        rows = rows[:max_pairs]

    pairs: list[FixedPair] = []
    for order, row in enumerate(rows, start=1):
        prompt_texts: dict[str, str] = {}
        for side in ("clean", "corrupt"):
            relative = Path(str(row[f"{side}_relpath"]))
            path = (dataset_root / relative).resolve()
            try:
                path.relative_to(dataset_root)
            except ValueError as exc:
                raise ValueError(f"{row['sample_id']}: prompt path escapes dataset root: {relative}") from exc
            if not path.is_file():
                raise FileNotFoundError(f"{row['sample_id']}: missing {side} prompt {path}")
            text = path.read_text(encoding="utf-8")
            actual_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
            expected_hash = str(row[f"{side}_prompt_sha256"])
            if actual_hash != expected_hash:
                raise ValueError(f"{row['sample_id']}: {side} prompt SHA-256 does not match manifest")
            prompt_texts[side] = text
        clean_tokens = model.to_tokens(prompt_texts["clean"], prepend_bos=False).detach().cpu()
        corrupt_tokens = model.to_tokens(prompt_texts["corrupt"], prepend_bos=False).detach().cpu()
        if clean_tokens.shape != corrupt_tokens.shape:
            raise ValueError(f"{row['sample_id']}: clean/corrupt token shapes differ")
        pairs.append(
            FixedPair(
                order=order,
                sample_id=str(row["sample_id"]),
                clean_text=prompt_texts["clean"],
                corrupt_text=prompt_texts["corrupt"],
                clean_tokens_cpu=clean_tokens,
                corrupt_tokens_cpu=corrupt_tokens,
                token_len=int(clean_tokens.shape[-1]),
                clean_candidate=str(row["clean_candidate"]),
                corrupt_candidate=str(row["corrupt_candidate"]),
            )
        )
    return pairs


def load_fixed_pairs(model, *, dataset_root: Path, split: str, max_pairs: int) -> list[FixedPair]:
    """Load a released split without any post-hoc behavioral filtering.

    v5 Qwen datasets use one root manifest with ``train`` and ``heldout``
    records. Legacy v4 layouts remain supported for historical reproduction.
    """

    dataset_root = dataset_root.resolve()
    if is_v5_model_specific_dataset(dataset_root):
        return _load_v5_pairs(model, dataset_root=dataset_root, split=split, max_pairs=max_pairs)

    manifest_path = dataset_root / split / "clean" / "manifest.jsonl"
    if manifest_path.exists():
        raw_pairs = load_sample_pairs(model, dataset_root=dataset_root, split=split, max_pairs=max_pairs)
        return [_coerce_pair(pair, index) for index, pair in enumerate(raw_pairs, start=1)]

    selected_path = dataset_root / "selected_pairs.jsonl"
    if not selected_path.exists():
        raise FileNotFoundError(
            f"{dataset_root}: neither {manifest_path.relative_to(dataset_root)} nor selected_pairs.jsonl exists"
        )
    pairs: list[FixedPair] = []
    with selected_path.open("r", encoding="utf-8") as handle:
        for row in (json.loads(line) for line in handle if line.strip()):
            if str(row.get("split")) != split:
                continue
            clean_text = str(row.get("clean_prompt", ""))
            corrupt_text = str(row.get("corrupt_prompt", ""))
            if not clean_text or not corrupt_text:
                raise ValueError(f"{dataset_root}: selected pair lacks rendered prompt text")
            clean_tokens = model.to_tokens(clean_text, prepend_bos=False).detach().cpu()
            corrupt_tokens = model.to_tokens(corrupt_text, prepend_bos=False).detach().cpu()
            if clean_tokens.shape != corrupt_tokens.shape:
                raise ValueError(f"{dataset_root}/{row.get('candidate_id')}: clean/corrupt token shapes differ")
            pairs.append(
                FixedPair(
                    order=len(pairs) + 1,
                    sample_id=str(row.get("candidate_id") or row.get("source_id") or len(pairs) + 1),
                    clean_text=clean_text,
                    corrupt_text=corrupt_text,
                    clean_tokens_cpu=clean_tokens,
                    corrupt_tokens_cpu=corrupt_tokens,
                    token_len=int(clean_tokens.shape[-1]),
                    clean_candidate=str(row.get("clean_verb")) if row.get("clean_verb") is not None else None,
                    corrupt_candidate=str(row.get("corrupt_verb")) if row.get("corrupt_verb") is not None else None,
                )
            )
            if len(pairs) >= int(max_pairs):
                break
    if not pairs:
        raise ValueError(f"{dataset_root}: no rows for split={split!r}")
    return pairs


def validate_fixed_pairs(pairs: Sequence[FixedPair], *, expected_count: int, label: str) -> dict[str, Any]:
    if len(pairs) != int(expected_count):
        raise ValueError(f"{label}: expected {expected_count} fixed pairs, found {len(pairs)}")
    sample_ids = [pair.sample_id for pair in pairs]
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError(f"{label}: duplicate sample IDs")
    differing_positions: set[int] = set()
    for pair in pairs:
        if pair.clean_tokens_cpu.shape != pair.corrupt_tokens_cpu.shape:
            raise ValueError(f"{label}/{pair.sample_id}: unequal clean/corrupt token shapes")
        diff = (pair.clean_tokens_cpu != pair.corrupt_tokens_cpu).nonzero(as_tuple=False)
        if int(diff.shape[0]) != 1:
            raise ValueError(
                f"{label}/{pair.sample_id}: expected exactly one differing original input token, found {diff.shape[0]}"
            )
        differing_positions.add(int(diff[0, -1].item()))
    return {
        "pair_count": len(pairs),
        "sample_id_count": len(set(sample_ids)),
        "token_length_min": min(pair.token_len for pair in pairs),
        "token_length_max": max(pair.token_len for pair in pairs),
        "unique_differing_token_positions": sorted(differing_positions),
    }


def validate_invariant_scaffold(pairs: Sequence[FixedPair], *, label: str) -> PromptParts:
    if not pairs:
        raise ValueError(f"{label}: no pairs")
    reference = parse_prompt_parts(pairs[0].clean_text)
    reference_chunks = (
        reference.system_open,
        reference.role_instructions,
        reference.tool_schema_block,
        reference.format_template,
        reference.system_to_user,
        reference.assistant_suffix,
    )
    for pair in pairs:
        for side, text in (("clean", pair.clean_text), ("corrupt", pair.corrupt_text)):
            parts = parse_prompt_parts(text)
            chunks = (
                parts.system_open,
                parts.role_instructions,
                parts.tool_schema_block,
                parts.format_template,
                parts.system_to_user,
                parts.assistant_suffix,
            )
            if chunks != reference_chunks:
                raise ValueError(f"{label}/{pair.sample_id}/{side}: scaffold is not invariant")
    return reference


def token_stats(logits: torch.Tensor, tool_token_id: int) -> dict[str, torch.Tensor]:
    final_logits = logits[:, -1, :].float()
    tool_logit = final_logits[:, tool_token_id]
    tool_prob = torch.softmax(final_logits, dim=-1)[:, tool_token_id]
    top1 = final_logits.argmax(dim=-1)
    tool_rank = (final_logits > tool_logit.unsqueeze(-1)).sum(dim=-1) + 1
    return {
        "tool_logit": tool_logit.detach().cpu(),
        "tool_prob": tool_prob.detach().cpu(),
        "top1": top1.detach().cpu(),
        "tool_rank": tool_rank.detach().cpu(),
    }


def flatten_metrics(prefix: str, metrics: dict[str, Any]) -> dict[str, Any]:
    return {f"{prefix}_{key}": value for key, value in metrics.items() if key != "rank_buckets"}


def prompt_batches(items: Sequence[PromptItem], batch_size: int) -> Iterable[list[PromptItem]]:
    buckets: dict[tuple[str, int], list[PromptItem]] = defaultdict(list)
    for item in items:
        group = str(item.metadata.get("request_type") or item.metadata.get("condition") or "all")
        buckets[(group, item.token_len)].append(item)
    for _group, _token_len in sorted(buckets):
        bucket = buckets[(_group, _token_len)]
        for start in range(0, len(bucket), max(int(batch_size), 1)):
            yield bucket[start : start + max(int(batch_size), 1)]


def evaluate_prompt_items(
    model,
    items: Sequence[PromptItem],
    *,
    batch_size: int,
    tool_token_id: int,
    progress_label: str,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    """Measure all four first-token metrics for arbitrary single prompts."""

    accumulators: dict[str, MetricAccumulator] = defaultdict(MetricAccumulator)
    rows: list[dict[str, Any]] = []
    batches = list(prompt_batches(items, batch_size))
    progress = tqdm(batches, desc=progress_label, dynamic_ncols=True)
    for batch in progress:
        tokens = torch.cat([item.tokens_cpu for item in batch], dim=0).to(model.W_U.device)
        with torch.no_grad():
            logits = model(tokens)
        stats = token_stats(logits, tool_token_id)
        for index, item in enumerate(batch):
            condition = str(item.metadata.get("condition") or item.metadata.get("request_type") or "all")
            individual = {
                "tool_logit": stats["tool_logit"][index : index + 1],
                "tool_prob": stats["tool_prob"][index : index + 1],
                "top1": stats["top1"][index : index + 1],
                "tool_rank": stats["tool_rank"][index : index + 1],
            }
            accumulators[condition].update(individual, tool_token_id=tool_token_id)
            rows.append(
                {
                    "sample_id": item.sample_id,
                    "token_length": item.token_len,
                    **item.metadata,
                    "tool_call_logit": float(individual["tool_logit"][0].item()),
                    "tool_call_prob": float(individual["tool_prob"][0].item()),
                    "tool_call_rank": int(individual["tool_rank"][0].item()),
                    "tool_call_top1": bool(int(individual["top1"][0].item()) == tool_token_id),
                    "top1_token_id": int(individual["top1"][0].item()),
                }
            )
        del tokens, logits, stats
        progress.set_postfix(tok=batch[0].token_len)
    return {key: accumulator.summary() for key, accumulator in accumulators.items()}, rows


def render_pair_for_scaffold(
    model,
    pair: FixedPair,
    *,
    clean_parts: PromptParts,
    corrupt_parts: PromptParts,
    include_role: bool,
    include_tool_schema: bool,
    include_format: bool,
    tool_schema_override: str | None = None,
) -> RenderedPair:
    clean_text = clean_parts.render(
        user_content=clean_parts.user_content,
        include_role=include_role,
        include_tool_schema=include_tool_schema,
        include_format=include_format,
        tool_schema_override=tool_schema_override,
    )
    corrupt_text = corrupt_parts.render(
        user_content=corrupt_parts.user_content,
        include_role=include_role,
        include_tool_schema=include_tool_schema,
        include_format=include_format,
        tool_schema_override=tool_schema_override,
    )
    clean_tokens = model.to_tokens(clean_text, prepend_bos=False).detach().cpu()
    corrupt_tokens = model.to_tokens(corrupt_text, prepend_bos=False).detach().cpu()
    if clean_tokens.shape != corrupt_tokens.shape:
        raise ValueError(f"{pair.sample_id}: scaffold rendering broke clean/corrupt token alignment")
    return RenderedPair(
        order=pair.order,
        sample_id=pair.sample_id,
        clean_text=clean_text,
        corrupt_text=corrupt_text,
        clean_tokens_cpu=clean_tokens,
        corrupt_tokens_cpu=corrupt_tokens,
        token_len=int(clean_tokens.shape[-1]),
        clean_candidate=pair.clean_candidate,
        corrupt_candidate=pair.corrupt_candidate,
    )


def validate_rendered_pairs(
    pairs: Sequence[RenderedPair], *, expected_count: int, label: str
) -> dict[str, Any]:
    if len(pairs) != int(expected_count):
        raise ValueError(f"{label}: expected {expected_count} pairs, found {len(pairs)}")
    differing_positions: set[int] = set()
    for pair in pairs:
        differences = (pair.clean_tokens_cpu != pair.corrupt_tokens_cpu).nonzero(as_tuple=False)
        if int(differences.shape[0]) != 1:
            raise ValueError(
                f"{label}/{pair.sample_id}: expected one differing input token after rendering, found {differences.shape[0]}"
            )
        differing_positions.add(int(differences[0, -1].item()))
    return {
        "pair_count": len(pairs),
        "token_length_min": min(pair.token_len for pair in pairs),
        "token_length_max": max(pair.token_len for pair in pairs),
        "unique_differing_token_positions": sorted(differing_positions),
    }


def collect_variant_vector(
    model,
    pairs: Sequence[RenderedPair],
    *,
    layer: int,
    hook_kind: str,
    batch_size: int,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    clean_resid, corrupt_resid = collect_residuals_at_hook(
        model, pairs, layer=layer, hook_kind=hook_kind, batch_size=batch_size
    )
    differences = clean_resid - corrupt_resid
    pca = compute_pca(differences, n_components=10)
    vector = pca["mean_diff"].detach().cpu().float().view(-1).contiguous()
    if vector.numel() != int(model.cfg.d_model) or not bool(torch.isfinite(vector).all()):
        raise ValueError("Invalid native clean-minus-corrupt vector")
    if float(vector.norm().item()) <= 0.0:
        raise ValueError("Native clean-minus-corrupt vector has zero norm")
    del clean_resid, corrupt_resid, differences
    clear_cuda()
    return vector, pca


def cosine_similarity(left: torch.Tensor, right: torch.Tensor) -> float:
    left_unit = left.float() / left.float().norm().clamp_min(1e-12)
    right_unit = right.float() / right.float().norm().clamp_min(1e-12)
    return float(torch.dot(left_unit, right_unit).clamp(-1.0, 1.0).item())


def _normalized_effect(numerator: float, denominator: float) -> float | None:
    if not math.isfinite(numerator) or not math.isfinite(denominator) or denominator <= 1e-8:
        return None
    return numerator / denominator


def evaluate_frozen_vector_causal(
    model,
    pairs: Sequence[RenderedPair],
    *,
    layer: int,
    hook_kind: str,
    frozen_vector: torch.Tensor,
    batch_size: int,
    tool_token_id: int,
    scaffold_label: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Evaluate behavior and original-vector add/remove within one scaffold cell."""

    clean_metrics = MetricAccumulator()
    corrupt_metrics = MetricAccumulator()
    add_metrics = MetricAccumulator()
    remove_metrics = MetricAccumulator()
    corrupt_non_tool = 0
    clean_tool = 0
    add_flips = 0
    remove_drops = 0
    rows: list[dict[str, Any]] = []
    residual_name = hook_name(layer, hook_kind)
    add_hook = (residual_name, make_last_token_add_hook(frozen_vector))
    remove_hook = (residual_name, make_last_token_add_hook(-frozen_vector))
    batches = build_pair_batches(pairs, batch_size=batch_size)
    progress = tqdm(batches, desc=f"{scaffold_label}: exec/analysis + frozen vector", dynamic_ncols=True)
    for batch in progress:
        clean_tokens = batch.clean_tokens_cpu.to(model.W_U.device)
        corrupt_tokens = batch.corrupt_tokens_cpu.to(model.W_U.device)
        with torch.no_grad():
            clean_logits = model(clean_tokens)
            corrupt_logits = model(corrupt_tokens)
            add_logits = model.run_with_hooks(corrupt_tokens, fwd_hooks=[add_hook])
            remove_logits = model.run_with_hooks(clean_tokens, fwd_hooks=[remove_hook])
        statistics = {
            "execution": token_stats(clean_logits, tool_token_id),
            "analysis": token_stats(corrupt_logits, tool_token_id),
            "orig_mu_add_to_analysis": token_stats(add_logits, tool_token_id),
            "orig_mu_remove_from_execution": token_stats(remove_logits, tool_token_id),
        }
        clean_metrics.update(statistics["execution"], tool_token_id=tool_token_id)
        corrupt_metrics.update(statistics["analysis"], tool_token_id=tool_token_id)
        add_metrics.update(statistics["orig_mu_add_to_analysis"], tool_token_id=tool_token_id)
        remove_metrics.update(statistics["orig_mu_remove_from_execution"], tool_token_id=tool_token_id)
        execution_top1 = statistics["execution"]["top1"] == tool_token_id
        analysis_top1 = statistics["analysis"]["top1"] == tool_token_id
        add_top1 = statistics["orig_mu_add_to_analysis"]["top1"] == tool_token_id
        remove_top1 = statistics["orig_mu_remove_from_execution"]["top1"] == tool_token_id
        clean_tool += int(execution_top1.sum().item())
        corrupt_non_tool += int((~analysis_top1).sum().item())
        add_flips += int(((~analysis_top1) & add_top1).sum().item())
        remove_drops += int((execution_top1 & (~remove_top1)).sum().item())
        for local_index, pair_index in enumerate(batch.indices):
            pair = pairs[int(pair_index)]
            for condition, stats in statistics.items():
                rows.append(
                    {
                        "scaffold": scaffold_label,
                        "sample_id": pair.sample_id,
                        "condition": condition,
                        "token_length": pair.token_len,
                        "tool_call_logit": float(stats["tool_logit"][local_index].item()),
                        "tool_call_prob": float(stats["tool_prob"][local_index].item()),
                        "tool_call_rank": int(stats["tool_rank"][local_index].item()),
                        "tool_call_top1": bool(int(stats["top1"][local_index].item()) == tool_token_id),
                        "top1_token_id": int(stats["top1"][local_index].item()),
                    }
                )
        del (
            clean_tokens,
            corrupt_tokens,
            clean_logits,
            corrupt_logits,
            add_logits,
            remove_logits,
            statistics,
            execution_top1,
            analysis_top1,
            add_top1,
            remove_top1,
        )
        progress.set_postfix(tok=batch.token_len)
    execution = clean_metrics.summary()
    analysis = corrupt_metrics.summary()
    added = add_metrics.summary()
    removed = remove_metrics.summary()
    gap = float(execution["mean_tool_call_logit"] - analysis["mean_tool_call_logit"])
    return (
        {
            "baseline": {"execution": execution, "analysis": analysis},
            "frozen_original_intervention": {
                "add_to_analysis": added,
                "remove_from_execution": removed,
                "add_strict_flip_rate": add_flips / max(corrupt_non_tool, 1),
                "remove_strict_drop_rate": remove_drops / max(clean_tool, 1),
                "condition_logit_gap_execution_minus_analysis": gap,
                "normalization_valid": gap > 1e-8,
                "sufficiency_normalized_logit_gap": _normalized_effect(
                    float(added["mean_tool_call_logit"] - analysis["mean_tool_call_logit"]), gap
                ),
                "necessity_normalized_logit_gap": _normalized_effect(
                    float(execution["mean_tool_call_logit"] - removed["mean_tool_call_logit"]), gap
                ),
            },
        },
        rows,
    )


def build_length_matched_neutral_block(model, parts: PromptParts) -> dict[str, Any]:
    """Replace T with non-tool English while preserving the full prompt length.

    The search is token-length based on the actual Qwen tokenizer and full
    prompt, not character length.  The resulting text deliberately avoids
    tool, function-call, XML, and format language.
    """

    original_tokens = model.to_tokens(parts.original(), prepend_bos=False)
    target_length = int(original_tokens.shape[-1])
    forbidden = ("tool", "function", "call", "format", "<", ">")
    bases = (
        "Background information is available for general context.",
        "General background material is available.",
        "Ordinary context is provided for reference.",
        "Background notes are included here.",
        "Context is available.",
        "",
    )
    fillers = (" background", " context", " detail", " note", " material", " reference", ".")
    candidates_checked = 0
    for base in bases:
        for filler in fillers:
            for count in range(0, max(target_length * 3, 64)):
                # Preserve the original T block's visual boundary before F.
                # The replacement is still ordinary English only, but F must
                # not become the suffix of the final English word.
                block = base + filler * count + "\n\n"
                if not block.strip() or any(token in block.casefold() for token in forbidden):
                    continue
                candidate = parts.render(
                    user_content=parts.user_content,
                    include_role=True,
                    include_tool_schema=False,
                    include_format=True,
                    tool_schema_override=block,
                )
                length = int(model.to_tokens(candidate, prepend_bos=False).shape[-1])
                candidates_checked += 1
                if length == target_length:
                    return {
                        "block": block,
                        "target_full_prompt_token_length": target_length,
                        "matched_full_prompt_token_length": length,
                        "base": base,
                        "filler": filler,
                        "repetitions": count,
                        "candidates_checked": candidates_checked,
                        "forbidden_substrings": list(forbidden),
                    }
    raise RuntimeError("Could not construct an exact-token-length neutral replacement for the T component")
