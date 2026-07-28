#!/usr/bin/env python3
"""Build a fresh, model-specific 200+200 τ² Telecom collection.

This is a *data-construction* program, not an intervention runner.  For each
target model it independently renders natural τ² histories with that model's
native tool template and retains two baseline-defined sets:

* 200 histories whose next decision is a native tool-call opening; and
* 200 histories whose next decision is strongly non-tool.

The source Qwen3.5 rollout is used only to find natural multi-turn histories.
The target model's own unmodified first decision defines the label.  Task
success/reward is retained only as source provenance and is never used for
selection.  Final sets contain at most one item per task and have disjoint
task IDs across the two directions.  Existing 50-example τ² sets are excluded
by task ID, so this collection is deliberately independent of them.

Run ``--tier primary`` first.  It screens one natural candidate per τ² task
for each arm.  If a model is short of 200 in an arm, run ``--tier supplement``
to screen alternate natural turns from the same raw trajectory pool; the final
selector still permits only one turn per task.
"""

from __future__ import annotations

import argparse
import collections
import gc
import hashlib
import importlib.util
import json
import os
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import run_tau2_cross_family as cross  # noqa: E402


RAW_ROOT = PROJECT_ROOT / "datasets" / "external" / "tau2_telecom_qwen35_9b" / "raw"
PREPARED_ROOT = PROJECT_ROOT / "datasets" / "external" / "tau2_telecom_qwen35_9b" / "prepared"
DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT
    / "datasets"
    / "external"
    / "tau2_telecom_qwen35_9b"
    / "collections"
    / "bidirectional_200_per_model_20260726"
)
DEFAULT_COLLECTION_NAME = "tau2_bidirectional_200_per_model_20260726"
SEED = 20260726
TOOL_CALL_TEXT = "<tool_call>"
MODEL_ROOT = Path(os.environ.get("MODEL_ROOT", PROJECT_ROOT / "external" / "models")).expanduser()


def model_path(environment_variable: str, directory_name: str) -> Path:
    """Resolve checkpoints from public configuration rather than one workstation."""

    return Path(os.environ.get(environment_variable, MODEL_ROOT / directory_name)).expanduser()


@dataclass(frozen=True)
class TargetSpec:
    key: str
    display_name: str
    model_path: Path
    family: str
    tool_token_text: str
    batch_size: int
    max_batch_tokens: int


SPECS: dict[str, TargetSpec] = {
    "qwen3_4b": TargetSpec(
        "qwen3_4b",
        "Qwen3-4B",
        model_path("QWEN3_4B_PATH", "Qwen3-4B"),
        "qwen3",
        TOOL_CALL_TEXT,
        4,
        56_000,
    ),
    "qwen3_8b": TargetSpec(
        "qwen3_8b",
        "Qwen3-8B",
        model_path("QWEN3_8B_PATH", "Qwen3-8B"),
        "qwen3",
        TOOL_CALL_TEXT,
        2,
        48_000,
    ),
    "qwen3_14b": TargetSpec(
        "qwen3_14b",
        "Qwen3-14B",
        model_path("QWEN3_14B_PATH", "Qwen3-14B"),
        "qwen3",
        TOOL_CALL_TEXT,
        2,
        40_000,
    ),
    "qwen35_4b": TargetSpec(
        "qwen35_4b",
        "Qwen3.5-4B",
        model_path("QWEN35_4B_PATH", "Qwen3.5-4B"),
        "qwen35",
        TOOL_CALL_TEXT,
        2,
        48_000,
    ),
    "qwen35_9b": TargetSpec(
        "qwen35_9b",
        "Qwen3.5-9B",
        model_path("QWEN35_9B_PATH", "Qwen3.5-9B"),
        "qwen35",
        TOOL_CALL_TEXT,
        2,
        40_000,
    ),
    "granite": TargetSpec(
        "granite",
        "Granite-3.3-8B-Instruct",
        model_path("GRANITE_3P3_8B_PATH", "granite-3.3-8b-instruct"),
        "granite",
        "<|tool_call|>",
        2,
        40_000,
    ),
    "mistral": TargetSpec(
        "mistral",
        "Mistral-Small-3.2-24B-Instruct-2506",
        model_path("MISTRAL_3P2_24B_PATH", "Mistral-Small-3.2-24B-Instruct-2506"),
        "mistral",
        "[TOOL_CALLS]",
        1,
        30_000,
    ),
    "devstral": TargetSpec(
        "devstral",
        "Devstral-Small-2-24B-Instruct-2512",
        model_path("DEVSTRAL_2_24B_PATH", "Devstral-Small-2-24B-Instruct-2512"),
        "devstral",
        "[TOOL_CALLS]",
        1,
        30_000,
    ),
}

# A full 1,600-by-1,600 scan is unnecessary for high-yield arms, but the
# scarce arms need a wider deterministic screen before we can know whether a
# supplementary pass is required.  Limits are *source rows*, after excluding
# the model's legacy-50 task IDs.  Zero means the complete primary pool.
PRIMARY_SCREEN_LIMITS: dict[str, dict[str, int]] = {
    "qwen3_4b": {"tool": 300, "direct": 300},
    "qwen3_8b": {"tool": 300, "direct": 300},
    "qwen3_14b": {"tool": 300, "direct": 300},
    # Qwen3.5-4B has a substantially lower natural tool-opener rate than the
    # Qwen3 checkpoints, so reserve a wider fixed primary sample.  This is
    # still task-stratified and remains well below the full pool.
    "qwen35_4b": {"tool": 900, "direct": 300},
    # Initial screening shows Qwen3.5-9B to be the sparse call-decision case;
    # audit the complete task-stratified primary pool rather than weaken the
    # top-1 tool-opener criterion.
    "qwen35_9b": {"tool": 0, "direct": 300},
    "granite": {"tool": 0, "direct": 300},
    "mistral": {"tool": 300, "direct": 0},
    "devstral": {"tool": 300, "direct": 900},
}


@dataclass(frozen=True)
class TauResources:
    system_prompt: str
    tools: list[dict[str, Any]]


@dataclass(frozen=True)
class PreparedPrefix:
    row: dict[str, Any]
    input_ids: list[int]
    adapter_audit: dict[str, Any]

    @property
    def length(self) -> int:
        return len(self.input_ids)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        default="all",
        help="Comma-separated model keys, or 'all'.",
    )
    parser.add_argument("--tier", choices=("primary", "closure", "broad"), required=True)
    parser.add_argument(
        "--arms",
        choices=("tool", "direct", "both"),
        default="both",
        help="Which arm(s) to screen in this pass.  Final selection is always re-audited for both arms.",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--collection-name",
        default=DEFAULT_COLLECTION_NAME,
        help="Audit label written into each model's metadata and selection summary.",
    )
    parser.add_argument("--raw-root", type=Path, default=RAW_ROOT)
    parser.add_argument("--prepared-root", type=Path, default=PREPARED_ROOT)
    parser.add_argument("--final-count", type=int, default=200)
    parser.add_argument("--max-context-tokens", type=int, default=30_000)
    parser.add_argument("--max-non-tool-probability", type=float, default=0.05)
    parser.add_argument("--render-chunk-size", type=int, default=100)
    parser.add_argument(
        "--source-limit",
        type=int,
        default=None,
        help="Override the per-model/per-arm adaptive source limit; zero means all available rows.",
    )
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                parsed = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}") from exc
            if not isinstance(parsed, dict):
                raise ValueError(f"Expected an object at {path}:{line_number}")
            rows.append(parsed)
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_rank(*parts: object) -> int:
    value = "|".join(str(part) for part in parts)
    return int.from_bytes(hashlib.sha256(value.encode("utf-8")).digest()[:8], "big")


def read_resources(raw_root: Path) -> TauResources:
    system_prompt = (raw_root / "tau2_system_prompt.txt").read_text(encoding="utf-8").strip()
    tools = json.loads((raw_root / "tau2_tool_schemas.json").read_text(encoding="utf-8"))
    if not system_prompt or not isinstance(tools, list) or not tools:
        raise ValueError("Invalid τ² system prompt or tool schemas")
    return TauResources(system_prompt=system_prompt, tools=tools)


def qwen_transformers() -> tuple[Any, Any]:
    """Match the narrow sklearn-import guard used by the existing Qwen runs."""
    original_find_spec = importlib.util.find_spec

    def patched_find_spec(name: str, package: str | None = None):
        if name == "sklearn" and os.environ.get("MECH_ENABLE_TRANSFORMERS_SKLEARN", "0") != "1":
            return None
        return original_find_spec(name, package)

    importlib.util.find_spec = patched_find_spec
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    finally:
        importlib.util.find_spec = original_find_spec
    return AutoModelForCausalLM, AutoTokenizer


def load_model_and_tokenizer(spec: TargetSpec) -> tuple[Any, Any, torch.device]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for τ² baseline screening")
    if spec.family in {"qwen3", "qwen35"}:
        AutoModelForCausalLM, AutoTokenizer = qwen_transformers()
        tokenizer = AutoTokenizer.from_pretrained(str(spec.model_path), trust_remote_code=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        model = AutoModelForCausalLM.from_pretrained(
            str(spec.model_path),
            torch_dtype=torch.bfloat16,
            device_map={"": 0},
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        ).eval()
        return model, tokenizer, torch.device("cuda:0")
    model, tokenizer, device = cross.load_tokenizer_and_model(cross.SPECS[spec.key])
    return model, tokenizer, device


def resolve_tool_token(tokenizer: Any, spec: TargetSpec) -> tuple[int, dict[str, Any]]:
    encoded = [int(token) for token in tokenizer.encode(spec.tool_token_text, add_special_tokens=False)]
    converted = int(tokenizer.convert_tokens_to_ids(spec.tool_token_text))
    if converted < 0:
        raise RuntimeError(f"Could not resolve native tool opener {spec.tool_token_text!r}")
    if spec.family in {"qwen3", "qwen35", "granite", "devstral"} and len(encoded) != 1:
        raise RuntimeError(
            f"{spec.display_name} native tool opener must encode as one token, got {encoded}"
        )
    return converted, {
        "semantic_tool_opening": spec.tool_token_text,
        "tool_token_id_via_convert": converted,
        "tool_token_ids_via_encode": encoded,
        "tool_token_encode_length": len(encoded),
        "decision_position": "final native prompt token / first generation decision",
    }


def adapt_qwen35_messages(row: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    source = row.get("messages")
    if not isinstance(source, list):
        raise TypeError(f"{row.get('candidate_id')} has no messages")
    messages: list[dict[str, Any]] = []
    historical_calls = 0
    for raw in source:
        role = str(raw["role"])
        converted: dict[str, Any] = {"role": role, "content": cross.safe_content(raw.get("content"))}
        calls = raw.get("tool_calls") or []
        if role == "assistant" and calls:
            converted_calls: list[dict[str, Any]] = []
            for call in calls:
                converted_calls.append(
                    {
                        "type": "function",
                        "function": {
                            "name": cross.function_name(call),
                            "arguments": cross.function_arguments(call),
                        },
                    }
                )
            converted["tool_calls"] = converted_calls
            historical_calls += len(converted_calls)
        messages.append(converted)
    return messages, {
        "adapter": "qwen35_native_tool_history",
        "historical_calls_serialized": historical_calls,
        "thinking_disabled_at_generation_prompt": True,
    }


def as_ids(value: Any) -> list[int]:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().tolist()
    if value and isinstance(value[0], list):
        value = value[0]
    return [int(token) for token in value]


def render_prefix(
    row: dict[str, Any],
    *,
    tokenizer: Any,
    spec: TargetSpec,
    resources: TauResources,
    max_context_tokens: int,
) -> tuple[PreparedPrefix | None, dict[str, Any] | None]:
    try:
        if spec.family == "qwen3":
            messages = [{"role": "system", "content": resources.system_prompt}, *row["messages"]]
            try:
                rendered = tokenizer.apply_chat_template(
                    messages,
                    tools=resources.tools,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            except TypeError:
                rendered = tokenizer.apply_chat_template(
                    messages,
                    tools=resources.tools,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            ids = as_ids(tokenizer(rendered, add_special_tokens=False)["input_ids"])
            adapter = {
                "adapter": "qwen3_native_tool_history",
                "thinking_disabled_at_generation_prompt": True,
            }
        elif spec.family == "qwen35":
            native_messages, adapter = adapt_qwen35_messages(row)
            encoded = tokenizer.apply_chat_template(
                [{"role": "system", "content": resources.system_prompt}, *native_messages],
                tools=resources.tools,
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=False,
                return_dict=True,
                return_tensors="pt",
            )
            ids = as_ids(encoded["input_ids"])
        else:
            native_messages, adapter = cross.adapt_tau2_messages(row, cross.SPECS[spec.key])
            messages = [{"role": "system", "content": resources.system_prompt}, *native_messages]
            if spec.family == "granite":
                try:
                    rendered = tokenizer.apply_chat_template(
                        messages,
                        tools=resources.tools,
                        tokenize=False,
                        add_generation_prompt=True,
                        thinking=False,
                    )
                except TypeError:
                    rendered = tokenizer.apply_chat_template(
                        messages,
                        tools=resources.tools,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                ids = as_ids(tokenizer(rendered, add_special_tokens=False)["input_ids"])
            else:
                encoded = tokenizer.apply_chat_template(
                    messages,
                    tools=resources.tools,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_dict=True,
                    return_tensors="pt",
                )
                ids = as_ids(encoded["input_ids"])
        if not ids:
            raise ValueError("Native template produced an empty prompt")
        if len(ids) > max_context_tokens:
            return None, {
                "candidate_id": row.get("candidate_id"),
                "reason": "context_too_long",
                "token_count": len(ids),
                "max_context_tokens": max_context_tokens,
                "adapter": adapter,
            }
        return PreparedPrefix(row=row, input_ids=ids, adapter_audit=adapter), None
    except Exception as exc:
        return None, {
            "candidate_id": row.get("candidate_id"),
            "reason": "render_error",
            "detail": repr(exc),
        }


def make_batches(
    prefixes: Sequence[PreparedPrefix], *, batch_size: int, max_batch_tokens: int
) -> list[list[PreparedPrefix]]:
    ordered = sorted(prefixes, key=lambda prefix: (prefix.length, str(prefix.row["candidate_id"])))
    batches: list[list[PreparedPrefix]] = []
    current: list[PreparedPrefix] = []
    current_length = 0
    for prefix in ordered:
        next_length = max(current_length, prefix.length)
        if current and (len(current) >= batch_size or (len(current) + 1) * next_length > max_batch_tokens):
            batches.append(current)
            current = []
            current_length = 0
        current.append(prefix)
        current_length = next_length
    if current:
        batches.append(current)
    return batches


def make_left_padded_batch(
    prefixes: Sequence[PreparedPrefix], *, pad_token_id: int, device: torch.device
) -> dict[str, torch.Tensor]:
    width = max(prefix.length for prefix in prefixes)
    input_ids = torch.full((len(prefixes), width), int(pad_token_id), dtype=torch.long)
    attention_mask = torch.zeros((len(prefixes), width), dtype=torch.long)
    for index, prefix in enumerate(prefixes):
        ids = torch.tensor(prefix.input_ids, dtype=torch.long)
        input_ids[index, -prefix.length :] = ids
        attention_mask[index, -prefix.length :] = 1
    return {"input_ids": input_ids.to(device), "attention_mask": attention_mask.to(device)}


def model_last_logits(model: Any, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
    with torch.inference_mode():
        try:
            output = model(**inputs, use_cache=False, logits_to_keep=1, return_dict=True)
        except TypeError:
            output = model(**inputs, use_cache=False, return_dict=True)
    logits = output.logits
    if logits.ndim != 3:
        raise RuntimeError(f"Unexpected logits shape {tuple(logits.shape)}")
    return logits[:, -1, :].float()


def screen_rows(
    rows: Sequence[dict[str, Any]],
    *,
    model: Any,
    tokenizer: Any,
    device: torch.device,
    spec: TargetSpec,
    resources: TauResources,
    tool_token_id: int,
    max_context_tokens: int,
    chunk_size: int,
    source_tier: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if tokenizer.pad_token_id is None:
        raise RuntimeError("Tokenizer has no pad token")
    results: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    total = len(rows)
    for chunk_start in range(0, total, chunk_size):
        chunk = rows[chunk_start : chunk_start + chunk_size]
        prepared: list[PreparedPrefix] = []
        for row in chunk:
            prefix, reason = render_prefix(
                row,
                tokenizer=tokenizer,
                spec=spec,
                resources=resources,
                max_context_tokens=max_context_tokens,
            )
            if prefix is None:
                assert reason is not None
                rejected.append(reason)
            else:
                prepared.append(prefix)
        for batch in make_batches(
            prepared,
            batch_size=spec.batch_size,
            max_batch_tokens=spec.max_batch_tokens,
        ):
            inputs = make_left_padded_batch(batch, pad_token_id=int(tokenizer.pad_token_id), device=device)
            logits = model_last_logits(model, inputs)
            top_values, top_ids = torch.max(logits, dim=-1)
            masked = logits.clone()
            masked[:, tool_token_id] = -torch.inf
            non_tool_values, non_tool_ids = torch.max(masked, dim=-1)
            tool_logits = logits[:, tool_token_id]
            tool_probabilities = torch.softmax(logits, dim=-1)[:, tool_token_id]
            for index, prefix in enumerate(batch):
                row = prefix.row
                top_id = int(top_ids[index].item())
                results.append(
                    {
                        "candidate_id": str(row["candidate_id"]),
                        "trace_index": int(row["trace_index"]),
                        "task_id": str(row["task_id"]),
                        "source_tier": source_tier,
                        "source_target_kind": row.get("target_kind"),
                        "target_tool_name": row.get("target_tool_name"),
                        "target_agent_tool_ordinal": row.get("target_agent_tool_ordinal"),
                        "prior_agent_tool_depth_group": row.get("prior_agent_tool_depth_group"),
                        "prompt_token_count": prefix.length,
                        "adapter": prefix.adapter_audit,
                        "baseline_model": spec.display_name,
                        "baseline_model_path": str(spec.model_path),
                        "baseline_tool_opening": spec.tool_token_text,
                        "baseline_tool_token_id": tool_token_id,
                        "baseline_tool_logit": float(tool_logits[index].item()),
                        "baseline_tool_probability": float(tool_probabilities[index].item()),
                        "baseline_top1_token_id": top_id,
                        "baseline_top1_token_text": tokenizer.decode(
                            [top_id], clean_up_tokenization_spaces=False
                        ),
                        "baseline_top1_logit": float(top_values[index].item()),
                        "baseline_is_tool_call_top1": bool(top_id == tool_token_id),
                        "baseline_best_non_tool_token_id": int(non_tool_ids[index].item()),
                        "baseline_best_non_tool_logit": float(non_tool_values[index].item()),
                        "baseline_tool_margin": float((tool_logits[index] - non_tool_values[index]).item()),
                    }
                )
            del inputs, logits, top_values, top_ids, masked, non_tool_values, non_tool_ids, tool_logits, tool_probabilities
            torch.cuda.empty_cache()
        completed = min(chunk_start + len(chunk), total)
        print(
            json.dumps(
                {
                    "model": spec.key,
                    "tier": source_tier,
                    "screened_source_rows": completed,
                    "source_rows_total": total,
                    "eligible_tool_so_far": sum(row["baseline_is_tool_call_top1"] for row in results),
                    "eligible_non_tool_so_far": sum(
                        not row["baseline_is_tool_call_top1"] and row["baseline_tool_probability"] <= 0.05
                        for row in results
                    ),
                }
            ),
            flush=True,
        )
        del prepared
        gc.collect()
    return results, rejected


def primary_paths(prepared_root: Path, arm: str) -> Path:
    if arm == "tool":
        return prepared_root / "screen_pool.jsonl"
    if arm == "direct":
        # Main induction pool: a real customer reports that the issue is
        # resolved and the source agent naturally replies in text.  This is a
        # semantically clean non-tool state.  Broader text replies remain
        # available only through the supplementary pass for model families
        # that cannot supply 200 from this strict pool.
        return prepared_root / "text_reply_induction_terminal" / "screen_pool.jsonl"
    raise ValueError(arm)


def all_paths(prepared_root: Path, arm: str) -> Path:
    if arm == "tool":
        return prepared_root / "candidate_prefixes.jsonl"
    if arm == "direct":
        return prepared_root / "text_reply_induction" / "candidate_prefixes.jsonl"
    raise ValueError(arm)


def closure_paths(prepared_root: Path, arm: str) -> Path:
    if arm != "direct":
        raise ValueError("The closure pool is defined only for the direct-reply arm")
    return prepared_root / "text_reply_induction_closure" / "screen_pool.jsonl"


def legacy_selection_paths(spec: TargetSpec) -> list[Path]:
    root = PROJECT_ROOT / "results" / "natural_trajectory"
    prepared = PREPARED_ROOT
    if spec.key == "qwen3_4b":
        base = root / "crossscale_qwen3_4b_14b_20260725" / "Qwen3-4B"
        return [base / "01_screen_tool" / "selected_50.jsonl", base / "02_screen_text" / "selected_50.jsonl"]
    if spec.key == "qwen3_8b":
        return [
            prepared / "selected_50_qwen3_8b.jsonl",
            prepared / "text_reply_induction_terminal" / "selected_50_qwen3_8b_text_reply.jsonl",
        ]
    if spec.key == "qwen3_14b":
        base = root / "crossscale_qwen3_4b_14b_20260725" / "Qwen3-14B"
        return [base / "07_screen_tool" / "selected_50.jsonl", base / "08_screen_text" / "selected_50.jsonl"]
    base = root / "cross_family_tau2_20260726" / spec.key
    return [base / "selected_tool_prefixes.jsonl", base / "selected_induction_prefixes.jsonl"]


def load_legacy_task_ids(spec: TargetSpec) -> tuple[set[str], dict[str, int]]:
    task_ids: set[str] = set()
    counts: dict[str, int] = {}
    for path in legacy_selection_paths(spec):
        if not path.exists():
            counts[str(path)] = 0
            continue
        rows = read_jsonl(path)
        found = {str(row["task_id"]) for row in rows if row.get("task_id") is not None}
        task_ids.update(found)
        counts[str(path)] = len(found)
    return task_ids, counts


def screen_file(model_root: Path, arm: str, tier: str) -> Path:
    return model_root / f"{arm}_baseline_screen_{tier}.jsonl"


def rejection_file(model_root: Path, arm: str, tier: str) -> Path:
    return model_root / f"{arm}_render_rejections_{tier}.json"


def existing_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {str(row["candidate_id"]) for row in read_jsonl(path)}


def merge_screen_records(model_root: Path, arm: str) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for tier in ("primary", "closure", "broad"):
        path = screen_file(model_root, arm, tier)
        if not path.exists():
            continue
        for row in read_jsonl(path):
            candidate_id = str(row["candidate_id"])
            if candidate_id in merged:
                raise RuntimeError(f"Duplicate cached baseline record for {candidate_id}")
            merged[candidate_id] = row
    return list(merged.values())


def group_key(row: dict[str, Any], arm: str) -> tuple[Any, ...]:
    if arm == "tool":
        return (str(row.get("target_tool_name") or "unknown"), int(row.get("target_agent_tool_ordinal") or 0))
    return (int(row.get("prior_agent_tool_depth_group") or 0),)


def choose_balanced(
    rows: Sequence[dict[str, Any]],
    *,
    arm: str,
    count: int,
    forbidden_task_ids: set[str],
    seed: int,
    model_key: str,
) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        task_id = str(row["task_id"])
        if task_id not in forbidden_task_ids:
            groups[group_key(row, arm)].append(row)
    keys = sorted(groups, key=lambda key: stable_rank(seed, model_key, arm, "group", *key))
    for key in keys:
        groups[key].sort(
            key=lambda row: stable_rank(seed, model_key, arm, "candidate", row["candidate_id"])
        )
    offsets = {key: 0 for key in keys}
    chosen: list[dict[str, Any]] = []
    used_task_ids = set(forbidden_task_ids)
    while len(chosen) < count:
        progress = False
        for key in keys:
            values = groups[key]
            while offsets[key] < len(values) and str(values[offsets[key]]["task_id"]) in used_task_ids:
                offsets[key] += 1
            if offsets[key] >= len(values):
                continue
            row = values[offsets[key]]
            offsets[key] += 1
            task_id = str(row["task_id"])
            if task_id in used_task_ids:
                continue
            chosen.append(row)
            used_task_ids.add(task_id)
            progress = True
            if len(chosen) == count:
                break
        if not progress:
            break
    return chosen


def unique_task_count(rows: Sequence[dict[str, Any]]) -> int:
    return len({str(row["task_id"]) for row in rows})


def raw_source_map(prepared_root: Path, arm: str) -> dict[str, dict[str, Any]]:
    rows = read_jsonl(all_paths(prepared_root, arm))
    mapping = {str(row["candidate_id"]): row for row in rows}
    if len(mapping) != len(rows):
        raise RuntimeError(f"Duplicate source candidate IDs in {arm} pool")
    return mapping


def attach_source(
    screen_rows: Sequence[dict[str, Any]], *, source_map: dict[str, dict[str, Any]], arm: str
) -> list[dict[str, Any]]:
    attached: list[dict[str, Any]] = []
    for screen in screen_rows:
        candidate_id = str(screen["candidate_id"])
        source = source_map.get(candidate_id)
        if source is None:
            raise KeyError(f"No {arm} source row for {candidate_id}")
        merged = dict(source)
        merged["tau2_200_baseline"] = screen
        attached.append(merged)
    return attached


def choose_final_sets(
    *,
    model_root: Path,
    spec: TargetSpec,
    prepared_root: Path,
    final_count: int,
    max_non_tool_probability: float,
    seed: int,
    collection_name: str,
) -> dict[str, Any]:
    legacy_task_ids, legacy_counts = load_legacy_task_ids(spec)
    tool_screen = merge_screen_records(model_root, "tool")
    direct_screen = merge_screen_records(model_root, "direct")
    tool_eligible = [row for row in tool_screen if bool(row["baseline_is_tool_call_top1"])]
    direct_eligible = [
        row
        for row in direct_screen
        if not bool(row["baseline_is_tool_call_top1"])
        and float(row["baseline_tool_probability"]) <= max_non_tool_probability
    ]
    tool_available = unique_task_count(
        [row for row in tool_eligible if str(row["task_id"]) not in legacy_task_ids]
    )
    direct_available = unique_task_count(
        [row for row in direct_eligible if str(row["task_id"]) not in legacy_task_ids]
    )

    # Allocate the scarcer arm first so disjointness cannot consume its only
    # eligible task IDs.  This ordering is deterministic and recorded.
    arm_order = ("tool", "direct") if tool_available <= direct_available else ("direct", "tool")
    arm_rows = {"tool": tool_eligible, "direct": direct_eligible}
    selections: dict[str, list[dict[str, Any]]] = {}
    used_task_ids = set(legacy_task_ids)
    for arm in arm_order:
        picked = choose_balanced(
            arm_rows[arm],
            arm=arm,
            count=final_count,
            forbidden_task_ids=used_task_ids,
            seed=seed,
            model_key=spec.key,
        )
        selections[arm] = picked
        used_task_ids.update(str(row["task_id"]) for row in picked)

    complete = all(len(selections.get(arm, [])) == final_count for arm in ("tool", "direct"))
    summary: dict[str, Any] = {
        "collection_name": collection_name,
        "model": spec.display_name,
        "model_key": spec.key,
        "target_per_arm": final_count,
        "max_non_tool_probability": max_non_tool_probability,
        "legacy_50_task_ids_excluded": len(legacy_task_ids),
        "legacy_selection_task_counts": legacy_counts,
        "screen_records": {"tool": len(tool_screen), "direct": len(direct_screen)},
        "eligible_unique_tasks_after_legacy_exclusion": {
            "tool": tool_available,
            "direct": direct_available,
        },
        "selection_priority": list(arm_order),
        "selected": {arm: len(selections.get(arm, [])) for arm in ("tool", "direct")},
        "complete": complete,
    }
    if not complete:
        write_json(model_root / "selection_summary.json", summary)
        return summary

    tool_task_ids = {str(row["task_id"]) for row in selections["tool"]}
    direct_task_ids = {str(row["task_id"]) for row in selections["direct"]}
    if tool_task_ids & direct_task_ids:
        raise RuntimeError("Final tool and direct arms share task IDs")
    if (tool_task_ids | direct_task_ids) & legacy_task_ids:
        raise RuntimeError("Final collection overlaps legacy 50-example task IDs")
    for arm in ("tool", "direct"):
        if unique_task_count(selections[arm]) != final_count:
            raise RuntimeError(f"{arm} selection has duplicate tasks")
        if any(int(row["prompt_token_count"]) > 30_000 for row in selections[arm]):
            raise RuntimeError(f"{arm} selection violates the 30K token ceiling")

    tool_source = raw_source_map(prepared_root, "tool")
    direct_source = raw_source_map(prepared_root, "direct")
    tool_output = attach_source(selections["tool"], source_map=tool_source, arm="tool")
    direct_output = attach_source(selections["direct"], source_map=direct_source, arm="direct")
    write_jsonl(model_root / "selected_tool_200.jsonl", tool_output)
    write_jsonl(model_root / "selected_direct_200.jsonl", direct_output)
    summary.update(
        {
            "task_overlap_between_arms": 0,
            "selected_token_lengths": {
                arm: {
                    "min": min(int(row["prompt_token_count"]) for row in selections[arm]),
                    "median": sorted(int(row["prompt_token_count"]) for row in selections[arm])[final_count // 2],
                    "max": max(int(row["prompt_token_count"]) for row in selections[arm]),
                }
                for arm in ("tool", "direct")
            },
            "selection_rule": {
                "tool": "target-model native tool opener is baseline top-1",
                "direct": "target-model native tool opener is not baseline top-1 and has probability <= cutoff",
                "independence": "one task_id per arm; arms and legacy 50 task IDs are disjoint",
                "source_success": "not used for selection",
            },
            "files": {
                "tool": str(model_root / "selected_tool_200.jsonl"),
                "direct": str(model_root / "selected_direct_200.jsonl"),
            },
        }
    )
    write_json(model_root / "selection_summary.json", summary)
    return summary


def source_rows_for_tier(
    prepared_root: Path,
    *,
    arm: str,
    tier: str,
    already_screened: set[str],
    excluded_task_ids: set[str],
    limit: int,
    seed: int,
    model_key: str,
) -> list[dict[str, Any]]:
    if tier == "primary":
        rows = read_jsonl(primary_paths(prepared_root, arm))
        rows = [row for row in rows if str(row["task_id"]) not in excluded_task_ids]
        # The primary sample is fixed before resume filtering.  Otherwise a
        # resumed run would silently advance to a different source subset.
        rows = balanced_source_subset(rows, arm=arm, limit=limit, seed=seed, model_key=model_key)
        return [row for row in rows if str(row["candidate_id"]) not in already_screened]
    if tier == "closure":
        if arm == "tool":
            return []
        rows = read_jsonl(closure_paths(prepared_root, arm))
        rows = [
            row
            for row in rows
            if str(row["candidate_id"]) not in already_screened
            and str(row["task_id"]) not in excluded_task_ids
        ]
        return balanced_source_subset(rows, arm=arm, limit=limit, seed=seed, model_key=model_key)
    else:
        rows = read_jsonl(all_paths(prepared_root, arm))
    rows = [
        row
        for row in rows
        if str(row["candidate_id"]) not in already_screened
        and str(row["task_id"]) not in excluded_task_ids
    ]
    return balanced_source_subset(rows, arm=arm, limit=limit, seed=seed, model_key=model_key)


def balanced_source_subset(
    rows: Sequence[dict[str, Any]], *, arm: str, limit: int, seed: int, model_key: str
) -> list[dict[str, Any]]:
    """Take a deterministic, stratified prefix without repeating a task.

    The primary source files already have at most one row per task.  The
    supplementary files have several turns per task, so this routine uses a
    task-level guard even before the final selection step.
    """
    if limit == 0 or limit >= len(rows):
        return list(rows)
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        groups[group_key(row, arm)].append(row)
    keys = sorted(groups, key=lambda key: stable_rank(seed, model_key, arm, "source-group", *key))
    for key in keys:
        groups[key].sort(
            key=lambda row: stable_rank(seed, model_key, arm, "source-candidate", row["candidate_id"])
        )
    offsets = {key: 0 for key in keys}
    used_tasks: set[str] = set()
    chosen: list[dict[str, Any]] = []
    while len(chosen) < limit:
        progress = False
        for key in keys:
            values = groups[key]
            while offsets[key] < len(values) and str(values[offsets[key]]["task_id"]) in used_tasks:
                offsets[key] += 1
            if offsets[key] >= len(values):
                continue
            row = values[offsets[key]]
            offsets[key] += 1
            task_id = str(row["task_id"])
            if task_id in used_tasks:
                continue
            chosen.append(row)
            used_tasks.add(task_id)
            progress = True
            if len(chosen) == limit:
                break
        if not progress:
            break
    return chosen


def source_limit_for(args: argparse.Namespace, *, spec: TargetSpec, arm: str) -> int:
    if args.source_limit is not None:
        return int(args.source_limit)
    if args.tier == "primary":
        return PRIMARY_SCREEN_LIMITS[spec.key][arm]
    if args.tier == "broad":
        # `candidate_prefixes` can contain several natural turns per task.
        # Make each broad pass task-first; subsequent resumed passes advance
        # to another real turn for those same tasks only if needed.
        return 1_625
    # The closure pool is already a compact, one-reply-per-trajectory pool.
    return 0


def parse_models(raw: str) -> list[TargetSpec]:
    if raw.strip() == "all":
        return [SPECS[key] for key in SPECS]
    keys = [key.strip() for key in raw.split(",") if key.strip()]
    unknown = [key for key in keys if key not in SPECS]
    if unknown:
        raise ValueError(f"Unknown model keys: {unknown}; choices are {sorted(SPECS)}")
    return [SPECS[key] for key in keys]


def run_model(args: argparse.Namespace, spec: TargetSpec, resources: TauResources) -> dict[str, Any]:
    model_root = args.output_root.resolve() / spec.key
    model_root.mkdir(parents=True, exist_ok=True)
    metadata_path = model_root / "collection_config.json"
    tool_source_path = primary_paths(args.prepared_root, "tool")
    direct_source_path = primary_paths(args.prepared_root, "direct")
    write_json(
        metadata_path,
        {
            "collection_name": args.collection_name,
            "model": spec.display_name,
            "model_key": spec.key,
            "model_path": str(spec.model_path),
            "tier_last_run": args.tier,
            "target_per_arm": args.final_count,
            "max_context_tokens": args.max_context_tokens,
            "max_non_tool_probability": args.max_non_tool_probability,
            "source_files": {
                "tool_primary": {"path": str(tool_source_path), "sha256": sha256_file(tool_source_path)},
                "direct_primary": {"path": str(direct_source_path), "sha256": sha256_file(direct_source_path)},
                "system_prompt": {
                    "path": str(args.raw_root / "tau2_system_prompt.txt"),
                    "sha256": sha256_file(args.raw_root / "tau2_system_prompt.txt"),
                },
                "tool_schemas": {
                    "path": str(args.raw_root / "tau2_tool_schemas.json"),
                    "sha256": sha256_file(args.raw_root / "tau2_tool_schemas.json"),
                },
            },
            "selection_scope": "Baseline-only target-model screening; no vector, layer, reward, or success is used.",
        },
    )
    model, tokenizer, device = load_model_and_tokenizer(spec)
    try:
        tool_token_id, token_audit = resolve_tool_token(tokenizer, spec)
        write_json(model_root / "tool_decision_spec.json", token_audit)
        legacy_task_ids, _legacy_counts = load_legacy_task_ids(spec)
        arms = ("tool", "direct") if args.arms == "both" else (args.arms,)
        for arm in arms:
            path = screen_file(model_root, arm, args.tier)
            previous = existing_ids(path) if args.resume else set()
            previous_all = (
                {str(row["candidate_id"]) for row in merge_screen_records(model_root, arm)}
                if args.resume
                else set()
            )
            source_limit = source_limit_for(args, spec=spec, arm=arm)
            source_rows = source_rows_for_tier(
                args.prepared_root,
                arm=arm,
                tier=args.tier,
                already_screened=previous_all,
                excluded_task_ids=legacy_task_ids,
                limit=source_limit,
                seed=args.seed,
                model_key=spec.key,
            )
            print(
                json.dumps(
                    {
                        "model": spec.key,
                        "arm": arm,
                        "tier": args.tier,
                        "cached_rows": len(previous),
                        "cached_rows_all_tiers": len(previous_all),
                        "source_limit": source_limit,
                        "rows_to_screen": len(source_rows),
                    }
                ),
                flush=True,
            )
            results, rejected = screen_rows(
                source_rows,
                model=model,
                tokenizer=tokenizer,
                device=device,
                spec=spec,
                resources=resources,
                tool_token_id=tool_token_id,
                max_context_tokens=args.max_context_tokens,
                chunk_size=args.render_chunk_size,
                source_tier=args.tier,
            )
            existing = read_jsonl(path) if args.resume and path.exists() else []
            combined = [*existing, *results]
            if len({str(row["candidate_id"]) for row in combined}) != len(combined):
                raise RuntimeError(f"Duplicate baseline candidate IDs in {path}")
            write_jsonl(path, combined)
            write_json(
                rejection_file(model_root, arm, args.tier),
                {
                    "source_rows": len(source_rows) + len(previous),
                    "newly_screened_rows": len(source_rows),
                    "rendered_rows": len(results),
                    "rejections": rejected,
                },
            )
        summary = choose_final_sets(
            model_root=model_root,
            spec=spec,
            prepared_root=args.prepared_root,
            final_count=args.final_count,
            max_non_tool_probability=args.max_non_tool_probability,
            seed=args.seed,
            collection_name=args.collection_name,
        )
        print(json.dumps({"model": spec.key, "selection": summary}, ensure_ascii=False), flush=True)
        return summary
    finally:
        del model
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> None:
    args = parse_args()
    if args.final_count < 1 or args.max_context_tokens < 1 or args.render_chunk_size < 1:
        raise ValueError("Counts and context ceiling must be positive")
    if not 0.0 <= args.max_non_tool_probability <= 1.0:
        raise ValueError("--max-non-tool-probability must lie in [0, 1]")
    args.raw_root = args.raw_root.resolve()
    args.prepared_root = args.prepared_root.resolve()
    args.output_root = args.output_root.resolve()
    for path in (
        args.raw_root / "tau2_system_prompt.txt",
        args.raw_root / "tau2_tool_schemas.json",
        primary_paths(args.prepared_root, "tool"),
        primary_paths(args.prepared_root, "direct"),
        all_paths(args.prepared_root, "tool"),
        all_paths(args.prepared_root, "direct"),
    ):
        if not path.exists():
            raise FileNotFoundError(path)
    resources = read_resources(args.raw_root)
    summaries = {
        spec.key: run_model(args, spec, resources)
        for spec in parse_models(args.models)
    }
    write_json(args.output_root / f"tier_{args.tier}_run_summary.json", {
        "completed_unix": time.time(),
        "tier": args.tier,
        "arms": args.arms,
        "models": summaries,
    })


if __name__ == "__main__":
    main()
