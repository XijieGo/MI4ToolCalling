#!/usr/bin/env python3
"""Cross-family τ² Telecom intervention experiment.

The experiment intentionally has two disjoint phases:

1. Estimate a tool-call direction on *coding* clean/corrupt pairs for each
   target model at that model family's previously coding-localized residual
   layer.  Telecom examples are never read in this phase.
2. Render natural Telecom histories with the target model's native tool-call
   template, screen model-specific positive/negative first-token baselines,
   and intervene only at the final prompt token.

The three target models use incompatible tool protocols.  The small adapters
below preserve the agent-visible history while converting only protocol-level
details (Mistral call IDs, Devstral's leading-greeting constraint, and
Granite's inline historical-call syntax).  All changes are recorded in the
run configuration and per-prefix render audit.
"""

from __future__ import annotations

import argparse
import collections
import contextlib
import csv
import gc
import hashlib
import json
import os
import platform
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_ROOT = PROJECT_ROOT / "datasets" / "external" / "tau2_telecom_qwen35_9b" / "raw"
PREPARED_ROOT = PROJECT_ROOT / "datasets" / "external" / "tau2_telecom_qwen35_9b" / "prepared"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "results" / "natural_trajectory" / "cross_family_tau2_20260726"
SEED = 20260726
MODEL_ROOT = Path(os.environ.get("MODEL_ROOT", PROJECT_ROOT / "external" / "models")).expanduser()


def model_path(environment_variable: str, directory_name: str) -> Path:
    return Path(os.environ.get(environment_variable, MODEL_ROOT / directory_name)).expanduser()


@dataclass(frozen=True)
class ModelSpec:
    key: str
    display_name: str
    model_path: Path
    family: str
    tool_token_text: str
    coding_layer: int
    coding_dataset_root: Path


SPECS: dict[str, ModelSpec] = {
    "granite": ModelSpec(
        key="granite",
        display_name="Granite-3.3-8B-Instruct",
        model_path=model_path("GRANITE_3P3_8B_PATH", "granite-3.3-8b-instruct"),
        family="granite",
        tool_token_text="<|tool_call|>",
        coding_layer=35,
        coding_dataset_root=PROJECT_ROOT / "results" / "section6_generalization" / "granite_3p3_8b" / "dataset",
    ),
    "mistral": ModelSpec(
        key="mistral",
        display_name="Mistral-Small-3.2-24B-Instruct-2506",
        model_path=model_path("MISTRAL_3P2_24B_PATH", "Mistral-Small-3.2-24B-Instruct-2506"),
        family="mistral",
        tool_token_text="[TOOL_CALLS]",
        coding_layer=26,
        coding_dataset_root=PROJECT_ROOT / "results" / "section6_generalization" / "mistral_3p2_24b" / "datasets",
    ),
    "devstral": ModelSpec(
        key="devstral",
        display_name="Devstral-Small-2-24B-Instruct-2512",
        model_path=model_path("DEVSTRAL_2_24B_PATH", "Devstral-Small-2-24B-Instruct-2512"),
        family="devstral",
        tool_token_text="[TOOL_CALLS]",
        coding_layer=33,
        coding_dataset_root=PROJECT_ROOT / "results" / "section6_generalization" / "devstral_2_24b" / "datasets",
    ),
}


@dataclass(frozen=True)
class PreparedPrefix:
    candidate_id: str
    input_ids: list[int]
    meta: dict[str, Any]

    @property
    def length(self) -> int:
        return len(self.input_ids)


@dataclass(frozen=True)
class CodingPair:
    pair_id: str
    split: str
    clean_ids: list[int]
    corrupt_ids: list[int]


@dataclass(frozen=True)
class Condition:
    name: str
    family: str
    alpha: float
    delta_cpu: torch.Tensor | None
    description: str

    @property
    def delta_norm(self) -> float:
        return 0.0 if self.delta_cpu is None else float(self.delta_cpu.norm().item())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=sorted(SPECS), required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--raw-root", type=Path, default=RAW_ROOT)
    parser.add_argument("--prepared-root", type=Path, default=PREPARED_ROOT)
    parser.add_argument("--coding-batch-size", type=int, default=4)
    parser.add_argument("--tau-batch-size", type=int, default=1)
    parser.add_argument("--max-context-tokens", type=int, default=16384)
    parser.add_argument(
        "--tool-screen-limit",
        type=int,
        default=400,
        help="Deterministic balanced cap for pre-tool-call screening; 0 means all rows.",
    )
    parser.add_argument(
        "--induction-screen-limit",
        type=int,
        default=0,
        help="Deterministic balanced cap for text-reply screening; 0 means all rows.",
    )
    parser.add_argument("--final-count", type=int, default=50)
    parser.add_argument("--max-baseline-tool-probability", type=float, default=0.05)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--skip-generation",
        action="store_true",
        help="Skip the alpha=1 induction continuation audit (diagnostic only).",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{number}") from exc
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_rank(*parts: object) -> int:
    value = "|".join(str(part) for part in parts)
    return int.from_bytes(hashlib.sha256(value.encode("utf-8")).digest()[:8], "big")


def normalize_arguments(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        parsed = json.loads(value)
        if not isinstance(parsed, dict):
            raise ValueError("Tool arguments must decode to an object")
        return parsed
    raise TypeError(f"Unsupported tool arguments type: {type(value).__name__}")


def function_name(call: dict[str, Any]) -> str:
    function = call.get("function") or {}
    name = call.get("name") or function.get("name")
    if not name:
        raise ValueError(f"Tool call has no name: {call}")
    return str(name)


def function_arguments(call: dict[str, Any]) -> dict[str, Any]:
    function = call.get("function") or {}
    return normalize_arguments(call.get("arguments", function.get("arguments")))


def safe_content(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def adapt_tau2_messages(row: dict[str, Any], spec: ModelSpec) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return native-model-compatible history plus a concise audit record."""
    source = row.get("messages")
    if not isinstance(source, list):
        raise TypeError(f"{row.get('candidate_id')} has no message list")

    if spec.family == "granite":
        messages: list[dict[str, Any]] = []
        historical_calls = 0
        for message in source:
            role = str(message["role"])
            content = safe_content(message.get("content"))
            calls = message.get("tool_calls") or []
            if role == "assistant" and calls:
                encoded_calls = [
                    {"name": function_name(call), "arguments": function_arguments(call)} for call in calls
                ]
                encoded = "<|tool_call|>" + json.dumps(encoded_calls, ensure_ascii=False, separators=(",", ":"))
                content = content + encoded if content else encoded
                historical_calls += len(encoded_calls)
            messages.append({"role": role, "content": content})
        return messages, {
            "adapter": "granite_inline_calls",
            "historical_calls_serialized": historical_calls,
            "leading_assistant_dropped": False,
            "mistral_call_ids_remapped": False,
        }

    if spec.family == "mistral":
        messages = []
        call_id_map: dict[str, str] = {}
        call_name_map: dict[str, str] = {}
        counter = 0
        for message in source:
            role = str(message["role"])
            converted: dict[str, Any] = {"role": role, "content": safe_content(message.get("content"))}
            calls = message.get("tool_calls") or []
            if role == "assistant" and calls:
                converted_calls: list[dict[str, Any]] = []
                for call in calls:
                    counter += 1
                    original_id = str(call.get("id") or f"source_call_{counter}")
                    # Mistral's native formatter requires exactly nine alphanumeric characters.
                    native_id = f"a{counter:08d}"
                    call_id_map[original_id] = native_id
                    call_name_map[original_id] = function_name(call)
                    converted_calls.append(
                        {
                            "id": native_id,
                            "type": "function",
                            "function": {
                                "name": function_name(call),
                                "arguments": json.dumps(function_arguments(call), ensure_ascii=False, separators=(",", ":")),
                            },
                        }
                    )
                converted["tool_calls"] = converted_calls
            elif role == "tool":
                original_id = str(message.get("tool_call_id") or message.get("id") or "")
                if original_id not in call_id_map:
                    raise ValueError(f"Unmatched Mistral tool result ID {original_id!r} in {row.get('candidate_id')}")
                converted["tool_call_id"] = call_id_map[original_id]
                converted["name"] = call_name_map[original_id]
            messages.append(converted)
        return messages, {
            "adapter": "mistral_native_tool_history",
            "historical_calls_serialized": counter,
            "leading_assistant_dropped": False,
            "mistral_call_ids_remapped": True,
        }

    if spec.family == "devstral":
        messages = []
        call_name_map: dict[str, str] = {}
        for message in source:
            role = str(message["role"])
            converted: dict[str, Any] = {"role": role, "content": safe_content(message.get("content"))}
            calls = message.get("tool_calls") or []
            if role == "assistant" and calls:
                converted["tool_calls"] = []
                for call in calls:
                    call_id = str(call.get("id") or "")
                    call_name_map[call_id] = function_name(call)
                    converted["tool_calls"].append(
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {
                                "name": function_name(call),
                                "arguments": function_arguments(call),
                            },
                        }
                    )
            elif role == "tool":
                call_id = str(message.get("tool_call_id") or message.get("id") or "")
                converted["tool_call_id"] = call_id
                converted["name"] = call_name_map.get(call_id, "unknown")
            messages.append(converted)
        # Devstral's supplied template requires the first non-system message to
        # be a user turn.  τ² logs an initial canned assistant greeting; it is
        # not a response to an agent-visible user message, so omit only that
        # incompatible greeting and preserve every subsequent history turn.
        dropped = bool(messages and messages[0].get("role") == "assistant")
        if dropped:
            messages = messages[1:]
        return messages, {
            "adapter": "devstral_native_tool_history",
            "historical_calls_serialized": len(call_name_map),
            "leading_assistant_dropped": dropped,
            "mistral_call_ids_remapped": False,
        }

    raise ValueError(f"Unsupported family: {spec.family}")


def load_tokenizer_and_model(spec: ModelSpec) -> tuple[Any, Any, torch.device]:
    if not torch.cuda.is_available():
        raise RuntimeError("τ² intervention requires a CUDA device")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(spec.model_path), trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    if spec.family == "granite":
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(
            str(spec.model_path),
            torch_dtype=torch.bfloat16,
            device_map={"": 0},
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        ).eval()
    else:
        from transformers import Mistral3ForConditionalGeneration

        kwargs: dict[str, Any] = {
            "trust_remote_code": True,
            "device_map": {"": 0},
        }
        if spec.family == "devstral":
            from transformers.utils.quantization_config import FineGrainedFP8Config

            kwargs["quantization_config"] = FineGrainedFP8Config(dequantize=True)
        try:
            model = Mistral3ForConditionalGeneration.from_pretrained(
                str(spec.model_path), dtype=torch.bfloat16, **kwargs
            ).eval()
        except TypeError:
            model = Mistral3ForConditionalGeneration.from_pretrained(
                str(spec.model_path), torch_dtype=torch.bfloat16, **kwargs
            ).eval()
    return model, tokenizer, torch.device("cuda:0")


def resolve_layers(model: Any) -> Any:
    candidates: list[Any] = []
    if hasattr(model, "model"):
        candidates.append(model.model)
        if hasattr(model.model, "language_model"):
            candidates.append(model.model.language_model)
    if hasattr(model, "language_model"):
        candidates.append(model.language_model)
    for candidate in candidates:
        if hasattr(candidate, "layers"):
            return candidate.layers
        if hasattr(candidate, "model") and hasattr(candidate.model, "layers"):
            return candidate.model.layers
    raise AttributeError("Could not resolve text decoder layers")


def hidden_size(model: Any) -> int:
    layers = resolve_layers(model)
    if not layers:
        raise ValueError("Model has no decoder layers")
    first = layers[0]
    if hasattr(first, "self_attn") and hasattr(first.self_attn, "q_proj"):
        return int(first.self_attn.q_proj.in_features)
    for candidate in (getattr(model, "config", None), getattr(getattr(model, "language_model", None), "config", None)):
        if candidate is not None and hasattr(candidate, "hidden_size"):
            return int(candidate.hidden_size)
    raise AttributeError("Could not resolve model hidden size")


def resolve_tool_token_id(tokenizer: Any, spec: ModelSpec) -> tuple[int, dict[str, Any]]:
    encoded = [int(value) for value in tokenizer.encode(spec.tool_token_text, add_special_tokens=False)]
    converted = int(tokenizer.convert_tokens_to_ids(spec.tool_token_text))
    if converted < 0:
        raise RuntimeError(f"Could not resolve {spec.tool_token_text!r}")
    if spec.family == "granite" and len(encoded) != 1:
        raise RuntimeError(f"Granite tool token must encode as one token, got {encoded}")
    if spec.family == "devstral" and len(encoded) != 1:
        raise RuntimeError(f"Devstral tool token must encode as one token, got {encoded}")
    return converted, {
        "tool_token_text": spec.tool_token_text,
        "tool_token_id_via_convert": converted,
        "tool_token_ids_via_encode": encoded,
        "tool_token_encode_len": len(encoded),
    }


def render_prefix(
    row: dict[str, Any], *, tokenizer: Any, spec: ModelSpec, raw_root: Path, max_context_tokens: int
) -> tuple[PreparedPrefix | None, dict[str, Any] | None]:
    try:
        system_prompt = (raw_root / "tau2_system_prompt.txt").read_text(encoding="utf-8").strip()
        tools = json.loads((raw_root / "tau2_tool_schemas.json").read_text(encoding="utf-8"))
        if not system_prompt or not isinstance(tools, list) or not tools:
            raise ValueError("Invalid Telecom system prompt or tool schema file")
        native_messages, audit = adapt_tau2_messages(row, spec)
        messages = [{"role": "system", "content": system_prompt}, *native_messages]
        if spec.family == "granite":
            try:
                rendered = tokenizer.apply_chat_template(
                    messages, tools=tools, tokenize=False, add_generation_prompt=True, thinking=False
                )
            except TypeError:
                rendered = tokenizer.apply_chat_template(messages, tools=tools, tokenize=False, add_generation_prompt=True)
            ids = [int(value) for value in tokenizer(rendered, add_special_tokens=False)["input_ids"]]
        else:
            encoded = tokenizer.apply_chat_template(
                messages,
                tools=tools,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
            ids = [int(value) for value in encoded["input_ids"][0].detach().cpu().tolist()]
        if not ids:
            raise ValueError("Template produced an empty prompt")
        if len(ids) > max_context_tokens:
            return None, {
                "candidate_id": row.get("candidate_id"),
                "reason": "context_too_long",
                "token_count": len(ids),
                "max_context_tokens": max_context_tokens,
                "adapter": audit,
            }
        meta = {
            "candidate_id": str(row["candidate_id"]),
            "trace_index": int(row["trace_index"]),
            "task_id": row.get("task_id"),
            "target_tool_name": row.get("target_tool_name"),
            "target_agent_tool_ordinal": row.get("target_agent_tool_ordinal"),
            "target_kind": row.get("target_kind"),
            "prior_agent_tool_depth_group": row.get("prior_agent_tool_depth_group"),
            "source_success": bool(row.get("source_success")),
            "adapter": audit,
        }
        return PreparedPrefix(candidate_id=str(row["candidate_id"]), input_ids=ids, meta=meta), None
    except Exception as exc:
        return None, {
            "candidate_id": row.get("candidate_id"),
            "reason": "render_error",
            "detail": repr(exc),
        }


def balanced_subset_tool(rows: list[dict[str, Any]], *, limit: int, seed: int) -> list[dict[str, Any]]:
    if limit <= 0 or limit >= len(rows):
        return list(rows)
    groups: dict[tuple[str, int], list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        groups[(str(row["target_tool_name"]), int(row["target_agent_tool_ordinal"]))].append(row)
    keys = sorted(groups, key=lambda key: stable_rank(seed, "tool-screen-group", *key))
    for key in keys:
        groups[key].sort(key=lambda row: stable_rank(seed, "tool-screen-row", row["candidate_id"]))
    chosen: list[dict[str, Any]] = []
    offsets = {key: 0 for key in keys}
    while len(chosen) < limit:
        progressed = False
        for key in keys:
            offset = offsets[key]
            if offset >= len(groups[key]):
                continue
            chosen.append(groups[key][offset])
            offsets[key] += 1
            progressed = True
            if len(chosen) == limit:
                break
        if not progressed:
            break
    return chosen


def balanced_subset_induction(rows: list[dict[str, Any]], *, limit: int, seed: int) -> list[dict[str, Any]]:
    if limit <= 0 or limit >= len(rows):
        return list(rows)
    groups: dict[int, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        groups[int(row["prior_agent_tool_depth_group"])].append(row)
    keys = sorted(groups, key=lambda key: stable_rank(seed, "induction-screen-group", key))
    for key in keys:
        groups[key].sort(key=lambda row: stable_rank(seed, "induction-screen-row", row["candidate_id"]))
    chosen: list[dict[str, Any]] = []
    offsets = {key: 0 for key in keys}
    while len(chosen) < limit:
        progressed = False
        for key in keys:
            offset = offsets[key]
            if offset >= len(groups[key]):
                continue
            chosen.append(groups[key][offset])
            offsets[key] += 1
            progressed = True
            if len(chosen) == limit:
                break
        if not progressed:
            break
    return chosen


def select_tool_rows(rows: list[dict[str, Any]], *, final_count: int, seed: int) -> list[dict[str, Any]]:
    eligible = [row for row in rows if bool(row["is_tool_call_top1"])]
    unique: list[dict[str, Any]] = []
    seen: set[int] = set()
    for row in eligible:
        trace = int(row["trace_index"])
        if trace not in seen:
            unique.append(row)
            seen.add(trace)
    groups: dict[tuple[str, int], list[dict[str, Any]]] = collections.defaultdict(list)
    for row in unique:
        groups[(str(row.get("target_tool_name")), int(row.get("target_agent_tool_ordinal") or 0))].append(row)
    keys = sorted(groups, key=lambda key: stable_rank(seed, "tool-final-group", *key))
    for key in keys:
        groups[key].sort(key=lambda row: stable_rank(seed, "tool-final-row", row["candidate_id"]))
    chosen: list[dict[str, Any]] = []
    offsets = {key: 0 for key in keys}
    while len(chosen) < final_count:
        progressed = False
        for key in keys:
            offset = offsets[key]
            if offset >= len(groups[key]):
                continue
            chosen.append(groups[key][offset])
            offsets[key] += 1
            progressed = True
            if len(chosen) == final_count:
                break
        if not progressed:
            break
    if len(chosen) < final_count:
        raise RuntimeError(
            f"Only {len(chosen)} independent model-positive Telecom tool prefixes; need {final_count}. "
            "Increase --tool-screen-limit."
        )
    return chosen


def select_induction_rows(
    rows: list[dict[str, Any]], *, final_count: int, max_probability: float, seed: int
) -> list[dict[str, Any]]:
    eligible = [
        row
        for row in rows
        if not bool(row["is_tool_call_top1"]) and float(row["tool_call_probability"]) <= max_probability
    ]
    if len({int(row["trace_index"]) for row in eligible}) != len(eligible):
        raise RuntimeError("Induction screen has duplicate source trajectories")
    chosen = balanced_subset_induction(eligible, limit=final_count, seed=seed)
    if len(chosen) < final_count:
        raise RuntimeError(
            f"Only {len(chosen)} strong non-tool Telecom prefixes; need {final_count}. "
            "Increase --induction-screen-limit or revise the cutoff."
        )
    return chosen


def make_batches(rows: Sequence[PreparedPrefix], batch_size: int) -> list[list[PreparedPrefix]]:
    if batch_size < 1:
        raise ValueError("Batch size must be positive")
    ordered = sorted(rows, key=lambda row: (row.length, row.candidate_id))
    return [ordered[index : index + batch_size] for index in range(0, len(ordered), batch_size)]


def make_left_padded_batch(rows: Sequence[PreparedPrefix], pad_token_id: int, device: torch.device) -> dict[str, torch.Tensor]:
    max_length = max(row.length for row in rows)
    input_ids = torch.full((len(rows), max_length), int(pad_token_id), dtype=torch.long)
    attention_mask = torch.zeros((len(rows), max_length), dtype=torch.long)
    for index, row in enumerate(rows):
        tokens = torch.tensor(row.input_ids, dtype=torch.long)
        input_ids[index, -row.length :] = tokens
        attention_mask[index, -row.length :] = 1
    return {"input_ids": input_ids.to(device), "attention_mask": attention_mask.to(device)}


def model_forward(model: Any, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
    kwargs: dict[str, Any] = {**inputs, "use_cache": False, "return_dict": True}
    with torch.inference_mode():
        try:
            output = model(**kwargs, logits_to_keep=1)
        except TypeError:
            output = model(**kwargs)
    logits = output.logits
    if logits.ndim != 3:
        raise RuntimeError(f"Expected [batch, seq, vocab] logits, got {tuple(logits.shape)}")
    return logits[:, -1, :].float()


@contextlib.contextmanager
def temporary_last_token_addition(
    model: Any, *, layer: int, delta_cpu: torch.Tensor | None, prefill_only: bool
) -> Iterator[dict[str, int]]:
    stats = {"hook_calls": 0, "modified_calls": 0}
    if delta_cpu is None:
        yield stats
        return
    layers = resolve_layers(model)
    if layer < 0 or layer >= len(layers):
        raise ValueError(f"Invalid layer {layer} for a {len(layers)}-layer model")

    def hook(_module: Any, _inputs: Any, output: Any) -> Any:
        stats["hook_calls"] += 1
        hidden = output[0] if isinstance(output, tuple) else output
        if not isinstance(hidden, torch.Tensor) or hidden.ndim != 3:
            raise RuntimeError(f"Unexpected layer output at L{layer}: {type(hidden)!r}")
        if prefill_only and int(hidden.shape[1]) <= 1:
            return output
        if hidden.shape[-1] != delta_cpu.numel():
            raise RuntimeError(f"Vector dim {delta_cpu.numel()} != layer dim {hidden.shape[-1]}")
        patched = hidden.clone()
        delta = delta_cpu.to(device=patched.device, dtype=patched.dtype)
        patched[:, -1, :] = patched[:, -1, :] + delta
        stats["modified_calls"] += 1
        if isinstance(output, tuple):
            return (patched, *output[1:])
        return patched

    handle = layers[layer].register_forward_hook(hook)
    try:
        yield stats
    finally:
        handle.remove()


@contextlib.contextmanager
def temporary_capture(model: Any, *, layer: int, store: dict[str, torch.Tensor]) -> Iterator[None]:
    layers = resolve_layers(model)
    if layer < 0 or layer >= len(layers):
        raise ValueError(f"Invalid layer {layer}")

    def hook(_module: Any, _inputs: Any, output: Any) -> None:
        hidden = output[0] if isinstance(output, tuple) else output
        if not isinstance(hidden, torch.Tensor) or hidden.ndim != 3:
            raise RuntimeError(f"Unexpected layer output at L{layer}: {type(hidden)!r}")
        store["state"] = hidden[:, -1, :].detach().float().cpu()

    handle = layers[layer].register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def token_text(tokenizer: Any, token_id: int) -> str:
    return tokenizer.decode([int(token_id)], clean_up_tokenization_spaces=False)


def evaluate_condition(
    prepared: Sequence[PreparedPrefix],
    *,
    model: Any,
    tokenizer: Any,
    device: torch.device,
    tool_token_id: int,
    layer: int,
    condition: Condition,
    batch_size: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    rows: list[dict[str, Any]] = []
    batches = make_batches(prepared, batch_size)
    with temporary_last_token_addition(
        model, layer=layer, delta_cpu=condition.delta_cpu, prefill_only=False
    ) as stats:
        for batch_index, batch in enumerate(batches, start=1):
            inputs = make_left_padded_batch(batch, int(tokenizer.pad_token_id), device)
            logits = model_forward(model, inputs)
            top_values, top_ids = torch.max(logits, dim=-1)
            tool_logits = logits[:, tool_token_id]
            tool_probabilities = torch.softmax(logits, dim=-1)[:, tool_token_id]
            non_tool_logits = logits.clone()
            non_tool_logits[:, tool_token_id] = -torch.inf
            non_tool_values, non_tool_ids = torch.max(non_tool_logits, dim=-1)
            for local_index, prefix in enumerate(batch):
                top_id = int(top_ids[local_index].item())
                rows.append(
                    {
                        **prefix.meta,
                        "prompt_token_count": prefix.length,
                        "condition": condition.name,
                        "condition_family": condition.family,
                        "alpha": condition.alpha,
                        "direction_description": condition.description,
                        "delta_norm": condition.delta_norm,
                        "tool_call_token_id": int(tool_token_id),
                        "tool_call_logit": float(tool_logits[local_index].item()),
                        "tool_call_probability": float(tool_probabilities[local_index].item()),
                        "is_tool_call_top1": bool(top_id == tool_token_id),
                        "top1_token_id": top_id,
                        "top1_token_text": token_text(tokenizer, top_id),
                        "top1_logit": float(top_values[local_index].item()),
                        "best_non_tool_token_id": int(non_tool_ids[local_index].item()),
                        "best_non_tool_token_text": token_text(tokenizer, int(non_tool_ids[local_index].item())),
                        "best_non_tool_logit": float(non_tool_values[local_index].item()),
                        "tool_call_margin": float((tool_logits[local_index] - non_tool_values[local_index]).item()),
                    }
                )
            del inputs, logits, top_values, top_ids, tool_logits, tool_probabilities, non_tool_logits, non_tool_values, non_tool_ids
            torch.cuda.empty_cache()
            if batch_index == len(batches) or batch_index % 50 == 0:
                print(json.dumps({"condition": condition.name, "batches": [batch_index, len(batches)]}), flush=True)
    by_id = {str(row["candidate_id"]) for row in rows}
    if len(by_id) != len(prepared):
        raise RuntimeError(f"Condition {condition.name} returned duplicate/missing rows")
    return rows, stats


def summarize_condition(rows: Sequence[dict[str, Any]], baseline: dict[str, dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("Cannot summarize empty condition")
    baseline_rows = [baseline[str(row["candidate_id"])] for row in rows]
    n = len(rows)
    tool_count = sum(bool(row["is_tool_call_top1"]) for row in rows)
    baseline_tool_count = sum(bool(row["is_tool_call_top1"]) for row in baseline_rows)
    drops = sum(
        bool(base["is_tool_call_top1"]) and not bool(row["is_tool_call_top1"])
        for row, base in zip(rows, baseline_rows)
    )
    gains = sum(
        not bool(base["is_tool_call_top1"]) and bool(row["is_tool_call_top1"])
        for row, base in zip(rows, baseline_rows)
    )
    return {
        "condition": rows[0]["condition"],
        "condition_family": rows[0]["condition_family"],
        "alpha": rows[0]["alpha"],
        "delta_norm": rows[0]["delta_norm"],
        "n": n,
        "tool_call_top1_count": tool_count,
        "tool_call_top1_rate": tool_count / n,
        "baseline_tool_call_top1_count": baseline_tool_count,
        "baseline_non_tool_top1_count": n - baseline_tool_count,
        "baseline_tool_to_non_tool_count": drops,
        "baseline_tool_to_non_tool_rate": drops / max(baseline_tool_count, 1),
        "baseline_non_tool_to_tool_count": gains,
        "baseline_non_tool_to_tool_rate": gains / max(n - baseline_tool_count, 1),
        "mean_tool_call_logit": sum(float(row["tool_call_logit"]) for row in rows) / n,
        "mean_tool_call_probability": sum(float(row["tool_call_probability"]) for row in rows) / n,
        "mean_tool_logit_change_vs_baseline": sum(
            float(row["tool_call_logit"]) - float(base["tool_call_logit"])
            for row, base in zip(rows, baseline_rows)
        )
        / n,
    }


def ids_from_text(tokenizer: Any, text: str) -> list[int]:
    encoded = tokenizer(text, add_special_tokens=False)["input_ids"]
    if isinstance(encoded, torch.Tensor):
        encoded = encoded.detach().cpu().tolist()
    if encoded and isinstance(encoded[0], list):
        encoded = encoded[0]
    return [int(value) for value in encoded]


def resolve_existing_path(path_text: str, fallbacks: Sequence[Path]) -> Path:
    path = Path(path_text)
    if path.exists():
        return path
    for candidate in fallbacks:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(path)


def load_coding_pairs(spec: ModelSpec, tokenizer: Any) -> list[CodingPair]:
    root = spec.coding_dataset_root
    pairs: list[CodingPair] = []
    if spec.family == "granite":
        for row in read_jsonl(root / "manifest.jsonl"):
            sample_id = str(row["sample_id"])
            clean = resolve_existing_path(
                str(row["clean_prompt_path"]), [root / "clean" / f"{sample_id}.txt"]
            ).read_text(encoding="utf-8")
            corrupt = resolve_existing_path(
                str(row["corrupt_prompt_path"]), [root / "corrupt" / f"{sample_id}.txt"]
            ).read_text(encoding="utf-8")
            pairs.append(
                CodingPair(
                    pair_id=sample_id,
                    split=str(row.get("split", "train")),
                    clean_ids=ids_from_text(tokenizer, clean),
                    corrupt_ids=ids_from_text(tokenizer, corrupt),
                )
            )
    elif spec.family == "mistral":
        # The flat files in ``datasets/`` retain the source model's textual
        # Qwen-style scaffold.  They are useful source records, but are not a
        # valid behavior prompt for Mistral: the native Mistral template opens
        # a tool call with the dedicated [TOOL_CALLS] special token.  Re-render
        # the canonical coding pairs with Mistral's own chat template so both
        # layer/vector fitting and the held-out coding audit use the same
        # semantic decision convention as the τ² evaluation.
        canonical_root = (
            PROJECT_ROOT
            / "results"
            / "section6_generalization"
            / "mistral_3p2_24b"
            / "converted_dataset"
        )
        verification = json.loads((canonical_root / "tokenizer_verification.json").read_text(encoding="utf-8"))
        system_prompt = str(verification.get("system_prompt_preview") or "")
        if not system_prompt:
            raise RuntimeError("Mistral coding prompt metadata has no system prompt")
        # Preserve the frozen 500-pair subset and its source-level split;
        # ``canonical_pairs.jsonl`` also contains the larger conversion pool.
        selected_sample_ids = {
            str(row["source_sample_id"])
            for row in read_jsonl(root / "pair_manifest.jsonl")
        }
        for row in read_jsonl(canonical_root / "canonical_pairs.jsonl"):
            if str(row.get("sample_id")) not in selected_sample_ids:
                continue
            tools = row.get("tools_schema")
            if not isinstance(tools, list) or not tools:
                raise RuntimeError(f"Mistral coding pair {row.get('sample_id')} has no tool schema")

            def render_native(user_content: Any) -> list[int]:
                encoded = tokenizer.apply_chat_template(
                    [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": str(user_content)},
                    ],
                    tools=tools,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_dict=True,
                    return_tensors="pt",
                )
                ids = encoded["input_ids"][0]
                return [int(value) for value in ids.detach().cpu().tolist()]

            sample_id = str(row["sample_id"])
            pairs.append(
                CodingPair(
                    pair_id=sample_id,
                    split=str(row.get("split", "train")),
                    clean_ids=render_native(row["user_content_clean"]),
                    corrupt_ids=render_native(row["user_content_corrupt"]),
                )
            )
    elif spec.family == "devstral":
        for row in read_jsonl(root / "manifest.jsonl"):
            pair_id = int(row["pair_id"])
            clean_path = resolve_existing_path(
                str(row["clean_path"]), [root / "clean" / f"clean_{pair_id}.txt"]
            )
            corrupt_path = resolve_existing_path(
                str(row["corrupt_path"]), [root / "corrupt" / f"corrupt_{pair_id}.txt"]
            )
            pairs.append(
                CodingPair(
                    pair_id=f"pair_{pair_id}",
                    split=str(row.get("split", "train")),
                    clean_ids=ids_from_text(tokenizer, clean_path.read_text(encoding="utf-8")),
                    corrupt_ids=ids_from_text(tokenizer, corrupt_path.read_text(encoding="utf-8")),
                )
            )
    else:
        raise ValueError(spec.family)
    if not pairs:
        raise RuntimeError(f"No coding pairs for {spec.key}")
    return pairs


def coding_prepared(pairs: Sequence[CodingPair], side: str) -> list[PreparedPrefix]:
    if side not in {"clean", "corrupt"}:
        raise ValueError(side)
    return [
        PreparedPrefix(
            candidate_id=f"coding_{side}_{pair.pair_id}",
            input_ids=pair.clean_ids if side == "clean" else pair.corrupt_ids,
            meta={"candidate_id": f"coding_{side}_{pair.pair_id}", "coding_pair_id": pair.pair_id, "coding_side": side},
        )
        for pair in pairs
    ]


def estimate_coding_vector(
    pairs: Sequence[CodingPair], *, model: Any, tokenizer: Any, device: torch.device, layer: int, batch_size: int
) -> torch.Tensor:
    clean = coding_prepared(pairs, "clean")
    corrupt = coding_prepared(pairs, "corrupt")

    def mean_state(prefixes: Sequence[PreparedPrefix]) -> tuple[torch.Tensor, int]:
        total: torch.Tensor | None = None
        n = 0
        for batch_index, batch in enumerate(make_batches(prefixes, batch_size), start=1):
            store: dict[str, torch.Tensor] = {}
            inputs = make_left_padded_batch(batch, int(tokenizer.pad_token_id), device)
            with temporary_capture(model, layer=layer, store=store):
                _ = model_forward(model, inputs)
            states = store.get("state")
            if states is None:
                raise RuntimeError("Capture hook did not run")
            current = states.sum(dim=0)
            total = current if total is None else total + current
            n += int(states.shape[0])
            del inputs, states, current
            torch.cuda.empty_cache()
            if batch_index % 50 == 0:
                print(json.dumps({"coding_capture_layer": layer, "batches": batch_index}), flush=True)
        if total is None:
            raise RuntimeError("No coding states captured")
        return total, n
    clean_sum, clean_count = mean_state(clean)
    corrupt_sum, corrupt_count = mean_state(corrupt)
    if clean_count != corrupt_count or clean_count != len(pairs):
        raise RuntimeError("Coding clean/corrupt count mismatch")
    vector = (clean_sum / clean_count - corrupt_sum / corrupt_count).float().contiguous()
    if not torch.isfinite(vector).all() or vector.norm().item() == 0:
        raise RuntimeError("Invalid coding vector")
    return vector


def coding_validation(
    pairs: Sequence[CodingPair],
    *,
    model: Any,
    tokenizer: Any,
    device: torch.device,
    tool_token_id: int,
    layer: int,
    vector: torch.Tensor,
    batch_size: int,
) -> dict[str, Any]:
    clean = coding_prepared(pairs, "clean")
    corrupt = coding_prepared(pairs, "corrupt")
    baseline = Condition("baseline", "baseline", 0.0, None, "no intervention")
    minus = Condition("minus_mean_diff_alpha_1", "coding_direction", 1.0, -vector, "subtract coding mean difference")
    plus = Condition("plus_mean_diff_alpha_1", "coding_direction", 1.0, vector, "add coding mean difference")
    clean_baseline, _ = evaluate_condition(
        clean, model=model, tokenizer=tokenizer, device=device, tool_token_id=tool_token_id, layer=layer, condition=baseline, batch_size=batch_size
    )
    clean_minus, _ = evaluate_condition(
        clean, model=model, tokenizer=tokenizer, device=device, tool_token_id=tool_token_id, layer=layer, condition=minus, batch_size=batch_size
    )
    corrupt_baseline, _ = evaluate_condition(
        corrupt, model=model, tokenizer=tokenizer, device=device, tool_token_id=tool_token_id, layer=layer, condition=baseline, batch_size=batch_size
    )
    corrupt_plus, _ = evaluate_condition(
        corrupt, model=model, tokenizer=tokenizer, device=device, tool_token_id=tool_token_id, layer=layer, condition=plus, batch_size=batch_size
    )
    clean_base = {row["coding_pair_id"]: row for row in clean_baseline}
    corrupt_base = {row["coding_pair_id"]: row for row in corrupt_baseline}
    strict_drop = sum(
        bool(clean_base[row["coding_pair_id"]]["is_tool_call_top1"]) and not bool(row["is_tool_call_top1"])
        for row in clean_minus
    )
    strict_flip = sum(
        not bool(corrupt_base[row["coding_pair_id"]]["is_tool_call_top1"]) and bool(row["is_tool_call_top1"])
        for row in corrupt_plus
    )
    n = len(pairs)
    return {
        "n_heldout_pairs": n,
        "clean_baseline_tool_top1": sum(bool(row["is_tool_call_top1"]) for row in clean_baseline),
        "clean_minus_tool_top1": sum(bool(row["is_tool_call_top1"]) for row in clean_minus),
        "strict_drop": strict_drop,
        "strict_drop_rate": strict_drop / max(sum(bool(row["is_tool_call_top1"]) for row in clean_baseline), 1),
        "corrupt_baseline_tool_top1": sum(bool(row["is_tool_call_top1"]) for row in corrupt_baseline),
        "corrupt_plus_tool_top1": sum(bool(row["is_tool_call_top1"]) for row in corrupt_plus),
        "strict_flip": strict_flip,
        "strict_flip_rate": strict_flip / max(sum(not bool(row["is_tool_call_top1"]) for row in corrupt_baseline), 1),
    }


def prepare_screen_pool(
    rows: Sequence[dict[str, Any]], *, tokenizer: Any, spec: ModelSpec, raw_root: Path, max_context_tokens: int
) -> tuple[list[PreparedPrefix], list[dict[str, Any]]]:
    prepared: list[PreparedPrefix] = []
    rejected: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        prefix, reason = render_prefix(
            row, tokenizer=tokenizer, spec=spec, raw_root=raw_root, max_context_tokens=max_context_tokens
        )
        if prefix is not None:
            prepared.append(prefix)
        else:
            assert reason is not None
            rejected.append(reason)
        if index % 200 == 0:
            print(json.dumps({"rendered_prefixes": index, "total": len(rows)}), flush=True)
    return prepared, rejected


def build_conditions(vector: torch.Tensor, random_unit: torch.Tensor, *, kind: str) -> list[Condition]:
    random_matched = random_unit * vector.norm()
    baseline = Condition("baseline_no_hook_alpha_0", "baseline", 0.0, None, "no intervention")
    if kind == "suppression":
        return [
            baseline,
            Condition("minus_mean_diff_alpha_1", "coding_direction_suppression", 1.0, -vector, "subtract coding vector"),
            Condition("minus_mean_diff_alpha_1.5", "coding_direction_suppression", 1.5, -1.5 * vector, "subtract 1.5x coding vector"),
            Condition("minus_random_norm_matched_alpha_1", "norm_matched_random_control", 1.0, -random_matched, "subtract norm-matched random direction"),
            Condition("minus_random_norm_matched_alpha_1.5", "norm_matched_random_control", 1.5, -1.5 * random_matched, "subtract 1.5x norm-matched random direction"),
        ]
    if kind == "induction":
        return [
            baseline,
            Condition("plus_mean_diff_alpha_1", "coding_direction_induction", 1.0, vector, "add coding vector"),
            Condition("plus_mean_diff_alpha_1.5", "coding_direction_induction", 1.5, 1.5 * vector, "add 1.5x coding vector"),
            Condition("plus_random_norm_matched_alpha_1", "norm_matched_random_control", 1.0, random_matched, "add norm-matched random direction"),
        ]
    raise ValueError(kind)


def run_intervention_suite(
    prepared: Sequence[PreparedPrefix],
    *,
    kind: str,
    model: Any,
    tokenizer: Any,
    device: torch.device,
    tool_token_id: int,
    layer: int,
    vector: torch.Tensor,
    random_unit: torch.Tensor,
    batch_size: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, dict[str, int]], dict[str, list[dict[str, Any]]]]:
    conditions = build_conditions(vector, random_unit, kind=kind)
    by_condition: dict[str, list[dict[str, Any]]] = {}
    hook_stats: dict[str, dict[str, int]] = {}
    all_rows: list[dict[str, Any]] = []
    for condition in conditions:
        rows, stats = evaluate_condition(
            prepared,
            model=model,
            tokenizer=tokenizer,
            device=device,
            tool_token_id=tool_token_id,
            layer=layer,
            condition=condition,
            batch_size=batch_size,
        )
        by_condition[condition.name] = rows
        hook_stats[condition.name] = stats
        all_rows.extend(rows)
    baseline = {row["candidate_id"]: row for row in by_condition["baseline_no_hook_alpha_0"]}
    if len(baseline) != len(prepared):
        raise RuntimeError("Duplicated intervention baseline candidates")
    summaries = [summarize_condition(by_condition[condition.name], baseline) for condition in conditions]
    return all_rows, summaries, hook_stats, by_condition


def parse_well_formed_call(text: str, spec: ModelSpec, tool_names: set[str]) -> tuple[bool, str | None]:
    if spec.family == "granite":
        if not text.startswith("<|tool_call|>"):
            return False, None
        payload = text[len("<|tool_call|>") :].strip()
        try:
            parsed, _ = json.JSONDecoder().raw_decode(payload)
        except (json.JSONDecodeError, ValueError):
            return False, None
        if not isinstance(parsed, list) or not parsed:
            return False, None
        first = parsed[0]
        if not isinstance(first, dict):
            return False, None
        name = first.get("name")
        args = first.get("arguments")
        return bool(isinstance(name, str) and name in tool_names and isinstance(args, dict)), str(name) if name else None
    if not text.startswith("[TOOL_CALLS]"):
        return False, None
    payload = text[len("[TOOL_CALLS]") :]
    match = re.match(r"([A-Za-z0-9_]+)(?:\[CALL_ID\][A-Za-z0-9]+)?\[ARGS\]", payload)
    if match is None:
        return False, None
    name = match.group(1)
    argument_text = payload[match.end() :]
    for marker in ("</s>", "[TOOL_RESULTS]", "<|end_of_text|>"):
        argument_text = argument_text.split(marker, 1)[0]
    try:
        arguments, _ = json.JSONDecoder().raw_decode(argument_text.lstrip())
    except (json.JSONDecodeError, ValueError):
        return False, name
    return bool(name in tool_names and isinstance(arguments, dict)), name


def greedy_induction_audit(
    prepared: Sequence[PreparedPrefix],
    *,
    model: Any,
    tokenizer: Any,
    device: torch.device,
    tool_token_id: int,
    layer: int,
    vector: torch.Tensor,
    spec: ModelSpec,
    raw_root: Path,
    max_new_tokens: int,
    direct_rows: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    tool_schemas = json.loads((raw_root / "tau2_tool_schemas.json").read_text(encoding="utf-8"))
    tool_names = {str(item["function"]["name"]) for item in tool_schemas}
    rows: list[dict[str, Any]] = []
    condition = Condition("plus_mean_diff_alpha_1", "coding_direction_induction", 1.0, vector, "add coding vector")
    with temporary_last_token_addition(model, layer=layer, delta_cpu=condition.delta_cpu, prefill_only=True) as stats:
        for index, prefix in enumerate(prepared, start=1):
            inputs = make_left_padded_batch([prefix], int(tokenizer.pad_token_id), device)
            prompt_length = int(inputs["input_ids"].shape[1])
            with torch.inference_mode():
                generated = model.generate(
                    **inputs,
                    do_sample=False,
                    use_cache=True,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=int(tokenizer.pad_token_id),
                    eos_token_id=tokenizer.eos_token_id,
                )
            new_ids = [int(value) for value in generated[0, prompt_length:].detach().cpu().tolist()]
            text = tokenizer.decode(new_ids, clean_up_tokenization_spaces=False)
            well_formed, tool_name = parse_well_formed_call(text, spec, tool_names)
            direct = direct_rows[prefix.candidate_id]
            rows.append(
                {
                    "candidate_id": prefix.candidate_id,
                    "trace_index": prefix.meta["trace_index"],
                    "first_generated_token_id": new_ids[0] if new_ids else None,
                    "first_generated_token_text": token_text(tokenizer, new_ids[0]) if new_ids else "",
                    "first_generated_token_is_tool_call": bool(new_ids and new_ids[0] == tool_token_id),
                    "contains_tool_call": bool(tool_token_id in new_ids),
                    "well_formed_tool_call": well_formed,
                    "parsed_tool_name": tool_name,
                    "generated_token_ids": new_ids,
                    "generated_text": text,
                    "first_token_matches_direct_forward": bool(new_ids and new_ids[0] == int(direct["top1_token_id"])),
                }
            )
            del inputs, generated
            torch.cuda.empty_cache()
            if index % 10 == 0 or index == len(prepared):
                print(json.dumps({"greedy_induction": [index, len(prepared)]}), flush=True)
    return rows, stats


def main() -> None:
    args = parse_args()
    if args.coding_batch_size < 1 or args.tau_batch_size < 1 or args.final_count < 1:
        raise ValueError("Batch sizes and final count must be positive")
    if not 0.0 <= args.max_baseline_tool_probability <= 1.0:
        raise ValueError("--max-baseline-tool-probability must be in [0, 1]")
    spec = SPECS[args.model]
    raw_root = args.raw_root.resolve()
    prepared_root = args.prepared_root.resolve()
    output_root = args.output_root.resolve() / spec.key
    output_root.mkdir(parents=True, exist_ok=True)
    required = [
        spec.model_path,
        raw_root / "tau2_system_prompt.txt",
        raw_root / "tau2_tool_schemas.json",
        prepared_root / "screen_pool.jsonl",
        prepared_root / "text_reply_induction_terminal" / "screen_pool.jsonl",
    ]
    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)

    model, tokenizer, device = load_tokenizer_and_model(spec)
    try:
        tool_token_id, token_info = resolve_tool_token_id(tokenizer, spec)
        if spec.coding_layer >= len(resolve_layers(model)):
            raise RuntimeError(f"L{spec.coding_layer} is not present in {spec.display_name}")
        run_config = {
            "started_unix": time.time(),
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "model": spec.display_name,
            "model_path": str(spec.model_path),
            "model_family": spec.family,
            "tool_token": token_info,
            "coding_localized_layer": spec.coding_layer,
            "coding_layer_source": "existing family-specific coding localization; Telecom data are never used for layer selection",
            "intervention_hook": f"post-output of target decoder layer L{spec.coding_layer}, final real prompt token only",
            "tau2_prompt_adapters": {
                "granite": "serialize historical assistant tool calls as <|tool_call|> JSON; retain role-tagged tool results",
                "mistral": "use native tool template and remap historical call IDs to Mistral's required nine alphanumeric characters",
                "devstral": "use native tool template and omit only τ²'s initial canned assistant greeting, required by its user-first alternation template",
            },
            "raw_files": {
                path.name: {"path": str(path), "sha256": sha256_file(path)}
                for path in (raw_root / "tau2_system_prompt.txt", raw_root / "tau2_tool_schemas.json")
            },
            "arguments": vars(args),
        }
        write_json(output_root / "run_config.json", run_config)

        # Phase 1: coding-only direction.  Use the source dataset's frozen
        # train/test partitions, so the validation below remains independent
        # of vector estimation as well as independent of τ².
        coding_pairs = load_coding_pairs(spec, tokenizer)
        coding_train = [pair for pair in coding_pairs if pair.split == "train"]
        coding_test = [pair for pair in coding_pairs if pair.split == "test"]
        if not coding_train or not coding_test:
            raise RuntimeError("Coding data must expose nonempty train/test splits")
        print(json.dumps({"coding_pairs": {"train": len(coding_train), "test": len(coding_test)}}), flush=True)
        vector = estimate_coding_vector(
            coding_train,
            model=model,
            tokenizer=tokenizer,
            device=device,
            layer=spec.coding_layer,
            batch_size=args.coding_batch_size,
        )
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(args.seed) + {"granite": 11, "mistral": 23, "devstral": 37}[spec.key])
        random_unit = torch.randn(vector.shape, generator=generator, dtype=torch.float32)
        random_unit = random_unit / random_unit.norm().clamp_min(1e-12)
        coding_validation_payload = coding_validation(
            coding_test,
            model=model,
            tokenizer=tokenizer,
            device=device,
            tool_token_id=tool_token_id,
            layer=spec.coding_layer,
            vector=vector,
            batch_size=args.coding_batch_size,
        )
        bundle = {
            "model": spec.display_name,
            "model_path": str(spec.model_path),
            "source_domain": "coding clean/corrupt pairs only",
            "fit_split": "train",
            "validation_split": "test",
            "layer": spec.coding_layer,
            "hook_kind": "post",
            "mean_diff": vector,
            "random_direction_unit": random_unit,
            "random_seed": int(args.seed),
            "tool_token": token_info,
        }
        vector_path = output_root / "coding_vector_bundle.pt"
        torch.save(bundle, vector_path)
        coding_summary = {
            "fit_pairs": len(coding_train),
            "heldout_pairs": len(coding_test),
            "layer": spec.coding_layer,
            "mean_diff_norm": float(vector.norm().item()),
            "validation": coding_validation_payload,
            "vector_bundle": str(vector_path),
        }
        write_json(output_root / "coding_vector_summary.json", coding_summary)

        # Phase 2a: each target model gets its own natural positive screen.
        tool_source = read_jsonl(prepared_root / "screen_pool.jsonl")
        tool_screen_input = balanced_subset_tool(tool_source, limit=args.tool_screen_limit, seed=args.seed)
        tool_prepared, tool_rejected = prepare_screen_pool(
            tool_screen_input, tokenizer=tokenizer, spec=spec, raw_root=raw_root, max_context_tokens=args.max_context_tokens
        )
        write_json(output_root / "tool_render_audit.json", {
            "source_rows": len(tool_source), "screen_input_rows": len(tool_screen_input), "rendered_rows": len(tool_prepared), "rejections": tool_rejected
        })
        baseline = Condition("baseline_no_hook_alpha_0", "baseline", 0.0, None, "no intervention")
        tool_screen, tool_screen_stats = evaluate_condition(
            tool_prepared,
            model=model,
            tokenizer=tokenizer,
            device=device,
            tool_token_id=tool_token_id,
            layer=spec.coding_layer,
            condition=baseline,
            batch_size=args.tau_batch_size,
        )
        write_jsonl(output_root / "tool_baseline_screen.jsonl", tool_screen)
        selected_tool_manifest = select_tool_rows(tool_screen, final_count=args.final_count, seed=args.seed)
        write_jsonl(output_root / "selected_tool_prefixes.jsonl", selected_tool_manifest)
        tool_by_id = {str(row["candidate_id"]): row for row in tool_source}
        selected_tool_rows = [tool_by_id[str(row["candidate_id"])] for row in selected_tool_manifest]
        tool_selected, selected_tool_rejected = prepare_screen_pool(
            selected_tool_rows, tokenizer=tokenizer, spec=spec, raw_root=raw_root, max_context_tokens=args.max_context_tokens
        )
        if selected_tool_rejected or len(tool_selected) != args.final_count:
            raise RuntimeError("Selected tool prefixes no longer render")

        # Phase 2b: independently screen natural text replies for induction.
        induction_root = prepared_root / "text_reply_induction_terminal"
        induction_source = read_jsonl(induction_root / "screen_pool.jsonl")
        induction_screen_input = balanced_subset_induction(
            induction_source, limit=args.induction_screen_limit, seed=args.seed
        )
        induction_prepared, induction_rejected = prepare_screen_pool(
            induction_screen_input, tokenizer=tokenizer, spec=spec, raw_root=raw_root, max_context_tokens=args.max_context_tokens
        )
        write_json(output_root / "induction_render_audit.json", {
            "source_rows": len(induction_source), "screen_input_rows": len(induction_screen_input), "rendered_rows": len(induction_prepared), "rejections": induction_rejected
        })
        induction_screen, induction_screen_stats = evaluate_condition(
            induction_prepared,
            model=model,
            tokenizer=tokenizer,
            device=device,
            tool_token_id=tool_token_id,
            layer=spec.coding_layer,
            condition=baseline,
            batch_size=args.tau_batch_size,
        )
        write_jsonl(output_root / "induction_baseline_screen.jsonl", induction_screen)
        selected_induction_manifest = select_induction_rows(
            induction_screen,
            final_count=args.final_count,
            max_probability=args.max_baseline_tool_probability,
            seed=args.seed,
        )
        write_jsonl(output_root / "selected_induction_prefixes.jsonl", selected_induction_manifest)
        induction_by_id = {str(row["candidate_id"]): row for row in induction_source}
        selected_induction_rows = [induction_by_id[str(row["candidate_id"])] for row in selected_induction_manifest]
        induction_selected, selected_induction_rejected = prepare_screen_pool(
            selected_induction_rows, tokenizer=tokenizer, spec=spec, raw_root=raw_root, max_context_tokens=args.max_context_tokens
        )
        if selected_induction_rejected or len(induction_selected) != args.final_count:
            raise RuntimeError("Selected induction prefixes no longer render")

        # Actual frozen-vector interventions; all Telecom material was screened
        # only after the vector and layer had been fixed above.
        tool_rows, tool_summaries, tool_hook_stats, tool_by_condition = run_intervention_suite(
            tool_selected,
            kind="suppression",
            model=model,
            tokenizer=tokenizer,
            device=device,
            tool_token_id=tool_token_id,
            layer=spec.coding_layer,
            vector=vector,
            random_unit=random_unit,
            batch_size=args.tau_batch_size,
        )
        induction_rows, induction_summaries, induction_hook_stats, induction_by_condition = run_intervention_suite(
            induction_selected,
            kind="induction",
            model=model,
            tokenizer=tokenizer,
            device=device,
            tool_token_id=tool_token_id,
            layer=spec.coding_layer,
            vector=vector,
            random_unit=random_unit,
            batch_size=args.tau_batch_size,
        )
        write_jsonl(output_root / "suppression_per_sample.jsonl", tool_rows)
        write_jsonl(output_root / "induction_per_sample.jsonl", induction_rows)
        write_csv(output_root / "suppression_summary.csv", tool_summaries)
        write_csv(output_root / "induction_summary.csv", induction_summaries)
        write_json(output_root / "suppression_summary.json", {"conditions": tool_summaries, "hook_stats": tool_hook_stats})
        write_json(output_root / "induction_summary.json", {"conditions": induction_summaries, "hook_stats": induction_hook_stats})

        greedy_rows: list[dict[str, Any]] = []
        greedy_stats = {"hook_calls": 0, "modified_calls": 0}
        if not args.skip_generation:
            greedy_rows, greedy_stats = greedy_induction_audit(
                induction_selected,
                model=model,
                tokenizer=tokenizer,
                device=device,
                tool_token_id=tool_token_id,
                layer=spec.coding_layer,
                vector=vector,
                spec=spec,
                raw_root=raw_root,
                max_new_tokens=args.max_new_tokens,
                direct_rows={row["candidate_id"]: row for row in induction_by_condition["plus_mean_diff_alpha_1"]},
            )
        write_jsonl(output_root / "induction_alpha1_generations.jsonl", greedy_rows)
        result = {
            "model": spec.display_name,
            "coding_vector": coding_summary,
            "screening": {
                "tool": {"screen_stats": tool_screen_stats, "selected": len(selected_tool_manifest), "screened": len(tool_screen)},
                "induction": {"screen_stats": induction_screen_stats, "selected": len(selected_induction_manifest), "screened": len(induction_screen)},
            },
            "suppression": tool_summaries,
            "induction": induction_summaries,
            "induction_alpha1_generation": {
                "n": len(greedy_rows),
                "first_tool_call": sum(bool(row["first_generated_token_is_tool_call"]) for row in greedy_rows),
                "well_formed_tool_call": sum(bool(row["well_formed_tool_call"]) for row in greedy_rows),
                "direct_forward_match": sum(bool(row["first_token_matches_direct_forward"]) for row in greedy_rows),
                "hook_stats": greedy_stats,
            },
            "completed_unix": time.time(),
        }
        write_json(output_root / "final_result.json", result)
        print(json.dumps({"completed": True, "model": spec.key, "result": result}, ensure_ascii=False), flush=True)
    finally:
        del model
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
