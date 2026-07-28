#!/usr/bin/env python3
"""Run Reviewer CJrQ's schema-identity table on audited v5 datasets.

This runner is intentionally separate from the historical v4 D1 script.  It
uses each target model's own v5 release, reserves disjoint training examples
for layer selection and vector estimation, and scores every one of the 300
held-out pairs.  The output directory is result-side only: no v5 file is ever
written or re-screened.

The reported table cells are ``cosine / strict flip / strict drop`` for:

* V1, renamed: change only the tool function name;
* V2, removed: opaque ``f1``, empty description, and neutral parameter names;
* V5, mismatched: a weather tool.

For native Mistral inputs V0 always uses the stored IDs.  Edited schemas are
rendered directly from ``messages`` and ``tools`` through
``apply_chat_template(tokenize=True)``.  It never decodes a prompt and feeds
that text back through a tokenizer.
"""

from __future__ import annotations

import argparse
import copy
import csv
import gc
import hashlib
import importlib.util
import json
import math
import os
import platform
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch


# A local sklearn installation is incompatible with the Transformers import
# chain in this environment.  This mirrors the narrowly-scoped workaround in
# the existing native-interface runner.
_ORIGINAL_FIND_SPEC = importlib.util.find_spec


def _patched_find_spec(name: str, package: str | None = None):
    if name == "sklearn" and os.environ.get("MECH_ENABLE_TRANSFORMERS_SKLEARN", "0") != "1":
        return None
    return _ORIGINAL_FIND_SPEC(name, package)


importlib.util.find_spec = _patched_find_spec
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, Mistral3ForConditionalGeneration
finally:
    importlib.util.find_spec = _ORIGINAL_FIND_SPEC


THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[2]
V5_ROOT = PROJECT_ROOT / "datasets" / "v5_model_specific_balanced"
VARIANT_ORDER = ("V0", "V1", "V2", "V5")
TABLE_VARIANTS = ("V1", "V2", "V5")
MODEL_ROOT = Path(os.environ.get("MODEL_ROOT", PROJECT_ROOT / "external" / "models")).expanduser()


def default_model_path(environment_variable: str, directory_name: str) -> str:
    """Resolve a portable checkpoint default while retaining a CLI override."""

    return str(Path(os.environ.get(environment_variable, MODEL_ROOT / directory_name)).expanduser())


MODEL_SPECS: dict[str, dict[str, Any]] = {
    "qwen3_4b": {
        "label": "Qwen3-4B",
        "family": "qwen_text",
        "loader": "auto",
        "model_path": default_model_path("QWEN3_4B_PATH", "Qwen3-4B"),
        "batch_size": 12,
    },
    "qwen3_8b": {
        "label": "Qwen3-8B",
        "family": "qwen_text",
        "loader": "auto",
        "model_path": default_model_path("QWEN3_8B_PATH", "Qwen3-8B"),
        "batch_size": 8,
    },
    "qwen3_14b": {
        "label": "Qwen3-14B",
        "family": "qwen_text",
        "loader": "auto",
        "model_path": default_model_path("QWEN3_14B_PATH", "Qwen3-14B"),
        "batch_size": 4,
    },
    "qwen35_4b": {
        "label": "Qwen3.5-4B",
        "family": "qwen_text",
        "loader": "auto",
        "model_path": default_model_path("QWEN35_4B_PATH", "Qwen3.5-4B"),
        "batch_size": 10,
    },
    "qwen35_9b": {
        "label": "Qwen3.5-9B",
        "family": "qwen_text",
        "loader": "auto",
        "model_path": default_model_path("QWEN35_9B_PATH", "Qwen3.5-9B"),
        "batch_size": 8,
    },
    "mistral_3p2_24b": {
        "label": "Mistral-Small-3.2-24B-Instruct-2506",
        "family": "mistral_native",
        "loader": "mistral",
        "model_path": default_model_path("MISTRAL_3P2_24B_PATH", "Mistral-Small-3.2-24B-Instruct-2506"),
        "batch_size": 4,
    },
    "granite_3p3_8b": {
        "label": "Granite-3.3-8B-Instruct",
        "family": "granite_text",
        "loader": "auto",
        "model_path": default_model_path("GRANITE_3P3_8B_PATH", "granite-3.3-8b-instruct"),
        "batch_size": 10,
    },
}


# These preserve the underlying affordance while changing the function-name
# string.  Descriptions and parameter schemas are unchanged in V1.
RENAMED_TOOL_NAMES = {
    "write_file": "apply_patch",
    "get_details_by_id": "lookup_record",
    "get_data_usage": "view_usage",
    "enable_roaming": "activate_roaming",
    "refuel_data": "top_up_data",
    "get_bills_for_customer": "list_customer_bills",
    "transfer_to_human_agents": "escalate_to_support",
    "can_send_mms": "check_mms_capability",
    "perform_account_action": "complete_account_request",
}

WEATHER_FUNCTION = {
    "name": "get_weather",
    "description": "Retrieve the current weather for a location.",
    "parameters": {
        "type": "object",
        "properties": {"location": {"type": "string"}},
        "required": ["location"],
    },
}


@dataclass(frozen=True)
class SourcePrompt:
    """An immutable prompt read from a v5 source file."""

    relpath: str
    file_sha256: str
    prompt_sha256: str
    data_format: str
    text: str | None
    messages: list[dict[str, Any]] | None
    tools: list[dict[str, Any]]
    stored_input_ids: tuple[int, ...]
    source_schema_sha256: str
    source_tool_name: str


@dataclass(frozen=True)
class PairRecord:
    order: int
    sample_id: str
    split: str
    manifest: dict[str, Any]
    clean: SourcePrompt
    corrupt: SourcePrompt


@dataclass(frozen=True)
class PromptItem:
    sample_id: str
    split: str
    side: str
    variant: str
    input_ids: tuple[int, ...]
    input_ids_sha256: str
    rendered_prompt_sha256: str
    source_prompt_sha256: str
    source_schema_sha256: str
    variant_schema_sha256: str
    render_method: str
    relpath: str
    source_tool_name: str
    variant_tool_name: str
    v0_template_roundtrip: bool | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-key", choices=tuple(MODEL_SPECS), required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--selection-pairs", type=int, default=64)
    parser.add_argument("--vector-pairs", type=int, default=136)
    parser.add_argument("--heldout-pairs", type=int, default=300)
    parser.add_argument("--allow-subset", action="store_true", help="Only for a smoke run; canonical v5 requires 64/136/300.")
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--attn-implementation", default="")
    parser.add_argument("--seed", type=int, default=20260728)
    return parser.parse_args()


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_json(value: Any) -> str:
    return sha256_text(canonical_json(value))


def sha256_input_ids(ids: Sequence[int]) -> str:
    # The v5 Mistral manifests hash a compact JSON-array representation.
    return sha256_text(json.dumps([int(value) for value in ids], separators=(",", ":")))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for field in row:
            if field not in seen:
                fields.append(field)
                seen.add(field)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path}: expected a JSON object")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise TypeError(f"{path}:{number}: expected a JSON object")
        rows.append(value)
    return rows


def run_command(command: Sequence[str]) -> str | None:
    try:
        completed = subprocess.run(command, cwd=PROJECT_ROOT, text=True, capture_output=True, check=False)
    except OSError:
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def git_provenance() -> dict[str, Any]:
    status = run_command(("git", "status", "--porcelain"))
    return {
        "commit": run_command(("git", "rev-parse", "HEAD")),
        "branch": run_command(("git", "branch", "--show-current")),
        "worktree_dirty": bool(status),
        "worktree_status_line_count": len(status.splitlines()) if status else 0,
    }


def gpu_snapshot() -> str | None:
    return run_command(("nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,utilization.gpu", "--format=csv,noheader"))


def clear_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def extract_text_tools_payload(text: str, family: str) -> tuple[int, int, str]:
    """Find the actual native tools payload, excluding prose mentioning tags."""

    if family == "granite_text":
        marker = "<|start_of_role|>available_tools<|end_of_role|>"
        closing = "<|end_of_text|>"
    elif family == "qwen_text":
        marker = "<tools>\n"
        closing = "</tools>"
    else:
        raise ValueError(f"Text-tools parser does not support family={family!r}")
    start = text.find(marker)
    if start < 0:
        raise ValueError(f"Prompt lacks native tools marker {marker!r}")
    start += len(marker)
    end = text.find(closing, start)
    if end < 0:
        raise ValueError(f"Prompt lacks native tools closing marker {closing!r}")
    return start, end, text[start:end]


def as_single_tool_list(value: Any, *, context: str) -> list[dict[str, Any]]:
    if isinstance(value, dict):
        tools = [value]
    elif isinstance(value, list):
        tools = value
    else:
        raise TypeError(f"{context}: tool payload is neither an object nor a list")
    if len(tools) != 1 or not isinstance(tools[0], dict):
        raise ValueError(f"{context}: this experiment requires exactly one current tool schema")
    function = tools[0].get("function")
    if not isinstance(function, dict) or not isinstance(function.get("name"), str):
        raise ValueError(f"{context}: tool lacks function.name")
    return tools


def source_tools_from_text(text: str, family: str, *, context: str) -> list[dict[str, Any]]:
    _start, _end, payload = extract_text_tools_payload(text, family)
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{context}: native tools payload is invalid JSON") from exc
    return as_single_tool_list(parsed, context=context)


def source_tool_name(tools: Sequence[dict[str, Any]], *, context: str) -> str:
    validated = as_single_tool_list(list(tools), context=context)
    return str(validated[0]["function"]["name"])


def safe_dataset_path(dataset_root: Path, relpath: str, *, context: str) -> Path:
    candidate = (dataset_root / relpath).resolve()
    try:
        candidate.relative_to(dataset_root.resolve())
    except ValueError as exc:
        raise ValueError(f"{context}: path escapes dataset root: {relpath}") from exc
    if not candidate.is_file():
        raise FileNotFoundError(f"{context}: missing source prompt {candidate}")
    return candidate


def load_source_prompt(
    dataset_root: Path,
    row: dict[str, Any],
    *,
    side: str,
    family: str,
) -> SourcePrompt:
    sample_id = str(row["sample_id"])
    relpath = str(row[f"{side}_relpath"])
    path = safe_dataset_path(dataset_root, relpath, context=f"{sample_id}/{side}")
    raw = path.read_text(encoding="utf-8")
    expected_prompt_hash = str(row[f"{side}_prompt_sha256"])
    if family == "mistral_native":
        payload = json.loads(raw)
        if not isinstance(payload, dict):
            raise TypeError(f"{sample_id}/{side}: native Mistral payload must be an object")
        if payload.get("prompt_format") != "mistral_native_input_ids":
            raise ValueError(f"{sample_id}/{side}: unexpected Mistral prompt format")
        messages = payload.get("messages")
        tools = payload.get("tools")
        input_ids = payload.get("input_ids")
        if not isinstance(messages, list) or not isinstance(tools, list) or not isinstance(input_ids, list):
            raise TypeError(f"{sample_id}/{side}: Mistral payload lacks messages/tools/input_ids lists")
        ids = tuple(int(value) for value in input_ids)
        if not ids:
            raise ValueError(f"{sample_id}/{side}: empty stored input_ids")
        actual_prompt_hash = sha256_input_ids(ids)
        if actual_prompt_hash != expected_prompt_hash:
            raise ValueError(f"{sample_id}/{side}: stored input_ids hash does not match manifest")
        declared_hash = payload.get("input_ids_sha256")
        if declared_hash is not None and str(declared_hash) != actual_prompt_hash:
            raise ValueError(f"{sample_id}/{side}: embedded input_ids hash does not match stored IDs")
        checked_tools = as_single_tool_list(tools, context=f"{sample_id}/{side}")
        return SourcePrompt(
            relpath=relpath,
            file_sha256=sha256_text(raw),
            prompt_sha256=actual_prompt_hash,
            data_format="mistral_native_input_ids",
            text=None,
            messages=copy.deepcopy(messages),
            tools=copy.deepcopy(checked_tools),
            stored_input_ids=ids,
            source_schema_sha256=sha256_json(checked_tools),
            source_tool_name=source_tool_name(checked_tools, context=f"{sample_id}/{side}"),
        )
    actual_prompt_hash = sha256_text(raw)
    if actual_prompt_hash != expected_prompt_hash:
        raise ValueError(f"{sample_id}/{side}: source prompt SHA-256 does not match manifest")
    tools = source_tools_from_text(raw, family, context=f"{sample_id}/{side}")
    return SourcePrompt(
        relpath=relpath,
        file_sha256=actual_prompt_hash,
        prompt_sha256=actual_prompt_hash,
        data_format="native_text",
        text=raw,
        messages=None,
        tools=copy.deepcopy(tools),
        stored_input_ids=(),
        source_schema_sha256=sha256_json(tools),
        source_tool_name=source_tool_name(tools, context=f"{sample_id}/{side}"),
    )


def load_pairs(dataset_root: Path, *, model_key: str, family: str) -> tuple[dict[str, Any], list[PairRecord]]:
    summary_path = dataset_root / "summary.json"
    manifest_path = dataset_root / "manifest.jsonl"
    if not summary_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(f"{dataset_root}: expected summary.json and manifest.jsonl")
    summary = read_json(summary_path)
    if summary.get("dataset_version") != "v5_model_specific_balanced":
        raise ValueError(f"{dataset_root}: not a v5 model-specific dataset")
    if str(summary.get("model_key")) != model_key:
        raise ValueError(f"{dataset_root}: summary model key does not match --model-key={model_key}")
    rows = read_jsonl(manifest_path)
    if len(rows) != 500:
        raise ValueError(f"{dataset_root}: expected 500 manifest pairs, found {len(rows)}")
    records: list[PairRecord] = []
    required = {
        "sample_id",
        "split",
        "clean_relpath",
        "corrupt_relpath",
        "clean_prompt_sha256",
        "corrupt_prompt_sha256",
        "clean_is_tool_top1",
        "corrupt_is_tool_top1",
    }
    for order, row in enumerate(rows):
        missing = required - set(row)
        if missing:
            raise KeyError(f"manifest row {order}: missing {sorted(missing)}")
        split = str(row["split"])
        if split not in {"train", "heldout"}:
            raise ValueError(f"{row['sample_id']}: unexpected split {split!r}")
        clean = load_source_prompt(dataset_root, row, side="clean", family=family)
        corrupt = load_source_prompt(dataset_root, row, side="corrupt", family=family)
        records.append(PairRecord(order=order, sample_id=str(row["sample_id"]), split=split, manifest=row, clean=clean, corrupt=corrupt))
    if len({record.sample_id for record in records}) != len(records):
        raise ValueError(f"{dataset_root}: duplicate sample IDs in manifest")
    counts = {split: sum(record.split == split for record in records) for split in ("train", "heldout")}
    if counts != {"train": 200, "heldout": 300}:
        raise ValueError(f"{dataset_root}: expected 200 train / 300 heldout, found {counts}")
    return summary, records


def neutralize_parameter_names(value: Any) -> Any:
    """Remove semantic parameter names/descriptions while retaining structure."""

    if isinstance(value, list):
        return [neutralize_parameter_names(item) for item in value]
    if not isinstance(value, dict):
        return copy.deepcopy(value)
    properties = value.get("properties")
    mapping: dict[str, str] = {}
    if isinstance(properties, dict):
        mapping = {name: f"arg_{index}" for index, name in enumerate(properties, start=1)}
    result: dict[str, Any] = {}
    for key, child in value.items():
        # These fields can carry the original affordance even after the top
        # level function description has been removed.
        if key in {"description", "title", "examples", "default"}:
            continue
        if key == "properties" and isinstance(child, dict):
            result[key] = {mapping[name]: neutralize_parameter_names(item) for name, item in child.items()}
        elif key == "required" and isinstance(child, list):
            result[key] = [mapping.get(str(item), str(item)) for item in child]
        else:
            result[key] = neutralize_parameter_names(child)
    return result


def mutate_tools(source: SourcePrompt, variant: str) -> list[dict[str, Any]]:
    tools = copy.deepcopy(source.tools)
    tool = as_single_tool_list(tools, context=f"{source.relpath}/{variant}")[0]
    function = tool["function"]
    old_name = str(function["name"])
    if variant == "V0":
        return tools
    if variant == "V1":
        try:
            function["name"] = RENAMED_TOOL_NAMES[old_name]
        except KeyError as exc:
            raise KeyError(f"No predeclared affordance-preserving rename for {old_name!r}") from exc
        return tools
    if variant == "V2":
        function["name"] = "f1"
        function["description"] = ""
        parameters = function.get("parameters")
        if not isinstance(parameters, dict):
            raise ValueError(f"{source.relpath}: V2 requires a function parameter object")
        function["parameters"] = neutralize_parameter_names(parameters)
        return tools
    if variant == "V5":
        tool["function"] = copy.deepcopy(WEATHER_FUNCTION)
        return tools
    raise ValueError(f"Unknown schema variant {variant!r}")


def serialize_text_tools(tools: list[dict[str, Any]], *, original_payload: str, family: str) -> str:
    value: Any = tools if family == "granite_text" else tools[0]
    if original_payload.lstrip().startswith("["):
        rendered = json.dumps(value, ensure_ascii=False, indent=4)
    elif '"type": ' in original_payload or '"name": ' in original_payload:
        rendered = json.dumps(value, ensure_ascii=False)
    else:
        rendered = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    if original_payload.endswith("\n") and not rendered.endswith("\n"):
        rendered += "\n"
    return rendered


def replace_first_function_name(payload: str, replacement: str) -> str:
    pattern = re.compile(r'("name"\s*:\s*")[^"]*(")')
    rendered, count = pattern.subn(lambda match: match.group(1) + replacement + match.group(2), payload, count=1)
    if count != 1:
        raise ValueError("Could not locate exactly one function.name in native tools payload")
    return rendered


def normalize_rendered_ids(value: Any) -> tuple[int, ...]:
    if isinstance(value, dict) or hasattr(value, "keys"):
        value = value["input_ids"]
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, tuple):
        value = list(value)
    if isinstance(value, list) and value and isinstance(value[0], list):
        if len(value) != 1:
            raise ValueError("Expected one rendered prompt, not a batch")
        value = value[0]
    if not isinstance(value, list) or not all(isinstance(item, int) for item in value):
        raise TypeError(f"Expected a list of token IDs, got {type(value).__name__}")
    if not value:
        raise ValueError("Rendered prompt has no token IDs")
    return tuple(int(item) for item in value)


def render_prompt_item(
    source: SourcePrompt,
    *,
    sample_id: str,
    split: str,
    side: str,
    variant: str,
    family: str,
    tokenizer,
) -> PromptItem:
    variant_tools = mutate_tools(source, variant)
    variant_schema_hash = sha256_json(variant_tools)
    variant_name = source_tool_name(variant_tools, context=f"{source.relpath}/{variant}")
    v0_template_roundtrip: bool | None = None
    if family == "mistral_native":
        if source.messages is None:
            raise AssertionError("Native Mistral source lacks messages")
        if variant == "V0":
            # Required invariant: consume this stored native sequence, never
            # a decoded text version.  We nevertheless audit the current
            # chat template against it so edited conditions are comparable.
            input_ids = source.stored_input_ids
            rerendered = normalize_rendered_ids(
                tokenizer.apply_chat_template(
                    source.messages,
                    tools=source.tools,
                    tokenize=True,
                    add_generation_prompt=True,
                )
            )
            v0_template_roundtrip = rerendered == input_ids
            if not v0_template_roundtrip:
                raise RuntimeError(
                    f"{source.relpath}: current apply_chat_template does not reproduce stored native input_ids"
                )
            render_method = "stored_native_input_ids"
        else:
            input_ids = normalize_rendered_ids(
                tokenizer.apply_chat_template(
                    source.messages,
                    tools=variant_tools,
                    tokenize=True,
                    add_generation_prompt=True,
                )
            )
            render_method = "apply_chat_template(tokenize=True,tools=mutated_tools)"
        rendered_hash = sha256_input_ids(input_ids)
    else:
        if source.text is None:
            raise AssertionError("Native text source lacks text")
        if variant == "V0":
            rendered_text = source.text
            render_method = "source_native_text"
        else:
            start, end, payload = extract_text_tools_payload(source.text, family)
            if variant == "V1":
                rendered_payload = replace_first_function_name(payload, variant_name)
                # V1's raw source differs only at the first schema name.
                parsed = source_tools_from_text(
                    source.text[:start] + rendered_payload + source.text[end:],
                    family,
                    context=f"{source.relpath}/{variant}",
                )
                if sha256_json(parsed) != variant_schema_hash:
                    raise AssertionError(f"{source.relpath}: V1 textual rename disagrees with structured mutation")
            else:
                rendered_payload = serialize_text_tools(variant_tools, original_payload=payload, family=family)
            rendered_text = source.text[:start] + rendered_payload + source.text[end:]
            render_method = "native_text_tools_payload_replacement"
        input_ids = tuple(int(token) for token in tokenizer.encode(rendered_text, add_special_tokens=False))
        if not input_ids:
            raise ValueError(f"{source.relpath}/{variant}: tokenizer produced no input IDs")
        rendered_hash = sha256_text(rendered_text)
    return PromptItem(
        sample_id=sample_id,
        split=split,
        side=side,
        variant=variant,
        input_ids=input_ids,
        input_ids_sha256=sha256_input_ids(input_ids),
        rendered_prompt_sha256=rendered_hash,
        source_prompt_sha256=source.prompt_sha256,
        source_schema_sha256=source.source_schema_sha256,
        variant_schema_sha256=variant_schema_hash,
        render_method=render_method,
        relpath=source.relpath,
        source_tool_name=source.source_tool_name,
        variant_tool_name=variant_name,
        v0_template_roundtrip=v0_template_roundtrip,
    )


def record_schema_catalog(catalog: dict[str, dict[str, Any]], source: SourcePrompt) -> None:
    entry = catalog.setdefault(
        source.source_schema_sha256,
        {
            "source_schema_sha256": source.source_schema_sha256,
            "source_tool_name": source.source_tool_name,
            "source_tools": source.tools,
            "source_prompt_count": 0,
            "variants": {},
        },
    )
    entry["source_prompt_count"] += 1
    for variant in VARIANT_ORDER:
        tools = mutate_tools(source, variant)
        entry["variants"][variant] = {
            "schema_sha256": sha256_json(tools),
            "tool_name": source_tool_name(tools, context=f"{source.relpath}/{variant}"),
            "tools": tools,
        }


def build_variant_items(
    pairs: Sequence[PairRecord],
    *,
    variant: str,
    family: str,
    tokenizer,
    catalog: dict[str, dict[str, Any]],
) -> tuple[dict[str, list[PromptItem]], list[dict[str, Any]]]:
    result = {"clean": [], "corrupt": []}
    provenance: list[dict[str, Any]] = []
    for pair in pairs:
        for side in ("clean", "corrupt"):
            source = pair.clean if side == "clean" else pair.corrupt
            # One catalog observation per source prompt, rather than one per
            # rendered variant.  The catalog itself contains all variants.
            if variant == "V0":
                record_schema_catalog(catalog, source)
            item = render_prompt_item(
                source,
                sample_id=pair.sample_id,
                split=pair.split,
                side=side,
                variant=variant,
                family=family,
                tokenizer=tokenizer,
            )
            result[side].append(item)
            provenance.append(
                {
                    "sample_id": item.sample_id,
                    "split": item.split,
                    "side": item.side,
                    "variant": item.variant,
                    "source_relpath": item.relpath,
                    "source_file_sha256": source.file_sha256,
                    "source_prompt_sha256": item.source_prompt_sha256,
                    "rendered_prompt_sha256": item.rendered_prompt_sha256,
                    "input_ids_sha256": item.input_ids_sha256,
                    "input_token_count": len(item.input_ids),
                    "source_schema_sha256": item.source_schema_sha256,
                    "variant_schema_sha256": item.variant_schema_sha256,
                    "source_tool_name": item.source_tool_name,
                    "variant_tool_name": item.variant_tool_name,
                    "render_method": item.render_method,
                    "v0_template_roundtrip": item.v0_template_roundtrip,
                }
            )
    return result, provenance


def dtype_from_name(name: str) -> torch.dtype:
    return getattr(torch, name)


def load_tokenizer_and_model(spec: dict[str, Any], args: argparse.Namespace):
    model_path = (args.model_path or Path(str(spec["model_path"]))).resolve()
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        raise RuntimeError("Tokenizer lacks a pad token and an EOS token")
    tokenizer.padding_side = "left"
    kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "device_map": {"": 0},
        "low_cpu_mem_usage": True,
    }
    if args.attn_implementation:
        kwargs["attn_implementation"] = args.attn_implementation
    model_class = Mistral3ForConditionalGeneration if spec["loader"] == "mistral" else AutoModelForCausalLM
    dtype = dtype_from_name(args.dtype)
    try:
        model = model_class.from_pretrained(str(model_path), dtype=dtype, **kwargs)
    except TypeError:
        model = model_class.from_pretrained(str(model_path), torch_dtype=dtype, **kwargs)
    model.eval()
    return tokenizer, model, model_path


def tool_token_id(tokenizer, *, marker: str, family: str) -> tuple[int, dict[str, Any]]:
    encoded = [int(token) for token in tokenizer.encode(marker, add_special_tokens=False)]
    if family == "mistral_native":
        token_id = int(tokenizer.convert_tokens_to_ids(marker))
        if token_id < 0 or token_id == int(getattr(tokenizer, "unk_token_id", -9999)):
            raise RuntimeError(f"Could not resolve native special token {marker!r}")
        return token_id, {
            "tool_call_marker": marker,
            "tool_call_token_id": token_id,
            "resolution": "convert_tokens_to_ids(native special token)",
            "encode_ids": encoded,
        }
    if len(encoded) != 1:
        raise RuntimeError(f"Expected {marker!r} to encode to one token, got {encoded}")
    return encoded[0], {
        "tool_call_marker": marker,
        "tool_call_token_id": encoded[0],
        "resolution": "tokenizer.encode",
        "encode_ids": encoded,
    }


def resolve_attr_chain(obj: Any, path: str) -> Any:
    for part in path.split("."):
        obj = getattr(obj, part)
    return obj


def get_layers(model) -> list[torch.nn.Module]:
    for path in (
        "model.language_model",
        "language_model",
        "model",
        "base_model.model",
        "base_model",
        "transformer",
    ):
        try:
            candidate = resolve_attr_chain(model, path)
        except AttributeError:
            continue
        if hasattr(candidate, "layers"):
            return list(candidate.layers)
        if hasattr(candidate, "model") and hasattr(candidate.model, "layers"):
            return list(candidate.model.layers)
    raise RuntimeError("Could not locate the decoder-layer sequence")


def model_device(model) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration as exc:
        raise RuntimeError("Loaded model has no parameters") from exc


def batch_inputs(items: Sequence[PromptItem], *, pad_token_id: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    if not items:
        raise ValueError("Cannot batch zero prompts")
    max_length = max(len(item.input_ids) for item in items)
    input_ids = torch.full((len(items), max_length), int(pad_token_id), dtype=torch.long)
    attention_mask = torch.zeros((len(items), max_length), dtype=torch.long)
    for row, item in enumerate(items):
        length = len(item.input_ids)
        input_ids[row, -length:] = torch.tensor(item.input_ids, dtype=torch.long)
        attention_mask[row, -length:] = 1
    return input_ids.to(device), attention_mask.to(device)


def chunks(values: Sequence[Any], size: int) -> Iterable[Sequence[Any]]:
    if size <= 0:
        raise ValueError("Batch size must be positive")
    for start in range(0, len(values), size):
        yield values[start : start + size]


def forward_model(model, *, input_ids: torch.Tensor, attention_mask: torch.Tensor):
    kwargs: dict[str, Any] = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "use_cache": False,
        "return_dict": True,
    }
    try:
        return model(**kwargs, logits_to_keep=1)
    except TypeError:
        return model(**kwargs)


def final_logits(outputs) -> torch.Tensor:
    logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
    if logits.ndim != 3:
        raise RuntimeError(f"Expected [batch, sequence, vocab] logits, got {tuple(logits.shape)}")
    return logits[:, -1, :].float()


def token_stats(logits: torch.Tensor, *, tool_id: int) -> dict[str, torch.Tensor]:
    target = logits[:, tool_id]
    top1 = logits.argmax(dim=-1)
    ranks = (logits > target.unsqueeze(-1)).sum(dim=-1) + 1
    probability = torch.exp(target - torch.logsumexp(logits, dim=-1))
    return {
        "tool_logit": target.detach().cpu(),
        "tool_prob": probability.detach().cpu(),
        "tool_rank": ranks.detach().cpu(),
        "top1": top1.detach().cpu(),
    }


def unwrap_hidden(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)) and output and isinstance(output[0], torch.Tensor):
        return output[0]
    raise TypeError(f"Unsupported layer hook output: {type(output).__name__}")


def replace_hidden_output(output: Any, hidden: torch.Tensor) -> Any:
    if isinstance(output, tuple):
        return (hidden, *output[1:])
    if isinstance(output, list):
        return [hidden, *output[1:]]
    return hidden


def append_metric_rows(
    destination: list[dict[str, Any]],
    items: Sequence[PromptItem],
    stats: dict[str, torch.Tensor],
    *,
    condition: str,
    layer: int,
    tool_id: int,
) -> None:
    for index, item in enumerate(items):
        top1 = int(stats["top1"][index].item())
        destination.append(
            {
                "sample_id": item.sample_id,
                "split": item.split,
                "side": item.side,
                "variant": item.variant,
                "condition": condition,
                "layer": layer,
                "tool_call_logit": float(stats["tool_logit"][index].item()),
                "tool_call_prob": float(stats["tool_prob"][index].item()),
                "tool_call_rank": int(stats["tool_rank"][index].item()),
                "tool_call_top1": bool(top1 == int(tool_id)),
                "top1_token_id": top1,
                "input_ids_sha256": item.input_ids_sha256,
                "variant_schema_sha256": item.variant_schema_sha256,
            }
        )


def evaluate_condition(
    model,
    items: Sequence[PromptItem],
    *,
    tokenizer,
    layer: torch.nn.Module | None,
    mode: str | None,
    value_cpu: torch.Tensor | None,
    tool_id: int,
    batch_size: int,
    condition: str,
    layer_id: int,
    progress_label: str,
) -> list[dict[str, Any]]:
    """Run one baseline/add/subtract/paired-replace condition."""

    if mode not in {None, "add", "subtract", "replace"}:
        raise ValueError(f"Unknown intervention mode {mode!r}")
    if mode is not None and (layer is None or value_cpu is None):
        raise ValueError("Intervention requires a layer and value")
    if mode == "replace" and value_cpu is not None and value_cpu.shape[0] != len(items):
        raise ValueError("Paired replacement states do not match prompt count")
    device = model_device(model)
    rows: list[dict[str, Any]] = []
    total = math.ceil(len(items) / batch_size)
    cursor = 0
    vector_gpu = None
    if mode in {"add", "subtract"} and value_cpu is not None:
        vector_gpu = value_cpu.to(device=device)
    for batch_index, batch in enumerate(chunks(items, batch_size), start=1):
        input_ids, attention_mask = batch_inputs(batch, pad_token_id=int(tokenizer.pad_token_id), device=device)
        handle = None
        if mode is not None:
            if mode == "replace":
                replacement_cpu = value_cpu[cursor : cursor + len(batch)]

                def hook_fn(_module, _inputs, output):
                    hidden = unwrap_hidden(output)
                    replacement = replacement_cpu.to(device=hidden.device, dtype=hidden.dtype)
                    edited = hidden.clone()
                    edited[:, -1, :] = replacement
                    return replace_hidden_output(output, edited)

            else:
                sign = 1.0 if mode == "add" else -1.0

                def hook_fn(_module, _inputs, output):
                    hidden = unwrap_hidden(output)
                    delta = vector_gpu.to(device=hidden.device, dtype=hidden.dtype)
                    edited = hidden.clone()
                    edited[:, -1, :] = edited[:, -1, :] + float(sign) * delta
                    return replace_hidden_output(output, edited)

            handle = layer.register_forward_hook(hook_fn)
        try:
            with torch.inference_mode():
                outputs = forward_model(model, input_ids=input_ids, attention_mask=attention_mask)
                stats = token_stats(final_logits(outputs), tool_id=tool_id)
        finally:
            if handle is not None:
                handle.remove()
        append_metric_rows(rows, batch, stats, condition=condition, layer=layer_id, tool_id=tool_id)
        cursor += len(batch)
        del input_ids, attention_mask, outputs, stats
        if batch_index == total or batch_index % max(total // 8, 1) == 0:
            print(f"{progress_label}: {batch_index}/{total} batches", flush=True)
    clear_cuda()
    return rows


def capture_layer_states(
    model,
    items: Sequence[PromptItem],
    *,
    tokenizer,
    layers: Sequence[torch.nn.Module],
    layer_ids: Sequence[int],
    batch_size: int,
    progress_label: str,
) -> dict[int, torch.Tensor]:
    """Capture several candidate block outputs in one no-grad forward sweep."""

    if not layer_ids:
        raise ValueError("No layers requested")
    device = model_device(model)
    captures: dict[int, list[torch.Tensor]] = {int(layer_id): [] for layer_id in layer_ids}
    total = math.ceil(len(items) / batch_size)
    for batch_index, batch in enumerate(chunks(items, batch_size), start=1):
        holders: dict[int, torch.Tensor] = {}
        handles = []
        for layer_id in layer_ids:
            current_layer_id = int(layer_id)

            def hook_fn(_module, _inputs, output, *, captured_layer=current_layer_id):
                holders[captured_layer] = unwrap_hidden(output)[:, -1, :].detach().cpu().float()

            handles.append(layers[current_layer_id].register_forward_hook(hook_fn))
        try:
            input_ids, attention_mask = batch_inputs(batch, pad_token_id=int(tokenizer.pad_token_id), device=device)
            with torch.inference_mode():
                outputs = forward_model(model, input_ids=input_ids, attention_mask=attention_mask)
        finally:
            for handle in handles:
                handle.remove()
        if set(holders) != set(captures):
            raise RuntimeError(f"{progress_label}: one or more layer hooks did not capture states")
        for layer_id, state in holders.items():
            captures[layer_id].append(state)
        del input_ids, attention_mask, outputs
        if batch_index == total or batch_index % max(total // 8, 1) == 0:
            print(f"{progress_label}: {batch_index}/{total} batches", flush=True)
    clear_cuda()
    return {layer_id: torch.cat(states, dim=0).contiguous() for layer_id, states in captures.items()}


def rate(rows: Sequence[dict[str, Any]], field: str) -> float:
    if not rows:
        raise ValueError("Cannot summarize zero rows")
    return sum(float(row[field]) for row in rows) / len(rows)


def summarize_rows(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("Cannot summarize zero rows")
    ranks = sorted(int(row["tool_call_rank"]) for row in rows)
    middle = len(ranks) // 2
    median = float(ranks[middle]) if len(ranks) % 2 else (ranks[middle - 1] + ranks[middle]) / 2.0
    return {
        "n": len(rows),
        "mean_tool_call_logit": rate(rows, "tool_call_logit"),
        "mean_tool_call_prob": rate(rows, "tool_call_prob"),
        "tool_call_top1_rate": rate(rows, "tool_call_top1"),
        "mean_tool_call_rank": rate(rows, "tool_call_rank"),
        "median_tool_call_rank": median,
        "tool_call_top3_rate": sum(int(row["tool_call_rank"]) <= 3 for row in rows) / len(rows),
    }


def candidate_layers(n_layers: int) -> list[int]:
    fractions = (0.45, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85)
    return sorted({min(max(int(round(fraction * n_layers)), 0), n_layers - 1) for fraction in fractions})


def select_layer(
    model,
    v0_selection: dict[str, list[PromptItem]],
    *,
    tokenizer,
    tool_id: int,
    batch_size: int,
    output_root: Path,
) -> int:
    """Select a block using only paired state replacement on 64 train pairs."""

    layers = get_layers(model)
    clean_items = v0_selection["clean"]
    corrupt_items = v0_selection["corrupt"]
    baseline = evaluate_condition(
        model,
        corrupt_items,
        tokenizer=tokenizer,
        layer=None,
        mode=None,
        value_cpu=None,
        tool_id=tool_id,
        batch_size=batch_size,
        condition="selection_baseline_corrupt",
        layer_id=-1,
        progress_label="layer selection baseline corrupt",
    )

    def measure(layer_ids: Sequence[int], phase: str) -> list[dict[str, Any]]:
        states = capture_layer_states(
            model,
            clean_items,
            tokenizer=tokenizer,
            layers=layers,
            layer_ids=layer_ids,
            batch_size=batch_size,
            progress_label=f"layer selection {phase} clean capture",
        )
        metrics: list[dict[str, Any]] = []
        for layer_id in layer_ids:
            patched = evaluate_condition(
                model,
                corrupt_items,
                tokenizer=tokenizer,
                layer=layers[int(layer_id)],
                mode="replace",
                value_cpu=states[int(layer_id)],
                tool_id=tool_id,
                batch_size=batch_size,
                condition="selection_paired_clean_state_replace",
                layer_id=int(layer_id),
                progress_label=f"layer selection {phase} L{layer_id} patch",
            )
            strict = sum(
                int(not bool(base["tool_call_top1"]) and bool(edited["tool_call_top1"]))
                for base, edited in zip(baseline, patched, strict=True)
            )
            metrics.append(
                {
                    "phase": phase,
                    "layer": int(layer_id),
                    "n_train_pairs": len(patched),
                    "baseline_corrupt_top1_rate": rate(baseline, "tool_call_top1"),
                    "paired_replace_top1_rate": rate(patched, "tool_call_top1"),
                    "strict_flip_count": strict,
                    "strict_flip_rate": strict / len(patched),
                    "mean_patched_tool_logit": rate(patched, "tool_call_logit"),
                    "mean_patched_tool_probability": rate(patched, "tool_call_prob"),
                }
            )
            del patched
            clear_cuda()
        del states
        clear_cuda()
        return metrics

    coarse_ids = candidate_layers(len(layers))
    coarse = measure(coarse_ids, "coarse_train")
    best_coarse = max(
        coarse,
        key=lambda row: (float(row["strict_flip_rate"]), float(row["mean_patched_tool_logit"]), -int(row["layer"])),
    )
    center = int(best_coarse["layer"])
    fine_ids = list(range(max(0, center - 2), min(len(layers), center + 3)))
    fine = measure(fine_ids, "fine_train")
    best = max(
        fine,
        key=lambda row: (float(row["strict_flip_rate"]), float(row["mean_patched_tool_logit"]), -int(row["layer"])),
    )
    all_rows = coarse + fine
    write_csv(output_root / "layer_selection_patch_sweep.csv", all_rows)
    write_json(
        output_root / "layer_selection.json",
        {
            "status": "complete",
            "selection_split": "first 64 manifest-order train pairs only",
            "position": "last non-padding prompt token; decoder-block output",
            "selection_rule": "max strict paired-state replacement flip; tie mean patched tool logit; tie earlier layer",
            "coarse_layers": coarse_ids,
            "fine_layers": fine_ids,
            "selected_layer": int(best["layer"]),
            "selected_metrics": best,
        },
    )
    return int(best["layer"])


def fit_vector(
    model,
    items: dict[str, list[PromptItem]],
    *,
    tokenizer,
    layer: torch.nn.Module,
    layer_id: int,
    batch_size: int,
    variant: str,
    output_root: Path,
) -> tuple[torch.Tensor, dict[str, Any]]:
    clean = capture_layer_states(
        model,
        items["clean"],
        tokenizer=tokenizer,
        layers=[layer],
        layer_ids=[0],
        batch_size=batch_size,
        progress_label=f"{variant} vector clean capture",
    )[0]
    corrupt = capture_layer_states(
        model,
        items["corrupt"],
        tokenizer=tokenizer,
        layers=[layer],
        layer_ids=[0],
        batch_size=batch_size,
        progress_label=f"{variant} vector corrupt capture",
    )[0]
    differences = clean - corrupt
    vector = differences.mean(dim=0).contiguous().float()
    if vector.ndim != 1 or not bool(torch.isfinite(vector).all()) or float(vector.norm().item()) <= 0.0:
        raise ValueError(f"{variant}: invalid mean clean-minus-corrupt vector")
    summary = {
        "variant": variant,
        "layer": layer_id,
        "n_vector_train_pairs": len(items["clean"]),
        "hidden_size": int(vector.numel()),
        "l2_norm": float(vector.norm().item()),
        "rms": float(vector.square().mean().sqrt().item()),
        "mean_pair_delta_l2_norm": float(differences.norm(dim=1).mean().item()),
        "vector_train_sample_ids": [item.sample_id for item in items["clean"]],
    }
    bundle_path = output_root / "vectors" / f"{variant}_mean_clean_minus_corrupt_L{layer_id}.pt"
    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"mean_diff": vector, **summary}, bundle_path)
    summary["bundle"] = str(bundle_path)
    del clean, corrupt, differences
    clear_cuda()
    return vector, summary


def cosine_similarity(left: torch.Tensor, right: torch.Tensor) -> float:
    left_unit = left.float() / left.float().norm().clamp_min(1e-12)
    right_unit = right.float() / right.float().norm().clamp_min(1e-12)
    return float(torch.dot(left_unit, right_unit).clamp(-1.0, 1.0).item())


def evaluate_variant(
    model,
    items: dict[str, list[PromptItem]],
    *,
    tokenizer,
    layer: torch.nn.Module,
    layer_id: int,
    frozen_v0: torch.Tensor,
    tool_id: int,
    batch_size: int,
    variant: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    clean = evaluate_condition(
        model,
        items["clean"],
        tokenizer=tokenizer,
        layer=None,
        mode=None,
        value_cpu=None,
        tool_id=tool_id,
        batch_size=batch_size,
        condition="baseline_clean",
        layer_id=layer_id,
        progress_label=f"{variant} heldout clean baseline",
    )
    corrupt = evaluate_condition(
        model,
        items["corrupt"],
        tokenizer=tokenizer,
        layer=None,
        mode=None,
        value_cpu=None,
        tool_id=tool_id,
        batch_size=batch_size,
        condition="baseline_corrupt",
        layer_id=layer_id,
        progress_label=f"{variant} heldout corrupt baseline",
    )
    added = evaluate_condition(
        model,
        items["corrupt"],
        tokenizer=tokenizer,
        layer=layer,
        mode="add",
        value_cpu=frozen_v0,
        tool_id=tool_id,
        batch_size=batch_size,
        condition="frozen_v0_add_to_corrupt",
        layer_id=layer_id,
        progress_label=f"{variant} heldout frozen V0 add",
    )
    removed = evaluate_condition(
        model,
        items["clean"],
        tokenizer=tokenizer,
        layer=layer,
        mode="subtract",
        value_cpu=frozen_v0,
        tool_id=tool_id,
        batch_size=batch_size,
        condition="frozen_v0_subtract_from_clean",
        layer_id=layer_id,
        progress_label=f"{variant} heldout frozen V0 subtract",
    )
    corrupt_non_tool = [not bool(row["tool_call_top1"]) for row in corrupt]
    clean_tool = [bool(row["tool_call_top1"]) for row in clean]
    flip_count = sum(
        int(eligible and bool(intervened["tool_call_top1"]))
        for eligible, intervened in zip(corrupt_non_tool, added, strict=True)
    )
    drop_count = sum(
        int(eligible and not bool(intervened["tool_call_top1"]))
        for eligible, intervened in zip(clean_tool, removed, strict=True)
    )
    flip_denominator = sum(corrupt_non_tool)
    drop_denominator = sum(clean_tool)
    clean_summary = summarize_rows(clean)
    corrupt_summary = summarize_rows(corrupt)
    added_summary = summarize_rows(added)
    removed_summary = summarize_rows(removed)
    gap = float(clean_summary["mean_tool_call_logit"] - corrupt_summary["mean_tool_call_logit"])
    summary = {
        "variant": variant,
        "n_heldout_pairs": len(clean),
        "baseline_clean": clean_summary,
        "baseline_corrupt": corrupt_summary,
        "frozen_v0_add_to_corrupt": added_summary,
        "frozen_v0_subtract_from_clean": removed_summary,
        "strict_flip_count": flip_count,
        "strict_flip_denominator": flip_denominator,
        "strict_flip_rate": flip_count / flip_denominator if flip_denominator else None,
        "strict_drop_count": drop_count,
        "strict_drop_denominator": drop_denominator,
        "strict_drop_rate": drop_count / drop_denominator if drop_denominator else None,
        "baseline_clean_minus_corrupt_logit_gap": gap,
        "normalized_sufficiency": (
            (float(added_summary["mean_tool_call_logit"]) - float(corrupt_summary["mean_tool_call_logit"])) / gap
            if gap > 1e-12
            else None
        ),
        "normalized_necessity": (
            (float(clean_summary["mean_tool_call_logit"]) - float(removed_summary["mean_tool_call_logit"])) / gap
            if gap > 1e-12
            else None
        ),
    }
    return summary, clean + corrupt + added + removed


def verify_v0_baseline(rows: Sequence[dict[str, Any]], pairs: Sequence[PairRecord], *, side: str) -> dict[str, Any]:
    expected_field = f"{side}_is_tool_top1"
    margin_field = f"{side}_margin_vs_best_non_tool"
    actual = {str(row["sample_id"]): bool(row["tool_call_top1"]) for row in rows}
    expected = {pair.sample_id: bool(pair.manifest[expected_field]) for pair in pairs}
    if set(actual) != set(expected):
        raise AssertionError(f"V0/{side}: evaluated sample IDs differ from held-out manifest")
    pair_by_id = {pair.sample_id: pair for pair in pairs}
    mismatches: list[dict[str, Any]] = []
    for sample_id in expected:
        if actual[sample_id] == expected[sample_id]:
            continue
        source = pair_by_id[sample_id]
        manifest_margin = float(source.manifest[margin_field])
        observed = next(row for row in rows if str(row["sample_id"]) == sample_id)
        detail = {
            "sample_id": sample_id,
            "manifest_tool_top1": expected[sample_id],
            "rerun_tool_top1": actual[sample_id],
            "manifest_margin_vs_best_non_tool": manifest_margin,
            "rerun_tool_logit": float(observed["tool_call_logit"]),
            "rerun_tool_rank": int(observed["tool_call_rank"]),
            "rerun_top1_token_id": int(observed["top1_token_id"]),
        }
        mismatches.append(detail)
    audit = {
        "side": side,
        "n": len(expected),
        "manifest_tool_top1_count": sum(expected.values()),
        "rerun_tool_top1_count": sum(actual.values()),
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
        "status": "exact_match" if not mismatches else "failed",
    }
    if mismatches:
        raise RuntimeError(
            f"V0/{side}: target-model baseline disagrees with the v5 release for {len(mismatches)} items"
        )
    return audit


def partitions(
    records: Sequence[PairRecord], *, selection_pairs: int, vector_pairs: int, heldout_pairs: int
) -> tuple[list[PairRecord], list[PairRecord], list[PairRecord]]:
    train = [record for record in records if record.split == "train"]
    heldout = [record for record in records if record.split == "heldout"]
    if selection_pairs <= 0 or vector_pairs <= 0 or heldout_pairs <= 0:
        raise ValueError("All partition cardinalities must be positive")
    if selection_pairs + vector_pairs > len(train):
        raise ValueError("Layer-selection and vector-fit partitions exceed 200 v5 train pairs")
    if heldout_pairs > len(heldout):
        raise ValueError("Requested heldout pairs exceed the 300-item v5 heldout split")
    selection = train[:selection_pairs]
    vector = train[selection_pairs : selection_pairs + vector_pairs]
    test = heldout[:heldout_pairs]
    ids = [record.sample_id for group in (selection, vector, test) for record in group]
    if len(ids) != len(set(ids)):
        raise AssertionError("Selection, vector, and heldout partitions overlap")
    return selection, vector, test


def dataset_provenance(dataset_root: Path, summary: dict[str, Any], records: Sequence[PairRecord]) -> dict[str, Any]:
    release_files = {
        "repository_readme": PROJECT_ROOT / "README.md",
        "v5_release_readme": dataset_root.parent / "README.md",
        "model_readme": dataset_root / "README.md",
        "summary": dataset_root / "summary.json",
        "manifest": dataset_root / "manifest.jsonl",
    }
    return {
        "dataset_version": summary["dataset_version"],
        "dataset_root": str(dataset_root),
        "model_key": summary["model_key"],
        "model_label": summary["model_label"],
        "model_path": summary["model_path"],
        "tool_call_marker": summary["tool_call_marker"],
        "tool_call_token_id": summary["tool_call_token_id"],
        "declared_counts": {"train": summary["n_train"], "heldout": summary["n_heldout"]},
        "manifest_counts": {
            "train": sum(record.split == "train" for record in records),
            "heldout": sum(record.split == "heldout" for record in records),
        },
        "release_files": {
            label: {"path": str(path.resolve()), "sha256": sha256_file(path)} for label, path in release_files.items()
        },
        "full_manifest_membership": {
            split: [
                {
                    "sample_id": record.sample_id,
                    "clean_relpath": record.clean.relpath,
                    "corrupt_relpath": record.corrupt.relpath,
                    "clean_prompt_sha256": record.clean.prompt_sha256,
                    "corrupt_prompt_sha256": record.corrupt.prompt_sha256,
                }
                for record in records
                if record.split == split
            ]
            for split in ("train", "heldout")
        },
    }


def main() -> None:
    args = parse_args()
    spec = MODEL_SPECS[args.model_key]
    dataset_root = (args.dataset_root or (V5_ROOT / args.model_key)).resolve()
    expected_model_path = Path(str(spec["model_path"])).resolve()
    model_path = (args.model_path or expected_model_path).resolve()
    output_root = args.output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite an existing result directory: {output_root}")
    if not dataset_root.is_dir():
        raise FileNotFoundError(dataset_root)
    if not model_path.is_dir():
        raise FileNotFoundError(model_path)
    if not args.allow_subset and (args.selection_pairs, args.vector_pairs, args.heldout_pairs) != (64, 136, 300):
        raise ValueError("Canonical v5 rerun requires --selection-pairs 64 --vector-pairs 136 --heldout-pairs 300")
    if model_path != expected_model_path:
        raise ValueError(f"{args.model_key}: v5 requires model path {expected_model_path}, received {model_path}")

    summary, records = load_pairs(dataset_root, model_key=args.model_key, family=str(spec["family"]))
    if str(summary.get("model_label")) != str(spec["label"]):
        raise ValueError(f"{args.model_key}: v5 summary label does not match runner model specification")
    if Path(str(summary.get("model_path"))).resolve() != model_path:
        raise ValueError(f"{args.model_key}: v5 summary model path does not match requested model")
    selection, vector_train, heldout = partitions(
        records,
        selection_pairs=args.selection_pairs,
        vector_pairs=args.vector_pairs,
        heldout_pairs=args.heldout_pairs,
    )
    batch_size = int(args.batch_size or spec["batch_size"])
    output_root.mkdir(parents=True, exist_ok=False)
    write_json(
        output_root / "run_config.json",
        {
            "experiment": "reviewer_cjrq_v5_tool_identity",
            "created_at": now_utc(),
            "model_key": args.model_key,
            "model_label": spec["label"],
            "model_path": str(model_path),
            "dataset_root": str(dataset_root),
            "dataset_version": "v5_model_specific_balanced",
            "family": spec["family"],
            "variants": list(VARIANT_ORDER),
            "table_variants": list(TABLE_VARIANTS),
            "metrics": "cosine_to_V0 / strict_flip / strict_drop",
            "selection_pairs": len(selection),
            "vector_pairs": len(vector_train),
            "heldout_pairs": len(heldout),
            "partition_rule": "manifest order: first train prefix for layer selection, following disjoint train prefix for vectors, all heldout rows",
            "selection_protocol": "paired clean-state replacement, V0 only, decoder-block output at final non-padding prompt token",
            "vector_protocol": "mean(clean state - corrupt state), variant-specific fit on disjoint train pairs at the selected V0 layer",
            "causal_protocol": "frozen V0 vector added to baseline-corrupt and subtracted from baseline-clean at the selected block output / final prompt token",
            "strict_denominators": "baseline-corrupt non-tool prompts for flip; baseline-clean tool prompts for drop, measured separately for every schema variant",
            "mistral_rule": "V0 stored native input_ids; V1/V2/V5 apply_chat_template(tokenize=True, tools=mutated_tools); no decoded-text re-encode",
            "batch_size": batch_size,
            "dtype": args.dtype,
            "attn_implementation": args.attn_implementation or None,
            "seed": args.seed,
            "allow_subset": bool(args.allow_subset),
            "runner_path": str(THIS_FILE),
            "runner_sha256": sha256_file(THIS_FILE),
            "python": sys.version,
            "platform": platform.platform(),
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "gpu_preflight": gpu_snapshot(),
            "git": git_provenance(),
        },
    )
    write_json(output_root / "dataset_provenance.json", dataset_provenance(dataset_root, summary, records))
    write_json(
        output_root / "membership.json",
        {
            "layer_selection_train_sample_ids": [record.sample_id for record in selection],
            "vector_fit_train_sample_ids": [record.sample_id for record in vector_train],
            "heldout_sample_ids": [record.sample_id for record in heldout],
            "disjoint": True,
        },
    )
    write_json(
        output_root / "variant_definitions.json",
        {
            "V0": {
                "label": "Original",
                "definition": "Native release schema preserved without alteration.",
            },
            "V1": {
                "label": "Renamed",
                "definition": "Only function.name changes; the original description and parameter schema are preserved byte-for-byte in text renderings and structurally unchanged in Mistral tools.",
                "rename_map": RENAMED_TOOL_NAMES,
            },
            "V2": {
                "label": "Removed",
                "definition": "Function name f1, empty function description, and recursively neutralized argument names/descriptions while retaining parameter type/property/required structure.",
            },
            "V5": {
                "label": "Mismatched",
                "definition": "A get_weather function with one required location string parameter.",
                "function": WEATHER_FUNCTION,
            },
        },
    )

    tokenizer = None
    model = None
    try:
        torch.manual_seed(int(args.seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(args.seed))
            torch.cuda.reset_peak_memory_stats()
        print(json.dumps({"event": "load_model", "model": spec["label"], "batch_size": batch_size}), flush=True)
        tokenizer, model, loaded_model_path = load_tokenizer_and_model(spec, args)
        if loaded_model_path != model_path:
            raise AssertionError("Loaded model path differs from requested model path")
        observed_tool_id, token_info = tool_token_id(
            tokenizer,
            marker=str(summary["tool_call_marker"]),
            family=str(spec["family"]),
        )
        if int(observed_tool_id) != int(summary["tool_call_token_id"]):
            raise RuntimeError(
                f"{args.model_key}: tokenizer tool marker ID {observed_tool_id} differs from v5 summary {summary['tool_call_token_id']}"
            )
        write_json(output_root / "tool_token.json", token_info)

        # Render every release prompt once under every condition before model
        # forwards.  This records the exact model input IDs and catches a
        # native-template mismatch before a costly causal run.
        catalog: dict[str, dict[str, Any]] = {}
        input_provenance: list[dict[str, Any]] = []
        item_sets: dict[str, dict[str, dict[str, list[PromptItem]]]] = {}
        for variant in VARIANT_ORDER:
            item_sets[variant] = {}
            for partition_name, partition_pairs in (
                ("layer_selection_train", selection),
                ("vector_fit_train", vector_train),
                ("heldout", heldout),
            ):
                rendered, provenance = build_variant_items(
                    partition_pairs,
                    variant=variant,
                    family=str(spec["family"]),
                    tokenizer=tokenizer,
                    catalog=catalog,
                )
                item_sets[variant][partition_name] = rendered
                for row in provenance:
                    row["partition"] = partition_name
                input_provenance.extend(provenance)
        write_jsonl(output_root / "input_provenance.jsonl", input_provenance)
        write_json(output_root / "schema_catalog.json", {key: catalog[key] for key in sorted(catalog)})
        mistral_roundtrip_rows = [
            row
            for row in input_provenance
            if row["variant"] == "V0" and row["v0_template_roundtrip"] is not None
        ]
        write_json(
            output_root / "prompt_render_audit.json",
            {
                "n_input_records": len(input_provenance),
                "expected_input_records": 2 * (len(selection) + len(vector_train) + len(heldout)) * len(VARIANT_ORDER),
                "source_schema_count": len(catalog),
                "mistral_v0_stored_id_roundtrip": {
                    "n": len(mistral_roundtrip_rows),
                    "exact_count": sum(bool(row["v0_template_roundtrip"]) for row in mistral_roundtrip_rows),
                },
                "v0_input_rule": "stored native IDs for Mistral; source text re-tokenized with add_special_tokens=False for native text families",
            },
        )

        layers = get_layers(model)
        layer_id = select_layer(
            model,
            item_sets["V0"]["layer_selection_train"],
            tokenizer=tokenizer,
            tool_id=observed_tool_id,
            batch_size=batch_size,
            output_root=output_root,
        )
        selected_layer = layers[layer_id]
        vectors: dict[str, torch.Tensor] = {}
        vector_summaries: dict[str, dict[str, Any]] = {}
        for variant in VARIANT_ORDER:
            vector_value, vector_summary = fit_vector(
                model,
                item_sets[variant]["vector_fit_train"],
                tokenizer=tokenizer,
                layer=selected_layer,
                layer_id=layer_id,
                batch_size=batch_size,
                variant=variant,
                output_root=output_root,
            )
            vectors[variant] = vector_value
            vector_summaries[variant] = vector_summary
        frozen_v0 = vectors["V0"].detach().cpu().float().contiguous()
        for variant, vector_value in vectors.items():
            vector_summaries[variant]["cosine_to_frozen_v0"] = cosine_similarity(vector_value, frozen_v0)
            vector_summaries[variant]["l2_norm_over_v0"] = float(vector_value.norm().item() / frozen_v0.norm().item())
        write_json(output_root / "vector_summaries.json", vector_summaries)

        all_sample_rows: list[dict[str, Any]] = []
        results: dict[str, dict[str, Any]] = {}
        baseline_audit: dict[str, Any] = {}
        summary_rows: list[dict[str, Any]] = []
        for variant in VARIANT_ORDER:
            result, sample_rows = evaluate_variant(
                model,
                item_sets[variant]["heldout"],
                tokenizer=tokenizer,
                layer=selected_layer,
                layer_id=layer_id,
                frozen_v0=frozen_v0,
                tool_id=observed_tool_id,
                batch_size=batch_size,
                variant=variant,
            )
            result["cosine_to_frozen_v0"] = vector_summaries[variant]["cosine_to_frozen_v0"]
            results[variant] = result
            all_sample_rows.extend(sample_rows)
            if variant == "V0":
                clean_rows = [row for row in sample_rows if row["condition"] == "baseline_clean"]
                corrupt_rows = [row for row in sample_rows if row["condition"] == "baseline_corrupt"]
                baseline_audit["clean"] = verify_v0_baseline(clean_rows, heldout, side="clean")
                baseline_audit["corrupt"] = verify_v0_baseline(corrupt_rows, heldout, side="corrupt")
            summary_rows.append(
                {
                    "variant": variant,
                    "cosine_to_frozen_v0": result["cosine_to_frozen_v0"],
                    "strict_flip_count": result["strict_flip_count"],
                    "strict_flip_denominator": result["strict_flip_denominator"],
                    "strict_flip_rate": result["strict_flip_rate"],
                    "strict_drop_count": result["strict_drop_count"],
                    "strict_drop_denominator": result["strict_drop_denominator"],
                    "strict_drop_rate": result["strict_drop_rate"],
                    "baseline_clean_top1_rate": result["baseline_clean"]["tool_call_top1_rate"],
                    "baseline_corrupt_top1_rate": result["baseline_corrupt"]["tool_call_top1_rate"],
                    "normalized_sufficiency": result["normalized_sufficiency"],
                    "normalized_necessity": result["normalized_necessity"],
                }
            )
            # Durable progress artifacts are helpful if a later model fails;
            # completion.json is the only completion signal used by the table
            # renderer.
            write_csv(output_root / "sample_metrics.partial.csv", all_sample_rows)
            write_csv(output_root / "intervention_long.partial.csv", summary_rows)

        write_json(output_root / "baseline_screening_replay.json", baseline_audit)
        write_json(output_root / "tool_identity_results.json", results)
        write_csv(output_root / "sample_metrics.csv", all_sample_rows)
        write_csv(output_root / "intervention_long.csv", summary_rows)
        table_metrics = {
            variant: {
                "label": {"V1": "Renamed", "V2": "Removed", "V5": "Mismatched"}.get(variant, "Original"),
                "cosine": results[variant]["cosine_to_frozen_v0"],
                "strict_flip": results[variant]["strict_flip_rate"],
                "strict_drop": results[variant]["strict_drop_rate"],
                "strict_flip_count": results[variant]["strict_flip_count"],
                "strict_flip_denominator": results[variant]["strict_flip_denominator"],
                "strict_drop_count": results[variant]["strict_drop_count"],
                "strict_drop_denominator": results[variant]["strict_drop_denominator"],
            }
            for variant in TABLE_VARIANTS
        }
        write_json(output_root / "table_metrics.json", table_metrics)
        peak_memory = float(torch.cuda.max_memory_allocated() / (1024**3)) if torch.cuda.is_available() else None
        write_json(
            output_root / "completion.json",
            {
                "status": "complete",
                "completed_at": now_utc(),
                "model_key": args.model_key,
                "model_label": spec["label"],
                "dataset_version": "v5_model_specific_balanced",
                "dataset_root": str(dataset_root),
                "selected_layer": layer_id,
                "layer_position": "decoder-block output at final non-padding prompt token",
                "n_layer_selection_train": len(selection),
                "n_vector_fit_train": len(vector_train),
                "n_heldout": len(heldout),
                "tool_call_token_id": observed_tool_id,
                "tool_call_marker": summary["tool_call_marker"],
                "baseline_screening_replay": baseline_audit,
                "table_metrics": table_metrics,
                "peak_cuda_memory_gib": peak_memory,
                "gpu_postrun": gpu_snapshot(),
            },
        )
        print(
            json.dumps(
                {
                    "status": "complete",
                    "model_key": args.model_key,
                    "output_root": str(output_root),
                    "selected_layer": layer_id,
                    "table_metrics": table_metrics,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    except BaseException as exc:
        write_json(
            output_root / "failure.json",
            {"status": "failed", "failed_at": now_utc(), "exception_type": type(exc).__name__, "message": str(exc)},
        )
        raise
    finally:
        del model
        clear_cuda()


if __name__ == "__main__":
    main()
