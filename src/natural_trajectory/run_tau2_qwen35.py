#!/usr/bin/env python3
"""Coding-only τ² Telecom interventions for Qwen3.5-9B and Qwen3.5-4B.

The two Qwen3.5 checkpoints share the native ``<tool_call>`` XML protocol,
but are calibrated independently.  For each checkpoint this runner:

1. screens only the checkpoint's coding clean/corrupt pairs;
2. chooses a residual layer using a coding-only fit/evaluation split;
3. fits a fresh coding mean-difference vector at that selected layer; and
4. freezes both choices before rendering, screening, and intervening on τ²
   Telecom trajectories with Qwen3.5's own chat template.

The first generated token after the native assistant generation prompt (whose
empty ``<think>`` block is explicitly closed) is the semantic decision point.
For Qwen3.5 this is the single special token ``<tool_call>`` when a function
call is selected.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import importlib.util
import json
import math
import os
import platform
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

import torch


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

# Reuse the family-neutral accounting, batching, intervention, and result
# serialization helpers.  Its model-layer resolver is replaced below with
# Qwen3.5's more explicit decoder-stack resolver.
import run_tau2_cross_family as common  # noqa: E402


_ORIGINAL_FIND_SPEC = importlib.util.find_spec


def _patched_find_spec(name: str, package: str | None = None):
    # Some local Transformer installations discover an incompatible sklearn
    # wheel while importing optional generation utilities.  Qwen's established
    # coding runners use this same narrow workaround.
    if name == "sklearn" and os.environ.get("MECH_ENABLE_TRANSFORMERS_SKLEARN", "0") != "1":
        return None
    return _ORIGINAL_FIND_SPEC(name, package)


importlib.util.find_spec = _patched_find_spec
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
finally:
    importlib.util.find_spec = _ORIGINAL_FIND_SPEC


RAW_ROOT = PROJECT_ROOT / "datasets" / "external" / "tau2_telecom_qwen35_9b" / "raw"
PREPARED_ROOT = PROJECT_ROOT / "datasets" / "external" / "tau2_telecom_qwen35_9b" / "prepared"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "results" / "natural_trajectory" / "cross_family_tau2_20260726"
SEED = 20260726
TOOL_CALL_TEXT = "<tool_call>"
MODEL_ROOT = Path(os.environ.get("MODEL_ROOT", PROJECT_ROOT / "external" / "models")).expanduser()


def model_path(environment_variable: str, directory_name: str) -> Path:
    return Path(os.environ.get(environment_variable, MODEL_ROOT / directory_name)).expanduser()


@dataclass(frozen=True)
class ModelSpec:
    key: str
    display_name: str
    model_path: Path
    coding_root: Path


SPECS: dict[str, ModelSpec] = {
    "qwen35_9b": ModelSpec(
        key="qwen35_9b",
        display_name="Qwen3.5-9B",
        model_path=model_path("QWEN35_9B_PATH", "Qwen3.5-9B"),
        coding_root=PROJECT_ROOT / "results" / "section6_generalization" / "qwen35_9b" / "datasets",
    ),
    "qwen35_4b": ModelSpec(
        key="qwen35_4b",
        display_name="Qwen3.5-4B",
        model_path=model_path("QWEN35_4B_PATH", "Qwen3.5-4B"),
        coding_root=PROJECT_ROOT / "results" / "runs" / "qwen35_4b_v2_1500_20260725" / "converted_dataset",
    ),
}


@dataclass(frozen=True)
class TauResources:
    system_prompt: str
    tools: list[dict[str, Any]]
    tool_names: set[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=sorted(SPECS), required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--raw-root", type=Path, default=RAW_ROOT)
    parser.add_argument("--prepared-root", type=Path, default=PREPARED_ROOT)
    parser.add_argument("--coding-batch-size", type=int, default=4)
    parser.add_argument("--tau-batch-size", type=int, default=1)
    parser.add_argument("--max-context-tokens", type=int, default=32768)
    parser.add_argument("--tool-screen-limit", type=int, default=0)
    parser.add_argument("--induction-screen-limit", type=int, default=0)
    parser.add_argument("--final-count", type=int, default=50)
    parser.add_argument("--max-baseline-tool-probability", type=float, default=0.05)
    parser.add_argument("--layer-fit-count", type=int, default=64)
    parser.add_argument("--layer-eval-count", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--skip-generation", action="store_true")
    return parser.parse_args()


def read_resources(raw_root: Path) -> TauResources:
    system_prompt = (raw_root / "tau2_system_prompt.txt").read_text(encoding="utf-8").strip()
    tools = json.loads((raw_root / "tau2_tool_schemas.json").read_text(encoding="utf-8"))
    if not system_prompt or not isinstance(tools, list) or not tools:
        raise ValueError("Invalid τ² system prompt or tool-schema resource")
    names = {str(item["function"]["name"]) for item in tools}
    return TauResources(system_prompt=system_prompt, tools=tools, tool_names=names)


def resolve_attr_chain(obj: Any, path: str) -> Any:
    for part in path.split("."):
        obj = getattr(obj, part)
    return obj


def resolve_qwen_layers(model: Any) -> Any:
    for path in (
        "model.language_model",
        "model",
        "base_model.model",
        "base_model",
        "language_model",
        "transformer",
    ):
        try:
            candidate = resolve_attr_chain(model, path)
        except AttributeError:
            continue
        if hasattr(candidate, "layers"):
            return candidate.layers
        if hasattr(candidate, "model") and hasattr(candidate.model, "layers"):
            return candidate.model.layers
    raise RuntimeError("Could not locate Qwen3.5 decoder layers")


# ``evaluate_condition`` and the hook helpers do global lookup inside the
# imported module, so this safely specializes all reused primitives.
common.resolve_layers = resolve_qwen_layers


def get_tool_token_id(tokenizer: Any) -> tuple[int, dict[str, Any]]:
    ids = [int(value) for value in tokenizer.encode(TOOL_CALL_TEXT, add_special_tokens=False)]
    converted = int(tokenizer.convert_tokens_to_ids(TOOL_CALL_TEXT))
    if len(ids) != 1 or ids[0] != converted:
        raise RuntimeError(f"Qwen3.5 {TOOL_CALL_TEXT!r} must be one special token, got {ids}, {converted}")
    return converted, {
        "tool_token_text": TOOL_CALL_TEXT,
        "tool_token_id": converted,
        "tool_token_ids_via_encode": ids,
        "semantic_position": "first token after native add_generation_prompt assistant prefix with enable_thinking=False",
    }


def load_model_and_tokenizer(spec: ModelSpec) -> tuple[Any, Any, torch.device]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for τ² intervention")
    model = AutoModelForCausalLM.from_pretrained(
        str(spec.model_path),
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(str(spec.model_path), trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return model, tokenizer, torch.device("cuda:0")


def ids_from_text(tokenizer: Any, text: str) -> list[int]:
    values = tokenizer(text, add_special_tokens=False)["input_ids"]
    if isinstance(values, torch.Tensor):
        values = values.detach().cpu().tolist()
    if values and isinstance(values[0], list):
        values = values[0]
    return [int(value) for value in values]


def load_coding_pairs(spec: ModelSpec, tokenizer: Any) -> list[common.CodingPair]:
    manifest = spec.coding_root / "manifest.jsonl"
    rows = common.read_jsonl(manifest)
    pairs: list[common.CodingPair] = []
    for row in rows:
        if spec.key == "qwen35_9b":
            pair_id = str(row["pair_id"])
            clean_path = spec.coding_root / str(row["clean_filename"])
            corrupt_path = spec.coding_root / str(row["corrupt_filename"])
        else:
            pair_id = str(row["sample_id"])
            clean_path = Path(str(row["clean_prompt_path"]))
            corrupt_path = Path(str(row["corrupt_prompt_path"]))
        if not clean_path.exists() or not corrupt_path.exists():
            raise FileNotFoundError(f"Missing coding pair files for {pair_id}: {clean_path}, {corrupt_path}")
        pairs.append(
            common.CodingPair(
                pair_id=pair_id,
                split=str(row.get("split", "train")),
                clean_ids=ids_from_text(tokenizer, clean_path.read_text(encoding="utf-8")),
                corrupt_ids=ids_from_text(tokenizer, corrupt_path.read_text(encoding="utf-8")),
            )
        )
    if not pairs:
        raise RuntimeError(f"No coding pairs in {manifest}")
    return pairs


def coding_baseline_screen(
    pairs: Sequence[common.CodingPair],
    *,
    model: Any,
    tokenizer: Any,
    device: torch.device,
    tool_token_id: int,
    batch_size: int,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], list[dict[str, Any]]]:
    baseline = common.Condition("baseline_no_hook_alpha_0", "baseline", 0.0, None, "no intervention")
    # L0 is only a bookkeeping argument here: a baseline condition registers no hook.
    clean_rows, _ = common.evaluate_condition(
        common.coding_prepared(pairs, "clean"),
        model=model,
        tokenizer=tokenizer,
        device=device,
        tool_token_id=tool_token_id,
        layer=0,
        condition=baseline,
        batch_size=batch_size,
    )
    corrupt_rows, _ = common.evaluate_condition(
        common.coding_prepared(pairs, "corrupt"),
        model=model,
        tokenizer=tokenizer,
        device=device,
        tool_token_id=tool_token_id,
        layer=0,
        condition=baseline,
        batch_size=batch_size,
    )
    clean_by_pair = {str(row["coding_pair_id"]): row for row in clean_rows}
    corrupt_by_pair = {str(row["coding_pair_id"]): row for row in corrupt_rows}
    if len(clean_by_pair) != len(pairs) or len(corrupt_by_pair) != len(pairs):
        raise RuntimeError("Duplicate or missing coding baseline rows")
    compact: list[dict[str, Any]] = []
    for pair in pairs:
        clean = clean_by_pair[pair.pair_id]
        corrupt = corrupt_by_pair[pair.pair_id]
        compact.append(
            {
                "coding_pair_id": pair.pair_id,
                "split": pair.split,
                "clean_is_tool_call_top1": bool(clean["is_tool_call_top1"]),
                "clean_tool_call_probability": float(clean["tool_call_probability"]),
                "clean_top1_token_text": str(clean["top1_token_text"]),
                "corrupt_is_tool_call_top1": bool(corrupt["is_tool_call_top1"]),
                "corrupt_tool_call_probability": float(corrupt["tool_call_probability"]),
                "corrupt_top1_token_text": str(corrupt["top1_token_text"]),
                "eligible_clean_tool_vs_corrupt_non_tool": bool(clean["is_tool_call_top1"])
                and not bool(corrupt["is_tool_call_top1"]),
            }
        )
    return clean_by_pair, corrupt_by_pair, compact


def stable_pairs(pairs: Sequence[common.CodingPair], seed: int, label: str) -> list[common.CodingPair]:
    return sorted(pairs, key=lambda pair: common.stable_rank(seed, label, pair.pair_id))


@contextlib.contextmanager
def capture_all_layers(model: Any) -> Iterator[list[torch.Tensor | None]]:
    layers = list(resolve_qwen_layers(model))
    captured: list[torch.Tensor | None] = [None] * len(layers)

    def make_hook(index: int):
        def hook(_module: Any, _inputs: Any, output: Any) -> None:
            hidden = output[0] if isinstance(output, tuple) else output
            if not isinstance(hidden, torch.Tensor) or hidden.ndim != 3:
                raise RuntimeError(f"Unexpected Qwen3.5 layer output at L{index}: {type(hidden)!r}")
            captured[index] = hidden[:, -1, :].detach().float().cpu()

        return hook

    handles = [layer.register_forward_hook(make_hook(index)) for index, layer in enumerate(layers)]
    try:
        yield captured
    finally:
        for handle in reversed(handles):
            handle.remove()


def mean_states_all_layers(
    pairs: Sequence[common.CodingPair],
    *,
    side: str,
    model: Any,
    tokenizer: Any,
    device: torch.device,
    batch_size: int,
) -> list[torch.Tensor]:
    prefixes = common.coding_prepared(pairs, side)
    layer_count = len(resolve_qwen_layers(model))
    sums: list[torch.Tensor | None] = [None] * layer_count
    count = 0
    for index, batch in enumerate(common.make_batches(prefixes, batch_size), start=1):
        inputs = common.make_left_padded_batch(batch, int(tokenizer.pad_token_id), device)
        with capture_all_layers(model) as states:
            logits = common.model_forward(model, inputs)
        del logits, inputs
        for layer_index, state in enumerate(states):
            if state is None:
                raise RuntimeError(f"Missing captured state at L{layer_index}")
            reduced = state.sum(dim=0)
            sums[layer_index] = reduced if sums[layer_index] is None else sums[layer_index] + reduced
        count += len(batch)
        torch.cuda.empty_cache()
        if index % 20 == 0 or index == math.ceil(len(prefixes) / batch_size):
            print(json.dumps({"all_layer_capture": side, "batches": index}), flush=True)
    if count != len(pairs) or any(value is None for value in sums):
        raise RuntimeError("Incomplete all-layer coding capture")
    return [(value / count).float().contiguous() for value in sums if value is not None]


def choose_layer_from_coding(
    eligible_train: Sequence[common.CodingPair],
    corrupt_baseline: dict[str, dict[str, Any]],
    *,
    model: Any,
    tokenizer: Any,
    device: torch.device,
    tool_token_id: int,
    batch_size: int,
    fit_count_requested: int,
    eval_count_requested: int,
    seed: int,
    output_root: Path,
) -> tuple[int, dict[str, Any]]:
    ordered = stable_pairs(eligible_train, seed, "qwen35-coding-layer-selection")
    if len(ordered) < 12:
        raise RuntimeError(f"Only {len(ordered)} coding behavior-positive pairs; need at least 12 for layer selection")
    max_fit = max(8, len(ordered) - 8)
    fit_count = min(max(8, int(fit_count_requested)), max_fit)
    eval_count = min(max(4, int(eval_count_requested)), len(ordered) - fit_count)
    if eval_count < 4:
        fit_count = max(8, len(ordered) - 4)
        eval_count = len(ordered) - fit_count
    fit_pairs = ordered[:fit_count]
    eval_pairs = ordered[fit_count : fit_count + eval_count]
    if not fit_pairs or not eval_pairs:
        raise RuntimeError("Could not form disjoint coding layer-selection partitions")

    clean_means = mean_states_all_layers(
        fit_pairs, side="clean", model=model, tokenizer=tokenizer, device=device, batch_size=batch_size
    )
    corrupt_means = mean_states_all_layers(
        fit_pairs, side="corrupt", model=model, tokenizer=tokenizer, device=device, batch_size=batch_size
    )
    if len(clean_means) != len(corrupt_means):
        raise RuntimeError("Layer-count mismatch while forming coding vectors")
    vectors = [(clean - corrupt).float().contiguous() for clean, corrupt in zip(clean_means, corrupt_means)]
    if any(not torch.isfinite(vector).all() or vector.norm().item() == 0 for vector in vectors):
        raise RuntimeError("Invalid candidate coding vector during layer selection")

    eval_prefixes = common.coding_prepared(eval_pairs, "corrupt")
    baseline_by_id = {
        f"coding_corrupt_{pair.pair_id}": corrupt_baseline[pair.pair_id]
        for pair in eval_pairs
    }
    rows: list[dict[str, Any]] = []
    for layer, vector in enumerate(vectors):
        condition = common.Condition(
            "plus_fit_mean_diff_alpha_1",
            "coding_layer_selection",
            1.0,
            vector,
            "add coding mean difference fit on disjoint coding subset",
        )
        intervened, hook_stats = common.evaluate_condition(
            eval_prefixes,
            model=model,
            tokenizer=tokenizer,
            device=device,
            tool_token_id=tool_token_id,
            layer=layer,
            condition=condition,
            batch_size=batch_size,
        )
        summary = common.summarize_condition(intervened, baseline_by_id)
        rows.append(
            {
                "layer": layer,
                "vector_norm": float(vector.norm().item()),
                **summary,
                "hook_stats": hook_stats,
            }
        )
        print(json.dumps({"coding_layer_selection": [layer + 1, len(vectors)], "layer": layer, "flips": summary["baseline_non_tool_to_tool_count"]}), flush=True)

    best = max(
        rows,
        key=lambda row: (
            int(row["baseline_non_tool_to_tool_count"]),
            int(row["tool_call_top1_count"]),
            float(row["mean_tool_call_logit"]),
            -int(row["layer"]),
        ),
    )
    payload = {
        "source_domain": "coding clean/corrupt pairs only",
        "selection_rule": "maximize coding held-out corrupt non-tool→tool flips, then tool top-1 count, then mean tool logit, then earliest layer",
        "fit_pair_ids": [pair.pair_id for pair in fit_pairs],
        "evaluation_pair_ids": [pair.pair_id for pair in eval_pairs],
        "fit_count": len(fit_pairs),
        "evaluation_count": len(eval_pairs),
        "candidate_layers": rows,
        "selected_layer": int(best["layer"]),
        "selected_row": best,
    }
    common.write_json(output_root / "coding_layer_selection.json", payload)
    return int(best["layer"]), payload


def adapt_qwen35_messages(row: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    source = row.get("messages")
    if not isinstance(source, list):
        raise TypeError(f"{row.get('candidate_id')} has no message list")
    messages: list[dict[str, Any]] = []
    historical_calls = 0
    for raw in source:
        role = str(raw["role"])
        converted: dict[str, Any] = {"role": role, "content": common.safe_content(raw.get("content"))}
        calls = raw.get("tool_calls") or []
        if role == "assistant" and calls:
            native_calls: list[dict[str, Any]] = []
            for call in calls:
                native_calls.append(
                    {
                        "type": "function",
                        "function": {
                            "name": common.function_name(call),
                            # Qwen3.5's template consumes a JSON object and
                            # serializes it into <parameter=...> XML itself.
                            "arguments": common.function_arguments(call),
                        },
                    }
                )
            converted["tool_calls"] = native_calls
            historical_calls += len(native_calls)
        messages.append(converted)
    return messages, {
        "adapter": "qwen35_native_tool_history",
        "historical_calls_serialized": historical_calls,
        "thinking_disabled_at_generation_prompt": True,
        "decision_token": TOOL_CALL_TEXT,
    }


def render_prefix(
    row: dict[str, Any],
    *,
    tokenizer: Any,
    resources: TauResources,
    max_context_tokens: int,
) -> tuple[common.PreparedPrefix | None, dict[str, Any] | None]:
    try:
        native_messages, audit = adapt_qwen35_messages(row)
        messages = [{"role": "system", "content": resources.system_prompt}, *native_messages]
        encoded = tokenizer.apply_chat_template(
            messages,
            tools=resources.tools,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
            return_dict=True,
            return_tensors="pt",
        )
        ids = [int(value) for value in encoded["input_ids"][0].detach().cpu().tolist()]
        if not ids:
            raise RuntimeError("Qwen3.5 chat template produced an empty prompt")
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
        return common.PreparedPrefix(candidate_id=str(row["candidate_id"]), input_ids=ids, meta=meta), None
    except Exception as exc:
        return None, {"candidate_id": row.get("candidate_id"), "reason": "render_error", "detail": repr(exc)}


def prepare_screen_pool(
    rows: Sequence[dict[str, Any]],
    *,
    tokenizer: Any,
    resources: TauResources,
    max_context_tokens: int,
) -> tuple[list[common.PreparedPrefix], list[dict[str, Any]]]:
    prepared: list[common.PreparedPrefix] = []
    rejected: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        prefix, reason = render_prefix(
            row, tokenizer=tokenizer, resources=resources, max_context_tokens=max_context_tokens
        )
        if prefix is not None:
            prepared.append(prefix)
        else:
            assert reason is not None
            rejected.append(reason)
        if index % 200 == 0 or index == len(rows):
            print(json.dumps({"rendered_tau2_prefixes": [index, len(rows)]}), flush=True)
    return prepared, rejected


def parse_well_formed_qwen_call(text: str, tool_names: set[str]) -> tuple[bool, str | None]:
    stripped = text.lstrip()
    if not stripped.startswith(TOOL_CALL_TEXT):
        return False, None
    match = re.search(r"<function=([^>\n]+)>", stripped)
    if match is None:
        return False, None
    name = match.group(1).strip()
    well_formed = name in tool_names and "</function>" in stripped and "</tool_call>" in stripped
    return well_formed, name


def greedy_induction_audit(
    prepared: Sequence[common.PreparedPrefix],
    *,
    model: Any,
    tokenizer: Any,
    device: torch.device,
    tool_token_id: int,
    layer: int,
    vector: torch.Tensor,
    resources: TauResources,
    max_new_tokens: int,
    direct_rows: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    condition = common.Condition("plus_mean_diff_alpha_1", "coding_direction_induction", 1.0, vector, "add coding vector")
    rows: list[dict[str, Any]] = []
    with common.temporary_last_token_addition(
        model, layer=layer, delta_cpu=condition.delta_cpu, prefill_only=True
    ) as stats:
        for index, prefix in enumerate(prepared, start=1):
            inputs = common.make_left_padded_batch([prefix], int(tokenizer.pad_token_id), device)
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
            well_formed, name = parse_well_formed_qwen_call(text, resources.tool_names)
            direct = direct_rows[prefix.candidate_id]
            rows.append(
                {
                    "candidate_id": prefix.candidate_id,
                    "trace_index": prefix.meta["trace_index"],
                    "first_generated_token_id": new_ids[0] if new_ids else None,
                    "first_generated_token_text": common.token_text(tokenizer, new_ids[0]) if new_ids else "",
                    "first_generated_token_is_tool_call": bool(new_ids and new_ids[0] == tool_token_id),
                    "contains_tool_call": bool(tool_token_id in new_ids),
                    "well_formed_tool_call": well_formed,
                    "parsed_tool_name": name,
                    "generated_token_ids": new_ids,
                    "generated_text": text,
                    "first_token_matches_direct_forward": bool(
                        new_ids and new_ids[0] == int(direct["top1_token_id"])
                    ),
                }
            )
            del inputs, generated
            torch.cuda.empty_cache()
            if index % 10 == 0 or index == len(prepared):
                print(json.dumps({"greedy_induction": [index, len(prepared)]}), flush=True)
    return rows, stats


def count_baseline_compact(rows: Sequence[dict[str, Any]], split: str) -> dict[str, int]:
    selected = [row for row in rows if row["split"] == split]
    return {
        "pairs": len(selected),
        "clean_tool_top1": sum(bool(row["clean_is_tool_call_top1"]) for row in selected),
        "corrupt_tool_top1": sum(bool(row["corrupt_is_tool_call_top1"]) for row in selected),
        "eligible_clean_tool_vs_corrupt_non_tool": sum(
            bool(row["eligible_clean_tool_vs_corrupt_non_tool"]) for row in selected
        ),
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    if min(args.coding_batch_size, args.tau_batch_size, args.final_count) < 1:
        raise ValueError("Batch sizes and final count must be positive")
    if args.layer_fit_count < 1 or args.layer_eval_count < 1:
        raise ValueError("Layer-selection counts must be positive")
    if not 0.0 <= args.max_baseline_tool_probability <= 1.0:
        raise ValueError("--max-baseline-tool-probability must be in [0, 1]")

    spec = SPECS[args.model]
    raw_root = args.raw_root.resolve()
    prepared_root = args.prepared_root.resolve()
    output_root = args.output_root.resolve() / spec.key
    output_root.mkdir(parents=True, exist_ok=True)
    required = [
        spec.model_path,
        spec.coding_root / "manifest.jsonl",
        raw_root / "tau2_system_prompt.txt",
        raw_root / "tau2_tool_schemas.json",
        prepared_root / "screen_pool.jsonl",
        prepared_root / "text_reply_induction_terminal" / "screen_pool.jsonl",
    ]
    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)

    model: Any | None = None
    tokenizer: Any | None = None
    try:
        resources = read_resources(raw_root)
        model, tokenizer, device = load_model_and_tokenizer(spec)
        tool_token_id, token_info = get_tool_token_id(tokenizer)
        layers = list(resolve_qwen_layers(model))
        run_config = {
            "started_unix": time.time(),
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "model": spec.display_name,
            "model_path": str(spec.model_path),
            "decoder_layer_count": len(layers),
            "tool_token": token_info,
            "coding_layer_source": "fresh coding-only disjoint fit/evaluation layer selection in this run",
            "coding_vector_source": "fresh coding-only clean-minus-corrupt mean difference in this run",
            "intervention_hook": "post-output of selected decoder layer; final real prompt token only",
            "tau2_prompt_adapter": "native Qwen3.5 chat template with historical tool_calls rendered as Qwen XML and enable_thinking=False",
            "raw_files": {
                path.name: {"path": str(path), "sha256": sha256_file(path)}
                for path in (raw_root / "tau2_system_prompt.txt", raw_root / "tau2_tool_schemas.json")
            },
            "arguments": vars(args),
        }
        common.write_json(output_root / "run_config.json", run_config)

        # Phase 1: no τ² material is accessed until both the layer and vector
        # have been fixed from this checkpoint's coding data.
        coding_pairs = load_coding_pairs(spec, tokenizer)
        coding_train = [pair for pair in coding_pairs if pair.split == "train"]
        coding_test = [pair for pair in coding_pairs if pair.split == "test"]
        if not coding_train or not coding_test:
            raise RuntimeError("Coding dataset must have nonempty train/test splits")
        print(json.dumps({"coding_pairs": {"train": len(coding_train), "test": len(coding_test)}}), flush=True)
        _clean_baseline, corrupt_baseline, compact_baseline = coding_baseline_screen(
            coding_pairs,
            model=model,
            tokenizer=tokenizer,
            device=device,
            tool_token_id=tool_token_id,
            batch_size=args.coding_batch_size,
        )
        common.write_jsonl(output_root / "coding_baseline_screen.jsonl", compact_baseline)
        eligible_ids = {
            str(row["coding_pair_id"])
            for row in compact_baseline
            if bool(row["eligible_clean_tool_vs_corrupt_non_tool"])
        }
        eligible_train = [pair for pair in coding_train if pair.pair_id in eligible_ids]
        eligible_test = [pair for pair in coding_test if pair.pair_id in eligible_ids]
        if len(eligible_train) < 12:
            raise RuntimeError(
                f"Only {len(eligible_train)} train coding pairs have clean tool / corrupt non-tool behavior; cannot localize"
            )
        layer, layer_selection = choose_layer_from_coding(
            eligible_train,
            corrupt_baseline,
            model=model,
            tokenizer=tokenizer,
            device=device,
            tool_token_id=tool_token_id,
            batch_size=args.coding_batch_size,
            fit_count_requested=args.layer_fit_count,
            eval_count_requested=args.layer_eval_count,
            seed=args.seed + (91 if spec.key == "qwen35_9b" else 47),
            output_root=output_root,
        )
        vector = common.estimate_coding_vector(
            eligible_train,
            model=model,
            tokenizer=tokenizer,
            device=device,
            layer=layer,
            batch_size=args.coding_batch_size,
        )
        generator = torch.Generator(device="cpu")
        generator.manual_seed(args.seed + (941 if spec.key == "qwen35_9b" else 947))
        random_unit = torch.randn(vector.shape, generator=generator, dtype=torch.float32)
        random_unit = random_unit / random_unit.norm().clamp_min(1e-12)
        full_validation = common.coding_validation(
            coding_test,
            model=model,
            tokenizer=tokenizer,
            device=device,
            tool_token_id=tool_token_id,
            layer=layer,
            vector=vector,
            batch_size=args.coding_batch_size,
        )
        eligible_validation = (
            common.coding_validation(
                eligible_test,
                model=model,
                tokenizer=tokenizer,
                device=device,
                tool_token_id=tool_token_id,
                layer=layer,
                vector=vector,
                batch_size=args.coding_batch_size,
            )
            if eligible_test
            else None
        )
        vector_path = output_root / "coding_vector_bundle.pt"
        torch.save(
            {
                "model": spec.display_name,
                "model_path": str(spec.model_path),
                "source_domain": "coding clean/corrupt pairs only",
                "fit_split": "train behavior-positive subset",
                "validation_split": "source test",
                "layer": layer,
                "hook_kind": "post",
                "mean_diff": vector,
                "random_direction_unit": random_unit,
                "random_seed": int(args.seed),
                "tool_token": token_info,
            },
            vector_path,
        )
        coding_summary = {
            "all_pair_baseline": {
                "train": count_baseline_compact(compact_baseline, "train"),
                "test": count_baseline_compact(compact_baseline, "test"),
            },
            "layer_selection": {
                "layer": layer,
                "fit_pairs": layer_selection["fit_count"],
                "evaluation_pairs": layer_selection["evaluation_count"],
                "selection_file": str(output_root / "coding_layer_selection.json"),
            },
            "vector_fit_pairs": len(eligible_train),
            "heldout_pairs": len(coding_test),
            "heldout_behavior_positive_pairs": len(eligible_test),
            "mean_diff_norm": float(vector.norm().item()),
            "validation_full_test": full_validation,
            "validation_behavior_positive_test": eligible_validation,
            "vector_bundle": str(vector_path),
        }
        common.write_json(output_root / "coding_vector_summary.json", coding_summary)

        # Phase 2a: model-specific natural tool-decision screen.
        tool_source = common.read_jsonl(prepared_root / "screen_pool.jsonl")
        tool_screen_input = common.balanced_subset_tool(
            tool_source, limit=args.tool_screen_limit, seed=args.seed
        )
        tool_prepared, tool_rejected = prepare_screen_pool(
            tool_screen_input,
            tokenizer=tokenizer,
            resources=resources,
            max_context_tokens=args.max_context_tokens,
        )
        common.write_json(
            output_root / "tool_render_audit.json",
            {
                "source_rows": len(tool_source),
                "screen_input_rows": len(tool_screen_input),
                "rendered_rows": len(tool_prepared),
                "rejections": tool_rejected,
            },
        )
        baseline = common.Condition("baseline_no_hook_alpha_0", "baseline", 0.0, None, "no intervention")
        tool_screen, tool_screen_stats = common.evaluate_condition(
            tool_prepared,
            model=model,
            tokenizer=tokenizer,
            device=device,
            tool_token_id=tool_token_id,
            layer=layer,
            condition=baseline,
            batch_size=args.tau_batch_size,
        )
        common.write_jsonl(output_root / "tool_baseline_screen.jsonl", tool_screen)
        selected_tool_manifest = common.select_tool_rows(
            tool_screen, final_count=args.final_count, seed=args.seed
        )
        common.write_jsonl(output_root / "selected_tool_prefixes.jsonl", selected_tool_manifest)
        tool_by_id = {str(row["candidate_id"]): row for row in tool_source}
        selected_tool_rows = [tool_by_id[str(row["candidate_id"])] for row in selected_tool_manifest]
        tool_selected, selected_tool_rejected = prepare_screen_pool(
            selected_tool_rows,
            tokenizer=tokenizer,
            resources=resources,
            max_context_tokens=args.max_context_tokens,
        )
        if selected_tool_rejected or len(tool_selected) != args.final_count:
            raise RuntimeError("Selected τ² tool prefixes no longer render")

        # Phase 2b: independent natural text-reply screen for induction.
        induction_source = common.read_jsonl(
            prepared_root / "text_reply_induction_terminal" / "screen_pool.jsonl"
        )
        induction_screen_input = common.balanced_subset_induction(
            induction_source, limit=args.induction_screen_limit, seed=args.seed
        )
        induction_prepared, induction_rejected = prepare_screen_pool(
            induction_screen_input,
            tokenizer=tokenizer,
            resources=resources,
            max_context_tokens=args.max_context_tokens,
        )
        common.write_json(
            output_root / "induction_render_audit.json",
            {
                "source_rows": len(induction_source),
                "screen_input_rows": len(induction_screen_input),
                "rendered_rows": len(induction_prepared),
                "rejections": induction_rejected,
            },
        )
        induction_screen, induction_screen_stats = common.evaluate_condition(
            induction_prepared,
            model=model,
            tokenizer=tokenizer,
            device=device,
            tool_token_id=tool_token_id,
            layer=layer,
            condition=baseline,
            batch_size=args.tau_batch_size,
        )
        common.write_jsonl(output_root / "induction_baseline_screen.jsonl", induction_screen)
        selected_induction_manifest = common.select_induction_rows(
            induction_screen,
            final_count=args.final_count,
            max_probability=args.max_baseline_tool_probability,
            seed=args.seed,
        )
        common.write_jsonl(output_root / "selected_induction_prefixes.jsonl", selected_induction_manifest)
        induction_by_id = {str(row["candidate_id"]): row for row in induction_source}
        selected_induction_rows = [
            induction_by_id[str(row["candidate_id"])] for row in selected_induction_manifest
        ]
        induction_selected, selected_induction_rejected = prepare_screen_pool(
            selected_induction_rows,
            tokenizer=tokenizer,
            resources=resources,
            max_context_tokens=args.max_context_tokens,
        )
        if selected_induction_rejected or len(induction_selected) != args.final_count:
            raise RuntimeError("Selected τ² induction prefixes no longer render")

        # Frozen-vector interventions.  τ² data have not informed the coding
        # layer/vector selection above.
        suppression_rows, suppression_summaries, suppression_hook_stats, _ = common.run_intervention_suite(
            tool_selected,
            kind="suppression",
            model=model,
            tokenizer=tokenizer,
            device=device,
            tool_token_id=tool_token_id,
            layer=layer,
            vector=vector,
            random_unit=random_unit,
            batch_size=args.tau_batch_size,
        )
        induction_rows, induction_summaries, induction_hook_stats, induction_by_condition = common.run_intervention_suite(
            induction_selected,
            kind="induction",
            model=model,
            tokenizer=tokenizer,
            device=device,
            tool_token_id=tool_token_id,
            layer=layer,
            vector=vector,
            random_unit=random_unit,
            batch_size=args.tau_batch_size,
        )
        common.write_jsonl(output_root / "suppression_per_sample.jsonl", suppression_rows)
        common.write_jsonl(output_root / "induction_per_sample.jsonl", induction_rows)
        common.write_csv(output_root / "suppression_summary.csv", suppression_summaries)
        common.write_csv(output_root / "induction_summary.csv", induction_summaries)
        common.write_json(
            output_root / "suppression_summary.json",
            {"conditions": suppression_summaries, "hook_stats": suppression_hook_stats},
        )
        common.write_json(
            output_root / "induction_summary.json",
            {"conditions": induction_summaries, "hook_stats": induction_hook_stats},
        )

        generation_rows: list[dict[str, Any]] = []
        generation_stats = {"hook_calls": 0, "modified_calls": 0}
        if not args.skip_generation:
            generation_rows, generation_stats = greedy_induction_audit(
                induction_selected,
                model=model,
                tokenizer=tokenizer,
                device=device,
                tool_token_id=tool_token_id,
                layer=layer,
                vector=vector,
                resources=resources,
                max_new_tokens=args.max_new_tokens,
                direct_rows={
                    str(row["candidate_id"]): row
                    for row in induction_by_condition["plus_mean_diff_alpha_1"]
                },
            )
        common.write_jsonl(output_root / "induction_alpha1_generations.jsonl", generation_rows)
        result = {
            "model": spec.display_name,
            "coding_vector": coding_summary,
            "screening": {
                "tool": {
                    "screen_stats": tool_screen_stats,
                    "selected": len(selected_tool_manifest),
                    "screened": len(tool_screen),
                },
                "induction": {
                    "screen_stats": induction_screen_stats,
                    "selected": len(selected_induction_manifest),
                    "screened": len(induction_screen),
                },
            },
            "suppression": suppression_summaries,
            "induction": induction_summaries,
            "induction_alpha1_generation": {
                "n": len(generation_rows),
                "first_tool_call": sum(bool(row["first_generated_token_is_tool_call"]) for row in generation_rows),
                "well_formed_tool_call": sum(bool(row["well_formed_tool_call"]) for row in generation_rows),
                "direct_forward_match": sum(bool(row["first_token_matches_direct_forward"]) for row in generation_rows),
                "hook_stats": generation_stats,
            },
            "completed_unix": time.time(),
        }
        common.write_json(output_root / "final_result.json", result)
        print(json.dumps({"completed": True, "model": spec.key, "result": result}, ensure_ascii=False), flush=True)
    finally:
        if model is not None:
            del model
        if tokenizer is not None:
            del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
