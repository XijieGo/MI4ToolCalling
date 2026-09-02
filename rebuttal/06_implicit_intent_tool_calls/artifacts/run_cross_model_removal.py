#!/usr/bin/env python3
"""Causal transfer of coding-derived tool-call vectors to implicit-intent requests.

The source set contains 600 deterministic, lexically audited implicit-intent
requests.  For each target model this runner:

1. renders the same request text with that model's native tool-call protocol;
2. baseline-screens all 600 requests;
3. takes up to 10 baseline-positive items per (domain, carrier-pattern) cell;
4. removes the frozen coding-derived direction at its model-localized layer;
5. compares mean-direction removal at alpha=1.0 and 1.5 with a same-norm
   random-direction removal at alpha=1.5.

No implicit-intent row is used to estimate, localize, orient, or tune a vector.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import importlib.util
import json
import os
import shutil
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

import torch


from release_paths import (  # noqa: E402
    D1_REFERENCE,
    DEFAULT_CROSS_MODEL_OUTPUT_ROOT,
    FROZEN_CANDIDATES,
    MISTRAL_SYSTEM_PATH,
    MODEL_PATHS,
    QWEN35_REFERENCE,
    VECTOR_PATHS,
)

from multidomain.common import TOOL_SCHEMAS  # noqa: E402


# The Qwen3.5 checkpoint import path probes sklearn in some local transformer
# versions, while sklearn is not needed for this inference-only runner.
_ORIGINAL_FIND_SPEC = importlib.util.find_spec


def _patched_find_spec(name: str, package: str | None = None):
    if name == "sklearn" and os.environ.get("MECH_ENABLE_TRANSFORMERS_SKLEARN", "0") != "1":
        return None
    return _ORIGINAL_FIND_SPEC(name, package)


importlib.util.find_spec = _patched_find_spec
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
finally:
    importlib.util.find_spec = _ORIGINAL_FIND_SPEC


DEFAULT_OUTPUT_ROOT = DEFAULT_CROSS_MODEL_OUTPUT_ROOT
RANDOM_SEED = 20260726
DOMAINS = ("D1", "D3", "D4", "D5")
PATTERNS = ("P1", "P2", "P3", "P4", "P5")
CONDITIONS = (("mean_diff_a1.0", "mean_diff", 1.0), ("mean_diff_a1.5", "mean_diff", 1.5), ("random_a1.5", "random", 1.5))


@dataclass(frozen=True)
class ModelSpec:
    key: str
    display_name: str
    model_path: Path
    vector_path: Path
    tool_token_text: str
    renderer: str
    batch_size: int


SPECS: dict[str, ModelSpec] = {
    "qwen3_4b": ModelSpec(
        key="qwen3_4b",
        display_name="Qwen3-4B",
        model_path=MODEL_PATHS["qwen3_4b"],
        vector_path=VECTOR_PATHS["qwen3_4b"],
        tool_token_text="<tool_call>",
        renderer="qwen3_source",
        batch_size=8,
    ),
    "qwen3_8b": ModelSpec(
        key="qwen3_8b",
        display_name="Qwen3-8B",
        model_path=MODEL_PATHS["qwen3_8b"],
        vector_path=VECTOR_PATHS["qwen3_8b"],
        tool_token_text="<tool_call>",
        renderer="qwen3_source",
        batch_size=6,
    ),
    "qwen3_14b": ModelSpec(
        key="qwen3_14b",
        display_name="Qwen3-14B",
        model_path=MODEL_PATHS["qwen3_14b"],
        vector_path=VECTOR_PATHS["qwen3_14b"],
        tool_token_text="<tool_call>",
        renderer="qwen3_source",
        batch_size=4,
    ),
    "qwen35_4b": ModelSpec(
        key="qwen35_4b",
        display_name="Qwen3.5-4B",
        model_path=MODEL_PATHS["qwen35_4b"],
        vector_path=VECTOR_PATHS["qwen35_4b"],
        tool_token_text="<tool_call>",
        renderer="qwen35_source_style",
        batch_size=8,
    ),
    "qwen35_9b": ModelSpec(
        key="qwen35_9b",
        display_name="Qwen3.5-9B",
        model_path=MODEL_PATHS["qwen35_9b"],
        vector_path=VECTOR_PATHS["qwen35_9b"],
        tool_token_text="<tool_call>",
        renderer="qwen35_source_style",
        batch_size=6,
    ),
    "mistral": ModelSpec(
        key="mistral",
        display_name="Mistral-Small-3.2-24B-Instruct-2506",
        model_path=MODEL_PATHS["mistral"],
        vector_path=VECTOR_PATHS["mistral"],
        tool_token_text="[TOOL_CALLS]",
        renderer="mistral_native",
        batch_size=2,
    ),
    "granite": ModelSpec(
        key="granite",
        display_name="Granite-3.3-8B-Instruct",
        model_path=MODEL_PATHS["granite"],
        vector_path=VECTOR_PATHS["granite"],
        tool_token_text="<|tool_call|>",
        renderer="granite_native",
        batch_size=6,
    ),
}


GRANITE_SYSTEM = (
    "Knowledge Cutoff Date: April 2024.\n"
    "Today's Date: May 05, 2026.\n"
    "You are Granite, developed by IBM. You are a helpful assistant with access to the following tools. "
    "When a tool is required to answer the user's query, respond only with <|tool_call|> followed "
    "by a JSON list of tools used. If a tool does not exist in the provided list of tools, notify the "
    "user that you do not have the ability to fulfill the request."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=sorted(SPECS), default=list(SPECS))
    parser.add_argument(
        "--source-path",
        type=Path,
        default=FROZEN_CANDIDATES,
        help="Frozen 600-item JSONL included in this release by default.",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--batch-size", type=int, default=0, help="Override the conservative model-specific batch size.")
    parser.add_argument("--limit", type=int, default=0, help="Optional smoke-test cap on source candidates; zero uses all 600.")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing per-model output directory.")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def clear_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def parse_d1_tool() -> dict[str, Any]:
    text = D1_REFERENCE.read_text(encoding="utf-8")
    payload = text.split("<tools>\n", 1)[1].split("\n</tools>", 1)[0]
    return json.loads(payload)


def source_tools() -> dict[str, list[dict[str, Any]]]:
    return {
        "D1": [parse_d1_tool()],
        "D3": [json.loads(TOOL_SCHEMAS["D3"])],
        "D4": [json.loads(TOOL_SCHEMAS["D4"])],
        "D5": [json.loads(TOOL_SCHEMAS["D5"])],
    }


def qwen_style_render(reference: Path, *, tools: list[dict[str, Any]], user_text: str) -> str:
    text = reference.read_text(encoding="utf-8")
    tool_open = "<tools>\n"
    tool_start = text.index(tool_open) + len(tool_open)
    tool_end = text.index("</tools>", tool_start)
    user_marker = "<|im_start|>user\n"
    user_start = text.index(user_marker, tool_end) + len(user_marker)
    assistant_marker = "<|im_end|>\n<|im_start|>assistant\n"
    user_end = text.index(assistant_marker, user_start)
    schema = "\n".join(json.dumps(tool, ensure_ascii=False) for tool in tools)
    return text[:tool_start] + schema + text[tool_end:user_start] + user_text.strip() + text[user_end:]


def mistral_system_prompt() -> str:
    payload = json.loads(MISTRAL_SYSTEM_PATH.read_text(encoding="utf-8"))
    value = str(payload.get("system_prompt_preview") or "")
    if not value:
        raise RuntimeError(f"No system_prompt_preview in {MISTRAL_SYSTEM_PATH}")
    return value


def render_native_ids(tokenizer: Any, spec: ModelSpec, row: dict[str, Any], tools: list[dict[str, Any]]) -> list[int]:
    if spec.renderer == "qwen3_source":
        return [int(value) for value in tokenizer(str(row["prompt"]), add_special_tokens=False)["input_ids"]]
    if spec.renderer == "qwen35_source_style":
        rendered = qwen_style_render(QWEN35_REFERENCE, tools=tools, user_text=str(row["text"]))
        return [int(value) for value in tokenizer(rendered, add_special_tokens=False)["input_ids"]]

    if spec.renderer == "mistral_native":
        messages = [
            {"role": "system", "content": mistral_system_prompt()},
            {"role": "user", "content": str(row["text"])},
        ]
        encoded = tokenizer.apply_chat_template(
            messages,
            tools=tools,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        return [int(value) for value in encoded["input_ids"][0].detach().cpu().tolist()]

    if spec.renderer == "granite_native":
        messages = [
            {"role": "system", "content": GRANITE_SYSTEM},
            {"role": "user", "content": str(row["text"])},
        ]
        try:
            rendered = tokenizer.apply_chat_template(
                messages, tools=tools, tokenize=False, add_generation_prompt=True, thinking=False
            )
        except TypeError:
            rendered = tokenizer.apply_chat_template(messages, tools=tools, tokenize=False, add_generation_prompt=True)
        return [int(value) for value in tokenizer(rendered, add_special_tokens=False)["input_ids"]]

    raise ValueError(f"Unknown renderer {spec.renderer!r}")


def resolve_layers(model: Any) -> Sequence[torch.nn.Module]:
    candidates: list[Any] = [model]
    for path in (
        ("model",),
        ("model", "model"),
        ("model", "language_model"),
        ("language_model",),
        ("base_model",),
        ("base_model", "model"),
    ):
        current = model
        try:
            for attr in path:
                current = getattr(current, attr)
            candidates.append(current)
        except AttributeError:
            pass
    for candidate in candidates:
        layers = getattr(candidate, "layers", None)
        if layers is not None:
            return layers
    raise RuntimeError("Could not locate the decoder layer list")


def get_tool_token_id(tokenizer: Any, token: str) -> tuple[int, dict[str, Any]]:
    converted = int(tokenizer.convert_tokens_to_ids(token))
    if converted < 0:
        raise RuntimeError(f"Could not resolve tool token {token!r}")
    encoded = [int(value) for value in tokenizer.encode(token, add_special_tokens=False)]
    return converted, {"text": token, "id_via_convert": converted, "ids_via_encode": encoded}


def load_vector(spec: ModelSpec) -> tuple[torch.Tensor, torch.Tensor, int, str, dict[str, Any]]:
    bundle = torch.load(spec.vector_path, map_location="cpu", weights_only=False)
    mean_diff = torch.as_tensor(bundle["mean_diff"], dtype=torch.float32).flatten().contiguous()
    # Older Qwen3 artifacts record a literal ``layer: None`` alongside the
    # operative ``patch_layer`` field, so ``dict.get(..., fallback)`` is not
    # sufficient here.
    layer = bundle.get("layer") or bundle.get("patch_layer")
    hook_kind = str(bundle.get("hook_kind", "pre"))
    if layer is None or hook_kind not in {"pre", "post"}:
        raise ValueError(f"Invalid vector metadata in {spec.vector_path}")
    random_unit = bundle.get("random_direction_unit")
    if random_unit is None:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(RANDOM_SEED)
        random_unit = torch.randn(mean_diff.shape, generator=generator, dtype=torch.float32)
    random_unit = torch.as_tensor(random_unit, dtype=torch.float32).flatten().contiguous()
    random_unit = random_unit / random_unit.norm().clamp_min(1e-12)
    random_direction = random_unit * mean_diff.norm()
    metadata = {
        "bundle_path": str(spec.vector_path),
        "bundle_sha256": sha256_file(spec.vector_path),
        "mean_diff_norm": float(mean_diff.norm().item()),
        "random_direction_norm": float(random_direction.norm().item()),
        "cosine_mean_random": float(torch.nn.functional.cosine_similarity(mean_diff, random_direction, dim=0).item()),
        "bundle_layer": int(layer),
        "bundle_hook_kind": hook_kind,
        "bundle_keys": sorted(bundle.keys()),
    }
    return mean_diff, random_direction, int(layer), hook_kind, metadata


@contextlib.contextmanager
def temporary_vector_removal(
    model: Any, *, layer_index: int, hook_kind: str, delta_cpu: torch.Tensor | None
) -> Iterator[dict[str, int]]:
    stats = {"hook_calls": 0, "modified_calls": 0}
    if delta_cpu is None:
        yield stats
        return
    layers = resolve_layers(model)
    if not 0 <= layer_index < len(layers):
        raise ValueError(f"Layer {layer_index} invalid for a {len(layers)}-layer model")

    if hook_kind == "pre":
        def pre_hook(_module: Any, args: tuple[Any, ...]):
            stats["hook_calls"] += 1
            hidden = args[0]
            if not isinstance(hidden, torch.Tensor) or hidden.ndim != 3:
                raise RuntimeError(f"Unexpected pre-hook input: {type(hidden)!r}")
            if hidden.shape[-1] != delta_cpu.numel():
                raise RuntimeError("Vector hidden dimension mismatch")
            patched = hidden.clone()
            patched[:, -1, :] += delta_cpu.to(device=patched.device, dtype=patched.dtype)
            stats["modified_calls"] += 1
            return (patched, *args[1:])

        handle = layers[layer_index].register_forward_pre_hook(pre_hook)
    else:
        def post_hook(_module: Any, _args: tuple[Any, ...], output: Any):
            stats["hook_calls"] += 1
            hidden = output[0] if isinstance(output, tuple) else output
            if not isinstance(hidden, torch.Tensor) or hidden.ndim != 3:
                raise RuntimeError(f"Unexpected post-hook output: {type(hidden)!r}")
            if hidden.shape[-1] != delta_cpu.numel():
                raise RuntimeError("Vector hidden dimension mismatch")
            patched = hidden.clone()
            patched[:, -1, :] += delta_cpu.to(device=patched.device, dtype=patched.dtype)
            stats["modified_calls"] += 1
            return (patched, *output[1:]) if isinstance(output, tuple) else patched

        handle = layers[layer_index].register_forward_hook(post_hook)
    try:
        yield stats
    finally:
        handle.remove()


def make_batches(rows: Sequence[dict[str, Any]], batch_size: int) -> Iterator[list[dict[str, Any]]]:
    ordered = sorted(rows, key=lambda row: len(row["input_ids"]))
    for start in range(0, len(ordered), batch_size):
        yield ordered[start : start + batch_size]


def forward_logits(model: Any, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    kwargs: dict[str, Any] = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "use_cache": False,
        "return_dict": True,
    }
    with torch.inference_mode():
        try:
            outputs = model(**kwargs, logits_to_keep=1)
        except TypeError:
            outputs = model(**kwargs)
    logits = outputs.logits
    if logits.ndim != 3:
        raise RuntimeError(f"Expected rank-3 logits, got {tuple(logits.shape)}")
    return logits[:, -1, :].float()


def evaluate(
    rows: Sequence[dict[str, Any]],
    *,
    model: Any,
    tokenizer: Any,
    tool_token_id: int,
    batch_size: int,
    layer: int,
    hook_kind: str,
    delta_cpu: torch.Tensor | None,
) -> tuple[dict[str, dict[str, Any]], dict[str, int]]:
    device = next(model.parameters()).device
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
    results: dict[str, dict[str, Any]] = {}
    aggregate_hooks = {"hook_calls": 0, "modified_calls": 0}
    for batch_index, batch in enumerate(make_batches(rows, batch_size), start=1):
        max_len = max(len(row["input_ids"]) for row in batch)
        input_ids = torch.full((len(batch), max_len), int(pad_id), dtype=torch.long, device=device)
        attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long, device=device)
        for index, row in enumerate(batch):
            token_ids = torch.tensor(row["input_ids"], dtype=torch.long, device=device)
            input_ids[index, -token_ids.numel() :] = token_ids
            attention_mask[index, -token_ids.numel() :] = 1
        with temporary_vector_removal(model, layer_index=layer, hook_kind=hook_kind, delta_cpu=delta_cpu) as hook_stats:
            logits = forward_logits(model, input_ids, attention_mask)
        aggregate_hooks["hook_calls"] += hook_stats["hook_calls"]
        aggregate_hooks["modified_calls"] += hook_stats["modified_calls"]
        top_values, top_ids = torch.max(logits, dim=-1)
        tool_logits = logits[:, tool_token_id]
        probs = torch.softmax(logits, dim=-1)[:, tool_token_id]
        for index, row in enumerate(batch):
            top_id = int(top_ids[index].item())
            results[str(row["item_id"])] = {
                "is_tool_call_top1": bool(top_id == tool_token_id),
                "top1_token_id": top_id,
                "top1_token_text": tokenizer.decode([top_id], clean_up_tokenization_spaces=False),
                "top1_logit": float(top_values[index].item()),
                "tool_call_logit": float(tool_logits[index].item()),
                "tool_call_probability": float(probs[index].item()),
            }
        del input_ids, attention_mask, logits
        # A full Python/CUDA collection after every singleton forward is much
        # slower than the model computation itself.  Periodic collection keeps
        # long 600-item runs stable without changing any logits.
        if batch_index % 32 == 0:
            clear_cuda()
    return results, aggregate_hooks


def select_removal_arm(rows: Sequence[dict[str, Any]], baseline: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for domain in DOMAINS:
        for pattern in PATTERNS:
            candidates = [
                row for row in rows
                if row["domain"] == domain and row["pattern"] == pattern and baseline[row["item_id"]]["is_tool_call_top1"]
            ]
            selected.extend(candidates[:10])
    return selected


def summarize(rows: Sequence[dict[str, Any]], results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    by_domain: dict[str, Any] = {}
    for domain in DOMAINS:
        domain_rows = [row for row in rows if row["domain"] == domain]
        n = len(domain_rows)
        drops = sum(not results[row["item_id"]]["is_tool_call_top1"] for row in domain_rows)
        by_domain[domain] = {"n": n, "strict_drop": drops, "strict_drop_rate": drops / n if n else None}
    n = len(rows)
    drops = sum(not results[row["item_id"]]["is_tool_call_top1"] for row in rows)
    return {"n": n, "strict_drop": drops, "strict_drop_rate": drops / n if n else None, "by_domain": by_domain}


def run_model(spec: ModelSpec, args: argparse.Namespace) -> dict[str, Any]:
    output_dir = args.output_root.resolve() / spec.key
    if output_dir.exists() and any(output_dir.iterdir()):
        if not args.overwrite:
            raise FileExistsError(f"Output exists: {output_dir}; pass --overwrite to replace it")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not spec.model_path.exists() or not spec.vector_path.exists():
        raise FileNotFoundError(f"Missing model or vector for {spec.key}")

    source_rows = read_jsonl(args.source_path)
    if args.limit:
        source_rows = source_rows[: args.limit]
    tools_by_domain = source_tools()
    tokenizer = AutoTokenizer.from_pretrained(str(spec.model_path), trust_remote_code=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    tool_token_id, token_info = get_tool_token_id(tokenizer, spec.tool_token_text)
    prepared: list[dict[str, Any]] = []
    for row in source_rows:
        input_ids = render_native_ids(tokenizer, spec, row, tools_by_domain[str(row["domain"])])
        if not input_ids:
            raise RuntimeError(f"Empty template output for {row['item_id']}")
        prepared.append({**row, "input_ids": input_ids, "prompt_token_count": len(input_ids)})
    write_jsonl(output_dir / "rendered_600.jsonl", [{k: v for k, v in row.items() if k != "input_ids"} for row in prepared])

    mean_diff, random_direction, layer, hook_kind, vector_meta = load_vector(spec)
    # Mistral-Small-3.2 is packaged as a Mistral3 multimodal checkpoint even
    # for text-only prompts, so it is intentionally loaded through its
    # conditional-generation class.  The other checkpoints are ordinary
    # causal-LM models.
    if spec.key == "mistral":
        from transformers import Mistral3ForConditionalGeneration

        mistral_kwargs: dict[str, Any] = {
            "device_map": {"": 0},
            "low_cpu_mem_usage": True,
            "trust_remote_code": True,
        }
        try:
            model = Mistral3ForConditionalGeneration.from_pretrained(
                str(spec.model_path), dtype=torch.bfloat16, **mistral_kwargs
            )
        except TypeError:
            model = Mistral3ForConditionalGeneration.from_pretrained(
                str(spec.model_path), torch_dtype=torch.bfloat16, **mistral_kwargs
            )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            str(spec.model_path), torch_dtype=torch.bfloat16, device_map={"": 0}, low_cpu_mem_usage=True, trust_remote_code=True
        )
    model.eval()
    layers = resolve_layers(model)
    if len(layers) <= layer:
        raise RuntimeError(f"{spec.key} has {len(layers)} layers, cannot hook L{layer}")
    hidden_size = int(getattr(model.config, "hidden_size", mean_diff.numel()))
    if hidden_size != mean_diff.numel():
        raise RuntimeError(f"Vector dim {mean_diff.numel()} != model hidden size {hidden_size}")
    batch_size = args.batch_size or spec.batch_size
    run_config = {
        "model": spec.display_name,
        "model_path": str(spec.model_path),
        "renderer": spec.renderer,
        "source_candidates": len(prepared),
        "source_path": str(args.source_path),
        "source_sha256": sha256_file(args.source_path),
        "tool_token": token_info,
        "layer": layer,
        "hook_kind": hook_kind,
        "decoder_layers": len(layers),
        "batch_size": batch_size,
        "vector": vector_meta,
        "conditions": [{"name": name, "direction": direction, "alpha": alpha} for name, direction, alpha in CONDITIONS],
        "started_unix": time.time(),
    }
    write_json(output_dir / "run_config.json", run_config)
    try:
        baseline, baseline_hooks = evaluate(
            prepared, model=model, tokenizer=tokenizer, tool_token_id=tool_token_id, batch_size=batch_size,
            layer=layer, hook_kind=hook_kind, delta_cpu=None
        )
        arm = select_removal_arm(prepared, baseline)
        intervention_results: dict[str, dict[str, dict[str, Any]]] = {}
        hook_audits: dict[str, dict[str, int]] = {"baseline": baseline_hooks}
        for name, direction_name, alpha in CONDITIONS:
            direction = mean_diff if direction_name == "mean_diff" else random_direction
            results, hook_audit = evaluate(
                arm, model=model, tokenizer=tokenizer, tool_token_id=tool_token_id, batch_size=batch_size,
                layer=layer, hook_kind=hook_kind, delta_cpu=-float(alpha) * direction,
            )
            intervention_results[name] = results
            hook_audits[name] = hook_audit

        output_rows: list[dict[str, Any]] = []
        for row in prepared:
            record = {key: value for key, value in row.items() if key != "input_ids"}
            record["baseline"] = baseline[row["item_id"]]
            output_rows.append(record)
        write_jsonl(output_dir / "baseline_600.jsonl", output_rows)
        arm_rows: list[dict[str, Any]] = []
        for row in arm:
            record = {key: value for key, value in row.items() if key != "input_ids"}
            record["baseline"] = baseline[row["item_id"]]
            record["conditions"] = {name: intervention_results[name][row["item_id"]] for name, _, _ in CONDITIONS}
            arm_rows.append(record)
        write_jsonl(output_dir / "removal_arm.jsonl", arm_rows)

        baseline_by_domain_pattern = {
            domain: {
                pattern: {
                    "n": sum(1 for row in prepared if row["domain"] == domain and row["pattern"] == pattern),
                    "baseline_tool_call_top1": sum(
                        baseline[row["item_id"]]["is_tool_call_top1"]
                        for row in prepared if row["domain"] == domain and row["pattern"] == pattern
                    ),
                }
                for pattern in PATTERNS
            }
            for domain in DOMAINS
        }
        summary = {
            "model": spec.display_name,
            "source_candidates": len(prepared),
            "baseline_tool_call_top1": sum(item["is_tool_call_top1"] for item in baseline.values()),
            "baseline_by_domain_pattern": baseline_by_domain_pattern,
            "removal_arm_n": len(arm),
            "conditions": {name: summarize(arm, intervention_results[name]) for name, _, _ in CONDITIONS},
            "hook_audits": hook_audits,
            "vector": vector_meta,
            "layer": layer,
            "hook_kind": hook_kind,
            "tool_token": token_info,
        }
        write_json(output_dir / "summary.json", summary)
        print(json.dumps({"model": spec.key, "summary": summary["conditions"], "n": len(arm)}, ensure_ascii=False), flush=True)
        return summary
    finally:
        del model
        clear_cuda()


def main() -> None:
    args = parse_args()
    aggregate_path = args.output_root.resolve() / "cross_model_summary.json"
    all_summaries: dict[str, Any] = (
        json.loads(aggregate_path.read_text(encoding="utf-8")) if aggregate_path.exists() else {}
    )
    for key in args.models:
        print(json.dumps({"event": "start", "model": key}, ensure_ascii=False), flush=True)
        all_summaries[key] = run_model(SPECS[key], args)
    write_json(aggregate_path, all_summaries)


if __name__ == "__main__":
    main()
