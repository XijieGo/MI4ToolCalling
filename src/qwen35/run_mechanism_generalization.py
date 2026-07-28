#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gc
import importlib.util
import json
import math
import os
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
from tqdm.auto import tqdm

import sys

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from artifact_paths import QWEN35_9B_PATH  # noqa: E402


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


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = QWEN35_9B_PATH
DATASET_ROOT = PROJECT_ROOT / "results" / "section6_generalization" / "qwen35_9b" / "datasets"
OUTPUT_ROOT = PROJECT_ROOT / "results" / "section6_generalization" / "qwen35_9b" / "mechanism_generalization_rerun"
TOOL_CALL_TEXT = "<tool_call>"
DEFAULT_BATCH_SIZE = 8
DEFAULT_DTYPE = "bfloat16"


@dataclass(frozen=True)
class PairRecord:
    pair_id: int
    sample_id: str
    split: str
    language: str
    clean_candidate: str
    corrupt_candidate: str
    clean_path: Path
    corrupt_path: Path
    clean_tokens: torch.Tensor
    corrupt_tokens: torch.Tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Qwen3.5-9B mechanism generalization with HF-native hooks.")
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH)
    parser.add_argument(
        "--model-label",
        type=str,
        default="Qwen3.5-9B",
        help="Human-readable model name recorded in generated summaries.",
    )
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--dtype", type=str, default=DEFAULT_DTYPE)
    parser.add_argument("--max-pairs", type=int, default=0)
    parser.add_argument(
        "--causal-metrics-only",
        action="store_true",
        help="Run localization plus vector sufficiency/necessity only; skip the upstream-projection stage.",
    )
    parser.add_argument(
        "--pad-batches",
        action="store_true",
        help="Left-pad causal batches with an attention mask so different prompt lengths can share a batch.",
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    ensure_dir(path.parent)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
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


def clear_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def mean(values: Iterable[float]) -> float:
    items = list(values)
    if not items:
        return math.nan
    return float(sum(items) / len(items))


def decode_token(tokenizer, token_id: int) -> str:
    return tokenizer.decode([int(token_id)], clean_up_tokenization_spaces=False)


def last_token_logits(outputs) -> torch.Tensor:
    logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
    if logits.ndim == 3:
        logits = logits[:, -1, :]
    return logits.detach().float().cpu()


def unwrap_hidden(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)) and output:
        first = output[0]
        if isinstance(first, torch.Tensor):
            return first
    raise TypeError(f"Unsupported hook output type: {type(output).__name__}")


def get_model_dtype(dtype_name: str) -> torch.dtype:
    if not hasattr(torch, dtype_name):
        raise ValueError(f"Unknown dtype: {dtype_name}")
    return getattr(torch, dtype_name)


def resolve_attr_chain(obj: Any, path: str) -> Any:
    for part in path.split("."):
        obj = getattr(obj, part)
    return obj


def resolve_text_model(model) -> Any:
    for path in ("model.language_model", "model", "base_model.model", "base_model", "language_model", "transformer"):
        try:
            candidate = resolve_attr_chain(model, path)
        except AttributeError:
            continue
        if hasattr(candidate, "layers"):
            return candidate
        if hasattr(candidate, "model") and hasattr(candidate.model, "layers"):
            return candidate.model
    raise RuntimeError("Could not locate the decoder stack on the loaded model.")


def get_model_device(model) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration as exc:
        raise RuntimeError("Model has no parameters.") from exc


def get_layers(model) -> list[torch.nn.Module]:
    text_model = resolve_text_model(model)
    layers = getattr(text_model, "layers", None)
    if layers is None:
        raise RuntimeError("Loaded model does not expose text_model.layers.")
    return list(layers)


def get_tool_token_id(tokenizer) -> int:
    token_ids = tokenizer.encode(TOOL_CALL_TEXT, add_special_tokens=False)
    if len(token_ids) != 1:
        raise ValueError(f"{TOOL_CALL_TEXT!r} is not a single token: {token_ids}")
    return int(token_ids[0])


def load_model(model_path: Path, dtype_name: str):
    dtype = get_model_dtype(dtype_name)
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        torch_dtype=dtype,
        device_map={"": 0},
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    # With left padding, the last tensor position is the prediction position
    # for every prompt in a mixed-length batch. The unpadded path is unchanged.
    tokenizer.padding_side = "left"
    return model, tokenizer


def load_pairs(dataset_root: Path, tokenizer) -> list[PairRecord]:
    manifest_path = dataset_root / "manifest.jsonl"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing dataset manifest: {manifest_path}")
    rows = read_jsonl(manifest_path)
    pairs: list[PairRecord] = []
    for row in rows:
        pair_id = int(row["pair_id"])
        sample_id = str(row["sample_id"])
        clean_path = dataset_root / str(row["clean_filename"])
        corrupt_path = dataset_root / str(row["corrupt_filename"])
        clean_text = clean_path.read_text(encoding="utf-8")
        corrupt_text = corrupt_path.read_text(encoding="utf-8")
        clean_tokens = tokenizer(clean_text, add_special_tokens=False, return_tensors="pt")["input_ids"][0].cpu()
        corrupt_tokens = tokenizer(corrupt_text, add_special_tokens=False, return_tensors="pt")["input_ids"][0].cpu()
        if int(clean_tokens.shape[0]) != int(corrupt_tokens.shape[0]):
            raise ValueError(f"Length mismatch for {sample_id}")
        pairs.append(
            PairRecord(
                pair_id=pair_id,
                sample_id=sample_id,
                split=str(row.get("split", "")),
                language=str(row.get("language", "")),
                clean_candidate=str(row.get("clean_candidate", "")),
                corrupt_candidate=str(row.get("corrupt_candidate", "")),
                clean_path=clean_path,
                corrupt_path=corrupt_path,
                clean_tokens=clean_tokens,
                corrupt_tokens=corrupt_tokens,
            )
        )
    pairs.sort(key=lambda item: item.pair_id)
    return pairs


def group_pairs_by_length(pairs: list[PairRecord]) -> list[tuple[int, list[int]]]:
    buckets: dict[int, list[int]] = {}
    for index, pair in enumerate(pairs):
        length = int(pair.clean_tokens.shape[0])
        buckets.setdefault(length, []).append(index)
    return sorted(buckets.items(), key=lambda item: item[0])


def left_pad_token_rows(
    token_rows: list[torch.Tensor],
    *,
    pad_token_id: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a left-padded batch whose final index is every prompt's final token."""
    if not token_rows:
        raise ValueError("Cannot pad an empty token batch.")
    max_length = max(int(tokens.shape[0]) for tokens in token_rows)
    input_ids = torch.full(
        (len(token_rows), max_length),
        int(pad_token_id),
        dtype=token_rows[0].dtype,
    )
    attention_mask = torch.zeros((len(token_rows), max_length), dtype=torch.long)
    for row_idx, tokens in enumerate(token_rows):
        length = int(tokens.shape[0])
        input_ids[row_idx, -length:] = tokens
        attention_mask[row_idx, -length:] = 1
    return input_ids.to(device), attention_mask.to(device)


def iter_causal_batches(
    pairs: list[PairRecord],
    *,
    batch_size: int,
    device: torch.device,
    pad_token_id: int,
    pad_batches: bool,
):
    """Yield paired tensors, optionally batching equal-endpoint left-padded prompts."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    if pad_batches:
        ordered_pairs = sorted(
            pairs,
            key=lambda pair: (int(pair.clean_tokens.shape[0]), pair.pair_id),
        )
        for start in range(0, len(ordered_pairs), batch_size):
            batch_pairs = ordered_pairs[start : start + batch_size]
            clean_tokens, clean_attention_mask = left_pad_token_rows(
                [pair.clean_tokens for pair in batch_pairs],
                pad_token_id=pad_token_id,
                device=device,
            )
            corrupt_tokens, corrupt_attention_mask = left_pad_token_rows(
                [pair.corrupt_tokens for pair in batch_pairs],
                pad_token_id=pad_token_id,
                device=device,
            )
            yield batch_pairs, clean_tokens, clean_attention_mask, corrupt_tokens, corrupt_attention_mask
        return

    for _length, indices in group_pairs_by_length(pairs):
        for start in range(0, len(indices), batch_size):
            batch_pairs = [pairs[idx] for idx in indices[start : start + batch_size]]
            clean_tokens = torch.stack([pair.clean_tokens for pair in batch_pairs], dim=0).to(device)
            corrupt_tokens = torch.stack([pair.corrupt_tokens for pair in batch_pairs], dim=0).to(device)
            yield batch_pairs, clean_tokens, None, corrupt_tokens, None


@contextmanager
def temporary_hooks(hooks: list[tuple[torch.nn.Module, str, Any]]):
    handles = []
    try:
        for module, kind, fn in hooks:
            if kind == "forward":
                handles.append(module.register_forward_hook(fn))
            elif kind == "pre":
                handles.append(module.register_forward_pre_hook(fn))
            else:
                raise ValueError(f"Unknown hook kind: {kind}")
        yield
    finally:
        for handle in reversed(handles):
            handle.remove()


def capture_layer_outputs(
    model,
    layers: list[torch.nn.Module],
    tokens: torch.Tensor,
    *,
    attention_mask: torch.Tensor | None = None,
    capture_hidden_states: bool = False,
):
    captured: list[torch.Tensor | None] = [None] * len(layers)

    def make_hook(layer_idx: int):
        def hook_fn(module, inputs, output):  # noqa: ANN001
            tensor = unwrap_hidden(output)
            captured[layer_idx] = tensor[:, -1, :].detach().cpu().float()
            return output

        return hook_fn

    hooks = [(layer, "forward", make_hook(idx)) for idx, layer in enumerate(layers)]
    with torch.inference_mode():
        with temporary_hooks(hooks):
            model_kwargs: dict[str, Any] = {
                "input_ids": tokens,
                "use_cache": False,
                "output_hidden_states": capture_hidden_states,
                "logits_to_keep": 1,
                "return_dict": True,
            }
            if attention_mask is not None:
                model_kwargs["attention_mask"] = attention_mask
            outputs = model(**model_kwargs)
    if any(item is None for item in captured):
        raise RuntimeError("Failed to capture all layer outputs.")
    return outputs, [item for item in captured if item is not None]


def capture_pass(
    model,
    layers: list[torch.nn.Module],
    tokens: torch.Tensor,
    *,
    attention_mask: torch.Tensor | None = None,
    capture_mlp_layers: set[int] | None = None,
    capture_attn_layers: set[int] | None = None,
    capture_hidden_states: bool = False,
):
    layer_outputs: list[torch.Tensor | None] = [None] * len(layers)
    mlp_outputs: dict[int, torch.Tensor] = {}
    attn_inputs: dict[int, torch.Tensor] = {}

    def make_layer_hook(layer_idx: int):
        def hook_fn(module, inputs, output):  # noqa: ANN001
            tensor = unwrap_hidden(output)
            layer_outputs[layer_idx] = tensor[:, -1, :].detach().cpu().float()
            return output

        return hook_fn

    def make_mlp_hook(layer_idx: int):
        def hook_fn(module, inputs, output):  # noqa: ANN001
            tensor = unwrap_hidden(output)
            mlp_outputs[layer_idx] = tensor[:, -1, :].detach().cpu().float()
            return output

        return hook_fn

    def make_pre_hook(layer_idx: int):
        def hook_fn(module, inputs):  # noqa: ANN001
            attn_inputs[layer_idx] = inputs[0][:, -1, :].detach().cpu().float()
            return None

        return hook_fn

    hook_specs: list[tuple[torch.nn.Module, str, Any]] = []
    for layer_idx, layer in enumerate(layers):
        hook_specs.append((layer, "forward", make_layer_hook(layer_idx)))
        if capture_mlp_layers is not None and layer_idx in capture_mlp_layers:
            hook_specs.append((layer.mlp, "forward", make_mlp_hook(layer_idx)))
        if capture_attn_layers is not None and layer_idx in capture_attn_layers:
            attn_module = layer.self_attn.o_proj if hasattr(layer, "self_attn") else layer.linear_attn.out_proj
            hook_specs.append((attn_module, "pre", make_pre_hook(layer_idx)))

    with torch.inference_mode():
        with temporary_hooks(hook_specs):
            model_kwargs: dict[str, Any] = {
                "input_ids": tokens,
                "use_cache": False,
                "output_hidden_states": capture_hidden_states,
                "logits_to_keep": 1,
                "return_dict": True,
            }
            if attention_mask is not None:
                model_kwargs["attention_mask"] = attention_mask
            outputs = model(**model_kwargs)

    if any(item is None for item in layer_outputs):
        raise RuntimeError("Failed to capture all decoder layer outputs.")
    return outputs, [item for item in layer_outputs if item is not None], mlp_outputs, attn_inputs


def token_stats(logits: torch.Tensor, tool_token_id: int, tokenizer) -> list[dict[str, Any]]:
    probs = torch.softmax(logits, dim=-1)
    top2_probs, top2_ids = torch.topk(probs, k=2, dim=-1)
    top1_ids = top2_ids[:, 0]
    top1_probs = top2_probs[:, 0]
    second_probs = top2_probs[:, 1] if top2_probs.shape[1] > 1 else torch.zeros_like(top1_probs)
    tool_probs = probs[:, tool_token_id]
    tool_logits = logits[:, tool_token_id]
    rows: list[dict[str, Any]] = []
    for row_idx in range(logits.shape[0]):
        rows.append(
            {
                "tool_token_id": int(tool_token_id),
                "tool_token_text": TOOL_CALL_TEXT,
                "tool_logit": float(tool_logits[row_idx].item()),
                "tool_prob": float(tool_probs[row_idx].item()),
                "top1_token_id": int(top1_ids[row_idx].item()),
                "top1_token_text": decode_token(tokenizer, int(top1_ids[row_idx].item())),
                "top1_prob": float(top1_probs[row_idx].item()),
                "top1_margin": float((top1_probs[row_idx] - second_probs[row_idx]).item()),
                "is_tool_call_top1": bool(int(top1_ids[row_idx].item()) == int(tool_token_id)),
            }
        )
    return rows


def make_replace_hook(source_cpu: torch.Tensor, row_slice: slice):
    def hook_fn(module, inputs, output):  # noqa: ANN001
        out = unwrap_hidden(output).clone()
        src = source_cpu.to(device=out.device, dtype=out.dtype)
        out[row_slice, -1, :] = src
        return out

    return hook_fn


def make_add_hook(delta_cpu: torch.Tensor):
    def hook_fn(module, inputs, output):  # noqa: ANN001
        out = unwrap_hidden(output).clone()
        delta = delta_cpu.to(device=out.device, dtype=out.dtype)
        if delta.ndim == 1:
            delta = delta.unsqueeze(0)
        out[:, -1, :] = out[:, -1, :] + delta
        return out

    return hook_fn


def run_state_patch_exp(
    model,
    tokenizer,
    pairs: list[PairRecord],
    *,
    batch_size: int,
    output_root: Path,
    pad_batches: bool = False,
):
    layers = get_layers(model)
    tool_token_id = get_tool_token_id(tokenizer)
    device = get_model_device(model)
    per_sample_rows: list[dict[str, Any]] = []
    summary_acc = {
        layer_idx: {"count": 0, "tool_top1": 0, "strict_flip": 0, "logit_sum": 0.0, "prob_sum": 0.0}
        for layer_idx in range(len(layers))
    }
    clean_top1_total = 0
    corrupt_top1_total = 0
    total_rows = 0

    for batch_pairs, clean_tokens, clean_attention_mask, corrupt_tokens, corrupt_attention_mask in tqdm(
        iter_causal_batches(
            pairs,
            batch_size=batch_size,
            device=device,
            pad_token_id=int(tokenizer.pad_token_id),
            pad_batches=pad_batches,
        ),
        desc="Exp A batches",
        dynamic_ncols=True,
    ):
            clean_outputs, clean_last_states = capture_layer_outputs(
                model,
                layers,
                clean_tokens,
                attention_mask=clean_attention_mask,
            )
            clean_logits = last_token_logits(clean_outputs)
            clean_rows = token_stats(clean_logits, tool_token_id, tokenizer)
            # Qwen3.5's hybrid decoder keeps a large transient state in the
            # model output. Layer-hook captures are already on CPU, so release
            # the clean forward before running the corrupt forward.
            del clean_outputs
            clear_cuda()
            corrupt_model_kwargs: dict[str, Any] = {
                "input_ids": corrupt_tokens,
                "use_cache": False,
                "logits_to_keep": 1,
                "return_dict": True,
            }
            if corrupt_attention_mask is not None:
                corrupt_model_kwargs["attention_mask"] = corrupt_attention_mask
            with torch.inference_mode():
                corrupt_outputs = model(**corrupt_model_kwargs)
            corrupt_logits = last_token_logits(corrupt_outputs)
            corrupt_rows = token_stats(corrupt_logits, tool_token_id, tokenizer)
            del corrupt_outputs
            clear_cuda()
            clean_top1 = torch.tensor([int(row["is_tool_call_top1"]) for row in clean_rows], dtype=torch.bool)
            corrupt_top1 = torch.tensor([int(row["is_tool_call_top1"]) for row in corrupt_rows], dtype=torch.bool)
            clean_top1_total += int(clean_top1.sum().item())
            corrupt_top1_total += int(corrupt_top1.sum().item())
            total_rows += len(batch_pairs)

            for layer_idx, layer_module in enumerate(layers):
                patch_hook = (layer_module, "forward", make_replace_hook(clean_last_states[layer_idx], slice(0, len(batch_pairs))))
                with torch.no_grad():
                    with temporary_hooks([patch_hook]):
                        patched_model_kwargs: dict[str, Any] = {
                            "input_ids": corrupt_tokens,
                            "use_cache": False,
                            "logits_to_keep": 1,
                            "return_dict": True,
                        }
                        if corrupt_attention_mask is not None:
                            patched_model_kwargs["attention_mask"] = corrupt_attention_mask
                        patched_outputs = model(**patched_model_kwargs)
                layer_logits = last_token_logits(patched_outputs)
                layer_rows = token_stats(layer_logits, tool_token_id, tokenizer)
                bucket = summary_acc[layer_idx]
                bucket["count"] += len(batch_pairs)
                bucket["tool_top1"] += sum(int(row["is_tool_call_top1"]) for row in layer_rows)
                bucket["strict_flip"] += sum(
                    int((not corrupt_rows[i]["is_tool_call_top1"]) and layer_rows[i]["is_tool_call_top1"])
                    for i in range(len(batch_pairs))
                )
                bucket["logit_sum"] += sum(float(row["tool_logit"]) for row in layer_rows)
                bucket["prob_sum"] += sum(float(row["tool_prob"]) for row in layer_rows)
                for i, pair in enumerate(batch_pairs):
                    per_sample_rows.append(
                        {
                            "pair_id": pair.pair_id,
                            "sample_id": pair.sample_id,
                            "layer": layer_idx,
                            "clean_tool_logit": float(clean_rows[i]["tool_logit"]),
                            "corrupt_tool_logit": float(corrupt_rows[i]["tool_logit"]),
                            "patched_tool_logit": float(layer_rows[i]["tool_logit"]),
                            "clean_is_tool_call_top1": int(clean_rows[i]["is_tool_call_top1"]),
                            "corrupt_is_tool_call_top1": int(corrupt_rows[i]["is_tool_call_top1"]),
                            "patched_is_tool_call_top1": int(layer_rows[i]["is_tool_call_top1"]),
                            "strict_flip": int((not corrupt_rows[i]["is_tool_call_top1"]) and layer_rows[i]["is_tool_call_top1"]),
                        }
                    )
                del patched_outputs, layer_logits

            del clean_logits, corrupt_logits, clean_tokens, corrupt_tokens, clean_last_states
            clear_cuda()

    summary_rows: list[dict[str, Any]] = []
    for layer_idx, bucket in summary_acc.items():
        count = max(int(bucket["count"]), 1)
        summary_rows.append(
            {
                "layer": layer_idx,
                "n": count,
                "tool_call_top1_rate": float(bucket["tool_top1"] / count),
                "strict_flip_rate": float(bucket["strict_flip"] / count),
                "mean_tool_call_logit": float(bucket["logit_sum"] / count),
                "mean_tool_call_prob": float(bucket["prob_sum"] / count),
                "baseline_clean_tool_call_top1_rate": float(clean_top1_total / max(total_rows, 1)),
                "baseline_corrupt_tool_call_top1_rate": float(corrupt_top1_total / max(total_rows, 1)),
            }
        )
    summary_rows.sort(key=lambda row: int(row["layer"]))
    best_row = max(
        summary_rows,
        key=lambda row: (
            float(row["tool_call_top1_rate"]),
            float(row["strict_flip_rate"]),
            float(row["mean_tool_call_logit"]),
            -int(row["layer"]),
        ),
    )
    write_csv(output_root / "exp_a_state_patch" / "patch_sweep.csv", summary_rows)
    write_csv(output_root / "exp_a_state_patch" / "patch_sweep_per_sample.csv", per_sample_rows)
    write_json(
        output_root / "exp_a_state_patch" / "summary.json",
        {
            "n_pairs": len(pairs),
            "tool_call_token_text": TOOL_CALL_TEXT,
            "tool_call_token_id": int(tool_token_id),
            "best_layer": int(best_row["layer"]),
            "selection_rule": "maximize tool_call_top1_rate, then strict_flip_rate, then mean_tool_call_logit, then earliest layer",
            "best_row": best_row,
            "baseline_clean_tool_call_top1_rate": float(clean_top1_total / max(total_rows, 1)),
            "baseline_corrupt_tool_call_top1_rate": float(corrupt_top1_total / max(total_rows, 1)),
        },
    )
    write_text(
        output_root / "exp_a_state_patch" / "summary.md",
        "\n".join(
            [
                "# Exp A Summary",
                "",
                f"- Samples: `{len(pairs)}`",
                f"- Tool token: `{TOOL_CALL_TEXT}` (`{tool_token_id}`)",
                f"- Best layer: `L{int(best_row['layer'])}`",
                f"- Best patched top-1 rate: `{float(best_row['tool_call_top1_rate']):.4f}`",
                f"- Best strict flip rate: `{float(best_row['strict_flip_rate']):.4f}`",
                f"- Mean tool logit: `{float(best_row['mean_tool_call_logit']):.4f}`",
            ]
        ),
    )
    return int(best_row["layer"]), summary_rows, best_row


def run_vector_exp(
    model,
    tokenizer,
    pairs: list[PairRecord],
    *,
    batch_size: int,
    output_root: Path,
    layer_star: int,
    pad_batches: bool = False,
):
    layers = get_layers(model)
    tool_token_id = get_tool_token_id(tokenizer)
    device = get_model_device(model)
    total_pairs = 0
    mu_delta_sum: torch.Tensor | None = None

    for batch_pairs, clean_tokens, clean_attention_mask, corrupt_tokens, corrupt_attention_mask in tqdm(
        iter_causal_batches(
            pairs,
            batch_size=batch_size,
            device=device,
            pad_token_id=int(tokenizer.pad_token_id),
            pad_batches=pad_batches,
        ),
        desc="Exp B mu_delta",
        dynamic_ncols=True,
    ):
        clean_outputs, clean_last_states = capture_layer_outputs(
            model,
            layers,
            clean_tokens,
            attention_mask=clean_attention_mask,
        )
        del clean_outputs
        clear_cuda()
        corrupt_outputs, corrupt_last_states = capture_layer_outputs(
            model,
            layers,
            corrupt_tokens,
            attention_mask=corrupt_attention_mask,
        )
        del corrupt_outputs
        clear_cuda()
        batch_delta = clean_last_states[layer_star] - corrupt_last_states[layer_star]
        if mu_delta_sum is None:
            mu_delta_sum = batch_delta.sum(dim=0)
        else:
            mu_delta_sum = mu_delta_sum + batch_delta.sum(dim=0)
        total_pairs += len(batch_pairs)
        del clean_tokens, corrupt_tokens, clean_last_states, corrupt_last_states
        clear_cuda()

    if mu_delta_sum is None or total_pairs == 0:
        raise RuntimeError("Failed to collect mu_delta.")
    mu_delta = mu_delta_sum / float(total_pairs)
    mu_delta_norm = float(mu_delta.norm().item())
    if mu_delta_norm == 0.0:
        raise RuntimeError("mu_delta norm is zero.")
    u = mu_delta / mu_delta.norm()

    pair_rows: list[dict[str, Any]] = []
    clean_logit_sum = 0.0
    corrupt_logit_sum = 0.0
    plus_logit_sum = 0.0
    minus_logit_sum = 0.0
    clean_count = 0
    plus_tool_top1 = 0
    minus_tool_top1 = 0
    plus_strict_flip = 0
    minus_strict_drop = 0
    pair_count = 0
    target_layer = layers[layer_star]

    for batch_pairs, clean_tokens, clean_attention_mask, corrupt_tokens, corrupt_attention_mask in tqdm(
        iter_causal_batches(
            pairs,
            batch_size=batch_size,
            device=device,
            pad_token_id=int(tokenizer.pad_token_id),
            pad_batches=pad_batches,
        ),
        desc="Exp B metrics",
        dynamic_ncols=True,
    ):
            clean_model_kwargs: dict[str, Any] = {
                "input_ids": clean_tokens,
                "use_cache": False,
                "logits_to_keep": 1,
                "return_dict": True,
            }
            corrupt_model_kwargs: dict[str, Any] = {
                "input_ids": corrupt_tokens,
                "use_cache": False,
                "logits_to_keep": 1,
                "return_dict": True,
            }
            if clean_attention_mask is not None:
                clean_model_kwargs["attention_mask"] = clean_attention_mask
            if corrupt_attention_mask is not None:
                corrupt_model_kwargs["attention_mask"] = corrupt_attention_mask
            with torch.inference_mode():
                clean_outputs = model(**clean_model_kwargs)
            clean_logits = last_token_logits(clean_outputs)
            clean_rows = token_stats(clean_logits, tool_token_id, tokenizer)
            clean_logit_sum += float(clean_logits[:, tool_token_id].sum().item())
            clean_count += len(batch_pairs)
            del clean_outputs
            clear_cuda()

            with torch.inference_mode():
                corrupt_outputs = model(**corrupt_model_kwargs)
            corrupt_logits = last_token_logits(corrupt_outputs)
            corrupt_rows = token_stats(corrupt_logits, tool_token_id, tokenizer)
            corrupt_logit_sum += float(corrupt_logits[:, tool_token_id].sum().item())
            del corrupt_outputs
            clear_cuda()

            plus_hook = (target_layer, "forward", make_add_hook(mu_delta))
            minus_hook = (target_layer, "forward", make_add_hook(-mu_delta))
            plus_model_kwargs: dict[str, Any] = {
                "input_ids": corrupt_tokens,
                "use_cache": False,
                "logits_to_keep": 1,
                "return_dict": True,
            }
            if corrupt_attention_mask is not None:
                plus_model_kwargs["attention_mask"] = corrupt_attention_mask
            with torch.no_grad():
                with temporary_hooks([plus_hook]):
                    plus_outputs = model(**plus_model_kwargs)
            plus_logits = last_token_logits(plus_outputs)
            plus_rows = token_stats(plus_logits, tool_token_id, tokenizer)
            plus_logit_sum += float(plus_logits[:, tool_token_id].sum().item())
            plus_tool_top1 += sum(int(row["is_tool_call_top1"]) for row in plus_rows)
            del plus_outputs
            clear_cuda()

            minus_model_kwargs: dict[str, Any] = {
                "input_ids": clean_tokens,
                "use_cache": False,
                "logits_to_keep": 1,
                "return_dict": True,
            }
            if clean_attention_mask is not None:
                minus_model_kwargs["attention_mask"] = clean_attention_mask
            with torch.no_grad():
                with temporary_hooks([minus_hook]):
                    minus_outputs = model(**minus_model_kwargs)
            minus_logits = last_token_logits(minus_outputs)
            minus_rows = token_stats(minus_logits, tool_token_id, tokenizer)
            minus_logit_sum += float(minus_logits[:, tool_token_id].sum().item())
            minus_tool_top1 += sum(int(row["is_tool_call_top1"]) for row in minus_rows)
            del minus_outputs
            clear_cuda()
            plus_strict_flip += sum(int((not corrupt_rows[i]["is_tool_call_top1"]) and plus_rows[i]["is_tool_call_top1"]) for i in range(len(batch_pairs)))
            minus_strict_drop += sum(
                int(bool(clean_rows[i]["is_tool_call_top1"]) and (not minus_rows[i]["is_tool_call_top1"]))
                for i in range(len(batch_pairs))
            )
            pair_count += len(batch_pairs)

            for i, pair in enumerate(batch_pairs):
                pair_rows.append(
                    {
                        "pair_id": pair.pair_id,
                        "sample_id": pair.sample_id,
                        "clean_tool_logit": float(clean_rows[i]["tool_logit"]),
                        "corrupt_tool_logit": float(corrupt_rows[i]["tool_logit"]),
                        "vector_plus_tool_logit": float(plus_rows[i]["tool_logit"]),
                        "vector_minus_tool_logit": float(minus_rows[i]["tool_logit"]),
                        "clean_is_tool_call_top1": int(clean_rows[i]["is_tool_call_top1"]),
                        "corrupt_is_tool_call_top1": int(corrupt_rows[i]["is_tool_call_top1"]),
                        "vector_plus_is_tool_call_top1": int(plus_rows[i]["is_tool_call_top1"]),
                        "vector_minus_is_tool_call_top1": int(minus_rows[i]["is_tool_call_top1"]),
                        "vector_plus_strict_flip": int(
                            (not corrupt_rows[i]["is_tool_call_top1"]) and plus_rows[i]["is_tool_call_top1"]
                        ),
                        "vector_minus_strict_drop": int(
                            bool(clean_rows[i]["is_tool_call_top1"]) and (not minus_rows[i]["is_tool_call_top1"])
                        ),
                    }
                )

            del clean_logits, corrupt_logits, plus_logits, minus_logits
            del clean_tokens, corrupt_tokens
            clear_cuda()

    Lc = clean_logit_sum / max(clean_count, 1)
    Lr = corrupt_logit_sum / max(clean_count, 1)
    Lplus = plus_logit_sum / max(clean_count, 1)
    Lminus = minus_logit_sum / max(clean_count, 1)
    denom = Lc - Lr
    suff = float((Lplus - Lr) / denom) if denom != 0 else math.nan
    necc = float((Lc - Lminus) / denom) if denom != 0 else math.nan

    write_csv(output_root / "exp_b_tool_call_vector" / "vector_per_sample.csv", pair_rows)
    write_csv(
        output_root / "exp_b_tool_call_vector" / "vector_summary.csv",
        [
            {
                "layer_star": layer_star,
                "mu_delta_norm": mu_delta_norm,
                "clean_mean_tool_logit": float(Lc),
                "corrupt_mean_tool_logit": float(Lr),
                "vector_plus_mean_tool_logit": float(Lplus),
                "vector_minus_mean_tool_logit": float(Lminus),
                "suff": suff,
                "necc": necc,
                "vector_plus_tool_call_top1_rate": float(plus_tool_top1 / max(pair_count, 1)),
                "vector_minus_tool_call_top1_rate": float(minus_tool_top1 / max(pair_count, 1)),
                "vector_plus_strict_flip_rate": float(plus_strict_flip / max(pair_count, 1)),
                "vector_minus_strict_drop_rate": float(minus_strict_drop / max(pair_count, 1)),
            }
        ],
    )
    write_json(
        output_root / "exp_b_tool_call_vector" / "summary.json",
        {
            "layer_star": int(layer_star),
            "mu_delta": [float(x) for x in mu_delta.tolist()],
            "mu_delta_norm": mu_delta_norm,
            "clean_mean_tool_logit": float(Lc),
            "corrupt_mean_tool_logit": float(Lr),
            "vector_plus_mean_tool_logit": float(Lplus),
            "vector_minus_mean_tool_logit": float(Lminus),
            "suff": suff,
            "necc": necc,
            "vector_plus_tool_call_top1_rate": float(plus_tool_top1 / max(pair_count, 1)),
            "vector_minus_tool_call_top1_rate": float(minus_tool_top1 / max(pair_count, 1)),
            "vector_plus_strict_flip_rate": float(plus_strict_flip / max(pair_count, 1)),
            "vector_minus_strict_drop_rate": float(minus_strict_drop / max(pair_count, 1)),
        },
    )
    write_text(
        output_root / "exp_b_tool_call_vector" / "summary.md",
        "\n".join(
            [
                "# Exp B Summary",
                "",
                f"- L*: `L{layer_star}`",
                f"- mu_delta norm: `{mu_delta_norm:.4f}`",
                f"- suff: `{suff:.4f}`",
                f"- necc: `{necc:.4f}`",
                f"- vector+ top1: `{plus_tool_top1 / max(pair_count, 1):.4f}`",
                f"- vector- top1: `{minus_tool_top1 / max(pair_count, 1):.4f}`",
            ]
        ),
    )
    return mu_delta, u


def get_attention_projection(model, layer: torch.nn.Module):
    text_config = getattr(model.config, "text_config", model.config)
    if hasattr(layer, "self_attn"):
        attn = layer.self_attn
        proj = attn.o_proj
        head_count = int(getattr(text_config, "num_attention_heads", getattr(attn, "num_attention_heads", 1)))
        head_dim = int(getattr(text_config, "head_dim", getattr(attn, "head_dim", proj.in_features // max(head_count, 1))))
        return proj, "full_attention", head_count, head_dim
    if hasattr(layer, "linear_attn"):
        attn = layer.linear_attn
        proj = attn.out_proj
        head_count = int(getattr(text_config, "linear_num_value_heads", getattr(attn, "num_v_heads", 1)))
        head_dim = int(getattr(text_config, "linear_value_head_dim", getattr(attn, "head_v_dim", proj.in_features // max(head_count, 1))))
        return proj, "linear_attention", head_count, head_dim
    raise RuntimeError("Layer does not expose attention projections.")


def run_upstream_exp(
    model,
    tokenizer,
    pairs: list[PairRecord],
    *,
    batch_size: int,
    output_root: Path,
    layer_star: int,
    mu_delta: torch.Tensor,
    u: torch.Tensor,
):
    layers = get_layers(model)
    tool_token_id = get_tool_token_id(tokenizer)
    device = get_model_device(model)
    active_layers = set(range(layer_star))
    trajectory_accum = torch.zeros(len(layers), dtype=torch.float64)
    mlp_accum = torch.zeros(len(layers), dtype=torch.float64)
    head_accum: dict[tuple[int, int], float] = {}
    head_meta: dict[tuple[int, int], dict[str, Any]] = {}

    for _length, indices in tqdm(group_pairs_by_length(pairs), desc="Exp C lengths", dynamic_ncols=True):
        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start : start + batch_size]
            batch_pairs = [pairs[idx] for idx in batch_indices]
            clean_tokens = torch.stack([pair.clean_tokens for pair in batch_pairs], dim=0).to(device)
            corrupt_tokens = torch.stack([pair.corrupt_tokens for pair in batch_pairs], dim=0).to(device)

            clean_outputs, clean_last_states, clean_mlp, clean_attn = capture_pass(
                model,
                layers,
                clean_tokens,
                capture_mlp_layers=active_layers,
                capture_attn_layers=active_layers,
            )
            corrupt_outputs, corrupt_last_states, corrupt_mlp, corrupt_attn = capture_pass(
                model,
                layers,
                corrupt_tokens,
                capture_mlp_layers=active_layers,
                capture_attn_layers=active_layers,
            )

            for layer_idx in range(len(layers)):
                delta = clean_last_states[layer_idx] - corrupt_last_states[layer_idx]
                trajectory_accum[layer_idx] += float((delta.to(dtype=torch.float32) @ u.cpu()).sum().item())

            for layer_idx in active_layers:
                clean_mlp_vec = clean_mlp[layer_idx]
                corrupt_mlp_vec = corrupt_mlp[layer_idx]
                mlp_delta = float(((clean_mlp_vec - corrupt_mlp_vec) @ u.cpu()).sum().item())
                mlp_accum[layer_idx] += mlp_delta

                proj_module, attn_kind, head_count, head_dim = get_attention_projection(model, layers[layer_idx])
                weight = proj_module.weight.detach().cpu().float()
                clean_in = clean_attn[layer_idx]
                corrupt_in = corrupt_attn[layer_idx]
                if clean_in.shape[-1] != head_count * head_dim:
                    raise RuntimeError(
                        f"Attention input shape mismatch at L{layer_idx}: {clean_in.shape[-1]} vs {head_count * head_dim}"
                    )
                weight_t_u = weight.t().contiguous() @ u.cpu().float()
                for head_idx in range(head_count):
                    start_dim = head_idx * head_dim
                    end_dim = start_dim + head_dim
                    head_u = weight_t_u[start_dim:end_dim]
                    clean_score = clean_in[:, start_dim:end_dim] @ head_u
                    corrupt_score = corrupt_in[:, start_dim:end_dim] @ head_u
                    delta = float((clean_score - corrupt_score).sum().item())
                    key = (layer_idx, head_idx)
                    head_accum[key] = head_accum.get(key, 0.0) + delta
                    head_meta[key] = {
                        "layer": layer_idx,
                        "head": head_idx,
                        "kind": attn_kind,
                        "head_label": f"L{layer_idx}H{head_idx}",
                    }

            del clean_outputs, corrupt_outputs, clean_tokens, corrupt_tokens
            clear_cuda()

    traj_rows = [
        {
            "layer": layer_idx,
            "trajectory": float(value / max(len(pairs), 1)),
        }
        for layer_idx, value in enumerate(trajectory_accum.tolist())
    ]
    mlp_rows = [
        {
            "layer": layer_idx,
            "mlp_delta": float(value / max(len(pairs), 1)),
        }
        for layer_idx, value in enumerate(mlp_accum.tolist())
        if layer_idx < layer_star
    ]
    head_rows = []
    for key, value in head_accum.items():
        meta = head_meta[key]
        head_rows.append(
            {
                **meta,
                "head_delta": float(value / max(len(pairs), 1)),
            }
        )
    mlp_rows.sort(key=lambda row: float(row["mlp_delta"]), reverse=True)
    head_rows.sort(key=lambda row: float(row["head_delta"]), reverse=True)

    write_csv(output_root / "exp_c_upstream_projection" / "trajectory.csv", traj_rows)
    write_csv(output_root / "exp_c_upstream_projection" / "mlp_contrib.csv", mlp_rows)
    write_csv(output_root / "exp_c_upstream_projection" / "head_contrib.csv", head_rows)
    write_csv(output_root / "exp_c_upstream_projection" / "top_mlp.csv", mlp_rows[:5])
    write_csv(output_root / "exp_c_upstream_projection" / "top_head.csv", head_rows[:10])
    write_json(
        output_root / "exp_c_upstream_projection" / "summary.json",
        {
            "layer_star": int(layer_star),
            "top_mlp": mlp_rows[:5],
            "top_head": head_rows[:10],
            "trajectory_peak_layer": int(max(traj_rows, key=lambda row: float(row["trajectory"]))["layer"]),
        },
    )
    top_mlp_text = "\n".join(
        [f"- L{int(row['layer'])}: {float(row['mlp_delta']):.4f}" for row in mlp_rows[:5]]
    )
    top_head_text = "\n".join(
        [f"- {row['head_label']}: {float(row['head_delta']):.4f}" for row in head_rows[:10]]
    )
    write_text(
        output_root / "exp_c_upstream_projection" / "summary.md",
        "\n".join(
            [
                "# Exp C Summary",
                "",
                f"- L*: `L{layer_star}`",
                "- Top-5 MLP:",
                top_mlp_text,
                "- Top-10 heads:",
                top_head_text,
            ]
        ),
    )
    return traj_rows, mlp_rows, head_rows


def build_root_summary(
    output_root: Path,
    *,
    model_label: str,
    model_path: Path,
    dataset_root: Path,
    tool_probe: dict[str, Any],
    exp_a_summary: dict[str, Any],
    exp_b_summary: dict[str, Any],
    exp_c_summary: dict[str, Any],
) -> None:
    payload = {
        "model_path": str(model_path),
        "dataset_root": str(dataset_root),
        "output_root": str(output_root),
        "tool_token_probe": tool_probe,
        "exp_a": exp_a_summary,
        "exp_b": exp_b_summary,
        "exp_c": exp_c_summary,
    }
    write_json(output_root / "summary.json", payload)
    write_text(
        output_root / "summary.md",
        "\n".join(
            [
                f"# {model_label} Mechanism Generalization",
                "",
                f"- Tool token: `{tool_probe['tool_token_text']}` (`{tool_probe['tool_token_id']}`)",
                f"- L*: `L{exp_a_summary['best_layer']}`",
                f"- suff: `{exp_b_summary['suff']:.4f}`",
                f"- necc: `{exp_b_summary['necc']:.4f}`",
                f"- Top MLP: `L{exp_c_summary['top_mlp'][0]['layer']}`",
                f"- Top head: `{exp_c_summary['top_head'][0]['head_label']}`",
            ]
        ),
    )


def build_causal_metrics_summary(
    output_root: Path,
    *,
    model_label: str,
    model_path: Path,
    dataset_root: Path,
    n_pairs: int,
    pad_batches: bool,
    exp_a_summary: dict[str, Any],
    exp_b_summary: dict[str, Any],
) -> None:
    """Write the two requested causal metrics with enough provenance to cite them."""
    necc = float(exp_b_summary["necc"])
    suff = float(exp_b_summary["suff"])
    payload = {
        "model": model_label,
        "model_path": str(model_path),
        "dataset_root": str(dataset_root),
        "n_pairs": int(n_pairs),
        "batching": "left_padded" if pad_batches else "equal_length_only",
        "best_layer": int(exp_a_summary["best_layer"]),
        "state_patch_recovery_top1_rate": float(exp_a_summary["best_row"]["tool_call_top1_rate"]),
        "suff": suff,
        "necc": necc,
        "ncc": necc,
        "metric_note": (
            "NCC is an explicit alias for the manuscript's normalized necessity metric Necc. "
            "Both metrics normalize the <tool_call> logit effect by the clean-corrupt logit gap."
        ),
        "sources": {
            "localization": "exp_a_state_patch/summary.json",
            "vector_intervention": "exp_b_tool_call_vector/summary.json",
        },
    }
    write_json(output_root / "causal_metrics.json", payload)
    write_text(
        output_root / "causal_metrics.md",
        "\n".join(
            [
                f"# {model_label} Causal Metrics",
                "",
                f"- Pairs: `{n_pairs}`",
                f"- Selected layer: `L{payload['best_layer']}`",
                f"- Suff: `{suff:.4f}`",
                f"- NCC / Necc: `{necc:.4f}`",
            ]
        ),
    )


def main() -> None:
    args = parse_args()
    ensure_dir(args.output_root)

    model, tokenizer = load_model(args.model_path, args.dtype)
    tool_token_id = get_tool_token_id(tokenizer)
    tool_probe = {
        "tool_token_text": TOOL_CALL_TEXT,
        "tool_token_id": int(tool_token_id),
        "single_token": True,
        "model_path": str(args.model_path),
        "dataset_root": str(args.dataset_root),
        "output_root": str(args.output_root),
        "dtype": args.dtype,
        "causal_metrics_only": bool(args.causal_metrics_only),
        "pad_batches": bool(args.pad_batches),
    }
    write_json(args.output_root / "tool_token_probe.json", tool_probe)
    write_text(
        args.output_root / "tool_token_probe.md",
        "\n".join(
            [
                "# Tool Token Probe",
                "",
                f"- token: `{TOOL_CALL_TEXT}`",
                f"- id: `{tool_token_id}`",
                f"- single token: `true`",
            ]
        ),
    )

    pairs = load_pairs(args.dataset_root, tokenizer)
    if int(args.max_pairs) > 0:
        pairs = pairs[: int(args.max_pairs)]
    write_json(
        args.output_root / "dataset_manifest.json",
        {
            "n_pairs": len(pairs),
            "dataset_root": str(args.dataset_root),
            "sample_ids": [pair.sample_id for pair in pairs],
            "pad_batches": bool(args.pad_batches),
        },
    )

    layer_star, exp_a_rows, _best_row = run_state_patch_exp(
        model,
        tokenizer,
        pairs,
        batch_size=args.batch_size,
        output_root=args.output_root,
        pad_batches=args.pad_batches,
    )
    mu_delta, u = run_vector_exp(
        model,
        tokenizer,
        pairs,
        batch_size=args.batch_size,
        output_root=args.output_root,
        layer_star=layer_star,
        pad_batches=args.pad_batches,
    )

    exp_a_summary = read_json(args.output_root / "exp_a_state_patch" / "summary.json")
    exp_b_summary = read_json(args.output_root / "exp_b_tool_call_vector" / "summary.json")
    if args.causal_metrics_only:
        build_causal_metrics_summary(
            args.output_root,
            model_label=args.model_label,
            model_path=args.model_path,
            dataset_root=args.dataset_root,
            n_pairs=len(pairs),
            pad_batches=args.pad_batches,
            exp_a_summary=exp_a_summary,
            exp_b_summary=exp_b_summary,
        )
    else:
        traj_rows, mlp_rows, head_rows = run_upstream_exp(
            model,
            tokenizer,
            pairs,
            batch_size=args.batch_size,
            output_root=args.output_root,
            layer_star=layer_star,
            mu_delta=mu_delta,
            u=u,
        )
        exp_c_summary = read_json(args.output_root / "exp_c_upstream_projection" / "summary.json")
        build_root_summary(
            args.output_root,
            model_label=args.model_label,
            model_path=args.model_path,
            dataset_root=args.dataset_root,
            tool_probe=tool_probe,
            exp_a_summary=exp_a_summary,
            exp_b_summary=exp_b_summary,
            exp_c_summary=exp_c_summary,
        )

    del model
    clear_cuda()


if __name__ == "__main__":
    main()
