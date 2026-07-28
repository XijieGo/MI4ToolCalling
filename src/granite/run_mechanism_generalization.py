#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
from tqdm.auto import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from granite_toolcall_common import DEFAULT_MODEL_PATH, TOOL_CALL_TOKEN, ensure_dir, read_jsonl, write_json


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "results" / "section6_generalization" / "granite_3p3_8b" / "dataset"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "results" / "section6_generalization" / "granite_3p3_8b" / "mechanism_generalization_rerun"


@dataclass
class PairRecord:
    index: int
    sample_id: str
    split: str
    dataset_name: str
    language: str
    template_kind: str
    clean_candidate: str
    corrupt_candidate: str
    clean_prompt_path: Path
    corrupt_prompt_path: Path
    clean_text: str
    corrupt_text: str
    clean_input_ids: list[int]
    corrupt_input_ids: list[int]
    clean_len: int
    corrupt_len: int
    max_len: int


@dataclass
class BatchRecord:
    indices: list[int]
    clean_input_ids_cpu: torch.Tensor
    clean_attention_mask_cpu: torch.Tensor
    corrupt_input_ids_cpu: torch.Tensor
    corrupt_attention_mask_cpu: torch.Tensor
    max_len: int


@dataclass
class BaselineBundle:
    clean_states: torch.Tensor
    corrupt_states: torch.Tensor
    clean_tool_logits: torch.Tensor
    corrupt_tool_logits: torch.Tensor
    clean_tool_probs: torch.Tensor
    corrupt_tool_probs: torch.Tensor
    clean_top1_ids: torch.Tensor
    corrupt_top1_ids: torch.Tensor
    clean_top1_text: list[str]
    corrupt_top1_text: list[str]
    per_pair_rows: list[dict[str, object]]
    summary: dict[str, object]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Granite mechanism generalization with HF-native hooks only."
    )
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=("bfloat16", "float16", "float32"))
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max-pairs", type=int, default=0)
    parser.add_argument("--wait-poll-seconds", type=int, default=30)
    return parser.parse_args()


def clear_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    ensure_dir(path.parent)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_text(path: Path, text: str) -> None:
    ensure_dir(path.parent)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def dtype_from_name(name: str) -> torch.dtype:
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    if name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def dtype_num_bytes(dtype: torch.dtype) -> int:
    if dtype == torch.bfloat16:
        return 2
    if dtype == torch.float16:
        return 2
    if dtype == torch.float32:
        return 4
    raise ValueError(f"Unsupported dtype for byte-size estimate: {dtype}")


def gib(bytes_value: float) -> float:
    return float(bytes_value) / float(1024**3)


def safe_ratio(numerator: float, denominator: float) -> float:
    if math.isclose(float(denominator), 0.0, abs_tol=1e-12):
        return float("nan")
    return float(numerator / denominator)


def query_free_vram_gib(device_index: int = 0) -> float:
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=memory.free",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    if device_index >= len(lines):
        raise IndexError(f"GPU index {device_index} out of range for nvidia-smi output.")
    free_mib = float(lines[device_index])
    return float(free_mib / 1024.0)


def wait_for_vram(required_gib: float, *, poll_seconds: int, device_index: int = 0) -> dict[str, float]:
    initial_free_gib = query_free_vram_gib(device_index=device_index)
    free_gib = initial_free_gib
    while free_gib < required_gib:
        print(
            json.dumps(
                {
                    "event": "wait_for_vram",
                    "required_gib": round(required_gib, 3),
                    "free_gib": round(free_gib, 3),
                    "sleep_seconds": int(poll_seconds),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        time.sleep(max(int(poll_seconds), 1))
        free_gib = query_free_vram_gib(device_index=device_index)
    return {
        "initial_free_gib": float(initial_free_gib),
        "ready_free_gib": float(free_gib),
        "required_gib": float(required_gib),
    }


def model_weight_bytes_on_disk(model_path: Path) -> int:
    candidates = list(model_path.glob("*.safetensors")) + list(model_path.glob("*.bin"))
    if not candidates:
        return 0
    return int(sum(path.stat().st_size for path in candidates))


def estimate_required_vram_gib(
    *,
    config,
    dtype: torch.dtype,
    batch_size: int,
    max_seq_len: int,
    model_path: Path,
) -> dict[str, float]:
    hidden_size = int(config.hidden_size)
    num_hidden_layers = int(config.num_hidden_layers)
    vocab_size = int(config.vocab_size)
    dtype_bytes = dtype_num_bytes(dtype)
    weight_bytes = model_weight_bytes_on_disk(model_path)
    hidden_state_bytes = (num_hidden_layers + 1) * batch_size * max_seq_len * hidden_size * dtype_bytes
    logits_bytes = batch_size * max_seq_len * vocab_size * dtype_bytes
    runtime_overhead_bytes = 6 * 1024**3
    estimated_peak_bytes = weight_bytes + hidden_state_bytes + logits_bytes + runtime_overhead_bytes
    return {
        "weight_gib_on_disk": gib(weight_bytes),
        "hidden_state_gib": gib(hidden_state_bytes),
        "logits_gib": gib(logits_bytes),
        "runtime_overhead_gib": gib(runtime_overhead_bytes),
        "estimated_peak_gib": gib(estimated_peak_bytes),
    }


def choose_best_layer(rows: Sequence[dict[str, object]], *, tolerance: float = 0.02) -> int:
    if not rows:
        raise ValueError("No rows supplied.")
    rows_sorted = sorted(rows, key=lambda row: int(row["layer"]))
    max_rate = max(float(row["tool_call_top1_rate"]) for row in rows_sorted)
    candidates = [
        row for row in rows_sorted if float(row["tool_call_top1_rate"]) >= max_rate - float(tolerance)
    ]
    if candidates:
        return int(candidates[0]["layer"])
    return int(max(rows_sorted, key=lambda row: float(row["tool_call_top1_rate"]))["layer"])


def top1_decode(tokenizer, token_ids: torch.Tensor) -> list[str]:
    return [
        tokenizer.decode([int(token_id)], clean_up_tokenization_spaces=False)
        for token_id in token_ids.detach().cpu().tolist()
    ]


def summarize_side(
    *,
    tool_logits: torch.Tensor,
    tool_probs: torch.Tensor,
    top1_ids: torch.Tensor,
    tool_token_id: int,
) -> dict[str, object]:
    is_tool = top1_ids == int(tool_token_id)
    return {
        "n": int(top1_ids.numel()),
        "tool_call_top1_count": int(is_tool.sum().item()),
        "tool_call_top1_rate": float(is_tool.float().mean().item()),
        "mean_tool_call_logit": float(tool_logits.float().mean().item()),
        "mean_tool_call_prob": float(tool_probs.float().mean().item()),
    }


def load_pairs(tokenizer, dataset_root: Path, *, max_pairs: int = 0) -> list[PairRecord]:
    manifest_path = dataset_root / "manifest.jsonl"
    canonical_path = dataset_root / "canonical_pairs.jsonl"
    manifest_rows = read_jsonl(manifest_path)
    canonical_rows = read_jsonl(canonical_path)
    manifest_sample_ids = {str(row["sample_id"]) for row in manifest_rows}
    canonical_sample_ids = {str(row["sample_id"]) for row in canonical_rows}
    if manifest_sample_ids != canonical_sample_ids:
        raise RuntimeError(
            "Granite dataset manifest.jsonl and canonical_pairs.jsonl do not reference the same 500 sample ids."
        )

    pairs: list[PairRecord] = []
    for index, row in enumerate(manifest_rows):
        clean_prompt_path = Path(str(row["clean_prompt_path"]))
        corrupt_prompt_path = Path(str(row["corrupt_prompt_path"]))
        clean_text = clean_prompt_path.read_text(encoding="utf-8")
        corrupt_text = corrupt_prompt_path.read_text(encoding="utf-8")
        clean_input_ids = list(tokenizer(clean_text, add_special_tokens=False)["input_ids"])
        corrupt_input_ids = list(tokenizer(corrupt_text, add_special_tokens=False)["input_ids"])
        pairs.append(
            PairRecord(
                index=index,
                sample_id=str(row["sample_id"]),
                split=str(row["split"]),
                dataset_name=str(row["dataset_name"]),
                language=str(row["language"]),
                template_kind=str(row["template_kind"]),
                clean_candidate=str(row["clean_candidate"]),
                corrupt_candidate=str(row["corrupt_candidate"]),
                clean_prompt_path=clean_prompt_path,
                corrupt_prompt_path=corrupt_prompt_path,
                clean_text=clean_text,
                corrupt_text=corrupt_text,
                clean_input_ids=clean_input_ids,
                corrupt_input_ids=corrupt_input_ids,
                clean_len=len(clean_input_ids),
                corrupt_len=len(corrupt_input_ids),
                max_len=max(len(clean_input_ids), len(corrupt_input_ids)),
            )
        )
        if max_pairs > 0 and len(pairs) >= max_pairs:
            break
    if not pairs:
        raise RuntimeError(f"No pairs found under {dataset_root}.")
    return pairs


def build_batches(tokenizer, pairs: Sequence[PairRecord], *, batch_size: int) -> list[BatchRecord]:
    order = sorted(range(len(pairs)), key=lambda idx: (pairs[idx].max_len, pairs[idx].sample_id))
    batches: list[BatchRecord] = []
    for start in range(0, len(order), max(int(batch_size), 1)):
        indices = order[start : start + max(int(batch_size), 1)]
        clean_encoded = tokenizer.pad(
            {"input_ids": [pairs[idx].clean_input_ids for idx in indices]},
            return_tensors="pt",
            padding=True,
        )
        corrupt_encoded = tokenizer.pad(
            {"input_ids": [pairs[idx].corrupt_input_ids for idx in indices]},
            return_tensors="pt",
            padding=True,
        )
        batches.append(
            BatchRecord(
                indices=indices,
                clean_input_ids_cpu=clean_encoded["input_ids"].detach().cpu(),
                clean_attention_mask_cpu=clean_encoded["attention_mask"].detach().cpu(),
                corrupt_input_ids_cpu=corrupt_encoded["input_ids"].detach().cpu(),
                corrupt_attention_mask_cpu=corrupt_encoded["attention_mask"].detach().cpu(),
                max_len=max(
                    int(clean_encoded["input_ids"].shape[1]),
                    int(corrupt_encoded["input_ids"].shape[1]),
                ),
            )
        )
    return batches


def load_model_and_tokenizer(
    *,
    model_path: Path,
    dtype: torch.dtype,
    device: str,
):
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    tool_token_ids = tokenizer.encode(TOOL_CALL_TOKEN, add_special_tokens=False)
    if len(tool_token_ids) != 1:
        raise RuntimeError(f"{TOOL_CALL_TOKEN!r} must be a single token, got {tool_token_ids}.")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        dtype=dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model.to(device)
    model.eval()
    return model, tokenizer, int(tool_token_ids[0])


def forward_model(
    model,
    *,
    input_ids_cpu: torch.Tensor,
    attention_mask_cpu: torch.Tensor,
    output_hidden_states: bool = False,
):
    device = next(model.parameters()).device
    input_ids = input_ids_cpu.to(device=device, non_blocking=True)
    attention_mask = attention_mask_cpu.to(device=device, non_blocking=True)
    with torch.inference_mode():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
        )
    return outputs


def extract_last_token_metrics(outputs, *, tool_token_id: int):
    last_logits = outputs.logits[:, -1, :].detach().float().cpu()
    tool_logits = last_logits[:, int(tool_token_id)].contiguous()
    probs = torch.softmax(last_logits, dim=-1)
    tool_probs = probs[:, int(tool_token_id)].contiguous()
    top1_ids = last_logits.argmax(dim=-1).contiguous()
    return tool_logits, tool_probs, top1_ids


def extract_last_token_states(outputs, *, num_layers: int) -> torch.Tensor:
    hidden_states = outputs.hidden_states
    if hidden_states is None or len(hidden_states) != num_layers + 1:
        raise RuntimeError("Unexpected hidden_states output shape.")
    states = torch.stack(
        [hidden_state[:, -1, :].detach().cpu().float() for hidden_state in hidden_states[1:]],
        dim=1,
    )
    return states


def peak_memory_gib(device: str) -> float:
    if not torch.cuda.is_available() or not str(device).startswith("cuda"):
        return 0.0
    dev = torch.device(device)
    return gib(torch.cuda.max_memory_allocated(dev))


def reset_peak_memory(device: str) -> None:
    if torch.cuda.is_available() and str(device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(torch.device(device))


def collect_baseline(
    model,
    tokenizer,
    pairs: Sequence[PairRecord],
    batches: Sequence[BatchRecord],
    *,
    tool_token_id: int,
    num_layers: int,
    device: str,
) -> BaselineBundle:
    n_pairs = len(pairs)
    d_model = int(model.config.hidden_size)
    clean_states = torch.empty((n_pairs, num_layers, d_model), dtype=torch.float32)
    corrupt_states = torch.empty((n_pairs, num_layers, d_model), dtype=torch.float32)
    clean_tool_logits = torch.empty(n_pairs, dtype=torch.float32)
    corrupt_tool_logits = torch.empty(n_pairs, dtype=torch.float32)
    clean_tool_probs = torch.empty(n_pairs, dtype=torch.float32)
    corrupt_tool_probs = torch.empty(n_pairs, dtype=torch.float32)
    clean_top1_ids = torch.empty(n_pairs, dtype=torch.long)
    corrupt_top1_ids = torch.empty(n_pairs, dtype=torch.long)
    clean_top1_text = [""] * n_pairs
    corrupt_top1_text = [""] * n_pairs

    progress = tqdm(batches, desc="Baseline states", dynamic_ncols=True)
    for batch in progress:
        clean_outputs = forward_model(
            model,
            input_ids_cpu=batch.clean_input_ids_cpu,
            attention_mask_cpu=batch.clean_attention_mask_cpu,
            output_hidden_states=True,
        )
        corrupt_outputs = forward_model(
            model,
            input_ids_cpu=batch.corrupt_input_ids_cpu,
            attention_mask_cpu=batch.corrupt_attention_mask_cpu,
            output_hidden_states=True,
        )
        clean_batch_states = extract_last_token_states(clean_outputs, num_layers=num_layers)
        corrupt_batch_states = extract_last_token_states(corrupt_outputs, num_layers=num_layers)
        clean_batch_logits, clean_batch_probs, clean_batch_top1 = extract_last_token_metrics(
            clean_outputs,
            tool_token_id=tool_token_id,
        )
        corrupt_batch_logits, corrupt_batch_probs, corrupt_batch_top1 = extract_last_token_metrics(
            corrupt_outputs,
            tool_token_id=tool_token_id,
        )
        clean_batch_text = top1_decode(tokenizer, clean_batch_top1)
        corrupt_batch_text = top1_decode(tokenizer, corrupt_batch_top1)

        for local_idx, pair_idx in enumerate(batch.indices):
            clean_states[pair_idx] = clean_batch_states[local_idx]
            corrupt_states[pair_idx] = corrupt_batch_states[local_idx]
            clean_tool_logits[pair_idx] = clean_batch_logits[local_idx]
            corrupt_tool_logits[pair_idx] = corrupt_batch_logits[local_idx]
            clean_tool_probs[pair_idx] = clean_batch_probs[local_idx]
            corrupt_tool_probs[pair_idx] = corrupt_batch_probs[local_idx]
            clean_top1_ids[pair_idx] = clean_batch_top1[local_idx]
            corrupt_top1_ids[pair_idx] = corrupt_batch_top1[local_idx]
            clean_top1_text[pair_idx] = clean_batch_text[local_idx]
            corrupt_top1_text[pair_idx] = corrupt_batch_text[local_idx]

        del clean_outputs, corrupt_outputs
        del clean_batch_states, corrupt_batch_states
        del clean_batch_logits, clean_batch_probs, clean_batch_top1, clean_batch_text
        del corrupt_batch_logits, corrupt_batch_probs, corrupt_batch_top1, corrupt_batch_text
        clear_cuda()
        progress.set_postfix(max_len=batch.max_len)

    per_pair_rows: list[dict[str, object]] = []
    for pair in pairs:
        per_pair_rows.append(
            {
                "sample_id": pair.sample_id,
                "split": pair.split,
                "dataset_name": pair.dataset_name,
                "language": pair.language,
                "template_kind": pair.template_kind,
                "clean_candidate": pair.clean_candidate,
                "corrupt_candidate": pair.corrupt_candidate,
                "clean_len": pair.clean_len,
                "corrupt_len": pair.corrupt_len,
                "clean_tool_call_logit": float(clean_tool_logits[pair.index].item()),
                "corrupt_tool_call_logit": float(corrupt_tool_logits[pair.index].item()),
                "clean_tool_call_prob": float(clean_tool_probs[pair.index].item()),
                "corrupt_tool_call_prob": float(corrupt_tool_probs[pair.index].item()),
                "clean_top1_token_id": int(clean_top1_ids[pair.index].item()),
                "corrupt_top1_token_id": int(corrupt_top1_ids[pair.index].item()),
                "clean_top1_token_text": clean_top1_text[pair.index],
                "corrupt_top1_token_text": corrupt_top1_text[pair.index],
                "clean_is_tool_call_top1": bool(clean_top1_ids[pair.index].item() == tool_token_id),
                "corrupt_is_tool_call_top1": bool(corrupt_top1_ids[pair.index].item() == tool_token_id),
            }
        )

    clean_summary = summarize_side(
        tool_logits=clean_tool_logits,
        tool_probs=clean_tool_probs,
        top1_ids=clean_top1_ids,
        tool_token_id=tool_token_id,
    )
    corrupt_summary = summarize_side(
        tool_logits=corrupt_tool_logits,
        tool_probs=corrupt_tool_probs,
        top1_ids=corrupt_top1_ids,
        tool_token_id=tool_token_id,
    )
    summary = {
        "n_pairs": int(n_pairs),
        "clean": clean_summary,
        "corrupt": corrupt_summary,
        "clean_minus_corrupt_top1_gap": float(
            clean_summary["tool_call_top1_rate"] - corrupt_summary["tool_call_top1_rate"]
        ),
        "clean_minus_corrupt_logit_gap": float(
            clean_summary["mean_tool_call_logit"] - corrupt_summary["mean_tool_call_logit"]
        ),
    }
    return BaselineBundle(
        clean_states=clean_states,
        corrupt_states=corrupt_states,
        clean_tool_logits=clean_tool_logits,
        corrupt_tool_logits=corrupt_tool_logits,
        clean_tool_probs=clean_tool_probs,
        corrupt_tool_probs=corrupt_tool_probs,
        clean_top1_ids=clean_top1_ids,
        corrupt_top1_ids=corrupt_top1_ids,
        clean_top1_text=clean_top1_text,
        corrupt_top1_text=corrupt_top1_text,
        per_pair_rows=per_pair_rows,
        summary=summary,
    )


def _replace_layer_output(output, replacement: torch.Tensor):
    if isinstance(output, tuple):
        tensor = output[0]
        out = tensor.clone()
        out[:, -1, :] = replacement
        return (out, *output[1:])
    out = output.clone()
    out[:, -1, :] = replacement
    return out


def _add_to_layer_output(output, delta: torch.Tensor):
    if isinstance(output, tuple):
        tensor = output[0]
        out = tensor.clone()
        out[:, -1, :] = out[:, -1, :] + delta
        return (out, *output[1:])
    out = output.clone()
    out[:, -1, :] = out[:, -1, :] + delta
    return out


def run_state_patch_experiment(
    model,
    tokenizer,
    pairs: Sequence[PairRecord],
    batches: Sequence[BatchRecord],
    baseline: BaselineBundle,
    *,
    tool_token_id: int,
    output_root: Path,
    device: str,
) -> tuple[int, list[dict[str, object]], dict[str, object]]:
    ensure_dir(output_root)
    write_csv(output_root / "baseline_pair_metrics.csv", baseline.per_pair_rows)
    write_json(output_root / "baseline_summary.json", baseline.summary)

    num_layers = int(model.config.num_hidden_layers)
    layer_rows: list[dict[str, object]] = []
    per_pair_rows: list[dict[str, object]] = []
    baseline_corrupt_is_tool = baseline.corrupt_top1_ids == int(tool_token_id)

    progress = tqdm(range(num_layers), desc="Exp A layer sweep", dynamic_ncols=True)
    for layer_idx in progress:
        patched_tool_top1 = 0
        strict_flip = 0
        top1_changed = 0
        logit_sum = 0.0
        prob_sum = 0.0
        count = 0
        for batch in batches:
            replacement_cpu = baseline.clean_states[batch.indices, layer_idx, :].contiguous()

            def hook_fn(_module, _inputs, output):
                replacement = replacement_cpu.to(device=next(model.parameters()).device, dtype=model.dtype)
                return _replace_layer_output(output, replacement)

            handle = model.model.layers[layer_idx].register_forward_hook(hook_fn)
            try:
                outputs = forward_model(
                    model,
                    input_ids_cpu=batch.corrupt_input_ids_cpu,
                    attention_mask_cpu=batch.corrupt_attention_mask_cpu,
                    output_hidden_states=False,
                )
            finally:
                handle.remove()

            batch_tool_logits, batch_tool_probs, batch_top1_ids = extract_last_token_metrics(
                outputs,
                tool_token_id=tool_token_id,
            )
            batch_top1_text = top1_decode(tokenizer, batch_top1_ids)
            batch_is_tool = batch_top1_ids == int(tool_token_id)
            batch_baseline_is_tool = baseline_corrupt_is_tool[batch.indices]
            batch_baseline_top1_ids = baseline.corrupt_top1_ids[batch.indices]
            patched_tool_top1 += int(batch_is_tool.sum().item())
            strict_flip += int((~batch_baseline_is_tool & batch_is_tool).sum().item())
            top1_changed += int((batch_baseline_top1_ids != batch_top1_ids).sum().item())
            logit_sum += float(batch_tool_logits.sum().item())
            prob_sum += float(batch_tool_probs.sum().item())
            count += len(batch.indices)

            for local_idx, pair_idx in enumerate(batch.indices):
                pair = pairs[pair_idx]
                per_pair_rows.append(
                    {
                        "sample_id": pair.sample_id,
                        "layer": int(layer_idx),
                        "baseline_corrupt_tool_call_logit": float(
                            baseline.corrupt_tool_logits[pair_idx].item()
                        ),
                        "baseline_corrupt_tool_call_prob": float(
                            baseline.corrupt_tool_probs[pair_idx].item()
                        ),
                        "baseline_corrupt_top1_token_id": int(
                            baseline.corrupt_top1_ids[pair_idx].item()
                        ),
                        "baseline_corrupt_top1_token_text": baseline.corrupt_top1_text[pair_idx],
                        "baseline_corrupt_is_tool_call_top1": bool(
                            baseline.corrupt_top1_ids[pair_idx].item() == tool_token_id
                        ),
                        "patched_tool_call_logit": float(batch_tool_logits[local_idx].item()),
                        "patched_tool_call_prob": float(batch_tool_probs[local_idx].item()),
                        "patched_top1_token_id": int(batch_top1_ids[local_idx].item()),
                        "patched_top1_token_text": batch_top1_text[local_idx],
                        "patched_is_tool_call_top1": bool(
                            batch_top1_ids[local_idx].item() == tool_token_id
                        ),
                        "strict_flip": bool(
                            baseline.corrupt_top1_ids[pair_idx].item() != tool_token_id
                            and batch_top1_ids[local_idx].item() == tool_token_id
                        ),
                        "top1_changed": bool(
                            baseline.corrupt_top1_ids[pair_idx].item() != batch_top1_ids[local_idx].item()
                        ),
                    }
                )

            del outputs, batch_tool_logits, batch_tool_probs, batch_top1_ids, batch_top1_text
            clear_cuda()

        layer_rows.append(
            {
                "layer": int(layer_idx),
                "n": int(count),
                "tool_call_top1_rate": float(patched_tool_top1 / max(count, 1)),
                "strict_flip_rate": float(strict_flip / max(count, 1)),
                "top1_changed_rate": float(top1_changed / max(count, 1)),
                "mean_tool_call_logit": float(logit_sum / max(count, 1)),
                "mean_tool_call_prob": float(prob_sum / max(count, 1)),
                "baseline_corrupt_tool_call_top1_rate": float(
                    baseline_corrupt_is_tool.float().mean().item()
                ),
                "baseline_corrupt_mean_tool_call_logit": float(
                    baseline.corrupt_tool_logits.mean().item()
                ),
                "baseline_corrupt_mean_tool_call_prob": float(
                    baseline.corrupt_tool_probs.mean().item()
                ),
            }
        )
    best_layer = choose_best_layer(layer_rows)
    best_row = next(row for row in layer_rows if int(row["layer"]) == best_layer)

    summary = {
        "best_layer": int(best_layer),
        "best_layer_metrics": best_row,
        "n_layers": int(num_layers),
        "n_pairs": int(len(pairs)),
        "tool_call_token": TOOL_CALL_TOKEN,
        "tool_call_token_id": int(tool_token_id),
        "patch_target": "model.model.layers[layer_idx]",
        "hidden_state_source": "output_hidden_states=True -> hidden_states[1:]",
        "observed_peak_memory_gib": peak_memory_gib(device),
    }
    write_csv(output_root / "patch_sweep.csv", layer_rows)
    write_csv(output_root / "patch_per_pair.csv", per_pair_rows)
    write_json(output_root / "summary.json", summary)
    write_text(
        output_root / "summary.md",
        "\n".join(
            [
                "# Exp A Summary",
                "",
                f"- n_pairs: {len(pairs)}",
                f"- best_layer: L{best_layer}",
                f"- best_tool_call_top1_rate: {float(best_row['tool_call_top1_rate']):.4f}",
                f"- best_strict_flip_rate: {float(best_row['strict_flip_rate']):.4f}",
                f"- best_mean_tool_call_logit: {float(best_row['mean_tool_call_logit']):.4f}",
                f"- baseline_corrupt_tool_call_top1_rate: {float(best_row['baseline_corrupt_tool_call_top1_rate']):.4f}",
                "",
                "State patch is implemented with a HuggingFace forward hook on model.model.layers[layer_idx].",
            ]
        ),
    )
    return best_layer, layer_rows, summary


def run_vector_experiment(
    model,
    tokenizer,
    pairs: Sequence[PairRecord],
    batches: Sequence[BatchRecord],
    baseline: BaselineBundle,
    *,
    best_layer: int,
    tool_token_id: int,
    output_root: Path,
    device: str,
) -> tuple[torch.Tensor, dict[str, object]]:
    ensure_dir(output_root)
    clean_layer_states = baseline.clean_states[:, best_layer, :]
    corrupt_layer_states = baseline.corrupt_states[:, best_layer, :]
    mu_delta = (clean_layer_states - corrupt_layer_states).mean(dim=0).contiguous()
    mean_clean_state = clean_layer_states.mean(dim=0).contiguous()
    mean_corrupt_state = corrupt_layer_states.mean(dim=0).contiguous()

    torch.save(
        {
            "layer": int(best_layer),
            "tool_call_token": TOOL_CALL_TOKEN,
            "tool_call_token_id": int(tool_token_id),
            "mean_clean": mean_clean_state,
            "mean_corrupt": mean_corrupt_state,
            "mean_diff": mu_delta,
            "n_pairs": int(len(pairs)),
        },
        output_root / "mu_delta_bundle.pt",
    )

    rows: list[dict[str, object]] = []
    plus_pair_rows: list[dict[str, object]] = []
    minus_pair_rows: list[dict[str, object]] = []

    def run_condition(*, side: str, delta_sign: float) -> tuple[dict[str, object], list[dict[str, object]]]:
        if side not in {"clean", "corrupt"}:
            raise ValueError(f"Unsupported side: {side}")
        tool_top1 = 0
        strict_event = 0
        logit_sum = 0.0
        prob_sum = 0.0
        count = 0
        condition_rows: list[dict[str, object]] = []
        iterator = tqdm(
            batches,
            desc=f"Exp B {side} {'add' if delta_sign > 0 else 'remove'}",
            dynamic_ncols=True,
        )
        for batch in iterator:
            delta_cpu = (mu_delta * float(delta_sign)).contiguous()

            def hook_fn(_module, _inputs, output):
                delta = delta_cpu.to(device=next(model.parameters()).device, dtype=model.dtype)
                return _add_to_layer_output(output, delta)

            handle = model.model.layers[best_layer].register_forward_hook(hook_fn)
            try:
                if side == "corrupt":
                    outputs = forward_model(
                        model,
                        input_ids_cpu=batch.corrupt_input_ids_cpu,
                        attention_mask_cpu=batch.corrupt_attention_mask_cpu,
                        output_hidden_states=False,
                    )
                    baseline_logits = baseline.corrupt_tool_logits
                    baseline_probs = baseline.corrupt_tool_probs
                    baseline_top1_ids = baseline.corrupt_top1_ids
                    baseline_top1_text = baseline.corrupt_top1_text
                    source_indices = batch.indices
                else:
                    outputs = forward_model(
                        model,
                        input_ids_cpu=batch.clean_input_ids_cpu,
                        attention_mask_cpu=batch.clean_attention_mask_cpu,
                        output_hidden_states=False,
                    )
                    baseline_logits = baseline.clean_tool_logits
                    baseline_probs = baseline.clean_tool_probs
                    baseline_top1_ids = baseline.clean_top1_ids
                    baseline_top1_text = baseline.clean_top1_text
                    source_indices = batch.indices
            finally:
                handle.remove()

            batch_tool_logits, batch_tool_probs, batch_top1_ids = extract_last_token_metrics(
                outputs,
                tool_token_id=tool_token_id,
            )
            batch_top1_text = top1_decode(tokenizer, batch_top1_ids)
            batch_is_tool = batch_top1_ids == int(tool_token_id)
            batch_baseline_is_tool = baseline_top1_ids[source_indices] == int(tool_token_id)
            tool_top1 += int(batch_is_tool.sum().item())
            if side == "corrupt":
                strict_event += int((~batch_baseline_is_tool & batch_is_tool).sum().item())
            else:
                strict_event += int((batch_baseline_is_tool & ~batch_is_tool).sum().item())
            logit_sum += float(batch_tool_logits.sum().item())
            prob_sum += float(batch_tool_probs.sum().item())
            count += len(source_indices)

            for local_idx, pair_idx in enumerate(source_indices):
                pair = pairs[pair_idx]
                condition_rows.append(
                    {
                        "sample_id": pair.sample_id,
                        "condition": (
                            "vector_plus_corrupt" if side == "corrupt" else "vector_minus_clean"
                        ),
                        "layer": int(best_layer),
                        "baseline_tool_call_logit": float(baseline_logits[pair_idx].item()),
                        "baseline_tool_call_prob": float(baseline_probs[pair_idx].item()),
                        "baseline_top1_token_id": int(baseline_top1_ids[pair_idx].item()),
                        "baseline_top1_token_text": baseline_top1_text[pair_idx],
                        "baseline_is_tool_call_top1": bool(
                            baseline_top1_ids[pair_idx].item() == tool_token_id
                        ),
                        "patched_tool_call_logit": float(batch_tool_logits[local_idx].item()),
                        "patched_tool_call_prob": float(batch_tool_probs[local_idx].item()),
                        "patched_top1_token_id": int(batch_top1_ids[local_idx].item()),
                        "patched_top1_token_text": batch_top1_text[local_idx],
                        "patched_is_tool_call_top1": bool(
                            batch_top1_ids[local_idx].item() == tool_token_id
                        ),
                        "strict_event": bool(
                            (
                                baseline_top1_ids[pair_idx].item() != tool_token_id
                                and batch_top1_ids[local_idx].item() == tool_token_id
                            )
                            if side == "corrupt"
                            else (
                                baseline_top1_ids[pair_idx].item() == tool_token_id
                                and batch_top1_ids[local_idx].item() != tool_token_id
                            )
                        ),
                    }
                )
            del outputs, batch_tool_logits, batch_tool_probs, batch_top1_ids, batch_top1_text
            clear_cuda()

        metric_name = (
            "vector_plus_strict_flip_rate" if side == "corrupt" else "vector_minus_strict_drop_rate"
        )
        top1_metric_name = (
            "vector_plus_tool_call_top1_rate" if side == "corrupt" else "vector_minus_tool_call_top1_rate"
        )
        condition_summary = {
            "condition": "vector_plus" if side == "corrupt" else "vector_minus",
            "layer": int(best_layer),
            "n": int(count),
            top1_metric_name: float(tool_top1 / max(count, 1)),
            metric_name: float(strict_event / max(count, 1)),
            "mean_tool_call_logit": float(logit_sum / max(count, 1)),
            "mean_tool_call_prob": float(prob_sum / max(count, 1)),
            "delta_sign": float(delta_sign),
        }
        return condition_summary, condition_rows

    plus_summary, plus_pair_rows = run_condition(side="corrupt", delta_sign=1.0)
    minus_summary, minus_pair_rows = run_condition(side="clean", delta_sign=-1.0)
    rows.extend([plus_summary, minus_summary])

    clean_mean_logit = float(baseline.clean_tool_logits.mean().item())
    corrupt_mean_logit = float(baseline.corrupt_tool_logits.mean().item())
    plus_mean_logit = float(plus_summary["mean_tool_call_logit"])
    minus_mean_logit = float(minus_summary["mean_tool_call_logit"])
    denominator = clean_mean_logit - corrupt_mean_logit
    suff = safe_ratio(plus_mean_logit - corrupt_mean_logit, denominator)
    necc = safe_ratio(clean_mean_logit - minus_mean_logit, denominator)

    summary = {
        "layer": int(best_layer),
        "n_pairs": int(len(pairs)),
        "mu_delta_norm": float(mu_delta.norm().item()),
        "clean_mean_tool_call_logit": clean_mean_logit,
        "corrupt_mean_tool_call_logit": corrupt_mean_logit,
        "vector_plus_mean_tool_call_logit": plus_mean_logit,
        "vector_minus_mean_tool_call_logit": minus_mean_logit,
        "suff": float(suff),
        "necc": float(necc),
        "vector_plus_tool_call_top1_rate": float(plus_summary["vector_plus_tool_call_top1_rate"]),
        "vector_plus_strict_flip_rate": float(plus_summary["vector_plus_strict_flip_rate"]),
        "vector_minus_tool_call_top1_rate": float(minus_summary["vector_minus_tool_call_top1_rate"]),
        "vector_minus_strict_drop_rate": float(minus_summary["vector_minus_strict_drop_rate"]),
        "observed_peak_memory_gib": peak_memory_gib(device),
    }
    write_csv(output_root / "vector_intervention_summary.csv", rows)
    write_csv(output_root / "vector_plus_per_pair.csv", plus_pair_rows)
    write_csv(output_root / "vector_minus_per_pair.csv", minus_pair_rows)
    write_json(output_root / "summary.json", summary)
    write_text(
        output_root / "summary.md",
        "\n".join(
            [
                "# Exp B Summary",
                "",
                f"- layer: L{best_layer}",
                f"- mu_delta_norm: {float(mu_delta.norm().item()):.6f}",
                f"- suff: {float(suff):.6f}",
                f"- necc: {float(necc):.6f}",
                f"- vector_plus_tool_call_top1_rate: {float(plus_summary['vector_plus_tool_call_top1_rate']):.4f}",
                f"- vector_plus_strict_flip_rate: {float(plus_summary['vector_plus_strict_flip_rate']):.4f}",
                f"- vector_minus_tool_call_top1_rate: {float(minus_summary['vector_minus_tool_call_top1_rate']):.4f}",
                f"- vector_minus_strict_drop_rate: {float(minus_summary['vector_minus_strict_drop_rate']):.4f}",
                "",
                "Vector interventions are implemented with a HuggingFace forward hook on model.model.layers[L*].",
            ]
        ),
    )
    return mu_delta, summary


def build_head_projection_vectors(model, layers: Sequence[int], direction_unit: torch.Tensor) -> dict[int, torch.Tensor]:
    projection_vectors: dict[int, torch.Tensor] = {}
    d_head = int(model.config.hidden_size // model.config.num_attention_heads)
    direction_device = direction_unit.to(device=next(model.parameters()).device, dtype=torch.float32)
    for layer in layers:
        o_proj_weight = model.model.layers[layer].self_attn.o_proj.weight.detach().to(
            device=direction_device.device,
            dtype=torch.float32,
        )
        layer_vectors = []
        for head_idx in range(int(model.config.num_attention_heads)):
            start = head_idx * d_head
            end = start + d_head
            layer_vectors.append(o_proj_weight[:, start:end].transpose(0, 1) @ direction_device)
        projection_vectors[layer] = torch.stack(layer_vectors, dim=0).contiguous()
    return projection_vectors


def collect_upstream_scores_for_side(
    model,
    batch: BatchRecord,
    *,
    side: str,
    upstream_layers: Sequence[int],
    direction_unit: torch.Tensor,
    head_projection_vectors: dict[int, torch.Tensor],
) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor]]:
    if side not in {"clean", "corrupt"}:
        raise ValueError(f"Unsupported side: {side}")

    mlp_scores: dict[int, torch.Tensor] = {}
    head_scores: dict[int, torch.Tensor] = {}
    direction_gpu = direction_unit.to(device=next(model.parameters()).device, dtype=torch.float32)
    handles = []

    def make_mlp_hook(layer_idx: int):
        def hook_fn(_module, _inputs, output):
            last = output[:, -1, :].detach().to(dtype=torch.float32)
            mlp_scores[layer_idx] = torch.matmul(last, direction_gpu).detach().cpu()
            return output

        return hook_fn

    def make_head_pre_hook(layer_idx: int):
        projection = head_projection_vectors[layer_idx]

        def hook_fn(_module, inputs):
            last = inputs[0][:, -1, :].detach().to(dtype=torch.float32)
            last = last.view(last.shape[0], int(model.config.num_attention_heads), -1)
            head_scores[layer_idx] = torch.einsum("bhd,hd->bh", last, projection).detach().cpu()
            return inputs

        return hook_fn

    for layer_idx in upstream_layers:
        handles.append(model.model.layers[layer_idx].mlp.register_forward_hook(make_mlp_hook(layer_idx)))
        handles.append(
            model.model.layers[layer_idx].self_attn.o_proj.register_forward_pre_hook(
                make_head_pre_hook(layer_idx)
            )
        )
    try:
        if side == "clean":
            _ = forward_model(
                model,
                input_ids_cpu=batch.clean_input_ids_cpu,
                attention_mask_cpu=batch.clean_attention_mask_cpu,
                output_hidden_states=False,
            )
        else:
            _ = forward_model(
                model,
                input_ids_cpu=batch.corrupt_input_ids_cpu,
                attention_mask_cpu=batch.corrupt_attention_mask_cpu,
                output_hidden_states=False,
            )
    finally:
        for handle in handles:
            handle.remove()

    return mlp_scores, head_scores


def run_upstream_projection_experiment(
    model,
    pairs: Sequence[PairRecord],
    batches: Sequence[BatchRecord],
    baseline: BaselineBundle,
    *,
    best_layer: int,
    mu_delta: torch.Tensor,
    output_root: Path,
    device: str,
) -> dict[str, object]:
    ensure_dir(output_root)
    direction_unit = mu_delta / mu_delta.norm().clamp_min(1e-12)
    num_layers = int(model.config.num_hidden_layers)
    trajectory_per_sample_rows: list[dict[str, object]] = []
    trajectory_rows: list[dict[str, object]] = []

    projections = torch.einsum("nld,d->nl", baseline.clean_states - baseline.corrupt_states, direction_unit.float())
    clean_proj = torch.einsum("nld,d->nl", baseline.clean_states, direction_unit.float())
    corrupt_proj = torch.einsum("nld,d->nl", baseline.corrupt_states, direction_unit.float())
    for layer_idx in range(num_layers):
        layer_values = projections[:, layer_idx]
        for pair_idx, pair in enumerate(pairs):
            trajectory_per_sample_rows.append(
                {
                    "sample_id": pair.sample_id,
                    "layer": int(layer_idx),
                    "trajectory_projection": float(layer_values[pair_idx].item()),
                    "clean_projection": float(clean_proj[pair_idx, layer_idx].item()),
                    "corrupt_projection": float(corrupt_proj[pair_idx, layer_idx].item()),
                }
            )
        trajectory_rows.append(
            {
                "layer": int(layer_idx),
                "mean_projection": float(layer_values.mean().item()),
                "std_projection": float(layer_values.std(unbiased=False).item()),
                "min_projection": float(layer_values.min().item()),
                "max_projection": float(layer_values.max().item()),
                "mean_clean_projection": float(clean_proj[:, layer_idx].mean().item()),
                "mean_corrupt_projection": float(corrupt_proj[:, layer_idx].mean().item()),
            }
        )

    upstream_layers = list(range(best_layer))
    mlp_per_sample_rows: list[dict[str, object]] = []
    head_per_sample_rows: list[dict[str, object]] = []
    mlp_clean_sum = {layer: 0.0 for layer in upstream_layers}
    mlp_corrupt_sum = {layer: 0.0 for layer in upstream_layers}
    head_clean_sum = {
        (layer, head): 0.0
        for layer in upstream_layers
        for head in range(int(model.config.num_attention_heads))
    }
    head_corrupt_sum = {
        (layer, head): 0.0
        for layer in upstream_layers
        for head in range(int(model.config.num_attention_heads))
    }

    head_projection_vectors = build_head_projection_vectors(model, upstream_layers, direction_unit)
    progress = tqdm(batches, desc="Exp C upstream capture", dynamic_ncols=True)
    for batch in progress:
        clean_mlp_scores, clean_head_scores = collect_upstream_scores_for_side(
            model,
            batch,
            side="clean",
            upstream_layers=upstream_layers,
            direction_unit=direction_unit,
            head_projection_vectors=head_projection_vectors,
        )
        corrupt_mlp_scores, corrupt_head_scores = collect_upstream_scores_for_side(
            model,
            batch,
            side="corrupt",
            upstream_layers=upstream_layers,
            direction_unit=direction_unit,
            head_projection_vectors=head_projection_vectors,
        )

        for layer_idx in upstream_layers:
            clean_layer = clean_mlp_scores[layer_idx]
            corrupt_layer = corrupt_mlp_scores[layer_idx]
            mlp_clean_sum[layer_idx] += float(clean_layer.sum().item())
            mlp_corrupt_sum[layer_idx] += float(corrupt_layer.sum().item())
            for local_idx, pair_idx in enumerate(batch.indices):
                pair = pairs[pair_idx]
                mlp_per_sample_rows.append(
                    {
                        "sample_id": pair.sample_id,
                        "layer": int(layer_idx),
                        "clean_projection": float(clean_layer[local_idx].item()),
                        "corrupt_projection": float(corrupt_layer[local_idx].item()),
                        "mlp_delta": float((clean_layer[local_idx] - corrupt_layer[local_idx]).item()),
                    }
                )

            clean_heads = clean_head_scores[layer_idx]
            corrupt_heads = corrupt_head_scores[layer_idx]
            for head_idx in range(int(model.config.num_attention_heads)):
                head_key = (layer_idx, head_idx)
                clean_head_col = clean_heads[:, head_idx]
                corrupt_head_col = corrupt_heads[:, head_idx]
                head_clean_sum[head_key] += float(clean_head_col.sum().item())
                head_corrupt_sum[head_key] += float(corrupt_head_col.sum().item())
                for local_idx, pair_idx in enumerate(batch.indices):
                    pair = pairs[pair_idx]
                    head_per_sample_rows.append(
                        {
                            "sample_id": pair.sample_id,
                            "layer": int(layer_idx),
                            "head": int(head_idx),
                            "clean_projection": float(clean_head_col[local_idx].item()),
                            "corrupt_projection": float(corrupt_head_col[local_idx].item()),
                            "head_delta": float((clean_head_col[local_idx] - corrupt_head_col[local_idx]).item()),
                        }
                    )

        del clean_mlp_scores, clean_head_scores, corrupt_mlp_scores, corrupt_head_scores
        clear_cuda()
        progress.set_postfix(max_len=batch.max_len)

    mlp_rows: list[dict[str, object]] = []
    for layer_idx in upstream_layers:
        clean_mean = mlp_clean_sum[layer_idx] / max(len(pairs), 1)
        corrupt_mean = mlp_corrupt_sum[layer_idx] / max(len(pairs), 1)
        mlp_rows.append(
            {
                "layer": int(layer_idx),
                "mean_clean_projection": float(clean_mean),
                "mean_corrupt_projection": float(corrupt_mean),
                "mlp_delta": float(clean_mean - corrupt_mean),
                "abs_mlp_delta": float(abs(clean_mean - corrupt_mean)),
            }
        )

    head_rows: list[dict[str, object]] = []
    for layer_idx in upstream_layers:
        for head_idx in range(int(model.config.num_attention_heads)):
            clean_mean = head_clean_sum[(layer_idx, head_idx)] / max(len(pairs), 1)
            corrupt_mean = head_corrupt_sum[(layer_idx, head_idx)] / max(len(pairs), 1)
            delta = clean_mean - corrupt_mean
            head_rows.append(
                {
                    "layer": int(layer_idx),
                    "head": int(head_idx),
                    "mean_clean_projection": float(clean_mean),
                    "mean_corrupt_projection": float(corrupt_mean),
                    "head_delta": float(delta),
                    "abs_head_delta": float(abs(delta)),
                }
            )

    top_mlp_rows = sorted(mlp_rows, key=lambda row: float(row["mlp_delta"]), reverse=True)[:5]
    top_head_rows = sorted(head_rows, key=lambda row: float(row["head_delta"]), reverse=True)[:10]

    summary = {
        "best_layer": int(best_layer),
        "n_pairs": int(len(pairs)),
        "trajectory_final_mean_projection": float(trajectory_rows[-1]["mean_projection"]),
        "top_mlp": top_mlp_rows,
        "top_heads": top_head_rows,
        "observed_peak_memory_gib": peak_memory_gib(device),
    }
    write_csv(output_root / "trajectory_per_sample.csv", trajectory_per_sample_rows)
    write_csv(output_root / "trajectory_summary.csv", trajectory_rows)
    write_csv(output_root / "mlp_contribution_per_sample.csv", mlp_per_sample_rows)
    write_csv(output_root / "mlp_contribution.csv", mlp_rows)
    write_csv(output_root / "top5_mlp.csv", top_mlp_rows)
    write_csv(output_root / "head_contribution_per_sample.csv", head_per_sample_rows)
    write_csv(output_root / "head_contribution.csv", head_rows)
    write_csv(output_root / "top10_heads.csv", top_head_rows)
    write_json(output_root / "summary.json", summary)
    top_mlp_labels = [f"L{int(row['layer'])}" for row in top_mlp_rows]
    top_head_labels = [f"L{int(row['layer'])}H{int(row['head'])}" for row in top_head_rows]
    write_text(
        output_root / "summary.md",
        "\n".join(
            [
                "# Exp C Summary",
                "",
                f"- best_layer: L{best_layer}",
                f"- upstream_layers: 0..{max(best_layer - 1, -1)}",
                f"- trajectory_final_mean_projection: {float(trajectory_rows[-1]['mean_projection']):.6f}",
                f"- top_mlp: {top_mlp_labels}",
                f"- top_heads: {top_head_labels}",
                "",
                "MLP outputs are captured with forward hooks on model.model.layers[l].mlp.",
                "Attention contributions are captured from self_attn.o_proj forward-pre-hook inputs and projected through W_O analytically.",
            ]
        ),
    )
    return summary


def build_root_summary(
    *,
    args: argparse.Namespace,
    pairs: Sequence[PairRecord],
    config,
    vram_wait: dict[str, float],
    vram_estimate: dict[str, float],
    exp_a_summary: dict[str, object],
    exp_b_summary: dict[str, object],
    exp_c_summary: dict[str, object],
) -> dict[str, object]:
    return {
        "model_path": str(args.model_path),
        "dataset_root": str(args.dataset_root),
        "output_root": str(args.output_root),
        "tool_call_token": TOOL_CALL_TOKEN,
        "n_pairs": int(len(pairs)),
        "splits_present": sorted({pair.split for pair in pairs}),
        "languages_present": sorted({pair.language for pair in pairs}),
        "model_config": {
            "num_hidden_layers": int(config.num_hidden_layers),
            "hidden_size": int(config.hidden_size),
            "num_attention_heads": int(config.num_attention_heads),
            "num_key_value_heads": int(getattr(config, "num_key_value_heads", 0)),
            "intermediate_size": int(config.intermediate_size),
            "vocab_size": int(config.vocab_size),
        },
        "batch_size": int(args.batch_size),
        "dtype": str(args.dtype),
        "vram_wait": vram_wait,
        "vram_estimate": vram_estimate,
        "exp_a": exp_a_summary,
        "exp_b": exp_b_summary,
        "exp_c": exp_c_summary,
        "path_note": (
            "The Granite dataset root contains 724 clean/corrupt text files, but manifest.jsonl and "
            "canonical_pairs.jsonl agree on the intended 500-pair selected subset. This run strictly uses "
            "the 500 prompts referenced by manifest/canonical metadata and does not read the extra leftover files."
        ),
    }


def main() -> None:
    args = parse_args()
    dtype = dtype_from_name(args.dtype)
    ensure_dir(args.output_root)

    config = AutoConfig.from_pretrained(str(args.model_path), trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(str(args.model_path), trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    pairs = load_pairs(tokenizer, args.dataset_root, max_pairs=args.max_pairs)
    max_seq_len = max(pair.max_len for pair in pairs)
    vram_estimate = estimate_required_vram_gib(
        config=config,
        dtype=dtype,
        batch_size=args.batch_size,
        max_seq_len=max_seq_len,
        model_path=args.model_path,
    )
    required_gib = float(vram_estimate["estimated_peak_gib"] * 1.1)
    vram_wait = wait_for_vram(
        required_gib,
        poll_seconds=args.wait_poll_seconds,
        device_index=0,
    )
    model, tokenizer, tool_token_id = load_model_and_tokenizer(
        model_path=args.model_path,
        dtype=dtype,
        device=args.device,
    )
    batches = build_batches(tokenizer, pairs, batch_size=args.batch_size)

    run_metadata = {
        "model_path": str(args.model_path),
        "dataset_root": str(args.dataset_root),
        "batch_size": int(args.batch_size),
        "dtype": args.dtype,
        "device": args.device,
        "n_pairs": int(len(pairs)),
        "max_seq_len": int(max_seq_len),
        "tool_call_token": TOOL_CALL_TOKEN,
        "tool_call_token_id": int(tool_token_id),
        "tokenizer_padding_side": tokenizer.padding_side,
        "vram_wait": vram_wait,
        "vram_estimate": vram_estimate,
    }
    write_json(args.output_root / "run_metadata.json", run_metadata)

    reset_peak_memory(args.device)
    baseline = collect_baseline(
        model,
        tokenizer,
        pairs,
        batches,
        tool_token_id=tool_token_id,
        num_layers=int(model.config.num_hidden_layers),
        device=args.device,
    )

    reset_peak_memory(args.device)
    exp_a_root = args.output_root / "exp_a_state_patch"
    best_layer, _patch_rows, exp_a_summary = run_state_patch_experiment(
        model,
        tokenizer,
        pairs,
        batches,
        baseline,
        tool_token_id=tool_token_id,
        output_root=exp_a_root,
        device=args.device,
    )

    reset_peak_memory(args.device)
    exp_b_root = args.output_root / "exp_b_tool_call_vector"
    mu_delta, exp_b_summary = run_vector_experiment(
        model,
        tokenizer,
        pairs,
        batches,
        baseline,
        best_layer=best_layer,
        tool_token_id=tool_token_id,
        output_root=exp_b_root,
        device=args.device,
    )

    reset_peak_memory(args.device)
    exp_c_root = args.output_root / "exp_c_upstream_projection"
    exp_c_summary = run_upstream_projection_experiment(
        model,
        pairs,
        batches,
        baseline,
        best_layer=best_layer,
        mu_delta=mu_delta,
        output_root=exp_c_root,
        device=args.device,
    )

    root_summary = build_root_summary(
        args=args,
        pairs=pairs,
        config=config,
        vram_wait=vram_wait,
        vram_estimate=vram_estimate,
        exp_a_summary=exp_a_summary,
        exp_b_summary=exp_b_summary,
        exp_c_summary=exp_c_summary,
    )
    write_json(args.output_root / "summary.json", root_summary)
    root_top_mlp_labels = [f"L{int(row['layer'])}" for row in exp_c_summary["top_mlp"]]
    root_top_head_labels = [
        f"L{int(row['layer'])}H{int(row['head'])}" for row in exp_c_summary["top_heads"]
    ]
    write_text(
        args.output_root / "summary.md",
        "\n".join(
            [
                "# Granite Mechanism Generalization",
                "",
                f"- n_pairs: {len(pairs)}",
                f"- L*: L{int(exp_a_summary['best_layer'])}",
                f"- mu_delta_norm: {float(exp_b_summary['mu_delta_norm']):.6f}",
                f"- suff: {float(exp_b_summary['suff']):.6f}",
                f"- necc: {float(exp_b_summary['necc']):.6f}",
                f"- top_mlp: {root_top_mlp_labels}",
                f"- top_heads: {root_top_head_labels}",
                "",
                root_summary["path_note"],
            ]
        ),
    )


if __name__ == "__main__":
    main()
