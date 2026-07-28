#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import math
import subprocess
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from tqdm.auto import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
import sys

SRC_ROOT = PROJECT_ROOT / "src"
for candidate in (SRC_ROOT, SCRIPT_DIR):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from artifact_paths import MISTRAL_3P2_24B_PATH  # noqa: E402

from dataset_utils import (
    DEFAULT_DATASET_ROOT,
    DEFAULT_METADATA_ROOT,
    build_pair_batches,
    collate_pair_batch,
    load_pairs,
    read_json,
    write_csv,
    write_json,
    write_text,
)
from hf_patch_utils import (
    TOOL_CALL_TEXT,
    get_d_model,
    get_head_dim,
    get_layer,
    get_last_token_positions,
    get_mlp,
    get_num_heads,
    get_num_layers,
    get_o_proj,
    load_model,
    load_tokenizer,
    make_capture_hook,
    make_last_token_add_hook,
    make_last_token_replace_hook,
    make_last_token_subtract_hook,
    make_pre_capture_hook,
    register_hooks,
    resolve_tool_token_id,
    to_token_text,
    tool_stats,
)


MODEL_PATH = MISTRAL_3P2_24B_PATH
OUTPUT_ROOT = PROJECT_ROOT / "results" / "section6_generalization" / "mistral_3p2_24b" / "mechanism_generalization_rerun"
TOKENIZER_VERIFICATION_PATH = (
    PROJECT_ROOT / "results" / "section6_generalization" / "mistral_3p2_24b" / "converted_dataset" / "tokenizer_verification.json"
)
COARSE_FRACTIONS = (0.45, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85)
ALPHAS = (0.5, 1.0, 1.5, 2.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mistral mechanism generalization runner with HF-native hooks only.")
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--metadata-root", type=Path, default=DEFAULT_METADATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--exp-a-split", type=str, default="all")
    parser.add_argument("--exp-b-split", type=str, default="all")
    parser.add_argument("--exp-c-split", type=str, default="all")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-pairs", type=int, default=0)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--device-map", type=str, default="auto")
    parser.add_argument("--wait-poll-seconds", type=int, default=30)
    parser.add_argument("--min-free-vram-gib", type=float, default=72.0)
    return parser.parse_args()


def dtype_from_name(name: str) -> torch.dtype:
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    if name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def dtype_num_bytes(dtype: torch.dtype) -> int:
    if dtype in {torch.bfloat16, torch.float16}:
        return 2
    if dtype == torch.float32:
        return 4
    raise ValueError(f"Unsupported dtype for byte-size estimate: {dtype}")


def gib(bytes_value: float) -> float:
    return float(bytes_value) / float(1024**3)


def clear_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


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
    candidates = [path for path in model_path.glob("*.safetensors") if path.name != "consolidated.safetensors"]
    if not candidates:
        candidates = list(model_path.glob("*.safetensors")) + list(model_path.glob("*.bin"))
    if not candidates:
        return 0
    return int(sum(path.stat().st_size for path in candidates))


def estimate_required_vram_gib(
    *,
    n_layers: int,
    hidden_size: int,
    vocab_size: int,
    dtype: torch.dtype,
    batch_size: int,
    max_seq_len: int,
    model_path: Path,
) -> dict[str, float]:
    dtype_bytes = dtype_num_bytes(dtype)
    weight_bytes = model_weight_bytes_on_disk(model_path)
    hidden_state_bytes = (n_layers + 1) * batch_size * max_seq_len * hidden_size * dtype_bytes
    logits_bytes = batch_size * max_seq_len * vocab_size * dtype_bytes
    runtime_overhead_bytes = 8 * 1024**3
    estimated_peak_bytes = weight_bytes + hidden_state_bytes + logits_bytes + runtime_overhead_bytes
    return {
        "weight_gib_on_disk": gib(weight_bytes),
        "hidden_state_gib": gib(hidden_state_bytes),
        "logits_gib": gib(logits_bytes),
        "runtime_overhead_gib": gib(runtime_overhead_bytes),
        "estimated_peak_gib": gib(estimated_peak_bytes),
    }


def build_model_and_tokenizer(args: argparse.Namespace):
    tokenizer = load_tokenizer(args.model_path)
    tool_token_id, tool_token_info = resolve_tool_token_id(tokenizer, TOOL_CALL_TEXT)
    if TOKENIZER_VERIFICATION_PATH.exists():
        verification = read_json(TOKENIZER_VERIFICATION_PATH)
        tool_token_info["tokenizer_verification_path"] = str(TOKENIZER_VERIFICATION_PATH.resolve())
        tool_token_info["tokenizer_verification_note"] = verification.get("note")
    dtype = dtype_from_name(args.dtype)
    model = load_model(
        args.model_path,
        dtype=dtype,
        device_map=args.device_map,
    )
    return model, tokenizer, tool_token_id, tool_token_info, dtype


def infer_device(model) -> torch.device:
    return next(model.parameters()).device


def to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def batch_logits(model, input_ids, attention_mask):
    with torch.inference_mode():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            output_hidden_states=False,
            return_dict=True,
        )
    return outputs.logits


def top1_rows(tokenizer, logits: torch.Tensor, tool_token_id: int, positions: torch.Tensor):
    tool_logit, tool_prob, top1_ids = tool_stats(logits, tool_token_id, positions)
    return tool_logit, tool_prob, top1_ids, [to_token_text(tokenizer, int(item)) for item in top1_ids]


def run_behavior_sanity_check(model, tokenizer, tool_token_id: int, pairs, batch_size: int) -> dict[str, Any]:
    batches = build_pair_batches(pairs, batch_size=batch_size)
    device = infer_device(model)
    clean_tool_top1 = 0
    corrupt_tool_top1 = 0
    clean_logit_sum = 0.0
    corrupt_logit_sum = 0.0
    n = 0
    for batch in batches:
        batch_inputs = collate_pair_batch(tokenizer, batch)
        clean = to_device({"input_ids": batch_inputs["clean_input_ids"], "attention_mask": batch_inputs["clean_attention_mask"]}, device)
        corrupt = to_device({"input_ids": batch_inputs["corrupt_input_ids"], "attention_mask": batch_inputs["corrupt_attention_mask"]}, device)
        clean_positions = get_last_token_positions(clean["attention_mask"]).detach().cpu()
        corrupt_positions = get_last_token_positions(corrupt["attention_mask"]).detach().cpu()
        clean_logits = batch_logits(model, clean["input_ids"], clean["attention_mask"])
        corrupt_logits = batch_logits(model, corrupt["input_ids"], corrupt["attention_mask"])
        clean_tool_logit, _, clean_top1 = tool_stats(clean_logits, tool_token_id, clean_positions)
        corrupt_tool_logit, _, corrupt_top1 = tool_stats(corrupt_logits, tool_token_id, corrupt_positions)
        clean_tool_top1 += int((clean_top1 == tool_token_id).sum().item())
        corrupt_tool_top1 += int((corrupt_top1 == tool_token_id).sum().item())
        clean_logit_sum += float(clean_tool_logit.sum().item())
        corrupt_logit_sum += float(corrupt_tool_logit.sum().item())
        n += int(clean_top1.shape[0])
        clear_cuda()
    return {
        "n_pairs": n,
        "clean_tool_call_top1_rate": float(clean_tool_top1 / max(n, 1)),
        "corrupt_tool_call_top1_rate": float(corrupt_tool_top1 / max(n, 1)),
        "clean_mean_tool_call_logit": float(clean_logit_sum / max(n, 1)),
        "corrupt_mean_tool_call_logit": float(corrupt_logit_sum / max(n, 1)),
    }


def choose_best_layer(rows: list[dict[str, object]]) -> int:
    phase_rows = [row for row in rows if row["phase"] == "final"]
    max_rate = max(float(row["tool_call_top1_rate"]) for row in phase_rows)
    candidates = [row for row in phase_rows if float(row["tool_call_top1_rate"]) >= max_rate - 1e-6]
    return int(sorted(candidates, key=lambda row: int(row["layer"]))[0]["layer"])


def build_patch_layers(n_layers: int) -> list[int]:
    coarse = []
    for frac in COARSE_FRACTIONS:
        layer = min(max(int(round(frac * n_layers)) - 1, 0), n_layers - 1)
        if layer not in coarse:
            coarse.append(layer)
    return coarse


def plot_exp_a(rows: list[dict[str, object]], path: Path) -> None:
    if not rows:
        return
    final_rows = sorted([row for row in rows if row["phase"] == "final"], key=lambda row: int(row["layer"]))
    layers = [int(row["layer"]) for row in final_rows]
    top1 = [float(row["tool_call_top1_rate"]) for row in final_rows]
    flip = [float(row["strict_flip_rate"]) for row in final_rows]
    logit = [float(row["mean_tool_call_logit"]) for row in final_rows]
    fig, ax1 = plt.subplots(figsize=(7.2, 4.6))
    ax1.plot(layers, top1, marker="o", label="tool-call top1")
    ax1.plot(layers, flip, marker="s", label="strict flip")
    ax1.set_xlabel("Layer")
    ax1.set_ylabel("Rate")
    ax1.set_ylim(0.0, 1.05)
    ax1.grid(alpha=0.25)
    ax2 = ax1.twinx()
    ax2.plot(layers, logit, marker="^", linestyle="--", color="tab:red", label="mean tool logit")
    ax2.set_ylabel("Mean tool logit")
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="best")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def summarize_exp_a_rows(rows: list[dict[str, object]], best_layer: int, split: str, tool_token_info: dict[str, Any]) -> str:
    best_row = next(row for row in rows if int(row["layer"]) == best_layer and row["phase"] == "final")
    correction_lines = []
    if tool_token_info.get("tool_token_encode_len", 1) != 1:
        correction_lines.append(
            f"- correction: local tokenizer `encode('[TOOL_CALLS]')` splits into `{tool_token_info['tool_token_encode_len']}` pieces, so behavior metrics use `convert_tokens_to_ids('[TOOL_CALLS]')={tool_token_info['tool_token_id_via_convert']}`."
        )
    correction_lines.append(
        "- correction: the actual text-layer hook path resolves to `model.language_model.layers[...]` on this local checkpoint."
    )
    return "\n".join(
        [
            "# Exp A Summary",
            "",
            f"- split: `{split}`",
            f"- best layer: `L{best_layer}`",
            f"- best tool-call top1: `{float(best_row['tool_call_top1_rate']):.4f}`",
            f"- best strict flip: `{float(best_row['strict_flip_rate']):.4f}`",
            f"- mean tool logit: `{float(best_row['mean_tool_call_logit']):.4f}`",
            f"- mean tool prob: `{float(best_row['mean_tool_call_prob']):.4f}`",
            "",
            *correction_lines,
        ]
    ) + "\n"


def summarize_main(best_layer: int, output_root: Path, tool_token_info: dict[str, Any]) -> str:
    correction_lines = []
    if tool_token_info.get("tool_token_encode_len", 1) != 1:
        correction_lines.append(
            f"- tokenizer correction: `[TOOL_CALLS]` is treated via special token id `{tool_token_info['tool_token_id_via_convert']}`, not raw `encode(...)`."
        )
    correction_lines.append(
        "- path correction: layer hooks resolve against `model.language_model.layers[...]` in the local Mistral checkpoint."
    )
    return "\n".join(
        [
            "# Mistral Mechanism Generalization",
            "",
            f"- best layer: `L{best_layer}`",
            f"- exp A: `{output_root / 'exp_a_state_patch'}`",
            f"- exp B: `{output_root / 'exp_b_tool_call_vector'}`",
            f"- exp C: `{output_root / 'exp_c_upstream_projection'}`",
            *correction_lines,
        ]
    ) + "\n"


def run_exp_a(model, tokenizer, tool_token_id: int, pairs, output_root: Path, batch_size: int, tool_token_info: dict[str, Any], args: argparse.Namespace):
    output_root.mkdir(parents=True, exist_ok=True)
    batches = build_pair_batches(pairs, batch_size=batch_size)
    n_layers = get_num_layers(model)
    coarse_layers = build_patch_layers(n_layers)
    rows: list[dict[str, object]] = []
    accum = {
        layer: {"count": 0, "tool_top1": 0, "strict_flip": 0, "logit_sum": 0.0, "prob_sum": 0.0}
        for layer in coarse_layers
    }
    baseline_clean_tool_top1 = 0
    baseline_corrupt_tool_top1 = 0
    baseline_count = 0

    for batch in tqdm(batches, desc="Exp A coarse", dynamic_ncols=True):
        batch_inputs = collate_pair_batch(tokenizer, batch)
        device = infer_device(model)
        clean = to_device({"input_ids": batch_inputs["clean_input_ids"], "attention_mask": batch_inputs["clean_attention_mask"]}, device)
        corrupt = to_device({"input_ids": batch_inputs["corrupt_input_ids"], "attention_mask": batch_inputs["corrupt_attention_mask"]}, device)
        clean_positions = get_last_token_positions(clean["attention_mask"]).detach().cpu()
        corrupt_positions = get_last_token_positions(corrupt["attention_mask"]).detach().cpu()

        clean_store: dict[int, torch.Tensor] = {}
        clean_hooks = [(get_layer(model, layer_idx), make_capture_hook(clean_store, layer_idx, clean_positions)) for layer_idx in coarse_layers]
        with register_hooks(clean_hooks):
            clean_logits = batch_logits(model, clean["input_ids"], clean["attention_mask"])
        corrupt_logits = batch_logits(model, corrupt["input_ids"], corrupt["attention_mask"])
        clean_tool_logit, clean_tool_prob, clean_top1 = tool_stats(clean_logits, tool_token_id, clean_positions)
        corrupt_tool_logit, corrupt_tool_prob, corrupt_top1 = tool_stats(corrupt_logits, tool_token_id, corrupt_positions)
        baseline_clean_tool_top1 += int((clean_top1 == tool_token_id).sum().item())
        baseline_corrupt_tool_top1 += int((corrupt_top1 == tool_token_id).sum().item())
        baseline_count += int(clean_top1.shape[0])

        for layer in coarse_layers:
            patch_source = clean_store[layer].detach().cpu()
            patch_hooks = [(get_layer(model, layer), make_last_token_replace_hook(patch_source, corrupt_positions))]
            with register_hooks(patch_hooks):
                patched_logits = batch_logits(model, corrupt["input_ids"], corrupt["attention_mask"])
            patched_tool_logit, patched_tool_prob, patched_top1 = tool_stats(patched_logits, tool_token_id, corrupt_positions)
            bucket = accum[layer]
            bucket["count"] += int(patched_top1.shape[0])
            bucket["tool_top1"] += int((patched_top1 == tool_token_id).sum().item())
            bucket["strict_flip"] += int(((corrupt_top1 != tool_token_id) & (patched_top1 == tool_token_id)).sum().item())
            bucket["logit_sum"] += float(patched_tool_logit.sum().item())
            bucket["prob_sum"] += float(patched_tool_prob.sum().item())
            del patched_logits, patched_tool_logit, patched_tool_prob, patched_top1
            clear_cuda()

        del clean_store, clean_logits, corrupt_logits, clean_tool_logit, clean_tool_prob, clean_top1, corrupt_tool_logit, corrupt_tool_prob, corrupt_top1
        clear_cuda()

    for layer in coarse_layers:
        bucket = accum[layer]
        count = max(int(bucket["count"]), 1)
        rows.append(
            {
                "phase": "coarse",
                "layer": layer,
                "n": count,
                "tool_call_top1_rate": float(bucket["tool_top1"] / count),
                "strict_flip_rate": float(bucket["strict_flip"] / count),
                "mean_tool_call_logit": float(bucket["logit_sum"] / count),
                "mean_tool_call_prob": float(bucket["prob_sum"] / count),
                "baseline_clean_tool_call_top1_rate": float(baseline_clean_tool_top1 / max(baseline_count, 1)),
                "baseline_corrupt_tool_call_top1_rate": float(baseline_corrupt_tool_top1 / max(baseline_count, 1)),
            }
        )

    best_coarse = max(rows, key=lambda row: float(row["tool_call_top1_rate"]))
    best_center = int(best_coarse["layer"])
    final_layers = sorted({max(best_center - 2, 0), max(best_center - 1, 0), best_center, min(best_center + 1, n_layers - 1), min(best_center + 2, n_layers - 1)})
    final_accum = {
        layer: {"count": 0, "tool_top1": 0, "strict_flip": 0, "logit_sum": 0.0, "prob_sum": 0.0}
        for layer in final_layers
    }

    for batch in tqdm(batches, desc="Exp A final", dynamic_ncols=True):
        batch_inputs = collate_pair_batch(tokenizer, batch)
        device = infer_device(model)
        clean = to_device({"input_ids": batch_inputs["clean_input_ids"], "attention_mask": batch_inputs["clean_attention_mask"]}, device)
        corrupt = to_device({"input_ids": batch_inputs["corrupt_input_ids"], "attention_mask": batch_inputs["corrupt_attention_mask"]}, device)
        clean_positions = get_last_token_positions(clean["attention_mask"]).detach().cpu()
        corrupt_positions = get_last_token_positions(corrupt["attention_mask"]).detach().cpu()
        clean_store: dict[int, torch.Tensor] = {}
        clean_hooks = [(get_layer(model, layer_idx), make_capture_hook(clean_store, layer_idx, clean_positions)) for layer_idx in final_layers]
        with register_hooks(clean_hooks):
            clean_logits = batch_logits(model, clean["input_ids"], clean["attention_mask"])
        corrupt_logits = batch_logits(model, corrupt["input_ids"], corrupt["attention_mask"])
        _, _, clean_top1 = tool_stats(clean_logits, tool_token_id, clean_positions)
        _, _, corrupt_top1 = tool_stats(corrupt_logits, tool_token_id, corrupt_positions)
        for layer in final_layers:
            patch_source = clean_store[layer].detach().cpu()
            patch_hooks = [(get_layer(model, layer), make_last_token_replace_hook(patch_source, corrupt_positions))]
            with register_hooks(patch_hooks):
                patched_logits = batch_logits(model, corrupt["input_ids"], corrupt["attention_mask"])
            patched_tool_logit, patched_tool_prob, patched_top1 = tool_stats(patched_logits, tool_token_id, corrupt_positions)
            bucket = final_accum[layer]
            bucket["count"] += int(patched_top1.shape[0])
            bucket["tool_top1"] += int((patched_top1 == tool_token_id).sum().item())
            bucket["strict_flip"] += int(((corrupt_top1 != tool_token_id) & (patched_top1 == tool_token_id)).sum().item())
            bucket["logit_sum"] += float(patched_tool_logit.sum().item())
            bucket["prob_sum"] += float(patched_tool_prob.sum().item())
            del patched_logits, patched_tool_logit, patched_tool_prob, patched_top1
            clear_cuda()
        del clean_store, clean_logits, corrupt_logits, clean_top1, corrupt_top1
        clear_cuda()

    for layer in final_layers:
        bucket = final_accum[layer]
        count = max(int(bucket["count"]), 1)
        rows.append(
            {
                "phase": "final",
                "layer": layer,
                "n": count,
                "tool_call_top1_rate": float(bucket["tool_top1"] / count),
                "strict_flip_rate": float(bucket["strict_flip"] / count),
                "mean_tool_call_logit": float(bucket["logit_sum"] / count),
                "mean_tool_call_prob": float(bucket["prob_sum"] / count),
                "baseline_clean_tool_call_top1_rate": float(baseline_clean_tool_top1 / max(baseline_count, 1)),
                "baseline_corrupt_tool_call_top1_rate": float(baseline_corrupt_tool_top1 / max(baseline_count, 1)),
            }
        )

    rows.sort(key=lambda row: (row["phase"], int(row["layer"])))
    best_layer = choose_best_layer(rows)
    write_csv(output_root / "patch_sweep.csv", rows)
    write_json(
        output_root / "metadata.json",
        {
            "model_path": str(args.model_path.resolve()),
            "dataset_root": str(args.dataset_root.resolve()),
            "split": args.exp_a_split,
            "batch_size": batch_size,
            "n_pairs": len(pairs),
            "n_layers": n_layers,
            "coarse_layers": coarse_layers,
            "final_layers": final_layers,
            "best_layer": best_layer,
            "tool_token_info": tool_token_info,
        },
    )
    write_text(output_root / "summary.md", summarize_exp_a_rows(rows, best_layer, args.exp_a_split, tool_token_info))
    plot_exp_a(rows, output_root / "plot_patch_sweep.pdf")
    return best_layer, rows


def run_exp_b(model, tokenizer, tool_token_id: int, pairs, layer_idx: int, output_root: Path, batch_size: int, tool_token_info: dict[str, Any], args: argparse.Namespace):
    output_root.mkdir(parents=True, exist_ok=True)
    batches = build_pair_batches(pairs, batch_size=batch_size)
    device = infer_device(model)
    mu_sum = None
    n_total = 0
    baseline_rows: dict[str, dict[str, object]] = {}
    per_sample_rows: list[dict[str, object]] = []
    per_sample_map: dict[str, dict[str, object]] = {}
    alpha_add_rows: list[dict[str, object]] = []
    alpha_sub_rows: list[dict[str, object]] = []

    for batch in tqdm(batches, desc="Exp B baseline", dynamic_ncols=True):
        batch_inputs = collate_pair_batch(tokenizer, batch)
        clean = to_device({"input_ids": batch_inputs["clean_input_ids"], "attention_mask": batch_inputs["clean_attention_mask"]}, device)
        corrupt = to_device({"input_ids": batch_inputs["corrupt_input_ids"], "attention_mask": batch_inputs["corrupt_attention_mask"]}, device)
        clean_positions = get_last_token_positions(clean["attention_mask"]).detach().cpu()
        corrupt_positions = get_last_token_positions(corrupt["attention_mask"]).detach().cpu()
        clean_store: dict[int, torch.Tensor] = {}
        corrupt_store: dict[int, torch.Tensor] = {}
        clean_hooks = [(get_layer(model, layer_idx), make_capture_hook(clean_store, layer_idx, clean_positions))]
        corrupt_hooks = [(get_layer(model, layer_idx), make_capture_hook(corrupt_store, layer_idx, corrupt_positions))]
        with register_hooks(clean_hooks):
            clean_logits = batch_logits(model, clean["input_ids"], clean["attention_mask"])
        with register_hooks(corrupt_hooks):
            corrupt_logits = batch_logits(model, corrupt["input_ids"], corrupt["attention_mask"])
        clean_tool_logit, clean_tool_prob, clean_top1, clean_text = top1_rows(tokenizer, clean_logits, tool_token_id, clean_positions)
        corrupt_tool_logit, corrupt_tool_prob, corrupt_top1, corrupt_text = top1_rows(tokenizer, corrupt_logits, tool_token_id, corrupt_positions)
        clean_state = clean_store[layer_idx].detach().float().cpu()
        corrupt_state = corrupt_store[layer_idx].detach().float().cpu()
        diff = clean_state - corrupt_state
        mu_sum = diff.sum(dim=0) if mu_sum is None else mu_sum + diff.sum(dim=0)
        n_total += int(diff.shape[0])
        for local_idx, pair in enumerate(batch.examples):
            baseline_rows[pair.sample_id] = {
                "pair_id": pair.pair_id,
                "sample_id": pair.sample_id,
                "split": pair.split,
                "language": pair.language,
                "dataset_name": pair.dataset_name,
                "clean_candidate": pair.clean_candidate,
                "corrupt_candidate": pair.corrupt_candidate,
                "clean_tool_call_logit": float(clean_tool_logit[local_idx].item()),
                "corrupt_tool_call_logit": float(corrupt_tool_logit[local_idx].item()),
                "clean_tool_call_prob": float(clean_tool_prob[local_idx].item()),
                "corrupt_tool_call_prob": float(corrupt_tool_prob[local_idx].item()),
                "clean_top1_token_id": int(clean_top1[local_idx].item()),
                "corrupt_top1_token_id": int(corrupt_top1[local_idx].item()),
                "clean_top1_token_text": clean_text[local_idx],
                "corrupt_top1_token_text": corrupt_text[local_idx],
                "clean_is_tool_call_top1": bool(int(clean_top1[local_idx].item()) == tool_token_id),
                "corrupt_is_tool_call_top1": bool(int(corrupt_top1[local_idx].item()) == tool_token_id),
            }
        del clean_logits, corrupt_logits, clean_state, corrupt_state, diff
        clear_cuda()

    mu_delta = (mu_sum / max(n_total, 1)).to(torch.float32)
    torch.save(mu_delta, output_root / "mu_delta.pt")

    def run_alpha(alpha: float, direction: str):
        direction_rows: list[dict[str, object]] = []
        accum = {
            "count": 0,
            "tool_top1": 0,
            "strict_flip": 0,
            "strict_drop": 0,
            "logit_sum": 0.0,
            "prob_sum": 0.0,
        }
        for batch in tqdm(batches, desc=f"Exp B {direction} alpha={alpha}", dynamic_ncols=True):
            batch_inputs = collate_pair_batch(tokenizer, batch)
            clean = to_device({"input_ids": batch_inputs["clean_input_ids"], "attention_mask": batch_inputs["clean_attention_mask"]}, device)
            corrupt = to_device({"input_ids": batch_inputs["corrupt_input_ids"], "attention_mask": batch_inputs["corrupt_attention_mask"]}, device)
            clean_positions = get_last_token_positions(clean["attention_mask"]).detach().cpu()
            corrupt_positions = get_last_token_positions(corrupt["attention_mask"]).detach().cpu()
            patch_source = mu_delta * float(alpha)
            if direction == "add":
                hooks = [(get_layer(model, layer_idx), make_last_token_add_hook(patch_source, corrupt_positions))]
                with register_hooks(hooks):
                    patched_logits = batch_logits(model, corrupt["input_ids"], corrupt["attention_mask"])
                _, _, patched_top1, patched_text = top1_rows(tokenizer, patched_logits, tool_token_id, corrupt_positions)
                patched_tool_logit, patched_tool_prob, _ = tool_stats(patched_logits, tool_token_id, corrupt_positions)
                for local_idx, pair in enumerate(batch.examples):
                    baseline = baseline_rows[pair.sample_id]
                    row = {
                        **baseline,
                        "alpha": alpha,
                        "direction": "add",
                        "vector_plus_tool_call_logit": float(patched_tool_logit[local_idx].item()),
                        "vector_plus_tool_call_prob": float(patched_tool_prob[local_idx].item()),
                        "vector_plus_top1_token_id": int(patched_top1[local_idx].item()),
                        "vector_plus_top1_token_text": patched_text[local_idx],
                        "vector_plus_is_tool_call_top1": bool(int(patched_top1[local_idx].item()) == tool_token_id),
                    }
                    direction_rows.append(row)
                    if alpha == 1.0:
                        merged = {
                            **row,
                            "vector_minus_tool_call_logit": None,
                            "vector_minus_tool_call_prob": None,
                            "vector_minus_top1_token_id": None,
                            "vector_minus_top1_token_text": None,
                            "vector_minus_is_tool_call_top1": None,
                        }
                        per_sample_rows.append(merged)
                        per_sample_map[pair.sample_id] = merged
                accum["count"] += int(patched_top1.shape[0])
                accum["tool_top1"] += int((patched_top1 == tool_token_id).sum().item())
                accum["strict_flip"] += int(
                    sum(
                        1
                        for local_idx, pair in enumerate(batch.examples)
                        if (not baseline_rows[pair.sample_id]["corrupt_is_tool_call_top1"]) and int(patched_top1[local_idx].item()) == tool_token_id
                    )
                )
            else:
                hooks = [(get_layer(model, layer_idx), make_last_token_subtract_hook(patch_source, clean_positions))]
                with register_hooks(hooks):
                    patched_logits = batch_logits(model, clean["input_ids"], clean["attention_mask"])
                _, _, patched_top1, patched_text = top1_rows(tokenizer, patched_logits, tool_token_id, clean_positions)
                patched_tool_logit, patched_tool_prob, _ = tool_stats(patched_logits, tool_token_id, clean_positions)
                for local_idx, pair in enumerate(batch.examples):
                    baseline = baseline_rows[pair.sample_id]
                    row = {
                        **baseline,
                        "alpha": alpha,
                        "direction": "subtract",
                        "vector_minus_tool_call_logit": float(patched_tool_logit[local_idx].item()),
                        "vector_minus_tool_call_prob": float(patched_tool_prob[local_idx].item()),
                        "vector_minus_top1_token_id": int(patched_top1[local_idx].item()),
                        "vector_minus_top1_token_text": patched_text[local_idx],
                        "vector_minus_is_tool_call_top1": bool(int(patched_top1[local_idx].item()) == tool_token_id),
                    }
                    direction_rows.append(row)
                    if alpha == 1.0 and pair.sample_id in per_sample_map:
                        per_sample_map[pair.sample_id].update(
                            {
                                "vector_minus_tool_call_logit": float(patched_tool_logit[local_idx].item()),
                                "vector_minus_tool_call_prob": float(patched_tool_prob[local_idx].item()),
                                "vector_minus_top1_token_id": int(patched_top1[local_idx].item()),
                                "vector_minus_top1_token_text": patched_text[local_idx],
                                "vector_minus_is_tool_call_top1": bool(int(patched_top1[local_idx].item()) == tool_token_id),
                            }
                        )
                accum["count"] += int(patched_top1.shape[0])
                accum["tool_top1"] += int((patched_top1 == tool_token_id).sum().item())
                accum["strict_drop"] += int(
                    sum(
                        1
                        for local_idx, pair in enumerate(batch.examples)
                        if baseline_rows[pair.sample_id]["clean_is_tool_call_top1"] and int(patched_top1[local_idx].item()) != tool_token_id
                    )
                )
            accum["logit_sum"] += float(patched_tool_logit.sum().item())
            accum["prob_sum"] += float(patched_tool_prob.sum().item())
            del patched_logits, patched_tool_logit, patched_tool_prob, patched_top1
            clear_cuda()
        rate_row = {
            "alpha": alpha,
            "direction": direction,
            "n": accum["count"],
            "tool_call_top1_rate": float(accum["tool_top1"] / max(accum["count"], 1)),
            "mean_tool_call_logit": float(accum["logit_sum"] / max(accum["count"], 1)),
            "mean_tool_call_prob": float(accum["prob_sum"] / max(accum["count"], 1)),
        }
        if direction == "add":
            rate_row["strict_flip_rate"] = float(accum["strict_flip"] / max(len(pairs), 1))
        else:
            rate_row["strict_drop_rate"] = float(accum["strict_drop"] / max(len(pairs), 1))
        return direction_rows, rate_row

    add_summary_rows = []
    sub_summary_rows = []
    alpha1_add = None
    alpha1_sub = None

    for alpha in ALPHAS:
        add_rows, add_rate = run_alpha(alpha, "add")
        sub_rows, sub_rate = run_alpha(alpha, "subtract")
        alpha_add_rows.extend(add_rows)
        alpha_sub_rows.extend(sub_rows)
        add_subset = [row for row in add_rows]
        sub_subset = [row for row in sub_rows]
        clean_mean = sum(float(v["clean_tool_call_logit"]) for v in baseline_rows.values()) / max(len(baseline_rows), 1)
        corrupt_mean = sum(float(v["corrupt_tool_call_logit"]) for v in baseline_rows.values()) / max(len(baseline_rows), 1)
        add_mean = sum(float(row["vector_plus_tool_call_logit"]) for row in add_subset) / max(len(add_subset), 1)
        sub_mean = sum(float(row["vector_minus_tool_call_logit"]) for row in sub_subset) / max(len(sub_subset), 1)
        add_summary = {
            **add_rate,
            "suff": float((add_mean - corrupt_mean) / max(clean_mean - corrupt_mean, 1e-12)),
        }
        sub_summary = {
            **sub_rate,
            "necc": float((clean_mean - sub_mean) / max(clean_mean - corrupt_mean, 1e-12)),
        }
        add_summary_rows.append(add_summary)
        sub_summary_rows.append(sub_summary)
        if alpha == 1.0:
            alpha1_add = add_summary
            alpha1_sub = sub_summary
        clear_cuda()

    clean_mean = sum(float(v["clean_tool_call_logit"]) for v in baseline_rows.values()) / max(len(baseline_rows), 1)
    corrupt_mean = sum(float(v["corrupt_tool_call_logit"]) for v in baseline_rows.values()) / max(len(baseline_rows), 1)
    write_csv(output_root / "per_sample_vector_intervention.csv", per_sample_rows)
    write_csv(output_root / "alpha_sweep_add.csv", add_summary_rows)
    write_csv(output_root / "alpha_sweep_subtract.csv", sub_summary_rows)
    write_json(
        output_root / "summary_metrics.json",
        {
            "model_path": str(args.model_path.resolve()),
            "dataset_root": str(args.dataset_root.resolve()),
            "split": args.exp_b_split,
            "batch_size": batch_size,
            "layer": layer_idx,
            "n_pairs": len(pairs),
            "clean_mean_tool_call_logit": clean_mean,
            "corrupt_mean_tool_call_logit": corrupt_mean,
            "mu_delta_norm": float(mu_delta.norm().item()),
            "vector_plus_tool_call_top1_rate_alpha_1": float(alpha1_add["tool_call_top1_rate"]) if alpha1_add else None,
            "vector_plus_strict_flip_rate_alpha_1": float(alpha1_add["strict_flip_rate"]) if alpha1_add else None,
            "vector_minus_tool_call_top1_rate_alpha_1": float(alpha1_sub["tool_call_top1_rate"]) if alpha1_sub else None,
            "vector_minus_strict_drop_rate_alpha_1": float(alpha1_sub["strict_drop_rate"]) if alpha1_sub else None,
            "suff_alpha_1": float(alpha1_add["suff"]) if alpha1_add else None,
            "necc_alpha_1": float(alpha1_sub["necc"]) if alpha1_sub else None,
            "tool_token_info": tool_token_info,
        },
    )
    summary_lines = [
        "# Exp B Summary",
        "",
        f"- layer: `L{layer_idx}`",
        f"- n_pairs: `{len(pairs)}`",
        f"- clean mean logit: `{clean_mean:.6f}`",
        f"- corrupt mean logit: `{corrupt_mean:.6f}`",
        f"- `alpha=1` suff: `{alpha1_add['suff']:.6f}`" if alpha1_add else "- `alpha=1` suff: `n/a`",
        f"- `alpha=1` necc: `{alpha1_sub['necc']:.6f}`" if alpha1_sub else "- `alpha=1` necc: `n/a`",
    ]
    if tool_token_info.get("tool_token_encode_len", 1) != 1:
        summary_lines.append(
            f"- correction: tool-call logit/prob use special token id `{tool_token_info['tool_token_id_via_convert']}` from `convert_tokens_to_ids`, because raw `encode('[TOOL_CALLS]')` is multi-piece locally."
        )
    write_text(output_root / "summary.md", "\n".join(summary_lines) + "\n")
    return mu_delta


def run_exp_c(model, tokenizer, tool_token_id: int, pairs, layer_idx: int, mu_delta: torch.Tensor, output_root: Path, batch_size: int, tool_token_info: dict[str, Any], args: argparse.Namespace):
    output_root.mkdir(parents=True, exist_ok=True)
    batches = build_pair_batches(pairs, batch_size=batch_size)
    device = infer_device(model)
    u = mu_delta.to(torch.float32)
    u = u / max(float(u.norm().item()), 1e-12)
    n_layers = get_num_layers(model)
    n_heads = get_num_heads(model)
    head_dim = get_head_dim(model)
    d_model = get_d_model(model)

    mlp_accum = {layer: {"clean": 0.0, "corrupt": 0.0, "count": 0} for layer in range(layer_idx)}
    head_accum = {(layer, head): {"clean": 0.0, "corrupt": 0.0, "count": 0} for layer in range(layer_idx) for head in range(n_heads)}
    traj_accum = {layer: {"clean": 0.0, "corrupt": 0.0, "count": 0} for layer in range(layer_idx + 1)}

    for batch in tqdm(batches, desc="Exp C", dynamic_ncols=True):
        batch_inputs = collate_pair_batch(tokenizer, batch)
        clean = to_device({"input_ids": batch_inputs["clean_input_ids"], "attention_mask": batch_inputs["clean_attention_mask"]}, device)
        corrupt = to_device({"input_ids": batch_inputs["corrupt_input_ids"], "attention_mask": batch_inputs["corrupt_attention_mask"]}, device)
        clean_positions = get_last_token_positions(clean["attention_mask"]).detach().cpu()
        corrupt_positions = get_last_token_positions(corrupt["attention_mask"]).detach().cpu()
        clean_layer_store: dict[int, torch.Tensor] = {}
        corrupt_layer_store: dict[int, torch.Tensor] = {}
        clean_mlp_store: dict[int, torch.Tensor] = {}
        corrupt_mlp_store: dict[int, torch.Tensor] = {}
        clean_attn_store: dict[int, torch.Tensor] = {}
        corrupt_attn_store: dict[int, torch.Tensor] = {}
        clean_specs = []
        clean_pre_specs = []
        corrupt_specs = []
        corrupt_pre_specs = []
        for layer in range(layer_idx + 1):
            clean_specs.append((get_layer(model, layer), make_capture_hook(clean_layer_store, layer, clean_positions)))
            corrupt_specs.append((get_layer(model, layer), make_capture_hook(corrupt_layer_store, layer, corrupt_positions)))
        for layer in range(layer_idx):
            clean_specs.append((get_mlp(model, layer), make_capture_hook(clean_mlp_store, layer, clean_positions)))
            corrupt_specs.append((get_mlp(model, layer), make_capture_hook(corrupt_mlp_store, layer, corrupt_positions)))
            clean_pre_specs.append((get_o_proj(model, layer), make_pre_capture_hook(clean_attn_store, layer, clean_positions)))
            corrupt_pre_specs.append((get_o_proj(model, layer), make_pre_capture_hook(corrupt_attn_store, layer, corrupt_positions)))
        with register_hooks(clean_specs, clean_pre_specs):
            _ = batch_logits(model, clean["input_ids"], clean["attention_mask"])
        with register_hooks(corrupt_specs, corrupt_pre_specs):
            _ = batch_logits(model, corrupt["input_ids"], corrupt["attention_mask"])
        for layer in range(layer_idx + 1):
            clean_state = clean_layer_store[layer].float()
            corrupt_state = corrupt_layer_store[layer].float()
            traj_accum[layer]["clean"] += float((clean_state @ u).sum().item())
            traj_accum[layer]["corrupt"] += float((corrupt_state @ u).sum().item())
            traj_accum[layer]["count"] += int(clean_state.shape[0])
        for layer in range(layer_idx):
            clean_mlp = clean_mlp_store[layer].float()
            corrupt_mlp = corrupt_mlp_store[layer].float()
            mlp_accum[layer]["clean"] += float((clean_mlp @ u).sum().item())
            mlp_accum[layer]["corrupt"] += float((corrupt_mlp @ u).sum().item())
            mlp_accum[layer]["count"] += int(clean_mlp.shape[0])
            clean_attn = clean_attn_store[layer].float()
            corrupt_attn = corrupt_attn_store[layer].float()
            w = get_o_proj(model, layer).weight.detach().cpu().float()
            for head in range(n_heads):
                start = head * head_dim
                end = start + head_dim
                proj_vec = w[:, start:end].T @ u
                head_accum[(layer, head)]["clean"] += float((clean_attn[:, start:end] @ proj_vec).sum().item())
                head_accum[(layer, head)]["corrupt"] += float((corrupt_attn[:, start:end] @ proj_vec).sum().item())
                head_accum[(layer, head)]["count"] += int(clean_attn.shape[0])
        clear_cuda()

    mlp_rows = []
    for layer in range(layer_idx):
        row = mlp_accum[layer]
        count = max(row["count"], 1)
        mlp_rows.append(
            {
                "layer": layer,
                "clean_projection": float(row["clean"] / count),
                "corrupt_projection": float(row["corrupt"] / count),
                "mlp_delta": float((row["clean"] - row["corrupt"]) / count),
                "count": int(count),
            }
        )
    head_rows = []
    for layer in range(layer_idx):
        for head in range(n_heads):
            row = head_accum[(layer, head)]
            count = max(row["count"], 1)
            head_rows.append(
                {
                    "layer": layer,
                    "head": head,
                    "clean_projection": float(row["clean"] / count),
                    "corrupt_projection": float(row["corrupt"] / count),
                    "head_delta": float((row["clean"] - row["corrupt"]) / count),
                    "count": int(count),
                }
            )
    traj_rows = []
    for layer in range(layer_idx + 1):
        row = traj_accum[layer]
        count = max(row["count"], 1)
        traj_rows.append(
            {
                "layer": layer,
                "clean_projection": float(row["clean"] / count),
                "corrupt_projection": float(row["corrupt"] / count),
                "traj": float((row["clean"] - row["corrupt"]) / count),
                "count": int(count),
            }
        )

    mlp_rows_sorted = sorted(mlp_rows, key=lambda row: row["mlp_delta"], reverse=True)
    head_rows_sorted = sorted(head_rows, key=lambda row: row["head_delta"], reverse=True)
    write_csv(output_root / "mlp_contribution.csv", mlp_rows)
    write_csv(output_root / "head_contribution.csv", head_rows)
    write_csv(output_root / "residual_trajectory.csv", traj_rows)
    write_csv(output_root / "top_mlp_blocks.csv", mlp_rows_sorted[:5])
    write_csv(output_root / "top_attention_heads.csv", head_rows_sorted[:10])
    write_json(
        output_root / "metadata.json",
        {
            "layer": layer_idx,
            "n_layers": n_layers,
            "n_heads": n_heads,
            "head_dim": head_dim,
            "d_model": d_model,
            "n_pairs": len(pairs),
            "tool_token_info": tool_token_info,
        },
    )
    summary_lines = [
        "# Exp C Summary",
        "",
        f"- layer: `L{layer_idx}`",
        f"- top-5 MLP: `{[row['layer'] for row in mlp_rows_sorted[:5]]}`",
        f"- top-10 heads: `{[(row['layer'], row['head']) for row in head_rows_sorted[:10]]}`",
    ]
    if tool_token_info.get("tool_token_encode_len", 1) != 1:
        summary_lines.append(
            f"- correction: upstream projection targets the same special tool-call id `{tool_token_info['tool_token_id_via_convert']}` used in Exp A/B."
        )
    write_text(output_root / "summary.md", "\n".join(summary_lines) + "\n")

    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    ax.plot([row["layer"] for row in traj_rows], [row["traj"] for row in traj_rows], marker="o")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean clean-corrupt projection")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_root / "plot_residual_trajectory.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    layers = sorted(set(row["layer"] for row in head_rows))
    heads = sorted(set(row["head"] for row in head_rows))
    matrix = torch.zeros((len(layers), len(heads)), dtype=torch.float32)
    for row in head_rows:
        matrix[layers.index(row["layer"]), heads.index(row["head"])] = float(row["head_delta"])
    im = ax.imshow(matrix.numpy(), aspect="auto", cmap="coolwarm")
    ax.set_xlabel("Head")
    ax.set_ylabel("Layer")
    ax.set_xticks(range(len(heads)))
    ax.set_xticklabels(heads, rotation=90, fontsize=6)
    ax.set_yticks(range(len(layers)))
    ax.set_yticklabels(layers, fontsize=6)
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(output_root / "plot_head_contribution_heatmap.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    ax.bar([row["layer"] for row in mlp_rows], [row["mlp_delta"] for row in mlp_rows])
    ax.set_xlabel("Layer")
    ax.set_ylabel("MLP delta")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_root / "plot_mlp_contribution.pdf")
    plt.close(fig)


def main():
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    exp_a_root = args.output_root / "exp_a_state_patch"
    exp_b_root = args.output_root / "exp_b_tool_call_vector"
    exp_c_root = args.output_root / "exp_c_upstream_projection"
    tokenizer = load_tokenizer(args.model_path)
    all_pairs = load_pairs(tokenizer, dataset_root=args.dataset_root, metadata_root=args.metadata_root, split="all", max_pairs=args.max_pairs)
    max_seq_len = max(max(pair.clean_len, pair.corrupt_len) for pair in all_pairs)
    tool_token_id, tool_token_info = resolve_tool_token_id(tokenizer, TOOL_CALL_TEXT)
    dtype = dtype_from_name(args.dtype)
    free_vram_info = wait_for_vram(args.min_free_vram_gib, poll_seconds=args.wait_poll_seconds)
    model = load_model(args.model_path, dtype=dtype, device_map=args.device_map)
    vram_estimate = estimate_required_vram_gib(
        n_layers=get_num_layers(model),
        hidden_size=get_d_model(model),
        vocab_size=int(model.config.text_config.vocab_size if hasattr(model.config, "text_config") else model.config.vocab_size),
        dtype=dtype,
        batch_size=args.batch_size,
        max_seq_len=max_seq_len,
        model_path=args.model_path,
    )
    write_json(
        args.output_root / "run_metadata.json",
        {
            "model_path": str(args.model_path.resolve()),
            "dataset_root": str(args.dataset_root.resolve()),
            "metadata_root": str(args.metadata_root.resolve()),
            "output_root": str(args.output_root.resolve()),
            "dtype": args.dtype,
            "device_map": args.device_map,
            "batch_size": args.batch_size,
            "max_pairs": args.max_pairs,
            "tool_token_info": tool_token_info,
            "free_vram_info": free_vram_info,
            "vram_estimate": vram_estimate,
        },
    )
    sanity = run_behavior_sanity_check(model, tokenizer, tool_token_id, all_pairs, args.batch_size)
    write_json(args.output_root / "behavior_sanity_check.json", sanity)
    if sanity["clean_tool_call_top1_rate"] <= 0.5 or sanity["corrupt_tool_call_top1_rate"] >= 0.5:
        raise RuntimeError(
            "Behavior sanity check failed for the selected dataset root: "
            f"clean_rate={sanity['clean_tool_call_top1_rate']:.4f}, "
            f"corrupt_rate={sanity['corrupt_tool_call_top1_rate']:.4f}. "
            "Do not run patch experiments on a non-separated dataset representation."
        )
    exp_a_pairs = [pair for pair in all_pairs if args.exp_a_split == "all" or pair.split == args.exp_a_split]
    exp_b_pairs = [pair for pair in all_pairs if args.exp_b_split == "all" or pair.split == args.exp_b_split]
    exp_c_pairs = [pair for pair in all_pairs if args.exp_c_split == "all" or pair.split == args.exp_c_split]
    best_layer, _ = run_exp_a(model, tokenizer, tool_token_id, exp_a_pairs, exp_a_root, args.batch_size, tool_token_info, args)
    mu_delta = run_exp_b(model, tokenizer, tool_token_id, exp_b_pairs, best_layer, exp_b_root, args.batch_size, tool_token_info, args)
    run_exp_c(model, tokenizer, tool_token_id, exp_c_pairs, best_layer, mu_delta, exp_c_root, args.batch_size, tool_token_info, args)
    write_text(args.output_root / "summary.md", summarize_main(best_layer, args.output_root, tool_token_info))


if __name__ == "__main__":
    main()
