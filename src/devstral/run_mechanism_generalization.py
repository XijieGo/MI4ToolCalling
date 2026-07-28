#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import math
import shutil
from collections import defaultdict
from datetime import datetime
from pathlib import Path

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

from artifact_paths import DEVSTRAL_2_24B_PATH  # noqa: E402

from dataset_utils import DEFAULT_DATASET_ROOT, build_pair_batches, collate_pair_batch, load_pairs, write_csv, write_json, write_text
from hf_patch_utils import (
    TOOL_CALL_TEXT,
    assert_single_token,
    get_d_model,
    get_head_dim,
    get_layer,
    get_layers,
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
    summarize_rate,
    to_token_text,
    tool_stats,
)


MODEL_PATH = DEVSTRAL_2_24B_PATH
OUTPUT_ROOT = PROJECT_ROOT / "results" / "section6_generalization" / "devstral_2_24b" / "mechanism_generalization_rerun"

COARSE_FRACTIONS = (0.45, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00)
ALPHAS = (0.5, 1.0, 1.5, 2.0)
TAIL_LAYER_SPAN = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Devstral mechanism generalization runner.")
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--exp-a-split", type=str, default="all")
    parser.add_argument("--exp-b-split", type=str, default="train")
    parser.add_argument("--exp-c-split", type=str, default="train")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-pairs", type=int, default=0)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--device-map", type=str, default="auto")
    parser.add_argument("--max-memory-gib", type=int, default=94)
    parser.add_argument("--run-tag", type=str, default="")
    parser.add_argument("--clear-output", action="store_true")
    return parser.parse_args()


def resolve_run_tag(args: argparse.Namespace) -> str:
    if args.run_tag:
        return str(args.run_tag)
    return datetime.utcnow().strftime("run_%Y%m%dT%H%M%SZ")


def prepare_output_roots(output_root: Path, run_tag: str, clear_output: bool) -> tuple[Path, Path, Path, Path]:
    if clear_output and output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    run_root = output_root / run_tag
    if run_root.exists():
        shutil.rmtree(run_root)
    run_root.mkdir(parents=True, exist_ok=True)
    exp_a_root = run_root / "exp_a_state_patch"
    exp_b_root = run_root / "exp_b_tool_call_vector"
    exp_c_root = run_root / "exp_c_upstream_projection"
    return run_root, exp_a_root, exp_b_root, exp_c_root


def build_model_and_tokenizer(args: argparse.Namespace):
    tokenizer = load_tokenizer(args.model_path)
    tool_token_id = assert_single_token(tokenizer, TOOL_CALL_TEXT)
    dtype = getattr(torch, args.dtype)
    model = load_model(
        args.model_path,
        dtype=dtype,
        device_map=args.device_map,
    )
    return model, tokenizer, tool_token_id


def infer_device(model) -> torch.device:
    return next(model.parameters()).device


def to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def layer_state_pass(model, input_ids, attention_mask, layer_idx: int):
    store: dict[int, torch.Tensor] = {}
    hooks = [(get_layer(model, layer_idx), make_capture_hook(store, layer_idx))]
    with register_hooks(hooks):
        with torch.inference_mode():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                logits_to_keep=1,
                output_hidden_states=True,
                return_dict=True,
            )
    logits = outputs.logits
    state = store[layer_idx]
    return logits, state


def collect_layer_states(model, input_ids, attention_mask, layer_indices: list[int]):
    stores: dict[int, torch.Tensor] = {}
    hooks = [(get_layer(model, layer_idx), make_capture_hook(stores, layer_idx)) for layer_idx in layer_indices]
    with register_hooks(hooks):
        with torch.inference_mode():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                logits_to_keep=1,
                output_hidden_states=True,
                return_dict=True,
            )
    return outputs.logits, stores


def make_layer_patch_hook(source_cpu: torch.Tensor):
    def hook_fn(module, inputs, output):  # noqa: ANN001
        out = output.clone()
        src = source_cpu.to(device=out.device, dtype=out.dtype)
        out[:, -1, :] = src
        return out

    return hook_fn


def batch_logits(model, input_ids, attention_mask):
    with torch.inference_mode():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            logits_to_keep=1,
            return_dict=True,
        )
    return outputs.logits


def capture_layer_hidden(model, batch, layer_idx: int):
    clean_store: dict[int, torch.Tensor] = {}
    corrupt_store: dict[int, torch.Tensor] = {}
    clean_hooks = [(get_layer(model, layer_idx), make_capture_hook(clean_store, layer_idx))]
    corrupt_hooks = [(get_layer(model, layer_idx), make_capture_hook(corrupt_store, layer_idx))]
    device = infer_device(model)
    clean_batch = to_device(batch["clean"], device)
    corrupt_batch = to_device(batch["corrupt"], device)
    with register_hooks(clean_hooks):
        with torch.inference_mode():
            clean_logits = model(
                input_ids=clean_batch["input_ids"],
                attention_mask=clean_batch["attention_mask"],
                use_cache=False,
                logits_to_keep=1,
                output_hidden_states=True,
                return_dict=True,
            ).logits
    with register_hooks(corrupt_hooks):
        with torch.inference_mode():
            corrupt_logits = model(
                input_ids=corrupt_batch["input_ids"],
                attention_mask=corrupt_batch["attention_mask"],
                use_cache=False,
                logits_to_keep=1,
                output_hidden_states=True,
                return_dict=True,
            ).logits
    return clean_logits, corrupt_logits, clean_store[layer_idx], corrupt_store[layer_idx]


def top1_rows(tokenizer, logits: torch.Tensor, tool_token_id: int):
    tool_logit, tool_prob, top1_ids, _ = tool_stats(logits, tool_token_id)
    return tool_logit, tool_prob, top1_ids, [to_token_text(tokenizer, int(item)) for item in top1_ids]


def summarize_exp_a_rows(rows: list[dict[str, object]], best_layer: int, split: str) -> str:
    best_row = next(row for row in rows if int(row["layer"]) == best_layer and row["phase"] == "final")
    return "\n".join(
        [
            "# Exp A Summary",
            "",
            f"- split: `{split}`",
            f"- best layer: `L{best_layer}`",
            f"- n_rows: `{len(rows)}`",
            f"- best tool-call top1: `{float(best_row['tool_call_top1_rate']):.4f}`",
            f"- best strict flip: `{float(best_row['strict_flip_rate']):.4f}`",
            f"- mean tool logit: `{float(best_row['mean_tool_call_logit']):.4f}`",
            f"- mean tool prob: `{float(best_row['mean_tool_call_prob']):.4f}`",
            "",
            f"- correction: the actual hook path is `model.model.language_model.model.layers[...]`.",
        ]
    ) + "\n"


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


def choose_best_layer(rows: list[dict[str, object]], *, phase: str = "final") -> int:
    phase_rows = [row for row in rows if row["phase"] == phase]
    max_rate = max(float(row["tool_call_top1_rate"]) for row in phase_rows)
    candidates = [row for row in phase_rows if float(row["tool_call_top1_rate"]) >= max_rate - 1e-6]
    return int(sorted(candidates, key=lambda row: int(row["layer"]))[0]["layer"])


def build_patch_layers(n_layers: int) -> list[int]:
    coarse = []
    for frac in COARSE_FRACTIONS:
        layer = min(max(int(round(frac * n_layers)) - 1, 0), n_layers - 1)
        if layer not in coarse:
            coarse.append(layer)
    tail_start = max(n_layers - TAIL_LAYER_SPAN, 0)
    for layer in range(tail_start, n_layers):
        if layer not in coarse:
            coarse.append(layer)
    return coarse


def validate_final_layer_patch_equivalence(model, tokenizer, tool_token_id: int, pairs, batch_size: int) -> dict[str, object]:
    if not pairs:
        return {
            "checked_pairs": 0,
            "layer": get_num_layers(model) - 1,
            "max_abs_clean_vs_patched": None,
            "mean_abs_clean_vs_patched": None,
            "clean_top1_rate": None,
            "patched_top1_rate": None,
        }

    layer_idx = get_num_layers(model) - 1
    batches = build_pair_batches(pairs, batch_size=batch_size)
    device = infer_device(model)
    max_abs = 0.0
    mean_abs_sum = 0.0
    n_batches = 0
    clean_tool_top1 = 0
    patched_tool_top1 = 0
    n_examples = 0

    for batch in batches:
        batch_inputs = collate_pair_batch(tokenizer, batch)
        clean = to_device({"input_ids": batch_inputs["clean_input_ids"], "attention_mask": batch_inputs["clean_attention_mask"]}, device)
        corrupt = to_device({"input_ids": batch_inputs["corrupt_input_ids"], "attention_mask": batch_inputs["corrupt_attention_mask"]}, device)
        clean_store: dict[int, torch.Tensor] = {}
        clean_hooks = [(get_layer(model, layer_idx), make_capture_hook(clean_store, layer_idx))]
        with register_hooks(clean_hooks):
            clean_logits = batch_logits(model, clean["input_ids"], clean["attention_mask"])
        patch_source = clean_store[layer_idx].detach().cpu()
        patch_hooks = [(get_layer(model, layer_idx), make_last_token_replace_hook(patch_source))]
        with register_hooks(patch_hooks):
            patched_logits = batch_logits(model, corrupt["input_ids"], corrupt["attention_mask"])

        clean_last = clean_logits[:, -1, :].float().cpu()
        patched_last = patched_logits[:, -1, :].float().cpu()
        diff = (clean_last - patched_last).abs()
        max_abs = max(max_abs, float(diff.max().item()))
        mean_abs_sum += float(diff.mean().item())
        n_batches += 1

        _, _, clean_top1, _ = tool_stats(clean_logits, tool_token_id)
        _, _, patched_top1, _ = tool_stats(patched_logits, tool_token_id)
        clean_tool_top1 += int((clean_top1 == tool_token_id).sum().item())
        patched_tool_top1 += int((patched_top1 == tool_token_id).sum().item())
        n_examples += int(clean_top1.shape[0])

        del clean_logits, patched_logits, clean_last, patched_last, diff, clean_top1, patched_top1
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return {
        "checked_pairs": n_examples,
        "layer": layer_idx,
        "max_abs_clean_vs_patched": max_abs,
        "mean_abs_clean_vs_patched": float(mean_abs_sum / max(n_batches, 1)),
        "clean_top1_rate": float(clean_tool_top1 / max(n_examples, 1)),
        "patched_top1_rate": float(patched_tool_top1 / max(n_examples, 1)),
    }


def run_exp_a(model, tokenizer, tool_token_id: int, pairs, output_root: Path, batch_size: int, *, model_path: Path, dataset_root: Path):
    output_root.mkdir(parents=True, exist_ok=True)
    batches = build_pair_batches(pairs, batch_size=batch_size)
    n_layers = get_num_layers(model)
    coarse_layers = build_patch_layers(n_layers)
    final_layer_equivalence = validate_final_layer_patch_equivalence(model, tokenizer, tool_token_id, pairs[: min(len(pairs), 32)], batch_size)
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

        clean_store: dict[int, torch.Tensor] = {}
        clean_hooks = [(get_layer(model, layer_idx), make_capture_hook(clean_store, layer_idx)) for layer_idx in coarse_layers]
        with register_hooks(clean_hooks):
            clean_logits = batch_logits(model, clean["input_ids"], clean["attention_mask"])
        corrupt_logits = batch_logits(model, corrupt["input_ids"], corrupt["attention_mask"])
        clean_tool_logit, clean_tool_prob, clean_top1, _ = tool_stats(clean_logits, tool_token_id)
        corrupt_tool_logit, corrupt_tool_prob, corrupt_top1, _ = tool_stats(corrupt_logits, tool_token_id)
        baseline_clean_tool_top1 += int((clean_top1 == tool_token_id).sum().item())
        baseline_corrupt_tool_top1 += int((corrupt_top1 == tool_token_id).sum().item())
        baseline_count += int(clean_top1.shape[0])

        for layer in coarse_layers:
            patch_source = clean_store[layer].detach().cpu()
            patch_hooks = [(get_layer(model, layer), make_last_token_replace_hook(patch_source))]
            with register_hooks(patch_hooks):
                patched_logits = batch_logits(model, corrupt["input_ids"], corrupt["attention_mask"])
            patched_tool_logit, patched_tool_prob, patched_top1, _ = tool_stats(patched_logits, tool_token_id)
            bucket = accum[layer]
            bucket["count"] += int(patched_top1.shape[0])
            bucket["tool_top1"] += int((patched_top1 == tool_token_id).sum().item())
            bucket["strict_flip"] += int(((corrupt_top1 != tool_token_id) & (patched_top1 == tool_token_id)).sum().item())
            bucket["logit_sum"] += float(patched_tool_logit.sum().item())
            bucket["prob_sum"] += float(patched_tool_prob.sum().item())
            del patched_logits, patched_tool_logit, patched_tool_prob, patched_top1
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        for layer in coarse_layers:
            del clean_store[layer]
        del clean_logits, corrupt_logits, clean_tool_logit, clean_tool_prob, clean_top1, corrupt_tool_logit, corrupt_tool_prob, corrupt_top1
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

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

    best_coarse = choose_best_layer(rows, phase="coarse")
    final_layers = sorted({max(best_coarse - 2, 0), max(best_coarse - 1, 0), best_coarse, min(best_coarse + 1, n_layers - 1), min(best_coarse + 2, n_layers - 1)})
    final_accum = {
        layer: {"count": 0, "tool_top1": 0, "strict_flip": 0, "logit_sum": 0.0, "prob_sum": 0.0}
        for layer in final_layers
    }

    for batch in tqdm(batches, desc="Exp A final", dynamic_ncols=True):
        batch_inputs = collate_pair_batch(tokenizer, batch)
        device = infer_device(model)
        clean = to_device({"input_ids": batch_inputs["clean_input_ids"], "attention_mask": batch_inputs["clean_attention_mask"]}, device)
        corrupt = to_device({"input_ids": batch_inputs["corrupt_input_ids"], "attention_mask": batch_inputs["corrupt_attention_mask"]}, device)
        clean_store: dict[int, torch.Tensor] = {}
        clean_hooks = [(get_layer(model, layer_idx), make_capture_hook(clean_store, layer_idx)) for layer_idx in final_layers]
        with register_hooks(clean_hooks):
            clean_logits = batch_logits(model, clean["input_ids"], clean["attention_mask"])
        corrupt_logits = batch_logits(model, corrupt["input_ids"], corrupt["attention_mask"])
        _, _, clean_top1, _ = tool_stats(clean_logits, tool_token_id)
        _, _, corrupt_top1, _ = tool_stats(corrupt_logits, tool_token_id)
        for layer in final_layers:
            patch_source = clean_store[layer].detach().cpu()
            patch_hooks = [(get_layer(model, layer), make_last_token_replace_hook(patch_source))]
            with register_hooks(patch_hooks):
                patched_logits = batch_logits(model, corrupt["input_ids"], corrupt["attention_mask"])
            patched_tool_logit, patched_tool_prob, patched_top1, _ = tool_stats(patched_logits, tool_token_id)
            bucket = final_accum[layer]
            bucket["count"] += int(patched_top1.shape[0])
            bucket["tool_top1"] += int((patched_top1 == tool_token_id).sum().item())
            bucket["strict_flip"] += int(((corrupt_top1 != tool_token_id) & (patched_top1 == tool_token_id)).sum().item())
            bucket["logit_sum"] += float(patched_tool_logit.sum().item())
            bucket["prob_sum"] += float(patched_tool_prob.sum().item())
            del patched_logits, patched_tool_logit, patched_tool_prob, patched_top1
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        del clean_store, clean_logits, corrupt_logits, clean_top1, corrupt_top1
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

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
    best_layer = choose_best_layer(rows, phase="final")
    write_csv(output_root / "patch_sweep.csv", rows)
    write_json(
        output_root / "metadata.json",
        {
            "model_path": str(model_path.resolve()),
            "dataset_root": str(dataset_root.resolve()),
            "split": "all",
            "batch_size": batch_size,
            "n_pairs": len(pairs),
            "n_layers": n_layers,
            "coarse_layers": coarse_layers,
            "final_layers": final_layers,
            "best_layer": best_layer,
            "final_layer_equivalence": final_layer_equivalence,
            "tool_token_text": TOOL_CALL_TEXT,
            "tool_token_id": tool_token_id,
        },
    )
    write_text(output_root / "summary.md", summarize_exp_a_rows(rows, best_layer, "all"))
    plot_exp_a(rows, output_root / "plot_patch_sweep.pdf")
    return best_layer, rows


def ensure_rank1_rows(alpha_rows: list[dict[str, object]], alpha: float, direction: str, rows: list[dict[str, object]]):
    for row in rows:
        row["alpha"] = alpha
        row["direction"] = direction
        alpha_rows.append(row)


def run_exp_b(
    model,
    tokenizer,
    tool_token_id: int,
    pairs,
    layer_idx: int,
    output_root: Path,
    batch_size: int,
    *,
    model_path: Path,
    dataset_root: Path,
):
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
        clean_store: dict[int, torch.Tensor] = {}
        corrupt_store: dict[int, torch.Tensor] = {}
        clean_hooks = [(get_layer(model, layer_idx), make_capture_hook(clean_store, layer_idx))]
        corrupt_hooks = [(get_layer(model, layer_idx), make_capture_hook(corrupt_store, layer_idx))]
        with register_hooks(clean_hooks):
            clean_logits = batch_logits(model, clean["input_ids"], clean["attention_mask"])
        with register_hooks(corrupt_hooks):
            corrupt_logits = batch_logits(model, corrupt["input_ids"], corrupt["attention_mask"])
        clean_tool_logit, clean_tool_prob, clean_top1, clean_text = top1_rows(tokenizer, clean_logits, tool_token_id)
        corrupt_tool_logit, corrupt_tool_prob, corrupt_top1, corrupt_text = top1_rows(tokenizer, corrupt_logits, tool_token_id)
        clean_state = clean_store[layer_idx].detach().float().cpu()
        corrupt_state = corrupt_store[layer_idx].detach().float().cpu()
        diff = clean_state - corrupt_state
        mu_sum = diff.sum(dim=0) if mu_sum is None else mu_sum + diff.sum(dim=0)
        n_total += int(diff.shape[0])
        for local_idx, pair in enumerate(batch.examples):
            baseline_rows[pair.sample_id] = {
                "sample_id": pair.sample_id,
                "split": pair.split,
                "language": pair.language,
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
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    mu_delta = (mu_sum / max(n_total, 1)).to(torch.float32)
    u = mu_delta / mu_delta.norm()
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
            if direction == "add":
                patch_source = mu_delta * float(alpha)
                hooks = [(get_layer(model, layer_idx), make_last_token_add_hook(patch_source))]
                with register_hooks(hooks):
                    patched_logits = batch_logits(model, corrupt["input_ids"], corrupt["attention_mask"])
                _, _, patched_top1, patched_text = top1_rows(tokenizer, patched_logits, tool_token_id)
                patched_tool_logit, patched_tool_prob, _, _ = tool_stats(patched_logits, tool_token_id)
                for local_idx, pair in enumerate(batch.examples):
                    baseline = baseline_rows[pair.sample_id]
                    corrupt_is_tool = bool(baseline["corrupt_is_tool_call_top1"])
                    strict_flip = bool((not corrupt_is_tool) and int(patched_top1[local_idx].item()) == tool_token_id)
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
                            "clean_tool_call_logit": baseline["clean_tool_call_logit"],
                            "corrupt_tool_call_logit": baseline["corrupt_tool_call_logit"],
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
                accum["logit_sum"] += float(patched_tool_logit.sum().item())
                accum["prob_sum"] += float(patched_tool_prob.sum().item())
            else:
                patch_source = mu_delta * float(alpha)
                hooks = [(get_layer(model, layer_idx), make_last_token_subtract_hook(patch_source))]
                with register_hooks(hooks):
                    patched_logits = batch_logits(model, clean["input_ids"], clean["attention_mask"])
                _, _, patched_top1, patched_text = top1_rows(tokenizer, patched_logits, tool_token_id)
                patched_tool_logit, patched_tool_prob, _, _ = tool_stats(patched_logits, tool_token_id)
                for local_idx, pair in enumerate(batch.examples):
                    baseline = baseline_rows[pair.sample_id]
                    clean_is_tool = bool(baseline["clean_is_tool_call_top1"])
                    strict_drop = bool(clean_is_tool and int(patched_top1[local_idx].item()) != tool_token_id)
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
                    if alpha == 1.0:
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
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
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

    for alpha in ALPHAS:
        add_rows, add_rate = run_alpha(alpha, "add")
        alpha_add_rows.extend(add_rows)
        alpha_sub_rows_for_alpha, sub_rate = run_alpha(alpha, "subtract")
        alpha_sub_rows.extend(alpha_sub_rows_for_alpha)
        del add_rows, alpha_sub_rows_for_alpha
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    clean_mean = sum(float(v["clean_tool_call_logit"]) for v in baseline_rows.values()) / max(len(baseline_rows), 1)
    corrupt_mean = sum(float(v["corrupt_tool_call_logit"]) for v in baseline_rows.values()) / max(len(baseline_rows), 1)
    alpha1_add = None
    alpha1_sub = None
    add_summary_rows = []
    sub_summary_rows = []
    for alpha in ALPHAS:
        add_subset = [row for row in alpha_add_rows if float(row["alpha"]) == float(alpha)]
        sub_subset = [row for row in alpha_sub_rows if float(row["alpha"]) == float(alpha)]
        if add_subset:
            add_mean = sum(float(row["vector_plus_tool_call_logit"]) for row in add_subset) / len(add_subset)
            add_rate = sum(int(bool(row["vector_plus_is_tool_call_top1"])) for row in add_subset) / len(add_subset)
            strict_flip_rate = sum(
                int((not bool(row["corrupt_is_tool_call_top1"])) and bool(row["vector_plus_is_tool_call_top1"]))
                for row in add_subset
            ) / len(add_subset)
            add_summary_rows.append({"alpha": alpha, "n": len(add_subset), "mean_tool_call_logit": add_mean, "tool_call_top1_rate": add_rate, "strict_flip_rate": strict_flip_rate, "suff": float((add_mean - corrupt_mean) / max(clean_mean - corrupt_mean, 1e-12))})
            if alpha == 1.0:
                alpha1_add = add_summary_rows[-1]
        if sub_subset:
            sub_mean = sum(float(row["vector_minus_tool_call_logit"]) for row in sub_subset) / len(sub_subset)
            sub_rate = sum(int(bool(row["vector_minus_is_tool_call_top1"])) for row in sub_subset) / len(sub_subset)
            strict_drop_rate = sum(
                int(bool(row["clean_is_tool_call_top1"]) and (not bool(row["vector_minus_is_tool_call_top1"])))
                for row in sub_subset
            ) / len(sub_subset)
            sub_summary_rows.append({"alpha": alpha, "n": len(sub_subset), "mean_tool_call_logit": sub_mean, "tool_call_top1_rate": sub_rate, "strict_drop_rate": strict_drop_rate, "necc": float((clean_mean - sub_mean) / max(clean_mean - corrupt_mean, 1e-12))})
            if alpha == 1.0:
                alpha1_sub = sub_summary_rows[-1]

    write_csv(output_root / "per_sample_vector_intervention.csv", per_sample_rows)
    write_csv(output_root / "alpha_sweep_add.csv", add_summary_rows)
    write_csv(output_root / "alpha_sweep_subtract.csv", sub_summary_rows)
    write_json(
        output_root / "summary_metrics.json",
        {
            "model_path": str(model_path.resolve()),
            "dataset_root": str(dataset_root.resolve()),
            "split": "train",
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
        },
    )
    summary_md = "\n".join(
        [
            "# Exp B Summary",
            "",
            f"- layer: `L{layer_idx}`",
            f"- n_pairs: `{len(pairs)}`",
            f"- clean mean logit: `{clean_mean:.6f}`",
            f"- corrupt mean logit: `{corrupt_mean:.6f}`",
            f"- `alpha=1` suff: `{alpha1_add['suff']:.6f}`" if alpha1_add else "- `alpha=1` suff: `n/a`",
            f"- `alpha=1` necc: `{alpha1_sub['necc']:.6f}`" if alpha1_sub else "- `alpha=1` necc: `n/a`",
        ]
    ) + "\n"
    write_text(output_root / "summary.md", summary_md)
    write_json(
        output_root / "metadata.json",
        {
            "layer": layer_idx,
            "tool_token_text": TOOL_CALL_TEXT,
            "tool_token_id": tool_token_id,
            "alphas": list(ALPHAS),
            "split": "train",
        },
    )
    return mu_delta


def run_exp_c(model, tokenizer, tool_token_id: int, pairs, layer_idx: int, mu_delta: torch.Tensor, output_root: Path, batch_size: int):
    output_root.mkdir(parents=True, exist_ok=True)
    batches = build_pair_batches(pairs, batch_size=batch_size)
    device = infer_device(model)
    u = mu_delta.to(torch.float32)
    u = u / u.norm()
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
        clean_layer_store: dict[int, torch.Tensor] = {}
        corrupt_layer_store: dict[int, torch.Tensor] = {}
        clean_mlp_store: dict[int, torch.Tensor] = {}
        corrupt_mlp_store: dict[int, torch.Tensor] = {}
        clean_attn_store: dict[int, torch.Tensor] = {}
        corrupt_attn_store: dict[int, torch.Tensor] = {}
        specs = []
        pre_specs = []
        for layer in range(layer_idx + 1):
            specs.append((get_layer(model, layer), make_capture_hook(clean_layer_store, layer)))
        for layer in range(layer_idx):
            specs.append((get_mlp(model, layer), make_capture_hook(clean_mlp_store, layer)))
            pre_specs.append((get_o_proj(model, layer), make_pre_capture_hook(clean_attn_store, layer)))
        with register_hooks(specs, pre_specs):
            clean_logits = batch_logits(model, clean["input_ids"], clean["attention_mask"])
        specs = []
        pre_specs = []
        for layer in range(layer_idx + 1):
            specs.append((get_layer(model, layer), make_capture_hook(corrupt_layer_store, layer)))
        for layer in range(layer_idx):
            specs.append((get_mlp(model, layer), make_capture_hook(corrupt_mlp_store, layer)))
            pre_specs.append((get_o_proj(model, layer), make_pre_capture_hook(corrupt_attn_store, layer)))
        with register_hooks(specs, pre_specs):
            corrupt_logits = batch_logits(model, corrupt["input_ids"], corrupt["attention_mask"])
        clean_tool_logit, clean_tool_prob, clean_top1, _ = tool_stats(clean_logits, tool_token_id)
        corrupt_tool_logit, corrupt_tool_prob, corrupt_top1, _ = tool_stats(corrupt_logits, tool_token_id)
        for layer in range(layer_idx + 1):
            clean_state = clean_layer_store[layer].float()
            corrupt_state = corrupt_layer_store[layer].float()
            delta = (clean_state - corrupt_state) @ u
            traj_accum[layer]["clean"] += float((clean_state @ u).sum().item())
            traj_accum[layer]["corrupt"] += float((corrupt_state @ u).sum().item())
            traj_accum[layer]["count"] += int(delta.shape[0])
        for layer in range(layer_idx):
            clean_mlp = clean_mlp_store[layer].float()
            corrupt_mlp = corrupt_mlp_store[layer].float()
            proj_clean = (clean_mlp @ u).sum().item()
            proj_corrupt = (corrupt_mlp @ u).sum().item()
            mlp_accum[layer]["clean"] += float(proj_clean)
            mlp_accum[layer]["corrupt"] += float(proj_corrupt)
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
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

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
        },
    )
    write_text(
        output_root / "summary.md",
        "\n".join(
            [
                "# Exp C Summary",
                "",
                f"- layer: `L{layer_idx}`",
                f"- n_pairs: `{len(pairs)}`",
                f"- top-5 MLP: `{[row['layer'] for row in mlp_rows_sorted[:5]]}`",
                f"- top-10 heads: `{[(row['layer'], row['head']) for row in head_rows_sorted[:10]]}`",
            ]
        )
        + "\n",
    )
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
    run_tag = resolve_run_tag(args)
    run_root, exp_a_root, exp_b_root, exp_c_root = prepare_output_roots(args.output_root, run_tag, args.clear_output)
    model, tokenizer, tool_token_id = build_model_and_tokenizer(args)
    all_pairs = load_pairs(tokenizer, dataset_root=args.dataset_root, split="all", max_pairs=args.max_pairs)
    exp_a_pairs = [pair for pair in all_pairs if args.exp_a_split == "all" or pair.split == args.exp_a_split]
    exp_b_pairs = [pair for pair in all_pairs if pair.split == args.exp_b_split]
    exp_c_pairs = [pair for pair in all_pairs if pair.split == args.exp_c_split]
    best_layer, _ = run_exp_a(
        model,
        tokenizer,
        tool_token_id,
        exp_a_pairs,
        exp_a_root,
        args.batch_size,
        model_path=args.model_path,
        dataset_root=args.dataset_root,
    )
    mu_delta = run_exp_b(
        model,
        tokenizer,
        tool_token_id,
        exp_b_pairs,
        best_layer,
        exp_b_root,
        args.batch_size,
        model_path=args.model_path,
        dataset_root=args.dataset_root,
    )
    run_exp_c(model, tokenizer, tool_token_id, exp_c_pairs, best_layer, mu_delta, exp_c_root, args.batch_size)
    write_text(
        run_root / "summary.md",
        "\n".join(
            [
                "# Devstral Mechanism Generalization",
                "",
                f"- run_tag: `{run_tag}`",
                f"- n_exp_a_pairs: `{len(exp_a_pairs)}`",
                f"- n_exp_b_pairs: `{len(exp_b_pairs)}`",
                f"- n_exp_c_pairs: `{len(exp_c_pairs)}`",
                f"- best layer: `L{best_layer}`",
                f"- exp A: `{exp_a_root}`",
                f"- exp B: `{exp_b_root}`",
                f"- exp C: `{exp_c_root}`",
            ]
        )
        + "\n",
    )
    write_text(args.output_root / "LATEST_RUN.txt", str(run_root.resolve()))


if __name__ == "__main__":
    main()
