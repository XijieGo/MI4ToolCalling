#!/usr/bin/env python3
"""Measure the fixed-layer residual write trajectory on Qwen3.5-9B."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from mi4tc.paths import model_path  # noqa: E402

from transformers import AutoModelForCausalLM, AutoTokenizer

COMMITMENT_LAYER = 31
FORMATION_LAYERS = [27, 28, 29, 30]


def load_heldout_items(dataset_root: Path) -> list[dict[str, Any]]:
    manifest_path = dataset_root / "manifest.jsonl"
    if not manifest_path.is_file():
        manifest_path = dataset_root / "pairs.jsonl"
    items = []
    with open(manifest_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("split", "heldout") == "heldout":
                items.append(row)
    return items


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, default=model_path("qwen35_9b"))
    parser.add_argument("--vector-path", type=Path, default=Path("results/transfer/qwen35_9b/coding_vector.pt"))
    parser.add_argument("--dataset-root", type=Path, default=Path("datasets/qwen35_9b/pair"))
    parser.add_argument("--output-root", type=Path, default=Path("results/qwen35_9b/formation_transcoder"))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-pairs", type=int, default=0)
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)

    print(f"Loading coding vector from {args.vector_path}...")
    vec_data = torch.load(args.vector_path, map_location="cpu")
    mu_Delta = vec_data["mean_diff"].float() if isinstance(vec_data, dict) else vec_data.float()
    mu_norm = float(mu_Delta.norm().item())
    u_Delta = mu_Delta / mu_norm
    print(f"Loaded vector norm = {mu_norm:.3f}")

    print(f"Loading Qwen3.5-9B from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        trust_remote_code=True,
    )
    model.eval()
    device = next(model.parameters()).device

    heldout_rows = load_heldout_items(args.dataset_root)
    if args.max_pairs > 0:
        heldout_rows = heldout_rows[: args.max_pairs]
    print(f"Loaded {len(heldout_rows)} held-out pairs from {args.dataset_root}")

    num_layers = len(model.model.layers)
    print(f"Sweeping layers 0..{COMMITMENT_LAYER} on {len(heldout_rows)} pairs...")

    layer_clean_writes = [0.0] * num_layers
    layer_corrupt_writes = [0.0] * num_layers

    for i in range(0, len(heldout_rows), args.batch_size):
        batch = heldout_rows[i : i + args.batch_size]
        clean_texts = [(args.dataset_root / r["clean_relpath"]).read_text(encoding="utf-8") for r in batch]
        corrupt_texts = [(args.dataset_root / r["corrupt_relpath"]).read_text(encoding="utf-8") for r in batch]

        def get_layer_projections(texts: list[str]) -> list[list[float]]:
            enc = tokenizer(texts, padding=True, return_tensors="pt")
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)

            layer_outs = {}
            handles = []
            for l_idx in range(COMMITMENT_LAYER + 1):
                def make_hook(idx: int):
                    def hook(module, args, output):
                        h = output[0] if isinstance(output, tuple) else output
                        proj = (h[:, -1, :].float() @ u_Delta.to(device=h.device)).detach().cpu().tolist()
                        layer_outs[idx] = proj
                    return hook
                handles.append(model.model.layers[l_idx].register_forward_hook(make_hook(l_idx)))

            with torch.no_grad():
                model(input_ids=input_ids, attention_mask=attention_mask)
            for h in handles:
                h.remove()

            return [layer_outs[l] for l in range(COMMITMENT_LAYER + 1)]

        clean_projs = get_layer_projections(clean_texts)
        corrupt_projs = get_layer_projections(corrupt_texts)

        for l_idx in range(COMMITMENT_LAYER + 1):
            layer_clean_writes[l_idx] += sum(clean_projs[l_idx])
            layer_corrupt_writes[l_idx] += sum(corrupt_projs[l_idx])

        if (i // args.batch_size) % max(1, (len(heldout_rows) // args.batch_size) // 4) == 0:
            print(f"  Processed {min(i + args.batch_size, len(heldout_rows))}/{len(heldout_rows)} pairs...", flush=True)

    n = len(heldout_rows)
    mean_clean_projs = [s / n for s in layer_clean_writes[:COMMITMENT_LAYER + 1]]
    mean_corrupt_projs = [s / n for s in layer_corrupt_writes[:COMMITMENT_LAYER + 1]]

    trajectory_rows = []
    prev_c, prev_k = 0.0, 0.0
    for l_idx in range(COMMITMENT_LAYER + 1):
        c_val = mean_clean_projs[l_idx]
        k_val = mean_corrupt_projs[l_idx]
        diff = c_val - k_val
        clean_delta = c_val - prev_c
        corrupt_delta = k_val - prev_k
        diff_delta = clean_delta - corrupt_delta
        prev_c, prev_k = c_val, k_val
        trajectory_rows.append({
            "layer": l_idx,
            "clean_proj": round(c_val, 4),
            "corrupt_proj": round(k_val, 4),
            "diff_proj": round(diff, 4),
            "clean_write": round(clean_delta, 4),
            "corrupt_write": round(corrupt_delta, 4),
            "diff_write": round(diff_delta, 4),
        })

    with open(args.output_root / "figure2_formation_trajectory.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["layer", "clean_proj", "corrupt_proj", "diff_proj", "clean_write", "corrupt_write", "diff_write"])
        writer.writeheader()
        writer.writerows(trajectory_rows)

    formation_write_sum = sum(trajectory_rows[l]["diff_write"] for l in FORMATION_LAYERS)
    print(f"\n=== Formation Window (L{FORMATION_LAYERS[0]}-L{FORMATION_LAYERS[-1]}) Writes Summary ===")
    print(f"  Total write along u_Delta: {formation_write_sum:.2f}")

    summary = {
        "model": "Qwen3.5-9B",
        "n_heldout": len(heldout_rows),
        "commitment_layer": COMMITMENT_LAYER,
        "formation_layers": FORMATION_LAYERS,
        "formation_window_write_sum": formation_write_sum,
    }
    with open(args.output_root / "formation_transcoder_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Results written to {args.output_root}")


if __name__ == "__main__":
    main()
