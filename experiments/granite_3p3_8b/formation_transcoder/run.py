#!/usr/bin/env python3
"""Measure fixed-layer MLP and attention writes on Granite-3.3-8B."""

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

COMMITMENT_LAYER = 35
FORMATION_LAYERS = [31, 32, 33, 34]


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
    parser.add_argument("--model-path", type=Path, default=model_path("granite"))
    parser.add_argument("--vector-path", type=Path, default=Path("results/transfer/granite_3p3_8b/coding_vector.pt"))
    parser.add_argument("--dataset-root", type=Path, default=Path("datasets/granite_3p3_8b/pair"))
    parser.add_argument("--output-root", type=Path, default=Path("results/granite_3p3_8b/formation_transcoder"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-pairs", type=int, default=0)
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)

    print(f"Loading coding vector from {args.vector_path}...")
    vec_data = torch.load(args.vector_path, map_location="cpu")
    mu_Delta = vec_data["mean_diff"].float() if isinstance(vec_data, dict) else vec_data.float()
    mu_norm = float(mu_Delta.norm().item())
    u_Delta = mu_Delta / mu_norm
    print(f"Loaded vector norm = {mu_norm:.3f}")

    print(f"Loading Granite-3.3-8B from {args.model_path}...")
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

    mlp_writes = {l: 0.0 for l in range(COMMITMENT_LAYER + 1)}
    attn_writes = {l: 0.0 for l in range(COMMITMENT_LAYER + 1)}

    for i in range(0, len(heldout_rows), args.batch_size):
        batch = heldout_rows[i : i + args.batch_size]
        clean_texts = [(args.dataset_root / r["clean_relpath"]).read_text(encoding="utf-8") for r in batch]
        corrupt_texts = [(args.dataset_root / r["corrupt_relpath"]).read_text(encoding="utf-8") for r in batch]

        def get_component_writes(texts: list[str]) -> tuple[dict[int, list[float]], dict[int, list[float]]]:
            enc = tokenizer(texts, padding=True, return_tensors="pt")
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)

            cur_mlp_outs = {}
            cur_attn_outs = {}
            handles = []

            for l_idx in range(COMMITMENT_LAYER + 1):
                def make_mlp_hook(idx: int):
                    def hook(module, args, output):
                        h = output[0] if isinstance(output, tuple) else output
                        proj = (h[:, -1, :].float() @ u_Delta.to(device=h.device)).detach().cpu().tolist()
                        cur_mlp_outs[idx] = proj
                    return hook

                def make_attn_hook(idx: int):
                    def hook(module, args, output):
                        h = output[0] if isinstance(output, tuple) else output
                        proj = (h[:, -1, :].float() @ u_Delta.to(device=h.device)).detach().cpu().tolist()
                        cur_attn_outs[idx] = proj
                    return hook

                handles.append(model.model.layers[l_idx].mlp.register_forward_hook(make_mlp_hook(l_idx)))
                handles.append(model.model.layers[l_idx].self_attn.register_forward_hook(make_attn_hook(l_idx)))

            with torch.no_grad():
                model(input_ids=input_ids, attention_mask=attention_mask)
            for h in handles:
                h.remove()

            return cur_mlp_outs, cur_attn_outs

        c_mlp, c_attn = get_component_writes(clean_texts)
        k_mlp, k_attn = get_component_writes(corrupt_texts)

        for l_idx in range(COMMITMENT_LAYER + 1):
            mlp_writes[l_idx] += sum(c - k for c, k in zip(c_mlp[l_idx], k_mlp[l_idx]))
            attn_writes[l_idx] += sum(c - k for c, k in zip(c_attn[l_idx], k_attn[l_idx]))

        if (i // args.batch_size) % max(1, (len(heldout_rows) // args.batch_size) // 4) == 0:
            print(f"  Processed {min(i + args.batch_size, len(heldout_rows))}/{len(heldout_rows)} pairs...", flush=True)

    n = len(heldout_rows)
    mean_mlp_writes = {l: mlp_writes[l] / n for l in range(COMMITMENT_LAYER + 1)}
    mean_attn_writes = {l: attn_writes[l] / n for l in range(COMMITMENT_LAYER + 1)}

    trajectory_rows = []
    for l_idx in range(COMMITMENT_LAYER + 1):
        mw = mean_mlp_writes[l_idx]
        aw = mean_attn_writes[l_idx]
        trajectory_rows.append({
            "layer": l_idx,
            "mlp_write": round(mw, 4),
            "attn_write": round(aw, 4),
            "total_write": round(mw + aw, 4),
        })

    with open(args.output_root / "figure2_formation_trajectory.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["layer", "mlp_write", "attn_write", "total_write"])
        writer.writeheader()
        writer.writerows(trajectory_rows)

    tot_mlp_form = sum(mean_mlp_writes[l] for l in FORMATION_LAYERS)
    tot_attn_form = sum(mean_attn_writes[l] for l in FORMATION_LAYERS)
    mlp_attn_ratio = tot_mlp_form / tot_attn_form if abs(tot_attn_form) > 1e-12 else None
    print(f"\n=== Formation Window (L{FORMATION_LAYERS[0]}-L{FORMATION_LAYERS[-1]}) Writes Summary ===")
    print(f"  Total MLP write: {tot_mlp_form:.2f}")
    print(f"  Total Attn write: {tot_attn_form:.2f}")
    print(f"  MLP/Attn ratio: {mlp_attn_ratio}")

    summary = {
        "model": "Granite-3.3-8B",
        "n_heldout": len(heldout_rows),
        "commitment_layer": COMMITMENT_LAYER,
        "formation_layers": FORMATION_LAYERS,
        "formation_mlp_write": tot_mlp_form,
        "formation_attn_write": tot_attn_form,
        "mlp_over_attn_ratio": mlp_attn_ratio,
    }
    with open(args.output_root / "formation_transcoder_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Results written to {args.output_root}")


if __name__ == "__main__":
    main()
