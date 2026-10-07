#!/usr/bin/env python3
"""Evaluate downstream readout mechanism on Qwen3.5-4B (Sec 6.1, 6.2, Figure 3)."""

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

TOOL_CALL_TOKEN = "<tool_call>"
TOOL_CALL_ID = 248058
INTERVENTION_LAYER = 31  # pre L31 equals the archived post L30 vector location.
FULL_ATTN_LAYERS = [23, 27, 31]


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


def identify_spans(text: str, offset_mapping: list[tuple[int, int]]) -> dict[str, list[int]]:
    system_open = "<|im_start|>system\n"
    system_to_user = "<|im_end|>\n<|im_start|>user\n"
    assistant_marker = "<|im_end|>\n<|im_start|>assistant\n"
    tools_open = "<tools>\n"
    tools_close = "</tools>"

    t_start = text.find(tools_open)
    t_end = text.find(tools_close) + len(tools_close) if text.find(tools_close) >= 0 else t_start
    u_start = text.find(system_to_user) + len(system_to_user)
    a_start = text.find(assistant_marker)

    r_range = (len(system_open), t_start if t_start >= 0 else u_start)
    t_range = (t_start, t_end) if t_start >= 0 else (0, 0)
    f_range = (t_end, u_start - len(system_to_user)) if t_start >= 0 else (0, 0)
    u_range = (u_start, a_start if a_start >= 0 else len(text))

    spans = {"R": [], "T": [], "F": [], "U": []}
    for tok_idx, (start, end) in enumerate(offset_mapping):
        if start == end:
            continue
        if r_range[0] <= start < r_range[1]:
            spans["R"].append(tok_idx)
        elif t_range[0] <= start < t_range[1]:
            spans["T"].append(tok_idx)
        elif f_range[0] <= start < f_range[1]:
            spans["F"].append(tok_idx)
        elif u_range[0] <= start < u_range[1]:
            spans["U"].append(tok_idx)
    return spans


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, default=model_path("qwen35_4b"))
    parser.add_argument("--vector-path", type=Path, default=Path("results/transfer/qwen35_4b/coding_vector.pt"))
    parser.add_argument("--dataset-root", type=Path, default=Path("datasets/qwen35_4b/pair"))
    parser.add_argument("--output-root", type=Path, default=Path("results/qwen35_4b/downstream_readout"))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-pairs", type=int, default=0)
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)

    print(f"Loading coding vector from {args.vector_path}...")
    vec_data = torch.load(args.vector_path, map_location="cpu")
    mu_Delta = vec_data["mean_diff"].float() if isinstance(vec_data, dict) else vec_data.float()
    print(f"Loaded vector norm = {float(mu_Delta.norm().item()):.3f}")

    print(f"Loading Qwen3.5-4B from {args.model_path} with eager attention...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        trust_remote_code=True,
        attn_implementation="eager",
    )
    model.eval()
    device = next(model.parameters()).device

    heldout_rows = load_heldout_items(args.dataset_root)
    if args.max_pairs > 0:
        heldout_rows = heldout_rows[: args.max_pairs]
    print(f"Loaded {len(heldout_rows)} held-out pairs from {args.dataset_root}")

    # Track attention shifts for full attention layers
    clean_span_attention = {l: {"R": 0.0, "T": 0.0, "F": 0.0, "U": 0.0} for l in FULL_ATTN_LAYERS}
    corrupt_span_attention = {l: {"R": 0.0, "T": 0.0, "F": 0.0, "U": 0.0} for l in FULL_ATTN_LAYERS}

    interv_recovered = 0
    total_eval = len(heldout_rows)

    print("\nProcessing held-out pairs for attention readout and vector intervention...")
    for idx, row in enumerate(heldout_rows, 1):
        clean_text = (args.dataset_root / row["clean_relpath"]).read_text(encoding="utf-8")
        corrupt_text = (args.dataset_root / row["corrupt_relpath"]).read_text(encoding="utf-8")

        c_enc = tokenizer(clean_text, return_offsets_mapping=True, add_special_tokens=False)
        k_enc = tokenizer(corrupt_text, return_offsets_mapping=True, add_special_tokens=False)

        c_spans = identify_spans(clean_text, c_enc["offset_mapping"])
        k_spans = identify_spans(corrupt_text, k_enc["offset_mapping"])

        c_ids = torch.tensor([c_enc["input_ids"]], device=device)
        k_ids = torch.tensor([k_enc["input_ids"]], device=device)

        with torch.no_grad():
            c_out = model(input_ids=c_ids, output_attentions=True)
            k_out = model(input_ids=k_ids, output_attentions=True)

        FULL_ATTN_MAP = {23: 5, 27: 6, 31: 7}
        for l in FULL_ATTN_LAYERS:
            attn_idx = FULL_ATTN_MAP.get(l)
            if hasattr(c_out, "attentions") and c_out.attentions is not None and attn_idx is not None and attn_idx < len(c_out.attentions):
                c_mat = c_out.attentions[attn_idx][0, :, -1, :].mean(dim=0)  # [seq_len] mean across heads
                k_mat = k_out.attentions[attn_idx][0, :, -1, :].mean(dim=0)
                for s in ("R", "T", "F", "U"):
                    if c_spans[s]:
                        clean_span_attention[l][s] += float(c_mat[c_spans[s]].sum().item())
                    if k_spans[s]:
                        corrupt_span_attention[l][s] += float(k_mat[k_spans[s]].sum().item())

        # Vector intervention at INTERVENTION_LAYER
        def add_hook(module, args):
            h = args[0].clone()
            h[0, -1] = h[0, -1] + mu_Delta.to(device=h.device, dtype=h.dtype)
            return (h, *args[1:])

        handle = model.model.layers[INTERVENTION_LAYER].register_forward_pre_hook(add_hook)
        with torch.no_grad():
            out_interv = model(input_ids=k_ids)
        handle.remove()

        if out_interv.logits[0, -1].argmax().item() == TOOL_CALL_ID:
            interv_recovered += 1

        if idx % max(1, total_eval // 5) == 0:
            print(f"  Processed {idx}/{total_eval} pairs...", flush=True)

    # Average span attentions
    span_rows = []
    max_shift_pp = 0.0
    for l in FULL_ATTN_LAYERS:
        for s in ("R", "T", "F", "U"):
            c_att = clean_span_attention[l][s] / max(1, total_eval)
            k_att = corrupt_span_attention[l][s] / max(1, total_eval)
            shift = (c_att - k_att) * 100.0
            if abs(shift) > max_shift_pp:
                max_shift_pp = abs(shift)
            span_rows.append({
                "layer": l,
                "span": s,
                "clean_attention": round(c_att, 4),
                "corrupt_attention": round(k_att, 4),
                "shift_pp": round(shift, 2),
            })

    with open(args.output_root / "figure3b_attention_spans.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["layer", "span", "clean_attention", "corrupt_attention", "shift_pp"])
        writer.writeheader()
        writer.writerows(span_rows)

    # Recovery rate
    recovery_rate = interv_recovered / max(1, total_eval) * 100.0

    readout_summary = {
        "model": "Qwen3.5-4B",
        "n_heldout": total_eval,
        "max_attention_shift_pp": round(max_shift_pp, 1),
        "intervention_layer": INTERVENTION_LAYER,
        "vector_recovery_rate_pct": round(recovery_rate, 1),
    }
    with open(args.output_root / "feature_readout.json", "w", encoding="utf-8") as f:
        json.dump(readout_summary, f, indent=2)

    md_lines = [
        "# Downstream Readout Mechanism on Qwen3.5-4B (Section 6)",
        "",
        f"Evaluated on {total_eval} held-out pairs from `{args.dataset_root}`.",
        "",
        "## 1. Attention Redistribution in Full-Attention Layers",
        "",
        "| Layer | Span | Clean Attention | Corrupt Attention | Shift (pp) |",
        "|---:|:---|---:|---:|---:|",
    ]
    for r in span_rows:
        md_lines.append(f"| L{r['layer']} | {r['span']} | {r['clean_attention']} | {r['corrupt_attention']} | {r['shift_pp']:+.2f} |")

    md_lines.extend([
        "",
        f"- **Max Attention Shift**: `{max_shift_pp:.1f} pp`.",
        f"- **Vector Intervention at L{INTERVENTION_LAYER}**: Recovers `<tool_call>` top-1 on **{recovery_rate:.1f}%** of held-out analysis prompts.",
    ])
    md_content = "\n".join(md_lines) + "\n"
    print("\n" + md_content)

    (args.output_root / "summary.md").write_text(md_content, encoding="utf-8")
    print(f"Results written to {args.output_root}")


if __name__ == "__main__":
    main()
