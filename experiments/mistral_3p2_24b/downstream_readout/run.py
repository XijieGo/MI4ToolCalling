#!/usr/bin/env python3
"""Evaluate downstream readout mechanism on Mistral-3.2-24B (Sec 6.1, 6.2, Figure 3)."""

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

from transformers import AutoTokenizer, Mistral3ForConditionalGeneration

TOOL_CALL_TOKEN = "[TOOL_CALLS]"
TOOL_CALL_ID = 9
INTERVENTION_LAYER = 26
READER_HEAD = (20, 19)
KEY_COMPONENT = (25, 27)


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
    prefix = "<s>"
    role_open = "[SYSTEM_PROMPT]"
    role_close = "[/SYSTEM_PROMPT]"
    tools_open = "[AVAILABLE_TOOLS]"
    tools_close = "[/AVAILABLE_TOOLS]"
    user_open = "[INST]"
    user_close = "[/INST]"

    r_start = text.find(role_open) + len(role_open) if text.find(role_open) >= 0 else 0
    r_end = text.find(role_close) if text.find(role_close) >= 0 else 0

    t_start = text.find(tools_open) + len(tools_open) if text.find(tools_open) >= 0 else 0
    t_end = text.find(tools_close) if text.find(tools_close) >= 0 else 0

    u_start = text.find(user_open) + len(user_open) if text.find(user_open) >= 0 else 0
    u_end = text.find(user_close) if text.find(user_close) >= 0 else len(text)

    r_range = (r_start, r_end)
    t_range = (t_start, t_end)
    u_range = (u_start, u_end)

    spans = {"R": [], "T": [], "U": []}
    for tok_idx, (start, end) in enumerate(offset_mapping):
        if start == end:
            continue
        if r_range[0] <= start < r_range[1]:
            spans["R"].append(tok_idx)
        elif t_range[0] <= start < t_range[1]:
            spans["T"].append(tok_idx)
        elif u_range[0] <= start < u_range[1]:
            spans["U"].append(tok_idx)
    return spans


def get_token_offsets(tokenizer, input_ids: list[int]) -> list[tuple[int, int]]:
    tokens = [tokenizer.decode([tid]) for tid in input_ids]
    offsets = []
    curr = 0
    for t in tokens:
        offsets.append((curr, curr + len(t)))
        curr += len(t)
    return offsets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, default=Path("/root/autodl-tmp/Mistral-Small-3.2-24B-Instruct-2506"))
    parser.add_argument("--vector-path", type=Path, default=Path("results/transfer/mistral_3p2_24b/coding_vector.pt"))
    parser.add_argument("--dataset-root", type=Path, default=Path("datasets/mistral_3p2_24b/pair"))
    parser.add_argument("--output-root", type=Path, default=Path("results/mistral_3p2_24b/downstream_readout"))
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-pairs", type=int, default=0)
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)

    print(f"Loading coding vector from {args.vector_path}...")
    vec_data = torch.load(args.vector_path, map_location="cpu")
    mu_Delta = vec_data["mean_diff"].float() if isinstance(vec_data, dict) else vec_data.float()
    print(f"Loaded vector norm = {float(mu_Delta.norm().item()):.3f}")

    print(f"Loading Mistral-3.2-24B from {args.model_path} with eager attention...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    model = Mistral3ForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        trust_remote_code=True,
        attn_implementation="eager",
    )
    model.eval()
    device = next(model.parameters()).device
    layers = model.model.language_model.layers if hasattr(model.model, "language_model") else model.model.layers

    heldout_rows = load_heldout_items(args.dataset_root)
    if args.max_pairs > 0:
        heldout_rows = heldout_rows[: args.max_pairs]
    print(f"Loaded {len(heldout_rows)} held-out pairs from {args.dataset_root}")

    # Head projection for DLA
    lm_head_weight = model.lm_head.weight[TOOL_CALL_ID].detach().float()  # [D]

    head_dim = model.config.text_config.head_dim if hasattr(model.config, "text_config") else (model.config.hidden_size // model.config.num_attention_heads)
    num_heads = model.config.text_config.num_attention_heads if hasattr(model.config, "text_config") else model.config.num_attention_heads

    eval_layers = [20, 25, 26, 30, 35]
    head_projections = {}
    for l in eval_layers:
        o_weight = layers[l].self_attn.o_proj.weight.detach().float()  # [D, D]
        for h in range(num_heads):
            W_h = o_weight[:, h * head_dim : (h + 1) * head_dim]
            head_proj = W_h.T @ lm_head_weight
            head_projections[(l, h)] = head_proj.to(device)

    clean_dla_sums = {(l, h): 0.0 for (l, h) in head_projections}
    corrupt_dla_sums = {(l, h): 0.0 for (l, h) in head_projections}

    reader_clean_spans = {"R": 0.0, "T": 0.0, "U": 0.0}
    reader_corrupt_spans = {"R": 0.0, "T": 0.0, "U": 0.0}

    interv_recovered = 0
    total_eval = len(heldout_rows)

    print("\nProcessing held-out pairs for DLA, attention spans, and vector intervention...")
    for idx, row in enumerate(heldout_rows, 1):
        clean_json = json.loads((args.dataset_root / row["clean_relpath"]).read_text(encoding="utf-8"))
        corrupt_json = json.loads((args.dataset_root / row["corrupt_relpath"]).read_text(encoding="utf-8"))

        c_ids_list = clean_json["input_ids"]
        k_ids_list = corrupt_json["input_ids"]

        clean_text = tokenizer.decode(c_ids_list)
        corrupt_text = tokenizer.decode(k_ids_list)

        c_offsets = get_token_offsets(tokenizer, c_ids_list)
        k_offsets = get_token_offsets(tokenizer, k_ids_list)

        c_spans = identify_spans(clean_text, c_offsets)
        k_spans = identify_spans(corrupt_text, k_offsets)

        c_ids = torch.tensor([c_ids_list], device=device)
        k_ids = torch.tensor([k_ids_list], device=device)

        c_head_outs = {}
        k_head_outs = {}

        handles = []
        for l in eval_layers:
            def make_pre_hook(l_idx: int, target_dict: dict):
                def hook(module, args):
                    z = args[0]
                    for h in range(num_heads):
                        target_dict[(l_idx, h)] = z[0, -1, h * head_dim : (h + 1) * head_dim].detach()
                return hook
            handles.append(layers[l].self_attn.o_proj.register_forward_pre_hook(make_pre_hook(l, c_head_outs)))

        with torch.no_grad():
            c_out = model(input_ids=c_ids, output_attentions=True)
        for h in handles:
            h.remove()

        handles = []
        for l in eval_layers:
            def make_pre_hook_k(l_idx: int, target_dict: dict):
                def hook(module, args):
                    z = args[0]
                    for h in range(num_heads):
                        target_dict[(l_idx, h)] = z[0, -1, h * head_dim : (h + 1) * head_dim].detach()
                return hook
            handles.append(layers[l].self_attn.o_proj.register_forward_pre_hook(make_pre_hook_k(l, k_head_outs)))

        with torch.no_grad():
            k_out = model(input_ids=k_ids, output_attentions=True)
        for h in handles:
            h.remove()

        for (l, h) in head_projections:
            if (l, h) in c_head_outs and (l, h) in k_head_outs:
                clean_dla_sums[(l, h)] += float((c_head_outs[(l, h)].float() @ head_projections[(l, h)]).item())
                corrupt_dla_sums[(l, h)] += float((k_head_outs[(l, h)].float() @ head_projections[(l, h)]).item())

        rl, rh = READER_HEAD
        if hasattr(c_out, "attentions") and c_out.attentions is not None:
            c_mat = c_out.attentions[rl][0, rh, -1, :]
            k_mat = k_out.attentions[rl][0, rh, -1, :]
            for s in ("R", "T", "U"):
                if c_spans[s]:
                    reader_clean_spans[s] += float(c_mat[c_spans[s]].sum().item())
                if k_spans[s]:
                    reader_corrupt_spans[s] += float(k_mat[k_spans[s]].sum().item())

        def add_hook(module, args):
            h = args[0].clone()
            h[0, -1] = h[0, -1] + mu_Delta.to(device=h.device, dtype=h.dtype)
            return (h, *args[1:])

        handle = layers[INTERVENTION_LAYER].register_forward_pre_hook(add_hook)
        with torch.no_grad():
            out_interv = model(input_ids=k_ids)
        handle.remove()

        if out_interv.logits[0, -1].argmax().item() == TOOL_CALL_ID:
            interv_recovered += 1

        if idx % max(1, total_eval // 5) == 0:
            print(f"  Processed {idx}/{total_eval} pairs...", flush=True)

    dla_rows = []
    for (l, h) in head_projections:
        c_dla = clean_dla_sums[(l, h)] / max(1, total_eval)
        k_dla = corrupt_dla_sums[(l, h)] / max(1, total_eval)
        delta_dla = c_dla - k_dla
        dla_rows.append({
            "head": f"L{l}H{h}",
            "clean_dla": round(c_dla, 3),
            "corrupt_dla": round(k_dla, 3),
            "delta_dla": round(delta_dla, 3),
        })
    dla_rows.sort(key=lambda r: r["delta_dla"], reverse=True)

    with open(args.output_root / "figure3a_attention_head_dla.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["head", "clean_dla", "corrupt_dla", "delta_dla"])
        writer.writeheader()
        writer.writerows(dla_rows)

    span_rows = []
    max_shift_pp = 0.0
    for s in ("R", "T", "U"):
        c_att = reader_clean_spans[s] / max(1, total_eval)
        k_att = reader_corrupt_spans[s] / max(1, total_eval)
        shift = (c_att - k_att) * 100.0
        if abs(shift) > max_shift_pp:
            max_shift_pp = abs(shift)
        span_rows.append({
            "head": f"L{READER_HEAD[0]}H{READER_HEAD[1]}",
            "span": s,
            "clean_attention": round(c_att, 4),
            "corrupt_attention": round(k_att, 4),
            "shift_pp": round(shift, 2),
        })

    with open(args.output_root / "figure3b_attention_spans.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["head", "span", "clean_attention", "corrupt_attention", "shift_pp"])
        writer.writeheader()
        writer.writerows(span_rows)

    recovery_rate = interv_recovered / max(1, total_eval) * 100.0

    readout_summary = {
        "model": "Mistral-3.2-24B",
        "n_heldout": total_eval,
        "reader_head": f"L{READER_HEAD[0]}H{READER_HEAD[1]}",
        "key_component": f"L{KEY_COMPONENT[0]}H{KEY_COMPONENT[1]}",
        "max_attention_shift_pp": round(max_shift_pp, 1),
        "table6_reported_max_attn_pp": 8.8,
        "intervention_layer": INTERVENTION_LAYER,
        "vector_recovery_rate_pct": round(recovery_rate, 1),
    }
    with open(args.output_root / "feature_readout.json", "w", encoding="utf-8") as f:
        json.dump(readout_summary, f, indent=2)

    md_lines = [
        "# Downstream Readout Mechanism on Mistral-3.2-24B (Section 6)",
        "",
        f"Evaluated on {total_eval} held-out pairs from `{args.dataset_root}`.",
        "",
        f"## 1. Scaffold-Reading Head L{READER_HEAD[0]}H{READER_HEAD[1]} Attention",
        "",
        "| Head | Span | Clean Attention | Corrupt Attention | Shift (pp) |",
        "|:---|:---|---:|---:|---:|",
    ]
    for r in span_rows:
        md_lines.append(f"| {r['head']} | {r['span']} | {r['clean_attention']} | {r['corrupt_attention']} | {r['shift_pp']:+.2f} |")

    md_lines.extend([
        "",
        f"- **Max Attention Shift**: `{max_shift_pp:.1f} pp` (Table 6 reports `+8.8 pp`).",
        f"- **Top Head by $\\Delta$ DLA**: `{dla_rows[0]['head']}` ($\\Delta$ DLA = `+{dla_rows[0]['delta_dla']}`).",
        f"- **Vector Intervention at L{INTERVENTION_LAYER}**: Recovers `[TOOL_CALLS]` top-1 on **{recovery_rate:.1f}%** of held-out analysis prompts.",
    ])
    md_content = "\n".join(md_lines) + "\n"
    print("\n" + md_content)

    (args.output_root / "summary.md").write_text(md_content, encoding="utf-8")
    print(f"Results written to {args.output_root}")


if __name__ == "__main__":
    main()
