#!/usr/bin/env python3
"""Baseline-screen the 200 implicit-intent prompts on Qwen3-8B.

For each prompt, run one forward pass and record whether the first generated
token's greedy top-1 is the start of ``<tool_call>``. This is the same
baseline-positive gate used for the tau2 natural-trajectory set: only items
that already call the tool at baseline are valid "removal" items.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = "/root/autodl-tmp/Qwen/Qwen3-8B"
OUT_DIR = Path(__file__).resolve().parent
IN_PATH = OUT_DIR / "implicit_intent_oversampled_600.jsonl"
OUT_PATH = OUT_DIR / "implicit_intent_oversampled_600_screened.jsonl"


def main() -> None:
    rows = [json.loads(l) for l in open(IN_PATH)]
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, device_map="cuda"
    )
    model.eval()

    tool_call_token = "<tool_call>"
    tool_call_ids = tokenizer.encode(tool_call_token, add_special_tokens=False)
    first_tool_call_id = tool_call_ids[0]

    results = []
    with torch.no_grad():
        for i, row in enumerate(rows):
            enc = tokenizer(row["prompt"], return_tensors="pt", add_special_tokens=False).to("cuda")
            out = model(**enc)
            logits = out.logits[0, -1, :]
            probs = torch.softmax(logits.float(), dim=-1)
            top1_id = int(torch.argmax(logits).item())
            top1_str = tokenizer.decode([top1_id])
            rank = int((logits > logits[first_tool_call_id]).sum().item()) + 1
            p_tool_call = float(probs[first_tool_call_id].item())
            is_call = top1_id == first_tool_call_id
            row_out = dict(row)
            row_out.update({
                "baseline_top1_id": top1_id,
                "baseline_top1_str": top1_str,
                "baseline_is_tool_call_top1": is_call,
                "baseline_tool_call_rank": rank,
                "baseline_tool_call_prob": p_tool_call,
            })
            results.append(row_out)
            if (i + 1) % 20 == 0:
                print(f"{i+1}/{len(rows)}")

    with OUT_PATH.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    by_domain = {}
    for r in results:
        d = by_domain.setdefault(r["domain"], {"n": 0, "call": 0})
        d["n"] += 1
        d["call"] += int(r["baseline_is_tool_call_top1"])
    for d, v in sorted(by_domain.items()):
        print(f"{d}: {v['call']}/{v['n']} baseline top-1 = <tool_call>")
    total_call = sum(v["call"] for v in by_domain.values())
    total_n = sum(v["n"] for v in by_domain.values())
    print(f"TOTAL: {total_call}/{total_n}")


if __name__ == "__main__":
    main()
