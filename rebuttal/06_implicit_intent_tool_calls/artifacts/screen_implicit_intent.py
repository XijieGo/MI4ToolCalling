#!/usr/bin/env python3
"""Baseline-screen frozen implicit-intent prompts on Qwen3-8B.

For each prompt, run one forward pass and record whether the first generated
token's greedy top-1 is the start of ``<tool_call>``. This is the same
baseline-positive gate used for the tau2 natural-trajectory set: only items
that already call the tool at baseline are valid "removal" items.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from release_paths import DEFAULT_QWEN3_8B_SCREEN_ROOT, FROZEN_CANDIDATES, MODEL_PATHS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=FROZEN_CANDIDATES)
    parser.add_argument("--output", type=Path, default=DEFAULT_QWEN3_8B_SCREEN_ROOT / "screened.jsonl")
    parser.add_argument("--model-path", type=Path, default=MODEL_PATHS["qwen3_8b"])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists: {args.output}; pass --overwrite to replace it")
    rows = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line.strip()]
    tokenizer = AutoTokenizer.from_pretrained(str(args.model_path))
    kwargs: dict[str, object] = {"torch_dtype": torch.bfloat16}
    if args.device.startswith("cuda"):
        kwargs["device_map"] = {"": 0}
    model = AutoModelForCausalLM.from_pretrained(str(args.model_path), **kwargs)
    model.eval()
    device = next(model.parameters()).device

    tool_call_token = "<tool_call>"
    tool_call_ids = tokenizer.encode(tool_call_token, add_special_tokens=False)
    first_tool_call_id = tool_call_ids[0]

    results = []
    with torch.no_grad():
        for i, row in enumerate(rows):
            enc = tokenizer(row["prompt"], return_tensors="pt", add_special_tokens=False).to(device)
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

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
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
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
