#!/usr/bin/env python3
"""Removal/necessity causal test on the 161-item implicit-intent set.

For every baseline-positive item (top-1 == <tool_call> with no intervention),
add -alpha * mean_diff to the final prompt token at layer 24's pre-hook
(the paper's frozen D1 vector, same one used in the tau2 natural-trajectory
and reasoning-trace probes) and record whether the top-1 token flips away
from <tool_call>. A norm-matched random direction at the same hook/position
is the control.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = "/root/autodl-tmp/Qwen/Qwen3-8B"
VECTOR_PATH = "./results/runs/rebuttal_scaffold_components_d1_v4_full_20260725/native_vectors/RTF_L24_pre.pt"
LAYER = 24
OUT_DIR = Path(__file__).resolve().parent
IN_PATH = OUT_DIR / "implicit_intent_removal_final.jsonl"
OUT_PATH = OUT_DIR / "implicit_intent_removal_results.jsonl"
ALPHAS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0)
SEED = 20260726


def main() -> None:
    rows = [json.loads(l) for l in open(IN_PATH)]
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, dtype=torch.bfloat16, device_map="cuda")
    model.eval()

    bundle = torch.load(VECTOR_PATH, map_location="cpu")
    mean_diff = bundle["mean_diff"].to(torch.float32)
    assert bundle["layer"] == LAYER and bundle["hook_kind"] == "pre"

    g = torch.Generator().manual_seed(SEED)
    random_dir = torch.randn(mean_diff.shape, generator=g)
    random_dir = random_dir / random_dir.norm() * mean_diff.norm()

    tool_call_ids = tokenizer.encode("<tool_call>", add_special_tokens=False)
    first_tool_call_id = tool_call_ids[0]

    target_layer = model.model.layers[LAYER]

    current_delta = {"value": None}

    def hook(module, args):
        hidden_states = args[0]
        if current_delta["value"] is None:
            return None
        delta = current_delta["value"].to(device=hidden_states.device, dtype=hidden_states.dtype).view(1, 1, -1)
        patched = hidden_states.clone()
        patched[:, -1:, :] = patched[:, -1:, :] + delta
        return (patched, *args[1:])

    handle = target_layer.register_forward_pre_hook(hook)

    results = []
    with torch.no_grad():
        for i, row in enumerate(rows):
            enc = tokenizer(row["prompt"], return_tensors="pt", add_special_tokens=False).to("cuda")
            row_result = {k: row[k] for k in ("domain", "pattern", "source_id", "item_id")}
            row_result["conditions"] = {}
            for direction_name, direction in (("mean_diff", mean_diff), ("random", random_dir)):
                for alpha in ALPHAS:
                    if alpha == 0.0 and direction_name == "random":
                        continue  # alpha=0 is identical to mean_diff's alpha=0; skip duplicate
                    current_delta["value"] = -alpha * direction
                    out = model(**enc)
                    logits = out.logits[0, -1, :]
                    top1_id = int(torch.argmax(logits).item())
                    is_call = top1_id == first_tool_call_id
                    key = f"{direction_name}_a{alpha}"
                    row_result["conditions"][key] = {
                        "top1_str": tokenizer.decode([top1_id]),
                        "is_tool_call_top1": is_call,
                    }
            current_delta["value"] = None
            results.append(row_result)
            if (i + 1) % 10 == 0:
                print(f"{i+1}/{len(rows)}")

    handle.remove()

    with OUT_PATH.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
