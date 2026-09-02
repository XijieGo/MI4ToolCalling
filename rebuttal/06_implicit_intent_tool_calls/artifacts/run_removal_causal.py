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

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from release_paths import (
    DEFAULT_QWEN3_8B_REMOVAL_ROOT,
    FROZEN_QWEN3_8B_REMOVAL_ARM,
    MODEL_PATHS,
    VECTOR_PATHS,
)


ALPHAS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0)
SEED = 20260726


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=FROZEN_QWEN3_8B_REMOVAL_ARM)
    parser.add_argument("--output", type=Path, default=DEFAULT_QWEN3_8B_REMOVAL_ROOT / "removal_results.jsonl")
    parser.add_argument("--model-path", type=Path, default=MODEL_PATHS["qwen3_8b"])
    parser.add_argument("--vector-path", type=Path, default=VECTOR_PATHS["qwen3_8b"])
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

    bundle = torch.load(args.vector_path, map_location="cpu", weights_only=False)
    mean_diff = bundle["mean_diff"].to(torch.float32)
    layer = bundle.get("layer") or bundle.get("patch_layer")
    if layer is None or str(bundle.get("hook_kind", "pre")) != "pre":
        raise ValueError(f"Expected a pre-hook vector with layer metadata: {args.vector_path}")

    g = torch.Generator().manual_seed(SEED)
    random_dir = torch.randn(mean_diff.shape, generator=g)
    random_dir = random_dir / random_dir.norm() * mean_diff.norm()

    tool_call_ids = tokenizer.encode("<tool_call>", add_special_tokens=False)
    first_tool_call_id = tool_call_ids[0]

    target_layer = model.model.layers[int(layer)]

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
            enc = tokenizer(row["prompt"], return_tensors="pt", add_special_tokens=False).to(device)
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

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
