#!/usr/bin/env python3
"""Run deterministic first-token behavioral screening for candidate pairs.

This script performs no semantic judging and no text generation.  It records
whether a target model itself produces the required first-token call/no-call
contrast for each already rule-valid candidate pair.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .common import read_jsonl, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--model-label", type=str, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--max-candidates", type=int, default=0, help="0 means all candidates")
    return parser.parse_args()


def dtype_from_name(name: str) -> torch.dtype:
    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[name]


def chunks(items: list[dict[str, Any]], size: int) -> Iterable[list[dict[str, Any]]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def token_alignment(tokenizer: Any, clean_prompt: str, corrupt_prompt: str) -> tuple[bool, int, list[int]]:
    clean_ids = tokenizer(clean_prompt, add_special_tokens=False)["input_ids"]
    corrupt_ids = tokenizer(corrupt_prompt, add_special_tokens=False)["input_ids"]
    if len(clean_ids) != len(corrupt_ids):
        return False, max(len(clean_ids), len(corrupt_ids)), []
    differences = [index for index, (left, right) in enumerate(zip(clean_ids, corrupt_ids)) if left != right]
    return len(differences) == 1, len(clean_ids), differences


def main() -> None:
    args = parse_args()
    candidates = list(read_jsonl(args.candidates.resolve()))
    if args.max_candidates > 0:
        candidates = candidates[: args.max_candidates]
    if not candidates:
        raise ValueError("No candidates to screen")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tool_id = int(tokenizer.convert_tokens_to_ids("<tool_call>"))
    if tool_id < 0 or tool_id == tokenizer.unk_token_id:
        raise ValueError(f"{args.model_label} lacks a dedicated <tool_call> token")

    alignment: dict[str, tuple[bool, int, list[int]]] = {
        row["candidate_id"]: token_alignment(tokenizer, row["clean_prompt"], row["corrupt_prompt"])
        for row in candidates
    }
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=dtype_from_name(args.dtype),
        trust_remote_code=True,
    )
    model.to(args.device)
    model.eval()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    valid_count = 0
    with args.output.open("w", encoding="utf-8") as handle, torch.inference_mode():
        for batch_index, batch in enumerate(chunks(candidates, args.batch_size), start=1):
            clean_encoded = tokenizer(
                [row["clean_prompt"] for row in batch],
                return_tensors="pt",
                padding=True,
                add_special_tokens=False,
            )
            corrupt_encoded = tokenizer(
                [row["corrupt_prompt"] for row in batch],
                return_tensors="pt",
                padding=True,
                add_special_tokens=False,
            )
            clean_encoded = {key: value.to(args.device) for key, value in clean_encoded.items()}
            corrupt_encoded = {key: value.to(args.device) for key, value in corrupt_encoded.items()}
            # Qwen3 supports ``logits_to_keep``.  We need only the decision
            # position, so avoid materializing logits for every prompt token;
            # this keeps the all-scale intersection screen practical on the
            # available GPU without changing the measured first-token rule.
            clean_logits = model(**clean_encoded, use_cache=False, logits_to_keep=1).logits[:, -1, :].float()
            corrupt_logits = model(**corrupt_encoded, use_cache=False, logits_to_keep=1).logits[:, -1, :].float()
            clean_top = clean_logits.argmax(dim=-1)
            corrupt_top = corrupt_logits.argmax(dim=-1)
            clean_tool = clean_logits[:, tool_id]
            corrupt_tool = corrupt_logits[:, tool_id]
            clean_without_tool = clean_logits.clone()
            corrupt_without_tool = corrupt_logits.clone()
            clean_without_tool[:, tool_id] = float("-inf")
            corrupt_without_tool[:, tool_id] = float("-inf")
            clean_best_other = clean_without_tool.max(dim=-1).values
            corrupt_best_other = corrupt_without_tool.max(dim=-1).values
            for offset, row in enumerate(batch):
                token_aligned, token_length, differing_positions = alignment[row["candidate_id"]]
                clean_is_call = int(clean_top[offset].item()) == tool_id
                corrupt_is_call = int(corrupt_top[offset].item()) == tool_id
                behavior_valid = token_aligned and clean_is_call and not corrupt_is_call
                valid_count += int(behavior_valid)
                payload = {
                    "candidate_id": row["candidate_id"],
                    "domain": row["domain"],
                    "source_id": row["source_id"],
                    "clean_verb": row["clean_verb"],
                    "corrupt_verb": row["corrupt_verb"],
                    "token_aligned": token_aligned,
                    "token_length": token_length,
                    "differing_token_positions": differing_positions,
                    "clean_is_tool_call_top1": clean_is_call,
                    "corrupt_is_tool_call_top1": corrupt_is_call,
                    "behavior_valid": behavior_valid,
                    "clean_top1_token_id": int(clean_top[offset].item()),
                    "corrupt_top1_token_id": int(corrupt_top[offset].item()),
                    "clean_top1_token_text": tokenizer.decode([int(clean_top[offset].item())], clean_up_tokenization_spaces=False),
                    "corrupt_top1_token_text": tokenizer.decode([int(corrupt_top[offset].item())], clean_up_tokenization_spaces=False),
                    "clean_tool_margin": float((clean_tool[offset] - clean_best_other[offset]).item()),
                    "corrupt_non_tool_margin": float((corrupt_best_other[offset] - corrupt_tool[offset]).item()),
                }
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
            if batch_index % 100 == 0:
                print(f"{args.model_label}: processed {batch_index * args.batch_size}/{len(candidates)} candidates", flush=True)

    write_json(
        args.output.with_suffix(args.output.suffix + ".metadata.json"),
        {
            "model_label": args.model_label,
            "model_path": str(args.model_path.resolve()),
            "candidates": str(args.candidates.resolve()),
            "n_candidates": len(candidates),
            "n_behavior_valid": valid_count,
            "tool_token_id": tool_id,
            "batch_size": args.batch_size,
            "device": args.device,
            "dtype": args.dtype,
        },
    )
    print(f"{args.model_label}: {valid_count}/{len(candidates)} behavior-valid pairs")


if __name__ == "__main__":
    main()
