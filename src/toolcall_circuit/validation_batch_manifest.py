#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

import torch
from tqdm.auto import tqdm

if __package__ in {None, ""}:
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from toolcall_circuit.dataset import get_tool_call_target_spec, load_toolcall_samples, resolve_distractor_token
from toolcall_circuit.single_sample import load_hooked_qwen3


def write_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def build_forward_summary(
    *,
    sample,
    target_token_id: int,
    distractor_token_id: int,
    tokenizer,
) -> Dict[str, object]:
    return {
        "sample_id": sample.sample_id,
        "sample_rank": sample.sample_rank,
        "filename": sample.filename,
        "direction": "forward_tool_call",
        "clean_role": "tool_call",
        "corrupt_role": "no_tool",
        "clean_prompt": str(sample.clean_path),
        "corrupt_prompt": str(sample.corrupt_path),
        "target_token_id": int(target_token_id),
        "target_token_str": tokenizer.decode([int(target_token_id)]),
        "distractor_token_id": int(distractor_token_id),
        "distractor_token_str": tokenizer.decode([int(distractor_token_id)]),
        "sample_catalog_record": sample.catalog_record(),
        "discovery_method": "validation_manifest_only",
    }


def build_reverse_summary(
    *,
    sample,
    target_token_id: int,
    distractor_token_id: int,
    tokenizer,
) -> Dict[str, object]:
    return {
        "sample_id": sample.sample_id,
        "sample_rank": sample.sample_rank,
        "filename": sample.filename,
        "direction": "reverse_no_tool",
        "clean_role": "no_tool",
        "corrupt_role": "tool_call",
        "clean_prompt": str(sample.corrupt_path),
        "corrupt_prompt": str(sample.clean_path),
        "target_token_id": int(distractor_token_id),
        "target_token_str": tokenizer.decode([int(distractor_token_id)]),
        "distractor_token_id": int(target_token_id),
        "distractor_token_str": tokenizer.decode([int(target_token_id)]),
        "sample_catalog_record": sample.catalog_record(),
        "discovery_method": "validation_manifest_only",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build lightweight batch manifests for test-side validation.")
    parser.add_argument("--dataset-root", type=str, required=True)
    parser.add_argument("--model-path", type=str, default="./external/models/Qwen3-1.7B")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output-root", type=str, required=True)
    parser.add_argument("--max-samples", type=int, default=0)
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root).resolve()
    out_root = Path(args.output_root).resolve()
    forward_root = out_root / "forward_batch"
    reverse_root = out_root / "reverse_batch"
    forward_root.mkdir(parents=True, exist_ok=True)
    reverse_root.mkdir(parents=True, exist_ok=True)

    samples = load_toolcall_samples(dataset_root=dataset_root)
    if args.max_samples > 0:
        samples = samples[: args.max_samples]

    model, tokenizer = load_hooked_qwen3(args.model_path, device=args.device, dtype=torch.bfloat16)
    target_spec = get_tool_call_target_spec(tokenizer, target_text="<tool_call>")
    if not target_spec.is_single_token:
        raise NotImplementedError(
            f"<tool_call> tokenization is no longer single-token ({target_spec.token_ids})."
        )
    tool_token = target_spec.primary_token_id

    rows: List[Dict[str, object]] = []
    pbar = tqdm(samples, desc="Build validation manifests", dynamic_ncols=True)
    for sample in pbar:
        clean_text = sample.clean_path.read_text(encoding="utf-8")
        corrupt_text = sample.corrupt_path.read_text(encoding="utf-8")
        clean_tokens = model.to_tokens(clean_text, prepend_bos=False)
        corrupt_tokens = model.to_tokens(corrupt_text, prepend_bos=False)
        if clean_tokens.shape != corrupt_tokens.shape:
            continue
        with torch.no_grad():
            corrupt_logits = model(corrupt_tokens)
        distractor = resolve_distractor_token(corrupt_logits[0, -1, :], tool_token)

        forward_summary = build_forward_summary(
            sample=sample,
            target_token_id=tool_token,
            distractor_token_id=distractor,
            tokenizer=tokenizer,
        )
        reverse_summary = build_reverse_summary(
            sample=sample,
            target_token_id=tool_token,
            distractor_token_id=distractor,
            tokenizer=tokenizer,
        )
        write_json(forward_root / sample.sample_id / "summary.json", forward_summary)
        write_json(reverse_root / sample.sample_id / "summary.json", reverse_summary)
        rows.append(
            {
                "sample_id": sample.sample_id,
                "sample_rank": sample.sample_rank,
                "filename": sample.filename,
                "clean_prompt": str(sample.clean_path),
                "corrupt_prompt": str(sample.corrupt_path),
                "target_token_id": int(tool_token),
                "distractor_token_id": int(distractor),
            }
        )
        pbar.set_postfix(sample=sample.sample_id)

    summary = {
        "dataset_root": str(dataset_root),
        "model_path": args.model_path,
        "device": args.device,
        "n_samples": len(rows),
        "forward_batch_root": str(forward_root),
        "reverse_batch_root": str(reverse_root),
        "manifest_rows": rows,
    }
    write_json(out_root / "manifest_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
