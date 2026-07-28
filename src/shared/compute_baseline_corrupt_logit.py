#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import torch
from tqdm.auto import tqdm

from multiscale_common import (
    DEFAULT_DATASET_ROOT,
    build_pair_batches,
    load_model_and_tokenizer,
    load_sample_pairs,
    tool_stats,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute baseline corrupt-side <tool_call> logits for Table 3.")
    parser.add_argument("--model-paths", type=Path, nargs="+", required=True)
    parser.add_argument("--size-labels", type=str, nargs="+", required=True)
    parser.add_argument("--gate-layers", type=int, nargs="+", required=True)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--eval-split", type=str, default="test")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def clear_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_one_model(
    *,
    model_path: Path,
    size_label: str,
    gate_layer: int,
    dataset_root: Path,
    eval_split: str,
    batch_size: int,
    device: str,
) -> dict[str, object]:
    model, tokenizer, tool_token_id = load_model_and_tokenizer(model_path=model_path, device=device)
    pairs = load_sample_pairs(model, dataset_root=dataset_root, split=eval_split, max_pairs=0)
    batches = build_pair_batches(pairs, batch_size=batch_size)

    all_tool_logits: list[torch.Tensor] = []
    all_tool_probs: list[torch.Tensor] = []
    all_top1: list[torch.Tensor] = []

    progress = tqdm(batches, desc=f"{size_label} baseline", dynamic_ncols=True)
    for batch in progress:
        tokens = batch.corrupt_tokens_cpu.to(model.W_U.device)
        with torch.no_grad():
            logits = model(tokens)
        tool_logit, tool_prob, top1 = tool_stats(logits, tool_token_id)
        all_tool_logits.append(tool_logit)
        all_tool_probs.append(tool_prob)
        all_top1.append(top1)
        del tokens, logits, tool_logit, tool_prob, top1
        clear_cuda()

    tool_logits = torch.cat(all_tool_logits, dim=0)
    tool_probs = torch.cat(all_tool_probs, dim=0)
    top1 = torch.cat(all_top1, dim=0)

    result = {
        "size_label": size_label,
        "model_path": str(model_path),
        "eval_split": eval_split,
        "gate_layer": gate_layer,
        "n_pairs": int(tool_logits.shape[0]),
        "tool_token_id": int(tool_token_id),
        "tool_token_text": tokenizer.decode([tool_token_id], clean_up_tokenization_spaces=False),
        "mean_tool_call_logit": float(tool_logits.mean().item()),
        "mean_tool_call_prob": float(tool_probs.mean().item()),
        "tool_call_top1_rate": float((top1 == tool_token_id).float().mean().item()),
    }

    del model, tokenizer, pairs, batches, tool_logits, tool_probs, top1
    clear_cuda()
    return result


def main() -> None:
    args = parse_args()
    if not (len(args.model_paths) == len(args.size_labels) == len(args.gate_layers)):
        raise ValueError("--model-paths, --size-labels, and --gate-layers must have the same length.")

    rows: list[dict[str, object]] = []
    for model_path, size_label, gate_layer in zip(args.model_paths, args.size_labels, args.gate_layers):
        rows.append(
            run_one_model(
                model_path=model_path,
                size_label=size_label,
                gate_layer=int(gate_layer),
                dataset_root=args.dataset_root,
                eval_split=args.eval_split,
                batch_size=args.batch_size,
                device=args.device,
            )
        )

    args.output_root.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_root / "baseline_corrupt_logit_summary.json"
    summary_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
