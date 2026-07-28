#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
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
    parser = argparse.ArgumentParser(description="Run a fixed-layer Qwen3 prediction-position residual state patch.")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--size-label", type=str, required=True)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--eval-split", type=str, default="test")
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--hook-kind", type=str, default="post")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-pairs", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def clear_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def make_last_token_replace_hook(source_cpu: torch.Tensor):
    def hook_fn(value: torch.Tensor, hook):  # noqa: ANN001
        out = value.clone()
        source = source_cpu.to(device=value.device, dtype=value.dtype)
        if source.ndim == 3:
            out[:, -1, :] = source[:, -1, :]
        elif source.ndim == 2:
            out[:, -1, :] = source
        else:
            raise ValueError(f"Unexpected source shape: {tuple(source.shape)}")
        return out

    return hook_fn


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_summary(path: Path, payload: dict[str, object]) -> None:
    lines = [
        f"# {payload['size_label']} State Patch L{payload['layer']} Summary",
        "",
        f"- Eval pairs: `{payload['n_eval_pairs']}`",
        f"- Hook: `blocks.{payload['layer']}.hook_resid_{payload['hook_kind']}`",
        f"- Baseline clean top-1: `{float(payload['baseline_clean_top1_rate']):.2%}`",
        f"- Baseline corrupt top-1: `{float(payload['baseline_corrupt_top1_rate']):.2%}`",
        f"- Patched top-1: `{float(payload['patched_top1_rate']):.2%}`",
        f"- Strict flip: `{float(payload['strict_flip_rate']):.2%}`",
        f"- Mean patched tool-call logit: `{float(payload['mean_tool_call_logit']):.6f}`",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    hook_name = f"blocks.{args.layer}.hook_resid_{args.hook_kind}"
    model, tokenizer, tool_token_id = load_model_and_tokenizer(model_path=args.model_path, device=args.device)
    pairs = load_sample_pairs(model, dataset_root=args.dataset_root, split=args.eval_split, max_pairs=args.max_pairs)
    batches = build_pair_batches(pairs, batch_size=args.batch_size)

    per_sample: list[dict[str, object]] = []
    clean_tool = 0
    corrupt_tool = 0
    patched_tool = 0
    strict_flip = 0
    clean_logit_sum = 0.0
    corrupt_logit_sum = 0.0
    patched_logit_sum = 0.0
    count = 0

    for batch in tqdm(batches, desc=f"{args.size_label} L{args.layer} patch", dynamic_ncols=True):
        clean_tokens = batch.clean_tokens_cpu.to(model.W_U.device)
        corrupt_tokens = batch.corrupt_tokens_cpu.to(model.W_U.device)
        with torch.no_grad():
            clean_logits, clean_cache = model.run_with_cache(clean_tokens, names_filter=lambda name: name == hook_name)
            corrupt_logits = model(corrupt_tokens)
            patched_logits = model.run_with_hooks(
                corrupt_tokens,
                fwd_hooks=[(hook_name, make_last_token_replace_hook(clean_cache[hook_name].detach().cpu()))],
            )

        clean_logit, clean_prob, clean_top1 = tool_stats(clean_logits, tool_token_id)
        corrupt_logit, corrupt_prob, corrupt_top1 = tool_stats(corrupt_logits, tool_token_id)
        patched_logit, patched_prob, patched_top1 = tool_stats(patched_logits, tool_token_id)
        batch_size = int(clean_top1.numel())
        for local_idx, sample_idx in enumerate(batch.indices):
            pair = pairs[int(sample_idx)]
            per_sample.append(
                {
                    "sample_id": pair.sample_id,
                    "clean_top1_is_tool": int(clean_top1[local_idx].item() == tool_token_id),
                    "corrupt_top1_is_tool": int(corrupt_top1[local_idx].item() == tool_token_id),
                    "patched_top1_is_tool": int(patched_top1[local_idx].item() == tool_token_id),
                    "strict_flip": int(
                        corrupt_top1[local_idx].item() != tool_token_id
                        and patched_top1[local_idx].item() == tool_token_id
                    ),
                    "clean_tool_logit": float(clean_logit[local_idx].item()),
                    "corrupt_tool_logit": float(corrupt_logit[local_idx].item()),
                    "patched_tool_logit": float(patched_logit[local_idx].item()),
                    "clean_tool_prob": float(clean_prob[local_idx].item()),
                    "corrupt_tool_prob": float(corrupt_prob[local_idx].item()),
                    "patched_tool_prob": float(patched_prob[local_idx].item()),
                }
            )

        clean_tool += int((clean_top1 == tool_token_id).sum().item())
        corrupt_tool += int((corrupt_top1 == tool_token_id).sum().item())
        patched_tool += int((patched_top1 == tool_token_id).sum().item())
        strict_flip += int(((corrupt_top1 != tool_token_id) & (patched_top1 == tool_token_id)).sum().item())
        clean_logit_sum += float(clean_logit.sum().item())
        corrupt_logit_sum += float(corrupt_logit.sum().item())
        patched_logit_sum += float(patched_logit.sum().item())
        count += batch_size
        del clean_tokens, corrupt_tokens, clean_logits, clean_cache, corrupt_logits, patched_logits
        clear_cuda()

    summary = {
        "size_label": args.size_label,
        "model_path": str(args.model_path),
        "dataset_root": str(args.dataset_root),
        "eval_split": args.eval_split,
        "layer": int(args.layer),
        "hook_kind": args.hook_kind,
        "hook_name": hook_name,
        "n_eval_pairs": int(count),
        "tool_token_id": int(tool_token_id),
        "tool_token_text": tokenizer.decode([tool_token_id], clean_up_tokenization_spaces=False),
        "baseline_clean_top1_rate": float(clean_tool / max(count, 1)),
        "baseline_corrupt_top1_rate": float(corrupt_tool / max(count, 1)),
        "patched_top1_rate": float(patched_tool / max(count, 1)),
        "strict_flip_rate": float(strict_flip / max(count, 1)),
        "baseline_clean_mean_tool_logit": float(clean_logit_sum / max(count, 1)),
        "baseline_corrupt_mean_tool_logit": float(corrupt_logit_sum / max(count, 1)),
        "mean_tool_call_logit": float(patched_logit_sum / max(count, 1)),
    }
    write_csv(args.output_root / f"patch_L{args.layer}_per_sample.csv", per_sample)
    write_json(args.output_root / f"patch_L{args.layer}_summary.json", summary)
    write_summary(args.output_root / "summary.md", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    del model
    clear_cuda()


if __name__ == "__main__":
    main()
