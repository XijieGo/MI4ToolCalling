#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gc
import json
from pathlib import Path

import torch
from safetensors.torch import load_file
from tqdm.auto import tqdm

from multiscale_common import (
    DEFAULT_DATASET_ROOT,
    build_pair_batches,
    load_model_and_tokenizer,
    load_sample_pairs,
    tool_stats,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate 4B late transcoder key-feature candidates with direct causal steering.")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--transcoder-path", type=Path, required=True)
    parser.add_argument("--candidate-csv", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--eval-split", type=str, default="test")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--top-n", type=int, default=12)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--require-positive-tool-proj", action="store_true")
    return parser.parse_args()


def clear_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def make_last_token_add_hook(delta_cpu: torch.Tensor):
    def hook_fn(value: torch.Tensor, hook):  # noqa: ANN001
        out = value.clone()
        delta = delta_cpu.to(device=value.device, dtype=value.dtype)
        if delta.ndim == 1:
            delta = delta.view(1, -1)
        out[:, -1, :] = out[:, -1, :] + delta
        return out

    return hook_fn


def load_candidates(path: Path, *, top_n: int, require_positive_tool_proj: bool) -> list[dict[str, object]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if require_positive_tool_proj:
        rows = [row for row in rows if float(row["tool_call_projection"]) > 0.0]
    rows = sorted(
        rows,
        key=lambda row: (
            float(row["abs_kappa"]),
            float(row["tool_call_projection"]),
        ),
        reverse=True,
    )
    return rows[:top_n]


def collect_baseline(
    model,
    batches,
    *,
    tool_token_id: int,
    side: str,
) -> dict[str, torch.Tensor | float]:
    logits_list: list[torch.Tensor] = []
    probs_list: list[torch.Tensor] = []
    top1_list: list[torch.Tensor] = []
    progress = tqdm(batches, desc=f"{side} baseline", dynamic_ncols=True)
    for batch in progress:
        tokens_cpu = batch.clean_tokens_cpu if side == "clean" else batch.corrupt_tokens_cpu
        tokens = tokens_cpu.to(model.W_U.device)
        with torch.no_grad():
            logits = model(tokens)
        tool_logit, tool_prob, top1 = tool_stats(logits, tool_token_id)
        logits_list.append(tool_logit)
        probs_list.append(tool_prob)
        top1_list.append(top1)
        del tokens, logits, tool_logit, tool_prob, top1
        clear_cuda()
    tool_logits = torch.cat(logits_list, dim=0)
    tool_probs = torch.cat(probs_list, dim=0)
    top1 = torch.cat(top1_list, dim=0)
    return {
        "tool_logits": tool_logits,
        "tool_probs": tool_probs,
        "top1": top1,
        "mean_tool_logit": float(tool_logits.mean().item()),
        "mean_tool_prob": float(tool_probs.mean().item()),
        "tool_call_top1_rate": float((top1 == tool_token_id).float().mean().item()),
    }


def evaluate_delta_vector(
    model,
    batches,
    *,
    tool_token_id: int,
    side: str,
    hook_name: str,
    delta_vector_cpu: torch.Tensor,
) -> dict[str, torch.Tensor | float]:
    logits_list: list[torch.Tensor] = []
    probs_list: list[torch.Tensor] = []
    top1_list: list[torch.Tensor] = []
    hooks = [(hook_name, make_last_token_add_hook(delta_vector_cpu))]
    progress = tqdm(batches, desc=f"{side} intervene", dynamic_ncols=True, leave=False)
    for batch in progress:
        tokens_cpu = batch.clean_tokens_cpu if side == "clean" else batch.corrupt_tokens_cpu
        tokens = tokens_cpu.to(model.W_U.device)
        with torch.no_grad():
            logits = model.run_with_hooks(tokens, fwd_hooks=hooks)
        tool_logit, tool_prob, top1 = tool_stats(logits, tool_token_id)
        logits_list.append(tool_logit)
        probs_list.append(tool_prob)
        top1_list.append(top1)
        del tokens, logits, tool_logit, tool_prob, top1
        clear_cuda()
    tool_logits = torch.cat(logits_list, dim=0)
    tool_probs = torch.cat(probs_list, dim=0)
    top1 = torch.cat(top1_list, dim=0)
    return {
        "tool_logits": tool_logits,
        "tool_probs": tool_probs,
        "top1": top1,
        "mean_tool_logit": float(tool_logits.mean().item()),
        "mean_tool_prob": float(tool_probs.mean().item()),
        "tool_call_top1_rate": float((top1 == tool_token_id).float().mean().item()),
    }


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    model, tokenizer, tool_token_id = load_model_and_tokenizer(model_path=args.model_path, device=args.device)
    pairs = load_sample_pairs(model, dataset_root=args.dataset_root, split=args.eval_split, max_pairs=0)
    batches = build_pair_batches(pairs, batch_size=args.batch_size)

    baseline_clean = collect_baseline(model, batches, tool_token_id=tool_token_id, side="clean")
    baseline_corrupt = collect_baseline(model, batches, tool_token_id=tool_token_id, side="corrupt")

    candidates = load_candidates(
        args.candidate_csv,
        top_n=args.top_n,
        require_positive_tool_proj=args.require_positive_tool_proj,
    )
    if not candidates:
        raise RuntimeError(f"No candidates found in {args.candidate_csv}")

    decoder_cache: dict[int, torch.Tensor] = {}
    results: list[dict[str, object]] = []
    for idx, row in enumerate(candidates, start=1):
        layer = int(row["layer"])
        feature_idx = int(row["feature_idx"])
        if layer not in decoder_cache:
            weights = load_file(str(args.transcoder_path / f"layer_{layer}.safetensors"))
            decoder_cache[layer] = weights["W_dec"].detach().cpu().float()
            del weights

        delta_activation = float(row["delta_activation"])
        decoder_row = decoder_cache[layer][feature_idx]
        delta_vector = decoder_row * delta_activation
        hook_name = f"blocks.{layer}.hook_mlp_out"

        corrupt_eval = evaluate_delta_vector(
            model,
            batches,
            tool_token_id=tool_token_id,
            side="corrupt",
            hook_name=hook_name,
            delta_vector_cpu=delta_vector,
        )
        clean_eval = evaluate_delta_vector(
            model,
            batches,
            tool_token_id=tool_token_id,
            side="clean",
            hook_name=hook_name,
            delta_vector_cpu=-delta_vector,
        )

        predicted_logit_delta = delta_activation * float(row["tool_call_projection"])
        result = {
            "rank_input": idx,
            "label": f"L{layer} F{feature_idx}",
            "layer": layer,
            "feature_idx": feature_idx,
            "delta_activation": delta_activation,
            "beta_mu": float(row["beta_mu"]),
            "tool_call_projection": float(row["tool_call_projection"]),
            "abs_kappa": float(row["abs_kappa"]),
            "predicted_tool_logit_delta": predicted_logit_delta,
            "corrupt_baseline_tool_call_top1_rate": float(baseline_corrupt["tool_call_top1_rate"]),
            "corrupt_intervene_tool_call_top1_rate": float(corrupt_eval["tool_call_top1_rate"]),
            "corrupt_top1_recovery_pp": 100.0 * (
                float(corrupt_eval["tool_call_top1_rate"]) - float(baseline_corrupt["tool_call_top1_rate"])
            ),
            "corrupt_baseline_mean_tool_logit": float(baseline_corrupt["mean_tool_logit"]),
            "corrupt_intervene_mean_tool_logit": float(corrupt_eval["mean_tool_logit"]),
            "corrupt_mean_tool_logit_delta": float(corrupt_eval["mean_tool_logit"]) - float(baseline_corrupt["mean_tool_logit"]),
            "clean_baseline_tool_call_top1_rate": float(baseline_clean["tool_call_top1_rate"]),
            "clean_intervene_tool_call_top1_rate": float(clean_eval["tool_call_top1_rate"]),
            "clean_top1_drop_pp": 100.0 * (
                float(baseline_clean["tool_call_top1_rate"]) - float(clean_eval["tool_call_top1_rate"])
            ),
            "clean_baseline_mean_tool_logit": float(baseline_clean["mean_tool_logit"]),
            "clean_intervene_mean_tool_logit": float(clean_eval["mean_tool_logit"]),
            "clean_mean_tool_logit_delta": float(clean_eval["mean_tool_logit"]) - float(baseline_clean["mean_tool_logit"]),
        }
        results.append(result)
        results.sort(
            key=lambda item: (
                float(item["corrupt_mean_tool_logit_delta"]),
                float(item["clean_top1_drop_pp"]),
                float(item["predicted_tool_logit_delta"]),
            ),
            reverse=True,
        )
        write_csv(args.output_root / "candidate_scores.csv", results)
        clear_cuda()

    best = results[0]
    summary = {
        "model_path": str(args.model_path),
        "transcoder_path": str(args.transcoder_path),
        "candidate_csv": str(args.candidate_csv),
        "eval_split": args.eval_split,
        "n_pairs": len(pairs),
        "tool_token_id": int(tool_token_id),
        "tool_token_text": tokenizer.decode([tool_token_id], clean_up_tokenization_spaces=False),
        "baseline_clean": {
            "mean_tool_logit": float(baseline_clean["mean_tool_logit"]),
            "mean_tool_prob": float(baseline_clean["mean_tool_prob"]),
            "tool_call_top1_rate": float(baseline_clean["tool_call_top1_rate"]),
        },
        "baseline_corrupt": {
            "mean_tool_logit": float(baseline_corrupt["mean_tool_logit"]),
            "mean_tool_prob": float(baseline_corrupt["mean_tool_prob"]),
            "tool_call_top1_rate": float(baseline_corrupt["tool_call_top1_rate"]),
        },
        "best_candidate": best,
    }
    (args.output_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.output_root / "summary.md").write_text(
        "\n".join(
            [
                "# 4B Key Feature Causal Screen",
                "",
                f"- Eval split: `{args.eval_split}` (`{len(pairs)}` pairs)",
                f"- Baseline clean top-1 `<tool_call>` rate: `{float(baseline_clean['tool_call_top1_rate']):.2%}`",
                f"- Baseline corrupt top-1 `<tool_call>` rate: `{float(baseline_corrupt['tool_call_top1_rate']):.2%}`",
                f"- Best candidate: `{best['label']}`",
                f"- Corrupt mean logit delta: `{float(best['corrupt_mean_tool_logit_delta']):+.3f}`",
                f"- Clean top-1 drop: `{float(best['clean_top1_drop_pp']):+.1f} pp`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
