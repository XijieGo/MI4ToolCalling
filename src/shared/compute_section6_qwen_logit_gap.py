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
    parser = argparse.ArgumentParser(description="Compute Section 6 Qwen3 logit-gap Suff./Necc. at alpha=1.")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--size-label", type=str, required=True)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--eval-split", type=str, default="test")
    parser.add_argument("--add-csv", type=Path, default=None)
    parser.add_argument("--remove-csv", type=Path, default=None)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--hook-kind", type=str, default="post")
    parser.add_argument("--pc-bundle", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-pairs", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def clear_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def read_alpha_row(path: Path, alpha: float = 1.0) -> dict[str, str]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if abs(float(row["alpha"]) - alpha) < 1e-9:
                return row
    raise KeyError(f"Could not find alpha={alpha:g} in {path}")


def make_last_token_add_hook(delta_cpu: torch.Tensor):
    def hook_fn(value: torch.Tensor, hook):  # noqa: ANN001
        out = value.clone()
        delta = delta_cpu.to(device=value.device, dtype=value.dtype)
        if delta.ndim == 1:
            delta = delta.view(1, -1)
        out[:, -1, :] = out[:, -1, :] + delta
        return out

    return hook_fn


def load_mu_delta(path: Path | None) -> torch.Tensor | None:
    if path is None:
        return None
    bundle = torch.load(path, map_location="cpu", weights_only=False)
    mean_diff = bundle.get("mean_diff")
    if mean_diff is None:
        return None
    if not isinstance(mean_diff, torch.Tensor):
        mean_diff = torch.tensor(mean_diff)
    return mean_diff.detach().cpu().float().view(-1)


def run_baselines(model, pairs, *, batch_size: int, tool_token_id: int) -> dict[str, float]:
    batches = build_pair_batches(pairs, batch_size=batch_size)
    clean_logit_sum = 0.0
    corrupt_logit_sum = 0.0
    clean_prob_sum = 0.0
    corrupt_prob_sum = 0.0
    clean_top1 = 0
    corrupt_top1 = 0
    count = 0
    for batch in tqdm(batches, desc="baseline clean/corrupt", dynamic_ncols=True):
        clean_tokens = batch.clean_tokens_cpu.to(model.W_U.device)
        corrupt_tokens = batch.corrupt_tokens_cpu.to(model.W_U.device)
        with torch.no_grad():
            clean_logits = model(clean_tokens)
            corrupt_logits = model(corrupt_tokens)
        c_logit, c_prob, c_top1 = tool_stats(clean_logits, tool_token_id)
        x_logit, x_prob, x_top1 = tool_stats(corrupt_logits, tool_token_id)
        clean_logit_sum += float(c_logit.sum().item())
        corrupt_logit_sum += float(x_logit.sum().item())
        clean_prob_sum += float(c_prob.sum().item())
        corrupt_prob_sum += float(x_prob.sum().item())
        clean_top1 += int((c_top1 == tool_token_id).sum().item())
        corrupt_top1 += int((x_top1 == tool_token_id).sum().item())
        count += int(c_top1.numel())
        del clean_tokens, corrupt_tokens, clean_logits, corrupt_logits, c_logit, c_prob, c_top1, x_logit, x_prob, x_top1
        clear_cuda()
    return {
        "n_pairs": float(count),
        "clean_mean_tool_logit": clean_logit_sum / max(count, 1),
        "corrupt_mean_tool_logit": corrupt_logit_sum / max(count, 1),
        "clean_mean_tool_prob": clean_prob_sum / max(count, 1),
        "corrupt_mean_tool_prob": corrupt_prob_sum / max(count, 1),
        "clean_top1_rate": clean_top1 / max(count, 1),
        "corrupt_top1_rate": corrupt_top1 / max(count, 1),
    }


def run_vector_sanity(
    model,
    pairs,
    *,
    batch_size: int,
    tool_token_id: int,
    hook_name: str,
    mu_delta: torch.Tensor,
) -> dict[str, float]:
    batches = build_pair_batches(pairs, batch_size=batch_size)
    plus_sum = 0.0
    minus_sum = 0.0
    count = 0
    for batch in tqdm(batches, desc="vector sanity", dynamic_ncols=True):
        clean_tokens = batch.clean_tokens_cpu.to(model.W_U.device)
        corrupt_tokens = batch.corrupt_tokens_cpu.to(model.W_U.device)
        with torch.no_grad():
            plus_logits = model.run_with_hooks(corrupt_tokens, fwd_hooks=[(hook_name, make_last_token_add_hook(mu_delta))])
            minus_logits = model.run_with_hooks(clean_tokens, fwd_hooks=[(hook_name, make_last_token_add_hook(-mu_delta))])
        plus_logit, _plus_prob, _plus_top1 = tool_stats(plus_logits, tool_token_id)
        minus_logit, _minus_prob, _minus_top1 = tool_stats(minus_logits, tool_token_id)
        plus_sum += float(plus_logit.sum().item())
        minus_sum += float(minus_logit.sum().item())
        count += int(plus_logit.numel())
        del clean_tokens, corrupt_tokens, plus_logits, minus_logits, plus_logit, minus_logit
        clear_cuda()
    return {
        "vector_plus_mean_tool_logit_recomputed": plus_sum / max(count, 1),
        "vector_minus_mean_tool_logit_recomputed": minus_sum / max(count, 1),
    }


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_summary(path: Path, payload: dict[str, object]) -> None:
    lines = [
        f"# {payload['size_label']} Section 6 Logit-Gap Suff./Necc.",
        "",
        f"- Layer: `L{payload['layer']}` at `hook_resid_{payload['hook_kind']}`",
        f"- Eval pairs: `{int(payload['n_pairs'])}`",
        f"- Clean mean tool logit: `{float(payload['clean_mean_tool_logit']):.6f}`",
        f"- Corrupt mean tool logit: `{float(payload['corrupt_mean_tool_logit']):.6f}`",
        f"- Vector-plus mean tool logit: `{float(payload['vector_plus_mean_tool_logit']):.6f}`",
        f"- Vector-minus mean tool logit: `{float(payload['vector_minus_mean_tool_logit']):.6f}`",
        f"- Suff.: `{float(payload['suff']):.6f}`",
        f"- Necc.: `{float(payload['necc']):.6f}`",
        f"- Source add CSV: `{payload['add_csv']}`",
        f"- Source remove CSV: `{payload['remove_csv']}`",
    ]
    if payload.get("vector_logit_recomputed"):
        lines.append("- Vector sanity logits were recomputed from the supplied `mu_delta` bundle.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    add_row = read_alpha_row(args.add_csv) if args.add_csv is not None else {}
    remove_row = read_alpha_row(args.remove_csv) if args.remove_csv is not None else {}
    if (args.add_csv is None or args.remove_csv is None) and args.pc_bundle is None:
        raise ValueError("When add/remove CSVs are omitted, --pc-bundle is required so vector logits can be recomputed.")
    model, tokenizer, tool_token_id = load_model_and_tokenizer(model_path=args.model_path, device=args.device)
    pairs = load_sample_pairs(model, dataset_root=args.dataset_root, split=args.eval_split, max_pairs=args.max_pairs)
    baseline = run_baselines(model, pairs, batch_size=args.batch_size, tool_token_id=tool_token_id)

    vector_plus = float(add_row["mean_tool_call_logit"]) if add_row else float("nan")
    vector_minus = float(remove_row["mean_tool_call_logit"]) if remove_row else float("nan")
    vector_logit_recomputed = False
    if args.pc_bundle is not None:
        mu_delta = load_mu_delta(args.pc_bundle)
        if mu_delta is not None:
            hook_name = f"blocks.{args.layer}.hook_resid_{args.hook_kind}"
            sanity = run_vector_sanity(
                model,
                pairs,
                batch_size=args.batch_size,
                tool_token_id=tool_token_id,
                hook_name=hook_name,
                mu_delta=mu_delta,
            )
            vector_plus = float(sanity["vector_plus_mean_tool_logit_recomputed"])
            vector_minus = float(sanity["vector_minus_mean_tool_logit_recomputed"])
            vector_logit_recomputed = True

    gap = float(baseline["clean_mean_tool_logit"] - baseline["corrupt_mean_tool_logit"])
    suff = (vector_plus - float(baseline["corrupt_mean_tool_logit"])) / gap
    necc = (float(baseline["clean_mean_tool_logit"]) - vector_minus) / gap
    payload = {
        "size_label": args.size_label,
        "model_path": str(args.model_path),
        "dataset_root": str(args.dataset_root),
        "eval_split": args.eval_split,
        "layer": int(args.layer),
        "hook_kind": args.hook_kind,
        "pc_bundle": str(args.pc_bundle) if args.pc_bundle is not None else None,
        "add_csv": str(args.add_csv) if args.add_csv is not None else None,
        "remove_csv": str(args.remove_csv) if args.remove_csv is not None else None,
        "tool_token_id": int(tool_token_id),
        "tool_token_text": tokenizer.decode([tool_token_id], clean_up_tokenization_spaces=False),
        "n_pairs": int(baseline["n_pairs"]),
        "clean_mean_tool_logit": float(baseline["clean_mean_tool_logit"]),
        "corrupt_mean_tool_logit": float(baseline["corrupt_mean_tool_logit"]),
        "clean_mean_tool_prob": float(baseline["clean_mean_tool_prob"]),
        "corrupt_mean_tool_prob": float(baseline["corrupt_mean_tool_prob"]),
        "clean_top1_rate": float(baseline["clean_top1_rate"]),
        "corrupt_top1_rate": float(baseline["corrupt_top1_rate"]),
        "vector_plus_mean_tool_logit": vector_plus,
        "vector_minus_mean_tool_logit": vector_minus,
        "suff": float(suff),
        "necc": float(necc),
        "vector_plus_top1_rate_csv": float(add_row.get("tool_call_top1_rate", "nan")) if add_row else None,
        "vector_minus_remaining_top1_rate_csv": float(remove_row.get("remaining_tool_call_top1_rate", "nan")) if remove_row else None,
        "vector_logit_recomputed": vector_logit_recomputed,
    }
    write_json(args.output_root / "logit_gap_metrics.json", payload)
    write_summary(args.output_root / "summary.md", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))

    del model
    clear_cuda()


if __name__ == "__main__":
    main()
