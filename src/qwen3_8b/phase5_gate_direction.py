#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

from task_attention_path_analysis import (
    DATASET_ROOT,
    MODEL_PATH,
    build_pair_batches,
    clear_cuda,
    ensure_dir,
    load_samples,
    set_seed,
    write_csv,
    write_text,
)

from phase4_reviewer_strengthening import (
    SEED,
    TOOL_CALL_STR,
    make_last_token_resid_add_hook,
    make_resid_last_capture,
    tool_stats,
)

import sys

LEGACY_SRC = Path("./src")
if str(LEGACY_SRC) not in sys.path:
    sys.path.insert(0, str(LEGACY_SRC))

from toolcall_circuit.single_sample import load_hooked_qwen3  # noqa: E402


PHASE5_ROOT = Path("./results/8B/phase5_neurips/exp_b_gate_direction")
PATCH_LAYER = 24
N_COMPONENTS = 10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 5 experiment B: L24 gate-direction semantics")
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH)
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--output-root", type=Path, default=PHASE5_ROOT)
    parser.add_argument("--patch-layer", type=int, default=PATCH_LAYER)
    parser.add_argument("--max-pairs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--n-components", type=int, default=N_COMPONENTS)
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.5, 1.0, 1.5, 2.0])
    parser.add_argument("--ks", nargs="+", default=["1", "2", "5", "10", "full"])
    return parser.parse_args()


def orient_components(components: torch.Tensor, diff_vectors: torch.Tensor) -> torch.Tensor:
    oriented = components.clone()
    for idx in range(oriented.shape[0]):
        direction = oriented[idx]
        if float((diff_vectors @ direction).mean().item()) < 0:
            oriented[idx] = -direction
    return oriented


def compute_pca(diff_vectors: torch.Tensor, *, n_components: int) -> dict[str, torch.Tensor]:
    centered = diff_vectors - diff_vectors.mean(dim=0, keepdim=True)
    _, singular_values, vh = torch.linalg.svd(centered, full_matrices=False)
    actual_components = min(n_components, int(vh.shape[0]))
    components = orient_components(vh[:actual_components].contiguous(), diff_vectors)
    exp_var = (singular_values[:actual_components] ** 2) / max(diff_vectors.shape[0] - 1, 1)
    total_var = float((centered.pow(2).sum().item()) / max(diff_vectors.shape[0] - 1, 1))
    exp_var_ratio = exp_var / total_var if total_var > 0 else torch.zeros_like(exp_var)
    return {
        "components": components,
        "mean_diff": diff_vectors.mean(dim=0),
        "explained_variance": exp_var,
        "explained_variance_ratio": exp_var_ratio,
        "centered": centered,
    }


def project_delta(diff_batch: torch.Tensor, pca: dict[str, torch.Tensor], *, k: str) -> torch.Tensor:
    if k == "full":
        return diff_batch
    k_int = int(k)
    components = pca["components"][:k_int]
    mean_diff = pca["mean_diff"].unsqueeze(0)
    centered = diff_batch - mean_diff
    coeff = centered @ components.T
    projected = coeff @ components + mean_diff
    return projected


def make_vocab_projection_rows(
    model,
    tokenizer,
    components: torch.Tensor,
) -> list[dict[str, object]]:
    w_u = model.W_U.to(device=model.W_U.device, dtype=model.W_U.dtype)
    special_ids = set(getattr(tokenizer, "all_special_ids", []))
    rows: list[dict[str, object]] = []
    for comp_idx in range(components.shape[0]):
        direction = components[comp_idx].to(device=w_u.device, dtype=w_u.dtype)
        scores = torch.matmul(w_u.transpose(0, 1), direction).detach().cpu().float()
        top_vals, top_idx = torch.topk(scores, k=50, largest=True)
        bot_vals, bot_idx = torch.topk(scores, k=50, largest=False)
        for rank, (token_id, score) in enumerate(zip(top_idx.tolist(), top_vals.tolist()), start=1):
            raw_token = tokenizer.convert_ids_to_tokens([token_id])[0]
            token_text = tokenizer.decode([token_id]).replace("\n", "\\n")
            rows.append(
                {
                    "component_idx": comp_idx + 1,
                    "direction": "top",
                    "rank": rank,
                    "token_id": token_id,
                    "token": token_text,
                    "token_raw": raw_token,
                    "score": float(score),
                    "is_special": int(token_id in special_ids),
                }
            )
        for rank, (token_id, score) in enumerate(zip(bot_idx.tolist(), bot_vals.tolist()), start=1):
            raw_token = tokenizer.convert_ids_to_tokens([token_id])[0]
            token_text = tokenizer.decode([token_id]).replace("\n", "\\n")
            rows.append(
                {
                    "component_idx": comp_idx + 1,
                    "direction": "bottom",
                    "rank": rank,
                    "token_id": token_id,
                    "token": token_text,
                    "token_raw": raw_token,
                    "score": float(score),
                    "is_special": int(token_id in special_ids),
                }
            )
    return rows


def token_list_for_summary(rows: list[dict[str, object]], *, component_idx: int, direction: str, limit: int = 12) -> list[str]:
    filtered: list[str] = []
    seen: set[str] = set()
    for row in rows:
        if int(row["component_idx"]) != component_idx or str(row["direction"]) != direction or int(row["is_special"]) != 0:
            continue
        token = str(row["token"]).strip()
        if not token or token in seen:
            continue
        seen.add(token)
        filtered.append(token)
        if len(filtered) >= limit:
            break
    return filtered


def main() -> None:
    args = parse_args()
    ensure_dir(args.output_root)
    set_seed(args.seed)

    model, tokenizer = load_hooked_qwen3(str(args.model_path), device=args.device, dtype=torch.bfloat16)
    tool_token_ids = tokenizer.encode(TOOL_CALL_STR, add_special_tokens=False)
    if len(tool_token_ids) != 1:
        raise ValueError(f"{TOOL_CALL_STR!r} maps to unexpected token ids: {tool_token_ids}")
    tool_token_id = int(tool_token_ids[0])

    samples = load_samples(args.dataset_root, model, tokenizer, max_pairs=args.max_pairs)
    pair_batches = build_pair_batches(samples, args.batch_size)
    resid_hook_name = f"blocks.{args.patch_layer}.hook_resid_pre"

    clean_resid = torch.empty((len(samples), int(model.cfg.d_model)), dtype=torch.float32)
    corrupt_resid = torch.empty((len(samples), int(model.cfg.d_model)), dtype=torch.float32)
    clean_top1 = torch.empty(len(samples), dtype=torch.long)
    corrupt_top1 = torch.empty(len(samples), dtype=torch.long)
    clean_tool_logit = torch.empty(len(samples), dtype=torch.float32)
    corrupt_tool_logit = torch.empty(len(samples), dtype=torch.float32)

    progress = tqdm(pair_batches, desc="Exp B capture residuals", dynamic_ncols=True)
    for batch in progress:
        clean_tokens = batch.clean_tokens_cpu.to(model.W_U.device)
        corrupt_tokens = batch.corrupt_tokens_cpu.to(model.W_U.device)
        clean_capture: dict[str, torch.Tensor] = {}
        corrupt_capture: dict[str, torch.Tensor] = {}
        with torch.no_grad():
            clean_logits = model.run_with_hooks(clean_tokens, fwd_hooks=[(resid_hook_name, make_resid_last_capture(clean_capture, "resid"))])
            corrupt_logits = model.run_with_hooks(
                corrupt_tokens,
                fwd_hooks=[(resid_hook_name, make_resid_last_capture(corrupt_capture, "resid"))],
            )
        clean_resid[batch.indices] = clean_capture["resid"]
        corrupt_resid[batch.indices] = corrupt_capture["resid"]
        clean_tool_logit_batch, clean_top1_batch = tool_stats(clean_logits, tool_token_id)
        corrupt_tool_logit_batch, corrupt_top1_batch = tool_stats(corrupt_logits, tool_token_id)
        clean_tool_logit[batch.indices] = clean_tool_logit_batch
        corrupt_tool_logit[batch.indices] = corrupt_tool_logit_batch
        clean_top1[batch.indices] = clean_top1_batch
        corrupt_top1[batch.indices] = corrupt_top1_batch
        clear_cuda()

    diff_vectors = clean_resid - corrupt_resid
    pca = compute_pca(diff_vectors, n_components=args.n_components)
    torch.save(
        {
            "patch_layer": args.patch_layer,
            "sample_ids": [sample.sample_id for sample in samples],
            "components": pca["components"],
            "mean_diff": pca["mean_diff"],
            "explained_variance": pca["explained_variance"],
            "explained_variance_ratio": pca["explained_variance_ratio"],
            "clean_tool_logit": clean_tool_logit,
            "corrupt_tool_logit": corrupt_tool_logit,
        },
        args.output_root / "pca_components.pt",
    )

    ev_rows = []
    cumulative = 0.0
    actual_components = int(pca["components"].shape[0])
    for idx in range(actual_components):
        ratio = float(pca["explained_variance_ratio"][idx].item())
        cumulative += ratio
        ev_rows.append(
            {
                "component_idx": idx + 1,
                "explained_var": float(pca["explained_variance"][idx].item()),
                "explained_var_ratio": ratio,
                "cumulative": cumulative,
            }
        )
    write_csv(args.output_root / "explained_variance.csv", ev_rows)

    vocab_rows = make_vocab_projection_rows(model, tokenizer, pca["components"])
    write_csv(args.output_root / "vocab_projections.csv", vocab_rows)

    sweep_rows: list[dict[str, object]] = []
    progress = tqdm(pair_batches, desc="Exp B low-rank patch sweep", dynamic_ncols=True)
    for batch in progress:
        corrupt_tokens = batch.corrupt_tokens_cpu.to(model.W_U.device)
        base_top1 = corrupt_top1[batch.indices]
        batch_diff = diff_vectors[batch.indices]
        for k in args.ks:
            projected = project_delta(batch_diff, pca, k=str(k))
            for alpha in args.alphas:
                condition = f"k={k},alpha={alpha:g}"
                hooks = [(resid_hook_name, make_last_token_resid_add_hook(projected * float(alpha)))]
                with torch.no_grad():
                    patched_logits = model.run_with_hooks(corrupt_tokens, fwd_hooks=hooks)
                tool_logit, top1 = tool_stats(patched_logits, tool_token_id)
                top1_is_tool = (top1 == tool_token_id)
                sweep_rows.append(
                    {
                        "condition": condition,
                        "k": str(k),
                        "alpha": float(alpha),
                        "batch_n": len(batch.indices),
                        "tool_call_top1_sum": int(top1_is_tool.sum().item()),
                        "strict_flip_sum": int(((base_top1 != tool_token_id) & top1_is_tool).sum().item()),
                        "tool_logit_sum": float(tool_logit.sum().item()),
                    }
                )
        clear_cuda()

    grouped: dict[tuple[str, float], dict[str, float]] = {}
    for row in sweep_rows:
        key = (str(row["k"]), float(row["alpha"]))
        acc = grouped.setdefault(
            key,
            {
                "n_samples": 0.0,
                "tool_call_top1_sum": 0.0,
                "strict_flip_sum": 0.0,
                "tool_logit_sum": 0.0,
            },
        )
        acc["n_samples"] += float(row["batch_n"])
        acc["tool_call_top1_sum"] += float(row["tool_call_top1_sum"])
        acc["strict_flip_sum"] += float(row["strict_flip_sum"])
        acc["tool_logit_sum"] += float(row["tool_logit_sum"])

    summary_rows: list[dict[str, object]] = []
    for k in args.ks:
        for alpha in args.alphas:
            acc = grouped[(str(k), float(alpha))]
            n = max(acc["n_samples"], 1.0)
            summary_rows.append(
                {
                    "k": str(k),
                    "alpha": float(alpha),
                    "tool_call_top1_rate": acc["tool_call_top1_sum"] / n,
                    "strict_flip_rate": acc["strict_flip_sum"] / n,
                    "mean_tool_logit": acc["tool_logit_sum"] / n,
                    "n_samples": int(n),
                }
            )
    write_csv(args.output_root / "rank_k_patch_sweep.csv", summary_rows)

    best_rank1 = max((row for row in summary_rows if row["k"] == "1"), key=lambda row: float(row["tool_call_top1_rate"]))
    best_overall = max(summary_rows, key=lambda row: float(row["tool_call_top1_rate"]))
    top_tokens = token_list_for_summary(vocab_rows, component_idx=1, direction="top")
    bottom_tokens = token_list_for_summary(vocab_rows, component_idx=1, direction="bottom")
    lines = [
        "# Experiment B: Gate Direction Semantics",
        "",
        f"- 模型: `{args.model_path}`",
        f"- 样本: `datasets/test` 前 `{len(samples)}` 对",
        f"- 层位点: `blocks.{args.patch_layer}.hook_resid_pre` 的 prediction position",
        "",
        "## Explained Variance",
        "",
        f"- PC1 explained variance ratio: `{float(ev_rows[0]['explained_var_ratio']):.4f}`",
        f"- Top-3 cumulative explained variance ratio: `{float(ev_rows[min(2, len(ev_rows)-1)]['cumulative']):.4f}`",
        "",
        "## PC1 Vocab Projection",
        "",
        f"- clean-direction top tokens: `{top_tokens}`",
        f"- corrupt-direction bottom tokens: `{bottom_tokens}`",
        "",
        "## Low-Rank Patch Sweep",
        "",
        f"- best rank-1 condition: `alpha={float(best_rank1['alpha']):g}` -> top1 `{float(best_rank1['tool_call_top1_rate']):.2%}`, strict flip `{float(best_rank1['strict_flip_rate']):.2%}`, mean logit `{float(best_rank1['mean_tool_logit']):.4f}`",
        f"- best overall condition: `k={best_overall['k']}, alpha={float(best_overall['alpha']):g}` -> top1 `{float(best_overall['tool_call_top1_rate']):.2%}`",
    ]
    write_text(args.output_root / "summary.md", "\n".join(lines))

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
