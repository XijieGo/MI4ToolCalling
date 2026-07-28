#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import torch
from tqdm.auto import tqdm

from multiscale_common import (
    DEFAULT_DATASET_ROOT,
    build_pair_batches,
    choose_best_layer,
    clear_cuda,
    ensure_dir,
    load_model_and_tokenizer,
    load_sample_pairs,
    set_seed,
    tool_stats,
    write_csv,
    write_json,
    write_text,
)


DEFAULT_ALPHAS = (0.5, 0.75, 1.0, 1.25, 1.5)
DEFAULT_F1_LAYERS = tuple(range(20, 29))
DEFAULT_F1_ANCHORS = (23, 26)


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 200,
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.6,
        }
    )


def parse_layers(text: str) -> list[int]:
    return [int(part.strip()) for part in str(text).split(",") if part.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Task-book-specific runner for cross-scale gate-fix experiments.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--model-path", type=Path, required=True)
    common.add_argument("--size-label", type=str, required=True)
    common.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    common.add_argument("--batch-size", type=int, default=8)
    common.add_argument("--device", type=str, default="cuda")
    common.add_argument("--seed", type=int, default=42)

    f1 = subparsers.add_parser("f1", parents=[common], help="Run the 4B state-patch sweep required by F1.")
    f1.add_argument("--eval-split", type=str, default="test")
    f1.add_argument("--output-root", type=Path, required=True)
    f1.add_argument("--hook-kind", type=str, choices=("pre", "post"), default="post")
    f1.add_argument("--coarse-pairs", type=int, default=200)
    f1.add_argument("--final-pairs", type=int, default=300)
    f1.add_argument("--coarse-layers", type=str, default="20,21,22,23,24,25,26,27,28")
    f1.add_argument("--anchor-layers", type=str, default="23,26")
    f1.add_argument("--best-layer-tolerance", type=float, default=0.01)

    f2 = subparsers.add_parser("f2", parents=[common], help="Run the bidirectional gate sweep required by F2.")
    f2.add_argument("--train-split", type=str, default="train")
    f2.add_argument("--eval-split", type=str, default="test")
    f2.add_argument("--output-root", type=Path, required=True)
    f2.add_argument("--pc-bundle", type=Path, default=None)
    f2.add_argument("--gate-layer", type=int, default=None)
    f2.add_argument("--hook-kind", type=str, choices=("pre", "post"), default=None)
    f2.add_argument("--recompute-mu-delta", action="store_true")
    f2.add_argument("--gate-train-pairs", type=int, default=200)
    f2.add_argument("--gate-eval-pairs", type=int, default=300)
    f2.add_argument("--alphas", type=float, nargs="+", default=list(DEFAULT_ALPHAS))
    f2.add_argument("--bundle-output-name", type=str, default="")
    return parser.parse_args()


def hook_name(layer: int, hook_kind: str) -> str:
    return f"blocks.{layer}.hook_resid_{hook_kind}"


def make_last_token_replace_hook(source_cpu: torch.Tensor):
    def hook_fn(value: torch.Tensor, hook):  # noqa: ANN001
        out = value.clone()
        source = source_cpu.to(device=value.device, dtype=value.dtype)
        if source.ndim == 3:
            out[:, -1, :] = source[:, -1, :]
        elif source.ndim == 2:
            out[:, -1, :] = source
        else:
            raise ValueError(f"Unexpected replace-hook source shape: {tuple(source.shape)}")
        return out

    return hook_fn


def make_last_token_add_hook(delta_cpu: torch.Tensor):
    def hook_fn(value: torch.Tensor, hook):  # noqa: ANN001
        out = value.clone()
        delta = delta_cpu.to(device=value.device, dtype=value.dtype)
        if delta.ndim == 1:
            delta = delta.view(1, -1)
        if delta.ndim != 2:
            raise ValueError(f"Unexpected add-hook delta shape: {tuple(delta.shape)}")
        out[:, -1, :] = out[:, -1, :] + delta
        return out

    return hook_fn


def load_gate_bundle(path: Path) -> dict[str, object]:
    return torch.load(path, map_location="cpu", weights_only=False)


def infer_hook_kind(bundle: dict[str, object]) -> str:
    return "pre" if "patch_layer" in bundle else "post"


def gate_layer_from_bundle(bundle: dict[str, object]) -> int:
    if "layer" in bundle:
        return int(bundle["layer"])
    if "patch_layer" in bundle:
        return int(bundle["patch_layer"])
    raise KeyError("Could not infer gate layer from bundle.")


def extract_mu_delta(bundle: dict[str, object]) -> torch.Tensor:
    mean_diff = bundle["mean_diff"]
    if not isinstance(mean_diff, torch.Tensor):
        mean_diff = torch.tensor(mean_diff)
    return mean_diff.detach().cpu().float().view(-1)


def orient_components(components: torch.Tensor, diff_vectors: torch.Tensor) -> torch.Tensor:
    oriented = components.clone()
    for idx in range(oriented.shape[0]):
        direction = oriented[idx]
        if float((diff_vectors @ direction).mean().item()) < 0:
            oriented[idx] = -direction
    return oriented


def compute_pca(diff_vectors: torch.Tensor, *, n_components: int = 10) -> dict[str, torch.Tensor]:
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
        "singular_values": singular_values[:actual_components],
    }


def collect_residuals_at_hook(
    model,
    pairs,
    *,
    layer: int,
    hook_kind: str,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    pair_batches = build_pair_batches(pairs, batch_size=batch_size)
    resid_name = hook_name(layer, hook_kind)
    d_model = int(model.cfg.d_model)
    clean_resid = torch.empty((len(pairs), d_model), dtype=torch.float32)
    corrupt_resid = torch.empty((len(pairs), d_model), dtype=torch.float32)

    progress = tqdm(pair_batches, desc=f"L{layer} {hook_kind} residual collect", dynamic_ncols=True)
    for batch in progress:
        clean_tokens = batch.clean_tokens_cpu.to(model.W_U.device)
        corrupt_tokens = batch.corrupt_tokens_cpu.to(model.W_U.device)
        with torch.no_grad():
            _, clean_cache = model.run_with_cache(clean_tokens, names_filter=lambda name: name == resid_name)
            _, corrupt_cache = model.run_with_cache(corrupt_tokens, names_filter=lambda name: name == resid_name)
        indices = torch.tensor(batch.indices, dtype=torch.long)
        clean_resid[indices] = clean_cache[resid_name][:, -1, :].detach().cpu().float()
        corrupt_resid[indices] = corrupt_cache[resid_name][:, -1, :].detach().cpu().float()
        del clean_tokens, corrupt_tokens, clean_cache, corrupt_cache, indices
        clear_cuda()
        progress.set_postfix(tok=batch.token_len)
    return clean_resid, corrupt_resid


def patch_sweep(
    model,
    pairs,
    *,
    layers: Sequence[int],
    hook_kind: str,
    batch_size: int,
    tool_token_id: int,
    phase: str,
) -> list[dict[str, object]]:
    pair_batches = build_pair_batches(pairs, batch_size=batch_size)
    wanted_hooks = [hook_name(layer, hook_kind) for layer in layers]
    accum = {
        int(layer): {"count": 0, "tool_top1": 0, "strict_flip": 0, "logit_sum": 0.0, "prob_sum": 0.0}
        for layer in layers
    }
    baseline_clean_tool_top1 = 0
    baseline_corrupt_tool_top1 = 0
    baseline_count = 0

    progress = tqdm(pair_batches, desc=f"{phase} state patch", dynamic_ncols=True)
    for batch in progress:
        clean_tokens = batch.clean_tokens_cpu.to(model.W_U.device)
        corrupt_tokens = batch.corrupt_tokens_cpu.to(model.W_U.device)
        with torch.no_grad():
            clean_logits, clean_cache = model.run_with_cache(clean_tokens, names_filter=lambda name: name in wanted_hooks)
            corrupt_logits = model(corrupt_tokens)
        _, _, clean_top1 = tool_stats(clean_logits, tool_token_id)
        corrupt_tool_logit, _, corrupt_top1 = tool_stats(corrupt_logits, tool_token_id)
        baseline_clean_tool_top1 += int((clean_top1 == tool_token_id).sum().item())
        baseline_corrupt_tool_top1 += int((corrupt_top1 == tool_token_id).sum().item())
        baseline_count += int(clean_top1.shape[0])

        for layer in layers:
            name = hook_name(int(layer), hook_kind)
            hooks = [(name, make_last_token_replace_hook(clean_cache[name].detach().cpu()))]
            with torch.no_grad():
                patched_logits = model.run_with_hooks(corrupt_tokens, fwd_hooks=hooks)
            patched_logit, patched_prob, patched_top1 = tool_stats(patched_logits, tool_token_id)
            bucket = accum[int(layer)]
            bucket["count"] += int(patched_top1.shape[0])
            bucket["tool_top1"] += int((patched_top1 == tool_token_id).sum().item())
            bucket["strict_flip"] += int(((corrupt_top1 != tool_token_id) & (patched_top1 == tool_token_id)).sum().item())
            bucket["logit_sum"] += float(patched_logit.sum().item())
            bucket["prob_sum"] += float(patched_prob.sum().item())
            del patched_logits, patched_logit, patched_prob, patched_top1
            clear_cuda()

        del clean_tokens, corrupt_tokens, clean_logits, clean_cache, corrupt_logits, corrupt_tool_logit, clean_top1, corrupt_top1
        clear_cuda()
        progress.set_postfix(tok=batch.token_len)

    rows: list[dict[str, object]] = []
    for layer in layers:
        bucket = accum[int(layer)]
        count = max(int(bucket["count"]), 1)
        rows.append(
            {
                "phase": phase,
                "layer": int(layer),
                "hook_kind": hook_kind,
                "n": count,
                "tool_call_top1_rate": float(bucket["tool_top1"] / count),
                "strict_flip_rate": float(bucket["strict_flip"] / count),
                "mean_tool_call_logit": float(bucket["logit_sum"] / count),
                "mean_tool_call_prob": float(bucket["prob_sum"] / count),
                "baseline_clean_tool_top1_rate": float(baseline_clean_tool_top1 / max(baseline_count, 1)),
                "baseline_corrupt_tool_top1_rate": float(baseline_corrupt_tool_top1 / max(baseline_count, 1)),
            }
        )
    return rows


def plot_patch_sweep(rows: list[dict[str, object]], path: Path) -> None:
    if not rows:
        return
    final_rows = [row for row in rows if str(row["phase"]) == "final"] or rows
    final_rows = sorted(final_rows, key=lambda row: int(row["layer"]))
    layers = [int(row["layer"]) for row in final_rows]
    top1 = [float(row["tool_call_top1_rate"]) for row in final_rows]
    strict_flip = [float(row["strict_flip_rate"]) for row in final_rows]
    logits = [float(row["mean_tool_call_logit"]) for row in final_rows]

    fig, ax1 = plt.subplots(figsize=(7.2, 4.4))
    ax1.plot(layers, top1, marker="o", linewidth=2.0, label="tool-call top1")
    ax1.plot(layers, strict_flip, marker="s", linewidth=1.8, label="strict flip")
    ax1.set_xlabel("Layer")
    ax1.set_ylabel("Rate")
    ax1.set_ylim(0.0, 1.05)
    ax2 = ax1.twinx()
    ax2.plot(layers, logits, marker="^", linewidth=1.5, linestyle="--", color="tab:red", label="mean tool logit")
    ax2.set_ylabel("Mean tool logit")
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="best")
    fig.tight_layout()
    ensure_dir(path.parent)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def row_by_layer(rows: Sequence[dict[str, object]], layer: int) -> dict[str, object] | None:
    for row in rows:
        if int(row["layer"]) == int(layer):
            return row
    return None


def build_f1_summary(
    *,
    size_label: str,
    hook_kind: str,
    best_row: dict[str, object],
    rows: list[dict[str, object]],
    anchor_layers: Sequence[int],
) -> str:
    coarse_rows = [row for row in rows if str(row["phase"]) == "coarse"]
    final_rows = [row for row in rows if str(row["phase"]) == "final"]
    lines = [
        "# Exp A Summary",
        "",
        f"- Size: `{size_label}`",
        f"- Hook kind: `hook_resid_{hook_kind}`",
        f"- Best layer: `L{int(best_row['layer'])}`",
        f"- Best patched top-1 `<tool_call>` rate: `{float(best_row['tool_call_top1_rate']):.2%}`",
        f"- Best strict flip rate: `{float(best_row['strict_flip_rate']):.2%}`",
        f"- Final baseline corrupt top-1 `<tool_call>` rate: `{float(best_row['baseline_corrupt_tool_top1_rate']):.2%}`",
        f"- Coarse sweep layers: `{[int(row['layer']) for row in coarse_rows]}`",
        f"- Final sweep layers: `{[int(row['layer']) for row in final_rows]}`",
    ]
    for layer in anchor_layers:
        anchor_row = row_by_layer(final_rows, int(layer)) or row_by_layer(coarse_rows, int(layer))
        if anchor_row is not None:
            lines.append(
                f"- L{int(layer)} result: top-1 `{float(anchor_row['tool_call_top1_rate']):.2%}`, strict flip `{float(anchor_row['strict_flip_rate']):.2%}`"
            )
    lines.extend(
        [
            "",
            "Interpretation:",
            "The dominant causal bottleneck remains a prediction-position residual state in the mid-to-late stack, and this rerun replaces the legacy 4B state-patch summary with a canonical sweep on the current phase6 dataset.",
        ]
    )
    return "\n".join(lines)


def run_f1(args: argparse.Namespace) -> None:
    configure_matplotlib()
    ensure_dir(args.output_root)
    coarse_layers = parse_layers(args.coarse_layers) or list(DEFAULT_F1_LAYERS)
    anchor_layers = parse_layers(args.anchor_layers) or list(DEFAULT_F1_ANCHORS)

    model, _tokenizer, tool_token_id = load_model_and_tokenizer(model_path=args.model_path, device=args.device)
    eval_pairs_full = load_sample_pairs(
        model,
        dataset_root=args.dataset_root,
        split=args.eval_split,
        max_pairs=args.final_pairs,
    )
    coarse_pairs = eval_pairs_full[: min(args.coarse_pairs, len(eval_pairs_full))]

    coarse_rows = patch_sweep(
        model,
        coarse_pairs,
        layers=coarse_layers,
        hook_kind=args.hook_kind,
        batch_size=args.batch_size,
        tool_token_id=tool_token_id,
        phase="coarse",
    )
    best_coarse = choose_best_layer(coarse_rows, tolerance=args.best_layer_tolerance)
    final_layers = sorted(
        {
            int(layer)
            for layer in ([best_coarse - 1, best_coarse, best_coarse + 1] + list(anchor_layers))
            if 0 <= int(layer) < int(model.cfg.n_layers)
        }
    )
    final_rows = patch_sweep(
        model,
        eval_pairs_full,
        layers=final_layers,
        hook_kind=args.hook_kind,
        batch_size=args.batch_size,
        tool_token_id=tool_token_id,
        phase="final",
    )
    selected_layer = choose_best_layer(final_rows, tolerance=args.best_layer_tolerance)
    best_row = next(row for row in final_rows if int(row["layer"]) == selected_layer)
    raw_peak = max(final_rows, key=lambda row: float(row["tool_call_top1_rate"]))
    rows = coarse_rows + final_rows

    write_csv(args.output_root / "patch_sweep.csv", rows)
    plot_patch_sweep(rows, args.output_root / "plot_patch_sweep.pdf")
    write_text(
        args.output_root / "summary.md",
        build_f1_summary(
            size_label=args.size_label,
            hook_kind=args.hook_kind,
            best_row=best_row,
            rows=rows,
            anchor_layers=anchor_layers,
        ),
    )
    write_json(
        args.output_root / "metadata.json",
        {
            "size_label": args.size_label,
            "model_path": str(args.model_path),
            "dataset_root": str(args.dataset_root),
            "eval_split": args.eval_split,
            "hook_kind": args.hook_kind,
            "coarse_pairs": len(coarse_pairs),
            "final_pairs": len(eval_pairs_full),
            "coarse_layers": coarse_layers,
            "fine_layers": final_layers,
            "best_layer": int(best_row["layer"]),
            "best_layer_raw_peak": int(raw_peak["layer"]),
            "selected_sample_ids": [pair.sample_id for pair in eval_pairs_full],
        },
    )

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def choose_bundle_path(args: argparse.Namespace) -> Path:
    if args.bundle_output_name:
        return args.output_root / args.bundle_output_name
    if args.gate_layer is None or args.hook_kind is None:
        raise ValueError("gate_layer and hook_kind are required when naming a recomputed bundle automatically.")
    return args.output_root / f"pca_components_L{int(args.gate_layer)}_{args.hook_kind}.pt"


def maybe_recompute_mu_delta(
    model,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, int, str, Path]:
    if not args.recompute_mu_delta:
        if args.pc_bundle is None:
            raise ValueError("pc_bundle is required unless --recompute-mu-delta is set.")
        bundle = load_gate_bundle(args.pc_bundle)
        gate_layer = args.gate_layer if args.gate_layer is not None else gate_layer_from_bundle(bundle)
        hook_kind = args.hook_kind if args.hook_kind is not None else infer_hook_kind(bundle)
        mu_delta = extract_mu_delta(bundle)
        return mu_delta, gate_layer, hook_kind, args.pc_bundle

    if args.gate_layer is None or args.hook_kind is None:
        raise ValueError("--gate-layer and --hook-kind are required when recomputing mu_delta.")
    train_pairs = load_sample_pairs(
        model,
        dataset_root=args.dataset_root,
        split=args.train_split,
        max_pairs=args.gate_train_pairs,
    )
    clean_resid, corrupt_resid = collect_residuals_at_hook(
        model,
        train_pairs,
        layer=args.gate_layer,
        hook_kind=args.hook_kind,
        batch_size=args.batch_size,
    )
    diff = clean_resid - corrupt_resid
    pca = compute_pca(diff, n_components=10)
    bundle = {
        "mean_diff": pca["mean_diff"],
        "components": pca["components"],
        "explained_variance": pca["explained_variance"],
        "explained_variance_ratio": pca["explained_variance_ratio"],
        "sample_ids_train": [pair.sample_id for pair in train_pairs],
    }
    if args.hook_kind == "pre":
        bundle["patch_layer"] = int(args.gate_layer)
    else:
        bundle["layer"] = int(args.gate_layer)
        bundle["singular_values"] = pca["singular_values"]
    bundle_path = choose_bundle_path(args)
    ensure_dir(bundle_path.parent)
    torch.save(bundle, bundle_path)
    return pca["mean_diff"].detach().cpu().float().view(-1), int(args.gate_layer), args.hook_kind, bundle_path


def alpha_key(alpha: float) -> str:
    return f"{float(alpha):.4f}"


def evaluate_bidirectional_gate(
    model,
    pairs,
    *,
    gate_layer: int,
    hook_kind: str,
    batch_size: int,
    tool_token_id: int,
    mu_delta: torch.Tensor,
    alphas: Sequence[float],
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, float]]:
    pair_batches = build_pair_batches(pairs, batch_size=batch_size)
    name = hook_name(gate_layer, hook_kind)

    add_acc = {
        alpha_key(alpha): {"alpha": float(alpha), "count": 0, "tool_top1": 0, "flip": 0, "logit_sum": 0.0, "prob_sum": 0.0}
        for alpha in alphas
    }
    remove_acc = {
        alpha_key(alpha): {"alpha": float(alpha), "count": 0, "tool_top1": 0, "drop": 0, "logit_sum": 0.0, "prob_sum": 0.0}
        for alpha in alphas
    }

    baseline_clean_tool_top1 = 0
    baseline_corrupt_tool_top1 = 0
    baseline_count = 0
    baseline_clean_tool_den = 0
    baseline_corrupt_non_tool_den = 0

    progress = tqdm(pair_batches, desc=f"bidirectional L{gate_layer} {hook_kind}", dynamic_ncols=True)
    for batch in progress:
        clean_tokens = batch.clean_tokens_cpu.to(model.W_U.device)
        corrupt_tokens = batch.corrupt_tokens_cpu.to(model.W_U.device)
        with torch.no_grad():
            clean_logits = model(clean_tokens)
            corrupt_logits = model(corrupt_tokens)
        _, _, clean_top1 = tool_stats(clean_logits, tool_token_id)
        _, _, corrupt_top1 = tool_stats(corrupt_logits, tool_token_id)
        clean_is_tool = clean_top1 == tool_token_id
        corrupt_is_tool = corrupt_top1 == tool_token_id
        batch_n = int(clean_top1.shape[0])
        baseline_clean_tool_top1 += int(clean_is_tool.sum().item())
        baseline_corrupt_tool_top1 += int(corrupt_is_tool.sum().item())
        baseline_clean_tool_den += int(clean_is_tool.sum().item())
        baseline_corrupt_non_tool_den += int((~corrupt_is_tool).sum().item())
        baseline_count += batch_n

        for alpha in alphas:
            key = alpha_key(alpha)
            delta = (mu_delta * float(alpha)).view(1, -1)

            with torch.no_grad():
                add_logits = model.run_with_hooks(corrupt_tokens, fwd_hooks=[(name, make_last_token_add_hook(delta))])
                remove_logits = model.run_with_hooks(clean_tokens, fwd_hooks=[(name, make_last_token_add_hook(-delta))])

            add_logit, add_prob, add_top1 = tool_stats(add_logits, tool_token_id)
            remove_logit, remove_prob, remove_top1 = tool_stats(remove_logits, tool_token_id)
            add_is_tool = add_top1 == tool_token_id
            remove_is_tool = remove_top1 == tool_token_id

            add_bucket = add_acc[key]
            add_bucket["count"] += batch_n
            add_bucket["tool_top1"] += int(add_is_tool.sum().item())
            add_bucket["flip"] += int(((~corrupt_is_tool) & add_is_tool).sum().item())
            add_bucket["logit_sum"] += float(add_logit.sum().item())
            add_bucket["prob_sum"] += float(add_prob.sum().item())

            remove_bucket = remove_acc[key]
            remove_bucket["count"] += batch_n
            remove_bucket["tool_top1"] += int(remove_is_tool.sum().item())
            remove_bucket["drop"] += int((clean_is_tool & (~remove_is_tool)).sum().item())
            remove_bucket["logit_sum"] += float(remove_logit.sum().item())
            remove_bucket["prob_sum"] += float(remove_prob.sum().item())

            del add_logits, remove_logits, add_logit, add_prob, add_top1, remove_logit, remove_prob, remove_top1
            clear_cuda()

        del clean_tokens, corrupt_tokens, clean_logits, corrupt_logits, clean_top1, corrupt_top1, clean_is_tool, corrupt_is_tool
        clear_cuda()
        progress.set_postfix(tok=batch.token_len)

    baseline = {
        "n": float(baseline_count),
        "baseline_clean_top1_rate": float(baseline_clean_tool_top1 / max(baseline_count, 1)),
        "baseline_corrupt_top1_rate": float(baseline_corrupt_tool_top1 / max(baseline_count, 1)),
        "baseline_clean_tool_count": float(baseline_clean_tool_den),
        "baseline_corrupt_non_tool_count": float(baseline_corrupt_non_tool_den),
    }

    add_rows: list[dict[str, object]] = []
    for alpha in alphas:
        bucket = add_acc[alpha_key(alpha)]
        count = max(int(bucket["count"]), 1)
        denom = max(int(baseline["baseline_corrupt_non_tool_count"]), 1)
        add_rows.append(
            {
                "alpha": float(alpha),
                "n": count,
                "tool_call_top1_rate": float(bucket["tool_top1"] / count),
                "strict_flip_rate": float(bucket["flip"] / denom),
                "baseline_corrupt_top1_rate": float(baseline["baseline_corrupt_top1_rate"]),
                "baseline_corrupt_non_tool_count": int(baseline["baseline_corrupt_non_tool_count"]),
                "mean_tool_call_logit": float(bucket["logit_sum"] / count),
                "mean_tool_call_prob": float(bucket["prob_sum"] / count),
            }
        )

    remove_rows: list[dict[str, object]] = []
    for alpha in alphas:
        bucket = remove_acc[alpha_key(alpha)]
        count = max(int(bucket["count"]), 1)
        denom = max(int(baseline["baseline_clean_tool_count"]), 1)
        remove_rows.append(
            {
                "alpha": float(alpha),
                "n": count,
                "remaining_tool_call_top1_rate": float(bucket["tool_top1"] / count),
                "strict_drop_rate": float(bucket["drop"] / denom),
                "baseline_clean_top1_rate": float(baseline["baseline_clean_top1_rate"]),
                "baseline_clean_tool_count": int(baseline["baseline_clean_tool_count"]),
                "mean_tool_call_logit": float(bucket["logit_sum"] / count),
                "mean_tool_call_prob": float(bucket["prob_sum"] / count),
            }
        )

    return add_rows, remove_rows, baseline


def plot_bidirectional_gate(add_rows: Sequence[dict[str, object]], remove_rows: Sequence[dict[str, object]], path: Path) -> None:
    if not add_rows or not remove_rows:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.2), sharex=True)

    add_x = [float(row["alpha"]) for row in add_rows]
    add_top1 = [float(row["tool_call_top1_rate"]) for row in add_rows]
    add_flip = [float(row["strict_flip_rate"]) for row in add_rows]
    axes[0].plot(add_x, add_top1, marker="o", linewidth=2.0, label="top-1 after add")
    axes[0].plot(add_x, add_flip, marker="s", linewidth=1.8, label="strict flip")
    axes[0].set_title("Sufficient Direction")
    axes[0].set_xlabel("alpha")
    axes[0].set_ylabel("Rate")
    axes[0].set_ylim(0.0, 1.05)
    axes[0].legend(frameon=False, loc="best")

    remove_x = [float(row["alpha"]) for row in remove_rows]
    remove_top1 = [float(row["remaining_tool_call_top1_rate"]) for row in remove_rows]
    remove_drop = [float(row["strict_drop_rate"]) for row in remove_rows]
    axes[1].plot(remove_x, remove_top1, marker="o", linewidth=2.0, label="remaining top-1")
    axes[1].plot(remove_x, remove_drop, marker="^", linewidth=1.8, label="strict drop")
    axes[1].set_title("Necessary Direction")
    axes[1].set_xlabel("alpha")
    axes[1].set_ylim(0.0, 1.05)
    axes[1].legend(frameon=False, loc="best")

    fig.tight_layout()
    ensure_dir(path.parent)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def row_at_alpha(rows: Sequence[dict[str, object]], alpha: float) -> dict[str, object]:
    target = alpha_key(alpha)
    for row in rows:
        if alpha_key(float(row["alpha"])) == target:
            return row
    raise KeyError(f"Could not find alpha={alpha:g} row.")


def summarize_strength(add_row: dict[str, object], remove_row: dict[str, object]) -> str:
    add_top1 = float(add_row["tool_call_top1_rate"])
    add_flip = float(add_row["strict_flip_rate"])
    rem_top1 = float(remove_row["remaining_tool_call_top1_rate"])
    rem_drop = float(remove_row["strict_drop_rate"])

    add_strong = add_top1 > 0.90 and add_flip > 0.85
    remove_strong = rem_top1 < 0.10 and rem_drop > 0.80
    add_weak = add_top1 > 0.80 and add_flip > 0.70
    remove_weak = rem_top1 < 0.30 and rem_drop > 0.60

    if add_strong and remove_strong:
        return "At alpha=1, both sufficient and necessary directions satisfy the task-book strong threshold."
    if add_weak and remove_weak:
        return "At alpha=1, both directions work, but at least one metric only reaches the weak threshold."
    if add_weak:
        return "At alpha=1, the sufficient direction is confirmed, while removal only partially suppresses tool calling."
    return "At alpha=1, the gate evidence is weaker than the task-book threshold and should be treated as diagnostic rather than definitive."


def build_bidirectional_summary(
    *,
    size_label: str,
    gate_layer: int,
    hook_kind: str,
    bundle_path: Path,
    baseline: dict[str, float],
    add_rows: Sequence[dict[str, object]],
    remove_rows: Sequence[dict[str, object]],
) -> str:
    add_alpha1 = row_at_alpha(add_rows, 1.0)
    remove_alpha1 = row_at_alpha(remove_rows, 1.0)
    lines = [
        "# Bidirectional Gate Summary",
        "",
        f"- Size: `{size_label}`",
        f"- Gate layer: `L{gate_layer}` at `hook_resid_{hook_kind}`",
        f"- Mu-delta bundle: `{bundle_path}`",
        f"- Baseline clean top-1: `{float(baseline['baseline_clean_top1_rate']):.2%}`",
        f"- Baseline corrupt top-1: `{float(baseline['baseline_corrupt_top1_rate']):.2%}`",
        "",
        "## Sufficient direction (gate add, alpha=1)",
        f"- top-1 rate after add: `{float(add_alpha1['tool_call_top1_rate']):.2%}`",
        f"- strict flip rate: `{float(add_alpha1['strict_flip_rate']):.2%}`",
        "",
        "## Necessary direction (gate remove, alpha=1)",
        f"- remaining top-1 rate after remove: `{float(remove_alpha1['remaining_tool_call_top1_rate']):.2%}`",
        f"- strict drop rate: `{float(remove_alpha1['strict_drop_rate']):.2%}`",
        "",
        "## Alpha sweep (add direction)",
        "| alpha | top-1 | strict flip |",
        "|---|---|---|",
    ]
    for row in add_rows:
        lines.append(
            f"| {float(row['alpha']):.2f} | {float(row['tool_call_top1_rate']):.2%} | {float(row['strict_flip_rate']):.2%} |"
        )
    lines.extend(
        [
            "",
            "## Alpha sweep (remove direction)",
            "| alpha | remaining top-1 | strict drop |",
            "|---|---|---|",
        ]
    )
    for row in remove_rows:
        lines.append(
            f"| {float(row['alpha']):.2f} | {float(row['remaining_tool_call_top1_rate']):.2%} | {float(row['strict_drop_rate']):.2%} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            summarize_strength(add_alpha1, remove_alpha1),
        ]
    )
    return "\n".join(lines)


def run_f2(args: argparse.Namespace) -> None:
    configure_matplotlib()
    ensure_dir(args.output_root)
    model, _tokenizer, tool_token_id = load_model_and_tokenizer(model_path=args.model_path, device=args.device)
    mu_delta, gate_layer, hook_kind, bundle_path = maybe_recompute_mu_delta(model, args)
    eval_pairs = load_sample_pairs(
        model,
        dataset_root=args.dataset_root,
        split=args.eval_split,
        max_pairs=args.gate_eval_pairs,
    )

    add_rows, remove_rows, baseline = evaluate_bidirectional_gate(
        model,
        eval_pairs,
        gate_layer=gate_layer,
        hook_kind=hook_kind,
        batch_size=args.batch_size,
        tool_token_id=tool_token_id,
        mu_delta=mu_delta,
        alphas=args.alphas,
    )

    write_csv(args.output_root / "alpha_sweep_add.csv", add_rows)
    write_csv(args.output_root / "alpha_sweep_remove.csv", remove_rows)
    plot_bidirectional_gate(add_rows, remove_rows, args.output_root / "plot_bidirectional_gate.pdf")
    write_text(
        args.output_root / "bidirectional_summary.md",
        build_bidirectional_summary(
            size_label=args.size_label,
            gate_layer=gate_layer,
            hook_kind=hook_kind,
            bundle_path=bundle_path,
            baseline=baseline,
            add_rows=add_rows,
            remove_rows=remove_rows,
        ),
    )
    write_json(
        args.output_root / "bidirectional_metadata.json",
        {
            "size_label": args.size_label,
            "model_path": str(args.model_path),
            "dataset_root": str(args.dataset_root),
            "train_split": args.train_split,
            "eval_split": args.eval_split,
            "gate_layer": gate_layer,
            "hook_kind": hook_kind,
            "bundle_path": str(bundle_path),
            "recomputed_mu_delta": bool(args.recompute_mu_delta),
            "gate_train_pairs": int(args.gate_train_pairs),
            "gate_eval_pairs": len(eval_pairs),
            "alphas": [float(alpha) for alpha in args.alphas],
            **baseline,
        },
    )

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    if args.command == "f1":
        run_f1(args)
        return
    if args.command == "f2":
        run_f2(args)
        return
    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
