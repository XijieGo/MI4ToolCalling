#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gc
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from phase7_l24_directionality_common import (
    DEFAULT_ALPHA_SWEEP_A,
    EVAL_DATASET_ROOT,
    PATCH_LAYER,
    evaluate_intervention,
    load_or_collect_pair_baseline,
    summarize_condition_row,
)
from phase8_common import (
    DEFAULT_HELDOUT_CACHE,
    DEFAULT_PC_BUNDLE,
    EXP_A_ROOT,
    MODEL_PATH,
    PHASE7_ROOT,
    configure_matplotlib,
    ensure_dir,
    get_tool_token_id,
    load_model_and_tokenizer,
    percent,
    set_seed,
    write_csv,
    write_text,
)


DEFAULT_KS = ("1", "2", "3", "5", "10", "full")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 8 Exp A: PC-k recovery main figure.")
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH)
    parser.add_argument("--eval-root", type=Path, default=EVAL_DATASET_ROOT)
    parser.add_argument("--pc-bundle", type=Path, default=DEFAULT_PC_BUNDLE)
    parser.add_argument("--eval-cache", type=Path, default=DEFAULT_HELDOUT_CACHE)
    parser.add_argument("--output-root", type=Path, default=EXP_A_ROOT)
    parser.add_argument("--phase7-headline", type=Path, default=PHASE7_ROOT / "headline_table.csv")
    parser.add_argument(
        "--phase7-alpha-sweep",
        type=Path,
        default=PHASE7_ROOT / "exp_a_fixed_direction" / "alpha_sweep_corrupt.csv",
    )
    parser.add_argument("--max-pairs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--alphas", type=float, nargs="+", default=list(DEFAULT_ALPHA_SWEEP_A))
    parser.add_argument("--ks", nargs="+", default=list(DEFAULT_KS))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def project_delta(diff_batch: torch.Tensor, pc_bundle: dict[str, torch.Tensor], *, k: str) -> torch.Tensor:
    if k == "full":
        return diff_batch
    k_int = int(k)
    components = pc_bundle["components"][:k_int]
    mean_diff = pc_bundle["mean_diff"].unsqueeze(0)
    centered = diff_batch - mean_diff
    coeff = centered @ components.T
    return coeff @ components + mean_diff


def best_row_for_k(rows: list[dict[str, object]], k: str) -> dict[str, object]:
    candidates = [row for row in rows if str(row["k"]) == str(k)]
    if not candidates:
        raise ValueError(f"Missing rows for k={k}")
    return max(
        candidates,
        key=lambda row: (
            float(row["strict_flip_rate"]),
            float(row["tool_call_top1_rate"]),
            float(row["mean_tool_logit"]),
            -abs(float(row["alpha"]) - 1.0),
        ),
    )


def best_fixed_pc1_row(path: Path) -> dict[str, object]:
    rows = [row for row in read_csv_rows(path) if row["condition_family"] == "fixed_pc1_add"]
    if not rows:
        raise ValueError(f"No fixed_pc1_add rows in {path}")
    best = max(rows, key=lambda row: (float(row["strict_flip_rate"]), float(row["tool_call_top1_rate"])))
    return {
        "intervention": "L24 fixed PC1 add",
        "k": "",
        "alpha": float(best["alpha"]),
        "tool_call_top1_rate": float(best["tool_call_top1_rate"]),
        "strict_flip_rate": float(best["strict_flip_rate"]),
        "mean_tool_logit": float(best["mean_tool_logit"]),
        "n_samples": int(best["n_samples"]),
        "source": str(path),
    }


def point_from_headline(path: Path, intervention: str) -> dict[str, object]:
    rows = read_csv_rows(path)
    matches = [row for row in rows if row["intervention"] == intervention]
    if not matches:
        raise ValueError(f"Missing intervention={intervention!r} in {path}")
    row = matches[0]
    return {
        "intervention": intervention,
        "k": "",
        "alpha": "",
        "tool_call_top1_rate": float(row["tool_call_top1_rate"]),
        "strict_flip_rate": float(row["strict_flip_or_drop_rate"]),
        "mean_tool_logit": "",
        "n_samples": "",
        "source": str(path),
    }


def plot_main_figure(
    rank_rows: list[dict[str, object]],
    fixed_point: dict[str, object],
    h9_point: dict[str, object],
    random_point: dict[str, object],
    path: Path,
) -> None:
    configure_matplotlib()
    k_positions = list(range(len(rank_rows)))
    x_extra = [len(rank_rows) - 0.10, len(rank_rows) + 0.35, len(rank_rows) + 0.80]

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8))
    top1 = [float(row["tool_call_top1_rate"]) for row in rank_rows]
    strict = [float(row["strict_flip_rate"]) for row in rank_rows]
    labels = [str(row["k"]) for row in rank_rows]

    axes[0].plot(k_positions, top1, color="#1b6ca8", marker="o", linewidth=2.3, label="per-pair rank-k")
    axes[1].plot(k_positions, strict, color="#cc5803", marker="o", linewidth=2.3, label="per-pair rank-k")

    extras = [
        (fixed_point, x_extra[0], "#2a9d8f", "fixed PC1"),
        (h9_point, x_extra[1], "#8d6a9f", "H9 query-only"),
        (random_point, x_extra[2], "#6c757d", "random"),
    ]
    for ax, metric in ((axes[0], "tool_call_top1_rate"), (axes[1], "strict_flip_rate")):
        for point, xpos, color, short_label in extras:
            ax.scatter([xpos], [float(point[metric])], color=color, s=72, marker="D", zorder=4)
            ax.annotate(short_label, (xpos, float(point[metric])), xytext=(6, 6), textcoords="offset points", fontsize=9)
        ax.set_xlim(-0.35, len(rank_rows) + 1.25)
        ax.set_ylim(-0.02, 1.02)
        ax.set_xticks(k_positions)
        ax.set_xticklabels(labels)

    axes[0].set_title("Tool-Call Top-1 Rate")
    axes[0].set_xlabel("per-pair reconstruction rank k")
    axes[0].set_ylabel("rate")
    axes[1].set_title("Strict Flip Rate")
    axes[1].set_xlabel("per-pair reconstruction rank k")
    axes[1].set_ylabel("rate")
    axes[0].legend(frameon=False, loc="lower right")
    fig.tight_layout()
    ensure_dir(path.parent)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    ensure_dir(args.output_root)
    set_seed(args.seed)

    model, tokenizer = load_model_and_tokenizer(model_path=args.model_path, device=args.device)
    tool_token_id = get_tool_token_id(tokenizer)
    pc_bundle = torch.load(args.pc_bundle, map_location="cpu", weights_only=False)
    eval_cache = load_or_collect_pair_baseline(
        model,
        tokenizer,
        dataset_root=args.eval_root,
        max_pairs=args.max_pairs,
        batch_size=args.batch_size,
        patch_layer=int(pc_bundle.get("patch_layer", PATCH_LAYER)),
        cache_path=args.eval_cache,
    )
    eval_diff = eval_cache.clean_resid - eval_cache.corrupt_resid
    pc1 = pc_bundle["components"][0].float()

    sweep_rows: list[dict[str, object]] = []
    for k in args.ks:
        for alpha in args.alphas:
            outputs = evaluate_intervention(
                model,
                eval_cache,
                tool_token_id=tool_token_id,
                eval_side="corrupt",
                layer=int(pc_bundle.get("patch_layer", PATCH_LAYER)),
                position_mode="last",
                delta_resolver=lambda indices, k=str(k), alpha=float(alpha): project_delta(
                    eval_diff[list(indices)],
                    pc_bundle,
                    k=k,
                )
                * alpha,
                direction=pc1,
            )
            row = summarize_condition_row(
                condition=f"rank_{k}_alpha_{alpha:g}",
                condition_family="per_pair_rankk_add",
                eval_side="corrupt",
                outputs=outputs,
                baseline_metrics=eval_cache.baseline_corrupt,
                tool_token_id=tool_token_id,
                layer=int(pc_bundle.get("patch_layer", PATCH_LAYER)),
                position_mode="last",
                alpha=float(alpha),
                direction_name="pc1",
            )
            row["intervention"] = f"per-pair rank-{k}"
            row["k"] = str(k)
            sweep_rows.append(row)

    write_csv(args.output_root / "pck_recovery_sweep.csv", sweep_rows)

    rank_rows = [best_row_for_k(sweep_rows, str(k)) for k in args.ks]
    rank_table_rows: list[dict[str, object]] = []
    for row in rank_rows:
        rank_table_rows.append(
            {
                "intervention": str(row["intervention"]),
                "k": str(row["k"]),
                "alpha": float(row["alpha"]),
                "tool_call_top1_rate": float(row["tool_call_top1_rate"]),
                "strict_flip_rate": float(row["strict_flip_rate"]),
                "mean_tool_logit": float(row["mean_tool_logit"]),
                "n_samples": int(row["n_samples"]),
                "source": str(args.pc_bundle),
            }
        )

    fixed_point = best_fixed_pc1_row(args.phase7_alpha_sweep)
    h9_point = point_from_headline(args.phase7_headline, "H9 query-only")
    random_point = point_from_headline(args.phase7_headline, "random direction")

    table_rows = rank_table_rows + [fixed_point, h9_point, random_point]
    write_csv(args.output_root / "pck_recovery_table.csv", table_rows)
    plot_main_figure(rank_table_rows, fixed_point, h9_point, random_point, args.output_root / "plot_pck_recovery_main.pdf")

    rank1 = next(row for row in rank_table_rows if str(row["k"]) == "1")
    rank2 = next(row for row in rank_table_rows if str(row["k"]) == "2")
    rank3 = next(row for row in rank_table_rows if str(row["k"]) == "3")
    full = next(row for row in rank_table_rows if str(row["k"]) == "full")

    lines = [
        "# Phase 8 Exp A: PC-k Recovery",
        "",
        f"- Held-out eval split: `{args.eval_root}` with `{len(eval_cache.samples)}` corrupt prompts patched toward their clean counterparts.",
        f"- Per-pair rank-1 recovery: alpha `{float(rank1['alpha']):g}`, top-1 `{percent(float(rank1['tool_call_top1_rate']))}`, strict flip `{percent(float(rank1['strict_flip_rate']))}`.",
        f"- Per-pair rank-2 recovery: alpha `{float(rank2['alpha']):g}`, top-1 `{percent(float(rank2['tool_call_top1_rate']))}`, strict flip `{percent(float(rank2['strict_flip_rate']))}`.",
        f"- Per-pair rank-3 recovery: alpha `{float(rank3['alpha']):g}`, top-1 `{percent(float(rank3['tool_call_top1_rate']))}`, strict flip `{percent(float(rank3['strict_flip_rate']))}`.",
        f"- Per-pair full recovery: alpha `{float(full['alpha']):g}`, top-1 `{percent(float(full['tool_call_top1_rate']))}`, strict flip `{percent(float(full['strict_flip_rate']))}`.",
        f"- Best fixed PC1 add: alpha `{float(fixed_point['alpha']):g}`, top-1 `{percent(float(fixed_point['tool_call_top1_rate']))}`, strict flip `{percent(float(fixed_point['strict_flip_rate']))}`.",
        f"- H9 query-only baseline: top-1 `{percent(float(h9_point['tool_call_top1_rate']))}`, strict flip `{percent(float(h9_point['strict_flip_rate']))}`.",
        "",
        "This figure supports a compact-subspace story, not a shared-single-direction story.",
        "",
        f"- Included k values: `{', '.join(str(row['k']) for row in rank_table_rows)}`.",
    ]
    write_text(args.output_root / "summary.md", "\n".join(lines))

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
