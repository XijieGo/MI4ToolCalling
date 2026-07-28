#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from phase4_reviewer_strengthening import TOOL_CALL_STR
from phase7_l24_directionality_common import (
    DEFAULT_ALPHA_SWEEP_B,
    EVAL_DATASET_ROOT,
    EXP_A_ROOT,
    EXP_B_ROOT,
    PHASE7_ROOT,
    PATCH_LAYER,
    baseline_summary_row,
    collect_pair_last_residuals,
    direction_scores,
    evaluate_intervention,
    load_or_collect_pair_baseline,
    manifest_pair_count,
    read_csv_rows,
    summarize_condition_row,
)
from task_attention_path_analysis import MODEL_PATH, ensure_dir, set_seed, write_csv, write_text


SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from toolcall_circuit.single_sample import load_hooked_qwen3  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 7 Exp B: L24 directional necessity on held-out clean prompts.")
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH)
    parser.add_argument("--eval-root", type=Path, default=EVAL_DATASET_ROOT)
    parser.add_argument("--output-root", type=Path, default=EXP_B_ROOT)
    parser.add_argument("--pc-bundle-path", type=Path, default=EXP_A_ROOT / "pc_bundle.pt")
    parser.add_argument("--eval-cache-path", type=Path, default=EXP_A_ROOT / "heldout_eval_cache.pt")
    parser.add_argument(
        "--phase4-query-root",
        type=Path,
        default=PROJECT_ROOT / "results" / "8b_main" / "phase4_reviewer" / "exp_h_query_specific",
        help="Fresh Phase 4 query-specific control output used for the H9 headline comparator.",
    )
    parser.add_argument("--eval-max-pairs", type=int, default=manifest_pair_count(EVAL_DATASET_ROOT))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--alphas", type=float, nargs="+", default=list(DEFAULT_ALPHA_SWEEP_B))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def raw_ablation_delta(resid: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
    coeff = torch.mv(resid.float(), direction.float())
    return -coeff.unsqueeze(1) * direction.unsqueeze(0)


def centered_ablation_delta(resid: torch.Tensor, direction: torch.Tensor, mean_mid: torch.Tensor) -> torch.Tensor:
    coeff = torch.mv((resid.float() - mean_mid.float().unsqueeze(0)), direction.float())
    return -coeff.unsqueeze(1) * direction.unsqueeze(0)


def repeat_direction(direction: torch.Tensor, batch_size: int, *, scale: float) -> torch.Tensor:
    return direction.unsqueeze(0).repeat(batch_size, 1) * float(scale)


def plot_main_results(main_rows: list[dict[str, object]], control_rows: list[dict[str, object]], path: Path) -> None:
    lookup = {str(row["condition"]): row for row in main_rows}
    signed_alpha1 = next(
        (row for row in main_rows if str(row["condition_family"]) == "signed_subtract_pc1" and float(row["alpha"]) == 1.0),
        None,
    )
    random_raw = next((row for row in control_rows if str(row["condition"]) == "clean_raw_ablation_random"), None)
    pc2_raw = next((row for row in control_rows if str(row["condition"]) == "clean_raw_ablation_pc2"), None)
    selected = [
        lookup["baseline_clean"],
        lookup["clean_raw_ablation_pc1"],
        lookup["clean_centered_ablation_pc1"],
        signed_alpha1 if signed_alpha1 is not None else lookup["clean_raw_ablation_pc1"],
        random_raw if random_raw is not None else lookup["clean_raw_ablation_pc1"],
        pc2_raw if pc2_raw is not None else lookup["clean_raw_ablation_pc1"],
    ]
    labels = [
        "baseline",
        "raw",
        "centered",
        "signed -1",
        "random raw",
        "PC2 raw",
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].bar(range(len(selected)), [float(row["tool_call_top1_rate"]) for row in selected])
    axes[0].set_title("Clean Top-1 Rate")
    axes[0].set_ylim(0, 1.02)
    axes[0].set_xticks(range(len(selected)))
    axes[0].set_xticklabels(labels, rotation=30, ha="right")

    axes[1].bar(range(len(selected)), [float(row["mean_tool_logit"]) for row in selected])
    axes[1].set_title("Clean Mean Tool Logit")
    axes[1].set_xticks(range(len(selected)))
    axes[1].set_xticklabels(labels, rotation=30, ha="right")

    for ax in axes:
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    ensure_dir(path.parent)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_control_results(control_rows: list[dict[str, object]], path: Path) -> None:
    clean_rows = [row for row in control_rows if str(row["eval_side"]) == "clean" and str(row["control_group"]) == "clean_controls"]
    mirror_rows = [row for row in control_rows if str(row["control_group"]) == "corrupt_mirror"]
    fig, axes = plt.subplots(2, 2, figsize=(16, 9))

    clean_labels = [str(row["condition"]) for row in clean_rows]
    mirror_labels = [str(row["condition"]) for row in mirror_rows]

    axes[0, 0].bar(range(len(clean_rows)), [float(row["tool_call_top1_rate"]) for row in clean_rows])
    axes[0, 0].set_title("Clean Controls: Top-1 Rate")
    axes[0, 0].set_ylim(0, 1.02)
    axes[0, 0].set_xticks(range(len(clean_rows)))
    axes[0, 0].set_xticklabels(clean_labels, rotation=45, ha="right")

    axes[0, 1].bar(
        range(len(clean_rows)),
        [0.0 if row["strict_drop_rate"] == "" else float(row["strict_drop_rate"]) for row in clean_rows],
    )
    axes[0, 1].set_title("Clean Controls: Strict Drop Rate")
    axes[0, 1].set_ylim(0, 1.02)
    axes[0, 1].set_xticks(range(len(clean_rows)))
    axes[0, 1].set_xticklabels(clean_labels, rotation=45, ha="right")

    axes[1, 0].bar(range(len(mirror_rows)), [float(row["tool_call_top1_rate"]) for row in mirror_rows])
    axes[1, 0].set_title("Corrupt Mirror: Top-1 Rate")
    axes[1, 0].set_ylim(0, 1.02)
    axes[1, 0].set_xticks(range(len(mirror_rows)))
    axes[1, 0].set_xticklabels(mirror_labels, rotation=45, ha="right")

    axes[1, 1].bar(range(len(mirror_rows)), [float(row["mean_tool_logit"]) for row in mirror_rows])
    axes[1, 1].set_title("Corrupt Mirror: Mean Tool Logit")
    axes[1, 1].set_xticks(range(len(mirror_rows)))
    axes[1, 1].set_xticklabels(mirror_labels, rotation=45, ha="right")

    for ax in axes.ravel():
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    ensure_dir(path.parent)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_direction_histogram(
    *,
    clean_scores: torch.Tensor,
    corrupt_scores: torch.Tensor,
    raw_clean_scores: torch.Tensor,
    shifted_corrupt_scores: torch.Tensor,
    path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    series = [
        ("baseline clean", clean_scores),
        ("baseline corrupt", corrupt_scores),
        ("clean after raw ablation", raw_clean_scores),
        ("corrupt after fixed add", shifted_corrupt_scores),
    ]
    all_values = torch.cat([tensor.float() for _, tensor in series])
    lo = float(all_values.min().item())
    hi = float(all_values.max().item())
    bins = 60

    for label, values in series:
        values = values.float()
        if float(values.std(unbiased=False).item()) < 1e-6:
            ax.axvline(float(values.mean().item()), label=label, linewidth=2.5)
        else:
            ax.hist(values.numpy(), bins=bins, range=(lo, hi), histtype="step", density=True, linewidth=2.0, label=label)

    ax.set_title("Direction Expression on Held-out Test")
    ax.set_xlabel("<x, v1>")
    ax.set_ylabel("density")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    ensure_dir(path.parent)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def build_headline_outputs(
    *,
    root: Path,
    exp_a_root: Path,
    phase4_query_root: Path,
    main_rows: list[dict[str, object]],
    control_rows: list[dict[str, object]],
) -> None:
    exp_a_rows = read_csv_rows(exp_a_root / "alpha_sweep_corrupt.csv")
    exp_a_controls = read_csv_rows(exp_a_root / "control_sweep.csv")
    fixed_best = max(
        (row for row in exp_a_rows if row["condition_family"] == "fixed_pc1_add"),
        key=lambda row: (float(row["tool_call_top1_rate"]), float(row["mean_tool_logit"])),
    )
    rank1_best = max(
        (row for row in exp_a_rows if row["condition_family"] == "per_pair_rank1_add"),
        key=lambda row: (float(row["tool_call_top1_rate"]), float(row["mean_tool_logit"])),
    )
    raw_row = next(row for row in main_rows if str(row["condition"]) == "clean_raw_ablation_pc1")
    centered_row = next(row for row in main_rows if str(row["condition"]) == "clean_centered_ablation_pc1")
    random_row = next(row for row in exp_a_controls if str(row["condition"]).startswith("corrupt_last_random_alpha_"))
    h9_rows = read_csv_rows(phase4_query_root / "query_specific_summary.csv")
    h9_row = next(row for row in h9_rows if row["condition"] == "h9_q_replace_exact")
    h9_baseline = next(row for row in h9_rows if row["condition"] == "baseline_corrupt")
    h9_per_sample_rows = read_csv_rows(
        phase4_query_root / "query_specific_per_sample.csv"
    )
    baseline_by_id = {
        row["sample_id"]: int(row["is_tool_call_top1"])
        for row in h9_per_sample_rows
        if row["condition"] == "baseline_corrupt"
    }
    h9_flip = sum(
        1
        for row in h9_per_sample_rows
        if row["condition"] == "h9_q_replace_exact"
        and baseline_by_id.get(row["sample_id"], 0) == 0
        and int(row["is_tool_call_top1"]) == 1
    ) / max(len(baseline_by_id), 1)

    headline_rows = []
    headline_rows.extend(
        [
            {
                "intervention": "L24 per-pair rank-1 patch",
                "split": "corrupt",
                "tool_call_top1_rate": float(rank1_best["tool_call_top1_rate"]),
                "strict_flip_or_drop_rate": float(rank1_best["strict_flip_rate"]),
                "meaning": "现有上界",
            },
            {
                "intervention": "L24 fixed PC1 add",
                "split": "corrupt",
                "tool_call_top1_rate": float(fixed_best["tool_call_top1_rate"]),
                "strict_flip_or_drop_rate": float(fixed_best["strict_flip_rate"]),
                "meaning": "shared-direction sufficiency",
            },
            {
                "intervention": "L24 raw ablation",
                "split": "clean",
                "tool_call_top1_rate": float(raw_row["tool_call_top1_rate"]),
                "strict_flip_or_drop_rate": float(raw_row["strict_drop_rate"]),
                "meaning": "directional necessity",
            },
            {
                "intervention": "L24 centered ablation",
                "split": "clean",
                "tool_call_top1_rate": float(centered_row["tool_call_top1_rate"]),
                "strict_flip_or_drop_rate": float(centered_row["strict_drop_rate"]),
                "meaning": "discriminative necessity",
            },
            {
                "intervention": "H9 query-only",
                "split": "corrupt",
                "tool_call_top1_rate": float(h9_row["tool_call_top1_rate"]),
                "strict_flip_or_drop_rate": float(h9_flip),
                "meaning": "partial bridge baseline",
            },
            {
                "intervention": "random direction",
                "split": "corrupt",
                "tool_call_top1_rate": float(random_row["tool_call_top1_rate"]),
                "strict_flip_or_drop_rate": float(random_row["strict_flip_rate"]),
                "meaning": "negative control",
            },
        ]
    )
    write_csv(root / "headline_table.csv", headline_rows)

    lines = [
        "# Phase 7 Headline Summary",
        "",
        f"- Fixed PC1 add best held-out result: top1 `{float(fixed_best['tool_call_top1_rate']):.2%}`, strict flip `{float(fixed_best['strict_flip_rate']):.2%}`",
        f"- Per-pair rank-1 upper bound: top1 `{float(rank1_best['tool_call_top1_rate']):.2%}`, strict flip `{float(rank1_best['strict_flip_rate']):.2%}`",
        f"- Raw necessity ablation: top1 `{float(raw_row['tool_call_top1_rate']):.2%}`, strict drop `{float(raw_row['strict_drop_rate']):.2%}`, mean logit delta `{float(raw_row['delta_tool_logit_vs_baseline']):+.4f}`",
        f"- Centered necessity ablation: top1 `{float(centered_row['tool_call_top1_rate']):.2%}`, strict drop `{float(centered_row['strict_drop_rate']):.2%}`, mean logit delta `{float(centered_row['delta_tool_logit_vs_baseline']):+.4f}`",
        f"- H9 query-only baseline: top1 `{float(h9_row['tool_call_top1_rate']):.2%}`, strict flip `{h9_flip:.2%}` over baseline corrupt top1 `{float(h9_baseline['tool_call_top1_rate']):.2%}`",
    ]
    write_text(root / "summary.md", "\n".join(lines))


def main() -> None:
    args = parse_args()
    ensure_dir(args.output_root)
    phase7_root = args.output_root.parent
    exp_a_root = args.pc_bundle_path.parent
    ensure_dir(phase7_root)
    set_seed(args.seed)

    if not args.pc_bundle_path.exists():
        raise FileNotFoundError(f"pc bundle not found: {args.pc_bundle_path}")
    pc_bundle = torch.load(args.pc_bundle_path, map_location="cpu", weights_only=False)

    model, tokenizer = load_hooked_qwen3(str(args.model_path), device=args.device, dtype=torch.bfloat16)
    tool_token_ids = tokenizer.encode(TOOL_CALL_STR, add_special_tokens=False)
    if len(tool_token_ids) != 1:
        raise ValueError(f"{TOOL_CALL_STR!r} maps to unexpected token ids: {tool_token_ids}")
    tool_token_id = int(tool_token_ids[0])

    eval_cache = load_or_collect_pair_baseline(
        model,
        tokenizer,
        dataset_root=args.eval_root,
        max_pairs=args.eval_max_pairs,
        batch_size=args.batch_size,
        patch_layer=PATCH_LAYER,
        cache_path=args.eval_cache_path,
    )

    pc1 = pc_bundle["components"][0].float()
    pc2 = pc_bundle["components"][1].float()
    pc3 = pc_bundle["components"][2].float()
    pc4 = pc_bundle["components"][3].float()
    pc5 = pc_bundle["components"][4].float()
    random_direction = pc_bundle["random_direction"].float()
    mean_mid = pc_bundle["mean_mid"].float()

    baseline_clean_score = direction_scores(eval_cache.clean_resid, pc1)
    baseline_corrupt_score = direction_scores(eval_cache.corrupt_resid, pc1)

    main_rows: list[dict[str, object]] = [
        baseline_summary_row(
            condition="baseline_clean",
            condition_family="baseline_clean",
            eval_side="clean",
            baseline_metrics=eval_cache.baseline_clean,
            tool_token_id=tool_token_id,
            direction_score=baseline_clean_score,
            alpha=0.0,
        )
    ]

    raw_outputs = evaluate_intervention(
        model,
        eval_cache,
        tool_token_id=tool_token_id,
        eval_side="clean",
        layer=PATCH_LAYER,
        position_mode="last",
        delta_resolver=lambda indices: raw_ablation_delta(eval_cache.clean_resid[list(indices)], pc1),
        direction=pc1,
    )
    main_rows.append(
        summarize_condition_row(
            condition="clean_raw_ablation_pc1",
            condition_family="raw_ablation_pc1",
            eval_side="clean",
            outputs=raw_outputs,
            baseline_metrics=eval_cache.baseline_clean,
            tool_token_id=tool_token_id,
            layer=PATCH_LAYER,
            position_mode="last",
            alpha=None,
            direction_name="pc1",
        )
    )

    centered_outputs = evaluate_intervention(
        model,
        eval_cache,
        tool_token_id=tool_token_id,
        eval_side="clean",
        layer=PATCH_LAYER,
        position_mode="last",
        delta_resolver=lambda indices: centered_ablation_delta(eval_cache.clean_resid[list(indices)], pc1, mean_mid),
        direction=pc1,
    )
    main_rows.append(
        summarize_condition_row(
            condition="clean_centered_ablation_pc1",
            condition_family="centered_ablation_pc1",
            eval_side="clean",
            outputs=centered_outputs,
            baseline_metrics=eval_cache.baseline_clean,
            tool_token_id=tool_token_id,
            layer=PATCH_LAYER,
            position_mode="last",
            alpha=None,
            direction_name="pc1",
        )
    )

    for alpha in args.alphas:
        signed_outputs = evaluate_intervention(
            model,
            eval_cache,
            tool_token_id=tool_token_id,
            eval_side="clean",
            layer=PATCH_LAYER,
            position_mode="last",
            delta_resolver=lambda indices, alpha=alpha: repeat_direction(pc1, len(indices), scale=-alpha),
            direction=pc1,
        )
        main_rows.append(
            summarize_condition_row(
                condition=f"clean_signed_subtract_pc1_alpha_{alpha:g}",
                condition_family="signed_subtract_pc1",
                eval_side="clean",
                outputs=signed_outputs,
                baseline_metrics=eval_cache.baseline_clean,
                tool_token_id=tool_token_id,
                layer=PATCH_LAYER,
                position_mode="last",
                alpha=alpha,
                direction_name="pc1",
            )
        )

    write_csv(args.output_root / "clean_necessity_results.csv", main_rows)

    best_signed = max(
        (row for row in main_rows if str(row["condition_family"]) == "signed_subtract_pc1"),
        key=lambda row: (float(row["strict_drop_rate"]), -float(row["mean_tool_logit"])),
    )
    best_signed_alpha = float(best_signed["alpha"])

    clean_l20, corrupt_l20 = collect_pair_last_residuals(model, eval_cache.pair_batches, layer=20)
    clean_l29, corrupt_l29 = collect_pair_last_residuals(model, eval_cache.pair_batches, layer=29)

    control_rows: list[dict[str, object]] = []

    def append_control(row: dict[str, object], group: str) -> None:
        row = dict(row)
        row["control_group"] = group
        control_rows.append(row)

    for condition_name, outputs in [
        (
            "clean_raw_ablation_pc1",
            raw_outputs,
        ),
        (
            "clean_centered_ablation_pc1",
            centered_outputs,
        ),
    ]:
        row = next(row for row in main_rows if str(row["condition"]) == condition_name)
        append_control(row, "clean_controls")

    signed_reference = next(
        row for row in main_rows if str(row["condition"]) == f"clean_signed_subtract_pc1_alpha_{best_signed_alpha:g}"
    )
    append_control(signed_reference, "clean_controls")

    for direction_name, direction in [
        ("random", random_direction),
        ("pc2", pc2),
        ("pc3", pc3),
        ("pc4", pc4),
        ("pc5", pc5),
    ]:
        outputs = evaluate_intervention(
            model,
            eval_cache,
            tool_token_id=tool_token_id,
            eval_side="clean",
            layer=PATCH_LAYER,
            position_mode="last",
            delta_resolver=lambda indices, direction=direction: raw_ablation_delta(eval_cache.clean_resid[list(indices)], direction),
            direction=None,
        )
        append_control(
            summarize_condition_row(
                condition=f"clean_raw_ablation_{direction_name}",
                condition_family="clean_raw_direction_control",
                eval_side="clean",
                outputs=outputs,
                baseline_metrics=eval_cache.baseline_clean,
                tool_token_id=tool_token_id,
                layer=PATCH_LAYER,
                position_mode="last",
                alpha=None,
                direction_name=direction_name,
            ),
            "clean_controls",
        )

    for layer, clean_resid in [(20, clean_l20), (29, clean_l29)]:
        outputs = evaluate_intervention(
            model,
            eval_cache,
            tool_token_id=tool_token_id,
            eval_side="clean",
            layer=layer,
            position_mode="last",
            delta_resolver=lambda indices, clean_resid=clean_resid: raw_ablation_delta(clean_resid[list(indices)], pc1),
            direction=None,
        )
        append_control(
            summarize_condition_row(
                condition=f"clean_raw_ablation_wrong_layer_L{layer}",
                condition_family="clean_wrong_layer",
                eval_side="clean",
                outputs=outputs,
                baseline_metrics=eval_cache.baseline_clean,
                tool_token_id=tool_token_id,
                layer=layer,
                position_mode="last",
                alpha=None,
                direction_name="pc1",
            ),
            "clean_controls",
        )

    for position_mode in ("all", "all_except_last"):
        outputs = evaluate_intervention(
            model,
            eval_cache,
            tool_token_id=tool_token_id,
            eval_side="clean",
            layer=PATCH_LAYER,
            position_mode=position_mode,
            delta_resolver=lambda indices: raw_ablation_delta(eval_cache.clean_resid[list(indices)], pc1),
            direction=pc1,
        )
        append_control(
            summarize_condition_row(
                condition=f"clean_raw_ablation_wrong_position_{position_mode}",
                condition_family="clean_wrong_position",
                eval_side="clean",
                outputs=outputs,
                baseline_metrics=eval_cache.baseline_clean,
                tool_token_id=tool_token_id,
                layer=PATCH_LAYER,
                position_mode=position_mode,
                alpha=None,
                direction_name="pc1",
            ),
            "clean_controls",
        )

    corrupt_raw_outputs = evaluate_intervention(
        model,
        eval_cache,
        tool_token_id=tool_token_id,
        eval_side="corrupt",
        layer=PATCH_LAYER,
        position_mode="last",
        delta_resolver=lambda indices: raw_ablation_delta(eval_cache.corrupt_resid[list(indices)], pc1),
        direction=pc1,
    )
    append_control(
        summarize_condition_row(
            condition="corrupt_raw_ablation_pc1",
            condition_family="corrupt_raw_mirror",
            eval_side="corrupt",
            outputs=corrupt_raw_outputs,
            baseline_metrics=eval_cache.baseline_corrupt,
            tool_token_id=tool_token_id,
            layer=PATCH_LAYER,
            position_mode="last",
            alpha=None,
            direction_name="pc1",
        ),
        "corrupt_mirror",
    )

    corrupt_centered_outputs = evaluate_intervention(
        model,
        eval_cache,
        tool_token_id=tool_token_id,
        eval_side="corrupt",
        layer=PATCH_LAYER,
        position_mode="last",
        delta_resolver=lambda indices: centered_ablation_delta(eval_cache.corrupt_resid[list(indices)], pc1, mean_mid),
        direction=pc1,
    )
    append_control(
        summarize_condition_row(
            condition="corrupt_centered_ablation_pc1",
            condition_family="corrupt_centered_mirror",
            eval_side="corrupt",
            outputs=corrupt_centered_outputs,
            baseline_metrics=eval_cache.baseline_corrupt,
            tool_token_id=tool_token_id,
            layer=PATCH_LAYER,
            position_mode="last",
            alpha=None,
            direction_name="pc1",
        ),
        "corrupt_mirror",
    )

    write_csv(args.output_root / "clean_necessity_controls.csv", control_rows)

    plot_main_results(main_rows, control_rows, args.output_root / "plot_clean_necessity_main.pdf")
    plot_control_results(control_rows, args.output_root / "plot_clean_necessity_controls.pdf")

    exp_a_rows = read_csv_rows(exp_a_root / "alpha_sweep_corrupt.csv")
    best_fixed_add = max(
        (row for row in exp_a_rows if row["condition_family"] == "fixed_pc1_add"),
        key=lambda row: (float(row["tool_call_top1_rate"]), float(row["mean_tool_logit"])),
    )
    best_fixed_alpha = float(best_fixed_add["alpha"])
    shifted_corrupt_scores = baseline_corrupt_score + best_fixed_alpha
    plot_direction_histogram(
        clean_scores=baseline_clean_score,
        corrupt_scores=baseline_corrupt_score,
        raw_clean_scores=raw_outputs["direction_score"],
        shifted_corrupt_scores=shifted_corrupt_scores,
        path=phase7_root / "direction_expression_histogram.pdf",
    )

    lines = [
        "# Exp B: Directional Necessity",
        "",
        f"- Held-out eval split: `{args.eval_root}` 前 `{args.eval_max_pairs}` 对",
        f"- PC bundle: `{args.pc_bundle_path}`",
        "",
        "## Main Results",
        "",
    ]
    raw_row = next(row for row in main_rows if str(row["condition"]) == "clean_raw_ablation_pc1")
    centered_row = next(row for row in main_rows if str(row["condition"]) == "clean_centered_ablation_pc1")
    lines.extend(
        [
            f"- Raw ablation: top1 `{float(raw_row['tool_call_top1_rate']):.2%}`, strict drop `{float(raw_row['strict_drop_rate']):.2%}`, mean logit `{float(raw_row['mean_tool_logit']):.4f}`, H9 delta `{float(raw_row['delta_system_attn_h9_vs_baseline']):+.4f}`, L33H29 DLA delta `{float(raw_row['delta_l33h29_dla_vs_baseline']):+.4f}`",
            f"- Centered ablation: top1 `{float(centered_row['tool_call_top1_rate']):.2%}`, strict drop `{float(centered_row['strict_drop_rate']):.2%}`, mean logit `{float(centered_row['mean_tool_logit']):.4f}`, H9 delta `{float(centered_row['delta_system_attn_h9_vs_baseline']):+.4f}`, L33H29 DLA delta `{float(centered_row['delta_l33h29_dla_vs_baseline']):+.4f}`",
            f"- Best signed subtraction: `alpha={best_signed_alpha:g}` -> top1 `{float(best_signed['tool_call_top1_rate']):.2%}`, strict drop `{float(best_signed['strict_drop_rate']):.2%}`, mean logit `{float(best_signed['mean_tool_logit']):.4f}`",
            "",
            "## Mirror Sanity Checks",
            "",
        ]
    )
    corrupt_raw_row = next(row for row in control_rows if str(row["condition"]) == "corrupt_raw_ablation_pc1")
    corrupt_centered_row = next(row for row in control_rows if str(row["condition"]) == "corrupt_centered_ablation_pc1")
    lines.extend(
        [
            f"- Corrupt raw ablation: top1 `{float(corrupt_raw_row['tool_call_top1_rate']):.2%}`, mean logit delta `{float(corrupt_raw_row['delta_tool_logit_vs_baseline']):+.4f}`",
            f"- Corrupt centered ablation: top1 `{float(corrupt_centered_row['tool_call_top1_rate']):.2%}`, mean logit delta `{float(corrupt_centered_row['delta_tool_logit_vs_baseline']):+.4f}`",
        ]
    )
    write_text(args.output_root / "summary.md", "\n".join(lines))

    build_headline_outputs(
        root=phase7_root,
        exp_a_root=exp_a_root,
        phase4_query_root=args.phase4_query_root,
        main_rows=main_rows,
        control_rows=control_rows,
    )

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
