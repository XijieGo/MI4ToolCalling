#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from phase7_l24_directionality_common import (
    EVAL_DATASET_ROOT,
    PATCH_LAYER,
    baseline_summary_row,
    evaluate_intervention,
    load_or_collect_pair_baseline,
    manifest_pair_count,
    project_delta,
    summarize_condition_row,
)
from phase8_common import (
    DEFAULT_HELDOUT_CACHE,
    DEFAULT_PC_BUNDLE,
    configure_matplotlib,
    ensure_dir,
    get_tool_token_id,
    load_model_and_tokenizer,
    percent,
    set_seed,
    write_csv,
    write_text,
)
from task_attention_path_analysis import MODEL_PATH


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = PROJECT_ROOT / "results" / "8b_main" / "phase10_necessity_decomposition"


CLEAN_CONDITIONS = [
    ("baseline_clean", "baseline"),
    ("subtract_mean_only", "mean"),
    ("subtract_mean_dir", "mean-dir"),
    ("subtract_v1_only", "v1-only"),
    ("subtract_both", "both"),
    ("subtract_mean_scaled_0.5", "0.5*mean"),
    ("subtract_mean_scaled_1.5", "1.5*mean"),
]

CORRUPT_CONDITIONS = [
    ("baseline_corrupt", "baseline"),
    ("add_mean_only", "mean"),
    ("add_mean_dir", "mean-dir"),
    ("add_v1_only", "v1-only"),
    ("add_both", "both"),
    ("add_mean_scaled_0.5", "0.5*mean"),
    ("add_mean_scaled_1.5", "1.5*mean"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 10: decompose the L24 rank-1 gate patch into mean vs. directional parts.")
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH)
    parser.add_argument("--eval-root", type=Path, default=EVAL_DATASET_ROOT)
    parser.add_argument("--pc-bundle", type=Path, default=DEFAULT_PC_BUNDLE)
    parser.add_argument("--eval-cache", type=Path, default=DEFAULT_HELDOUT_CACHE)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--max-pairs", type=int, default=manifest_pair_count(EVAL_DATASET_ROOT))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def repeat_vector(vector: torch.Tensor, batch_size: int, *, scale: float = 1.0) -> torch.Tensor:
    return vector.float().unsqueeze(0).repeat(batch_size, 1) * float(scale)


def remove_projection(resid: torch.Tensor, direction_unit: torch.Tensor) -> torch.Tensor:
    coeff = torch.mv(resid.float(), direction_unit.float())
    return -coeff.unsqueeze(1) * direction_unit.float().unsqueeze(0)


def centered_v1_component(diff_batch: torch.Tensor, mu_delta: torch.Tensor, v1: torch.Tensor) -> torch.Tensor:
    coeff = torch.mv((diff_batch.float() - mu_delta.float().unsqueeze(0)), v1.float())
    return coeff.unsqueeze(1) * v1.float().unsqueeze(0)


def lookup_row(rows: list[dict[str, object]], condition: str) -> dict[str, object]:
    for row in rows:
        if str(row["condition"]) == condition:
            return row
    raise KeyError(f"Missing condition={condition!r}")


def rate(row: dict[str, object], key: str) -> float:
    value = row[key]
    if value in {"", None}:
        return float("nan")
    return float(value)


def table_lines(
    rows: list[dict[str, object]],
    *,
    title: str,
    effect_key: str,
    order: list[tuple[str, str]],
) -> list[str]:
    lines = [f"## {title}", "", "| Condition | Top-1 | Effect | Mean tool logit | Delta vs baseline |", "|---|---:|---:|---:|---:|"]
    for condition, label in order:
        row = lookup_row(rows, condition)
        lines.append(
            "| "
            f"`{label}` | "
            f"{percent(rate(row, 'tool_call_top1_rate'))} | "
            f"{percent(rate(row, effect_key))} | "
            f"{rate(row, 'mean_tool_logit'):.4f} | "
            f"{rate(row, 'delta_tool_logit_vs_baseline'):+.4f} |"
        )
    lines.append("")
    return lines


def plot_top1_bars(
    clean_rows: list[dict[str, object]],
    corrupt_rows: list[dict[str, object]],
    path: Path,
) -> None:
    configure_matplotlib()
    clean_labels = [label for _, label in CLEAN_CONDITIONS]
    corrupt_labels = [label for _, label in CORRUPT_CONDITIONS]
    clean_values = [rate(lookup_row(clean_rows, condition), "tool_call_top1_rate") for condition, _ in CLEAN_CONDITIONS]
    corrupt_values = [rate(lookup_row(corrupt_rows, condition), "tool_call_top1_rate") for condition, _ in CORRUPT_CONDITIONS]

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.8))
    colors = ["#7f8c8d", "#2a9d8f", "#5b8ff9", "#e76f51", "#264653", "#8ab17d", "#cdb4db"]

    axes[0].bar(range(len(clean_values)), clean_values, color=colors[: len(clean_values)])
    axes[0].set_title("Clean: Tool-Call Top-1 After Subtraction")
    axes[0].set_ylabel("tool_call top-1 rate")
    axes[0].set_ylim(0.0, 1.02)
    axes[0].set_xticks(range(len(clean_labels)))
    axes[0].set_xticklabels(clean_labels, rotation=30, ha="right")

    axes[1].bar(range(len(corrupt_values)), corrupt_values, color=colors[: len(corrupt_values)])
    axes[1].set_title("Corrupt: Tool-Call Top-1 After Addition")
    axes[1].set_ylim(0.0, 1.02)
    axes[1].set_xticks(range(len(corrupt_labels)))
    axes[1].set_xticklabels(corrupt_labels, rotation=30, ha="right")

    for ax in axes:
        ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    ensure_dir(path.parent)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def build_summary(
    *,
    clean_rows: list[dict[str, object]],
    corrupt_rows: list[dict[str, object]],
    output_root: Path,
    model_path: Path,
    eval_root: Path,
    patch_layer: int,
    pc_bundle_path: Path,
    eval_cache_path: Path,
) -> None:
    add_mean = lookup_row(corrupt_rows, "add_mean_only")
    add_mean_dir = lookup_row(corrupt_rows, "add_mean_dir")
    add_v1 = lookup_row(corrupt_rows, "add_v1_only")
    add_both = lookup_row(corrupt_rows, "add_both")

    sub_mean = lookup_row(clean_rows, "subtract_mean_only")
    sub_mean_dir = lookup_row(clean_rows, "subtract_mean_dir")
    sub_v1 = lookup_row(clean_rows, "subtract_v1_only")
    sub_both = lookup_row(clean_rows, "subtract_both")

    mean_dir_match = (
        abs(rate(add_mean, "tool_call_top1_rate") - rate(add_mean_dir, "tool_call_top1_rate")) < 1e-9
        and abs(rate(add_mean, "mean_tool_logit") - rate(add_mean_dir, "mean_tool_logit")) < 1e-6
    )

    lines = [
        "# Phase 10 Necessity Decomposition",
        "",
        "## Scope",
        "",
        f"- Model: `{model_path}`",
        f"- Eval split: `{eval_root}` (`{int(clean_rows[0]['n_samples'])}` held-out pairs)",
        f"- Patch layer / position: `L{patch_layer}` prediction position only",
        f"- PC bundle: `{pc_bundle_path}`",
        f"- Eval cache: `{eval_cache_path}`",
        "",
    ]
    lines.extend(table_lines(clean_rows, title="Clean-Side Suppression", effect_key="strict_drop_rate", order=CLEAN_CONDITIONS))
    lines.extend(table_lines(corrupt_rows, title="Corrupt-Side Recovery", effect_key="strict_flip_rate", order=CORRUPT_CONDITIONS))
    lines.extend(
        [
            "## Headline",
            "",
            f"- `subtract_mean_only`: top-1 `{percent(rate(sub_mean, 'tool_call_top1_rate'))}`, strict drop `{percent(rate(sub_mean, 'strict_drop_rate'))}`.",
            f"- `subtract_mean_dir`: top-1 `{percent(rate(sub_mean_dir, 'tool_call_top1_rate'))}`, strict drop `{percent(rate(sub_mean_dir, 'strict_drop_rate'))}`.",
            f"- `subtract_v1_only`: top-1 `{percent(rate(sub_v1, 'tool_call_top1_rate'))}`, strict drop `{percent(rate(sub_v1, 'strict_drop_rate'))}`.",
            f"- `subtract_both`: top-1 `{percent(rate(sub_both, 'tool_call_top1_rate'))}`, strict drop `{percent(rate(sub_both, 'strict_drop_rate'))}`.",
            f"- `add_mean_only`: top-1 `{percent(rate(add_mean, 'tool_call_top1_rate'))}`, strict flip `{percent(rate(add_mean, 'strict_flip_rate'))}`.",
            f"- `add_v1_only`: top-1 `{percent(rate(add_v1, 'tool_call_top1_rate'))}`, strict flip `{percent(rate(add_v1, 'strict_flip_rate'))}`.",
            f"- `add_both`: top-1 `{percent(rate(add_both, 'tool_call_top1_rate'))}`, strict flip `{percent(rate(add_both, 'strict_flip_rate'))}`.",
            "",
            "## Notes",
            "",
            "- `add_mean_dir = ||mu_delta|| * mu_hat` is algebraically identical to `add_mean_only = mu_delta`.",
            f"- Observed `add_mean_dir` vs `add_mean_only` equality check: `{'passed' if mean_dir_match else 'failed'}`.",
            "- `subtract_mean_dir` is the closest local analogue of the Arditi-style directional ablation: remove the normalized mean-difference direction from the clean residual at a single layer-position.",
        ]
    )
    write_text(output_root / "summary.md", "\n".join(lines))


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

    mu_delta = (pc_bundle["mean_clean"] - pc_bundle["mean_corrupt"]).float()
    mu_hat = mu_delta / mu_delta.norm().clamp_min(1e-12)
    v1 = pc_bundle["components"][0].float()
    eval_diff = eval_cache.clean_resid - eval_cache.corrupt_resid

    patch_layer = int(pc_bundle.get("patch_layer", PATCH_LAYER))

    clean_rows: list[dict[str, object]] = [
        baseline_summary_row(
            condition="baseline_clean",
            condition_family="baseline_clean",
            eval_side="clean",
            baseline_metrics=eval_cache.baseline_clean,
            tool_token_id=tool_token_id,
            layer=patch_layer,
            position_mode="last",
            alpha=0.0,
            direction_name="none",
        )
    ]

    corrupt_rows: list[dict[str, object]] = [
        baseline_summary_row(
            condition="baseline_corrupt",
            condition_family="baseline_corrupt",
            eval_side="corrupt",
            baseline_metrics=eval_cache.baseline_corrupt,
            tool_token_id=tool_token_id,
            layer=patch_layer,
            position_mode="last",
            alpha=0.0,
            direction_name="none",
        )
    ]

    clean_specs = [
        (
            "subtract_mean_only",
            "subtract_mean_only",
            "mu_delta",
            lambda indices: repeat_vector(mu_delta, len(indices), scale=-1.0),
        ),
        (
            "subtract_mean_dir",
            "subtract_mean_dir",
            "mu_hat",
            lambda indices: remove_projection(eval_cache.clean_resid[list(indices)], mu_hat),
        ),
        (
            "subtract_v1_only",
            "subtract_v1_only",
            "v1_centered_component",
            lambda indices: -centered_v1_component(eval_diff[list(indices)], mu_delta, v1),
        ),
        (
            "subtract_both",
            "subtract_both",
            "rank1_projected_delta",
            lambda indices: -project_delta(eval_diff[list(indices)], pc_bundle, k=1),
        ),
        (
            "subtract_mean_scaled_0.5",
            "subtract_mean_scaled",
            "mu_delta",
            lambda indices: repeat_vector(mu_delta, len(indices), scale=-0.5),
        ),
        (
            "subtract_mean_scaled_1.5",
            "subtract_mean_scaled",
            "mu_delta",
            lambda indices: repeat_vector(mu_delta, len(indices), scale=-1.5),
        ),
    ]

    corrupt_specs = [
        (
            "add_mean_only",
            "add_mean_only",
            "mu_delta",
            lambda indices: repeat_vector(mu_delta, len(indices), scale=1.0),
        ),
        (
            "add_mean_dir",
            "add_mean_dir",
            "mu_hat",
            lambda indices: repeat_vector(mu_hat * mu_delta.norm(), len(indices), scale=1.0),
        ),
        (
            "add_v1_only",
            "add_v1_only",
            "v1_centered_component",
            lambda indices: centered_v1_component(eval_diff[list(indices)], mu_delta, v1),
        ),
        (
            "add_both",
            "add_both",
            "rank1_projected_delta",
            lambda indices: project_delta(eval_diff[list(indices)], pc_bundle, k=1),
        ),
        (
            "add_mean_scaled_0.5",
            "add_mean_scaled",
            "mu_delta",
            lambda indices: repeat_vector(mu_delta, len(indices), scale=0.5),
        ),
        (
            "add_mean_scaled_1.5",
            "add_mean_scaled",
            "mu_delta",
            lambda indices: repeat_vector(mu_delta, len(indices), scale=1.5),
        ),
    ]

    for condition, family, direction_name, delta_resolver in clean_specs:
        outputs = evaluate_intervention(
            model,
            eval_cache,
            tool_token_id=tool_token_id,
            eval_side="clean",
            layer=patch_layer,
            position_mode="last",
            delta_resolver=delta_resolver,
            direction=None,
        )
        alpha = None
        if condition.endswith("_0.5"):
            alpha = 0.5
        elif condition.endswith("_1.5"):
            alpha = 1.5
        clean_rows.append(
            summarize_condition_row(
                condition=condition,
                condition_family=family,
                eval_side="clean",
                outputs=outputs,
                baseline_metrics=eval_cache.baseline_clean,
                tool_token_id=tool_token_id,
                layer=patch_layer,
                position_mode="last",
                alpha=alpha,
                direction_name=direction_name,
            )
        )

    for condition, family, direction_name, delta_resolver in corrupt_specs:
        outputs = evaluate_intervention(
            model,
            eval_cache,
            tool_token_id=tool_token_id,
            eval_side="corrupt",
            layer=patch_layer,
            position_mode="last",
            delta_resolver=delta_resolver,
            direction=None,
        )
        alpha = None
        if condition.endswith("_0.5"):
            alpha = 0.5
        elif condition.endswith("_1.5"):
            alpha = 1.5
        corrupt_rows.append(
            summarize_condition_row(
                condition=condition,
                condition_family=family,
                eval_side="corrupt",
                outputs=outputs,
                baseline_metrics=eval_cache.baseline_corrupt,
                tool_token_id=tool_token_id,
                layer=patch_layer,
                position_mode="last",
                alpha=alpha,
                direction_name=direction_name,
            )
        )

    write_csv(args.output_root / "clean_interventions.csv", clean_rows)
    write_csv(args.output_root / "corrupt_interventions.csv", corrupt_rows)
    plot_top1_bars(clean_rows, corrupt_rows, args.output_root / "plot_decomposition.pdf")
    build_summary(
        clean_rows=clean_rows,
        corrupt_rows=corrupt_rows,
        output_root=args.output_root,
        model_path=args.model_path,
        eval_root=args.eval_root,
        patch_layer=patch_layer,
        pc_bundle_path=args.pc_bundle,
        eval_cache_path=args.eval_cache,
    )

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
