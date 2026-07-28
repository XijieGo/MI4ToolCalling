#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from tqdm.auto import tqdm

from phase4_reviewer_strengthening import TOOL_CALL_STR, make_resid_last_capture
from phase7_l24_directionality_common import (
    DEFAULT_ALPHA_SWEEP_A,
    DEFAULT_N_COMPONENTS,
    DISCOVERY_DATASET_ROOT,
    EVAL_DATASET_ROOT,
    EXP_A_ROOT,
    PATCH_LAYER,
    baseline_summary_row,
    compute_pca,
    direction_scores,
    evaluate_intervention,
    load_or_collect_pair_baseline,
    manifest_pair_count,
    normalized_random_direction,
    project_delta,
    summarize_condition_row,
    tool_stats_with_margin,
)
from task_attention_path_analysis import (
    MODEL_PATH,
    build_pair_batches,
    clear_cuda,
    ensure_dir,
    load_samples,
    set_seed,
    write_csv,
    write_text,
)


SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from toolcall_circuit.single_sample import load_hooked_qwen3  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 7 Exp A: held-out fixed L24 direction sufficiency.")
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH)
    parser.add_argument("--discovery-root", type=Path, default=DISCOVERY_DATASET_ROOT)
    parser.add_argument("--eval-root", type=Path, default=EVAL_DATASET_ROOT)
    parser.add_argument("--output-root", type=Path, default=EXP_A_ROOT)
    parser.add_argument("--discovery-max-pairs", type=int, default=200)
    parser.add_argument("--eval-max-pairs", type=int, default=manifest_pair_count(EVAL_DATASET_ROOT))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--patch-layer", type=int, default=PATCH_LAYER)
    parser.add_argument("--n-components", type=int, default=DEFAULT_N_COMPONENTS)
    parser.add_argument("--alphas", type=float, nargs="+", default=list(DEFAULT_ALPHA_SWEEP_A))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def capture_discovery_residuals(
    model,
    tokenizer,
    *,
    dataset_root: Path,
    max_pairs: int,
    batch_size: int,
    patch_layer: int,
    tool_token_id: int,
) -> dict[str, object]:
    samples = load_samples(dataset_root, model, tokenizer, max_pairs=max_pairs)
    pair_batches = build_pair_batches(samples, batch_size)
    resid_hook_name = f"blocks.{patch_layer}.hook_resid_pre"
    d_model = int(model.cfg.d_model)
    clean_resid = torch.empty((len(samples), d_model), dtype=torch.float32)
    corrupt_resid = torch.empty((len(samples), d_model), dtype=torch.float32)
    clean_tool_logit = torch.empty(len(samples), dtype=torch.float32)
    corrupt_tool_logit = torch.empty(len(samples), dtype=torch.float32)
    clean_top1 = torch.empty(len(samples), dtype=torch.long)
    corrupt_top1 = torch.empty(len(samples), dtype=torch.long)

    progress = tqdm(pair_batches, desc="Discovery residual capture", dynamic_ncols=True)
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
        clean_tool_logit_batch, clean_top1_batch, _ = tool_stats_with_margin(clean_logits, tool_token_id)
        corrupt_tool_logit_batch, corrupt_top1_batch, _ = tool_stats_with_margin(corrupt_logits, tool_token_id)
        clean_tool_logit[batch.indices] = clean_tool_logit_batch
        corrupt_tool_logit[batch.indices] = corrupt_tool_logit_batch
        clean_top1[batch.indices] = clean_top1_batch
        corrupt_top1[batch.indices] = corrupt_top1_batch
        clear_cuda()

    return {
        "samples": samples,
        "clean_resid": clean_resid,
        "corrupt_resid": corrupt_resid,
        "clean_tool_logit": clean_tool_logit,
        "corrupt_tool_logit": corrupt_tool_logit,
        "clean_top1": clean_top1,
        "corrupt_top1": corrupt_top1,
    }


def plot_alpha_sweep(corrupt_rows: list[dict[str, object]], clean_rows: list[dict[str, object]], path: Path) -> None:
    main_corrupt = [row for row in corrupt_rows if str(row["condition_family"]) == "fixed_pc1_add"]
    rank1_corrupt = [row for row in corrupt_rows if str(row["condition_family"]) == "per_pair_rank1_add"]
    main_clean = [row for row in clean_rows if str(row["condition_family"]) == "fixed_pc1_subtract"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    def series(rows: list[dict[str, object]], key: str) -> tuple[list[float], list[float]]:
        ordered = sorted(rows, key=lambda row: float(row["alpha"]))
        return [float(row["alpha"]) for row in ordered], [float(row[key]) for row in ordered]

    x_fixed, y_fixed_top1 = series(main_corrupt, "tool_call_top1_rate")
    _, y_fixed_flip = series(main_corrupt, "strict_flip_rate")
    _, y_fixed_logit = series(main_corrupt, "mean_tool_logit")
    x_rank1, y_rank1_top1 = series(rank1_corrupt, "tool_call_top1_rate")
    _, y_rank1_flip = series(rank1_corrupt, "strict_flip_rate")
    _, y_rank1_logit = series(rank1_corrupt, "mean_tool_logit")
    x_clean, y_clean_top1 = series(main_clean, "tool_call_top1_rate")
    _, y_clean_drop = series(main_clean, "strict_drop_rate")
    _, y_clean_logit = series(main_clean, "mean_tool_logit")

    axes[0].plot(x_fixed, y_fixed_top1, marker="o", label="corrupt + PC1")
    axes[0].plot(x_rank1, y_rank1_top1, marker="s", label="corrupt + rank1 upper bound")
    axes[0].plot(x_clean, y_clean_top1, marker="^", label="clean - PC1")
    axes[0].set_title("Top-1 Rate")
    axes[0].set_xlabel("alpha")
    axes[0].set_ylabel("tool_call top-1 rate")
    axes[0].set_ylim(-0.02, 1.02)

    axes[1].plot(x_fixed, y_fixed_flip, marker="o", label="corrupt strict flip")
    axes[1].plot(x_rank1, y_rank1_flip, marker="s", label="rank1 strict flip")
    axes[1].plot(x_clean, y_clean_drop, marker="^", label="clean strict drop")
    axes[1].set_title("Effect Rate")
    axes[1].set_xlabel("alpha")
    axes[1].set_ylabel("flip / drop rate")
    axes[1].set_ylim(-0.02, 1.02)

    axes[2].plot(x_fixed, y_fixed_logit, marker="o", label="corrupt + PC1")
    axes[2].plot(x_rank1, y_rank1_logit, marker="s", label="corrupt + rank1 upper bound")
    axes[2].plot(x_clean, y_clean_logit, marker="^", label="clean - PC1")
    axes[2].set_title("Mean Tool Logit")
    axes[2].set_xlabel("alpha")
    axes[2].set_ylabel("mean <tool_call> logit")

    for ax in axes:
        ax.grid(alpha=0.25)
    axes[0].legend(frameon=False, fontsize=9)
    fig.tight_layout()
    ensure_dir(path.parent)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_control_bars(control_rows: list[dict[str, object]], path: Path) -> None:
    corrupt_rows = [row for row in control_rows if str(row["eval_side"]) == "corrupt"]
    clean_rows = [row for row in control_rows if str(row["eval_side"]) == "clean"]
    corrupt_labels = [str(row["condition"]) for row in corrupt_rows]
    clean_labels = [str(row["condition"]) for row in clean_rows]

    fig, axes = plt.subplots(2, 2, figsize=(16, 9))

    axes[0, 0].bar(range(len(corrupt_rows)), [float(row["tool_call_top1_rate"]) for row in corrupt_rows])
    axes[0, 0].set_title("Corrupt Controls: Top-1 Rate")
    axes[0, 0].set_ylim(0, 1.02)
    axes[0, 0].set_xticks(range(len(corrupt_rows)))
    axes[0, 0].set_xticklabels(corrupt_labels, rotation=45, ha="right")

    axes[0, 1].bar(range(len(corrupt_rows)), [float(row["mean_tool_logit"]) for row in corrupt_rows])
    axes[0, 1].set_title("Corrupt Controls: Mean Tool Logit")
    axes[0, 1].set_xticks(range(len(corrupt_rows)))
    axes[0, 1].set_xticklabels(corrupt_labels, rotation=45, ha="right")

    axes[1, 0].bar(range(len(clean_rows)), [float(row["tool_call_top1_rate"]) for row in clean_rows])
    axes[1, 0].set_title("Clean Controls: Top-1 Rate")
    axes[1, 0].set_ylim(0, 1.02)
    axes[1, 0].set_xticks(range(len(clean_rows)))
    axes[1, 0].set_xticklabels(clean_labels, rotation=45, ha="right")

    axes[1, 1].bar(range(len(clean_rows)), [0.0 if row["strict_drop_rate"] == "" else float(row["strict_drop_rate"]) for row in clean_rows])
    axes[1, 1].set_title("Clean Controls: Strict Drop Rate")
    axes[1, 1].set_ylim(0, 1.02)
    axes[1, 1].set_xticks(range(len(clean_rows)))
    axes[1, 1].set_xticklabels(clean_labels, rotation=45, ha="right")

    for ax in axes.ravel():
        ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    ensure_dir(path.parent)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def repeat_direction(direction: torch.Tensor, batch_size: int, *, scale: float) -> torch.Tensor:
    return direction.unsqueeze(0).repeat(batch_size, 1) * float(scale)


def main() -> None:
    args = parse_args()
    ensure_dir(args.output_root)
    set_seed(args.seed)

    model, tokenizer = load_hooked_qwen3(str(args.model_path), device=args.device, dtype=torch.bfloat16)
    tool_token_ids = tokenizer.encode(TOOL_CALL_STR, add_special_tokens=False)
    if len(tool_token_ids) != 1:
        raise ValueError(f"{TOOL_CALL_STR!r} maps to unexpected token ids: {tool_token_ids}")
    tool_token_id = int(tool_token_ids[0])

    discovery = capture_discovery_residuals(
        model,
        tokenizer,
        dataset_root=args.discovery_root,
        max_pairs=args.discovery_max_pairs,
        batch_size=args.batch_size,
        patch_layer=args.patch_layer,
        tool_token_id=tool_token_id,
    )
    diff_vectors = discovery["clean_resid"] - discovery["corrupt_resid"]
    pca = compute_pca(diff_vectors, n_components=args.n_components)
    components = pca["components"].float()
    mean_clean = discovery["clean_resid"].mean(dim=0)
    mean_corrupt = discovery["corrupt_resid"].mean(dim=0)
    mean_mid = 0.5 * (mean_clean + mean_corrupt)
    random_direction = normalized_random_direction(int(components.shape[1]), seed=args.seed + 1000)
    pc_bundle = {
        "patch_layer": int(args.patch_layer),
        "discovery_dataset_root": str(args.discovery_root),
        "discovery_sample_ids": [sample.sample_id for sample in discovery["samples"]],
        "components": components,
        "mean_diff": pca["mean_diff"].float(),
        "explained_variance": pca["explained_variance"].float(),
        "explained_variance_ratio": pca["explained_variance_ratio"].float(),
        "mean_clean": mean_clean.float(),
        "mean_corrupt": mean_corrupt.float(),
        "mean_mid": mean_mid.float(),
        "random_direction": random_direction.float(),
        "seed": int(args.seed),
    }
    torch.save(pc_bundle, args.output_root / "pc_bundle.pt")

    eval_cache = load_or_collect_pair_baseline(
        model,
        tokenizer,
        dataset_root=args.eval_root,
        max_pairs=args.eval_max_pairs,
        batch_size=args.batch_size,
        patch_layer=args.patch_layer,
        cache_path=args.output_root / "heldout_eval_cache.pt",
    )
    pc1 = components[0].float()
    eval_diff = eval_cache.clean_resid - eval_cache.corrupt_resid

    baseline_clean_score = direction_scores(eval_cache.clean_resid, pc1)
    baseline_corrupt_score = direction_scores(eval_cache.corrupt_resid, pc1)

    corrupt_rows: list[dict[str, object]] = [
        baseline_summary_row(
            condition="baseline_corrupt",
            condition_family="baseline_corrupt",
            eval_side="corrupt",
            baseline_metrics=eval_cache.baseline_corrupt,
            tool_token_id=tool_token_id,
            direction_score=baseline_corrupt_score,
            alpha=0.0,
        )
    ]
    clean_rows: list[dict[str, object]] = [
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

    for alpha in args.alphas:
        fixed_outputs = evaluate_intervention(
            model,
            eval_cache,
            tool_token_id=tool_token_id,
            eval_side="corrupt",
            layer=args.patch_layer,
            position_mode="last",
            delta_resolver=lambda indices, alpha=alpha: repeat_direction(pc1, len(indices), scale=alpha),
            direction=pc1,
        )
        corrupt_rows.append(
            summarize_condition_row(
                condition=f"corrupt_add_pc1_alpha_{alpha:g}",
                condition_family="fixed_pc1_add",
                eval_side="corrupt",
                outputs=fixed_outputs,
                baseline_metrics=eval_cache.baseline_corrupt,
                tool_token_id=tool_token_id,
                layer=args.patch_layer,
                position_mode="last",
                alpha=alpha,
                direction_name="pc1",
            )
        )

        rank1_outputs = evaluate_intervention(
            model,
            eval_cache,
            tool_token_id=tool_token_id,
            eval_side="corrupt",
            layer=args.patch_layer,
            position_mode="last",
            delta_resolver=lambda indices, alpha=alpha: project_delta(eval_diff[list(indices)], pc_bundle, k=1) * float(alpha),
            direction=pc1,
        )
        corrupt_rows.append(
            summarize_condition_row(
                condition=f"corrupt_add_rank1_alpha_{alpha:g}",
                condition_family="per_pair_rank1_add",
                eval_side="corrupt",
                outputs=rank1_outputs,
                baseline_metrics=eval_cache.baseline_corrupt,
                tool_token_id=tool_token_id,
                layer=args.patch_layer,
                position_mode="last",
                alpha=alpha,
                direction_name="rank1_projected_delta",
            )
        )

        clean_outputs = evaluate_intervention(
            model,
            eval_cache,
            tool_token_id=tool_token_id,
            eval_side="clean",
            layer=args.patch_layer,
            position_mode="last",
            delta_resolver=lambda indices, alpha=alpha: repeat_direction(pc1, len(indices), scale=-alpha),
            direction=pc1,
        )
        clean_rows.append(
            summarize_condition_row(
                condition=f"clean_subtract_pc1_alpha_{alpha:g}",
                condition_family="fixed_pc1_subtract",
                eval_side="clean",
                outputs=clean_outputs,
                baseline_metrics=eval_cache.baseline_clean,
                tool_token_id=tool_token_id,
                layer=args.patch_layer,
                position_mode="last",
                alpha=alpha,
                direction_name="pc1",
            )
        )

    write_csv(args.output_root / "alpha_sweep_corrupt.csv", corrupt_rows)
    write_csv(args.output_root / "alpha_sweep_clean.csv", clean_rows)

    best_fixed_corrupt = max(
        (row for row in corrupt_rows if str(row["condition_family"]) == "fixed_pc1_add"),
        key=lambda row: (float(row["tool_call_top1_rate"]), float(row["mean_tool_logit"])),
    )
    best_clean_subtract = max(
        (row for row in clean_rows if str(row["condition_family"]) == "fixed_pc1_subtract"),
        key=lambda row: (float(row["strict_drop_rate"]), -float(row["mean_tool_logit"])),
    )
    best_corrupt_alpha = float(best_fixed_corrupt["alpha"])
    best_clean_alpha = float(best_clean_subtract["alpha"])

    control_rows: list[dict[str, object]] = []
    direction_controls = [
        ("pc1", pc1),
        ("pc2", components[1].float()),
        ("pc3", components[2].float()),
        ("pc4", components[3].float()),
        ("pc5", components[4].float()),
        ("random", random_direction),
    ]

    for direction_name, direction in direction_controls:
        outputs = evaluate_intervention(
            model,
            eval_cache,
            tool_token_id=tool_token_id,
            eval_side="corrupt",
            layer=args.patch_layer,
            position_mode="last",
            delta_resolver=lambda indices, direction=direction: repeat_direction(direction, len(indices), scale=best_corrupt_alpha),
            direction=pc1 if direction_name == "pc1" else None,
        )
        control_rows.append(
            summarize_condition_row(
                condition=f"corrupt_last_{direction_name}_alpha_{best_corrupt_alpha:g}",
                condition_family="corrupt_direction_control",
                eval_side="corrupt",
                outputs=outputs,
                baseline_metrics=eval_cache.baseline_corrupt,
                tool_token_id=tool_token_id,
                layer=args.patch_layer,
                position_mode="last",
                alpha=best_corrupt_alpha,
                direction_name=direction_name,
            )
        )

    for layer in (20, 29):
        outputs = evaluate_intervention(
            model,
            eval_cache,
            tool_token_id=tool_token_id,
            eval_side="corrupt",
            layer=layer,
            position_mode="last",
            delta_resolver=lambda indices, alpha=best_corrupt_alpha: repeat_direction(pc1, len(indices), scale=alpha),
            direction=None,
        )
        control_rows.append(
            summarize_condition_row(
                condition=f"corrupt_wrong_layer_L{layer}_pc1",
                condition_family="corrupt_wrong_layer",
                eval_side="corrupt",
                outputs=outputs,
                baseline_metrics=eval_cache.baseline_corrupt,
                tool_token_id=tool_token_id,
                layer=layer,
                position_mode="last",
                alpha=best_corrupt_alpha,
                direction_name="pc1",
            )
        )

    for position_mode in ("all", "all_except_last"):
        outputs = evaluate_intervention(
            model,
            eval_cache,
            tool_token_id=tool_token_id,
            eval_side="corrupt",
            layer=args.patch_layer,
            position_mode=position_mode,
            delta_resolver=lambda indices, alpha=best_corrupt_alpha: repeat_direction(pc1, len(indices), scale=alpha),
            direction=pc1,
        )
        control_rows.append(
            summarize_condition_row(
                condition=f"corrupt_wrong_position_{position_mode}_pc1",
                condition_family="corrupt_wrong_position",
                eval_side="corrupt",
                outputs=outputs,
                baseline_metrics=eval_cache.baseline_corrupt,
                tool_token_id=tool_token_id,
                layer=args.patch_layer,
                position_mode=position_mode,
                alpha=best_corrupt_alpha,
                direction_name="pc1",
            )
        )

    for direction_name, direction in direction_controls:
        outputs = evaluate_intervention(
            model,
            eval_cache,
            tool_token_id=tool_token_id,
            eval_side="clean",
            layer=args.patch_layer,
            position_mode="last",
            delta_resolver=lambda indices, direction=direction: repeat_direction(direction, len(indices), scale=-best_clean_alpha),
            direction=pc1 if direction_name == "pc1" else None,
        )
        control_rows.append(
            summarize_condition_row(
                condition=f"clean_last_{direction_name}_alpha_{best_clean_alpha:g}",
                condition_family="clean_direction_control",
                eval_side="clean",
                outputs=outputs,
                baseline_metrics=eval_cache.baseline_clean,
                tool_token_id=tool_token_id,
                layer=args.patch_layer,
                position_mode="last",
                alpha=best_clean_alpha,
                direction_name=direction_name,
            )
        )

    for layer in (20, 29):
        outputs = evaluate_intervention(
            model,
            eval_cache,
            tool_token_id=tool_token_id,
            eval_side="clean",
            layer=layer,
            position_mode="last",
            delta_resolver=lambda indices, alpha=best_clean_alpha: repeat_direction(pc1, len(indices), scale=-alpha),
            direction=None,
        )
        control_rows.append(
            summarize_condition_row(
                condition=f"clean_wrong_layer_L{layer}_pc1",
                condition_family="clean_wrong_layer",
                eval_side="clean",
                outputs=outputs,
                baseline_metrics=eval_cache.baseline_clean,
                tool_token_id=tool_token_id,
                layer=layer,
                position_mode="last",
                alpha=best_clean_alpha,
                direction_name="pc1",
            )
        )

    for position_mode in ("all", "all_except_last"):
        outputs = evaluate_intervention(
            model,
            eval_cache,
            tool_token_id=tool_token_id,
            eval_side="clean",
            layer=args.patch_layer,
            position_mode=position_mode,
            delta_resolver=lambda indices, alpha=best_clean_alpha: repeat_direction(pc1, len(indices), scale=-alpha),
            direction=pc1,
        )
        control_rows.append(
            summarize_condition_row(
                condition=f"clean_wrong_position_{position_mode}_pc1",
                condition_family="clean_wrong_position",
                eval_side="clean",
                outputs=outputs,
                baseline_metrics=eval_cache.baseline_clean,
                tool_token_id=tool_token_id,
                layer=args.patch_layer,
                position_mode=position_mode,
                alpha=best_clean_alpha,
                direction_name="pc1",
            )
        )

    write_csv(args.output_root / "control_sweep.csv", control_rows)

    plot_alpha_sweep(corrupt_rows, clean_rows, args.output_root / "plot_fixed_direction_sufficiency.pdf")
    plot_control_bars(control_rows, args.output_root / "plot_fixed_direction_controls.pdf")

    best_rank1_corrupt = max(
        (row for row in corrupt_rows if str(row["condition_family"]) == "per_pair_rank1_add"),
        key=lambda row: (float(row["tool_call_top1_rate"]), float(row["mean_tool_logit"])),
    )
    ev_ratio = pc_bundle["explained_variance_ratio"]
    lines = [
        "# Exp A: Fixed L24 Direction Sufficiency",
        "",
        f"- Discovery split: `{args.discovery_root}` 前 `{args.discovery_max_pairs}` 对",
        f"- Held-out eval split: `{args.eval_root}` 前 `{args.eval_max_pairs}` 对",
        f"- Intervention site: `blocks.{args.patch_layer}.hook_resid_pre` prediction position",
        "",
        "## Discovery PCA",
        "",
        f"- PC1 explained variance ratio: `{float(ev_ratio[0].item()):.4f}`",
        f"- Top-3 cumulative explained variance ratio: `{float(ev_ratio[:3].sum().item()):.4f}`",
        f"- Held-out mean PC1 score: clean `{float(baseline_clean_score.mean().item()):.4f}` vs corrupt `{float(baseline_corrupt_score.mean().item()):.4f}`",
        "",
        "## Held-out Main Results",
        "",
        f"- Best fixed PC1 add: `alpha={best_corrupt_alpha:g}` -> top1 `{float(best_fixed_corrupt['tool_call_top1_rate']):.2%}`, strict flip `{float(best_fixed_corrupt['strict_flip_rate']):.2%}`, mean logit `{float(best_fixed_corrupt['mean_tool_logit']):.4f}`, H9 system attn `{float(best_fixed_corrupt['mean_system_attn_h9']):.4f}`, L33H29 DLA `{float(best_fixed_corrupt['mean_l33h29_dla']):.4f}`",
        f"- Best per-pair rank-1 upper bound: `alpha={float(best_rank1_corrupt['alpha']):g}` -> top1 `{float(best_rank1_corrupt['tool_call_top1_rate']):.2%}`, strict flip `{float(best_rank1_corrupt['strict_flip_rate']):.2%}`, mean logit `{float(best_rank1_corrupt['mean_tool_logit']):.4f}`",
        f"- Best clean subtraction: `alpha={best_clean_alpha:g}` -> top1 `{float(best_clean_subtract['tool_call_top1_rate']):.2%}`, strict drop `{float(best_clean_subtract['strict_drop_rate']):.2%}`, mean logit `{float(best_clean_subtract['mean_tool_logit']):.4f}`",
        "",
        "## Matched-Alpha Controls",
        "",
    ]
    corrupt_non_pc1 = [
        row for row in control_rows if str(row["condition_family"]) == "corrupt_direction_control" and str(row["direction_name"]) != "pc1"
    ]
    clean_non_pc1 = [
        row for row in control_rows if str(row["condition_family"]) == "clean_direction_control" and str(row["direction_name"]) != "pc1"
    ]
    best_control_corrupt = max(corrupt_non_pc1, key=lambda row: float(row["tool_call_top1_rate"]))
    best_control_clean = max(clean_non_pc1, key=lambda row: float(row["strict_drop_rate"]))
    lines.extend(
        [
            f"- Corrupt matched-alpha controls use `alpha={best_corrupt_alpha:g}`. Best non-PC1 direction: `{best_control_corrupt['condition']}` -> top1 `{float(best_control_corrupt['tool_call_top1_rate']):.2%}`, strict flip `{0.0 if best_control_corrupt['strict_flip_rate'] == '' else float(best_control_corrupt['strict_flip_rate']):.2%}`",
            f"- Clean matched-alpha controls use `alpha={best_clean_alpha:g}`. Best non-PC1 direction: `{best_control_clean['condition']}` -> strict drop `{0.0 if best_control_clean['strict_drop_rate'] == '' else float(best_control_clean['strict_drop_rate']):.2%}`",
        ]
    )
    write_text(args.output_root / "summary.md", "\n".join(lines))

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
