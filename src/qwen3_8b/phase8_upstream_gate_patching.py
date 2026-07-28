#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm.auto import tqdm

from phase8_common import (
    DEFAULT_PC_BUNDLE,
    EVAL_DATASET_ROOT,
    EXP_E_ROOT,
    L24_LAYER,
    MODEL_PATH,
    build_pair_batches,
    configure_matplotlib,
    ensure_dir,
    get_tool_token_id,
    load_gate_direction,
    load_model_and_tokenizer,
    load_samples,
    make_last_token_vector_capture,
    make_last_token_vector_replace,
    manifest_pair_count,
    projection,
    set_seed,
    tool_stats,
    write_csv,
    write_text,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 8 Exp E: upstream causal patching to the L24 gate score.")
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH)
    parser.add_argument("--dataset-root", type=Path, default=EVAL_DATASET_ROOT)
    parser.add_argument("--pc-bundle", type=Path, default=DEFAULT_PC_BUNDLE)
    parser.add_argument("--output-root", type=Path, default=EXP_E_ROOT)
    parser.add_argument("--max-pairs", type=int, default=manifest_pair_count(EVAL_DATASET_ROOT))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--components", nargs="+", default=["resid_pre"])
    parser.add_argument("--max-layer", type=int, default=L24_LAYER)
    return parser.parse_args()


def hook_name_for(component: str, layer: int) -> str:
    if component == "resid_pre":
        return f"blocks.{layer}.hook_resid_pre"
    if component == "resid_mid":
        return f"blocks.{layer}.hook_resid_mid"
    if component == "resid_post":
        return f"blocks.{layer}.hook_resid_post"
    if component == "mlp_out":
        return f"blocks.{layer}.hook_mlp_out"
    raise ValueError(f"Unsupported component: {component}")


def valid_layers(component: str, max_layer: int) -> list[int]:
    if component == "resid_pre":
        return list(range(max_layer + 1))
    return list(range(max_layer))


def plot_patch_sweep(rows: list[dict[str, object]], path: Path) -> None:
    configure_matplotlib()
    components = sorted({str(row["component"]) for row in rows})
    component_colors = {
        "resid_pre": "#1b6ca8",
        "resid_mid": "#8d6a9f",
        "resid_post": "#6c757d",
        "mlp_out": "#cc5803",
    }

    fig, axes = plt.subplots(2, 1, figsize=(10.6, 7.6), sharex=True)
    for component in components:
        comp_rows = sorted(
            [row for row in rows if str(row["component"]) == component],
            key=lambda row: int(row["layer"]),
        )
        layers = [int(row["layer"]) for row in comp_rows]
        recovery = [float(row["mean_gate_recovery"]) for row in comp_rows]
        top1 = [float(row["tool_call_top1_rate"]) for row in comp_rows]
        color = component_colors.get(component, "#333333")
        axes[0].plot(layers, recovery, marker="o", linewidth=2.0, color=color, label=component)
        axes[1].plot(layers, top1, marker="o", linewidth=2.0, color=color, label=component)

    axes[0].axhline(0.0, color="black", linewidth=0.8, alpha=0.35)
    axes[0].set_ylabel("mean gate recovery")
    axes[0].set_title("Upstream Patching Recovery of the L24 Gate Score")
    axes[0].legend(frameon=False)

    axes[1].axhline(0.0, color="black", linewidth=0.8, alpha=0.35)
    axes[1].set_xlabel("patched layer")
    axes[1].set_ylabel("tool-call top-1 rate")
    axes[1].set_ylim(-0.02, 1.02)
    axes[1].set_title("Behavior After Gate-Targeted Upstream Patching")

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
    gate_direction = load_gate_direction(args.pc_bundle)
    samples = load_samples(args.dataset_root, model, tokenizer, max_pairs=args.max_pairs)
    pair_batches = build_pair_batches(samples, args.batch_size)

    gate_hook_name = f"blocks.{L24_LAYER}.hook_resid_pre"
    requested_components = [str(component) for component in args.components]
    component_layers = {component: valid_layers(component, args.max_layer) for component in requested_components}
    target_hooks = {component: [hook_name_for(component, layer) for layer in layers] for component, layers in component_layers.items()}

    per_sample_rows: list[dict[str, object]] = []
    summary_store: dict[tuple[str, int], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    progress = tqdm(pair_batches, desc="Gate patch sweep", dynamic_ncols=True)
    for batch in progress:
        clean_capture: dict[str, torch.Tensor] = {}
        corrupt_capture: dict[str, torch.Tensor] = {}
        baseline_clean_capture: dict[str, torch.Tensor] = {}
        baseline_corrupt_capture: dict[str, torch.Tensor] = {}

        clean_hooks = [(gate_hook_name, make_last_token_vector_capture(baseline_clean_capture, gate_hook_name))]
        corrupt_hooks = [(gate_hook_name, make_last_token_vector_capture(baseline_corrupt_capture, gate_hook_name))]
        for hooks_dict, hook_list in ((clean_capture, clean_hooks), (corrupt_capture, corrupt_hooks)):
            for component in requested_components:
                for hook_name in target_hooks[component]:
                    hook_list.append((hook_name, make_last_token_vector_capture(hooks_dict, hook_name)))

        clean_tokens = batch.clean_tokens_cpu.to(model.W_U.device)
        corrupt_tokens = batch.corrupt_tokens_cpu.to(model.W_U.device)
        with torch.no_grad():
            clean_logits = model.run_with_hooks(clean_tokens, fwd_hooks=clean_hooks)
            corrupt_logits = model.run_with_hooks(corrupt_tokens, fwd_hooks=corrupt_hooks)

        clean_gate_scores = projection(baseline_clean_capture[gate_hook_name], gate_direction)
        corrupt_gate_scores = projection(baseline_corrupt_capture[gate_hook_name], gate_direction)
        clean_tool_logit, clean_top1 = tool_stats(clean_logits, tool_token_id)
        corrupt_tool_logit, corrupt_top1 = tool_stats(corrupt_logits, tool_token_id)

        for component in requested_components:
            for layer in component_layers[component]:
                hook_name = hook_name_for(component, layer)
                patched_capture: dict[str, torch.Tensor] = {}
                hooks = [
                    (hook_name, make_last_token_vector_replace(clean_capture[hook_name])),
                    (gate_hook_name, make_last_token_vector_capture(patched_capture, gate_hook_name)),
                ]
                with torch.no_grad():
                    patched_logits = model.run_with_hooks(corrupt_tokens, fwd_hooks=hooks)

                patched_gate_scores = projection(patched_capture[gate_hook_name], gate_direction)
                patched_tool_logit, patched_top1 = tool_stats(patched_logits, tool_token_id)
                recovery = (patched_gate_scores - corrupt_gate_scores) / (clean_gate_scores - corrupt_gate_scores).clamp_min(1e-6)

                store = summary_store[(component, layer)]
                store["delta_gate_score"].extend((patched_gate_scores - corrupt_gate_scores).tolist())
                store["gate_recovery"].extend(recovery.tolist())
                store["delta_tool_logit"].extend((patched_tool_logit - corrupt_tool_logit).tolist())
                store["patched_tool_logit"].extend(patched_tool_logit.tolist())
                store["patched_is_tool"].extend((patched_top1 == tool_token_id).float().tolist())
                store["strict_flip"].extend(((corrupt_top1 != tool_token_id) & (patched_top1 == tool_token_id)).float().tolist())

                for local_idx, sample_idx in enumerate(batch.indices):
                    per_sample_rows.append(
                        {
                            "sample_id": samples[sample_idx].sample_id,
                            "component": component,
                            "layer": layer,
                            "clean_gate_score": float(clean_gate_scores[local_idx].item()),
                            "corrupt_gate_score": float(corrupt_gate_scores[local_idx].item()),
                            "patched_gate_score": float(patched_gate_scores[local_idx].item()),
                            "gate_recovery": float(recovery[local_idx].item()),
                            "delta_gate_score": float((patched_gate_scores - corrupt_gate_scores)[local_idx].item()),
                            "delta_tool_logit": float((patched_tool_logit - corrupt_tool_logit)[local_idx].item()),
                            "patched_is_tool_call_top1": int(patched_top1[local_idx].item() == tool_token_id),
                            "strict_flip": int(
                                (corrupt_top1[local_idx].item() != tool_token_id)
                                and (patched_top1[local_idx].item() == tool_token_id)
                            ),
                        }
                    )

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary_rows: list[dict[str, object]] = []
    for (component, layer), store in sorted(summary_store.items(), key=lambda item: (item[0][0], item[0][1])):
        summary_rows.append(
            {
                "component": component,
                "layer": layer,
                "mean_delta_gate_score": float(np.mean(store["delta_gate_score"])),
                "mean_gate_recovery": float(np.mean(store["gate_recovery"])),
                "mean_delta_tool_logit": float(np.mean(store["delta_tool_logit"])),
                "mean_patched_tool_logit": float(np.mean(store["patched_tool_logit"])),
                "tool_call_top1_rate": float(np.mean(store["patched_is_tool"])),
                "strict_flip_rate": float(np.mean(store["strict_flip"])),
                "n_samples": len(store["patched_is_tool"]),
            }
        )

    write_csv(args.output_root / "gate_score_patch_per_sample.csv", per_sample_rows)
    write_csv(args.output_root / "gate_score_patch_table.csv", summary_rows)
    plot_patch_sweep(summary_rows, args.output_root / "plot_gate_score_patch_sweep.pdf")

    best_row = max(summary_rows, key=lambda row: float(row["mean_gate_recovery"]))
    lines = [
        "# Phase 8 Exp E: Upstream Causal Patching to the L24 Gate Score",
        "",
        f"- Eval split: `{args.dataset_root}` with `{len(samples)}` held-out pairs.",
        f"- Components swept: `{', '.join(requested_components)}`.",
        (
            f"- Best gate recovery: `{best_row['component']}` at `L{int(best_row['layer'])}` "
            f"with recovery `{float(best_row['mean_gate_recovery']):.4f}`, "
            f"top-1 `{float(best_row['tool_call_top1_rate']):.2%}`, "
            f"strict flip `{float(best_row['strict_flip_rate']):.2%}`."
        ),
        "- Interpretation: high upstream gate recovery identifies layers that are causally close to L24 gate formation.",
    ]
    write_text(args.output_root / "summary.md", "\n".join(lines))

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
