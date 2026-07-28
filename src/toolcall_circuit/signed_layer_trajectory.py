#!/usr/bin/env python3
"""
Layer-wise margin trajectories for the signed circuit story.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm.auto import tqdm

if __package__ in {None, ""}:
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from toolcall_circuit.bidirectional_causal_eval import collect_cache_cpu_for_nodes, load_sample_paths
from toolcall_circuit.objective import DistributionObjective, build_bidirectional_endpoint_objectives
from toolcall_circuit.signed_circuit import derive_signed_groups
from toolcall_circuit.single_sample import load_hooked_qwen3, objective_from_logits, parse_head


def finite(values: Iterable[float]) -> List[float]:
    return [float(v) for v in values if isinstance(v, (int, float)) and math.isfinite(float(v))]


def median(values: Iterable[float]) -> float:
    vals = finite(values)
    return float(np.median(vals)) if vals else float("nan")


def build_patch_hooks(
    patch_nodes: Sequence[str],
    source_cache_cpu: Dict[str, torch.Tensor],
    device: str,
) -> List[tuple[str, object]]:
    heads_by_layer: Dict[int, List[int]] = {}
    mlp_layers: List[int] = []
    for node in patch_nodes:
        if node.startswith("MLP"):
            mlp_layers.append(int(node[3:]))
        else:
            layer, head = parse_head(node)
            heads_by_layer.setdefault(layer, []).append(head)

    hooks: List[tuple[str, object]] = []
    for layer, heads in heads_by_layer.items():
        cache_name = f"blocks.{layer}.attn.hook_z"
        src_act = source_cache_cpu[cache_name].to(device)
        uniq = sorted(set(heads))

        def make_head_hook(src: torch.Tensor, hs: Sequence[int]):
            def hook_fn(z: torch.Tensor, hook):  # noqa: ANN001
                out = z.clone()
                for h in hs:
                    out[:, -1, h, :] = src[:, -1, h, :]
                return out

            return hook_fn

        hooks.append((cache_name, make_head_hook(src_act, uniq)))

    for layer in sorted(set(mlp_layers)):
        cache_name = f"blocks.{layer}.hook_mlp_out"
        src_act = source_cache_cpu[cache_name].to(device)

        def make_mlp_hook(src: torch.Tensor):
            def hook_fn(mlp_out: torch.Tensor, hook):  # noqa: ANN001
                out = mlp_out.clone()
                out[:, -1, :] = src[:, -1, :]
                return out

            return hook_fn

        hooks.append((cache_name, make_mlp_hook(src_act)))

    return hooks


def run_layer_margins(
    model,
    tokens: torch.Tensor,
    tool_objective: DistributionObjective,
    no_tool_objective: DistributionObjective,
    patch_nodes: Sequence[str] | None = None,
    source_cache_cpu: Dict[str, torch.Tensor] | None = None,
) -> tuple[List[str], np.ndarray]:
    hooks = []
    if patch_nodes:
        if source_cache_cpu is None:
            raise ValueError("source_cache_cpu is required when patch_nodes is provided.")
        hooks = build_patch_hooks(patch_nodes, source_cache_cpu, str(tokens.device))

    with torch.no_grad():
        if hooks:
            with model.hooks(fwd_hooks=hooks):
                _, cache = model.run_with_cache(tokens)
        else:
            _, cache = model.run_with_cache(tokens)
        stack, labels = cache.accumulated_resid(apply_ln=False, pos_slice=-1, return_labels=True)
        stack = stack.permute(1, 0, 2).to(dtype=model.W_U.dtype)
        stack_final = model.ln_final(stack)
        stack_logits = model.unembed(stack_final)
    stage_logits = stack_logits[0]
    deltas: List[float] = []
    for idx in range(stage_logits.shape[0]):
        logits_one = stage_logits[idx].unsqueeze(0).unsqueeze(0)
        tool_score = float(objective_from_logits(logits_one, tool_objective).item())
        no_tool_score = float(objective_from_logits(logits_one, no_tool_objective).item())
        deltas.append(tool_score - no_tool_score)
    return labels, np.asarray(deltas, dtype=np.float32)


def plot_curves(layers: Sequence[int], curves: Dict[str, Sequence[float]], out_path: Path) -> None:
    style = {
        "tool_base": ("#c0392b", "tool base"),
        "tool_plus_symmetric": ("#e67e22", "tool base + symmetric"),
        "tool_plus_no_tool_bias": ("#2980b9", "tool base + symmetric + no-tool bias"),
        "no_tool_base": ("#34495e", "no-tool base"),
        "no_tool_plus_symmetric": ("#7f8c8d", "no-tool base + symmetric"),
        "no_tool_plus_tool_bias": ("#d35400", "no-tool base + symmetric + tool bias"),
    }
    plt.style.use("default")
    fig, ax = plt.subplots(figsize=(11.4, 5.8), constrained_layout=True)
    for key, vals in curves.items():
        color, label = style.get(key, ("#555555", key))
        ax.plot(layers, vals, linewidth=2.0, color=color, label=label)
    ax.axhline(0.0, color="#444444", linewidth=1.0)
    ax.set_xlabel("Layer / residual stage")
    ax.set_ylabel("Endpoint score delta: tool_call - no_tool")
    ax.set_title("Signed Circuit Layer Trajectory")
    if len(layers) > 12:
        step = max(1, len(layers) // 8)
        ax.set_xticks(list(layers[::step]))
    ax.legend(loc="best")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Layer-wise trajectory analysis for the signed circuit.")
    parser.add_argument("--forward-batch-root", type=str, required=True)
    parser.add_argument("--bidirectional-summary", type=str, required=True)
    parser.add_argument("--model-path", type=str, default="./external/models/Qwen3-1.7B")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max-samples", type=int, default=200)
    parser.add_argument("--output-root", type=str, required=True)
    args = parser.parse_args()

    out_root = Path(args.output_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    bidi = json.loads(Path(args.bidirectional_summary).resolve().read_text(encoding="utf-8"))
    groups = derive_signed_groups(bidi)
    symmetric = groups.get("symmetric_backbone", [])
    tool_bias = groups.get("tool_bias_backbone", [])
    no_tool_bias = groups.get("no_tool_bias_backbone", [])
    if not symmetric or not tool_bias or not no_tool_bias:
        raise ValueError("Expected symmetric_backbone, tool_bias_backbone, and no_tool_bias_backbone.")

    combos = {
        "tool_base": None,
        "tool_plus_symmetric": sorted(symmetric),
        "tool_plus_no_tool_bias": sorted(set(symmetric) | set(no_tool_bias)),
        "no_tool_base": None,
        "no_tool_plus_symmetric": sorted(symmetric),
        "no_tool_plus_tool_bias": sorted(set(symmetric) | set(tool_bias)),
    }

    forward_batch_root = Path(args.forward_batch_root).resolve()
    reverse_batch_root = Path(args.bidirectional_summary).resolve().parent.parent / "reverse_batch"
    samples = load_sample_paths(forward_batch_root, reverse_batch_root, max_samples=args.max_samples)

    all_nodes = sorted({n for nodes in combos.values() if nodes for n in nodes})
    model, tokenizer = load_hooked_qwen3(args.model_path, device=args.device, dtype=torch.bfloat16)
    _ = tokenizer

    labels_ref: List[str] | None = None
    curve_values: Dict[str, List[np.ndarray]] = {k: [] for k in combos}

    pbar = tqdm(samples, desc="Signed layer trajectory", dynamic_ncols=True)
    for sp in pbar:
        try:
            tool_text = sp.tool_call_prompt.read_text(encoding="utf-8")
            no_tool_text = sp.no_tool_prompt.read_text(encoding="utf-8")
        except Exception:
            continue

        tool_tokens = model.to_tokens(tool_text, prepend_bos=False)
        no_tool_tokens = model.to_tokens(no_tool_text, prepend_bos=False)
        if tool_tokens.shape != no_tool_tokens.shape:
            continue

        with torch.no_grad():
            tool_logits = model(tool_tokens)
            no_tool_logits = model(no_tool_tokens)
        tool_objective, no_tool_objective = build_bidirectional_endpoint_objectives(
            tool_logits,
            no_tool_logits,
            tokenizer=tokenizer,
        )

        tool_cache = collect_cache_cpu_for_nodes(model, tool_tokens, all_nodes)
        no_tool_cache = collect_cache_cpu_for_nodes(model, no_tool_tokens, all_nodes)

        config = [
            ("tool_base", tool_tokens, None, None),
            ("tool_plus_symmetric", tool_tokens, combos["tool_plus_symmetric"], no_tool_cache),
            ("tool_plus_no_tool_bias", tool_tokens, combos["tool_plus_no_tool_bias"], no_tool_cache),
            ("no_tool_base", no_tool_tokens, None, None),
            ("no_tool_plus_symmetric", no_tool_tokens, combos["no_tool_plus_symmetric"], tool_cache),
            ("no_tool_plus_tool_bias", no_tool_tokens, combos["no_tool_plus_tool_bias"], tool_cache),
        ]
        for key, tokens, patch_nodes, source_cache in config:
            labels, margins = run_layer_margins(
                model,
                tokens,
                tool_objective,
                no_tool_objective,
                patch_nodes=patch_nodes,
                source_cache_cpu=source_cache,
            )
            if labels_ref is None:
                labels_ref = list(labels)
            curve_values[key].append(margins)

        pbar.set_postfix(sample=sp.sample_id)

    if labels_ref is None:
        raise ValueError("No valid samples for signed layer trajectory.")

    summary_curves: Dict[str, List[float]] = {}
    for key, arrs in curve_values.items():
        if not arrs:
            continue
        arr = np.stack(arrs, axis=0)
        summary_curves[key] = [float(np.median(arr[:, i])) for i in range(arr.shape[1])]

    layers = list(range(len(labels_ref)))
    plot_curves(layers, summary_curves, out_root / "signed_layer_trajectory.png")

    report = {
        "n_samples": len(samples),
        "labels": labels_ref,
        "layers": layers,
        "curves": summary_curves,
        "artifacts": {
            "trajectory_png": str(out_root / "signed_layer_trajectory.png"),
            "summary_json": str(out_root / "signed_layer_trajectory_report.json"),
        },
    }
    (out_root / "signed_layer_trajectory_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
