#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gc
import json
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from multiscale_common import (
    DEFAULT_DATASET_ROOT,
    build_pair_batches,
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze whether the gate is mu_delta and plot its formation trajectory.")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--size-label", type=str, required=True)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--eval-split", type=str, default="test")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--pc-bundle", type=Path, required=True)
    parser.add_argument("--max-pairs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.5, 1.0, 1.5, 2.0])
    return parser.parse_args()


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


def load_gate_bundle(path: Path) -> dict[str, object]:
    return torch.load(path, map_location="cpu", weights_only=False)


def infer_hook_kind(bundle: dict[str, object]) -> str:
    return "pre" if "patch_layer" in bundle else "post"


def gate_layer_from_bundle(bundle: dict[str, object]) -> int:
    if "layer" in bundle:
        return int(bundle["layer"])
    if "patch_layer" in bundle:
        return int(bundle["patch_layer"])
    raise KeyError("Could not infer gate layer from pca bundle.")


def extract_mu_delta(bundle: dict[str, object]) -> torch.Tensor:
    mean_diff = bundle["mean_diff"]
    if not isinstance(mean_diff, torch.Tensor):
        mean_diff = torch.tensor(mean_diff)
    mean_diff = mean_diff.detach().cpu().float().view(-1)
    return mean_diff


def extract_pc1(bundle: dict[str, object]) -> torch.Tensor | None:
    components = bundle.get("components")
    if components is None:
        return None
    if not isinstance(components, torch.Tensor):
        components = torch.tensor(components)
    components = components.detach().cpu().float()
    if components.ndim != 2:
        return None
    if components.shape[0] <= components.shape[1]:
        return components[0].contiguous().view(-1)
    return components[:, 0].contiguous().view(-1)


def unit(vector: torch.Tensor) -> torch.Tensor:
    return vector / vector.norm().clamp_min(1e-12)


def make_last_token_add_hook(delta_cpu: torch.Tensor):
    def hook_fn(value: torch.Tensor, hook):  # noqa: ANN001
        out = value.clone()
        delta = delta_cpu.to(device=value.device, dtype=value.dtype)
        if delta.ndim == 1:
            delta = delta.view(1, -1)
        out[:, -1, :] = out[:, -1, :] + delta
        return out

    return hook_fn


def projection(scores: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
    return torch.mv(scores.float(), direction.float())


def cosine_to_direction(scores: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
    direction_batch = direction.float().unsqueeze(0).expand_as(scores)
    return F.cosine_similarity(scores.float(), direction_batch, dim=-1)


def make_last_token_vector_capture(capture: dict[str, torch.Tensor], key: str):
    def hook_fn(value: torch.Tensor, hook):  # noqa: ANN001
        capture[key] = value[:, -1, :].detach().cpu().float()
        return value

    return hook_fn


def collect_pair_last_token_activations(
    model,
    pair_batches,
    *,
    hook_names: Sequence[str],
    side: str,
    desc: str,
) -> dict[str, torch.Tensor]:
    if side not in {"clean", "corrupt"}:
        raise ValueError(f"Unsupported side: {side}")

    n_samples = sum(len(batch.indices) for batch in pair_batches)
    d_model = int(model.cfg.d_model)
    outputs = {name: torch.empty((n_samples, d_model), dtype=torch.float32) for name in hook_names}

    progress = tqdm(pair_batches, desc=desc, dynamic_ncols=True)
    for batch in progress:
        capture: dict[str, torch.Tensor] = {}
        hooks = [(name, make_last_token_vector_capture(capture, name)) for name in hook_names]
        tokens_cpu = batch.clean_tokens_cpu if side == "clean" else batch.corrupt_tokens_cpu
        with torch.no_grad():
            _ = model.run_with_hooks(tokens_cpu.to(model.W_U.device), fwd_hooks=hooks)
        for name in hook_names:
            outputs[name][batch.indices] = capture[name]
        clear_cuda()
        progress.set_postfix(tok=batch.token_len)
    return outputs


def find_first_threshold_layer(values: Sequence[float], *, threshold_ratio: float = 0.1) -> int | None:
    if not values:
        return None
    final = float(values[-1])
    if np.isclose(final, 0.0):
        return None
    threshold = threshold_ratio * final
    for idx, value in enumerate(values):
        if value >= threshold:
            return idx
    return None


def largest_jump(values: Sequence[float]) -> tuple[int, float] | None:
    if len(values) < 2:
        return None
    deltas = [float(values[idx + 1] - values[idx]) for idx in range(len(values) - 1)]
    layer = int(np.argmax(deltas))
    return layer, float(deltas[layer])


def stage_abs_mass(values: Sequence[float], stage: range) -> float:
    return float(sum(abs(float(values[idx])) for idx in stage if idx < len(values)))


def plot_trajectory(rows: list[dict[str, object]], path: Path, *, gate_layer: int) -> None:
    configure_matplotlib()
    layers = [int(row["layer"]) for row in rows]
    mean_proj = [float(row["mean_projection"]) for row in rows]
    std_proj = [float(row["std_projection"]) for row in rows]
    mean_cos = [float(row["mean_cosine"]) for row in rows]
    mean_norm = [float(row["mean_delta_norm"]) for row in rows]

    fig, axes = plt.subplots(2, 1, figsize=(10.5, 7.2), sharex=True)

    axes[0].plot(layers, mean_proj, color="#1b6ca8", marker="o", linewidth=2.2)
    axes[0].fill_between(
        layers,
        np.asarray(mean_proj) - np.asarray(std_proj),
        np.asarray(mean_proj) + np.asarray(std_proj),
        color="#1b6ca8",
        alpha=0.16,
        linewidth=0,
    )
    axes[0].axhline(0.0, color="black", linewidth=0.8, alpha=0.35)
    axes[0].axvline(gate_layer, color="black", linestyle="--", linewidth=1.0, alpha=0.45)
    axes[0].set_ylabel(r"mean $\langle \Delta^{(l)}, \hat{\mu}_\Delta \rangle$")
    axes[0].set_title("Projection onto the Final Gate Direction")

    ax2 = axes[1]
    ax2b = ax2.twinx()
    line1 = ax2.plot(layers, mean_cos, color="#cc5803", marker="o", linewidth=2.0, label="mean cosine")
    line2 = ax2b.plot(layers, mean_norm, color="#2a9d8f", marker="s", linewidth=2.0, label="mean ||delta||")
    ax2.axvline(gate_layer, color="black", linestyle="--", linewidth=1.0, alpha=0.45)
    ax2.axhline(0.0, color="black", linewidth=0.8, alpha=0.35)
    ax2.set_ylabel("mean cosine")
    ax2b.set_ylabel("mean ||delta||")
    ax2.set_xlabel("layer")
    ax2.set_title("Alignment vs Magnitude")

    handles = line1 + line2
    labels = [line.get_label() for line in handles]
    ax2.legend(handles, labels, frameon=False, loc="upper left")
    fig.tight_layout()
    ensure_dir(path.parent)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_components(rows: list[dict[str, object]], path: Path) -> None:
    configure_matplotlib()
    layers = [int(row["layer"]) for row in rows]
    attn = [float(row["mean_attn_contrib"]) for row in rows]
    mlp = [float(row["mean_mlp_contrib"]) for row in rows]
    cum_attn = [float(row["cumulative_attn"]) for row in rows]
    cum_mlp = [float(row["cumulative_mlp"]) for row in rows]
    cum_total = [float(row["cumulative_total"]) for row in rows]

    fig, axes = plt.subplots(2, 1, figsize=(10.8, 7.5), sharex=True)
    width = 0.38
    axes[0].bar(np.asarray(layers) - width / 2, attn, width=width, color="#1b6ca8", label="attention")
    axes[0].bar(np.asarray(layers) + width / 2, mlp, width=width, color="#cc5803", label="MLP")
    axes[0].axhline(0.0, color="black", linewidth=0.8, alpha=0.35)
    axes[0].set_ylabel(r"mean signed contribution to $\hat{\mu}_\Delta$")
    axes[0].set_title("Per-Layer Attention / MLP Contributions into the Gate")
    axes[0].legend(frameon=False)

    axes[1].plot(layers, cum_attn, color="#1b6ca8", marker="o", linewidth=2.0, label="cumulative attention")
    axes[1].plot(layers, cum_mlp, color="#cc5803", marker="s", linewidth=2.0, label="cumulative MLP")
    axes[1].plot(layers, cum_total, color="#2a9d8f", marker="^", linewidth=2.2, label="cumulative total")
    axes[1].axhline(0.0, color="black", linewidth=0.8, alpha=0.35)
    axes[1].set_xlabel("layer")
    axes[1].set_ylabel("cumulative mean contribution")
    axes[1].set_title("Cumulative Gate Formation")
    axes[1].legend(frameon=False, loc="upper left")

    fig.tight_layout()
    ensure_dir(path.parent)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def gate_hook_name(layer: int, hook_kind: str) -> str:
    return f"blocks.{layer}.hook_resid_{hook_kind}"


def trajectory_hook_names(gate_layer: int, hook_kind: str) -> list[str]:
    return [gate_hook_name(layer, hook_kind) for layer in range(gate_layer + 1)]


def component_hook_names(gate_layer: int, hook_kind: str) -> tuple[list[str], list[str], list[str]]:
    max_layer = gate_layer if hook_kind == "post" else gate_layer - 1
    if max_layer < 0:
        return [], [], []
    pre_hooks = [f"blocks.{layer}.hook_resid_pre" for layer in range(max_layer + 1)]
    mid_hooks = [f"blocks.{layer}.hook_resid_mid" for layer in range(max_layer + 1)]
    post_hooks = [f"blocks.{layer}.hook_resid_post" for layer in range(max_layer + 1)]
    return pre_hooks, mid_hooks, post_hooks


def run_fixed_direction_sweep(model, pairs, *, hook_name: str, mu_delta: torch.Tensor, tool_token_id: int, alphas: Sequence[float], batch_size: int) -> list[dict[str, object]]:
    pair_batches = build_pair_batches(pairs, batch_size=batch_size)
    rows: list[dict[str, object]] = []
    for alpha in alphas:
        tool_top1 = 0
        strict_flip = 0
        logit_sum = 0.0
        prob_sum = 0.0
        count = 0
        progress = tqdm(pair_batches, desc=f"mu_delta alpha={alpha:g}", dynamic_ncols=True)
        for batch in progress:
            corrupt_tokens = batch.corrupt_tokens_cpu.to(model.W_U.device)
            hooks = [(hook_name, make_last_token_add_hook(mu_delta * float(alpha)))]
            with torch.no_grad():
                corrupt_logits = model(corrupt_tokens)
                patched_logits = model.run_with_hooks(corrupt_tokens, fwd_hooks=hooks)
            _base_logit, _base_prob, base_top1 = tool_stats(corrupt_logits, tool_token_id)
            patched_logit, patched_prob, patched_top1 = tool_stats(patched_logits, tool_token_id)
            tool_top1 += int((patched_top1 == tool_token_id).sum().item())
            strict_flip += int(((base_top1 != tool_token_id) & (patched_top1 == tool_token_id)).sum().item())
            logit_sum += float(patched_logit.sum().item())
            prob_sum += float(patched_prob.sum().item())
            count += int(patched_top1.shape[0])
            del corrupt_tokens, corrupt_logits, patched_logits, base_top1, patched_logit, patched_prob, patched_top1
            clear_cuda()
            progress.set_postfix(tok=batch.token_len)
        rows.append(
            {
                "direction": "mu_delta",
                "alpha": float(alpha),
                "n": count,
                "tool_call_top1_rate": float(tool_top1 / max(count, 1)),
                "strict_flip_rate": float(strict_flip / max(count, 1)),
                "mean_tool_call_logit": float(logit_sum / max(count, 1)),
                "mean_tool_call_prob": float(prob_sum / max(count, 1)),
            }
        )
    return rows


def build_trajectory_outputs(model, pairs, *, hook_kind: str, gate_layer: int, gate_direction_unit: torch.Tensor, batch_size: int) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    pair_batches = build_pair_batches(pairs, batch_size=batch_size)

    traj_hooks = trajectory_hook_names(gate_layer, hook_kind)
    clean_traj = collect_pair_last_token_activations(model, pair_batches, hook_names=traj_hooks, side="clean", desc=f"{hook_kind} clean trajectory")
    corrupt_traj = collect_pair_last_token_activations(model, pair_batches, hook_names=traj_hooks, side="corrupt", desc=f"{hook_kind} corrupt trajectory")

    per_sample_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for layer, hook_name in enumerate(traj_hooks):
        clean_resid = clean_traj[hook_name]
        corrupt_resid = corrupt_traj[hook_name]
        delta = clean_resid - corrupt_resid
        proj = projection(delta, gate_direction_unit)
        cos = cosine_to_direction(delta, gate_direction_unit)
        norm = delta.norm(dim=-1)
        clean_score = projection(clean_resid, gate_direction_unit)
        corrupt_score = projection(corrupt_resid, gate_direction_unit)

        for sample_idx, pair in enumerate(pairs):
            per_sample_rows.append(
                {
                    "sample_id": pair.sample_id,
                    "layer": layer,
                    "projection": float(proj[sample_idx].item()),
                    "cosine": float(cos[sample_idx].item()),
                    "delta_norm": float(norm[sample_idx].item()),
                    "clean_gate_score": float(clean_score[sample_idx].item()),
                    "corrupt_gate_score": float(corrupt_score[sample_idx].item()),
                }
            )

        summary_rows.append(
            {
                "layer": layer,
                "mean_projection": float(proj.mean().item()),
                "std_projection": float(proj.std(unbiased=True).item()),
                "mean_cosine": float(cos.mean().item()),
                "std_cosine": float(cos.std(unbiased=True).item()),
                "mean_delta_norm": float(norm.mean().item()),
                "std_delta_norm": float(norm.std(unbiased=True).item()),
                "mean_clean_gate_score": float(clean_score.mean().item()),
                "mean_corrupt_gate_score": float(corrupt_score.mean().item()),
                "n_samples": len(pairs),
            }
        )

    pre_hooks, mid_hooks, post_hooks = component_hook_names(gate_layer, hook_kind)
    hook_names = pre_hooks + mid_hooks + post_hooks
    if not hook_names:
        return per_sample_rows, summary_rows, []
    clean_comp = collect_pair_last_token_activations(model, pair_batches, hook_names=hook_names, side="clean", desc="component clean")
    corrupt_comp = collect_pair_last_token_activations(model, pair_batches, hook_names=hook_names, side="corrupt", desc="component corrupt")

    component_rows: list[dict[str, object]] = []
    cumulative_attn = 0.0
    cumulative_mlp = 0.0
    max_layer = gate_layer if hook_kind == "post" else gate_layer - 1
    for layer in range(max_layer + 1):
        pre_name = f"blocks.{layer}.hook_resid_pre"
        mid_name = f"blocks.{layer}.hook_resid_mid"
        post_name = f"blocks.{layer}.hook_resid_post"

        attn_clean = clean_comp[mid_name] - clean_comp[pre_name]
        attn_corrupt = corrupt_comp[mid_name] - corrupt_comp[pre_name]
        mlp_clean = clean_comp[post_name] - clean_comp[mid_name]
        mlp_corrupt = corrupt_comp[post_name] - corrupt_comp[mid_name]

        delta_attn = attn_clean - attn_corrupt
        delta_mlp = mlp_clean - mlp_corrupt
        gate_attn = projection(delta_attn, gate_direction_unit)
        gate_mlp = projection(delta_mlp, gate_direction_unit)
        gate_total = gate_attn + gate_mlp

        mean_attn = float(gate_attn.mean().item())
        mean_mlp = float(gate_mlp.mean().item())
        cumulative_attn += mean_attn
        cumulative_mlp += mean_mlp
        component_rows.append(
            {
                "layer": layer,
                "mean_attn_contrib": mean_attn,
                "std_attn_contrib": float(gate_attn.std(unbiased=True).item()),
                "mean_mlp_contrib": mean_mlp,
                "std_mlp_contrib": float(gate_mlp.std(unbiased=True).item()),
                "mean_total_contrib": float(gate_total.mean().item()),
                "std_total_contrib": float(gate_total.std(unbiased=True).item()),
                "cumulative_attn": cumulative_attn,
                "cumulative_mlp": cumulative_mlp,
                "cumulative_total": cumulative_attn + cumulative_mlp,
                "n_samples": len(pairs),
            }
        )
    return per_sample_rows, summary_rows, component_rows


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    ensure_dir(args.output_root)

    bundle = load_gate_bundle(args.pc_bundle)
    hook_kind = infer_hook_kind(bundle)
    gate_layer = gate_layer_from_bundle(bundle)
    mu_delta = extract_mu_delta(bundle)
    pc1 = extract_pc1(bundle)
    gate_direction_unit = unit(mu_delta)

    model, _tokenizer, tool_token_id = load_model_and_tokenizer(model_path=args.model_path, device=args.device)
    pairs = load_sample_pairs(model, dataset_root=args.dataset_root, split=args.eval_split, max_pairs=args.max_pairs)

    hook_name = gate_hook_name(gate_layer, hook_kind)
    fixed_rows = run_fixed_direction_sweep(
        model,
        pairs,
        hook_name=hook_name,
        mu_delta=mu_delta,
        tool_token_id=tool_token_id,
        alphas=args.alphas,
        batch_size=args.batch_size,
    )
    write_csv(args.output_root / "fixed_direction_sweep.csv", fixed_rows)

    traj_per_sample, traj_summary, component_rows = build_trajectory_outputs(
        model,
        pairs,
        hook_kind=hook_kind,
        gate_layer=gate_layer,
        gate_direction_unit=gate_direction_unit,
        batch_size=args.batch_size,
    )
    write_csv(args.output_root / "trajectory_per_sample.csv", traj_per_sample)
    write_csv(args.output_root / "trajectory_metrics.csv", traj_summary)
    plot_trajectory(traj_summary, args.output_root / "plot_gate_trajectory.pdf", gate_layer=gate_layer)

    if component_rows:
        write_csv(args.output_root / "component_contributions.csv", component_rows)
        plot_components(component_rows, args.output_root / "plot_gate_component_contributions.pdf")

    cos_mu_pc1 = None
    if pc1 is not None:
        cos_mu_pc1 = float(F.cosine_similarity(unit(mu_delta).view(1, -1), unit(pc1).view(1, -1), dim=-1).item())

    projections = [float(row["mean_projection"]) for row in traj_summary]
    threshold_layer = find_first_threshold_layer(projections, threshold_ratio=0.1)
    jump = largest_jump(projections)

    attn_series = [float(row["mean_attn_contrib"]) for row in component_rows]
    mlp_series = [float(row["mean_mlp_contrib"]) for row in component_rows]
    attn_mass = sum(abs(x) for x in attn_series)
    mlp_mass = sum(abs(x) for x in mlp_series)
    best_mu = max(fixed_rows, key=lambda row: float(row["tool_call_top1_rate"]))
    lines = [
        f"# {args.size_label} Mu-Delta and Gate-Formation Summary",
        "",
        f"- Gate layer: `L{gate_layer}` at `hook_resid_{hook_kind}`.",
        f"- Best `mu_delta` fixed-direction condition: `alpha={float(best_mu['alpha']):g}` -> top1 `{float(best_mu['tool_call_top1_rate']):.2%}`, strict flip `{float(best_mu['strict_flip_rate']):.2%}`.",
        f"- Final mean projection onto `hat(mu_delta)`: `{projections[-1]:.4f}`.",
        f"- First layer above 10% of final projection: `L{threshold_layer}`." if threshold_layer is not None else "- First layer above 10% of final projection: `n/a`.",
        (
            f"- Largest consecutive jump: `L{jump[0]} -> L{jump[0] + 1}` with `+{jump[1]:.4f}`."
            if jump is not None
            else "- Largest consecutive jump: `n/a`."
        ),
        f"- Total absolute contribution mass: attention `{attn_mass:.4f}`, MLP `{mlp_mass:.4f}`.",
        "- Note: centered PCA components need not align numerically with `mu_delta`; the fixed-direction intervention is the relevant diagnostic for whether the gate is carried by `mu_delta` itself.",
    ]
    if mlp_mass > attn_mass:
        lines.append("- Interpretation: gate formation is MLP-dominant in aggregate.")
    else:
        lines.append("- Interpretation: gate formation is not MLP-dominant in aggregate.")
    write_text(args.output_root / "mu_delta_summary.md", "\n".join(lines))

    write_json(
        args.output_root / "mu_delta_metadata.json",
        {
            "size_label": args.size_label,
            "model_path": str(args.model_path),
            "dataset_root": str(args.dataset_root),
            "eval_split": args.eval_split,
            "max_pairs": len(pairs),
            "pc_bundle": str(args.pc_bundle),
            "hook_kind": hook_kind,
            "gate_layer": gate_layer,
            "cos_mu_delta_pc1": cos_mu_pc1,
        },
    )

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
