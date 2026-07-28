#!/usr/bin/env python3
"""Sweep clean-to-corrupt residual-state patches for the paper's Figure 2.

This is the canonical implementation of the localization experiment in
Section 4.1.  For every selected layer it replaces either (a) the token whose
clean/corrupt IDs differ (the instruction verb) or (b) the final prompt token
state in a corrupt forward pass with the matched clean state.  It deliberately
uses the frozen ``datasets/{train,test}`` layout and writes a fresh table and
figure; it never reads archived result tables.
"""
from __future__ import annotations

import argparse
import gc
from collections import Counter
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import torch
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


VALID_POSITIONS = ("verb", "prediction")


def parse_layers(raw: str, *, n_layers: int) -> list[int]:
    text = str(raw).strip().lower()
    if text in {"all", "*"}:
        return list(range(n_layers))

    layers: set[int] = set()
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            start_text, end_text = item.split("-", 1)
            start, end = int(start_text), int(end_text)
            if end < start:
                raise ValueError(f"Invalid decreasing layer range: {item!r}")
            layers.update(range(start, end + 1))
        else:
            layers.add(int(item))
    selected = sorted(layers)
    if not selected:
        raise ValueError("No layers selected.")
    invalid = [layer for layer in selected if layer < 0 or layer >= n_layers]
    if invalid:
        raise ValueError(f"Layer(s) outside [0, {n_layers - 1}]: {invalid}")
    return selected


def parse_positions(raw: str) -> list[str]:
    positions = [item.strip().lower() for item in str(raw).split(",") if item.strip()]
    if not positions:
        raise ValueError("At least one patch position is required.")
    invalid = [item for item in positions if item not in VALID_POSITIONS]
    if invalid:
        raise ValueError(f"Unknown position(s): {invalid}; choose from {VALID_POSITIONS}")
    return list(dict.fromkeys(positions))


def first_changed_token_position(pair) -> int:
    """Return the only changed token position for an aligned contrastive pair.

    The frozen v2 manifests were constructed to differ in exactly one first
    user-instruction token.  Failing loudly here is preferable to silently
    treating a multi-token or misaligned edit as the paper's verb intervention.
    """

    clean = pair.clean_tokens_cpu.squeeze(0)
    corrupt = pair.corrupt_tokens_cpu.squeeze(0)
    if clean.shape != corrupt.shape:
        raise ValueError(
            f"{pair.sample_id}: clean/corrupt token shapes differ: "
            f"{tuple(clean.shape)} vs {tuple(corrupt.shape)}"
        )
    changed = torch.nonzero(clean != corrupt, as_tuple=False).flatten().tolist()
    if len(changed) != 1:
        raise ValueError(
            f"{pair.sample_id}: expected exactly one changed instruction token, found {len(changed)} at {changed}"
        )
    return int(changed[0])


def make_clean_source_capture(
    capture: dict[int, dict[str, torch.Tensor]],
    *,
    layer: int,
    verb_positions: Sequence[int],
    positions: Sequence[str],
):
    """Capture only the source vectors needed for the two patch locations."""

    def hook_fn(value: torch.Tensor, hook):  # noqa: ANN001
        sources: dict[str, torch.Tensor] = {}
        batch_indices = torch.arange(value.shape[0], device=value.device)
        if "verb" in positions:
            pos = torch.tensor(verb_positions, device=value.device, dtype=torch.long)
            sources["verb"] = value[batch_indices, pos, :].detach().cpu()
        if "prediction" in positions:
            sources["prediction"] = value[:, -1, :].detach().cpu()
        capture[layer] = sources
        return value

    return hook_fn


def make_batched_replace_hook(source_cpu: torch.Tensor, positions: Sequence[int]):
    """Patch one possibly different sequence position for each batch item."""

    def hook_fn(value: torch.Tensor, hook):  # noqa: ANN001
        if source_cpu.ndim != 2:
            raise ValueError(f"Expected [batch, d_model] source, got {tuple(source_cpu.shape)}")
        if source_cpu.shape[0] != value.shape[0]:
            raise ValueError(
                f"Source batch size {source_cpu.shape[0]} does not match activation batch size {value.shape[0]}"
            )
        out = value.clone()
        source = source_cpu.to(device=value.device, dtype=value.dtype)
        batch_indices = torch.arange(out.shape[0], device=out.device)
        pos = torch.tensor(positions, device=out.device, dtype=torch.long)
        out[batch_indices, pos, :] = source
        return out

    return hook_fn


def add_baseline_rows(
    rows: list[dict[str, object]],
    *,
    pairs,
    batch_indices: Sequence[int],
    condition: str,
    logits: torch.Tensor,
    probs: torch.Tensor,
    top1: torch.Tensor,
    tool_token_id: int,
    write_per_sample: bool,
) -> None:
    if not write_per_sample:
        return
    for local_idx, pair_idx in enumerate(batch_indices):
        pair = pairs[pair_idx]
        rows.append(
            {
                "sample_id": pair.sample_id,
                "condition": condition,
                "patch_position": "baseline",
                "layer": "",
                "tool_call_logit": float(logits[local_idx].item()),
                "tool_call_prob": float(probs[local_idx].item()),
                "is_tool_call_top1": int(top1[local_idx].item() == tool_token_id),
                "strict_flip_from_corrupt": 0,
            }
        )


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
        }
    )


def plot_sweep(rows: Sequence[dict[str, object]], output_root: Path) -> None:
    configure_matplotlib()
    fig, ax = plt.subplots(figsize=(7.4, 4.5))
    style = {
        "verb": {"color": "tab:blue", "marker": "o", "label": "instruction-verb position"},
        "prediction": {"color": "tab:orange", "marker": "s", "label": "prediction position"},
    }
    for position in VALID_POSITIONS:
        subset = sorted(
            (row for row in rows if str(row["patch_position"]) == position),
            key=lambda row: int(row["layer"]),
        )
        if not subset:
            continue
        ax.plot(
            [int(row["layer"]) for row in subset],
            [float(row["recovery_top1_rate"]) for row in subset],
            linewidth=2.0,
            markersize=4.5,
            **style[position],
        )
    ax.set_xlabel("Layer (hook_resid_pre)")
    ax.set_ylabel("Patched <tool_call> top-1 recovery")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    for suffix in ("pdf", "png"):
        fig.savefig(output_root / f"activation_patch_sweep.{suffix}", bbox_inches="tight")
    plt.close(fig)


def earliest_near_peak(rows: Iterable[dict[str, object]], *, tolerance: float = 0.01) -> dict[str, object] | None:
    candidates = list(rows)
    if not candidates:
        return None
    peak = max(float(row["recovery_top1_rate"]) for row in candidates)
    close = [row for row in candidates if float(row["recovery_top1_rate"]) >= peak - tolerance]
    return min(close, key=lambda row: int(row["layer"]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Canonical clean-to-corrupt residual patch sweep for the paper's activation-patching figure."
    )
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--size-label", type=str, default="Qwen3-8B")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--split", choices=("train", "test"), default="test")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--layers", type=str, default="all", help="Comma/range list such as 0-35, or 'all'.")
    parser.add_argument("--positions", type=str, default="verb,prediction")
    parser.add_argument("--hook-kind", choices=("pre",), default="pre")
    parser.add_argument("--max-pairs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-per-sample", action="store_true", help="Do not write the audit-level per-sample CSV.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    ensure_dir(args.output_root)

    model, _tokenizer, tool_token_id = load_model_and_tokenizer(model_path=args.model_path, device=args.device)
    layers = parse_layers(args.layers, n_layers=int(model.cfg.n_layers))
    positions = parse_positions(args.positions)
    pairs = load_sample_pairs(
        model,
        dataset_root=args.dataset_root,
        split=args.split,
        max_pairs=args.max_pairs,
    )
    verb_positions = [first_changed_token_position(pair) for pair in pairs]
    batches = build_pair_batches(pairs, batch_size=args.batch_size)

    stats: dict[tuple[int, str], dict[str, float]] = {
        (layer, position): {
            "count": 0.0,
            "tool_top1": 0.0,
            "strict_flip": 0.0,
            "tool_logit_sum": 0.0,
            "tool_prob_sum": 0.0,
        }
        for layer in layers
        for position in positions
    }
    baseline = {
        "count": 0.0,
        "clean_tool_top1": 0.0,
        "corrupt_tool_top1": 0.0,
        "corrupt_non_tool": 0.0,
        "clean_logit_sum": 0.0,
        "corrupt_logit_sum": 0.0,
        "clean_prob_sum": 0.0,
        "corrupt_prob_sum": 0.0,
    }
    per_sample_rows: list[dict[str, object]] = []

    hook_names = {layer: f"blocks.{layer}.hook_resid_{args.hook_kind}" for layer in layers}
    progress = tqdm(batches, desc=f"{args.size_label} activation-patch sweep", dynamic_ncols=True)
    for batch in progress:
        batch_verb_positions = [verb_positions[idx] for idx in batch.indices]
        batch_prediction_positions = [int(batch.token_len) - 1] * len(batch.indices)
        clean_tokens = batch.clean_tokens_cpu.to(model.W_U.device)
        corrupt_tokens = batch.corrupt_tokens_cpu.to(model.W_U.device)
        source_capture: dict[int, dict[str, torch.Tensor]] = {}
        clean_hooks = [
            (
                hook_names[layer],
                make_clean_source_capture(
                    source_capture,
                    layer=layer,
                    verb_positions=batch_verb_positions,
                    positions=positions,
                ),
            )
            for layer in layers
        ]
        with torch.no_grad():
            clean_logits = model.run_with_hooks(clean_tokens, fwd_hooks=clean_hooks)
            corrupt_logits = model(corrupt_tokens)
        clean_logit, clean_prob, clean_top1 = tool_stats(clean_logits, tool_token_id)
        corrupt_logit, corrupt_prob, corrupt_top1 = tool_stats(corrupt_logits, tool_token_id)

        batch_count = float(len(batch.indices))
        baseline["count"] += batch_count
        baseline["clean_tool_top1"] += float((clean_top1 == tool_token_id).sum().item())
        baseline["corrupt_tool_top1"] += float((corrupt_top1 == tool_token_id).sum().item())
        baseline["corrupt_non_tool"] += float((corrupt_top1 != tool_token_id).sum().item())
        baseline["clean_logit_sum"] += float(clean_logit.sum().item())
        baseline["corrupt_logit_sum"] += float(corrupt_logit.sum().item())
        baseline["clean_prob_sum"] += float(clean_prob.sum().item())
        baseline["corrupt_prob_sum"] += float(corrupt_prob.sum().item())
        add_baseline_rows(
            per_sample_rows,
            pairs=pairs,
            batch_indices=batch.indices,
            condition="clean",
            logits=clean_logit,
            probs=clean_prob,
            top1=clean_top1,
            tool_token_id=tool_token_id,
            write_per_sample=not args.no_per_sample,
        )
        add_baseline_rows(
            per_sample_rows,
            pairs=pairs,
            batch_indices=batch.indices,
            condition="corrupt",
            logits=corrupt_logit,
            probs=corrupt_prob,
            top1=corrupt_top1,
            tool_token_id=tool_token_id,
            write_per_sample=not args.no_per_sample,
        )

        for layer in layers:
            for position in positions:
                patch_positions = batch_verb_positions if position == "verb" else batch_prediction_positions
                with torch.no_grad():
                    patched_logits = model.run_with_hooks(
                        corrupt_tokens,
                        fwd_hooks=[
                            (
                                hook_names[layer],
                                make_batched_replace_hook(source_capture[layer][position], patch_positions),
                            )
                        ],
                    )
                patched_logit, patched_prob, patched_top1 = tool_stats(patched_logits, tool_token_id)
                strict_flip = (corrupt_top1 != tool_token_id) & (patched_top1 == tool_token_id)
                bucket = stats[(layer, position)]
                bucket["count"] += batch_count
                bucket["tool_top1"] += float((patched_top1 == tool_token_id).sum().item())
                bucket["strict_flip"] += float(strict_flip.sum().item())
                bucket["tool_logit_sum"] += float(patched_logit.sum().item())
                bucket["tool_prob_sum"] += float(patched_prob.sum().item())

                if not args.no_per_sample:
                    for local_idx, pair_idx in enumerate(batch.indices):
                        pair = pairs[pair_idx]
                        per_sample_rows.append(
                            {
                                "sample_id": pair.sample_id,
                                "condition": "corrupt_with_clean_patch",
                                "patch_position": position,
                                "layer": int(layer),
                                "patched_token_position": int(patch_positions[local_idx]),
                                "tool_call_logit": float(patched_logit[local_idx].item()),
                                "tool_call_prob": float(patched_prob[local_idx].item()),
                                "is_tool_call_top1": int(patched_top1[local_idx].item() == tool_token_id),
                                "strict_flip_from_corrupt": int(strict_flip[local_idx].item()),
                            }
                        )
                del patched_logits, patched_logit, patched_prob, patched_top1, strict_flip
                clear_cuda()

        del clean_tokens, corrupt_tokens, clean_logits, corrupt_logits, source_capture
        clear_cuda()
        progress.set_postfix(token_len=batch.token_len)

    n = max(int(baseline["count"]), 1)
    corrupt_non_tool = max(int(baseline["corrupt_non_tool"]), 1)
    summary_rows: list[dict[str, object]] = []
    for layer in layers:
        for position in positions:
            bucket = stats[(layer, position)]
            count = max(int(bucket["count"]), 1)
            summary_rows.append(
                {
                    "layer": int(layer),
                    "patch_position": position,
                    "hook_name": hook_names[layer],
                    "n_pairs": count,
                    "recovery_top1_rate": float(bucket["tool_top1"] / count),
                    "strict_flip_rate": float(bucket["strict_flip"] / corrupt_non_tool),
                    "mean_tool_call_logit": float(bucket["tool_logit_sum"] / count),
                    "mean_tool_call_prob": float(bucket["tool_prob_sum"] / count),
                    "baseline_clean_top1_rate": float(baseline["clean_tool_top1"] / n),
                    "baseline_corrupt_top1_rate": float(baseline["corrupt_tool_top1"] / n),
                    "baseline_clean_mean_tool_logit": float(baseline["clean_logit_sum"] / n),
                    "baseline_corrupt_mean_tool_logit": float(baseline["corrupt_logit_sum"] / n),
                }
            )

    write_csv(args.output_root / "activation_patch_sweep.csv", summary_rows)
    if not args.no_per_sample:
        write_csv(args.output_root / "activation_patch_sweep_per_sample.csv", per_sample_rows)
    plot_sweep(summary_rows, args.output_root)

    selected: dict[str, dict[str, object]] = {}
    for position in positions:
        candidate = earliest_near_peak(
            (row for row in summary_rows if str(row["patch_position"]) == position)
        )
        if candidate is not None:
            selected[position] = candidate
    mismatch_counts = Counter(verb_positions)
    metadata = {
        "purpose": "Section 4.1 / Figure 2 clean-to-corrupt residual-state localization sweep",
        "size_label": args.size_label,
        "model_path": str(args.model_path),
        "dataset_root": str(args.dataset_root),
        "split": args.split,
        "n_pairs": n,
        "layers": layers,
        "positions": positions,
        "hook_kind": args.hook_kind,
        "hook_semantics": "TransformerLens blocks.L.hook_resid_pre; clean state is inserted into the corrupt forward pass.",
        "verb_token_position_counts": {str(key): int(value) for key, value in sorted(mismatch_counts.items())},
        "selection_rule": "earliest layer within 1 percentage point of that position's peak top-1 recovery",
        "selected_near_peak": selected,
        "tool_token_id": int(tool_token_id),
        "seed": int(args.seed),
    }
    write_json(args.output_root / "metadata.json", metadata)

    lines = [
        "# Activation-Patching Localization Sweep",
        "",
        f"- Model: `{args.size_label}` (`{args.model_path}`)",
        f"- Data: frozen v2 `{args.split}` split, `{n}` paired prompts",
        f"- Patch: matched clean `hook_resid_pre` state injected into the corrupt pass",
        f"- Instruction edit invariant: exactly one token differs in every retained pair; position distribution `{dict(sorted(mismatch_counts.items()))}`",
        f"- Baseline clean `<tool_call>` top-1: `{baseline['clean_tool_top1'] / n:.2%}`",
        f"- Baseline corrupt `<tool_call>` top-1: `{baseline['corrupt_tool_top1'] / n:.2%}`",
        "",
        "| position | selected near-peak layer | recovery | strict flip |",
        "|---|---:|---:|---:|",
    ]
    for position in positions:
        row = selected.get(position)
        if row is not None:
            lines.append(
                f"| {position} | L{int(row['layer'])} | {float(row['recovery_top1_rate']):.2%} | {float(row['strict_flip_rate']):.2%} |"
            )
    lines.extend(
        [
            "",
            "`activation_patch_sweep.csv` is the numerical source for the two curves; the PDF/PNG are generated views and are not copied into the manuscript automatically.",
        ]
    )
    write_text(args.output_root / "summary.md", "\n".join(lines))

    print(
        f"Wrote {len(summary_rows)} layer/position rows for {n} pairs to "
        f"{args.output_root / 'activation_patch_sweep.csv'}"
    )
    del model
    gc.collect()
    clear_cuda()


if __name__ == "__main__":
    main()
