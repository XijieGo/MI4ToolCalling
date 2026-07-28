#!/usr/bin/env python3
"""
Late-writer steering audit for MLP27.

We keep the prompt and all upstream computation fixed, and only modify the
last-token MLP27 activation by interpolating toward the clean tool-call source.
"""

from __future__ import annotations

import argparse
import csv
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

from toolcall_circuit.bidirectional_causal_eval import load_sample_paths
from toolcall_circuit.objective import build_bidirectional_endpoint_objectives
from toolcall_circuit.semantic_factorized_counterfactual import build_variants, safe_float
from toolcall_circuit.single_sample import load_hooked_qwen3, objective_from_logits


def write_csv(rows: Sequence[Dict[str, object]], out_path: Path) -> None:
    if not rows:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def median(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if isinstance(v, (int, float)) and math.isfinite(float(v))]
    return float(np.median(vals)) if vals else float("nan")


def safe_rate(values: Iterable[bool]) -> float:
    vals = [1.0 if bool(v) else 0.0 for v in values]
    return float(np.mean(vals)) if vals else float("nan")


def cache_mlp27(model, tokens: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens, names_filter=lambda n: n == "blocks.27.hook_mlp_out")
    return cache["blocks.27.hook_mlp_out"].detach().cpu()


def run_with_mlp27_interpolation(model, base_tokens: torch.Tensor, base_cache: torch.Tensor, source_cache: torch.Tensor, alpha: float) -> torch.Tensor:
    base_cache = base_cache.to(base_tokens.device)
    source_cache = source_cache.to(base_tokens.device)
    base_last = base_cache[:, -1, :]
    source_last = source_cache[:, -1, :]
    target = base_last + alpha * (source_last - base_last)

    def hook_fn(mlp_out: torch.Tensor, hook):  # noqa: ANN001
        out = mlp_out.clone()
        out[:, -1, :] = target
        return out

    with torch.no_grad():
        return model.run_with_hooks(base_tokens, fwd_hooks=[("blocks.27.hook_mlp_out", hook_fn)])


def plot_curves(summary_rows: Sequence[Dict[str, object]], out_path: Path) -> None:
    by_base: Dict[str, List[Dict[str, object]]] = {}
    for row in summary_rows:
        by_base.setdefault(str(row["base_variant"]), []).append(row)
    plt.style.use("default")
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8), constrained_layout=True)
    for base_variant, rows in by_base.items():
        rows = sorted(rows, key=lambda r: float(r["alpha"]))
        xs = [float(r["alpha"]) for r in rows]
        axes[0].plot(xs, [float(r["decision_score_median"]) for r in rows], marker="o", label=base_variant)
        axes[1].plot(xs, [float(r["tool_top1_rate"]) for r in rows], marker="o", label=base_variant)
    axes[0].axhline(0.0, color="#888888", linewidth=1.0)
    axes[0].set_title("MLP27 Steering: Decision Score")
    axes[0].set_xlabel("alpha")
    axes[0].set_ylabel("median (S_tool - S_no_tool)")
    axes[1].set_title("MLP27 Steering: Tool Top-1 Rate")
    axes[1].set_xlabel("alpha")
    axes[1].set_ylabel("tool top-1 rate")
    for ax in axes:
        ax.legend(frameon=False)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="MLP27 steering audit.")
    parser.add_argument("--run-root", type=str, required=True)
    parser.add_argument("--model-path", type=str, default="./external/models/Qwen3-1.7B")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--alphas", type=str, default="0.0,0.25,0.5,0.75,1.0,1.25,1.5")
    parser.add_argument("--output-root", type=str, required=True)
    args = parser.parse_args()

    alphas = [float(x.strip()) for x in args.alphas.split(",") if x.strip()]
    run_root = Path(args.run_root).resolve()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    samples = load_sample_paths(run_root / "forward_batch", run_root / "reverse_batch", max_samples=args.max_samples)
    model, tokenizer = load_hooked_qwen3(args.model_path, device=args.device, dtype=torch.bfloat16)
    _ = tokenizer

    per_sample_rows: List[Dict[str, object]] = []
    pbar = tqdm(samples, desc="MLP27 steering", dynamic_ncols=True)
    for sp in pbar:
        try:
            clean_text = sp.tool_call_prompt.read_text(encoding="utf-8")
            corrupt_text = sp.no_tool_prompt.read_text(encoding="utf-8")
        except Exception:
            continue
        variants = build_variants(clean_text, corrupt_text)

        clean_tokens = model.to_tokens(variants["clean_full"], prepend_bos=False)
        corrupt_tokens = model.to_tokens(variants["corrupt_full"], prepend_bos=False)
        with torch.no_grad():
            clean_logits = model(clean_tokens)
            corrupt_logits = model(corrupt_tokens)
        tool_objective, no_tool_objective = build_bidirectional_endpoint_objectives(clean_logits, corrupt_logits, tokenizer=tokenizer)

        source_cache = cache_mlp27(model, clean_tokens)
        for base_variant in ["corrupt_full", "clean_no_schema", "clean_schema_mismatch", "clean_no_protocol"]:
            base_tokens = model.to_tokens(variants[base_variant], prepend_bos=False)
            base_cache = cache_mlp27(model, base_tokens)
            for alpha in alphas:
                logits = run_with_mlp27_interpolation(model, base_tokens, base_cache, source_cache, alpha)
                tool_score = float(objective_from_logits(logits, tool_objective).item())
                no_tool_score = float(objective_from_logits(logits, no_tool_objective).item())
                per_sample_rows.append(
                    {
                        "sample_id": sp.sample_id,
                        "base_variant": base_variant,
                        "alpha": alpha,
                        "tool_score": tool_score,
                        "no_tool_score": no_tool_score,
                        "decision_score": tool_score - no_tool_score,
                        "tool_top1": int(logits[0, -1].argmax().item()) == sp.target_tool_call,
                        "boundary_flip": tool_score > no_tool_score,
                    }
                )
        pbar.set_postfix(sample=sp.sample_id)

    write_csv(per_sample_rows, output_root / "mlp27_steering_per_sample.csv")

    summary_rows: List[Dict[str, object]] = []
    for base_variant in sorted({str(r["base_variant"]) for r in per_sample_rows}):
        for alpha in sorted({float(r["alpha"]) for r in per_sample_rows if str(r["base_variant"]) == base_variant}):
            rows = [
                r
                for r in per_sample_rows
                if str(r["base_variant"]) == base_variant and float(r["alpha"]) == alpha
            ]
            summary_rows.append(
                {
                    "base_variant": base_variant,
                    "alpha": alpha,
                    "n_samples": len(rows),
                    "tool_score_median": median(safe_float(r["tool_score"]) for r in rows),
                    "no_tool_score_median": median(safe_float(r["no_tool_score"]) for r in rows),
                    "decision_score_median": median(safe_float(r["decision_score"]) for r in rows),
                    "tool_top1_rate": safe_rate(bool(r["tool_top1"]) for r in rows),
                    "boundary_flip_rate": safe_rate(bool(r["boundary_flip"]) for r in rows),
                }
            )

    write_csv(summary_rows, output_root / "mlp27_steering_summary.csv")
    plot_curves(summary_rows, output_root / "mlp27_steering_curves.png")

    summary = {
        "summary_rows": summary_rows,
        "artifacts": {
            "summary_csv": str(output_root / "mlp27_steering_summary.csv"),
            "per_sample_csv": str(output_root / "mlp27_steering_per_sample.csv"),
            "plot": str(output_root / "mlp27_steering_curves.png"),
        },
    }
    (output_root / "mlp27_steering_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = ["# MLP27 Steering Report", ""]
    for row in summary_rows:
        lines.append(
            f"- `{row['base_variant']}` / alpha `{row['alpha']}`: "
            f"decision `{row['decision_score_median']:.3f}`, "
            f"tool-top1 `{row['tool_top1_rate']:.3f}`, "
            f"boundary `{row['boundary_flip_rate']:.3f}`"
        )
    lines.append("")
    markdown = "\n".join(lines)
    (output_root / "mlp27_steering_report.md").write_text(markdown, encoding="utf-8")
    print(markdown)


if __name__ == "__main__":
    main()
