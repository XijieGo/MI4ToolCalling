#!/usr/bin/env python3
"""
Stagewise schema-path rescue on factorized prompt variants.

This isolates the two-step schema branch found in the semantic chain analysis
and asks how much each step rescues tool-calling behavior when:
- schema is removed
- schema is mismatched
- protocol cue is removed
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

from toolcall_circuit.bidirectional_causal_eval import collect_cache_cpu_for_nodes, load_sample_paths
from toolcall_circuit.bidirectional_token_flip import run_logits_on_base_with_source
from toolcall_circuit.objective import build_bidirectional_endpoint_objectives
from toolcall_circuit.semantic_factorized_counterfactual import build_variants, safe_float, median, safe_rate
from toolcall_circuit.single_sample import load_hooked_qwen3, objective_from_logits


def write_csv(rows: Sequence[Dict[str, object]], out_path: Path) -> None:
    if not rows:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def load_schema_path(run_root: Path) -> List[str]:
    chain = json.loads((run_root / "semantic_chain" / "semantic_chain_summary.json").read_text(encoding="utf-8"))
    for path in chain.get("paths", []):
        if str(path["key"]) == "schema_tool_path":
            return [str(x) for x in path["nodes"] if str(x) != "Residual Output: decision"]
    raise ValueError("schema_tool_path not found in semantic chain summary.")


def plot(rows: Sequence[Dict[str, object]], out_path: Path) -> None:
    by_base: Dict[str, List[Dict[str, object]]] = {}
    for row in rows:
        by_base.setdefault(str(row["base_variant"]), []).append(row)
    plt.style.use("default")
    fig, ax = plt.subplots(figsize=(8.8, 4.8), constrained_layout=True)
    for base_variant, base_rows in by_base.items():
        base_rows = sorted(base_rows, key=lambda r: int(r["step_idx"]))
        xs = np.arange(1, len(base_rows) + 1)
        ys = [float(r["rescue_ratio_median"]) for r in base_rows]
        ax.plot(xs, ys, marker="o", label=base_variant)
    ax.set_xlabel("Schema path step")
    ax.set_ylabel("Median rescue ratio")
    ax.set_title("Stagewise Schema Rescue on Factorized Variants")
    ax.axhline(0.0, color="#888888", linewidth=1.0)
    ax.legend(frameon=False)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Stagewise schema rescue on factorized variants.")
    parser.add_argument("--run-root", type=str, required=True)
    parser.add_argument("--model-path", type=str, default="./external/models/Qwen3-1.7B")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--output-root", type=str, required=True)
    args = parser.parse_args()

    run_root = Path(args.run_root).resolve()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    schema_nodes = load_schema_path(run_root)
    samples = load_sample_paths(run_root / "forward_batch", run_root / "reverse_batch", max_samples=args.max_samples)
    model, tokenizer = load_hooked_qwen3(args.model_path, device=args.device, dtype=torch.bfloat16)

    per_sample_rows: List[Dict[str, object]] = []
    pbar = tqdm(samples, desc="Schema stagewise", dynamic_ncols=True)
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
        tool_gap = float(objective_from_logits(clean_logits, tool_objective).item()) - float(
            objective_from_logits(corrupt_logits, tool_objective).item()
        )
        if not math.isfinite(tool_gap) or abs(tool_gap) < 1e-8:
            continue

        clean_cache = collect_cache_cpu_for_nodes(model, clean_tokens, schema_nodes)
        for base_variant in ["clean_no_schema", "clean_schema_mismatch", "clean_no_protocol"]:
            base_tokens = model.to_tokens(variants[base_variant], prepend_bos=False)
            with torch.no_grad():
                base_logits = model(base_tokens)
            base_score = float(objective_from_logits(base_logits, tool_objective).item())
            cumulative_nodes: List[str] = []
            for step_idx, node in enumerate(schema_nodes, start=1):
                cumulative_nodes.append(node)
                logits = run_logits_on_base_with_source(model, base_tokens, clean_cache, cumulative_nodes)
                patched_tool = float(objective_from_logits(logits, tool_objective).item())
                patched_no_tool = float(objective_from_logits(logits, no_tool_objective).item())
                per_sample_rows.append(
                    {
                        "sample_id": sp.sample_id,
                        "base_variant": base_variant,
                        "step_idx": step_idx,
                        "node": node,
                        "rescue_ratio": (patched_tool - base_score) / tool_gap,
                        "top1_is_tool": int(logits[0, -1].argmax().item()) == sp.target_tool_call,
                        "boundary_flip": patched_tool > patched_no_tool,
                    }
                )
        pbar.set_postfix(sample=sp.sample_id)

    write_csv(per_sample_rows, output_root / "schema_stagewise_per_sample.csv")

    summary_rows: List[Dict[str, object]] = []
    for base_variant in sorted({str(r["base_variant"]) for r in per_sample_rows}):
        for step_idx in sorted({int(r["step_idx"]) for r in per_sample_rows if str(r["base_variant"]) == base_variant}):
            rows = [
                r
                for r in per_sample_rows
                if str(r["base_variant"]) == base_variant and int(r["step_idx"]) == step_idx
            ]
            summary_rows.append(
                {
                    "base_variant": base_variant,
                    "step_idx": step_idx,
                    "node": str(rows[0]["node"]),
                    "n_samples": len(rows),
                    "rescue_ratio_median": median(safe_float(r["rescue_ratio"]) for r in rows),
                    "top1_rate": safe_rate(bool(r["top1_is_tool"]) for r in rows),
                    "boundary_flip_rate": safe_rate(bool(r["boundary_flip"]) for r in rows),
                }
            )
    write_csv(summary_rows, output_root / "schema_stagewise_summary.csv")
    plot(summary_rows, output_root / "schema_stagewise_plot.png")

    summary = {
        "schema_nodes": schema_nodes,
        "summary_rows": summary_rows,
        "artifacts": {
            "summary_csv": str(output_root / "schema_stagewise_summary.csv"),
            "per_sample_csv": str(output_root / "schema_stagewise_per_sample.csv"),
            "plot": str(output_root / "schema_stagewise_plot.png"),
        },
    }
    (output_root / "schema_stagewise_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = ["# Schema Stagewise Report", ""]
    for row in summary_rows:
        lines.append(
            f"- `{row['base_variant']}` step {row['step_idx']} / `{row['node']}`: "
            f"rescue `{row['rescue_ratio_median']:.3f}`, top1 `{row['top1_rate']:.3f}`, "
            f"boundary `{row['boundary_flip_rate']:.3f}`"
        )
    lines.append("")
    markdown = "\n".join(lines)
    (output_root / "schema_stagewise_report.md").write_text(markdown, encoding="utf-8")
    print(markdown)


if __name__ == "__main__":
    main()
