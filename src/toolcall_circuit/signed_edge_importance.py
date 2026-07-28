#!/usr/bin/env python3
"""
Edge-level mediation analysis for the final signed circuit.

For each edge (u -> v):
1) source-only: patch source node from the clean endpoint;
2) blocked: patch source from clean, but force target back to the base endpoint;
3) mediated = source-only - blocked.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm.auto import tqdm

if __package__ in {None, ""}:
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from toolcall_circuit.bidirectional_causal_eval import collect_cache_cpu_for_nodes, load_sample_paths
from toolcall_circuit.objective import build_bidirectional_endpoint_objectives
from toolcall_circuit.single_sample import load_hooked_qwen3, objective_from_logits, parse_head


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def finite(values: Iterable[float]) -> List[float]:
    return [float(v) for v in values if isinstance(v, (int, float)) and math.isfinite(float(v))]


def median(values: Iterable[float]) -> float:
    vals = finite(values)
    return float(np.median(vals)) if vals else float("nan")


def write_csv(rows: Sequence[Dict[str, object]], out_path: Path) -> None:
    if not rows:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_logits_with_assignments(
    model,
    base_tokens: torch.Tensor,
    clean_cache_cpu: Dict[str, torch.Tensor],
    corrupt_cache_cpu: Dict[str, torch.Tensor],
    patch_clean_nodes: Sequence[str],
    patch_corrupt_nodes: Sequence[str],
) -> torch.Tensor:
    heads_by_layer_clean: Dict[int, List[int]] = defaultdict(list)
    heads_by_layer_corrupt: Dict[int, List[int]] = defaultdict(list)
    mlp_layers_clean: List[int] = []
    mlp_layers_corrupt: List[int] = []

    for node in patch_clean_nodes:
        if node.startswith("MLP"):
            mlp_layers_clean.append(int(node[3:]))
        else:
            layer, head = parse_head(node)
            heads_by_layer_clean[layer].append(head)
    for node in patch_corrupt_nodes:
        if node.startswith("MLP"):
            mlp_layers_corrupt.append(int(node[3:]))
        else:
            layer, head = parse_head(node)
            heads_by_layer_corrupt[layer].append(head)

    hooks = []

    def add_head_hooks(layer_to_heads: Dict[int, List[int]], cache_cpu: Dict[str, torch.Tensor]) -> None:
        for layer, heads in layer_to_heads.items():
            cache_name = f"blocks.{layer}.attn.hook_z"
            src_act = cache_cpu[cache_name].to(base_tokens.device)
            uniq = sorted(set(heads))

            def make_head_hook(src: torch.Tensor, hs: Sequence[int]):
                def hook_fn(z: torch.Tensor, hook):  # noqa: ANN001
                    out = z.clone()
                    for h in hs:
                        out[:, -1, h, :] = src[:, -1, h, :]
                    return out

                return hook_fn

            hooks.append((cache_name, make_head_hook(src_act, uniq)))

    def add_mlp_hooks(layers: Sequence[int], cache_cpu: Dict[str, torch.Tensor]) -> None:
        for layer in sorted(set(layers)):
            cache_name = f"blocks.{layer}.hook_mlp_out"
            src_act = cache_cpu[cache_name].to(base_tokens.device)

            def make_mlp_hook(src: torch.Tensor):
                def hook_fn(mlp_out: torch.Tensor, hook):  # noqa: ANN001
                    out = mlp_out.clone()
                    out[:, -1, :] = src[:, -1, :]
                    return out

                return hook_fn

            hooks.append((cache_name, make_mlp_hook(src_act)))

    add_head_hooks(heads_by_layer_clean, clean_cache_cpu)
    add_head_hooks(heads_by_layer_corrupt, corrupt_cache_cpu)
    add_mlp_hooks(mlp_layers_clean, clean_cache_cpu)
    add_mlp_hooks(mlp_layers_corrupt, corrupt_cache_cpu)

    with torch.no_grad():
        return model.run_with_hooks(base_tokens, fwd_hooks=hooks)


def plot_edge_heatmap(rows: Sequence[Dict[str, object]], out_path: Path) -> None:
    labels = [str(r["edge"]) for r in rows]
    mat = np.array(
        [
            [
                float(r["promote_mediated_ratio_median"]),
                float(r["suppress_mediated_ratio_median"]),
            ]
            for r in rows
        ],
        dtype=float,
    )
    finite_vals = mat[np.isfinite(mat)]
    vmax = max(float(np.percentile(np.abs(finite_vals), 98)), 1e-6) if finite_vals.size else 1.0
    plt.style.use("default")
    fig, ax = plt.subplots(figsize=(8.8, max(5.5, 0.33 * len(labels) + 1.8)), constrained_layout=True)
    im = ax.imshow(mat, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(np.arange(2))
    ax.set_xticklabels(["promote mediated", "suppress mediated"], rotation=25, ha="right")
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_title("Signed Edge Mediation")
    cbar = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
    cbar.set_label("KL recovery drop")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Edge-level mediation for the signed circuit.")
    parser.add_argument("--forward-batch-root", type=str, required=True)
    parser.add_argument("--signed-edges-csv", type=str, required=True)
    parser.add_argument("--model-path", type=str, default="./external/models/Qwen3-1.7B")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max-samples", type=int, default=200)
    parser.add_argument("--max-edges", type=int, default=0)
    parser.add_argument("--output-root", type=str, required=True)
    args = parser.parse_args()

    out_root = Path(args.output_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    edge_rows = [
        row
        for row in read_csv_rows(Path(args.signed_edges_csv).resolve())
        if str(row["target"]) != "Residual Output: decision"
    ]
    edge_rows.sort(key=lambda r: float(r["union_support_max"]), reverse=True)
    if args.max_edges > 0:
        edge_rows = edge_rows[: args.max_edges]

    reverse_batch_root = Path(args.signed_edges_csv).resolve().parent.parent / "reverse_batch"
    if not reverse_batch_root.exists():
        reverse_batch_root = Path(args.forward_batch_root).resolve().parent / "reverse_batch"
    samples = load_sample_paths(Path(args.forward_batch_root).resolve(), reverse_batch_root, max_samples=args.max_samples)

    all_nodes = sorted({str(r["source"]) for r in edge_rows} | {str(r["target"]) for r in edge_rows})
    model, tokenizer = load_hooked_qwen3(args.model_path, device=args.device, dtype=torch.bfloat16)
    _ = tokenizer

    per_sample_rows: List[Dict[str, object]] = []
    pbar = tqdm(samples, desc="Signed edge mediation", dynamic_ncols=True)
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
        tool_base = float(objective_from_logits(no_tool_logits, tool_objective).item())
        tool_gap = float(objective_from_logits(tool_logits, tool_objective).item()) - tool_base
        no_tool_base = float(objective_from_logits(tool_logits, no_tool_objective).item())
        no_tool_gap = float(objective_from_logits(no_tool_logits, no_tool_objective).item()) - no_tool_base
        if (
            not math.isfinite(tool_gap)
            or abs(tool_gap) < 1e-8
            or not math.isfinite(no_tool_gap)
            or abs(no_tool_gap) < 1e-8
        ):
            continue

        tool_cache = collect_cache_cpu_for_nodes(model, tool_tokens, all_nodes)
        no_tool_cache = collect_cache_cpu_for_nodes(model, no_tool_tokens, all_nodes)

        for row in edge_rows:
            source = str(row["source"])
            target = str(row["target"])

            promote_source_logits = run_logits_with_assignments(
                model, no_tool_tokens, tool_cache, no_tool_cache, [source], []
            )
            promote_source_ratio = (
                float(objective_from_logits(promote_source_logits, tool_objective).item()) - tool_base
            ) / tool_gap

            promote_blocked_logits = run_logits_with_assignments(
                model, no_tool_tokens, tool_cache, no_tool_cache, [source], [target]
            )
            promote_blocked_ratio = (
                float(objective_from_logits(promote_blocked_logits, tool_objective).item()) - tool_base
            ) / tool_gap

            suppress_source_logits = run_logits_with_assignments(
                model, tool_tokens, no_tool_cache, tool_cache, [source], []
            )
            suppress_source_ratio = (
                float(objective_from_logits(suppress_source_logits, no_tool_objective).item()) - no_tool_base
            ) / no_tool_gap

            suppress_blocked_logits = run_logits_with_assignments(
                model, tool_tokens, no_tool_cache, tool_cache, [source], [target]
            )
            suppress_blocked_ratio = (
                float(objective_from_logits(suppress_blocked_logits, no_tool_objective).item()) - no_tool_base
            ) / no_tool_gap

            per_sample_rows.append(
                {
                    "sample_id": sp.sample_id,
                    "edge": f"{source}->{target}",
                    "source": source,
                    "target": target,
                    "sign": str(row["sign"]),
                    "promote_source_ratio": promote_source_ratio,
                    "promote_blocked_ratio": promote_blocked_ratio,
                    "promote_mediated_ratio": promote_source_ratio - promote_blocked_ratio,
                    "suppress_source_ratio": suppress_source_ratio,
                    "suppress_blocked_ratio": suppress_blocked_ratio,
                    "suppress_mediated_ratio": suppress_source_ratio - suppress_blocked_ratio,
                }
            )

        pbar.set_postfix(sample=sp.sample_id)

    by_edge: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in per_sample_rows:
        by_edge[str(row["edge"])].append(row)

    summary_rows: List[Dict[str, object]] = []
    for row in edge_rows:
        edge = f"{row['source']}->{row['target']}"
        rows = by_edge[edge]
        summary_rows.append(
            {
                "edge": edge,
                "source": str(row["source"]),
                "target": str(row["target"]),
                "sign": str(row["sign"]),
                "n_samples": len(rows),
                "promote_source_ratio_median": median(r["promote_source_ratio"] for r in rows),
                "promote_blocked_ratio_median": median(r["promote_blocked_ratio"] for r in rows),
                "promote_mediated_ratio_median": median(r["promote_mediated_ratio"] for r in rows),
                "suppress_source_ratio_median": median(r["suppress_source_ratio"] for r in rows),
                "suppress_blocked_ratio_median": median(r["suppress_blocked_ratio"] for r in rows),
                "suppress_mediated_ratio_median": median(r["suppress_mediated_ratio"] for r in rows),
            }
        )
    summary_rows.sort(
        key=lambda r: abs(float(r["promote_mediated_ratio_median"])) + abs(float(r["suppress_mediated_ratio_median"])),
        reverse=True,
    )

    write_csv(per_sample_rows, out_root / "signed_edge_mediation_per_sample.csv")
    write_csv(summary_rows, out_root / "signed_edge_mediation_summary.csv")
    plot_edge_heatmap(summary_rows, out_root / "signed_edge_mediation_heatmap.png")

    summary = {
        "n_samples": len(samples),
        "artifacts": {
            "per_sample_csv": str(out_root / "signed_edge_mediation_per_sample.csv"),
            "summary_csv": str(out_root / "signed_edge_mediation_summary.csv"),
            "summary_json": str(out_root / "signed_edge_mediation_report.json"),
            "heatmap_png": str(out_root / "signed_edge_mediation_heatmap.png"),
        },
        "summary_rows": summary_rows,
    }
    (out_root / "signed_edge_mediation_report.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
