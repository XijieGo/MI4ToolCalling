#!/usr/bin/env python3
"""
Analyze where direction-selective heads read from.

We compute attention mass at the *next-token prediction position* (query pos = -1)
to several interpretable position sets derived from the prompt:
- contrast positions (where tool-call vs no-tool prompts differ)
- <tool_call> / </tool_call> tag positions
- <tools> ... </tools> schema block
- last user block
- prefix_16 and recent_32 baselines

Outputs:
- per-head CSV with median masses and deltas (no-tool minus tool-call)
- per-group CSV (reverse-only, reverse-selective, forward-selective, shared-core)
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
from statistics import median
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm.auto import tqdm

if __package__ in {None, ""}:
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from toolcall_circuit.dataset import build_position_sets, load_summary_records
from toolcall_circuit.single_sample import load_hooked_qwen3


def finite(xs: Iterable[float]) -> List[float]:
    return [float(x) for x in xs if isinstance(x, (int, float)) and math.isfinite(float(x))]


def med(xs: Iterable[float]) -> float:
    vals = finite(xs)
    return float(median(vals)) if vals else float("nan")


def mean(xs: Iterable[float]) -> float:
    vals = finite(xs)
    return float(np.mean(vals)) if vals else float("nan")


def parse_head(node: str) -> Tuple[int, int]:
    body = node[1:]
    layer_s, head_s = body.split("H")
    return int(layer_s), int(head_s)


def head_nodes(nodes: Sequence[str]) -> List[str]:
    return sorted({str(n) for n in nodes if str(n).startswith("L")})


def load_groups(forward_agg: Path, reverse_agg: Path, bidirectional_summary: Path) -> Dict[str, List[str]]:
    fwd = json.loads(forward_agg.read_text(encoding="utf-8"))
    rev = json.loads(reverse_agg.read_text(encoding="utf-8"))
    bidi = json.loads(bidirectional_summary.read_text(encoding="utf-8"))

    f_core = set(head_nodes(fwd.get("core_nodes", [])))
    r_core = set(head_nodes(rev.get("core_nodes", [])))

    groups: Dict[str, List[str]] = {
        "forward_only_core_heads": sorted(f_core - r_core),
        "reverse_only_core_heads": sorted(r_core - f_core),
        "shared_core_heads": sorted(f_core & r_core),
        "forward_selective_heads": head_nodes(bidi.get("support_analysis", {}).get("forward_selective_nodes", [])),
        "reverse_selective_heads": head_nodes(bidi.get("support_analysis", {}).get("reverse_selective_nodes", [])),
    }
    return {k: v for k, v in groups.items() if v}


def position_sets_from_prompts(model, tokenizer, tool_text: str, no_tool_text: str) -> Dict[str, List[int]]:
    tool_tokens = model.to_tokens(tool_text, prepend_bos=False)
    no_tool_tokens = model.to_tokens(no_tool_text, prepend_bos=False)
    if tool_tokens.shape != no_tool_tokens.shape:
        raise ValueError(f"Token shapes differ: {tool_tokens.shape} vs {no_tool_tokens.shape}")
    ids_tool = [int(x) for x in tool_tokens[0].tolist()]
    ids_no = [int(x) for x in no_tool_tokens[0].tolist()]
    return build_position_sets(ids_tool, ids_no, tokenizer, clean_text=tool_text)


def needed_pattern_names(layers: Sequence[int]) -> List[str]:
    return [f"blocks.{int(l)}.attn.hook_pattern" for l in sorted(set(int(x) for x in layers))]


def mass_to_positions(pattern_vec: torch.Tensor, positions: Sequence[int]) -> float:
    if not positions:
        return float("nan")
    pos = [int(p) for p in positions if 0 <= int(p) < int(pattern_vec.shape[0])]
    if not pos:
        return float("nan")
    return float(pattern_vec[pos].sum().item())


def apply_plot_style() -> None:
    plt.style.use("default")
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 16,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        }
    )


def save_heatmap(matrix: np.ndarray, row_labels: Sequence[str], col_labels: Sequence[str], title: str, out_path: Path) -> None:
    apply_plot_style()
    finite_vals = matrix[np.isfinite(matrix)]
    vmax = float(np.percentile(np.abs(finite_vals), 98)) if finite_vals.size else 1.0
    vmax = max(vmax, 1e-6)
    fig, ax = plt.subplots(figsize=(1.35 * max(5, len(col_labels)), 0.55 * max(5, len(row_labels)) + 2))
    im = ax.imshow(matrix, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_title(title)
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=25, ha="right")
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels)
    cbar = fig.colorbar(im, ax=ax, fraction=0.06, pad=0.03)
    cbar.set_label("Delta attention mass (no-tool minus tool-call)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Bidirectional head read analysis.")
    parser.add_argument("--dataset-root", type=str, default="datasets")
    parser.add_argument("--reverse-batch-root", type=str, required=True)
    parser.add_argument("--forward-aggregate-summary", type=str, required=True)
    parser.add_argument("--reverse-aggregate-summary", type=str, required=True)
    parser.add_argument("--bidirectional-summary", type=str, required=True)
    parser.add_argument("--model-path", type=str, default="./external/models/Qwen3-1.7B")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--output-root", type=str, required=True)
    args = parser.parse_args()

    out_root = Path(args.output_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    groups = load_groups(
        Path(args.forward_aggregate_summary).resolve(),
        Path(args.reverse_aggregate_summary).resolve(),
        Path(args.bidirectional_summary).resolve(),
    )
    heads_all = sorted({h for hs in groups.values() for h in hs})
    if not heads_all:
        raise ValueError("No head nodes found in any group.")

    head_specs = [(h, *parse_head(h)) for h in heads_all]
    layers_needed = [layer for _, layer, _ in head_specs]
    needed_names = needed_pattern_names(layers_needed)

    reverse_batch_root = Path(args.reverse_batch_root).resolve()
    sample_ids = [r.sample_id for r in load_summary_records(reverse_batch_root)]
    if args.max_samples > 0:
        sample_ids = sample_ids[: args.max_samples]
    if not sample_ids:
        raise ValueError(f"No samples found under {reverse_batch_root}")

    dataset_root = Path(args.dataset_root).resolve()
    clean_dir = dataset_root / "clean"
    corrupt_dir = dataset_root / "corrupt"

    model, tokenizer = load_hooked_qwen3(args.model_path, device=args.device, dtype=torch.bfloat16)

    set_names = [
        "contrast",
        "tool_call_tags",
        "tools_block",
        "user_block",
        "prefix_16",
        "recent_32",
    ]

    # Store per-head per-set per-condition values.
    acc_tool: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    acc_no: Dict[Tuple[str, str], List[float]] = defaultdict(list)

    pbar = tqdm(sample_ids, desc="Head reads", dynamic_ncols=True)
    for sample_id in pbar:
        tool_path = clean_dir / f"{sample_id}.txt"
        no_path = corrupt_dir / f"{sample_id}.txt"
        if not tool_path.exists() or not no_path.exists():
            continue
        tool_text = tool_path.read_text(encoding="utf-8")
        no_text = no_path.read_text(encoding="utf-8")

        try:
            pos_sets = position_sets_from_prompts(model, tokenizer, tool_text, no_text)
        except Exception:
            continue

        tool_tokens = model.to_tokens(tool_text, prepend_bos=False)
        no_tokens = model.to_tokens(no_text, prepend_bos=False)
        if tool_tokens.shape != no_tokens.shape:
            continue

        for label, tokens in [("tool", tool_tokens), ("no", no_tokens)]:
            with torch.no_grad():
                _, cache = model.run_with_cache(tokens, names_filter=lambda n: n in set(needed_names))

            for head, layer, head_idx in head_specs:
                hook = f"blocks.{layer}.attn.hook_pattern"
                if hook not in cache:
                    continue
                pattern = cache[hook][0, head_idx, -1, :]  # [seq]
                for sname in set_names:
                    positions = pos_sets.get(sname, [])
                    val = mass_to_positions(pattern, positions)
                    if label == "tool":
                        acc_tool[(head, sname)].append(val)
                    else:
                        acc_no[(head, sname)].append(val)

            del cache
            torch.cuda.empty_cache()
            model.reset_hooks()

    # Per-head rows.
    head_rows: List[Dict[str, object]] = []
    for head, layer, head_idx in head_specs:
        _ = head_idx
        for sname in set_names:
            tool_vals = acc_tool.get((head, sname), [])
            no_vals = acc_no.get((head, sname), [])
            tool_med = med(tool_vals)
            no_med = med(no_vals)
            head_rows.append(
                {
                    "head": head,
                    "layer": layer,
                    "set": sname,
                    "n": min(len(tool_vals), len(no_vals)),
                    "tool_mass_median": tool_med,
                    "no_tool_mass_median": no_med,
                    "delta_median_no_minus_tool": no_med - tool_med if math.isfinite(tool_med) and math.isfinite(no_med) else float("nan"),
                    "tool_mass_mean": mean(tool_vals),
                    "no_tool_mass_mean": mean(no_vals),
                }
            )

    head_rows.sort(key=lambda r: (r["layer"], r["head"], r["set"]))
    with (out_root / "per_head_read_mass.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(head_rows[0].keys()))
        writer.writeheader()
        writer.writerows(head_rows)

    # Per-group aggregation (pool heads in group, then median across all head-sample points).
    group_rows: List[Dict[str, object]] = []
    for gname, heads in groups.items():
        for sname in set_names:
            tool_vals: List[float] = []
            no_vals: List[float] = []
            for head in heads:
                tool_vals.extend(acc_tool.get((head, sname), []))
                no_vals.extend(acc_no.get((head, sname), []))
            tmed = med(tool_vals)
            nmed = med(no_vals)
            group_rows.append(
                {
                    "group": gname,
                    "set": sname,
                    "n_points": min(len(tool_vals), len(no_vals)),
                    "tool_mass_median": tmed,
                    "no_tool_mass_median": nmed,
                    "delta_median_no_minus_tool": nmed - tmed if math.isfinite(tmed) and math.isfinite(nmed) else float("nan"),
                }
            )

    group_rows.sort(key=lambda r: (r["group"], r["set"]))
    with (out_root / "per_group_read_mass.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(group_rows[0].keys()))
        writer.writeheader()
        writer.writerows(group_rows)

    # Heatmap: heads x sets for delta.
    heads_labels = [h for h, _, _ in head_specs]
    cols = set_names
    mat = np.full((len(heads_labels), len(cols)), np.nan, dtype=np.float64)
    for i, head in enumerate(heads_labels):
        for j, sname in enumerate(cols):
            row = next((r for r in head_rows if r["head"] == head and r["set"] == sname), None)
            if row is None:
                continue
            mat[i, j] = float(row["delta_median_no_minus_tool"])
    save_heatmap(
        mat,
        row_labels=heads_labels,
        col_labels=cols,
        title="Head Read Delta by Position Set (median over samples)",
        out_path=out_root / "head_read_delta_heatmap.png",
    )

    report = {
        "n_samples": len(sample_ids),
        "groups": groups,
        "set_names": set_names,
        "artifacts": {
            "per_head_csv": str(out_root / "per_head_read_mass.csv"),
            "per_group_csv": str(out_root / "per_group_read_mass.csv"),
            "heatmap_png": str(out_root / "head_read_delta_heatmap.png"),
        },
    }
    (out_root / "head_read_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
