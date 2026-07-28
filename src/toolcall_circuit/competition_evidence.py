#!/usr/bin/env python3
"""
Competition / co-activation evidence analysis for the dual-pathway narrative.

Task-aligned outputs:
- coactivation_per_sample.csv
- coactivation_summary.csv
- pathway_correlation.csv
- margin_conditioned_activation.csv
- coactivation_distribution.png
- margin_conditioned_heatmap.png
- competition_evidence_summary.json

The local route geometry is always fit on the train split and then applied to
both train and test, following the task specification.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm.auto import tqdm

try:
    from scipy.stats import pearsonr
except Exception:  # pragma: no cover - scipy is expected, but keep a fallback.
    pearsonr = None

if __package__ in {None, ""}:
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from toolcall_circuit.dataset import ToolCallSample, load_dataset_samples
from toolcall_circuit.paths import DATASETS_ROOT, MODEL_PATH_DEFAULT, RESULTS_ROOT
from toolcall_circuit.single_sample import load_hooked_qwen3, parse_head


ROUTE_DISPATCH_NODES = ("MLP11", "MLP16")
CONSTRUCTION_NODES = ("MLP19", "L20H5", "L21H1", "L21H12", "L24H6", "MLP27")
SUPPRESSION_NODES = ("L16H4", "MLP17", "L23H6")
NODE_ORDER = ROUTE_DISPATCH_NODES + CONSTRUCTION_NODES + SUPPRESSION_NODES

PATHWAY_BY_NODE = {
    "MLP11": "route_dispatch",
    "MLP16": "route_dispatch",
    "MLP19": "construction",
    "L20H5": "construction",
    "L21H1": "construction",
    "L21H12": "construction",
    "L24H6": "construction",
    "MLP27": "construction",
    "L16H4": "suppression",
    "MLP17": "suppression",
    "L23H6": "suppression",
}

NODE_SPECS: Dict[str, Tuple[str, int, int | None]] = {
    "MLP11": ("mlp", 11, None),
    "MLP16": ("mlp", 16, None),
    "MLP19": ("mlp", 19, None),
    "L20H5": ("head", 20, 5),
    "L21H1": ("head", 21, 1),
    "L21H12": ("head", 21, 12),
    "L24H6": ("head", 24, 6),
    "MLP27": ("mlp", 27, None),
    "L16H4": ("head", 16, 4),
    "MLP17": ("mlp", 17, None),
    "L23H6": ("head", 23, 6),
}

MARGIN_BINS: Tuple[Tuple[str, float, float], ...] = (
    ("strong_clean", 0.8, math.inf),
    ("weak_clean", 0.2, 0.8),
    ("ambiguous", -0.2, 0.2),
    ("weak_corrupt", -0.8, -0.2),
    ("strong_corrupt", -math.inf, -0.8),
)

CONDITION_COLORS = {"clean": "#1f77b4", "corrupt": "#d95f02"}
SPLIT_LINESTYLES = {"train": "-", "test": "--"}
HEATMAP_CMAP = "YlOrRd"


def finite(values: Iterable[float]) -> List[float]:
    out: List[float] = []
    for value in values:
        try:
            number = float(value)
        except Exception:
            continue
        if math.isfinite(number):
            out.append(number)
    return out


def mean(values: Iterable[float]) -> float:
    vals = finite(values)
    return float(np.mean(vals)) if vals else float("nan")


def stddev(values: Iterable[float]) -> float:
    vals = finite(values)
    if not vals:
        return float("nan")
    return float(np.std(np.asarray(vals, dtype=float), ddof=0))


def safe_rate(values: Iterable[bool]) -> float:
    vals = [1.0 if bool(v) else 0.0 for v in values]
    return float(np.mean(vals)) if vals else float("nan")


def unit(vec: torch.Tensor) -> torch.Tensor:
    denom = float(vec.norm().item())
    if denom < 1e-8:
        return torch.zeros_like(vec)
    return vec / denom


def maybe_cap_gpu_memory(device: str, max_gpu_memory_gb: float) -> Dict[str, object]:
    if max_gpu_memory_gb <= 0 or not str(device).startswith("cuda") or not torch.cuda.is_available():
        return {"enabled": False}

    index = 0
    if ":" in str(device):
        try:
            index = int(str(device).split(":")[-1])
        except Exception:
            index = 0
    props = torch.cuda.get_device_properties(index)
    total_gb = float(props.total_memory) / (1024**3)
    fraction = min(max_gpu_memory_gb / max(total_gb, 1e-6), 0.98)
    torch.cuda.set_per_process_memory_fraction(fraction, index)
    return {
        "enabled": True,
        "device_index": index,
        "device_name": props.name,
        "total_gpu_memory_gb": round(total_gb, 3),
        "requested_gpu_memory_gb": float(max_gpu_memory_gb),
        "memory_fraction": round(fraction, 6),
    }


def hook_name(node: str) -> str:
    kind, layer, _head = NODE_SPECS[node]
    if kind == "mlp":
        return f"blocks.{layer}.hook_mlp_out"
    return f"blocks.{layer}.attn.hook_z"


def collect_cache(model, tokens: torch.Tensor, hook_names: Sequence[str]) -> Dict[str, torch.Tensor]:
    wanted = set(hook_names)
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens, names_filter=lambda name: name in wanted)
    return {name: act.detach().cpu() for name, act in cache.items()}


def extract_node(cache: Mapping[str, torch.Tensor], node: str) -> torch.Tensor:
    kind, layer, head = NODE_SPECS[node]
    if kind == "mlp":
        return cache[f"blocks.{layer}.hook_mlp_out"][0, -1, :].float()
    return cache[f"blocks.{layer}.attn.hook_z"][0, -1, int(head), :].float()


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def resolve_condition_candidate(sample: ToolCallSample, condition: str) -> str:
    clean_manifest = sample.clean_manifest or {}
    corrupt_manifest = sample.corrupt_manifest or {}
    if condition == "clean":
        return str(clean_manifest.get("clean_candidate") or "")
    return str(
        clean_manifest.get("corrupt_candidate")
        or corrupt_manifest.get("assigned_candidate")
        or corrupt_manifest.get("clean_candidate")
        or ""
    )


def prompt_preview(text: str, limit: int = 160) -> str:
    preview = " ".join(text.replace("\n", " ").replace("\t", " ").split())
    return preview[:limit]


def load_or_extract_split_cache(
    *,
    split_name: str,
    dataset_root: Path,
    cache_path: Path,
    model=None,
    max_samples: int,
    use_cache: bool,
) -> Dict[str, object]:
    if use_cache and cache_path.exists():
        return torch.load(cache_path, map_location="cpu")
    if model is None:
        raise ValueError(f"{split_name} cache missing at {cache_path}, but no model was provided for extraction.")

    samples = load_dataset_samples(dataset_root.resolve())
    if max_samples > 0:
        samples = samples[:max_samples]
    if not samples:
        raise ValueError(f"{split_name} split has no samples.")

    hook_names = sorted({hook_name(node) for node in NODE_ORDER})
    act_store: Dict[str, List[torch.Tensor]] = {node: [] for node in NODE_ORDER}
    metadata_rows: List[Dict[str, object]] = []

    pbar = tqdm(samples, desc=f"Extract {split_name}", dynamic_ncols=True)
    for sample_idx, sample in enumerate(pbar, start=1):
        texts = {
            "clean": sample.clean_path.read_text(encoding="utf-8"),
            "corrupt": sample.corrupt_path.read_text(encoding="utf-8"),
        }
        for condition, text in texts.items():
            tokens = model.to_tokens(text, prepend_bos=False)
            cache = collect_cache(model, tokens, hook_names)
            metadata_rows.append(
                {
                    "sample_id": sample.sample_id,
                    "sample_rank": int(sample.sample_rank),
                    "split": split_name,
                    "condition": condition,
                    "label": 1 if condition == "clean" else 0,
                    "language": str((sample.clean_manifest or {}).get("language") or (sample.corrupt_manifest or {}).get("language") or ""),
                    "dataset_name": str((sample.clean_manifest or {}).get("dataset_name") or (sample.clean_manifest or {}).get("dataset") or ""),
                    "candidate_text": resolve_condition_candidate(sample, condition),
                    "prompt_token_length": int(tokens.shape[-1]),
                    "prompt_preview": prompt_preview(text),
                }
            )
            for node in NODE_ORDER:
                act_store[node].append(extract_node(cache, node).to(dtype=torch.float16))
            del tokens
            del cache
        if sample_idx % 25 == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    cache_dict = {
        "split": split_name,
        "dataset_root": str(dataset_root.resolve()),
        "n_pairs": len(samples),
        "n_rows": len(metadata_rows),
        "node_order": list(NODE_ORDER),
        "metadata": metadata_rows,
        "activations": {node: torch.stack(rows, dim=0) for node, rows in act_store.items()},
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache_dict, cache_path)
    return cache_dict


def build_route_geometry(train_cache: Mapping[str, object]) -> Dict[str, Dict[str, torch.Tensor | float]]:
    metadata = train_cache["metadata"]  # type: ignore[index]
    labels = torch.tensor([int(row["label"]) for row in metadata], dtype=torch.bool)
    geometry: Dict[str, Dict[str, torch.Tensor | float]] = {}
    for node in NODE_ORDER:
        acts = train_cache["activations"][node].float()  # type: ignore[index]
        clean = acts[labels]
        corrupt = acts[~labels]
        mu_clean = clean.mean(dim=0)
        mu_corrupt = corrupt.mean(dim=0)
        direction = unit(mu_clean - mu_corrupt)
        midpoint = 0.5 * (mu_clean + mu_corrupt)
        scale = float(torch.dot(mu_clean - midpoint, direction).item())
        geometry[node] = {
            "mu_clean": mu_clean,
            "mu_corrupt": mu_corrupt,
            "direction": direction,
            "midpoint": midpoint,
            "scale": scale,
        }
    return geometry


def assign_margin_bin(score: float) -> str:
    for name, lower, upper in MARGIN_BINS:
        if lower == -math.inf and score < upper:
            return name
        if upper == math.inf and score > lower:
            return name
        if lower <= score <= upper:
            return name
    return "unknown"


def summarize_rows(rows: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[Tuple[str, str, str, str], List[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        key = (str(row["split"]), str(row["node"]), str(row["pathway"]), str(row["condition"]))
        grouped[key].append(row)

    summary_rows: List[Dict[str, object]] = []
    for split, node, pathway, condition in sorted(grouped.keys()):
        chunk = grouped[(split, node, pathway, condition)]
        route_scores = [float(row["route_score"]) for row in chunk]
        direction_projections = [float(row["direction_projection"]) for row in chunk]
        activation_norms = [float(row["activation_norm"]) for row in chunk]
        summary_rows.append(
            {
                "split": split,
                "node": node,
                "pathway": pathway,
                "condition": condition,
                "n_samples": len(chunk),
                "route_score_mean": mean(route_scores),
                "route_score_std": stddev(route_scores),
                "direction_projection_mean": mean(direction_projections),
                "direction_projection_std": stddev(direction_projections),
                "activation_norm_mean": mean(activation_norms),
                "activation_norm_std": stddev(activation_norms),
                "nonzero_rate": safe_rate(abs(v) > 0.1 for v in route_scores),
            }
        )
    return summary_rows


def safe_pearsonr(xs: Sequence[float], ys: Sequence[float]) -> Tuple[float, float]:
    x = np.asarray([float(v) for v in xs], dtype=float)
    y = np.asarray([float(v) for v in ys], dtype=float)
    if x.size < 2 or y.size < 2:
        return float("nan"), float("nan")
    if pearsonr is not None:
        result = pearsonr(x, y)
        try:
            return float(result.statistic), float(result.pvalue)
        except AttributeError:
            return float(result[0]), float(result[1])
    x = x - x.mean()
    y = y - y.mean()
    denom = math.sqrt(float((x * x).sum()) * float((y * y).sum()))
    if denom < 1e-12:
        return float("nan"), float("nan")
    r = float((x * y).sum() / denom)
    return r, float("nan")


def build_pathway_correlation(rows: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[Tuple[str, str], List[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        split = str(row["split"])
        grouped[(split, "all")].append(row)
        grouped[(split, str(row["condition"]))].append(row)
        if str(row["margin_bin"]) == "ambiguous":
            grouped[(split, "ambiguous")].append(row)

    corr_rows: List[Dict[str, object]] = []
    for (split, subset), chunk in sorted(grouped.items()):
        by_key = {(str(row["sample_id"]), str(row["condition"]), str(row["node"])): row for row in chunk}
        for construction_node in CONSTRUCTION_NODES:
            for suppression_node in SUPPRESSION_NODES:
                common_keys = []
                for row in chunk:
                    if str(row["node"]) != construction_node:
                        continue
                    key = (str(row["sample_id"]), str(row["condition"]), suppression_node)
                    if key in by_key:
                        common_keys.append((str(row["sample_id"]), str(row["condition"])))
                if not common_keys:
                    continue
                construction_scores = []
                suppression_scores = []
                suppression_competition_scores = []
                for sample_id, condition in common_keys:
                    c_row = by_key[(sample_id, condition, construction_node)]
                    s_row = by_key[(sample_id, condition, suppression_node)]
                    c_score = float(c_row["route_score"])
                    s_score = float(s_row["route_score"])
                    construction_scores.append(c_score)
                    suppression_scores.append(s_score)
                    # Sign-align suppression so positive means "more suppressive".
                    suppression_competition_scores.append(-s_score)
                raw_r, raw_p = safe_pearsonr(construction_scores, suppression_scores)
                aligned_r, aligned_p = safe_pearsonr(construction_scores, suppression_competition_scores)
                corr_rows.append(
                    {
                        "split": split,
                        "subset": subset,
                        "construction_node": construction_node,
                        "suppression_node": suppression_node,
                        "pearson_r": raw_r,
                        "p_value": raw_p,
                        "competition_aligned_r": aligned_r,
                        "competition_aligned_p_value": aligned_p,
                        "n_samples": len(construction_scores),
                    }
                )
    return corr_rows


def build_margin_conditioned_activation(rows: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[Tuple[str, str, str], List[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["split"]), str(row["margin_bin"]), str(row["node"]))].append(row)

    out: List[Dict[str, object]] = []
    for split, margin_bin, node in sorted(grouped.keys()):
        chunk = grouped[(split, margin_bin, node)]
        values = [float(row["direction_projection"]) for row in chunk]
        out.append(
            {
                "split": split,
                "margin_bin": margin_bin,
                "node": node,
                "pathway": PATHWAY_BY_NODE[node],
                "mean_direction_projection": mean(values),
                "std": stddev(values),
                "n_samples": len(chunk),
            }
        )
    return out


def collect_per_sample_rows(
    cache: Mapping[str, object],
    geometry: Mapping[str, Mapping[str, torch.Tensor | float]],
) -> List[Dict[str, object]]:
    metadata = cache["metadata"]  # type: ignore[index]
    score_map: Dict[str, np.ndarray] = {}
    norm_map: Dict[str, np.ndarray] = {}

    for node in NODE_ORDER:
        acts = cache["activations"][node].float()  # type: ignore[index]
        direction = geometry[node]["direction"]  # type: ignore[index]
        midpoint = geometry[node]["midpoint"]  # type: ignore[index]
        scale = float(geometry[node]["scale"])
        centered = torch.matmul(acts - midpoint, direction)
        route_scores = centered / scale
        score_map[node] = route_scores.cpu().numpy()
        norm_map[node] = acts.norm(dim=1).cpu().numpy()

    rows: List[Dict[str, object]] = []
    for idx, meta in enumerate(metadata):
        margin_score = float(score_map["MLP16"][idx])
        margin_bin = assign_margin_bin(margin_score)
        for node in NODE_ORDER:
            route_score = float(score_map[node][idx])
            rows.append(
                {
                    "sample_id": meta["sample_id"],
                    "split": meta["split"],
                    "condition": meta["condition"],
                    "node": node,
                    "pathway": PATHWAY_BY_NODE[node],
                    "route_score": route_score,
                    "activation_norm": float(norm_map[node][idx]),
                    "direction_projection": abs(route_score),
                    "margin_bin": margin_bin,
                }
            )
    return rows


def plot_distribution_panel(
    ax: plt.Axes,
    *,
    rows: Sequence[Mapping[str, object]],
    node: str,
) -> None:
    node_rows = [row for row in rows if str(row["node"]) == node]
    values = finite(float(row["route_score"]) for row in node_rows)
    if not values:
        ax.set_visible(False)
        return
    q_lo, q_hi = np.quantile(np.asarray(values, dtype=float), [0.01, 0.99])
    lo = float(min(q_lo, -1.5))
    hi = float(max(q_hi, 1.5))
    bins = np.linspace(lo, hi, 40)

    for split in ("train", "test"):
        for condition in ("clean", "corrupt"):
            subset = [
                float(row["route_score"])
                for row in node_rows
                if str(row["split"]) == split and str(row["condition"]) == condition
            ]
            if len(subset) < 2:
                continue
            hist, edges = np.histogram(np.asarray(subset, dtype=float), bins=bins, density=True)
            centers = 0.5 * (edges[:-1] + edges[1:])
            ax.plot(
                centers,
                hist,
                color=CONDITION_COLORS[condition],
                linestyle=SPLIT_LINESTYLES[split],
                linewidth=1.8,
                label=f"{split}-{condition}",
            )

    ax.axvline(0.0, color="#666666", linewidth=1.0, alpha=0.7)
    ax.set_title(node, fontsize=10)
    ax.set_xlim(lo, hi)
    ax.tick_params(labelsize=8)


def plot_coactivation_distribution(rows: Sequence[Mapping[str, object]], out_path: Path) -> None:
    fig, axes = plt.subplots(2, 6, figsize=(18, 6), constrained_layout=True)

    for col, node in enumerate(CONSTRUCTION_NODES):
        plot_distribution_panel(axes[0, col], rows=rows, node=node)
        axes[0, col].set_xlabel("route_score", fontsize=9)
        if col == 0:
            axes[0, col].set_ylabel("density", fontsize=9)

    for col, node in enumerate(SUPPRESSION_NODES):
        plot_distribution_panel(axes[1, col], rows=rows, node=node)
        axes[1, col].set_xlabel("route_score", fontsize=9)
        if col == 0:
            axes[1, col].set_ylabel("density", fontsize=9)

    for col in range(len(SUPPRESSION_NODES), 6):
        axes[1, col].axis("off")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, fontsize=9)

    fig.suptitle(
        "Natural forward route-score distributions\nTop: construction nodes | Bottom: suppression nodes",
        fontsize=13,
        y=1.02,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_margin_conditioned_heatmap(
    rows: Sequence[Mapping[str, object]],
    out_path: Path,
) -> None:
    bin_names = [name for name, _lo, _hi in MARGIN_BINS]
    split_names = ["train", "test"]
    value_map = {
        (str(row["split"]), str(row["margin_bin"]), str(row["node"])): float(row["mean_direction_projection"])
        for row in rows
    }

    matrices: Dict[str, np.ndarray] = {}
    for split in split_names:
        matrix = np.full((len(bin_names), len(NODE_ORDER)), np.nan, dtype=float)
        for i, margin_bin in enumerate(bin_names):
            for j, node in enumerate(NODE_ORDER):
                matrix[i, j] = value_map.get((split, margin_bin, node), float("nan"))
        matrices[split] = matrix

    finite_vals = np.concatenate([m[np.isfinite(m)] for m in matrices.values() if np.isfinite(m).any()])
    vmax = float(np.max(finite_vals)) if finite_vals.size else 1.0

    fig, axes = plt.subplots(1, 2, figsize=(16, 5), constrained_layout=True)
    for ax, split in zip(axes, split_names):
        image = ax.imshow(matrices[split], aspect="auto", cmap=HEATMAP_CMAP, vmin=0.0, vmax=vmax)
        ax.set_title(split, fontsize=12)
        ax.set_xticks(np.arange(len(NODE_ORDER)))
        ax.set_xticklabels(NODE_ORDER, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(np.arange(len(bin_names)))
        ax.set_yticklabels(bin_names, fontsize=9)
    cbar = fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.92)
    cbar.set_label("mean |route_score|", fontsize=10)
    fig.suptitle("Margin-conditioned node activation (binned by MLP16 route score)", fontsize=13)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def first_row(
    rows: Sequence[Mapping[str, object]],
    **conditions: object,
) -> Mapping[str, object] | None:
    for row in rows:
        if all(str(row.get(key)) == str(value) for key, value in conditions.items()):
            return row
    return None


def summarize_key_findings(
    *,
    per_sample_rows: Sequence[Mapping[str, object]],
    summary_rows: Sequence[Mapping[str, object]],
    correlation_rows: Sequence[Mapping[str, object]],
    margin_rows: Sequence[Mapping[str, object]],
) -> Dict[str, object]:
    available_splits = sorted({str(row["split"]) for row in per_sample_rows})

    def summary_lookup(split: str, node: str, condition: str) -> Mapping[str, object] | None:
        return first_row(summary_rows, split=split, node=node, condition=condition)

    findings: Dict[str, object] = {
        "suppression_on_clean": {},
        "construction_on_corrupt": {},
        "ambiguous_bin": {},
        "correlation_highlights": {},
    }

    for split in available_splits:
        findings["suppression_on_clean"][split] = {
            node: {
                "route_score_mean": float(summary_lookup(split, node, "clean")["route_score_mean"]),  # type: ignore[index]
                "direction_projection_mean": float(summary_lookup(split, node, "clean")["direction_projection_mean"]),  # type: ignore[index]
                "nonzero_rate": float(summary_lookup(split, node, "clean")["nonzero_rate"]),  # type: ignore[index]
                "suppress_direction_rate": safe_rate(
                    float(row["route_score"]) < -0.1
                    for row in per_sample_rows
                    if str(row["split"]) == split and str(row["node"]) == node and str(row["condition"]) == "clean"
                ),
            }
            for node in SUPPRESSION_NODES
        }
        findings["construction_on_corrupt"][split] = {
            node: {
                "route_score_mean": float(summary_lookup(split, node, "corrupt")["route_score_mean"]),  # type: ignore[index]
                "direction_projection_mean": float(summary_lookup(split, node, "corrupt")["direction_projection_mean"]),  # type: ignore[index]
                "nonzero_rate": float(summary_lookup(split, node, "corrupt")["nonzero_rate"]),  # type: ignore[index]
                "construct_direction_rate": safe_rate(
                    float(row["route_score"]) > 0.1
                    for row in per_sample_rows
                    if str(row["split"]) == split and str(row["node"]) == node and str(row["condition"]) == "corrupt"
                ),
            }
            for node in CONSTRUCTION_NODES
        }

        split_margin_rows = [row for row in margin_rows if str(row["split"]) == split and str(row["margin_bin"]) == "ambiguous"]
        findings["ambiguous_bin"][split] = {
            "n_rows": int(
                len(
                    {
                        (str(row["sample_id"]), str(row["condition"]))
                        for row in per_sample_rows
                        if str(row["split"]) == split and str(row["margin_bin"]) == "ambiguous"
                    }
                )
            ),
            "construction_mean_direction_projection": mean(
                float(row["mean_direction_projection"])
                for row in split_margin_rows
                if str(row["pathway"]) == "construction"
            ),
            "suppression_mean_direction_projection": mean(
                float(row["mean_direction_projection"])
                for row in split_margin_rows
                if str(row["pathway"]) == "suppression"
            ),
            "route_dispatch_mean_direction_projection": mean(
                float(row["mean_direction_projection"])
                for row in split_margin_rows
                if str(row["pathway"]) == "route_dispatch"
            ),
        }

        split_corr = [row for row in correlation_rows if str(row["split"]) == split and str(row["subset"]) == "ambiguous"]
        split_corr_sorted = sorted(split_corr, key=lambda row: float(row["competition_aligned_r"]))
        findings["correlation_highlights"][split] = {
            "most_negative_ambiguous_pair": split_corr_sorted[0] if split_corr_sorted else None,
            "most_positive_ambiguous_pair": split_corr_sorted[-1] if split_corr_sorted else None,
        }

    # Heuristic recommendation for the paper narrative.
    clean_suppress_rates = finite(
        findings["suppression_on_clean"][split][node]["suppress_direction_rate"]
        for split in available_splits
        for node in SUPPRESSION_NODES
    )
    corrupt_construct_rates = finite(
        findings["construction_on_corrupt"][split][node]["construct_direction_rate"]
        for split in available_splits
        for node in CONSTRUCTION_NODES
    )
    ambiguous_construction = finite(
        findings["ambiguous_bin"][split]["construction_mean_direction_projection"] for split in available_splits
    )
    ambiguous_suppression = finite(
        findings["ambiguous_bin"][split]["suppression_mean_direction_projection"] for split in available_splits
    )
    if (
        mean(clean_suppress_rates) <= 0.15
        and mean(corrupt_construct_rates) <= 0.05
        and mean(ambiguous_construction) >= 0.4
        and mean(ambiguous_suppression) >= 0.4
    ):
        recommendation = "conditional_dual_pathway_with_boundary_local_weak_competition"
    elif (
        mean(clean_suppress_rates) >= 0.2
        and mean(corrupt_construct_rates) >= 0.2
        and mean(ambiguous_construction) >= 0.2
        and mean(ambiguous_suppression) >= 0.2
    ):
        recommendation = "competition"
    elif mean(clean_suppress_rates) <= 0.05 and mean(corrupt_construct_rates) <= 0.05:
        recommendation = "conditional_dual_pathway"
    else:
        recommendation = "mixed_or_weak_competition"
    findings["recommended_narrative"] = recommendation
    return findings


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH_DEFAULT)
    parser.add_argument("--train-dataset-root", type=Path, default=DATASETS_ROOT / "train")
    parser.add_argument("--test-dataset-root", type=Path, default=DATASETS_ROOT / "test")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=RESULTS_ROOT / "split" / "competition_evidence",
    )
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-test-samples", type=int, default=0)
    parser.add_argument("--max-gpu-memory-gb", type=float, default=24.0)
    parser.add_argument("--disable-cache", action="store_true")
    parser.add_argument("--skip-test", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_argparser()
    args = parser.parse_args(argv)

    args.output_root = args.output_root.resolve()
    cache_dir = (args.cache_dir or (args.output_root / "cache")).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    args.output_root.mkdir(parents=True, exist_ok=True)

    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    if args.dtype not in dtype_map:
        raise ValueError(f"Unsupported dtype: {args.dtype}")
    dtype = dtype_map[args.dtype]

    train_cache_path = cache_dir / "train_activation_cache.pt"
    test_cache_path = cache_dir / "test_activation_cache.pt"
    need_model = args.disable_cache or not train_cache_path.exists() or (not args.skip_test and not test_cache_path.exists())
    gpu_limit_info = {
        "enabled": False,
        "reason": "cache_only_run",
        "requested_gpu_memory_gb": float(args.max_gpu_memory_gb),
    }
    model = None
    if need_model:
        gpu_limit_info = maybe_cap_gpu_memory(args.device, args.max_gpu_memory_gb)
        model, _tokenizer = load_hooked_qwen3(str(args.model_path), device=args.device, dtype=dtype)

    train_cache = load_or_extract_split_cache(
        split_name="train",
        dataset_root=args.train_dataset_root,
        cache_path=train_cache_path,
        model=model,
        max_samples=args.max_train_samples,
        use_cache=not args.disable_cache,
    )
    test_cache = None
    if not args.skip_test:
        test_cache = load_or_extract_split_cache(
            split_name="test",
            dataset_root=args.test_dataset_root,
            cache_path=test_cache_path,
            model=model,
            max_samples=args.max_test_samples,
            use_cache=not args.disable_cache,
        )

    geometry = build_route_geometry(train_cache)
    per_sample_rows = collect_per_sample_rows(train_cache, geometry)
    if test_cache is not None:
        per_sample_rows.extend(collect_per_sample_rows(test_cache, geometry))
    summary_rows = summarize_rows(per_sample_rows)
    correlation_rows = build_pathway_correlation(per_sample_rows)
    margin_rows = build_margin_conditioned_activation(per_sample_rows)

    write_csv(args.output_root / "coactivation_per_sample.csv", per_sample_rows)
    write_csv(args.output_root / "coactivation_summary.csv", summary_rows)
    write_csv(args.output_root / "pathway_correlation.csv", correlation_rows)
    write_csv(args.output_root / "margin_conditioned_activation.csv", margin_rows)

    plot_coactivation_distribution(per_sample_rows, args.output_root / "coactivation_distribution.png")
    plot_margin_conditioned_heatmap(margin_rows, args.output_root / "margin_conditioned_heatmap.png")

    findings = summarize_key_findings(
        per_sample_rows=per_sample_rows,
        summary_rows=summary_rows,
        correlation_rows=correlation_rows,
        margin_rows=margin_rows,
    )

    summary_json = {
        "metadata": {
            "model_path": str(args.model_path),
            "train_dataset_root": str(args.train_dataset_root.resolve()),
            "test_dataset_root": str(args.test_dataset_root.resolve()),
            "output_root": str(args.output_root),
            "route_geometry_source": "train_split_only",
            "attention_representation": "hook_z",
            "mlp_representation": "hook_mlp_out",
            "direction_projection_definition": "abs(route_score)",
            "dtype": args.dtype,
            "device": args.device,
            "gpu_limit": gpu_limit_info,
        },
        "counts": {
            "train_pairs": int(train_cache["n_pairs"]),
            "test_pairs": int(test_cache["n_pairs"]) if test_cache is not None else 0,
            "train_rows": int(train_cache["n_rows"]),
            "test_rows": int(test_cache["n_rows"]) if test_cache is not None else 0,
            "per_sample_rows": len(per_sample_rows),
            "summary_rows": len(summary_rows),
            "correlation_rows": len(correlation_rows),
            "margin_rows": len(margin_rows),
        },
        "artifacts": {
            "coactivation_per_sample_csv": str(args.output_root / "coactivation_per_sample.csv"),
            "coactivation_summary_csv": str(args.output_root / "coactivation_summary.csv"),
            "pathway_correlation_csv": str(args.output_root / "pathway_correlation.csv"),
            "margin_conditioned_activation_csv": str(args.output_root / "margin_conditioned_activation.csv"),
            "coactivation_distribution_png": str(args.output_root / "coactivation_distribution.png"),
            "margin_conditioned_heatmap_png": str(args.output_root / "margin_conditioned_heatmap.png"),
            "train_activation_cache": str(train_cache_path),
            "test_activation_cache": str(test_cache_path),
        },
        "key_findings": findings,
    }
    (args.output_root / "competition_evidence_summary.json").write_text(
        json.dumps(summary_json, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
