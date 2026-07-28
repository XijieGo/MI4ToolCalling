#!/usr/bin/env python3
"""
Task 4: bidirectional vs. unidirectional circuit faithfulness comparison.

This analysis adds three artifacts on top of the existing split pipeline:

1. Bidirectional sufficiency comparison for forward / reverse / unified circuits.
2. Forward-only vs unified suppression stagewise comparison.
3. Forward-only vs unified construction stagewise comparison.

We keep the hook convention fixed to the existing codebase:
- attention heads: `attn.hook_z`
- MLPs: `hook_mlp_out`
- patch position: last token only
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm.auto import tqdm

if __package__ in {None, ""}:
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from toolcall_circuit.bidirectional_causal_eval import collect_cache_cpu_for_nodes
from toolcall_circuit.bidirectional_token_flip import run_logits_on_base_with_source
from toolcall_circuit.dataset import load_dataset_samples
from toolcall_circuit.objective import build_bidirectional_endpoint_objectives
from toolcall_circuit.signed_edge_importance import run_logits_with_assignments
from toolcall_circuit.single_sample import load_hooked_qwen3, parse_head


EXPECTED_FORWARD = {
    "MLP16",
    "MLP17",
    "L17H8",
    "MLP19",
    "L21H1",
    "L21H12",
    "L23H6",
    "L24H6",
    "MLP27",
}
EXPECTED_REVERSE = {
    "MLP16",
    "L16H4",
    "L16H8",
    "MLP17",
    "L17H8",
    "MLP19",
    "L20H5",
    "L21H1",
    "L21H12",
    "L23H6",
    "L24H6",
    "MLP27",
}
EXPECTED_UNIFIED = {
    "L2H14",
    "MLP11",
    "L12H6",
    "MLP12",
    "L13H9",
    "L15H5",
    "L16H13",
    "L16H4",
    "L16H8",
    "L16H9",
    "MLP16",
    "L17H2",
    "L17H8",
    "MLP17",
    "L18H14",
    "MLP19",
    "L20H5",
    "L21H1",
    "L21H12",
    "MLP21",
    "L23H5",
    "L23H6",
    "L24H6",
    "MLP27",
}

FORWARD_CONSTRUCTION_STAGES = [
    ("MLP19", ["MLP19"]),
    ("L21H1", ["MLP19", "L21H1"]),
    ("L21H12", ["MLP19", "L21H1", "L21H12"]),
    ("L24H6", ["MLP19", "L21H1", "L21H12", "L24H6"]),
    ("MLP27", ["MLP19", "L21H1", "L21H12", "L24H6", "MLP27"]),
]

SUPPRESSION_NODE_SPECS: Dict[str, Tuple[str, int, int | None]] = {
    "L16H4": ("head", 16, 4),
    "MLP17": ("mlp", 17, None),
    "L23H6": ("head", 23, 6),
}
FORWARD_SUPPRESSION_STAGES = [
    ("MLP17", ["MLP17"]),
    ("L23H6", ["MLP17", "L23H6"]),
]
SUPPRESSION_DIRECTION_NODES = ["L16H4", "MLP17", "L23H6"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare forward / reverse / unified circuit faithfulness.")
    parser.add_argument(
        "--train-dataset-root",
        type=Path,
        default=Path("./datasets/train"),
    )
    parser.add_argument(
        "--test-dataset-root",
        type=Path,
        default=Path("./datasets/test"),
    )
    parser.add_argument(
        "--forward-summary",
        type=Path,
        default=Path("./results/split/pipeline/forward_aggregate/global_core_summary.json"),
    )
    parser.add_argument(
        "--reverse-summary",
        type=Path,
        default=Path("./results/split/pipeline/reverse_aggregate/global_core_summary.json"),
    )
    parser.add_argument(
        "--unified-nodes-csv",
        type=Path,
        default=Path("./results/split/pipeline/final_signed_circuit/final_signed_nodes.csv"),
    )
    parser.add_argument(
        "--construction-train-summary",
        type=Path,
        default=Path("./results/split/tool_call_construction/construction_stagewise_summary.csv"),
    )
    parser.add_argument(
        "--construction-test-summary",
        type=Path,
        default=Path("./results/split/test_validation/construction_stagewise_test.csv"),
    )
    parser.add_argument(
        "--suppression-train-summary",
        type=Path,
        default=Path("./results/split/tool_call_suppression/suppression_stagewise_summary.csv"),
    )
    parser.add_argument(
        "--suppression-test-summary",
        type=Path,
        default=Path("./results/split/test_validation/suppression_stagewise_test.csv"),
    )
    parser.add_argument("--model-path", type=str, default="./external/models/Qwen3-1.7B")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("./results/split/bidirectional_comparison"),
    )
    return parser.parse_args()


def read_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(rows: Sequence[Dict[str, object]], path: Path) -> None:
    if not rows:
        return
    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def finite(values: Iterable[float]) -> List[float]:
    out: List[float] = []
    for value in values:
        try:
            num = float(value)
        except Exception:
            continue
        if math.isfinite(num):
            out.append(num)
    return out


def median(values: Iterable[float]) -> float:
    vals = finite(values)
    return float(np.median(vals)) if vals else float("nan")


def safe_rate(values: Iterable[bool]) -> float:
    vals = [1.0 if bool(v) else 0.0 for v in values]
    return float(np.mean(vals)) if vals else float("nan")


def bootstrap_rate_ci(values: Sequence[bool], n_boot: int, seed: int) -> Tuple[float, float]:
    vals = [1.0 if bool(v) else 0.0 for v in values]
    if not vals:
        return float("nan"), float("nan")
    rng = random.Random(seed)
    n = len(vals)
    boots: List[float] = []
    for _ in range(max(1, n_boot)):
        sample = [vals[rng.randrange(n)] for __ in range(n)]
        boots.append(float(np.mean(sample)))
    boots.sort()
    lo = boots[max(0, int(0.025 * len(boots)))]
    hi = boots[min(len(boots) - 1, int(0.975 * len(boots)))]
    return float(lo), float(hi)


def decode_token(tokenizer, token_id: int) -> str:
    try:
        return tokenizer.decode([int(token_id)]).replace("\n", "\\n")
    except Exception:
        return str(token_id)


def node_sort_key(node: str) -> Tuple[int, int, int]:
    if node.startswith("MLP"):
        return (int(node[3:]), 1, 0)
    layer, head = parse_head(node)
    return (layer, 0, head)


def load_circuit_variants(args: argparse.Namespace) -> Dict[str, List[str]]:
    forward_nodes = list(read_json(args.forward_summary.resolve()).get("core_nodes", []))
    reverse_nodes = list(read_json(args.reverse_summary.resolve()).get("core_nodes", []))
    unified_nodes = [str(row["node"]) for row in read_csv_rows(args.unified_nodes_csv.resolve())]

    if set(forward_nodes) != EXPECTED_FORWARD:
        raise ValueError(f"Unexpected forward circuit: {forward_nodes}")
    if set(reverse_nodes) != EXPECTED_REVERSE:
        raise ValueError(f"Unexpected reverse circuit: {reverse_nodes}")
    if set(unified_nodes) != EXPECTED_UNIFIED:
        raise ValueError(f"Unexpected unified circuit: {unified_nodes}")

    return {
        "forward": forward_nodes,
        "reverse": reverse_nodes,
        "unified": unified_nodes,
    }


def target_vs_competitor_margin(logits: torch.Tensor, target_token_id: int) -> Tuple[float, int]:
    last = logits[0, -1].float()
    masked = last.clone()
    masked[int(target_token_id)] = float("-inf")
    competitor_id = int(masked.argmax().item())
    return float((last[int(target_token_id)] - last[competitor_id]).item()), competitor_id


def no_tool_vs_tool_margin(logits: torch.Tensor, no_tool_token_id: int, tool_token_id: int) -> float:
    last = logits[0, -1].float()
    return float((last[int(no_tool_token_id)] - last[int(tool_token_id)]).item())


def extract_node_vector(cache: Dict[str, torch.Tensor], node: str) -> torch.Tensor:
    kind, layer, head = SUPPRESSION_NODE_SPECS[node]
    if kind == "mlp":
        return cache[f"blocks.{layer}.hook_mlp_out"][0, -1, :].float()
    return cache[f"blocks.{layer}.attn.hook_z"][0, -1, int(head), :].float()


def unit(vec: torch.Tensor) -> torch.Tensor:
    denom = float(vec.norm().item())
    if denom < 1e-8:
        return torch.zeros_like(vec)
    return vec / denom


def projection_delta(base_vec: torch.Tensor, source_vec: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
    if float(direction.norm().item()) < 1e-8:
        return torch.zeros_like(base_vec)
    d = unit(direction)
    return (torch.dot(source_vec, d) - torch.dot(base_vec, d)) * d


def run_logits_with_direction_edits(
    model,
    base_tokens: torch.Tensor,
    edits: Dict[str, torch.Tensor],
) -> torch.Tensor:
    hooks = []
    for node, delta in edits.items():
        kind, layer, head = SUPPRESSION_NODE_SPECS[node]
        delta = delta.to(base_tokens.device)
        if kind == "mlp":
            hook_name = f"blocks.{layer}.hook_mlp_out"

            def make_mlp_hook(delta_vec: torch.Tensor):
                def hook_fn(mlp_out: torch.Tensor, hook):  # noqa: ANN001
                    out = mlp_out.clone()
                    out[:, -1, :] = (out[:, -1, :].float() + delta_vec.unsqueeze(0)).to(dtype=out.dtype)
                    return out

                return hook_fn

            hooks.append((hook_name, make_mlp_hook(delta)))
        else:
            hook_name = f"blocks.{layer}.attn.hook_z"
            head_idx = int(head)

            def make_head_hook(delta_vec: torch.Tensor, h: int):
                def hook_fn(z: torch.Tensor, hook):  # noqa: ANN001
                    out = z.clone()
                    out[:, -1, h, :] = (out[:, -1, h, :].float() + delta_vec.unsqueeze(0)).to(dtype=out.dtype)
                    return out

                return hook_fn

            hooks.append((hook_name, make_head_hook(delta, head_idx)))

    with torch.no_grad():
        return model.run_with_hooks(base_tokens, fwd_hooks=hooks)


def build_train_suppression_directions(model, train_samples, max_samples: int) -> Tuple[Dict[str, torch.Tensor], Dict[str, int]]:
    direction_sums: Dict[str, torch.Tensor | None] = {node: None for node in SUPPRESSION_DIRECTION_NODES}
    n_used = 0
    n_total = 0
    pbar = tqdm(train_samples, desc="Build suppression dirs", dynamic_ncols=True)
    for sample in pbar:
        if max_samples > 0 and n_total >= max_samples:
            break
        n_total += 1
        clean_text = sample.clean_path.read_text(encoding="utf-8")
        corrupt_text = sample.corrupt_path.read_text(encoding="utf-8")
        clean_tokens = model.to_tokens(clean_text, prepend_bos=False)
        corrupt_tokens = model.to_tokens(corrupt_text, prepend_bos=False)
        clean_cache = collect_cache_cpu_for_nodes(model, clean_tokens, SUPPRESSION_DIRECTION_NODES)
        corrupt_cache = collect_cache_cpu_for_nodes(model, corrupt_tokens, SUPPRESSION_DIRECTION_NODES)
        for node in SUPPRESSION_DIRECTION_NODES:
            diff = extract_node_vector(corrupt_cache, node) - extract_node_vector(clean_cache, node)
            direction_sums[node] = diff if direction_sums[node] is None else direction_sums[node] + diff
        n_used += 1
    directions = {
        node: unit(vec) for node, vec in direction_sums.items() if vec is not None
    }
    return directions, {"n_direction_samples_total": n_total, "n_direction_samples_used": n_used}


def summarize_sufficiency(
    rows: Sequence[Dict[str, object]],
    bootstrap: int,
    seed: int,
) -> List[Dict[str, object]]:
    grouped: Dict[Tuple[str, str, str], List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["circuit"]), str(row["direction"]), str(row["split"]))].append(dict(row))
    summary_rows: List[Dict[str, object]] = []
    for (circuit, direction, split), members in sorted(grouped.items()):
        top1_vals = [bool(r["top1_is_target"]) for r in members]
        rate = safe_rate(top1_vals)
        stable_offset = sum(ord(ch) for ch in f"{circuit}:{direction}:{split}")
        ci_lo, ci_hi = bootstrap_rate_ci(top1_vals, n_boot=bootstrap, seed=seed + stable_offset)
        summary_rows.append(
            {
                "circuit": circuit,
                "direction": direction,
                "split": split,
                "n_samples": len(members),
                "top1_rate": rate,
                "logit_margin_median": median(float(r["logit_margin"]) for r in members),
                "top1_rate_ci_lo": ci_lo,
                "top1_rate_ci_hi": ci_hi,
            }
        )
    return summary_rows


def evaluate_split(
    *,
    split_name: str,
    samples,
    model,
    tokenizer,
    circuit_variants: Dict[str, List[str]],
    suppression_dirs: Dict[str, torch.Tensor],
    max_samples: int,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], List[Dict[str, object]], Dict[str, int]]:
    needed_nodes = sorted(
        {
            node
            for nodes in circuit_variants.values()
            for node in nodes
        }
        | {node for _stage, nodes in FORWARD_CONSTRUCTION_STAGES for node in nodes}
        | set(SUPPRESSION_DIRECTION_NODES)
    , key=node_sort_key)

    suff_rows: List[Dict[str, object]] = []
    construction_rows: List[Dict[str, object]] = []
    suppression_rows: List[Dict[str, object]] = []
    n_total = 0
    n_used = 0
    n_shape_mismatch = 0

    pbar = tqdm(samples, desc=f"{split_name} bidirectional", dynamic_ncols=True)
    for sample in pbar:
        if max_samples > 0 and n_total >= max_samples:
            break
        n_total += 1
        clean_text = sample.clean_path.read_text(encoding="utf-8")
        corrupt_text = sample.corrupt_path.read_text(encoding="utf-8")
        clean_tokens = model.to_tokens(clean_text, prepend_bos=False)
        corrupt_tokens = model.to_tokens(corrupt_text, prepend_bos=False)
        if clean_tokens.shape != corrupt_tokens.shape:
            n_shape_mismatch += 1
            continue

        needed_hook_names = set()
        for node in needed_nodes:
            if node.startswith("MLP"):
                needed_hook_names.add(f"blocks.{int(node[3:])}.hook_mlp_out")
            else:
                layer, _head = parse_head(node)
                needed_hook_names.add(f"blocks.{layer}.attn.hook_z")
        with torch.no_grad():
            clean_logits, clean_cache_gpu = model.run_with_cache(clean_tokens, names_filter=lambda name: name in needed_hook_names)
            corrupt_logits, corrupt_cache_gpu = model.run_with_cache(corrupt_tokens, names_filter=lambda name: name in needed_hook_names)
        clean_cache = {k: v.detach().cpu() for k, v in clean_cache_gpu.items()}
        corrupt_cache = {k: v.detach().cpu() for k, v in corrupt_cache_gpu.items()}
        del clean_cache_gpu, corrupt_cache_gpu

        tool_objective, no_tool_objective = build_bidirectional_endpoint_objectives(
            clean_logits,
            corrupt_logits,
            tokenizer=tokenizer,
        )
        tool_token_id = int(tool_objective.top_token_id or int(clean_logits[0, -1].argmax().item()))
        no_tool_token_id = int(no_tool_objective.top_token_id or int(corrupt_logits[0, -1].argmax().item()))
        if tool_token_id == no_tool_token_id:
            continue

        n_used += 1

        for circuit_name, nodes in circuit_variants.items():
            tool_patched = run_logits_on_base_with_source(model, corrupt_tokens, clean_cache, nodes)
            tool_top1 = int(tool_patched[0, -1].argmax().item())
            tool_margin, competitor_id = target_vs_competitor_margin(tool_patched, tool_token_id)
            suff_rows.append(
                {
                    "sample_id": sample.sample_id,
                    "circuit": circuit_name,
                    "direction": "tool_call",
                    "split": split_name,
                    "n_nodes": len(nodes),
                    "target_token_id": tool_token_id,
                    "target_token": decode_token(tokenizer, tool_token_id),
                    "top1_token_id": tool_top1,
                    "top1_token": decode_token(tokenizer, tool_top1),
                    "top1_is_target": tool_top1 == tool_token_id,
                    "logit_margin": tool_margin,
                    "competitor_token_id": competitor_id,
                    "competitor_token": decode_token(tokenizer, competitor_id),
                    "tool_logit": float(tool_patched[0, -1, tool_token_id].item()),
                    "no_tool_logit": float(tool_patched[0, -1, no_tool_token_id].item()),
                }
            )

            no_tool_patched = run_logits_on_base_with_source(model, clean_tokens, corrupt_cache, nodes)
            no_tool_top1 = int(no_tool_patched[0, -1].argmax().item())
            no_tool_margin = no_tool_vs_tool_margin(no_tool_patched, no_tool_token_id, tool_token_id)
            suff_rows.append(
                {
                    "sample_id": sample.sample_id,
                    "circuit": circuit_name,
                    "direction": "no_tool",
                    "split": split_name,
                    "n_nodes": len(nodes),
                    "target_token_id": no_tool_token_id,
                    "target_token": decode_token(tokenizer, no_tool_token_id),
                    "top1_token_id": no_tool_top1,
                    "top1_token": decode_token(tokenizer, no_tool_top1),
                    "top1_is_target": no_tool_top1 == no_tool_token_id,
                    "logit_margin": no_tool_margin,
                    "tool_token_id": tool_token_id,
                    "tool_token": decode_token(tokenizer, tool_token_id),
                    "tool_logit": float(no_tool_patched[0, -1, tool_token_id].item()),
                    "no_tool_logit": float(no_tool_patched[0, -1, no_tool_token_id].item()),
                }
            )

        for stage_idx, (stage_name, nodes) in enumerate(FORWARD_CONSTRUCTION_STAGES, start=1):
            stage_logits = run_logits_with_assignments(
                model,
                corrupt_tokens,
                clean_cache,
                corrupt_cache,
                nodes,
                [],
            )
            top1_id = int(stage_logits[0, -1].argmax().item())
            construction_rows.append(
                {
                    "split": split_name,
                    "circuit": "forward_only",
                    "stage": stage_name,
                    "stage_idx": stage_idx,
                    "nodes": "|".join(nodes),
                    "sample_id": sample.sample_id,
                    "tool_top1": top1_id == tool_token_id,
                }
            )

        clean_vecs = {node: extract_node_vector(clean_cache, node) for node in SUPPRESSION_DIRECTION_NODES}
        corrupt_vecs = {node: extract_node_vector(corrupt_cache, node) for node in SUPPRESSION_DIRECTION_NODES}
        for stage_idx, (stage_name, nodes) in enumerate(FORWARD_SUPPRESSION_STAGES, start=1):
            edits = {
                node: projection_delta(clean_vecs[node], corrupt_vecs[node], suppression_dirs[node])
                for node in nodes
            }
            stage_logits = run_logits_with_direction_edits(model, clean_tokens, edits)
            top1_id = int(stage_logits[0, -1].argmax().item())
            suppression_rows.append(
                {
                    "split": split_name,
                    "circuit": "forward_only",
                    "stage": stage_name,
                    "stage_idx": stage_idx,
                    "nodes": "|".join(nodes),
                    "sample_id": sample.sample_id,
                    "tool_top1": top1_id == tool_token_id,
                    "no_tool_top1": top1_id == no_tool_token_id,
                }
            )

        pbar.set_postfix(sample=sample.sample_id)

    return (
        suff_rows,
        construction_rows,
        suppression_rows,
        {
            "n_samples_total": n_total,
            "n_samples_used": n_used,
            "n_shape_mismatch": n_shape_mismatch,
        },
    )


def summarize_stage_rows(
    rows: Sequence[Dict[str, object]],
    *,
    include_no_tool: bool,
) -> List[Dict[str, object]]:
    grouped: Dict[Tuple[str, str, str], List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["split"]), str(row["circuit"]), str(row["stage"]))].append(dict(row))
    out: List[Dict[str, object]] = []
    for (split, circuit, stage), members in sorted(grouped.items()):
        summary = {
            "split": split,
            "circuit": circuit,
            "stage": stage,
            "n_samples": len(members),
            "tool_top1": safe_rate(bool(r["tool_top1"]) for r in members),
        }
        if include_no_tool:
            summary["no_tool_top1"] = safe_rate(bool(r["no_tool_top1"]) for r in members)
        out.append(summary)
    return out


def load_official_construction_rows(train_path: Path, test_path: Path) -> List[Dict[str, object]]:
    mapping = {
        "plus_MLP19": "MLP19",
        "plus_L20H5": "L20H5",
        "plus_L21H1": "L21H1",
        "plus_L21H12": "L21H12",
        "plus_L24H6": "L24H6",
        "plus_MLP27": "MLP27",
    }
    out: List[Dict[str, object]] = []
    for split, path in (("train", train_path), ("test", test_path)):
        for row in read_csv_rows(path.resolve()):
            step_label = str(row.get("step_label", ""))
            if step_label not in mapping:
                continue
            out.append(
                {
                    "split": split,
                    "circuit": "unified",
                    "stage": mapping[step_label],
                    "n_samples": int(float(row["n_samples"])),
                    "tool_top1": float(row["tool_top1_rate"]),
                }
            )
    return out


def load_official_suppression_rows(train_path: Path, test_path: Path) -> List[Dict[str, object]]:
    mapping = {
        "read_only": "L16H4",
        "writer_added": "MLP17",
        "late_relay_added": "L23H6",
    }
    out: List[Dict[str, object]] = []
    for split, path in (("train", train_path), ("test", test_path)):
        for row in read_csv_rows(path.resolve()):
            stage_label = str(row.get("stage_label", ""))
            if stage_label not in mapping:
                continue
            out.append(
                {
                    "split": split,
                    "circuit": "unified",
                    "stage": mapping[stage_label],
                    "n_samples": int(float(row["n_samples"])),
                    "tool_top1": float(row["tool_top1_rate"]),
                    "no_tool_top1": float(row["no_tool_top1_rate"]),
                }
            )
    return out


def plot_all(
    suff_summary: Sequence[Dict[str, object]],
    construction_rows: Sequence[Dict[str, object]],
    suppression_rows: Sequence[Dict[str, object]],
    out_path: Path,
) -> None:
    plt.style.use("default")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        }
    )

    fig, axes = plt.subplots(1, 3, figsize=(17.0, 4.8), constrained_layout=True)

    # Panel A: sufficiency
    circuit_order = ["forward", "reverse", "unified"]
    x = np.arange(len(circuit_order))
    width = 0.18
    split_offset = {"train": -0.5 * width, "test": 0.5 * width}
    direction_offset = {"tool_call": -0.5 * width, "no_tool": 0.5 * width}
    colors = {"tool_call": "#2f6db3", "no_tool": "#d9822b"}
    hatches = {"train": "", "test": "//"}
    row_map = {(str(r["circuit"]), str(r["direction"]), str(r["split"])): r for r in suff_summary}
    for split in ("train", "test"):
        for direction in ("tool_call", "no_tool"):
            ys = []
            yerr_lo = []
            yerr_hi = []
            for circuit in circuit_order:
                row = row_map[(circuit, direction, split)]
                ys.append(float(row["top1_rate"]))
                yerr_lo.append(float(row["top1_rate"]) - float(row["top1_rate_ci_lo"]))
                yerr_hi.append(float(row["top1_rate_ci_hi"]) - float(row["top1_rate"]))
            xpos = x + split_offset[split] + direction_offset[direction]
            axes[0].bar(
                xpos,
                ys,
                width=width,
                color=colors[direction],
                hatch=hatches[split],
                edgecolor="#1f1f1f",
                linewidth=0.8,
                label=f"{split} {direction}" if split == "train" else None,
            )
            axes[0].errorbar(
                xpos,
                ys,
                yerr=np.vstack([yerr_lo, yerr_hi]),
                fmt="none",
                ecolor="#1f1f1f",
                elinewidth=0.9,
                capsize=2.5,
            )
    axes[0].set_xticks(x, circuit_order)
    axes[0].set_ylim(0.0, 1.05)
    axes[0].set_ylabel("Top-1 recovery rate")
    axes[0].set_title("Bidirectional Sufficiency")
    legend_items = [
        plt.Rectangle((0, 0), 1, 1, facecolor=colors["tool_call"], edgecolor="#1f1f1f", label="tool-call"),
        plt.Rectangle((0, 0), 1, 1, facecolor=colors["no_tool"], edgecolor="#1f1f1f", label="no-tool"),
        plt.Rectangle((0, 0), 1, 1, facecolor="#ffffff", hatch="", edgecolor="#1f1f1f", label="train"),
        plt.Rectangle((0, 0), 1, 1, facecolor="#ffffff", hatch="//", edgecolor="#1f1f1f", label="test"),
    ]
    axes[0].legend(handles=legend_items, frameon=False, ncol=2, loc="upper left")

    # Panel B: suppression
    suppress_order = ["L16H4", "MLP17", "L23H6"]
    suppress_x = np.arange(len(suppress_order))
    suppress_map = {(str(r["split"]), str(r["circuit"]), str(r["stage"])): r for r in suppression_rows}
    suppress_colors = {"unified": "#c44e52", "forward_only": "#4c72b0"}
    for split, linestyle in (("train", "-"), ("test", "--")):
        for circuit in ("unified", "forward_only"):
            ys = []
            xs = []
            for idx, stage in enumerate(suppress_order):
                row = suppress_map.get((split, circuit, stage))
                if row is None:
                    continue
                xs.append(idx)
                ys.append(float(row["no_tool_top1"]))
            axes[1].plot(
                xs,
                ys,
                marker="o",
                linestyle=linestyle,
                color=suppress_colors[circuit],
                linewidth=2.0,
                label=f"{split} {circuit}",
            )
    axes[1].set_xticks(suppress_x, suppress_order)
    axes[1].set_ylim(0.0, 1.0)
    axes[1].set_ylabel("No-tool top-1")
    axes[1].set_title("Suppression Stagewise")
    axes[1].legend(frameon=False, fontsize=9, loc="lower right")

    # Panel C: construction
    construction_order = ["MLP19", "L20H5", "L21H1", "L21H12", "L24H6", "MLP27"]
    construction_x = np.arange(len(construction_order))
    construction_map = {(str(r["split"]), str(r["circuit"]), str(r["stage"])): r for r in construction_rows}
    construction_colors = {"unified": "#3a923a", "forward_only": "#7a5195"}
    for split, linestyle in (("train", "-"), ("test", "--")):
        for circuit in ("unified", "forward_only"):
            ys = []
            xs = []
            for idx, stage in enumerate(construction_order):
                row = construction_map.get((split, circuit, stage))
                if row is None:
                    continue
                xs.append(idx)
                ys.append(float(row["tool_top1"]))
            axes[2].plot(
                xs,
                ys,
                marker="o",
                linestyle=linestyle,
                color=construction_colors[circuit],
                linewidth=2.0,
                label=f"{split} {circuit}",
            )
    axes[2].set_xticks(construction_x, construction_order, rotation=20, ha="right")
    axes[2].set_ylim(0.0, 1.0)
    axes[2].set_ylabel("<tool_call> top-1")
    axes[2].set_title("Construction Stagewise")
    axes[2].legend(frameon=False, fontsize=9, loc="lower right")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def build_summary_json(
    *,
    output_root: Path,
    circuit_variants: Dict[str, List[str]],
    suff_summary: Sequence[Dict[str, object]],
    construction_rows: Sequence[Dict[str, object]],
    suppression_rows: Sequence[Dict[str, object]],
    split_counts: Dict[str, Dict[str, int]],
    direction_counts: Dict[str, int],
) -> Dict[str, object]:
    suff_map = {(str(r["circuit"]), str(r["direction"]), str(r["split"])): r for r in suff_summary}
    construction_map = {(str(r["split"]), str(r["circuit"]), str(r["stage"])): r for r in construction_rows}
    suppression_map = {(str(r["split"]), str(r["circuit"]), str(r["stage"])): r for r in suppression_rows}

    highlights: Dict[str, object] = {}
    for split in ("train", "test"):
        highlights[f"{split}_forward_tool_call_top1"] = float(suff_map[("forward", "tool_call", split)]["top1_rate"])
        highlights[f"{split}_forward_no_tool_top1"] = float(suff_map[("forward", "no_tool", split)]["top1_rate"])
        highlights[f"{split}_reverse_tool_call_top1"] = float(suff_map[("reverse", "tool_call", split)]["top1_rate"])
        highlights[f"{split}_reverse_no_tool_top1"] = float(suff_map[("reverse", "no_tool", split)]["top1_rate"])
        highlights[f"{split}_unified_tool_call_top1"] = float(suff_map[("unified", "tool_call", split)]["top1_rate"])
        highlights[f"{split}_unified_no_tool_top1"] = float(suff_map[("unified", "no_tool", split)]["top1_rate"])
        highlights[f"{split}_suppression_forward_final_no_tool_top1"] = float(
            suppression_map[(split, "forward_only", "L23H6")]["no_tool_top1"]
        )
        highlights[f"{split}_suppression_unified_final_no_tool_top1"] = float(
            suppression_map[(split, "unified", "L23H6")]["no_tool_top1"]
        )
        highlights[f"{split}_construction_forward_final_tool_top1"] = float(
            construction_map[(split, "forward_only", "MLP27")]["tool_top1"]
        )
        highlights[f"{split}_construction_unified_final_tool_top1"] = float(
            construction_map[(split, "unified", "MLP27")]["tool_top1"]
        )
        highlights[f"{split}_forward_vs_unified_no_tool_gap_pp"] = 100.0 * (
            float(suff_map[("unified", "no_tool", split)]["top1_rate"])
            - float(suff_map[("forward", "no_tool", split)]["top1_rate"])
        )
        highlights[f"{split}_reverse_vs_unified_tool_gap_pp"] = 100.0 * (
            float(suff_map[("unified", "tool_call", split)]["top1_rate"])
            - float(suff_map[("reverse", "tool_call", split)]["top1_rate"])
        )
        highlights[f"{split}_suppression_missing_L16H4_gap_pp"] = 100.0 * (
            float(suppression_map[(split, "unified", "L23H6")]["no_tool_top1"])
            - float(suppression_map[(split, "forward_only", "L23H6")]["no_tool_top1"])
        )
        highlights[f"{split}_construction_missing_L20H5_gap_pp"] = 100.0 * (
            float(construction_map[(split, "unified", "MLP27")]["tool_top1"])
            - float(construction_map[(split, "forward_only", "MLP27")]["tool_top1"])
        )

    return {
        "task": "task4_bidirectional_vs_unidirectional_faithfulness",
        "circuits": {
            name: {"n_nodes": len(nodes), "nodes": nodes}
            for name, nodes in circuit_variants.items()
        },
        "split_counts": split_counts,
        "direction_counts": direction_counts,
        "artifacts": {
            "sufficiency_summary_csv": str(output_root / "bidirectional_sufficiency_summary.csv"),
            "sufficiency_per_sample_csv": str(output_root / "bidirectional_sufficiency_per_sample.csv"),
            "suppression_stagewise_csv": str(output_root / "suppression_stagewise_comparison.csv"),
            "construction_stagewise_csv": str(output_root / "construction_stagewise_comparison.csv"),
            "figure_png": str(output_root / "bidirectional_comparison.png"),
            "summary_json": str(output_root / "bidirectional_comparison_summary.json"),
        },
        "highlights": highlights,
    }


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    circuit_variants = load_circuit_variants(args)
    train_samples = load_dataset_samples(args.train_dataset_root.resolve())
    test_samples = load_dataset_samples(args.test_dataset_root.resolve())
    model, tokenizer = load_hooked_qwen3(args.model_path, device=args.device, dtype=torch.bfloat16)

    suppression_dirs, direction_counts = build_train_suppression_directions(
        model,
        train_samples,
        max_samples=args.max_samples,
    )

    train_suff, train_construction, train_suppression, train_counts = evaluate_split(
        split_name="train",
        samples=train_samples,
        model=model,
        tokenizer=tokenizer,
        circuit_variants=circuit_variants,
        suppression_dirs=suppression_dirs,
        max_samples=args.max_samples,
    )
    test_suff, test_construction, test_suppression, test_counts = evaluate_split(
        split_name="test",
        samples=test_samples,
        model=model,
        tokenizer=tokenizer,
        circuit_variants=circuit_variants,
        suppression_dirs=suppression_dirs,
        max_samples=args.max_samples,
    )

    suff_rows = train_suff + test_suff
    suff_summary = summarize_sufficiency(suff_rows, bootstrap=args.bootstrap, seed=args.seed)
    write_csv(suff_rows, output_root / "bidirectional_sufficiency_per_sample.csv")
    write_csv(suff_summary, output_root / "bidirectional_sufficiency_summary.csv")

    construction_rows = summarize_stage_rows(train_construction + test_construction, include_no_tool=False)
    construction_rows.extend(
        load_official_construction_rows(args.construction_train_summary, args.construction_test_summary)
    )
    construction_rows.sort(key=lambda r: (str(r["split"]), str(r["circuit"]), str(r["stage"])))
    write_csv(construction_rows, output_root / "construction_stagewise_comparison.csv")

    suppression_rows = summarize_stage_rows(train_suppression + test_suppression, include_no_tool=True)
    suppression_rows.extend(
        load_official_suppression_rows(args.suppression_train_summary, args.suppression_test_summary)
    )
    suppression_rows.sort(key=lambda r: (str(r["split"]), str(r["circuit"]), str(r["stage"])))
    write_csv(suppression_rows, output_root / "suppression_stagewise_comparison.csv")

    plot_all(
        suff_summary=suff_summary,
        construction_rows=construction_rows,
        suppression_rows=suppression_rows,
        out_path=output_root / "bidirectional_comparison.png",
    )

    summary = build_summary_json(
        output_root=output_root,
        circuit_variants=circuit_variants,
        suff_summary=suff_summary,
        construction_rows=construction_rows,
        suppression_rows=suppression_rows,
        split_counts={"train": train_counts, "test": test_counts},
        direction_counts=direction_counts,
    )
    (output_root / "bidirectional_comparison_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    if output_root.name == "bidirectional_comparison":
        final_fig = Path("./results/legacy_figures/figure_11_bidirectional_comparison.png")
        final_fig.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(output_root / "bidirectional_comparison.png", final_fig)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
