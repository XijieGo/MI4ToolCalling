#!/usr/bin/env python3
"""
Token-level behavioral evaluation for bidirectional circuit groups.

Unlike the margin-only causal summaries, this script measures whether a patch
actually flips the *predicted first token* to the source-side target:

- promotion: does patching tool-call-side activations make `<tool_call>` top-1?
- suppression: does patching no-tool-side activations make the no-tool target top-1?

It also evaluates a cleaner branch decomposition that removes shared backbone
nodes from the selective groups.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from tqdm.auto import tqdm

if __package__ in {None, ""}:
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from toolcall_circuit.bidirectional_causal_eval import (
    collect_cache_cpu_for_nodes,
    filter_valid_nodes,
    load_sample_paths,
)
from toolcall_circuit.objective import build_bidirectional_endpoint_objectives
from toolcall_circuit.single_sample import load_hooked_qwen3, objective_from_logits, parse_head


def finite(values: Iterable[float]) -> List[float]:
    return [float(v) for v in values if isinstance(v, (int, float)) and math.isfinite(float(v))]


def median(values: Iterable[float]) -> float:
    vals = finite(values)
    return float(np.median(vals)) if vals else float("nan")


def mean(values: Iterable[float]) -> float:
    vals = finite(values)
    return float(np.mean(vals)) if vals else float("nan")


def safe_rate(values: Iterable[bool]) -> float:
    vals = [1.0 if bool(v) else 0.0 for v in values]
    return float(np.mean(vals)) if vals else float("nan")


def write_csv(rows: Sequence[Dict[str, object]], out_path: Path) -> None:
    if not rows:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_logits_on_base_with_source(
    model,
    base_tokens: torch.Tensor,
    source_cache_cpu: Dict[str, torch.Tensor],
    patch_nodes: Sequence[str],
) -> torch.Tensor:
    heads_by_layer: Dict[int, List[int]] = {}
    mlp_layers: List[int] = []

    for node in patch_nodes:
        if node.startswith("MLP"):
            mlp_layers.append(int(node[3:]))
        else:
            layer, head = parse_head(node)
            heads_by_layer.setdefault(layer, []).append(head)

    hooks = []
    for layer, heads in heads_by_layer.items():
        cache_name = f"blocks.{layer}.attn.hook_z"
        clean_act = source_cache_cpu[cache_name].to(base_tokens.device)
        heads = sorted(set(heads))

        def make_head_hook(src: torch.Tensor, hs: Sequence[int]):
            def hook_fn(z: torch.Tensor, hook):  # noqa: ANN001
                out = z.clone()
                for h in hs:
                    out[:, -1, h, :] = src[:, -1, h, :]
                return out

            return hook_fn

        hooks.append((cache_name, make_head_hook(clean_act, heads)))

    for layer in sorted(set(mlp_layers)):
        cache_name = f"blocks.{layer}.hook_mlp_out"
        clean_act = source_cache_cpu[cache_name].to(base_tokens.device)

        def make_mlp_hook(src: torch.Tensor):
            def hook_fn(mlp_out: torch.Tensor, hook):  # noqa: ANN001
                out = mlp_out.clone()
                out[:, -1, :] = src[:, -1, :]
                return out

            return hook_fn

        hooks.append((cache_name, make_mlp_hook(clean_act)))

    with torch.no_grad():
        return model.run_with_hooks(base_tokens, fwd_hooks=hooks)


def main() -> None:
    parser = argparse.ArgumentParser(description="Token-level behavioral evaluation for bidirectional groups.")
    parser.add_argument("--forward-batch-root", type=str, required=True)
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

    forward_batch_root = Path(args.forward_batch_root).resolve()
    reverse_batch_root = Path(args.bidirectional_summary).resolve().parent.parent / "reverse_batch"
    if not reverse_batch_root.exists():
        raise ValueError(f"Missing reverse batch root: {reverse_batch_root}")

    fwd_agg = json.loads(Path(args.forward_aggregate_summary).resolve().read_text(encoding="utf-8"))
    rev_agg = json.loads(Path(args.reverse_aggregate_summary).resolve().read_text(encoding="utf-8"))
    bidi = json.loads(Path(args.bidirectional_summary).resolve().read_text(encoding="utf-8"))

    forward_core = set(filter_valid_nodes(list(fwd_agg.get("core_nodes", []))))
    reverse_core = set(filter_valid_nodes(list(rev_agg.get("core_nodes", []))))
    forward_only_core = sorted(forward_core - reverse_core)
    reverse_only_core = sorted(reverse_core - forward_core)

    support = bidi.get("support_analysis", {})
    shared_backbone = set(filter_valid_nodes(list(support.get("shared_backbone_nodes", []))))
    forward_selective = set(filter_valid_nodes(list(support.get("forward_selective_nodes", []))))
    reverse_selective = set(filter_valid_nodes(list(support.get("reverse_selective_nodes", []))))

    groups: Dict[str, List[str]] = {
        "shared_backbone": sorted(shared_backbone),
        "forward_selective": sorted(forward_selective),
        "reverse_selective": sorted(reverse_selective),
        "forward_selective_unique": sorted(forward_selective - shared_backbone),
        "reverse_selective_unique": sorted(reverse_selective - shared_backbone),
        "forward_branch_unique": sorted(set(forward_only_core) | (forward_selective - shared_backbone)),
        "reverse_branch_unique": sorted(set(reverse_only_core) | (reverse_selective - shared_backbone)),
        "shared_backbone_exclusive": sorted(shared_backbone - forward_selective - reverse_selective),
    }
    groups = {k: v for k, v in groups.items() if v}

    promote_groups = list(groups.keys())
    suppress_groups = list(groups.keys())
    nodes_tool_source = sorted({n for g in promote_groups for n in groups[g]})
    nodes_no_tool_source = sorted({n for g in suppress_groups for n in groups[g]})

    samples = load_sample_paths(forward_batch_root, reverse_batch_root, max_samples=args.max_samples)
    model, tokenizer = load_hooked_qwen3(args.model_path, device=args.device, dtype=torch.bfloat16)

    per_sample_rows: List[Dict[str, object]] = []
    pbar = tqdm(samples, desc="Token flip eval", dynamic_ncols=True)
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
        m_tool_tool = float(objective_from_logits(tool_logits, tool_objective).item())
        m_tool_no = float(objective_from_logits(no_tool_logits, tool_objective).item())
        gap_tool = m_tool_tool - m_tool_no
        m_no_tool_tool = float(objective_from_logits(tool_logits, no_tool_objective).item())
        m_no_tool_no = float(objective_from_logits(no_tool_logits, no_tool_objective).item())
        gap_no_tool = m_no_tool_no - m_no_tool_tool
        if (
            not math.isfinite(gap_tool)
            or abs(gap_tool) < 1e-8
            or not math.isfinite(gap_no_tool)
            or abs(gap_no_tool) < 1e-8
        ):
            continue

        tool_cache = collect_cache_cpu_for_nodes(model, tool_tokens, nodes_tool_source)
        no_tool_cache = collect_cache_cpu_for_nodes(model, no_tool_tokens, nodes_no_tool_source)

        base_tool_top1 = int(tool_logits[0, -1].argmax().item())
        base_no_tool_top1 = int(no_tool_logits[0, -1].argmax().item())

        row: Dict[str, object] = {
            "sample_id": sp.sample_id,
            "target_tool_call_id": sp.target_tool_call,
            "target_tool_call_str": tokenizer.decode([sp.target_tool_call]),
            "target_no_tool_id": sp.distractor,
            "target_no_tool_str": tokenizer.decode([sp.distractor]),
            "tool_endpoint_tool_score": m_tool_tool,
            "tool_endpoint_no_tool_score": m_tool_no,
            "tool_endpoint_gap": gap_tool,
            "no_tool_endpoint_tool_score": m_no_tool_tool,
            "no_tool_endpoint_no_tool_score": m_no_tool_no,
            "no_tool_endpoint_gap": gap_no_tool,
            "tool_base_top1_id": base_tool_top1,
            "tool_base_top1_str": tokenizer.decode([base_tool_top1]),
            "no_tool_base_top1_id": base_no_tool_top1,
            "no_tool_base_top1_str": tokenizer.decode([base_no_tool_top1]),
        }

        for group_name, nodes in groups.items():
            promote_logits = run_logits_on_base_with_source(model, no_tool_tokens, tool_cache, nodes)
            promote_tool_score = float(objective_from_logits(promote_logits, tool_objective).item())
            promote_no_tool_score = float(objective_from_logits(promote_logits, no_tool_objective).item())
            promote_top1 = int(promote_logits[0, -1].argmax().item())
            row[f"promote__{group_name}__ratio"] = (promote_tool_score - m_tool_no) / gap_tool
            row[f"promote__{group_name}__tool_score"] = promote_tool_score
            row[f"promote__{group_name}__no_tool_score"] = promote_no_tool_score
            row[f"promote__{group_name}__top1_id"] = promote_top1
            row[f"promote__{group_name}__top1_str"] = tokenizer.decode([promote_top1])
            row[f"promote__{group_name}__top1_is_tool"] = promote_top1 == sp.target_tool_call
            row[f"promote__{group_name}__boundary_flip"] = promote_tool_score > promote_no_tool_score

            suppress_logits = run_logits_on_base_with_source(model, tool_tokens, no_tool_cache, nodes)
            suppress_tool_score = float(objective_from_logits(suppress_logits, tool_objective).item())
            suppress_no_tool_score = float(objective_from_logits(suppress_logits, no_tool_objective).item())
            suppress_top1 = int(suppress_logits[0, -1].argmax().item())
            row[f"suppress__{group_name}__ratio"] = (suppress_no_tool_score - m_no_tool_tool) / gap_no_tool
            row[f"suppress__{group_name}__tool_score"] = suppress_tool_score
            row[f"suppress__{group_name}__no_tool_score"] = suppress_no_tool_score
            row[f"suppress__{group_name}__top1_id"] = suppress_top1
            row[f"suppress__{group_name}__top1_str"] = tokenizer.decode([suppress_top1])
            row[f"suppress__{group_name}__top1_is_no_tool"] = suppress_top1 == sp.distractor
            row[f"suppress__{group_name}__boundary_flip"] = suppress_no_tool_score > suppress_tool_score

        per_sample_rows.append(row)
        pbar.set_postfix(sample=sp.sample_id)

    summary_rows: List[Dict[str, object]] = []
    for group_name in groups:
        promote_rows = per_sample_rows
        suppress_rows = per_sample_rows
        summary_rows.append(
            {
                "group": group_name,
                "n_nodes": len(groups[group_name]),
                "nodes": ",".join(groups[group_name]),
                "promote_ratio_median": median(r[f"promote__{group_name}__ratio"] for r in promote_rows),
                "promote_boundary_flip_rate": safe_rate(r[f"promote__{group_name}__boundary_flip"] for r in promote_rows),
                "promote_tool_top1_rate": safe_rate(r[f"promote__{group_name}__top1_is_tool"] for r in promote_rows),
                "suppress_ratio_median": median(r[f"suppress__{group_name}__ratio"] for r in suppress_rows),
                "suppress_boundary_flip_rate": safe_rate(r[f"suppress__{group_name}__boundary_flip"] for r in suppress_rows),
                "suppress_no_tool_top1_rate": safe_rate(r[f"suppress__{group_name}__top1_is_no_tool"] for r in suppress_rows),
            }
        )

    summary_rows.sort(
        key=lambda r: (
            float(r["promote_tool_top1_rate"]) + float(r["suppress_no_tool_top1_rate"]),
            float(r["promote_boundary_flip_rate"]) + float(r["suppress_boundary_flip_rate"]),
        ),
        reverse=True,
    )

    summary = {
        "n_samples": len(per_sample_rows),
        "groups": groups,
        "artifacts": {
            "per_sample_json": str(out_root / "per_sample_token_flip.json"),
            "summary_csv": str(out_root / "group_token_flip_summary.csv"),
            "summary_json": str(out_root / "group_token_flip_summary.json"),
        },
        "summary_rows": summary_rows,
    }

    (out_root / "per_sample_token_flip.json").write_text(
        json.dumps(per_sample_rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_csv(summary_rows, out_root / "group_token_flip_summary.csv")
    (out_root / "group_token_flip_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
