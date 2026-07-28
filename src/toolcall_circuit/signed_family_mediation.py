#!/usr/bin/env python3
"""
Group-to-group signed family mediation analysis.

For a family (source_group -> target_group, sign), we measure whether source-group
influence on the decision margin is mediated through the target-group nodes.
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


def mean(values: Iterable[float]) -> float:
    vals = finite(values)
    return float(np.mean(vals)) if vals else float("nan")


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Signed family mediation analysis.")
    parser.add_argument("--forward-batch-root", type=str, required=True)
    parser.add_argument("--bidirectional-summary", type=str, required=True)
    parser.add_argument("--signed-family-summary-csv", type=str, required=True)
    parser.add_argument("--model-path", type=str, default="./external/models/Qwen3-1.7B")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max-samples", type=int, default=200)
    parser.add_argument("--min-family-edges", type=int, default=2)
    parser.add_argument("--min-support", type=float, default=0.25)
    parser.add_argument("--output-root", type=str, required=True)
    args = parser.parse_args()

    out_root = Path(args.output_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    bidi = json.loads(Path(args.bidirectional_summary).resolve().read_text(encoding="utf-8"))
    groups = bidi["support_analysis"]
    # Build the same group mapping as the final signed circuit.
    shared = set(map(str, groups.get("shared_backbone_nodes", [])))
    forward = set(map(str, groups.get("forward_selective_nodes", [])))
    reverse = set(map(str, groups.get("reverse_selective_nodes", [])))
    group_nodes = {
        "symmetric_backbone": sorted(shared - forward - reverse),
        "tool_bias_backbone": sorted((shared & forward) - reverse),
        "no_tool_bias_backbone": sorted((shared & reverse) - forward),
        "tool_tail": sorted(forward - shared),
        "no_tool_tail": sorted(reverse - shared),
    }
    group_nodes = {k: v for k, v in group_nodes.items() if v}

    family_rows = read_csv_rows(Path(args.signed_family_summary_csv).resolve())
    selected = [
        r
        for r in family_rows
        if int(r["n_edges"]) >= args.min_family_edges and float(r["union_support_median"]) >= args.min_support
    ]
    selected.sort(key=lambda r: (float(r["union_support_median"]), int(r["n_edges"])), reverse=True)

    families = []
    for r in selected:
        sg = str(r["source_group"])
        tg = str(r["target_group"])
        families.append(
            {
                "family": f"{sg}->{tg} ({r['sign']})",
                "source_group": sg,
                "target_group": tg,
                "sign": str(r["sign"]),
                "source_nodes": group_nodes[sg],
                "target_nodes": group_nodes[tg],
            }
        )

    reverse_batch_root = Path(args.bidirectional_summary).resolve().parent.parent / "reverse_batch"
    samples = load_sample_paths(Path(args.forward_batch_root).resolve(), reverse_batch_root, max_samples=args.max_samples)

    all_nodes = sorted({n for fam in families for n in fam["source_nodes"] + fam["target_nodes"]})
    model, tokenizer = load_hooked_qwen3(args.model_path, device=args.device, dtype=torch.bfloat16)
    _ = tokenizer

    per_sample_rows: List[Dict[str, object]] = []
    pbar = tqdm(samples, desc="Signed family mediation", dynamic_ncols=True)
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

        tool_cache = collect_cache_cpu_for_nodes(model, tool_tokens, all_nodes)
        no_tool_cache = collect_cache_cpu_for_nodes(model, no_tool_tokens, all_nodes)

        for fam in families:
            src_nodes = list(fam["source_nodes"])
            tgt_nodes = list(fam["target_nodes"])

            promote_src_logits = run_logits_with_assignments(
                model, no_tool_tokens, tool_cache, no_tool_cache, src_nodes, []
            )
            promote_src_margin = float(objective_from_logits(promote_src_logits, tool_objective).item())
            promote_src_ratio = (promote_src_margin - m_tool_no) / gap_tool

            promote_blocked_logits = run_logits_with_assignments(
                model, no_tool_tokens, tool_cache, no_tool_cache, src_nodes, tgt_nodes
            )
            promote_blocked_margin = float(objective_from_logits(promote_blocked_logits, tool_objective).item())
            promote_blocked_ratio = (promote_blocked_margin - m_tool_no) / gap_tool

            suppress_src_logits = run_logits_with_assignments(
                model, tool_tokens, no_tool_cache, tool_cache, src_nodes, []
            )
            suppress_src_margin = float(objective_from_logits(suppress_src_logits, no_tool_objective).item())
            suppress_src_ratio = (suppress_src_margin - m_no_tool_tool) / gap_no_tool

            suppress_blocked_logits = run_logits_with_assignments(
                model, tool_tokens, no_tool_cache, tool_cache, src_nodes, tgt_nodes
            )
            suppress_blocked_margin = float(objective_from_logits(suppress_blocked_logits, no_tool_objective).item())
            suppress_blocked_ratio = (suppress_blocked_margin - m_no_tool_tool) / gap_no_tool

            per_sample_rows.append(
                {
                    "sample_id": sp.sample_id,
                    "family": fam["family"],
                    "promote_source_ratio": promote_src_ratio,
                    "promote_blocked_ratio": promote_blocked_ratio,
                    "promote_mediated_ratio": promote_src_ratio - promote_blocked_ratio,
                    "suppress_source_ratio": suppress_src_ratio,
                    "suppress_blocked_ratio": suppress_blocked_ratio,
                    "suppress_mediated_ratio": suppress_src_ratio - suppress_blocked_ratio,
                }
            )

        pbar.set_postfix(sample=sp.sample_id)

    by_family: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in per_sample_rows:
        by_family[str(row["family"])].append(row)

    summary_rows: List[Dict[str, object]] = []
    for fam in families:
        key = str(fam["family"])
        rows = by_family[key]
        summary_rows.append(
            {
                "family": key,
                "n_samples": len(rows),
                "promote_source_ratio_median": median(r["promote_source_ratio"] for r in rows),
                "promote_blocked_ratio_median": median(r["promote_blocked_ratio"] for r in rows),
                "promote_mediated_ratio_median": median(r["promote_mediated_ratio"] for r in rows),
                "suppress_source_ratio_median": median(r["suppress_source_ratio"] for r in rows),
                "suppress_blocked_ratio_median": median(r["suppress_blocked_ratio"] for r in rows),
                "suppress_mediated_ratio_median": median(r["suppress_mediated_ratio"] for r in rows),
            }
        )
    summary_rows.sort(key=lambda r: abs(float(r["promote_mediated_ratio_median"])) + abs(float(r["suppress_mediated_ratio_median"])), reverse=True)

    write_csv(per_sample_rows, out_root / "signed_family_mediation_per_sample.csv")
    write_csv(summary_rows, out_root / "signed_family_mediation_summary.csv")
    summary = {
        "n_samples": len(samples),
        "families": families,
        "artifacts": {
            "per_sample_csv": str(out_root / "signed_family_mediation_per_sample.csv"),
            "summary_csv": str(out_root / "signed_family_mediation_summary.csv"),
            "summary_json": str(out_root / "signed_family_mediation_report.json"),
        },
        "summary_rows": summary_rows,
    }
    (out_root / "signed_family_mediation_report.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
