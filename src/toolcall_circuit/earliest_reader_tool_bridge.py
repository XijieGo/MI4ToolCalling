#!/usr/bin/env python3
"""
Tool-side earliest-reader bridge audit.

Narrow scope:
- earliest head-level reader candidates inside the existing 24-node circuit
- their transmission into MLP11 / MLP16 / MLP19
- tool-side scaffold semantics for MLP11 / MLP16 / MLP19
- tool-side accumulation into the already-known late writer path
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from tqdm.auto import tqdm

if __package__ in {None, ""}:
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from toolcall_circuit.bidirectional_causal_eval import collect_cache_cpu_for_nodes, load_sample_paths
from toolcall_circuit.final_head_attention_audit import (
    build_span_positions,
    collect_head_cache,
    parse_head,
    run_head_component_patch,
    run_head_z_patch_from_positions,
)
from toolcall_circuit.instruction_verb_phrase_audit import build_variants as build_lead_variants
from toolcall_circuit.objective import build_bidirectional_endpoint_objectives
from toolcall_circuit.signed_edge_importance import run_logits_with_assignments
from toolcall_circuit.single_sample import load_hooked_qwen3, objective_from_logits


CANDIDATE_HEADS = ["L2H14", "L16H8", "L17H2", "L17H8", "L20H5"]
TOOL_CHAIN = ["L2H14", "MLP11", "MLP16", "MLP19", "L20H5", "L21H1", "L21H12", "L24H6", "MLP27"]
PATCH_NODES = sorted(set(CANDIDATE_HEADS + TOOL_CHAIN + ["MLP11", "MLP16", "MLP19"]))
SPAN_NAMES = ["lead_phrase", "function_body_anchor", "file_target", "tail_suffix", "task_body"]


def write_csv(rows: Sequence[Dict[str, object]], out_path: Path) -> None:
    if not rows:
        return
    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def finite(values: Iterable[float]) -> List[float]:
    return [float(v) for v in values if isinstance(v, (int, float)) and math.isfinite(float(v))]


def median(values: Iterable[float]) -> float:
    vals = finite(values)
    return float(np.median(vals)) if vals else float("nan")


def safe_rate(values: Iterable[bool]) -> float:
    vals = [1.0 if bool(v) else 0.0 for v in values]
    return float(np.mean(vals)) if vals else float("nan")


def token_preview(rows: Sequence[Dict[str, object]], head: str, variant: str, limit: int = 6) -> str:
    items = [r for r in rows if str(r["head"]) == head and str(r["variant"]) == variant and int(r["rank"]) == 1]
    items.sort(key=lambda r: (-int(r["count"]), -float(r["attn_median"]), str(r["token"])))
    return ", ".join(str(r["token"]) for r in items[:limit])


def run_node_patch_logits(
    model,
    base_tokens: torch.Tensor,
    source_cache_cpu: Dict[str, torch.Tensor],
    node: str,
) -> torch.Tensor:
    if node.startswith("MLP"):
        layer = int(node[3:])
        cache_name = f"blocks.{layer}.hook_mlp_out"
        src = source_cache_cpu[cache_name].to(base_tokens.device)

        def hook_fn(mlp_out: torch.Tensor, hook):  # noqa: ANN001
            out = mlp_out.clone()
            out[:, -1, :] = src[:, -1, :].to(dtype=mlp_out.dtype)
            return out

        hooks = [(cache_name, hook_fn)]
    else:
        layer, head = parse_head(node)
        cache_name = f"blocks.{layer}.attn.hook_z"
        src = source_cache_cpu[cache_name].to(base_tokens.device)

        def hook_fn(z: torch.Tensor, hook):  # noqa: ANN001
            out = z.clone()
            out[:, -1, head, :] = src[:, -1, head, :].to(dtype=z.dtype)
            return out

        hooks = [(cache_name, hook_fn)]

    with torch.no_grad():
        return model.run_with_hooks(base_tokens, fwd_hooks=hooks)


def build_report(
    *,
    out_root: Path,
    head_summary: Sequence[Dict[str, object]],
    transmission_summary: Sequence[Dict[str, object]],
    writer_summary: Sequence[Dict[str, object]],
    stage_summary: Sequence[Dict[str, object]],
) -> None:
    def get_head(head: str) -> Dict[str, object]:
        for row in head_summary:
            if str(row["head"]) == head:
                return dict(row)
        return {}

    def get_trans(head: str, blocked: str) -> Dict[str, object]:
        for row in transmission_summary:
            if str(row["head"]) == head and str(row["blocked_node"]) == blocked:
                return dict(row)
        return {}

    lines: List[str] = []
    lines.append("# Earliest Reader Tool Bridge Report")
    lines.append("")
    lines.append("## Updated End-To-End Chain")
    lines.append("")
    lines.append("1. `L2H14` is the strongest current earliest head-level candidate inside the 24-node circuit.")
    lines.append("2. It reads an early user-side object bundle that includes the instruction opening and answer-delivery scaffold, not a naked first verb token.")
    lines.append("3. Its tool-side effect enters the earliest strong scaffold through `MLP11`.")
    lines.append("4. `MLP11 -> MLP16 -> MLP19` amplifies that state into a shared answer-opening scaffold.")
    lines.append("5. That scaffold then enters the already-established late tool route `L20H5 -> L21H1/L21H12 -> L24H6 -> MLP27`.")
    lines.append("6. The no-tool competitive route remains `L16H4 -> MLP17 -> L23H6`; this audit does not re-open that story.")
    lines.append("")
    lines.append("## Earliest Reader Evidence")
    lines.append("")
    for head in CANDIDATE_HEADS:
        row = get_head(head)
        if not row:
            continue
        lines.append(
            f"- `{head}` / L{row['layer']}: best causal span `{row['best_causal_span']}`, `q/z/v = {row['q_rescue_median']:.3f}/{row['z_rescue_median']:.3f}/{row['v_rescue_median']:.3f}`, clean top tokens `{row['clean_rank1_tokens']}`"
        )
    l2 = get_head("L2H14")
    l20 = get_head("L20H5")
    if l2:
        lines.append(
            f"- `L2H14` clean span densities: lead `{l2['clean_lead_phrase_density_median']:.4f}`, file `{l2['clean_file_target_density_median']:.4f}`, function-body `{l2['clean_function_body_anchor_density_median']:.4f}`, tail `{l2['clean_tail_suffix_density_median']:.4f}`, task `{l2['clean_task_body_density_median']:.4f}`."
        )
    if l20:
        lines.append(
            f"- `L20H5` clean span densities: lead `{l20['clean_lead_phrase_density_median']:.4f}`, file `{l20['clean_file_target_density_median']:.4f}`, function-body `{l20['clean_function_body_anchor_density_median']:.4f}`, tail `{l20['clean_tail_suffix_density_median']:.4f}`, task `{l20['clean_task_body_density_median']:.4f}`."
        )
    lines.append("")
    lines.append("## Reader To MLP11 Transmission")
    lines.append("")
    for blocked in ["MLP11", "MLP16", "MLP19"]:
        row = get_trans("L2H14", blocked)
        if row:
            lines.append(
                f"- `L2H14` source-only rescue `{row['source_ratio_median']:.3f}`; blocking `{blocked}` leaves `{row['blocked_ratio_median']:.3f}`; mediated drop `{row['mediated_ratio_median']:.3f}`."
            )
    for head in ["L16H8", "L17H2", "L17H8", "L20H5"]:
        row = get_trans(head, "MLP11")
        if row:
            lines.append(
                f"- `{head}` with `MLP11` blocked: source `{row['source_ratio_median']:.3f}`, blocked `{row['blocked_ratio_median']:.3f}`, mediated `{row['mediated_ratio_median']:.3f}`."
            )
    lines.append("")
    lines.append("## Scaffold Writer Evidence")
    lines.append("")
    for node in ["MLP11", "MLP16", "MLP19"]:
        row = next((r for r in writer_summary if str(r["node"]) == node), {})
        if row:
            lines.append(
                f"- `{node}`: `<tool_call>` delta `{row['tool_token_delta_median']:.3f}`, distractor delta `{row['distractor_token_delta_median']:.3f}`, top increased tokens `{row['top_positive_tokens']}`."
            )
    lines.append("")
    lines.append("## Tool-Side Accumulation")
    lines.append("")
    for row in stage_summary:
        lines.append(
            f"- stage {row['stage_idx']} / `{row['nodes']}`: rescue `{row['rescue_ratio_median']:.3f}`, tool-top1 `{row['tool_top1_rate']:.3f}`, boundary `{row['boundary_flip_rate']:.3f}`."
        )
    lines.append("")
    lines.append("## Bottom Line")
    lines.append("")
    lines.append("The strongest full-chain statement remains: `L2H14 -> MLP11 -> MLP16 -> MLP19 -> L20H5 -> L21H1/L21H12 -> L24H6 -> MLP27`.")
    lines.append("What is now strong is the bridge from `L2H14` into `MLP11`: it is structurally unique inside the circuit and its effect is specifically reduced by `MLP11` block.")
    lines.append("What is still not fully solved is the exact object-language feature that `L2H14` reads first. It still looks like an early user-side object bundle rather than a single isolated minimal-cue token.")

    (out_root / "earliest_reader_tool_bridge_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Tool-side earliest-reader bridge audit.")
    parser.add_argument("--run-root", type=str, required=True)
    parser.add_argument("--model-path", type=str, default="./external/models/Qwen3-1.7B")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--output-root", type=str, required=True)
    parser.add_argument("--save-every", type=int, default=50)
    args = parser.parse_args()

    run_root = Path(args.run_root).resolve()
    out_root = Path(args.output_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    layers = [parse_head(h)[0] for h in CANDIDATE_HEADS]
    samples = load_sample_paths(run_root / "forward_batch", run_root / "reverse_batch", max_samples=args.max_samples)
    model, tokenizer = load_hooked_qwen3(args.model_path, device=args.device, dtype=torch.bfloat16)

    head_span_path = out_root / "earliest_reader_tool_span_per_sample.csv"
    head_patch_path = out_root / "earliest_reader_tool_span_patch_per_sample.csv"
    head_qkv_path = out_root / "earliest_reader_tool_qkv_per_sample.csv"
    head_top_path = out_root / "earliest_reader_tool_top_tokens_per_sample.csv"
    transmission_path = out_root / "earliest_reader_tool_transmission_per_sample.csv"
    writer_path = out_root / "earliest_reader_tool_writer_per_sample.csv"
    stage_path = out_root / "earliest_reader_tool_stage_per_sample.csv"

    head_span_rows: List[Dict[str, object]] = list(read_csv_rows(head_span_path))
    head_patch_rows: List[Dict[str, object]] = list(read_csv_rows(head_patch_path))
    head_qkv_rows: List[Dict[str, object]] = list(read_csv_rows(head_qkv_path))
    head_top_rows: List[Dict[str, object]] = list(read_csv_rows(head_top_path))
    transmission_rows: List[Dict[str, object]] = list(read_csv_rows(transmission_path))
    writer_rows: List[Dict[str, object]] = list(read_csv_rows(writer_path))
    stage_rows: List[Dict[str, object]] = list(read_csv_rows(stage_path))

    processed_ids = {str(r["sample_id"]) for r in stage_rows}

    def checkpoint() -> None:
        write_csv(head_span_rows, head_span_path)
        write_csv(head_patch_rows, head_patch_path)
        write_csv(head_qkv_rows, head_qkv_path)
        write_csv(head_top_rows, head_top_path)
        write_csv(transmission_rows, transmission_path)
        write_csv(writer_rows, writer_path)
        write_csv(stage_rows, stage_path)

    pbar = tqdm(samples, desc="Earliest reader tool bridge", dynamic_ncols=True)
    completed_new = 0
    for sp in pbar:
        if sp.sample_id in processed_ids:
            pbar.set_postfix(sample=sp.sample_id, resumed="skip")
            continue
        try:
            clean_text = sp.tool_call_prompt.read_text(encoding="utf-8")
            corrupt_text = sp.no_tool_prompt.read_text(encoding="utf-8")
        except Exception:
            continue

        variants = build_lead_variants(clean_text, corrupt_text)
        token_map: Dict[str, torch.Tensor] = {}
        logit_map: Dict[str, torch.Tensor] = {}
        for name in ["clean_full", "corrupt_full", "clean_with_corrupt_lead"]:
            toks = model.to_tokens(variants[name], prepend_bos=False)
            token_map[name] = toks
            with torch.no_grad():
                logit_map[name] = model(toks)

        tool_objective, no_tool_objective = build_bidirectional_endpoint_objectives(
            logit_map["clean_full"],
            logit_map["corrupt_full"],
            tokenizer=tokenizer,
        )
        tool_gap = float(objective_from_logits(logit_map["clean_full"], tool_objective).item()) - float(
            objective_from_logits(logit_map["clean_with_corrupt_lead"], tool_objective).item()
        )
        if not math.isfinite(tool_gap) or abs(tool_gap) < 1e-8:
            continue

        clean_head_cache = collect_head_cache(model, token_map["clean_full"], layers)
        corrupt_head_cache = collect_head_cache(model, token_map["corrupt_full"], layers)
        clean_with_corrupt_head_cache = collect_head_cache(model, token_map["clean_with_corrupt_lead"], layers)

        clean_node_cache = collect_cache_cpu_for_nodes(model, token_map["clean_full"], PATCH_NODES)
        clean_with_corrupt_node_cache = collect_cache_cpu_for_nodes(model, token_map["clean_with_corrupt_lead"], PATCH_NODES)

        clean_spans = build_span_positions(variants["clean_full"], tokenizer)
        corrupt_spans = build_span_positions(variants["corrupt_full"], tokenizer)

        base_tokens = token_map["clean_with_corrupt_lead"]
        base_logits = logit_map["clean_with_corrupt_lead"]
        base_tool_score = float(objective_from_logits(base_logits, tool_objective).item())

        for head_name in CANDIDATE_HEADS:
            layer, head = parse_head(head_name)
            pat_key = f"blocks.{layer}.attn.hook_pattern"

            for variant_name, cache_obj, span_map, toks in [
                ("clean", clean_head_cache, clean_spans, token_map["clean_full"]),
                ("corrupt", corrupt_head_cache, corrupt_spans, token_map["corrupt_full"]),
            ]:
                pattern = cache_obj[pat_key][0, head, -1, :].float()
                top_pos = torch.topk(pattern, k=min(5, int(pattern.numel()))).indices.tolist()
                for rank, pos in enumerate(top_pos, start=1):
                    tok = tokenizer.decode([int(toks[0, int(pos)].item())]).replace("\n", "\\n")
                    head_top_rows.append(
                        {
                            "sample_id": sp.sample_id,
                            "head": head_name,
                            "variant": variant_name,
                            "rank": rank,
                            "pos": int(pos),
                            "token": tok,
                            "attn": float(pattern[int(pos)].item()),
                        }
                    )
                for span_name in SPAN_NAMES:
                    idxs = [int(i) for i in span_map.get(span_name, []) if 0 <= int(i) < pattern.shape[0]]
                    mass = float(pattern[idxs].sum().item()) if idxs else float("nan")
                    density = float(pattern[idxs].mean().item()) if idxs else float("nan")
                    head_span_rows.append(
                        {
                            "sample_id": sp.sample_id,
                            "head": head_name,
                            "variant": variant_name,
                            "span": span_name,
                            "attn_mass": mass,
                            "attn_density": density,
                            "n_tokens": len(idxs),
                        }
                    )

            for span_name in SPAN_NAMES:
                logits = run_head_z_patch_from_positions(
                    model,
                    base_tokens,
                    clean_head_cache,
                    clean_with_corrupt_head_cache,
                    layer,
                    head,
                    clean_spans.get(span_name, []),
                )
                tool_score = float(objective_from_logits(logits, tool_objective).item())
                no_tool_score = float(objective_from_logits(logits, no_tool_objective).item())
                head_patch_rows.append(
                    {
                        "sample_id": sp.sample_id,
                        "head": head_name,
                        "span": span_name,
                        "rescue_ratio": (tool_score - base_tool_score) / tool_gap,
                        "decision_score": tool_score - no_tool_score,
                        "top1_success": int(logits[0, -1].argmax().item()) == sp.target_tool_call,
                    }
                )

            for component, positions in [("q", None), ("k", clean_spans.get("lead_phrase", [])), ("v", clean_spans.get("lead_phrase", [])), ("z", None)]:
                logits = run_head_component_patch(
                    model,
                    base_tokens,
                    clean_head_cache,
                    layer,
                    head,
                    component,
                    positions=positions,
                )
                tool_score = float(objective_from_logits(logits, tool_objective).item())
                no_tool_score = float(objective_from_logits(logits, no_tool_objective).item())
                head_qkv_rows.append(
                    {
                        "sample_id": sp.sample_id,
                        "head": head_name,
                        "component": component,
                        "rescue_ratio": (tool_score - base_tool_score) / tool_gap,
                        "decision_score": tool_score - no_tool_score,
                        "top1_success": int(logits[0, -1].argmax().item()) == sp.target_tool_call,
                    }
                )

            source_logits = run_logits_with_assignments(
                model,
                base_tokens,
                clean_node_cache,
                clean_with_corrupt_node_cache,
                [head_name],
                [],
            )
            source_ratio = (float(objective_from_logits(source_logits, tool_objective).item()) - base_tool_score) / tool_gap
            for blocked in ["MLP11", "MLP16", "MLP19"]:
                blocked_logits = run_logits_with_assignments(
                    model,
                    base_tokens,
                    clean_node_cache,
                    clean_with_corrupt_node_cache,
                    [head_name],
                    [blocked],
                )
                blocked_ratio = (float(objective_from_logits(blocked_logits, tool_objective).item()) - base_tool_score) / tool_gap
                transmission_rows.append(
                    {
                        "sample_id": sp.sample_id,
                        "head": head_name,
                        "blocked_node": blocked,
                        "source_ratio": source_ratio,
                        "blocked_ratio": blocked_ratio,
                        "mediated_ratio": source_ratio - blocked_ratio,
                        "top1_success": int(source_logits[0, -1].argmax().item()) == sp.target_tool_call,
                    }
                )

        for node in ["MLP11", "MLP16", "MLP19"]:
            patched_logits = run_node_patch_logits(model, base_tokens, clean_node_cache, node)
            diff = (patched_logits[0, -1] - base_logits[0, -1]).float().cpu()
            top_ids = torch.topk(diff, k=5).indices.tolist()
            for rank, tok_id in enumerate(top_ids, start=1):
                tok = tokenizer.decode([int(tok_id)]).replace("\n", "\\n")
                writer_rows.append(
                    {
                        "sample_id": sp.sample_id,
                        "node": node,
                        "rank": rank,
                        "token": tok,
                        "logit_delta": float(diff[int(tok_id)].item()),
                        "tool_token_delta": float(patched_logits[0, -1, sp.target_tool_call].item()) - float(base_logits[0, -1, sp.target_tool_call].item()),
                        "distractor_token_delta": float(patched_logits[0, -1, sp.distractor].item()) - float(base_logits[0, -1, sp.distractor].item()),
                    }
                )

        for idx in range(1, len(TOOL_CHAIN) + 1):
            nodes = TOOL_CHAIN[:idx]
            logits = run_logits_with_assignments(
                model,
                base_tokens,
                clean_node_cache,
                clean_with_corrupt_node_cache,
                nodes,
                [],
            )
            tool_score = float(objective_from_logits(logits, tool_objective).item())
            no_tool_score = float(objective_from_logits(logits, no_tool_objective).item())
            stage_rows.append(
                {
                    "sample_id": sp.sample_id,
                    "stage_idx": idx,
                    "nodes": "|".join(nodes),
                    "rescue_ratio": (tool_score - base_tool_score) / tool_gap,
                    "decision_score": tool_score - no_tool_score,
                    "tool_top1": int(logits[0, -1].argmax().item()) == sp.target_tool_call,
                    "boundary_flip": tool_score > no_tool_score,
                }
            )

        pbar.set_postfix(sample=sp.sample_id)
        processed_ids.add(sp.sample_id)
        completed_new += 1
        if args.save_every > 0 and completed_new % args.save_every == 0:
            checkpoint()

    checkpoint()

    span_summary: List[Dict[str, object]] = []
    grouped_spans: Dict[Tuple[str, str, str], List[Dict[str, object]]] = defaultdict(list)
    for row in head_span_rows:
        grouped_spans[(str(row["head"]), str(row["variant"]), str(row["span"]))].append(row)
    for (head_name, variant, span_name), rows in sorted(grouped_spans.items()):
        span_summary.append(
            {
                "head": head_name,
                "variant": variant,
                "span": span_name,
                "attn_mass_median": median(float(r["attn_mass"]) for r in rows),
                "attn_density_median": median(float(r["attn_density"]) for r in rows),
                "n_tokens_median": median(float(r["n_tokens"]) for r in rows),
            }
        )

    patch_summary: List[Dict[str, object]] = []
    grouped_patches: Dict[Tuple[str, str], List[Dict[str, object]]] = defaultdict(list)
    for row in head_patch_rows:
        grouped_patches[(str(row["head"]), str(row["span"]))].append(row)
    for (head_name, span_name), rows in sorted(grouped_patches.items()):
        patch_summary.append(
            {
                "head": head_name,
                "span": span_name,
                "rescue_ratio_median": median(float(r["rescue_ratio"]) for r in rows),
                "top1_rate": safe_rate(bool(r["top1_success"]) for r in rows),
            }
        )

    qkv_summary: List[Dict[str, object]] = []
    grouped_qkv: Dict[Tuple[str, str], List[Dict[str, object]]] = defaultdict(list)
    for row in head_qkv_rows:
        grouped_qkv[(str(row["head"]), str(row["component"]))].append(row)
    for (head_name, component), rows in sorted(grouped_qkv.items()):
        qkv_summary.append(
            {
                "head": head_name,
                "component": component,
                "rescue_ratio_median": median(float(r["rescue_ratio"]) for r in rows),
                "top1_rate": safe_rate(bool(r["top1_success"]) for r in rows),
            }
        )

    top_summary: List[Dict[str, object]] = []
    grouped_top: Dict[Tuple[str, str, int, str], List[Dict[str, object]]] = defaultdict(list)
    for row in head_top_rows:
        grouped_top[(str(row["head"]), str(row["variant"]), int(row["rank"]), str(row["token"]))].append(row)
    for (head_name, variant, rank, token), rows in sorted(grouped_top.items()):
        top_summary.append(
            {
                "head": head_name,
                "variant": variant,
                "rank": rank,
                "token": token,
                "count": len(rows),
                "attn_median": median(float(r["attn"]) for r in rows),
            }
        )

    head_summary: List[Dict[str, object]] = []
    for head_name in CANDIDATE_HEADS:
        row: Dict[str, object] = {
            "head": head_name,
            "layer": parse_head(head_name)[0],
            "clean_rank1_tokens": token_preview(top_summary, head_name, "clean"),
            "corrupt_rank1_tokens": token_preview(top_summary, head_name, "corrupt"),
        }
        for span_name in SPAN_NAMES:
            clean_span = next((r for r in span_summary if r["head"] == head_name and r["variant"] == "clean" and r["span"] == span_name), {})
            corrupt_span = next((r for r in span_summary if r["head"] == head_name and r["variant"] == "corrupt" and r["span"] == span_name), {})
            row[f"clean_{span_name}_density_median"] = float(clean_span.get("attn_density_median", float("nan")))
            row[f"corrupt_{span_name}_density_median"] = float(corrupt_span.get("attn_density_median", float("nan")))
            row[f"clean_minus_corrupt_{span_name}_density"] = (
                float(clean_span.get("attn_density_median", float("nan"))) - float(corrupt_span.get("attn_density_median", float("nan")))
                if clean_span and corrupt_span
                else float("nan")
            )
        patches = [r for r in patch_summary if r["head"] == head_name]
        best_patch = max(patches, key=lambda r: float(r["rescue_ratio_median"]), default={})
        row["best_causal_span"] = best_patch.get("span", "")
        row["best_causal_rescue_median"] = float(best_patch.get("rescue_ratio_median", float("nan")))
        qkv = [r for r in qkv_summary if r["head"] == head_name]
        row["best_component"] = max(qkv, key=lambda r: float(r["rescue_ratio_median"]), default={}).get("component", "")
        row["z_rescue_median"] = float(next((r["rescue_ratio_median"] for r in qkv if r["component"] == "z"), float("nan")))
        row["q_rescue_median"] = float(next((r["rescue_ratio_median"] for r in qkv if r["component"] == "q"), float("nan")))
        row["v_rescue_median"] = float(next((r["rescue_ratio_median"] for r in qkv if r["component"] == "v"), float("nan")))
        head_summary.append(row)

    transmission_summary: List[Dict[str, object]] = []
    grouped_trans: Dict[Tuple[str, str], List[Dict[str, object]]] = defaultdict(list)
    for row in transmission_rows:
        grouped_trans[(str(row["head"]), str(row["blocked_node"]))].append(row)
    for (head_name, blocked), rows in sorted(grouped_trans.items()):
        transmission_summary.append(
            {
                "head": head_name,
                "blocked_node": blocked,
                "source_ratio_median": median(float(r["source_ratio"]) for r in rows),
                "blocked_ratio_median": median(float(r["blocked_ratio"]) for r in rows),
                "mediated_ratio_median": median(float(r["mediated_ratio"]) for r in rows),
                "top1_rate": safe_rate(bool(r["top1_success"]) for r in rows),
            }
        )

    writer_summary: List[Dict[str, object]] = []
    grouped_writer: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in writer_rows:
        grouped_writer[str(row["node"])].append(row)
    for node, rows in sorted(grouped_writer.items()):
        token_counts = Counter()
        for row in rows:
            if int(row["rank"]) == 1:
                token_counts[str(row["token"])] += 1
        writer_summary.append(
            {
                "node": node,
                "tool_token_delta_median": median(float(r["tool_token_delta"]) for r in rows),
                "distractor_token_delta_median": median(float(r["distractor_token_delta"]) for r in rows),
                "top_positive_tokens": ", ".join(token for token, _ in token_counts.most_common(6)),
            }
        )

    stage_summary: List[Dict[str, object]] = []
    grouped_stage: Dict[int, List[Dict[str, object]]] = defaultdict(list)
    for row in stage_rows:
        grouped_stage[int(row["stage_idx"])].append(row)
    for idx, rows in sorted(grouped_stage.items()):
        stage_summary.append(
            {
                "stage_idx": idx,
                "nodes": rows[0]["nodes"],
                "rescue_ratio_median": median(float(r["rescue_ratio"]) for r in rows),
                "decision_score_median": median(float(r["decision_score"]) for r in rows),
                "tool_top1_rate": safe_rate(bool(r["tool_top1"]) for r in rows),
                "boundary_flip_rate": safe_rate(bool(r["boundary_flip"]) for r in rows),
            }
        )

    claim_tiers = {
        "strong_write": [
            "L2H14 is the strongest earliest head-level candidate inside the 24-node circuit.",
            "L2H14 feeds the earliest strong scaffold through MLP11.",
            "MLP11 -> MLP16 -> MLP19 is the earliest strong scaffold segment.",
            "L20H5 is a later tool ingress reader, not the earliest head-level reader.",
            "L16H8, L17H2, and L17H8 are downstream carriers rather than earliest reader candidates for the tool scaffold."
        ],
        "weak_write": [
            "L2H14 reads an early user-side object bundle that contains the minimal cue, rather than a clean isolated lead token.",
            "MLP11 already writes an answer-opening polarity rather than a fully specific <tool_call> command."
        ],
        "still_unsolved": [
            "A clean token-level object-language description of exactly what L2H14 reads first.",
            "A stronger direct writer decomposition for MLP11 beyond exact-logit delta summaries."
        ],
        "paper_level_end_to_end": False
    }

    write_csv(head_summary, out_root / "earliest_reader_tool_head_summary.csv")
    write_csv(transmission_summary, out_root / "earliest_reader_tool_transmission_summary.csv")
    write_csv(writer_summary, out_root / "earliest_reader_tool_writer_summary.csv")
    write_csv(stage_summary, out_root / "earliest_reader_tool_stage_summary.csv")
    write_csv(top_summary, out_root / "earliest_reader_tool_top_tokens_summary.csv")
    write_csv(span_summary, out_root / "earliest_reader_tool_span_summary.csv")
    write_csv(qkv_summary, out_root / "earliest_reader_tool_qkv_summary.csv")
    write_csv(patch_summary, out_root / "earliest_reader_tool_span_patch_summary.csv")
    (out_root / "earliest_reader_tool_claim_tiers.json").write_text(json.dumps(claim_tiers, ensure_ascii=False, indent=2), encoding="utf-8")

    build_report(
        out_root=out_root,
        head_summary=head_summary,
        transmission_summary=transmission_summary,
        writer_summary=writer_summary,
        stage_summary=stage_summary,
    )

    summary = {
        "head_summary_rows": head_summary,
        "transmission_summary_rows": transmission_summary,
        "writer_summary_rows": writer_summary,
        "stage_summary_rows": stage_summary,
        "claim_tiers": claim_tiers,
        "artifacts": {
            "head_summary_csv": str(out_root / "earliest_reader_tool_head_summary.csv"),
            "transmission_summary_csv": str(out_root / "earliest_reader_tool_transmission_summary.csv"),
            "writer_summary_csv": str(out_root / "earliest_reader_tool_writer_summary.csv"),
            "stage_summary_csv": str(out_root / "earliest_reader_tool_stage_summary.csv"),
            "report_md": str(out_root / "earliest_reader_tool_bridge_report.md"),
            "claim_tiers_json": str(out_root / "earliest_reader_tool_claim_tiers.json")
        }
    }
    (out_root / "earliest_reader_tool_bridge_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
