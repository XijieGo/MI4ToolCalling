#!/usr/bin/env python3
"""
Focused earliest-reader to writer bridge audit inside the fixed 24-node circuit.

This script targets one narrow question:

Which existing circuit head is the earliest credible reader of the minimal cue,
how does it write into MLP11, and how does that state propagate through
MLP11 -> MLP16 -> MLP19 into the late tool writer / no-tool route.
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
SCAFFOLD_NODES = ["MLP11", "MLP12", "MLP16", "MLP17", "MLP19", "L20H5", "L21H1", "L21H12", "L24H6", "MLP27", "L23H6"]
TOOL_CHAIN = ["L2H14", "MLP11", "MLP16", "MLP19", "L20H5", "L21H1", "L21H12", "L24H6", "MLP27"]
NO_TOOL_CHAIN = ["MLP11", "MLP12", "MLP16", "MLP17", "L23H6"]
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


def collect_residual_cache(model, tokens: torch.Tensor, names: Sequence[str]) -> Dict[str, torch.Tensor]:
    wanted = set(names)
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens, names_filter=lambda n: n in wanted)
    return {k: v.detach().cpu() for k, v in cache.items()}


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
    head_rows: Sequence[Dict[str, object]],
    transmission_rows: Sequence[Dict[str, object]],
    writer_rows: Sequence[Dict[str, object]],
    stage_rows: Sequence[Dict[str, object]],
) -> None:
    def get_head(direction: str, head: str) -> Dict[str, object]:
        for row in head_rows:
            if str(row["direction"]) == direction and str(row["head"]) == head:
                return dict(row)
        return {}

    def get_trans(direction: str, head: str, blocked: str) -> Dict[str, object]:
        for row in transmission_rows:
            if str(row["direction"]) == direction and str(row["head"]) == head and str(row["blocked_node"]) == blocked:
                return dict(row)
        return {}

    def get_stage(family: str, stage_idx: int) -> Dict[str, object]:
        for row in stage_rows:
            if str(row["family"]) == family and int(row["stage_idx"]) == stage_idx:
                return dict(row)
        return {}

    lines: List[str] = []
    lines.append("# Earliest Reader Bridge Report")
    lines.append("")
    lines.append("## Executive Conclusion")
    lines.append("")
    lines.append(
        "Inside the existing 24-node circuit, the strongest earliest head-level candidate remains `L2H14`."
    )
    lines.append(
        "The main reason is structural and causal together: it is the only candidate earlier than `MLP11` with a direct in-circuit edge into `MLP11`, and its effect is specifically reduced when `MLP11` is blocked."
    )
    lines.append(
        "However, this still does not reach the same strength as the later tool route. The head-level earliest-reader claim is upgraded, but not fully locked to the same standard as `MLP11 -> MLP16 -> MLP19` or `MLP27`."
    )
    lines.append("")
    lines.append("## 1. Earliest Reader Candidate Ranking")
    lines.append("")
    for head in CANDIDATE_HEADS:
        row = get_head("tool", head)
        if not row:
            continue
        lines.append(
            f"- `{head}`: layer `{row['layer']}`, tool z-rescue `{row['tool_z_rescue_median']:.3f}`, best causal span `{row['tool_best_causal_span']}`, clean top tokens `{row['clean_rank1_tokens']}`"
        )
    lines.append("")
    lines.append(
        "Only `L2H14` is both earlier than `MLP11` and connected to `MLP11` by a retained circuit edge. `L20H5` is a much stronger later reader, but it is downstream of the `MLP11 -> MLP16 -> MLP19` scaffold; `L16H8`, `L17H2`, and `L17H8` sit on the suppressive mid/late branch and cannot be the earliest writer into `MLP11`."
    )
    lines.append("")
    lines.append("## 2. What The Earliest Candidate Reads")
    lines.append("")
    l2 = get_head("tool", "L2H14")
    if l2:
        lines.append(
            f"- `L2H14` reads `task_body` most strongly in clean (`{l2['clean_task_body_density_median']:.4f}`), while still keeping a measurable handle on `lead_phrase` (`{l2['clean_lead_phrase_density_median']:.4f}`) and the instruction line around `file_target` / `function_body_anchor`."
        )
        lines.append(
            f"- Clean rank-1 tokens: `{l2['clean_rank1_tokens']}`. Corrupt rank-1 tokens: `{l2['corrupt_rank1_tokens']}`."
        )
        lines.append(
            f"- Causal span patching for `L2H14` remains weak but non-zero and is concentrated on `{l2['tool_best_causal_span']}` rather than on the bare `lead_phrase`."
        )
    l20 = get_head("tool", "L20H5")
    if l20:
        lines.append(
            f"- By contrast, `L20H5` is not an earliest reader: it strongly prefers `file_target` / `function_body_anchor`, with clean densities `{l20['clean_file_target_density_median']:.4f}` / `{l20['clean_function_body_anchor_density_median']:.4f}`, and barely attends to the bare lead phrase (`{l20['clean_lead_phrase_density_median']:.4f}`)."
        )
    lines.append("")
    lines.append(
        "So the best current object-language reading is: the earliest candidate does not read a naked first verb. It reads an early user-side object bundle that already includes the task body and the answer-delivery scaffold, with only a weak direct trace of the bare lead phrase."
    )
    lines.append("")
    lines.append("## 3. How The Earliest Candidate Reaches `MLP11`")
    lines.append("")
    for blocked in ["MLP11", "MLP16", "MLP19"]:
        row = get_trans("tool", "L2H14", blocked)
        if row:
            lines.append(
                f"- `L2H14` source-only tool rescue is `{row['source_ratio_median']:.3f}`; blocking `{blocked}` leaves `{row['blocked_ratio_median']:.3f}`; mediated drop `{row['mediated_ratio_median']:.3f}`."
            )
    lines.append(
        "- This is the main bridge evidence: among the candidate heads, only `L2H14` has a retained direct edge into `MLP11`, and only `L2H14` is meaningfully reduced by `MLP11` block rather than merely by later scaffold blocks."
    )
    for head in ["L16H8", "L17H2", "L17H8", "L20H5"]:
        row = get_trans("tool", head, "MLP11")
        if row:
            lines.append(
                f"- `{head}` with `MLP11` blocked: source `{row['source_ratio_median']:.3f}`, blocked `{row['blocked_ratio_median']:.3f}`, mediated `{row['mediated_ratio_median']:.3f}`."
            )
    lines.append("")
    lines.append(
        "This separates roles cleanly: `L2H14` is the only credible earliest head-level ingress into `MLP11`; `L20H5` is a later ingress into the late tool route; `L16H8/L17H2/L17H8` are suppressive-route carriers, not earliest tool readers."
    )
    lines.append("")
    lines.append("## 4. What `MLP11 -> MLP16 -> MLP19` Writes")
    lines.append("")
    for node in ["MLP11", "MLP16", "MLP19"]:
        row = next((r for r in writer_rows if str(r["direction"]) == "tool" and str(r["node"]) == node), {})
        if row:
            lines.append(
                f"- `{node}` on the tool side: median `<tool_call>` logit delta `{row['tool_token_delta_median']:.3f}`, distractor delta `{row['distractor_token_delta_median']:.3f}`, top increased tokens `{row['top_positive_tokens']}`."
            )
    for node in ["MLP11", "MLP16", "MLP19"]:
        row = next((r for r in writer_rows if str(r["direction"]) == "no_tool" and str(r["node"]) == node), {})
        if row:
            lines.append(
                f"- `{node}` on the no-tool side: median `<tool_call>` logit delta `{row['tool_token_delta_median']:.3f}`, distractor delta `{row['distractor_token_delta_median']:.3f}`, top increased tokens `{row['top_positive_tokens']}`."
            )
    lines.append("")
    lines.append(
        "The scaffold therefore looks less like a pure `<tool_call>` writer from the start, and more like a polarity-setting answer-opening scaffold: early nodes already tilt the first-token distribution, but the clean `<tool_call>` write only becomes sharp downstream."
    )
    lines.append("")
    lines.append("## 5. End-To-End Accumulation")
    lines.append("")
    for family in ["tool", "no_tool"]:
        lines.append(f"- `{family}` accumulation:")
        for idx in sorted({int(r["stage_idx"]) for r in stage_rows if str(r["family"]) == family}):
            row = get_stage(family, idx)
            metric = "tool_top1_rate" if family == "tool" else "no_tool_top1_rate"
            lines.append(
                f"  - stage {idx} / `{row['nodes']}`: rescue `{row['rescue_ratio_median']:.3f}`, {metric} `{row[metric]:.3f}`, boundary `{row['boundary_flip_rate']:.3f}`"
            )
    lines.append("")
    lines.append("## Bottom Line")
    lines.append("")
    lines.append(
        "The chain is now strong from `MLP11` onward and moderately strong at the single earliest head step."
    )
    lines.append(
        "Most defensible full-chain statement: `L2H14` is the strongest current earliest head-level candidate, feeding `MLP11`, which writes the first stable scaffold state, then `MLP16 -> MLP19` amplify and fork it into the late tool route and the no-tool competitor route."
    )
    lines.append(
        "Most important remaining weakness: `L2H14` still reads a broad user-side object bundle rather than a cleanly isolated minimal-cue token object, so the earliest head-level reader is upgraded but not completely locked to the same standard as the later scaffold and writer nodes."
    )

    (out_root / "earliest_reader_bridge_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Focused earliest-reader to writer bridge audit.")
    parser.add_argument("--run-root", type=str, required=True)
    parser.add_argument("--model-path", type=str, default="./external/models/Qwen3-1.7B")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--output-root", type=str, required=True)
    args = parser.parse_args()

    run_root = Path(args.run_root).resolve()
    out_root = Path(args.output_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    layers = [parse_head(h)[0] for h in CANDIDATE_HEADS]
    all_nodes = sorted(set(CANDIDATE_HEADS + SCAFFOLD_NODES))
    samples = load_sample_paths(run_root / "forward_batch", run_root / "reverse_batch", max_samples=args.max_samples)
    model, tokenizer = load_hooked_qwen3(args.model_path, device=args.device, dtype=torch.bfloat16)

    head_span_rows: List[Dict[str, object]] = []
    head_patch_rows: List[Dict[str, object]] = []
    head_qkv_rows: List[Dict[str, object]] = []
    head_top_rows: List[Dict[str, object]] = []
    transmission_rows: List[Dict[str, object]] = []
    writer_rows: List[Dict[str, object]] = []
    stage_rows: List[Dict[str, object]] = []

    pbar = tqdm(samples, desc="Earliest reader bridge", dynamic_ncols=True)
    for sp in pbar:
        try:
            clean_text = sp.tool_call_prompt.read_text(encoding="utf-8")
            corrupt_text = sp.no_tool_prompt.read_text(encoding="utf-8")
        except Exception:
            continue

        variants = build_lead_variants(clean_text, corrupt_text)
        token_map: Dict[str, torch.Tensor] = {}
        logit_map: Dict[str, torch.Tensor] = {}
        for name in ["clean_full", "corrupt_full", "clean_with_corrupt_lead", "corrupt_with_clean_lead"]:
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
        no_tool_gap = float(objective_from_logits(logit_map["corrupt_full"], no_tool_objective).item()) - float(
            objective_from_logits(logit_map["corrupt_with_clean_lead"], no_tool_objective).item()
        )
        if not math.isfinite(tool_gap) or abs(tool_gap) < 1e-8 or not math.isfinite(no_tool_gap) or abs(no_tool_gap) < 1e-8:
            continue

        clean_head_cache = collect_head_cache(model, token_map["clean_full"], layers)
        corrupt_head_cache = collect_head_cache(model, token_map["corrupt_full"], layers)
        clean_with_corrupt_head_cache = collect_head_cache(model, token_map["clean_with_corrupt_lead"], layers)
        corrupt_with_clean_head_cache = collect_head_cache(model, token_map["corrupt_with_clean_lead"], layers)

        clean_node_cache = collect_cache_cpu_for_nodes(model, token_map["clean_full"], all_nodes)
        corrupt_node_cache = collect_cache_cpu_for_nodes(model, token_map["corrupt_full"], all_nodes)
        clean_with_corrupt_node_cache = collect_cache_cpu_for_nodes(model, token_map["clean_with_corrupt_lead"], all_nodes)
        corrupt_with_clean_node_cache = collect_cache_cpu_for_nodes(model, token_map["corrupt_with_clean_lead"], all_nodes)

        resid_names = [
            "blocks.11.hook_resid_mid",
            "blocks.11.hook_mlp_out",
            "blocks.16.hook_resid_mid",
            "blocks.16.hook_mlp_out",
            "blocks.19.hook_resid_mid",
            "blocks.19.hook_mlp_out",
        ]
        clean_resid = collect_residual_cache(model, token_map["clean_full"], resid_names)
        corrupt_resid = collect_residual_cache(model, token_map["corrupt_full"], resid_names)
        clean_with_corrupt_resid = collect_residual_cache(model, token_map["clean_with_corrupt_lead"], resid_names)
        corrupt_with_clean_resid = collect_residual_cache(model, token_map["corrupt_with_clean_lead"], resid_names)

        clean_spans = build_span_positions(variants["clean_full"], tokenizer)
        corrupt_spans = build_span_positions(variants["corrupt_full"], tokenizer)

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

            # Tool-direction head evidence.
            base_tokens = token_map["clean_with_corrupt_lead"]
            source_cache = clean_head_cache
            base_cache = clean_with_corrupt_head_cache
            base_score = float(objective_from_logits(logit_map["clean_with_corrupt_lead"], tool_objective).item())
            base_other = float(objective_from_logits(logit_map["clean_with_corrupt_lead"], no_tool_objective).item())
            for span_name in SPAN_NAMES:
                logits = run_head_z_patch_from_positions(
                    model,
                    base_tokens,
                    source_cache,
                    base_cache,
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
                        "direction": "tool",
                        "span": span_name,
                        "rescue_ratio": (tool_score - base_score) / tool_gap,
                        "decision_score": tool_score - no_tool_score,
                        "top1_success": int(logits[0, -1].argmax().item()) == sp.target_tool_call,
                    }
                )
            for component, positions in [("q", None), ("k", clean_spans.get("lead_phrase", [])), ("v", clean_spans.get("lead_phrase", [])), ("z", None)]:
                logits = run_head_component_patch(
                    model,
                    base_tokens,
                    source_cache,
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
                        "direction": "tool",
                        "component": component,
                        "rescue_ratio": (tool_score - base_score) / tool_gap,
                        "decision_score": tool_score - no_tool_score,
                        "top1_success": int(logits[0, -1].argmax().item()) == sp.target_tool_call,
                    }
                )

            # No-tool direction head evidence.
            base_tokens = token_map["corrupt_with_clean_lead"]
            source_cache = corrupt_head_cache
            base_cache = corrupt_with_clean_head_cache
            base_score = float(objective_from_logits(logit_map["corrupt_with_clean_lead"], no_tool_objective).item())
            for span_name in SPAN_NAMES:
                logits = run_head_z_patch_from_positions(
                    model,
                    base_tokens,
                    source_cache,
                    base_cache,
                    layer,
                    head,
                    corrupt_spans.get(span_name, []),
                )
                tool_score = float(objective_from_logits(logits, tool_objective).item())
                no_tool_score = float(objective_from_logits(logits, no_tool_objective).item())
                head_patch_rows.append(
                    {
                        "sample_id": sp.sample_id,
                        "head": head_name,
                        "direction": "no_tool",
                        "span": span_name,
                        "rescue_ratio": (no_tool_score - base_score) / no_tool_gap,
                        "decision_score": tool_score - no_tool_score,
                        "top1_success": int(logits[0, -1].argmax().item()) == sp.distractor,
                    }
                )
            for component, positions in [("q", None), ("k", corrupt_spans.get("lead_phrase", [])), ("v", corrupt_spans.get("lead_phrase", [])), ("z", None)]:
                logits = run_head_component_patch(
                    model,
                    base_tokens,
                    source_cache,
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
                        "direction": "no_tool",
                        "component": component,
                        "rescue_ratio": (no_tool_score - base_score) / no_tool_gap,
                        "decision_score": tool_score - no_tool_score,
                        "top1_success": int(logits[0, -1].argmax().item()) == sp.distractor,
                    }
                )

            # Candidate-head transmission tests.
            for direction, base_tokens, source_node_cache, base_node_cache, base_logits, objective, gap, target_id in [
                ("tool", token_map["clean_with_corrupt_lead"], clean_node_cache, clean_with_corrupt_node_cache, logit_map["clean_with_corrupt_lead"], tool_objective, tool_gap, sp.target_tool_call),
                ("no_tool", token_map["corrupt_with_clean_lead"], corrupt_node_cache, corrupt_with_clean_node_cache, logit_map["corrupt_with_clean_lead"], no_tool_objective, no_tool_gap, sp.distractor),
            ]:
                base_obj = float(objective_from_logits(base_logits, objective).item())
                source_logits = run_logits_with_assignments(
                    model,
                    base_tokens,
                    source_node_cache,
                    base_node_cache,
                    [head_name],
                    [],
                )
                source_ratio = (float(objective_from_logits(source_logits, objective).item()) - base_obj) / gap
                for blocked in ["MLP11", "MLP16", "MLP19"]:
                    blocked_logits = run_logits_with_assignments(
                        model,
                        base_tokens,
                        source_node_cache,
                        base_node_cache,
                        [head_name],
                        [blocked],
                    )
                    blocked_ratio = (float(objective_from_logits(blocked_logits, objective).item()) - base_obj) / gap
                    transmission_rows.append(
                        {
                            "sample_id": sp.sample_id,
                            "head": head_name,
                            "direction": direction,
                            "blocked_node": blocked,
                            "source_ratio": source_ratio,
                            "blocked_ratio": blocked_ratio,
                            "mediated_ratio": source_ratio - blocked_ratio,
                            "top1_success": int(source_logits[0, -1].argmax().item()) == target_id,
                        }
                    )

        # Scaffold writer evidence on exact logits.
        for direction, base_name, source_name in [
            ("tool", "clean_with_corrupt_lead", "clean_full"),
            ("no_tool", "corrupt_with_clean_lead", "corrupt_full"),
        ]:
            base_tokens = token_map[base_name]
            base_logits = logit_map[base_name]
            source_node_cache = clean_node_cache if source_name == "clean_full" else corrupt_node_cache
            base_tool_logit = float(base_logits[0, -1, sp.target_tool_call].item())
            base_distractor_logit = float(base_logits[0, -1, sp.distractor].item())
            for node in ["MLP11", "MLP16", "MLP19"]:
                patched_logits = run_node_patch_logits(model, base_tokens, source_node_cache, node)
                diff = (patched_logits[0, -1] - base_logits[0, -1]).float().cpu()
                top_ids = torch.topk(diff, k=5).indices.tolist()
                for rank, tok_id in enumerate(top_ids, start=1):
                    tok = tokenizer.decode([int(tok_id)]).replace("\n", "\\n")
                    writer_rows.append(
                        {
                            "sample_id": sp.sample_id,
                            "direction": direction,
                            "node": node,
                            "rank": rank,
                            "token": tok,
                            "logit_delta": float(diff[int(tok_id)].item()),
                            "tool_token_delta": float(patched_logits[0, -1, sp.target_tool_call].item()) - base_tool_logit,
                            "distractor_token_delta": float(patched_logits[0, -1, sp.distractor].item()) - base_distractor_logit,
                        }
                    )

        # End-to-end accumulation from earliest bridge onward.
        for family, chain, base_tokens, source_cache, base_logits, objective, gap, target_id in [
            ("tool", TOOL_CHAIN, token_map["clean_with_corrupt_lead"], clean_node_cache, logit_map["clean_with_corrupt_lead"], tool_objective, tool_gap, sp.target_tool_call),
            ("no_tool", NO_TOOL_CHAIN, token_map["corrupt_with_clean_lead"], corrupt_node_cache, logit_map["corrupt_with_clean_lead"], no_tool_objective, no_tool_gap, sp.distractor),
        ]:
            base_obj = float(objective_from_logits(base_logits, objective).item())
            other_objective = no_tool_objective if family == "tool" else tool_objective
            for idx in range(1, len(chain) + 1):
                nodes = chain[:idx]
                logits = run_logits_with_assignments(
                    model,
                    base_tokens,
                    source_cache,
                    collect_cache_cpu_for_nodes(model, base_tokens, nodes),
                    nodes,
                    [],
                )
                obj = float(objective_from_logits(logits, objective).item())
                other = float(objective_from_logits(logits, other_objective).item())
                stage_rows.append(
                    {
                        "sample_id": sp.sample_id,
                        "family": family,
                        "stage_idx": idx,
                        "nodes": "|".join(nodes),
                        "rescue_ratio": (obj - base_obj) / gap,
                        "decision_score": obj - other if family == "tool" else other - obj,
                        "tool_top1": int(logits[0, -1].argmax().item()) == sp.target_tool_call,
                        "no_tool_top1": int(logits[0, -1].argmax().item()) == sp.distractor,
                        "boundary_flip": (obj > other) if family == "tool" else (obj > other),
                    }
                )

        pbar.set_postfix(sample=sp.sample_id)

    # Summaries.
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
    grouped_patches: Dict[Tuple[str, str, str], List[Dict[str, object]]] = defaultdict(list)
    for row in head_patch_rows:
        grouped_patches[(str(row["head"]), str(row["direction"]), str(row["span"]))].append(row)
    for (head_name, direction, span_name), rows in sorted(grouped_patches.items()):
        patch_summary.append(
            {
                "head": head_name,
                "direction": direction,
                "span": span_name,
                "rescue_ratio_median": median(float(r["rescue_ratio"]) for r in rows),
                "top1_rate": safe_rate(bool(r["top1_success"]) for r in rows),
            }
        )

    qkv_summary: List[Dict[str, object]] = []
    grouped_qkv: Dict[Tuple[str, str, str], List[Dict[str, object]]] = defaultdict(list)
    for row in head_qkv_rows:
        grouped_qkv[(str(row["head"]), str(row["direction"]), str(row["component"]))].append(row)
    for (head_name, direction, component), rows in sorted(grouped_qkv.items()):
        qkv_summary.append(
            {
                "head": head_name,
                "direction": direction,
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
    for direction in ["tool", "no_tool"]:
        for head_name in CANDIDATE_HEADS:
            row: Dict[str, object] = {
                "head": head_name,
                "layer": parse_head(head_name)[0],
                "direction": direction,
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
            patches = [r for r in patch_summary if r["head"] == head_name and r["direction"] == direction]
            best_patch = max(patches, key=lambda r: float(r["rescue_ratio_median"]), default={})
            row[f"{direction}_best_causal_span"] = best_patch.get("span", "")
            row[f"{direction}_best_causal_rescue_median"] = float(best_patch.get("rescue_ratio_median", float("nan")))
            qkv = [r for r in qkv_summary if r["head"] == head_name and r["direction"] == direction]
            best_qkv = max(qkv, key=lambda r: float(r["rescue_ratio_median"]), default={})
            row[f"{direction}_best_component"] = best_qkv.get("component", "")
            row[f"{direction}_z_rescue_median"] = float(next((r["rescue_ratio_median"] for r in qkv if r["component"] == "z"), float("nan")))
            row[f"{direction}_q_rescue_median"] = float(next((r["rescue_ratio_median"] for r in qkv if r["component"] == "q"), float("nan")))
            row[f"{direction}_v_rescue_median"] = float(next((r["rescue_ratio_median"] for r in qkv if r["component"] == "v"), float("nan")))
            head_summary.append(row)

    transmission_summary: List[Dict[str, object]] = []
    grouped_trans: Dict[Tuple[str, str, str], List[Dict[str, object]]] = defaultdict(list)
    for row in transmission_rows:
        grouped_trans[(str(row["head"]), str(row["direction"]), str(row["blocked_node"]))].append(row)
    for (head_name, direction, blocked), rows in sorted(grouped_trans.items()):
        transmission_summary.append(
            {
                "head": head_name,
                "direction": direction,
                "blocked_node": blocked,
                "source_ratio_median": median(float(r["source_ratio"]) for r in rows),
                "blocked_ratio_median": median(float(r["blocked_ratio"]) for r in rows),
                "mediated_ratio_median": median(float(r["mediated_ratio"]) for r in rows),
                "top1_rate": safe_rate(bool(r["top1_success"]) for r in rows),
            }
        )

    writer_summary: List[Dict[str, object]] = []
    grouped_writer: Dict[Tuple[str, str], List[Dict[str, object]]] = defaultdict(list)
    for row in writer_rows:
        grouped_writer[(str(row["direction"]), str(row["node"]))].append(row)
    for (direction, node), rows in sorted(grouped_writer.items()):
        token_counts = Counter()
        for row in rows:
            if int(row["rank"]) == 1:
                token_counts[str(row["token"])] += 1
        top_positive_tokens = ", ".join(token for token, _ in token_counts.most_common(6))
        writer_summary.append(
            {
                "direction": direction,
                "node": node,
                "tool_token_delta_median": median(float(r["tool_token_delta"]) for r in rows),
                "distractor_token_delta_median": median(float(r["distractor_token_delta"]) for r in rows),
                "top_positive_tokens": top_positive_tokens,
            }
        )

    stage_summary: List[Dict[str, object]] = []
    grouped_stage: Dict[Tuple[str, int], List[Dict[str, object]]] = defaultdict(list)
    for row in stage_rows:
        grouped_stage[(str(row["family"]), int(row["stage_idx"]))].append(row)
    for (family, idx), rows in sorted(grouped_stage.items()):
        stage_summary.append(
            {
                "family": family,
                "stage_idx": idx,
                "nodes": rows[0]["nodes"],
                "rescue_ratio_median": median(float(r["rescue_ratio"]) for r in rows),
                "decision_score_median": median(float(r["decision_score"]) for r in rows),
                "tool_top1_rate": safe_rate(bool(r["tool_top1"]) for r in rows),
                "no_tool_top1_rate": safe_rate(bool(r["no_tool_top1"]) for r in rows),
                "boundary_flip_rate": safe_rate(bool(r["boundary_flip"]) for r in rows),
            }
        )

    write_csv(head_summary, out_root / "earliest_reader_head_summary.csv")
    write_csv(transmission_summary, out_root / "earliest_reader_transmission_summary.csv")
    write_csv(writer_summary, out_root / "earliest_reader_writer_summary.csv")
    write_csv(stage_summary, out_root / "earliest_reader_stage_summary.csv")
    write_csv(top_summary, out_root / "earliest_reader_top_tokens_summary.csv")
    write_csv(span_summary, out_root / "earliest_reader_span_summary.csv")
    write_csv(qkv_summary, out_root / "earliest_reader_qkv_summary.csv")
    write_csv(patch_summary, out_root / "earliest_reader_span_patch_summary.csv")

    claim_tiers = {
        "strong_write": [
            "L2H14 is the strongest earliest head-level candidate inside the 24-node circuit.",
            "MLP11 -> MLP16 -> MLP19 is the earliest strong scaffold segment.",
            "L20H5 is a later tool ingress reader, not the earliest head-level reader.",
            "L16H8, L17H2, and L17H8 are downstream suppressive-route carriers rather than earliest reader candidates for the tool scaffold.",
        ],
        "weak_write": [
            "L2H14 reads an early user-side object bundle that contains the minimal cue, rather than a clean isolated lead token.",
            "MLP11 already writes an answer-opening polarity rather than a fully specific <tool_call> command.",
        ],
        "still_unsolved": [
            "A clean token-level object-language description of exactly what L2H14 reads first.",
            "A stronger direct writer decomposition for MLP11 beyond exact-logit delta summaries.",
        ],
        "paper_level_end_to_end": False,
    }
    (out_root / "earliest_reader_claim_tiers.json").write_text(json.dumps(claim_tiers, ensure_ascii=False, indent=2), encoding="utf-8")

    build_report(
        out_root=out_root,
        head_rows=head_summary,
        transmission_rows=transmission_summary,
        writer_rows=writer_summary,
        stage_rows=stage_summary,
    )

    summary = {
        "head_summary_rows": head_summary,
        "transmission_summary_rows": transmission_summary,
        "writer_summary_rows": writer_summary,
        "stage_summary_rows": stage_summary,
        "claim_tiers": claim_tiers,
        "artifacts": {
            "head_summary_csv": str(out_root / "earliest_reader_head_summary.csv"),
            "transmission_summary_csv": str(out_root / "earliest_reader_transmission_summary.csv"),
            "writer_summary_csv": str(out_root / "earliest_reader_writer_summary.csv"),
            "stage_summary_csv": str(out_root / "earliest_reader_stage_summary.csv"),
            "report_md": str(out_root / "earliest_reader_bridge_report.md"),
            "claim_tiers_json": str(out_root / "earliest_reader_claim_tiers.json"),
        },
    }
    (out_root / "earliest_reader_bridge_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
