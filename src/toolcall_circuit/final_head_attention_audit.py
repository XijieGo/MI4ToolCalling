#!/usr/bin/env python3
"""
Final attention-head audit aligned to the refined mechanism story.

For each key attention head, this script builds a stronger evidence package:

1. What span it reads:
   - lead phrase
   - file target token span
   - shared instruction suffix (excluding file target)
   - task body

2. Which source span causally matters:
   - patch only the head contribution from that span

3. Which attention component matters:
   - query patch
   - key patch on lead positions
   - value patch on lead positions
   - z patch
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from tqdm.auto import tqdm

if __package__ in {None, ""}:
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from toolcall_circuit.bidirectional_causal_eval import load_sample_paths
from toolcall_circuit.instruction_verb_phrase_audit import build_variants as build_lead_variants
from toolcall_circuit.objective import build_bidirectional_endpoint_objectives
from toolcall_circuit.single_sample import load_hooked_qwen3, objective_from_logits


TOOL_HEADS = ["L20H5", "L21H1", "L21H12", "L24H6"]
NO_TOOL_HEADS = ["L16H4", "L23H6"]
ALL_HEADS = TOOL_HEADS + NO_TOOL_HEADS


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


def parse_head(node: str) -> Tuple[int, int]:
    body = node[1:]
    layer_s, head_s = body.split("H")
    return int(layer_s), int(head_s)


def split_prompt_outer(text: str) -> Tuple[str, str, str]:
    user_marker = "<|im_start|>user\n"
    assistant_marker = "<|im_end|>\n<|im_start|>assistant\n"
    start = text.find(user_marker)
    if start < 0:
        raise ValueError("missing user marker")
    user_start = start + len(user_marker)
    end = text.find(assistant_marker, user_start)
    if end < 0:
        raise ValueError("missing assistant marker")
    return text[:user_start], text[user_start:end], text[end:]


def split_instruction_line(user_content: str) -> Tuple[str, str]:
    if "\n" not in user_content:
        return user_content, ""
    first, rest = user_content.split("\n", 1)
    return first, rest


def token_positions_for_char_span(text: str, start: int, end: int, tokenizer) -> List[int]:
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    positions: List[int] = []
    for idx, (tok_start, tok_end) in enumerate(enc["offset_mapping"]):
        if int(tok_start) < int(end) and int(tok_end) > int(start):
            positions.append(int(idx))
    return positions


def build_span_positions(text: str, tokenizer) -> Dict[str, List[int]]:
    outer_prefix, user_content, _outer_suffix = split_prompt_outer(text)
    line, rest = split_instruction_line(user_content)
    line_start = len(outer_prefix)
    line_end = line_start + len(line)
    task_start = line_end + (1 if rest else 0)
    task_end = task_start + len(rest)

    m = re.search(r"solve\.(?:py|cpp|java)", line)
    if not m:
        raise ValueError("failed to find solve file target in instruction line")
    file_start = line_start + m.start()
    file_end = line_start + m.end()

    lead_end = line.find("the function body")
    if lead_end < 0:
        lead_end = line.find("function body")
    if lead_end < 0:
        lead_end = max(1, line.find("solve."))
    lead_start = line_start
    lead_span_end = line_start + lead_end

    instruction_line_positions = set(token_positions_for_char_span(text, line_start, line_end, tokenizer))
    lead_positions = set(token_positions_for_char_span(text, lead_start, lead_span_end, tokenizer))
    file_positions = set(token_positions_for_char_span(text, file_start, file_end, tokenizer))
    task_positions = set(token_positions_for_char_span(text, task_start, task_end, tokenizer))
    func_m = re.search(r"function body", line)
    func_positions: set[int] = set()
    if func_m:
        func_positions = set(
            token_positions_for_char_span(
                text,
                line_start + func_m.start(),
                line_start + func_m.end(),
                tokenizer,
            )
        )
    tail_positions = sorted(instruction_line_positions - lead_positions - file_positions - func_positions)

    return {
        "lead_phrase": sorted(lead_positions),
        "function_body_anchor": sorted(func_positions),
        "file_target": sorted(file_positions),
        "tail_suffix": tail_positions,
        "task_body": sorted(task_positions),
    }


def collect_head_cache(model, tokens: torch.Tensor, layers: Sequence[int]) -> Dict[str, torch.Tensor]:
    names = set()
    for layer in sorted(set(layers)):
        names.add(f"blocks.{layer}.attn.hook_pattern")
        names.add(f"blocks.{layer}.attn.hook_q")
        names.add(f"blocks.{layer}.attn.hook_k")
        names.add(f"blocks.{layer}.attn.hook_v")
        names.add(f"blocks.{layer}.attn.hook_z")
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens, names_filter=lambda n: n in names)
    return {k: v.detach().cpu() for k, v in cache.items()}


def run_head_z_patch_from_positions(
    model,
    base_tokens: torch.Tensor,
    source_cache: Dict[str, torch.Tensor],
    base_cache: Dict[str, torch.Tensor],
    layer: int,
    head: int,
    positions: Sequence[int],
) -> torch.Tensor:
    pat_key = f"blocks.{layer}.attn.hook_pattern"
    v_key = f"blocks.{layer}.attn.hook_v"
    z_key = f"blocks.{layer}.attn.hook_z"

    q_heads = int(source_cache[pat_key].shape[1])
    kv_heads = int(source_cache[v_key].shape[2])
    group = max(1, q_heads // kv_heads)
    kv_index = head // group

    pat_source = source_cache[pat_key][0, head, -1, :].float().to(base_tokens.device)
    pat_base = base_cache[pat_key][0, head, -1, :].float().to(base_tokens.device)
    v_source = source_cache[v_key][0, :, kv_index, :].float().to(base_tokens.device)
    v_base = base_cache[v_key][0, :, kv_index, :].float().to(base_tokens.device)
    z_base = base_cache[z_key][0, -1, head, :].float().to(base_tokens.device)

    seq_len = int(v_source.shape[0])
    valid_positions = [int(p) for p in positions if 0 <= int(p) < seq_len]
    delta_by_pos = pat_source[:, None] * v_source - pat_base[:, None] * v_base
    target_z = z_base.clone()
    if valid_positions:
        target_z = target_z + delta_by_pos[valid_positions].sum(dim=0)
    target_z = target_z.to(dtype=base_cache[z_key].dtype)

    hook_name = z_key

    def hook_fn(z: torch.Tensor, hook):  # noqa: ANN001
        out = z.clone()
        out[:, -1, head, :] = target_z
        return out

    with torch.no_grad():
        return model.run_with_hooks(base_tokens, fwd_hooks=[(hook_name, hook_fn)])


def run_head_component_patch(
    model,
    base_tokens: torch.Tensor,
    source_cache: Dict[str, torch.Tensor],
    layer: int,
    head: int,
    component: str,
    positions: Sequence[int] | None = None,
) -> torch.Tensor:
    q_heads = int(source_cache[f"blocks.{layer}.attn.hook_q"].shape[2])
    kv_heads = int(source_cache[f"blocks.{layer}.attn.hook_k"].shape[2])
    group = max(1, q_heads // kv_heads)
    kv_index = head // group

    hooks = []
    if component == "q":
        src = source_cache[f"blocks.{layer}.attn.hook_q"].to(base_tokens.device)

        def hook_fn(q: torch.Tensor, hook):  # noqa: ANN001
            out = q.clone()
            out[:, -1, head, :] = src[:, -1, head, :].to(dtype=q.dtype)
            return out

        hooks.append((f"blocks.{layer}.attn.hook_q", hook_fn))
    elif component == "k":
        src = source_cache[f"blocks.{layer}.attn.hook_k"].to(base_tokens.device)
        valid_positions = None if positions is None else [int(p) for p in positions]

        def hook_fn(k: torch.Tensor, hook):  # noqa: ANN001
            out = k.clone()
            if valid_positions is None:
                out[:, :, kv_index, :] = src[:, :, kv_index, :].to(dtype=k.dtype)
            else:
                for p in valid_positions:
                    if 0 <= p < out.shape[1]:
                        out[:, p, kv_index, :] = src[:, p, kv_index, :].to(dtype=k.dtype)
            return out

        hooks.append((f"blocks.{layer}.attn.hook_k", hook_fn))
    elif component == "v":
        src = source_cache[f"blocks.{layer}.attn.hook_v"].to(base_tokens.device)
        valid_positions = None if positions is None else [int(p) for p in positions]

        def hook_fn(v: torch.Tensor, hook):  # noqa: ANN001
            out = v.clone()
            if valid_positions is None:
                out[:, :, kv_index, :] = src[:, :, kv_index, :].to(dtype=v.dtype)
            else:
                for p in valid_positions:
                    if 0 <= p < out.shape[1]:
                        out[:, p, kv_index, :] = src[:, p, kv_index, :].to(dtype=v.dtype)
            return out

        hooks.append((f"blocks.{layer}.attn.hook_v", hook_fn))
    elif component == "z":
        src = source_cache[f"blocks.{layer}.attn.hook_z"].to(base_tokens.device)

        def hook_fn(z: torch.Tensor, hook):  # noqa: ANN001
            out = z.clone()
            out[:, -1, head, :] = src[:, -1, head, :].to(dtype=z.dtype)
            return out

        hooks.append((f"blocks.{layer}.attn.hook_z", hook_fn))
    else:
        raise ValueError(f"unknown component: {component}")

    with torch.no_grad():
        return model.run_with_hooks(base_tokens, fwd_hooks=hooks)


def main() -> None:
    parser = argparse.ArgumentParser(description="Final attention-head audit for the refined mechanism story.")
    parser.add_argument("--run-root", type=str, required=True)
    parser.add_argument("--model-path", type=str, default="./external/models/Qwen3-1.7B")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--output-root", type=str, required=True)
    args = parser.parse_args()

    run_root = Path(args.run_root).resolve()
    out_root = Path(args.output_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    layers = [parse_head(h)[0] for h in ALL_HEADS]
    samples = load_sample_paths(run_root / "forward_batch", run_root / "reverse_batch", max_samples=args.max_samples)
    model, tokenizer = load_hooked_qwen3(args.model_path, device=args.device, dtype=torch.bfloat16)

    span_rows: List[Dict[str, object]] = []
    patch_rows: List[Dict[str, object]] = []
    qkv_rows: List[Dict[str, object]] = []
    top_token_rows: List[Dict[str, object]] = []

    pbar = tqdm(samples, desc="Final head attention audit", dynamic_ncols=True)
    for sp in pbar:
        try:
            clean_text = sp.tool_call_prompt.read_text(encoding="utf-8")
            corrupt_text = sp.no_tool_prompt.read_text(encoding="utf-8")
        except Exception:
            continue

        lead_variants = build_lead_variants(clean_text, corrupt_text)
        clean_tokens = model.to_tokens(lead_variants["clean_full"], prepend_bos=False)
        corrupt_tokens = model.to_tokens(lead_variants["corrupt_full"], prepend_bos=False)
        clean_with_corrupt_lead_tokens = model.to_tokens(lead_variants["clean_with_corrupt_lead"], prepend_bos=False)
        corrupt_with_clean_lead_tokens = model.to_tokens(lead_variants["corrupt_with_clean_lead"], prepend_bos=False)

        with torch.no_grad():
            clean_logits = model(clean_tokens)
            corrupt_logits = model(corrupt_tokens)
            clean_with_corrupt_lead_logits = model(clean_with_corrupt_lead_tokens)
            corrupt_with_clean_lead_logits = model(corrupt_with_clean_lead_tokens)

        tool_objective, no_tool_objective = build_bidirectional_endpoint_objectives(
            clean_logits,
            corrupt_logits,
            tokenizer=tokenizer,
        )
        tool_gap = float(objective_from_logits(clean_logits, tool_objective).item()) - float(
            objective_from_logits(corrupt_logits, tool_objective).item()
        )
        no_tool_gap = float(objective_from_logits(corrupt_logits, no_tool_objective).item()) - float(
            objective_from_logits(clean_logits, no_tool_objective).item()
        )
        if (
            not math.isfinite(tool_gap)
            or abs(tool_gap) < 1e-8
            or not math.isfinite(no_tool_gap)
            or abs(no_tool_gap) < 1e-8
        ):
            continue

        clean_cache = collect_head_cache(model, clean_tokens, layers)
        corrupt_cache = collect_head_cache(model, corrupt_tokens, layers)
        clean_with_corrupt_lead_cache = collect_head_cache(model, clean_with_corrupt_lead_tokens, layers)
        corrupt_with_clean_lead_cache = collect_head_cache(model, corrupt_with_clean_lead_tokens, layers)

        clean_spans = build_span_positions(lead_variants["clean_full"], tokenizer)
        corrupt_spans = build_span_positions(lead_variants["corrupt_full"], tokenizer)
        clean_with_corrupt_lead_spans = build_span_positions(lead_variants["clean_with_corrupt_lead"], tokenizer)
        corrupt_with_clean_lead_spans = build_span_positions(lead_variants["corrupt_with_clean_lead"], tokenizer)

        for head_name in ALL_HEADS:
            layer, head = parse_head(head_name)
            pat_key = f"blocks.{layer}.attn.hook_pattern"

            if head_name in TOOL_HEADS:
                family = "tool"
                source_cache = clean_cache
                base_cache = clean_with_corrupt_lead_cache
                base_tokens = clean_with_corrupt_lead_tokens
                base_logits = clean_with_corrupt_lead_logits
                base_spans = clean_with_corrupt_lead_spans
                source_spans = clean_spans
                gap = tool_gap
                objective = tool_objective
                base_score = float(objective_from_logits(base_logits, tool_objective).item())
                compare_score = float(objective_from_logits(base_logits, no_tool_objective).item())
                target_id = sp.target_tool_call
            else:
                family = "no_tool"
                source_cache = corrupt_cache
                base_cache = corrupt_with_clean_lead_cache
                base_tokens = corrupt_with_clean_lead_tokens
                base_logits = corrupt_with_clean_lead_logits
                base_spans = corrupt_with_clean_lead_spans
                source_spans = corrupt_spans
                gap = no_tool_gap
                objective = no_tool_objective
                base_score = float(objective_from_logits(base_logits, no_tool_objective).item())
                compare_score = float(objective_from_logits(base_logits, tool_objective).item())
                target_id = sp.distractor

            base_pattern = base_cache[pat_key][0, head, -1, :].float()
            top_pos = torch.topk(base_pattern, k=min(5, int(base_pattern.numel()))).indices.tolist()
            for rank, pos in enumerate(top_pos, start=1):
                tok = tokenizer.decode([int(base_tokens[0, int(pos)].item())]).replace("\n", "\\n")
                top_token_rows.append(
                    {
                        "sample_id": sp.sample_id,
                        "head": head_name,
                        "family": family,
                        "rank": rank,
                        "pos": int(pos),
                        "token": tok,
                        "attn": float(base_pattern[int(pos)].item()),
                    }
                )

            for span_name in ["lead_phrase", "function_body_anchor", "file_target", "tail_suffix", "task_body"]:
                idxs = [int(i) for i in base_spans.get(span_name, []) if 0 <= int(i) < base_pattern.shape[0]]
                mass = float(base_pattern[idxs].sum().item()) if idxs else float("nan")
                density = float(base_pattern[idxs].mean().item()) if idxs else float("nan")
                span_rows.append(
                    {
                        "sample_id": sp.sample_id,
                        "head": head_name,
                        "family": family,
                        "span": span_name,
                        "n_tokens": len(idxs),
                        "attn_mass_base": mass,
                        "attn_density_base": density,
                    }
                )

            for span_name in ["lead_phrase", "function_body_anchor", "file_target", "tail_suffix", "task_body"]:
                logits = run_head_z_patch_from_positions(
                    model,
                    base_tokens,
                    source_cache,
                    base_cache,
                    layer,
                    head,
                    source_spans.get(span_name, []),
                )
                patched_score = float(objective_from_logits(logits, objective).item())
                other_score = float(objective_from_logits(logits, no_tool_objective if family == "tool" else tool_objective).item())
                patch_rows.append(
                    {
                        "sample_id": sp.sample_id,
                        "head": head_name,
                        "family": family,
                        "span": span_name,
                        "rescue_ratio": (patched_score - base_score) / gap,
                        "decision_score": (patched_score - other_score) if family == "tool" else (patched_score - other_score),
                        "top1_success": int(logits[0, -1].argmax().item()) == target_id,
                    }
                )

            lead_positions = source_spans.get("lead_phrase", [])
            for component, positions in [("q", None), ("k", lead_positions), ("v", lead_positions), ("z", None)]:
                logits = run_head_component_patch(
                    model,
                    base_tokens,
                    source_cache,
                    layer,
                    head,
                    component,
                    positions=positions,
                )
                patched_score = float(objective_from_logits(logits, objective).item())
                other_score = float(objective_from_logits(logits, no_tool_objective if family == "tool" else tool_objective).item())
                qkv_rows.append(
                    {
                        "sample_id": sp.sample_id,
                        "head": head_name,
                        "family": family,
                        "component": component,
                        "rescue_ratio": (patched_score - base_score) / gap,
                        "decision_score": (patched_score - other_score) if family == "tool" else (patched_score - other_score),
                        "top1_success": int(logits[0, -1].argmax().item()) == target_id,
                    }
                )

        pbar.set_postfix(sample=sp.sample_id)

    write_csv(span_rows, out_root / "head_span_attention_per_sample.csv")
    write_csv(patch_rows, out_root / "head_span_patch_per_sample.csv")
    write_csv(qkv_rows, out_root / "head_qkv_patch_per_sample.csv")
    write_csv(top_token_rows, out_root / "head_top_tokens_per_sample.csv")

    span_summary: List[Dict[str, object]] = []
    grouped_spans: Dict[Tuple[str, str, str], List[Dict[str, object]]] = defaultdict(list)
    for row in span_rows:
        grouped_spans[(str(row["head"]), str(row["family"]), str(row["span"]))].append(row)
    for (head_name, family, span_name), rows in sorted(grouped_spans.items()):
        span_summary.append(
            {
                "head": head_name,
                "family": family,
                "span": span_name,
                "attn_mass_median": median(float(r["attn_mass_base"]) for r in rows if math.isfinite(float(r["attn_mass_base"]))),
                "attn_density_median": median(float(r["attn_density_base"]) for r in rows if math.isfinite(float(r["attn_density_base"]))),
                "n_tokens_median": median(float(r["n_tokens"]) for r in rows if math.isfinite(float(r["n_tokens"]))),
            }
        )

    patch_summary: List[Dict[str, object]] = []
    grouped_patches: Dict[Tuple[str, str, str], List[Dict[str, object]]] = defaultdict(list)
    for row in patch_rows:
        grouped_patches[(str(row["head"]), str(row["family"]), str(row["span"]))].append(row)
    for (head_name, family, span_name), rows in sorted(grouped_patches.items()):
        patch_summary.append(
            {
                "head": head_name,
                "family": family,
                "span": span_name,
                "rescue_ratio_median": median(float(r["rescue_ratio"]) for r in rows),
                "top1_rate": safe_rate(bool(r["top1_success"]) for r in rows),
            }
        )

    qkv_summary: List[Dict[str, object]] = []
    grouped_qkv: Dict[Tuple[str, str, str], List[Dict[str, object]]] = defaultdict(list)
    for row in qkv_rows:
        grouped_qkv[(str(row["head"]), str(row["family"]), str(row["component"]))].append(row)
    for (head_name, family, component), rows in sorted(grouped_qkv.items()):
        qkv_summary.append(
            {
                "head": head_name,
                "family": family,
                "component": component,
                "rescue_ratio_median": median(float(r["rescue_ratio"]) for r in rows),
                "top1_rate": safe_rate(bool(r["top1_success"]) for r in rows),
            }
        )

    final_rows: List[Dict[str, object]] = []
    for head_name in ALL_HEADS:
        family = "tool" if head_name in TOOL_HEADS else "no_tool"
        head_spans = [r for r in span_summary if r["head"] == head_name and r["family"] == family]
        head_patches = [r for r in patch_summary if r["head"] == head_name and r["family"] == family]
        head_qkv = [r for r in qkv_summary if r["head"] == head_name and r["family"] == family]
        best_span_read = max(head_spans, key=lambda r: float(r["attn_density_median"]), default={})
        best_span_patch = max(head_patches, key=lambda r: float(r["rescue_ratio_median"]), default={})
        best_component = max(head_qkv, key=lambda r: float(r["rescue_ratio_median"]), default={})
        final_rows.append(
            {
                "head": head_name,
                "family": family,
                "best_read_span": best_span_read.get("span", ""),
                "best_read_mass_median": best_span_read.get("attn_mass_median", float("nan")),
                "best_read_density_median": best_span_read.get("attn_density_median", float("nan")),
                "best_causal_span": best_span_patch.get("span", ""),
                "best_causal_span_rescue_median": best_span_patch.get("rescue_ratio_median", float("nan")),
                "best_causal_span_top1_rate": best_span_patch.get("top1_rate", float("nan")),
                "best_qkv_component": best_component.get("component", ""),
                "best_qkv_rescue_median": best_component.get("rescue_ratio_median", float("nan")),
                "best_qkv_top1_rate": best_component.get("top1_rate", float("nan")),
            }
        )

    write_csv(span_summary, out_root / "head_span_attention_summary.csv")
    write_csv(patch_summary, out_root / "head_span_patch_summary.csv")
    write_csv(qkv_summary, out_root / "head_qkv_patch_summary.csv")
    write_csv(final_rows, out_root / "head_final_audit_summary.csv")
    top_summary: List[Dict[str, object]] = []
    grouped_top: Dict[Tuple[str, str, int, str], List[Dict[str, object]]] = defaultdict(list)
    for row in top_token_rows:
        grouped_top[(str(row["head"]), str(row["family"]), int(row["rank"]), str(row["token"]))].append(row)
    for (head_name, family, rank, token), rows in sorted(grouped_top.items()):
        top_summary.append(
            {
                "head": head_name,
                "family": family,
                "rank": rank,
                "token": token,
                "count": len(rows),
                "attn_median": median(float(r["attn"]) for r in rows),
            }
        )
    write_csv(top_summary, out_root / "head_top_tokens_summary.csv")

    summary = {
        "span_summary_rows": span_summary,
        "patch_summary_rows": patch_summary,
        "qkv_summary_rows": qkv_summary,
        "final_rows": final_rows,
        "top_token_summary_rows": top_summary,
        "artifacts": {
            "span_attention_csv": str(out_root / "head_span_attention_summary.csv"),
            "span_patch_csv": str(out_root / "head_span_patch_summary.csv"),
            "qkv_patch_csv": str(out_root / "head_qkv_patch_summary.csv"),
            "final_audit_csv": str(out_root / "head_final_audit_summary.csv"),
            "top_token_csv": str(out_root / "head_top_tokens_summary.csv"),
        },
    }
    (out_root / "head_final_audit_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = ["# Final Head Attention Audit", ""]
    for row in final_rows:
        top_tokens = [r["token"] for r in top_summary if r["head"] == row["head"] and int(r["rank"]) <= 3][:3]
        lines.append(
            f"- `{row['head']}` ({row['family']}): read `{row['best_read_span']}` mass `{row['best_read_mass_median']:.3f}`, "
            f"causal span `{row['best_causal_span']}` rescue `{row['best_causal_span_rescue_median']:.3f}`, "
            f"best component `{row['best_qkv_component']}` rescue `{row['best_qkv_rescue_median']:.3f}`, "
            f"top tokens `{', '.join(top_tokens)}`."
        )
    (out_root / "head_final_audit_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
