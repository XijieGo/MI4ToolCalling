#!/usr/bin/env python3
"""
Minimal instruction lead-phrase audit.

Starting from the existing stable clean/corrupt pairs, this script extracts the
shared suffix of the first instruction line and swaps only the differing lead
phrase. This approximates the smallest human-readable cue that distinguishes:

- file-committing / execution-leaning phrasing
- text-authoring / non-committing phrasing
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
from toolcall_circuit.signed_edge_importance import run_logits_with_assignments
from toolcall_circuit.single_sample import load_hooked_qwen3, objective_from_logits


QUERY_CHAIN = ["L20H5", "L21H1", "L21H12", "L24H6", "MLP27"]


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


def longest_common_suffix(a: str, b: str) -> str:
    i = 0
    limit = min(len(a), len(b))
    while i < limit and a[-1 - i] == b[-1 - i]:
        i += 1
    return a[len(a) - i :] if i > 0 else ""


def build_variants(clean_text: str, corrupt_text: str) -> Dict[str, str]:
    clean_prefix_outer, clean_user, clean_suffix_outer = split_prompt_outer(clean_text)
    corrupt_prefix_outer, corrupt_user, corrupt_suffix_outer = split_prompt_outer(corrupt_text)
    clean_line, clean_rest = split_instruction_line(clean_user)
    corrupt_line, corrupt_rest = split_instruction_line(corrupt_user)
    shared_suffix = longest_common_suffix(clean_line, corrupt_line)
    clean_lead = clean_line[: len(clean_line) - len(shared_suffix)] if shared_suffix else clean_line
    corrupt_lead = corrupt_line[: len(corrupt_line) - len(shared_suffix)] if shared_suffix else corrupt_line

    clean_line_with_corrupt_lead = corrupt_lead + shared_suffix
    corrupt_line_with_clean_lead = clean_lead + shared_suffix
    return {
        "clean_full": clean_text,
        "corrupt_full": corrupt_text,
        "clean_with_corrupt_lead": clean_prefix_outer + clean_line_with_corrupt_lead + ("\n" + clean_rest if clean_rest else "") + clean_suffix_outer,
        "corrupt_with_clean_lead": corrupt_prefix_outer + corrupt_line_with_clean_lead + ("\n" + corrupt_rest if corrupt_rest else "") + corrupt_suffix_outer,
        "clean_lead": clean_lead,
        "corrupt_lead": corrupt_lead,
        "shared_suffix": shared_suffix,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Instruction lead-phrase audit.")
    parser.add_argument("--run-root", type=str, required=True)
    parser.add_argument("--model-path", type=str, default="./external/models/Qwen3-1.7B")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--output-root", type=str, required=True)
    args = parser.parse_args()

    run_root = Path(args.run_root).resolve()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    samples = load_sample_paths(run_root / "forward_batch", run_root / "reverse_batch", max_samples=args.max_samples)
    model, tokenizer = load_hooked_qwen3(args.model_path, device=args.device, dtype=torch.bfloat16)
    _ = tokenizer

    variant_rows: List[Dict[str, object]] = []
    query_rows: List[Dict[str, object]] = []
    phrase_rows: List[Dict[str, object]] = []

    pbar = tqdm(samples, desc="Instruction lead audit", dynamic_ncols=True)
    for sp in pbar:
        try:
            clean_text = sp.tool_call_prompt.read_text(encoding="utf-8")
            corrupt_text = sp.no_tool_prompt.read_text(encoding="utf-8")
        except Exception:
            continue
        variants = build_variants(clean_text, corrupt_text)
        phrase_rows.append(
            {
                "sample_id": sp.sample_id,
                "clean_lead": variants["clean_lead"],
                "corrupt_lead": variants["corrupt_lead"],
                "shared_suffix": variants["shared_suffix"],
            }
        )

        tokens_by_variant: Dict[str, torch.Tensor] = {}
        logits_by_variant: Dict[str, torch.Tensor] = {}
        for name in ["clean_full", "corrupt_full", "clean_with_corrupt_lead", "corrupt_with_clean_lead"]:
            tokens = model.to_tokens(variants[name], prepend_bos=False)
            tokens_by_variant[name] = tokens
            with torch.no_grad():
                logits_by_variant[name] = model(tokens)

        tool_objective, no_tool_objective = build_bidirectional_endpoint_objectives(
            logits_by_variant["clean_full"],
            logits_by_variant["corrupt_full"],
            tokenizer=tokenizer,
        )
        tool_gap = float(objective_from_logits(logits_by_variant["clean_full"], tool_objective).item()) - float(
            objective_from_logits(logits_by_variant["corrupt_full"], tool_objective).item()
        )
        if not math.isfinite(tool_gap) or abs(tool_gap) < 1e-8:
            continue

        for name in ["clean_full", "clean_with_corrupt_lead", "corrupt_with_clean_lead", "corrupt_full"]:
            logits = logits_by_variant[name]
            tool_score = float(objective_from_logits(logits, tool_objective).item())
            no_tool_score = float(objective_from_logits(logits, no_tool_objective).item())
            variant_rows.append(
                {
                    "sample_id": sp.sample_id,
                    "variant": name,
                    "tool_score": tool_score,
                    "no_tool_score": no_tool_score,
                    "decision_score": tool_score - no_tool_score,
                    "tool_top1": int(logits[0, -1].argmax().item()) == sp.target_tool_call,
                    "no_tool_top1": int(logits[0, -1].argmax().item()) == sp.distractor,
                }
            )

        clean_cache = collect_cache_cpu_for_nodes(model, tokens_by_variant["clean_full"], QUERY_CHAIN)
        base_cache = collect_cache_cpu_for_nodes(model, tokens_by_variant["clean_with_corrupt_lead"], QUERY_CHAIN)
        base_tool = float(objective_from_logits(logits_by_variant["clean_with_corrupt_lead"], tool_objective).item())
        base_no_tool = float(objective_from_logits(logits_by_variant["clean_with_corrupt_lead"], no_tool_objective).item())
        for step_idx in range(1, len(QUERY_CHAIN) + 1):
            nodes = QUERY_CHAIN[:step_idx]
            logits = run_logits_with_assignments(
                model,
                tokens_by_variant["clean_with_corrupt_lead"],
                clean_cache,
                base_cache,
                nodes,
                [],
            )
            tool_score = float(objective_from_logits(logits, tool_objective).item())
            no_tool_score = float(objective_from_logits(logits, no_tool_objective).item())
            query_rows.append(
                {
                    "sample_id": sp.sample_id,
                    "step_idx": step_idx,
                    "nodes": "|".join(nodes),
                    "rescue_ratio": (tool_score - base_tool) / tool_gap,
                    "decision_score": tool_score - no_tool_score,
                    "tool_top1": int(logits[0, -1].argmax().item()) == sp.target_tool_call,
                    "boundary_flip": tool_score > no_tool_score,
                }
            )
        pbar.set_postfix(sample=sp.sample_id)

    write_csv(variant_rows, output_root / "instruction_lead_variant_per_sample.csv")
    write_csv(query_rows, output_root / "instruction_lead_query_per_sample.csv")
    write_csv(phrase_rows, output_root / "instruction_lead_pairs.csv")

    variant_summary = []
    by_variant: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in variant_rows:
        by_variant[str(row["variant"])].append(row)
    for variant, rows in sorted(by_variant.items()):
        variant_summary.append(
            {
                "variant": variant,
                "n_samples": len(rows),
                "decision_score_median": median(float(r["decision_score"]) for r in rows),
                "tool_top1_rate": safe_rate(bool(r["tool_top1"]) for r in rows),
                "no_tool_top1_rate": safe_rate(bool(r["no_tool_top1"]) for r in rows),
            }
        )

    query_summary = []
    by_step: Dict[int, List[Dict[str, object]]] = defaultdict(list)
    for row in query_rows:
        by_step[int(row["step_idx"])].append(row)
    for step_idx, rows in sorted(by_step.items()):
        query_summary.append(
            {
                "step_idx": step_idx,
                "nodes": rows[0]["nodes"],
                "rescue_ratio_median": median(float(r["rescue_ratio"]) for r in rows),
                "decision_score_median": median(float(r["decision_score"]) for r in rows),
                "tool_top1_rate": safe_rate(bool(r["tool_top1"]) for r in rows),
                "boundary_flip_rate": safe_rate(bool(r["boundary_flip"]) for r in rows),
            }
        )

    summary = {
        "variant_summary_rows": variant_summary,
        "query_summary_rows": query_summary,
        "artifacts": {
            "variant_per_sample_csv": str(output_root / "instruction_lead_variant_per_sample.csv"),
            "query_per_sample_csv": str(output_root / "instruction_lead_query_per_sample.csv"),
            "pair_csv": str(output_root / "instruction_lead_pairs.csv"),
        },
    }
    (output_root / "instruction_lead_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = ["# Instruction Lead-Phrase Audit", ""]
    variant_map = {str(r["variant"]): r for r in variant_summary}
    for key in ["clean_full", "clean_with_corrupt_lead", "corrupt_with_clean_lead", "corrupt_full"]:
        row = variant_map.get(key)
        if row:
            lines.append(
                f"- `{key}`: decision `{row['decision_score_median']:.3f}`, tool-top1 `{row['tool_top1_rate']:.3f}`, no-tool-top1 `{row['no_tool_top1_rate']:.3f}`."
            )
    if query_summary:
        row = query_summary[-1]
        lines.append(
            f"- Query chain on `clean_with_corrupt_lead`: `{row['nodes']}` -> rescue `{row['rescue_ratio_median']:.3f}`, tool-top1 `{row['tool_top1_rate']:.3f}`."
        )
    (output_root / "instruction_lead_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
