#!/usr/bin/env python3
"""
Instruction-line commitment audit.

This script stays within the existing stable clean/corrupt pairs and isolates
the most human-interpretable part of the user prompt: the first instruction
line. For this dataset, that line often carries the difference between:

- perform the task into `solve.py` / `solve.cpp`
- describe / implement the function body in text

We swap only that instruction line while keeping the same task body, then audit
whether the late query-conditioned chain follows the instruction-line swap.
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

import matplotlib.pyplot as plt
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
NO_TOOL_CHAIN = ["L16H4", "MLP17", "L23H6"]
READ_HEADS = ["L20H5", "L21H1", "L21H12", "L16H4"]


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
    outer_prefix = text[:user_start]
    user_content = text[user_start:end]
    outer_suffix = text[end:]
    return outer_prefix, user_content, outer_suffix


def split_instruction_line(user_content: str) -> Tuple[str, str]:
    if "\n" not in user_content:
        return user_content, ""
    first, rest = user_content.split("\n", 1)
    return first, rest


def build_instruction_variants(clean_text: str, corrupt_text: str) -> Dict[str, str]:
    clean_outer_prefix, clean_user, clean_outer_suffix = split_prompt_outer(clean_text)
    corrupt_outer_prefix, corrupt_user, corrupt_outer_suffix = split_prompt_outer(corrupt_text)
    clean_line, clean_rest = split_instruction_line(clean_user)
    corrupt_line, corrupt_rest = split_instruction_line(corrupt_user)
    # The system prompt is aligned in the stable dataset, so we can use the
    # matching outer shells from each side.
    return {
        "clean_full": clean_text,
        "corrupt_full": corrupt_text,
        "clean_with_corrupt_instruction": clean_outer_prefix + corrupt_line + ("\n" + clean_rest if clean_rest else "") + clean_outer_suffix,
        "corrupt_with_clean_instruction": corrupt_outer_prefix + clean_line + ("\n" + corrupt_rest if corrupt_rest else "") + corrupt_outer_suffix,
        "clean_instruction": clean_line,
        "corrupt_instruction": corrupt_line,
    }


def token_positions_for_char_span(text: str, start: int, end: int, tokenizer) -> List[int]:
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    positions: List[int] = []
    for idx, (tok_start, tok_end) in enumerate(enc["offset_mapping"]):
        if int(tok_start) < int(end) and int(tok_end) > int(start):
            positions.append(int(idx))
    return positions


def instruction_task_positions(text: str, tokenizer) -> Dict[str, List[int]]:
    _outer_prefix, user_content, _outer_suffix = split_prompt_outer(text)
    line, rest = split_instruction_line(user_content)
    user_marker = "<|im_start|>user\n"
    start = text.find(user_marker)
    user_start = start + len(user_marker)
    line_start = user_start
    line_end = line_start + len(line)
    task_start = line_end + (1 if rest else 0)
    task_end = task_start + len(rest)
    return {
        "instruction_line": token_positions_for_char_span(text, line_start, line_end, tokenizer),
        "task_body": token_positions_for_char_span(text, task_start, task_end, tokenizer),
    }


def read_mass_summary(model, tokenizer, text: str, heads: Sequence[str]) -> Dict[Tuple[str, str], float]:
    if not heads:
        return {}
    tokens = model.to_tokens(text, prepend_bos=False)
    positions = instruction_task_positions(text, tokenizer)
    layers = sorted({int(h[1:].split("H")[0]) for h in heads})
    needed = {f"blocks.{layer}.attn.hook_pattern" for layer in layers}
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens, names_filter=lambda n: n in needed)
    out: Dict[Tuple[str, str], float] = {}
    for head in heads:
        layer, head_idx = map(int, head[1:].split("H"))
        pattern = cache[f"blocks.{layer}.attn.hook_pattern"][0, head_idx, -1, :]
        for set_name in ["instruction_line", "task_body"]:
            idxs = [int(p) for p in positions.get(set_name, []) if 0 <= int(p) < int(pattern.shape[0])]
            out[(head, set_name)] = float(pattern[idxs].sum().item()) if idxs else float("nan")
    model.reset_hooks()
    return out


def plot_variant_summary(summary_rows: Sequence[Dict[str, object]], out_path: Path) -> None:
    order = [
        "clean_full",
        "clean_with_corrupt_instruction",
        "corrupt_with_clean_instruction",
        "corrupt_full",
    ]
    row_map = {str(r["variant"]): r for r in summary_rows}
    xs = np.arange(len(order))
    decision = [float(row_map[name]["decision_score_median"]) for name in order if name in row_map]
    tool = [float(row_map[name]["tool_top1_rate"]) for name in order if name in row_map]
    labels = [name for name in order if name in row_map]
    plt.style.use("default")
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.8), constrained_layout=True)
    axes[0].plot(np.arange(len(labels)), decision, marker="o")
    axes[0].axhline(0.0, color="#888888", linewidth=1.0)
    axes[0].set_xticks(np.arange(len(labels)))
    axes[0].set_xticklabels(labels, rotation=25, ha="right")
    axes[0].set_title("Instruction Swap: Decision Score")
    axes[1].plot(np.arange(len(labels)), tool, marker="o")
    axes[1].set_xticks(np.arange(len(labels)))
    axes[1].set_xticklabels(labels, rotation=25, ha="right")
    axes[1].set_title("Instruction Swap: Tool Top-1")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Instruction-line commitment audit on the stable clean/corrupt pairs.")
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

    all_nodes = sorted(set(QUERY_CHAIN + NO_TOOL_CHAIN))
    variant_rows: List[Dict[str, object]] = []
    read_rows: List[Dict[str, object]] = []
    query_step_rows: List[Dict[str, object]] = []
    no_tool_step_rows: List[Dict[str, object]] = []
    instruction_rows: List[Dict[str, object]] = []

    pbar = tqdm(samples, desc="Instruction commitment audit", dynamic_ncols=True)
    for sp in pbar:
        try:
            clean_text = sp.tool_call_prompt.read_text(encoding="utf-8")
            corrupt_text = sp.no_tool_prompt.read_text(encoding="utf-8")
        except Exception:
            continue
        variants = build_instruction_variants(clean_text, corrupt_text)
        instruction_rows.append(
            {
                "sample_id": sp.sample_id,
                "clean_instruction": variants["clean_instruction"],
                "corrupt_instruction": variants["corrupt_instruction"],
            }
        )

        tokens_by_variant: Dict[str, torch.Tensor] = {}
        logits_by_variant: Dict[str, torch.Tensor] = {}
        for name in [
            "clean_full",
            "corrupt_full",
            "clean_with_corrupt_instruction",
            "corrupt_with_clean_instruction",
        ]:
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
        no_tool_gap = float(objective_from_logits(logits_by_variant["corrupt_full"], no_tool_objective).item()) - float(
            objective_from_logits(logits_by_variant["clean_full"], no_tool_objective).item()
        )
        if (
            not math.isfinite(tool_gap)
            or abs(tool_gap) < 1e-8
            or not math.isfinite(no_tool_gap)
            or abs(no_tool_gap) < 1e-8
        ):
            continue

        clean_cache = collect_cache_cpu_for_nodes(model, tokens_by_variant["clean_full"], all_nodes)
        corrupt_cache = collect_cache_cpu_for_nodes(model, tokens_by_variant["corrupt_full"], all_nodes)
        swap_clean_cache = collect_cache_cpu_for_nodes(model, tokens_by_variant["clean_with_corrupt_instruction"], all_nodes)
        swap_corrupt_cache = collect_cache_cpu_for_nodes(model, tokens_by_variant["corrupt_with_clean_instruction"], all_nodes)

        for variant_name in [
            "clean_full",
            "clean_with_corrupt_instruction",
            "corrupt_with_clean_instruction",
            "corrupt_full",
        ]:
            logits = logits_by_variant[variant_name]
            tool_score = float(objective_from_logits(logits, tool_objective).item())
            no_tool_score = float(objective_from_logits(logits, no_tool_objective).item())
            variant_rows.append(
                {
                    "sample_id": sp.sample_id,
                    "variant": variant_name,
                    "tool_score": tool_score,
                    "no_tool_score": no_tool_score,
                    "decision_score": tool_score - no_tool_score,
                    "tool_top1": int(logits[0, -1].argmax().item()) == sp.target_tool_call,
                    "no_tool_top1": int(logits[0, -1].argmax().item()) == sp.distractor,
                }
            )
            masses = read_mass_summary(model, tokenizer, variants[variant_name], READ_HEADS)
            for head in READ_HEADS:
                for set_name in ["instruction_line", "task_body"]:
                    read_rows.append(
                        {
                            "sample_id": sp.sample_id,
                            "variant": variant_name,
                            "component": head,
                            "set": set_name,
                            "mass": masses.get((head, set_name), float("nan")),
                        }
                    )

        # Query-chain rescue when the clean task body is kept but the corrupt instruction is inserted.
        base_tokens = tokens_by_variant["clean_with_corrupt_instruction"]
        base_tool = float(objective_from_logits(logits_by_variant["clean_with_corrupt_instruction"], tool_objective).item())
        base_no_tool = float(objective_from_logits(logits_by_variant["clean_with_corrupt_instruction"], no_tool_objective).item())
        for step_idx in range(1, len(QUERY_CHAIN) + 1):
            nodes = QUERY_CHAIN[:step_idx]
            logits = run_logits_with_assignments(model, base_tokens, clean_cache, swap_clean_cache, nodes, [])
            tool_score = float(objective_from_logits(logits, tool_objective).item())
            no_tool_score = float(objective_from_logits(logits, no_tool_objective).item())
            query_step_rows.append(
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

        # No-tool-chain rescue when the corrupt task body is kept but the clean instruction is inserted.
        base_tokens = tokens_by_variant["corrupt_with_clean_instruction"]
        base_tool = float(objective_from_logits(logits_by_variant["corrupt_with_clean_instruction"], tool_objective).item())
        base_no_tool = float(objective_from_logits(logits_by_variant["corrupt_with_clean_instruction"], no_tool_objective).item())
        for step_idx in range(1, len(NO_TOOL_CHAIN) + 1):
            nodes = NO_TOOL_CHAIN[:step_idx]
            logits = run_logits_with_assignments(model, base_tokens, corrupt_cache, swap_corrupt_cache, nodes, [])
            tool_score = float(objective_from_logits(logits, tool_objective).item())
            no_tool_score = float(objective_from_logits(logits, no_tool_objective).item())
            no_tool_step_rows.append(
                {
                    "sample_id": sp.sample_id,
                    "step_idx": step_idx,
                    "nodes": "|".join(nodes),
                    "rescue_ratio": (no_tool_score - base_no_tool) / no_tool_gap,
                    "decision_score": no_tool_score - tool_score,
                    "no_tool_top1": int(logits[0, -1].argmax().item()) == sp.distractor,
                    "boundary_flip": no_tool_score > tool_score,
                }
            )
        pbar.set_postfix(sample=sp.sample_id)

    write_csv(variant_rows, output_root / "instruction_variant_per_sample.csv")
    write_csv(read_rows, output_root / "instruction_read_per_sample.csv")
    write_csv(query_step_rows, output_root / "instruction_query_chain_per_sample.csv")
    write_csv(no_tool_step_rows, output_root / "instruction_no_tool_chain_per_sample.csv")
    write_csv(instruction_rows, output_root / "instruction_pairs.csv")

    variant_summary = []
    by_variant: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in variant_rows:
        by_variant[str(row["variant"])].append(row)
    for variant, rows in sorted(by_variant.items()):
        variant_summary.append(
            {
                "variant": variant,
                "n_samples": len(rows),
                "tool_score_median": median(float(r["tool_score"]) for r in rows),
                "no_tool_score_median": median(float(r["no_tool_score"]) for r in rows),
                "decision_score_median": median(float(r["decision_score"]) for r in rows),
                "tool_top1_rate": safe_rate(bool(r["tool_top1"]) for r in rows),
                "no_tool_top1_rate": safe_rate(bool(r["no_tool_top1"]) for r in rows),
            }
        )

    read_summary = []
    by_read: Dict[Tuple[str, str, str], List[Dict[str, object]]] = defaultdict(list)
    for row in read_rows:
        by_read[(str(row["variant"]), str(row["component"]), str(row["set"]))].append(row)
    for (variant, component, set_name), rows in sorted(by_read.items()):
        read_summary.append(
            {
                "variant": variant,
                "component": component,
                "set": set_name,
                "mass_median": median(float(r["mass"]) for r in rows if math.isfinite(float(r["mass"]))),
            }
        )

    query_summary = []
    by_q: Dict[int, List[Dict[str, object]]] = defaultdict(list)
    for row in query_step_rows:
        by_q[int(row["step_idx"])].append(row)
    for step_idx, rows in sorted(by_q.items()):
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

    no_tool_summary = []
    by_n: Dict[int, List[Dict[str, object]]] = defaultdict(list)
    for row in no_tool_step_rows:
        by_n[int(row["step_idx"])].append(row)
    for step_idx, rows in sorted(by_n.items()):
        no_tool_summary.append(
            {
                "step_idx": step_idx,
                "nodes": rows[0]["nodes"],
                "rescue_ratio_median": median(float(r["rescue_ratio"]) for r in rows),
                "decision_score_median": median(float(r["decision_score"]) for r in rows),
                "no_tool_top1_rate": safe_rate(bool(r["no_tool_top1"]) for r in rows),
                "boundary_flip_rate": safe_rate(bool(r["boundary_flip"]) for r in rows),
            }
        )

    clean_instruction_counter = Counter(str(r["clean_instruction"]) for r in instruction_rows)
    corrupt_instruction_counter = Counter(str(r["corrupt_instruction"]) for r in instruction_rows)

    write_csv(variant_summary, output_root / "instruction_variant_summary.csv")
    write_csv(read_summary, output_root / "instruction_read_summary.csv")
    write_csv(query_summary, output_root / "instruction_query_chain_summary.csv")
    write_csv(no_tool_summary, output_root / "instruction_no_tool_chain_summary.csv")
    plot_variant_summary(variant_summary, output_root / "instruction_variant_effects.png")

    summary = {
        "variant_summary_rows": variant_summary,
        "read_summary_rows": read_summary,
        "query_summary_rows": query_summary,
        "no_tool_summary_rows": no_tool_summary,
        "top_clean_instructions": clean_instruction_counter.most_common(20),
        "top_corrupt_instructions": corrupt_instruction_counter.most_common(20),
        "artifacts": {
            "variant_summary_csv": str(output_root / "instruction_variant_summary.csv"),
            "read_summary_csv": str(output_root / "instruction_read_summary.csv"),
            "query_summary_csv": str(output_root / "instruction_query_chain_summary.csv"),
            "no_tool_summary_csv": str(output_root / "instruction_no_tool_chain_summary.csv"),
            "variant_plot": str(output_root / "instruction_variant_effects.png"),
            "instruction_pairs_csv": str(output_root / "instruction_pairs.csv"),
        },
    }
    (output_root / "instruction_commitment_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    variant_map = {str(r["variant"]): r for r in variant_summary}
    q_final = query_summary[-1] if query_summary else {}
    n_final = no_tool_summary[-1] if no_tool_summary else {}
    lines = ["# Instruction-Line Commitment Audit", ""]
    for key in ["clean_full", "clean_with_corrupt_instruction", "corrupt_with_clean_instruction", "corrupt_full"]:
        row = variant_map.get(key)
        if row:
            lines.append(
                f"- `{key}`: decision `{row['decision_score_median']:.3f}`, "
                f"tool-top1 `{row['tool_top1_rate']:.3f}`, no-tool-top1 `{row['no_tool_top1_rate']:.3f}`."
            )
    lines.append("")
    if q_final:
        lines.append(
            f"- Query chain on `clean_with_corrupt_instruction`: `{q_final['nodes']}` -> "
            f"rescue `{q_final['rescue_ratio_median']:.3f}`, tool-top1 `{q_final['tool_top1_rate']:.3f}`."
        )
    if n_final:
        lines.append(
            f"- No-tool chain on `corrupt_with_clean_instruction`: `{n_final['nodes']}` -> "
            f"rescue `{n_final['rescue_ratio_median']:.3f}`, no-tool-top1 `{n_final['no_tool_top1_rate']:.3f}`."
        )
    lines.append("")
    lines.append("Interpretation:")
    lines.append(
        "- If swapping only the first instruction line moves the decision strongly, the dataset's key user-side variable is not the problem body but the instruction-level commitment cue."
    )
    lines.append(
        "- If the fixed-schema query chain rescues `clean_with_corrupt_instruction`, then that chain carries the missing commitment signal from the instruction line into the late writer."
    )
    (output_root / "instruction_commitment_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
