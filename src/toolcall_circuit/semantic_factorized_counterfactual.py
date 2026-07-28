#!/usr/bin/env python3
"""
Factorized counterfactual analysis for the semantic tool-call mechanism.

This script isolates three factors in the system prompt / user prompt:

1. user request wording (clean vs corrupt)
2. tool schema availability / matching
3. tool-call protocol cue

It then evaluates:
- direct endpoint score changes under these variants
- representative head read masses
- path-level rescue for the extracted semantic branches
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm.auto import tqdm

if __package__ in {None, ""}:
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from toolcall_circuit.bidirectional_causal_eval import collect_cache_cpu_for_nodes, load_sample_paths
from toolcall_circuit.bidirectional_token_flip import run_logits_on_base_with_source
from toolcall_circuit.dataset import build_position_sets
from toolcall_circuit.objective import build_bidirectional_endpoint_objectives
from toolcall_circuit.single_sample import load_hooked_qwen3, objective_from_logits, parse_head


UNRELATED_TOOL_SCHEMA = """{"type":"function","function":{"name":"lookup_weather","description":"Look up the weather in a city.","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}"""
NO_PROTOCOL_TEXT = "Respond directly to the user in natural language. Do not emit tool-call XML."


def write_csv(rows: Sequence[Dict[str, object]], out_path: Path) -> None:
    if not rows:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


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


def safe_float(value: object, default: float = float("nan")) -> float:
    try:
        value = float(value)
    except Exception:
        return default
    return value if math.isfinite(value) else default


def read_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def split_prompt(text: str) -> tuple[str, str]:
    marker = "<|im_end|>\n<|im_start|>user\n"
    idx = text.find(marker)
    if idx < 0:
        raise ValueError("Failed to find system/user boundary.")
    system_block = text[: idx + len("<|im_end|>")]
    suffix = text[idx + len("<|im_end|>") :]
    return system_block, suffix


def replace_tools_block(system_block: str, replacement_body: str) -> str:
    return re.sub(
        r"<tools>\n.*?\n</tools>",
        f"<tools>\n{replacement_body}\n</tools>",
        system_block,
        flags=re.S,
    )


def remove_protocol_block(system_block: str) -> str:
    return re.sub(
        r"\nFor each function call, return .*?</tool_call>",
        "\n" + NO_PROTOCOL_TEXT,
        system_block,
        flags=re.S,
    )


def build_variants(clean_text: str, corrupt_text: str) -> Dict[str, str]:
    clean_system, clean_suffix = split_prompt(clean_text)
    corrupt_system, corrupt_suffix = split_prompt(corrupt_text)
    _ = corrupt_system  # system prompts should be matched already

    clean_no_schema = replace_tools_block(clean_system, "") + clean_suffix
    clean_schema_mismatch = replace_tools_block(clean_system, UNRELATED_TOOL_SCHEMA) + clean_suffix
    clean_no_protocol = remove_protocol_block(clean_system) + clean_suffix

    return {
        "clean_full": clean_text,
        "corrupt_full": corrupt_text,
        "clean_no_schema": clean_no_schema,
        "clean_schema_mismatch": clean_schema_mismatch,
        "clean_no_protocol": clean_no_protocol,
    }


def head_pattern_names(heads: Sequence[str]) -> List[str]:
    layers = sorted({parse_head(h)[0] for h in heads if h.startswith("L")})
    return [f"blocks.{layer}.attn.hook_pattern" for layer in layers]


def token_positions_for_char_span(text: str, start: int, end: int, tokenizer) -> List[int]:
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    positions: List[int] = []
    for idx, (tok_start, tok_end) in enumerate(enc["offset_mapping"]):
        if int(tok_start) < int(end) and int(tok_end) > int(start):
            positions.append(int(idx))
    return positions


def inner_payload_positions(text: str, open_tag: str, close_tag: str, tokenizer) -> List[int]:
    start = text.find(open_tag)
    end = text.find(close_tag)
    if start < 0 or end < 0 or end <= start:
        return []
    payload_start = start + len(open_tag)
    payload_end = end
    return token_positions_for_char_span(text, payload_start, payload_end, tokenizer)


def head_mass_summary(model, tokenizer, text: str, heads: Sequence[str]) -> Dict[tuple[str, str], float]:
    if not heads:
        return {}
    tokens = model.to_tokens(text, prepend_bos=False)
    ids = [int(x) for x in tokens[0].tolist()]
    pos_sets = build_position_sets(ids, ids, tokenizer, clean_text=text)
    pos_sets["tools_payload"] = inner_payload_positions(text, "<tools>", "</tools>", tokenizer)
    pos_sets["protocol_payload"] = inner_payload_positions(text, "<tool_call>", "</tool_call>", tokenizer)
    needed = set(head_pattern_names(heads))
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens, names_filter=lambda n: n in needed)
    out: Dict[tuple[str, str], float] = {}
    for head in heads:
        layer, head_idx = parse_head(head)
        pattern = cache[f"blocks.{layer}.attn.hook_pattern"][0, head_idx, -1, :]
        for set_name in ["tools_block", "tools_payload", "tool_call_tags", "protocol_payload", "user_block", "prefix_16"]:
            positions = [int(p) for p in pos_sets.get(set_name, []) if 0 <= int(p) < int(pattern.shape[0])]
            out[(head, set_name)] = float(pattern[positions].sum().item()) if positions else float("nan")
    model.reset_hooks()
    return out


def choose_representative_heads(run_root: Path) -> Dict[str, str]:
    chain = read_json(run_root / "semantic_chain" / "semantic_chain_summary.json")
    out: Dict[str, str] = {}
    for path in chain.get("paths", []):
        nodes = [n for n in path.get("nodes", []) if str(n).startswith("L")]
        if not nodes:
            continue
        if path["key"] == "query_tool_path":
            out["query_reader"] = nodes[0]
        elif path["key"] == "schema_tool_path":
            out["schema_reader"] = nodes[0]
        elif path["key"] == "no_tool_path":
            out["no_tool_reader"] = nodes[0]
    return out


def choose_paths(run_root: Path) -> Dict[str, List[str]]:
    chain = read_json(run_root / "semantic_chain" / "semantic_chain_summary.json")
    return {
        str(path["key"]): [str(x) for x in path["nodes"] if str(x) != "Residual Output: decision"]
        for path in chain.get("paths", [])
    }


def plot_variant_effects(summary_rows: Sequence[Dict[str, object]], out_path: Path) -> None:
    order = ["clean_full", "corrupt_full", "clean_no_schema", "clean_schema_mismatch", "clean_no_protocol"]
    rows = {str(r["variant"]): r for r in summary_rows}
    xs = np.arange(len(order))
    tool_vals = [safe_float(rows.get(name, {}).get("tool_endpoint_score_median")) for name in order]
    no_tool_vals = [safe_float(rows.get(name, {}).get("no_tool_endpoint_score_median")) for name in order]
    plt.style.use("default")
    fig, ax = plt.subplots(figsize=(9.2, 4.8), constrained_layout=True)
    ax.plot(xs, tool_vals, marker="o", label="tool endpoint score")
    ax.plot(xs, no_tool_vals, marker="o", label="no-tool endpoint score")
    ax.set_xticks(xs)
    ax.set_xticklabels(order, rotation=25, ha="right")
    ax.set_ylabel("Endpoint score")
    ax.set_title("Direct Endpoint Effects of Factorized Variants")
    ax.legend(frameon=False)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Factorized semantic counterfactual analysis.")
    parser.add_argument("--run-root", type=str, required=True)
    parser.add_argument("--model-path", type=str, default="./external/models/Qwen3-1.7B")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--output-root", type=str, required=True)
    args = parser.parse_args()

    run_root = Path(args.run_root).resolve()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    rep_heads = choose_representative_heads(run_root)
    paths = choose_paths(run_root)
    samples = load_sample_paths(run_root / "forward_batch", run_root / "reverse_batch", max_samples=args.max_samples)
    model, tokenizer = load_hooked_qwen3(args.model_path, device=args.device, dtype=torch.bfloat16)

    all_path_nodes = sorted({node for nodes in paths.values() for node in nodes})
    rep_head_list = sorted(rep_heads.values())

    variant_rows: List[Dict[str, object]] = []
    head_rows: List[Dict[str, object]] = []
    rescue_rows: List[Dict[str, object]] = []

    pbar = tqdm(samples, desc="Semantic factorized eval", dynamic_ncols=True)
    for sp in pbar:
        try:
            clean_text = sp.tool_call_prompt.read_text(encoding="utf-8")
            corrupt_text = sp.no_tool_prompt.read_text(encoding="utf-8")
        except Exception:
            continue

        variants = build_variants(clean_text, corrupt_text)
        logits_by_variant: Dict[str, torch.Tensor] = {}
        tokens_by_variant: Dict[str, torch.Tensor] = {}
        for name, text in variants.items():
            tokens = model.to_tokens(text, prepend_bos=False)
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

        for name, logits in logits_by_variant.items():
            variant_rows.append(
                {
                    "sample_id": sp.sample_id,
                    "variant": name,
                    "tool_endpoint_score": float(objective_from_logits(logits, tool_objective).item()),
                    "no_tool_endpoint_score": float(objective_from_logits(logits, no_tool_objective).item()),
                    "tool_top1_id": int(logits[0, -1].argmax().item()),
                    "tool_top1_is_tool_call": int(logits[0, -1].argmax().item()) == sp.target_tool_call,
                    "tool_boundary_flip": float(objective_from_logits(logits, tool_objective).item())
                    > float(objective_from_logits(logits, no_tool_objective).item()),
                }
            )

        for name, text in variants.items():
            mass = head_mass_summary(model, tokenizer, text, rep_head_list)
            for role, head in rep_heads.items():
                for set_name in ["tools_block", "tools_payload", "tool_call_tags", "protocol_payload", "user_block", "prefix_16"]:
                    head_rows.append(
                        {
                            "sample_id": sp.sample_id,
                            "variant": name,
                            "role": role,
                            "head": head,
                            "set": set_name,
                            "mass": mass.get((head, set_name), float("nan")),
                        }
                    )

        clean_cache = collect_cache_cpu_for_nodes(model, tokens_by_variant["clean_full"], all_path_nodes)
        corrupt_cache = collect_cache_cpu_for_nodes(model, tokens_by_variant["corrupt_full"], all_path_nodes)

        rescue_specs = [
            ("query_tool_path", "corrupt_full", "clean_full", "tool"),
            ("schema_tool_path", "clean_no_schema", "clean_full", "tool"),
            ("schema_tool_path", "clean_schema_mismatch", "clean_full", "tool"),
            ("schema_tool_path", "clean_no_protocol", "clean_full", "tool"),
            ("no_tool_path", "clean_full", "corrupt_full", "no_tool"),
        ]
        for path_key, base_variant, source_variant, direction in rescue_specs:
            nodes = paths.get(path_key, [])
            if not nodes:
                continue
            base_tokens = tokens_by_variant[base_variant]
            source_cache = clean_cache if source_variant == "clean_full" else corrupt_cache
            logits = run_logits_on_base_with_source(model, base_tokens, source_cache, nodes)
            if direction == "tool":
                base_score = float(objective_from_logits(logits_by_variant[base_variant], tool_objective).item())
                patched_score = float(objective_from_logits(logits, tool_objective).item())
                ratio = (patched_score - base_score) / tool_gap
                top1 = int(logits[0, -1].argmax().item()) == sp.target_tool_call
                boundary_flip = float(objective_from_logits(logits, tool_objective).item()) > float(
                    objective_from_logits(logits, no_tool_objective).item()
                )
            else:
                base_score = float(objective_from_logits(logits_by_variant[base_variant], no_tool_objective).item())
                patched_score = float(objective_from_logits(logits, no_tool_objective).item())
                ratio = (patched_score - base_score) / no_tool_gap
                top1 = int(logits[0, -1].argmax().item()) == sp.distractor
                boundary_flip = float(objective_from_logits(logits, no_tool_objective).item()) > float(
                    objective_from_logits(logits, tool_objective).item()
                )
            rescue_rows.append(
                {
                    "sample_id": sp.sample_id,
                    "path_key": path_key,
                    "base_variant": base_variant,
                    "source_variant": source_variant,
                    "direction": direction,
                    "ratio": ratio,
                    "top1": top1,
                    "boundary_flip": boundary_flip,
                }
            )
        pbar.set_postfix(sample=sp.sample_id)

    write_csv(variant_rows, output_root / "variant_effects_per_sample.csv")
    write_csv(head_rows, output_root / "representative_head_reads_per_sample.csv")
    write_csv(rescue_rows, output_root / "path_rescue_per_sample.csv")

    variant_summary: List[Dict[str, object]] = []
    for variant in sorted({str(r["variant"]) for r in variant_rows}):
        rows = [r for r in variant_rows if str(r["variant"]) == variant]
        variant_summary.append(
            {
                "variant": variant,
                "n_samples": len(rows),
                "tool_endpoint_score_median": median(safe_float(r["tool_endpoint_score"]) for r in rows),
                "no_tool_endpoint_score_median": median(safe_float(r["no_tool_endpoint_score"]) for r in rows),
                "tool_top1_rate": safe_rate(bool(r["tool_top1_is_tool_call"]) for r in rows),
                "boundary_flip_rate": safe_rate(bool(r["tool_boundary_flip"]) for r in rows),
            }
        )
    write_csv(variant_summary, output_root / "variant_effects_summary.csv")

    head_summary: List[Dict[str, object]] = []
    for role in sorted({str(r["role"]) for r in head_rows}):
        for variant in sorted({str(r["variant"]) for r in head_rows}):
            for set_name in ["tools_block", "tools_payload", "tool_call_tags", "protocol_payload", "user_block", "prefix_16"]:
                rows = [
                    r
                    for r in head_rows
                    if str(r["role"]) == role and str(r["variant"]) == variant and str(r["set"]) == set_name
                ]
                if not rows:
                    continue
                head_summary.append(
                    {
                        "role": role,
                        "head": str(rows[0]["head"]),
                        "variant": variant,
                        "set": set_name,
                        "n_samples": len(rows),
                        "mass_median": median(safe_float(r["mass"]) for r in rows),
                    }
                )
    write_csv(head_summary, output_root / "representative_head_reads_summary.csv")

    rescue_summary: List[Dict[str, object]] = []
    for path_key in sorted({str(r["path_key"]) for r in rescue_rows}):
        for base_variant in sorted({str(r["base_variant"]) for r in rescue_rows if str(r["path_key"]) == path_key}):
            rows = [
                r
                for r in rescue_rows
                if str(r["path_key"]) == path_key and str(r["base_variant"]) == base_variant
            ]
            rescue_summary.append(
                {
                    "path_key": path_key,
                    "base_variant": base_variant,
                    "direction": str(rows[0]["direction"]),
                    "n_samples": len(rows),
                    "rescue_ratio_median": median(safe_float(r["ratio"]) for r in rows),
                    "top1_rate": safe_rate(bool(r["top1"]) for r in rows),
                    "boundary_flip_rate": safe_rate(bool(r["boundary_flip"]) for r in rows),
                }
            )
    write_csv(rescue_summary, output_root / "path_rescue_summary.csv")

    plot_variant_effects(variant_summary, output_root / "variant_effects_plot.png")

    summary = {
        "representative_heads": rep_heads,
        "paths": paths,
        "variant_summary_rows": variant_summary,
        "head_summary_rows": head_summary,
        "rescue_summary_rows": rescue_summary,
        "artifacts": {
            "variant_effects_summary_csv": str(output_root / "variant_effects_summary.csv"),
            "head_reads_summary_csv": str(output_root / "representative_head_reads_summary.csv"),
            "path_rescue_summary_csv": str(output_root / "path_rescue_summary.csv"),
            "variant_effects_plot": str(output_root / "variant_effects_plot.png"),
        },
    }
    (output_root / "semantic_factorized_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    lines: List[str] = []
    lines.append("# Semantic Factorized Counterfactual Report")
    lines.append("")
    lines.append("## Direct Variant Effects")
    for row in variant_summary:
        lines.append(
            f"- `{row['variant']}`: tool `{row['tool_endpoint_score_median']:.3f}`, "
            f"no-tool `{row['no_tool_endpoint_score_median']:.3f}`, "
            f"tool-top1 `{row['tool_top1_rate']:.3f}`, boundary `{row['boundary_flip_rate']:.3f}`"
        )
    lines.append("")
    lines.append("## Representative Head Reads")
    for row in head_summary:
        lines.append(
            f"- `{row['role']}` / `{row['head']}` / `{row['variant']}` / `{row['set']}`: "
            f"`{row['mass_median']:.3f}`"
        )
    lines.append("")
    lines.append("## Path Rescue")
    for row in rescue_summary:
        lines.append(
            f"- `{row['path_key']}` on `{row['base_variant']}`: "
            f"rescue `{row['rescue_ratio_median']:.3f}`, "
            f"top1 `{row['top1_rate']:.3f}`, boundary `{row['boundary_flip_rate']:.3f}`"
        )
    lines.append("")
    markdown = "\n".join(lines) + "\n"
    (output_root / "semantic_factorized_report.md").write_text(markdown, encoding="utf-8")
    print(markdown)


if __name__ == "__main__":
    main()
