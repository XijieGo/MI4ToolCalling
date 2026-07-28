#!/usr/bin/env python3
"""
Within-opening matched counterfactual audit.

The goal is to separate three factors inside the instruction opening:
- lexical surface
- opening-frame position / syntax
- answer-delivery semantics

We keep the entire prompt fixed except for the opening phrase of the first
instruction line, and replace that opening with a matched set of tool-like and
no-tool-like variants.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

if __package__ in {None, ""}:
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from toolcall_circuit.bidirectional_causal_eval import load_sample_paths
from toolcall_circuit.final_head_attention_audit import build_span_positions
from toolcall_circuit.instruction_verb_phrase_audit import split_instruction_line, split_prompt_outer
from toolcall_circuit.objective import build_bidirectional_endpoint_objectives
from toolcall_circuit.single_sample import load_hooked_qwen3, objective_from_logits


L2H14_HEAD = 14
L2H14_KV_INDEX = 7
CANONICAL_VARIANTS = [
    {"variant": "tool_out", "semantic": "tool", "frame": "out", "opening": "Write out"},
    {"variant": "no_tool_out", "semantic": "no_tool", "frame": "out", "opening": "Build out"},
    {"variant": "tool_manually", "semantic": "tool", "frame": "manually", "opening": "Manually build"},
    {"variant": "no_tool_manually", "semantic": "no_tool", "frame": "manually", "opening": "Manually develop"},
    {"variant": "tool_properly", "semantic": "tool", "frame": "properly", "opening": "Properly add"},
    {"variant": "no_tool_properly", "semantic": "no_tool", "frame": "properly", "opening": "Properly develop"},
]
REFERENCE_VARIANTS = [
    {"variant": "original_clean", "semantic": "tool", "frame": "original", "opening": None},
    {"variant": "original_corrupt", "semantic": "no_tool", "frame": "original", "opening": None},
]
ALL_VARIANTS = REFERENCE_VARIANTS + CANONICAL_VARIANTS


def finite(values: Iterable[float]) -> List[float]:
    return [float(v) for v in values if isinstance(v, (int, float)) and math.isfinite(float(v))]


def median(values: Iterable[float]) -> float:
    vals = finite(values)
    return float(np.median(vals)) if vals else float("nan")


def safe_rate(values: Iterable[bool]) -> float:
    vals = [1.0 if bool(v) else 0.0 for v in values]
    return float(np.mean(vals)) if vals else float("nan")


def to_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes"}
    return bool(value)


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


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


def token_preview(rows: Sequence[Dict[str, object]], variant: str, limit: int = 6) -> str:
    grouped: Counter[str] = Counter()
    for row in rows:
        if str(row["variant"]) != variant or int(row["rank"]) != 1:
            continue
        grouped[str(row["token"])] += 1
    return ", ".join(token for token, _count in grouped.most_common(limit))


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item())


def collect_cache(model, tokens: torch.Tensor, names: Sequence[str]) -> Dict[str, torch.Tensor]:
    wanted = set(names)
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens, names_filter=lambda n: n in wanted)
    return {k: v.detach().cpu() for k, v in cache.items()}


def extract_object_phrase(line: str) -> str:
    m = re.search(r"the function body .*$", line)
    if not m:
        raise ValueError("failed to locate object phrase")
    return line[m.start() :]


def build_matched_prompts(clean_text: str, corrupt_text: str) -> Dict[str, str]:
    clean_prefix, clean_user, clean_suffix = split_prompt_outer(clean_text)
    _corrupt_prefix, corrupt_user, _corrupt_suffix = split_prompt_outer(corrupt_text)
    clean_line, clean_rest = split_instruction_line(clean_user)
    corrupt_line, _corrupt_rest = split_instruction_line(corrupt_user)
    object_phrase = extract_object_phrase(clean_line)
    prompts = {
        "original_clean": clean_text,
        "original_corrupt": corrupt_text,
    }
    for meta in CANONICAL_VARIANTS:
        line = f"{meta['opening']} {object_phrase}"
        user = line + ("\n" + clean_rest if clean_rest else "")
        prompts[str(meta["variant"])] = clean_prefix + user + clean_suffix
    return prompts


def main() -> None:
    parser = argparse.ArgumentParser(description="Within-opening matched counterfactual audit.")
    parser.add_argument("--run-root", type=str, required=True)
    parser.add_argument("--model-path", type=str, default="./external/models/Qwen3-1.7B")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--output-root", type=str, required=True)
    parser.add_argument("--save-every", type=int, default=25)
    args = parser.parse_args()

    run_root = Path(args.run_root).resolve()
    out_root = Path(args.output_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    samples = load_sample_paths(run_root / "forward_batch", run_root / "reverse_batch", max_samples=args.max_samples)
    model, tokenizer = load_hooked_qwen3(args.model_path, device=args.device, dtype=torch.bfloat16)

    hook_names = [
        "blocks.2.attn.hook_pattern",
        "blocks.2.attn.hook_k",
        "blocks.2.attn.hook_z",
        "blocks.11.hook_mlp_out",
    ]

    variant_path = out_root / "opening_matched_variant_per_sample.csv"
    pair_path = out_root / "opening_matched_representation_per_sample.csv"
    align_path = out_root / "opening_matched_alignment_per_sample.csv"
    top_path = out_root / "opening_matched_top_tokens_per_sample.csv"

    variant_rows: List[Dict[str, object]] = list(read_csv_rows(variant_path))
    pair_rows: List[Dict[str, object]] = list(read_csv_rows(pair_path))
    align_rows: List[Dict[str, object]] = list(read_csv_rows(align_path))
    top_rows: List[Dict[str, object]] = list(read_csv_rows(top_path))
    processed_ids = {str(r["sample_id"]) for r in align_rows}

    def checkpoint() -> None:
        write_csv(variant_rows, variant_path)
        write_csv(pair_rows, pair_path)
        write_csv(align_rows, align_path)
        write_csv(top_rows, top_path)

    pbar = tqdm(samples, desc="Opening matched semantics", dynamic_ncols=True)
    completed_new = 0
    for sp in pbar:
        if sp.sample_id in processed_ids:
            pbar.set_postfix(sample=sp.sample_id, resumed="skip")
            continue
        try:
            clean_text = sp.tool_call_prompt.read_text(encoding="utf-8")
            corrupt_text = sp.no_tool_prompt.read_text(encoding="utf-8")
            prompt_map = build_matched_prompts(clean_text, corrupt_text)
        except Exception:
            continue

        token_map: Dict[str, torch.Tensor] = {}
        logits_map: Dict[str, torch.Tensor] = {}
        cache_map: Dict[str, Dict[str, torch.Tensor]] = {}
        spans_map: Dict[str, Dict[str, List[int]]] = {}
        lead_k_map: Dict[str, torch.Tensor] = {}
        z_map: Dict[str, torch.Tensor] = {}
        mlp11_map: Dict[str, torch.Tensor] = {}

        for meta in ALL_VARIANTS:
            name = str(meta["variant"])
            text = prompt_map[name]
            toks = model.to_tokens(text, prepend_bos=False)
            token_map[name] = toks
            with torch.no_grad():
                logits_map[name] = model(toks)
            cache = collect_cache(model, toks, hook_names)
            cache_map[name] = cache
            spans_map[name] = build_span_positions(text, tokenizer)
            lead_positions = [int(i) for i in spans_map[name].get("lead_phrase", [])]
            k_tensor = cache["blocks.2.attn.hook_k"][0, :, L2H14_KV_INDEX, :].float()
            if lead_positions:
                lead_k_map[name] = k_tensor[lead_positions].mean(dim=0)
            else:
                lead_k_map[name] = k_tensor[0]
            z_map[name] = cache["blocks.2.attn.hook_z"][0, -1, L2H14_HEAD, :].float()
            mlp11_map[name] = cache["blocks.11.hook_mlp_out"][0, -1, :].float()

            pattern = cache["blocks.2.attn.hook_pattern"][0, L2H14_HEAD, -1, :].float()
            top_pos = torch.topk(pattern, k=min(5, int(pattern.numel()))).indices.tolist()
            for rank, pos in enumerate(top_pos, start=1):
                tok = tokenizer.decode([int(toks[0, int(pos)].item())]).replace("\n", "\\n")
                top_rows.append(
                    {
                        "sample_id": sp.sample_id,
                        "variant": name,
                        "semantic": meta["semantic"],
                        "frame": meta["frame"],
                        "rank": rank,
                        "pos": int(pos),
                        "token": tok,
                        "attn": float(pattern[int(pos)].item()),
                    }
                )

        tool_objective, no_tool_objective = build_bidirectional_endpoint_objectives(
            logits_map["original_clean"],
            logits_map["original_corrupt"],
            tokenizer=tokenizer,
        )

        for meta in ALL_VARIANTS:
            name = str(meta["variant"])
            logits = logits_map[name]
            tool_score = float(objective_from_logits(logits, tool_objective).item())
            no_tool_score = float(objective_from_logits(logits, no_tool_objective).item())
            pattern = cache_map[name]["blocks.2.attn.hook_pattern"][0, L2H14_HEAD, -1, :].float()
            spans = spans_map[name]

            def density(span_name: str) -> float:
                idxs = [int(i) for i in spans.get(span_name, []) if 0 <= int(i) < pattern.shape[0]]
                return float(pattern[idxs].mean().item()) if idxs else float("nan")

            variant_rows.append(
                {
                    "sample_id": sp.sample_id,
                    "variant": name,
                    "semantic": meta["semantic"],
                    "frame": meta["frame"],
                    "tool_score": tool_score,
                    "no_tool_score": no_tool_score,
                    "decision_score": tool_score - no_tool_score,
                    "tool_top1": int(logits[0, -1].argmax().item()) == sp.target_tool_call,
                    "no_tool_top1": int(logits[0, -1].argmax().item()) == sp.distractor,
                    "l2h14_lead_density": density("lead_phrase"),
                    "l2h14_file_density": density("file_target"),
                    "l2h14_function_density": density("function_body_anchor"),
                    "l2h14_tail_density": density("tail_suffix"),
                }
            )

        tool_variants = [m["variant"] for m in CANONICAL_VARIANTS if m["semantic"] == "tool"]
        no_tool_variants = [m["variant"] for m in CANONICAL_VARIANTS if m["semantic"] == "no_tool"]
        frame_pairs = [
            ("tool_out", "no_tool_out"),
            ("tool_manually", "no_tool_manually"),
            ("tool_properly", "no_tool_properly"),
        ]
        raw_spaces = [("l2h14_lead_k", lead_k_map), ("l2h14_z", z_map), ("mlp11_out", mlp11_map)]
        centered_spaces = []
        canonical_names = [str(m["variant"]) for m in CANONICAL_VARIANTS]
        for space_name, vec_map in raw_spaces:
            center = torch.stack([vec_map[v] for v in canonical_names], dim=0).mean(dim=0)
            centered_spaces.append((f"{space_name}_centered", {k: v - center for k, v in vec_map.items()}))

        for space_name, vec_map in raw_spaces + centered_spaces:
            same_sem = []
            for group in [tool_variants, no_tool_variants]:
                for i in range(len(group)):
                    for j in range(i + 1, len(group)):
                        same_sem.append(cosine(vec_map[group[i]], vec_map[group[j]]))
            same_frame = [cosine(vec_map[a], vec_map[b]) for a, b in frame_pairs]
            pair_rows.append(
                {
                    "sample_id": sp.sample_id,
                    "space": space_name,
                    "same_semantic_cross_frame_cosine": float(np.mean(same_sem)),
                    "same_frame_opposite_semantic_cosine": float(np.mean(same_frame)),
                    "semantic_minus_frame_cosine": float(np.mean(same_sem) - np.mean(same_frame)),
                    "semantic_wins": float(np.mean(same_sem) - np.mean(same_frame)) > 0.0,
                }
            )

            for meta in CANONICAL_VARIANTS:
                name = str(meta["variant"])
                same_group = [v for v in (tool_variants if meta["semantic"] == "tool" else no_tool_variants) if v != name]
                other_group = no_tool_variants if meta["semantic"] == "tool" else tool_variants
                same_centroid = torch.stack([vec_map[v] for v in same_group], dim=0).mean(dim=0)
                other_centroid = torch.stack([vec_map[v] for v in other_group], dim=0).mean(dim=0)
                same_sim = cosine(vec_map[name], same_centroid)
                other_sim = cosine(vec_map[name], other_centroid)
                align_rows.append(
                    {
                        "sample_id": sp.sample_id,
                        "space": space_name,
                        "variant": name,
                        "semantic": meta["semantic"],
                        "frame": meta["frame"],
                        "same_centroid_similarity": same_sim,
                        "other_centroid_similarity": other_sim,
                        "semantic_margin": same_sim - other_sim,
                        "semantic_correct": same_sim > other_sim,
                    }
                )

            tool_centroid = torch.stack([vec_map[v] for v in tool_variants], dim=0).mean(dim=0)
            no_tool_centroid = torch.stack([vec_map[v] for v in no_tool_variants], dim=0).mean(dim=0)
            for meta in REFERENCE_VARIANTS:
                name = str(meta["variant"])
                tool_sim = cosine(vec_map[name], tool_centroid)
                no_tool_sim = cosine(vec_map[name], no_tool_centroid)
                align_rows.append(
                    {
                        "sample_id": sp.sample_id,
                        "space": space_name,
                        "variant": name,
                        "semantic": meta["semantic"],
                        "frame": meta["frame"],
                        "same_centroid_similarity": tool_sim if meta["semantic"] == "tool" else no_tool_sim,
                        "other_centroid_similarity": no_tool_sim if meta["semantic"] == "tool" else tool_sim,
                        "semantic_margin": (tool_sim - no_tool_sim) if meta["semantic"] == "tool" else (no_tool_sim - tool_sim),
                        "semantic_correct": tool_sim > no_tool_sim if meta["semantic"] == "tool" else no_tool_sim > tool_sim,
                    }
                )

        pbar.set_postfix(sample=sp.sample_id)
        processed_ids.add(sp.sample_id)
        completed_new += 1
        if args.save_every > 0 and completed_new % args.save_every == 0:
            checkpoint()

    checkpoint()

    variant_summary: List[Dict[str, object]] = []
    by_variant: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in variant_rows:
        by_variant[str(row["variant"])].append(row)
    for meta in ALL_VARIANTS:
        name = str(meta["variant"])
        rows = by_variant.get(name, [])
        if not rows:
            continue
        variant_summary.append(
            {
                "variant": name,
                "semantic": meta["semantic"],
                "frame": meta["frame"],
                "n_samples": len(rows),
                "decision_score_median": median(float(r["decision_score"]) for r in rows),
                "tool_top1_rate": safe_rate(to_bool(r["tool_top1"]) for r in rows),
                "no_tool_top1_rate": safe_rate(to_bool(r["no_tool_top1"]) for r in rows),
                "l2h14_lead_density_median": median(float(r["l2h14_lead_density"]) for r in rows),
                "l2h14_file_density_median": median(float(r["l2h14_file_density"]) for r in rows),
                "l2h14_function_density_median": median(float(r["l2h14_function_density"]) for r in rows),
            }
        )

    representation_summary: List[Dict[str, object]] = []
    by_space: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in pair_rows:
        by_space[str(row["space"])].append(row)
    for space, rows in sorted(by_space.items()):
        representation_summary.append(
            {
                "space": space,
                "same_semantic_cross_frame_cosine_median": median(float(r["same_semantic_cross_frame_cosine"]) for r in rows),
                "same_frame_opposite_semantic_cosine_median": median(float(r["same_frame_opposite_semantic_cosine"]) for r in rows),
                "semantic_minus_frame_cosine_median": median(float(r["semantic_minus_frame_cosine"]) for r in rows),
                "semantic_wins_rate": safe_rate(to_bool(r["semantic_wins"]) for r in rows),
            }
        )

    alignment_summary: List[Dict[str, object]] = []
    by_align: Dict[tuple[str, str], List[Dict[str, object]]] = defaultdict(list)
    for row in align_rows:
        by_align[(str(row["space"]), str(row["variant"]))].append(row)
    for (space, variant), rows in sorted(by_align.items()):
        alignment_summary.append(
            {
                "space": space,
                "variant": variant,
                "semantic": rows[0]["semantic"],
                "frame": rows[0]["frame"],
                "semantic_margin_median": median(float(r["semantic_margin"]) for r in rows),
                "semantic_correct_rate": safe_rate(to_bool(r["semantic_correct"]) for r in rows),
            }
        )

    top_summary: List[Dict[str, object]] = []
    grouped_top: Dict[tuple[str, int, str], List[Dict[str, object]]] = defaultdict(list)
    for row in top_rows:
        grouped_top[(str(row["variant"]), int(row["rank"]), str(row["token"]))].append(row)
    for (variant, rank, token), rows in sorted(grouped_top.items()):
        top_summary.append(
            {
                "variant": variant,
                "rank": rank,
                "token": token,
                "count": len(rows),
                "attn_median": median(float(r["attn"]) for r in rows),
            }
        )

    write_csv(variant_summary, out_root / "opening_matched_variant_summary.csv")
    write_csv(representation_summary, out_root / "opening_matched_representation_summary.csv")
    write_csv(alignment_summary, out_root / "opening_matched_alignment_summary.csv")
    write_csv(top_summary, out_root / "opening_matched_top_tokens_summary.csv")

    lines = ["# Within-Opening Matched Counterfactual Audit", ""]
    lines.append("## Main Result")
    lines.append("")
    lines.append("This audit holds the full prompt fixed and changes only the instruction opening with matched tool-like and no-tool-like openings.")
    lines.append("It asks whether `L2H14` and `MLP11` group variants by semantic class or by local opening frame.")
    lines.append("")
    lines.append("## Behavior")
    lines.append("")
    for row in variant_summary:
        lines.append(
            f"- `{row['variant']}`: decision `{row['decision_score_median']:.3f}`, tool-top1 `{row['tool_top1_rate']:.3f}`, no-tool-top1 `{row['no_tool_top1_rate']:.3f}`, L2H14 lead density `{row['l2h14_lead_density_median']:.4f}`."
        )
    lines.append("")
    lines.append("## Semantic vs Frame Clustering")
    lines.append("")
    for row in representation_summary:
        lines.append(
            f"- `{row['space']}`: same-semantic cross-frame cosine `{row['same_semantic_cross_frame_cosine_median']:.3f}`, same-frame opposite-semantic cosine `{row['same_frame_opposite_semantic_cosine_median']:.3f}`, gap `{row['semantic_minus_frame_cosine_median']:.3f}`, semantic-wins `{row['semantic_wins_rate']:.3f}`."
        )
    lines.append("")
    lines.append("## Centroid Alignment")
    lines.append("")
    for variant in ["original_clean", "original_corrupt", "tool_out", "no_tool_out", "tool_manually", "no_tool_manually", "tool_properly", "no_tool_properly"]:
        for row in alignment_summary:
            if row["variant"] != variant:
                continue
            lines.append(
                f"- `{row['space']}` / `{variant}`: semantic-margin `{row['semantic_margin_median']:.3f}`, semantic-correct `{row['semantic_correct_rate']:.3f}`."
            )
    lines.append("")
    lines.append("## L2H14 Top Tokens")
    lines.append("")
    for variant in [m["variant"] for m in ALL_VARIANTS]:
        lines.append(f"- `{variant}` rank-1 tokens: `{token_preview(top_rows, str(variant))}`")
    lines.append("")
    lines.append("## Bottom Line")
    lines.append("")
    lines.append("If `L2H14` and `MLP11` group same-semantic openings across different local frames more tightly than matched frame-opposites, then the earliest reader is tracking opening semantics rather than only lexical surface.")
    (out_root / "opening_matched_counterfactual_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    summary = {
        "variant_summary_rows": variant_summary,
        "representation_summary_rows": representation_summary,
        "alignment_summary_rows": alignment_summary,
        "artifacts": {
            "variant_per_sample_csv": str(variant_path),
            "representation_per_sample_csv": str(pair_path),
            "alignment_per_sample_csv": str(align_path),
            "top_tokens_per_sample_csv": str(top_path),
            "report_md": str(out_root / "opening_matched_counterfactual_report.md"),
        },
    }
    (out_root / "opening_matched_counterfactual_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
