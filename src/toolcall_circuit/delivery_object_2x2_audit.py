#!/usr/bin/env python3
"""
Delivery-object 2x2 audit.

We construct four controlled first-line variants per sample:
- Write  x file-target
- Write  x inline-below
- Develop x file-target
- Develop x inline-below

This isolates whether early nodes group prompts by:
- opening frame (`Write` / `Develop`)
- delivery object (`in solve.py` / `below`)
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
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

if __package__ in {None, ""}:
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from toolcall_circuit.bidirectional_causal_eval import collect_cache_cpu_for_nodes, load_sample_paths
from toolcall_circuit.final_head_attention_audit import token_positions_for_char_span
from toolcall_circuit.instruction_verb_phrase_audit import split_instruction_line, split_prompt_outer
from toolcall_circuit.objective import build_bidirectional_endpoint_objectives
from toolcall_circuit.signed_edge_importance import run_logits_with_assignments
from toolcall_circuit.single_sample import load_hooked_qwen3, objective_from_logits


L2H14_HEAD = 14
L2H14_KV_INDEX = 7
FRAMES = [
    {"frame": "write", "opening": "Write"},
    {"frame": "develop", "opening": "Develop"},
]
OBJECTS = [
    {"delivery_object": "file", "clause_template": "the function body in {solve_file} based on the function definition and docstring:"},
    {"delivery_object": "answer", "clause_template": "the function body in your answer below based on the function definition and docstring:"},
]
NODES = ["L2H14", "MLP11"]
CANONICAL_VARIANTS = tuple(f"{frame['frame']}_{obj['delivery_object']}" for frame in FRAMES for obj in OBJECTS)
CANONICAL_SPACES = (
    "l2h14_lead_k",
    "l2h14_z",
    "mlp11_out",
    "l2h14_lead_k_centered",
    "l2h14_z_centered",
    "mlp11_out_centered",
)


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


def filter_rows(rows: Sequence[Dict[str, str]], *, field: str, allowed: Sequence[str]) -> List[Dict[str, str]]:
    allowed_set = set(allowed)
    return [row for row in rows if str(row.get(field, "")) in allowed_set]


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
    return ", ".join(token for token, _ in grouped.most_common(limit))


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item())


def parse_head(node: str) -> Tuple[int, int]:
    body = node[1:]
    layer_s, head_s = body.split("H")
    return int(layer_s), int(head_s)


def collect_cache(model, tokens: torch.Tensor, names: Sequence[str]) -> Dict[str, torch.Tensor]:
    wanted = set(names)
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens, names_filter=lambda n: n in wanted)
    return {k: v.detach().cpu() for k, v in cache.items()}


def detect_solve_file(line: str) -> str:
    m = re.search(r"solve\.(?:py|cpp|java)", line)
    if not m:
        raise ValueError("failed to find solve file")
    return m.group(0)


def build_prompts(clean_text: str) -> Dict[str, str]:
    prefix, user_content, suffix = split_prompt_outer(clean_text)
    line, rest = split_instruction_line(user_content)
    solve_file = detect_solve_file(line)
    prompts: Dict[str, str] = {}
    for frame in FRAMES:
        for obj in OBJECTS:
            name = f"{frame['frame']}_{obj['delivery_object']}"
            line_text = f"{frame['opening']} {obj['clause_template'].format(solve_file=solve_file)}"
            body = line_text + ("\n" + rest if rest else "")
            prompts[name] = prefix + body + suffix
    return prompts


def build_object_spans(text: str, tokenizer) -> Dict[str, List[int]]:
    prefix, user_content, _suffix = split_prompt_outer(text)
    line, _rest = split_instruction_line(user_content)
    line_start = len(prefix)

    first_space = line.find(" ")
    if first_space < 0:
        raise ValueError("missing opening space")
    opening_start = line_start
    opening_end = line_start + first_space

    m_file = re.search(r"in solve\.(?:py|cpp|java)", line)
    m_answer = re.search(r"in your answer below", line)
    if m_file:
        object_start = line_start + m_file.start()
        object_end = line_start + m_file.end()
        object_kind = "file"
    elif m_answer:
        object_start = line_start + m_answer.start()
        object_end = line_start + m_answer.end()
        object_kind = "answer"
    else:
        raise ValueError("failed to find delivery object")

    return {
        "opening": token_positions_for_char_span(text, opening_start, opening_end, tokenizer),
        "delivery_object": token_positions_for_char_span(text, object_start, object_end, tokenizer),
        "delivery_kind": [object_kind],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Delivery-object 2x2 audit.")
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

    variant_path = out_root / "delivery_object_2x2_variant_per_sample.csv"
    repr_path = out_root / "delivery_object_2x2_representation_per_sample.csv"
    align_path = out_root / "delivery_object_2x2_alignment_per_sample.csv"
    patch_path = out_root / "delivery_object_2x2_patch_per_sample.csv"
    top_path = out_root / "delivery_object_2x2_top_tokens_per_sample.csv"

    variant_rows = filter_rows(read_csv_rows(variant_path), field="variant", allowed=CANONICAL_VARIANTS)
    repr_rows = filter_rows(read_csv_rows(repr_path), field="space", allowed=CANONICAL_SPACES)
    align_rows = filter_rows(read_csv_rows(align_path), field="variant", allowed=CANONICAL_VARIANTS)
    align_rows = filter_rows(align_rows, field="space", allowed=CANONICAL_SPACES)
    patch_rows = [row for row in read_csv_rows(patch_path) if str(row.get("frame", "")) in {f["frame"] for f in FRAMES} and str(row.get("node", "")) in set(NODES)]
    top_rows = filter_rows(read_csv_rows(top_path), field="variant", allowed=CANONICAL_VARIANTS)

    variant_counts: Counter[str] = Counter()
    patch_counts: Counter[str] = Counter()
    for row in variant_rows:
        variant_counts[str(row["sample_id"])] += 1
    for row in patch_rows:
        patch_counts[str(row["sample_id"])] += 1
    processed_ids = {
        sample_id
        for sample_id, count in variant_counts.items()
        if count == len(CANONICAL_VARIANTS) and patch_counts.get(sample_id, 0) == len(FRAMES) * len(NODES)
    }

    def checkpoint() -> None:
        write_csv(variant_rows, variant_path)
        write_csv(repr_rows, repr_path)
        write_csv(align_rows, align_path)
        write_csv(patch_rows, patch_path)
        write_csv(top_rows, top_path)

    pbar = tqdm(samples, desc="Delivery object 2x2", dynamic_ncols=True)
    completed_new = 0
    for sp in pbar:
        if sp.sample_id in processed_ids:
            pbar.set_postfix(sample=sp.sample_id, resumed="skip")
            continue
        try:
            clean_text = sp.tool_call_prompt.read_text(encoding="utf-8")
            prompts = build_prompts(clean_text)
        except Exception:
            continue

        token_map: Dict[str, torch.Tensor] = {}
        logits_map: Dict[str, torch.Tensor] = {}
        cache_map: Dict[str, Dict[str, torch.Tensor]] = {}
        node_cache_map: Dict[str, Dict[str, torch.Tensor]] = {}
        spans_map: Dict[str, Dict[str, List[int]]] = {}
        lead_k_map: Dict[str, torch.Tensor] = {}
        z_map: Dict[str, torch.Tensor] = {}
        mlp11_map: Dict[str, torch.Tensor] = {}

        for frame in FRAMES:
            for obj in OBJECTS:
                name = f"{frame['frame']}_{obj['delivery_object']}"
                text = prompts[name]
                toks = model.to_tokens(text, prepend_bos=False)
                token_map[name] = toks
                with torch.no_grad():
                    logits_map[name] = model(toks)
                cache = collect_cache(model, toks, hook_names)
                cache_map[name] = cache
                node_cache_map[name] = collect_cache_cpu_for_nodes(model, toks, NODES)
                spans = build_object_spans(text, tokenizer)
                spans_map[name] = spans

                opening_positions = [int(i) for i in spans["opening"]]
                k_tensor = cache["blocks.2.attn.hook_k"][0, :, L2H14_KV_INDEX, :].float()
                lead_k_map[name] = k_tensor[opening_positions].mean(dim=0)
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
                            "frame": frame["frame"],
                            "delivery_object": obj["delivery_object"],
                            "rank": rank,
                            "pos": int(pos),
                            "token": tok,
                            "attn": float(pattern[int(pos)].item()),
                        }
                    )

        # Behavior summary on the original tool/no-tool axis.
        tool_objective, no_tool_objective = build_bidirectional_endpoint_objectives(
            logits_map["write_file"],
            logits_map["develop_answer"],
            tokenizer=tokenizer,
        )
        for frame in FRAMES:
            for obj in OBJECTS:
                name = f"{frame['frame']}_{obj['delivery_object']}"
                logits = logits_map[name]
                tool_score = float(objective_from_logits(logits, tool_objective).item())
                no_tool_score = float(objective_from_logits(logits, no_tool_objective).item())
                pattern = cache_map[name]["blocks.2.attn.hook_pattern"][0, L2H14_HEAD, -1, :].float()
                spans = spans_map[name]

                def density(idxs: Sequence[int]) -> float:
                    use = [int(i) for i in idxs if 0 <= int(i) < pattern.shape[0]]
                    return float(pattern[use].mean().item()) if use else float("nan")

                variant_rows.append(
                    {
                        "sample_id": sp.sample_id,
                        "variant": name,
                        "frame": frame["frame"],
                        "delivery_object": obj["delivery_object"],
                        "tool_score": tool_score,
                        "no_tool_score": no_tool_score,
                        "decision_score": tool_score - no_tool_score,
                        "tool_top1": int(logits[0, -1].argmax().item()) == sp.target_tool_call,
                        "no_tool_top1": int(logits[0, -1].argmax().item()) == sp.distractor,
                        "l2h14_opening_density": density(spans["opening"]),
                        "l2h14_object_density": density(spans["delivery_object"]),
                    }
                )

        # Representation grouping by delivery object vs frame.
        object_pairs = [("write_file", "develop_file"), ("write_answer", "develop_answer")]
        frame_pairs = [("write_file", "write_answer"), ("develop_file", "develop_answer")]
        raw_spaces = [("l2h14_lead_k", lead_k_map), ("l2h14_z", z_map), ("mlp11_out", mlp11_map)]
        centered_spaces = []
        canonical_names = ["write_file", "write_answer", "develop_file", "develop_answer"]
        for space_name, vec_map in raw_spaces:
            center = torch.stack([vec_map[v] for v in canonical_names], dim=0).mean(dim=0)
            centered_spaces.append((f"{space_name}_centered", {k: v - center for k, v in vec_map.items()}))

        for space_name, vec_map in raw_spaces + centered_spaces:
            same_object = [cosine(vec_map[a], vec_map[b]) for a, b in object_pairs]
            same_frame = [cosine(vec_map[a], vec_map[b]) for a, b in frame_pairs]
            repr_rows.append(
                {
                    "sample_id": sp.sample_id,
                    "space": space_name,
                    "same_object_cross_frame_cosine": float(np.mean(same_object)),
                    "same_frame_cross_object_cosine": float(np.mean(same_frame)),
                    "object_minus_frame_cosine": float(np.mean(same_object) - np.mean(same_frame)),
                    "object_wins": float(np.mean(same_object) - np.mean(same_frame)) > 0.0,
                }
            )

            file_centroid = torch.stack([vec_map["write_file"], vec_map["develop_file"]], dim=0).mean(dim=0)
            answer_centroid = torch.stack([vec_map["write_answer"], vec_map["develop_answer"]], dim=0).mean(dim=0)
            for name in canonical_names:
                obj = "file" if name.endswith("file") else "answer"
                file_sim = cosine(vec_map[name], file_centroid)
                answer_sim = cosine(vec_map[name], answer_centroid)
                align_rows.append(
                    {
                        "sample_id": sp.sample_id,
                        "space": space_name,
                        "variant": name,
                        "frame": name.split("_")[0],
                        "delivery_object": obj,
                        "object_margin": (file_sim - answer_sim) if obj == "file" else (answer_sim - file_sim),
                        "object_correct": file_sim > answer_sim if obj == "file" else answer_sim > file_sim,
                    }
                )

        # Object-axis patch within each frame.
        for frame in FRAMES:
            base_name = f"{frame['frame']}_answer"
            source_name = f"{frame['frame']}_file"
            file_objective, below_objective = build_bidirectional_endpoint_objectives(
                logits_map[source_name],
                logits_map[base_name],
                tokenizer=tokenizer,
            )
            base_file_score = float(objective_from_logits(logits_map[base_name], file_objective).item())
            base_below_score = float(objective_from_logits(logits_map[base_name], below_objective).item())
            file_gap = float(objective_from_logits(logits_map[source_name], file_objective).item()) - base_file_score
            if not math.isfinite(file_gap) or abs(file_gap) < 1e-8:
                continue
            for node in NODES:
                logits = run_logits_with_assignments(
                    model,
                    token_map[base_name],
                    node_cache_map[source_name],
                    node_cache_map[base_name],
                    [node],
                    [],
                )
                file_score = float(objective_from_logits(logits, file_objective).item())
                below_score = float(objective_from_logits(logits, below_objective).item())
                patch_rows.append(
                    {
                        "sample_id": sp.sample_id,
                        "frame": frame["frame"],
                        "node": node,
                        "file_rescue_ratio": (file_score - base_file_score) / file_gap,
                        "object_decision_score": file_score - below_score,
                        "object_boundary_flip": file_score > below_score,
                        "tool_top1": int(logits[0, -1].argmax().item()) == sp.target_tool_call,
                        "no_tool_top1": int(logits[0, -1].argmax().item()) == sp.distractor,
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
    for name in ["write_file", "write_answer", "develop_file", "develop_answer"]:
        rows = by_variant.get(name, [])
        if not rows:
            continue
        variant_summary.append(
            {
                "variant": name,
                "frame": rows[0]["frame"],
                "delivery_object": rows[0]["delivery_object"],
                "n_samples": len(rows),
                "decision_score_median": median(float(r["decision_score"]) for r in rows),
                "tool_top1_rate": safe_rate(to_bool(r["tool_top1"]) for r in rows),
                "no_tool_top1_rate": safe_rate(to_bool(r["no_tool_top1"]) for r in rows),
                "l2h14_opening_density_median": median(float(r["l2h14_opening_density"]) for r in rows),
                "l2h14_object_density_median": median(float(r["l2h14_object_density"]) for r in rows),
            }
        )

    representation_summary: List[Dict[str, object]] = []
    by_space: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in repr_rows:
        by_space[str(row["space"])].append(row)
    for space, rows in sorted(by_space.items()):
        representation_summary.append(
            {
                "space": space,
                "same_object_cross_frame_cosine_median": median(float(r["same_object_cross_frame_cosine"]) for r in rows),
                "same_frame_cross_object_cosine_median": median(float(r["same_frame_cross_object_cosine"]) for r in rows),
                "object_minus_frame_cosine_median": median(float(r["object_minus_frame_cosine"]) for r in rows),
                "object_wins_rate": safe_rate(to_bool(r["object_wins"]) for r in rows),
            }
        )

    alignment_summary: List[Dict[str, object]] = []
    by_align: Dict[Tuple[str, str], List[Dict[str, object]]] = defaultdict(list)
    for row in align_rows:
        by_align[(str(row["space"]), str(row["variant"]))].append(row)
    for (space, variant), rows in sorted(by_align.items()):
        alignment_summary.append(
            {
                "space": space,
                "variant": variant,
                "frame": rows[0]["frame"],
                "delivery_object": rows[0]["delivery_object"],
                "object_margin_median": median(float(r["object_margin"]) for r in rows),
                "object_correct_rate": safe_rate(to_bool(r["object_correct"]) for r in rows),
            }
        )

    patch_summary: List[Dict[str, object]] = []
    by_patch: Dict[Tuple[str, str], List[Dict[str, object]]] = defaultdict(list)
    for row in patch_rows:
        by_patch[(str(row["frame"]), str(row["node"]))].append(row)
    for (frame, node), rows in sorted(by_patch.items()):
        patch_summary.append(
            {
                "frame": frame,
                "node": node,
                "file_rescue_ratio_median": median(float(r["file_rescue_ratio"]) for r in rows),
                "object_decision_score_median": median(float(r["object_decision_score"]) for r in rows),
                "object_boundary_flip_rate": safe_rate(to_bool(r["object_boundary_flip"]) for r in rows),
                "tool_top1_rate": safe_rate(to_bool(r["tool_top1"]) for r in rows),
            }
        )

    top_summary: List[Dict[str, object]] = []
    grouped_top: Dict[Tuple[str, int, str], List[Dict[str, object]]] = defaultdict(list)
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

    write_csv(variant_summary, out_root / "delivery_object_2x2_variant_summary.csv")
    write_csv(representation_summary, out_root / "delivery_object_2x2_representation_summary.csv")
    write_csv(alignment_summary, out_root / "delivery_object_2x2_alignment_summary.csv")
    write_csv(patch_summary, out_root / "delivery_object_2x2_patch_summary.csv")
    write_csv(top_summary, out_root / "delivery_object_2x2_top_tokens_summary.csv")

    lines = ["# Delivery Object 2x2 Audit", ""]
    lines.append("## Main Result")
    lines.append("")
    lines.append("This audit isolates `opening frame` and `delivery object` by constructing `Write/Develop x in solve.py/below` variants.")
    lines.append("")
    lines.append("## Behavior")
    lines.append("")
    for row in variant_summary:
        lines.append(
            f"- `{row['variant']}`: decision `{row['decision_score_median']:.3f}`, tool-top1 `{row['tool_top1_rate']:.3f}`, no-tool-top1 `{row['no_tool_top1_rate']:.3f}`, L2H14 opening density `{row['l2h14_opening_density_median']:.4f}`, object density `{row['l2h14_object_density_median']:.4f}`."
        )
    lines.append("")
    lines.append("## Object vs Frame Grouping")
    lines.append("")
    for row in representation_summary:
        lines.append(
            f"- `{row['space']}`: same-object cross-frame `{row['same_object_cross_frame_cosine_median']:.3f}`, same-frame cross-object `{row['same_frame_cross_object_cosine_median']:.3f}`, gap `{row['object_minus_frame_cosine_median']:.3f}`, object-wins `{row['object_wins_rate']:.3f}`."
        )
    lines.append("")
    lines.append("## Object-Axis Patch")
    lines.append("")
    for row in patch_summary:
        lines.append(
            f"- frame `{row['frame']}` / node `{row['node']}`: file-rescue `{row['file_rescue_ratio_median']:.3f}`, object-decision `{row['object_decision_score_median']:.3f}`, boundary `{row['object_boundary_flip_rate']:.3f}`."
        )
    lines.append("")
    lines.append("## L2H14 Top Tokens")
    lines.append("")
    for variant in ["write_file", "write_answer", "develop_file", "develop_answer"]:
        lines.append(f"- `{variant}` rank-1 tokens: `{token_preview(top_rows, variant)}`")
    lines.append("")
    lines.append("## Bottom Line")
    lines.append("")
    lines.append("If `MLP11` groups same-object variants across different openings and object-axis patching at `MLP11` is stronger than at `L2H14`, then delivery-object semantics first stabilizes at `MLP11`, not `L2H14`.")
    (out_root / "delivery_object_2x2_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    summary = {
        "variant_summary_rows": variant_summary,
        "representation_summary_rows": representation_summary,
        "alignment_summary_rows": alignment_summary,
        "patch_summary_rows": patch_summary,
        "artifacts": {
            "variant_per_sample_csv": str(variant_path),
            "representation_per_sample_csv": str(repr_path),
            "alignment_per_sample_csv": str(align_path),
            "patch_per_sample_csv": str(patch_path),
            "top_tokens_per_sample_csv": str(top_path),
            "report_md": str(out_root / "delivery_object_2x2_report.md"),
        },
    }
    (out_root / "delivery_object_2x2_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
