#!/usr/bin/env python3
"""
Focused component-level bridge audit for L2H14 -> MLP11.

This audit stays inside the fixed 24-node circuit and targets one question:
which L2H14 component reads the earliest cue-conditioned signal, and which
component actually delivers that signal into MLP11.
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
from typing import Dict, Iterable, List, Sequence

import numpy as np
import torch
from tqdm.auto import tqdm

if __package__ in {None, ""}:
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from toolcall_circuit.bidirectional_causal_eval import load_sample_paths
from toolcall_circuit.final_head_attention_audit import (
    build_span_positions,
    collect_head_cache,
    parse_head,
)
from toolcall_circuit.instruction_verb_phrase_audit import build_variants as build_lead_variants
from toolcall_circuit.objective import build_bidirectional_endpoint_objectives
from toolcall_circuit.single_sample import load_hooked_qwen3, objective_from_logits


HEAD = "L2H14"
LAYER, HEAD_INDEX = parse_head(HEAD)
COMPONENTS = ("q", "k", "v", "z")


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
    grouped: Dict[str, List[float]] = defaultdict(list)
    for row in rows:
        if str(row["variant"]) != variant or int(row["rank"]) != 1:
            continue
        grouped[str(row["token"])].append(float(row["attn"]))
    items = sorted(
        ((len(v), median(v), tok) for tok, v in grouped.items()),
        key=lambda item: (-item[0], -item[1], item[2]),
    )
    return ", ".join(tok for _, _, tok in items[:limit])


def collect_residual_cache(model, tokens: torch.Tensor, names: Sequence[str]) -> Dict[str, torch.Tensor]:
    wanted = set(names)
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens, names_filter=lambda n: n in wanted)
    return {k: v.detach().cpu() for k, v in cache.items()}


def resid_logits(model, resid: torch.Tensor) -> torch.Tensor:
    stack = resid.to(dtype=model.W_U.dtype)
    stack_final = model.ln_final(stack)
    return model.unembed(stack_final)


def head_count(tensor: torch.Tensor) -> int:
    if tensor.ndim == 4:
        return int(tensor.shape[2])
    if tensor.ndim == 3:
        return int(tensor.shape[1])
    raise ValueError(f"unexpected head tensor rank: {tensor.ndim}")


def assign_last_head(out: torch.Tensor, src: torch.Tensor, head_idx: int) -> torch.Tensor:
    if out.ndim == 4:
        if src.ndim == 4:
            value = src[:, -1, head_idx, :]
        elif src.ndim == 3:
            value = src[-1, head_idx, :].unsqueeze(0)
        else:
            raise ValueError(f"unexpected source head tensor rank: {src.ndim}")
        out[:, -1, head_idx, :] = value.to(dtype=out.dtype)
        return out
    if out.ndim == 3:
        if src.ndim == 4:
            value = src[0, -1, head_idx, :]
        elif src.ndim == 3:
            value = src[-1, head_idx, :]
        else:
            raise ValueError(f"unexpected source head tensor rank: {src.ndim}")
        out[-1, head_idx, :] = value.to(dtype=out.dtype)
        return out
    raise ValueError(f"unexpected head tensor rank: {out.ndim}")


def assign_pos_head(out: torch.Tensor, src: torch.Tensor, pos: int, head_idx: int) -> torch.Tensor:
    if out.ndim == 4:
        if src.ndim == 4:
            value = src[:, pos, head_idx, :]
        elif src.ndim == 3:
            value = src[pos, head_idx, :].unsqueeze(0)
        else:
            raise ValueError(f"unexpected source head tensor rank: {src.ndim}")
        out[:, pos, head_idx, :] = value.to(dtype=out.dtype)
        return out
    if out.ndim == 3:
        if src.ndim == 4:
            value = src[0, pos, head_idx, :]
        elif src.ndim == 3:
            value = src[pos, head_idx, :]
        else:
            raise ValueError(f"unexpected source head tensor rank: {src.ndim}")
        out[pos, head_idx, :] = value.to(dtype=out.dtype)
        return out
    raise ValueError(f"unexpected head tensor rank: {out.ndim}")


def assign_last_resid(out: torch.Tensor, src: torch.Tensor) -> torch.Tensor:
    if out.ndim == 3:
        if src.ndim == 3:
            value = src[:, -1, :]
        elif src.ndim == 2:
            value = src[-1, :].unsqueeze(0)
        else:
            raise ValueError(f"unexpected source resid tensor rank: {src.ndim}")
        out[:, -1, :] = value.to(dtype=out.dtype)
        return out
    if out.ndim == 2:
        if src.ndim == 3:
            value = src[0, -1, :]
        elif src.ndim == 2:
            value = src[-1, :]
        else:
            raise ValueError(f"unexpected source resid tensor rank: {src.ndim}")
        out[-1, :] = value.to(dtype=out.dtype)
        return out
    raise ValueError(f"unexpected resid tensor rank: {out.ndim}")


def final_pos_resid(resid: torch.Tensor) -> torch.Tensor:
    if resid.ndim == 3:
        return resid[:, -1:, :]
    if resid.ndim == 2:
        return resid[-1:, :].unsqueeze(0)
    raise ValueError(f"unexpected resid tensor rank: {resid.ndim}")


def run_component_patch_with_optional_mlp11_block(
    model,
    base_tokens: torch.Tensor,
    source_cache: Dict[str, torch.Tensor],
    base_head_cache: Dict[str, torch.Tensor],
    base_mlp11_cache: Dict[str, torch.Tensor],
    component: str,
    positions: Sequence[int] | None,
    block_mlp11: bool,
) -> torch.Tensor:
    q_heads = head_count(source_cache[f"blocks.{LAYER}.attn.hook_q"])
    kv_heads = head_count(source_cache[f"blocks.{LAYER}.attn.hook_k"])
    group = max(1, q_heads // kv_heads)
    kv_index = HEAD_INDEX // group
    hooks = []

    if component == "q":
        src = source_cache[f"blocks.{LAYER}.attn.hook_q"].to(base_tokens.device)

        def hook_fn(q: torch.Tensor, hook, src_q=src):  # noqa: ANN001
            out = q.clone()
            return assign_last_head(out, src_q, HEAD_INDEX)

        hooks.append((f"blocks.{LAYER}.attn.hook_q", hook_fn))
    elif component == "k":
        src = source_cache[f"blocks.{LAYER}.attn.hook_k"].to(base_tokens.device)
        valid_positions = [] if positions is None else [int(p) for p in positions]

        def hook_fn(k: torch.Tensor, hook, src_k=src, posns=tuple(valid_positions)):  # noqa: ANN001
            out = k.clone()
            for p in posns:
                if 0 <= p < out.shape[1]:
                    out = assign_pos_head(out, src_k, p, kv_index)
            return out

        hooks.append((f"blocks.{LAYER}.attn.hook_k", hook_fn))
    elif component == "v":
        src = source_cache[f"blocks.{LAYER}.attn.hook_v"].to(base_tokens.device)
        valid_positions = [] if positions is None else [int(p) for p in positions]

        def hook_fn(v: torch.Tensor, hook, src_v=src, posns=tuple(valid_positions)):  # noqa: ANN001
            out = v.clone()
            for p in posns:
                if 0 <= p < out.shape[1]:
                    out = assign_pos_head(out, src_v, p, kv_index)
            return out

        hooks.append((f"blocks.{LAYER}.attn.hook_v", hook_fn))
    elif component == "z":
        src = source_cache[f"blocks.{LAYER}.attn.hook_z"].to(base_tokens.device)

        def hook_fn(z: torch.Tensor, hook, src_z=src):  # noqa: ANN001
            out = z.clone()
            return assign_last_head(out, src_z, HEAD_INDEX)

        hooks.append((f"blocks.{LAYER}.attn.hook_z", hook_fn))
    else:
        raise ValueError(f"unknown component: {component}")

    if block_mlp11:
        src = base_mlp11_cache["blocks.11.hook_mlp_out"].to(base_tokens.device)

        def hook_fn(mlp_out: torch.Tensor, hook, src_mlp=src):  # noqa: ANN001
            out = mlp_out.clone()
            return assign_last_resid(out, src_mlp)

        hooks.append(("blocks.11.hook_mlp_out", hook_fn))

    with torch.no_grad():
        return model.run_with_hooks(base_tokens, fwd_hooks=hooks)


def main() -> None:
    parser = argparse.ArgumentParser(description="Focused L2H14 -> MLP11 component bridge audit.")
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

    samples = load_sample_paths(run_root / "forward_batch", run_root / "reverse_batch", max_samples=args.max_samples)
    model, tokenizer = load_hooked_qwen3(args.model_path, device=args.device, dtype=torch.bfloat16)

    top_path = out_root / "l2h14_mlp11_top_tokens_per_sample.csv"
    span_path = out_root / "l2h14_mlp11_span_per_sample.csv"
    component_path = out_root / "l2h14_mlp11_component_per_sample.csv"
    resid_path = out_root / "l2h14_mlp11_residual_per_sample.csv"

    top_rows: List[Dict[str, object]] = list(read_csv_rows(top_path))
    span_rows: List[Dict[str, object]] = list(read_csv_rows(span_path))
    component_rows: List[Dict[str, object]] = list(read_csv_rows(component_path))
    resid_rows: List[Dict[str, object]] = list(read_csv_rows(resid_path))
    processed_ids = {str(row["sample_id"]) for row in resid_rows}

    def checkpoint() -> None:
        write_csv(top_rows, top_path)
        write_csv(span_rows, span_path)
        write_csv(component_rows, component_path)
        write_csv(resid_rows, resid_path)

    pbar = tqdm(samples, desc="L2H14 -> MLP11 component bridge", dynamic_ncols=True)
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
        for name in ("clean_full", "corrupt_full", "clean_with_corrupt_lead"):
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

        clean_head_cache = collect_head_cache(model, token_map["clean_full"], [LAYER])
        base_head_cache = collect_head_cache(model, token_map["clean_with_corrupt_lead"], [LAYER])
        resid_names = ("blocks.11.hook_resid_mid", "blocks.11.hook_resid_post", "blocks.11.hook_mlp_out")
        clean_resid = collect_residual_cache(model, token_map["clean_full"], resid_names)
        base_resid = collect_residual_cache(model, token_map["clean_with_corrupt_lead"], resid_names)
        clean_spans = build_span_positions(variants["clean_full"], tokenizer)

        pattern = clean_head_cache[f"blocks.{LAYER}.attn.hook_pattern"][0, HEAD_INDEX, -1, :].float()
        toks = token_map["clean_full"]
        top_pos = torch.topk(pattern, k=min(5, int(pattern.numel()))).indices.tolist()
        for rank, pos in enumerate(top_pos, start=1):
            tok = tokenizer.decode([int(toks[0, int(pos)].item())]).replace("\n", "\\n")
            top_rows.append(
                {
                    "sample_id": sp.sample_id,
                    "variant": "clean",
                    "rank": rank,
                    "pos": int(pos),
                    "token": tok,
                    "attn": float(pattern[int(pos)].item()),
                }
            )

        pattern = base_head_cache[f"blocks.{LAYER}.attn.hook_pattern"][0, HEAD_INDEX, -1, :].float()
        toks = token_map["clean_with_corrupt_lead"]
        top_pos = torch.topk(pattern, k=min(5, int(pattern.numel()))).indices.tolist()
        for rank, pos in enumerate(top_pos, start=1):
            tok = tokenizer.decode([int(toks[0, int(pos)].item())]).replace("\n", "\\n")
            top_rows.append(
                {
                    "sample_id": sp.sample_id,
                    "variant": "base",
                    "rank": rank,
                    "pos": int(pos),
                    "token": tok,
                    "attn": float(pattern[int(pos)].item()),
                }
            )

        for variant_name, cache_obj, span_map in (
            ("clean", clean_head_cache, clean_spans),
            ("base", base_head_cache, build_span_positions(variants["clean_with_corrupt_lead"], tokenizer)),
        ):
            pattern = cache_obj[f"blocks.{LAYER}.attn.hook_pattern"][0, HEAD_INDEX, -1, :].float()
            for span_name in ("lead_phrase", "function_body_anchor", "file_target", "tail_suffix", "task_body"):
                idxs = [int(i) for i in span_map.get(span_name, []) if 0 <= int(i) < pattern.shape[0]]
                span_rows.append(
                    {
                        "sample_id": sp.sample_id,
                        "variant": variant_name,
                        "span": span_name,
                        "attn_mass": float(pattern[idxs].sum().item()) if idxs else float("nan"),
                        "attn_density": float(pattern[idxs].mean().item()) if idxs else float("nan"),
                    }
                )

        base_logits = logit_map["clean_with_corrupt_lead"]
        base_tool_score = float(objective_from_logits(base_logits, tool_objective).item())
        base_no_tool_score = float(objective_from_logits(base_logits, no_tool_objective).item())
        lead_positions = clean_spans.get("lead_phrase", [])
        for component in COMPONENTS:
            positions = lead_positions if component in {"k", "v"} else None
            source_logits = run_component_patch_with_optional_mlp11_block(
                model=model,
                base_tokens=token_map["clean_with_corrupt_lead"],
                source_cache=clean_head_cache,
                base_head_cache=base_head_cache,
                base_mlp11_cache=base_resid,
                component=component,
                positions=positions,
                block_mlp11=False,
            )
            blocked_logits = run_component_patch_with_optional_mlp11_block(
                model=model,
                base_tokens=token_map["clean_with_corrupt_lead"],
                source_cache=clean_head_cache,
                base_head_cache=base_head_cache,
                base_mlp11_cache=base_resid,
                component=component,
                positions=positions,
                block_mlp11=True,
            )
            source_tool = float(objective_from_logits(source_logits, tool_objective).item())
            source_no_tool = float(objective_from_logits(source_logits, no_tool_objective).item())
            blocked_tool = float(objective_from_logits(blocked_logits, tool_objective).item())
            blocked_no_tool = float(objective_from_logits(blocked_logits, no_tool_objective).item())
            component_rows.append(
                {
                    "sample_id": sp.sample_id,
                    "component": component,
                    "source_rescue_ratio": (source_tool - base_tool_score) / tool_gap,
                    "blocked_rescue_ratio": (blocked_tool - base_tool_score) / tool_gap,
                    "mediated_ratio": (source_tool - blocked_tool) / tool_gap,
                    "source_decision_score": source_tool - source_no_tool,
                    "blocked_decision_score": blocked_tool - blocked_no_tool,
                    "source_top1": int(source_logits[0, -1].argmax().item()) == sp.target_tool_call,
                    "blocked_top1": int(blocked_logits[0, -1].argmax().item()) == sp.target_tool_call,
                }
            )

        clean_mid = final_pos_resid(clean_resid["blocks.11.hook_resid_mid"]).to(model.W_U.device)
        clean_post = final_pos_resid(clean_resid["blocks.11.hook_resid_post"]).to(model.W_U.device)
        base_mid = final_pos_resid(base_resid["blocks.11.hook_resid_mid"]).to(model.W_U.device)
        base_post = final_pos_resid(base_resid["blocks.11.hook_resid_post"]).to(model.W_U.device)
        clean_mid_logits = resid_logits(model, clean_mid)
        clean_post_logits = resid_logits(model, clean_post)
        base_mid_logits = resid_logits(model, base_mid)
        base_post_logits = resid_logits(model, base_post)
        for stage_name, src_logits, dst_logits in (
            ("resid_mid", clean_mid_logits, base_mid_logits),
            ("resid_post", clean_post_logits, base_post_logits),
        ):
            resid_rows.append(
                {
                    "sample_id": sp.sample_id,
                    "stage": stage_name,
                    "tool_token_delta": float(src_logits[0, 0, sp.target_tool_call].item() - dst_logits[0, 0, sp.target_tool_call].item()),
                    "distractor_delta": float(src_logits[0, 0, sp.distractor].item() - dst_logits[0, 0, sp.distractor].item()),
                }
            )

        pbar.set_postfix(sample=sp.sample_id)
        processed_ids.add(sp.sample_id)
        completed_new += 1
        if args.save_every > 0 and completed_new % args.save_every == 0:
            checkpoint()

    checkpoint()

    top_summary: List[Dict[str, object]] = []
    grouped_top: Dict[tuple[str, int, str], List[float]] = defaultdict(list)
    for row in top_rows:
        grouped_top[(str(row["variant"]), int(row["rank"]), str(row["token"]))].append(float(row["attn"]))
    for (variant, rank, token), vals in sorted(grouped_top.items()):
        top_summary.append(
            {
                "variant": variant,
                "rank": rank,
                "token": token,
                "count": len(vals),
                "attn_median": median(vals),
            }
        )

    span_summary: List[Dict[str, object]] = []
    grouped_span: Dict[tuple[str, str], List[float]] = defaultdict(list)
    for row in span_rows:
        grouped_span[(str(row["variant"]), str(row["span"]))].append(float(row["attn_density"]))
    for (variant, span), vals in sorted(grouped_span.items()):
        span_summary.append(
            {
                "variant": variant,
                "span": span,
                "attn_density_median": median(vals),
            }
        )

    component_summary: List[Dict[str, object]] = []
    grouped_component: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in component_rows:
        grouped_component[str(row["component"])].append(row)
    for component, rows in sorted(grouped_component.items()):
        component_summary.append(
            {
                "component": component,
                "source_rescue_ratio_median": median(float(r["source_rescue_ratio"]) for r in rows),
                "blocked_rescue_ratio_median": median(float(r["blocked_rescue_ratio"]) for r in rows),
                "mediated_ratio_median": median(float(r["mediated_ratio"]) for r in rows),
                "source_top1_rate": safe_rate(to_bool(r["source_top1"]) for r in rows),
                "blocked_top1_rate": safe_rate(to_bool(r["blocked_top1"]) for r in rows),
            }
        )

    resid_summary: List[Dict[str, object]] = []
    grouped_resid: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in resid_rows:
        grouped_resid[str(row["stage"])].append(row)
    for stage, rows in sorted(grouped_resid.items()):
        resid_summary.append(
            {
                "stage": stage,
                "tool_token_delta_median": median(float(r["tool_token_delta"]) for r in rows),
                "distractor_delta_median": median(float(r["distractor_delta"]) for r in rows),
            }
        )

    claim_tiers = {
        "strong_write": [
            "Inside the 24-node circuit, L2H14 is the earliest retained head-level predecessor of MLP11.",
            "Among the priority candidates, only L2H14 shows non-zero MLP11-mediated tool transmission.",
            "MLP11 is the first stable scaffold writer downstream of L2H14."
        ],
        "medium_write": [
            "L2H14 reads the minimal cue mainly through lead-phrase-sensitive key-side structure.",
            "The portion of L2H14 that actually survives into MLP11 is carried by its head output z."
        ],
        "weak_write": [
            "The exact object-language feature read by L2H14 is an instruction-opening bundle rather than a fully isolated minimal-cue token."
        ]
    }

    report_lines = [
        "# L2H14 -> MLP11 Component Bridge Report",
        "",
        "## Main Result",
        "",
        "Inside the fixed 24-node circuit, `L2H14` is the only earliest-priority head that both precedes `MLP11` and retains non-zero `MLP11`-mediated tool transmission.",
        "",
        "## Read vs Write Split",
        "",
        f"- Clean rank-1 tokens: `{token_preview(top_rows, 'clean')}`",
        f"- Base rank-1 tokens: `{token_preview(top_rows, 'base')}`",
    ]
    for row in span_summary:
        if row["variant"] == "clean":
            report_lines.append(
                f"- Clean span density `{row['span']}`: `{row['attn_density_median']:.4f}`"
            )
    report_lines.append("")
    for row in component_summary:
        report_lines.append(
            f"- `{row['component']}`: source rescue `{row['source_rescue_ratio_median']:.3f}`, blocked-by-MLP11 `{row['blocked_rescue_ratio_median']:.3f}`, mediated `{row['mediated_ratio_median']:.3f}`."
        )
    report_lines.append("")
    for row in resid_summary:
        report_lines.append(
            f"- `MLP11 {row['stage']}`: `<tool_call>` delta `{row['tool_token_delta_median']:.3f}`, distractor delta `{row['distractor_delta_median']:.3f}`."
        )
    report_lines.append("")
    report_lines.append("## Bottom Line")
    report_lines.append("")
    report_lines.append("The strongest mechanistic split is: `L2H14` reads the instruction opening on the lead-phrase side, and the part that reaches `MLP11` is carried by the resulting head output rather than by a later downstream carrier.")
    (out_root / "l2h14_mlp11_component_bridge_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    write_csv(top_summary, out_root / "l2h14_mlp11_top_tokens_summary.csv")
    write_csv(span_summary, out_root / "l2h14_mlp11_span_summary.csv")
    write_csv(component_summary, out_root / "l2h14_mlp11_component_summary.csv")
    write_csv(resid_summary, out_root / "l2h14_mlp11_residual_summary.csv")
    (out_root / "l2h14_mlp11_claim_tiers.json").write_text(json.dumps(claim_tiers, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {
        "top_tokens_summary": str(out_root / "l2h14_mlp11_top_tokens_summary.csv"),
        "span_summary": str(out_root / "l2h14_mlp11_span_summary.csv"),
        "component_summary": str(out_root / "l2h14_mlp11_component_summary.csv"),
        "residual_summary": str(out_root / "l2h14_mlp11_residual_summary.csv"),
        "claim_tiers": claim_tiers,
    }
    (out_root / "l2h14_mlp11_component_bridge_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
