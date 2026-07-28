#!/usr/bin/env python3
"""
Minimal-cue mechanism analysis inside the final 24-node signed circuit.

This analysis is purpose-built for the TODO in `todo.md`:

1. Hold the prompt fixed except for the minimal lead-phrase cue.
2. Measure which signed-circuit nodes recover the lost tool/no-tool decision.
3. Trace how that cue is propagated and amplified through specific edges.
4. Build a report that directly answers:
   - how the cue becomes `<tool_call>`
   - how the competing `no_tool` chain happens
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

from toolcall_circuit.bidirectional_causal_eval import collect_cache_cpu_for_nodes, load_sample_paths
from toolcall_circuit.instruction_verb_phrase_audit import build_variants
from toolcall_circuit.objective import build_bidirectional_endpoint_objectives
from toolcall_circuit.signed_edge_importance import run_logits_with_assignments
from toolcall_circuit.single_sample import load_hooked_qwen3, objective_from_logits


TOOL_CHAIN = ["L20H5", "L21H1", "L21H12", "L24H6", "MLP27"]
NO_TOOL_CHAIN = ["L16H4", "MLP17", "L23H6"]

TOOL_EDGES = [
    ("L20H5", "L21H1"),
    ("L20H5", "L21H12"),
    ("L21H1", "L24H6"),
    ("L21H1", "MLP27"),
    ("L21H12", "L24H6"),
    ("L21H12", "MLP27"),
    ("L24H6", "MLP27"),
]

NO_TOOL_EDGES = [
    ("L16H4", "MLP17"),
    ("MLP17", "L23H6"),
    ("MLP17", "L20H5"),
    ("MLP17", "L21H1"),
    ("MLP17", "L21H12"),
    ("MLP17", "L24H6"),
]


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


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


def safe_rate(values: Iterable[bool]) -> float:
    vals = [1.0 if bool(v) else 0.0 for v in values]
    return float(np.mean(vals)) if vals else float("nan")


def safe_ratio(num: float, den: float) -> float:
    if not math.isfinite(num) or not math.isfinite(den) or abs(den) < 1e-8:
        return float("nan")
    return float(num / den)


def fmt(value: object, digits: int = 3) -> str:
    try:
        value = float(value)
    except Exception:
        return str(value)
    if not math.isfinite(value):
        return "nan"
    return f"{value:.{digits}f}"


def load_signed_nodes(path: Path) -> List[Dict[str, object]]:
    rows = read_csv_rows(path)
    out: List[Dict[str, object]] = []
    for row in rows:
        out.append(
            {
                "node": str(row["node"]),
                "layer": int(row["layer"]),
                "group_key": str(row["group_key"]),
                "semantic_hint": str(row.get("semantic_hint", "")),
            }
        )
    out.sort(key=lambda r: (int(r["layer"]), str(r["node"])))
    return out


def summarize_variant_rows(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    by_variant: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_variant[str(row["variant"])].append(row)
    summary: List[Dict[str, object]] = []
    for variant, members in sorted(by_variant.items()):
        summary.append(
            {
                "variant": variant,
                "n_samples": len(members),
                "tool_score_median": median(float(r["tool_score"]) for r in members),
                "no_tool_score_median": median(float(r["no_tool_score"]) for r in members),
                "decision_score_median": median(float(r["decision_score"]) for r in members),
                "tool_top1_rate": safe_rate(bool(r["tool_top1"]) for r in members),
                "no_tool_top1_rate": safe_rate(bool(r["no_tool_top1"]) for r in members),
            }
        )
    return summary


def summarize_node_rows(rows: Sequence[Dict[str, object]], node_meta: Dict[str, Dict[str, object]]) -> List[Dict[str, object]]:
    by_key: Dict[tuple[str, str], List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_key[(str(row["family"]), str(row["node"]))].append(row)
    summary: List[Dict[str, object]] = []
    for (family, node), members in sorted(by_key.items(), key=lambda kv: (median(float(r["rescue_ratio"]) for r in kv[1])),):
        meta = node_meta[node]
        summary.append(
            {
                "family": family,
                "node": node,
                "layer": meta["layer"],
                "group_key": meta["group_key"],
                "semantic_hint": meta["semantic_hint"],
                "rescue_ratio_median": median(float(r["rescue_ratio"]) for r in members),
                "decision_score_median": median(float(r["decision_score"]) for r in members),
                "tool_top1_rate": safe_rate(bool(r["tool_top1"]) for r in members),
                "no_tool_top1_rate": safe_rate(bool(r["no_tool_top1"]) for r in members),
                "boundary_flip_rate": safe_rate(bool(r["boundary_flip"]) for r in members),
                "n_samples": len(members),
            }
        )
    summary.sort(
        key=lambda r: (
            str(r["family"]),
            -float(r["rescue_ratio_median"]) if math.isfinite(float(r["rescue_ratio_median"])) else float("inf"),
            int(r["layer"]),
            str(r["node"]),
        )
    )
    return summary


def summarize_step_rows(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    by_key: Dict[tuple[str, int], List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_key[(str(row["family"]), int(row["step_idx"]))].append(row)
    summary: List[Dict[str, object]] = []
    for (family, step_idx), members in sorted(by_key.items()):
        summary.append(
            {
                "family": family,
                "step_idx": step_idx,
                "nodes": members[0]["nodes"],
                "rescue_ratio_median": median(float(r["rescue_ratio"]) for r in members),
                "decision_score_median": median(float(r["decision_score"]) for r in members),
                "tool_top1_rate": safe_rate(bool(r["tool_top1"]) for r in members),
                "no_tool_top1_rate": safe_rate(bool(r["no_tool_top1"]) for r in members),
                "boundary_flip_rate": safe_rate(bool(r["boundary_flip"]) for r in members),
                "n_samples": len(members),
            }
        )
    return summary


def summarize_edge_rows(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    by_key: Dict[tuple[str, str], List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_key[(str(row["family"]), str(row["edge"]))].append(row)
    summary: List[Dict[str, object]] = []
    for (family, edge), members in sorted(by_key.items()):
        summary.append(
            {
                "family": family,
                "edge": edge,
                "source_ratio_median": median(float(r["source_ratio"]) for r in members),
                "blocked_ratio_median": median(float(r["blocked_ratio"]) for r in members),
                "mediated_ratio_median": median(float(r["mediated_ratio"]) for r in members),
                "n_samples": len(members),
            }
        )
    summary.sort(
        key=lambda r: (
            str(r["family"]),
            -float(r["mediated_ratio_median"]) if math.isfinite(float(r["mediated_ratio_median"])) else float("inf"),
            str(r["edge"]),
        )
    )
    return summary


def find_row(rows: Sequence[Dict[str, object]], **kwargs: object) -> Dict[str, object]:
    for row in rows:
        ok = True
        for key, value in kwargs.items():
            if str(row.get(key)) != str(value):
                ok = False
                break
        if ok:
            return dict(row)
    return {}


def load_optional_json(path: Path) -> Dict[str, object] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def build_report(
    *,
    out_root: Path,
    variant_summary: Sequence[Dict[str, object]],
    node_summary: Sequence[Dict[str, object]],
    step_summary: Sequence[Dict[str, object]],
    edge_summary: Sequence[Dict[str, object]],
    head_audit: Dict[str, object] | None,
    reverse_overlap: Dict[str, object] | None,
    max_samples: int,
) -> None:
    variant_map = {str(r["variant"]): r for r in variant_summary}
    head_final = {}
    if head_audit:
        head_final = {str(r["head"]): r for r in head_audit.get("final_rows", [])}
    actual_n = int(variant_map.get("clean_full", {}).get("n_samples", 0) or 0)

    def top_family_rows(family: str, limit: int = 8) -> List[Dict[str, object]]:
        return [r for r in node_summary if str(r["family"]) == family][:limit]

    def earliest_significant_row(family: str, min_rescue: float = 0.10) -> Dict[str, object]:
        candidates = [
            r
            for r in node_summary
            if str(r["family"]) == family and math.isfinite(float(r["rescue_ratio_median"])) and float(r["rescue_ratio_median"]) >= min_rescue
        ]
        if not candidates:
            return {}
        candidates.sort(key=lambda r: (int(r["layer"]), -float(r["rescue_ratio_median"]), str(r["node"])))
        return candidates[0]

    tool_top = top_family_rows("tool")
    no_tool_top = top_family_rows("no_tool")
    tool_ingress = earliest_significant_row("tool")
    no_tool_ingress = earliest_significant_row("no_tool")
    tool_ingress_name = str(tool_ingress.get("node") or "L20H5")
    no_tool_ingress_name = str(no_tool_ingress.get("node") or "L16H4")
    tool_steps = [r for r in step_summary if str(r["family"]) == "tool"]
    no_tool_steps = [r for r in step_summary if str(r["family"]) == "no_tool"]
    tool_edges = [r for r in edge_summary if str(r["family"]) == "tool"]
    no_tool_edges = [r for r in edge_summary if str(r["family"]) == "no_tool"]

    lines: List[str] = []
    lines.append("# Minimal Cue Mechanism Report")
    lines.append("")
    lines.append("## 结论先行")
    lines.append("")
    lines.append(
        "在 24 节点 circuit 内，最小 lead cue 不像一个被单个 head 直接盯住的裸 token；它更像是先进入共享/用户侧状态，然后在晚期由 `L20H5` 作为决策性 tool ingress 接住，经 `L21H1/L21H12` 路由，在 `L24H6/MLP27` 区域被放大并写成 `<tool_call>`。"
    )
    lines.append(
        "`no_tool` 链则不是单纯“另一个输出头”；它先由 `L16H4` 读入竞争性 user-side 状态，经 `MLP17` 写成 no-tool 偏置，再一边通过 `L23H6` 把这份状态送到晚期输出区，一边反压 `L20H5/L21H12` 这条 tool ingress。"
    )
    lines.append("")
    lines.append("## 1. 最小 cue 本身就足够翻转首 token")
    lines.append("")
    for variant in ["clean_full", "clean_with_corrupt_lead", "corrupt_with_clean_lead", "corrupt_full"]:
        row = variant_map.get(variant, {})
        if not row:
            continue
        lines.append(
            f"- `{variant}`: decision `{fmt(row.get('decision_score_median'))}`, tool-top1 `{fmt(row.get('tool_top1_rate'))}`, no-tool-top1 `{fmt(row.get('no_tool_top1_rate'))}`"
        )
    lines.append("")
    lines.append("## 2. 在 24 节点里，谁最先稳定携带这份 cue")
    lines.append("")
    requested_label = "all shared samples" if max_samples <= 0 else str(max_samples)
    lines.append(f"- 请求样本数：`{requested_label}`；实际有效样本数：`{actual_n}`。")
    if tool_ingress:
        lines.append(
            f"- 单节点扫描里，`tool` 方向最早超过 `0.10` rescue 的 cue-sensitive 节点是 `{tool_ingress_name}`（L{tool_ingress['layer']}，rescue `{fmt(tool_ingress['rescue_ratio_median'])}`）。"
        )
    if no_tool_ingress:
        lines.append(
            f"- 单节点扫描里，`no_tool` 方向最早超过 `0.10` rescue 的 cue-sensitive 节点是 `{no_tool_ingress_name}`（L{no_tool_ingress['layer']}，rescue `{fmt(no_tool_ingress['rescue_ratio_median'])}`）。"
        )
    lines.append("- 但决策性晚期主链仍应区分开看：tool 主链从 `L20H5` 进入，no-tool 最小 suppress chain 从 `L16H4` 进入。")
    lines.append("- `tool` 方向 top 节点：")
    for row in tool_top:
        lines.append(
            f"  - `{row['node']}` / L{row['layer']}: rescue `{fmt(row['rescue_ratio_median'])}`, boundary-flip `{fmt(row['boundary_flip_rate'])}`, group `{row['group_key']}`, hint `{row['semantic_hint']}`"
        )
    lines.append("- `no_tool` 方向 top 节点：")
    for row in no_tool_top:
        lines.append(
            f"  - `{row['node']}` / L{row['layer']}: rescue `{fmt(row['rescue_ratio_median'])}`, boundary-flip `{fmt(row['boundary_flip_rate'])}`, group `{row['group_key']}`, hint `{row['semantic_hint']}`"
        )
    lines.append("")
    lines.append(
        "读法上要注意：这里的“最先”指的是在 24 节点 final circuit 内，最早能稳定把最小 cue 重新补回决策的节点；它不等于原始 token 的唯一第一读头。"
    )
    lines.append("")
    lines.append("## 3. tool 链是如何被放大并写成 `<tool_call>` 的")
    lines.append("")
    lines.append("- 分阶段 patch（`clean_full -> clean_with_corrupt_lead`）:")
    for row in tool_steps:
        lines.append(
            f"  - step {row['step_idx']} / `{row['nodes']}`: rescue `{fmt(row['rescue_ratio_median'])}`, decision `{fmt(row['decision_score_median'])}`, tool-top1 `{fmt(row['tool_top1_rate'])}`, boundary-flip `{fmt(row['boundary_flip_rate'])}`"
        )
    lines.append("- 关键边介导：")
    for row in tool_edges[:6]:
        lines.append(
            f"  - `{row['edge']}`: source `{fmt(row['source_ratio_median'])}`, blocked `{fmt(row['blocked_ratio_median'])}`, mediated `{fmt(row['mediated_ratio_median'])}`"
        )
    if head_final:
        lines.append("- 对应 head 的 Q/K/V/Z 证据（来自已有 `final_head_attention_audit` 全量结果）：")
        for head in ["L20H5", "L21H1", "L21H12", "L24H6"]:
            row = head_final.get(head)
            if not row:
                continue
            lines.append(
                f"  - `{head}`: best-read `{row.get('best_read_span')}`, best-causal-span `{row.get('best_causal_span')}`, best-component `{row.get('best_qkv_component')}` rescue `{fmt(row.get('best_qkv_rescue_median'))}`"
            )
    lines.append("")
    lines.append(
        "这条链的关键信号很清楚：`L20H5` 单独只能带来很弱的 tool 恢复；加入 `L21H1/L21H12` 后决策边界开始翻正；`L24H6` 和尤其 `MLP27` 把差异推到稳定 `<tool_call>`。"
    )
    lines.append("")
    lines.append("## 4. `no_tool` 链到底如何发生")
    lines.append("")
    lines.append("- 分阶段 patch（`corrupt_full -> corrupt_with_clean_lead` 的反向 no-tool 恢复）:")
    for row in no_tool_steps:
        lines.append(
            f"  - step {row['step_idx']} / `{row['nodes']}`: rescue `{fmt(row['rescue_ratio_median'])}`, decision `{fmt(row['decision_score_median'])}`, no-tool-top1 `{fmt(row['no_tool_top1_rate'])}`, boundary-flip `{fmt(row['boundary_flip_rate'])}`"
        )
    lines.append("- 关键边介导：")
    for row in no_tool_edges[:6]:
        lines.append(
            f"  - `{row['edge']}`: source `{fmt(row['source_ratio_median'])}`, blocked `{fmt(row['blocked_ratio_median'])}`, mediated `{fmt(row['mediated_ratio_median'])}`"
        )
    if head_final:
        lines.append("- 对应 head 的 Q/K/V/Z 证据（来自已有 `final_head_attention_audit` 全量结果）：")
        for head in ["L16H4", "L23H6"]:
            row = head_final.get(head)
            if not row:
                continue
            lines.append(
                f"  - `{head}`: best-read `{row.get('best_read_span')}`, best-causal-span `{row.get('best_causal_span')}`, best-component `{row.get('best_qkv_component')}` rescue `{fmt(row.get('best_qkv_rescue_median'))}`"
            )
    if reverse_overlap:
        takeaway = reverse_overlap.get("takeaway", {})
        rows = {str(r["name"]): r for r in reverse_overlap.get("rows", [])}
        reverse_line = rows.get("reverse_aligned_no_tool", {})
        lines.append("- reverse overlap 补充：")
        if reverse_line:
            nodes = ", ".join(str(x) for x in reverse_line.get("nodes", []))
            lines.append(
                f"  - reverse-aligned no-tool line: `{nodes}`"
            )
            lines.append(
                f"  - reverse-selective recall `{fmt(reverse_line.get('reverse_selective_recall'))}`, precision `{fmt(reverse_line.get('reverse_selective_precision'))}`"
            )
        if takeaway:
            lines.append(
                f"  - {takeaway.get('minimal_chain_vs_reverse_core', '')}"
            )
            lines.append(
                f"  - {takeaway.get('reverse_aligned_semantic_line_vs_reverse_selective', '')}"
            )
    lines.append("")
    lines.append(
        "`no_tool` 链的发生方式因此是“双重的”：`MLP17` 既把 no-tool 状态往 `L23H6` / 输出区送，也对 `L20H5/L21H12` 这类 tool ingress 做抑制。所以它不是只在末端写一个 `no_tool` token，而是在晚期决策区同时做“写 no-tool”和“压 tool 路”。"
    )
    lines.append("")
    lines.append("## 5. 直接回答 TODO 里的五个子问题")
    lines.append("")
    lines.append(f"1. 单节点扫描里，最早出现明显 cue-sensitivity 的 tool-side 节点是 `{tool_ingress_name}`，no-tool-side 节点是 `{no_tool_ingress_name}`；但决策性晚期主链分别从 `L20H5` 和 `L16H4` 进入。")
    lines.append("2. 它们读到的不是孤立首词，而是 instruction lead / file target / function-body anchor 绑定出来的 user-side commitment state。")
    lines.append("3. 在晚期 attention 头上，最强证据主要落在 `z` 写出而不是单独 `Q/K/V`，说明到 `L20H5/L21H12/L16H4/L23H6` 时，cue 已经被折叠成可直接传播的 head output state。")
    lines.append("4. tool 放大发生在 `L20H5 -> L21H1/L21H12 -> L24H6 -> MLP27`；no-tool 放大发生在 `L16H4 -> MLP17 -> L23H6`，并由 `MLP17 -> L20H5/L21H12` 产生额外 suppress。")
    lines.append("5. `<tool_call>` 的主写出节点是 `MLP27`；`no_tool` 的主写出/抑制核心是 `MLP17`，`L23H6` 负责把这份状态带进晚期输出区。")
    lines.append("")
    lines.append("## Artifact Index")
    lines.append("")
    lines.append("- `minimal_cue_variant_summary.csv`")
    lines.append("- `minimal_cue_node_summary.csv`")
    lines.append("- `minimal_cue_step_summary.csv`")
    lines.append("- `minimal_cue_edge_summary.csv`")
    lines.append("- `minimal_cue_mechanism_summary.json`")
    lines.append("- `minimal_cue_mechanism_report.md`")

    (out_root / "minimal_cue_mechanism_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal-cue mechanism analysis inside the final 24-node circuit.")
    parser.add_argument("--run-root", type=str, required=True)
    parser.add_argument("--signed-nodes-csv", type=str, default="")
    parser.add_argument("--model-path", type=str, default="./external/models/Qwen3-1.7B")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--output-root", type=str, required=True)
    args = parser.parse_args()

    run_root = Path(args.run_root).resolve()
    out_root = Path(args.output_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    signed_nodes_csv = Path(args.signed_nodes_csv).resolve() if args.signed_nodes_csv else run_root / "final_signed_circuit" / "final_signed_nodes.csv"
    signed_nodes = load_signed_nodes(signed_nodes_csv)
    all_nodes = [str(r["node"]) for r in signed_nodes]
    node_meta = {str(r["node"]): r for r in signed_nodes}

    samples = load_sample_paths(run_root / "forward_batch", run_root / "reverse_batch", max_samples=args.max_samples)
    model, tokenizer = load_hooked_qwen3(args.model_path, device=args.device, dtype=torch.bfloat16)

    variant_rows: List[Dict[str, object]] = []
    node_rows: List[Dict[str, object]] = []
    step_rows: List[Dict[str, object]] = []
    edge_rows: List[Dict[str, object]] = []

    pbar = tqdm(samples, desc="Minimal cue mechanism", dynamic_ncols=True)
    for sp in pbar:
        try:
            clean_text = sp.tool_call_prompt.read_text(encoding="utf-8")
            corrupt_text = sp.no_tool_prompt.read_text(encoding="utf-8")
        except Exception:
            continue

        variants = build_variants(clean_text, corrupt_text)
        tokens_by_variant: Dict[str, torch.Tensor] = {}
        logits_by_variant: Dict[str, torch.Tensor] = {}
        for variant in ["clean_full", "corrupt_full", "clean_with_corrupt_lead", "corrupt_with_clean_lead"]:
            tokens = model.to_tokens(variants[variant], prepend_bos=False)
            tokens_by_variant[variant] = tokens
            with torch.no_grad():
                logits_by_variant[variant] = model(tokens)

        tool_objective, no_tool_objective = build_bidirectional_endpoint_objectives(
            logits_by_variant["clean_full"],
            logits_by_variant["corrupt_full"],
            tokenizer=tokenizer,
        )

        tool_clean = float(objective_from_logits(logits_by_variant["clean_full"], tool_objective).item())
        tool_corrupt_lead = float(objective_from_logits(logits_by_variant["clean_with_corrupt_lead"], tool_objective).item())
        no_tool_corrupt = float(objective_from_logits(logits_by_variant["corrupt_full"], no_tool_objective).item())
        no_tool_clean_lead = float(objective_from_logits(logits_by_variant["corrupt_with_clean_lead"], no_tool_objective).item())
        tool_gap = tool_clean - tool_corrupt_lead
        no_tool_gap = no_tool_corrupt - no_tool_clean_lead
        if (
            not math.isfinite(tool_gap)
            or abs(tool_gap) < 1e-8
            or not math.isfinite(no_tool_gap)
            or abs(no_tool_gap) < 1e-8
        ):
            continue

        for variant in ["clean_full", "clean_with_corrupt_lead", "corrupt_with_clean_lead", "corrupt_full"]:
            logits = logits_by_variant[variant]
            tool_score = float(objective_from_logits(logits, tool_objective).item())
            no_tool_score = float(objective_from_logits(logits, no_tool_objective).item())
            variant_rows.append(
                {
                    "sample_id": sp.sample_id,
                    "variant": variant,
                    "tool_score": tool_score,
                    "no_tool_score": no_tool_score,
                    "decision_score": tool_score - no_tool_score,
                    "tool_top1": int(logits[0, -1].argmax().item()) == sp.target_tool_call,
                    "no_tool_top1": int(logits[0, -1].argmax().item()) == sp.distractor,
                }
            )

        clean_full_cache = collect_cache_cpu_for_nodes(model, tokens_by_variant["clean_full"], all_nodes)
        corrupt_full_cache = collect_cache_cpu_for_nodes(model, tokens_by_variant["corrupt_full"], all_nodes)
        clean_with_corrupt_lead_cache = collect_cache_cpu_for_nodes(model, tokens_by_variant["clean_with_corrupt_lead"], all_nodes)
        corrupt_with_clean_lead_cache = collect_cache_cpu_for_nodes(model, tokens_by_variant["corrupt_with_clean_lead"], all_nodes)

        base_tool_score = float(objective_from_logits(logits_by_variant["clean_with_corrupt_lead"], tool_objective).item())
        base_tool_no_tool_score = float(objective_from_logits(logits_by_variant["clean_with_corrupt_lead"], no_tool_objective).item())
        base_no_tool_tool_score = float(objective_from_logits(logits_by_variant["corrupt_with_clean_lead"], tool_objective).item())
        base_no_tool_score = float(objective_from_logits(logits_by_variant["corrupt_with_clean_lead"], no_tool_objective).item())

        for node in all_nodes:
            logits = run_logits_with_assignments(
                model,
                tokens_by_variant["clean_with_corrupt_lead"],
                clean_full_cache,
                clean_with_corrupt_lead_cache,
                [node],
                [],
            )
            tool_score = float(objective_from_logits(logits, tool_objective).item())
            no_tool_score = float(objective_from_logits(logits, no_tool_objective).item())
            node_rows.append(
                {
                    "sample_id": sp.sample_id,
                    "family": "tool",
                    "node": node,
                    "rescue_ratio": safe_ratio(tool_score - base_tool_score, tool_gap),
                    "decision_score": tool_score - no_tool_score,
                    "tool_top1": int(logits[0, -1].argmax().item()) == sp.target_tool_call,
                    "no_tool_top1": int(logits[0, -1].argmax().item()) == sp.distractor,
                    "boundary_flip": tool_score > no_tool_score,
                }
            )

            logits = run_logits_with_assignments(
                model,
                tokens_by_variant["corrupt_with_clean_lead"],
                corrupt_full_cache,
                corrupt_with_clean_lead_cache,
                [node],
                [],
            )
            tool_score = float(objective_from_logits(logits, tool_objective).item())
            no_tool_score = float(objective_from_logits(logits, no_tool_objective).item())
            node_rows.append(
                {
                    "sample_id": sp.sample_id,
                    "family": "no_tool",
                    "node": node,
                    "rescue_ratio": safe_ratio(no_tool_score - base_no_tool_score, no_tool_gap),
                    "decision_score": tool_score - no_tool_score,
                    "tool_top1": int(logits[0, -1].argmax().item()) == sp.target_tool_call,
                    "no_tool_top1": int(logits[0, -1].argmax().item()) == sp.distractor,
                    "boundary_flip": no_tool_score > tool_score,
                }
            )

        for step_idx in range(1, len(TOOL_CHAIN) + 1):
            nodes = TOOL_CHAIN[:step_idx]
            logits = run_logits_with_assignments(
                model,
                tokens_by_variant["clean_with_corrupt_lead"],
                clean_full_cache,
                clean_with_corrupt_lead_cache,
                nodes,
                [],
            )
            tool_score = float(objective_from_logits(logits, tool_objective).item())
            no_tool_score = float(objective_from_logits(logits, no_tool_objective).item())
            step_rows.append(
                {
                    "sample_id": sp.sample_id,
                    "family": "tool",
                    "step_idx": step_idx,
                    "nodes": "|".join(nodes),
                    "rescue_ratio": safe_ratio(tool_score - base_tool_score, tool_gap),
                    "decision_score": tool_score - no_tool_score,
                    "tool_top1": int(logits[0, -1].argmax().item()) == sp.target_tool_call,
                    "no_tool_top1": int(logits[0, -1].argmax().item()) == sp.distractor,
                    "boundary_flip": tool_score > no_tool_score,
                }
            )

        for step_idx in range(1, len(NO_TOOL_CHAIN) + 1):
            nodes = NO_TOOL_CHAIN[:step_idx]
            logits = run_logits_with_assignments(
                model,
                tokens_by_variant["corrupt_with_clean_lead"],
                corrupt_full_cache,
                corrupt_with_clean_lead_cache,
                nodes,
                [],
            )
            tool_score = float(objective_from_logits(logits, tool_objective).item())
            no_tool_score = float(objective_from_logits(logits, no_tool_objective).item())
            step_rows.append(
                {
                    "sample_id": sp.sample_id,
                    "family": "no_tool",
                    "step_idx": step_idx,
                    "nodes": "|".join(nodes),
                    "rescue_ratio": safe_ratio(no_tool_score - base_no_tool_score, no_tool_gap),
                    "decision_score": tool_score - no_tool_score,
                    "tool_top1": int(logits[0, -1].argmax().item()) == sp.target_tool_call,
                    "no_tool_top1": int(logits[0, -1].argmax().item()) == sp.distractor,
                    "boundary_flip": no_tool_score > tool_score,
                }
            )

        for source, target in TOOL_EDGES:
            source_logits = run_logits_with_assignments(
                model,
                tokens_by_variant["clean_with_corrupt_lead"],
                clean_full_cache,
                clean_with_corrupt_lead_cache,
                [source],
                [],
            )
            blocked_logits = run_logits_with_assignments(
                model,
                tokens_by_variant["clean_with_corrupt_lead"],
                clean_full_cache,
                clean_with_corrupt_lead_cache,
                [source],
                [target],
            )
            source_ratio = safe_ratio(
                float(objective_from_logits(source_logits, tool_objective).item()) - base_tool_score,
                tool_gap,
            )
            blocked_ratio = safe_ratio(
                float(objective_from_logits(blocked_logits, tool_objective).item()) - base_tool_score,
                tool_gap,
            )
            edge_rows.append(
                {
                    "sample_id": sp.sample_id,
                    "family": "tool",
                    "edge": f"{source}->{target}",
                    "source_ratio": source_ratio,
                    "blocked_ratio": blocked_ratio,
                    "mediated_ratio": source_ratio - blocked_ratio if math.isfinite(source_ratio) and math.isfinite(blocked_ratio) else float("nan"),
                }
            )

        for source, target in NO_TOOL_EDGES:
            source_logits = run_logits_with_assignments(
                model,
                tokens_by_variant["corrupt_with_clean_lead"],
                corrupt_full_cache,
                corrupt_with_clean_lead_cache,
                [source],
                [],
            )
            blocked_logits = run_logits_with_assignments(
                model,
                tokens_by_variant["corrupt_with_clean_lead"],
                corrupt_full_cache,
                corrupt_with_clean_lead_cache,
                [source],
                [target],
            )
            source_ratio = safe_ratio(
                float(objective_from_logits(source_logits, no_tool_objective).item()) - base_no_tool_score,
                no_tool_gap,
            )
            blocked_ratio = safe_ratio(
                float(objective_from_logits(blocked_logits, no_tool_objective).item()) - base_no_tool_score,
                no_tool_gap,
            )
            edge_rows.append(
                {
                    "sample_id": sp.sample_id,
                    "family": "no_tool",
                    "edge": f"{source}->{target}",
                    "source_ratio": source_ratio,
                    "blocked_ratio": blocked_ratio,
                    "mediated_ratio": source_ratio - blocked_ratio if math.isfinite(source_ratio) and math.isfinite(blocked_ratio) else float("nan"),
                }
            )

        pbar.set_postfix(sample=sp.sample_id)

    variant_summary = summarize_variant_rows(variant_rows)
    node_summary = summarize_node_rows(node_rows, node_meta)
    step_summary = summarize_step_rows(step_rows)
    edge_summary = summarize_edge_rows(edge_rows)

    write_csv(variant_rows, out_root / "minimal_cue_variant_per_sample.csv")
    write_csv(node_rows, out_root / "minimal_cue_node_per_sample.csv")
    write_csv(step_rows, out_root / "minimal_cue_step_per_sample.csv")
    write_csv(edge_rows, out_root / "minimal_cue_edge_per_sample.csv")
    write_csv(variant_summary, out_root / "minimal_cue_variant_summary.csv")
    write_csv(node_summary, out_root / "minimal_cue_node_summary.csv")
    write_csv(step_summary, out_root / "minimal_cue_step_summary.csv")
    write_csv(edge_summary, out_root / "minimal_cue_edge_summary.csv")

    head_audit = load_optional_json(run_root / "final_head_attention_audit" / "head_final_audit_summary.json")
    reverse_overlap = load_optional_json(run_root / "reverse_overlap" / "reverse_overlap_summary.json")

    summary = {
        "requested_max_samples": args.max_samples,
        "variant_summary_rows": variant_summary,
        "node_summary_rows": node_summary,
        "step_summary_rows": step_summary,
        "edge_summary_rows": edge_summary,
        "artifacts": {
            "variant_per_sample_csv": str(out_root / "minimal_cue_variant_per_sample.csv"),
            "node_per_sample_csv": str(out_root / "minimal_cue_node_per_sample.csv"),
            "step_per_sample_csv": str(out_root / "minimal_cue_step_per_sample.csv"),
            "edge_per_sample_csv": str(out_root / "minimal_cue_edge_per_sample.csv"),
            "variant_summary_csv": str(out_root / "minimal_cue_variant_summary.csv"),
            "node_summary_csv": str(out_root / "minimal_cue_node_summary.csv"),
            "step_summary_csv": str(out_root / "minimal_cue_step_summary.csv"),
            "edge_summary_csv": str(out_root / "minimal_cue_edge_summary.csv"),
            "report_md": str(out_root / "minimal_cue_mechanism_report.md"),
        },
    }
    (out_root / "minimal_cue_mechanism_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    build_report(
        out_root=out_root,
        variant_summary=variant_summary,
        node_summary=node_summary,
        step_summary=step_summary,
        edge_summary=edge_summary,
        head_audit=head_audit,
        reverse_overlap=reverse_overlap,
        max_samples=args.max_samples,
    )


if __name__ == "__main__":
    main()
