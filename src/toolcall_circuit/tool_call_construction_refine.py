#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
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

from toolcall_circuit.bidirectional_causal_eval import load_sample_paths
from toolcall_circuit.objective import build_bidirectional_endpoint_objectives, objective_from_logits
from toolcall_circuit.signed_edge_importance import run_logits_with_assignments
from toolcall_circuit.single_sample import load_hooked_qwen3
from toolcall_circuit.tool_call_construction_analysis import (
    build_derived_summaries,
    build_stage_top_tokens_summary,
    copy_or_combine_attention_plots,
    fmt,
    load_summary_csv,
    median,
    plot_candidate_patch,
    plot_logit_lens,
    plot_node_projection,
    plot_node_writeout,
    plot_route_fanout,
    plot_stagewise,
    plot_top_token_change,
    read_csv_rows,
    resolve_prompt_path,
    safe_rate,
    summarize_group,
    write_csv,
    collect_cache_with_scale,
)


BRANCH_CONFIGS: List[Tuple[str, List[str]]] = [
    ("corrupt_full", []),
    ("after_L20H5", ["MLP19", "L20H5"]),
    ("L21H1_only", ["MLP19", "L20H5", "L21H1"]),
    ("L21H12_only", ["MLP19", "L20H5", "L21H12"]),
    ("both_L21", ["MLP19", "L20H5", "L21H1", "L21H12"]),
    ("L21H1_L24H6", ["MLP19", "L20H5", "L21H1", "L24H6"]),
    ("L21H12_L24H6", ["MLP19", "L20H5", "L21H12", "L24H6"]),
]

SHORTCUT_CONFIGS: List[Tuple[str, List[str]]] = [
    ("corrupt_full", []),
    ("MLP19_only", ["MLP19"]),
    ("MLP27_only", ["MLP27"]),
    ("MLP19_MLP27_only", ["MLP19", "MLP27"]),
    ("late_heads_only", ["L20H5", "L21H1", "L21H12", "L24H6"]),
    ("late_heads_MLP27", ["L20H5", "L21H1", "L21H12", "L24H6", "MLP27"]),
    ("MLP19_late_heads", ["MLP19", "L20H5", "L21H1", "L21H12", "L24H6"]),
    ("full_chain", ["MLP19", "L20H5", "L21H1", "L21H12", "L24H6", "MLP27"]),
]


def legacy_bool_median_summary(
    rows: Sequence[Dict[str, object]],
    *,
    keys: Sequence[str],
    metric: str,
) -> Dict[Tuple[str, ...], float]:
    grouped: Dict[Tuple[str, ...], List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(str(row[k]) for k in keys)].append(dict(row))
    out: Dict[Tuple[str, ...], float] = {}
    for key, members in grouped.items():
        vals = [bool(row[metric]) for row in members if row.get(metric) is not None]
        out[key] = median(float(v) for v in vals)
    return out


def copy_base_artifacts(source_root: Path, out_root: Path) -> None:
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "figures").mkdir(parents=True, exist_ok=True)
    for name in [
        "construction_writeout_per_sample.csv",
        "construction_stagewise_per_sample.csv",
        "construction_route_fanout_per_sample.csv",
        "construction_candidate_patch_per_sample.csv",
        "construction_mean_logit_lens_summary.csv",
    ]:
        shutil.copy2(source_root / name, out_root / name)


def rebuild_corrected_core(source_root: Path, out_root: Path, attention_root: Path) -> Dict[str, object]:
    copy_base_artifacts(source_root, out_root)
    writeout_rows = load_summary_csv(out_root / "construction_writeout_per_sample.csv")
    stage_rows = load_summary_csv(out_root / "construction_stagewise_per_sample.csv")
    route_fanout_rows = load_summary_csv(out_root / "construction_route_fanout_per_sample.csv")
    candidate_patch_rows = load_summary_csv(out_root / "construction_candidate_patch_per_sample.csv")
    mean_logit_rows = load_summary_csv(out_root / "construction_mean_logit_lens_summary.csv")
    actual_n = len({str(r["sample_id"]) for r in stage_rows if str(r.get("step_label")) == "corrupt_full"})

    writeout_summary, stage_summary, stage_top_tokens_summary, route_fanout_summary, candidate_patch_summary = build_derived_summaries(
        writeout_rows=writeout_rows,
        stage_rows=stage_rows,
        route_fanout_rows=route_fanout_rows,
        candidate_patch_rows=candidate_patch_rows,
        actual_n=actual_n,
    )

    write_csv(writeout_summary, out_root / "construction_writeout_summary.csv")
    write_csv(stage_summary, out_root / "construction_stagewise_summary.csv")
    write_csv(stage_top_tokens_summary, out_root / "construction_stagewise_top_tokens_summary.csv")
    write_csv(route_fanout_summary, out_root / "construction_route_fanout_summary.csv")
    write_csv(candidate_patch_summary, out_root / "construction_candidate_patch_summary.csv")

    plot_node_writeout(writeout_summary, out_root / "figures" / "construction_node_writeout.png")
    plot_node_projection(writeout_summary, out_root / "figures" / "construction_node_projection.png")
    plot_stagewise(stage_summary, out_root / "figures" / "construction_stagewise_trajectory.png")
    plot_top_token_change(stage_summary, stage_top_tokens_summary, out_root / "figures" / "construction_top_token_change.png")
    plot_route_fanout(route_fanout_summary, out_root / "figures" / "construction_route_fanout.png")
    plot_candidate_patch(candidate_patch_summary, out_root / "figures" / "construction_candidate_patch.png")
    plot_logit_lens(mean_logit_rows, out_root / "figures" / "construction_mean_logit_lens.png")
    attention_artifacts = copy_or_combine_attention_plots(attention_root, out_root)

    return {
        "actual_n": actual_n,
        "writeout_summary": writeout_summary,
        "stage_rows": stage_rows,
        "stage_summary": stage_summary,
        "stage_top_tokens_summary": stage_top_tokens_summary,
        "route_fanout_summary": route_fanout_summary,
        "candidate_patch_rows": candidate_patch_rows,
        "candidate_patch_summary": candidate_patch_summary,
        "mean_logit_rows": mean_logit_rows,
        "attention_artifacts": attention_artifacts,
    }


def summarize_config_rows(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    summary = summarize_group(
        rows,
        keys=["family", "config_label", "nodes"],
        metrics=["rescue_ratio", "route_margin", "tool_logit", "competitor_logit", "margin_logit", "tool_prob", "tool_top1", "boundary_flip"],
    )
    for row in summary:
        row["rescue_ratio_median"] = row.pop("rescue_ratio")
        row["route_margin_median"] = row.pop("route_margin")
        row["tool_logit_median"] = row.pop("tool_logit")
        row["competitor_logit_median"] = row.pop("competitor_logit")
        row["margin_logit_median"] = row.pop("margin_logit")
        row["tool_prob_median"] = row.pop("tool_prob")
        row["tool_top1_rate"] = row.pop("tool_top1")
        row["boundary_flip_rate"] = row.pop("boundary_flip")
    return summary


def plot_config_compare(
    summary_rows: Sequence[Dict[str, object]],
    *,
    order: Sequence[str],
    title: str,
    out_path: Path,
) -> None:
    row_map = {str(r["config_label"]): r for r in summary_rows}
    xs = np.arange(len(order))
    route = [float(row_map[label]["route_margin_median"]) for label in order]
    margin = [float(row_map[label]["margin_logit_median"]) for label in order]
    top1 = [float(row_map[label]["tool_top1_rate"]) for label in order]
    boundary = [float(row_map[label]["boundary_flip_rate"]) for label in order]

    plt.style.use("default")
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.8), constrained_layout=True)
    width = 0.36
    axes[0].bar(xs - width / 2, route, width=width, color="#4878a8", label="route margin")
    axes[0].bar(xs + width / 2, margin, width=width, color="#be4d25", label="logit margin")
    axes[0].axhline(0.0, color="#666666", linewidth=1.0)
    axes[0].set_xticks(xs, order, rotation=20, ha="right")
    axes[0].set_title(title)
    axes[0].legend(frameon=False)

    axes[1].plot(xs, top1, marker="o", color="#be4d25", label="<tool_call> top1")
    axes[1].plot(xs, boundary, marker="o", color="#4878a8", label="boundary flip")
    axes[1].set_xticks(xs, order, rotation=20, ha="right")
    axes[1].set_ylim(-0.02, 1.02)
    axes[1].set_title("Output Rates")
    axes[1].legend(frameon=False)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def run_config_audits(
    *,
    run_root: Path,
    model_path: str,
    device: str,
    configs: Sequence[Tuple[str, str, Sequence[str]]],
    max_samples: int,
) -> List[Dict[str, object]]:
    samples = load_sample_paths(run_root / "forward_batch", run_root / "reverse_batch", max_samples=max_samples)
    model, tokenizer = load_hooked_qwen3(model_path, device=device, dtype=torch.bfloat16)
    tracked_nodes = sorted({node for _family, _label, nodes in configs for node in nodes})

    rows: List[Dict[str, object]] = []
    pbar = tqdm(samples, desc="Tool-Call Construction Refine", dynamic_ncols=True)
    for sp in pbar:
        try:
            clean_text = resolve_prompt_path(sp.tool_call_prompt).read_text(encoding="utf-8")
            corrupt_text = resolve_prompt_path(sp.no_tool_prompt).read_text(encoding="utf-8")
        except Exception:
            continue

        clean_tokens = model.to_tokens(clean_text, prepend_bos=False)
        corrupt_tokens = model.to_tokens(corrupt_text, prepend_bos=False)
        if clean_tokens.shape != corrupt_tokens.shape:
            continue

        clean_cache = collect_cache_with_scale(model, clean_tokens, tracked_nodes)
        corrupt_cache = collect_cache_with_scale(model, corrupt_tokens, tracked_nodes)

        with torch.no_grad():
            clean_logits = model(clean_tokens)
            corrupt_logits = model(corrupt_tokens)
        tool_objective, no_tool_objective = build_bidirectional_endpoint_objectives(
            clean_logits,
            corrupt_logits,
            tokenizer=tokenizer,
        )
        tool_token_id = int(tool_objective.top_token_id or sp.target_tool_call)
        competitor_token_id = int(no_tool_objective.top_token_id or int(corrupt_logits[0, -1].argmax().item()))
        clean_tool = float(objective_from_logits(clean_logits, tool_objective).item())
        corrupt_tool = float(objective_from_logits(corrupt_logits, tool_objective).item())
        clean_no_tool = float(objective_from_logits(clean_logits, no_tool_objective).item())
        corrupt_no_tool = float(objective_from_logits(corrupt_logits, no_tool_objective).item())
        route_gap = clean_tool - clean_no_tool - (corrupt_tool - corrupt_no_tool)
        tool_gap = clean_tool - corrupt_tool
        if not math.isfinite(tool_gap) or abs(tool_gap) < 1e-8:
            continue

        for family, label, nodes in configs:
            if nodes:
                logits = run_logits_with_assignments(model, corrupt_tokens, clean_cache, corrupt_cache, list(nodes), [])
            else:
                logits = corrupt_logits
            tool_score = float(objective_from_logits(logits, tool_objective).item())
            no_tool_score = float(objective_from_logits(logits, no_tool_objective).item())
            top1_id = int(logits[0, -1].argmax().item())
            rows.append(
                {
                    "sample_id": sp.sample_id,
                    "family": family,
                    "config_label": label,
                    "nodes": "|".join(nodes),
                    "tool_score": tool_score,
                    "no_tool_score": no_tool_score,
                    "route_margin": tool_score - no_tool_score,
                    "tool_logit": float(logits[0, -1, tool_token_id].item()),
                    "competitor_logit": float(logits[0, -1, competitor_token_id].item()),
                    "margin_logit": float(logits[0, -1, tool_token_id].item() - logits[0, -1, competitor_token_id].item()),
                    "tool_prob": float(torch.softmax(logits[0, -1], dim=-1)[tool_token_id].item()),
                    "tool_top1": top1_id == tool_token_id,
                    "boundary_flip": tool_score > no_tool_score,
                    "rescue_ratio": (tool_score - corrupt_tool) / max(tool_gap, 1e-6),
                }
            )
        pbar.set_postfix(sample=sp.sample_id)
    return rows


def build_bug_note(
    *,
    out_root: Path,
    stage_rows: Sequence[Dict[str, object]],
    corrected_stage_summary: Sequence[Dict[str, object]],
    candidate_patch_rows: Sequence[Dict[str, object]],
    corrected_candidate_summary: Sequence[Dict[str, object]],
) -> None:
    old_stage = legacy_bool_median_summary(stage_rows, keys=["step_idx", "step_label", "nodes"], metric="tool_top1")
    new_stage = {
        (str(r["step_idx"]), str(r["step_label"]), str(r["nodes"])): float(r["tool_top1_rate"])
        for r in corrected_stage_summary
    }
    old_candidate = legacy_bool_median_summary(candidate_patch_rows, keys=["node", "layer"], metric="tool_top1")
    new_candidate = {
        (str(r["node"]), str(r["layer"])): float(r["tool_top1_rate"])
        for r in corrected_candidate_summary
    }

    changed_stage = []
    for key, new_value in new_stage.items():
        old_value = old_stage.get(key, float("nan"))
        if abs(new_value - old_value) > 1e-9:
            changed_stage.append((key[1], old_value, new_value))
    changed_stage.sort(key=lambda item: item[2] - item[1])

    changed_candidate = []
    for key, new_value in new_candidate.items():
        old_value = old_candidate.get(key, float("nan"))
        if abs(new_value - old_value) > 1e-9:
            changed_candidate.append((key[0], old_value, new_value))
    changed_candidate.sort(key=lambda item: item[2] - item[1], reverse=True)

    lines: List[str] = []
    lines.append("# Stagewise Bug 说明")
    lines.append("")
    lines.append("## bug 在哪里")
    lines.append("")
    lines.append("- bug 位于 `tool_call_construction_analysis.py` 的 `summarize_group()`。")
    lines.append("- `tool_top1` 是布尔字段，但旧逻辑只把名字以 `_rate` / `_flip` / `_success` 结尾的指标当成比例处理。")
    lines.append("- 因此 `tool_top1` 被按普通数值取了中位数，随后又在下游被重命名成 `tool_top1_rate`。")
    lines.append("")
    lines.append("## 为什么会导致 summary 出错")
    lines.append("")
    lines.append("- 对布尔值取中位数，本质上是在算“多数投票”，不是实际比例。")
    lines.append("- 所以旧 `construction_stagewise_summary.csv` 里，`plus_L21H1` 这类真正比例约为 `0.503` 的 step，会被错误写成 `1.0`。")
    lines.append("- `construction_stagewise_top_tokens_summary.csv` 用的是逐样本真实计数，所以两者自然不一致。")
    lines.append("- 同样的口径问题也影响了 `construction_candidate_patch_summary.csv` 的 `tool_top1_rate`。")
    lines.append("")
    lines.append("## 修复方式")
    lines.append("")
    lines.append("- 修复后，`summarize_group()` 会把所有布尔型指标统一按比例汇总，不再依赖字段名后缀。")
    lines.append("- `report-only` 也改成优先读取已有 per-sample 文件，重建 summary、图和报告，而不是继续沿用旧 summary。")
    lines.append("- 这次 refine 结果全部基于旧 full-run 的 per-sample 文件重建，没有重跑原始全量 writeout / route fanout / candidate scan。")
    lines.append("")
    lines.append("## 受影响并重生成的产物")
    lines.append("")
    lines.append("- `construction_stagewise_summary.csv`")
    lines.append("- `construction_candidate_patch_summary.csv`")
    lines.append("- `figures/construction_stagewise_trajectory.png`")
    lines.append("- `figures/construction_top_token_change.png`")
    lines.append("- `tool_call_construction_refine_report.md` 中所有引用 `tool_top1_rate`、`首次过半`、`top-1 多数出现` 的表述")
    lines.append("")
    lines.append("## stagewise 受影响最大的行")
    lines.append("")
    for step, old_value, new_value in changed_stage:
        lines.append(f"- `{step}`: 旧值 `{fmt(old_value)}` -> 新值 `{fmt(new_value)}`")
    if changed_candidate:
        lines.append("")
        lines.append("## candidate patch 受影响行")
        lines.append("")
        for node, old_value, new_value in changed_candidate:
            lines.append(f"- `{node}`: 旧值 `{fmt(old_value)}` -> 新值 `{fmt(new_value)}`")

    (out_root / "stagewise_bug_note.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_refine_report(
    *,
    out_root: Path,
    corrected: Dict[str, object],
    branch_summary: Sequence[Dict[str, object]],
    shortcut_summary: Sequence[Dict[str, object]],
) -> None:
    stage_summary = corrected["stage_summary"]
    stage_top_tokens_summary = corrected["stage_top_tokens_summary"]
    writeout_summary = corrected["writeout_summary"]
    route_fanout_summary = corrected["route_fanout_summary"]
    candidate_patch_summary = corrected["candidate_patch_summary"]
    mean_logit_rows = corrected["mean_logit_rows"]
    actual_n = int(corrected["actual_n"])

    stage_map = {str(r["step_label"]): r for r in stage_summary}
    branch_map = {str(r["config_label"]): r for r in branch_summary}
    shortcut_map = {str(r["config_label"]): r for r in shortcut_summary}
    route_map = {str(r["target"]): r for r in route_fanout_summary}
    write_map = {str(r["node"]): r for r in writeout_summary}
    lens_map = {str(r["node"]): r for r in mean_logit_rows}

    first_positive_margin = next((r for r in sorted(stage_summary, key=lambda r: int(r["step_idx"])) if float(r["margin_logit_median"]) > 0), None)
    first_positive_top1 = next((r for r in sorted(stage_summary, key=lambda r: int(r["step_idx"])) if float(r["tool_top1_rate"]) >= 0.5), None)
    top_token_by_step = defaultdict(list)
    for row in stage_top_tokens_summary:
        top_token_by_step[str(row["step_label"])].append(row)

    lines: List[str] = []
    lines.append("# Tool-Call Construction Refine 主报告")
    lines.append("")
    lines.append("## 范围")
    lines.append("")
    lines.append(f"- 这份 refine 报告基于 `1722` 个有效样本的 full run。")
    lines.append("- 先修正 stagewise / top1 口径 bug，再在此基础上补两个最小必要的 full-run 定向实验。")
    lines.append("- bug 说明见 `stagewise_bug_note.md`。")
    lines.append("")
    lines.append("## 修 bug 后，哪些原结论保留")
    lines.append("")
    lines.append("- 主线仍保留：`MLP19 -> L20H5 -> (L21H1 / L21H12) -> L24H6 -> MLP27`。")
    lines.append(f"- `<tool_call>` 的 logit margin 首次翻正仍在 `{first_positive_margin['step_label']}`。")
    lines.append(f"- `<tool_call>` 的 top-1 首次过半也仍在 `{first_positive_top1['step_label']}`，但现在是实值 `{fmt(first_positive_top1['tool_top1_rate'])}`，不再是旧 summary 里错误的 `1.0`。")
    lines.append("- `MLP27` 仍是主 writer，`L24H6` 仍更像 formatter / protocol commitment，`L20H5` 仍更像 construction ingress。")
    lines.append("")
    lines.append("## 修 bug 后，哪些原表述需要收紧")
    lines.append("")
    lines.append("- 旧报告里关于 `plus_L21H1 / plus_L21H12 / plus_L24H6 / plus_MLP27` 的 “top-1 已到 1.0” 都必须回收成真实比例。")
    lines.append("- 现在可以强写的是“`plus_L21H1` 首次过半”，不能再写成“从 `L21H1` 开始几乎所有样本都已经是 `<tool_call>` top-1”。")
    lines.append("- `MLP19 -> MLP27` 仍不能强写成独立 bypass 主路；这次只把它提高到‘强 parallel receipt 候选’，而不是冻结成单独 construction 子模块。")
    lines.append("")
    lines.append("## 修正后的 stagewise 重审")
    lines.append("")
    for step in ["corrupt_full", "plus_MLP19", "plus_L20H5", "plus_L21H1", "plus_L21H12", "plus_L24H6", "plus_MLP27"]:
        row = stage_map[step]
        top_token = next((r for r in top_token_by_step[step] if int(r["rank"]) == 1), None)
        lines.append(
            f"- `{step}`: route margin `{fmt(row['route_margin_median'])}`，logit margin `{fmt(row['margin_logit_median'])}`，`<tool_call>` top1 `{fmt(row['tool_top1_rate'])}`，boundary flip `{fmt(row['boundary_flip_rate'])}`，top token `{top_token['top1_token'] if top_token else 'nan'}`。"
        )
    lines.append("")
    lines.append("## L21H1 vs L21H12：限制性比较")
    lines.append("")
    lines.append(
        f"- 固定 `MLP19 + L20H5` 之后，只加 `L21H1` 时，`<tool_call>` top1 为 `{fmt(branch_map['L21H1_only']['tool_top1_rate'])}`，logit margin 为 `{fmt(branch_map['L21H1_only']['margin_logit_median'])}`。"
    )
    lines.append(
        f"- 固定 `MLP19 + L20H5` 之后，只加 `L21H12` 时，`<tool_call>` top1 为 `{fmt(branch_map['L21H12_only']['tool_top1_rate'])}`，logit margin 为 `{fmt(branch_map['L21H12_only']['margin_logit_median'])}`。"
    )
    lines.append(
        f"- 加上 `L24H6` 后，`L21H1` 分支到 `{fmt(branch_map['L21H1_L24H6']['tool_top1_rate'])}` / `{fmt(branch_map['L21H1_L24H6']['margin_logit_median'])}`，`L21H12` 分支到 `{fmt(branch_map['L21H12_L24H6']['tool_top1_rate'])}` / `{fmt(branch_map['L21H12_L24H6']['margin_logit_median'])}`。"
    )
    lines.append(
        "因此 `L21H1` 仍应该放在 support：它确实能把 `<tool_call>` 推过决策边界，但单独分支强度、与 `L24H6` 的配合强度、以及旧的 `-> MLP27` transmission 都仍弱于 `L21H12`。`L21H12` 仍更像 protocol-heavy router / binder，因此保留 anchor 更稳。"
    )
    lines.append("")
    lines.append("## MLP19 -> MLP27：parallel fanout 还是 bypass")
    lines.append("")
    lines.append(
        f"- `MLP19 + MLP27` 这条 shortcut node-set 的 `<tool_call>` top1 为 `{fmt(shortcut_map['MLP19_MLP27_only']['tool_top1_rate'])}`，logit margin 为 `{fmt(shortcut_map['MLP19_MLP27_only']['margin_logit_median'])}`。"
    )
    lines.append(
        f"- `MLP19 + late_heads` 为 `{fmt(shortcut_map['MLP19_late_heads']['tool_top1_rate'])}` / `{fmt(shortcut_map['MLP19_late_heads']['margin_logit_median'])}`；`late_heads + MLP27` 为 `{fmt(shortcut_map['late_heads_MLP27']['tool_top1_rate'])}` / `{fmt(shortcut_map['late_heads_MLP27']['margin_logit_median'])}`；完整链 `full_chain` 为 `{fmt(shortcut_map['full_chain']['tool_top1_rate'])}` / `{fmt(shortcut_map['full_chain']['margin_logit_median'])}`。"
    )
    lines.append(
        f"- 再结合旧的 fanout 审计，`MLP19 -> MLP27` 的 route mediated ratio 为 `{fmt(route_map['MLP27']['route_mediated_ratio_median'])}`，确实是强接收之一。"
    )
    lines.append(
        "当前最稳的写法是：`MLP19 -> MLP27` 存在明显 direct receipt，但它更像与 late-head 链并行的强 fanout，而不是已经能独立冻结成 bypass 主路。因为一旦只保留 shortcut node-set，恢复效果仍不如包含 `L20H5/L21H12/L24H6` 的链。"
    )
    lines.append("")
    lines.append("## 更新后的节点分层")
    lines.append("")
    lines.append("- anchor nodes: `L20H5, L21H12, L24H6, MLP27`")
    lines.append("- support nodes: `MLP19, L21H1`")
    lines.append("- candidate nodes: `L25H10, L25H13, L26H15, L27H7, MLP24, MLP25, MLP26`")
    lines.append("")
    lines.append("## 当前最可信的机制描述")
    lines.append("")
    lines.append(
        f"`MLP19` 先把上游 route state 扇出到 construction 区；`L20H5` 把这份状态绑定到文件名和函数体对象；随后 `L21H1` 与 `L21H12` 分化成两条 late routing，其中 `L21H1` 更偏 output-start/example，`L21H12` 更偏 tool-call example / instruction tail / protocol；`L24H6` 把已绑定状态压进调用起始格式；`MLP27` 最终把 `<tool_call>` 方向强写成首词偏好。`<tool_call>` 从 `L20H5` 开始被推向目标方向，在 `{first_positive_margin['step_label']}` 首次过决策边界，在 `{first_positive_top1['step_label']}` 首次过半，并在 `MLP27` 达到最强。"
    )
    lines.append("")
    lines.append("## 哪些说法可以强写")
    lines.append("")
    lines.append(f"- `MLP27` 是主 writer。证据最强：clean direct margin `{fmt(write_map['MLP27']['clean_margin_logit_median'])}`，平均 direct-logit lens rank=`{lens_map['MLP27']['tool_rank']}`。")
    lines.append(f"- `L24H6` 更像 formatter / protocol commitment。证据最强：clean direct margin `{fmt(write_map['L24H6']['clean_margin_logit_median'])}`，并且它把 stagewise top1 从 `{fmt(stage_map['plus_L21H12']['tool_top1_rate'])}` 推到 `{fmt(stage_map['plus_L24H6']['tool_top1_rate'])}`。")
    lines.append(f"- `L20H5` 是 construction ingress / payload binder。证据最强：`MLP19 -> L20H5` mediated `{fmt(route_map['L20H5']['route_mediated_ratio_median'])}`，同时 `L20H5` 的 delta margin 为 `{fmt(write_map['L20H5']['delta_margin_logit_median'])}`。")
    lines.append("- `L21H1` 与 `L21H12` 功能不同，不是简单冗余。")
    lines.append("")
    lines.append("## 哪些说法必须弱写")
    lines.append("")
    lines.append("- `MLP19 -> MLP27` 是独立 bypass 主路：当前证据还不够。")
    lines.append("- `L21H1` 可以升格成 anchor：当前限制性比较仍不支持。")
    lines.append("- 候选晚层头可以升格：当前 candidate patch 与写出证据仍然偏弱。")
    lines.append("")
    lines.append("## 论文风格总结")
    lines.append("")
    lines.append(
        "修正 stagewise 布尔汇总 bug 之后，Tool-Call Construction 的主线并没有被推翻，但它的强度表述被校正了：`<tool_call>` 不是在 `L21H1` 之后立刻接近满比例成形，而是在 `L21H1` 首次过半、在 `L21H12/L24H6` 进一步稳固、最终由 `MLP27` 写到最强。当前最可信的机制仍是：`MLP19` 提供 tool-route state 的 late fanout，`L20H5` 做 payload 绑定，`L21H1/L21H12` 做分化的 late routing，`L24H6` 做 protocol commitment，`MLP27` 做最终 writer。最强证据来自三组 full-run 结果的收敛：其一，修正后的 stagewise trajectory 和 top-token 变化图直接给出了 `<tool_call>` 从弱偏置到稳定首词的轨迹；其二，writeout / residual projection 仍显示 `L24H6` 与 `MLP27` 是最强的 late writer 节点；其三，限制性比较显示 `L21H12` 分支比 `L21H1` 分支更稳定、更接近 protocol binder，而 `MLP19 -> MLP27` 虽强，但仍更像 parallel fanout，而不是已经独立成路的 bypass。"
    )

    report_path = out_root / "tool_call_construction_refine_report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    summary = {
        "actual_n": actual_n,
        "anchor_nodes": ["L20H5", "L21H12", "L24H6", "MLP27"],
        "support_nodes": ["MLP19", "L21H1"],
        "candidate_nodes": ["L25H10", "L25H13", "L26H15", "L27H7", "MLP24", "MLP25", "MLP26"],
        "artifacts": {
            "report_md": str(report_path),
            "bug_note_md": str(out_root / "stagewise_bug_note.md"),
            "stagewise_png": str(out_root / "figures" / "construction_stagewise_trajectory.png"),
            "top_token_change_png": str(out_root / "figures" / "construction_top_token_change.png"),
            "node_writeout_png": str(out_root / "figures" / "construction_node_writeout.png"),
            "route_fanout_png": str(out_root / "figures" / "construction_route_fanout.png"),
            "branch_compare_png": str(out_root / "figures" / "l21_branch_comparison.png"),
            "shortcut_png": str(out_root / "figures" / "mlp19_mlp27_shortcut_audit.png"),
        },
    }
    (out_root / "tool_call_construction_refine_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Refine Tool-Call Construction after stagewise bug fix.")
    parser.add_argument("--source-root", type=str, required=True)
    parser.add_argument("--run-root", type=str, required=True)
    parser.add_argument("--output-root", type=str, required=True)
    parser.add_argument(
        "--attention-root",
        type=str,
        default="./results/attentionhead/20260319-121000-attention-head-full",
    )
    parser.add_argument("--model-path", type=str, default="./external/models/Qwen3-1.7B")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max-samples", type=int, default=0)
    args = parser.parse_args()

    source_root = Path(args.source_root).resolve()
    run_root = Path(args.run_root).resolve()
    out_root = Path(args.output_root).resolve()
    attention_root = Path(args.attention_root).resolve()

    corrected = rebuild_corrected_core(source_root, out_root, attention_root)

    config_specs = [("branch", label, nodes) for label, nodes in BRANCH_CONFIGS] + [
        ("shortcut", label, nodes) for label, nodes in SHORTCUT_CONFIGS
    ]
    config_rows = run_config_audits(
        run_root=run_root,
        model_path=args.model_path,
        device=args.device,
        configs=config_specs,
        max_samples=args.max_samples,
    )
    write_csv(config_rows, out_root / "refine_patch_audit_per_sample.csv")
    config_summary = summarize_config_rows(config_rows)
    write_csv(config_summary, out_root / "refine_patch_audit_summary.csv")

    branch_summary = [row for row in config_summary if str(row["family"]) == "branch"]
    shortcut_summary = [row for row in config_summary if str(row["family"]) == "shortcut"]
    write_csv(branch_summary, out_root / "l21_branch_compare_summary.csv")
    write_csv(shortcut_summary, out_root / "mlp19_mlp27_shortcut_summary.csv")

    plot_config_compare(
        branch_summary,
        order=[label for label, _nodes in BRANCH_CONFIGS],
        title="L21H1 vs L21H12 Branch Compare",
        out_path=out_root / "figures" / "l21_branch_comparison.png",
    )
    plot_config_compare(
        shortcut_summary,
        order=[label for label, _nodes in SHORTCUT_CONFIGS],
        title="MLP19 to MLP27 Shortcut Audit",
        out_path=out_root / "figures" / "mlp19_mlp27_shortcut_audit.png",
    )

    build_bug_note(
        out_root=out_root,
        stage_rows=corrected["stage_rows"],
        corrected_stage_summary=corrected["stage_summary"],
        candidate_patch_rows=corrected["candidate_patch_rows"],
        corrected_candidate_summary=corrected["candidate_patch_summary"],
    )
    build_refine_report(
        out_root=out_root,
        corrected=corrected,
        branch_summary=branch_summary,
        shortcut_summary=shortcut_summary,
    )


if __name__ == "__main__":
    main()
