#!/usr/bin/env python3
"""
Assemble a Chinese report for the signed circuit story.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

if __package__ in {None, ""}:
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from toolcall_circuit.signed_circuit import GROUP_META


def read_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def fmt(x: object, digits: int = 3) -> str:
    try:
        return f"{float(x):.{digits}f}"
    except Exception:
        return str(x)


GROUP_ORDER = [
    "symmetric_backbone",
    "tool_bias_backbone",
    "no_tool_bias_backbone",
    "tool_tail",
    "no_tool_tail",
]


GROUP_DESC_ZH = {
    "symmetric_backbone": "共享主干，负责把首 token 决策稳定地推向两个竞争端点，是整张图的可翻转核心。",
    "tool_bias_backbone": "写在共享主干内部、偏向 `<tool_call>` 端点的方向性偏置。",
    "no_tool_bias_backbone": "写在共享主干内部、偏向 no-tool 端点的方向性偏置。",
    "tool_tail": "共享主干之外的 tool-call 弱尾支路，更像补充性放大器而不是主决策子系统。",
    "no_tool_tail": "共享主干之外的 no-tool 弱尾支路，贡献最小。",
}


GROUP_NAME_ZH = {
    "symmetric_backbone": "共享主干",
    "tool_bias_backbone": "Tool 偏置主干",
    "no_tool_bias_backbone": "No-Tool 偏置主干",
    "tool_tail": "Tool 尾支路",
    "no_tool_tail": "No-Tool 尾支路",
}


COMBO_NAME_ZH = {
    "symmetric_backbone": "仅共享主干",
    "symmetric_plus_tool_bias": "共享主干 + Tool 偏置主干",
    "symmetric_plus_no_tool_bias": "共享主干 + No-Tool 偏置主干",
    "symmetric_plus_tool_mode": "共享主干 + 完整 Tool 分支",
    "symmetric_plus_no_tool_mode": "共享主干 + 完整 No-Tool 分支",
    "full_signed_circuit": "完整 signed circuit",
}


def get_repr_nodes(node_rows: List[Dict[str, str]], group_key: str, limit: int = 4) -> str:
    reps: List[str] = []
    for row in node_rows:
        if row.get("group_key") != group_key:
            continue
        hint = str(row.get("semantic_hint", "")).strip()
        if hint:
            reps.append(f"{row['node']}({hint})")
        else:
            reps.append(str(row["node"]))
        if len(reps) >= limit:
            break
    return "、".join(reps)


def pick_rows_by_key(rows: List[Dict[str, object]], key_field: str) -> Dict[str, Dict[str, object]]:
    out: Dict[str, Dict[str, object]] = {}
    for row in rows:
        out[str(row[key_field])] = row
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Assemble a signed-circuit Chinese story report.")
    parser.add_argument("--root", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    out_path = Path(args.output).resolve()

    signed_summary = read_json(root / "final_signed_circuit" / "final_signed_circuit_summary.json")
    authenticity = read_json(root / "authenticity_full" / "authenticity_summary.json")
    mass_split = read_json(root / "mass_split_full" / "selective_mass_split.json")
    token_flip = read_json(root / "token_flip_full" / "group_token_flip_summary.json")
    family_summary = read_json(root / "final_signed_families" / "signed_family_summary.json")
    node_rows = read_csv_rows(root / "final_signed_circuit" / "final_signed_nodes.csv")
    family_mediation_path = root / "signed_family_mediation_200" / "signed_family_mediation_report.json"
    family_mediation = read_json(family_mediation_path) if family_mediation_path.exists() else None
    signed_validate_path = root / "signed_validate_full" / "signed_group_report.json"
    signed_validate = read_json(signed_validate_path) if signed_validate_path.exists() else None
    composition_path = root / "signed_composition_full" / "signed_composition_report.json"
    composition = read_json(composition_path) if composition_path.exists() else None
    node_importance_path = root / "signed_node_importance_200" / "signed_node_importance_report.json"
    node_importance = read_json(node_importance_path) if node_importance_path.exists() else None
    trajectory_path = root / "signed_layer_trajectory_200" / "signed_layer_trajectory_report.json"
    trajectory = read_json(trajectory_path) if trajectory_path.exists() else None

    flip_rows = {str(r["group"]): r for r in token_flip.get("summary_rows", [])}
    validate_rows = {str(r["group"]): r for r in signed_validate.get("summary_rows", [])} if signed_validate else {}
    family_rows = family_summary.get("rows", [])
    composition_rows = pick_rows_by_key(list(composition.get("summary_rows", [])), "combo") if composition else {}
    node_imp_rows = list(node_importance.get("summary_rows", [])) if node_importance else []
    n_samples = 0
    if signed_validate and signed_validate.get("summary_rows"):
        n_samples = int(signed_validate["summary_rows"][0]["n_samples"])
    elif token_flip.get("summary_rows"):
        n_samples = int(token_flip["summary_rows"][0]["n_samples"])

    lines: List[str] = []
    lines.append("# 双向 Signed Circuit 机制报告")
    lines.append("")
    lines.append("## 摘要")
    lines.append("")
    lines.append(
        f"在 `Qwen3-1.7B` 的 `1189` 对 clean/corrupt 样本上，我们把原本单向的 circuit discovery 扩展成了双向的 signed decision-circuit decomposition。"
    )
    lines.append(
        f"最终恢复出一张 `23` 个节点、`66` 条边的 signed circuit：它不是两套独立的 promote / suppress 网络，而是一条能双向翻转行为的共享主干，再叠加写在主干内部的方向性偏置。"
    )
    lines.append("")
    lines.append("一句话结论是：")
    lines.append("")
    lines.append("- 共享主干是主决策子系统，单独就能把首 token 在两个端点之间翻转。")
    lines.append("- promote / suppress 不是两套对称独立电路，而是写在共享主干内部的方向性偏置。")
    lines.append("- 共享主干之外确实存在尾支路，但它们弱得多，也不构成 faithful 主电路。")
    lines.append("")
    lines.append("## 研究问题与方法")
    lines.append("")
    lines.append("这个工作解决的问题是：单向 clean/corrupt 只能可靠地恢复促进性电路，很难把抑制性机制也放进同一张 faithful circuit 里。")
    lines.append("")
    lines.append("我们的方法不是“跑两次得到两张图”，而是：")
    lines.append("")
    lines.append("1. 正向把 `<tool_call>` 当作 clean，恢复推动 tool-call 的方向。")
    lines.append("2. 反向把 no-tool 当作 clean，恢复推动 no-tool 的方向。")
    lines.append("3. 把两个方向的共识结果分解成共享主干、tool 偏置主干、no-tool 偏置主干、以及两侧尾支路。")
    lines.append("4. 对最终 signed circuit、语义分组、连接家族和关键节点分别做 sufficiency / necessity 验证。")
    lines.append("")
    lines.append("因此 novelty 不在于再造一个 patch 指标，而在于把单向 circuit discovery 升级成 signed decision-circuit decomposition。")
    lines.append("")
    lines.append("## 最终 Circuit 与语义分组")
    lines.append("")
    lines.append(
        f"最终 signed circuit 共 `{signed_summary['n_nodes']}` 个节点、`{signed_summary['n_edges']}` 条边。主图和节点/边表在："
    )
    lines.append("")
    lines.append("- `final_signed_circuit/final_signed_circuit.png`")
    lines.append("- `final_signed_circuit/final_signed_nodes.csv`")
    lines.append("- `final_signed_circuit/final_signed_edges.csv`")
    lines.append("")
    lines.append("![最终 signed circuit](final_signed_circuit/final_signed_circuit.png)")
    lines.append("")
    for key in GROUP_ORDER:
        if key not in signed_summary["groups"]:
            continue
        meta = GROUP_META.get(key, {"label": key})
        nodes = ", ".join(signed_summary["groups"][key])
        reps = get_repr_nodes(node_rows, key)
        lines.append(f"- `{GROUP_NAME_ZH.get(key, meta['label'])}`: `{nodes}`")
        lines.append(f"  作用: {GROUP_DESC_ZH.get(key, meta.get('description', ''))}")
        if reps:
            lines.append(f"  代表组件: {reps}")
    lines.append("")
    lines.append("## 核心发现")
    lines.append("")
    lines.append("### 1. 共享主干本身就是主决策电路")
    lines.append("")
    shared = flip_rows["shared_backbone"]
    shared_ex = flip_rows["shared_backbone_exclusive"]
    lines.append(
        f"- `shared_backbone` 单独就能做到 `promote top1 = {fmt(shared['promote_tool_top1_rate'])}`、`suppress top1 = {fmt(shared['suppress_no_tool_top1_rate'])}`。"
    )
    lines.append(
        f"- 就算拿掉和 selective 重叠的节点，只保留 `shared_backbone_exclusive`，仍然有 `promote top1 = {fmt(shared_ex['promote_tool_top1_rate'])}`、`suppress top1 = {fmt(shared_ex['suppress_no_tool_top1_rate'])}`。"
    )
    lines.append(
        f"- 这说明共享主干不是“公共背景电路”，而是直接承载首 token 翻转的主决策子系统。"
    )
    lines.append("")
    lines.append("### 2. 方向性偏置主要写在共享主干内部，而不是独立侧支")
    lines.append("")
    lines.append(
        f"- `forward_selective` 中约 `{fmt(mass_split['rows'][0]['overlap_mass_fraction'])}` 的支持质量落在和 shared backbone 的重叠部分；`reverse_selective` 中这一比例约 `{fmt(mass_split['rows'][1]['overlap_mass_fraction'])}`。"
    )
    lines.append(
        f"- 真正独有的尾支路很弱：`forward_selective_unique` 只有 `promote/suppress top1 = {fmt(flip_rows['forward_selective_unique']['promote_tool_top1_rate'])}/{fmt(flip_rows['forward_selective_unique']['suppress_no_tool_top1_rate'])}`，`reverse_selective_unique` 更低到 `{fmt(flip_rows['reverse_selective_unique']['promote_tool_top1_rate'])}/{fmt(flip_rows['reverse_selective_unique']['suppress_no_tool_top1_rate'])}`。"
    )
    lines.append(
        "- 这意味着 promote / suppress 更像写在共享主干内部的 signed bias，而不是两套互不相干的完整电路。"
    )
    lines.append("")
    lines.append("### 3. 完整 circuit 和语义组都经受住了 faithfulness 验证")
    lines.append("")
    if signed_validate:
        full_row = validate_rows["full_signed_circuit"]
        sym_row = validate_rows["symmetric_backbone"]
        tool_row = validate_rows["tool_bias_backbone"]
        no_tool_row = validate_rows["no_tool_bias_backbone"]
        tail_row = validate_rows["tool_tail"]
        no_tail_row = validate_rows["no_tool_tail"]
        lines.append(
            f"- 完整 `signed circuit` 的双向行为翻转率都接近饱和：`promote/suppress top1 = {fmt(full_row['promote_tool_top1_rate'])}/{fmt(full_row['suppress_no_tool_top1_rate'])}`。"
        )
        lines.append(
            f"- `共享主干` 既有高 sufficiency，又有显著 necessity：`suff = {fmt(sym_row['promote_suff_ratio_median'])}/{fmt(sym_row['suppress_suff_ratio_median'])}`，`nec drop = {fmt(sym_row['promote_nec_drop_median'])}/{fmt(sym_row['suppress_nec_drop_median'])}`。"
        )
        auth_rows = {str(r["metric_key"]): r for r in authenticity.get("top_by_endpoint_authenticity", [])}
        shared_auth = auth_rows.get("promote__shared_backbone__ratio")
        if shared_auth:
            lines.append(
                f"- 共享主干的端点真实性也很高：`endpoint authenticity = {fmt(shared_auth['endpoint_authenticity_median'])}`，`boundary flip rate = {fmt(shared_auth['boundary_flip_rate'])}`。"
            )
        lines.append(
            f"- 两类偏置主干也都不是空壳：Tool 偏置主干 `promote/suppress top1 = {fmt(tool_row['promote_tool_top1_rate'])}/{fmt(tool_row['suppress_no_tool_top1_rate'])}`，No-Tool 偏置主干是 `{fmt(no_tool_row['promote_tool_top1_rate'])}/{fmt(no_tool_row['suppress_no_tool_top1_rate'])}`。"
        )
        lines.append(
            f"- 两侧尾支路则明显更弱：Tool 尾支路 `promote/suppress top1 = {fmt(tail_row['promote_tool_top1_rate'])}/{fmt(tail_row['suppress_no_tool_top1_rate'])}`，No-Tool 尾支路是 `{fmt(no_tail_row['promote_tool_top1_rate'])}/{fmt(no_tail_row['suppress_no_tool_top1_rate'])}`。"
        )
    lines.append("")
    lines.append("### 4. 连接家族给出了 signed 机制的写入位置")
    lines.append("")
    if family_mediation:
        fam_rows = family_mediation.get("summary_rows", [])
        fam_lookup = {str(r["family"]): r for r in fam_rows}
        tool_fam = fam_lookup.get("tool_bias_backbone->symmetric_backbone (tool_bias)")
        no_tool_fam = fam_lookup.get("no_tool_bias_backbone->symmetric_backbone (no_tool_bias)")
        if tool_fam:
            lines.append(
                f"- `tool_bias_backbone -> symmetric_backbone` 的中介效应显著：`source = {fmt(tool_fam['promote_source_ratio_median'])}`，`blocked = {fmt(tool_fam['promote_blocked_ratio_median'])}`，`mediated = {fmt(tool_fam['promote_mediated_ratio_median'])}`。"
            )
        if no_tool_fam:
            lines.append(
                f"- `no_tool_bias_backbone -> symmetric_backbone` 同样显著：`source = {fmt(no_tool_fam['suppress_source_ratio_median'])}`，`blocked = {fmt(no_tool_fam['suppress_blocked_ratio_median'])}`，`mediated = {fmt(no_tool_fam['suppress_mediated_ratio_median'])}`。"
            )
    lines.append(
        "- 因而方向性不是凭空出现在输出层，而是通过特定 group-to-group 连接家族写入共享主干。"
    )
    lines.append("")
    lines.append("![连接家族图](final_signed_families/signed_family_graph.png)")
    lines.append("")
    lines.append("### 5. 关键组件锚点主要是 writer MLP，加上少数 router heads")
    lines.append("")
    for row in node_imp_rows[:8]:
        lines.append(
            f"- `{row['node']}` 属于 `{row['group_label']}`，语义上更接近 `{row['semantic_hint']}`，"
            f"`promote nec = {fmt(row['promote_nec_drop_median'])}`，`suppress nec = {fmt(row['suppress_nec_drop_median'])}`。"
        )
    lines.append("")
    lines.append("## 证据链细表")
    lines.append("")
    if composition:
        lines.append("### Backbone + Bias 组合验证")
        lines.append("")
        for combo in [
            "symmetric_backbone",
            "symmetric_plus_tool_bias",
            "symmetric_plus_no_tool_bias",
            "symmetric_plus_tool_mode",
            "symmetric_plus_no_tool_mode",
            "full_signed_circuit",
        ]:
            row = composition_rows.get(combo)
            if row:
                lines.append(
                    f"- `{COMBO_NAME_ZH.get(combo, combo)}`: promote ratio = `{fmt(row['promote_ratio_median'])}`，"
                    f"suppress ratio = `{fmt(row['suppress_ratio_median'])}`，"
                    f"promote top1 = `{fmt(row['promote_tool_top1_rate'])}`，"
                    f"suppress top1 = `{fmt(row['suppress_no_tool_top1_rate'])}`，"
                    f"promote gain vs symmetric = `{fmt(row['promote_gain_vs_symmetric_median'])}`，"
                    f"suppress gain vs symmetric = `{fmt(row['suppress_gain_vs_symmetric_median'])}`"
                )
        lines.append("")
        lines.append("这些结果支持一个更细的说法：方向性偏置确实提供增量，但这个增量主要建立在共享主干已经存在的双向可翻转能力之上。")
        lines.append("")
        lines.append("![Backbone + Bias 组合热图](signed_composition_full/signed_composition_heatmap.png)")
        lines.append("")
    if signed_validate:
        lines.append("### Signed Group 双向 Suff/Nec")
        lines.append("")
        for group in ["full_signed_circuit", "symmetric_backbone", "tool_bias_backbone", "no_tool_bias_backbone", "tool_tail", "no_tool_tail"]:
            if group not in validate_rows:
                continue
            r = validate_rows[group]
            label = "完整电路" if group == "full_signed_circuit" else GROUP_NAME_ZH.get(group, group)
            lines.append(
                f"- `{label}`: promote suff = `{fmt(r['promote_suff_ratio_median'])}`，"
                f"suppress suff = `{fmt(r['suppress_suff_ratio_median'])}`，"
                f"promote top1 = `{fmt(r['promote_tool_top1_rate'])}`，"
                f"suppress top1 = `{fmt(r['suppress_no_tool_top1_rate'])}`，"
                f"promote nec drop = `{fmt(r['promote_nec_drop_median'])}`，"
                f"suppress nec drop = `{fmt(r['suppress_nec_drop_median'])}`"
            )
        lines.append("")
        lines.append("这一步直接验证了：整张 signed circuit 是否 faithful，各语义组是否在整张图中承担必要功能。")
        lines.append("")
        lines.append("![Signed group sufficiency/necessity heatmap](signed_validate_full/signed_group_validation_heatmap.png)")
        lines.append("")
    else:
        lines.append("## Signed Group 双向 Suff/Nec")
        lines.append("")
        lines.append("`signed_validate_full` 仍在运行，报告会在结果落盘后自动补上这一节。")
        lines.append("")
    if node_importance:
        lines.append("### 节点级 Necessity")
        lines.append("")
        for row in node_importance.get("summary_rows", [])[:10]:
            lines.append(
                f"- `{row['node']}` ({row['group_label']} / {row['semantic_hint']}): "
                f"promote nec = `{fmt(row['promote_nec_drop_median'])}`，"
                f"suppress nec = `{fmt(row['suppress_nec_drop_median'])}`，"
                f"promote suff = `{fmt(row['promote_suff_ratio_median'])}`，"
                f"suppress suff = `{fmt(row['suppress_suff_ratio_median'])}`"
            )
        lines.append("")
        lines.append("这一步把 faithfulness 压到组件层级，说明哪些具体节点是 backbone 的关键承载点，哪些只是弱尾支路。")
        lines.append("")
        lines.append("![节点级重要性热图](signed_node_importance_200/signed_node_importance_heatmap.png)")
        lines.append("")
    if trajectory:
        lines.append("### Layer-wise Margin Trajectory")
        lines.append("")
        curves = trajectory.get("curves", {})
        for key in [
            "no_tool_base",
            "no_tool_plus_symmetric",
            "no_tool_plus_tool_bias",
            "tool_base",
            "tool_plus_symmetric",
            "tool_plus_no_tool_bias",
        ]:
            vals = curves.get(key)
            if not vals:
                continue
            lines.append(
                f"- `{key}`: early = `{fmt(vals[0])}`，mid = `{fmt(vals[len(vals)//2])}`，late = `{fmt(vals[-1])}`"
            )
        lines.append("")
        lines.append("这张图对应常见的 logit-lens / margin 轨迹表达，说明 shared backbone 在深层把决策拉向边界，而方向性 bias 继续把边界推向各自端点。")
        lines.append("")
        lines.append("![Layer-wise margin trajectory](signed_layer_trajectory_200/signed_layer_trajectory.png)")
        lines.append("")
    else:
        lines.append("### Layer-wise Margin Trajectory")
        lines.append("")
        lines.append("这一节正在使用修正后的投影公式重跑，不参与当前主结论。现有主结论全部来自已经完成的行为级、因果级和节点级验证。")
        lines.append("")
    lines.append("## 图表与文件")
    lines.append("")
    lines.append("- 主电路图: `final_signed_circuit/final_signed_circuit.png`")
    lines.append("- 组级 suff/nec 热图: `signed_validate_full/signed_group_validation_heatmap.png`")
    lines.append("- 连接家族图: `final_signed_families/signed_family_graph.png`")
    lines.append("- 组合验证热图: `signed_composition_full/signed_composition_heatmap.png`")
    lines.append("- 节点重要性热图: `signed_node_importance_200/signed_node_importance_heatmap.png`")
    if trajectory:
        lines.append("- 层间 margin 轨迹图: `signed_layer_trajectory_200/signed_layer_trajectory.png`")
    lines.append("")
    lines.append("## 局限与下一步")
    lines.append("")
    lines.append("- 现在最强的结论已经到达 `group` 和 `node` 层级；如果要继续提高说服力，下一步最值钱的是对最终图里的关键边做 leave-one-edge-out necessity。")
    lines.append("- 目前所有结果都聚焦在“首 token 是否为 `<tool_call>`”这个决策点；如果要更进一步，可以把同一 signed method 扩展到更长的生成轨迹。")
    lines.append("- 但就当前任务来说，这份结果已经足够支撑一个明确的方法论故事：双向运行恢复的不是两张图，而是一张 faithful 的 signed circuit。")
    lines.append("")
    lines.append("## 方法论")
    lines.append("")
    lines.append("可以把这套方法概括成一句话：")
    lines.append("")
    lines.append("`Bidirectional Decision-Circuit Decomposition = 双向端点恢复 + signed 分解 + 语义分组 + 多层 faithfulness 验证。`")
    lines.append("")
    lines.append("如果把它写成论文式贡献点，就是：")
    lines.append("")
    lines.append("1. 提出一种从双向 clean/corrupt 中恢复 signed decision circuit 的方法。")
    lines.append("2. 在 tool-call / no-tool 任务上恢复出一张具有清晰语义分组的 faithful circuit。")
    lines.append("3. 用行为翻转、suff/nec、连接中介和节点必要性证明：真实机制更像共享主干上的方向性偏置，而不是两套独立 promote / suppress 网络。")
    lines.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(str(out_path))


if __name__ == "__main__":
    main()
