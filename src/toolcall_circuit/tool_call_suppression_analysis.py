#!/usr/bin/env python3
"""
Build a paper-facing Tool-Call Suppression package from existing full-run results.

This script deliberately reuses the existing 1722-sample full-run suppression
artifacts from the legacy final package instead of rerunning the model. The goal
is to reorganize them into the current module-oriented workflow and add the
missing pieces needed for a standalone suppression module package:

1. unified node tiers;
2. candidate comparison;
3. paper-facing figures;
4. a report aligned with the current 4-module story.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


LEGACY_DATA_ROOT = Path("./results/legacy/final/data")
ATTN_FULL_ROOT = Path("./results/attentionhead/20260319-121000-attention-head-full")
SUPPRESSION_ROOT = Path("./results/tool_call_suppression")

ANCHOR_NODES = ["L16H4", "MLP17", "L23H6"]
SUPPORT_NODES = ["MLP16"]
CANDIDATE_NODES = ["MLP12", "L16H8", "L15H5", "L16H13", "L16H9", "L17H2"]
DOWNSTREAM_TOOL_NODES = ["L20H5", "L21H1", "L21H12", "L24H6", "MLP27"]
ATTN_PANEL_HEADS = ["L16H4", "L23H6", "L16H8"]


def fmt(value: object, digits: int = 3) -> str:
    try:
        value = float(value)
    except Exception:
        return str(value)
    if not np.isfinite(value):
        return "nan"
    return f"{value:.{digits}f}"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(data: dict, path: Path) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_df(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def stage_order_key(label: str) -> int:
    order = {
        "read_only": 1,
        "writer_added": 2,
        "late_relay_added": 3,
    }
    return order.get(label, 999)


def build_candidate_table(
    minimal_cue_summary: dict,
    focused_table: pd.DataFrame,
    attn_overview: pd.DataFrame,
) -> pd.DataFrame:
    node_rows = pd.DataFrame(minimal_cue_summary["node_summary_rows"])
    node_rows = node_rows[node_rows["family"] == "no_tool"].copy()
    focused_map = focused_table.set_index("id").to_dict(orient="index")
    attn_map = attn_overview.set_index("head").to_dict(orient="index") if "head" in attn_overview.columns else {}

    tier_map = {node: "anchor" for node in ANCHOR_NODES}
    tier_map.update({node: "support" for node in SUPPORT_NODES})
    tier_map.update({node: "candidate" for node in CANDIDATE_NODES})

    rows: List[Dict[str, object]] = []
    for node in ANCHOR_NODES + SUPPORT_NODES + CANDIDATE_NODES:
        base = node_rows[node_rows["node"] == node]
        if base.empty:
            row = {
                "node": node,
                "tier": tier_map[node],
                "group_key": "",
                "semantic_hint": "",
                "rescue_ratio_median": np.nan,
                "no_tool_top1_rate": np.nan,
                "boundary_flip_rate": np.nan,
            }
        else:
            src = base.iloc[0]
            row = {
                "node": node,
                "tier": tier_map[node],
                "group_key": src.get("group_key", ""),
                "semantic_hint": src.get("semantic_hint", ""),
                "rescue_ratio_median": src.get("rescue_ratio_median", np.nan),
                "no_tool_top1_rate": src.get("no_tool_top1_rate", np.nan),
                "boundary_flip_rate": src.get("boundary_flip_rate", np.nan),
            }

        focused = focused_map.get(node, {})
        row["focused_role"] = focused.get("role", "")
        row["focused_reads"] = focused.get("reads", "")
        row["focused_writes"] = focused.get("writes", "")
        row["focused_claim_tier"] = focused.get("claim_tier", "")

        if node in attn_map:
            attn = attn_map[node]
            row["clean_density_top_span"] = attn.get("clean_density_top_span", "")
            row["decision_density_delta_pos_span"] = attn.get("decision_density_delta_pos_span", "")
            row["decision_density_delta_neg_span"] = attn.get("decision_density_delta_neg_span", "")
        else:
            row["clean_density_top_span"] = ""
            row["decision_density_delta_pos_span"] = ""
            row["decision_density_delta_neg_span"] = ""

        if node == "MLP16":
            row["tier_rationale"] = "decision-to-suppression 边界节点；`MLP16->MLP17` 是强 fork edge，但 `MLP16` 本体不算 suppression 模块主体。"
        elif node == "L16H4":
            row["tier_rationale"] = "最稳的 suppressive reader / ingress，task-body 与 tail-suffix 读入最清楚。"
        elif node == "MLP17":
            row["tier_rationale"] = "主 suppressive writer；既抬高 no-tool，也压低 <tool_call>。"
        elif node == "L23H6":
            row["tier_rationale"] = "late suppressive relay；更像把已写好的 suppressive state 送到输出附近。"
        elif node == "MLP12":
            row["tier_rationale"] = "更像早期 no-tool seed candidate，模块边界上游，证据不足以并入 suppression 主体。"
        elif node == "L16H8":
            row["tier_rationale"] = "有 no-tool bias，但 attention 更混杂，读入对象不够像 plain-answer suppressive reader。"
        else:
            row["tier_rationale"] = "有 no-tool 痕迹，但 patch / writeout / stagewise 证据不足。"

        rows.append(row)

    return pd.DataFrame(rows)


def plot_stagewise(stage_df: pd.DataFrame, out_path: Path) -> None:
    df = stage_df.copy().sort_values("step_idx")
    x = np.arange(len(df))
    labels = ["L16H4", "+MLP17", "+L23H6"]

    plt.style.use("default")
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), constrained_layout=True)

    axes[0].plot(x, df["tool_top1_rate"], marker="o", linewidth=2.2, color="#b23b2a", label="<tool_call> top1")
    axes[0].plot(x, df["no_tool_top1_rate"], marker="o", linewidth=2.2, color="#3568b0", label="no-tool top1")
    axes[0].set_xticks(x, labels)
    axes[0].set_ylim(-0.02, 1.02)
    axes[0].set_title("Stagewise Top-1 Shift")
    axes[0].legend(frameon=False)

    axes[1].plot(x, df["tool_token_delta_median"], marker="o", linewidth=2.2, color="#b23b2a", label="<tool_call> logit")
    axes[1].plot(x, df["no_tool_token_delta_median"], marker="o", linewidth=2.2, color="#3568b0", label="no-tool logit")
    axes[1].plot(x, df["decision_score_delta_median"], marker="o", linewidth=2.2, color="#5e8d3a", label="decision")
    axes[1].axhline(0.0, color="#666666", linewidth=1.0)
    axes[1].set_xticks(x, labels)
    axes[1].set_title("Stagewise Logit / Decision Shift")
    axes[1].legend(frameon=False)

    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_node_writeout(proj_df: pd.DataFrame, int_df: pd.DataFrame, out_path: Path) -> None:
    order = ["L16H4", "MLP17", "L23H6"]
    proj_map = proj_df.set_index("node").to_dict(orient="index")
    int_map = int_df.set_index("node").to_dict(orient="index")
    x = np.arange(len(order))
    width = 0.25

    plt.style.use("default")
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), constrained_layout=True)

    axes[0].bar(x - width / 2, [proj_map[n]["tool_logit_delta_median"] for n in order], width=width, color="#b23b2a", label="clean→corrupt <tool_call>")
    axes[0].bar(x + width / 2, [proj_map[n]["no_tool_logit_delta_median"] for n in order], width=width, color="#3568b0", label="clean→corrupt no-tool")
    axes[0].axhline(0.0, color="#666666", linewidth=1.0)
    axes[0].set_xticks(x, order)
    axes[0].set_title("Direct Residual Logit Shift")
    axes[0].legend(frameon=False)

    axes[1].bar(x - width, [int_map[n]["tool_token_delta_median"] for n in order], width=width, color="#b23b2a", label="inject: <tool_call>")
    axes[1].bar(x, [int_map[n]["no_tool_token_delta_median"] for n in order], width=width, color="#3568b0", label="inject: no-tool")
    axes[1].bar(x + width, [int_map[n]["decision_score_delta_median"] for n in order], width=width, color="#5e8d3a", label="inject: decision")
    axes[1].axhline(0.0, color="#666666", linewidth=1.0)
    axes[1].set_xticks(x, order)
    axes[1].set_title("Direction Injection Effect")
    axes[1].legend(frameon=False)

    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_downstream_heatmap(int_df: pd.DataFrame, out_path: Path) -> None:
    order = ["L16H4", "MLP17", "L23H6"]
    cols = [
        ("tool_token_delta_median", "<tool_call>"),
        ("no_tool_token_delta_median", "no-tool"),
        ("decision_score_delta_median", "decision"),
        ("L20H5_projection_delta_median", "L20H5"),
        ("L21H1_projection_delta_median", "L21H1"),
        ("L21H12_projection_delta_median", "L21H12"),
        ("L24H6_projection_delta_median", "L24H6"),
        ("MLP27_projection_delta_median", "MLP27"),
    ]
    row_map = int_df.set_index("node").to_dict(orient="index")
    data = np.array([[float(row_map[node][col]) for col, _ in cols] for node in order], dtype=float)

    plt.style.use("default")
    fig, ax = plt.subplots(figsize=(10.8, 4.6), constrained_layout=True)
    vmax = max(abs(float(np.nanmin(data))), abs(float(np.nanmax(data))))
    im = ax.imshow(data, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(np.arange(len(cols)), [label for _, label in cols], rotation=25, ha="right")
    ax.set_yticks(np.arange(len(order)), order)
    ax.set_title("Downstream Suppression Heatmap")
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(j, i, fmt(data[i, j], 2), ha="center", va="center", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_candidate_comparison(candidate_df: pd.DataFrame, out_path: Path) -> None:
    df = candidate_df[candidate_df["tier"] == "candidate"].copy()
    if df.empty:
        return
    x = np.arange(len(df))

    plt.style.use("default")
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), constrained_layout=True)
    axes[0].bar(x, df["rescue_ratio_median"], color="#4c72b0")
    axes[0].set_xticks(x, df["node"], rotation=25, ha="right")
    axes[0].axhline(0.0, color="#666666", linewidth=1.0)
    axes[0].set_title("Candidate Rescue Ratio")

    axes[1].bar(x, df["no_tool_top1_rate"], color="#3568b0")
    axes[1].set_xticks(x, df["node"], rotation=25, ha="right")
    axes[1].set_ylim(0.0, max(0.05, float(df["no_tool_top1_rate"].max()) * 1.15))
    axes[1].set_title("Candidate No-Tool Top1 Rate")

    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_attention_panel(attention_root: Path, out_path: Path) -> None:
    plot_root = attention_root / "plots"
    rows = []
    for head in ATTN_PANEL_HEADS:
        layer = int(head[1:].split("H")[0])
        rows.append(
            (
                head,
                plot_root / f"layer_{layer:02d}" / head / "density_heatmap.png",
                plot_root / f"layer_{layer:02d}" / head / "decision_row.png",
            )
        )

    plt.style.use("default")
    fig, axes = plt.subplots(len(rows), 2, figsize=(12.0, 4.0 * len(rows)), constrained_layout=True)
    if len(rows) == 1:
        axes = np.array([axes])

    for i, (head, density_path, decision_path) in enumerate(rows):
        for j, img_path in enumerate([density_path, decision_path]):
            ax = axes[i, j]
            img = mpimg.imread(img_path)
            ax.imshow(img)
            ax.axis("off")
            ax.set_title(f"{head} - {'density' if j == 0 else 'decision-row'}")

    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def build_report(
    out_root: Path,
    proj_df: pd.DataFrame,
    dir_df: pd.DataFrame,
    int_df: pd.DataFrame,
    stage_df: pd.DataFrame,
    candidate_df: pd.DataFrame,
    focused_df: pd.DataFrame,
) -> None:
    proj_map = proj_df.set_index("node").to_dict(orient="index")
    dir_map = dir_df.set_index("node").to_dict(orient="index")
    int_map = int_df.set_index("node").to_dict(orient="index")
    focused_map = focused_df.set_index("id").to_dict(orient="index")

    l16 = int_map["L16H4"]
    mlp17 = int_map["MLP17"]
    l23 = int_map["L23H6"]
    step1 = stage_df[stage_df["step_idx"] == 1].iloc[0]
    step2 = stage_df[stage_df["step_idx"] == 2].iloc[0]
    step3 = stage_df[stage_df["step_idx"] == 3].iloc[0]

    lines: List[str] = []
    lines.append("# Tool-Call Suppression 主报告")
    lines.append("")
    lines.append("## 模块定义")
    lines.append("")
    lines.append(
        "本报告把 Tool-Call Suppression 定义为：在上游 `Output-Route Decision` 已经偏向 direct-response route 之后，"
        "一条 competing no-tool 线路把 ordinary-answer 侧的状态写强，并同时把 `<tool_call>` 路线压低。"
        "对当前数据最可信的最小主链是 `L16H4 -> MLP17 -> L23H6`；其中 `MLP16 -> MLP17` 是从 Decision 模块进入 suppressive line 的边界 fork，而不算 suppression 模块主体。"
    )
    lines.append("")
    lines.append("## 样本与结果范围")
    lines.append("")
    lines.append("- 全部结论基于 `1722` 个有效样本的 full-run suppression 结果。")
    lines.append("- 这次不重跑大模型；直接复用 legacy full-run 的 per-sample 与 summary 结果，并按当前模块标准重组。")
    lines.append("- attention 可视化同时复用 `experiment/results/attentionhead/20260319-121000-attention-head-full/`。")
    lines.append("")
    lines.append("## 节点分层")
    lines.append("")
    lines.append("- anchor nodes: `L16H4, MLP17, L23H6`")
    lines.append("- support nodes: `MLP16`")
    lines.append("- candidate nodes: `MLP12, L16H8, L15H5, L16H13, L16H9, L17H2`")
    lines.append("")
    lines.append("### 分层理由")
    lines.append("")
    lines.append("- `L16H4` 是最稳的 suppressive reader / ingress：它主要读 task-body / tail-suffix 一带的 ordinary-answer evidence，而不是 tool schema。")
    lines.append("- `MLP17` 是主 suppressive writer：它既抬高 no-tool，也压低 `<tool_call>`，并直接扰动 tool-side ingress 与 late writer。")
    lines.append("- `L23H6` 更像 late suppressive relay，而不是新的 reader 或主要 writer。")
    lines.append("- `MLP16` 是 support，因为 `MLP16->MLP17` 是强 fork edge，但 `MLP16` 更适合被写成 Decision 到 Suppression 的边界节点。")
    lines.append("- 其余候选节点虽然带有 no-tool bias 或 no-tool tail 痕迹，但当前 patch / writeout / stagewise 证据不足以升格。")
    lines.append("")
    lines.append("## 核心结论")
    lines.append("")
    lines.append("### 1. 旧主线是否仍成立")
    lines.append("")
    lines.append(
        "成立。当前最稳的 suppressive 主链仍是 `L16H4 -> MLP17 -> L23H6`，上游通过 `MLP16->MLP17` 从 `Output-Route Decision` 分叉进来。"
    )
    lines.append("")
    lines.append("### 2. 这个模块到底在做什么")
    lines.append("")
    lines.append(
        "它不是单纯“把另一个 token 顶上去”。当前最可信的写法是："
        "`MLP17` 同时做两件事，一是把 direct-answer / no-tool 一侧写强，二是主动压低 `<tool_call>`，"
        "而 `L16H4` 提供 ordinary-answer 侧读入，`L23H6` 负责把已写好的 suppressive state 送到输出附近。"
    )
    lines.append("")
    lines.append(
        f"- `L16H4` inject into clean: `<tool_call>` `{fmt(l16['tool_token_delta_median'])}`, no-tool `{fmt(l16['no_tool_token_delta_median'])}`, decision `{fmt(l16['decision_score_delta_median'])}`。"
    )
    lines.append(
        f"- `MLP17` inject into clean: `<tool_call>` `{fmt(mlp17['tool_token_delta_median'])}`, no-tool `{fmt(mlp17['no_tool_token_delta_median'])}`, decision `{fmt(mlp17['decision_score_delta_median'])}`。"
    )
    lines.append(
        f"- `L23H6` inject into clean: `<tool_call>` `{fmt(l23['tool_token_delta_median'])}`, no-tool `{fmt(l23['no_tool_token_delta_median'])}`, decision `{fmt(l23['decision_score_delta_median'])}`。"
    )
    lines.append("")
    lines.append("### 3. `L16H4` 在读什么")
    lines.append("")
    lines.append(
        "`L16H4` 最可信的定位仍是 ordinary-answer reader / branch ingress。"
        "旧 full-run attention 审计显示它主要读 task-body 和 tail-suffix，一点也不像 tool schema reader；"
        "QKV 结果也显示它主要靠 `z` 带出 suppressive state。"
    )
    lines.append(
        f"对照证据：`L16H4->MLP17` 是强 no-tool ingress edge，focused table 给出 `MLP16->MLP17` `{focused_map['MLP16->MLP17']['evidence']}`，"
        f"`L16H4->MLP17` `{focused_map['L16H4->MLP17']['evidence']}`。"
    )
    lines.append("")
    lines.append("### 4. `MLP17` 是不是主 suppressive writer")
    lines.append("")
    lines.append("是，而且这一点是当前 suppressive 模块里最硬的结论。")
    lines.append(
        f"`MLP17` 的 clean→corrupt residual writeout 对 no-tool 是 `{fmt(proj_map['MLP17']['no_tool_logit_delta_median'])}`，"
        f"注入 suppressive direction 时 no-tool token 变化 `{fmt(mlp17['no_tool_token_delta_median'])}`，"
        f"`<tool_call>` token 变化 `{fmt(mlp17['tool_token_delta_median'])}`。"
    )
    lines.append(
        f"更关键的是，它会同时把 construction 线往 no-tool 侧推：`L20H5` `{fmt(mlp17['L20H5_projection_delta_median'])}`，"
        f"`L21H1` `{fmt(mlp17['L21H1_projection_delta_median'])}`，`L21H12` `{fmt(mlp17['L21H12_projection_delta_median'])}`，"
        f"`L24H6` `{fmt(mlp17['L24H6_projection_delta_median'])}`，`MLP27` `{fmt(mlp17['MLP27_projection_delta_median'])}`。"
    )
    lines.append("")
    lines.append("### 5. `L23H6` 是主要 writer 还是 late relay")
    lines.append("")
    lines.append(
        "更像 late relay。它当然有 suppressive 作用，但它不像 `MLP17` 那样直接把 no-tool 一侧写强；"
        "它更像把已经写好的 suppressive state 送到输出附近。"
    )
    lines.append(
        f"一个直接迹象是：clean→corrupt residual writeout 上，`L23H6` 对 `<tool_call>` 的变化很大 "
        f"`{fmt(proj_map['L23H6']['tool_logit_delta_median'])}`，但对 no-tool 并不是 strongest writer 样子 "
        f"`{fmt(proj_map['L23H6']['no_tool_logit_delta_median'])}`；"
        f"而方向注入时，它能把 no-tool 顶上去 `{fmt(l23['no_tool_token_delta_median'])}`，说明它在运输已成形的 suppressive state。"
    )
    lines.append("")
    lines.append("### 6. suppressive state 是单一方向还是多节点共同状态")
    lines.append("")
    lines.append(
        "更稳的写法是：存在节点局部一致的 suppressive direction，但 token-level 后果是分阶段积累出来的。"
        "也就是说，它不是单个节点瞬时完成，而是 `L16H4` 先读入、`MLP17` 主写、`L23H6` 后送。"
    )
    lines.append(
        f"方向一致性也支持这点：`L16H4` alignment `{fmt(dir_map['L16H4']['direction_alignment_median'])}`，"
        f"`MLP17` `{fmt(dir_map['MLP17']['direction_alignment_median'])}`，"
        f"`L23H6` `{fmt(dir_map['L23H6']['direction_alignment_median'])}`。"
    )
    lines.append("")
    lines.append("### 7. stagewise 上它是如何影响首个输出词的")
    lines.append("")
    lines.append(
        f"- `L16H4` alone: `<tool_call>` top1 `{fmt(step1['tool_top1_rate'])}`, no-tool top1 `{fmt(step1['no_tool_top1_rate'])}`, decision `{fmt(step1['decision_score_delta_median'])}`。"
    )
    lines.append(
        f"- `+MLP17`: `<tool_call>` top1 `{fmt(step2['tool_top1_rate'])}`, no-tool top1 `{fmt(step2['no_tool_top1_rate'])}`, decision `{fmt(step2['decision_score_delta_median'])}`。"
    )
    lines.append(
        f"- `+L23H6`: `<tool_call>` top1 `{fmt(step3['tool_top1_rate'])}`, no-tool top1 `{fmt(step3['no_tool_top1_rate'])}`, decision `{fmt(step3['decision_score_delta_median'])}`。"
    )
    lines.append(
        "这说明 token-level 后果并不是在 reader 阶段就出现，而是在 `MLP17` 加入后开始变得明显，最后由 `L23H6` 把 clean prompt 大范围推回 no-tool 一侧。"
    )
    lines.append("")
    lines.append("## 读什么 / 写什么 / 怎么传")
    lines.append("")
    lines.append("| 节点 | 读什么 | 写什么 | 怎样传给下游 | 当前定位 |")
    lines.append("| --- | --- | --- | --- | --- |")
    lines.append("| `L16H4` | task-body / tail-suffix 的 ordinary-answer evidence | 早期 suppressive routed state | 主要送到 `MLP17` | anchor / suppressive reader-ingress |")
    lines.append("| `MLP17` | 上游 ordinary-answer / direct-response state | no-tool-favoring residual，并同时压 `<tool_call>` | 送到 `L23H6`，并回扰 `L20H5/L21H1/L21H12/L24H6/MLP27` | anchor / main suppressive writer |")
    lines.append("| `L23H6` | 已写好的 suppressive state | 输出附近的 late suppressive relay state | 进入 output-adjacent region | anchor / late suppressive relay |")
    lines.append("| `MLP16` | shared route score | suppressive fork input | 主要送到 `MLP17` | support / boundary fork node |")
    lines.append("")
    lines.append("## 与 Output-Route Decision 的连接")
    lines.append("")
    lines.append(
        "当前最稳的连接写法是：`MLP16 -> MLP17` 是 direct-answer / suppressive 分支的强 fork edge。"
        "`MLP16` 还属于 `Output-Route Decision`，但 `MLP17` 开始，状态才真正进入 suppression 模块。"
    )
    lines.append(
        f"旧 full-run focused table 明确把 `MLP16->MLP17` 写成 strong edge，证据为 `{focused_map['MLP16->MLP17']['evidence']}`。"
    )
    lines.append("")
    lines.append("## 候选节点比较")
    lines.append("")
    lines.append(
        "这轮没有发现足以推翻旧主线的新 anchor。`MLP12` 更像更早的 no-tool seed，`L16H8` 虽有 no-tool bias，但 attention 更混杂，"
        "`L15H5 / L16H13 / L16H9 / L17H2` 主要保留为 no-tool tail 候选。"
    )
    lines.append("")
    lines.append("## 写出类可视化结论")
    lines.append("")
    lines.append(f"- stagewise 轨迹见 `{out_root / 'figures' / 'suppression_stagewise_trajectory.png'}`。这张图最直接回答 suppressive state 何时开始具备 token-level 后果。")
    lines.append(f"- 节点 writeout 见 `{out_root / 'figures' / 'suppression_node_writeout.png'}`。这张图最直接区分 reader、writer、relay。")
    lines.append(f"- downstream suppression heatmap 见 `{out_root / 'figures' / 'suppression_downstream_heatmap.png'}`。这张图回答 `MLP17` 是否真的在压 tool path。")
    lines.append(f"- candidate comparison 见 `{out_root / 'figures' / 'suppression_candidate_comparison.png'}`。这张图回答为什么旧主线没有被候选节点推翻。")
    lines.append(f"- attention panel 见 `{out_root / 'figures' / 'suppression_attention_panels.png'}`。这张图回答 `L16H4` 与 `L23H6` 的读入差别，以及 `L16H8` 为何停在 candidate。")
    lines.append("")
    lines.append("## 未解决问题")
    lines.append("")
    lines.append("- `L16H4` 读入的 ordinary-answer evidence 还不能被强写成更窄的单一 microfeature。")
    lines.append("- `L23H6` 的精确 transported microfeature 仍更适合写成“suppressive state”，不宜过度命名。")
    lines.append("- 候选 no-tool 头虽然存在，但当前没有一个具备足够强的 patching + writeout 证据来升格。")
    lines.append("")
    lines.append("## 论文风格总结")
    lines.append("")
    lines.append(
        "当前最可信的 Tool-Call Suppression 机制是：`MLP16` 把 direct-response 一侧的 fork 输入送到 `MLP17` 后，"
        "`L16H4` 提供 ordinary-answer 侧的 suppressive 读入，`MLP17` 把这份状态写成同时抬高 no-tool、压低 `<tool_call>` 的主 suppressive direction，"
        "随后 `L23H6` 把已写好的 suppressive state 送到输出附近。最强证据来自三类 full-run 结果同时收敛：其一，reader / writer / relay 三者在 attention、direction inject、stagewise 上有清楚分工；"
        "其二，`MLP17` 的 intervention 不只改变 `<tool_call>` 与 no-tool token，还会同步把 `L20H5/L21H1/L21H12/L24H6/MLP27` 推向各自的 local no-tool 方向；"
        "其三，stagewise 结果显示 suppressive token-level 后果不是在 reader 阶段完成，而是在 `MLP17` 加入后 sharply 出现，再由 `L23H6` 扩展到输出附近。"
        "当前还不能强写的是：`L16H4` 的精确 microfeature 名称，以及任何候选 no-tool 头已经足以改写这条主链。"
    )
    lines.append("")
    (out_root / "tool_call_suppression_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a paper-facing Tool-Call Suppression package from existing full-run results.")
    parser.add_argument("--legacy-data-root", type=str, default=str(LEGACY_DATA_ROOT))
    parser.add_argument("--attention-root", type=str, default=str(ATTN_FULL_ROOT))
    parser.add_argument("--output-root", type=str, default="")
    args = parser.parse_args()

    legacy_root = Path(args.legacy_data_root).resolve()
    attention_root = Path(args.attention_root).resolve()

    timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    out_root = Path(args.output_root).resolve() if args.output_root else (SUPPRESSION_ROOT / f"{timestamp}-tool-call-suppression")
    figures_root = out_root / "figures"
    out_root.mkdir(parents=True, exist_ok=True)
    figures_root.mkdir(parents=True, exist_ok=True)

    proj_df = pd.read_csv(legacy_root / "suppression_projection_summary.csv")
    dir_df = pd.read_csv(legacy_root / "suppression_direction_summary.csv")
    int_df = pd.read_csv(legacy_root / "suppression_intervention_summary.csv")
    stage_df = pd.read_csv(legacy_root / "suppression_stagewise_summary.csv").sort_values("step_idx")
    focused_df = pd.read_csv(legacy_root / "focused_mechanism_table.csv")
    minimal_cue_summary = load_json(legacy_root / "minimal_cue_mechanism_summary.json")
    attn_overview = pd.read_csv(attention_root / "analysis" / "head_overview.csv")
    claim_tiers = load_json(legacy_root / "suppression_claim_tiers.json")

    candidate_df = build_candidate_table(minimal_cue_summary, focused_df, attn_overview)

    write_df(proj_df, out_root / "suppression_projection_summary.csv")
    write_df(dir_df, out_root / "suppression_direction_summary.csv")
    write_df(int_df, out_root / "suppression_intervention_summary.csv")
    write_df(stage_df, out_root / "suppression_stagewise_summary.csv")
    write_df(candidate_df, out_root / "suppression_candidate_tier_table.csv")
    write_json(claim_tiers, out_root / "suppression_claim_tiers.json")

    plot_stagewise(stage_df, figures_root / "suppression_stagewise_trajectory.png")
    plot_node_writeout(proj_df, int_df, figures_root / "suppression_node_writeout.png")
    plot_downstream_heatmap(int_df, figures_root / "suppression_downstream_heatmap.png")
    plot_candidate_comparison(candidate_df, figures_root / "suppression_candidate_comparison.png")
    plot_attention_panel(attention_root, figures_root / "suppression_attention_panels.png")

    build_report(
        out_root=out_root,
        proj_df=proj_df,
        dir_df=dir_df,
        int_df=int_df,
        stage_df=stage_df,
        candidate_df=candidate_df,
        focused_df=focused_df,
    )

    summary = {
        "actual_n": int(stage_df["n_samples"].iloc[0]),
        "anchor_nodes": ANCHOR_NODES,
        "support_nodes": SUPPORT_NODES,
        "candidate_nodes": CANDIDATE_NODES,
        "artifacts": {
            "report_md": str(out_root / "tool_call_suppression_report.md"),
            "projection_summary_csv": str(out_root / "suppression_projection_summary.csv"),
            "direction_summary_csv": str(out_root / "suppression_direction_summary.csv"),
            "intervention_summary_csv": str(out_root / "suppression_intervention_summary.csv"),
            "stagewise_summary_csv": str(out_root / "suppression_stagewise_summary.csv"),
            "candidate_tier_csv": str(out_root / "suppression_candidate_tier_table.csv"),
            "claim_tiers_json": str(out_root / "suppression_claim_tiers.json"),
            "stagewise_png": str(figures_root / "suppression_stagewise_trajectory.png"),
            "node_writeout_png": str(figures_root / "suppression_node_writeout.png"),
            "downstream_heatmap_png": str(figures_root / "suppression_downstream_heatmap.png"),
            "candidate_comparison_png": str(figures_root / "suppression_candidate_comparison.png"),
            "attention_panel_png": str(figures_root / "suppression_attention_panels.png"),
        },
    }
    write_json(summary, out_root / "tool_call_suppression_summary.json")


if __name__ == "__main__":
    main()
