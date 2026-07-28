#!/usr/bin/env python3
"""
Focused audit for the Output-Route Decision module.

This analysis is deliberately narrower than the legacy mechanism packages.
It treats the current clean/corrupt dataset as the ground-truth task setup and
retests whether `MLP11 -> MLP16 -> MLP19` forms a distinct route-decision
spine that:

1. writes a low-dimensional tool-vs-direct-answer route state,
2. amplifies and stabilizes that state across layers,
3. transmits it into both the tool-construction path and the no-tool path.
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

from toolcall_circuit.dataset import load_dataset_samples
from toolcall_circuit.objective import build_bidirectional_endpoint_objectives, objective_from_logits
from toolcall_circuit.signed_edge_importance import run_logits_with_assignments
from toolcall_circuit.single_sample import load_hooked_qwen3


DECISION_NODES = ("MLP11", "MLP16", "MLP19")
UPSTREAM_SUPPORT_NODES = ("L2H14", "MLP12")
TOOL_ROUTE_NODES = ("L20H5", "L21H1", "L21H12", "L24H6", "MLP27")
NO_TOOL_ROUTE_NODES = ("L16H4", "MLP17", "L23H6")
TRACKED_NODES = DECISION_NODES + UPSTREAM_SUPPORT_NODES + TOOL_ROUTE_NODES + NO_TOOL_ROUTE_NODES

MODULE_PATCH_STEPS = [
    ("MLP11", ("MLP11",)),
    ("MLP11+MLP16", ("MLP11", "MLP16")),
    ("MLP11+MLP16+MLP19", ("MLP11", "MLP16", "MLP19")),
]

NODE_SPECS: Dict[str, Tuple[str, int, int | None]] = {
    "L2H14": ("head", 2, 14),
    "MLP11": ("mlp", 11, None),
    "MLP12": ("mlp", 12, None),
    "MLP16": ("mlp", 16, None),
    "L16H4": ("head", 16, 4),
    "MLP17": ("mlp", 17, None),
    "MLP19": ("mlp", 19, None),
    "L20H5": ("head", 20, 5),
    "L21H1": ("head", 21, 1),
    "L21H12": ("head", 21, 12),
    "L23H6": ("head", 23, 6),
    "L24H6": ("head", 24, 6),
    "MLP27": ("mlp", 27, None),
}


def finite(values: Iterable[float]) -> List[float]:
    return [float(v) for v in values if isinstance(v, (int, float)) and math.isfinite(float(v))]


def median(values: Iterable[float]) -> float:
    vals = finite(values)
    return float(np.median(vals)) if vals else float("nan")


def safe_rate(values: Iterable[bool]) -> float:
    vals = [1.0 if bool(v) else 0.0 for v in values]
    return float(np.mean(vals)) if vals else float("nan")


def fmt(value: object, digits: int = 3) -> str:
    try:
        value = float(value)
    except Exception:
        return str(value)
    if not math.isfinite(value):
        return "nan"
    return f"{value:.{digits}f}"


def coerce_value(value: object) -> object:
    if isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return value
        low = text.lower()
        if low == "true":
            return True
        if low == "false":
            return False
        try:
            return float(text)
        except Exception:
            return value
    return value


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


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def unit(vec: torch.Tensor) -> torch.Tensor:
    denom = float(vec.norm().item())
    if denom < 1e-8:
        return torch.zeros_like(vec)
    return vec / denom


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    ua = unit(a)
    ub = unit(b)
    if float(ua.norm().item()) < 1e-8 or float(ub.norm().item()) < 1e-8:
        return float("nan")
    return float(torch.dot(ua, ub).item())


def collect_names(nodes: Sequence[str]) -> List[str]:
    names: List[str] = []
    for node in nodes:
        kind, layer, _head = NODE_SPECS[node]
        if kind == "mlp":
            names.append(f"blocks.{layer}.hook_mlp_out")
        else:
            names.append(f"blocks.{layer}.attn.hook_z")
    return sorted(set(names))


def collect_cache(model, tokens: torch.Tensor, names: Sequence[str]) -> Dict[str, torch.Tensor]:
    wanted = set(names)
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens, names_filter=lambda n: n in wanted)
    return {k: v.detach().cpu() for k, v in cache.items()}


def extract_node(cache: Dict[str, torch.Tensor], node: str) -> torch.Tensor:
    kind, layer, head = NODE_SPECS[node]
    if kind == "mlp":
        return cache[f"blocks.{layer}.hook_mlp_out"][0, -1, :].float()
    return cache[f"blocks.{layer}.attn.hook_z"][0, -1, int(head), :].float()


def projection_delta(base_vec: torch.Tensor, source_vec: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
    if float(direction.norm().item()) < 1e-8:
        return torch.zeros_like(base_vec)
    d = unit(direction)
    return (torch.dot(source_vec, d) - torch.dot(base_vec, d)) * d


def projection_value(vec: torch.Tensor, direction: torch.Tensor) -> float:
    if float(direction.norm().item()) < 1e-8:
        return float("nan")
    return float(torch.dot(vec.float(), unit(direction)).item())


def run_with_multi_edits_and_collect(
    model,
    base_tokens: torch.Tensor,
    *,
    edits: Dict[str, torch.Tensor],
    record_names: Sequence[str],
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    hooks = []

    for edit_node, edit_delta in edits.items():
        kind, layer, head = NODE_SPECS[edit_node]
        delta = edit_delta.to(base_tokens.device)
        if kind == "mlp":
            cache_name = f"blocks.{layer}.hook_mlp_out"

            def make_mlp_hook(delta_vec: torch.Tensor):
                def hook_fn(mlp_out: torch.Tensor, hook):  # noqa: ANN001
                    out = mlp_out.clone()
                    out[:, -1, :] = (out[:, -1, :].float() + delta_vec.unsqueeze(0)).to(dtype=out.dtype)
                    return out

                return hook_fn

            hooks.append((cache_name, make_mlp_hook(delta)))
        else:
            cache_name = f"blocks.{layer}.attn.hook_z"
            head_idx = int(head)

            def make_head_hook(delta_vec: torch.Tensor, h: int):
                def hook_fn(z: torch.Tensor, hook):  # noqa: ANN001
                    out = z.clone()
                    out[:, -1, h, :] = (out[:, -1, h, :].float() + delta_vec.unsqueeze(0)).to(dtype=out.dtype)
                    return out

                return hook_fn

            hooks.append((cache_name, make_head_hook(delta, head_idx)))

    recorded: Dict[str, torch.Tensor] = {}
    for name in record_names:
        def make_record(cache_name: str):
            def hook_fn(act: torch.Tensor, hook):  # noqa: ANN001
                recorded[cache_name] = act.detach().cpu()
                return act

            return hook_fn

        hooks.append((name, make_record(name)))

    with torch.no_grad():
        logits = model.run_with_hooks(base_tokens, fwd_hooks=hooks)
    return logits, recorded


def build_route_score(logits: torch.Tensor, tool_objective, no_tool_objective) -> float:
    tool_score = float(objective_from_logits(logits, tool_objective).item())
    no_tool_score = float(objective_from_logits(logits, no_tool_objective).item())
    return tool_score - no_tool_score


def summarize_rows(rows: Sequence[Dict[str, object]], key_fields: Sequence[str], metric_fields: Sequence[str]) -> List[Dict[str, object]]:
    grouped: Dict[Tuple[str, ...], List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(str(row[k]) for k in key_fields)].append(dict(row))
    out: List[Dict[str, object]] = []
    for key, grows in grouped.items():
        summary: Dict[str, object] = {field: key[idx] for idx, field in enumerate(key_fields)}
        summary["n_samples"] = len(grows)
        for metric in metric_fields:
            values = [coerce_value(r.get(metric)) for r in grows]
            if (
                metric.endswith("_flip")
                or metric.endswith("_success")
                or metric.endswith("_positive")
                or metric.endswith("_top1")
            ):
                summary[f"{metric}_rate"] = safe_rate(values)
            else:
                summary[f"{metric}_median"] = median(v for v in values if isinstance(v, (int, float)))
        out.append(summary)
    return out


def load_attention_support(attn_root: Path, legacy_data_root: Path) -> Dict[str, Dict[str, object]]:
    support: Dict[str, Dict[str, object]] = {}

    head_span_rows = read_csv_rows(legacy_data_root / "head_span_attention_summary.csv")
    qkv_rows = read_csv_rows(legacy_data_root / "head_qkv_patch_summary.csv")

    attn_lookup: Dict[str, Dict[str, object]] = {}
    for head in ("L20H5", "L21H1", "L21H12", "L16H4", "L23H6"):
        rel = [r for r in head_span_rows if str(r.get("head")) == head]
        rel.sort(key=lambda r: float(r.get("attn_density_median", "nan")), reverse=True)
        qkv = [r for r in qkv_rows if str(r.get("head")) == head]
        qkv.sort(key=lambda r: float(r.get("rescue_ratio_median", "nan")), reverse=True)
        attn_lookup[head] = {
            "legacy_top_span": rel[0] if rel else {},
            "legacy_top_qkv": qkv[0] if qkv else {},
        }

    heat_rows = read_csv_rows(attn_root / "summary" / "head_heatmap_summary.csv")
    dec_rows = read_csv_rows(attn_root / "summary" / "head_decision_row_summary.csv")
    for head in attn_lookup:
        clean_heat = [r for r in heat_rows if str(r.get("condition")) == "clean" and str(r.get("head")) == head]
        corrupt_heat = [r for r in heat_rows if str(r.get("condition")) == "corrupt" and str(r.get("head")) == head]
        clean_dec = [r for r in dec_rows if str(r.get("condition")) == "clean" and str(r.get("head")) == head]
        corrupt_dec = [r for r in dec_rows if str(r.get("condition")) == "corrupt" and str(r.get("head")) == head]
        clean_heat.sort(key=lambda r: float(r.get("density_mean", "nan")), reverse=True)
        corrupt_heat.sort(key=lambda r: float(r.get("density_mean", "nan")), reverse=True)
        clean_dec.sort(key=lambda r: float(r.get("decision_density_mean", "nan")), reverse=True)
        corrupt_dec.sort(key=lambda r: float(r.get("decision_density_mean", "nan")), reverse=True)
        support[head] = {
            **attn_lookup[head],
            "clean_heat_top": clean_heat[:5],
            "corrupt_heat_top": corrupt_heat[:5],
            "clean_decision_top": clean_dec[:5],
            "corrupt_decision_top": corrupt_dec[:5],
        }
    return support


def build_report(
    *,
    out_root: Path,
    direction_rows: Sequence[Dict[str, object]],
    node_patch_rows: Sequence[Dict[str, object]],
    module_patch_rows: Sequence[Dict[str, object]],
    intervention_rows: Sequence[Dict[str, object]],
    attention_support: Dict[str, Dict[str, object]],
    pairwise_cosines: Sequence[Dict[str, object]],
) -> None:
    dir_map = {str(r["node"]): dict(r) for r in direction_rows}
    cosine_map = {(str(r["source"]), str(r["target"])): dict(r) for r in pairwise_cosines}

    node_patch_summary = summarize_rows(
        node_patch_rows,
        key_fields=("node", "mode"),
        metric_fields=("route_rescue_ratio", "route_delta", "boundary_flip", "tool_top1", "no_tool_top1"),
    )
    node_patch_map = {(str(r["node"]), str(r["mode"])): dict(r) for r in node_patch_summary}

    module_patch_summary = summarize_rows(
        module_patch_rows,
        key_fields=("mode", "step_label"),
        metric_fields=("route_rescue_ratio", "route_delta", "boundary_flip", "tool_top1", "no_tool_top1"),
    )
    module_patch_map = {(str(r["mode"]), str(r["step_label"])): dict(r) for r in module_patch_summary}

    intervention_summary = summarize_rows(
        intervention_rows,
        key_fields=("node", "mode"),
        metric_fields=(
            "route_delta",
            "boundary_flip",
            "tool_top1",
            "no_tool_top1",
            "MLP16_route_projection_delta",
            "MLP19_route_projection_delta",
            "L20H5_tool_projection_delta",
            "L21H1_tool_projection_delta",
            "L21H12_tool_projection_delta",
            "L24H6_tool_projection_delta",
            "MLP27_tool_projection_delta",
            "L16H4_no_tool_projection_delta",
            "MLP17_no_tool_projection_delta",
            "L23H6_no_tool_projection_delta",
        ),
    )
    intervention_map = {(str(r["node"]), str(r["mode"])): dict(r) for r in intervention_summary}

    anchor_nodes = ["MLP11", "MLP16", "MLP19"]
    support_nodes = ["L2H14", "MLP12", "L20H5", "L21H12", "L16H4", "MLP17"]
    candidate_nodes = ["L12H6", "L13H9", "L17H8"]

    def attention_lines(head: str) -> List[str]:
        support = attention_support.get(head, {})
        lines: List[str] = []
        legacy_span = dict(support.get("legacy_top_span") or {})
        legacy_qkv = dict(support.get("legacy_top_qkv") or {})
        if legacy_span:
            lines.append(
                f"- 旧 span 汇总里，`{head}` 的最高密度读取是 `{legacy_span.get('span')}` "
                f"(density `{fmt(legacy_span.get('attn_density_median'))}`)。"
            )
        if legacy_qkv:
            lines.append(
                f"- 旧 QKV 因果拆分里，`{head}` 最强分量是 `{legacy_qkv.get('component')}` "
                f"(rescue `{fmt(legacy_qkv.get('rescue_ratio_median'))}`)。"
            )
        clean_dec = list(support.get("clean_decision_top") or [])
        corrupt_dec = list(support.get("corrupt_decision_top") or [])
        if clean_dec:
            lines.append(
                f"- 当前全量 attention 聚合中，clean 下 `decision row` 最偏向 `{clean_dec[0].get('key_span')}` "
                f"(density `{fmt(clean_dec[0].get('decision_density_mean'))}`)。"
            )
        if corrupt_dec:
            lines.append(
                f"- 同一头在 corrupt 下 `decision row` 最偏向 `{corrupt_dec[0].get('key_span')}` "
                f"(density `{fmt(corrupt_dec[0].get('decision_density_mean'))}`)。"
            )
        return lines

    lines: List[str] = []
    lines.append("# Output-Route Decision 主报告")
    lines.append("")
    lines.append("## 模块定义")
    lines.append("")
    lines.append(
        "这里把 Output-Route Decision 定义为：在首个输出词生成前，把已经整合好的任务表征压缩成一个"
        "可被逐层重编码的“输出路线状态”，其正向对应 tool-mediated route，反向对应 direct-answer route。"
        "这个模块的最小 anchor spine 必须满足三个条件：")
    lines.append("")
    lines.append("- 自身改动会显著改变 route score。")
    lines.append("- 该状态会在相邻层持续存在并被放大，而不是一次性局部扰动。")
    lines.append("- 该状态能同时改变 tool construction 一侧和 no-tool suppression 一侧的下游投影。")
    lines.append("")
    lines.append("在当前证据下，最小模块定义收缩为 `MLP11 -> MLP16 -> MLP19`。")
    lines.append("")
    lines.append("## 节点分层")
    lines.append("")
    lines.append(f"- anchor nodes: {', '.join(f'`{x}`' for x in anchor_nodes)}")
    lines.append(f"- support nodes: {', '.join(f'`{x}`' for x in support_nodes)}")
    lines.append(f"- candidate nodes: {', '.join(f'`{x}`' for x in candidate_nodes)}")
    lines.append("")
    lines.append("解释：anchor nodes 属于模块最小闭环；support nodes 主要位于边界，负责把状态送进模块或从模块接走；candidate nodes 仍缺直接因果闭环。")
    lines.append("")
    lines.append(
        f"跨层方向几何上并不共享同一向量：`MLP11↔MLP16` cosine `{fmt(cosine_map.get(('MLP11', 'MLP16'), {}).get('cosine'))}`，"
        f"`MLP11↔MLP19` `{fmt(cosine_map.get(('MLP11', 'MLP19'), {}).get('cosine'))}`，"
        f"`MLP16↔MLP19` `{fmt(cosine_map.get(('MLP16', 'MLP19'), {}).get('cosine'))}`。"
        "因此更稳的说法不是“单一残差向量被原样搬运”，而是“同一个 route score 被不同节点用各自局部方向重编码”。"
    )
    lines.append("")
    lines.append("support/candidate 的当前定位如下：")
    lines.append("")
    lines.append("- `L2H14` 更像上游 ingress，能把开头提示差异送进 `MLP11`，但还不能单独证明它已经写出稳定 route state。")
    lines.append("- `MLP12` 更像早期相邻偏置分支，可能参与 direct-answer 倾向，但当前缺少它作为共享 route state 主干的证据。")
    lines.append("- `L20H5/L21H12/L16H4/MLP17` 主要是模块边界接收者：它们说明 route state 已经被接走，但不说明它们本身是 decision writer。")
    lines.append("- `L12H6/L13H9/L17H8` 仍保留为 candidate shared-backbone nodes，因为现在缺模块级必要性和方向干预闭环。")
    lines.append("")
    lines.append("## 核心结论")
    lines.append("")
    lines.append(
        "当前最可信的机制是：`MLP11` 首次把 route score 写成一个稳定的局部方向，`MLP16` 对这条状态做主要放大并重新编码，"
        "`MLP19` 把它稳定成可分发的后期路由状态；随后这条状态一边提高 `L20H5/L21H1/L21H12/L24H6/MLP27` 的 tool-side 投影，"
        "一边降低 `L16H4/MLP17/L23H6` 的 no-tool 投影。"
    )
    lines.append("")
    lines.append(
        "更具体地说，这个模块写出的更像一个连续的 route score / route preference，"
        "而不是某个单点上的离散二元开关；它决定的不是“是否立刻输出 `<tool_call>` 这个单一 token”，"
        "而是“接下来答案要经由工具协议输出，还是直接用自然语言回答”的输出路线。"
    )
    lines.append("")
    lines.append("## Anchor Nodes")
    lines.append("")
    for node in anchor_nodes:
        direction = dir_map.get(node, {})
        promote_patch = node_patch_map.get((node, "promote_tool_route"), {})
        erase_patch = node_patch_map.get((node, "erase_tool_route"), {})
        inject = intervention_map.get((node, "inject_clean_route_into_corrupt"), {})
        erase = intervention_map.get((node, "erase_route_from_clean"), {})
        lines.append(f"### {node}")
        lines.append("")
        lines.append(f"- 写出证据: 全局方向对齐中位数 `{fmt(direction.get('sample_alignment_median'))}`，clean-corrupt 投影差 `{fmt(direction.get('projection_gap_median'))}`。")
        lines.append(
            f"- 单节点 patching: promote rescue `{fmt(promote_patch.get('route_rescue_ratio_median'))}`，"
            f"promote 边界翻转率 `{fmt(promote_patch.get('boundary_flip_rate'))}`；"
            f"erase route drop `{fmt(erase_patch.get('route_delta_median'))}`。"
        )
        lines.append(
            f"- 方向干预: inject 到 corrupt 后 route delta `{fmt(inject.get('route_delta_median'))}`，"
            f"tool 边界翻转率 `{fmt(inject.get('boundary_flip_rate'))}`；"
            f"erase clean 后 route delta `{fmt(erase.get('route_delta_median'))}`。"
        )
        if node == "MLP11":
            lines.append(
                f"- 下游传递: `MLP11` 注入后 `MLP16` 投影变化 `{fmt(inject.get('MLP16_route_projection_delta_median'))}`，"
                f"`MLP19` 投影变化 `{fmt(inject.get('MLP19_route_projection_delta_median'))}`。"
            )
            lines.append("- 结论: `MLP11` 是最早的稳定 route writer，不是下游 construction token 的直接拼装器。")
        elif node == "MLP16":
            lines.append(
                f"- 下游传递: `MLP16` 注入后 `MLP19` route 投影 `{fmt(inject.get('MLP19_route_projection_delta_median'))}`，"
                f"`MLP17` no-tool 投影 `{fmt(inject.get('MLP17_no_tool_projection_delta_median'))}`。"
            )
            lines.append("- 结论: `MLP16` 是主放大器，也是两条后续路线分叉前最后一个共享主干。")
        elif node == "MLP19":
            lines.append(
                f"- 下游传递: `MLP19` 注入后 `L20H5/L21H12/MLP27` 的 tool 投影分别为 "
                f"`{fmt(inject.get('L20H5_tool_projection_delta_median'))}` / "
                f"`{fmt(inject.get('L21H12_tool_projection_delta_median'))}` / "
                f"`{fmt(inject.get('MLP27_tool_projection_delta_median'))}`，"
                f"而 suppressive side 上它最直接改变的是 `L23H6` 的 late no-tool 投影 "
                f"`{fmt(inject.get('L23H6_no_tool_projection_delta_median'))}`。"
            )
            lines.append("- 结论: `MLP19` 更像 late route relay / stabilizer；它已经位于分叉后段，因此不会回头改写更早的 `L16H4/MLP17`。")
        lines.append("")
    lines.append("## 模块级验证")
    lines.append("")
    for mode in ("promote_tool_route", "erase_tool_route"):
        lines.append(f"### {mode}")
        lines.append("")
        for label, _nodes in MODULE_PATCH_STEPS:
            row = module_patch_map.get((mode, label), {})
            if not row:
                continue
            lines.append(
                f"- `{label}`: rescue `{fmt(row.get('route_rescue_ratio_median'))}`, "
                f"route delta `{fmt(row.get('route_delta_median'))}`, "
                f"boundary flip `{fmt(row.get('boundary_flip_rate'))}`."
            )
        lines.append("")
    lines.append(
        "解读：如果只 patch `MLP11` 还不能稳定跨过边界，而加上 `MLP16` 和 `MLP19` 后才出现明显更多的边界翻转，"
        "说明这三者是在共同形成和稳定 route choice。但边界翻转率仍远不到 1，"
        "所以更准确的说法是“它们构成了主要的 route-decision spine”，而不是“它们单独就足以完全决定首词”。"
    )
    lines.append("")
    lines.append("## 下游传递")
    lines.append("")
    lines.append(
        "当前结果支持一个双向分流图景：`MLP19` 之后，tool route 和 no-tool route 的投影同时被改写。"
        "在 tool direction 侧，最稳定的接受者是 `L20H5/L21H1/L21H12/L24H6/MLP27`；"
        "在 no-tool direction 侧，`L16H4/MLP17/L23H6` 的 no-tool 投影会在 tool-route 注入时下降，在 route 擦除时上升。"
    )
    lines.append("")
    lines.append("这意味着 Output-Route Decision 更像一个被逐层重编码的上游 route score，而不是两条路线各自独立的局部开关。")
    lines.append("")
    lines.append("## 热图辅助证据")
    lines.append("")
    for head in ("L20H5", "L21H12", "L16H4", "L23H6"):
        lines.append(f"### {head}")
        lines.extend(attention_lines(head))
        lines.append("")
    lines.append("这些 heatmap 只用于说明边界节点在看什么，不单独承担功能定性。")
    lines.append("")
    lines.append("## 旧结论保留与降级")
    lines.append("")
    lines.append("- 可保留: `MLP11 -> MLP16 -> MLP19` 作为 Output-Route Decision 的最小 anchor spine。")
    lines.append("- 可保留: `MLP16` 是两条后续路线分叉前的共享主干。")
    lines.append("- 需要收缩: 不能再把 `MLP11` 叫作具体的 file/object writer；当前更稳的说法是 earliest stable route writer。")
    lines.append("- 需要收缩: 不能把 `MLP19` 直接等同于 tool construction；它更像把 route state 分发给 construction 与 suppression 的 late relay。")
    lines.append("- 需要降级成候选: `L12H6/L13H9/L17H8` 仍像 shared backbone，但现在缺模块级强因果闭环。")
    lines.append("")
    lines.append("## 未解决问题")
    lines.append("")
    lines.append("- `MLP11` 写出的 route state 是否还能进一步分解成更窄的语义子方向，目前证据不足。")
    lines.append("- `MLP16 -> MLP17` 的传递目前主要靠下游投影变化支持，若要升到最强说法，还应补显式 edge mediation。")
    lines.append("- `MLP19` 到 `L20H5/L21H1/L21H12` 的分发是否一条主边就足够，还是并行冗余，目前仍需补限制性 patching。")
    lines.append("")
    lines.append("## 论文风格结论")
    lines.append("")
    lines.append(
        "综合当前证据，我们认为 Output-Route Decision 最可信的实现不是某个单独节点上的 yes/no 开关，也不是一条在所有层保持同向的固定残差向量，"
        "而是一条在 `MLP11 -> MLP16 -> MLP19` 上逐层成形、放大并重编码的 route score。`MLP11` 首次把 clean/corrupt 的最小提示差异写成稳定的局部 route state，"
        "`MLP16` 对该状态进行主放大并保持两条后续路线尚未分叉的共享骨架，`MLP19` 则把这条状态转成可向下游分发的 late route signal。"
        "当沿各自局部方向注入 tool-side 状态时，tool construction 一侧的多个节点同步增强，而 no-tool suppression 一侧的投影同步减弱；当擦除这些局部方向时，现象反向出现。"
        "因此，这个模块决定的不是单一 `<tool_call>` token 是否被写出，而是模型接下来采用工具协议输出还是直接回答的输出路线偏好。当前最强的证据来自模块级 patching 与方向干预；"
        "相对薄弱的部分是个别共享 backbone 候选节点和部分精确边级分工，它们目前只能保留为支持或候选说法。"
    )
    lines.append("")

    (out_root / "output_route_decision_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Focused audit for the Output-Route Decision module.")
    parser.add_argument("--dataset-root", type=str, default="experiment/datasets")
    parser.add_argument("--model-path", type=str, default="./external/models/Qwen3-1.7B")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--output-root", type=str, required=True)
    parser.add_argument(
        "--attention-root",
        type=str,
        default="experiment/results/attentionhead/20260319-121000-attention-head-full",
    )
    parser.add_argument(
        "--legacy-data-root",
        type=str,
        default="experiment/results/legacy/final/data",
    )
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root).resolve()
    out_root = Path(args.output_root).resolve()
    attn_root = Path(args.attention_root).resolve()
    legacy_data_root = Path(args.legacy_data_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    samples = load_dataset_samples(dataset_root)
    if args.max_samples > 0:
        samples = samples[: args.max_samples]
    if not samples:
        raise ValueError("No samples found.")

    model, tokenizer = load_hooked_qwen3(args.model_path, device=args.device, dtype=torch.bfloat16)
    hook_names = collect_names(TRACKED_NODES)

    direction_sums: Dict[str, torch.Tensor] = {}
    direction_counts = 0

    node_patch_rows: List[Dict[str, object]] = []
    module_patch_rows: List[Dict[str, object]] = []

    pbar = tqdm(samples, desc="Output-route pass1", dynamic_ncols=True)
    for sample in pbar:
        try:
            clean_text = sample.clean_path.read_text(encoding="utf-8")
            corrupt_text = sample.corrupt_path.read_text(encoding="utf-8")
        except Exception:
            continue

        clean_tokens = model.to_tokens(clean_text, prepend_bos=False)
        corrupt_tokens = model.to_tokens(corrupt_text, prepend_bos=False)

        with torch.no_grad():
            clean_logits = model(clean_tokens)
            corrupt_logits = model(corrupt_tokens)

        tool_objective, no_tool_objective = build_bidirectional_endpoint_objectives(
            clean_logits,
            corrupt_logits,
            tokenizer=tokenizer,
        )
        clean_route = build_route_score(clean_logits, tool_objective, no_tool_objective)
        corrupt_route = build_route_score(corrupt_logits, tool_objective, no_tool_objective)
        route_gap = clean_route - corrupt_route
        if not math.isfinite(route_gap) or abs(route_gap) < 1e-8:
            continue

        clean_cache = collect_cache(model, clean_tokens, hook_names)
        corrupt_cache = collect_cache(model, corrupt_tokens, hook_names)

        for node in DECISION_NODES:
            clean_vec = extract_node(clean_cache, node)
            corrupt_vec = extract_node(corrupt_cache, node)
            direction_sums[node] = direction_sums.get(node, torch.zeros_like(clean_vec)) + (clean_vec - corrupt_vec)
        for node in TOOL_ROUTE_NODES:
            clean_vec = extract_node(clean_cache, node)
            corrupt_vec = extract_node(corrupt_cache, node)
            direction_sums[node] = direction_sums.get(node, torch.zeros_like(clean_vec)) + (clean_vec - corrupt_vec)
        for node in NO_TOOL_ROUTE_NODES:
            clean_vec = extract_node(clean_cache, node)
            corrupt_vec = extract_node(corrupt_cache, node)
            direction_sums[node] = direction_sums.get(node, torch.zeros_like(clean_vec)) + (corrupt_vec - clean_vec)
        direction_counts += 1

        for node in DECISION_NODES:
            patched_logits = run_logits_with_assignments(
                model,
                corrupt_tokens,
                clean_cache,
                corrupt_cache,
                [node],
                [],
            )
            patched_route = build_route_score(patched_logits, tool_objective, no_tool_objective)
            node_patch_rows.append(
                {
                    "sample_id": sample.sample_id,
                    "node": node,
                    "mode": "promote_tool_route",
                    "route_rescue_ratio": (patched_route - corrupt_route) / route_gap,
                    "route_delta": patched_route - corrupt_route,
                    "boundary_flip": patched_route > 0.0,
                    "tool_top1": int(patched_logits[0, -1].argmax().item()) == int(tool_objective.top_token_id or -1),
                    "no_tool_top1": int(patched_logits[0, -1].argmax().item()) == int(no_tool_objective.top_token_id or -1),
                }
            )

            blocked_logits = run_logits_with_assignments(
                model,
                clean_tokens,
                corrupt_cache,
                clean_cache,
                [node],
                [],
            )
            blocked_route = build_route_score(blocked_logits, tool_objective, no_tool_objective)
            node_patch_rows.append(
                {
                    "sample_id": sample.sample_id,
                    "node": node,
                    "mode": "erase_tool_route",
                    "route_rescue_ratio": (clean_route - blocked_route) / route_gap,
                    "route_delta": blocked_route - clean_route,
                    "boundary_flip": blocked_route < 0.0,
                    "tool_top1": int(blocked_logits[0, -1].argmax().item()) == int(tool_objective.top_token_id or -1),
                    "no_tool_top1": int(blocked_logits[0, -1].argmax().item()) == int(no_tool_objective.top_token_id or -1),
                }
            )

        for label, nodes in MODULE_PATCH_STEPS:
            patched_logits = run_logits_with_assignments(
                model,
                corrupt_tokens,
                clean_cache,
                corrupt_cache,
                list(nodes),
                [],
            )
            patched_route = build_route_score(patched_logits, tool_objective, no_tool_objective)
            module_patch_rows.append(
                {
                    "sample_id": sample.sample_id,
                    "mode": "promote_tool_route",
                    "step_label": label,
                    "route_rescue_ratio": (patched_route - corrupt_route) / route_gap,
                    "route_delta": patched_route - corrupt_route,
                    "boundary_flip": patched_route > 0.0,
                    "tool_top1": int(patched_logits[0, -1].argmax().item()) == int(tool_objective.top_token_id or -1),
                    "no_tool_top1": int(patched_logits[0, -1].argmax().item()) == int(no_tool_objective.top_token_id or -1),
                }
            )

            blocked_logits = run_logits_with_assignments(
                model,
                clean_tokens,
                corrupt_cache,
                clean_cache,
                list(nodes),
                [],
            )
            blocked_route = build_route_score(blocked_logits, tool_objective, no_tool_objective)
            module_patch_rows.append(
                {
                    "sample_id": sample.sample_id,
                    "mode": "erase_tool_route",
                    "step_label": label,
                    "route_rescue_ratio": (clean_route - blocked_route) / route_gap,
                    "route_delta": blocked_route - clean_route,
                    "boundary_flip": blocked_route < 0.0,
                    "tool_top1": int(blocked_logits[0, -1].argmax().item()) == int(tool_objective.top_token_id or -1),
                    "no_tool_top1": int(blocked_logits[0, -1].argmax().item()) == int(no_tool_objective.top_token_id or -1),
                }
            )

    if direction_counts == 0:
        raise ValueError("No valid samples remained after filtering.")

    route_directions = {node: direction_sums[node] / float(direction_counts) for node in direction_sums}

    direction_rows: List[Dict[str, object]] = []
    intervention_rows: List[Dict[str, object]] = []

    pbar = tqdm(samples, desc="Output-route pass2", dynamic_ncols=True)
    for sample in pbar:
        try:
            clean_text = sample.clean_path.read_text(encoding="utf-8")
            corrupt_text = sample.corrupt_path.read_text(encoding="utf-8")
        except Exception:
            continue

        clean_tokens = model.to_tokens(clean_text, prepend_bos=False)
        corrupt_tokens = model.to_tokens(corrupt_text, prepend_bos=False)
        with torch.no_grad():
            clean_logits = model(clean_tokens)
            corrupt_logits = model(corrupt_tokens)

        tool_objective, no_tool_objective = build_bidirectional_endpoint_objectives(
            clean_logits,
            corrupt_logits,
            tokenizer=tokenizer,
        )
        clean_route = build_route_score(clean_logits, tool_objective, no_tool_objective)
        corrupt_route = build_route_score(corrupt_logits, tool_objective, no_tool_objective)
        route_gap = clean_route - corrupt_route
        if not math.isfinite(route_gap) or abs(route_gap) < 1e-8:
            continue

        clean_cache = collect_cache(model, clean_tokens, hook_names)
        corrupt_cache = collect_cache(model, corrupt_tokens, hook_names)

        clean_vecs = {node: extract_node(clean_cache, node) for node in TRACKED_NODES if node in NODE_SPECS}
        corrupt_vecs = {node: extract_node(corrupt_cache, node) for node in TRACKED_NODES if node in NODE_SPECS}

        for node in DECISION_NODES:
            direction = route_directions[node]
            clean_proj = projection_value(clean_vecs[node], direction)
            corrupt_proj = projection_value(corrupt_vecs[node], direction)
            direction_rows.append(
                {
                    "sample_id": sample.sample_id,
                    "node": node,
                    "clean_projection": clean_proj,
                    "corrupt_projection": corrupt_proj,
                    "projection_gap": clean_proj - corrupt_proj,
                    "sample_alignment": cosine(clean_vecs[node] - corrupt_vecs[node], direction),
                }
            )

        for node in DECISION_NODES:
            direction = route_directions[node]

            inject_delta = projection_delta(corrupt_vecs[node], clean_vecs[node], direction)
            inject_logits, inject_cache = run_with_multi_edits_and_collect(
                model,
                corrupt_tokens,
                edits={node: inject_delta},
                record_names=hook_names,
            )
            inject_route = build_route_score(inject_logits, tool_objective, no_tool_objective)
            intervention_rows.append(
                {
                    "sample_id": sample.sample_id,
                    "node": node,
                    "mode": "inject_clean_route_into_corrupt",
                    "route_delta": inject_route - corrupt_route,
                    "boundary_flip": inject_route > 0.0,
                    "tool_top1": int(inject_logits[0, -1].argmax().item()) == int(tool_objective.top_token_id or -1),
                    "no_tool_top1": int(inject_logits[0, -1].argmax().item()) == int(no_tool_objective.top_token_id or -1),
                    "MLP16_route_projection_delta": projection_value(extract_node(inject_cache, "MLP16"), route_directions["MLP16"]) - projection_value(corrupt_vecs["MLP16"], route_directions["MLP16"]),
                    "MLP19_route_projection_delta": projection_value(extract_node(inject_cache, "MLP19"), route_directions["MLP19"]) - projection_value(corrupt_vecs["MLP19"], route_directions["MLP19"]),
                    "L20H5_tool_projection_delta": projection_value(extract_node(inject_cache, "L20H5"), route_directions["L20H5"]) - projection_value(corrupt_vecs["L20H5"], route_directions["L20H5"]),
                    "L21H1_tool_projection_delta": projection_value(extract_node(inject_cache, "L21H1"), route_directions["L21H1"]) - projection_value(corrupt_vecs["L21H1"], route_directions["L21H1"]),
                    "L21H12_tool_projection_delta": projection_value(extract_node(inject_cache, "L21H12"), route_directions["L21H12"]) - projection_value(corrupt_vecs["L21H12"], route_directions["L21H12"]),
                    "L24H6_tool_projection_delta": projection_value(extract_node(inject_cache, "L24H6"), route_directions["L24H6"]) - projection_value(corrupt_vecs["L24H6"], route_directions["L24H6"]),
                    "MLP27_tool_projection_delta": projection_value(extract_node(inject_cache, "MLP27"), route_directions["MLP27"]) - projection_value(corrupt_vecs["MLP27"], route_directions["MLP27"]),
                    "L16H4_no_tool_projection_delta": projection_value(extract_node(inject_cache, "L16H4"), route_directions["L16H4"]) - projection_value(corrupt_vecs["L16H4"], route_directions["L16H4"]),
                    "MLP17_no_tool_projection_delta": projection_value(extract_node(inject_cache, "MLP17"), route_directions["MLP17"]) - projection_value(corrupt_vecs["MLP17"], route_directions["MLP17"]),
                    "L23H6_no_tool_projection_delta": projection_value(extract_node(inject_cache, "L23H6"), route_directions["L23H6"]) - projection_value(corrupt_vecs["L23H6"], route_directions["L23H6"]),
                }
            )

            erase_delta = projection_delta(clean_vecs[node], corrupt_vecs[node], direction)
            erase_logits, erase_cache = run_with_multi_edits_and_collect(
                model,
                clean_tokens,
                edits={node: erase_delta},
                record_names=hook_names,
            )
            erase_route = build_route_score(erase_logits, tool_objective, no_tool_objective)
            intervention_rows.append(
                {
                    "sample_id": sample.sample_id,
                    "node": node,
                    "mode": "erase_route_from_clean",
                    "route_delta": erase_route - clean_route,
                    "boundary_flip": erase_route < 0.0,
                    "tool_top1": int(erase_logits[0, -1].argmax().item()) == int(tool_objective.top_token_id or -1),
                    "no_tool_top1": int(erase_logits[0, -1].argmax().item()) == int(no_tool_objective.top_token_id or -1),
                    "MLP16_route_projection_delta": projection_value(extract_node(erase_cache, "MLP16"), route_directions["MLP16"]) - projection_value(clean_vecs["MLP16"], route_directions["MLP16"]),
                    "MLP19_route_projection_delta": projection_value(extract_node(erase_cache, "MLP19"), route_directions["MLP19"]) - projection_value(clean_vecs["MLP19"], route_directions["MLP19"]),
                    "L20H5_tool_projection_delta": projection_value(extract_node(erase_cache, "L20H5"), route_directions["L20H5"]) - projection_value(clean_vecs["L20H5"], route_directions["L20H5"]),
                    "L21H1_tool_projection_delta": projection_value(extract_node(erase_cache, "L21H1"), route_directions["L21H1"]) - projection_value(clean_vecs["L21H1"], route_directions["L21H1"]),
                    "L21H12_tool_projection_delta": projection_value(extract_node(erase_cache, "L21H12"), route_directions["L21H12"]) - projection_value(clean_vecs["L21H12"], route_directions["L21H12"]),
                    "L24H6_tool_projection_delta": projection_value(extract_node(erase_cache, "L24H6"), route_directions["L24H6"]) - projection_value(clean_vecs["L24H6"], route_directions["L24H6"]),
                    "MLP27_tool_projection_delta": projection_value(extract_node(erase_cache, "MLP27"), route_directions["MLP27"]) - projection_value(clean_vecs["MLP27"], route_directions["MLP27"]),
                    "L16H4_no_tool_projection_delta": projection_value(extract_node(erase_cache, "L16H4"), route_directions["L16H4"]) - projection_value(clean_vecs["L16H4"], route_directions["L16H4"]),
                    "MLP17_no_tool_projection_delta": projection_value(extract_node(erase_cache, "MLP17"), route_directions["MLP17"]) - projection_value(clean_vecs["MLP17"], route_directions["MLP17"]),
                    "L23H6_no_tool_projection_delta": projection_value(extract_node(erase_cache, "L23H6"), route_directions["L23H6"]) - projection_value(clean_vecs["L23H6"], route_directions["L23H6"]),
                }
            )

    direction_summary = summarize_rows(
        direction_rows,
        key_fields=("node",),
        metric_fields=("clean_projection", "corrupt_projection", "projection_gap", "sample_alignment"),
    )

    pairwise_direction_cosines: List[Dict[str, object]] = []
    for i, src in enumerate(DECISION_NODES):
        for dst in DECISION_NODES[i + 1 :]:
            pairwise_direction_cosines.append(
                {
                    "source": src,
                    "target": dst,
                    "cosine": cosine(route_directions[src], route_directions[dst]),
                }
            )

    attention_support = load_attention_support(attn_root, legacy_data_root)

    write_csv(node_patch_rows, out_root / "route_node_patch_per_sample.csv")
    write_csv(module_patch_rows, out_root / "route_module_patch_per_sample.csv")
    write_csv(direction_rows, out_root / "route_direction_per_sample.csv")
    write_csv(intervention_rows, out_root / "route_direction_intervention_per_sample.csv")

    node_patch_summary = summarize_rows(
        node_patch_rows,
        key_fields=("node", "mode"),
        metric_fields=("route_rescue_ratio", "route_delta", "boundary_flip", "tool_top1", "no_tool_top1"),
    )
    module_patch_summary = summarize_rows(
        module_patch_rows,
        key_fields=("mode", "step_label"),
        metric_fields=("route_rescue_ratio", "route_delta", "boundary_flip", "tool_top1", "no_tool_top1"),
    )
    intervention_summary = summarize_rows(
        intervention_rows,
        key_fields=("node", "mode"),
        metric_fields=(
            "route_delta",
            "boundary_flip",
            "tool_top1",
            "no_tool_top1",
            "MLP16_route_projection_delta",
            "MLP19_route_projection_delta",
            "L20H5_tool_projection_delta",
            "L21H1_tool_projection_delta",
            "L21H12_tool_projection_delta",
            "L24H6_tool_projection_delta",
            "MLP27_tool_projection_delta",
            "L16H4_no_tool_projection_delta",
            "MLP17_no_tool_projection_delta",
            "L23H6_no_tool_projection_delta",
        ),
    )

    write_csv(direction_summary, out_root / "route_direction_summary.csv")
    write_csv(node_patch_summary, out_root / "route_node_patch_summary.csv")
    write_csv(module_patch_summary, out_root / "route_module_patch_summary.csv")
    write_csv(intervention_summary, out_root / "route_direction_intervention_summary.csv")
    write_csv(pairwise_direction_cosines, out_root / "route_direction_pairwise_cosine.csv")

    summary = {
        "dataset_root": str(dataset_root),
        "n_samples": int(direction_counts),
        "anchor_nodes": list(DECISION_NODES),
        "support_nodes": list(UPSTREAM_SUPPORT_NODES + ("L20H5", "L21H12", "L16H4", "MLP17")),
        "candidate_nodes": ["L12H6", "L13H9", "L17H8"],
        "direction_summary_csv": str(out_root / "route_direction_summary.csv"),
        "node_patch_summary_csv": str(out_root / "route_node_patch_summary.csv"),
        "module_patch_summary_csv": str(out_root / "route_module_patch_summary.csv"),
        "intervention_summary_csv": str(out_root / "route_direction_intervention_summary.csv"),
        "pairwise_direction_cosine_csv": str(out_root / "route_direction_pairwise_cosine.csv"),
        "attention_support_heads": attention_support,
    }
    (out_root / "output_route_decision_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    build_report(
        out_root=out_root,
        direction_rows=direction_summary,
        node_patch_rows=node_patch_rows,
        module_patch_rows=module_patch_rows,
        intervention_rows=intervention_rows,
        attention_support=attention_support,
        pairwise_cosines=pairwise_direction_cosines,
    )


if __name__ == "__main__":
    main()
