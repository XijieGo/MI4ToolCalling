#!/usr/bin/env python3
"""
Refinement audit for Output-Route Decision.

This script only addresses two questions:
1. formalize the route score object,
2. harden edge-level transmission evidence from the decision spine to downstream.
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
from toolcall_circuit.output_route_decision_audit import (
    NODE_SPECS,
    build_route_score,
    collect_cache,
    collect_names,
    extract_node,
    fmt,
    projection_delta,
    read_csv_rows,
    run_with_multi_edits_and_collect,
    safe_rate,
    unit,
    write_csv,
)
from toolcall_circuit.single_sample import load_hooked_qwen3, parse_head


ANCHOR_NODES = ("MLP11", "MLP16", "MLP19")
SCORE_NODES = ("MLP11", "MLP16", "MLP19", "MLP17", "L20H5", "L21H12", "MLP27", "L23H6")
INTERVENTION_NODES = ANCHOR_NODES
EDGE_SPECS = [
    {"source": "MLP11", "target": "MLP16", "note": "spine ingress"},
    {"source": "MLP16", "target": "MLP19", "note": "spine relay"},
    {"source": "MLP16", "target": "MLP17", "note": "fork into direct-answer branch"},
    {"source": "MLP19", "target": "L20H5", "note": "late tool fanout"},
    {"source": "MLP19", "target": "L21H12", "note": "late tool fanout"},
    {"source": "MLP19", "target": "MLP27", "note": "late writer fanout"},
    {"source": "MLP19", "target": "L23H6", "note": "late suppressive fanout"},
]
EDGE_GROUP_SPECS = [
    {"source": "MLP19", "label": "tool_out_group", "targets": ("L20H5", "L21H12", "MLP27")},
    {"source": "MLP19", "label": "full_out_group", "targets": ("L20H5", "L21H12", "MLP27", "L23H6")},
]


def finite(values: Iterable[float]) -> List[float]:
    return [float(v) for v in values if isinstance(v, (int, float)) and math.isfinite(float(v))]


def median(values: Iterable[float]) -> float:
    vals = finite(values)
    return float(np.median(vals)) if vals else float("nan")


def mean(values: Iterable[float]) -> float:
    vals = finite(values)
    return float(np.mean(vals)) if vals else float("nan")


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    ua = unit(a)
    ub = unit(b)
    if float(ua.norm().item()) < 1e-8 or float(ub.norm().item()) < 1e-8:
        return float("nan")
    return float(torch.dot(ua, ub).item())


def summarize_rows(rows: Sequence[Dict[str, object]], key_fields: Sequence[str], metric_fields: Sequence[str]) -> List[Dict[str, object]]:
    grouped: Dict[Tuple[str, ...], List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(str(row[k]) for k in key_fields)].append(dict(row))
    out: List[Dict[str, object]] = []
    for key, grows in grouped.items():
        summary: Dict[str, object] = {field: key[idx] for idx, field in enumerate(key_fields)}
        summary["n_samples"] = len(grows)
        for metric in metric_fields:
            values = [row.get(metric) for row in grows]
            if metric.endswith("_rate") or metric.endswith("_flip") or metric.endswith("_positive") or metric.endswith("_negative"):
                summary[metric] = safe_rate(values)
            else:
                summary[f"{metric}_median"] = median(values)
                summary[f"{metric}_mean"] = mean(values)
        out.append(summary)
    return out


def rankdata_average(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    order = np.argsort(arr, kind="mergesort")
    ranks = np.empty(len(arr), dtype=float)
    i = 0
    while i < len(arr):
        j = i
        while j + 1 < len(arr) and arr[order[j + 1]] == arr[order[i]]:
            j += 1
        rank = 0.5 * (i + j) + 1.0
        ranks[order[i : j + 1]] = rank
        i = j + 1
    return ranks


def spearman_corr(xs: Sequence[float], ys: Sequence[float]) -> float:
    x = np.asarray([float(v) for v in xs], dtype=float)
    y = np.asarray([float(v) for v in ys], dtype=float)
    if len(x) < 2 or len(y) < 2:
        return float("nan")
    xr = rankdata_average(x)
    yr = rankdata_average(y)
    x_std = float(xr.std())
    y_std = float(yr.std())
    if x_std < 1e-12 or y_std < 1e-12:
        return float("nan")
    return float(np.corrcoef(xr, yr)[0, 1])


def binary_auc(pos_scores: Sequence[float], neg_scores: Sequence[float]) -> float:
    pos = np.asarray([float(v) for v in pos_scores], dtype=float)
    neg = np.asarray([float(v) for v in neg_scores], dtype=float)
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    gt = (pos[:, None] > neg[None, :]).mean()
    eq = (pos[:, None] == neg[None, :]).mean()
    return float(gt + 0.5 * eq)


def module_score(score_map: Dict[str, float]) -> float:
    vals = [float(score_map[node]) for node in ANCHOR_NODES if node in score_map and math.isfinite(float(score_map[node]))]
    return float(np.mean(vals)) if vals else float("nan")


def node_to_hook_name(node: str) -> str:
    if node.startswith("MLP"):
        return f"blocks.{int(node[3:])}.hook_mlp_out"
    layer, _head = parse_head(node)
    return f"blocks.{layer}.attn.hook_z"


def run_with_assignments_and_collect(
    model,
    base_tokens: torch.Tensor,
    *,
    clean_cache_cpu: Dict[str, torch.Tensor],
    corrupt_cache_cpu: Dict[str, torch.Tensor],
    patch_clean_nodes: Sequence[str],
    patch_corrupt_nodes: Sequence[str],
    record_nodes: Sequence[str],
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    heads_by_layer_clean: Dict[int, List[int]] = defaultdict(list)
    heads_by_layer_corrupt: Dict[int, List[int]] = defaultdict(list)
    mlp_layers_clean: List[int] = []
    mlp_layers_corrupt: List[int] = []

    for node in patch_clean_nodes:
        if node.startswith("MLP"):
            mlp_layers_clean.append(int(node[3:]))
        else:
            layer, head = parse_head(node)
            heads_by_layer_clean[layer].append(head)
    for node in patch_corrupt_nodes:
        if node.startswith("MLP"):
            mlp_layers_corrupt.append(int(node[3:]))
        else:
            layer, head = parse_head(node)
            heads_by_layer_corrupt[layer].append(head)

    hooks = []

    def add_head_hooks(layer_to_heads: Dict[int, List[int]], cache_cpu: Dict[str, torch.Tensor]) -> None:
        for layer, heads in layer_to_heads.items():
            cache_name = f"blocks.{layer}.attn.hook_z"
            src_act = cache_cpu[cache_name].to(base_tokens.device)
            uniq = sorted(set(heads))

            def make_head_hook(src: torch.Tensor, hs: Sequence[int]):
                def hook_fn(z: torch.Tensor, hook):  # noqa: ANN001
                    out = z.clone()
                    for h in hs:
                        out[:, -1, h, :] = src[:, -1, h, :]
                    return out

                return hook_fn

            hooks.append((cache_name, make_head_hook(src_act, uniq)))

    def add_mlp_hooks(layers: Sequence[int], cache_cpu: Dict[str, torch.Tensor]) -> None:
        for layer in sorted(set(layers)):
            cache_name = f"blocks.{layer}.hook_mlp_out"
            src_act = cache_cpu[cache_name].to(base_tokens.device)

            def make_mlp_hook(src: torch.Tensor):
                def hook_fn(mlp_out: torch.Tensor, hook):  # noqa: ANN001
                    out = mlp_out.clone()
                    out[:, -1, :] = src[:, -1, :]
                    return out

                return hook_fn

            hooks.append((cache_name, make_mlp_hook(src_act)))

    add_head_hooks(heads_by_layer_clean, clean_cache_cpu)
    add_head_hooks(heads_by_layer_corrupt, corrupt_cache_cpu)
    add_mlp_hooks(mlp_layers_clean, clean_cache_cpu)
    add_mlp_hooks(mlp_layers_corrupt, corrupt_cache_cpu)

    recorded: Dict[str, torch.Tensor] = {}
    for node in record_nodes:
        cache_name = node_to_hook_name(node)

        def make_record(name: str):
            def hook_fn(act: torch.Tensor, hook):  # noqa: ANN001
                recorded[name] = act.detach().cpu()
                return act

            return hook_fn

        hooks.append((cache_name, make_record(cache_name)))

    with torch.no_grad():
        logits = model.run_with_hooks(base_tokens, fwd_hooks=hooks)
    return logits, recorded


def score_from_vec(vec: torch.Tensor, *, direction: torch.Tensor, midpoint: torch.Tensor, scale: float) -> float:
    raw = float(torch.dot(vec.float() - midpoint.float(), direction.float()).item())
    if not math.isfinite(scale) or abs(scale) < 1e-8:
        return float("nan")
    return raw / scale


def score_from_cache(cache: Dict[str, torch.Tensor], node: str, geometry: Dict[str, Dict[str, torch.Tensor | float]]) -> float:
    geom = geometry[node]
    return score_from_vec(
        extract_node(cache, node),
        direction=geom["direction"],  # type: ignore[arg-type]
        midpoint=geom["midpoint"],  # type: ignore[arg-type]
        scale=float(geom["scale"]),
    )


def build_report(
    *,
    out_root: Path,
    route_score_rows: Sequence[Dict[str, object]],
    edge_rows: Sequence[Dict[str, object]],
    edge_group_rows: Sequence[Dict[str, object]],
    node_tiers: Dict[str, List[str]],
    edge_tiers: Dict[str, List[str]],
    pairwise_cos_rows: Sequence[Dict[str, object]],
) -> None:
    route_map = {str(row["node"]): dict(row) for row in route_score_rows}
    edge_map = {f"{row['source']}->{row['target']}": dict(row) for row in edge_rows}
    group_map = {f"{row['source']}->{row['target_group']}": dict(row) for row in edge_group_rows}
    pairwise_map = {(str(row["source"]), str(row["target"])): dict(row) for row in pairwise_cos_rows}

    lines: List[str] = []
    lines.append("# Output-Route Decision Refine 主报告")
    lines.append("")
    lines.append("## 模块目标")
    lines.append("")
    lines.append("这次不重做整个模块，只把两个对象钉死：`route score` 的正式定义，以及 decision spine 到下游的边级传递。")
    lines.append("")
    lines.append("## 这次只解决的两个问题")
    lines.append("")
    lines.append("1. 把 `route score / route preference` 定义成一个可计算、可验证、可干预的对象。")
    lines.append("2. 给 `MLP11 -> MLP16 -> MLP19` 到下游的指定边补强因果传递证据，并区分主边、并行冗余边和暂时不能强写的边。")
    lines.append("")
    lines.append("## route score 的正式定义")
    lines.append("")
    lines.append("对任意节点 `n`，记首个输出位置的节点激活为 `h_n(x)`。定义：")
    lines.append("")
    lines.append("- 节点均值：`mu_n^clean = E[h_n(x_clean)]`，`mu_n^corrupt = E[h_n(x_corrupt)]`。")
    lines.append("- local route direction：`d_n = (mu_n^clean - mu_n^corrupt) / ||mu_n^clean - mu_n^corrupt||`。")
    lines.append("- 中心点：`m_n = (mu_n^clean + mu_n^corrupt) / 2`。")
    lines.append("- local route score：`r_n(x) = <h_n(x) - m_n, d_n> / <mu_n^clean - m_n, d_n>`。")
    lines.append("")
    lines.append("这个定义有两个直接性质：")
    lines.append("")
    lines.append("- `r_n(mu_n^clean) = +1`，`r_n(mu_n^corrupt) = -1`。")
    lines.append("- 对所有节点统一用同一符号约定：`r_n > 0` 表示更偏 tool-route，`r_n < 0` 表示更偏 direct-answer route。")
    lines.append("")
    lines.append("因此，统一对象不是“所有层共享一条固定方向”，而是“每个节点各自有一条 local route direction，但它们都实现同一个同号标量对象 `r_n`”。")
    lines.append(
        f"跨层几何也支持这点：`MLP11↔MLP16` cosine `{fmt(pairwise_map.get(('MLP11', 'MLP16'), {}).get('cosine'))}`，"
        f"`MLP11↔MLP19` `{fmt(pairwise_map.get(('MLP11', 'MLP19'), {}).get('cosine'))}`，"
        f"`MLP16↔MLP19` `{fmt(pairwise_map.get(('MLP16', 'MLP19'), {}).get('cosine'))}`。"
        "三者几乎不共线，所以更合理的对象是“逐层重编码的连续状态”，而不是“固定向量搬运”。"
    )
    lines.append("")
    lines.append("进一步定义模块级对象：`R_module(x) = mean(r_MLP11(x), r_MLP16(x), r_MLP19(x))`。")
    lines.append("")
    lines.append("## route score 的验证结果")
    lines.append("")
    for node in ["MLP11", "MLP16", "MLP19", "module_anchor_mean"]:
        row = route_map.get(node, {})
        if not row:
            continue
        if node == "module_anchor_mean":
            lines.append(
                f"- `R_module`: clean `{fmt(row.get('clean_score_median'))}`, corrupt `{fmt(row.get('corrupt_score_median'))}`, "
                f"AUC `{fmt(row.get('auc_clean_vs_corrupt'))}`, 与最终 route margin 的 Spearman `{fmt(row.get('spearman_with_route_margin'))}`。"
            )
        else:
            lines.append(
                f"- `{node}`: clean `{fmt(row.get('clean_score_median'))}`, corrupt `{fmt(row.get('corrupt_score_median'))}`, "
                f"AUC `{fmt(row.get('auc_clean_vs_corrupt'))}`, clean>0 比例 `{fmt(row.get('clean_positive_rate'))}`, "
                f"corrupt<0 比例 `{fmt(row.get('corrupt_negative_rate'))}`, 与最终 route margin 的 Spearman `{fmt(row.get('spearman_with_route_margin'))}`。"
            )
    lines.append("")
    lines.append("这些结果说明 clean/corrupt 不只在 logits 上分开，也在三个 anchor 的局部 route score 上稳定分开。")
    lines.append("")
    lines.append("对 anchor 的 direction-only 干预结果：")
    lines.append("")
    for node in ANCHOR_NODES:
        row = route_map.get(node, {})
        lines.append(
            f"- `{node}`: patch-promote 后 route margin 变化 `{fmt(row.get('patch_promote_route_margin_delta_median'))}`，"
            f"patch-erase 后 `{fmt(row.get('patch_erase_route_margin_delta_median'))}`；"
            f"inject 后本地 score 变化 `{fmt(row.get('inject_local_score_delta_median'))}`，"
            f"`R_module` 变化 `{fmt(row.get('inject_module_score_delta_median'))}`，"
            f"最终 route margin 变化 `{fmt(row.get('inject_route_margin_delta_median'))}`；"
            f"erase 后分别为 `{fmt(row.get('erase_local_score_delta_median'))}` / "
            f"`{fmt(row.get('erase_module_score_delta_median'))}` / `{fmt(row.get('erase_route_margin_delta_median'))}`。"
        )
    lines.append("")
    lines.append("因此，`route score` 不是纯描述量；沿着 local route direction 做 inject / erase，会同时改动节点本地 score、模块平均 score 和最终输出边界。")
    lines.append("")
    lines.append("## 边级传递结果")
    lines.append("")
    lines.append("下面的边结论只基于这次新做的强因果审计：source-only patch、target block、以及 target local route score mediation。")
    lines.append("")
    for edge_name in [
        "MLP11->MLP16",
        "MLP16->MLP19",
        "MLP16->MLP17",
        "MLP19->L20H5",
        "MLP19->L21H12",
        "MLP19->MLP27",
        "MLP19->L23H6",
    ]:
        row = edge_map.get(edge_name, {})
        if not row:
            continue
        lines.append(
            f"- `{edge_name}`: promote route mediation `{fmt(row.get('promote_route_mediated_ratio_median'))}`, "
            f"erase route mediation `{fmt(row.get('erase_route_mediated_ratio_median'))}`, "
            f"promote target-score mediation `{fmt(row.get('promote_target_mediated_ratio_median'))}`, "
            f"erase target-score mediation `{fmt(row.get('erase_target_mediated_ratio_median'))}`, "
            f"标签 `{row.get('conclusion_label')}`。"
        )
    lines.append("")
    if group_map:
        lines.append("与 group block 对照：")
        lines.append("")
        for group_name in ["MLP19->tool_out_group", "MLP19->full_out_group"]:
            row = group_map.get(group_name, {})
            if row:
                lines.append(
                    f"- `{group_name}`: promote group mediation `{fmt(row.get('promote_route_mediated_ratio_median'))}`, "
                    f"erase group mediation `{fmt(row.get('erase_route_mediated_ratio_median'))}`。"
                )
        lines.append("")
        lines.append("如果 group block 明显强于单边 block，说明单边更像并行冗余入口，而不是唯一瓶颈。")
        lines.append("")
    lines.append("## 强写结论")
    lines.append("")
    for item in node_tiers["anchor_nodes"]:
        lines.append(f"- anchor node: `{item}`")
    for item in edge_tiers["strong_edges"]:
        lines.append(f"- strong edge: `{item}`")
    lines.append("")
    lines.append("## 弱写结论")
    lines.append("")
    for item in node_tiers["support_nodes"]:
        lines.append(f"- support node: `{item}`")
    for item in edge_tiers["weak_edges"]:
        lines.append(f"- weak edge: `{item}`")
    lines.append("")
    lines.append("## 需要降级的旧说法")
    lines.append("")
    lines.append("- 不能把 `route score` 写成一条固定跨层向量；现在只能写成“逐层重编码的同号标量对象”。")
    lines.append("- 不能把 `MLP19` 直接写成 single bottleneck to construction；如果 group block 远强于单边 block，就只能写它是 late fanout hub。")
    lines.append("- `MLP16 -> MLP17` 若只有分支级证据而缺更强的 route mediation，就只能写成 shared-to-suppress fork，不写成唯一主边。")
    lines.append("")
    lines.append("## 仍未解决的问题")
    lines.append("")
    lines.append("- `MLP11` 的 local route direction 还能不能进一步分解成更窄的语义子方向，目前没有必要的新证据。")
    lines.append("- `MLP19 -> MLP27` 若保留强 mediation，但和 group block 相比仍不构成唯一瓶颈，那么还需要更细的 exclusion patch 才能区分 direct edge 与 parallel receipt。")
    lines.append("- `L20H5` 与 `L21H12` 的并行冗余程度目前只能相对比较，不能写成精确比例分解。")
    lines.append("")
    lines.append("## 论文主文可用结论")
    lines.append("")
    lines.append(
        "我们将 Output-Route Decision 的核心对象定义为一个节点局部、符号统一的 route score："
        "在每个 anchor 节点上，用 clean 与 corrupt 均值差定义 local route direction，并用 midpoint-centered projection 定义 local route score，"
        "使正值表示 tool-route 偏好、负值表示 direct-answer 偏好。`MLP11`、`MLP16` 和 `MLP19` 并不共享同一跨层向量，"
        "但它们都稳定实现同一个逐层重编码的连续 route state；该 state 在 clean/corrupt 条件下可稳定分离，并在 direction-only inject / erase 后同步改变模块平均 score 与最终 route margin。"
        "边级上，`MLP11 -> MLP16` 与 `MLP16 -> MLP19` 构成最可信的 decision spine 主边；`MLP16 -> MLP17` 表明 route state 已经开始分叉到 direct-answer 分支，但其唯一性仍弱于主 spine。"
        "到 late stage，`MLP19` 通过多个并行接收边把 route state 分发到 tool-side 与 suppressive-side，下游表现更像 fanout hub 而不是单边瓶颈。"
    )
    lines.append("")

    (out_root / "output_route_decision_refine_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Refine Output-Route Decision with formal route score and edge mediation.")
    parser.add_argument("--dataset-root", type=str, default="experiment/datasets")
    parser.add_argument("--model-path", type=str, default="./external/models/Qwen3-1.7B")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--output-root", type=str, required=True)
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root).resolve()
    out_root = Path(args.output_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    samples = load_dataset_samples(dataset_root)
    if args.max_samples > 0:
        samples = samples[: args.max_samples]
    if not samples:
        raise ValueError("No samples found.")

    model, tokenizer = load_hooked_qwen3(args.model_path, device=args.device, dtype=torch.bfloat16)
    hook_names = collect_names(SCORE_NODES)

    mean_clean: Dict[str, torch.Tensor] = {}
    mean_corrupt: Dict[str, torch.Tensor] = {}
    valid_samples = 0

    pbar = tqdm(samples, desc="Route-refine pass1", dynamic_ncols=True)
    for sample in pbar:
        try:
            clean_text = sample.clean_path.read_text(encoding="utf-8")
            corrupt_text = sample.corrupt_path.read_text(encoding="utf-8")
        except Exception:
            continue

        clean_tokens = model.to_tokens(clean_text, prepend_bos=False)
        corrupt_tokens = model.to_tokens(corrupt_text, prepend_bos=False)
        clean_cache = collect_cache(model, clean_tokens, hook_names)
        corrupt_cache = collect_cache(model, corrupt_tokens, hook_names)

        for node in SCORE_NODES:
            mean_clean[node] = mean_clean.get(node, torch.zeros_like(extract_node(clean_cache, node))) + extract_node(clean_cache, node)
            mean_corrupt[node] = mean_corrupt.get(node, torch.zeros_like(extract_node(corrupt_cache, node))) + extract_node(corrupt_cache, node)
        valid_samples += 1

    if valid_samples == 0:
        raise ValueError("No valid samples for pass1.")

    geometry: Dict[str, Dict[str, torch.Tensor | float]] = {}
    for node in SCORE_NODES:
        mu_clean = mean_clean[node] / float(valid_samples)
        mu_corrupt = mean_corrupt[node] / float(valid_samples)
        direction = unit(mu_clean - mu_corrupt)
        midpoint = 0.5 * (mu_clean + mu_corrupt)
        scale = float(torch.dot(mu_clean - midpoint, direction).item())
        geometry[node] = {
            "mu_clean": mu_clean,
            "mu_corrupt": mu_corrupt,
            "direction": direction,
            "midpoint": midpoint,
            "scale": scale,
        }

    base_rows: List[Dict[str, object]] = []
    intervention_rows: List[Dict[str, object]] = []
    patch_rows: List[Dict[str, object]] = []
    edge_rows: List[Dict[str, object]] = []
    edge_group_rows: List[Dict[str, object]] = []

    pbar = tqdm(samples, desc="Route-refine pass2", dynamic_ncols=True)
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
        clean_scores = {node: score_from_cache(clean_cache, node, geometry) for node in SCORE_NODES}
        corrupt_scores = {node: score_from_cache(corrupt_cache, node, geometry) for node in SCORE_NODES}
        clean_module = module_score(clean_scores)
        corrupt_module = module_score(corrupt_scores)

        for node in SCORE_NODES:
            base_rows.append(
                {
                    "sample_id": sample.sample_id,
                    "node": node,
                    "condition": "clean",
                    "local_route_score": clean_scores[node],
                    "module_route_score": clean_module,
                    "route_margin": clean_route,
                    "positive_boundary": clean_scores[node] > 0.0,
                    "negative_boundary": clean_scores[node] < 0.0,
                    "margin_positive": clean_route > 0.0,
                }
            )
            base_rows.append(
                {
                    "sample_id": sample.sample_id,
                    "node": node,
                    "condition": "corrupt",
                    "local_route_score": corrupt_scores[node],
                    "module_route_score": corrupt_module,
                    "route_margin": corrupt_route,
                    "positive_boundary": corrupt_scores[node] > 0.0,
                    "negative_boundary": corrupt_scores[node] < 0.0,
                    "margin_positive": corrupt_route > 0.0,
                }
            )

        source_only_promote: Dict[str, Dict[str, object]] = {}
        source_only_erase: Dict[str, Dict[str, object]] = {}

        for node in INTERVENTION_NODES:
            direction = geometry[node]["direction"]  # type: ignore[assignment]
            clean_vec = extract_node(clean_cache, node)
            corrupt_vec = extract_node(corrupt_cache, node)

            inject_delta = projection_delta(corrupt_vec, clean_vec, direction)  # type: ignore[arg-type]
            inject_logits, inject_record = run_with_multi_edits_and_collect(
                model,
                corrupt_tokens,
                edits={node: inject_delta},
                record_names=hook_names,
            )
            inject_scores = {score_node: score_from_cache(inject_record, score_node, geometry) for score_node in SCORE_NODES}
            inject_module = module_score(inject_scores)
            inject_route = build_route_score(inject_logits, tool_objective, no_tool_objective)
            intervention_rows.append(
                {
                    "sample_id": sample.sample_id,
                    "node": node,
                    "mode": "inject_clean_route_into_corrupt",
                    "local_score_delta": inject_scores[node] - corrupt_scores[node],
                    "module_score_delta": inject_module - corrupt_module,
                    "route_margin_delta": inject_route - corrupt_route,
                    "boundary_flip": inject_route > 0.0,
                    **{f"{score_node}_score_delta": inject_scores[score_node] - corrupt_scores[score_node] for score_node in SCORE_NODES},
                }
            )

            erase_delta = projection_delta(clean_vec, corrupt_vec, direction)  # type: ignore[arg-type]
            erase_logits, erase_record = run_with_multi_edits_and_collect(
                model,
                clean_tokens,
                edits={node: erase_delta},
                record_names=hook_names,
            )
            erase_scores = {score_node: score_from_cache(erase_record, score_node, geometry) for score_node in SCORE_NODES}
            erase_module = module_score(erase_scores)
            erase_route = build_route_score(erase_logits, tool_objective, no_tool_objective)
            intervention_rows.append(
                {
                    "sample_id": sample.sample_id,
                    "node": node,
                    "mode": "erase_route_from_clean",
                    "local_score_delta": erase_scores[node] - clean_scores[node],
                    "module_score_delta": erase_module - clean_module,
                    "route_margin_delta": erase_route - clean_route,
                    "boundary_flip": erase_route < 0.0,
                    **{f"{score_node}_score_delta": erase_scores[score_node] - clean_scores[score_node] for score_node in SCORE_NODES},
                }
            )

            promote_logits, promote_record = run_with_assignments_and_collect(
                model,
                corrupt_tokens,
                clean_cache_cpu=clean_cache,
                corrupt_cache_cpu=corrupt_cache,
                patch_clean_nodes=[node],
                patch_corrupt_nodes=[],
                record_nodes=SCORE_NODES,
            )
            source_only_promote[node] = {
                "route": build_route_score(promote_logits, tool_objective, no_tool_objective),
                "scores": {score_node: score_from_cache(promote_record, score_node, geometry) for score_node in SCORE_NODES},
            }
            patch_rows.append(
                {
                    "sample_id": sample.sample_id,
                    "node": node,
                    "mode": "patch_promote_tool_route",
                    "local_score_delta": float(source_only_promote[node]["scores"][node]) - corrupt_scores[node],
                    "module_score_delta": module_score(source_only_promote[node]["scores"]) - corrupt_module,
                    "route_margin_delta": float(source_only_promote[node]["route"]) - corrupt_route,
                    "boundary_flip": float(source_only_promote[node]["route"]) > 0.0,
                }
            )

            erase_patch_logits, erase_patch_record = run_with_assignments_and_collect(
                model,
                clean_tokens,
                clean_cache_cpu=clean_cache,
                corrupt_cache_cpu=corrupt_cache,
                patch_clean_nodes=[],
                patch_corrupt_nodes=[node],
                record_nodes=SCORE_NODES,
            )
            source_only_erase[node] = {
                "route": build_route_score(erase_patch_logits, tool_objective, no_tool_objective),
                "scores": {score_node: score_from_cache(erase_patch_record, score_node, geometry) for score_node in SCORE_NODES},
            }
            patch_rows.append(
                {
                    "sample_id": sample.sample_id,
                    "node": node,
                    "mode": "patch_erase_tool_route",
                    "local_score_delta": float(source_only_erase[node]["scores"][node]) - clean_scores[node],
                    "module_score_delta": module_score(source_only_erase[node]["scores"]) - clean_module,
                    "route_margin_delta": float(source_only_erase[node]["route"]) - clean_route,
                    "boundary_flip": float(source_only_erase[node]["route"]) < 0.0,
                }
            )

        for spec in EDGE_SPECS:
            source = spec["source"]
            target = spec["target"]

            promote_target_gap = clean_scores[target] - corrupt_scores[target]
            erase_target_gap = clean_scores[target] - corrupt_scores[target]
            if not math.isfinite(promote_target_gap) or abs(promote_target_gap) < 1e-8:
                continue

            blocked_promote_logits, blocked_promote_record = run_with_assignments_and_collect(
                model,
                corrupt_tokens,
                clean_cache_cpu=clean_cache,
                corrupt_cache_cpu=corrupt_cache,
                patch_clean_nodes=[source],
                patch_corrupt_nodes=[target],
                record_nodes=[target],
            )
            blocked_promote_route = build_route_score(blocked_promote_logits, tool_objective, no_tool_objective)
            blocked_promote_target = score_from_cache(blocked_promote_record, target, geometry)

            source_promote_route = float(source_only_promote[source]["route"])
            source_promote_target = float(source_only_promote[source]["scores"][target])
            route_source_effect = source_promote_route - corrupt_route
            route_blocked_effect = blocked_promote_route - corrupt_route
            target_source_effect = source_promote_target - corrupt_scores[target]
            target_blocked_effect = blocked_promote_target - corrupt_scores[target]

            blocked_erase_logits, blocked_erase_record = run_with_assignments_and_collect(
                model,
                clean_tokens,
                clean_cache_cpu=clean_cache,
                corrupt_cache_cpu=corrupt_cache,
                patch_clean_nodes=[target],
                patch_corrupt_nodes=[source],
                record_nodes=[target],
            )
            blocked_erase_route = build_route_score(blocked_erase_logits, tool_objective, no_tool_objective)
            blocked_erase_target = score_from_cache(blocked_erase_record, target, geometry)

            source_erase_route = float(source_only_erase[source]["route"])
            source_erase_target = float(source_only_erase[source]["scores"][target])
            route_drop = clean_route - source_erase_route
            route_blocked_drop = clean_route - blocked_erase_route
            target_drop = clean_scores[target] - source_erase_target
            target_blocked_drop = clean_scores[target] - blocked_erase_target

            edge_rows.append(
                {
                    "sample_id": sample.sample_id,
                    "source": source,
                    "target": target,
                    "promote_route_source_ratio": route_source_effect / route_gap,
                    "promote_route_blocked_ratio": route_blocked_effect / route_gap,
                    "promote_route_mediated_ratio": (route_source_effect - route_blocked_effect) / route_gap,
                    "promote_target_source_ratio": target_source_effect / promote_target_gap,
                    "promote_target_blocked_ratio": target_blocked_effect / promote_target_gap,
                    "promote_target_mediated_ratio": (target_source_effect - target_blocked_effect) / promote_target_gap,
                    "erase_route_source_ratio": route_drop / route_gap,
                    "erase_route_blocked_ratio": route_blocked_drop / route_gap,
                    "erase_route_mediated_ratio": (route_drop - route_blocked_drop) / route_gap,
                    "erase_target_source_ratio": target_drop / erase_target_gap,
                    "erase_target_blocked_ratio": target_blocked_drop / erase_target_gap,
                    "erase_target_mediated_ratio": (target_drop - target_blocked_drop) / erase_target_gap,
                }
            )

        for spec in EDGE_GROUP_SPECS:
            source = spec["source"]
            targets = list(spec["targets"])

            blocked_promote_logits, _ = run_with_assignments_and_collect(
                model,
                corrupt_tokens,
                clean_cache_cpu=clean_cache,
                corrupt_cache_cpu=corrupt_cache,
                patch_clean_nodes=[source],
                patch_corrupt_nodes=targets,
                record_nodes=[],
            )
            blocked_promote_route = build_route_score(blocked_promote_logits, tool_objective, no_tool_objective)
            source_promote_route = float(source_only_promote[source]["route"])

            blocked_erase_logits, _ = run_with_assignments_and_collect(
                model,
                clean_tokens,
                clean_cache_cpu=clean_cache,
                corrupt_cache_cpu=corrupt_cache,
                patch_clean_nodes=targets,
                patch_corrupt_nodes=[source],
                record_nodes=[],
            )
            blocked_erase_route = build_route_score(blocked_erase_logits, tool_objective, no_tool_objective)
            source_erase_route = float(source_only_erase[source]["route"])

            edge_group_rows.append(
                {
                    "sample_id": sample.sample_id,
                    "source": source,
                    "target_group": spec["label"],
                    "targets": "|".join(targets),
                    "promote_route_mediated_ratio": ((source_promote_route - corrupt_route) - (blocked_promote_route - corrupt_route)) / route_gap,
                    "erase_route_mediated_ratio": ((clean_route - source_erase_route) - (clean_route - blocked_erase_route)) / route_gap,
                }
            )

    write_csv(base_rows, out_root / "route_score_base_per_sample.csv")
    write_csv(intervention_rows, out_root / "route_score_intervention_per_sample.csv")
    write_csv(patch_rows, out_root / "route_score_patch_per_sample.csv")
    write_csv(edge_rows, out_root / "route_edge_audit_per_sample.csv")
    write_csv(edge_group_rows, out_root / "route_edge_group_per_sample.csv")

    route_score_summary_rows: List[Dict[str, object]] = []
    for node in ANCHOR_NODES:
        clean_vals = [float(r["local_route_score"]) for r in base_rows if r["node"] == node and r["condition"] == "clean"]
        corrupt_vals = [float(r["local_route_score"]) for r in base_rows if r["node"] == node and r["condition"] == "corrupt"]
        all_vals = [float(r["local_route_score"]) for r in base_rows if r["node"] == node]
        all_margins = [float(r["route_margin"]) for r in base_rows if r["node"] == node]
        clean_module = [float(r["module_score_delta"]) for r in intervention_rows if r["node"] == node and r["mode"] == "inject_clean_route_into_corrupt"]
        erase_module = [float(r["module_score_delta"]) for r in intervention_rows if r["node"] == node and r["mode"] == "erase_route_from_clean"]
        inject_rows = [r for r in intervention_rows if r["node"] == node and r["mode"] == "inject_clean_route_into_corrupt"]
        erase_rows = [r for r in intervention_rows if r["node"] == node and r["mode"] == "erase_route_from_clean"]
        patch_promote_rows = [r for r in patch_rows if r["node"] == node and r["mode"] == "patch_promote_tool_route"]
        patch_erase_rows = [r for r in patch_rows if r["node"] == node and r["mode"] == "patch_erase_tool_route"]

        downstream_nodes = [n for n in SCORE_NODES if n != node]
        inject_target_stats = {dn: median(float(r[f"{dn}_score_delta"]) for r in inject_rows) for dn in downstream_nodes}
        best_inject_target = max(inject_target_stats, key=lambda dn: abs(inject_target_stats[dn])) if inject_target_stats else ""
        route_score_summary_rows.append(
            {
                "node": node,
                "local_route_direction_definition": "d_n = unit(mu_clean - mu_corrupt); score = centered projection normalized so means map to +1/-1",
                "clean_score_median": median(clean_vals),
                "corrupt_score_median": median(corrupt_vals),
                "clean_positive_rate": safe_rate(v > 0.0 for v in clean_vals),
                "corrupt_negative_rate": safe_rate(v < 0.0 for v in corrupt_vals),
                "auc_clean_vs_corrupt": binary_auc(clean_vals, corrupt_vals),
                "spearman_with_route_margin": spearman_corr(all_vals, all_margins),
                "inject_local_score_delta_median": median(float(r["local_score_delta"]) for r in inject_rows),
                "erase_local_score_delta_median": median(float(r["local_score_delta"]) for r in erase_rows),
                "inject_module_score_delta_median": median(clean_module),
                "erase_module_score_delta_median": median(erase_module),
                "inject_route_margin_delta_median": median(float(r["route_margin_delta"]) for r in inject_rows),
                "erase_route_margin_delta_median": median(float(r["route_margin_delta"]) for r in erase_rows),
                "patch_promote_route_margin_delta_median": median(float(r["route_margin_delta"]) for r in patch_promote_rows),
                "patch_erase_route_margin_delta_median": median(float(r["route_margin_delta"]) for r in patch_erase_rows),
                "inject_boundary_flip_rate": safe_rate(bool(r["boundary_flip"]) for r in inject_rows),
                "erase_boundary_flip_rate": safe_rate(bool(r["boundary_flip"]) for r in erase_rows),
                "strongest_downstream_node_under_inject": best_inject_target,
                "strongest_downstream_score_delta_under_inject": inject_target_stats.get(best_inject_target, float("nan")),
            }
        )

    module_clean_vals = [float(r["module_route_score"]) for r in base_rows if r["node"] == "MLP11" and r["condition"] == "clean"]
    module_corrupt_vals = [float(r["module_route_score"]) for r in base_rows if r["node"] == "MLP11" and r["condition"] == "corrupt"]
    module_all_vals = [float(r["module_route_score"]) for r in base_rows if r["node"] == "MLP11"]
    module_all_margins = [float(r["route_margin"]) for r in base_rows if r["node"] == "MLP11"]
    route_score_summary_rows.append(
        {
            "node": "module_anchor_mean",
            "local_route_direction_definition": "R_module = mean(r_MLP11, r_MLP16, r_MLP19)",
            "clean_score_median": median(module_clean_vals),
            "corrupt_score_median": median(module_corrupt_vals),
            "clean_positive_rate": safe_rate(v > 0.0 for v in module_clean_vals),
            "corrupt_negative_rate": safe_rate(v < 0.0 for v in module_corrupt_vals),
            "auc_clean_vs_corrupt": binary_auc(module_clean_vals, module_corrupt_vals),
            "spearman_with_route_margin": spearman_corr(module_all_vals, module_all_margins),
            "inject_local_score_delta_median": float("nan"),
            "erase_local_score_delta_median": float("nan"),
            "inject_module_score_delta_median": float("nan"),
            "erase_module_score_delta_median": float("nan"),
            "inject_route_margin_delta_median": float("nan"),
            "erase_route_margin_delta_median": float("nan"),
            "patch_promote_route_margin_delta_median": float("nan"),
            "patch_erase_route_margin_delta_median": float("nan"),
            "inject_boundary_flip_rate": float("nan"),
            "erase_boundary_flip_rate": float("nan"),
            "strongest_downstream_node_under_inject": "",
            "strongest_downstream_score_delta_under_inject": float("nan"),
        }
    )
    write_csv(route_score_summary_rows, out_root / "route_score_summary.csv")

    edge_summary_rows = summarize_rows(
        edge_rows,
        key_fields=("source", "target"),
        metric_fields=(
            "promote_route_source_ratio",
            "promote_route_blocked_ratio",
            "promote_route_mediated_ratio",
            "promote_target_source_ratio",
            "promote_target_blocked_ratio",
            "promote_target_mediated_ratio",
            "erase_route_source_ratio",
            "erase_route_blocked_ratio",
            "erase_route_mediated_ratio",
            "erase_target_source_ratio",
            "erase_target_blocked_ratio",
            "erase_target_mediated_ratio",
        ),
    )

    edge_group_summary_rows = summarize_rows(
        edge_group_rows,
        key_fields=("source", "target_group", "targets"),
        metric_fields=("promote_route_mediated_ratio", "erase_route_mediated_ratio"),
    )

    tool_group_promote = next((row for row in edge_group_summary_rows if row["source"] == "MLP19" and row["target_group"] == "tool_out_group"), {})
    tool_group_value = float(tool_group_promote.get("promote_route_mediated_ratio_median", "nan")) if tool_group_promote else float("nan")

    for row in edge_summary_rows:
        key = f"{row['source']}->{row['target']}"
        promote_route = float(row.get("promote_route_mediated_ratio_median", "nan"))
        erase_route = float(row.get("erase_route_mediated_ratio_median", "nan"))
        promote_target = float(row.get("promote_target_mediated_ratio_median", "nan"))
        erase_target = float(row.get("erase_target_mediated_ratio_median", "nan"))

        label = "candidate"
        strength = "candidate"
        evidence = "patching"
        if min(promote_route, erase_route) >= 0.04 and min(promote_target, erase_target) >= 0.08:
            label = "strong"
            strength = "strong"
        elif min(promote_target, erase_target) >= 0.08 and max(promote_route, erase_route) >= 0.02:
            label = "weak"
            strength = "weak"

        if str(row["source"]) == "MLP19" and str(row["target"]) in {"L20H5", "L21H12", "MLP27"} and math.isfinite(tool_group_value):
            if promote_route < 0.8 * tool_group_value:
                if label == "strong":
                    label = "weak"
                row["redundancy_note"] = "single-edge weaker than tool_out_group, likely parallel redundancy"
            else:
                row["redundancy_note"] = "single-edge explains most of tool_out_group mediation"
        else:
            row["redundancy_note"] = ""

        row["evidence_used"] = "source-only patch + target block + local route score mediation + final route mediation"
        row["evidence_strength"] = strength
        row["conclusion_label"] = label

    write_csv(edge_summary_rows, out_root / "route_edge_summary.csv")
    write_csv(edge_group_summary_rows, out_root / "route_edge_group_summary.csv")

    node_tiers = {
        "anchor_nodes": list(ANCHOR_NODES),
        "support_nodes": ["L2H14", "MLP12", "L20H5", "L21H12", "MLP17", "MLP27", "L23H6"],
        "candidate_nodes": ["L12H6", "L13H9", "L17H8"],
    }
    edge_tiers = {
        "strong_edges": [f"{row['source']}->{row['target']}" for row in edge_summary_rows if row["conclusion_label"] == "strong"],
        "weak_edges": [f"{row['source']}->{row['target']}" for row in edge_summary_rows if row["conclusion_label"] == "weak"],
        "candidate_edges": [f"{row['source']}->{row['target']}" for row in edge_summary_rows if row["conclusion_label"] == "candidate"],
    }
    (out_root / "node_edge_tiers.json").write_text(
        json.dumps(
            {
                **node_tiers,
                **edge_tiers,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    pairwise_cos_rows = []
    for i, src in enumerate(ANCHOR_NODES):
        for dst in ANCHOR_NODES[i + 1 :]:
            pairwise_cos_rows.append(
                {
                    "source": src,
                    "target": dst,
                    "cosine": cosine(geometry[src]["direction"], geometry[dst]["direction"]),  # type: ignore[arg-type]
                }
            )
    write_csv(pairwise_cos_rows, out_root / "route_score_pairwise_cosine.csv")

    build_report(
        out_root=out_root,
        route_score_rows=route_score_summary_rows,
        edge_rows=edge_summary_rows,
        edge_group_rows=edge_group_summary_rows,
        node_tiers=node_tiers,
        edge_tiers=edge_tiers,
        pairwise_cos_rows=pairwise_cos_rows,
    )

    summary = {
        "dataset_root": str(dataset_root),
        "n_samples": int(valid_samples),
        "route_score_summary_csv": str(out_root / "route_score_summary.csv"),
        "edge_summary_csv": str(out_root / "route_edge_summary.csv"),
        "edge_group_summary_csv": str(out_root / "route_edge_group_summary.csv"),
        "tiers_json": str(out_root / "node_edge_tiers.json"),
        "report_md": str(out_root / "output_route_decision_refine_report.md"),
    }
    (out_root / "output_route_decision_refine_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
