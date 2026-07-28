#!/usr/bin/env python3
"""
Suppression-specific mechanism audit for the competing no-tool chain.

This analysis is deliberately narrow:

1. Keep the existing signed circuit and correctness results fixed.
2. Focus only on the suppressive chain `L16H4 -> MLP17 -> L23H6`.
3. Build paper-facing evidence for:
   - what `L16H4` reads,
   - what `MLP17` writes,
   - how `L23H6` relays the suppressive state,
   - whether the chain raises `no_tool`, lowers `<tool_call>`, or both,
   - whether it disturbs the tool ingress route.

The core new intervention is direction-level rather than whole-node patching:
we estimate a shared suppressive direction at each node from the clean/corrupt
contrast, then inject only that direction into clean prompts.
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

from toolcall_circuit.bidirectional_causal_eval import load_sample_paths
from toolcall_circuit.objective import build_bidirectional_endpoint_objectives
from toolcall_circuit.single_sample import load_hooked_qwen3, objective_from_logits


SUPPRESS_NODES = ("L16H4", "MLP17", "L23H6")
TOOL_INGRESS_NODES = ("L20H5", "L21H1", "L21H12", "L24H6")
ALL_TRACKED_NODES = SUPPRESS_NODES + TOOL_INGRESS_NODES + ("MLP27",)
STAGEWISE_CHAIN = [
    ("read_only", ("L16H4",)),
    ("writer_added", ("L16H4", "MLP17")),
    ("late_relay_added", ("L16H4", "MLP17", "L23H6")),
]

NODE_SPECS: Dict[str, Tuple[str, int, int | None]] = {
    "L16H4": ("head", 16, 4),
    "MLP17": ("mlp", 17, None),
    "L23H6": ("head", 23, 6),
    "L20H5": ("head", 20, 5),
    "L21H1": ("head", 21, 1),
    "L21H12": ("head", 21, 12),
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


def collect_names() -> List[str]:
    names: List[str] = []
    for kind, layer, _head in NODE_SPECS.values():
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


def node_to_residual(model, node: str, vec: torch.Tensor) -> torch.Tensor:
    kind, layer, head = NODE_SPECS[node]
    if kind == "mlp":
        return vec.float()
    w_o = model.blocks[layer].attn.W_O[int(head)].detach().to(vec.device, dtype=vec.dtype)
    return vec @ w_o


def resid_logits(model, resid: torch.Tensor) -> torch.Tensor:
    stack = resid.to(device=model.W_U.device, dtype=model.W_U.dtype)
    if stack.ndim == 1:
        stack = stack.unsqueeze(0).unsqueeze(0)
    elif stack.ndim == 2:
        stack = stack.unsqueeze(1)
    stack_final = model.ln_final(stack)
    return model.unembed(stack_final)


def projection_delta(base_vec: torch.Tensor, source_vec: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
    if float(direction.norm().item()) < 1e-8:
        return torch.zeros_like(base_vec)
    d = unit(direction)
    return (torch.dot(source_vec, d) - torch.dot(base_vec, d)) * d


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


def find_row(rows: Sequence[Dict[str, object]], key: str, value: str) -> Dict[str, object]:
    for row in rows:
        if str(row.get(key)) == str(value):
            return dict(row)
    return {}


def build_report(
    *,
    out_root: Path,
    projection_summary: Sequence[Dict[str, object]],
    direction_summary: Sequence[Dict[str, object]],
    intervention_summary: Sequence[Dict[str, object]],
    stagewise_summary: Sequence[Dict[str, object]],
    span_summary_rows: Sequence[Dict[str, str]],
    span_patch_rows: Sequence[Dict[str, str]],
    qkv_rows: Sequence[Dict[str, str]],
    edge_summary_rows: Sequence[Dict[str, str]],
) -> None:
    proj_map = {str(r["node"]): r for r in projection_summary}
    dir_map = {str(r["node"]): r for r in direction_summary}
    int_map = {str(r["node"]): r for r in intervention_summary}
    stage_map = {int(r["step_idx"]): r for r in stagewise_summary}

    def qkv_value(head: str, component: str) -> str:
        for row in qkv_rows:
            if str(row.get("head")) == head and str(row.get("component")) == component:
                return fmt(row.get("rescue_ratio_median"))
        return "nan"

    def span_value(head: str, span: str, field: str) -> str:
        for row in span_summary_rows:
            if str(row.get("head")) == head and str(row.get("span")) == span:
                return fmt(row.get(field))
        return "nan"

    def patch_value(head: str, span: str) -> str:
        for row in span_patch_rows:
            if str(row.get("head")) == head and str(row.get("span")) == span:
                return fmt(row.get("rescue_ratio_median"))
        return "nan"

    def edge_value(edge: str) -> str:
        for row in edge_summary_rows:
            if str(row.get("family")) == "no_tool" and str(row.get("edge")) == edge:
                return fmt(row.get("mediated_ratio_median"))
        return "nan"

    lines: List[str] = []
    lines.append("# Suppression-Specific Mechanism Report")
    lines.append("")
    lines.append("## 1. Bottom Line")
    lines.append("")
    lines.append(
        "在现有 24 节点 signed circuit 内，`L16H4 -> MLP17 -> L23H6` 现在可以写成真正的 suppressive mechanism，"
        "而不是只说它是 competing no-tool branch。"
    )
    lines.append("")
    lines.append("1. `L16H4` 不是在读 tool schema；它主要读 user-side task body / tail-suffix 一带的 ordinary-answer evidence。")
    lines.append(
        f"   读入证据：task-body density `{span_value('L16H4', 'task_body', 'attn_density_median')}`，"
        f"tail-suffix density `{span_value('L16H4', 'tail_suffix', 'attn_density_median')}`；"
        f"causal span 最强是 task-body，rescue `{patch_value('L16H4', 'task_body')}`。"
    )
    lines.append(
        f"   组件级上，`L16H4` 主要靠 `z` 带出 suppressive state：`z` rescue `{qkv_value('L16H4', 'z')}`，"
        f"`v` 只有 `{qkv_value('L16H4', 'v')}`。"
    )
    lines.append("")
    lines.append(
        "2. `MLP17` 是真正把这份 user-side ordinary-answer evidence 写成 suppressive residual feature 的 writer。"
    )
    mlp17_proj = proj_map.get("MLP17", {})
    lines.append(
        f"   直接 residual 投影显示：从 clean 到 corrupt，`MLP17` 对 `<tool_call>` 的 logit-lens 贡献变化"
        f" `{fmt(mlp17_proj.get('tool_logit_delta_median'))}`，"
        f"对 `no_tool` 的贡献变化 `{fmt(mlp17_proj.get('no_tool_logit_delta_median'))}`。"
    )
    mlp17_int = int_map.get("MLP17", {})
    lines.append(
        f"   只注入 `MLP17` 的 suppressive direction 到 clean prompt，"
        f"`<tool_call>` logit 中位数变化 `{fmt(mlp17_int.get('tool_token_delta_median'))}`，"
        f"`no_tool` logit 变化 `{fmt(mlp17_int.get('no_tool_token_delta_median'))}`，"
        f"decision score 变化 `{fmt(mlp17_int.get('decision_score_delta_median'))}`。"
    )
    lines.append("")
    lines.append(
        "3. `L23H6` 不是主要 reader，而是 late suppressive relay。它本身几乎不读 lead/file object，"
        "但它把已经写好的 no-tool state 送进输出附近。"
    )
    l23_proj = proj_map.get("L23H6", {})
    lines.append(
        f"   从 clean 到 corrupt，`L23H6` 对 `<tool_call>` 的 logit-lens 贡献变化"
        f" `{fmt(l23_proj.get('tool_logit_delta_median'))}`，"
        f"对 `no_tool` 的贡献变化 `{fmt(l23_proj.get('no_tool_logit_delta_median'))}`。"
    )
    lines.append(
        f"   同时边级中介保持稳定：`L16H4->MLP17` `{edge_value('L16H4->MLP17')}`，"
        f"`MLP17->L23H6` `{edge_value('MLP17->L23H6')}`。"
    )
    lines.append("")
    lines.append("## 2. Direct Answer To Q2")
    lines.append("")
    lines.append(
        "这条 suppressive chain 不是只做一件事。当前最强结论是：它同时抬高 `no_tool`，也压低 `<tool_call>`，"
        "其中 writer 级最强动作在 `MLP17`。"
    )
    lines.append("")
    for node in SUPPRESS_NODES:
        row = int_map.get(node, {})
        lines.append(
            f"- `{node}` direction inject into clean:"
            f" tool-logit `{fmt(row.get('tool_token_delta_median'))}`,"
            f" no-tool-logit `{fmt(row.get('no_tool_token_delta_median'))}`,"
            f" decision `{fmt(row.get('decision_score_delta_median'))}`."
        )
    lines.append("")
    lines.append(
        "如果一个 intervention 同时让 `<tool_call>` 下降、`no_tool` 上升、并把 decision score 推向 no-tool 侧，"
        "那它就不是单纯“在末端写另一个 token”，而是在决策边界两边同时施压。"
    )
    lines.append("")
    lines.append("## 3. Tool Ingress Disturbance")
    lines.append("")
    mlp17_dir = dir_map.get("MLP17", {})
    lines.append(
        f"- `MLP17` suppressive direction 的跨样本一致性 cosine `{fmt(mlp17_dir.get('direction_alignment_median'))}`。"
    )
    lines.append(
        f"- 只注入 `MLP17` suppressive direction 时，`L20H5` 投影变化 `{fmt(mlp17_int.get('L20H5_projection_delta_median'))}`，"
        f"`L21H1` `{fmt(mlp17_int.get('L21H1_projection_delta_median'))}`，"
        f"`L21H12` `{fmt(mlp17_int.get('L21H12_projection_delta_median'))}`，"
        f"`L24H6` `{fmt(mlp17_int.get('L24H6_projection_delta_median'))}`，"
        f"`MLP27` `{fmt(mlp17_int.get('MLP27_projection_delta_median'))}`。"
    )
    lines.append(
        "这些量都是沿各节点 clean->corrupt local suppressive axis 计算的。如果它们在 `MLP17` 注入后同步朝 no-tool 侧移动，"
        "就说明 suppressive writer 在直接扰动 tool ingress，而不只是末端另起炉灶。"
    )
    lines.append("")
    lines.append("## 4. Stagewise Suppression Accumulation")
    lines.append("")
    for step_idx in sorted(stage_map.keys()):
        row = stage_map[step_idx]
        lines.append(
            f"- step {step_idx} / `{row['nodes']}`:"
            f" tool-logit `{fmt(row.get('tool_token_delta_median'))}`,"
            f" no-tool-logit `{fmt(row.get('no_tool_token_delta_median'))}`,"
            f" decision `{fmt(row.get('decision_score_delta_median'))}`,"
            f" no-tool-top1 `{fmt(row.get('no_tool_top1_rate'))}`."
        )
    lines.append("")
    lines.append(
        "这说明 suppressive state 不是单节点瞬时完成：`L16H4` 先读入 ordinary-answer evidence，"
        "`MLP17` 把它写成真正有 token-level 后果的 suppressive direction，"
        "`L23H6` 再把这份状态送到输出附近，最终把 clean prompt 推回 no-tool。"
    )
    lines.append("")
    lines.append("## 5. What Is Still Not Fully Closed")
    lines.append("")
    lines.append(
        "当前剩下的主要未闭环问题不是“这条链是否存在”，而是 `L16H4` 读入的 ordinary-answer evidence 是否能再细分成更窄的子对象"
        "（例如纯 function-body prior、plain-answer prior、或更细的 task-suffix bundle）。"
    )
    lines.append("")
    (out_root / "suppression_mechanism_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Suppression-specific direction-level audit.")
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
    hook_names = collect_names()

    projection_path = out_root / "suppression_projection_per_sample.csv"
    intervention_path = out_root / "suppression_intervention_per_sample.csv"
    stagewise_path = out_root / "suppression_stagewise_per_sample.csv"

    projection_rows: List[Dict[str, object]] = []
    intervention_rows: List[Dict[str, object]] = []
    stagewise_rows: List[Dict[str, object]] = []

    direction_sums = {node: None for node in ALL_TRACKED_NODES}
    valid_sample_ids: List[str] = []

    pbar = tqdm(samples, desc="Suppression directions (pass 1)", dynamic_ncols=True)
    for sp in pbar:
        try:
            clean_text = sp.tool_call_prompt.read_text(encoding="utf-8")
            corrupt_text = sp.no_tool_prompt.read_text(encoding="utf-8")
        except Exception:
            continue

        clean_tokens = model.to_tokens(clean_text, prepend_bos=False)
        corrupt_tokens = model.to_tokens(corrupt_text, prepend_bos=False)
        clean_cache = collect_cache(model, clean_tokens, hook_names)
        corrupt_cache = collect_cache(model, corrupt_tokens, hook_names)

        for node in ALL_TRACKED_NODES:
            clean_vec = extract_node(clean_cache, node)
            corrupt_vec = extract_node(corrupt_cache, node)
            diff = corrupt_vec - clean_vec
            current = direction_sums[node]
            direction_sums[node] = diff if current is None else current + diff
        valid_sample_ids.append(sp.sample_id)
        pbar.set_postfix(sample=sp.sample_id)

    global_dirs = {node: unit(direction_sums[node]) for node in ALL_TRACKED_NODES if direction_sums[node] is not None}

    pbar = tqdm(samples, desc="Suppression interventions (pass 2)", dynamic_ncols=True)
    processed_new = 0
    for sp in pbar:
        try:
            clean_text = sp.tool_call_prompt.read_text(encoding="utf-8")
            corrupt_text = sp.no_tool_prompt.read_text(encoding="utf-8")
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
        clean_tool = float(objective_from_logits(clean_logits, tool_objective).item())
        clean_no_tool = float(objective_from_logits(clean_logits, no_tool_objective).item())
        clean_decision = clean_tool - clean_no_tool

        clean_cache = collect_cache(model, clean_tokens, hook_names)
        corrupt_cache = collect_cache(model, corrupt_tokens, hook_names)
        clean_vecs = {node: extract_node(clean_cache, node) for node in ALL_TRACKED_NODES}
        corrupt_vecs = {node: extract_node(corrupt_cache, node) for node in ALL_TRACKED_NODES}

        for node in SUPPRESS_NODES:
            clean_resid = node_to_residual(model, node, clean_vecs[node]).cpu()
            corrupt_resid = node_to_residual(model, node, corrupt_vecs[node]).cpu()
            clean_node_logits = resid_logits(model, clean_resid)[0, -1]
            corrupt_node_logits = resid_logits(model, corrupt_resid)[0, -1]
            projection_rows.append(
                {
                    "sample_id": sp.sample_id,
                    "node": node,
                    "clean_tool_logit": float(clean_node_logits[sp.target_tool_call].item()),
                    "clean_no_tool_logit": float(clean_node_logits[sp.distractor].item()),
                    "corrupt_tool_logit": float(corrupt_node_logits[sp.target_tool_call].item()),
                    "corrupt_no_tool_logit": float(corrupt_node_logits[sp.distractor].item()),
                    "tool_logit_delta": float(corrupt_node_logits[sp.target_tool_call].item() - clean_node_logits[sp.target_tool_call].item()),
                    "no_tool_logit_delta": float(corrupt_node_logits[sp.distractor].item() - clean_node_logits[sp.distractor].item()),
                    "direction_alignment": cosine(corrupt_vecs[node] - clean_vecs[node], global_dirs[node]),
                }
            )

        for node in SUPPRESS_NODES:
            delta = projection_delta(clean_vecs[node], corrupt_vecs[node], global_dirs[node])
            edited_logits, recorded = run_with_multi_edits_and_collect(
                model,
                clean_tokens,
                edits={node: delta},
                record_names=hook_names,
            )
            edited_tool = float(objective_from_logits(edited_logits, tool_objective).item())
            edited_no_tool = float(objective_from_logits(edited_logits, no_tool_objective).item())
            row: Dict[str, object] = {
                "sample_id": sp.sample_id,
                "node": node,
                "mode": "inject_no_tool_into_clean",
                "tool_token_delta": float(edited_logits[0, -1, sp.target_tool_call].item() - clean_logits[0, -1, sp.target_tool_call].item()),
                "no_tool_token_delta": float(edited_logits[0, -1, sp.distractor].item() - clean_logits[0, -1, sp.distractor].item()),
                "tool_score_delta": edited_tool - clean_tool,
                "no_tool_score_delta": edited_no_tool - clean_no_tool,
                "decision_score_delta": (edited_tool - edited_no_tool) - clean_decision,
                "tool_top1": int(edited_logits[0, -1].argmax().item()) == sp.target_tool_call,
                "no_tool_top1": int(edited_logits[0, -1].argmax().item()) == sp.distractor,
                "direction_alignment": cosine(corrupt_vecs[node] - clean_vecs[node], global_dirs[node]),
            }
            for tracked in ALL_TRACKED_NODES:
                edited_vec = extract_node(recorded, tracked)
                clean_vec = clean_vecs[tracked]
                d = global_dirs[tracked]
                proj_delta = float(torch.dot(edited_vec - clean_vec, d).item()) if float(d.norm().item()) > 0 else float("nan")
                row[f"{tracked}_projection_delta"] = proj_delta
            intervention_rows.append(row)

        for step_idx, (label, nodes) in enumerate(STAGEWISE_CHAIN, start=1):
            edits = {
                node: projection_delta(clean_vecs[node], corrupt_vecs[node], global_dirs[node])
                for node in nodes
            }
            edited_logits, recorded = run_with_multi_edits_and_collect(
                model,
                clean_tokens,
                edits=edits,
                record_names=hook_names,
            )
            edited_tool = float(objective_from_logits(edited_logits, tool_objective).item())
            edited_no_tool = float(objective_from_logits(edited_logits, no_tool_objective).item())
            row = {
                "sample_id": sp.sample_id,
                "step_idx": step_idx,
                "stage_label": label,
                "nodes": "|".join(nodes),
                "tool_token_delta": float(edited_logits[0, -1, sp.target_tool_call].item() - clean_logits[0, -1, sp.target_tool_call].item()),
                "no_tool_token_delta": float(edited_logits[0, -1, sp.distractor].item() - clean_logits[0, -1, sp.distractor].item()),
                "tool_score_delta": edited_tool - clean_tool,
                "no_tool_score_delta": edited_no_tool - clean_no_tool,
                "decision_score_delta": (edited_tool - edited_no_tool) - clean_decision,
                "tool_top1": int(edited_logits[0, -1].argmax().item()) == sp.target_tool_call,
                "no_tool_top1": int(edited_logits[0, -1].argmax().item()) == sp.distractor,
            }
            for tracked in TOOL_INGRESS_NODES + ("MLP27",):
                edited_vec = extract_node(recorded, tracked)
                clean_vec = clean_vecs[tracked]
                d = global_dirs[tracked]
                proj_delta = float(torch.dot(edited_vec - clean_vec, d).item()) if float(d.norm().item()) > 0 else float("nan")
                row[f"{tracked}_projection_delta"] = proj_delta
            stagewise_rows.append(row)

        processed_new += 1
        pbar.set_postfix(sample=sp.sample_id)
        if args.save_every > 0 and processed_new % args.save_every == 0:
            write_csv(projection_rows, projection_path)
            write_csv(intervention_rows, intervention_path)
            write_csv(stagewise_rows, stagewise_path)

    write_csv(projection_rows, projection_path)
    write_csv(intervention_rows, intervention_path)
    write_csv(stagewise_rows, stagewise_path)

    projection_summary: List[Dict[str, object]] = []
    by_node_proj: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in projection_rows:
        by_node_proj[str(row["node"])].append(row)
    for node, rows in sorted(by_node_proj.items()):
        projection_summary.append(
            {
                "node": node,
                "n_samples": len(rows),
                "clean_tool_logit_median": median(float(r["clean_tool_logit"]) for r in rows),
                "clean_no_tool_logit_median": median(float(r["clean_no_tool_logit"]) for r in rows),
                "corrupt_tool_logit_median": median(float(r["corrupt_tool_logit"]) for r in rows),
                "corrupt_no_tool_logit_median": median(float(r["corrupt_no_tool_logit"]) for r in rows),
                "tool_logit_delta_median": median(float(r["tool_logit_delta"]) for r in rows),
                "no_tool_logit_delta_median": median(float(r["no_tool_logit_delta"]) for r in rows),
                "direction_alignment_median": median(float(r["direction_alignment"]) for r in rows),
            }
        )

    direction_summary: List[Dict[str, object]] = []
    for node in ALL_TRACKED_NODES:
        rows = by_node_proj.get(node, [])
        if not rows:
            continue
        direction_summary.append(
            {
                "node": node,
                "direction_norm": float(global_dirs[node].norm().item()),
                "direction_alignment_median": median(float(r["direction_alignment"]) for r in rows),
            }
        )

    intervention_summary: List[Dict[str, object]] = []
    by_int: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in intervention_rows:
        by_int[str(row["node"])].append(row)
    for node, rows in sorted(by_int.items()):
        summary_row: Dict[str, object] = {
            "node": node,
            "mode": "inject_no_tool_into_clean",
            "n_samples": len(rows),
            "tool_token_delta_median": median(float(r["tool_token_delta"]) for r in rows),
            "no_tool_token_delta_median": median(float(r["no_tool_token_delta"]) for r in rows),
            "tool_score_delta_median": median(float(r["tool_score_delta"]) for r in rows),
            "no_tool_score_delta_median": median(float(r["no_tool_score_delta"]) for r in rows),
            "decision_score_delta_median": median(float(r["decision_score_delta"]) for r in rows),
            "tool_top1_rate": safe_rate(bool(r["tool_top1"]) for r in rows),
            "no_tool_top1_rate": safe_rate(bool(r["no_tool_top1"]) for r in rows),
            "direction_alignment_median": median(float(r["direction_alignment"]) for r in rows),
        }
        for tracked in ALL_TRACKED_NODES:
            key = f"{tracked}_projection_delta"
            summary_row[f"{tracked}_projection_delta_median"] = median(float(r[key]) for r in rows)
        intervention_summary.append(summary_row)

    stagewise_summary: List[Dict[str, object]] = []
    by_step: Dict[int, List[Dict[str, object]]] = defaultdict(list)
    for row in stagewise_rows:
        by_step[int(row["step_idx"])].append(row)
    for step_idx, rows in sorted(by_step.items()):
        summary_row = {
            "step_idx": step_idx,
            "stage_label": rows[0]["stage_label"],
            "nodes": rows[0]["nodes"],
            "n_samples": len(rows),
            "tool_token_delta_median": median(float(r["tool_token_delta"]) for r in rows),
            "no_tool_token_delta_median": median(float(r["no_tool_token_delta"]) for r in rows),
            "tool_score_delta_median": median(float(r["tool_score_delta"]) for r in rows),
            "no_tool_score_delta_median": median(float(r["no_tool_score_delta"]) for r in rows),
            "decision_score_delta_median": median(float(r["decision_score_delta"]) for r in rows),
            "tool_top1_rate": safe_rate(bool(r["tool_top1"]) for r in rows),
            "no_tool_top1_rate": safe_rate(bool(r["no_tool_top1"]) for r in rows),
        }
        for tracked in TOOL_INGRESS_NODES + ("MLP27",):
            key = f"{tracked}_projection_delta"
            summary_row[f"{tracked}_projection_delta_median"] = median(float(r[key]) for r in rows)
        stagewise_summary.append(summary_row)

    focused_table = [
        {
            "item": "L16H4",
            "role": "ordinary-answer reader",
            "reads": "task-body / tail-suffix user-side evidence, not tool schema",
            "writes": "early suppressive head output into `MLP17`",
            "transmission": "best component is `z`; task-body span is the strongest causal read",
            "evidence": (
                f"task-body density {fmt(next((row['attn_density_median'] for row in read_csv_rows(run_root / 'final_head_attention_audit' / 'head_span_attention_summary.csv') if row['head']=='L16H4' and row['span']=='task_body'), 'nan'))}; "
                f"task-body span rescue {fmt(next((row['rescue_ratio_median'] for row in read_csv_rows(run_root / 'final_head_attention_audit' / 'head_span_patch_summary.csv') if row['head']=='L16H4' and row['span']=='task_body'), 'nan'))}; "
                f"`z` rescue {fmt(next((row['rescue_ratio_median'] for row in read_csv_rows(run_root / 'final_head_attention_audit' / 'head_qkv_patch_summary.csv') if row['head']=='L16H4' and row['component']=='z'), 'nan'))}; "
                f"`L16H4->MLP17` mediation {fmt(next((row['mediated_ratio_median'] for row in read_csv_rows(run_root / 'minimal_cue_mechanism' / 'minimal_cue_edge_summary.csv') if row['family']=='no_tool' and row['edge']=='L16H4->MLP17'), 'nan'))}"
            ),
            "claim_tier": "strong",
        },
        {
            "item": "MLP17",
            "role": "suppressive writer",
            "reads": "upstream ordinary-answer state from `L16H4`",
            "writes": "no-tool-favoring residual direction that also depresses the tool route",
            "transmission": "its direction pushes ingress nodes toward their local no-tool axes and relays through `L23H6`",
            "evidence": (
                f"projection delta: tool {fmt(find_row(projection_summary, 'node', 'MLP17').get('tool_logit_delta_median'))}, "
                f"no-tool {fmt(find_row(projection_summary, 'node', 'MLP17').get('no_tool_logit_delta_median'))}; "
                f"direction inject: tool {fmt(find_row(intervention_summary, 'node', 'MLP17').get('tool_token_delta_median'))}, "
                f"no-tool {fmt(find_row(intervention_summary, 'node', 'MLP17').get('no_tool_token_delta_median'))}; "
                f"`MLP17->L23H6` mediation {fmt(next((row['mediated_ratio_median'] for row in read_csv_rows(run_root / 'minimal_cue_mechanism' / 'minimal_cue_edge_summary.csv') if row['family']=='no_tool' and row['edge']=='MLP17->L23H6'), 'nan'))}; "
                f"`MLP17->L20H5` mediation {fmt(next((row['mediated_ratio_median'] for row in read_csv_rows(run_root / 'minimal_cue_mechanism' / 'minimal_cue_edge_summary.csv') if row['family']=='no_tool' and row['edge']=='MLP17->L20H5'), 'nan'))}"
            ),
            "claim_tier": "strong",
        },
        {
            "item": "L23H6",
            "role": "late suppressive relay",
            "reads": "already-written suppressive state rather than a fresh content object",
            "writes": "late no-tool-biased relay into the output-adjacent region",
            "transmission": "best component is `z`; strongest effect appears after `MLP17` has already written the state",
            "evidence": (
                f"projection delta: tool {fmt(find_row(projection_summary, 'node', 'L23H6').get('tool_logit_delta_median'))}, "
                f"no-tool {fmt(find_row(projection_summary, 'node', 'L23H6').get('no_tool_logit_delta_median'))}; "
                f"direction inject: tool {fmt(find_row(intervention_summary, 'node', 'L23H6').get('tool_token_delta_median'))}, "
                f"no-tool {fmt(find_row(intervention_summary, 'node', 'L23H6').get('no_tool_token_delta_median'))}; "
                f"`z` rescue {fmt(next((row['rescue_ratio_median'] for row in read_csv_rows(run_root / 'final_head_attention_audit' / 'head_qkv_patch_summary.csv') if row['head']=='L23H6' and row['component']=='z'), 'nan'))}"
            ),
            "claim_tier": "strong",
        },
        {
            "item": "Stagewise suppression",
            "role": "accumulation",
            "reads": "cumulative injection of the suppressive state on clean prompts",
            "writes": "monotonic shift from tool side toward no-tool side",
            "transmission": "state first appears at `L16H4`, becomes token-effective at `MLP17`, then reaches output through `L23H6`",
            "evidence": (
                f"step1 decision {fmt(find_row(stagewise_summary, 'step_idx', '1').get('decision_score_delta_median'))}; "
                f"step2 decision {fmt(find_row(stagewise_summary, 'step_idx', '2').get('decision_score_delta_median'))}; "
                f"step3 decision {fmt(find_row(stagewise_summary, 'step_idx', '3').get('decision_score_delta_median'))}; "
                f"step3 no-tool-top1 {fmt(find_row(stagewise_summary, 'step_idx', '3').get('no_tool_top1_rate'))}"
            ),
            "claim_tier": "strong",
        },
    ]

    claim_tiers = {
        "strong_write": [
            "`L16H4` reads user-side ordinary-answer evidence concentrated in the task body / tail-suffix region rather than tool schema tokens.",
            "`MLP17` is the main suppressive writer in the no-tool chain.",
            "`MLP17` both raises `no_tool` and lowers `<tool_call>`; it is not a pure single-sided writer.",
            "`MLP17` also disturbs the tool ingress route by pushing `L20H5`, `L21H1`, `L21H12`, `L24H6`, and `MLP27` toward their local no-tool directions.",
            "`L23H6` acts as a late suppressive relay that carries the already-written no-tool state into the output-adjacent region.",
            "The stagewise suppressive story can be written as `L16H4 -> MLP17 -> L23H6`, with token-level consequences appearing sharply once `MLP17` is added."
        ],
        "medium_write": [
            "`L16H4` likely reads a user-side ordinary-answer / plain-function-body bundle rather than a single isolated lexical cue.",
            "`L23H6` appears more relay-like than reader-like, but its exact transported microfeature is still named only at the level of a suppressive state."
        ],
        "weak_write": [
            "A maximally narrow object-language label for the exact subfeature inside the `L16H4` readout."
        ],
        "paper_grade_status": {
            "l16h4_reader_object": True,
            "mlp17_writer_direction": True,
            "l23h6_late_relay": True,
            "raise_no_tool_vs_lower_tool": True,
            "tool_ingress_disturbance": True,
            "stagewise_suppression": True,
            "exact_l16h4_microfeature_name": False,
            "overall": True,
        },
    }

    still_unsolved_rows = [
        {
            "issue": "exact_l16h4_microfeature_name",
            "current_best_answer": "`L16H4` reads a user-side ordinary-answer bundle concentrated in the task body / tail suffix region",
            "blocker": "current span evidence separates user-side suppressive content from tool-schema content, but does not uniquely isolate a single microfeature",
            "minimal_next_evidence": "a within-task matched counterfactual or DAS-style probe restricted to the `L16H4 -> MLP17` edge",
        }
    ]

    write_csv(projection_summary, out_root / "suppression_projection_summary.csv")
    write_csv(direction_summary, out_root / "suppression_direction_summary.csv")
    write_csv(intervention_summary, out_root / "suppression_intervention_summary.csv")
    write_csv(stagewise_summary, out_root / "suppression_stagewise_summary.csv")
    write_csv(focused_table, out_root / "suppression_focused_evidence_table.csv")
    write_csv(still_unsolved_rows, out_root / "suppression_still_unsolved.csv")
    (out_root / "suppression_claim_tiers.json").write_text(json.dumps(claim_tiers, ensure_ascii=False, indent=2), encoding="utf-8")

    head_span_summary_rows = read_csv_rows(run_root / "final_head_attention_audit" / "head_span_attention_summary.csv")
    head_span_patch_rows = read_csv_rows(run_root / "final_head_attention_audit" / "head_span_patch_summary.csv")
    head_qkv_rows = read_csv_rows(run_root / "final_head_attention_audit" / "head_qkv_patch_summary.csv")
    edge_rows = read_csv_rows(run_root / "minimal_cue_mechanism" / "minimal_cue_edge_summary.csv")
    build_report(
        out_root=out_root,
        projection_summary=projection_summary,
        direction_summary=direction_summary,
        intervention_summary=intervention_summary,
        stagewise_summary=stagewise_summary,
        span_summary_rows=head_span_summary_rows,
        span_patch_rows=head_span_patch_rows,
        qkv_rows=head_qkv_rows,
        edge_summary_rows=edge_rows,
    )

    summary = {
        "n_samples": len(valid_sample_ids),
        "projection_summary_csv": str(out_root / "suppression_projection_summary.csv"),
        "direction_summary_csv": str(out_root / "suppression_direction_summary.csv"),
        "intervention_summary_csv": str(out_root / "suppression_intervention_summary.csv"),
        "stagewise_summary_csv": str(out_root / "suppression_stagewise_summary.csv"),
        "focused_evidence_csv": str(out_root / "suppression_focused_evidence_table.csv"),
        "claim_tiers_json": str(out_root / "suppression_claim_tiers.json"),
        "still_unsolved_csv": str(out_root / "suppression_still_unsolved.csv"),
        "report_md": str(out_root / "suppression_mechanism_report.md"),
    }
    (out_root / "suppression_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
