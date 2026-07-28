#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from tqdm.auto import tqdm

from toolcall_circuit.bidirectional_causal_eval import collect_cache_cpu_for_nodes
from toolcall_circuit.dataset import (
    ToolCallSample,
    get_tool_call_target_spec,
    resolve_distractor_token,
)
from toolcall_circuit.objective import (
    DistributionObjective,
    build_distribution_objective,
    logits_to_probs,
    objective_from_logits,
)
from toolcall_circuit.single_sample import parse_head

FORWARD_OUTPUT_NODE = "Residual Output: <tool_call>"
REVERSE_OUTPUT_NODE = "Residual Output: no_tool"

METRIC_ORDER = ("endpoint_kl_score", "logit_margin", "target_prob")


@dataclass(frozen=True)
class DirectionSpec:
    name: str
    clean_role: str
    corrupt_role: str
    objective_endpoint: str
    output_node_label: str
    decision_label: str


@dataclass
class DirectionCase:
    sample: ToolCallSample
    spec: DirectionSpec
    clean_text: str
    corrupt_text: str
    clean_tokens: torch.Tensor
    corrupt_tokens: torch.Tensor
    clean_logits: torch.Tensor
    corrupt_logits: torch.Tensor
    endpoint_objective: DistributionObjective
    target_token_id: int
    distractor_token_id: int
    clean_metric_raws: Dict[str, float]
    corrupt_metric_raws: Dict[str, float]


def get_direction_spec(direction: str) -> DirectionSpec:
    direction = str(direction).strip().lower()
    if direction == "forward":
        return DirectionSpec(
            name="forward",
            clean_role="tool_call",
            corrupt_role="no_tool",
            objective_endpoint="tool_call",
            output_node_label=FORWARD_OUTPUT_NODE,
            decision_label="tool_call_vs_no_tool",
        )
    if direction == "reverse":
        return DirectionSpec(
            name="reverse",
            clean_role="no_tool",
            corrupt_role="tool_call",
            objective_endpoint="no_tool",
            output_node_label=REVERSE_OUTPUT_NODE,
            decision_label="no_tool_vs_tool_call",
        )
    raise ValueError(f"Unknown direction: {direction}")


def finite(values: Iterable[float]) -> List[float]:
    out: List[float] = []
    for value in values:
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            out.append(float(value))
    return out


def median(values: Iterable[float]) -> float:
    vals = finite(values)
    return float(np.median(vals)) if vals else float("nan")


def mean(values: Iterable[float]) -> float:
    vals = finite(values)
    return float(np.mean(vals)) if vals else float("nan")


def safe_rate(values: Iterable[bool]) -> float:
    vals = [1.0 if bool(v) else 0.0 for v in values]
    return float(np.mean(vals)) if vals else float("nan")


def write_csv(rows: Sequence[Dict[str, object]], out_path: Path) -> None:
    if not rows:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(obj: object, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def run_logits_with_patched_nodes(
    model,
    base_tokens: torch.Tensor,
    source_cache_cpu: Dict[str, torch.Tensor],
    patch_nodes: Sequence[str],
) -> torch.Tensor:
    heads_by_layer: Dict[int, List[int]] = {}
    mlp_layers: List[int] = []
    for node in patch_nodes:
        if node.startswith("MLP"):
            mlp_layers.append(int(node[3:]))
        else:
            layer, head = parse_head(node)
            heads_by_layer.setdefault(layer, []).append(head)

    hooks: List[Tuple[str, object]] = []
    for layer, heads in heads_by_layer.items():
        cache_name = f"blocks.{layer}.attn.hook_z"
        source_act = source_cache_cpu[cache_name].to(base_tokens.device)
        head_ids = sorted(set(int(h) for h in heads))

        def make_head_hook(src: torch.Tensor, target_heads: Sequence[int]):
            def hook_fn(z: torch.Tensor, hook):  # noqa: ANN001
                out = z.clone()
                for head in target_heads:
                    out[:, -1, head, :] = src[:, -1, head, :]
                return out

            return hook_fn

        hooks.append((cache_name, make_head_hook(source_act, head_ids)))

    for layer in sorted(set(int(layer) for layer in mlp_layers)):
        cache_name = f"blocks.{layer}.hook_mlp_out"
        source_act = source_cache_cpu[cache_name].to(base_tokens.device)

        def make_mlp_hook(src: torch.Tensor):
            def hook_fn(mlp_out: torch.Tensor, hook):  # noqa: ANN001
                out = mlp_out.clone()
                out[:, -1, :] = src[:, -1, :]
                return out

            return hook_fn

        hooks.append((cache_name, make_mlp_hook(source_act)))

    with torch.no_grad():
        return model.run_with_hooks(base_tokens, fwd_hooks=hooks)


def metric_raws_from_logits(
    logits: torch.Tensor,
    *,
    objective: DistributionObjective,
    target_token_id: int,
    distractor_token_id: int,
) -> Dict[str, float]:
    probs = logits_to_probs(
        logits,
        temperature=objective.temperature,
        masked_token_ids=objective.masked_token_ids,
    )[0]
    last_logits = logits[0, -1, :]
    return {
        "endpoint_kl_score": float(objective_from_logits(logits, objective).item()),
        "logit_margin": float((last_logits[int(target_token_id)] - last_logits[int(distractor_token_id)]).item()),
        "target_prob": float(probs[int(target_token_id)].item()),
    }


def build_direction_case(
    model,
    tokenizer,
    sample: ToolCallSample,
    direction: str,
) -> DirectionCase:
    spec = get_direction_spec(direction)
    tool_text = sample.clean_path.read_text(encoding="utf-8")
    no_tool_text = sample.corrupt_path.read_text(encoding="utf-8")
    if spec.name == "forward":
        clean_text = tool_text
        corrupt_text = no_tool_text
    else:
        clean_text = no_tool_text
        corrupt_text = tool_text

    clean_tokens = model.to_tokens(clean_text, prepend_bos=False)
    corrupt_tokens = model.to_tokens(corrupt_text, prepend_bos=False)
    if clean_tokens.shape != corrupt_tokens.shape:
        raise ValueError(
            f"{spec.name} clean/corrupt token shapes differ for {sample.sample_id}: "
            f"{tuple(clean_tokens.shape)} vs {tuple(corrupt_tokens.shape)}"
        )

    with torch.no_grad():
        clean_logits = model(clean_tokens)
        corrupt_logits = model(corrupt_tokens)

    tool_spec = get_tool_call_target_spec(tokenizer, target_text="<tool_call>")
    if not tool_spec.is_single_token:
        raise NotImplementedError(
            f"<tool_call> tokenization is no longer single-token ({tool_spec.token_ids})."
        )
    tool_token_id = int(tool_spec.primary_token_id)
    if spec.name == "forward":
        target_token_id = tool_token_id
        distractor_token_id = resolve_distractor_token(corrupt_logits[0, -1, :], tool_token_id)
    else:
        target_token_id = resolve_distractor_token(clean_logits[0, -1, :], tool_token_id)
        distractor_token_id = tool_token_id

    endpoint_objective = build_distribution_objective(
        clean_logits,
        endpoint_label=spec.objective_endpoint,
        tokenizer=tokenizer,
        temperature=1.0,
        masked_token_ids=(),
    )
    clean_metric_raws = metric_raws_from_logits(
        clean_logits,
        objective=endpoint_objective,
        target_token_id=target_token_id,
        distractor_token_id=distractor_token_id,
    )
    corrupt_metric_raws = metric_raws_from_logits(
        corrupt_logits,
        objective=endpoint_objective,
        target_token_id=target_token_id,
        distractor_token_id=distractor_token_id,
    )
    return DirectionCase(
        sample=sample,
        spec=spec,
        clean_text=clean_text,
        corrupt_text=corrupt_text,
        clean_tokens=clean_tokens,
        corrupt_tokens=corrupt_tokens,
        clean_logits=clean_logits,
        corrupt_logits=corrupt_logits,
        endpoint_objective=endpoint_objective,
        target_token_id=target_token_id,
        distractor_token_id=distractor_token_id,
        clean_metric_raws=clean_metric_raws,
        corrupt_metric_raws=corrupt_metric_raws,
    )


def _ratio(numerator: float, denominator: float) -> float:
    if not math.isfinite(numerator) or not math.isfinite(denominator) or abs(denominator) <= 1e-8:
        return float("nan")
    return float(numerator / denominator)


def build_metric_transition(
    *,
    clean_raw: float,
    corrupt_raw: float,
    patched_suff_raw: float,
    patched_nec_raw: float,
) -> Dict[str, float]:
    gap = clean_raw - corrupt_raw
    suff_recovery_raw = patched_suff_raw - corrupt_raw
    nec_drop_raw = clean_raw - patched_nec_raw
    return {
        "clean_raw": clean_raw,
        "corrupt_raw": corrupt_raw,
        "gap_raw": gap,
        "suff_patched_raw": patched_suff_raw,
        "suff_recovery_raw": suff_recovery_raw,
        "suff_recovery_ratio": _ratio(suff_recovery_raw, gap),
        "nec_patched_raw": patched_nec_raw,
        "nec_drop_raw": nec_drop_raw,
        "nec_drop_ratio": _ratio(nec_drop_raw, gap),
    }


def evaluate_node_set_for_sample(
    model,
    tokenizer,
    sample: ToolCallSample,
    direction: str,
    nodes: Sequence[str],
) -> Dict[str, object]:
    case = build_direction_case(model, tokenizer, sample, direction)
    clean_cache = collect_cache_cpu_for_nodes(model, case.clean_tokens, nodes)
    corrupt_cache = collect_cache_cpu_for_nodes(model, case.corrupt_tokens, nodes)
    suff_logits = run_logits_with_patched_nodes(
        model,
        case.corrupt_tokens,
        clean_cache,
        nodes,
    )
    nec_logits = run_logits_with_patched_nodes(
        model,
        case.clean_tokens,
        corrupt_cache,
        nodes,
    )
    suff_raws = metric_raws_from_logits(
        suff_logits,
        objective=case.endpoint_objective,
        target_token_id=case.target_token_id,
        distractor_token_id=case.distractor_token_id,
    )
    nec_raws = metric_raws_from_logits(
        nec_logits,
        objective=case.endpoint_objective,
        target_token_id=case.target_token_id,
        distractor_token_id=case.distractor_token_id,
    )
    row: Dict[str, object] = {
        "sample_id": sample.sample_id,
        "sample_rank": sample.sample_rank,
        "filename": sample.filename,
        "direction": case.spec.name,
        "n_nodes": len(nodes),
        "target_token_id": case.target_token_id,
        "target_token_str": tokenizer.decode([case.target_token_id]),
        "distractor_token_id": case.distractor_token_id,
        "distractor_token_str": tokenizer.decode([case.distractor_token_id]),
        "suff_top1_id": int(suff_logits[0, -1].argmax().item()),
        "nec_top1_id": int(nec_logits[0, -1].argmax().item()),
    }
    row["suff_top1_str"] = tokenizer.decode([int(row["suff_top1_id"])])
    row["nec_top1_str"] = tokenizer.decode([int(row["nec_top1_id"])])
    row["suff_target_top1"] = int(row["suff_top1_id"]) == case.target_token_id
    row["nec_target_top1_retained"] = int(row["nec_top1_id"]) == case.target_token_id

    for metric_name in METRIC_ORDER:
        metric = build_metric_transition(
            clean_raw=float(case.clean_metric_raws[metric_name]),
            corrupt_raw=float(case.corrupt_metric_raws[metric_name]),
            patched_suff_raw=float(suff_raws[metric_name]),
            patched_nec_raw=float(nec_raws[metric_name]),
        )
        for key, value in metric.items():
            row[f"{metric_name}__{key}"] = value
    return row


def summarize_behavior_rows(
    rows: Sequence[Dict[str, object]],
    *,
    label: str,
    nodes: Sequence[str],
    split_label: str,
) -> Dict[str, object]:
    summary: Dict[str, object] = {
        "label": label,
        "split": split_label,
        "n_samples": len(rows),
        "n_nodes": len(nodes),
        "nodes": list(nodes),
        "target_top1_rate_suff": safe_rate(bool(r.get("suff_target_top1")) for r in rows),
        "target_top1_rate_nec_retained": safe_rate(bool(r.get("nec_target_top1_retained")) for r in rows),
    }
    for metric_name in METRIC_ORDER:
        for suffix in (
            "clean_raw",
            "corrupt_raw",
            "gap_raw",
            "suff_patched_raw",
            "suff_recovery_raw",
            "suff_recovery_ratio",
            "nec_patched_raw",
            "nec_drop_raw",
            "nec_drop_ratio",
        ):
            values = [float(r[f"{metric_name}__{suffix}"]) for r in rows if math.isfinite(float(r[f"{metric_name}__{suffix}"]))]
            summary[f"{metric_name}__{suffix}__median"] = float(np.median(values)) if values else float("nan")
            summary[f"{metric_name}__{suffix}__mean"] = float(np.mean(values)) if values else float("nan")
    return summary


def evaluate_node_set_on_samples(
    model,
    tokenizer,
    samples: Sequence[ToolCallSample],
    direction: str,
    nodes: Sequence[str],
    *,
    label: str,
    split_label: str,
    progress_desc: str,
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    pbar = tqdm(samples, desc=progress_desc, dynamic_ncols=True)
    for sample in pbar:
        row = evaluate_node_set_for_sample(model, tokenizer, sample, direction, nodes)
        rows.append(row)
        pbar.set_postfix(sample=sample.sample_id)
    summary = summarize_behavior_rows(rows, label=label, nodes=nodes, split_label=split_label)
    return rows, summary
