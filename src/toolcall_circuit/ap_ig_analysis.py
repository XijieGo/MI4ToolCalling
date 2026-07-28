#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from tqdm.auto import tqdm

if __package__ in {None, ""}:
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from toolcall_circuit.ap_ig_common import (
    METRIC_ORDER,
    build_direction_case,
    build_metric_transition,
    get_direction_spec,
    metric_raws_from_logits,
    run_logits_with_patched_nodes,
    summarize_behavior_rows,
    write_csv,
    write_json,
)
from toolcall_circuit.bidirectional_causal_eval import collect_cache_cpu_for_nodes
from toolcall_circuit.dataset import ToolCallSample, load_dataset_samples, load_summary_records, select_samples
from toolcall_circuit.graph_utils import INPUT_NODE, draw_circuit_with_output, node_layer
from toolcall_circuit.paths import MODEL_PATH_DEFAULT
from toolcall_circuit.single_sample import load_hooked_qwen3


@dataclass(frozen=True)
class DiscoveryRecord:
    sample_id: str
    sample_rank: int
    summary: Dict[str, object]
    candidate_nodes: List[str]
    head_scores: torch.Tensor
    mlp_scores: torch.Tensor


def parse_int_list(raw: str) -> List[int]:
    out: List[int] = []
    for chunk in str(raw or "").split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        out.append(int(chunk))
    return out


def parse_float_list(raw: str) -> List[float]:
    out: List[float] = []
    for chunk in str(raw or "").split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        out.append(float(chunk))
    return out


def load_discovery_records(root: Path) -> List[DiscoveryRecord]:
    records: List[DiscoveryRecord] = []
    for summary_record in load_summary_records(root):
        summary = summary_record.summary
        node_score_path = Path(str(summary["artifacts"]["node_scores_pt"]))
        payload = torch.load(node_score_path, map_location="cpu")
        records.append(
            DiscoveryRecord(
                sample_id=summary_record.sample_id,
                sample_rank=int(summary_record.sample_rank or 0),
                summary=summary,
                candidate_nodes=[str(node) for node in summary.get("candidate_nodes", [])],
                head_scores=payload["head_scores"].float().cpu(),
                mlp_scores=payload["mlp_scores"].float().cpu(),
            )
        )
    records.sort(key=lambda record: (record.sample_rank if record.sample_rank > 0 else 10**9, record.sample_id))
    return records


def flatten_record_scores(record: DiscoveryRecord) -> Dict[str, float]:
    gap = float(record.summary.get("gap", float("nan")))
    score_map: Dict[str, float] = {}
    if not math.isfinite(gap) or abs(gap) <= 1e-8:
        return score_map
    for layer in range(record.head_scores.shape[0]):
        for head in range(record.head_scores.shape[1]):
            score_map[f"L{layer}H{head}"] = float(record.head_scores[layer, head].item() / gap)
    for layer in range(record.mlp_scores.shape[0]):
        score_map[f"MLP{layer}"] = float(record.mlp_scores[layer].item() / gap)
    return score_map


def aggregate_node_statistics(records: Sequence[DiscoveryRecord]) -> List[Dict[str, object]]:
    if not records:
        return []
    score_lists: Dict[str, List[float]] = defaultdict(list)
    candidate_counts: Dict[str, int] = defaultdict(int)
    weight_sums: Dict[str, float] = defaultdict(float)
    total_weight = 0.0
    for record in records:
        score_map = flatten_record_scores(record)
        weight = float(record.summary.get("gap", 0.0))
        if math.isfinite(weight) and weight > 0:
            total_weight += weight
        for node, score in score_map.items():
            score_lists[node].append(float(score))
        for node in record.candidate_nodes:
            candidate_counts[node] += 1
            if math.isfinite(weight) and weight > 0:
                weight_sums[node] += weight

    rows: List[Dict[str, object]] = []
    n_records = len(records)
    for node, values in score_lists.items():
        positives = [value for value in values if value > 0]
        rows.append(
            {
                "node": node,
                "layer": node_layer(node),
                "kind": "mlp" if node.startswith("MLP") else "head",
                "candidate_support_rate": candidate_counts.get(node, 0) / max(1, n_records),
                "candidate_support_gap_weighted": weight_sums.get(node, 0.0) / max(total_weight, 1e-8),
                "positive_rate": sum(1 for value in values if value > 0) / max(1, len(values)),
                "score_norm_median": float(np.median(values)) if values else float("nan"),
                "score_norm_mean": float(np.mean(values)) if values else float("nan"),
                "score_norm_positive_median": float(np.median(positives)) if positives else 0.0,
                "score_norm_positive_mean": float(np.mean(positives)) if positives else 0.0,
            }
        )
    rows.sort(
        key=lambda row: (
            float(row["candidate_support_gap_weighted"]),
            float(row["candidate_support_rate"]),
            float(row["score_norm_positive_median"]),
            float(row["score_norm_mean"]),
        ),
        reverse=True,
    )
    return rows


def top_k_nodes(node_rows: Sequence[Dict[str, object]], k: int) -> List[str]:
    return [str(row["node"]) for row in list(node_rows)[: max(0, int(k))]]


def sample_ids_from_records(records: Sequence[DiscoveryRecord]) -> List[str]:
    return [record.sample_id for record in records]


def select_dataset_samples_by_ids(dataset_root: Path, sample_ids: Sequence[str]) -> List[ToolCallSample]:
    samples = load_dataset_samples(dataset_root.resolve())
    return select_samples(samples, sample_ids=list(sample_ids))


def build_behavior_row(
    *,
    case,
    tokenizer,
    nodes: Sequence[str],
    label: str,
    suff_logits: torch.Tensor,
    nec_logits: torch.Tensor,
) -> Dict[str, object]:
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
        "sample_id": case.sample.sample_id,
        "sample_rank": case.sample.sample_rank,
        "filename": case.sample.filename,
        "direction": case.spec.name,
        "label": label,
        "n_nodes": len(nodes),
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


def evaluate_node_sets_jointly(
    model,
    tokenizer,
    samples: Sequence[ToolCallSample],
    direction: str,
    circuits: Dict[str, Sequence[str]],
    *,
    split_label: str,
    progress_desc: str,
) -> Tuple[Dict[str, List[Dict[str, object]]], Dict[str, Dict[str, object]]]:
    all_nodes = sorted({node for nodes in circuits.values() for node in nodes}, key=lambda node: (node_layer(node), node))
    rows_by_label: Dict[str, List[Dict[str, object]]] = {label: [] for label in circuits}
    pbar = tqdm(samples, desc=progress_desc, dynamic_ncols=True)
    for sample in pbar:
        case = build_direction_case(model, tokenizer, sample, direction)
        clean_cache = collect_cache_cpu_for_nodes(model, case.clean_tokens, all_nodes)
        corrupt_cache = collect_cache_cpu_for_nodes(model, case.corrupt_tokens, all_nodes)
        for label, nodes in circuits.items():
            suff_logits = run_logits_with_patched_nodes(model, case.corrupt_tokens, clean_cache, nodes)
            nec_logits = run_logits_with_patched_nodes(model, case.clean_tokens, corrupt_cache, nodes)
            rows_by_label[label].append(
                build_behavior_row(
                    case=case,
                    tokenizer=tokenizer,
                    nodes=nodes,
                    label=label,
                    suff_logits=suff_logits,
                    nec_logits=nec_logits,
                )
            )
        pbar.set_postfix(sample=sample.sample_id)

    summary_by_label: Dict[str, Dict[str, object]] = {}
    for label, rows in rows_by_label.items():
        summary_by_label[label] = summarize_behavior_rows(
            rows,
            label=label,
            nodes=list(circuits[label]),
            split_label=split_label,
        )
    return rows_by_label, summary_by_label


def combined_node_score(summary: Dict[str, object]) -> float:
    return float(
        0.35 * float(summary["logit_margin__suff_recovery_ratio__median"])
        + 0.20 * float(summary["target_prob__suff_recovery_ratio__median"])
        + 0.15 * float(summary["endpoint_kl_score__suff_recovery_ratio__median"])
        + 0.20 * float(summary["logit_margin__nec_drop_ratio__median"])
        + 0.10 * float(summary["target_prob__nec_drop_ratio__median"])
    )


def choose_best_k(sweep_rows: Sequence[Dict[str, object]], tolerance: float = 0.015) -> Dict[str, object]:
    if not sweep_rows:
        raise ValueError("Empty node-K sweep rows.")
    best_score = max(float(row["combined_score"]) for row in sweep_rows)
    eligible = [row for row in sweep_rows if float(row["combined_score"]) >= best_score - tolerance]
    eligible.sort(key=lambda row: (int(row["k"]), -float(row["combined_score"])))
    return dict(eligible[0])


def candidate_pairs(nodes: Sequence[str]) -> List[Tuple[str, str]]:
    ordered = sorted(nodes, key=lambda node: (node_layer(node), 0 if node.startswith("MLP") else 1, node))
    pairs: List[Tuple[str, str]] = []
    for idx, source in enumerate(ordered):
        for target in ordered[idx + 1 :]:
            if node_layer(source) < node_layer(target):
                pairs.append((source, target))
    return pairs


def compute_edge_mediation_rows(
    model,
    tokenizer,
    samples: Sequence[ToolCallSample],
    direction: str,
    nodes: Sequence[str],
    *,
    progress_desc: str,
) -> List[Dict[str, object]]:
    node_pairs = candidate_pairs(nodes)
    rows: List[Dict[str, object]] = []
    pbar = tqdm(samples, desc=progress_desc, dynamic_ncols=True)
    for sample in pbar:
        case = build_direction_case(model, tokenizer, sample, direction)
        clean_cache = collect_cache_cpu_for_nodes(model, case.clean_tokens, nodes)
        solo_cache: Dict[str, Dict[str, float]] = {}
        for _, target in node_pairs:
            if target in solo_cache:
                continue
            logits = run_logits_with_patched_nodes(model, case.corrupt_tokens, clean_cache, [target])
            solo_cache[target] = metric_raws_from_logits(
                logits,
                objective=case.endpoint_objective,
                target_token_id=case.target_token_id,
                distractor_token_id=case.distractor_token_id,
            )

        for source, target in node_pairs:
            uv_logits = run_logits_with_patched_nodes(model, case.corrupt_tokens, clean_cache, [source, target])
            uv_raws = metric_raws_from_logits(
                uv_logits,
                objective=case.endpoint_objective,
                target_token_id=case.target_token_id,
                distractor_token_id=case.distractor_token_id,
            )
            row: Dict[str, object] = {
                "sample_id": sample.sample_id,
                "direction": direction,
                "source": source,
                "target": target,
            }
            for metric_name in METRIC_ORDER:
                clean_raw = float(case.clean_metric_raws[metric_name])
                corrupt_raw = float(case.corrupt_metric_raws[metric_name])
                denom = clean_raw - corrupt_raw
                mediated_raw = float(uv_raws[metric_name] - solo_cache[target][metric_name])
                row[f"{metric_name}__target_only_raw"] = float(solo_cache[target][metric_name])
                row[f"{metric_name}__joint_raw"] = float(uv_raws[metric_name])
                row[f"{metric_name}__mediated_raw"] = mediated_raw
                row[f"{metric_name}__mediated_ratio"] = mediated_raw / denom if math.isfinite(denom) and abs(denom) > 1e-8 else float("nan")
            rows.append(row)
        pbar.set_postfix(sample=sample.sample_id)
    return rows


def summarize_edge_mediation_rows(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    buckets: Dict[Tuple[str, str], List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        buckets[(str(row["source"]), str(row["target"]))].append(row)
    summary_rows: List[Dict[str, object]] = []
    for (source, target), items in buckets.items():
        row: Dict[str, object] = {
            "source": source,
            "target": target,
            "layer_source": node_layer(source),
            "layer_target": node_layer(target),
            "n_samples": len(items),
        }
        for metric_name in METRIC_ORDER:
            raw_key = f"{metric_name}__mediated_raw"
            ratio_key = f"{metric_name}__mediated_ratio"
            raw_vals = [float(item[raw_key]) for item in items if math.isfinite(float(item[raw_key]))]
            ratio_vals = [float(item[ratio_key]) for item in items if math.isfinite(float(item[ratio_key]))]
            row[f"{metric_name}__mediated_raw__median"] = float(np.median(raw_vals)) if raw_vals else float("nan")
            row[f"{metric_name}__mediated_raw__mean"] = float(np.mean(raw_vals)) if raw_vals else float("nan")
            row[f"{metric_name}__mediated_ratio__median"] = float(np.median(ratio_vals)) if ratio_vals else float("nan")
            row[f"{metric_name}__mediated_ratio__mean"] = float(np.mean(ratio_vals)) if ratio_vals else float("nan")
            row[f"{metric_name}__mediated_ratio__positive_rate"] = sum(1 for value in ratio_vals if value > 0) / max(1, len(ratio_vals))
        summary_rows.append(row)
    summary_rows.sort(key=lambda row: float(row["logit_margin__mediated_ratio__median"]), reverse=True)
    return summary_rows


def repair_directed_dag(
    *,
    nodes: Sequence[str],
    direct_edges: Sequence[Tuple[str, str]],
    output_node_label: str,
) -> List[Dict[str, object]]:
    node_list = sorted(nodes, key=lambda node: (node_layer(node), 0 if node.startswith("MLP") else 1, node))
    direct_set = []
    seen = set()
    for source, target in direct_edges:
        if (source, target) in seen:
            continue
        seen.add((source, target))
        direct_set.append((source, target))

    indeg: Dict[str, int] = defaultdict(int)
    outdeg: Dict[str, int] = defaultdict(int)
    for source, target in direct_set:
        indeg[target] += 1
        outdeg[source] += 1

    rows: List[Dict[str, object]] = [
        {"source": source, "target": target, "edge_type": "mediated"} for source, target in direct_set
    ]
    roots = [node for node in node_list if indeg.get(node, 0) == 0]
    sinks = [node for node in node_list if outdeg.get(node, 0) == 0]
    for node in roots:
        rows.append({"source": INPUT_NODE, "target": node, "edge_type": "added_input"})
    for node in sinks:
        rows.append({"source": node, "target": output_node_label, "edge_type": "added_output"})
    return rows


def threshold_sweep_rows(
    edge_summary_rows: Sequence[Dict[str, object]],
    *,
    nodes: Sequence[str],
    thresholds: Sequence[float],
    output_node_label: str,
) -> List[Dict[str, object]]:
    total_positive_mass = sum(
        max(0.0, float(row["logit_margin__mediated_ratio__median"]))
        for row in edge_summary_rows
        if math.isfinite(float(row["logit_margin__mediated_ratio__median"]))
    )
    if total_positive_mass <= 0:
        total_positive_mass = 1.0
    total_positive_prob_mass = sum(
        max(0.0, float(row["target_prob__mediated_ratio__median"]))
        for row in edge_summary_rows
        if math.isfinite(float(row["target_prob__mediated_ratio__median"]))
    )
    if total_positive_prob_mass <= 0:
        total_positive_prob_mass = 1.0

    sweep_rows: List[Dict[str, object]] = []
    for threshold in thresholds:
        direct_edges = [
            (str(row["source"]), str(row["target"]))
            for row in edge_summary_rows
            if float(row["logit_margin__mediated_ratio__median"]) > float(threshold)
        ]
        dag_rows = repair_directed_dag(nodes=nodes, direct_edges=direct_edges, output_node_label=output_node_label)
        retained_mass = sum(
            max(0.0, float(row["logit_margin__mediated_ratio__median"]))
            for row in edge_summary_rows
            if float(row["logit_margin__mediated_ratio__median"]) > float(threshold)
        )
        retained_prob_mass = sum(
            max(0.0, float(row["target_prob__mediated_ratio__median"]))
            for row in edge_summary_rows
            if float(row["logit_margin__mediated_ratio__median"]) > float(threshold)
        )
        added_edges = [row for row in dag_rows if row["edge_type"] != "mediated"]
        sweep_rows.append(
            {
                "threshold": float(threshold),
                "n_nodes": len(nodes),
                "direct_edge_count": len(direct_edges),
                "added_edge_count": len(added_edges),
                "total_edge_count": len(dag_rows),
                "positive_mass_retained": float(retained_mass / total_positive_mass),
                "positive_prob_mass_retained": float(retained_prob_mass / total_positive_prob_mass),
            }
        )
    return sweep_rows


def choose_edge_threshold(sweep_rows: Sequence[Dict[str, object]]) -> Dict[str, object]:
    if not sweep_rows:
        raise ValueError("Empty edge threshold sweep rows.")
    eligible = [
        row
        for row in sweep_rows
        if float(row["positive_mass_retained"]) >= 0.55
        and int(row["direct_edge_count"]) >= max(1, int(row["n_nodes"]) - 1)
    ]
    if eligible:
        eligible.sort(
            key=lambda row: (
                int(row["total_edge_count"]),
                int(row["added_edge_count"]),
                -float(row["threshold"]),
            )
        )
        return dict(eligible[0])
    ranked = sorted(
        sweep_rows,
        key=lambda row: (
            float(row["positive_mass_retained"]) - 0.03 * float(row["total_edge_count"]),
            -float(row["threshold"]),
        ),
        reverse=True,
    )
    return dict(ranked[0])


def load_baseline_directional_circuits(pipeline_root: Path) -> Dict[str, object]:
    forward = json.loads((pipeline_root / "forward_aggregate" / "global_core_summary.json").read_text(encoding="utf-8"))
    reverse = json.loads((pipeline_root / "reverse_aggregate" / "global_core_summary.json").read_text(encoding="utf-8"))
    bidirectional = json.loads((pipeline_root / "bidirectional" / "bidirectional_summary.json").read_text(encoding="utf-8"))
    final_nodes: List[str] = []
    with (pipeline_root / "final_signed_circuit" / "final_signed_nodes.csv").open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            final_nodes.append(str(row["node"]))
    final_edges: List[Tuple[str, str]] = []
    with (pipeline_root / "final_signed_circuit" / "final_signed_edges.csv").open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            final_edges.append((str(row["source"]), str(row["target"])))
    return {
        "forward_nodes": [str(node) for node in forward["core_nodes"]],
        "forward_edges": [tuple(edge) for edge in forward["core_edges"]],
        "reverse_nodes": [str(node) for node in reverse["core_nodes"]],
        "reverse_edges": [tuple(edge) for edge in reverse["core_edges"]],
        "union_nodes": final_nodes,
        "union_edges": final_edges,
        "shared_nodes": [str(node) for node in bidirectional["support_analysis"]["shared_backbone_nodes"]],
        "promote_nodes": [str(node) for node in bidirectional["support_analysis"]["forward_selective_nodes"]],
        "suppress_nodes": [str(node) for node in bidirectional["support_analysis"]["reverse_selective_nodes"]],
        "shared_edges": [tuple(edge) for edge in bidirectional["support_analysis"]["shared_backbone_edges"]],
        "promote_edges": [tuple(edge) for edge in bidirectional["support_analysis"]["forward_selective_edges"]],
        "suppress_edges": [tuple(edge) for edge in bidirectional["support_analysis"]["reverse_selective_edges"]],
    }


def overlap_stats(pred: Sequence[str] | Sequence[Tuple[str, str]], ref: Sequence[str] | Sequence[Tuple[str, str]]) -> Dict[str, object]:
    pred_set = {tuple(item) if isinstance(item, list) else item for item in pred}
    ref_set = {tuple(item) if isinstance(item, list) else item for item in ref}
    inter = pred_set & ref_set
    union = pred_set | ref_set
    return {
        "pred_count": len(pred_set),
        "ref_count": len(ref_set),
        "overlap_count": len(inter),
        "precision": len(inter) / max(1, len(pred_set)),
        "recall": len(inter) / max(1, len(ref_set)),
        "jaccard": len(inter) / max(1, len(union)),
        "overlap_items": sorted(inter),
        "pred_only": sorted(pred_set - ref_set),
        "ref_only": sorted(ref_set - pred_set),
    }


def edge_list_from_dag_rows(rows: Sequence[Dict[str, object]], *, include_added: bool) -> List[Tuple[str, str]]:
    edge_rows = rows if include_added else [row for row in rows if row["edge_type"] == "mediated"]
    return [(str(row["source"]), str(row["target"])) for row in edge_rows]


def build_markdown_table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> List[str]:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return lines


def format_float(value: object, digits: int = 4) -> str:
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        return "nan"
    return f"{float(value):.{digits}f}"


def run_pilot(args) -> None:
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    batch_roots = {
        ("forward", 5): Path(args.forward_ig5_root).resolve(),
        ("reverse", 5): Path(args.reverse_ig5_root).resolve(),
        ("forward", 10): Path(args.forward_ig10_root).resolve(),
        ("reverse", 10): Path(args.reverse_ig10_root).resolve(),
    }
    k_values = parse_int_list(args.k_values)
    threshold_values = parse_float_list(args.edge_thresholds)
    all_records = {key: load_discovery_records(root) for key, root in batch_roots.items()}
    pilot_sample_ids = sample_ids_from_records(all_records[("forward", 10)])
    pilot_samples = select_dataset_samples_by_ids(Path(args.dataset_root), pilot_sample_ids)
    model, tokenizer = load_hooked_qwen3(args.model_path, device=args.device, dtype=torch.bfloat16)

    node_sweep_rows: List[Dict[str, object]] = []
    chosen_k_by_config: Dict[Tuple[str, int], Dict[str, object]] = {}
    node_tables: Dict[Tuple[str, int], List[Dict[str, object]]] = {}
    for (direction, ig_steps), records in all_records.items():
        node_rows = aggregate_node_statistics(records)
        node_tables[(direction, ig_steps)] = node_rows
        write_csv(node_rows, output_root / f"{direction}_ig{ig_steps}_node_table.csv")
        circuits = {f"k_{k}": top_k_nodes(node_rows, k) for k in k_values}
        _, summary_by_label = evaluate_node_sets_jointly(
            model,
            tokenizer,
            pilot_samples,
            direction,
            circuits,
            split_label="pilot",
            progress_desc=f"Pilot node sweep {direction} ig{ig_steps}",
        )
        local_rows: List[Dict[str, object]] = []
        for label, summary in summary_by_label.items():
            k = int(label.split("_")[1])
            row = {
                "direction": direction,
                "ig_steps": ig_steps,
                "k": k,
                "n_nodes": summary["n_nodes"],
                "combined_score": combined_node_score(summary),
                "logit_margin_suff_ratio_median": summary["logit_margin__suff_recovery_ratio__median"],
                "logit_margin_nec_ratio_median": summary["logit_margin__nec_drop_ratio__median"],
                "prob_suff_ratio_median": summary["target_prob__suff_recovery_ratio__median"],
                "prob_nec_ratio_median": summary["target_prob__nec_drop_ratio__median"],
                "endpoint_suff_ratio_median": summary["endpoint_kl_score__suff_recovery_ratio__median"],
                "target_top1_rate_suff": summary["target_top1_rate_suff"],
            }
            local_rows.append(row)
            node_sweep_rows.append(row)
        local_rows.sort(key=lambda row: (float(row["combined_score"]), -int(row["k"])), reverse=True)
        chosen_k_by_config[(direction, ig_steps)] = choose_best_k(local_rows)

    write_csv(node_sweep_rows, output_root / "pilot_node_k_sweep.csv")

    avg_score_by_step: Dict[int, float] = {}
    for ig_steps in (5, 10):
        avg_score_by_step[ig_steps] = float(
            np.mean(
                [
                    float(chosen_k_by_config[("forward", ig_steps)]["combined_score"]),
                    float(chosen_k_by_config[("reverse", ig_steps)]["combined_score"]),
                ]
            )
        )
    best_step_score = max(avg_score_by_step.values())
    candidate_steps = [step for step, score in avg_score_by_step.items() if score >= best_step_score - 0.015]
    recommended_ig_steps = min(candidate_steps)

    edge_sweep_rows: List[Dict[str, object]] = []
    threshold_choice: Dict[str, Dict[str, object]] = {}
    for direction in ("forward", "reverse"):
        chosen_k = int(chosen_k_by_config[(direction, recommended_ig_steps)]["k"])
        chosen_nodes = top_k_nodes(node_tables[(direction, recommended_ig_steps)], chosen_k)
        edge_rows = compute_edge_mediation_rows(
            model,
            tokenizer,
            pilot_samples,
            direction,
            chosen_nodes,
            progress_desc=f"Pilot edge mediation {direction} ig{recommended_ig_steps}",
        )
        edge_summary = summarize_edge_mediation_rows(edge_rows)
        write_csv(edge_summary, output_root / f"pilot_{direction}_edge_summary.csv")
        sweep = threshold_sweep_rows(
            edge_summary,
            nodes=chosen_nodes,
            thresholds=threshold_values,
            output_node_label=get_direction_spec(direction).output_node_label,
        )
        for row in sweep:
            row["direction"] = direction
            row["ig_steps"] = recommended_ig_steps
        edge_sweep_rows.extend(sweep)
        threshold_choice[direction] = choose_edge_threshold(sweep)

    write_csv(edge_sweep_rows, output_root / "pilot_edge_threshold_sweep.csv")

    stability = {}
    for direction in ("forward", "reverse"):
        k = int(chosen_k_by_config[(direction, recommended_ig_steps)]["k"])
        nodes_5 = set(top_k_nodes(node_tables[(direction, 5)], k))
        nodes_10 = set(top_k_nodes(node_tables[(direction, 10)], k))
        stability[direction] = overlap_stats(sorted(nodes_5), sorted(nodes_10))

    recommended = {
        "recommended_ig_steps": recommended_ig_steps,
        "avg_score_by_step": avg_score_by_step,
        "chosen_k": {
            direction: int(chosen_k_by_config[(direction, recommended_ig_steps)]["k"])
            for direction in ("forward", "reverse")
        },
        "chosen_threshold": {
            direction: float(threshold_choice[direction]["threshold"]) for direction in ("forward", "reverse")
        },
        "stability_ig5_vs_ig10": stability,
    }
    write_json(recommended, output_root / "recommended_config.json")

    summary_lines = [
        "# AP-IG Pilot Summary",
        "",
        f"- pilot 样本数：{len(pilot_samples)}",
        f"- 推荐 ig_steps：`{recommended_ig_steps}`",
        f"- forward 推荐 K：`{recommended['chosen_k']['forward']}`，阈值：`{format_float(recommended['chosen_threshold']['forward'], 3)}`",
        f"- reverse 推荐 K：`{recommended['chosen_k']['reverse']}`，阈值：`{format_float(recommended['chosen_threshold']['reverse'], 3)}`",
        f"- ig5 vs ig10 平均 node-validate 分数：`{format_float(avg_score_by_step[5])}` / `{format_float(avg_score_by_step[10])}`",
        "",
        "## Node Stability",
        "",
    ]
    for direction in ("forward", "reverse"):
        stats = stability[direction]
        summary_lines.append(
            f"- `{direction}`: overlap `{stats['overlap_count']}` / union `{stats['pred_count'] + stats['ref_count'] - stats['overlap_count']}`, "
            f"Jaccard=`{format_float(stats['jaccard'])}`."
        )
    (output_root / "pilot_summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")


def run_finalize(args) -> None:
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    config = json.loads(Path(args.config_json).resolve().read_text(encoding="utf-8"))
    recommended_ig_steps = int(config["recommended_ig_steps"])
    chosen_k = {direction: int(value) for direction, value in config["chosen_k"].items()}
    chosen_threshold = {direction: float(value) for direction, value in config["chosen_threshold"].items()}

    batch_roots = {
        ("forward", 5): Path(args.forward_ig5_root).resolve(),
        ("reverse", 5): Path(args.reverse_ig5_root).resolve(),
        ("forward", 10): Path(args.forward_ig10_root).resolve(),
        ("reverse", 10): Path(args.reverse_ig10_root).resolve(),
    }
    train_records = {key: load_discovery_records(root) for key, root in batch_roots.items()}
    train_samples = load_dataset_samples(Path(args.train_dataset_root).resolve())
    test_samples = load_dataset_samples(Path(args.test_dataset_root).resolve())
    if int(args.train_max_samples) > 0:
        train_samples = train_samples[: int(args.train_max_samples)]
    if int(args.test_max_samples) > 0:
        test_samples = test_samples[: int(args.test_max_samples)]
    baseline = load_baseline_directional_circuits(Path(args.baseline_pipeline_root).resolve())

    final_node_tables: Dict[Tuple[str, int], List[Dict[str, object]]] = {}
    final_nodes: Dict[str, List[str]] = {}
    for direction in ("forward", "reverse"):
        for step in (5, 10):
            node_rows = aggregate_node_statistics(train_records[(direction, step)])
            final_node_tables[(direction, step)] = node_rows
            write_csv(node_rows, output_root / "node_tables" / f"{direction}_ig{step}_node_table.csv")
        final_nodes[direction] = top_k_nodes(final_node_tables[(direction, recommended_ig_steps)], chosen_k[direction])

    stability_full = {}
    for direction in ("forward", "reverse"):
        nodes_5 = set(top_k_nodes(final_node_tables[(direction, 5)], chosen_k[direction]))
        nodes_10 = set(top_k_nodes(final_node_tables[(direction, 10)], chosen_k[direction]))
        stability_full[direction] = overlap_stats(sorted(nodes_5), sorted(nodes_10))

    for direction in ("forward", "reverse"):
        write_csv(
            [{"node": node} for node in final_nodes[direction]],
            output_root / "final_nodes" / f"{direction}_final_nodes.csv",
        )

    model, tokenizer = load_hooked_qwen3(args.model_path, device=args.device, dtype=torch.bfloat16)
    final_edge_rows: Dict[str, List[Dict[str, object]]] = {}
    final_dag_rows: Dict[str, List[Dict[str, object]]] = {}
    edge_summaries: Dict[str, List[Dict[str, object]]] = {}
    for direction in ("forward", "reverse"):
        edge_rows = compute_edge_mediation_rows(
            model,
            tokenizer,
            train_samples,
            direction,
            final_nodes[direction],
            progress_desc=f"Full edge mediation {direction}",
        )
        edge_summary = summarize_edge_mediation_rows(edge_rows)
        edge_summaries[direction] = edge_summary
        write_csv(edge_rows, output_root / "edge_discovery" / f"{direction}_edge_per_sample.csv")
        write_csv(edge_summary, output_root / "edge_discovery" / f"{direction}_edge_summary.csv")
        direct_edges = [
            (str(row["source"]), str(row["target"]))
            for row in edge_summary
            if float(row["logit_margin__mediated_ratio__median"]) > chosen_threshold[direction]
        ]
        dag_rows = repair_directed_dag(
            nodes=final_nodes[direction],
            direct_edges=direct_edges,
            output_node_label=get_direction_spec(direction).output_node_label,
        )
        final_edge_rows[direction] = [
            row
            for row in dag_rows
            if row["edge_type"] == "mediated"
        ]
        final_dag_rows[direction] = dag_rows
        write_csv(dag_rows, output_root / "edge_discovery" / f"{direction}_final_dag_edges.csv")
        draw_circuit_with_output(
            nodes=final_nodes[direction],
            edges=[(str(row["source"]), str(row["target"])) for row in dag_rows],
            out_path=output_root / "edge_discovery" / f"{direction}_final_circuit.png",
            title=f"AP-IG {direction.title()} Circuit",
            input_node=INPUT_NODE,
            output_node=get_direction_spec(direction).output_node_label,
        )

    ap_ig_groups = {
        "shared_nodes": sorted(set(final_nodes["forward"]) & set(final_nodes["reverse"])),
        "promote_nodes": sorted(set(final_nodes["forward"]) - set(final_nodes["reverse"])),
        "suppress_nodes": sorted(set(final_nodes["reverse"]) - set(final_nodes["forward"])),
        "shared_edges": sorted(
            set(edge_list_from_dag_rows(final_dag_rows["forward"], include_added=False))
            & set(edge_list_from_dag_rows(final_dag_rows["reverse"], include_added=False))
        ),
        "promote_edges": sorted(
            set(edge_list_from_dag_rows(final_dag_rows["forward"], include_added=False))
            - set(edge_list_from_dag_rows(final_dag_rows["reverse"], include_added=False))
        ),
        "suppress_edges": sorted(
            set(edge_list_from_dag_rows(final_dag_rows["reverse"], include_added=False))
            - set(edge_list_from_dag_rows(final_dag_rows["forward"], include_added=False))
        ),
    }
    write_json(ap_ig_groups, output_root / "group_overlap" / "ap_ig_groups.json")

    direction_circuit_maps = {
        "forward": {
            "ap_ig_forward": final_nodes["forward"],
            "baseline_forward": baseline["forward_nodes"],
            "ap_ig_union": sorted(set(final_nodes["forward"]) | set(final_nodes["reverse"])),
            "baseline_union": baseline["union_nodes"],
        },
        "reverse": {
            "ap_ig_reverse": final_nodes["reverse"],
            "baseline_reverse": baseline["reverse_nodes"],
            "ap_ig_union": sorted(set(final_nodes["forward"]) | set(final_nodes["reverse"])),
            "baseline_union": baseline["union_nodes"],
        },
    }

    behavior_summary_rows: List[Dict[str, object]] = []
    behavior_summary_map: Dict[Tuple[str, str, str], Dict[str, object]] = {}
    for split_label, split_samples in (("train", train_samples), ("test", test_samples)):
        for direction in ("forward", "reverse"):
            rows_by_label, summaries = evaluate_node_sets_jointly(
                model,
                tokenizer,
                split_samples,
                direction,
                direction_circuit_maps[direction],
                split_label=split_label,
                progress_desc=f"Evaluate {split_label} {direction}",
            )
            for label, rows in rows_by_label.items():
                write_csv(rows, output_root / "behavior_eval" / f"{split_label}_{direction}_{label}_per_sample.csv")
            for label, summary in summaries.items():
                behavior_summary_map[(split_label, direction, label)] = summary
                behavior_summary_rows.append(
                    {
                        "split": split_label,
                        "direction": direction,
                        "label": label,
                        "n_nodes": summary["n_nodes"],
                        "logit_margin_suff_ratio_median": summary["logit_margin__suff_recovery_ratio__median"],
                        "logit_margin_nec_ratio_median": summary["logit_margin__nec_drop_ratio__median"],
                        "prob_suff_ratio_median": summary["target_prob__suff_recovery_ratio__median"],
                        "prob_nec_ratio_median": summary["target_prob__nec_drop_ratio__median"],
                        "endpoint_suff_ratio_median": summary["endpoint_kl_score__suff_recovery_ratio__median"],
                        "target_top1_rate_suff": summary["target_top1_rate_suff"],
                    }
                )
                write_json(summary, output_root / "behavior_eval" / f"{split_label}_{direction}_{label}_summary.json")
    write_csv(behavior_summary_rows, output_root / "behavior_eval" / "behavior_summary.csv")

    structure = {
        "forward_vs_baseline": {
            "nodes": overlap_stats(final_nodes["forward"], baseline["forward_nodes"]),
            "direct_edges": overlap_stats(
                edge_list_from_dag_rows(final_dag_rows["forward"], include_added=False),
                baseline["forward_edges"],
            ),
            "full_dag_edges": overlap_stats(
                edge_list_from_dag_rows(final_dag_rows["forward"], include_added=True),
                baseline["forward_edges"],
            ),
        },
        "reverse_vs_baseline": {
            "nodes": overlap_stats(final_nodes["reverse"], baseline["reverse_nodes"]),
            "direct_edges": overlap_stats(
                edge_list_from_dag_rows(final_dag_rows["reverse"], include_added=False),
                baseline["reverse_edges"],
            ),
            "full_dag_edges": overlap_stats(
                edge_list_from_dag_rows(final_dag_rows["reverse"], include_added=True),
                baseline["reverse_edges"],
            ),
        },
        "union_vs_baseline": {
            "nodes": overlap_stats(
                sorted(set(final_nodes["forward"]) | set(final_nodes["reverse"])),
                baseline["union_nodes"],
            ),
        },
        "group_overlap": {
            "shared_nodes": overlap_stats(ap_ig_groups["shared_nodes"], baseline["shared_nodes"]),
            "promote_nodes": overlap_stats(ap_ig_groups["promote_nodes"], baseline["promote_nodes"]),
            "suppress_nodes": overlap_stats(ap_ig_groups["suppress_nodes"], baseline["suppress_nodes"]),
            "shared_edges": overlap_stats(ap_ig_groups["shared_edges"], baseline["shared_edges"]),
            "promote_edges": overlap_stats(ap_ig_groups["promote_edges"], baseline["promote_edges"]),
            "suppress_edges": overlap_stats(ap_ig_groups["suppress_edges"], baseline["suppress_edges"]),
        },
        "ig5_vs_ig10_full_stability": stability_full,
    }
    write_json(structure, output_root / "comparison" / "structure_comparison.json")

    behavior_compare = {
        "train_forward": {
            "ap_ig": behavior_summary_map[("train", "forward", "ap_ig_forward")],
            "baseline": behavior_summary_map[("train", "forward", "baseline_forward")],
        },
        "test_forward": {
            "ap_ig": behavior_summary_map[("test", "forward", "ap_ig_forward")],
            "baseline": behavior_summary_map[("test", "forward", "baseline_forward")],
        },
        "train_reverse": {
            "ap_ig": behavior_summary_map[("train", "reverse", "ap_ig_reverse")],
            "baseline": behavior_summary_map[("train", "reverse", "baseline_reverse")],
        },
        "test_reverse": {
            "ap_ig": behavior_summary_map[("test", "reverse", "ap_ig_reverse")],
            "baseline": behavior_summary_map[("test", "reverse", "baseline_reverse")],
        },
        "train_union_forward": {
            "ap_ig": behavior_summary_map[("train", "forward", "ap_ig_union")],
            "baseline": behavior_summary_map[("train", "forward", "baseline_union")],
        },
        "test_union_forward": {
            "ap_ig": behavior_summary_map[("test", "forward", "ap_ig_union")],
            "baseline": behavior_summary_map[("test", "forward", "baseline_union")],
        },
        "train_union_reverse": {
            "ap_ig": behavior_summary_map[("train", "reverse", "ap_ig_union")],
            "baseline": behavior_summary_map[("train", "reverse", "baseline_union")],
        },
        "test_union_reverse": {
            "ap_ig": behavior_summary_map[("test", "reverse", "ap_ig_union")],
            "baseline": behavior_summary_map[("test", "reverse", "baseline_union")],
        },
    }
    write_json(behavior_compare, output_root / "comparison" / "behavior_comparison.json")

    final_report_lines = [
        "# AP-IG Discovery Comparison",
        "",
        f"- 训练集：`{len(train_samples)}`，测试集：`{len(test_samples)}`。",
        f"- 主发现积分步数：`{recommended_ig_steps}`。",
        f"- forward 最终节点数：`{len(final_nodes['forward'])}`；reverse 最终节点数：`{len(final_nodes['reverse'])}`。",
        f"- forward mediated 阈值：`{format_float(chosen_threshold['forward'], 3)}`；reverse mediated 阈值：`{format_float(chosen_threshold['reverse'], 3)}`。",
        "",
        "## 关键指标",
        "",
    ]
    metric_rows = []
    for split_label, direction, ap_label, base_label in (
        ("train", "forward", "ap_ig_forward", "baseline_forward"),
        ("test", "forward", "ap_ig_forward", "baseline_forward"),
        ("train", "reverse", "ap_ig_reverse", "baseline_reverse"),
        ("test", "reverse", "ap_ig_reverse", "baseline_reverse"),
    ):
        ap = behavior_summary_map[(split_label, direction, ap_label)]
        base = behavior_summary_map[(split_label, direction, base_label)]
        metric_rows.append(
            [
                f"{split_label}-{direction}",
                format_float(ap["logit_margin__suff_recovery_ratio__median"]),
                format_float(base["logit_margin__suff_recovery_ratio__median"]),
                format_float(ap["logit_margin__nec_drop_ratio__median"]),
                format_float(base["logit_margin__nec_drop_ratio__median"]),
                format_float(ap["target_prob__suff_recovery_ratio__median"]),
                format_float(base["target_prob__suff_recovery_ratio__median"]),
            ]
        )
    final_report_lines.extend(
        build_markdown_table(
            [
                "split-direction",
                "AP-IG margin suff",
                "Baseline margin suff",
                "AP-IG margin nec",
                "Baseline margin nec",
                "AP-IG prob suff",
                "Baseline prob suff",
            ],
            metric_rows,
        )
    )
    final_report_lines.extend(
        [
            "",
            "## 结构对比",
            "",
            f"- forward 节点 overlap：`{structure['forward_vs_baseline']['nodes']['overlap_count']}` / `{structure['forward_vs_baseline']['nodes']['ref_count']}`，Jaccard=`{format_float(structure['forward_vs_baseline']['nodes']['jaccard'])}`。",
            f"- reverse 节点 overlap：`{structure['reverse_vs_baseline']['nodes']['overlap_count']}` / `{structure['reverse_vs_baseline']['nodes']['ref_count']}`，Jaccard=`{format_float(structure['reverse_vs_baseline']['nodes']['jaccard'])}`。",
            f"- forward 直接 mediated 边 overlap：`{structure['forward_vs_baseline']['direct_edges']['overlap_count']}` / `{structure['forward_vs_baseline']['direct_edges']['ref_count']}`。",
            f"- reverse 直接 mediated 边 overlap：`{structure['reverse_vs_baseline']['direct_edges']['overlap_count']}` / `{structure['reverse_vs_baseline']['direct_edges']['ref_count']}`。",
            "",
            "## 结论",
            "",
        ]
    )
    forward_stable = structure["ig5_vs_ig10_full_stability"]["forward"]["jaccard"]
    reverse_stable = structure["ig5_vs_ig10_full_stability"]["reverse"]["jaccard"]
    final_report_lines.append(
        f"- AP-IG node discovery 稳定性：forward ig5/ig10 Jaccard=`{format_float(forward_stable)}`，reverse=`{format_float(reverse_stable)}`。"
    )
    final_report_lines.append(
        f"- mediated-edge discovery 恢复原主干的程度：forward 直接边 recall=`{format_float(structure['forward_vs_baseline']['direct_edges']['recall'])}`，reverse=`{format_float(structure['reverse_vs_baseline']['direct_edges']['recall'])}`。"
    )
    test_forward_ap = behavior_summary_map[("test", "forward", "ap_ig_forward")]
    test_forward_base = behavior_summary_map[("test", "forward", "baseline_forward")]
    test_reverse_ap = behavior_summary_map[("test", "reverse", "ap_ig_reverse")]
    test_reverse_base = behavior_summary_map[("test", "reverse", "baseline_reverse")]
    final_report_lines.append(
        f"- 行为上，forward test margin suff：AP-IG `{format_float(test_forward_ap['logit_margin__suff_recovery_ratio__median'])}` vs baseline `{format_float(test_forward_base['logit_margin__suff_recovery_ratio__median'])}`；"
        f"reverse test margin suff：AP-IG `{format_float(test_reverse_ap['logit_margin__suff_recovery_ratio__median'])}` vs baseline `{format_float(test_reverse_base['logit_margin__suff_recovery_ratio__median'])}`。"
    )
    ap_union_forward = behavior_summary_map[("test", "forward", "ap_ig_union")]
    base_union_forward = behavior_summary_map[("test", "forward", "baseline_union")]
    if float(ap_union_forward["logit_margin__suff_recovery_ratio__median"]) >= float(base_union_forward["logit_margin__suff_recovery_ratio__median"]) - 0.02:
        verdict = "值得继续全量推进"
    else:
        verdict = "不建议替代原方法，只建议保留为补充对照"
    final_report_lines.append(f"- 继续推进判断：`{verdict}`。")
    (output_root / "comparison" / "final_comparison_summary.md").write_text(
        "\n".join(final_report_lines) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Pilot / finalize analysis for the AP-IG discovery pipeline.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    pilot = subparsers.add_parser("pilot", help="Run pilot node/edge sweeps and choose config.")
    pilot.add_argument("--dataset-root", type=str, required=True)
    pilot.add_argument("--forward-ig5-root", type=str, required=True)
    pilot.add_argument("--reverse-ig5-root", type=str, required=True)
    pilot.add_argument("--forward-ig10-root", type=str, required=True)
    pilot.add_argument("--reverse-ig10-root", type=str, required=True)
    pilot.add_argument("--model-path", type=str, default=str(MODEL_PATH_DEFAULT))
    pilot.add_argument("--device", type=str, default="cuda")
    pilot.add_argument("--k-values", type=str, default="6,8,10,12,14,16")
    pilot.add_argument("--edge-thresholds", type=str, default="0.02,0.04,0.06,0.08,0.10,0.12,0.15,0.20")
    pilot.add_argument("--output-root", type=str, required=True)

    finalize = subparsers.add_parser("finalize", help="Run full-train edge discovery, train/test eval and comparison.")
    finalize.add_argument("--train-dataset-root", type=str, required=True)
    finalize.add_argument("--test-dataset-root", type=str, required=True)
    finalize.add_argument("--forward-ig5-root", type=str, required=True)
    finalize.add_argument("--reverse-ig5-root", type=str, required=True)
    finalize.add_argument("--forward-ig10-root", type=str, required=True)
    finalize.add_argument("--reverse-ig10-root", type=str, required=True)
    finalize.add_argument("--config-json", type=str, required=True)
    finalize.add_argument("--baseline-pipeline-root", type=str, default="./results/split/pipeline")
    finalize.add_argument("--model-path", type=str, default=str(MODEL_PATH_DEFAULT))
    finalize.add_argument("--device", type=str, default="cuda")
    finalize.add_argument("--train-max-samples", type=int, default=0)
    finalize.add_argument("--test-max-samples", type=int, default=0)
    finalize.add_argument("--output-root", type=str, required=True)

    args = parser.parse_args()
    if args.command == "pilot":
        run_pilot(args)
        return
    if args.command == "finalize":
        run_finalize(args)
        return
    raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
