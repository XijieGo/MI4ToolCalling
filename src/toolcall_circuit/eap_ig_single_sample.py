#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import torch

if __package__ in {None, ""}:
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

PROJECT_ROOT = Path(__file__).resolve().parents[4]
EAP_SRC = PROJECT_ROOT / "experiment" / "code" / "vendor" / "EAP-IG" / "src"
if str(EAP_SRC) not in sys.path:
    sys.path.insert(0, str(EAP_SRC))

from eap.attribute import attribute  # noqa: E402
from eap.graph import Graph  # noqa: E402

from toolcall_circuit.dataset import (
    ToolCallSample,
    build_position_sets,
    decode_token_at,
    get_tool_call_target_spec,
    resolve_distractor_token,
)
from toolcall_circuit.eap_ig_dataset import (
    build_single_pair_dataloader,
    resolve_direction_spec,
    sample_label,
)
from toolcall_circuit.eap_ig_export import (
    aggregate_node_scores,
    all_real_raw_edge_rows,
    fold_selected_edges,
    top_raw_edge_rows,
    write_csv,
    write_json,
)
from toolcall_circuit.graph_utils import remap_output_node
from toolcall_circuit.eap_ig_objective import (
    build_directional_objective,
    candidate_summary_row,
    endpoint_pair_summary,
    metric_name,
    negative_kl_to_clean_endpoint,
    objective_from_logits,
    summarize_candidates,
)
from toolcall_circuit.single_sample import build_edges, collect_clean_cache_cpu, evaluate_on_base_with_source


def single_token_spec(tokenizer, token_id: int) -> Dict[str, object]:
    token_id = int(token_id)
    return {
        "text": tokenizer.decode([token_id]),
        "token_ids": [token_id],
        "tokens": tokenizer.convert_ids_to_tokens([token_id]),
        "length": 1,
        "is_single_token": True,
    }


def normalize_edge_counts(edge_counts: Sequence[int]) -> List[int]:
    out = sorted({int(x) for x in edge_counts if int(x) > 0})
    if not out:
        raise ValueError("edge_counts must contain at least one positive value")
    return out


def pick_selected_edge_count(edge_counts: Sequence[int], selected_edge_count: int) -> int:
    if selected_edge_count > 0:
        if int(selected_edge_count) not in set(int(x) for x in edge_counts):
            raise ValueError(f"selected_edge_count={selected_edge_count} is not in edge_counts={list(edge_counts)}")
        return int(selected_edge_count)
    return max(int(x) for x in edge_counts)


def _top_head_probe(top_node_scores: Sequence[Dict[str, object]]) -> str:
    for row in top_node_scores:
        name = str(row.get("name", ""))
        if name.startswith("L"):
            return name
    return "L0H0"


def _candidate_artifact_stem(edge_count: int) -> str:
    return f"edge_count_{int(edge_count):05d}"


def _node_sort_key(node_name: str) -> tuple[int, int, str]:
    if node_name.startswith("MLP"):
        return int(node_name[3:]), 0, node_name
    layer_s, head_s = node_name[1:].split("H")
    return int(layer_s), 1, f"{int(layer_s):02d}:{int(head_s):02d}"


def run_one_eap_ig_sample(
    *,
    sample: ToolCallSample,
    out_dir: Path,
    model,
    tokenizer,
    model_path: str,
    direction: str,
    ig_steps: int,
    edge_counts: Sequence[int],
    selected_edge_count: int = 0,
    raw_topk: int = 256,
) -> Dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    edge_counts = normalize_edge_counts(edge_counts)
    selected_edge_count = pick_selected_edge_count(edge_counts, selected_edge_count)
    rough_edge_count = min(edge_counts)

    spec = resolve_direction_spec(sample, direction)

    clean_tokens = model.to_tokens(spec.clean_text, prepend_bos=model.cfg.default_prepend_bos)
    corrupt_tokens = model.to_tokens(spec.corrupt_text, prepend_bos=model.cfg.default_prepend_bos)
    if clean_tokens.shape != corrupt_tokens.shape:
        raise ValueError(
            f"Token shapes differ for {sample.sample_id} ({direction}): "
            f"{tuple(clean_tokens.shape)} vs {tuple(corrupt_tokens.shape)}"
        )

    ids_clean = [int(x) for x in clean_tokens[0].tolist()]
    ids_corrupt = [int(x) for x in corrupt_tokens[0].tolist()]
    pos_sets = build_position_sets(ids_clean, ids_corrupt, tokenizer, clean_text=spec.clean_text)

    with torch.no_grad():
        clean_logits = model(clean_tokens)
        corrupt_logits = model(corrupt_tokens)

    tool_spec = get_tool_call_target_spec(tokenizer, target_text="<tool_call>")
    if not tool_spec.is_single_token:
        raise NotImplementedError(
            f"<tool_call> tokenization is not single-token: {tool_spec.token_ids}. "
            "Update downstream top1/token-gap tooling before using this workflow."
        )
    tool_token = tool_spec.primary_token_id
    no_tool_token = resolve_distractor_token(clean_logits[0, -1, :], tool_token)

    endpoint_temperature = 1.0
    endpoint_masked_token_ids: tuple[int, ...] = ()
    endpoint_objective = build_directional_objective(
        clean_logits=clean_logits,
        tokenizer=tokenizer,
        endpoint_label=spec.objective_endpoint,
        temperature=endpoint_temperature,
        masked_token_ids=endpoint_masked_token_ids,
    )
    endpoint_summary = endpoint_pair_summary(
        direction=direction,
        clean_logits=clean_logits,
        corrupt_logits=corrupt_logits,
        tokenizer=tokenizer,
        temperature=endpoint_temperature,
        masked_token_ids=endpoint_masked_token_ids,
    )

    clean_obj = float(objective_from_logits(clean_logits, endpoint_objective).item())
    corrupt_obj = float(objective_from_logits(corrupt_logits, endpoint_objective).item())
    gap = clean_obj - corrupt_obj
    if not math.isfinite(gap) or abs(gap) <= 1e-8:
        raise ValueError(f"Degenerate endpoint gap for {sample.sample_id} ({direction}): {gap}")

    # The endpoint objective only needs summary stats from these logits; releasing them before
    # EAP-IG attribution prevents borderline samples from OOMing during backward.
    del clean_logits
    del corrupt_logits
    if clean_tokens.is_cuda:
        torch.cuda.empty_cache()

    single_loader = build_single_pair_dataloader(
        clean_text=spec.clean_text,
        corrupt_text=spec.corrupt_text,
        label=sample_label(sample, direction),
    )

    scored_graph = Graph.from_model(model)
    attribute(
        model,
        scored_graph,
        single_loader,
        metric=negative_kl_to_clean_endpoint,
        method="EAP-IG-inputs",
        ig_steps=int(ig_steps),
        quiet=True,
    )

    top_raw_edges = top_raw_edge_rows(scored_graph, output_node_label=spec.output_node_label, topk=raw_topk)
    write_csv(top_raw_edges, out_dir / "top_raw_edges.csv")
    write_json(top_raw_edges, out_dir / "top_raw_edges.json")

    all_raw_edges = all_real_raw_edge_rows(scored_graph, output_node_label=spec.output_node_label)
    all_folded_rows = fold_selected_edges(all_raw_edges)
    folded_lookup = {(str(row["source"]), str(row["target"])): row for row in all_folded_rows}
    all_node_score_rows = aggregate_node_scores(scored_graph, output_node_label=spec.output_node_label)
    top_node_scores = all_node_score_rows[:64]
    node_score_lookup = {str(row["name"]): float(row["score"]) for row in all_node_score_rows}
    clean_cache_cpu = collect_clean_cache_cpu(model, clean_tokens)
    corrupt_cache_cpu = collect_clean_cache_cpu(model, corrupt_tokens)

    candidate_rows: List[Dict[str, object]] = []
    candidate_lookup: Dict[int, Dict[str, object]] = {}
    for edge_count in edge_counts:
        selected_seed_rows = list(all_folded_rows[: int(edge_count)])
        selected_nodes_set = {
            str(node)
            for row in selected_seed_rows
            for node in (row["source"], row["target"])
            if str(node) not in {"Input Embed", spec.output_node_label}
        }
        for row in all_node_score_rows:
            if len(selected_nodes_set) >= max(6, int(edge_count) // 2):
                break
            selected_nodes_set.add(str(row["name"]))
        nodes = sorted(selected_nodes_set, key=_node_sort_key)

        detailed_edges = build_edges(nodes, score_lookup=node_score_lookup, max_parents=2)
        if direction == "reverse":
            detailed_edges = remap_output_node(detailed_edges, target_output=spec.output_node_label)
        detailed_edge_pairs = [(str(src), str(dst)) for src, dst in detailed_edges]

        raw_rows = [
            row
            for row in all_raw_edges
            if (str(row["source_local"]), str(row["target_local"])) in set(detailed_edge_pairs)
        ]
        folded_rows = []
        for src, dst in detailed_edge_pairs:
            if (src, dst) in folded_lookup:
                folded_rows.append(dict(folded_lookup[(src, dst)]))
            else:
                folded_rows.append(
                    {
                        "source": src,
                        "target": dst,
                        "qkv_channels": [],
                        "qkv_count": 0,
                        "score_sum": 0.0,
                        "score_max": 0.0,
                        "score_mean": 0.0,
                        "raw_edge_count": 0,
                        "raw_edges": [],
                    }
                )

        sufficiency_obj = evaluate_on_base_with_source(
            model=model,
            base_tokens=corrupt_tokens,
            source_cache_cpu=clean_cache_cpu,
            patch_nodes=nodes,
            target_token=endpoint_objective,
            distractor_token=None,
        )
        sufficiency_ratio = (sufficiency_obj - corrupt_obj) / gap

        clean_without_selected = evaluate_on_base_with_source(
            model,
            clean_tokens,
            corrupt_cache_cpu,
            nodes,
            endpoint_objective,
            None,
        )
        necessity_drop = clean_obj - clean_without_selected
        necessity_ratio = necessity_drop / gap

        stem = _candidate_artifact_stem(edge_count)
        write_csv(raw_rows, out_dir / f"{stem}_raw_edges.csv")
        write_json(raw_rows, out_dir / f"{stem}_raw_edges.json")
        write_csv(folded_rows, out_dir / f"{stem}_folded_edges.csv")
        write_json(folded_rows, out_dir / f"{stem}_folded_edges.json")

        row = candidate_summary_row(
            edge_count=int(edge_count),
            sufficiency_obj=sufficiency_obj,
            sufficiency_ratio=sufficiency_ratio,
            necessity_obj=clean_without_selected,
            necessity_drop=necessity_drop,
            necessity_ratio=necessity_ratio,
            node_count=len(nodes),
            folded_edge_count=len(detailed_edge_pairs),
        )
        row.update(
            {
                "raw_edge_count": len(raw_rows),
                "artifacts": {
                    "raw_edges_csv": str(out_dir / f"{stem}_raw_edges.csv"),
                    "raw_edges_json": str(out_dir / f"{stem}_raw_edges.json"),
                    "folded_edges_csv": str(out_dir / f"{stem}_folded_edges.csv"),
                    "folded_edges_json": str(out_dir / f"{stem}_folded_edges.json"),
                },
            }
        )
        candidate_rows.append(row)
        candidate_lookup[int(edge_count)] = {
            "nodes": nodes,
            "folded_rows": folded_rows,
            "folded_pairs": detailed_edge_pairs,
            "raw_rows": raw_rows,
            **row,
        }

    write_json(candidate_rows, out_dir / "candidate_edge_sweep.json")

    detailed = candidate_lookup[int(selected_edge_count)]
    rough = candidate_lookup[int(rough_edge_count)]
    probe_head = _top_head_probe(top_node_scores)

    target_token_id = tool_token if direction == "forward" else no_tool_token
    distractor_token_id = no_tool_token if direction == "forward" else tool_token

    contrast_token_details = [
        {
            "position": int(pos),
            "clean_token": decode_token_at(tokenizer, ids_clean, int(pos)),
            "corrupt_token": decode_token_at(tokenizer, ids_corrupt, int(pos)),
        }
        for pos in pos_sets["contrast"]
    ]

    direction_label = "forward_tool_call" if direction == "forward" else "reverse_no_tool"
    summary: Dict[str, object] = {
        "sample_id": sample.sample_id,
        "sample_rank": sample.sample_rank,
        "filename": sample.filename,
        "source_kind": sample.source_kind,
        "q_index": sample.legacy_index,
        "direction": direction_label,
        "decision_label": spec.decision_label,
        "output_node_label": spec.output_node_label,
        "clean_role": spec.clean_role,
        "corrupt_role": spec.corrupt_role,
        "clean_prompt": str(spec.clean_prompt_path),
        "corrupt_prompt": str(spec.corrupt_prompt_path),
        "model_path": model_path,
        "clean_prompt_token_length": int(clean_tokens.shape[1]),
        "corrupt_prompt_token_length": int(corrupt_tokens.shape[1]),
        "token_lengths_aligned": bool(clean_tokens.shape == corrupt_tokens.shape),
        "target_tokenization": single_token_spec(tokenizer, target_token_id),
        "target_token_id": int(target_token_id),
        "target_token_str": tokenizer.decode([int(target_token_id)]),
        "distractor_token_id": int(distractor_token_id),
        "distractor_token_str": tokenizer.decode([int(distractor_token_id)]),
        "tool_call_tokenization": tool_spec.to_dict(tokenizer),
        "ap_mode": "full",
        "quality_mode": "full",
        "discovery_method": "EAP-IG-inputs",
        "graph_level": "edge_folded_to_node",
        "ig_steps": int(ig_steps),
        "topn_rule": f"top_folded_edges_by_raw_score={selected_edge_count}",
        "edge_counts_evaluated": [int(x) for x in edge_counts],
        "selected_edge_count": int(selected_edge_count),
        "rough_edge_count": int(rough_edge_count),
        "folded_edge_policy": "collapse_qkv_to_node_pair_keep_raw_provenance",
        "objective_mode": metric_name(direction),
        "objective_endpoint": spec.objective_endpoint,
        "objective_temperature": endpoint_temperature,
        "objective_masked_token_ids": list(endpoint_masked_token_ids),
        "clean_obj": clean_obj,
        "corrupt_obj": corrupt_obj,
        "gap": gap,
        "clean_kl_to_endpoint": -clean_obj,
        "corrupt_kl_to_endpoint": -corrupt_obj,
        "endpoint_js_divergence": endpoint_summary["endpoint_js_divergence"],
        "detailed_obj": float(detailed["sufficiency_obj"]),
        "detailed_ratio_vs_gap": float(detailed["sufficiency_ratio_vs_gap"]),
        "detailed_kl_recovery_ratio": float(detailed["sufficiency_ratio_vs_gap"]),
        "rough_obj": float(rough["sufficiency_obj"]),
        "rough_ratio_vs_gap": float(rough["sufficiency_ratio_vs_gap"]),
        "rough_kl_recovery_ratio": float(rough["sufficiency_ratio_vs_gap"]),
        "clean_with_detailed_corrupted": float(detailed["necessity_obj"]),
        "necessity_drop": float(detailed["necessity_drop"]),
        "necessity_ratio_vs_gap": float(detailed["necessity_ratio_vs_gap"]),
        "necessity_kl_drop_ratio": float(detailed["necessity_ratio_vs_gap"]),
        "clean_with_rough_corrupted": float(rough["necessity_obj"]),
        "rough_necessity_drop": float(rough["necessity_drop"]),
        "rough_necessity_ratio_vs_gap": float(rough["necessity_ratio_vs_gap"]),
        "rough_necessity_kl_drop_ratio": float(rough["necessity_ratio_vs_gap"]),
        "probe_head": probe_head,
        "probe_corrupt_with_head": float("nan"),
        "probe_clean_with_head_corrupt": float("nan"),
        "detailed_nodes": list(detailed["nodes"]),
        "detailed_edges": [list(edge) for edge in detailed["folded_pairs"]],
        "rough_nodes": list(rough["nodes"]),
        "rough_edges": [list(edge) for edge in rough["folded_pairs"]],
        "contrast_positions": list(pos_sets["contrast"]),
        "contrast_spans": list(pos_sets["contrast_spans"]),
        "contrast_token_details": contrast_token_details,
        "tool_call_open_positions": list(pos_sets["tool_call_open"]),
        "tool_call_close_positions": list(pos_sets["tool_call_close"]),
        "tool_call_tag_positions": list(pos_sets["tool_call_tags"]),
        "tools_block_positions": list(pos_sets["tools_block"]),
        "user_block_positions": list(pos_sets["user_block"]),
        "sample_catalog_record": sample.catalog_record(),
        "endpoint_distribution_summary": endpoint_summary,
        "top_node_scores": top_node_scores,
        "candidate_edge_sweep": candidate_rows,
        "candidate_edge_sweep_summary": summarize_candidates(candidate_rows),
        "artifacts": {
            "top_raw_edges_csv": str(out_dir / "top_raw_edges.csv"),
            "top_raw_edges_json": str(out_dir / "top_raw_edges.json"),
            "candidate_edge_sweep_json": str(out_dir / "candidate_edge_sweep.json"),
            "selected_raw_edges_csv": detailed["artifacts"]["raw_edges_csv"],
            "selected_raw_edges_json": detailed["artifacts"]["raw_edges_json"],
            "selected_folded_edges_csv": detailed["artifacts"]["folded_edges_csv"],
            "selected_folded_edges_json": detailed["artifacts"]["folded_edges_json"],
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


__all__ = ["normalize_edge_counts", "pick_selected_edge_count", "run_one_eap_ig_sample"]
