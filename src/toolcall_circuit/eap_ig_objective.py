#!/usr/bin/env python3
from __future__ import annotations

import math
from typing import Dict, Iterable, List, Sequence

import torch

from toolcall_circuit.objective import (
    DistributionObjective,
    build_distribution_objective,
    logits_to_log_probs,
    objective_from_logits,
    summarize_endpoint_pair,
)


def negative_kl_to_clean_endpoint(
    logits: torch.Tensor,
    clean_logits: torch.Tensor | None,
    input_lengths: torch.Tensor,
    labels,
) -> torch.Tensor:
    del input_lengths, labels
    if clean_logits is None:
        raise ValueError("clean_logits is required for the EAP-IG endpoint metric")
    ref_log_probs = logits_to_log_probs(clean_logits)
    ref_probs = ref_log_probs.exp()
    log_q = logits_to_log_probs(logits)
    kl = (ref_probs * (ref_log_probs - log_q)).sum(dim=-1)
    return -kl


def metric_name(direction: str) -> str:
    if direction == "forward":
        return "negative_kl_to_tool_endpoint"
    if direction == "reverse":
        return "negative_kl_to_no_tool_endpoint"
    raise ValueError(f"Unknown direction: {direction}")


def build_directional_objective(
    *,
    clean_logits: torch.Tensor,
    tokenizer,
    endpoint_label: str,
    temperature: float = 1.0,
    masked_token_ids: Sequence[int] = (),
) -> DistributionObjective:
    return build_distribution_objective(
        clean_logits,
        endpoint_label=endpoint_label,
        tokenizer=tokenizer,
        temperature=temperature,
        masked_token_ids=masked_token_ids,
    )


def candidate_summary_row(
    *,
    edge_count: int,
    sufficiency_obj: float,
    sufficiency_ratio: float,
    necessity_obj: float,
    necessity_drop: float,
    necessity_ratio: float,
    node_count: int,
    folded_edge_count: int,
) -> Dict[str, object]:
    return {
        "edge_count": int(edge_count),
        "sufficiency_obj": float(sufficiency_obj),
        "sufficiency_ratio_vs_gap": float(sufficiency_ratio),
        "necessity_obj": float(necessity_obj),
        "necessity_drop": float(necessity_drop),
        "necessity_ratio_vs_gap": float(necessity_ratio),
        "node_count": int(node_count),
        "folded_edge_count": int(folded_edge_count),
    }


def summarize_candidates(rows: Sequence[Dict[str, object]]) -> Dict[str, object]:
    if not rows:
        return {"candidate_count": 0}
    suff = [float(r["sufficiency_ratio_vs_gap"]) for r in rows if math.isfinite(float(r["sufficiency_ratio_vs_gap"]))]
    nec = [float(r["necessity_ratio_vs_gap"]) for r in rows if math.isfinite(float(r["necessity_ratio_vs_gap"]))]
    return {
        "candidate_count": len(rows),
        "sufficiency_ratio_min": min(suff) if suff else float("nan"),
        "sufficiency_ratio_max": max(suff) if suff else float("nan"),
        "necessity_ratio_min": min(nec) if nec else float("nan"),
        "necessity_ratio_max": max(nec) if nec else float("nan"),
    }


def top_token_strings(tokenizer, token_ids: Iterable[int]) -> List[str]:
    out: List[str] = []
    for token_id in token_ids:
        try:
            out.append(tokenizer.decode([int(token_id)]))
        except Exception:
            out.append("")
    return out


def endpoint_pair_summary(
    *,
    direction: str,
    clean_logits: torch.Tensor,
    corrupt_logits: torch.Tensor,
    tokenizer,
    temperature: float = 1.0,
    masked_token_ids: Sequence[int] = (),
) -> Dict[str, object]:
    if direction == "forward":
        return summarize_endpoint_pair(
            tool_logits=clean_logits,
            no_tool_logits=corrupt_logits,
            tokenizer=tokenizer,
            temperature=temperature,
            masked_token_ids=masked_token_ids,
            topk=12,
        )
    if direction == "reverse":
        return summarize_endpoint_pair(
            tool_logits=corrupt_logits,
            no_tool_logits=clean_logits,
            tokenizer=tokenizer,
            temperature=temperature,
            masked_token_ids=masked_token_ids,
            topk=12,
        )
    raise ValueError(f"Unknown direction: {direction}")


__all__ = [
    "build_directional_objective",
    "candidate_summary_row",
    "endpoint_pair_summary",
    "metric_name",
    "negative_kl_to_clean_endpoint",
    "objective_from_logits",
    "summarize_candidates",
]
