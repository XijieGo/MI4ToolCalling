#!/usr/bin/env python3
"""
Distributional objective helpers for tool-call circuit experiments.

The core scoring function is a directional endpoint score:

    score(q; p_ref) = -KL(p_ref || q)

where `p_ref` is the clean endpoint next-token distribution and `q` is the
current next-token distribution at the first generation position.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Sequence

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class DistributionObjective:
    """Reference distribution for one decision endpoint."""

    endpoint_label: str
    reference_probs: torch.Tensor
    reference_log_probs: torch.Tensor
    temperature: float = 1.0
    masked_token_ids: tuple[int, ...] = ()
    top_token_id: int | None = None
    top_token_str: str | None = None


def _last_token_logits(logits: torch.Tensor) -> torch.Tensor:
    if logits.ndim != 3:
        raise ValueError(f"Expected logits with shape [batch, pos, vocab], got {tuple(logits.shape)}")
    return logits[:, -1, :]


def _masked_logits(
    logits: torch.Tensor,
    *,
    temperature: float,
    masked_token_ids: Sequence[int] = (),
) -> torch.Tensor:
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")
    out = logits / temperature
    if masked_token_ids:
        out = out.clone()
        out[..., list(masked_token_ids)] = float("-inf")
    return out


def logits_to_log_probs(
    logits: torch.Tensor,
    *,
    temperature: float = 1.0,
    masked_token_ids: Sequence[int] = (),
) -> torch.Tensor:
    last_logits = _last_token_logits(logits)
    masked = _masked_logits(last_logits, temperature=temperature, masked_token_ids=masked_token_ids)
    return F.log_softmax(masked, dim=-1)


def logits_to_probs(
    logits: torch.Tensor,
    *,
    temperature: float = 1.0,
    masked_token_ids: Sequence[int] = (),
) -> torch.Tensor:
    return logits_to_log_probs(
        logits,
        temperature=temperature,
        masked_token_ids=masked_token_ids,
    ).exp()


def build_distribution_objective(
    logits: torch.Tensor,
    *,
    endpoint_label: str,
    tokenizer=None,
    temperature: float = 1.0,
    masked_token_ids: Sequence[int] = (),
) -> DistributionObjective:
    ref_log_probs = logits_to_log_probs(
        logits,
        temperature=temperature,
        masked_token_ids=masked_token_ids,
    )[0].detach()
    ref_probs = ref_log_probs.exp()
    top_token_id = int(ref_probs.argmax().item())
    top_token_str = None
    if tokenizer is not None:
        try:
            top_token_str = tokenizer.decode([top_token_id])
        except Exception:
            top_token_str = None
    return DistributionObjective(
        endpoint_label=endpoint_label,
        reference_probs=ref_probs,
        reference_log_probs=ref_log_probs,
        temperature=float(temperature),
        masked_token_ids=tuple(int(x) for x in masked_token_ids),
        top_token_id=top_token_id,
        top_token_str=top_token_str,
    )


def kl_from_logits(logits: torch.Tensor, objective: DistributionObjective) -> torch.Tensor:
    log_q = logits_to_log_probs(
        logits,
        temperature=objective.temperature,
        masked_token_ids=objective.masked_token_ids,
    )
    ref_probs = objective.reference_probs.to(device=log_q.device, dtype=log_q.dtype).unsqueeze(0)
    ref_log_probs = objective.reference_log_probs.to(device=log_q.device, dtype=log_q.dtype).unsqueeze(0)
    kl = (ref_probs * (ref_log_probs - log_q)).sum(dim=-1)
    return kl


def objective_vector_from_logits(
    logits: torch.Tensor,
    target_token_or_objective: int | DistributionObjective,
    distractor_token: int | None = None,
) -> torch.Tensor:
    if isinstance(target_token_or_objective, DistributionObjective):
        return -kl_from_logits(logits, target_token_or_objective)

    target_token = int(target_token_or_objective)
    if distractor_token is None:
        raise ValueError("distractor_token is required for the legacy logit-gap objective.")
    return _last_token_logits(logits)[:, target_token] - _last_token_logits(logits)[:, int(distractor_token)]


def objective_from_logits(
    logits: torch.Tensor,
    target_token_or_objective: int | DistributionObjective,
    distractor_token: int | None = None,
) -> torch.Tensor:
    return objective_vector_from_logits(logits, target_token_or_objective, distractor_token).mean()


def endpoint_score_delta_from_logits(
    logits: torch.Tensor,
    tool_objective: DistributionObjective,
    no_tool_objective: DistributionObjective,
) -> torch.Tensor:
    return objective_from_logits(logits, tool_objective) - objective_from_logits(logits, no_tool_objective)


def js_divergence_between_logits(
    logits_a: torch.Tensor,
    logits_b: torch.Tensor,
    *,
    temperature: float = 1.0,
    masked_token_ids: Sequence[int] = (),
) -> float:
    probs_a = logits_to_probs(logits_a, temperature=temperature, masked_token_ids=masked_token_ids)[0]
    probs_b = logits_to_probs(logits_b, temperature=temperature, masked_token_ids=masked_token_ids)[0]
    mix = 0.5 * (probs_a + probs_b)
    log_a = probs_a.clamp_min(1e-12).log()
    log_b = probs_b.clamp_min(1e-12).log()
    log_mix = mix.clamp_min(1e-12).log()
    js = 0.5 * torch.sum(probs_a * (log_a - log_mix)) + 0.5 * torch.sum(probs_b * (log_b - log_mix))
    return float(js.item())


def topk_distribution_snapshot(
    logits: torch.Tensor,
    tokenizer,
    *,
    k: int = 10,
    temperature: float = 1.0,
    masked_token_ids: Sequence[int] = (),
) -> list[Dict[str, object]]:
    probs = logits_to_probs(logits, temperature=temperature, masked_token_ids=masked_token_ids)[0]
    k = min(max(1, int(k)), int(probs.numel()))
    values, indices = torch.topk(probs, k)
    rows: list[Dict[str, object]] = []
    for value, index in zip(values.tolist(), indices.tolist()):
        token_id = int(index)
        try:
            token_text = tokenizer.decode([token_id])
        except Exception:
            token_text = ""
        rows.append(
            {
                "token_id": token_id,
                "token": token_text,
                "prob": float(value),
            }
        )
    return rows


def objective_metadata(objective: DistributionObjective) -> Dict[str, object]:
    return {
        "mode": "negative_kl_to_endpoint",
        "endpoint_label": objective.endpoint_label,
        "temperature": float(objective.temperature),
        "masked_token_ids": list(objective.masked_token_ids),
        "top_token_id": objective.top_token_id,
        "top_token_str": objective.top_token_str,
    }


def summarize_endpoint_pair(
    *,
    tool_logits: torch.Tensor,
    no_tool_logits: torch.Tensor,
    tokenizer,
    temperature: float = 1.0,
    masked_token_ids: Sequence[int] = (),
    topk: int = 10,
    ) -> Dict[str, object]:
    tool_objective, no_tool_objective = build_bidirectional_endpoint_objectives(
        tool_logits,
        no_tool_logits,
        tokenizer=tokenizer,
        temperature=temperature,
        masked_token_ids=masked_token_ids,
    )
    return {
        "tool_objective": objective_metadata(tool_objective),
        "no_tool_objective": objective_metadata(no_tool_objective),
        "tool_topk": topk_distribution_snapshot(
            tool_logits,
            tokenizer,
            k=topk,
            temperature=temperature,
            masked_token_ids=masked_token_ids,
        ),
        "no_tool_topk": topk_distribution_snapshot(
            no_tool_logits,
            tokenizer,
            k=topk,
            temperature=temperature,
            masked_token_ids=masked_token_ids,
        ),
        "endpoint_js_divergence": js_divergence_between_logits(
            tool_logits,
            no_tool_logits,
            temperature=temperature,
            masked_token_ids=masked_token_ids,
        ),
    }


def build_bidirectional_endpoint_objectives(
    tool_logits: torch.Tensor,
    no_tool_logits: torch.Tensor,
    *,
    tokenizer=None,
    temperature: float = 1.0,
    masked_token_ids: Sequence[int] = (),
) -> tuple[DistributionObjective, DistributionObjective]:
    tool_objective = build_distribution_objective(
        tool_logits,
        endpoint_label="tool_call",
        tokenizer=tokenizer,
        temperature=temperature,
        masked_token_ids=masked_token_ids,
    )
    no_tool_objective = build_distribution_objective(
        no_tool_logits,
        endpoint_label="no_tool",
        tokenizer=tokenizer,
        temperature=temperature,
        masked_token_ids=masked_token_ids,
    )
    return tool_objective, no_tool_objective
