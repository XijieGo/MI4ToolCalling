"""Metrics for the first-token call/no-call target."""

from __future__ import annotations

from typing import Any


def last_logits(logits: Any) -> Any:
    return logits[:, -1, :] if getattr(logits, "ndim", 0) == 3 else logits


def tool_call_stats(logits: Any, tool_token_id: int) -> dict[str, Any]:
    import torch

    values = last_logits(logits).float()
    token = int(tool_token_id)
    target = values[:, token]
    top1 = values.argmax(dim=-1)
    rank = (values > target.unsqueeze(-1)).sum(dim=-1) + 1
    probability = torch.softmax(values, dim=-1)[:, token]
    return {
        "tool_logit": target.detach().cpu(),
        "tool_probability": probability.detach().cpu(),
        "tool_rank": rank.detach().cpu(),
        "tool_top1": (top1 == token).detach().cpu(),
    }


def rate(values: Any) -> float:
    import torch

    tensor = torch.as_tensor(values)
    return float(tensor.float().mean().item()) if tensor.numel() else float("nan")


def strict_flip_rate(before: Any, after: Any, tool_token_id: int) -> float:
    import torch

    before_top1 = torch.as_tensor(before) == int(tool_token_id)
    after_top1 = torch.as_tensor(after) == int(tool_token_id)
    eligible = ~before_top1
    return float((after_top1 & eligible).float().sum().item() / max(int(eligible.sum().item()), 1))
