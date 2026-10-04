"""Explicit direction estimators used by the studies."""

from __future__ import annotations

from typing import Any


def mean_difference(clean: Any, corrupt: Any) -> Any:
    """Return the unnormalised mean clean-minus-corrupt direction."""

    import torch

    clean_tensor = torch.as_tensor(clean).float()
    corrupt_tensor = torch.as_tensor(corrupt).float()
    if clean_tensor.shape != corrupt_tensor.shape:
        raise ValueError(f"Direction inputs have different shapes: {clean_tensor.shape} vs {corrupt_tensor.shape}")
    if clean_tensor.ndim < 2:
        raise ValueError("Direction inputs need a sample dimension and a feature dimension")
    return (clean_tensor - corrupt_tensor).mean(dim=0)


def unit(direction: Any, *, eps: float = 1e-12) -> Any:
    import torch

    vector = torch.as_tensor(direction).float()
    norm = torch.linalg.vector_norm(vector)
    if float(norm) <= eps:
        raise ValueError("Cannot normalise a near-zero direction")
    return vector / norm


def add_at_position(state: Any, direction: Any, position: int, *, alpha: float = 1.0) -> Any:
    import torch

    output = torch.as_tensor(state).clone()
    delta = torch.as_tensor(direction, device=output.device, dtype=output.dtype)
    if output.ndim != 3 or delta.ndim != 1 or output.shape[-1] != delta.shape[0]:
        raise ValueError(f"Expected state [batch, sequence, hidden] and direction [hidden], got {output.shape}, {delta.shape}")
    output[:, position, :] += float(alpha) * delta
    return output


def subtract_at_position(state: Any, direction: Any, position: int, *, alpha: float = 1.0) -> Any:
    return add_at_position(state, -direction, position, alpha=alpha)
