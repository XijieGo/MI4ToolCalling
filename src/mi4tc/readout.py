"""Small, model-agnostic readout calculations."""

from __future__ import annotations

from typing import Any, Iterable


def attention_mass(pattern: Any, positions: Iterable[int]) -> Any:
    """Sum attention over a declared token-position set for each batch/head."""

    import torch

    values = torch.as_tensor(pattern)
    if values.ndim < 3:
        raise ValueError(f"Expected attention tensor with batch/head/position axes, got {values.shape}")
    index = torch.as_tensor(list(positions), dtype=torch.long, device=values.device)
    if index.numel() == 0:
        return torch.zeros(values.shape[:-1], dtype=values.dtype, device=values.device)
    return values.index_select(-1, index).sum(dim=-1)


def normalized_shift(clean: Any, corrupt: Any) -> Any:
    """Clean-minus-corrupt shift, leaving normalization to the caller."""

    import torch

    clean_tensor = torch.as_tensor(clean).float()
    corrupt_tensor = torch.as_tensor(corrupt).float()
    if clean_tensor.shape != corrupt_tensor.shape:
        raise ValueError(f"Readout arrays differ: {clean_tensor.shape} vs {corrupt_tensor.shape}")
    return clean_tensor - corrupt_tensor
