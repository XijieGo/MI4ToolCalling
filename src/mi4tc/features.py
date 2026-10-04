"""Pure feature-level bookkeeping shared by formation audits.

The functions operate on already-collected feature activations and decoder
directions.  They intentionally do not load a Transcoder checkpoint; model and
checkpoint adapters belong to a study and remain optional external inputs.
"""

from __future__ import annotations

from typing import Any


def kappa(activation_delta: Any, decoder_projection: Any) -> Any:
    """Signed feature contribution: activation clean-minus-corrupt × write projection."""

    import torch

    delta = torch.as_tensor(activation_delta).float()
    projection = torch.as_tensor(decoder_projection).float()
    if delta.shape != projection.shape:
        raise ValueError(f"Activation and decoder arrays differ: {delta.shape} vs {projection.shape}")
    return delta * projection


def side_masses(activation_clean: Any, activation_corrupt: Any, decoder_projection: Any) -> dict[str, float]:
    """Return the paper's signed-side K masses and aligned S/E masses."""

    import torch

    clean = torch.as_tensor(activation_clean).float()
    corrupt = torch.as_tensor(activation_corrupt).float()
    projection = torch.as_tensor(decoder_projection).float()
    values = kappa(clean - corrupt, projection)
    corrupt_higher = corrupt > clean
    clean_higher = clean > corrupt
    return {
        "K_corrupt": float(values[corrupt_higher].abs().sum().item()),
        "K_clean": float(values[clean_higher].abs().sum().item()),
        "suppressor_aligned_mass": float(values[(corrupt_higher) & (projection < 0)].abs().sum().item()),
        "driver_aligned_mass": float(values[(clean_higher) & (projection > 0)].abs().sum().item()),
    }


def select_top_features(values: Any, *, k: int = 20) -> Any:
    import torch

    tensor = torch.as_tensor(values).float()
    if tensor.ndim != 1:
        raise ValueError(f"Expected one-dimensional feature scores, got {tensor.shape}")
    return torch.topk(tensor.abs(), min(int(k), tensor.numel())).indices
