#!/usr/bin/env python3
"""Copy a causal-vector bundle after matching its L2 norm to a reference bundle.

The source direction is retained exactly; only ``mean_diff`` is rescaled.  The
output is intentionally a new bundle so the source and target-native vectors
remain immutable provenance artifacts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rescale one vector bundle to another bundle's L2 norm.")
    parser.add_argument("--source-bundle", type=Path, required=True)
    parser.add_argument("--reference-bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def vector_from_bundle(bundle: dict[str, Any], label: str) -> torch.Tensor:
    vector = bundle.get("mean_diff")
    if not isinstance(vector, torch.Tensor):
        vector = torch.as_tensor(vector)
    vector = vector.detach().cpu()
    if vector.ndim != 1 or vector.numel() == 0:
        raise ValueError(f"{label} mean_diff must be a nonempty rank-1 tensor")
    if not bool(torch.isfinite(vector).all()):
        raise ValueError(f"{label} mean_diff contains non-finite values")
    return vector


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")

    source_path = args.source_bundle.resolve()
    reference_path = args.reference_bundle.resolve()
    source = torch.load(source_path, map_location="cpu", weights_only=False)
    reference = torch.load(reference_path, map_location="cpu", weights_only=False)
    if not isinstance(source, dict) or not isinstance(reference, dict):
        raise TypeError("Both bundles must be dictionaries")

    source_vector = vector_from_bundle(source, "source")
    reference_vector = vector_from_bundle(reference, "reference")
    if source_vector.shape != reference_vector.shape:
        raise ValueError(f"Vector shape mismatch: {tuple(source_vector.shape)} vs {tuple(reference_vector.shape)}")
    source_layer = int(source.get("patch_layer", source.get("layer", -1)))
    reference_layer = int(reference.get("patch_layer", reference.get("layer", -1)))
    if source_layer != reference_layer:
        raise ValueError(f"Layer mismatch: source L{source_layer}, reference L{reference_layer}")

    source_norm = float(source_vector.float().norm().item())
    reference_norm = float(reference_vector.float().norm().item())
    if source_norm <= 0.0:
        raise ValueError("Cannot align a zero-norm source vector")
    scale = reference_norm / source_norm
    aligned_vector = source_vector * scale

    aligned = dict(source)
    aligned["mean_diff"] = aligned_vector
    aligned["norm_alignment"] = {
        "method": "source_direction_target_l2_norm",
        "source_bundle": str(source_path),
        "reference_bundle": str(reference_path),
        "source_l2_norm": source_norm,
        "reference_l2_norm": reference_norm,
        "applied_scale": scale,
        "result_l2_norm": float(aligned_vector.float().norm().item()),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(aligned, output)
    print(json.dumps(aligned["norm_alignment"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
