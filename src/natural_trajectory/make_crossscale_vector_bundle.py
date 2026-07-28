#!/usr/bin/env python3
"""Wrap a frozen cross-scale ``mean_diff`` vector for natural-trajectory runs.

The natural trajectory runners require a model-local vector plus a saved,
deterministic random direction for equal-norm controls.  The source artifacts
are the frozen code-completion vectors used by the paper's cross-scale
experiments; this utility does not fit a vector on Telecom data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-label", type=str, required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--hook-kind", choices=("pre", "post"), required=True)
    parser.add_argument("--random-seed", type=int, default=20260726)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    source_path = args.source_bundle.resolve()
    output_path = args.output.resolve()
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite {output_path}; pass --overwrite to replace it")

    source = torch.load(source_path, map_location="cpu", weights_only=False)
    if not isinstance(source, dict) or "mean_diff" not in source:
        raise ValueError(f"{source_path} does not contain a mean_diff vector")
    vector = torch.as_tensor(source["mean_diff"], dtype=torch.float32).reshape(-1).contiguous()
    if vector.numel() == 0 or not bool(torch.isfinite(vector).all()) or float(vector.norm().item()) <= 0.0:
        raise ValueError(f"Invalid mean_diff in {source_path}")
    stored_layer = source.get("patch_layer", source.get("layer"))
    if stored_layer is not None and int(stored_layer) != int(args.layer):
        raise ValueError(f"Source layer L{stored_layer} does not match requested L{args.layer}")

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(args.random_seed))
    random_direction = torch.randn(vector.shape, generator=generator, dtype=torch.float32)
    random_direction = (random_direction / random_direction.norm().clamp_min(1e-12)).contiguous()

    bundle: dict[str, Any] = {
        "format": "natural_trajectory_crossscale_bundle_v1",
        "model_label": args.model_label,
        "hook_kind": args.hook_kind,
        "patch_layer": int(args.layer) if args.hook_kind == "pre" else None,
        "layer": int(args.layer) if args.hook_kind == "post" else None,
        "mean_diff": vector,
        "random_direction": random_direction,
        "components": source.get("components"),
        "source_bundle": str(source_path),
        "source_bundle_sha256": sha256_file(source_path),
        "source_mean_diff_norm": float(vector.norm().item()),
        "source_sample_ids_train": source.get("sample_ids_train", source.get("sample_ids")),
        "random_seed": int(args.random_seed),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, output_path)
    print(
        json.dumps(
            {
                "output": str(output_path),
                "model_label": args.model_label,
                "layer": int(args.layer),
                "hook_kind": args.hook_kind,
                "hidden_size": int(vector.numel()),
                "mean_diff_norm": float(vector.norm().item()),
                "random_seed": int(args.random_seed),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
