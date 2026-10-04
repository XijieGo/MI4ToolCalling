#!/usr/bin/env python3
"""Minimal, explicit HF implementation of the localization/vector study.

This is deliberately a small study runner rather than a compatibility wrapper
around the old phase scripts.  It uses a declared Hugging Face decoder-block
input hook, estimates a mean clean-minus-corrupt direction on train, and
evaluates held-out full-state patches plus direction addition/removal.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from mi4tc.directions import mean_difference  # noqa: E402
from mi4tc.io import write_json  # noqa: E402
from mi4tc.metrics import tool_call_stats  # noqa: E402
from mi4tc.model import load_causal_lm  # noqa: E402
from mi4tc.pairs import changed_token_positions, load_pairs  # noqa: E402


def parse_layers(raw: str) -> list[int]:
    layers: set[int] = set()
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = (int(item) for item in part.split("-", 1))
            if start < 0 or end < start:
                raise ValueError(f"Invalid layer range: {part!r}")
            layers.update(range(start, end + 1))
        else:
            layer = int(part)
            if layer < 0:
                raise ValueError(f"Layer must be non-negative: {layer}")
            layers.add(layer)
    if not layers:
        raise ValueError("At least one layer is required")
    return sorted(layers)


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def scalar(value: Any) -> float | int | bool:
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    return float(value)


def one_stats(logits: Any, token_id: int) -> dict[str, Any]:
    stats = tool_call_stats(logits, token_id)
    return {key: scalar(value[0]) for key, value in stats.items()}


def resolve_tool_token_id(tokenizer: Any, marker: str) -> int:
    ids = tokenizer.encode(marker, add_special_tokens=False)
    if len(ids) != 1:
        raise ValueError(f"{marker!r} is not one tokenizer token: {ids}")
    return int(ids[0])


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the portable localization/vector study.")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--layers", default="24", help="Comma/range list, e.g. 20-25")
    parser.add_argument("--direction-layer", type=int, default=24)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-train-pairs", type=int, default=0)
    parser.add_argument("--max-test-pairs", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--tool-call-token", default="<tool_call>")
    args = parser.parse_args()

    layers = parse_layers(args.layers)
    if args.direction_layer not in layers:
        layers.append(args.direction_layer)
        layers.sort()
    train_root = REPO_ROOT / "datasets/qwen3_8b/controlled"
    train_pairs = load_pairs(train_root, "train", max_pairs=args.max_train_pairs)
    test_pairs = load_pairs(train_root, "test", max_pairs=args.max_test_pairs)
    adapter = load_causal_lm(args.model_path, device=args.device, dtype=args.dtype)
    if max(layers) >= adapter.n_layers:
        raise ValueError(f"Requested layer {max(layers)} but model has {adapter.n_layers} layers")
    tool_token_id = resolve_tool_token_id(adapter.tokenizer, args.tool_call_token)

    def prepare(pair):
        clean_tokens = adapter.encode(pair.clean_text)
        corrupt_tokens = adapter.encode(pair.corrupt_text)
        if clean_tokens.shape != corrupt_tokens.shape:
            raise ValueError(f"{pair.sample_id}: clean/corrupt token lengths differ")
        positions = changed_token_positions(adapter.tokenizer, pair)
        if len(positions) != 1:
            raise ValueError(f"{pair.sample_id}: expected one changed token, found {positions}")
        return clean_tokens, corrupt_tokens, positions[0]

    # Fit the direction once on the declared training split.
    clean_train_states: list[Any] = []
    corrupt_train_states: list[Any] = []
    for index, pair in enumerate(train_pairs, start=1):
        clean_tokens, corrupt_tokens, _ = prepare(pair)
        clean_state = adapter.capture(clean_tokens, args.direction_layer)
        corrupt_state = adapter.capture(corrupt_tokens, args.direction_layer)
        clean_train_states.append(clean_state[0, -1, :])
        corrupt_train_states.append(corrupt_state[0, -1, :])
        if index % 100 == 0:
            print(f"fit direction: {index}/{len(train_pairs)}", flush=True)

    import torch

    direction = mean_difference(torch.stack(clean_train_states), torch.stack(corrupt_train_states))
    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "direction": direction.cpu(),
            "estimator": "mean(clean - corrupt)",
            "fit_split": "datasets/qwen3_8b/controlled/train",
            "layer": args.direction_layer,
            "hook_convention": adapter.hook_convention,
        },
        output_root / "mean_direction.pt",
    )

    rows: list[dict[str, Any]] = []
    aggregate: dict[tuple[int, str], dict[str, float]] = {}

    def record(
        layer: int,
        intervention: str,
        pair,
        stats: dict[str, Any],
        baseline: dict[str, Any],
        expected_transition: str,
    ) -> None:
        if expected_transition == "gain":
            transition = (not baseline["tool_top1"]) and stats["tool_top1"]
        elif expected_transition == "loss":
            transition = baseline["tool_top1"] and not stats["tool_top1"]
        else:
            raise ValueError(f"unknown expected transition: {expected_transition}")
        key = (layer, intervention)
        bucket = aggregate.setdefault(key, {"n": 0.0, "top1": 0.0, "transition": 0.0, "logit": 0.0, "probability": 0.0})
        bucket["n"] += 1
        bucket["top1"] += float(bool(stats["tool_top1"]))
        bucket["transition"] += float(transition)
        bucket["logit"] += float(stats["tool_logit"])
        bucket["probability"] += float(stats["tool_probability"])
        rows.append(
            {
                "sample_id": pair.sample_id,
                "layer": layer,
                "intervention": intervention,
                "tool_top1": int(bool(stats["tool_top1"])),
                "tool_rank": int(stats["tool_rank"]),
                "tool_probability": float(stats["tool_probability"]),
                "tool_logit": float(stats["tool_logit"]),
                "baseline_tool_top1": int(bool(baseline["tool_top1"])),
                "expected_transition": expected_transition,
                "strict_transition": int(transition),
            }
        )

    for index, pair in enumerate(test_pairs, start=1):
        clean_tokens, corrupt_tokens, verb_position = prepare(pair)
        corrupt_baseline = one_stats(adapter.logits(corrupt_tokens), tool_token_id)
        for layer in layers:
            clean_state = adapter.capture(clean_tokens, layer)
            corrupt_state = adapter.capture(corrupt_tokens, layer)
            for name, position, replacement in (
                ("full_patch_verb", verb_position, clean_state[0, verb_position, :]),
                ("full_patch_prediction", clean_state.shape[1] - 1, clean_state[0, -1, :]),
            ):
                stats = one_stats(adapter.patched_logits(corrupt_tokens, layer=layer, position=position, replacement=replacement), tool_token_id)
                record(layer, name, pair, stats, corrupt_baseline, "gain")
            if layer == args.direction_layer:
                stats = one_stats(
                    adapter.patched_logits(
                        corrupt_tokens,
                        layer=layer,
                        position=corrupt_state.shape[1] - 1,
                        replacement=corrupt_state[0, -1, :] + direction,
                    ),
                    tool_token_id,
                )
                record(layer, "add_mean_direction", pair, stats, corrupt_baseline, "gain")
                clean_baseline = one_stats(adapter.logits(clean_tokens), tool_token_id)
                stats = one_stats(
                    adapter.patched_logits(
                        clean_tokens,
                        layer=layer,
                        position=clean_state.shape[1] - 1,
                        replacement=clean_state[0, -1, :] - direction,
                    ),
                    tool_token_id,
                )
                record(layer, "remove_mean_direction", pair, stats, clean_baseline, "loss")
        if index % 25 == 0:
            print(f"evaluate heldout: {index}/{len(test_pairs)}", flush=True)

    summary = []
    for (layer, intervention), values in sorted(aggregate.items()):
        n = max(values["n"], 1.0)
        summary.append(
            {
                "layer": layer,
                "intervention": intervention,
                "n": int(values["n"]),
                "tool_top1_rate": values["top1"] / n,
                "strict_transition_rate": values["transition"] / n,
                "mean_tool_logit": values["logit"] / n,
                "mean_tool_probability": values["probability"] / n,
            }
        )
    write_rows(output_root / "per_sample.csv", rows)
    write_json(
        output_root / "summary.json",
        {
            "model_path": str(args.model_path),
            "train_pairs": len(train_pairs),
            "test_pairs": len(test_pairs),
            "layers": layers,
            "direction_layer": args.direction_layer,
            "tool_call_token": args.tool_call_token,
            "hook_convention": adapter.hook_convention,
            "direction_estimator": "mean(clean - corrupt) at the prediction position on datasets/qwen3_8b/controlled/train",
            "summary": summary,
        },
    )
    del adapter
    gc.collect()
    print(f"Wrote pair localization results to {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
