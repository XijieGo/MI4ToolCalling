#!/usr/bin/env python3
"""Run a declared native-pair localization audit for Granite or Mistral.

The model-native collection is already behavior-screened.  This runner never
re-screens, replaces, or drops a prompt because of a fresh baseline decision.
It fits a clean-minus-corrupt direction on the frozen 200-pair train split and
evaluates interventions on the frozen 300-pair held-out split.

``prediction`` alignment uses all selected pairs and operates only at the
prediction position.  ``strict`` additionally requires equal token lengths
and exactly one changed token before it runs the verb-position patch.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from mi4tc.directions import mean_difference  # noqa: E402
from mi4tc.io import write_json  # noqa: E402
from mi4tc.metrics import tool_call_stats  # noqa: E402
from mi4tc.model import CausalLMAdapter, load_causal_lm, load_mistral3  # noqa: E402
from mi4tc.pairs import (  # noqa: E402
    ModelNativePairs,
    NativePair,
    changed_token_positions_from_ids,
    load_model_native_pairs,
    native_pair_token_ids,
)


@dataclass(frozen=True)
class NativeModelSpec:
    model_key: str
    dataset_root: Path
    marker: str
    marker_id: int
    prompt_format: str
    loader: Callable[..., CausalLMAdapter]


NATIVE_MODELS = {
    "granite_3p3_8b": NativeModelSpec(
        model_key="granite_3p3_8b",
        dataset_root=REPO_ROOT / "datasets/granite_3p3_8b/pair",
        marker="<|tool_call|>",
        marker_id=49154,
        prompt_format="native_text",
        loader=load_causal_lm,
    ),
    "mistral_3p2_24b": NativeModelSpec(
        model_key="mistral_3p2_24b",
        dataset_root=REPO_ROOT / "datasets/mistral_3p2_24b/pair",
        marker="[TOOL_CALLS]",
        marker_id=9,
        prompt_format="mistral_native_input_ids",
        loader=load_mistral3,
    ),
}


@dataclass(frozen=True)
class PreparedPair:
    pair: NativePair
    clean_ids: tuple[int, ...]
    corrupt_ids: tuple[int, ...]
    changed_token_position: int | None


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


def validate_model_contract(collection: ModelNativePairs, spec: NativeModelSpec) -> None:
    observed = {
        "model_key": collection.model_key,
        "marker": collection.tool_call_marker,
        "marker_id": collection.tool_call_token_id,
        "prompt_format": collection.prompt_format,
    }
    expected = {
        "model_key": spec.model_key,
        "marker": spec.marker,
        "marker_id": spec.marker_id,
        "prompt_format": spec.prompt_format,
    }
    if observed != expected:
        raise ValueError(f"Model-native collection does not match the declared adapter: {observed}")


def prepare_pairs(
    pairs: list[NativePair],
    *,
    tokenizer: Any | None,
    alignment_policy: str,
) -> tuple[list[PreparedPair], dict[str, int]]:
    """Materialize prompt IDs and apply the declared, pre-model alignment policy."""

    report = {
        "input_pairs": len(pairs),
        "used_pairs": 0,
        "excluded_length_mismatch": 0,
        "excluded_non_single_token_edit": 0,
    }
    prepared: list[PreparedPair] = []
    for pair in pairs:
        clean_ids, corrupt_ids = native_pair_token_ids(pair, tokenizer=tokenizer)
        changed_position: int | None = None
        if alignment_policy == "strict":
            if len(clean_ids) != len(corrupt_ids):
                report["excluded_length_mismatch"] += 1
                continue
            changed = changed_token_positions_from_ids(clean_ids, corrupt_ids)
            if len(changed) != 1:
                report["excluded_non_single_token_edit"] += 1
                continue
            changed_position = changed[0]
        prepared.append(
            PreparedPair(
                pair=pair,
                clean_ids=clean_ids,
                corrupt_ids=corrupt_ids,
                changed_token_position=changed_position,
            )
        )
    report["used_pairs"] = len(prepared)
    return prepared, report


def check_granite_marker(adapter: CausalLMAdapter, spec: NativeModelSpec) -> None:
    """Confirm that Granite's rendered text still resolves to its declared ID."""

    if adapter.tokenizer is None:
        raise RuntimeError("Granite adapter did not expose a tokenizer")
    marker_ids = adapter.tokenizer.encode(spec.marker, add_special_tokens=False)
    if len(marker_ids) != 1 or int(marker_ids[0]) != spec.marker_id:
        raise ValueError(
            f"Granite marker {spec.marker!r} must encode as [{spec.marker_id}], got {list(marker_ids)}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-key", choices=tuple(NATIVE_MODELS), required=True)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--layers", default="", help="Comma/range list, e.g. 20-26")
    parser.add_argument("--direction-layer", type=int)
    parser.add_argument("--alignment-policy", choices=("prediction", "strict"), default="prediction")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--max-train-pairs", type=int, default=0)
    parser.add_argument("--max-heldout-pairs", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate the frozen 200/300 inputs and print the run contract without loading a model.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    spec = NATIVE_MODELS[args.model_key]
    collection = load_model_native_pairs(spec.dataset_root)
    validate_model_contract(collection, spec)
    source_train = collection.split_pairs("train", max_pairs=args.max_train_pairs)
    source_heldout = collection.split_pairs("heldout", max_pairs=args.max_heldout_pairs)

    if args.dry_run:
        print(
            json.dumps(
                {
                    "run_kind": "native_pair_localization",
                    "model_key": spec.model_key,
                    "dataset_root": str(spec.dataset_root.relative_to(REPO_ROOT)),
                    "tool_call_marker": collection.tool_call_marker,
                    "tool_call_token_id": collection.tool_call_token_id,
                    "prompt_format": collection.prompt_format,
                    "frozen_train_pairs": len(source_train),
                    "frozen_heldout_pairs": len(source_heldout),
                    "alignment_policy": args.alignment_policy,
                    "alignment_note": (
                        "prediction uses every frozen pair"
                        if args.alignment_policy == "prediction"
                        else "strict token alignment is resolved after loading the model-native tokenizer or IDs"
                    ),
                    "selection": "frozen behavior-screened pairs; no fresh screening or baseline-based exclusion",
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    if args.model_path is None:
        raise ValueError("--model-path is required unless --dry-run is used")
    if args.output_root is None:
        raise ValueError("--output-root is required unless --dry-run is used")
    if not args.layers:
        raise ValueError("--layers is required unless --dry-run is used")
    if args.direction_layer is None:
        raise ValueError("--direction-layer is required unless --dry-run is used")
    if args.max_train_pairs < 0 or args.max_heldout_pairs < 0:
        raise ValueError("max pair counts must be non-negative")

    layers = parse_layers(args.layers)
    if args.direction_layer not in layers:
        layers.append(args.direction_layer)
        layers.sort()
    adapter = spec.loader(args.model_path, device=args.device, dtype=args.dtype)
    if max(layers) >= adapter.n_layers:
        raise ValueError(f"Requested layer {max(layers)} but model has {adapter.n_layers} layers")
    if spec.model_key == "granite_3p3_8b":
        check_granite_marker(adapter, spec)

    train_pairs, train_alignment = prepare_pairs(
        source_train,
        tokenizer=adapter.tokenizer,
        alignment_policy=args.alignment_policy,
    )
    heldout_pairs, heldout_alignment = prepare_pairs(
        source_heldout,
        tokenizer=adapter.tokenizer,
        alignment_policy=args.alignment_policy,
    )
    if not train_pairs or not heldout_pairs:
        raise ValueError(f"No usable pairs under alignment policy {args.alignment_policy!r}")

    clean_train_states: list[Any] = []
    corrupt_train_states: list[Any] = []
    for index, prepared in enumerate(train_pairs, start=1):
        clean_tokens = adapter.input_ids(prepared.clean_ids)
        corrupt_tokens = adapter.input_ids(prepared.corrupt_ids)
        clean_state = adapter.capture(clean_tokens, args.direction_layer)
        corrupt_state = adapter.capture(corrupt_tokens, args.direction_layer)
        clean_train_states.append(clean_state[0, -1, :])
        corrupt_train_states.append(corrupt_state[0, -1, :])
        if index % 25 == 0 or index == len(train_pairs):
            print(f"fit direction: {index}/{len(train_pairs)}", flush=True)

    import torch

    direction = mean_difference(torch.stack(clean_train_states), torch.stack(corrupt_train_states))
    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "direction": direction.cpu(),
            "estimator": "mean(clean - corrupt) at the final prediction position",
            "fit_split": f"datasets/{spec.model_key}/pair/train",
            "model_key": spec.model_key,
            "layer": args.direction_layer,
            "alignment_policy": args.alignment_policy,
            "train_pairs": len(train_pairs),
            "hook_convention": adapter.hook_convention,
        },
        output_root / "mean_direction.pt",
    )

    rows: list[dict[str, Any]] = []
    aggregate: dict[tuple[int, str], dict[str, float]] = {}

    def record(
        layer: int,
        intervention: str,
        prepared: PreparedPair,
        stats: dict[str, Any],
        baseline: dict[str, Any],
        expected_transition: str,
    ) -> None:
        if expected_transition == "gain":
            transition = (not baseline["tool_top1"]) and stats["tool_top1"]
        elif expected_transition == "loss":
            transition = baseline["tool_top1"] and not stats["tool_top1"]
        else:
            raise ValueError(f"Unknown expected transition: {expected_transition}")
        key = (layer, intervention)
        bucket = aggregate.setdefault(
            key,
            {"n": 0.0, "top1": 0.0, "transition": 0.0, "logit": 0.0, "probability": 0.0},
        )
        bucket["n"] += 1
        bucket["top1"] += float(bool(stats["tool_top1"]))
        bucket["transition"] += float(transition)
        bucket["logit"] += float(stats["tool_logit"])
        bucket["probability"] += float(stats["tool_probability"])
        rows.append(
            {
                "sample_id": prepared.pair.sample_id,
                "split": prepared.pair.split,
                "model_key": spec.model_key,
                "layer": layer,
                "intervention": intervention,
                "alignment_policy": args.alignment_policy,
                "clean_token_count": len(prepared.clean_ids),
                "corrupt_token_count": len(prepared.corrupt_ids),
                "changed_token_position": prepared.changed_token_position,
                "tool_top1": int(bool(stats["tool_top1"])),
                "tool_rank": int(stats["tool_rank"]),
                "tool_probability": float(stats["tool_probability"]),
                "tool_logit": float(stats["tool_logit"]),
                "baseline_tool_top1": int(bool(baseline["tool_top1"])),
                "expected_transition": expected_transition,
                "strict_transition": int(transition),
            }
        )

    for index, prepared in enumerate(heldout_pairs, start=1):
        clean_tokens = adapter.input_ids(prepared.clean_ids)
        corrupt_tokens = adapter.input_ids(prepared.corrupt_ids)
        clean_baseline = one_stats(adapter.logits(clean_tokens), spec.marker_id)
        corrupt_baseline = one_stats(adapter.logits(corrupt_tokens), spec.marker_id)
        for layer in layers:
            clean_state = adapter.capture(clean_tokens, layer)
            corrupt_state = adapter.capture(corrupt_tokens, layer)
            prediction_stats = one_stats(
                adapter.patched_logits(
                    corrupt_tokens,
                    layer=layer,
                    position=corrupt_state.shape[1] - 1,
                    replacement=clean_state[0, -1, :],
                ),
                spec.marker_id,
            )
            record(layer, "full_patch_prediction", prepared, prediction_stats, corrupt_baseline, "gain")
            if prepared.changed_token_position is not None:
                verb_stats = one_stats(
                    adapter.patched_logits(
                        corrupt_tokens,
                        layer=layer,
                        position=prepared.changed_token_position,
                        replacement=clean_state[0, prepared.changed_token_position, :],
                    ),
                    spec.marker_id,
                )
                record(layer, "full_patch_verb", prepared, verb_stats, corrupt_baseline, "gain")
            if layer == args.direction_layer:
                add_stats = one_stats(
                    adapter.patched_logits(
                        corrupt_tokens,
                        layer=layer,
                        position=corrupt_state.shape[1] - 1,
                        replacement=corrupt_state[0, -1, :] + direction,
                    ),
                    spec.marker_id,
                )
                record(layer, "add_mean_direction", prepared, add_stats, corrupt_baseline, "gain")
                remove_stats = one_stats(
                    adapter.patched_logits(
                        clean_tokens,
                        layer=layer,
                        position=clean_state.shape[1] - 1,
                        replacement=clean_state[0, -1, :] - direction,
                    ),
                    spec.marker_id,
                )
                record(layer, "remove_mean_direction", prepared, remove_stats, clean_baseline, "loss")
        if index % 25 == 0 or index == len(heldout_pairs):
            print(f"evaluate heldout: {index}/{len(heldout_pairs)}", flush=True)

    summary_rows = []
    for (layer, intervention), values in sorted(aggregate.items()):
        n = max(values["n"], 1.0)
        summary_rows.append(
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
            "run_kind": "native_pair_localization",
            "model_key": spec.model_key,
            "model_path": str(args.model_path),
            "dataset_root": str(spec.dataset_root.relative_to(REPO_ROOT)),
            "tool_call_marker": collection.tool_call_marker,
            "tool_call_token_id": collection.tool_call_token_id,
            "prompt_format": collection.prompt_format,
            "input_handling": (
                "exact stored native input_ids"
                if collection.prompt_format == "mistral_native_input_ids"
                else "native rendered text encoded with add_special_tokens=False"
            ),
            "selection": "frozen behavior-screened pairs; no fresh screening or baseline-based exclusion",
            "train_pairs": len(train_pairs),
            "heldout_pairs": len(heldout_pairs),
            "source_train_pairs": len(source_train),
            "source_heldout_pairs": len(source_heldout),
            "train_alignment": train_alignment,
            "heldout_alignment": heldout_alignment,
            "alignment_policy": args.alignment_policy,
            "layers": layers,
            "direction_layer": args.direction_layer,
            "hook_convention": adapter.hook_convention,
            "model_family": adapter.model_family,
            "direction_estimator": "mean(clean - corrupt) at the final prediction position on frozen train pairs",
            "summary": summary_rows,
        },
    )
    del adapter
    gc.collect()
    print(f"Wrote native localization results to {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
