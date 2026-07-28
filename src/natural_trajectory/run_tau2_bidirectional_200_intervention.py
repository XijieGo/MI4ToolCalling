#!/usr/bin/env python3
"""Causal first-token interventions on the new τ² 200+200 collections.

The collections were constructed independently for each target model by the
unmodified native first-token decision.  This runner deliberately does not
screen, tune, or estimate anything on τ².  It only:

1. loads a frozen model-local Coding vector and its Coding-selected layer;
2. re-renders the selected natural τ² histories with the target model's
   native template; and
3. measures first-token switches under signed Code and equal-norm Random
   directions.

The two arms are kept separate:

* ``suppression``: naturally tool-positive prefixes, with ``-alpha * v``;
* ``induction``: naturally non-tool prefixes, with ``+alpha * v``.

Each condition is checkpointed independently.  ``--resume`` can therefore
continue after an interrupted GPU job without recomputing completed arms or
conditions.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import gc
import hashlib
import json
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import torch


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import build_tau2_bidirectional_200 as collection  # noqa: E402


SEED = 20260726
TELECOM_COLLECTION_ROOT = (
    PROJECT_ROOT
    / "datasets"
    / "external"
    / "tau2_telecom_qwen35_9b"
    / "collections"
    / "bidirectional_200_per_model_20260726"
)
RETAIL_COLLECTION_ROOT = (
    PROJECT_ROOT
    / "datasets"
    / "external"
    / "tau2_retail_qwen35_9b"
    / "collections"
    / "bidirectional_200_per_model_20260726"
)
TELECOM_CANONICAL_COLLECTION_ROOT = TELECOM_COLLECTION_ROOT.with_name(
    TELECOM_COLLECTION_ROOT.name + "_canonical_batch1"
)
RETAIL_CANONICAL_COLLECTION_ROOT = RETAIL_COLLECTION_ROOT.with_name(
    RETAIL_COLLECTION_ROOT.name + "_canonical_batch1"
)
TELECOM_RAW_ROOT = PROJECT_ROOT / "datasets" / "external" / "tau2_telecom_qwen35_9b" / "raw"
RETAIL_RAW_ROOT = PROJECT_ROOT / "datasets" / "external" / "tau2_retail_qwen35_9b" / "raw"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "results" / "natural_trajectory" / "tau2_bidirectional_200_interventions_20260727"
FROZEN_VECTOR_ROOT = (
    PROJECT_ROOT
    / "artifacts"
    / "05_tau2_bench_200"
    / "coding_direction_inputs"
    / "cross_family_tau2_20260726"
)

# Every path below was learned/frozen on Coding only.  The Qwen3-8B bundle
# predates the common post-hook format and is intentionally a pre-hook bundle;
# all remaining bundles carry their own post-hook layer in the metadata.
VECTOR_BUNDLES: dict[str, Path] = {
    "qwen3_4b": FROZEN_VECTOR_ROOT / "qwen3_4b" / "coding_vector_bundle.pt",
    "qwen3_8b": FROZEN_VECTOR_ROOT / "qwen3_8b" / "coding_vector_bundle.pt",
    "qwen3_14b": FROZEN_VECTOR_ROOT / "qwen3_14b" / "coding_vector_bundle.pt",
    "qwen35_4b": FROZEN_VECTOR_ROOT / "qwen35_4b" / "coding_vector_bundle.pt",
    "qwen35_9b": FROZEN_VECTOR_ROOT / "qwen35_9b" / "coding_vector_bundle.pt",
    "granite": FROZEN_VECTOR_ROOT / "granite" / "coding_vector_bundle.pt",
    "mistral": FROZEN_VECTOR_ROOT / "mistral" / "coding_vector_bundle.pt",
    "devstral": FROZEN_VECTOR_ROOT / "devstral" / "coding_vector_bundle.pt",
}


@dataclass(frozen=True)
class FrozenVector:
    bundle: dict[str, Any]
    path: Path
    layer: int
    hook_kind: str
    mean_diff: torch.Tensor
    random_unit: torch.Tensor


@dataclass(frozen=True)
class Condition:
    name: str
    family: str
    alpha: float
    delta_cpu: torch.Tensor | None
    description: str

    @property
    def delta_norm(self) -> float:
        return 0.0 if self.delta_cpu is None else float(self.delta_cpu.norm().item())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        default="all",
        help="Comma-separated collection model keys, or 'all'.",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--selection-version",
        choices=("original", "canonical_batch1"),
        default="original",
        help="Use the original screen-defined set or the batch-size-one revalidated set.",
    )
    parser.add_argument("--max-context-tokens", type=int, default=30_000)
    parser.add_argument(
        "--arms", choices=("both", "suppression", "induction"), default="both"
    )
    parser.add_argument(
        "--alphas",
        type=float,
        nargs="+",
        default=[1.0, 1.5],
        help="Positive amplitudes for the frozen Coding direction.",
    )
    parser.add_argument(
        "--random-alphas",
        type=float,
        nargs="+",
        default=[1.0, 1.5],
        help="Positive amplitudes for the equal-norm seeded Random control.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=0,
        help="Override model-specific batch size; 0 keeps the screened collection setting.",
    )
    parser.add_argument(
        "--max-batch-tokens",
        type=int,
        default=0,
        help="Override model-specific batch-token cap; 0 keeps the screened collection setting.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Diagnostic-only cap per arm; 0 evaluates the full pre-registered 200 rows.",
    )
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--allow-baseline-drift",
        action="store_true",
        help=(
            "Keep an original screen-defined 200-ID collection even if its fresh "
            "baseline under this fixed inference protocol differs for a small number "
            "of near-tie samples. The final audit records every mismatch, and switch "
            "rates are computed against the fresh observed baseline rather than the "
            "stored label."
        ),
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return collection.read_jsonl(path)


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def parse_models(raw: str) -> list[collection.TargetSpec]:
    if raw.strip() == "all":
        return [collection.SPECS[key] for key in collection.SPECS]
    keys = [value.strip() for value in raw.split(",") if value.strip()]
    unknown = [key for key in keys if key not in collection.SPECS]
    if unknown:
        raise ValueError(f"Unknown model keys: {unknown}; choices are {sorted(collection.SPECS)}")
    return [collection.SPECS[key] for key in keys]


def roots_for(spec: collection.TargetSpec, *, selection_version: str) -> tuple[Path, Path, str]:
    if spec.key.startswith("qwen"):
        collection_root = (
            TELECOM_COLLECTION_ROOT
            if selection_version == "original"
            else TELECOM_CANONICAL_COLLECTION_ROOT
        )
        return collection_root, TELECOM_RAW_ROOT, "telecom"
    collection_root = (
        RETAIL_COLLECTION_ROOT
        if selection_version == "original"
        else RETAIL_CANONICAL_COLLECTION_ROOT
    )
    return collection_root, RETAIL_RAW_ROOT, "retail"


def selection_paths(
    spec: collection.TargetSpec, *, selection_version: str
) -> tuple[Path, Path, Path, Path, str]:
    collection_root, raw_root, source_domain = roots_for(spec, selection_version=selection_version)
    model_root = collection_root / spec.key
    return (
        model_root / "selected_tool_200.jsonl",
        model_root / "selected_direct_200.jsonl",
        raw_root,
        model_root / "selection_summary.json",
        source_domain,
    )


def load_vector(path: Path) -> FrozenVector:
    if not path.is_file():
        raise FileNotFoundError(path)
    bundle = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(bundle, dict):
        raise TypeError(f"Expected a mapping bundle in {path}, found {type(bundle)!r}")
    if "mean_diff" not in bundle:
        raise KeyError(f"{path} has no mean_diff")
    mean_diff = torch.as_tensor(bundle["mean_diff"], dtype=torch.float32).reshape(-1).contiguous()
    if not bool(torch.isfinite(mean_diff).all()) or float(mean_diff.norm().item()) <= 0.0:
        raise ValueError(f"Invalid mean_diff in {path}")

    random_value = bundle.get("random_direction", bundle.get("random_direction_unit"))
    if random_value is None:
        raise KeyError(f"{path} has no saved random direction")
    random_unit = torch.as_tensor(random_value, dtype=torch.float32).reshape(-1).contiguous()
    if random_unit.numel() != mean_diff.numel() or not bool(torch.isfinite(random_unit).all()):
        raise ValueError(f"Invalid random direction in {path}")
    random_unit = random_unit / random_unit.norm().clamp_min(1e-12)

    inferred_hook = "pre" if bundle.get("patch_layer") is not None and bundle.get("layer") is None else "post"
    hook_kind = str(bundle.get("hook_kind", inferred_hook))
    if hook_kind not in {"pre", "post"}:
        raise ValueError(f"Unsupported hook kind {hook_kind!r} in {path}")
    layer_value = bundle.get("patch_layer") if hook_kind == "pre" else bundle.get("layer")
    if layer_value is None:
        raise ValueError(f"{path} has no layer for hook kind {hook_kind}")
    layer = int(layer_value)
    if layer < 0:
        raise ValueError(f"Invalid layer {layer} in {path}")
    return FrozenVector(
        bundle=bundle,
        path=path.resolve(),
        layer=layer,
        hook_kind=hook_kind,
        mean_diff=mean_diff,
        random_unit=random_unit,
    )


def resolve_layers(model: Any) -> Any:
    candidates: list[Any] = []
    if hasattr(model, "model"):
        candidates.append(model.model)
        if hasattr(model.model, "language_model"):
            candidates.append(model.model.language_model)
    if hasattr(model, "language_model"):
        candidates.append(model.language_model)
    for candidate in candidates:
        if hasattr(candidate, "layers"):
            return candidate.layers
        if hasattr(candidate, "model") and hasattr(candidate.model, "layers"):
            return candidate.model.layers
    raise AttributeError("Could not resolve decoder layers")


@contextlib.contextmanager
def temporary_last_token_addition(
    model: Any,
    *,
    layer: int,
    hook_kind: str,
    delta_cpu: torch.Tensor | None,
) -> Iterator[dict[str, int]]:
    """Add ``delta_cpu`` only at the last real token of the prompt prefill."""

    stats = {"hook_calls": 0, "modified_calls": 0}
    if delta_cpu is None:
        yield stats
        return
    layers = resolve_layers(model)
    if layer < 0 or layer >= len(layers):
        raise ValueError(f"Invalid L{layer}; model has {len(layers)} decoder layers")
    if hook_kind not in {"pre", "post"}:
        raise ValueError(f"Unsupported hook kind {hook_kind!r}")

    def add_delta(hidden: Any) -> torch.Tensor:
        if not isinstance(hidden, torch.Tensor) or hidden.ndim != 3:
            raise RuntimeError(f"Expected [batch, position, hidden] at L{layer}, got {type(hidden)!r}")
        if int(hidden.shape[-1]) != int(delta_cpu.numel()):
            raise RuntimeError(
                f"Vector dim {delta_cpu.numel()} does not match L{layer} hidden dim {hidden.shape[-1]}"
            )
        patched = hidden.clone()
        patched[:, -1, :] = patched[:, -1, :] + delta_cpu.to(device=patched.device, dtype=patched.dtype)
        stats["modified_calls"] += 1
        return patched

    if hook_kind == "pre":

        def pre_hook(_module: Any, args: tuple[Any, ...], kwargs: dict[str, Any]):
            stats["hook_calls"] += 1
            hidden = args[0] if args else kwargs.get("hidden_states")
            patched = add_delta(hidden)
            if args:
                return (patched, *args[1:]), kwargs
            updated = dict(kwargs)
            updated["hidden_states"] = patched
            return args, updated

        handle = layers[layer].register_forward_pre_hook(pre_hook, with_kwargs=True)
    else:

        def post_hook(_module: Any, _args: tuple[Any, ...], _kwargs: dict[str, Any], output: Any):
            stats["hook_calls"] += 1
            hidden = output[0] if isinstance(output, tuple) else output
            patched = add_delta(hidden)
            if isinstance(output, tuple):
                return (patched, *output[1:])
            return patched

        handle = layers[layer].register_forward_hook(post_hook, with_kwargs=True)
    try:
        yield stats
    finally:
        handle.remove()


def build_conditions(
    vector: FrozenVector, *, arm: str, alphas: Sequence[float], random_alphas: Sequence[float]
) -> list[Condition]:
    sign = -1.0 if arm == "suppression" else 1.0
    verb = "subtract" if arm == "suppression" else "add"
    direction_name = "coding_direction_suppression" if arm == "suppression" else "coding_direction_induction"
    conditions = [Condition("baseline_no_hook_alpha_0", "baseline", 0.0, None, "no intervention")]
    seen: set[float] = set()
    for raw_alpha in alphas:
        alpha = float(raw_alpha)
        if alpha <= 0.0:
            raise ValueError("--alphas must be positive")
        if alpha in seen:
            continue
        seen.add(alpha)
        prefix = "minus" if arm == "suppression" else "plus"
        conditions.append(
            Condition(
                f"{prefix}_mean_diff_alpha_{alpha:g}",
                direction_name,
                alpha,
                (sign * alpha * vector.mean_diff).contiguous(),
                f"{verb} {alpha:g}x frozen Coding mean-difference vector",
            )
        )
    random_matched = vector.random_unit * vector.mean_diff.norm()
    seen.clear()
    for raw_alpha in random_alphas:
        alpha = float(raw_alpha)
        if alpha <= 0.0:
            raise ValueError("--random-alphas must be positive")
        if alpha in seen:
            continue
        seen.add(alpha)
        prefix = "minus" if arm == "suppression" else "plus"
        conditions.append(
            Condition(
                f"{prefix}_random_norm_matched_alpha_{alpha:g}",
                "norm_matched_random_control",
                alpha,
                (sign * alpha * random_matched).contiguous(),
                f"{verb} {alpha:g}x equal-norm saved Random direction",
            )
        )
    return conditions


def token_text(tokenizer: Any, token_id: int) -> str:
    return tokenizer.decode([int(token_id)], clean_up_tokenization_spaces=False)


def prepare_rows(
    rows: Sequence[dict[str, Any]],
    *,
    tokenizer: Any,
    spec: collection.TargetSpec,
    resources: collection.TauResources,
    max_context_tokens: int,
) -> list[collection.PreparedPrefix]:
    prepared: list[collection.PreparedPrefix] = []
    rejected: list[dict[str, Any]] = []
    for row in rows:
        prefix, reason = collection.render_prefix(
            row,
            tokenizer=tokenizer,
            spec=spec,
            resources=resources,
            max_context_tokens=max_context_tokens,
        )
        if prefix is None:
            assert reason is not None
            rejected.append(reason)
        else:
            prepared.append(prefix)
    if rejected:
        raise RuntimeError(f"Selected τ² prefixes no longer render: {json.dumps(rejected[:3], ensure_ascii=False)}")
    if len(prepared) != len(rows):
        raise RuntimeError(f"Rendered {len(prepared)} of {len(rows)} selected prefixes")
    ids = [str(prefix.row["candidate_id"]) for prefix in prepared]
    if len(ids) != len(set(ids)):
        raise RuntimeError("Selected prefixes contain duplicate candidate IDs")
    return prepared


def evaluate_condition(
    prepared: Sequence[collection.PreparedPrefix],
    *,
    model: Any,
    tokenizer: Any,
    device: torch.device,
    tool_token_id: int,
    vector: FrozenVector,
    condition: Condition,
    batch_size: int,
    max_batch_tokens: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    rows: list[dict[str, Any]] = []
    batches = collection.make_batches(
        prepared, batch_size=batch_size, max_batch_tokens=max_batch_tokens
    )
    with temporary_last_token_addition(
        model,
        layer=vector.layer,
        hook_kind=vector.hook_kind,
        delta_cpu=condition.delta_cpu,
    ) as hook_stats:
        for batch_index, batch in enumerate(batches, start=1):
            inputs = collection.make_left_padded_batch(
                batch, pad_token_id=int(tokenizer.pad_token_id), device=device
            )
            logits = collection.model_last_logits(model, inputs)
            top_values, top_ids = torch.max(logits, dim=-1)
            tool_logits = logits[:, tool_token_id]
            tool_probabilities = torch.softmax(logits, dim=-1)[:, tool_token_id]
            non_tool_logits = logits.clone()
            non_tool_logits[:, tool_token_id] = -torch.inf
            non_tool_values, non_tool_ids = torch.max(non_tool_logits, dim=-1)
            for index, prefix in enumerate(batch):
                source = prefix.row
                stored = source.get("tau2_200_baseline") or {}
                top_id = int(top_ids[index].item())
                rows.append(
                    {
                        "candidate_id": str(source["candidate_id"]),
                        "task_id": str(source["task_id"]),
                        "source_domain": source.get("source_domain"),
                        "source_base_task_id": source.get("source_base_task_id"),
                        "trace_index": source.get("trace_index"),
                        "target_tool_name": source.get("target_tool_name"),
                        "target_agent_tool_ordinal": source.get("target_agent_tool_ordinal"),
                        "target_kind": source.get("target_kind"),
                        "prior_agent_tool_depth_group": source.get("prior_agent_tool_depth_group"),
                        "prompt_token_count": int(prefix.length),
                        "adapter": prefix.adapter_audit,
                        "condition": condition.name,
                        "condition_family": condition.family,
                        "alpha": condition.alpha,
                        "direction_description": condition.description,
                        "delta_norm": condition.delta_norm,
                        "tool_call_token_id": int(tool_token_id),
                        "tool_call_logit": float(tool_logits[index].item()),
                        "tool_call_probability": float(tool_probabilities[index].item()),
                        "is_tool_call_top1": bool(top_id == tool_token_id),
                        "top1_token_id": top_id,
                        "top1_token_text": token_text(tokenizer, top_id),
                        "top1_logit": float(top_values[index].item()),
                        "best_non_tool_token_id": int(non_tool_ids[index].item()),
                        "best_non_tool_token_text": token_text(tokenizer, int(non_tool_ids[index].item())),
                        "best_non_tool_logit": float(non_tool_values[index].item()),
                        "tool_call_margin": float((tool_logits[index] - non_tool_values[index]).item()),
                        "stored_baseline_tool_logit": stored.get("baseline_tool_logit"),
                        "stored_baseline_tool_probability": stored.get("baseline_tool_probability"),
                        "stored_baseline_is_tool_call_top1": stored.get("baseline_is_tool_call_top1"),
                    }
                )
            del (
                inputs,
                logits,
                top_values,
                top_ids,
                tool_logits,
                tool_probabilities,
                non_tool_logits,
                non_tool_values,
                non_tool_ids,
            )
            torch.cuda.empty_cache()
            if batch_index % 20 == 0 or batch_index == len(batches):
                print(
                    json.dumps(
                        {
                            "condition": condition.name,
                            "batches_complete": batch_index,
                            "batches_total": len(batches),
                        }
                    ),
                    flush=True,
                )
    if len(rows) != len(prepared):
        raise RuntimeError(f"{condition.name}: produced {len(rows)} rows for {len(prepared)} prefixes")
    if len({row["candidate_id"] for row in rows}) != len(rows):
        raise RuntimeError(f"{condition.name}: duplicate candidate IDs")
    return rows, hook_stats


def summary_row(
    rows: Sequence[dict[str, Any]], baseline_by_id: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    if not rows:
        raise ValueError("Cannot summarize an empty condition")
    baseline = [baseline_by_id[str(row["candidate_id"])] for row in rows]
    n = len(rows)
    tool_count = sum(bool(row["is_tool_call_top1"]) for row in rows)
    baseline_tool_count = sum(bool(row["is_tool_call_top1"]) for row in baseline)
    drops = sum(
        bool(base["is_tool_call_top1"]) and not bool(row["is_tool_call_top1"])
        for row, base in zip(rows, baseline)
    )
    gains = sum(
        not bool(base["is_tool_call_top1"]) and bool(row["is_tool_call_top1"])
        for row, base in zip(rows, baseline)
    )
    return {
        "condition": rows[0]["condition"],
        "condition_family": rows[0]["condition_family"],
        "alpha": rows[0]["alpha"],
        "delta_norm": rows[0]["delta_norm"],
        "n": n,
        "tool_call_top1_count": tool_count,
        "tool_call_top1_rate": tool_count / n,
        "baseline_tool_call_top1_count": baseline_tool_count,
        "baseline_non_tool_top1_count": n - baseline_tool_count,
        "baseline_tool_to_non_tool_count": drops,
        "baseline_tool_to_non_tool_rate": drops / max(baseline_tool_count, 1),
        "baseline_non_tool_to_tool_count": gains,
        "baseline_non_tool_to_tool_rate": gains / max(n - baseline_tool_count, 1),
        "mean_tool_call_logit": sum(float(row["tool_call_logit"]) for row in rows) / n,
        "mean_tool_call_probability": sum(float(row["tool_call_probability"]) for row in rows) / n,
        "mean_tool_logit_change_vs_baseline": sum(
            float(row["tool_call_logit"]) - float(base["tool_call_logit"])
            for row, base in zip(rows, baseline)
        )
        / n,
    }


def validate_fresh_baseline(
    baseline_rows: Sequence[dict[str, Any]],
    *,
    arm: str,
    max_non_tool_probability: float,
    allow_baseline_drift: bool,
) -> dict[str, Any]:
    if arm == "suppression":
        invalid = [row["candidate_id"] for row in baseline_rows if not bool(row["is_tool_call_top1"])]
    else:
        invalid = [
            row["candidate_id"]
            for row in baseline_rows
            if bool(row["is_tool_call_top1"])
            or float(row["tool_call_probability"]) > max_non_tool_probability
        ]
    if invalid and not allow_baseline_drift:
        raise RuntimeError(f"Fresh {arm} baseline no longer matches selection: {invalid[:10]}")
    top1_match = all(
        bool(row["is_tool_call_top1"]) == bool(row["stored_baseline_is_tool_call_top1"])
        for row in baseline_rows
    )
    logit_diffs = [
        abs(float(row["tool_call_logit"]) - float(row["stored_baseline_tool_logit"]))
        for row in baseline_rows
        if row["stored_baseline_tool_logit"] is not None
    ]
    probability_diffs = [
        abs(float(row["tool_call_probability"]) - float(row["stored_baseline_tool_probability"]))
        for row in baseline_rows
        if row["stored_baseline_tool_probability"] is not None
    ]
    if not top1_match and not allow_baseline_drift:
        raise RuntimeError(f"Fresh {arm} top-1 decisions disagree with stored baseline")
    return {
        "all_top1_match": top1_match,
        "baseline_drift_allowed": bool(allow_baseline_drift),
        "fresh_baseline_mismatch_count": len(invalid),
        "fresh_baseline_mismatch_candidate_ids": invalid,
        "fresh_tool_call_top1_count": sum(bool(row["is_tool_call_top1"]) for row in baseline_rows),
        "fresh_non_tool_top1_count": sum(
            not bool(row["is_tool_call_top1"]) for row in baseline_rows
        ),
        "max_abs_tool_logit_difference": max(logit_diffs, default=0.0),
        "max_abs_tool_probability_difference": max(probability_diffs, default=0.0),
    }


def cached_condition(path: Path, candidate_ids: set[str]) -> list[dict[str, Any]] | None:
    if not path.exists():
        return None
    rows = read_jsonl(path)
    if len(rows) != len(candidate_ids) or {str(row.get("candidate_id")) for row in rows} != candidate_ids:
        return None
    return rows


def run_arm(
    *,
    arm: str,
    prepared: Sequence[collection.PreparedPrefix],
    output_root: Path,
    model: Any,
    tokenizer: Any,
    device: torch.device,
    tool_token_id: int,
    vector: FrozenVector,
    alphas: Sequence[float],
    random_alphas: Sequence[float],
    batch_size: int,
    max_batch_tokens: int,
    max_non_tool_probability: float,
    resume: bool,
    allow_baseline_drift: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, dict[str, int]], dict[str, Any]]:
    conditions = build_conditions(vector, arm=arm, alphas=alphas, random_alphas=random_alphas)
    candidate_ids = {str(prefix.row["candidate_id"]) for prefix in prepared}
    if len(candidate_ids) != len(prepared):
        raise RuntimeError(f"{arm} has duplicate candidates")
    by_condition: dict[str, list[dict[str, Any]]] = {}
    hook_stats: dict[str, dict[str, int]] = {}
    checkpoint_root = output_root / "condition_checkpoints"
    for condition in conditions:
        checkpoint = checkpoint_root / f"{arm}_{condition.name}.jsonl"
        rows = cached_condition(checkpoint, candidate_ids) if resume else None
        if rows is not None:
            print(json.dumps({"resuming_cached_condition": condition.name, "arm": arm}), flush=True)
            hook_stats[condition.name] = {"hook_calls": 0, "modified_calls": 0, "cached": 1}
        else:
            print(json.dumps({"starting_condition": condition.name, "arm": arm, "n": len(prepared)}), flush=True)
            rows, stats = evaluate_condition(
                prepared,
                model=model,
                tokenizer=tokenizer,
                device=device,
                tool_token_id=tool_token_id,
                vector=vector,
                condition=condition,
                batch_size=batch_size,
                max_batch_tokens=max_batch_tokens,
            )
            write_jsonl(checkpoint, rows)
            hook_stats[condition.name] = stats
        by_condition[condition.name] = rows
    baseline_name = "baseline_no_hook_alpha_0"
    baseline = {str(row["candidate_id"]): row for row in by_condition[baseline_name]}
    if len(baseline) != len(prepared):
        raise RuntimeError(f"{arm} baseline contains duplicate candidates")
    baseline_audit = validate_fresh_baseline(
        by_condition[baseline_name],
        arm=arm,
        max_non_tool_probability=max_non_tool_probability,
        allow_baseline_drift=allow_baseline_drift,
    )
    summaries = [summary_row(by_condition[condition.name], baseline) for condition in conditions]
    all_rows = [row for condition in conditions for row in by_condition[condition.name]]
    return all_rows, summaries, hook_stats, baseline_audit


def run_model(args: argparse.Namespace, spec: collection.TargetSpec) -> dict[str, Any]:
    tool_path, direct_path, raw_root, selection_summary_path, source_domain = selection_paths(
        spec, selection_version=args.selection_version
    )
    vector = load_vector(VECTOR_BUNDLES[spec.key])
    required = [tool_path, direct_path, raw_root / "tau2_system_prompt.txt", raw_root / "tau2_tool_schemas.json", selection_summary_path]
    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)
    selection_summary = json.loads(selection_summary_path.read_text(encoding="utf-8"))
    if not bool(selection_summary.get("complete")):
        raise RuntimeError(f"Selection is incomplete: {selection_summary_path}")
    source_rows = {"suppression": read_jsonl(tool_path), "induction": read_jsonl(direct_path)}
    for arm, rows in source_rows.items():
        if len(rows) != 200:
            raise RuntimeError(f"Expected exactly 200 {arm} rows, found {len(rows)} in {tool_path if arm == 'suppression' else direct_path}")
    if args.max_samples < 0:
        raise ValueError("--max-samples must be nonnegative")
    if args.max_samples:
        source_rows = {arm: rows[: int(args.max_samples)] for arm, rows in source_rows.items()}
    output_root = args.output_root.resolve() / spec.key
    output_root.mkdir(parents=True, exist_ok=True)
    resources = collection.read_resources(raw_root)
    model = None
    tokenizer = None
    try:
        model, tokenizer, device = collection.load_model_and_tokenizer(spec)
        tool_token_id, token_audit = collection.resolve_tool_token(tokenizer, spec)
        layers = resolve_layers(model)
        if vector.layer >= len(layers):
            raise RuntimeError(f"Vector layer L{vector.layer} is outside {spec.display_name}'s {len(layers)} layers")
        batch_size = int(args.batch_size) if args.batch_size else int(spec.batch_size)
        max_batch_tokens = int(args.max_batch_tokens) if args.max_batch_tokens else int(spec.max_batch_tokens)
        if batch_size < 1 or max_batch_tokens < 1:
            raise ValueError("Batch settings must be positive")
        config = {
            "started_unix": time.time(),
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "model": spec.display_name,
            "model_key": spec.key,
            "model_path": str(spec.model_path),
            "source_domain": source_domain,
            "selection_version": args.selection_version,
            "selection_files": {
                "tool": {"path": str(tool_path), "sha256": sha256_file(tool_path)},
                "direct": {"path": str(direct_path), "sha256": sha256_file(direct_path)},
                "summary": {"path": str(selection_summary_path), "sha256": sha256_file(selection_summary_path)},
            },
            "raw_files": {
                "system_prompt": {"path": str(raw_root / "tau2_system_prompt.txt"), "sha256": sha256_file(raw_root / "tau2_system_prompt.txt")},
                "tool_schemas": {"path": str(raw_root / "tau2_tool_schemas.json"), "sha256": sha256_file(raw_root / "tau2_tool_schemas.json")},
            },
            "frozen_coding_vector": {
                "path": str(vector.path),
                "sha256": sha256_file(vector.path),
                "layer": vector.layer,
                "hook_kind": vector.hook_kind,
                "mean_diff_norm": float(vector.mean_diff.norm().item()),
                "bundle_source_domain": vector.bundle.get("source_domain", "Coding-only artifact"),
                "bundle_fit_split": vector.bundle.get("fit_split"),
                "bundle_model": vector.bundle.get("model", vector.bundle.get("model_label")),
            },
            "tool_decision": token_audit,
            "intervention_hook": f"{vector.hook_kind}-boundary of decoder L{vector.layer}, final real prompt token only",
            "selection_policy": "Frozen Coding vector/layer; τ² is not used to estimate, tune, or select the vector/layer.",
            "baseline_drift_policy": (
                "Permit original screen-defined IDs with an explicit fresh-baseline audit; "
                "switch rates use the fresh observed baseline."
                if args.allow_baseline_drift
                else "Require every fresh baseline decision to match the stored collection label."
            ),
            "conditions": {
                "coding_alphas": list(args.alphas),
                "random_alphas": list(args.random_alphas),
                "random_control": "saved seeded unit vector scaled to ||Coding mean-difference||",
            },
            "batch_size": batch_size,
            "max_batch_tokens": max_batch_tokens,
            "max_context_tokens": int(args.max_context_tokens),
            "arguments": vars(args),
        }
        write_json(output_root / "run_config.json", config)
        arms = ("suppression", "induction") if args.arms == "both" else (args.arms,)
        result: dict[str, Any] = {
            "model": spec.display_name,
            "model_key": spec.key,
            "source_domain": source_domain,
            "n_per_arm": len(source_rows["suppression"]),
            "frozen_coding_vector": config["frozen_coding_vector"],
            "results": {},
        }
        for arm in arms:
            prepared = prepare_rows(
                source_rows[arm],
                tokenizer=tokenizer,
                spec=spec,
                resources=resources,
                max_context_tokens=int(args.max_context_tokens),
            )
            all_rows, summaries, hook_stats, baseline_audit = run_arm(
                arm=arm,
                prepared=prepared,
                output_root=output_root,
                model=model,
                tokenizer=tokenizer,
                device=device,
                tool_token_id=tool_token_id,
                vector=vector,
                alphas=args.alphas,
                random_alphas=args.random_alphas,
                batch_size=batch_size,
                max_batch_tokens=max_batch_tokens,
                max_non_tool_probability=0.05,
                resume=bool(args.resume),
                allow_baseline_drift=bool(args.allow_baseline_drift),
            )
            file_prefix = "suppression" if arm == "suppression" else "induction"
            write_jsonl(output_root / f"{file_prefix}_per_sample.jsonl", all_rows)
            write_csv(output_root / f"{file_prefix}_summary.csv", summaries)
            write_json(
                output_root / f"{file_prefix}_summary.json",
                {
                    "conditions": summaries,
                    "hook_stats": hook_stats,
                    "fresh_baseline_vs_selection": baseline_audit,
                },
            )
            result["results"][arm] = {
                "conditions": summaries,
                "hook_stats": hook_stats,
                "fresh_baseline_vs_selection": baseline_audit,
            }
            write_json(output_root / "partial_result.json", result)
        result["completed_unix"] = time.time()
        result["complete"] = args.arms == "both"
        if result["complete"]:
            write_json(output_root / "final_result.json", result)
        else:
            write_json(output_root / "partial_result.json", result)
        print(json.dumps({"completed": True, "model": spec.key, "output_root": str(output_root)}, ensure_ascii=False), flush=True)
        return result
    finally:
        if model is not None:
            del model
        if tokenizer is not None:
            del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> None:
    args = parse_args()
    if args.max_context_tokens < 1:
        raise ValueError("--max-context-tokens must be positive")
    outcomes = {spec.key: run_model(args, spec) for spec in parse_models(args.models)}
    write_json(args.output_root.resolve() / "run_summary.json", {"models": outcomes, "completed_unix": time.time()})


if __name__ == "__main__":
    main()
