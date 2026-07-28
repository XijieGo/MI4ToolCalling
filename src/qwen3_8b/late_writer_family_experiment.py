#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import torch
import torch.nn.functional as F
from safetensors.torch import load_file

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from differential_feature_mechanism import build_pair_batches, run_intervention  # noqa: E402
from phase6_common import (  # noqa: E402
    CACHE_PATH,
    DTYPE,
    MODEL_PATH,
    TOOL_CALL_TOKEN,
    TRANSCODER_DIR,
    build_sample_pairs,
    clear_cuda,
    dominant_peak_token_type,
    ensure_dir,
    ensure_phase6_cache,
    infer_context_semantic_label,
    load_feature_index,
    normalize_token,
    read_feature_metadata,
    tool_call_token_id,
    top_quantile_examples,
)
from toolcall_circuit.single_sample import load_hooked_qwen3  # noqa: E402
from artifact_paths import ARTIFACT_ROOT  # noqa: E402


PHASE6_EXPB_CSV = ARTIFACT_ROOT / "results" / "8b_main" / "phase6" / "expB_clean_selective.csv"
OUTPUT_ROOT = ARTIFACT_ROOT / "results" / "8b_main" / "section53_late_writer_family"
TARGET_LAYERS = (34, 35)

SCHEMA_TERMS = {
    "<tools",
    "</tools",
    "tool",
    "tools",
    "function",
    "functions",
    "parameter",
    "parameters",
    "argument",
    "arguments",
    "properties",
    "required",
    "schema",
    "json",
    "xml",
    "user",
    "assistant",
    "name",
    "description",
    "type",
}

BOUNDARY_TERMS = {
    "<|im_start|>",
    "<|im_end|>",
    "<think",
    "</think",
    "<tool_call>",
    "```",
    ":\\n",
    ">\\n",
    "?\\n",
    ".\\n",
    "!\\n",
    "user\\n",
    "assistant\\n",
    "tools>\\n",
    "}\\n",
    ")\\n",
    "]\\n",
}


@dataclass(frozen=True)
class FeatureCandidate:
    layer: int
    feature_id: int
    mean_clean: float
    mean_corrupt: float
    delta: float
    ratio: float
    frac_positive_in_corrupt: float
    tool_call_projection: float
    causal_score: float
    dominant_peak_token_type: str
    context_semantic_label: str
    family_label: str
    structural_subtype: str
    schema_hits: int
    boundary_hits: int
    tool_schema_hits: int
    chat_boundary_hits: int
    think_boundary_hits: int
    tool_call_hits: int
    top_logits: str


@dataclass
class LayerPool:
    layer: int
    feature_ids: List[int]
    id_to_local: Dict[int, int]
    clean_values: torch.Tensor
    corrupt_values: torch.Tensor
    W_enc: torch.Tensor
    b_enc: torch.Tensor
    W_dec: torch.Tensor
    row_lookup: Dict[int, FeatureCandidate]


@dataclass(frozen=True)
class SelectionScope:
    name: str
    layers_label: str
    candidates: List[FeatureCandidate]
    control_pool: List[FeatureCandidate]


def write_rows(path: Path, fieldnames: Sequence[str], rows: Iterable[Dict[str, object]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_text(path: Path, text: str) -> None:
    ensure_dir(path.parent)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def write_json(path: Path, payload: Dict[str, object]) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def parse_int_list(raw: str) -> List[int]:
    values = sorted({int(item.strip()) for item in raw.split(",") if item.strip()})
    if not values:
        raise ValueError("At least one integer value is required.")
    return values


def safe_float(value: str | float | int) -> float:
    if value == "" or value is None:
        return float("nan")
    return float(value)


def join_example_tokens(tokens: Sequence[object]) -> str:
    return "".join(normalize_token(token) for token in tokens)


def classify_family(meta: dict) -> tuple[str, str, int, int, int, int, int, int, str, str]:
    dominant_type = dominant_peak_token_type(meta)
    semantic_label = infer_context_semantic_label(meta)
    top_logits_raw = list(meta.get("top_logits") or [])
    top_logits_text = " | ".join(str(item) for item in top_logits_raw[:10])
    top_logits_lower = top_logits_text.lower()

    schema_hits = 0
    boundary_hits = 0
    tool_schema_hits = 0
    chat_boundary_hits = 0
    think_boundary_hits = 0
    tool_call_hits = 0
    examples = top_quantile_examples(meta, limit=5)
    threshold = max(1, min(2, len(examples)))

    for example in examples:
        tokens = list(example.get("tokens") or [])
        if not tokens:
            continue
        peak_index = int(example.get("train_token_ind") or 0)
        left = max(0, peak_index - 4)
        right = min(len(tokens), peak_index + 5)
        window = join_example_tokens(tokens[left:right]).lower()
        full_context = join_example_tokens(tokens).lower()
        peak_token = normalize_token(tokens[peak_index]).lower()

        if "<tools" in full_context and "function" in full_context:
            tool_schema_hits += 1
        if "<|im_end|>" in full_context and "<|im_start|>" in full_context and "user" in full_context:
            chat_boundary_hits += 1
        if "</think>" in full_context:
            think_boundary_hits += 1
        if "<tool_call>" in full_context:
            tool_call_hits += 1

        if semantic_label == "tool_schema" or any(term in window or term in full_context or term in top_logits_lower for term in SCHEMA_TERMS):
            schema_hits += 1

        if (
            any(term in window or term in full_context for term in BOUNDARY_TERMS)
            or ("\n" in peak_token)
            or ("<|im_end|>" in full_context and "<|im_start|>" in full_context)
            or ("<tools" in full_context and ">\n" in full_context)
        ):
            boundary_hits += 1

    if schema_hits >= threshold and boundary_hits >= threshold:
        family = "boundary_schema"
    elif schema_hits >= threshold:
        family = "schema_adjacent"
    elif boundary_hits >= threshold:
        family = "boundary_format"
    else:
        family = "other"

    if tool_schema_hits >= threshold:
        structural_subtype = "tool_schema"
    elif tool_call_hits >= threshold:
        structural_subtype = "tool_call_boundary"
    elif chat_boundary_hits >= threshold and think_boundary_hits < threshold:
        structural_subtype = "chat_turn_boundary"
    elif think_boundary_hits >= threshold:
        structural_subtype = "think_boundary"
    elif boundary_hits >= threshold and schema_hits >= threshold:
        structural_subtype = "generic_boundary_schema"
    elif boundary_hits >= threshold:
        structural_subtype = "generic_boundary"
    elif schema_hits >= threshold:
        structural_subtype = "generic_schema_adjacent"
    else:
        structural_subtype = "other"

    return (
        family,
        structural_subtype,
        schema_hits,
        boundary_hits,
        tool_schema_hits,
        chat_boundary_hits,
        think_boundary_hits,
        tool_call_hits,
        dominant_type,
        semantic_label,
    )


def load_clean_selective_rows(path: Path) -> List[Dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return [row for row in rows if int(row["layer"]) in TARGET_LAYERS]


def build_candidates(model, *, phase6_expb_csv: Path) -> List[FeatureCandidate]:
    feature_index = load_feature_index()
    tool_token_id = tool_call_token_id(model)
    tool_writer = model.W_U[:, tool_token_id].detach().cpu().float()
    rows = load_clean_selective_rows(phase6_expb_csv)
    by_layer: Dict[int, List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_layer[int(row["layer"])].append(row)

    projections: Dict[tuple[int, int], float] = {}
    for layer, layer_rows in by_layer.items():
        weights = load_file(str(TRANSCODER_DIR / f"layer_{layer}.safetensors"))
        W_dec = weights["W_dec"].detach().cpu().float()
        feature_ids = torch.tensor([int(row["feature_id"]) for row in layer_rows], dtype=torch.long)
        proj = torch.mv(W_dec[feature_ids], tool_writer)
        for feature_id, value in zip(feature_ids.tolist(), proj.tolist()):
            projections[(layer, int(feature_id))] = float(value)
        del weights, W_dec, feature_ids, proj
        clear_cuda()

    candidates: List[FeatureCandidate] = []
    for row in rows:
        layer = int(row["layer"])
        feature_id = int(row["feature_id"])
        meta = read_feature_metadata(feature_index, layer, feature_id)
        (
            family,
            structural_subtype,
            schema_hits,
            boundary_hits,
            tool_schema_hits,
            chat_boundary_hits,
            think_boundary_hits,
            tool_call_hits,
            dominant_type,
            semantic_label,
        ) = classify_family(meta)
        projection = projections[(layer, feature_id)]
        delta = safe_float(row["delta"])
        causal_score = delta * max(projection, 0.0)
        candidates.append(
            FeatureCandidate(
                layer=layer,
                feature_id=feature_id,
                mean_clean=safe_float(row["mean_clean"]),
                mean_corrupt=safe_float(row["mean_corrupt"]),
                delta=delta,
                ratio=safe_float(row["ratio"]),
                frac_positive_in_corrupt=safe_float(row["frac_positive_in_corrupt"]),
                tool_call_projection=projection,
                causal_score=causal_score,
                dominant_peak_token_type=dominant_type,
                context_semantic_label=semantic_label,
                family_label=family,
                structural_subtype=structural_subtype,
                schema_hits=schema_hits,
                boundary_hits=boundary_hits,
                tool_schema_hits=tool_schema_hits,
                chat_boundary_hits=chat_boundary_hits,
                think_boundary_hits=think_boundary_hits,
                tool_call_hits=tool_call_hits,
                top_logits=" | ".join(str(item) for item in list(meta.get("top_logits") or [])[:5]),
            )
        )
    candidates.sort(key=lambda item: item.causal_score, reverse=True)
    return candidates


def compute_selected_activations(
    inputs: torch.Tensor,
    W_enc: torch.Tensor,
    b_enc: torch.Tensor,
    *,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    n_samples = int(inputs.shape[0])
    n_features = int(W_enc.shape[0])
    outputs = torch.empty((n_samples, n_features), dtype=torch.float32)

    W_enc_gpu = W_enc.to(device=device, dtype=DTYPE)
    b_enc_gpu = b_enc.to(device=device, dtype=DTYPE)
    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        batch_inputs = inputs[start:end].to(device=device, dtype=DTYPE)
        with torch.no_grad():
            batch_out = torch.relu(F.linear(batch_inputs, W_enc_gpu, b_enc_gpu))
        outputs[start:end] = batch_out.detach().cpu().float()
        del batch_inputs, batch_out
    del W_enc_gpu, b_enc_gpu
    clear_cuda()
    return outputs


def build_layer_pools(
    cache: dict,
    candidates: Sequence[FeatureCandidate],
    *,
    device: torch.device,
    batch_size: int,
) -> Dict[int, LayerPool]:
    by_layer: Dict[int, List[FeatureCandidate]] = defaultdict(list)
    for candidate in candidates:
        by_layer[candidate.layer].append(candidate)

    pools: Dict[int, LayerPool] = {}
    for layer, layer_rows in by_layer.items():
        layer_rows = sorted(layer_rows, key=lambda item: item.feature_id)
        feature_ids = [item.feature_id for item in layer_rows]
        idx_tensor = torch.tensor(feature_ids, dtype=torch.long)
        weights = load_file(str(TRANSCODER_DIR / f"layer_{layer}.safetensors"))
        W_enc = weights["W_enc"][idx_tensor].detach().cpu().to(torch.bfloat16)
        b_enc = weights["b_enc"][idx_tensor].detach().cpu().to(torch.bfloat16)
        W_dec = weights["W_dec"][idx_tensor].detach().cpu().to(torch.bfloat16)
        clean_values = compute_selected_activations(
            cache["mlp_in"]["clean"][layer],
            W_enc,
            b_enc,
            device=device,
            batch_size=batch_size,
        )
        corrupt_values = compute_selected_activations(
            cache["mlp_in"]["corrupt"][layer],
            W_enc,
            b_enc,
            device=device,
            batch_size=batch_size,
        )
        pools[layer] = LayerPool(
            layer=layer,
            feature_ids=feature_ids,
            id_to_local={feature_id: local_idx for local_idx, feature_id in enumerate(feature_ids)},
            clean_values=clean_values,
            corrupt_values=corrupt_values,
            W_enc=W_enc,
            b_enc=b_enc,
            W_dec=W_dec,
            row_lookup={item.feature_id: item for item in layer_rows},
        )
        del weights, idx_tensor
        clear_cuda()
    return pools


def payload_from_candidates(
    candidates: Sequence[FeatureCandidate],
    pools: Dict[int, LayerPool],
) -> Dict[int, Dict[str, torch.Tensor]]:
    grouped: Dict[int, List[FeatureCandidate]] = defaultdict(list)
    for candidate in candidates:
        grouped[candidate.layer].append(candidate)

    payload: Dict[int, Dict[str, torch.Tensor]] = {}
    for layer, layer_candidates in grouped.items():
        pool = pools[layer]
        local_indices = [pool.id_to_local[item.feature_id] for item in layer_candidates]
        idx_tensor = torch.tensor(local_indices, dtype=torch.long)
        payload[layer] = {
            "W_enc": pool.W_enc[idx_tensor].contiguous(),
            "b_enc": pool.b_enc[idx_tensor].contiguous(),
            "W_dec": pool.W_dec[idx_tensor].contiguous(),
            "clean_values": pool.clean_values[:, idx_tensor].contiguous(),
            "corrupt_values": pool.corrupt_values[:, idx_tensor].contiguous(),
        }
    return payload


def score_matched_control(
    selected: Sequence[FeatureCandidate],
    control_pool: Sequence[FeatureCandidate],
) -> List[FeatureCandidate]:
    by_layer_pool: Dict[int, List[FeatureCandidate]] = defaultdict(list)
    for candidate in control_pool:
        by_layer_pool[candidate.layer].append(candidate)
    for layer in by_layer_pool:
        by_layer_pool[layer].sort(key=lambda item: item.causal_score)

    matched: List[FeatureCandidate] = []
    used: set[tuple[int, int]] = set()
    for candidate in selected:
        pool = by_layer_pool[candidate.layer]
        best_idx = None
        best_gap = None
        for idx, control_candidate in enumerate(pool):
            key = (control_candidate.layer, control_candidate.feature_id)
            if key in used:
                continue
            gap = abs(control_candidate.causal_score - candidate.causal_score)
            if best_gap is None or gap < best_gap:
                best_gap = gap
                best_idx = idx
        if best_idx is None:
            continue
        chosen = pool[best_idx]
        used.add((chosen.layer, chosen.feature_id))
        matched.append(chosen)
    return matched


def subset_global_topk(
    candidates: Sequence[FeatureCandidate],
    *,
    layers: Sequence[int] | None = None,
    k: int,
) -> List[FeatureCandidate]:
    filtered = [item for item in candidates if layers is None or item.layer in layers]
    filtered.sort(key=lambda item: item.causal_score, reverse=True)
    return filtered[: min(k, len(filtered))]


def family_counts(candidates: Sequence[FeatureCandidate]) -> Dict[str, int]:
    counts: Dict[str, int] = defaultdict(int)
    for candidate in candidates:
        counts[candidate.family_label] += 1
    return dict(sorted(counts.items()))


def subtype_counts(candidates: Sequence[FeatureCandidate]) -> Dict[str, int]:
    counts: Dict[str, int] = defaultdict(int)
    for candidate in candidates:
        counts[candidate.structural_subtype] += 1
    return dict(sorted(counts.items()))


def run_targeted_suite(
    *,
    model,
    pair_batches,
    baseline: dict,
    pools: Dict[int, LayerPool],
    scopes: Sequence[SelectionScope],
    k_values: Sequence[int],
    tool_token_id_value: int,
) -> tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    rows: List[Dict[str, object]] = []
    selection_rows: List[Dict[str, object]] = []
    for scope in scopes:
        if not scope.candidates:
            continue
        effective_ks = sorted({min(k, len(scope.candidates)) for k in k_values if min(k, len(scope.candidates)) > 0})
        for k in effective_ks:
            family_selected = subset_global_topk(scope.candidates, k=k)
            control_selected = score_matched_control(family_selected, scope.control_pool)
            compared_sets = (
                ("family", family_selected),
                ("matched_control", control_selected),
            )
            for comparison_name, selected in compared_sets:
                if not selected:
                    continue
                for candidate in selected:
                    selection_rows.append(
                        {
                            "scope": scope.name,
                            "comparison": comparison_name,
                            "layers": scope.layers_label,
                            "k": k,
                            "layer": candidate.layer,
                            "feature_id": candidate.feature_id,
                            "family_label": candidate.family_label,
                            "structural_subtype": candidate.structural_subtype,
                            "delta": candidate.delta,
                            "ratio": candidate.ratio,
                            "tool_call_projection": candidate.tool_call_projection,
                            "causal_score": candidate.causal_score,
                        }
                    )
                payload = payload_from_candidates(selected, pools)
                layer_counts = {layer: 0 for layer in TARGET_LAYERS}
                for candidate in selected:
                    layer_counts[candidate.layer] += 1
                for mode, side, source_side, baseline_logits in (
                    ("ablate", "clean", None, baseline["clean_tool_logit"]),
                    ("inject", "corrupt", "clean", baseline["corrupt_tool_logit"]),
                ):
                    result = run_intervention(
                        model,
                        pair_batches,
                        baseline_logits,
                        side=side,
                        tool_token_id=tool_token_id_value,
                        layer_payloads=payload,
                        mode=mode,
                        source_side=source_side,
                    )
                    rows.append(
                        {
                            "scope": scope.name,
                            "comparison": comparison_name,
                            "layers": scope.layers_label,
                            "mode": mode,
                            "side": side,
                            "k": k,
                            "feature_count": len(selected),
                            "L34_count": layer_counts[34],
                            "L35_count": layer_counts[35],
                            **result,
                        }
                    )
    return rows, selection_rows


def build_summary(
    *,
    baseline: dict,
    all_candidates: Sequence[FeatureCandidate],
    structural_candidates: Sequence[FeatureCandidate],
    control_candidates: Sequence[FeatureCandidate],
    result_rows: Sequence[Dict[str, object]],
    selection_rows: Sequence[Dict[str, object]],
    k_values: Sequence[int],
) -> str:
    lines: List[str] = []
    lines.append("# L34/L35 Late Writer Family Summary")
    lines.append("")
    lines.append("## Setup")
    lines.append("")
    lines.append(f"- Layers: `L34`, `L35`")
    lines.append(f"- Eval pairs: `{baseline['n_pairs']}`")
    lines.append(f"- Clean baseline `<tool_call>` top-1: `{baseline['clean_tool_top1_rate']:.2%}`")
    lines.append(f"- Corrupt baseline `<tool_call>` top-1: `{baseline['corrupt_tool_top1_rate']:.2%}`")
    lines.append(f"- Scope k-sweep: `{k_values}`")
    lines.append("")
    lines.append("## Candidate Pool")
    lines.append("")
    lines.append(f"- Total L34/L35 clean-selective features from `phase6/expB_clean_selective.csv`: `{len(all_candidates)}`")
    lines.append(f"- Family counts: `{family_counts(all_candidates)}`")
    lines.append(f"- Structural subtype counts: `{subtype_counts(all_candidates)}`")
    lines.append(f"- Positive-projection strict structural candidates (`tool_schema` or `chat_turn_boundary`): `{len(structural_candidates)}`")
    lines.append(f"- Positive-projection matched-control pool (all non-structural candidates): `{len(control_candidates)}`")
    lines.append("")
    lines.append("## Top Family Features")
    lines.append("")
    top_family = sorted(structural_candidates, key=lambda item: item.causal_score, reverse=True)[:10]
    for candidate in top_family:
        lines.append(
            f"- L{candidate.layer} F{candidate.feature_id}: "
            f"subtype=`{candidate.structural_subtype}`, family=`{candidate.family_label}`, delta={candidate.delta:.2f}, "
            f"ratio={candidate.ratio:.3f}, proj={candidate.tool_call_projection:.2f}, "
            f"score={candidate.causal_score:.2f}, semantic=`{candidate.context_semantic_label}`"
        )
    lines.append("")
    lines.append("## Intervention Results")
    lines.append("")

    def find_row(scope: str, comparison: str, mode: str, k: int | None = None) -> Dict[str, object] | None:
        for row in result_rows:
            if row["scope"] != scope or row["comparison"] != comparison or row["mode"] != mode:
                continue
            if k is not None and int(row["k"]) != k:
                continue
            return row
        return None

    scope_rows: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in result_rows:
        scope_rows[str(row["scope"])].append(row)

    for scope_name in sorted(scope_rows):
        inject_rows = [
            row
            for row in scope_rows[scope_name]
            if row["mode"] == "inject" and row["comparison"] == "family"
        ]
        if not inject_rows:
            continue
        best_inject = max(inject_rows, key=lambda row: float(row["after_tool_top1_rate"]))
        k = int(best_inject["k"])
        family_ablate = find_row(scope_name, "family", "ablate", k)
        control_ablate = find_row(scope_name, "matched_control", "ablate", k)
        control_inject = find_row(scope_name, "matched_control", "inject", k)
        if not family_ablate or not control_ablate or not control_inject:
            continue
        lines.append(
            f"- {scope_name} top-{k}: family ablate clean flip `{float(family_ablate['flip_rate']):.2%}` "
            f"vs control `{float(control_ablate['flip_rate']):.2%}`; "
            f"family inject corrupt recovery `{float(best_inject['after_tool_top1_rate']):.2%}` "
            f"vs control `{float(control_inject['after_tool_top1_rate']):.2%}`."
        )

    lines.append("")
    lines.append("## Selected Features")
    lines.append("")
    grouped_selection: Dict[tuple[str, int], List[Dict[str, object]]] = defaultdict(list)
    for row in selection_rows:
        if row["comparison"] != "family":
            continue
        grouped_selection[(str(row["scope"]), int(row["k"]))].append(row)
    for (scope_name, k), rows_for_scope in sorted(grouped_selection.items()):
        if not rows_for_scope:
            continue
        feature_text = ", ".join(f"L{int(row['layer'])} F{int(row['feature_id'])}" for row in rows_for_scope)
        lines.append(f"- {scope_name} top-{k}: {feature_text}")

    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append(
        "The strict structural family is defined directly from Transcoder feature semantics: "
        "clean-higher L34/L35 features that peak either on `<tools> ... function ...` schema contexts or on `<|im_end|> ... <|im_start|>user` chat-turn boundaries, "
        "with positive decoder projection onto `<tool_call>`."
    )
    lines.append(
        "The central diagnostic is whether these strict structural features outperform matched non-structural controls on both clean ablation and clean-to-corrupt injection. "
        "If they do, Section 5.3 can claim a specific late writer family rather than a generic 'any late clean feature' effect."
    )
    lines.append(
        "Layer-specific scopes separate the L34 schema/boundary writer from the L35 boundary writer, which matters because the smoke run showed that naive cross-layer unions can dilute the rescue effect."
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Targeted L34/L35 late writer family experiment for Section 5.3.")
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--cache-path", type=Path, default=CACHE_PATH)
    parser.add_argument("--phase6-expb-csv", type=Path, default=PHASE6_EXPB_CSV)
    parser.add_argument("--k-values", type=str, default="1,3,5,10")
    parser.add_argument("--per-layer-k", type=int, default=20)
    parser.add_argument("--feature-batch-size", type=int, default=128)
    parser.add_argument("--forward-batch-size", type=int, default=12)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(42)
    torch.manual_seed(42)
    ensure_dir(args.output_root)

    print("[stage] load phase6 cache", flush=True)
    cache = ensure_phase6_cache(cache_path=args.cache_path, force=False)
    print("[stage] load model", flush=True)
    model, _tokenizer = load_hooked_qwen3(str(MODEL_PATH), args.device, DTYPE)
    tool_token_id_value = tool_call_token_id(model)
    print("[stage] tokenize aligned test pairs", flush=True)
    pairs = build_sample_pairs(model)
    if list(cache["sample_ids"]) != [pair.sample_id for pair in pairs]:
        raise RuntimeError("Phase 6 cache sample order does not match freshly built sample pairs.")
    pair_batches = build_pair_batches(pairs, batch_size=args.forward_batch_size)

    baseline = {
        "clean_tool_logit": cache["tool_logit"]["clean"].float(),
        "corrupt_tool_logit": cache["tool_logit"]["corrupt"].float(),
        "clean_top1": cache["top1"]["clean"].long(),
        "corrupt_top1": cache["top1"]["corrupt"].long(),
        "n_pairs": len(pairs),
        "clean_tool_top1_rate": float((cache["top1"]["clean"].long() == tool_token_id_value).float().mean().item()),
        "corrupt_tool_top1_rate": float((cache["top1"]["corrupt"].long() == tool_token_id_value).float().mean().item()),
        "tool_token_id": tool_token_id_value,
        "tool_token": TOOL_CALL_TOKEN,
    }

    print("[stage] classify L34/L35 clean-selective features", flush=True)
    all_candidates = build_candidates(model, phase6_expb_csv=args.phase6_expb_csv)
    structural_candidates = [
        candidate
        for candidate in all_candidates
        if candidate.structural_subtype in {"tool_schema", "chat_turn_boundary", "tool_call_boundary"}
        and candidate.tool_call_projection > 0
    ]
    control_candidates = [
        candidate
        for candidate in all_candidates
        if candidate.structural_subtype not in {"tool_schema", "chat_turn_boundary", "tool_call_boundary"}
        and candidate.tool_call_projection > 0
    ]
    if not structural_candidates:
        raise RuntimeError("No positive-projection strict structural candidates were found.")
    if not control_candidates:
        raise RuntimeError("No positive-projection non-family control candidates were found.")

    layer34_structural = [candidate for candidate in structural_candidates if candidate.layer == 34]
    layer35_structural = [candidate for candidate in structural_candidates if candidate.layer == 35]
    layer34_schema = [candidate for candidate in layer34_structural if candidate.structural_subtype == "tool_schema"]
    layer34_chat_boundary = [candidate for candidate in layer34_structural if candidate.structural_subtype == "chat_turn_boundary"]
    layer35_chat_boundary = [candidate for candidate in layer35_structural if candidate.structural_subtype == "chat_turn_boundary"]

    scopes = [
        SelectionScope(
            name="union_structural",
            layers_label="L34+L35",
            candidates=structural_candidates,
            control_pool=control_candidates,
        ),
        SelectionScope(
            name="L34_structural",
            layers_label="L34",
            candidates=layer34_structural,
            control_pool=[candidate for candidate in control_candidates if candidate.layer == 34],
        ),
        SelectionScope(
            name="L35_structural",
            layers_label="L35",
            candidates=layer35_structural,
            control_pool=[candidate for candidate in control_candidates if candidate.layer == 35],
        ),
        SelectionScope(
            name="L34_schema_only",
            layers_label="L34",
            candidates=layer34_schema,
            control_pool=[candidate for candidate in control_candidates if candidate.layer == 34],
        ),
        SelectionScope(
            name="L34_chat_boundary_only",
            layers_label="L34",
            candidates=layer34_chat_boundary,
            control_pool=[candidate for candidate in control_candidates if candidate.layer == 34],
        ),
        SelectionScope(
            name="L35_chat_boundary_only",
            layers_label="L35",
            candidates=layer35_chat_boundary,
            control_pool=[candidate for candidate in control_candidates if candidate.layer == 35],
        ),
    ]

    print(
        f"[stage] build layer pools: structural={len(structural_candidates)} control={len(control_candidates)}",
        flush=True,
    )
    pooled_lookup: Dict[tuple[int, int], FeatureCandidate] = {}
    for candidate in structural_candidates + control_candidates:
        pooled_lookup[(candidate.layer, candidate.feature_id)] = candidate
    pooled_candidates = sorted(pooled_lookup.values(), key=lambda item: (item.layer, item.feature_id))
    pools = build_layer_pools(
        cache,
        pooled_candidates,
        device=torch.device(args.device),
        batch_size=args.feature_batch_size,
    )

    print("[stage] run targeted interventions", flush=True)
    result_rows, selection_rows = run_targeted_suite(
        model=model,
        pair_batches=pair_batches,
        baseline=baseline,
        pools=pools,
        scopes=scopes,
        k_values=parse_int_list(args.k_values),
        tool_token_id_value=tool_token_id_value,
    )

    print("[stage] write artifacts", flush=True)
    candidate_rows = [
        {
            "layer": candidate.layer,
            "feature_id": candidate.feature_id,
            "family_label": candidate.family_label,
            "structural_subtype": candidate.structural_subtype,
            "mean_clean": candidate.mean_clean,
            "mean_corrupt": candidate.mean_corrupt,
            "delta": candidate.delta,
            "ratio": candidate.ratio,
            "frac_positive_in_corrupt": candidate.frac_positive_in_corrupt,
            "tool_call_projection": candidate.tool_call_projection,
            "causal_score": candidate.causal_score,
            "dominant_peak_token_type": candidate.dominant_peak_token_type,
            "context_semantic_label": candidate.context_semantic_label,
            "schema_hits": candidate.schema_hits,
            "boundary_hits": candidate.boundary_hits,
            "tool_schema_hits": candidate.tool_schema_hits,
            "chat_boundary_hits": candidate.chat_boundary_hits,
            "think_boundary_hits": candidate.think_boundary_hits,
            "tool_call_hits": candidate.tool_call_hits,
            "top_logits": candidate.top_logits,
        }
        for candidate in all_candidates
    ]
    write_rows(
        args.output_root / "candidate_catalog.csv",
        [
            "layer",
            "feature_id",
            "family_label",
            "structural_subtype",
            "mean_clean",
            "mean_corrupt",
            "delta",
            "ratio",
            "frac_positive_in_corrupt",
            "tool_call_projection",
            "causal_score",
            "dominant_peak_token_type",
            "context_semantic_label",
            "schema_hits",
            "boundary_hits",
            "tool_schema_hits",
            "chat_boundary_hits",
            "think_boundary_hits",
            "tool_call_hits",
            "top_logits",
        ],
        candidate_rows,
    )
    write_rows(
        args.output_root / "intervention_results.csv",
        [
            "scope",
            "comparison",
            "layers",
            "mode",
            "side",
            "k",
            "feature_count",
            "L34_count",
            "L35_count",
            "flip_rate",
            "after_tool_top1_rate",
            "mean_logit_delta",
        ],
        result_rows,
    )
    write_rows(
        args.output_root / "selected_scope_catalog.csv",
        [
            "scope",
            "comparison",
            "layers",
            "k",
            "layer",
            "feature_id",
            "family_label",
            "structural_subtype",
            "delta",
            "ratio",
            "tool_call_projection",
            "causal_score",
        ],
        selection_rows,
    )

    max_union_k = min(max(parse_int_list(args.k_values)), len(structural_candidates))
    matched_control = score_matched_control(
        subset_global_topk(structural_candidates, k=max_union_k),
        control_candidates,
    )
    write_rows(
        args.output_root / "matched_control_catalog.csv",
        [
            "layer",
            "feature_id",
            "family_label",
            "structural_subtype",
            "delta",
            "tool_call_projection",
            "causal_score",
        ],
        [
            {
                "layer": candidate.layer,
                "feature_id": candidate.feature_id,
                "family_label": candidate.family_label,
                "structural_subtype": candidate.structural_subtype,
                "delta": candidate.delta,
                "tool_call_projection": candidate.tool_call_projection,
                "causal_score": candidate.causal_score,
            }
            for candidate in matched_control
        ],
    )
    write_json(
        args.output_root / "metadata.json",
        {
            "model_path": str(MODEL_PATH),
            "transcoder_dir": str(TRANSCODER_DIR),
            "target_layers": list(TARGET_LAYERS),
            "k_values": parse_int_list(args.k_values),
            "per_layer_k": args.per_layer_k,
            "n_pairs": len(pairs),
            "tool_token_id": tool_token_id_value,
            "structural_candidate_count": len(structural_candidates),
            "control_candidate_count": len(control_candidates),
            "matched_control_count_at_max_k": len(matched_control),
            "scope_counts": {scope.name: len(scope.candidates) for scope in scopes},
        },
    )
    write_text(
        args.output_root / "summary.md",
        build_summary(
            baseline=baseline,
            all_candidates=all_candidates,
            structural_candidates=structural_candidates,
            control_candidates=control_candidates,
            result_rows=result_rows,
            selection_rows=selection_rows,
            k_values=parse_int_list(args.k_values),
        ),
    )
    print("[done] late writer family experiment complete", flush=True)


if __name__ == "__main__":
    main()
