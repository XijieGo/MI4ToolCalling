#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from tqdm.auto import tqdm

from differential_feature_mechanism import (
    TRANSCODER_DIR,
    clear_cuda,
    collect_layer_inputs_and_baseline,
    compute_dense_features,
    ensure_dir,
    load_dataset_metadata,
    write_text,
)
from phase4_reviewer_strengthening import (
    make_last_token_resid_add_hook,
    make_resid_last_capture,
)
from phase7_l24_directionality_common import (
    PATCH_LAYER,
    direction_scores,
    load_or_collect_pair_baseline,
    project_delta,
)
from phase8_common import (
    DEFAULT_HELDOUT_CACHE,
    DEFAULT_PC_BUNDLE,
    EVAL_DATASET_ROOT,
    configure_matplotlib,
    get_tool_token_id,
    load_gate_bundle,
    load_model_and_tokenizer,
    make_last_token_vector_replace,
    set_seed,
)
from task_attention_path_analysis import (
    REGIONS,
    build_pair_batches,
    load_samples,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DISCOVERY_ROOT = PROJECT_ROOT / "datasets" / "train"
OUTPUT_ROOT = PROJECT_ROOT / "results" / "8b_main" / "phase10_section4_summaries" / "rerun"

NO_NEED_KEYWORDS = {
    "no",
    "none",
    "nothing",
    "not",
    "never",
    "neither",
    "cannot",
    "can't",
    "nowhere",
    "nonexistent",
    "ineligible",
    "impossible",
    "rarely",
    "无需",
    "不需要",
    "不必",
    "不会",
    "不可能",
    "不存在",
    "没有",
    "没有任何",
    "没有人",
    "找不到",
    "无关",
    "不在",
    "根本没有",
}
EXECUTION_HINTS = {
    "write",
    "build",
    "add",
    "save",
    "implement",
    "create",
    "make",
    "complete",
    "function",
    "code",
    "solve",
}
DISCOURSE_KEYWORDS = {
    "narration",
    "paragraph",
    "paragraphs",
    "sentences",
    "written",
    "summary",
    "summaries",
    "past",
    "case",
    "cases",
    "discourse",
    "description",
    "descriptions",
    "叙述",
    "讲话",
    "科普",
    "文字",
    "语气",
    "发言",
    "一篇文章",
    "讲课",
    "谈话",
    "播报",
}
SCHEMA_FORMAT_KEYWORDS = {
    "####",
    "###",
    "**",
    "definitions",
    "definition",
    "concept",
    "concepts",
    "clarification",
    "schema",
    "json",
    "tool",
    "function",
    "functions",
    "arguments",
    "parameters",
}
ACTION_VERBS = {"add", "build", "complete", "create", "implement", "make", "process", "save", "write"}
ANALYSIS_VERBS = {"analyze", "analyse", "benchmark", "clarify", "detail", "discuss", "evaluate", "explore", "inspect", "parse", "review", "study"}


@dataclass(frozen=True)
class FeatureCandidate:
    family_name: str
    layer: int
    feature_idx: int
    kappa: float
    beta: float
    delta_activation: float
    beta_abs: float
    pattern: str
    semantic_score: float
    top_tokens: tuple[str, ...]
    bottom_tokens: tuple[str, ...]
    common_verbs: tuple[str, ...]


@dataclass
class FamilyLayerBundle:
    layer: int
    feature_ids: list[int]
    scores: list[float]
    clean_values: torch.Tensor
    corrupt_values: torch.Tensor
    W_enc: torch.Tensor
    b_enc: torch.Tensor
    W_dec: torch.Tensor


@dataclass
class FamilyEvalBundle:
    family_name: str
    role: str
    active_side: str
    members: list[FeatureCandidate]
    layers: dict[int, FamilyLayerBundle]

    @property
    def n_members(self) -> int:
        return len(self.members)


@dataclass(frozen=True)
class ComboSpec:
    name: str
    heads: tuple[tuple[int, int], ...]
    mlp_layers: tuple[int, ...]
    rationale: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 10 Section 4: feature-family mechanism experiments.")
    parser.add_argument("--model-path", type=Path, default=Path("./external/models/Qwen3-8B"))
    parser.add_argument("--discovery-root", type=Path, default=DISCOVERY_ROOT)
    parser.add_argument("--eval-root", type=Path, default=EVAL_DATASET_ROOT)
    parser.add_argument("--pc-bundle", type=Path, default=DEFAULT_PC_BUNDLE)
    parser.add_argument("--heldout-cache", type=Path, default=DEFAULT_HELDOUT_CACHE)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--layers", type=str, default="21-24")
    parser.add_argument("--causal-max-layer", type=int, default=23)
    parser.add_argument("--discovery-max-pairs", type=int, default=200)
    parser.add_argument("--eval-max-pairs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--feature-compute-batch-size", type=int, default=32)
    parser.add_argument("--candidate-limit-per-side", type=int, default=80)
    parser.add_argument("--family-max-features", type=int, default=80)
    parser.add_argument("--k-values", type=str, default="1,5,10,20,50")
    parser.add_argument("--include-families", type=str, default="")
    parser.add_argument("--include-combos", type=str, default="")
    parser.add_argument("--skip-position-trace", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def parse_layers(raw: str) -> list[int]:
    raw = raw.strip()
    if "-" in raw:
        start_s, end_s = raw.split("-", 1)
        return list(range(int(start_s), int(end_s) + 1))
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def parse_ks(raw: str) -> list[int]:
    return sorted({int(item.strip()) for item in raw.split(",") if item.strip()})


def parse_name_filter(raw: str) -> set[str]:
    return {item.strip() for item in raw.split(",") if item.strip()}


def write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[dict[str, object]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def safe_json_list(value: str) -> list[str]:
    try:
        parsed = json.loads(value)
    except Exception:
        return []
    if not isinstance(parsed, list):
        return []
    return [str(item) for item in parsed]


def normalize_token(token: str) -> str:
    return token.strip().lower()


def count_keyword_hits(tokens: Sequence[str], keywords: set[str]) -> int:
    merged = " ".join(normalize_token(token) for token in tokens)
    hits = 0
    for keyword in keywords:
        if keyword.lower() in merged:
            hits += 1
    return hits


def alignment_pattern(delta_activation: float, beta: float) -> str:
    if delta_activation >= 0.0 and beta >= 0.0:
        return "clean_higher_write_toward_gate"
    if delta_activation < 0.0 and beta < 0.0:
        return "corrupt_higher_write_away_from_gate"
    if delta_activation >= 0.0 and beta < 0.0:
        return "clean_higher_write_away_from_gate"
    return "corrupt_higher_write_toward_gate"


def decode_tokens(tokenizer, token_ids: Sequence[int]) -> tuple[str, ...]:
    return tuple(tokenizer.decode([int(token_id)]) for token_id in token_ids)


def feature_rows_for_layer(
    *,
    layer: int,
    beta: torch.Tensor,
    clean_dense: torch.Tensor,
    corrupt_dense: torch.Tensor,
) -> list[dict[str, object]]:
    mean_clean = clean_dense.mean(dim=0)
    mean_corrupt = corrupt_dense.mean(dim=0)
    delta_activation = mean_clean - mean_corrupt
    active_rate_clean = (clean_dense > 0).float().mean(dim=0)
    active_rate_corrupt = (corrupt_dense > 0).float().mean(dim=0)
    kappa = ((clean_dense - corrupt_dense) * beta.unsqueeze(0)).mean(dim=0)

    rows: list[dict[str, object]] = []
    for feature_idx in range(int(beta.shape[0])):
        delta_val = float(delta_activation[feature_idx].item())
        beta_val = float(beta[feature_idx].item())
        rows.append(
            {
                "layer": layer,
                "feature_idx": feature_idx,
                "mean_clean": float(mean_clean[feature_idx].item()),
                "mean_corrupt": float(mean_corrupt[feature_idx].item()),
                "delta_activation": delta_val,
                "beta": beta_val,
                "beta_abs": abs(beta_val),
                "kappa": float(kappa[feature_idx].item()),
                "active_rate_clean": float(active_rate_clean[feature_idx].item()),
                "active_rate_corrupt": float(active_rate_corrupt[feature_idx].item()),
                "pattern": alignment_pattern(delta_val, beta_val),
            }
        )
    return rows


def topk_feature_rows(rows: Sequence[dict[str, object]], *, positive: bool, limit: int) -> list[dict[str, object]]:
    if positive:
        pool = [row for row in rows if float(row["kappa"]) > 0]
        return sorted(pool, key=lambda row: float(row["kappa"]), reverse=True)[:limit]
    pool = [row for row in rows if float(row["kappa"]) < 0]
    return sorted(pool, key=lambda row: float(row["kappa"]))[:limit]


def build_common_verbs(example_payloads: Sequence[dict[str, object]]) -> tuple[str, ...]:
    counter: Counter[str] = Counter()
    for payload in example_payloads:
        verb = str(payload.get("verb") or "").lower().strip()
        if verb:
            counter[verb] += 1
    return tuple(verb for verb, _count in counter.most_common(3))


def score_family_membership(
    *,
    pattern: str,
    top_tokens: Sequence[str],
    bottom_tokens: Sequence[str],
    common_verbs: Sequence[str],
) -> dict[str, float]:
    top_hits_no_need = count_keyword_hits(top_tokens, NO_NEED_KEYWORDS)
    top_hits_discourse = count_keyword_hits(top_tokens + tuple(bottom_tokens), DISCOURSE_KEYWORDS)
    top_hits_schema = count_keyword_hits(top_tokens + tuple(bottom_tokens), SCHEMA_FORMAT_KEYWORDS)
    top_hits_exec = count_keyword_hits(top_tokens + tuple(bottom_tokens), EXECUTION_HINTS)
    action_hits = sum(verb in ACTION_VERBS for verb in common_verbs)
    analysis_hits = sum(verb in ANALYSIS_VERBS for verb in common_verbs)

    no_need = 2.0 * top_hits_no_need + 1.0 * analysis_hits
    if pattern == "corrupt_higher_write_away_from_gate":
        no_need += 2.0

    execution = 3.0 * action_hits + 1.0 * top_hits_exec
    if pattern == "clean_higher_write_toward_gate":
        execution += 2.0

    analysis = 2.0 * top_hits_discourse + 1.0 * analysis_hits
    if pattern in {"corrupt_higher_write_toward_gate", "clean_higher_write_away_from_gate"}:
        analysis += 1.0

    schema = 2.0 * top_hits_schema
    if pattern in {"corrupt_higher_write_away_from_gate", "clean_higher_write_toward_gate"}:
        schema += 0.5

    return {
        "no_need_non_existence": no_need,
        "execution_request": execution,
        "analysis_plain_discourse": analysis,
        "schema_boundary_formatting": schema,
    }


def family_threshold(family_name: str) -> float:
    if family_name == "execution_request":
        return 4.0
    if family_name == "schema_boundary_formatting":
        return 3.0
    return 4.0


def family_role(family_name: str) -> tuple[str, str]:
    mapping = {
        "no_need_non_existence": ("gate-supporting differential family", "corrupt"),
        "execution_request": ("clean-side supporting family", "clean"),
        "analysis_plain_discourse": ("gate-opposing candidate family", "corrupt"),
        "schema_boundary_formatting": ("supporting/structuring candidate family", "corrupt"),
    }
    return mapping[family_name]


def discover_feature_families(
    *,
    model,
    tokenizer,
    gate_direction: torch.Tensor,
    dataset_root: Path,
    layers: Sequence[int],
    max_pairs: int,
    batch_size: int,
    feature_compute_batch_size: int,
    candidate_limit_per_side: int,
    family_max_features: int,
) -> tuple[dict[str, list[FeatureCandidate]], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    samples = load_samples(dataset_root, model, tokenizer, max_pairs=max_pairs)
    pair_batches = build_pair_batches(samples, batch_size)
    pair_metadata = load_dataset_metadata(dataset_root)
    for sample in samples:
        pair_metadata.setdefault(sample.sample_id, {})
        pair_metadata[sample.sample_id]["sample_id"] = sample.sample_id
        pair_metadata[sample.sample_id]["clean_candidate"] = sample.clean_verb
        pair_metadata[sample.sample_id]["corrupt_candidate"] = sample.corrupt_verb

    tool_token_id = get_tool_token_id(tokenizer)
    clean_inputs, corrupt_inputs, _baseline = collect_layer_inputs_and_baseline(
        model,
        pair_batches,
        layers=layers,
        n_samples=len(samples),
        tool_token_id=tool_token_id,
    )

    family_candidates: dict[str, list[FeatureCandidate]] = defaultdict(list)
    raw_feature_rows: list[dict[str, object]] = []
    token_rows: list[dict[str, object]] = []
    example_rows: list[dict[str, object]] = []

    for layer in tqdm(layers, desc="Family discovery", dynamic_ncols=True):
        tc_weights = load_file(str(TRANSCODER_DIR / f"layer_{layer}.safetensors"))
        W_enc = tc_weights["W_enc"].detach().cpu()
        b_enc = tc_weights["b_enc"].detach().cpu()
        W_dec = tc_weights["W_dec"].detach().cpu()

        clean_dense = compute_dense_features(
            clean_inputs[layer],
            W_enc,
            b_enc,
            device=model.W_U.device,
            compute_batch_size=feature_compute_batch_size,
        )
        corrupt_dense = compute_dense_features(
            corrupt_inputs[layer],
            W_enc,
            b_enc,
            device=model.W_U.device,
            compute_batch_size=feature_compute_batch_size,
        )

        beta = torch.mv(W_dec.float(), gate_direction.float())
        layer_rows = feature_rows_for_layer(layer=layer, beta=beta, clean_dense=clean_dense, corrupt_dense=corrupt_dense)
        candidate_rows = topk_feature_rows(layer_rows, positive=True, limit=candidate_limit_per_side) + topk_feature_rows(
            layer_rows,
            positive=False,
            limit=candidate_limit_per_side,
        )
        candidate_rows = sorted(candidate_rows, key=lambda row: abs(float(row["kappa"])), reverse=True)
        candidate_ids = [int(row["feature_idx"]) for row in candidate_rows]
        decoder_rows = W_dec[torch.tensor(candidate_ids, dtype=torch.long)].to(device=model.W_U.device, dtype=torch.bfloat16)
        with torch.no_grad():
            token_scores = (decoder_rows @ model.W_U.to(dtype=torch.bfloat16)).detach().cpu().float()

        row_lookup = {int(row["feature_idx"]): row for row in candidate_rows}
        for local_idx, feature_idx in enumerate(candidate_ids):
            base_row = row_lookup[feature_idx]
            top_ids = torch.topk(token_scores[local_idx], k=20).indices.tolist()
            bottom_ids = torch.topk(token_scores[local_idx], k=20, largest=False).indices.tolist()
            top_tokens = decode_tokens(tokenizer, top_ids)
            bottom_tokens = decode_tokens(tokenizer, bottom_ids)

            clean_vals = clean_dense[:, feature_idx]
            corrupt_vals = corrupt_dense[:, feature_idx]
            combined: list[tuple[float, str, str]] = []
            for sample_idx, sample in enumerate(samples):
                combined.append((float(clean_vals[sample_idx].item()), "clean", sample.sample_id))
                combined.append((float(corrupt_vals[sample_idx].item()), "corrupt", sample.sample_id))
            combined.sort(key=lambda item: item[0], reverse=True)
            top_examples = []
            for activation, side, sample_id in combined[:5]:
                meta = pair_metadata.get(sample_id, {})
                verb_key = "clean_candidate" if side == "clean" else "corrupt_candidate"
                top_examples.append(
                    {
                        "sample_id": sample_id,
                        "side": side,
                        "activation": activation,
                        "verb": meta.get(verb_key),
                        "language": meta.get("language"),
                        "dataset_name": meta.get("dataset_name"),
                    }
                )
            common_verbs = build_common_verbs(top_examples)
            family_scores = score_family_membership(
                pattern=str(base_row["pattern"]),
                top_tokens=top_tokens,
                bottom_tokens=bottom_tokens,
                common_verbs=common_verbs,
            )
            best_family = max(family_scores.items(), key=lambda item: item[1])[0]
            best_score = float(family_scores[best_family])
            assigned_family = best_family if best_score >= family_threshold(best_family) else ""

            raw_feature_rows.append(
                {
                    "layer": layer,
                    "feature_idx": feature_idx,
                    "kappa": float(base_row["kappa"]),
                    "beta": float(base_row["beta"]),
                    "delta_activation": float(base_row["delta_activation"]),
                    "pattern": str(base_row["pattern"]),
                    "assigned_family": assigned_family,
                    "semantic_score": best_score,
                    "family_scores_json": json.dumps(family_scores, ensure_ascii=False),
                    "common_verbs": json.dumps(list(common_verbs), ensure_ascii=False),
                }
            )

            if assigned_family:
                family_candidates[assigned_family].append(
                    FeatureCandidate(
                        family_name=assigned_family,
                        layer=layer,
                        feature_idx=feature_idx,
                        kappa=float(base_row["kappa"]),
                        beta=float(base_row["beta"]),
                        delta_activation=float(base_row["delta_activation"]),
                        beta_abs=float(base_row["beta_abs"]),
                        pattern=str(base_row["pattern"]),
                        semantic_score=best_score,
                        top_tokens=top_tokens,
                        bottom_tokens=bottom_tokens,
                        common_verbs=common_verbs,
                    )
                )
                token_rows.append(
                    {
                        "family_name": assigned_family,
                        "layer": layer,
                        "feature_idx": feature_idx,
                        "kappa": float(base_row["kappa"]),
                        "beta": float(base_row["beta"]),
                        "pattern": str(base_row["pattern"]),
                        "top_tokens": json.dumps(list(top_tokens), ensure_ascii=False),
                        "bottom_tokens": json.dumps(list(bottom_tokens), ensure_ascii=False),
                    }
                )
                example_rows.append(
                    {
                        "family_name": assigned_family,
                        "layer": layer,
                        "feature_idx": feature_idx,
                        "kappa": float(base_row["kappa"]),
                        "beta": float(base_row["beta"]),
                        "pattern": str(base_row["pattern"]),
                        "common_verbs": json.dumps(list(common_verbs), ensure_ascii=False),
                        "top_examples": json.dumps(top_examples, ensure_ascii=False),
                    }
                )

        del clean_dense, corrupt_dense, W_enc, b_enc, W_dec, tc_weights, decoder_rows, token_scores
        clear_cuda()

    pruned: dict[str, list[FeatureCandidate]] = {}
    for family_name, members in family_candidates.items():
        unique_map: dict[tuple[int, int], FeatureCandidate] = {}
        for member in members:
            key = (member.layer, member.feature_idx)
            current = unique_map.get(key)
            if current is None or (member.semantic_score, abs(member.kappa)) > (current.semantic_score, abs(current.kappa)):
                unique_map[key] = member
        ordered = sorted(unique_map.values(), key=lambda item: (item.semantic_score, abs(item.kappa), item.beta_abs), reverse=True)
        pruned[family_name] = ordered[:family_max_features]
    return pruned, raw_feature_rows, token_rows, example_rows


def build_family_catalog_rows(
    families: dict[str, list[FeatureCandidate]],
    *,
    causal_max_layer: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for family_name, members in families.items():
        role, active_side = family_role(family_name)
        layers = sorted({member.layer for member in members})
        causal_members = [member for member in members if member.layer <= causal_max_layer]
        patterns = Counter(member.pattern for member in members)
        rows.append(
            {
                "family_name": family_name,
                "role": role,
                "active_side": active_side,
                "n_members": len(members),
                "n_causal_members_lte_l23": len(causal_members),
                "layers": json.dumps(layers),
                "dominant_patterns": json.dumps(dict(patterns), ensure_ascii=False),
                "representative_features": json.dumps(
                    [f"L{member.layer}F{member.feature_idx}" for member in members[:6]],
                    ensure_ascii=False,
                ),
                "representative_verbs": json.dumps(
                    list(dict.fromkeys(verb for member in members[:10] for verb in member.common_verbs if verb))[:6],
                    ensure_ascii=False,
                ),
            }
        )
    return rows


def build_family_eval_bundles(
    *,
    model,
    tokenizer,
    gate_direction: torch.Tensor,
    dataset_root: Path,
    max_pairs: int,
    batch_size: int,
    feature_compute_batch_size: int,
    families: dict[str, list[FeatureCandidate]],
    causal_max_layer: int,
    cache_root: Path,
) -> tuple[dict[str, FamilyEvalBundle], dict[str, object], list]:
    pair_cache = load_or_collect_pair_baseline(
        model,
        tokenizer,
        dataset_root=dataset_root,
        max_pairs=max_pairs,
        batch_size=batch_size,
        patch_layer=PATCH_LAYER,
        cache_path=cache_root / f"{dataset_root.name}_pair_baseline_{max_pairs}.pt",
    )
    tool_token_id = get_tool_token_id(tokenizer)
    eval_clean_inputs, eval_corrupt_inputs, _baseline = collect_layer_inputs_and_baseline(
        model,
        pair_cache.pair_batches,
        layers=sorted({member.layer for members in families.values() for member in members}),
        n_samples=len(pair_cache.samples),
        tool_token_id=tool_token_id,
    )

    bundles: dict[str, FamilyEvalBundle] = {}
    for family_name, members in families.items():
        role, active_side = family_role(family_name)
        causal_members = [member for member in members if member.layer <= causal_max_layer]
        if not causal_members:
            continue
        layer_to_members: dict[int, list[FeatureCandidate]] = defaultdict(list)
        for member in causal_members:
            layer_to_members[member.layer].append(member)
        layer_bundles: dict[int, FamilyLayerBundle] = {}
        for layer, layer_members in sorted(layer_to_members.items()):
            tc_weights = load_file(str(TRANSCODER_DIR / f"layer_{layer}.safetensors"))
            feature_ids = [member.feature_idx for member in layer_members]
            idx_tensor = torch.tensor(feature_ids, dtype=torch.long)
            W_enc_full = tc_weights["W_enc"].detach().cpu()[idx_tensor].contiguous()
            b_enc_full = tc_weights["b_enc"].detach().cpu()[idx_tensor].contiguous()
            W_dec_full = tc_weights["W_dec"].detach().cpu()[idx_tensor].contiguous()
            clean_values = compute_dense_features(
                eval_clean_inputs[layer],
                W_enc_full,
                b_enc_full,
                device=model.W_U.device,
                compute_batch_size=feature_compute_batch_size,
            )
            corrupt_values = compute_dense_features(
                eval_corrupt_inputs[layer],
                W_enc_full,
                b_enc_full,
                device=model.W_U.device,
                compute_batch_size=feature_compute_batch_size,
            )
            layer_bundles[layer] = FamilyLayerBundle(
                layer=layer,
                feature_ids=feature_ids,
                scores=[abs(member.kappa) for member in layer_members],
                clean_values=clean_values,
                corrupt_values=corrupt_values,
                W_enc=W_enc_full,
                b_enc=b_enc_full,
                W_dec=W_dec_full,
            )
            del tc_weights
            clear_cuda()

        bundles[family_name] = FamilyEvalBundle(
            family_name=family_name,
            role=role,
            active_side=active_side,
            members=causal_members,
            layers=layer_bundles,
        )
    return bundles, {"pair_cache": pair_cache}, pair_cache.pair_batches


def build_random_control_bundle(
    *,
    model,
    eval_clean_inputs: dict[int, torch.Tensor],
    eval_corrupt_inputs: dict[int, torch.Tensor],
    candidate_pool: Sequence[FeatureCandidate],
    n_members: int,
    feature_compute_batch_size: int,
    seed: int,
) -> FamilyEvalBundle:
    rng = np.random.default_rng(seed)
    picked = sorted(
        rng.choice(np.arange(len(candidate_pool)), size=min(n_members, len(candidate_pool)), replace=False).tolist()
    )
    members = [candidate_pool[idx] for idx in picked]
    layer_to_members: dict[int, list[FeatureCandidate]] = defaultdict(list)
    for member in members:
        layer_to_members[member.layer].append(member)
    layer_bundles: dict[int, FamilyLayerBundle] = {}
    for layer, layer_members in sorted(layer_to_members.items()):
        tc_weights = load_file(str(TRANSCODER_DIR / f"layer_{layer}.safetensors"))
        feature_ids = [member.feature_idx for member in layer_members]
        idx_tensor = torch.tensor(feature_ids, dtype=torch.long)
        W_enc_full = tc_weights["W_enc"].detach().cpu()[idx_tensor].contiguous()
        b_enc_full = tc_weights["b_enc"].detach().cpu()[idx_tensor].contiguous()
        W_dec_full = tc_weights["W_dec"].detach().cpu()[idx_tensor].contiguous()
        clean_values = compute_dense_features(
            eval_clean_inputs[layer],
            W_enc_full,
            b_enc_full,
            device=model.W_U.device,
            compute_batch_size=feature_compute_batch_size,
        )
        corrupt_values = compute_dense_features(
            eval_corrupt_inputs[layer],
            W_enc_full,
            b_enc_full,
            device=model.W_U.device,
            compute_batch_size=feature_compute_batch_size,
        )
        layer_bundles[layer] = FamilyLayerBundle(
            layer=layer,
            feature_ids=feature_ids,
            scores=[abs(member.kappa) for member in layer_members],
            clean_values=clean_values,
            corrupt_values=corrupt_values,
            W_enc=W_enc_full,
            b_enc=b_enc_full,
            W_dec=W_dec_full,
        )
        del tc_weights
        clear_cuda()
    return FamilyEvalBundle(
        family_name="random_matched_control",
        role="random matched control",
        active_side="corrupt",
        members=members,
        layers=layer_bundles,
    )


def layer_member_count(bundle: FamilyEvalBundle, k: int) -> dict[int, int]:
    counts: dict[int, int] = {}
    for member in bundle.members[:k]:
        counts[member.layer] = counts.get(member.layer, 0) + 1
    return counts


def reorder_bundle_global(bundle: FamilyEvalBundle) -> FamilyEvalBundle:
    ordered_members = sorted(bundle.members, key=lambda item: (item.semantic_score, abs(item.kappa), item.beta_abs), reverse=True)
    layer_to_members: dict[int, list[FeatureCandidate]] = defaultdict(list)
    for member in ordered_members:
        layer_to_members[member.layer].append(member)
    new_layers: dict[int, FamilyLayerBundle] = {}
    for layer, layer_members in layer_to_members.items():
        old = bundle.layers[layer]
        old_index = {feature_id: idx for idx, feature_id in enumerate(old.feature_ids)}
        local_indices = [old_index[member.feature_idx] for member in layer_members]
        new_layers[layer] = FamilyLayerBundle(
            layer=layer,
            feature_ids=[member.feature_idx for member in layer_members],
            scores=[abs(member.kappa) for member in layer_members],
            clean_values=old.clean_values[:, local_indices].contiguous(),
            corrupt_values=old.corrupt_values[:, local_indices].contiguous(),
            W_enc=old.W_enc[local_indices].contiguous(),
            b_enc=old.b_enc[local_indices].contiguous(),
            W_dec=old.W_dec[local_indices].contiguous(),
        )
    return FamilyEvalBundle(
        family_name=bundle.family_name,
        role=bundle.role,
        active_side=bundle.active_side,
        members=ordered_members,
        layers=dict(sorted(new_layers.items())),
    )


def make_family_feature_hooks(
    bundle: FamilyEvalBundle,
    batch_indices: Sequence[int],
    *,
    k: int,
    mode: str,
    source_side: str | None,
):
    hooks = []
    state: dict[int, torch.Tensor] = {}
    counts = layer_member_count(bundle, k)
    for layer, layer_bundle in sorted(bundle.layers.items()):
        count = counts.get(layer, 0)
        if count <= 0:
            continue
        W_enc = layer_bundle.W_enc[:count]
        b_enc = layer_bundle.b_enc[:count]
        W_dec = layer_bundle.W_dec[:count]
        source_values = None
        if mode == "inject":
            if source_side == "clean":
                source_values = layer_bundle.clean_values[list(batch_indices), :count]
            elif source_side == "corrupt":
                source_values = layer_bundle.corrupt_values[list(batch_indices), :count]
            else:
                raise ValueError("Feature injection requires source_side.")

        def make_in_hook(layer_key: int):
            def in_hook(value: torch.Tensor, hook):  # noqa: ANN001
                state[layer_key] = value[:, -1, :].detach()
                return value

            return in_hook

        def make_out_hook(
            layer_key: int,
            W_enc_cpu: torch.Tensor,
            b_enc_cpu: torch.Tensor,
            W_dec_cpu: torch.Tensor,
            source_cpu: torch.Tensor | None,
        ):
            def out_hook(value: torch.Tensor, hook):  # noqa: ANN001
                mlp_in = state.pop(layer_key)
                W_enc_dev = W_enc_cpu.to(device=value.device, dtype=torch.bfloat16)
                b_enc_dev = b_enc_cpu.to(device=value.device, dtype=torch.bfloat16)
                W_dec_dev = W_dec_cpu.to(device=value.device, dtype=torch.bfloat16)
                acts = torch.relu(F.linear(mlp_in.to(dtype=torch.bfloat16), W_enc_dev, b_enc_dev))
                current_contrib = acts @ W_dec_dev
                out = value.clone()
                if mode == "ablate":
                    out[:, -1, :] = out[:, -1, :] - current_contrib.to(dtype=out.dtype)
                elif mode == "inject":
                    if source_cpu is None:
                        raise RuntimeError("Missing feature source values for injection.")
                    target_contrib = source_cpu.to(device=value.device, dtype=torch.bfloat16) @ W_dec_dev
                    out[:, -1, :] = out[:, -1, :] - current_contrib.to(dtype=out.dtype) + target_contrib.to(dtype=out.dtype)
                else:
                    raise ValueError(f"Unsupported feature mode: {mode}")
                return out

            return out_hook

        hooks.append((f"blocks.{layer}.hook_mlp_in", make_in_hook(layer)))
        hooks.append((f"blocks.{layer}.hook_mlp_out", make_out_hook(layer, W_enc, b_enc, W_dec, source_values)))
    return hooks


def evaluate_feature_intervention(
    *,
    model,
    pair_cache,
    gate_direction: torch.Tensor,
    tool_token_id: int,
    family_bundle: FamilyEvalBundle,
    k: int,
    eval_side: str,
    mode: str,
    source_side: str | None,
    extra_gate_hook=None,
) -> tuple[dict[str, float], torch.Tensor, torch.Tensor]:
    gate_hook_name = f"blocks.{PATCH_LAYER}.hook_resid_pre"
    patched_side_gate = torch.empty(len(pair_cache.samples), dtype=torch.float32)
    patched_tool_logit = torch.empty(len(pair_cache.samples), dtype=torch.float32)
    patched_top1 = torch.empty(len(pair_cache.samples), dtype=torch.long)

    baseline_clean_gate = direction_scores(pair_cache.clean_resid, gate_direction)
    baseline_corrupt_gate = direction_scores(pair_cache.corrupt_resid, gate_direction)
    baseline_clean_top1 = pair_cache.baseline_clean["top1"]
    baseline_corrupt_top1 = pair_cache.baseline_corrupt["top1"]
    baseline_clean_logit = pair_cache.baseline_clean["tool_logit"]
    baseline_corrupt_logit = pair_cache.baseline_corrupt["tool_logit"]

    progress = tqdm(pair_cache.pair_batches, desc=f"{family_bundle.family_name}:{eval_side}:{mode}:k={k}", dynamic_ncols=True, leave=False)
    for batch in progress:
        tokens_cpu = batch.clean_tokens_cpu if eval_side == "clean" else batch.corrupt_tokens_cpu
        tokens = tokens_cpu.to(model.W_U.device)
        hooks = make_family_feature_hooks(
            family_bundle,
            batch.indices,
            k=k,
            mode=mode,
            source_side=source_side,
        )
        capture: dict[str, torch.Tensor] = {}
        if extra_gate_hook is not None:
            hooks.append((gate_hook_name, extra_gate_hook(batch.indices)))
        hooks.append((gate_hook_name, make_resid_last_capture(capture, "gate")))
        with torch.no_grad():
            logits = model.run_with_hooks(tokens, fwd_hooks=hooks)
        tool_logit = logits[:, -1, tool_token_id].detach().cpu().float()
        top1 = logits[:, -1, :].argmax(dim=-1).detach().cpu()
        gate_score = torch.mv(capture["gate"].float(), gate_direction.float())
        patched_side_gate[batch.indices] = gate_score
        patched_tool_logit[batch.indices] = tool_logit
        patched_top1[batch.indices] = top1
        clear_cuda()

    if eval_side == "corrupt":
        patched_delta_g24 = baseline_clean_gate - patched_side_gate
        baseline_delta_g24 = baseline_clean_gate - baseline_corrupt_gate
        strict_flip_rate = float(((baseline_corrupt_top1 != tool_token_id) & (patched_top1 == tool_token_id)).float().mean().item())
        mean_tool_logit_shift = float((patched_tool_logit - baseline_corrupt_logit).mean().item())
        tool_call_top1_rate = float((patched_top1 == tool_token_id).float().mean().item())
    else:
        patched_delta_g24 = patched_side_gate - baseline_corrupt_gate
        baseline_delta_g24 = baseline_clean_gate - baseline_corrupt_gate
        strict_flip_rate = float(((baseline_clean_top1 == tool_token_id) & (patched_top1 != tool_token_id)).float().mean().item())
        mean_tool_logit_shift = float((patched_tool_logit - baseline_clean_logit).mean().item())
        tool_call_top1_rate = float((patched_top1 == tool_token_id).float().mean().item())

    delta_shift = patched_delta_g24 - baseline_delta_g24
    summary = {
        "mean_patched_delta_g24": float(patched_delta_g24.mean().item()),
        "mean_baseline_delta_g24": float(baseline_delta_g24.mean().item()),
        "mean_delta_g24_shift": float(delta_shift.mean().item()),
        "mean_delta_g24_fraction": float((delta_shift / baseline_delta_g24.clamp_min(1e-6)).mean().item()),
        "mean_side_gate_score": float(patched_side_gate.mean().item()),
        "mean_tool_logit_shift": mean_tool_logit_shift,
        "tool_call_top1_rate": tool_call_top1_rate,
        "strict_flip_rate": strict_flip_rate,
    }
    return summary, patched_side_gate, patched_tool_logit


def plot_family_effects(rows: Sequence[dict[str, object]], path: Path) -> None:
    configure_matplotlib()
    named_rows = [row for row in rows if str(row["family_name"]) != "random_matched_control" and int(row["k"]) == int(row["selected_k"])]
    if not named_rows:
        return
    labels = [f"{row['family_name']}\n{row['intervention']}" for row in named_rows]
    delta = np.asarray([float(row["mean_delta_g24_shift"]) for row in named_rows], dtype=np.float32)
    flip = np.asarray([float(row["strict_flip_rate"]) for row in named_rows], dtype=np.float32)
    x = np.arange(len(named_rows))
    fig, axes = plt.subplots(2, 1, figsize=(max(10, len(named_rows) * 1.2), 7), constrained_layout=True)
    axes[0].bar(x, delta, color="#1b6ca8")
    axes[0].axhline(0.0, color="#222222", linewidth=1.0)
    axes[0].set_ylabel("mean Δg24 shift")
    axes[0].set_title("Family Causality Toward the L24 Gate")
    axes[1].bar(x, flip, color="#c65d00")
    axes[1].set_ylabel("strict flip rate")
    axes[1].set_ylim(0.0, 1.05)
    axes[1].set_xticks(x, labels, rotation=30, ha="right")
    for ax in axes:
        ax.grid(axis="y", alpha=0.25)
    ensure_dir(path.parent)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def pick_best_rows_for_families(rows: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        key = (str(row["family_name"]), str(row["intervention"]))
        grouped[key].append(dict(row))
    selected: list[dict[str, object]] = []
    for key, grow in grouped.items():
        best = max(grow, key=lambda row: abs(float(row["mean_delta_g24_shift"])))
        selected.append(best)
    return sorted(selected, key=lambda row: (row["family_name"], row["intervention"]))


def build_matched_gate_restore_hook(
    *,
    gate_shift_vectors: torch.Tensor,
):
    def resolver(batch_indices: Sequence[int]):
        return make_last_token_resid_add_hook(gate_shift_vectors[list(batch_indices)])

    return resolver


def run_gate_mediation(
    *,
    model,
    pair_cache,
    gate_bundle: dict[str, torch.Tensor],
    gate_direction: torch.Tensor,
    tool_token_id: int,
    family_bundle: FamilyEvalBundle,
    intervention_row: dict[str, object],
) -> dict[str, object]:
    k = int(intervention_row["k"])
    eval_side = str(intervention_row["eval_side"])
    mode = str(intervention_row["mode"])
    source_side = intervention_row["source_side"] or None
    family_only, patched_side_gate, patched_tool_logit = evaluate_feature_intervention(
        model=model,
        pair_cache=pair_cache,
        gate_direction=gate_direction,
        tool_token_id=tool_token_id,
        family_bundle=family_bundle,
        k=k,
        eval_side=eval_side,
        mode=mode,
        source_side=source_side,
    )

    pair_diff = pair_cache.clean_resid - pair_cache.corrupt_resid
    rank1_delta = project_delta(pair_diff, gate_bundle, k=1)
    rank1_gate = torch.mv(rank1_delta.float(), gate_direction.float())
    baseline_target_gate = direction_scores(pair_cache.clean_resid if eval_side == "clean" else pair_cache.corrupt_resid, gate_direction)
    desired_shift = baseline_target_gate - patched_side_gate
    scale = desired_shift / rank1_gate.clamp_min(1e-6)
    gate_restore_vectors = scale.unsqueeze(1) * rank1_delta

    gate_restore, gate_restore_side_gate, gate_restore_tool_logit = evaluate_feature_intervention(
        model=model,
        pair_cache=pair_cache,
        gate_direction=gate_direction,
        tool_token_id=tool_token_id,
        family_bundle=family_bundle,
        k=k,
        eval_side=eval_side,
        mode=mode,
        source_side=source_side,
        extra_gate_hook=build_matched_gate_restore_hook(gate_shift_vectors=gate_restore_vectors),
    )

    baseline_resid = pair_cache.clean_resid if eval_side == "clean" else pair_cache.corrupt_resid

    def full_restore_resolver(batch_indices: Sequence[int]):
        return make_last_token_vector_replace(baseline_resid[list(batch_indices)])

    full_restore, full_restore_side_gate, full_restore_tool_logit = evaluate_feature_intervention(
        model=model,
        pair_cache=pair_cache,
        gate_direction=gate_direction,
        tool_token_id=tool_token_id,
        family_bundle=family_bundle,
        k=k,
        eval_side=eval_side,
        mode=mode,
        source_side=source_side,
        extra_gate_hook=full_restore_resolver,
    )

    baseline_delta = float(family_only["mean_baseline_delta_g24"])
    family_delta = float(family_only["mean_patched_delta_g24"])
    gate_delta = float(gate_restore["mean_patched_delta_g24"])
    full_delta = float(full_restore["mean_patched_delta_g24"])
    baseline_logit = float((pair_cache.baseline_clean["tool_logit"] if eval_side == "clean" else pair_cache.baseline_corrupt["tool_logit"]).mean().item())
    family_logit = float((patched_tool_logit).mean().item())
    gate_logit = float(gate_restore_tool_logit.mean().item())
    full_logit = float(full_restore_tool_logit.mean().item())
    gate_closed = 1.0 - abs(gate_delta - baseline_delta) / max(abs(family_delta - baseline_delta), 1e-6)
    full_closed = 1.0 - abs(full_delta - baseline_delta) / max(abs(family_delta - baseline_delta), 1e-6)
    gate_logit_closed = 1.0 - abs(gate_logit - baseline_logit) / max(abs(family_logit - baseline_logit), 1e-6)
    full_logit_closed = 1.0 - abs(full_logit - baseline_logit) / max(abs(family_logit - baseline_logit), 1e-6)
    return {
        "family_name": family_bundle.family_name,
        "intervention": str(intervention_row["intervention"]),
        "eval_side": eval_side,
        "k": k,
        "baseline_delta_g24": baseline_delta,
        "family_only_delta_g24": family_delta,
        "gate_restore_delta_g24": gate_delta,
        "full_restore_delta_g24": full_delta,
        "gate_restore_closed_fraction": gate_closed,
        "full_restore_closed_fraction": full_closed,
        "baseline_tool_logit": baseline_logit,
        "family_only_tool_logit": family_logit,
        "gate_restore_tool_logit": gate_logit,
        "full_restore_tool_logit": full_logit,
        "gate_restore_logit_closed_fraction": gate_logit_closed,
        "full_restore_logit_closed_fraction": full_logit_closed,
        "family_only_top1_rate": float(family_only["tool_call_top1_rate"]),
        "gate_restore_top1_rate": float(gate_restore["tool_call_top1_rate"]),
        "full_restore_top1_rate": float(full_restore["tool_call_top1_rate"]),
    }


def plot_mediation(rows: Sequence[dict[str, object]], path: Path) -> None:
    if not rows:
        return
    configure_matplotlib()
    labels = [str(row["family_name"]) for row in rows]
    x = np.arange(len(rows))
    width = 0.25
    family_only = np.asarray([float(row["family_only_delta_g24"]) for row in rows], dtype=np.float32)
    gate_restore = np.asarray([float(row["gate_restore_delta_g24"]) for row in rows], dtype=np.float32)
    full_restore = np.asarray([float(row["full_restore_delta_g24"]) for row in rows], dtype=np.float32)
    baseline = np.asarray([float(row["baseline_delta_g24"]) for row in rows], dtype=np.float32)
    fig, ax = plt.subplots(figsize=(max(8, len(rows) * 1.5), 4.2))
    ax.bar(x - width, family_only, width=width, color="#b22222", label="family only")
    ax.bar(x, gate_restore, width=width, color="#1b6ca8", label="family + gate restore")
    ax.bar(x + width, full_restore, width=width, color="#5aa469", label="family + full restore")
    ax.plot(x, baseline, color="#222222", marker="o", linewidth=1.5, label="baseline")
    ax.set_ylabel("mean Δg24")
    ax.set_xticks(x, labels)
    ax.set_title("Gate Mediation Test")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    ensure_dir(path.parent)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def build_region_masks_with_prediction(sample, side: str) -> dict[str, torch.Tensor]:
    base_masks = sample.clean_region_masks if side == "clean" else sample.corrupt_region_masks
    seq_len = int(sample.clean_tokens_cpu.shape[-1])
    pred_mask = torch.zeros(seq_len, dtype=torch.bool)
    pred_mask[-1] = True
    region_masks: dict[str, torch.Tensor] = {"prediction_position": pred_mask}
    occupied = pred_mask.clone()
    for region in REGIONS:
        mask = base_masks[region].clone()
        mask[-1] = False
        region_masks[region] = mask
        occupied |= mask
    region_masks["other_non_prediction"] = ~occupied
    return region_masks


def run_position_trace(
    *,
    model,
    pair_cache,
    top_families: Sequence[tuple[FamilyEvalBundle, int]],
    batch_size: int,
) -> list[dict[str, object]]:
    hook_layers = sorted({layer for family_bundle, _k in top_families for layer in family_bundle.layers.keys()} | {24})
    hook_names = [f"blocks.{layer}.hook_mlp_in" for layer in hook_layers]
    rows: list[dict[str, object]] = []

    progress = tqdm(pair_cache.pair_batches, desc="Feature position trace", dynamic_ncols=True)
    for batch in progress:
        for side in ("clean", "corrupt"):
            tokens_cpu = batch.clean_tokens_cpu if side == "clean" else batch.corrupt_tokens_cpu
            tokens = tokens_cpu.to(model.W_U.device)
            with torch.no_grad():
                _, cache = model.run_with_cache(tokens, names_filter=lambda name: name in hook_names)
            for family_bundle, k in top_families:
                counts = layer_member_count(family_bundle, k)
                for layer, layer_bundle in sorted(family_bundle.layers.items()):
                    count = counts.get(layer, 0)
                    if count <= 0:
                        continue
                    hidden = cache[f"blocks.{layer}.hook_mlp_in"].detach()
                    W_enc = layer_bundle.W_enc[:count].to(device=hidden.device, dtype=torch.bfloat16)
                    b_enc = layer_bundle.b_enc[:count].to(device=hidden.device, dtype=torch.bfloat16)
                    acts = torch.relu(F.linear(hidden.to(dtype=torch.bfloat16), W_enc, b_enc)).float().sum(dim=-1).detach().cpu()
                    for local_idx, sample_idx in enumerate(batch.indices):
                        sample = pair_cache.samples[sample_idx]
                        region_masks = build_region_masks_with_prediction(sample, side)
                        total = float(acts[local_idx].sum().item())
                        for region_name, mask in region_masks.items():
                            rows.append(
                                {
                                    "family_name": family_bundle.family_name,
                                    "side": side,
                                    "layer": layer,
                                    "k": k,
                                    "sample_id": sample.sample_id,
                                    "region": region_name,
                                    "activation_mass": float(acts[local_idx, mask].sum().item()),
                                    "activation_share": float(acts[local_idx, mask].sum().item() / max(total, 1e-6)),
                                    "total_activation": total,
                                }
                            )
                    clear_cuda()
            del cache
            clear_cuda()
    return rows


def summarize_position_rows(rows: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, int, int, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        key = (str(row["family_name"]), str(row["side"]), int(row["layer"]), int(row["k"]), str(row["region"]))
        grouped[key].append(dict(row))
    out: list[dict[str, object]] = []
    for key, grow in sorted(grouped.items()):
        out.append(
            {
                "family_name": key[0],
                "side": key[1],
                "layer": key[2],
                "k": key[3],
                "region": key[4],
                "mean_activation_mass": float(np.mean([float(row["activation_mass"]) for row in grow])),
                "mean_activation_share": float(np.mean([float(row["activation_share"]) for row in grow])),
                "n_samples": len(grow),
            }
        )
    return out


def plot_position_flow(rows: Sequence[dict[str, object]], top_families: Sequence[tuple[FamilyEvalBundle, int]], path: Path) -> None:
    if not rows:
        return
    configure_matplotlib()
    active_side_by_family = {bundle.family_name: bundle.active_side for bundle, _k in top_families}
    family_names = [bundle.family_name for bundle, _k in top_families]
    fig, axes = plt.subplots(len(family_names), 1, figsize=(8.5, 3.2 * len(family_names)), sharex=True)
    if len(family_names) == 1:
        axes = [axes]
    for ax, family_name in zip(axes, family_names):
        active_side = active_side_by_family[family_name]
        family_rows = [
            row
            for row in rows
            if str(row["family_name"]) == family_name and str(row["side"]) == active_side and str(row["region"]) in {"prediction_position", "task_desc", "schema", "verb"}
        ]
        layers = sorted({int(row["layer"]) for row in family_rows})
        for region, color in (
            ("prediction_position", "#1b6ca8"),
            ("task_desc", "#c65d00"),
            ("schema", "#5aa469"),
            ("verb", "#b22222"),
        ):
            values = [
                float(
                    next(
                        row["mean_activation_share"]
                        for row in family_rows
                        if int(row["layer"]) == layer and str(row["region"]) == region
                    )
                )
                for layer in layers
            ]
            ax.plot(layers, values, marker="o", linewidth=2.0, color=color, label=region)
        ax.set_ylabel(f"{family_name}\nshare")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, ncol=2)
    axes[-1].set_xlabel("layer")
    fig.suptitle("Family Position Flow")
    ensure_dir(path.parent)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def make_head_patch_hook(heads: Sequence[int], source_cpu: torch.Tensor):
    def hook_fn(value: torch.Tensor, hook):  # noqa: ANN001
        out = value.clone()
        src = source_cpu.to(device=value.device, dtype=value.dtype)
        for head in heads:
            out[:, -1, head, :] = src[:, -1, head, :]
        return out

    return hook_fn


def make_head_zero_hook(heads: Sequence[int]):
    def hook_fn(value: torch.Tensor, hook):  # noqa: ANN001
        out = value.clone()
        for head in heads:
            out[:, -1, head, :] = 0
        return out

    return hook_fn


def make_mlp_out_patch_hook(source_cpu: torch.Tensor):
    def hook_fn(value: torch.Tensor, hook):  # noqa: ANN001
        out = value.clone()
        src = source_cpu.to(device=value.device, dtype=value.dtype)
        out[:, -1, :] = src[:, -1, :]
        return out

    return hook_fn


def make_mlp_out_zero_hook():
    def hook_fn(value: torch.Tensor, hook):  # noqa: ANN001
        out = value.clone()
        out[:, -1, :] = 0
        return out

    return hook_fn


def run_combo_interventions(
    *,
    model,
    pair_cache,
    gate_direction: torch.Tensor,
    tool_token_id: int,
    combos: Sequence[ComboSpec],
) -> list[dict[str, object]]:
    gate_hook_name = f"blocks.{PATCH_LAYER}.hook_resid_pre"
    baseline_clean_gate = direction_scores(pair_cache.clean_resid, gate_direction)
    baseline_corrupt_gate = direction_scores(pair_cache.corrupt_resid, gate_direction)
    baseline_clean_top1 = pair_cache.baseline_clean["top1"]
    baseline_corrupt_top1 = pair_cache.baseline_corrupt["top1"]
    baseline_clean_logit = pair_cache.baseline_clean["tool_logit"]
    baseline_corrupt_logit = pair_cache.baseline_corrupt["tool_logit"]
    combo_results: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    hook_names = sorted(
        {gate_hook_name}
        | {f"blocks.{layer}.attn.hook_z" for combo in combos for layer, _head in combo.heads}
        | {f"blocks.{layer}.hook_mlp_out" for combo in combos for layer in combo.mlp_layers}
    )

    progress = tqdm(pair_cache.pair_batches, desc="Combo interventions", dynamic_ncols=True)
    for batch in progress:
        clean_tokens = batch.clean_tokens_cpu.to(model.W_U.device)
        corrupt_tokens = batch.corrupt_tokens_cpu.to(model.W_U.device)
        with torch.no_grad():
            _, clean_cache = model.run_with_cache(clean_tokens, names_filter=lambda name: name in hook_names)
            _, corrupt_cache = model.run_with_cache(corrupt_tokens, names_filter=lambda name: name in hook_names)
        clean_cache_cpu = {name: clean_cache[name].detach().cpu() for name in hook_names if name in clean_cache}
        corrupt_cache_cpu = {name: corrupt_cache[name].detach().cpu() for name in hook_names if name in corrupt_cache}

        for combo in combos:
            heads_by_layer: dict[int, list[int]] = defaultdict(list)
            for layer, head in combo.heads:
                heads_by_layer[layer].append(head)

            # clean -> corrupt patch
            patch_hooks = []
            for layer, heads in heads_by_layer.items():
                patch_hooks.append((f"blocks.{layer}.attn.hook_z", make_head_patch_hook(heads, clean_cache_cpu[f"blocks.{layer}.attn.hook_z"])))
            for layer in combo.mlp_layers:
                patch_hooks.append((f"blocks.{layer}.hook_mlp_out", make_mlp_out_patch_hook(clean_cache_cpu[f"blocks.{layer}.hook_mlp_out"])))
            capture_patch: dict[str, torch.Tensor] = {}
            patch_hooks.append((gate_hook_name, make_resid_last_capture(capture_patch, "gate")))
            with torch.no_grad():
                patch_logits = model.run_with_hooks(corrupt_tokens, fwd_hooks=patch_hooks)
            patch_tool_logit = patch_logits[:, -1, tool_token_id].detach().cpu().float()
            patch_top1 = patch_logits[:, -1, :].argmax(dim=-1).detach().cpu()
            patch_gate = torch.mv(capture_patch["gate"].float(), gate_direction.float())
            patch_delta = baseline_clean_gate[batch.indices] - patch_gate
            baseline_delta = baseline_clean_gate[batch.indices] - baseline_corrupt_gate[batch.indices]
            combo_results[(combo.name, "clean_to_corrupt_patch")]["delta_g24_shift"].extend((patch_delta - baseline_delta).tolist())
            combo_results[(combo.name, "clean_to_corrupt_patch")]["tool_logit_shift"].extend((patch_tool_logit - baseline_corrupt_logit[batch.indices]).tolist())
            combo_results[(combo.name, "clean_to_corrupt_patch")]["top1"].extend((patch_top1 == tool_token_id).float().tolist())
            combo_results[(combo.name, "clean_to_corrupt_patch")]["strict_flip"].extend(((baseline_corrupt_top1[batch.indices] != tool_token_id) & (patch_top1 == tool_token_id)).float().tolist())

            # corrupt -> clean patch
            reverse_hooks = []
            for layer, heads in heads_by_layer.items():
                reverse_hooks.append((f"blocks.{layer}.attn.hook_z", make_head_patch_hook(heads, corrupt_cache_cpu[f"blocks.{layer}.attn.hook_z"])))
            for layer in combo.mlp_layers:
                reverse_hooks.append((f"blocks.{layer}.hook_mlp_out", make_mlp_out_patch_hook(corrupt_cache_cpu[f"blocks.{layer}.hook_mlp_out"])))
            capture_reverse: dict[str, torch.Tensor] = {}
            reverse_hooks.append((gate_hook_name, make_resid_last_capture(capture_reverse, "gate")))
            with torch.no_grad():
                reverse_logits = model.run_with_hooks(clean_tokens, fwd_hooks=reverse_hooks)
            reverse_tool_logit = reverse_logits[:, -1, tool_token_id].detach().cpu().float()
            reverse_top1 = reverse_logits[:, -1, :].argmax(dim=-1).detach().cpu()
            reverse_gate = torch.mv(capture_reverse["gate"].float(), gate_direction.float())
            reverse_delta = reverse_gate - baseline_corrupt_gate[batch.indices]
            combo_results[(combo.name, "corrupt_to_clean_patch")]["delta_g24_shift"].extend((reverse_delta - baseline_delta).tolist())
            combo_results[(combo.name, "corrupt_to_clean_patch")]["tool_logit_shift"].extend((reverse_tool_logit - baseline_clean_logit[batch.indices]).tolist())
            combo_results[(combo.name, "corrupt_to_clean_patch")]["top1"].extend((reverse_top1 == tool_token_id).float().tolist())
            combo_results[(combo.name, "corrupt_to_clean_patch")]["strict_flip"].extend(((baseline_clean_top1[batch.indices] == tool_token_id) & (reverse_top1 != tool_token_id)).float().tolist())

            # clean ablation
            zero_hooks = []
            for layer, heads in heads_by_layer.items():
                zero_hooks.append((f"blocks.{layer}.attn.hook_z", make_head_zero_hook(heads)))
            for layer in combo.mlp_layers:
                zero_hooks.append((f"blocks.{layer}.hook_mlp_out", make_mlp_out_zero_hook()))
            capture_zero: dict[str, torch.Tensor] = {}
            zero_hooks.append((gate_hook_name, make_resid_last_capture(capture_zero, "gate")))
            with torch.no_grad():
                zero_logits = model.run_with_hooks(clean_tokens, fwd_hooks=zero_hooks)
            zero_tool_logit = zero_logits[:, -1, tool_token_id].detach().cpu().float()
            zero_top1 = zero_logits[:, -1, :].argmax(dim=-1).detach().cpu()
            zero_gate = torch.mv(capture_zero["gate"].float(), gate_direction.float())
            zero_delta = zero_gate - baseline_corrupt_gate[batch.indices]
            combo_results[(combo.name, "clean_ablate")]["delta_g24_shift"].extend((zero_delta - baseline_delta).tolist())
            combo_results[(combo.name, "clean_ablate")]["tool_logit_shift"].extend((zero_tool_logit - baseline_clean_logit[batch.indices]).tolist())
            combo_results[(combo.name, "clean_ablate")]["top1"].extend((zero_top1 == tool_token_id).float().tolist())
            combo_results[(combo.name, "clean_ablate")]["strict_flip"].extend(((baseline_clean_top1[batch.indices] == tool_token_id) & (zero_top1 != tool_token_id)).float().tolist())

        del clean_cache, corrupt_cache, clean_cache_cpu, corrupt_cache_cpu
        clear_cuda()

    rows: list[dict[str, object]] = []
    combo_map = {combo.name: combo for combo in combos}
    for (combo_name, intervention), store in sorted(combo_results.items()):
        combo = combo_map[combo_name]
        rows.append(
            {
                "combo_name": combo_name,
                "intervention": intervention,
                "heads": json.dumps([f"L{layer}H{head}" for layer, head in combo.heads]),
                "mlp_layers": json.dumps(list(combo.mlp_layers)),
                "mean_delta_g24_shift": float(np.mean(store["delta_g24_shift"])),
                "mean_tool_logit_shift": float(np.mean(store["tool_logit_shift"])),
                "tool_call_top1_rate": float(np.mean(store["top1"])),
                "strict_flip_rate": float(np.mean(store["strict_flip"])),
                "rationale": combo.rationale,
            }
        )
    return rows


def plot_combo_effects(rows: Sequence[dict[str, object]], path: Path) -> None:
    if not rows:
        return
    configure_matplotlib()
    patch_rows = [row for row in rows if str(row["intervention"]) == "clean_to_corrupt_patch"]
    labels = [str(row["combo_name"]) for row in patch_rows]
    vals = np.asarray([float(row["mean_delta_g24_shift"]) for row in patch_rows], dtype=np.float32)
    fig, ax = plt.subplots(figsize=(max(9, len(labels) * 1.4), 4.2))
    ax.bar(np.arange(len(labels)), vals, color="#1b6ca8")
    ax.axhline(0.0, color="#222222", linewidth=1.0)
    ax.set_ylabel("mean Δg24 shift")
    ax.set_xticks(np.arange(len(labels)), labels, rotation=25, ha="right")
    ax.set_title("Candidate Upstream Bridge Combos")
    ax.grid(axis="y", alpha=0.25)
    ensure_dir(path.parent)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def build_combo_specs() -> list[ComboSpec]:
    return [
        ComboSpec(
            name="heads_L20H10_L21H18",
            heads=((20, 10), (21, 18)),
            mlp_layers=(),
            rationale="shared forward/reverse mid-window heads from the legacy circuit summary",
        ),
        ComboSpec(
            name="heads_plus_MLP22",
            heads=((20, 10), (21, 18)),
            mlp_layers=(22,),
            rationale="add the first late MLP writer candidate to the shared mid-window heads",
        ),
        ComboSpec(
            name="MLP22_plus_MLP23",
            heads=(),
            mlp_layers=(22, 23),
            rationale="pure late-MLP bridge motivated by the sharp gate crystallization at L23",
        ),
        ComboSpec(
            name="L23H15_plus_MLP23",
            heads=((23, 15),),
            mlp_layers=(23,),
            rationale="late near-gate head plus the strongest late MLP write",
        ),
        ComboSpec(
            name="full_midwindow_bridge",
            heads=((20, 10), (21, 18), (23, 15)),
            mlp_layers=(22, 23),
            rationale="compact cross-layer bridge spanning the main formation window",
        ),
        ComboSpec(
            name="L24H30_control",
            heads=((24, 30),),
            mlp_layers=(),
            rationale="near-target downstream-ish control; should have little effect on L24 resid_pre gate",
        ),
    ]


def main() -> None:
    args = parse_args()
    ensure_dir(args.output_root)
    cache_root = args.output_root / "cache"
    ensure_dir(cache_root)
    set_seed(args.seed)
    layers = parse_layers(args.layers)
    k_values = parse_ks(args.k_values)
    family_filter = parse_name_filter(args.include_families)
    combo_filter = parse_name_filter(args.include_combos)

    model, tokenizer = load_model_and_tokenizer(model_path=args.model_path, device=args.device)
    gate_bundle = load_gate_bundle(args.pc_bundle)
    gate_direction = gate_bundle["components"][0].detach().cpu().float()
    tool_token_id = get_tool_token_id(tokenizer)
    print(f"[stage] model loaded on {args.device}; discovery={args.discovery_max_pairs}, eval={args.eval_max_pairs}", flush=True)

    # Exp 6: family discovery
    families_raw, raw_feature_rows, token_rows, example_rows = discover_feature_families(
        model=model,
        tokenizer=tokenizer,
        gate_direction=gate_direction,
        dataset_root=args.discovery_root,
        layers=layers,
        max_pairs=args.discovery_max_pairs,
        batch_size=args.batch_size,
        feature_compute_batch_size=args.feature_compute_batch_size,
        candidate_limit_per_side=args.candidate_limit_per_side,
        family_max_features=args.family_max_features,
    )
    preferred_order = ["no_need_non_existence", "execution_request", "analysis_plain_discourse", "schema_boundary_formatting"]
    families = {name: families_raw[name] for name in preferred_order if name in families_raw and families_raw[name]}
    if family_filter:
        families = {name: members for name, members in families.items() if name in family_filter}
    print(
        "[stage] discovered families: "
        + ", ".join(f"{name}={len(members)}" for name, members in families.items()),
        flush=True,
    )
    random_pool = [
        FeatureCandidate(
            family_name="random_pool",
            layer=int(row["layer"]),
            feature_idx=int(row["feature_idx"]),
            kappa=float(row["kappa"]),
            beta=float(row["beta"]),
            delta_activation=float(row["delta_activation"]),
            beta_abs=abs(float(row["beta"])),
            pattern=str(row["pattern"]),
            semantic_score=float(row["semantic_score"]),
            top_tokens=(),
            bottom_tokens=(),
            common_verbs=(),
        )
        for row in raw_feature_rows
        if not str(row["assigned_family"]) and int(row["layer"]) <= args.causal_max_layer
    ]
    catalog_rows = build_family_catalog_rows(families, causal_max_layer=args.causal_max_layer)
    write_csv(
        args.output_root / "exp6_feature_families" / "feature_family_catalog.csv",
        [
            "family_name",
            "role",
            "active_side",
            "n_members",
            "n_causal_members_lte_l23",
            "layers",
            "dominant_patterns",
            "representative_features",
            "representative_verbs",
        ],
        catalog_rows,
    )
    write_csv(
        args.output_root / "exp6_feature_families" / "top_feature_tokens.csv",
        ["family_name", "layer", "feature_idx", "kappa", "beta", "pattern", "top_tokens", "bottom_tokens"],
        token_rows,
    )
    write_csv(
        args.output_root / "exp6_feature_families" / "top_feature_examples.csv",
        ["family_name", "layer", "feature_idx", "kappa", "beta", "pattern", "common_verbs", "top_examples"],
        example_rows,
    )

    # Eval bundles
    eval_bundles, eval_payload, _pair_batches = build_family_eval_bundles(
        model=model,
        tokenizer=tokenizer,
        gate_direction=gate_direction,
        dataset_root=args.eval_root,
        max_pairs=args.eval_max_pairs,
        batch_size=args.batch_size,
        feature_compute_batch_size=args.feature_compute_batch_size,
        families=families,
        causal_max_layer=args.causal_max_layer,
        cache_root=cache_root,
    )
    pair_cache = eval_payload["pair_cache"]
    eval_tool_token_id = tool_token_id
    print(
        "[stage] eval bundles ready: "
        + ", ".join(f"{name}={bundle.n_members}" for name, bundle in eval_bundles.items()),
        flush=True,
    )

    # Reorder bundles globally and prepare random control from all named members.
    eval_bundles = {name: reorder_bundle_global(bundle) for name, bundle in eval_bundles.items()}
    all_named_members = [member for bundle in eval_bundles.values() for member in bundle.members]
    if all_named_members:
        tool_token_id = get_tool_token_id(tokenizer)
        eval_clean_inputs, eval_corrupt_inputs, _ = collect_layer_inputs_and_baseline(
            model,
            pair_cache.pair_batches,
            layers=sorted({member.layer for member in all_named_members}),
            n_samples=len(pair_cache.samples),
            tool_token_id=tool_token_id,
        )
        max_named_size = max(bundle.n_members for bundle in eval_bundles.values())
        eval_bundles["random_matched_control"] = reorder_bundle_global(
            build_random_control_bundle(
                model=model,
                eval_clean_inputs=eval_clean_inputs,
                eval_corrupt_inputs=eval_corrupt_inputs,
                candidate_pool=random_pool if len(random_pool) >= max_named_size else all_named_members,
                n_members=max_named_size,
                feature_compute_batch_size=args.feature_compute_batch_size,
                seed=args.seed + 99,
            )
        )
    else:
        eval_clean_inputs = {}
        eval_corrupt_inputs = {}
    print(
        "[stage] random control ready; running family causality for "
        + ", ".join(eval_bundles.keys()),
        flush=True,
    )

    # Exp 7: family causality
    family_k_rows: list[dict[str, object]] = []
    best_rows_for_plot: list[dict[str, object]] = []
    for family_name, bundle in eval_bundles.items():
        if bundle.n_members == 0:
            continue
        valid_ks = [k for k in k_values if k <= bundle.n_members]
        if not valid_ks:
            valid_ks.append(bundle.n_members)
        valid_ks = sorted(set(valid_ks))
        for intervention, eval_side, mode, source_side in (
            ("clean_ablate", "clean", "ablate", None),
            ("corrupt_ablate", "corrupt", "ablate", None),
            ("corrupt_clean_patch", "corrupt", "inject", "clean"),
        ):
            for k in valid_ks:
                summary, _patched_gate, _patched_tool = evaluate_feature_intervention(
                    model=model,
                    pair_cache=pair_cache,
                    gate_direction=gate_direction,
                    tool_token_id=eval_tool_token_id,
                    family_bundle=bundle,
                    k=k,
                    eval_side=eval_side,
                    mode=mode,
                    source_side=source_side,
                )
                family_k_rows.append(
                    {
                        "family_name": family_name,
                        "role": bundle.role,
                        "active_side": bundle.active_side,
                        "intervention": intervention,
                        "eval_side": eval_side,
                        "mode": mode,
                        "source_side": source_side or "",
                        "k": k,
                        "selected_k": k,
                        "n_family_members": bundle.n_members,
                        **summary,
                    }
                )
        print(f"[stage] finished family causality for {family_name}", flush=True)

    best_rows = pick_best_rows_for_families(family_k_rows)
    write_csv(
        args.output_root / "exp7_family_causality" / "family_k_sweep.csv",
        [
            "family_name",
            "role",
            "active_side",
            "intervention",
            "eval_side",
            "mode",
            "source_side",
            "k",
            "selected_k",
            "n_family_members",
            "mean_patched_delta_g24",
            "mean_baseline_delta_g24",
            "mean_delta_g24_shift",
            "mean_delta_g24_fraction",
            "mean_side_gate_score",
            "mean_tool_logit_shift",
            "tool_call_top1_rate",
            "strict_flip_rate",
        ],
        family_k_rows,
    )
    write_csv(
        args.output_root / "exp7_family_causality" / "family_gate_effects.csv",
        [
            "family_name",
            "role",
            "active_side",
            "intervention",
            "eval_side",
            "mode",
            "source_side",
            "k",
            "selected_k",
            "n_family_members",
            "mean_patched_delta_g24",
            "mean_baseline_delta_g24",
            "mean_delta_g24_shift",
            "mean_delta_g24_fraction",
            "mean_side_gate_score",
            "mean_tool_logit_shift",
            "tool_call_top1_rate",
            "strict_flip_rate",
        ],
        best_rows,
    )
    plot_family_effects(best_rows, args.output_root / "exp7_family_causality" / "plot_family_gate_effects.pdf")
    print("[stage] exp7 written", flush=True)

    # Exp 8: mediation on top two named families.
    mediation_source_rows = [
        row
        for row in best_rows
        if str(row["family_name"]) in eval_bundles and str(row["family_name"]) != "random_matched_control"
    ]
    mediation_source_rows = sorted(mediation_source_rows, key=lambda row: abs(float(row["mean_delta_g24_shift"])), reverse=True)
    mediation_rows: list[dict[str, object]] = []
    used_families: set[str] = set()
    for row in mediation_source_rows:
        family_name = str(row["family_name"])
        if family_name in used_families:
            continue
        used_families.add(family_name)
        mediation_rows.append(
            run_gate_mediation(
                model=model,
                pair_cache=pair_cache,
                gate_bundle=gate_bundle,
                gate_direction=gate_direction,
                tool_token_id=eval_tool_token_id,
                family_bundle=eval_bundles[family_name],
                intervention_row=row,
            )
        )
        if len(mediation_rows) >= 2:
            break
    write_csv(
        args.output_root / "exp8_gate_mediation" / "mediation_table.csv",
        [
            "family_name",
            "intervention",
            "eval_side",
            "k",
            "baseline_delta_g24",
            "family_only_delta_g24",
            "gate_restore_delta_g24",
            "full_restore_delta_g24",
            "gate_restore_closed_fraction",
            "full_restore_closed_fraction",
            "baseline_tool_logit",
            "family_only_tool_logit",
            "gate_restore_tool_logit",
            "full_restore_tool_logit",
            "gate_restore_logit_closed_fraction",
            "full_restore_logit_closed_fraction",
            "family_only_top1_rate",
            "gate_restore_top1_rate",
            "full_restore_top1_rate",
        ],
        mediation_rows,
    )
    plot_mediation(mediation_rows, args.output_root / "exp8_gate_mediation" / "plot_gate_mediation.pdf")
    print("[stage] exp8 written", flush=True)

    # Exp 9: position tracing for strongest two named families.
    trace_targets: list[tuple[FamilyEvalBundle, int]] = []
    trace_seen: set[str] = set()
    for row in mediation_source_rows:
        family_name = str(row["family_name"])
        if family_name not in eval_bundles or family_name in trace_seen:
            continue
        trace_seen.add(family_name)
        trace_targets.append((eval_bundles[family_name], int(row["k"])))
        if len(trace_targets) >= 2:
            break
    position_summary_rows: list[dict[str, object]] = []
    if not args.skip_position_trace:
        position_detail_rows = run_position_trace(
            model=model,
            pair_cache=pair_cache,
            top_families=trace_targets,
            batch_size=args.batch_size,
        ) if trace_targets else []
        position_summary_rows = summarize_position_rows(position_detail_rows)
        write_csv(
            args.output_root / "exp9_feature_position_trace" / "feature_position_stats.csv",
            [
                "family_name",
                "side",
                "layer",
                "k",
                "region",
                "mean_activation_mass",
                "mean_activation_share",
                "n_samples",
            ],
            position_summary_rows,
        )
        plot_position_flow(position_summary_rows, trace_targets, args.output_root / "exp9_feature_position_trace" / "plot_feature_position_flow.pdf")
        print("[stage] exp9 written", flush=True)

    # Exp 10: candidate bridge combos.
    combo_specs = build_combo_specs()
    if combo_filter:
        combo_specs = [combo for combo in combo_specs if combo.name in combo_filter]
    combo_rows = run_combo_interventions(
        model=model,
        pair_cache=pair_cache,
        gate_direction=gate_direction,
        tool_token_id=eval_tool_token_id,
        combos=combo_specs,
    )
    write_csv(
        args.output_root / "exp10_upstream_bridge_combos" / "combo_gate_effects.csv",
        [
            "combo_name",
            "intervention",
            "heads",
            "mlp_layers",
            "mean_delta_g24_shift",
            "mean_tool_logit_shift",
            "tool_call_top1_rate",
            "strict_flip_rate",
            "rationale",
        ],
        combo_rows,
    )
    plot_combo_effects(combo_rows, args.output_root / "exp10_upstream_bridge_combos" / "plot_combo_gate_effects.pdf")
    print("[stage] exp10 written", flush=True)

    # Summaries
    family_summary_lines = [
        "# Exp 6: Candidate Upstream Feature Families",
        "",
        f"- Discovery split: `{args.discovery_root}` first `{args.discovery_max_pairs}` pairs.",
        f"- Eval split for downstream causal experiments: `{args.eval_root}` first `{args.eval_max_pairs}` pairs.",
        "",
        "## Discovered Families",
    ]
    for row in catalog_rows:
        family_summary_lines.append(
            f"- `{row['family_name']}`: members `{row['n_members']}`, causal members `<=L23` = `{row['n_causal_members_lte_l23']}`, "
            f"layers `{row['layers']}`, representative `{row['representative_features']}`"
        )
    write_text(args.output_root / "exp6_feature_families" / "summary.md", "\n".join(family_summary_lines))

    exp7_lines = [
        "# Exp 7: Family -> Gate Causality",
        "",
        f"- Eval split: `{args.eval_root}` first `{args.eval_max_pairs}` pairs.",
        f"- K sweep: `{k_values}`.",
        "",
        "## Strongest Rows",
    ]
    for row in mediation_source_rows[:8]:
        exp7_lines.append(
            f"- `{row['family_name']}` / `{row['intervention']}` / `k={row['k']}`: "
            f"`Δg24 shift={float(row['mean_delta_g24_shift']):+.4f}`, "
            f"`top1={float(row['tool_call_top1_rate']):.2%}`, `strict flip={float(row['strict_flip_rate']):.2%}`"
        )
    write_text(args.output_root / "exp7_family_causality" / "summary.md", "\n".join(exp7_lines))

    exp8_lines = [
        "# Exp 8: Gate Mediation",
        "",
    ]
    for row in mediation_rows:
        exp8_lines.append(
            f"- `{row['family_name']}` / `{row['intervention']}`: "
            f"gate-restore closes `{float(row['gate_restore_closed_fraction']):.2%}` of the Δg24 gap; "
            f"full-state closes `{float(row['full_restore_closed_fraction']):.2%}`."
        )
    write_text(args.output_root / "exp8_gate_mediation" / "summary.md", "\n".join(exp8_lines))

    exp9_lines = [
        "# Exp 9: Feature Position Trace",
        "",
    ]
    if args.skip_position_trace:
        exp9_lines.append("- skipped in this rerun")
    else:
        for family_bundle, k in trace_targets:
            active_side = family_bundle.active_side
            pred_rows = [
                row
                for row in position_summary_rows
                if str(row["family_name"]) == family_bundle.family_name and str(row["side"]) == active_side and str(row["region"]) == "prediction_position"
            ]
            if pred_rows:
                pred_rows = sorted(pred_rows, key=lambda row: int(row["layer"]))
                exp9_lines.append(
                    f"- `{family_bundle.family_name}` ({active_side}, k={k}): prediction-position share "
                    + " -> ".join(f"L{int(row['layer'])}:{float(row['mean_activation_share']):.3f}" for row in pred_rows)
                )
    write_text(args.output_root / "exp9_feature_position_trace" / "summary.md", "\n".join(exp9_lines))

    exp10_lines = [
        "# Exp 10: Upstream Bridge Combos",
        "",
        "## Strongest clean->corrupt patches",
    ]
    for row in sorted(
        [row for row in combo_rows if str(row["intervention"]) == "clean_to_corrupt_patch"],
        key=lambda row: abs(float(row["mean_delta_g24_shift"])),
        reverse=True,
    )[:8]:
        exp10_lines.append(
            f"- `{row['combo_name']}`: `Δg24 shift={float(row['mean_delta_g24_shift']):+.4f}`, "
            f"`top1={float(row['tool_call_top1_rate']):.2%}`, `strict flip={float(row['strict_flip_rate']):.2%}`"
        )
    write_text(args.output_root / "exp10_upstream_bridge_combos" / "summary.md", "\n".join(exp10_lines))


if __name__ == "__main__":
    main()
