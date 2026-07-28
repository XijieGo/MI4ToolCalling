#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Sequence

import torch
from safetensors.torch import load_file
from tqdm.auto import tqdm

from differential_feature_mechanism import (
    MODEL_PATH,
    TRANSCODER_DIR,
    clear_cuda,
    collect_layer_inputs_and_baseline,
    compute_dense_features,
    ensure_dir,
    load_dataset_metadata,
    write_text,
)
from phase8_common import DEFAULT_PC_BUNDLE, get_tool_token_id, load_gate_bundle, load_model_and_tokenizer, set_seed
from task_attention_path_analysis import build_pair_batches, load_samples


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = PROJECT_ROOT / "results" / "8b_main" / "phase9_gate_interpretability_summaries" / "exp_b_transcoder_gate_features_rerun"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 9 Exp B: gate-aligned transcoder feature analysis.")
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH)
    parser.add_argument("--transcoder-path", type=Path, default=TRANSCODER_DIR)
    parser.add_argument("--dataset-root", type=Path, default=PROJECT_ROOT / "datasets" / "test")
    parser.add_argument("--pc-bundle", type=Path, default=DEFAULT_PC_BUNDLE)
    parser.add_argument("--layers", type=str, default="21-24")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-pairs", type=int, default=300)
    parser.add_argument("--feature-compute-batch-size", type=int, default=32)
    parser.add_argument("--top-features", type=int, default=20)
    parser.add_argument("--token-topk", type=int, default=20)
    parser.add_argument("--example-topk", type=int, default=5)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def parse_layers(raw: str) -> list[int]:
    raw = raw.strip()
    if "-" in raw:
        start_s, end_s = raw.split("-", 1)
        start = int(start_s)
        end = int(end_s)
        return list(range(start, end + 1))
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[dict[str, object]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def append_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[dict[str, object]], *, write_header: bool) -> None:
    ensure_dir(path.parent)
    mode = "w" if write_header else "a"
    with path.open(mode, encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


def alignment_pattern(delta_activation: float, beta: float) -> str:
    if delta_activation >= 0.0 and beta >= 0.0:
        return "clean_higher_write_toward_gate"
    if delta_activation < 0.0 and beta < 0.0:
        return "corrupt_higher_write_away_from_gate"
    if delta_activation >= 0.0 and beta < 0.0:
        return "clean_higher_write_away_from_gate"
    return "corrupt_higher_write_toward_gate"


def decode_tokens(tokenizer, token_ids: Sequence[int]) -> str:
    return json.dumps([tokenizer.decode([int(token_id)]) for token_id in token_ids], ensure_ascii=False)


def topk_feature_rows(rows: Sequence[dict[str, object]], *, positive: bool, limit: int) -> list[dict[str, object]]:
    if positive:
        pool = [row for row in rows if float(row["kappa"]) > 0]
        return sorted(pool, key=lambda row: float(row["kappa"]), reverse=True)[:limit]
    pool = [row for row in rows if float(row["kappa"]) < 0]
    return sorted(pool, key=lambda row: float(row["kappa"]))[:limit]


def build_feature_index(rows: Sequence[dict[str, object]]) -> dict[int, dict[int, dict[str, object]]]:
    out: dict[int, dict[int, dict[str, object]]] = defaultdict(dict)
    for row in rows:
        out[int(row["layer"])][int(row["feature_idx"])] = dict(row)
    return out


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


def sample_payload(meta: dict[str, object], *, side: str, activation: float) -> dict[str, object]:
    first_line_key = "clean_rendered_first_user_line" if side == "clean" else "corrupt_rendered_first_user_line"
    verb_key = "clean_candidate" if side == "clean" else "corrupt_candidate"
    return {
        "sample_id": meta.get("sample_id"),
        "side": side,
        "activation": activation,
        "verb": meta.get(verb_key),
        "language": meta.get("language"),
        "dataset_name": meta.get("dataset_name"),
        "first_user_line": meta.get(first_line_key),
    }


def main() -> None:
    args = parse_args()
    ensure_dir(args.output_root)
    set_seed(args.seed)
    layers = parse_layers(args.layers)

    model, tokenizer = load_model_and_tokenizer(model_path=args.model_path, device=args.device)
    tool_token_id = get_tool_token_id(tokenizer)
    gate_bundle = load_gate_bundle(args.pc_bundle)
    v1 = gate_bundle["components"][0].detach().cpu().float()

    samples = load_samples(args.dataset_root, model, tokenizer, max_pairs=args.max_pairs)
    pair_batches = build_pair_batches(samples, args.batch_size)
    pair_metadata = load_dataset_metadata(args.dataset_root)
    for sample in samples:
        pair_metadata.setdefault(sample.sample_id, {})
        pair_metadata[sample.sample_id]["sample_id"] = sample.sample_id
        pair_metadata[sample.sample_id]["clean_candidate"] = sample.clean_verb
        pair_metadata[sample.sample_id]["corrupt_candidate"] = sample.corrupt_verb

    clean_inputs, corrupt_inputs, baseline = collect_layer_inputs_and_baseline(
        model,
        pair_batches,
        layers=layers,
        n_samples=len(samples),
        tool_token_id=tool_token_id,
    )

    fieldnames = [
        "layer",
        "feature_idx",
        "mean_clean",
        "mean_corrupt",
        "delta_activation",
        "beta",
        "beta_abs",
        "kappa",
        "active_rate_clean",
        "active_rate_corrupt",
        "pattern",
    ]
    feature_csv = args.output_root / "feature_gate_alignment.csv"
    all_rows: list[dict[str, object]] = []
    first_write = True
    layer_candidates: dict[int, list[int]] = defaultdict(list)

    for layer in tqdm(layers, desc="Gate feature rows", dynamic_ncols=True):
        tc_weights = load_file(str(args.transcoder_path / f"layer_{layer}.safetensors"))
        W_enc = tc_weights["W_enc"].detach().cpu()
        b_enc = tc_weights["b_enc"].detach().cpu()
        W_dec = tc_weights["W_dec"].detach().cpu()

        clean_dense = compute_dense_features(
            clean_inputs[layer],
            W_enc,
            b_enc,
            device=model.W_U.device,
            compute_batch_size=args.feature_compute_batch_size,
        )
        corrupt_dense = compute_dense_features(
            corrupt_inputs[layer],
            W_enc,
            b_enc,
            device=model.W_U.device,
            compute_batch_size=args.feature_compute_batch_size,
        )
        beta = torch.mv(W_dec.float(), v1)
        layer_rows = feature_rows_for_layer(layer=layer, beta=beta, clean_dense=clean_dense, corrupt_dense=corrupt_dense)
        append_csv(feature_csv, fieldnames, layer_rows, write_header=first_write)
        first_write = False
        all_rows.extend(layer_rows)

        pos = topk_feature_rows(layer_rows, positive=True, limit=max(args.top_features, 50))
        neg = topk_feature_rows(layer_rows, positive=False, limit=max(args.top_features, 50))
        layer_candidates[layer].extend(int(row["feature_idx"]) for row in pos)
        layer_candidates[layer].extend(int(row["feature_idx"]) for row in neg)

        del clean_dense, corrupt_dense, W_enc, b_enc, W_dec, tc_weights, beta
        clear_cuda()

    top_support = topk_feature_rows(all_rows, positive=True, limit=args.top_features)
    top_oppose = topk_feature_rows(all_rows, positive=False, limit=args.top_features)
    write_csv(args.output_root / "top_gate_supporting_features.csv", fieldnames, top_support)
    write_csv(args.output_root / "top_gate_opposing_features.csv", fieldnames, top_oppose)

    selected_lookup = build_feature_index(top_support + top_oppose)
    token_rows: list[dict[str, object]] = []
    example_rows: list[dict[str, object]] = []

    for layer in tqdm(sorted(selected_lookup.keys()), desc="Feature semantics", dynamic_ncols=True):
        feature_ids = sorted(selected_lookup[layer].keys())
        tc_weights = load_file(str(args.transcoder_path / f"layer_{layer}.safetensors"))
        W_enc = tc_weights["W_enc"].detach().cpu()
        b_enc = tc_weights["b_enc"].detach().cpu()
        W_dec = tc_weights["W_dec"].detach().cpu()

        clean_dense = compute_dense_features(
            clean_inputs[layer],
            W_enc,
            b_enc,
            device=model.W_U.device,
            compute_batch_size=args.feature_compute_batch_size,
        )
        corrupt_dense = compute_dense_features(
            corrupt_inputs[layer],
            W_enc,
            b_enc,
            device=model.W_U.device,
            compute_batch_size=args.feature_compute_batch_size,
        )

        decoder_rows = W_dec[torch.tensor(feature_ids, dtype=torch.long)].to(device=model.W_U.device, dtype=torch.bfloat16)
        with torch.no_grad():
            token_scores = (decoder_rows @ model.W_U.to(dtype=torch.bfloat16)).detach().cpu().float()

        for local_idx, feature_idx in enumerate(feature_ids):
            row = selected_lookup[layer][feature_idx]
            top_ids = torch.topk(token_scores[local_idx], k=args.token_topk).indices.tolist()
            bottom_ids = torch.topk(token_scores[local_idx], k=args.token_topk, largest=False).indices.tolist()
            token_rows.append(
                {
                    "layer": layer,
                    "feature_idx": feature_idx,
                    "kappa": row["kappa"],
                    "beta": row["beta"],
                    "pattern": row["pattern"],
                    "top_tokens": decode_tokens(tokenizer, top_ids),
                    "bottom_tokens": decode_tokens(tokenizer, bottom_ids),
                }
            )

            clean_vals = clean_dense[:, feature_idx]
            corrupt_vals = corrupt_dense[:, feature_idx]
            combined: list[tuple[float, str, str]] = []
            for sample_idx, sample in enumerate(samples):
                combined.append((float(clean_vals[sample_idx].item()), "clean", sample.sample_id))
                combined.append((float(corrupt_vals[sample_idx].item()), "corrupt", sample.sample_id))
            combined.sort(key=lambda item: item[0], reverse=True)
            top_examples = []
            verb_counter: Counter[str] = Counter()
            for activation, side, sample_id in combined[: args.example_topk]:
                meta = pair_metadata.get(sample_id, {})
                payload = sample_payload(meta, side=side, activation=activation)
                top_examples.append(payload)
                verb = str(payload.get("verb") or "")
                if verb:
                    verb_counter[verb] += 1
            example_rows.append(
                {
                    "layer": layer,
                    "feature_idx": feature_idx,
                    "kappa": row["kappa"],
                    "beta": row["beta"],
                    "pattern": row["pattern"],
                    "common_verbs": json.dumps([verb for verb, _count in verb_counter.most_common(3)], ensure_ascii=False),
                    "top_examples": json.dumps(top_examples, ensure_ascii=False),
                }
            )

        del clean_dense, corrupt_dense, W_enc, b_enc, W_dec, tc_weights, decoder_rows, token_scores
        clear_cuda()

    write_csv(
        args.output_root / "top_feature_tokens.csv",
        ["layer", "feature_idx", "kappa", "beta", "pattern", "top_tokens", "bottom_tokens"],
        token_rows,
    )
    write_csv(
        args.output_root / "top_feature_examples.csv",
        ["layer", "feature_idx", "kappa", "beta", "pattern", "common_verbs", "top_examples"],
        example_rows,
    )

    support_patterns = Counter(str(row["pattern"]) for row in top_support)
    oppose_patterns = Counter(str(row["pattern"]) for row in top_oppose)
    lines = [
        "# Phase 9 Exp B: Gate-Aligned Transcoder Features",
        "",
        "## Setup",
        f"- Eval split: `{args.dataset_root}` with `{len(samples)}` held-out pairs.",
        f"- Model: `{args.model_path}`.",
        f"- Transcoder: `{args.transcoder_path}`.",
        f"- Layers: `{layers}`.",
        f"- Gate direction: PC1 from `{args.pc_bundle}`.",
        f"- Clean baseline `<tool_call>` top-1: {float((baseline['clean_top1'] == tool_token_id).float().mean().item()):.2%}.",
        f"- Corrupt baseline `<tool_call>` top-1: {float((baseline['corrupt_top1'] == tool_token_id).float().mean().item()):.2%}.",
        "",
        "## Top Gate-Supporting Features",
    ]
    for row in top_support[:10]:
        lines.append(
            f"- L{int(row['layer'])} F{int(row['feature_idx'])}: kappa={float(row['kappa']):+.4f}, "
            f"beta={float(row['beta']):+.4f}, delta_act={float(row['delta_activation']):+.4f}, "
            f"pattern={row['pattern']}"
        )
    lines.extend(
        [
            "",
            "## Top Gate-Opposing Features",
        ]
    )
    for row in top_oppose[:10]:
        lines.append(
            f"- L{int(row['layer'])} F{int(row['feature_idx'])}: kappa={float(row['kappa']):+.4f}, "
            f"beta={float(row['beta']):+.4f}, delta_act={float(row['delta_activation']):+.4f}, "
            f"pattern={row['pattern']}"
        )
    lines.extend(
        [
            "",
            "## Pattern Counts",
            f"- Supporting top-{args.top_features}: {dict(support_patterns)}",
            f"- Opposing top-{args.top_features}: {dict(oppose_patterns)}",
        ]
    )
    write_text(args.output_root / "summary.md", "\n".join(lines))
    (args.output_root / "run_metadata.json").write_text(
        json.dumps(
            {
                "model_path": str(args.model_path),
                "transcoder_path": str(args.transcoder_path),
                "dataset_root": str(args.dataset_root),
                "pc_bundle": str(args.pc_bundle),
                "layers": layers,
                "n_pairs": len(samples),
                "seed": args.seed,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
