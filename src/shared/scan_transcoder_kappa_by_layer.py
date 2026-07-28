#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gc
import json
from pathlib import Path

import torch
from safetensors.torch import load_file
from tqdm.auto import tqdm

from compute_transcoder_table3_metrics import clear_cuda, collect_layer_inputs, mean_dense_features, top_rows_for_mask
from multiscale_common import build_pair_batches, load_model_and_tokenizer, load_sample_pairs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scan per-layer Qwen3 Transcoder kappa ratios for a contiguous layer range.")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--transcoder-path", type=Path, required=True)
    parser.add_argument("--pc-bundle", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--size-label", type=str, required=True)
    parser.add_argument("--layer-start", type=int, required=True)
    parser.add_argument("--layer-end", type=int, required=True)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--direction", choices=("mean-diff", "pc1"), default="mean-diff")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--feature-batch-size", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    layers = list(range(args.layer_start, args.layer_end + 1))
    model, tokenizer, tool_token_id = load_model_and_tokenizer(model_path=args.model_path, device=args.device)
    pairs = load_sample_pairs(model, dataset_root=args.dataset_root, split=args.split, max_pairs=0)
    pair_batches = build_pair_batches(pairs, batch_size=args.batch_size)

    clean_inputs, corrupt_inputs = collect_layer_inputs(
        model,
        pair_batches,
        layers=layers,
        n_pairs=len(pairs),
    )

    gate_bundle = torch.load(args.pc_bundle, map_location="cpu", weights_only=False)
    if args.direction == "mean-diff":
        direction_raw = gate_bundle.get("mean_diff")
        if direction_raw is None:
            raise KeyError(f"{args.pc_bundle} does not contain mean_diff.")
        if not isinstance(direction_raw, torch.Tensor):
            direction_raw = torch.tensor(direction_raw)
        direction = direction_raw.detach().cpu().float().view(-1)
    else:
        direction = gate_bundle["components"][0].detach().cpu().float().view(-1)
    direction_norm = float(direction.norm().item())
    direction = direction / max(direction_norm, 1e-12)
    tool_vector = model.W_U[:, tool_token_id].detach().cpu().float()

    layer_rows: list[dict[str, object]] = []
    for layer in tqdm(layers, desc="Scanning per-layer kappa", dynamic_ncols=True):
        weights = load_file(str(args.transcoder_path / f"layer_{layer}.safetensors"))
        W_enc = weights["W_enc"].detach().cpu()
        b_enc = weights["b_enc"].detach().cpu()
        W_dec = weights["W_dec"].detach().cpu().float()

        mean_clean = mean_dense_features(
            clean_inputs[layer],
            W_enc,
            b_enc,
            device=model.W_U.device,
            compute_batch_size=args.feature_batch_size,
        )
        mean_corrupt = mean_dense_features(
            corrupt_inputs[layer],
            W_enc,
            b_enc,
            device=model.W_U.device,
            compute_batch_size=args.feature_batch_size,
        )

        delta = mean_clean - mean_corrupt
        beta_mu = torch.mv(W_dec, direction)
        tool_proj = torch.mv(W_dec, tool_vector)
        kappa = delta * beta_mu
        corrupt_mask = (delta < 0) & (beta_mu < 0)
        clean_mask = (delta > 0) & (beta_mu > 0)
        top_corrupt = top_rows_for_mask(
            layer=layer,
            score=kappa,
            delta=delta,
            beta_mu=beta_mu,
            tool_proj=tool_proj,
            mask=corrupt_mask,
            limit=args.top_k,
        )
        top_clean = top_rows_for_mask(
            layer=layer,
            score=kappa,
            delta=delta,
            beta_mu=beta_mu,
            tool_proj=tool_proj,
            mask=clean_mask,
            limit=args.top_k,
        )
        top_corrupt.sort(key=lambda row: float(row["abs_kappa"]), reverse=True)
        top_clean.sort(key=lambda row: float(row["abs_kappa"]), reverse=True)
        kappa_c = float(sum(float(row["abs_kappa"]) for row in top_corrupt[: args.top_k]))
        kappa_p = float(sum(float(row["abs_kappa"]) for row in top_clean[: args.top_k]))
        layer_rows.append(
            {
                "layer": layer,
                "n_pairs": len(pairs),
                "kappa_corrupt_higher_topk_abs_sum": kappa_c,
                "kappa_clean_higher_topk_abs_sum": kappa_p,
                "kappa_ratio_text": f"{kappa_c:.1f} / {kappa_p:.1f}",
                "n_corrupt_features": int(corrupt_mask.sum().item()),
                "n_clean_features": int(clean_mask.sum().item()),
                "direction_norm_before_unit": direction_norm,
                "pc_bundle": str(args.pc_bundle),
            }
        )

        del W_enc, b_enc, W_dec, mean_clean, mean_corrupt, delta, beta_mu, tool_proj, kappa, weights
        clear_cuda()

    args.output_root.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_root / "layer_metrics.csv", layer_rows)
    (args.output_root / "layer_metrics.json").write_text(
        json.dumps(
            {
                "size_label": args.size_label,
                "model_path": str(args.model_path),
                "transcoder_path": str(args.transcoder_path),
                "dataset_root": str(args.dataset_root),
                "split": args.split,
                "tool_token_id": int(tool_token_id),
                "tool_token_text": tokenizer.decode([tool_token_id], clean_up_tokenization_spaces=False),
                "direction": args.direction,
                "layers": layer_rows,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    md_lines = [
        f"# {args.size_label} Per-Layer Kappa Scan",
        "",
        "| Layer | K_corrupt | K_clean | Ratio |",
        "|---:|---:|---:|---|",
    ]
    for row in layer_rows:
        md_lines.append(
            f"| {row['layer']} | {row['kappa_corrupt_higher_topk_abs_sum']:.1f} | {row['kappa_clean_higher_topk_abs_sum']:.1f} | {row['kappa_ratio_text']} |"
        )
    (args.output_root / "layer_metrics.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
