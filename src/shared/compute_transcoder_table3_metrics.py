#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gc
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from tqdm.auto import tqdm

from multiscale_common import (
    DEFAULT_DATASET_ROOT,
    build_pair_batches,
    load_model_and_tokenizer,
    load_sample_pairs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute Transcoder kappa and late-writer metrics from a current-run gate bundle.")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--transcoder-path", type=Path, required=True)
    parser.add_argument("--pc-bundle", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--size-label", type=str, required=True)
    parser.add_argument("--formation-start", type=int, default=None, help="Explicit inclusive formation-window start (legacy/manual mode).")
    parser.add_argument("--formation-end", type=int, default=None, help="Explicit inclusive formation-window end (legacy/manual mode).")
    parser.add_argument(
        "--formation-pre-layers",
        type=int,
        default=None,
        help="Fresh-run mode: use this many layers immediately before the gate bundle's localized commitment layer.",
    )
    parser.add_argument("--late-start", type=int, default=None)
    parser.add_argument(
        "--late-start-offset",
        type=int,
        default=1,
        help="When --late-start is omitted, begin the late-readout window this many layers after the localized commitment layer.",
    )
    parser.add_argument("--late-end", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--direction", choices=("mean-diff", "pc1"), default="mean-diff")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--feature-batch-size", type=int, default=16)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def clear_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def layer_range(start: int, end: int) -> list[int]:
    if end < start:
        raise ValueError(f"Invalid layer range: {start}..{end}")
    return list(range(start, end + 1))


def resolve_windows(args: argparse.Namespace, *, commitment_layer: int, n_layers: int) -> tuple[list[int], list[int], str]:
    """Resolve explicit legacy windows or a pre-registered relative protocol."""

    explicit_start = args.formation_start
    explicit_end = args.formation_end
    if args.formation_pre_layers is not None:
        if explicit_start is not None or explicit_end is not None:
            raise ValueError("Use either --formation-pre-layers or --formation-start/--formation-end, not both")
        if args.formation_pre_layers <= 0:
            raise ValueError("--formation-pre-layers must be positive")
        formation_start = commitment_layer - args.formation_pre_layers
        formation_end = commitment_layer - 1
        protocol = f"relative_precommit_{args.formation_pre_layers}"
    else:
        if explicit_start is None or explicit_end is None:
            raise ValueError("Provide --formation-pre-layers or both --formation-start and --formation-end")
        formation_start = explicit_start
        formation_end = explicit_end
        protocol = "explicit"
    if formation_start < 0 or formation_end < formation_start or formation_end >= n_layers:
        raise ValueError(
            f"Invalid formation window {formation_start}..{formation_end} for commitment layer L{commitment_layer} and {n_layers} layers"
        )

    late_start = args.late_start if args.late_start is not None else commitment_layer + args.late_start_offset
    late_end = args.late_end if args.late_end is not None else n_layers - 1
    if late_start < 0 or late_end < late_start or late_end >= n_layers:
        raise ValueError(f"Invalid late window {late_start}..{late_end} for a {n_layers}-layer model")
    return layer_range(formation_start, formation_end), layer_range(late_start, late_end), protocol


def collect_layer_inputs(
    model,
    pair_batches,
    *,
    layers: list[int],
    n_pairs: int,
) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor]]:
    if hasattr(model, "set_use_hook_mlp_in"):
        model.set_use_hook_mlp_in(True)
    if hasattr(model, "cfg") and hasattr(model.cfg, "use_hook_mlp_in"):
        model.cfg.use_hook_mlp_in = True

    hook_names = [f"blocks.{layer}.hook_mlp_in" for layer in layers]
    d_model = int(model.cfg.d_model)
    clean_inputs = {layer: torch.empty((n_pairs, d_model), dtype=torch.bfloat16) for layer in layers}
    corrupt_inputs = {layer: torch.empty((n_pairs, d_model), dtype=torch.bfloat16) for layer in layers}

    progress = tqdm(pair_batches, desc="Collecting MLP inputs", dynamic_ncols=True)
    for batch in progress:
        tokens = torch.cat([batch.clean_tokens_cpu, batch.corrupt_tokens_cpu], dim=0).to(model.W_U.device)
        batch_size = int(batch.clean_tokens_cpu.shape[0])
        with torch.no_grad():
            _logits, cache = model.run_with_cache(tokens, names_filter=lambda name: name in hook_names)
        for layer in layers:
            hook_name = f"blocks.{layer}.hook_mlp_in"
            layer_inputs = cache[hook_name][:, -1, :].detach().cpu().to(torch.bfloat16)
            clean_inputs[layer][batch.indices] = layer_inputs[:batch_size]
            corrupt_inputs[layer][batch.indices] = layer_inputs[batch_size:]
        del tokens, cache, layer_inputs
        clear_cuda()

    return clean_inputs, corrupt_inputs


def mean_dense_features(
    inputs: torch.Tensor,
    W_enc_cpu: torch.Tensor,
    b_enc_cpu: torch.Tensor,
    *,
    device: torch.device,
    compute_batch_size: int,
) -> torch.Tensor:
    W_enc = W_enc_cpu.to(device=device, dtype=torch.bfloat16)
    b_enc = b_enc_cpu.to(device=device, dtype=torch.bfloat16)
    total = torch.zeros(W_enc.shape[0], dtype=torch.float32)
    n_samples = int(inputs.shape[0])
    for start in range(0, n_samples, compute_batch_size):
        end = min(start + compute_batch_size, n_samples)
        batch_in = inputs[start:end].to(device=device, dtype=torch.bfloat16)
        with torch.no_grad():
            batch_out = torch.relu(F.linear(batch_in, W_enc, b_enc))
        total += batch_out.detach().cpu().float().sum(dim=0)
        del batch_in, batch_out
    del W_enc, b_enc
    clear_cuda()
    return total / max(n_samples, 1)


def top_rows_for_mask(
    *,
    layer: int,
    score: torch.Tensor,
    delta: torch.Tensor,
    beta_mu: torch.Tensor,
    tool_proj: torch.Tensor,
    mask: torch.Tensor,
    limit: int,
) -> list[dict[str, object]]:
    feature_ids = torch.nonzero(mask, as_tuple=False).squeeze(-1)
    if feature_ids.numel() == 0:
        return []
    order = torch.argsort(score[feature_ids].abs(), descending=True)
    picked = feature_ids[order[:limit]]
    rows: list[dict[str, object]] = []
    for feature_idx in picked.tolist():
        rows.append(
            {
                "layer": layer,
                "feature_idx": int(feature_idx),
                "delta_activation": float(delta[feature_idx].item()),
                "beta_mu": float(beta_mu[feature_idx].item()),
                "tool_call_projection": float(tool_proj[feature_idx].item()),
                "kappa": float(score[feature_idx].item()),
                "abs_kappa": float(abs(score[feature_idx].item())),
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    gate_bundle = torch.load(args.pc_bundle, map_location="cpu", weights_only=False)
    if "layer" not in gate_bundle:
        raise KeyError(f"{args.pc_bundle} does not record its localized commitment layer")
    commitment_layer = int(gate_bundle["layer"])

    model, tokenizer, tool_token_id = load_model_and_tokenizer(model_path=args.model_path, device=args.device)
    pairs = load_sample_pairs(model, dataset_root=args.dataset_root, split=args.split, max_pairs=0)
    pair_batches = build_pair_batches(pairs, batch_size=args.batch_size)

    formation_layers, late_layers, window_protocol = resolve_windows(
        args,
        commitment_layer=commitment_layer,
        n_layers=int(model.cfg.n_layers),
    )
    all_layers = sorted(set(formation_layers + late_layers))

    clean_inputs, corrupt_inputs = collect_layer_inputs(
        model,
        pair_batches,
        layers=all_layers,
        n_pairs=len(pairs),
    )

    if args.direction == "mean-diff":
        direction_raw = gate_bundle.get("mean_diff")
        if direction_raw is None:
            raise KeyError(f"{args.pc_bundle} does not contain mean_diff; rerun with --direction pc1 only if documenting PC1 proxy.")
        if not isinstance(direction_raw, torch.Tensor):
            direction_raw = torch.tensor(direction_raw)
        v1 = direction_raw.detach().cpu().float().view(-1)
    else:
        v1 = gate_bundle["components"][0].detach().cpu().float().view(-1)
    direction_norm = float(v1.norm().item())
    v1 = v1 / max(direction_norm, 1e-12)
    tool_vector = model.W_U[:, tool_token_id].detach().cpu().float()

    formation_corrupt_rows: list[dict[str, object]] = []
    formation_clean_rows: list[dict[str, object]] = []
    late_candidate_rows: list[dict[str, object]] = []

    for layer in tqdm(all_layers, desc="Scoring transcoder layers", dynamic_ncols=True):
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
        beta_mu = torch.mv(W_dec, v1)
        tool_proj = torch.mv(W_dec, tool_vector)
        kappa = delta * beta_mu

        if layer in formation_layers:
            corrupt_mask = (delta < 0) & (beta_mu < 0)
            clean_mask = (delta > 0) & (beta_mu > 0)
            formation_corrupt_rows.extend(
                top_rows_for_mask(
                    layer=layer,
                    score=kappa,
                    delta=delta,
                    beta_mu=beta_mu,
                    tool_proj=tool_proj,
                    mask=corrupt_mask,
                    limit=args.top_k,
                )
            )
            formation_clean_rows.extend(
                top_rows_for_mask(
                    layer=layer,
                    score=kappa,
                    delta=delta,
                    beta_mu=beta_mu,
                    tool_proj=tool_proj,
                    mask=clean_mask,
                    limit=args.top_k,
                )
            )

        if layer in late_layers:
            late_mask = (delta > 0) & (beta_mu > 0)
            late_candidate_rows.extend(
                top_rows_for_mask(
                    layer=layer,
                    score=kappa,
                    delta=delta,
                    beta_mu=beta_mu,
                    tool_proj=tool_proj,
                    mask=late_mask,
                    limit=max(args.top_k, 100),
                )
            )

        del W_enc, b_enc, W_dec, mean_clean, mean_corrupt, delta, beta_mu, tool_proj, kappa, weights
        clear_cuda()

    formation_corrupt_rows.sort(key=lambda row: float(row["abs_kappa"]), reverse=True)
    formation_clean_rows.sort(key=lambda row: float(row["abs_kappa"]), reverse=True)
    top_corrupt = formation_corrupt_rows[: args.top_k]
    top_clean = formation_clean_rows[: args.top_k]
    kappa_c = float(sum(float(row["abs_kappa"]) for row in top_corrupt))
    kappa_plus = float(sum(float(row["abs_kappa"]) for row in top_clean))

    late_candidate_rows.sort(
        key=lambda row: (
            float(row["delta_activation"]) * float(max(row["tool_call_projection"], 0.0)),
            float(row["abs_kappa"]),
            float(max(row["tool_call_projection"], 0.0)),
        ),
        reverse=True,
    )
    key_feat = late_candidate_rows[0] if late_candidate_rows else None

    args.output_root.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_root / "formation_top_corrupt_higher.csv", top_corrupt)
    write_csv(args.output_root / "formation_top_clean_higher.csv", top_clean)
    write_csv(args.output_root / "late_key_feature_candidates.csv", late_candidate_rows[:100])

    summary = {
        "size_label": args.size_label,
        "model_path": str(args.model_path),
        "transcoder_path": str(args.transcoder_path),
        "pc_bundle": str(args.pc_bundle),
        "commitment_layer": commitment_layer,
        "window_protocol": window_protocol,
        "direction": args.direction,
        "direction_norm_before_unit": direction_norm,
        "dataset_root": str(args.dataset_root),
        "split": args.split,
        "n_pairs": len(pairs),
        "formation_layers": formation_layers,
        "late_layers": late_layers,
        "tool_token_id": int(tool_token_id),
        "tool_token_text": tokenizer.decode([tool_token_id], clean_up_tokenization_spaces=False),
        "kappa_corrupt_higher_topk_abs_sum": kappa_c,
        "kappa_clean_higher_topk_abs_sum": kappa_plus,
        "kappa_ratio_text": f"{kappa_c:.1f} / {kappa_plus:.1f}",
        "key_feat": {
            "label": (f"L{int(key_feat['layer'])} F{int(key_feat['feature_idx'])}" if key_feat else None),
            **(key_feat or {}),
        },
    }
    (args.output_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.output_root / "summary.md").write_text(
        "\n".join(
            [
                f"# {args.size_label} Transcoder Table 3 Metrics",
                "",
                f"- Split: `{args.split}` (`{len(pairs)}` pairs)",
                f"- Localized commitment layer: `L{commitment_layer}`",
                f"- Window protocol: `{window_protocol}`",
                f"- Formation layers: `{formation_layers}`",
                f"- Late layers: `{late_layers}`",
                f"- Direction: `{args.direction}` from `{args.pc_bundle}`",
                f"- `kappa_c / kappa_+`: `{kappa_c:.1f} / {kappa_plus:.1f}`",
                f"- Key feat: `{summary['key_feat']['label']}`" if key_feat else "- Key feat: `None`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
