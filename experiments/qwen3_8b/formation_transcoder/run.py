#!/usr/bin/env python3
"""Vector formation & Transcoder feature analysis on Qwen3-8B (Sec 5.2, 5.3, Figure 2, Table 5).

This script performs:
1. Section 5.2 (Figure 2):
   - Computes the residual execution-minus-analysis gap projection:
     g_l = (1/N) * sum_i <h_p^(l)(x_i^c) - h_p^(l)(x_i^*), mu_hat> for l = 0 ... 24
   - Projects MLP and Attention writes onto mu_hat across layers
   - Computes formation window (L20-L23) write shares: MLP % vs Attention %
2. Section 5.3 (Table 5):
   - Loads Transcoder checkpoints for layers 20-23
   - Decomposes MLP writes into sparse features: kappa_lf = (a_clean - a_corrupt) * <w_dec, mu_hat>
   - Computes K_corrupt, K_clean, ratio K_corrupt / K_clean, layer share (%), and semantic labels
   - Feature ablation: suppresses top-5 analysis-active features on corrupt prompts to measure
     the restoration rate of <tool_call> and shift in g_l.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from safetensors.torch import load_file

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from mi4tc.paths import model_path, transcoder_root  # noqa: E402

TOOL_CALL_TOKEN = "<tool_call>"
TOOL_CALL_ID = 151657


def load_pairs(dataset_root: Path, split: str, max_pairs: int = 0) -> list[dict[str, Any]]:
    """Load one explicit split from the model-specific paired dataset."""
    pairs_file = dataset_root / "manifest.jsonl" if (dataset_root / "manifest.jsonl").exists() else dataset_root / "pairs.jsonl"
    rows = []
    with pairs_file.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                if item.get("split") == split:
                    rows.append(item)
    if max_pairs > 0:
        rows = rows[:max_pairs]
    return rows


def load_heldout_pairs(dataset_root: Path, max_pairs: int = 0) -> list[dict[str, Any]]:
    return load_pairs(dataset_root, "heldout", max_pairs=max_pairs)


def load_coding_vector(vector_path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    """Return raw vector (mu_Delta) and normalized unit direction (mu_hat)."""
    payload = torch.load(str(vector_path), map_location="cpu", weights_only=False)
    raw = payload["mean_diff"] if isinstance(payload, dict) and "mean_diff" in payload else payload
    raw = raw.float()
    unit = raw / raw.norm().clamp_min(1e-12)
    return raw, unit


def make_input_batch(
    tokenizer: Any,
    pairs: list[dict[str, Any]],
    dataset_root: Path,
    side: str,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Right-pad a batch while retaining each prompt's prediction position."""
    rel_key = "clean_relpath" if side == "clean" else "corrupt_relpath"
    prompts = [(dataset_root / pair[rel_key]).read_text(encoding="utf-8") for pair in pairs]
    tokenized = [tokenizer.encode(prompt, add_special_tokens=False) for prompt in prompts]
    if not tokenized or any(not tokens for tokens in tokenized):
        raise ValueError(f"Encountered an empty {side} prompt batch")

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    max_len = max(len(tokens) for tokens in tokenized)
    batch_size = len(tokenized)
    input_ids = torch.full((batch_size, max_len), pad_id, dtype=torch.long, device=device)
    attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long, device=device)
    last_pos = torch.zeros(batch_size, dtype=torch.long, device=device)
    for idx, tokens in enumerate(tokenized):
        input_ids[idx, : len(tokens)] = torch.tensor(tokens, dtype=torch.long, device=device)
        attention_mask[idx, : len(tokens)] = 1
        last_pos[idx] = len(tokens) - 1
    return input_ids, attention_mask, last_pos


def capture_formation_mlp_inputs(
    model: Any,
    tokenizer: Any,
    pairs: list[dict[str, Any]],
    dataset_root: Path,
    batch_size: int,
    device: torch.device,
) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor]]:
    """Capture L20--L23 MLP inputs for train-only feature selection."""
    formation_layers = (20, 21, 22, 23)
    captured: dict[str, dict[int, list[torch.Tensor]]] = {
        side: {layer: [] for layer in formation_layers} for side in ("clean", "corrupt")
    }

    print(f"Capturing L20--L23 MLP inputs on {len(pairs)} train pairs for feature selection...", flush=True)
    for start in range(0, len(pairs), batch_size):
        batch_pairs = pairs[start : start + batch_size]
        for side in ("clean", "corrupt"):
            input_ids, attention_mask, last_pos = make_input_batch(tokenizer, batch_pairs, dataset_root, side, device)
            rows = torch.arange(len(batch_pairs), device=device)
            holders: dict[int, torch.Tensor] = {}
            handles = []
            for layer in formation_layers:
                def make_hook(layer_idx: int):
                    def hook(module, args):
                        holders[layer_idx] = args[0][rows, last_pos].detach().float().cpu()
                    return hook

                handles.append(model.model.layers[layer].mlp.register_forward_pre_hook(make_hook(layer)))
            try:
                with torch.no_grad():
                    model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
            finally:
                for handle in handles:
                    handle.remove()

            for layer in formation_layers:
                captured[side][layer].append(holders[layer])

        if (start // batch_size + 1) % max(1, len(pairs) // (batch_size * 4)) == 0:
            print(f"  Selected-input capture processed {min(start + batch_size, len(pairs))}/{len(pairs)}", flush=True)

    return (
        {layer: torch.cat(captured["clean"][layer]) for layer in formation_layers},
        {layer: torch.cat(captured["corrupt"][layer]) for layer in formation_layers},
    )


def capture_trajectory_and_writes(
    model: Any,
    tokenizer: Any,
    pairs: list[dict[str, Any]],
    dataset_root: Path,
    mu_hat: torch.Tensor,
    max_layer: int = 24,
    batch_size: int = 8,
    device: torch.device = torch.device("cuda"),
) -> dict[str, Any]:
    """Capture residual states, MLP writes, and Attn writes for clean and corrupt pairs."""
    layers_to_track = list(range(max_layer + 1))
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    mu_dev = mu_hat.to(device=device, dtype=torch.float32)

    # Accumulators for projections
    clean_resid_projs = {l: [] for l in layers_to_track}
    corrupt_resid_projs = {l: [] for l in layers_to_track}
    clean_mlp_projs = {l: [] for l in layers_to_track}
    corrupt_mlp_projs = {l: [] for l in layers_to_track}
    clean_attn_projs = {l: [] for l in layers_to_track}
    corrupt_attn_projs = {l: [] for l in layers_to_track}

    # Store raw inputs to MLPs at layers 20-23 for Transcoder analysis
    formation_layers = [20, 21, 22, 23]
    clean_mlp_inputs = {l: [] for l in formation_layers}
    corrupt_mlp_inputs = {l: [] for l in formation_layers}

    # We also keep corrupt states at L24 for ablation testing
    corrupt_l24_states = []
    clean_l24_states = []

    print(f"Sweeping layers 0..{max_layer} on {len(pairs)} pairs...", flush=True)

    for b_start in range(0, len(pairs), batch_size):
        b_pairs = pairs[b_start : b_start + batch_size]
        for side in ("clean", "corrupt"):
            prompts = []
            for p in b_pairs:
                rel = p["clean_relpath"] if side == "clean" else p["corrupt_relpath"]
                text = (dataset_root / rel).read_text(encoding="utf-8")
                prompts.append(text)

            tokenized = [tokenizer.encode(p, add_special_tokens=False) for p in prompts]
            max_len = max(len(t) for t in tokenized)
            B = len(prompts)
            input_ids = torch.full((B, max_len), pad_id, dtype=torch.long, device=device)
            attention_mask = torch.zeros((B, max_len), dtype=torch.long, device=device)
            last_pos = torch.zeros(B, dtype=torch.long, device=device)
            for i, tok in enumerate(tokenized):
                input_ids[i, : len(tok)] = torch.tensor(tok, dtype=torch.long, device=device)
                attention_mask[i, : len(tok)] = 1
                last_pos[i] = len(tok) - 1

            # Register hooks
            resid_holders = {}
            mlp_holders = {}
            attn_holders = {}
            mlp_in_holders = {}
            handles = []

            for l in layers_to_track:
                def make_resid_hook(layer_idx: int):
                    def hook(module, args):
                        hidden = args[0]
                        rows = torch.arange(B, device=device)
                        resid_holders[layer_idx] = hidden[rows, last_pos].detach().float()
                    return hook

                def make_mlp_hook(layer_idx: int):
                    def hook(module, inp, out):
                        rows = torch.arange(B, device=device)
                        mlp_holders[layer_idx] = out[rows, last_pos].detach().float()
                    return hook

                def make_attn_hook(layer_idx: int):
                    def hook(module, inp, out):
                        rows = torch.arange(B, device=device)
                        attn_out = out[0] if isinstance(out, tuple) else out
                        attn_holders[layer_idx] = attn_out[rows, last_pos].detach().float()
                    return hook

                dec_layer = model.model.layers[l]
                handles.append(dec_layer.register_forward_pre_hook(make_resid_hook(l)))
                handles.append(dec_layer.mlp.register_forward_hook(make_mlp_hook(l)))
                handles.append(dec_layer.self_attn.register_forward_hook(make_attn_hook(l)))

                if l in formation_layers:
                    def make_mlp_in_hook(layer_idx: int):
                        def hook(module, args):
                            rows = torch.arange(B, device=device)
                            mlp_in_holders[layer_idx] = args[0][rows, last_pos].detach().float().cpu()
                        return hook
                    handles.append(dec_layer.mlp.register_forward_pre_hook(make_mlp_in_hook(l)))

            with torch.no_grad():
                model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)

            for h in handles:
                h.remove()

            # Record projections
            rows = torch.arange(B, device=device)
            target_resid = clean_resid_projs if side == "clean" else corrupt_resid_projs
            target_mlp = clean_mlp_projs if side == "clean" else corrupt_mlp_projs
            target_attn = clean_attn_projs if side == "clean" else corrupt_attn_projs
            target_mlp_in = clean_mlp_inputs if side == "clean" else corrupt_mlp_inputs

            for l in layers_to_track:
                target_resid[l].append((resid_holders[l] @ mu_dev).cpu())
                target_mlp[l].append((mlp_holders[l] @ mu_dev).cpu())
                target_attn[l].append((attn_holders[l] @ mu_dev).cpu())

            for l in formation_layers:
                target_mlp_in[l].append(mlp_in_holders[l])

            if side == "corrupt":
                corrupt_l24_states.append(resid_holders[24].cpu())
            else:
                clean_l24_states.append(resid_holders[24].cpu())

        if (b_start // batch_size + 1) % max(1, len(pairs) // (batch_size * 4)) == 0:
            print(f"  Processed {min(b_start + batch_size, len(pairs))}/{len(pairs)} pairs...", flush=True)

    # Concatenate results
    trajectory_data = []
    total_mlp_formation = 0.0
    total_attn_formation = 0.0

    for l in layers_to_track:
        c_res = torch.cat(clean_resid_projs[l])
        k_res = torch.cat(corrupt_resid_projs[l])
        c_mlp = torch.cat(clean_mlp_projs[l])
        k_mlp = torch.cat(corrupt_mlp_projs[l])
        c_attn = torch.cat(clean_attn_projs[l])
        k_attn = torch.cat(corrupt_attn_projs[l])

        g_l = float((c_res - k_res).mean().item())
        w_mlp = float((c_mlp - k_mlp).mean().item())
        w_attn = float((c_attn - k_attn).mean().item())

        if l in formation_layers:
            total_mlp_formation += w_mlp
            total_attn_formation += w_attn

        trajectory_data.append({
            "layer": l,
            "residual_gap_gl": g_l,
            "mlp_write": w_mlp,
            "attn_write": w_attn,
            "total_layer_write": w_mlp + w_attn,
        })

    tot_formation = total_mlp_formation + total_attn_formation
    mlp_share = (total_mlp_formation / tot_formation * 100.0) if tot_formation > 0 else 0.0
    attn_share = (total_attn_formation / tot_formation * 100.0) if tot_formation > 0 else 0.0

    # Max single write layer in formation window
    formation_writes = [r for r in trajectory_data if r["layer"] in formation_layers]
    max_write_layer = max(formation_writes, key=lambda r: r["mlp_write"])

    return {
        "trajectory": trajectory_data,
        "formation_summary": {
            "formation_layers": formation_layers,
            "total_mlp_write": total_mlp_formation,
            "total_attn_write": total_attn_formation,
            "total_formation_write": tot_formation,
            "mlp_share_pct": mlp_share,
            "attn_share_pct": attn_share,
            "max_mlp_write_layer": max_write_layer["layer"],
            "max_mlp_write_value": max_write_layer["mlp_write"],
        },
        "clean_mlp_inputs": {l: torch.cat(clean_mlp_inputs[l]) for l in formation_layers},
        "corrupt_mlp_inputs": {l: torch.cat(corrupt_mlp_inputs[l]) for l in formation_layers},
        "corrupt_l24_states": torch.cat(corrupt_l24_states),
        "clean_l24_states": torch.cat(clean_l24_states),
    }


def analyze_transcoder_layers(
    transcoder_dir: Path,
    clean_mlp_inputs: dict[int, torch.Tensor],
    corrupt_mlp_inputs: dict[int, torch.Tensor],
    mu_hat: torch.Tensor,
    formation_mlp_write: float,
    device: torch.device,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Compute Table 5 metrics across layers 20-23 using Transcoder checkpoints."""
    mu_dev = mu_hat.to(device=device, dtype=torch.float32)
    table5_rows = []
    top_features = []
    all_suppressor_features = []


    layer_total_kappas = {}

    for layer in (20, 21, 22, 23):
        tc_path = transcoder_dir / f"layer_{layer}.safetensors"
        if not tc_path.is_file():
            raise FileNotFoundError(f"Missing transcoder checkpoint: {tc_path}")

        tc = load_file(str(tc_path))
        W_enc = tc["W_enc"].to(device=device, dtype=torch.bfloat16)  # [n_feat, d_model]
        b_enc = tc["b_enc"].to(device=device, dtype=torch.bfloat16)
        W_dec = tc["W_dec"].to(device=device, dtype=torch.bfloat16)  # [n_feat, d_model]

        # Feature activations: a = ReLU(x @ W_enc.T + b_enc)
        # Compute in chunks to avoid GPU OOM
        n_samples = clean_mlp_inputs[layer].shape[0]
        n_feat = W_enc.shape[0]

        a_clean_sum = torch.zeros(n_feat, device="cpu", dtype=torch.float32)
        a_corrupt_sum = torch.zeros(n_feat, device="cpu", dtype=torch.float32)
        chunk_size = 16

        for start in range(0, n_samples, chunk_size):
            end = min(start + chunk_size, n_samples)
            c_in = clean_mlp_inputs[layer][start:end].to(device=device, dtype=torch.bfloat16)
            k_in = corrupt_mlp_inputs[layer][start:end].to(device=device, dtype=torch.bfloat16)

            with torch.no_grad():
                c_act = F.relu(F.linear(c_in, W_enc, b_enc)).float().sum(dim=0).cpu()
                k_act = F.relu(F.linear(k_in, W_enc, b_enc)).float().sum(dim=0).cpu()

            a_clean_sum += c_act
            a_corrupt_sum += k_act

        bar_a_clean = a_clean_sum / n_samples
        bar_a_corrupt = a_corrupt_sum / n_samples
        delta_a = bar_a_clean - bar_a_corrupt  # [n_feat]

        # Decoder write along mu_hat: beta_f = <W_dec[f], mu_hat>
        with torch.no_grad():
            beta = (W_dec.float() @ mu_dev).cpu()  # [n_feat]

        # kappa_f = delta_a * beta
        kappa = delta_a * beta

        corrupt_higher = bar_a_corrupt > bar_a_clean
        clean_higher = bar_a_clean > bar_a_corrupt

        K_corrupt = float(kappa[corrupt_higher].abs().sum().item())
        K_clean = float(kappa[clean_higher].abs().sum().item())
        ratio = (K_corrupt / K_clean) if K_clean > 0 else float("inf")
        dominant = "Clean" if K_clean > K_corrupt else "Corrupt"

        total_kappa_layer = float(kappa.sum().item())
        layer_total_kappas[layer] = total_kappa_layer

        # Top suppressor features: corrupt_higher and beta < 0 (opposing mu_hat)
        suppressor_mask = corrupt_higher & (beta < 0)
        suppressor_indices = torch.where(suppressor_mask)[0]
        suppressor_scores = kappa[suppressor_indices].abs()
        # Retain a broader train-selected pool so the primary top-five set and
        # a layer-matched random control can be evaluated on held-out prompts.
        top_k_supp = torch.topk(suppressor_scores, min(100, len(suppressor_indices)))

        for idx, score in zip(top_k_supp.indices.tolist(), top_k_supp.values.tolist()):
            f_idx = int(suppressor_indices[idx].item())
            all_suppressor_features.append({
                "layer": layer,
                "feature_idx": f_idx,
                "abs_kappa": float(score),
                "kappa": float(kappa[f_idx].item()),
                "delta_a": float(delta_a[f_idx].item()),
                "beta": float(beta[f_idx].item()),
                "mean_a_corrupt": float(bar_a_corrupt[f_idx].item()),
                "mean_a_clean": float(bar_a_clean[f_idx].item()),
                "decoder_vector": W_dec[f_idx].detach().cpu(),
            })

        # Top 5 overall features for logging
        top_overall = torch.topk(kappa.abs(), 5)
        for f_idx in top_overall.indices.tolist():
            top_features.append({
                "layer": layer,
                "feature_idx": int(f_idx),
                "kappa": float(kappa[f_idx].item()),
                "abs_kappa": float(abs(kappa[f_idx].item())),
                "delta_a": float(delta_a[f_idx].item()),
                "beta": float(beta[f_idx].item()),
                "dominant_side": "clean" if clean_higher[f_idx] else "corrupt",
            })

        table5_rows.append({
            "layer": f"L{layer}",
            "dominant": dominant,
            "K_corrupt": K_corrupt,
            "K_clean": K_clean,
            "K_corrupt_over_K_clean": ratio,
            "share_pct": 0.0,  # Will compute after sum

            "_total_kappa": total_kappa_layer,
        })

        del W_enc, b_enc, W_dec, tc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Normalize shares
    total_formation_kappa = sum(r["_total_kappa"] for r in table5_rows)
    for r in table5_rows:
        if total_formation_kappa != 0:
            r["share_pct"] = (r["_total_kappa"] / total_formation_kappa) * 100.0
        else:
            r["share_pct"] = 100.0 / len(table5_rows)

    return table5_rows, top_features, all_suppressor_features


def _feature_metadata(features: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{key: value for key, value in item.items() if key != "decoder_vector"} for item in features]


def select_layer_matched_random_control(
    suppressor_features: list[dict[str, Any]],
    selected: list[dict[str, Any]],
    *,
    seed: int,
) -> list[dict[str, Any]]:
    """Sample a non-selected suppressor control with the same layer counts."""
    selected_keys = {(int(item["layer"]), int(item["feature_idx"])) for item in selected}
    candidates_by_layer: dict[int, list[dict[str, Any]]] = {}
    for item in suppressor_features:
        key = (int(item["layer"]), int(item["feature_idx"]))
        if key not in selected_keys:
            candidates_by_layer.setdefault(int(item["layer"]), []).append(item)

    rng = random.Random(seed)
    control: list[dict[str, Any]] = []
    for item in selected:
        layer = int(item["layer"])
        pool = candidates_by_layer.get(layer, [])
        if not pool:
            raise RuntimeError(f"No non-selected suppressor features remain for layer {layer}")
        chosen = pool.pop(rng.randrange(len(pool)))
        control.append(chosen)
    return control


def load_feature_payloads(
    transcoder_dir: Path,
    features: list[dict[str, Any]],
    device: torch.device,
) -> dict[int, dict[str, torch.Tensor]]:
    """Load only the selected Transcoder rows for exact per-sample zeroing."""
    by_layer: dict[int, list[int]] = {}
    for item in features:
        by_layer.setdefault(int(item["layer"]), []).append(int(item["feature_idx"]))

    weight_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    payloads: dict[int, dict[str, torch.Tensor]] = {}
    for layer, feature_ids in by_layer.items():
        weights = load_file(str(transcoder_dir / f"layer_{layer}.safetensors"))
        indices = torch.tensor(feature_ids, dtype=torch.long)
        payloads[layer] = {
            "W_enc": weights["W_enc"][indices].to(device=device, dtype=weight_dtype),
            "b_enc": weights["b_enc"][indices].to(device=device, dtype=weight_dtype),
            "W_dec": weights["W_dec"][indices].to(device=device, dtype=weight_dtype),
        }
    return payloads


def forward_with_l24_capture(
    model: Any,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    last_pos: torch.Tensor,
    *,
    feature_payloads: dict[int, dict[str, torch.Tensor]] | None = None,
) -> dict[str, torch.Tensor]:
    """Run a forward pass and optionally zero selected feature writes in-place.

    The intervention is applied at each source MLP output.  For a selected
    Transcoder feature f, its *current per-sample* contribution
    a_f(x) W_dec[f] is subtracted from the actual MLP output at the prediction
    position.  Capturing L24's input then measures the resulting state rather
    than an algebraic proxy for it.
    """
    batch_size = int(input_ids.shape[0])
    device = input_ids.device
    rows = torch.arange(batch_size, device=device)
    captured: dict[str, torch.Tensor] = {}
    handles = []

    def capture_l24_pre(module, args):
        captured["l24_pre"] = args[0][rows, last_pos].detach().float().cpu()

    handles.append(model.model.layers[24].register_forward_pre_hook(capture_l24_pre))

    if feature_payloads:
        for layer, payload in feature_payloads.items():
            def make_zero_hook(layer_payload: dict[str, torch.Tensor]):
                def hook(module, args, output):
                    mlp_input = args[0][rows, last_pos]
                    activations = F.relu(
                        F.linear(
                            mlp_input.to(dtype=layer_payload["W_enc"].dtype),
                            layer_payload["W_enc"],
                            layer_payload["b_enc"],
                        )
                    )
                    contribution = activations @ layer_payload["W_dec"]
                    out = output.clone()
                    out[rows, last_pos] = out[rows, last_pos] - contribution.to(dtype=out.dtype)
                    return out

                return hook

            handles.append(model.model.layers[layer].mlp.register_forward_hook(make_zero_hook(payload)))

    try:
        with torch.no_grad():
            output = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    finally:
        for handle in handles:
            handle.remove()

    logits = output.logits if hasattr(output, "logits") else output[0]
    last_logits = logits[rows, last_pos].float()
    tool_logits = last_logits[:, TOOL_CALL_ID]
    competitors = last_logits.clone()
    competitors[:, TOOL_CALL_ID] = -torch.inf
    return {
        "l24_pre": captured["l24_pre"],
        "top1": last_logits.argmax(dim=-1).detach().cpu(),
        "tool_logit": tool_logits.detach().cpu(),
        "tool_margin": (tool_logits - competitors.max(dim=-1).values).detach().cpu(),
    }


def collect_l24_baselines(
    model: Any,
    tokenizer: Any,
    pairs: list[dict[str, Any]],
    dataset_root: Path,
    batch_size: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Collect matched clean/corrupt prediction states and behavior once."""
    collected: dict[str, list[torch.Tensor]] = {
        f"{side}_{name}": []
        for side in ("clean", "corrupt")
        for name in ("l24_pre", "top1", "tool_logit", "tool_margin")
    }
    print(f"Collecting clean/corrupt L24 baselines on {len(pairs)} held-out pairs...", flush=True)
    for start in range(0, len(pairs), batch_size):
        batch_pairs = pairs[start : start + batch_size]
        for side in ("clean", "corrupt"):
            input_ids, attention_mask, last_pos = make_input_batch(tokenizer, batch_pairs, dataset_root, side, device)
            result = forward_with_l24_capture(model, input_ids, attention_mask, last_pos)
            for name, value in result.items():
                collected[f"{side}_{name}"].append(value)
    return {key: torch.cat(values) for key, values in collected.items()}


def run_zero_ablation(
    model: Any,
    tokenizer: Any,
    pairs: list[dict[str, Any]],
    dataset_root: Path,
    batch_size: int,
    device: torch.device,
    feature_payloads: dict[int, dict[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    collected: dict[str, list[torch.Tensor]] = {name: [] for name in ("l24_pre", "top1", "tool_logit", "tool_margin")}
    for start in range(0, len(pairs), batch_size):
        batch_pairs = pairs[start : start + batch_size]
        input_ids, attention_mask, last_pos = make_input_batch(tokenizer, batch_pairs, dataset_root, "corrupt", device)
        result = forward_with_l24_capture(
            model,
            input_ids,
            attention_mask,
            last_pos,
            feature_payloads=feature_payloads,
        )
        for name, value in result.items():
            collected[name].append(value)
    return {key: torch.cat(values) for key, values in collected.items()}


def summarize_zero_ablation(
    baseline: dict[str, torch.Tensor],
    ablated: dict[str, torch.Tensor],
    mu_hat: torch.Tensor,
) -> dict[str, Any]:
    mu_cpu = mu_hat.detach().float().cpu()
    clean_projection = baseline["clean_l24_pre"] @ mu_cpu
    corrupt_projection = baseline["corrupt_l24_pre"] @ mu_cpu
    ablated_projection = ablated["l24_pre"] @ mu_cpu
    state_delta = ablated_projection - corrupt_projection
    clean_minus_corrupt_gap = clean_projection - corrupt_projection
    mean_gap = float(clean_minus_corrupt_gap.mean().item())
    mean_delta = float(state_delta.mean().item())

    baseline_corrupt_top1 = baseline["corrupt_top1"] == TOOL_CALL_ID
    ablated_top1 = ablated["top1"] == TOOL_CALL_ID
    strict_recovery = (~baseline_corrupt_top1) & ablated_top1
    n_samples = int(ablated_top1.shape[0])

    return {
        "intervention": "zero each selected feature's current a_f(x) W_dec[f] at its source MLP output",
        "n_heldout": n_samples,
        "baseline_corrupt_tool_call_top1_count": int(baseline_corrupt_top1.sum().item()),
        "baseline_corrupt_tool_call_top1_rate": float(baseline_corrupt_top1.float().mean().item() * 100.0),
        "post_ablation_tool_call_top1_count": int(ablated_top1.sum().item()),
        "post_ablation_tool_call_top1_rate": float(ablated_top1.float().mean().item() * 100.0),
        "strict_recovery_count": int(strict_recovery.sum().item()),
        "strict_recovery_rate": float(strict_recovery.float().mean().item() * 100.0),
        "mean_tool_call_logit_delta": float((ablated["tool_logit"] - baseline["corrupt_tool_logit"]).mean().item()),
        "mean_tool_call_margin_delta": float((ablated["tool_margin"] - baseline["corrupt_tool_margin"]).mean().item()),
        "mean_clean_l24_projection": float(clean_projection.mean().item()),
        "mean_corrupt_l24_projection": float(corrupt_projection.mean().item()),
        "mean_post_ablation_l24_projection": float(ablated_projection.mean().item()),
        "mean_clean_minus_corrupt_l24_gap": mean_gap,
        "mean_l24_delta_along_mu_hat": mean_delta,
        "fraction_of_l24_gap_closed": (mean_delta / mean_gap) if abs(mean_gap) > 1e-12 else None,
    }


def evaluate_feature_ablation(
    model: Any,
    tokenizer: Any,
    pairs: list[dict[str, Any]],
    dataset_root: Path,
    transcoder_dir: Path,
    suppressor_features: list[dict[str, Any]],
    mu_hat: torch.Tensor,
    device: torch.device,
    *,
    selection_pair_count: int,
    batch_size: int,
    control_seed: int = 42,
) -> dict[str, Any]:
    """Evaluate a train-selected top-five zero ablation on held-out pairs."""
    suppressor_features.sort(key=lambda item: float(item["abs_kappa"]), reverse=True)
    top5 = suppressor_features[:5]
    labels = [f"L{item['layer']}/F{item['feature_idx']}" for item in top5]
    print(f"\nTrain-selected top-5 suppressor features across L20-L23: {labels}", flush=True)
    control = select_layer_matched_random_control(suppressor_features, top5, seed=control_seed)
    control_labels = [f"L{item['layer']}/F{item['feature_idx']}" for item in control]
    print(f"Layer-matched random suppressor control: {control_labels}", flush=True)

    baseline = collect_l24_baselines(model, tokenizer, pairs, dataset_root, batch_size, device)
    top5_payloads = load_feature_payloads(transcoder_dir, top5, device)
    control_payloads = load_feature_payloads(transcoder_dir, control, device)

    print("Evaluating per-sample top-5 zero ablation...", flush=True)
    top5_result = run_zero_ablation(model, tokenizer, pairs, dataset_root, batch_size, device, top5_payloads)
    print("Evaluating layer-matched random zero-ablation control...", flush=True)
    control_result = run_zero_ablation(model, tokenizer, pairs, dataset_root, batch_size, device, control_payloads)

    top5_summary = summarize_zero_ablation(baseline, top5_result, mu_hat)
    control_summary = summarize_zero_ablation(baseline, control_result, mu_hat)
    return {
        "selection_split": "train",
        "selection_pair_count": selection_pair_count,
        "selection_rule": "top five corrupt-higher, mu-hat-opposing Transcoder features by |kappa|",
        "top5_features": _feature_metadata(top5),
        "layer_matched_random_suppressor_control_features": _feature_metadata(control),
        "top5_zero_ablation": top5_summary,
        "layer_matched_random_suppressor_zero_control": control_summary,
        # Clear aliases for the quantities previously reported under ambiguous
        # names in the original summary.
        "baseline_corrupt_call_rate": top5_summary["baseline_corrupt_tool_call_top1_rate"],
        "restored_tool_call_rate": top5_summary["post_ablation_tool_call_top1_rate"],
        "strict_recovery_rate": top5_summary["strict_recovery_rate"],
        "shift_along_mu_hat": top5_summary["mean_l24_delta_along_mu_hat"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Vector formation & Transcoder features (Sec 5.2 & 5.3)")
    parser.add_argument("--model-path", type=Path, default=model_path("qwen3_8b"))
    parser.add_argument("--transcoder-dir", type=Path, default=transcoder_root() / "Qwen3-8B")
    parser.add_argument("--vector-path", type=Path, default=REPO_ROOT / "results/transfer/qwen3_8b/coding_vector.pt")
    parser.add_argument("--dataset-root", type=Path, default=REPO_ROOT / "datasets/qwen3_8b/pair")
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "results/qwen3_8b/formation_transcoder")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-pairs", type=int, default=0)
    parser.add_argument("--selection-max-pairs", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="bfloat16")
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    print(f"Loading coding vector from {args.vector_path}...")
    raw_vector, mu_hat = load_coding_vector(args.vector_path)
    print(f"Loaded vector norm = {raw_vector.norm().item():.3f}")

    print(f"Loading Qwen3-8B from {args.model_path}...")
    import transformers
    torch_dtype = getattr(torch, args.dtype)
    tokenizer = transformers.AutoTokenizer.from_pretrained(str(args.model_path), trust_remote_code=True)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        str(args.model_path),
        torch_dtype=torch_dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()

    pairs = load_heldout_pairs(args.dataset_root, max_pairs=args.max_pairs)
    selection_pairs = load_pairs(args.dataset_root, "train", max_pairs=args.selection_max_pairs)
    print(f"Loaded {len(pairs)} held-out pairs and {len(selection_pairs)} train pairs from {args.dataset_root}")

    # 1. Formation Trajectory & Component Writes (Figure 2)
    sweep_results = capture_trajectory_and_writes(
        model=model,
        tokenizer=tokenizer,
        pairs=pairs,
        dataset_root=args.dataset_root,
        mu_hat=mu_hat,
        max_layer=24,
        batch_size=args.batch_size,
        device=device,
    )

    trajectory = sweep_results["trajectory"]
    formation_summary = sweep_results["formation_summary"]

    # Write Figure 2 CSV
    traj_csv = args.output_root / "figure2_formation_trajectory.csv"
    with traj_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(trajectory[0].keys()))
        writer.writeheader()
        writer.writerows(trajectory)

    print("\n=== Formation Window (L20-L23) Writes Summary ===")
    print(f"  Total MLP write: {formation_summary['total_mlp_write']:.2f} ({formation_summary['mlp_share_pct']:.1f}%)")
    print(f"  Total Attn write: {formation_summary['total_attn_write']:.2f} ({formation_summary['attn_share_pct']:.1f}%)")
    print(f"  Max MLP single write: Layer {formation_summary['max_mlp_write_layer']} ({formation_summary['max_mlp_write_value']:.2f})")

    # 2. Descriptive held-out Table-5 accounting.  The causal top-five set is
    # selected separately on train pairs below, so held-out evaluation stays
    # independent of feature selection.
    print(f"\nAnalyzing held-out Transcoder features from {args.transcoder_dir}...")
    table5_rows, top_features, _heldout_suppressor_features = analyze_transcoder_layers(
        transcoder_dir=args.transcoder_dir,
        clean_mlp_inputs=sweep_results["clean_mlp_inputs"],
        corrupt_mlp_inputs=sweep_results["corrupt_mlp_inputs"],
        mu_hat=mu_hat,
        formation_mlp_write=formation_summary["total_mlp_write"],
        device=device,
    )

    selection_clean_inputs, selection_corrupt_inputs = capture_formation_mlp_inputs(
        model=model,
        tokenizer=tokenizer,
        pairs=selection_pairs,
        dataset_root=args.dataset_root,
        batch_size=args.batch_size,
        device=device,
    )
    print("Analyzing train-only Transcoder features for causal-set selection...", flush=True)
    selection_table5_rows, _selection_top_features, selection_suppressor_features = analyze_transcoder_layers(
        transcoder_dir=args.transcoder_dir,
        clean_mlp_inputs=selection_clean_inputs,
        corrupt_mlp_inputs=selection_corrupt_inputs,
        mu_hat=mu_hat,
        formation_mlp_write=formation_summary["total_mlp_write"],
        device=device,
    )

    # 3. Train-selected, held-out per-sample Top-5 Suppressor Ablation
    ablation_summary = evaluate_feature_ablation(
        model=model,
        tokenizer=tokenizer,
        pairs=pairs,
        dataset_root=args.dataset_root,
        transcoder_dir=args.transcoder_dir,
        suppressor_features=selection_suppressor_features,
        mu_hat=mu_hat,
        device=device,
        selection_pair_count=len(selection_pairs),
        batch_size=args.batch_size,
    )

    # Write Table 5 Markdown
    md_lines = [
        "# Transcoder feature contributions",
        "",
        f"Evaluated on {len(pairs)} held-out pairs from `datasets/qwen3_8b/pair`.",
        "",
        "| Layer | Dominant | $K_{\\mathrm{corrupt}}$ | $K_{\\mathrm{clean}}$ | $K_{\\mathrm{corrupt}}/K_{\\mathrm{clean}}$ | Share (%) |",
        "|:---|:---|---:|---:|---:|---:|",
    ]
    clean_table5_rows = []
    for r in table5_rows:
        row_dict = {
            "layer": r["layer"],
            "dominant": r["dominant"],
            "K_corrupt": round(r["K_corrupt"], 2),
            "K_clean": round(r["K_clean"], 2),
            "K_corrupt_over_K_clean": round(r["K_corrupt_over_K_clean"], 2),
            "share_pct": round(r["share_pct"], 1),

        }
        clean_table5_rows.append(row_dict)
        md_lines.append(
            f"| {row_dict['layer']} | {row_dict['dominant']} | {row_dict['K_corrupt']} | "
            f"{row_dict['K_clean']} | {row_dict['K_corrupt_over_K_clean']} | {row_dict['share_pct']} |"
        )
    md_lines.append("")
    md_content = "\n".join(md_lines)
    (args.output_root / "table5_transcoder_features.md").write_text(md_content, encoding="utf-8")
    print("\n" + md_content)

    # Write Table 5 CSV
    with (args.output_root / "table5_transcoder_features.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(clean_table5_rows[0].keys()))
        writer.writeheader()
        writer.writerows(clean_table5_rows)

    # Write Top Features CSV
    with (args.output_root / "top_features_l20_l23.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(top_features[0].keys()))
        writer.writeheader()
        writer.writerows(top_features)

    # Write Full Summary JSON
    (args.output_root / "formation_transcoder_summary.json").write_text(
        json.dumps({
            "model_path": str(args.model_path),
            "vector_path": str(args.vector_path),
            "n_heldout": len(pairs),
            "n_train_for_feature_selection": len(selection_pairs),
            "formation_window_summary": formation_summary,
            "heldout_table5": clean_table5_rows,
            "train_selection_table5": [
                {
                    "layer": row["layer"],
                    "dominant": row["dominant"],
                    "K_corrupt": round(row["K_corrupt"], 2),
                    "K_clean": round(row["K_clean"], 2),
                    "K_corrupt_over_K_clean": round(row["K_corrupt_over_K_clean"], 2),
                    "share_pct": round(row["share_pct"], 1),

                }
                for row in selection_table5_rows
            ],
            "top5_feature_ablation": ablation_summary,
        }, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(f"\nFeature ablation restoration rate: {ablation_summary['restored_tool_call_rate']:.2f}%")
    print(f"Results written to {args.output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
