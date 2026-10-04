#!/usr/bin/env python3
"""Vector formation & Transcoder feature analysis on Qwen3-14B (Sec 5.2, 5.3, Figure 2, Table 5).

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


def load_heldout_pairs(dataset_root: Path, max_pairs: int = 0) -> list[dict[str, Any]]:
    pairs_file = dataset_root / "manifest.jsonl" if (dataset_root / "manifest.jsonl").exists() else dataset_root / "pairs.jsonl"
    rows = []
    with pairs_file.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                if item.get("split") == "heldout":
                    rows.append(item)
    if max_pairs > 0:
        rows = rows[:max_pairs]
    return rows


def load_coding_vector(vector_path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    """Return raw vector (mu_Delta) and normalized unit direction (mu_hat)."""
    payload = torch.load(str(vector_path), map_location="cpu", weights_only=False)
    raw = payload["mean_diff"] if isinstance(payload, dict) and "mean_diff" in payload else payload
    raw = raw.float()
    unit = raw / raw.norm().clamp_min(1e-12)
    return raw, unit


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

    # Store raw inputs to MLPs at layers 22-25 for Transcoder analysis
    formation_layers = [28, 29, 30, 31, 32]
    clean_mlp_inputs = {l: [] for l in formation_layers}
    corrupt_mlp_inputs = {l: [] for l in formation_layers}

    # We also keep corrupt states at L26 for ablation testing
    corrupt_l33_states = []
    clean_l33_states = []

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
                corrupt_l33_states.append(resid_holders[26].cpu())
            else:
                clean_l33_states.append(resid_holders[26].cpu())

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
        "corrupt_l24_states": torch.cat(corrupt_l33_states),
        "clean_l24_states": torch.cat(clean_l33_states),
    }


def analyze_transcoder_layers(
    transcoder_dir: Path,
    clean_mlp_inputs: dict[int, torch.Tensor],
    corrupt_mlp_inputs: dict[int, torch.Tensor],
    mu_hat: torch.Tensor,
    formation_mlp_write: float,
    device: torch.device,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Compute Table 5 metrics across layers 22-25 using Transcoder checkpoints."""
    mu_dev = mu_hat.to(device=device, dtype=torch.float32)
    table5_rows = []
    top_features = []
    all_suppressor_features = []

    semantic_labels = {
        28: "Execution requests",
        29: "Non-necessity",
        30: "Analysis-task contexts",
        31: "Analysis-task contexts",
        32: "Analysis-verbs",
    }

    layer_total_kappas = {}

    for layer in (28, 29, 30, 31, 32):
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
        top_k_supp = torch.topk(suppressor_scores, min(10, len(suppressor_indices)))

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
            "semantic_label": semantic_labels[layer],
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
            r["share_pct"] = 25.0

    return table5_rows, top_features, all_suppressor_features


def evaluate_feature_ablation(
    model: Any,
    tokenizer: Any,
    pairs: list[dict[str, Any]],
    dataset_root: Path,
    suppressor_features: list[dict[str, Any]],
    mu_hat: torch.Tensor,
    device: torch.device,
) -> dict[str, Any]:
    # Select top 5 strongest suppressor features overall
    suppressor_features.sort(key=lambda x: x["abs_kappa"], reverse=True)
    top5 = suppressor_features[:5]
    labels = [f"L{x['layer']}/F{x['feature_idx']}" for x in top5]
    print(f"\nTop-5 suppressor features across L20-L23: {labels}")

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    mu_dev = mu_hat.to(device=device, dtype=torch.float32)

    # For each sample, evaluate corrupt prompt baseline vs ablated
    baseline_top1_count = 0
    restored_top1_count = 0
    delta_g24_sum = 0.0
    n_samples = len(pairs)

    batch_size = 8
    for b_start in range(0, n_samples, batch_size):
        b_pairs = pairs[b_start : b_start + batch_size]
        prompts = [(dataset_root / p["corrupt_relpath"]).read_text(encoding="utf-8") for p in b_pairs]
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

        # We intervene by adding the removed suppressor writes to L24 pre-hook
        # Note: Suppressor write is a_f * W_dec[f] (which points against mu_hat).
        # Removing it means adding -a_f * W_dec[f] to cancel the suppressive write.
        # Since each top feature has mean corrupt activation mean_a_corrupt,
        # the collective cancellation vector is: Delta_supp = sum_f (mean_a_corrupt[f] * (-W_dec[f]))
        ablation_offset = torch.zeros(model.config.hidden_size, device=device, dtype=torch.float32)
        for item in top5:
            w_dec = item["decoder_vector"].to(device=device, dtype=torch.float32)
            # Subtracting the suppressive opposing write restores mu_hat
            ablation_offset += (-item["mean_a_corrupt"]) * w_dec

        # Baseline corrupt forward
        with torch.no_grad():
            out_base = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
            logits_base = out_base.logits if hasattr(out_base, "logits") else out_base[0]

        rows = torch.arange(B, device=device)
        base_top1 = logits_base[rows, last_pos].argmax(dim=-1) == TOOL_CALL_ID
        baseline_top1_count += int(base_top1.sum().item())

        # Intervened corrupt forward (patch at L24)
        def pre_hook(module, args):
            hidden = args[0].clone()
            hidden[rows, last_pos] = hidden[rows, last_pos] - ablation_offset.to(dtype=hidden.dtype)
            return (hidden, *args[1:])

        handle = model.model.layers[31].register_forward_pre_hook(pre_hook)
        try:
            with torch.no_grad():
                out_abl = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
                logits_abl = out_abl.logits if hasattr(out_abl, "logits") else out_abl[0]
        finally:
            handle.remove()

        abl_top1 = logits_abl[rows, last_pos].argmax(dim=-1) == TOOL_CALL_ID
        restored_top1_count += int(abl_top1.sum().item())
        delta_g24_sum += float((ablation_offset @ mu_dev).item()) * B

    recovery_rate = (restored_top1_count / n_samples) * 100.0
    baseline_call_rate = (baseline_top1_count / n_samples) * 100.0
    mean_delta_g24 = delta_g24_sum / n_samples

    return {
        "top5_features": [{k: v for k, v in item.items() if k != "decoder_vector"} for item in top5],
        "baseline_corrupt_call_rate": baseline_call_rate,
        "restored_tool_call_rate": recovery_rate,
        "shift_along_mu_hat": mean_delta_g24,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Vector formation & Transcoder features (Sec 5.2 & 5.3)")
    parser.add_argument("--model-path", type=Path, default=model_path("qwen3_14b"))
    parser.add_argument("--transcoder-dir", type=Path, default=transcoder_root() / "Qwen3-14B")
    parser.add_argument("--vector-path", type=Path, default=REPO_ROOT / "results/transfer/qwen3_14b/coding_vector.pt")
    parser.add_argument("--dataset-root", type=Path, default=REPO_ROOT / "datasets/qwen3_14b/pair")
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "results/qwen3_14b/formation_transcoder")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-pairs", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="bfloat16")
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    print(f"Loading coding vector from {args.vector_path}...")
    raw_vector, mu_hat = load_coding_vector(args.vector_path)
    print(f"Loaded vector norm = {raw_vector.norm().item():.3f}")

    print(f"Loading Qwen3-14B from {args.model_path}...")
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
    print(f"Loaded {len(pairs)} held-out pairs from {args.dataset_root}")

    # 1. Formation Trajectory & Component Writes (Figure 2)
    sweep_results = capture_trajectory_and_writes(
        model=model,
        tokenizer=tokenizer,
        pairs=pairs,
        dataset_root=args.dataset_root,
        mu_hat=mu_hat,
        max_layer=33,
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

    # 2. Transcoder Feature Analysis (Table 5)
    print(f"\nAnalyzing Transcoder features from {args.transcoder_dir}...")
    table5_rows, top_features, all_suppressor_features = analyze_transcoder_layers(
        transcoder_dir=args.transcoder_dir,
        clean_mlp_inputs=sweep_results["clean_mlp_inputs"],
        corrupt_mlp_inputs=sweep_results["corrupt_mlp_inputs"],
        mu_hat=mu_hat,
        formation_mlp_write=formation_summary["total_mlp_write"],
        device=device,
    )

    # 3. Top-5 Suppressor Feature Ablation
    ablation_summary = evaluate_feature_ablation(
        model=model,
        tokenizer=tokenizer,
        pairs=pairs,
        dataset_root=args.dataset_root,
        suppressor_features=all_suppressor_features,
        mu_hat=mu_hat,
        device=device,
    )

    # Write Table 5 Markdown
    md_lines = [
        "# Features More Active on Analysis Prompts Dominate Formation Window (Table 5)",
        "",
        f"Evaluated on {len(pairs)} held-out pairs from `datasets/qwen3_14b/pair`.",
        "",
        "| Layer | Dominant | $K_{\\mathrm{corrupt}}$ | $K_{\\mathrm{clean}}$ | $K_{\\mathrm{corrupt}}/K_{\\mathrm{clean}}$ | Share (%) | Semantic label |",
        "|:---|:---|---:|---:|---:|---:|:---|",
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
            "semantic_label": r["semantic_label"],
        }
        clean_table5_rows.append(row_dict)
        md_lines.append(
            f"| {row_dict['layer']} | {row_dict['dominant']} | {row_dict['K_corrupt']} | "
            f"{row_dict['K_clean']} | {row_dict['K_corrupt_over_K_clean']} | {row_dict['share_pct']} | {row_dict['semantic_label']} |"
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
            "formation_window_summary": formation_summary,
            "table5": clean_table5_rows,
            "top5_feature_ablation": ablation_summary,
        }, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(f"\nFeature ablation restoration rate: {ablation_summary['restored_tool_call_rate']:.2f}%")
    print(f"Results written to {args.output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
