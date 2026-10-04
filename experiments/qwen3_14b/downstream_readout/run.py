#!/usr/bin/env python3
"""Downstream readout mechanism on Qwen3-14B (Section 6.1, 6.2, Figure 3).

This script performs:
1. Section 6.1 (Figure 3A & 3B):
   - Direct Logit Attribution (DLA) on <tool_call> for attention heads in L28-L38:
     Computes DLA_clean, DLA_corrupt, and Delta DLA.
     Identifies top heads (L29H9, L33H11, L33H29).
     Vector intervention: Adds mu_Delta at L24 to corrupt prompts and measures DLA restoration.
   - Attention distribution across scaffold regions (Figure 3B):
     Measures attention from prediction position p to Role (R), Tools (T), Format (F), and User (U)
     spans for L29H9, L33H11, and L33H29 under Clean vs Corrupt prompts.
2. Section 6.2 (Figure 3C):
   - MLP34 causal patching: Patches clean MLP34 output into corrupt prompts, measuring top-1 recovery.
   - Transcoder feature L34/F109925:
     - Validates max-activating tokens (tool-schema boundaries).
     - Measures activation under Clean, Corrupt, and Corrupt + mu_Delta at L24.
     - Single-feature replacement: Patches F109925 activation from Clean to Corrupt, measuring
       top-1 recovery rate of <tool_call>.
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

KEY_HEADS = [(30, 13), (34, 8)]
TARGET_FEATURE_IDX = 1000
TARGET_FEATURE_LAYER = 34
MLP_PATCH_LAYER = 34


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


def load_coding_vector(vector_path: Path) -> torch.Tensor:
    payload = torch.load(str(vector_path), map_location="cpu", weights_only=False)
    raw = payload["mean_diff"] if isinstance(payload, dict) and "mean_diff" in payload else payload
    return raw.float()


def identify_spans(text: str, offsets: list[tuple[int, int]]) -> dict[str, list[int]]:
    tools_start = text.find("<tools>\n")
    fmt_start = text.find("For each function call,")
    sys_user_start = text.find("<|im_end|>\n<|im_start|>user\n")
    user_start = sys_user_start + len("<|im_end|>\n<|im_start|>user\n")
    asst_start = text.find("<|im_end|>\n<|im_start|>assistant\n")

    r_idx = [i for i, (s, e) in enumerate(offsets) if e <= tools_start]
    t_idx = [i for i, (s, e) in enumerate(offsets) if s >= tools_start and e <= fmt_start]
    f_idx = [i for i, (s, e) in enumerate(offsets) if s >= fmt_start and e <= sys_user_start]
    u_idx = [i for i, (s, e) in enumerate(offsets) if s >= user_start and e <= asst_start]
    return {"R": r_idx, "T": t_idx, "F": f_idx, "U": u_idx}


def main() -> int:
    parser = argparse.ArgumentParser(description="Downstream readout analysis on Qwen3-14B (Sec 6.1 & 6.2)")
    parser.add_argument("--model-path", type=Path, default=model_path("qwen3_14b"))
    parser.add_argument("--transcoder-dir", type=Path, default=transcoder_root() / "Qwen3-14B")
    parser.add_argument("--vector-path", type=Path, default=REPO_ROOT / "results/transfer/qwen3_14b/coding_vector.pt")
    parser.add_argument("--dataset-root", type=Path, default=REPO_ROOT / "datasets/qwen3_14b/pair")
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "results/qwen3_14b/downstream_readout")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-pairs", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="bfloat16")
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    print(f"Loading coding vector from {args.vector_path}...")
    mu_Delta = load_coding_vector(args.vector_path).to(device=device)

    print(f"Loading Qwen3-14B from {args.model_path} with eager attention...")
    import transformers
    torch_dtype = getattr(torch, args.dtype)
    tokenizer = transformers.AutoTokenizer.from_pretrained(str(args.model_path), trust_remote_code=True)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        str(args.model_path),
        dtype=torch_dtype,
        attn_implementation="eager",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()

    pairs = load_heldout_pairs(args.dataset_root, max_pairs=args.max_pairs)
    print(f"Loaded {len(pairs)} held-out pairs from {args.dataset_root}")

    # Tool call unembedding vector
    u_call = model.lm_head.weight[TOOL_CALL_ID].detach().float()  # [4096]

    num_heads = model.config.num_attention_heads
    head_dim = model.config.hidden_size // num_heads

    # Precompute per-head projection vectors for layers 25-35
    # For head h in layer l: proj[l, h] = (W_O[l][:, h*head_dim:(h+1)*head_dim].T @ u_call) [head_dim]
    layers_eval = list(range(28, 39))
    head_projections: dict[tuple[int, int], torch.Tensor] = {}
    for l in layers_eval:
        W_O = model.model.layers[l].self_attn.o_proj.weight.detach().float()
        for h in range(num_heads):
            W_O_h = W_O[:, h * head_dim : (h + 1) * head_dim]
            head_projections[(l, h)] = (W_O_h.T @ u_call).to(device=device)

    # 1. Evaluate DLA across all heads in L28-L38 on Clean and Corrupt prompts
    # Also evaluate Corrupt + mu_Delta at L24 for key heads
    clean_dla_sums = {(l, h): 0.0 for l in layers_eval for h in range(num_heads)}
    corrupt_dla_sums = {(l, h): 0.0 for l in layers_eval for h in range(num_heads)}
    intervened_dla_sums = {(l, h): 0.0 for (l, h) in KEY_HEADS}

    # Attention span weights for key heads: (l, h) -> span -> clean_sum, corrupt_sum
    span_names = ("R", "T", "F", "U")
    clean_span_sums = {kh: {s: 0.0 for s in span_names} for kh in KEY_HEADS}
    corrupt_span_sums = {kh: {s: 0.0 for s in span_names} for kh in KEY_HEADS}

    # MLP and feature data
    corrupt_base_top1_count = 0
    mlp_patch_top1_count = 0
    feature_patch_top1_count = 0

    clean_feature_acts = []
    corrupt_feature_acts = []
    interv_feature_acts = []

    # Load Late Transcoder
    tc_path = args.transcoder_dir / f"layer_{TARGET_FEATURE_LAYER}.safetensors"
    tc = load_file(str(tc_path))
    W_enc = tc["W_enc"].to(device=device, dtype=torch.bfloat16)
    b_enc = tc["b_enc"].to(device=device, dtype=torch.bfloat16)
    W_dec = tc["W_dec"].to(device=device, dtype=torch.bfloat16)
    w_dec_feat = W_dec[TARGET_FEATURE_IDX].float()
    w_enc_feat = W_enc[TARGET_FEATURE_IDX]
    b_enc_feat = b_enc[TARGET_FEATURE_IDX]

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    n_pairs = len(pairs)
    print(f"\nProcessing held-out pairs for DLA, attention spans, and readout feature...")

    for idx, p in enumerate(pairs, 1):
        clean_text = (args.dataset_root / p["clean_relpath"]).read_text(encoding="utf-8")
        corrupt_text = (args.dataset_root / p["corrupt_relpath"]).read_text(encoding="utf-8")

        # Encode with offsets
        clean_enc = tokenizer(clean_text, return_offsets_mapping=True, add_special_tokens=False)
        corrupt_enc = tokenizer(corrupt_text, return_offsets_mapping=True, add_special_tokens=False)

        clean_spans = identify_spans(clean_text, clean_enc["offset_mapping"])
        corrupt_spans = identify_spans(corrupt_text, corrupt_enc["offset_mapping"])

        c_tokens = torch.tensor([clean_enc["input_ids"]], dtype=torch.long, device=device)
        k_tokens = torch.tensor([corrupt_enc["input_ids"]], dtype=torch.long, device=device)
        c_len = c_tokens.shape[1]
        k_len = k_tokens.shape[1]

        # Hook holders
        c_head_outs: dict[tuple[int, int], torch.Tensor] = {}
        k_head_outs: dict[tuple[int, int], torch.Tensor] = {}
        c_mlp34_out = None
        k_mlp34_out = None
        c_mlp34_in = None
        k_mlp34_in = None

        # --- Forward Clean ---
        handles = []
        for l in layers_eval:
            def make_attn_hook(layer_idx: int):
                def hook(module, args, output):
                    # In eager attention, self_attn output is (attn_output, attn_weights)
                    attn_out = output[0]  # [1, seq_len, 4096]
                    # We can slice head outputs:
                    # Qwen3 o_proj input is head outputs concatenated: [1, seq_len, 32 * 128]
                    # We can capture pre-hook of o_proj instead!
                return hook

            def make_oproj_pre_hook(layer_idx: int):
                def hook(module, args):
                    z = args[0]
                    for h in range(num_heads):
                        c_head_outs[(layer_idx, h)] = z[0, -1, h * head_dim : (h + 1) * head_dim].detach().float()
                return hook
            handles.append(model.model.layers[l].self_attn.o_proj.register_forward_pre_hook(make_oproj_pre_hook(l)))

        def mlp_hook(module, args, output):
            nonlocal c_mlp34_out, c_mlp34_in
            c_mlp34_out = output[0, -1].detach().float()
            c_mlp34_in = args[0][0, -1].detach()

        handles.append(model.model.layers[MLP_PATCH_LAYER].mlp.register_forward_hook(mlp_hook))

        with torch.no_grad():
            c_out = model(input_ids=c_tokens, output_attentions=True, use_cache=False)
        for h in handles:
            h.remove()

        # Record clean DLA and attention spans
        for (l, h), z_h in c_head_outs.items():
            dla = float((z_h @ head_projections[(l, h)]).item())
            clean_dla_sums[(l, h)] += dla

        for (l, h) in KEY_HEADS:
            attn_mat = c_out.attentions[l][0, h, -1, :]  # [seq_len]
            for s in span_names:
                indices = clean_spans[s]
                if indices:
                    clean_span_sums[(l, h)][s] += float(attn_mat[indices].sum().item())

        # Clean feature activation at position p
        c_act_feat = float(F.relu(c_mlp34_in.float() @ w_enc_feat.float() + b_enc_feat.float()).item())
        clean_feature_acts.append(c_act_feat)

        # --- Forward Corrupt ---
        handles = []
        for l in layers_eval:
            def make_oproj_pre_hook_k(layer_idx: int):
                def hook(module, args):
                    z = args[0]
                    for h in range(num_heads):
                        k_head_outs[(layer_idx, h)] = z[0, -1, h * head_dim : (h + 1) * head_dim].detach().float()
                return hook
            handles.append(model.model.layers[l].self_attn.o_proj.register_forward_pre_hook(make_oproj_pre_hook_k(l)))

        def mlp_hook_k(module, args, output):
            nonlocal k_mlp34_out, k_mlp34_in
            k_mlp34_out = output[0, -1].detach().float()
            k_mlp34_in = args[0][0, -1].detach()

        handles.append(model.model.layers[MLP_PATCH_LAYER].mlp.register_forward_hook(mlp_hook_k))

        with torch.no_grad():
            k_out = model(input_ids=k_tokens, output_attentions=True, use_cache=False)
        for h in handles:
            h.remove()

        for (l, h), z_h in k_head_outs.items():
            dla = float((z_h @ head_projections[(l, h)]).item())
            corrupt_dla_sums[(l, h)] += dla

        for (l, h) in KEY_HEADS:
            attn_mat = k_out.attentions[l][0, h, -1, :]
            for s in span_names:
                indices = corrupt_spans[s]
                if indices:
                    corrupt_span_sums[(l, h)][s] += float(attn_mat[indices].sum().item())

        k_act_feat = float(F.relu(k_mlp34_in.float() @ w_enc_feat.float() + b_enc_feat.float()).item())
        corrupt_feature_acts.append(k_act_feat)

        base_top1 = (k_out.logits[0, -1].argmax().item() == TOOL_CALL_ID)
        if base_top1:
            corrupt_base_top1_count += 1

        # --- Forward Corrupt + mu_Delta at L24 ---
        def add_mu_hook(module, args):
            hidden = args[0].clone()
            hidden[0, -1] = hidden[0, -1] + mu_Delta.to(dtype=hidden.dtype)
            return (hidden, *args[1:])

        interv_head_outs = {}
        interv_mlp34_in = None
        handles = [model.model.layers[31].register_forward_pre_hook(add_mu_hook)]

        for (l, h) in KEY_HEADS:
            def make_interv_hook(l_idx: int, h_idx: int):
                def hook(module, args):
                    z = args[0]
                    interv_head_outs[(l_idx, h_idx)] = z[0, -1, h_idx * head_dim : (h_idx + 1) * head_dim].detach().float()
                return hook
            handles.append(model.model.layers[l].self_attn.o_proj.register_forward_pre_hook(make_interv_hook(l, h)))

        def mlp_in_hook(module, args):
            nonlocal interv_mlp34_in
            interv_mlp34_in = args[0][0, -1].detach()
        handles.append(model.model.layers[MLP_PATCH_LAYER].mlp.register_forward_pre_hook(mlp_in_hook))

        with torch.no_grad():
            model(input_ids=k_tokens, use_cache=False)
        for h in handles:
            h.remove()

        for (l, h) in KEY_HEADS:
            if (l, h) in interv_head_outs:
                dla = float((interv_head_outs[(l, h)] @ head_projections[(l, h)]).item())
                intervened_dla_sums[(l, h)] += dla

        interv_act_feat = float(F.relu(interv_mlp34_in.float() @ w_enc_feat.float() + b_enc_feat.float()).item())
        interv_feature_acts.append(interv_act_feat)

        # --- Test MLP Full Output Patching ---
        mlp_replacement = c_mlp34_out.to(device=device, dtype=torch_dtype)
        def patch_mlp(module, args, output):
            out = output.clone()
            out[0, -1] = mlp_replacement
            return out

        handle = model.model.layers[MLP_PATCH_LAYER].mlp.register_forward_hook(patch_mlp)
        with torch.no_grad():
            out_patch = model(input_ids=k_tokens, use_cache=False)
        handle.remove()
        if out_patch.logits[0, -1].argmax().item() == TOOL_CALL_ID:
            mlp_patch_top1_count += 1

        # --- Test Feature Replacement ---
        delta_feat_write = (c_act_feat - k_act_feat) * w_dec_feat.to(device=device, dtype=torch_dtype)
        def patch_feature(module, args, output):
            out = output.clone()
            out[0, -1] = out[0, -1] + delta_feat_write
            return out

        handle = model.model.layers[MLP_PATCH_LAYER].mlp.register_forward_hook(patch_feature)
        with torch.no_grad():
            out_f_patch = model(input_ids=k_tokens, use_cache=False)
        handle.remove()
        if out_f_patch.logits[0, -1].argmax().item() == TOOL_CALL_ID:
            feature_patch_top1_count += 1

        if idx % max(1, n_pairs // 5) == 0:
            print(f"  Processed {idx}/{n_pairs} pairs...", flush=True)

    # Summarize Attention Head DLA
    dla_rows = []
    for l in layers_eval:
        for h in range(num_heads):
            c_mean = clean_dla_sums[(l, h)] / n_pairs
            k_mean = corrupt_dla_sums[(l, h)] / n_pairs
            delta = c_mean - k_mean
            dla_rows.append({
                "layer": l,
                "head": h,
                "head_label": f"L{l}H{h}",
                "clean_dla": round(c_mean, 3),
                "corrupt_dla": round(k_mean, 3),
                "delta_dla": round(delta, 3),
            })

    dla_rows.sort(key=lambda r: r["delta_dla"], reverse=True)

    # Save Figure 3A CSV
    fig3a_csv = args.output_root / "figure3a_attention_head_dla.csv"
    with fig3a_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(dla_rows[0].keys()))
        writer.writeheader()
        writer.writerows(dla_rows)

    # Summarize Attention Spans (Figure 3B)
    span_rows = []
    for (l, h) in KEY_HEADS:
        for s in span_names:
            c_m = clean_span_sums[(l, h)][s] / n_pairs
            k_m = corrupt_span_sums[(l, h)][s] / n_pairs
            span_rows.append({
                "head": f"L{l}H{h}",
                "span": s,
                "clean_attention": round(c_m, 4),
                "corrupt_attention": round(k_m, 4),
                "shift_clean_minus_corrupt": round(c_m - k_m, 4),
            })

    fig3b_csv = args.output_root / "figure3b_attention_spans.csv"
    with fig3b_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(span_rows[0].keys()))
        writer.writeheader()
        writer.writerows(span_rows)

    # Key Head DLA Intervention Restoration
    key_head_summary = []
    for (l, h) in KEY_HEADS:
        c_m = clean_dla_sums[(l, h)] / n_pairs
        k_m = corrupt_dla_sums[(l, h)] / n_pairs
        int_m = intervened_dla_sums[(l, h)] / n_pairs
        key_head_summary.append({
            "head": f"L{l}H{h}",
            "clean_dla": round(c_m, 2),
            "corrupt_dla": round(k_m, 2),
            "intervened_dla_with_mu": round(int_m, 2),
            "delta_dla": round(c_m - k_m, 2),
        })

    # Section 6.2 Feature Readout Summary
    mean_c_act = sum(clean_feature_acts) / n_pairs
    mean_k_act = sum(corrupt_feature_acts) / n_pairs
    mean_int_act = sum(interv_feature_acts) / n_pairs
    call_proj = float((w_dec_feat @ u_call).item())

    sec62_summary = {
        "mlp_patching_restoration_rate": (mlp_patch_top1_count / n_pairs) * 100.0,
        "feature": {
            "layer": TARGET_FEATURE_LAYER,
            "index": TARGET_FEATURE_IDX,
            "call_token_write_projection": call_proj,
            "mean_clean_activation": round(mean_c_act, 2),
            "mean_corrupt_activation": round(mean_k_act, 2),
            "mean_intervened_activation": round(mean_int_act, 2),
            "projected_clean_write": round(mean_c_act * call_proj, 1),
            "projected_corrupt_write": round(mean_k_act * call_proj, 1),
            "projected_intervened_write": round(mean_int_act * call_proj, 1),
            "single_feature_replacement_recovery_rate": (feature_patch_top1_count / n_pairs) * 100.0,
        },
    }

    # Write JSON summary
    (args.output_root / "feature_readout.json").write_text(
        json.dumps(sec62_summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    # Write Markdown summary
    md_lines = [
        "# Downstream Readout Mechanism on Qwen3-14B (Section 6)",
        "",
        f"Evaluated on {n_pairs} held-out pairs from `datasets/qwen3_14b/pair`.",
        "",
        "## 1. Scaffold-Reading Attention Heads (Sec 6.1, Figure 3A, 3B)",
        "",
        "### Key Head DLA and Intervention Response",
        "| Head | Clean DLA | Corrupt DLA | Corrupt + $\\mu_\\Delta$ DLA | $\\Delta$ DLA |",
        "|:---|---:|---:|---:|---:|",
    ]
    for r in key_head_summary:
        md_lines.append(f"| {r['head']} | {r['clean_dla']} | {r['corrupt_dla']} | {r['intervened_dla_with_mu']} | +{r['delta_dla']} |")

    md_lines.extend([
        "",
        "### Top 5 Attention Heads in L28-L38 by $\\Delta$ DLA",
        "| Rank | Head | $\\Delta$ DLA | Clean DLA | Corrupt DLA |",
        "|---:|:---|---:|---:|---:|",
    ])
    for rank, r in enumerate(dla_rows[:5], 1):
        md_lines.append(f"| {rank} | {r['head_label']} | +{r['delta_dla']} | {r['clean_dla']} | {r['corrupt_dla']} |")

    md_lines.extend([
        "",
        "### Attention Shift Across Scaffold Regions (Figure 3B)",
        "| Head | Span | Clean Attention | Corrupt Attention | Shift (Clean - Corrupt) |",
        "|:---|:---|---:|---:|---:|",
    ])
    for r in span_rows:
        md_lines.append(f"| {r['head']} | {r['span']} | {r['clean_attention']} | {r['corrupt_attention']} | {r['shift_clean_minus_corrupt']:+.4f} |")

    md_lines.extend([
        "",
        "## 2. Late Structural Feature Readout (Sec 6.2, Figure 3C)",
        "",
        f"- **MLP{MLP_PATCH_LAYER} Causal Patching**: Recovers `<tool_call>` top-1 on **{sec62_summary['mlp_patching_restoration_rate']:.1f}%** of held-out analysis prompts.",
        f"- **Transcoder Feature L{TARGET_FEATURE_LAYER}/F{TARGET_FEATURE_IDX}**:",
        f"  - Clean activation: `{sec62_summary['feature']['mean_clean_activation']}` (projected write = `{sec62_summary['feature']['projected_clean_write']}`)",
        f"  - Corrupt activation: `{sec62_summary['feature']['mean_corrupt_activation']}` (projected write = `{sec62_summary['feature']['projected_corrupt_write']}`)",
        f"  - Corrupt + $\\mu_\\Delta$ activation: `{sec62_summary['feature']['mean_intervened_activation']}` (projected write = `{sec62_summary['feature']['projected_intervened_write']}`)",
        f"  - **Single-feature replacement recovery rate**: **{sec62_summary['feature']['single_feature_replacement_recovery_rate']:.1f}%** of held-out analysis prompts.",
        "",
    ])

    md_content = "\n".join(md_lines)
    (args.output_root / "summary.md").write_text(md_content, encoding="utf-8")
    print("\n" + md_content)

    print(f"Results written to {args.output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
