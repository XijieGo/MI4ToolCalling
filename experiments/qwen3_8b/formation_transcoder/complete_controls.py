#!/usr/bin/env python3
"""Fixed-window mediation and L34 single-feature tests on the current holdout."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open

import reanalyze as common

FEATURES = (109925, 91365)  # Original structural feature and its fixed historical control.


def forward(model, ids, mask, pos, *, mlp_swaps=None, restore=None, replace_state=None, capture_l34=False):
    rows = torch.arange(len(ids), device=ids.device)
    saved, handles = {}, []
    if restore is not None or replace_state is not None:
        def patch_state(module, inputs):
            hidden = inputs[0].clone()
            if replace_state is not None:
                hidden[rows, pos] = replace_state.to(hidden)
            else:
                hidden[rows, pos] = (hidden[rows, pos].float() + restore.to(hidden.device).float()).to(hidden.dtype)
            return (hidden, *inputs[1:])
        handles.append(model.model.layers[24].register_forward_pre_hook(patch_state))
    def capture_state(module, inputs):
        saved["l24_pre"] = inputs[0][rows, pos].detach().float().cpu()
    handles.append(model.model.layers[24].register_forward_pre_hook(capture_state))
    if mlp_swaps:
        for layer, value in mlp_swaps.items():
            def patch_output(target):
                def hook(module, inputs, output):
                    out = output.clone()
                    out[rows, pos] = target.to(out)
                    return out
                return hook
            handles.append(model.model.layers[layer].mlp.register_forward_hook(patch_output(value)))
    if capture_l34:
        def save_mlp(module, inputs, output):
            saved["mlp_input"] = inputs[0][rows, pos].detach().cpu()
            saved["mlp_output"] = output[rows, pos].detach().cpu()
        handles.append(model.model.layers[34].mlp.register_forward_hook(save_mlp))
    try:
        with torch.inference_mode():
            hidden = model.model(input_ids=ids, attention_mask=mask, use_cache=False).last_hidden_state
            logits = model.lm_head(hidden[rows, pos]).float()
            saved["top1"] = logits.argmax(-1).cpu()
            saved["tool_logit"] = logits[:, common.original.TOOL_CALL_ID].cpu()
            logits[:, common.original.TOOL_CALL_ID] = -torch.inf
            saved["tool_margin"] = saved["tool_logit"] - logits.max(-1).values.cpu()
    finally:
        for handle in handles:
            handle.remove()
    return saved


def mediation(args):
    cache = common.merge_capture(args)
    model, tokenizer = common.load_model(args)
    data = cache["splits"]["heldout"]
    indices = list(range(args.shard, len(data["pairs"]), args.num_shards))
    mu = cache["mu_hat"].float()
    collected = {}
    for side in ("clean", "corrupt"):
        other = "corrupt" if side == "clean" else "clean"
        for start in range(0, len(indices), args.batch_size):
            batch_indices = indices[start:start + args.batch_size]
            pairs = [data["pairs"][i] for i in batch_indices]
            ids, mask, pos = common.original.make_input_batch(tokenizer, pairs, args.dataset_root, side, torch.device(args.device))
            swaps = {layer: data[other][f"mlp_output_{layer}"][batch_indices] for layer in common.LAYERS}
            baseline_state = data[side]["l24_pre"][batch_indices].float()
            patched = forward(model, ids, mask, pos, mlp_swaps=swaps)
            compensation = ((baseline_state - patched["l24_pre"]) @ mu)[:, None] * mu[None, :]
            total_delta = baseline_state - patched["l24_pre"]
            runs = {
                "full_window_swap": patched,
                "mu_restore": forward(model, ids, mask, pos, mlp_swaps=swaps, restore=compensation),
                "orthogonal_restore": forward(model, ids, mask, pos, mlp_swaps=swaps, restore=total_delta - compensation),
                "full_state_restore": forward(model, ids, mask, pos, mlp_swaps=swaps, replace_state=baseline_state),
            }
            for mode, result in runs.items():
                for key, value in result.items():
                    collected.setdefault(f"{side}:{mode}:{key}", []).append(value)
            print(f"mediation shard {args.shard} {side} {min(start + len(pairs), len(indices))}/{len(indices)}", flush=True)
    torch.save({"indices": indices, "results": {k: torch.cat(v) for k, v in collected.items()},
                "dataset_sha256": cache["dataset_sha256"], "vector_sha256": cache["vector_sha256"]}, args.output_root / f"mediation_{args.shard:02d}.pt")
    print(f"mediation shard {args.shard} complete", flush=True)


def readout(args):
    cache = common.merge_capture(args)
    model, tokenizer = common.load_model(args)
    data = cache["splits"]["heldout"]
    indices = list(range(args.shard, len(data["pairs"]), args.num_shards))
    with safe_open(args.transcoder_dir / "layer_34.safetensors", framework="pt") as f:
        rows = torch.tensor(FEATURES)
        weights = {k: f.get_tensor(k)[rows].to(args.device) for k in ("W_enc", "b_enc", "W_dec")}
    u_call = model.lm_head.weight[common.original.TOOL_CALL_ID].detach().float()
    beta_call = (weights["W_dec"].float() @ u_call).cpu()
    beta_mu = (weights["W_dec"].float() @ cache["mu_hat"].to(args.device)).cpu()
    collected = {}
    for start in range(0, len(indices), args.batch_size):
        batch_indices = indices[start:start + args.batch_size]
        pairs = [data["pairs"][i] for i in batch_indices]
        batches = {side: common.original.make_input_batch(tokenizer, pairs, args.dataset_root, side, torch.device(args.device)) for side in ("clean", "corrupt")}
        runs = {side: forward(model, *batches[side], capture_l34=True) for side in ("clean", "corrupt")}
        raw = cache["mu_raw"].expand(len(pairs), -1)
        runs["corrupt_plus_mu"] = forward(model, *batches["corrupt"], restore=raw, capture_l34=True)
        activations = {}
        for name in ("clean", "corrupt", "corrupt_plus_mu"):
            x = runs[name]["mlp_input"].to(device=args.device, dtype=weights["W_enc"].dtype)
            activations[name] = F.relu(F.linear(x, weights["W_enc"], weights["b_enc"])).float().cpu()
            for key in ("top1", "tool_logit", "tool_margin"):
                collected.setdefault(f"{name}:{key}", []).append(runs[name][key])
            collected.setdefault(f"{name}:activations", []).append(activations[name])
        ids, mask, pos = batches["corrupt"]
        for j, fid in enumerate(FEATURES):
            delta = (activations["clean"][:, j] - activations["corrupt"][:, j]).to(args.device)
            contribution = delta[:, None] * weights["W_dec"][j].float()[None, :]
            rows = torch.arange(len(pairs), device=args.device)
            def add_feature(module, inputs, output):
                patched = output.clone()
                patched[rows, pos] = (patched[rows, pos].float() + contribution).to(patched.dtype)
                return patched
            handle = model.model.layers[34].mlp.register_forward_hook(add_feature)
            try:
                result = forward(model, ids, mask, pos)
            finally:
                handle.remove()
            for key in ("top1", "tool_logit", "tool_margin"):
                collected.setdefault(f"replace_F{fid}:{key}", []).append(result[key])
        result = forward(model, ids, mask, pos, mlp_swaps={34: runs["clean"]["mlp_output"]})
        for key in ("top1", "tool_logit", "tool_margin"):
            collected.setdefault(f"full_mlp34_patch:{key}", []).append(result[key])
        print(f"readout shard {args.shard} {min(start + len(pairs), len(indices))}/{len(indices)}", flush=True)
    torch.save({"indices": indices, "features": FEATURES, "beta_call": beta_call, "beta_mu": beta_mu,
                "results": {k: torch.cat(v) for k, v in collected.items()}, "dataset_sha256": cache["dataset_sha256"],
                "vector_sha256": cache["vector_sha256"]}, args.output_root / f"readout34_{args.shard:02d}.pt")
    print(f"readout shard {args.shard} complete", flush=True)


def collect_parts(args, prefix):
    cache = common.merge_capture(args)
    parts = [torch.load(args.output_root / f"{prefix}_{i:02d}.pt", weights_only=False) for i in range(args.num_shards)]
    for p in parts:
        assert p["dataset_sha256"] == cache["dataset_sha256"]
        assert p["vector_sha256"] == cache["vector_sha256"]
    indices = torch.tensor([i for p in parts for i in p["indices"]])
    order = indices.argsort()
    assert torch.equal(indices[order], torch.arange(200))
    values = {k: torch.cat([p["results"][k] for p in parts])[order] for k in parts[0]["results"]}
    return cache, values, parts[0]


def collect(args):
    cache, med, _ = collect_parts(args, "mediation")
    mu = cache["mu_hat"]
    data = cache["splits"]["heldout"]
    gap = float(((data["clean"]["l24_pre"].float() - data["corrupt"]["l24_pre"].float()) @ mu).mean())
    mediation_summary = {"dataset_root": str(args.dataset_root), "n_heldout": 200, "formation_layers": list(common.LAYERS),
                         "state": "L24 pre, last non-padding input token", "vector_sha256": cache["vector_sha256"],
                         "method": "Swap all L20--L23 MLP outputs at the prediction position with paired opposite-side baseline outputs; compensate the measured L24 projection change along the original mu_hat. Orthogonal compensation and full-state replacement are controls.",
                         "g_l_baseline": gap, "arms": {}}
    for side in ("clean", "corrupt"):
        baseline_proj = data[side]["l24_pre"].float() @ mu
        baseline_call = data[side]["top1"] == common.original.TOOL_CALL_ID
        patched_proj = med[f"{side}:full_window_swap:l24_pre"] @ mu
        target_other_proj = data["corrupt" if side == "clean" else "clean"]["l24_pre"].float() @ mu
        arm = {"baseline_calls": int(baseline_call.sum()), "baseline_call_rate": 100 * float(baseline_call.float().mean()), "modes": {}}
        for mode in ("full_window_swap", "mu_restore", "orthogonal_restore", "full_state_restore"):
            proj = med[f"{side}:{mode}:l24_pre"] @ mu
            calls = med[f"{side}:{mode}:top1"] == common.original.TOOL_CALL_ID
            residual_error = float((proj - baseline_proj).mean())
            lost = float((patched_proj - baseline_proj).mean())
            arm["modes"][mode] = {
                "calls": int(calls.sum()), "call_rate": 100 * float(calls.float().mean()),
                "decision_agreement_with_baseline": 100 * float((calls == baseline_call).float().mean()),
                "g_l": float((proj - target_other_proj).mean()) * (1 if side == "clean" else -1),
                "mean_projection_error_vs_baseline": residual_error,
                "projection_effect_recovered_fraction": 1 - abs(residual_error) / abs(lost) if lost else None,
                "mean_logit_delta": float((med[f"{side}:{mode}:tool_logit"] - data[side]["tool_logit"]).mean()),
                "mean_margin_delta": float((med[f"{side}:{mode}:tool_margin"] - data[side]["tool_margin"]).mean()),
            }
        mediation_summary["arms"][side] = arm
    cache, readout_data, first = collect_parts(args, "readout34")
    for side in ("clean", "corrupt"):
        assert torch.equal(readout_data[f"{side}:top1"], data[side]["top1"])
    readout_summary = {"dataset_root": str(args.dataset_root), "n_heldout": 200, "layer": 34,
                       "intervention_state": "L24 pre", "feature_intervention_position": "Last non-padding input token only",
                       "encoding": "ReLU of the native normalized MLP input", "target_feature": FEATURES[0], "fixed_control_feature": FEATURES[1],
                       "control_source": "Original L34_structural / k=1 non-structural control catalogue; fixed before this run", "features": {}, "interventions": {}}
    baseline = readout_data["corrupt:top1"] == common.original.TOOL_CALL_ID
    readout_summary["baseline_corrupt_calls"] = int(baseline.sum())
    for j, fid in enumerate(FEATURES):
        means = {name: float(readout_data[f"{name}:activations"][:, j].mean()) for name in ("clean", "corrupt", "corrupt_plus_mu")}
        readout_summary["features"][str(fid)] = {
            "mean_activations": means, "call_token_projection_per_unit": float(first["beta_call"][j]),
            "mean_projected_call_token_writes": {k: v * float(first["beta_call"][j]) for k, v in means.items()},
            "mu_projection_per_unit": float(first["beta_mu"][j]),
        }
    for mode in ("corrupt_plus_mu", f"replace_F{FEATURES[0]}", f"replace_F{FEATURES[1]}", "full_mlp34_patch"):
        calls = readout_data[f"{mode}:top1"] == common.original.TOOL_CALL_ID
        strict = (~baseline) & calls
        readout_summary["interventions"][mode] = {
            "post_calls": int(calls.sum()), "post_call_rate": 100 * float(calls.float().mean()),
            "strict_recoveries": int(strict.sum()), "strict_recovery_rate": 100 * float(strict.float().mean()),
            "mean_logit_delta": float((readout_data[f"{mode}:tool_logit"] - readout_data["corrupt:tool_logit"]).mean()),
            "mean_margin_delta": float((readout_data[f"{mode}:tool_margin"] - readout_data["corrupt:tool_margin"]).mean()),
        }
    common.dump_json(args.output_root / "results/mediation_summary.json", mediation_summary)
    common.dump_json(args.output_root / "results/readout34_summary.json", readout_summary)
    per_sample = []
    for i, p in enumerate(data["pairs"]):
        row = {"sample_id": p["sample_id"], "baseline_corrupt_top1": int(data["corrupt"]["top1"][i])}
        for mode in (f"replace_F{FEATURES[0]}", f"replace_F{FEATURES[1]}", "full_mlp34_patch"):
            row[f"{mode}_top1"] = int(readout_data[f"{mode}:top1"][i])
        for mode in ("full_window_swap", "mu_restore", "orthogonal_restore", "full_state_restore"):
            row[f"clean_{mode}_top1"] = int(med[f"clean:{mode}:top1"][i])
        per_sample.append(row)
    common.write_csv(args.output_root / "results/additional_controls_per_sample.csv", per_sample)
    print(json.dumps({"mediation": mediation_summary, "readout34": readout_summary}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("mediation", "readout", "collect"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, default=common.ROOT / "datasets/qwen3_8b/pair")
    parser.add_argument("--vector-path", type=Path, default=common.ROOT / "results/transfer/qwen3_8b/coding_vector.pt")
    parser.add_argument("--model-path", type=Path, default=common.model_path("qwen3_8b"))
    parser.add_argument("--transcoder-dir", type=Path, default=common.transcoder_root() / "Qwen3-8B")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    assert args.dataset_root.resolve() == (common.ROOT / "datasets/qwen3_8b/pair").resolve()
    {"mediation": mediation, "readout": readout, "collect": collect}[args.phase](args)


if __name__ == "__main__":
    main()
