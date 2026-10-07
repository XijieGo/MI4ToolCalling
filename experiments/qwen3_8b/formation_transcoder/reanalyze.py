#!/usr/bin/env python3
"""Recompute fixed-window Qwen3-8B Transcoder accounting and causal effects."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[3]
LAYERS = (20, 21, 22, 23)
spec = importlib.util.spec_from_file_location("qwen3_formation_run", ROOT / "experiments/qwen3_8b/formation_transcoder/run.py")
original = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(original)

from mi4tc.paths import model_path, transcoder_root


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def dump_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def load_model(args):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(args.model_path), local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(args.model_path), dtype=torch.bfloat16,
        attn_implementation="sdpa", local_files_only=True,
    ).to(args.device).eval()
    if tokenizer.convert_tokens_to_ids("<tool_call>") != original.TOOL_CALL_ID:
        raise ValueError("Unexpected tool-call token ID")
    return model, tokenizer


def capture(args) -> None:
    model, tokenizer = load_model(args)
    mu_raw, mu_hat = original.load_coding_vector(args.vector_path)
    result = {
        "shard": args.shard, "num_shards": args.num_shards,
        "layers": list(LAYERS), "commitment_layer": 24, "commitment_position": "pre",
        "mu_raw": mu_raw, "mu_hat": mu_hat, "vector_sha256": digest(args.vector_path),
        "dataset_sha256": digest(args.dataset_root / "pairs.jsonl"),
        "script_sha256": digest(Path(__file__)), "torch_version": torch.__version__,
        "model_path": str(args.model_path), "gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "splits": {},
    }
    for split in ("train", "heldout"):
        all_pairs = original.load_pairs(args.dataset_root, split)
        indices = list(range(args.shard, len(all_pairs), args.num_shards))
        pairs = [all_pairs[i] for i in indices]
        collected = {side: {} for side in ("clean", "corrupt")}
        for start in range(0, len(pairs), args.batch_size):
            batch = pairs[start:start + args.batch_size]
            for side in ("clean", "corrupt"):
                ids, mask, pos = original.make_input_batch(tokenizer, batch, args.dataset_root, side, torch.device(args.device))
                rows = torch.arange(len(batch), device=args.device)
                holders, handles = {}, []
                def save_pre(name):
                    def hook(module, inputs):
                        holders[name] = inputs[0][rows, pos].detach().cpu()
                    return hook
                def save_output(name):
                    def hook(module, inputs, output):
                        value = output[0] if isinstance(output, tuple) else output
                        holders[name] = value[rows, pos].detach().cpu()
                    return hook
                for layer in LAYERS:
                    block = model.model.layers[layer]
                    handles.append(block.post_attention_layernorm.register_forward_pre_hook(save_pre(f"pre_norm_{layer}")))
                    handles.append(block.mlp.register_forward_pre_hook(save_pre(f"mlp_input_{layer}")))
                    handles.append(block.mlp.register_forward_hook(save_output(f"mlp_output_{layer}")))
                    handles.append(block.self_attn.register_forward_hook(save_output(f"attn_output_{layer}")))
                handles.append(model.model.layers[24].register_forward_pre_hook(save_pre("l24_pre")))
                try:
                    with torch.inference_mode():
                        hidden = model.model(input_ids=ids, attention_mask=mask, use_cache=False).last_hidden_state
                        logits = model.lm_head(hidden[rows, pos]).float()
                        holders["top1"] = logits.argmax(-1).cpu()
                        holders["tool_logit"] = logits[:, original.TOOL_CALL_ID].cpu()
                        logits[:, original.TOOL_CALL_ID] = -torch.inf
                        holders["tool_margin"] = holders["tool_logit"] - logits.max(-1).values.cpu()
                finally:
                    for handle in handles:
                        handle.remove()
                for name, value in holders.items():
                    collected[side].setdefault(name, []).append(value)
            print(f"capture shard {args.shard} {split} {min(start + len(batch), len(pairs))}/{len(pairs)}", flush=True)
        result["splits"][split] = {
            "indices": indices, "pairs": pairs,
            **{side: {name: torch.cat(values) for name, values in data.items()} for side, data in collected.items()},
        }
        torch.save(result, args.output_root / f"capture_{args.shard:02d}.pt")
    print(f"capture shard {args.shard} complete", flush=True)


def merge_capture(args):
    pieces = [torch.load(args.output_root / f"capture_{i:02d}.pt", weights_only=False) for i in range(args.num_shards)]
    for p in pieces:
        assert p["layers"] == list(LAYERS)
        assert p["commitment_position"] == "pre"
        assert p["vector_sha256"] == pieces[0]["vector_sha256"] == digest(args.vector_path)
        assert p["dataset_sha256"] == pieces[0]["dataset_sha256"] == digest(args.dataset_root / "pairs.jsonl")
    merged = {k: v for k, v in pieces[0].items() if k not in ("splits", "shard", "gpu")}
    merged["splits"] = {}
    for split in ("train", "heldout"):
        parts = [p["splits"][split] for p in pieces]
        indices = torch.tensor([i for p in parts for i in p["indices"]])
        order = indices.argsort()
        assert torch.equal(indices[order], torch.arange(len(indices)))
        pairs = [item for p in parts for item in p["pairs"]]
        pairs = [pairs[i] for i in order.tolist()]
        assert pairs == original.load_pairs(args.dataset_root, split)
        merged["splits"][split] = {
            "pairs": pairs,
            **{side: {name: torch.cat([p[side][name] for p in parts])[order] for name in parts[0][side]} for side in ("clean", "corrupt")},
        }
    return merged


def encode_dataset(weights, xs, ys, device, batch_size):
    dense, predictions = [], []
    for start in range(0, len(xs), batch_size):
        x = xs[start:start + batch_size].to(device=device, dtype=weights["W_enc"].dtype)
        with torch.inference_mode():
            a = F.relu(F.linear(x, weights["W_enc"], weights["b_enc"]))
            yhat = a @ weights["W_dec"] + weights["b_dec"]
        dense.append(a.cpu())
        predictions.append(yhat.cpu())
    a = torch.cat(dense)
    pred = torch.cat(predictions).double()
    target = ys.double()
    sse = (pred - target).square().sum().item()
    denom_global = (target - target.mean()).square().sum().item()
    denom_channel = (target - target.mean(0)).square().sum().item()
    return a, {
        "n_prompts": len(xs), "mean_l0": float((a > 0).sum(1).double().mean()),
        "mse": sse / target.numel(), "variance_explained_global": 1 - sse / denom_global,
        "variance_explained_channel": 1 - sse / denom_channel,
        "mean_output_cosine": float(F.cosine_similarity(pred.float(), target.float()).mean()),
        "mean_prediction_norm": float(pred.norm(dim=1).mean()),
        "mean_target_norm": float(target.norm(dim=1).mean()),
    }, pred


def layer_analysis(args):
    from safetensors.torch import load_file
    data = merge_capture(args)
    layer = args.layer
    if layer not in LAYERS:
        raise ValueError("Only fixed L20--L23 may be analyzed")
    path = args.transcoder_dir / f"layer_{layer}.safetensors"
    if Path(str(path) + ".aria2").exists():
        raise ValueError("Checkpoint download is incomplete")
    weights = {k: v.to(args.device) for k, v in load_file(str(path)).items()}
    if set(weights) != {"W_enc", "b_enc", "W_dec", "b_dec"}:
        raise ValueError("Unexpected activation function or skip-connection parameters")
    mu = data["mu_hat"].to(args.device, dtype=torch.float32)
    with torch.inference_mode():
        beta = (weights["W_dec"].float() @ mu).cpu().double()
    eps = json.loads((args.model_path / "config.json").read_text())["rms_norm_eps"]
    summary = {"layer": layer, "checkpoint": str(path), "checkpoint_sha256": digest(path),
               "activation_function": "ReLU", "fixed_vector_sha256": data["vector_sha256"],
               "dataset_sha256": data["dataset_sha256"], "splits": {}, "input_diagnostics": {}}
    stats = {"beta": beta, "layer": layer, "splits": {}}
    for split, d in data["splits"].items():
        all_acts, quality = {}, {}
        for side in ("clean", "corrupt"):
            x = d[side][f"mlp_input_{layer}"]
            y = d[side][f"mlp_output_{layer}"]
            a, q, pred = encode_dataset(weights, x, y, args.device, args.batch_size)
            all_acts[side] = a
            quality[side] = q
            print(f"L{layer} {split} {side}: VE={q['variance_explained_global']:.4f}, L0={q['mean_l0']:.1f}", flush=True)
        mean_clean = all_acts["clean"].double().mean(0)
        mean_corrupt = all_acts["corrupt"].double().mean(0)
        delta = mean_clean - mean_corrupt
        kappa = delta * beta
        kc = float(kappa[delta < 0].abs().sum())
        ke = float(kappa[delta > 0].abs().sum())
        actual_write = float(((d["clean"][f"mlp_output_{layer}"].float() - d["corrupt"][f"mlp_output_{layer}"].float()) @ data["mu_hat"]).double().mean())
        summary["splits"][split] = {
            "n_pairs": len(d["pairs"]), "K_corrupt": kc, "K_clean": ke,
            "K_corrupt_over_K_clean": kc / ke, "all_feature_abs_kappa": float(kappa.abs().sum()),
            "net_feature_write": float(kappa.sum()), "actual_mlp_write": actual_write,
            "quality": quality,
        }
        assert abs(kc + ke - float(kappa.abs().sum())) < 1e-9
        stats["splits"][split] = {"mean_clean": mean_clean, "mean_corrupt": mean_corrupt, "delta": delta, "kappa": kappa,
                                  "activation_clean": all_acts["clean"], "activation_corrupt": all_acts["corrupt"]}
    # Diagnose the three distinct hook conventions on a fixed training subset.
    # These measurements never choose an intervention layer or select features.
    d = data["splits"]["train"]
    for variant in ("rms_without_affine", "pre_norm"):
        for side in ("clean", "corrupt"):
            raw = d[side][f"pre_norm_{layer}"][:32]
            x = raw
            if variant == "rms_without_affine":
                x = (raw.float() * (raw.float().square().mean(-1, keepdim=True) + eps).rsqrt()).to(raw.dtype)
            _, q, _ = encode_dataset(weights, x, d[side][f"mlp_output_{layer}"][:32], args.device, args.batch_size)
            summary["input_diagnostics"][f"{variant}_{side}"] = q
            print(f"L{layer} diagnostic {variant} {side}: VE={q['variance_explained_global']:.4f}, L0={q['mean_l0']:.1f}", flush=True)
    torch.save(stats, args.output_root / f"feature_stats_L{layer}.pt")
    dump_json(args.output_root / f"layer_L{layer}.json", summary)
    print(f"L{layer} analysis complete", flush=True)


def select_features(args):
    candidates = []
    for layer in LAYERS:
        s = torch.load(args.output_root / f"feature_stats_L{layer}.pt", weights_only=False)
        train = s["splits"]["train"]
        ids = torch.where((train["delta"] < 0) & (s["beta"] < 0))[0]
        ids = ids[train["kappa"][ids].abs().argsort(descending=True)[:100]]
        for f in ids.tolist():
            candidates.append({
                "layer": layer, "feature_idx": f, "abs_kappa": float(train["kappa"][f].abs()),
                "kappa": float(train["kappa"][f]), "delta_a": float(train["delta"][f]),
                "beta": float(s["beta"][f]), "mean_a_clean": float(train["mean_clean"][f]),
                "mean_a_corrupt": float(train["mean_corrupt"][f]),
            })
    candidates.sort(key=lambda x: x["abs_kappa"], reverse=True)
    top5 = candidates[:5]
    control = original.select_layer_matched_random_control(candidates, top5, seed=42)
    selected = {
        "selection_split": "train", "selection_pair_count": 300,
        "selection_rule": "Top five corrupt-higher, mu-hat-opposing features by training |kappa| over fixed L20--L23",
        "control_rule": "Seed-42 sampling from the remaining top-100 training suppressors per layer, with identical layer counts",
        "top5_features": top5, "layer_matched_random_suppressor_control_features": control,
    }
    dump_json(args.output_root / "feature_selection.json", selected)
    print(json.dumps(selected, indent=2), flush=True)


def causal_forward(model, ids, mask, pos, payloads):
    rows = torch.arange(len(ids), device=ids.device)
    captured, handles = {}, []
    def capture_l24(module, inputs):
        captured["l24_pre"] = inputs[0][rows, pos].detach().float().cpu()
    handles.append(model.model.layers[24].register_forward_pre_hook(capture_l24))
    for layer, payload in payloads.items():
        def zero_write(p):
            def hook(module, inputs, output):
                acts = F.relu(F.linear(inputs[0][rows, pos].to(p["W_enc"].dtype), p["W_enc"], p["b_enc"]))
                delta = acts @ p["W_dec"]
                result = output.clone()
                result[rows, pos] -= delta.to(result.dtype)
                return result
            return hook
        handles.append(model.model.layers[layer].mlp.register_forward_hook(zero_write(payload)))
    try:
        with torch.inference_mode():
            hidden = model.model(input_ids=ids, attention_mask=mask, use_cache=False).last_hidden_state
            logits = model.lm_head(hidden[rows, pos]).float()
            captured["top1"] = logits.argmax(-1).cpu()
            captured["tool_logit"] = logits[:, original.TOOL_CALL_ID].cpu()
            logits[:, original.TOOL_CALL_ID] = -torch.inf
            captured["tool_margin"] = captured["tool_logit"] - logits.max(-1).values.cpu()
    finally:
        for handle in handles:
            handle.remove()
    return captured


def causal(args):
    data = merge_capture(args)
    selected = json.loads((args.output_root / "feature_selection.json").read_text())
    model, tokenizer = load_model(args)
    d = data["splits"]["heldout"]
    indices = list(range(args.shard, len(d["pairs"]), args.num_shards))
    pairs = [d["pairs"][i] for i in indices]
    baseline = {f"{side}_{name}": d[side][name][indices].float() if name != "top1" else d[side][name][indices]
                for side in ("clean", "corrupt") for name in ("l24_pre", "top1", "tool_logit", "tool_margin")}
    results = {}
    for name, key in [("top5", "top5_features"), ("random", "layer_matched_random_suppressor_control_features")]:
        payloads = original.load_feature_payloads(args.transcoder_dir, selected[key], torch.device(args.device))
        collected = {}
        for start in range(0, len(pairs), args.batch_size):
            batch = pairs[start:start + args.batch_size]
            ids, mask, pos = original.make_input_batch(tokenizer, batch, args.dataset_root, "corrupt", torch.device(args.device))
            result = causal_forward(model, ids, mask, pos, payloads)
            for k, v in result.items():
                collected.setdefault(k, []).append(v)
            print(f"causal shard {args.shard} {name} {min(start + len(batch), len(pairs))}/{len(pairs)}", flush=True)
        results[name] = {k: torch.cat(v) for k, v in collected.items()}
    torch.save({"indices": indices, "baseline": baseline, "results": results}, args.output_root / f"causal_{args.shard:02d}.pt")
    for name, result in results.items():
        print(name, json.dumps(original.summarize_zero_ablation(baseline, result, data["mu_hat"])), flush=True)


def family_rules():
    """Load the fixed semantic rules shipped alongside this runner."""
    path = Path(__file__).with_name("family_rules.py")
    spec = importlib.util.spec_from_file_location("mi4tc_family_rules", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return vars(module), path


def family_analysis(args):
    from safetensors import safe_open
    from transformers import AutoTokenizer
    from collections import Counter
    data = merge_capture(args)
    layer = args.layer
    s = torch.load(args.output_root / f"feature_stats_L{layer}.pt", weights_only=False)
    tr = s["splits"]["train"]
    rule, rule_path = family_rules()
    positive = torch.where(tr["kappa"] > 0)[0]
    negative = torch.where(tr["kappa"] < 0)[0]
    ids = torch.cat((positive[tr["kappa"][positive].argsort(descending=True)[:80]], negative[tr["kappa"][negative].argsort()[:80]]))
    ids = ids[tr["kappa"][ids].abs().argsort(descending=True)]
    with safe_open(args.transcoder_dir / f"layer_{layer}.safetensors", framework="pt") as f:
        decoder = f.get_tensor("W_dec")[ids].to(args.device)
    weight_map = json.loads((args.model_path / "model.safetensors.index.json").read_text())["weight_map"]
    with safe_open(args.model_path / weight_map["lm_head.weight"], framework="pt") as f:
        unembedding = f.get_tensor("lm_head.weight").to(args.device)
    tokenizer = AutoTokenizer.from_pretrained(str(args.model_path), local_files_only=True)
    with torch.inference_mode():
        scores = decoder @ unembedding.T
        top = scores.topk(20, dim=1).indices.cpu()
        bottom = scores.topk(20, dim=1, largest=False).indices.cpu()
    pairs = data["splits"]["train"]["pairs"]
    candidates = []
    for j, fid in enumerate(ids.tolist()):
        tokens_top = tuple(tokenizer.decode([v]) for v in top[j].tolist())
        tokens_bottom = tuple(tokenizer.decode([v]) for v in bottom[j].tolist())
        acts = torch.cat((tr["activation_clean"][:, fid], tr["activation_corrupt"][:, fid])).float()
        examples = []
        for pidx in torch.argsort(acts, descending=True, stable=True)[:5].tolist():
            side = "clean" if pidx < len(pairs) else "corrupt"
            pair = pairs[pidx % len(pairs)]
            examples.append({"sample_id": pair["sample_id"], "side": side, "activation": float(acts[pidx]),
                             "verb": pair[f"{side}_verb"], "prompt": (args.dataset_root / pair[f"{side}_relpath"]).read_text()[-1600:]})
        verbs = tuple(v for v, _ in Counter(e["verb"] for e in examples).most_common(3))
        pattern = rule["alignment_pattern"](float(tr["delta"][fid]), float(s["beta"][fid]))
        fs = rule["score_family_membership"](pattern=pattern, top_tokens=tokens_top, bottom_tokens=tokens_bottom, common_verbs=verbs)
        best = max(fs, key=fs.get)
        family = best if fs[best] >= rule["family_threshold"](best) else ""
        if family == "no_need_non_existence":
            family = "analysis_non_execution"
        candidates.append({
            "layer": layer, "feature_idx": fid, "assigned_family": family, "semantic_score": fs[best],
            "family_scores": fs, "pattern": pattern, "train_kappa": float(tr["kappa"][fid]),
            "heldout_kappa": float(s["splits"]["heldout"]["kappa"][fid]),
            "beta": float(s["beta"][fid]), "common_verbs": verbs,
            "top_decoder_tokens": tokens_top, "bottom_decoder_tokens": tokens_bottom,
            "train_max_activating_examples": examples,
        })
    dump_json(args.output_root / f"family_candidates_L{layer}.json", {
        "rule_source": str(rule_path), "rule_source_sha256": digest(rule_path),
        "selection_split": "train", "candidate_limit_per_sign_per_layer": 80,
        "decoder_token_projection": "Native decoder row dotted with the native HF lm_head row",
        "candidates": candidates,
    })
    print(f"L{layer} semantic candidates complete: {Counter(x['assigned_family'] for x in candidates)}", flush=True)


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def finalize(args):
    from collections import defaultdict
    import shutil
    data = merge_capture(args)
    assert args.dataset_root.resolve() == (ROOT / "datasets/qwen3_8b/pair").resolve()
    assert len(data["splits"]["train"]["pairs"]) == 300
    assert len(data["splits"]["heldout"]["pairs"]) == 200
    evaluated = data["splits"]["heldout"]
    mismatches = []
    for split, d in data["splits"].items():
        for p in d["pairs"]:
            for side in ("clean", "corrupt"):
                path = args.dataset_root / p[f"{side}_relpath"]
                assert path.is_relative_to(args.dataset_root / split)
                if digest(path) != p[f"{side}_prompt_sha256"]:
                    mismatches.append(str(path))
    if mismatches:
        raise ValueError(f"Prompt hash mismatch: {mismatches[:3]}")
    stats = {layer: torch.load(args.output_root / f"feature_stats_L{layer}.pt", weights_only=False) for layer in LAYERS}
    summary = {
        "model": "Qwen3-8B", "dataset_root": str(args.dataset_root.resolve()),
        "evaluation_directory": str((args.dataset_root / "heldout").resolve()),
        "dataset_sha256": data["dataset_sha256"], "vector_path": str(args.vector_path),
        "vector_sha256": data["vector_sha256"], "n_train_for_feature_selection": 300, "n_heldout": 200,
        "formation_layers": list(LAYERS), "intervention_state": "L24 pre at the prediction position",
        "transcoder_dir": str(args.transcoder_dir), "transcoder_activation": "ReLU",
        "transcoder_input": "Native normalized MLP input, including the learned RMSNorm weight",
        "kappa_definition": "(mean_clean_activation - mean_corrupt_activation) * dot(W_dec[f], mu_hat)",
        "heldout_table5": [], "train_selection_table5": [],
    }
    for split, table_key in (("heldout", "heldout_table5"), ("train", "train_selection_table5")):
        net = sum(float(s["splits"][split]["kappa"].sum()) for s in stats.values())
        for layer, s in stats.items():
            tr = s["splits"][split]
            kc = float(tr["kappa"][tr["delta"] < 0].abs().sum())
            ke = float(tr["kappa"][tr["delta"] > 0].abs().sum())
            row = {"layer": f"L{layer}", "dominant": "Corrupt" if kc > ke else "Clean", "K_corrupt": kc, "K_clean": ke,
                   "K_corrupt_over_K_clean": kc / ke, "share_pct": 100 * float(tr["kappa"].sum()) / net,
                   "net_feature_write": float(tr["kappa"].sum())}
            summary[table_key].append(row)
        kc = sum(r["K_corrupt"] for r in summary[table_key])
        ke = sum(r["K_clean"] for r in summary[table_key])
        summary[f"{split}_totals"] = {"K_corrupt": kc, "K_clean": ke, "K_corrupt_over_K_clean": kc / ke,
                                     "all_feature_abs_kappa": kc + ke, "net_feature_write": net}
    parts = [torch.load(args.output_root / f"causal_{i:02d}.pt", weights_only=False) for i in range(args.num_shards)]
    indices = torch.tensor([i for p in parts for i in p["indices"]])
    order = indices.argsort()
    assert torch.equal(indices[order], torch.arange(200))
    baseline = {k: torch.cat([p["baseline"][k] for p in parts])[order] for k in parts[0]["baseline"]}
    for side in ("clean", "corrupt"):
        for k in ("l24_pre", "top1", "tool_logit", "tool_margin"):
            assert torch.equal(baseline[f"{side}_{k}"], evaluated[side][k].float() if k != "top1" else evaluated[side][k])
    selection = json.loads((args.output_root / "feature_selection.json").read_text())
    summary["top5_feature_ablation"] = dict(selection)
    per_sample = [{"sample_id": p["sample_id"]} for p in evaluated["pairs"]]
    for name, key in (("top5", "top5_zero_ablation"), ("random", "layer_matched_random_suppressor_zero_control")):
        result = {k: torch.cat([p["results"][name][k] for p in parts])[order] for k in parts[0]["results"][name]}
        measurements = original.summarize_zero_ablation(baseline, result, data["mu_hat"])
        measurements["intervention_position"] = "Only each prompt's last non-padding input token at its source MLP output"
        measurements["baseline_clean_tool_call_top1_count"] = int((baseline["clean_top1"] == original.TOOL_CALL_ID).sum())
        measurements["eligible_strict_recovery_count"] = int((baseline["corrupt_top1"] != original.TOOL_CALL_ID).sum())
        measurements["strict_recovery_rate_among_eligible"] = 100 * measurements["strict_recovery_count"] / measurements["eligible_strict_recovery_count"]
        summary["top5_feature_ablation"][key] = measurements
        shifts = (result["l24_pre"] - baseline["corrupt_l24_pre"]) @ data["mu_hat"]
        for i, row in enumerate(per_sample):
            row[f"{name}_baseline_top1"] = int(baseline["corrupt_top1"][i])
            row[f"{name}_post_top1"] = int(result["top1"][i])
            row[f"{name}_strict_recovery"] = int(baseline["corrupt_top1"][i] != original.TOOL_CALL_ID and result["top1"][i] == original.TOOL_CALL_ID)
            row[f"{name}_state_shift"] = float(shifts[i])
            row[f"{name}_tool_margin_shift"] = float(result["tool_margin"][i] - baseline["corrupt_tool_margin"][i])
    groups = defaultdict(list)
    for layer in LAYERS:
        candidates = json.loads((args.output_root / f"family_candidates_L{layer}.json").read_text())["candidates"]
        for row in candidates:
            if row["assigned_family"]:
                groups[row["assigned_family"]].append(row)
    family_rows = []
    family_members = []
    assigned_keys = set()
    for family, members in sorted(groups.items()):
        total = 0.0
        for member in members:
            layer, f = member["layer"], member["feature_idx"]
            assert (layer, f) not in assigned_keys
            assigned_keys.add((layer, f))
            heldout_k = float(stats[layer]["splits"]["heldout"]["kappa"][f])
            assert heldout_k == member["heldout_kappa"]
            total += abs(heldout_k)
            family_members.append({"family": family, "layer": layer, "feature_idx": f,
                                   "train_kappa": member["train_kappa"], "heldout_kappa": heldout_k,
                                   "abs_heldout_kappa": abs(heldout_k), "pattern": member["pattern"]})
        assert total <= summary["heldout_totals"]["all_feature_abs_kappa"] + 1e-9
        family_rows.append({"family": family, "n_features": len(members), "heldout_abs_kappa_sum": total,
                            "heldout_mean_abs_kappa": total / len(members),
                            "train_abs_kappa_sum": sum(abs(m["train_kappa"]) for m in members),
                            "n_train_corrupt_higher": sum(m["pattern"].startswith("corrupt_higher") for m in members),
                            "n_train_suppressors": sum(m["pattern"] == "corrupt_higher_write_away_from_gate" for m in members)})
    family_total = sum(r["heldout_abs_kappa_sum"] for r in family_rows)
    assert family_total <= summary["heldout_totals"]["all_feature_abs_kappa"] + 1e-9
    summary["feature_families"] = {
        "selection_split": "train", "evaluation_split": "heldout",
        "selection_rule": "Original semantic keyword and activation-pattern rules applied to the 80 largest positive and 80 largest negative training-kappa candidates per layer; all assigned candidates retained, without a family-size cap",
        "rows": family_rows, "assigned_abs_kappa_sum": family_total,
        "unassigned_abs_kappa_sum": summary["heldout_totals"]["all_feature_abs_kappa"] - family_total,
        "subset_total_le_all_feature_total": True,
    }
    d = evaluated
    mu = data["mu_hat"]
    component_rows = []
    for layer in LAYERS:
        writes = {component: float(((d["clean"][f"{component}_output_{layer}"].float() - d["corrupt"][f"{component}_output_{layer}"].float()) @ mu).double().mean()) for component in ("mlp", "attn")}
        component_rows.append({"layer": layer, "mlp_write": writes["mlp"], "attn_write": writes["attn"]})
        layer_summary = json.loads((args.output_root / f"layer_L{layer}.json").read_text())
        layer_summary["splits"]["heldout"]["actual_mlp_write"] = writes["mlp"]
        dump_json(args.output_root / f"layer_L{layer}.json", layer_summary)
    mlp_sum, attn_sum = (sum(r[f"{k}_write"] for r in component_rows) for k in ("mlp", "attn"))
    summary["formation_window_summary"] = {"formation_layers": list(LAYERS), "total_mlp_write": mlp_sum, "total_attn_write": attn_sum,
                                            "mlp_share_pct": 100 * mlp_sum / (mlp_sum + attn_sum), "attn_share_pct": 100 * attn_sum / (mlp_sum + attn_sum),
                                            "mlp_over_attn": mlp_sum / attn_sum}
    summary["verification"] = {"prompt_hash_mismatches": 0, "heldout_directory_only": True,
                                "full_feature_count_per_layer": 163840, "full_feature_count_window": 4 * 163840,
                                "feature_mass_partition_identity": True, "family_subset_bound": True,
                                "train_only_feature_selection": True, "causal_baseline_matches_capture": True,
                                "fixed_formation_layers": True, "fixed_vector_and_pre_state": True}
    dest = args.output_root / "results"
    dest.mkdir(exist_ok=True)
    dump_json(dest / "formation_transcoder_summary.json", summary)
    write_csv(dest / "table5_transcoder_features.csv", summary["heldout_table5"])
    write_csv(dest / "train_selection_table5.csv", summary["train_selection_table5"])
    write_csv(dest / "feature_family_summary.csv", family_rows)
    write_csv(dest / "feature_family_members.csv", family_members)
    write_csv(dest / "feature_ablation_per_sample.csv", per_sample)
    write_csv(dest / "formation_window_writes.csv", component_rows)
    top_rows = []
    for layer, s in stats.items():
        for f in s["splits"]["heldout"]["kappa"].abs().argsort(descending=True)[:20].tolist():
            tr = s["splits"]["heldout"]
            top_rows.append({"layer": layer, "feature_idx": f, "kappa": float(tr["kappa"][f]), "abs_kappa": float(tr["kappa"][f].abs()),
                             "delta_a": float(tr["delta"][f]), "beta": float(s["beta"][f]),
                             "dominant_side": "clean" if tr["delta"][f] > 0 else "corrupt"})
    write_csv(dest / "top_features_l20_l23.csv", top_rows)
    lines = ["# Qwen3-8B L20--L23 Transcoder feature analysis", "", f"Evaluation: `{args.dataset_root / 'heldout'}` (200 pairs). Feature selection: `train` (300 pairs).", "",
             "Vector: saved coding direction; intervention state: L24 pre. Features are encoded from the native normalized MLP input with ReLU. Kappa is evaluated at each prompt's last non-padding input token.", "",
             "| Layer | K_corrupt | K_clean | Ratio | Net-write share (%) |", "|---|---:|---:|---:|---:|"]
    for r in summary["heldout_table5"]:
        lines.append(f"| {r['layer']} | {r['K_corrupt']:.4f} | {r['K_clean']:.4f} | {r['K_corrupt_over_K_clean']:.4f} | {r['share_pct']:.2f} |")
    totals = summary["heldout_totals"]
    lines += ["", f"Window totals: K_corrupt={totals['K_corrupt']:.8f}, K_clean={totals['K_clean']:.8f}, ratio={totals['K_corrupt_over_K_clean']:.8f}, total |kappa|={totals['all_feature_abs_kappa']:.8f}.", "",
              "Family labels use the original fixed semantic rules on training candidates. Counts refer to those labelled candidates; all family means and sums below use exactly the same held-out kappa tensors as the full-feature accounting. No family-size cap or held-out selection is applied.", "",
              "| Family | Features | Mean heldout abs(kappa) | Sum heldout abs(kappa) |", "|---|---:|---:|---:|"]
    for r in family_rows:
        lines.append(f"| {r['family']} | {r['n_features']} | {r['heldout_mean_abs_kappa']:.8f} | {r['heldout_abs_kappa_sum']:.8f} |")
    lines += ["", f"All labelled-family contributions sum to {family_total:.8f}, bounded by the full-feature total {totals['all_feature_abs_kappa']:.8f}. Individual feature IDs and their unrounded contributions are in `feature_family_members.csv`; every feature's activation and contribution is retained in the parent `feature_stats_L*.pt` files.", ""]
    for name, key in (("Top-five zero ablation", "top5_zero_ablation"), ("Layer-matched random control", "layer_matched_random_suppressor_zero_control")):
        r = summary["top5_feature_ablation"][key]
        lines.append(f"{name}: strict recovery {r['strict_recovery_count']}/200 ({r['strict_recovery_rate']:.1f}%); post-intervention call rate {r['post_ablation_tool_call_top1_rate']:.1f}%; mean L24 shift {r['mean_l24_delta_along_mu_hat']:.6f}; gap closure {100*r['fraction_of_l24_gap_closed']:.3f}%; mean margin shift {r['mean_tool_call_margin_delta']:.6f}.")
    lines += ["", "The analysis reports four-layer feature accounting, training-selected family annotations, top-five zero ablation and a matched random control. Companion mediation and L34 readout analyses are provided by `complete_controls.py`.", "",
              "Per-layer reconstruction metrics and input-encoding diagnostics are recorded in the `layer_L*.json` outputs.", ""]
    (dest / "README.md").write_text("\n".join(lines))
    shutil.copy2(Path(__file__), args.output_root / "analysis_source.py")
    print(json.dumps({"results": str(dest), "totals": totals, "families": family_rows,
                      "top5": summary["top5_feature_ablation"]["top5_zero_ablation"],
                      "random": summary["top5_feature_ablation"]["layer_matched_random_suppressor_zero_control"],
                      "verification": summary["verification"]}, indent=2), flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("capture", "layer", "select", "causal", "families", "finalize"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, default=model_path("qwen3_8b"))
    parser.add_argument("--transcoder-dir", type=Path, default=transcoder_root() / "Qwen3-8B")
    parser.add_argument("--vector-path", type=Path, default=ROOT / "results/transfer/qwen3_8b/coding_vector.pt")
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "datasets/qwen3_8b/pair")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--layer", type=int)
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    return args


if __name__ == "__main__":
    args = parse_args()
    {"capture": capture, "layer": layer_analysis, "select": select_features, "causal": causal, "families": family_analysis, "finalize": finalize}[args.phase](args)
