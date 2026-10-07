#!/usr/bin/env python3
"""Measure fixed-layer scaffold, formation and readout experiments.

Outputs record component writes and feature contributions computed from
checkpoints or supplied feature-summary artifacts.
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).parent))
from mi4tc.pairs import load_model_native_pairs, native_pair_token_ids
from mi4tc.paths import released_transcoder_root
from tool_call_vector.run import (LOCKED, LayerSweep, check_marker, hidden_from_call,
                                 iter_batches, load_adapter, replace_hidden, replace_output, tensor_from_output)
from measure_mlp_attn_and_max_attn import CONFIGS
from recompute_scaffolds import (CONDITIONS, native_variant, neutral_text,
                                render_text_variant, token_regions)

RELEASE = released_transcoder_root()
TC_FAMILIES = {"qwen35_4b": "qwen35-4b", "qwen35_9b": "qwen35-9b",
               "granite_3p3_8b": "granite-3.3-8b-instruct", "mistral_3p2_24b": "mistral-small-3.2-24b"}
READOUT_LAYERS = {"qwen3_4b": list(range(25, 36)), "qwen3_14b": list(range(28, 40)),
                 "granite_3p3_8b": list(range(30, 40))}
ORIGINAL_KEY_HEADS = {"qwen3_4b": [(29, 9), (34, 1), (34, 15)],
                      "qwen3_14b": [(30, 13), (34, 8)], "granite_3p3_8b": [(34, 25), (34, 26)]}
ORIGINAL_LATE_MLP = {"qwen3_4b": 35, "qwen3_14b": 34}


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    tmp.replace(path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No measured rows for {path}")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def ratio(a: float, b: float) -> float | None:
    return a / b if abs(b) > 1e-12 else None


def stats(result: dict[str, torch.Tensor]) -> dict[str, float]:
    return {"mean_prob": float(result["tool_prob"].float().mean()),
            "top1_rate": float(result["tool_top1"].float().mean()),
            "mean_logit": float(result["tool_logit"].float().mean())}


class Context:
    def __init__(self, model: str, stage: str, run_dir: Path, feature_summary: Path | None = None):
        if model == "qwen3_8b":
            raise ValueError("Use experiments/qwen3_8b/ for Qwen3-8B mechanism stages")
        self.key, self.stage, self.run_dir = model, stage, run_dir
        self.feature_summary = feature_summary
        self.spec = LOCKED[model]
        locked = json.loads((run_dir / "fixed_layers.json").read_text())[model]
        if (locked["layer"], locked["hook"]) != (self.spec["layer"], self.spec["hook"]):
            raise ValueError("Locked layer/hook mismatch")
        self.pairs = load_model_native_pairs(self.spec["dataset"])
        assert len(self.pairs.train) == 300 and len(self.pairs.heldout) == 200
        print(f"LOAD {model} {stage}: train=300 heldout=200 layer={locked}", flush=True)
        self.adapter, _ = load_adapter(Path(self.spec["path"]), self.spec["loader"], "cuda")
        check_marker(self.adapter, self.spec["marker"], self.spec["marker_id"])
        self.unit = self.direction = None
        if stage != "scaffold":
            saved = torch.load(ROOT / "results/tool_call_vector" / model / "directions.pt", map_location="cpu", weights_only=True)
            assert saved["hook"] == locked["hook"] and set(saved["directions"]) == {locked["layer"]}
            self.direction = saved["directions"][locked["layer"]].float()
            if not torch.isfinite(self.direction).all() or self.direction.norm() <= 0:
                raise ValueError("Invalid fitted training vector")
            self.unit = self.direction / self.direction.norm()
        self.seqs = {}
        for split in ("train", "heldout"):
            values = [native_pair_token_ids(p, tokenizer=self.adapter.tokenizer) for p in self.pairs.split_pairs(split)]
            self.seqs[split] = {"clean": [list(x[0]) for x in values], "corrupt": [list(x[1]) for x in values]}
        self.sweep = LayerSweep(self.adapter, self.spec["marker_id"], [], 4096, self.spec["hook"], last_token_only=True)
        self.out = ROOT / "results" / model / {"scaffold": "scaffold_ablation", "formation": "formation_transcoder", "readout": "downstream_readout"}[stage]
        self.out.mkdir(parents=True, exist_ok=True)
        self.metadata = {"model_key": model, "layer": locked["layer"], "hook": locked["hook"],
                         "layer_index": "0-based", "n_train": 300, "n_heldout": 200,
                         "dataset": str(self.spec["dataset"]),
                         "manifest_sha256": hashlib.sha256((self.spec["dataset"] / "manifest.jsonl").read_bytes()).hexdigest(),
                         "run_dir": str(run_dir), "measurement_status": "computed", "fit_split": "train",
                         "forward_protocol": "same_length_batches_without_padding" if model.startswith("qwen35") else "left_padding_last_token_logits"}

    def run(self, sequences: list[list[int]], observer=None, patch=False,
            mlp_patches: dict[int, Any] | None = None, max_batch: int | None = None) -> dict[str, torch.Tensor]:
        # Group hybrid-model prompts by exact length. This uses larger batches
        # without adding padding tokens to recurrent-state evolution.
        maximum = max_batch or (8 if self.key.startswith("qwen35") else 4)
        results: dict[str, list[Any]] = {k: [None] * len(sequences) for k in ("tool_logit", "tool_prob", "tool_top1")}

        def consume(indices: list[int]) -> None:
            ids, mask, positions = self.sweep._pad([sequences[i] for i in indices])
            handles = []
            if observer is not None:
                observer.begin(indices, positions, ids.shape[1])
            if patch:
                layer = self.adapter.layers[self.spec["layer"]]
                def change(h):
                    h = h.clone()
                    h[:, -1] += self.direction.to(h.device, h.dtype)
                    return h
                if self.spec["hook"] == "pre":
                    handles.append(layer.register_forward_pre_hook(lambda m, a, k: replace_hidden(a, k, change(hidden_from_call(a, k))), with_kwargs=True))
                else:
                    handles.append(layer.register_forward_hook(lambda m, a, k, o: replace_output(o, change(tensor_from_output(o))), with_kwargs=True))
            for layer_index, ablator in (mlp_patches or {}).items():
                saved = {}
                def remember(m, a, k, saved=saved):
                    saved["x"] = hidden_from_call(a, k)[:, -1].detach()
                def ablate(m, a, k, o, saved=saved, ablator=ablator):
                    output = tensor_from_output(o).clone()
                    output[:, -1] -= ablator(saved["x"]).to(output.dtype)
                    return replace_output(o, output)
                mlp = self.adapter.layers[layer_index].mlp
                handles.extend([mlp.register_forward_pre_hook(remember, with_kwargs=True),
                                mlp.register_forward_hook(ablate, with_kwargs=True)])
            try:
                with torch.inference_mode():
                    logits = self.sweep._forward(ids, mask)[:, -1].float()
                    batch = {"tool_logit": logits[:, self.spec["marker_id"]],
                             "tool_prob": logits.softmax(-1)[:, self.spec["marker_id"]],
                             "tool_top1": logits.argmax(-1) == self.spec["marker_id"]}
                if observer is not None:
                    observer.end()
                for local, index in enumerate(indices):
                    for name in results:
                        results[name][index] = batch[name][local].detach().cpu()
            except torch.cuda.OutOfMemoryError:
                for h in handles:
                    h.remove()
                handles.clear()
                torch.cuda.empty_cache()
                if len(indices) == 1:
                    raise
                mid = len(indices)//2
                consume(indices[:mid]); consume(indices[mid:])
            finally:
                for h in handles:
                    h.remove()
        if self.key.startswith("qwen35"):
            groups = {}
            for i, seq in enumerate(sequences):
                groups.setdefault(len(seq), []).append(i)
            batches = []
            for length, group in sorted(groups.items()):
                size = max(1, min(maximum, 4096 // length))
                batches.extend(group[start:start+size] for start in range(0, len(group), size))
        else:
            batches = iter_batches([len(s) for s in sequences], 4096, max_batch=maximum)
        for step, batch in enumerate(batches):
            consume(batch)
            if step % 50 == 0:
                print(f"forward {step} / {len(sequences)} sequences", flush=True)
        return {name: torch.stack(values) for name, values in results.items()}


def scaffold(ctx: Context) -> None:
    import copy
    conditions = list(CONDITIONS)
    if ctx.key.startswith(("granite", "mistral")):
        # These native formats encode the call convention inside role/schema;
        # they do not expose a separate F component.
        conditions = ["RTF", "R-F", "R--", "-T-", "---", "R_TLEN_F"]
    requests = {c: {r: [] for r in ("neutral", "analysis", "execution")} for c in conditions}
    for i, pair in enumerate(ctx.pairs.heldout):
        if ctx.key.startswith("mistral"):
            clean = json.loads(pair.clean_path.read_text())
            corrupt = json.loads(pair.corrupt_path.read_text())
            idx = max(j for j, m in enumerate(clean["messages"]) if m["role"] == "user")
            neutral = neutral_text(clean["messages"][idx]["content"], corrupt["messages"][idx]["content"], i)
            for condition in conditions:
                requests[condition]["execution"].append(native_variant(ctx.adapter.tokenizer, clean, condition))
                requests[condition]["analysis"].append(native_variant(ctx.adapter.tokenizer, corrupt, condition))
                requests[condition]["neutral"].append(native_variant(ctx.adapter.tokenizer, clean, condition, neutral))
        else:
            neutral = neutral_text(pair.clean_text, pair.corrupt_text, i)
            for condition in conditions:
                for request, text in (("execution", pair.clean_text), ("analysis", pair.corrupt_text), ("neutral", neutral)):
                    requests[condition][request].append(render_text_variant(ctx.key, ctx.adapter.tokenizer, text, condition))
        if requests["RTF"]["execution"][-1] != ctx.seqs["heldout"]["clean"][i] or requests["RTF"]["analysis"][-1] != ctx.seqs["heldout"]["corrupt"][i]:
            raise ValueError(f"Full native scaffold changed sample {pair.sample_id}")
    report = dict(ctx.metadata, conditions={}, table4=[], distinct_format_component=ctx.key.startswith("qwen"))
    records = []
    for condition in conditions:
        condition_result = {}
        for request, sequences in requests[condition].items():
            print(f"scaffold {condition} {request}", flush=True)
            result = ctx.run(sequences)
            condition_result[request] = stats(result)
            for i, pair in enumerate(ctx.pairs.heldout):
                records.append({"sample_id": pair.sample_id, "scaffold_condition": condition, "request_type": request,
                                "tool_prob": float(result["tool_prob"][i]), "tool_logit": float(result["tool_logit"][i]), "top1": int(result["tool_top1"][i])})
        condition_result["delta_p"] = condition_result["neutral"]["mean_prob"] - condition_result["analysis"]["mean_prob"]
        report["conditions"][condition] = condition_result
        row = {"scaffold": condition, "neutral_p_call": condition_result["neutral"]["mean_prob"],
               "analysis_p_call": condition_result["analysis"]["mean_prob"], "execution_p_call": condition_result["execution"]["mean_prob"],
               "delta_p": condition_result["delta_p"]}
        row.update({f"{k}_top1": v["top1_rate"] for k, v in condition_result.items() if isinstance(v, dict)})
        report["table4"].append(row)
        write_json(ctx.out / "scaffold_ablation_summary.json", report)
    write_csv(ctx.out / "table4_scaffold_ablation.csv", report["table4"])
    write_csv(ctx.out / "sample_scores.csv", records)
    lines = [f"# {ctx.key}: native scaffold ablation", "", "Train 300 / held-out 200. The full condition reproduces every stored native prompt.", "",
             "| scaffold | neutral P(call) | analysis P(call) | execution P(call) | delta P |", "|---|---:|---:|---:|---:|"]
    for row in report["table4"]:
        lines.append(f"| {row['scaffold']} | {row['neutral_p_call']:.6f} | {row['analysis_p_call']:.6f} | {row['execution_p_call']:.6f} | {row['delta_p']:.6f} |")
    (ctx.out / "table4_scaffold_ablation.md").write_text("\n".join(lines)+"\n")


class FormationObserver:
    def __init__(self, ctx: Context, count: int, capture_layers: list[int]):
        self.ctx = ctx
        self.projections = {kind: torch.empty(count, len(ctx.adapter.layers)) for kind in ("pre", "post", "mlp", "attn")}
        self.inputs = {layer: torch.empty(count, ctx.direction.numel()) for layer in capture_layers}
        self.outputs = {layer: torch.empty(count, ctx.direction.numel()) for layer in capture_layers}
        self.handles = []
        self.scale = float(getattr(ctx.adapter.model.config, "residual_multiplier", 1.0))
        for index, block in enumerate(ctx.adapter.layers):
            def pre(m, a, k, index=index):
                self.seen.add((index, "pre"))
                h = hidden_from_call(a, k)[:, -1].detach().float().cpu()
                self.projections["pre"][self.indices, index] = h @ ctx.unit
            def post(m, a, k, o, index=index):
                self.seen.add((index, "post"))
                h = tensor_from_output(o)[:, -1].detach().float().cpu()
                self.projections["post"][self.indices, index] = h @ ctx.unit
            def mlp_pre(m, a, k, index=index):
                if index in self.inputs:
                    self.inputs[index][self.indices] = hidden_from_call(a, k)[:, -1].detach().float().cpu()
            def mlp_post(m, a, k, o, index=index):
                self.seen.add((index, "mlp"))
                h = tensor_from_output(o)[:, -1].detach().float().cpu()
                self.projections["mlp"][self.indices, index] = (h @ ctx.unit) * self.scale
                if index in self.outputs:
                    self.outputs[index][self.indices] = h
            def attn_post(m, a, k, o, index=index):
                self.seen.add((index, "attn"))
                h = tensor_from_output(o)[:, -1].detach().float().cpu()
                self.projections["attn"][self.indices, index] = (h @ ctx.unit) * self.scale
            attn = getattr(block, "self_attn", None) or getattr(block, "linear_attn", None)
            if attn is None:
                raise ValueError(f"Missing attention module at {index}")
            self.handles.extend([block.register_forward_pre_hook(pre, with_kwargs=True), block.register_forward_hook(post, with_kwargs=True),
                                 block.mlp.register_forward_pre_hook(mlp_pre, with_kwargs=True), block.mlp.register_forward_hook(mlp_post, with_kwargs=True),
                                 attn.register_forward_hook(attn_post, with_kwargs=True)])

    def begin(self, indices, positions, width):
        self.indices = indices
        self.seen = set()

    def end(self):
        expected = {(i, kind) for i in range(len(self.ctx.adapter.layers)) for kind in self.projections}
        if self.seen != expected:
            raise RuntimeError(f"Missing formation captures: {expected-self.seen}")

    def close(self):
        for h in self.handles:
            h.remove()


class Transcoder:
    def __init__(self, path: Path, device: str = "cuda"):
        self.path = path
        payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        args = payload["args"]
        if not isinstance(args, dict):
            args = vars(args)
        self.layer = int(payload["layer"])
        self.topk = int(args.get("topk") or 0)
        self.topk_input = args.get("topk_input") or "relu"
        weights = payload["transcoder"]
        self.enc = weights["W_enc"].to(device)
        self.enc_bias = weights["b_enc"].to(device)
        self.dec = weights["W_dec"].to(device)
        self.dec_bias = weights["b_dec"].to(device)
        self.features = self.enc.shape[0]
        self.args = {"topk": self.topk, "topk_input": self.topk_input, "d_feature": self.features,
                     "d_model": self.enc.shape[1], "step": int(payload["step"])}

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        pre = F.linear(inputs.to(self.enc.device, self.enc.dtype), self.enc, self.enc_bias)
        if self.topk > 0:
            scores = pre.relu() if self.topk_input == "relu" else pre
            values, indices = scores.topk(min(self.topk, self.features), dim=-1)
            return torch.zeros_like(scores).scatter_(-1, indices, values)
        return pre.relu()

    def measure(self, inputs: torch.Tensor, outputs: torch.Tensor) -> tuple[torch.Tensor, dict]:
        means = torch.zeros(self.features, device=self.enc.device, dtype=torch.float64)
        squared_error = 0.0
        active = 0
        for start in range(0, len(inputs), 32):
            x, actual = inputs[start:start+32], outputs[start:start+32].to(self.enc.device).float()
            with torch.inference_mode():
                activation = self.encode(x)
                means += activation.double().sum(0)
                active += int((activation != 0).sum())
                predicted = F.linear(activation.to(self.dec.dtype), self.dec.T, self.dec_bias).float()
                squared_error += float((predicted - actual).square().sum())
        # Match train_transcoder.py: target.var(unbiased=False) over tokens
        # and channels. Per-channel centering is a different metric and must
        # not be compared to the release's training evaluation.
        actual = outputs.float()
        denominator = float((actual - actual.mean()).square().sum())
        channel_denominator = float((actual - actual.mean(0)).square().sum())
        quality = {
            "variance_explained": 1.0 - ratio(squared_error, denominator) if denominator > 0 else None,
            "variance_explained_per_channel": 1.0 - ratio(squared_error, channel_denominator) if channel_denominator > 0 else None,
            "normalized_mse": ratio(squared_error, float(actual.square().sum())),
            "mean_active_features": active / len(inputs),
        }
        return (means / len(inputs)).float().cpu(), quality

    def contribution(self, clean, corrupt, unit, scale):
        beta = (self.dec.float() @ unit.to(self.dec.device)).cpu() * scale
        kappa = (clean - corrupt) * beta
        corrupt_higher, clean_higher = corrupt > clean, clean > corrupt
        kc = float(kappa[corrupt_higher].abs().sum())
        ke = float(kappa[clean_higher].abs().sum())
        return {"K_corrupt": kc, "K_clean": ke, "K_corrupt_over_K_clean": ratio(kc, ke),
                "suppressor_mass": float(kappa[corrupt_higher & (beta < 0)].abs().sum()),
                "driver_mass": float(kappa[clean_higher & (beta > 0)].abs().sum())}, kappa, beta


def feature_summary_metrics(path: Path) -> tuple[dict, dict]:
    """Read K accounting and ablation measurements from an input artifact."""
    payload = json.loads(path.read_text())
    table = payload.get("heldout_table5") or payload.get("table5") or []
    if table:
        kc = sum(float(row["K_corrupt"]) for row in table)
        ke = sum(float(row["K_clean"]) for row in table)
        metrics = {"K_corrupt": kc, "K_clean": ke,
                   "K_corrupt_over_K_clean": ratio(kc, ke),
                   "summary_layers": [row["layer"] for row in table]}
    else:
        metrics = {key: value for key, value in payload.get("transcoder", {}).items()
                   if key in {"K_corrupt", "K_clean", "K_corrupt_over_K_clean"}}
    value = metrics.get("K_corrupt_over_K_clean")
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"Expected finite K accounting in {path}")
    metrics.update(status="from_feature_summary", source=str(path.resolve()),
                   source_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    return metrics, payload.get("top5_feature_ablation", payload.get("feature_ablation", {}))


def formation(ctx: Context) -> None:
    paths = sorted((RELEASE / TC_FAMILIES[ctx.key]).glob("transcoder_layer*.pt")) if ctx.key in TC_FAMILIES else []
    capture_layers = sorted(int(p.stem.replace("transcoder_layer", "")) for p in paths)
    if ctx.key in TC_FAMILIES and not paths:
        raise FileNotFoundError(f"Missing released Transcoders: {ctx.key}")
    supplied_metrics = supplied_ablation = None
    if not paths:
        if ctx.feature_summary is None:
            raise ValueError(f"Provide --feature-summary from experiments/{ctx.key}/formation_transcoder/run.py")
        supplied_metrics, supplied_ablation = feature_summary_metrics(ctx.feature_summary)
    captured = {}
    baselines = {}
    for split in ("train", "heldout"):
        if split == "train" and not paths:
            continue
        captured[split] = {}
        for side in ("clean", "corrupt"):
            print(f"formation capture {split} {side}", flush=True)
            observer = FormationObserver(ctx, len(ctx.seqs[split][side]), capture_layers)
            try:
                baselines[split + "_" + side] = stats(ctx.run(ctx.seqs[split][side], observer))
            finally:
                observer.close()
            captured[split][side] = observer
    hc, ha = captured["heldout"]["clean"], captured["heldout"]["corrupt"]
    window = CONFIGS[ctx.key]["formation_layers"]
    rows = []
    for layer in range(len(ctx.adapter.layers)):
        row = {"layer": layer, "in_fixed_formation_window": layer in window}
        row.update({kind + "_write" if kind in {"mlp", "attn"} else kind + "_gap": float((hc.projections[kind][:, layer] - ha.projections[kind][:, layer]).mean()) for kind in hc.projections})
        rows.append(row)
    window_rows = [row for row in rows if row["in_fixed_formation_window"]]
    mlp = sum(row["mlp_write"] for row in window_rows)
    attn = sum(row["attn_write"] for row in window_rows)
    report = dict(ctx.metadata, formation_layers=window, trajectory=rows, baseline=baselines,
                  formation={"formation_layers": window, "total_mlp_write": mlp, "total_attn_write": attn,
                             "mlp_attn_ratio": ratio(mlp, attn), "mlp_share_pct": 100 * ratio(mlp, mlp + attn) if ratio(mlp, mlp + attn) is not None else None},
                  transcoder_layers=capture_layers, table5=[], selected_features=[])
    write_csv(ctx.out / "figure2_formation_trajectory.csv", rows)
    # Use released checkpoints literally, including each checkpoint's own
    # activation mode/top-k. The optimizer tensors remain memory-mapped on CPU.
    selection = {"suppressor": [], "driver": []}
    for path in paths:
        print(f"transcoder {path}", flush=True)
        tc = Transcoder(path)
        assert tc.layer in capture_layers
        measured = {}
        train_kappa = train_beta = None
        for split in ("train", "heldout"):
            means, quality = {}, {}
            for side in ("clean", "corrupt"):
                obs = captured[split][side]
                means[side], quality[side] = tc.measure(obs.inputs[tc.layer], obs.outputs[tc.layer])
            metrics, kappa, beta = tc.contribution(means["clean"], means["corrupt"], ctx.unit, hc.scale)
            measured[split] = dict(metrics, variance_explained_clean=quality["clean"]["variance_explained"],
                                   variance_explained_corrupt=quality["corrupt"]["variance_explained"],
                                   reconstruction_quality=quality)
            if split == "train":
                train_kappa, train_beta = kappa, beta
                for name, mask in (("suppressor", (means["corrupt"] > means["clean"]) & (beta < 0)),
                                   ("driver", (means["clean"] > means["corrupt"]) & (beta > 0))):
                    if tc.layer not in window:
                        continue
                    indices = torch.where(mask)[0]
                    if len(indices):
                        top = indices[kappa[indices].abs().topk(min(5, len(indices))).indices]
                        for f in top.tolist():
                            selection[name].append({"layer": tc.layer, "feature": f, "abs_kappa_train": float(kappa[f].abs()), "beta": float(beta[f])})
        top_indices = train_kappa.abs().topk(20).indices.tolist()
        row = {"layer": tc.layer, "in_fixed_formation_window": tc.layer in window, **measured["heldout"], **tc.args,
               "checkpoint": str(path), "train": measured["train"], "selection_split": "train",
               "top_features_selected_on_train": [{"feature": f, "kappa_train": float(train_kappa[f]), "beta": float(train_beta[f])} for f in top_indices]}
        report["table5"].append(row)
        write_json(ctx.out / "formation_transcoder_summary.json", report)
        del tc
        gc.collect(); torch.cuda.empty_cache()
    if paths:
        available_window = [r for r in report["table5"] if r["in_fixed_formation_window"]]
        kc, ke = sum(r["K_corrupt"] for r in available_window), sum(r["K_clean"] for r in available_window)
        fullkc, fullke = sum(r["K_corrupt"] for r in report["table5"]), sum(r["K_clean"] for r in report["table5"])
        report["transcoder"] = {"status": "measured_from_release", "K_corrupt": kc, "K_clean": ke, "K_corrupt_over_K_clean": ratio(kc, ke),
                                "measured_window_layers": [r["layer"] for r in available_window], "missing_window_layers": sorted(set(window)-set(capture_layers)),
                                "released_layer_K_corrupt": fullkc, "released_layer_K_clean": fullke, "released_layer_ratio": ratio(fullkc, fullke)}
        flagged = [r["layer"] for r in report["table5"]
                      if any(q["variance_explained"] is None or q["variance_explained"] < 0
                             for q in r["train"]["reconstruction_quality"].values())]
        report["transcoder"].update(
            variance_explained_definition="1 - SSE / sum((MLP_output - global_mean)**2), matching the trainer",
            reconstruction_flagged_release_layers=flagged,
            reconstruction_flagged_window_layers=sorted(set(flagged) & set(window)),
            quality_flag_split="train",
            quality_flag_criterion="Training clean or corrupt reconstruction VE < 0 or undefined; K aggregates all supplied fixed-window layers",
        )
        flat_rows = [{k: v for k, v in r.items() if not isinstance(v, (dict, list))} for r in report["table5"]]
        write_csv(ctx.out / "table5_transcoder_features.csv", flat_rows)
        # Feature candidates are ranked using train only; held-out prompts
        # never choose a feature. Save selections and matched random controls.
        for kind in selection:
            selection[kind] = sorted(selection[kind], key=lambda x: x["abs_kappa_train"], reverse=True)[:5]
        report["selected_features"] = selection
        report["feature_selection_layers"] = window
        report["feature_ablation"] = feature_ablation(ctx, paths, selection)
    else:
        report["transcoder"] = supplied_metrics
        report["feature_ablation"] = supplied_ablation
        write_csv(ctx.out / "table5_transcoder_features.csv", [{"model": ctx.key, **supplied_metrics}])
    write_json(ctx.out / "formation_transcoder_summary.json", report)
    (ctx.out / "summary.md").write_text(f"# {ctx.key}: fixed-layer formation\n\nFormation layers: {window}. Vector layer {ctx.spec['layer']} ({ctx.spec['hook']}).\n\nMLP write: {mlp:.6f}; attention write: {attn:.6f}; ratio: {ratio(mlp, attn)}.\n\nTranscoder: {json.dumps(report['transcoder'], ensure_ascii=False)}\n")


def feature_ablation(ctx, paths, selections):
    by_layer = {int(p.stem.replace("transcoder_layer", "")): p for p in paths}
    result = {"selection_split": "train", "selection_layers": CONFIGS[ctx.key]["formation_layers"],
              "n_features": 5, "random_seed": 20261006,
              "patch": "actual MLP output minus selected activation*decoder; reconstruction error retained", "arms": {}}
    generator = torch.Generator().manual_seed(20261006)
    for kind, features in selections.items():
        layers = sorted(set(f["layer"] for f in features))
        tcs = {}
        decoders = {}
        selected = {layer: [f["feature"] for f in features if f["layer"] == layer] for layer in layers}
        randoms = {}
        for layer in layers:
            tc = Transcoder(by_layer[layer])
            tcs[layer] = tc
            excluded = {f["feature"] for group in selections.values() for f in group if f["layer"] == layer}
            pool = [i for i in range(tc.features) if i not in excluded]
            order = torch.randperm(len(pool), generator=generator)[:len(selected[layer])]
            randoms[layer] = [pool[i] for i in order.tolist()]
            decoders[layer] = {"selected": tc.dec[selected[layer]].clone(), "random": tc.dec[randoms[layer]].clone()}
            # Only the encoder and a few decoder rows are needed to zero
            # selected features. Four full Mistral decoders waste 12 GiB.
            tc.dec = tc.dec_bias = None
            torch.cuda.empty_cache()
        side = "corrupt" if kind == "suppressor" else "clean"
        for name, ids in ((kind, selected), (kind + "_random", randoms)):
            patches = {}
            for layer in layers:
                tc, feature_ids = tcs[layer], ids[layer]
                decoder = decoders[layer]["random" if name.endswith("_random") else "selected"]
                def subtract(x, tc=tc, feature_ids=feature_ids, decoder=decoder):
                    with torch.inference_mode():
                        a = tc.encode(x)[:, feature_ids]
                        return a.to(decoder.dtype) @ decoder
                patches[layer] = subtract
            print(f"feature ablation {name}: {ids}", flush=True)
            measured = stats(ctx.run(ctx.seqs["heldout"][side], mlp_patches=patches))
            result["arms"][name] = dict(measured, side=side, features=ids)
        del tcs, patches, decoders
        if layers:
            del tc, decoder, subtract
        gc.collect(); torch.cuda.empty_cache()
    return result


class ReadoutObserver:
    """Observe last-query attention while retaining the original SDPA forward."""
    def __init__(self, ctx: Context, sequences: list[list[int]], side: str):
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
        from transformers.integrations.sdpa_attention import sdpa_attention_forward, repeat_kv
        self.registry = ALL_ATTENTION_FUNCTIONS
        self.original = self.registry["sdpa"]
        self.ctx, self.side = ctx, side
        self.layers = READOUT_LAYERS.get(ctx.key, CONFIGS[ctx.key]["readout_layers"])
        count = len(sequences)
        self.attention, self.heads, self.handles = {}, {}, []
        self.norm_rms = torch.empty(count)
        self.spans = []
        for i, pair in enumerate(ctx.pairs.heldout):
            text = pair.clean_text if side == "clean" else pair.corrupt_text
            self.spans.append(token_regions(ctx.key, ctx.adapter.tokenizer, sequences[i], text))
        self.lengths = [len(s) for s in sequences]
        parent = None
        for module in ctx.adapter.model.modules():
            layers = getattr(module, "layers", None)
            if layers is not None and len(layers) == len(ctx.adapter.layers) and layers[0] is ctx.adapter.layers[0]:
                parent = module
                break
        norm = getattr(parent, "norm", None)
        if norm is None:
            raise ValueError("Missing final residual normalization")
        self.eps = float(getattr(norm, "eps", getattr(norm, "variance_epsilon", 1e-6)))
        gain = norm.weight.detach().float()
        if "Qwen3_5RMSNorm" in type(norm).__name__:
            gain = 1 + gain
        unembed = ctx.adapter.model.get_output_embeddings().weight[ctx.spec["marker_id"]].detach().float()
        scale = float(getattr(ctx.adapter.model.config, "residual_multiplier", 1.0))
        logit_scale = float(getattr(ctx.adapter.model.config, "logits_scaling", 1.0))
        self.projectors = {}
        tracked = {}
        for layer in self.layers:
            module = getattr(ctx.adapter.layers[layer], "self_attn", None)
            if module is None:
                raise ValueError(f"Fixed readout layer {layer} is not full attention")
            head_dim = int(module.head_dim)
            heads = module.o_proj.in_features // head_dim
            if heads * head_dim != module.o_proj.in_features:
                raise ValueError("Head output size mismatch")
            tracked[id(module)] = layer
            self.attention[layer] = torch.empty(count, heads, 5)
            self.heads[layer] = torch.empty(count, heads)
            self.projectors[layer] = (module.o_proj.weight.detach().float().T @ (unembed * gain)).reshape(heads, head_dim) * scale / logit_scale
            def head_capture(m, a, k, layer=layer, heads=heads, head_dim=head_dim):
                self.seen.add((layer, "head"))
                h = hidden_from_call(a, k)[:, -1].float().reshape(-1, heads, head_dim)
                self.heads[layer][self.indices] = (h * self.projectors[layer]).sum(-1).detach().cpu()
            self.handles.append(module.o_proj.register_forward_pre_hook(head_capture, with_kwargs=True))
        def norm_capture(m, a, k):
            self.seen.add((0, "norm"))
            h = hidden_from_call(a, k)[:, -1].detach().float()
            self.norm_rms[self.indices] = (h.square().mean(-1) + self.eps).sqrt().cpu()
        self.handles.append(norm.register_forward_pre_hook(norm_capture, with_kwargs=True))

        def capture_attention(module, query, key, value, attention_mask, dropout=0.0,
                              scaling=None, is_causal=None, position_bias=None, **kwargs):
            layer = tracked.get(id(module))
            if layer is not None:
                self.seen.add((layer, "attention"))
                # Compute one query row, after RoPE and any Q/K normalization.
                # Never materialize all S*S weights just to retain the last row.
                repeated_key = repeat_kv(key, query.shape[1] // key.shape[1])
                scores = query[:, :, -1:, :].float() @ repeated_key.float().transpose(-1, -2)
                scores *= scaling if scaling is not None else query.shape[-1] ** -0.5
                if attention_mask is not None:
                    mask = attention_mask[..., -1:, :key.shape[-2]]
                    if mask.dtype == torch.bool:
                        scores.masked_fill_(~mask, float("-inf"))
                    else:
                        scores += mask.float()
                if position_bias is not None:
                    scores += position_bias[..., -1:, :key.shape[-2]].float()
                weights = scores.softmax(-1)[:, :, 0].detach().cpu()
                for local, index in enumerate(self.indices):
                    offset = self.width - self.lengths[index]
                    for j, name in enumerate(("R", "T", "F", "U", "history")):
                        tokens = [i + offset for i in self.spans[index][name]]
                        self.attention[layer][index, :, j] = weights[local, :, tokens].sum(-1) if tokens else 0.0
            return sdpa_attention_forward(module, query, key, value, attention_mask, dropout=dropout,
                                          scaling=scaling, is_causal=is_causal, position_bias=position_bias, **kwargs)
        self.registry.register("sdpa", capture_attention)
        self.seen = set()
        # Force the interface used by these observations. No eager S*S matrices.
        ctx.adapter.model.set_attn_implementation("sdpa")

    def begin(self, indices, positions, width):
        self.indices, self.width = indices, width
        self.seen = set()

    def end(self):
        expected = {(layer, kind) for layer in self.layers for kind in ("head", "attention")} | {(0, "norm")}
        if self.seen != expected:
            raise RuntimeError(f"Missing readout captures: {expected-self.seen}")

    def close(self):
        self.registry.register("sdpa", self.original)
        for h in self.handles:
            h.remove()

    def dla(self, layer):
        return self.heads[layer] / self.norm_rms[:, None]


def readout(ctx: Context) -> None:
    captures, scored = {}, {}
    for condition, side, patch in (("clean", "clean", False), ("corrupt", "corrupt", False), ("add_vector", "corrupt", True)):
        print(f"readout {condition}", flush=True)
        obs = ReadoutObserver(ctx, ctx.seqs["heldout"][side], side)
        try:
            scored[condition] = stats(ctx.run(ctx.seqs["heldout"][side], obs, patch=patch))
        finally:
            obs.close()
        captures[condition] = obs
    dla_rows, span_rows = [], []
    target = CONFIGS[ctx.key]["target_span"]
    span_index = ("R", "T", "F", "U", "history").index(target)
    clean, corrupt, added = (captures[k] for k in ("clean", "corrupt", "add_vector"))
    for layer in clean.layers:
        after_vector = layer >= ctx.spec["layer"] if ctx.spec["hook"] == "pre" else layer > ctx.spec["layer"]
        if not after_vector:
            torch.testing.assert_close(added.attention[layer], corrupt.attention[layer], rtol=1e-5, atol=1e-6,
                                       msg=f"Vector changed attention before its fixed intervention at layer {layer}")
        dc, da, dv = clean.dla(layer).mean(0), corrupt.dla(layer).mean(0), added.dla(layer).mean(0)
        ac, aa, av = (o.attention[layer].mean(0) for o in (clean, corrupt, added))
        if not all(torch.isfinite(t).all() for t in (dc, da, dv, ac, aa, av)):
            raise ValueError(f"Non-finite readout at layer {layer}")
        for head in range(len(dc)):
            dla_rows.append({"layer": layer, "head": head, "causally_after_vector": after_vector, "clean_dla": float(dc[head]), "corrupt_dla": float(da[head]),
                             "add_vector_dla": float(dv[head]), "delta_clean_corrupt": float(dc[head]-da[head]),
                             "delta_add_corrupt": float(dv[head]-da[head])})
            row = {"layer": layer, "head": head, "causally_after_vector": after_vector, "target_span": target,
                   "target_shift_clean_corrupt_pp": 100 * float(ac[head, span_index]-aa[head, span_index]),
                   "target_shift_add_corrupt_pp": 100 * float(av[head, span_index]-aa[head, span_index])}
            for i, name in enumerate(("R", "T", "F", "U", "history")):
                row.update({f"{name}_clean": float(ac[head, i]), f"{name}_corrupt": float(aa[head, i]), f"{name}_add_vector": float(av[head, i])})
            span_rows.append(row)
    key_layer, key_head = CONFIGS[ctx.key]["key_head"]
    key_dla = next(r for r in dla_rows if (r["layer"], r["head"]) == (key_layer, key_head))
    key_attention = next(r for r in span_rows if (r["layer"], r["head"]) == (key_layer, key_head))
    gap = scored["clean"]["mean_logit"] - scored["corrupt"]["mean_logit"]
    causal = None
    if ctx.key in ORIGINAL_LATE_MLP:
        late_layer = ORIGINAL_LATE_MLP[ctx.key]
        patched = late_mlp_patch(ctx, late_layer)
        causal = {"layer": late_layer, "intervention": "clean-to-corrupt last-token MLP output patch at the original fixed late MLP layer",
                  **patched, "logit_recovery": ratio(patched["mean_logit"]-scored["corrupt"]["mean_logit"], gap)}
    original_heads = ORIGINAL_KEY_HEADS.get(ctx.key, [(key_layer, key_head)])
    report = dict(ctx.metadata, readout_layers=clean.layers, target_span=target,
                  key_head={"layer": key_layer, "head": key_head}, key_head_dla=key_dla, key_head_attention=key_attention,
                  original_key_heads=[{"layer": layer, "head": head} for layer, head in original_heads],
                  original_key_heads_dla=[r for r in dla_rows if (r["layer"], r["head"]) in original_heads],
                  original_key_heads_attention=[r for r in span_rows if (r["layer"], r["head"]) in original_heads],
                  max_attention_shift_pp=max(r["target_shift_clean_corrupt_pp"] for r in span_rows),
                  max_attention_shift_with_vector_pp=max(r["target_shift_add_corrupt_pp"] for r in span_rows),
                  baseline=scored, fixed_downstream_mlp_patch=causal,
                  vector_causality="Heads before the fixed hook keep their native attention; changes in their RMS-scaled DLA can follow a change in final normalization",
                  attention_protocol="last query after native RoPE/QK normalization, observed alongside unchanged SDPA",
                  dla_protocol="head output through o_proj and marker unembedding, frozen final RMS scale; native model scaling retained")
    write_csv(ctx.out / "figure3a_attention_head_dla.csv", dla_rows)
    write_csv(ctx.out / "figure3b_attention_spans.csv", span_rows)
    write_json(ctx.out / "feature_readout.json", report)
    write_json(ROOT / "results/formation_readout" / ctx.key / "summary.json", report)
    (ctx.out / "summary.md").write_text(f"# {ctx.key}: fixed downstream readout\n\nVector: layer {ctx.spec['layer']} {ctx.spec['hook']}. Held-out: 200.\n\nExisting key head L{key_layer}H{key_head}; target {target}; clean-corrupt attention shift {key_attention['target_shift_clean_corrupt_pp']:.4f} pp.\n\nMarker DLA: {json.dumps(key_dla)}\n\nCausal late MLP patch: {json.dumps(causal)}\n")


def late_mlp_patch(ctx: Context, late_layer: int) -> dict[str, float]:
    """Keep Qwen3-4B's L35 and Qwen3-14B's L34 causal patch locations."""
    stored_clean = torch.empty(len(ctx.pairs.heldout), ctx.direction.numel())
    class MLPCapture:
        def begin(self, indices, positions, width):
            self.indices = indices
        def end(self):
            pass
    observer = MLPCapture()
    def save_mlp(m, a, k, o):
        stored_clean[observer.indices] = tensor_from_output(o)[:, -1].detach().float().cpu()
    handle = ctx.adapter.layers[late_layer].mlp.register_forward_hook(save_mlp, with_kwargs=True)
    try:
        ctx.run(ctx.seqs["heldout"]["clean"], observer)
    finally:
        handle.remove()
    class MLPPatch:
        def begin(self, indices, positions, width):
            self.indices = indices
        def end(self):
            pass
    patch_observer = MLPPatch()
    def patch_mlp(m, a, k, o):
        hidden = tensor_from_output(o).clone()
        hidden[:, -1] = stored_clean[patch_observer.indices].to(hidden.device, hidden.dtype)
        return replace_output(o, hidden)
    handle = ctx.adapter.layers[late_layer].mlp.register_forward_hook(patch_mlp, with_kwargs=True)
    try:
        patched = stats(ctx.run(ctx.seqs["heldout"]["corrupt"], patch_observer))
    finally:
        handle.remove()
    return patched


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-key", choices=[k for k in LOCKED if k != "qwen3_8b"], required=True)
    parser.add_argument("--stage", choices=["scaffold", "formation", "readout"], required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--feature-summary", type=Path,
                        help="Per-model feature accounting JSON for safetensors-based Transcoders")
    args = parser.parse_args()
    started = time.time()
    if args.stage == "formation" and args.model_key not in TC_FAMILIES and args.feature_summary is None:
        parser.error("--feature-summary is required for this model's formation stage")
    run_dir = args.run_dir.resolve()
    fixed_layers = run_dir / "fixed_layers.json"
    if not fixed_layers.exists():
        write_json(fixed_layers, {key: {"layer": spec["layer"], "hook": spec["hook"]}
                                  for key, spec in LOCKED.items()})
    ctx = Context(args.model_key, args.stage, run_dir, args.feature_summary)
    {"scaffold": scaffold, "formation": formation, "readout": readout}[args.stage](ctx)
    print(f"DONE {args.model_key} {args.stage} seconds={time.time()-started:.1f}", flush=True)


if __name__ == "__main__":
    main()
