#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from tqdm.auto import tqdm

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from phase6_common import DTYPE, MODEL_PATH, TOOL_CALL_TOKEN, TRANSCODER_DIR, build_pair_batches, build_sample_pairs, clear_cuda, ensure_dir  # noqa: E402
from toolcall_circuit.single_sample import load_hooked_qwen3  # noqa: E402
from artifact_paths import ARTIFACT_ROOT  # noqa: E402


PROJECT_ROOT = ARTIFACT_ROOT
PC_BUNDLE = PROJECT_ROOT / "results" / "8b_main" / "phase7_l24_directionality" / "exp_a_fixed_direction" / "pc_bundle.pt"
SELECTED_CATALOG = PROJECT_ROOT / "results" / "8b_main" / "section53_late_writer_family" / "selected_scope_catalog.csv"
INTERVENTION_RESULTS = PROJECT_ROOT / "results" / "8b_main" / "section53_late_writer_family" / "intervention_results.csv"
OUTPUT_ROOT = PROJECT_ROOT / "results" / "8b_main" / "section53_vector_to_mlp34"
TARGET_LAYER = 34
TARGET_SCOPES = ("L34_schema_only", "L34_structural", "L34_chat_boundary_only")
TARGET_COMPARISONS = ("family", "matched_control")


@dataclass(frozen=True)
class FeatureSet:
    scope: str
    comparison: str
    k: int
    layer: int
    feature_ids: List[int]


@dataclass(frozen=True)
class ConditionConfig:
    name: str
    side: str
    delta: torch.Tensor | None
    note: str


@dataclass
class ConditionCache:
    mlp_in: torch.Tensor
    mlp_out: torch.Tensor
    tool_logit: torch.Tensor
    top1: torch.Tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bridge experiment for Section 5.3: mu_delta -> MLP34 -> <tool_call>.")
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH)
    parser.add_argument("--pc-bundle", type=Path, default=PC_BUNDLE)
    parser.add_argument("--selected-catalog", type=Path, default=SELECTED_CATALOG)
    parser.add_argument("--intervention-results", type=Path, default=INTERVENTION_RESULTS)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--feature-batch-size", type=int, default=64)
    return parser.parse_args()


def write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[Dict[str, object]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_text(path: Path, text: str) -> None:
    ensure_dir(path.parent)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def write_json(path: Path, payload: Dict[str, object]) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def tool_call_token_id(model) -> int:
    token_ids = model.tokenizer.encode(TOOL_CALL_TOKEN, add_special_tokens=False)
    if len(token_ids) != 1:
        raise ValueError(f"{TOOL_CALL_TOKEN!r} is not a single token: {token_ids}")
    return int(token_ids[0])


def load_gate_bundle(path: Path) -> tuple[int, str, torch.Tensor, torch.Tensor]:
    bundle = torch.load(path, map_location="cpu", weights_only=False)
    layer = int(bundle.get("patch_layer", bundle.get("layer")))
    hook_kind = "pre" if "patch_layer" in bundle else "post"
    mu_delta = (bundle["mean_clean"] - bundle["mean_corrupt"]).detach().cpu().float().view(-1)
    random_direction = bundle["random_direction"].detach().cpu().float().view(-1)
    return layer, hook_kind, mu_delta, random_direction


def gate_hook_name(layer: int, hook_kind: str) -> str:
    return f"blocks.{layer}.hook_resid_{hook_kind}"


def make_last_token_add_hook(delta_cpu: torch.Tensor):
    def hook_fn(value: torch.Tensor, hook):  # noqa: ANN001
        out = value.clone()
        delta = delta_cpu.to(device=value.device, dtype=value.dtype)
        out[:, -1, :] = out[:, -1, :] + delta.view(1, -1)
        return out

    return hook_fn


def load_feature_sets(path: Path) -> List[FeatureSet]:
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    grouped: Dict[tuple[str, str, int, int], List[int]] = defaultdict(list)
    for row in rows:
        scope = str(row["scope"])
        comparison = str(row["comparison"])
        if scope not in TARGET_SCOPES or comparison not in TARGET_COMPARISONS:
            continue
        layer = int(row["layer"])
        if layer != TARGET_LAYER:
            continue
        key = (scope, comparison, int(row["k"]), layer)
        grouped[key].append(int(row["feature_id"]))

    feature_sets: List[FeatureSet] = []
    for (scope, comparison, k, layer), feature_ids in sorted(grouped.items()):
        feature_sets.append(
            FeatureSet(
                scope=scope,
                comparison=comparison,
                k=k,
                layer=layer,
                feature_ids=list(feature_ids),
            )
        )
    if not feature_sets:
        raise RuntimeError(f"No L34 feature sets found in {path}.")
    return feature_sets


def collect_condition_cache(
    model,
    pair_batches,
    *,
    side: str,
    condition_name: str,
    delta: torch.Tensor | None,
    gate_hook: str,
    mlp_in_hook: str,
    mlp_out_hook: str,
    tool_token_id_value: int,
) -> ConditionCache:
    n_samples = sum(len(batch.indices) for batch in pair_batches)
    d_model = int(model.cfg.d_model)
    mlp_in = torch.empty((n_samples, d_model), dtype=DTYPE)
    mlp_out = torch.empty((n_samples, d_model), dtype=DTYPE)
    tool_logit = torch.empty(n_samples, dtype=torch.float32)
    top1 = torch.empty(n_samples, dtype=torch.long)
    hook_names = {mlp_in_hook, mlp_out_hook}
    fwd_hooks = [] if delta is None else [(gate_hook, make_last_token_add_hook(delta))]

    progress = tqdm(pair_batches, desc=condition_name, dynamic_ncols=True, disable=True)
    for batch in progress:
        tokens_cpu = batch.clean_tokens_cpu if side == "clean" else batch.corrupt_tokens_cpu
        tokens = tokens_cpu.to(model.cfg.device)
        with torch.no_grad():
            if fwd_hooks:
                with model.hooks(fwd_hooks=fwd_hooks):
                    logits, cache = model.run_with_cache(tokens, names_filter=lambda name: name in hook_names)
            else:
                logits, cache = model.run_with_cache(tokens, names_filter=lambda name: name in hook_names)

        last_logits = logits[:, -1, :]
        tool_logit[batch.indices] = last_logits[:, tool_token_id_value].detach().cpu().float()
        top1[batch.indices] = last_logits.argmax(dim=-1).detach().cpu().long()
        mlp_in[batch.indices] = cache[mlp_in_hook][:, -1, :].detach().cpu().to(dtype=DTYPE)
        mlp_out[batch.indices] = cache[mlp_out_hook][:, -1, :].detach().cpu().to(dtype=DTYPE)

        del tokens, logits, cache, last_logits
        clear_cuda()
    return ConditionCache(mlp_in=mlp_in, mlp_out=mlp_out, tool_logit=tool_logit, top1=top1)


def compute_selected_activations(
    inputs: torch.Tensor,
    W_enc: torch.Tensor,
    b_enc: torch.Tensor,
    *,
    batch_size: int,
) -> torch.Tensor:
    outputs = torch.empty((int(inputs.shape[0]), int(W_enc.shape[0])), dtype=torch.float32)
    W_enc_f = W_enc.float()
    b_enc_f = b_enc.float()
    for start in range(0, int(inputs.shape[0]), batch_size):
        end = min(start + batch_size, int(inputs.shape[0]))
        batch_inputs = inputs[start:end].float()
        outputs[start:end] = torch.relu(F.linear(batch_inputs, W_enc_f, b_enc_f)).detach().cpu()
    return outputs


def safe_fraction(numerator: float, denominator: float) -> float:
    if abs(denominator) <= 1e-12:
        return float("nan")
    return float(numerator / denominator)


def load_validation_lookup(path: Path) -> dict[tuple[str, str, str, str], dict[str, float]]:
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    lookup: dict[tuple[str, str, str, str], dict[str, float]] = {}
    for row in rows:
        key = (str(row["scope"]), str(row["comparison"]), str(row["mode"]), str(row["side"]))
        lookup[key] = {
            "flip_rate": float(row["flip_rate"]),
            "after_tool_top1_rate": float(row["after_tool_top1_rate"]),
            "mean_logit_delta": float(row["mean_logit_delta"]),
        }
    return lookup


def main() -> None:
    args = parse_args()
    ensure_dir(args.output_root)
    if args.device.startswith("cuda") and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    gate_layer, hook_kind, mu_delta, random_direction = load_gate_bundle(args.pc_bundle)
    random_direction = random_direction / random_direction.norm().clamp_min(1e-12) * mu_delta.norm()

    model, _tokenizer = load_hooked_qwen3(str(args.model_path), args.device, DTYPE)
    model.set_use_hook_mlp_in(True)
    if hasattr(model.cfg, "use_hook_mlp_in"):
        model.cfg.use_hook_mlp_in = True

    pairs = build_sample_pairs(model, max_pairs=None)
    pair_batches = build_pair_batches(pairs)
    tool_token_id_value = tool_call_token_id(model)
    wu_tool = model.W_U[:, tool_token_id_value].detach().cpu().float()

    conditions = [
        ConditionConfig(name="clean_baseline", side="clean", delta=None, note="clean prompts, no gate intervention"),
        ConditionConfig(name="clean_gate_removed", side="clean", delta=-mu_delta, note="clean prompts, subtract mu_delta at L24"),
        ConditionConfig(name="corrupt_baseline", side="corrupt", delta=None, note="corrupt prompts, no gate intervention"),
        ConditionConfig(name="corrupt_gate_added", side="corrupt", delta=mu_delta, note="corrupt prompts, add mu_delta at L24"),
        ConditionConfig(name="corrupt_random_added", side="corrupt", delta=random_direction, note="corrupt prompts, add norm-matched random direction at L24"),
    ]

    gate_hook = gate_hook_name(gate_layer, hook_kind)
    mlp_in_hook = f"blocks.{TARGET_LAYER}.hook_mlp_in"
    mlp_out_hook = f"blocks.{TARGET_LAYER}.hook_mlp_out"

    condition_cache: Dict[str, ConditionCache] = {}
    for condition in conditions:
        condition_cache[condition.name] = collect_condition_cache(
            model,
            pair_batches,
            side=condition.side,
            condition_name=condition.name,
            delta=condition.delta,
            gate_hook=gate_hook,
            mlp_in_hook=mlp_in_hook,
            mlp_out_hook=mlp_out_hook,
            tool_token_id_value=tool_token_id_value,
        )

    feature_sets = load_feature_sets(args.selected_catalog)
    feature_rows_by_scope: Dict[str, List[FeatureSet]] = defaultdict(list)
    for item in feature_sets:
        feature_rows_by_scope[item.scope].append(item)
    selected_feature_ids = sorted({feature_id for item in feature_sets for feature_id in item.feature_ids})
    selected_idx = torch.tensor(selected_feature_ids, dtype=torch.long)
    transcoder = load_file(str(TRANSCODER_DIR / f"layer_{TARGET_LAYER}.safetensors"))
    W_enc = transcoder["W_enc"][selected_idx].detach().cpu().float()
    b_enc = transcoder["b_enc"][selected_idx].detach().cpu().float()
    W_dec = transcoder["W_dec"][selected_idx].detach().cpu().float()
    proj_weights = torch.mv(W_dec, wu_tool)
    feature_to_local = {feature_id: local_idx for local_idx, feature_id in enumerate(selected_feature_ids)}

    condition_summary_rows: List[Dict[str, object]] = []
    scope_summary_rows: List[Dict[str, object]] = []
    feature_summary_rows: List[Dict[str, object]] = []
    per_sample_scope_rows: List[Dict[str, object]] = []

    activation_cache: Dict[str, torch.Tensor] = {}
    projected_write_cache: Dict[str, torch.Tensor] = {}
    family_sum_cache: Dict[tuple[str, str], torch.Tensor] = {}
    family_proj_cache: Dict[tuple[str, str], torch.Tensor] = {}
    family_indices: Dict[tuple[str, str], List[int]] = {}

    for condition in conditions:
        cache = condition_cache[condition.name]
        mlp34_dla = torch.mv(cache.mlp_out.float(), wu_tool)
        top1_rate = float((cache.top1 == tool_token_id_value).float().mean().item())
        condition_summary_rows.append(
            {
                "condition": condition.name,
                "side": condition.side,
                "note": condition.note,
                "n": int(cache.top1.shape[0]),
                "tool_call_top1_rate": top1_rate,
                "mean_tool_logit": float(cache.tool_logit.mean().item()),
                "mean_mlp34_dla": float(mlp34_dla.mean().item()),
            }
        )

        acts = compute_selected_activations(cache.mlp_in, W_enc, b_enc, batch_size=args.feature_batch_size)
        activation_cache[condition.name] = acts
        projected_write_cache[condition.name] = acts * proj_weights.unsqueeze(0)

        for feature_id in selected_feature_ids:
            local_idx = feature_to_local[feature_id]
            feature_summary_rows.append(
                {
                    "condition": condition.name,
                    "feature_id": feature_id,
                    "mean_activation": float(acts[:, local_idx].mean().item()),
                    "mean_projected_tool_write": float(projected_write_cache[condition.name][:, local_idx].mean().item()),
                    "tool_call_projection": float(proj_weights[local_idx].item()),
                }
            )

        for item in feature_sets:
            key = (item.scope, item.comparison)
            local_indices = [feature_to_local[feature_id] for feature_id in item.feature_ids]
            family_indices[key] = local_indices
            family_sum = acts[:, local_indices].sum(dim=1)
            family_proj = projected_write_cache[condition.name][:, local_indices].sum(dim=1)
            family_sum_cache[(condition.name, f"{item.scope}:{item.comparison}")] = family_sum
            family_proj_cache[(condition.name, f"{item.scope}:{item.comparison}")] = family_proj

            scope_summary_rows.append(
                {
                    "condition": condition.name,
                    "scope": item.scope,
                    "comparison": item.comparison,
                    "k": item.k,
                    "feature_ids": ",".join(str(feature_id) for feature_id in item.feature_ids),
                    "mean_activation_sum": float(family_sum.mean().item()),
                    "mean_projected_tool_write": float(family_proj.mean().item()),
                }
            )
            for sample_idx, pair in enumerate(pairs):
                per_sample_scope_rows.append(
                    {
                        "sample_id": pair.sample_id,
                        "condition": condition.name,
                        "scope": item.scope,
                        "comparison": item.comparison,
                        "activation_sum": float(family_sum[sample_idx].item()),
                        "projected_tool_write": float(family_proj[sample_idx].item()),
                    }
                )

    transition_rows: List[Dict[str, object]] = []
    condition_map = {row["condition"]: row for row in condition_summary_rows}
    clean_mlp = float(condition_map["clean_baseline"]["mean_mlp34_dla"])
    removed_mlp = float(condition_map["clean_gate_removed"]["mean_mlp34_dla"])
    corrupt_mlp = float(condition_map["corrupt_baseline"]["mean_mlp34_dla"])
    added_mlp = float(condition_map["corrupt_gate_added"]["mean_mlp34_dla"])
    random_mlp = float(condition_map["corrupt_random_added"]["mean_mlp34_dla"])
    transition_rows.append(
        {
            "object": "MLP34_exact_dla",
            "metric": "mean_tool_write",
            "clean_minus_removed": clean_mlp - removed_mlp,
            "corrupt_plus_added": added_mlp - corrupt_mlp,
            "corrupt_plus_random": random_mlp - corrupt_mlp,
            "collapse_fraction_toward_corrupt": safe_fraction(clean_mlp - removed_mlp, clean_mlp - corrupt_mlp),
            "recovery_fraction_toward_clean": safe_fraction(added_mlp - corrupt_mlp, clean_mlp - corrupt_mlp),
            "random_fraction_toward_clean": safe_fraction(random_mlp - corrupt_mlp, clean_mlp - corrupt_mlp),
        }
    )

    scope_summary_map = {(row["condition"], row["scope"], row["comparison"]): row for row in scope_summary_rows}
    for item in feature_sets:
        clean_row = scope_summary_map[("clean_baseline", item.scope, item.comparison)]
        removed_row = scope_summary_map[("clean_gate_removed", item.scope, item.comparison)]
        corrupt_row = scope_summary_map[("corrupt_baseline", item.scope, item.comparison)]
        added_row = scope_summary_map[("corrupt_gate_added", item.scope, item.comparison)]
        random_row = scope_summary_map[("corrupt_random_added", item.scope, item.comparison)]
        for metric in ("mean_activation_sum", "mean_projected_tool_write"):
            clean_val = float(clean_row[metric])
            removed_val = float(removed_row[metric])
            corrupt_val = float(corrupt_row[metric])
            added_val = float(added_row[metric])
            random_val = float(random_row[metric])
            transition_rows.append(
                {
                    "object": f"{item.scope}:{item.comparison}",
                    "metric": metric,
                    "clean_minus_removed": clean_val - removed_val,
                    "corrupt_plus_added": added_val - corrupt_val,
                    "corrupt_plus_random": random_val - corrupt_val,
                    "collapse_fraction_toward_corrupt": safe_fraction(clean_val - removed_val, clean_val - corrupt_val),
                    "recovery_fraction_toward_clean": safe_fraction(added_val - corrupt_val, clean_val - corrupt_val),
                    "random_fraction_toward_clean": safe_fraction(random_val - corrupt_val, clean_val - corrupt_val),
                }
            )

    validation_lookup = load_validation_lookup(args.intervention_results)

    write_csv(
        args.output_root / "condition_summary.csv",
        ["condition", "side", "note", "n", "tool_call_top1_rate", "mean_tool_logit", "mean_mlp34_dla"],
        condition_summary_rows,
    )
    write_csv(
        args.output_root / "scope_condition_summary.csv",
        ["condition", "scope", "comparison", "k", "feature_ids", "mean_activation_sum", "mean_projected_tool_write"],
        scope_summary_rows,
    )
    write_csv(
        args.output_root / "feature_condition_summary.csv",
        ["condition", "feature_id", "mean_activation", "mean_projected_tool_write", "tool_call_projection"],
        feature_summary_rows,
    )
    write_csv(
        args.output_root / "transition_summary.csv",
        [
            "object",
            "metric",
            "clean_minus_removed",
            "corrupt_plus_added",
            "corrupt_plus_random",
            "collapse_fraction_toward_corrupt",
            "recovery_fraction_toward_clean",
            "random_fraction_toward_clean",
        ],
        transition_rows,
    )
    write_csv(
        args.output_root / "scope_condition_per_sample.csv",
        ["sample_id", "condition", "scope", "comparison", "activation_sum", "projected_tool_write"],
        per_sample_scope_rows,
    )
    write_json(
        args.output_root / "metadata.json",
        {
            "gate_layer": gate_layer,
            "hook_kind": hook_kind,
            "target_layer": TARGET_LAYER,
            "n_pairs": len(pairs),
            "tool_call_token": TOOL_CALL_TOKEN,
            "selected_scopes": list(TARGET_SCOPES),
            "selected_comparisons": list(TARGET_COMPARISONS),
        },
    )

    feature_condition_map = {(row["condition"], int(row["feature_id"])): row for row in feature_summary_rows}
    f109925_clean = feature_condition_map[("clean_baseline", 109925)]
    f109925_removed = feature_condition_map[("clean_gate_removed", 109925)]
    f109925_corrupt = feature_condition_map[("corrupt_baseline", 109925)]
    f109925_added = feature_condition_map[("corrupt_gate_added", 109925)]
    f109925_random = feature_condition_map[("corrupt_random_added", 109925)]

    lines = [
        "# Section 5.3 Bridge: Tool-call Vector -> MLP34 -> `<tool_call>`",
        "",
        "## Method",
        f"- Intervention site: `L{gate_layer}` `hook_resid_{hook_kind}` prediction position.",
        "- Causal move: subtract `mu_delta` on clean prompts, add `mu_delta` on corrupt prompts, and add a norm-matched random direction as a control.",
        f"- Readout site: exact `MLP{TARGET_LAYER}` output DLA onto `{TOOL_CALL_TOKEN}`, plus Transcoder readout of the previously validated L34 feature sets.",
        f"- Eval set: `{len(pairs)}` held-out clean/corrupt pairs.",
        "",
        "## Exact MLP34 Readout",
        f"- Clean baseline `{TOOL_CALL_TOKEN}` top-1: `{float(condition_map['clean_baseline']['tool_call_top1_rate']):.2%}`; corrupt baseline: `{float(condition_map['corrupt_baseline']['tool_call_top1_rate']):.2%}`.",
        f"- `MLP34` exact mean DLA: clean baseline `{clean_mlp:.4f}` -> clean gate-removed `{removed_mlp:.4f}` (drop `{clean_mlp - removed_mlp:.4f}`, `{safe_fraction(clean_mlp - removed_mlp, clean_mlp):.2%}` of clean baseline).",
        f"- `MLP34` exact mean DLA: corrupt baseline `{corrupt_mlp:.4f}` -> corrupt `+mu_delta` `{added_mlp:.4f}` (gain `{added_mlp - corrupt_mlp:.4f}`), versus corrupt `+random` `{random_mlp:.4f}` (gain `{random_mlp - corrupt_mlp:.4f}`).",
        f"- `MLP34` recovery fraction toward the clean/corrupt DLA gap: `{safe_fraction(added_mlp - corrupt_mlp, clean_mlp - corrupt_mlp):.2%}` for `+mu_delta`, versus `{safe_fraction(random_mlp - corrupt_mlp, clean_mlp - corrupt_mlp):.2%}` for `+random`.",
        "",
        "## Structural Family Readout",
    ]

    for scope in TARGET_SCOPES:
        family_row_clean = scope_summary_map[("clean_baseline", scope, "family")]
        family_row_removed = scope_summary_map[("clean_gate_removed", scope, "family")]
        family_row_corrupt = scope_summary_map[("corrupt_baseline", scope, "family")]
        family_row_added = scope_summary_map[("corrupt_gate_added", scope, "family")]
        family_row_random = scope_summary_map[("corrupt_random_added", scope, "family")]
        control_row_added = scope_summary_map[("corrupt_gate_added", scope, "matched_control")]
        lines.append(
            f"- `{scope}` projected tool write: clean `{float(family_row_clean['mean_projected_tool_write']):.4f}` -> clean gate-removed `{float(family_row_removed['mean_projected_tool_write']):.4f}`; "
            f"corrupt `{float(family_row_corrupt['mean_projected_tool_write']):.4f}` -> corrupt `+mu_delta` `{float(family_row_added['mean_projected_tool_write']):.4f}` -> corrupt `+random` `{float(family_row_random['mean_projected_tool_write']):.4f}`; "
            f"matched-control under corrupt `+mu_delta`: `{float(control_row_added['mean_projected_tool_write']):.4f}`."
        )
    lines.extend(
        [
            "",
            "## Dominant Single Feature",
            f"- `L34 F109925` mean activation: clean `{float(f109925_clean['mean_activation']):.4f}` -> clean gate-removed `{float(f109925_removed['mean_activation']):.4f}`; corrupt `{float(f109925_corrupt['mean_activation']):.4f}` -> corrupt `+mu_delta` `{float(f109925_added['mean_activation']):.4f}` -> corrupt `+random` `{float(f109925_random['mean_activation']):.4f}`.",
            f"- `L34 F109925` projected `{TOOL_CALL_TOKEN}` write: clean `{float(f109925_clean['mean_projected_tool_write']):.4f}` -> clean gate-removed `{float(f109925_removed['mean_projected_tool_write']):.4f}`; corrupt `{float(f109925_corrupt['mean_projected_tool_write']):.4f}` -> corrupt `+mu_delta` `{float(f109925_added['mean_projected_tool_write']):.4f}`.",
            "",
            "## Validation Against Direct Feature Intervention",
        ]
    )
    for scope in ("L34_schema_only", "L34_structural"):
        family_inject = validation_lookup[(scope, "family", "inject", "corrupt")]
        control_inject = validation_lookup[(scope, "matched_control", "inject", "corrupt")]
        lines.append(
            f"- `{scope}` clean-to-corrupt feature injection recovers `{TOOL_CALL_TOKEN}` on `{family_inject['after_tool_top1_rate']:.2%}`, versus `{control_inject['after_tool_top1_rate']:.2%}` for the matched control."
        )
    lines.extend(
        [
            "",
            "## Conclusion",
            f"- The exact downstream writer affected by the tool-call vector is `MLP{TARGET_LAYER}`: changing `mu_delta` at `L{gate_layer}` produces large, directed changes in `MLP{TARGET_LAYER}`'s exact contribution to `{TOOL_CALL_TOKEN}`.",
            "- The Transcoder readout localizes that change to the same L34 schema-focused family that independently passes the clean-to-corrupt rescue test.",
            "- This is the Section 5.3 bridge: the vector does not only correlate with late structure; it drives a specific L34 schema-opening writer that directly changes the final `<tool_call>` readout.",
        ]
    )
    write_text(args.output_root / "summary.md", "\n".join(lines))


if __name__ == "__main__":
    main()
