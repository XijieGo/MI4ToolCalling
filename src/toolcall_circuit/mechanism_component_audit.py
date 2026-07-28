#!/usr/bin/env python3
"""
Mechanism evidence audit for the current candidate tool-call chains.

This script does not assume the current story is correct. Instead it audits the
current candidate components on four axes:

1. Read evidence: what span/factor the component tracks
2. Write evidence: what endpoint score the component can directly move
3. Path evidence: whether key edges mediate the effect
4. Counterfactual evidence: which alternative explanations survive

Outputs:
- component_evidence_table.csv/json
- edge_evidence_table.csv/json
- claim_tiers.json
- writing_boundary.md
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

import numpy as np
import torch
from tqdm.auto import tqdm

if __package__ in {None, ""}:
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from toolcall_circuit.bidirectional_causal_eval import collect_cache_cpu_for_nodes, load_sample_paths
from toolcall_circuit.objective import build_bidirectional_endpoint_objectives
from toolcall_circuit.semantic_factorized_counterfactual import (
    build_variants,
    head_mass_summary,
    safe_float,
)
from toolcall_circuit.semantic_causal_chain import read_csv_rows
from toolcall_circuit.signed_edge_importance import run_logits_with_assignments
from toolcall_circuit.bidirectional_token_flip import run_logits_on_base_with_source
from toolcall_circuit.single_sample import load_hooked_qwen3, objective_from_logits


COMPONENTS = [
    "L2H14",
    "MLP11",
    "MLP16",
    "L21H12",
    "MLP27",
    "L16H4",
    "MLP17",
    "L23H6",
    "L24H6",
]

EDGE_CHAINS = {
    "query_chain": [
        ("L2H14", "MLP11"),
        ("MLP11", "MLP16"),
        ("MLP16", "L24H6"),
        ("L24H6", "Residual Output: decision"),
    ],
    "schema_chain": [
        ("L21H12", "MLP27"),
        ("MLP27", "Residual Output: decision"),
    ],
    "no_tool_chain": [
        ("L16H4", "MLP17"),
        ("MLP17", "L23H6"),
        ("L23H6", "Residual Output: decision"),
    ],
}

COMPONENT_CLAIMS = {
    "L2H14": "candidate query-need reader",
    "MLP11": "candidate early tool-biased writer",
    "MLP16": "candidate shared relay writer",
    "L21H12": "candidate schema/protocol reader",
    "MLP27": "candidate late decision writer",
    "L16H4": "candidate no-tool prior reader",
    "MLP17": "candidate no-tool writer",
    "L23H6": "candidate late suppressive relay",
    "L24H6": "candidate late relay / writer",
}

COMPONENT_ALTERNATIVES = {
    "L2H14": "general user-block salience rather than action-demand",
    "MLP11": "correlated early amplifier rather than tool-favoring writer",
    "MLP16": "shared late relay with weak semantic specificity",
    "L21H12": "format/protocol-tag reader rather than tool-availability reader",
    "MLP27": "strong late correlated node rather than unique late writer bottleneck",
    "L16H4": "anti-protocol / anti-special-token bias rather than ordinary-answer prior",
    "MLP17": "generic negative writer rather than no-tool-specific writer",
    "L23H6": "mixed late reader/writer rather than dedicated suppressive relay",
    "L24H6": "late correlated relay rather than functional writer",
}

EDGE_ALTERNATIVES = {
    "L2H14->MLP11": "coincidental query-branch adjacency rather than functional ingress edge",
    "MLP11->MLP16": "correlated middle edge rather than required relay",
    "MLP16->L24H6": "shared late correlation rather than relay edge",
    "L21H12->MLP27": "schema/protocol co-activation rather than causal main edge",
    "L16H4->MLP17": "shared no-tool correlation rather than ingress edge",
    "MLP17->L23H6": "shared suppressive correlation rather than relay edge",
}


def write_csv(rows: Sequence[Dict[str, object]], out_path: Path) -> None:
    if not rows:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def median(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if isinstance(v, (int, float)) and math.isfinite(float(v))]
    return float(np.median(vals)) if vals else float("nan")


def mean(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if isinstance(v, (int, float)) and math.isfinite(float(v))]
    return float(np.mean(vals)) if vals else float("nan")


def safe_rate(values: Iterable[bool]) -> float:
    vals = [1.0 if bool(v) else 0.0 for v in values]
    return float(np.mean(vals)) if vals else float("nan")


def load_node_meta(run_root: Path) -> Dict[str, Dict[str, str]]:
    rows = read_csv_rows(run_root / "functional_groups" / "functional_node_table.csv")
    return {str(r["node"]): r for r in rows}


def load_component_paths(run_root: Path) -> Dict[str, List[str]]:
    chain = json.loads((run_root / "semantic_chain" / "semantic_chain_summary.json").read_text(encoding="utf-8"))
    return {
        str(path["key"]): [str(x) for x in path["nodes"] if str(x) != "Residual Output: decision"]
        for path in chain.get("paths", [])
    }


def choose_source_variant(component: str) -> str:
    if component in {"L16H4", "MLP17", "L23H6"}:
        return "corrupt_full"
    return "clean_full"


def choose_base_variants(component: str) -> List[str]:
    if component in {"L21H12", "MLP27"}:
        return ["corrupt_full", "clean_no_schema", "clean_schema_mismatch", "clean_no_protocol"]
    if component in {"L16H4", "MLP17", "L23H6"}:
        return ["clean_full", "clean_no_schema", "clean_schema_mismatch", "clean_no_protocol"]
    return ["corrupt_full", "clean_no_schema", "clean_schema_mismatch", "clean_no_protocol"]


def endpoint_direction(component: str) -> str:
    if component in {"L16H4", "MLP17", "L23H6"}:
        return "no_tool"
    return "tool"


def variant_family_signature(component: str, rows: Sequence[Dict[str, object]]) -> str:
    best_by_base = {str(r["base_variant"]): safe_float(r["rescue_ratio_median"], float("nan")) for r in rows}
    if component in {"L21H12", "MLP27"}:
        no_schema = best_by_base.get("clean_no_schema", float("nan"))
        mismatch = best_by_base.get("clean_schema_mismatch", float("nan"))
        no_protocol = best_by_base.get("clean_no_protocol", float("nan"))
        if no_protocol > no_schema + 0.8 and no_protocol > mismatch + 0.8:
            return "protocol-heavy"
        if mismatch > no_schema + 0.2:
            return "schema-match-sensitive"
        return "tool-availability-sensitive"
    if component in {"L16H4", "MLP17", "L23H6"}:
        clean_full = best_by_base.get("clean_full", float("nan"))
        no_protocol = best_by_base.get("clean_no_protocol", float("nan"))
        if no_protocol > clean_full + 0.2:
            return "anti-protocol-leaning"
        return "ordinary-answer / no-tool leaning"
    corrupt = best_by_base.get("corrupt_full", float("nan"))
    no_schema = best_by_base.get("clean_no_schema", float("nan"))
    if corrupt > no_schema + 0.2:
        return "query-conditioned"
    return "mixed tool-conditioned"


def classify_tier(component_row: Dict[str, object]) -> str:
    path = safe_float(component_row["path_mediation_strength"], 0.0)
    write = safe_float(component_row["direct_write_strength"], 0.0)
    top1 = safe_float(component_row["strongest_top1_rate"], 0.0)
    excluded = bool(component_row["primary_alternative_excluded"])
    if path >= 0.15 and write >= 0.35 and top1 >= 0.45 and excluded:
        return "A"
    if path >= 0.05 and write >= 0.15:
        return "B"
    return "C"


def edge_tier(edge_row: Dict[str, object]) -> str:
    mediated = safe_float(edge_row["best_mediated_ratio"], 0.0)
    if mediated >= 0.15 and bool(edge_row["alternative_excluded"]):
        return "A"
    if mediated >= 0.05:
        return "B"
    return "C"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a mechanism evidence audit on the stable tool-call dataset.")
    parser.add_argument("--run-root", type=str, required=True)
    parser.add_argument("--model-path", type=str, default="./external/models/Qwen3-1.7B")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--output-root", type=str, required=True)
    args = parser.parse_args()

    run_root = Path(args.run_root).resolve()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    node_meta = load_node_meta(run_root)
    paths = load_component_paths(run_root)
    all_nodes = sorted({n for n in COMPONENTS})
    all_heads = [n for n in COMPONENTS if n.startswith("L")]

    samples = load_sample_paths(run_root / "forward_batch", run_root / "reverse_batch", max_samples=args.max_samples)
    model, tokenizer = load_hooked_qwen3(args.model_path, device=args.device, dtype=torch.bfloat16)

    component_patch_rows: List[Dict[str, object]] = []
    edge_patch_rows: List[Dict[str, object]] = []
    head_read_rows: List[Dict[str, object]] = []

    pbar = tqdm(samples, desc="Mechanism component audit", dynamic_ncols=True)
    for sp in pbar:
        try:
            clean_text = sp.tool_call_prompt.read_text(encoding="utf-8")
            corrupt_text = sp.no_tool_prompt.read_text(encoding="utf-8")
        except Exception:
            continue
        variants = build_variants(clean_text, corrupt_text)

        logits_by_variant: Dict[str, torch.Tensor] = {}
        tokens_by_variant: Dict[str, torch.Tensor] = {}
        for name, text in variants.items():
            tokens = model.to_tokens(text, prepend_bos=False)
            tokens_by_variant[name] = tokens
            with torch.no_grad():
                logits_by_variant[name] = model(tokens)

        tool_objective, no_tool_objective = build_bidirectional_endpoint_objectives(
            logits_by_variant["clean_full"],
            logits_by_variant["corrupt_full"],
            tokenizer=tokenizer,
        )
        tool_gap = float(objective_from_logits(logits_by_variant["clean_full"], tool_objective).item()) - float(
            objective_from_logits(logits_by_variant["corrupt_full"], tool_objective).item()
        )
        no_tool_gap = float(objective_from_logits(logits_by_variant["corrupt_full"], no_tool_objective).item()) - float(
            objective_from_logits(logits_by_variant["clean_full"], no_tool_objective).item()
        )
        if (
            not math.isfinite(tool_gap)
            or abs(tool_gap) < 1e-8
            or not math.isfinite(no_tool_gap)
            or abs(no_tool_gap) < 1e-8
        ):
            continue

        clean_cache = collect_cache_cpu_for_nodes(model, tokens_by_variant["clean_full"], all_nodes)
        corrupt_cache = collect_cache_cpu_for_nodes(model, tokens_by_variant["corrupt_full"], all_nodes)
        variant_caches = {
            "clean_full": clean_cache,
            "corrupt_full": corrupt_cache,
            "clean_no_schema": collect_cache_cpu_for_nodes(model, tokens_by_variant["clean_no_schema"], all_nodes),
            "clean_schema_mismatch": collect_cache_cpu_for_nodes(model, tokens_by_variant["clean_schema_mismatch"], all_nodes),
            "clean_no_protocol": collect_cache_cpu_for_nodes(model, tokens_by_variant["clean_no_protocol"], all_nodes),
        }

        # Head read evidence across factorized variants.
        for variant_name, text in variants.items():
            masses = head_mass_summary(model, tokenizer, text, all_heads)
            for head in all_heads:
                for set_name in ["user_block", "tools_block", "tools_payload", "tool_call_tags", "protocol_payload", "prefix_16"]:
                    head_read_rows.append(
                        {
                            "sample_id": sp.sample_id,
                            "component": head,
                            "variant": variant_name,
                            "set": set_name,
                            "mass": masses.get((head, set_name), float("nan")),
                        }
                    )

        # Direct write audits for each component on relevant base variants.
        for component in COMPONENTS:
            direction = endpoint_direction(component)
            source_variant = choose_source_variant(component)
            source_cache = clean_cache if source_variant == "clean_full" else corrupt_cache
            for base_variant in choose_base_variants(component):
                base_tokens = tokens_by_variant[base_variant]
                base_tool = float(objective_from_logits(logits_by_variant[base_variant], tool_objective).item())
                base_no_tool = float(objective_from_logits(logits_by_variant[base_variant], no_tool_objective).item())
                patched_logits = run_logits_on_base_with_source(model, base_tokens, source_cache, [component])
                patched_tool = float(objective_from_logits(patched_logits, tool_objective).item())
                patched_no_tool = float(objective_from_logits(patched_logits, no_tool_objective).item())
                if direction == "tool":
                    ratio = (patched_tool - base_tool) / tool_gap
                    top1 = int(patched_logits[0, -1].argmax().item()) == sp.target_tool_call
                    boundary = patched_tool > patched_no_tool
                else:
                    ratio = (patched_no_tool - base_no_tool) / no_tool_gap
                    top1 = int(patched_logits[0, -1].argmax().item()) == sp.distractor
                    boundary = patched_no_tool > patched_tool
                component_patch_rows.append(
                    {
                        "sample_id": sp.sample_id,
                        "component": component,
                        "direction": direction,
                        "source_variant": source_variant,
                        "base_variant": base_variant,
                        "rescue_ratio": ratio,
                        "tool_score_delta": patched_tool - base_tool,
                        "no_tool_score_delta": patched_no_tool - base_no_tool,
                        "top1_success": top1,
                        "boundary_flip": boundary,
                    }
                )

        # Key edge blocked mediation on semantically relevant base variants.
        edge_specs = [
            ("query_chain", "corrupt_full", "clean_full", "tool"),
            ("schema_chain", "clean_no_schema", "clean_full", "tool"),
            ("schema_chain", "clean_schema_mismatch", "clean_full", "tool"),
            ("schema_chain", "clean_no_protocol", "clean_full", "tool"),
            ("no_tool_chain", "clean_full", "corrupt_full", "no_tool"),
        ]
        for chain_key, base_variant, source_variant, direction in edge_specs:
            source_cache = variant_caches[source_variant]
            base_cache = variant_caches[base_variant]
            base_tokens = tokens_by_variant[base_variant]
            base_tool = float(objective_from_logits(logits_by_variant[base_variant], tool_objective).item())
            base_no_tool = float(objective_from_logits(logits_by_variant[base_variant], no_tool_objective).item())
            gap = tool_gap if direction == "tool" else no_tool_gap
            objective = tool_objective if direction == "tool" else no_tool_objective
            for source, target in EDGE_CHAINS[chain_key]:
                if target == "Residual Output: decision":
                    continue
                source_only_logits = run_logits_with_assignments(
                    model,
                    base_tokens,
                    source_cache,
                    base_cache,
                    [source],
                    [],
                )
                blocked_logits = run_logits_with_assignments(
                    model,
                    base_tokens,
                    source_cache,
                    base_cache,
                    [source],
                    [target],
                )
                if direction == "tool":
                    source_ratio = (float(objective_from_logits(source_only_logits, objective).item()) - base_tool) / gap
                    blocked_ratio = (float(objective_from_logits(blocked_logits, objective).item()) - base_tool) / gap
                else:
                    source_ratio = (float(objective_from_logits(source_only_logits, objective).item()) - base_no_tool) / gap
                    blocked_ratio = (float(objective_from_logits(blocked_logits, objective).item()) - base_no_tool) / gap
                edge_patch_rows.append(
                    {
                        "sample_id": sp.sample_id,
                        "chain_key": chain_key,
                        "edge": f"{source}->{target}",
                        "base_variant": base_variant,
                        "direction": direction,
                        "source_ratio": source_ratio,
                        "blocked_ratio": blocked_ratio,
                        "mediated_ratio": source_ratio - blocked_ratio,
                    }
                )
        pbar.set_postfix(sample=sp.sample_id)

    write_csv(component_patch_rows, output_root / "component_patch_per_sample.csv")
    write_csv(edge_patch_rows, output_root / "edge_patch_per_sample.csv")
    write_csv(head_read_rows, output_root / "component_head_reads_per_sample.csv")

    # Summaries
    component_rows: List[Dict[str, object]] = []
    for component in COMPONENTS:
        meta = node_meta.get(component, {})
        component_patch = [r for r in component_patch_rows if str(r["component"]) == component]
        by_base = defaultdict(list)
        for row in component_patch:
            by_base[str(row["base_variant"])].append(row)
        rescue_summary = {base: median(safe_float(r["rescue_ratio"]) for r in rows) for base, rows in by_base.items()}
        top1_summary = {base: safe_rate(bool(r["top1_success"]) for r in rows) for base, rows in by_base.items()}
        head_rows = [r for r in head_read_rows if str(r["component"]) == component]
        read_by_variant_set = defaultdict(list)
        for row in head_rows:
            read_by_variant_set[(str(row["variant"]), str(row["set"]))].append(safe_float(row["mass"]))
        component_read_summary = {
            f"{variant}::{set_name}": median(vals)
            for (variant, set_name), vals in read_by_variant_set.items()
        }

        edge_rows = [r for r in edge_patch_rows if component in str(r["edge"])]
        best_mediated = max((safe_float(r["mediated_ratio"], 0.0) for r in edge_rows), default=0.0)
        strongest_base = max(rescue_summary.items(), key=lambda kv: safe_float(kv[1], float("-inf")))[0] if rescue_summary else ""
        strongest_ratio = rescue_summary.get(strongest_base, float("nan"))
        strongest_top1 = top1_summary.get(strongest_base, float("nan"))
        signature = variant_family_signature(component, [
            {"base_variant": base, "rescue_ratio_median": value}
            for base, value in rescue_summary.items()
        ])

        read_object = "undetermined"
        if component.startswith("L"):
            clean_user = component_read_summary.get("clean_full::user_block", float("nan"))
            clean_tools = component_read_summary.get("clean_full::tools_payload", float("nan"))
            clean_protocol = component_read_summary.get("clean_full::protocol_payload", float("nan"))
            if clean_user > max(clean_tools, clean_protocol) + 0.10:
                read_object = "user-side content / action-demand-like cue"
            elif clean_tools > max(clean_user, clean_protocol) + 0.02:
                read_object = "tool schema payload"
            elif clean_protocol > max(clean_user, clean_tools) + 0.02:
                read_object = "protocol / tool-call payload"
            else:
                read_object = "mixed cue"
        write_object = "undetermined"
        if component in {"MLP27", "MLP11", "MLP16", "L24H6", "L21H12"}:
            write_object = "<tool_call>-favoring residual direction"
        elif component in {"L16H4", "MLP17", "L23H6"}:
            write_object = "no_tool-favoring residual direction"

        alternative_excluded = False
        if component == "L21H12":
            alternative_excluded = signature in {"schema-match-sensitive", "tool-availability-sensitive"}
        elif component == "L16H4":
            alternative_excluded = signature == "ordinary-answer / no-tool leaning"
        elif component == "MLP27":
            alternative_excluded = safe_float(strongest_ratio, 0.0) >= 0.5 and best_mediated >= 0.15
        elif component == "L2H14":
            alternative_excluded = signature == "query-conditioned"
        else:
            alternative_excluded = safe_float(strongest_ratio, 0.0) >= 0.2 and best_mediated >= 0.05

        component_row = {
            "component": component,
            "current_claim": COMPONENT_CLAIMS.get(component, ""),
            "functional_group": str(meta.get("functional_group", "")),
            "structural_group": str(meta.get("structural_group", "")),
            "object_language_function": "",
            "read_object": read_object,
            "write_object": write_object,
            "strongest_support_context": strongest_base,
            "strongest_rescue_ratio": strongest_ratio,
            "strongest_top1_rate": strongest_top1,
            "direct_write_strength": median(safe_float(r["rescue_ratio"]) for r in component_patch if str(r["base_variant"]) == strongest_base),
            "path_mediation_strength": best_mediated,
            "counterfactual_signature": signature,
            "strongest_supporting_evidence": "",
            "direct_read_write_evidence": "",
            "path_evidence": "",
            "counterfactual_evidence": "",
            "excluded_alternative": COMPONENT_ALTERNATIVES.get(component, ""),
            "primary_alternative_excluded": alternative_excluded,
            "remaining_risk": "",
            "sufficient_for_main_text": "",
            "tier": "",
        }

        if component == "L21H12":
            component_row["object_language_function"] = "reads schema/protocol availability cues and sends them to the late tool writer"
            component_row["strongest_supporting_evidence"] = (
                f"schema step-1 rescue on no-schema/mismatch/no-protocol variants; best rescue {safe_float(strongest_ratio):.3f}"
            )
            component_row["direct_read_write_evidence"] = (
                f"tools_payload={component_read_summary.get('clean_full::tools_payload', float('nan')):.3f}, "
                f"protocol_payload={component_read_summary.get('clean_no_protocol::protocol_payload', float('nan')):.3f}, "
                f"best direct rescue={safe_float(strongest_ratio):.3f}"
            )
            component_row["path_evidence"] = f"L21H12->MLP27 mediated ratio up to {best_mediated:.3f}"
            component_row["counterfactual_evidence"] = signature
            component_row["remaining_risk"] = "schema content vs protocol cue still partially entangled"
        elif component == "MLP27":
            component_row["object_language_function"] = "late writer that converts schema-conditioned state into a tool-call-favoring output direction"
            component_row["strongest_supporting_evidence"] = f"schema path final rescue {safe_float(strongest_ratio):.3f}"
            component_row["direct_read_write_evidence"] = f"best direct rescue={safe_float(strongest_ratio):.3f}, top1={safe_float(strongest_top1):.3f}"
            component_row["path_evidence"] = f"receives mediated input from L21H12 with ratio up to {best_mediated:.3f}"
            component_row["counterfactual_evidence"] = signature
            component_row["remaining_risk"] = "not yet proven unique bottleneck; backup writer search still missing"
        elif component == "L16H4":
            component_row["object_language_function"] = "reads no-tool / ordinary-answer evidence from the user-side prompt and feeds the no-tool chain"
            component_row["strongest_supporting_evidence"] = f"best no-tool rescue {safe_float(strongest_ratio):.3f}"
            component_row["direct_read_write_evidence"] = (
                f"user_block mass on clean_full={component_read_summary.get('clean_full::user_block', float('nan')):.3f}, "
                f"on corrupt_full={component_read_summary.get('corrupt_full::user_block', float('nan')):.3f}"
            )
            component_row["path_evidence"] = f"L16H4->MLP17 mediated ratio up to {best_mediated:.3f}"
            component_row["counterfactual_evidence"] = signature
            component_row["remaining_risk"] = "ordinary-answer prior vs anti-protocol bias not fully separated"
        elif component == "MLP17":
            component_row["object_language_function"] = "writes a no-tool-favoring residual state inside the suppression chain"
            component_row["strongest_supporting_evidence"] = f"best no-tool rescue {safe_float(strongest_ratio):.3f}"
            component_row["direct_read_write_evidence"] = f"best direct rescue={safe_float(strongest_ratio):.3f}, top1={safe_float(strongest_top1):.3f}"
            component_row["path_evidence"] = f"MLP17->L23H6 mediated ratio up to {best_mediated:.3f}"
            component_row["counterfactual_evidence"] = signature
            component_row["remaining_risk"] = "could still be a strong generic negative writer"
        elif component == "L23H6":
            component_row["object_language_function"] = "late suppressive relay that carries no-tool-biased state toward the output"
            component_row["strongest_supporting_evidence"] = f"best no-tool rescue {safe_float(strongest_ratio):.3f}"
            component_row["direct_read_write_evidence"] = f"best direct rescue={safe_float(strongest_ratio):.3f}"
            component_row["path_evidence"] = f"receives MLP17 signal with mediated ratio up to {best_mediated:.3f}"
            component_row["counterfactual_evidence"] = signature
            component_row["remaining_risk"] = "may still mix suppressive relay with late reader behavior"
        elif component == "L2H14":
            component_row["object_language_function"] = "candidate reader of user-side action-demand / execution intent"
            component_row["strongest_supporting_evidence"] = f"best tool rescue on corrupt_full={rescue_summary.get('corrupt_full', float('nan')):.3f}"
            component_row["direct_read_write_evidence"] = (
                f"user_block mass={component_read_summary.get('clean_full::user_block', float('nan')):.3f}, "
                f"tools_payload={component_read_summary.get('clean_full::tools_payload', float('nan')):.3f}"
            )
            component_row["path_evidence"] = f"L2H14->MLP11 mediated ratio up to {best_mediated:.3f}"
            component_row["counterfactual_evidence"] = signature
            component_row["remaining_risk"] = "current stable data does not isolate action-demand cleanly"
        elif component == "MLP11":
            component_row["object_language_function"] = "candidate early tool-favoring writer downstream of the query-side reader"
            component_row["strongest_supporting_evidence"] = f"best tool rescue={safe_float(strongest_ratio):.3f}"
            component_row["direct_read_write_evidence"] = f"best direct rescue={safe_float(strongest_ratio):.3f}"
            component_row["path_evidence"] = f"MLP11->MLP16 mediated ratio up to {best_mediated:.3f}"
            component_row["counterfactual_evidence"] = signature
            component_row["remaining_risk"] = "not yet isolated from broader early tool branch"
        elif component == "MLP16":
            component_row["object_language_function"] = "shared late relay/writer that transports upstream tool-biased state toward the output-adjacent region"
            component_row["strongest_supporting_evidence"] = f"best tool rescue={safe_float(strongest_ratio):.3f}"
            component_row["direct_read_write_evidence"] = f"best direct rescue={safe_float(strongest_ratio):.3f}"
            component_row["path_evidence"] = f"incident mediated ratio up to {best_mediated:.3f}"
            component_row["counterfactual_evidence"] = signature
            component_row["remaining_risk"] = "semantic specificity remains weak; likely relay rather than reader"
        elif component == "L24H6":
            component_row["object_language_function"] = "late relay/writer that helps carry tool-biased state into the final output region"
            component_row["strongest_supporting_evidence"] = f"best tool rescue={safe_float(strongest_ratio):.3f}"
            component_row["direct_read_write_evidence"] = f"best direct rescue={safe_float(strongest_ratio):.3f}"
            component_row["path_evidence"] = f"incident mediated ratio up to {best_mediated:.3f}"
            component_row["counterfactual_evidence"] = signature
            component_row["remaining_risk"] = "writer vs relay role still not fully separated"

        component_row["tier"] = classify_tier(component_row)
        component_row["sufficient_for_main_text"] = "yes" if component_row["tier"] == "A" else ("cautious" if component_row["tier"] == "B" else "no")
        component_rows.append(component_row)

    edge_rows_summary: List[Dict[str, object]] = []
    for chain_key, chain_edges in EDGE_CHAINS.items():
        for source, target in chain_edges:
            if target == "Residual Output: decision":
                continue
            edge_name = f"{source}->{target}"
            rows = [r for r in edge_patch_rows if str(r["edge"]) == edge_name]
            best_mediated = max((safe_float(r["mediated_ratio"], 0.0) for r in rows), default=0.0)
            best_base = ""
            if rows:
                by_base = defaultdict(list)
                for row in rows:
                    by_base[str(row["base_variant"])].append(row)
                best_base = max(by_base.items(), key=lambda kv: median(safe_float(r["mediated_ratio"]) for r in kv[1]))[0]
            alternative_excluded = best_mediated >= 0.10
            edge_row = {
                "edge": edge_name,
                "chain_key": chain_key,
                "current_claim": "candidate causal edge in the current mechanism chain",
                "object_language_function": "",
                "best_base_variant": best_base,
                "best_mediated_ratio": best_mediated,
                "strongest_supporting_evidence": "",
                "path_evidence": "",
                "counterfactual_evidence": best_base,
                "excluded_alternative": EDGE_ALTERNATIVES.get(edge_name, ""),
                "alternative_excluded": alternative_excluded,
                "remaining_risk": "",
                "tier": "",
                "sufficient_for_main_text": "",
            }
            if edge_name == "L21H12->MLP27":
                edge_row["object_language_function"] = "carries schema/protocol-conditioned signal into the final late writer"
                edge_row["strongest_supporting_evidence"] = f"best mediated ratio {best_mediated:.3f} on {best_base}"
                edge_row["path_evidence"] = f"blocked target sharply reduces source rescue on {best_base}"
                edge_row["remaining_risk"] = "still need q/k/v-specific route decomposition"
            elif edge_name == "L16H4->MLP17":
                edge_row["object_language_function"] = "passes no-tool-biased user-side evidence into the no-tool writer"
                edge_row["strongest_supporting_evidence"] = f"best mediated ratio {best_mediated:.3f} on {best_base}"
                edge_row["path_evidence"] = f"source rescue loses strength when MLP17 is blocked on {best_base}"
                edge_row["remaining_risk"] = "ordinary-answer prior vs anti-protocol signal still entangled"
            elif edge_name == "MLP17->L23H6":
                edge_row["object_language_function"] = "passes no-tool-biased written state into the late suppressive relay"
                edge_row["strongest_supporting_evidence"] = f"best mediated ratio {best_mediated:.3f} on {best_base}"
                edge_row["path_evidence"] = f"blocking L23H6 removes part of MLP17 rescue on {best_base}"
                edge_row["remaining_risk"] = "late suppressive chain may still have parallel backup routes"
            elif edge_name == "L2H14->MLP11":
                edge_row["object_language_function"] = "candidate ingress edge from query-side reader into the early tool writer"
                edge_row["strongest_supporting_evidence"] = f"best mediated ratio {best_mediated:.3f} on {best_base}"
                edge_row["path_evidence"] = f"partial mediation on {best_base}"
                edge_row["remaining_risk"] = "action-demand semantics not yet isolated"
            elif edge_name == "MLP11->MLP16":
                edge_row["object_language_function"] = "candidate relay edge from early tool write into the shared late relay"
                edge_row["strongest_supporting_evidence"] = f"best mediated ratio {best_mediated:.3f} on {best_base}"
                edge_row["path_evidence"] = f"partial mediation on {best_base}"
                edge_row["remaining_risk"] = "could still be correlated branch structure"
            elif edge_name == "MLP16->L24H6":
                edge_row["object_language_function"] = "candidate late relay edge inside the query-conditioned branch"
                edge_row["strongest_supporting_evidence"] = f"best mediated ratio {best_mediated:.3f} on {best_base}"
                edge_row["path_evidence"] = f"partial mediation on {best_base}"
                edge_row["remaining_risk"] = "writer vs relay distinction unresolved"

            edge_row["tier"] = edge_tier(edge_row)
            edge_row["sufficient_for_main_text"] = "yes" if edge_row["tier"] == "A" else ("cautious" if edge_row["tier"] == "B" else "no")
            edge_rows_summary.append(edge_row)

    # Claim tiers and writing boundary
    level_a = [row["component"] for row in component_rows if row["tier"] == "A"] + [
        row["edge"] for row in edge_rows_summary if row["tier"] == "A"
    ]
    level_b = [row["component"] for row in component_rows if row["tier"] == "B"] + [
        row["edge"] for row in edge_rows_summary if row["tier"] == "B"
    ]
    level_c = [row["component"] for row in component_rows if row["tier"] == "C"] + [
        row["edge"] for row in edge_rows_summary if row["tier"] == "C"
    ]
    claim_tiers = {
        "level_A": level_a,
        "level_B": level_b,
        "level_C": level_c,
    }

    boundary_lines = [
        "# Writing Boundary",
        "",
        "## Strong",
    ]
    for row in component_rows:
        if row["tier"] == "A":
            boundary_lines.append(f"- `{row['component']}`: {row['object_language_function']}")
    for row in edge_rows_summary:
        if row["tier"] == "A":
            boundary_lines.append(f"- `{row['edge']}`: {row['object_language_function']}")
    boundary_lines.append("")
    boundary_lines.append("## Weak")
    for row in component_rows:
        if row["tier"] == "B":
            boundary_lines.append(f"- `{row['component']}`: evidence suggests {row['object_language_function']}")
    for row in edge_rows_summary:
        if row["tier"] == "B":
            boundary_lines.append(f"- `{row['edge']}`: evidence suggests {row['object_language_function']}")
    boundary_lines.append("")
    boundary_lines.append("## Do Not Write Strongly")
    boundary_lines.append("- unified mode switcher")
    boundary_lines.append("- arbitration zone")
    boundary_lines.append("- decision boundary module")
    boundary_lines.append("- default conservative branch")
    for row in component_rows:
        if row["tier"] == "C":
            boundary_lines.append(f"- `{row['component']}` strong functional claim")
    for row in edge_rows_summary:
        if row["tier"] == "C":
            boundary_lines.append(f"- `{row['edge']}` strong causal-edge claim")

    write_csv(component_rows, output_root / "component_evidence_table.csv")
    write_csv(edge_rows_summary, output_root / "edge_evidence_table.csv")
    (output_root / "component_evidence_table.json").write_text(json.dumps(component_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_root / "edge_evidence_table.json").write_text(json.dumps(edge_rows_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_root / "claim_tiers.json").write_text(json.dumps(claim_tiers, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_root / "writing_boundary.md").write_text("\n".join(boundary_lines) + "\n", encoding="utf-8")

    summary = {
        "component_rows": component_rows,
        "edge_rows": edge_rows_summary,
        "claim_tiers": claim_tiers,
        "artifacts": {
            "component_csv": str(output_root / "component_evidence_table.csv"),
            "edge_csv": str(output_root / "edge_evidence_table.csv"),
            "tiers_json": str(output_root / "claim_tiers.json"),
            "boundary_md": str(output_root / "writing_boundary.md"),
        },
    }
    (output_root / "mechanism_audit_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
