#!/usr/bin/env python3
"""
Assemble a concise Markdown report for the final tool-call circuit deliverable.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

if __package__ in {None, ""}:
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


def read_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def fmt(value: object, digits: int = 3) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return str(value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the final Markdown report for the tool-call circuit run.")
    parser.add_argument("--run-root", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    run_root = Path(args.run_root).resolve()
    out_path = Path(args.output).resolve()

    bidi = read_json(run_root / "bidirectional" / "bidirectional_summary.json")
    signed = read_json(run_root / "final_signed_circuit" / "final_signed_circuit_summary.json")
    signed_validate = read_json(run_root / "signed_validate" / "signed_group_report.json")
    token_flip = read_json(run_root / "token_flip" / "group_token_flip_summary.json")
    node_importance = read_json(run_root / "node_importance" / "signed_node_importance_report.json")
    functional = read_json(run_root / "functional_groups" / "functional_group_summary.json")
    functional_validate_path = run_root / "functional_validate" / "functional_group_report.json"
    functional_validate = read_json(functional_validate_path) if functional_validate_path.exists() else {"summary_rows": []}
    semantic_chain_path = run_root / "semantic_chain" / "semantic_chain_summary.json"
    semantic_chain = read_json(semantic_chain_path) if semantic_chain_path.exists() else {"paths": [], "summary_rows": []}
    semantic_factorized_path = run_root / "semantic_factorized" / "semantic_factorized_summary.json"
    semantic_factorized = read_json(semantic_factorized_path) if semantic_factorized_path.exists() else {
        "variant_summary_rows": [],
        "head_summary_rows": [],
        "rescue_summary_rows": [],
    }
    schema_stagewise_path = run_root / "schema_stagewise" / "schema_stagewise_summary.json"
    schema_stagewise = read_json(schema_stagewise_path) if schema_stagewise_path.exists() else {"summary_rows": []}
    mechanism_audit_path = run_root / "mechanism_audit" / "mechanism_audit_summary.json"
    mechanism_audit = read_json(mechanism_audit_path) if mechanism_audit_path.exists() else {
        "component_rows": [],
        "edge_rows": [],
        "claim_tiers": {},
    }
    mlp27_steering_path = run_root / "mlp27_steering" / "mlp27_steering_summary.json"
    mlp27_steering = read_json(mlp27_steering_path) if mlp27_steering_path.exists() else {"summary_rows": []}
    late_writer_backup_path = run_root / "late_writer_backup" / "late_writer_backup_summary.json"
    late_writer_backup = read_json(late_writer_backup_path) if late_writer_backup_path.exists() else {
        "candidate_summary_rows": [],
        "base_summary_rows": [],
    }
    query_decision_path = run_root / "query_decision_chain" / "query_decision_summary.json"
    query_decision = read_json(query_decision_path) if query_decision_path.exists() else {
        "component_summary_rows": [],
        "step_summary_rows": [],
        "edge_summary_rows": [],
        "key_findings": {},
    }
    instruction_commitment_path = run_root / "instruction_commitment" / "instruction_commitment_summary.json"
    instruction_commitment = read_json(instruction_commitment_path) if instruction_commitment_path.exists() else {
        "variant_summary_rows": [],
        "query_summary_rows": [],
        "no_tool_summary_rows": [],
        "top_clean_instructions": [],
        "top_corrupt_instructions": [],
    }
    instruction_lead_path = run_root / "instruction_lead" / "instruction_lead_summary.json"
    instruction_lead = read_json(instruction_lead_path) if instruction_lead_path.exists() else {
        "variant_summary_rows": [],
        "query_summary_rows": [],
    }

    signed_rows = {str(r["group"]): r for r in signed_validate.get("summary_rows", [])}
    flip_rows = {str(r["group"]): r for r in token_flip.get("summary_rows", [])}
    node_rows = list(node_importance.get("summary_rows", []))
    func_rows = list(functional.get("summary_rows", []))
    func_validate_rows = {str(r["group"]): r for r in functional_validate.get("summary_rows", [])}
    chain_rows_by_path = {}
    for row in semantic_chain.get("summary_rows", []):
        chain_rows_by_path.setdefault(str(row["path_key"]), []).append(row)

    lines: List[str] = []
    lines.append("# Tool-Call Signed Circuit Final Report")
    lines.append("")
    lines.append("## Executive Summary")
    lines.append("")
    lines.append(
        f"- Dataset: `{bidi['forward']['n_samples']}` discovery samples per direction in this run root."
    )
    lines.append(
        f"- Final structural signed circuit: `{signed['n_nodes']}` nodes / `{signed['n_edges']}` edges."
    )
    full_row = signed_rows.get("full_signed_circuit", {})
    lines.append(
        f"- Full-circuit KL recovery: promote `{fmt(full_row.get('promote_suff_ratio_median'))}`, suppress `{fmt(full_row.get('suppress_suff_ratio_median'))}`."
    )
    lines.append(
        f"- Full-circuit top-1 flips: promote `{fmt(full_row.get('promote_tool_top1_rate'))}`, suppress `{fmt(full_row.get('suppress_no_tool_top1_rate'))}`."
    )
    lines.append("")
    lines.append("## Structural Story")
    lines.append("")
    lines.append("- Structural groups remain internal analysis tags:")
    for key, nodes in signed.get("groups", {}).items():
        lines.append(f"  - `{key}`: `{len(nodes)}` nodes")
    lines.append("")
    lines.append("Core structural artifacts:")
    lines.append("- `final_signed_circuit/final_signed_circuit.png`")
    lines.append("- `bidirectional/bidirectional_summary.json`")
    lines.append("- `signed_validate/signed_group_report.json`")
    lines.append("")
    lines.append("## Functional Semantic Groups")
    lines.append("")
    for row in func_rows:
        lines.append(
            f"- `{row['functional_label']}`: `{row['n_nodes']}` nodes, "
            f"promote median `{fmt(row['promote_strength_median'])}`, "
            f"suppress median `{fmt(row['suppress_strength_median'])}`."
        )
    lines.append("")
    lines.append("Functional artifacts:")
    lines.append("- `functional_groups/functional_group_graph.png`")
    lines.append("- `functional_groups/functional_node_table.csv`")
    lines.append("- `functional_validate/functional_group_report.json`")
    lines.append("")
    lines.append("## Faithfulness Checks")
    lines.append("")
    lines.append("- Structural group validation:")
    for key in ["symmetric_backbone", "tool_bias_backbone", "no_tool_bias_backbone", "tool_tail", "no_tool_tail"]:
        if key not in signed_rows:
            continue
        row = signed_rows[key]
        lines.append(
            f"  - `{key}`: suff `{fmt(row['promote_suff_ratio_median'])}/{fmt(row['suppress_suff_ratio_median'])}`, "
            f"nec `{fmt(row['promote_nec_drop_median'])}/{fmt(row['suppress_nec_drop_median'])}`."
        )
    lines.append("- Functional group validation:")
    for key in functional.get("groups", {}).keys():
        row = func_validate_rows.get(key)
        if not row:
            continue
        lines.append(
            f"  - `{row['functional_label']}`: suff `{fmt(row['promote_suff_ratio_median'])}/{fmt(row['suppress_suff_ratio_median'])}`, "
            f"nec `{fmt(row['promote_nec_drop_median'])}/{fmt(row['suppress_nec_drop_median'])}`."
        )
    lines.append("")
    lines.append("## Behavioral Evidence")
    lines.append("")
    for key in ["shared_backbone", "shared_backbone_exclusive", "forward_selective", "reverse_selective"]:
        row = flip_rows.get(key)
        if not row:
            continue
        lines.append(
            f"- `{key}`: top-1 `{fmt(row['promote_tool_top1_rate'])}/{fmt(row['suppress_no_tool_top1_rate'])}`, "
            f"boundary `{fmt(row['promote_boundary_flip_rate'])}/{fmt(row['suppress_boundary_flip_rate'])}`."
        )
    lines.append("")
    if semantic_chain.get("paths"):
        lines.append("## Initial Extracted Chain Candidates")
        lines.append("")
        if query_decision.get("step_summary_rows") or instruction_commitment.get("variant_summary_rows"):
            lines.append("- These are discovery-time chain candidates. The refined fixed-schema and instruction-level analyses below supersede the early `L2H14`-anchored query interpretation.")
        for path in semantic_chain.get("paths", []):
            rows = sorted(chain_rows_by_path.get(str(path["key"]), []), key=lambda r: int(r["step_idx"]))
            final_row = rows[-1] if rows else {}
            lines.append(
                f"- `{path['label']}`: `{' -> '.join(path['nodes'])}`, "
                f"final cumulative `{fmt(final_row.get('cumulative_ratio_median'))}`, "
                f"top1 `{fmt(final_row.get('top1_rate'))}`."
            )
        lines.append("- Detailed chain report: `semantic_chain/semantic_chain_report.md`")
        lines.append("- Progression plot: `semantic_chain/semantic_chain_progression.png`")
        lines.append("")
    if semantic_factorized.get("variant_summary_rows"):
        variant_rows = {str(r["variant"]): r for r in semantic_factorized.get("variant_summary_rows", [])}
        rescue_rows = semantic_factorized.get("rescue_summary_rows", [])
        lines.append("## Factorized Counterfactuals")
        lines.append("")
        for key in ["clean_full", "corrupt_full", "clean_no_schema", "clean_schema_mismatch", "clean_no_protocol"]:
            row = variant_rows.get(key)
            if not row:
                continue
            lines.append(
                f"- `{key}`: tool `{fmt(row.get('tool_endpoint_score_median'))}`, "
                f"no-tool `{fmt(row.get('no_tool_endpoint_score_median'))}`, "
                f"tool-top1 `{fmt(row.get('tool_top1_rate'))}`."
            )
        for row in rescue_rows:
            lines.append(
                f"- rescue `{row['path_key']}` on `{row['base_variant']}`: "
                f"`{fmt(row.get('rescue_ratio_median'))}` with top1 `{fmt(row.get('top1_rate'))}`."
            )
        if schema_stagewise.get("summary_rows"):
            for row in schema_stagewise.get("summary_rows", []):
                lines.append(
                    f"- schema step `{row['step_idx']}` / `{row['node']}` on `{row['base_variant']}`: "
                    f"rescue `{fmt(row.get('rescue_ratio_median'))}`, top1 `{fmt(row.get('top1_rate'))}`."
                )
            lines.append("- Detailed schema-stagewise report: `schema_stagewise/schema_stagewise_report.md`")
        lines.append("- Detailed factorized report: `semantic_factorized/semantic_factorized_report.md`")
        lines.append("")
    if mechanism_audit.get("component_rows") or mlp27_steering.get("summary_rows") or late_writer_backup.get("base_summary_rows"):
        lines.append("## Mechanism Audit")
        lines.append("")
        claim_tiers = mechanism_audit.get("claim_tiers", {})
        if claim_tiers:
            lines.append(
                f"- Claim tiers: A=`{len(claim_tiers.get('level_A', []))}`, "
                f"B=`{len(claim_tiers.get('level_B', []))}`, C=`{len(claim_tiers.get('level_C', []))}`."
            )
        component_rows = list(mechanism_audit.get("component_rows", []))
        edge_rows = list(mechanism_audit.get("edge_rows", []))
        if component_rows:
            lines.append("- Main component findings:")
            for row in component_rows:
                if str(row.get("tier")) not in {"A", "B"}:
                    continue
                lines.append(
                    f"  - `{row['component']}` [{row['tier']}]: {row['object_language_function']} "
                    f"(direct `{fmt(row['direct_write_strength'])}`, path `{fmt(row['path_mediation_strength'])}`)."
                )
        if edge_rows:
            lines.append("- Main edge findings:")
            for row in edge_rows:
                if str(row.get("tier")) != "A":
                    continue
                lines.append(
                    f"  - `{row['edge']}`: {row['object_language_function']} "
                    f"(mediated `{fmt(row['best_mediated_ratio'])}` on `{row['best_base_variant']}`)."
                )
        if mlp27_steering.get("summary_rows"):
            by_base = {}
            for row in mlp27_steering.get("summary_rows", []):
                base = str(row["base_variant"])
                alpha = float(row["alpha"])
                if alpha == 1.5:
                    by_base[base] = row
            if by_base:
                lines.append("- MLP27 steering at alpha `1.5`:")
                for base in ["corrupt_full", "clean_no_schema", "clean_schema_mismatch", "clean_no_protocol"]:
                    row = by_base.get(base)
                    if not row:
                        continue
                    lines.append(
                        f"  - `{base}`: decision `{fmt(row['decision_score_median'])}`, "
                        f"tool-top1 `{fmt(row['tool_top1_rate'])}`, boundary `{fmt(row['boundary_flip_rate'])}`."
                    )
        if late_writer_backup.get("base_summary_rows"):
            lines.append("- Late writer backup search:")
            for row in late_writer_backup.get("base_summary_rows", []):
                lines.append(
                    f"  - `{row['base_variant']}`: MLP27 direct `{fmt(row['mlp27_direct_rescue_median'])}`, "
                    f"best alt direct `{row['best_alt_direct_candidate']}`=`{fmt(row['best_alt_direct_rescue_median'])}`, "
                    f"best alt with MLP27 blocked `{row['best_alt_independent_candidate']}`=`{fmt(row['best_alt_independent_rescue_median'])}`."
                )
        lines.append("- Mechanism audit table: `mechanism_audit/component_evidence_table.csv`")
        lines.append("- Writing boundary: `mechanism_audit/writing_boundary.md`")
        lines.append("- MLP27 steering: `mlp27_steering/mlp27_steering_report.md`")
        lines.append("- Late writer backup search: `late_writer_backup/late_writer_backup_report.md`")
        lines.append("")
    if query_decision.get("step_summary_rows"):
        lines.append("## Fixed-Schema Query Decision")
        lines.append("")
        key_findings = query_decision.get("key_findings", {})
        lines.append(
            f"- `L20H5` clean->corrupt rescue `{fmt(key_findings.get('query_l20h5_rescue_median'))}`; "
            f"`L21H1` `{fmt(key_findings.get('query_l21h1_rescue_median'))}`; "
            f"`L21H12` `{fmt(key_findings.get('query_l21h12_rescue_median'))}`; "
            f"`MLP27` `{fmt(key_findings.get('query_mlp27_rescue_median'))}`."
        )
        lines.append(
            f"- Cumulative fixed-schema query chain tool-top1 `{fmt(key_findings.get('query_chain_final_top1_rate'))}`."
        )
        lines.append(
            f"- Competing suppressive chain no-tool top1 `{fmt(key_findings.get('suppress_chain_final_top1_rate'))}`."
        )
        lines.append(
            f"- Key edge mediation: `L20H5->L21H12` `{fmt(key_findings.get('edge_l20h5_l21h12_mediated'))}`, "
            f"`L21H12->MLP27` `{fmt(key_findings.get('edge_l21h12_mlp27_mediated'))}`, "
            f"`MLP17->L20H5` `{fmt(key_findings.get('edge_mlp17_l20h5_mediated'))}`."
        )
        lines.append("- Detailed report: `query_decision_chain/query_decision_report.md`")
        lines.append("- Stepwise plot: `query_decision_chain/query_decision_stepwise.png`")
        lines.append("")
    if instruction_commitment.get("variant_summary_rows"):
        lines.append("## Instruction-Level Commitment")
        lines.append("")
        variant_rows = {str(r["variant"]): r for r in instruction_commitment.get("variant_summary_rows", [])}
        for key in ["clean_full", "clean_with_corrupt_instruction", "corrupt_with_clean_instruction", "corrupt_full"]:
            row = variant_rows.get(key)
            if not row:
                continue
            lines.append(
                f"- `{key}`: decision `{fmt(row['decision_score_median'])}`, "
                f"tool-top1 `{fmt(row['tool_top1_rate'])}`, no-tool-top1 `{fmt(row['no_tool_top1_rate'])}`."
            )
        q_rows = instruction_commitment.get("query_summary_rows", [])
        n_rows = instruction_commitment.get("no_tool_summary_rows", [])
        if q_rows:
            row = q_rows[-1]
            lines.append(
                f"- Query chain on corrupt instruction swap: `{row['nodes']}` -> "
                f"rescue `{fmt(row['rescue_ratio_median'])}`, tool-top1 `{fmt(row['tool_top1_rate'])}`."
            )
        if n_rows:
            row = n_rows[-1]
            lines.append(
                f"- No-tool chain on clean instruction swap: `{row['nodes']}` -> "
                f"rescue `{fmt(row['rescue_ratio_median'])}`, no-tool-top1 `{fmt(row['no_tool_top1_rate'])}`."
            )
        top_clean = instruction_commitment.get("top_clean_instructions", [])[:3]
        top_corrupt = instruction_commitment.get("top_corrupt_instructions", [])[:3]
        if top_clean:
            lines.append("- Most common clean instruction lines:")
            for text, count in top_clean:
                lines.append(f"  - `{count}`x `{text}`")
        if top_corrupt:
            lines.append("- Most common corrupt instruction lines:")
            for text, count in top_corrupt:
                lines.append(f"  - `{count}`x `{text}`")
        lines.append("- Detailed report: `instruction_commitment/instruction_commitment_report.md`")
        lines.append("- Variant effect plot: `instruction_commitment/instruction_variant_effects.png`")
        lines.append("")
    if instruction_lead.get("variant_summary_rows"):
        lines.append("## Minimal Lead Cue")
        lines.append("")
        variant_rows = {str(r["variant"]): r for r in instruction_lead.get("variant_summary_rows", [])}
        for key in ["clean_full", "clean_with_corrupt_lead", "corrupt_with_clean_lead", "corrupt_full"]:
            row = variant_rows.get(key)
            if not row:
                continue
            lines.append(
                f"- `{key}`: decision `{fmt(row['decision_score_median'])}`, "
                f"tool-top1 `{fmt(row['tool_top1_rate'])}`, no-tool-top1 `{fmt(row['no_tool_top1_rate'])}`."
            )
        q_rows = instruction_lead.get("query_summary_rows", [])
        if q_rows:
            row = q_rows[-1]
            lines.append(
                f"- Query chain on corrupt lead swap: `{row['nodes']}` -> "
                f"rescue `{fmt(row['rescue_ratio_median'])}`, tool-top1 `{fmt(row['tool_top1_rate'])}`."
            )
        lines.append("- Detailed report: `instruction_lead/instruction_lead_report.md`")
        lines.append("")
    lines.append("## Node / Edge Diagnostics")
    lines.append("")
    lines.append("- Top node diagnostics:")
    for row in node_rows[:10]:
        lines.append(
            f"  - `{row['node']}`: suff `{fmt(row['promote_suff_ratio_median'])}/{fmt(row['suppress_suff_ratio_median'])}`, "
            f"nec `{fmt(row['promote_nec_drop_median'])}/{fmt(row['suppress_nec_drop_median'])}`."
        )
    edge_report_path = run_root / "edge_importance" / "signed_edge_mediation_report.json"
    if edge_report_path.exists():
        edge_report = read_json(edge_report_path)
        edge_rows = list(edge_report.get("summary_rows", []))
        lines.append("- Top edge diagnostics:")
        for row in edge_rows[:10]:
            lines.append(
                f"  - `{row['edge']}`: mediated `{fmt(row['promote_mediated_ratio_median'])}/{fmt(row['suppress_mediated_ratio_median'])}`."
            )
    lines.append("")
    method_report_path = run_root / "method_benchmark" / "benchmark_summary.json"
    if method_report_path.exists():
        method_report = read_json(method_report_path)
        lines.append("## Method Comparison")
        lines.append("")
        for row in method_report.get("summary_rows", []):
            lines.append(
                f"- `{row['method']}`: faithfulness `{fmt(row.get('faithfulness_median'))}`, "
                f"sparsity `{fmt(row.get('n_nodes'))}`, semantic `{fmt(row.get('semantic_coherence'))}`, "
                f"runtime `{fmt(row.get('runtime_seconds'))}` sec."
            )
        lines.append("")
    lines.append("## Artifact Index")
    lines.append("")
    lines.append(f"- Run root: `{run_root}`")
    if (run_root / "FINAL_MECHANISTIC_RESULT.md").exists():
        lines.append("- Final mechanistic result: `FINAL_MECHANISTIC_RESULT.md`")
    lines.append("- Main figure: `final_signed_circuit/final_signed_circuit.png`")
    lines.append("- Structural validation: `signed_validate/signed_group_validation_heatmap.png`")
    lines.append("- Functional graph: `functional_groups/functional_group_graph.png`")
    lines.append("- Functional validation: `functional_validate/functional_group_validation_heatmap.png`")
    if semantic_chain.get("paths"):
        lines.append("- Semantic chain: `semantic_chain/semantic_chain_report.md`")
    if semantic_factorized.get("variant_summary_rows"):
        lines.append("- Factorized counterfactuals: `semantic_factorized/semantic_factorized_report.md`")
    if schema_stagewise.get("summary_rows"):
        lines.append("- Schema stagewise: `schema_stagewise/schema_stagewise_report.md`")
    if mechanism_audit.get("component_rows"):
        lines.append("- Mechanism audit: `mechanism_audit/mechanism_audit_summary.json`")
    if mlp27_steering.get("summary_rows"):
        lines.append("- MLP27 steering: `mlp27_steering/mlp27_steering_report.md`")
    if late_writer_backup.get("base_summary_rows"):
        lines.append("- Late writer backup: `late_writer_backup/late_writer_backup_report.md`")
    if query_decision.get("step_summary_rows"):
        lines.append("- Fixed-schema query decision: `query_decision_chain/query_decision_report.md`")
    if instruction_commitment.get("variant_summary_rows"):
        lines.append("- Instruction commitment: `instruction_commitment/instruction_commitment_report.md`")
    if instruction_lead.get("variant_summary_rows"):
        lines.append("- Instruction lead cue: `instruction_lead/instruction_lead_report.md`")
    if (run_root / "final_head_attention_audit" / "head_final_audit_summary.json").exists():
        lines.append("- Final head attention audit: `final_head_attention_audit/head_final_audit_report.md`")
    if (run_root / "final_mechanism_evidence" / "final_mechanism_evidence_summary.json").exists():
        lines.append("- Final mechanism evidence: `final_mechanism_evidence/final_mechanism_evidence_summary.json`")
    lines.append("- Token flips: `token_flip/group_token_flip_summary.csv`")
    lines.append("- Node importance: `node_importance/signed_node_importance_heatmap.png`")
    lines.append("- Trajectory: `signed_layer_trajectory/signed_layer_trajectory.png`")
    lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(str(out_path))


if __name__ == "__main__":
    main()
