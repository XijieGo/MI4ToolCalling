#!/usr/bin/env python3
"""
Assemble a final evidence table aligned to the refined mechanism story.
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


def write_csv(rows: List[Dict[str, object]], out_path: Path) -> None:
    if not rows:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: object, digits: int = 3) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return str(value)


def find_row(rows: List[Dict[str, object]], key: str, value: str) -> Dict[str, object]:
    for row in rows:
        if str(row.get(key)) == value:
            return row
    return {}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the final evidence table aligned to the refined mechanism story.")
    parser.add_argument("--run-root", type=str, required=True)
    parser.add_argument("--output-root", type=str, required=True)
    args = parser.parse_args()

    run_root = Path(args.run_root).resolve()
    out_root = Path(args.output_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    query_decision = read_json(run_root / "query_decision_chain" / "query_decision_summary.json")
    instruction_commitment = read_json(run_root / "instruction_commitment" / "instruction_commitment_summary.json")
    instruction_lead = read_json(run_root / "instruction_lead" / "instruction_lead_summary.json")
    late_writer_backup = read_json(run_root / "late_writer_backup" / "late_writer_backup_summary.json")
    mechanism_audit = read_json(run_root / "mechanism_audit" / "mechanism_audit_summary.json")
    mlp27_steering = read_json(run_root / "mlp27_steering" / "mlp27_steering_summary.json")
    head_audit_path = run_root / "final_head_attention_audit" / "head_final_audit_summary.json"
    head_audit = read_json(head_audit_path) if head_audit_path.exists() else {
        "final_rows": [],
        "span_summary_rows": [],
        "qkv_summary_rows": [],
    }

    q_comp = list(query_decision.get("component_summary_rows", []))
    q_edges = list(query_decision.get("edge_summary_rows", []))
    q_steps = list(query_decision.get("step_summary_rows", []))
    ic_read = list(instruction_commitment.get("read_summary_rows", []))
    ic_vars = {str(r["variant"]): r for r in instruction_commitment.get("variant_summary_rows", [])}
    il_vars = {str(r["variant"]): r for r in instruction_lead.get("variant_summary_rows", [])}
    backup_by_base = {str(r["base_variant"]): r for r in late_writer_backup.get("base_summary_rows", [])}
    mech_comp = {str(r["component"]): r for r in mechanism_audit.get("component_rows", [])}
    mech_edge = {str(r["edge"]): r for r in mechanism_audit.get("edge_rows", [])}
    head_final = {str(r["head"]): r for r in head_audit.get("final_rows", [])}
    head_span = list(head_audit.get("span_summary_rows", []))
    head_qkv = list(head_audit.get("qkv_summary_rows", []))
    steer_by_base = {
        str(r["base_variant"]): r
        for r in mlp27_steering.get("summary_rows", [])
        if float(r.get("alpha", -1.0)) == 1.5
    }

    read_lookup = {
        (str(r["variant"]), str(r["component"]), str(r["set"])): r
        for r in ic_read
    }

    component_rows: List[Dict[str, object]] = []
    component_specs = [
        (
            "L20H5",
            "late user-conditioned ingress point",
            "reads the instruction-level commitment cue from the user request and injects it into the tool-writing route",
            "instruction-level commitment cue",
            "early tool-biased state that can be routed downstream",
            "general user-instruction salience rather than commitment-specific cue",
            "weak-main-text",
        ),
        (
            "L21H1",
            "late query router",
            "relays the user-conditioned ingress state toward the late tool writer",
            "mixed user-side / instruction-conditioned state",
            "tool-biased routed state",
            "late correlated router rather than specific commitment relay",
            "weak-main-text",
        ),
        (
            "L21H12",
            "late commitment/schema router",
            "combines the user-conditioned commitment signal with the existing tool-call channel and sends it to MLP27",
            "instruction-conditioned late state plus tool-call channel state",
            "tool-biased routed state into MLP27",
            "mostly protocol-tag reader rather than commitment-conditioned router",
            "main-text",
        ),
        (
            "L24H6",
            "late pre-writer relay",
            "carries the user-conditioned tool state into the output-adjacent writer region",
            "late tool-biased routed state",
            "tool-biased pre-output state",
            "late correlated relay rather than functional pre-writer",
            "weak-main-text",
        ),
        (
            "MLP27",
            "primary late writer",
            "writes the final tool-call-favoring state into the output direction",
            "late commitment-conditioned tool state",
            "<tool_call>-favoring residual direction",
            "strong late correlated node rather than primary writer",
            "main-text",
        ),
        (
            "L16H4",
            "no-tool-biased user-side reader",
            "reads a competing no-tool-biased user-side state and seeds the suppressive route",
            "user-side no-tool-biased state",
            "suppression route state",
            "anti-protocol cue rather than no-tool-biased reader",
            "weak-main-text",
        ),
        (
            "MLP17",
            "no-tool writer",
            "writes the competing no-tool-favoring state",
            "no-tool-biased user-side state",
            "no_tool-favoring residual direction",
            "generic negative writer rather than no-tool-specific writer",
            "main-text",
        ),
        (
            "L23H6",
            "late suppressive relay",
            "relays the no-tool written state toward the output region and suppresses the tool route",
            "no-tool written state",
            "late suppressive state",
            "mixed late node rather than dedicated suppressive relay",
            "main-text",
        ),
    ]

    for component, label, claim, read_obj, write_obj, alt, strength in component_specs:
        q_row = find_row(q_comp, "component", component)
        direct = q_row.get("rescue_ratio_median", "")
        top1 = q_row.get("top1_rate", "")
        read_evidence = ""
        if component in {"L20H5", "L21H1", "L21H12", "L16H4"}:
            clean_inst = read_lookup.get(("clean_full", component, "instruction_line"), {}).get("mass_median")
            corrupt_inst = read_lookup.get(("clean_with_corrupt_instruction", component, "instruction_line"), {}).get("mass_median")
            clean_task = read_lookup.get(("clean_full", component, "task_body"), {}).get("mass_median")
            read_evidence = (
                f"instruction mass clean `{fmt(clean_inst)}` -> corrupt-instruction `{fmt(corrupt_inst)}`; "
                f"task-body mass clean `{fmt(clean_task)}`"
            )
        if component in head_final:
            hf = head_final[component]
            read_evidence = (
                f"best read span `{hf.get('best_read_span')}` with density `{fmt(hf.get('best_read_density_median'))}`; "
                f"causal span `{hf.get('best_causal_span')}` rescue `{fmt(hf.get('best_causal_span_rescue_median'))}`"
            )
        path_evidence = ""
        if component == "L20H5":
            path_evidence = (
                f"`L20H5->L21H12` mediation `{fmt(find_row(q_edges, 'edge', 'L20H5->L21H12').get('mediated_ratio_median'))}`; "
                f"`L20H5->L21H1` `{fmt(find_row(q_edges, 'edge', 'L20H5->L21H1').get('mediated_ratio_median'))}`"
            )
        elif component == "L21H1":
            path_evidence = f"`L21H1->MLP27` mediation `{fmt(find_row(q_edges, 'edge', 'L21H1->MLP27').get('mediated_ratio_median'))}`"
        elif component == "L21H12":
            path_evidence = f"`L21H12->MLP27` mediation `{fmt(find_row(q_edges, 'edge', 'L21H12->MLP27').get('mediated_ratio_median'))}`"
        elif component == "MLP27":
            row = backup_by_base.get("corrupt_full", {})
            path_evidence = (
                f"steering `corrupt_full` tool-top1 `{fmt(steer_by_base.get('corrupt_full', {}).get('tool_top1_rate'))}`; "
                f"backup search best independent alt `{row.get('best_alt_independent_candidate')}`=`{fmt(row.get('best_alt_independent_rescue_median'))}`"
            )
        elif component == "L16H4":
            path_evidence = f"`L16H4->MLP17` mediation `{fmt(mech_edge.get('L16H4->MLP17', {}).get('best_mediated_ratio'))}`"
        elif component == "MLP17":
            path_evidence = (
                f"`MLP17->L23H6` mediation `{fmt(mech_edge.get('MLP17->L23H6', {}).get('best_mediated_ratio'))}`; "
                f"`MLP17->L20H5` mediation `{fmt(find_row(q_edges, 'edge', 'MLP17->L20H5').get('mediated_ratio_median'))}`"
            )
        elif component == "L23H6":
            path_evidence = f"`MLP17->L23H6` mediation `{fmt(mech_edge.get('MLP17->L23H6', {}).get('best_mediated_ratio'))}`"
        elif component == "L24H6":
            path_evidence = f"full query route reaches tool-top1 `{fmt(find_row(q_steps, 'step_label', 'query_to_late_router').get('top1_rate'))}` before MLP27"
        if component in head_final:
            qkv_best = head_final[component]
            path_evidence = path_evidence + (
                ("; " if path_evidence else "")
                + f"best QKV component `{qkv_best.get('best_qkv_component')}` rescue `{fmt(qkv_best.get('best_qkv_rescue_median'))}`"
            )

        counterfactual = ""
        if component in {"L20H5", "L21H1", "L21H12", "L24H6", "MLP27"}:
            counterfactual = (
                f"instruction swap flips to tool-top1 `{fmt(ic_vars.get('corrupt_with_clean_instruction', {}).get('tool_top1_rate'))}`; "
                f"lead swap flips to tool-top1 `{fmt(il_vars.get('corrupt_with_clean_lead', {}).get('tool_top1_rate'))}`"
            )
        else:
            counterfactual = (
                f"clean-with-corrupt-instruction flips to no-tool-top1 `{fmt(ic_vars.get('clean_with_corrupt_instruction', {}).get('no_tool_top1_rate'))}`"
            )

        risk = ""
        if component == "L20H5":
            risk = "still not proven to encode abstract action-demand beyond this dataset's commitment cue"
        elif component == "L21H1":
            risk = "routing role is clearer than semantic specificity"
        elif component == "L21H12":
            risk = "still mixes commitment-conditioned state with existing protocol/schema channel"
        elif component == "MLP27":
            risk = "primary writer, but not proven unique bottleneck on no-schema variants"
        elif component == "L16H4":
            risk = "ordinary-answer prior vs anti-tool bias remains partially entangled"
        elif component == "MLP17":
            risk = "could still be a strong generic negative writer"
        elif component == "L23H6":
            risk = "late relay may still mix read and write behavior"
        elif component == "L24H6":
            risk = "writer vs relay distinction remains unresolved"

        component_rows.append(
            {
                "component": component,
                "final_function": label,
                "object_language_claim": claim,
                "reads": read_obj,
                "writes": write_obj,
                "direct_evidence": f"rescue `{fmt(direct)}`, top1 `{fmt(top1)}`",
                "read_evidence": read_evidence,
                "path_evidence": path_evidence,
                "counterfactual_evidence": counterfactual,
                "alternative_excluded": alt,
                "remaining_risk": risk,
                "write_strength": strength,
            }
        )

    edge_rows: List[Dict[str, object]] = []
    edge_specs = [
        ("L20H5->L21H1", "passes the instruction-conditioned ingress state into a late query router", "weak-main-text"),
        ("L20H5->L21H12", "passes the instruction-conditioned ingress state into the main commitment-conditioned late router", "main-text"),
        ("L21H1->MLP27", "sends late user-conditioned routed state into the final writer", "weak-main-text"),
        ("L21H12->MLP27", "sends the main commitment-conditioned late state into the final writer", "main-text"),
        ("L16H4->MLP17", "passes the competing no-tool-biased user-side state into the no-tool writer", "main-text"),
        ("MLP17->L23H6", "passes the written no-tool state into the late suppressive relay", "main-text"),
        ("MLP17->L20H5", "suppresses the user-conditioned tool ingress route", "weak-main-text"),
    ]
    for edge, claim, strength in edge_specs:
        row = find_row(q_edges, "edge", edge)
        if not row:
            row = mech_edge.get(edge, {})
        edge_rows.append(
            {
                "edge": edge,
                "object_language_claim": claim,
                "mediated_evidence": fmt(row.get("mediated_ratio_median", row.get("best_mediated_ratio"))),
                "strength": strength,
            }
        )

    claim_tree = {
        "level_A": [
            "instruction-level commitment cue alone flips the first-token decision",
            "minimal lead phrase alone flips the first-token decision",
            "user-conditioned late ingress route `L20H5 -> L21H1/L21H12 -> L24H6 -> MLP27` drives the tool-call decision under fixed schema/protocol",
            "MLP27 acts as a primary late writer for the tool-call decision",
            "a competing no-tool chain `L16H4 -> MLP17 -> L23H6` can suppress that route",
        ],
        "level_B": [
            "`L20H5` specifically encodes performative demand rather than broader instruction salience",
            "`L16H4` specifically encodes ordinary-answer prior rather than anti-tool bias",
            "`direct-answer sufficiency` is the exact abstract latent variable under the observed commitment cue",
        ],
        "level_C": [
            "unified mode switcher",
            "generic arbitration zone",
            "early `L2H14` branch as the main final query story",
        ],
    }

    write_csv(component_rows, out_root / "final_component_evidence_table.csv")
    write_csv(edge_rows, out_root / "final_edge_evidence_table.csv")
    (out_root / "final_claim_tree.json").write_text(json.dumps(claim_tree, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {
        "component_rows": component_rows,
        "edge_rows": edge_rows,
        "claim_tree": claim_tree,
        "artifacts": {
            "component_csv": str(out_root / "final_component_evidence_table.csv"),
            "edge_csv": str(out_root / "final_edge_evidence_table.csv"),
            "claim_tree_json": str(out_root / "final_claim_tree.json"),
        },
    }
    (out_root / "final_mechanism_evidence_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
