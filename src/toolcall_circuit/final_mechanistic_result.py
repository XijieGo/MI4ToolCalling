#!/usr/bin/env python3
"""
Build a single mechanistic result report for the refined tool-call story.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

if __package__ in {None, ""}:
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


def read_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def fmt(value: object, digits: int = 3) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return str(value)


def find_step(rows: List[Dict[str, object]], step_idx: int) -> Dict[str, object]:
    for row in rows:
        if int(row.get("step_idx", -1)) == int(step_idx):
            return row
    return {}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the refined final mechanistic result report.")
    parser.add_argument("--run-root", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    run_root = Path(args.run_root).resolve()
    out_path = Path(args.output).resolve()

    signed_validate = read_json(run_root / "signed_validate" / "signed_group_report.json")
    mechanism_audit = read_json(run_root / "mechanism_audit" / "mechanism_audit_summary.json")
    mlp27_steering = read_json(run_root / "mlp27_steering" / "mlp27_steering_summary.json")
    late_writer_backup = read_json(run_root / "late_writer_backup" / "late_writer_backup_summary.json")
    query_decision = read_json(run_root / "query_decision_chain" / "query_decision_summary.json")
    instruction_commitment = read_json(run_root / "instruction_commitment" / "instruction_commitment_summary.json")
    instruction_lead = read_json(run_root / "instruction_lead" / "instruction_lead_summary.json")

    signed_rows = {str(r["group"]): r for r in signed_validate.get("summary_rows", [])}
    full_row = signed_rows.get("full_signed_circuit", {})
    claim_tiers = mechanism_audit.get("claim_tiers", {})
    key_findings = query_decision.get("key_findings", {})
    query_steps = list(query_decision.get("step_summary_rows", []))
    suppress_steps = [r for r in query_steps if str(r.get("family")) == "suppress"]
    query_steps = [r for r in query_steps if str(r.get("family")) == "query"]
    if not suppress_steps:
        suppress_steps = list(query_decision.get("step_summary_rows", []))[len(query_steps):]

    instruction_variants = {str(r["variant"]): r for r in instruction_commitment.get("variant_summary_rows", [])}
    lead_variants = {str(r["variant"]): r for r in instruction_lead.get("variant_summary_rows", [])}
    backup_rows = {str(r["base_variant"]): r for r in late_writer_backup.get("base_summary_rows", [])}
    steer_rows = [r for r in mlp27_steering.get("summary_rows", []) if float(r.get("alpha", -1.0)) == 1.5]
    steer_by_base = {str(r["base_variant"]): r for r in steer_rows}

    lines: List[str] = []
    lines.append("# Final Mechanistic Result")
    lines.append("")
    lines.append("## Core Claim")
    lines.append("")
    lines.append(
        "In this fixed tool environment, the decisive variable is not schema availability itself. "
        "The decisive variable is an instruction-level commitment cue in the user request: whether the model is being asked to commit to an external file-writing / execution-style delivery, or merely to author the function body in text."
    )
    lines.append("")
    lines.append(
        f"The resulting signed circuit remains highly faithful: full-circuit KL recovery `{fmt(full_row.get('promote_suff_ratio_median'))}` / `{fmt(full_row.get('suppress_suff_ratio_median'))}`, "
        f"top-1 `{fmt(full_row.get('promote_tool_top1_rate'))}` / `{fmt(full_row.get('suppress_no_tool_top1_rate'))}`."
    )
    lines.append("")
    lines.append("## Final 4-Step Mechanistic Chain")
    lines.append("")
    lines.append(
        "1. The model reads an instruction-level commitment cue from the first user instruction line. "
        "Swapping only that line is enough to flip the first-token decision across the whole dataset."
    )
    if instruction_variants:
        lines.append(
            f"   Evidence: `clean_full` decision `{fmt(instruction_variants['clean_full']['decision_score_median'])}` and tool-top1 `{fmt(instruction_variants['clean_full']['tool_top1_rate'])}`, "
            f"but `clean_with_corrupt_instruction` drops to decision `{fmt(instruction_variants['clean_with_corrupt_instruction']['decision_score_median'])}` and tool-top1 `{fmt(instruction_variants['clean_with_corrupt_instruction']['tool_top1_rate'])}`; "
            f"`corrupt_with_clean_instruction` flips back to tool-top1 `{fmt(instruction_variants['corrupt_with_clean_instruction']['tool_top1_rate'])}`."
        )
    if lead_variants:
        lines.append(
            f"   Stronger evidence: swapping only the minimal lead phrase has the same effect: "
            f"`clean_with_corrupt_lead` tool-top1 `{fmt(lead_variants['clean_with_corrupt_lead']['tool_top1_rate'])}`, "
            f"`corrupt_with_clean_lead` tool-top1 `{fmt(lead_variants['corrupt_with_clean_lead']['tool_top1_rate'])}`."
        )
    lines.append("")
    lines.append(
        "2. That commitment cue enters the tool-writing route through a user-conditioned late ingress path."
    )
    lines.append(
        "   The best-supported path is `L20H5 -> L21H1/L21H12 -> L24H6 -> MLP27`."
    )
    if key_findings:
        lines.append(
            f"   Evidence: `L20H5` rescue `{fmt(key_findings.get('query_l20h5_rescue_median'))}`, "
            f"`L21H1` `{fmt(key_findings.get('query_l21h1_rescue_median'))}`, "
            f"`L21H12` `{fmt(key_findings.get('query_l21h12_rescue_median'))}`, "
            f"`L21H12->MLP27` mediation `{fmt(key_findings.get('edge_l21h12_mlp27_mediated'))}`."
        )
    if query_steps:
        q_final = query_steps[-1]
        q_mid = find_step(query_steps, 4) or find_step(query_steps, 3)
        if q_mid:
            lines.append(
                f"   Stepwise buildup: by `{q_mid['nodes']}` the decision is already `{fmt(q_mid['decision_score_median'])}` with tool-top1 `{fmt(q_mid['top1_rate'])}`; "
                f"the full route reaches tool-top1 `{fmt(q_final['top1_rate'])}`."
            )
    lines.append("")
    lines.append(
        "3. `MLP27` writes this state into a `<tool_call>`-favoring output direction."
    )
    if steer_by_base:
        lines.append(
            f"   Evidence: at steering alpha `1.5`, `corrupt_full` reaches tool-top1 `{fmt(steer_by_base['corrupt_full']['tool_top1_rate'])}`, "
            f"`clean_schema_mismatch` reaches `{fmt(steer_by_base['clean_schema_mismatch']['tool_top1_rate'])}`, "
            f"`clean_no_protocol` still reaches `{fmt(steer_by_base['clean_no_protocol']['tool_top1_rate'])}`."
        )
    if backup_rows:
        row = backup_rows.get("corrupt_full", {})
        if row:
            lines.append(
                f"   Backup search: on `corrupt_full`, MLP27 direct rescue is `{fmt(row.get('mlp27_direct_rescue_median'))}` while the best independent alternative under MLP27 block is only `{fmt(row.get('best_alt_independent_rescue_median'))}` from `{row.get('best_alt_independent_candidate')}`."
            )
        row = backup_rows.get("clean_no_schema", {})
        if row:
            lines.append(
                f"   Caveat: on `clean_no_schema`, alternatives remain more competitive; MLP27 is therefore best described as a primary late writer, not a proven unique bottleneck."
            )
    lines.append("")
    lines.append(
        "4. A competing no-tool chain pushes the same fixed tool environment back toward `no_tool`."
    )
    lines.append(
        "   The best-supported path is `L16H4 -> MLP17 -> L23H6`."
    )
    if key_findings:
        lines.append(
            f"   Evidence: `MLP17` rescue `{fmt(key_findings.get('suppress_mlp17_rescue_median'))}`, "
            f"`MLP17->L20H5` mediation `{fmt(key_findings.get('edge_mlp17_l20h5_mediated'))}`, "
            f"full suppressive route no-tool top1 `{fmt(key_findings.get('suppress_chain_final_top1_rate'))}`."
        )
    lines.append("")
    lines.append("## Human-Level Interpretation")
    lines.append("")
    lines.append(
        "The strongest interpretation is that the circuit is tracking a performative distinction: "
        "is the user asking for a committed file/environment action, or merely for authored content? "
        "In this dataset, that distinction is carried overwhelmingly by the opening instruction phrase."
    )
    lines.append("")
    lines.append(
        "This supports three human-readable latent variables, with different confidence levels:"
    )
    lines.append(
        "- Strongest: `performative demand` or `instruction-level commitment cue`."
    )
    lines.append(
        "- Strong: `execution commitment threshold`, written by the late writer path around `MLP27`."
    )
    lines.append(
        "- Plausible but weaker: `direct-answer sufficiency`; it likely underlies the commitment cue, but the current dataset isolates instruction phrasing more directly than abstract answer sufficiency."
    )
    lines.append("")
    lines.append("## What We Can Write Strongly")
    lines.append("")
    level_a = claim_tiers.get("level_A", [])
    level_b = claim_tiers.get("level_B", [])
    level_c = claim_tiers.get("level_C", [])
    lines.append(
        f"- Claim tiers from the mechanism audit: A=`{len(level_a)}`, B=`{len(level_b)}`, C=`{len(level_c)}`."
    )
    lines.append(
        "- Strongest main-text statement: under fixed schema/protocol, swapping only the instruction-level commitment cue is sufficient to flip the model between `<tool_call>` and `no_tool`."
    )
    lines.append(
        "- Strong main-text statement: the user-conditioned route enters through `L20H5`, is routed by `L21H1/L21H12`, and is written out by `MLP27`."
    )
    lines.append(
        "- Strong main-text statement: `L16H4 -> MLP17 -> L23H6` is a competing no-tool route that can suppress the tool-writing path."
    )
    lines.append("")
    lines.append("## What Must Stay Weak")
    lines.append("")
    lines.append(
        "- `L20H5` should be described as a user-conditioned late ingress point, not yet as a pure action-demand reader."
    )
    lines.append(
        "- `L16H4` should be described as a no-tool-biased user-side reader, not yet as a pure ordinary-answer prior."
    )
    lines.append(
        "- `MLP27` should be described as a primary late writer, not a proven unique bottleneck."
    )
    lines.append("")
    lines.append("## What Should Not Be Claimed")
    lines.append("")
    lines.append("- Do not claim a unified mode switcher.")
    lines.append("- Do not claim a generic arbitration zone.")
    lines.append("- Do not center the final story on the early `L2H14` branch.")
    lines.append("")
    lines.append("## Artifact Index")
    lines.append("")
    lines.append("- `query_decision_chain/query_decision_report.md`")
    lines.append("- `instruction_commitment/instruction_commitment_report.md`")
    lines.append("- `instruction_lead/instruction_lead_report.md`")
    lines.append("- `late_writer_backup/late_writer_backup_report.md`")
    lines.append("- `mechanism_audit/component_evidence_table.csv`")
    lines.append("- `mechanism_audit/writing_boundary.md`")
    if (run_root / "final_head_attention_audit" / "head_final_audit_summary.json").exists():
        lines.append("- `final_head_attention_audit/head_final_audit_report.md`")
        lines.append("- `final_head_attention_audit/head_final_audit_summary.json`")
    lines.append("- `final_mechanism_evidence/final_component_evidence_table.csv`")
    lines.append("- `final_mechanism_evidence/final_edge_evidence_table.csv`")
    lines.append("- `final_mechanism_evidence/final_claim_tree.json`")
    lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(str(out_path))


if __name__ == "__main__":
    main()
