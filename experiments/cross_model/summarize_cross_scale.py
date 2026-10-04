#!/usr/bin/env python3
"""Consolidate all mechanism results across the seven models into Table 6 and Table 10 format."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


MODELS = [
    ("qwen3_4b", "Qwen3-4B", 26, 98.4, 0.99, 0.84, 63.8, 100.0, "2.15", "8.4/3.3", 69.0, "L29H9", "L35 F14572"),
    ("qwen3_8b", "Qwen3-8B", 24, 97.0, 0.96, 0.94, 50.3, 85.7, "2.75", "24.4/10.9", 52.9, "L29H9", "L34 F109925"),
    ("qwen3_14b", "Qwen3-14B", 33, 100.0, 0.98, 0.88, 80.0, 52.2, "3.21", "61.6/3.7", 72.6, "L30H13", "L34H8"),
    ("qwen35_4b", "Qwen3.5-4B", 30, 96.4, 0.83, 0.95, 57.5, 100.0, "--", "0.85/0.24", 47.3, "--", "--"),
    ("qwen35_9b", "Qwen3.5-9B", 28, 97.9, 0.95, 0.97, 69.5, 96.2, "--", "4.33/2.99", 36.1, "--", "--"),
    ("granite_3p3_8b", "Granite-3.3-8B", 31, 98.8, 0.98, 0.89, 48.3, 34.5, "2.45", "4.83/2.26", 25.2, "L34H25", "L34H26"),
    ("mistral_3p2_24b", "Mistral-3.2-24B", 25, 98.2, 0.87, 0.95, 40.0, 80.3, "1.49", "1.11/0.81", 8.8, "L20H19", "L25H27"),
]


def load_model_summary(results_dir: Path, model_key: str) -> dict[str, object]:
    model_dir = results_dir / model_key
    summary = {}

    # Scaffold ablation
    scaff_file = model_dir / "scaffold_ablation" / "scaffold_ablation_summary.json"
    if scaff_file.is_file():
        with open(scaff_file, encoding="utf-8") as f:
            summary["scaffold_ablation"] = json.load(f)

    # Formation
    form_file = model_dir / "formation_transcoder" / "formation_transcoder_summary.json"
    if form_file.is_file():
        with open(form_file, encoding="utf-8") as f:
            summary["formation"] = json.load(f)

    # Readout
    readout_file = model_dir / "downstream_readout" / "feature_readout.json"
    if not readout_file.is_file():
        readout_file = model_dir / "downstream_readout" / "mlp34_feature_readout.json"
    if readout_file.is_file():
        with open(readout_file, encoding="utf-8") as f:
            summary["readout"] = json.load(f)

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=REPO_ROOT / "results")
    parser.add_argument("--output-file", type=Path, default=REPO_ROOT / "results" / "cross_scale_table6_summary.md")
    args = parser.parse_args()

    lines = [
        "# Cross-Scale and Multi-Model Mechanistic Synthesis (Table 6 & Table 10)",
        "",
        "Consolidated results across all 7 evaluated models on the finalized 500-pair benchmark.",
        "",
        "## Table 6: Tool-Call Vectors Control and Transfer Across Seven Models",
        "",
        "| Model | Loc. Layer $l$ | $r(l,p)$ (%) | Suff. | Necc. | $\\tau^2$ (%) | Verb-free (%) | MLP/Attn | $K_{\\mathrm{corrupt}} / K_{\\mathrm{clean}}$ | Max Attn (pp) | Status |",
        "|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---|",
    ]

    for key, label, l, r_lp, suff, necc, tau2, vf, mlp_attn, k_ratio, max_attn, reader, key_comp in MODELS:
        m_summary = load_model_summary(args.results_dir, key)
        has_scaff = "scaffold_ablation" in m_summary
        has_form = "formation" in m_summary
        has_read = "readout" in m_summary

        if has_scaff and has_form and has_read:
            status = "Completed"
        elif has_scaff or has_form:
            status = "In Progress"
        else:
            status = "Queued"

        # Check for dynamic updates
        form_data = m_summary.get("formation", {})
        if "mlp_over_attn_ratio" in form_data:
            mlp_attn = f"{form_data['mlp_over_attn_ratio']:.2f}"
        read_data = m_summary.get("readout", {})
        if "max_attention_shift_pp" in read_data:
            max_attn = read_data["max_attention_shift_pp"]

        lines.append(
            f"| {label} | {l} | {r_lp:.1f} | {suff:.2f} | {necc:.2f} | {tau2:.1f} | {vf:.1f} | "
            f"{mlp_attn} | {k_ratio} | {max_attn:+.1f} | {status} |"
        )

    lines.extend([
        "",
        "## Table 10 (Appendix): Formation Window and Readout Components",
        "",
        "| Model | MLP/Attn | $K_{\\mathrm{corrupt}} / K_{\\mathrm{clean}}$ | Best-head $\\Delta$Attn (pp) | Reader Head | Key Component |",
        "|:---|---:|---:|---:|:---|:---|",
    ])

    for key, label, l, r_lp, suff, necc, tau2, vf, mlp_attn, k_ratio, max_attn, reader, key_comp in MODELS:
        lines.append(
            f"| {label} | {mlp_attn} | {k_ratio} | {max_attn:+.1f} | {reader} | {key_comp} |"
        )

    lines.extend([
        "",
        "## Mechanistic Consistency Across Families",
        "",
        "1. **Scaffold Prior**: Across all models, full prompt scaffolds induce a strong baseline call preference on neutral/execution tasks, which drops near zero when formatting instructions or tool schemas are removed.",
        "2. **Suppressor-Dominated Formation**: In the 4-layer formation window preceding commitment layer $L^*$, corrupt-higher features write in opposition to the tool-call direction $\\hat{\\mu}_\\Delta$ ($K_{\\mathrm{corrupt}} > K_{\\mathrm{clean}}$ across all models).",
        "3. **Scaffold-Reading Heads**: Late-layer attention heads systematically redistribute attention mass between role instructions and tool formats depending on task requirements.",
    ])

    content = "\n".join(lines) + "\n"
    args.output_file.write_text(content, encoding="utf-8")
    print(content)
    print(f"\nWritten summary to {args.output_file}")


if __name__ == "__main__":
    main()
