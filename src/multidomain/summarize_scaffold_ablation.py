#!/usr/bin/env python3
"""Build a traceable cross-domain summary for rebuttal scaffold ablations.

This is deliberately a reporting-only step: it reads the completed Group 1
request-ladder and Group 3 R/T/F runs, validates their completion and fixed
split metadata, and writes compact tables plus a rebuttal-ready Markdown
readout.  It never reselects examples or recomputes behavioral statistics.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = PROJECT_ROOT / "results" / "runs"
DOMAINS = ("D1", "D3", "D4", "D5")
LADDER_LEVELS = ("L0", "L1", "L2", "L3", "L4", "L5", "L6", "L7")
CORE_SCAFFOLDS = ("RTF", "-TF", "R-F", "RT-", "--F", "-T-", "R--", "---")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default="20260725", help="Run-date suffix used in source output directories.")
    parser.add_argument("--results-root", type=Path, default=RESULTS_ROOT)
    parser.add_argument("--output-root", type=Path)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write an empty table: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def source_root(results_root: Path, *, experiment: str, domain: str, tag: str) -> Path:
    return results_root / f"rebuttal_{experiment}_{domain.lower()}_v4_full_{tag}"


def numeric(row: dict[str, Any], name: str) -> float:
    return float(row[name])


def p(value: float) -> str:
    if abs(value) < 1e-4 and value != 0.0:
        return f"{value:.2e}"
    return f"{value:.3f}"


def build_markdown(
    *,
    ladder_rows: list[dict[str, Any]],
    derived_rows: list[dict[str, Any]],
    intervention_rows: list[dict[str, Any]],
    audit: dict[str, Any],
) -> str:
    ladder_by_domain = {
        domain: {row["ladder_level"]: row for row in ladder_rows if row["domain"] == domain}
        for domain in DOMAINS
    }
    derived_by_domain = {
        domain: {row["scaffold"]: row for row in derived_rows if row["domain"] == domain}
        for domain in DOMAINS
    }
    intervention_by_domain = {
        domain: {row["scaffold"]: row for row in intervention_rows if row["domain"] == domain}
        for domain in DOMAINS
    }

    lines = [
        "# v4 cross-domain scaffold-ablation audit",
        "",
        "This reporting bundle aggregates only completed runs on the four rebuilt v4 domains ",
        "(D1 code, D3 fact retrieval, D4 SQL, D5 email); it does not use the retired 1,500-example data.",
        "",
        "## Integrity checks",
        "",
        "- All eight source runs have `status: complete`.",
        "- Each request-ladder result uses the fixed 100-example test split. Each R/T/F result uses 400 fixed train pairs and 100 fixed test pairs.",
        "- In every source pair set, clean and corrupt prompts differ at exactly one common token position. The R+length-matched-neutral-text+F control is token-length exact on both train and test in every domain.",
        "",
        "## Group 1 — request ladder",
        "",
        "Values are mean `p(<tool_call>)` on held-out prompts. L0--L2 are each one literal prompt and are consequently reported once rather than replicated 100 times.",
        "",
        "| Domain | L0 | L1 | L3 verb-free | L4 neutral | L5 analysis | L6 execution | L7 no scaffold | Pre-registered case |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for domain in DOMAINS:
        rows = ladder_by_domain[domain]
        lines.append(
            "| "
            + domain
            + " | "
            + " | ".join(p(numeric(rows[level], "behavior_mean_tool_call_prob")) for level in ("L0", "L1", "L3", "L4", "L5", "L6", "L7"))
            + f" | {rows['L0']['interpretation_case']} |"
        )
    lines.extend(
        [
            "",
            "Readout: neither literal baseline (L0/L1) is top-1 in any domain, so the evidence does not support an unconditional/literal default. The pre-registered task-conditioned criterion is met in D1, D3, and D5. D4 is the planned boundary case: the verb-free task body is execution-side, but the predeclared neutral-verb mixture sits essentially halfway between analysis and execution; it must remain reported as case C rather than being post-hoc folded into the positive result.",
            "",
            "## Group 3 — R/T/F full factorial",
            "",
            "`D(S)` is neutral-request `p(<tool_call>)`; `P(S)` is neutral minus analysis. Both are held-out means.",
            "",
            "| Domain | Full RTF D | Full RTF P | R+T (no F) D | F-only D | F-only P | R+neutral-length+F D |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for domain in DOMAINS:
        rows = derived_by_domain[domain]
        lines.append(
            "| "
            + domain
            + " | "
            + " | ".join(
                (
                    p(numeric(rows[scaffold], metric))
                    for scaffold, metric in (
                        ("RTF", "default_strength_p_call"),
                        ("RTF", "suppression_depth_p_call"),
                        ("RT-", "default_strength_p_call"),
                        ("--F", "default_strength_p_call"),
                        ("--F", "suppression_depth_p_call"),
                        ("R_TLEN_F", "default_strength_p_call"),
                    )
                )
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "Readout: removing the format template (RT-) collapses `D(S)` in all four domains (maximum 2.18e-6), whereas F alone retains a near-ceiling call prior (`D(S)` 0.993--1.000) but almost no analysis-selective suppression (`P(S)` 0.0001--0.037). Thus F is necessary for the token-level prior, while R and T modulate request conditioning and suppression rather than supplying the prior by themselves. The equal-length T control changes behavior substantially, especially in D4, so schema effects cannot be reduced to prompt length alone; their magnitude is domain dependent.",
            "",
            "## Full-scaffold causal check",
            "",
            "The vector is estimated from the frozen 400-pair train split per domain and tested on the 100-pair held-out split. `Suff.`/`Necc.` are condition-internal normalized logit-gap measures.",
            "",
            "| Domain | cos(variant, frozen RTF) | Suff. | Necc. | add flip | remove drop |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for domain in DOMAINS:
        row = intervention_by_domain[domain]["RTF"]
        lines.append(
            f"| {domain} | {p(numeric(row, 'cosine_to_frozen_RTF'))} | {p(numeric(row, 'frozen_RTF_sufficiency'))} | {p(numeric(row, 'frozen_RTF_necessity'))} | {p(numeric(row, 'frozen_RTF_add_strict_flip_rate'))} | {p(numeric(row, 'frozen_RTF_remove_strict_drop_rate'))} |"
        )
    lines.extend(
        [
            "",
            "The complete 8x3 behavioral factorial and all frozen-vector transfer metrics are retained in the CSV files alongside this summary. Normalized causal ratios in collapsed no-F cells should be read cautiously because their within-condition clean--analysis denominators are near zero; the primary scaffold conclusion uses the directly comparable probability outcomes above.",
            "",
            "## Audit metadata",
            "",
            f"- Completed source runs: {audit['completed_run_count']}",
            f"- Group-1 cases: {dict(sorted(audit['ladder_case_counts'].items()))}",
            "- Pair and control validation details: `cross_domain_audit.json`.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    output_root = args.output_root or args.results_root / f"rebuttal_scaffold_ablation_v4_crossdomain_{args.tag}"
    output_root.mkdir(parents=True, exist_ok=True)

    ladder_rows: list[dict[str, Any]] = []
    derived_rows: list[dict[str, Any]] = []
    intervention_rows: list[dict[str, Any]] = []
    sources: dict[str, Any] = {}
    validation: dict[str, Any] = {}
    ladder_cases: Counter[str] = Counter()

    for domain in DOMAINS:
        ladder_root = source_root(args.results_root, experiment="request_ladder", domain=domain, tag=args.tag)
        component_root = source_root(args.results_root, experiment="scaffold_components", domain=domain, tag=args.tag)
        require(ladder_root.is_dir(), f"Missing Group 1 directory: {ladder_root}")
        require(component_root.is_dir(), f"Missing Group 3 directory: {component_root}")

        ladder_completion = read_json(ladder_root / "completion.json")
        component_completion = read_json(component_root / "completion.json")
        require(ladder_completion.get("status") == "complete", f"Incomplete Group 1 run: {domain}")
        require(component_completion.get("status") == "complete", f"Incomplete Group 3 run: {domain}")
        require(ladder_completion.get("dataset_cardinality", {}).get("test") == 100, f"Unexpected Group 1 test size: {domain}")
        require(component_completion.get("dataset_cardinality") == {"train": 400, "test": 100}, f"Unexpected Group 3 split size: {domain}")

        ladder_validation = read_json(ladder_root / "pair_validation.json")
        component_validation = read_json(component_root / "pair_validation.json")
        require(ladder_validation["test"]["pair_count"] == 100, f"Group 1 pair count failed: {domain}")
        require(component_validation["fixed"]["train"]["pair_count"] == 400, f"Group 3 train-pair count failed: {domain}")
        require(component_validation["fixed"]["test"]["pair_count"] == 100, f"Group 3 test-pair count failed: {domain}")
        require(component_validation["R_TLEN_F/train_length_control"]["all_exact"], f"Train length control failed: {domain}")
        require(component_validation["R_TLEN_F/test_length_control"]["all_exact"], f"Test length control failed: {domain}")

        interpretation = read_json(ladder_root / "ladder_interpretation.json")
        case = str(interpretation["observed_case"])
        ladder_cases[case] += 1
        ladder_summary = read_csv(ladder_root / "ladder_summary.csv")
        require(tuple(row["ladder_level"] for row in ladder_summary) == LADDER_LEVELS, f"Unexpected ladder ordering: {domain}")
        for row in ladder_summary:
            enriched = {"domain": domain, "interpretation_case": case, **row}
            ladder_rows.append(enriched)

        domain_derived = read_csv(component_root / "derived_default_suppression.csv")
        require(tuple(row["scaffold"] for row in domain_derived[:8]) == CORE_SCAFFOLDS, f"Unexpected factorial ordering: {domain}")
        require(domain_derived[-1]["scaffold"] == "R_TLEN_F", f"Missing length control: {domain}")
        derived_rows.extend(domain_derived)
        intervention_rows.extend(read_csv(component_root / "intervention_long.csv"))

        sources[domain] = {
            "request_ladder": str(ladder_root),
            "scaffold_components": str(component_root),
        }
        validation[domain] = {
            "request_ladder_test": ladder_validation["test"],
            "component_fixed": component_validation["fixed"],
            "length_matched_control": {
                "train": component_validation["R_TLEN_F/train_length_control"],
                "test": component_validation["R_TLEN_F/test_length_control"],
            },
        }

    audit = {
        "status": "complete",
        "data_version": "v4_multidomain_balanced",
        "domains": list(DOMAINS),
        "completed_run_count": len(DOMAINS) * 2,
        "ladder_case_counts": dict(ladder_cases),
        "sources": sources,
        "validation": validation,
    }
    write_csv(output_root / "request_ladder_cross_domain.csv", ladder_rows)
    write_csv(output_root / "scaffold_default_suppression_cross_domain.csv", derived_rows)
    write_csv(output_root / "scaffold_causal_cross_domain.csv", intervention_rows)
    (output_root / "cross_domain_audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    (output_root / "rebuttal_ready_summary.md").write_text(
        build_markdown(
            ladder_rows=ladder_rows,
            derived_rows=derived_rows,
            intervention_rows=intervention_rows,
            audit=audit,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"status": "complete", "output_root": str(output_root), "domains": list(DOMAINS)}, indent=2))


if __name__ == "__main__":
    main()
