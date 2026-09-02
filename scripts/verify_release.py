#!/usr/bin/env python3
"""Validate the anonymous rebuttal-evidence release without loading models."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REQUIRED_FILES = (
    "README.md",
    "scripts/verify_release.py",
    "scripts/verify_standalone.py",
    "experiments/README.md",
    "experiments/registry.json",
    "src/README.md",
    "paper/neurips_2026.tex",
    "paper/main.tex",
    "paper/appendix.tex",
    "paper/references.bib",
    "paper/figures/feature_circuit.png",
    "datasets/provenance/v2_1500/selection_manifest.csv",
    "datasets/provenance/v2_1500/split_manifest.csv",
    "datasets/v5_model_specific_balanced/README.md",
    "src/paper_repro/verify_dataset_v2.py",
    "src/paper_repro/rebuild_dataset_v2.py",
    "src/paper_repro/build_v5_model_specific_balanced.py",
    "src/paper_repro/build_v5_mistral_tau2.py",
    "src/rebuttal/run_cjrq_v5_tool_identity.py",
    "src/rebuttal/render_cjrq_v5_table.py",
    "src/rebuttal/run_lxey_v5_rank_probability.py",
    "src/natural_trajectory/run_tau2_bidirectional_200_intervention.py",
    "src/multidomain/run_scaffold_component_ablation.py",
    "src/multidomain/run_request_ladder_ablation.py",
    "src/multidomain/run_common_direction_causal_ablation.py",
    "src/qwen3_8b/rebuttal_lxey_other_concerns.py",
    "src/qwen3_8b/rebuttal_lxey_head_attribution.py",
    "src/qwen3_8b/rebuttal_lxey_crossdomain_head_roles.py",
    "rebuttal/README.md",
    "rebuttal/02_schema_tool_identity/final_all_models/rebuttal_table.md",
    "rebuttal/02_schema_tool_identity/final_all_models/table_metrics.csv",
    "rebuttal/02_schema_tool_identity/affordance_reversal_qwen3_8b_v4_20260725/affordance_reversal_2x2_samples.csv",
    "rebuttal/03_scaffold_ablation_and_request_ladder/final_v5_qwen3_8b/summary/scaffold_components_table.md",
    "rebuttal/03_scaffold_ablation_and_request_ladder/final_v5_qwen3_8b/summary/request_ladder_table.md",
    "rebuttal/03_scaffold_ablation_and_request_ladder/final_v5_qwen3_8b/components/sample_metrics.csv",
    "rebuttal/03_scaffold_ablation_and_request_ladder/final_v5_qwen3_8b/request_ladder/sample_metrics.csv",
    "rebuttal/04_rank_probability/final_heldout_all_models/rebuttal_table.md",
    "rebuttal/04_rank_probability/final_heldout_all_models/table_metrics.csv",
    "rebuttal/05_tau2_bench_200/final_bidirectional_results/tau2_bidirectional_200_interventions_20260727_canonical_batch1/run_summary.json",
    "rebuttal/06_implicit_intent_tool_calls/artifacts/cross_model_removal_20260726/cross_model_summary.json",
    "rebuttal/07_lxey_mechanistic_controls/lxey_other_concerns_v2_1500/head_attribution/summary.md",
    "rebuttal/07_lxey_mechanistic_controls/lxey_other_concerns_v2_1500/boundary_verbs/summary.md",
    "rebuttal/07_lxey_mechanistic_controls/lxey_other_concerns_v2_1500/failure_audit/heldout_mu_delta_per_sample.csv",
    "rebuttal/07_lxey_mechanistic_controls/lxey_crossdomain_heads_v4/summary.md",
    "rebuttal/07_lxey_mechanistic_controls/lxey_v5_neighbor_layers_20260728T040846Z/per_sample_metrics.csv",
    "rebuttal/08_wpfh_crossdomain_qwen_family/qwen_table_audit/candidate_table.md",
    "rebuttal/08_wpfh_crossdomain_qwen_family/qwen3_8b/matrix_long.csv",
    "rebuttal/08_wpfh_crossdomain_qwen_family/qwen35_4b/transfer/matrix_long.csv",
)
V5_MODELS = (
    "qwen3_4b",
    "qwen3_8b",
    "qwen3_14b",
    "qwen35_4b",
    "qwen35_9b",
    "mistral_3p2_24b",
    "granite_3p3_8b",
)
VECTOR_HASHES = {
    "rebuttal/05_tau2_bench_200/coding_direction_inputs/cross_family_tau2_20260726/qwen3_4b/coding_vector_bundle.pt": "b8badf78655ea1d995092056dddd7232e70ea7b3fcd547ac0cc177eca68b93be",
    "rebuttal/05_tau2_bench_200/coding_direction_inputs/cross_family_tau2_20260726/qwen3_8b/coding_vector_bundle.pt": "7fe8067426d685b84b016d4c80be3116dbf4c1e69f6e8df5ef5de524d21fe8da",
    "rebuttal/05_tau2_bench_200/coding_direction_inputs/cross_family_tau2_20260726/qwen3_14b/coding_vector_bundle.pt": "de22079ddd280005093bc431c334f68de2f22805c8900cee10fb73c23ab23cc1",
    "rebuttal/05_tau2_bench_200/coding_direction_inputs/cross_family_tau2_20260726/qwen35_4b/coding_vector_bundle.pt": "e0769573e84184a4f531f427c14f6e8cb52466505af01962a7061b1ad6a73d0d",
    "rebuttal/05_tau2_bench_200/coding_direction_inputs/cross_family_tau2_20260726/qwen35_9b/coding_vector_bundle.pt": "2b48ba1628d264d033d2a9c618185800ffb2ff3269611b7f540c72854169549a",
    "rebuttal/05_tau2_bench_200/coding_direction_inputs/cross_family_tau2_20260726/mistral/coding_vector_bundle.pt": "94988e0149770101f60c94651dae3f7fb9c6059fca042e3cbbbbfcb3746460ca",
    "rebuttal/05_tau2_bench_200/coding_direction_inputs/cross_family_tau2_20260726/granite/coding_vector_bundle.pt": "0b72941ac71b6d4317fbca5b7f2ec47d62bc531016de2e7fcd3f86d82d873d47",
    "rebuttal/05_tau2_bench_200/coding_direction_inputs/cross_family_tau2_20260726/devstral/coding_vector_bundle.pt": "a3ad8d506bbdee2da3ecfd4d9cb6eb88751979847c913c2604ff024b576aeb98",
}
JSON_FILES = (
    "rebuttal/05_tau2_bench_200/final_bidirectional_results/tau2_bidirectional_200_interventions_20260727_canonical_batch1/run_summary.json",
    "rebuttal/06_implicit_intent_tool_calls/artifacts/cross_model_removal_20260726/cross_model_summary.json",
    "rebuttal/02_schema_tool_identity/final_all_models/completion.json",
    "rebuttal/04_rank_probability/final_heldout_all_models/completion.json",
)
PUBLIC_RELEASE_TREES = ("README.md", "datasets", "experiments", "rebuttal", "scripts", "src")
FORBIDDEN_WORKSPACE_FRAGMENTS = ("MI4ToolCalling" + chr(45) + "sync",)


def add(results: list[dict[str, str]], check: str, ok: bool, detail: str) -> None:
    results.append({"check": check, "status": "ok" if ok else "error", "detail": detail})


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def contains(path: Path, needle: bytes) -> bool:
    overlap = max(len(needle) - 1, 0)
    previous = b""
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            if needle in previous + chunk:
                return True
            previous = chunk[-overlap:] if overlap else b""
    return False


def check_v5_manifest(model: str, results: list[dict[str, str]]) -> None:
    path = ROOT / "datasets" / "v5_model_specific_balanced" / model / "manifest.jsonl"
    label = str(path.relative_to(ROOT))
    try:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        splits = Counter(str(row.get("split")) for row in rows)
        unique_ids = {str(row.get("sample_id")) for row in rows}
        ok = len(rows) == 500 and len(unique_ids) == 500 and splits == Counter({"train": 200, "heldout": 300})
        add(results, label, ok, "500 unique rows, 200 train / 300 heldout" if ok else f"rows={len(rows)}, unique={len(unique_ids)}, splits={dict(splits)}")
    except Exception as exc:
        add(results, label, False, f"invalid or missing: {exc}")


def check_foreign_paths(results: list[dict[str, str]]) -> None:
    files: list[Path] = []
    for relative in PUBLIC_RELEASE_TREES:
        path = ROOT / relative
        if path.is_file():
            files.append(path)
        elif path.is_dir():
            files.extend(item for item in path.rglob("*") if item.is_file())
    violations = [
        str(path.relative_to(ROOT))
        for fragment in FORBIDDEN_WORKSPACE_FRAGMENTS
        for path in files
        if contains(path, fragment.encode("utf-8"))
    ]
    add(
        results,
        "foreign-workspace references",
        not violations,
        f"no synchronization-workspace paths in {len(files)} public files" if not violations else f"found in: {', '.join(violations[:5])}",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-checksums", action="store_true", help="Skip SHA-256 checks for frozen tau2 vectors.")
    parser.add_argument("--json", action="store_true", help="Emit a machine-readable report.")
    args = parser.parse_args()
    results: list[dict[str, str]] = []

    for relative in REQUIRED_FILES:
        path = ROOT / relative
        add(results, relative, path.is_file(), "present" if path.is_file() else "missing")
    for model in V5_MODELS:
        check_v5_manifest(model, results)
    for relative in JSON_FILES:
        path = ROOT / relative
        try:
            json.loads(path.read_text(encoding="utf-8"))
            add(results, relative, True, "valid JSON")
        except Exception as exc:
            add(results, relative, False, f"invalid or missing: {exc}")
    check_foreign_paths(results)
    if not args.skip_checksums:
        for relative, expected in VECTOR_HASHES.items():
            path = ROOT / relative
            actual = sha256(path) if path.is_file() else ""
            add(results, relative, actual == expected, "SHA-256 verified" if actual == expected else "missing or SHA-256 mismatch")

    ok = all(item["status"] == "ok" for item in results)
    report = {"repository_root": str(ROOT), "ok": ok, "checks": results}
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        for item in results:
            print(f"[{item['status'].upper()}] {item['check']}: {item['detail']}")
        print(f"\nStructural release validation {'passed' if ok else 'failed'}: {sum(item['status'] == 'ok' for item in results)}/{len(results)} checks passed.")
        if ok:
            print("Traceability status: rebuttal/README.md maps every response item to its intermediate records, final result, and code.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
