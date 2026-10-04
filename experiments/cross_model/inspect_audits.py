#!/usr/bin/env python3
"""Compare compact cross-family Transcoder summaries without hiding protocol differences."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from mi4tc.io import load_json, write_json  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect cross-model Transcoder audit summaries.")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    rows = []
    for path in sorted((REPO_ROOT / "experiments").glob("**/summary.json")):
        loaded = load_json(path)
        payloads = loaded if isinstance(loaded, list) else [loaded]
        for payload in payloads:
            heldout = payload.get("paper_K_heldout") or {}
            topk = payload.get("table3_topk") or {}
            native = "S_over_E" in payload
            rows.append(
                {
                    "artifact": str(path.relative_to(REPO_ROOT)),
                    "model": payload.get("model_label", payload.get("model_key")),
                    "alignment_policy": payload.get("alignment_policy") or ("native_se_dominance" if native else "unspecified_in_source_summary"),
                    "train_pairs": payload.get("train_pairs", payload.get("full_train_pairs_before_alignment")),
                    "heldout_pairs": payload.get("heldout_pairs", payload.get("full_heldout_pairs_before_alignment")),
                    "reference_layer": payload.get("reference_layer"),
                    "available_layers": payload.get("available_transcoder_layers"),
                    "all_feature_ratio": heldout.get("K_corrupt_clean_ratio"),
                    "top20_ratio": topk.get("heldout_frozen_ratio_corrupt_over_clean"),
                    "top20_status": topk.get("status"),
                    "native_S_over_E": payload.get("S_over_E"),
                }
            )
    causal_artifacts = []
    for path in sorted((REPO_ROOT / "experiments").glob("**/causal_intervention_summary.json")):
        loaded = load_json(path)
        records = loaded if isinstance(loaded, list) else [loaded]
        causal_artifacts.append(
            {
                "artifact": str(path.relative_to(REPO_ROOT)),
                "groups": [record.get("group") for record in records if isinstance(record, dict)],
                "n_by_group": {
                    record.get("group"): record.get("n")
                    for record in records
                    if isinstance(record, dict) and record.get("group") is not None
                },
            }
        )
    policies = sorted({str(row["alignment_policy"]) for row in rows})
    report = {
        "protocols_present": policies,
        "rows": rows,
        "causal_artifacts": causal_artifacts,
        "why_not_promoted": [
            "Available Transcoder layers do not always include the historical reference layer.",
            "Strict and allow-unequal token-alignment protocols are both present.",
            "All-feature K and top-20 K are different estimands.",
            "Native S/E dominance is retained as a separate estimand and is not pooled with K ratios.",
        ],
    }
    if args.output:
        write_json(args.output, report)
    else:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
