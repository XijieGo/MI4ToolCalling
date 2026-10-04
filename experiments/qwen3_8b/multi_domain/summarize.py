#!/usr/bin/env python3
"""Make a compact index of scaffold, schema, and transfer evidence."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from mi4tc.io import parse_csv_rows, write_json  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize the scaffold and schema tables.")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    scaffold = here / "scaffold_prior/scaffold_components_table.md"
    ladder = here / "scaffold_prior/request_ladder_table.md"
    schema = here / "schema_identity/table_metrics.csv"
    reversal = here / "schema_identity/affordance_reversal_2x2.csv"
    report = {
        "scaffold_table": str(scaffold.relative_to(REPO_ROOT)),
        "request_ladder_table": str(ladder.relative_to(REPO_ROOT)),
        "schema_rows": len(parse_csv_rows(schema)),
        "affordance_reversal_rows": len(parse_csv_rows(reversal)),
        "scope": "The scaffold tables are Qwen3-8B evidence; schema identity tables include model-specific conditions.",
        "interpretation": "Use this evidence for a selective scaffold-conditioned prior, not an unconditional call default.",
    }
    if args.output:
        write_json(args.output, report)
    else:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
