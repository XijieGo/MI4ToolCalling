#!/usr/bin/env python3
"""Make a deterministic, source-first reserve pool for the 8B screen.

The primary-stage target is 900 behavior-valid *sources*, not exhaustive
measurement of every Cartesian verb combination.  This helper takes a fixed
number of source records and a fixed number of evenly rotated verb pairs per
source.  It performs no model or semantic selection.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .common import read_jsonl, stable_rank, write_json, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--domain", required=True)
    parser.add_argument("--source-limit", type=int, required=True)
    parser.add_argument("--pairs-per-source", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def choose(rows: list[dict[str, Any]], *, source_limit: int, pairs_per_source: int, seed: int) -> list[dict[str, Any]]:
    if source_limit <= 0 or pairs_per_source <= 0:
        raise ValueError("--source-limit and --pairs-per-source must be positive")
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_source[str(row["source_id"])].append(row)
    source_ids = sorted(by_source, key=lambda source_id: (stable_rank(seed, source_id), source_id))[:source_limit]
    selected: list[dict[str, Any]] = []
    for source_id in source_ids:
        options = sorted(
            by_source[source_id],
            key=lambda row: (row["clean_verb"], row["corrupt_verb"], row["candidate_id"]),
        )
        if not options:
            continue
        # Rotate the lexicographically fixed verb grid by source hash.  Across
        # sources this gives each pair comparable exposure without looking at
        # content or model behavior.
        offset = int(stable_rank(seed, f"pair-offset:{source_id}")[:16], 16) % len(options)
        count = min(pairs_per_source, len(options))
        selected.extend(options[(offset + index) % len(options)] for index in range(count))
    return selected


def main() -> None:
    args = parse_args()
    rows = list(read_jsonl(args.candidates.resolve()))
    if any(row.get("domain") != args.domain for row in rows):
        raise ValueError(f"Candidate file contains a domain other than {args.domain}")
    selected = choose(
        rows,
        source_limit=args.source_limit,
        pairs_per_source=args.pairs_per_source,
        seed=args.seed,
    )
    if not selected:
        raise RuntimeError("No reserve-pool candidates selected")
    write_jsonl(args.output.resolve(), (row for row in selected))
    write_json(
        args.output.with_suffix(args.output.suffix + ".manifest.json"),
        {
            "domain": args.domain,
            "seed": args.seed,
            "input_candidate_count": len(rows),
            "source_limit": args.source_limit,
            "pairs_per_source": args.pairs_per_source,
            "selected_source_count": len({row["source_id"] for row in selected}),
            "selected_candidate_count": len(selected),
            "verb_pair_counts": dict(Counter(f"{row['clean_verb']}/{row['corrupt_verb']}" for row in selected)),
        },
    )
    print(
        f"{args.domain}: {len({row['source_id'] for row in selected})} sources, "
        f"{len(selected)} deterministic primary-screen candidates"
    )


if __name__ == "__main__":
    main()
