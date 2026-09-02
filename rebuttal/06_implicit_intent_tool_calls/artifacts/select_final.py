#!/usr/bin/env python3
"""Select a Qwen3-8B removal arm from an independently frozen screen."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from release_paths import DEFAULT_QWEN3_8B_REMOVAL_ROOT, FROZEN_SCREENED_CANDIDATES


DOMAINS = ("D1", "D3", "D4", "D5")
PATTERNS = ("P1", "P2", "P3", "P4", "P5")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=FROZEN_SCREENED_CANDIDATES)
    parser.add_argument("--output", type=Path, default=DEFAULT_QWEN3_8B_REMOVAL_ROOT / "selected_arm.jsonl")
    parser.add_argument("--per-cell", type=int, default=10)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists: {args.output}; pass --overwrite to replace it")
    rows = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line.strip()]
    by_cell: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        if row.get("baseline_is_tool_call_top1"):
            by_cell[(str(row["domain"]), str(row["pattern"]))].append(row)

    final: list[dict[str, object]] = []
    print(f"{'domain':6}" + "".join(f"{pattern:>18}" for pattern in PATTERNS) + "   total")
    for domain in DOMAINS:
        counts = [len(by_cell.get((domain, pattern), [])[: args.per_cell]) for pattern in PATTERNS]
        available = [len(by_cell.get((domain, pattern), [])) for pattern in PATTERNS]
        cells = [f"{count}/{args.per_cell}({avail} avail)" for count, avail in zip(counts, available)]
        print(f"{domain:6}" + "".join(f"{cell:>18}" for cell in cells) + f"   {sum(counts)}")
        for pattern in PATTERNS:
            final.extend(by_cell.get((domain, pattern), [])[: args.per_cell])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in final:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"\nTotal removal-arm items: {len(final)}")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
