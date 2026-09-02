"""Summarize results produced by the standalone Qwen3-8B removal runner."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from release_paths import DEFAULT_QWEN3_8B_REMOVAL_ROOT

ALPHAS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0)
DOMAINS = ("D1", "D3", "D4", "D5")
PATTERNS = ("P1", "P2", "P3", "P4", "P5")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_QWEN3_8B_REMOVAL_ROOT / "removal_results.jsonl")
    return parser.parse_args()


def flip_rate(rows: list[dict[str, object]], key: str) -> tuple[int, int]:
    n = len(rows)
    flips = sum(1 for r in rows if not r["conditions"][key]["is_tool_call_top1"])
    return flips, n


def main() -> None:
    args = parse_args()
    rows = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line.strip()]
    print("=== Per-domain, mean_diff removal ===")
    print("domain  n  " + "  ".join(f"a={alpha}" for alpha in ALPHAS))
    by_domain: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_domain[str(row["domain"])].append(row)
    for domain in DOMAINS:
        domain_rows = by_domain[domain]
        cells = []
        for alpha in ALPHAS:
            flips, n = flip_rate(domain_rows, "mean_diff_a" + str(alpha))
            cells.append(f"{flips}/{n}")
        print(f"{domain:6}  {len(domain_rows):3} " + "  ".join(cells))

    print(f"\n=== Combined (N={len(rows)}), mean_diff vs random ===")
    for direction in ("mean_diff", "random"):
        cells = []
        for alpha in ALPHAS:
            if alpha == 0.0 and direction == "random":
                continue
            flips, n = flip_rate(rows, f"{direction}_a{alpha}")
            cells.append(f"a={alpha}: {flips}/{n} ({100 * flips / n:.1f}%)")
        print(direction, cells)

    print("\n=== Per-domain x pattern, mean_diff a=1.5 ===")
    by_cell: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_cell[(str(row["domain"]), str(row["pattern"]))].append(row)
    print(f"{'domain':6}" + "".join(f"{pattern:>12}" for pattern in PATTERNS))
    for domain in DOMAINS:
        cells = []
        for pattern in PATTERNS:
            cell_rows = by_cell.get((domain, pattern), [])
            if not cell_rows:
                cells.append("n/a")
                continue
            flips, n = flip_rate(cell_rows, "mean_diff_a1.5")
            cells.append(f"{flips}/{n}")
        print(f"{domain:6}" + "".join(f"{cell:>12}" for cell in cells))


if __name__ == "__main__":
    main()
