#!/usr/bin/env python3
"""Select the final removal-arm set: up to 10 baseline-positive items per
(domain, pattern) cell, in the original deterministic construction order.
Reports the achieved count honestly -- some cells may fall short of 10.
"""
import json
from pathlib import Path
from collections import defaultdict

OUT_DIR = Path(__file__).resolve().parent
rows = [json.loads(l) for l in open(OUT_DIR / "implicit_intent_oversampled_600_screened.jsonl")]

by_cell = defaultdict(list)
for r in rows:
    if r["baseline_is_tool_call_top1"]:
        by_cell[(r["domain"], r["pattern"])].append(r)

DOMAINS = ("D1", "D3", "D4", "D5")
PATTERNS = ("P1", "P2", "P3", "P4", "P5")

final = []
table = {}
for d in DOMAINS:
    table[d] = {}
    for p in PATTERNS:
        cell = by_cell.get((d, p), [])
        take = cell[:10]
        table[d][p] = f"{len(take)}/10 (of {len(cell)} available, {30} attempted)"
        final.extend(take)

print(f"{'domain':6}" + "".join(f"{p:>18}" for p in PATTERNS) + "   total")
for d in DOMAINS:
    counts = [len(by_cell.get((d, p), [])[:10]) for p in PATTERNS]
    avail = [len(by_cell.get((d, p), [])) for p in PATTERNS]
    cells = [f"{c}/10({a} avail)" for c, a in zip(counts, avail)]
    print(f"{d:6}" + "".join(f"{c:>18}" for c in cells) + f"   {sum(counts)}")

print(f"\nTotal removal-arm items: {len(final)}")

out_path = OUT_DIR / "implicit_intent_removal_final.jsonl"
with out_path.open("w", encoding="utf-8") as f:
    for r in final:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"Wrote {out_path}")
