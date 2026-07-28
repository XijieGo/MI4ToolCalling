import json
from pathlib import Path
from collections import defaultdict

OUT_DIR = Path(__file__).resolve().parent
rows = [json.loads(l) for l in open(OUT_DIR / "implicit_intent_removal_results.jsonl")]

ALPHAS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0)
DOMAINS = ("D1", "D3", "D4", "D5")

def flip_rate(rows, key):
    n = len(rows)
    flips = sum(1 for r in rows if not r["conditions"][key]["is_tool_call_top1"])
    return flips, n

print("=== Per-domain, mean_diff removal ===")
header = "domain  n  " + "  ".join(f"a={a}" for a in ALPHAS)
print(header)
by_domain = defaultdict(list)
for r in rows:
    by_domain[r["domain"]].append(r)
for d in DOMAINS:
    rs = by_domain[d]
    cells = []
    for a in ALPHAS:
        key = "mean_diff_a" + str(a)
        flips, n = flip_rate(rs, key)
        cells.append(f"{flips}/{n}")
    print(f"{d:6}  {len(rs):3} " + "  ".join(cells))

print("\n=== Combined (N=161), mean_diff vs random ===")
for direction in ("mean_diff", "random"):
    cells = []
    for a in ALPHAS:
        if a == 0.0 and direction == "random":
            continue
        key = f"{direction}_a{a}"
        flips, n = flip_rate(rows, key)
        cells.append(f"a={a}: {flips}/{n} ({100*flips/n:.1f}%)")
    print(direction, cells)

print("\n=== Per-domain x pattern, mean_diff a=1.5 ===")
by_cell = defaultdict(list)
for r in rows:
    by_cell[(r["domain"], r["pattern"])].append(r)
PATTERNS = ("P1", "P2", "P3", "P4", "P5")
print(f"{'domain':6}" + "".join(f"{p:>12}" for p in PATTERNS))
for d in DOMAINS:
    cells = []
    for p in PATTERNS:
        rs = by_cell.get((d, p), [])
        if not rs:
            cells.append("n/a")
            continue
        flips, n = flip_rate(rs, "mean_diff_a1.5")
        cells.append(f"{flips}/{n}")
    print(f"{d:6}" + "".join(f"{c:>12}" for c in cells))
