import json
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent
screened = {r["item_id"]: r for r in map(json.loads, open(OUT_DIR / "implicit_intent_removal_final.jsonl"))}
causal = {r["item_id"]: r for r in map(json.loads, open(OUT_DIR / "implicit_intent_removal_results.jsonl"))}

merged = []
for item_id, base in screened.items():
    row = dict(base)
    row["causal"] = causal[item_id]["conditions"]
    merged.append(row)

merged.sort(key=lambda r: (r["domain"], r["pattern"], r["item_id"]))

dest = Path("./outputs/implicit_intent_transfer")
dest.mkdir(parents=True, exist_ok=True)
with (dest / "removal_arm_161.jsonl").open("w", encoding="utf-8") as f:
    for r in merged:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"Wrote {dest / 'removal_arm_161.jsonl'} ({len(merged)} items)")
