"""Merge an independently selected arm and its causal results into one JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from release_paths import DEFAULT_QWEN3_8B_REMOVAL_ROOT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, default=DEFAULT_QWEN3_8B_REMOVAL_ROOT / "selected_arm.jsonl")
    parser.add_argument("--causal", type=Path, default=DEFAULT_QWEN3_8B_REMOVAL_ROOT / "removal_results.jsonl")
    parser.add_argument("--output", type=Path, default=DEFAULT_QWEN3_8B_REMOVAL_ROOT / "merged_removal_arm.jsonl")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> dict[str, dict[str, object]]:
    return {
        str(row["item_id"]): row
        for row in (json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    }


def main() -> None:
    args = parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists: {args.output}; pass --overwrite to replace it")
    selected = read_jsonl(args.selection)
    causal = read_jsonl(args.causal)
    missing = sorted(set(selected) - set(causal))
    if missing:
        raise ValueError(f"Causal results are missing {len(missing)} selected item(s), e.g. {missing[:3]}")
    merged = []
    for item_id, base in selected.items():
        row = dict(base)
        row["causal"] = causal[item_id]["conditions"]
        merged.append(row)
    merged.sort(key=lambda row: (str(row["domain"]), str(row["pattern"]), str(row["item_id"])))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in merged:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {args.output} ({len(merged)} items)")


if __name__ == "__main__":
    main()
