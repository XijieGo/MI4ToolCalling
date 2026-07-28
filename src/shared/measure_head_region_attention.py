#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from multiscale_common import DEFAULT_DATASET_ROOT, load_model_and_tokenizer, load_sample_pairs
from run_core_generalization import analyze_top_head_regions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure clean/corrupt system/schema attention for specified heads.")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--size-label", type=str, required=True)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--eval-split", type=str, default="test")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--heads", type=str, required=True, help="Comma-separated L:H entries, e.g. 20:14,21:1")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def parse_heads(text: str) -> list[dict[str, int]]:
    rows: list[dict[str, int]] = []
    for chunk in str(text).split(","):
        piece = chunk.strip()
        if not piece:
            continue
        layer_str, head_str = piece.split(":")
        rows.append({"layer": int(layer_str), "head": int(head_str)})
    if not rows:
        raise ValueError("No heads parsed from --heads.")
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    model, tokenizer, _tool_token_id = load_model_and_tokenizer(model_path=args.model_path, device=args.device)
    pairs = load_sample_pairs(model, dataset_root=args.dataset_root, split=args.eval_split, max_pairs=0)
    top_rows = parse_heads(args.heads)
    rows = analyze_top_head_regions(
        model,
        tokenizer,
        pairs,
        top_rows=top_rows,
        batch_size=args.batch_size,
    )
    rows.sort(key=lambda row: float(row["delta_system_attention"]), reverse=True)
    write_csv(args.output_root / "head_region_attention.csv", rows)
    summary = {
        "size_label": args.size_label,
        "model_path": str(args.model_path),
        "eval_split": args.eval_split,
        "n_pairs": len(pairs),
        "heads": top_rows,
        "rows": rows,
    }
    (args.output_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
