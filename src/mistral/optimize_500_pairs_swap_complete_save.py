#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


USER_MARKER = "<|im_start|>user\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Swap failing complete pairs with save pairs inside each corrupt-verb group.")
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("./results/Mistral-Small-3.2-24B-Instruct-2506/datasets"),
    )
    parser.add_argument(
        "--eval-root",
        type=Path,
        default=Path("./results/Mistral-Small-3.2-24B-Instruct-2506/datasets_mistral_vibe_eval"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("./results/Mistral-Small-3.2-24B-Instruct-2506/datasets_v2"),
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_manifest(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def read_eval_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def extract_verb(text: str) -> str:
    tail = text.split(USER_MARKER, 1)[1]
    return tail.split(None, 1)[0]


def replace_verb(text: str, new_verb: str) -> str:
    prefix, tail = text.split(USER_MARKER, 1)
    tail = re.sub(r"^\S+", new_verb.capitalize(), tail, count=1)
    return prefix + USER_MARKER + tail


def main() -> None:
    args = parse_args()
    ensure_dir(args.output_root)

    manifest_rows = read_manifest(args.source_root / "pair_manifest.jsonl")
    eval_rows = read_eval_rows(args.eval_root / "pair_decisions.csv")
    eval_by_pair = {int(row["sample_id"].split("_")[1]): row for row in eval_rows}
    manifest_by_pair = {int(row["pair_id"]): row for row in manifest_rows}

    complete_fail_by_corrupt: dict[str, list[int]] = defaultdict(list)
    save_success_by_corrupt: dict[str, list[int]] = defaultdict(list)

    for pair_id, row in eval_by_pair.items():
        manifest = manifest_by_pair[pair_id]
        clean = manifest["assigned_clean_candidate"]
        corrupt = manifest["assigned_corrupt_candidate"]
        clean_ok = str(row["clean_is_tool_call_top1"]).lower() == "true"
        corrupt_ok = str(row["corrupt_is_tool_call_top1"]).lower() == "true"
        if clean == "complete" and (not clean_ok) and (not corrupt_ok):
            complete_fail_by_corrupt[corrupt].append(pair_id)
        if clean == "save" and clean_ok and (not corrupt_ok):
            save_success_by_corrupt[corrupt].append(pair_id)

    swaps: list[tuple[int, int]] = []
    for corrupt in sorted(complete_fail_by_corrupt):
        fails = complete_fail_by_corrupt[corrupt]
        saves = save_success_by_corrupt[corrupt]
        n = min(len(fails), len(saves))
        for i in range(n):
            swaps.append((fails[i], saves[i]))

    updated_rows = [dict(row) for row in manifest_rows]
    row_index_by_pair = {int(row["pair_id"]): idx for idx, row in enumerate(updated_rows)}

    for complete_pair, save_pair in swaps:
        c_idx = row_index_by_pair[complete_pair]
        s_idx = row_index_by_pair[save_pair]
        c_row = updated_rows[c_idx]
        s_row = updated_rows[s_idx]
        c_row["assigned_clean_candidate"] = "save"
        s_row["assigned_clean_candidate"] = "complete"

    for row in updated_rows:
        pair_id = int(row["pair_id"])
        clean_src = (args.source_root / row["clean_filename"]).read_text(encoding="utf-8")
        corrupt_src = (args.source_root / row["corrupt_filename"]).read_text(encoding="utf-8")
        clean_text = replace_verb(clean_src, str(row["assigned_clean_candidate"]))
        corrupt_text = replace_verb(corrupt_src, str(row["assigned_corrupt_candidate"]))
        (args.output_root / row["clean_filename"]).write_text(clean_text, encoding="utf-8")
        (args.output_root / row["corrupt_filename"]).write_text(corrupt_text, encoding="utf-8")

    with (args.output_root / "pair_manifest.jsonl").open("w", encoding="utf-8") as handle:
        for row in updated_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "n_pairs": len(updated_rows),
        "n_swaps": len(swaps),
        "swaps": [{"complete_pair_id": c, "save_pair_id": s} for c, s in swaps],
        "selection_rule": "Swap failing complete pairs with successful save pairs within the same corrupt-verb bucket.",
    }
    (args.output_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
