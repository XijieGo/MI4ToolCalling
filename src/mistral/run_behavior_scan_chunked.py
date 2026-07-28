#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
from pathlib import Path
from typing import Any


DEFAULT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = Path(
    os.environ.get(
        "MISTRAL_3P2_24B_PATH",
        str(DEFAULT_ROOT / "external" / "models" / "Mistral-Small-3.2-24B-Instruct-2506"),
    )
)
DEFAULT_CONVERTED_ROOT = DEFAULT_ROOT / "results" / "Mistral-Small-3.2-24B-Instruct-2506" / "converted_dataset"
DEFAULT_OUTPUT_ROOT = DEFAULT_ROOT / "results" / "Mistral-Small-3.2-24B-Instruct-2506" / "behavior_scan"
SCRIPT_PATH = DEFAULT_ROOT / "src" / "mistral" / "run_behavior_scan.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Mistral behavior scan in stable chunks and merge outputs.")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--converted-root", type=Path, default=DEFAULT_CONVERTED_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--chunk-size", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--device-map", type=str, default="auto")
    parser.add_argument(
        "--system-prompt-mode",
        type=str,
        choices=("official", "toolcall_short", "vibe"),
        default="vibe",
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def read_csv_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def build_summary(rows: list[dict[str, Any]], tool_token_text: str, tool_token_id: int) -> dict[str, Any]:
    clean_flags = [str(row["clean_is_tool_call_top1"]).lower() == "true" for row in rows]
    corrupt_flags = [str(row["corrupt_is_tool_call_top1"]).lower() == "true" for row in rows]
    clean_rate = mean([1.0 if x else 0.0 for x in clean_flags])
    corrupt_rate = mean([1.0 if x else 0.0 for x in corrupt_flags])

    def top_rows(side: str) -> list[dict[str, Any]]:
        counts: dict[tuple[str, str], int] = {}
        for row in rows:
            key = (str(row[f"{side}_top1_token_id"]), str(row[f"{side}_top1_token_text"]))
            counts[key] = counts.get(key, 0) + 1
        ordered = sorted(counts.items(), key=lambda item: (-item[1], int(item[0][0])))
        return [
            {
                "token_id": int(token_id),
                "count": count,
                "rate": count / max(len(rows), 1),
                "token_text": token_text,
            }
            for (token_id, token_text), count in ordered[:10]
        ]

    summary = {
        "n_pairs": len(rows),
        "tool_calls_token_text": tool_token_text,
        "tool_calls_token_id": int(tool_token_id),
        "clean_tool_call_rate": clean_rate,
        "corrupt_tool_call_rate": corrupt_rate,
        "gap_top1_pp": (clean_rate - corrupt_rate) * 100.0,
        "borderline_pair_count": sum(
            1
            for row in rows
            if float(row["clean_top1_margin"]) < 0.05 or float(row["corrupt_top1_margin"]) < 0.05
        ),
        "borderline_definition": "Pairs where clean or corrupt prompt has top1 probability margin under 0.05.",
        "clean": {
            "mean_tool_call_prob": mean([float(row["clean_tool_token_prob"]) for row in rows]),
            "mean_tool_call_logit": mean([float(row["clean_tool_token_logit"]) for row in rows]),
            "top1_mode_rows": top_rows("clean"),
        },
        "corrupt": {
            "mean_tool_call_prob": mean([float(row["corrupt_tool_token_prob"]) for row in rows]),
            "mean_tool_call_logit": mean([float(row["corrupt_tool_token_logit"]) for row in rows]),
            "top1_mode_rows": top_rows("corrupt"),
        },
        "per_split": {},
    }
    for split in sorted({str(row["split"]) for row in rows}):
        split_rows = [row for row in rows if str(row["split"]) == split]
        summary["per_split"][split] = {
            "n_pairs": len(split_rows),
            "clean_tool_call_rate": mean([1.0 if str(row["clean_is_tool_call_top1"]).lower() == "true" else 0.0 for row in split_rows]),
            "corrupt_tool_call_rate": mean([1.0 if str(row["corrupt_is_tool_call_top1"]).lower() == "true" else 0.0 for row in split_rows]),
        }
    return summary


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    clean_rate = summary["clean_tool_call_rate"] * 100.0
    corrupt_rate = summary["corrupt_tool_call_rate"] * 100.0
    gap = summary["gap_top1_pp"]
    clean_mode = summary["clean"]["top1_mode_rows"][0]["token_text"] if summary["clean"]["top1_mode_rows"] else "N/A"
    corrupt_mode = summary["corrupt"]["top1_mode_rows"][0]["token_text"] if summary["corrupt"]["top1_mode_rows"] else "N/A"
    if gap >= 30.0:
        verdict = "clean/corrupt still show substantial separation on Mistral-Small-3.2-24B."
    elif gap >= 10.0:
        verdict = "clean/corrupt remain partially separated, but the gap is much weaker than a clean transfer."
    else:
        verdict = "clean/corrupt are not meaningfully separated on Mistral-Small-3.2-24B."
    if clean_rate < 50.0 and corrupt_rate < 50.0:
        collapse = "Both sides tend to collapse away from tool calls."
    elif clean_rate >= 50.0 and corrupt_rate >= 50.0:
        collapse = "Both sides tend to collapse into tool calls."
    elif clean_rate < 50.0:
        collapse = "The clean side is the one that collapses first."
    else:
        collapse = "The corrupt side is the one that collapses first."
    if gap >= 30.0:
        next_step = "This dataset is likely worth deeper Mistral-family follow-up."
    elif gap >= 10.0:
        next_step = "This dataset may support limited follow-up, but it is not a clean drop-in for mechanism work."
    else:
        next_step = "This dataset is not a strong candidate for deeper Mistral-specific mechanism experiments without re-filtering."
    text = (
        "# Mistral-Small-3.2-24B Tool-Call Generalization Scan\n\n"
        f"- Pairs evaluated: `{summary['n_pairs']}`\n"
        f"- `[TOOL_CALLS]` token id: `{summary['tool_calls_token_id']}`\n"
        f"- Clean tool-call rate: `{clean_rate:.2f}%`\n"
        f"- Corrupt tool-call rate: `{corrupt_rate:.2f}%`\n"
        f"- Clean minus corrupt gap: `{gap:.2f} pp`\n"
        f"- Clean top-1 mode: `{clean_mode}`\n"
        f"- Corrupt top-1 mode: `{corrupt_mode}`\n"
        f"- Borderline pairs: `{summary['borderline_pair_count']}`\n\n"
        f"{verdict}\n\n"
        f"{collapse}\n\n"
        f"{next_step}\n"
    )
    path.write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    canonical_rows = read_jsonl(args.converted_root / "canonical_pairs.jsonl")
    n_pairs = len(canonical_rows)
    chunk_root = args.output_root / "chunks"
    ensure_dir(chunk_root)
    logs_root = args.output_root.parent / "logs"
    ensure_dir(logs_root)
    total_chunks = (n_pairs + args.chunk_size - 1) // args.chunk_size

    for chunk_idx, start in enumerate(range(0, n_pairs, args.chunk_size), start=1):
        chunk_output = chunk_root / f"chunk_{start:04d}_{min(start + args.chunk_size, n_pairs):04d}"
        ensure_dir(chunk_output)
        log_path = logs_root / f"behavior_scan_chunk_{start:04d}.log"
        stop = min(start + args.chunk_size, n_pairs)
        print(
            f"[chunk {chunk_idx}/{total_chunks}] start={start} stop={stop} output={chunk_output}",
            flush=True,
        )
        cmd = [
            "python",
            str(SCRIPT_PATH),
            "--model-path",
            str(args.model_path),
            "--converted-root",
            str(args.converted_root),
            "--output-root",
            str(chunk_output),
            "--batch-size",
            str(args.batch_size),
            "--dtype",
            args.dtype,
            "--device-map",
            args.device_map,
            "--system-prompt-mode",
            args.system_prompt_mode,
            "--start-index",
            str(start),
            "--max-pairs",
            str(min(args.chunk_size, n_pairs - start)),
        ]
        with log_path.open("w", encoding="utf-8") as log_handle:
            completed = subprocess.run(cmd, cwd=str(DEFAULT_ROOT), stdout=log_handle, stderr=subprocess.STDOUT)
        if completed.returncode != 0:
            raise RuntimeError(f"Chunk starting at {start} failed with exit code {completed.returncode}. See {log_path}")
        print(
            f"[chunk {chunk_idx}/{total_chunks}] completed log={log_path}",
            flush=True,
        )

    merged_rows: list[dict[str, Any]] = []
    tool_token_text = None
    tool_token_id = None
    for start in range(0, n_pairs, args.chunk_size):
        chunk_output = chunk_root / f"chunk_{start:04d}_{min(start + args.chunk_size, n_pairs):04d}"
        merged_rows.extend(read_csv_rows(chunk_output / "pair_decisions.csv"))
        summary = json.loads((chunk_output / "aggregate_summary.json").read_text(encoding="utf-8"))
        tool_token_text = summary["tool_calls_token_text"]
        tool_token_id = summary["tool_calls_token_id"]

    if tool_token_text is None or tool_token_id is None:
        raise RuntimeError("No chunk summaries were produced.")

    write_csv(args.output_root / "pair_decisions.csv", merged_rows)
    summary = build_summary(merged_rows, str(tool_token_text), int(tool_token_id))
    (args.output_root / "aggregate_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_markdown(args.output_root / "aggregate_summary.md", summary)
    print(
        f"[merge] completed n_pairs={summary['n_pairs']} clean_rate={summary['clean_tool_call_rate']:.6f} "
        f"corrupt_rate={summary['corrupt_tool_call_rate']:.6f} gap_pp={summary['gap_top1_pp']:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
