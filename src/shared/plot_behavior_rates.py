#!/usr/bin/env python3
"""Build the paper's first-token behavior-rate figure from current-run files.

This script intentionally consumes raw screening/decision outputs rather than
selected mechanism subsets.  A subset is normally constructed by retaining
clean-only behavior, so plotting only its manifest would create a tautological
100%/0% figure.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


MODEL_ORDER = {
    "Qwen3-1.7B": 0,
    "Qwen3-4B": 1,
    "Qwen3-8B": 2,
    "Qwen3-14B": 3,
    "Mistral-3.2-24B": 4,
    "Devstral-2-24B": 5,
    "Granite-3.3-8B": 6,
    "Qwen3.5-9B": 7,
}


def named_path(raw: str) -> tuple[str, Path]:
    if "=" not in raw:
        raise argparse.ArgumentTypeError("Use MODEL=PATH")
    label, raw_path = raw.split("=", 1)
    label = label.strip()
    path = Path(raw_path).expanduser()
    if not label or not raw_path:
        raise argparse.ArgumentTypeError("MODEL and PATH must both be non-empty")
    return label, path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot first-token tool-call rates from fresh behavior outputs.")
    parser.add_argument(
        "--qwen-screen-csv",
        action="append",
        type=Path,
        default=[],
        help="CSV from paper_repro.screen_candidates; may be supplied more than once.",
    )
    parser.add_argument(
        "--paired-csv",
        action="append",
        type=named_path,
        default=[],
        metavar="MODEL=PATH",
        help="CSV with clean_is_tool_call_top1 and corrupt_is_tool_call_top1 columns.",
    )
    parser.add_argument(
        "--long-csv",
        action="append",
        type=named_path,
        default=[],
        metavar="MODEL=PATH",
        help="Long-form CSV with pool={clean,corrupt} and is_tool_call_top1 columns.",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def as_bool(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def rate(values: list[bool]) -> float:
    if not values:
        raise ValueError("Cannot compute a rate from zero rows")
    return float(sum(values) / len(values))


def qwen_rows(path: Path) -> list[dict[str, object]]:
    rows = read_csv(path)
    if not rows:
        raise ValueError(f"No screening rows in {path}")
    fields = set(rows[0])
    if "pool" not in fields:
        raise ValueError(f"{path} is not a candidate-screen CSV: missing pool column")
    model_labels = sorted(field[: -len("_is_tool_call_top1")] for field in fields if field.endswith("_is_tool_call_top1"))
    if not model_labels:
        raise ValueError(f"{path} has no *_is_tool_call_top1 columns")
    output: list[dict[str, object]] = []
    for label in model_labels:
        clean = [as_bool(row[f"{label}_is_tool_call_top1"]) for row in rows if row.get("pool") == "clean"]
        corrupt = [as_bool(row[f"{label}_is_tool_call_top1"]) for row in rows if row.get("pool") == "corrupt"]
        output.append(
            {
                "model": label,
                "clean_rate": rate(clean),
                "corrupt_rate": rate(corrupt),
                "clean_n": len(clean),
                "corrupt_n": len(corrupt),
                "source_kind": "qwen_candidate_screen",
                "source_path": str(path.resolve()),
            }
        )
    return output


def paired_row(label: str, path: Path) -> dict[str, object]:
    rows = read_csv(path)
    if not rows:
        raise ValueError(f"No paired behavior rows in {path}")
    required = {"clean_is_tool_call_top1", "corrupt_is_tool_call_top1"}
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"{path} lacks required paired behavior columns: {sorted(missing)}")
    clean = [as_bool(row["clean_is_tool_call_top1"]) for row in rows]
    corrupt = [as_bool(row["corrupt_is_tool_call_top1"]) for row in rows]
    return {
        "model": label,
        "clean_rate": rate(clean),
        "corrupt_rate": rate(corrupt),
        "clean_n": len(clean),
        "corrupt_n": len(corrupt),
        "source_kind": "paired_behavior_scan",
        "source_path": str(path.resolve()),
    }


def long_row(label: str, path: Path) -> dict[str, object]:
    rows = read_csv(path)
    if not rows:
        raise ValueError(f"No long-form behavior rows in {path}")
    required = {"pool", "is_tool_call_top1"}
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"{path} lacks required long-form behavior columns: {sorted(missing)}")
    clean = [as_bool(row["is_tool_call_top1"]) for row in rows if str(row["pool"]).lower() == "clean"]
    corrupt = [as_bool(row["is_tool_call_top1"]) for row in rows if str(row["pool"]).lower() == "corrupt"]
    return {
        "model": label,
        "clean_rate": rate(clean),
        "corrupt_rate": rate(corrupt),
        "clean_n": len(clean),
        "corrupt_n": len(corrupt),
        "source_kind": "long_behavior_screen",
        "source_path": str(path.resolve()),
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def render(rows: list[dict[str, object]], output_root: Path) -> None:
    labels = [str(row["model"]) for row in rows]
    execution = [float(row["clean_rate"]) for row in rows]
    analysis = [float(row["corrupt_rate"]) for row in rows]
    x = np.arange(len(rows), dtype=np.float64)
    width = 0.36
    fig, ax = plt.subplots(figsize=(max(8.0, len(rows) * 1.25), 5.0))
    ax.bar(x - width / 2, execution, width, label="Execution verbs", color="#2878B5")
    ax.bar(x + width / 2, analysis, width, label="Analysis verbs", color="#D62728")
    ax.set_ylabel("First-token <tool_call> rate")
    ax.set_ylim(0.0, 1.05)
    ax.set_yticks(np.linspace(0.0, 1.0, 6), [f"{value:.0%}" for value in np.linspace(0.0, 1.0, 6)])
    ax.set_xticks(x, labels, rotation=25, ha="right")
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.18), ncol=2, frameon=False)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(output_root / f"figure1_behavior_rates.{suffix}", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for path in args.qwen_screen_csv:
        rows.extend(qwen_rows(path.resolve()))
    rows.extend(paired_row(label, path.resolve()) for label, path in args.paired_csv)
    rows.extend(long_row(label, path.resolve()) for label, path in args.long_csv)
    if not rows:
        raise ValueError("Supply at least one behavior input")
    labels = [str(row["model"]) for row in rows]
    duplicate_labels = sorted({label for label in labels if labels.count(label) > 1})
    if duplicate_labels:
        raise ValueError(f"Duplicate model labels across behavior inputs: {duplicate_labels}")
    rows.sort(key=lambda row: (MODEL_ORDER.get(str(row["model"]), 10_000), str(row["model"])))
    write_csv(output_root / "behavior_rates.csv", rows)
    (output_root / "behavior_rates.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_root / "behavior_rates.md").write_text(
        "\n".join(
            [
                "# First-token Behavior Rates",
                "",
                "| Model | Execution rate | Analysis rate | n (execution / analysis) | Source |",
                "|---|---:|---:|---:|---|",
                *[
                    f"| {row['model']} | {float(row['clean_rate']):.2%} | {float(row['corrupt_rate']):.2%} | "
                    f"{int(row['clean_n'])} / {int(row['corrupt_n'])} | {row['source_kind']} |"
                    for row in rows
                ],
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    render(rows, output_root)
    print(f"wrote {output_root / 'behavior_rates.csv'}")


if __name__ == "__main__":
    main()
