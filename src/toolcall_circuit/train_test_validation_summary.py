#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

if __package__ in {None, ""}:
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: object, digits: int = 3) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return str(value)


def build_rows(train_rows: Dict[str, Dict[str, Any]], test_rows: Dict[str, Dict[str, Any]], label_key: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for key in sorted(set(train_rows) | set(test_rows)):
        tr = train_rows.get(key, {})
        te = test_rows.get(key, {})
        out.append(
            {
                "group": key,
                "label": te.get(label_key) or tr.get(label_key) or key,
                "train_promote_suff": tr.get("promote_suff_ratio_median"),
                "test_promote_suff": te.get("promote_suff_ratio_median"),
                "delta_promote_suff": (
                    float(te["promote_suff_ratio_median"]) - float(tr["promote_suff_ratio_median"])
                    if tr.get("promote_suff_ratio_median") is not None and te.get("promote_suff_ratio_median") is not None
                    else None
                ),
                "train_suppress_suff": tr.get("suppress_suff_ratio_median"),
                "test_suppress_suff": te.get("suppress_suff_ratio_median"),
                "delta_suppress_suff": (
                    float(te["suppress_suff_ratio_median"]) - float(tr["suppress_suff_ratio_median"])
                    if tr.get("suppress_suff_ratio_median") is not None and te.get("suppress_suff_ratio_median") is not None
                    else None
                ),
                "train_promote_top1": tr.get("promote_tool_top1_rate"),
                "test_promote_top1": te.get("promote_tool_top1_rate"),
                "delta_promote_top1": (
                    float(te["promote_tool_top1_rate"]) - float(tr["promote_tool_top1_rate"])
                    if tr.get("promote_tool_top1_rate") is not None and te.get("promote_tool_top1_rate") is not None
                    else None
                ),
                "train_suppress_top1": tr.get("suppress_no_tool_top1_rate"),
                "test_suppress_top1": te.get("suppress_no_tool_top1_rate"),
                "delta_suppress_top1": (
                    float(te["suppress_no_tool_top1_rate"]) - float(tr["suppress_no_tool_top1_rate"])
                    if tr.get("suppress_no_tool_top1_rate") is not None and te.get("suppress_no_tool_top1_rate") is not None
                    else None
                ),
            }
        )
    return out


def build_markdown(struct_rows: List[Dict[str, Any]], func_rows: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    lines.append("# EAP-IG Train/Test Validation Summary")
    lines.append("")
    lines.append("## Structural Groups")
    lines.append("")
    for row in struct_rows:
        lines.append(
            f"- `{row['group']}`: promote suff `{fmt(row['train_promote_suff'])} -> {fmt(row['test_promote_suff'])}` "
            f"(Δ `{fmt(row['delta_promote_suff'])}`), "
            f"suppress suff `{fmt(row['train_suppress_suff'])} -> {fmt(row['test_suppress_suff'])}` "
            f"(Δ `{fmt(row['delta_suppress_suff'])}`), "
            f"promote top1 `{fmt(row['train_promote_top1'])} -> {fmt(row['test_promote_top1'])}` "
            f"(Δ `{fmt(row['delta_promote_top1'])}`), "
            f"suppress top1 `{fmt(row['train_suppress_top1'])} -> {fmt(row['test_suppress_top1'])}` "
            f"(Δ `{fmt(row['delta_suppress_top1'])}`)."
        )
    lines.append("")
    lines.append("## Functional Groups")
    lines.append("")
    for row in func_rows:
        lines.append(
            f"- `{row['label']}`: promote suff `{fmt(row['train_promote_suff'])} -> {fmt(row['test_promote_suff'])}` "
            f"(Δ `{fmt(row['delta_promote_suff'])}`), "
            f"suppress suff `{fmt(row['train_suppress_suff'])} -> {fmt(row['test_suppress_suff'])}` "
            f"(Δ `{fmt(row['delta_suppress_suff'])}`), "
            f"promote top1 `{fmt(row['train_promote_top1'])} -> {fmt(row['test_promote_top1'])}` "
            f"(Δ `{fmt(row['delta_promote_top1'])}`), "
            f"suppress top1 `{fmt(row['train_suppress_top1'])} -> {fmt(row['test_suppress_top1'])}` "
            f"(Δ `{fmt(row['delta_suppress_top1'])}`)."
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build train/test comparison summary for EAP-IG validation.")
    parser.add_argument("--train-signed", type=str, required=True)
    parser.add_argument("--test-signed", type=str, required=True)
    parser.add_argument("--train-functional", type=str, required=True)
    parser.add_argument("--test-functional", type=str, required=True)
    parser.add_argument("--output-root", type=str, required=True)
    args = parser.parse_args()

    output_root = Path(args.output_root).resolve()
    train_signed = read_json(Path(args.train_signed).resolve())
    test_signed = read_json(Path(args.test_signed).resolve())
    train_functional = read_json(Path(args.train_functional).resolve())
    test_functional = read_json(Path(args.test_functional).resolve())

    train_signed_rows = {str(r["group"]): r for r in train_signed.get("summary_rows", [])}
    test_signed_rows = {str(r["group"]): r for r in test_signed.get("summary_rows", [])}
    train_functional_rows = {str(r["group"]): r for r in train_functional.get("summary_rows", [])}
    test_functional_rows = {str(r["group"]): r for r in test_functional.get("summary_rows", [])}

    structural_rows = build_rows(train_signed_rows, test_signed_rows, "group_label")
    functional_rows = build_rows(train_functional_rows, test_functional_rows, "functional_label")

    payload = {
        "structural_rows": structural_rows,
        "functional_rows": functional_rows,
        "artifacts": {
            "summary_json": str(output_root / "train_test_validation_summary.json"),
            "structural_csv": str(output_root / "train_test_structural_summary.csv"),
            "functional_csv": str(output_root / "train_test_functional_summary.csv"),
            "summary_md": str(output_root / "TRAIN_TEST_VALIDATION_SUMMARY.md"),
        },
    }
    write_json(output_root / "train_test_validation_summary.json", payload)
    write_csv(output_root / "train_test_structural_summary.csv", structural_rows)
    write_csv(output_root / "train_test_functional_summary.csv", functional_rows)
    (output_root / "TRAIN_TEST_VALIDATION_SUMMARY.md").write_text(
        build_markdown(structural_rows, functional_rows),
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
