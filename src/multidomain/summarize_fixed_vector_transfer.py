#!/usr/bin/env python3
"""Summarize one fixed source vector's held-out cross-domain interventions.

This is deliberately a result-only postprocessor.  It reads the existing
bidirectional intervention CSVs and their companion logit-gap JSON files,
then writes one wide CSV row so a source vector can be reported across several
target domains without re-running a model.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import torch


def parse_metric_spec(value: str) -> tuple[str, Path]:
    domain, separator, raw_path = value.partition("=")
    if not separator or not domain or not raw_path:
        raise argparse.ArgumentTypeError("Expected DOMAIN=/absolute/or/relative/logit_gap_metrics.json")
    return domain.upper(), Path(raw_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a one-row summary for a fixed cross-domain vector transfer.")
    parser.add_argument("--source-domain", required=True)
    parser.add_argument("--source-bundle", type=Path, required=True)
    parser.add_argument("--metric", action="append", type=parse_metric_spec, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def alpha_one_row(path: Path, *, required_field: str) -> dict[str, str]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if abs(float(row["alpha"]) - 1.0) < 1e-9:
                if required_field not in row:
                    raise KeyError(f"{path} lacks {required_field}")
                return row
    raise KeyError(f"{path} lacks alpha=1.0")


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected an object in {path}")
    return payload


def format_percent(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def main() -> None:
    args = parse_args()
    output_root = args.output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite output root: {output_root}")
    bundle = torch.load(args.source_bundle, map_location="cpu", weights_only=False)
    mu = bundle.get("mean_diff")
    if not isinstance(mu, torch.Tensor):
        mu = torch.tensor(mu)
    if not bool(torch.isfinite(mu).all()):
        raise ValueError("Source vector contains non-finite values")
    vector_layer = int(bundle.get("patch_layer", bundle.get("layer", -1)))
    train_ids = bundle.get("sample_ids_train", [])
    metrics = dict(args.metric)
    if len(metrics) != len(args.metric):
        raise ValueError("Each target domain may be supplied only once")

    row: dict[str, Any] = {
        "source_domain": args.source_domain.upper(),
        "source_bundle": str(args.source_bundle.resolve()),
        "source_train_pairs": len(train_ids),
        "source_vector_dimension": int(mu.numel()),
        "source_vector_l2_norm": float(mu.float().norm().item()),
        "layer": vector_layer,
        "hook_kind": "pre" if "patch_layer" in bundle else "post",
        "position": "last_prompt_token",
        "alpha": 1.0,
    }
    markdown_rows: list[dict[str, Any]] = []
    for domain in sorted(metrics):
        metric_path = metrics[domain].resolve()
        payload = load_json(metric_path)
        add_row = alpha_one_row(Path(str(payload["add_csv"])), required_field="tool_call_top1_rate")
        remove_row = alpha_one_row(Path(str(payload["remove_csv"])), required_field="remaining_tool_call_top1_rate")
        prefix = domain.lower()
        values = {
            "n": int(payload["n_pairs"]),
            "baseline_clean_top1": float(payload["clean_top1_rate"]),
            "baseline_corrupt_top1": float(payload["corrupt_top1_rate"]),
            "add_top1": float(add_row["tool_call_top1_rate"]),
            "strict_flip": float(add_row["strict_flip_rate"]),
            "remove_remaining_top1": float(remove_row["remaining_tool_call_top1_rate"]),
            "strict_drop": float(remove_row["strict_drop_rate"]),
            "suff": float(payload["suff"]),
            "necc": float(payload["necc"]),
        }
        row.update({f"{prefix}_{name}": value for name, value in values.items()})
        markdown_rows.append({"domain": domain, **values})

    output_root.mkdir(parents=True, exist_ok=False)
    csv_path = output_root / "one_row_fixed_vector_transfer.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    json_path = output_root / "one_row_fixed_vector_transfer.json"
    json_path.write_text(json.dumps(row, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# D1 Fixed Vector Transfer — One-row Result",
        "",
        f"- Source: `{row['source_domain']}` train mean-difference vector (`n={row['source_train_pairs']}`, L2 norm `{row['source_vector_l2_norm']:.6f}`).",
        f"- Intervention: `L{row['layer']} hook_resid_{row['hook_kind']}`, final prompt position, raw vector, `alpha=1.0`.",
        "- `Suff.` and `Necc.` use the pre-specified normalization by each target domain's clean–corrupt tool-call logit gap.",
        "",
        "| target | n | clean / corrupt top-1 | +vector top-1 / flip | −vector remaining / drop | Suff. | Necc. |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for item in markdown_rows:
        lines.append(
            "| {domain} | {n} | {clean} / {corrupt} | {add} / {flip} | {remain} / {drop} | {suff:.3f} | {necc:.3f} |".format(
                domain=item["domain"],
                n=item["n"],
                clean=format_percent(item["baseline_clean_top1"]),
                corrupt=format_percent(item["baseline_corrupt_top1"]),
                add=format_percent(item["add_top1"]),
                flip=format_percent(item["strict_flip"]),
                remain=format_percent(item["remove_remaining_top1"]),
                drop=format_percent(item["strict_drop"]),
                suff=item["suff"],
                necc=item["necc"],
            )
        )
    (output_root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(row, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
