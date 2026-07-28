#!/usr/bin/env python3
"""Summarize raw and target-norm-aligned cross-domain amplitude sweeps.

For each target domain, this records the behavioral intervention metrics and
the paper's clean--corrupt logit-gap normalization at every requested alpha.
It is result-only: model outputs and vector bundles are read, never changed.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import torch


def parse_domain_path(value: str) -> tuple[str, Path]:
    domain, separator, raw_path = value.partition("=")
    if not separator or not domain or not raw_path:
        raise argparse.ArgumentTypeError("Expected DOMAIN=/path/to/artifact")
    return domain.upper(), Path(raw_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Make a provenance-rich amplitude-sweep table.")
    parser.add_argument("--source-bundle", type=Path, required=True)
    parser.add_argument("--target-native-bundle", action="append", type=parse_domain_path, required=True)
    parser.add_argument("--baseline-metrics", action="append", type=parse_domain_path, required=True)
    parser.add_argument("--raw-sweep", action="append", type=parse_domain_path, required=True)
    parser.add_argument("--aligned-bundle", action="append", type=parse_domain_path, required=True)
    parser.add_argument("--aligned-sweep", action="append", type=parse_domain_path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def unique_specs(specs: list[tuple[str, Path]], label: str) -> dict[str, Path]:
    result = dict(specs)
    if len(result) != len(specs):
        raise ValueError(f"Duplicate domain supplied for {label}")
    return result


def load_bundle_vector(path: Path) -> torch.Tensor:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a dictionary bundle: {path}")
    vector = payload.get("mean_diff")
    if not isinstance(vector, torch.Tensor):
        vector = torch.as_tensor(vector)
    vector = vector.detach().cpu().float()
    if vector.ndim != 1 or vector.numel() == 0 or not bool(torch.isfinite(vector).all()):
        raise ValueError(f"Invalid mean_diff in {path}")
    return vector


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return payload


def read_rows(path: Path) -> dict[float, dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    result: dict[float, dict[str, str]] = {}
    for row in rows:
        alpha = float(row["alpha"])
        if alpha in result:
            raise ValueError(f"Duplicate alpha={alpha} in {path}")
        result[alpha] = row
    if not result:
        raise ValueError(f"No rows found in {path}")
    return result


def pct(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def main() -> None:
    args = parse_args()
    output_root = args.output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite output root: {output_root}")

    native_paths = unique_specs(args.target_native_bundle, "target-native-bundle")
    baseline_paths = unique_specs(args.baseline_metrics, "baseline-metrics")
    raw_paths = unique_specs(args.raw_sweep, "raw-sweep")
    aligned_bundle_paths = unique_specs(args.aligned_bundle, "aligned-bundle")
    aligned_paths = unique_specs(args.aligned_sweep, "aligned-sweep")
    expected_domains = set(native_paths)
    for label, mapping in [
        ("baseline-metrics", baseline_paths),
        ("raw-sweep", raw_paths),
        ("aligned-bundle", aligned_bundle_paths),
        ("aligned-sweep", aligned_paths),
    ]:
        if set(mapping) != expected_domains:
            raise ValueError(f"Domains for {label} differ from target-native-bundle: {sorted(mapping)} vs {sorted(expected_domains)}")

    source_path = args.source_bundle.resolve()
    source_norm = float(load_bundle_vector(source_path).norm().item())
    rows: list[dict[str, Any]] = []
    for domain in sorted(expected_domains):
        target_norm = float(load_bundle_vector(native_paths[domain].resolve()).norm().item())
        baseline = read_json(baseline_paths[domain].resolve())
        clean_logit = float(baseline["clean_mean_tool_logit"])
        corrupt_logit = float(baseline["corrupt_mean_tool_logit"])
        gap = clean_logit - corrupt_logit
        if gap <= 0.0:
            raise ValueError(f"Non-positive clean--corrupt logit gap for {domain}: {gap}")

        conditions = [
            ("raw", raw_paths[domain].resolve(), source_norm),
            ("target_norm_aligned", aligned_paths[domain].resolve(), float(load_bundle_vector(aligned_bundle_paths[domain].resolve()).norm().item())),
        ]
        for condition, sweep_root, vector_norm in conditions:
            add_rows = read_rows(sweep_root / "alpha_sweep_add.csv")
            remove_rows = read_rows(sweep_root / "alpha_sweep_remove.csv")
            if set(add_rows) != set(remove_rows):
                raise ValueError(f"Add/remove alphas differ in {sweep_root}")
            for alpha in sorted(add_rows):
                add = add_rows[alpha]
                remove = remove_rows[alpha]
                if int(add["n"]) != int(remove["n"]):
                    raise ValueError(f"Add/remove n differs at {domain} {condition} alpha={alpha}")
                add_logit = float(add["mean_tool_call_logit"])
                remove_logit = float(remove["mean_tool_call_logit"])
                rows.append(
                    {
                        "target_domain": domain,
                        "condition": condition,
                        "alpha": alpha,
                        "n": int(add["n"]),
                        "source_l2_norm": source_norm,
                        "target_native_l2_norm": target_norm,
                        "vector_l2_norm_before_alpha": vector_norm,
                        "effective_l2_norm": vector_norm * alpha,
                        "effective_norm_over_target_native": vector_norm * alpha / target_norm,
                        "baseline_clean_mean_tool_logit": clean_logit,
                        "baseline_corrupt_mean_tool_logit": corrupt_logit,
                        "add_tool_call_top1_rate": float(add["tool_call_top1_rate"]),
                        "add_strict_flip_rate": float(add["strict_flip_rate"]),
                        "add_mean_tool_call_logit": add_logit,
                        "sufficiency_normalized_logit_gap": (add_logit - corrupt_logit) / gap,
                        "remove_remaining_tool_call_top1_rate": float(remove["remaining_tool_call_top1_rate"]),
                        "remove_strict_drop_rate": float(remove["strict_drop_rate"]),
                        "remove_mean_tool_call_logit": remove_logit,
                        "necessity_normalized_logit_gap": (clean_logit - remove_logit) / gap,
                    }
                )

    output_root.mkdir(parents=True, exist_ok=False)
    csv_path = output_root / "amplitude_sweep.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metadata = {
        "source_bundle": str(source_path),
        "source_l2_norm": source_norm,
        "target_native_bundles": {domain: str(path.resolve()) for domain, path in native_paths.items()},
        "baseline_metrics": {domain: str(path.resolve()) for domain, path in baseline_paths.items()},
        "raw_sweeps": {domain: str(path.resolve()) for domain, path in raw_paths.items()},
        "aligned_bundles": {domain: str(path.resolve()) for domain, path in aligned_bundle_paths.items()},
        "aligned_sweeps": {domain: str(path.resolve()) for domain, path in aligned_paths.items()},
        "normalization": "suff=(add_logit-corrupt_logit)/(clean_logit-corrupt_logit); necc=(clean_logit-remove_logit)/(clean_logit-corrupt_logit)",
    }
    (output_root / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Cross-domain Amplitude Sweep",
        "",
        "D1 direction is tested raw and after matching its L2 norm to each target domain's native L24 vector.",
        "`Suff.` and `Necc.` are normalized by that target domain's clean--corrupt tool-call logit gap.",
        "",
        "| target | condition | α | effective L2 / target L2 | + top-1 / flip | Suff. | − remaining / drop | Necc. |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {target_domain} | {condition} | {alpha:.1f} | {effective_l2_norm:.2f} / {target_native_l2_norm:.2f} ({effective_norm_over_target_native:.2f}×) | {add} / {flip} | {suff:.3f} | {remain} / {drop} | {necc:.3f} |".format(
                **row,
                add=pct(row["add_tool_call_top1_rate"]),
                flip=pct(row["add_strict_flip_rate"]),
                suff=row["sufficiency_normalized_logit_gap"],
                remain=pct(row["remove_remaining_tool_call_top1_rate"]),
                drop=pct(row["remove_strict_drop_rate"]),
                necc=row["necessity_normalized_logit_gap"],
            )
        )
    (output_root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(csv_path)


if __name__ == "__main__":
    main()
