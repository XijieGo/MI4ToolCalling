#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import torch

import run_mechanism_generalization as mech


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "results" / "Qwen3.5-9B" / "fixed_layer_followup"
DEFAULT_PATCH_SWEEP = PROJECT_ROOT / "results" / "Qwen3.5-9B" / "mechanism_generalization" / "exp_a_state_patch" / "patch_sweep.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run fixed-layer Qwen3.5-9B follow-up metrics on the 500-pair set.")
    parser.add_argument("--model-path", type=Path, default=mech.MODEL_PATH)
    parser.add_argument("--dataset-root", type=Path, default=mech.DATASET_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--patch-sweep", type=Path, default=DEFAULT_PATCH_SWEEP)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=mech.DEFAULT_BATCH_SIZE)
    parser.add_argument("--dtype", type=str, default=mech.DEFAULT_DTYPE)
    parser.add_argument("--max-pairs", type=int, default=0)
    return parser.parse_args()


def read_patch_row(path: Path, layer: int) -> dict[str, str]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if int(row["layer"]) == layer:
                return row
    raise KeyError(f"Layer L{layer} not found in {path}")


def compute_mlp_attn_ratio(mlp_rows: list[dict[str, object]], head_rows: list[dict[str, object]]) -> float:
    mlp_mass = sum(abs(float(row["mlp_delta"])) for row in mlp_rows)
    attn_mass = sum(abs(float(row["head_delta"])) for row in head_rows)
    if attn_mass == 0.0:
        return math.nan
    return float(mlp_mass / attn_mass)


def build_summary_text(
    *,
    layer: int,
    n_pairs: int,
    patch_row: dict[str, str],
    exp_b_summary: dict[str, object],
    exp_c_summary: dict[str, object],
    mlp_attn: float,
) -> str:
    top_mlp = exp_c_summary.get("top_mlp", [])
    top_head = exp_c_summary.get("top_head", [])
    return "\n".join(
        [
            "# Qwen3.5-9B Fixed-Layer Follow-up",
            "",
            f"- layer: `L{layer}`",
            f"- n_pairs: `{n_pairs}`",
            f"- r(l,p): `{100.0 * float(patch_row['tool_call_top1_rate']):.1f}`",
            f"- strict flip: `{100.0 * float(patch_row['strict_flip_rate']):.1f}`",
            f"- clean baseline top1: `{100.0 * float(patch_row['baseline_clean_tool_call_top1_rate']):.1f}`",
            f"- corrupt baseline top1: `{100.0 * float(patch_row['baseline_corrupt_tool_call_top1_rate']):.1f}`",
            f"- suff: `{float(exp_b_summary['suff']):.4f}`",
            f"- necc: `{float(exp_b_summary['necc']):.4f}`",
            f"- vector+ top1: `{100.0 * float(exp_b_summary['vector_plus_tool_call_top1_rate']):.1f}`",
            f"- vector- top1: `{100.0 * float(exp_b_summary['vector_minus_tool_call_top1_rate']):.1f}`",
            f"- MLP/Attn: `{mlp_attn:.4f}`",
            f"- trajectory peak layer: `L{int(exp_c_summary['trajectory_peak_layer'])}`",
            f"- top MLP: `{top_mlp[0]['layer'] if top_mlp else '--'}`",
            f"- top head: `{top_head[0]['head_label'] if top_head else '--'}`",
        ]
    )


def main() -> None:
    args = parse_args()
    layer_root = args.output_root / f"L{args.layer}"
    mech.ensure_dir(layer_root)

    model, tokenizer = mech.load_model(args.model_path, args.dtype)
    pairs = mech.load_pairs(args.dataset_root, tokenizer)
    if args.max_pairs > 0:
        pairs = pairs[: args.max_pairs]

    patch_row = read_patch_row(args.patch_sweep, args.layer)
    mu_delta, u = mech.run_vector_exp(
        model,
        tokenizer,
        pairs,
        batch_size=args.batch_size,
        output_root=layer_root,
        layer_star=args.layer,
    )
    _traj_rows, mlp_rows, head_rows = mech.run_upstream_exp(
        model,
        tokenizer,
        pairs,
        batch_size=args.batch_size,
        output_root=layer_root,
        layer_star=args.layer,
        mu_delta=mu_delta,
        u=u,
    )

    exp_b_summary = mech.read_json(layer_root / "exp_b_tool_call_vector" / "summary.json")
    exp_c_summary = mech.read_json(layer_root / "exp_c_upstream_projection" / "summary.json")
    mlp_attn = compute_mlp_attn_ratio(mlp_rows, head_rows)

    summary_payload = {
        "layer": int(args.layer),
        "n_pairs": len(pairs),
        "r_lp_percent": 100.0 * float(patch_row["tool_call_top1_rate"]),
        "strict_flip_percent": 100.0 * float(patch_row["strict_flip_rate"]),
        "baseline_clean_top1_percent": 100.0 * float(patch_row["baseline_clean_tool_call_top1_rate"]),
        "baseline_corrupt_top1_percent": 100.0 * float(patch_row["baseline_corrupt_tool_call_top1_rate"]),
        "suff": float(exp_b_summary["suff"]),
        "necc": float(exp_b_summary["necc"]),
        "vector_plus_top1_percent": 100.0 * float(exp_b_summary["vector_plus_tool_call_top1_rate"]),
        "vector_minus_top1_percent": 100.0 * float(exp_b_summary["vector_minus_tool_call_top1_rate"]),
        "mlp_attn": mlp_attn,
        "source_patch_sweep": str(args.patch_sweep),
        "source_exp_b": str(layer_root / "exp_b_tool_call_vector" / "summary.json"),
        "source_exp_c": str(layer_root / "exp_c_upstream_projection" / "summary.json"),
        "top_mlp": exp_c_summary.get("top_mlp", []),
        "top_head": exp_c_summary.get("top_head", []),
        "trajectory_peak_layer": int(exp_c_summary["trajectory_peak_layer"]),
    }
    mech.write_json(layer_root / "fixed_layer_summary.json", summary_payload)
    mech.write_text(
        layer_root / "fixed_layer_summary.md",
        build_summary_text(
            layer=args.layer,
            n_pairs=len(pairs),
            patch_row=patch_row,
            exp_b_summary=exp_b_summary,
            exp_c_summary=exp_c_summary,
            mlp_attn=mlp_attn,
        ),
    )

    del model, mu_delta, u
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
