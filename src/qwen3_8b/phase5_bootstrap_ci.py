#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from task_attention_path_analysis import ensure_dir, write_csv, write_text


SEED = 42
PHASE5_ROOT = Path("./results/8B/phase5_neurips/exp_a_bootstrap_ci")
QUERY_SHIFT_ROOT = Path("./results/8B/query_shift_source")
PHASE4_ROOT = Path("./results/8B/phase4_reviewer")


@dataclass(frozen=True)
class MetricSpec:
    metric: str
    source_file: Path
    definition: str
    extractor: Callable[[], pd.DataFrame]


def bootstrap_ci(values: np.ndarray, *, n_boot: int, seed: int) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, values.size, size=(n_boot, values.size))
    boots = values[idx].mean(axis=1)
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return float(lo), float(hi)


def build_single_metric_df(
    df: pd.DataFrame,
    *,
    sample_col: str,
    value_col: str,
) -> pd.DataFrame:
    out = df[[sample_col, value_col]].copy()
    out = out.rename(columns={sample_col: "sample_id", value_col: "value"})
    out["value"] = out["value"].astype(np.float64)
    return out.sort_values("sample_id").reset_index(drop=True)


def build_paired_flip_df(
    baseline_df: pd.DataFrame,
    intervention_df: pd.DataFrame,
    *,
    sample_col: str,
    baseline_col: str,
    intervention_col: str,
) -> pd.DataFrame:
    merged = (
        baseline_df[[sample_col, baseline_col]]
        .rename(columns={baseline_col: "baseline_value"})
        .merge(
            intervention_df[[sample_col, intervention_col]].rename(columns={intervention_col: "intervention_value"}),
            on=sample_col,
            how="inner",
        )
        .rename(columns={sample_col: "sample_id"})
        .sort_values("sample_id")
        .reset_index(drop=True)
    )
    merged["value"] = (
        (merged["baseline_value"].astype(int) == 0) & (merged["intervention_value"].astype(int) == 1)
    ).astype(np.float64)
    return merged[["sample_id", "value"]]


def make_metric_specs() -> list[MetricSpec]:
    patch_sweep_per_sample = pd.read_csv(QUERY_SHIFT_ROOT / "patch_sweep_per_sample.csv")
    query_specific = pd.read_csv(PHASE4_ROOT / "exp_h_query_specific" / "query_specific_per_sample.csv")
    forcing = pd.read_csv(PHASE4_ROOT / "exp_i_h9_forcing" / "forcing_per_sample.csv")
    schema_control = pd.read_csv(PHASE4_ROOT / "exp_l_schema_control" / "schema_control_per_sample.csv")
    cross_position = pd.read_csv(PHASE4_ROOT / "exp_m_cross_position_control" / "cross_position_per_sample.csv")
    forcing_dep = pd.read_csv(PHASE4_ROOT / "exp_i_h9_forcing" / "forcing_l33_dependency.csv")

    return [
        MetricSpec(
            metric="l24_state_patch_flip_rate",
            source_file=QUERY_SHIFT_ROOT / "patch_sweep_per_sample.csv",
            definition="Phase 3 headline number; uses patch_layer==24 flip_from_corrupt",
            extractor=lambda: build_single_metric_df(
                patch_sweep_per_sample[
                    (patch_sweep_per_sample["condition"] == "corrupt")
                    & (patch_sweep_per_sample["patch_layer"].astype(str) == "24")
                ].copy(),
                sample_col="sample_id",
                value_col="flip_from_corrupt",
            ),
        ),
        MetricSpec(
            metric="h9_query_patch_tool_call_rate",
            source_file=PHASE4_ROOT / "exp_h_query_specific" / "query_specific_per_sample.csv",
            definition="Phase 4 reported 19.33%; intervention condition h9_q_replace_exact top-1 tool-call rate",
            extractor=lambda: build_single_metric_df(
                query_specific[query_specific["condition"] == "h9_q_replace_exact"].copy(),
                sample_col="sample_id",
                value_col="is_tool_call_top1",
            ),
        ),
        MetricSpec(
            metric="h9_query_patch_strict_flip_rate",
            source_file=PHASE4_ROOT / "exp_h_query_specific" / "query_specific_per_sample.csv",
            definition="Strict paired flip rate against baseline_corrupt for h9_q_replace_exact",
            extractor=lambda: build_paired_flip_df(
                query_specific[query_specific["condition"] == "baseline_corrupt"].copy(),
                query_specific[query_specific["condition"] == "h9_q_replace_exact"].copy(),
                sample_col="sample_id",
                baseline_col="is_tool_call_top1",
                intervention_col="is_tool_call_top1",
            ),
        ),
        MetricSpec(
            metric="h9_pattern_forcing_tool_call_rate",
            source_file=PHASE4_ROOT / "exp_i_h9_forcing" / "forcing_per_sample.csv",
            definition="Phase 4 reported 19.13%; intervention condition h9_pattern_replace_exact top-1 tool-call rate",
            extractor=lambda: build_single_metric_df(
                forcing[forcing["condition"] == "h9_pattern_replace_exact"].copy(),
                sample_col="sample_id",
                value_col="is_tool_call_top1",
            ),
        ),
        MetricSpec(
            metric="h9_pattern_forcing_strict_flip_rate",
            source_file=PHASE4_ROOT / "exp_i_h9_forcing" / "forcing_per_sample.csv",
            definition="Strict paired flip rate against baseline_corrupt for h9_pattern_replace_exact",
            extractor=lambda: build_paired_flip_df(
                forcing[forcing["condition"] == "baseline_corrupt"].copy(),
                forcing[forcing["condition"] == "h9_pattern_replace_exact"].copy(),
                sample_col="sample_id",
                baseline_col="is_tool_call_top1",
                intervention_col="is_tool_call_top1",
            ),
        ),
        MetricSpec(
            metric="schema_removed_corrupt_tool_call_rate",
            source_file=PHASE4_ROOT / "exp_l_schema_control" / "schema_control_per_sample.csv",
            definition="Phase 4 reported 24.46%; corrupt_analysis_side + schema_removed top-1 tool-call rate",
            extractor=lambda: build_single_metric_df(
                schema_control[
                    (schema_control["verb_condition"] == "corrupt_analysis_side")
                    & (schema_control["condition"] == "schema_removed")
                ].copy(),
                sample_col="sample_id",
                value_col="is_tool_call_top1",
            ),
        ),
        MetricSpec(
            metric="unrelated_schema_clean_tool_call_rate",
            source_file=PHASE4_ROOT / "exp_l_schema_control" / "schema_control_per_sample.csv",
            definition="Phase 4 reported 0.39%; clean_action_side + unrelated_valid_schema_length_matched top-1 tool-call rate",
            extractor=lambda: build_single_metric_df(
                schema_control[
                    (schema_control["verb_condition"] == "clean_action_side")
                    & (schema_control["condition"] == "unrelated_valid_schema_length_matched")
                ].copy(),
                sample_col="sample_id",
                value_col="is_tool_call_top1",
            ),
        ),
        MetricSpec(
            metric="all_except_last_tool_call_rate",
            source_file=PHASE4_ROOT / "exp_m_cross_position_control" / "cross_position_per_sample.csv",
            definition="Phase 4 reported 33.33%; all_except_last_patch top-1 tool-call rate",
            extractor=lambda: build_single_metric_df(
                cross_position[cross_position["condition"] == "all_except_last_patch"].copy(),
                sample_col="sample_id",
                value_col="is_tool_call_top1",
            ),
        ),
        MetricSpec(
            metric="all_except_last_strict_flip_rate",
            source_file=PHASE4_ROOT / "exp_m_cross_position_control" / "cross_position_per_sample.csv",
            definition="Strict paired flip rate against baseline_corrupt for all_except_last_patch",
            extractor=lambda: build_paired_flip_df(
                cross_position[cross_position["condition"] == "baseline_corrupt"].copy(),
                cross_position[cross_position["condition"] == "all_except_last_patch"].copy(),
                sample_col="sample_id",
                baseline_col="is_tool_call_top1",
                intervention_col="is_tool_call_top1",
            ),
        ),
        MetricSpec(
            metric="h9_system_attention_clean",
            source_file=PHASE4_ROOT / "exp_h_query_specific" / "query_specific_per_sample.csv",
            definition="Mean H9 system attention under baseline_clean",
            extractor=lambda: build_single_metric_df(
                query_specific[query_specific["condition"] == "baseline_clean"].copy(),
                sample_col="sample_id",
                value_col="system_attn_h9",
            ),
        ),
        MetricSpec(
            metric="h9_system_attention_corrupt",
            source_file=PHASE4_ROOT / "exp_h_query_specific" / "query_specific_per_sample.csv",
            definition="Mean H9 system attention under baseline_corrupt",
            extractor=lambda: build_single_metric_df(
                query_specific[query_specific["condition"] == "baseline_corrupt"].copy(),
                sample_col="sample_id",
                value_col="system_attn_h9",
            ),
        ),
        MetricSpec(
            metric="l33h29_dla_baseline_corrupt",
            source_file=PHASE4_ROOT / "exp_i_h9_forcing" / "forcing_per_sample.csv",
            definition="Mean L33H29 DLA under baseline_corrupt",
            extractor=lambda: build_single_metric_df(
                forcing[forcing["condition"] == "baseline_corrupt"].copy(),
                sample_col="sample_id",
                value_col="l33h29_dla",
            ),
        ),
        MetricSpec(
            metric="l33h29_dla_h9_pattern_replace",
            source_file=PHASE4_ROOT / "exp_i_h9_forcing" / "forcing_per_sample.csv",
            definition="Mean L33H29 DLA after h9_pattern_replace_exact",
            extractor=lambda: build_single_metric_df(
                forcing[forcing["condition"] == "h9_pattern_replace_exact"].copy(),
                sample_col="sample_id",
                value_col="l33h29_dla",
            ),
        ),
        MetricSpec(
            metric="l33h29_dla_table_baseline_corrupt",
            source_file=PHASE4_ROOT / "exp_i_h9_forcing" / "forcing_l33_dependency.csv",
            definition="Summary-table baseline_corrupt mean from forcing_l33_dependency.csv",
            extractor=lambda: pd.DataFrame(
                {
                    "sample_id": ["summary_row"],
                    "value": [
                        float(
                            forcing_dep.loc[
                                forcing_dep["condition"] == "baseline_corrupt", "mean_l33h29_dla"
                            ].iloc[0]
                        )
                    ],
                }
            ),
        ),
        MetricSpec(
            metric="l33h29_dla_table_h9_pattern_replace",
            source_file=PHASE4_ROOT / "exp_i_h9_forcing" / "forcing_l33_dependency.csv",
            definition="Summary-table h9_pattern_replace_exact mean from forcing_l33_dependency.csv",
            extractor=lambda: pd.DataFrame(
                {
                    "sample_id": ["summary_row"],
                    "value": [
                        float(
                            forcing_dep.loc[
                                forcing_dep["condition"] == "h9_pattern_replace_exact", "mean_l33h29_dla"
                            ].iloc[0]
                        )
                    ],
                }
            ),
        ),
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 5 experiment A: bootstrap CIs for headline metrics")
    parser.add_argument("--output-root", type=Path, default=PHASE5_ROOT)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dir(args.output_root)

    rows: list[dict[str, object]] = []
    for idx, spec in enumerate(make_metric_specs()):
        metric_df = spec.extractor()
        values = metric_df["value"].to_numpy(dtype=np.float64)
        point = float(values.mean()) if values.size else float("nan")
        lo, hi = bootstrap_ci(values, n_boot=args.bootstrap_samples, seed=args.seed + idx * 97)
        rows.append(
            {
                "metric": spec.metric,
                "definition": spec.definition,
                "point_estimate": point,
                "ci_lower": lo,
                "ci_upper": hi,
                "n_samples": int(values.size),
                "source_file": str(spec.source_file),
            }
        )

    write_csv(args.output_root / "headline_ci_table.csv", rows)

    lookup = {str(row["metric"]): row for row in rows}
    lines = [
        "# Experiment A: Bootstrap Confidence Intervals",
        "",
        f"- bootstrap samples: `{args.bootstrap_samples}`",
        "",
        "## Headline CIs",
        "",
        "| metric | point | 95% CI | n |",
        "|---|---:|---:|---:|",
    ]
    ordered_metrics = [
        "l24_state_patch_flip_rate",
        "h9_query_patch_tool_call_rate",
        "h9_query_patch_strict_flip_rate",
        "h9_pattern_forcing_tool_call_rate",
        "h9_pattern_forcing_strict_flip_rate",
        "schema_removed_corrupt_tool_call_rate",
        "unrelated_schema_clean_tool_call_rate",
        "all_except_last_tool_call_rate",
        "all_except_last_strict_flip_rate",
        "h9_system_attention_clean",
        "h9_system_attention_corrupt",
        "l33h29_dla_baseline_corrupt",
        "l33h29_dla_h9_pattern_replace",
    ]
    for metric in ordered_metrics:
        row = lookup[metric]
        point = float(row["point_estimate"])
        lo = float(row["ci_lower"])
        hi = float(row["ci_upper"])
        if "attention" in metric or "dla" in metric:
            point_str = f"{point:.4f}"
            ci_str = f"[{lo:.4f}, {hi:.4f}]"
        else:
            point_str = f"{point:.2%}"
            ci_str = f"[{lo:.2%}, {hi:.2%}]"
        lines.append(f"| {metric} | {point_str} | {ci_str} | {int(row['n_samples'])} |")

    lines.extend(
        [
            "",
            "## Notes",
            "",
            f"- 继续沿用当前文档口径时，`h9_query_patch_tool_call_rate` = `{float(lookup['h9_query_patch_tool_call_rate']['point_estimate']):.2%}`，"
            f"其严格 paired flip 版本是 `{float(lookup['h9_query_patch_strict_flip_rate']['point_estimate']):.2%}`。",
            f"- `h9_pattern_forcing_tool_call_rate` = `{float(lookup['h9_pattern_forcing_tool_call_rate']['point_estimate']):.2%}`，"
            f"其严格 paired flip 版本是 `{float(lookup['h9_pattern_forcing_strict_flip_rate']['point_estimate']):.2%}`。",
            f"- `all_except_last_tool_call_rate` = `{float(lookup['all_except_last_tool_call_rate']['point_estimate']):.2%}`，"
            f"其严格 paired flip 版本是 `{float(lookup['all_except_last_strict_flip_rate']['point_estimate']):.2%}`。",
        ]
    )
    write_text(args.output_root / "summary.md", "\n".join(lines))


if __name__ == "__main__":
    main()
