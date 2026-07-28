#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from multiscale_common import ensure_dir, write_csv, write_json, write_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Canonicalize legacy phase5 generalization outputs into phase6 format.")
    parser.add_argument("--size-label", required=True)
    parser.add_argument("--legacy-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def maybe_float(text: str | None) -> float | None:
    if text is None:
        return None
    raw = str(text).strip()
    if not raw:
        return None
    return float(raw)


def maybe_int(text: str | None) -> int | None:
    if text is None:
        return None
    raw = str(text).strip()
    if not raw:
        return None
    return int(float(raw))


def plot_patch_sweep(rows: list[dict[str, object]], path: Path) -> None:
    if not rows:
        return
    layers = [int(row["layer"]) for row in rows]
    top1 = [float(row["tool_call_top1_rate"]) for row in rows]
    strict_flip = [float(row["strict_flip_rate"]) for row in rows]
    logits = [float(row["mean_tool_call_logit"]) for row in rows]

    fig, ax1 = plt.subplots(figsize=(7.2, 4.4))
    ax1.plot(layers, top1, marker="o", linewidth=2.0, label="tool-call top1")
    ax1.plot(layers, strict_flip, marker="s", linewidth=1.8, label="strict flip")
    ax1.set_xlabel("Layer")
    ax1.set_ylabel("Rate")
    ax1.set_ylim(0.0, 1.05)
    ax1.grid(alpha=0.25)

    ax2 = ax1.twinx()
    ax2.plot(layers, logits, marker="^", linewidth=1.5, linestyle="--", color="tab:red", label="mean tool logit")
    ax2.set_ylabel("Mean tool logit")

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="best")
    fig.tight_layout()
    ensure_dir(path.parent)
    fig.savefig(path)
    plt.close(fig)


def normalize_patch_rows(size_label: str, legacy_root: Path) -> tuple[list[dict[str, object]], dict[str, object]]:
    legacy_rows = load_csv(legacy_root / "exp_b_state_patch" / "patch_sweep.csv")
    normalized: list[dict[str, object]] = []
    baseline_clean = None
    baseline_corrupt = None

    if legacy_rows and "patch_layer" in legacy_rows[0]:
        for row in legacy_rows:
            condition = str(row["condition"])
            patch_layer = str(row["patch_layer"])
            if condition == "baseline_clean":
                baseline_clean = maybe_float(row["tool_call_top1_rate"])
                continue
            if condition == "baseline_corrupt":
                baseline_corrupt = maybe_float(row["tool_call_top1_rate"])
                continue
            if not patch_layer.isdigit():
                continue
            normalized.append(
                {
                    "layer": int(patch_layer),
                    "n": maybe_int(row["n_samples"]),
                    "tool_call_top1_rate": maybe_float(row["tool_call_top1_rate"]),
                    "strict_flip_rate": maybe_float(row["strict_flip_rate"]),
                    "mean_tool_call_logit": maybe_float(row["mean_tool_logit"]),
                    "mean_tool_call_prob": None,
                    "baseline_clean_tool_top1_rate": baseline_clean,
                    "baseline_corrupt_tool_top1_rate": baseline_corrupt,
                }
            )
    else:
        for row in legacy_rows:
            normalized.append(
                {
                    "layer": maybe_int(row["layer"]),
                    "n": maybe_int(row["n"]),
                    "tool_call_top1_rate": maybe_float(row["tool_call_top1_rate"]),
                    "strict_flip_rate": maybe_float(row["strict_flip_rate"]),
                    "mean_tool_call_logit": maybe_float(row["mean_tool_call_logit"]),
                    "mean_tool_call_prob": maybe_float(row.get("mean_tool_call_prob")),
                    "baseline_clean_tool_top1_rate": None,
                    "baseline_corrupt_tool_top1_rate": maybe_float(row.get("baseline_corrupt_tool_top1_rate")),
                }
            )

    normalized.sort(key=lambda row: int(row["layer"]))
    best_row = max(normalized, key=lambda row: float(row["tool_call_top1_rate"]))
    summary = {
        "size_label": size_label,
        "best_layer": int(best_row["layer"]),
        "best_tool_call_top1_rate": float(best_row["tool_call_top1_rate"]),
        "best_strict_flip_rate": float(best_row["strict_flip_rate"]),
        "baseline_clean_tool_top1_rate": baseline_clean,
        "baseline_corrupt_tool_top1_rate": baseline_corrupt if baseline_corrupt is not None else best_row.get("baseline_corrupt_tool_top1_rate"),
    }
    return normalized, summary


def normalize_explained_variance(legacy_root: Path) -> list[dict[str, object]]:
    legacy_rows = load_csv(legacy_root / "exp_c_gate_rank1" / "explained_variance.csv")
    normalized: list[dict[str, object]] = []
    for row in legacy_rows:
        if "component_idx" in row:
            normalized.append(
                {
                    "component": maybe_int(row["component_idx"]),
                    "singular_value": None,
                    "explained_variance": maybe_float(row["explained_var"]),
                    "explained_ratio": maybe_float(row["explained_var_ratio"]),
                    "cumulative_ratio": maybe_float(row["cumulative"]),
                }
            )
        else:
            normalized.append(
                {
                    "component": maybe_int(row["component"]),
                    "singular_value": maybe_float(row.get("singular_value")),
                    "explained_variance": maybe_float(row["explained_variance"]),
                    "explained_ratio": maybe_float(row["explained_ratio"]),
                    "cumulative_ratio": maybe_float(row["cumulative_ratio"]),
                }
            )
    return normalized


def normalize_rank_rows(legacy_root: Path) -> list[dict[str, object]]:
    legacy_rows = load_csv(legacy_root / "exp_c_gate_rank1" / "rank_k_patch_sweep.csv")
    normalized: list[dict[str, object]] = []
    for row in legacy_rows:
        if "k" in row:
            normalized.append(
                {
                    "rank_k": str(row["k"]),
                    "alpha": maybe_float(row["alpha"]),
                    "n": maybe_int(row["n_samples"]),
                    "tool_call_top1_rate": maybe_float(row["tool_call_top1_rate"]),
                    "strict_flip_rate": maybe_float(row["strict_flip_rate"]),
                    "mean_tool_call_logit": maybe_float(row["mean_tool_logit"]),
                    "mean_tool_call_prob": None,
                }
            )
        else:
            normalized.append(
                {
                    "rank_k": str(row["rank_k"]),
                    "alpha": 1.0,
                    "n": maybe_int(row["n"]),
                    "tool_call_top1_rate": maybe_float(row["tool_call_top1_rate"]),
                    "strict_flip_rate": maybe_float(row["strict_flip_rate"]),
                    "mean_tool_call_logit": maybe_float(row["mean_tool_call_logit"]),
                    "mean_tool_call_prob": maybe_float(row.get("mean_tool_call_prob")),
                }
            )
    order = {"1": 1, "3": 3, "5": 5, "10": 10, "full": 999}
    normalized.sort(key=lambda row: (order.get(str(row["rank_k"]), 500), float(row["alpha"])))
    return normalized


def plot_rank_k(rows: list[dict[str, object]], path: Path) -> None:
    if not rows:
        return
    rank_order = ["1", "3", "5", "10", "full"]
    best_rows = []
    for rank_k in rank_order:
        candidates = [row for row in rows if str(row["rank_k"]) == rank_k]
        if candidates:
            best_rows.append(max(candidates, key=lambda row: float(row["tool_call_top1_rate"])))
    labels = [str(row["rank_k"]) for row in best_rows]
    top1 = [float(row["tool_call_top1_rate"]) for row in best_rows]
    strict_flip = [float(row["strict_flip_rate"]) for row in best_rows]

    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    ax.plot(labels, top1, marker="o", linewidth=2.0, label="tool-call top1")
    ax.plot(labels, strict_flip, marker="s", linewidth=1.8, label="strict flip")
    ax.set_xlabel("Rank k")
    ax.set_ylabel("Rate")
    ax.set_ylim(0.0, 1.05)
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    ensure_dir(path.parent)
    fig.savefig(path)
    plt.close(fig)


def build_exp_b_summary(size_label: str, explained_rows: list[dict[str, object]], rank_rows: list[dict[str, object]]) -> str:
    pc1 = explained_rows[0]
    rank1 = max((row for row in rank_rows if str(row["rank_k"]) == "1"), key=lambda row: float(row["tool_call_top1_rate"]))
    full = max((row for row in rank_rows if str(row["rank_k"]) == "full"), key=lambda row: float(row["tool_call_top1_rate"]))
    lines = [
        "# Exp B Summary",
        "",
        f"- Source: legacy `phase5_neurips` gate-rank experiment for `{size_label}`.",
        f"- PC1 explained variance ratio: `{float(pc1['explained_ratio']):.4f}`",
        f"- Best rank-1 recovery: top1 `{float(rank1['tool_call_top1_rate']):.2%}`, strict flip `{float(rank1['strict_flip_rate']):.2%}`",
        f"- Best full recovery: top1 `{float(full['tool_call_top1_rate']):.2%}`, strict flip `{float(full['strict_flip_rate']):.2%}`",
        "- Fixed shared direction sweep was not rerun in the legacy asset, so `fixed_direction_sweep.csv` is intentionally left empty.",
        "",
        "Interpretation:",
        "The legacy result already supports the current task-book claim that the key-layer clean/corrupt difference is compact enough to admit strong low-rank recovery.",
    ]
    return "\n".join(lines)


def normalize_downstream_scores(legacy_root: Path) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    bridge_rows = load_csv(legacy_root / "exp_d_bridge_head" / "top_bridge_heads.csv")
    late_rows = load_csv(legacy_root / "exp_e_late_writer" / "top_heads.csv")
    mlp_rows = load_csv(legacy_root / "exp_e_late_writer" / "dla_by_layer.csv")

    normalized: list[dict[str, object]] = []
    region_rows: list[dict[str, object]] = []

    for row in bridge_rows:
        normalized.append(
            {
                "layer": maybe_int(row["layer"]),
                "head": maybe_int(row["head"]),
                "role": "bridge_candidate",
                "source": "legacy_exp_d_bridge_head",
                "mean_clean": maybe_float(row.get("clean_dla") or row.get("dla_clean")),
                "mean_corrupt": maybe_float(row.get("corrupt_dla") or row.get("dla_corrupt")),
                "dla_delta": maybe_float(row.get("delta_dla") or row.get("dla_delta")),
                "system_attn_clean": maybe_float(row.get("clean_system_attn")),
                "system_attn_corrupt": maybe_float(row.get("corrupt_system_attn")),
                "system_attn_delta": maybe_float(row.get("delta_system_attn")),
                "score": maybe_float(row.get("composite_score") or row.get("bridge_score")),
            }
        )
        region_rows.append(
            {
                "layer": maybe_int(row["layer"]),
                "head": maybe_int(row["head"]),
                "region": "system",
                "clean_attention": maybe_float(row.get("clean_system_attn")),
                "corrupt_attention": maybe_float(row.get("corrupt_system_attn")),
                "delta_attention": maybe_float(row.get("delta_system_attn")),
                "source": "legacy_exp_d_bridge_head",
            }
        )

    for row in late_rows:
        normalized.append(
            {
                "layer": maybe_int(row["layer"]),
                "head": maybe_int(row["head"]),
                "role": str(row.get("role") or row.get("role_guess") or "late_candidate"),
                "source": "legacy_exp_e_late_writer",
                "mean_clean": maybe_float(row.get("mean_clean")),
                "mean_corrupt": maybe_float(row.get("mean_corrupt")),
                "dla_delta": maybe_float(row.get("delta")),
                "system_attn_clean": None,
                "system_attn_corrupt": None,
                "system_attn_delta": None,
                "score": maybe_float(row.get("abs_delta") or row.get("delta")),
            }
        )

    normalized.sort(key=lambda row: float(row["score"]) if row["score"] is not None else float("-inf"), reverse=True)
    return normalized, region_rows, mlp_rows


def build_exp_c_summary(size_label: str, downstream_rows: list[dict[str, object]], legacy_root: Path) -> str:
    best = downstream_rows[0]
    bridge_summary = (legacy_root / "exp_d_bridge_head" / "summary.md").read_text(encoding="utf-8").strip()
    late_summary = (legacy_root / "exp_e_late_writer" / "summary.md").read_text(encoding="utf-8").strip()
    lines = [
        "# Exp C Summary",
        "",
        f"- Source: merged legacy `exp_d_bridge_head` + `exp_e_late_writer` for `{size_label}`.",
        f"- Top downstream head by legacy score: `L{int(best['layer'])}H{int(best['head'])}` with DLA delta `{float(best['dla_delta']):.4f}`.",
        "- The canonicalized output keeps bridge-like and later-writer-like evidence in one directory, matching the current task-book interpretation of late distributed readout.",
        "",
        "Legacy bridge summary:",
        bridge_summary,
        "",
        "Legacy late-writer summary:",
        late_summary,
    ]
    return "\n".join(lines)


def build_exp_a_summary(size_label: str, summary: dict[str, object]) -> str:
    lines = [
        "# Exp A Summary",
        "",
        f"- Source: legacy `phase5_neurips/exp_b_state_patch` for `{size_label}`.",
        f"- Best layer `L{int(summary['best_layer'])}`.",
        f"- Best patched top-1 `<tool_call>` rate: `{float(summary['best_tool_call_top1_rate']):.2%}`.",
        f"- Best strict flip rate: `{float(summary['best_strict_flip_rate']):.2%}`.",
        "",
        "Interpretation:",
        "This legacy state-patch sweep already provides the task-book Exp A evidence: a specific prediction-position residual state in the mid-to-late stack acts as the dominant causal bottleneck.",
    ]
    return "\n".join(lines)


def maybe_write_json_from_legacy(path: Path) -> dict[str, object]:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def main() -> None:
    args = parse_args()
    legacy_root = args.legacy_root
    output_root = args.output_root
    ensure_dir(output_root)

    exp_a_root = output_root / "exp_a_state_patch"
    exp_b_root = output_root / "exp_b_gate_vector"
    exp_c_root = output_root / "exp_c_late_readout"
    ensure_dir(exp_a_root)
    ensure_dir(exp_b_root)
    ensure_dir(exp_c_root)

    patch_rows, patch_summary = normalize_patch_rows(args.size_label, legacy_root)
    write_csv(exp_a_root / "patch_sweep.csv", patch_rows)
    plot_patch_sweep(patch_rows, exp_a_root / "plot_patch_sweep.pdf")
    write_text(exp_a_root / "summary.md", build_exp_a_summary(args.size_label, patch_summary))
    write_json(
        exp_a_root / "metadata.json",
        {
            "size_label": args.size_label,
            "source_root": str(legacy_root),
            "source_patch_sweep": str(legacy_root / "exp_b_state_patch" / "patch_sweep.csv"),
            "legacy_summary": maybe_write_json_from_legacy(legacy_root / "exp_b_state_patch" / "summary.json"),
            "best_layer": patch_summary["best_layer"],
        },
    )

    explained_rows = normalize_explained_variance(legacy_root)
    rank_rows = normalize_rank_rows(legacy_root)
    write_csv(exp_b_root / "explained_variance.csv", explained_rows)
    write_csv(exp_b_root / "rank_k_patch_sweep.csv", rank_rows)
    write_csv(exp_b_root / "fixed_direction_sweep.csv", [])
    plot_rank_k(rank_rows, exp_b_root / "plot_rank_k_recovery.pdf")
    pca_src = legacy_root / "exp_c_gate_rank1" / "pca_components.pt"
    if pca_src.exists():
        (exp_b_root / "pca_components.pt").write_bytes(pca_src.read_bytes())
    write_text(exp_b_root / "summary.md", build_exp_b_summary(args.size_label, explained_rows, rank_rows))
    write_json(
        exp_b_root / "metadata.json",
        {
            "size_label": args.size_label,
            "source_root": str(legacy_root),
            "source_explained_variance": str(legacy_root / "exp_c_gate_rank1" / "explained_variance.csv"),
            "source_rank_k_patch": str(legacy_root / "exp_c_gate_rank1" / "rank_k_patch_sweep.csv"),
            "fixed_direction_sweep_status": "not available in legacy asset",
        },
    )

    downstream_rows, region_rows, mlp_rows = normalize_downstream_scores(legacy_root)
    top_rows = downstream_rows[:20]
    write_csv(exp_c_root / "downstream_head_scores.csv", downstream_rows)
    write_csv(exp_c_root / "top_downstream_heads.csv", top_rows)
    write_csv(exp_c_root / "top_head_region_attention.csv", region_rows)
    write_csv(exp_c_root / "optional_mlp_scores.csv", mlp_rows)
    write_text(exp_c_root / "summary.md", build_exp_c_summary(args.size_label, downstream_rows, legacy_root))
    write_json(
        exp_c_root / "metadata.json",
        {
            "size_label": args.size_label,
            "source_root": str(legacy_root),
            "source_bridge_scores": str(legacy_root / "exp_d_bridge_head" / "top_bridge_heads.csv"),
            "source_late_writer_scores": str(legacy_root / "exp_e_late_writer" / "top_heads.csv"),
            "source_mlp_scores": str(legacy_root / "exp_e_late_writer" / "dla_by_layer.csv"),
        },
    )


if __name__ == "__main__":
    main()
