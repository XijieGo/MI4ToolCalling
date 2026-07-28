#!/usr/bin/env python3
"""Render the eight full-transfer tables and per-model PCA summaries.

Input is the completed output of ``run_full_transfer_matrix.py`` for one or
more Qwen3 scales.  The renderer is deliberately result-only: it verifies
that each model has all 32 directed condition cells, then creates eight 4x4
tables/figures (four scales x two treatments) and PCA/cosine summaries.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import numpy as np


DOMAINS = ("D1", "D3", "D4", "D5")
CONDITIONS = ("target_norm_aligned", "raw_1p5")
DISPLAY_CONDITION = {
    "target_norm_aligned": "Target-norm aligned (α=1)",
    "raw_1p5": "Raw source vector (α=1.5)",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render completed full-transfer matrix results.")
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return payload


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No rows in {path}")
    return rows


def model_roots(run_root: Path) -> list[tuple[str, Path]]:
    roots: list[tuple[str, Path]] = []
    for path in sorted(run_root.iterdir()):
        if not path.is_dir():
            continue
        completion = path / "completion.json"
        if completion.is_file():
            payload = read_json(completion)
            if payload.get("status") != "complete":
                raise ValueError(f"Incomplete model result: {path}")
            roots.append((path.name, path))
    if not roots:
        raise ValueError(f"No completed model roots in {run_root}")
    return roots


def condition_matrix(rows: list[dict[str, str]], condition: str) -> dict[tuple[str, str], dict[str, str]]:
    subset = [row for row in rows if row["condition"] == condition]
    expected = {(source, target) for source in DOMAINS for target in DOMAINS}
    observed = {(row["source_domain"], row["target_domain"]) for row in subset}
    if observed != expected or len(subset) != 16:
        raise ValueError(f"{condition}: expected 16 complete directed cells, found {len(subset)}")
    return {(row["source_domain"], row["target_domain"]): row for row in subset}


def cell_text(row: dict[str, str]) -> str:
    return (
        f"S {float(row['sufficiency_normalized_logit_gap']):.2f}\n"
        f"N {float(row['necessity_normalized_logit_gap']):.2f}\n"
        f"F/D {100 * float(row['add_strict_flip_rate']):.0f}/{100 * float(row['remove_strict_drop_rate']):.0f}%"
    )


def clipped_score(row: dict[str, str]) -> float:
    suff = min(max(float(row["sufficiency_normalized_logit_gap"]), 0.0), 1.0)
    necc = min(max(float(row["necessity_normalized_logit_gap"]), 0.0), 1.0)
    return 0.5 * (suff + necc)


def render_matrix_png(model_label: str, condition: str, matrix: dict[tuple[str, str], dict[str, str]], output: Path) -> None:
    fig, ax = plt.subplots(figsize=(12.6, 8.8))
    ax.axis("off")
    text = [[cell_text(matrix[(source, target)]) for target in DOMAINS] for source in DOMAINS]
    table = ax.table(
        cellText=text,
        rowLabels=DOMAINS,
        colLabels=DOMAINS,
        cellLoc="center",
        rowLoc="center",
        loc="center",
        bbox=[0.055, 0.08, 0.84, 0.79],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    cmap = plt.get_cmap("YlGnBu")
    norm = Normalize(vmin=0.0, vmax=1.0)
    for row_idx, source in enumerate(DOMAINS, start=1):
        for col_idx, target in enumerate(DOMAINS):
            cell = table[(row_idx, col_idx)]
            cell.set_facecolor(cmap(norm(clipped_score(matrix[(source, target)]))))
            cell.set_edgecolor("#ffffff")
            cell.set_linewidth(1.0)
    for col_idx in range(len(DOMAINS)):
        table[(0, col_idx)].set_facecolor("#e8edf3")
        table[(0, col_idx)].set_text_props(weight="bold")
    for row_idx in range(1, len(DOMAINS) + 1):
        table[(row_idx, -1)].set_facecolor("#e8edf3")
        table[(row_idx, -1)].set_text_props(weight="bold")
    fig.text(0.5, 0.95, f"{model_label} — {DISPLAY_CONDITION[condition]}", ha="center", va="top", fontsize=18, weight="bold")
    fig.text(
        0.5,
        0.905,
        "Rows = source-domain vector; columns = held-out target domain.  S = normalized sufficiency; N = normalized necessity; F/D = strict flip/drop.",
        ha="center",
        va="top",
        fontsize=10.5,
    )
    colorbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax, fraction=0.045, pad=0.02)
    colorbar.set_label("Mean of clipped S and N (color only; values are printed)", fontsize=9)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=230, bbox_inches="tight")
    plt.close(fig)


def write_matrix_csv(path: Path, matrix: dict[tuple[str, str], dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["source_domain", *DOMAINS])
        for source in DOMAINS:
            writer.writerow(
                [
                    source,
                    *[
                        "S={:.6f}; N={:.6f}; flip={:.6f}; drop={:.6f}".format(
                            float(matrix[(source, target)]["sufficiency_normalized_logit_gap"]),
                            float(matrix[(source, target)]["necessity_normalized_logit_gap"]),
                            float(matrix[(source, target)]["add_strict_flip_rate"]),
                            float(matrix[(source, target)]["remove_strict_drop_rate"]),
                        )
                        for target in DOMAINS
                    ],
                ]
            )


def markdown_matrix(model_label: str, condition: str, matrix: dict[tuple[str, str], dict[str, str]]) -> list[str]:
    lines = [
        f"## {model_label} — {DISPLAY_CONDITION[condition]}",
        "",
        "Each cell: `S` normalized sufficiency / `N` normalized necessity; `F/D` strict flip/drop.",
        "",
        "| source \\ target | " + " | ".join(DOMAINS) + " |",
        "|---|" + "|".join(["---:"] * len(DOMAINS)) + "|",
    ]
    for source in DOMAINS:
        values = []
        for target in DOMAINS:
            row = matrix[(source, target)]
            values.append(
                "S={:.3f}<br>N={:.3f}<br>F/D={:.0f}/{:.0f}%".format(
                    float(row["sufficiency_normalized_logit_gap"]),
                    float(row["necessity_normalized_logit_gap"]),
                    100 * float(row["add_strict_flip_rate"]),
                    100 * float(row["remove_strict_drop_rate"]),
                )
            )
        lines.append("| " + source + " | " + " | ".join(values) + " |")
    lines.append("")
    return lines


def render_pca_png(model_label: str, pca: dict[str, Any], output: Path) -> None:
    cosine = np.array([[float(pca["pairwise_cosine"][source][target]) for target in DOMAINS] for source in DOMAINS])
    fig, (ax_cos, ax_pca) = plt.subplots(1, 2, figsize=(12, 5.2), gridspec_kw={"width_ratios": [1.2, 1]})
    image = ax_cos.imshow(cosine, cmap="coolwarm", vmin=-1.0, vmax=1.0)
    ax_cos.set_xticks(range(len(DOMAINS)), DOMAINS)
    ax_cos.set_yticks(range(len(DOMAINS)), DOMAINS)
    ax_cos.set_title("Pairwise cosine of native directions")
    for row_idx in range(len(DOMAINS)):
        for col_idx in range(len(DOMAINS)):
            value = cosine[row_idx, col_idx]
            ax_cos.text(col_idx, row_idx, f"{value:.2f}", ha="center", va="center", color="black" if abs(value) < 0.55 else "white")
    fig.colorbar(image, ax=ax_cos, fraction=0.045, pad=0.04)

    rank1 = 100 * float(pca["rank1_explained_energy_ratio"])
    centered = pca.get("centered_pca_explained_variance_ratio", [])
    centered_pc1 = 100 * float(centered[0]) if centered else 0.0
    bars = ax_pca.bar(["Shared rank-1\nenergy", "Centered PCA\nPC1 variance"], [rank1, centered_pc1], color=["#2166ac", "#b2182b"])
    ax_pca.axhline(70.0, linestyle="--", color="#666666", linewidth=1, label="70% reference")
    ax_pca.axhline(80.0, linestyle=":", color="#666666", linewidth=1, label="80% reference")
    ax_pca.set_ylim(0, 105)
    ax_pca.set_ylabel("Explained percentage")
    ax_pca.set_title("PCA / common-direction summary")
    ax_pca.legend(fontsize=8, loc="lower right")
    for bar, value in zip(bars, [rank1, centered_pc1]):
        ax_pca.text(bar.get_x() + bar.get_width() / 2, value + 2, f"{value:.1f}%", ha="center", va="bottom", weight="bold")
    fig.suptitle(f"{model_label} — L24 native vectors", fontsize=16, weight="bold")
    fig.text(0.5, 0.015, "Shared rank-1 energy uses uncentered SVD of L2-normalized domain vectors; it directly tests a common direction.", ha="center", fontsize=9)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=230, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    run_root = args.run_root.resolve()
    output_root = args.output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite existing output root: {output_root}")
    roots = model_roots(run_root)
    if len(roots) != 4:
        raise ValueError(f"Expected four completed Qwen3 model results, found {len(roots)}")
    output_root.mkdir(parents=True, exist_ok=False)
    figures_root = output_root / "figures"
    tables_root = output_root / "tables"
    matrix_markdown: list[str] = [
        "# Full Cross-domain Causal-transfer Matrices",
        "",
        "Each model has two 4×4 directed tables. The intervention layer is L24 `hook_resid_pre` at the final prompt token.",
        "`target_norm_aligned`: source direction rescaled to the target-native L2 norm. `raw_1p5`: native source vector ×1.5.",
        "",
    ]
    pca_rows: list[dict[str, Any]] = []
    aggregate_rows: list[dict[str, Any]] = []
    pdf_path = output_root / "eight_transfer_matrices.pdf"
    with PdfPages(pdf_path) as pdf:
        for model_label, root in roots:
            rows = read_csv(root / "matrix_long.csv")
            completion = read_json(root / "completion.json")
            if int(completion["observed_cells"]) != 32 or len(rows) != 32:
                raise ValueError(f"{model_label}: invalid matrix cardinality")
            for condition in CONDITIONS:
                matrix = condition_matrix(rows, condition)
                csv_path = tables_root / f"{model_label}_{condition}.csv"
                write_matrix_csv(csv_path, matrix)
                png_path = figures_root / f"{model_label}_{condition}.png"
                render_matrix_png(model_label, condition, matrix, png_path)
                # PdfPages needs the rendered figure itself; re-open the PNG on a
                # compact canvas rather than duplicate the table-drawing logic.
                image = plt.imread(png_path)
                fig, ax = plt.subplots(figsize=(12.6, 8.8))
                ax.imshow(image)
                ax.axis("off")
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)
                matrix_markdown.extend(markdown_matrix(model_label, condition, matrix))
                for scope, cell_rows in (
                    ("all", list(matrix.values())),
                    ("diagonal", [matrix[(domain, domain)] for domain in DOMAINS]),
                    ("off_diagonal", [matrix[(source, target)] for source in DOMAINS for target in DOMAINS if source != target]),
                ):
                    suff = np.array([float(row["sufficiency_normalized_logit_gap"]) for row in cell_rows])
                    necc = np.array([float(row["necessity_normalized_logit_gap"]) for row in cell_rows])
                    flip = np.array([float(row["add_strict_flip_rate"]) for row in cell_rows])
                    drop = np.array([float(row["remove_strict_drop_rate"]) for row in cell_rows])
                    aggregate_rows.append(
                        {
                            "model": model_label,
                            "condition": condition,
                            "scope": scope,
                            "cell_count": len(cell_rows),
                            "mean_sufficiency": float(np.mean(suff)),
                            "median_sufficiency": float(np.median(suff)),
                            "mean_necessity": float(np.mean(necc)),
                            "median_necessity": float(np.median(necc)),
                            "mean_strict_flip_rate": float(np.mean(flip)),
                            "mean_strict_drop_rate": float(np.mean(drop)),
                            "both_behavioral_rate_one_fraction": float(np.mean((flip >= 1.0 - 1e-12) & (drop >= 1.0 - 1e-12))),
                        }
                    )

            pca = read_json(root / "pca_common_direction.json")
            pca_png = figures_root / f"{model_label}_pca_common_direction.png"
            render_pca_png(model_label, pca, pca_png)
            cosine_values = [
                float(pca["pairwise_cosine"][source][target])
                for idx, source in enumerate(DOMAINS)
                for target in DOMAINS[idx + 1 :]
            ]
            pca_rows.append(
                {
                    "model": model_label,
                    "rank1_common_direction_explained_energy": float(pca["rank1_explained_energy_ratio"]),
                    "centered_pca_pc1_explained_variance": float(pca["centered_pca_explained_variance_ratio"][0]),
                    "mean_pairwise_cosine": float(np.mean(cosine_values)),
                    "min_pairwise_cosine": float(np.min(cosine_values)),
                    "min_cosine_to_common_direction": min(float(value) for value in pca["cosine_to_common_direction"].values()),
                }
            )
    (output_root / "eight_matrix_tables.md").write_text("\n".join(matrix_markdown) + "\n", encoding="utf-8")
    with (output_root / "pca_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(pca_rows[0]))
        writer.writeheader()
        writer.writerows(pca_rows)
    with (output_root / "aggregate_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(aggregate_rows[0]))
        writer.writeheader()
        writer.writerows(aggregate_rows)
    pca_lines = [
        "# PCA Common-direction Summary",
        "",
        "The primary common-direction measure is the rank-1 explained energy of an uncentered SVD over the four L2-normalized domain vectors. It directly measures whether one direction reconstructs all four vectors; the conventional centered-PCA value is included separately.",
        "",
        "| model | rank-1 common energy | centered PC1 variance | mean pairwise cosine | min pairwise cosine | min cosine to common |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in pca_rows:
        pca_lines.append(
            "| {model} | {rank1_common_direction_explained_energy:.1%} | {centered_pca_pc1_explained_variance:.1%} | {mean_pairwise_cosine:.3f} | {min_pairwise_cosine:.3f} | {min_cosine_to_common_direction:.3f} |".format(**row)
        )
    (output_root / "pca_summary.md").write_text("\n".join(pca_lines) + "\n", encoding="utf-8")
    aggregate_lines = [
        "# Aggregate Matrix Metrics",
        "",
        "Off-diagonal rows summarize the 12 genuine cross-domain directions; strict flip/drop use the behavioral top-1 decision.",
        "",
        "| model | condition | scope | mean Suff. | mean Necc. | mean flip | mean drop | both 100% |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in aggregate_rows:
        aggregate_lines.append(
            "| {model} | {condition} | {scope} | {mean_sufficiency:.3f} | {mean_necessity:.3f} | {mean_strict_flip_rate:.1%} | {mean_strict_drop_rate:.1%} | {both_behavioral_rate_one_fraction:.1%} |".format(**row)
        )
    (output_root / "aggregate_metrics.md").write_text("\n".join(aggregate_lines) + "\n", encoding="utf-8")
    write_manifest = {
        "run_root": str(run_root),
        "models": [label for label, _root in roots],
        "matrix_figures": [str(path.relative_to(output_root)) for path in sorted(figures_root.glob("*_*.*"))],
        "matrix_csv_tables": [str(path.relative_to(output_root)) for path in sorted(tables_root.glob("*.csv"))],
        "pca_summary": "pca_summary.md",
        "matrix_summary": "eight_matrix_tables.md",
        "aggregate_metrics": "aggregate_metrics.md",
    }
    (output_root / "render_manifest.json").write_text(json.dumps(write_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(output_root)


if __name__ == "__main__":
    main()
