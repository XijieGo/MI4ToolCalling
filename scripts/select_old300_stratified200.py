#!/usr/bin/env python3
"""Create an auditable, outcome-blind 200-pair subset of the historical v2 set.

The subset balances clean/corrupt verbs as far as source availability allows,
retains the original source-domain mix, and preserves the corrupt-baseline
margin quintiles. Treatment/intervention outcomes are not part of selection.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import Bounds, LinearConstraint, milp


REPO_ROOT = Path(__file__).resolve().parents[1]
OLD_DATA_ROOT = Path("/root/autodl-tmp/MI4ToolCalling-sync-v0/datasets/test")
HIST_ROOT = REPO_ROOT / "results/qwen3_8b/historical_source_audit_20261005/v2_300_fixed_v1_protocol"
BASE_CACHE_PATH = HIST_ROOT / "phase6_cache.pt"
BASE_MLP_CSV = HIST_ROOT / "mlp34_patching/mlp34_patch_per_sample.csv"
OUT_ROOT = REPO_ROOT / "results/qwen3_8b/old300_verb_balanced200_20261005"
SEED = "qwen3_8b_old300_to_200_verb_balanced_20261005_v1"

# Hamilton-style domain quotas from the old 300-pair proportions.
QUOTAS = {
    "dataset": {"codecontests": 135, "apps": 51, "mbpp": 10, "humaneval": 4},
    # build/complete have only 20 examples each in the source 300; include all
    # available rows and split the remaining 160 as evenly as possible.
    "clean_verb": {"add": 53, "build": 20, "complete": 20, "save": 53, "write": 54},
    "corrupt_verb": {"discuss": 40, "explore": 40, "inspect": 40, "review": 40, "study": 40},
    "language": {"cpp": 67, "java": 68, "python": 65},
    "corrupt_margin_quintile": {str(i): 40 for i in range(1, 6)},
}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def select_rows(clean_rows: list[dict], corrupt_rows: list[dict], mlp_rows: dict[str, dict]) -> tuple[list[dict], dict]:
    if len(clean_rows) != 300 or len(corrupt_rows) != 300:
        raise ValueError("Expected the historical v2 set to contain 300 aligned pairs.")
    rows = []
    for clean, corrupt in zip(clean_rows, corrupt_rows):
        filename = clean.get("output_filename") or clean.get("source_filename")
        if filename != (corrupt.get("output_filename") or corrupt.get("source_filename")):
            raise ValueError("Clean/corrupt manifests are not aligned by filename.")
        sample_id = Path(str(filename)).stem
        if sample_id not in mlp_rows:
            raise ValueError(f"Missing baseline margin for {sample_id}.")
        rows.append(
            {
                "sample_id": sample_id,
                "dataset": clean["dataset_name"],
                "language": clean["language"],
                "clean_verb": clean["clean_candidate_slug"],
                "corrupt_verb": corrupt["assigned_candidate_slug"],
                "corrupt_baseline_margin": float(mlp_rows[sample_id]["corrupt_margin"]),
                "clean_manifest_row": clean,
                "corrupt_manifest_row": corrupt,
            }
        )

    # Equal-sized quintiles of a pre-intervention quantity, with sample_id as
    # a deterministic tie-breaker for repeated margin values.
    ranked = sorted(range(len(rows)), key=lambda i: (rows[i]["corrupt_baseline_margin"], rows[i]["sample_id"]))
    for rank, row_index in enumerate(ranked):
        rows[row_index]["corrupt_margin_quintile"] = str(rank // 60 + 1)

    constraint_rows: list[list[int]] = []
    lower: list[float] = []
    upper: list[float] = []
    for field, values in QUOTAS.items():
        for value, target in values.items():
            constraint_rows.append([int(str(row[field]) == str(value)) for row in rows])
            lower.append(target)
            upper.append(target)

    objective = np.asarray(
        [
            int(hashlib.sha256(f"{SEED}:{row['sample_id']}".encode("utf-8")).hexdigest()[:13], 16)
            / float(16**13)
            for row in rows
        ],
        dtype=np.float64,
    )
    result = milp(
        objective,
        integrality=np.ones(len(rows), dtype=np.int8),
        bounds=Bounds(np.zeros(len(rows)), np.ones(len(rows))),
        constraints=LinearConstraint(np.asarray(constraint_rows, dtype=np.float64), lower, upper),
        options={"time_limit": 60},
    )
    if not result.success or result.x is None:
        raise RuntimeError(f"Could not satisfy the predeclared sampling quotas: {result.message}")
    selected_indices = [i for i, value in enumerate(result.x) if value > 0.5]
    if len(selected_indices) != 200:
        raise RuntimeError(f"Expected 200 selected pairs, got {len(selected_indices)}")
    selected = [rows[i] for i in selected_indices]  # retain original manifest order
    return selected, {"status": result.message, "objective": float(result.fun)}


def save_selected_dataset(selected: list[dict], clean_rows: list[dict], corrupt_rows: list[dict]) -> Path:
    dataset_root = OUT_ROOT / "dataset"
    clean_by_id = {Path(str(row["output_filename"])).stem: row for row in clean_rows}
    corrupt_by_id = {Path(str(row["output_filename"])).stem: row for row in corrupt_rows}
    for condition, source_root in (("clean", OLD_DATA_ROOT / "clean"), ("corrupt", OLD_DATA_ROOT / "corrupt")):
        destination = dataset_root / condition
        destination.mkdir(parents=True, exist_ok=True)
        selected_manifest_rows = []
        for selected_row in selected:
            sample_id = selected_row["sample_id"]
            manifest_row = (clean_by_id if condition == "clean" else corrupt_by_id)[sample_id]
            selected_manifest_rows.append(manifest_row)
            filename = manifest_row["output_filename"]
            link = destination / filename
            source = source_root / filename
            if link.exists() or link.is_symlink():
                link.unlink()
            link.symlink_to(source)
        write_jsonl(destination / "manifest.jsonl", selected_manifest_rows)

    pairs = []
    for row in selected:
        clean_path = dataset_root / "clean" / f"{row['sample_id']}.txt"
        corrupt_path = dataset_root / "corrupt" / f"{row['sample_id']}.txt"
        pairs.append(
            {
                "sample_id": row["sample_id"],
                "source_sample_id": row["sample_id"],
                "split": "heldout",
                "clean_verb": row["clean_verb"],
                "corrupt_verb": row["corrupt_verb"],
                "source_dataset": row["dataset"],
                "source_language": row["language"],
                "clean_relpath": f"clean/{row['sample_id']}.txt",
                "corrupt_relpath": f"corrupt/{row['sample_id']}.txt",
                "clean_prompt_sha256": sha256(clean_path),
                "corrupt_prompt_sha256": sha256(corrupt_path),
            }
        )
    write_jsonl(dataset_root / "pairs.jsonl", pairs)
    return dataset_root


def subset_phase6_cache(selected: list[dict]) -> Path:
    cache = torch.load(BASE_CACHE_PATH, map_location="cpu", weights_only=False)
    old_ids = list(cache["sample_ids"])
    selected_ids = [row["sample_id"] for row in selected]
    if len(old_ids) != 300 or len(set(old_ids)) != 300:
        raise ValueError("Unexpected historical phase6 cache sample index.")
    index_by_id = {sample_id: i for i, sample_id in enumerate(old_ids)}
    indices = torch.tensor([index_by_id[sample_id] for sample_id in selected_ids], dtype=torch.long)
    subset = {
        "sample_ids": selected_ids,
        "clean_candidates": [cache["clean_candidates"][i] for i in indices.tolist()],
        "corrupt_candidates": [cache["corrupt_candidates"][i] for i in indices.tolist()],
        "token_lengths": cache["token_lengths"][indices],
        "layers": list(cache["layers"]),
        "mlp_in": {
            side: {int(layer): value[indices] for layer, value in by_layer.items()}
            for side, by_layer in cache["mlp_in"].items()
        },
        "l33h29_out": {side: value[indices] for side, value in cache["l33h29_out"].items()},
        "tool_logit": {side: value[indices] for side, value in cache["tool_logit"].items()},
        "top1": {side: value[indices] for side, value in cache["top1"].items()},
        "tool_token_id": cache["tool_token_id"],
    }
    if subset["sample_ids"] != selected_ids:
        raise RuntimeError("Subset cache order mismatch.")
    path = OUT_ROOT / "phase6_cache_subset.pt"
    torch.save(subset, path)
    return path


def save_selection_artifacts(selected: list[dict], solver: dict, dataset_root: Path, cache_path: Path) -> None:
    records = []
    for row in selected:
        records.append(
            {
                "sample_id": row["sample_id"],
                "source_dataset": row["dataset"],
                "source_language": row["language"],
                "clean_verb": row["clean_verb"],
                "corrupt_verb": row["corrupt_verb"],
                "corrupt_baseline_margin": row["corrupt_baseline_margin"],
                "corrupt_margin_quintile": int(row["corrupt_margin_quintile"]),
            }
        )
    write_jsonl(OUT_ROOT / "selected_pairs.jsonl", records)
    payload = {
        "n_source_pairs": 300,
        "n_selected": 200,
        "seed": SEED,
        "selection_rule": "Integer-programmed minimum-hash-rank sample subject to fixed metadata margins; no post-intervention outcome was used.",
        "balanced_variables": ["source_dataset", "source_language", "clean_verb", "corrupt_verb", "corrupt_baseline_margin_quintile"],
        "quotas": QUOTAS,
        "solver": solver,
        "input_manifests": {
            "clean_sha256": sha256(OLD_DATA_ROOT / "clean/manifest.jsonl"),
            "corrupt_sha256": sha256(OLD_DATA_ROOT / "corrupt/manifest.jsonl"),
        },
        "input_mlp_baseline_csv_sha256": sha256(BASE_MLP_CSV),
        "dataset_root": str(dataset_root),
        "subset_phase6_cache": str(cache_path),
        "selected_counts": {
            "source_dataset": dict(Counter(row["dataset"] for row in selected)),
            "source_language": dict(Counter(row["language"] for row in selected)),
            "clean_verb": dict(Counter(row["clean_verb"] for row in selected)),
            "corrupt_verb": dict(Counter(row["corrupt_verb"] for row in selected)),
            "corrupt_baseline_margin_quintile": dict(Counter(row["corrupt_margin_quintile"] for row in selected)),
        },
    }
    (OUT_ROOT / "sampling_protocol.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def save_mlp_subset(selected: list[dict], mlp_rows: dict[str, dict]) -> None:
    rows = [mlp_rows[row["sample_id"]] for row in selected]
    tool_id = "151657"
    n = len(rows)
    baseline_top1 = sum(row["corrupt_top1"] == tool_id for row in rows)
    patched_top1 = sum(row["prediction_position_top1"] == tool_id for row in rows)
    strict = sum(
        row["corrupt_top1"] != tool_id and row["prediction_position_top1"] == tool_id
        for row in rows
    )
    summary = {
        "n_pairs": n,
        "source": "Subset of the already completed historical v2 300-pair prediction-position MLP34 patch run; no model rerun required.",
        "baseline_corrupt_tool_top1_count": baseline_top1,
        "baseline_corrupt_tool_top1_rate": baseline_top1 / n,
        "patched_tool_top1_count": patched_top1,
        "patched_tool_top1_rate": patched_top1 / n,
        "strict_recovery_count": strict,
        "strict_recovery_rate": strict / n,
        "mean_tool_logit_delta": sum(float(row["prediction_position_tool_logit"]) - float(row["corrupt_tool_logit"]) for row in rows) / n,
        "mean_margin_delta": sum(float(row["prediction_position_margin"]) - float(row["corrupt_margin"]) for row in rows) / n,
    }
    output = OUT_ROOT / "mlp34_patching"
    output.mkdir(parents=True, exist_ok=True)
    with (output / "mlp34_patch_per_sample.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / "mlp34_patch_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    if OUT_ROOT.exists():
        raise FileExistsError(f"Refusing to overwrite existing sampling artifacts: {OUT_ROOT}")
    clean_rows = read_jsonl(OLD_DATA_ROOT / "clean/manifest.jsonl")
    corrupt_rows = read_jsonl(OLD_DATA_ROOT / "corrupt/manifest.jsonl")
    with BASE_MLP_CSV.open(newline="", encoding="utf-8") as handle:
        mlp_rows = {row["sample_id"]: row for row in csv.DictReader(handle)}
    selected, solver = select_rows(clean_rows, corrupt_rows, mlp_rows)
    OUT_ROOT.mkdir(parents=True, exist_ok=False)
    dataset_root = save_selected_dataset(selected, clean_rows, corrupt_rows)
    cache_path = subset_phase6_cache(selected)
    save_selection_artifacts(selected, solver, dataset_root, cache_path)
    save_mlp_subset(selected, mlp_rows)
    print(json.dumps({"output_root": str(OUT_ROOT), "n_selected": len(selected), "solver": solver}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
