#!/usr/bin/env python3
"""Build the rerun's tables strictly from per-model experiment artifacts."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path

from result_metadata import transcoder_metadata, transcoder_notes

ROOT = Path(__file__).resolve().parents[2]
MODELS = ("qwen3_4b", "qwen3_8b", "qwen3_14b", "qwen35_4b", "qwen35_9b", "granite_3p3_8b", "mistral_3p2_24b")


def read(path: Path):
    if not path.is_file():
        return None
    return json.loads(path.read_text())


def fmt(value, places=3):
    return "—" if value is None else f"{value:.{places}f}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    status = read(run_dir / "status.json")
    if not status or not all(t["state"] == "completed" for t in status["tasks"]):
        raise RuntimeError("Complete all rerun tasks before publishing the measured summary")
    snapshot = read(run_dir / "qwen3_8b_preserved_sha256.json")
    # Understand either a direct mapping or the explicit files wrapper.
    files = snapshot.get("files", snapshot)
    if isinstance(files, list):
        files = {r["path"]: r["sha256"] for r in files}
    for relative, expected in files.items():
        path = ROOT / relative
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError(f"Protected Qwen3-8B artifact changed: {relative}")
    models, rows = {}, []
    lines = ["# Full benchmark summary", "", "Seven-model benchmark: train 300 / held-out 200.",
             "Vector layers and hooks are specified in this run's fixed_layers.json.",
             "K sources distinguish reference formation summaries from released Transcoder checkpoint measurements.", "",
             "| Model | Layer/hook | Suff. | Necc. | MLP/Attn | K corrupt/clean | K source | Attention shift (pp) |", "|---|---|---:|---:|---:|---:|---|---:|"]
    for model in MODELS:
        vector_path = ROOT / "results/tool_call_vector" / model / "summary.json"
        formation_path = ROOT / "results" / model / "formation_transcoder/formation_transcoder_summary.json"
        readout_path = ROOT / "results" / model / "downstream_readout/feature_readout.json"
        if model == "qwen3_8b":
            readout_path = ROOT / "results" / model / "downstream_readout/mlp34_feature_readout.json"
        scaffold_path = ROOT / "results" / model / "scaffold_ablation/scaffold_ablation_summary.json"
        transfer_path = ROOT / "results/transfer" / model / "summary.json"
        vector, formation, readout, scaffold, transfer = (read(p) for p in (vector_path, formation_path, readout_path, scaffold_path, transfer_path))
        if model != "qwen3_8b" and not all(x is not None for x in (vector, formation, readout, scaffold, transfer)):
            raise RuntimeError(f"Missing completed result for {model}")
        vector = vector or {}
        formation = formation or {}
        readout = readout or {}
        tc = transcoder_metadata(formation.get("transcoder", {}))
        form = formation.get("formation", {})
        first = (vector.get("layers") or [{}])[0]
        row = {"model": model, "result_status": "reference" if model == "qwen3_8b" else "completed",
               "layer": vector.get("layer"), "hook": vector.get("hook"), "n_train": vector.get("n_train"), "n_heldout": vector.get("n_heldout"),
               "suff": first.get("suff"), "necc": first.get("necc"), "mlp_attn_ratio": form.get("mlp_attn_ratio"),
               "K_corrupt_over_K_clean": tc.get("K_corrupt_over_K_clean"), "K_source": tc.get("status", "reference"),
               "max_attention_shift_pp": readout.get("max_attention_shift_pp")}
        if model == "qwen3_8b":
            # Read the Qwen3-8B per-layer contribution schema.
            old_table = formation.get("heldout_table5", [])
            kc = sum(float(r["K_corrupt"]) for r in old_table)
            ke = sum(float(r["K_clean"]) for r in old_table)
            row["K_corrupt_over_K_clean"] = kc / ke if ke else None
            row["K_source"] = "reference"
            old_form = formation.get("formation_window_summary", {})
            a, b = old_form.get("total_mlp_write"), old_form.get("total_attn_write")
            row["mlp_attn_ratio"] = a / b if a is not None and b else None
            combined = read(ROOT / "results/formation_readout/qwen3_8b/summary.json") or {}
            row["max_attention_shift_pp"] = combined.get("max_attn_pp")
        rows.append(row)
        models[model] = {"metrics": row, "artifacts": {"vector": str(vector_path.relative_to(ROOT)), "scaffold": str(scaffold_path.relative_to(ROOT)),
                                                       "formation": str(formation_path.relative_to(ROOT)), "readout": str(readout_path.relative_to(ROOT)), "transfer": str(transfer_path.relative_to(ROOT))},
                         "transcoder": tc, "transfer": transfer, "readout": readout}
        lines.append(f"| {model} | {row['layer']} {row['hook']} | {fmt(row['suff'])} | {fmt(row['necc'])} | {fmt(row['mlp_attn_ratio'])} | {fmt(row['K_corrupt_over_K_clean'])} | {row['K_source']} | {fmt(row['max_attention_shift_pp'], 2)} |")
    lines.extend(["", "Artifact paths and detailed results follow. Attention shifts use all 200 held-out pairs; feature selection uses train only.", ""])
    for model, result in models.items():
        lines.extend([f"## {model}", ""])
        for name, path in result["artifacts"].items():
            lines.append(f"- {name}: [{path}]({path.removeprefix('results/')})")
        tc = result["transcoder"]
        for note in transcoder_notes(tc):
            lines.append(f"- {note}")
        lines.append("")
    root = ROOT / "results"
    report = {"run_dir": str(run_dir), "fixed_layers": read(run_dir / "fixed_layers.json"), "qwen3_8b_preserved_files_verified": len(files), "models": models}
    (root / "FULL_BENCHMARK_MEASURED_SUMMARY.md").write_text("\n".join(lines)+"\n")
    (root / "FULL_BENCHMARK_MEASURED_SUMMARY.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)+"\n")
    with (root / "cross_scale_measured_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    combined_dir = root / "formation_readout"
    (combined_dir / "summary.json").write_text(json.dumps({"models": rows, "run_dir": str(run_dir)}, indent=2, ensure_ascii=False, allow_nan=False)+"\n")
    (combined_dir / "summary.md").write_text("\n".join(lines[:lines.index("Artifact paths and detailed results follow. Attention shifts use all 200 held-out pairs; feature selection uses train only.")])+"\n")
    (run_dir / "completed").write_text("All tasks completed and protected Qwen3-8B hashes verified.\n")
    print(f"Summary written; {len(files)} Qwen3-8B artifact hashes verified.")


if __name__ == "__main__":
    main()
