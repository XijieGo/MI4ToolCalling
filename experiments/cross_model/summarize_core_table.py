#!/usr/bin/env python3
"""Generate the paper's complete cross-model table from completed artifacts."""
from __future__ import annotations

import argparse
import csv
import json
from decimal import Decimal, ROUND_HALF_UP
from fractions import Fraction
from pathlib import Path

from result_metadata import transcoder_metadata, transcoder_notes

ROOT = Path(__file__).resolve().parents[2]
MODELS = (
    ("qwen3_4b", "Qwen3-4B"), ("qwen3_8b", "Qwen3-8B"),
    ("qwen3_14b", "Qwen3-14B"), ("qwen35_4b", "Qwen3.5-4B"),
    ("qwen35_9b", "Qwen3.5-9B"), ("mistral_3p2_24b", "Mistral-3.2-24B"),
    ("granite_3p3_8b", "Granite-3.3-8B"),
)
HEADER = (
    r"Model & $l$ & $r(l,p)$ (\%) & Suff. & Necc. & Domains & $\tau^2$ & "
    r"Verb-free & MLP/Attn & $K_{\mathrm{corrupt}}/K_{\mathrm{clean}}$ & Max Attn (pp) \\"
)


def read(path: Path):
    return json.loads(path.read_text())


def conditional_rate(arm: dict, intervention: str) -> Fraction:
    """Recover integer flip counts; serialized rates have float32 rounding."""
    key, count = (("suppression_among_calls", "n_baseline_call")
                  if intervention == "suppression"
                  else ("induction_among_quiet", "n_baseline_quiet"))
    denominator, rate = int(arm[count]), arm[key]
    if denominator == 0 or rate is None:
        raise ValueError(f"Undefined {intervention} rate")
    numerator = round(rate * denominator)
    exact = Fraction(numerator, denominator)
    if not 0 <= numerator <= denominator or abs(float(exact)-rate) > 1e-6:
        raise ValueError("Rate cannot be reconstructed from its baseline count")
    return exact


def formatted(value, digits: int) -> str:
    if value is None:
        return "--"
    number = (Decimal(value.numerator)/Decimal(value.denominator)
              if isinstance(value, Fraction) else Decimal(str(value)))
    return str(number.quantize(Decimal(1).scaleb(-digits), rounding=ROUND_HALF_UP))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    state = read(run/"status.json")
    if not all(t["state"] == "completed" for t in state["tasks"]) or not (run/"completed").exists():
        raise RuntimeError("Complete the rerun and verified measured summary first")
    fixed = read(run/"fixed_layers.json")
    output = ROOT/"results"
    rows, detailed = [], []
    tex = [HEADER, r"\midrule"]
    for key, label in MODELS:
        paths = {
            "vector": output/"tool_call_vector"/key/"summary.json",
            "formation": output/key/"formation_transcoder/formation_transcoder_summary.json",
            "transfer": output/"transfer"/key/"summary.json",
            "readout": output/key/"downstream_readout/feature_readout.json",
        }
        v, f, t = (read(paths[k]) for k in ("vector","formation","transfer"))
        if (v["layer"],v["hook"]) != (fixed[key]["layer"],fixed[key]["hook"]):
            raise ValueError(f"Fixed layer/hook changed for {key}")
        domains = t["multi_domain"]["domains"]
        domain_rate = sum(
            (conditional_rate(r["induction"],"induction") +
             conditional_rate(r["suppression"],"suppression"))
            for r in domains.values()
        ) / (2*len(domains))
        tau_rate = (conditional_rate(t["tau2"]["removal"],"suppression") +
                    conditional_rate(t["tau2"]["induction"],"induction"))/2
        if abs(float(tau_rate)-t["tau2"]["score"]) > 1e-6:
            raise ValueError(f"Tau2 score disagrees with measured arms for {key}")
        verb_rate = conditional_rate(t["verb_free"]["alpha_1"],"suppression")
        tc = transcoder_metadata(f.get("transcoder", {}))
        if key == "qwen3_8b":
            form = f["formation_window_summary"]
            mlp = form["total_mlp_write"]/form["total_attn_write"]
            k_ratio = sum(r["K_corrupt"] for r in f["heldout_table5"])/sum(r["K_clean"] for r in f["heldout_table5"])
            paths["readout"] = output/"formation_readout"/key/"summary.json"
            attention = read(paths["readout"])["max_attn_pp"]
            tc = {"status": "reference", "source": "heldout_table5",
                  "K_corrupt_over_K_clean": k_ratio}
        else:
            form = f["formation"]
            mlp = form["mlp_attn_ratio"]
            k_ratio = f["transcoder"]["K_corrupt_over_K_clean"]
            attention = read(paths["readout"])["max_attention_shift_pp"]
        notes = transcoder_notes(tc)
        if key.startswith("qwen35"):
            mlp = None
            notes.append("The hybrid architecture uses the component convention in the paper caption.")
        first = v["layers"][0]
        values = [label,str(v["layer"]),formatted(first["r_lp"]*100,1),
                  formatted(first["suff"],2),formatted(first["necc"],2),
                  formatted(domain_rate*100,1),formatted(tau_rate*100,1),
                  formatted(verb_rate*100,1),formatted(mlp,2),
                  formatted(k_ratio,2),formatted(attention,1)]
        row = dict(zip(("Model","l","r(l,p) (%)","Suff.","Necc.","Domains (%)",
                        "tau2 (%)","Verb-free (%)","MLP/Attn",
                        "K_corrupt/K_clean","Max Attn (pp)"),values))
        rows.append(row)
        tex_values = values.copy()
        if tc.get("missing_window_layers"):
            tex_values[9] += r"$^{\dagger}$"
        if tc.get("reconstruction_flagged_window_layers"):
            tex_values[9] += r"$^{\ddagger}$"
        tex.append(" & ".join(tex_values) + r" \\")
        detailed.append({
            "model_key": key, "display": row, "layer_hook": fixed[key],
            "n_train": v["n_train"], "n_heldout": v["n_heldout"],
            "exact_conditional_rates": {
                "domains": str(domain_rate), "tau2": str(tau_rate), "verb_free": str(verb_rate)},
            "MLP_Attn_measured": form.get("mlp_attn_ratio",mlp),
            "K_ratio": k_ratio, "max_attn_pp": attention,
            "K_metadata": tc,
            "sources": {k:str(p.relative_to(ROOT)) for k,p in paths.items()}, "notes": notes,
        })
    tex.extend([
        r"\bottomrule",
        "% Domains: mean of the six conditional induction/suppression rates; all gains are 1.",
        "% Tau2: mean of its two conditional arm rates. Verb-free: suppression at gain 1.",
        "% Max Attn: clean-minus-corrupt target-scaffold attention, not vector-induced shift.",
        "% Display rounding uses ROUND_HALF_UP on exact decision counts where available.",
    ])
    if any(row["K_metadata"].get("missing_window_layers") for row in detailed):
        tex.append("% dagger: K checkpoint layers and the fixed formation window are listed in core_cross_model_table.json.")
    if any(row["K_metadata"].get("reconstruction_flagged_window_layers") for row in detailed):
        tex.append("% ddagger: training reconstruction VE < 0 or undefined; K aggregates all supplied fixed-window layers.")
    (output/"core_cross_model_table.tex").write_text("\n".join(tex)+"\n")
    with (output/"core_cross_model_table.csv").open("w",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    (output/"core_cross_model_table.json").write_text(json.dumps(
        {"run_dir":str(run),"rows":detailed},indent=2,ensure_ascii=False,allow_nan=False)+"\n")
    from summarize_cross_scale import render_table6
    render_table6({"run_dir": str(run), "rows": detailed}, output/"cross_scale_table6_summary.md")
    print("\n".join(tex[:9]))


if __name__ == "__main__":
    main()
