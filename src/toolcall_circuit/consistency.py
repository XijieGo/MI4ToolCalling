#!/usr/bin/env python3
"""
Evaluate cross-sample consistency and causal metrics for tool-call circuits.

Example:
python scripts/evaluate_toolcall_consistency.py \
  --base-root results/manual_run/batch \
  --variant refined=results/manual_run/refined_consistent \
  --output results/manual_run/consistency_eval.json
"""

from __future__ import annotations

import os
import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from toolcall_circuit.dataset import load_summary_records
from toolcall_circuit.paths import RESULTS_ROOT


def median(xs: Sequence[float]) -> float:
    return float(np.median(xs)) if xs else float("nan")


def mean(xs: Sequence[float]) -> float:
    return float(np.mean(xs)) if xs else float("nan")


def jaccard(a: Sequence[str], b: Sequence[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def pairwise_jaccard(nodes: Sequence[Sequence[str]]) -> Tuple[float, float]:
    vals: List[float] = []
    n = len(nodes)
    for i in range(n):
        for j in range(i + 1, n):
            vals.append(jaccard(nodes[i], nodes[j]))
    if not vals:
        return float("nan"), float("nan")
    return float(np.mean(vals)), float(np.median(vals))


def finite(xs: Sequence[float]) -> List[float]:
    return [float(x) for x in xs if isinstance(x, (int, float)) and math.isfinite(float(x))]


def bootstrap_ci(
    records: Sequence[Dict[str, object]],
    stat_fn,
    n_boot: int,
    seed: int,
) -> Dict[str, float]:
    rng = random.Random(seed)
    n = len(records)
    if n == 0:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan")}
    vals: List[float] = []
    for _ in range(n_boot):
        sample = [records[rng.randrange(n)] for __ in range(n)]
        vals.append(float(stat_fn(sample)))
    vals.sort()
    lo_idx = max(0, int(0.025 * n_boot))
    hi_idx = min(n_boot - 1, int(0.975 * n_boot))
    return {"mean": float(np.mean(vals)), "lo": vals[lo_idx], "hi": vals[hi_idx]}


def load_base_records(base_root: Path, gap_min: float) -> Dict[str, Dict[str, object]]:
    out: Dict[str, Dict[str, object]] = {}
    for summary_record in load_summary_records(base_root):
        summary = summary_record.summary
        gap = float(summary.get("gap", float("nan")))
        if not math.isfinite(gap) or gap <= gap_min:
            continue
        suff = float(summary.get("detailed_ratio_vs_gap", float("nan")))
        nec = float(summary.get("necessity_ratio_vs_gap", float("nan")))
        if not (math.isfinite(suff) and math.isfinite(nec)):
            continue
        sample_id = summary_record.sample_id
        out[sample_id] = {
            "sample_id": sample_id,
            "sample_rank": summary_record.sample_rank,
            "q_index": summary_record.legacy_index,
            "gap": gap,
            "suff": suff,
            "nec": nec,
            "nodes": list(summary.get("detailed_nodes", [])),
            "node_count": len(summary.get("detailed_nodes", [])),
            "k_local": None,
        }
    return out


def load_variant_records(
    variant_root: Path,
    sample_ids: Sequence[str],
) -> Dict[str, Dict[str, object]]:
    out: Dict[str, Dict[str, object]] = {}
    for sample_id in sample_ids:
        path = variant_root / sample_id / "summary_refined.json"
        if not path.exists():
            continue
        summary = json.loads(path.read_text(encoding="utf-8"))
        suff = float(summary.get("refined_suff_ratio_vs_gap", float("nan")))
        nec = float(summary.get("refined_necessity_ratio_vs_gap", float("nan")))
        if not (math.isfinite(suff) and math.isfinite(nec)):
            continue
        nodes = list(summary.get("refined_nodes", []))
        out[sample_id] = {
            "sample_id": sample_id,
            "sample_rank": summary.get("sample_rank"),
            "q_index": summary.get("q_index"),
            "suff": suff,
            "nec": nec,
            "nodes": nodes,
            "node_count": len(nodes),
            "k_local": summary.get("k_local"),
        }
    return out


def summarize(records: Sequence[Dict[str, object]], n_boot: int, seed: int) -> Dict[str, object]:
    suff = finite([r["suff"] for r in records])
    nec = finite([r["nec"] for r in records])
    nodes = [list(r["nodes"]) for r in records]
    node_counts = finite([r["node_count"] for r in records])
    ks = finite([r["k_local"] for r in records if r.get("k_local") is not None])
    j_mean, j_median = pairwise_jaccard(nodes)
    k_hist: Dict[str, int] = {}
    for k in ks:
        k_hist[str(int(k))] = k_hist.get(str(int(k)), 0) + 1

    return {
        "n_samples": len(records),
        "suff": {
            "median": median(suff),
            "mean": mean(suff),
            "bootstrap_median": bootstrap_ci(records, lambda rs: median([float(x["suff"]) for x in rs]), n_boot, seed),
        },
        "nec": {
            "median": median(nec),
            "mean": mean(nec),
            "bootstrap_median": bootstrap_ci(records, lambda rs: median([float(x["nec"]) for x in rs]), n_boot, seed + 1),
        },
        "consistency": {
            "pairwise_jaccard_mean": j_mean,
            "pairwise_jaccard_median": j_median,
            "bootstrap_jaccard_mean": bootstrap_ci(
                records,
                lambda rs: pairwise_jaccard([list(x["nodes"]) for x in rs])[0],
                n_boot,
                seed + 2,
            ),
        },
        "node_count": {"median": median(node_counts), "mean": mean(node_counts)},
        "k_local": {"mean": mean(ks), "hist": k_hist} if ks else None,
    }


def build_delta(
    base_records: Sequence[Dict[str, object]],
    variant_records: Sequence[Dict[str, object]],
    n_boot: int,
    seed: int,
) -> Dict[str, object]:
    base_by_sample = {str(r["sample_id"]): r for r in base_records}
    var_by_sample = {str(r["sample_id"]): r for r in variant_records}
    shared = sorted(set(base_by_sample) & set(var_by_sample))
    paired = [(base_by_sample[sample_id], var_by_sample[sample_id]) for sample_id in shared]
    if not paired:
        return {"n_shared": 0}

    def d_suff(ps) -> float:
        return median([float(v["suff"]) for _, v in ps]) - median([float(b["suff"]) for b, _ in ps])

    def d_nec(ps) -> float:
        return median([float(v["nec"]) for _, v in ps]) - median([float(b["nec"]) for b, _ in ps])

    def d_j(ps) -> float:
        b_nodes = [list(b["nodes"]) for b, _ in ps]
        v_nodes = [list(v["nodes"]) for _, v in ps]
        return pairwise_jaccard(v_nodes)[0] - pairwise_jaccard(b_nodes)[0]

    rng = random.Random(seed)
    boot_vals_s: List[float] = []
    boot_vals_n: List[float] = []
    boot_vals_j: List[float] = []
    n = len(paired)
    for _ in range(n_boot):
        sample = [paired[rng.randrange(n)] for __ in range(n)]
        boot_vals_s.append(float(d_suff(sample)))
        boot_vals_n.append(float(d_nec(sample)))
        boot_vals_j.append(float(d_j(sample)))
    boot_vals_s.sort()
    boot_vals_n.sort()
    boot_vals_j.sort()
    lo = int(0.025 * n_boot)
    hi = min(n_boot - 1, int(0.975 * n_boot))

    return {
        "n_shared": n,
        "delta_suff_median": float(d_suff(paired)),
        "delta_nec_median": float(d_nec(paired)),
        "delta_jaccard_mean": float(d_j(paired)),
        "bootstrap": {
            "delta_suff_median": {"mean": float(np.mean(boot_vals_s)), "lo": boot_vals_s[lo], "hi": boot_vals_s[hi]},
            "delta_nec_median": {"mean": float(np.mean(boot_vals_n)), "lo": boot_vals_n[lo], "hi": boot_vals_n[hi]},
            "delta_jaccard_mean": {"mean": float(np.mean(boot_vals_j)), "lo": boot_vals_j[lo], "hi": boot_vals_j[hi]},
        },
    }


def parse_variant_arg(raw: str) -> Tuple[str, Path]:
    if "=" not in raw:
        raise ValueError(f"Variant must be LABEL=PATH, got: {raw}")
    label, path = raw.split("=", 1)
    label = label.strip()
    path = path.strip()
    if not label or not path:
        raise ValueError(f"Variant must be LABEL=PATH, got: {raw}")
    return label, Path(path).resolve()


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate consistency/causal tradeoff for tool-call circuits.")
    parser.add_argument("--base-root", type=str, default=str(RESULTS_ROOT / "manual_run" / "batch"))
    parser.add_argument(
        "--variant",
        action="append",
        default=[],
        help="Variant in LABEL=PATH format. Can pass multiple times.",
    )
    parser.add_argument("--gap-min", type=float, default=0.5)
    parser.add_argument("--bootstrap", type=int, default=500)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output", type=str, default=str(RESULTS_ROOT / "manual_run" / "consistency_eval.json"))
    args = parser.parse_args()

    base_root = Path(args.base_root).resolve()
    base_records_map = load_base_records(base_root, gap_min=args.gap_min)
    sample_ids = sorted(base_records_map.keys())
    base_records = [base_records_map[sample_id] for sample_id in sample_ids]

    report: Dict[str, object] = {
        "base_root": str(base_root),
        "gap_min": args.gap_min,
        "bootstrap": args.bootstrap,
        "seed": args.seed,
        "base": summarize(base_records, n_boot=args.bootstrap, seed=args.seed),
        "variants": {},
        "deltas_vs_base": {},
    }

    for raw_variant in args.variant:
        label, root = parse_variant_arg(raw_variant)
        var_map = load_variant_records(root, sample_ids=sample_ids)
        shared_ids = sorted(set(var_map) & set(base_records_map))
        var_records = [var_map[sample_id] for sample_id in shared_ids]
        base_shared = [base_records_map[sample_id] for sample_id in shared_ids]
        report["variants"][label] = {
            "root": str(root),
            "summary": summarize(var_records, n_boot=args.bootstrap, seed=args.seed + 11),
        }
        report["deltas_vs_base"][label] = build_delta(
            base_records=base_shared,
            variant_records=var_records,
            n_boot=args.bootstrap,
            seed=args.seed + 23,
        )

    out_path = Path(args.output).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] wrote: {out_path}")


if __name__ == "__main__":
    main()
