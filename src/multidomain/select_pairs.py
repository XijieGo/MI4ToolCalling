#!/usr/bin/env python3
"""Select balanced behavior-valid pairs for the primary Qwen3-8B stage."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .common import read_jsonl, stable_rank, write_json, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--screen", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-count", type=int, default=900)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--domain", type=str, required=True)
    parser.add_argument(
        "--exact-grid",
        action="store_true",
        help="Jointly enforce equal counts for every requested clean×corrupt verb cell.",
    )
    parser.add_argument("--grid-clean-verbs", nargs="+", default=None)
    parser.add_argument("--grid-corrupt-verbs", nargs="+", default=None)
    return parser.parse_args()


def stability_margin(screen: dict[str, Any]) -> float:
    """Margin by which the pair satisfies both sides of the decision rule."""

    return min(float(screen["clean_tool_margin"]), float(screen["corrupt_non_tool_margin"]))


def choose_balanced(rows: list[dict[str, Any]], *, target_count: int, seed: int) -> list[dict[str, Any]]:
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_source[row["source_id"]].append(row)
    selected: list[dict[str, Any]] = []
    combo_counts: Counter[tuple[str, str]] = Counter()
    clean_counts: Counter[str] = Counter()
    corrupt_counts: Counter[str] = Counter()
    # First retain the most decisive valid source rows.  Verb diversity is a
    # tiebreaker, never a reason to exchange an unambiguous behavior pair for
    # a marginal top-1 flip.
    ordered_source_ids = sorted(
        by_source,
        key=lambda source_id: (
            -max(float(row["primary_stability_margin"]) for row in by_source[source_id]),
            stable_rank(seed, source_id),
            source_id,
        ),
    )
    for source_id in ordered_source_ids:
        options = by_source[source_id]
        chosen = min(
            options,
            key=lambda row: (
                -float(row["primary_stability_margin"]),
                combo_counts[(row["clean_verb"], row["corrupt_verb"])],
                clean_counts[row["clean_verb"]],
                corrupt_counts[row["corrupt_verb"]],
                stable_rank(seed, row["candidate_id"]),
            ),
        )
        selected.append(chosen)
        combo_counts[(chosen["clean_verb"], chosen["corrupt_verb"])] += 1
        clean_counts[chosen["clean_verb"]] += 1
        corrupt_counts[chosen["corrupt_verb"]] += 1
        if len(selected) == target_count:
            break
    if len(selected) < target_count:
        raise RuntimeError(f"Only {len(selected)} distinct source records are behavior-valid; need {target_count}")
    return selected


def choose_exact_grid(
    rows: list[dict[str, Any]],
    *,
    target_count: int,
    clean_verbs: list[str],
    corrupt_verbs: list[str],
    seed: int,
) -> list[dict[str, Any]]:
    """Select one source per row while enforcing an equal full verb grid."""

    try:
        import numpy as np
        from scipy.optimize import Bounds, LinearConstraint, milp
        from scipy.sparse import lil_matrix
    except ImportError as exc:  # pragma: no cover - scipy is a project dependency.
        raise RuntimeError("Exact-grid selection requires scipy") from exc

    combinations = [(clean, corrupt) for clean in clean_verbs for corrupt in corrupt_verbs]
    if not combinations or target_count <= 0 or target_count % len(combinations) != 0:
        raise ValueError(
            f"target-count={target_count} must be positive and divisible by the {len(combinations)} requested cells"
        )
    per_cell = target_count // len(combinations)
    grid_rows = [
        row
        for row in rows
        if (str(row["clean_verb"]), str(row["corrupt_verb"])) in combinations
    ]
    by_source: dict[str, list[int]] = defaultdict(list)
    by_combo: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, row in enumerate(grid_rows):
        pair = (str(row["clean_verb"]), str(row["corrupt_verb"]))
        by_source[str(row["source_id"])].append(index)
        by_combo[pair].append(index)
    missing = {f"{clean}/{corrupt}": len(by_combo[(clean, corrupt)]) for clean, corrupt in combinations if len(by_combo[(clean, corrupt)]) < per_cell}
    if missing:
        raise RuntimeError(f"Insufficient primary-model-valid candidates for an exact grid: {missing}")

    constraint_count = len(by_source) + len(combinations)
    matrix = lil_matrix((constraint_count, len(grid_rows)), dtype=float)
    lower: list[float] = []
    upper: list[float] = []
    row_index = 0
    for source_id in sorted(by_source):
        for column in by_source[source_id]:
            matrix[row_index, column] = 1.0
        lower.append(0.0)
        upper.append(1.0)
        row_index += 1
    for pair in combinations:
        for column in by_combo[pair]:
            matrix[row_index, column] = 1.0
        lower.append(float(per_cell))
        upper.append(float(per_cell))
        row_index += 1

    costs: list[float] = []
    for row in grid_rows:
        rank = int(stable_rank(seed, str(row["candidate_id"]))[:16], 16) / 2**64
        costs.append(-float(row["primary_stability_margin"]) + 1e-8 * rank)
    result = milp(
        c=np.asarray(costs, dtype=float),
        integrality=np.ones(len(grid_rows), dtype=int),
        bounds=Bounds(0.0, 1.0),
        constraints=LinearConstraint(matrix.tocsr(), np.asarray(lower), np.asarray(upper)),
    )
    if result.status != 0 or result.x is None:
        raise RuntimeError(f"Could not satisfy exact primary grid: {result.message}")
    selected = [row for index, row in enumerate(grid_rows) if result.x[index] > 0.5]
    if len(selected) != target_count or len({str(row["source_id"]) for row in selected}) != target_count:
        raise AssertionError("Exact-grid selection did not return target-count source-distinct rows")
    expected = Counter({pair: per_cell for pair in combinations})
    observed = Counter((str(row["clean_verb"]), str(row["corrupt_verb"])) for row in selected)
    if observed != expected:
        raise AssertionError("Exact-grid selection did not meet every verb-pair quota")
    return selected


def main() -> None:
    args = parse_args()
    candidates = {row["candidate_id"]: row for row in read_jsonl(args.candidates.resolve())}
    screened = list(read_jsonl(args.screen.resolve()))
    eligible: list[dict[str, Any]] = []
    for screen in screened:
        if not screen.get("behavior_valid"):
            continue
        candidate = candidates.get(screen["candidate_id"])
        if candidate is None:
            raise KeyError(f"Screen row is absent from candidates: {screen['candidate_id']}")
        if candidate["domain"] != args.domain:
            raise ValueError(f"Expected {args.domain}, got {candidate['domain']}")
        candidate = dict(candidate)
        candidate["primary_screen"] = screen
        candidate["primary_stability_margin"] = stability_margin(screen)
        eligible.append(candidate)
    if args.exact_grid:
        if not args.grid_clean_verbs or not args.grid_corrupt_verbs:
            raise ValueError("--exact-grid requires --grid-clean-verbs and --grid-corrupt-verbs")
        selected = choose_exact_grid(
            eligible,
            target_count=args.target_count,
            clean_verbs=list(args.grid_clean_verbs),
            corrupt_verbs=list(args.grid_corrupt_verbs),
            seed=args.seed,
        )
        selection_policy = "exact_clean_x_corrupt_grid"
    else:
        selected = choose_balanced(eligible, target_count=args.target_count, seed=args.seed)
        selection_policy = "margin_first_greedy"
    write_jsonl(args.output.resolve(), (row for row in selected))
    write_json(
        args.output.with_suffix(args.output.suffix + ".manifest.json"),
        {
            "domain": args.domain,
            "seed": args.seed,
            "target_count": args.target_count,
            "candidate_count": len(candidates),
            "behavior_valid_candidate_count": len(eligible),
            "selected_count": len(selected),
            "selection_policy": selection_policy,
            "grid_clean_verbs": args.grid_clean_verbs,
            "grid_corrupt_verbs": args.grid_corrupt_verbs,
            "selected_stability_margin": {
                "minimum": min(float(row["primary_stability_margin"]) for row in selected),
                "median": sorted(float(row["primary_stability_margin"]) for row in selected)[len(selected) // 2],
            },
            "clean_verb_counts": dict(Counter(row["clean_verb"] for row in selected)),
            "corrupt_verb_counts": dict(Counter(row["corrupt_verb"] for row in selected)),
            "verb_pair_counts": dict(Counter(f"{row['clean_verb']}/{row['corrupt_verb']}" for row in selected)),
        },
    )
    print(f"{args.domain}: selected {len(selected)} primary-model-valid pairs")


if __name__ == "__main__":
    main()
