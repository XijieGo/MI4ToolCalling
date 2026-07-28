#!/usr/bin/env python3
"""Take the cross-scale intersection and materialize a 400/100 common dataset."""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .common import read_jsonl, stable_rank, write_json, write_jsonl


def parse_screen(raw: str) -> tuple[str, Path]:
    if "=" not in raw:
        raise argparse.ArgumentTypeError("Use LABEL=PATH for --screen")
    label, path = raw.split("=", 1)
    if not label or not path:
        raise argparse.ArgumentTypeError("Screen LABEL and PATH must be non-empty")
    return label, Path(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary-selection", type=Path, required=True)
    parser.add_argument("--screen", action="append", type=parse_screen, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--domain", type=str, required=True)
    parser.add_argument("--target-count", type=int, default=500)
    parser.add_argument("--train-count", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--exact-grid",
        action="store_true",
        help="Select an equal count from every requested clean×corrupt verb cell.",
    )
    parser.add_argument("--grid-clean-verbs", nargs="+", default=None)
    parser.add_argument("--grid-corrupt-verbs", nargs="+", default=None)
    return parser.parse_args()


def load_screens(specs: list[tuple[str, Path]]) -> dict[str, dict[str, dict[str, Any]]]:
    result: dict[str, dict[str, dict[str, Any]]] = {}
    for label, path in specs:
        rows = {row["candidate_id"]: row for row in read_jsonl(path.resolve())}
        result[label] = rows
    return result


def crossscale_stability_margin(row: dict[str, Any]) -> float:
    """Worst decision margin across every model required by the intersection."""

    return min(
        min(float(screen["clean_tool_margin"]), float(screen["corrupt_non_tool_margin"]))
        for screen in row["crossscale_screens"].values()
    )


def choose(rows: list[dict[str, Any]], *, target_count: int, seed: int) -> list[dict[str, Any]]:
    # Primary selection already guarantees one pair per source.  Rebalance the
    # retained intersection rather than simply taking its file order.
    ordered = sorted(
        rows,
        key=lambda row: (
            -float(row["crossscale_min_stability_margin"]),
            stable_rank(seed, row["candidate_id"]),
            row["candidate_id"],
        ),
    )
    combo_counts: Counter[tuple[str, str]] = Counter()
    clean_counts: Counter[str] = Counter()
    corrupt_counts: Counter[str] = Counter()
    remaining = list(ordered)
    selected: list[dict[str, Any]] = []
    while remaining and len(selected) < target_count:
        chosen = min(
            remaining,
            key=lambda row: (
                -float(row["crossscale_min_stability_margin"]),
                combo_counts[(row["clean_verb"], row["corrupt_verb"])],
                clean_counts[row["clean_verb"]],
                corrupt_counts[row["corrupt_verb"]],
                stable_rank(seed, row["candidate_id"]),
            ),
        )
        remaining.remove(chosen)
        selected.append(chosen)
        combo_counts[(chosen["clean_verb"], chosen["corrupt_verb"])] += 1
        clean_counts[chosen["clean_verb"]] += 1
        corrupt_counts[chosen["corrupt_verb"]] += 1
    if len(selected) < target_count:
        raise RuntimeError(f"Only {len(selected)} cross-scale-valid pairs; need {target_count}")
    return selected


def choose_exact_grid(
    rows: list[dict[str, Any]],
    *,
    target_count: int,
    clean_verbs: list[str],
    corrupt_verbs: list[str],
    seed: int,
) -> list[dict[str, Any]]:
    """Take an exact full grid from the cross-scale behavioral intersection."""

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
        raise RuntimeError(f"Insufficient cross-scale-valid candidates for an exact grid: {missing}")

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
        costs.append(-float(row["crossscale_min_stability_margin"]) + 1e-8 * rank)
    result = milp(
        c=np.asarray(costs, dtype=float),
        integrality=np.ones(len(grid_rows), dtype=int),
        bounds=Bounds(0.0, 1.0),
        constraints=LinearConstraint(matrix.tocsr(), np.asarray(lower), np.asarray(upper)),
    )
    if result.status != 0 or result.x is None:
        raise RuntimeError(f"Could not satisfy exact cross-scale grid: {result.message}")
    selected = [row for index, row in enumerate(grid_rows) if result.x[index] > 0.5]
    if len(selected) != target_count or len({str(row["source_id"]) for row in selected}) != target_count:
        raise AssertionError("Exact-grid selection did not return target-count source-distinct rows")
    expected = Counter({pair: per_cell for pair in combinations})
    observed = Counter((str(row["clean_verb"]), str(row["corrupt_verb"])) for row in selected)
    if observed != expected:
        raise AssertionError("Exact-grid selection did not meet every verb-pair quota")
    return selected


def split_groups(
    rows: list[dict[str, Any]],
    *,
    test_count: int,
    seed: int,
    exact_test_cell_count: int | None = None,
) -> dict[str, str]:
    """Make an exact-size train/test split without source-group leakage.

    The task bodies in D3 can share an entity, D4 rows can share a database,
    and D5 messages can share a thread.  A small subset-sum pass assigns each
    whole group to one split while retaining the requested 400/100 size.
    """

    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_group[str(row["source_group"])].append(row)
    if exact_test_cell_count is not None:
        try:
            import numpy as np
            from scipy.optimize import Bounds, LinearConstraint, milp
            from scipy.sparse import lil_matrix
        except ImportError as exc:  # pragma: no cover - scipy is a project dependency.
            raise RuntimeError("Exact-grid group split requires scipy") from exc

        cells = sorted({(str(row["clean_verb"]), str(row["corrupt_verb"])) for row in rows})
        expected_total = exact_test_cell_count * len(cells)
        if expected_total != test_count:
            raise ValueError(
                "Exact split cell count is incompatible with test count: "
                f"{exact_test_cell_count} * {len(cells)} != {test_count}"
            )
        groups = sorted(by_group, key=lambda group: (stable_rank(seed + 1, group), group))
        matrix = lil_matrix((1 + len(cells), len(groups)), dtype=float)
        cell_index = {cell: index for index, cell in enumerate(cells, start=1)}
        for column, group in enumerate(groups):
            matrix[0, column] = len(by_group[group])
            for row in by_group[group]:
                cell = (str(row["clean_verb"]), str(row["corrupt_verb"]))
                matrix[cell_index[cell], column] += 1.0
        lower = np.asarray([float(test_count)] + [float(exact_test_cell_count)] * len(cells))
        costs = np.asarray(
            [int(stable_rank(seed + 1, group)[:16], 16) / 2**64 for group in groups], dtype=float
        )
        result = milp(
            c=costs,
            integrality=np.ones(len(groups), dtype=int),
            bounds=Bounds(0.0, 1.0),
            constraints=LinearConstraint(matrix.tocsr(), lower, lower),
        )
        if result.status != 0 or result.x is None:
            raise RuntimeError(
                "Could not form an exact group-disjoint grid-balanced test split: " + result.message
            )
        test_groups = {groups[index] for index, value in enumerate(result.x) if value > 0.5}
        if sum(len(by_group[group]) for group in test_groups) != test_count:
            raise AssertionError("Exact grid split did not meet the test-count quota")
        observed = Counter(
            (str(row["clean_verb"]), str(row["corrupt_verb"]))
            for group in test_groups
            for row in by_group[group]
        )
        expected = Counter({cell: exact_test_cell_count for cell in cells})
        if observed != expected:
            raise AssertionError("Exact grid split did not meet every test-cell quota")
        return {
            row["candidate_id"]: "test" if str(row["source_group"]) in test_groups else "train"
            for row in rows
        }

    ordered_groups = sorted(by_group, key=lambda group: (stable_rank(seed + 1, group), group))
    reachable: dict[int, tuple[str, ...]] = {0: ()}
    for group in ordered_groups:
        size = len(by_group[group])
        for total, chosen in sorted(list(reachable.items()), reverse=True):
            next_total = total + size
            if next_total <= test_count and next_total not in reachable:
                reachable[next_total] = chosen + (group,)
    if test_count not in reachable:
        group_sizes = sorted(len(value) for value in by_group.values())
        raise RuntimeError(
            f"Could not form an exact group-disjoint test split of {test_count}; "
            f"group sizes are {group_sizes}"
        )
    test_groups = set(reachable[test_count])
    return {
        row["candidate_id"]: "test" if str(row["source_group"]) in test_groups else "train"
        for row in rows
    }


def prepare_output(root: Path, *, overwrite: bool) -> Path:
    if root.exists():
        if not overwrite:
            raise FileExistsError(f"Output exists: {root}; use --overwrite for this exact directory")
        import shutil

        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=False)
    return root


def main() -> None:
    args = parse_args()
    if args.train_count <= 0 or args.train_count >= args.target_count:
        raise ValueError("--train-count must be strictly between zero and --target-count")
    primary = list(read_jsonl(args.primary_selection.resolve()))
    screens = load_screens(args.screen)
    common: list[dict[str, Any]] = []
    for row in primary:
        per_model = {label: screen.get(row["candidate_id"]) for label, screen in screens.items()}
        if all(item is not None and item.get("behavior_valid") for item in per_model.values()):
            item = dict(row)
            item["crossscale_screens"] = per_model
            item["crossscale_min_stability_margin"] = crossscale_stability_margin(item)
            common.append(item)
    if args.exact_grid:
        if not args.grid_clean_verbs or not args.grid_corrupt_verbs:
            raise ValueError("--exact-grid requires --grid-clean-verbs and --grid-corrupt-verbs")
        selected = choose_exact_grid(
            common,
            target_count=args.target_count,
            clean_verbs=list(args.grid_clean_verbs),
            corrupt_verbs=list(args.grid_corrupt_verbs),
            seed=args.seed,
        )
        selection_policy = "exact_clean_x_corrupt_grid"
    else:
        selected = choose(common, target_count=args.target_count, seed=args.seed)
        selection_policy = "margin_first_greedy"
    test_count = args.target_count - args.train_count
    exact_test_cell_count: int | None = None
    if args.exact_grid:
        cell_count = len(args.grid_clean_verbs or []) * len(args.grid_corrupt_verbs or [])
        if cell_count and test_count % cell_count == 0:
            exact_test_cell_count = test_count // cell_count
    split_by_id = split_groups(
        selected,
        test_count=test_count,
        seed=args.seed,
        exact_test_cell_count=exact_test_cell_count,
    )
    for row in selected:
        row["split"] = split_by_id[row["candidate_id"]]

    root = prepare_output(args.output_root.resolve(), overwrite=args.overwrite)
    write_jsonl(root / "selected_pairs.jsonl", (row for row in selected))
    manifest_rows: list[dict[str, Any]] = []
    for number, row in enumerate(sorted(selected, key=lambda item: item["candidate_id"]), start=1):
        stem = f"{args.domain.lower()}_{number:04d}"
        for condition, prompt in (("clean", row["clean_prompt"]), ("corrupt", row["corrupt_prompt"])):
            path = root / row["split"] / condition / f"{stem}.txt"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(prompt, encoding="utf-8")
        manifest_rows.append(
            {
                "filename": f"{stem}.txt",
                "candidate_id": row["candidate_id"],
                "domain": row["domain"],
                "source_id": row["source_id"],
                "source_group": row["source_group"],
                "split": row["split"],
                "clean_verb": row["clean_verb"],
                "corrupt_verb": row["corrupt_verb"],
                "models": ";".join(sorted(screens)),
            }
        )
    with (root / "selection_manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0]))
        writer.writeheader()
        writer.writerows(manifest_rows)
    write_json(
        root / "selection_summary.json",
        {
            "domain": args.domain,
            "seed": args.seed,
            "primary_selection_count": len(primary),
            "crossscale_intersection_count": len(common),
            "selected_count": len(selected),
            "selection_policy": selection_policy,
            "grid_clean_verbs": args.grid_clean_verbs,
            "grid_corrupt_verbs": args.grid_corrupt_verbs,
            "selected_crossscale_stability_margin": {
                "minimum": min(float(row["crossscale_min_stability_margin"]) for row in selected),
                "median": sorted(float(row["crossscale_min_stability_margin"]) for row in selected)[len(selected) // 2],
            },
            "train_count": args.train_count,
            "test_count": args.target_count - args.train_count,
            "train_source_group_count": len({row["source_group"] for row in selected if row["split"] == "train"}),
            "test_source_group_count": len({row["source_group"] for row in selected if row["split"] == "test"}),
            "group_disjoint_split": True,
            "split_policy": (
                "group_disjoint_exact_verb_grid" if exact_test_cell_count is not None else "group_disjoint_exact_size"
            ),
            "models": sorted(screens),
            "clean_verb_counts": dict(Counter(row["clean_verb"] for row in selected)),
            "corrupt_verb_counts": dict(Counter(row["corrupt_verb"] for row in selected)),
            "verb_pair_counts": dict(Counter(f"{row['clean_verb']}/{row['corrupt_verb']}" for row in selected)),
            "test_verb_pair_counts": dict(
                Counter(f"{row['clean_verb']}/{row['corrupt_verb']}" for row in selected if row["split"] == "test")
            ),
        },
    )
    print(
        f"{args.domain}: intersection {len(common)}/{len(primary)}, "
        f"materialized {len(selected)} pairs ({args.train_count}/{args.target_count - args.train_count})"
    )


if __name__ == "__main__":
    main()
