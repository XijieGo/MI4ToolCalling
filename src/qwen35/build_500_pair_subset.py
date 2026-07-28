#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CANONICAL_PATH = PROJECT_ROOT / "results" / "Qwen3.5-9B" / "converted_dataset" / "canonical_pairs.jsonl"
CONVERTED_MANIFEST_PATH = PROJECT_ROOT / "results" / "Qwen3.5-9B" / "converted_dataset" / "manifest.jsonl"
BEHAVIOR_PATH = PROJECT_ROOT / "results" / "Qwen3.5-9B" / "behavior_scan" / "per_pair_results.jsonl"
OUTPUT_ROOT = PROJECT_ROOT / "results" / "Qwen3.5-9B" / "datasets"

CLEAN_CANDIDATES = ("add", "build", "complete", "save", "write")
CORRUPT_CANDIDATES = ("discuss", "explore", "inspect", "review", "study")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select a fresh behavior-valid, complete verb-grid Qwen3.5 subset."
    )
    parser.add_argument("--canonical-path", type=Path, default=CANONICAL_PATH)
    parser.add_argument("--converted-manifest-path", type=Path, default=CONVERTED_MANIFEST_PATH)
    parser.add_argument("--behavior-path", type=Path, default=BEHAVIOR_PATH)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument(
        "--model-label",
        type=str,
        default="Qwen3.5-9B",
        help="Model whose behavior scan defines the eligible pool; recorded in the selection manifest.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--target-pairs",
        type=int,
        default=0,
        help=(
            "Exact size with equal clean-verb and corrupt-verb marginals; 0 selects the largest "
            "complete clean×corrupt verb grid."
        ),
    )
    parser.add_argument(
        "--allow-relaxed-clean-balance",
        action="store_true",
        help=(
            "Keep exact corrupt-verb marginals but relax clean-verb equality when a model's "
            "behavior-valid pool cannot supply the uniform clean quota. The min-cost flow keeps "
            "clean counts as close to uniform as feasible."
        ),
    )
    parser.add_argument(
        "--min-clean-top1-margin",
        type=float,
        default=0.0,
        help=(
            "Require the clean prompt's <tool_call> top-1 probability margin to be at least this value. "
            "This can exclude numerically borderline behavior-pass examples."
        ),
    )
    parser.add_argument(
        "--min-corrupt-top1-margin",
        type=float,
        default=0.0,
        help=(
            "Require the corrupt prompt's non-<tool_call> top-1 probability margin to be at least this value. "
            "This can exclude numerically borderline behavior-pass examples."
        ),
    )
    parser.add_argument(
        "--rank-by-min-top1-margin",
        action="store_true",
        help=(
            "Within each clean×corrupt cell, select the examples with the largest minimum of the clean and "
            "corrupt top-1 margins instead of choosing them randomly."
        ),
    )
    parser.add_argument(
        "--exclude-sample-id",
        action="append",
        default=[],
        help=(
            "Exclude a source sample ID from the eligible pool. Repeat this option when an exact "
            "causal-batch baseline audit identifies a numerically unstable otherwise-valid example."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace only this selector's clean_*.txt, corrupt_*.txt and manifest outputs.",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def first_word_and_tail(text: str) -> tuple[str, str]:
    head, sep, tail = text.partition(" ")
    if not sep:
        raise ValueError(f"Could not split first word from text: {text!r}")
    return head, tail


def verify_pair_equivalence(
    clean_user: str,
    corrupt_user: str,
    clean_prompt: str,
    corrupt_prompt: str,
) -> None:
    clean_word, clean_tail = first_word_and_tail(clean_user)
    corrupt_word, corrupt_tail = first_word_and_tail(corrupt_user)
    if clean_tail != corrupt_tail:
        raise ValueError("Clean/corrupt user contents differ beyond the first word.")
    clean_idx = clean_prompt.find(clean_user)
    corrupt_idx = corrupt_prompt.find(corrupt_user)
    if clean_idx < 0 or corrupt_idx < 0:
        raise ValueError("Failed to locate user content inside converted prompts.")
    clean_prefix = clean_prompt[:clean_idx]
    corrupt_prefix = corrupt_prompt[:corrupt_idx]
    clean_suffix = clean_prompt[clean_idx + len(clean_user) :]
    corrupt_suffix = corrupt_prompt[corrupt_idx + len(corrupt_user) :]
    if clean_prefix != corrupt_prefix or clean_suffix != corrupt_suffix:
        raise ValueError("Converted clean/corrupt prompts differ outside the user-content verb span.")
    if clean_word.lower() == corrupt_word.lower():
        raise ValueError("Clean/corrupt first words are identical; expected a verb flip.")


def prepare_output_root(output_root: Path, *, overwrite: bool) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    targets = [*output_root.glob("clean_*.txt"), *output_root.glob("corrupt_*.txt")]
    targets.extend(output_root / name for name in ("manifest.jsonl", "selection_summary.json"))
    existing = [path for path in targets if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            f"Selection output already exists under {output_root}; use --overwrite for this explicit target."
        )
    for path in existing:
        path.unlink()


def solve_balanced_marginal_counts(
    candidates: dict[tuple[str, str], list[dict[str, Any]]],
    *,
    target_pairs: int,
) -> tuple[dict[tuple[str, str], int], int]:
    """Allocate a feasible, near-uniform 5×5 grid with exactly balanced marginals.

    The v2 prompt set is deliberately not uniform over verb pairs, so a fixed
    20-per-cell selection can be infeasible even when there are far more than
    500 behavior-valid pairs. A small min-cost flow chooses a feasible matrix
    while minimizing squared deviation from the uniform per-cell target.
    """
    combinations = [(clean, corrupt) for clean in CLEAN_CANDIDATES for corrupt in CORRUPT_CANDIDATES]
    n_clean = len(CLEAN_CANDIDATES)
    n_corrupt = len(CORRUPT_CANDIDATES)
    n_cells = len(combinations)
    if target_pairs % n_clean != 0 or target_pairs % n_corrupt != 0:
        raise ValueError(
            f"target_pairs={target_pairs} cannot give equal marginals over {n_clean} clean and {n_corrupt} corrupt verbs"
        )
    if target_pairs % n_cells != 0:
        raise ValueError(
            f"target_pairs={target_pairs} must be divisible by {n_cells} to define the near-uniform cell target"
        )

    clean_quota = target_pairs // n_clean
    corrupt_quota = target_pairs // n_corrupt
    uniform_cell_target = target_pairs // n_cells
    try:
        import networkx as nx
    except ImportError as exc:  # pragma: no cover - the project environment provides networkx transitively.
        raise RuntimeError("Balanced subset selection requires the installed networkx package.") from exc

    source = "source"
    sink = "sink"
    clean_nodes = {clean: f"clean::{clean}" for clean in CLEAN_CANDIDATES}
    corrupt_nodes = {corrupt: f"corrupt::{corrupt}" for corrupt in CORRUPT_CANDIDATES}
    graph = nx.MultiDiGraph()
    graph.add_node(source, demand=-target_pairs)
    graph.add_node(sink, demand=target_pairs)
    for clean in CLEAN_CANDIDATES:
        graph.add_edge(source, clean_nodes[clean], capacity=clean_quota, weight=0)
    for corrupt in CORRUPT_CANDIDATES:
        graph.add_edge(corrupt_nodes[corrupt], sink, capacity=corrupt_quota, weight=0)
    for clean, corrupt in combinations:
        # Unit-capacity parallel edges make the cost convex: the kth selected
        # example from a cell has the exact incremental squared-error cost.
        for unit_index in range(len(candidates[(clean, corrupt)])):
            incremental_cost = 2 * unit_index + 1 - 2 * uniform_cell_target
            graph.add_edge(
                clean_nodes[clean],
                corrupt_nodes[corrupt],
                capacity=1,
                weight=incremental_cost,
            )
    try:
        flow = nx.min_cost_flow(graph)
    except nx.NetworkXUnfeasible as exc:
        raise RuntimeError(
            "No behavior-valid subset can satisfy the requested balanced clean/corrupt marginals."
        ) from exc

    counts = {
        (clean, corrupt): int(sum(flow[clean_nodes[clean]][corrupt_nodes[corrupt]].values()))
        for clean, corrupt in combinations
    }
    if sum(counts.values()) != target_pairs:
        raise RuntimeError("Balanced selection did not reach the requested target size.")
    for clean in CLEAN_CANDIDATES:
        if sum(counts[(clean, corrupt)] for corrupt in CORRUPT_CANDIDATES) != clean_quota:
            raise RuntimeError(f"Clean marginal is not balanced for {clean}.")
    for corrupt in CORRUPT_CANDIDATES:
        if sum(counts[(clean, corrupt)] for clean in CLEAN_CANDIDATES) != corrupt_quota:
            raise RuntimeError(f"Corrupt marginal is not balanced for {corrupt}.")
    return counts, uniform_cell_target


def solve_relaxed_clean_marginal_counts(
    candidates: dict[tuple[str, str], list[dict[str, Any]]],
    *,
    target_pairs: int,
) -> tuple[dict[tuple[str, str], int], int, int]:
    """Select a strict-valid target when one clean verb cannot meet its equal quota.

    The corrupt verb marginals remain exactly equal.  Clean marginals are chosen by
    the flow itself, with convex costs that minimize their squared deviation from
    target_pairs / 5; cell counts are simultaneously kept near target_pairs / 25.
    """
    combinations = [(clean, corrupt) for clean in CLEAN_CANDIDATES for corrupt in CORRUPT_CANDIDATES]
    n_clean = len(CLEAN_CANDIDATES)
    n_corrupt = len(CORRUPT_CANDIDATES)
    if target_pairs % n_corrupt != 0 or target_pairs % (n_clean * n_corrupt) != 0:
        raise ValueError("Relaxed selection requires a target divisible by both 5 and 25.")
    try:
        import networkx as nx
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Balanced subset selection requires the installed networkx package.") from exc

    clean_target = target_pairs // n_clean
    corrupt_target = target_pairs // n_corrupt
    uniform_cell_target = target_pairs // (n_clean * n_corrupt)
    source = "source"
    sink = "sink"
    clean_nodes = {clean: f"clean::{clean}" for clean in CLEAN_CANDIDATES}
    corrupt_nodes = {corrupt: f"corrupt::{corrupt}" for corrupt in CORRUPT_CANDIDATES}
    graph = nx.MultiDiGraph()
    graph.add_node(source, demand=-target_pairs)
    graph.add_node(sink, demand=target_pairs)
    for clean in CLEAN_CANDIDATES:
        available = sum(len(candidates[(clean, corrupt)]) for corrupt in CORRUPT_CANDIDATES)
        for unit_index in range(available):
            # Incremental squared-error cost for choosing the next clean-side example.
            graph.add_edge(
                source,
                clean_nodes[clean],
                capacity=1,
                weight=2 * unit_index + 1 - 2 * clean_target,
            )
    for corrupt in CORRUPT_CANDIDATES:
        graph.add_edge(corrupt_nodes[corrupt], sink, capacity=corrupt_target, weight=0)
    for clean, corrupt in combinations:
        for unit_index in range(len(candidates[(clean, corrupt)])):
            # Also discourage an unnecessarily concentrated clean×corrupt grid.
            graph.add_edge(
                clean_nodes[clean],
                corrupt_nodes[corrupt],
                capacity=1,
                weight=2 * unit_index + 1 - 2 * uniform_cell_target,
            )
    try:
        flow = nx.min_cost_flow(graph)
    except nx.NetworkXUnfeasible as exc:
        raise RuntimeError("No behavior-valid subset can satisfy the requested corrupt marginals.") from exc
    counts = {
        (clean, corrupt): int(sum(flow[clean_nodes[clean]][corrupt_nodes[corrupt]].values()))
        for clean, corrupt in combinations
    }
    if sum(counts.values()) != target_pairs:
        raise RuntimeError("Relaxed selection did not reach the requested target size.")
    for corrupt in CORRUPT_CANDIDATES:
        if sum(counts[(clean, corrupt)] for clean in CLEAN_CANDIDATES) != corrupt_target:
            raise RuntimeError(f"Corrupt marginal is not balanced for {corrupt}.")
    return counts, uniform_cell_target, clean_target


def main() -> None:
    args = parse_args()
    if args.target_pairs < 0:
        raise ValueError("--target-pairs must be non-negative")
    if args.min_clean_top1_margin < 0 or args.min_corrupt_top1_margin < 0:
        raise ValueError("Top-1 margin thresholds must be non-negative")
    excluded_sample_ids = {str(sample_id) for sample_id in args.exclude_sample_id}
    canonical_rows = read_jsonl(args.canonical_path)
    manifest_rows = read_jsonl(args.converted_manifest_path)
    behavior_rows = read_jsonl(args.behavior_path)

    canonical_by_id = {str(row["sample_id"]): row for row in canonical_rows}
    manifest_by_id = {str(row["sample_id"]): row for row in manifest_rows}
    behavior_by_id = {str(row["sample_id"]): row for row in behavior_rows}

    candidates: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    behavior_valid_count = 0
    for sample_id, canonical_row in canonical_by_id.items():
        if sample_id in excluded_sample_ids:
            continue
        manifest_row = manifest_by_id[sample_id]
        behavior_row = behavior_by_id[sample_id]
        if not behavior_row["clean_is_tool_call_top1"]:
            continue
        if behavior_row["corrupt_is_tool_call_top1"]:
            continue
        behavior_valid_count += 1
        clean_top1_margin = float(behavior_row["clean_top1_margin"])
        corrupt_top1_margin = float(behavior_row["corrupt_top1_margin"])
        if clean_top1_margin < args.min_clean_top1_margin:
            continue
        if corrupt_top1_margin < args.min_corrupt_top1_margin:
            continue
        clean_user = str(canonical_row["user_content_clean"])
        corrupt_user = str(canonical_row["user_content_corrupt"])
        clean_prompt_path = Path(str(manifest_row["clean_prompt_path"]))
        corrupt_prompt_path = Path(str(manifest_row["corrupt_prompt_path"]))
        clean_prompt = clean_prompt_path.read_text(encoding="utf-8")
        corrupt_prompt = corrupt_prompt_path.read_text(encoding="utf-8")
        verify_pair_equivalence(clean_user, corrupt_user, clean_prompt, corrupt_prompt)

        clean_word, _ = first_word_and_tail(clean_user)
        corrupt_word, _ = first_word_and_tail(corrupt_user)
        if clean_word.lower() != str(canonical_row["clean_candidate"]):
            raise ValueError(f"Clean first word mismatch for {sample_id}")
        if corrupt_word.lower() != str(canonical_row["corrupt_candidate"]):
            raise ValueError(f"Corrupt first word mismatch for {sample_id}")

        key = (str(canonical_row["clean_candidate"]), str(canonical_row["corrupt_candidate"]))
        candidates[key].append(
            {
                "sample_id": sample_id,
                "split": str(canonical_row["split"]),
                "language": str(canonical_row["language"]),
                "clean_candidate": key[0],
                "corrupt_candidate": key[1],
                "clean_prompt_path": str(clean_prompt_path),
                "corrupt_prompt_path": str(corrupt_prompt_path),
                "clean_tool_token_prob": float(behavior_row["clean_tool_token_prob"]),
                "corrupt_tool_token_prob": float(behavior_row["corrupt_tool_token_prob"]),
                "clean_top1_margin": clean_top1_margin,
                "corrupt_top1_margin": corrupt_top1_margin,
                "clean_top1_token_text": str(behavior_row["clean_top1_token_text"]),
                "corrupt_top1_token_text": str(behavior_row["corrupt_top1_token_text"]),
            }
        )

    combinations = [(clean, corrupt) for clean in CLEAN_CANDIDATES for corrupt in CORRUPT_CANDIDATES]
    missing = sorted(set(combinations) - set(candidates))
    if missing and not args.target_pairs:
        raise ValueError(f"Missing required verb combinations: {missing}")

    rng = random.Random(args.seed)
    selected: list[dict[str, Any]] = []
    if args.target_pairs:
        if args.allow_relaxed_clean_balance:
            selected_counts, uniform_cell_target, clean_target = solve_relaxed_clean_marginal_counts(
                candidates,
                target_pairs=args.target_pairs,
            )
            selection_mode = "exact_corrupt_marginals_relaxed_clean_marginals"
        else:
            selected_counts, uniform_cell_target = solve_balanced_marginal_counts(
                candidates,
                target_pairs=args.target_pairs,
            )
            clean_target = args.target_pairs // len(CLEAN_CANDIDATES)
            selection_mode = "balanced_marginals_with_near_uniform_cells"
    else:
        per_combo = min(len(candidates[key]) for key in combinations)
        if per_combo <= 0:
            raise RuntimeError("No complete behavior-valid Qwen3.5 verb grid is available")
        selected_counts = {key: per_combo for key in combinations}
        uniform_cell_target = per_combo
        clean_target = per_combo * len(CORRUPT_CANDIDATES)
        selection_mode = "largest_complete_grid"

    for key in combinations:
        pool = list(candidates[key])
        count = selected_counts[key]
        if len(pool) < count:
            raise ValueError(f"Insufficient pool for {key}: need {count}, have {len(pool)}")
        if args.rank_by_min_top1_margin:
            chosen = sorted(
                pool,
                key=lambda row: (
                    -min(float(row["clean_top1_margin"]), float(row["corrupt_top1_margin"])),
                    -float(row["clean_top1_margin"]),
                    -float(row["corrupt_top1_margin"]),
                    row["sample_id"],
                ),
            )[:count]
        else:
            rng.shuffle(pool)
            chosen = pool[:count]
        chosen = sorted(chosen, key=lambda row: (row["split"], row["language"], row["sample_id"]))
        selected.extend(chosen)

    sample_ids = [str(row["sample_id"]) for row in selected]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("A source sample was selected for more than one Qwen3.5 verb pair")

    selected.sort(key=lambda row: (row["clean_candidate"], row["corrupt_candidate"], row["sample_id"]))
    prepare_output_root(args.output_root, overwrite=args.overwrite)

    manifest_records: list[dict[str, Any]] = []
    for idx, row in enumerate(selected, start=1):
        clean_out = args.output_root / f"clean_{idx}.txt"
        corrupt_out = args.output_root / f"corrupt_{idx}.txt"
        clean_text = Path(row["clean_prompt_path"]).read_text(encoding="utf-8")
        corrupt_text = Path(row["corrupt_prompt_path"]).read_text(encoding="utf-8")
        clean_out.write_text(clean_text, encoding="utf-8")
        corrupt_out.write_text(corrupt_text, encoding="utf-8")
        manifest_records.append(
            {
                "pair_id": idx,
                "sample_id": row["sample_id"],
                "split": row["split"],
                "language": row["language"],
                "clean_candidate": row["clean_candidate"],
                "corrupt_candidate": row["corrupt_candidate"],
                "clean_filename": clean_out.name,
                "corrupt_filename": corrupt_out.name,
                "clean_source_path": row["clean_prompt_path"],
                "corrupt_source_path": row["corrupt_prompt_path"],
                "clean_tool_token_prob": row["clean_tool_token_prob"],
                "corrupt_tool_token_prob": row["corrupt_tool_token_prob"],
                "clean_top1_margin": row["clean_top1_margin"],
                "corrupt_top1_margin": row["corrupt_top1_margin"],
                "clean_top1_token_text": row["clean_top1_token_text"],
                "corrupt_top1_token_text": row["corrupt_top1_token_text"],
            }
        )

    with (args.output_root / "manifest.jsonl").open("w", encoding="utf-8") as handle:
        for row in manifest_records:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    clean_counter = Counter(row["clean_candidate"] for row in selected)
    corrupt_counter = Counter(row["corrupt_candidate"] for row in selected)
    combo_counter = Counter((row["clean_candidate"], row["corrupt_candidate"]) for row in selected)
    summary = {
        "n_pairs": len(selected),
        "output_root": str(args.output_root),
        "clean_candidate_counts": dict(sorted(clean_counter.items())),
        "corrupt_candidate_counts": dict(sorted(corrupt_counter.items())),
        "combination_counts": {
            f"{clean}|{corrupt}": combo_counter[(clean, corrupt)]
            for clean, corrupt in combinations
        },
        "selection_policy": (
            f"Selected from {args.model_label} behavior-pass pairs only. Every kept pair satisfies "
            "clean=<tool_call> top1 and corrupt!=<tool_call> top1, and the rendered prompts are "
            "verified to differ only in the first user-content word (the verb). "
            + (
                "For this target, corrupt verb marginals are exactly balanced while clean marginals are "
                "minimized for deviation from the uniform quota because strict clean balance is infeasible "
                "for this model's behavior-valid pool."
                if args.allow_relaxed_clean_balance
                else "For this target, the selector balances each clean and corrupt verb marginal exactly, "
                "then keeps cell counts as close as possible to a uniform grid subject to behavior-valid availability."
            )
        ),
        "target_pairs_requested": args.target_pairs,
        "selection_mode": selection_mode,
        "uniform_cell_target": uniform_cell_target,
        "uniform_clean_target": clean_target,
        "allow_relaxed_clean_balance": bool(args.allow_relaxed_clean_balance),
        "behavior_valid_pairs_before_margin_filter": behavior_valid_count,
        "margin_filter": {
            "min_clean_top1_margin": float(args.min_clean_top1_margin),
            "min_corrupt_top1_margin": float(args.min_corrupt_top1_margin),
            "rank_by_min_top1_margin": bool(args.rank_by_min_top1_margin),
            "eligible_pairs_after_margin_filter": int(sum(len(pool) for pool in candidates.values())),
        },
        "excluded_sample_ids": sorted(excluded_sample_ids),
        "seed": args.seed,
    }
    (args.output_root / "selection_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
