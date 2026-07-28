#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import torch
from tqdm.auto import tqdm

from phase7_l24_directionality_common import (
    PATCH_LAYER,
    PairBaselineCache,
    load_or_collect_pair_baseline,
    make_position_resid_add_hook,
    project_delta,
)
from phase8_common import (
    DEFAULT_HELDOUT_CACHE,
    DEFAULT_PC_BUNDLE,
    EVAL_DATASET_ROOT,
    EXP_A_ROOT as PHASE8_EXP_A_ROOT,
    MODEL_PATH,
    configure_matplotlib,
    ensure_dir,
    get_tool_token_id,
    load_gate_bundle,
    load_model_and_tokenizer,
    percent,
    set_seed,
    write_csv,
    write_text,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = PROJECT_ROOT / "results" / "8b_main" / "phase9_gate_interpretability_summaries" / "exp_a_causal_vocab_projection_rerun"
DEFAULT_ALPHA_TABLE = PHASE8_EXP_A_ROOT / "pck_recovery_table.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 9 Exp A: causal vocabulary projection from true rank-1 gate interventions.")
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH)
    parser.add_argument("--dataset-root", type=Path, default=EVAL_DATASET_ROOT)
    parser.add_argument("--pc-bundle", type=Path, default=DEFAULT_PC_BUNDLE)
    parser.add_argument("--heldout-cache", type=Path, default=DEFAULT_HELDOUT_CACHE)
    parser.add_argument("--alpha-table", type=Path, default=DEFAULT_ALPHA_TABLE)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-pairs", type=int, default=300)
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    import csv

    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def infer_best_rank1_alpha(path: Path) -> float:
    rows = read_csv_rows(path)
    matches = [row for row in rows if row.get("intervention") == "per-pair rank-1"]
    if not matches:
        raise ValueError(f"Could not find per-pair rank-1 row in {path}")
    return float(matches[0]["alpha"])


def decode_token(tokenizer, token_id: int) -> str:
    return tokenizer.decode([int(token_id)])


def token_filter_reason(token_text: str, token_id: int, special_ids: set[int], tool_token_id: int) -> str:
    if int(token_id) == int(tool_token_id):
        return ""
    if int(token_id) in special_ids:
        return "special"
    stripped = token_text.strip()
    if not stripped:
        return "blank"
    if "\ufffd" in token_text or "�" in token_text:
        return "bad_decode"
    if token_text.startswith("<|") and token_text.endswith("|>"):
        return "special_like"
    if not any(ch.isalnum() for ch in stripped):
        return "format_only"
    return ""


def top_display_rows(rows: Sequence[dict[str, object]], *, descending: bool, limit: int) -> list[dict[str, object]]:
    ordered = sorted(rows, key=lambda row: float(row["mean_delta_logit"]), reverse=descending)
    kept = [row for row in ordered if not row["filter_reason"]]
    return kept[:limit]


def top_shift_rows(rows: Sequence[dict[str, object]], *, descending: bool, limit: int) -> list[dict[str, object]]:
    ordered = sorted(rows, key=lambda row: float(row["top1_rate_shift"]), reverse=descending)
    kept = [
        row
        for row in ordered
        if not row["filter_reason"] and abs(float(row["top1_rate_shift"])) > 1e-9
    ]
    return kept[:limit]


def format_token_triplets(rows: Sequence[dict[str, object]], *, limit: int = 8) -> str:
    parts: list[str] = []
    for row in list(rows)[:limit]:
        token = str(row["token_text_repr"])
        parts.append(
            f"{token} (Δ={float(row['mean_delta_logit']):+.3f}, ρ={float(row['support_rate']):.3f}, "
            f"top1Δ={float(row['top1_rate_shift']):+.3f}, "
            f"base={float(row['baseline_top1_rate']):.3f}, patch={float(row['patched_top1_rate']):.3f})"
        )
    return "; ".join(parts) if parts else "n/a"


def bfloat_last_logits(logits: torch.Tensor) -> torch.Tensor:
    return logits[:, -1, :].detach().cpu().float()


def run_intervention_vocab_sweep(
    model,
    baseline_cache: PairBaselineCache,
    *,
    tool_token_id: int,
    eval_side: str,
    patch_layer: int,
    delta_resolver,
    topk: int,
) -> dict[str, object]:
    if eval_side not in {"clean", "corrupt"}:
        raise ValueError(f"Unsupported eval_side={eval_side}")

    vocab_size = int(model.cfg.d_vocab)
    n_samples = len(baseline_cache.samples)
    hook_name = f"blocks.{patch_layer}.hook_resid_pre"

    sum_delta = torch.zeros(vocab_size, dtype=torch.float64)
    promote_counts = torch.zeros(vocab_size, dtype=torch.int32)
    suppress_counts = torch.zeros(vocab_size, dtype=torch.int32)
    baseline_top1 = torch.empty(n_samples, dtype=torch.long)
    patched_top1 = torch.empty(n_samples, dtype=torch.long)

    progress = tqdm(
        baseline_cache.pair_batches,
        desc=f"Vocab projection: {eval_side}",
        dynamic_ncols=True,
    )
    for batch in progress:
        tokens_cpu = batch.clean_tokens_cpu if eval_side == "clean" else batch.corrupt_tokens_cpu
        tokens = tokens_cpu.to(model.W_U.device)
        delta_batch = delta_resolver(batch.indices).float()
        hooks = [(hook_name, make_position_resid_add_hook(delta_batch, "last"))]
        with torch.no_grad():
            baseline_logits = model(tokens)
            patched_logits = model.run_with_hooks(tokens, fwd_hooks=hooks)

        baseline_last = bfloat_last_logits(baseline_logits)
        patched_last = bfloat_last_logits(patched_logits)
        delta_last = patched_last - baseline_last

        sum_delta += delta_last.sum(dim=0, dtype=torch.float64)
        promote_ids = torch.topk(delta_last, k=topk, dim=-1).indices.reshape(-1)
        suppress_ids = torch.topk(delta_last, k=topk, dim=-1, largest=False).indices.reshape(-1)
        promote_counts += torch.bincount(promote_ids, minlength=vocab_size).to(torch.int32)
        suppress_counts += torch.bincount(suppress_ids, minlength=vocab_size).to(torch.int32)
        baseline_top1[batch.indices] = baseline_last.argmax(dim=-1)
        patched_top1[batch.indices] = patched_last.argmax(dim=-1)

    return {
        "mean_delta": (sum_delta / float(n_samples)).float(),
        "support_rate_promote": promote_counts.float() / float(n_samples),
        "support_rate_suppress": suppress_counts.float() / float(n_samples),
        "baseline_top1": baseline_top1,
        "patched_top1": patched_top1,
    }


def build_token_rows(
    tokenizer,
    *,
    tool_token_id: int,
    special_ids: set[int],
    stats: dict[str, object],
) -> list[dict[str, object]]:
    mean_delta = stats["mean_delta"]
    support_promote = stats["support_rate_promote"]
    support_suppress = stats["support_rate_suppress"]
    baseline_top1 = stats["baseline_top1"]
    patched_top1 = stats["patched_top1"]
    vocab_size = int(mean_delta.shape[0])

    baseline_counts = torch.bincount(baseline_top1, minlength=vocab_size).float()
    patched_counts = torch.bincount(patched_top1, minlength=vocab_size).float()
    order_desc = torch.argsort(mean_delta, descending=True)
    order_asc = torch.argsort(mean_delta, descending=False)
    rank_promoted = torch.empty(vocab_size, dtype=torch.long)
    rank_suppressed = torch.empty(vocab_size, dtype=torch.long)
    rank_promoted[order_desc] = torch.arange(vocab_size)
    rank_suppressed[order_asc] = torch.arange(vocab_size)

    rows: list[dict[str, object]] = []
    for token_id in range(vocab_size):
        token_text = decode_token(tokenizer, token_id)
        filter_reason = token_filter_reason(token_text, token_id, special_ids, tool_token_id)
        rows.append(
            {
                "token_id": token_id,
                "token_text": token_text,
                "token_text_repr": json.dumps(token_text, ensure_ascii=False),
                "mean_delta_logit": float(mean_delta[token_id].item()),
                "support_rate": float(support_promote[token_id].item()),
                "support_rate_suppressed": float(support_suppress[token_id].item()),
                "baseline_top1_rate": float((baseline_counts[token_id] / baseline_top1.numel()).item()),
                "patched_top1_rate": float((patched_counts[token_id] / patched_top1.numel()).item()),
                "top1_rate_shift": float(((patched_counts[token_id] - baseline_counts[token_id]) / baseline_top1.numel()).item()),
                "promoted_rank": int(rank_promoted[token_id].item()) + 1,
                "suppressed_rank": int(rank_suppressed[token_id].item()) + 1,
                "is_special": int(token_id in special_ids),
                "filter_reason": filter_reason,
            }
        )
    return rows


def build_common_no_tool_group(rows: Sequence[dict[str, object]], *, tool_token_id: int) -> list[int]:
    candidates = [
        row
        for row in rows
        if int(row["token_id"]) != int(tool_token_id)
        and not row["filter_reason"]
        and float(row["baseline_top1_rate"]) > 0.0
    ]
    candidates = sorted(candidates, key=lambda row: float(row["baseline_top1_rate"]), reverse=True)
    selected: list[int] = []
    mass = 0.0
    for row in candidates:
        token_id = int(row["token_id"])
        rate = float(row["baseline_top1_rate"])
        if rate < 0.005 and selected:
            break
        selected.append(token_id)
        mass += rate
        if len(selected) >= 10 or mass >= 0.9:
            break
    return selected


def build_group_rows(
    tokenizer,
    addition_rows: Sequence[dict[str, object]],
    removal_rows: Sequence[dict[str, object]],
    *,
    tool_token_id: int,
) -> list[dict[str, object]]:
    add_by_id = {int(row["token_id"]): row for row in addition_rows}
    rem_by_id = {int(row["token_id"]): row for row in removal_rows}
    common_no_tool = build_common_no_tool_group(addition_rows, tool_token_id=tool_token_id)
    token_lookup = {
        str(row["token_text"]): int(row["token_id"])
        for row in addition_rows
    }

    def ids_for_texts(texts: Sequence[str]) -> list[int]:
        seen: list[int] = []
        for text in texts:
            token_id = token_lookup.get(text)
            if token_id is not None and token_id not in seen:
                seen.append(token_id)
        return seen

    groups: list[tuple[str, str, list[int]]] = [
        ("tool_call_entry", "single_token", [int(tool_token_id)]),
        ("common_no_tool_starters", "data_driven_family", common_no_tool),
        (
            "ordinary_prose_starters",
            "observed_readable_family",
            ids_for_texts(["The", "To", "There", "Based", "Looking", "So", "Hmm"]),
        ),
        (
            "no_need_starters",
            "observed_readable_family",
            ids_for_texts(["No", "None", "Nothing"]),
        ),
    ]
    groups.extend(
        (
            f"starter_{decode_token(tokenizer, token_id).strip().replace(' ', '_')[:24] or token_id}",
            "singleton_from_baseline",
            [int(token_id)],
        )
        for token_id in common_no_tool[:5]
    )

    rows: list[dict[str, object]] = []
    for group_name, group_source, token_ids in groups:
        if not token_ids:
            continue
        add_group = [add_by_id[token_id] for token_id in token_ids]
        rem_group = [rem_by_id[token_id] for token_id in token_ids]
        rows.append(
            {
                "group_name": group_name,
                "group_source": group_source,
                "n_tokens": len(token_ids),
                "token_ids": json.dumps(token_ids, ensure_ascii=False),
                "token_texts": json.dumps([decode_token(tokenizer, token_id) for token_id in token_ids], ensure_ascii=False),
                "addition_mean_group_delta": float(sum(float(row["mean_delta_logit"]) for row in add_group)),
                "addition_baseline_top1_rate": float(sum(float(row["baseline_top1_rate"]) for row in add_group)),
                "addition_patched_top1_rate": float(sum(float(row["patched_top1_rate"]) for row in add_group)),
                "addition_top1_rate_shift": float(sum(float(row["top1_rate_shift"]) for row in add_group)),
                "removal_mean_group_delta": float(sum(float(row["mean_delta_logit"]) for row in rem_group)),
                "removal_baseline_top1_rate": float(sum(float(row["baseline_top1_rate"]) for row in rem_group)),
                "removal_patched_top1_rate": float(sum(float(row["patched_top1_rate"]) for row in rem_group)),
                "removal_top1_rate_shift": float(sum(float(row["top1_rate_shift"]) for row in rem_group)),
            }
        )
    return rows


def plot_group_effects(group_rows: Sequence[dict[str, object]], path: Path) -> None:
    configure_matplotlib()
    labels = [str(row["group_name"]) for row in group_rows]
    addition = [float(row["addition_mean_group_delta"]) for row in group_rows]
    removal = [float(row["removal_mean_group_delta"]) for row in group_rows]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].bar(range(len(labels)), addition, color="#2b6cb0")
    axes[0].set_title("Addition on Corrupt")
    axes[0].set_ylabel("Mean grouped logit shift")
    axes[0].set_xticks(range(len(labels)))
    axes[0].set_xticklabels(labels, rotation=35, ha="right")

    axes[1].bar(range(len(labels)), removal, color="#c05621")
    axes[1].set_title("Removal on Clean")
    axes[1].set_ylabel("Mean grouped logit shift")
    axes[1].set_xticks(range(len(labels)))
    axes[1].set_xticklabels(labels, rotation=35, ha="right")

    for ax in axes:
        ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.4)
        ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    ensure_dir(path.parent)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    ensure_dir(args.output_root)
    set_seed(args.seed)

    alpha = float(args.alpha) if args.alpha is not None else infer_best_rank1_alpha(args.alpha_table)
    model, tokenizer = load_model_and_tokenizer(model_path=args.model_path, device=args.device)
    tool_token_id = get_tool_token_id(tokenizer)
    special_ids = set(getattr(tokenizer, "all_special_ids", []))
    pc_bundle = load_gate_bundle(args.pc_bundle)

    baseline_cache = load_or_collect_pair_baseline(
        model,
        tokenizer,
        dataset_root=args.dataset_root,
        max_pairs=args.max_pairs,
        batch_size=args.batch_size,
        patch_layer=PATCH_LAYER,
        cache_path=args.heldout_cache,
    )
    diff_vectors = baseline_cache.clean_resid - baseline_cache.corrupt_resid

    addition_stats = run_intervention_vocab_sweep(
        model,
        baseline_cache,
        tool_token_id=tool_token_id,
        eval_side="corrupt",
        patch_layer=PATCH_LAYER,
        delta_resolver=lambda indices: project_delta(diff_vectors[list(indices)], pc_bundle, k=1) * alpha,
        topk=args.topk,
    )
    removal_stats = run_intervention_vocab_sweep(
        model,
        baseline_cache,
        tool_token_id=tool_token_id,
        eval_side="clean",
        patch_layer=PATCH_LAYER,
        delta_resolver=lambda indices: project_delta(diff_vectors[list(indices)], pc_bundle, k=1) * (-alpha),
        topk=args.topk,
    )

    addition_rows = build_token_rows(
        tokenizer,
        tool_token_id=tool_token_id,
        special_ids=special_ids,
        stats=addition_stats,
    )
    removal_rows = build_token_rows(
        tokenizer,
        tool_token_id=tool_token_id,
        special_ids=special_ids,
        stats=removal_stats,
    )
    group_rows = build_group_rows(
        tokenizer,
        addition_rows,
        removal_rows,
        tool_token_id=tool_token_id,
    )

    write_csv(args.output_root / "token_logit_shift_addition.csv", addition_rows)
    write_csv(args.output_root / "token_logit_shift_removal.csv", removal_rows)
    write_csv(args.output_root / "token_group_summary.csv", group_rows)
    plot_group_effects(group_rows, args.output_root / "plot_token_group_effects.pdf")

    add_top = top_display_rows(addition_rows, descending=True, limit=12)
    add_bottom = top_display_rows(addition_rows, descending=False, limit=12)
    rem_top = top_display_rows(removal_rows, descending=True, limit=12)
    rem_bottom = top_display_rows(removal_rows, descending=False, limit=12)
    add_top1_shift = top_shift_rows(addition_rows, descending=True, limit=8)
    add_top1_drop = top_shift_rows(addition_rows, descending=False, limit=8)
    rem_top1_shift = top_shift_rows(removal_rows, descending=True, limit=8)
    rem_top1_drop = top_shift_rows(removal_rows, descending=False, limit=8)

    group_lookup = {str(row["group_name"]): row for row in group_rows}
    tool_group = group_lookup.get("tool_call_entry")
    no_tool_group = group_lookup.get("common_no_tool_starters")
    prose_group = group_lookup.get("ordinary_prose_starters")
    no_need_group = group_lookup.get("no_need_starters")
    no_tool_tokens = []
    if no_tool_group is not None:
        no_tool_tokens = json.loads(str(no_tool_group["token_texts"]))

    lines = [
        "# Phase 9 Exp A: Causal Vocabulary Projection",
        "",
        "## Setup",
        f"- Eval split: `{args.dataset_root}` with `{len(baseline_cache.samples)}` held-out pairs.",
        f"- Intervention site: `blocks.{PATCH_LAYER}.hook_resid_pre` prediction position.",
        f"- Gate object: per-pair rank-1 reconstruction from `{args.pc_bundle}`.",
        f"- Alpha: `{alpha:g}` from `{args.alpha_table}`.",
        f"- TopK support metric: `K={args.topk}`.",
        "",
        "## Addition on Corrupt",
        f"- Top promoted raw tokens: {format_token_triplets(add_top)}",
        f"- Top suppressed raw tokens: {format_token_triplets(add_bottom)}",
        f"- Largest top1-rate increases: {format_token_triplets(add_top1_shift)}",
        f"- Largest top1-rate decreases: {format_token_triplets(add_top1_drop)}",
        "",
        "## Removal on Clean",
        f"- Top promoted raw tokens: {format_token_triplets(rem_top)}",
        f"- Top suppressed raw tokens: {format_token_triplets(rem_bottom)}",
        f"- Largest top1-rate increases: {format_token_triplets(rem_top1_shift)}",
        f"- Largest top1-rate decreases: {format_token_triplets(rem_top1_drop)}",
        "",
        "## Grouped Readout",
    ]
    if tool_group is not None:
        lines.append(
            f"- `tool_call_entry`: addition Δ={float(tool_group['addition_mean_group_delta']):+.3f}, "
            f"top1 {percent(float(tool_group['addition_baseline_top1_rate']))} -> {percent(float(tool_group['addition_patched_top1_rate']))}; "
            f"removal Δ={float(tool_group['removal_mean_group_delta']):+.3f}, "
            f"top1 {percent(float(tool_group['removal_baseline_top1_rate']))} -> {percent(float(tool_group['removal_patched_top1_rate']))}."
        )
    if no_tool_group is not None:
        lines.append(
            f"- `common_no_tool_starters`={json.dumps(no_tool_tokens, ensure_ascii=False)}: "
            f"addition Δ={float(no_tool_group['addition_mean_group_delta']):+.3f}, "
            f"top1 {percent(float(no_tool_group['addition_baseline_top1_rate']))} -> {percent(float(no_tool_group['addition_patched_top1_rate']))}; "
            f"removal Δ={float(no_tool_group['removal_mean_group_delta']):+.3f}, "
            f"top1 {percent(float(no_tool_group['removal_baseline_top1_rate']))} -> {percent(float(no_tool_group['removal_patched_top1_rate']))}."
        )
    if prose_group is not None:
        lines.append(
            f"- `ordinary_prose_starters`: addition Δ={float(prose_group['addition_mean_group_delta']):+.3f}, "
            f"top1 shift {float(prose_group['addition_top1_rate_shift']):+.3f}; "
            f"removal Δ={float(prose_group['removal_mean_group_delta']):+.3f}, "
            f"top1 shift {float(prose_group['removal_top1_rate_shift']):+.3f}."
        )
    if no_need_group is not None:
        lines.append(
            f"- `no_need_starters`: addition Δ={float(no_need_group['addition_mean_group_delta']):+.3f}, "
            f"top1 shift {float(no_need_group['addition_top1_rate_shift']):+.3f}; "
            f"removal Δ={float(no_need_group['removal_mean_group_delta']):+.3f}, "
            f"top1 shift {float(no_need_group['removal_top1_rate_shift']):+.3f}."
        )

    write_text(args.output_root / "summary.md", "\n".join(lines))


if __name__ == "__main__":
    main()
