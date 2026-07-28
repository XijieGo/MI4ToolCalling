#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from tqdm.auto import tqdm


COMMON_DIR = Path(__file__).resolve().parents[1] / "common"
if str(COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(COMMON_DIR))
SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from multiscale_common import (  # noqa: E402
    DEFAULT_DATASET_ROOT,
    build_pair_batches,
    clear_cuda,
    load_manifest_rows,
    load_model_and_tokenizer,
    load_sample_pairs,
    precompute_head_tool_projections,
    run_with_hooks_and_cache,
    set_seed,
    tool_stats,
    write_csv,
    write_json,
    write_text,
)
from artifact_paths import ARTIFACT_ROOT, QWEN3_8B_PATH  # noqa: E402


PROJECT_ROOT = ARTIFACT_ROOT
MODEL_PATH = QWEN3_8B_PATH
OUTPUT_ROOT = PROJECT_ROOT / "results" / "8b_main" / "claim_upgrade_suite"
PC_BUNDLE = PROJECT_ROOT / "results" / "8b_main" / "phase7_l24_directionality" / "exp_a_fixed_direction" / "pc_bundle.pt"
EARLY_HEAD_CSV = PROJECT_ROOT / "results" / "8b_main" / "attention_analysis" / "ablation_per_head.csv"

DEFAULT_EXPERIMENTS = ("exp1", "exp2", "exp3", "exp4")
EXP2_LAYERS = (24, 26, 28, 29, 30, 32, 34, 35)
EXP4_LAYERS = (29, 30, 31, 32, 33, 34, 35)
LATE_HEAD_TARGETS = ((33, 29), (33, 11))
LATE_MLP_LAYER = 34


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the 8B claim-upgrade experiment suite.")
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--pc-bundle", type=Path, default=PC_BUNDLE)
    parser.add_argument(
        "--early-head-csv",
        type=Path,
        default=EARLY_HEAD_CSV,
        help="Fresh attention-head ablation table used to choose the early bridge-head control set.",
    )
    parser.add_argument("--experiments", nargs="+", default=list(DEFAULT_EXPERIMENTS))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--rank-batch-size", type=int, default=16)
    parser.add_argument("--heavy-batch-size", type=int, default=6)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def normalize_experiments(items: Sequence[str]) -> list[str]:
    normalized: list[str] = []
    for item in items:
        key = str(item).strip().lower()
        if key == "all":
            return list(DEFAULT_EXPERIMENTS)
        if key not in DEFAULT_EXPERIMENTS:
            raise ValueError(f"Unknown experiment label: {item}")
        if key not in normalized:
            normalized.append(key)
    return normalized


def hook_name(layer: int, hook_kind: str) -> str:
    return f"blocks.{layer}.hook_resid_{hook_kind}"


def load_bundle(path: Path) -> tuple[int, str, torch.Tensor]:
    bundle = torch.load(path, map_location="cpu", weights_only=False)
    layer = int(bundle["patch_layer"] if "patch_layer" in bundle else bundle["layer"])
    hook_kind = "pre" if "patch_layer" in bundle else "post"
    mean_diff = bundle["mean_diff"]
    if not isinstance(mean_diff, torch.Tensor):
        mean_diff = torch.tensor(mean_diff)
    return layer, hook_kind, mean_diff.detach().cpu().float().view(-1)


def make_last_token_add_hook(delta_cpu: torch.Tensor):
    def hook_fn(value: torch.Tensor, hook):  # noqa: ANN001
        out = value.clone()
        delta = delta_cpu.to(device=value.device, dtype=value.dtype)
        if delta.ndim == 1:
            delta = delta.view(1, -1)
        out[:, -1, :] = out[:, -1, :] + delta
        return out

    return hook_fn


def make_zero_heads_hook(heads: Sequence[int]):
    head_ids = [int(head) for head in heads]

    def hook_fn(value: torch.Tensor, hook):  # noqa: ANN001
        out = value.clone()
        out[:, -1, head_ids, :] = 0
        return out

    return hook_fn


def make_zero_last_token_hook():
    def hook_fn(value: torch.Tensor, hook):  # noqa: ANN001
        out = value.clone()
        out[:, -1, :] = 0
        return out

    return hook_fn


def manifest_lookup(dataset_root: Path, split: str) -> dict[str, dict]:
    rows = load_manifest_rows(dataset_root, split, "clean")
    lookup: dict[str, dict] = {}
    for row in rows:
        filename = str(row.get("output_filename") or row.get("source_filename") or "")
        if filename:
            lookup[Path(filename).stem] = row
    return lookup


def language_of(sample_id: str, lookup: dict[str, dict]) -> str:
    row = lookup.get(sample_id, {})
    return str(row.get("language") or "unknown")


def aggregate_rows(rows: Sequence[dict[str, object]], group_keys: Sequence[str], metric_keys: Sequence[str]) -> list[dict[str, object]]:
    groups: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        key = tuple(row[key_name] for key_name in group_keys)
        groups[key].append(row)

    out: list[dict[str, object]] = []
    for key, group in sorted(groups.items()):
        record = {name: value for name, value in zip(group_keys, key)}
        record["n"] = len(group)
        for metric_key in metric_keys:
            values = np.asarray([float(item[metric_key]) for item in group], dtype=np.float64)
            record[f"{metric_key}_mean"] = float(values.mean())
            record[f"{metric_key}_median"] = float(np.median(values))
        out.append(record)
    return out


def collect_side_residuals(
    model,
    pairs,
    *,
    side: str,
    layer: int,
    hook_kind: str,
    batch_size: int,
) -> torch.Tensor:
    if side not in {"clean", "corrupt"}:
        raise ValueError(f"Unsupported side: {side}")
    pair_batches = build_pair_batches(pairs, batch_size=batch_size)
    resid = torch.empty((len(pairs), int(model.cfg.d_model)), dtype=torch.float32)
    target_hook = hook_name(layer, hook_kind)

    progress = tqdm(pair_batches, desc=f"Collect {side} L{layer} {hook_kind}", dynamic_ncols=True)
    for batch in progress:
        tokens_cpu = batch.clean_tokens_cpu if side == "clean" else batch.corrupt_tokens_cpu
        with torch.no_grad():
            _, cache = model.run_with_cache(
                tokens_cpu.to(model.W_U.device),
                names_filter=lambda name: name == target_hook,
            )
        resid[batch.indices] = cache[target_hook][:, -1, :].detach().cpu().float()
        clear_cuda()
        progress.set_postfix(tok=batch.token_len)
    return resid


def evaluate_delta_on_corrupt(
    model,
    pairs,
    *,
    layer: int,
    hook_kind: str,
    delta: torch.Tensor,
    batch_size: int,
    tool_token_id: int,
    split: str,
    target_label: str,
    donor_label: str,
    metadata_lookup: dict[str, dict],
) -> tuple[dict[str, object], list[dict[str, object]]]:
    pair_batches = build_pair_batches(pairs, batch_size=batch_size)
    target_hook = hook_name(layer, hook_kind)
    per_sample_rows: list[dict[str, object]] = []
    count = 0
    top1_sum = 0
    strict_flip_sum = 0
    logit_sum = 0.0
    baseline_top1_sum = 0

    progress = tqdm(pair_batches, desc=f"{donor_label} -> {target_label} ({split})", dynamic_ncols=True)
    for batch in progress:
        corrupt_tokens = batch.corrupt_tokens_cpu.to(model.W_U.device)
        with torch.no_grad():
            baseline_logits = model(corrupt_tokens)
        baseline_logit, _baseline_prob, baseline_top1 = tool_stats(baseline_logits, tool_token_id)
        hooks = [(target_hook, make_last_token_add_hook(delta))]
        with torch.no_grad():
            patched_logits = model.run_with_hooks(corrupt_tokens, fwd_hooks=hooks)
        patched_logit, _patched_prob, patched_top1 = tool_stats(patched_logits, tool_token_id)
        strict_flip = (baseline_top1 != tool_token_id) & (patched_top1 == tool_token_id)

        count += int(patched_top1.shape[0])
        top1_sum += int((patched_top1 == tool_token_id).sum().item())
        strict_flip_sum += int(strict_flip.sum().item())
        logit_sum += float(patched_logit.sum().item())
        baseline_top1_sum += int((baseline_top1 == tool_token_id).sum().item())

        for local_idx, pair_idx in enumerate(batch.indices):
            pair = pairs[pair_idx]
            per_sample_rows.append(
                {
                    "sample_id": pair.sample_id,
                    "language": language_of(pair.sample_id, metadata_lookup),
                    "split": split,
                    "target_label": target_label,
                    "donor_label": donor_label,
                    "tool_logit": float(patched_logit[local_idx].item()),
                    "is_tool_call_top1": int(patched_top1[local_idx].item() == tool_token_id),
                    "strict_flip": int(strict_flip[local_idx].item()),
                    "baseline_is_tool_call_top1": int(baseline_top1[local_idx].item() == tool_token_id),
                }
            )
        clear_cuda()
        progress.set_postfix(tok=batch.token_len)

    summary = {
        "split": split,
        "target_label": target_label,
        "donor_label": donor_label,
        "n_pairs": count,
        "tool_call_top1_rate": float(top1_sum / max(count, 1)),
        "strict_flip_rate": float(strict_flip_sum / max(count, 1)),
        "mean_tool_logit": float(logit_sum / max(count, 1)),
        "baseline_corrupt_top1_rate": float(baseline_top1_sum / max(count, 1)),
    }
    return summary, per_sample_rows


def run_exp1_cross_verb(model, args: argparse.Namespace, *, tool_token_id: int, gate_layer: int, hook_kind: str) -> None:
    exp_root = args.output_root / "exp1_cross_verb_generalization"
    train_pairs = load_sample_pairs(model, dataset_root=args.dataset_root, split="train", max_pairs=0)
    test_pairs = load_sample_pairs(model, dataset_root=args.dataset_root, split="test", max_pairs=0)
    train_lookup = manifest_lookup(args.dataset_root, "train")
    test_lookup = manifest_lookup(args.dataset_root, "test")

    rng = random.Random(args.seed)
    donor_write_all = [pair for pair in train_pairs if pair.clean_candidate == "write"]
    donor_write = [donor_write_all[idx] for idx in sorted(rng.sample(range(len(donor_write_all)), 200))]
    donor_bc = [pair for pair in train_pairs if pair.clean_candidate in {"build", "complete"}]
    full_train = [train_pairs[idx] for idx in sorted(rng.sample(range(len(train_pairs)), 200))]

    target_train_bc = [pair for pair in train_pairs if pair.clean_candidate in {"build", "complete"}]
    target_test_bc = [pair for pair in test_pairs if pair.clean_candidate in {"build", "complete"}]
    target_train_write = [pair for pair in train_pairs if pair.clean_candidate == "write"]
    target_test_write = [pair for pair in test_pairs if pair.clean_candidate == "write"]

    donor_write_clean = collect_side_residuals(model, donor_write, side="clean", layer=gate_layer, hook_kind=hook_kind, batch_size=args.batch_size)
    donor_write_corrupt = collect_side_residuals(model, donor_write, side="corrupt", layer=gate_layer, hook_kind=hook_kind, batch_size=args.batch_size)
    mu_write = (donor_write_clean - donor_write_corrupt).mean(dim=0)

    donor_bc_clean = collect_side_residuals(model, donor_bc, side="clean", layer=gate_layer, hook_kind=hook_kind, batch_size=args.batch_size)
    donor_bc_corrupt = collect_side_residuals(model, donor_bc, side="corrupt", layer=gate_layer, hook_kind=hook_kind, batch_size=args.batch_size)
    mu_bc = (donor_bc_clean - donor_bc_corrupt).mean(dim=0)

    full_clean = collect_side_residuals(model, full_train, side="clean", layer=gate_layer, hook_kind=hook_kind, batch_size=args.batch_size)
    full_corrupt = collect_side_residuals(model, full_train, side="corrupt", layer=gate_layer, hook_kind=hook_kind, batch_size=args.batch_size)
    mu_full = (full_clean - full_corrupt).mean(dim=0)

    summary_rows: list[dict[str, object]] = []
    per_sample_rows: list[dict[str, object]] = []
    configs = [
        ("train", "build_complete", "mu_write", target_train_bc, mu_write, train_lookup),
        ("test", "build_complete", "mu_write", target_test_bc, mu_write, test_lookup),
        ("test", "build_complete", "mu_full", target_test_bc, mu_full, test_lookup),
        ("train", "write", "mu_build_complete", target_train_write, mu_bc, train_lookup),
        ("test", "write", "mu_build_complete", target_test_write, mu_bc, test_lookup),
        ("test", "write", "mu_full", target_test_write, mu_full, test_lookup),
    ]
    for split, target_label, donor_label, target_pairs, delta, lookup in configs:
        summary, rows = evaluate_delta_on_corrupt(
            model,
            target_pairs,
            layer=gate_layer,
            hook_kind=hook_kind,
            delta=delta,
            batch_size=args.batch_size,
            tool_token_id=tool_token_id,
            split=split,
            target_label=target_label,
            donor_label=donor_label,
            metadata_lookup=lookup,
        )
        summary_rows.append(summary)
        per_sample_rows.extend(rows)

    write_csv(exp_root / "cross_verb_summary.csv", summary_rows)
    write_csv(exp_root / "cross_verb_per_sample.csv", per_sample_rows)
    write_json(
        exp_root / "metadata.json",
        {
            "gate_layer": gate_layer,
            "hook_kind": hook_kind,
            "donor_write_n": len(donor_write),
            "donor_build_complete_n": len(donor_bc),
            "full_train_n": len(full_train),
            "target_train_build_complete_n": len(target_train_bc),
            "target_test_build_complete_n": len(target_test_bc),
            "target_train_write_n": len(target_train_write),
            "target_test_write_n": len(target_test_write),
        },
    )

    row_map = {(row["split"], row["target_label"], row["donor_label"]): row for row in summary_rows}
    heldout_write = row_map[("test", "build_complete", "mu_write")]
    heldout_full = row_map[("test", "build_complete", "mu_full")]
    gap_pp = 100.0 * (float(heldout_full["strict_flip_rate"]) - float(heldout_write["strict_flip_rate"]))
    lines = [
        "# Exp 1: Cross-Verb Generalization",
        "",
        f"- gate layer: `L{gate_layer}` (`hook_resid_{hook_kind}`)",
        f"- donor `write` train pairs: `{len(donor_write)}`",
        f"- donor `build+complete` train pairs: `{len(donor_bc)}`",
        f"- held-out `build+complete` pairs: `{len(target_test_bc)}`",
        f"- held-out `write` pairs: `{len(target_test_write)}`",
        "",
        "## Main held-out result",
        "",
        f"- `mu_write -> build+complete (test)` top1: `{float(heldout_write['tool_call_top1_rate']):.2%}`",
        f"- `mu_write -> build+complete (test)` strict flip: `{float(heldout_write['strict_flip_rate']):.2%}`",
        f"- `mu_full -> build+complete (test)` strict flip: `{float(heldout_full['strict_flip_rate']):.2%}`",
        f"- gap vs `mu_full`: `{gap_pp:+.2f} pp`",
        "",
        "## Reverse check",
        "",
        f"- `mu_build_complete -> write (test)` strict flip: `{float(row_map[('test', 'write', 'mu_build_complete')]['strict_flip_rate']):.2%}`",
        "",
        "Interpretation:",
        "If the held-out write-derived direction remains strong on build/complete targets, the gate is better described as a cross-verb action-affordance direction than a verb-specific lexical template.",
    ]
    write_text(exp_root / "summary.md", "\n".join(lines))


def run_exp2_rank(model, args: argparse.Namespace, *, tool_token_id: int) -> None:
    exp_root = args.output_root / "exp2_toolcall_rank"
    test_pairs = load_sample_pairs(model, dataset_root=args.dataset_root, split="test", max_pairs=0)
    test_lookup = manifest_lookup(args.dataset_root, "test")
    pair_batches = build_pair_batches(test_pairs, batch_size=args.rank_batch_size)
    hook_names = [f"blocks.{layer}.hook_resid_post" for layer in EXP2_LAYERS]
    per_sample_rows: list[dict[str, object]] = []

    progress = tqdm(pair_batches, desc="Exp2 tool-call rank", dynamic_ncols=True)
    for batch in progress:
        for side in ("clean", "corrupt"):
            tokens_cpu = batch.clean_tokens_cpu if side == "clean" else batch.corrupt_tokens_cpu
            with torch.no_grad():
                _, cache = model.run_with_cache(
                    tokens_cpu.to(model.W_U.device),
                    names_filter=lambda name: name in hook_names,
                )
            for layer in EXP2_LAYERS:
                resid = cache[f"blocks.{layer}.hook_resid_post"][:, -1, :].to(device=model.W_U.device, dtype=model.W_U.dtype)
                logits = model.unembed(model.ln_final(resid.unsqueeze(1)))[:, 0, :].float()
                tool_logit = logits[:, tool_token_id]
                rank = 1 + (logits > tool_logit.unsqueeze(-1)).sum(dim=-1)
                top3 = torch.topk(logits, k=3, dim=-1).indices
                top1 = logits.argmax(dim=-1)
                tool_in_top3 = top3.eq(tool_token_id).any(dim=-1)
                for local_idx, pair_idx in enumerate(batch.indices):
                    pair = test_pairs[pair_idx]
                    per_sample_rows.append(
                        {
                            "sample_id": pair.sample_id,
                            "language": language_of(pair.sample_id, test_lookup),
                            "side": side,
                            "layer": layer,
                            "tool_logit": float(tool_logit[local_idx].item()),
                            "rank": int(rank[local_idx].item()),
                            "is_tool_call_top1": int(top1[local_idx].item() == tool_token_id),
                            "is_tool_call_top3": int(tool_in_top3[local_idx].item()),
                        }
                    )
                del resid, logits, tool_logit, rank, top3, top1, tool_in_top3
                clear_cuda()
            clear_cuda()
        progress.set_postfix(tok=batch.token_len)

    side_layer_rows: list[dict[str, object]] = []
    groups: dict[tuple[str, int], list[dict[str, object]]] = defaultdict(list)
    lang_groups: dict[tuple[str, str, int], list[dict[str, object]]] = defaultdict(list)
    for row in per_sample_rows:
        groups[(str(row["side"]), int(row["layer"]))].append(row)
        lang_groups[(str(row["language"]), str(row["side"]), int(row["layer"]))].append(row)

    for (side, layer), group in sorted(groups.items()):
        ranks = np.asarray([int(item["rank"]) for item in group], dtype=np.int64)
        top1 = np.asarray([int(item["is_tool_call_top1"]) for item in group], dtype=np.int64)
        top3 = np.asarray([int(item["is_tool_call_top3"]) for item in group], dtype=np.int64)
        logits = np.asarray([float(item["tool_logit"]) for item in group], dtype=np.float64)
        side_layer_rows.append(
            {
                "side": side,
                "layer": layer,
                "n": len(group),
                "tool_call_top1_rate": float(top1.mean()),
                "tool_call_top3_rate": float(top3.mean()),
                "rank_mean": float(ranks.mean()),
                "rank_median": float(np.median(ranks)),
                "mean_tool_logit": float(logits.mean()),
            }
        )

    language_rows: list[dict[str, object]] = []
    for (language, side, layer), group in sorted(lang_groups.items()):
        ranks = np.asarray([int(item["rank"]) for item in group], dtype=np.int64)
        top1 = np.asarray([int(item["is_tool_call_top1"]) for item in group], dtype=np.int64)
        top3 = np.asarray([int(item["is_tool_call_top3"]) for item in group], dtype=np.int64)
        language_rows.append(
            {
                "language": language,
                "side": side,
                "layer": layer,
                "n": len(group),
                "tool_call_top1_rate": float(top1.mean()),
                "tool_call_top3_rate": float(top3.mean()),
                "rank_mean": float(ranks.mean()),
                "rank_median": float(np.median(ranks)),
            }
        )

    comparison_rows: list[dict[str, object]] = []
    summary_map = {(str(row["side"]), int(row["layer"])): row for row in side_layer_rows}
    for layer in EXP2_LAYERS:
        clean_row = summary_map[("clean", layer)]
        corrupt_row = summary_map[("corrupt", layer)]
        comparison_rows.append(
            {
                "layer": layer,
                "clean_top1_rate": float(clean_row["tool_call_top1_rate"]),
                "corrupt_top1_rate": float(corrupt_row["tool_call_top1_rate"]),
                "clean_top3_rate": float(clean_row["tool_call_top3_rate"]),
                "corrupt_top3_rate": float(corrupt_row["tool_call_top3_rate"]),
                "clean_rank_median": float(clean_row["rank_median"]),
                "corrupt_rank_median": float(corrupt_row["rank_median"]),
                "clean_rank_mean": float(clean_row["rank_mean"]),
                "corrupt_rank_mean": float(corrupt_row["rank_mean"]),
            }
        )

    write_csv(exp_root / "rank_per_sample.csv", per_sample_rows)
    write_csv(exp_root / "rank_layer_side_summary.csv", side_layer_rows)
    write_csv(exp_root / "rank_language_summary.csv", language_rows)
    write_csv(exp_root / "rank_comparison_table.csv", comparison_rows)
    write_json(
        exp_root / "metadata.json",
        {
            "layers": list(EXP2_LAYERS),
            "n_pairs": len(test_pairs),
            "batch_size": args.rank_batch_size,
        },
    )

    corrupt_best = min(
        (row for row in comparison_rows if int(row["layer"]) in {28, 29, 30}),
        key=lambda row: float(row["corrupt_rank_median"]),
    )
    lines = [
        "# Exp 2: Corrupt-Side Mid-Layer `<tool_call>` Rank",
        "",
        f"- eval pairs: `{len(test_pairs)}`",
        f"- layers: `{list(EXP2_LAYERS)}`",
        f"- best corrupt median rank within L28-L30: `L{int(corrupt_best['layer'])}` -> `{float(corrupt_best['corrupt_rank_median']):.2f}`",
        f"- corresponding corrupt top1 rate: `{float(corrupt_best['corrupt_top1_rate']):.2%}`",
        "",
        "| Layer | Clean top-1 | Corrupt top-1 | Clean top-3 | Corrupt top-3 | Clean rank(med) | Corrupt rank(med) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in comparison_rows:
        lines.append(
            f"| L{int(row['layer'])} | {float(row['clean_top1_rate']):.2%} | {float(row['corrupt_top1_rate']):.2%} | "
            f"{float(row['clean_top3_rate']):.2%} | {float(row['corrupt_top3_rate']):.2%} | "
            f"{float(row['clean_rank_median']):.2f} | {float(row['corrupt_rank_median']):.2f} |"
        )
    lines.extend(
        [
            "",
            "Interpretation:",
            "If corrupt prompts already push `<tool_call>` into the top few candidates in mid layers, the late stack is better described as overriding a default tool-ready tendency than creating the tool-call option from scratch.",
        ]
    )
    write_text(exp_root / "summary.md", "\n".join(lines))


def select_early_heads(path: Path, *, top_k: int = 5) -> list[tuple[int, int]]:
    rows: list[tuple[float, int, int]] = []
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            layer = int(row["layer"])
            head = int(row["head"])
            if 25 <= layer <= 29:
                score = abs(float(row["clean_logit_delta_mean"]))
                rows.append((score, layer, head))
    rows.sort(reverse=True)
    return [(layer, head) for _score, layer, head in rows[:top_k]]


def group_heads_by_layer(heads: Sequence[tuple[int, int]]) -> dict[int, list[int]]:
    grouped: dict[int, list[int]] = defaultdict(list)
    for layer, head in heads:
        grouped[int(layer)].append(int(head))
    return {layer: sorted(head_list) for layer, head_list in sorted(grouped.items())}


def summarize_condition(rows: Sequence[dict[str, object]], condition: str) -> dict[str, object]:
    group = [row for row in rows if str(row["condition"]) == condition]
    top1 = np.asarray([int(row["is_tool_call_top1"]) for row in group], dtype=np.int64)
    logits = np.asarray([float(row["tool_logit"]) for row in group], dtype=np.float64)
    out = {
        "condition": condition,
        "n": len(group),
        "tool_call_top1_rate": float(top1.mean()),
        "mean_tool_logit": float(logits.mean()),
    }
    if "strict_drop" in group[0]:
        drop = np.asarray([int(row["strict_drop"]) for row in group], dtype=np.int64)
        out["strict_drop_rate"] = float(drop.mean())
    if "l33h29_dla" in group[0]:
        out["mean_l33h29_dla"] = float(np.asarray([float(row["l33h29_dla"]) for row in group], dtype=np.float64).mean())
        out["mean_l33h11_dla"] = float(np.asarray([float(row["l33h11_dla"]) for row in group], dtype=np.float64).mean())
        out["mean_mlp34_dla"] = float(np.asarray([float(row["mlp34_dla"]) for row in group], dtype=np.float64).mean())
    return out


def run_exp3_parallel_readout(model, args: argparse.Namespace, *, tool_token_id: int) -> None:
    exp_root = args.output_root / "exp3_parallel_readout"
    test_pairs = load_sample_pairs(model, dataset_root=args.dataset_root, split="test", max_pairs=0)
    early_heads = select_early_heads(args.early_head_csv, top_k=5)
    early_by_layer = group_heads_by_layer(early_heads)

    head_proj = precompute_head_tool_projections(model, [33], tool_token_id)[33].detach().cpu().float()
    wu_tool = model.W_U[:, tool_token_id].detach().cpu().float()
    pair_batches = build_pair_batches(test_pairs, batch_size=args.heavy_batch_size)
    target_hooks = ["blocks.33.attn.hook_z", f"blocks.{LATE_MLP_LAYER}.hook_mlp_out"]

    baseline_rows: list[dict[str, object]] = []
    early_rows: list[dict[str, object]] = []
    behavior_rows: list[dict[str, object]] = []

    progress = tqdm(pair_batches, desc="Exp3 baseline + early ablation", dynamic_ncols=True)
    for batch in progress:
        clean_tokens = batch.clean_tokens_cpu.to(model.W_U.device)
        baseline_top1_current: torch.Tensor | None = None

        for condition, fwd_hooks, storage in (
            ("baseline_clean", [], baseline_rows),
            (
                "early_band_zero",
                [(f"blocks.{layer}.attn.hook_z", make_zero_heads_hook(heads)) for layer, heads in early_by_layer.items()],
                early_rows,
            ),
        ):
            with torch.no_grad():
                logits, cache = run_with_hooks_and_cache(model, clean_tokens, hook_names=target_hooks, fwd_hooks=fwd_hooks)
            tool_logit, _tool_prob, top1 = tool_stats(logits, tool_token_id)
            z_last = cache["blocks.33.attn.hook_z"][:, -1, :, :].detach().cpu().float()
            mlp_last = cache[f"blocks.{LATE_MLP_LAYER}.hook_mlp_out"][:, -1, :].detach().cpu().float()
            l33h29 = torch.einsum("bd,d->b", z_last[:, 29, :], head_proj[29])
            l33h11 = torch.einsum("bd,d->b", z_last[:, 11, :], head_proj[11])
            mlp34 = torch.einsum("bd,d->b", mlp_last, wu_tool)
            if condition == "baseline_clean":
                baseline_top1_current = top1.clone()
            for local_idx, pair_idx in enumerate(batch.indices):
                strict_drop = 0
                if condition == "early_band_zero" and baseline_top1_current is not None:
                    strict_drop = int(
                        baseline_top1_current[local_idx].item() == tool_token_id
                        and top1[local_idx].item() != tool_token_id
                    )
                storage.append(
                    {
                        "sample_id": test_pairs[pair_idx].sample_id,
                        "condition": condition,
                        "tool_logit": float(tool_logit[local_idx].item()),
                        "is_tool_call_top1": int(top1[local_idx].item() == tool_token_id),
                        "strict_drop": strict_drop,
                        "l33h29_dla": float(l33h29[local_idx].item()),
                        "l33h11_dla": float(l33h11[local_idx].item()),
                        "mlp34_dla": float(mlp34[local_idx].item()),
                    }
                )
            clear_cuda()

        baseline_top1_flags = torch.tensor([int(row["is_tool_call_top1"]) for row in baseline_rows[-len(batch.indices) :]], dtype=torch.long)
        behavior_conditions = [
            ("late_band_zero", [(f"blocks.33.attn.hook_z", make_zero_heads_hook([29, 11])), (f"blocks.{LATE_MLP_LAYER}.hook_mlp_out", make_zero_last_token_hook())]),
            (
                "both_zero",
                [(f"blocks.{layer}.attn.hook_z", make_zero_heads_hook(heads)) for layer, heads in early_by_layer.items()]
                + [(f"blocks.33.attn.hook_z", make_zero_heads_hook([29, 11])), (f"blocks.{LATE_MLP_LAYER}.hook_mlp_out", make_zero_last_token_hook())],
            ),
        ]
        for condition, fwd_hooks in behavior_conditions:
            with torch.no_grad():
                logits = model.run_with_hooks(clean_tokens, fwd_hooks=fwd_hooks)
            tool_logit, _tool_prob, top1 = tool_stats(logits, tool_token_id)
            strict_drop = (baseline_top1_flags == 1) & (top1 != tool_token_id)
            for local_idx, pair_idx in enumerate(batch.indices):
                behavior_rows.append(
                    {
                        "sample_id": test_pairs[pair_idx].sample_id,
                        "condition": condition,
                        "tool_logit": float(tool_logit[local_idx].item()),
                        "is_tool_call_top1": int(top1[local_idx].item() == tool_token_id),
                        "strict_drop": int(strict_drop[local_idx].item()),
                    }
                )
            clear_cuda()

        progress.set_postfix(tok=batch.token_len)

    baseline_rows = [
        {
            **row,
            "strict_drop": 0,
        }
        for row in baseline_rows
    ]
    combined_rows = baseline_rows + early_rows + behavior_rows
    condition_summaries = [summarize_condition(combined_rows, condition) for condition in ("baseline_clean", "early_band_zero", "late_band_zero", "both_zero")]

    summary_map = {str(row["condition"]): row for row in condition_summaries}
    retention_rows = []
    for metric in ("mean_l33h29_dla", "mean_l33h11_dla", "mean_mlp34_dla"):
        baseline_value = float(summary_map["baseline_clean"][metric])
        early_value = float(summary_map["early_band_zero"][metric])
        retention_rows.append(
            {
                "metric": metric,
                "baseline_value": baseline_value,
                "early_band_zero_value": early_value,
                "retention_ratio": float(early_value / baseline_value) if baseline_value != 0.0 else float("nan"),
            }
        )

    write_csv(exp_root / "selected_early_heads.csv", [{"layer": layer, "head": head} for layer, head in early_heads])
    write_csv(exp_root / "per_sample_summary.csv", combined_rows)
    write_csv(exp_root / "condition_summary.csv", condition_summaries)
    write_csv(exp_root / "retention_summary.csv", retention_rows)
    write_json(
        exp_root / "metadata.json",
        {
            "early_head_csv": str(args.early_head_csv),
            "selected_early_heads": early_heads,
            "late_head_targets": list(LATE_HEAD_TARGETS),
            "late_mlp_layer": LATE_MLP_LAYER,
            "n_pairs": len(test_pairs),
        },
    )

    lines = [
        "# Exp 3: Sequential vs Parallel Readout",
        "",
        f"- selected early heads (from `{args.early_head_csv}`): `{early_heads}`",
        f"- eval pairs: `{len(test_pairs)}`",
        "",
        "## Retention after early-band ablation",
        "",
        "| metric | baseline | early-band zero | retention |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in retention_rows:
        lines.append(
            f"| {row['metric']} | {float(row['baseline_value']):.4f} | {float(row['early_band_zero_value']):.4f} | {float(row['retention_ratio']):.4f} |"
        )
    lines.extend(
        [
            "",
            "## Behavior controls",
            "",
            "| condition | top-1 | strict drop | mean tool logit |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for row in condition_summaries:
        strict_drop = float(row.get("strict_drop_rate", 0.0))
        lines.append(
            f"| {row['condition']} | {float(row['tool_call_top1_rate']):.2%} | {strict_drop:.2%} | {float(row['mean_tool_logit']):.4f} |"
        )
    lines.extend(
        [
            "",
            "Interpretation:",
            "If late-writer DLA remains high after ablating the strongest early-band heads, the late band is better described as parallel readout from the shared gate/state rather than a strict serial relay through the early attention band.",
        ]
    )
    write_text(exp_root / "summary.md", "\n".join(lines))


def run_exp4_gate_to_dla(model, args: argparse.Namespace, *, tool_token_id: int, gate_layer: int, hook_kind: str, mu_delta: torch.Tensor) -> None:
    exp_root = args.output_root / "exp4_gate_to_late_dla"
    test_pairs = load_sample_pairs(model, dataset_root=args.dataset_root, split="test", max_pairs=0)
    test_lookup = manifest_lookup(args.dataset_root, "test")
    pair_batches = build_pair_batches(test_pairs, batch_size=args.heavy_batch_size)
    z_layers = list(EXP4_LAYERS)
    hook_names = [f"blocks.{layer}.attn.hook_z" for layer in z_layers] + [f"blocks.{layer}.hook_mlp_out" for layer in z_layers]
    head_proj = precompute_head_tool_projections(model, z_layers, tool_token_id)
    wu_tool = model.W_U[:, tool_token_id].detach().cpu().float()
    gate_hook = hook_name(gate_layer, hook_kind)

    per_sample_rows: list[dict[str, object]] = []
    condition_rows: list[dict[str, object]] = []

    progress = tqdm(pair_batches, desc="Exp4 gate removal -> late DLA", dynamic_ncols=True)
    for batch in progress:
        clean_tokens = batch.clean_tokens_cpu.to(model.W_U.device)
        for condition, fwd_hooks in (
            ("baseline_clean", []),
            ("gate_removed", [(gate_hook, make_last_token_add_hook(-mu_delta))]),
        ):
            with torch.no_grad():
                logits, cache = run_with_hooks_and_cache(model, clean_tokens, hook_names=hook_names, fwd_hooks=fwd_hooks)
            tool_logit, _tool_prob, top1 = tool_stats(logits, tool_token_id)
            baseline_top1 = None
            if condition == "baseline_clean":
                baseline_top1 = top1.clone()
            condition_rows.extend(
                [
                    {
                        "sample_id": test_pairs[pair_idx].sample_id,
                        "language": language_of(test_pairs[pair_idx].sample_id, test_lookup),
                        "condition": condition,
                        "tool_logit": float(tool_logit[local_idx].item()),
                        "is_tool_call_top1": int(top1[local_idx].item() == tool_token_id),
                    }
                    for local_idx, pair_idx in enumerate(batch.indices)
                ]
            )
            for layer in z_layers:
                z_last = cache[f"blocks.{layer}.attn.hook_z"][:, -1, :, :].detach().cpu().float()
                mlp_last = cache[f"blocks.{layer}.hook_mlp_out"][:, -1, :].detach().cpu().float()
                head_dla = torch.einsum("bhd,hd->bh", z_last, head_proj[layer].detach().cpu().float()).sum(dim=-1)
                mlp_dla = torch.einsum("bd,d->b", mlp_last, wu_tool)
                total_dla = head_dla + mlp_dla
                for local_idx, pair_idx in enumerate(batch.indices):
                    per_sample_rows.append(
                        {
                            "sample_id": test_pairs[pair_idx].sample_id,
                            "language": language_of(test_pairs[pair_idx].sample_id, test_lookup),
                            "condition": condition,
                            "layer": layer,
                            "head_total_dla": float(head_dla[local_idx].item()),
                            "mlp_dla": float(mlp_dla[local_idx].item()),
                            "total_dla": float(total_dla[local_idx].item()),
                        }
                    )
            clear_cuda()
        progress.set_postfix(tok=batch.token_len)

    cond_groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in condition_rows:
        cond_groups[str(row["condition"])].append(row)
    condition_summary_rows = []
    baseline_map = {str(row["sample_id"]): int(row["is_tool_call_top1"]) for row in cond_groups["baseline_clean"]}
    for condition, group in sorted(cond_groups.items()):
        tool_top1 = np.asarray([int(item["is_tool_call_top1"]) for item in group], dtype=np.int64)
        tool_logit = np.asarray([float(item["tool_logit"]) for item in group], dtype=np.float64)
        if condition == "baseline_clean":
            strict_drop = np.zeros_like(tool_top1)
        else:
            strict_drop = np.asarray([int(baseline_map[str(item["sample_id"])] == 1 and int(item["is_tool_call_top1"]) == 0) for item in group], dtype=np.int64)
        condition_summary_rows.append(
            {
                "condition": condition,
                "n": len(group),
                "tool_call_top1_rate": float(tool_top1.mean()),
                "strict_drop_rate": float(strict_drop.mean()),
                "mean_tool_logit": float(tool_logit.mean()),
            }
        )

    per_layer_summary: list[dict[str, object]] = []
    per_language_summary: list[dict[str, object]] = []
    layer_groups: dict[tuple[int, str], list[dict[str, object]]] = defaultdict(list)
    language_groups: dict[tuple[str, int, str], list[dict[str, object]]] = defaultdict(list)
    for row in per_sample_rows:
        layer_groups[(int(row["layer"]), str(row["condition"]))].append(row)
        language_groups[(str(row["language"]), int(row["layer"]), str(row["condition"]))].append(row)

    layer_summary_map: dict[tuple[int, str], dict[str, object]] = {}
    for (layer, condition), group in sorted(layer_groups.items()):
        head_vals = np.asarray([float(item["head_total_dla"]) for item in group], dtype=np.float64)
        mlp_vals = np.asarray([float(item["mlp_dla"]) for item in group], dtype=np.float64)
        total_vals = np.asarray([float(item["total_dla"]) for item in group], dtype=np.float64)
        row = {
            "layer": layer,
            "condition": condition,
            "n": len(group),
            "mean_head_total_dla": float(head_vals.mean()),
            "mean_mlp_dla": float(mlp_vals.mean()),
            "mean_total_dla": float(total_vals.mean()),
        }
        per_layer_summary.append(row)
        layer_summary_map[(layer, condition)] = row

    for (language, layer, condition), group in sorted(language_groups.items()):
        total_vals = np.asarray([float(item["total_dla"]) for item in group], dtype=np.float64)
        per_language_summary.append(
            {
                "language": language,
                "layer": layer,
                "condition": condition,
                "n": len(group),
                "mean_total_dla": float(total_vals.mean()),
            }
        )

    drop_rows = []
    for layer in z_layers:
        baseline_row = layer_summary_map[(layer, "baseline_clean")]
        removed_row = layer_summary_map[(layer, "gate_removed")]
        baseline_total = float(baseline_row["mean_total_dla"])
        removed_total = float(removed_row["mean_total_dla"])
        drop_rows.append(
            {
                "layer": layer,
                "baseline_total_dla": baseline_total,
                "gate_removed_total_dla": removed_total,
                "absolute_drop": float(baseline_total - removed_total),
                "drop_fraction": float((baseline_total - removed_total) / baseline_total) if baseline_total != 0.0 else float("nan"),
            }
        )

    baseline_sum = sum(float(row["baseline_total_dla"]) for row in drop_rows if int(row["layer"]) >= 30)
    removed_sum = sum(float(row["gate_removed_total_dla"]) for row in drop_rows if int(row["layer"]) >= 30)
    late_drop_fraction = float((baseline_sum - removed_sum) / baseline_sum) if baseline_sum != 0.0 else float("nan")

    write_csv(exp_root / "late_dla_per_sample.csv", per_sample_rows)
    write_csv(exp_root / "condition_summary.csv", condition_summary_rows)
    write_csv(exp_root / "late_dla_layer_summary.csv", per_layer_summary)
    write_csv(exp_root / "late_dla_drop_summary.csv", drop_rows)
    write_csv(exp_root / "late_dla_language_summary.csv", per_language_summary)
    write_json(
        exp_root / "metadata.json",
        {
            "gate_layer": gate_layer,
            "hook_kind": hook_kind,
            "layers": list(z_layers),
            "n_pairs": len(test_pairs),
            "late_drop_fraction_l30_l35": late_drop_fraction,
        },
    )

    cond_map = {str(row["condition"]): row for row in condition_summary_rows}
    strongest_drop = max(drop_rows, key=lambda row: float(row["absolute_drop"]))
    lines = [
        "# Exp 4: Gate Removal -> Late-Layer DLA Drop",
        "",
        f"- gate layer: `L{gate_layer}` (`hook_resid_{hook_kind}`)",
        f"- eval pairs: `{len(test_pairs)}`",
        f"- baseline clean top-1: `{float(cond_map['baseline_clean']['tool_call_top1_rate']):.2%}`",
        f"- gate-removed top-1: `{float(cond_map['gate_removed']['tool_call_top1_rate']):.2%}`",
        f"- gate-removed strict drop: `{float(cond_map['gate_removed']['strict_drop_rate']):.2%}`",
        f"- late-layer drop fraction (L30-L35 total): `{late_drop_fraction:.2%}`",
        f"- strongest single-layer drop: `L{int(strongest_drop['layer'])}` -> `{float(strongest_drop['absolute_drop']):.4f}`",
        "",
        "| Layer | Baseline total DLA | Gate-removed total DLA | Absolute drop | Drop fraction |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in drop_rows:
        lines.append(
            f"| L{int(row['layer'])} | {float(row['baseline_total_dla']):.4f} | {float(row['gate_removed_total_dla']):.4f} | "
            f"{float(row['absolute_drop']):.4f} | {float(row['drop_fraction']):.2%} |"
        )
    lines.extend(
        [
            "",
            "Interpretation:",
            "If subtracting the gate suppresses the late-layer DLA mass, the late writers look causally downstream of the gate rather than an independent formatting circuit that merely correlates with the decision state.",
        ]
    )
    write_text(exp_root / "summary.md", "\n".join(lines))


def main() -> None:
    args = parse_args()
    experiments = normalize_experiments(args.experiments)
    args.output_root.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)

    if args.device.startswith("cuda") and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    gate_layer, hook_kind, mu_delta = load_bundle(args.pc_bundle)
    model, _tokenizer, tool_token_id = load_model_and_tokenizer(model_path=args.model_path, device=args.device)

    if "exp1" in experiments:
        run_exp1_cross_verb(model, args, tool_token_id=tool_token_id, gate_layer=gate_layer, hook_kind=hook_kind)
    if "exp2" in experiments:
        run_exp2_rank(model, args, tool_token_id=tool_token_id)
    if "exp3" in experiments:
        run_exp3_parallel_readout(model, args, tool_token_id=tool_token_id)
    if "exp4" in experiments:
        run_exp4_gate_to_dla(model, args, tool_token_id=tool_token_id, gate_layer=gate_layer, hook_kind=hook_kind, mu_delta=mu_delta)


if __name__ == "__main__":
    main()
