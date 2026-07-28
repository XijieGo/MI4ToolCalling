#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm.auto import tqdm

LEGACY_SRC = Path("./src")
if str(LEGACY_SRC) not in sys.path:
    sys.path.insert(0, str(LEGACY_SRC))

from toolcall_circuit.single_sample import load_hooked_qwen3  # noqa: E402


MODEL_PATH = Path("./external/models/Qwen3-8B")
DATASET_ROOT = Path("./datasets")
OUTPUT_ROOT = Path("./results/8B/full_gap_analysis")
DEFAULT_BATCH_SIZE = 24


@dataclass(frozen=True)
class PairRecord:
    sample_id: str
    clean_path: Path
    corrupt_path: Path
    token_length: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Full-dataset 8B gap analysis with 95% CI")
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH)
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--max-pairs", type=int, default=0, help="Optional cap for smoke tests; 0 means full dataset.")
    return parser.parse_args()


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, data: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def iter_manifest_rows(path: Path) -> Iterable[Dict[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def load_pairs(dataset_root: Path) -> List[PairRecord]:
    clean_root = dataset_root / "clean"
    corrupt_root = dataset_root / "corrupt"
    manifest_path = clean_root / "manifest.jsonl"
    pairs: List[PairRecord] = []
    seen_sample_ids: set[str] = set()

    for row in iter_manifest_rows(manifest_path):
        if not bool(row.get("token_length_aligned", False)):
            continue
        filename = str(row["output_filename"])
        clean_len = int(row["clean_prompt_token_length"])
        corrupt_len = int(row["corrupt_prompt_token_length"])
        if clean_len != corrupt_len:
            continue
        sample_id = filename.rsplit(".", 1)[0]
        if sample_id in seen_sample_ids:
            raise ValueError(f"Duplicate sample_id in manifest: {sample_id}")
        clean_path = clean_root / filename
        corrupt_path = corrupt_root / filename
        if not clean_path.is_file() or not corrupt_path.is_file():
            raise FileNotFoundError(f"Missing pair files for {sample_id}")
        pairs.append(
            PairRecord(
                sample_id=sample_id,
                clean_path=clean_path,
                corrupt_path=corrupt_path,
                token_length=clean_len,
            )
        )
        seen_sample_ids.add(sample_id)

    if not pairs:
        raise RuntimeError(f"No aligned pairs found in {manifest_path}")
    return pairs


def retokenize_pairs(tokenizer, pairs: Sequence[PairRecord]) -> List[PairRecord]:
    aligned: List[PairRecord] = []
    pbar = tqdm(pairs, desc="Retokenizing pairs", dynamic_ncols=True)
    for pair in pbar:
        clean_text = pair.clean_path.read_text(encoding="utf-8")
        corrupt_text = pair.corrupt_path.read_text(encoding="utf-8")
        clean_len = len(tokenizer.encode(clean_text, add_special_tokens=False))
        corrupt_len = len(tokenizer.encode(corrupt_text, add_special_tokens=False))
        if clean_len != corrupt_len:
            continue
        aligned.append(
            PairRecord(
                sample_id=pair.sample_id,
                clean_path=pair.clean_path,
                corrupt_path=pair.corrupt_path,
                token_length=clean_len,
            )
        )
        pbar.set_postfix(kept=len(aligned), token_length=clean_len)
    return aligned


def bucket_pairs_by_length(pairs: Sequence[PairRecord]) -> List[tuple[int, List[PairRecord]]]:
    buckets: Dict[int, List[PairRecord]] = {}
    for pair in pairs:
        buckets.setdefault(pair.token_length, []).append(pair)
    return sorted(buckets.items(), key=lambda item: (item[0], item[1][0].sample_id))


def load_batch_tokens(model, paths: Sequence[Path]) -> torch.Tensor:
    token_rows = [model.to_tokens(path.read_text(encoding="utf-8"), prepend_bos=False) for path in paths]
    return torch.cat(token_rows, dim=0)


def collect_resid_tool_logits(model, tokens: torch.Tensor, tool_token_id: int) -> np.ndarray:
    n_layers = int(model.cfg.n_layers)
    resid_store: Dict[int, torch.Tensor] = {}
    hooks = []

    for layer in range(n_layers):
        hook_name = f"blocks.{layer}.hook_resid_post"

        def make_hook(layer_idx: int):
            def hook_fn(act: torch.Tensor, hook):  # noqa: ANN001
                resid_store[layer_idx] = act[:, -1, :].detach()
                return act

            return hook_fn

        hooks.append((hook_name, make_hook(layer)))

    with torch.inference_mode():
        _ = model.run_with_hooks(tokens, fwd_hooks=hooks)
        resid = torch.stack([resid_store[layer] for layer in range(n_layers)], dim=1)
        flat_resid = resid.reshape(-1, resid.shape[-1]).unsqueeze(1)
        flat_logits = model.unembed(model.ln_final(flat_resid))[:, 0, tool_token_id]
        logits = flat_logits.reshape(tokens.shape[0], n_layers).detach().cpu().float().numpy()

    resid_store.clear()
    return logits


def compute_ci(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = values.shape[0]
    mean = values.mean(axis=0)
    std = values.std(axis=0, ddof=1)
    stderr = std / math.sqrt(n)
    margin = 1.96 * stderr
    return mean, mean - margin, mean + margin


def summarize_layers(clean: np.ndarray, corrupt: np.ndarray, gap: np.ndarray, diff: np.ndarray) -> List[Dict[str, object]]:
    clean_mean, clean_ci_low, clean_ci_high = compute_ci(clean)
    corrupt_mean, corrupt_ci_low, corrupt_ci_high = compute_ci(corrupt)
    gap_mean, gap_ci_low, gap_ci_high = compute_ci(gap)
    diff_mean, diff_ci_low, diff_ci_high = compute_ci(diff)
    clean_std = clean.std(axis=0, ddof=1)
    corrupt_std = corrupt.std(axis=0, ddof=1)
    gap_std = gap.std(axis=0, ddof=1)
    diff_std = diff.std(axis=0, ddof=1)

    rows: List[Dict[str, object]] = []
    for layer in range(gap.shape[1]):
        rows.append(
            {
                "layer": layer,
                "clean_mean": float(clean_mean[layer]),
                "clean_std": float(clean_std[layer]),
                "clean_ci_low": float(clean_ci_low[layer]),
                "clean_ci_high": float(clean_ci_high[layer]),
                "corrupt_mean": float(corrupt_mean[layer]),
                "corrupt_std": float(corrupt_std[layer]),
                "corrupt_ci_low": float(corrupt_ci_low[layer]),
                "corrupt_ci_high": float(corrupt_ci_high[layer]),
                "gap_mean": float(gap_mean[layer]),
                "gap_std": float(gap_std[layer]),
                "gap_ci_low": float(gap_ci_low[layer]),
                "gap_ci_high": float(gap_ci_high[layer]),
                "diff_mean": float(diff_mean[layer]),
                "diff_std": float(diff_std[layer]),
                "diff_ci_low": float(diff_ci_low[layer]),
                "diff_ci_high": float(diff_ci_high[layer]),
            }
        )
    return rows


def first_layer_ci_above_zero(rows: Sequence[Dict[str, object]], key: str) -> int | None:
    for row in rows:
        if float(row[key]) > 0.0:
            return int(row["layer"])
    return None


def top_layers_by_diff(rows: Sequence[Dict[str, object]], top_k: int = 5) -> List[Dict[str, object]]:
    return sorted(rows, key=lambda row: float(row["diff_mean"]), reverse=True)[:top_k]


def build_summary(rows: Sequence[Dict[str, object]], n_pairs: int) -> str:
    final = rows[-1]
    first_gap_pos = first_layer_ci_above_zero(rows, "gap_ci_low")
    first_diff_pos = first_layer_ci_above_zero(rows, "diff_ci_low")
    top_layers = top_layers_by_diff(rows, top_k=5)
    top_desc = ", ".join(f"L{int(row['layer'])} ({float(row['diff_mean']):.3f})" for row in top_layers)
    return (
        f"样本 {n_pairs} 对。最终层 gap 均值 {float(final['gap_mean']):.3f}"
        f"（95% CI [{float(final['gap_ci_low']):.3f}, {float(final['gap_ci_high']):.3f}]）。"
        f"gap 的 95% CI 下界首次大于 0 出现在 L{first_gap_pos if first_gap_pos is not None else 'NA'}；"
        f"单层增量 diff 的 95% CI 下界首次大于 0 出现在 L{first_diff_pos if first_diff_pos is not None else 'NA'}。"
        f"增量最大的层：{top_desc}。"
    )


def plot_gap_summary(rows: Sequence[Dict[str, object]], path: Path) -> None:
    layers = np.array([int(row["layer"]) for row in rows], dtype=np.int64)
    diff_mean = np.array([float(row["diff_mean"]) for row in rows], dtype=np.float64)
    diff_ci_low = np.array([float(row["diff_ci_low"]) for row in rows], dtype=np.float64)
    diff_ci_high = np.array([float(row["diff_ci_high"]) for row in rows], dtype=np.float64)
    gap_mean = np.array([float(row["gap_mean"]) for row in rows], dtype=np.float64)
    gap_ci_low = np.array([float(row["gap_ci_low"]) for row in rows], dtype=np.float64)
    gap_ci_high = np.array([float(row["gap_ci_high"]) for row in rows], dtype=np.float64)

    diff_yerr = np.vstack([diff_mean - diff_ci_low, diff_ci_high - diff_mean])
    gap_yerr = np.vstack([gap_mean - gap_ci_low, gap_ci_high - gap_mean])
    colors = np.where(diff_mean >= 0.0, "#2f855a", "#dd6b20")

    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True, height_ratios=[1.0, 1.2])

    axes[0].bar(layers, diff_mean, color=colors, alpha=0.9, width=0.82)
    axes[0].errorbar(layers, diff_mean, yerr=diff_yerr, fmt="none", ecolor="#1a202c", capsize=3, linewidth=1.1)
    axes[0].axhline(0.0, color="#1a202c", linewidth=1.0, alpha=0.7)
    axes[0].set_ylabel("Mean layer diff")
    axes[0].set_title("8B Full-Dataset Gap Dynamics (1500 Pairs)")
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].bar(layers, gap_mean, color="#3182ce", alpha=0.85, width=0.82)
    axes[1].errorbar(layers, gap_mean, yerr=gap_yerr, fmt="none", ecolor="#1a202c", capsize=3, linewidth=1.1)
    axes[1].axhline(0.0, color="#1a202c", linewidth=1.0, alpha=0.7)
    axes[1].set_xlabel("Layer")
    axes[1].set_ylabel("Cumulative gap")
    axes[1].grid(axis="y", alpha=0.25)

    xticks = layers[::2] if len(layers) > 18 else layers
    axes[1].set_xticks(xticks)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_logit_lens_with_ci(rows: Sequence[Dict[str, object]], path: Path) -> None:
    layers = np.array([int(row["layer"]) for row in rows], dtype=np.int64)
    clean_mean = np.array([float(row["clean_mean"]) for row in rows], dtype=np.float64)
    clean_ci_low = np.array([float(row["clean_ci_low"]) for row in rows], dtype=np.float64)
    clean_ci_high = np.array([float(row["clean_ci_high"]) for row in rows], dtype=np.float64)
    corrupt_mean = np.array([float(row["corrupt_mean"]) for row in rows], dtype=np.float64)
    corrupt_ci_low = np.array([float(row["corrupt_ci_low"]) for row in rows], dtype=np.float64)
    corrupt_ci_high = np.array([float(row["corrupt_ci_high"]) for row in rows], dtype=np.float64)
    gap_mean = np.array([float(row["gap_mean"]) for row in rows], dtype=np.float64)
    gap_ci_low = np.array([float(row["gap_ci_low"]) for row in rows], dtype=np.float64)
    gap_ci_high = np.array([float(row["gap_ci_high"]) for row in rows], dtype=np.float64)

    plt.figure(figsize=(11, 6.5))

    plt.plot(layers, clean_mean, color="#3182ce", linewidth=2.3, label="clean mean")
    plt.plot(layers, corrupt_mean, color="#dd6b20", linewidth=2.3, label="corrupt mean")
    plt.plot(layers, gap_mean, color="#2f855a", linewidth=2.3, linestyle="--", label="gap mean")

    plt.plot(layers, clean_ci_low, color="#3182ce", linewidth=1.0, linestyle=":", alpha=0.9, label="clean 95% CI")
    plt.plot(layers, clean_ci_high, color="#3182ce", linewidth=1.0, linestyle=":", alpha=0.9)
    plt.plot(layers, corrupt_ci_low, color="#dd6b20", linewidth=1.0, linestyle=":", alpha=0.9, label="corrupt 95% CI")
    plt.plot(layers, corrupt_ci_high, color="#dd6b20", linewidth=1.0, linestyle=":", alpha=0.9)
    plt.plot(layers, gap_ci_low, color="#2f855a", linewidth=1.1, linestyle="-.", alpha=0.9, label="gap 95% CI")
    plt.plot(layers, gap_ci_high, color="#2f855a", linewidth=1.1, linestyle="-.", alpha=0.9)

    plt.fill_between(layers, clean_ci_low, clean_ci_high, color="#3182ce", alpha=0.08)
    plt.fill_between(layers, corrupt_ci_low, corrupt_ci_high, color="#dd6b20", alpha=0.08)
    plt.fill_between(layers, gap_ci_low, gap_ci_high, color="#2f855a", alpha=0.08)

    plt.axhline(0.0, color="#1a202c", linewidth=0.9, alpha=0.5)
    plt.xlabel("Layer")
    plt.ylabel("<tool_call> logit")
    plt.title("8B Full-Dataset Logit Lens with 95% CI (1500 Pairs)")
    plt.xticks(layers[::2] if len(layers) > 18 else layers)
    plt.grid(alpha=0.25)
    plt.legend(ncol=2)
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=220)
    plt.close()


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    model, tokenizer = load_hooked_qwen3(str(args.model_path), device=args.device, dtype=torch.bfloat16)
    model.eval()

    tool_token_ids = tokenizer.encode("<tool_call>", add_special_tokens=False)
    if len(tool_token_ids) != 1:
        raise ValueError(f"<tool_call> is not a single token: {tool_token_ids}")
    tool_token_id = int(tool_token_ids[0])

    candidate_pairs = load_pairs(args.dataset_root)
    if args.max_pairs > 0:
        candidate_pairs = candidate_pairs[: args.max_pairs]
    pairs = retokenize_pairs(tokenizer, candidate_pairs)
    if len(pairs) < 2:
        raise RuntimeError(f"Need at least 2 aligned pairs after retokenization, got {len(pairs)}")
    pair_rows = [
        {
            "order": idx,
            "sample_id": pair.sample_id,
            "clean_path": str(pair.clean_path),
            "corrupt_path": str(pair.corrupt_path),
            "token_length": pair.token_length,
        }
        for idx, pair in enumerate(pairs)
    ]
    write_csv(args.output_root / "pair_manifest.csv", pair_rows)

    buckets = bucket_pairs_by_length(pairs)
    clean_batches: List[np.ndarray] = []
    corrupt_batches: List[np.ndarray] = []
    gap_batches: List[np.ndarray] = []

    pbar = tqdm(buckets, desc="Processing length buckets", dynamic_ncols=True)
    for token_length, bucket in pbar:
        for start in range(0, len(bucket), args.batch_size):
            batch_pairs = bucket[start : start + args.batch_size]
            clean_tokens = load_batch_tokens(model, [pair.clean_path for pair in batch_pairs])
            corrupt_tokens = load_batch_tokens(model, [pair.corrupt_path for pair in batch_pairs])
            clean_logits = collect_resid_tool_logits(model, clean_tokens, tool_token_id)
            corrupt_logits = collect_resid_tool_logits(model, corrupt_tokens, tool_token_id)
            clean_batches.append(clean_logits)
            corrupt_batches.append(corrupt_logits)
            gap_batches.append(clean_logits - corrupt_logits)

            pbar.set_postfix(length=token_length, batch=f"{start + len(batch_pairs)}/{len(bucket)}")
            del clean_tokens, corrupt_tokens, clean_logits, corrupt_logits
            torch.cuda.empty_cache()

    clean = np.concatenate(clean_batches, axis=0)
    corrupt = np.concatenate(corrupt_batches, axis=0)
    gap = np.concatenate(gap_batches, axis=0)
    diff = np.diff(gap, axis=1, prepend=np.zeros((gap.shape[0], 1), dtype=gap.dtype))
    layer_rows = summarize_layers(clean=clean, corrupt=corrupt, gap=gap, diff=diff)
    write_csv(args.output_root / "layer_gap_stats.csv", layer_rows)

    summary = build_summary(layer_rows, n_pairs=gap.shape[0])
    (args.output_root / "summary.md").write_text(summary + "\n", encoding="utf-8")
    plot_gap_summary(layer_rows, args.output_root / "gap_diff_cumulative_8B.png")
    plot_logit_lens_with_ci(layer_rows, args.output_root / "logit_lens_full_ci_8B.png")

    metadata = {
        "model_path": str(args.model_path),
        "dataset_root": str(args.dataset_root),
        "output_root": str(args.output_root),
        "tool_token_id": tool_token_id,
        "candidate_pairs": int(len(candidate_pairs)),
        "n_pairs": int(gap.shape[0]),
        "dropped_after_retokenization": int(len(candidate_pairs) - gap.shape[0]),
        "n_layers": int(gap.shape[1]),
        "batch_size": int(args.batch_size),
        "outputs": {
            "pair_manifest_csv": str(args.output_root / "pair_manifest.csv"),
            "layer_gap_stats_csv": str(args.output_root / "layer_gap_stats.csv"),
            "summary_md": str(args.output_root / "summary.md"),
            "plot_png": str(args.output_root / "gap_diff_cumulative_8B.png"),
            "logit_lens_ci_png": str(args.output_root / "logit_lens_full_ci_8B.png"),
        },
    }
    write_json(args.output_root / "metadata.json", metadata)

    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
