#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

from multiscale_common import (
    DEFAULT_DATASET_ROOT,
    SamplePair,
    load_model_and_tokenizer,
    load_sample_pairs,
    set_seed,
    write_csv,
    write_json,
    write_text,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Triplet diagnosis for Qwen3 multi-scale tool-call analysis.")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--size-label", type=str, required=True)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--train-split", type=str, default="train")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-pairs", type=int, default=200)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def collect_last_token_vectors(model, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    n_layers = int(model.cfg.n_layers)
    resid_store: dict[int, torch.Tensor] = {}
    mlp_store: dict[int, torch.Tensor] = {}
    hooks = []

    for layer in range(n_layers):
        resid_name = f"blocks.{layer}.hook_resid_post"
        mlp_name = f"blocks.{layer}.hook_mlp_out"

        def make_hook(store: dict[int, torch.Tensor], layer_idx: int):
            def hook_fn(act: torch.Tensor, hook):  # noqa: ANN001
                store[layer_idx] = act[0, -1, :].detach().cpu().float()
                return act

            return hook_fn

        hooks.append((resid_name, make_hook(resid_store, layer)))
        hooks.append((mlp_name, make_hook(mlp_store, layer)))

    with torch.no_grad():
        _ = model.run_with_hooks(tokens, fwd_hooks=hooks)

    resid = torch.stack([resid_store[layer] for layer in range(n_layers)], dim=0)
    mlp = torch.stack([mlp_store[layer] for layer in range(n_layers)], dim=0)
    return resid, mlp


def tool_logits_from_resid(model, resid_stack: torch.Tensor, tool_token_id: int) -> np.ndarray:
    with torch.no_grad():
        resid_gpu = resid_stack.to(device=model.W_U.device, dtype=model.W_U.dtype).unsqueeze(1)
        logits = model.unembed(model.ln_final(resid_gpu))[:, 0, tool_token_id]
    return logits.detach().cpu().float().numpy()


def summarize_mean_std(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return values.mean(axis=0), values.std(axis=0, ddof=1)


def plot_logit_lens(rows: list[dict[str, object]], path: Path, size_label: str) -> None:
    layers = [int(row["layer"]) for row in rows]
    clean = [float(row["clean_mean"]) for row in rows]
    corrupt = [float(row["corrupt_mean"]) for row in rows]
    gap = [float(row["gap"]) for row in rows]

    plt.figure(figsize=(9, 5))
    plt.plot(layers, clean, label="clean", linewidth=2.0)
    plt.plot(layers, corrupt, label="corrupt", linewidth=2.0)
    plt.plot(layers, gap, label="gap", linewidth=2.0, linestyle="--")
    plt.xlabel("Layer")
    plt.ylabel("<tool_call> logit")
    plt.title(f"{size_label} Logit Lens")
    plt.xticks(layers[::2] if len(layers) > 12 else layers)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_probe(rows: list[dict[str, object]], path: Path, size_label: str) -> None:
    layers = [int(row["layer"]) for row in rows]
    cv_mean = [float(row["cv_auc_mean"]) for row in rows]
    cv_std = [float(row["cv_auc_std"]) for row in rows]
    train_auc = [float(row["train_auc"]) for row in rows]

    plt.figure(figsize=(9, 5))
    plt.plot(layers, cv_mean, label="5-fold CV AUC", linewidth=2.0)
    plt.fill_between(layers, np.array(cv_mean) - np.array(cv_std), np.array(cv_mean) + np.array(cv_std), alpha=0.2)
    plt.plot(layers, train_auc, label="train AUC", linewidth=1.8, linestyle="--")
    plt.axhline(0.9, color="gray", linestyle=":", linewidth=1.2)
    plt.xlabel("Layer")
    plt.ylabel("AUC")
    plt.title(f"{size_label} Linear Probe")
    plt.xticks(layers[::2] if len(layers) > 12 else layers)
    plt.ylim(0.45, 1.01)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_dla(rows: list[dict[str, object]], path: Path, size_label: str) -> None:
    layers = [int(row["layer"]) for row in rows]
    delta = [float(row["delta"]) for row in rows]

    plt.figure(figsize=(10, 5))
    plt.bar(layers, delta, width=0.8)
    plt.xlabel("MLP Layer")
    plt.ylabel("clean - corrupt DLA")
    plt.title(f"{size_label} Direct Logit Attribution")
    plt.xticks(layers[::2] if len(layers) > 12 else layers)
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def first_layer_over(rows: list[dict[str, object]], key: str, threshold: float) -> int | None:
    for row in rows:
        if float(row[key]) > threshold:
            return int(row["layer"])
    return None


def build_summary(
    size_label: str,
    n_pairs: int,
    logit_rows: list[dict[str, object]],
    probe_rows: list[dict[str, object]],
    dla_rows: list[dict[str, object]],
) -> str:
    gap_layer = first_layer_over(logit_rows, "gap", 1.0)
    auc_layer = first_layer_over(probe_rows, "cv_auc_mean", 0.9)
    top3 = sorted(dla_rows, key=lambda row: float(row["delta"]), reverse=True)[:3]
    top_layers = ", ".join(f"L{int(row['layer'])}" for row in top3)
    top_values = ", ".join(f"{float(row['delta']):.3f}" for row in top3)
    lines = [
        f"# {size_label} Triplet Summary",
        "",
        f"- Samples: `{n_pairs}` train pairs.",
        f"- First layer with logit gap > 1.0: `L{gap_layer}`." if gap_layer is not None else "- First layer with logit gap > 1.0: `NA`.",
        f"- First layer with probe AUC > 0.9: `L{auc_layer}`." if auc_layer is not None else "- First layer with probe AUC > 0.9: `NA`.",
        f"- Top DLA layers: `{top_layers}` with deltas `{top_values}`.",
        "",
        "Interpretation:",
        "This diagnostic output is intended to anchor the cross-scale layer range, not to serve as the main causal evidence.",
    ]
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    args.output_root.mkdir(parents=True, exist_ok=True)

    model, _tokenizer, tool_token_id = load_model_and_tokenizer(model_path=args.model_path, device=args.device)
    selected: list[SamplePair] = load_sample_pairs(
        model,
        dataset_root=args.dataset_root,
        split=args.train_split,
        max_pairs=args.max_pairs,
    )

    manifest_rows = [
        {
            "order": idx,
            "sample_id": pair.sample_id,
            "clean_path": str(pair.clean_path),
            "corrupt_path": str(pair.corrupt_path),
            "clean_tokens": pair.token_len,
            "corrupt_tokens": pair.token_len,
        }
        for idx, pair in enumerate(selected)
    ]
    write_csv(args.output_root / "sample_manifest.csv", manifest_rows)

    n_pairs = len(selected)
    n_layers = int(model.cfg.n_layers)
    d_model = int(model.cfg.d_model)
    clean_resid = np.zeros((n_pairs, n_layers, d_model), dtype=np.float32)
    corrupt_resid = np.zeros((n_pairs, n_layers, d_model), dtype=np.float32)
    clean_mlp = np.zeros((n_pairs, n_layers, d_model), dtype=np.float32)
    corrupt_mlp = np.zeros((n_pairs, n_layers, d_model), dtype=np.float32)
    clean_logits = np.zeros((n_pairs, n_layers), dtype=np.float32)
    corrupt_logits = np.zeros((n_pairs, n_layers), dtype=np.float32)

    pbar = tqdm(selected, desc=f"{args.size_label} collect", dynamic_ncols=True)
    for idx, pair in enumerate(pbar):
        clean_tokens = model.to_tokens(pair.clean_text, prepend_bos=False)
        corrupt_tokens = model.to_tokens(pair.corrupt_text, prepend_bos=False)

        resid_clean, mlp_clean = collect_last_token_vectors(model, clean_tokens)
        resid_corrupt, mlp_corrupt = collect_last_token_vectors(model, corrupt_tokens)

        clean_resid[idx] = resid_clean.numpy()
        corrupt_resid[idx] = resid_corrupt.numpy()
        clean_mlp[idx] = mlp_clean.numpy()
        corrupt_mlp[idx] = mlp_corrupt.numpy()
        clean_logits[idx] = tool_logits_from_resid(model, resid_clean, tool_token_id)
        corrupt_logits[idx] = tool_logits_from_resid(model, resid_corrupt, tool_token_id)

        pbar.set_postfix(sample=pair.sample_id)
        if idx % 25 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()

    clean_mean, _ = summarize_mean_std(clean_logits)
    corrupt_mean, _ = summarize_mean_std(corrupt_logits)
    gap = clean_logits - corrupt_logits
    gap_mean, gap_std = summarize_mean_std(gap)
    logit_rows = [
        {
            "layer": layer,
            "clean_mean": float(clean_mean[layer]),
            "corrupt_mean": float(corrupt_mean[layer]),
            "gap": float(gap_mean[layer]),
            "gap_std": float(gap_std[layer]),
        }
        for layer in range(n_layers)
    ]
    logit_csv = args.output_root / f"logit_lens_{args.size_label}.csv"
    write_csv(logit_csv, logit_rows)
    plot_logit_lens(logit_rows, args.output_root / f"logit_lens_{args.size_label}.png", args.size_label)

    probe_rows: list[dict[str, object]] = []
    y = np.concatenate([np.ones(n_pairs, dtype=np.int64), np.zeros(n_pairs, dtype=np.int64)])
    n_splits = min(5, n_pairs)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=args.seed)
    probe_pbar = tqdm(range(n_layers), desc=f"{args.size_label} probe", dynamic_ncols=True)
    for layer in probe_pbar:
        X = np.concatenate([clean_resid[:, layer, :], corrupt_resid[:, layer, :]], axis=0)
        aucs: list[float] = []
        for train_idx, test_idx in cv.split(X, y):
            clf = make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=1000, random_state=args.seed, solver="liblinear"),
            )
            clf.fit(X[train_idx], y[train_idx])
            probs = clf.predict_proba(X[test_idx])[:, 1]
            aucs.append(float(roc_auc_score(y[test_idx], probs)))
        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, random_state=args.seed, solver="liblinear"),
        )
        clf.fit(X, y)
        train_probs = clf.predict_proba(X)[:, 1]
        probe_rows.append(
            {
                "layer": layer,
                "cv_auc_mean": float(np.mean(aucs)),
                "cv_auc_std": float(np.std(aucs, ddof=1)),
                "train_auc": float(roc_auc_score(y, train_probs)),
            }
        )
    probe_csv = args.output_root / f"probe_auc_{args.size_label}.csv"
    write_csv(probe_csv, probe_rows)
    plot_probe(probe_rows, args.output_root / f"probe_auc_{args.size_label}.png", args.size_label)

    wu_tool = model.W_U[:, tool_token_id].detach().cpu().float()
    clean_attr = (torch.tensor(clean_mlp) * wu_tool.view(1, 1, -1)).sum(dim=-1)
    corrupt_attr = (torch.tensor(corrupt_mlp) * wu_tool.view(1, 1, -1)).sum(dim=-1)
    clean_dla_mean = clean_attr.mean(dim=0).numpy()
    corrupt_dla_mean = corrupt_attr.mean(dim=0).numpy()
    delta = clean_dla_mean - corrupt_dla_mean
    dla_rows = [
        {
            "layer": layer,
            "clean_mean": float(clean_dla_mean[layer]),
            "corrupt_mean": float(corrupt_dla_mean[layer]),
            "delta": float(delta[layer]),
        }
        for layer in range(n_layers)
    ]
    dla_csv = args.output_root / f"dla_{args.size_label}.csv"
    write_csv(dla_csv, dla_rows)
    plot_dla(dla_rows, args.output_root / f"dla_{args.size_label}.png", args.size_label)

    summary = build_summary(args.size_label, n_pairs, logit_rows, probe_rows, dla_rows)
    write_text(args.output_root / "summary.md", summary)

    metadata = {
        "seed": args.seed,
        "size_label": args.size_label,
        "model_path": str(args.model_path),
        "dataset_root": str(args.dataset_root),
        "train_split": args.train_split,
        "output_root": str(args.output_root),
        "tool_token_id": tool_token_id,
        "n_layers": n_layers,
        "d_model": d_model,
        "n_pairs": n_pairs,
        "selected_sample_ids": [pair.sample_id for pair in selected],
    }
    write_json(args.output_root / "metadata.json", metadata)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
