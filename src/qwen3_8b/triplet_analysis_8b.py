#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from toolcall_circuit.single_sample import load_hooked_qwen3  # noqa: E402
from artifact_paths import ARTIFACT_ROOT, QWEN3_8B_PATH  # noqa: E402


SEED = 42
MAX_PAIRS = 200
MODEL_PATH = QWEN3_8B_PATH
DATASET_ROOT = ARTIFACT_ROOT / "datasets" / "train"
OUTPUT_ROOT = ARTIFACT_ROOT / "results" / "8b_main" / "triplet_analysis"


@dataclass
class SamplePair:
    sample_id: str
    clean_path: Path
    corrupt_path: Path
    clean_tokens: int
    corrupt_tokens: int


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def list_candidate_pairs(clean_root: Path, corrupt_root: Path) -> List[tuple[str, Path, Path]]:
    clean = {path.name: path for path in clean_root.iterdir() if path.is_file()}
    corrupt = {path.name: path for path in corrupt_root.iterdir() if path.is_file()}
    shared = sorted(set(clean) & set(corrupt))
    return [(name.rsplit(".", 1)[0], clean[name], corrupt[name]) for name in shared]


def select_sample_pairs(model, candidates: Sequence[tuple[str, Path, Path]], limit: int) -> List[SamplePair]:
    selected: List[SamplePair] = []
    pbar = tqdm(candidates, desc="Selecting matched pairs", dynamic_ncols=True)
    for sample_id, clean_path, corrupt_path in pbar:
        clean_text = clean_path.read_text(encoding="utf-8")
        corrupt_text = corrupt_path.read_text(encoding="utf-8")
        clean_tokens = model.to_tokens(clean_text, prepend_bos=False)
        corrupt_tokens = model.to_tokens(corrupt_text, prepend_bos=False)
        clean_len = int(clean_tokens.shape[-1])
        corrupt_len = int(corrupt_tokens.shape[-1])
        if clean_len != corrupt_len:
            continue
        selected.append(
            SamplePair(
                sample_id=sample_id,
                clean_path=clean_path,
                corrupt_path=corrupt_path,
                clean_tokens=clean_len,
                corrupt_tokens=corrupt_len,
            )
        )
        pbar.set_postfix(selected=len(selected), sample=sample_id, tok=clean_len)
        if len(selected) >= limit:
            break
    return selected


def collect_last_token_vectors(model, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    n_layers = int(model.cfg.n_layers)
    resid_store: Dict[int, torch.Tensor] = {}
    mlp_store: Dict[int, torch.Tensor] = {}
    hooks = []

    for layer in range(n_layers):
        resid_name = f"blocks.{layer}.hook_resid_post"
        mlp_name = f"blocks.{layer}.hook_mlp_out"

        def make_hook(store: Dict[int, torch.Tensor], layer_idx: int):
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


def plot_logit_lens(rows: Sequence[Dict[str, object]], path: Path) -> None:
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
    plt.title("8B Logit Lens")
    plt.xticks(layers[::2] if len(layers) > 12 else layers)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_probe(rows: Sequence[Dict[str, object]], path: Path) -> None:
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
    plt.title("8B Linear Probe")
    plt.xticks(layers[::2] if len(layers) > 12 else layers)
    plt.ylim(0.45, 1.01)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_dla(rows: Sequence[Dict[str, object]], path: Path) -> None:
    layers = [int(row["layer"]) for row in rows]
    delta = [float(row["delta"]) for row in rows]

    plt.figure(figsize=(10, 5))
    plt.bar(layers, delta, width=0.8)
    plt.xlabel("MLP Layer")
    plt.ylabel("clean - corrupt DLA")
    plt.title("8B Direct Logit Attribution")
    plt.xticks(layers[::2] if len(layers) > 12 else layers)
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def first_layer_over(rows: Sequence[Dict[str, object]], key: str, threshold: float) -> int | None:
    for row in rows:
        if float(row[key]) > threshold:
            return int(row["layer"])
    return None


def build_summary(
    n_pairs: int,
    logit_rows: Sequence[Dict[str, object]],
    probe_rows: Sequence[Dict[str, object]],
    dla_rows: Sequence[Dict[str, object]],
) -> str:
    gap_layer = first_layer_over(logit_rows, "gap", 1.0)
    auc_layer = first_layer_over(probe_rows, "cv_auc_mean", 0.9)
    top3 = sorted(dla_rows, key=lambda row: float(row["delta"]), reverse=True)[:3]
    top3_layers = [str(int(row["layer"])) for row in top3]
    top3_values = [f"{float(row['delta']):.3f}" for row in top3]
    text = (
        f"样本 {n_pairs} 对。Logit lens: gap>1 首次在 L{gap_layer if gap_layer is not None else 'NA'}；"
        f"Probe: AUC>0.9 首次在 L{auc_layer if auc_layer is not None else 'NA'}；"
        f"DLA top3: L{top3_layers[0]}, L{top3_layers[1]}, L{top3_layers[2]} "
        f"(Δ={top3_values[0]}, {top3_values[1]}, {top3_values[2]})。"
    )
    return text[:200]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="8B triplet analysis")
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH)
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--max-pairs", type=int, default=MAX_PAIRS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    args.output_root.mkdir(parents=True, exist_ok=True)

    model, tokenizer = load_hooked_qwen3(str(args.model_path), device=args.device, dtype=torch.bfloat16)
    tool_token_ids = tokenizer.encode("<tool_call>", add_special_tokens=False)
    if len(tool_token_ids) != 1:
        raise ValueError(f"<tool_call> is not a single token: {tool_token_ids}")
    tool_token_id = int(tool_token_ids[0])
    n_layers = int(model.cfg.n_layers)
    d_model = int(model.cfg.d_model)

    candidates = list_candidate_pairs(args.dataset_root / "clean", args.dataset_root / "corrupt")
    selected = select_sample_pairs(model, candidates, args.max_pairs)
    if len(selected) < args.max_pairs:
        raise RuntimeError(f"Only found {len(selected)} matched pairs with equal token length")

    manifest_rows = [
        {
            "order": idx,
            "sample_id": pair.sample_id,
            "clean_path": str(pair.clean_path),
            "corrupt_path": str(pair.corrupt_path),
            "clean_tokens": pair.clean_tokens,
            "corrupt_tokens": pair.corrupt_tokens,
        }
        for idx, pair in enumerate(selected)
    ]
    write_csv(args.output_root / "sample_manifest.csv", manifest_rows)

    n_pairs = len(selected)
    clean_resid = np.zeros((n_pairs, n_layers, d_model), dtype=np.float32)
    corrupt_resid = np.zeros((n_pairs, n_layers, d_model), dtype=np.float32)
    clean_mlp = np.zeros((n_pairs, n_layers, d_model), dtype=np.float32)
    corrupt_mlp = np.zeros((n_pairs, n_layers, d_model), dtype=np.float32)
    clean_logits = np.zeros((n_pairs, n_layers), dtype=np.float32)
    corrupt_logits = np.zeros((n_pairs, n_layers), dtype=np.float32)

    pbar = tqdm(selected, desc="Collecting activations", dynamic_ncols=True)
    for idx, pair in enumerate(pbar):
        clean_text = pair.clean_path.read_text(encoding="utf-8")
        corrupt_text = pair.corrupt_path.read_text(encoding="utf-8")
        clean_tokens = model.to_tokens(clean_text, prepend_bos=False)
        corrupt_tokens = model.to_tokens(corrupt_text, prepend_bos=False)

        resid_clean, mlp_clean = collect_last_token_vectors(model, clean_tokens)
        resid_corrupt, mlp_corrupt = collect_last_token_vectors(model, corrupt_tokens)

        clean_resid[idx] = resid_clean.numpy()
        corrupt_resid[idx] = resid_corrupt.numpy()
        clean_mlp[idx] = mlp_clean.numpy()
        corrupt_mlp[idx] = mlp_corrupt.numpy()
        clean_logits[idx] = tool_logits_from_resid(model, resid_clean, tool_token_id)
        corrupt_logits[idx] = tool_logits_from_resid(model, resid_corrupt, tool_token_id)

        pbar.set_postfix(sample=pair.sample_id)
        if idx % 25 == 0:
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
    write_csv(args.output_root / "logit_lens_8B.csv", logit_rows)
    plot_logit_lens(logit_rows, args.output_root / "logit_lens_8B.png")

    probe_rows: List[Dict[str, object]] = []
    y = np.concatenate([np.ones(n_pairs, dtype=np.int64), np.zeros(n_pairs, dtype=np.int64)])
    n_splits = min(5, n_pairs)
    if n_splits < 2:
        raise RuntimeError("Need at least 2 matched pairs for probe CV")
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    probe_pbar = tqdm(range(n_layers), desc="Training probes", dynamic_ncols=True)
    for layer in probe_pbar:
        X = np.concatenate([clean_resid[:, layer, :], corrupt_resid[:, layer, :]], axis=0)
        aucs: List[float] = []
        for train_idx, test_idx in cv.split(X, y):
            clf = make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=1000, random_state=SEED, solver="liblinear"),
            )
            clf.fit(X[train_idx], y[train_idx])
            probs = clf.predict_proba(X[test_idx])[:, 1]
            aucs.append(float(roc_auc_score(y[test_idx], probs)))
        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, random_state=SEED, solver="liblinear"),
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
    write_csv(args.output_root / "probe_auc_8B.csv", probe_rows)
    plot_probe(probe_rows, args.output_root / "probe_auc_8B.png")

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
    write_csv(args.output_root / "dla_8B.csv", dla_rows)
    plot_dla(dla_rows, args.output_root / "dla_8B.png")

    summary = build_summary(n_pairs, logit_rows, probe_rows, dla_rows)
    (args.output_root / "summary.md").write_text(summary + "\n", encoding="utf-8")

    metadata = {
        "seed": args.seed,
        "model_path": str(args.model_path),
        "dataset_root": str(args.dataset_root),
        "output_root": str(args.output_root),
        "tool_token_id": tool_token_id,
        "n_layers": n_layers,
        "d_model": d_model,
        "n_pairs": n_pairs,
        "selected_sample_ids": [pair.sample_id for pair in selected],
        "outputs": {
            "logit_lens_csv": str(args.output_root / "logit_lens_8B.csv"),
            "probe_auc_csv": str(args.output_root / "probe_auc_8B.csv"),
            "dla_csv": str(args.output_root / "dla_8B.csv"),
            "summary_md": str(args.output_root / "summary.md"),
            "sample_manifest_csv": str(args.output_root / "sample_manifest.csv"),
        },
    }
    write_json(args.output_root / "metadata.json", metadata)

    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
