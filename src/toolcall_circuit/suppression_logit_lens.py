#!/usr/bin/env python3
"""
Layer-wise `<tool_call>` logit lens for the no-tool suppression route.

This analysis follows the split protocol:
- discover / characterize on train;
- validate the same trajectory shape on test.

It only tracks corrupt (no-tool) prompts and always measures the same
`<tool_call>` token logit, so results can be aggregated across samples.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm.auto import tqdm

if __package__ in {None, ""}:
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from toolcall_circuit.dataset import load_dataset_samples
from toolcall_circuit.single_sample import load_hooked_qwen3


def finite(values: Iterable[float]) -> List[float]:
    out: List[float] = []
    for value in values:
        try:
            num = float(value)
        except Exception:
            continue
        if math.isfinite(num):
            out.append(num)
    return out


def median(values: Iterable[float]) -> float:
    vals = finite(values)
    return float(np.median(vals)) if vals else float("nan")


def quantile(values: Iterable[float], q: float) -> float:
    vals = finite(values)
    return float(np.quantile(vals, q)) if vals else float("nan")


def fmt(value: object, digits: int = 3) -> str:
    try:
        num = float(value)
    except Exception:
        return str(value)
    if not math.isfinite(num):
        return "nan"
    return f"{num:.{digits}f}"


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, data: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def collect_stage_names(n_layers: int) -> List[str]:
    names = [f"blocks.{layer}.hook_resid_pre" for layer in range(n_layers)]
    names.append(f"blocks.{n_layers - 1}.hook_resid_post")
    return names


def stage_label_from_hook_name(hook_name: str) -> str:
    if hook_name.endswith("hook_resid_post"):
        return "final_post"
    chunk = hook_name.split(".")
    return f"{int(chunk[1])}_pre"


def display_index_from_label(label: str) -> int:
    if label == "final_post":
        return 28
    return int(label.split("_", 1)[0])


def collect_final_pos_residuals(model, tokens: torch.Tensor, stage_names: Sequence[str]) -> List[torch.Tensor]:
    recorded: Dict[str, torch.Tensor] = {}
    hooks = []
    for name in stage_names:
        def make_hook(cache_name: str):
            def hook_fn(act: torch.Tensor, hook):  # noqa: ANN001
                recorded[cache_name] = act[0, -1, :].detach().cpu().float()
                return act

            return hook_fn

        hooks.append((name, make_hook(name)))

    with torch.no_grad():
        _ = model.run_with_hooks(tokens, fwd_hooks=hooks)

    missing = [name for name in stage_names if name not in recorded]
    if missing:
        raise KeyError(f"Missing residual hooks: {missing}")
    return [recorded[name] for name in stage_names]


def tool_logits_from_residuals(model, residuals: Sequence[torch.Tensor], tool_token_id: int) -> List[float]:
    stack = torch.stack(list(residuals), dim=0).to(device=model.W_U.device, dtype=model.W_U.dtype)
    stack = stack.unsqueeze(1)
    with torch.no_grad():
        logits = model.unembed(model.ln_final(stack))
    return [float(x) for x in logits[:, 0, int(tool_token_id)].detach().cpu().tolist()]


def collect_split_rows(
    *,
    model,
    samples,
    split_name: str,
    tool_token_id: int,
) -> List[Dict[str, object]]:
    stage_names = collect_stage_names(int(model.cfg.n_layers))
    stage_labels = [stage_label_from_hook_name(name) for name in stage_names]
    rows: List[Dict[str, object]] = []

    pbar = tqdm(samples, desc=f"Suppression logit lens [{split_name}]", dynamic_ncols=True)
    for sample in pbar:
        corrupt_text = sample.corrupt_path.read_text(encoding="utf-8")
        tokens = model.to_tokens(corrupt_text, prepend_bos=False)
        residuals = collect_final_pos_residuals(model, tokens, stage_names)
        tool_logits = tool_logits_from_residuals(model, residuals, tool_token_id)

        for stage_idx, (stage_label, tool_logit) in enumerate(zip(stage_labels, tool_logits)):
            rows.append(
                {
                    "split": split_name,
                    "sample_id": sample.sample_id,
                    "stage_idx": stage_idx,
                    "stage_label": stage_label,
                    "display_idx": display_index_from_label(stage_label),
                    "tool_logit": tool_logit,
                }
            )
        pbar.set_postfix(sample=sample.sample_id)

    return rows


def summarize_rows(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    by_stage: Dict[int, List[Dict[str, object]]] = defaultdict(list)
    sample_stage: Dict[str, Dict[int, float]] = defaultdict(dict)
    for row in rows:
        stage_idx = int(row["stage_idx"])
        by_stage[stage_idx].append(dict(row))
        sample_stage[str(row["sample_id"])][stage_idx] = float(row["tool_logit"])

    summary: List[Dict[str, object]] = []
    for stage_idx in sorted(by_stage):
        members = by_stage[stage_idx]
        stage_label = str(members[0]["stage_label"])
        split_name = str(members[0]["split"])
        logits = [float(r["tool_logit"]) for r in members]
        if stage_idx == 0:
            delta_from_prev = float("nan")
        else:
            delta_from_prev = median(
                stage_map[stage_idx] - stage_map[stage_idx - 1]
                for stage_map in sample_stage.values()
                if stage_idx in stage_map and (stage_idx - 1) in stage_map
            )
        summary.append(
            {
                "split": split_name,
                "stage_idx": stage_idx,
                "stage_label": stage_label,
                "display_idx": int(members[0]["display_idx"]),
                "n_samples": len(members),
                "tool_logit_median": median(logits),
                "tool_logit_p25": quantile(logits, 0.25),
                "tool_logit_p75": quantile(logits, 0.75),
                "delta_from_prev_median": delta_from_prev,
            }
        )
    return summary


def summary_map(summary_rows: Sequence[Dict[str, object]]) -> Dict[str, Dict[str, object]]:
    return {str(row["stage_label"]): dict(row) for row in summary_rows}


def split_metrics(summary_rows: Sequence[Dict[str, object]]) -> Dict[str, object]:
    stage_map = summary_map(summary_rows)
    ordered = sorted(summary_rows, key=lambda row: int(row["stage_idx"]))
    min_row = min(ordered, key=lambda row: float(row["tool_logit_median"]))
    first_positive = next((row for row in ordered if float(row["tool_logit_median"]) > 0), None)
    metrics = {
        "start": stage_map["0_pre"],
        "after_l16": stage_map["17_pre"],
        "after_mlp17_layer": stage_map["18_pre"],
        "before_l23h6_layer": stage_map["23_pre"],
        "after_l23h6_layer": stage_map["24_pre"],
        "final": stage_map["final_post"],
        "min_row": min_row,
        "first_positive": first_positive,
    }
    return {
        "tool_logit_start_median": float(metrics["start"]["tool_logit_median"]),
        "tool_logit_after_l16_median": float(metrics["after_l16"]["tool_logit_median"]),
        "tool_logit_after_mlp17_layer_median": float(metrics["after_mlp17_layer"]["tool_logit_median"]),
        "tool_logit_before_l23h6_layer_median": float(metrics["before_l23h6_layer"]["tool_logit_median"]),
        "tool_logit_after_l23h6_layer_median": float(metrics["after_l23h6_layer"]["tool_logit_median"]),
        "tool_logit_final_median": float(metrics["final"]["tool_logit_median"]),
        "tool_logit_min_stage": str(metrics["min_row"]["stage_label"]),
        "tool_logit_min_median": float(metrics["min_row"]["tool_logit_median"]),
        "tool_logit_first_positive_stage": str(metrics["first_positive"]["stage_label"]) if metrics["first_positive"] else "",
        "tool_logit_first_positive_median": float(metrics["first_positive"]["tool_logit_median"]) if metrics["first_positive"] else float("nan"),
    }


def plot_logit_lens(
    *,
    train_summary: Sequence[Dict[str, object]],
    test_summary: Sequence[Dict[str, object]],
    out_path: Path,
) -> None:
    plt.style.use("default")
    fig, axes = plt.subplots(2, 1, figsize=(12.0, 7.2), sharex=True, sharey=True, constrained_layout=True)

    configs = [
        ("train", train_summary, axes[0], "#b23b2a"),
        ("test", test_summary, axes[1], "#3568b0"),
    ]

    combined_vals: List[float] = []
    for _split_name, rows, _ax, _color in configs:
        combined_vals.extend(float(row["tool_logit_p25"]) for row in rows)
        combined_vals.extend(float(row["tool_logit_p75"]) for row in rows)
        combined_vals.extend(float(row["tool_logit_median"]) for row in rows)

    finite_vals = finite(combined_vals)
    ymin = min(finite_vals) if finite_vals else -1.0
    ymax = max(finite_vals) if finite_vals else 1.0
    pad = max(0.3, 0.08 * (ymax - ymin if ymax > ymin else 1.0))

    anchor_lines = [
        (17, "after L16 / L16H4"),
        (18, "after L17 / MLP17"),
        (24, "after L23 / L23H6"),
    ]

    for split_name, rows, ax, color in configs:
        xs = [int(row["display_idx"]) for row in rows]
        med = [float(row["tool_logit_median"]) for row in rows]
        p25 = [float(row["tool_logit_p25"]) for row in rows]
        p75 = [float(row["tool_logit_p75"]) for row in rows]

        ax.fill_between(xs, p25, p75, color=color, alpha=0.18, linewidth=0.0)
        ax.plot(xs, med, color=color, linewidth=2.3, marker="o", markersize=3.2)
        ax.axhline(0.0, color="#666666", linewidth=1.0)
        for x, label in anchor_lines:
            ax.axvline(x, color="#999999", linestyle="--", linewidth=1.0)
            ax.text(x + 0.15, ymax + pad * 0.05, label, rotation=90, va="bottom", ha="left", fontsize=9, color="#444444")
        ax.set_title(f"{split_name.upper()} corrupt prompts")
        ax.set_ylabel("<tool_call> direct logit")
        ax.set_ylim(ymin - pad, ymax + pad)

    axes[-1].set_xlabel("Residual stage (`k_pre`, plus `final_post`)")
    xticks = [0, 4, 8, 12, 16, 20, 24, 28]
    xtick_labels = ["0_pre", "4_pre", "8_pre", "12_pre", "16_pre", "20_pre", "24_pre", "final"]
    axes[-1].set_xticks(xticks, xtick_labels)
    fig.suptitle("Suppression Logit Lens on Corrupt Prompts", fontsize=14)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build split train/test no-tool logit-lens summaries for the suppression module.")
    parser.add_argument(
        "--train-dataset-root",
        type=Path,
        default=Path("./datasets/train"),
    )
    parser.add_argument(
        "--test-dataset-root",
        type=Path,
        default=Path("./datasets/test"),
    )
    parser.add_argument("--model-path", type=str, default="./external/models/Qwen3-1.7B")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--train-output-root",
        type=Path,
        default=Path("./results/split/tool_call_suppression"),
    )
    parser.add_argument(
        "--test-output-root",
        type=Path,
        default=Path("./results/split/test_validation"),
    )
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-test-samples", type=int, default=0)
    args = parser.parse_args()

    train_samples = load_dataset_samples(args.train_dataset_root.resolve())
    test_samples = load_dataset_samples(args.test_dataset_root.resolve())
    if args.max_train_samples > 0:
        train_samples = train_samples[: args.max_train_samples]
    if args.max_test_samples > 0:
        test_samples = test_samples[: args.max_test_samples]

    model, tokenizer = load_hooked_qwen3(args.model_path, device=args.device, dtype=torch.bfloat16)
    tool_token_ids = tokenizer.encode("<tool_call>", add_special_tokens=False)
    if len(tool_token_ids) != 1:
        raise ValueError(f"`<tool_call>` is not a single token: {tool_token_ids}")
    tool_token_id = int(tool_token_ids[0])

    train_rows = collect_split_rows(
        model=model,
        samples=train_samples,
        split_name="train",
        tool_token_id=tool_token_id,
    )
    test_rows = collect_split_rows(
        model=model,
        samples=test_samples,
        split_name="test",
        tool_token_id=tool_token_id,
    )

    train_summary = summarize_rows(train_rows)
    test_summary = summarize_rows(test_rows)

    train_root = args.train_output_root.resolve()
    test_root = args.test_output_root.resolve()
    write_csv(train_root / "suppression_logit_lens_train_per_sample.csv", train_rows)
    write_csv(train_root / "suppression_logit_lens_train_summary.csv", train_summary)
    write_csv(test_root / "suppression_logit_lens_test_per_sample.csv", test_rows)
    write_csv(test_root / "suppression_logit_lens_test_summary.csv", test_summary)

    fig_path = train_root / "figures" / "suppression_logit_lens.png"
    plot_logit_lens(train_summary=train_summary, test_summary=test_summary, out_path=fig_path)

    report = {
        "tool_token_text": "<tool_call>",
        "tool_token_id": tool_token_id,
        "train_n_samples": len(train_samples),
        "test_n_samples": len(test_samples),
        "train_metrics": split_metrics(train_summary),
        "test_metrics": split_metrics(test_summary),
        "artifacts": {
            "train_per_sample_csv": str(train_root / "suppression_logit_lens_train_per_sample.csv"),
            "train_summary_csv": str(train_root / "suppression_logit_lens_train_summary.csv"),
            "test_per_sample_csv": str(test_root / "suppression_logit_lens_test_per_sample.csv"),
            "test_summary_csv": str(test_root / "suppression_logit_lens_test_summary.csv"),
            "figure_png": str(fig_path),
        },
    }
    write_json(train_root / "suppression_logit_lens_summary.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
