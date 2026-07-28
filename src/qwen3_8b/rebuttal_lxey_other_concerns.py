#!/usr/bin/env python3
"""Focused follow-up experiments for Reviewer LxEy's remaining concerns.

This runner deliberately uses the frozen v2_1500 split (1,200 train / 300
test) end to end.  Earlier exploratory artifacts in this repository used the
historical v1_1711 split, so they must not be mixed with the current test
manifest when describing held-out failures.

It produces three linked artifacts:

* a fresh L24 mean-difference vector fit on all 1,200 train pairs;
* a per-sample audit of the held-out ``+mu_delta`` intervention, including
  task-source, language, verb, and prompt-length failure summaries; and
* a verb-boundary sweep on the held-out task bodies, reporting each verb's
  L24 projection onto the frozen direction and its first-token behaviour.

The attention-head transfer experiment is intentionally in the companion
``rebuttal_lxey_head_attribution.py`` runner.  Keeping it separate allows the
much smaller failure audit to finish and remain inspectable independently.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import random
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
from scipy.stats import fisher_exact, spearmanr
from tqdm.auto import tqdm

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from artifact_paths import ARTIFACT_ROOT, QWEN3_8B_PATH  # noqa: E402
from toolcall_circuit.single_sample import load_hooked_qwen3  # noqa: E402
from qwen3_8b.task_attention_path_analysis import (  # noqa: E402
    PairBatch,
    Sample,
    build_pair_batches,
    clear_cuda,
    load_samples,
    set_seed,
)


TOOL_CALL = "<tool_call>"
PATCH_LAYER = 24
DEFAULT_OUTPUT_ROOT = ARTIFACT_ROOT / "results" / "runs" / "rebuttal_lxey_other_concerns_v2_1500_20260727"


def default_model_path() -> Path:
    """Use the configured path, with the workspace's standard model cache fallback."""

    candidates = (QWEN3_8B_PATH, ARTIFACT_ROOT.parent / "Qwen" / "Qwen3-8B")
    return next((candidate for candidate in candidates if candidate.is_dir()), QWEN3_8B_PATH)

# The anchors reproduce the five retained verbs on each side.  The remaining
# verbs are the Qwen3-8B intermediate-rate verbs highlighted by Tables 5--6
# (Appendix A), including ``inspect``.  Every prompt is rendered from a held-
# out v2 task body; these variants never contribute to fitting mu_delta.
DEFAULT_BOUNDARY_VERBS = (
    "add",
    "build",
    "complete",
    "save",
    "write",
    "create",
    "generate",
    "implement",
    "modify",
    "update",
    "discuss",
    "explore",
    "review",
    "study",
    "analyze",
    "assess",
    "compare",
    "examine",
    "inspect",
    "summarize",
)
ANCHOR_EXEC = {"add", "build", "complete", "save", "write"}
ANCHOR_ANALYSIS = {"discuss", "explore", "review", "study"}
BOUNDARY_EXEC = {"create", "generate", "implement", "modify", "update"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=default_model_path())
    parser.add_argument("--train-root", type=Path, default=ARTIFACT_ROOT / "datasets" / "train")
    parser.add_argument("--test-root", type=Path, default=ARTIFACT_ROOT / "datasets" / "test")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=("all", "vector", "failures", "boundary"),
        default=["all"],
        help="Stages to run. `all` runs vector, failures, then boundary.",
    )
    parser.add_argument("--train-pairs", type=int, default=1200)
    parser.add_argument("--test-pairs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--boundary-batch-size", type=int, default=8)
    parser.add_argument("--boundary-verbs", nargs="+", default=list(DEFAULT_BOUNDARY_VERBS))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse a completed vector or completed stage in --output-root after validating its metadata.",
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    ensure_dir(path.parent)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, object]) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    ensure_dir(path.parent)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def manifest_hashes(dataset_root: Path) -> dict[str, str]:
    return {
        side: sha256_file(dataset_root / side / "manifest.jsonl")
        for side in ("clean", "corrupt")
    }


def load_manifest_map(dataset_root: Path, side: str) -> dict[str, dict[str, Any]]:
    path = dataset_root / side / "manifest.jsonl"
    result: dict[str, dict[str, Any]] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        row = json.loads(raw)
        name = str(row.get("output_filename") or row.get("source_filename") or "")
        if not name:
            raise ValueError(f"Manifest row lacks output filename: {row}")
        result[Path(name).stem] = row
    return result


def mean_ci(values: np.ndarray) -> tuple[float, float]:
    if values.size == 0:
        return float("nan"), float("nan")
    mean = float(np.mean(values))
    if values.size == 1:
        return mean, 0.0
    return mean, float(np.std(values, ddof=1) / math.sqrt(values.size))


def get_tool_token_id(tokenizer) -> int:  # noqa: ANN001
    token_ids = tokenizer.encode(TOOL_CALL, add_special_tokens=False)
    if len(token_ids) != 1:
        raise ValueError(f"{TOOL_CALL!r} must be a single token, found {token_ids}")
    return int(token_ids[0])


def last_token_metrics(logits: torch.Tensor, tool_token_id: int) -> dict[str, torch.Tensor]:
    last = logits[:, -1, :].float()
    tool_logit = last[:, tool_token_id]
    top1 = last.argmax(dim=-1)
    tool_prob = torch.softmax(last, dim=-1)[:, tool_token_id]
    rank = (last > tool_logit.unsqueeze(1)).sum(dim=1).to(dtype=torch.long) + 1
    non_tool = last.clone()
    non_tool[:, tool_token_id] = -torch.inf
    best_non_tool = non_tool.max(dim=-1).values
    return {
        "tool_logit": tool_logit.detach().cpu(),
        "top1": top1.detach().cpu(),
        "tool_prob": tool_prob.detach().cpu(),
        "tool_rank": rank.detach().cpu(),
        "margin_vs_best_non_tool": (tool_logit - best_non_tool).detach().cpu(),
    }


def make_capture_hook(capture: dict[str, torch.Tensor], key: str):
    def hook_fn(value: torch.Tensor, hook):  # noqa: ANN001
        capture[key] = value[:, -1, :].detach().cpu().float()
        return value

    return hook_fn


def make_add_capture_hook(delta_cpu: torch.Tensor, capture: dict[str, torch.Tensor], key: str):
    def hook_fn(value: torch.Tensor, hook):  # noqa: ANN001
        out = value.clone()
        delta = delta_cpu.to(device=value.device, dtype=value.dtype)
        out[:, -1, :] = out[:, -1, :] + delta
        capture[key] = out[:, -1, :].detach().cpu().float()
        return out

    return hook_fn


def capture_pair_residuals(
    model,
    pairs: Sequence[Sample],
    *,
    patch_layer: int,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Capture clean/corrupt L24 pre-residuals in canonical sample order."""

    batches = build_pair_batches(pairs, batch_size)
    d_model = int(model.cfg.d_model)
    clean_resid = torch.empty((len(pairs), d_model), dtype=torch.float32)
    corrupt_resid = torch.empty((len(pairs), d_model), dtype=torch.float32)
    hook_name = f"blocks.{patch_layer}.hook_resid_pre"
    progress = tqdm(batches, desc=f"Capture L{patch_layer} train residuals", dynamic_ncols=True)
    for batch in progress:
        for side, target in (("clean", clean_resid), ("corrupt", corrupt_resid)):
            tokens_cpu = batch.clean_tokens_cpu if side == "clean" else batch.corrupt_tokens_cpu
            capture: dict[str, torch.Tensor] = {}
            with torch.no_grad():
                _ = model.run_with_hooks(
                    tokens_cpu.to(model.W_U.device),
                    fwd_hooks=[(hook_name, make_capture_hook(capture, "resid"))],
                )
            target[batch.indices] = capture["resid"]
            clear_cuda()
        progress.set_postfix(tok=batch.token_len)
    return clean_resid, corrupt_resid


def vector_paths(output_root: Path) -> tuple[Path, Path]:
    root = output_root / "vector_fit"
    return root / "vector_bundle.pt", root / "train_l24_residuals.pt"


def fit_or_load_vector(
    model,
    tokenizer,
    *,
    model_path: Path,
    train_root: Path,
    train_pairs: int,
    batch_size: int,
    output_root: Path,
    seed: int,
    resume: bool,
) -> dict[str, Any]:
    bundle_path, residual_path = vector_paths(output_root)
    expected_hashes = manifest_hashes(train_root)
    if resume and bundle_path.exists():
        bundle = torch.load(bundle_path, map_location="cpu", weights_only=False)
        if (
            isinstance(bundle, dict)
            and bundle.get("dataset_manifest_sha256") == expected_hashes
            and int(bundle.get("n_train_pairs", -1)) == int(train_pairs)
            and int(bundle.get("patch_layer", -1)) == PATCH_LAYER
        ):
            print(f"[resume] Reusing coherent vector bundle: {bundle_path}", flush=True)
            return bundle
        raise RuntimeError(
            f"Refusing to reuse incompatible vector bundle {bundle_path}. "
            "Use a new output root or rerun without --resume."
        )

    train = load_samples(train_root, model, tokenizer, max_pairs=train_pairs)
    clean_resid, corrupt_resid = capture_pair_residuals(
        model,
        train,
        patch_layer=PATCH_LAYER,
        batch_size=batch_size,
    )
    mean_clean = clean_resid.mean(dim=0)
    mean_corrupt = corrupt_resid.mean(dim=0)
    mean_diff = mean_clean - mean_corrupt
    norm = mean_diff.norm().clamp_min(1e-12)
    generator = torch.Generator(device="cpu").manual_seed(seed + 10_003)
    random_unit = torch.randn(mean_diff.shape, generator=generator, dtype=torch.float32)
    random_unit = random_unit / random_unit.norm().clamp_min(1e-12)
    bundle: dict[str, Any] = {
        "schema_version": 1,
        "model_path": str(model_path),
        "patch_layer": PATCH_LAYER,
        "hook": f"blocks.{PATCH_LAYER}.hook_resid_pre",
        "fit_split": "v2_1500/train",
        "n_train_pairs": len(train),
        "dataset_manifest_sha256": expected_hashes,
        "sample_ids": [sample.sample_id for sample in train],
        "mean_clean": mean_clean.float(),
        "mean_corrupt": mean_corrupt.float(),
        "mean_diff": mean_diff.float(),
        "mean_diff_norm": float(norm.item()),
        "unit_direction": (mean_diff / norm).float(),
        "random_unit_direction": random_unit.float(),
        "seed": int(seed),
    }
    ensure_dir(bundle_path.parent)
    torch.save(bundle, bundle_path)
    torch.save(
        {
            "sample_ids": bundle["sample_ids"],
            "patch_layer": PATCH_LAYER,
            "clean_resid": clean_resid,
            "corrupt_resid": corrupt_resid,
            "dataset_manifest_sha256": expected_hashes,
        },
        residual_path,
    )
    write_json(
        bundle_path.with_suffix(".json"),
        {
            key: value
            for key, value in bundle.items()
            if key not in {"mean_clean", "mean_corrupt", "mean_diff", "unit_direction", "random_unit_direction"}
        }
        | {"mean_diff_norm": float(norm.item())},
    )
    return bundle


def evaluate_heldout_recovery(
    model,
    tokenizer,
    *,
    test_root: Path,
    test_pairs: int,
    batch_size: int,
    bundle: dict[str, Any],
    output_root: Path,
    resume: bool,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    stage_root = output_root / "failure_audit"
    per_sample_path = stage_root / "heldout_mu_delta_per_sample.csv"
    summary_path = stage_root / "summary.json"
    expected_hashes = manifest_hashes(test_root)
    if resume and per_sample_path.exists() and summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("dataset_manifest_sha256") == expected_hashes and int(summary.get("n_test_pairs", -1)) == test_pairs:
            with per_sample_path.open(encoding="utf-8", newline="") as handle:
                rows = [dict(row) for row in csv.DictReader(handle)]
            print(f"[resume] Reusing held-out failure audit: {stage_root}", flush=True)
            return rows, summary
        raise RuntimeError(f"Refusing to reuse incompatible failure audit at {stage_root}")

    samples = load_samples(test_root, model, tokenizer, max_pairs=test_pairs)
    clean_meta = load_manifest_map(test_root, "clean")
    corrupt_meta = load_manifest_map(test_root, "corrupt")
    tool_token_id = get_tool_token_id(tokenizer)
    d_model = int(model.cfg.d_model)
    n = len(samples)
    hook_name = f"blocks.{PATCH_LAYER}.hook_resid_pre"
    mu_delta = torch.as_tensor(bundle["mean_diff"], dtype=torch.float32).reshape(-1)
    if mu_delta.numel() != d_model:
        raise ValueError(f"Vector dimension {mu_delta.numel()} does not match model d_model={d_model}")

    baseline_resid = torch.empty((n, d_model), dtype=torch.float32)
    patched_resid = torch.empty((n, d_model), dtype=torch.float32)
    metric_keys = ("tool_logit", "top1", "tool_prob", "tool_rank", "margin_vs_best_non_tool")
    baseline = {
        key: torch.empty((n,), dtype=torch.long if key in {"top1", "tool_rank"} else torch.float32)
        for key in metric_keys
    }
    patched = {
        key: torch.empty((n,), dtype=torch.long if key in {"top1", "tool_rank"} else torch.float32)
        for key in metric_keys
    }

    batches = build_pair_batches(samples, batch_size)
    progress = tqdm(batches, desc="Held-out +mu_delta failure audit", dynamic_ncols=True)
    for batch in progress:
        tokens = batch.corrupt_tokens_cpu.to(model.W_U.device)
        base_capture: dict[str, torch.Tensor] = {}
        with torch.no_grad():
            base_logits = model.run_with_hooks(tokens, fwd_hooks=[(hook_name, make_capture_hook(base_capture, "resid"))])
        base_metrics = last_token_metrics(base_logits, tool_token_id)
        baseline_resid[batch.indices] = base_capture["resid"]
        for key in metric_keys:
            baseline[key][batch.indices] = base_metrics[key]

        patch_capture: dict[str, torch.Tensor] = {}
        delta = mu_delta.unsqueeze(0).repeat(len(batch.indices), 1)
        with torch.no_grad():
            patch_logits = model.run_with_hooks(
                tokens,
                fwd_hooks=[(hook_name, make_add_capture_hook(delta, patch_capture, "resid"))],
            )
        patch_metrics = last_token_metrics(patch_logits, tool_token_id)
        patched_resid[batch.indices] = patch_capture["resid"]
        for key in metric_keys:
            patched[key][batch.indices] = patch_metrics[key]
        clear_cuda()
        progress.set_postfix(tok=batch.token_len)

    mean_corrupt = torch.as_tensor(bundle["mean_corrupt"], dtype=torch.float32)
    norm_sq = float(mu_delta.dot(mu_delta).item())
    if norm_sq <= 0:
        raise ValueError("mean_diff has zero norm")
    baseline_coord = ((baseline_resid - mean_corrupt) @ mu_delta) / norm_sq
    patched_coord = ((patched_resid - mean_corrupt) @ mu_delta) / norm_sq
    token_lengths = np.asarray([int(sample.clean_tokens_cpu.shape[-1]) for sample in samples], dtype=np.int64)
    cut1, cut2 = np.quantile(token_lengths, [1 / 3, 2 / 3])

    rows: list[dict[str, object]] = []
    for idx, sample in enumerate(samples):
        clean_row = clean_meta[sample.sample_id]
        corrupt_row = corrupt_meta[sample.sample_id]
        base_is_tool = bool(int(baseline["top1"][idx].item()) == tool_token_id)
        patch_is_tool = bool(int(patched["top1"][idx].item()) == tool_token_id)
        length = int(token_lengths[idx])
        length_bin = "short" if length <= cut1 else "medium" if length <= cut2 else "long"
        rows.append(
            {
                "sample_id": sample.sample_id,
                "task_source": str(clean_row.get("dataset_name") or clean_row.get("dataset") or "unknown"),
                "language": str(clean_row.get("language") or "unknown"),
                "clean_verb": str(clean_row.get("clean_candidate") or sample.clean_verb),
                "corrupt_verb": str(corrupt_row.get("assigned_candidate") or sample.corrupt_verb),
                "prompt_tokens": length,
                "prompt_chars": len(sample.corrupt_text),
                "length_bin": length_bin,
                "baseline_tool_top1": int(base_is_tool),
                "patched_tool_top1": int(patch_is_tool),
                "strict_recovery": int((not base_is_tool) and patch_is_tool),
                "recovery_failure": int((not base_is_tool) and (not patch_is_tool)),
                "patched_non_tool": int(not patch_is_tool),
                "baseline_tool_logit": float(baseline["tool_logit"][idx].item()),
                "patched_tool_logit": float(patched["tool_logit"][idx].item()),
                "tool_logit_change": float((patched["tool_logit"][idx] - baseline["tool_logit"][idx]).item()),
                "baseline_tool_probability": float(baseline["tool_prob"][idx].item()),
                "patched_tool_probability": float(patched["tool_prob"][idx].item()),
                "baseline_tool_rank": int(baseline["tool_rank"][idx].item()),
                "patched_tool_rank": int(patched["tool_rank"][idx].item()),
                "baseline_margin": float(baseline["margin_vs_best_non_tool"][idx].item()),
                "patched_margin": float(patched["margin_vs_best_non_tool"][idx].item()),
                "baseline_mu_coordinate": float(baseline_coord[idx].item()),
                "patched_mu_coordinate": float(patched_coord[idx].item()),
            }
        )

    ensure_dir(stage_root)
    write_csv(per_sample_path, rows)
    torch.save(
        {
            "sample_ids": [sample.sample_id for sample in samples],
            "baseline_l24_pre": baseline_resid,
            "patched_l24_pre": patched_resid,
            "dataset_manifest_sha256": expected_hashes,
            "vector_bundle": str(vector_paths(output_root)[0]),
        },
        stage_root / "heldout_l24_states.pt",
    )
    summary: dict[str, object] = {
        "schema_version": 1,
        "n_test_pairs": n,
        "dataset_manifest_sha256": expected_hashes,
        "vector_bundle": str(vector_paths(output_root)[0]),
        "patch_layer": PATCH_LAYER,
        "baseline_tool_top1_count": int(sum(int(row["baseline_tool_top1"]) for row in rows)),
        "patched_tool_top1_count": int(sum(int(row["patched_tool_top1"]) for row in rows)),
        "strict_recovery_count": int(sum(int(row["strict_recovery"]) for row in rows)),
        "recovery_failure_count": int(sum(int(row["recovery_failure"]) for row in rows)),
        "patched_non_tool_count": int(sum(int(row["patched_non_tool"]) for row in rows)),
        "baseline_tool_top1_rate": float(np.mean([row["baseline_tool_top1"] for row in rows])),
        "patched_tool_top1_rate": float(np.mean([row["patched_tool_top1"] for row in rows])),
        "strict_recovery_rate": float(np.mean([row["strict_recovery"] for row in rows])),
        "mean_tool_logit_change": float(np.mean([row["tool_logit_change"] for row in rows])),
        "mean_baseline_mu_coordinate": float(np.mean([row["baseline_mu_coordinate"] for row in rows])),
        "mean_patched_mu_coordinate": float(np.mean([row["patched_mu_coordinate"] for row in rows])),
        "length_tertiles": {"short_max": float(cut1), "medium_max": float(cut2)},
    }
    write_json(summary_path, summary)
    return rows, summary


def cluster_failure_rows(rows: Sequence[dict[str, object]], group_key: str) -> list[dict[str, object]]:
    total_n = len(rows)
    total_failures = sum(int(row["recovery_failure"]) for row in rows)
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row[group_key])].append(row)
    out: list[dict[str, object]] = []
    for value, group in sorted(grouped.items()):
        n = len(group)
        failures = sum(int(row["recovery_failure"]) for row in group)
        nonfailures = n - failures
        rest_n = total_n - n
        rest_failures = total_failures - failures
        if rest_n > 0:
            odds_ratio, p_value = fisher_exact(
                [[failures, nonfailures], [rest_failures, rest_n - rest_failures]],
                alternative="two-sided",
            )
        else:
            odds_ratio, p_value = float("nan"), float("nan")
        out.append(
            {
                "group_key": group_key,
                "group": value,
                "n": n,
                "recovery_failure_count": failures,
                "recovery_failure_rate": failures / n,
                "patched_tool_top1_rate": float(np.mean([int(row["patched_tool_top1"]) for row in group])),
                "mean_tool_logit_change": float(np.mean([float(row["tool_logit_change"]) for row in group])),
                "mean_patched_mu_coordinate": float(np.mean([float(row["patched_mu_coordinate"]) for row in group])),
                "odds_ratio_vs_rest": float(odds_ratio),
                "fisher_p_vs_rest": float(p_value),
            }
        )
    return out


def write_failure_cluster_report(
    rows: Sequence[dict[str, object]],
    summary: dict[str, object],
    *,
    output_root: Path,
) -> None:
    stage_root = output_root / "failure_audit"
    failure_rows = [row for row in rows if int(row["recovery_failure"])]
    write_csv(stage_root / "failure_cases.csv", failure_rows)
    all_clusters: list[dict[str, object]] = []
    for key in ("task_source", "language", "corrupt_verb", "clean_verb", "length_bin"):
        all_clusters.extend(cluster_failure_rows(rows, key))
    write_csv(stage_root / "failure_cluster_summary.csv", all_clusters)

    failure_count = int(summary["recovery_failure_count"])
    lines = [
        "# Held-out L24 +μΔ failure audit",
        "",
        f"- Split: frozen v2_1500 test manifest, N={int(summary['n_test_pairs'])}; the vector is fit only on v2_1500 train.",
        f"- Baseline `<tool_call>` top-1: {float(summary['baseline_tool_top1_rate']):.2%} ({int(summary['baseline_tool_top1_count'])}/{int(summary['n_test_pairs'])}).",
        f"- After `+μΔ`: {float(summary['patched_tool_top1_rate']):.2%} ({int(summary['patched_tool_top1_count'])}/{int(summary['n_test_pairs'])}).",
        f"- Strict non-tool→tool recoveries: {float(summary['strict_recovery_rate']):.2%} ({int(summary['strict_recovery_count'])}/{int(summary['n_test_pairs'])}).",
        f"- Unrecovered baseline-non-tool cases: {failure_count}.",
        "",
        "The CSV files retain every held-out item and the exact contingency summaries. With a small failure count, the per-category Fisher tests are descriptive rather than evidence for a broad population claim.",
    ]
    if failure_count:
        lines.extend(["", "## Failure IDs", ""])
        for row in failure_rows:
            lines.append(
                f"- `{row['sample_id']}`: source={row['task_source']}, language={row['language']}, "
                f"corrupt verb={row['corrupt_verb']}, tokens={row['prompt_tokens']}, "
                f"patched rank={row['patched_tool_rank']}, coordinate={float(row['patched_mu_coordinate']):.3f}."
            )
    write_text(stage_root / "summary.md", "\n".join(lines))


def verb_class(verb: str) -> str:
    if verb in ANCHOR_EXEC:
        return "anchor_execution"
    if verb in ANCHOR_ANALYSIS:
        return "anchor_analysis"
    if verb in BOUNDARY_EXEC:
        return "boundary_execution"
    return "boundary_analysis"


def replace_leading_request_verb(text: str, verb: str) -> str:
    marker = "<|im_start|>user\n"
    start = text.find(marker)
    if start < 0:
        raise ValueError("Could not find user turn marker")
    start += len(marker)
    end = text.find("\n", start)
    if end < 0:
        end = len(text)
    line = text[start:end]
    match = re.match(r"([A-Za-z]+)", line)
    if match is None:
        raise ValueError(f"Could not find leading request verb in line: {line!r}")
    replacement = verb.capitalize() if match.group(1)[0].isupper() else verb
    return text[:start] + replacement + line[match.end() :] + text[end:]


def build_variant_batches(
    model,
    samples: Sequence[Sample],
    *,
    verb: str,
    batch_size: int,
) -> tuple[list[PairBatch], list[str], list[int]]:
    """Construct one held-out variant per test body, bucketed by token length.

    PairBatch is reused as a tiny generic batch container; its corrupt tensor is
    intentionally unused here.
    """

    ids: list[str] = []
    token_lengths: list[int] = []
    token_rows: list[torch.Tensor] = []
    for sample in samples:
        rendered = replace_leading_request_verb(sample.clean_text, verb)
        tokens = model.to_tokens(rendered, prepend_bos=False).detach().cpu()
        ids.append(sample.sample_id)
        token_lengths.append(int(tokens.shape[-1]))
        token_rows.append(tokens)

    buckets: dict[int, list[tuple[int, torch.Tensor]]] = defaultdict(list)
    for idx, tokens in enumerate(token_rows):
        buckets[int(tokens.shape[-1])].append((idx, tokens))
    batches: list[PairBatch] = []
    for token_len in sorted(buckets):
        group = buckets[token_len]
        for offset in range(0, len(group), batch_size):
            chunk = group[offset : offset + batch_size]
            tensor = torch.cat([item[1] for item in chunk], dim=0)
            batches.append(
                PairBatch(
                    indices=[item[0] for item in chunk],
                    clean_tokens_cpu=tensor,
                    corrupt_tokens_cpu=tensor,
                    token_len=token_len,
                )
            )
    return batches, ids, token_lengths


def run_boundary_projection_sweep(
    model,
    tokenizer,
    *,
    test_root: Path,
    test_pairs: int,
    boundary_verbs: Sequence[str],
    batch_size: int,
    bundle: dict[str, Any],
    output_root: Path,
    resume: bool,
) -> None:
    stage_root = output_root / "boundary_verbs"
    summary_csv = stage_root / "projection_by_verb.csv"
    metadata_path = stage_root / "metadata.json"
    expected_hashes = manifest_hashes(test_root)
    clean_meta = load_manifest_map(test_root, "clean")
    if resume and summary_csv.exists() and metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if (
            metadata.get("dataset_manifest_sha256") == expected_hashes
            and int(metadata.get("n_test_pairs", -1)) == test_pairs
            and list(metadata.get("verbs", [])) == list(boundary_verbs)
        ):
            print(f"[resume] Reusing boundary-verb sweep: {stage_root}", flush=True)
            return
        raise RuntimeError(f"Refusing to reuse incompatible boundary sweep at {stage_root}")

    samples = load_samples(test_root, model, tokenizer, max_pairs=test_pairs)
    tool_token_id = get_tool_token_id(tokenizer)
    mu_delta = torch.as_tensor(bundle["mean_diff"], dtype=torch.float32).reshape(-1)
    mean_corrupt = torch.as_tensor(bundle["mean_corrupt"], dtype=torch.float32).reshape(-1)
    norm_sq = float(mu_delta.dot(mu_delta).item())
    if norm_sq <= 0:
        raise ValueError("mean_diff has zero norm")
    hook_name = f"blocks.{PATCH_LAYER}.hook_resid_pre"
    all_rows: list[dict[str, object]] = []

    seen: set[str] = set()
    verbs = [str(verb).lower() for verb in boundary_verbs if not (str(verb).lower() in seen or seen.add(str(verb).lower()))]
    for verb in verbs:
        batches, ids, token_lengths = build_variant_batches(model, samples, verb=verb, batch_size=batch_size)
        n = len(ids)
        d_model = int(model.cfg.d_model)
        residuals = torch.empty((n, d_model), dtype=torch.float32)
        metrics = {
            key: torch.empty((n,), dtype=torch.long if key in {"top1", "tool_rank"} else torch.float32)
            for key in ("tool_logit", "top1", "tool_prob", "tool_rank", "margin_vs_best_non_tool")
        }
        progress = tqdm(batches, desc=f"Boundary verb: {verb}", dynamic_ncols=True)
        for batch in progress:
            capture: dict[str, torch.Tensor] = {}
            with torch.no_grad():
                logits = model.run_with_hooks(
                    batch.clean_tokens_cpu.to(model.W_U.device),
                    fwd_hooks=[(hook_name, make_capture_hook(capture, "resid"))],
                )
            values = last_token_metrics(logits, tool_token_id)
            residuals[batch.indices] = capture["resid"]
            for key in metrics:
                metrics[key][batch.indices] = values[key]
            clear_cuda()
            progress.set_postfix(tok=batch.token_len)

        coordinate = ((residuals - mean_corrupt) @ mu_delta) / norm_sq
        for idx, sample_id in enumerate(ids):
            meta = clean_meta[sample_id]
            all_rows.append(
                {
                    "sample_id": sample_id,
                    "verb": verb,
                    "verb_class": verb_class(verb),
                    "task_source": str(meta.get("dataset_name") or meta.get("dataset") or "unknown"),
                    "language": str(meta.get("language") or "unknown"),
                    "prompt_tokens": int(token_lengths[idx]),
                    "tool_call_top1": int(int(metrics["top1"][idx].item()) == tool_token_id),
                    "tool_probability": float(metrics["tool_prob"][idx].item()),
                    "tool_logit": float(metrics["tool_logit"][idx].item()),
                    "tool_rank": int(metrics["tool_rank"][idx].item()),
                    "margin_vs_best_non_tool": float(metrics["margin_vs_best_non_tool"][idx].item()),
                    "l24_mu_coordinate": float(coordinate[idx].item()),
                }
            )

    ensure_dir(stage_root)
    write_csv(stage_root / "projection_per_sample.csv", all_rows)
    by_verb: list[dict[str, object]] = []
    for verb in verbs:
        rows = [row for row in all_rows if row["verb"] == verb]
        coordinates = np.asarray([float(row["l24_mu_coordinate"]) for row in rows], dtype=np.float64)
        probs = np.asarray([float(row["tool_probability"]) for row in rows], dtype=np.float64)
        ranks = np.asarray([int(row["tool_rank"]) for row in rows], dtype=np.float64)
        mean_coord, se_coord = mean_ci(coordinates)
        by_verb.append(
            {
                "verb": verb,
                "verb_class": verb_class(verb),
                "n": len(rows),
                "tool_call_top1_count": int(sum(int(row["tool_call_top1"]) for row in rows)),
                "tool_call_top1_rate": float(np.mean([int(row["tool_call_top1"]) for row in rows])),
                "mean_tool_probability": float(np.mean(probs)),
                "median_tool_probability": float(np.median(probs)),
                "mean_tool_rank": float(np.mean(ranks)),
                "median_tool_rank": float(np.median(ranks)),
                "mean_l24_mu_coordinate": mean_coord,
                "se_l24_mu_coordinate": se_coord,
                "median_l24_mu_coordinate": float(np.median(coordinates)),
            }
        )
    by_verb.sort(key=lambda row: float(row["mean_l24_mu_coordinate"]))
    write_csv(summary_csv, by_verb)

    top1_rates = np.asarray([float(row["tool_call_top1_rate"]) for row in by_verb], dtype=np.float64)
    projections = np.asarray([float(row["mean_l24_mu_coordinate"]) for row in by_verb], dtype=np.float64)
    rho, p_value = spearmanr(top1_rates, projections)
    metadata = {
        "schema_version": 1,
        "dataset_manifest_sha256": expected_hashes,
        "n_test_pairs": len(samples),
        "verbs": verbs,
        "vector_bundle": str(vector_paths(output_root)[0]),
        "patch_layer": PATCH_LAYER,
        "coordinate_definition": "dot(h_L24_pre - train_mean_corrupt, mu_delta) / ||mu_delta||^2; train corrupt mean=0 and clean mean=1",
        "spearman_top1_rate_vs_mean_coordinate": float(rho),
        "spearman_p_value": float(p_value),
    }
    write_json(metadata_path, metadata)
    lines = [
        "# Boundary-verb L24 projection sweep",
        "",
        "Every variant substitutes only the leading request verb in a held-out v2 test task body. The L24 direction is frozen from all 1,200 v2 train pairs.",
        "",
        "| verb | class | top-1 tool-call | mean L24 coordinate | SE | mean rank |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in by_verb:
        lines.append(
            f"| {row['verb']} | {row['verb_class']} | {float(row['tool_call_top1_rate']):.1%} | "
            f"{float(row['mean_l24_mu_coordinate']):.3f} | {float(row['se_l24_mu_coordinate']):.3f} | "
            f"{float(row['mean_tool_rank']):.1f} |"
        )
    lines.extend(
        [
            "",
            f"Across the {len(by_verb)} verbs, Spearman ρ between top-1 call rate and mean L24 coordinate is {float(rho):.3f} (two-sided p={float(p_value):.3g}).",
            "The raw per-item values are retained in `projection_per_sample.csv`.",
        ]
    )
    write_text(stage_root / "summary.md", "\n".join(lines))


def normalize_stages(raw: Iterable[str]) -> list[str]:
    items = list(raw)
    if "all" in items:
        return ["vector", "failures", "boundary"]
    return list(dict.fromkeys(items))


def main() -> None:
    args = parse_args()
    stages = normalize_stages(args.stages)
    if args.train_pairs <= 0 or args.test_pairs <= 0:
        raise ValueError("--train-pairs and --test-pairs must be positive")
    if args.batch_size <= 0 or args.boundary_batch_size <= 0:
        raise ValueError("batch sizes must be positive")
    if not args.model_path.is_dir():
        raise FileNotFoundError(f"Model path does not exist: {args.model_path}")
    ensure_dir(args.output_root)
    set_seed(args.seed)

    model, tokenizer = load_hooked_qwen3(str(args.model_path), device=args.device, dtype=torch.bfloat16)
    model.eval()
    bundle = fit_or_load_vector(
        model,
        tokenizer,
        model_path=args.model_path,
        train_root=args.train_root,
        train_pairs=args.train_pairs,
        batch_size=args.batch_size,
        output_root=args.output_root,
        seed=args.seed,
        resume=args.resume,
    )
    if "failures" in stages:
        rows, summary = evaluate_heldout_recovery(
            model,
            tokenizer,
            test_root=args.test_root,
            test_pairs=args.test_pairs,
            batch_size=args.batch_size,
            bundle=bundle,
            output_root=args.output_root,
            resume=args.resume,
        )
        write_failure_cluster_report(rows, summary, output_root=args.output_root)
    if "boundary" in stages:
        run_boundary_projection_sweep(
            model,
            tokenizer,
            test_root=args.test_root,
            test_pairs=args.test_pairs,
            boundary_verbs=args.boundary_verbs,
            batch_size=args.boundary_batch_size,
            bundle=bundle,
            output_root=args.output_root,
            resume=args.resume,
        )

    write_json(
        args.output_root / "run_metadata.json",
        {
            "stages": stages,
            "model_path": str(args.model_path),
            "train_root": str(args.train_root),
            "test_root": str(args.test_root),
            "train_pairs": int(args.train_pairs),
            "test_pairs": int(args.test_pairs),
            "batch_size": int(args.batch_size),
            "boundary_batch_size": int(args.boundary_batch_size),
            "seed": int(args.seed),
            "device": args.device,
            "resume": bool(args.resume),
        },
    )
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
