#!/usr/bin/env python3
"""Audit MLP34 patching on an explicit paired-prompt test set.

The historical routing script replaces the entire MLP activation sequence.
The manuscript describes a patch at the next-token prediction position.  This
runner evaluates both meanings on exactly the same clean/corrupt pairs and
reports post-intervention top-1 and strict recovery separately.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch


ROOT = Path("/root/autodl-tmp")
LEGACY_SRC = ROOT / "project-new/tool-call-mechanism-artifact/src"
MODEL_PATH = ROOT / "Qwen/Qwen3-8B"
TOOL_CALL = "<tool_call>"
MLP_LAYER = 34
DEFAULT_OUTPUT_ROOT = ROOT / "MI4Toolcalling/results/qwen3_8b/historical_source_audit_20261005"


@dataclass(frozen=True)
class Pair:
    index: int
    sample_id: str
    token_length: int
    clean_tokens: torch.Tensor
    corrupt_tokens: torch.Tensor


@dataclass(frozen=True)
class Batch:
    indices: list[int]
    sample_ids: list[str]
    token_length: int
    clean_tokens: torch.Tensor
    corrupt_tokens: torch.Tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--use-hook-mlp-in",
        action="store_true",
        help=(
            "Enable TransformerLens's hook_mlp_in compatibility setting. "
            "The preserved phase-6 and routing scripts enable it, so this flag "
            "audits that exact loader state separately from the default state."
        ),
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_pairs(model, dataset_root: Path) -> list[Pair]:
    manifest = dataset_root / "clean" / "manifest.jsonl"
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    pairs: list[Pair] = []
    with manifest.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            filename = row.get("output_filename") or row.get("source_filename")
            if not filename:
                continue
            sample_id = Path(str(filename)).stem
            clean_path = dataset_root / "clean" / f"{sample_id}.txt"
            corrupt_path = dataset_root / "corrupt" / f"{sample_id}.txt"
            if not clean_path.is_file() or not corrupt_path.is_file():
                continue
            clean_tokens = model.to_tokens(clean_path.read_text(encoding="utf-8"), prepend_bos=False).detach().cpu()
            corrupt_tokens = model.to_tokens(corrupt_path.read_text(encoding="utf-8"), prepend_bos=False).detach().cpu()
            if clean_tokens.shape[-1] != corrupt_tokens.shape[-1]:
                continue
            pairs.append(
                Pair(
                    index=len(pairs),
                    sample_id=sample_id,
                    token_length=int(clean_tokens.shape[-1]),
                    clean_tokens=clean_tokens,
                    corrupt_tokens=corrupt_tokens,
                )
            )
    if not pairs:
        raise RuntimeError(f"No aligned pairs found below {dataset_root}")
    return pairs


def make_batches(pairs: list[Pair], batch_size: int) -> list[Batch]:
    buckets: dict[int, list[Pair]] = defaultdict(list)
    for pair in pairs:
        buckets[pair.token_length].append(pair)
    batches: list[Batch] = []
    for token_length in sorted(buckets):
        bucket = buckets[token_length]
        for start in range(0, len(bucket), batch_size):
            chunk = bucket[start : start + batch_size]
            batches.append(
                Batch(
                    indices=[pair.index for pair in chunk],
                    sample_ids=[pair.sample_id for pair in chunk],
                    token_length=token_length,
                    clean_tokens=torch.cat([pair.clean_tokens for pair in chunk], dim=0),
                    corrupt_tokens=torch.cat([pair.corrupt_tokens for pair in chunk], dim=0),
                )
            )
    return batches


def logit_stats(logits: torch.Tensor, tool_id: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    last = logits[:, -1, :].float()
    tool_logit = last[:, tool_id]
    competitors = last.clone()
    competitors[:, tool_id] = -torch.inf
    margin = tool_logit - competitors.max(dim=-1).values
    top1 = last.argmax(dim=-1)
    return tool_logit.detach().cpu(), margin.detach().cpu(), top1.detach().cpu()


def make_patch_hook(source: torch.Tensor, *, prediction_position_only: bool) -> Callable:
    def hook_fn(value: torch.Tensor, hook):  # noqa: ANN001
        out = value.clone()
        src = source.to(device=value.device, dtype=value.dtype)
        if prediction_position_only:
            out[:, -1, :] = src[:, -1, :]
        else:
            out.copy_(src)
        return out

    return hook_fn


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summarize(records: list[dict[str, object]], key: str, tool_id: int) -> dict[str, float | int]:
    n = len(records)
    base_tool = sum(int(record["corrupt_top1"]) == tool_id for record in records)
    patched_tool = sum(int(record[f"{key}_top1"]) == tool_id for record in records)
    strict = sum(
        int(record["corrupt_top1"]) != tool_id and int(record[f"{key}_top1"]) == tool_id
        for record in records
    )
    return {
        "n_pairs": n,
        "baseline_corrupt_tool_top1_count": base_tool,
        "baseline_corrupt_tool_top1_rate": base_tool / n,
        "patched_tool_top1_count": patched_tool,
        "patched_tool_top1_rate": patched_tool / n,
        "strict_recovery_count": strict,
        "strict_recovery_rate": strict / n,
        "mean_tool_logit_delta": sum(float(record[f"{key}_tool_logit"]) - float(record["corrupt_tool_logit"]) for record in records) / n,
        "mean_margin_delta": sum(float(record[f"{key}_margin"]) - float(record["corrupt_margin"]) for record in records) / n,
    }


def main() -> None:
    args = parse_args()
    output_root = args.output_root / args.label / "mlp34_patching"
    output_root.mkdir(parents=True, exist_ok=True)
    if str(LEGACY_SRC) not in sys.path:
        sys.path.insert(0, str(LEGACY_SRC))
    from toolcall_circuit.single_sample import load_hooked_qwen3

    dtype = torch.bfloat16
    model, _tokenizer = load_hooked_qwen3(str(MODEL_PATH), args.device, dtype)
    if args.use_hook_mlp_in:
        model.set_use_hook_mlp_in(True)
    token_ids = model.tokenizer.encode(TOOL_CALL, add_special_tokens=False)
    if len(token_ids) != 1:
        raise ValueError(f"{TOOL_CALL} is not a single token: {token_ids}")
    tool_id = int(token_ids[0])
    pairs = load_pairs(model, args.dataset_root)
    batches = make_batches(pairs, args.batch_size)
    hook_name = f"blocks.{MLP_LAYER}.hook_mlp_out"
    records: list[dict[str, object] | None] = [None] * len(pairs)

    print(f"[setup] {len(pairs)} aligned pairs in {len(batches)} equal-length batches", flush=True)
    for batch_index, batch in enumerate(batches, start=1):
        clean = batch.clean_tokens.to(model.cfg.device)
        corrupt = batch.corrupt_tokens.to(model.cfg.device)
        with torch.no_grad():
            clean_logits, clean_cache = model.run_with_cache(clean, names_filter=lambda name: name == hook_name)
            corrupt_logits, _corrupt_cache = model.run_with_cache(corrupt, names_filter=lambda name: name == hook_name)
            source = clean_cache[hook_name].detach()
            prediction_logits = model.run_with_hooks(
                corrupt,
                fwd_hooks=[(hook_name, make_patch_hook(source, prediction_position_only=True))],
            )
            full_logits = model.run_with_hooks(
                corrupt,
                fwd_hooks=[(hook_name, make_patch_hook(source, prediction_position_only=False))],
            )

        clean_tool, clean_margin, clean_top1 = logit_stats(clean_logits, tool_id)
        corrupt_tool, corrupt_margin, corrupt_top1 = logit_stats(corrupt_logits, tool_id)
        prediction_tool, prediction_margin, prediction_top1 = logit_stats(prediction_logits, tool_id)
        full_tool, full_margin, full_top1 = logit_stats(full_logits, tool_id)
        for local, pair_index in enumerate(batch.indices):
            records[pair_index] = {
                "sample_id": batch.sample_ids[local],
                "token_length": batch.token_length,
                "clean_tool_logit": float(clean_tool[local]),
                "clean_margin": float(clean_margin[local]),
                "clean_top1": int(clean_top1[local]),
                "corrupt_tool_logit": float(corrupt_tool[local]),
                "corrupt_margin": float(corrupt_margin[local]),
                "corrupt_top1": int(corrupt_top1[local]),
                "prediction_position_tool_logit": float(prediction_tool[local]),
                "prediction_position_margin": float(prediction_margin[local]),
                "prediction_position_top1": int(prediction_top1[local]),
                "full_sequence_tool_logit": float(full_tool[local]),
                "full_sequence_margin": float(full_margin[local]),
                "full_sequence_top1": int(full_top1[local]),
            }
        del clean, corrupt, clean_logits, corrupt_logits, prediction_logits, full_logits, clean_cache, _corrupt_cache, source
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if batch_index % 25 == 0 or batch_index == len(batches):
            print(f"[progress] {batch_index}/{len(batches)} batches", flush=True)

    finished_records = [record for record in records if record is not None]
    if len(finished_records) != len(pairs):
        raise RuntimeError("Missing per-sample results")
    rows = [dict(record) for record in finished_records]
    prediction = summarize(rows, "prediction_position", tool_id)
    full = summarize(rows, "full_sequence", tool_id)
    clean_top1_count = sum(int(row["clean_top1"]) == tool_id for row in rows)
    metadata = {
        "label": args.label,
        "dataset_root": str(args.dataset_root.resolve()),
        "clean_manifest_sha256": sha256_file(args.dataset_root / "clean/manifest.jsonl"),
        "corrupt_manifest_sha256": sha256_file(args.dataset_root / "corrupt/manifest.jsonl"),
        "model_path": str(MODEL_PATH),
        "model_dtype": str(dtype),
        "use_hook_mlp_in": bool(args.use_hook_mlp_in),
        "tool_token": TOOL_CALL,
        "tool_token_id": tool_id,
        "layer": MLP_LAYER,
        "n_pairs": len(rows),
        "clean_tool_top1_count": clean_top1_count,
        "clean_tool_top1_rate": clean_top1_count / len(rows),
        "patches": {
            "prediction_position_only": prediction,
            "full_sequence_legacy": full,
        },
    }
    (output_root / "mlp34_patch_summary.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    write_csv(output_root / "mlp34_patch_per_sample.csv", rows)
    md = [
        "# MLP34 Patching Audit",
        "",
        f"- Evaluation set: `{len(rows)}` aligned pairs from `{args.dataset_root}`.",
        f"- Patch hook: `{hook_name}`.",
        "- Both variants use the clean activation from the paired prompt; only the patched sequence positions differ.",
        "",
        "| Variant | Post-patch `<tool_call>` top-1 | Strict recovery | Mean call-logit delta | Mean margin delta |",
        "|---|---:|---:|---:|---:|",
    ]
    for label, result in (("prediction position only", prediction), ("full sequence (legacy)", full)):
        md.append(
            f"| {label} | {int(result['patched_tool_top1_count'])}/{len(rows)} ({float(result['patched_tool_top1_rate']):.2%}) | "
            f"{int(result['strict_recovery_count'])}/{len(rows)} ({float(result['strict_recovery_rate']):.2%}) | "
            f"{float(result['mean_tool_logit_delta']):+.4f} | {float(result['mean_margin_delta']):+.4f} |"
        )
    md.extend(
        [
            "",
            f"Corrupt baseline `<tool_call>` top-1: {int(prediction['baseline_corrupt_tool_top1_count'])}/{len(rows)} ({float(prediction['baseline_corrupt_tool_top1_rate']):.2%}).",
            "",
            "The full-sequence row is included solely to audit the historical implementation. The prediction-position row is the causal patch described in the manuscript.",
        ]
    )
    (output_root / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps(metadata["patches"], indent=2), flush=True)


if __name__ == "__main__":
    main()
