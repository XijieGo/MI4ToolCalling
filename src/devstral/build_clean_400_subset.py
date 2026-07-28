#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any

import torch

from dataset_utils import read_jsonl, write_json, write_text
from hf_patch_utils import TOOL_CALL_TEXT, assert_single_token, load_model, load_tokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = Path(
    os.environ.get(
        "DEVSTRAL_2_24B_PATH",
        str(PROJECT_ROOT / "external" / "models" / "Devstral-Small-2-24B-Instruct-2512"),
    )
)
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "results" / "Devstral-Small-2-24B-Instruct-2512" / "datasets"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "results" / "Devstral-Small-2-24B-Instruct-2512" / "datasets_clean400"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a 400-pair clean Devstral subset from the current 500-pair prompt set.")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--target-pairs", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--device-map", type=str, default="auto")
    parser.add_argument("--clear-output", action="store_true")
    return parser.parse_args()


def collate(tokenizer, texts: list[str]) -> dict[str, torch.Tensor]:
    encoded = [tokenizer(text, add_special_tokens=False, return_tensors="pt") for text in texts]
    return tokenizer.pad(
        {
            "input_ids": [item["input_ids"][0] for item in encoded],
            "attention_mask": [item["attention_mask"][0] for item in encoded],
        },
        padding=True,
        return_tensors="pt",
    )


def to_rows(manifest_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for row in manifest_rows:
        clean_path = Path(row["clean_path"])
        corrupt_path = Path(row["corrupt_path"])
        rows.append(
            {
                **row,
                "clean_path": str(clean_path.resolve()),
                "corrupt_path": str(corrupt_path.resolve()),
                "clean_text": clean_path.read_text(encoding="utf-8"),
                "corrupt_text": corrupt_path.read_text(encoding="utf-8"),
            }
        )
    rows.sort(key=lambda item: int(item["pair_id"]))
    return rows


def scan_side(model, tokenizer, rows: list[dict[str, Any]], *, side: str, tool_token_id: int, batch_size: int) -> dict[str, dict[str, Any]]:
    device = next(model.parameters()).device
    out: dict[str, dict[str, Any]] = {}
    for start in range(0, len(rows), batch_size):
        batch_rows = rows[start : start + batch_size]
        texts = [row[f"{side}_text"] for row in batch_rows]
        batch = collate(tokenizer, texts)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                logits_to_keep=1,
                return_dict=True,
            )
        next_token_logits = outputs.logits[:, -1, :].float()
        probs = torch.softmax(next_token_logits, dim=-1)
        top1 = next_token_logits.argmax(dim=-1)
        for idx, row in enumerate(batch_rows):
            top1_id = int(top1[idx].item())
            out[str(row["sample_id"])] = {
                f"{side}_tool_prob": float(probs[idx, tool_token_id].item()),
                f"{side}_tool_logit": float(next_token_logits[idx, tool_token_id].item()),
                f"{side}_top1_token_id": top1_id,
                f"{side}_top1_token_text": tokenizer.decode([top1_id], clean_up_tokenization_spaces=False),
                f"{side}_is_tool_call_top1": bool(top1_id == tool_token_id),
            }
        del input_ids, attention_mask, outputs, next_token_logits, probs, top1
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return out


def prepare_output_root(path: Path, clear_output: bool) -> None:
    if clear_output and path.exists():
        shutil.rmtree(path)
    (path / "clean").mkdir(parents=True, exist_ok=True)
    (path / "corrupt").mkdir(parents=True, exist_ok=True)


def materialize_subset(rows: list[dict[str, Any]], output_root: Path) -> list[dict[str, Any]]:
    manifest_rows: list[dict[str, Any]] = []
    for idx, row in enumerate(rows, start=1):
        clean_src = Path(row["clean_path"])
        corrupt_src = Path(row["corrupt_path"])
        clean_dst = output_root / "clean" / f"clean_{idx}.txt"
        corrupt_dst = output_root / "corrupt" / f"corrupt_{idx}.txt"
        shutil.copyfile(clean_src, clean_dst)
        shutil.copyfile(corrupt_src, corrupt_dst)
        manifest_rows.append(
            {
                "pair_id": idx,
                "sample_id": row["sample_id"],
                "split": row["split"],
                "language": row["language"],
                "clean_verb": row.get("clean_verb"),
                "corrupt_verb": row.get("corrupt_verb"),
                "clean_path": str(clean_dst.resolve()),
                "corrupt_path": str(corrupt_dst.resolve()),
                "clean_tool_prob": row["clean_tool_prob"],
                "corrupt_tool_prob": row["corrupt_tool_prob"],
                "clean_top1_token_text": row["clean_top1_token_text"],
                "corrupt_top1_token_text": row["corrupt_top1_token_text"],
                "score_margin": row["score_margin"],
                "selection_rank": idx,
                "selection_reason": "clean_top1_is_tool_and_corrupt_top1_is_not_tool_sorted_by_margin",
            }
        )
    return manifest_rows


def main() -> None:
    args = parse_args()
    tokenizer = load_tokenizer(args.model_path)
    tool_token_id = assert_single_token(tokenizer, TOOL_CALL_TEXT)
    dtype = getattr(torch, args.dtype)
    model = load_model(args.model_path, dtype=dtype, device_map=args.device_map, attn_implementation="eager")

    manifest_rows = read_jsonl(args.dataset_root / "manifest.jsonl")
    rows = to_rows(manifest_rows)
    clean_stats = scan_side(model, tokenizer, rows, side="clean", tool_token_id=tool_token_id, batch_size=args.batch_size)
    corrupt_stats = scan_side(model, tokenizer, rows, side="corrupt", tool_token_id=tool_token_id, batch_size=args.batch_size)

    rescored_rows = []
    for row in rows:
        sample_id = str(row["sample_id"])
        merged = {**row, **clean_stats[sample_id], **corrupt_stats[sample_id]}
        merged["score_margin"] = float(merged["clean_tool_prob"] - merged["corrupt_tool_prob"])
        rescored_rows.append(merged)

    eligible = [
        row
        for row in rescored_rows
        if row["clean_is_tool_call_top1"] and (not row["corrupt_is_tool_call_top1"])
    ]
    eligible.sort(
        key=lambda row: (
            -float(row["score_margin"]),
            -float(row["clean_tool_prob"]),
            float(row["corrupt_tool_prob"]),
            str(row["sample_id"]),
        )
    )
    selected = eligible[: args.target_pairs]
    if len(selected) < args.target_pairs:
        raise RuntimeError(f"Only found {len(selected)} eligible pairs, fewer than target {args.target_pairs}.")
    if len(selected) != len(set(str(row["sample_id"]) for row in selected)):
        raise RuntimeError("Selected subset contains duplicate sample_id values.")

    prepare_output_root(args.output_root, args.clear_output)
    new_manifest = materialize_subset(selected, args.output_root)
    manifest_path = args.output_root / "manifest.jsonl"
    manifest_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in new_manifest),
        encoding="utf-8",
    )

    summary = {
        "model_path": str(args.model_path.resolve()),
        "dataset_root": str(args.dataset_root.resolve()),
        "output_root": str(args.output_root.resolve()),
        "target_pairs": args.target_pairs,
        "total_rows": len(rows),
        "eligible_rows": len(eligible),
        "selected_rows": len(selected),
        "tool_token_text": TOOL_CALL_TEXT,
        "tool_token_id": tool_token_id,
        "clean_top1_tool_count": sum(int(row["clean_is_tool_call_top1"]) for row in rescored_rows),
        "corrupt_non_tool_count": sum(int(not row["corrupt_is_tool_call_top1"]) for row in rescored_rows),
    }
    write_json(args.output_root / "subset_summary.json", summary)
    write_text(
        args.output_root / "subset_summary.md",
        "\n".join(
            [
                "# Devstral Clean-400 Subset",
                "",
                f"- Source rows: `{len(rows)}`",
                f"- Eligible rows: `{len(eligible)}`",
                f"- Selected rows: `{len(selected)}`",
                f"- Criterion: `clean top1 = {TOOL_CALL_TEXT}` and `corrupt top1 != {TOOL_CALL_TEXT}` on rescored real prompts.",
                f"- Ranking: descending `clean_tool_prob - corrupt_tool_prob`.",
            ]
        ),
    )


if __name__ == "__main__":
    main()
