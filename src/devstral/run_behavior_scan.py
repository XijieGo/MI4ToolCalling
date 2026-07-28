#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer, Mistral3ForConditionalGeneration
from transformers.utils import logging as transformers_logging

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
try:
    transformers_logging.disable_progress_bar()
except Exception:
    pass
try:
    from huggingface_hub.utils import disable_progress_bars

    disable_progress_bars()
except Exception:
    pass


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = Path(
    os.environ.get(
        "DEVSTRAL_2_24B_PATH",
        str(PROJECT_ROOT / "external" / "models" / "Devstral-Small-2-24B-Instruct-2512"),
    )
)
DEFAULT_CONVERTED_ROOT = PROJECT_ROOT / "results" / "Devstral-Small-2-24B-Instruct-2512" / "converted_dataset"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "results" / "Devstral-Small-2-24B-Instruct-2512" / "behavior_scan"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scan Devstral first-token tool-call behavior on converted prompts.")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--converted-root", type=Path, default=DEFAULT_CONVERTED_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--device-map", type=str, default="auto")
    parser.add_argument("--max-pairs", type=int, default=0)
    parser.add_argument("--gpu-max-memory", type=str, default="")
    parser.add_argument("--cpu-max-memory", type=str, default="")
    parser.add_argument("--offload-folder", type=Path, default=PROJECT_ROOT / "results" / "Devstral-Small-2-24B-Instruct-2512" / "offload")
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_OUTPUT_ROOT / "checkpoints")
    parser.add_argument("--reset-checkpoints", action="store_true")
    parser.add_argument("--progress-every", type=int, default=25)
    return parser.parse_args()


def clear_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def unlink_if_exists(path: Path) -> None:
    if path.exists():
        path.unlink()


def build_model(
    model_path: Path,
    dtype_name: str,
    device_map: str,
    *,
    gpu_max_memory: str = "",
    cpu_max_memory: str = "",
    offload_folder: Path | None = None,
):
    dtype = getattr(torch, dtype_name)
    extra_kwargs: dict[str, Any] = {}
    if gpu_max_memory or cpu_max_memory:
        max_memory: dict[Any, str] = {}
        if gpu_max_memory:
            max_memory[0] = gpu_max_memory
        if cpu_max_memory:
            max_memory["cpu"] = cpu_max_memory
        extra_kwargs["max_memory"] = max_memory
    if offload_folder is not None:
        offload_folder.mkdir(parents=True, exist_ok=True)
        extra_kwargs["offload_folder"] = str(offload_folder)
    try:
        model = Mistral3ForConditionalGeneration.from_pretrained(
            str(model_path),
            dtype=dtype,
            device_map=device_map,
            trust_remote_code=True,
            **extra_kwargs,
        )
    except TypeError:
        model = Mistral3ForConditionalGeneration.from_pretrained(
            str(model_path),
            torch_dtype=dtype,
            device_map=device_map,
            trust_remote_code=True,
            **extra_kwargs,
        )
    model.eval()
    return model


def collate_batch(tokenizer, items: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    batch = tokenizer.pad(
        {
            "input_ids": [item["input_ids"] for item in items],
            "attention_mask": [item["attention_mask"] for item in items],
        },
        padding=True,
        return_tensors="pt",
    )
    return batch


def load_examples(tokenizer, manifest_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    clean_examples: list[dict[str, Any]] = []
    corrupt_examples: list[dict[str, Any]] = []
    for row in manifest_rows:
        clean_prompt_path = Path(row["clean_prompt_path"])
        corrupt_prompt_path = Path(row["corrupt_prompt_path"])
        clean_prompt = clean_prompt_path.read_text(encoding="utf-8")
        corrupt_prompt = corrupt_prompt_path.read_text(encoding="utf-8")
        clean_encoded = tokenizer(clean_prompt, add_special_tokens=False, return_tensors="pt")
        corrupt_encoded = tokenizer(corrupt_prompt, add_special_tokens=False, return_tensors="pt")

        shared = {
            "sample_id": row["sample_id"],
            "split": row["split"],
            "language": row["language"],
            "clean_candidate": row["clean_candidate"],
            "corrupt_candidate": row["corrupt_candidate"],
        }
        clean_examples.append(
            {
                **shared,
                "side": "clean",
                "prompt_path": str(clean_prompt_path.resolve()),
                "input_ids": clean_encoded["input_ids"][0],
                "attention_mask": clean_encoded["attention_mask"][0],
                "token_length": int(clean_encoded["attention_mask"][0].sum().item()),
            }
        )
        corrupt_examples.append(
            {
                **shared,
                "side": "corrupt",
                "prompt_path": str(corrupt_prompt_path.resolve()),
                "input_ids": corrupt_encoded["input_ids"][0],
                "attention_mask": corrupt_encoded["attention_mask"][0],
                "token_length": int(corrupt_encoded["attention_mask"][0].sum().item()),
            }
        )
    clean_examples.sort(key=lambda item: (item["token_length"], item["sample_id"]))
    corrupt_examples.sort(key=lambda item: (item["token_length"], item["sample_id"]))
    return clean_examples, corrupt_examples


def evaluate_side(
    *,
    model,
    tokenizer,
    examples: list[dict[str, Any]],
    tool_token_id: int,
    batch_size: int,
    checkpoint_path: Path,
    progress_every: int,
    side_name: str,
) -> dict[str, dict[str, Any]]:
    device = next(model.parameters()).device
    results: dict[str, dict[str, Any]] = {}
    if checkpoint_path.exists():
        for row in load_jsonl(checkpoint_path):
            sample_id = row.get("sample_id")
            if isinstance(sample_id, str):
                results[sample_id] = row["result"]
    remaining_examples = [item for item in examples if item["sample_id"] not in results]
    print(
        f"{side_name}: loaded {len(results)} checkpointed results, "
        f"remaining {len(remaining_examples)} / {len(examples)}."
    , flush=True)

    if not remaining_examples:
        return results

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    batch_count = (len(remaining_examples) + batch_size - 1) // batch_size
    start_time = time.time()
    with checkpoint_path.open("a", encoding="utf-8") as checkpoint_handle:
        for batch_number, start in enumerate(range(0, len(remaining_examples), batch_size), start=1):
            batch_items = remaining_examples[start : start + batch_size]
            batch_start = time.time()
            batch = collate_batch(tokenizer, batch_items)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            with torch.no_grad():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                    logits_to_keep=1,
                )

            next_token_logits = outputs.logits[:, -1, :].float()
            probabilities = torch.softmax(next_token_logits, dim=-1)
            top1_ids = next_token_logits.argmax(dim=-1)

            for item_index, item in enumerate(batch_items):
                sample_top1_id = int(top1_ids[item_index].item())
                result_row = {
                    "prompt_path": item["prompt_path"],
                    "prompt_token_length": item["token_length"],
                    "tool_token_id": tool_token_id,
                    "tool_token_logit": float(next_token_logits[item_index, tool_token_id].item()),
                    "tool_token_prob": float(probabilities[item_index, tool_token_id].item()),
                    "top1_token_id": sample_top1_id,
                    "top1_token_text": tokenizer.decode(
                        [sample_top1_id],
                        clean_up_tokenization_spaces=False,
                    ),
                    "is_tool_call_top1": bool(sample_top1_id == tool_token_id),
                }
                results[item["sample_id"]] = result_row
                checkpoint_handle.write(
                    json.dumps(
                        {
                            "sample_id": item["sample_id"],
                            "result": result_row,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            checkpoint_handle.flush()

            del batch, input_ids, attention_mask, outputs, next_token_logits, probabilities, top1_ids
            clear_cuda()
            if batch_number % max(progress_every, 1) == 0 or batch_number == batch_count:
                elapsed = time.time() - start_time
                just_elapsed = time.time() - batch_start
                print(
                    f"{side_name}: batch {batch_number}/{batch_count}, "
                    f"done {len(results)}/{len(examples)}, "
                    f"last_batch_sec={just_elapsed:.2f}, total_elapsed_min={elapsed / 60.0:.2f}"
                , flush=True)
    return results


def mode_summary(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    counter = Counter(row[key] for row in rows)
    return [
        {
            "value": value,
            "count": count,
            "rate": float(count / max(len(rows), 1)),
        }
        for value, count in counter.most_common(10)
    ]


def per_group_rates(rows: list[dict[str, Any]], field: str, side_field: str) -> dict[str, float]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for row in rows:
        grouped[str(row[field])].append(int(bool(row[side_field])))
    return {
        key: float(sum(values) / len(values))
        for key, values in sorted(grouped.items())
    }


def write_pair_decisions(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "sample_id",
        "split",
        "language",
        "clean_candidate",
        "corrupt_candidate",
        "clean_prompt_path",
        "corrupt_prompt_path",
        "clean_prompt_token_length",
        "corrupt_prompt_token_length",
        "clean_tool_token_id",
        "corrupt_tool_token_id",
        "clean_tool_token_logit",
        "corrupt_tool_token_logit",
        "clean_tool_token_prob",
        "corrupt_tool_token_prob",
        "clean_top1_token_id",
        "corrupt_top1_token_id",
        "clean_top1_token_text",
        "corrupt_top1_token_text",
        "clean_is_tool_call_top1",
        "corrupt_is_tool_call_top1",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_markdown(summary: dict[str, Any]) -> str:
    clean_rate = 100.0 * float(summary["clean_tool_call_rate"])
    corrupt_rate = 100.0 * float(summary["corrupt_tool_call_rate"])
    gap = 100.0 * float(summary["clean_minus_corrupt_gap"])
    worthwhile = "值得" if gap >= 30.0 and clean_rate > corrupt_rate else "暂不值得"
    still_separated = "仍然分离" if gap > 0 else "没有分离"
    corrupt_collapse = "没有大规模塌缩到工具调用" if corrupt_rate < 50.0 else "有明显塌缩到工具调用"
    return "\n".join(
        [
            "# Devstral 工具调用率泛化摘要",
            "",
            f"- n_pairs = {summary['n_pairs']}",
            f"- tool token = `[TOOL_CALLS]` (id = {summary['tool_token_id']})",
            f"- clean tool-call rate = {clean_rate:.2f}%",
            f"- corrupt tool-call rate = {corrupt_rate:.2f}%",
            f"- clean minus corrupt gap = {gap:.2f} pp",
            f"- borderline samples (0.25 <= p(tool) <= 0.75): clean = {summary['clean_borderline_count']}, corrupt = {summary['corrupt_borderline_count']}",
            "",
            f"1. Devstral 上这 1500 对样本 {still_separated}。",
            f"2. clean 侧{'明显偏向' if clean_rate > corrupt_rate else '没有明显偏向'}工具调用。",
            f"3. corrupt 侧{corrupt_collapse}。",
            f"4. 按当前 gap 判断，后续对 Devstral 做更深入机制实验{worthwhile}。",
            "",
            "## Top-1 token mode",
            "",
            f"- clean top-1 mode: {summary['clean_top1_token_mode'][0]['value']} ({100.0 * summary['clean_top1_token_mode'][0]['rate']:.2f}%)",
            f"- corrupt top-1 mode: {summary['corrupt_top1_token_mode'][0]['value']} ({100.0 * summary['corrupt_top1_token_mode'][0]['rate']:.2f}%)",
        ]
    ) + "\n"


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.offload_folder.mkdir(parents=True, exist_ok=True)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(str(args.model_path), trust_remote_code=True)
    tool_token_ids = tokenizer.encode("[TOOL_CALLS]", add_special_tokens=False)
    if len(tool_token_ids) != 1:
        raise RuntimeError(f"[TOOL_CALLS] should map to exactly one token, got {tool_token_ids}")
    tool_token_id = int(tool_token_ids[0])
    clean_checkpoint_path = args.checkpoint_dir / "clean_results.jsonl"
    corrupt_checkpoint_path = args.checkpoint_dir / "corrupt_results.jsonl"
    if args.reset_checkpoints:
        unlink_if_exists(clean_checkpoint_path)
        unlink_if_exists(corrupt_checkpoint_path)

    manifest_rows = load_jsonl(args.converted_root / "manifest.jsonl")
    if len(manifest_rows) != 1500:
        raise RuntimeError(f"Expected 1500 pairs in manifest, found {len(manifest_rows)}")
    if args.max_pairs > 0:
        manifest_rows = manifest_rows[: args.max_pairs]

    print(f"Loaded manifest with {len(manifest_rows)} pairs.", flush=True)
    clean_examples, corrupt_examples = load_examples(tokenizer, manifest_rows)
    print(
        f"Prepared tokenized prompts: clean={len(clean_examples)}, corrupt={len(corrupt_examples)}, "
        f"batch_size={args.batch_size}."
    , flush=True)
    model = build_model(
        args.model_path,
        args.dtype,
        args.device_map,
        gpu_max_memory=args.gpu_max_memory,
        cpu_max_memory=args.cpu_max_memory,
        offload_folder=args.offload_folder,
    )
    print("Model loaded. Starting clean-side forward pass.", flush=True)

    clean_results = evaluate_side(
        model=model,
        tokenizer=tokenizer,
        examples=clean_examples,
        tool_token_id=tool_token_id,
        batch_size=args.batch_size,
        checkpoint_path=clean_checkpoint_path,
        progress_every=args.progress_every,
        side_name="clean",
    )
    print("Clean-side forward pass finished. Starting corrupt-side forward pass.", flush=True)
    corrupt_results = evaluate_side(
        model=model,
        tokenizer=tokenizer,
        examples=corrupt_examples,
        tool_token_id=tool_token_id,
        batch_size=args.batch_size,
        checkpoint_path=corrupt_checkpoint_path,
        progress_every=args.progress_every,
        side_name="corrupt",
    )
    print("Corrupt-side forward pass finished. Building summaries.", flush=True)

    pair_rows: list[dict[str, Any]] = []
    for row in sorted(manifest_rows, key=lambda item: (item["split"], item["sample_id"])):
        clean = clean_results[row["sample_id"]]
        corrupt = corrupt_results[row["sample_id"]]
        pair_rows.append(
            {
                "sample_id": row["sample_id"],
                "split": row["split"],
                "language": row["language"],
                "clean_candidate": row["clean_candidate"],
                "corrupt_candidate": row["corrupt_candidate"],
                "clean_prompt_path": clean["prompt_path"],
                "corrupt_prompt_path": corrupt["prompt_path"],
                "clean_prompt_token_length": clean["prompt_token_length"],
                "corrupt_prompt_token_length": corrupt["prompt_token_length"],
                "clean_tool_token_id": clean["tool_token_id"],
                "corrupt_tool_token_id": corrupt["tool_token_id"],
                "clean_tool_token_logit": clean["tool_token_logit"],
                "corrupt_tool_token_logit": corrupt["tool_token_logit"],
                "clean_tool_token_prob": clean["tool_token_prob"],
                "corrupt_tool_token_prob": corrupt["tool_token_prob"],
                "clean_top1_token_id": clean["top1_token_id"],
                "corrupt_top1_token_id": corrupt["top1_token_id"],
                "clean_top1_token_text": clean["top1_token_text"],
                "corrupt_top1_token_text": corrupt["top1_token_text"],
                "clean_is_tool_call_top1": clean["is_tool_call_top1"],
                "corrupt_is_tool_call_top1": corrupt["is_tool_call_top1"],
            }
        )

    clean_rate = sum(int(bool(row["clean_is_tool_call_top1"])) for row in pair_rows) / len(pair_rows)
    corrupt_rate = sum(int(bool(row["corrupt_is_tool_call_top1"])) for row in pair_rows) / len(pair_rows)
    summary = {
        "model_path": str(args.model_path.resolve()),
        "converted_root": str(args.converted_root.resolve()),
        "output_root": str(args.output_root.resolve()),
        "n_pairs": len(pair_rows),
        "tool_token_text": "[TOOL_CALLS]",
        "tool_token_id": tool_token_id,
        "clean_tool_call_rate": float(clean_rate),
        "corrupt_tool_call_rate": float(corrupt_rate),
        "clean_minus_corrupt_gap": float(clean_rate - corrupt_rate),
        "clean_mean_tool_prob": float(sum(row["clean_tool_token_prob"] for row in pair_rows) / len(pair_rows)),
        "corrupt_mean_tool_prob": float(sum(row["corrupt_tool_token_prob"] for row in pair_rows) / len(pair_rows)),
        "clean_mean_tool_logit": float(sum(row["clean_tool_token_logit"] for row in pair_rows) / len(pair_rows)),
        "corrupt_mean_tool_logit": float(sum(row["corrupt_tool_token_logit"] for row in pair_rows) / len(pair_rows)),
        "clean_top1_token_mode": mode_summary(pair_rows, "clean_top1_token_text"),
        "corrupt_top1_token_mode": mode_summary(pair_rows, "corrupt_top1_token_text"),
        "clean_borderline_count": sum(1 for row in pair_rows if 0.25 <= row["clean_tool_token_prob"] <= 0.75),
        "corrupt_borderline_count": sum(1 for row in pair_rows if 0.25 <= row["corrupt_tool_token_prob"] <= 0.75),
        "clean_rate_by_split": per_group_rates(pair_rows, "split", "clean_is_tool_call_top1"),
        "corrupt_rate_by_split": per_group_rates(pair_rows, "split", "corrupt_is_tool_call_top1"),
        "clean_rate_by_language": per_group_rates(pair_rows, "language", "clean_is_tool_call_top1"),
        "corrupt_rate_by_language": per_group_rates(pair_rows, "language", "corrupt_is_tool_call_top1"),
        "clean_rate_by_candidate": per_group_rates(pair_rows, "clean_candidate", "clean_is_tool_call_top1"),
        "corrupt_rate_by_candidate": per_group_rates(pair_rows, "corrupt_candidate", "corrupt_is_tool_call_top1"),
    }

    write_pair_decisions(args.output_root / "pair_decisions.csv", pair_rows)
    (args.output_root / "aggregate_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_root / "aggregate_summary.md").write_text(
        build_markdown(summary),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
