#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gc
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "external" / "models" / "Qwen3.5-9B"
DEFAULT_CONVERTED_ROOT = PROJECT_ROOT / "results" / "Qwen3.5-9B" / "converted_dataset"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "results" / "Qwen3.5-9B" / "behavior_scan"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scan Qwen3.5-9B first-step tool-call behavior on converted prompts.")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--converted-root", type=Path, default=DEFAULT_CONVERTED_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--device-map", type=str, default="auto")
    parser.add_argument("--max-pairs", type=int, default=0)
    parser.add_argument("--start-index", type=int, default=0)
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def clear_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def decode_token(tokenizer, token_id: int) -> str:
    return tokenizer.decode([int(token_id)], clean_up_tokenization_spaces=False)


def build_max_memory(device_map: str) -> dict[Any, str] | None:
    if device_map != "auto" or not torch.cuda.is_available():
        return None
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    free_gib = max(int(free_bytes // (1024**3)) - 4, 16)
    total_gib = int(total_bytes // (1024**3))
    free_gib = min(free_gib, total_gib - 2)
    return {0: f"{free_gib}GiB", "cpu": "400GiB"}


def load_model(model_path: Path, dtype_name: str, device_map: str, output_root: Path):
    dtype = getattr(torch, dtype_name)
    kwargs: dict[str, Any] = {
        "torch_dtype": dtype,
        "device_map": device_map,
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
    }
    max_memory = build_max_memory(device_map)
    if max_memory is not None:
        offload_folder = output_root / "offload"
        ensure_dir(offload_folder)
        kwargs["max_memory"] = max_memory
        kwargs["offload_folder"] = str(offload_folder)
    model = AutoModelForCausalLM.from_pretrained(str(model_path), **kwargs)
    model.eval()
    return model


def build_prompt_tensors(
    tokenizer,
    canonical_rows: list[dict[str, Any]],
    manifest_by_id: dict[str, dict[str, Any]],
    *,
    max_pairs: int,
    start_index: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    stop_index = start_index + max_pairs if max_pairs > 0 else len(canonical_rows)
    for canonical_row in canonical_rows[start_index:stop_index]:
        sample_id = str(canonical_row["sample_id"])
        manifest_row = manifest_by_id[sample_id]
        clean_prompt = Path(manifest_row["clean_prompt_path"]).read_text(encoding="utf-8")
        corrupt_prompt = Path(manifest_row["corrupt_prompt_path"]).read_text(encoding="utf-8")
        clean_enc = tokenizer(clean_prompt, add_special_tokens=False, return_tensors="pt")
        corrupt_enc = tokenizer(corrupt_prompt, add_special_tokens=False, return_tensors="pt")
        rows.append(
            {
                **canonical_row,
                **manifest_row,
                "clean_input_ids": clean_enc["input_ids"][0],
                "clean_attention_mask": clean_enc["attention_mask"][0],
                "corrupt_input_ids": corrupt_enc["input_ids"][0],
                "corrupt_attention_mask": corrupt_enc["attention_mask"][0],
            }
        )
    return rows


def iter_chunks(items: list[dict[str, Any]], batch_size: int):
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def get_model_input_device(model) -> torch.device:
    try:
        return model.get_input_embeddings().weight.device
    except Exception:
        return next(model.parameters()).device


def evaluate_batch(model, tokenizer, prompts: list[dict[str, Any]], tool_token_id: int, side: str) -> list[dict[str, Any]]:
    input_ids = [row[f"{side}_input_ids"] for row in prompts]
    attention_masks = [row[f"{side}_attention_mask"] for row in prompts]
    padded = tokenizer.pad(
        {"input_ids": input_ids, "attention_mask": attention_masks},
        return_tensors="pt",
        padding=True,
    )
    device = get_model_input_device(model)
    input_ids_tensor = padded["input_ids"].to(device)
    attention_mask_tensor = padded["attention_mask"].to(device)
    with torch.inference_mode():
        outputs = model(
            input_ids=input_ids_tensor,
            attention_mask=attention_mask_tensor,
            use_cache=False,
            logits_to_keep=1,
        )
    logits = outputs.logits[:, -1, :].float().cpu()
    attention_mask_cpu = padded["attention_mask"].cpu()
    lengths = attention_mask_cpu.sum(dim=1).long()
    results: list[dict[str, Any]] = []
    for idx, row in enumerate(prompts):
        last_logits = logits[idx]
        probs = torch.softmax(last_logits, dim=-1)
        top2_probs, top2_ids = torch.topk(probs, k=2)
        top1_id = int(top2_ids[0].item())
        top1_prob = float(top2_probs[0].item())
        second_prob = float(top2_probs[1].item()) if top2_probs.numel() > 1 else 0.0
        tool_prob = float(probs[tool_token_id].item())
        tool_logit = float(last_logits[tool_token_id].item())
        results.append(
            {
                f"{side}_prompt_token_length": int(lengths[idx].item()),
                f"{side}_tool_token_id": int(tool_token_id),
                f"{side}_tool_token_logit": tool_logit,
                f"{side}_tool_token_prob": tool_prob,
                f"{side}_top1_token_id": top1_id,
                f"{side}_top1_token_text": decode_token(tokenizer, top1_id),
                f"{side}_top1_prob": top1_prob,
                f"{side}_top1_margin": top1_prob - second_prob,
                f"{side}_is_tool_call_top1": bool(top1_id == tool_token_id),
            }
        )
    del padded, input_ids_tensor, attention_mask_tensor, outputs, logits
    clear_cuda()
    return results


def top_token_rows(rows: list[dict[str, Any]], side: str) -> list[dict[str, Any]]:
    counter = Counter(int(row[f"{side}_top1_token_id"]) for row in rows)
    total = max(len(rows), 1)
    sample_row = {int(row[f"{side}_top1_token_id"]): row[f"{side}_top1_token_text"] for row in rows}
    output: list[dict[str, Any]] = []
    for token_id, count in counter.most_common(10):
        output.append(
            {
                "token_id": int(token_id),
                "count": int(count),
                "rate": float(count / total),
                "token_text": sample_row[token_id],
            }
        )
    return output


def mean(values: list[float]) -> float:
    if not values:
        return math.nan
    return float(sum(values) / len(values))


def build_summary(rows: list[dict[str, Any]], tool_token_text: str, tool_token_id: int) -> dict[str, Any]:
    clean_rate = mean([1.0 if row["clean_is_tool_call_top1"] else 0.0 for row in rows])
    corrupt_rate = mean([1.0 if row["corrupt_is_tool_call_top1"] else 0.0 for row in rows])
    borderline_count = sum(
        1
        for row in rows
        if row["clean_top1_margin"] < 0.05 or row["corrupt_top1_margin"] < 0.05
    )
    summary = {
        "n_pairs": len(rows),
        "tool_call_token_text": tool_token_text,
        "tool_call_token_id": int(tool_token_id),
        "clean_tool_call_rate": clean_rate,
        "corrupt_tool_call_rate": corrupt_rate,
        "gap_top1_pp": (clean_rate - corrupt_rate) * 100.0,
        "borderline_pair_count": int(borderline_count),
        "borderline_definition": "Pairs where clean or corrupt prompt has top1 probability margin under 0.05.",
        "clean": {
            "mean_tool_call_prob": mean([row["clean_tool_token_prob"] for row in rows]),
            "mean_tool_call_logit": mean([row["clean_tool_token_logit"] for row in rows]),
            "top1_mode_rows": top_token_rows(rows, "clean"),
        },
        "corrupt": {
            "mean_tool_call_prob": mean([row["corrupt_tool_token_prob"] for row in rows]),
            "mean_tool_call_logit": mean([row["corrupt_tool_token_logit"] for row in rows]),
            "top1_mode_rows": top_token_rows(rows, "corrupt"),
        },
        "per_split": {},
    }
    for split in sorted({str(row["split"]) for row in rows}):
        split_rows = [row for row in rows if str(row["split"]) == split]
        summary["per_split"][split] = {
            "n_pairs": len(split_rows),
            "clean_tool_call_rate": mean([1.0 if row["clean_is_tool_call_top1"] else 0.0 for row in split_rows]),
            "corrupt_tool_call_rate": mean([1.0 if row["corrupt_is_tool_call_top1"] else 0.0 for row in split_rows]),
        }
    return summary


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    ensure_dir(args.output_root)
    tokenizer = AutoTokenizer.from_pretrained(str(args.model_path), trust_remote_code=True)
    tokenizer.padding_side = "left"
    model = load_model(args.model_path, args.dtype, args.device_map, args.output_root)

    manifest_rows = read_jsonl(args.converted_root / "manifest.jsonl")
    canonical_rows = read_jsonl(args.converted_root / "canonical_pairs.jsonl")
    conversion_summary = read_json(args.converted_root / "conversion_summary.json")
    manifest_by_id = {str(row["sample_id"]): row for row in manifest_rows}

    tool_token_text = str(conversion_summary["tool_call_token_text"])
    tool_token_id = int(conversion_summary["tool_call_token_id"])

    prompt_rows = build_prompt_tensors(
        tokenizer,
        canonical_rows,
        manifest_by_id,
        max_pairs=args.max_pairs,
        start_index=args.start_index,
    )

    result_rows: list[dict[str, Any]] = []
    for chunk in iter_chunks(prompt_rows, args.batch_size):
        clean_stats = evaluate_batch(model, tokenizer, chunk, tool_token_id, "clean")
        corrupt_stats = evaluate_batch(model, tokenizer, chunk, tool_token_id, "corrupt")
        for row, clean_row, corrupt_row in zip(chunk, clean_stats, corrupt_stats):
            result_rows.append(
                {
                    "sample_id": row["sample_id"],
                    "split": row["split"],
                    "language": row["language"],
                    "clean_candidate": row["clean_candidate"],
                    "corrupt_candidate": row["corrupt_candidate"],
                    **clean_row,
                    **corrupt_row,
                }
            )

    result_rows.sort(key=lambda row: str(row["sample_id"]))
    write_csv(args.output_root / "per_pair_results.csv", result_rows)
    (args.output_root / "per_pair_results.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in result_rows) + ("\n" if result_rows else ""),
        encoding="utf-8",
    )

    summary = build_summary(result_rows, tool_token_text, tool_token_id)
    summary.update(
        {
            "model_path": str(args.model_path.resolve()),
            "converted_root": str(args.converted_root.resolve()),
            "output_root": str(args.output_root.resolve()),
            "batch_size": int(args.batch_size),
            "dtype": args.dtype,
            "device_map": args.device_map,
            "start_index": int(args.start_index),
            "max_pairs": int(args.max_pairs),
        }
    )
    (args.output_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
