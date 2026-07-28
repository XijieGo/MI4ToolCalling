#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
from collections import Counter
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import torch
from tqdm.auto import tqdm
from transformers import AutoTokenizer, Mistral3ForConditionalGeneration


SHORT_TOOLCALL_SYSTEM_PROMPT = """# Tools

You may call one or more functions to assist with the user query.

Use the available functions when the user asks you to create, write, update, complete, or save file content.

If tool use is appropriate, make the tool call. Otherwise answer normally."""
PROJECT_ROOT = Path(__file__).resolve().parents[2]
VIBE_SYSTEM_PROMPT_PATH = PROJECT_ROOT / "configs" / "prompts" / "mistral_vibe_system_prompt.txt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Mistral-Small-3.2-24B tool-call behavior scan from scratch.")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path(
            os.environ.get(
                "MISTRAL_3P2_24B_PATH",
                str(PROJECT_ROOT / "external" / "models" / "Mistral-Small-3.2-24B-Instruct-2506"),
            )
        ),
    )
    parser.add_argument(
        "--converted-root",
        type=Path,
        default=PROJECT_ROOT / "results" / "Mistral-Small-3.2-24B-Instruct-2506" / "converted_dataset",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "results" / "Mistral-Small-3.2-24B-Instruct-2506" / "behavior_scan",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--device-map", type=str, default="auto")
    parser.add_argument("--max-pairs", type=int, default=0)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument(
        "--system-prompt-mode",
        type=str,
        choices=("official", "toolcall_short", "vibe"),
        default="vibe",
    )
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


def format_system_prompt(
    template_path: Path,
    mode: str,
    *,
    converted_verification: dict[str, Any] | None = None,
) -> str:
    if mode == "toolcall_short":
        return SHORT_TOOLCALL_SYSTEM_PROMPT
    if mode == "vibe":
        if converted_verification is not None:
            preview = str(converted_verification.get("system_prompt_preview", "")).strip()
            if preview:
                return preview
        if VIBE_SYSTEM_PROMPT_PATH.exists():
            prompt = VIBE_SYSTEM_PROMPT_PATH.read_text(encoding="utf-8").strip()
            if prompt:
                return prompt
        raise FileNotFoundError(
            "System prompt preview not found in the converted dataset or "
            f"versioned prompt file {VIBE_SYSTEM_PROMPT_PATH}"
        )
    today = date.today()
    yesterday = today - timedelta(days=1)
    template = template_path.read_text(encoding="utf-8")
    return template.format(
        name="Mistral-Small-3.2-24B-Instruct-2506",
        today=today.isoformat(),
        yesterday=yesterday.isoformat(),
    )


def build_max_memory(device_map: str) -> dict[Any, str] | None:
    if device_map != "auto" or not torch.cuda.is_available():
        return None
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    free_gib = max(int(free_bytes // (1024**3)) - 2, 8)
    total_gib = int(total_bytes // (1024**3))
    free_gib = min(free_gib, total_gib - 1)
    return {0: f"{free_gib}GiB", "cpu": "400GiB"}


def load_model(model_path: Path, dtype_name: str, device_map: str, output_root: Path):
    dtype = getattr(torch, dtype_name)
    kwargs: dict[str, Any] = {
        "dtype": dtype,
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
    model = Mistral3ForConditionalGeneration.from_pretrained(str(model_path), **kwargs)
    model.eval()
    return model


def build_prompt_tensors(
    tokenizer,
    canonical_rows: list[dict[str, Any]],
    manifest_by_id: dict[str, dict[str, Any]],
    *,
    system_prompt: str,
    max_pairs: int,
    start_index: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    stop_index = start_index + max_pairs if max_pairs > 0 else len(canonical_rows)
    for canonical_row in canonical_rows[start_index:stop_index]:
        sample_id = str(canonical_row["sample_id"])
        manifest_row = manifest_by_id[sample_id]
        tools = canonical_row["tools_schema"]
        clean_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": canonical_row["user_content_clean"]},
        ]
        corrupt_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": canonical_row["user_content_corrupt"]},
        ]
        clean_enc = tokenizer.apply_chat_template(
            clean_messages,
            tools=tools,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        corrupt_enc = tokenizer.apply_chat_template(
            corrupt_messages,
            tools=tools,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
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
        outputs = model(input_ids=input_ids_tensor, attention_mask=attention_mask_tensor)
    logits = outputs.logits.float().cpu()
    attention_mask_cpu = padded["attention_mask"].cpu()
    lengths = attention_mask_cpu.sum(dim=1).long()
    results: list[dict[str, Any]] = []
    for idx, row in enumerate(prompts):
        last_pos = int(lengths[idx].item()) - 1
        last_logits = logits[idx, last_pos, :]
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
        "tool_calls_token_text": tool_token_text,
        "tool_calls_token_id": int(tool_token_id),
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
    ensure_dir(path.parent)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    clean_rate = summary["clean_tool_call_rate"] * 100.0
    corrupt_rate = summary["corrupt_tool_call_rate"] * 100.0
    gap = summary["gap_top1_pp"]
    clean_mode = summary["clean"]["top1_mode_rows"][0]["token_text"] if summary["clean"]["top1_mode_rows"] else "N/A"
    corrupt_mode = summary["corrupt"]["top1_mode_rows"][0]["token_text"] if summary["corrupt"]["top1_mode_rows"] else "N/A"
    if gap >= 30.0:
        verdict = "clean/corrupt still show substantial separation on Mistral-Small-3.2-24B."
    elif gap >= 10.0:
        verdict = "clean/corrupt remain partially separated, but the gap is much weaker than a clean transfer."
    else:
        verdict = "clean/corrupt are not meaningfully separated on Mistral-Small-3.2-24B."
    if clean_rate < 50.0 and corrupt_rate < 50.0:
        collapse = "Both sides tend to collapse away from tool calls."
    elif clean_rate >= 50.0 and corrupt_rate >= 50.0:
        collapse = "Both sides tend to collapse into tool calls."
    elif clean_rate < 50.0:
        collapse = "The clean side is the one that collapses first."
    else:
        collapse = "The corrupt side is the one that collapses first."
    if gap >= 30.0:
        next_step = "This dataset is likely worth deeper Mistral-family follow-up."
    elif gap >= 10.0:
        next_step = "This dataset may support limited follow-up, but it is not a clean drop-in for mechanism work."
    else:
        next_step = "This dataset is not a strong candidate for deeper Mistral-specific mechanism experiments without re-filtering."
    text = (
        "# Mistral-Small-3.2-24B Tool-Call Generalization Scan\n\n"
        f"- Pairs evaluated: `{summary['n_pairs']}`\n"
        f"- `[TOOL_CALLS]` token id: `{summary['tool_calls_token_id']}`\n"
        f"- Clean tool-call rate: `{clean_rate:.2f}%`\n"
        f"- Corrupt tool-call rate: `{corrupt_rate:.2f}%`\n"
        f"- Clean minus corrupt gap: `{gap:.2f} pp`\n"
        f"- Clean top-1 mode: `{clean_mode}`\n"
        f"- Corrupt top-1 mode: `{corrupt_mode}`\n"
        f"- Borderline pairs: `{summary['borderline_pair_count']}`\n\n"
        f"{verdict}\n\n"
        f"{collapse}\n\n"
        f"{next_step}\n"
    )
    path.write_text(text, encoding="utf-8")


def checkpoint_outputs(output_root: Path, pair_rows: list[dict[str, Any]], tool_token_text: str, tool_token_id: int) -> None:
    if not pair_rows:
        return
    write_csv(output_root / "pair_decisions.csv", pair_rows)
    summary = build_summary(pair_rows, tool_token_text, tool_token_id)
    (output_root / "aggregate_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_markdown(output_root / "aggregate_summary.md", summary)


def main() -> None:
    args = parse_args()
    ensure_dir(args.output_root)
    tokenizer = AutoTokenizer.from_pretrained(str(args.model_path), trust_remote_code=True)
    manifest_rows = read_jsonl(args.converted_root / "manifest.jsonl")
    canonical_rows = read_jsonl(args.converted_root / "canonical_pairs.jsonl")
    verification = read_json(args.converted_root / "tokenizer_verification.json")
    tool_token_text = str(verification["tool_token_text"])
    tool_token_id = int(verification["tool_token_id_via_convert"])
    system_prompt = format_system_prompt(
        args.model_path / "SYSTEM_PROMPT.txt",
        args.system_prompt_mode,
        converted_verification=verification,
    )
    manifest_by_id = {str(row["sample_id"]): row for row in manifest_rows}
    model = load_model(args.model_path, args.dtype, args.device_map, args.output_root)
    prepared_rows = build_prompt_tensors(
        tokenizer,
        canonical_rows,
        manifest_by_id,
        system_prompt=system_prompt,
        max_pairs=args.max_pairs,
        start_index=args.start_index,
    )

    pair_rows: list[dict[str, Any]] = []
    progress = tqdm(list(iter_chunks(prepared_rows, args.batch_size)), desc="Mistral behavior scan", dynamic_ncols=True)
    for batch_idx, chunk in enumerate(progress, start=1):
        clean_stats = evaluate_batch(model, tokenizer, chunk, tool_token_id, "clean")
        corrupt_stats = evaluate_batch(model, tokenizer, chunk, tool_token_id, "corrupt")
        for base_row, clean_row, corrupt_row in zip(chunk, clean_stats, corrupt_stats):
            pair_rows.append(
                {
                    "sample_id": base_row["sample_id"],
                    "split": base_row["split"],
                    "language": base_row["language"],
                    "clean_candidate": base_row["clean_candidate"],
                    "corrupt_candidate": base_row["corrupt_candidate"],
                    "clean_prompt_path": base_row["clean_prompt_path"],
                    "corrupt_prompt_path": base_row["corrupt_prompt_path"],
                    **clean_row,
                    **corrupt_row,
                }
            )
        if batch_idx % 20 == 0 or batch_idx == len(progress):
            checkpoint_outputs(args.output_root, pair_rows, tool_token_text, tool_token_id)

    checkpoint_outputs(args.output_root, pair_rows, tool_token_text, tool_token_id)


if __name__ == "__main__":
    main()
