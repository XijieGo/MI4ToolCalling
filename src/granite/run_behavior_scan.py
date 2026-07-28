#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gc
import json
from collections import Counter, defaultdict
from pathlib import Path

import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from granite_toolcall_common import (
    DEFAULT_BEHAVIOR_ROOT,
    DEFAULT_CONVERTED_ROOT,
    DEFAULT_MODEL_PATH,
    TOOL_CALL_TOKEN,
    ensure_dir,
    read_jsonl,
    read_text,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scan Granite next-token tool-call behavior on converted prompts.")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--converted-root", type=Path, default=DEFAULT_CONVERTED_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_BEHAVIOR_ROOT)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--device-map", type=str, default="auto")
    parser.add_argument("--max-samples", type=int, default=0)
    return parser.parse_args()


def clear_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    ensure_dir(path.parent)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def tokenize_text(tokenizer, text: str) -> list[int]:
    return list(tokenizer(text, add_special_tokens=False)["input_ids"])


def pad_sequences(tokenizer, sequences: list[list[int]]) -> dict[str, torch.Tensor]:
    return tokenizer.pad({"input_ids": sequences}, return_tensors="pt", padding=True)


def decode_token(tokenizer, token_id: int) -> str:
    text = tokenizer.decode([int(token_id)], clean_up_tokenization_spaces=False)
    return text.encode("unicode_escape").decode("ascii")


def is_oom_error(exc: BaseException) -> bool:
    message = str(exc).lower()
    return isinstance(exc, torch.OutOfMemoryError) or "out of memory" in message


def forward_batch(
    *,
    model,
    tokenizer,
    sequences: list[list[int]],
    tool_token_id: int,
    device: torch.device,
) -> list[dict[str, object]]:
    encoded = pad_sequences(tokenizer, sequences)
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    with torch.inference_mode():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    # Find the last non-padding token index robustly for both left- and right-padding.
    reversed_offsets = torch.argmax(attention_mask.flip(dims=[1]), dim=1)
    last_indices = attention_mask.shape[1] - 1 - reversed_offsets
    batch_indices = torch.arange(input_ids.shape[0], device=outputs.logits.device)
    last_logits = outputs.logits[batch_indices, last_indices].float()
    last_probs = torch.softmax(last_logits, dim=-1)
    top1_probs, top1_ids = torch.max(last_probs, dim=-1)
    tool_logits = last_logits[:, tool_token_id]
    tool_probs = last_probs[:, tool_token_id]

    batch_rows: list[dict[str, object]] = []
    for index in range(len(sequences)):
        top1_token_id = int(top1_ids[index].item())
        batch_rows.append(
            {
                "tool_token_logit": float(tool_logits[index].item()),
                "tool_token_prob": float(tool_probs[index].item()),
                "top1_token_id": top1_token_id,
                "top1_token_text": decode_token(tokenizer, top1_token_id),
                "top1_token_prob": float(top1_probs[index].item()),
                "is_tool_call_top1": bool(top1_token_id == tool_token_id),
            }
        )
    del encoded, input_ids, attention_mask, outputs, last_logits, last_probs, top1_probs, top1_ids, tool_logits, tool_probs
    clear_cuda()
    return batch_rows


def evaluate_batch_rows_adaptive(
    *,
    model,
    tokenizer,
    batch_rows: list[dict[str, object]],
    tool_token_id: int,
    device: torch.device,
    progress_bar,
) -> list[dict[str, object]]:
    sequences = [row["clean_input_ids"] for row in batch_rows] + [
        row["corrupt_input_ids"] for row in batch_rows
    ]
    try:
        batch_stats = forward_batch(
            model=model,
            tokenizer=tokenizer,
            sequences=sequences,
            tool_token_id=tool_token_id,
            device=device,
        )
        if len(batch_rows) > 1:
            progress_bar.set_postfix_str(f"effective_batch={len(batch_rows)}")
        return batch_stats
    except Exception as exc:
        if not is_oom_error(exc):
            raise
        clear_cuda()
        if len(batch_rows) <= 1:
            raise
        split_point = max(len(batch_rows) // 2, 1)
        progress_bar.write(
            f"OOM on batch of {len(batch_rows)} pairs; retrying with {split_point} + {len(batch_rows) - split_point}."
        )
        left_stats = evaluate_batch_rows_adaptive(
            model=model,
            tokenizer=tokenizer,
            batch_rows=batch_rows[:split_point],
            tool_token_id=tool_token_id,
            device=device,
            progress_bar=progress_bar,
        )
        right_stats = evaluate_batch_rows_adaptive(
            model=model,
            tokenizer=tokenizer,
            batch_rows=batch_rows[split_point:],
            tool_token_id=tool_token_id,
            device=device,
            progress_bar=progress_bar,
        )
        return left_stats + right_stats


def summarize_side(rows: list[dict[str, object]], prefix: str) -> dict[str, object]:
    top1_counter = Counter(str(row[f"{prefix}_top1_token_text"]) for row in rows)
    n = len(rows)
    tool_count = sum(int(bool(row[f"{prefix}_is_tool_call_top1"])) for row in rows)
    borderline_count = sum(
        1 for row in rows if 0.25 <= float(row[f"{prefix}_tool_token_prob"]) <= 0.75
    )
    mode_token, mode_count = ("", 0)
    if top1_counter:
        mode_token, mode_count = top1_counter.most_common(1)[0]
    return {
        "n": n,
        "tool_call_top1_count": tool_count,
        "tool_call_top1_rate": float(tool_count / n) if n else 0.0,
        "mean_tool_token_prob": float(
            sum(float(row[f"{prefix}_tool_token_prob"]) for row in rows) / n
        )
        if n
        else 0.0,
        "mean_tool_token_logit": float(
            sum(float(row[f"{prefix}_tool_token_logit"]) for row in rows) / n
        )
        if n
        else 0.0,
        "borderline_25_75_count": borderline_count,
        "borderline_25_75_rate": float(borderline_count / n) if n else 0.0,
        "top1_token_mode": mode_token,
        "top1_token_mode_count": mode_count,
        "top1_token_mode_rate": float(mode_count / n) if n else 0.0,
    }


def summarize_pairs(rows: list[dict[str, object]]) -> dict[str, object]:
    clean_summary = summarize_side(rows, "clean")
    corrupt_summary = summarize_side(rows, "corrupt")
    n = len(rows)
    both_tool_call_count = sum(
        1
        for row in rows
        if bool(row["clean_is_tool_call_top1"]) and bool(row["corrupt_is_tool_call_top1"])
    )
    neither_tool_call_count = sum(
        1
        for row in rows
        if (not bool(row["clean_is_tool_call_top1"])) and (not bool(row["corrupt_is_tool_call_top1"]))
    )
    either_borderline_count = sum(
        1
        for row in rows
        if 0.25 <= float(row["clean_tool_token_prob"]) <= 0.75
        or 0.25 <= float(row["corrupt_tool_token_prob"]) <= 0.75
    )
    both_borderline_count = sum(
        1
        for row in rows
        if 0.25 <= float(row["clean_tool_token_prob"]) <= 0.75
        and 0.25 <= float(row["corrupt_tool_token_prob"]) <= 0.75
    )
    return {
        "n_pairs": n,
        "clean": clean_summary,
        "corrupt": corrupt_summary,
        "clean_minus_corrupt_gap": float(
            clean_summary["tool_call_top1_rate"] - corrupt_summary["tool_call_top1_rate"]
        ),
        "pairs_both_sides_tool_call_top1_count": both_tool_call_count,
        "pairs_both_sides_tool_call_top1_rate": float(both_tool_call_count / n) if n else 0.0,
        "pairs_neither_side_tool_call_top1_count": neither_tool_call_count,
        "pairs_neither_side_tool_call_top1_rate": float(neither_tool_call_count / n) if n else 0.0,
        "pairs_either_side_borderline_25_75_count": either_borderline_count,
        "pairs_either_side_borderline_25_75_rate": float(either_borderline_count / n) if n else 0.0,
        "pairs_both_sides_borderline_25_75_count": both_borderline_count,
        "pairs_both_sides_borderline_25_75_rate": float(both_borderline_count / n) if n else 0.0,
    }


def build_group_rows(pair_rows: list[dict[str, object]], key: str) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in pair_rows:
        grouped[str(row.get(key, ""))].append(row)
    output_rows: list[dict[str, object]] = []
    for value, rows in sorted(grouped.items(), key=lambda item: (-len(item[1]), item[0])):
        summary = summarize_pairs(rows)
        output_rows.append(
            {
                key: value,
                "n_pairs": int(summary["n_pairs"]),
                "clean_tool_call_top1_rate": float(summary["clean"]["tool_call_top1_rate"]),
                "corrupt_tool_call_top1_rate": float(summary["corrupt"]["tool_call_top1_rate"]),
                "clean_minus_corrupt_gap": float(summary["clean_minus_corrupt_gap"]),
                "clean_mode_token": str(summary["clean"]["top1_token_mode"]),
                "corrupt_mode_token": str(summary["corrupt"]["top1_token_mode"]),
                "either_borderline_25_75_count": int(summary["pairs_either_side_borderline_25_75_count"]),
                "both_sides_tool_call_top1_count": int(summary["pairs_both_sides_tool_call_top1_count"]),
            }
        )
    return output_rows


def recommendation_from_summary(summary: dict[str, object]) -> str:
    clean_rate = float(summary["clean"]["tool_call_top1_rate"])
    corrupt_rate = float(summary["corrupt"]["tool_call_top1_rate"])
    gap = float(summary["clean_minus_corrupt_gap"])
    both_rate = float(summary["pairs_both_sides_tool_call_top1_rate"])
    if clean_rate >= 0.8 and corrupt_rate <= 0.2 and gap >= 0.6 and both_rate <= 0.1:
        return "worth_followup_mechanistic_analysis"
    if gap >= 0.3 and both_rate <= 0.3:
        return "borderline_but_maybe_worth_screening_or_resampling"
    return "not_recommended_for_immediate_mechanistic_followup"


def build_markdown_summary(
    *,
    model_path: Path,
    tool_token_id: int,
    summary: dict[str, object],
) -> str:
    clean = summary["clean"]
    corrupt = summary["corrupt"]
    gap = float(summary["clean_minus_corrupt_gap"])
    both_tool = int(summary["pairs_both_sides_tool_call_top1_count"])
    either_borderline = int(summary["pairs_either_side_borderline_25_75_count"])
    n_pairs = int(summary["n_pairs"])
    separated = bool(gap >= 0.5 and float(clean["tool_call_top1_rate"]) > float(corrupt["tool_call_top1_rate"]))
    collapse = bool(both_tool > 0)
    if separated:
        separation_text = "Granite 上 clean/corrupt 仍然显著分离。"
    else:
        separation_text = "Granite 上 clean/corrupt 没有形成足够强的首 token 分离。"
    if collapse:
        collapse_text = (
            f"存在两侧都塌到工具调用的样本，共 {both_tool}/{n_pairs} 对，"
            f"占比 {float(summary['pairs_both_sides_tool_call_top1_rate']):.4f}。"
        )
    else:
        collapse_text = "没有出现两侧都塌到工具调用的样本。"
    recommendation = recommendation_from_summary(summary)
    if recommendation == "worth_followup_mechanistic_analysis":
        recommendation_text = "当前行为筛查结果支持继续做后续机制分析。"
    elif recommendation == "borderline_but_maybe_worth_screening_or_resampling":
        recommendation_text = "当前结果偏边缘，建议先做进一步筛样或分组诊断，再决定是否进入机制分析。"
    else:
        recommendation_text = "当前结果不支持直接进入后续机制分析，建议先重筛数据或更换模型家族。"

    lines = [
        "# granite-3.3-8b-instruct 工具调用率泛化筛查",
        "",
        f"- model_path: `{model_path}`",
        f"- tool_call_token: `{TOOL_CALL_TOKEN}` (token_id={tool_token_id})",
        f"- n_pairs: {n_pairs}",
        f"- clean tool-call rate: {float(clean['tool_call_top1_rate']):.4f} ({int(clean['tool_call_top1_count'])}/{n_pairs})",
        f"- corrupt tool-call rate: {float(corrupt['tool_call_top1_rate']):.4f} ({int(corrupt['tool_call_top1_count'])}/{n_pairs})",
        f"- clean minus corrupt gap: {gap:.4f}",
        f"- clean top-1 mode: `{clean['top1_token_mode']}` ({int(clean['top1_token_mode_count'])}/{n_pairs})",
        f"- corrupt top-1 mode: `{corrupt['top1_token_mode']}` ({int(corrupt['top1_token_mode_count'])}/{n_pairs})",
        f"- borderline pairs (0.25 <= p(tool) <= 0.75 on either side): {either_borderline}/{n_pairs}",
        "",
        "## Interpretation",
        "",
        separation_text,
        collapse_text,
        recommendation_text,
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    ensure_dir(args.output_root)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    tokenizer = AutoTokenizer.from_pretrained(str(args.model_path), trust_remote_code=True)
    tool_token_ids = tokenizer.encode(TOOL_CALL_TOKEN, add_special_tokens=False)
    if len(tool_token_ids) != 1:
        raise RuntimeError(f"{TOOL_CALL_TOKEN!r} must be a single token, got {tool_token_ids}")
    tool_token_id = int(tool_token_ids[0])

    manifest_rows = read_jsonl(args.converted_root / "manifest.jsonl")
    if args.max_samples > 0:
        manifest_rows = manifest_rows[: args.max_samples]
    if not manifest_rows:
        raise RuntimeError(f"No converted prompts found under {args.converted_root}")

    prepared_rows: list[dict[str, object]] = []
    prep_progress = tqdm(manifest_rows, desc="Tokenizing prompts", dynamic_ncols=True)
    for row in prep_progress:
        clean_prompt_path = Path(str(row["clean_prompt_path"]))
        corrupt_prompt_path = Path(str(row["corrupt_prompt_path"]))
        clean_prompt_text = read_text(clean_prompt_path)
        corrupt_prompt_text = read_text(corrupt_prompt_path)
        clean_input_ids = tokenize_text(tokenizer, clean_prompt_text)
        corrupt_input_ids = tokenize_text(tokenizer, corrupt_prompt_text)
        prepared_rows.append(
            {
                **row,
                "clean_prompt_path": str(clean_prompt_path.resolve()),
                "corrupt_prompt_path": str(corrupt_prompt_path.resolve()),
                "clean_input_ids": clean_input_ids,
                "corrupt_input_ids": corrupt_input_ids,
                "clean_prompt_token_length": len(clean_input_ids),
                "corrupt_prompt_token_length": len(corrupt_input_ids),
            }
        )

    dtype = getattr(torch, args.dtype)
    model = AutoModelForCausalLM.from_pretrained(
        str(args.model_path),
        torch_dtype=dtype,
        device_map=args.device_map,
        trust_remote_code=True,
    )
    model.eval()
    device = next(model.parameters()).device

    pair_rows: list[dict[str, object]] = []
    scan_progress = tqdm(
        range(0, len(prepared_rows), args.batch_size),
        desc="Granite behavior scan",
        dynamic_ncols=True,
    )
    for start in scan_progress:
        batch_rows = prepared_rows[start : start + args.batch_size]
        batch_stats = evaluate_batch_rows_adaptive(
            model=model,
            tokenizer=tokenizer,
            batch_rows=batch_rows,
            tool_token_id=tool_token_id,
            device=device,
            progress_bar=scan_progress,
        )
        split_point = len(batch_rows)
        clean_stats = batch_stats[:split_point]
        corrupt_stats = batch_stats[split_point:]
        for row, clean_stat, corrupt_stat in zip(batch_rows, clean_stats, corrupt_stats, strict=True):
            pair_rows.append(
                {
                    "sample_id": row["sample_id"],
                    "global_row_id": row.get("global_row_id"),
                    "split": row["split"],
                    "dataset_name": row.get("dataset_name"),
                    "language": row.get("language"),
                    "template_kind": row.get("template_kind"),
                    "clean_candidate": row.get("clean_candidate"),
                    "corrupt_candidate": row.get("corrupt_candidate"),
                    "clean_prompt_path": row["clean_prompt_path"],
                    "corrupt_prompt_path": row["corrupt_prompt_path"],
                    "clean_prompt_token_length": row["clean_prompt_token_length"],
                    "corrupt_prompt_token_length": row["corrupt_prompt_token_length"],
                    "clean_tool_token_logit": clean_stat["tool_token_logit"],
                    "corrupt_tool_token_logit": corrupt_stat["tool_token_logit"],
                    "clean_tool_token_prob": clean_stat["tool_token_prob"],
                    "corrupt_tool_token_prob": corrupt_stat["tool_token_prob"],
                    "clean_top1_token_id": clean_stat["top1_token_id"],
                    "corrupt_top1_token_id": corrupt_stat["top1_token_id"],
                    "clean_top1_token_text": clean_stat["top1_token_text"],
                    "corrupt_top1_token_text": corrupt_stat["top1_token_text"],
                    "clean_top1_token_prob": clean_stat["top1_token_prob"],
                    "corrupt_top1_token_prob": corrupt_stat["top1_token_prob"],
                    "clean_is_tool_call_top1": clean_stat["is_tool_call_top1"],
                    "corrupt_is_tool_call_top1": corrupt_stat["is_tool_call_top1"],
                }
            )

    summary = summarize_pairs(pair_rows)
    summary["model_path"] = str(args.model_path)
    summary["converted_root"] = str(args.converted_root)
    summary["output_root"] = str(args.output_root)
    summary["tool_call_token"] = TOOL_CALL_TOKEN
    summary["tool_call_token_id"] = tool_token_id
    summary["borderline_definition"] = "0.25 <= p(<|tool_call|>) <= 0.75"
    summary["recommendation"] = recommendation_from_summary(summary)

    by_split_rows = build_group_rows(pair_rows, "split")
    by_language_rows = build_group_rows(pair_rows, "language")
    by_clean_candidate_rows = build_group_rows(pair_rows, "clean_candidate")
    by_corrupt_candidate_rows = build_group_rows(pair_rows, "corrupt_candidate")
    summary["by_split"] = {
        row["split"]: {
            "n_pairs": row["n_pairs"],
            "clean_tool_call_top1_rate": row["clean_tool_call_top1_rate"],
            "corrupt_tool_call_top1_rate": row["corrupt_tool_call_top1_rate"],
            "clean_minus_corrupt_gap": row["clean_minus_corrupt_gap"],
        }
        for row in by_split_rows
    }
    summary["by_language"] = {
        row["language"]: {
            "n_pairs": row["n_pairs"],
            "clean_tool_call_top1_rate": row["clean_tool_call_top1_rate"],
            "corrupt_tool_call_top1_rate": row["corrupt_tool_call_top1_rate"],
            "clean_minus_corrupt_gap": row["clean_minus_corrupt_gap"],
        }
        for row in by_language_rows
    }

    write_csv(args.output_root / "pair_decisions.csv", pair_rows)
    write_csv(args.output_root / "summary_by_split.csv", by_split_rows)
    write_csv(args.output_root / "summary_by_language.csv", by_language_rows)
    write_csv(args.output_root / "summary_by_clean_candidate.csv", by_clean_candidate_rows)
    write_csv(args.output_root / "summary_by_corrupt_candidate.csv", by_corrupt_candidate_rows)
    write_json(args.output_root / "aggregate_summary.json", summary)
    (args.output_root / "aggregate_summary.md").write_text(
        build_markdown_summary(model_path=args.model_path, tool_token_id=tool_token_id, summary=summary),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "n_pairs": summary["n_pairs"],
                "clean_tool_call_top1_rate": summary["clean"]["tool_call_top1_rate"],
                "corrupt_tool_call_top1_rate": summary["corrupt"]["tool_call_top1_rate"],
                "clean_minus_corrupt_gap": summary["clean_minus_corrupt_gap"],
                "recommendation": summary["recommendation"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
