#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import re
from collections import Counter, defaultdict
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer, Mistral3ForConditionalGeneration
from transformers.utils import logging as transformers_logging

try:
    transformers_logging.disable_progress_bar()
except Exception:
    pass


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = Path(
    os.environ.get(
        "DEVSTRAL_2_24B_PATH",
        str(PROJECT_ROOT / "external" / "models" / "Devstral-Small-2-24B-Instruct-2512"),
    )
)
DEFAULT_CANONICAL_PATH = PROJECT_ROOT / "results" / "Devstral-Small-2-24B-Instruct-2512" / "converted_dataset" / "canonical_pairs.jsonl"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "results" / "Devstral-Small-2-24B-Instruct-2512" / "verb_subset_scan_50"
DEFAULT_EXECUTION_VERBS = (
    "add",
    "apply",
    "assemble",
    "author",
    "build",
    "code",
    "complete",
    "compose",
    "construct",
    "create",
    "develop",
    "draft",
    "fill",
    "generate",
    "implement",
    "make",
    "produce",
    "program",
    "update",
    "write",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scan execution-like verbs on a 50-sample Devstral subset.")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--canonical-path", type=Path, default=DEFAULT_CANONICAL_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--system-prompt-file", type=str, default="CHAT_SYSTEM_PROMPT.txt")
    parser.add_argument("--subset-size", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--device-map", type=str, default="auto")
    parser.add_argument("--gpu-max-memory", type=str, default="")
    parser.add_argument("--cpu-max-memory", type=str, default="")
    parser.add_argument("--verbs", nargs="*", default=list(DEFAULT_EXECUTION_VERBS))
    parser.add_argument("--today", type=str, default="")
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def resolve_system_prompt(path: Path, today_override: str = "") -> str:
    today_value = date.fromisoformat(today_override) if today_override else date.today()
    yesterday_value = today_value - timedelta(days=1)
    template = path.read_text(encoding="utf-8")
    return template.replace("{today}", today_value.isoformat()).replace("{yesterday}", yesterday_value.isoformat())


def pick_subset(rows: list[dict[str, Any]], subset_size: int) -> list[dict[str, Any]]:
    test_rows = [row for row in rows if row["split"] == "test"]
    by_language: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in test_rows:
        by_language[row["language"]].append(row)
    for group in by_language.values():
        group.sort(key=lambda row: row["sample_id"])

    languages = sorted(by_language)
    target = subset_size // max(len(languages), 1)
    remainder = subset_size % max(len(languages), 1)
    selected: list[dict[str, Any]] = []
    for idx, language in enumerate(languages):
        take = target + (1 if idx < remainder else 0)
        selected.extend(by_language[language][:take])
    selected.sort(key=lambda row: (row["language"], row["sample_id"]))
    return selected[:subset_size]


def replace_first_verb(user_text: str, new_verb: str) -> str:
    lines = user_text.splitlines()
    if not lines:
        raise ValueError("Empty user content.")
    first_line = lines[0]
    match = re.match(r"^(\s*)(\S+)(.*)$", first_line)
    if match is None:
        raise ValueError(f"Could not parse first line: {first_line!r}")
    prefix, _old, suffix = match.groups()
    replacement = new_verb[:1].upper() + new_verb[1:]
    lines[0] = prefix + replacement + suffix
    return "\n".join(lines)


def build_model(model_path: Path, dtype_name: str, device_map: str, gpu_max_memory: str, cpu_max_memory: str):
    dtype = getattr(torch, dtype_name)
    extra_kwargs: dict[str, Any] = {}
    if gpu_max_memory or cpu_max_memory:
        max_memory: dict[Any, str] = {}
        if gpu_max_memory:
            max_memory[0] = gpu_max_memory
        if cpu_max_memory:
            max_memory["cpu"] = cpu_max_memory
        extra_kwargs["max_memory"] = max_memory
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
    return tokenizer.pad(
        {
            "input_ids": [item["input_ids"] for item in items],
            "attention_mask": [item["attention_mask"] for item in items],
        },
        padding=True,
        return_tensors="pt",
    )


def evaluate_prompts(model, tokenizer, prompts: list[dict[str, Any]], tool_token_id: int, batch_size: int) -> list[dict[str, Any]]:
    device = next(model.parameters()).device
    outputs: list[dict[str, Any]] = []
    for start in range(0, len(prompts), batch_size):
        batch_items = prompts[start : start + batch_size]
        batch = collate_batch(tokenizer, batch_items)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        with torch.no_grad():
            result = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                logits_to_keep=1,
            )
        logits = result.logits[:, -1, :].float()
        probs = torch.softmax(logits, dim=-1)
        top1 = logits.argmax(dim=-1)
        for idx, item in enumerate(batch_items):
            top1_id = int(top1[idx].item())
            outputs.append(
                {
                    **item,
                    "tool_token_id": tool_token_id,
                    "tool_token_prob": float(probs[idx, tool_token_id].item()),
                    "tool_token_logit": float(logits[idx, tool_token_id].item()),
                    "top1_token_id": top1_id,
                    "top1_token_text": tokenizer.decode([top1_id], clean_up_tokenization_spaces=False),
                    "is_tool_call_top1": bool(top1_id == tool_token_id),
                }
            )
    return outputs


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    canonical_rows = load_jsonl(args.canonical_path)
    subset_rows = pick_subset(canonical_rows, args.subset_size)
    system_prompt = resolve_system_prompt(args.model_path / args.system_prompt_file, args.today)

    tokenizer = AutoTokenizer.from_pretrained(str(args.model_path), trust_remote_code=True)
    tool_token_ids = tokenizer.encode("[TOOL_CALLS]", add_special_tokens=False)
    if len(tool_token_ids) != 1:
        raise RuntimeError(f"[TOOL_CALLS] should map to one token, got {tool_token_ids}")
    tool_token_id = int(tool_token_ids[0])

    model = build_model(
        args.model_path,
        args.dtype,
        args.device_map,
        args.gpu_max_memory,
        args.cpu_max_memory,
    )

    detail_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    subset_manifest_rows = []
    for verb in args.verbs:
        prompt_rows: list[dict[str, Any]] = []
        for row in subset_rows:
            rendered_user = replace_first_verb(row["clean_user_content"], verb)
            prompt_text = tokenizer.apply_chat_template(
                conversation=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": rendered_user},
                ],
                tools=row["tools_schema"],
                tokenize=False,
                add_generation_prompt=True,
            )
            encoded = tokenizer(prompt_text, add_special_tokens=False, return_tensors="pt")
            prompt_rows.append(
                {
                    "sample_id": row["sample_id"],
                    "split": row["split"],
                    "language": row["language"],
                    "candidate_verb": verb,
                    "prompt_path": "",
                    "prompt_token_length": int(encoded["attention_mask"][0].sum().item()),
                    "input_ids": encoded["input_ids"][0],
                    "attention_mask": encoded["attention_mask"][0],
                    "rendered_first_line": rendered_user.splitlines()[0],
                }
            )
            subset_manifest_rows.append(
                {
                    "sample_id": row["sample_id"],
                    "language": row["language"],
                    "split": row["split"],
                }
            )
        outputs = evaluate_prompts(model, tokenizer, prompt_rows, tool_token_id, args.batch_size)
        detail_rows.extend(
            {
                "sample_id": item["sample_id"],
                "split": item["split"],
                "language": item["language"],
                "candidate_verb": item["candidate_verb"],
                "rendered_first_line": item["rendered_first_line"],
                "prompt_token_length": item["prompt_token_length"],
                "tool_token_id": item["tool_token_id"],
                "tool_token_prob": item["tool_token_prob"],
                "tool_token_logit": item["tool_token_logit"],
                "top1_token_id": item["top1_token_id"],
                "top1_token_text": item["top1_token_text"],
                "is_tool_call_top1": item["is_tool_call_top1"],
            }
            for item in outputs
        )
        by_lang = defaultdict(list)
        for item in outputs:
            by_lang[item["language"]].append(int(item["is_tool_call_top1"]))
        top1_counter = Counter(item["top1_token_text"] for item in outputs)
        summary_rows.append(
            {
                "candidate_verb": verb,
                "n_samples": len(outputs),
                "tool_call_top1_rate": sum(int(item["is_tool_call_top1"]) for item in outputs) / len(outputs),
                "mean_tool_token_prob": sum(item["tool_token_prob"] for item in outputs) / len(outputs),
                "borderline_25_75_count": sum(1 for item in outputs if 0.25 <= item["tool_token_prob"] <= 0.75),
                "python_tool_call_rate": (sum(by_lang["python"]) / len(by_lang["python"])) if by_lang["python"] else None,
                "java_tool_call_rate": (sum(by_lang["java"]) / len(by_lang["java"])) if by_lang["java"] else None,
                "cpp_tool_call_rate": (sum(by_lang["cpp"]) / len(by_lang["cpp"])) if by_lang["cpp"] else None,
                "top1_mode_token": top1_counter.most_common(1)[0][0],
                "tool_call_count": sum(int(item["is_tool_call_top1"]) for item in outputs),
            }
        )

    summary_rows.sort(key=lambda row: (-row["tool_call_top1_rate"], -row["mean_tool_token_prob"], row["candidate_verb"]))
    write_csv(args.output_root / "verb_summary.csv", summary_rows)
    write_csv(args.output_root / "verb_details.csv", detail_rows)
    (args.output_root / "subset_manifest.json").write_text(
        json.dumps(
            {
                "subset_size": len(subset_rows),
                "sample_ids": [row["sample_id"] for row in subset_rows],
                "language_counts": dict(Counter(row["language"] for row in subset_rows)),
                "split_counts": dict(Counter(row["split"] for row in subset_rows)),
                "system_prompt_file": args.system_prompt_file,
                "verbs": list(args.verbs),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (args.output_root / "summary.md").write_text(
        "\n".join(
            [
                "# Devstral 50-sample execution-verb scan",
                "",
                f"- subset size: {len(subset_rows)}",
                f"- language counts: {dict(Counter(row['language'] for row in subset_rows))}",
                "",
            ]
            + [
                f"- `{row['candidate_verb']}`: top-1 tool-call rate = {100.0 * row['tool_call_top1_rate']:.2f}%, mean p(tool) = {row['mean_tool_token_prob']:.4f}, top1 mode = `{row['top1_mode_token']}`"
                for row in summary_rows
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary_rows[:10], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
