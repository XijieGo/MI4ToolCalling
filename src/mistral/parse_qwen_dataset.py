#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[2]


QWEN_SYSTEM_RE = re.compile(r"<\|im_start\|>system\n(.*?)<\|im_end\|>", re.DOTALL)
QWEN_USER_RE = re.compile(r"<\|im_start\|>user\n(.*?)<\|im_end\|>", re.DOTALL)
TOOLS_RE = re.compile(r"<tools>\s*(.*?)\s*</tools>", re.DOTALL)
SHORT_TOOLCALL_SYSTEM_PROMPT = """# Tools

You may call one or more functions to assist with the user query.

Use the available functions when the user asks you to create, write, update, complete, or save file content.

If tool use is appropriate, make the tool call. Otherwise answer normally."""
VIBE_SYSTEM_PROMPT_PATH = PROJECT_ROOT / "results" / "section6_generalization" / "mistral_3p2_24b" / "datasets" / "tokenizer_verification.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert Qwen-format tool-call pairs into Mistral-native prompts.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=PROJECT_ROOT / "datasets",
    )
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
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "results" / "Mistral-Small-3.2-24B-Instruct-2506" / "converted_dataset",
    )
    parser.add_argument(
        "--system-prompt-mode",
        type=str,
        choices=("official", "toolcall_short", "vibe"),
        default="vibe",
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def parse_tools_payload(payload: str) -> list[dict[str, Any]]:
    payload = payload.strip()
    if not payload:
        raise ValueError("Empty tools payload.")
    if payload.startswith("["):
        data = json.loads(payload)
        if not isinstance(data, list):
            raise ValueError(f"Expected list tools payload, got {type(data).__name__}.")
        return data
    decoder = json.JSONDecoder()
    items: list[dict[str, Any]] = []
    idx = 0
    while idx < len(payload):
        while idx < len(payload) and payload[idx].isspace():
            idx += 1
        if idx >= len(payload):
            break
        item, idx = decoder.raw_decode(payload, idx)
        if not isinstance(item, dict):
            raise ValueError(f"Expected dict tool schema, got {type(item).__name__}.")
        items.append(item)
    if not items:
        raise ValueError("No tool objects decoded from payload.")
    return items


def parse_qwen_prompt(text: str) -> tuple[str, list[dict[str, Any]]]:
    system_match = QWEN_SYSTEM_RE.search(text)
    user_match = QWEN_USER_RE.search(text)
    if system_match is None or user_match is None:
        raise ValueError("Failed to parse Qwen system/user blocks.")
    system_text = system_match.group(1).strip()
    user_text = user_match.group(1).strip()
    tool_matches = [match.strip() for match in TOOLS_RE.findall(system_text)]
    tool_matches = [match for match in tool_matches if match]
    if not tool_matches:
        raise ValueError("Failed to find <tools> payload.")
    tools = parse_tools_payload(max(tool_matches, key=len))
    return user_text, tools


def format_system_prompt(template_path: Path, mode: str) -> str:
    if mode == "toolcall_short":
        return SHORT_TOOLCALL_SYSTEM_PROMPT
    if mode == "vibe":
        payload = json.loads(VIBE_SYSTEM_PROMPT_PATH.read_text(encoding="utf-8"))
        preview = str(payload.get("system_prompt_preview", "")).strip()
        if preview:
            return preview
        raise ValueError(f"Missing system prompt preview in {VIBE_SYSTEM_PROMPT_PATH}")
    today = date.today()
    yesterday = today - timedelta(days=1)
    template = template_path.read_text(encoding="utf-8")
    return template.format(
        name="Mistral-Small-3.2-24B-Instruct-2506",
        today=today.isoformat(),
        yesterday=yesterday.isoformat(),
    )


def build_split_lookup(dataset_root: Path) -> dict[str, str]:
    split_lookup: dict[str, str] = {}
    for split in ("train", "test"):
        manifest_path = dataset_root / split / "clean" / "manifest.jsonl"
        for row in read_jsonl(manifest_path):
            filename = str(row["output_filename"])
            sample_id = Path(filename).stem
            split_lookup[sample_id] = split
    return split_lookup

def main() -> None:
    args = parse_args()
    ensure_dir(args.output_root)
    ensure_dir(args.output_root / "clean")
    ensure_dir(args.output_root / "corrupt")

    tokenizer = AutoTokenizer.from_pretrained(str(args.model_path), trust_remote_code=True)
    system_prompt = format_system_prompt(args.model_path / "SYSTEM_PROMPT.txt", args.system_prompt_mode)
    split_lookup = build_split_lookup(args.dataset_root)
    clean_rows = read_jsonl(args.dataset_root / "clean" / "manifest.jsonl")

    canonical_path = args.output_root / "canonical_pairs.jsonl"
    manifest_path = args.output_root / "manifest.jsonl"
    verification_path = args.output_root / "tokenizer_verification.json"

    tool_token_text = "[TOOL_CALLS]"
    tool_token_encode_ids = tokenizer.encode(tool_token_text, add_special_tokens=False)
    tool_token_id = tokenizer.convert_tokens_to_ids(tool_token_text)

    verification_payload = {
        "model_path": str(args.model_path),
        "tool_token_text": tool_token_text,
        "tool_token_id_via_convert": int(tool_token_id),
        "tool_token_ids_via_encode": [int(x) for x in tool_token_encode_ids],
        "tool_token_encode_len": len(tool_token_encode_ids),
        "available_tools_token_id": int(tokenizer.convert_tokens_to_ids("[AVAILABLE_TOOLS]")),
        "inst_token_id": int(tokenizer.convert_tokens_to_ids("[INST]")),
        "system_prompt_token_id": int(tokenizer.convert_tokens_to_ids("[SYSTEM_PROMPT]")),
        "system_prompt_mode": args.system_prompt_mode,
        "system_prompt_preview": system_prompt,
        "note": (
            "In this local MistralCommon backend, convert_tokens_to_ids resolves [TOOL_CALLS] to the "
            "dedicated special token id, while encode(add_special_tokens=False) splits the literal string "
            "into multiple pieces. Downstream behavior scanning therefore uses the dedicated special token id."
        ),
    }
    verification_path.write_text(json.dumps(verification_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    canonical_handle = canonical_path.open("w", encoding="utf-8")
    manifest_handle = manifest_path.open("w", encoding="utf-8")
    try:
        for row in clean_rows:
            filename = str(row["output_filename"])
            sample_id = Path(filename).stem
            clean_prompt_path = args.dataset_root / "clean" / filename
            corrupt_prompt_path = args.dataset_root / "corrupt" / filename
            clean_text = clean_prompt_path.read_text(encoding="utf-8")
            corrupt_text = corrupt_prompt_path.read_text(encoding="utf-8")

            clean_user, clean_tools = parse_qwen_prompt(clean_text)
            corrupt_user, corrupt_tools = parse_qwen_prompt(corrupt_text)
            if clean_tools != corrupt_tools:
                raise ValueError(f"Tool schema mismatch in pair {sample_id}.")

            clean_messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": clean_user},
            ]
            corrupt_messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": corrupt_user},
            ]

            clean_tokens = tokenizer.apply_chat_template(
                clean_messages,
                tools=clean_tools,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
            clean_rendered = tokenizer.decode(
                clean_tokens["input_ids"][0].tolist(),
                clean_up_tokenization_spaces=False,
            )
            corrupt_tokens = tokenizer.apply_chat_template(
                corrupt_messages,
                tools=clean_tools,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
            corrupt_rendered = tokenizer.decode(
                corrupt_tokens["input_ids"][0].tolist(),
                clean_up_tokenization_spaces=False,
            )

            clean_output_path = args.output_root / "clean" / filename
            corrupt_output_path = args.output_root / "corrupt" / filename
            clean_output_path.write_text(clean_rendered, encoding="utf-8")
            corrupt_output_path.write_text(corrupt_rendered, encoding="utf-8")

            split = split_lookup.get(sample_id, "unknown")
            canonical_record = {
                "sample_id": sample_id,
                "split": split,
                "language": row.get("language"),
                "clean_candidate": row.get("clean_candidate"),
                "corrupt_candidate": row.get("corrupt_candidate"),
                "user_content_clean": clean_user,
                "user_content_corrupt": corrupt_user,
                "tools_schema": clean_tools,
            }
            manifest_record = {
                "sample_id": sample_id,
                "split": split,
                "language": row.get("language"),
                "clean_candidate": row.get("clean_candidate"),
                "corrupt_candidate": row.get("corrupt_candidate"),
                "clean_prompt_path": str(clean_output_path),
                "corrupt_prompt_path": str(corrupt_output_path),
                "clean_prompt_token_length": int(clean_tokens["attention_mask"][0].sum().item()),
                "corrupt_prompt_token_length": int(corrupt_tokens["attention_mask"][0].sum().item()),
            }
            canonical_handle.write(json.dumps(canonical_record, ensure_ascii=False) + "\n")
            manifest_handle.write(json.dumps(manifest_record, ensure_ascii=False) + "\n")
    finally:
        canonical_handle.close()
        manifest_handle.close()


if __name__ == "__main__":
    main()
