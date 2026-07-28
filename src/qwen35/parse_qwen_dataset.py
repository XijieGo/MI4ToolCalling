#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "datasets"
DEFAULT_MODEL_PATH = Path(
    os.environ.get(
        "QWEN35_9B_PATH",
        str(PROJECT_ROOT / "external" / "models" / "Qwen3.5-9B"),
    )
)
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "results" / "Qwen3.5-9B" / "converted_dataset"

QWEN_SYSTEM_RE = re.compile(r"<\|im_start\|>system\n(.*?)<\|im_end\|>", re.DOTALL)
QWEN_USER_RE = re.compile(r"<\|im_start\|>user\n(.*?)<\|im_end\|>", re.DOTALL)
TOOLS_RE = re.compile(r"<tools>\s*(.*?)\s*</tools>", re.DOTALL)


@dataclass(frozen=True)
class PairRecord:
    sample_id: str
    global_row_id: int
    split: str
    language: str
    clean_candidate: str
    corrupt_candidate: str
    clean_source_path: str
    corrupt_source_path: str
    user_content_clean: str
    user_content_corrupt: str
    tools_schema: list[dict[str, Any]]
    metadata: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Parse Qwen3 clean/corrupt pairs and rerender them with Qwen3.5-9B's native chat template."
    )
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help="Enable Qwen3.5 thinking mode in add_generation_prompt output. Disabled by default.",
    )
    return parser.parse_args()


def parse_concatenated_json_objects(payload: str) -> list[dict[str, Any]]:
    decoder = json.JSONDecoder()
    items: list[dict[str, Any]] = []
    idx = 0
    text = payload.strip()
    while idx < len(text):
        while idx < len(text) and text[idx].isspace():
            idx += 1
        if idx >= len(text):
            break
        obj, next_idx = decoder.raw_decode(text, idx)
        if not isinstance(obj, dict):
            raise ValueError(f"Expected dict in tools payload, got {type(obj).__name__}")
        items.append(obj)
        idx = next_idx
    if not items:
        raise ValueError("No JSON tool objects found in payload.")
    return items


def parse_tools_payload(system_text: str) -> list[dict[str, Any]]:
    matches = [match.strip() for match in TOOLS_RE.findall(system_text) if match.strip()]
    if not matches:
        raise ValueError("Could not extract <tools>...</tools> payload.")
    payload = max(matches, key=len)
    if payload.startswith("["):
        tools = json.loads(payload)
        if not isinstance(tools, list):
            raise ValueError(f"Expected tools list, got {type(tools).__name__}")
        return tools
    return parse_concatenated_json_objects(payload)


def parse_qwen_prompt(path: Path) -> tuple[list[dict[str, Any]], str]:
    text = path.read_text(encoding="utf-8")
    system_match = QWEN_SYSTEM_RE.search(text)
    user_match = QWEN_USER_RE.search(text)
    if system_match is None or user_match is None:
        raise ValueError(f"Failed to parse Qwen prompt blocks from {path}")
    system_text = system_match.group(1).strip()
    user_text = user_match.group(1).strip()
    tools = parse_tools_payload(system_text)
    return tools, user_text


def make_sample_id(source_filename: str, global_row_id: int | None) -> str:
    stem = Path(source_filename).stem
    if global_row_id is None:
        return stem
    return f"gid{int(global_row_id)}_{stem}"


def build_pair_records(dataset_root: Path) -> list[PairRecord]:
    records: list[PairRecord] = []
    for split in ("train", "test"):
        clean_manifest_path = dataset_root / split / "clean" / "manifest.jsonl"
        with clean_manifest_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                source_filename = row["source_filename"]
                output_filename = row["output_filename"]
                clean_prompt_path = dataset_root / split / "clean" / output_filename
                corrupt_prompt_path = dataset_root / split / "corrupt" / output_filename
                clean_prompt_text = clean_prompt_path.read_text(encoding="utf-8")
                corrupt_prompt_text = corrupt_prompt_path.read_text(encoding="utf-8")
                clean_tools, clean_user = parse_qwen_prompt(clean_prompt_path)
                corrupt_tools, corrupt_user = parse_qwen_prompt(corrupt_prompt_path)
                clean_tools_json = json.dumps(clean_tools, ensure_ascii=False, sort_keys=True)
                corrupt_tools_json = json.dumps(corrupt_tools, ensure_ascii=False, sort_keys=True)
                if clean_tools_json != corrupt_tools_json:
                    raise ValueError(f"Tool schema mismatch between clean and corrupt for {source_filename}")
                metadata = {
                    "dataset": row.get("dataset"),
                    "dataset_name": row.get("dataset_name"),
                    "source_filename": source_filename,
                    "output_filename": output_filename,
                    "global_row_id": row.get("global_row_id"),
                    "sample_index": row.get("sample_index"),
                    "template_kind": row.get("template_kind"),
                    "original_source_id": row.get("original_source_id"),
                    "original_source_split": row.get("original_source_split"),
                    "clean_prompt_sha1": row.get("clean_prompt_sha1"),
                    "corrupt_prompt_sha1": row.get("corrupt_prompt_sha1"),
                    "clean_source_prompt_char_length": len(clean_prompt_text),
                    "corrupt_source_prompt_char_length": len(corrupt_prompt_text),
                }
                records.append(
                    PairRecord(
                        sample_id=make_sample_id(source_filename, row.get("global_row_id")),
                        global_row_id=int(row.get("global_row_id")),
                        split=split,
                        language=row["language"],
                        clean_candidate=row["clean_candidate"],
                        corrupt_candidate=row["corrupt_candidate"],
                        clean_source_path=str(clean_prompt_path.resolve()),
                        corrupt_source_path=str(corrupt_prompt_path.resolve()),
                        user_content_clean=clean_user,
                        user_content_corrupt=corrupt_user,
                        tools_schema=clean_tools,
                        metadata=metadata,
                    )
                )
    records.sort(key=lambda record: (0 if record.split == "train" else 1, record.global_row_id, record.sample_id))
    return records


def ensure_directories(output_root: Path) -> None:
    (output_root / "clean").mkdir(parents=True, exist_ok=True)
    (output_root / "corrupt").mkdir(parents=True, exist_ok=True)


def render_prompt(
    tokenizer: AutoTokenizer,
    user_content: str,
    tools_schema: list[dict[str, Any]],
    enable_thinking: bool,
) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": user_content}],
        tools=tools_schema,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    ensure_directories(args.output_root)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    tool_call_token_ids = tokenizer.encode("<tool_call>", add_special_tokens=False)
    tool_response_token_ids = tokenizer.encode("<tool_response>", add_special_tokens=False)
    think_token_ids = tokenizer.encode("<think>", add_special_tokens=False)
    if len(tool_call_token_ids) != 1:
        raise ValueError(f"<tool_call> is not single-token for Qwen3.5-9B: {tool_call_token_ids}")
    if len(tool_response_token_ids) != 1:
        raise ValueError(f"<tool_response> is not single-token for Qwen3.5-9B: {tool_response_token_ids}")
    if len(think_token_ids) != 1:
        raise ValueError(f"<think> is not single-token for Qwen3.5-9B: {think_token_ids}")

    pair_records = build_pair_records(args.dataset_root)
    canonical_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []

    for record in pair_records:
        clean_prompt = render_prompt(tokenizer, record.user_content_clean, record.tools_schema, args.enable_thinking)
        corrupt_prompt = render_prompt(tokenizer, record.user_content_corrupt, record.tools_schema, args.enable_thinking)

        clean_output_path = args.output_root / "clean" / f"{record.sample_id}.txt"
        corrupt_output_path = args.output_root / "corrupt" / f"{record.sample_id}.txt"
        write_text(clean_output_path, clean_prompt)
        write_text(corrupt_output_path, corrupt_prompt)

        clean_token_length = len(tokenizer.encode(clean_prompt, add_special_tokens=False))
        corrupt_token_length = len(tokenizer.encode(corrupt_prompt, add_special_tokens=False))

        canonical_rows.append(
            {
                "sample_id": record.sample_id,
                "split": record.split,
                "language": record.language,
                "clean_candidate": record.clean_candidate,
                "corrupt_candidate": record.corrupt_candidate,
                "user_content_clean": record.user_content_clean,
                "user_content_corrupt": record.user_content_corrupt,
                "tools_schema": record.tools_schema,
                "clean_source_path": record.clean_source_path,
                "corrupt_source_path": record.corrupt_source_path,
                **record.metadata,
            }
        )
        manifest_rows.append(
            {
                "sample_id": record.sample_id,
                "split": record.split,
                "language": record.language,
                "clean_candidate": record.clean_candidate,
                "corrupt_candidate": record.corrupt_candidate,
                "clean_source_path": record.clean_source_path,
                "corrupt_source_path": record.corrupt_source_path,
                "clean_prompt_path": str(clean_output_path.resolve()),
                "corrupt_prompt_path": str(corrupt_output_path.resolve()),
                "clean_prompt_token_length": clean_token_length,
                "corrupt_prompt_token_length": corrupt_token_length,
                "tool_token_id": int(tool_call_token_ids[0]),
                "tool_token_text": "<tool_call>",
                "tool_response_token_id": int(tool_response_token_ids[0]),
                "tool_response_token_text": "<tool_response>",
                "think_token_id": int(think_token_ids[0]),
                "think_token_text": "<think>",
                "enable_thinking": bool(args.enable_thinking),
                **record.metadata,
            }
        )

    write_jsonl(args.output_root / "canonical_pairs.jsonl", canonical_rows)
    write_jsonl(args.output_root / "manifest.jsonl", manifest_rows)

    summary = {
        "model_path": str(args.model_path.resolve()),
        "output_root": str(args.output_root.resolve()),
        "n_pairs": len(pair_records),
        "splits": {
            "train": sum(1 for record in pair_records if record.split == "train"),
            "test": sum(1 for record in pair_records if record.split == "test"),
        },
        "languages": {
            language: sum(1 for record in pair_records if record.language == language)
            for language in sorted({record.language for record in pair_records})
        },
        "tool_call_token_text": "<tool_call>",
        "tool_call_token_id": int(tool_call_token_ids[0]),
        "tool_response_token_text": "<tool_response>",
        "tool_response_token_id": int(tool_response_token_ids[0]),
        "think_token_text": "<think>",
        "think_token_id": int(think_token_ids[0]),
        "enable_thinking": bool(args.enable_thinking),
        "template_behavior_note": (
            "Prompts are rerendered with Qwen3.5-9B's native chat_template using tools=... and "
            "add_generation_prompt=True. By default thinking is disabled to keep the assistant prefill "
            "closer to a first-token tool-call/no-tool decision interface."
        ),
    }
    write_text(args.output_root / "conversion_summary.json", json.dumps(summary, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
