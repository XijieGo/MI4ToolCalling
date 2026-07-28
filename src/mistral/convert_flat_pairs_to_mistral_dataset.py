#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[2]
QWEN_SYSTEM_RE = re.compile(r"<\|im_start\|>system\n(.*?)<\|im_end\|>", re.DOTALL)
QWEN_USER_RE = re.compile(r"<\|im_start\|>user\n(.*?)<\|im_end\|>", re.DOTALL)
TOOLS_RE = re.compile(r"<tools>\s*(.*?)\s*</tools>", re.DOTALL)
DEFAULT_SYSTEM_PROMPT_FILE = PROJECT_ROOT / "configs" / "prompts" / "mistral_vibe_system_prompt.txt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert flat clean_N/corrupt_N pairs into a Mistral evaluation dataset.")
    parser.add_argument(
        "--pairs-root",
        type=Path,
        default=PROJECT_ROOT / "results" / "Mistral-Small-3.2-24B-Instruct-2506" / "datasets",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "results" / "Mistral-Small-3.2-24B-Instruct-2506" / "datasets_mistral_vibe",
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
    prompt_source = parser.add_mutually_exclusive_group()
    prompt_source.add_argument(
        "--system-prompt-file",
        type=Path,
        default=None,
        help="Versioned plain-text VIBE system prompt used for fresh native rendering.",
    )
    prompt_source.add_argument(
        "--system-prompt-metadata",
        type=Path,
        default=None,
        help="Legacy JSON fallback containing `system_prompt_preview`; not used by the canonical runner.",
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def parse_tools_payload(payload: str) -> list[dict[str, Any]]:
    payload = payload.strip()
    if payload.startswith("["):
        data = json.loads(payload)
        if not isinstance(data, list):
            raise ValueError(f"Expected list tools payload, got {type(data).__name__}.")
        return data
    decoder = json.JSONDecoder()
    idx = 0
    items: list[dict[str, Any]] = []
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
        raise ValueError("No tool objects decoded.")
    return items


def parse_qwen_prompt(text: str) -> tuple[str, list[dict[str, Any]]]:
    system_match = QWEN_SYSTEM_RE.search(text)
    user_match = QWEN_USER_RE.search(text)
    if system_match is None or user_match is None:
        raise ValueError("Failed to parse system/user blocks.")
    system_text = system_match.group(1).strip()
    user_text = user_match.group(1).strip()
    tool_matches = [match.strip() for match in TOOLS_RE.findall(system_text)]
    tool_matches = [match for match in tool_matches if match]
    if not tool_matches:
        raise ValueError("Missing tools payload.")
    tools = parse_tools_payload(max(tool_matches, key=len))
    return user_text, tools


def load_system_prompt(*, text_path: Path | None, metadata_path: Path | None) -> str:
    if text_path is not None:
        prompt = text_path.read_text(encoding="utf-8").strip()
        if prompt:
            return prompt
        raise ValueError(f"System prompt file is empty: {text_path}")
    if metadata_path is None:
        raise ValueError("Provide --system-prompt-file or --system-prompt-metadata.")
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    prompt = str(payload.get("system_prompt_preview", "")).strip()
    if not prompt:
        raise ValueError(f"Missing system_prompt_preview in {metadata_path}")
    return prompt


def main() -> None:
    args = parse_args()
    ensure_dir(args.output_root)
    ensure_dir(args.output_root / "clean")
    ensure_dir(args.output_root / "corrupt")

    tokenizer = AutoTokenizer.from_pretrained(str(args.model_path), trust_remote_code=True)
    prompt_file = args.system_prompt_file
    if prompt_file is None and args.system_prompt_metadata is None:
        prompt_file = DEFAULT_SYSTEM_PROMPT_FILE
    system_prompt = load_system_prompt(text_path=prompt_file, metadata_path=args.system_prompt_metadata)
    pair_manifest = [json.loads(line) for line in (args.pairs_root / "pair_manifest.jsonl").read_text().splitlines() if line.strip()]

    canonical_path = args.output_root / "canonical_pairs.jsonl"
    manifest_path = args.output_root / "manifest.jsonl"
    verification_path = args.output_root / "tokenizer_verification.json"

    verification_payload = {
        "model_path": str(args.model_path),
        "tool_token_text": "[TOOL_CALLS]",
        "tool_token_id_via_convert": int(tokenizer.convert_tokens_to_ids("[TOOL_CALLS]")),
        "tool_token_ids_via_encode": tokenizer.encode("[TOOL_CALLS]", add_special_tokens=False),
        "system_prompt_mode": "vibe",
        "system_prompt_preview": system_prompt,
    }
    verification_path.write_text(json.dumps(verification_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    with canonical_path.open("w", encoding="utf-8") as canonical_handle, manifest_path.open("w", encoding="utf-8") as manifest_handle:
        for row in pair_manifest:
            pair_id = int(row["pair_id"])
            clean_src = (args.pairs_root / row["clean_filename"]).read_text(encoding="utf-8")
            corrupt_src = (args.pairs_root / row["corrupt_filename"]).read_text(encoding="utf-8")
            clean_user, clean_tools = parse_qwen_prompt(clean_src)
            corrupt_user, corrupt_tools = parse_qwen_prompt(corrupt_src)
            if clean_tools != corrupt_tools:
                raise ValueError(f"Tool schema mismatch in pair {pair_id}")

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
            corrupt_tokens = tokenizer.apply_chat_template(
                corrupt_messages,
                tools=clean_tools,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
            clean_rendered = tokenizer.decode(clean_tokens["input_ids"][0].tolist(), clean_up_tokenization_spaces=False)
            corrupt_rendered = tokenizer.decode(corrupt_tokens["input_ids"][0].tolist(), clean_up_tokenization_spaces=False)

            clean_out = args.output_root / "clean" / row["clean_filename"]
            corrupt_out = args.output_root / "corrupt" / row["corrupt_filename"]
            clean_out.write_text(clean_rendered, encoding="utf-8")
            corrupt_out.write_text(corrupt_rendered, encoding="utf-8")

            canonical_record = {
                "sample_id": f"pair_{pair_id}",
                "split": "custom500",
                "language": row["language"],
                "clean_candidate": row["assigned_clean_candidate"],
                "corrupt_candidate": row["assigned_corrupt_candidate"],
                "user_content_clean": clean_user,
                "user_content_corrupt": corrupt_user,
                "tools_schema": clean_tools,
                "source_sample_id": row["source_sample_id"],
                "source_filename": row["source_filename"],
                "dataset_name": row["dataset_name"],
            }
            manifest_record = {
                "sample_id": f"pair_{pair_id}",
                "split": "custom500",
                "language": row["language"],
                "clean_candidate": row["assigned_clean_candidate"],
                "corrupt_candidate": row["assigned_corrupt_candidate"],
                "clean_prompt_path": str(clean_out),
                "corrupt_prompt_path": str(corrupt_out),
                "clean_prompt_token_length": int(clean_tokens["attention_mask"][0].sum().item()),
                "corrupt_prompt_token_length": int(corrupt_tokens["attention_mask"][0].sum().item()),
                "source_sample_id": row["source_sample_id"],
                "source_filename": row["source_filename"],
                "dataset_name": row["dataset_name"],
            }
            canonical_handle.write(json.dumps(canonical_record, ensure_ascii=False) + "\n")
            manifest_handle.write(json.dumps(manifest_record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
