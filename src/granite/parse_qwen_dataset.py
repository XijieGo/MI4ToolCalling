#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
from datetime import datetime, timezone
from pathlib import Path

from transformers import AutoTokenizer

from granite_toolcall_common import (
    DEFAULT_CONVERTED_ROOT,
    DEFAULT_DATASET_ROOT,
    DEFAULT_MODEL_PATH,
    TOOL_CALL_TOKEN,
    ensure_dir,
    load_pair_metadata,
    normalize_json,
    parse_qwen_prompt,
    read_text,
    render_granite_prompt,
    write_json,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert Qwen tool-call pairs into Granite native prompts.")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_CONVERTED_ROOT)
    parser.add_argument("--max-samples", type=int, default=0)
    return parser.parse_args()


def sha1_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def main() -> None:
    args = parse_args()
    ensure_dir(args.output_root)
    ensure_dir(args.output_root / "clean")
    ensure_dir(args.output_root / "corrupt")

    tokenizer = AutoTokenizer.from_pretrained(str(args.model_path), trust_remote_code=True)
    tool_token_ids = tokenizer.encode(TOOL_CALL_TOKEN, add_special_tokens=False)
    if len(tool_token_ids) != 1:
        raise RuntimeError(f"{TOOL_CALL_TOKEN!r} must be a single token, got {tool_token_ids}")

    pair_rows = load_pair_metadata(args.dataset_root)
    if args.max_samples > 0:
        pair_rows = pair_rows[: args.max_samples]

    canonical_rows: list[dict] = []
    manifest_rows: list[dict] = []

    for order, pair in enumerate(pair_rows, start=1):
        clean_raw_text = read_text(pair["clean_source_path"])
        corrupt_raw_text = read_text(pair["corrupt_source_path"])
        clean_tools, clean_user_text = parse_qwen_prompt(clean_raw_text)
        corrupt_tools, corrupt_user_text = parse_qwen_prompt(corrupt_raw_text)
        if normalize_json(clean_tools) != normalize_json(corrupt_tools):
            raise ValueError(f"Tool schema mismatch for {pair['sample_id']}")

        rendered_clean_prompt = render_granite_prompt(
            tokenizer,
            user_text=clean_user_text,
            tools=clean_tools,
        )
        rendered_corrupt_prompt = render_granite_prompt(
            tokenizer,
            user_text=corrupt_user_text,
            tools=clean_tools,
        )

        clean_prompt_relpath = Path("clean") / pair["filename"]
        corrupt_prompt_relpath = Path("corrupt") / pair["filename"]
        clean_prompt_path = args.output_root / clean_prompt_relpath
        corrupt_prompt_path = args.output_root / corrupt_prompt_relpath
        clean_prompt_path.write_text(rendered_clean_prompt, encoding="utf-8")
        corrupt_prompt_path.write_text(rendered_corrupt_prompt, encoding="utf-8")

        clean_prompt_token_length = len(
            tokenizer(rendered_clean_prompt, add_special_tokens=False)["input_ids"]
        )
        corrupt_prompt_token_length = len(
            tokenizer(rendered_corrupt_prompt, add_special_tokens=False)["input_ids"]
        )

        canonical_row = {
            "order": order,
            "global_row_id": pair["global_row_id"],
            "sample_id": pair["sample_id"],
            "split": pair["split"],
            "dataset_name": pair["dataset_name"],
            "language": pair["language"],
            "template_kind": pair["template_kind"],
            "clean_candidate": pair["clean_candidate"],
            "corrupt_candidate": pair["corrupt_candidate"],
            "filename": pair["filename"],
            "tool_schema": clean_tools,
            "clean_user_content": clean_user_text,
            "corrupt_user_content": corrupt_user_text,
            "clean_source_path": str(pair["clean_source_path"]),
            "corrupt_source_path": str(pair["corrupt_source_path"]),
        }
        manifest_row = {
            "order": order,
            "global_row_id": pair["global_row_id"],
            "sample_id": pair["sample_id"],
            "split": pair["split"],
            "dataset_name": pair["dataset_name"],
            "language": pair["language"],
            "template_kind": pair["template_kind"],
            "clean_candidate": pair["clean_candidate"],
            "corrupt_candidate": pair["corrupt_candidate"],
            "filename": pair["filename"],
            "clean_prompt_relpath": str(clean_prompt_relpath),
            "corrupt_prompt_relpath": str(corrupt_prompt_relpath),
            "clean_prompt_path": str(clean_prompt_path.resolve()),
            "corrupt_prompt_path": str(corrupt_prompt_path.resolve()),
            "clean_prompt_token_length": clean_prompt_token_length,
            "corrupt_prompt_token_length": corrupt_prompt_token_length,
            "clean_prompt_char_length": len(rendered_clean_prompt),
            "corrupt_prompt_char_length": len(rendered_corrupt_prompt),
            "clean_prompt_sha1": sha1_text(rendered_clean_prompt),
            "corrupt_prompt_sha1": sha1_text(rendered_corrupt_prompt),
        }
        canonical_rows.append(canonical_row)
        manifest_rows.append(manifest_row)

    write_jsonl(args.output_root / "canonical_pairs.jsonl", canonical_rows)
    write_jsonl(args.output_root / "manifest.jsonl", manifest_rows)
    write_json(
        args.output_root / "conversion_summary.json",
        {
            "model_path": str(args.model_path),
            "dataset_root": str(args.dataset_root),
            "output_root": str(args.output_root),
            "n_pairs": len(manifest_rows),
            "tool_call_token": TOOL_CALL_TOKEN,
            "tool_call_token_id": int(tool_token_ids[0]),
            "render_policy": {
                "explicit_system_message": False,
                "add_generation_prompt": True,
                "thinking": False,
            },
            "rendered_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    print(f"Converted {len(manifest_rows)} pairs into {args.output_root}")


if __name__ == "__main__":
    main()
