#!/usr/bin/env python3
"""Render the v2 1,500 Qwen contrastive pairs with a model's native tool template.

The source prompts are used only as a transport format for the paired user turn
and JSON tool schema.  Each target model receives its own chat-template system
prompt, tool serialization, and non-thinking assistant prefill.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[2]
QWEN_SYSTEM_RE = re.compile(r"<\|im_start\|>system\n(.*?)<\|im_end\|>", re.DOTALL)
QWEN_USER_RE = re.compile(r"<\|im_start\|>user\n(.*?)<\|im_end\|>", re.DOTALL)
TOOLS_RE = re.compile(r"<tools>\s*(.*?)\s*</tools>", re.DOTALL)
TOOL_CALL_TEXT = "<tool_call>"


@dataclass(frozen=True)
class SourcePair:
    sample_id: str
    global_row_id: int
    split: str
    language: str
    clean_candidate: str
    corrupt_candidate: str
    clean_user_content: str
    corrupt_user_content: str
    tools_schema: list[dict[str, Any]]
    clean_source_path: Path
    corrupt_source_path: Path
    metadata: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render v2 Qwen tool-call pairs through a target model's native chat template."
    )
    parser.add_argument("--dataset-root", type=Path, default=PROJECT_ROOT / "datasets")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--model-label", type=str, required=True)
    parser.add_argument("--template-mode", choices=("hermes", "smollm3"), required=True)
    parser.add_argument(
        "--system-prompt-file",
        type=Path,
        default=None,
        help=(
            "Optional model-native system instruction. Hermes uses it in place of its default system "
            "message; SmolLM3 uses it as its system message (include /no_think when desired)."
        ),
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Explicitly replace a previous converted dataset at --output-root.",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def parse_concatenated_json_objects(payload: str) -> list[dict[str, Any]]:
    decoder = json.JSONDecoder()
    values: list[dict[str, Any]] = []
    index = 0
    while index < len(payload):
        while index < len(payload) and payload[index].isspace():
            index += 1
        if index >= len(payload):
            break
        value, index = decoder.raw_decode(payload, index)
        if not isinstance(value, dict):
            raise ValueError(f"Expected a JSON object in <tools>, got {type(value).__name__}.")
        values.append(value)
    if not values:
        raise ValueError("No tool schemas found in source prompt.")
    return values


def parse_source_prompt(path: Path) -> tuple[list[dict[str, Any]], str]:
    text = path.read_text(encoding="utf-8")
    system_match = QWEN_SYSTEM_RE.search(text)
    user_match = QWEN_USER_RE.search(text)
    if system_match is None or user_match is None:
        raise ValueError(f"Could not parse Qwen source prompt: {path}")
    payloads = [match.strip() for match in TOOLS_RE.findall(system_match.group(1)) if match.strip()]
    if not payloads:
        raise ValueError(f"Could not find <tools> schema in {path}")
    payload = max(payloads, key=len)
    if payload.startswith("["):
        tools = json.loads(payload)
        if not isinstance(tools, list) or not all(isinstance(tool, dict) for tool in tools):
            raise ValueError(f"Invalid list-form tool schema in {path}")
    else:
        tools = parse_concatenated_json_objects(payload)
    return tools, user_match.group(1).strip()


def sample_id(source_filename: str, global_row_id: int) -> str:
    return f"gid{global_row_id}_{Path(source_filename).stem}"


def load_pairs(dataset_root: Path) -> list[SourcePair]:
    pairs: list[SourcePair] = []
    for split in ("train", "test"):
        manifest_path = dataset_root / split / "clean" / "manifest.jsonl"
        for row in read_jsonl(manifest_path):
            filename = str(row["output_filename"])
            clean_source_path = dataset_root / split / "clean" / filename
            corrupt_source_path = dataset_root / split / "corrupt" / filename
            clean_tools, clean_user = parse_source_prompt(clean_source_path)
            corrupt_tools, corrupt_user = parse_source_prompt(corrupt_source_path)
            if json.dumps(clean_tools, sort_keys=True, ensure_ascii=False) != json.dumps(
                corrupt_tools, sort_keys=True, ensure_ascii=False
            ):
                raise ValueError(f"Tool schema differs within source pair {filename}")
            global_row_id = int(row["global_row_id"])
            pairs.append(
                SourcePair(
                    sample_id=sample_id(str(row["source_filename"]), global_row_id),
                    global_row_id=global_row_id,
                    split=split,
                    language=str(row["language"]),
                    clean_candidate=str(row["clean_candidate"]),
                    corrupt_candidate=str(row["corrupt_candidate"]),
                    clean_user_content=clean_user,
                    corrupt_user_content=corrupt_user,
                    tools_schema=clean_tools,
                    clean_source_path=clean_source_path,
                    corrupt_source_path=corrupt_source_path,
                    metadata={
                        "dataset": row.get("dataset"),
                        "dataset_name": row.get("dataset_name"),
                        "source_filename": row.get("source_filename"),
                        "output_filename": filename,
                        "template_kind": row.get("template_kind"),
                        "original_source_id": row.get("original_source_id"),
                        "original_source_split": row.get("original_source_split"),
                    },
                )
            )
    pairs.sort(key=lambda pair: (0 if pair.split == "train" else 1, pair.global_row_id, pair.sample_id))
    return pairs


def render_prompt(
    tokenizer,
    *,
    template_mode: str,
    user_content: str,
    tools_schema: list[dict[str, Any]],
    system_instruction: str | None,
) -> tuple[str, dict[str, Any]]:
    if template_mode == "hermes":
        messages = (
            [{"role": "system", "content": system_instruction}, {"role": "user", "content": user_content}]
            if system_instruction is not None
            else [{"role": "user", "content": user_content}]
        )
        kwargs: dict[str, Any] = {"tools": tools_schema, "thinking": False}
        policy = {
            "messages": (
                "custom Hermes system instruction plus one user message"
                if system_instruction is not None
                else "native default Hermes system prompt plus one user message"
            ),
            "template_kwargs": {"tools": "source schema", "thinking": False},
            "non_thinking": True,
        }
    elif template_mode == "smollm3":
        smollm_system = system_instruction or "/no_think"
        messages = [
            {"role": "system", "content": smollm_system},
            {"role": "user", "content": user_content},
        ]
        kwargs = {"xml_tools": tools_schema, "enable_thinking": False}
        policy = {
            "messages": "native SmolLM3 system instruction plus one user message",
            "template_kwargs": {"xml_tools": "source schema", "enable_thinking": False},
            "non_thinking": True,
        }
    else:  # pragma: no cover - argparse enforces the possible modes.
        raise ValueError(f"Unsupported template mode: {template_mode}")
    return (
        tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **kwargs),
        policy,
    )


def assert_only_verb_diff(clean_user: str, corrupt_user: str, clean_prompt: str, corrupt_prompt: str) -> None:
    clean_word, clean_sep, clean_tail = clean_user.partition(" ")
    corrupt_word, corrupt_sep, corrupt_tail = corrupt_user.partition(" ")
    if not clean_sep or not corrupt_sep or clean_tail != corrupt_tail or clean_word.lower() == corrupt_word.lower():
        raise ValueError("Source pair is not a one-word clean/corrupt contrast.")
    clean_index = clean_prompt.find(clean_user)
    corrupt_index = corrupt_prompt.find(corrupt_user)
    if clean_index < 0 or corrupt_index < 0:
        raise ValueError("Rendered prompt does not contain the source user message.")
    if (
        clean_prompt[:clean_index] != corrupt_prompt[:corrupt_index]
        or clean_prompt[clean_index + len(clean_user) :] != corrupt_prompt[corrupt_index + len(corrupt_user) :]
    ):
        raise ValueError("Rendered clean/corrupt prompts differ outside the user verb span.")


def prepare_output_root(path: Path, *, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise FileExistsError(f"Output root already contains files: {path}; pass --overwrite to replace it.")
        for child in sorted(path.iterdir(), reverse=True):
            if child.is_dir():
                for nested in sorted(child.rglob("*"), reverse=True):
                    if nested.is_file() or nested.is_symlink():
                        nested.unlink()
                    elif nested.is_dir():
                        nested.rmdir()
                child.rmdir()
            else:
                child.unlink()
    (path / "clean").mkdir(parents=True, exist_ok=True)
    (path / "corrupt").mkdir(parents=True, exist_ok=True)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(str(args.model_path), trust_remote_code=True)
    tool_token_ids = tokenizer.encode(TOOL_CALL_TEXT, add_special_tokens=False)
    if len(tool_token_ids) != 1:
        raise ValueError(f"{TOOL_CALL_TEXT} must be one target token, got {tool_token_ids}")
    pairs = load_pairs(args.dataset_root)
    system_instruction = (
        args.system_prompt_file.read_text(encoding="utf-8").strip()
        if args.system_prompt_file is not None
        else None
    )
    if args.template_mode == "smollm3" and system_instruction is not None and "/no_think" not in system_instruction:
        raise ValueError("The SmolLM3 system-prompt file must include /no_think for this first-token setup.")
    prepare_output_root(args.output_root, overwrite=args.overwrite)

    canonical_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    unequal_token_length_count = 0
    render_policy: dict[str, Any] | None = None
    for pair in pairs:
        clean_prompt, policy = render_prompt(
            tokenizer,
            template_mode=args.template_mode,
            user_content=pair.clean_user_content,
            tools_schema=pair.tools_schema,
            system_instruction=system_instruction,
        )
        corrupt_prompt, _ = render_prompt(
            tokenizer,
            template_mode=args.template_mode,
            user_content=pair.corrupt_user_content,
            tools_schema=pair.tools_schema,
            system_instruction=system_instruction,
        )
        assert_only_verb_diff(
            pair.clean_user_content, pair.corrupt_user_content, clean_prompt, corrupt_prompt
        )
        clean_path = args.output_root / "clean" / f"{pair.sample_id}.txt"
        corrupt_path = args.output_root / "corrupt" / f"{pair.sample_id}.txt"
        clean_path.write_text(clean_prompt, encoding="utf-8")
        corrupt_path.write_text(corrupt_prompt, encoding="utf-8")
        clean_length = len(tokenizer.encode(clean_prompt, add_special_tokens=False))
        corrupt_length = len(tokenizer.encode(corrupt_prompt, add_special_tokens=False))
        unequal_token_length_count += int(clean_length != corrupt_length)
        render_policy = policy
        canonical_rows.append(
            {
                "sample_id": pair.sample_id,
                "global_row_id": pair.global_row_id,
                "split": pair.split,
                "language": pair.language,
                "clean_candidate": pair.clean_candidate,
                "corrupt_candidate": pair.corrupt_candidate,
                "user_content_clean": pair.clean_user_content,
                "user_content_corrupt": pair.corrupt_user_content,
                "tools_schema": pair.tools_schema,
                "clean_source_path": str(pair.clean_source_path.resolve()),
                "corrupt_source_path": str(pair.corrupt_source_path.resolve()),
                **pair.metadata,
            }
        )
        manifest_rows.append(
            {
                "sample_id": pair.sample_id,
                "global_row_id": pair.global_row_id,
                "split": pair.split,
                "language": pair.language,
                "clean_candidate": pair.clean_candidate,
                "corrupt_candidate": pair.corrupt_candidate,
                "clean_prompt_path": str(clean_path.resolve()),
                "corrupt_prompt_path": str(corrupt_path.resolve()),
                "clean_prompt_token_length": clean_length,
                "corrupt_prompt_token_length": corrupt_length,
                "tool_token_text": TOOL_CALL_TEXT,
                "tool_token_id": int(tool_token_ids[0]),
            }
        )
    write_jsonl(args.output_root / "canonical_pairs.jsonl", canonical_rows)
    write_jsonl(args.output_root / "manifest.jsonl", manifest_rows)
    summary = {
        "model_label": args.model_label,
        "model_path": str(args.model_path.resolve()),
        "template_mode": args.template_mode,
        "render_policy": render_policy,
        "system_prompt_file": str(args.system_prompt_file) if args.system_prompt_file is not None else None,
        "system_prompt_sha256": (
            hashlib.sha256(system_instruction.encode("utf-8")).hexdigest()
            if system_instruction is not None
            else None
        ),
        "conversion_date": date.today().isoformat(),
        "n_pairs": len(pairs),
        "splits": {split: sum(pair.split == split for pair in pairs) for split in ("train", "test")},
        "tool_call_token_text": TOOL_CALL_TEXT,
        "tool_call_token_id": int(tool_token_ids[0]),
        "unequal_clean_corrupt_token_length_count": unequal_token_length_count,
    }
    (args.output_root / "conversion_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
