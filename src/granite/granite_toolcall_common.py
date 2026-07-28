#!/usr/bin/env python3
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable

import sys

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from artifact_paths import GRANITE_3P3_8B_PATH  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "datasets"
DEFAULT_MODEL_PATH = GRANITE_3P3_8B_PATH
DEFAULT_RESULTS_ROOT = PROJECT_ROOT / "results" / "section6_generalization" / "granite_3p3_8b"
DEFAULT_CONVERTED_ROOT = DEFAULT_RESULTS_ROOT / "converted_dataset"
DEFAULT_BEHAVIOR_ROOT = DEFAULT_RESULTS_ROOT / "behavior_scan"
DEFAULT_LOG_ROOT = DEFAULT_RESULTS_ROOT / "logs"
TOOL_CALL_TOKEN = "<|tool_call|>"

QWEN_USER_RE = re.compile(r"<\|im_start\|>user\n(.*?)<\|im_end\|>", re.DOTALL)
TOOLS_RE = re.compile(r"<tools>\s*(.*?)\s*</tools>", re.DOTALL)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, payload: dict) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def normalize_json(payload: object) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def filename_from_row(row: dict) -> str:
    filename = str(row.get("output_filename") or row.get("source_filename") or "").strip()
    if not filename:
        raise ValueError(f"Row missing filename: {row}")
    return filename


def sample_id_from_filename(filename: str) -> str:
    return Path(filename).stem


def parse_tools_payload(text: str) -> list[dict[str, object]]:
    decoder = json.JSONDecoder()
    idx = 0
    items: list[dict[str, object]] = []
    payload = text.strip()
    while idx < len(payload):
        while idx < len(payload) and payload[idx].isspace():
            idx += 1
        if idx >= len(payload):
            break
        item, offset = decoder.raw_decode(payload, idx)
        if not isinstance(item, dict):
            raise ValueError(f"Unexpected tool payload item type: {type(item).__name__}")
        items.append(item)
        idx = offset
    if not items:
        raise ValueError("No JSON tool objects found in payload.")
    return items


def parse_qwen_prompt(text: str) -> tuple[list[dict[str, object]], str]:
    user_match = QWEN_USER_RE.search(text)
    if user_match is None:
        raise ValueError("Could not parse Qwen user block.")
    user_text = user_match.group(1).strip()

    tool_matches = [match.strip() for match in TOOLS_RE.findall(text)]
    tool_matches = [match for match in tool_matches if match]
    if not tool_matches:
        raise ValueError("Could not extract <tools> block.")
    tools_payload = max(tool_matches, key=len)
    if tools_payload.startswith("["):
        tools_obj = json.loads(tools_payload)
        if not isinstance(tools_obj, list):
            raise ValueError(f"Unexpected tools payload type: {type(tools_obj).__name__}")
        tools = tools_obj
    else:
        tools = parse_tools_payload(tools_payload)
    return tools, user_text


def render_granite_prompt(tokenizer, *, user_text: str, tools: list[dict[str, object]]) -> str:
    messages = [{"role": "user", "content": user_text}]
    try:
        rendered = tokenizer.apply_chat_template(
            messages,
            tools=tools,
            tokenize=False,
            add_generation_prompt=True,
            thinking=False,
        )
    except TypeError:
        rendered = tokenizer.apply_chat_template(
            messages,
            tools=tools,
            tokenize=False,
            add_generation_prompt=True,
        )
    if not isinstance(rendered, str) or not rendered.strip():
        raise ValueError("Granite chat template returned an empty prompt.")
    return rendered


def load_split_map(dataset_root: Path) -> dict[str, str]:
    split_map: dict[str, str] = {}
    for split in ("train", "test"):
        manifest_path = dataset_root / split / "clean" / "manifest.jsonl"
        for row in read_jsonl(manifest_path):
            filename = filename_from_row(row)
            if filename in split_map:
                raise ValueError(f"Duplicate filename across splits: {filename}")
            split_map[filename] = split
    return split_map


def load_pair_metadata(dataset_root: Path) -> list[dict]:
    # The frozen v2 dataset is split as ``train/`` and ``test/``.  Earlier
    # Granite utilities expected a temporary flat ``clean/`` directory, which
    # made the canonical dataset unusable as a direct input.  Prefer the
    # split-aware layout while keeping the old branch for historical reruns.
    if (dataset_root / "train" / "clean" / "manifest.jsonl").exists():
        pairs: list[dict] = []
        for split in ("train", "test"):
            manifest_path = dataset_root / split / "clean" / "manifest.jsonl"
            if not manifest_path.exists():
                raise FileNotFoundError(manifest_path)
            for row in read_jsonl(manifest_path):
                filename = filename_from_row(row)
                clean_source_path = dataset_root / split / "clean" / filename
                corrupt_source_path = dataset_root / split / "corrupt" / filename
                if not clean_source_path.exists() or not corrupt_source_path.exists():
                    raise FileNotFoundError(
                        f"Missing v2 clean/corrupt pair for {split}/{filename}"
                    )
                pairs.append(
                    {
                        "global_row_id": row.get("global_row_id"),
                        "sample_id": sample_id_from_filename(filename),
                        "filename": filename,
                        "split": split,
                        "dataset_name": row.get("dataset_name") or row.get("dataset"),
                        "language": row.get("language"),
                        "template_kind": row.get("template_kind"),
                        "clean_candidate": row.get("clean_candidate"),
                        "corrupt_candidate": row.get("corrupt_candidate"),
                        "clean_source_path": clean_source_path.resolve(),
                        "corrupt_source_path": corrupt_source_path.resolve(),
                        "source_filename": row.get("source_filename"),
                        "original_first_user_line": row.get("original_first_user_line"),
                        "clean_rendered_first_user_line": row.get("clean_rendered_first_user_line"),
                        "corrupt_rendered_first_user_line": row.get("corrupt_rendered_first_user_line"),
                    }
                )
        if not pairs:
            raise RuntimeError(f"No split pairs found under {dataset_root}")
        return pairs

    split_map = load_split_map(dataset_root)
    rows = read_jsonl(dataset_root / "clean" / "manifest.jsonl")
    pairs: list[dict] = []
    for row in rows:
        filename = filename_from_row(row)
        clean_source_path = dataset_root / "clean" / filename
        corrupt_source_path = dataset_root / "corrupt" / filename
        if not clean_source_path.exists():
            raise FileNotFoundError(clean_source_path)
        if not corrupt_source_path.exists():
            raise FileNotFoundError(corrupt_source_path)
        split = split_map.get(filename)
        if split is None:
            raise ValueError(f"No split assignment found for {filename}")
        pairs.append(
            {
                "global_row_id": row.get("global_row_id"),
                "sample_id": sample_id_from_filename(filename),
                "filename": filename,
                "split": split,
                "dataset_name": row.get("dataset_name") or row.get("dataset"),
                "language": row.get("language"),
                "template_kind": row.get("template_kind"),
                "clean_candidate": row.get("clean_candidate"),
                "corrupt_candidate": row.get("corrupt_candidate"),
                "clean_source_path": clean_source_path.resolve(),
                "corrupt_source_path": corrupt_source_path.resolve(),
                "source_filename": row.get("source_filename"),
                "original_first_user_line": row.get("original_first_user_line"),
                "clean_rendered_first_user_line": row.get("clean_rendered_first_user_line"),
                "corrupt_rendered_first_user_line": row.get("corrupt_rendered_first_user_line"),
            }
        )
    if not pairs:
        raise RuntimeError(f"No root pairs found under {dataset_root}")
    return pairs
