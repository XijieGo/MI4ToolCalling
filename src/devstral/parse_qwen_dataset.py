#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "datasets"
DEFAULT_MODEL_PATH = Path(
    os.environ.get(
        "DEVSTRAL_2_24B_PATH",
        str(PROJECT_ROOT / "external" / "models" / "Devstral-Small-2-24B-Instruct-2512"),
    )
)
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "results" / "Devstral-Small-2-24B-Instruct-2512" / "converted_dataset"
DEFAULT_SYSTEM_PROMPT_FILE = PROJECT_ROOT / "configs" / "prompts" / "devstral_chat_system_prompt.txt"

QWEN_SYSTEM_RE = re.compile(r"<\|im_start\|>system\n(.*?)<\|im_end\|>", re.DOTALL)
QWEN_USER_RE = re.compile(r"<\|im_start\|>user\n(.*?)<\|im_end\|>", re.DOTALL)
TOOLS_RE = re.compile(r"<tools>\s*(.*?)\s*</tools>", re.DOTALL)


@dataclass(frozen=True)
class CanonicalPair:
    sample_id: str
    split: str
    language: str
    clean_candidate: str
    corrupt_candidate: str
    clean_user_content: str
    corrupt_user_content: str
    tools_schema: list[dict[str, Any]]
    clean_source_path: str
    corrupt_source_path: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert Qwen3 tool-call dataset into Devstral-native prompts.")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--system-prompt-file",
        type=Path,
        default=DEFAULT_SYSTEM_PROMPT_FILE,
        help="Versioned Devstral chat-system template; date placeholders are resolved with --today.",
    )
    parser.add_argument("--today", type=str, default="")
    return parser.parse_args()


def load_sample_ids(path: Path) -> list[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not all(isinstance(item, str) for item in data):
        raise ValueError(f"Unexpected sample id format in {path}")
    return data


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


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


def resolve_system_prompt(path: Path, today_override: str = "") -> tuple[str, str, str]:
    today_value = date.fromisoformat(today_override) if today_override else date.today()
    yesterday_value = today_value - timedelta(days=1)
    template = path.read_text(encoding="utf-8")
    resolved = template.replace("{today}", today_value.isoformat()).replace(
        "{yesterday}",
        yesterday_value.isoformat(),
    )
    return resolved, today_value.isoformat(), yesterday_value.isoformat()


def infer_language(sample_id: str) -> str:
    parts = sample_id.split("_")
    if len(parts) < 3:
        raise ValueError(f"Unexpected sample id format: {sample_id}")
    return parts[1]


def infer_candidate(user_text: str) -> str:
    first_line = user_text.splitlines()[0].strip()
    if not first_line:
        raise ValueError("User content is empty.")
    return first_line.split()[0].lower()


def build_pairs_from_v2_layout(dataset_root: Path) -> list[CanonicalPair]:
    """Load the canonical ``datasets/{train,test}`` layout directly."""

    pairs: list[CanonicalPair] = []
    for split in ("train", "test"):
        manifest_path = dataset_root / split / "clean" / "manifest.jsonl"
        if not manifest_path.exists():
            raise FileNotFoundError(manifest_path)
        for row in read_jsonl(manifest_path):
            filename = str(row["output_filename"])
            clean_path = dataset_root / split / "clean" / filename
            corrupt_path = dataset_root / split / "corrupt" / filename
            if not clean_path.exists() or not corrupt_path.exists():
                raise FileNotFoundError(f"Missing clean/corrupt files for {split}/{filename}")
            clean_tools, clean_user = parse_qwen_prompt(clean_path)
            corrupt_tools, corrupt_user = parse_qwen_prompt(corrupt_path)
            if json.dumps(clean_tools, ensure_ascii=False, sort_keys=True) != json.dumps(
                corrupt_tools, ensure_ascii=False, sort_keys=True
            ):
                raise ValueError(f"Tool schema mismatch between clean and corrupt for {filename}")
            pairs.append(
                CanonicalPair(
                    sample_id=Path(filename).stem,
                    split=split,
                    language=str(row.get("language", "unknown")),
                    clean_candidate=str(row.get("clean_candidate") or infer_candidate(clean_user)),
                    corrupt_candidate=str(row.get("corrupt_candidate") or infer_candidate(corrupt_user)),
                    clean_user_content=clean_user,
                    corrupt_user_content=corrupt_user,
                    tools_schema=clean_tools,
                    clean_source_path=str(clean_path.resolve()),
                    corrupt_source_path=str(corrupt_path.resolve()),
                )
            )
    return pairs


def build_pairs_from_legacy_layout(dataset_root: Path) -> list[CanonicalPair]:
    train_ids = load_sample_ids(dataset_root / "aligned_shared_qwen3" / "train" / "sample_ids.json")
    test_ids = load_sample_ids(dataset_root / "aligned_shared_qwen3" / "test" / "sample_ids.json")
    split_by_id = {sample_id: "train" for sample_id in train_ids}
    overlap = set(split_by_id).intersection(test_ids)
    if overlap:
        raise ValueError(f"Sample ids appear in both train and test: {sorted(overlap)[:5]}")
    split_by_id.update({sample_id: "test" for sample_id in test_ids})

    pairs: list[CanonicalPair] = []
    for sample_id in sorted(split_by_id):
        clean_path = dataset_root / "clean" / f"{sample_id}.txt"
        corrupt_path = dataset_root / "corrupt" / f"{sample_id}.txt"
        if not clean_path.exists() or not corrupt_path.exists():
            raise FileNotFoundError(f"Missing clean/corrupt files for {sample_id}")

        clean_tools, clean_user = parse_qwen_prompt(clean_path)
        corrupt_tools, corrupt_user = parse_qwen_prompt(corrupt_path)
        clean_tools_json = json.dumps(clean_tools, ensure_ascii=False, sort_keys=True)
        corrupt_tools_json = json.dumps(corrupt_tools, ensure_ascii=False, sort_keys=True)
        if clean_tools_json != corrupt_tools_json:
            raise ValueError(f"Tool schema mismatch between clean and corrupt for {sample_id}")

        pairs.append(
            CanonicalPair(
                sample_id=sample_id,
                split=split_by_id[sample_id],
                language=infer_language(sample_id),
                clean_candidate=infer_candidate(clean_user),
                corrupt_candidate=infer_candidate(corrupt_user),
                clean_user_content=clean_user,
                corrupt_user_content=corrupt_user,
                tools_schema=clean_tools,
                clean_source_path=str(clean_path.resolve()),
                corrupt_source_path=str(corrupt_path.resolve()),
            )
        )
    return pairs


def build_pairs(dataset_root: Path) -> list[CanonicalPair]:
    """Load v2 data by default while retaining the historical Devstral layout."""

    if (dataset_root / "train" / "clean" / "manifest.jsonl").exists():
        return build_pairs_from_v2_layout(dataset_root)
    return build_pairs_from_legacy_layout(dataset_root)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    clean_output_root = args.output_root / "clean"
    corrupt_output_root = args.output_root / "corrupt"
    clean_output_root.mkdir(parents=True, exist_ok=True)
    corrupt_output_root.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(str(args.model_path), trust_remote_code=True)
    system_prompt_path = args.system_prompt_file.expanduser().resolve()
    system_prompt, today_value, yesterday_value = resolve_system_prompt(
        system_prompt_path,
        args.today,
    )
    pairs = build_pairs(args.dataset_root)

    canonical_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    for pair in pairs:
        clean_prompt = tokenizer.apply_chat_template(
            conversation=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": pair.clean_user_content},
            ],
            tools=pair.tools_schema,
            tokenize=False,
            add_generation_prompt=True,
        )
        corrupt_prompt = tokenizer.apply_chat_template(
            conversation=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": pair.corrupt_user_content},
            ],
            tools=pair.tools_schema,
            tokenize=False,
            add_generation_prompt=True,
        )
        clean_output_path = clean_output_root / f"{pair.sample_id}.txt"
        corrupt_output_path = corrupt_output_root / f"{pair.sample_id}.txt"
        clean_output_path.write_text(clean_prompt, encoding="utf-8")
        corrupt_output_path.write_text(corrupt_prompt, encoding="utf-8")

        canonical_rows.append(
            {
                "sample_id": pair.sample_id,
                "split": pair.split,
                "language": pair.language,
                "clean_candidate": pair.clean_candidate,
                "corrupt_candidate": pair.corrupt_candidate,
                "clean_user_content": pair.clean_user_content,
                "corrupt_user_content": pair.corrupt_user_content,
                "tools_schema": pair.tools_schema,
                "clean_source_path": pair.clean_source_path,
                "corrupt_source_path": pair.corrupt_source_path,
            }
        )
        manifest_rows.append(
            {
                "sample_id": pair.sample_id,
                "split": pair.split,
                "language": pair.language,
                "clean_candidate": pair.clean_candidate,
                "corrupt_candidate": pair.corrupt_candidate,
                "system_prompt_path": str(system_prompt_path.resolve()),
                "system_prompt_file": str(system_prompt_path),
                "system_prompt_today": today_value,
                "system_prompt_yesterday": yesterday_value,
                "clean_source_path": pair.clean_source_path,
                "corrupt_source_path": pair.corrupt_source_path,
                "clean_prompt_path": str(clean_output_path.resolve()),
                "corrupt_prompt_path": str(corrupt_output_path.resolve()),
                "add_generation_prompt": True,
            }
        )

    write_jsonl(args.output_root / "canonical_pairs.jsonl", canonical_rows)
    write_jsonl(args.output_root / "manifest.jsonl", manifest_rows)

    summary = {
        "n_pairs": len(pairs),
        "splits": {
            "train": sum(1 for pair in pairs if pair.split == "train"),
            "test": sum(1 for pair in pairs if pair.split == "test"),
        },
        "languages": {
            language: sum(1 for pair in pairs if pair.language == language)
            for language in sorted({pair.language for pair in pairs})
        },
        "clean_candidates": {
            candidate: sum(1 for pair in pairs if pair.clean_candidate == candidate)
            for candidate in sorted({pair.clean_candidate for pair in pairs})
        },
        "corrupt_candidates": {
            candidate: sum(1 for pair in pairs if pair.corrupt_candidate == candidate)
            for candidate in sorted({pair.corrupt_candidate for pair in pairs})
        },
        "model_path": str(args.model_path.resolve()),
        "output_root": str(args.output_root.resolve()),
        "system_prompt_file": str(system_prompt_path),
        "system_prompt_today": today_value,
        "system_prompt_yesterday": yesterday_value,
    }
    (args.output_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
