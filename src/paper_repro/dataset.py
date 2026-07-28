"""Version-aware dataset utilities used by the paper reproduction entry points."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Iterable, Iterator


USER_MARKER = "<|im_start|>user\n"
ASSISTANT_MARKER = "<|im_end|>\n<|im_start|>assistant\n"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Iterable[dict[str, object]], fieldnames: list[str] | None = None) -> None:
    materialized = list(rows)
    if fieldnames is None:
        fieldnames = list(materialized[0]) if materialized else []
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(materialized)


def iter_jsonl(path: Path) -> Iterator[dict[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def user_instruction_line(prompt: str) -> tuple[int, int, str]:
    """Return the character span and contents of the first user-instruction line.

    The archived Qwen prompts have a stable system/tool/user/assistant layout.
    Replacing the first user-line token, rather than using a character suffix
    heuristic, is important: the latter once produced corrupt strings such as
    ``Save e the function ...`` for ``Save``/``Explore`` pairs.
    """

    marker_start = prompt.find(USER_MARKER)
    if marker_start < 0:
        raise ValueError("Prompt does not contain the Qwen user marker")
    content_start = marker_start + len(USER_MARKER)
    content_end = prompt.find(ASSISTANT_MARKER, content_start)
    if content_end < 0:
        raise ValueError("Prompt does not contain the Qwen assistant marker")
    line_end = prompt.find("\n", content_start, content_end)
    if line_end < 0:
        line_end = content_end
    return content_start, line_end, prompt[content_start:line_end]


def replace_instruction_verb(prompt: str, verb: str) -> tuple[str, str]:
    """Replace only the leading word of the first user instruction line.

    Returns the complete prompt and the rendered first line.  This reproduces
    all 1,500 frozen v2 clean/corrupt prompts byte-for-byte from the v1 base
    prompt archive when combined with the v2 selection manifest.
    """

    start, end, line = user_instruction_line(prompt)
    if not line or line[0].isspace():
        raise ValueError(f"Unexpected empty/indented instruction line: {line!r}")
    separator = len(line)
    for index, char in enumerate(line):
        if char.isspace():
            separator = index
            break
    tail = line[separator:]
    normalized = str(verb).strip().lower()
    if not normalized:
        raise ValueError("Verb must be non-empty")
    rendered_verb = normalized[:1].upper() + normalized[1:]
    rendered_line = rendered_verb + tail
    return prompt[:start] + rendered_line + prompt[end:], rendered_line


def dataset_file_map(root: Path, *, include_manifests: bool = False) -> dict[str, Path]:
    """Map a dataset tree to relative paths, excluding generated reports by default."""

    result: dict[str, Path] = {}
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        # Provenance is deliberately adjacent to the active dataset, but it is
        # not part of the active v2 tree that a rebuild should compare against.
        if relative.parts and relative.parts[0] == "provenance":
            continue
        if not include_manifests and path.name in {"manifest.jsonl", "rebuild_report.json"}:
            continue
        result[str(relative)] = path
    return result
