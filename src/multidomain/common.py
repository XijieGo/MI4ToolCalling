"""Shared definitions for the multi-domain paired-prompt construction.

The construction deliberately contains no LLM-based rewriting or judging.  A
source record is accepted through domain-specific field rules, then rendered
into clean/corrupt prompts that differ only in their leading user-turn verb.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator


TOOL_SCHEMAS = {
    "D3": (
        '{"type":"function","function":{"name":"web_search",'
        '"description":"Search web.","parameters":{"type":"object",'
        '"properties":{"query":{"type":"string"},"max_results":'
        '{"type":"integer"}},"required":["query"]}}}'
    ),
    "D4": (
        '{"type":"function","function":{"name":"run_sql",'
        '"description":"Run query.","parameters":{"type":"object",'
        '"properties":{"database":{"type":"string"},"query":'
        '{"type":"string"}},"required":["database","query"]}}}'
    ),
    "D5": (
        '{"type":"function","function":{"name":"send_email",'
        '"description":"Send email.","parameters":{"type":"object",'
        '"properties":{"to":{"type":"string"},"subject":'
        '{"type":"string"},"body":{"type":"string"}},'
        '"required":["to","subject","body"]}}}'
    ),
}


@dataclass(frozen=True)
class DomainConfig:
    name: str
    schema: str
    tail: str
    clean_verbs: tuple[str, ...]
    corrupt_verbs: tuple[str, ...]


DOMAINS = {
    "D3": DomainConfig(
        name="D3",
        schema=TOOL_SCHEMAS["D3"],
        # The body supplies a literal search-query string.  Calling it a
        # completed query, rather than merely a claim, makes the tool input
        # explicit for every Qwen3 scale while retaining a fixed object for
        # the no-call analysis request.
        tail="the completed search query shown below:",
        clean_verbs=("search",),
        corrupt_verbs=("describe", "review"),
    ),
    "D4": DomainConfig(
        name="D4",
        schema=TOOL_SCHEMAS["D4"],
        tail="the completed SQL query shown below:",
        clean_verbs=("run", "execute", "query", "fetch"),
        corrupt_verbs=("describe", "discuss", "review", "detail"),
    ),
    "D5": DomainConfig(
        name="D5",
        schema=TOOL_SCHEMAS["D5"],
        tail="the email message shown below to the recipient given:",
        clean_verbs=("send", "mail", "forward", "dispatch", "submit"),
        corrupt_verbs=("describe", "discuss", "review", "inspect"),
    ),
}


@dataclass(frozen=True)
class PromptTemplate:
    """The byte-level scaffold obtained from an existing Qwen v2 prompt."""

    prefix_before_schema: str
    after_schema_before_user: str
    assistant_suffix: str
    reference_path: str
    reference_sha256: str


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    domain: str
    source_id: str
    source_group: str
    source_metadata: dict[str, Any]
    body: str
    clean_verb: str
    corrupt_verb: str
    clean_prompt: str
    corrupt_prompt: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def stable_rank(seed: int, key: str) -> str:
    return sha256_text(f"{seed}:{key}")


def capitalize_verb(verb: str) -> str:
    normalized = verb.strip().lower()
    if not normalized or any(char.isspace() for char in normalized):
        raise ValueError(f"Expected a single non-empty verb, got {verb!r}")
    return normalized[:1].upper() + normalized[1:]


def build_template(reference_path: Path) -> PromptTemplate:
    text = reference_path.read_text(encoding="utf-8")
    tool_open = "<tools>\n"
    schema_start = text.find(tool_open)
    if schema_start < 0:
        raise ValueError(f"Could not find {tool_open!r} in {reference_path}")
    schema_start += len(tool_open)
    schema_end = text.find("</tools>", schema_start)
    if schema_end < 0:
        raise ValueError(f"Could not find </tools> in {reference_path}")
    # Keep the separator immediately before ``</tools>`` outside the schema
    # slot.  The reference scaffold uses a newline there; dropping it would
    # silently change the prompt tokens to ``...}}</tools>``.
    separator_start = schema_end
    if schema_end > schema_start and text[schema_end - 1] == "\n":
        separator_start -= 1
    post_schema = text[separator_start:]
    user_marker = "<|im_start|>user\n"
    user_start = post_schema.find(user_marker)
    if user_start < 0:
        raise ValueError(f"Could not find user turn in {reference_path}")
    user_content_start = user_start + len(user_marker)
    assistant_marker = "<|im_end|>\n<|im_start|>assistant\n"
    user_end = post_schema.find(assistant_marker, user_content_start)
    if user_end < 0:
        raise ValueError(f"Could not find assistant turn in {reference_path}")
    return PromptTemplate(
        prefix_before_schema=text[:schema_start],
        after_schema_before_user=post_schema[:user_content_start],
        assistant_suffix=post_schema[user_end:],
        reference_path=str(reference_path.resolve()),
        reference_sha256=sha256_text(text),
    )


def render_prompt(template: PromptTemplate, *, schema: str, verb: str, tail: str, body: str) -> str:
    rendered_line = f"{capitalize_verb(verb)} {tail}"
    return (
        template.prefix_before_schema
        + schema
        + template.after_schema_before_user
        + rendered_line
        + "\n"
        + body.rstrip()
        + "\n"
        + template.assistant_suffix
    )


def user_instruction_span(prompt: str) -> tuple[int, int, str]:
    marker = "<|im_start|>user\n"
    start = prompt.find(marker)
    if start < 0:
        raise ValueError("Prompt does not contain a user turn")
    start += len(marker)
    end = prompt.find("\n", start)
    if end < 0:
        raise ValueError("Prompt user instruction has no line ending")
    return start, end, prompt[start:end]


def assert_character_minimal_pair(clean_prompt: str, corrupt_prompt: str, clean_verb: str, corrupt_verb: str) -> None:
    """Require that the rendered prompts differ only in the first user word."""

    c_start, c_end, c_line = user_instruction_span(clean_prompt)
    r_start, r_end, r_line = user_instruction_span(corrupt_prompt)
    expected_clean = capitalize_verb(clean_verb)
    expected_corrupt = capitalize_verb(corrupt_verb)
    if not c_line.startswith(expected_clean + " "):
        raise ValueError(f"Unexpected clean user line: {c_line!r}")
    if not r_line.startswith(expected_corrupt + " "):
        raise ValueError(f"Unexpected corrupt user line: {r_line!r}")
    c_tail = c_line[len(expected_clean) :]
    r_tail = r_line[len(expected_corrupt) :]
    if c_tail != r_tail:
        raise ValueError("Clean/corrupt user instruction tails differ")
    rebuilt = clean_prompt[:c_start] + expected_corrupt + clean_prompt[c_start + len(expected_clean) :]
    if rebuilt != corrupt_prompt:
        raise ValueError("Clean/corrupt prompts differ outside the leading user verb")
    if c_end - c_start != len(c_line) or r_end - r_start != len(r_line):
        raise AssertionError("Invalid user-line span")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)
