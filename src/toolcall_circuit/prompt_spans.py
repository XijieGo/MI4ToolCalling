#!/usr/bin/env python3
from __future__ import annotations

import re
from collections import OrderedDict
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple


SPAN_NAMES: Tuple[str, ...] = (
    "system_preamble",
    "tools_block",
    "tool_instruction",
    "tool_call_example",
    "user_lead_phrase",
    "function_body_anchor",
    "file_target",
    "instruction_suffix",
    "task_body",
    "assistant_prefix",
)

SPAN_LABELS: Mapping[str, str] = OrderedDict(
    [
        ("system_preamble", "System"),
        ("tools_block", "Tools"),
        ("tool_instruction", "Tool Instr."),
        ("tool_call_example", "Tool Example"),
        ("user_lead_phrase", "Lead Phrase"),
        ("function_body_anchor", "Function Body"),
        ("file_target", "Filename"),
        ("instruction_suffix", "Request Suffix"),
        ("task_body", "Task Description"),
        ("assistant_prefix", "Assistant Prefix"),
    ]
)

SYSTEM_TOOLS_OPEN = "<tools>"
SYSTEM_TOOLS_CLOSE = "</tools>"
SYSTEM_TOOL_CALL_OPEN = "<tool_call>"
USER_MARKER = "<|im_start|>user\n"
ASSISTANT_MARKER = "<|im_end|>\n<|im_start|>assistant\n"
FILE_TARGET_RE = re.compile(r"solve\.(?:py|cpp|java)")


def find_required_substring(text: str, needle: str, start: int = 0, *, use_last: bool = False) -> int:
    idx = text.rfind(needle, start) if use_last else text.find(needle, start)
    if idx < 0:
        raise ValueError(f"Failed to find required substring {needle!r}")
    return idx


def validate_char_spans(spans: Mapping[str, Tuple[int, int]], text_len: int) -> None:
    start = 0
    for name in SPAN_NAMES:
        lo, hi = spans[name]
        if lo != start:
            raise ValueError(f"Span {name} starts at {lo}, expected {start}")
        if hi <= lo:
            raise ValueError(f"Span {name} is empty or negative: {(lo, hi)}")
        start = hi
    if start != text_len:
        raise ValueError(f"Char spans end at {start}, expected {text_len}")


def build_char_spans(text: str) -> Dict[str, Tuple[int, int]]:
    user_marker_idx = text.find(USER_MARKER)
    if user_marker_idx < 0:
        raise ValueError("Failed to find user marker")
    user_content_start = user_marker_idx + len(USER_MARKER)
    assistant_marker_idx = text.find(ASSISTANT_MARKER, user_content_start)
    if assistant_marker_idx < 0:
        raise ValueError("Failed to find assistant marker")

    tools_open_idx = find_required_substring(text[:user_content_start], SYSTEM_TOOLS_OPEN, use_last=True)
    tools_close_idx = find_required_substring(text[:user_content_start], SYSTEM_TOOLS_CLOSE, tools_open_idx)
    tools_close_end = tools_close_idx + len(SYSTEM_TOOLS_CLOSE)
    tool_example_open_idx = find_required_substring(text[:user_content_start], SYSTEM_TOOL_CALL_OPEN, use_last=True)

    user_content = text[user_content_start:assistant_marker_idx]
    first_line_end_rel = user_content.find("\n")
    if first_line_end_rel < 0:
        first_line_end_rel = len(user_content)
    first_line = user_content[:first_line_end_rel]
    line_start_abs = user_content_start
    line_end_abs = user_content_start + first_line_end_rel

    file_match = FILE_TARGET_RE.search(first_line)
    if file_match is None:
        raise ValueError("Failed to find file target in user instruction line")
    file_start_abs = line_start_abs + file_match.start()
    file_end_abs = line_start_abs + file_match.end()

    anchor_rel = first_line.find("the function body")
    if anchor_rel < 0:
        anchor_rel = first_line.find("function body")
    if anchor_rel < 0:
        raise ValueError("Failed to find function body anchor in user instruction line")
    anchor_start_abs = line_start_abs + anchor_rel

    spans = {
        "system_preamble": (0, tools_open_idx),
        "tools_block": (tools_open_idx, tools_close_end),
        "tool_instruction": (tools_close_end, tool_example_open_idx),
        "tool_call_example": (tool_example_open_idx, user_content_start),
        "user_lead_phrase": (user_content_start, anchor_start_abs),
        "function_body_anchor": (anchor_start_abs, file_start_abs),
        "file_target": (file_start_abs, file_end_abs),
        "instruction_suffix": (file_end_abs, line_end_abs),
        "task_body": (line_end_abs, assistant_marker_idx),
        "assistant_prefix": (assistant_marker_idx, len(text)),
    }
    validate_char_spans(spans, len(text))
    return spans


def token_span_positions(text: str, tokenizer) -> Dict[str, List[int]]:
    spans = build_char_spans(text)
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    offsets = [(int(start), int(end)) for start, end in encoded["offset_mapping"]]
    positions = {name: [] for name in SPAN_NAMES}

    for tok_idx, (tok_start, tok_end) in enumerate(offsets):
        probe = float(tok_start) if tok_end <= tok_start else tok_start + 0.5 * float(tok_end - tok_start)
        assigned = None
        for name in SPAN_NAMES:
            lo, hi = spans[name]
            if lo <= probe < hi:
                assigned = name
                break
        if assigned is None and tok_end == len(text):
            assigned = SPAN_NAMES[-1]
        if assigned is None:
            raise ValueError(f"Failed to assign token {tok_idx} offset={(tok_start, tok_end)}")
        positions[assigned].append(tok_idx)

    total_positions = sum(len(pos) for pos in positions.values())
    if total_positions != len(offsets):
        raise ValueError(f"Assigned {total_positions} tokens, expected {len(offsets)}")
    for name in SPAN_NAMES:
        if not positions[name]:
            raise ValueError(f"Span {name} has zero tokens after token assignment")
    return positions


def merge_span_positions(
    span_positions: Mapping[str, Sequence[int]],
    groups: Mapping[str, Iterable[str]],
) -> Dict[str, List[int]]:
    merged: Dict[str, List[int]] = {}
    for group_name, group_spans in groups.items():
        positions: List[int] = []
        for span_name in group_spans:
            positions.extend(int(pos) for pos in span_positions[span_name])
        uniq = sorted(set(positions))
        if not uniq:
            raise ValueError(f"Span group {group_name} has zero tokens")
        merged[group_name] = uniq
    return merged
