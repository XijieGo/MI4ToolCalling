"""Scaffold variants that preserve native token IDs and multi-turn history."""
from __future__ import annotations

import copy
import difflib
import re
from typing import Any

NEUTRAL = ("Consider", "Handle", "Take", "Use", "Process")
CONDITIONS = {
    "RTF": (True, True, True, False),
    "RT-": (True, True, False, False),
    "R-F": (True, False, True, False),
    "--F": (False, False, True, False),
    "R_TLEN_F": (True, False, True, True),
    "R--": (True, False, False, False),
    "-T-": (False, True, False, False),
    "---": (False, False, False, False),
}


def neutral_text(clean: str, corrupt: str, index: int) -> str:
    # Replace the changed action/analysis verb, including changes in a final
    # user turn of a long tool conversation. Never replace an earlier turn.
    edits = [(i, j) for tag, i, j, _, _ in difflib.SequenceMatcher(None, clean, corrupt, autojunk=False).get_opcodes() if tag != "equal"]
    if not edits:
        raise ValueError("Pair contains no instruction difference")
    start, end = edits[0][0], edits[-1][1]
    while start and clean[start - 1].isalpha():
        start -= 1
    while end < len(clean) and clean[end].isalpha():
        end += 1
    if end - start > 120:
        raise ValueError(f"Unexpected instruction change: {clean[start:end]!r}")
    verb = NEUTRAL[index % len(NEUTRAL)]
    if clean[start:start+1].islower():
        verb = verb.lower()
    return clean[:start] + verb + clean[end:]


def system_regions(model: str, text: str) -> list[tuple[int, int, str]]:
    """Return disjoint R/T/F character regions; all other history is untouched."""
    if model.startswith("qwen"):
        open_tag = "<|im_start|>system\n"
        if not text.startswith(open_tag):
            raise ValueError("Missing native Qwen system turn")
        start, end = len(open_tag), text.index("<|im_end|>")
        content = text[start:end]
        t0 = content.find("# Tools")
        if t0 < 0:
            t0 = content.index("<tools>")
        t1 = content.index("</tools>", t0) + len("</tools>")
        f0 = content.find("If you choose to call", t1)
        if f0 < 0:
            f0 = content.index("For each function call", t1)
        important = content.find("</IMPORTANT>", f0)
        f1 = important + len("</IMPORTANT>") if important >= 0 else len(content)
        regions = [(start + t0, start + t1, "T"), (start + f0, start + f1, "F")]
        cursor = start
        for a, b, _ in sorted(regions):
            if cursor < a:
                regions.append((cursor, a, "R"))
            cursor = b
        if cursor < end:
            regions.append((cursor, end, "R"))
        return sorted(regions)
    if model.startswith("granite"):
        end_tag = "<|end_of_text|>"
        role_tag = "<|start_of_role|>system<|end_of_role|>"
        tools_tag = "<|start_of_role|>available_tools<|end_of_role|>"
        r0 = text.index(role_tag)
        r1 = text.index(end_tag, r0) + len(end_tag)
        t0 = text.index(tools_tag)
        t1 = text.index(end_tag, t0) + len(end_tag)
        # Granite's call format instructions are part of its role message;
        # it has no separate format-template component to ablate.
        return [(r0, r1, "R"), (t0, t1, "T")]
    raise ValueError(model)


def filler_ids(tokenizer: Any, count: int) -> list[int]:
    base = tokenizer.encode("Background context is provided for reference. General information is available. ", add_special_tokens=False)
    if not base:
        raise ValueError("Empty neutral filler")
    return (base * ((count + len(base) - 1) // len(base)))[:count]


def token_regions(model: str, tokenizer: Any, ids: list[int], text: str | None) -> dict[str, list[int]]:
    """Partition tokens into scaffold components, final user request, history."""
    result = {k: [] for k in ("R", "T", "F", "U", "history")}
    if model.startswith("mistral"):
        text, offsets = tokenizer.token_offsets(ids)
        tags = (("[SYSTEM_PROMPT]", "[/SYSTEM_PROMPT]", "R"), ("[AVAILABLE_TOOLS]", "[/AVAILABLE_TOOLS]", "T"))
        regions = []
        for left, right, key in tags:
            a = text.find(left)
            if a >= 0:
                b = text.index(right, a) + len(right)
                regions.append((a, b, key))
        u0 = text.rfind("[INST]")
        u1 = text.find("[/INST]", u0)
        if u0 >= 0 and u1 >= 0:
            regions.append((u0, u1 + len("[/INST]"), "U"))
    else:
        assert text is not None
        encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        if list(encoded["input_ids"]) != ids:
            raise ValueError("Offset tokenization changed the native prompt")
        offsets = encoded["offset_mapping"]
        regions = system_regions(model, text)
        tag = "<|im_start|>user\n" if model.startswith("qwen") else "<|start_of_role|>user<|end_of_role|>"
        end_tag = "<|im_end|>" if model.startswith("qwen") else "<|end_of_text|>"
        u0 = text.rfind(tag)
        u1 = text.find(end_tag, u0)
        if u0 >= 0 and u1 >= 0:
            regions.append((u0, u1 + len(end_tag), "U"))
    for i, (a, b) in enumerate(offsets):
        key = next((k for start, end, k in regions if a < end and b > start), "history")
        result[key].append(i)
    return result


def render_text_variant(model: str, tokenizer: Any, text: str, condition: str) -> list[int]:
    include_r, include_t, include_f, length_match = CONDITIONS[condition]
    if condition == "RTF":
        return tokenizer.encode(text, add_special_tokens=False)
    regions = system_regions(model, text)
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    if length_match:
        # Replace the exact schema-token interval, retaining its exact token
        # count. Whitespace character length is not a token-length control.
        span = next((a, b) for a, b, k in regions if k == "T")
        if model.startswith("granite"):
            span = (span[0] + len("<|start_of_role|>available_tools<|end_of_role|>"),
                    span[1] - len("<|end_of_text|>"))
        indices = [i for i, (a, b) in enumerate(encoded["offset_mapping"]) if a < span[1] and b > span[0]]
        if not indices:
            raise ValueError("No tool schema tokens")
        a, b = min(indices), max(indices) + 1
        return encoded["input_ids"][:a] + filler_ids(tokenizer, b-a) + encoded["input_ids"][b:]
    keep = {"R": include_r, "T": include_t, "F": include_f}
    if model.startswith("qwen") and not any(keep.values()):
        end = text.index("<|im_end|>") + len("<|im_end|>\n")
        return tokenizer.encode(text[end:], add_special_tokens=False)
    # Delete ranges in reverse order so the original R/T/F ordering survives.
    rendered = text
    for a, b, key in sorted(regions, reverse=True):
        if not keep[key]:
            rendered = rendered[:a] + rendered[b:]
    return tokenizer.encode(rendered, add_special_tokens=False)


def native_variant(tokenizer: Any, payload: dict[str, Any], condition: str, neutral: str | None = None) -> list[int]:
    include_r, include_t, _, length_match = CONDITIONS[condition]
    messages = copy.deepcopy(payload["messages"])
    if neutral is not None:
        index = max(i for i, msg in enumerate(messages) if msg["role"] == "user")
        messages[index]["content"] = neutral
    original = tokenizer.apply_chat_template(messages, tools=payload["tools"], tokenize=True)
    if neutral is None and original != payload["input_ids"]:
        raise ValueError("Native Mistral rendering does not reproduce the stored IDs")
    if length_match:
        left = tokenizer.convert_tokens_to_ids("[AVAILABLE_TOOLS]")
        right = tokenizer.convert_tokens_to_ids("[/AVAILABLE_TOOLS]")
        a, b = original.index(left) + 1, original.index(right)
        return original[:a] + filler_ids(tokenizer, b-a) + original[b:]
    if not include_r:
        messages = [msg for msg in messages if msg["role"] != "system"]
    return tokenizer.apply_chat_template(messages, tools=payload["tools"] if include_t else None, tokenize=True)
