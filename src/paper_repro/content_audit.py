"""Deterministic content checks for the archived code-task prompts.

The historical ``v1_1711`` pool was behavior-screened, but it did not have a
separate source-text audit.  This module deliberately does *not* call a model
or attempt to decide whether a programming problem is factually correct.  It
checks only properties that can be established from the prompt bytes:

* the Qwen scaffold and the first user instruction are well formed;
* the user already contains a complete code object (signature, description,
  and an implementation placeholder);
* prompt-control/injection strings, credentials, and obvious direct contact
  data are absent from the user payload; and
* length and character checks do not indicate truncation or corruption.

The result is an audit trail, rather than an automatic rewrite.  A rejected
row is kept in the report with stable reason codes so that the 500-row
selection can be reproduced and reviewed.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from .dataset import ASSISTANT_MARKER, USER_MARKER, user_instruction_line


WORD_RE = re.compile(r"\b[\w'-]+\b", flags=re.UNICODE)

# The v1 prompt scaffold has one fixed user-line template per language.  The
# verb is intentionally not fixed here: it is replaced later by the behavior
# screen's selected clean/corrupt candidate.
CODE_INSTRUCTION_RE = re.compile(
    r"^(?:Write|Save|Add|Complete|Build) the function body in "
    r"solve\.(?:py|cpp|java) based on the function definition and docstring below:$"
)

PROMPT_CONTROL_RE = re.compile(
    r"<\|im_start\|>|<\|im_end\|>|<tool_call>|</?tools>|<\|endoftext\|>",
    flags=re.IGNORECASE,
)

# Inline Markdown code spans (for example, ``at least `digit` digits``) are
# ordinary benchmark prose.  A backtick inserted *inside* a word, however, is
# a known corruption in one archived instruction (``docstrin`g``).
WORD_SPLIT_BACKTICK_RE = re.compile(r"\w`\w")

# These patterns are deliberately contextual.  A coding problem may mention
# an innocuous variable named ``password`` or an algorithm called
# ``minMalwareSpread``; neither is evidence of a compliance problem.  We only
# reject strings that look like an instruction override, a literal secret, or
# direct identifying/contact data.
PROMPT_INJECTION_RE = re.compile(
    r"(?:"
    r"ignore\s+(?:all\s+)?(?:the\s+)?(?:previous|prior|above)\s+instructions?"
    r"|system\s+prompt"
    r"|developer\s+(?:message|instruction)"
    r"|(?:reveal|print|show)\s+(?:the\s+)?(?:hidden|system)\s+prompt"
    r"|jailbreak"
    r"|do\s+not\s+follow\s+(?:the\s+)?(?:above|previous)"
    r")",
    flags=re.IGNORECASE,
)

EMAIL_RE = re.compile(r"(?<![\w.+-])[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}(?![\w.-])")
PRIVATE_KEY_RE = re.compile(r"-----BEGIN(?: [A-Z]+)* PRIVATE KEY-----", flags=re.IGNORECASE)
LITERAL_SECRET_RE = re.compile(
    r"(?:"
    r"\b(?:api|secret|access)[_ -]?(?:key|token)\b\s*[:=]\s*['\"]?[A-Za-z0-9_./+=-]{12,}"
    r"|\bauthorization\s*[:=]\s*['\"]?bearer\s+[A-Za-z0-9._~+/=-]{12,}"
    r"|\bsk-[A-Za-z0-9]{16,}\b"
    r")",
    flags=re.IGNORECASE,
)

# Only formatted numbers with an explicit phone/contact context are treated as
# direct contact data.  Long digit strings in CodeContests examples are test
# inputs, not personal phone numbers, and are intentionally retained.
FORMATTED_PHONE_RE = re.compile(
    r"\b(?:phone|telephone|mobile|cell)\s*(?:number|no\.?|#)?\s*(?:is|:|=)?\s*"
    r"(?:\+\d{1,3}[\s.-]?)?(?:\(\d{2,4}\)[\s.-]?)?\d{3}[\s.-]\d{3,4}[\s.-]\d{4}\b",
    flags=re.IGNORECASE,
)
SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")

EXPLICIT_HARM_RE = re.compile(
    r"(?:"
    r"\b(?:ransomware|keylogger|botnet)\b"
    r"|\breverse\s+shell\b"
    r"|\b(?:steal|stealing|harvest|harvesting)\s+(?:user\s+)?passwords?\b"
    r"|\bbypass(?:ing)?\s+(?:authentication|security)\b"
    r"|\bddos\s+attack\b"
    r")",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True)
class ContentAudit:
    """The deterministic audit result for one source scaffold."""

    passed: bool
    reasons: tuple[str, ...]
    user_line: str
    body_word_count: int
    body_char_count: int
    language: str

    def as_dict(self, *, filename: str, sample_id: str, dataset_name: str) -> dict[str, object]:
        return {
            "filename": filename,
            "sample_id": sample_id,
            "language": self.language,
            "dataset_name": dataset_name,
            "content_audit_passed": self.passed,
            "content_audit_reasons": "|".join(self.reasons),
            "user_instruction": self.user_line,
            "body_word_count": self.body_word_count,
            "body_char_count": self.body_char_count,
        }


def extract_user_payload(prompt: str) -> tuple[str, str]:
    """Return the first user instruction line and its body from a scaffold."""

    start, end, line = user_instruction_line(prompt)
    assistant_start = prompt.find(ASSISTANT_MARKER, end)
    if assistant_start < 0:
        raise ValueError("Prompt does not contain the Qwen assistant marker")
    body_start = end + 1 if end < len(prompt) and prompt[end] == "\n" else end
    body = prompt[body_start:assistant_start]
    return line, body.rstrip("\n")


def _language_from_line(user_line: str) -> str:
    match = re.search(r"solve\.(py|cpp|java)\b", user_line)
    return match.group(1) if match else "unknown"


def _has_code_signature(body: str, language: str) -> bool:
    if language == "py":
        return bool(re.search(r"\b(?:async\s+)?def\s+[A-Za-z_]\w*\s*\(", body))
    # Both C++ and Java adapters expose a solve(...) function.  Requiring the
    # function name avoids accepting a truncated comment-only task body.
    return bool(re.search(r"\bsolve\s*\(", body))


def _has_description(body: str, language: str) -> bool:
    if language == "py":
        return '"""' in body or "'''" in body
    return "/*" in body and "*/" in body


def _has_placeholder(body: str, language: str) -> bool:
    if language == "py":
        return bool(re.search(r"\bpass\b", body))
    return bool(re.search(r"\bTODO\b", body, flags=re.IGNORECASE)) and bool(
        re.search(r"\breturn\s+['\"]{2}\s*;", body)
    )


def _has_invisible_corruption(value: str) -> bool:
    for character in value:
        category = unicodedata.category(character)
        # Newline/tab are valid prompt formatting; other Cc characters and
        # zero-width formatting characters are not useful in a source task.
        if category == "Cc" and character not in {"\n", "\r", "\t"}:
            return True
        if category == "Cf":
            return True
    return False


def audit_code_scaffold(prompt: str, *, min_words: int = 12, max_words: int = 130) -> ContentAudit:
    """Audit one archived code prompt without model or external services."""

    reasons: list[str] = []
    user_line = ""
    body = ""
    language = "unknown"
    try:
        user_line, body = extract_user_payload(prompt)
        language = _language_from_line(user_line)
    except (ValueError, IndexError):
        reasons.append("malformed_turn_structure")

    # Exact counts protect against nested prompt content and accidental
    # concatenation of two examples.
    if prompt.count(USER_MARKER) != 1 or prompt.count(ASSISTANT_MARKER) != 1:
        reasons.append("unexpected_turn_count")
    # The system prose itself contains the literal phrase ``<tools></tools>``;
    # count the line-delimited XML tags rather than the prose mention.
    if prompt.count("<tools>\n") != 1 or prompt.count("\n</tools>\n") != 1:
        reasons.append("malformed_tool_schema_scaffold")

    if not CODE_INSTRUCTION_RE.fullmatch(user_line):
        reasons.append("malformed_instruction_line")
    if language == "unknown":
        reasons.append("unknown_code_language")

    body_word_count = len(WORD_RE.findall(body))
    body_char_count = len(body)
    if not min_words <= body_word_count <= max_words:
        reasons.append("body_word_count_out_of_range")
    if not body.strip():
        reasons.append("empty_body")
    if not _has_code_signature(body, language):
        reasons.append("missing_function_signature")
    if not _has_description(body, language):
        reasons.append("missing_problem_description")
    if not _has_placeholder(body, language):
        reasons.append("missing_implementation_placeholder")

    payload = f"{user_line}\n{body}"
    if WORD_SPLIT_BACKTICK_RE.search(payload):
        reasons.append("stray_backtick_or_markup")
    if PROMPT_CONTROL_RE.search(payload):
        reasons.append("prompt_control_marker_in_user_payload")
    if PROMPT_INJECTION_RE.search(payload):
        reasons.append("prompt_injection_text")
    if EMAIL_RE.search(payload):
        reasons.append("literal_email_address")
    if PRIVATE_KEY_RE.search(payload) or LITERAL_SECRET_RE.search(payload):
        reasons.append("literal_credential_or_secret")
    if FORMATTED_PHONE_RE.search(payload) or SSN_RE.search(payload):
        reasons.append("direct_contact_or_identity_number")
    if EXPLICIT_HARM_RE.search(payload):
        reasons.append("explicit_high_risk_instruction")
    if _has_invisible_corruption(payload):
        reasons.append("invisible_control_character")

    # Preserve first-occurrence order for stable CSV reports while avoiding
    # duplicate reasons if a future check is extended.
    unique_reasons = tuple(dict.fromkeys(reasons))
    return ContentAudit(
        passed=not unique_reasons,
        reasons=unique_reasons,
        user_line=user_line,
        body_word_count=body_word_count,
        body_char_count=body_char_count,
        language=language,
    )
