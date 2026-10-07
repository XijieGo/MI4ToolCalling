"""Fixed semantic rules for train-selected Transcoder feature families."""
from __future__ import annotations

from collections import Counter
from typing import Sequence


NO_NEED_KEYWORDS = {
    "no",
    "none",
    "nothing",
    "not",
    "never",
    "neither",
    "cannot",
    "can't",
    "nowhere",
    "nonexistent",
    "ineligible",
    "impossible",
    "rarely",
    "无需",
    "不需要",
    "不必",
    "不会",
    "不可能",
    "不存在",
    "没有",
    "没有任何",
    "没有人",
    "找不到",
    "无关",
    "不在",
    "根本没有",
}

EXECUTION_HINTS = {
    "write",
    "build",
    "add",
    "save",
    "implement",
    "create",
    "make",
    "complete",
    "function",
    "code",
    "solve",
}

DISCOURSE_KEYWORDS = {
    "narration",
    "paragraph",
    "paragraphs",
    "sentences",
    "written",
    "summary",
    "summaries",
    "past",
    "case",
    "cases",
    "discourse",
    "description",
    "descriptions",
    "叙述",
    "讲话",
    "科普",
    "文字",
    "语气",
    "发言",
    "一篇文章",
    "讲课",
    "谈话",
    "播报",
}

SCHEMA_FORMAT_KEYWORDS = {
    "####",
    "###",
    "**",
    "definitions",
    "definition",
    "concept",
    "concepts",
    "clarification",
    "schema",
    "json",
    "tool",
    "function",
    "functions",
    "arguments",
    "parameters",
}

ACTION_VERBS = {"add", "build", "complete", "create", "implement", "make", "process", "save", "write"}

ANALYSIS_VERBS = {"analyze", "analyse", "benchmark", "clarify", "detail", "discuss", "evaluate", "explore", "inspect", "parse", "review", "study"}

def normalize_token(token: str) -> str:
    return token.strip().lower()

def count_keyword_hits(tokens: Sequence[str], keywords: set[str]) -> int:
    merged = " ".join(normalize_token(token) for token in tokens)
    hits = 0
    for keyword in keywords:
        if keyword.lower() in merged:
            hits += 1
    return hits

def alignment_pattern(delta_activation: float, beta: float) -> str:
    if delta_activation >= 0.0 and beta >= 0.0:
        return "clean_higher_write_toward_gate"
    if delta_activation < 0.0 and beta < 0.0:
        return "corrupt_higher_write_away_from_gate"
    if delta_activation >= 0.0 and beta < 0.0:
        return "clean_higher_write_away_from_gate"
    return "corrupt_higher_write_toward_gate"

def build_common_verbs(example_payloads: Sequence[dict[str, object]]) -> tuple[str, ...]:
    counter: Counter[str] = Counter()
    for payload in example_payloads:
        verb = str(payload.get("verb") or "").lower().strip()
        if verb:
            counter[verb] += 1
    return tuple(verb for verb, _count in counter.most_common(3))

def score_family_membership(
    *,
    pattern: str,
    top_tokens: Sequence[str],
    bottom_tokens: Sequence[str],
    common_verbs: Sequence[str],
) -> dict[str, float]:
    top_hits_no_need = count_keyword_hits(top_tokens, NO_NEED_KEYWORDS)
    top_hits_discourse = count_keyword_hits(top_tokens + tuple(bottom_tokens), DISCOURSE_KEYWORDS)
    top_hits_schema = count_keyword_hits(top_tokens + tuple(bottom_tokens), SCHEMA_FORMAT_KEYWORDS)
    top_hits_exec = count_keyword_hits(top_tokens + tuple(bottom_tokens), EXECUTION_HINTS)
    action_hits = sum(verb in ACTION_VERBS for verb in common_verbs)
    analysis_hits = sum(verb in ANALYSIS_VERBS for verb in common_verbs)

    no_need = 2.0 * top_hits_no_need + 1.0 * analysis_hits
    if pattern == "corrupt_higher_write_away_from_gate":
        no_need += 2.0

    execution = 3.0 * action_hits + 1.0 * top_hits_exec
    if pattern == "clean_higher_write_toward_gate":
        execution += 2.0

    analysis = 2.0 * top_hits_discourse + 1.0 * analysis_hits
    if pattern in {"corrupt_higher_write_toward_gate", "clean_higher_write_away_from_gate"}:
        analysis += 1.0

    schema = 2.0 * top_hits_schema
    if pattern in {"corrupt_higher_write_away_from_gate", "clean_higher_write_toward_gate"}:
        schema += 0.5

    return {
        "no_need_non_existence": no_need,
        "execution_request": execution,
        "analysis_plain_discourse": analysis,
        "schema_boundary_formatting": schema,
    }

def family_threshold(family_name: str) -> float:
    if family_name == "execution_request":
        return 4.0
    if family_name == "schema_boundary_formatting":
        return 3.0
    return 4.0
