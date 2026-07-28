#!/usr/bin/env python3
"""Extract natural τ² Telecom prefixes before an agent's *text-only* reply.

This is the converse data construction to ``build_tau2_telecom_prefixes.py``.
Each candidate ends immediately before a Qwen3.5-9B agent response that chose
ordinary text rather than an agent tool call.  The agent-visible history,
system policy, and tool schemas are retained exactly as in the pre-tool-call
set.  A later Qwen3-8B screen, not the source rollout, determines whether the
target model also makes a non-tool next-token decision.

The resulting examples support a natural positive-direction experiment:
whether adding the frozen paper ``mean_diff`` direction can induce a
structured ``<tool_call>`` where Qwen3-8B otherwise begins a normal reply.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from build_tau2_telecom_prefixes import (  # noqa: E402
    DEFAULT_RAW_ROOT,
    DEFAULT_OUTPUT_ROOT,
    SOURCE_AGENT_MODEL,
    SOURCE_HF_REVISION,
    is_agent_visible,
    normalize_visible_message,
    raw_tool_name,
    read_jsonl,
    sha256_file,
    source_model,
    stable_rank,
    write_json,
    write_jsonl,
)


DEFAULT_REPLY_OUTPUT_ROOT = DEFAULT_OUTPUT_ROOT / "text_reply_induction"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_REPLY_OUTPUT_ROOT)
    parser.add_argument(
        "--require-prior-agent-tool",
        action="store_true",
        help="Exclude early no-tool account-clarification turns. Off by default to retain both natural reply regimes.",
    )
    parser.add_argument(
        "--terminal-confirmation-only",
        action="store_true",
        help="Retain only user turns that naturally report a resolved/successful outcome and close the request.",
    )
    parser.add_argument("--seed", type=int, default=20260725)
    return parser.parse_args()


def prior_tool_group(count: int) -> int:
    """Compact the naturally sparse tool-depth values into balanced strata."""
    if count <= 0:
        return 0
    if count <= 3:
        return 3
    if count == 4:
        return 4
    return 5


def content_preview(content: str, *, limit: int = 240) -> str:
    text = " ".join(content.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


_TERMINAL_SUCCESS = re.compile(
    r"\b("
    r"(?:issue|problem).{0,32}\b(?:resolved|fixed)\b|"
    r"\b(?:resolved|fixed)\b.{0,32}(?:issue|problem)\b|"
    r"\b(?:can now|now can|working (?:perfectly|great|well)? now|works now)\b.{0,80}\b(?:send|mms|message)\b|"
    r"\b(?:excellent|very fast)\b.{0,80}\b(?:speed|connection|internet|mobile data)\b|"
    r"\b(?:speed|connection|internet|mobile data)\b.{0,80}\b(?:excellent|very fast)\b"
    r")",
    flags=re.IGNORECASE,
)
_TERMINAL_CLOSURE = re.compile(
    r"\b(thank(?:s| you)?|anything else|anything more|nothing else|no further|all set|concludes? my request|consider (?:this )?resolved)\b",
    flags=re.IGNORECASE,
)
_TERMINAL_NEGATION = re.compile(
    r"\b(still (?:can(?:not|'t)|is(?: not|n't)|not working)|what (?:should|do) I do next|what(?:'s| is) the next step)\b",
    flags=re.IGNORECASE,
)


def is_terminal_confirmation(user_text: str) -> bool:
    """Conservative lexical selector for real post-resolution user turns.

    This only narrows an external raw trace pool.  The decisive behavior
    filter remains Qwen3-8B's own baseline next-token prediction.
    """
    return bool(
        _TERMINAL_SUCCESS.search(user_text)
        and _TERMINAL_CLOSURE.search(user_text)
        and not _TERMINAL_NEGATION.search(user_text)
    )


def extract_candidates(
    traces: Iterable[dict[str, Any]], *, require_prior_agent_tool: bool, terminal_confirmation_only: bool
) -> tuple[list[dict[str, Any]], collections.Counter[str]]:
    candidates: list[dict[str, Any]] = []
    exclusions: collections.Counter[str] = collections.Counter()

    for trace_index, trace in enumerate(traces):
        visible_history: list[dict[str, Any]] = []
        prior_agent_tool_names: list[str] = []
        agent_text_ordinal = 0

        for source_message_index, message in enumerate(trace.get("messages") or []):
            role = message.get("role")
            calls = message.get("tool_calls") or []
            qwen_agent = role == "assistant" and source_model(message) == SOURCE_AGENT_MODEL
            content = str(message.get("content") or "")

            # Keep only an actual Qwen agent text turn, never the initial
            # canned greeting and never a mixed text/tool turn.
            if qwen_agent and not calls and content.strip():
                agent_text_ordinal += 1
                previous_role = str(visible_history[-1].get("role")) if visible_history else "none"
                if previous_role != "user":
                    exclusions[f"target_preceded_by_{previous_role}"] += 1
                elif terminal_confirmation_only and not is_terminal_confirmation(
                    str(visible_history[-1].get("content") or "")
                ):
                    exclusions["target_not_terminal_confirmation"] += 1
                elif require_prior_agent_tool and not prior_agent_tool_names:
                    exclusions["target_without_prior_agent_tool"] += 1
                else:
                    candidate_id = f"telecom_reply_t{trace_index:04d}_m{source_message_index:02d}"
                    candidates.append(
                        {
                            "candidate_id": candidate_id,
                            "trace_index": trace_index,
                            "task_id": trace.get("task_id"),
                            "task_description": trace.get("task_description"),
                            "target_source_message_index": source_message_index,
                            "target_turn_idx": message.get("turn_idx"),
                            "target_kind": "qwen_text_reply_after_user",
                            "source_target_text": content,
                            "source_target_text_preview": content_preview(content),
                            "source_target_text_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
                            "source_agent_text_ordinal": agent_text_ordinal,
                            "prior_agent_tool_call_count": len(prior_agent_tool_names),
                            "prior_agent_tool_depth_group": prior_tool_group(len(prior_agent_tool_names)),
                            "prior_agent_tool_names": list(prior_agent_tool_names),
                            "source_agent_model": source_model(message),
                            "source_reward": trace.get("reward"),
                            "source_success": trace.get("success"),
                            "source_termination_reason": trace.get("termination_reason"),
                            "source_tool_call_count": trace.get("tool_call_count"),
                            "source_num_messages": trace.get("num_messages"),
                            "visible_message_count": len(visible_history),
                            "messages": list(visible_history),
                        }
                    )

            if qwen_agent and calls:
                prior_agent_tool_names.extend(raw_tool_name(call) for call in calls if raw_tool_name(call))

            if is_agent_visible(message):
                visible_history.append(normalize_visible_message(message))

    return candidates, exclusions


def build_screen_pool(candidates: list[dict[str, Any]], *, seed: int) -> list[dict[str, Any]]:
    """Choose one source-independent, depth-balanced text target per trace."""
    by_trace: dict[int, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in candidates:
        by_trace[int(row["trace_index"])].append(row)

    target_groups = (0, 3, 4, 5)
    pool: list[dict[str, Any]] = []
    for trace_index, rows in sorted(by_trace.items()):
        desired_group = target_groups[stable_rank(seed, "reply-depth", trace_index) % len(target_groups)]
        chosen = min(
            rows,
            key=lambda row: (
                int(row["prior_agent_tool_depth_group"]) != desired_group,
                # Prefer a shorter source history only after preserving the
                # intended depth; screening will later enforce an exact token
                # limit with Qwen3-8B's own template.
                int(row["visible_message_count"]),
                stable_rank(seed, "reply-pool", row["candidate_id"]),
            ),
        )
        pool.append(chosen)
    return pool


def counter_as_dict(counter: collections.Counter[Any]) -> dict[str, int]:
    return {str(key): int(value) for key, value in sorted(counter.items(), key=lambda item: str(item[0]))}


def main() -> None:
    args = parse_args()
    raw_root = args.raw_root.resolve()
    output_root = args.output_root.resolve()
    trace_path = raw_root / "tau2_baseline_judged.jsonl"
    system_path = raw_root / "tau2_system_prompt.txt"
    tools_path = raw_root / "tau2_tool_schemas.json"
    for path in (trace_path, system_path, tools_path):
        if not path.exists():
            raise FileNotFoundError(path)

    tool_schemas = json.loads(tools_path.read_text(encoding="utf-8"))
    if not isinstance(tool_schemas, list) or not tool_schemas:
        raise ValueError("Expected a nonempty Telecom tool schema list")
    if not system_path.read_text(encoding="utf-8").strip():
        raise ValueError("System prompt is empty")

    traces = list(read_jsonl(trace_path))
    candidates, exclusions = extract_candidates(
        traces,
        require_prior_agent_tool=bool(args.require_prior_agent_tool),
        terminal_confirmation_only=bool(args.terminal_confirmation_only),
    )
    if not candidates:
        raise RuntimeError("No eligible text-reply prefixes were extracted")
    screen_pool = build_screen_pool(candidates, seed=args.seed)

    output_root.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_root / "candidate_prefixes.jsonl", candidates)
    write_jsonl(output_root / "screen_pool.jsonl", screen_pool)
    provenance = {
        "dataset": "KermitCO/qwen3.5-9B-tau2bench-telecom-traces",
        "hf_revision": SOURCE_HF_REVISION,
        "source_agent_model": SOURCE_AGENT_MODEL,
        "raw_files": {
            path.name: {"sha256": sha256_file(path), "bytes": path.stat().st_size}
            for path in (trace_path, system_path, tools_path)
        },
        "agent_visible_history_rule": {
            "assistant": "include",
            "user": "include only if it is not a user-simulator tool call",
            "tool": "include only if requestor == assistant",
            "stored_reasoning_content": "excluded",
        },
        "target_rule": {
            "source_model": SOURCE_AGENT_MODEL,
            "assistant_content": "nonempty text",
            "assistant_tool_calls": "absent/empty",
            "preceding_visible_role": "user",
            "require_prior_agent_tool": bool(args.require_prior_agent_tool),
            "terminal_confirmation_only": bool(args.terminal_confirmation_only),
        },
    }
    summary = {
        "traces": len(traces),
        "candidate_prefixes": len(candidates),
        "screen_pool_prefixes": len(screen_pool),
        "screen_pool_unit": "one pre-text-reply candidate per source trajectory",
        "screen_pool_seed": args.seed,
        "candidate_depth_group_counts": counter_as_dict(
            collections.Counter(int(row["prior_agent_tool_depth_group"]) for row in candidates)
        ),
        "pool_depth_group_counts": counter_as_dict(
            collections.Counter(int(row["prior_agent_tool_depth_group"]) for row in screen_pool)
        ),
        "candidate_success_counts": counter_as_dict(
            collections.Counter(bool(row["source_success"]) for row in candidates)
        ),
        "exclusions": counter_as_dict(exclusions),
    }
    write_json(output_root / "provenance.json", provenance)
    write_json(output_root / "extraction_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
