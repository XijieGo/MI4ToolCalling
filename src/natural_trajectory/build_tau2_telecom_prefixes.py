#!/usr/bin/env python3
"""Extract agent-visible, pre-tool-call prefixes from τ² Telecom traces.

The raw τ² trace contains two actors: the customer-service agent and a user
simulator.  A user-simulator tool invocation is *not* visible to the agent on
its next turn.  This script follows τ²'s `is_valid_agent_history_message`
rule, so every exported prefix contains only:

* ordinary user messages;
* prior assistant (agent) messages; and
* tool results requested by the assistant.

Each candidate ends immediately before one Qwen3.5-9B assistant tool call.
The target tool call itself is metadata, never part of the model input.  We
only retain tool-only assistant turns with exactly one call, and target the
second through fourth assistant tool calls.  Thus each candidate has a real
multi-turn/tool history, but does not require replaying the rest of a trace.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW_ROOT = PROJECT_ROOT / "datasets" / "external" / "tau2_telecom_qwen35_9b" / "raw"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "datasets" / "external" / "tau2_telecom_qwen35_9b" / "prepared"
SOURCE_AGENT_MODEL = "Qwen/Qwen3.5-9B"
SOURCE_HF_REVISION = "0e538681426672acc942acf6d34e73d65501bfe1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--min-agent-tool-ordinal",
        type=int,
        default=2,
        help="First eligible agent-tool ordinal (inclusive).",
    )
    parser.add_argument(
        "--max-agent-tool-ordinal",
        type=int,
        default=4,
        help="Last eligible agent-tool ordinal (inclusive).",
    )
    parser.add_argument("--seed", type=int, default=20260725)
    return parser.parse_args()


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}") from exc


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def stable_rank(*parts: object) -> int:
    text = "|".join(str(part) for part in parts)
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")


def is_agent_visible(message: dict[str, Any]) -> bool:
    """Mirror τ²'s `is_valid_agent_history_message` predicate."""
    role = message.get("role")
    if role == "assistant":
        return True
    if role == "user":
        return not bool(message.get("tool_calls"))
    if role == "tool":
        return message.get("requestor") == "assistant"
    return False


def normalize_arguments(value: Any) -> dict[str, Any]:
    """Canonicalize OpenAI-style serialized arguments for HF chat templates."""
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        parsed = json.loads(value)
        if not isinstance(parsed, dict):
            raise ValueError(f"Tool arguments must decode to a mapping, got {type(parsed).__name__}")
        return parsed
    raise TypeError(f"Unsupported tool-argument type: {type(value).__name__}")


def normalize_tool_call(call: dict[str, Any]) -> dict[str, Any]:
    """Produce the structure consumed by both Qwen3 and Qwen3.5 templates."""
    function = call.get("function") or {}
    name = call.get("name") or function.get("name")
    if not name:
        raise ValueError(f"Tool call without a name: {call}")
    arguments = call.get("arguments")
    if arguments is None:
        arguments = function.get("arguments")
    return {
        "id": str(call.get("id") or ""),
        "type": "function",
        "function": {
            "name": str(name),
            # Qwen3.5's native template expects a mapping here.  Qwen3's
            # template also accepts it and serializes it to JSON.
            "arguments": normalize_arguments(arguments),
        },
    }


def normalize_visible_message(message: dict[str, Any]) -> dict[str, Any]:
    role = str(message["role"])
    if role == "assistant":
        result: dict[str, Any] = {"role": "assistant", "content": message.get("content")}
        calls = message.get("tool_calls")
        if calls:
            result["tool_calls"] = [normalize_tool_call(call) for call in calls]
        return result
    if role == "user":
        return {"role": "user", "content": message.get("content")}
    if role == "tool":
        return {
            "role": "tool",
            "content": message.get("content") or "",
            "tool_call_id": str(message.get("id") or message.get("tool_call_id") or ""),
        }
    raise ValueError(f"Unexpected visible role: {role!r}")


def raw_tool_name(call: dict[str, Any]) -> str:
    return str(call.get("name") or (call.get("function") or {}).get("name") or "")


def raw_tool_arguments(call: dict[str, Any]) -> dict[str, Any]:
    function = call.get("function") or {}
    return normalize_arguments(call.get("arguments", function.get("arguments")))


def source_model(message: dict[str, Any]) -> str | None:
    raw_data = message.get("raw_data") or {}
    model = raw_data.get("model")
    return str(model) if model is not None else None


def extract_candidates(
    traces: Iterable[dict[str, Any]],
    *,
    min_agent_tool_ordinal: int,
    max_agent_tool_ordinal: int,
) -> tuple[list[dict[str, Any]], collections.Counter[str]]:
    candidates: list[dict[str, Any]] = []
    exclusions: collections.Counter[str] = collections.Counter()

    for trace_index, trace in enumerate(traces):
        visible_history: list[dict[str, Any]] = []
        prior_agent_tool_names: list[str] = []
        agent_tool_ordinal = 0

        for source_message_index, message in enumerate(trace.get("messages") or []):
            role = message.get("role")
            calls = message.get("tool_calls") or []
            is_qwen_agent_tool_turn = role == "assistant" and bool(calls) and source_model(message) == SOURCE_AGENT_MODEL

            if is_qwen_agent_tool_turn:
                agent_tool_ordinal += 1
                content = str(message.get("content") or "")
                if len(calls) != 1:
                    exclusions["qwen_target_multi_tool"] += 1
                elif content.strip():
                    exclusions["qwen_target_mixed_text_and_tool"] += 1
                elif not (min_agent_tool_ordinal <= agent_tool_ordinal <= max_agent_tool_ordinal):
                    exclusions["outside_selected_tool_ordinal_window"] += 1
                else:
                    target_call = calls[0]
                    target_name = raw_tool_name(target_call)
                    if not target_name:
                        exclusions["qwen_target_missing_tool_name"] += 1
                    else:
                        candidate_id = f"telecom_t{trace_index:04d}_m{source_message_index:02d}"
                        candidates.append(
                            {
                                "candidate_id": candidate_id,
                                "trace_index": trace_index,
                                "task_id": trace.get("task_id"),
                                "task_description": trace.get("task_description"),
                                "target_source_message_index": source_message_index,
                                "target_turn_idx": message.get("turn_idx"),
                                "target_tool_name": target_name,
                                "target_tool_arguments": raw_tool_arguments(target_call),
                                "target_agent_tool_ordinal": agent_tool_ordinal,
                                "prior_agent_tool_call_count": len(prior_agent_tool_names),
                                "prior_agent_tool_names": list(prior_agent_tool_names),
                                "source_agent_model": source_model(message),
                                "source_reward": trace.get("reward"),
                                "source_success": trace.get("success"),
                                "source_termination_reason": trace.get("termination_reason"),
                                "source_tool_call_count": trace.get("tool_call_count"),
                                "source_num_messages": trace.get("num_messages"),
                                "visible_message_count": len(visible_history),
                                # System prompt and tool schemas are deliberately not duplicated
                                # per row; provenance.json fixes the files and their checksums.
                                "messages": list(visible_history),
                            }
                        )
                prior_agent_tool_names.extend(raw_tool_name(call) for call in calls if raw_tool_name(call))

            elif role == "assistant" and calls:
                exclusions["non_qwen_assistant_tool_turn"] += 1

            if is_agent_visible(message):
                visible_history.append(normalize_visible_message(message))

    return candidates, exclusions


def build_screen_pool(candidates: list[dict[str, Any]], *, seed: int) -> list[dict[str, Any]]:
    """Pick one pre-screen candidate per trajectory without using model outputs.

    The pool preferentially retains rare tool identities, then balances the
    second/third/fourth tool-call depth with a stable seeded tie-break.  This
    avoids pseudo-replication while keeping the later model screen tractable.
    """
    global_tool_frequency = collections.Counter(str(row["target_tool_name"]) for row in candidates)
    by_trace: dict[int, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in candidates:
        by_trace[int(row["trace_index"])].append(row)

    selected: list[dict[str, Any]] = []
    for trace_index in sorted(by_trace):
        rows = by_trace[trace_index]
        desired_ordinal = 2 + (stable_rank(seed, "desired-ordinal", trace_index) % 3)
        chosen = min(
            rows,
            key=lambda row: (
                global_tool_frequency[str(row["target_tool_name"])],
                int(row["target_agent_tool_ordinal"]) != desired_ordinal,
                stable_rank(seed, "screen-pool", row["candidate_id"]),
            ),
        )
        selected.append(chosen)
    return selected


def counter_as_dict(counter: collections.Counter[Any]) -> dict[str, int]:
    return {str(key): int(value) for key, value in sorted(counter.items(), key=lambda item: str(item[0]))}


def main() -> None:
    args = parse_args()
    if args.min_agent_tool_ordinal < 1 or args.max_agent_tool_ordinal < args.min_agent_tool_ordinal:
        raise ValueError("Require 1 <= min-agent-tool-ordinal <= max-agent-tool-ordinal")

    raw_root = args.raw_root.resolve()
    output_root = args.output_root.resolve()
    trace_path = raw_root / "tau2_baseline_judged.jsonl"
    system_path = raw_root / "tau2_system_prompt.txt"
    tools_path = raw_root / "tau2_tool_schemas.json"
    for path in (trace_path, system_path, tools_path):
        if not path.exists():
            raise FileNotFoundError(path)

    # Parse schemas once here as a structural validation; rendering is done by
    # the screening script with each target model's native chat template.
    tool_schemas = json.loads(tools_path.read_text(encoding="utf-8"))
    if not isinstance(tool_schemas, list) or not tool_schemas:
        raise ValueError("Expected a non-empty list of tool schemas")
    system_prompt = system_path.read_text(encoding="utf-8").strip()
    if not system_prompt:
        raise ValueError("System prompt is empty")

    traces = list(read_jsonl(trace_path))
    candidates, exclusions = extract_candidates(
        traces,
        min_agent_tool_ordinal=args.min_agent_tool_ordinal,
        max_agent_tool_ordinal=args.max_agent_tool_ordinal,
    )
    if not candidates:
        raise RuntimeError("No eligible candidate prefixes were extracted")
    screen_pool = build_screen_pool(candidates, seed=args.seed)

    output_root.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_root / "candidate_prefixes.jsonl", candidates)
    write_jsonl(output_root / "screen_pool.jsonl", screen_pool)

    candidate_tool_counts = collections.Counter(str(row["target_tool_name"]) for row in candidates)
    pool_tool_counts = collections.Counter(str(row["target_tool_name"]) for row in screen_pool)
    candidate_ordinal_counts = collections.Counter(int(row["target_agent_tool_ordinal"]) for row in candidates)
    pool_ordinal_counts = collections.Counter(int(row["target_agent_tool_ordinal"]) for row in screen_pool)

    provenance = {
        "dataset": "KermitCO/qwen3.5-9B-tau2bench-telecom-traces",
        "hf_revision": SOURCE_HF_REVISION,
        "source_agent_model": SOURCE_AGENT_MODEL,
        "raw_files": {
            path.name: {"sha256": sha256_file(path), "bytes": path.stat().st_size}
            for path in (trace_path, system_path, tools_path)
        },
        "tau2_reference_commit": "1d244f5dca42944b67a379b44bfeb9f5748f189d",
        "agent_visible_history_rule": {
            "assistant": "include",
            "user": "include only if it is not a user-simulator tool call",
            "tool": "include only if requestor == assistant",
            "stored_reasoning_content": "excluded; τ² sends only role/content/tool_calls to LiteLLM",
        },
        "target_rule": {
            "source_model": SOURCE_AGENT_MODEL,
            "assistant_content": "empty/whitespace only",
            "tool_calls_per_target": 1,
            "agent_tool_ordinal_range": [args.min_agent_tool_ordinal, args.max_agent_tool_ordinal],
        },
        "tool_schema_count": len(tool_schemas),
        "system_prompt_bytes": len(system_prompt.encode("utf-8")),
    }
    summary = {
        "traces": len(traces),
        "candidate_prefixes": len(candidates),
        "screen_pool_prefixes": len(screen_pool),
        "screen_pool_unit": "one pre-intervention candidate per source trajectory",
        "screen_pool_seed": args.seed,
        "candidate_tool_counts": counter_as_dict(candidate_tool_counts),
        "screen_pool_tool_counts": counter_as_dict(pool_tool_counts),
        "candidate_ordinal_counts": counter_as_dict(candidate_ordinal_counts),
        "screen_pool_ordinal_counts": counter_as_dict(pool_ordinal_counts),
        "exclusions": counter_as_dict(exclusions),
    }
    write_json(output_root / "provenance.json", provenance)
    write_json(output_root / "extraction_summary.json", summary)

    print(
        json.dumps(
            {
                "output_root": str(output_root),
                "traces": len(traces),
                "candidates": len(candidates),
                "screen_pool": len(screen_pool),
                "tool_identities_in_screen_pool": len(pool_tool_counts),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
