#!/usr/bin/env python3
"""Prepare task/rollout-disjoint τ² Retail prefixes from public Qwen3.5-9B traces.

The input release contains repeated independent simulations of some τ² Retail
task IDs.  A ``task_id`` in the prepared files is therefore a *rollout ID*
(``retail:<base-task>:trace:<n>``), not the benchmark's base task ID.  This
lets downstream screening keep one decision point per independent trajectory
while retaining the base task ID for audit.

No reward, judge, or success field is read for candidate selection.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(".")
DEFAULT_RAW = PROJECT_ROOT / "datasets/external/tau2_retail_qwen35_9b/raw/tau2_retail_judged_v2.jsonl"
DEFAULT_OUTPUT = PROJECT_ROOT / "datasets/external/tau2_retail_qwen35_9b/prepared"
SEED = 20260726


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def stable_rank(*parts: object) -> str:
    payload = "\x1f".join(str(part) for part in (SEED, *parts))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def compact_message(message: dict[str, Any]) -> dict[str, Any]:
    """Keep exactly the fields needed by each native chat-template adaptor."""
    kept: dict[str, Any] = {
        "role": str(message["role"]),
        "content": message.get("content"),
    }
    if message.get("tool_calls"):
        kept["tool_calls"] = message["tool_calls"]
    # τ² Retail uses the tool-result message's id as its call ID.
    for field in ("id", "tool_call_id", "name"):
        if message.get(field) is not None:
            kept[field] = message[field]
    return kept


def prior_tool_depth(messages: list[dict[str, Any]]) -> int:
    return sum(
        len(message.get("tool_calls") or [])
        for message in messages
        if message.get("role") == "assistant"
    )


def source_rollout_id(base_task_id: Any, trace_index: int) -> str:
    return f"retail:{base_task_id}:trace:{trace_index}"


def candidate_row(
    *,
    trace_index: int,
    base_task_id: Any,
    target_index: int,
    target: dict[str, Any],
    prefix: list[dict[str, Any]],
    kind: str,
) -> dict[str, Any]:
    calls = target.get("tool_calls") or []
    tool_name = None
    if calls:
        first = calls[0]
        function = first.get("function") or {}
        tool_name = first.get("name") or function.get("name")
    rollout_id = source_rollout_id(base_task_id, trace_index)
    return {
        "candidate_id": f"retail_t{trace_index:04d}_m{target_index:02d}_{kind}",
        "trace_index": trace_index,
        "task_id": rollout_id,
        "source_base_task_id": str(base_task_id),
        "source_domain": "retail",
        "messages": prefix,
        "target_kind": kind,
        "target_tool_name": str(tool_name) if tool_name is not None else None,
        # Human-readable tool depth is 1-indexed for real calls.
        "target_agent_tool_ordinal": prior_tool_depth(prefix) + 1 if calls else None,
        "prior_agent_tool_depth_group": prior_tool_depth(prefix),
        "source_success": None,
    }


def one_per_rollout(rows: list[dict[str, Any]], *, arm: str) -> list[dict[str, Any]]:
    """Choose a deterministic, semantically clean primary point per rollout."""
    grouped: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        grouped[str(row["task_id"])].append(row)
    selected: list[dict[str, Any]] = []
    for rollout_id in sorted(grouped, key=lambda value: stable_rank(arm, "rollout", value)):
        values = grouped[rollout_id]
        if arm == "tool":
            # The last source-agent call is closest to the final unresolved
            # subgoal and yields a clean natural decision point.
            values.sort(key=lambda row: (-int(row["target_agent_tool_ordinal"] or 0), stable_rank(arm, row["candidate_id"])))
        else:
            # The last natural text turn is a direct-answer state, often after
            # the customer confirms/clarifies a request.
            values.sort(key=lambda row: (-int(str(row["candidate_id"]).split("_m")[1].split("_")[0]), stable_rank(arm, row["candidate_id"])))
        selected.append(values[0])
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    traces = read_jsonl(args.raw)
    tools: list[dict[str, Any]] = []
    direct: list[dict[str, Any]] = []
    for trace_index, trace in enumerate(traces):
        source = trace.get("messages")
        if not isinstance(source, list):
            continue
        base_task_id = trace.get("task_id")
        compact = [compact_message(message) for message in source]
        for target_index, target in enumerate(compact):
            if target.get("role") != "assistant":
                continue
            prefix = compact[:target_index]
            if not prefix:
                continue
            if target.get("tool_calls"):
                tools.append(candidate_row(
                    trace_index=trace_index,
                    base_task_id=base_task_id,
                    target_index=target_index,
                    target=target,
                    prefix=prefix,
                    kind="qwen35_retail_tool_call",
                ))
                continue
            # Omit the canned greeting only.  Every retained text candidate has
            # a real preceding user or tool state and is an observed direct reply.
            if target.get("content") and any(message.get("role") == "user" for message in prefix):
                direct.append(candidate_row(
                    trace_index=trace_index,
                    base_task_id=base_task_id,
                    target_index=target_index,
                    target=target,
                    prefix=prefix,
                    kind="qwen35_retail_text_reply",
                ))

    output = args.output.resolve()
    tool_primary = one_per_rollout(tools, arm="tool")
    direct_primary = one_per_rollout(direct, arm="direct")
    write_jsonl(output / "candidate_prefixes.jsonl", tools)
    write_jsonl(output / "screen_pool.jsonl", tool_primary)
    write_jsonl(output / "text_reply_induction/candidate_prefixes.jsonl", direct)
    write_jsonl(output / "text_reply_induction_terminal/screen_pool.jsonl", direct_primary)
    summary = {
        "source": str(args.raw.resolve()),
        "trace_count": len(traces),
        "base_task_count": len({str(row.get("task_id")) for row in traces}),
        "independent_rollout_count": len({str(row["task_id"]) for row in tool_primary}),
        "tool_candidates": len(tools),
        "tool_primary_rows": len(tool_primary),
        "direct_candidates": len(direct),
        "direct_primary_rows": len(direct_primary),
        "rule": "Candidates are all observed Qwen3.5-9B agent tool/text turns; no reward or judge field is read.",
    }
    (output / "preparation_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
