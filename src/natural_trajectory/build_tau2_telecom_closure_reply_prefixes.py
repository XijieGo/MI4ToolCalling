#!/usr/bin/env python3
"""Derive a broader natural direct-reply pool from existing τ² traces.

The strict terminal pool requires both a success statement and a closure.  For
the 200-example collection, this companion pool retains any real user turn
that closes the interaction (for example, thanks/all set/no further help)
immediately before the historical Qwen3.5 agent's text-only reply.  It never
uses reward, task success, or reply correctness.  The target model's own
baseline screen remains the decisive non-tool label.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = (
    PROJECT_ROOT
    / "datasets"
    / "external"
    / "tau2_telecom_qwen35_9b"
    / "prepared"
    / "text_reply_induction"
    / "candidate_prefixes.jsonl"
)
DEFAULT_OUTPUT = DEFAULT_SOURCE.parent.parent / "text_reply_induction_closure"
SEED = 20260726

_CLOSURE = re.compile(
    r"\b(thank(?:s| you)?|anything else|anything more|nothing else|no further|all set|"
    r"concludes? my request|consider (?:this )?resolved)\b",
    flags=re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected an object at {path}:{line_number}")
            rows.append(row)
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def stable_rank(*parts: object) -> int:
    text = "|".join(str(part) for part in parts)
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")


def preceding_user_text(row: dict[str, Any]) -> str:
    messages = row.get("messages")
    if not isinstance(messages, list) or not messages:
        return ""
    last = messages[-1]
    if not isinstance(last, dict) or last.get("role") != "user":
        return ""
    return str(last.get("content") or "")


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    output = args.output_root.resolve()
    rows = read_jsonl(source)
    selected = [row for row in rows if _CLOSURE.search(preceding_user_text(row))]
    if not selected:
        raise RuntimeError("No natural closure-reply candidates found")
    by_trace: dict[int, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in selected:
        by_trace[int(row["trace_index"])].append(row)
    screen_pool: list[dict[str, Any]] = []
    for trace_index, values in sorted(by_trace.items()):
        screen_pool.append(
            min(
                values,
                key=lambda row: (
                    int(row.get("prior_agent_tool_depth_group") or 0),
                    stable_rank(args.seed, "closure-reply", trace_index, row["candidate_id"]),
                ),
            )
        )
    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(output / "candidate_prefixes.jsonl", selected)
    write_jsonl(output / "screen_pool.jsonl", screen_pool)
    summary = {
        "source": str(source),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "selector": "real preceding user turn contains an interaction-closure phrase; source agent's next turn is the existing text-only reply",
        "uses_source_success_or_reward": False,
        "candidate_rows": len(selected),
        "independent_trajectory_count": len(by_trace),
        "independent_task_id_count": len({str(row["task_id"]) for row in screen_pool}),
        "screen_pool_rows": len(screen_pool),
        "seed": args.seed,
    }
    (output / "extraction_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
