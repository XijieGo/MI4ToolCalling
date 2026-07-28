#!/usr/bin/env python3
"""Build the Qwen3.5-4B v5 dataset from its native TAU2 tool context pool.

Qwen3.5-4B does not reliably call a tool on the code-writing prompt format
used by the Qwen3 family.  This builder instead uses short, model-native
telecom histories and changes only the final request verb.  It writes the same
v5 schema as ``build_v5_model_specific_balanced.py``.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

if __package__:
    from .build_v5_model_specific_balanced import (
        DEFAULT_OUTPUT_ROOT,
        MODEL_SPECS,
        choose_largest_feasible_vocabulary,
        evaluate_texts,
        load_model_and_tokenizer,
        release_model,
        sha256_text,
        split_selected,
        summary_for_rows,
        write_csv,
        write_json,
        write_jsonl,
    )
else:  # Support direct invocation as a script as well as ``python -m``.
    from build_v5_model_specific_balanced import (
        DEFAULT_OUTPUT_ROOT,
        MODEL_SPECS,
        choose_largest_feasible_vocabulary,
        evaluate_texts,
        load_model_and_tokenizer,
        release_model,
        sha256_text,
        split_selected,
        summary_for_rows,
        write_csv,
        write_json,
        write_jsonl,
    )


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SPEC = MODEL_SPECS["qwen35_4b"]
SOURCE_ROOT = PROJECT_ROOT / "datasets" / "external" / "tau2_telecom_qwen35_9b"
SOURCE_ROWS = SOURCE_ROOT / "prepared" / "candidate_prefixes.jsonl"
SYSTEM_POLICY = (
    "You are a telecom support agent. Use the provided functions when the user asks you to "
    "perform an account action. When the user asks only to discuss, explain, review, summarize, "
    "or describe an action, answer in text without using a function."
)
CLEAN_VERBS = ("execute", "perform")
CORRUPT_VERBS = ("discuss", "explain", "summarize")
FINAL_REQUEST = "Please {verb} the next appropriate account action now."


@dataclass(frozen=True)
class TauCandidate:
    candidate_id: str
    source_candidate_id: str
    task_id: str
    trace_index: int
    target_tool_name: str
    clean_verb: str
    corrupt_verb: str
    clean_text: str
    corrupt_text: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--candidates-per-cell", type=int, default=250)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-prompt-tokens", type=int, default=2048)
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--min-clean-margin", type=float, default=0.0)
    parser.add_argument("--min-corrupt-non-tool-margin", type=float, default=0.0)
    parser.add_argument("--clean-verbs", nargs="+", default=list(CLEAN_VERBS))
    parser.add_argument("--corrupt-verbs", nargs="+", default=list(CORRUPT_VERBS))
    parser.add_argument(
        "--screening-report",
        type=Path,
        help="Write all rendered candidate screening rows here before balanced selection.",
    )
    parser.add_argument(
        "--screening-input",
        type=Path,
        help="Reuse a previously written complete screening CSV with the same seed, verbs, and candidates-per-cell.",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_screening_csv(path: Path) -> list[dict[str, Any]]:
    bool_fields = {"clean_is_tool_top1", "corrupt_is_tool_top1", "behavior_valid"}
    int_fields = {"clean_tool_rank", "clean_top1_token_id", "corrupt_tool_rank", "corrupt_top1_token_id", "source_trace_index"}
    float_fields = {
        "clean_tool_logit",
        "clean_tool_probability",
        "clean_margin_vs_best_non_tool",
        "corrupt_tool_logit",
        "corrupt_tool_probability",
        "corrupt_margin_vs_best_non_tool",
        "quality",
    }
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    if not rows:
        raise ValueError(f"{path}: screening CSV has no rows")
    for row in rows:
        for field in bool_fields:
            if field in row:
                if row[field] not in {"True", "False"}:
                    raise ValueError(f"{path}: {field} is not a boolean literal")
                row[field] = row[field] == "True"
        for field in int_fields:
            if field in row and row[field] != "":
                row[field] = int(row[field])
        for field in float_fields:
            if field in row and row[field] != "":
                row[field] = float(row[field])
    return rows


def load_sources(tool_names: set[str]) -> list[dict[str, Any]]:
    rows = []
    seen: set[str] = set()
    for row in read_jsonl(SOURCE_ROWS):
        candidate_id = str(row.get("candidate_id") or "")
        target_tool = str(row.get("target_tool_name") or "")
        messages = row.get("messages")
        if not candidate_id or candidate_id in seen or target_tool not in tool_names or not isinstance(messages, list):
            continue
        seen.add(candidate_id)
        rows.append(row)
    if len(rows) < 500:
        raise RuntimeError(f"Only {len(rows)} usable TAU2 source states were found")
    return rows


def native_messages(messages: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    for raw in messages:
        role = str(raw.get("role") or "")
        if not role:
            raise ValueError("A TAU2 message has no role")
        item: dict[str, Any] = {"role": role, "content": raw.get("content") or ""}
        calls = raw.get("tool_calls") or []
        if role == "assistant" and calls:
            item["tool_calls"] = [
                {
                    "type": "function",
                    "function": {
                        "name": str(call.get("function", {}).get("name") or call.get("name") or ""),
                        "arguments": call.get("function", {}).get("arguments") or call.get("arguments") or "{}",
                    },
                }
                for call in calls
            ]
        converted.append(item)
    return converted


def render_prompt(row: dict[str, Any], verb: str, *, tokenizer: Any, tool_by_name: dict[str, dict[str, Any]]) -> str:
    target_tool = str(row["target_tool_name"])
    # Four final turns retain the locally relevant account state while keeping
    # each screening prompt short enough to evaluate a broad candidate pool.
    messages = [
        {"role": "system", "content": SYSTEM_POLICY},
        *native_messages(row["messages"][-4:]),
        {"role": "user", "content": FINAL_REQUEST.format(verb=verb)},
    ]
    kwargs = {
        "tools": [tool_by_name[target_tool]],
        "tokenize": False,
        "add_generation_prompt": True,
        "enable_thinking": False,
    }
    try:
        rendered = tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError:
        kwargs.pop("enable_thinking")
        rendered = tokenizer.apply_chat_template(messages, **kwargs)
    if not isinstance(rendered, str) or not rendered:
        raise ValueError("Native template did not produce a prompt string")
    return rendered


def build_candidates(
    sources: Sequence[dict[str, Any]],
    *,
    tokenizer: Any,
    tool_by_name: dict[str, dict[str, Any]],
    candidates_per_cell: int,
    max_prompt_tokens: int,
    clean_verbs: Sequence[str],
    corrupt_verbs: Sequence[str],
    seed: int,
) -> tuple[list[TauCandidate], list[dict[str, Any]]]:
    randomizer = random.Random(seed)
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in sources:
        by_task[str(row["task_id"])].append(row)
    task_ids = list(by_task)
    randomizer.shuffle(task_ids)
    cells = [(clean, corrupt) for clean in clean_verbs for corrupt in corrupt_verbs]
    needed = len(cells) * candidates_per_cell
    if needed > len(task_ids):
        raise ValueError(f"Need {needed} task-disjoint source states, only {len(task_ids)} TAU2 tasks are available")
    candidates: list[TauCandidate] = []
    rejected: list[dict[str, Any]] = []
    cursor = 0
    for slot in range(candidates_per_cell):
        for clean_verb, corrupt_verb in cells:
            while cursor < len(task_ids):
                task_id = task_ids[cursor]
                cursor += 1
                options = list(by_task[task_id])
                randomizer.shuffle(options)
                for row in options:
                    try:
                        clean_text = render_prompt(row, clean_verb, tokenizer=tokenizer, tool_by_name=tool_by_name)
                        corrupt_text = render_prompt(row, corrupt_verb, tokenizer=tokenizer, tool_by_name=tool_by_name)
                        clean_length = len(tokenizer.encode(clean_text, add_special_tokens=False))
                        corrupt_length = len(tokenizer.encode(corrupt_text, add_special_tokens=False))
                        if max(clean_length, corrupt_length) > max_prompt_tokens:
                            raise ValueError(f"prompt_too_long:{clean_length}/{corrupt_length}")
                    except (KeyError, TypeError, ValueError) as exc:
                        rejected.append(
                            {
                                "source_candidate_id": row.get("candidate_id"),
                                "source_task_id": task_id,
                                "clean_verb": clean_verb,
                                "corrupt_verb": corrupt_verb,
                                "reason": str(exc),
                            }
                        )
                        continue
                    source_id = str(row["candidate_id"])
                    candidates.append(
                        TauCandidate(
                            candidate_id=f"{source_id}__{clean_verb}__{corrupt_verb}",
                            source_candidate_id=source_id,
                            task_id=task_id,
                            trace_index=int(row.get("trace_index") or -1),
                            target_tool_name=str(row["target_tool_name"]),
                            clean_verb=clean_verb,
                            corrupt_verb=corrupt_verb,
                            clean_text=clean_text,
                            corrupt_text=corrupt_text,
                        )
                    )
                    break
                else:
                    continue
                break
            else:
                raise RuntimeError(
                    f"Ran out of valid unique TAU2 source states after constructing {len(candidates)} candidates"
                )
    return candidates, rejected


def screen_candidates(
    candidates: Sequence[TauCandidate], *, model: Any, tokenizer: Any, tool_token_id: int, batch_size: int
) -> list[dict[str, Any]]:
    clean_metrics = evaluate_texts(
        model,
        tokenizer,
        [candidate.clean_text for candidate in candidates],
        tool_token_id=tool_token_id,
        batch_size=batch_size,
        progress_label="qwen35_4b clean screening",
    )
    corrupt_metrics = evaluate_texts(
        model,
        tokenizer,
        [candidate.corrupt_text for candidate in candidates],
        tool_token_id=tool_token_id,
        batch_size=batch_size,
        progress_label="qwen35_4b corrupt screening",
    )
    rows: list[dict[str, Any]] = []
    for candidate, clean, corrupt in zip(candidates, clean_metrics, corrupt_metrics, strict=True):
        quality = float(clean["margin_vs_best_non_tool"]) - float(corrupt["margin_vs_best_non_tool"])
        rows.append(
            {
                "candidate_id": candidate.candidate_id,
                "source_sample_id": candidate.source_candidate_id,
                "source_filename": f"{candidate.source_candidate_id}.jsonl",
                "source_dataset": "tau2_telecom",
                "source_language": "en",
                "source_split": "natural_tool_history",
                "source_task_id": candidate.task_id,
                "source_trace_index": candidate.trace_index,
                "target_tool_name": candidate.target_tool_name,
                "clean_verb": candidate.clean_verb,
                "corrupt_verb": candidate.corrupt_verb,
                "clean_prompt_sha256": sha256_text(candidate.clean_text),
                "corrupt_prompt_sha256": sha256_text(candidate.corrupt_text),
                **{f"clean_{key}": value for key, value in clean.items()},
                **{f"corrupt_{key}": value for key, value in corrupt.items()},
                "behavior_valid": bool(clean["is_tool_top1"]) and not bool(corrupt["is_tool_top1"]),
                "quality": quality,
            }
        )
    return rows


def write_dataset(
    destination: Path,
    *,
    candidates: Sequence[TauCandidate],
    screening: Sequence[dict[str, Any]],
    selected: Sequence[dict[str, Any]],
    tool_token_id: int,
    candidates_per_cell: int,
    selected_clean_verbs: Sequence[str],
    selected_corrupt_verbs: Sequence[str],
    seed: int,
    min_clean_margin: float,
    min_corrupt_non_tool_margin: float,
    rejected: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    destination.mkdir(parents=True, exist_ok=False)
    for split in ("train", "heldout"):
        (destination / split / "clean").mkdir(parents=True)
        (destination / split / "corrupt").mkdir(parents=True)
    candidate_map = {candidate.candidate_id: candidate for candidate in candidates}
    manifest: list[dict[str, Any]] = []
    ordered = sorted(
        selected,
        key=lambda row: (str(row["split"]), str(row["clean_verb"]), str(row["corrupt_verb"]), str(row["candidate_id"])),
    )
    for index, row in enumerate(ordered, start=1):
        candidate = candidate_map[str(row["candidate_id"])]
        split = str(row["split"])
        filename = f"{index:03d}_{candidate.source_candidate_id}.txt"
        clean_path = destination / split / "clean" / filename
        corrupt_path = destination / split / "corrupt" / filename
        clean_path.write_text(candidate.clean_text, encoding="utf-8")
        corrupt_path.write_text(candidate.corrupt_text, encoding="utf-8")
        manifest.append(
            {
                "sample_id": f"v5_qwen35_4b_{index:03d}",
                "split": split,
                "filename": filename,
                "clean_relpath": str(clean_path.relative_to(destination)),
                "corrupt_relpath": str(corrupt_path.relative_to(destination)),
                "source_sample_id": candidate.source_candidate_id,
                "source_dataset": "tau2_telecom",
                "source_task_id": candidate.task_id,
                "source_trace_index": candidate.trace_index,
                "source_candidate_file": str(SOURCE_ROWS),
                "target_tool_name": candidate.target_tool_name,
                "template": "four_turn_tau2_history_plus_single_final_verb_edit",
                "clean_candidate": candidate.clean_verb,
                "corrupt_candidate": candidate.corrupt_verb,
                "tool_call_marker": SPEC.tool_marker,
                "tool_call_token_id": tool_token_id,
                **{key: value for key, value in row.items() if key not in {"split", "source_sample_id", "source_filename", "source_dataset", "source_language", "source_split", "source_task_id", "source_trace_index", "target_tool_name"}},
            }
        )
    write_jsonl(destination / "manifest.jsonl", manifest)
    write_csv(destination / "candidate_screening.csv", list(screening))
    if rejected:
        write_jsonl(destination / "candidate_construction_rejections.jsonl", rejected)
    summaries = {split: summary_for_rows([row for row in selected if row["split"] == split]) for split in ("train", "heldout")}
    for split, summary in summaries.items():
        if summary["clean_tool_call_top1_rate"] != 1.0 or summary["corrupt_tool_call_top1_rate"] != 0.0:
            raise AssertionError(f"{split} contains a behavior-invalid selected row")
    summary = {
        "schema_version": 1,
        "dataset_version": "v5_model_specific_balanced",
        "model_key": SPEC.key,
        "model_label": SPEC.label,
        "model_path": str(SPEC.model_path),
        "source_root": str(SOURCE_ROOT),
        "source_candidate_file": str(SOURCE_ROWS),
        "tool_call_marker": SPEC.tool_marker,
        "tool_call_token_id": tool_token_id,
        "seed": seed,
        "candidate_per_verb_cell": candidates_per_cell,
        "selected_clean_verbs": list(selected_clean_verbs),
        "selected_corrupt_verbs": list(selected_corrupt_verbs),
        "selection_rule": "Target-model first-token clean-call/corrupt-non-call screen with required safety margins, followed by maximum-quality exact marginal verb balancing.",
        "screening_margin_requirements": {
            "minimum_clean_tool_minus_best_non_tool": min_clean_margin,
            "minimum_corrupt_best_non_tool_minus_tool": min_corrupt_non_tool_margin,
        },
        "template": "four_turn_tau2_history_plus_single_final_verb_edit",
        "n_pairs": len(selected),
        "n_train": sum(row["split"] == "train" for row in selected),
        "n_heldout": sum(row["split"] == "heldout" for row in selected),
        "all": summary_for_rows(selected),
        "splits": summaries,
    }
    write_json(destination / "summary.json", summary)
    (destination / "README.md").write_text(
        "\n".join(
            [
                "# Qwen3.5-4B v5 dataset",
                "",
                "This is a Qwen3.5-4B-specific, behavior-screened TAU2 telecom dataset.",
                "",
                f"- Pairs: `{summary['n_pairs']}` (`{summary['n_train']}` train, `{summary['n_heldout']}` held-out).",
                f"- Native first-token call marker: `{SPEC.tool_marker}` (ID `{tool_token_id}`).",
                "- Each paired prompt changes only the final action/discussion verb.",
                "- Admission rule: clean marker is top-1 and corrupt marker is not top-1.",
                "- Verb counts are balanced independently on clean and corrupt sides in every split.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    args = parse_args()
    if args.min_clean_margin < 0.0 or args.min_corrupt_non_tool_margin < 0.0:
        raise ValueError("Margin requirements must be non-negative")
    clean_verbs = tuple(str(value).lower() for value in args.clean_verbs)
    corrupt_verbs = tuple(str(value).lower() for value in args.corrupt_verbs)
    if len(clean_verbs) < 2 or len(corrupt_verbs) < 2:
        raise ValueError("At least two clean and two corrupt verbs are required for balanced v5 splits")
    if len(set(clean_verbs)) != len(clean_verbs) or len(set(corrupt_verbs)) != len(corrupt_verbs):
        raise ValueError("Clean and corrupt verb lists must not contain duplicates")
    output_root = args.output_root.resolve()
    destination = output_root / SPEC.key
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {destination}")
    tool_schemas = json.loads((SOURCE_ROOT / "raw" / "tau2_tool_schemas.json").read_text(encoding="utf-8"))
    tool_by_name = {str(item["function"]["name"]): item for item in tool_schemas}
    sources = load_sources(set(tool_by_name))
    print(f"[{SPEC.key}] loading {SPEC.label}; {len(sources)} native TAU2 source states", flush=True)
    model, tokenizer, tool_token_id = load_model_and_tokenizer(SPEC, str(args.dtype))
    try:
        candidates, rejected = build_candidates(
            sources,
            tokenizer=tokenizer,
            tool_by_name=tool_by_name,
            candidates_per_cell=int(args.candidates_per_cell),
            max_prompt_tokens=int(args.max_prompt_tokens),
            clean_verbs=clean_verbs,
            corrupt_verbs=corrupt_verbs,
            seed=int(args.seed),
        )
        print(f"[{SPEC.key}] rendered {len(candidates)} candidates; {len(rejected)} source states skipped", flush=True)
        if args.screening_input:
            screening = read_screening_csv(args.screening_input.resolve())
        else:
            screening = screen_candidates(
                candidates,
                model=model,
                tokenizer=tokenizer,
                tool_token_id=tool_token_id,
                batch_size=int(args.batch_size),
            )
    finally:
        release_model(model)
    candidate_ids = {candidate.candidate_id for candidate in candidates}
    screening_ids = {str(row.get("candidate_id")) for row in screening}
    if len(screening) != len(candidate_ids) or screening_ids != candidate_ids:
        raise ValueError("Screening rows do not exactly match the deterministic rendered candidate set")
    if args.screening_report:
        report_path = args.screening_report.resolve()
        if report_path.exists():
            raise FileExistsError(f"Refusing to overwrite screening report: {report_path}")
        write_csv(report_path, screening)
    valid = [
        row
        for row in screening
        if bool(row["behavior_valid"])
        and float(row["clean_margin_vs_best_non_tool"]) > float(args.min_clean_margin)
        and -float(row["corrupt_margin_vs_best_non_tool"]) > float(args.min_corrupt_non_tool_margin)
    ]
    print(
        f"[{SPEC.key}] valid target behavior with clean>{args.min_clean_margin:g}, "
        f"corrupt-non-tool>{args.min_corrupt_non_tool_margin:g}: {len(valid)}/{len(screening)} candidates",
        flush=True,
    )
    selected, clean_verbs, corrupt_verbs = choose_largest_feasible_vocabulary(
        valid,
        preferred_clean_verbs=clean_verbs,
        preferred_corrupt_verbs=corrupt_verbs,
        total=500,
    )
    selected = split_selected(selected, clean_verbs=clean_verbs, corrupt_verbs=corrupt_verbs, seed=int(args.seed) + 1)
    summary = write_dataset(
        destination,
        candidates=candidates,
        screening=screening,
        selected=selected,
        tool_token_id=tool_token_id,
        candidates_per_cell=int(args.candidates_per_cell),
        selected_clean_verbs=clean_verbs,
        selected_corrupt_verbs=corrupt_verbs,
        seed=int(args.seed),
        min_clean_margin=float(args.min_clean_margin),
        min_corrupt_non_tool_margin=float(args.min_corrupt_non_tool_margin),
        rejected=rejected,
    )
    print(
        f"[{SPEC.key}] complete: {summary['n_train']} train / {summary['n_heldout']} held-out, "
        f"clean={summary['all']['clean_tool_call_top1_rate']:.1%}, "
        f"corrupt={summary['all']['corrupt_tool_call_top1_rate']:.1%}",
        flush=True,
    )


if __name__ == "__main__":
    main()
