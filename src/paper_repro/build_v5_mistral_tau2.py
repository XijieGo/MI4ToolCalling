#!/usr/bin/env python3
"""Build a native-tokenized, behavior-screened Mistral v5 dataset.

Mistral's local tokenizer cannot losslessly round-trip its chat template through
plain text: special control tokens such as ``[AVAILABLE_TOOLS]`` are split by a
subsequent normal encode call.  This builder therefore stores each prompt as
the native chat messages, tool schema, and exact input IDs used for screening.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch

if __package__:
    from .build_v5_model_specific_balanced import (
        DEFAULT_OUTPUT_ROOT,
        MODEL_SPECS,
        choose_largest_feasible_vocabulary,
        forward_logits,
        load_model_and_tokenizer,
        model_input_device,
        release_model,
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
        forward_logits,
        load_model_and_tokenizer,
        model_input_device,
        release_model,
        split_selected,
        summary_for_rows,
        write_csv,
        write_json,
        write_jsonl,
    )


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SPEC = MODEL_SPECS["mistral_3p2_24b"]
SOURCE_ROOT = PROJECT_ROOT / "datasets" / "external" / "tau2_telecom_qwen35_9b"
SOURCE_ROWS = SOURCE_ROOT / "prepared" / "text_reply_induction" / "candidate_prefixes.jsonl"
SYSTEM_POLICY = (
    "You are a telecom support agent. Use the provided function when the user asks you to "
    "perform an account action. When the user asks only to discuss, explain, review, or summarize "
    "an action, respond in text without using a function."
)
TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "perform_account_action",
        "description": "Perform the appropriate account action requested by the customer. Use only for an action request, not for a request to discuss or explain.",
        "parameters": {
            "type": "object",
            "properties": {"request": {"type": "string", "description": "The requested account action."}},
            "required": ["request"],
        },
    },
}
CLEAN_VERBS = ("execute", "perform", "complete", "process")
CORRUPT_VERBS = ("discuss", "explain", "review", "summarize")
FINAL_REQUEST = "Please {verb} the next appropriate account action now."


@dataclass(frozen=True)
class MistralCandidate:
    candidate_id: str
    source_candidate_id: str
    task_id: str
    trace_index: int
    clean_verb: str
    corrupt_verb: str
    clean_messages: list[dict[str, str]]
    corrupt_messages: list[dict[str, str]]
    clean_ids: list[int]
    corrupt_ids: list[int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--candidates-per-cell", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-prompt-tokens", type=int, default=2048)
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--min-clean-margin", type=float, default=0.0)
    parser.add_argument("--min-corrupt-non-tool-margin", type=float, default=0.0)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def hash_ids(ids: Sequence[int]) -> str:
    payload = json.dumps(list(ids), separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def load_sources() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in read_jsonl(SOURCE_ROWS):
        source_id = str(row.get("candidate_id") or "")
        task_id = str(row.get("task_id") or "")
        messages = row.get("messages")
        if (
            not source_id
            or not task_id
            or source_id in seen
            or not isinstance(messages, list)
            or len(messages) < 2
            or str(messages[-1].get("role")) != "user"
            or str(messages[-2].get("role")) != "assistant"
        ):
            continue
        seen.add(source_id)
        rows.append(row)
    if len({str(row["task_id"]) for row in rows}) < 500:
        raise RuntimeError("The Mistral TAU2 source pool does not contain 500 distinct tasks")
    return rows


def native_messages(row: dict[str, Any], verb: str) -> list[dict[str, str]]:
    # The prepared text-reply pool ends with an assistant/user exchange and
    # contains no calls in those two final turns, so it can be made into a
    # compact valid Mistral chat without importing unrelated prior history.
    history = [
        {"role": str(message["role"]), "content": str(message.get("content") or "")}
        for message in row["messages"][-2:]
    ]
    history[-1]["content"] += "\n\n" + FINAL_REQUEST.format(verb=verb)
    return [{"role": "system", "content": SYSTEM_POLICY}, *history]


def render_ids(messages: Sequence[dict[str, str]], *, tokenizer: Any) -> list[int]:
    encoded = tokenizer.apply_chat_template(
        list(messages),
        tools=[TOOL_SCHEMA],
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    ids = encoded["input_ids"][0]
    return [int(value) for value in ids.detach().cpu().tolist()]


def build_candidates(
    sources: Sequence[dict[str, Any]],
    *,
    tokenizer: Any,
    candidates_per_cell: int,
    max_prompt_tokens: int,
    seed: int,
) -> tuple[list[MistralCandidate], list[dict[str, Any]]]:
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in sources:
        by_task[str(row["task_id"])].append(row)
    randomizer = random.Random(seed)
    task_ids = list(by_task)
    randomizer.shuffle(task_ids)
    cells = [(clean, corrupt) for clean in CLEAN_VERBS for corrupt in CORRUPT_VERBS]
    needed = len(cells) * candidates_per_cell
    if needed > len(task_ids):
        raise ValueError(f"Need {needed} task-disjoint source prompts, only {len(task_ids)} tasks are available")
    candidates: list[MistralCandidate] = []
    rejected: list[dict[str, Any]] = []
    cursor = 0
    for _ in range(candidates_per_cell):
        for clean_verb, corrupt_verb in cells:
            while cursor < len(task_ids):
                task_id = task_ids[cursor]
                cursor += 1
                options = list(by_task[task_id])
                randomizer.shuffle(options)
                for row in options:
                    try:
                        clean_messages = native_messages(row, clean_verb)
                        corrupt_messages = native_messages(row, corrupt_verb)
                        clean_ids = render_ids(clean_messages, tokenizer=tokenizer)
                        corrupt_ids = render_ids(corrupt_messages, tokenizer=tokenizer)
                        if max(len(clean_ids), len(corrupt_ids)) > max_prompt_tokens:
                            raise ValueError(f"prompt_too_long:{len(clean_ids)}/{len(corrupt_ids)}")
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
                        MistralCandidate(
                            candidate_id=f"{source_id}__{clean_verb}__{corrupt_verb}",
                            source_candidate_id=source_id,
                            task_id=task_id,
                            trace_index=int(row.get("trace_index") or -1),
                            clean_verb=clean_verb,
                            corrupt_verb=corrupt_verb,
                            clean_messages=clean_messages,
                            corrupt_messages=corrupt_messages,
                            clean_ids=clean_ids,
                            corrupt_ids=corrupt_ids,
                        )
                    )
                    break
                else:
                    continue
                break
            else:
                raise RuntimeError(f"Ran out of valid task-disjoint sources after {len(candidates)} candidates")
    return candidates, rejected


def evaluate_ids(
    model: Any,
    sequences: Sequence[Sequence[int]],
    *,
    tokenizer: Any,
    tool_token_id: int,
    batch_size: int,
    label: str,
) -> list[dict[str, Any]]:
    if tokenizer.pad_token_id is None:
        raise ValueError("Mistral tokenizer has no pad token")
    device = model_input_device(model)
    results: list[dict[str, Any]] = []
    total_batches = (len(sequences) + batch_size - 1) // batch_size
    for batch_index, start in enumerate(range(0, len(sequences), batch_size), start=1):
        batch_sequences = sequences[start : start + batch_size]
        width = max(len(sequence) for sequence in batch_sequences)
        input_ids = torch.full((len(batch_sequences), width), int(tokenizer.pad_token_id), dtype=torch.long)
        attention_mask = torch.zeros_like(input_ids)
        for index, sequence in enumerate(batch_sequences):
            values = torch.tensor(sequence, dtype=torch.long)
            input_ids[index, -len(sequence) :] = values
            attention_mask[index, -len(sequence) :] = 1
        with torch.inference_mode():
            logits = forward_logits(model, input_ids.to(device), attention_mask.to(device))
        target = logits[:, tool_token_id]
        top1 = logits.argmax(dim=-1)
        ranks = (logits > target.unsqueeze(-1)).sum(dim=-1) + 1
        non_target = logits.clone()
        non_target[:, tool_token_id] = -torch.inf
        best_non_target = non_target.max(dim=-1).values
        probabilities = torch.softmax(logits, dim=-1)[:, tool_token_id]
        for index in range(logits.shape[0]):
            token_id = int(top1[index].item())
            results.append(
                {
                    "tool_logit": float(target[index].item()),
                    "tool_probability": float(probabilities[index].item()),
                    "tool_rank": int(ranks[index].item()),
                    "top1_token_id": token_id,
                    "top1_token_text": tokenizer.decode([token_id], clean_up_tokenization_spaces=False),
                    "is_tool_top1": bool(token_id == tool_token_id),
                    "margin_vs_best_non_tool": float((target[index] - best_non_target[index]).item()),
                }
            )
        if batch_index == total_batches or batch_index % max(total_batches // 10, 1) == 0:
            print(f"{label}: {batch_index}/{total_batches} batches", flush=True)
    return results


def screen_candidates(
    candidates: Sequence[MistralCandidate], *, model: Any, tokenizer: Any, tool_token_id: int, batch_size: int
) -> list[dict[str, Any]]:
    clean_metrics = evaluate_ids(
        model,
        [candidate.clean_ids for candidate in candidates],
        tokenizer=tokenizer,
        tool_token_id=tool_token_id,
        batch_size=batch_size,
        label="mistral clean screening",
    )
    corrupt_metrics = evaluate_ids(
        model,
        [candidate.corrupt_ids for candidate in candidates],
        tokenizer=tokenizer,
        tool_token_id=tool_token_id,
        batch_size=batch_size,
        label="mistral corrupt screening",
    )
    rows: list[dict[str, Any]] = []
    for candidate, clean, corrupt in zip(candidates, clean_metrics, corrupt_metrics, strict=True):
        rows.append(
            {
                "candidate_id": candidate.candidate_id,
                "source_sample_id": candidate.source_candidate_id,
                "source_filename": f"{candidate.source_candidate_id}.jsonl",
                "source_dataset": "tau2_telecom",
                "source_language": "en",
                "source_split": "natural_text_reply_history",
                "source_task_id": candidate.task_id,
                "source_trace_index": candidate.trace_index,
                "clean_verb": candidate.clean_verb,
                "corrupt_verb": candidate.corrupt_verb,
                "clean_prompt_sha256": hash_ids(candidate.clean_ids),
                "corrupt_prompt_sha256": hash_ids(candidate.corrupt_ids),
                **{f"clean_{key}": value for key, value in clean.items()},
                **{f"corrupt_{key}": value for key, value in corrupt.items()},
                "behavior_valid": bool(clean["is_tool_top1"]) and not bool(corrupt["is_tool_top1"]),
                "quality": float(clean["margin_vs_best_non_tool"]) - float(corrupt["margin_vs_best_non_tool"]),
            }
        )
    return rows


def prompt_payload(messages: Sequence[dict[str, str]], input_ids: Sequence[int]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "prompt_format": "mistral_native_input_ids",
        "renderer": "AutoTokenizer.apply_chat_template(tokenize=True, add_generation_prompt=True)",
        "messages": list(messages),
        "tools": [TOOL_SCHEMA],
        "input_ids": list(input_ids),
        "input_ids_sha256": hash_ids(input_ids),
    }


def write_dataset(
    destination: Path,
    *,
    candidates: Sequence[MistralCandidate],
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
        filename = f"{index:03d}_{candidate.source_candidate_id}.json"
        clean_path = destination / split / "clean" / filename
        corrupt_path = destination / split / "corrupt" / filename
        clean_path.write_text(json.dumps(prompt_payload(candidate.clean_messages, candidate.clean_ids), ensure_ascii=False) + "\n", encoding="utf-8")
        corrupt_path.write_text(json.dumps(prompt_payload(candidate.corrupt_messages, candidate.corrupt_ids), ensure_ascii=False) + "\n", encoding="utf-8")
        manifest.append(
            {
                "sample_id": f"v5_mistral_3p2_24b_{index:03d}",
                "split": split,
                "filename": filename,
                "clean_relpath": str(clean_path.relative_to(destination)),
                "corrupt_relpath": str(corrupt_path.relative_to(destination)),
                "prompt_format": "mistral_native_input_ids",
                "source_sample_id": candidate.source_candidate_id,
                "source_dataset": "tau2_telecom",
                "source_task_id": candidate.task_id,
                "source_trace_index": candidate.trace_index,
                "source_candidate_file": str(SOURCE_ROWS),
                "template": "two_turn_tau2_history_plus_single_final_verb_edit",
                "clean_candidate": candidate.clean_verb,
                "corrupt_candidate": candidate.corrupt_verb,
                "clean_token_count": len(candidate.clean_ids),
                "corrupt_token_count": len(candidate.corrupt_ids),
                "tool_call_marker": SPEC.tool_marker,
                "tool_call_token_id": tool_token_id,
                **{key: value for key, value in row.items() if key not in {"split", "source_sample_id", "source_filename", "source_dataset", "source_language", "source_split", "source_task_id", "source_trace_index"}},
            }
        )
    write_jsonl(destination / "manifest.jsonl", manifest)
    write_csv(destination / "candidate_screening.csv", list(screening))
    if rejected:
        write_jsonl(destination / "candidate_construction_rejections.jsonl", rejected)
    split_summaries = {split: summary_for_rows([row for row in selected if row["split"] == split]) for split in ("train", "heldout")}
    for split, summary in split_summaries.items():
        if summary["clean_tool_call_top1_rate"] != 1.0 or summary["corrupt_tool_call_top1_rate"] != 0.0:
            raise AssertionError(f"{split} contains a behavior-invalid selected pair")
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
        "template": "two_turn_tau2_history_plus_single_final_verb_edit",
        "prompt_format": "mistral_native_input_ids",
        "n_pairs": len(selected),
        "n_train": sum(row["split"] == "train" for row in selected),
        "n_heldout": sum(row["split"] == "heldout" for row in selected),
        "all": summary_for_rows(selected),
        "splits": split_summaries,
    }
    write_json(destination / "summary.json", summary)
    (destination / "README.md").write_text(
        "\n".join(
            [
                "# Mistral-Small-3.2-24B v5 dataset",
                "",
                "This model-specific TAU2 dataset is stored as native Mistral token IDs.",
                "",
                f"- Pairs: `{summary['n_pairs']}` (`{summary['n_train']}` train, `{summary['n_heldout']}` held-out).",
                f"- Native first-token call marker: `{SPEC.tool_marker}` (ID `{tool_token_id}`).",
                "- Each paired prompt changes only the final action/discussion verb.",
                "- Every prompt JSON contains the original messages, tool schema, and exact native `input_ids`.",
                "- Use the stored IDs directly or re-render from `messages` and `tools` with `apply_chat_template(tokenize=True)`.",
                "- Do not encode a decoded plain-text rendering: this backend does not round-trip special tokens losslessly.",
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
    output_root = args.output_root.resolve()
    destination = output_root / SPEC.key
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {destination}")
    sources = load_sources()
    print(f"[{SPEC.key}] loading {SPEC.label}; {len(sources)} native TAU2 source states", flush=True)
    model, tokenizer, tool_token_id = load_model_and_tokenizer(SPEC, str(args.dtype))
    try:
        candidates, rejected = build_candidates(
            sources,
            tokenizer=tokenizer,
            candidates_per_cell=int(args.candidates_per_cell),
            max_prompt_tokens=int(args.max_prompt_tokens),
            seed=int(args.seed),
        )
        print(f"[{SPEC.key}] rendered {len(candidates)} candidates; {len(rejected)} source states skipped", flush=True)
        screening = screen_candidates(
            candidates,
            model=model,
            tokenizer=tokenizer,
            tool_token_id=tool_token_id,
            batch_size=int(args.batch_size),
        )
    finally:
        release_model(model)
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
        preferred_clean_verbs=CLEAN_VERBS,
        preferred_corrupt_verbs=CORRUPT_VERBS,
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
