#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
from collections import Counter, defaultdict
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer, Mistral3ForConditionalGeneration
from transformers.utils import logging as transformers_logging
from transformers.utils.quantization_config import FineGrainedFP8Config

try:
    transformers_logging.disable_progress_bar()
except Exception:
    pass


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = Path(
    os.environ.get(
        "DEVSTRAL_2_24B_PATH",
        str(PROJECT_ROOT / "external" / "models" / "Devstral-Small-2-24B-Instruct-2512"),
    )
)
DEFAULT_CANONICAL_PATH = PROJECT_ROOT / "results" / "Devstral-Small-2-24B-Instruct-2512" / "converted_dataset" / "canonical_pairs.jsonl"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "results" / "Devstral-Small-2-24B-Instruct-2512" / "datasets"
DEFAULT_VIBE_SYSTEM_PROMPT_FILE = PROJECT_ROOT / "configs" / "prompts" / "mistral_vibe_system_prompt.txt"
DEFAULT_CLEAN_VERBS = ("save", "update", "write", "add", "complete")
DEFAULT_CORRUPT_VERBS = ("reason", "describe", "think", "outline", "discuss")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a behavior-valid balanced Devstral VIBE subset.")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--canonical-path", type=Path, default=DEFAULT_CANONICAL_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--system-prompt-file",
        type=Path,
        default=DEFAULT_VIBE_SYSTEM_PROMPT_FILE,
        help="Versioned VIBE system prompt used for the behavior-selection render.",
    )
    parser.add_argument("--clean-verbs", nargs="*", default=list(DEFAULT_CLEAN_VERBS))
    parser.add_argument("--corrupt-verbs", nargs="*", default=list(DEFAULT_CORRUPT_VERBS))
    parser.add_argument(
        "--target-pairs",
        type=int,
        default=0,
        help="Exact requested size, divisible by clean_verbs × corrupt_verbs; 0 finds the largest complete balanced grid.",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--device-map", type=str, default="auto")
    parser.add_argument("--gpu-max-memory", type=str, default="60GiB")
    parser.add_argument("--cpu-max-memory", type=str, default="400GiB")
    parser.add_argument("--today", type=str, default="")
    parser.add_argument("--overwrite", action="store_true", help="Replace only this explicit output directory.")
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def resolve_system_prompt(path: Path, today_override: str = "") -> str:
    today_value = date.fromisoformat(today_override) if today_override else date.today()
    yesterday_value = today_value - timedelta(days=1)
    template = path.read_text(encoding="utf-8")
    return template.replace("{today}", today_value.isoformat()).replace("{yesterday}", yesterday_value.isoformat())


def build_model(model_path: Path, dtype_name: str, device_map: str, gpu_max_memory: str, cpu_max_memory: str):
    dtype = getattr(torch, dtype_name)
    extra_kwargs: dict[str, Any] = {}
    if gpu_max_memory or cpu_max_memory:
        max_memory: dict[Any, str] = {}
        if gpu_max_memory:
            max_memory[0] = gpu_max_memory
        if cpu_max_memory:
            max_memory["cpu"] = cpu_max_memory
        extra_kwargs["max_memory"] = max_memory
    # Devstral is stored with fine-grained FP8 weights.  Dequantizing at load
    # time keeps this behavior-only scan independent of the optional Triton
    # FP8 kernel, matching the repository's shared Devstral loader.
    extra_kwargs["quantization_config"] = FineGrainedFP8Config(dequantize=True)
    try:
        model = Mistral3ForConditionalGeneration.from_pretrained(
            str(model_path),
            dtype=dtype,
            device_map=device_map,
            trust_remote_code=True,
            **extra_kwargs,
        )
    except TypeError:
        model = Mistral3ForConditionalGeneration.from_pretrained(
            str(model_path),
            torch_dtype=dtype,
            device_map=device_map,
            trust_remote_code=True,
            **extra_kwargs,
        )
    model.eval()
    return model


def extract_shared_user_template(clean_text: str, corrupt_text: str) -> tuple[str, str, str]:
    clean_lines = clean_text.splitlines()
    corrupt_lines = corrupt_text.splitlines()
    if not clean_lines or not corrupt_lines:
        raise ValueError("Empty user text.")

    pattern = re.compile(r"^(\s*)(\S+)(.*)$")
    clean_match = pattern.match(clean_lines[0])
    corrupt_match = pattern.match(corrupt_lines[0])
    if clean_match is None or corrupt_match is None:
        raise ValueError("Failed to parse first line.")

    clean_prefix, _clean_verb, clean_suffix = clean_match.groups()
    corrupt_prefix, _corrupt_verb, corrupt_suffix = corrupt_match.groups()
    if clean_prefix != corrupt_prefix or clean_suffix != corrupt_suffix:
        raise ValueError("First line differs beyond the leading verb.")

    clean_rest = "\n".join(clean_lines[1:])
    corrupt_rest = "\n".join(corrupt_lines[1:])
    if clean_rest != corrupt_rest:
        raise ValueError("User contents differ beyond the first-line verb.")
    return clean_prefix, clean_suffix, clean_rest


def render_user_from_template(prefix: str, suffix: str, rest: str, verb: str) -> str:
    first_line = prefix + verb[:1].upper() + verb[1:] + suffix
    if rest:
        return first_line + "\n" + rest
    return first_line


def collate_batch(tokenizer, items: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    return tokenizer.pad(
        {
            "input_ids": [item["input_ids"] for item in items],
            "attention_mask": [item["attention_mask"] for item in items],
        },
        padding=True,
        return_tensors="pt",
    )


def evaluate_prompts(model, tokenizer, prompts: list[dict[str, Any]], tool_token_id: int, batch_size: int) -> list[dict[str, Any]]:
    device = next(model.parameters()).device
    outputs: list[dict[str, Any]] = []
    for start in range(0, len(prompts), batch_size):
        batch_items = prompts[start : start + batch_size]
        batch = collate_batch(tokenizer, batch_items)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        with torch.no_grad():
            result = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                logits_to_keep=1,
            )
        logits = result.logits[:, -1, :].float()
        probs = torch.softmax(logits, dim=-1)
        top1 = logits.argmax(dim=-1)
        for idx, item in enumerate(batch_items):
            top1_id = int(top1[idx].item())
            outputs.append(
                {
                    **item,
                    "tool_token_id": tool_token_id,
                    "tool_token_prob": float(probs[idx, tool_token_id].item()),
                    "tool_token_logit": float(logits[idx, tool_token_id].item()),
                    "top1_token_id": top1_id,
                    "top1_token_text": tokenizer.decode([top1_id], clean_up_tokenization_spaces=False),
                    "is_tool_call_top1": bool(top1_id == tool_token_id),
                }
            )
    return outputs


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def prepare_output_root(output_root: Path, *, overwrite: bool) -> Path:
    """Create one explicit selection directory without mixing stale prompts."""

    resolved = output_root.resolve()
    forbidden = {PROJECT_ROOT.resolve(), (PROJECT_ROOT / "datasets").resolve(), (PROJECT_ROOT / "results").resolve()}
    if resolved in forbidden:
        raise ValueError(f"Refusing broad output directory: {resolved}")
    if resolved.exists() and any(resolved.iterdir()):
        if not overwrite:
            raise FileExistsError(f"Output already exists: {resolved}; pass --overwrite for this exact directory.")
        shutil.rmtree(resolved)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def exact_balanced_selection(
    candidates_by_combo: dict[tuple[str, str], list[dict[str, Any]]],
    combos: list[tuple[str, str]],
    *,
    per_combo: int,
) -> list[dict[str, Any]] | None:
    """Find a behavior-valid selection with one source example per pair.

    A source prompt can pass several verb combinations.  Treating each cell
    independently would reuse it and silently overstate the fresh subset size.
    This deterministic bipartite matching gives every clean×corrupt cell the
    requested capacity while each source ``sample_id`` has capacity one.
    """

    if per_combo <= 0:
        return []
    if any(len(candidates_by_combo[combo]) < per_combo for combo in combos):
        return None

    slots = [(combo, slot_index) for combo in combos for slot_index in range(per_combo)]
    slots.sort(key=lambda item: (len(candidates_by_combo[item[0]]), item[0], item[1]))
    assigned_candidate: dict[tuple[tuple[str, str], int], dict[str, Any]] = {}
    assigned_slot_by_sample: dict[str, tuple[tuple[str, str], int]] = {}

    def assign(slot: tuple[tuple[str, str], int], seen_samples: set[str]) -> bool:
        combo, _slot_index = slot
        for candidate in candidates_by_combo[combo]:
            sample_id = str(candidate["sample_id"])
            if sample_id in seen_samples:
                continue
            seen_samples.add(sample_id)
            occupied_slot = assigned_slot_by_sample.get(sample_id)
            if occupied_slot is None or assign(occupied_slot, seen_samples):
                assigned_candidate[slot] = candidate
                assigned_slot_by_sample[sample_id] = slot
                return True
        return False

    for slot in slots:
        if not assign(slot, set()):
            return None
    return [assigned_candidate[slot] for slot in slots]


def choose_largest_complete_grid(
    candidates_by_combo: dict[tuple[str, str], list[dict[str, Any]]],
    combos: list[tuple[str, str]],
) -> tuple[int, list[dict[str, Any]]]:
    upper = min(len(candidates_by_combo[combo]) for combo in combos)
    low, high = 0, upper
    while low < high:
        middle = (low + high + 1) // 2
        if exact_balanced_selection(candidates_by_combo, combos, per_combo=middle) is None:
            high = middle - 1
        else:
            low = middle
    selected = exact_balanced_selection(candidates_by_combo, combos, per_combo=low)
    if not selected:
        raise RuntimeError("No complete behavior-valid clean×corrupt verb grid can be selected")
    return low, selected


def main() -> None:
    args = parse_args()
    if args.target_pairs < 0:
        raise ValueError("--target-pairs must be non-negative")
    args.output_root = prepare_output_root(args.output_root, overwrite=args.overwrite)
    clean_output_root = args.output_root / "clean"
    corrupt_output_root = args.output_root / "corrupt"
    clean_output_root.mkdir(parents=True, exist_ok=True)
    corrupt_output_root.mkdir(parents=True, exist_ok=True)

    canonical_rows = load_jsonl(args.canonical_path)
    system_prompt = resolve_system_prompt(args.system_prompt_file.expanduser().resolve(), args.today)
    tokenizer = AutoTokenizer.from_pretrained(str(args.model_path), trust_remote_code=True)
    tool_token_ids = tokenizer.encode("[TOOL_CALLS]", add_special_tokens=False)
    if len(tool_token_ids) != 1:
        raise RuntimeError(f"[TOOL_CALLS] should map to one token, got {tool_token_ids}")
    tool_token_id = int(tool_token_ids[0])

    prepared_rows: list[dict[str, Any]] = []
    for row in canonical_rows:
        try:
            prefix, suffix, rest = extract_shared_user_template(
                row["clean_user_content"],
                row["corrupt_user_content"],
            )
        except ValueError:
            continue
        prepared_rows.append(
            {
                **row,
                "user_prefix": prefix,
                "user_suffix": suffix,
                "user_rest": rest,
            }
        )

    model = build_model(
        args.model_path,
        args.dtype,
        args.device_map,
        args.gpu_max_memory,
        args.cpu_max_memory,
    )

    clean_scores: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    corrupt_scores: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)

    for verb, target_store in [(verb, clean_scores) for verb in args.clean_verbs] + [(verb, corrupt_scores) for verb in args.corrupt_verbs]:
        prompt_rows: list[dict[str, Any]] = []
        for row in prepared_rows:
            rendered_user = render_user_from_template(
                row["user_prefix"],
                row["user_suffix"],
                row["user_rest"],
                verb,
            )
            prompt_text = tokenizer.apply_chat_template(
                conversation=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": rendered_user},
                ],
                tools=row["tools_schema"],
                tokenize=False,
                add_generation_prompt=True,
            )
            encoded = tokenizer(prompt_text, add_special_tokens=False, return_tensors="pt")
            prompt_rows.append(
                {
                    "sample_id": row["sample_id"],
                    "split": row["split"],
                    "language": row["language"],
                    "candidate_verb": verb,
                    "prompt_text": prompt_text,
                    "rendered_user": rendered_user,
                    "input_ids": encoded["input_ids"][0],
                    "attention_mask": encoded["attention_mask"][0],
                }
            )
        outputs = evaluate_prompts(model, tokenizer, prompt_rows, tool_token_id, args.batch_size)
        for item in outputs:
            target_store[item["sample_id"]][verb] = item
        print(
            f"scored verb={verb} side={'clean' if target_store is clean_scores else 'corrupt'} "
            f"tool_rate={sum(int(x['is_tool_call_top1']) for x in outputs) / len(outputs):.4f}",
            flush=True,
        )

    # Preserve the full behavior screen before selecting a balanced subset.
    # Figure 1 reports behavioral rates; using only the subsequently selected
    # clean-only pairs would mechanically turn that figure into 100% / 0%.
    behavior_screen_rows: list[dict[str, Any]] = []
    for pool, verbs, score_store in (
        ("clean", args.clean_verbs, clean_scores),
        ("corrupt", args.corrupt_verbs, corrupt_scores),
    ):
        for row in prepared_rows:
            for verb in verbs:
                score = score_store[row["sample_id"]][verb]
                behavior_screen_rows.append(
                    {
                        "sample_id": row["sample_id"],
                        "split": row["split"],
                        "language": row["language"],
                        "pool": pool,
                        "candidate": verb,
                        "tool_token_id": score["tool_token_id"],
                        "tool_token_prob": score["tool_token_prob"],
                        "tool_token_logit": score["tool_token_logit"],
                        "top1_token_id": score["top1_token_id"],
                        "top1_token_text": score["top1_token_text"],
                        "is_tool_call_top1": score["is_tool_call_top1"],
                    }
                )
    write_csv(args.output_root / "behavior_screen.csv", behavior_screen_rows)

    combos = [(clean_verb, corrupt_verb) for clean_verb in args.clean_verbs for corrupt_verb in args.corrupt_verbs]

    candidates_by_combo: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for clean_verb, corrupt_verb in combos:
        combo_rows: list[dict[str, Any]] = []
        for row in prepared_rows:
            sample_id = row["sample_id"]
            clean_result = clean_scores[sample_id][clean_verb]
            corrupt_result = corrupt_scores[sample_id][corrupt_verb]
            if not clean_result["is_tool_call_top1"]:
                continue
            if corrupt_result["is_tool_call_top1"]:
                continue
            combo_rows.append(
                {
                    "sample_id": sample_id,
                    "language": row["language"],
                    "split": row["split"],
                    "clean_verb": clean_verb,
                    "corrupt_verb": corrupt_verb,
                    "clean_prompt_text": clean_result["prompt_text"],
                    "corrupt_prompt_text": corrupt_result["prompt_text"],
                    "clean_rendered_user": clean_result["rendered_user"],
                    "corrupt_rendered_user": corrupt_result["rendered_user"],
                    "clean_tool_prob": clean_result["tool_token_prob"],
                    "corrupt_tool_prob": corrupt_result["tool_token_prob"],
                    "clean_top1_token_text": clean_result["top1_token_text"],
                    "corrupt_top1_token_text": corrupt_result["top1_token_text"],
                    "score_margin": clean_result["tool_token_prob"] - corrupt_result["tool_token_prob"],
                }
            )
        combo_rows.sort(key=lambda item: (-item["score_margin"], -item["clean_tool_prob"], item["corrupt_tool_prob"], item["sample_id"]))
        candidates_by_combo[(clean_verb, corrupt_verb)] = combo_rows

    if args.target_pairs:
        if args.target_pairs % len(combos) != 0:
            raise ValueError("target_pairs must be divisible by clean_verbs * corrupt_verbs for a complete balanced grid.")
        per_combo_target = args.target_pairs // len(combos)
        selected_rows = exact_balanced_selection(
            candidates_by_combo,
            combos,
            per_combo=per_combo_target,
        )
        if selected_rows is None:
            raise RuntimeError(
                f"Could not build the requested complete behavior-valid grid: "
                f"{per_combo_target} pairs per each of {len(combos)} verb cells."
            )
    else:
        per_combo_target, selected_rows = choose_largest_complete_grid(candidates_by_combo, combos)

    selected_rows.sort(key=lambda item: (item["clean_verb"], item["corrupt_verb"], item["sample_id"]))
    combo_counts = Counter((row["clean_verb"], row["corrupt_verb"]) for row in selected_rows)
    clean_counts = Counter(row["clean_verb"] for row in selected_rows)
    corrupt_counts = Counter(row["corrupt_verb"] for row in selected_rows)
    manifest_rows: list[dict[str, Any]] = []
    for pair_index, row in enumerate(selected_rows, start=1):
        clean_path = clean_output_root / f"clean_{pair_index}.txt"
        corrupt_path = corrupt_output_root / f"corrupt_{pair_index}.txt"
        clean_path.write_text(row["clean_prompt_text"], encoding="utf-8")
        corrupt_path.write_text(row["corrupt_prompt_text"], encoding="utf-8")
        manifest_rows.append(
            {
                "pair_id": pair_index,
                "sample_id": row["sample_id"],
                "split": row["split"],
                "language": row["language"],
                "clean_verb": row["clean_verb"],
                "corrupt_verb": row["corrupt_verb"],
                "clean_path": str(clean_path.resolve()),
                "corrupt_path": str(corrupt_path.resolve()),
                "clean_tool_prob": row["clean_tool_prob"],
                "corrupt_tool_prob": row["corrupt_tool_prob"],
                "clean_top1_token_text": row["clean_top1_token_text"],
                "corrupt_top1_token_text": row["corrupt_top1_token_text"],
                "score_margin": row["score_margin"],
            }
        )

    write_jsonl(args.output_root / "manifest.jsonl", manifest_rows)
    write_csv(args.output_root / "manifest.csv", manifest_rows)

    combo_summary_rows = [
        {
            "clean_verb": clean_verb,
            "corrupt_verb": corrupt_verb,
            "selected_pairs": combo_counts[(clean_verb, corrupt_verb)],
            "available_valid_candidates": len(candidates_by_combo[(clean_verb, corrupt_verb)]),
        }
        for clean_verb, corrupt_verb in combos
    ]
    write_csv(args.output_root / "combo_summary.csv", combo_summary_rows)

    summary = {
        "target_pairs_requested": args.target_pairs,
        "target_pairs": len(manifest_rows),
        "selected_pairs": len(manifest_rows),
        "system_prompt_file": str(args.system_prompt_file.expanduser().resolve()),
        "clean_verbs": list(args.clean_verbs),
        "corrupt_verbs": list(args.corrupt_verbs),
        "per_combo_target": per_combo_target,
        "clean_counts": dict(clean_counts),
        "corrupt_counts": dict(corrupt_counts),
        "language_counts": dict(Counter(row["language"] for row in selected_rows)),
        "split_counts": dict(Counter(row["split"] for row in selected_rows)),
        "behavior_screen_csv": str((args.output_root / "behavior_screen.csv").resolve()),
    }
    (args.output_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.output_root / "summary.md").write_text(
        "\n".join(
            [
                "# Devstral Behavior-Validated Balanced Dataset",
                "",
                f"- selected pairs: {len(manifest_rows)}",
                f"- system prompt: `{args.system_prompt_file}`",
                f"- clean verbs: {', '.join(args.clean_verbs)}",
                f"- corrupt verbs: {', '.join(args.corrupt_verbs)}",
                f"- language counts: {dict(Counter(row['language'] for row in selected_rows))}",
                f"- split counts: {dict(Counter(row['split'] for row in selected_rows))}",
                "",
                "## Clean Verb Counts",
                "",
            ]
            + [f"- `{verb}`: {clean_counts[verb]}" for verb in args.clean_verbs]
            + [
                "",
                "## Corrupt Verb Counts",
                "",
            ]
            + [f"- `{verb}`: {corrupt_counts[verb]}" for verb in args.corrupt_verbs]
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
