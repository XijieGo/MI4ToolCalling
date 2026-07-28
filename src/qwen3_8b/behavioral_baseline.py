#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import random
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch
from tqdm.auto import tqdm

LEGACY_SRC = Path("./src")
if str(LEGACY_SRC) not in sys.path:
    sys.path.insert(0, str(LEGACY_SRC))

from toolcall_circuit.single_sample import load_hooked_qwen3  # noqa: E402


SEED = 42
MAX_SAMPLES = 100
MODEL_PATH = Path("./external/models/Qwen3-8B")
DATASET_DIR = Path("./datasets/train/clean")
OUTPUT_ROOT = Path("./results/8B/behavioral_baseline")
TOOL_CALL_STR = "<tool_call>"
NEUTRAL_VERB = "Process"
NO_TOOL_SYSTEM_TEXT = "You are a helpful assistant."

SYSTEM_RE = re.compile(r"(<\|im_start\|>system\n).*?(<\|im_end\|>)", re.DOTALL)
USER_MARKER = "<|im_start|>user\n"

CONDITION_ORDER = ["with_tools", "no_tools", "neutral_verb", "no_tools_neutral"]
EXPERIMENT_CONDITIONS = {
    "exp1_no_tools": ["with_tools", "no_tools"],
    "exp2_neutral_verb": ["with_tools", "neutral_verb"],
    "exp3_no_tools_neutral": ["with_tools", "no_tools_neutral"],
}
CONDITION_LABELS = {
    "with_tools": "原始 prompt（含 tool schema + 原始动词）",
    "no_tools": "去 tool schema",
    "neutral_verb": "保留 tool schema，首动词替换为 Process",
    "no_tools_neutral": "同时去 tool schema + 首动词替换为 Process",
}
@dataclass(frozen=True)
class PromptVariants:
    sample_id: str
    original_verb: str
    prompts: Dict[str, str]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: List[str] = []
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


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def replace_system_prompt(text: str) -> str:
    updated, count = SYSTEM_RE.subn(r"\1" + NO_TOOL_SYSTEM_TEXT + r"\2", text, count=1)
    if count != 1:
        raise ValueError("Failed to replace system prompt")
    return updated


def replace_first_user_verb(text: str, new_verb: str) -> tuple[str, str]:
    marker_pos = text.find(USER_MARKER)
    if marker_pos < 0:
        raise ValueError("Missing user marker")
    user_start = marker_pos + len(USER_MARKER)
    tail = text[user_start:]
    if not tail:
        raise ValueError("Empty user content")

    verb_end = 0
    while verb_end < len(tail) and not tail[verb_end].isspace():
        verb_end += 1
    if verb_end <= 0:
        raise ValueError("Failed to parse first user verb")

    original_verb = tail[:verb_end]
    updated = text[:user_start] + new_verb + tail[verb_end:]
    return updated, original_verb


def build_prompt_variants(sample_id: str, text: str) -> PromptVariants:
    neutral_text, original_verb = replace_first_user_verb(text, NEUTRAL_VERB)
    no_tools_text = replace_system_prompt(text)
    no_tools_neutral_text = replace_system_prompt(neutral_text)
    return PromptVariants(
        sample_id=sample_id,
        original_verb=original_verb,
        prompts={
            "with_tools": text,
            "no_tools": no_tools_text,
            "neutral_verb": neutral_text,
            "no_tools_neutral": no_tools_neutral_text,
        },
    )


def load_prompt_variants(dataset_dir: Path, max_samples: int) -> List[PromptVariants]:
    paths = sorted(dataset_dir.glob("*.txt"))[:max_samples]
    if len(paths) < max_samples:
        raise RuntimeError(f"Expected at least {max_samples} prompt files, found {len(paths)}")
    return [
        build_prompt_variants(path.stem, path.read_text(encoding="utf-8"))
        for path in paths
    ]


def decode_token(tokenizer, token_id: int) -> str:
    return tokenizer.decode([token_id], clean_up_tokenization_spaces=False)


def evaluate_prompt(model, tokenizer, sample_id: str, original_verb: str, condition: str, prompt: str, tool_token_id: int) -> Dict[str, object]:
    tokens = model.to_tokens(prompt, prepend_bos=False)
    with torch.inference_mode():
        logits = model(tokens)
    last_logits = logits[0, -1, :].detach().float().cpu()
    top1_id = int(last_logits.argmax().item())
    tool_logit = float(last_logits[tool_token_id].item())
    tool_rank = int((last_logits > last_logits[tool_token_id]).sum().item()) + 1
    return {
        "file": sample_id,
        "condition": condition,
        "top1_token": decode_token(tokenizer, top1_id),
        "top1_id": top1_id,
        "is_tool_call_top1": int(top1_id == tool_token_id),
        "tool_call_logit": tool_logit,
        "tool_call_rank": tool_rank,
        "original_verb": original_verb,
        "effective_verb": NEUTRAL_VERB if condition in {"neutral_verb", "no_tools_neutral"} else original_verb,
        "prompt_token_len": int(tokens.shape[-1]),
    }


def condition_metrics(rows: Sequence[Dict[str, object]]) -> Dict[str, float]:
    if not rows:
        raise ValueError("Cannot summarize empty rows")
    n = len(rows)
    top1_rate = sum(int(row["is_tool_call_top1"]) for row in rows) / n
    mean_logit = sum(float(row["tool_call_logit"]) for row in rows) / n
    mean_rank = sum(float(row["tool_call_rank"]) for row in rows) / n
    return {
        "n": float(n),
        "tool_call_top1_rate": top1_rate,
        "mean_tool_call_logit": mean_logit,
        "mean_tool_call_rank": mean_rank,
    }


def percent(value: float) -> str:
    return f"{value * 100:.1f}%"


def fmt(value: float) -> str:
    return f"{value:.3f}"


def build_interpretation(metrics_by_condition: Dict[str, Dict[str, float]]) -> tuple[str, List[str]]:
    baseline = metrics_by_condition["with_tools"]
    no_tools = metrics_by_condition["no_tools"]
    neutral = metrics_by_condition["neutral_verb"]
    no_tools_neutral = metrics_by_condition["no_tools_neutral"]

    reasons: List[str] = []
    score_b = 0.0
    score_c = 0.0

    no_tools_drop = baseline["mean_tool_call_logit"] - no_tools["mean_tool_call_logit"]
    if no_tools["tool_call_top1_rate"] <= 0.05 and no_tools_drop >= 1.0:
        score_c += 1.5
        reasons.append(
            f"Exp1 去 schema 后 `<tool_call>` top-1 降到 {percent(no_tools['tool_call_top1_rate'])}，均值 logit 下降 {fmt(no_tools_drop)}，说明 schema 不是可有可无。"
        )
    elif no_tools["tool_call_top1_rate"] >= 0.20:
        score_b += 1.0
        reasons.append(
            f"Exp1 去 schema 后仍有 {percent(no_tools['tool_call_top1_rate'])} 样本把 `<tool_call>` 置为 top-1，说明动词本身保留了明显触发力。"
        )
    else:
        reasons.append(
            f"Exp1 去 schema 后 `<tool_call>` top-1 为 {percent(no_tools['tool_call_top1_rate'])}，但还需要结合 Exp2 判断究竟是默认态还是弱触发。"
        )

    neutral_rate = neutral["tool_call_top1_rate"]
    if neutral_rate >= 0.90:
        score_c += 2.0
        reasons.append(
            f"Exp2 中性动词条件下 `<tool_call>` top-1 仍有 {percent(neutral_rate)}，更像是 tool schema + 任务语境驱动的默认行为。"
        )
    elif 0.25 <= neutral_rate <= 0.75:
        score_b += 1.5
        reasons.append(
            f"Exp2 中性动词条件下 `<tool_call>` top-1 为 {percent(neutral_rate)}，接近任务书里描述的双向分类区间。"
        )
    elif neutral_rate <= 0.10:
        reasons.append(
            f"Exp2 中性动词条件下 `<tool_call>` top-1 只有 {percent(neutral_rate)}，说明首动词几乎是必要输入，但这也不完全吻合任务书中预设的 B/C 两种极简图景。"
        )
    else:
        score_c += 0.5
        reasons.append(
            f"Exp2 中性动词条件下 `<tool_call>` top-1 为 {percent(neutral_rate)}，没有掉到随机附近，更偏向默认态但仍保留部分动词敏感性。"
        )

    if no_tools_neutral["tool_call_top1_rate"] <= 0.05:
        score_c += 0.5
        reasons.append(
            f"Exp3 同时去 schema 和动词后 `<tool_call>` top-1 仅 {percent(no_tools_neutral['tool_call_top1_rate'])}，控制组与 C 假说一致。"
        )
    else:
        score_b += 0.5
        reasons.append(
            f"Exp3 同时去 schema 和动词后仍有 {percent(no_tools_neutral['tool_call_top1_rate'])} 的 `<tool_call>` top-1，说明存在超出 schema/动词的残余触发。"
        )

    if score_c > score_b:
        verdict = "偏 C"
    elif score_b > score_c:
        verdict = "偏 B"
    else:
        verdict = "暂不确定"
    return verdict, reasons


def build_summary(
    samples: Sequence[PromptVariants],
    metrics_by_condition: Dict[str, Dict[str, float]],
    *,
    model_path: Path,
    dataset_dir: Path,
) -> str:
    baseline = metrics_by_condition["with_tools"]
    verdict, reasons = build_interpretation(metrics_by_condition)
    verb_counts = Counter(sample.original_verb for sample in samples)
    verb_summary = ", ".join(f"{verb}:{count}" for verb, count in sorted(verb_counts.items()))

    lines = [
        "# Phase 1 Behavioral Baseline Summary",
        "",
        "## Setup",
        f"- Model: `{model_path}`",
        f"- Dataset: `{dataset_dir}` 前 {len(samples)} 个 clean prompt",
        f"- Neutral verb: `{NEUTRAL_VERB}`",
        f"- Baseline `<tool_call>` top-1 rate: {percent(baseline['tool_call_top1_rate'])}",
        f"- Original verb distribution: {verb_summary}",
        "",
        "## Metrics",
        "| Condition | Description | `<tool_call>` top-1 rate | Mean `<tool_call>` logit | Mean rank |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for condition in CONDITION_ORDER:
        stats = metrics_by_condition[condition]
        lines.append(
            f"| `{condition}` | {CONDITION_LABELS[condition]} | {percent(stats['tool_call_top1_rate'])} | {fmt(stats['mean_tool_call_logit'])} | {fmt(stats['mean_tool_call_rank'])} |"
        )

    lines.extend(
        [
            "",
            "## Experiment Readout",
            f"- Exp1 (`with_tools` vs `no_tools`): top-1 rate {percent(metrics_by_condition['with_tools']['tool_call_top1_rate'])} -> {percent(metrics_by_condition['no_tools']['tool_call_top1_rate'])}; mean logit {fmt(metrics_by_condition['with_tools']['mean_tool_call_logit'])} -> {fmt(metrics_by_condition['no_tools']['mean_tool_call_logit'])}.",
            f"- Exp2 (`with_tools` vs `neutral_verb`): top-1 rate {percent(metrics_by_condition['with_tools']['tool_call_top1_rate'])} -> {percent(metrics_by_condition['neutral_verb']['tool_call_top1_rate'])}; mean logit {fmt(metrics_by_condition['with_tools']['mean_tool_call_logit'])} -> {fmt(metrics_by_condition['neutral_verb']['mean_tool_call_logit'])}.",
            f"- Exp3 (`with_tools` vs `no_tools_neutral`): top-1 rate {percent(metrics_by_condition['with_tools']['tool_call_top1_rate'])} -> {percent(metrics_by_condition['no_tools_neutral']['tool_call_top1_rate'])}; mean logit {fmt(metrics_by_condition['with_tools']['mean_tool_call_logit'])} -> {fmt(metrics_by_condition['no_tools_neutral']['mean_tool_call_logit'])}.",
            "",
            "## Preliminary Judgment",
            f"- Verdict: **{verdict}**",
        ]
    )
    lines.extend(f"- {reason}" for reason in reasons)
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 1 behavioral baseline for Qwen3-8B tool-call behavior")
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH)
    parser.add_argument("--dataset-dir", type=Path, default=DATASET_DIR)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--max-samples", type=int, default=MAX_SAMPLES)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    args.output_root.mkdir(parents=True, exist_ok=True)

    samples = load_prompt_variants(args.dataset_dir, args.max_samples)
    model, tokenizer = load_hooked_qwen3(str(args.model_path), device=args.device, dtype=torch.bfloat16)
    tool_token_ids = tokenizer.encode(TOOL_CALL_STR, add_special_tokens=False)
    if len(tool_token_ids) != 1:
        raise ValueError(f"{TOOL_CALL_STR!r} encoded to {tool_token_ids}, expected one token.")
    tool_token_id = int(tool_token_ids[0])

    all_rows: List[Dict[str, object]] = []
    progress = tqdm(samples, desc="Behavioral baseline", dynamic_ncols=True)
    for sample in progress:
        for condition in CONDITION_ORDER:
            row = evaluate_prompt(
                model,
                tokenizer,
                sample.sample_id,
                sample.original_verb,
                condition,
                sample.prompts[condition],
                tool_token_id,
            )
            all_rows.append(row)
        progress.set_postfix(sample=sample.sample_id, verb=sample.original_verb)

    rows_by_condition: Dict[str, List[Dict[str, object]]] = {condition: [] for condition in CONDITION_ORDER}
    for row in all_rows:
        rows_by_condition[str(row["condition"])].append(row)

    for experiment_name, conditions in EXPERIMENT_CONDITIONS.items():
        rows = []
        for condition in conditions:
            rows.extend(rows_by_condition[condition])
        rows.sort(key=lambda row: (str(row["file"]), conditions.index(str(row["condition"]))))
        write_csv(args.output_root / f"{experiment_name}.csv", rows)

    metrics_by_condition = {
        condition: condition_metrics(rows_by_condition[condition])
        for condition in CONDITION_ORDER
    }
    summary = build_summary(
        samples,
        metrics_by_condition,
        model_path=args.model_path,
        dataset_dir=args.dataset_dir,
    )
    write_text(args.output_root / "summary.md", summary)


if __name__ == "__main__":
    main()
