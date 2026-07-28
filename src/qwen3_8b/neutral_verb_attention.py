#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from tqdm.auto import tqdm

from scan_neutral_verbs import DATASET_DIR as TRAIN_CLEAN_DIR
from scan_neutral_verbs import OUTPUT_ROOT as DEFAULT_SCAN_OUTPUT_ROOT
from scan_neutral_verbs import PromptTemplate, load_templates
from task_attention_path_analysis import (
    ATTN_LAYER,
    MODEL_PATH,
    TARGET_HEAD,
    build_region_masks,
    clear_cuda,
    ensure_dir,
    resolve_pattern_head_idx,
    set_seed,
    write_csv,
    write_json,
    write_text,
)

import sys

LEGACY_SRC = Path("./src")
if str(LEGACY_SRC) not in sys.path:
    sys.path.insert(0, str(LEGACY_SRC))

from toolcall_circuit.single_sample import load_hooked_qwen3  # noqa: E402


SEED = 42
TOOL_CALL_STR = "<tool_call>"
DEFAULT_OUTPUT_ROOT = DEFAULT_SCAN_OUTPUT_ROOT


@dataclass
class VerbPrompt:
    verb: str
    role: str
    scan_tool_call_rate: float
    sample_id: str
    text: str
    tokens_cpu: torch.Tensor
    region_masks: dict[str, torch.Tensor]


def load_scan_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def select_verbs(
    rows: Sequence[dict[str, str]],
    *,
    top_k: int,
    clean_control: str,
    corrupt_control: str,
) -> list[tuple[str, str, float]]:
    normalized = {str(row["candidate"]): row for row in rows}
    ranked = sorted(
        rows,
        key=lambda row: (
            abs(float(row["tool_call_top1_rate"]) - 0.5),
            float(row["composite_mid_score"]),
            str(row["candidate"]),
        ),
    )
    chosen: list[tuple[str, str, float]] = []
    used = {clean_control, corrupt_control}
    for row in ranked:
        candidate = str(row["candidate"])
        if candidate in used:
            continue
        chosen.append((candidate, "neutral", float(row["tool_call_top1_rate"])))
        used.add(candidate)
        if len(chosen) >= top_k:
            break
    for candidate, role in ((clean_control, "clean_control"), (corrupt_control, "corrupt_control")):
        if candidate not in normalized:
            raise ValueError(f"Control verb {candidate!r} not found in scan summary")
        chosen.append((candidate, role, float(normalized[candidate]["tool_call_top1_rate"])))
    return chosen


def build_prompts(
    model,
    tokenizer,
    templates: Sequence[PromptTemplate],
    selected_verbs: Sequence[tuple[str, str, float]],
) -> list[VerbPrompt]:
    special_ids = set(getattr(tokenizer, "all_special_ids", []))
    prompts: list[VerbPrompt] = []
    for verb, role, scan_rate in selected_verbs:
        for template in templates:
            text = template.render(verb)
            tokens_cpu = model.to_tokens(text, prepend_bos=False).detach().cpu()
            encoding = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
            region_masks = build_region_masks(
                text,
                [(int(a), int(b)) for a, b in encoding["offset_mapping"]],
                special_ids,
                encoding["input_ids"],
                verb,
            )
            prompts.append(
                VerbPrompt(
                    verb=verb,
                    role=role,
                    scan_tool_call_rate=scan_rate,
                    sample_id=template.sample_id,
                    text=text,
                    tokens_cpu=tokens_cpu,
                    region_masks=region_masks,
                )
            )
    return prompts


def build_batches(prompts: Sequence[VerbPrompt], batch_size: int) -> list[tuple[list[int], torch.Tensor]]:
    buckets: dict[int, list[tuple[int, VerbPrompt]]] = defaultdict(list)
    for idx, prompt in enumerate(prompts):
        buckets[int(prompt.tokens_cpu.shape[-1])].append((idx, prompt))
    batches: list[tuple[list[int], torch.Tensor]] = []
    for token_len in sorted(buckets):
        group = buckets[token_len]
        for start in range(0, len(group), batch_size):
            chunk = group[start : start + batch_size]
            batches.append(([idx for idx, _ in chunk], torch.cat([prompt.tokens_cpu for _, prompt in chunk], dim=0)))
    return batches


def tool_stats(logits: torch.Tensor, tool_token_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    last_logits = logits[:, -1, :]
    return (
        last_logits[:, tool_token_id].detach().cpu().float(),
        last_logits.argmax(dim=-1).detach().cpu(),
    )


def make_pattern_capture(capture: dict[str, torch.Tensor]):
    def hook_fn(value: torch.Tensor, hook):  # noqa: ANN001
        capture["pattern"] = value.detach().cpu().float()
        return value

    return hook_fn


def run_attention_spectrum(
    model,
    tokenizer,
    prompts: Sequence[VerbPrompt],
    *,
    batch_size: int,
    tool_token_id: int,
    output_root: Path,
) -> dict[str, object]:
    ensure_dir(output_root)
    batches = build_batches(prompts, batch_size)
    pattern_hook_name = f"blocks.{ATTN_LAYER}.attn.hook_pattern"

    per_prompt_rows: list[dict[str, object]] = []
    progress = tqdm(batches, desc="Neutral verb attention", dynamic_ncols=True)
    for batch_indices, tokens_cpu in progress:
        tokens = tokens_cpu.to(model.W_U.device)
        capture: dict[str, torch.Tensor] = {}
        with torch.no_grad():
            logits = model.run_with_hooks(tokens, fwd_hooks=[(pattern_hook_name, make_pattern_capture(capture))])
        tool_logit, top1 = tool_stats(logits, tool_token_id)
        pattern = capture["pattern"]
        head_idx = resolve_pattern_head_idx(TARGET_HEAD, int(pattern.shape[1]), int(getattr(model.cfg, "n_heads", pattern.shape[1])))
        for local_idx, prompt_idx in enumerate(batch_indices):
            prompt = prompts[prompt_idx]
            weights = pattern[local_idx, head_idx, -1, :]
            per_prompt_rows.append(
                {
                    "verb": prompt.verb,
                    "role": prompt.role,
                    "sample_id": prompt.sample_id,
                    "scan_tool_call_rate": prompt.scan_tool_call_rate,
                    "system_attn_h9": float(weights[prompt.region_masks["system"]].sum().item()),
                    "special_attn_h9": float(weights[prompt.region_masks["special"]].sum().item()),
                    "verb_attn_h9": float(weights[prompt.region_masks["verb"]].sum().item()),
                    "tool_logit": float(tool_logit[local_idx].item()),
                    "is_tool_call_top1": int(top1[local_idx].item() == tool_token_id),
                }
            )
        del tokens, logits, pattern
        clear_cuda()

    write_csv(output_root / "verb_attention_per_prompt.csv", per_prompt_rows)

    summary_rows: list[dict[str, object]] = []
    verbs = []
    for row in per_prompt_rows:
        if row["verb"] not in verbs:
            verbs.append(str(row["verb"]))
    for verb in verbs:
        rows = [row for row in per_prompt_rows if row["verb"] == verb]
        system = np.asarray([float(row["system_attn_h9"]) for row in rows], dtype=np.float64)
        special = np.asarray([float(row["special_attn_h9"]) for row in rows], dtype=np.float64)
        verb_attn = np.asarray([float(row["verb_attn_h9"]) for row in rows], dtype=np.float64)
        tool_logit = np.asarray([float(row["tool_logit"]) for row in rows], dtype=np.float64)
        top1 = np.asarray([float(row["is_tool_call_top1"]) for row in rows], dtype=np.float64)
        summary_rows.append(
            {
                "verb": verb,
                "role": rows[0]["role"],
                "tool_call_rate": float(rows[0]["scan_tool_call_rate"]),
                "sample_tool_call_rate_50": float(top1.mean()),
                "mean_system_attn_h9": float(system.mean()),
                "std_system_attn_h9": float(system.std(ddof=1)),
                "mean_special_attn_h9": float(special.mean()),
                "mean_verb_attn_h9": float(verb_attn.mean()),
                "mean_tool_logit": float(tool_logit.mean()),
                "std_tool_logit": float(tool_logit.std(ddof=1)),
                "n_prompts": int(len(rows)),
            }
        )
    summary_rows.sort(key=lambda row: float(row["tool_call_rate"]), reverse=True)
    write_csv(output_root / "verb_attention_spectrum.csv", summary_rows)

    rates = np.asarray([float(row["tool_call_rate"]) for row in summary_rows], dtype=np.float64)
    system_attn = np.asarray([float(row["mean_system_attn_h9"]) for row in summary_rows], dtype=np.float64)
    pearson = float(np.corrcoef(rates, system_attn)[0, 1]) if len(summary_rows) >= 2 else float("nan")

    lines = [
        "# Neutral Verb Attention Spectrum",
        "",
        f"- 样本模板: `{len(summary_rows) and int(summary_rows[0]['n_prompts'])}` per verb",
        f"- verbs: `{', '.join(row['verb'] for row in summary_rows)}`",
        f"- Pearson(tool_call_rate, system_attn) = `{pearson:.4f}`",
        "",
        "| verb | role | scan rate | sample rate | mean system attn | mean tool logit |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['verb']} | {row['role']} | {float(row['tool_call_rate']):.4f} | {float(row['sample_tool_call_rate_50']):.4f} | "
            f"{float(row['mean_system_attn_h9']):.4f} | {float(row['mean_tool_logit']):.4f} |"
        )
    lines.extend(
        [
            "",
            "## 结论",
            "",
            "- 若中性词落在 clean / corrupt 两端之间，并且 system attention 随 tool-call rate 单调变化，则支持 `H9 system attention` 是连续代理变量。",
        ]
    )
    write_text(output_root / "verb_attention_summary.md", "\n".join(lines))

    return {
        "n_verbs": len(summary_rows),
        "n_prompts_total": len(prompts),
        "pearson_tool_rate_vs_system_attn": pearson,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure L29H9 system attention for neutral verbs.")
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH)
    parser.add_argument("--dataset-dir", type=Path, default=TRAIN_CLEAN_DIR)
    parser.add_argument("--scan-output-root", type=Path, default=DEFAULT_SCAN_OUTPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--limit-prompts", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--clean-control", type=str, default="save")
    parser.add_argument("--corrupt-control", type=str, default="review")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    scan_rows = load_scan_rows(args.scan_output_root / "verb_summary.csv")
    selected_verbs = select_verbs(
        scan_rows,
        top_k=args.top_k,
        clean_control=args.clean_control,
        corrupt_control=args.corrupt_control,
    )
    templates = load_templates(args.dataset_dir.resolve(), args.limit_prompts)

    model, tokenizer = load_hooked_qwen3(str(args.model_path), device=args.device, dtype=torch.bfloat16)
    tool_token_ids = tokenizer.encode(TOOL_CALL_STR, add_special_tokens=False)
    if len(tool_token_ids) != 1:
        raise ValueError(f"{TOOL_CALL_STR!r} encoded to {tool_token_ids}, expected one token")
    tool_token_id = int(tool_token_ids[0])

    prompts = build_prompts(model, tokenizer, templates, selected_verbs)
    metadata = run_attention_spectrum(
        model,
        tokenizer,
        prompts,
        batch_size=args.batch_size,
        tool_token_id=tool_token_id,
        output_root=args.output_root,
    )
    write_json(
        args.output_root / "verb_attention_metadata.json",
        {
            "seed": args.seed,
            "model_path": str(args.model_path),
            "dataset_dir": str(args.dataset_dir),
            "scan_output_root": str(args.scan_output_root),
            "limit_prompts": args.limit_prompts,
            "batch_size": args.batch_size,
            "selected_verbs": [
                {
                    "verb": verb,
                    "role": role,
                    "scan_tool_call_rate": rate,
                }
                for verb, role, rate in selected_verbs
            ],
            **metadata,
        },
    )

    del model
    clear_cuda()


if __name__ == "__main__":
    main()
