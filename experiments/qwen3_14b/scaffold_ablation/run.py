#!/usr/bin/env python3
"""Scaffold ablation experiment for Qwen3-14B (Section 5.1 & Table 4).

Evaluates first-token tool-call probability and top-1 rates across factorial
combinations of system scaffold components:
  - Role instructions (R)
  - Tool schema (T)
  - Format template (F)
Crossed with:
  - Neutral requests (assigned from 'Consider', 'Handle', 'Take', 'Use', 'Process')
  - Analysis requests (corrupt verb from pair)
  - Execution requests (clean verb from pair)

Evaluated on the held-out split of datasets/qwen3_14b/pair.
Outputs Table 4 in Markdown, CSV, and full summary JSON.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from mi4tc.paths import model_path  # noqa: E402

SYSTEM_OPEN = "<|im_start|>system\n"
TOOLS_OPEN = "<tools>\n"
FORMAT_OPEN = "For each function call, return a json object with function name and arguments within "
SYSTEM_TO_USER = "<|im_end|>\n<|im_start|>user\n"
USER_OPEN = "<|im_start|>user\n"
ASSISTANT_BOUNDARY = "<|im_end|>\n<|im_start|>assistant\n"

NEUTRAL_VERBS = ("Consider", "Handle", "Take", "Use", "Process")
TOOL_CALL_TOKEN = "<tool_call>"
TOOL_CALL_ID = 151657


@dataclass(frozen=True)
class PromptParts:
    system_open: str
    role_instructions: str
    tool_schema_block: str
    format_template: str
    system_to_user: str
    user_content: str
    assistant_suffix: str

    def render(
        self,
        *,
        user_content: str,
        include_role: bool,
        include_tool_schema: bool,
        include_format: bool,
        tool_schema_override: str | None = None,
    ) -> str:
        if not (include_role or include_tool_schema or include_format or tool_schema_override):
            return USER_OPEN + user_content + self.assistant_suffix

        tool_block = (
            self.tool_schema_block
            if include_tool_schema
            else (tool_schema_override if tool_schema_override is not None else "")
        )
        return (
            self.system_open
            + (self.role_instructions if include_role else "")
            + tool_block
            + (self.format_template if include_format else "")
            + self.system_to_user
            + user_content
            + self.assistant_suffix
        )


def parse_prompt_parts(prompt: str) -> PromptParts:
    if not prompt.startswith(SYSTEM_OPEN):
        raise ValueError("Expected prompt to start with system turn boundary")
    tools_start = prompt.find(TOOLS_OPEN)
    if tools_start < 0:
        raise ValueError("Could not find <tools> block in prompt")
    format_start = prompt.find(FORMAT_OPEN, tools_start)
    if format_start < 0:
        raise ValueError("Could not find format template in prompt")
    sys_to_user = prompt.find(SYSTEM_TO_USER, format_start)
    if sys_to_user < 0:
        raise ValueError("Could not find system-to-user boundary in prompt")
    user_content_start = sys_to_user + len(SYSTEM_TO_USER)
    asst_start = prompt.find(ASSISTANT_BOUNDARY, user_content_start)
    if asst_start < 0:
        raise ValueError("Could not find assistant boundary in prompt")

    parts = PromptParts(
        system_open=SYSTEM_OPEN,
        role_instructions=prompt[len(SYSTEM_OPEN) : tools_start],
        tool_schema_block=prompt[tools_start:format_start],
        format_template=prompt[format_start:sys_to_user],
        system_to_user=SYSTEM_TO_USER,
        user_content=prompt[user_content_start:asst_start],
        assistant_suffix=prompt[asst_start:],
    )
    return parts


def replace_leading_verb(user_content: str, new_verb: str) -> str:
    space = user_content.find(" ")
    line_end = user_content.find("\n")
    if space < 0 or (line_end >= 0 and space > line_end):
        raise ValueError(f"Could not find leading verb in {user_content[:60]!r}")
    return new_verb.strip() + user_content[space:]


def build_neutral_filler(tokenizer: Any, target_len: int) -> str:
    base = "Background context is provided for reference. General information is available. "
    filler = base
    while len(tokenizer.encode(filler, add_special_tokens=False)) < target_len:
        filler += base
    # Truncate tokens to match target_len
    tokens = tokenizer.encode(filler, add_special_tokens=False)[:target_len]
    return tokenizer.decode(tokens) + "\n\n"


@dataclass(frozen=True)
class ScaffoldCondition:
    key: str
    label: str
    include_role: bool
    include_tool_schema: bool
    include_format: bool
    length_matched_control: bool = False


CONDITIONS = [
    ScaffoldCondition("RTF", "R + T + F (full)", True, True, True),
    ScaffoldCondition("RT-", "R + T (no format template)", True, True, False),
    ScaffoldCondition("R-F", "R + F (no tool schema)", True, False, True),
    ScaffoldCondition("--F", "F only", False, False, True),
    ScaffoldCondition("R_TLEN_F", "R + length-matched neutral text + F", True, False, True, length_matched_control=True),
    ScaffoldCondition("R--", "R only", True, False, False),
    ScaffoldCondition("-T-", "T only", False, True, False),
    ScaffoldCondition("---", "Empty system scaffold", False, False, False),
]


def load_heldout_items(dataset_root: Path, max_pairs: int = 0) -> list[dict[str, Any]]:
    pairs_file = dataset_root / "manifest.jsonl" if (dataset_root / "manifest.jsonl").exists() else dataset_root / "pairs.jsonl"
    rows = []
    with pairs_file.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                if item.get("split") == "heldout":
                    rows.append(item)
    if max_pairs > 0:
        rows = rows[:max_pairs]
    return rows


def evaluate_batch(
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    device: torch.device,
    tool_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    tokenized = [tokenizer.encode(p, add_special_tokens=False) for p in prompts]
    max_len = max(len(t) for t in tokenized)

    batch_size = len(prompts)
    input_ids = torch.full((batch_size, max_len), pad_id, dtype=torch.long, device=device)
    attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long, device=device)
    last_positions = torch.zeros(batch_size, dtype=torch.long, device=device)

    for i, tok in enumerate(tokenized):
        input_ids[i, : len(tok)] = torch.tensor(tok, dtype=torch.long, device=device)
        attention_mask[i, : len(tok)] = 1
        last_positions[i] = len(tok) - 1

    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        logits = out.logits if hasattr(out, "logits") else out[0]

    rows = torch.arange(batch_size, device=device)
    last_logits = logits[rows, last_positions].float()

    tool_logits = last_logits[:, tool_token_id].cpu()
    probs = F.softmax(last_logits, dim=-1)[:, tool_token_id].cpu()
    top1 = (last_logits.argmax(dim=-1) == tool_token_id).cpu()
    return tool_logits, probs, top1


def main() -> int:
    parser = argparse.ArgumentParser(description="Scaffold component ablation on Qwen3-14B")
    parser.add_argument("--model-path", type=Path, default=model_path("qwen3_14b"))
    parser.add_argument("--dataset-root", type=Path, default=REPO_ROOT / "datasets/qwen3_14b/pair")
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "results/qwen3_14b/scaffold_ablation")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-pairs", type=int, default=0, help="0 for all held-out pairs")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="bfloat16")
    args = parser.parse_args()

    print(f"Loading Qwen3-14B from {args.model_path} onto {args.device}...")
    import transformers

    torch_dtype = getattr(torch, args.dtype)
    tokenizer = transformers.AutoTokenizer.from_pretrained(str(args.model_path), trust_remote_code=True)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        str(args.model_path),
        torch_dtype=torch_dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    ).to(args.device)
    model.eval()

    heldout_meta = load_heldout_items(args.dataset_root, max_pairs=args.max_pairs)
    print(f"Loaded {len(heldout_meta)} held-out pairs from {args.dataset_root}")

    # Parse and prepare test items
    items = []
    for idx, row in enumerate(heldout_meta):
        clean_file = args.dataset_root / row["clean_relpath"]
        corrupt_file = args.dataset_root / row["corrupt_relpath"]
        clean_text = clean_file.read_text(encoding="utf-8")
        corrupt_text = corrupt_file.read_text(encoding="utf-8")

        parts = parse_prompt_parts(clean_text)
        corrupt_parts = parse_prompt_parts(corrupt_text)

        neutral_verb = NEUTRAL_VERBS[idx % len(NEUTRAL_VERBS)]
        neutral_user = replace_leading_verb(parts.user_content, neutral_verb)

        schema_len = len(tokenizer.encode(parts.tool_schema_block, add_special_tokens=False))
        filler_block = build_neutral_filler(tokenizer, schema_len)

        items.append({
            "sample_id": row["sample_id"],
            "parts": parts,
            "corrupt_parts": corrupt_parts,
            "neutral_user": neutral_user,
            "clean_user": parts.user_content,
            "corrupt_user": corrupt_parts.user_content,
            "filler_block": filler_block,
            "neutral_verb": neutral_verb,
            "clean_verb": row.get("clean_verb"),
            "corrupt_verb": row.get("corrupt_verb"),
        })

    args.output_root.mkdir(parents=True, exist_ok=True)
    sample_records: list[dict[str, Any]] = []
    condition_summaries: list[dict[str, Any]] = []

    print("\nEvaluating scaffold conditions...")
    for cond in CONDITIONS:
        print(f"  Condition: {cond.label} ...", flush=True)

        for req_type in ("neutral", "analysis", "execution"):
            prompts = []
            for it in items:
                parts = it["parts"]
                if req_type == "neutral":
                    u = it["neutral_user"]
                elif req_type == "analysis":
                    u = it["corrupt_user"]
                else:
                    u = it["clean_user"]

                override = it["filler_block"] if cond.length_matched_control else None
                rendered = parts.render(
                    user_content=u,
                    include_role=cond.include_role,
                    include_tool_schema=cond.include_tool_schema,
                    include_format=cond.include_format,
                    tool_schema_override=override,
                )
                prompts.append(rendered)

            # Batched evaluation
            all_tool_logits = []
            all_probs = []
            all_top1 = []
            for b_start in range(0, len(prompts), args.batch_size):
                b_prompts = prompts[b_start : b_start + args.batch_size]
                t_logits, probs, top1 = evaluate_batch(
                    model, tokenizer, b_prompts, torch.device(args.device), TOOL_CALL_ID
                )
                all_tool_logits.append(t_logits)
                all_probs.append(probs)
                all_top1.append(top1)

            tool_logits = torch.cat(all_tool_logits)
            probs = torch.cat(all_probs)
            top1 = torch.cat(all_top1)

            mean_prob = float(probs.mean().item())
            top1_rate = float(top1.float().mean().item())
            mean_logit = float(tool_logits.mean().item())

            for i, it in enumerate(items):
                sample_records.append({
                    "sample_id": it["sample_id"],
                    "scaffold_condition": cond.key,
                    "request_type": req_type,
                    "tool_prob": float(probs[i].item()),
                    "top1": int(top1[i].item()),
                    "tool_logit": float(tool_logits[i].item()),
                })

            cond_key = f"{cond.key}_{req_type}"
            # Store summary in accumulator
            # We'll merge neutral, analysis, execution into condition_summaries later
            cond.__dict__.setdefault("_res", {})[req_type] = {
                "mean_prob": mean_prob,
                "top1_rate": top1_rate,
                "mean_logit": mean_logit,
            }

    # Format Table 4 summary
    table4_rows = []
    full_summary = {}

    for cond in CONDITIONS:
        res = getattr(cond, "_res")
        neut = res["neutral"]["mean_prob"]
        anal = res["analysis"]["mean_prob"]
        exec_prob = res["execution"]["mean_prob"]
        p_diff = neut - anal

        neut_str = f"{neut:.4f}" if neut >= 1e-4 else f"{neut:.2e}"
        anal_str = f"{anal:.4f}" if anal >= 1e-4 else f"{anal:.2e}"
        diff_str = f"{p_diff:.4f}" if abs(p_diff) >= 1e-4 else f"{p_diff:.2e}"

        table4_rows.append({
            "scaffold": cond.label,
            "neutral_p_call": neut,
            "analysis_p_call": anal,
            "execution_p_call": exec_prob,
            "delta_p": p_diff,
            "neutral_top1_rate": res["neutral"]["top1_rate"],
            "analysis_top1_rate": res["analysis"]["top1_rate"],
            "execution_top1_rate": res["execution"]["top1_rate"],
        })
        full_summary[cond.key] = {
            "label": cond.label,
            "neutral": res["neutral"],
            "analysis": res["analysis"],
            "execution": res["execution"],
            "P_neutral_minus_analysis": p_diff,
        }

    # Write Markdown table
    md_lines = [
        "# Scaffold-Component Ablation on Qwen3-14B (Table 4)",
        "",
        f"Evaluated on {len(items)} held-out tasks from `datasets/qwen3_14b/pair`.",
        "",
        "| Scaffold | Neutral request | Analysis request | P = Neutral - Analysis | Execution request |",
        "|:---|---:|---:|---:|---:|",
    ]
    for r in table4_rows:
        n_s = f"{r['neutral_p_call']:.4f}" if r['neutral_p_call'] >= 1e-4 else f"{r['neutral_p_call']:.2e}"
        a_s = f"{r['analysis_p_call']:.4f}" if r['analysis_p_call'] >= 1e-4 else f"{r['analysis_p_call']:.2e}"
        d_s = f"{r['delta_p']:.4f}" if abs(r['delta_p']) >= 1e-4 else f"{r['delta_p']:.2e}"
        e_s = f"{r['execution_p_call']:.4f}" if r['execution_p_call'] >= 1e-4 else f"{r['execution_p_call']:.2e}"
        md_lines.append(f"| {r['scaffold']} | {n_s} | {a_s} | {d_s} | {e_s} |")
    md_lines.append("")

    md_content = "\n".join(md_lines)
    (args.output_root / "table4_scaffold_ablation.md").write_text(md_content, encoding="utf-8")
    print("\n" + md_content)

    # Write CSV
    with (args.output_root / "table4_scaffold_ablation.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(table4_rows[0].keys()))
        writer.writeheader()
        writer.writerows(table4_rows)

    # Write sample metrics
    with (args.output_root / "sample_metrics.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(sample_records[0].keys()))
        writer.writeheader()
        writer.writerows(sample_records)

    # Write JSON summary
    (args.output_root / "scaffold_ablation_summary.json").write_text(
        json.dumps({
            "model_path": str(args.model_path),
            "n_heldout": len(items),
            "table4": table4_rows,
            "conditions": full_summary,
        }, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(f"Results written to {args.output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
