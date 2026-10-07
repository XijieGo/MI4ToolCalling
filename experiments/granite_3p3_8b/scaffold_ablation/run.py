#!/usr/bin/env python3
"""Evaluate scaffold component ablations on Granite-3.3-8B held-out tasks (Table 4 style)."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from mi4tc.paths import model_path  # noqa: E402

from transformers import AutoModelForCausalLM, AutoTokenizer

NEUTRAL_VERBS = ("Consider", "Handle", "Take", "Use", "Process")
TOOL_CALL_TOKEN = "<|tool_call|>"
TOOL_CALL_ID = 49154


@dataclass(frozen=True)
class NativeParts:
    prefix: str
    role_open: str
    role_text: str
    role_close: str
    tools_open: str
    tools_payload: str
    tools_close: str
    between_tools_and_user: str
    user_open: str
    user_content: str
    user_close: str
    assistant_suffix: str

    def original(self) -> str:
        return (
            self.prefix
            + self.role_open
            + self.role_text
            + self.role_close
            + self.tools_open
            + self.tools_payload
            + self.tools_close
            + self.between_tools_and_user
            + self.user_open
            + self.user_content
            + self.user_close
            + self.assistant_suffix
        )

    def render(self, condition: str, user_content: str | None = None) -> str:
        user = self.user_content if user_content is None else user_content
        payload = self.tools_payload
        include_role = True
        include_tools = True

        if condition == "full":
            pass
        elif condition == "no_tools":
            include_tools = False
        elif condition == "no_role":
            include_role = False
        elif condition == "empty_system":
            return self.prefix + self.user_open + user + self.user_close + self.assistant_suffix
        elif condition == "neutral_length_matched":
            payload = "The following documentation is provided for reference only.\n" + (" " * max(0, len(self.tools_payload) - 70))
        elif condition == "tools_only":
            include_role = False
        elif condition == "role_only":
            include_tools = False
        else:
            raise ValueError(f"Unknown condition: {condition}")

        role_segment = self.role_open + self.role_text + self.role_close if include_role else ""
        tools_segment = self.tools_open + payload + self.tools_close if include_tools else ""
        return (
            self.prefix
            + role_segment
            + tools_segment
            + self.between_tools_and_user
            + self.user_open
            + user
            + self.user_close
            + self.assistant_suffix
        )


def parse_native_parts(text: str) -> NativeParts:
    role_open = "<|start_of_role|>system<|end_of_role|>"
    available_open = "<|start_of_role|>available_tools<|end_of_role|>"
    user_open = "<|start_of_role|>user<|end_of_role|>"
    assistant_open = "<|start_of_role|>assistant<|end_of_role|>"
    end = "<|end_of_text|>"

    if not text.startswith(role_open):
        raise ValueError("Granite native prompt does not start with a system role")
    role_text, found_role_end, remainder = text[len(role_open) :].partition(end)
    if not found_role_end:
        raise ValueError("Granite native prompt lacks available_tools role")
    after_role, found_available, remainder = remainder.partition(available_open)
    if not found_available:
        raise ValueError("Granite native prompt lacks available_tools segment")
    payload, found_tools_end, remainder = remainder.partition(end)
    if not found_tools_end:
        raise ValueError("Granite native prompt lacks available_tools terminator")
    before_user, found_user, remainder = remainder.partition(user_open)
    if not found_user:
        raise ValueError("Granite native prompt lacks user role")
    user, found_user_end, remainder = remainder.partition(end)
    if not found_user_end:
        raise ValueError("Granite native prompt lacks user terminator")

    return NativeParts(
        prefix="",
        role_open=role_open,
        role_text=role_text,
        role_close=end + after_role,
        tools_open=available_open,
        tools_payload=payload,
        tools_close=end,
        between_tools_and_user=before_user,
        user_open=user_open,
        user_content=user,
        user_close=end,
        assistant_suffix=remainder,
    )


def replace_instruction_verb(user_content: str, source_verb: str, target_verb: str) -> str:
    lines = user_content.splitlines(keepends=True)
    for i, line in enumerate(lines):
        if line.startswith(source_verb):
            lines[i] = target_verb + line[len(source_verb) :]
            return "".join(lines)
        if line.startswith(source_verb.capitalize()):
            lines[i] = target_verb.capitalize() + line[len(source_verb) :]
            return "".join(lines)
    return user_content.replace(source_verb, target_verb, 1)


def load_heldout_items(dataset_root: Path) -> list[dict[str, Any]]:
    manifest_path = dataset_root / "manifest.jsonl"
    if not manifest_path.is_file():
        manifest_path = dataset_root / "pairs.jsonl"
    items = []
    with open(manifest_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("split", "heldout") == "heldout":
                items.append(row)
    return items


def evaluate_prompts(model, tokenizer, prompts: list[str], batch_size: int = 8) -> list[float]:
    probs = []
    device = next(model.parameters()).device

    for i in range(0, len(prompts), batch_size):
        batch_texts = prompts[i : i + batch_size]
        enc = tokenizer(batch_texts, padding=True, return_tensors="pt")
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits[:, -1, :]
            batch_probs = F.softmax(logits, dim=-1)[:, TOOL_CALL_ID]
            probs.extend(batch_probs.detach().cpu().tolist())

    return probs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, default=model_path("granite"))
    parser.add_argument("--dataset-root", type=Path, default=Path("datasets/granite_3p3_8b/pair"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/granite_3p3_8b/scaffold_ablation"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-pairs", type=int, default=0)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading Granite-3.3-8B from {args.model_path} onto cuda...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        trust_remote_code=True,
    )
    model.eval()

    heldout_rows = load_heldout_items(args.dataset_root)
    if args.max_pairs > 0:
        heldout_rows = heldout_rows[: args.max_pairs]
    print(f"Loaded {len(heldout_rows)} held-out pairs from {args.dataset_root}")

    conditions = [
        ("Full native scaffold", "full"),
        ("No available tools (R + F)", "no_tools"),
        ("No system role (T + F)", "no_role"),
        ("Length-matched neutral tools", "neutral_length_matched"),
        ("Tools only", "tools_only"),
        ("Role only", "role_only"),
        ("Empty scaffold (user only)", "empty_system"),
    ]

    print("\nEvaluating scaffold conditions...")
    results_table = []

    for label, cond_key in conditions:
        print(f"  Condition: {label} ...", flush=True)

        neutral_prompts = []
        analysis_prompts = []
        execution_prompts = []

        for row in heldout_rows:
            clean_text = (args.dataset_root / row["clean_relpath"]).read_text(encoding="utf-8")
            corrupt_text = (args.dataset_root / row["corrupt_relpath"]).read_text(encoding="utf-8")
            clean_verb = row.get("clean_verb", row.get("clean_candidate", "add"))
            corrupt_verb = row.get("corrupt_verb", row.get("corrupt_candidate", "discuss"))

            clean_parts = parse_native_parts(clean_text)
            corrupt_parts = parse_native_parts(corrupt_text)

            execution_prompts.append(clean_parts.render(cond_key))
            analysis_prompts.append(corrupt_parts.render(cond_key))

            target_neutral_verb = NEUTRAL_VERBS[hash(row.get("sample_id", "")) % len(NEUTRAL_VERBS)]
            neutral_user = replace_instruction_verb(clean_parts.user_content, clean_verb, target_neutral_verb)
            neutral_prompts.append(clean_parts.render(cond_key, user_content=neutral_user))

        exec_probs = evaluate_prompts(model, tokenizer, execution_prompts, batch_size=args.batch_size)
        analysis_probs = evaluate_prompts(model, tokenizer, analysis_prompts, batch_size=args.batch_size)
        neutral_probs = evaluate_prompts(model, tokenizer, neutral_prompts, batch_size=args.batch_size)

        mean_exec = sum(exec_probs) / len(exec_probs)
        mean_analysis = sum(analysis_probs) / len(analysis_probs)
        mean_neutral = sum(neutral_probs) / len(neutral_probs)
        delta_p = mean_neutral - mean_analysis

        results_table.append({
            "scaffold": label,
            "neutral_request": mean_neutral,
            "analysis_request": mean_analysis,
            "delta_p": delta_p,
            "execution_request": mean_exec,
        })

    md_lines = [
        "# Scaffold-Component Ablation on Granite-3.3-8B (Table 4 Style)",
        "",
        f"Evaluated on {len(heldout_rows)} held-out tasks from `{args.dataset_root}`.",
        "",
        "| Scaffold | Neutral request | Analysis request | P = Neutral - Analysis | Execution request |",
        "|:---|---:|---:|---:|---:|",
    ]
    for row in results_table:
        def fmt(val: float) -> str:
            if abs(val) < 1e-3 and val != 0:
                return f"{val:.2e}"
            return f"{val:.4f}"
        md_lines.append(
            f"| {row['scaffold']} | {fmt(row['neutral_request'])} | {fmt(row['analysis_request'])} | "
            f"{fmt(row['delta_p'])} | {fmt(row['execution_request'])} |"
        )
    md_content = "\n".join(md_lines) + "\n"
    print("\n" + md_content)

    (args.output_dir / "table4_scaffold_ablation.md").write_text(md_content, encoding="utf-8")

    with open(args.output_dir / "table4_scaffold_ablation.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["scaffold", "neutral_request", "analysis_request", "delta_p", "execution_request"]
        )
        writer.writeheader()
        writer.writerows(results_table)

    with open(args.output_dir / "scaffold_ablation_summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "model": "Granite-3.3-8B",
                "n_tasks": len(heldout_rows),
                "rows": results_table,
            },
            f,
            indent=2,
        )

    print(f"Results written to {args.output_dir}")


if __name__ == "__main__":
    main()
