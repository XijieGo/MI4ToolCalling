#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from tqdm.auto import tqdm
from transformers import AutoTokenizer, Mistral3ForConditionalGeneration


PROJECT_ROOT = Path(__file__).resolve().parents[2]
QWEN_SYSTEM_RE = re.compile(r"<\|im_start\|>system\n(.*?)<\|im_end\|>", re.DOTALL)
QWEN_USER_RE = re.compile(r"<\|im_start\|>user\n(.*?)<\|im_end\|>", re.DOTALL)
TOOLS_RE = re.compile(r"<tools>\s*(.*?)\s*</tools>", re.DOTALL)

SHORT_TOOLCALL_SYSTEM_PROMPT = """# Tools

You may call one or more functions to assist with the user query.

Use the available functions when the user asks you to create, write, update, complete, or save file content.

If tool use is appropriate, make the tool call. Otherwise answer normally."""

EXECUTION_VERBS = (
    "add",
    "build",
    "write",
    "complete",
    "create",
    "implement",
    "generate",
    "construct",
    "compose",
    "craft",
    "assemble",
    "produce",
    "develop",
    "draft",
    "define",
    "devise",
    "formulate",
    "prepare",
    "design",
    "fill",
    "extend",
    "rewrite",
    "modify",
    "revise",
    "edit",
    "adjust",
    "fix",
    "patch",
    "improve",
    "refine",
    "update",
    "adapt",
    "rework",
    "reshape",
    "tune",
    "optimize",
    "polish",
    "correct",
    "repair",
    "streamline",
    "simplify",
    "debug",
    "test",
    "validate",
    "verify",
    "check",
    "handle",
    "manage",
    "address",
    "tackle",
    "process",
    "approach",
    "treat",
    "structure",
    "restructure",
    "organize",
    "arrange",
)


@dataclass(frozen=True)
class SampleTemplate:
    sample_id: str
    split: str
    language: str
    dataset_name: str
    original_clean_candidate: str
    original_corrupt_candidate: str
    user_content: str
    tools_schema: list[dict[str, Any]]

    def render_user_content(self, verb: str) -> str:
        candidate = verb[:1].upper() + verb[1:]
        return re.sub(r"^\S+", candidate, self.user_content, count=1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scan execution-related verbs on Mistral-Small-3.2 with short tool-call prompt.")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path(
            os.environ.get(
                "MISTRAL_3P2_24B_PATH",
                str(PROJECT_ROOT / "external" / "models" / "Mistral-Small-3.2-24B-Instruct-2506"),
            )
        ),
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=PROJECT_ROOT / "datasets",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "results" / "Mistral-Small-3.2-24B-Instruct-2506" / "verb_scan_short_prompt_50",
    )
    parser.add_argument("--n-samples", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--device-map", type=str, default="cuda:0")
    parser.add_argument("--max-verbs", type=int, default=0)
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def parse_tools_payload(payload: str) -> list[dict[str, Any]]:
    payload = payload.strip()
    if payload.startswith("["):
        data = json.loads(payload)
        if not isinstance(data, list):
            raise ValueError(f"Expected list tools payload, got {type(data).__name__}.")
        return data
    decoder = json.JSONDecoder()
    idx = 0
    items: list[dict[str, Any]] = []
    while idx < len(payload):
        while idx < len(payload) and payload[idx].isspace():
            idx += 1
        if idx >= len(payload):
            break
        item, idx = decoder.raw_decode(payload, idx)
        if not isinstance(item, dict):
            raise ValueError(f"Expected dict tool schema, got {type(item).__name__}.")
        items.append(item)
    if not items:
        raise ValueError("No tool items decoded.")
    return items


def parse_qwen_prompt(text: str) -> tuple[str, list[dict[str, Any]]]:
    system_match = QWEN_SYSTEM_RE.search(text)
    user_match = QWEN_USER_RE.search(text)
    if system_match is None or user_match is None:
        raise ValueError("Failed to parse Qwen prompt blocks.")
    system_text = system_match.group(1).strip()
    user_text = user_match.group(1).strip()
    tool_matches = [match.strip() for match in TOOLS_RE.findall(system_text)]
    tool_matches = [match for match in tool_matches if match]
    if not tool_matches:
        raise ValueError("Missing <tools> payload.")
    tools = parse_tools_payload(max(tool_matches, key=len))
    return user_text, tools


def select_templates(dataset_root: Path, n_samples: int, seed: int) -> list[SampleTemplate]:
    manifest_rows = read_jsonl(dataset_root / "clean" / "manifest.jsonl")
    by_language: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in manifest_rows:
        by_language[str(row["language"])].append(row)

    rng = random.Random(seed)
    targets = {"cpp": n_samples // 3, "java": n_samples // 3, "python": n_samples - 2 * (n_samples // 3)}
    selected_rows: list[dict[str, Any]] = []
    for language, target in targets.items():
        pool = list(by_language[language])
        rng.shuffle(pool)
        selected_rows.extend(pool[:target])
    selected_rows = sorted(selected_rows, key=lambda row: str(row["output_filename"]))

    templates: list[SampleTemplate] = []
    for row in selected_rows:
        filename = str(row["output_filename"])
        clean_text = (dataset_root / "clean" / filename).read_text(encoding="utf-8")
        user_content, tools_schema = parse_qwen_prompt(clean_text)
        templates.append(
            SampleTemplate(
                sample_id=Path(filename).stem,
                split="test" if (dataset_root / "test" / "clean" / filename).exists() else "train",
                language=str(row["language"]),
                dataset_name=str(row.get("dataset_name") or row.get("dataset") or ""),
                original_clean_candidate=str(row["clean_candidate"]),
                original_corrupt_candidate=str(row["corrupt_candidate"]),
                user_content=user_content,
                tools_schema=tools_schema,
            )
        )
    return templates


def build_max_memory(device_map: str) -> dict[Any, str] | None:
    if device_map != "auto" or not torch.cuda.is_available():
        return None
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    free_gib = max(int(free_bytes // (1024**3)) - 2, 8)
    total_gib = int(total_bytes // (1024**3))
    free_gib = min(free_gib, total_gib - 1)
    return {0: f"{free_gib}GiB", "cpu": "400GiB"}


def load_model(model_path: Path, dtype_name: str, device_map: str, output_root: Path):
    dtype = getattr(torch, dtype_name)
    kwargs: dict[str, Any] = {
        "dtype": dtype,
        "device_map": device_map,
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
    }
    max_memory = build_max_memory(device_map)
    if max_memory is not None:
        offload_folder = output_root / "offload"
        ensure_dir(offload_folder)
        kwargs["max_memory"] = max_memory
        kwargs["offload_folder"] = str(offload_folder)
    model = Mistral3ForConditionalGeneration.from_pretrained(str(model_path), **kwargs)
    model.eval()
    return model


def get_model_input_device(model) -> torch.device:
    try:
        return model.get_input_embeddings().weight.device
    except Exception:
        return next(model.parameters()).device


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def main() -> None:
    args = parse_args()
    ensure_dir(args.output_root)

    tokenizer = AutoTokenizer.from_pretrained(str(args.model_path), trust_remote_code=True)
    tool_token_text = "[TOOL_CALLS]"
    tool_token_id = int(tokenizer.convert_tokens_to_ids(tool_token_text))
    if tool_token_id < 0:
        raise ValueError("Failed to resolve [TOOL_CALLS] special token id.")

    model = load_model(args.model_path, args.dtype, args.device_map, args.output_root)
    input_device = get_model_input_device(model)
    templates = select_templates(args.dataset_root, args.n_samples, args.seed)
    verbs = list(EXECUTION_VERBS)
    if args.max_verbs > 0:
        verbs = verbs[: args.max_verbs]

    (args.output_root / "subset_manifest.json").write_text(
        json.dumps(
            [
                {
                    "sample_id": row.sample_id,
                    "split": row.split,
                    "language": row.language,
                    "dataset_name": row.dataset_name,
                    "original_clean_candidate": row.original_clean_candidate,
                    "original_corrupt_candidate": row.original_corrupt_candidate,
                }
                for row in templates
            ],
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    per_prompt_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    for verb in tqdm(verbs, desc="Mistral execution verbs", dynamic_ncols=True):
        prepared: list[dict[str, Any]] = []
        for template in templates:
            user_content = template.render_user_content(verb)
            messages = [
                {"role": "system", "content": SHORT_TOOLCALL_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ]
            encoded = tokenizer.apply_chat_template(
                messages,
                tools=template.tools_schema,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
            prepared.append(
                {
                    "template": template,
                    "verb": verb,
                    "user_content": user_content,
                    "input_ids": encoded["input_ids"][0],
                    "attention_mask": encoded["attention_mask"][0],
                }
            )

        verb_rows: list[dict[str, Any]] = []
        for start in range(0, len(prepared), args.batch_size):
            batch = prepared[start : start + args.batch_size]
            padded = tokenizer.pad(
                {
                    "input_ids": [row["input_ids"] for row in batch],
                    "attention_mask": [row["attention_mask"] for row in batch],
                },
                return_tensors="pt",
                padding=True,
            )
            input_ids = padded["input_ids"].to(input_device)
            attention_mask = padded["attention_mask"].to(input_device)
            with torch.inference_mode():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits.float().cpu()
            lengths = padded["attention_mask"].sum(dim=1).long().cpu()
            for idx, row in enumerate(batch):
                last_pos = int(lengths[idx].item()) - 1
                last_logits = logits[idx, last_pos, :]
                probs = torch.softmax(last_logits, dim=-1)
                top2_probs, top2_ids = torch.topk(probs, k=2)
                top1_id = int(top2_ids[0].item())
                top1_prob = float(top2_probs[0].item())
                second_prob = float(top2_probs[1].item())
                tool_prob = float(probs[tool_token_id].item())
                template = row["template"]
                record = {
                    "verb": verb,
                    "sample_id": template.sample_id,
                    "split": template.split,
                    "language": template.language,
                    "dataset_name": template.dataset_name,
                    "original_clean_candidate": template.original_clean_candidate,
                    "original_corrupt_candidate": template.original_corrupt_candidate,
                    "prompt_token_length": int(lengths[idx].item()),
                    "tool_token_id": tool_token_id,
                    "tool_token_prob": tool_prob,
                    "tool_token_logit": float(last_logits[tool_token_id].item()),
                    "top1_token_id": top1_id,
                    "top1_token_text": tokenizer.decode([top1_id], clean_up_tokenization_spaces=False),
                    "top1_prob": top1_prob,
                    "top1_margin": top1_prob - second_prob,
                    "is_tool_call_top1": bool(top1_id == tool_token_id),
                }
                verb_rows.append(record)
            del padded, input_ids, attention_mask, outputs, logits
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        per_prompt_rows.extend(verb_rows)

        by_language: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in verb_rows:
            by_language[str(row["language"])].append(row)

        summary_rows.append(
            {
                "verb": verb,
                "n_samples": len(verb_rows),
                "tool_call_top1_rate": mean([1.0 if row["is_tool_call_top1"] else 0.0 for row in verb_rows]),
                "tool_call_count": sum(1 for row in verb_rows if row["is_tool_call_top1"]),
                "mean_tool_call_prob": mean([float(row["tool_token_prob"]) for row in verb_rows]),
                "mean_tool_call_logit": mean([float(row["tool_token_logit"]) for row in verb_rows]),
                "mean_top1_margin": mean([float(row["top1_margin"]) for row in verb_rows]),
                "top1_mode_token": Counter(str(row["top1_token_text"]) for row in verb_rows).most_common(1)[0][0],
                "python_tool_call_top1_rate": mean([1.0 if row["is_tool_call_top1"] else 0.0 for row in by_language.get("python", [])]),
                "java_tool_call_top1_rate": mean([1.0 if row["is_tool_call_top1"] else 0.0 for row in by_language.get("java", [])]),
                "cpp_tool_call_top1_rate": mean([1.0 if row["is_tool_call_top1"] else 0.0 for row in by_language.get("cpp", [])]),
            }
        )

    summary_rows.sort(key=lambda row: (-float(row["tool_call_top1_rate"]), -float(row["mean_tool_call_prob"]), str(row["verb"])))
    write_csv(args.output_root / "summary_by_verb.csv", summary_rows)
    write_csv(args.output_root / "per_prompt.csv", per_prompt_rows)

    top10 = summary_rows[:10]
    md_lines = [
        "# Mistral Short-Prompt Execution Verb Scan",
        "",
        f"- Model: `{args.model_path}`",
        f"- Samples: `{args.n_samples}`",
        f"- Candidate verbs: `{len(verbs)}`",
        f"- Short system prompt: enabled",
        f"- Tool token: `{tool_token_text}` (id `{tool_token_id}`)",
        "",
        "## Top Verbs",
        "| verb | top1 rate | mean tool prob | mode token | python | java | cpp |",
        "| --- | ---: | ---: | --- | ---: | ---: | ---: |",
    ]
    for row in top10:
        md_lines.append(
            f"| `{row['verb']}` | {100.0 * float(row['tool_call_top1_rate']):.1f}% | {float(row['mean_tool_call_prob']):.4f} | `{row['top1_mode_token']}` | "
            f"{100.0 * float(row['python_tool_call_top1_rate']):.1f}% | {100.0 * float(row['java_tool_call_top1_rate']):.1f}% | {100.0 * float(row['cpp_tool_call_top1_rate']):.1f}% |"
        )
    (args.output_root / "summary.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
