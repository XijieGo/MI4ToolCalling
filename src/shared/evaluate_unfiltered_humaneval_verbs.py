#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import csv
import gc
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import pyarrow.parquet as pq
import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


ACTION_VERBS: Tuple[str, ...] = ("add", "build", "complete", "save", "write")
ANALYSIS_VERBS: Tuple[str, ...] = ("discuss", "explore", "inspect", "review", "study")
DEFAULT_MODELS: Tuple[str, ...] = (
    "Qwen3-1.7B",
    "Qwen3-4B",
    "Qwen3-8B",
    "Qwen3-14B",
    "Qwen3-32B",
)

USER_MARKER = "<|im_start|>user\n"
ASSISTANT_MARKER = "<|im_end|>\n<|im_start|>assistant\n"
FIRST_LINE_TEMPLATE = "{verb} the function body in solve.py based on the function definition and docstring below:"


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key in seen:
                continue
            seen.add(key)
            fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def safe_mean(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    if not vals:
        return float("nan")
    return float(sum(vals) / len(vals))


def safe_median(values: Iterable[float]) -> float:
    vals = sorted(float(v) for v in values if math.isfinite(float(v)))
    if not vals:
        return float("nan")
    n = len(vals)
    mid = n // 2
    if n % 2 == 1:
        return vals[mid]
    return float((vals[mid - 1] + vals[mid]) / 2.0)


def safe_rate(values: Iterable[bool]) -> float:
    vals = [1.0 if bool(v) else 0.0 for v in values]
    if not vals:
        return float("nan")
    return float(sum(vals) / len(vals))


def split_prompt_outer(text: str) -> Tuple[str, str]:
    if USER_MARKER not in text or ASSISTANT_MARKER not in text:
        raise ValueError("template prompt is missing expected markers")
    prefix, after_user = text.split(USER_MARKER, 1)
    _user_content, suffix_tail = after_user.split(ASSISTANT_MARKER, 1)
    return prefix + USER_MARKER, ASSISTANT_MARKER + suffix_tail


def discover_template_parts(dataset_root: Path) -> Tuple[str, str]:
    prompt_files = sorted((dataset_root / "clean").glob("humaneval_python_*.txt"))
    if not prompt_files:
        raise FileNotFoundError(f"no Humaneval prompt template found under {dataset_root / 'clean'}")
    template_text = prompt_files[0].read_text(encoding="utf-8")
    return split_prompt_outer(template_text)


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def first_sentence(text: str) -> str:
    cleaned = normalize_whitespace(text)
    if not cleaned:
        return ""
    match = re.search(r"(.+?[.!?])(?:\s|$)", cleaned)
    if match:
        return match.group(1).strip()
    return cleaned


def first_doctest_block(lines: Sequence[str]) -> List[str]:
    start_idx = -1
    for idx, line in enumerate(lines):
        if line.lstrip().startswith(">>>"):
            start_idx = idx
            break
    if start_idx < 0:
        return []

    block: List[str] = []
    block.append(lines[start_idx].rstrip())
    idx = start_idx + 1
    while idx < len(lines):
        line = lines[idx]
        stripped = line.strip()
        if not stripped:
            break
        if line.lstrip().startswith(">>>"):
            break
        block.append(line.rstrip())
        idx += 1
    return block


def compress_docstring(raw_docstring: str) -> str:
    lines = [line.rstrip() for line in raw_docstring.strip().splitlines()]
    doctest_block = first_doctest_block(lines)

    description_lines: List[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith(">>>"):
            break
        if not stripped:
            continue
        lowered = stripped.lower()
        if lowered in {"examples", "example", "note:", "notes:"}:
            continue
        description_lines.append(stripped)

    summary = first_sentence(" ".join(description_lines))
    if doctest_block:
        if summary:
            return summary + "\n\n" + "\n".join(doctest_block)
        return "\n".join(doctest_block)
    return summary


def render_docstring_block(
    doc_content: str,
    *,
    indent: str,
    inline_opening: bool,
) -> List[str]:
    content = doc_content.replace('"""', '\\"""').strip("\n")
    lines = content.splitlines() if content else []
    if not lines:
        return [indent + '"""', indent + '"""']
    if inline_opening:
        out = [indent + '"""' + lines[0]]
        out.extend(indent + line for line in lines[1:])
        out.append(indent + '"""')
        return out
    out = [indent + '"""']
    out.extend(indent + line for line in lines)
    out.append(indent + '"""')
    return out


def extract_target_stub(raw_prompt: str) -> str:
    module = ast.parse(raw_prompt)
    function_nodes = [node for node in module.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
    if not function_nodes:
        raise ValueError("failed to find target function in Humaneval prompt")

    target = function_nodes[-1]
    doc_node = None
    for stmt in target.body:
        if (
            isinstance(stmt, ast.Expr)
            and isinstance(getattr(stmt, "value", None), ast.Constant)
            and isinstance(stmt.value.value, str)
        ):
            doc_node = stmt
            break
    if doc_node is None:
        raise ValueError("target function is missing a recoverable docstring body")

    raw_docstring = str(doc_node.value.value or "")
    compressed_docstring = compress_docstring(raw_docstring)
    if not compressed_docstring:
        compressed_docstring = normalize_whitespace(raw_docstring)

    source_lines = raw_prompt.splitlines()
    start_line = min([target.lineno] + [dec.lineno for dec in target.decorator_list]) - 1
    doc_line = doc_node.lineno - 1
    header_lines = source_lines[start_line:doc_line]
    if not header_lines:
        raise ValueError("failed to recover target function header")

    first_doc_line = source_lines[doc_line] if doc_line < len(source_lines) else ""
    inline_opening = bool(re.match(r'^\s*(?:"""|\'\'\')\S', first_doc_line))
    indent = re.match(r"^(\s*)", first_doc_line).group(1) if first_doc_line else "    "

    rendered_lines = [line.rstrip() for line in header_lines]
    rendered_lines.extend(
        render_docstring_block(
            compressed_docstring,
            indent=indent,
            inline_opening=inline_opening,
        )
    )
    rendered_lines.append(indent + "pass")
    return "\n".join(rendered_lines).rstrip() + "\n"


def load_humaneval_rows(parquet_path: Path) -> List[Dict[str, object]]:
    table = pq.read_table(parquet_path)
    rows: List[Dict[str, object]] = []
    for idx in range(table.num_rows):
        task_id = str(table["task_id"][idx].as_py())
        prompt = str(table["prompt"][idx].as_py())
        entry_point = str(table["entry_point"][idx].as_py())
        sample_index = idx + 1
        stub_text = extract_target_stub(prompt)
        rows.append(
            {
                "sample_id": f"humaneval_python_{sample_index}",
                "sample_index": sample_index,
                "task_id": task_id,
                "entry_point": entry_point,
                "stub_text": stub_text,
                "raw_prompt_char_length": len(prompt),
                "stub_char_length": len(stub_text),
            }
        )
    return rows


def build_prompt_rows(
    humaneval_rows: Sequence[Dict[str, object]],
    *,
    prompt_prefix: str,
    prompt_suffix: str,
    tokenizer,
) -> List[Dict[str, object]]:
    out_rows: List[Dict[str, object]] = []
    for row in humaneval_rows:
        stub_text = str(row["stub_text"]).rstrip("\n")
        for verb_group, verbs in [("action", ACTION_VERBS), ("analysis", ANALYSIS_VERBS)]:
            for verb in verbs:
                first_line = FIRST_LINE_TEMPLATE.format(verb=verb.capitalize())
                prompt_text = prompt_prefix + first_line + "\n" + stub_text + "\n" + prompt_suffix
                input_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
                out_rows.append(
                    {
                        "sample_id": row["sample_id"],
                        "sample_index": int(row["sample_index"]),
                        "task_id": row["task_id"],
                        "entry_point": row["entry_point"],
                        "verb": verb,
                        "verb_group": verb_group,
                        "first_user_line": first_line,
                        "prompt_token_length": len(input_ids),
                        "raw_prompt_char_length": int(row["raw_prompt_char_length"]),
                        "stub_char_length": int(row["stub_char_length"]),
                        "prompt_text": prompt_text,
                    }
                )
    return out_rows


def ensure_padding_token(tokenizer) -> None:
    if tokenizer.pad_token_id is not None:
        return
    if tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    elif tokenizer.bos_token is not None:
        tokenizer.pad_token = tokenizer.bos_token
    else:
        tokenizer.add_special_tokens({"pad_token": "<|pad|>"})


def load_model_and_tokenizer(model_path: Path, device: str):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    ensure_padding_token(tokenizer)
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map="auto" if device.startswith("cuda") else None,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    if not device.startswith("cuda"):
        model.to(device)
    model.eval()
    return model, tokenizer


def evaluate_prompt_rows(
    model,
    tokenizer,
    rows: Sequence[Dict[str, object]],
    *,
    tool_token_id: int,
    batch_size: int,
    desc: str,
) -> List[Dict[str, object]]:
    out_rows: List[Dict[str, object]] = []
    start = 0
    current_batch_size = max(1, int(batch_size))

    pbar = tqdm(total=len(rows), desc=desc, dynamic_ncols=True)
    while start < len(rows):
        this_batch_size = min(current_batch_size, len(rows) - start)
        batch_rows = rows[start : start + this_batch_size]
        texts = [str(row["prompt_text"]) for row in batch_rows]
        try:
            encoded = tokenizer(
                texts,
                padding=True,
                truncation=False,
                return_tensors="pt",
                add_special_tokens=False,
            )
            encoded = {key: value.to(model.device) for key, value in encoded.items()}

            with torch.inference_mode():
                outputs = model(**encoded, use_cache=False)
                logits = outputs.logits

            last_positions = encoded["attention_mask"].sum(dim=1) - 1
            batch_indices = torch.arange(logits.shape[0], device=logits.device)
            last_logits = logits[batch_indices, last_positions]
            last_logits_fp32 = last_logits.float()
            probs = torch.softmax(last_logits_fp32, dim=-1)
            tool_probs = probs[:, tool_token_id].detach().cpu()
            top_ids = last_logits.argmax(dim=-1).detach().cpu()
            top_probs = probs.max(dim=-1).values.detach().cpu()

            for local_idx, row in enumerate(batch_rows):
                top_id = int(top_ids[local_idx].item())
                out_rows.append(
                    {
                        **{key: value for key, value in row.items() if key != "prompt_text"},
                        "tool_call_prob": float(tool_probs[local_idx].item()),
                        "is_tool_call_top1": int(top_id == tool_token_id),
                        "top1_token_id": top_id,
                        "top1_token_text": tokenizer.decode([top_id]),
                        "top1_token_piece": tokenizer.convert_ids_to_tokens([top_id])[0],
                        "top1_prob": float(top_probs[local_idx].item()),
                    }
                )

            start += len(batch_rows)
            pbar.update(len(batch_rows))
            pbar.set_postfix(batch=this_batch_size)

            del encoded
            del outputs
            del logits
            del last_logits
            del last_logits_fp32
            del probs
            del tool_probs
            del top_ids
            del top_probs
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower():
                raise
            if this_batch_size <= 1:
                raise
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            current_batch_size = max(1, this_batch_size // 2)
            pbar.write(f"[warn] OOM at batch_size={this_batch_size}; retry with {current_batch_size}")

    pbar.close()
    return out_rows


def summarize_rows(rows: Sequence[Dict[str, object]], group_keys: Sequence[str]) -> List[Dict[str, object]]:
    grouped: Dict[Tuple[object, ...], List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(key) for key in group_keys)].append(dict(row))

    out_rows: List[Dict[str, object]] = []
    for group_values, group_rows in sorted(grouped.items(), key=lambda item: tuple("" if v is None else str(v) for v in item[0])):
        token_counter = Counter(str(r["top1_token_text"]) for r in group_rows)
        mode_token, mode_count = ("", 0)
        if token_counter:
            mode_token, mode_count = token_counter.most_common(1)[0]
        out_rows.append(
            {
                **{key: value for key, value in zip(group_keys, group_values)},
                "n_samples": len(group_rows),
                "tool_call_top1_count": sum(int(r["is_tool_call_top1"]) for r in group_rows),
                "tool_call_top1_rate": safe_rate(bool(r["is_tool_call_top1"]) for r in group_rows),
                "tool_call_prob_mean": safe_mean(float(r["tool_call_prob"]) for r in group_rows),
                "tool_call_prob_median": safe_median(float(r["tool_call_prob"]) for r in group_rows),
                "borderline_25_75_count": sum(1 for r in group_rows if 0.25 <= float(r["tool_call_prob"]) <= 0.75),
                "borderline_25_75_rate": safe_rate(0.25 <= float(r["tool_call_prob"]) <= 0.75 for r in group_rows),
                "top1_token_mode": mode_token,
                "top1_token_mode_count": mode_count,
                "top1_token_mode_share": safe_rate(str(r["top1_token_text"]) == mode_token for r in group_rows),
                "top1_prob_mean": safe_mean(float(r["top1_prob"]) for r in group_rows),
                "prompt_token_length_mean": safe_mean(float(r["prompt_token_length"]) for r in group_rows),
            }
        )
    return out_rows


def model_batch_size(model_name: str, override: int) -> int:
    if override > 0:
        return override
    if "32b" in model_name.lower():
        return 8
    if "14b" in model_name.lower():
        return 16
    if "8b" in model_name.lower():
        return 24
    if "4b" in model_name.lower():
        return 32
    return 48


def cleanup_model(model, tokenizer) -> None:
    del model
    del tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def build_report(model_summaries: Dict[str, List[Dict[str, object]]], output_root: Path) -> None:
    lines = ["# Unfiltered HumanEval Verb Evaluation", ""]
    for model_name in sorted(model_summaries.keys(), key=lambda name: DEFAULT_MODELS.index(name) if name in DEFAULT_MODELS else name):
        lines.append(f"## {model_name}")
        rows = model_summaries[model_name]
        for row in rows:
            lines.append(
                "- `{verb}` ({verb_group}): top-1 `<tool_call>` {tool_call_top1_count}/{n_samples} "
                "({tool_call_top1_rate:.4f}), mean p(`<tool_call>`)={tool_call_prob_mean:.4f}, "
                "borderline={borderline_25_75_rate:.4f}, mode=`{top1_token_mode}`.".format(**row)
            )
        lines.append("")
    (output_root / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate first-token tool-call behavior for 10 verbs on unfiltered HumanEval.")
    parser.add_argument(
        "--humaneval-parquet",
        type=Path,
        default=Path("./datasets/humaneval/openai_humaneval/test-00000-of-00001.parquet"),
    )
    parser.add_argument(
        "--template-dataset-root",
        type=Path,
        default=Path("./datasets"),
    )
    parser.add_argument(
        "--qwen-root",
        type=Path,
        default=Path("./external/models"),
    )
    parser.add_argument(
        "--models",
        type=str,
        default=",".join(DEFAULT_MODELS),
        help="Comma-separated model directory names under --qwen-root.",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("./results/humaneval_unfiltered_verb_eval"),
    )
    args = parser.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = True

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    model_names = [chunk.strip() for chunk in args.models.split(",") if chunk.strip()]
    model_paths = []
    for model_name in model_names:
        model_path = (args.qwen_root / model_name).resolve()
        if not model_path.exists():
            raise FileNotFoundError(f"model path does not exist: {model_path}")
        model_paths.append((model_name, model_path))

    prompt_prefix, prompt_suffix = discover_template_parts(args.template_dataset_root.resolve())
    template_tokenizer = AutoTokenizer.from_pretrained(str(model_paths[0][1]), trust_remote_code=True)
    humaneval_rows = load_humaneval_rows(args.humaneval_parquet.resolve())
    if args.max_samples > 0:
        humaneval_rows = humaneval_rows[: args.max_samples]
    prompt_rows = build_prompt_rows(
        humaneval_rows,
        prompt_prefix=prompt_prefix,
        prompt_suffix=prompt_suffix,
        tokenizer=template_tokenizer,
    )
    del template_tokenizer

    write_jsonl(output_root / "generated_prompts.jsonl", prompt_rows)
    write_jsonl(
        output_root / "generated_prompt_manifest.jsonl",
        [{key: value for key, value in row.items() if key != "prompt_text"} for row in prompt_rows],
    )

    model_level_summary: Dict[str, List[Dict[str, object]]] = {}
    cross_model_rows: List[Dict[str, object]] = []

    for model_name, model_path in model_paths:
        model_out = output_root / model_name
        model_out.mkdir(parents=True, exist_ok=True)
        model, tokenizer = load_model_and_tokenizer(model_path, args.device)
        tool_token_ids = tokenizer.encode("<tool_call>", add_special_tokens=False)
        if len(tool_token_ids) != 1:
            raise ValueError(f"<tool_call> is not single-token for {model_name}: {tool_token_ids}")
        tool_token_id = int(tool_token_ids[0])

        rows_for_eval = []
        for row in prompt_rows:
            copied = dict(row)
            copied["prompt_token_length"] = len(tokenizer(str(copied["prompt_text"]), add_special_tokens=False)["input_ids"])
            rows_for_eval.append(copied)

        eval_rows = evaluate_prompt_rows(
            model,
            tokenizer,
            rows_for_eval,
            tool_token_id=tool_token_id,
            batch_size=model_batch_size(model_name, args.batch_size),
            desc=f"{model_name} Humaneval verbs",
        )
        by_verb_rows = summarize_rows(eval_rows, group_keys=["verb_group", "verb"])
        by_group_rows = summarize_rows(eval_rows, group_keys=["verb_group"])
        by_verb_rows = [{**row, "model_name": model_name} for row in by_verb_rows]
        by_group_rows = [{**row, "model_name": model_name} for row in by_group_rows]

        write_csv(model_out / "per_prompt.csv", eval_rows)
        write_csv(model_out / "summary_by_verb.csv", by_verb_rows)
        write_csv(model_out / "summary_by_group.csv", by_group_rows)
        write_json(
            model_out / "summary.json",
            {
                "model_name": model_name,
                "model_path": str(model_path),
                "tool_token_id": tool_token_id,
                "n_prompts": len(eval_rows),
                "n_samples": len(humaneval_rows),
                "verbs": {
                    "action": list(ACTION_VERBS),
                    "analysis": list(ANALYSIS_VERBS),
                },
                "artifacts": {
                    "per_prompt_csv": str(model_out / "per_prompt.csv"),
                    "summary_by_verb_csv": str(model_out / "summary_by_verb.csv"),
                    "summary_by_group_csv": str(model_out / "summary_by_group.csv"),
                },
            },
        )

        model_level_summary[model_name] = by_verb_rows
        cross_model_rows.extend(by_verb_rows)
        cleanup_model(model, tokenizer)

    write_csv(output_root / "summary_all_models_by_verb.csv", cross_model_rows)
    build_report(model_level_summary, output_root)
    write_json(
        output_root / "run_summary.json",
        {
            "humaneval_parquet": str(args.humaneval_parquet.resolve()),
            "template_dataset_root": str(args.template_dataset_root.resolve()),
            "qwen_root": str(args.qwen_root.resolve()),
            "models": [name for name, _ in model_paths],
            "n_samples": len(humaneval_rows),
            "n_prompts_per_model": len(prompt_rows),
            "verbs": {
                "action": list(ACTION_VERBS),
                "analysis": list(ANALYSIS_VERBS),
            },
            "artifacts": {
                "generated_prompts_jsonl": str(output_root / "generated_prompts.jsonl"),
                "generated_prompt_manifest_jsonl": str(output_root / "generated_prompt_manifest.jsonl"),
                "summary_all_models_by_verb_csv": str(output_root / "summary_all_models_by_verb.csv"),
                "report_md": str(output_root / "report.md"),
            },
        },
    )


if __name__ == "__main__":
    main()
