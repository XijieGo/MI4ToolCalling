#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = Path(os.environ.get("QWEN3_8B_PATH", str(PROJECT_ROOT / "external" / "models" / "Qwen3-8B")))
DATASET_DIR = PROJECT_ROOT / "datasets" / "train" / "clean"
CANDIDATE_FILE = PROJECT_ROOT / "src" / "8B" / "neutral_verb_candidates_100.txt"
OUTPUT_ROOT = PROJECT_ROOT / "results" / "8B" / "neutral_verb_scan_train"
TOOL_CALL_STR = "<tool_call>"
USER_MARKER = "<|im_start|>user\n"
MANIFEST_NAME = "manifest.jsonl"


@dataclass(frozen=True)
class PromptTemplate:
    sample_id: str
    prompt_prefix: str
    prompt_suffix: str
    original_verb: str
    language: str
    dataset_name: str

    def render(self, candidate: str) -> str:
        rendered = candidate[:1].upper() + candidate[1:]
        return self.prompt_prefix + rendered + self.prompt_suffix


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scan candidate neutral verbs on Qwen3-8B.")
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH)
    parser.add_argument("--dataset-dir", type=Path, default=DATASET_DIR)
    parser.add_argument("--candidate-file", type=Path, default=CANDIDATE_FILE)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="bfloat16" if torch.cuda.is_available() else "float32")
    parser.add_argument("--limit-samples", type=int, default=0)
    parser.add_argument("--max-candidates", type=int, default=0)
    parser.add_argument("--write-details", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--candidates", nargs="*", default=None)
    return parser.parse_args()


def resolve_dtype(raw: str) -> torch.dtype:
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    key = raw.strip().lower()
    if key not in mapping:
        raise ValueError(f"Unsupported dtype: {raw}")
    return mapping[key]


def load_manifest_rows(path: Path) -> Dict[str, Dict[str, object]]:
    rows: Dict[str, Dict[str, object]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            filename = str(row.get("output_filename") or row.get("source_filename") or "")
            if filename:
                rows[filename] = row
    return rows


def extract_prompt_template(sample_id: str, text: str, metadata: Dict[str, object]) -> PromptTemplate:
    marker_pos = text.find(USER_MARKER)
    if marker_pos < 0:
        raise ValueError(f"{sample_id}: missing user marker")
    user_start = marker_pos + len(USER_MARKER)
    tail = text[user_start:]
    if not tail:
        raise ValueError(f"{sample_id}: empty user content")
    verb_end = 0
    while verb_end < len(tail) and not tail[verb_end].isspace():
        verb_end += 1
    if verb_end <= 0:
        raise ValueError(f"{sample_id}: failed to parse first verb")
    return PromptTemplate(
        sample_id=sample_id,
        prompt_prefix=text[:user_start],
        prompt_suffix=tail[verb_end:],
        original_verb=tail[:verb_end],
        language=str(metadata.get("language") or ""),
        dataset_name=str(metadata.get("dataset_name") or metadata.get("dataset") or ""),
    )


def load_templates(dataset_dir: Path, limit_samples: int) -> List[PromptTemplate]:
    manifest_path = dataset_dir / MANIFEST_NAME
    manifest_rows = load_manifest_rows(manifest_path)
    paths = sorted(p for p in dataset_dir.glob("*.txt"))
    if limit_samples > 0:
        paths = paths[:limit_samples]
    templates: List[PromptTemplate] = []
    for path in paths:
        metadata = manifest_rows.get(path.name, {})
        templates.append(
            extract_prompt_template(
                sample_id=path.stem,
                text=path.read_text(encoding="utf-8"),
                metadata=metadata,
            )
        )
    return templates


def load_candidates(args: argparse.Namespace) -> List[str]:
    if args.candidates:
        candidates = args.candidates
    else:
        with args.candidate_file.open("r", encoding="utf-8") as handle:
            candidates = [line.strip() for line in handle if line.strip() and not line.lstrip().startswith("#")]

    normalized: List[str] = []
    seen = set()
    for raw in candidates:
        candidate = raw.strip().lower()
        if not candidate:
            continue
        if any(ch.isspace() for ch in candidate):
            raise ValueError(f"Candidate must be a single word: {raw!r}")
        if candidate in seen:
            continue
        seen.add(candidate)
        normalized.append(candidate)
    if args.max_candidates > 0:
        normalized = normalized[: args.max_candidates]
    if not normalized:
        raise ValueError("No candidates loaded")
    return normalized


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


def load_completed_candidates(summary_path: Path) -> Dict[str, Dict[str, str]]:
    if not summary_path.exists():
        return {}
    with summary_path.open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return {str(row["candidate"]): row for row in rows}


def binary_entropy(probs: np.ndarray) -> np.ndarray:
    clipped = np.clip(probs.astype(np.float64, copy=False), 1e-9, 1.0 - 1e-9)
    return -(clipped * np.log2(clipped) + (1.0 - clipped) * np.log2(1.0 - clipped))


def summarize_candidate(
    candidate: str,
    templates: Sequence[PromptTemplate],
    probs: np.ndarray,
    top1_is_tool: np.ndarray,
    tokenizer,
) -> Dict[str, object]:
    entropy = binary_entropy(probs)
    top1_rate = float(top1_is_tool.mean())
    mean_prob = float(probs.mean())
    mean_entropy = float(entropy.mean())
    by_language: Dict[str, List[int]] = defaultdict(list)
    for template, flag in zip(templates, top1_is_tool.tolist()):
        by_language[template.language].append(int(flag))
    language_rates = {
        language: (sum(flags) / len(flags) if flags else float("nan"))
        for language, flags in sorted(by_language.items())
    }
    finite_language_rates = [value for value in language_rates.values() if np.isfinite(value)]
    language_balance_std = float(np.std(finite_language_rates)) if finite_language_rates else float("nan")
    word_token_length = len(tokenizer.encode(candidate[:1].upper() + candidate[1:], add_special_tokens=False))
    prob_mid_distance = abs(mean_prob - 0.5)
    top1_mid_distance = abs(top1_rate - 0.5)
    entropy_gap = 1.0 - mean_entropy if np.isfinite(mean_entropy) else 1.0
    composite_mid_score = prob_mid_distance + top1_mid_distance + entropy_gap
    return {
        "candidate": candidate,
        "n_samples": int(len(templates)),
        "word_token_length": int(word_token_length),
        "tool_call_top1_rate": top1_rate,
        "tool_call_count": int(top1_is_tool.sum()),
        "no_tool_count": int(len(top1_is_tool) - top1_is_tool.sum()),
        "mean_tool_call_prob": mean_prob,
        "median_tool_call_prob": float(np.median(probs)),
        "prob_q25": float(np.quantile(probs, 0.25)),
        "prob_q75": float(np.quantile(probs, 0.75)),
        "prob_std": float(np.std(probs)),
        "mean_binary_entropy_bits": mean_entropy,
        "borderline_10_90_rate": float(((probs >= 0.10) & (probs <= 0.90)).mean()),
        "borderline_25_75_rate": float(((probs >= 0.25) & (probs <= 0.75)).mean()),
        "prob_mid_distance": prob_mid_distance,
        "top1_mid_distance": top1_mid_distance,
        "language_balance_std": language_balance_std,
        "python_top1_rate": float(language_rates.get("python", float("nan"))),
        "java_top1_rate": float(language_rates.get("java", float("nan"))),
        "cpp_top1_rate": float(language_rates.get("cpp", float("nan"))),
        "composite_mid_score": composite_mid_score,
    }


def build_details_rows(
    candidate: str,
    templates: Sequence[PromptTemplate],
    probs: np.ndarray,
    top1_is_tool: np.ndarray,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for template, prob, flag in zip(templates, probs.tolist(), top1_is_tool.tolist()):
        rows.append(
            {
                "candidate": candidate,
                "sample_id": template.sample_id,
                "language": template.language,
                "dataset_name": template.dataset_name,
                "original_verb": template.original_verb,
                "tool_call_prob": float(prob),
                "is_tool_call_top1": int(flag),
            }
        )
    return rows


def build_markdown_summary(
    summary_rows: Sequence[Dict[str, object]],
    *,
    args: argparse.Namespace,
    templates: Sequence[PromptTemplate],
    candidates: Sequence[str],
) -> str:
    ranked = sorted(summary_rows, key=lambda row: (float(row["composite_mid_score"]), str(row["candidate"])))
    top_rows = ranked[:20]
    lines = [
        "# Neutral Verb Scan",
        "",
        "## Setup",
        f"- Model: `{args.model_path}`",
        f"- Dataset: `{args.dataset_dir}`",
        f"- Samples: `{len(templates)}`",
        f"- Candidates: `{len(candidates)}`",
        f"- Batch size: `{args.batch_size}`",
        f"- Ranking: `abs(top1_rate-0.5) + abs(mean_prob-0.5) + (1 - mean_binary_entropy_bits)`",
        "",
        "## Top Candidates",
        "| candidate | top1_rate | mean_prob | entropy(bits) | 25-75 rate | token_len | composite |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in top_rows:
        lines.append(
            f"| `{row['candidate']}` | {float(row['tool_call_top1_rate']):.4f} | {float(row['mean_tool_call_prob']):.4f} | {float(row['mean_binary_entropy_bits']):.4f} | {float(row['borderline_25_75_rate']):.4f} | {int(row['word_token_length'])} | {float(row['composite_mid_score']):.4f} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    dtype = resolve_dtype(args.dtype)
    if args.device.startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    templates = load_templates(args.dataset_dir.resolve(), args.limit_samples)
    candidates = load_candidates(args)
    args.output_root.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(str(args.model_path), trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.bos_token or "<|endoftext|>"
    tokenizer.padding_side = "left"

    tool_token_ids = tokenizer.encode(TOOL_CALL_STR, add_special_tokens=False)
    if len(tool_token_ids) != 1:
        raise ValueError(f"{TOOL_CALL_STR!r} encoded to {tool_token_ids}, expected one token")
    tool_token_id = int(tool_token_ids[0])

    model = AutoModelForCausalLM.from_pretrained(
        str(args.model_path),
        dtype=dtype,
        trust_remote_code=True,
    )
    model.to(args.device)
    model.eval()

    summary_path = args.output_root / "verb_summary.csv"
    details_dir = args.output_root / "per_candidate"
    completed = load_completed_candidates(summary_path) if args.resume else {}
    summary_rows: List[Dict[str, object]] = list(completed.values())

    started_at = time.time()
    for candidate in tqdm(candidates, desc="Scanning candidates", dynamic_ncols=True):
        if candidate in completed:
            continue

        prompts = [template.render(candidate) for template in templates]
        probs_parts: List[np.ndarray] = []
        top1_parts: List[np.ndarray] = []

        for start in range(0, len(prompts), args.batch_size):
            batch_prompts = prompts[start : start + args.batch_size]
            encoded = tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                add_special_tokens=False,
            )
            encoded = {key: value.to(args.device) for key, value in encoded.items()}
            with torch.inference_mode():
                outputs = model(**encoded, use_cache=False, logits_to_keep=1)
            last_logits = outputs.logits[:, -1, :].float()
            del outputs

            top1_ids = last_logits.argmax(dim=-1)
            tool_logits = last_logits[:, tool_token_id]
            tool_log_probs = tool_logits - torch.logsumexp(last_logits, dim=-1)
            tool_probs = torch.exp(tool_log_probs)

            probs_parts.append(tool_probs.detach().cpu().numpy())
            top1_parts.append(top1_ids.eq(tool_token_id).detach().cpu().numpy().astype(np.int8))
            del encoded, last_logits, top1_ids, tool_logits, tool_log_probs, tool_probs

        probs = np.concatenate(probs_parts, axis=0)
        top1_is_tool = np.concatenate(top1_parts, axis=0)
        summary_row = summarize_candidate(candidate, templates, probs, top1_is_tool, tokenizer)
        summary_rows.append(summary_row)
        summary_rows.sort(key=lambda row: (float(row["composite_mid_score"]), str(row["candidate"])))
        write_csv(summary_path, summary_rows)

        if args.write_details:
            details_rows = build_details_rows(candidate, templates, probs, top1_is_tool)
            write_csv(details_dir / f"{candidate}.csv", details_rows)

        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    elapsed = time.time() - started_at
    summary_rows.sort(key=lambda row: (float(row["composite_mid_score"]), str(row["candidate"])))
    write_csv(summary_path, summary_rows)
    markdown = build_markdown_summary(summary_rows, args=args, templates=templates, candidates=candidates)
    (args.output_root / "summary.md").write_text(markdown, encoding="utf-8")
    run_info = {
        "model_path": str(args.model_path),
        "dataset_dir": str(args.dataset_dir),
        "n_samples": len(templates),
        "n_candidates": len(candidates),
        "batch_size": args.batch_size,
        "device": args.device,
        "dtype": args.dtype,
        "elapsed_seconds": elapsed,
    }
    (args.output_root / "run_info.json").write_text(json.dumps(run_info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
