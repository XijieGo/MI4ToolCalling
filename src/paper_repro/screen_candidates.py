#!/usr/bin/env python3
"""Re-screen candidate clean/corrupt verbs from the archived base-pair pool.

This is the forward data-generation step omitted from the earlier sync.  It is
deliberately separate from the frozen v2 replay: model updates can change which
pairs pass, so a fresh screen must be recorded as a *new* dataset version.
"""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .dataset import replace_instruction_verb, write_json
from .paths import PROVENANCE_ROOT


CLEAN_CANDIDATES = ("complete", "build", "write", "save", "add")
CORRUPT_CANDIDATES = ("discuss", "explore", "inspect", "review", "study")

# The submitted appendix reports a wider one-verb-at-a-time audit than the
# five-by-five pool used to construct paired data.  Keep both protocols here
# rather than silently deriving one from a historical CSV.
APPENDIX_CLEAN_CANDIDATES = (
    "add",
    "build",
    "complete",
    "create",
    "generate",
    "implement",
    "modify",
    "save",
    "update",
    "write",
)
APPENDIX_CORRUPT_CANDIDATES = (
    "analyze",
    "assess",
    "compare",
    "discuss",
    "examine",
    "explore",
    "inspect",
    "review",
    "summarize",
    "study",
)
CANDIDATE_PROFILES: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "final_pool": (CLEAN_CANDIDATES, CORRUPT_CANDIDATES),
    "appendix_audit": (APPENDIX_CLEAN_CANDIDATES, APPENDIX_CORRUPT_CANDIDATES),
}


@dataclass(frozen=True)
class CandidatePrompt:
    sample_id: str
    filename: str
    language: str
    dataset_name: str
    pool: str
    candidate: str
    prompt: str
    rendered_first_user_line: str


def parse_model_spec(raw: str) -> tuple[str, Path]:
    if "=" not in raw:
        raise argparse.ArgumentTypeError("Use NAME=PATH, e.g. Qwen3-8B=/models/Qwen3-8B")
    name, raw_path = raw.split("=", 1)
    name = name.strip()
    path = Path(raw_path).expanduser()
    if not name or not raw_path:
        raise argparse.ArgumentTypeError("Model name and path must both be non-empty")
    return name, path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run first-token behavioral screening for all v1 base-pair verb candidates.")
    parser.add_argument(
        "--source-root",
        type=Path,
        default=PROVENANCE_ROOT / "v1_1711" / "base_pairs",
        help="Archived base pairs; only this source is used to render candidate prompts.",
    )
    parser.add_argument(
        "--metadata-csv",
        type=Path,
        default=PROVENANCE_ROOT / "v1_1711" / "metadata" / "dataset_assignment_audit.csv",
        help="Provides stable source/language metadata for the archived base pairs.",
    )
    parser.add_argument("--model", action="append", type=parse_model_spec, required=True, metavar="NAME=PATH")
    parser.add_argument(
        "--candidate-profile",
        choices=tuple(CANDIDATE_PROFILES),
        default="final_pool",
        help="Verb protocol: final_pool is used for fresh paired-data selection; appendix_audit reproduces the 10+10 Appendix A screen.",
    )
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--metadata-output", type=Path, default=None)
    parser.add_argument("--max-pairs", type=int, default=0, help="0 means all archived base pairs.")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    return parser.parse_args()


def dtype_from_name(name: str) -> torch.dtype:
    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[name]


def source_metadata(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {row["filename"]: row for row in rows}


def build_candidate_prompts(
    source_root: Path,
    metadata: dict[str, dict[str, str]],
    max_pairs: int,
    *,
    clean_candidates: tuple[str, ...],
    corrupt_candidates: tuple[str, ...],
) -> list[CandidatePrompt]:
    prompts: list[CandidatePrompt] = []
    files = sorted((source_root / "clean").glob("*.txt"))
    if max_pairs > 0:
        files = files[:max_pairs]
    for clean_path in files:
        row = metadata.get(clean_path.name)
        if row is None:
            raise KeyError(f"No archived metadata row for {clean_path.name}")
        scaffold = clean_path.read_text(encoding="utf-8")
        for pool, candidates in (("clean", clean_candidates), ("corrupt", corrupt_candidates)):
            for candidate in candidates:
                prompt, rendered_line = replace_instruction_verb(scaffold, candidate)
                prompts.append(
                    CandidatePrompt(
                        sample_id=row["sample_id"],
                        filename=clean_path.name,
                        language=row["language"],
                        dataset_name=row["dataset_name"],
                        pool=pool,
                        candidate=candidate,
                        prompt=prompt,
                        rendered_first_user_line=rendered_line,
                    )
                )
    return prompts


def batched(items: list[CandidatePrompt], size: int) -> Iterable[list[CandidatePrompt]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def evaluate_model(
    *, model_name: str, model_path: Path, prompts: list[CandidatePrompt], batch_size: int, device: str, dtype: torch.dtype
) -> tuple[dict[tuple[str, str, str], dict[str, object]], dict[str, object]]:
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tool_token_id = int(tokenizer.convert_tokens_to_ids("<tool_call>"))
    if tool_token_id < 0 or tool_token_id == tokenizer.unk_token_id:
        raise ValueError(f"{model_name}: <tool_call> is not a dedicated tokenizer token")
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype, trust_remote_code=True)
    model.to(device)
    model.eval()
    result: dict[tuple[str, str, str], dict[str, object]] = {}
    with torch.inference_mode():
        for batch in batched(prompts, batch_size):
            encoded = tokenizer([item.prompt for item in batch], return_tensors="pt", padding=True, add_special_tokens=False)
            encoded = {key: value.to(device) for key, value in encoded.items()}
            # Qwen3 supports ``logits_to_keep``; retaining only the decision
            # position avoids materializing logits for every prompt token,
            # which is important when screening the full 1,711-row pool with
            # the 14B checkpoint.  Keep a compatibility fallback for older
            # Transformers/model implementations.
            try:
                logits = model(**encoded, use_cache=False, logits_to_keep=1).logits[:, -1, :].float()
            except TypeError:
                logits = model(**encoded, use_cache=False).logits[:, -1, :].float()
            probabilities = torch.softmax(logits, dim=-1)
            top_probs, top_ids = probabilities.max(dim=-1)
            tool_probs = probabilities[:, tool_token_id]
            for index, item in enumerate(batch):
                top_id = int(top_ids[index].item())
                result[(item.pool, item.candidate, item.sample_id)] = {
                    "tool_call_prob": float(tool_probs[index].item()),
                    "is_tool_call_top1": top_id == tool_token_id,
                    "top1_token_id": top_id,
                    "top1_token_text": tokenizer.decode([top_id], clean_up_tokenization_spaces=False),
                    "top1_prob": float(top_probs[index].item()),
                }
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result, {"model_name": model_name, "model_path": str(model_path), "tool_token_id": tool_token_id}


def main() -> None:
    args = parse_args()
    model_specs: list[tuple[str, Path]] = args.model
    model_names = [name for name, _path in model_specs]
    if len(model_names) != len(set(model_names)):
        raise ValueError("Model names must be unique")
    clean_candidates, corrupt_candidates = CANDIDATE_PROFILES[args.candidate_profile]
    prompts = build_candidate_prompts(
        args.source_root.resolve(),
        source_metadata(args.metadata_csv.resolve()),
        args.max_pairs,
        clean_candidates=clean_candidates,
        corrupt_candidates=corrupt_candidates,
    )
    model_results: dict[str, dict[tuple[str, str, str], dict[str, object]]] = {}
    model_metadata: list[dict[str, object]] = []
    for model_name, model_path in model_specs:
        output, metadata = evaluate_model(
            model_name=model_name,
            model_path=model_path,
            prompts=prompts,
            batch_size=args.batch_size,
            device=args.device,
            dtype=dtype_from_name(args.dtype),
        )
        model_results[model_name] = output
        model_metadata.append(metadata)

    rows: list[dict[str, object]] = []
    for item in prompts:
        row: dict[str, object] = {
            "sample_id": item.sample_id,
            "filename": item.filename,
            "language": item.language,
            "dataset_name": item.dataset_name,
            "pool": item.pool,
            "candidate": item.candidate,
        }
        checks: list[bool] = []
        key = (item.pool, item.candidate, item.sample_id)
        for model_name in model_names:
            metrics = model_results[model_name][key]
            row.update({f"{model_name}_{name}": value for name, value in metrics.items()})
            expected_tool_call = item.pool == "clean"
            checks.append(bool(metrics["is_tool_call_top1"]) == expected_tool_call)
        row["rendered_first_user_line"] = item.rendered_first_user_line
        row["all_models_valid"] = all(checks)
        rows.append(row)
    fieldnames = list(rows[0]) if rows else []
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    metadata_path = args.metadata_output or args.output_csv.with_suffix(".metadata.json")
    write_json(
        metadata_path,
        {
            "source_root": str(args.source_root.resolve()),
            "metadata_csv": str(args.metadata_csv.resolve()),
            "candidate_profile": args.candidate_profile,
            "candidate_pools": {"clean": list(clean_candidates), "corrupt": list(corrupt_candidates)},
            "n_base_pairs": len({item.filename for item in prompts}),
            "n_candidate_prompts": len(prompts),
            "models": model_metadata,
            "device": args.device,
            "dtype": args.dtype,
        },
    )
    print(f"Wrote {len(rows)} screening rows to {args.output_csv}")


if __name__ == "__main__":
    main()
