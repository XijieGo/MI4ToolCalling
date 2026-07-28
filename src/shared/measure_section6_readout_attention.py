#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gc
import importlib.util
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from tqdm.auto import tqdm

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from artifact_paths import (
    DEVSTRAL_2_24B_PATH,
    GRANITE_3P3_8B_PATH,
    MISTRAL_3P2_24B_PATH,
    QWEN35_9B_PATH,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_CONFIGS = {
    "mistral": {
        "display_model": "Mistral-3.2-24B",
        "model_path": MISTRAL_3P2_24B_PATH,
        "loader": "mistral",
    },
    "devstral": {
        "display_model": "Devstral-2-24B",
        "model_path": DEVSTRAL_2_24B_PATH,
        "loader": "devstral",
    },
    "granite": {
        "display_model": "Granite-3.3-8B",
        "model_path": GRANITE_3P3_8B_PATH,
        "loader": "auto",
    },
    "qwen35": {
        "display_model": "Qwen3.5-9B",
        "model_path": QWEN35_9B_PATH,
        "loader": "qwen35",
    },
}


@dataclass(frozen=True)
class PairRecord:
    pair_id: int
    sample_id: str
    clean_text: str
    corrupt_text: str
    clean_ids: torch.Tensor
    corrupt_ids: torch.Tensor
    clean_path: str
    corrupt_path: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure true readout attention mass to tool/system/schema regions.")
    parser.add_argument("--model-key", choices=sorted(MODEL_CONFIGS), required=True)
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--dataset-root", type=Path, required=True, help="Fresh model-native selected-pair root.")
    parser.add_argument("--candidate-csv", type=Path, required=True, help="Current-run mechanism candidate-head CSV.")
    parser.add_argument("--output-root", type=Path, required=True, help="Fresh output directory.")
    parser.add_argument("--max-pairs", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--top-heads", type=int, default=10)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--attn-implementation", type=str, default="eager")
    parser.add_argument("--min-free-vram-gib", type=float, default=0.0)
    parser.add_argument("--wait-poll-seconds", type=int, default=30)
    return parser.parse_args()


def clear_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def dtype_from_name(name: str) -> torch.dtype:
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    if name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def query_free_vram_gib(device_index: int = 0) -> float:
    output = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
        text=True,
    )
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    if device_index >= len(lines):
        raise IndexError(f"GPU index {device_index} out of range.")
    return float(lines[device_index]) / 1024.0


def wait_for_vram(required_gib: float, poll_seconds: int) -> None:
    if required_gib <= 0:
        return
    free_gib = query_free_vram_gib()
    while free_gib < required_gib:
        print(
            json.dumps(
                {"event": "wait_for_vram", "required_gib": required_gib, "free_gib": free_gib},
                ensure_ascii=False,
            ),
            flush=True,
        )
        time.sleep(max(int(poll_seconds), 1))
        free_gib = query_free_vram_gib()


def import_from_dir(module_dir: Path, module_name: str):
    if str(module_dir) not in sys.path:
        sys.path.insert(0, str(module_dir))
    return __import__(module_name)


def load_model_and_tokenizer(model_key: str, model_path: Path, dtype: torch.dtype, attn_implementation: str):
    loader = MODEL_CONFIGS[model_key]["loader"]
    if loader == "mistral":
        helper_dir = PROJECT_ROOT / "src/mistral"
        helper = import_from_dir(helper_dir, "hf_patch_utils")
        tokenizer = helper.load_tokenizer(model_path)
        try:
            model = helper.load_model(
                model_path,
                dtype=dtype,
                device_map={"": 0},
                attn_implementation=attn_implementation,
            )
        except TypeError:
            model = helper.load_model(model_path, dtype=dtype, device_map={"": 0})
    elif loader == "devstral":
        helper_dir = PROJECT_ROOT / "src/devstral"
        helper = import_from_dir(helper_dir, "hf_patch_utils")
        tokenizer = helper.load_tokenizer(model_path)
        try:
            model = helper.load_model(
                model_path,
                dtype=dtype,
                device_map={"": 0},
                attn_implementation=attn_implementation,
            )
        except TypeError:
            model = helper.load_model(model_path, dtype=dtype, device_map={"": 0})
    else:
        if loader == "qwen35":
            original_find_spec = importlib.util.find_spec

            def patched_find_spec(name: str, package: str | None = None):
                if name == "sklearn" and os.environ.get("MECH_ENABLE_TRANSFORMERS_SKLEARN", "0") != "1":
                    return None
                return original_find_spec(name, package)

            importlib.util.find_spec = patched_find_spec
            try:
                from transformers import AutoModelForCausalLM, AutoTokenizer
            finally:
                importlib.util.find_spec = original_find_spec
        else:
            from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
        kwargs: dict[str, Any] = {
            "torch_dtype": dtype,
            "device_map": {"": 0},
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
        }
        if attn_implementation:
            kwargs["attn_implementation"] = attn_implementation
        try:
            model = AutoModelForCausalLM.from_pretrained(str(model_path), **kwargs)
        except TypeError:
            kwargs.pop("attn_implementation", None)
            model = AutoModelForCausalLM.from_pretrained(str(model_path), **kwargs)

    model.eval()
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    if not getattr(tokenizer, "padding_side", None):
        tokenizer.padding_side = "right"
    try:
        model.config.output_attentions = True
    except Exception:
        pass
    return model, tokenizer


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def encode_text(tokenizer, text: str) -> torch.Tensor:
    return tokenizer(text, add_special_tokens=False, return_tensors="pt")["input_ids"][0].detach().cpu()


def resolve_prompt_path(raw_path: str | Path, dataset_root: Path) -> Path:
    path = Path(raw_path)
    if path.exists():
        return path
    if path.is_absolute():
        return path
    artifact_candidate = PROJECT_ROOT / path
    if artifact_candidate.exists():
        return artifact_candidate
    dataset_candidate = dataset_root / path
    if dataset_candidate.exists():
        return dataset_candidate
    return path


def index_numbered_texts(root: Path, prefix: str) -> dict[int, Path]:
    direct = {int(path.stem.split("_")[1]): path for path in root.glob(f"{prefix}_*.txt") if "_" in path.stem}
    if direct:
        return direct
    nested = root / prefix
    if nested.exists():
        return {int(path.stem.split("_")[1]): path for path in nested.glob(f"{prefix}_*.txt") if "_" in path.stem}
    return {}


def load_pairs(model_key: str, dataset_root: Path, tokenizer, max_pairs: int) -> list[PairRecord]:
    pairs: list[PairRecord] = []
    if model_key == "mistral":
        clean_paths = index_numbered_texts(dataset_root, "clean")
        corrupt_paths = index_numbered_texts(dataset_root, "corrupt")
        for pair_id in sorted(set(clean_paths) & set(corrupt_paths)):
            clean_path = clean_paths[pair_id]
            corrupt_path = corrupt_paths[pair_id]
            clean_text = clean_path.read_text(encoding="utf-8")
            corrupt_text = corrupt_path.read_text(encoding="utf-8")
            pairs.append(
                PairRecord(
                    pair_id=pair_id,
                    sample_id=f"pair_{pair_id}",
                    clean_text=clean_text,
                    corrupt_text=corrupt_text,
                    clean_ids=encode_text(tokenizer, clean_text),
                    corrupt_ids=encode_text(tokenizer, corrupt_text),
                    clean_path=str(clean_path),
                    corrupt_path=str(corrupt_path),
                )
            )
            if max_pairs > 0 and len(pairs) >= max_pairs:
                break
    elif model_key == "devstral":
        rows = sorted(read_jsonl(dataset_root / "manifest.jsonl"), key=lambda row: int(row.get("pair_id", 0)))
        for row in rows:
            clean_path = resolve_prompt_path(row["clean_path"], dataset_root)
            corrupt_path = resolve_prompt_path(row["corrupt_path"], dataset_root)
            clean_text = clean_path.read_text(encoding="utf-8")
            corrupt_text = corrupt_path.read_text(encoding="utf-8")
            pairs.append(
                PairRecord(
                    pair_id=int(row.get("pair_id", len(pairs) + 1)),
                    sample_id=str(row.get("sample_id", f"pair_{len(pairs) + 1}")),
                    clean_text=clean_text,
                    corrupt_text=corrupt_text,
                    clean_ids=encode_text(tokenizer, clean_text),
                    corrupt_ids=encode_text(tokenizer, corrupt_text),
                    clean_path=str(clean_path),
                    corrupt_path=str(corrupt_path),
                )
            )
            if max_pairs > 0 and len(pairs) >= max_pairs:
                break
    elif model_key == "granite":
        rows = sorted(read_jsonl(dataset_root / "manifest.jsonl"), key=lambda row: int(row.get("dataset_index", 0)))
        for row in rows:
            clean_path = resolve_prompt_path(row["clean_prompt_path"], dataset_root)
            corrupt_path = resolve_prompt_path(row["corrupt_prompt_path"], dataset_root)
            clean_text = clean_path.read_text(encoding="utf-8")
            corrupt_text = corrupt_path.read_text(encoding="utf-8")
            pairs.append(
                PairRecord(
                    pair_id=int(row.get("dataset_index", len(pairs) + 1)),
                    sample_id=str(row.get("sample_id", f"pair_{len(pairs) + 1}")),
                    clean_text=clean_text,
                    corrupt_text=corrupt_text,
                    clean_ids=encode_text(tokenizer, clean_text),
                    corrupt_ids=encode_text(tokenizer, corrupt_text),
                    clean_path=str(clean_path),
                    corrupt_path=str(corrupt_path),
                )
            )
            if max_pairs > 0 and len(pairs) >= max_pairs:
                break
    elif model_key == "qwen35":
        rows = sorted(read_jsonl(dataset_root / "manifest.jsonl"), key=lambda row: int(row.get("pair_id", 0)))
        for row in rows:
            clean_path = dataset_root / str(row["clean_filename"])
            corrupt_path = dataset_root / str(row["corrupt_filename"])
            clean_text = clean_path.read_text(encoding="utf-8")
            corrupt_text = corrupt_path.read_text(encoding="utf-8")
            pairs.append(
                PairRecord(
                    pair_id=int(row.get("pair_id", len(pairs) + 1)),
                    sample_id=str(row.get("sample_id", f"pair_{len(pairs) + 1}")),
                    clean_text=clean_text,
                    corrupt_text=corrupt_text,
                    clean_ids=encode_text(tokenizer, clean_text),
                    corrupt_ids=encode_text(tokenizer, corrupt_text),
                    clean_path=str(clean_path),
                    corrupt_path=str(corrupt_path),
                )
            )
            if max_pairs > 0 and len(pairs) >= max_pairs:
                break
    else:
        raise ValueError(f"Unsupported model key: {model_key}")
    if not pairs:
        raise RuntimeError(f"No pairs loaded from {dataset_root}")
    return pairs


def load_candidate_heads(path: Path, top_heads: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if "layer" not in row or "head" not in row:
                continue
            kind = str(row.get("kind", "")).strip()
            if kind and kind != "full_attention":
                continue
            score = row.get("head_delta") or row.get("abs_head_delta") or row.get("dla_delta") or "0"
            rows.append(
                {
                    "layer": int(row["layer"]),
                    "head": int(row["head"]),
                    "score": float(score),
                    "kind": kind or "attention",
                    "source_row": dict(row),
                }
            )
    rows.sort(key=lambda item: abs(float(item["score"])), reverse=True)
    return rows[: max(int(top_heads), 1)]


def locate_region_span(model_key: str, text: str) -> tuple[int, int, str]:
    if "[AVAILABLE_TOOLS]" in text and "[/AVAILABLE_TOOLS]" in text:
        start = text.index("[AVAILABLE_TOOLS]")
        end = text.index("[/AVAILABLE_TOOLS]", start) + len("[/AVAILABLE_TOOLS]")
        return start, end, "available_tools"
    if "<|start_of_role|>available_tools<|end_of_role|>" in text:
        start = text.index("<|start_of_role|>available_tools<|end_of_role|>")
        marker = "<|end_of_text|>"
        end = text.find(marker, start)
        if end >= 0:
            return start, end + len(marker), "granite_available_tools"
    if "<|im_start|>system" in text:
        start = text.index("<|im_start|>system")
        end = text.find("<|im_end|>", start)
        if end >= 0:
            return start, end + len("<|im_end|>"), "qwen_system"
    if "<tools>" in text and "</tools>" in text:
        start = text.index("<tools>")
        end = text.index("</tools>", start) + len("</tools>")
        return start, end, "tools_xml"
    if '{"type"' in text and "function" in text:
        start = text.index('{"type"')
        end_candidates = [text.find("</tools>", start), text.find("[/AVAILABLE_TOOLS]", start), text.find("<|end_of_text|>", start)]
        end_candidates = [value for value in end_candidates if value >= 0]
        if end_candidates:
            return start, min(end_candidates), "json_schema"
    return 0, 0, "missing"


def token_positions_for_span(tokenizer, text: str, start: int, end: int) -> tuple[list[int], str]:
    if start < 0 or end <= start:
        return [], "missing"
    try:
        encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        offsets = encoded["offset_mapping"]
        positions = [
            idx
            for idx, (tok_start, tok_end) in enumerate(offsets)
            if int(tok_start) < int(end) and int(tok_end) > int(start)
        ]
        return positions, "offset_mapping"
    except Exception:
        prefix_ids = tokenizer(text[:start], add_special_tokens=False)["input_ids"]
        span_ids = tokenizer(text[start:end], add_special_tokens=False)["input_ids"]
        start_idx = len(prefix_ids)
        return list(range(start_idx, start_idx + len(span_ids))), "prefix_token_count"


def collate(tokenizer, ids_list: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, list[int], list[int]]:
    masks = [torch.ones_like(ids, dtype=torch.long) for ids in ids_list]
    batch = tokenizer.pad(
        {"input_ids": ids_list, "attention_mask": masks},
        padding=True,
        return_tensors="pt",
    )
    max_len = int(batch["input_ids"].shape[1])
    lengths = [int(ids.numel()) for ids in ids_list]
    if tokenizer.padding_side == "left":
        offsets = [max_len - length for length in lengths]
        query_positions = [max_len - 1 for _length in lengths]
    else:
        offsets = [0 for _length in lengths]
        query_positions = [length - 1 for length in lengths]
    return batch["input_ids"], batch["attention_mask"], offsets, query_positions


def resolve_head_idx(requested_head: int, pattern_head_count: int, num_attention_heads: int | None) -> int:
    if requested_head < pattern_head_count:
        return requested_head
    if num_attention_heads and num_attention_heads % pattern_head_count == 0:
        return requested_head // (num_attention_heads // pattern_head_count)
    raise ValueError(f"Cannot map requested head {requested_head} into pattern head count {pattern_head_count}")


def attention_tuple_index(model, layer: int) -> int:
    config = getattr(model, "config", None)
    layer_types = getattr(config, "layer_types", None)
    if layer_types is None and getattr(config, "text_config", None) is not None:
        layer_types = getattr(config.text_config, "layer_types", None)
    if not layer_types:
        return int(layer)
    if int(layer) >= len(layer_types):
        raise IndexError(f"Layer {layer} outside layer_types length {len(layer_types)}")
    if layer_types[int(layer)] != "full_attention":
        raise ValueError(f"Layer {layer} is {layer_types[int(layer)]}, not full_attention.")
    return sum(1 for idx in range(int(layer) + 1) if layer_types[idx] == "full_attention") - 1


def compute_side_masses(
    model,
    tokenizer,
    examples: list[PairRecord],
    *,
    side: str,
    candidate_heads: list[dict[str, Any]],
) -> tuple[dict[tuple[int, int, str], float], list[dict[str, Any]]]:
    ids_list = [example.clean_ids if side == "clean" else example.corrupt_ids for example in examples]
    texts = [example.clean_text if side == "clean" else example.corrupt_text for example in examples]
    input_ids, attention_mask, offsets, query_positions = collate(tokenizer, ids_list)
    device = next(model.parameters()).device
    with torch.inference_mode():
        outputs = model(
            input_ids=input_ids.to(device),
            attention_mask=attention_mask.to(device),
            output_attentions=True,
            use_cache=False,
            return_dict=True,
        )
    attentions = outputs.attentions
    if attentions is None:
        raise RuntimeError("Model did not return attentions; ensure eager attention and output_attentions=True.")

    config = getattr(model, "config", None)
    num_attention_heads = getattr(config, "num_attention_heads", None)
    masses: dict[tuple[int, int, str], float] = {}
    span_rows: list[dict[str, Any]] = []
    for batch_idx, (example, text) in enumerate(zip(examples, texts)):
        start, end, region_kind = locate_region_span("", text)
        raw_positions, position_method = token_positions_for_span(tokenizer, text, start, end)
        padded_positions = [pos + offsets[batch_idx] for pos in raw_positions if pos >= 0]
        valid_positions = [pos for pos in padded_positions if 0 <= pos < int(input_ids.shape[1])]
        span_rows.append(
            {
                "sample_id": example.sample_id,
                "pair_id": example.pair_id,
                "side": side,
                "region_kind": region_kind,
                "region_char_start": int(start),
                "region_char_end": int(end),
                "region_token_start": int(min(raw_positions)) if raw_positions else -1,
                "region_token_end_exclusive": int(max(raw_positions) + 1) if raw_positions else -1,
                "n_region_tokens": len(raw_positions),
                "position_method": position_method,
                "sequence_length": int(ids_list[batch_idx].numel()),
                "padded_sequence_length": int(input_ids.shape[1]),
                "pad_offset": int(offsets[batch_idx]),
                "query_position": int(query_positions[batch_idx]),
            }
        )
        for item in candidate_heads:
            layer = int(item["layer"])
            head = int(item["head"])
            pattern = attentions[attention_tuple_index(model, layer)].detach()
            effective_head = resolve_head_idx(head, int(pattern.shape[1]), num_attention_heads)
            if valid_positions:
                mass = float(
                    pattern[
                        batch_idx,
                        effective_head,
                        int(query_positions[batch_idx]),
                        torch.tensor(valid_positions, device=pattern.device, dtype=torch.long),
                    ]
                    .float()
                    .sum()
                    .item()
                )
            else:
                mass = 0.0
            masses[(layer, head, example.sample_id)] = mass
    del outputs, attentions, input_ids, attention_mask
    clear_cuda()
    return masses, span_rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    cfg = dict(MODEL_CONFIGS[args.model_key])
    model_path = args.model_path or cfg["model_path"]
    dataset_root = args.dataset_root
    candidate_csv = args.candidate_csv
    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    wait_for_vram(args.min_free_vram_gib, args.wait_poll_seconds)

    dtype = dtype_from_name(args.dtype)
    candidate_heads = load_candidate_heads(candidate_csv, args.top_heads)
    write_csv(output_root / "candidate_heads.csv", candidate_heads)

    model, tokenizer = load_model_and_tokenizer(args.model_key, model_path, dtype, args.attn_implementation)
    pairs = load_pairs(args.model_key, dataset_root, tokenizer, args.max_pairs)

    region_rows: list[dict[str, Any]] = []
    span_rows_all: list[dict[str, Any]] = []
    head_acc = {
        (int(item["layer"]), int(item["head"])): {"clean": 0.0, "corrupt": 0.0, "n": 0}
        for item in candidate_heads
    }
    step = max(int(args.batch_size), 1)
    for start in tqdm(range(0, len(pairs), step), desc=f"{cfg['display_model']} attention", dynamic_ncols=True):
        chunk = pairs[start : start + step]
        clean_masses, clean_spans = compute_side_masses(model, tokenizer, chunk, side="clean", candidate_heads=candidate_heads)
        corrupt_masses, corrupt_spans = compute_side_masses(
            model,
            tokenizer,
            chunk,
            side="corrupt",
            candidate_heads=candidate_heads,
        )
        span_rows_all.extend(clean_spans)
        span_rows_all.extend(corrupt_spans)
        span_by_key = {(row["sample_id"], row["side"]): row for row in clean_spans + corrupt_spans}
        for example in chunk:
            for item in candidate_heads:
                layer = int(item["layer"])
                head = int(item["head"])
                clean_mass = float(clean_masses[(layer, head, example.sample_id)])
                corrupt_mass = float(corrupt_masses[(layer, head, example.sample_id)])
                delta = clean_mass - corrupt_mass
                acc = head_acc[(layer, head)]
                acc["clean"] += clean_mass
                acc["corrupt"] += corrupt_mass
                acc["n"] += 1
                clean_span = span_by_key[(example.sample_id, "clean")]
                corrupt_span = span_by_key[(example.sample_id, "corrupt")]
                region_rows.append(
                    {
                        "sample_id": example.sample_id,
                        "pair_id": example.pair_id,
                        "layer": layer,
                        "head": head,
                        "clean_region_attention": clean_mass,
                        "corrupt_region_attention": corrupt_mass,
                        "delta_attention": delta,
                        "clean_region_char_start": clean_span["region_char_start"],
                        "clean_region_char_end": clean_span["region_char_end"],
                        "clean_region_token_start": clean_span["region_token_start"],
                        "clean_region_token_end_exclusive": clean_span["region_token_end_exclusive"],
                        "clean_region_kind": clean_span["region_kind"],
                        "corrupt_region_char_start": corrupt_span["region_char_start"],
                        "corrupt_region_char_end": corrupt_span["region_char_end"],
                        "corrupt_region_token_start": corrupt_span["region_token_start"],
                        "corrupt_region_token_end_exclusive": corrupt_span["region_token_end_exclusive"],
                        "corrupt_region_kind": corrupt_span["region_kind"],
                    }
                )

    head_summaries: list[dict[str, Any]] = []
    for item in candidate_heads:
        layer = int(item["layer"])
        head = int(item["head"])
        acc = head_acc[(layer, head)]
        n = max(int(acc["n"]), 1)
        clean_mean = float(acc["clean"] / n)
        corrupt_mean = float(acc["corrupt"] / n)
        head_summaries.append(
            {
                "layer": layer,
                "head": head,
                "n_pairs": int(acc["n"]),
                "clean_region_attention": clean_mean,
                "corrupt_region_attention": corrupt_mean,
                "delta_attention": clean_mean - corrupt_mean,
                "max_attn_pp_candidate": 100.0 * (clean_mean - corrupt_mean),
                "candidate_score": float(item["score"]),
            }
        )
    head_summaries.sort(key=lambda row: float(row["delta_attention"]), reverse=True)
    max_row = head_summaries[0] if head_summaries else None
    summary = {
        "model_key": args.model_key,
        "display_model": cfg["display_model"],
        "model_path": str(model_path),
        "dataset_root": str(dataset_root),
        "candidate_csv": str(candidate_csv),
        "n_pairs": len(pairs),
        "batch_size": int(args.batch_size),
        "dtype": args.dtype,
        "attn_implementation": args.attn_implementation,
        "candidate_heads": candidate_heads,
        "head_summaries": head_summaries,
        "max_attn_pp": float(max_row["max_attn_pp_candidate"]) if max_row else None,
        "max_head": max_row,
        "definition": "100 * max over candidate heads of mean(clean region attention - corrupt region attention)",
        "region_definition": "tool/system/schema region located per prompt by chat-template markers.",
    }
    write_csv(output_root / "region_attention.csv", region_rows)
    write_csv(output_root / "region_spans.csv", span_rows_all)
    write_json = lambda path, payload: path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_json(output_root / "summary.json", summary)
    lines = [
        f"# {cfg['display_model']} Section 6 Readout Attention",
        "",
        f"- Pairs: `{len(pairs)}`",
        f"- Candidate source: `{candidate_csv}`",
        f"- Max Attn (pp): `{float(summary['max_attn_pp']):.4f}`" if summary["max_attn_pp"] is not None else "- Max Attn (pp): `n/a`",
    ]
    if max_row:
        lines.append(
            f"- Max head: `L{int(max_row['layer'])}H{int(max_row['head'])}` with clean `{float(max_row['clean_region_attention']):.6f}`, corrupt `{float(max_row['corrupt_region_attention']):.6f}`."
        )
    lines.append("- This is attention region mass, not head projection or DLA.")
    (output_root / "summary.md").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    del model
    clear_cuda()


if __name__ == "__main__":
    main()
