#!/usr/bin/env python3
"""Run frozen-Coding cross-domain transfer for one Qwen3.5 checkpoint.

The runner consumes a tokenizer-audited, read-only v4 pair view.  It freezes
the model-specific Coding layer from an earlier Coding-only bundle, estimates
the D1 Coding and D3/D4/D5 native mean-difference vectors on their 400-pair
training splits, then evaluates D1->target transfer on 100 held-out pairs.

For each target it reports the frozen Coding direction, rescaled to the
target native L2 norm, and a single seeded Gaussian random direction with
the identical target norm.  No target-domain examples affect layer selection.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import gc
import importlib.util
import json
import math
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

import torch
from tqdm.auto import tqdm


# Match the narrow workaround used by the established Qwen3.5 causal runner.
_ORIGINAL_FIND_SPEC = importlib.util.find_spec


def _patched_find_spec(name: str, package: str | None = None):
    if name == "sklearn" and os.environ.get("MECH_ENABLE_TRANSFORMERS_SKLEARN", "0") != "1":
        return None
    return _ORIGINAL_FIND_SPEC(name, package)


importlib.util.find_spec = _patched_find_spec
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
finally:
    importlib.util.find_spec = _ORIGINAL_FIND_SPEC


DOMAINS = ("D1", "D3", "D4", "D5")
TARGET_DOMAINS = ("D3", "D4", "D5")
TABLE_DOMAIN = {"D3": "Retrieval", "D4": "SQL", "D5": "Email"}
TOOL_CALL_TEXT = "<tool_call>"


@dataclass(frozen=True)
class Pair:
    sample_id: str
    candidate_id: str
    clean_ids: list[int]
    corrupt_ids: list[int]

    @property
    def token_len(self) -> int:
        return len(self.clean_ids)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--dataset-view-root", type=Path, required=True)
    parser.add_argument("--frozen-coding-bundle", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--targets", nargs="+", choices=TARGET_DOMAINS, default=list(TARGET_DOMAINS))
    parser.add_argument("--train-pairs", type=int, default=400)
    parser.add_argument("--eval-pairs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--random-seed", type=int, default=20260727)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if text:
                rows.append(json.loads(text))
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def ids_from_text(tokenizer: Any, text: str) -> list[int]:
    values = tokenizer(text, add_special_tokens=False)["input_ids"]
    if isinstance(values, torch.Tensor):
        values = values.detach().cpu().tolist()
    if values and isinstance(values[0], list):
        values = values[0]
    return [int(value) for value in values]


def load_pairs(tokenizer: Any, *, view_root: Path, domain: str, split: str, expected_count: int) -> tuple[list[Pair], dict[str, Any]]:
    clean_root = view_root / domain / split / "clean"
    corrupt_root = view_root / domain / split / "corrupt"
    clean_manifest = read_jsonl(clean_root / "manifest.jsonl")
    corrupt_manifest = {str(row["output_filename"]): row for row in read_jsonl(corrupt_root / "manifest.jsonl")}
    pairs: list[Pair] = []
    differing_positions: set[int] = set()
    for row in sorted(clean_manifest, key=lambda item: str(item["output_filename"])):
        filename = str(row["output_filename"])
        if filename not in corrupt_manifest:
            raise ValueError(f"{domain}/{split}: no corrupt manifest row for {filename}")
        clean_path = clean_root / filename
        corrupt_path = corrupt_root / filename
        clean_ids = ids_from_text(tokenizer, clean_path.read_text(encoding="utf-8"))
        corrupt_ids = ids_from_text(tokenizer, corrupt_path.read_text(encoding="utf-8"))
        if len(clean_ids) != len(corrupt_ids):
            raise ValueError(f"{domain}/{split}/{filename}: unequal token lengths")
        differences = [index for index, (left, right) in enumerate(zip(clean_ids, corrupt_ids)) if left != right]
        if len(differences) != 1:
            raise ValueError(f"{domain}/{split}/{filename}: expected one differing token, got {len(differences)}")
        differing_positions.add(differences[0])
        pairs.append(
            Pair(
                sample_id=str(row.get("sample_id") or Path(filename).stem),
                candidate_id=str(row.get("candidate_id") or Path(filename).stem),
                clean_ids=clean_ids,
                corrupt_ids=corrupt_ids,
            )
        )
    if len(pairs) != int(expected_count):
        raise ValueError(f"{domain}/{split}: expected {expected_count} pairs, got {len(pairs)}")
    return pairs, {
        "pair_count": len(pairs),
        "token_length_min": min(pair.token_len for pair in pairs),
        "token_length_max": max(pair.token_len for pair in pairs),
        "unique_differing_token_positions": sorted(differing_positions),
    }


def batches(pairs: Sequence[Pair], *, batch_size: int) -> list[list[Pair]]:
    if batch_size < 1:
        raise ValueError("batch size must be positive")
    ordered = sorted(pairs, key=lambda pair: (pair.token_len, pair.candidate_id))
    return [ordered[start : start + batch_size] for start in range(0, len(ordered), batch_size)]


def make_inputs(rows: Sequence[Pair], *, side: str, pad_token_id: int, device: torch.device) -> dict[str, torch.Tensor]:
    selected = [pair.clean_ids if side == "clean" else pair.corrupt_ids for pair in rows]
    max_len = max(len(value) for value in selected)
    input_ids = torch.full((len(rows), max_len), int(pad_token_id), dtype=torch.long)
    attention_mask = torch.zeros((len(rows), max_len), dtype=torch.long)
    for index, values in enumerate(selected):
        input_ids[index, -len(values) :] = torch.tensor(values, dtype=torch.long)
        attention_mask[index, -len(values) :] = 1
    return {"input_ids": input_ids.to(device), "attention_mask": attention_mask.to(device)}


def resolve_layers(model: Any) -> Any:
    for path in ("model.language_model", "model", "base_model.model", "base_model", "language_model", "transformer"):
        candidate = model
        try:
            for component in path.split("."):
                candidate = getattr(candidate, component)
        except AttributeError:
            continue
        if hasattr(candidate, "layers"):
            return candidate.layers
        if hasattr(candidate, "model") and hasattr(candidate.model, "layers"):
            return candidate.model.layers
    raise RuntimeError("Could not resolve Qwen3.5 decoder layers")


def forward_last_logits(model: Any, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
    with torch.inference_mode():
        try:
            output = model(**inputs, use_cache=False, return_dict=True, logits_to_keep=1)
        except TypeError:
            output = model(**inputs, use_cache=False, return_dict=True)
    logits = output.logits
    if logits.ndim != 3:
        raise RuntimeError(f"Unexpected logits shape {tuple(logits.shape)}")
    return logits[:, -1, :].float()


@contextlib.contextmanager
def capture_last_state(model: Any, *, layer: int) -> Iterator[dict[str, torch.Tensor]]:
    layers = resolve_layers(model)
    if layer < 0 or layer >= len(layers):
        raise ValueError(f"L{layer} is invalid for {len(layers)} layers")
    captured: dict[str, torch.Tensor] = {}

    def hook(_module: Any, _inputs: Any, output: Any) -> None:
        hidden = output[0] if isinstance(output, tuple) else output
        if not isinstance(hidden, torch.Tensor) or hidden.ndim != 3:
            raise RuntimeError(f"Unexpected layer output at L{layer}: {type(hidden)!r}")
        captured["state"] = hidden[:, -1, :].detach().float().cpu()

    handle = layers[layer].register_forward_hook(hook)
    try:
        yield captured
    finally:
        handle.remove()


@contextlib.contextmanager
def temporary_last_token_addition(model: Any, *, layer: int, vector_cpu: torch.Tensor) -> Iterator[dict[str, int]]:
    layers = resolve_layers(model)
    if layer < 0 or layer >= len(layers):
        raise ValueError(f"L{layer} is invalid for {len(layers)} layers")
    stats = {"hook_calls": 0, "modified_calls": 0}

    def hook(_module: Any, _inputs: Any, output: Any) -> Any:
        stats["hook_calls"] += 1
        hidden = output[0] if isinstance(output, tuple) else output
        if not isinstance(hidden, torch.Tensor) or hidden.ndim != 3:
            raise RuntimeError(f"Unexpected layer output at L{layer}: {type(hidden)!r}")
        if hidden.shape[-1] != vector_cpu.numel():
            raise ValueError(f"L{layer}: vector dim {vector_cpu.numel()} != hidden dim {hidden.shape[-1]}")
        patched = hidden.clone()
        patched[:, -1, :] += vector_cpu.to(device=patched.device, dtype=patched.dtype)
        stats["modified_calls"] += 1
        if isinstance(output, tuple):
            return (patched, *output[1:])
        return patched

    handle = layers[layer].register_forward_hook(hook)
    try:
        yield stats
    finally:
        handle.remove()


def mean_state(
    model: Any,
    pairs: Sequence[Pair],
    *,
    side: str,
    layer: int,
    tokenizer: Any,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    total: torch.Tensor | None = None
    count = 0
    for batch_index, batch in enumerate(batches(pairs, batch_size=batch_size), start=1):
        inputs = make_inputs(batch, side=side, pad_token_id=int(tokenizer.pad_token_id), device=device)
        with capture_last_state(model, layer=layer) as captured:
            logits = forward_last_logits(model, inputs)
        state = captured.get("state")
        if state is None:
            raise RuntimeError(f"L{layer}: forward hook did not capture a state")
        reduced = state.sum(dim=0)
        total = reduced if total is None else total + reduced
        count += len(batch)
        del inputs, logits, state
        torch.cuda.empty_cache()
        if batch_index % 25 == 0 or batch_index == math.ceil(len(pairs) / batch_size):
            print(json.dumps({"mean_state": side, "batches": batch_index}), flush=True)
    if total is None or count != len(pairs):
        raise RuntimeError(f"Incomplete mean state for {side}")
    return (total / count).float().contiguous()


def native_vector(
    model: Any,
    pairs: Sequence[Pair],
    *,
    layer: int,
    tokenizer: Any,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    vector = mean_state(
        model, pairs, side="clean", layer=layer, tokenizer=tokenizer, device=device, batch_size=batch_size
    ) - mean_state(
        model, pairs, side="corrupt", layer=layer, tokenizer=tokenizer, device=device, batch_size=batch_size
    )
    if not bool(torch.isfinite(vector).all()) or float(vector.norm().item()) <= 0.0:
        raise ValueError("Invalid native mean-difference vector")
    return vector.contiguous()


def tool_metrics(logits: torch.Tensor, *, tool_token_id: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return logits[:, tool_token_id], torch.softmax(logits, dim=-1)[:, tool_token_id], logits.argmax(dim=-1)


def seeded_unit_vector(dim: int, *, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    value = torch.randn((int(dim),), generator=generator, dtype=torch.float32)
    if float(value.norm().item()) <= 0.0:
        raise RuntimeError("Degenerate random direction")
    return (value / value.norm()).contiguous()


def evaluate_target(
    model: Any,
    pairs: Sequence[Pair],
    *,
    target_domain: str,
    code_vector: torch.Tensor,
    random_vector: torch.Tensor,
    source_norm: float,
    target_norm: float,
    layer: int,
    tokenizer: Any,
    device: torch.device,
    tool_token_id: int,
    batch_size: int,
    random_seed: int,
    random_cosine_to_code: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    treatments = {"Code": code_vector.contiguous(), "Random": random_vector.contiguous()}
    accum: dict[str, dict[str, float]] = {name: defaultdict(float) for name in treatments}
    clean_logit_sum = corrupt_logit_sum = 0.0
    clean_prob_sum = corrupt_prob_sum = 0.0
    clean_tool_count = corrupt_non_tool_count = 0
    clean_tool_top1 = corrupt_tool_top1 = 0
    count = 0
    hook_stats: dict[str, dict[str, int]] = {}

    for batch in tqdm(batches(pairs, batch_size=batch_size), desc=f"{target_domain}: Qwen3.5 transfer", dynamic_ncols=True):
        clean_inputs = make_inputs(batch, side="clean", pad_token_id=int(tokenizer.pad_token_id), device=device)
        corrupt_inputs = make_inputs(batch, side="corrupt", pad_token_id=int(tokenizer.pad_token_id), device=device)
        clean_logits = forward_last_logits(model, clean_inputs)
        corrupt_logits = forward_last_logits(model, corrupt_inputs)
        clean_logit, clean_prob, clean_top1_ids = tool_metrics(clean_logits, tool_token_id=tool_token_id)
        corrupt_logit, corrupt_prob, corrupt_top1_ids = tool_metrics(corrupt_logits, tool_token_id=tool_token_id)
        clean_is_tool = clean_top1_ids == tool_token_id
        corrupt_is_tool = corrupt_top1_ids == tool_token_id
        count += len(batch)
        clean_logit_sum += float(clean_logit.sum().item())
        corrupt_logit_sum += float(corrupt_logit.sum().item())
        clean_prob_sum += float(clean_prob.sum().item())
        corrupt_prob_sum += float(corrupt_prob.sum().item())
        clean_tool_count += int(clean_is_tool.sum().item())
        corrupt_non_tool_count += int((~corrupt_is_tool).sum().item())
        clean_tool_top1 += int(clean_is_tool.sum().item())
        corrupt_tool_top1 += int(corrupt_is_tool.sum().item())

        for direction, vector in treatments.items():
            with temporary_last_token_addition(model, layer=layer, vector_cpu=vector) as add_stats:
                add_logits = forward_last_logits(model, corrupt_inputs)
            with temporary_last_token_addition(model, layer=layer, vector_cpu=-vector) as remove_stats:
                remove_logits = forward_last_logits(model, clean_inputs)
            prior_add = hook_stats.setdefault(f"{direction}/add", {"hook_calls": 0, "modified_calls": 0})
            prior_remove = hook_stats.setdefault(f"{direction}/remove", {"hook_calls": 0, "modified_calls": 0})
            for key in prior_add:
                prior_add[key] += int(add_stats[key])
                prior_remove[key] += int(remove_stats[key])
            add_logit, add_prob, add_top1_ids = tool_metrics(add_logits, tool_token_id=tool_token_id)
            remove_logit, remove_prob, remove_top1_ids = tool_metrics(remove_logits, tool_token_id=tool_token_id)
            bucket = accum[direction]
            bucket["add_tool_top1"] += float((add_top1_ids == tool_token_id).sum().item())
            bucket["add_strict_flip"] += float(((~corrupt_is_tool) & (add_top1_ids == tool_token_id)).sum().item())
            bucket["add_logit_sum"] += float(add_logit.sum().item())
            bucket["add_prob_sum"] += float(add_prob.sum().item())
            bucket["remove_tool_top1"] += float((remove_top1_ids == tool_token_id).sum().item())
            bucket["remove_strict_drop"] += float((clean_is_tool & (remove_top1_ids != tool_token_id)).sum().item())
            bucket["remove_logit_sum"] += float(remove_logit.sum().item())
            bucket["remove_prob_sum"] += float(remove_prob.sum().item())
            del add_logits, remove_logits, add_logit, add_prob, add_top1_ids, remove_logit, remove_prob, remove_top1_ids
            torch.cuda.empty_cache()

        del clean_inputs, corrupt_inputs, clean_logits, corrupt_logits
        del clean_logit, clean_prob, clean_top1_ids, corrupt_logit, corrupt_prob, corrupt_top1_ids, clean_is_tool, corrupt_is_tool
        torch.cuda.empty_cache()

    if count != len(pairs):
        raise AssertionError(f"{target_domain}: processed {count} instead of {len(pairs)}")
    if clean_tool_count == 0 or corrupt_non_tool_count == 0:
        raise ValueError(
            f"{target_domain}: cannot form strict rates; clean tool={clean_tool_count}, corrupt non-tool={corrupt_non_tool_count}"
        )
    baseline = {
        "target_domain": target_domain,
        "n": count,
        "clean_mean_tool_logit": clean_logit_sum / count,
        "corrupt_mean_tool_logit": corrupt_logit_sum / count,
        "clean_mean_tool_prob": clean_prob_sum / count,
        "corrupt_mean_tool_prob": corrupt_prob_sum / count,
        "clean_top1_rate": clean_tool_top1 / count,
        "corrupt_top1_rate": corrupt_tool_top1 / count,
        "clean_tool_count": clean_tool_count,
        "corrupt_non_tool_count": corrupt_non_tool_count,
    }
    logit_gap = float(baseline["clean_mean_tool_logit"] - baseline["corrupt_mean_tool_logit"])
    if logit_gap <= 0.0:
        raise ValueError(f"{target_domain}: non-positive clean/corrupt logit gap {logit_gap}")

    rows: list[dict[str, Any]] = []
    for direction, vector in treatments.items():
        bucket = accum[direction]
        add_logit = bucket["add_logit_sum"] / count
        remove_logit = bucket["remove_logit_sum"] / count
        effective_norm = float(vector.norm().item())
        rows.append(
            {
                "source_domain": "D1",
                "target_domain": target_domain,
                "table_domain": TABLE_DOMAIN[target_domain],
                "direction": direction,
                "condition": "target_norm_aligned" if direction == "Code" else "random_target_norm",
                "n": count,
                "source_l2_norm": source_norm if direction == "Code" else 1.0,
                "target_native_l2_norm": target_norm,
                "applied_scale_from_source": target_norm / source_norm if direction == "Code" else target_norm,
                "effective_l2_norm": effective_norm,
                "effective_norm_over_target_native": effective_norm / target_norm,
                "random_seed": int(random_seed) if direction == "Random" else None,
                "random_cosine_to_code": random_cosine_to_code if direction == "Random" else None,
                "baseline_clean_mean_tool_logit": baseline["clean_mean_tool_logit"],
                "baseline_corrupt_mean_tool_logit": baseline["corrupt_mean_tool_logit"],
                "baseline_clean_top1_rate": baseline["clean_top1_rate"],
                "baseline_corrupt_top1_rate": baseline["corrupt_top1_rate"],
                "baseline_clean_tool_count": clean_tool_count,
                "baseline_corrupt_non_tool_count": corrupt_non_tool_count,
                "add_tool_call_top1_rate": bucket["add_tool_top1"] / count,
                "add_strict_flip_count": int(bucket["add_strict_flip"]),
                "add_strict_flip_rate": bucket["add_strict_flip"] / corrupt_non_tool_count,
                "add_mean_tool_call_logit": add_logit,
                "add_mean_tool_call_prob": bucket["add_prob_sum"] / count,
                "sufficiency_normalized_logit_gap": (add_logit - baseline["corrupt_mean_tool_logit"]) / logit_gap,
                "remove_remaining_tool_call_top1_rate": bucket["remove_tool_top1"] / count,
                "remove_strict_drop_count": int(bucket["remove_strict_drop"]),
                "remove_strict_drop_rate": bucket["remove_strict_drop"] / clean_tool_count,
                "remove_mean_tool_call_logit": remove_logit,
                "remove_mean_tool_call_prob": bucket["remove_prob_sum"] / count,
                "necessity_normalized_logit_gap": (baseline["clean_mean_tool_logit"] - remove_logit) / logit_gap,
                "hook_add_calls": hook_stats[f"{direction}/add"]["hook_calls"],
                "hook_remove_calls": hook_stats[f"{direction}/remove"]["hook_calls"],
                "hook_add_modified": hook_stats[f"{direction}/add"]["modified_calls"],
                "hook_remove_modified": hook_stats[f"{direction}/remove"]["modified_calls"],
            }
        )
    return baseline, rows


def frozen_layer_bundle(path: Path) -> tuple[int, str, int]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a mapping in {path}")
    if "layer" not in payload or "mean_diff" not in payload:
        raise KeyError(f"{path}: missing layer or mean_diff")
    layer = int(payload["layer"])
    hook_kind = str(payload.get("hook_kind", "post"))
    dim = int(torch.as_tensor(payload["mean_diff"]).numel())
    if hook_kind != "post":
        raise ValueError(f"{path}: expected a post-output Qwen3.5 bundle, got {hook_kind}")
    return layer, hook_kind, dim


def load_model_and_tokenizer(model_path: Path) -> tuple[Any, Any, torch.device, int]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path), torch_dtype=torch.bfloat16, device_map={"": 0}, low_cpu_mem_usage=True, trust_remote_code=True
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    encoded = [int(value) for value in tokenizer.encode(TOOL_CALL_TEXT, add_special_tokens=False)]
    tool_token_id = int(tokenizer.convert_tokens_to_ids(TOOL_CALL_TEXT))
    if len(encoded) != 1 or encoded[0] != tool_token_id:
        raise RuntimeError(f"{TOOL_CALL_TEXT} must be a single Qwen3.5 token, got {encoded}/{tool_token_id}")
    return model, tokenizer, torch.device("cuda:0"), tool_token_id


def write_summary(path: Path, *, model_label: str, rows: Sequence[dict[str, Any]]) -> None:
    lines = [
        f"# {model_label}: frozen Coding cross-domain transfer",
        "",
        "Each row is evaluated on the fixed 100-pair held-out split. Code is the D1 mean-difference direction at the previously frozen Coding layer, rescaled to the target native norm. Random is a seeded equal-norm Gaussian direction.",
        "",
        "| domain | direction | strict flip / strict drop | eligible corrupt / clean |",
        "|---|---|---:|---:|",
    ]
    for domain in TARGET_DOMAINS:
        for direction in ("Code", "Random"):
            row = next(item for item in rows if item["target_domain"] == domain and item["direction"] == direction)
            lines.append(
                f"| {row['table_domain']} | {direction} | {100 * float(row['add_strict_flip_rate']):.0f}/{100 * float(row['remove_strict_drop_rate']):.0f} | {row['baseline_corrupt_non_tool_count']}/{row['baseline_clean_tool_count']} |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    targets = tuple(dict.fromkeys(args.targets))
    output_root = args.output_root.resolve()
    view_root = args.dataset_view_root.resolve()
    bundle_path = args.frozen_coding_bundle.resolve()
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite existing output root: {output_root}")
    if not view_root.is_dir():
        raise FileNotFoundError(view_root)
    layer, hook_kind, bundle_dim = frozen_layer_bundle(bundle_path)
    output_root.mkdir(parents=True, exist_ok=False)

    model: Any | None = None
    try:
        model, tokenizer, device, tool_token_id = load_model_and_tokenizer(args.model_path.resolve())
        layers = resolve_layers(model)
        if layer < 0 or layer >= len(layers):
            raise ValueError(f"Frozen L{layer} is invalid for {len(layers)} decoder layers")

        d1_train, d1_train_validation = load_pairs(
            tokenizer, view_root=view_root, domain="D1", split="train", expected_count=args.train_pairs
        )
        target_train: dict[str, list[Pair]] = {}
        target_test: dict[str, list[Pair]] = {}
        validation: dict[str, Any] = {"D1/train": d1_train_validation}
        for domain in targets:
            train_pairs, train_validation = load_pairs(
                tokenizer, view_root=view_root, domain=domain, split="train", expected_count=args.train_pairs
            )
            test_pairs, test_validation = load_pairs(
                tokenizer, view_root=view_root, domain=domain, split="test", expected_count=args.eval_pairs
            )
            target_train[domain] = train_pairs
            target_test[domain] = test_pairs
            validation[f"{domain}/train"] = train_validation
            validation[f"{domain}/test"] = test_validation

        d1_vector = native_vector(
            model, d1_train, layer=layer, tokenizer=tokenizer, device=device, batch_size=args.batch_size
        )
        if d1_vector.numel() != bundle_dim:
            raise ValueError(f"D1 vector dim {d1_vector.numel()} != frozen bundle dim {bundle_dim}")
        d1_norm = float(d1_vector.norm().item())
        random_unit = seeded_unit_vector(d1_vector.numel(), seed=args.random_seed)
        code_unit = d1_vector / d1_vector.norm().clamp_min(1e-12)
        random_cosine = float(torch.dot(random_unit, code_unit).item())
        native_vectors = {"D1": d1_vector}
        for domain in targets:
            native_vectors[domain] = native_vector(
                model,
                target_train[domain],
                layer=layer,
                tokenizer=tokenizer,
                device=device,
                batch_size=args.batch_size,
            )
        torch.save(
            {
                "mean_diff": d1_vector,
                "layer": layer,
                "hook_kind": hook_kind,
                "domain": "D1",
                "train_pair_count": len(d1_train),
            },
            output_root / f"D1_L{layer}_post.pt",
        )
        for domain in targets:
            torch.save(
                {
                    "mean_diff": native_vectors[domain],
                    "layer": layer,
                    "hook_kind": hook_kind,
                    "domain": domain,
                    "train_pair_count": len(target_train[domain]),
                },
                output_root / f"{domain}_L{layer}_post.pt",
            )
        torch.save(
            {"random_unit": random_unit, "seed": args.random_seed, "cosine_to_code": random_cosine},
            output_root / "random_direction.pt",
        )
        write_json(
            output_root / "run_config.json",
            {
                "model_label": args.model_label,
                "model_path": str(args.model_path.resolve()),
                "dataset_view_root": str(view_root),
                "prompt_interface": "frozen v4 Qwen <|im_start|> prompt serialization",
                "frozen_coding_bundle": str(bundle_path),
                "frozen_coding_layer": layer,
                "hook_kind": hook_kind,
                "train_pairs": args.train_pairs,
                "eval_pairs": args.eval_pairs,
                "targets": list(targets),
                "random_control": {
                    "kind": "one seeded Gaussian unit vector per model",
                    "seed": args.random_seed,
                    "cosine_to_v4_d1_code": random_cosine,
                    "per_target_scaling": "target native-vector L2 norm",
                },
            },
        )

        baseline_rows: list[dict[str, Any]] = []
        result_rows: list[dict[str, Any]] = []
        for domain in targets:
            target_norm = float(native_vectors[domain].norm().item())
            baseline, rows = evaluate_target(
                model,
                target_test[domain],
                target_domain=domain,
                code_vector=code_unit * target_norm,
                random_vector=random_unit * target_norm,
                source_norm=d1_norm,
                target_norm=target_norm,
                layer=layer,
                tokenizer=tokenizer,
                device=device,
                tool_token_id=tool_token_id,
                batch_size=args.batch_size,
                random_seed=args.random_seed,
                random_cosine_to_code=random_cosine,
            )
            baseline_rows.append(baseline)
            result_rows.extend(rows)
            write_csv(output_root / "matrix_long.partial.csv", result_rows)
            write_csv(output_root / "target_baselines.partial.csv", baseline_rows)
            torch.cuda.empty_cache()

        expected_rows = len(targets) * 2
        if len(result_rows) != expected_rows:
            raise AssertionError(f"Expected {expected_rows} result rows, found {len(result_rows)}")
        write_csv(output_root / "matrix_long.csv", result_rows)
        write_csv(output_root / "target_baselines.csv", baseline_rows)
        write_json(output_root / "pair_validation.json", validation)
        write_json(
            output_root / "native_vectors.json",
            {
                domain: {
                    "l2_norm": float(vector.norm().item()),
                    "dimension": vector.numel(),
                    "train_pair_count": args.train_pairs,
                }
                for domain, vector in native_vectors.items()
            },
        )
        write_summary(output_root / "summary.md", model_label=args.model_label, rows=result_rows)
        write_json(
            output_root / "completion.json",
            {
                "status": "complete",
                "expected_rows": expected_rows,
                "observed_rows": len(result_rows),
                "tool_token_id": tool_token_id,
                "layer": layer,
                "hook_kind": hook_kind,
                "decoder_layer_count": len(layers),
                "model_hidden_size": d1_vector.numel(),
            },
        )
        print(json.dumps({"status": "complete", "output_root": str(output_root), "rows": len(result_rows)}))
    finally:
        if model is not None:
            del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
