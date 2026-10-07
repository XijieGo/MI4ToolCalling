#!/usr/bin/env python3
"""Apply each model's locked coding vector to multi-domain, verb-free, and tau2.

The direction is fit on the locked coding train split and is not re-estimated
on the transfer sets. Multi-domain keeps that direction and rescales it to the
target domain's clean-minus-corrupt residual norm (train pairs when the split
exists, otherwise the eval pairs). Verb-free baselines all 600 requests, keeps
up to 10 fresh tool-call top-1 rows per domain x pattern, and removes the
coding vector at alpha 1 and 1.5. Tau2 uses the raw coding vector at alpha 1:
subtract on the call arm, add on the text arm. The control is a same-norm
Gaussian orthogonal to the coding direction, seed 20260726.
"""

from __future__ import annotations

import argparse
import fcntl
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
VECTOR_RUN = REPO_ROOT / "experiments/cross_model/tool_call_vector/run.py"
def _resolve_tau2(name: str) -> Path:
    default = Path(__file__).resolve().parent / "templates" / name
    return Path(os.environ.get(f"MI4TC_TAU2_{name.upper()}_ROOT", str(default))).expanduser()

TAU2_RAW = {
    "telecom": _resolve_tau2("telecom"),
    "retail": _resolve_tau2("retail"),
}
RANDOM_SEED = 20260726
TRANSFER_DOMAINS = ("retrieval", "operations", "communication")
VERB_FREE_DOMAINS = ("D1", "D3", "D4", "D5")
VERB_FREE_PATTERNS = ("P1", "P2", "P3", "P4", "P5")
PROTOCOL = "transfer-locked-v1"
FAMILY = {
    "qwen3_4b": "qwen3",
    "qwen3_8b": "qwen3",
    "qwen3_14b": "qwen3",
    "qwen35_4b": "qwen35",
    "qwen35_9b": "qwen35",
    "granite_3p3_8b": "granite",
    "mistral_3p2_24b": "mistral",
}
# Qwen3-4B and Qwen3-14B read the same rendered Qwen3 multi-domain prompts as Qwen3-8B.
MULTI_DOMAIN_SOURCE = {
    "qwen3_4b": "qwen3_8b",
    "qwen3_8b": "qwen3_8b",
    "qwen3_14b": "qwen3_8b",
}


def load_vector_module() -> Any:
    spec = importlib.util.spec_from_file_location("locked_vector", VECTOR_RUN)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {VECTOR_RUN}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def limit_rows(rows: list[Any], count: int) -> list[Any]:
    return rows[:count] if count else rows


def safe_content(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def normalize_arguments(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        parsed = json.loads(value)
        if not isinstance(parsed, dict):
            raise ValueError("Tool arguments must decode to an object")
        return parsed
    raise TypeError(f"Unsupported tool arguments type: {type(value).__name__}")


def function_name(call: dict[str, Any]) -> str:
    function = call.get("function") or {}
    name = call.get("name") or function.get("name")
    if not name:
        raise ValueError(f"Tool call has no name: {call}")
    return str(name)


def function_arguments(call: dict[str, Any]) -> dict[str, Any]:
    function = call.get("function") or {}
    if "arguments" in call and call.get("arguments") is not None:
        return normalize_arguments(call.get("arguments"))
    return normalize_arguments(function.get("arguments"))


def adapt_qwen35_messages(source: list[dict[str, Any]]) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    for raw in source:
        role = str(raw["role"])
        converted: dict[str, Any] = {"role": role, "content": safe_content(raw.get("content"))}
        calls = raw.get("tool_calls") or []
        if role == "assistant" and calls:
            converted["tool_calls"] = [
                {
                    "type": "function",
                    "function": {
                        "name": function_name(call),
                        "arguments": function_arguments(call),
                    },
                }
                for call in calls
            ]
        messages.append(converted)
    return messages


def adapt_granite_messages(source: list[dict[str, Any]]) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    for raw in source:
        role = str(raw["role"])
        content = safe_content(raw.get("content"))
        calls = raw.get("tool_calls") or []
        if role == "assistant" and calls:
            encoded_calls = [
                {"name": function_name(call), "arguments": function_arguments(call)} for call in calls
            ]
            encoded = "<|tool_call|>" + json.dumps(encoded_calls, ensure_ascii=False, separators=(",", ":"))
            content = content + encoded if content else encoded
        messages.append({"role": role, "content": content})
    return messages


def adapt_mistral_messages(source: list[dict[str, Any]]) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    call_id_map: dict[str, str] = {}
    call_name_map: dict[str, str] = {}
    counter = 0
    for raw in source:
        role = str(raw["role"])
        converted: dict[str, Any] = {"role": role, "content": safe_content(raw.get("content"))}
        calls = raw.get("tool_calls") or []
        if role == "assistant" and calls:
            converted_calls: list[dict[str, Any]] = []
            for call in calls:
                counter += 1
                original_id = str(call.get("id") or f"source_call_{counter}")
                native_id = f"a{counter:08d}"
                call_id_map[original_id] = native_id
                call_name_map[original_id] = function_name(call)
                converted_calls.append(
                    {
                        "id": native_id,
                        "type": "function",
                        "function": {
                            "name": function_name(call),
                            "arguments": json.dumps(
                                function_arguments(call), ensure_ascii=False, separators=(",", ":")
                            ),
                        },
                    }
                )
            converted["tool_calls"] = converted_calls
        elif role == "tool":
            original_id = str(raw.get("tool_call_id") or raw.get("id") or "")
            if original_id not in call_id_map:
                raise ValueError(f"Unmatched Mistral tool result id {original_id!r}")
            converted["tool_call_id"] = call_id_map[original_id]
            converted["name"] = call_name_map[original_id]
        messages.append(converted)
    return messages


def ids_from_encoded(encoded: Any) -> list[int]:
    if torch.is_tensor(encoded):
        values = encoded[0].detach().cpu().tolist() if encoded.ndim == 2 else encoded.detach().cpu().tolist()
        return [int(token) for token in values]
    input_ids = encoded["input_ids"] if not isinstance(encoded, list) else encoded
    if torch.is_tensor(input_ids):
        values = input_ids[0].detach().cpu().tolist() if input_ids.ndim == 2 else input_ids.detach().cpu().tolist()
        return [int(token) for token in values]
    if input_ids and isinstance(input_ids[0], list):
        input_ids = input_ids[0]
    return [int(token) for token in input_ids]


def orthogonal_match(vector: torch.Tensor, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    noise = torch.randn(vector.shape, generator=generator, dtype=torch.float32)
    unit = vector / vector.norm().clamp_min(1e-12)
    noise = noise - torch.dot(noise, unit) * unit
    return noise / noise.norm().clamp_min(1e-12) * vector.norm()


def fraction(mask: torch.Tensor) -> float | None:
    if int(mask.numel()) == 0:
        return None
    return float(mask.float().mean().item())


def arm_metrics(before_top1: torch.Tensor, after_top1: torch.Tensor, before_logit: torch.Tensor, after_logit: torch.Tensor) -> dict[str, Any]:
    callers = before_top1.bool()
    quiet = ~callers
    dropped = callers & ~after_top1.bool()
    induced = quiet & after_top1.bool()
    return {
        "n": int(before_top1.numel()),
        "top1_before": fraction(callers),
        "top1_after": fraction(after_top1.bool()),
        "n_baseline_call": int(callers.sum().item()),
        "n_baseline_quiet": int(quiet.sum().item()),
        "suppression_among_calls": fraction(dropped[callers]) if int(callers.sum()) else None,
        "induction_among_quiet": fraction(induced[quiet]) if int(quiet.sum()) else None,
        "mean_logit_before": float(before_logit.float().mean().item()),
        "mean_logit_after": float(after_logit.float().mean().item()),
        "mean_logit_delta": float((after_logit - before_logit).float().mean().item()),
    }


def fit_coding_vector(module: Any, adapter: Any, spec: dict[str, Any], layer: int) -> torch.Tensor:
    dataset = spec["dataset"]
    if spec["layout"] == "text_dir":
        train_rows, _held_rows = module.load_text_dir(dataset, 0, 0)
        train_clean, train_corrupt = module.ids_from_text(train_rows, adapter.tokenizer)
    else:
        collection = module.load_model_native_pairs(dataset)
        train_clean, train_corrupt, _held_clean, _held_corrupt = module.ids_from_native(
            collection, adapter.tokenizer, 0, 0
        )
    sweep = module.LayerSweep(adapter, spec["marker_id"], [layer], spec["token_budget"], hook=spec["hook"])
    padding_diff = sweep.padding_check(train_clean)
    print(f"padding check max |Δlogit|={padding_diff:.4f}", flush=True)
    if padding_diff > 0.5:
        raise RuntimeError(f"Padded batch disagrees with unpadded forward by {padding_diff:.4f}")
    print(f"fit coding vector on {len(train_clean)} train pairs", flush=True)
    clean = sweep.capture(train_clean)
    corrupt = sweep.capture(train_corrupt)
    vector = (clean["states"][layer] - corrupt["states"][layer]).mean(dim=0).float()
    if not torch.isfinite(vector).all() or float(vector.norm()) == 0.0:
        raise RuntimeError("Coding vector is empty or non-finite")
    return vector


def pct(value: float | None) -> str:
    return "—" if value is None else f"{value:.1%}"


def sequence_from_prompt_file(module: Any, tokenizer: Any, path: Path) -> list[int]:
    if path.suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if "input_ids" not in payload:
            raise KeyError(f"{path} has no stored input_ids")
        return [int(token) for token in payload["input_ids"]]
    return module.token_ids(tokenizer, path.read_text(encoding="utf-8"))


def pair_ids(module: Any, tokenizer: Any, domain_dir: Path, row: dict[str, Any]) -> tuple[list[int], list[int]]:
    if "clean_prompt" in row and "corrupt_prompt" in row:
        return (
            module.token_ids(tokenizer, row["clean_prompt"]),
            module.token_ids(tokenizer, row["corrupt_prompt"]),
        )
    return (
        sequence_from_prompt_file(module, tokenizer, domain_dir / row["clean_relpath"]),
        sequence_from_prompt_file(module, tokenizer, domain_dir / row["corrupt_relpath"]),
    )


def domain_pair_rows(domain_dir: Path, max_count: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str]:
    selected = domain_dir / "selected_pairs.jsonl"
    if not selected.is_file():
        raise FileNotFoundError(selected)
    rows = read_jsonl(selected)
    eval_rows = [row for row in rows if row.get("split", "test") in {"test", "heldout"}]
    train_rows = [row for row in rows if row.get("split") == "train"]
    norm_source = "train" if train_rows else "eval"
    return limit_rows(train_rows or eval_rows, max_count), limit_rows(eval_rows, max_count), norm_source


def run_pair_arm(
    sweep: Any,
    layer: int,
    clean_ids: list[list[int]],
    corrupt_ids: list[list[int]],
    direction: torch.Tensor,
    random_direction: torch.Tensor,
) -> dict[str, Any]:
    clean = sweep.capture(clean_ids)
    corrupt = sweep.capture(corrupt_ids)
    gap = float((clean["tool_logit"] - corrupt["tool_logit"]).mean().item())

    def apply(side: dict[str, Any], sequences: list[list[int]], delta: torch.Tensor) -> dict[str, Any]:
        replacement = side["states"][layer] + delta
        scored = sweep.intervene(sequences, layer, replacement)
        metrics = arm_metrics(side["tool_top1"], scored["tool_top1"], side["tool_logit"], scored["tool_logit"])
        metrics["normalized_logit_shift"] = None if abs(gap) < 1e-6 else metrics["mean_logit_delta"] / gap
        return metrics

    return {
        "n_eval": len(clean_ids),
        "logit_gap": gap,
        "clean_top1": fraction(clean["tool_top1"]),
        "corrupt_top1": fraction(corrupt["tool_top1"]),
        "induction": apply(corrupt, corrupt_ids, direction),
        "suppression": apply(clean, clean_ids, -direction),
        "random_induction": apply(corrupt, corrupt_ids, random_direction),
        "random_suppression": apply(clean, clean_ids, -random_direction),
    }


def multi_domain(
    module: Any,
    adapter: Any,
    spec: dict[str, Any],
    layer: int,
    coding: torch.Tensor,
    max_count: int,
) -> dict[str, Any] | None:
    source_key = MULTI_DOMAIN_SOURCE.get(spec_name(spec), spec_name(spec))
    root = REPO_ROOT / "datasets" / source_key / "multi_domain"
    if not root.is_dir():
        print(f"multi_domain: absent for {source_key}", flush=True)
        return None
    sweep = module.LayerSweep(adapter, spec["marker_id"], [layer], spec["token_budget"], hook=spec["hook"])
    domains: dict[str, Any] = {}
    for name in TRANSFER_DOMAINS:
        domain_dir = root / name
        if not (domain_dir / "selected_pairs.jsonl").is_file():
            print(f"multi_domain: skip {name}", flush=True)
            continue
        norm_rows, eval_rows, norm_source = domain_pair_rows(domain_dir, max_count)
        print(f"multi_domain {name}: norm={norm_source} n_norm={len(norm_rows)} n_eval={len(eval_rows)}", flush=True)
        norm_clean = [pair_ids(module, adapter.tokenizer, domain_dir, row)[0] for row in norm_rows]
        norm_corrupt = [pair_ids(module, adapter.tokenizer, domain_dir, row)[1] for row in norm_rows]
        clean_state = sweep.capture(norm_clean)["states"][layer]
        corrupt_state = sweep.capture(norm_corrupt)["states"][layer]
        target = (clean_state - corrupt_state).mean(dim=0).float()
        scale = float(target.norm().item()) / float(coding.norm().clamp_min(1e-12).item())
        direction = coding * scale
        random_direction = orthogonal_match(direction, RANDOM_SEED)
        eval_clean = [pair_ids(module, adapter.tokenizer, domain_dir, row)[0] for row in eval_rows]
        eval_corrupt = [pair_ids(module, adapter.tokenizer, domain_dir, row)[1] for row in eval_rows]
        scored = run_pair_arm(sweep, layer, eval_clean, eval_corrupt, direction, random_direction)
        scored["norm_source"] = norm_source
        scored["n_norm"] = len(norm_rows)
        scored["scale"] = scale
        scored["aligned_norm"] = float(direction.norm().item())
        domains[name] = scored
        print(
            f"multi_domain {name}: ind {scored['induction']['top1_before']:.3f}->{scored['induction']['top1_after']:.3f} "
            f"sup {scored['suppression']['top1_before']:.3f}->{scored['suppression']['top1_after']:.3f}",
            flush=True,
        )
    return {"source": f"datasets/{source_key}/multi_domain", "domains": domains}


def spec_name(spec: dict[str, Any]) -> str:
    return spec["model_key"]


def verb_free_rows(model_key: str) -> tuple[list[dict[str, Any]], str]:
    root = REPO_ROOT / "datasets" / model_key / "verb_free"
    detailed = root / "implicit_intent.jsonl"
    requests = root / "requests.jsonl"
    if detailed.is_file():
        rows = read_jsonl(detailed)
        if rows and "input_ids" in rows[0]:
            return rows, "implicit_intent.input_ids"
    if requests.is_file():
        return read_jsonl(requests), "requests.prompt"
    if detailed.is_file():
        return read_jsonl(detailed), "implicit_intent.prompt"
    raise FileNotFoundError(root)


def sequence_from_row(module: Any, tokenizer: Any, row: dict[str, Any], source: str) -> list[int]:
    if source.endswith("input_ids"):
        return [int(token) for token in row["input_ids"]]
    return module.token_ids(tokenizer, row["prompt"])


def run_delta_arm(sweep: Any, layer: int, sequences: list[list[int]], baseline: dict[str, Any], delta: torch.Tensor) -> dict[str, Any]:
    scored = sweep.intervene(sequences, layer, baseline["states"][layer] + delta)
    return arm_metrics(baseline["tool_top1"], scored["tool_top1"], baseline["tool_logit"], scored["tool_logit"])


def select_removal_arm(rows: list[dict[str, Any]], baseline_top1: torch.Tensor) -> list[int]:
    selected: list[int] = []
    for domain in VERB_FREE_DOMAINS:
        for pattern in VERB_FREE_PATTERNS:
            kept = 0
            for index, row in enumerate(rows):
                if row.get("domain") != domain or row.get("pattern") != pattern or not bool(baseline_top1[index]):
                    continue
                selected.append(index)
                kept += 1
                if kept == 10:
                    break
    return selected


def slice_capture(captured: dict[str, Any], indices: list[int]) -> dict[str, Any]:
    picker = torch.tensor(indices, dtype=torch.long)
    return {
        "tool_logit": captured["tool_logit"].index_select(0, picker),
        "tool_top1": captured["tool_top1"].index_select(0, picker),
        "states": {layer: state.index_select(0, picker) for layer, state in captured["states"].items()},
    }


def verb_free(module: Any, adapter: Any, spec: dict[str, Any], layer: int, coding: torch.Tensor, max_count: int) -> dict[str, Any]:
    rows, source = verb_free_rows(spec_name(spec))
    rows = limit_rows(rows, max_count)
    sequences = [sequence_from_row(module, adapter.tokenizer, row, source) for row in rows]
    print(f"verb_free: baseline {len(sequences)} from {source}", flush=True)
    sweep = module.LayerSweep(adapter, spec["marker_id"], [layer], spec["token_budget"], hook=spec["hook"])
    baseline = sweep.capture(sequences)
    indices = select_removal_arm(rows, baseline["tool_top1"])
    if not indices:
        labels = sorted({(row.get("domain"), row.get("pattern")) for row in rows})
        raise RuntimeError(f"Verb-free removal arm is empty. Cells in file: {labels[:12]}")
    arm_sequences = [sequences[index] for index in indices]
    arm_baseline = slice_capture(baseline, indices)
    random_direction = orthogonal_match(coding, RANDOM_SEED)
    result = {
        "source": source,
        "arm_rule": "up to 10 fresh baseline tool-call top-1 rows per domain x pattern, source order",
        "baseline_calls_all": int(baseline["tool_top1"].sum().item()),
        "arm_n": len(indices),
        "alpha_1": run_delta_arm(sweep, layer, arm_sequences, arm_baseline, -coding),
        "alpha_1_5": run_delta_arm(sweep, layer, arm_sequences, arm_baseline, -1.5 * coding),
        "random_alpha_1": run_delta_arm(sweep, layer, arm_sequences, arm_baseline, -random_direction),
        "random_alpha_1_5": run_delta_arm(sweep, layer, arm_sequences, arm_baseline, -1.5 * random_direction),
    }
    print(
        f"verb_free: baseline_calls={result['baseline_calls_all']} arm={result['arm_n']} "
        f"drop@1={result['alpha_1']['suppression_among_calls']} "
        f"drop@1.5={result['alpha_1_5']['suppression_among_calls']} "
        f"random@1.5={result['random_alpha_1_5']['suppression_among_calls']}",
        flush=True,
    )
    return result


def render_tau2(tokenizer: Any, rows: list[dict[str, Any]], domain: str, family: str) -> list[list[int]]:
    raw = TAU2_RAW[domain]
    system_prompt = (raw / "tau2_system_prompt.txt").read_text(encoding="utf-8").strip()
    tools = json.loads((raw / "tau2_tool_schemas.json").read_text(encoding="utf-8"))
    if not system_prompt or not isinstance(tools, list) or not tools:
        raise ValueError(f"Invalid tau2 template in {raw}")
    rendered: list[list[int]] = []
    for row in rows:
        source = row["messages"]
        if family == "qwen35":
            messages = [{"role": "system", "content": system_prompt}, *adapt_qwen35_messages(source)]
            encoded = tokenizer.apply_chat_template(
                messages,
                tools=tools,
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=False,
                return_dict=True,
            )
            ids = ids_from_encoded(encoded)
        elif family == "granite":
            messages = [{"role": "system", "content": system_prompt}, *adapt_granite_messages(source)]
            try:
                text = tokenizer.apply_chat_template(
                    messages, tools=tools, tokenize=False, add_generation_prompt=True, thinking=False
                )
            except TypeError:
                text = tokenizer.apply_chat_template(messages, tools=tools, tokenize=False, add_generation_prompt=True)
            if not isinstance(text, str):
                raise TypeError(f"Granite chat template returned {type(text).__name__}")
            ids = [int(token) for token in tokenizer(text, add_special_tokens=False)["input_ids"]]
        elif family == "mistral":
            messages = [{"role": "system", "content": system_prompt}, *adapt_mistral_messages(source)]
            encoded = tokenizer.apply_chat_template(
                messages,
                tools=tools,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
            )
            ids = ids_from_encoded(encoded)
        elif family == "qwen3":
            messages = [{"role": "system", "content": system_prompt}, *source]
            kwargs = {"tools": tools, "tokenize": False, "add_generation_prompt": True}
            try:
                text = tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
            except TypeError:
                text = tokenizer.apply_chat_template(messages, **kwargs)
            if not isinstance(text, str):
                raise TypeError(f"Qwen3 chat template returned {type(text).__name__}")
            ids = [int(token) for token in tokenizer(text, add_special_tokens=False)["input_ids"]]
        else:
            raise ValueError(f"Unsupported tau2 family {family!r}")
        if not ids:
            raise RuntimeError(f"Empty tau2 render for {row.get('candidate_id')}")
        rendered.append(ids)
    return rendered


def length_stats(sequences: list[list[int]]) -> str:
    lengths = sorted(len(sequence) for sequence in sequences)
    return f"min={lengths[0]} med={lengths[len(lengths) // 2]} max={lengths[-1]}"


def tau2_domain(rows: list[dict[str, Any]]) -> str:
    sample = rows[0]
    domain = sample.get("source_domain") or str(sample.get("candidate_id", "")).split("_")[0]
    if domain not in TAU2_RAW:
        raise ValueError(f"No tau2 template for domain {domain!r}")
    return domain


def tau2(module: Any, adapter: Any, spec: dict[str, Any], layer: int, coding: torch.Tensor, max_count: int) -> dict[str, Any]:
    root = REPO_ROOT / "datasets" / spec_name(spec) / "tau2_bench"
    call_rows = limit_rows(read_jsonl(root / "native-call-200.jsonl"), max_count)
    text_rows = limit_rows(read_jsonl(root / "native-text-200.jsonl"), max_count)
    domain = tau2_domain(call_rows)
    if tau2_domain(text_rows) != domain:
        raise ValueError("Call and text tau2 arms use different domains")
    family = FAMILY[spec_name(spec)]
    print(f"tau2: render {domain} family={family} call={len(call_rows)} text={len(text_rows)}", flush=True)
    call_ids = render_tau2(adapter.tokenizer, call_rows, domain, family)
    text_ids_rows = render_tau2(adapter.tokenizer, text_rows, domain, family)
    print(f"tau2 tokens call {length_stats(call_ids)} text {length_stats(text_ids_rows)}", flush=True)
    sweep = module.LayerSweep(
        adapter,
        spec["marker_id"],
        [layer],
        spec["token_budget"],
        hook=spec["hook"],
        last_token_only=True,
    )
    call_base = sweep.capture(call_ids)
    text_base = sweep.capture(text_ids_rows)
    random_direction = orthogonal_match(coding, RANDOM_SEED)

    def score_alpha(alpha: float) -> dict[str, Any]:
        removal_arm = run_delta_arm(sweep, layer, call_ids, call_base, -alpha * coding)
        induction_arm = run_delta_arm(sweep, layer, text_ids_rows, text_base, alpha * coding)
        paired = None
        if removal_arm["suppression_among_calls"] is not None and induction_arm["induction_among_quiet"] is not None:
            paired = 0.5 * (removal_arm["suppression_among_calls"] + induction_arm["induction_among_quiet"])
        print(
            f"tau2 alpha {alpha:g}: sup {removal_arm['suppression_among_calls']} ind {induction_arm['induction_among_quiet']}",
            flush=True,
        )
        return {"removal": removal_arm, "induction": induction_arm, "score": paired}

    strength = {str(float(alpha)): score_alpha(alpha) for alpha in (1.0, 1.5, 3.0)}
    alpha_one = strength["1.0"]
    result = {
        "domain": domain,
        "family": family,
        "template": str(TAU2_RAW[domain]),
        "alpha": 1.0,
        "forward": "left_pad_last_token",
        "removal": alpha_one["removal"],
        "induction": alpha_one["induction"],
        "score": alpha_one["score"],
        "random_removal": run_delta_arm(sweep, layer, call_ids, call_base, -random_direction),
        "random_induction": run_delta_arm(sweep, layer, text_ids_rows, text_base, random_direction),
        "strength": {key: strength[key] for key in ("1.5", "3.0")},
    }
    alpha_path = REPO_ROOT / "results" / "transfer" / spec_name(spec) / "tau2_alphas.json"
    alpha_path.parent.mkdir(parents=True, exist_ok=True)
    alpha_report = {
        "model_key": spec_name(spec),
        "layer": layer,
        "hook": spec["hook"],
        "coding_norm": float(coding.norm().item()),
        "domain": domain,
        "forward": "left_pad_last_token",
        "token_budget": spec["token_budget"],
        "alphas": strength,
    }
    alpha_path.write_text(json.dumps(alpha_report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(
        f"tau2: remove {result['removal']['suppression_among_calls']} induce {result['induction']['induction_among_quiet']}",
        flush=True,
    )
    return result


def write_model_summary(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [
        f"# {report['model_key']} transfer",
        "",
        f"Layer {report['layer']} hook `{report['hook']}`. Coding norm {report['coding_norm']:.3f}.",
        "",
    ]
    domains = (report.get("multi_domain") or {}).get("domains") or {}
    if domains:
        lines.extend(
            [
                "| domain | scale | induction | suppression | random induction | random suppression |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for name, row in domains.items():
            lines.append(
                "| {name} | {scale:.3f} | {ib:.1%}→{ia:.1%} | {sb:.1%}→{sa:.1%} | {rib:.1%}→{ria:.1%} | {rsb:.1%}→{rsa:.1%} |".format(
                    name=name,
                    scale=row["scale"],
                    ib=row["induction"]["top1_before"],
                    ia=row["induction"]["top1_after"],
                    sb=row["suppression"]["top1_before"],
                    sa=row["suppression"]["top1_after"],
                    rib=row["random_induction"]["top1_before"],
                    ria=row["random_induction"]["top1_after"],
                    rsb=row["random_suppression"]["top1_before"],
                    rsa=row["random_suppression"]["top1_after"],
                )
            )
        lines.append("")
    if report.get("verb_free"):
        free = report["verb_free"]
        arm = free["alpha_1"]
        stronger = free["alpha_1_5"]
        random_arm = free["random_alpha_1_5"]
        lines.append(
            f"Verb-free arm {free['arm_n']} of {free['baseline_calls_all']} fresh calls: "
            f"1× {pct(arm['suppression_among_calls'])}, 1.5× {pct(stronger['suppression_among_calls'])}, "
            f"random 1.5× {pct(random_arm['suppression_among_calls'])}."
        )
        lines.append("")
    if report.get("tau2"):
        removal = report["tau2"]["removal"]
        induction = report["tau2"]["induction"]
        lines.append(
            f"Tau2 {report['tau2']['domain']}: suppression {pct(removal['suppression_among_calls'])} "
            f"of {removal['n_baseline_call']} baseline calls; induction {pct(induction['induction_among_quiet'])} "
            f"of {induction['n_baseline_quiet']} baseline text turns; score {pct(report['tau2']['score'])}."
        )
        lines.append("")
    path.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")


def refresh_overview(output_root: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    lock_path = output_root / ".overview.lock"
    with lock_path.open("a", encoding="utf-8") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        _write_overview(output_root)


def _write_overview(output_root: Path) -> None:
    rows = []
    for path in sorted(output_root.glob("*/summary.json")):
        rows.append(json.loads(path.read_text(encoding="utf-8")))
    lines = [
        "# Transfer of the locked coding vector",
        "",
        "Direction is the coding train mean at the locked layer. It is not refit on these sets.",
        "Multi-domain uses that direction rescaled to the target domain contrast norm.",
        "Verb-free and tau2 use alpha 1 of the coding vector. Random controls match that norm and are orthogonal to it.",
        "",
        "Domain cells are induction among quiet prompts / suppression among calling prompts. Tau2 is the same pair of rates. Verb-free is removal among baseline calls.",
        "",
        "| model | layer | hook | retrieval ind/sup | operations ind/sup | communication ind/sup | tau2 sup/ind | verb-free drop 1× / 1.5× |",
        "|---|---:|---|---|---|---|---|---|",
    ]

    def pair(block: dict[str, Any] | None) -> str:
        if not block:
            return "—"
        return f"{rate_text(block['induction']['induction_among_quiet'])} / {rate_text(block['suppression']['suppression_among_calls'])}"

    def rate_text(value: float | None) -> str:
        return "—" if value is None else f"{value:.0%}"

    for report in rows:
        domains = (report.get("multi_domain") or {}).get("domains") or {}
        tau = report.get("tau2") or {}
        free = report.get("verb_free") or {}
        lines.append(
            "| {model} | {layer} | {hook} | {retrieval} | {operations} | {communication} | {tau} | {free} |".format(
                model=report["model_key"],
                layer=report["layer"],
                hook=report["hook"],
                retrieval=pair(domains.get("retrieval")),
                operations=pair(domains.get("operations")),
                communication=pair(domains.get("communication")),
                tau=(
                    f"{rate_text(tau['removal']['suppression_among_calls'])} / {rate_text(tau['induction']['induction_among_quiet'])}"
                    if tau
                    else "—"
                ),
                free=(
                    f"{rate_text(free['alpha_1']['suppression_among_calls'])} / {rate_text(free['alpha_1_5']['suppression_among_calls'])}"
                    if free
                    else "—"
                ),
            )
        )
    lines.append("")
    (output_root / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def checkpoint(destination: Path, report: dict[str, Any], coding: torch.Tensor, layer: int, hook: str) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    torch.save({"mean_diff": coding, "layer": layer, "hook": hook, "protocol": PROTOCOL}, destination / "coding_vector.pt")
    write_model_summary(destination / "summary.json", report)
    refresh_overview(destination.parent)


def load_checkpoint(destination: Path, layer: int, hook: str) -> tuple[dict[str, Any] | None, torch.Tensor | None]:
    report_path = destination / "summary.json"
    vector_path = destination / "coding_vector.pt"
    if not report_path.is_file() or not vector_path.is_file():
        return None, None
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("protocol") != PROTOCOL or int(report.get("layer", -1)) != layer or report.get("hook") != hook:
        return None, None
    blob = torch.load(vector_path, map_location="cpu", weights_only=False)
    if int(blob.get("layer", -1)) != layer or blob.get("hook") != hook or blob.get("protocol") != PROTOCOL:
        return None, None
    vector = torch.as_tensor(blob["mean_diff"], dtype=torch.float32).flatten().contiguous()
    return report, vector


def run_one(
    model_key: str,
    output_root: Path,
    device: str,
    max_count: int,
    arms: set[str],
    token_budget: int = 0,
) -> dict[str, Any]:
    module = load_vector_module()
    spec = dict(module.LOCKED[model_key])
    spec["model_key"] = model_key
    if token_budget:
        spec["token_budget"] = token_budget
    layer = int(spec["layer"])
    started = time.time()
    destination = output_root / model_key
    report, coding = (None, None) if max_count else load_checkpoint(destination, layer, spec["hook"])
    print(f"load {model_key}", flush=True)
    adapter, loader_name = module.load_adapter(Path(spec["path"]), spec["loader"], device)
    module.check_marker(adapter, spec["marker"], spec["marker_id"])
    if coding is None:
        direction_path = REPO_ROOT / "results" / "tool_call_vector" / model_key / "directions.pt"
        if direction_path.exists():
            blob = torch.load(direction_path, map_location="cpu", weights_only=False)
            if int(layer) in blob.get("directions", {}) and blob.get("hook") == spec["hook"]:
                print(f"loaded pre-computed locked vector from {direction_path}", flush=True)
                coding = blob["directions"][int(layer)].float().contiguous()
        if coding is None:
            coding = fit_coding_vector(module, adapter, spec, layer)
        report = {
            "protocol": PROTOCOL,
            "model_key": model_key,
            "layer": layer,
            "hook": spec["hook"],
            "coding_dataset": str(spec["dataset"].relative_to(module.REPO_ROOT)),
            "coding_norm": float(coding.norm().item()),
            "loader": loader_name,
            "token_budget": spec["token_budget"],
        }
        checkpoint(destination, report, coding, layer, spec["hook"])
    else:
        print(f"resume checkpoint layer={layer} hook={spec['hook']}", flush=True)
        report["loader"] = loader_name
        report["token_budget"] = spec["token_budget"]
    print(f"coding norm={float(coding.norm()):.3f} layer={layer} hook={spec['hook']} loader={loader_name}", flush=True)
    if "multi_domain" in arms and "multi_domain" not in report:
        report["multi_domain"] = multi_domain(module, adapter, spec, layer, coding, max_count)
        checkpoint(destination, report, coding, layer, spec["hook"])
    if "verb_free" in arms and "verb_free" not in report:
        report["verb_free"] = verb_free(module, adapter, spec, layer, coding, max_count)
        checkpoint(destination, report, coding, layer, spec["hook"])
    if "tau2" in arms and "tau2" not in report:
        report["tau2"] = tau2(module, adapter, spec, layer, coding, max_count)
        checkpoint(destination, report, coding, layer, spec["hook"])
    report["elapsed_sec"] = round(time.time() - started, 1)
    checkpoint(destination, report, coding, layer, spec["hook"])
    del adapter
    torch.cuda.empty_cache()
    print(f"MODEL_DONE {model_key} elapsed={report['elapsed_sec']}", flush=True)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-key", required=True)
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "results" / "transfer")
    parser.add_argument("--max-per-arm", type=int, default=0)
    parser.add_argument("--arms", default="multi_domain,verb_free,tau2")
    parser.add_argument("--token-budget", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    arms = {arm.strip() for arm in args.arms.split(",") if arm.strip()}
    unknown = arms - {"multi_domain", "verb_free", "tau2"}
    if unknown:
        raise ValueError(f"Unknown arms: {sorted(unknown)}")
    run_one(args.model_key, args.output_root, args.device, args.max_per_arm, arms, args.token_budget)
    return 0


if __name__ == "__main__":
    sys.exit(main())
