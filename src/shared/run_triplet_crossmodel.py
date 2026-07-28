#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gc
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "datasets"


@dataclass(frozen=True)
class PairRecord:
    order: int
    sample_id: str
    clean_path: Path
    corrupt_path: Path
    clean_text: str
    corrupt_text: str
    clean_input_ids: torch.Tensor
    corrupt_input_ids: torch.Tensor
    clean_len: int
    corrupt_len: int
    reference_clean_path: str
    reference_corrupt_path: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cross-model triplet diagnostics from HF hidden states."
    )
    parser.add_argument("--model_path", "--model-path", type=Path, required=True)
    parser.add_argument("--model_name", "--model-name", type=str, required=True)
    parser.add_argument("--output_dir", "--output-dir", type=Path, required=True)
    parser.add_argument("--n_pairs", "--n-pairs", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tool_token_str", "--tool-token-str", type=str, required=True)
    parser.add_argument("--dataset_root", "--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument(
        "--pair-manifest",
        type=Path,
        default=None,
        help="Current-run native JSONL manifest. Defaults to PROMPT_ROOT/manifest.jsonl.",
    )
    parser.add_argument(
        "--sample_manifest",
        "--sample-manifest",
        type=Path,
        default=None,
        help="Legacy CSV manifest support; do not use it for a fresh paper rerun.",
    )
    parser.add_argument(
        "--prompt_root",
        "--prompt-root",
        type=Path,
        required=True,
        help="Current-run native prompt directory containing clean/corrupt prompts.",
    )
    parser.add_argument("--batch_size", "--batch-size", type=int, default=1)
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=("bfloat16", "float16", "float32"))
    parser.add_argument("--device_map", "--device-map", type=str, default="cuda:0")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
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


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def resolve_tool_token_id(tokenizer, requested_text: str, model_path: Path) -> tuple[int, dict[str, Any]]:
    candidates = [requested_text]
    lowered_name = model_path.name.lower()
    if "granite" in lowered_name:
        candidates.extend(["<|tool_call|>", "<tool_call>"])
    if "mistral" in lowered_name or "devstral" in lowered_name:
        candidates.extend(["[TOOL_CALLS]"])

    seen: set[str] = set()
    checked: list[dict[str, Any]] = []
    unk_id = tokenizer.unk_token_id if tokenizer.unk_token_id is not None else -1
    for text in candidates:
        if text in seen:
            continue
        seen.add(text)
        converted = tokenizer.convert_tokens_to_ids(text)
        encode_ids = tokenizer.encode(text, add_special_tokens=False)
        checked.append(
            {
                "text": text,
                "convert_tokens_to_ids": None if converted is None else int(converted),
                "encode_ids": [int(item) for item in encode_ids],
                "encode_len": len(encode_ids),
            }
        )
        if converted is not None and int(converted) >= 0 and int(converted) != int(unk_id):
            return int(converted), {
                "requested_tool_token_text": requested_text,
                "resolved_tool_token_text": text,
                "resolution_method": "convert_tokens_to_ids",
                "checked": checked,
            }
        if len(encode_ids) == 1 and int(encode_ids[0]) != int(unk_id):
            return int(encode_ids[0]), {
                "requested_tool_token_text": requested_text,
                "resolved_tool_token_text": text,
                "resolution_method": "encode_single",
                "checked": checked,
            }
    raise RuntimeError(f"Could not resolve a single tool token for {requested_text!r}; checked={checked}")


def metadata_dataset_root(dataset_root: Path, split: str) -> Path:
    split_root = dataset_root / split
    return split_root if split_root.exists() else dataset_root


def _path_from_manifest_row(row: dict[str, Any], *, side: str, prompt_root: Path) -> Path:
    """Resolve a native prompt using only the current-run prompt directory.

    The Mistral, Devstral, Granite, and Qwen3.5 renderers use different
    manifest field names.  A stale absolute path is therefore reduced to its
    filename and resolved inside ``prompt_root`` rather than resurrecting an
    old project checkout.
    """

    raw_values = [
        row.get(f"{side}_prompt_path"),
        row.get(f"{side}_path"),
        row.get(f"{side}_prompt_relpath"),
        row.get(f"{side}_filename"),
        row.get("filename"),
    ]
    candidates: list[Path] = []
    resolved_root = prompt_root.resolve()
    for raw in raw_values:
        if raw is None or not str(raw).strip():
            continue
        value = Path(str(raw))
        if value.is_file():
            resolved_value = value.resolve()
            if resolved_value.is_relative_to(resolved_root):
                return resolved_value
        if not value.is_absolute():
            candidates.append(prompt_root / value)
        candidates.extend((prompt_root / value.name, prompt_root / side / value.name))
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        f"Could not resolve {side} prompt for sample={row.get('sample_id')!r} under {prompt_root}; "
        f"tried {[str(item) for item in candidates[:8]]}"
    )


def _normalize_manifest_row(row: dict[str, Any], order: int, prompt_root: Path) -> dict[str, Any]:
    sample_id = str(row.get("sample_id") or row.get("source_sample_id") or row.get("pair_id") or f"pair_{order}").strip()
    if not sample_id:
        raise ValueError(f"Manifest row {order} has no usable sample identifier")
    return {
        "order": int(row.get("order") or row.get("pair_id") or order),
        "sample_id": sample_id,
        "clean_path": _path_from_manifest_row(row, side="clean", prompt_root=prompt_root),
        "corrupt_path": _path_from_manifest_row(row, side="corrupt", prompt_root=prompt_root),
        "reference_clean_path": str(row.get("clean_source_path") or row.get("clean_path") or ""),
        "reference_corrupt_path": str(row.get("corrupt_source_path") or row.get("corrupt_path") or ""),
    }


def _discover_prompt_rows(prompt_root: Path) -> list[dict[str, Any]]:
    manifest_path = prompt_root / "manifest.jsonl"
    if manifest_path.exists():
        return [
            _normalize_manifest_row(row, index, prompt_root)
            for index, row in enumerate(read_jsonl(manifest_path), start=1)
        ]

    clean_root = prompt_root / "clean"
    corrupt_root = prompt_root / "corrupt"
    if not clean_root.is_dir() or not corrupt_root.is_dir():
        raise FileNotFoundError(
            f"Expected {manifest_path} or clean/corrupt directories under current prompt root {prompt_root}"
        )
    clean_by_key = {path.name.removeprefix("clean_"): path for path in clean_root.glob("*.txt")}
    corrupt_by_key = {path.name.removeprefix("corrupt_"): path for path in corrupt_root.glob("*.txt")}
    common = sorted(set(clean_by_key) & set(corrupt_by_key))
    if not common:
        raise RuntimeError(f"No matching clean/corrupt prompt pairs under {prompt_root}")
    return [
        {
            "order": index,
            "sample_id": Path(key).stem,
            "clean_path": clean_by_key[key].resolve(),
            "corrupt_path": corrupt_by_key[key].resolve(),
            "reference_clean_path": "",
            "reference_corrupt_path": "",
        }
        for index, key in enumerate(common, start=1)
    ]


def load_pairs(
    tokenizer,
    *,
    prompt_root: Path,
    pair_manifest: Path | None,
    sample_manifest: Path | None,
    n_pairs: int,
) -> list[PairRecord]:
    if pair_manifest is not None and sample_manifest is not None:
        raise ValueError("Pass at most one of --pair-manifest and --sample-manifest")
    if pair_manifest is not None:
        rows = [
            _normalize_manifest_row(row, index, prompt_root)
            for index, row in enumerate(read_jsonl(pair_manifest), start=1)
        ]
    elif sample_manifest is not None:
        rows = [
            _normalize_manifest_row(dict(row), index, prompt_root)
            for index, row in enumerate(read_csv(sample_manifest), start=1)
        ]
    else:
        rows = _discover_prompt_rows(prompt_root)
    if n_pairs > 0:
        rows = rows[: min(int(n_pairs), len(rows))]

    pairs: list[PairRecord] = []
    for row in rows:
        clean_path = Path(row["clean_path"])
        corrupt_path = Path(row["corrupt_path"])
        clean_text = clean_path.read_text(encoding="utf-8")
        corrupt_text = corrupt_path.read_text(encoding="utf-8")
        clean_ids = tokenizer(clean_text, add_special_tokens=False, return_tensors="pt")["input_ids"][0].detach().cpu()
        corrupt_ids = tokenizer(corrupt_text, add_special_tokens=False, return_tensors="pt")["input_ids"][0].detach().cpu()
        pairs.append(
            PairRecord(
                order=int(row["order"]),
                sample_id=str(row["sample_id"]),
                clean_path=clean_path,
                corrupt_path=corrupt_path,
                clean_text=clean_text,
                corrupt_text=corrupt_text,
                clean_input_ids=clean_ids,
                corrupt_input_ids=corrupt_ids,
                clean_len=int(clean_ids.numel()),
                corrupt_len=int(corrupt_ids.numel()),
                reference_clean_path=str(row["reference_clean_path"]),
                reference_corrupt_path=str(row["reference_corrupt_path"]),
            )
        )
    if not pairs:
        raise RuntimeError(f"No usable pairs found in {prompt_root}")
    return pairs


def make_batch(tokenizer, tensors: Sequence[torch.Tensor]) -> dict[str, torch.Tensor]:
    return tokenizer.pad(
        {
            "input_ids": [item.detach().cpu() for item in tensors],
            "attention_mask": [torch.ones_like(item, dtype=torch.long) for item in tensors],
        },
        padding=True,
        return_tensors="pt",
    )


def infer_device(model) -> torch.device:
    return next(model.parameters()).device


def last_token_positions(attention_mask: torch.Tensor) -> torch.Tensor:
    seq_len = int(attention_mask.shape[1])
    idx = torch.arange(seq_len, device=attention_mask.device, dtype=torch.long).view(1, seq_len)
    return (idx * attention_mask.long()).max(dim=1).values


def get_layers_module(model):
    candidates = []
    if hasattr(model, "model"):
        candidates.append(model.model)
        if hasattr(model.model, "language_model"):
            candidates.append(model.model.language_model)
        if hasattr(model.model, "model"):
            candidates.append(model.model.model)
    if hasattr(model, "language_model"):
        candidates.append(model.language_model)
        if hasattr(model.language_model, "model"):
            candidates.append(model.language_model.model)
    for candidate in candidates:
        if hasattr(candidate, "layers"):
            return candidate.layers
    raise AttributeError("Could not locate decoder layers on model.")


def get_text_config(model):
    text_model = None
    try:
        layers = get_layers_module(model)
        # ModuleList parent is not exposed, so fall through to candidate search
        # and use the first config attached to the text stack.
        del layers
    except Exception:
        pass
    candidates = []
    if hasattr(model, "model"):
        candidates.append(model.model)
        if hasattr(model.model, "language_model"):
            candidates.append(model.model.language_model)
        if hasattr(model.model, "model"):
            candidates.append(model.model.model)
    if hasattr(model, "language_model"):
        candidates.append(model.language_model)
        if hasattr(model.language_model, "model"):
            candidates.append(model.language_model.model)
    candidates.append(model)
    for candidate in candidates:
        cfg = getattr(candidate, "config", None)
        if cfg is not None and hasattr(cfg, "hidden_size"):
            text_model = candidate
            break
    if text_model is None:
        raise AttributeError("Could not locate text model config with hidden_size.")
    return text_model.config


def get_hidden_size(model) -> int:
    return int(get_text_config(model).hidden_size)


def get_input_embedding_module(model):
    if hasattr(model, "get_input_embeddings"):
        return model.get_input_embeddings()
    raise AttributeError("Model does not expose get_input_embeddings().")


def get_final_norm(model):
    candidates = []
    if hasattr(model, "model"):
        candidates.append(model.model)
        if hasattr(model.model, "language_model"):
            candidates.append(model.model.language_model)
        if hasattr(model.model, "model"):
            candidates.append(model.model.model)
    if hasattr(model, "language_model"):
        candidates.append(model.language_model)
        if hasattr(model.language_model, "model"):
            candidates.append(model.language_model.model)
    candidates.append(model)
    for candidate in candidates:
        for attr in ("norm", "ln_f", "final_layernorm", "ln_final"):
            module = getattr(candidate, attr, None)
            if module is not None:
                return module
    raise AttributeError("Could not locate final layer norm.")


def get_lm_head_weight(model) -> torch.Tensor:
    if hasattr(model, "lm_head") and hasattr(model.lm_head, "weight"):
        return model.lm_head.weight
    if hasattr(model, "get_output_embeddings"):
        output_embeddings = model.get_output_embeddings()
        if output_embeddings is not None and hasattr(output_embeddings, "weight"):
            return output_embeddings.weight
    raise AttributeError("Could not locate lm_head weight.")


def load_hf_model(model_path: Path, *, dtype: torch.dtype, device_map: str):
    model_kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
    }
    if "devstral" in model_path.name.lower():
        from transformers.utils.quantization_config import FineGrainedFP8Config

        model_kwargs["quantization_config"] = FineGrainedFP8Config(dequantize=True)
    if device_map == "cuda:0":
        model_kwargs["device_map"] = {"": 0}
    else:
        model_kwargs["device_map"] = device_map

    config = AutoConfig.from_pretrained(str(model_path), trust_remote_code=True)
    if getattr(config, "model_type", "") == "mistral3":
        from transformers import Mistral3ForConditionalGeneration

        try:
            return Mistral3ForConditionalGeneration.from_pretrained(
                str(model_path),
                dtype=dtype,
                **model_kwargs,
            )
        except TypeError:
            return Mistral3ForConditionalGeneration.from_pretrained(
                str(model_path),
                torch_dtype=dtype,
                **model_kwargs,
            )

    try:
        return AutoModelForCausalLM.from_pretrained(
            str(model_path),
            dtype=dtype,
            **model_kwargs,
        )
    except TypeError:
        return AutoModelForCausalLM.from_pretrained(
            str(model_path),
            torch_dtype=dtype,
            **model_kwargs,
        )


def collect_hidden_states(
    model,
    tokenizer,
    tensors: Sequence[torch.Tensor],
    *,
    batch_size: int,
) -> np.ndarray:
    layers = get_layers_module(model)
    n_layers = len(layers)
    d_model = get_hidden_size(model)
    out = np.zeros((len(tensors), n_layers + 1, d_model), dtype=np.float32)
    device = infer_device(model)

    ordered = sorted(enumerate(tensors), key=lambda item: int(item[1].numel()))
    for start in tqdm(range(0, len(ordered), max(int(batch_size), 1)), desc="collect hidden", dynamic_ncols=True):
        chunk = ordered[start : start + max(int(batch_size), 1)]
        indices = [idx for idx, _tensor in chunk]
        batch = make_batch(tokenizer, [tensor for _idx, tensor in chunk])
        batch = {key: value.to(device) for key, value in batch.items()}
        positions = last_token_positions(batch["attention_mask"])
        batch_index = torch.arange(batch["input_ids"].shape[0], device=device)

        with torch.inference_mode():
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                use_cache=False,
                output_hidden_states=True,
                return_dict=True,
            )
        hidden_states = outputs.hidden_states
        if len(hidden_states) != n_layers + 1:
            raise RuntimeError(f"Expected {n_layers + 1} hidden states, got {len(hidden_states)}")
        for layer_idx, hidden in enumerate(hidden_states):
            selected = hidden[batch_index, positions, :].detach().float().cpu().numpy()
            out[indices, layer_idx, :] = selected
        del outputs, hidden_states, batch
        clear_cuda()
    return out


def logit_lens_values(
    final_norm,
    lm_head_weight: torch.Tensor,
    resid: np.ndarray,
    tool_token_id: int,
    *,
    batch_layers: int = 8,
) -> np.ndarray:
    device = lm_head_weight.device
    dtype = lm_head_weight.dtype
    direction = lm_head_weight[int(tool_token_id)].detach()
    values = np.zeros((resid.shape[0], resid.shape[1]), dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, resid.shape[1], max(int(batch_layers), 1)):
            chunk = torch.as_tensor(resid[:, start : start + batch_layers, :], device=device, dtype=dtype)
            normed = final_norm(chunk)
            logits = torch.einsum("nld,d->nl", normed, direction)
            values[:, start : start + batch_layers] = logits.detach().float().cpu().numpy()
    return values


def project_writes_to_direction(
    writes: np.ndarray,
    direction: np.ndarray,
    *,
    batch_layers: int = 8,
) -> np.ndarray:
    n_items, n_layers, _d_model = writes.shape
    values = np.zeros((n_items, n_layers), dtype=np.float32)
    direction_t = torch.as_tensor(np.asarray(direction, dtype=np.float32), dtype=torch.float32)
    with torch.inference_mode():
        for start in range(0, n_layers, max(int(batch_layers), 1)):
            stop = min(start + max(int(batch_layers), 1), n_layers)
            chunk = torch.as_tensor(np.ascontiguousarray(writes[:, start:stop, :]), dtype=torch.float32)
            values[:, start:stop] = torch.matmul(chunk, direction_t).cpu().numpy()
    return values


def first_layer_over(rows: Sequence[dict[str, object]], key: str, threshold: float) -> int | None:
    for row in rows:
        if float(row[key]) > float(threshold):
            return int(row["layer"])
    return None


def build_summary(
    *,
    model_name: str,
    model_path: Path,
    n_pairs: int,
    tool_token_id: int,
    tool_token_info: dict[str, Any],
    logit_rows: Sequence[dict[str, object]],
    probe_rows: Sequence[dict[str, object]],
    dla_rows: Sequence[dict[str, object]],
) -> str:
    auc_layer = first_layer_over(probe_rows, "cv_auc_mean", 0.9)
    gap_layer = first_layer_over(logit_rows, "gap", 1.0)
    top3 = sorted(dla_rows, key=lambda row: abs(float(row["dla_delta"])), reverse=True)[:3]
    top3_text = ", ".join(f"L{int(row['layer'])} ({float(row['dla_delta']):+.3f})" for row in top3)
    lines = [
        f"# {model_name} Triplet Summary",
        "",
        f"- n pairs used: `{n_pairs}`",
        f"- model path: `{model_path}`",
        f"- requested tool token: `{tool_token_info.get('requested_tool_token_text')}`",
        f"- resolved tool token: `{tool_token_info.get('resolved_tool_token_text')}`",
        f"- tool token id: `{tool_token_id}`",
        f"- First layer where probe AUC > 0.9: `L{auc_layer}`." if auc_layer is not None else "- First layer where probe AUC > 0.9: `NA`.",
        f"- First layer where logit lens gap > 1.0: `L{gap_layer}`." if gap_layer is not None else "- First layer where logit lens gap > 1.0: `NA`.",
        f"- Top 3 DLA layers by |dla_delta|: {top3_text}.",
    ]
    return "\n".join(lines)


def run_probe(clean_resid: np.ndarray, corrupt_resid: np.ndarray, seed: int) -> list[dict[str, object]]:
    n_pairs, n_layers, _d_model = clean_resid.shape
    y = np.concatenate([np.ones(n_pairs, dtype=np.int64), np.zeros(n_pairs, dtype=np.int64)])
    cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=seed)
    rows: list[dict[str, object]] = []
    for layer in tqdm(range(n_layers), desc="linear probe", dynamic_ncols=True):
        X = np.concatenate([clean_resid[:, layer, :], corrupt_resid[:, layer, :]], axis=0)
        aucs: list[float] = []
        for train_idx, test_idx in cv.split(X, y):
            clf = make_pipeline(
                StandardScaler(),
                LogisticRegression(C=1.0, max_iter=1000, random_state=seed, solver="liblinear"),
            )
            clf.fit(X[train_idx], y[train_idx])
            probs = clf.predict_proba(X[test_idx])[:, 1]
            aucs.append(float(roc_auc_score(y[test_idx], probs)))
        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=1.0, max_iter=1000, random_state=seed, solver="liblinear"),
        )
        clf.fit(X, y)
        train_probs = clf.predict_proba(X)[:, 1]
        rows.append(
            {
                "layer": layer,
                "cv_auc_mean": float(np.mean(aucs)),
                "cv_auc_std": float(np.std(aucs, ddof=1)),
                "train_auc": float(roc_auc_score(y, train_probs)),
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dtype = dtype_from_name(args.dtype)
    tokenizer = AutoTokenizer.from_pretrained(str(args.model_path), trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    tool_token_id, tool_token_info = resolve_tool_token_id(tokenizer, args.tool_token_str, args.model_path)
    prompt_root = args.prompt_root.resolve()

    print(json.dumps({"event": "load_pairs", "model_name": args.model_name}, ensure_ascii=False), flush=True)
    pairs = load_pairs(
        tokenizer,
        prompt_root=prompt_root,
        pair_manifest=args.pair_manifest.resolve() if args.pair_manifest is not None else None,
        sample_manifest=args.sample_manifest,
        n_pairs=args.n_pairs,
    )
    manifest_rows = [
        {
            "order": idx,
            "sample_id": pair.sample_id,
            "clean_path": str(pair.clean_path),
            "corrupt_path": str(pair.corrupt_path),
            "clean_tokens": pair.clean_len,
            "corrupt_tokens": pair.corrupt_len,
            "reference_clean_path": pair.reference_clean_path,
            "reference_corrupt_path": pair.reference_corrupt_path,
        }
        for idx, pair in enumerate(pairs)
    ]
    write_csv(args.output_dir / "sample_manifest.csv", manifest_rows)

    print(json.dumps({"event": "load_model", "model_path": str(args.model_path)}, ensure_ascii=False), flush=True)
    model = load_hf_model(args.model_path, dtype=dtype, device_map=args.device_map)
    model.eval()

    n_layers = len(get_layers_module(model))
    d_model = get_hidden_size(model)
    print(json.dumps({"event": "collect_clean", "n_pairs": len(pairs), "n_layers": n_layers}, ensure_ascii=False), flush=True)
    clean_resid_all = collect_hidden_states(
        model,
        tokenizer,
        [pair.clean_input_ids for pair in pairs],
        batch_size=args.batch_size,
    )
    print(json.dumps({"event": "collect_corrupt", "n_pairs": len(pairs), "n_layers": n_layers}, ensure_ascii=False), flush=True)
    corrupt_resid_all = collect_hidden_states(
        model,
        tokenizer,
        [pair.corrupt_input_ids for pair in pairs],
        batch_size=args.batch_size,
    )

    final_norm = get_final_norm(model)
    lm_head_weight = get_lm_head_weight(model)
    clean_logits = logit_lens_values(final_norm, lm_head_weight, clean_resid_all, tool_token_id)
    corrupt_logits = logit_lens_values(final_norm, lm_head_weight, corrupt_resid_all, tool_token_id)

    clean_resid = clean_resid_all[:, 1:, :]
    corrupt_resid = corrupt_resid_all[:, 1:, :]
    clean_logit_layers = clean_logits[:, 1:]
    corrupt_logit_layers = corrupt_logits[:, 1:]

    gap = clean_logit_layers - corrupt_logit_layers
    logit_rows = [
        {
            "layer": layer,
            "clean_mean": float(clean_logit_layers[:, layer].mean()),
            "corrupt_mean": float(corrupt_logit_layers[:, layer].mean()),
            "gap": float(gap[:, layer].mean()),
            "gap_std": float(gap[:, layer].std(ddof=1)),
        }
        for layer in range(n_layers)
    ]
    logit_csv = args.output_dir / f"logit_lens_{args.model_name}.csv"
    write_csv(logit_csv, logit_rows)

    probe_rows = run_probe(clean_resid, corrupt_resid, args.seed)
    probe_csv = args.output_dir / f"probe_auc_{args.model_name}.csv"
    write_csv(probe_csv, probe_rows)

    unembed_direction = get_lm_head_weight(model)[int(tool_token_id)].detach().float().cpu().numpy()
    clean_writes = clean_resid_all[:, 1:, :] - clean_resid_all[:, :-1, :]
    corrupt_writes = corrupt_resid_all[:, 1:, :] - corrupt_resid_all[:, :-1, :]
    clean_dla = project_writes_to_direction(clean_writes, unembed_direction)
    corrupt_dla = project_writes_to_direction(corrupt_writes, unembed_direction)
    dla_rows = [
        {
            "layer": layer,
            "dla_clean_mean": float(clean_dla[:, layer].mean()),
            "dla_corrupt_mean": float(corrupt_dla[:, layer].mean()),
            "dla_delta": float((clean_dla[:, layer] - corrupt_dla[:, layer]).mean()),
            "clean_mean": float(clean_dla[:, layer].mean()),
            "corrupt_mean": float(corrupt_dla[:, layer].mean()),
            "delta": float((clean_dla[:, layer] - corrupt_dla[:, layer]).mean()),
        }
        for layer in range(n_layers)
    ]
    dla_csv = args.output_dir / f"dla_{args.model_name}.csv"
    write_csv(dla_csv, dla_rows)

    summary = build_summary(
        model_name=args.model_name,
        model_path=args.model_path,
        n_pairs=len(pairs),
        tool_token_id=tool_token_id,
        tool_token_info=tool_token_info,
        logit_rows=logit_rows,
        probe_rows=probe_rows,
        dla_rows=dla_rows,
    )
    summary_path = args.output_dir / "summary.md"
    write_text(summary_path, summary)

    metadata = {
        "seed": args.seed,
        "model_name": args.model_name,
        "model_path": str(args.model_path),
        "dataset_root": str(metadata_dataset_root(args.dataset_root, args.split)),
        "pair_manifest": str(args.pair_manifest) if args.pair_manifest is not None else None,
        "sample_manifest_csv": str(args.sample_manifest) if args.sample_manifest is not None else None,
        "prompt_root": str(prompt_root),
        "output_root": str(args.output_dir),
        "tool_token_id": int(tool_token_id),
        "tool_token_info": tool_token_info,
        "n_layers": int(n_layers),
        "d_model": int(d_model),
        "n_pairs": int(len(pairs)),
        "selected_sample_ids": [pair.sample_id for pair in pairs],
        "outputs": {
            "logit_lens_csv": str(logit_csv),
            "probe_auc_csv": str(probe_csv),
            "dla_csv": str(dla_csv),
            "summary_md": str(summary_path),
            "sample_manifest_csv": str(args.output_dir / "sample_manifest.csv"),
        },
    }
    write_json(args.output_dir / "metadata.json", metadata)

    del clean_resid_all, corrupt_resid_all, model
    clear_cuda()


if __name__ == "__main__":
    main()
