#!/usr/bin/env python3
"""Fit one locked tool-call vector per model and score sufficiency / necessity.

Each model uses the layer and hook chosen after the layer search. The
vector is the mean train residual difference, clean minus corrupt, at the
last real prompt token. Held-out sufficiency adds that vector to corrupt
prompts. Necessity subtracts it from clean prompts. Pairs are not dropped
when a fresh baseline disagrees with the frozen screen.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from mi4tc.io import write_json  # noqa: E402
from mi4tc.model import CausalLMAdapter, _find_decoder_layers, load_mistral3  # noqa: E402
from mi4tc.pairs import load_model_native_pairs, native_pair_token_ids, token_ids  # noqa: E402


HOOK_CONVENTION = {
    "pre": "HuggingFace decoder-block input (pre-block residual state)",
    "post": "HuggingFace decoder-block output (post-block residual state)",
}

# Layer indices are 0-based. Qwen3, Granite, and Mistral use the block input.
# Qwen3.5 uses the block output. Splits are the ones on which the layer was chosen.
def _model_dir(name: str, fallback: str) -> str:
    local_p = Path("/home/xijie/models") / name
    if local_p.exists():
        return str(local_p)
    return fallback

# Layer indices are 0-based.
LOCKED: dict[str, dict[str, Any]] = {
    "qwen3_4b": {
        "path": _model_dir("Qwen3-4B", "/root/autodl-tmp/Qwen/Qwen3-4B"),
        "dataset": REPO_ROOT / "datasets/qwen3_4b/pair",
        "layout": "native",
        "loader": "causal",
        "marker": "<tool_call>",
        "marker_id": 151657,
        "token_budget": 131072,
        "layer": 26,
        "hook": "pre",
    },
    "qwen3_8b": {
        "path": _model_dir("Qwen3-8B", "/root/autodl-tmp/Qwen/Qwen3-8B"),
        "dataset": REPO_ROOT / "datasets/qwen3_8b/pair",
        "layout": "text_dir",
        "loader": "causal",
        "marker": "<tool_call>",
        "marker_id": 151657,
        "token_budget": 98304,
        "layer": 24,
        "hook": "pre",
    },
    "qwen3_14b": {
        "path": _model_dir("Qwen3-14B", "/root/autodl-tmp/Qwen/Qwen3-14B"),
        "dataset": REPO_ROOT / "datasets/qwen3_14b/pair",
        "layout": "native",
        "loader": "causal",
        "marker": "<tool_call>",
        "marker_id": 151657,
        "token_budget": 49152,
        "layer": 33,
        "hook": "pre",
    },
    "qwen35_4b": {
        "path": _model_dir("Qwen3.5-4B", "/root/autodl-tmp/Qwen/Qwen3.5-4B"),
        "dataset": REPO_ROOT / "trash/selected_500/qwen35_4b",
        "layout": "text_dir",
        "loader": "auto",
        "marker": "<tool_call>",
        "marker_id": 248058,
        "token_budget": 65536,
        "layer": 31,
        "hook": "post",
    },
    "qwen35_9b": {
        "path": _model_dir("Qwen3.5-9B", "/root/autodl-tmp/Qwen/Qwen3.5-9B"),
        "dataset": REPO_ROOT / "trash/selected_500/qwen35_9b",
        "layout": "text_dir",
        "loader": "auto",
        "marker": "<tool_call>",
        "marker_id": 248058,
        "token_budget": 16384,
        "layer": 31,
        "hook": "post",
    },
    "granite_3p3_8b": {
        "path": _model_dir("granite-3.3-8b-instruct", "/root/autodl-tmp/Granite/granite-3.3-8b-instruct"),
        "dataset": REPO_ROOT / "datasets/granite_3p3_8b/pair",
        "layout": "native",
        "loader": "causal",
        "marker": "<|tool_call|>",
        "marker_id": 49154,
        "token_budget": 65536,
        "layer": 35,
        "hook": "pre",
    },
    "mistral_3p2_24b": {
        "path": _model_dir("Mistral-Small-3.2-24B-Instruct-2506", "/root/autodl-tmp/Mistral-Small-3.2-24B-Instruct-2506"),
        "dataset": REPO_ROOT / "datasets/mistral_3p2_24b/pair",
        "layout": "native",
        "loader": "mistral",
        "marker": "[TOOL_CALLS]",
        "marker_id": 9,
        "token_budget": 24576,
        "layer": 25,
        "hook": "pre",
    },
}


def tensor_from_output(output: Any) -> torch.Tensor:
    hidden = output[0] if isinstance(output, tuple) else output
    if not torch.is_tensor(hidden) or hidden.ndim != 3:
        raise TypeError(f"Decoder block output is not a rank-3 residual tensor: {type(output)!r}")
    return hidden


def replace_output(output: Any, hidden: torch.Tensor) -> Any:
    if isinstance(output, tuple):
        return (hidden, *output[1:])
    return hidden


def load_text_dir(dataset_root: Path, max_train: int, max_heldout: int) -> tuple[list[tuple[str, str, str]], list[tuple[str, str, str]]]:
    """Load Qwen3-8B pair text. That directory has no native manifest."""

    loaded: dict[str, list[tuple[str, str, str]]] = {}
    for split in ("train", "heldout"):
        clean_dir = dataset_root / split / "clean"
        corrupt_dir = dataset_root / split / "corrupt"
        clean_names = {path.name for path in clean_dir.glob("*.txt")}
        corrupt_names = {path.name for path in corrupt_dir.glob("*.txt")}
        if clean_names != corrupt_names:
            raise ValueError(f"{split}: clean/corrupt filenames differ")
        rows = []
        for name in sorted(clean_names):
            rows.append(
                (
                    Path(name).stem,
                    (clean_dir / name).read_text(encoding="utf-8"),
                    (corrupt_dir / name).read_text(encoding="utf-8"),
                )
            )
        loaded[split] = rows
    if max_train:
        loaded["train"] = loaded["train"][:max_train]
    if max_heldout:
        loaded["heldout"] = loaded["heldout"][:max_heldout]
    if not loaded["train"] or not loaded["heldout"]:
        raise ValueError(f"{dataset_root}: empty train or held-out split")
    return loaded["train"], loaded["heldout"]


def load_adapter(path: Path, loader: str, device: str) -> tuple[CausalLMAdapter, str]:
    if loader == "mistral":
        return load_mistral3(path, device=device, dtype="bfloat16"), "mistral3"

    import transformers

    kwargs = {
        "dtype": torch.bfloat16,
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
    }
    names = ("AutoModelForCausalLM",) if loader == "causal" else ("AutoModelForCausalLM", "AutoModelForImageTextToText")
    errors: list[str] = []
    for name in names:
        factory = getattr(transformers, name, None)
        if factory is None:
            continue
        try:
            model = factory.from_pretrained(str(path), **kwargs)
            model.to(device)
            model.eval()
            tokenizer = transformers.AutoTokenizer.from_pretrained(str(path), trust_remote_code=True)
            adapter = CausalLMAdapter(
                model=model,
                tokenizer=tokenizer,
                device=torch.device(device),
                layers=_find_decoder_layers(model),
            )
            return adapter, name
        except Exception as exc:
            errors.append(f"{name}: {type(exc).__name__}: {exc}")
    raise RuntimeError("Could not load model\n" + "\n".join(errors))


def hidden_from_call(args: tuple[Any, ...], kwargs: dict[str, Any]) -> torch.Tensor:
    if args and torch.is_tensor(args[0]) and args[0].ndim == 3:
        return args[0]
    hidden = kwargs.get("hidden_states")
    if torch.is_tensor(hidden) and hidden.ndim == 3:
        return hidden
    raise TypeError("Decoder block input is not a rank-3 residual tensor")


def replace_hidden(args: tuple[Any, ...], kwargs: dict[str, Any], hidden: torch.Tensor) -> tuple[tuple[Any, ...], dict[str, Any]]:
    if args and torch.is_tensor(args[0]) and args[0].ndim == 3:
        return (hidden, *args[1:]), kwargs
    updated = dict(kwargs)
    updated["hidden_states"] = hidden
    return args, updated


def pad_id_for(adapter: CausalLMAdapter) -> int:
    tokenizer = adapter.tokenizer
    if tokenizer is not None:
        for value in (tokenizer.pad_token_id, tokenizer.eos_token_id):
            if value is not None:
                return int(value)
    config = adapter.model.config
    for value in (getattr(config, "pad_token_id", None), getattr(config, "eos_token_id", None)):
        if isinstance(value, int):
            return value
    text_config = getattr(config, "text_config", None)
    if text_config is not None:
        for value in (getattr(text_config, "pad_token_id", None), getattr(text_config, "eos_token_id", None)):
            if isinstance(value, int):
                return value
    return 0


def check_marker(adapter: CausalLMAdapter, marker: str, marker_id: int) -> None:
    if adapter.tokenizer is None:
        return
    encoded = token_ids(adapter.tokenizer, marker)
    converted = adapter.tokenizer.convert_tokens_to_ids(marker)
    if encoded != [marker_id] and converted != marker_id:
        raise ValueError(f"{marker!r} encoded as {encoded}, convert_tokens_to_ids={converted}, expected {marker_id}")


def iter_batches(lengths: list[int], token_budget: int, max_batch: int) -> list[list[int]]:
    order = sorted(range(len(lengths)), key=lambda index: lengths[index])
    batches: list[list[int]] = []
    current: list[int] = []
    for index in order:
        if not current:
            current = [index]
            continue
        longest = max(lengths[item] for item in current)
        padded = (len(current) + 1) * max(longest, lengths[index])
        if padded > token_budget or len(current) >= max_batch:
            batches.append(current)
            current = [index]
        else:
            current.append(index)
    if current:
        batches.append(current)
    return batches


class LayerSweep:
    def __init__(
        self,
        adapter: CausalLMAdapter,
        tool_id: int,
        layers: list[int],
        token_budget: int,
        hook: str = "pre",
        last_token_only: bool = False,
    ) -> None:
        if hook not in {"pre", "post"}:
            raise ValueError(f"Unsupported hook {hook!r}")
        self.adapter = adapter
        self.tool_id = tool_id
        self.layers = layers
        self.token_budget = token_budget
        self.hook = hook
        self.last_token_only = last_token_only
        self.pad_id = pad_id_for(adapter)
        self.device = adapter.device

    def _forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        kwargs: dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "use_cache": False,
        }
        if self.last_token_only:
            kwargs["logits_to_keep"] = 1
        try:
            output = self.adapter.model(**kwargs)
        except TypeError:
            output = self.adapter.model(input_ids=input_ids, attention_mask=attention_mask)
        logits = output.logits if hasattr(output, "logits") else output[0]
        if logits.ndim != 3:
            raise RuntimeError(f"Expected logits [batch, sequence, vocab], got {tuple(logits.shape)}")
        return logits

    def _pad(self, sequences: list[list[int]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        width = max(len(sequence) for sequence in sequences)
        input_ids = torch.full((len(sequences), width), self.pad_id, dtype=torch.long, device=self.device)
        attention_mask = torch.zeros((len(sequences), width), dtype=torch.long, device=self.device)
        positions = torch.empty(len(sequences), dtype=torch.long, device=self.device)
        for row, sequence in enumerate(sequences):
            if not sequence:
                raise ValueError("Empty token sequence")
            tokens = torch.tensor(sequence, dtype=torch.long, device=self.device)
            if self.last_token_only:
                input_ids[row, -len(sequence) :] = tokens
                attention_mask[row, -len(sequence) :] = 1
                positions[row] = width - 1
            else:
                input_ids[row, : len(sequence)] = tokens
                attention_mask[row, : len(sequence)] = 1
                positions[row] = len(sequence) - 1
        return input_ids, attention_mask, positions

    def _run_batch(
        self,
        sequences: list[list[int]],
        *,
        capture: bool,
        patch_layer: int | None = None,
        replacement: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        input_ids, attention_mask, positions = self._pad(sequences)
        captured: dict[int, torch.Tensor] = {}
        calls: dict[int, int] = {}
        handles = []
        if capture:
            for layer in self.layers:
                def make_capture(layer_index: int):
                    def remember(hidden: torch.Tensor) -> None:
                        calls[layer_index] = calls.get(layer_index, 0) + 1
                        if calls[layer_index] != 1:
                            return
                        rows = torch.arange(hidden.shape[0], device=hidden.device)
                        captured[layer_index] = hidden[rows, positions].detach().to(dtype=torch.float32).cpu()

                    def pre_hook(module: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
                        remember(hidden_from_call(args, kwargs))
                        return None

                    def post_hook(module: Any, args: tuple[Any, ...], kwargs: dict[str, Any], output: Any) -> None:
                        remember(tensor_from_output(output))
                        return None

                    return post_hook if self.hook == "post" else pre_hook

                register = (
                    self.adapter.layers[layer].register_forward_hook
                    if self.hook == "post"
                    else self.adapter.layers[layer].register_forward_pre_hook
                )
                handles.append(register(make_capture(layer), with_kwargs=True))
        if patch_layer is not None:
            if replacement is None or replacement.shape[0] != len(sequences):
                raise ValueError("Patch replacement must align with the batch")
            replacement_cpu = replacement.detach().to(dtype=torch.float32).cpu()

            def write_replacement(hidden: torch.Tensor) -> torch.Tensor:
                patched = hidden.clone()
                rows = torch.arange(patched.shape[0], device=patched.device)
                value = replacement_cpu.to(device=patched.device, dtype=patched.dtype)
                patched[rows, positions] = value
                return patched

            if self.hook == "post":

                def post_patch(module: Any, args: tuple[Any, ...], kwargs: dict[str, Any], output: Any) -> Any:
                    return replace_output(output, write_replacement(tensor_from_output(output)))

                handles.append(
                    self.adapter.layers[patch_layer].register_forward_hook(post_patch, with_kwargs=True)
                )
            else:

                def pre_patch(module: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> tuple[tuple[Any, ...], dict[str, Any]]:
                    return replace_hidden(args, kwargs, write_replacement(hidden_from_call(args, kwargs)))

                handles.append(
                    self.adapter.layers[patch_layer].register_forward_pre_hook(pre_patch, with_kwargs=True)
                )
        try:
            with torch.inference_mode():
                logits = self._forward(input_ids, attention_mask)
            if logits.shape[1] == 1:
                last = logits[:, 0, :].float()
            else:
                rows = torch.arange(logits.shape[0], device=logits.device)
                last = logits[rows, positions].float()
            tool_logit = last[:, self.tool_id].detach().cpu()
            tool_top1 = (last.argmax(dim=-1) == self.tool_id).detach().cpu()
        finally:
            for handle in handles:
                handle.remove()
        if capture:
            missing = [layer for layer in self.layers if layer not in captured]
            if missing:
                raise RuntimeError(f"Capture missed layers {missing}")
            repeated = {layer: count for layer, count in calls.items() if count != 1}
            if repeated:
                print(f"warning: decoder layers fired more than once, kept the first call: {repeated}", flush=True)
        return {"tool_logit": tool_logit, "tool_top1": tool_top1, "states": captured}

    def _map(self, sequences: list[list[int]], **kwargs: Any) -> dict[str, Any]:
        logit_parts: list[torch.Tensor | None] = [None] * len(sequences)
        top1_parts: list[torch.Tensor | None] = [None] * len(sequences)
        state_parts: dict[int, list[torch.Tensor | None]] = {layer: [None] * len(sequences) for layer in self.layers}
        batches = iter_batches([len(sequence) for sequence in sequences], self.token_budget, max_batch=128)

        def consume(indices: list[int]) -> None:
            try:
                batch_sequences = [sequences[index] for index in indices]
                replacement = None
                if kwargs.get("replacement") is not None:
                    replacement = kwargs["replacement"][indices]
                result = self._run_batch(
                    batch_sequences,
                    capture=kwargs.get("capture", False),
                    patch_layer=kwargs.get("patch_layer"),
                    replacement=replacement,
                )
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if len(indices) == 1:
                    raise
                middle = len(indices) // 2
                print(f"OOM at batch {len(indices)}; splitting", flush=True)
                consume(indices[:middle])
                consume(indices[middle:])
                return
            for local, index in enumerate(indices):
                logit_parts[index] = result["tool_logit"][local]
                top1_parts[index] = result["tool_top1"][local]
                for layer in self.layers:
                    if layer in result["states"]:
                        state_parts[layer][index] = result["states"][layer][local]

        for batch in batches:
            consume(batch)
        if any(part is None for part in logit_parts):
            raise RuntimeError("A sequence was not scored")
        states = {}
        if kwargs.get("capture", False):
            states = {layer: torch.stack(state_parts[layer]).float() for layer in self.layers}
        return {
            "tool_logit": torch.stack(logit_parts).float(),
            "tool_top1": torch.stack(top1_parts).bool(),
            "states": states,
        }

    def capture(self, sequences: list[list[int]]) -> dict[str, Any]:
        return self._map(sequences, capture=True)

    def intervene(self, sequences: list[list[int]], layer: int, replacement: torch.Tensor) -> dict[str, Any]:
        return self._map(sequences, capture=False, patch_layer=layer, replacement=replacement)

    def padding_check(self, sequences: list[list[int]]) -> float:
        """Compare one right-padded batch of two prompts with two unpadded forwards."""

        if len(sequences) < 2:
            return 0.0
        sample = sequences[:2]
        alone = [float(self._run_batch([sequence], capture=False)["tool_logit"][0]) for sequence in sample]
        paired = self._run_batch(sample, capture=False)["tool_logit"]
        return max(abs(alone[0] - float(paired[0])), abs(alone[1] - float(paired[1])))


def ids_from_native(collection: Any, tokenizer: Any, max_train: int, max_heldout: int) -> tuple[list[list[int]], list[list[int]], list[list[int]], list[list[int]]]:
    def take(split: str, limit: int) -> tuple[list[list[int]], list[list[int]]]:
        clean_rows: list[list[int]] = []
        corrupt_rows: list[list[int]] = []
        for pair in collection.split_pairs(split, max_pairs=limit):
            clean_ids, corrupt_ids = native_pair_token_ids(pair, tokenizer=tokenizer)
            clean_rows.append(list(clean_ids))
            corrupt_rows.append(list(corrupt_ids))
        return clean_rows, corrupt_rows

    train_clean, train_corrupt = take("train", max_train)
    held_clean, held_corrupt = take("heldout", max_heldout)
    return train_clean, train_corrupt, held_clean, held_corrupt


def ids_from_text(rows: list[tuple[str, str, str]], tokenizer: Any) -> tuple[list[list[int]], list[list[int]]]:
    clean_rows = [token_ids(tokenizer, clean) for _sample_id, clean, _corrupt in rows]
    corrupt_rows = [token_ids(tokenizer, corrupt) for _sample_id, _clean, corrupt in rows]
    return clean_rows, corrupt_rows


def mean(values: torch.Tensor) -> float:
    return float(values.float().mean().item())


def rate(values: torch.Tensor) -> float:
    return float(values.float().mean().item())


def write_table(path: Path, report: dict[str, Any]) -> None:
    lines = [
        f"# {report['model_key']} locked layer",
        "",
        f"Layer {report['layer']} is 0-based. Hook: {report['hook']}. Depth is `(layer + 1) / {report['n_layers']}`.",
        f"Train {report['n_train']} / held-out {report['n_heldout']}.",
        f"Vector: mean train clean−corrupt at the last real token. Suff adds it to held-out corrupt prompts. Necc subtracts it from held-out clean prompts.",
        f"Baseline held-out top-1: clean {report['clean_top1_before']:.3f}, corrupt {report['corrupt_top1_before']:.3f}.",
        f"Logit gap (clean − corrupt): {report['logit_gap']:.3f}.",
        "",
        "| layer | depth | r(l,p) | suff | necc | add top-1 | remove top-1 | Δlogit add | Δlogit remove |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["layers"]:
        r_lp_val = row.get("r_lp")
        r_lp_str = f"{r_lp_val:.1%}" if r_lp_val is not None else "—"
        lines.append(
            "| {layer} | {depth:.1%} | {r_lp_str} | {suff:.3f} | {necc:.3f} | {corrupt_top1_before:.1%}→{add_top1:.1%} | {clean_top1_before:.1%}→{remove_top1:.1%} | {add_logit_delta:+.3f} | {remove_logit_delta:+.3f} |".format(
                r_lp_str=r_lp_str,
                **row
            )
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def refresh_vector_overview(output_root: Path) -> None:
    reports = []
    for path in sorted(output_root.glob("*/summary.json")):
        try:
            reports.append(json.loads(path.read_text(encoding="utf-8")))
        except Exception:
            pass
    if not reports:
        return
    lines = [
        "# Locked Layer Tool Call Vector Summary",
        "",
        "| Model | Layer | Depth | Hook | r(l,p) | Suff. | Necc. | Add top-1 | Remove top-1 |",
        "|---|---:|---:|---|---:|---:|---:|---|---|",
    ]
    models_summary = []
    for r in reports:
        row = r["layers"][0] if r.get("layers") else {}
        r_lp_val = row.get("r_lp")
        r_lp_str = f"{r_lp_val:.1%}" if r_lp_val is not None else "—"
        suff_val = row.get("suff")
        suff_str = f"{suff_val:.3f}" if suff_val is not None else "—"
        necc_val = row.get("necc")
        necc_str = f"{necc_val:.3f}" if necc_val is not None else "—"
        add_str = f"{row.get('corrupt_top1_before', 0):.1%}→{row.get('add_top1', 0):.1%}"
        rem_str = f"{row.get('clean_top1_before', 0):.1%}→{row.get('remove_top1', 0):.1%}"
        lines.append(
            f"| {r['model_key']} | {r['layer']} | {r['depth']:.1%} | {r['hook']} | {r_lp_str} | {suff_str} | {necc_str} | {add_str} | {rem_str} |"
        )
        models_summary.append({
            "model_key": r["model_key"],
            "layer": r["layer"],
            "depth": r["depth"],
            "hook": r["hook"],
            "r_lp": r_lp_val,
            "suff": suff_val,
            "necc": necc_val,
            "add_top1": row.get("add_top1"),
            "remove_top1": row.get("remove_top1"),
        })
    lines.append("")
    (output_root / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    (output_root / "summary.json").write_text(json.dumps({"models": models_summary}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def run_model(
    model_key: str,
    output_root: Path,
    max_train: int,
    max_heldout: int,
    device: str,
) -> dict[str, Any]:
    spec = LOCKED[model_key]
    dataset = spec["dataset"]
    layout = spec["layout"]
    hook = spec["hook"]
    layer = int(spec["layer"])
    started = time.time()
    print(f"load {model_key} from {spec['path']}", flush=True)
    adapter, loader_name = load_adapter(Path(spec["path"]), spec["loader"], device)
    check_marker(adapter, spec["marker"], spec["marker_id"])
    if layer < 0 or layer >= adapter.n_layers:
        raise ValueError(f"{model_key}: locked layer {layer} is outside 0..{adapter.n_layers - 1}")
    layers = [layer]
    print(
        f"{model_key}: n_layers={adapter.n_layers} layer={layer} hook={hook} loader={loader_name}",
        flush=True,
    )
    if layout == "text_dir":
        train_rows, held_rows = load_text_dir(dataset, max_train, max_heldout)
        train_clean, train_corrupt = ids_from_text(train_rows, adapter.tokenizer)
        held_clean, held_corrupt = ids_from_text(held_rows, adapter.tokenizer)
    else:
        collection = load_model_native_pairs(dataset)
        if collection.tool_call_token_id != spec["marker_id"] or collection.tool_call_marker != spec["marker"]:
            raise ValueError(
                f"{model_key}: dataset marker {collection.tool_call_marker}/{collection.tool_call_token_id} "
                f"!= {spec['marker']}/{spec['marker_id']}"
            )
        train_clean, train_corrupt, held_clean, held_corrupt = ids_from_native(
            collection, adapter.tokenizer, max_train, max_heldout
        )
    sweep = LayerSweep(adapter, spec["marker_id"], layers, spec["token_budget"], hook=hook)
    padding_diff = sweep.padding_check(held_clean)
    print(f"{model_key}: padding check max |Δlogit|={padding_diff:.4f}", flush=True)
    if padding_diff > 0.5:
        raise RuntimeError(f"{model_key}: padded batch disagrees with unpadded forward by {padding_diff:.4f}")

    print(f"{model_key}: fit direction on {len(train_clean)} train pairs", flush=True)
    train_clean_cap = sweep.capture(train_clean)
    train_corrupt_cap = sweep.capture(train_corrupt)
    directions = {
        layer: train_clean_cap["states"][layer] - train_corrupt_cap["states"][layer]
        for layer in layers
    }
    direction_mean = {layer: values.mean(dim=0) for layer, values in directions.items()}

    print(f"{model_key}: held-out baseline on {len(held_clean)} pairs", flush=True)
    held_clean_cap = sweep.capture(held_clean)
    held_corrupt_cap = sweep.capture(held_corrupt)
    clean_logit = held_clean_cap["tool_logit"]
    corrupt_logit = held_corrupt_cap["tool_logit"]
    clean_top1 = held_clean_cap["tool_top1"]
    corrupt_top1 = held_corrupt_cap["tool_top1"]
    gap = mean(clean_logit) - mean(corrupt_logit)
    clean_top1_rate = rate(clean_top1)
    corrupt_top1_rate = rate(corrupt_top1)
    print(
        f"{model_key}: baseline clean_top1={clean_top1_rate:.3f} corrupt_top1={corrupt_top1_rate:.3f} gap={gap:.3f}",
        flush=True,
    )
    if clean_top1_rate < 0.80 or corrupt_top1_rate > 0.20:
        raise RuntimeError(
            f"{model_key}: held-out baseline is not the screened contrast "
            f"(clean_top1={clean_top1_rate:.3f}, corrupt_top1={corrupt_top1_rate:.3f})"
        )
    if abs(gap) < 1e-3:
        raise RuntimeError(f"{model_key}: clean/corrupt logit gap is {gap}")

    rows = []
    output_root.mkdir(parents=True, exist_ok=True)
    for layer in layers:
        direction = direction_mean[layer]
        added = sweep.intervene(held_corrupt, layer, held_corrupt_cap["states"][layer] + direction)
        removed = sweep.intervene(held_clean, layer, held_clean_cap["states"][layer] - direction)
        patched = sweep.intervene(held_corrupt, layer, held_clean_cap["states"][layer])
        add_delta = mean(added["tool_logit"]) - mean(corrupt_logit)
        remove_delta = mean(clean_logit) - mean(removed["tool_logit"])
        r_lp = rate(patched["tool_top1"])
        row = {
            "layer": layer,
            "depth": (layer + 1) / adapter.n_layers,
            "r_lp": r_lp,
            "suff": add_delta / gap,
            "necc": remove_delta / gap,
            "add_top1": rate(added["tool_top1"]),
            "remove_top1": rate(removed["tool_top1"]),
            "clean_top1_before": clean_top1_rate,
            "corrupt_top1_before": corrupt_top1_rate,
            "add_logit_delta": add_delta,
            "remove_logit_delta": remove_delta,
            "direction_norm": float(torch.linalg.vector_norm(direction).item()),
        }
        rows.append(row)
        print(
            f"{model_key}: layer {layer} depth={row['depth']:.1%} r(l,p)={r_lp:.1%} suff={row['suff']:.3f} necc={row['necc']:.3f} "
            f"add_top1={row['add_top1']:.3f} remove_top1={row['remove_top1']:.3f}",
            flush=True,
        )
        partial = {
            "model_key": model_key,
            "n_layers": adapter.n_layers,
            "layer": layer,
            "hook": hook,
            "n_train": len(train_clean),
            "n_heldout": len(held_clean),
            "clean_top1_before": clean_top1_rate,
            "corrupt_top1_before": corrupt_top1_rate,
            "logit_gap": gap,
            "layers": rows,
        }
        write_table(output_root / "summary.md", partial)

    report = {
        "model_key": model_key,
        "model_path": spec["path"],
        "dataset": str(dataset.resolve().relative_to(REPO_ROOT)),
        "loader": loader_name,
        "hook": hook,
        "hook_convention": HOOK_CONVENTION[hook],
        "tool_call_marker": spec["marker"],
        "tool_call_token_id": spec["marker_id"],
        "n_layers": adapter.n_layers,
        "layer": layer,
        "depth": (layer + 1) / adapter.n_layers,
        "n_train": len(train_clean),
        "n_heldout": len(held_clean),
        "clean_top1_before": clean_top1_rate,
        "corrupt_top1_before": corrupt_top1_rate,
        "clean_logit": mean(clean_logit),
        "corrupt_logit": mean(corrupt_logit),
        "logit_gap": gap,
        "padding_check_max_abs_logit_diff": padding_diff,
        "token_budget": sweep.token_budget,
        "elapsed_sec": round(time.time() - started, 1),
        "layers": rows,
    }
    write_json(output_root / "summary.json", report)
    write_table(output_root / "summary.md", report)
    torch.save(
        {
            "directions": {layer: direction_mean[layer].cpu() for layer in layers},
            "estimator": "mean(clean - corrupt) at the last real token",
            "fit_split": "train",
            "hook": hook,
            "hook_convention": HOOK_CONVENTION[hook],
            "layer_index": "0-based",
        },
        output_root / "directions.pt",
    )
    refresh_vector_overview(output_root.parent)
    del adapter
    gc.collect()
    torch.cuda.empty_cache()
    print(f"MODEL_DONE {model_key} clean_top1={clean_top1_rate:.3f} corrupt_top1={corrupt_top1_rate:.3f} elapsed={report['elapsed_sec']}", flush=True)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-key", required=True, choices=tuple(LOCKED))
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--max-train-pairs", type=int, default=0)
    parser.add_argument("--max-heldout-pairs", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.max_train_pairs < 0 or args.max_heldout_pairs < 0:
        raise ValueError("pair limits must be non-negative")
    output_root = args.output_root or (REPO_ROOT / "results" / "tool_call_vector" / args.model_key)
    run_model(
        args.model_key,
        output_root,
        args.max_train_pairs,
        args.max_heldout_pairs,
        args.device,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
