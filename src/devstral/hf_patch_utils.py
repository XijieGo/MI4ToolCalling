from __future__ import annotations

from contextlib import ExitStack
from pathlib import Path
from typing import Any, Callable

import torch
from transformers import AutoTokenizer, Mistral3ForConditionalGeneration
from transformers.utils.quantization_config import FineGrainedFP8Config
from transformers.utils import logging as transformers_logging

import sys

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from artifact_paths import DEVSTRAL_2_24B_PATH  # noqa: E402


try:
    transformers_logging.disable_progress_bar()
except Exception:
    pass


MODEL_PATH = DEVSTRAL_2_24B_PATH
TOOL_CALL_TEXT = "[TOOL_CALLS]"


def load_tokenizer(model_path: Path = MODEL_PATH):
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


def load_model(
    model_path: Path = MODEL_PATH,
    *,
    dtype: torch.dtype = torch.bfloat16,
    device_map: object = None,
    max_memory_gib: int | None = None,
    attn_implementation: str | None = None,
):
    del max_memory_gib
    resolved_map = device_map if isinstance(device_map, dict) else {"": 0}
    kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "device_map": resolved_map,
        "quantization_config": FineGrainedFP8Config(dequantize=True),
    }
    if attn_implementation:
        kwargs["attn_implementation"] = attn_implementation
    try:
        model = Mistral3ForConditionalGeneration.from_pretrained(
            str(model_path),
            dtype=dtype,
            **kwargs,
        )
    except TypeError:
        model = Mistral3ForConditionalGeneration.from_pretrained(
            str(model_path),
            torch_dtype=dtype,
            **kwargs,
        )
    model.eval()
    return model


def assert_single_token(tokenizer, token_text: str = TOOL_CALL_TEXT) -> int:
    token_ids = tokenizer.encode(token_text, add_special_tokens=False)
    if len(token_ids) != 1:
        raise RuntimeError(f"{token_text!r} should be one token, got {token_ids}")
    return int(token_ids[0])


def resolve_text_model(model):
    candidates = []
    if hasattr(model, "model"):
        candidates.append(model.model)
        if hasattr(model.model, "language_model"):
            candidates.append(model.model.language_model)
    if hasattr(model, "language_model"):
        candidates.append(model.language_model)
    for candidate in candidates:
        if hasattr(candidate, "layers"):
            return candidate
        if hasattr(candidate, "model") and hasattr(candidate.model, "layers"):
            return candidate.model
    raise AttributeError("Could not resolve Devstral text model.")


def get_layers(model):
    return resolve_text_model(model).layers


def get_layer(model, layer_idx: int):
    return get_layers(model)[int(layer_idx)]


def get_mlp(model, layer_idx: int):
    return get_layer(model, layer_idx).mlp


def get_o_proj(model, layer_idx: int):
    return get_layer(model, layer_idx).self_attn.o_proj


def get_num_layers(model) -> int:
    return len(get_layers(model))


def get_num_heads(model) -> int:
    text_model = resolve_text_model(model)
    return int(text_model.config.num_attention_heads)


def get_head_dim(model) -> int:
    text_model = resolve_text_model(model)
    return int(text_model.config.head_dim)


def get_d_model(model) -> int:
    text_model = resolve_text_model(model)
    return int(text_model.config.hidden_size)


def get_output_weight(model, layer_idx: int) -> torch.Tensor:
    return get_o_proj(model, layer_idx).weight.detach()


def make_last_token_replace_hook(source_cpu: torch.Tensor):
    def hook_fn(module, inputs, output):  # noqa: ANN001
        out = output.clone()
        source = source_cpu.to(device=out.device, dtype=out.dtype)
        out[:, -1, :] = source
        return out

    return hook_fn


def make_last_token_add_hook(delta_cpu: torch.Tensor):
    def hook_fn(module, inputs, output):  # noqa: ANN001
        out = output.clone()
        delta = delta_cpu.to(device=out.device, dtype=out.dtype)
        if delta.ndim == 1:
            delta = delta.view(1, -1)
        out[:, -1, :] = out[:, -1, :] + delta
        return out

    return hook_fn


def make_last_token_subtract_hook(delta_cpu: torch.Tensor):
    def hook_fn(module, inputs, output):  # noqa: ANN001
        out = output.clone()
        delta = delta_cpu.to(device=out.device, dtype=out.dtype)
        if delta.ndim == 1:
            delta = delta.view(1, -1)
        out[:, -1, :] = out[:, -1, :] - delta
        return out

    return hook_fn


def make_capture_hook(store: dict[int, torch.Tensor], layer_idx: int):
    def hook_fn(module, inputs, output):  # noqa: ANN001
        store[int(layer_idx)] = output[:, -1, :].detach().cpu()

    return hook_fn


def make_pre_capture_hook(store: dict[int, torch.Tensor], layer_idx: int):
    def hook_fn(module, inputs):  # noqa: ANN001
        store[int(layer_idx)] = inputs[0][:, -1, :].detach().cpu()

    return hook_fn


def register_hooks(specs: list[tuple[torch.nn.Module, Callable]], pre_specs: list[tuple[torch.nn.Module, Callable]] | None = None):
    stack = ExitStack()
    pre_specs = pre_specs or []
    for module, fn in pre_specs:
        stack.enter_context(module.register_forward_pre_hook(fn))
    for module, fn in specs:
        stack.enter_context(module.register_forward_hook(fn))
    return stack


def tool_stats(logits: torch.Tensor, tool_token_id: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    last_logits = logits[:, -1, :].float()
    tool_logit = last_logits[:, tool_token_id].detach().cpu()
    tool_prob = torch.softmax(last_logits, dim=-1)[:, tool_token_id].detach().cpu()
    top1 = last_logits.argmax(dim=-1).detach().cpu()
    top1_text = torch.tensor(top1.tolist(), dtype=torch.long)
    return tool_logit, tool_prob, top1, top1_text


def to_token_text(tokenizer, token_id: int) -> str:
    return tokenizer.decode([int(token_id)], clean_up_tokenization_spaces=False)


def summarize_rate(numer: int, denom: int) -> float:
    return float(numer / max(int(denom), 1))
