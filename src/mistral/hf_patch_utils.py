from __future__ import annotations

from contextlib import ExitStack
from pathlib import Path
from typing import Any, Callable

import torch
from transformers import AutoTokenizer, Mistral3ForConditionalGeneration
from transformers.utils import logging as transformers_logging

import sys

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from artifact_paths import MISTRAL_3P2_24B_PATH  # noqa: E402


try:
    transformers_logging.disable_progress_bar()
except Exception:
    pass


MODEL_PATH = MISTRAL_3P2_24B_PATH
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
    attn_implementation: str | None = None,
):
    resolved_map = device_map if isinstance(device_map, dict) else {"": 0}
    kwargs: dict[str, Any] = {"trust_remote_code": True, "device_map": resolved_map}
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


def resolve_tool_token_id(tokenizer, token_text: str = TOOL_CALL_TEXT) -> tuple[int, dict[str, Any]]:
    encode_ids = tokenizer.encode(token_text, add_special_tokens=False)
    convert_id = int(tokenizer.convert_tokens_to_ids(token_text))
    info = {
        "tool_token_text": token_text,
        "tool_token_id_via_convert": convert_id,
        "tool_token_ids_via_encode": [int(item) for item in encode_ids],
        "tool_token_encode_len": len(encode_ids),
        "uses_convert_tokens_to_ids": True,
    }
    if convert_id is None or convert_id < 0:
        raise RuntimeError(f"Failed to resolve special token id for {token_text!r}")
    return convert_id, info


def resolve_text_model(model):
    candidates = []
    if hasattr(model, "language_model"):
        candidates.append(model.language_model)
    if hasattr(model, "model"):
        candidates.append(model.model)
        if hasattr(model.model, "language_model"):
            candidates.append(model.model.language_model)
    for candidate in candidates:
        if hasattr(candidate, "layers"):
            return candidate
        if hasattr(candidate, "model") and hasattr(candidate.model, "layers"):
            return candidate.model
    raise AttributeError("Could not resolve Mistral text model.")


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


def get_last_token_positions(attention_mask: torch.Tensor) -> torch.Tensor:
    seq_len = int(attention_mask.shape[1])
    position_ids = torch.arange(seq_len, device=attention_mask.device, dtype=torch.long).view(1, seq_len)
    masked = position_ids * attention_mask.long()
    return masked.max(dim=1).values


def make_last_token_replace_hook(source_cpu: torch.Tensor, positions_cpu: torch.Tensor):
    def hook_fn(module, inputs, output):  # noqa: ANN001
        if isinstance(output, tuple):
            hidden = output[0].clone()
            source = source_cpu.to(device=hidden.device, dtype=hidden.dtype)
            positions = positions_cpu.to(device=hidden.device, dtype=torch.long)
            batch_index = torch.arange(hidden.shape[0], device=hidden.device)
            hidden[batch_index, positions, :] = source
            return (hidden, *output[1:])
        out = output.clone()
        source = source_cpu.to(device=out.device, dtype=out.dtype)
        positions = positions_cpu.to(device=out.device, dtype=torch.long)
        batch_index = torch.arange(out.shape[0], device=out.device)
        out[batch_index, positions, :] = source
        return out

    return hook_fn


def make_last_token_add_hook(delta_cpu: torch.Tensor, positions_cpu: torch.Tensor):
    def hook_fn(module, inputs, output):  # noqa: ANN001
        if isinstance(output, tuple):
            hidden = output[0].clone()
            delta = delta_cpu.to(device=hidden.device, dtype=hidden.dtype)
            if delta.ndim == 1:
                delta = delta.view(1, -1)
            positions = positions_cpu.to(device=hidden.device, dtype=torch.long)
            batch_index = torch.arange(hidden.shape[0], device=hidden.device)
            hidden[batch_index, positions, :] = hidden[batch_index, positions, :] + delta
            return (hidden, *output[1:])
        out = output.clone()
        delta = delta_cpu.to(device=out.device, dtype=out.dtype)
        if delta.ndim == 1:
            delta = delta.view(1, -1)
        positions = positions_cpu.to(device=out.device, dtype=torch.long)
        batch_index = torch.arange(out.shape[0], device=out.device)
        out[batch_index, positions, :] = out[batch_index, positions, :] + delta
        return out

    return hook_fn


def make_last_token_subtract_hook(delta_cpu: torch.Tensor, positions_cpu: torch.Tensor):
    def hook_fn(module, inputs, output):  # noqa: ANN001
        if isinstance(output, tuple):
            hidden = output[0].clone()
            delta = delta_cpu.to(device=hidden.device, dtype=hidden.dtype)
            if delta.ndim == 1:
                delta = delta.view(1, -1)
            positions = positions_cpu.to(device=hidden.device, dtype=torch.long)
            batch_index = torch.arange(hidden.shape[0], device=hidden.device)
            hidden[batch_index, positions, :] = hidden[batch_index, positions, :] - delta
            return (hidden, *output[1:])
        out = output.clone()
        delta = delta_cpu.to(device=out.device, dtype=out.dtype)
        if delta.ndim == 1:
            delta = delta.view(1, -1)
        positions = positions_cpu.to(device=out.device, dtype=torch.long)
        batch_index = torch.arange(out.shape[0], device=out.device)
        out[batch_index, positions, :] = out[batch_index, positions, :] - delta
        return out

    return hook_fn


def make_capture_hook(store: dict[int, torch.Tensor], layer_idx: int, positions_cpu: torch.Tensor):
    def hook_fn(module, inputs, output):  # noqa: ANN001
        tensor = output[0] if isinstance(output, tuple) else output
        positions = positions_cpu.to(device=tensor.device, dtype=torch.long)
        batch_index = torch.arange(tensor.shape[0], device=tensor.device)
        store[int(layer_idx)] = tensor[batch_index, positions, :].detach().cpu()

    return hook_fn


def make_pre_capture_hook(store: dict[int, torch.Tensor], layer_idx: int, positions_cpu: torch.Tensor):
    def hook_fn(module, inputs):  # noqa: ANN001
        tensor = inputs[0]
        positions = positions_cpu.to(device=tensor.device, dtype=torch.long)
        batch_index = torch.arange(tensor.shape[0], device=tensor.device)
        store[int(layer_idx)] = tensor[batch_index, positions, :].detach().cpu()

    return hook_fn


def register_hooks(specs: list[tuple[torch.nn.Module, Callable]], pre_specs: list[tuple[torch.nn.Module, Callable]] | None = None):
    stack = ExitStack()
    pre_specs = pre_specs or []
    for module, fn in pre_specs:
        stack.enter_context(module.register_forward_pre_hook(fn))
    for module, fn in specs:
        stack.enter_context(module.register_forward_hook(fn))
    return stack


def tool_stats(logits: torch.Tensor, tool_token_id: int, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pos = positions.to(device=logits.device, dtype=torch.long)
    batch_index = torch.arange(logits.shape[0], device=logits.device)
    selected_logits = logits[batch_index, pos, :].float()
    tool_logit = selected_logits[:, tool_token_id].detach().cpu()
    tool_prob = torch.softmax(selected_logits, dim=-1)[:, tool_token_id].detach().cpu()
    top1 = selected_logits.argmax(dim=-1).detach().cpu()
    return tool_logit, tool_prob, top1


def to_token_text(tokenizer, token_id: int) -> str:
    return tokenizer.decode([int(token_id)], clean_up_tokenization_spaces=False)
