"""Optional Hugging Face model adapter for small, explicit intervention runs.

The repository validator never imports this module's heavy dependencies.  The
adapter uses Hugging Face block-input hooks rather than importing the old
TransformerLens experiment stack.  Its hook convention is recorded in every
run manifest so a result cannot be confused with the historical `hook_resid_*`
variants.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence


@dataclass
class CausalLMAdapter:
    model: Any
    tokenizer: Any | None
    device: Any
    layers: list[Any]
    hook_convention: str = "HuggingFace decoder-block input (pre-block residual state)"
    model_family: str = "causal_lm"

    @property
    def n_layers(self) -> int:
        return len(self.layers)

    def encode(self, text: str) -> Any:
        if self.tokenizer is None:
            raise RuntimeError("This adapter accepts native input IDs rather than rendered prompt text")
        encoded = self.tokenizer(text, add_special_tokens=False, return_tensors="pt")
        return encoded["input_ids"].to(self.device)

    def input_ids(self, ids: Sequence[int]) -> Any:
        """Move one already-tokenized native prompt onto the model device."""

        import torch

        values = [int(value) for value in ids]
        if not values:
            raise ValueError("input_ids must not be empty")
        return torch.tensor([values], dtype=torch.long, device=self.device)

    def logits(self, tokens: Any) -> Any:
        import torch

        with torch.no_grad():
            output = self.model(input_ids=tokens, use_cache=False)
        return output.logits if hasattr(output, "logits") else output[0]

    @contextmanager
    def capture_block_input(self, layer: int) -> Iterator[dict[str, Any]]:
        capture: dict[str, Any] = {}

        def hook(_module: Any, inputs: tuple[Any, ...]) -> None:
            if not inputs or not hasattr(inputs[0], "detach"):
                raise TypeError("The selected decoder block did not receive a tensor as its first input")
            capture["state"] = inputs[0].detach().float().cpu()

        handle = self.layers[layer].register_forward_pre_hook(hook)
        try:
            yield capture
        finally:
            handle.remove()

    @contextmanager
    def patch_block_input(self, layer: int, position: int, replacement: Any) -> Iterator[None]:
        import torch

        replacement_cpu = torch.as_tensor(replacement).detach().cpu()

        def hook(_module: Any, inputs: tuple[Any, ...]) -> tuple[Any, ...]:
            if not inputs or not hasattr(inputs[0], "clone"):
                raise TypeError("The selected decoder block did not receive patchable tensor input")
            hidden = inputs[0].clone()
            replacement_device = replacement_cpu.to(device=hidden.device, dtype=hidden.dtype)
            if replacement_device.ndim == 1:
                hidden[:, position, :] = replacement_device
            elif replacement_device.ndim == 2:
                hidden[:, position, :] = replacement_device
            else:
                raise ValueError(f"Replacement must be [hidden] or [batch, hidden], got {replacement_device.shape}")
            return (hidden, *inputs[1:])

        handle = self.layers[layer].register_forward_pre_hook(hook)
        try:
            yield
        finally:
            handle.remove()

    def capture(self, tokens: Any, layer: int) -> Any:
        with self.capture_block_input(layer) as capture:
            self.logits(tokens)
        if "state" not in capture:
            raise RuntimeError("The decoder block hook did not capture a state")
        return capture["state"]

    def patched_logits(self, tokens: Any, *, layer: int, position: int, replacement: Any) -> Any:
        with self.patch_block_input(layer, position, replacement):
            return self.logits(tokens)


def _find_decoder_layers(model: Any) -> list[Any]:
    candidates = [
        getattr(getattr(model, "model", None), "layers", None),
        getattr(model, "layers", None),
        getattr(getattr(getattr(model, "model", None), "decoder", None), "layers", None),
        getattr(getattr(getattr(model, "model", None), "language_model", None), "layers", None),
        getattr(getattr(model, "language_model", None), "layers", None),
    ]
    for layers in candidates:
        if layers is not None:
            return list(layers)
    raise AttributeError(
        "Could not locate decoder layers; expected model.layers, model.model.layers, "
        "or model.model.language_model.layers"
    )


def load_causal_lm(model_path: str | Path, *, device: str = "cuda", dtype: str = "bfloat16") -> CausalLMAdapter:
    """Load a causal LM lazily; raises an actionable error if optional deps are absent."""

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:  # pragma: no cover - depends on optional environment
        raise RuntimeError("Model studies require the optional 'model' dependencies from pyproject.toml") from exc

    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    try:
        torch_dtype = dtype_map[dtype]
    except KeyError as exc:
        raise ValueError(f"Unsupported dtype {dtype!r}; choose from {sorted(dtype_map)}") from exc

    requested_device = torch.device(device)
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    )
    model.to(requested_device)
    model.eval()
    return CausalLMAdapter(model=model, tokenizer=tokenizer, device=requested_device, layers=_find_decoder_layers(model))


def load_mistral3(
    model_path: str | Path,
    *,
    device: str = "cuda",
    dtype: str = "bfloat16",
) -> CausalLMAdapter:
    """Load Mistral-Small-3.2 while preserving its native token-ID interface.

    Mistral's tool-control tokens are not recoverable through ordinary text
    encoding.  The model-native study therefore supplies exact IDs directly.

    Transfer arms still need to render plain text (the tau2 prompts), so a text
    tokenizer is loaded when the checkpoint ships one. Arms that only need IDs
    keep working when it does not.
    """

    try:
        import torch
        from transformers import Mistral3ForConditionalGeneration
    except ImportError as exc:  # pragma: no cover - depends on optional environment
        raise RuntimeError(
            "Mistral studies require a Transformers build exposing Mistral3ForConditionalGeneration "
            "and the optional 'model' dependencies from pyproject.toml"
        ) from exc

    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    try:
        torch_dtype = dtype_map[dtype]
    except KeyError as exc:
        raise ValueError(f"Unsupported dtype {dtype!r}; choose from {sorted(dtype_map)}") from exc

    try:
        model = Mistral3ForConditionalGeneration.from_pretrained(
            str(model_path),
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        )
    except TypeError:
        # Some supported Transformers releases expose the newer ``dtype``
        # keyword instead of ``torch_dtype``.
        model = Mistral3ForConditionalGeneration.from_pretrained(
            str(model_path),
            dtype=torch_dtype,
            trust_remote_code=True,
        )
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    except Exception:  # pragma: no cover - only when the checkpoint ships no tokenizer
        tokenizer = None
    requested_device = torch.device(device)
    model.to(requested_device)
    model.eval()
    return CausalLMAdapter(
        model=model,
        tokenizer=tokenizer,
        device=requested_device,
        layers=_find_decoder_layers(model),
        model_family="mistral3_conditional_generation",
    )
