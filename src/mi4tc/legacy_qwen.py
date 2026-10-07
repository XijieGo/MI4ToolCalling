"""TransformerLens Qwen3 loader for the historical MLP patch comparison."""
from __future__ import annotations

import gc
import inspect
import os
from pathlib import Path
from typing import Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import logging as hf_logging
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from transformer_lens import HookedTransformer
import transformer_lens.loading_from_pretrained as tl_loading


def patch_qwen3_rope_theta() -> None:
    """Backfill `rope_theta` only for older Qwen3 configs that don't expose it."""
    if "rope_theta" in inspect.signature(Qwen3Config.__init__).parameters:
        return

    def _get_rope_theta(self) -> float:
        stored = getattr(self, "_acdc_rope_theta", None)
        if stored is not None:
            return stored
        return (
            (getattr(self, "rope_scaling", None) or {}).get("rope_theta")
            or (getattr(self, "rope_parameters", None) or {}).get("rope_theta")
            or 1_000_000
        )

    def _set_rope_theta(self, value: float) -> None:
        self._acdc_rope_theta = value

    Qwen3Config.rope_theta = property(_get_rope_theta, _set_rope_theta)

def load_hooked_qwen3(model_path: str, device: str, dtype: torch.dtype) -> Tuple[HookedTransformer, AutoTokenizer]:
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    hf_logging.set_verbosity_error()
    try:
        hf_logging.disable_progress_bar()
    except Exception:
        pass

    patch_qwen3_rope_theta()

    hf_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map="cpu",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.bos_token is None:
        tokenizer.bos_token = "<|endoftext|>"
    tokenizer.add_bos_token = True

    if Path(model_path).exists():
        cfg = tl_loading.get_pretrained_model_config(
            model_path,
            hf_cfg=hf_model.config.to_dict(),
            fold_ln=False,
            device=device,
            n_devices=1,
            dtype=dtype,
            trust_remote_code=True,
        )
        state_dict = tl_loading.get_pretrained_state_dict(
            model_path,
            cfg,
            hf_model=hf_model,
            dtype=dtype,
            trust_remote_code=True,
        )
        model = HookedTransformer(
            cfg,
            tokenizer,
            move_to_device=False,
            default_padding_side="right",
        )
        model.load_and_process_state_dict(
            state_dict,
            fold_ln=False,
            center_writing_weights=False,
            center_unembed=False,
        )
        model.move_model_modules_to_device()
    else:
        model = HookedTransformer.from_pretrained(
            model_path,
            hf_model=hf_model,
            tokenizer=tokenizer,
            device=device,
            dtype=dtype,
            fold_ln=False,
            center_writing_weights=False,
            center_unembed=False,
            trust_remote_code=True,
        )
    # `move_model_modules_to_device()` is not always sufficient in this environment;
    # force the final TransformerLens module onto the requested runtime device.
    model.to(device, print_details=False)
    # These workflows only read `hook_z` and `hook_mlp_out`; the heavier
    # attention-result and split-qkv hooks materially increase activation
    # memory for long prompts without improving the current analyses.
    model.set_use_attn_result(False)
    model.set_use_split_qkv_input(False)
    model.set_use_hook_mlp_in(False)
    model.eval()

    # Free HF model copy ASAP.
    del hf_model
    gc.collect()
    torch.cuda.empty_cache()

    return model, tokenizer
