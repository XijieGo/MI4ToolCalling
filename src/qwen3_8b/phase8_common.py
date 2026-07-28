#!/usr/bin/env python3
from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from phase4_reviewer_strengthening import L24_LAYER, SEED, TOOL_CALL_STR, tool_stats
from task_attention_path_analysis import (
    MODEL_PATH,
    build_pair_batches,
    clear_cuda,
    ensure_dir,
    load_samples,
    set_seed,
    write_csv,
    write_text,
)


from toolcall_circuit.single_sample import load_hooked_qwen3  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DISCOVERY_DATASET_ROOT = PROJECT_ROOT / "datasets" / "train"
EVAL_DATASET_ROOT = PROJECT_ROOT / "datasets" / "test"
PHASE7_ROOT = PROJECT_ROOT / "results" / "8b_main" / "phase7_l24_directionality"
PHASE8_ROOT = PROJECT_ROOT / "results" / "8b_main" / "phase8_upstream"

DEFAULT_PC_BUNDLE = PHASE7_ROOT / "exp_a_fixed_direction" / "pc_bundle.pt"
DEFAULT_HELDOUT_CACHE = PHASE7_ROOT / "exp_a_fixed_direction" / "heldout_eval_cache.pt"

EXP_A_ROOT = PHASE8_ROOT / "exp_a_pck_recovery"
EXP_B_ROOT = PHASE8_ROOT / "exp_b_gate_trajectory"
EXP_C_ROOT = PHASE8_ROOT / "exp_c_gate_components"
EXP_D_ROOT = PHASE8_ROOT / "exp_d_gate_behavior"
EXP_E_ROOT = PHASE8_ROOT / "exp_e_gate_patching"


def manifest_pair_count(dataset_root: Path) -> int:
    manifest_path = dataset_root / "clean" / "manifest.jsonl"
    with manifest_path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 200,
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.6,
        }
    )


def load_model_and_tokenizer(*, model_path: Path = MODEL_PATH, device: str = "cuda"):
    return load_hooked_qwen3(str(model_path), device=device, dtype=torch.bfloat16)


def get_tool_token_id(tokenizer) -> int:
    tool_token_ids = tokenizer.encode(TOOL_CALL_STR, add_special_tokens=False)
    if len(tool_token_ids) != 1:
        raise ValueError(f"{TOOL_CALL_STR!r} maps to unexpected token ids: {tool_token_ids}")
    return int(tool_token_ids[0])


def load_gate_bundle(path: Path) -> dict[str, torch.Tensor]:
    bundle = torch.load(path, map_location="cpu", weights_only=False)
    if "components" not in bundle:
        raise KeyError(f"Missing components in {path}")
    return bundle


def load_gate_direction(path: Path) -> torch.Tensor:
    bundle = load_gate_bundle(path)
    return bundle["components"][0].float()


def percent(value: float) -> str:
    return f"{value:.2%}"


def safe_mean(values: Sequence[float]) -> float:
    if not values:
        return float("nan")
    return float(np.asarray(values, dtype=np.float64).mean())


def safe_corrcoef(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or y.size < 2:
        return float("nan")
    if np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def projection(scores: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
    return torch.mv(scores.float(), direction.float())


def cosine_to_direction(scores: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
    direction_batch = direction.float().unsqueeze(0).expand_as(scores)
    return F.cosine_similarity(scores.float(), direction_batch, dim=-1)


def make_last_token_vector_capture(capture: dict[str, torch.Tensor], key: str):
    def hook_fn(value: torch.Tensor, hook):  # noqa: ANN001
        capture[key] = value[:, -1, :].detach().cpu().float()
        return value

    return hook_fn


def make_last_token_vector_replace(source_cpu: torch.Tensor):
    def hook_fn(value: torch.Tensor, hook):  # noqa: ANN001
        out = value.clone()
        source = source_cpu.to(device=value.device, dtype=value.dtype)
        out[:, -1, :] = source
        return out

    return hook_fn


def collect_pair_last_token_activations(
    model,
    pair_batches,
    *,
    hook_names: Sequence[str],
    side: str,
    desc: str,
) -> dict[str, torch.Tensor]:
    if side not in {"clean", "corrupt"}:
        raise ValueError(f"Unsupported side: {side}")

    n_samples = sum(len(batch.indices) for batch in pair_batches)
    d_model = int(model.cfg.d_model)
    outputs = {name: torch.empty((n_samples, d_model), dtype=torch.float32) for name in hook_names}

    progress = tqdm(pair_batches, desc=desc, dynamic_ncols=True)
    for batch in progress:
        capture: dict[str, torch.Tensor] = {}
        hooks = [(name, make_last_token_vector_capture(capture, name)) for name in hook_names]
        tokens_cpu = batch.clean_tokens_cpu if side == "clean" else batch.corrupt_tokens_cpu
        with torch.no_grad():
            _ = model.run_with_hooks(tokens_cpu.to(model.W_U.device), fwd_hooks=hooks)
        for name in hook_names:
            outputs[name][batch.indices] = capture[name]
        clear_cuda()
    return outputs


def add_gate_marker(ax, *, gate_layer: int = L24_LAYER) -> None:
    ax.axvline(gate_layer, color="black", linestyle="--", linewidth=1.0, alpha=0.45)


def find_first_threshold_layer(values: Sequence[float], *, threshold_ratio: float = 0.1) -> int | None:
    if not values:
        return None
    final = float(values[-1])
    if math.isclose(final, 0.0):
        return None
    threshold = threshold_ratio * final
    for idx, value in enumerate(values):
        if value >= threshold:
            return idx
    return None


def largest_jump(values: Sequence[float]) -> tuple[int, float] | None:
    if len(values) < 2:
        return None
    deltas = [float(values[idx + 1] - values[idx]) for idx in range(len(values) - 1)]
    layer = int(np.argmax(deltas))
    return layer, float(deltas[layer])


def stage_abs_mass(values: Sequence[float], stage: range) -> float:
    return float(sum(abs(float(values[idx])) for idx in stage if idx < len(values)))


__all__ = [
    "DEFAULT_HELDOUT_CACHE",
    "DEFAULT_PC_BUNDLE",
    "DISCOVERY_DATASET_ROOT",
    "EVAL_DATASET_ROOT",
    "EXP_A_ROOT",
    "EXP_B_ROOT",
    "EXP_C_ROOT",
    "EXP_D_ROOT",
    "EXP_E_ROOT",
    "L24_LAYER",
    "MODEL_PATH",
    "PHASE7_ROOT",
    "PROJECT_ROOT",
    "SEED",
    "TOOL_CALL_STR",
    "add_gate_marker",
    "clear_cuda",
    "collect_pair_last_token_activations",
    "configure_matplotlib",
    "cosine_to_direction",
    "ensure_dir",
    "find_first_threshold_layer",
    "get_tool_token_id",
    "largest_jump",
    "load_gate_bundle",
    "load_gate_direction",
    "load_model_and_tokenizer",
    "load_samples",
    "make_last_token_vector_capture",
    "make_last_token_vector_replace",
    "manifest_pair_count",
    "percent",
    "projection",
    "safe_corrcoef",
    "safe_mean",
    "set_seed",
    "stage_abs_mass",
    "tool_stats",
    "write_csv",
    "write_text",
    "build_pair_batches",
]
