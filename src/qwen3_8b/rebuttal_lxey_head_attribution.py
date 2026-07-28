#!/usr/bin/env python3
"""Head-level verb-to-prediction attribution for Reviewer LxEy.

For every Qwen3-8B attention head in L5--L20, this runner measures two
quantities on the held-out v2_1500 test pairs:

1. prediction-position attention from the final prompt token to the one-token
   clean/corrupt verb span; and
2. the head's *linear residual write* along the frozen L24 tool-call direction,
   ``<W_O z_{l,h,p}, mu_hat>``.

The product of the paired clean-minus-corrupt shifts is a transparent ranking
criterion for heads that both read the changed verb and write into the later
decision subspace.  The highest-ranked heads then receive a held-out causal
check: their prediction-position ``z`` output is patched from the clean run
into the corrupt run, and we measure the L24-coordinate and final tool-call
changes.  This does not claim a complete circuit; it identifies and tests the
direct head-level bridge evaluated in this release.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch
from tqdm.auto import tqdm

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from artifact_paths import ARTIFACT_ROOT, QWEN3_8B_PATH  # noqa: E402
from toolcall_circuit.single_sample import load_hooked_qwen3  # noqa: E402
from qwen3_8b.rebuttal_lxey_other_concerns import (  # noqa: E402
    DEFAULT_OUTPUT_ROOT,
    PATCH_LAYER,
    TOOL_CALL,
    default_model_path,
    get_tool_token_id,
    last_token_metrics,
    manifest_hashes,
    make_capture_hook,
    write_csv,
    write_json,
    write_text,
)
from qwen3_8b.task_attention_path_analysis import (  # noqa: E402
    PairBatch,
    Sample,
    build_pair_batches,
    clear_cuda,
    load_samples,
    set_seed,
)


DEFAULT_LAYERS = tuple(range(5, 21))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=default_model_path())
    parser.add_argument("--test-root", type=Path, default=ARTIFACT_ROOT / "datasets" / "test")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--vector-bundle", type=Path, default=None)
    parser.add_argument("--layers", type=int, nargs="+", default=list(DEFAULT_LAYERS))
    parser.add_argument("--test-pairs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--causal-top-k", type=int, default=12)
    parser.add_argument("--causal-candidates-per-forward", type=int, default=4)
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=("all", "sweep", "validate"),
        default=["all"],
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def get_w_o_layer(model, layer: int) -> torch.Tensor:  # noqa: ANN001
    if hasattr(model, "W_O"):
        return model.W_O[layer]
    attn = model.blocks[layer].attn
    w_o = attn.W_O
    n_heads = int(model.cfg.n_heads)
    return w_o.view(n_heads, int(model.cfg.d_head), int(model.cfg.d_model))


def pattern_head_map(n_heads: int, pattern_head_count: int) -> torch.Tensor:
    """Map query-head indices to KV-head pattern indices under GQA."""

    if pattern_head_count == n_heads:
        return torch.arange(n_heads, dtype=torch.long)
    if pattern_head_count <= 0 or n_heads % pattern_head_count:
        raise ValueError(
            f"Cannot map {n_heads} query heads onto {pattern_head_count} attention-pattern heads"
        )
    group_size = n_heads // pattern_head_count
    return torch.arange(n_heads, dtype=torch.long) // group_size


def pair_diff_positions(batch: PairBatch, local_idx: int) -> torch.Tensor:
    diff = torch.nonzero(
        batch.clean_tokens_cpu[local_idx] != batch.corrupt_tokens_cpu[local_idx],
        as_tuple=False,
    ).flatten()
    if diff.numel() == 0:
        raise ValueError(f"Pair at local index {local_idx} has no token-level difference")
    return diff


def normalize_stages(raw: Iterable[str]) -> list[str]:
    values = list(raw)
    if "all" in values:
        return ["sweep", "validate"]
    return list(dict.fromkeys(values))


def vector_path(args: argparse.Namespace) -> Path:
    return args.vector_bundle or (args.output_root / "vector_fit" / "vector_bundle.pt")


def head_stage_root(output_root: Path) -> Path:
    return output_root / "head_attribution"


def cache_path(output_root: Path) -> Path:
    return head_stage_root(output_root) / "head_sweep_cache.pt"


def output_projection_vectors(model, layers: Sequence[int], unit_direction: torch.Tensor) -> torch.Tensor:
    """Return q-head x d_head vectors whose dot product with z is mu attribution."""

    vectors: list[torch.Tensor] = []
    target = unit_direction.to(device=model.W_U.device, dtype=torch.float32)
    for layer in layers:
        w_o = get_w_o_layer(model, int(layer)).to(device=model.W_U.device, dtype=torch.float32)
        vectors.append(torch.einsum("hde,e->hd", w_o, target).detach().cpu().float())
    return torch.stack(vectors, dim=0)


def collect_head_sweep(
    model,
    tokenizer,
    *,
    test_root: Path,
    test_pairs: int,
    layers: Sequence[int],
    batch_size: int,
    vector_bundle_path: Path,
    output_root: Path,
    resume: bool,
) -> tuple[dict[str, Any], list[dict[str, object]], list[tuple[int, int]]]:
    root = head_stage_root(output_root)
    cpath = cache_path(output_root)
    expected_hashes = manifest_hashes(test_root)
    vector_hash = sha256_file(vector_bundle_path)
    bundle = torch.load(vector_bundle_path, map_location="cpu", weights_only=False)
    if not isinstance(bundle, dict):
        raise TypeError(f"Expected mapping at {vector_bundle_path}")
    unit_direction = torch.as_tensor(bundle["unit_direction"], dtype=torch.float32).reshape(-1)
    if resume and cpath.exists():
        payload = torch.load(cpath, map_location="cpu", weights_only=False)
        if (
            payload.get("dataset_manifest_sha256") == expected_hashes
            and payload.get("vector_bundle_sha256") == vector_hash
            and list(payload.get("layers", [])) == [int(layer) for layer in layers]
            and int(payload.get("n_test_pairs", -1)) == test_pairs
        ):
            print(f"[resume] Reusing head-sweep cache: {cpath}", flush=True)
            summary_rows = summarize_head_sweep(payload)
            candidates = select_candidates(summary_rows, top_k=12)
            return payload, summary_rows, candidates
        raise RuntimeError(f"Refusing to reuse incompatible head sweep cache: {cpath}")

    samples = load_samples(test_root, model, tokenizer, max_pairs=test_pairs)
    n = len(samples)
    n_layers = len(layers)
    n_heads = int(model.cfg.n_heads)
    d_head = int(model.cfg.d_head)
    d_model = int(model.cfg.d_model)
    if unit_direction.numel() != d_model:
        raise ValueError("Direction and model dimensions differ")
    head_projection = output_projection_vectors(model, layers, unit_direction)

    attention = {
        side: torch.empty((n, n_layers, n_heads), dtype=torch.float32)
        for side in ("clean", "corrupt")
    }
    linear_write = {
        side: torch.empty((n, n_layers, n_heads), dtype=torch.float32)
        for side in ("clean", "corrupt")
    }
    # This is only ~40 MiB in bfloat16 and permits a fast causal follow-up
    # without recomputing clean outputs per candidate head.
    clean_z = torch.empty((n, n_layers, n_heads, d_head), dtype=torch.bfloat16)
    baseline_metrics = {
        side: {
            key: torch.empty((n,), dtype=torch.long if key in {"top1", "tool_rank"} else torch.float32)
            for key in ("tool_logit", "top1", "tool_prob", "tool_rank", "margin_vs_best_non_tool")
        }
        for side in ("clean", "corrupt")
    }
    hook_names: set[str] = set()
    for layer in layers:
        hook_names.add(f"blocks.{layer}.attn.hook_pattern")
        hook_names.add(f"blocks.{layer}.attn.hook_z")
    batches = build_pair_batches(samples, batch_size)
    tool_token_id = get_tool_token_id(tokenizer)
    observed_pattern_map: dict[int, list[int]] = {}

    progress = tqdm(batches, desc="L5-L20 linear head attribution", dynamic_ncols=True)
    for batch in progress:
        diff_positions = [pair_diff_positions(batch, local_idx) for local_idx in range(len(batch.indices))]
        for side in ("clean", "corrupt"):
            tokens_cpu = batch.clean_tokens_cpu if side == "clean" else batch.corrupt_tokens_cpu
            with torch.no_grad():
                logits, cache = model.run_with_cache(
                    tokens_cpu.to(model.W_U.device),
                    names_filter=lambda name: name in hook_names,
                )
            metric_values = last_token_metrics(logits, tool_token_id)
            for key, value in metric_values.items():
                baseline_metrics[side][key][batch.indices] = value
            for layer_idx, layer in enumerate(layers):
                z_name = f"blocks.{layer}.attn.hook_z"
                pattern_name = f"blocks.{layer}.attn.hook_pattern"
                z_last = cache[z_name][:, -1, :, :].detach().float()
                writes = torch.einsum(
                    "bhd,hd->bh",
                    z_last,
                    head_projection[layer_idx].to(device=z_last.device),
                ).detach().cpu()
                pattern = cache[pattern_name].detach().cpu().float()
                head_map = pattern_head_map(n_heads, int(pattern.shape[1]))
                observed_pattern_map.setdefault(int(layer), [int(value) for value in head_map.tolist()])
                for local_idx, sample_idx in enumerate(batch.indices):
                    diff = diff_positions[local_idx]
                    attention_value = pattern[local_idx, head_map, -1, :].index_select(1, diff).sum(dim=1)
                    attention[side][sample_idx, layer_idx] = attention_value
                    linear_write[side][sample_idx, layer_idx] = writes[local_idx]
                    if side == "clean":
                        clean_z[sample_idx, layer_idx] = z_last[local_idx].detach().cpu().to(torch.bfloat16)
            del cache, logits
            clear_cuda()
        progress.set_postfix(tok=batch.token_len)

    payload: dict[str, Any] = {
        "schema_version": 1,
        "dataset_manifest_sha256": expected_hashes,
        "vector_bundle_sha256": vector_hash,
        "vector_bundle": str(vector_bundle_path),
        "n_test_pairs": n,
        "sample_ids": [sample.sample_id for sample in samples],
        "layers": [int(layer) for layer in layers],
        "n_heads": n_heads,
        "d_head": d_head,
        "pattern_head_map": observed_pattern_map,
        "attention_clean": attention["clean"],
        "attention_corrupt": attention["corrupt"],
        "linear_write_clean": linear_write["clean"],
        "linear_write_corrupt": linear_write["corrupt"],
        "clean_z": clean_z,
        "baseline_clean": baseline_metrics["clean"],
        "baseline_corrupt": baseline_metrics["corrupt"],
    }
    ensure_dir(root)
    torch.save(payload, cpath)
    summary_rows = summarize_head_sweep(payload)
    candidates = select_candidates(summary_rows, top_k=12)
    return payload, summary_rows, candidates


def _ordinal_ranks(values: torch.Tensor) -> torch.Tensor:
    """Return deterministic ordinal ranks without requiring NumPy/SciPy.

    The system image currently contains an inconsistent NumPy 1.x/2.x module
    mix that is triggered by ``Tensor.numpy()``.  All statistics here are
    small, descriptive held-out summaries, so native Torch is both sufficient
    and avoids that unrelated environment failure.  Exact ties are rare for
    continuous activations; stable ordinal ranks make their handling explicit.
    """

    return torch.argsort(torch.argsort(values, stable=True), stable=True).to(torch.float64)


def safe_correlation(x: torch.Tensor, y: torch.Tensor) -> tuple[float, float]:
    """Pearson and ordinal-rank (Spearman-style) correlations, or NaNs."""

    x = x.detach().to(dtype=torch.float64, device="cpu").flatten()
    y = y.detach().to(dtype=torch.float64, device="cpu").flatten()
    if x.numel() < 3 or torch.allclose(x, x[0]) or torch.allclose(y, y[0]):
        return float("nan"), float("nan")

    def correlation(left: torch.Tensor, right: torch.Tensor) -> float:
        centered_left = left - left.mean()
        centered_right = right - right.mean()
        denominator = torch.linalg.vector_norm(centered_left) * torch.linalg.vector_norm(centered_right)
        if not bool(torch.isfinite(denominator)) or float(denominator.item()) == 0.0:
            return float("nan")
        return float((centered_left @ centered_right / denominator).item())

    return correlation(x, y), correlation(_ordinal_ranks(x), _ordinal_ranks(y))


def summarize_head_sweep(payload: dict[str, Any]) -> list[dict[str, object]]:
    layers = [int(layer) for layer in payload["layers"]]
    attention_clean = torch.as_tensor(payload["attention_clean"], dtype=torch.float64)
    attention_corrupt = torch.as_tensor(payload["attention_corrupt"], dtype=torch.float64)
    write_clean = torch.as_tensor(payload["linear_write_clean"], dtype=torch.float64)
    write_corrupt = torch.as_tensor(payload["linear_write_corrupt"], dtype=torch.float64)
    delta_attention = attention_clean - attention_corrupt
    delta_write = write_clean - write_corrupt
    rows: list[dict[str, object]] = []
    for layer_idx, layer in enumerate(layers):
        for head in range(delta_attention.shape[2]):
            attn = delta_attention[:, layer_idx, head]
            write = delta_write[:, layer_idx, head]
            pearson_r, spearman_r = safe_correlation(attn, write)
            mean_attn = float(attn.mean().item())
            mean_write = float(write.mean().item())
            # The score is deliberately descriptive and sign-agnostic: a
            # useful transfer head can either raise or lower attention/output
            # on the clean side, but it must do both reliably.
            transport_score = abs(mean_attn) * abs(mean_write) * (abs(pearson_r) if math.isfinite(pearson_r) else 0.0)
            rows.append(
                {
                    "layer": layer,
                    "head": head,
                    "mean_clean_attention_to_verb": float(attention_clean[:, layer_idx, head].mean().item()),
                    "mean_corrupt_attention_to_verb": float(attention_corrupt[:, layer_idx, head].mean().item()),
                    "delta_attention_clean_minus_corrupt": mean_attn,
                    "mean_clean_mu_linear_write": float(write_clean[:, layer_idx, head].mean().item()),
                    "mean_corrupt_mu_linear_write": float(write_corrupt[:, layer_idx, head].mean().item()),
                    "delta_mu_linear_write_clean_minus_corrupt": mean_write,
                    "pearson_pairwise_delta_attention_write": pearson_r,
                    "pearson_p": "not_computed",
                    "spearman_pairwise_delta_attention_write": spearman_r,
                    "spearman_p": "not_computed",
                    "linear_transport_score": transport_score,
                }
            )
    rows.sort(key=lambda row: float(row["linear_transport_score"]), reverse=True)
    for rank, row in enumerate(rows, start=1):
        row["linear_transport_rank"] = rank
    return rows


def select_candidates(rows: Sequence[dict[str, object]], *, top_k: int) -> list[tuple[int, int]]:
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    selected = [(int(row["layer"]), int(row["head"])) for row in rows[:top_k]]
    if not selected:
        raise RuntimeError("No heads available for causal validation")
    return selected


def make_chunk_patch_hook(
    *,
    layer: int,
    layer_to_candidate_indices: dict[int, list[int]],
    candidates: Sequence[tuple[int, int]],
    layer_to_index: dict[int, int],
    clean_z: torch.Tensor,
    sample_indices: Sequence[int],
    original_batch_size: int,
):
    candidate_indices = list(layer_to_candidate_indices.get(layer, []))

    def hook_fn(value: torch.Tensor, hook):  # noqa: ANN001
        out = value.clone()
        for candidate_idx in candidate_indices:
            _candidate_layer, head = candidates[candidate_idx]
            start = candidate_idx * original_batch_size
            stop = start + original_batch_size
            source = clean_z[list(sample_indices), layer_to_index[layer], head].to(device=value.device, dtype=value.dtype)
            out[start:stop, -1, head, :] = source
        return out

    return hook_fn


def validate_candidates(
    model,
    tokenizer,
    *,
    test_root: Path,
    test_pairs: int,
    payload: dict[str, Any],
    candidates: Sequence[tuple[int, int]],
    vector_bundle_path: Path,
    batch_size: int,
    candidates_per_forward: int,
    output_root: Path,
) -> list[dict[str, object]]:
    if candidates_per_forward <= 0:
        raise ValueError("--causal-candidates-per-forward must be positive")
    root = head_stage_root(output_root)
    samples = load_samples(test_root, model, tokenizer, max_pairs=test_pairs)
    expected_ids = [sample.sample_id for sample in samples]
    if list(payload["sample_ids"]) != expected_ids:
        raise RuntimeError("Head-sweep cache sample order does not match the active test manifest")
    bundle = torch.load(vector_bundle_path, map_location="cpu", weights_only=False)
    mu_delta = torch.as_tensor(bundle["mean_diff"], dtype=torch.float32).reshape(-1)
    mean_corrupt = torch.as_tensor(bundle["mean_corrupt"], dtype=torch.float32).reshape(-1)
    norm_sq = float(mu_delta.dot(mu_delta).item())
    if norm_sq <= 0:
        raise ValueError("Invalid zero-norm mean difference")

    layers = [int(layer) for layer in payload["layers"]]
    layer_to_index = {layer: idx for idx, layer in enumerate(layers)}
    clean_z = torch.as_tensor(payload["clean_z"])
    baseline = payload["baseline_corrupt"]
    baseline_tool_logit = torch.as_tensor(baseline["tool_logit"], dtype=torch.float32)
    baseline_top1 = torch.as_tensor(baseline["top1"], dtype=torch.long)
    tool_token_id = get_tool_token_id(tokenizer)
    n = len(samples)
    n_candidates = len(candidates)
    patched_tool_logit = torch.empty((n_candidates, n), dtype=torch.float32)
    patched_top1 = torch.empty((n_candidates, n), dtype=torch.long)
    patched_coordinate = torch.empty((n_candidates, n), dtype=torch.float32)
    baseline_resid_coordinate: torch.Tensor | None = None

    batches = build_pair_batches(samples, batch_size)
    for start_candidate in range(0, n_candidates, candidates_per_forward):
        chunk = list(candidates[start_candidate : start_candidate + candidates_per_forward])
        chunk_size = len(chunk)
        chunk_layers: dict[int, list[int]] = defaultdict(list)
        for local_idx, (layer, _head) in enumerate(chunk):
            chunk_layers[layer].append(local_idx)
        progress = tqdm(
            batches,
            desc=f"Causal head-z patch {start_candidate + 1}-{start_candidate + chunk_size}",
            dynamic_ncols=True,
        )
        for batch in progress:
            original_size = len(batch.indices)
            expanded_tokens = batch.corrupt_tokens_cpu.to(model.W_U.device).repeat((chunk_size, 1))
            capture: dict[str, torch.Tensor] = {}
            hooks: list[tuple[str, Any]] = []
            for layer in sorted(chunk_layers):
                hooks.append(
                    (
                        f"blocks.{layer}.attn.hook_z",
                        make_chunk_patch_hook(
                            layer=layer,
                            layer_to_candidate_indices=chunk_layers,
                            candidates=chunk,
                            layer_to_index=layer_to_index,
                            clean_z=clean_z,
                            sample_indices=batch.indices,
                            original_batch_size=original_size,
                        ),
                    )
                )
            hooks.append((f"blocks.{PATCH_LAYER}.hook_resid_pre", make_capture_hook(capture, "l24")))
            with torch.no_grad():
                logits = model.run_with_hooks(expanded_tokens, fwd_hooks=hooks)
            metrics = last_token_metrics(logits, tool_token_id)
            l24 = capture["l24"]
            if l24.shape[0] != chunk_size * original_size:
                raise RuntimeError("Unexpected causal validation capture shape")
            coord = ((l24 - mean_corrupt.unsqueeze(0)) @ mu_delta) / norm_sq
            if baseline_resid_coordinate is None:
                # A no-patch forward is already available from the sweep cache
                # only as logits, so capture it once per bucket without adding a
                # material extra model pass: the first expansion's unmodified
                # L24 value is reconstructed below in a dedicated hook pass.
                baseline_resid_coordinate = torch.full((n,), float("nan"), dtype=torch.float32)
            for local_idx in range(chunk_size):
                global_idx = start_candidate + local_idx
                offset = local_idx * original_size
                target = batch.indices
                patched_tool_logit[global_idx, target] = metrics["tool_logit"][offset : offset + original_size]
                patched_top1[global_idx, target] = metrics["top1"][offset : offset + original_size]
                patched_coordinate[global_idx, target] = coord[offset : offset + original_size]
            clear_cuda()
            progress.set_postfix(tok=batch.token_len)

    # Capture the corrupt L24 coordinate once. This is separate from the
    # selected head patches, so the resulting delta is interpretable at the
    # decision site rather than as a comparison between candidate heads.
    baseline_l24 = torch.empty((n, int(model.cfg.d_model)), dtype=torch.float32)
    progress = tqdm(batches, desc="Baseline corrupt L24 for head validation", dynamic_ncols=True)
    for batch in progress:
        capture: dict[str, torch.Tensor] = {}
        with torch.no_grad():
            _ = model.run_with_hooks(
                batch.corrupt_tokens_cpu.to(model.W_U.device),
                fwd_hooks=[(f"blocks.{PATCH_LAYER}.hook_resid_pre", make_capture_hook(capture, "l24"))],
            )
        baseline_l24[batch.indices] = capture["l24"]
        clear_cuda()
    baseline_coord = ((baseline_l24 - mean_corrupt.unsqueeze(0)) @ mu_delta) / norm_sq

    rows: list[dict[str, object]] = []
    for idx, (layer, head) in enumerate(candidates):
        is_tool = patched_top1[idx] == tool_token_id
        strict_recovery = (baseline_top1 != tool_token_id) & is_tool
        delta_logit = patched_tool_logit[idx] - baseline_tool_logit
        delta_coord = patched_coordinate[idx] - baseline_coord
        rows.append(
            {
                "layer": int(layer),
                "head": int(head),
                "n_test_pairs": n,
                "patched_tool_call_top1_rate": float(is_tool.float().mean().item()),
                "strict_recovery_rate": float(strict_recovery.float().mean().item()),
                "mean_tool_logit_delta_vs_corrupt": float(delta_logit.mean().item()),
                "mean_l24_mu_coordinate_delta_vs_corrupt": float(delta_coord.mean().item()),
                "median_l24_mu_coordinate_delta_vs_corrupt": float(delta_coord.median().item()),
            }
        )
    rows.sort(key=lambda row: abs(float(row["mean_l24_mu_coordinate_delta_vs_corrupt"])), reverse=True)
    write_csv(root / "causal_head_z_patch_summary.csv", rows)
    return rows


def write_reports(
    payload: dict[str, Any],
    summary_rows: Sequence[dict[str, object]],
    candidates: Sequence[tuple[int, int]],
    causal_rows: Sequence[dict[str, object]] | None,
    *,
    output_root: Path,
) -> None:
    root = head_stage_root(output_root)
    write_csv(root / "head_linear_attribution_summary.csv", list(summary_rows))
    selected = set(candidates)
    attention_clean = torch.as_tensor(payload["attention_clean"], dtype=torch.float32)
    attention_corrupt = torch.as_tensor(payload["attention_corrupt"], dtype=torch.float32)
    write_clean = torch.as_tensor(payload["linear_write_clean"], dtype=torch.float32)
    write_corrupt = torch.as_tensor(payload["linear_write_corrupt"], dtype=torch.float32)
    sample_ids = [str(value) for value in payload["sample_ids"]]
    layers = [int(value) for value in payload["layers"]]
    layer_to_index = {layer: idx for idx, layer in enumerate(layers)}
    per_sample: list[dict[str, object]] = []
    for layer, head in candidates:
        li = layer_to_index[layer]
        for sample_idx, sample_id in enumerate(sample_ids):
            per_sample.append(
                {
                    "sample_id": sample_id,
                    "layer": layer,
                    "head": head,
                    "clean_attention_to_verb": float(attention_clean[sample_idx, li, head].item()),
                    "corrupt_attention_to_verb": float(attention_corrupt[sample_idx, li, head].item()),
                    "delta_attention": float((attention_clean[sample_idx, li, head] - attention_corrupt[sample_idx, li, head]).item()),
                    "clean_mu_linear_write": float(write_clean[sample_idx, li, head].item()),
                    "corrupt_mu_linear_write": float(write_corrupt[sample_idx, li, head].item()),
                    "delta_mu_linear_write": float((write_clean[sample_idx, li, head] - write_corrupt[sample_idx, li, head]).item()),
                }
            )
    write_csv(root / "top_head_per_sample_attribution.csv", per_sample)

    lookup = {(int(row["layer"]), int(row["head"])): row for row in summary_rows}
    lines = [
        "# L5--L20 verb-to-prediction head attribution",
        "",
        "For each head, `attention` is the final-prompt-position mass on the changed verb token(s). `mu linear write` is `<W_O z, mu_hat>` at the same prediction position. Both are evaluated on the held-out v2_1500 test pairs.",
        "",
        "## Top linear transport heads",
        "",
        "| rank | head | Δ attention (clean-corrupt) | Δ μ write | paired Pearson r | transport score |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for row in list(summary_rows)[:12]:
        lines.append(
            f"| {int(row['linear_transport_rank'])} | L{int(row['layer'])}H{int(row['head'])} | "
            f"{float(row['delta_attention_clean_minus_corrupt']):+.5f} | "
            f"{float(row['delta_mu_linear_write_clean_minus_corrupt']):+.5f} | "
            f"{float(row['pearson_pairwise_delta_attention_write']):+.3f} | "
            f"{float(row['linear_transport_score']):.6g} |"
        )
    if causal_rows is not None:
        lines.extend(
            [
                "",
                "## Causal check: clean head-z patched into corrupt run",
                "",
                "| head | Δ L24 μ coordinate | Δ tool logit | strict recovery |",
                "|---|---:|---:|---:|",
            ]
        )
        for row in causal_rows:
            lines.append(
                f"| L{int(row['layer'])}H{int(row['head'])} | "
                f"{float(row['mean_l24_mu_coordinate_delta_vs_corrupt']):+.4f} | "
                f"{float(row['mean_tool_logit_delta_vs_corrupt']):+.4f} | "
                f"{float(row['strict_recovery_rate']):.2%} |"
            )
    lines.extend(
        [
            "",
            "The causal check validates head outputs at the prediction position. It is deliberately narrower than a full edge-patching proof, so the rebuttal should describe it as a tested bridge rather than a complete upstream circuit.",
        ]
    )
    write_text(root / "summary.md", "\n".join(lines))
    write_json(
        root / "metadata.json",
        {
            "n_test_pairs": int(payload["n_test_pairs"]),
            "layers": layers,
            "n_heads": int(payload["n_heads"]),
            "vector_bundle": payload["vector_bundle"],
            "vector_bundle_sha256": payload["vector_bundle_sha256"],
            "dataset_manifest_sha256": payload["dataset_manifest_sha256"],
            "selection": "abs(mean clean-corrupt verb attention) * abs(mean clean-corrupt mu linear write) * abs(pairwise Pearson r)",
            "selected_heads": [f"L{layer}H{head}" for layer, head in candidates],
            "causal_validation_present": causal_rows is not None,
        },
    )


def main() -> None:
    args = parse_args()
    stages = normalize_stages(args.stages)
    layers = tuple(sorted(set(int(layer) for layer in args.layers)))
    if not layers or min(layers) < 0 or max(layers) >= 36:
        raise ValueError(f"Invalid layer list: {layers}")
    if args.test_pairs <= 0 or args.batch_size <= 0:
        raise ValueError("--test-pairs and --batch-size must be positive")
    if not args.model_path.is_dir():
        raise FileNotFoundError(f"Model directory not found: {args.model_path}")
    bundle_path = vector_path(args)
    if not bundle_path.is_file():
        raise FileNotFoundError(
            f"Missing coherent vector bundle: {bundle_path}. Run rebuttal_lxey_other_concerns.py first."
        )
    ensure_dir(head_stage_root(args.output_root))
    set_seed(args.seed)
    model, tokenizer = load_hooked_qwen3(str(args.model_path), device=args.device, dtype=torch.bfloat16)
    model.eval()

    payload, summary_rows, default_candidates = collect_head_sweep(
        model,
        tokenizer,
        test_root=args.test_root,
        test_pairs=args.test_pairs,
        layers=layers,
        batch_size=args.batch_size,
        vector_bundle_path=bundle_path,
        output_root=args.output_root,
        resume=args.resume,
    )
    candidates = select_candidates(summary_rows, top_k=args.causal_top_k)
    causal_rows: list[dict[str, object]] | None = None
    causal_path = head_stage_root(args.output_root) / "causal_head_z_patch_summary.csv"
    if "validate" in stages:
        if args.resume and causal_path.exists():
            with causal_path.open(encoding="utf-8", newline="") as handle:
                causal_rows = [dict(row) for row in csv.DictReader(handle)]
            print(f"[resume] Reusing causal head validation: {causal_path}", flush=True)
        else:
            causal_rows = validate_candidates(
                model,
                tokenizer,
                test_root=args.test_root,
                test_pairs=args.test_pairs,
                payload=payload,
                candidates=candidates,
                vector_bundle_path=bundle_path,
                batch_size=args.batch_size,
                candidates_per_forward=args.causal_candidates_per_forward,
                output_root=args.output_root,
            )
    write_reports(payload, summary_rows, candidates, causal_rows, output_root=args.output_root)
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
