#!/usr/bin/env python3
"""Test the three LxEy candidate bridge heads on v4 domains D3--D5.

The heads were identified on the code-domain v2_1500 experiment:
``L19H31``, ``L20H29``, and ``L20H14``.  This runner keeps those identities
fixed and evaluates their role on the frozen held-out v4 web-verification,
SQL, and email pairs.  It deliberately does *not* reuse the code-domain
``mu_delta`` direction: cross-domain evidence is instead reported in the
common output space--attention to the changed verb, raw unembedding write to
``<tool_call>``, and bidirectional causal head-z patching.

For each head (and the three-head set), the script measures:

* final-position attention mass to the sole changed verb token;
* a DLA-style raw ``<W_O z, W_U[:, tool_call]>`` linear write; and
* clean-z -> corrupt recovery and corrupt-z -> clean ablation at the final
  prediction position.

The v4 membership was filtered before this analysis so every held-out clean
prompt is tool-call top-1 and every corrupt prompt is non-tool top-1 for the
Qwen3 scales.  The script nevertheless records fresh baselines rather than
assuming that invariant.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch
from tqdm.auto import tqdm


SRC_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SRC_ROOT.parent
for _path in (SRC_ROOT, SRC_ROOT / "shared", SRC_ROOT / "multidomain"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from multiscale_common import build_pair_batches  # noqa: E402
from scaffold_ablation_common import FixedPair, load_fixed_pairs, validate_fixed_pairs  # noqa: E402
from toolcall_circuit.single_sample import load_hooked_qwen3  # noqa: E402


TOOL_CALL = "<tool_call>"
DOMAINS = ("D3", "D4", "D5")
HEADS = ((19, 31), (20, 29), (20, 14))
DEFAULT_DATA_ROOT = PROJECT_ROOT / "datasets" / "v4_multidomain_balanced"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "results" / "runs" / "rebuttal_lxey_crossdomain_heads_v4_20260727"


def default_model_path() -> Path:
    candidates = (
        Path(os.environ.get("QWEN3_8B_PATH", PROJECT_ROOT / "external" / "models" / "Qwen3-8B")).expanduser(),
    )
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    return candidates[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=default_model_path())
    parser.add_argument("--dataset-view-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--domains", nargs="+", choices=DOMAINS, default=list(DOMAINS))
    parser.add_argument("--split", choices=("train", "test"), default="test")
    parser.add_argument("--pairs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--variants-per-forward", type=int, default=2)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    ensure_dir(path.parent)
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


def write_json(path: Path, payload: dict[str, object]) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    ensure_dir(path.parent)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def clear_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def head_name(head: tuple[int, int]) -> str:
    return f"L{head[0]}H{head[1]}"


def get_tool_token_id(tokenizer) -> int:  # noqa: ANN001
    token_ids = tokenizer.encode(TOOL_CALL, add_special_tokens=False)
    if len(token_ids) != 1:
        raise ValueError(f"{TOOL_CALL!r} is not a single tokenizer token: {token_ids}")
    return int(token_ids[0])


def last_token_metrics(logits: torch.Tensor, tool_token_id: int) -> dict[str, torch.Tensor]:
    final_logits = logits[:, -1, :].float()
    tool_logit = final_logits[:, tool_token_id]
    best_non_tool = final_logits.clone()
    best_non_tool[:, tool_token_id] = -torch.inf
    return {
        "tool_logit": tool_logit.detach().cpu(),
        "tool_prob": torch.softmax(final_logits, dim=-1)[:, tool_token_id].detach().cpu(),
        "top1": final_logits.argmax(dim=-1).detach().cpu(),
        "tool_rank": ((final_logits > tool_logit.unsqueeze(-1)).sum(dim=-1) + 1).detach().cpu(),
        "margin_vs_best_non_tool": (tool_logit - best_non_tool.max(dim=-1).values).detach().cpu(),
    }


def get_w_o_layer(model, layer: int) -> torch.Tensor:  # noqa: ANN001
    if hasattr(model, "W_O"):
        return model.W_O[layer]
    value = model.blocks[layer].attn.W_O
    return value.view(int(model.cfg.n_heads), int(model.cfg.d_head), int(model.cfg.d_model))


def pattern_head_map(n_heads: int, pattern_head_count: int) -> torch.Tensor:
    if pattern_head_count == n_heads:
        return torch.arange(n_heads, dtype=torch.long)
    if pattern_head_count <= 0 or n_heads % pattern_head_count:
        raise ValueError(f"Cannot map {n_heads} Q heads to {pattern_head_count} pattern heads")
    return torch.arange(n_heads, dtype=torch.long) // (n_heads // pattern_head_count)


def variants() -> list[tuple[str, tuple[tuple[int, int], ...]]]:
    return [(head_name(head), (head,)) for head in HEADS] + [("all_3", HEADS)]


METRIC_KEYS = ("tool_logit", "tool_prob", "top1", "tool_rank", "margin_vs_best_non_tool")


def empty_metric_store(n: int) -> dict[str, torch.Tensor]:
    return {
        key: torch.empty((n,), dtype=torch.long if key in {"top1", "tool_rank"} else torch.float32)
        for key in METRIC_KEYS
    }


def differing_positions(batch, local_idx: int) -> torch.Tensor:  # noqa: ANN001
    changed = torch.nonzero(
        batch.clean_tokens_cpu[local_idx] != batch.corrupt_tokens_cpu[local_idx],
        as_tuple=False,
    ).flatten()
    if changed.numel() != 1:
        raise ValueError(f"Expected one changed token, found {changed.numel()}")
    return changed


def source_capture(
    model,
    pairs: Sequence[FixedPair],
    *,
    batch_size: int,
    tool_token_id: int,
) -> dict[str, Any]:
    """Capture selected heads and fresh clean/corrupt metrics once per domain."""

    n = len(pairs)
    d_head = int(model.cfg.d_head)
    layers = sorted({layer for layer, _head in HEADS})
    projection: dict[tuple[int, int], torch.Tensor] = {}
    wu_tool = model.W_U[:, tool_token_id].detach().to(dtype=torch.float32)
    for layer, head in HEADS:
        w_o = get_w_o_layer(model, layer)[head].detach().to(dtype=torch.float32)
        projection[(layer, head)] = torch.einsum("de,e->d", w_o, wu_tool).cpu()

    attention = {side: {head: torch.empty((n,), dtype=torch.float32) for head in HEADS} for side in ("clean", "corrupt")}
    raw_write = {side: {head: torch.empty((n,), dtype=torch.float32) for head in HEADS} for side in ("clean", "corrupt")}
    head_z = {
        side: {head: torch.empty((n, d_head), dtype=torch.bfloat16) for head in HEADS}
        for side in ("clean", "corrupt")
    }
    metrics = {side: empty_metric_store(n) for side in ("clean", "corrupt")}
    hook_names = {
        f"blocks.{layer}.attn.{kind}"
        for layer in layers
        for kind in ("hook_pattern", "hook_z")
    }
    batches = build_pair_batches(pairs, batch_size)
    observed_pattern_map: dict[int, list[int]] = {}
    progress = tqdm(batches, desc="Capture cross-domain selected heads", dynamic_ncols=True)
    for batch in progress:
        changed_positions = [differing_positions(batch, local_idx) for local_idx in range(len(batch.indices))]
        for side in ("clean", "corrupt"):
            tokens = batch.clean_tokens_cpu if side == "clean" else batch.corrupt_tokens_cpu
            with torch.no_grad():
                logits, cache = model.run_with_cache(
                    tokens.to(model.W_U.device),
                    names_filter=lambda name: name in hook_names,
                )
            values = last_token_metrics(logits, tool_token_id)
            for key, value in values.items():
                metrics[side][key][batch.indices] = value
            z_by_layer = {
                layer: cache[f"blocks.{layer}.attn.hook_z"][:, -1].detach().float()
                for layer in layers
            }
            pattern_by_layer = {
                layer: cache[f"blocks.{layer}.attn.hook_pattern"].detach().cpu().float()
                for layer in layers
            }
            for layer in layers:
                pattern = pattern_by_layer[layer]
                mapped_heads = pattern_head_map(int(model.cfg.n_heads), int(pattern.shape[1]))
                observed_pattern_map.setdefault(layer, [int(value) for value in mapped_heads.tolist()])
                for head_tuple in (head for head in HEADS if head[0] == layer):
                    _head_layer, head = head_tuple
                    z = z_by_layer[layer][:, head, :]
                    raw_write[side][head_tuple][batch.indices] = (
                        z @ projection[head_tuple].to(device=z.device)
                    ).detach().cpu()
                    head_z[side][head_tuple][batch.indices] = z.detach().cpu().to(torch.bfloat16)
                    pattern_head = int(mapped_heads[head].item())
                    for local_idx, pair_idx in enumerate(batch.indices):
                        attention[side][head_tuple][pair_idx] = pattern[local_idx, pattern_head, -1, changed_positions[local_idx]].sum()
            del cache, logits
            clear_cuda()
        progress.set_postfix(tok=batch.token_len)
    return {
        "attention": attention,
        "raw_write": raw_write,
        "head_z": head_z,
        "metrics": metrics,
        "pattern_head_map": observed_pattern_map,
        "batches": batches,
    }


def make_patch_hook(
    *,
    layer: int,
    chunk: Sequence[tuple[str, tuple[tuple[int, int], ...]]],
    source_z: dict[tuple[int, int], torch.Tensor],
    batch_indices: Sequence[int],
    original_batch_size: int,
):
    per_variant = [
        (variant_idx, [head for head in heads if head[0] == layer])
        for variant_idx, (_name, heads) in enumerate(chunk)
    ]

    def hook_fn(value: torch.Tensor, hook):  # noqa: ANN001
        patched = value.clone()
        for variant_idx, heads in per_variant:
            if not heads:
                continue
            start = variant_idx * original_batch_size
            stop = start + original_batch_size
            for head_tuple in heads:
                _head_layer, head = head_tuple
                source = source_z[head_tuple][list(batch_indices)].to(device=value.device, dtype=value.dtype)
                patched[start:stop, -1, head, :] = source
        return patched

    return hook_fn


def causal_patch(
    model,
    pairs: Sequence[FixedPair],
    *,
    batches: Sequence[Any],
    source_z: dict[tuple[int, int], torch.Tensor],
    base_side: str,
    variants_per_forward: int,
    tool_token_id: int,
) -> dict[str, dict[str, torch.Tensor]]:
    if base_side not in {"clean", "corrupt"}:
        raise ValueError(base_side)
    if variants_per_forward <= 0:
        raise ValueError("--variants-per-forward must be positive")
    n = len(pairs)
    result = {name: empty_metric_store(n) for name, _heads in variants()}
    all_variants = variants()
    for start in range(0, len(all_variants), variants_per_forward):
        chunk = all_variants[start : start + variants_per_forward]
        active_layers = sorted({layer for _name, heads in chunk for layer, _head in heads})
        progress = tqdm(
            batches,
            desc=f"{base_side} causal patch {start + 1}-{start + len(chunk)}",
            dynamic_ncols=True,
        )
        for batch in progress:
            original_size = len(batch.indices)
            base = batch.corrupt_tokens_cpu if base_side == "corrupt" else batch.clean_tokens_cpu
            expanded = base.to(model.W_U.device).repeat((len(chunk), 1))
            hooks = [
                (
                    f"blocks.{layer}.attn.hook_z",
                    make_patch_hook(
                        layer=layer,
                        chunk=chunk,
                        source_z=source_z,
                        batch_indices=batch.indices,
                        original_batch_size=original_size,
                    ),
                )
                for layer in active_layers
            ]
            with torch.no_grad():
                logits = model.run_with_hooks(expanded, fwd_hooks=hooks)
            values = last_token_metrics(logits, tool_token_id)
            for variant_idx, (name, _heads) in enumerate(chunk):
                offset = variant_idx * original_size
                for key, value in values.items():
                    result[name][key][batch.indices] = value[offset : offset + original_size]
            del logits
            clear_cuda()
            progress.set_postfix(tok=batch.token_len)
    return result


def mean(values: torch.Tensor) -> float:
    return float(values.detach().float().mean().item())


def rate(values: torch.Tensor) -> float:
    return mean(values.float())


def domain_rows(
    domain: str,
    pairs: Sequence[FixedPair],
    capture: dict[str, Any],
    rescue: dict[str, dict[str, torch.Tensor]],
    ablation: dict[str, dict[str, torch.Tensor]],
    *,
    tool_token_id: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    """Construct per-head observations, causal rows, and concise summaries."""

    observations: list[dict[str, object]] = []
    causal_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    baseline_clean = capture["metrics"]["clean"]
    baseline_corrupt = capture["metrics"]["corrupt"]
    for pair_idx, pair in enumerate(pairs):
        for head in HEADS:
            observations.append(
                {
                    "domain": domain,
                    "sample_id": pair.sample_id,
                    "clean_verb": pair.clean_candidate or "",
                    "corrupt_verb": pair.corrupt_candidate or "",
                    "prompt_tokens": pair.token_len,
                    "head": head_name(head),
                    "clean_attention_to_changed_verb": float(capture["attention"]["clean"][head][pair_idx].item()),
                    "corrupt_attention_to_changed_verb": float(capture["attention"]["corrupt"][head][pair_idx].item()),
                    "delta_attention_clean_minus_corrupt": float(
                        (capture["attention"]["clean"][head][pair_idx] - capture["attention"]["corrupt"][head][pair_idx]).item()
                    ),
                    "clean_raw_tool_unembedding_write": float(capture["raw_write"]["clean"][head][pair_idx].item()),
                    "corrupt_raw_tool_unembedding_write": float(capture["raw_write"]["corrupt"][head][pair_idx].item()),
                    "delta_raw_tool_unembedding_write": float(
                        (capture["raw_write"]["clean"][head][pair_idx] - capture["raw_write"]["corrupt"][head][pair_idx]).item()
                    ),
                }
            )
    for direction, patched, baseline in (
        ("rescue_clean_z_into_corrupt", rescue, baseline_corrupt),
        ("ablate_corrupt_z_into_clean", ablation, baseline_clean),
    ):
        for variant, _heads in variants():
            for pair_idx, pair in enumerate(pairs):
                baseline_is_tool = int(baseline["top1"][pair_idx].item()) == tool_token_id
                patched_is_tool = int(patched[variant]["top1"][pair_idx].item()) == tool_token_id
                strict_transition = (
                    int((not baseline_is_tool) and patched_is_tool)
                    if direction.startswith("rescue")
                    else int(baseline_is_tool and not patched_is_tool)
                )
                causal_rows.append(
                    {
                        "domain": domain,
                        "sample_id": pair.sample_id,
                        "direction": direction,
                        "variant": variant,
                        "baseline_tool_top1": int(baseline_is_tool),
                        "patched_tool_top1": int(patched_is_tool),
                        "strict_transition": strict_transition,
                        "baseline_tool_logit": float(baseline["tool_logit"][pair_idx].item()),
                        "patched_tool_logit": float(patched[variant]["tool_logit"][pair_idx].item()),
                        "tool_logit_delta": float((patched[variant]["tool_logit"][pair_idx] - baseline["tool_logit"][pair_idx]).item()),
                        "baseline_tool_rank": int(baseline["tool_rank"][pair_idx].item()),
                        "patched_tool_rank": int(patched[variant]["tool_rank"][pair_idx].item()),
                    }
                )
    for variant, variant_heads in variants():
        attention_delta = sum(
            capture["attention"]["clean"][head] - capture["attention"]["corrupt"][head]
            for head in variant_heads
        )
        raw_delta = sum(
            capture["raw_write"]["clean"][head] - capture["raw_write"]["corrupt"][head]
            for head in variant_heads
        )
        rescue_metrics = rescue[variant]
        ablation_metrics = ablation[variant]
        clean_is_tool = baseline_clean["top1"] == tool_token_id
        corrupt_is_tool = baseline_corrupt["top1"] == tool_token_id
        rescue_is_tool = rescue_metrics["top1"] == tool_token_id
        ablation_is_tool = ablation_metrics["top1"] == tool_token_id
        summary_rows.append(
            {
                "domain": domain,
                "variant": variant,
                "heads": "+".join(head_name(head) for head in variant_heads),
                "n_heads": len(variant_heads),
                "n_test_pairs": len(pairs),
                "baseline_clean_tool_top1_rate": rate(clean_is_tool),
                "baseline_corrupt_tool_top1_rate": rate(corrupt_is_tool),
                "sum_delta_attention_clean_minus_corrupt": mean(attention_delta),
                "sum_delta_raw_tool_unembedding_write": mean(raw_delta),
                "rescue_patched_tool_top1_rate": rate(rescue_is_tool),
                "rescue_strict_recovery_rate": rate((~corrupt_is_tool) & rescue_is_tool),
                "rescue_mean_tool_logit_delta": mean(rescue_metrics["tool_logit"] - baseline_corrupt["tool_logit"]),
                "ablation_patched_tool_top1_rate": rate(ablation_is_tool),
                "ablation_strict_loss_rate": rate(clean_is_tool & (~ablation_is_tool)),
                "ablation_mean_tool_logit_delta": mean(ablation_metrics["tool_logit"] - baseline_clean["tool_logit"]),
            }
        )
    return observations, causal_rows, summary_rows


def load_complete_summary(domain_root: Path) -> list[dict[str, object]]:
    with (domain_root / "summary.csv").open(encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def run_domain(model, tokenizer, args: argparse.Namespace, domain: str) -> list[dict[str, object]]:  # noqa: ANN001
    domain_root = args.output_root / domain
    complete_path = domain_root / "completion.json"
    if complete_path.exists():
        state = json.loads(complete_path.read_text(encoding="utf-8"))
        if state.get("status") == "complete":
            if not args.resume:
                raise FileExistsError(f"Completed domain output exists: {domain_root}; pass --resume to reuse it")
            print(f"[resume] Reusing completed {domain}: {domain_root}", flush=True)
            return load_complete_summary(domain_root)
    if domain_root.exists():
        raise FileExistsError(f"Refusing to mix with incomplete output directory: {domain_root}")
    ensure_dir(domain_root)
    dataset_root = args.dataset_view_root / domain
    pairs = load_fixed_pairs(model, dataset_root=dataset_root, split=args.split, max_pairs=args.pairs)
    validation = validate_fixed_pairs(pairs, expected_count=args.pairs, label=f"{domain}/{args.split}")
    capture = source_capture(
        model,
        pairs,
        batch_size=args.batch_size,
        tool_token_id=get_tool_token_id(tokenizer),
    )
    tool_token_id = get_tool_token_id(tokenizer)
    rescue = causal_patch(
        model,
        pairs,
        batches=capture["batches"],
        source_z=capture["head_z"]["clean"],
        base_side="corrupt",
        variants_per_forward=args.variants_per_forward,
        tool_token_id=tool_token_id,
    )
    ablation = causal_patch(
        model,
        pairs,
        batches=capture["batches"],
        source_z=capture["head_z"]["corrupt"],
        base_side="clean",
        variants_per_forward=args.variants_per_forward,
        tool_token_id=tool_token_id,
    )
    observations, causal_rows, summary_rows = domain_rows(
        domain,
        pairs,
        capture,
        rescue,
        ablation,
        tool_token_id=tool_token_id,
    )
    write_csv(domain_root / "head_observations_per_sample.csv", observations)
    write_csv(domain_root / "causal_patch_per_sample.csv", causal_rows)
    write_csv(domain_root / "summary.csv", summary_rows)
    selected_path = dataset_root / "selected_pairs.jsonl"
    write_json(
        domain_root / "metadata.json",
        {
            "domain": domain,
            "split": args.split,
            "n_pairs": len(pairs),
            "heads": [head_name(head) for head in HEADS],
            "variants": [name for name, _heads in variants()],
            "dataset_root": str(dataset_root.resolve()),
            "selected_pairs_sha256": sha256_file(selected_path) if selected_path.is_file() else None,
            "validation": validation,
            "pattern_head_map": capture["pattern_head_map"],
            "attention_definition": "Final-prediction-position mass on the sole changed verb token.",
            "raw_dla_definition": "dot(W_O z_head, W_U[:, <tool_call>]); a raw linear unembedding projection, not an exact final-layernorm logit decomposition.",
            "causal_definition": "Patch final-position head z from the opposite verb condition; rescue uses clean z -> corrupt and ablation uses corrupt z -> clean.",
        },
    )
    write_json(domain_root / "completion.json", {"status": "complete", "domain": domain, "n_test_pairs": len(pairs)})
    return summary_rows


def write_global_summary(output_root: Path, rows: Sequence[dict[str, object]]) -> None:
    write_csv(output_root / "crossdomain_head_role_summary.csv", list(rows))
    lines = [
        "# Fixed LxEy head roles across v4 domains",
        "",
        "The three heads were selected on the code-domain v2_1500 result and held fixed here. Every row uses the frozen held-out v4 test pairs; no D3/D4/D5 head selection or behavioral filtering occurs in this analysis.",
        "",
        "`Δ attention` and `Δ raw write` are clean minus corrupt. Rescue patches clean head-z into corrupt prompts; ablation patches corrupt head-z into clean prompts.",
        "",
        "| domain | head(s) | Δ attention | Δ raw tool write | rescue Δ logit | rescue strict recovery | ablation Δ logit | ablation strict loss |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['domain']} | {row['heads']} | "
            f"{float(row['sum_delta_attention_clean_minus_corrupt']):+.4f} | "
            f"{float(row['sum_delta_raw_tool_unembedding_write']):+.4f} | "
            f"{float(row['rescue_mean_tool_logit_delta']):+.4f} | "
            f"{float(row['rescue_strict_recovery_rate']):.1%} | "
            f"{float(row['ablation_mean_tool_logit_delta']):+.4f} | "
            f"{float(row['ablation_strict_loss_rate']):.1%} |"
        )
    lines.extend(
        [
            "",
            "Interpret rescue and ablation together: a head has a cross-domain bridge role only where clean-z raises tool evidence on corrupt prompts and corrupt-z removes it from clean prompts. The three-head row tests their joint effect; it should not be read as a complete circuit proof.",
        ]
    )
    write_text(output_root / "summary.md", "\n".join(lines))


def main() -> None:
    args = parse_args()
    args.output_root = args.output_root.resolve()
    args.dataset_view_root = args.dataset_view_root.resolve()
    if args.pairs <= 0 or args.batch_size <= 0:
        raise ValueError("--pairs and --batch-size must be positive")
    if not args.model_path.is_dir():
        raise FileNotFoundError(f"Model path does not exist: {args.model_path}")
    if args.output_root.exists() and not args.resume:
        raise FileExistsError(f"Output root already exists: {args.output_root}; pass --resume to reuse completed domains")
    ensure_dir(args.output_root)
    set_seed(args.seed)
    model, tokenizer = load_hooked_qwen3(str(args.model_path), device=args.device, dtype=torch.bfloat16)
    model.eval()
    try:
        all_rows: list[dict[str, object]] = []
        for domain in args.domains:
            all_rows.extend(run_domain(model, tokenizer, args, domain))
            clear_cuda()
        write_global_summary(args.output_root, all_rows)
        write_json(
            args.output_root / "run_metadata.json",
            {
                "domains": list(args.domains),
                "split": args.split,
                "pairs_per_domain": args.pairs,
                "model_path": str(args.model_path.resolve()),
                "heads": [head_name(head) for head in HEADS],
                "batch_size": args.batch_size,
                "variants_per_forward": args.variants_per_forward,
                "seed": args.seed,
                "device": args.device,
            },
        )
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
