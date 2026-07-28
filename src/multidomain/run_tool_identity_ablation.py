#!/usr/bin/env python3
"""Run the D1 tool-identity scaffold ablation on v4 500-pair data.

This is the D1 tool-identity rebuttal experiment documented in
``rebuttal/README.md``.
It deliberately keeps the v4 D1 membership fixed (400 train / 100 test) and
changes only the JSON payload inside the system ``<tools>`` block.  In
particular, it never re-screens or re-selects prompts after a scaffold change.

For V0--V6 the runner reports three distinct quantities:

1. behavior in the variant itself (tool-token logit, probability, rank, top-1);
2. whether a clean-minus-corrupt L24 vector re-estimated in that variant is
   aligned with V0; and
3. whether the *frozen V0 vector* remains causally sufficient / necessary in
   the variant, with the normalization denominator measured in that same
   variant.

It also writes the requested 2x2 affordance-reversal behavior control using
the held-out D1 task bodies.  The control holds the user turn fixed within a
row and swaps only ``write_file`` against ``submit_review``.

The script is intentionally D1-specific for its first execution.  Extending
the successful protocol to D3--D5 should reuse its measurement conventions,
but requires domain-appropriate matched/reversed schema definitions rather
than silently reusing code-tool semantics.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch
from tqdm.auto import tqdm


THIS_DIR = Path(__file__).resolve().parent
SHARED_DIR = THIS_DIR.parent / "shared"
if str(SHARED_DIR) not in sys.path:
    sys.path.insert(0, str(SHARED_DIR))

from multiscale_common import (  # noqa: E402
    build_pair_batches,
    clear_cuda,
    load_model_and_tokenizer,
    load_sample_pairs,
    set_seed,
    write_csv,
    write_json,
)
from run_cross_scale_fix_gate_and_patch import (  # noqa: E402
    collect_residuals_at_hook,
    compute_pca,
    hook_name,
    make_last_token_add_hook,
)


PROJECT_ROOT = THIS_DIR.parents[1]
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "datasets" / "v4_multidomain_balanced" / "D1"
DEFAULT_MODEL_PATH = Path(
    os.environ.get("QWEN3_8B_PATH", PROJECT_ROOT / "external" / "models" / "Qwen3-8B")
).expanduser()
VARIANT_ORDER = ("V0", "V1", "V2", "V3", "V4", "V5", "V6")


@dataclass(frozen=True)
class VariantSpec:
    key: str
    label: str
    description: str
    schema: str | None


@dataclass
class RenderedPair:
    """A schema-rendered counterpart of a fixed v4 source pair."""

    order: int
    sample_id: str
    clean_text: str
    corrupt_text: str
    clean_tokens_cpu: torch.Tensor
    corrupt_tokens_cpu: torch.Tensor
    token_len: int
    clean_candidate: str | None
    corrupt_candidate: str | None


@dataclass
class PromptItem:
    sample_id: str
    prompt: str
    tokens_cpu: torch.Tensor
    token_len: int
    schema_label: str
    user_verb: str


@dataclass
class MetricAccumulator:
    """Aggregate first-token tool-call measurements without losing rank data."""

    n: int = 0
    tool_logit_sum: float = 0.0
    tool_prob_sum: float = 0.0
    top1_count: int = 0
    ranks: list[int] = field(default_factory=list)

    def update(self, stats: dict[str, torch.Tensor], *, tool_token_id: int) -> None:
        self.n += int(stats["tool_logit"].shape[0])
        self.tool_logit_sum += float(stats["tool_logit"].sum().item())
        self.tool_prob_sum += float(stats["tool_prob"].sum().item())
        self.top1_count += int((stats["top1"] == tool_token_id).sum().item())
        self.ranks.extend(int(value) for value in stats["tool_rank"].tolist())

    def summary(self) -> dict[str, Any]:
        if self.n <= 0:
            raise ValueError("Cannot summarize an empty metric accumulator")
        ranks = sorted(self.ranks)
        midpoint = len(ranks) // 2
        if len(ranks) % 2:
            median_rank: float = float(ranks[midpoint])
        else:
            median_rank = (ranks[midpoint - 1] + ranks[midpoint]) / 2.0
        buckets = {
            "rank_1": sum(rank == 1 for rank in ranks),
            "rank_2": sum(rank == 2 for rank in ranks),
            "rank_3": sum(rank == 3 for rank in ranks),
            "rank_4_to_10": sum(4 <= rank <= 10 for rank in ranks),
            "rank_above_10": sum(rank > 10 for rank in ranks),
        }
        return {
            "n": self.n,
            "mean_tool_call_logit": self.tool_logit_sum / self.n,
            "mean_tool_call_prob": self.tool_prob_sum / self.n,
            "tool_call_top1_rate": self.top1_count / self.n,
            "mean_tool_call_rank": sum(ranks) / self.n,
            "median_tool_call_rank": median_rank,
            "tool_call_top3_rate": sum(rank <= 3 for rank in ranks) / self.n,
            "rank_buckets": buckets,
        }


# Keep every JSON string explicit.  The exact text is written to variants.json
# so that a rebuttal table is auditable without reading this source file.
D1_VARIANTS: dict[str, VariantSpec] = {
    "V0": VariantSpec(
        key="V0",
        label="original write_file",
        description="Original v4 D1 schema, preserved byte-for-byte.",
        schema=None,
    ),
    "V1": VariantSpec(
        key="V1",
        label="apply_patch",
        description="Code-edit affordance without literal overlap with write_file.",
        schema=(
            '{"type":"function","function":{"name":"apply_patch",'
            '"description":"Apply a patch to a file.","parameters":{"type":"object",'
            '"properties":{"file_path":{"type":"string"},"patch":{"type":"string"}},'
            '"required":["file_path","patch"]}}}'
        ),
    ),
    "V2": VariantSpec(
        key="V2",
        label="f1 / empty description",
        description="A semantically opaque tool name and empty description; only schema structure remains.",
        schema=(
            '{"type":"function","function":{"name":"f1","description":"",'
            '"parameters":{"type":"object","properties":{"file_path":{"type":"string"},'
            '"content":{"type":"string"}},"required":["file_path","content"]}}}'
        ),
    ),
    "V3": VariantSpec(
        key="V3",
        label="write_file / unrelated description",
        description="Preserves the original tool name but replaces its description with unrelated weather content.",
        schema=(
            '{"type":"function","function":{"name":"write_file",'
            '"description":"Get the current weather forecast.","parameters":{"type":"object",'
            '"properties":{"file_path":{"type":"string"},"content":{"type":"string"}},'
            '"required":["file_path","content"]}}}'
        ),
    ),
    "V4": VariantSpec(
        key="V4",
        label="submit_review",
        description="Affordance reversal: a review-submission tool replaces the code-writing tool.",
        schema=(
            '{"type":"function","function":{"name":"submit_review",'
            '"description":"Submit review.","parameters":{"type":"object",'
            '"properties":{"file_path":{"type":"string"},"comments":{"type":"string"}},'
            '"required":["file_path","comments"]}}}'
        ),
    ),
    "V5": VariantSpec(
        key="V5",
        label="get_weather",
        description="A deliberately mismatched weather tool.",
        schema=(
            '{"type":"function","function":{"name":"get_weather",'
            '"description":"Get weather.","parameters":{"type":"object",'
            '"properties":{"location":{"type":"string"}},"required":["location"]}}}'
        ),
    ),
    "V6": VariantSpec(
        key="V6",
        label="write_file + explain_code",
        description="Two tools: original code-writing affordance plus an analysis-oriented explanation tool.",
        schema=(
            '{"type":"function","function":{"name":"write_file","description":"Write file.",'
            '"parameters":{"type":"object","properties":{"file_path":{"type":"string"},'
            '"content":{"type":"string"}},"required":["file_path","content"]}}}'
            "\n"
            '{"type":"function","function":{"name":"explain_code","description":"Explain code.",'
            '"parameters":{"type":"object","properties":{"file_path":{"type":"string"},'
            '"explanation":{"type":"string"}},"required":["file_path","explanation"]}}}'
        ),
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the v4 D1 tool-identity scaffold ablation.")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--model-label", type=str, default="Qwen3-8B")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=24)
    parser.add_argument("--hook-kind", choices=("pre", "post"), default="pre")
    parser.add_argument("--train-pairs", type=int, default=400)
    parser.add_argument("--eval-pairs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--variants", nargs="+", choices=VARIANT_ORDER, default=list(VARIANT_ORDER))
    parser.add_argument(
        "--skip-affordance-reversal",
        action="store_true",
        help="Skip the held-out Write/Review x write_file/submit_review behavior grid.",
    )
    return parser.parse_args()


def schema_bounds(prompt: str) -> tuple[int, int]:
    """Return the exact replaceable schema region, excluding its final newline."""

    open_marker = "<tools>\n"
    start = prompt.find(open_marker)
    if start < 0:
        raise ValueError("Prompt does not contain the expected <tools> opening")
    start += len(open_marker)
    close = prompt.find("</tools>", start)
    if close < 0:
        raise ValueError("Prompt does not contain </tools>")
    end = close - 1 if close > start and prompt[close - 1] == "\n" else close
    if end <= start:
        raise ValueError("Tool schema is empty")
    return start, end


def extract_schema(prompt: str) -> str:
    start, end = schema_bounds(prompt)
    return prompt[start:end]


def replace_schema(prompt: str, schema: str) -> str:
    if not schema or "<tools>" in schema or "</tools>" in schema:
        raise ValueError("Variant schema must be non-empty tool payload text")
    start, end = schema_bounds(prompt)
    return prompt[:start] + schema + prompt[end:]


def replace_leading_user_verb(prompt: str, verb: str) -> str:
    """Replace only the leading word of the first user instruction line."""

    normalized = verb.strip()
    if not normalized or any(char.isspace() for char in normalized):
        raise ValueError(f"Expected one non-empty verb, got {verb!r}")
    marker = "<|im_start|>user\n"
    start = prompt.find(marker)
    if start < 0:
        raise ValueError("Prompt does not contain a user turn")
    start += len(marker)
    space = prompt.find(" ", start)
    line_end = prompt.find("\n", start)
    if space < 0 or (line_end >= 0 and space > line_end):
        raise ValueError("Could not isolate the leading user verb")
    return prompt[:start] + normalized + prompt[space:]


def token_stats(logits: torch.Tensor, tool_token_id: int) -> dict[str, torch.Tensor]:
    """Return full-distribution tool-token measurements for the final prompt token."""

    final_logits = logits[:, -1, :].float()
    tool_logit = final_logits[:, tool_token_id]
    tool_prob = torch.softmax(final_logits, dim=-1)[:, tool_token_id]
    top1 = final_logits.argmax(dim=-1)
    # Strict comparison makes rank 1 exactly the top-1 status except for an
    # exact logit tie, which is immaterial here and is retained transparently.
    tool_rank = (final_logits > tool_logit.unsqueeze(-1)).sum(dim=-1) + 1
    return {
        "tool_logit": tool_logit.detach().cpu(),
        "tool_prob": tool_prob.detach().cpu(),
        "top1": top1.detach().cpu(),
        "tool_rank": tool_rank.detach().cpu(),
    }


def validate_pairs(pairs: Sequence[RenderedPair], *, expected_count: int, label: str) -> dict[str, Any]:
    if len(pairs) != int(expected_count):
        raise ValueError(f"{label}: expected {expected_count} pairs, found {len(pairs)}")
    positions: set[int] = set()
    for pair in pairs:
        if pair.clean_tokens_cpu.shape != pair.corrupt_tokens_cpu.shape:
            raise ValueError(f"{label}/{pair.sample_id}: clean/corrupt token shapes differ")
        differences = (pair.clean_tokens_cpu != pair.corrupt_tokens_cpu).nonzero(as_tuple=False)
        if int(differences.shape[0]) != 1:
            raise ValueError(
                f"{label}/{pair.sample_id}: expected one differing input token, found {differences.shape[0]}"
            )
        positions.add(int(differences[0, -1].item()))
    return {
        "pair_count": len(pairs),
        "token_length_min": min(pair.token_len for pair in pairs),
        "token_length_max": max(pair.token_len for pair in pairs),
        "unique_differing_token_positions": sorted(positions),
    }


def render_pairs_for_variant(
    model,
    base_pairs: Sequence[Any],
    *,
    variant: VariantSpec,
) -> list[RenderedPair]:
    rendered: list[RenderedPair] = []
    for base in base_pairs:
        if variant.schema is None:
            clean_text = base.clean_text
            corrupt_text = base.corrupt_text
        else:
            clean_text = replace_schema(base.clean_text, variant.schema)
            corrupt_text = replace_schema(base.corrupt_text, variant.schema)
        clean_tokens = model.to_tokens(clean_text, prepend_bos=False).detach().cpu()
        corrupt_tokens = model.to_tokens(corrupt_text, prepend_bos=False).detach().cpu()
        if int(clean_tokens.shape[-1]) != int(corrupt_tokens.shape[-1]):
            raise ValueError(f"{variant.key}/{base.sample_id}: schema rendering broke pair token alignment")
        rendered.append(
            RenderedPair(
                order=int(base.order),
                sample_id=str(base.sample_id),
                clean_text=clean_text,
                corrupt_text=corrupt_text,
                clean_tokens_cpu=clean_tokens,
                corrupt_tokens_cpu=corrupt_tokens,
                token_len=int(clean_tokens.shape[-1]),
                clean_candidate=base.clean_candidate,
                corrupt_candidate=base.corrupt_candidate,
            )
        )
    return rendered


def append_sample_rows(
    rows: list[dict[str, Any]],
    *,
    pairs: Sequence[RenderedPair],
    indices: Sequence[int],
    variant: str,
    split: str,
    condition: str,
    stats: dict[str, torch.Tensor],
    tool_token_id: int,
) -> None:
    for local_idx, pair_idx in enumerate(indices):
        pair = pairs[int(pair_idx)]
        rows.append(
            {
                "variant": variant,
                "split": split,
                "condition": condition,
                "sample_id": pair.sample_id,
                "clean_verb": pair.clean_candidate or "",
                "corrupt_verb": pair.corrupt_candidate or "",
                "tool_call_logit": float(stats["tool_logit"][local_idx].item()),
                "tool_call_prob": float(stats["tool_prob"][local_idx].item()),
                "tool_call_rank": int(stats["tool_rank"][local_idx].item()),
                "tool_call_top1": bool(int(stats["top1"][local_idx].item()) == tool_token_id),
                "top1_token_id": int(stats["top1"][local_idx].item()),
            }
        )


def collect_variant_vector(
    model,
    pairs: Sequence[RenderedPair],
    *,
    layer: int,
    hook_kind: str,
    batch_size: int,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    clean_resid, corrupt_resid = collect_residuals_at_hook(
        model,
        pairs,
        layer=layer,
        hook_kind=hook_kind,
        batch_size=batch_size,
    )
    diff = clean_resid - corrupt_resid
    pca = compute_pca(diff, n_components=10)
    vector = pca["mean_diff"].detach().cpu().float().view(-1).contiguous()
    if vector.numel() != int(model.cfg.d_model) or not bool(torch.isfinite(vector).all()):
        raise ValueError("Invalid re-estimated mean-difference vector")
    if float(vector.norm().item()) <= 0.0:
        raise ValueError("Re-estimated mean-difference vector has zero norm")
    del clean_resid, corrupt_resid, diff
    clear_cuda()
    return vector, pca


def flatten_metrics(prefix: str, metrics: dict[str, Any]) -> dict[str, Any]:
    return {f"{prefix}_{key}": value for key, value in metrics.items() if key != "rank_buckets"}


def normalized_effect(numerator: float, denominator: float) -> float | None:
    """Only score Suff./Necc. when clean remains the higher-logit side."""

    if not math.isfinite(numerator) or not math.isfinite(denominator) or denominator <= 1e-8:
        return None
    return numerator / denominator


def evaluate_variant(
    model,
    pairs: Sequence[RenderedPair],
    *,
    variant: str,
    layer: int,
    hook_kind: str,
    frozen_v0: torch.Tensor,
    batch_size: int,
    tool_token_id: int,
    sample_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Score behavior plus frozen-V0 addition/removal on one held-out variant."""

    clean_metrics = MetricAccumulator()
    corrupt_metrics = MetricAccumulator()
    add_metrics = MetricAccumulator()
    remove_metrics = MetricAccumulator()
    corrupt_non_tool_count = 0
    clean_tool_count = 0
    add_strict_flip_count = 0
    remove_strict_drop_count = 0
    name = hook_name(layer, hook_kind)
    add_hook = (name, make_last_token_add_hook(frozen_v0))
    remove_hook = (name, make_last_token_add_hook(-frozen_v0))
    batches = build_pair_batches(pairs, batch_size=batch_size)

    progress = tqdm(batches, desc=f"{variant}: held-out behavior + frozen V0", dynamic_ncols=True)
    for batch in progress:
        clean_tokens = batch.clean_tokens_cpu.to(model.W_U.device)
        corrupt_tokens = batch.corrupt_tokens_cpu.to(model.W_U.device)
        with torch.no_grad():
            clean_logits = model(clean_tokens)
            corrupt_logits = model(corrupt_tokens)
            add_logits = model.run_with_hooks(corrupt_tokens, fwd_hooks=[add_hook])
            remove_logits = model.run_with_hooks(clean_tokens, fwd_hooks=[remove_hook])

        clean_stats = token_stats(clean_logits, tool_token_id)
        corrupt_stats = token_stats(corrupt_logits, tool_token_id)
        add_stats = token_stats(add_logits, tool_token_id)
        remove_stats = token_stats(remove_logits, tool_token_id)
        clean_metrics.update(clean_stats, tool_token_id=tool_token_id)
        corrupt_metrics.update(corrupt_stats, tool_token_id=tool_token_id)
        add_metrics.update(add_stats, tool_token_id=tool_token_id)
        remove_metrics.update(remove_stats, tool_token_id=tool_token_id)

        clean_is_tool = clean_stats["top1"] == tool_token_id
        corrupt_is_tool = corrupt_stats["top1"] == tool_token_id
        add_is_tool = add_stats["top1"] == tool_token_id
        remove_is_tool = remove_stats["top1"] == tool_token_id
        clean_tool_count += int(clean_is_tool.sum().item())
        corrupt_non_tool_count += int((~corrupt_is_tool).sum().item())
        add_strict_flip_count += int(((~corrupt_is_tool) & add_is_tool).sum().item())
        remove_strict_drop_count += int((clean_is_tool & (~remove_is_tool)).sum().item())

        append_sample_rows(
            sample_rows,
            pairs=pairs,
            indices=batch.indices,
            variant=variant,
            split="test",
            condition="baseline_clean",
            stats=clean_stats,
            tool_token_id=tool_token_id,
        )
        append_sample_rows(
            sample_rows,
            pairs=pairs,
            indices=batch.indices,
            variant=variant,
            split="test",
            condition="baseline_corrupt",
            stats=corrupt_stats,
            tool_token_id=tool_token_id,
        )
        append_sample_rows(
            sample_rows,
            pairs=pairs,
            indices=batch.indices,
            variant=variant,
            split="test",
            condition="frozen_v0_add_to_corrupt",
            stats=add_stats,
            tool_token_id=tool_token_id,
        )
        append_sample_rows(
            sample_rows,
            pairs=pairs,
            indices=batch.indices,
            variant=variant,
            split="test",
            condition="frozen_v0_remove_from_clean",
            stats=remove_stats,
            tool_token_id=tool_token_id,
        )

        del (
            clean_tokens,
            corrupt_tokens,
            clean_logits,
            corrupt_logits,
            add_logits,
            remove_logits,
            clean_stats,
            corrupt_stats,
            add_stats,
            remove_stats,
            clean_is_tool,
            corrupt_is_tool,
            add_is_tool,
            remove_is_tool,
        )
        progress.set_postfix(tok=batch.token_len)

    clean = clean_metrics.summary()
    corrupt = corrupt_metrics.summary()
    added = add_metrics.summary()
    removed = remove_metrics.summary()
    logit_gap = float(clean["mean_tool_call_logit"] - corrupt["mean_tool_call_logit"])
    sufficiency = normalized_effect(
        float(added["mean_tool_call_logit"] - corrupt["mean_tool_call_logit"]), logit_gap
    )
    necessity = normalized_effect(
        float(clean["mean_tool_call_logit"] - removed["mean_tool_call_logit"]), logit_gap
    )
    return {
        "variant": variant,
        "n": len(pairs),
        "baseline": {"clean": clean, "corrupt": corrupt},
        "frozen_v0_intervention": {
            "add_to_corrupt": added,
            "remove_from_clean": removed,
            "add_strict_flip_rate": add_strict_flip_count / max(corrupt_non_tool_count, 1),
            "remove_strict_drop_rate": remove_strict_drop_count / max(clean_tool_count, 1),
            "condition_logit_gap_clean_minus_corrupt": logit_gap,
            "normalization_valid": logit_gap > 1e-8,
            "sufficiency_normalized_logit_gap": sufficiency,
            "necessity_normalized_logit_gap": necessity,
        },
    }


def build_affordance_prompts(
    model,
    base_pairs: Sequence[Any],
    *,
    original_schema: str,
) -> dict[tuple[str, str], list[PromptItem]]:
    """Build held-out 2x2 Write/Review x write_file/submit_review prompts."""

    schemas = {
        "write_file": original_schema,
        "submit_review": D1_VARIANTS["V4"].schema,
    }
    if schemas["submit_review"] is None:
        raise AssertionError("V4 must supply a schema")
    output: dict[tuple[str, str], list[PromptItem]] = {}
    for user_verb in ("Write", "Review"):
        for schema_label, schema in schemas.items():
            prompts: list[PromptItem] = []
            for pair in base_pairs:
                prompt = replace_leading_user_verb(pair.clean_text, user_verb)
                if schema_label != "write_file":
                    prompt = replace_schema(prompt, schema)
                tokens = model.to_tokens(prompt, prepend_bos=False).detach().cpu()
                prompts.append(
                    PromptItem(
                        sample_id=str(pair.sample_id),
                        prompt=prompt,
                        tokens_cpu=tokens,
                        token_len=int(tokens.shape[-1]),
                        schema_label=schema_label,
                        user_verb=user_verb,
                    )
                )
            output[(user_verb, schema_label)] = prompts
    return output


def prompt_batches(items: Sequence[PromptItem], batch_size: int) -> Iterable[list[PromptItem]]:
    buckets: dict[int, list[PromptItem]] = defaultdict(list)
    for item in items:
        buckets[item.token_len].append(item)
    for token_len in sorted(buckets):
        group = buckets[token_len]
        for start in range(0, len(group), max(int(batch_size), 1)):
            yield group[start : start + max(int(batch_size), 1)]


def evaluate_affordance_reversal(
    model,
    base_pairs: Sequence[Any],
    *,
    original_schema: str,
    batch_size: int,
    tool_token_id: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    prompt_sets = build_affordance_prompts(model, base_pairs, original_schema=original_schema)
    rows: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []
    for (user_verb, schema_label), items in prompt_sets.items():
        accumulator = MetricAccumulator()
        progress = tqdm(
            list(prompt_batches(items, batch_size)),
            desc=f"affordance {user_verb} / {schema_label}",
            dynamic_ncols=True,
        )
        for batch in progress:
            tokens = torch.cat([item.tokens_cpu for item in batch], dim=0).to(model.W_U.device)
            with torch.no_grad():
                logits = model(tokens)
            stats = token_stats(logits, tool_token_id)
            accumulator.update(stats, tool_token_id=tool_token_id)
            for idx, item in enumerate(batch):
                sample_rows.append(
                    {
                        "sample_id": item.sample_id,
                        "user_verb": item.user_verb,
                        "schema": item.schema_label,
                        "tool_call_logit": float(stats["tool_logit"][idx].item()),
                        "tool_call_prob": float(stats["tool_prob"][idx].item()),
                        "tool_call_rank": int(stats["tool_rank"][idx].item()),
                        "tool_call_top1": bool(int(stats["top1"][idx].item()) == tool_token_id),
                        "top1_token_id": int(stats["top1"][idx].item()),
                    }
                )
            del tokens, logits, stats
        rows.append(
            {
                "user_verb": user_verb,
                "schema": schema_label,
                **flatten_metrics("behavior", accumulator.summary()),
            }
        )
    return rows, sample_rows


def cosine_similarity(left: torch.Tensor, right: torch.Tensor) -> float:
    left_unit = left.float() / left.float().norm().clamp_min(1e-12)
    right_unit = right.float() / right.float().norm().clamp_min(1e-12)
    # Float32 reduction can overshoot one by a few ULPs for V0 versus itself.
    return float(torch.dot(left_unit, right_unit).clamp(-1.0, 1.0).item())


def main() -> None:
    args = parse_args()
    selected_variants = tuple(args.variants)
    if "V0" not in selected_variants:
        raise ValueError("V0 is required because every causal test uses the frozen original V0 vector")
    if len(set(selected_variants)) != len(selected_variants):
        raise ValueError("Duplicate --variants entries are not allowed")
    output_root = args.output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite existing output root: {output_root}")
    dataset_root = args.dataset_root.resolve()
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")
    if int(args.train_pairs) <= 0 or int(args.eval_pairs) <= 0:
        raise ValueError("--train-pairs and --eval-pairs must be positive")

    output_root.mkdir(parents=True, exist_ok=False)
    write_json(
        output_root / "run_config.json",
        {
            "experiment": "v4_d1_tool_identity_ablation",
            "dataset_root": str(dataset_root),
            "model_path": str(args.model_path.resolve()),
            "model_label": args.model_label,
            "layer": args.layer,
            "hook_kind": args.hook_kind,
            "train_pairs": args.train_pairs,
            "eval_pairs": args.eval_pairs,
            "batch_size": args.batch_size,
            "seed": args.seed,
            "variants": list(selected_variants),
            "frozen_vector": "V0 mean(clean L24 pre residual - corrupt L24 pre residual) on fixed D1 v4 train split",
            "selection_rule": "No schema-variant behavioral re-screening or re-selection.",
        },
    )
    write_json(
        output_root / "variants.json",
        {
            key: {
                "label": D1_VARIANTS[key].label,
                "description": D1_VARIANTS[key].description,
                "schema": D1_VARIANTS[key].schema,
            }
            for key in selected_variants
        },
    )

    set_seed(args.seed)
    model, _tokenizer, tool_token_id = load_model_and_tokenizer(
        model_path=args.model_path.resolve(), device=args.device
    )
    if args.layer < 0 or args.layer >= int(model.cfg.n_layers):
        raise ValueError(f"L{args.layer} is invalid for this model ({model.cfg.n_layers} layers)")

    try:
        base_train = load_sample_pairs(model, dataset_root=dataset_root, split="train", max_pairs=args.train_pairs)
        base_test = load_sample_pairs(model, dataset_root=dataset_root, split="test", max_pairs=args.eval_pairs)
        if len(base_train) != int(args.train_pairs) or len(base_test) != int(args.eval_pairs):
            raise ValueError("Dataset did not yield the requested fixed train/test cardinalities")
        original_schema = extract_schema(base_train[0].clean_text)
        for base in (*base_train, *base_test):
            if extract_schema(base.clean_text) != original_schema or extract_schema(base.corrupt_text) != original_schema:
                raise ValueError(f"{base.sample_id}: original D1 schema is not invariant across fixed data")

        pair_validation: dict[str, Any] = {}
        vectors: dict[str, torch.Tensor] = {}
        vector_summaries: dict[str, dict[str, Any]] = {}
        vector_root = output_root / "native_vectors"
        vector_root.mkdir(parents=True, exist_ok=True)

        for variant_key in selected_variants:
            variant = D1_VARIANTS[variant_key]
            rendered_train = render_pairs_for_variant(model, base_train, variant=variant)
            pair_validation[f"{variant_key}/train"] = validate_pairs(
                rendered_train, expected_count=args.train_pairs, label=f"{variant_key}/train"
            )
            vector, pca = collect_variant_vector(
                model,
                rendered_train,
                layer=args.layer,
                hook_kind=args.hook_kind,
                batch_size=args.batch_size,
            )
            vector_path = vector_root / f"{variant_key}_L{args.layer}_{args.hook_kind}.pt"
            torch.save(
                {
                    "mean_diff": vector,
                    "components": pca["components"].detach().cpu(),
                    "explained_variance": pca["explained_variance"].detach().cpu(),
                    "explained_variance_ratio": pca["explained_variance_ratio"].detach().cpu(),
                    "singular_values": pca["singular_values"].detach().cpu(),
                    "variant": variant_key,
                    "domain": "D1",
                    "dataset_root": str(dataset_root),
                    "train_sample_ids": [pair.sample_id for pair in rendered_train],
                    "layer": args.layer,
                    "hook_kind": args.hook_kind,
                    "model_d_model": int(model.cfg.d_model),
                },
                vector_path,
            )
            vectors[variant_key] = vector
            vector_summaries[variant_key] = {
                "bundle": str(vector_path),
                "l2_norm": float(vector.norm().item()),
                "rms": float(vector.square().mean().sqrt().item()),
                "train_pair_count": len(rendered_train),
                "pca_explained_variance_ratio": [
                    float(value) for value in pca["explained_variance_ratio"].detach().cpu().tolist()
                ],
            }
            del rendered_train, vector, pca
            clear_cuda()

        frozen_v0 = vectors["V0"].detach().cpu().float().contiguous()
        for variant_key, vector in vectors.items():
            vector_summaries[variant_key]["cosine_to_frozen_v0"] = cosine_similarity(vector, frozen_v0)
            vector_summaries[variant_key]["l2_norm_over_v0"] = float(vector.norm().item() / frozen_v0.norm().item())
        write_json(output_root / "native_vectors.json", vector_summaries)

        behavior_rows: list[dict[str, Any]] = []
        intervention_rows: list[dict[str, Any]] = []
        sample_rows: list[dict[str, Any]] = []
        detailed_results: dict[str, Any] = {}
        for variant_key in selected_variants:
            variant = D1_VARIANTS[variant_key]
            rendered_test = render_pairs_for_variant(model, base_test, variant=variant)
            pair_validation[f"{variant_key}/test"] = validate_pairs(
                rendered_test, expected_count=args.eval_pairs, label=f"{variant_key}/test"
            )
            result = evaluate_variant(
                model,
                rendered_test,
                variant=variant_key,
                layer=args.layer,
                hook_kind=args.hook_kind,
                frozen_v0=frozen_v0,
                batch_size=args.batch_size,
                tool_token_id=tool_token_id,
                sample_rows=sample_rows,
            )
            detailed_results[variant_key] = result
            clean = result["baseline"]["clean"]
            corrupt = result["baseline"]["corrupt"]
            intervention = result["frozen_v0_intervention"]
            behavior_rows.extend(
                [
                    {"variant": variant_key, "side": "clean", **flatten_metrics("behavior", clean)},
                    {"variant": variant_key, "side": "corrupt", **flatten_metrics("behavior", corrupt)},
                ]
            )
            intervention_rows.append(
                {
                    "variant": variant_key,
                    "cosine_to_frozen_v0": vector_summaries[variant_key]["cosine_to_frozen_v0"],
                    "native_l2_norm": vector_summaries[variant_key]["l2_norm"],
                    "native_l2_norm_over_v0": vector_summaries[variant_key]["l2_norm_over_v0"],
                    "baseline_clean_top1_rate": clean["tool_call_top1_rate"],
                    "baseline_corrupt_top1_rate": corrupt["tool_call_top1_rate"],
                    "baseline_clean_mean_tool_prob": clean["mean_tool_call_prob"],
                    "baseline_corrupt_mean_tool_prob": corrupt["mean_tool_call_prob"],
                    "condition_logit_gap_clean_minus_corrupt": intervention[
                        "condition_logit_gap_clean_minus_corrupt"
                    ],
                    "normalization_valid": intervention["normalization_valid"],
                    "frozen_v0_add_top1_rate": intervention["add_to_corrupt"]["tool_call_top1_rate"],
                    "frozen_v0_add_mean_tool_prob": intervention["add_to_corrupt"]["mean_tool_call_prob"],
                    "frozen_v0_add_strict_flip_rate": intervention["add_strict_flip_rate"],
                    "frozen_v0_sufficiency": intervention["sufficiency_normalized_logit_gap"],
                    "frozen_v0_remove_remaining_top1_rate": intervention["remove_from_clean"][
                        "tool_call_top1_rate"
                    ],
                    "frozen_v0_remove_mean_tool_prob": intervention["remove_from_clean"][
                        "mean_tool_call_prob"
                    ],
                    "frozen_v0_remove_strict_drop_rate": intervention["remove_strict_drop_rate"],
                    "frozen_v0_necessity": intervention["necessity_normalized_logit_gap"],
                }
            )
            write_csv(output_root / "behavior_long.partial.csv", behavior_rows)
            write_csv(output_root / "intervention_long.partial.csv", intervention_rows)
            write_csv(output_root / "sample_metrics.partial.csv", sample_rows)
            del rendered_test
            clear_cuda()

        write_json(output_root / "pair_validation.json", pair_validation)
        write_json(output_root / "tool_identity_results.json", detailed_results)
        write_csv(output_root / "behavior_long.csv", behavior_rows)
        write_csv(output_root / "intervention_long.csv", intervention_rows)
        write_csv(output_root / "sample_metrics.csv", sample_rows)

        affordance_rows: list[dict[str, Any]] = []
        affordance_sample_rows: list[dict[str, Any]] = []
        if not args.skip_affordance_reversal:
            affordance_rows, affordance_sample_rows = evaluate_affordance_reversal(
                model,
                base_test,
                original_schema=original_schema,
                batch_size=args.batch_size,
                tool_token_id=tool_token_id,
            )
            write_csv(output_root / "affordance_reversal_2x2.csv", affordance_rows)
            write_csv(output_root / "affordance_reversal_2x2_samples.csv", affordance_sample_rows)

        d1_gate = {
            "pre_registered_thresholds": {
                "vector_survival_cosine": ">= 0.6 (V2 may be lower, per plan)",
                "frozen_v0_sufficiency": ">= 0.85 for semantically matched V1--V3; < 0.5 is falsifying",
                "frozen_v0_necessity": "reported with the same condition-specific logit-gap normalization",
                "falsifier": "cosine < 0.3 or Suff. < 0.5 means the V0 vector is write_file-specific in that variant",
            },
            "observed": {
                row["variant"]: {
                    "cosine_to_frozen_v0": row["cosine_to_frozen_v0"],
                    "frozen_v0_sufficiency": row["frozen_v0_sufficiency"],
                    "frozen_v0_necessity": row["frozen_v0_necessity"],
                    "normalization_valid": row["normalization_valid"],
                }
                for row in intervention_rows
            },
        }
        write_json(output_root / "d1_expansion_gate.json", d1_gate)
        write_json(
            output_root / "completion.json",
            {
                "status": "complete",
                "domain": "D1",
                "dataset_cardinality": {"train": len(base_train), "test": len(base_test)},
                "tool_token_id": tool_token_id,
                "layer": args.layer,
                "hook_kind": args.hook_kind,
                "variants": list(selected_variants),
                "affordance_reversal_completed": not args.skip_affordance_reversal,
                "model_d_model": int(model.cfg.d_model),
                "model_n_layers": int(model.cfg.n_layers),
            },
        )
        print(
            json.dumps(
                {
                    "status": "complete",
                    "output_root": str(output_root),
                    "variants": list(selected_variants),
                    "domain": "D1",
                },
                ensure_ascii=False,
            )
        )
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
