#!/usr/bin/env python3
"""
Cross-directional causal evaluation for bidirectional circuit discovery.

We evaluate how node groups discovered in each direction move the *tool-call margin*:
  m_tool = logit(<tool_call>) - logit(distractor)

For each sample pair (tool-call prompt vs no-tool prompt), we run two symmetric tests:

1) Promotion (tool-call direction):
   Base = no-tool prompt
   Source = tool-call prompt activations
   Expect: m_tool increases.

2) Suppression (no-tool direction):
   Base = tool-call prompt
   Source = no-tool prompt activations
   Expect: m_tool decreases.

We report effects normalized by the per-sample gap:
  gap = m_tool(tool_call_prompt) - m_tool(no_tool_prompt)
  ratio = (m_tool(patched) - m_tool(base)) / gap
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from tqdm.auto import tqdm

if __package__ in {None, ""}:
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from toolcall_circuit.dataset import load_summary_records
from toolcall_circuit.objective import build_bidirectional_endpoint_objectives
from toolcall_circuit.single_sample import evaluate_on_base_with_source, load_hooked_qwen3, objective_from_logits


def finite(vals: Iterable[float]) -> List[float]:
    return [float(v) for v in vals if isinstance(v, (int, float)) and math.isfinite(float(v))]


def median(vals: Sequence[float]) -> float:
    return float(np.median(vals)) if vals else float("nan")


def mean(vals: Sequence[float]) -> float:
    return float(np.mean(vals)) if vals else float("nan")


def bootstrap_ci(values: Sequence[float], n_boot: int, seed: int) -> Dict[str, float]:
    vals = finite(values)
    if not vals:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan")}
    rng = random.Random(seed)
    n = len(vals)
    boots: List[float] = []
    for _ in range(n_boot):
        sample = [vals[rng.randrange(n)] for __ in range(n)]
        boots.append(float(np.median(sample)))
    boots.sort()
    lo = boots[max(0, int(0.025 * n_boot))]
    hi = boots[min(n_boot - 1, int(0.975 * n_boot))]
    return {"mean": float(np.mean(boots)), "lo": float(lo), "hi": float(hi)}


def parse_head(node: str) -> Tuple[int, int]:
    body = node[1:]
    layer_s, head_s = body.split("H")
    return int(layer_s), int(head_s)


def node_to_hook_name(node: str) -> str:
    if node.startswith("MLP"):
        layer = int(node[3:])
        return f"blocks.{layer}.hook_mlp_out"
    if node.startswith("L"):
        layer, _ = parse_head(node)
        return f"blocks.{layer}.attn.hook_z"
    raise ValueError(f"Unknown node format: {node}")


def filter_valid_nodes(nodes: Sequence[str]) -> List[str]:
    out: List[str] = []
    for n in nodes:
        n = str(n)
        if n.startswith("MLP") or n.startswith("L"):
            out.append(n)
    return sorted(set(out))


def collect_cache_cpu_for_nodes(model, tokens: torch.Tensor, nodes: Sequence[str]) -> Dict[str, torch.Tensor]:
    needed = sorted({node_to_hook_name(n) for n in filter_valid_nodes(nodes)})
    needed_set = set(needed)
    if not needed:
        return {}
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens, names_filter=lambda name: name in needed_set)
    return {k: v.detach().cpu() for k, v in cache.items()}


@dataclass(frozen=True)
class SamplePaths:
    sample_id: str
    tool_call_prompt: Path
    no_tool_prompt: Path
    target_tool_call: int
    distractor: int


def load_sample_paths(forward_batch_root: Path, reverse_batch_root: Path, max_samples: int) -> List[SamplePaths]:
    reverse_ids = {r.sample_id for r in load_summary_records(reverse_batch_root)}
    samples: List[SamplePaths] = []
    for rec in load_summary_records(forward_batch_root):
        if rec.sample_id not in reverse_ids:
            continue
        s = rec.summary
        tool_prompt = Path(s["clean_prompt"])
        no_tool_prompt = Path(s["corrupt_prompt"])
        samples.append(
            SamplePaths(
                sample_id=rec.sample_id,
                tool_call_prompt=tool_prompt,
                no_tool_prompt=no_tool_prompt,
                target_tool_call=int(s["target_token_id"]),
                distractor=int(s["distractor_token_id"]),
            )
        )
        if max_samples > 0 and len(samples) >= max_samples:
            break
    if not samples:
        raise ValueError("No shared samples between forward and reverse roots.")
    return samples


def safe_read_text(path: Path) -> Optional[str]:
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return None


def compute_margin(model, tokens: torch.Tensor, target, distractor: int | None = None) -> float:
    with torch.no_grad():
        logits = model(tokens)
    return float(objective_from_logits(logits, target, distractor).item())


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-directional causal eval for bidirectional circuits.")
    parser.add_argument("--forward-batch-root", type=str, required=True)
    parser.add_argument("--forward-aggregate-summary", type=str, required=True)
    parser.add_argument("--reverse-aggregate-summary", type=str, required=True)
    parser.add_argument("--bidirectional-summary", type=str, required=True)
    parser.add_argument("--model-path", type=str, default="./external/models/Qwen3-1.7B")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--bootstrap", type=int, default=400)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output-root", type=str, required=True)
    parser.add_argument(
        "--matrix",
        choices=["aligned", "full"],
        default="aligned",
        help=(
            "aligned: evaluate forward-groups in promotion direction and reverse-groups in suppression direction; "
            "full: evaluate all groups in both directions (includes off-diagonal necessity-like tests)."
        ),
    )
    args = parser.parse_args()

    out_root = Path(args.output_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    forward_batch_root = Path(args.forward_batch_root).resolve()
    reverse_batch_root = Path(Path(args.reverse_aggregate_summary).resolve().parent.parent / "reverse_batch")
    if not reverse_batch_root.exists():
        reverse_batch_root = Path(args.forward_batch_root).resolve()  # placeholder; will error later if wrong

    fwd_agg = json.loads(Path(args.forward_aggregate_summary).resolve().read_text(encoding="utf-8"))
    rev_agg = json.loads(Path(args.reverse_aggregate_summary).resolve().read_text(encoding="utf-8"))
    bidi = json.loads(Path(args.bidirectional_summary).resolve().read_text(encoding="utf-8"))

    forward_core = filter_valid_nodes(list(fwd_agg.get("core_nodes", [])))
    reverse_core = filter_valid_nodes(list(rev_agg.get("core_nodes", [])))
    shared_core = sorted(set(forward_core) & set(reverse_core))
    forward_only_core = sorted(set(forward_core) - set(reverse_core))
    reverse_only_core = sorted(set(reverse_core) - set(forward_core))

    forward_selective = filter_valid_nodes(list(bidi.get("support_analysis", {}).get("forward_selective_nodes", [])))
    reverse_selective = filter_valid_nodes(list(bidi.get("support_analysis", {}).get("reverse_selective_nodes", [])))
    shared_backbone = filter_valid_nodes(list(bidi.get("support_analysis", {}).get("shared_backbone_nodes", [])))

    groups: Dict[str, List[str]] = {
        "forward_only_core": forward_only_core,
        "reverse_only_core": reverse_only_core,
        "shared_core": shared_core,
        "forward_selective": forward_selective,
        "reverse_selective": reverse_selective,
        "shared_backbone": shared_backbone,
    }
    groups = {k: v for k, v in groups.items() if v}

    # Nodes needed for source caches in each direction.
    if args.matrix == "full":
        promote_groups = list(groups.keys())
        suppress_groups = list(groups.keys())
    else:
        promote_groups = ["forward_only_core", "forward_selective", "shared_core", "shared_backbone"]
        suppress_groups = ["reverse_only_core", "reverse_selective", "shared_core", "shared_backbone"]

    nodes_tool_source = sorted({n for g in promote_groups for n in groups.get(g, [])})
    nodes_no_tool_source = sorted({n for g in suppress_groups for n in groups.get(g, [])})

    # Determine which samples to run on (use reverse-batch sample ids as the shared set).
    reverse_batch_root = Path(args.bidirectional_summary).resolve().parent.parent / "reverse_batch"
    if not reverse_batch_root.exists():
        # fallback: caller may pass reverse-batch root as output-root parent
        reverse_batch_root = Path(args.output_root).resolve().parent / "reverse_batch"
    samples = load_sample_paths(forward_batch_root, reverse_batch_root, max_samples=args.max_samples)

    model, tokenizer = load_hooked_qwen3(args.model_path, device=args.device, dtype=torch.bfloat16)
    _ = tokenizer

    per_sample: List[Dict[str, object]] = []
    pbar = tqdm(samples, desc="Cross-eval samples", dynamic_ncols=True)
    for sp in pbar:
        tool_text = safe_read_text(sp.tool_call_prompt)
        no_tool_text = safe_read_text(sp.no_tool_prompt)
        if tool_text is None or no_tool_text is None:
            continue

        tool_tokens = model.to_tokens(tool_text, prepend_bos=False)
        no_tool_tokens = model.to_tokens(no_tool_text, prepend_bos=False)
        if tool_tokens.shape != no_tool_tokens.shape:
            continue

        with torch.no_grad():
            tool_logits = model(tool_tokens)
            no_tool_logits = model(no_tool_tokens)
        tool_objective, no_tool_objective = build_bidirectional_endpoint_objectives(
            tool_logits,
            no_tool_logits,
            tokenizer=tokenizer,
        )
        m_tool_tool = float(objective_from_logits(tool_logits, tool_objective).item())
        m_tool_no = float(objective_from_logits(no_tool_logits, tool_objective).item())
        gap_tool = m_tool_tool - m_tool_no
        m_no_tool_tool = float(objective_from_logits(tool_logits, no_tool_objective).item())
        m_no_tool_no = float(objective_from_logits(no_tool_logits, no_tool_objective).item())
        gap_no_tool = m_no_tool_no - m_no_tool_tool
        if (
            not math.isfinite(gap_tool)
            or abs(gap_tool) < 1e-8
            or not math.isfinite(gap_no_tool)
            or abs(gap_no_tool) < 1e-8
        ):
            continue

        tool_cache = collect_cache_cpu_for_nodes(model, tool_tokens, nodes_tool_source)
        no_tool_cache = collect_cache_cpu_for_nodes(model, no_tool_tokens, nodes_no_tool_source)

        row: Dict[str, object] = {
            "sample_id": sp.sample_id,
            "tool_endpoint_tool_score": m_tool_tool,
            "tool_endpoint_no_tool_score": m_tool_no,
            "tool_endpoint_gap": gap_tool,
            "no_tool_endpoint_tool_score": m_no_tool_tool,
            "no_tool_endpoint_no_tool_score": m_no_tool_no,
            "no_tool_endpoint_gap": gap_no_tool,
        }

        # Promotion: patch tool-call activations into no-tool base.
        for g in promote_groups:
            nodes = groups.get(g, [])
            if not nodes:
                continue
            patched = evaluate_on_base_with_source(
                model=model,
                base_tokens=no_tool_tokens,
                source_cache_cpu=tool_cache,
                patch_nodes=nodes,
                target_token=tool_objective,
                distractor_token=None,
            )
            row[f"promote__{g}__ratio"] = (float(patched) - m_tool_no) / gap_tool
            row[f"promote__{g}__kl"] = -float(patched)

        # Suppression: patch no-tool activations into tool-call base.
        for g in suppress_groups:
            nodes = groups.get(g, [])
            if not nodes:
                continue
            patched = evaluate_on_base_with_source(
                model=model,
                base_tokens=tool_tokens,
                source_cache_cpu=no_tool_cache,
                patch_nodes=nodes,
                target_token=no_tool_objective,
                distractor_token=None,
            )
            row[f"suppress__{g}__ratio"] = (float(patched) - m_no_tool_tool) / gap_no_tool
            row[f"suppress__{g}__kl"] = -float(patched)

        per_sample.append(row)
        pbar.set_postfix(tool_gap=f"{gap_tool:.2f}", no_tool_gap=f"{gap_no_tool:.2f}")

        model.reset_hooks()
        torch.cuda.empty_cache()

    # Summaries.
    def summarize_key(key: str) -> Dict[str, float]:
        vals = finite([float(r.get(key, float("nan"))) for r in per_sample])
        return {
            "n": len(vals),
            "median": median(vals),
            "mean": mean(vals),
            "bootstrap_median": bootstrap_ci(vals, n_boot=args.bootstrap, seed=args.seed + hash(key) % 10_000),
        }

    summary: Dict[str, object] = {
        "n_samples": len(per_sample),
        "groups": groups,
        "matrix": args.matrix,
        "metrics": {},
        "artifacts": {
            "per_sample_json": str(out_root / "per_sample_cross_eval.json"),
            "summary_json": str(out_root / "cross_eval_summary.json"),
        },
    }

    keys = sorted({k for row in per_sample for k in row.keys() if k.endswith("__ratio")})
    for k in keys:
        summary["metrics"][k] = summarize_key(k)

    (out_root / "per_sample_cross_eval.json").write_text(json.dumps(per_sample, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_root / "cross_eval_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
