#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

if __package__ in {None, ""}:
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

PROJECT_ROOT = Path(__file__).resolve().parents[4]
EAP_SRC = PROJECT_ROOT / "experiment" / "code" / "vendor" / "EAP-IG" / "src"
RUNNER_PATH = PROJECT_ROOT / "experiment" / "code" / "scripts" / "run_toolcall_eap_ig_original_circuit.py"
if str(EAP_SRC) not in sys.path:
    sys.path.insert(0, str(EAP_SRC))

from eap.attribute import attribute  # noqa: E402
from eap.evaluate import evaluate_graph  # noqa: E402
from eap.graph import Graph  # noqa: E402
from eap.utils import tokenize_plus  # noqa: E402

from toolcall_circuit.dataset import ToolCallSample, load_toolcall_samples
from toolcall_circuit.eap_ig_dataset import resolve_direction_spec
from toolcall_circuit.eap_ig_export import clone_graph
from toolcall_circuit.eap_ig_objective import negative_kl_to_clean_endpoint
from toolcall_circuit.paths import DATASETS_ROOT, MODEL_PATH_DEFAULT, RESULTS_ROOT
from toolcall_circuit.single_sample import load_hooked_qwen3


MODES = {
    "full",
    "discover_shard",
    "merge_direction",
    "finalize_direction",
    "full_sharded",
}


class DirectionalPromptDataset(Dataset):
    def __init__(self, samples: Sequence[ToolCallSample], direction: str) -> None:
        self.samples = list(samples)
        self.direction = direction

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        spec = resolve_direction_spec(sample, self.direction)
        label = {
            "sample_id": sample.sample_id,
            "sample_rank": sample.sample_rank,
            "direction": self.direction,
        }
        return spec.clean_text, spec.corrupt_text, label


def collate_rows(rows):
    clean, corrupt, label = zip(*rows)
    return list(clean), list(corrupt), list(label)


def parse_topn_values(raw: str) -> List[int]:
    values = sorted({int(x.strip()) for x in raw.split(",") if x.strip()})
    if not values or any(v <= 0 for v in values):
        raise ValueError(f"Invalid topn values: {raw}")
    return values


def build_samples(dataset_root: Path, max_samples: int) -> List[ToolCallSample]:
    samples = load_toolcall_samples(dataset_root=dataset_root)
    if max_samples > 0:
        samples = samples[: int(max_samples)]
    if not samples:
        raise ValueError(f"No samples found under {dataset_root}")
    return samples


def build_dataloader(samples: Sequence[ToolCallSample], direction: str, batch_size: int) -> DataLoader:
    dataset = DirectionalPromptDataset(samples=samples, direction=direction)
    return DataLoader(
        dataset,
        batch_size=max(1, int(batch_size)),
        shuffle=False,
        collate_fn=collate_rows,
    )


def finite_mean(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if isinstance(v, (int, float)) and math.isfinite(float(v))]
    return float(np.mean(vals)) if vals else float("nan")


def finite_median(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if isinstance(v, (int, float)) and math.isfinite(float(v))]
    return float(np.median(vals)) if vals else float("nan")


def tensor_to_list(values: torch.Tensor) -> List[float]:
    return [float(v) for v in values.detach().cpu().tolist()]


def write_json(path: Path, payload: Dict[str, object] | List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def direction_root(out_root: Path, direction: str) -> Path:
    return out_root / direction


def discovery_root(out_root: Path, direction: str) -> Path:
    return direction_root(out_root, direction) / "discovery"


def shards_root(out_root: Path, direction: str) -> Path:
    return discovery_root(out_root, direction) / "shards"


def shard_specs(n_samples: int, shard_size: int) -> List[Dict[str, int | str]]:
    if shard_size <= 0:
        raise ValueError(f"shard_size must be > 0, got {shard_size}")
    total_shards = int(math.ceil(float(n_samples) / float(shard_size)))
    specs: List[Dict[str, int | str]] = []
    for shard_index in range(total_shards):
        start = shard_index * shard_size
        end = min(n_samples, start + shard_size)
        specs.append(
            {
                "shard_index": shard_index,
                "total_shards": total_shards,
                "start_index": start,
                "end_index_exclusive": end,
                "name": f"shard_{shard_index:04d}_of_{total_shards:04d}",
            }
        )
    return specs


def get_shard_spec(n_samples: int, shard_size: int, shard_index: int) -> Dict[str, int | str]:
    specs = shard_specs(n_samples, shard_size)
    if shard_index < 0 or shard_index >= len(specs):
        raise ValueError(f"Invalid shard_index={shard_index}; valid range is [0, {len(specs) - 1}]")
    return specs[shard_index]


def get_shard_samples(
    samples: Sequence[ToolCallSample],
    *,
    shard_size: int,
    shard_index: int,
) -> tuple[List[ToolCallSample], Dict[str, int | str]]:
    spec = get_shard_spec(len(samples), shard_size, shard_index)
    start = int(spec["start_index"])
    end = int(spec["end_index_exclusive"])
    return list(samples[start:end]), spec


def shard_output_root(out_root: Path, direction: str, shard_spec: Dict[str, int | str]) -> Path:
    return shards_root(out_root, direction) / str(shard_spec["name"])


def persist_graph(graph: Graph, out_root: Path, stem: str) -> None:
    out_root.mkdir(parents=True, exist_ok=True)
    graph.to_pt(str(out_root / f"{stem}.pt"))
    graph.to_json(str(out_root / f"{stem}.json"))


def load_model_for_eap(args) -> tuple[object, object]:
    dtype = getattr(torch, str(args.dtype))
    model, tokenizer = load_hooked_qwen3(
        args.model_path,
        device=args.device,
        dtype=dtype,
        use_attn_result=True,
        use_split_qkv_input=True,
        use_hook_mlp_in=True,
        ungroup_gqa=True,
    )
    for param in model.parameters():
        param.requires_grad_(False)
    model.zero_grad(set_to_none=True)
    return model, tokenizer


def clear_runtime_state(model=None) -> None:
    if model is not None:
        try:
            model.reset_hooks()
        except Exception:
            pass
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def evaluate_baselines_with_metric(model, dataloader: DataLoader) -> Dict[str, torch.Tensor]:
    clean_scores: List[torch.Tensor] = []
    corrupted_scores: List[torch.Tensor] = []
    for clean, corrupted, label in tqdm(dataloader, desc="Baseline", dynamic_ncols=True):
        clean_tokens, attention_mask, input_lengths, _ = tokenize_plus(model, clean)
        corrupted_tokens, _, _, _ = tokenize_plus(model, corrupted)
        with torch.inference_mode():
            clean_logits = model(clean_tokens, attention_mask=attention_mask)
            corrupted_logits = model(corrupted_tokens, attention_mask=attention_mask)
            clean_score = negative_kl_to_clean_endpoint(clean_logits, clean_logits, input_lengths, label).detach().cpu()
            corrupted_score = negative_kl_to_clean_endpoint(corrupted_logits, clean_logits, input_lengths, label).detach().cpu()
        clean_scores.append(clean_score)
        corrupted_scores.append(corrupted_score)
    return {
        "clean": torch.cat(clean_scores) if clean_scores else torch.empty(0),
        "corrupted": torch.cat(corrupted_scores) if corrupted_scores else torch.empty(0),
    }


def summarize_score_tensor(name: str, values: torch.Tensor) -> Dict[str, object]:
    rows = tensor_to_list(values)
    return {
        "name": name,
        "n_samples": len(rows),
        "mean": finite_mean(rows),
        "median": finite_median(rows),
        "min": min(rows) if rows else float("nan"),
        "max": max(rows) if rows else float("nan"),
    }


def recovery_values(circuit_scores: torch.Tensor, clean_scores: torch.Tensor, corrupted_scores: torch.Tensor) -> List[float]:
    out: List[float] = []
    for circuit, clean, corrupted in zip(
        circuit_scores.detach().cpu().tolist(),
        clean_scores.detach().cpu().tolist(),
        corrupted_scores.detach().cpu().tolist(),
    ):
        gap = float(clean) - float(corrupted)
        if abs(gap) <= 1e-12:
            continue
        out.append((float(circuit) - float(corrupted)) / gap)
    return out


def discover_direction_graph(
    *,
    model,
    samples: Sequence[ToolCallSample],
    direction: str,
    batch_size: int,
    ig_steps: int,
    out_root: Path,
) -> Graph:
    dataloader = build_dataloader(samples=samples, direction=direction, batch_size=batch_size)
    graph = Graph.from_model(model)
    attribute(
        model,
        graph,
        dataloader,
        metric=negative_kl_to_clean_endpoint,
        method="EAP-IG-inputs",
        ig_steps=int(ig_steps),
        quiet=False,
    )
    persist_graph(graph, out_root, "scored_graph")
    return graph


def sweep_topn_on_split(
    *,
    model,
    scored_graph: Graph,
    samples: Sequence[ToolCallSample],
    direction: str,
    batch_size: int,
    topn_values: Sequence[int],
    out_root: Path,
    selection_method: str,
) -> List[Dict[str, object]]:
    baseline_loader = build_dataloader(samples=samples, direction=direction, batch_size=batch_size)
    baselines = evaluate_baselines_with_metric(model, baseline_loader)
    clean_scores = baselines["clean"]
    corrupted_scores = baselines["corrupted"]

    rows: List[Dict[str, object]] = []
    clean_list = tensor_to_list(clean_scores)
    corrupted_list = tensor_to_list(corrupted_scores)
    for topn in topn_values:
        graph = clone_graph(scored_graph, with_scores=True, with_in_graph=False)
        if selection_method == "greedy":
            graph.apply_greedy(int(topn), absolute=True, reset=True, prune=True)
        elif selection_method == "topn":
            graph.apply_topn(int(topn), absolute=True, level="edge", reset=True, prune=True)
        else:
            raise ValueError(f"Unknown selection_method: {selection_method}")
        dataloader = build_dataloader(samples=samples, direction=direction, batch_size=batch_size)
        circuit_scores = evaluate_graph(
            model,
            graph,
            dataloader,
            negative_kl_to_clean_endpoint,
            quiet=False,
            intervention="patching",
            skip_clean=False,
        )
        recovery = recovery_values(circuit_scores, clean_scores, corrupted_scores)
        rows.append(
            {
                "topn_edges": int(topn),
                "n_edges": int(graph.count_included_edges()),
                "n_nodes": int(graph.count_included_nodes()),
                "weighted_edge_count": float(graph.weighted_edge_count()),
                "clean_mean": finite_mean(clean_list),
                "corrupted_mean": finite_mean(corrupted_list),
                "circuit_mean": finite_mean(tensor_to_list(circuit_scores)),
                "clean_median": finite_median(clean_list),
                "corrupted_median": finite_median(corrupted_list),
                "circuit_median": finite_median(tensor_to_list(circuit_scores)),
                "recovery_mean": finite_mean(recovery),
                "recovery_median": finite_median(recovery),
            }
        )

    write_csv(out_root / "topn_sweep.csv", rows)
    write_json(out_root / "topn_sweep.json", rows)
    write_json(
        out_root / "baseline_summary.json",
        {
            "clean": summarize_score_tensor("clean", clean_scores),
            "corrupted": summarize_score_tensor("corrupted", corrupted_scores),
        },
    )
    return rows


def select_topn(rows: Sequence[Dict[str, object]], recovery_target: float) -> Dict[str, object]:
    if not rows:
        raise ValueError("No sweep rows to select from.")
    rows = sorted(rows, key=lambda r: int(r["topn_edges"]))
    nonempty_rows = [r for r in rows if int(r["n_edges"]) > 0]
    candidate_rows = nonempty_rows or list(rows)
    best_recovery = max(float(r["recovery_mean"]) for r in candidate_rows if math.isfinite(float(r["recovery_mean"])))
    threshold = min(float(recovery_target), best_recovery)
    eligible = [
        r
        for r in candidate_rows
        if math.isfinite(float(r["recovery_mean"])) and float(r["recovery_mean"]) >= threshold
    ]
    if eligible:
        return min(eligible, key=lambda r: int(r["topn_edges"]))
    return max(candidate_rows, key=lambda r: float(r["recovery_mean"]))


def materialize_selected_graph(
    *,
    scored_graph: Graph,
    selected_topn: int,
    out_root: Path,
    selection_method: str,
) -> Graph:
    graph = clone_graph(scored_graph, with_scores=True, with_in_graph=False)
    if selection_method == "greedy":
        graph.apply_greedy(int(selected_topn), absolute=True, reset=True, prune=True)
    elif selection_method == "topn":
        graph.apply_topn(int(selected_topn), absolute=True, level="edge", reset=True, prune=True)
    else:
        raise ValueError(f"Unknown selection_method: {selection_method}")
    persist_graph(graph, out_root, "selected_circuit")
    graph.to_image(str(out_root / "selected_circuit.png"))
    return graph


def evaluate_selected_graph(
    *,
    model,
    graph: Graph,
    samples: Sequence[ToolCallSample],
    direction: str,
    batch_size: int,
    split_name: str,
    out_root: Path,
) -> Dict[str, object]:
    baseline_loader = build_dataloader(samples=samples, direction=direction, batch_size=batch_size)
    baselines = evaluate_baselines_with_metric(model, baseline_loader)
    dataloader = build_dataloader(samples=samples, direction=direction, batch_size=batch_size)
    circuit_scores = evaluate_graph(
        model,
        graph,
        dataloader,
        negative_kl_to_clean_endpoint,
        quiet=False,
        intervention="patching",
        skip_clean=False,
    )

    clean_scores = baselines["clean"]
    corrupted_scores = baselines["corrupted"]
    recovery = recovery_values(circuit_scores, clean_scores, corrupted_scores)
    per_sample_rows: List[Dict[str, object]] = []
    for idx, sample in enumerate(samples):
        if idx >= circuit_scores.numel():
            break
        per_sample_rows.append(
            {
                "sample_id": sample.sample_id,
                "sample_rank": sample.sample_rank,
                "clean_metric": float(clean_scores[idx].item()),
                "corrupted_metric": float(corrupted_scores[idx].item()),
                "circuit_metric": float(circuit_scores[idx].item()),
                "recovery_ratio": recovery[idx] if idx < len(recovery) else float("nan"),
            }
        )
    write_csv(out_root / f"{split_name}_per_sample.csv", per_sample_rows)
    summary = {
        "split": split_name,
        "n_samples": len(per_sample_rows),
        "n_edges": int(graph.count_included_edges()),
        "n_nodes": int(graph.count_included_nodes()),
        "clean_mean": finite_mean(tensor_to_list(clean_scores)),
        "corrupted_mean": finite_mean(tensor_to_list(corrupted_scores)),
        "circuit_mean": finite_mean(tensor_to_list(circuit_scores)),
        "clean_median": finite_median(tensor_to_list(clean_scores)),
        "corrupted_median": finite_median(tensor_to_list(corrupted_scores)),
        "circuit_median": finite_median(tensor_to_list(circuit_scores)),
        "recovery_mean": finite_mean(recovery),
        "recovery_median": finite_median(recovery),
        "per_sample_csv": str(out_root / f"{split_name}_per_sample.csv"),
    }
    write_json(out_root / f"{split_name}_summary.json", summary)
    return summary


def merged_graph_manifest_path(out_root: Path, direction: str) -> Path:
    return discovery_root(out_root, direction) / "merged_shards_manifest.json"


def merge_scored_graph_shards(
    *,
    direction: str,
    out_root: Path,
) -> tuple[Graph, Dict[str, object]]:
    shard_dirs = sorted([path for path in shards_root(out_root, direction).glob("shard_*") if path.is_dir()])
    if not shard_dirs:
        raise ValueError(f"No shard directories found under {shards_root(out_root, direction)}")

    total_samples = 0
    acc_scores = None
    merged_graph = None
    shard_rows: List[Dict[str, object]] = []
    expected_total_shards = None
    seen_indices = set()
    for shard_dir in shard_dirs:
        meta_path = shard_dir / "shard_meta.json"
        graph_path = shard_dir / "scored_graph.pt"
        if not meta_path.exists() or not graph_path.exists():
            raise ValueError(f"Missing shard artifacts in {shard_dir}")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        shard_index = int(meta["shard_index"])
        total_shards = int(meta["total_shards"])
        if expected_total_shards is None:
            expected_total_shards = total_shards
        elif total_shards != expected_total_shards:
            raise ValueError(f"Inconsistent total_shards in {meta_path}: {total_shards} vs {expected_total_shards}")
        seen_indices.add(shard_index)
        shard_graph = Graph.from_pt(str(graph_path))
        shard_n = int(meta["n_samples"])
        if shard_n <= 0:
            raise ValueError(f"Invalid shard n_samples={shard_n} in {meta_path}")
        if merged_graph is None:
            merged_graph = clone_graph(shard_graph, with_scores=False, with_in_graph=False)
            acc_scores = torch.zeros_like(shard_graph.scores, dtype=torch.float64, device="cpu")
        acc_scores += shard_graph.scores.detach().cpu().to(torch.float64) * float(shard_n)
        total_samples += shard_n
        shard_rows.append(meta)

    if merged_graph is None or acc_scores is None or total_samples <= 0:
        raise ValueError(f"Failed to merge shards for direction={direction}")
    if expected_total_shards is None:
        raise ValueError(f"Failed to infer expected_total_shards for direction={direction}")
    if len(seen_indices) != expected_total_shards:
        raise ValueError(
            f"Shard set incomplete for direction={direction}: found {len(seen_indices)} / {expected_total_shards} shards"
        )

    merged_graph.scores[:] = (acc_scores / float(total_samples)).to(dtype=merged_graph.scores.dtype)
    merged_graph.in_graph[:] = False
    merged_graph.nodes_in_graph[:] = False

    out_dir = discovery_root(out_root, direction)
    persist_graph(merged_graph, out_dir, "scored_graph")
    manifest = {
        "direction": direction,
        "merge_mode": "weighted_mean_by_n_samples",
        "n_shards": len(shard_rows),
        "n_samples_total": total_samples,
        "scored_graph_pt": str(out_dir / "scored_graph.pt"),
        "scored_graph_json": str(out_dir / "scored_graph.json"),
        "shards": shard_rows,
    }
    write_json(merged_graph_manifest_path(out_root, direction), manifest)
    return merged_graph, manifest


def load_or_merge_scored_graph(
    *,
    direction: str,
    out_root: Path,
) -> tuple[Graph, Dict[str, object] | None]:
    graph_pt = discovery_root(out_root, direction) / "scored_graph.pt"
    manifest_path = merged_graph_manifest_path(out_root, direction)
    if graph_pt.exists():
        manifest = None
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        return Graph.from_pt(str(graph_pt)), manifest
    return merge_scored_graph_shards(direction=direction, out_root=out_root)


def finalize_direction_from_scored_graph(
    *,
    model,
    train_samples: Sequence[ToolCallSample],
    test_samples: Sequence[ToolCallSample],
    scored_graph: Graph,
    direction: str,
    eval_batch_size: int,
    topn_values: Sequence[int],
    recovery_target: float,
    out_root: Path,
    selection_method: str,
    discovery_manifest: Dict[str, object] | None = None,
) -> Dict[str, object]:
    sweep_root = out_root / "train_sweep"
    sweep_rows = sweep_topn_on_split(
        model=model,
        scored_graph=scored_graph,
        samples=train_samples,
        direction=direction,
        batch_size=eval_batch_size,
        topn_values=topn_values,
        out_root=sweep_root,
        selection_method=selection_method,
    )
    selected = select_topn(sweep_rows, recovery_target=recovery_target)

    selected_root = out_root / "selected_circuit"
    selected_graph = materialize_selected_graph(
        scored_graph=scored_graph,
        selected_topn=int(selected["topn_edges"]),
        out_root=selected_root,
        selection_method=selection_method,
    )

    train_summary = evaluate_selected_graph(
        model=model,
        graph=selected_graph,
        samples=train_samples,
        direction=direction,
        batch_size=eval_batch_size,
        split_name="train",
        out_root=out_root / "eval",
    )
    test_summary = evaluate_selected_graph(
        model=model,
        graph=selected_graph,
        samples=test_samples,
        direction=direction,
        batch_size=eval_batch_size,
        split_name="test",
        out_root=out_root / "eval",
    )

    discovery_artifacts = {
        "scored_graph_pt": str(discovery_root(out_root.parent, direction) / "scored_graph.pt"),
        "scored_graph_json": str(discovery_root(out_root.parent, direction) / "scored_graph.json"),
        "selected_circuit_pt": str(selected_root / "selected_circuit.pt"),
        "selected_circuit_json": str(selected_root / "selected_circuit.json"),
        "selected_circuit_png": str(selected_root / "selected_circuit.png"),
    }
    if discovery_manifest is not None:
        discovery_artifacts["merged_shards_manifest"] = str(merged_graph_manifest_path(out_root.parent, direction))

    payload = {
        "direction": direction,
        "selection_method": selection_method,
        "selected_topn": int(selected["topn_edges"]),
        "selected_row": selected,
        "discovery_artifacts": discovery_artifacts,
        "train_summary": train_summary,
        "test_summary": test_summary,
        "topn_sweep_csv": str(sweep_root / "topn_sweep.csv"),
        "topn_sweep_json": str(sweep_root / "topn_sweep.json"),
    }
    if discovery_manifest is not None:
        payload["discovery_manifest"] = discovery_manifest
    write_json(out_root / "direction_summary.json", payload)
    return payload


def run_direction_single_process(
    *,
    model,
    train_samples: Sequence[ToolCallSample],
    test_samples: Sequence[ToolCallSample],
    direction: str,
    attribution_batch_size: int,
    eval_batch_size: int,
    ig_steps: int,
    topn_values: Sequence[int],
    recovery_target: float,
    out_root: Path,
    selection_method: str,
) -> Dict[str, object]:
    discovery_dir = discovery_root(out_root, direction)
    scored_graph = discover_direction_graph(
        model=model,
        samples=train_samples,
        direction=direction,
        batch_size=attribution_batch_size,
        ig_steps=ig_steps,
        out_root=discovery_dir,
    )
    return finalize_direction_from_scored_graph(
        model=model,
        train_samples=train_samples,
        test_samples=test_samples,
        scored_graph=scored_graph,
        direction=direction,
        eval_batch_size=eval_batch_size,
        topn_values=topn_values,
        recovery_target=recovery_target,
        out_root=direction_root(out_root, direction),
        selection_method=selection_method,
        discovery_manifest=None,
    )


def build_summary_markdown(payloads: Sequence[Dict[str, object]]) -> str:
    lines = [
        "# Original EAP-IG Circuit Summary",
        "",
        "只保留原始 EAP-IG 发现和原始 EAP-IG `evaluate_graph` 指标。",
        "",
    ]
    for payload in payloads:
        train = payload["train_summary"]
        test = payload["test_summary"]
        lines.extend(
            [
                f"## {payload['direction']}",
                "",
                f"- selected top-n edges: `{payload['selected_topn']}`",
                f"- selected circuit size: `{train['n_nodes']}` nodes / `{train['n_edges']}` edges",
                f"- train raw metric mean: clean `{train['clean_mean']:.6f}`, corrupted `{train['corrupted_mean']:.6f}`, circuit `{train['circuit_mean']:.6f}`",
                f"- train recovery mean / median: `{train['recovery_mean']:.6f}` / `{train['recovery_median']:.6f}`",
                f"- test raw metric mean: clean `{test['clean_mean']:.6f}`, corrupted `{test['corrupted_mean']:.6f}`, circuit `{test['circuit_mean']:.6f}`",
                f"- test recovery mean / median: `{test['recovery_mean']:.6f}` / `{test['recovery_median']:.6f}`",
                f"- graph image: `{payload['discovery_artifacts']['selected_circuit_png']}`",
                "",
            ]
        )
    return "\n".join(lines) + "\n"


def build_common_cli_args(args) -> List[str]:
    return [
        "--train-dataset-root",
        str(args.train_dataset_root),
        "--test-dataset-root",
        str(args.test_dataset_root),
        "--model-path",
        str(args.model_path),
        "--device",
        str(args.device),
        "--dtype",
        str(args.dtype),
        "--ig-steps",
        str(args.ig_steps),
        "--attribution-batch-size",
        str(args.attribution_batch_size),
        "--eval-batch-size",
        str(args.eval_batch_size),
        "--topn-values",
        str(args.topn_values),
        "--selection-method",
        str(args.selection_method),
        "--recovery-target",
        str(args.recovery_target),
        "--max-train-samples",
        str(args.max_train_samples),
        "--max-test-samples",
        str(args.max_test_samples),
        "--out-root",
        str(args.out_root),
        "--shard-size",
        str(args.shard_size),
    ]


def run_discover_shard(args) -> Dict[str, object]:
    train_samples = build_samples(Path(args.train_dataset_root).resolve(), args.max_train_samples)
    shard_samples, shard_spec = get_shard_samples(
        train_samples,
        shard_size=int(args.shard_size),
        shard_index=int(args.shard_index),
    )
    shard_root = shard_output_root(Path(args.out_root).resolve(), args.direction, shard_spec)
    graph_path = shard_root / "scored_graph.pt"
    meta_path = shard_root / "shard_meta.json"
    if args.resume and graph_path.exists() and meta_path.exists():
        return json.loads(meta_path.read_text(encoding="utf-8"))

    model = None
    try:
        model, _ = load_model_for_eap(args)
        discover_direction_graph(
            model=model,
            samples=shard_samples,
            direction=args.direction,
            batch_size=args.attribution_batch_size,
            ig_steps=args.ig_steps,
            out_root=shard_root,
        )
    finally:
        clear_runtime_state(model)
        del model

    meta = {
        "direction": args.direction,
        "shard_index": int(shard_spec["shard_index"]),
        "total_shards": int(shard_spec["total_shards"]),
        "start_index": int(shard_spec["start_index"]),
        "end_index_exclusive": int(shard_spec["end_index_exclusive"]),
        "n_samples": len(shard_samples),
        "ig_steps": int(args.ig_steps),
        "attribution_batch_size": int(args.attribution_batch_size),
        "model_path": str(args.model_path),
        "dtype": str(args.dtype),
        "sample_ids": [sample.sample_id for sample in shard_samples],
        "scored_graph_pt": str(graph_path),
        "scored_graph_json": str(shard_root / "scored_graph.json"),
    }
    write_json(meta_path, meta)
    return meta


def run_merge_direction(args) -> Dict[str, object]:
    _, manifest = merge_scored_graph_shards(direction=args.direction, out_root=Path(args.out_root).resolve())
    return manifest


def run_finalize_direction(args) -> Dict[str, object]:
    train_samples = build_samples(Path(args.train_dataset_root).resolve(), args.max_train_samples)
    test_samples = build_samples(Path(args.test_dataset_root).resolve(), args.max_test_samples)
    topn_values = parse_topn_values(args.topn_values)
    scored_graph, manifest = load_or_merge_scored_graph(direction=args.direction, out_root=Path(args.out_root).resolve())

    model = None
    try:
        model, _ = load_model_for_eap(args)
        payload = finalize_direction_from_scored_graph(
            model=model,
            train_samples=train_samples,
            test_samples=test_samples,
            scored_graph=scored_graph,
            direction=args.direction,
            eval_batch_size=args.eval_batch_size,
            topn_values=topn_values,
            recovery_target=float(args.recovery_target),
            out_root=direction_root(Path(args.out_root).resolve(), args.direction),
            selection_method=args.selection_method,
            discovery_manifest=manifest,
        )
    finally:
        clear_runtime_state(model)
        del model
    return payload


def run_full_single_process(args) -> List[Dict[str, object]]:
    train_samples = build_samples(Path(args.train_dataset_root).resolve(), args.max_train_samples)
    test_samples = build_samples(Path(args.test_dataset_root).resolve(), args.max_test_samples)
    topn_values = parse_topn_values(args.topn_values)
    out_root = Path(args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    model = None
    payloads: List[Dict[str, object]] = []
    try:
        model, _ = load_model_for_eap(args)
        for direction in ("forward", "reverse"):
            payloads.append(
                run_direction_single_process(
                    model=model,
                    train_samples=train_samples,
                    test_samples=test_samples,
                    direction=direction,
                    attribution_batch_size=args.attribution_batch_size,
                    eval_batch_size=args.eval_batch_size,
                    ig_steps=args.ig_steps,
                    topn_values=topn_values,
                    recovery_target=float(args.recovery_target),
                    out_root=out_root,
                    selection_method=args.selection_method,
                )
            )
            clear_runtime_state(model)
    finally:
        clear_runtime_state(model)
        del model

    write_json(out_root / "summary.json", payloads)
    (out_root / "SUMMARY.md").write_text(build_summary_markdown(payloads), encoding="utf-8")
    return payloads


def run_full_sharded(args) -> List[Dict[str, object]]:
    out_root = Path(args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    train_samples = build_samples(Path(args.train_dataset_root).resolve(), args.max_train_samples)
    specs = shard_specs(len(train_samples), int(args.shard_size))
    write_json(
        out_root / "shard_plan.json",
        {
            "n_samples": len(train_samples),
            "shard_size": int(args.shard_size),
            "n_shards": len(specs),
            "shards": specs,
        },
    )

    base_args = build_common_cli_args(args)
    for direction in ("forward", "reverse"):
        for spec in specs:
            cmd = [
                sys.executable,
                str(RUNNER_PATH),
                "--mode",
                "discover_shard",
                "--direction",
                direction,
                "--shard-index",
                str(spec["shard_index"]),
            ] + base_args
            if args.resume:
                cmd.append("--resume")
            subprocess.run(cmd, check=True)

    payloads: List[Dict[str, object]] = []
    for direction in ("forward", "reverse"):
        finalize_args = argparse.Namespace(**vars(args))
        finalize_args.direction = direction
        payloads.append(run_finalize_direction(finalize_args))

    write_json(out_root / "summary.json", payloads)
    (out_root / "SUMMARY.md").write_text(build_summary_markdown(payloads), encoding="utf-8")
    return payloads


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Original EAP-IG circuit discovery and evaluation.")
    parser.add_argument("--mode", choices=sorted(MODES), default="full")
    parser.add_argument("--train-dataset-root", type=str, default=str(DATASETS_ROOT / "train"))
    parser.add_argument("--test-dataset-root", type=str, default=str(DATASETS_ROOT / "test"))
    parser.add_argument("--model-path", type=str, default=str(MODEL_PATH_DEFAULT))
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--ig-steps", type=int, default=2)
    parser.add_argument("--attribution-batch-size", type=int, default=1)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--topn-values", type=str, default="64,128,256,512")
    parser.add_argument("--selection-method", choices=["greedy", "topn"], default="greedy")
    parser.add_argument("--recovery-target", type=float, default=0.95)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-test-samples", type=int, default=0)
    parser.add_argument("--direction", choices=["forward", "reverse"], default="forward")
    parser.add_argument("--shard-size", type=int, default=64)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--out-root",
        type=str,
        default=str(RESULTS_ROOT / "eap-ig" / "original_eap_ig"),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.mode == "discover_shard":
        run_discover_shard(args)
        return
    if args.mode == "merge_direction":
        run_merge_direction(args)
        return
    if args.mode == "finalize_direction":
        run_finalize_direction(args)
        return
    if args.mode == "full_sharded":
        run_full_sharded(args)
        return
    if args.mode == "full":
        run_full_single_process(args)
        return
    raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
