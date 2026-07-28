#!/usr/bin/env python3
"""
Direction-level delivery-object intervention audit.

The goal is to distinguish:
- `L2H14`: earliest frame-heavy reader
- `MLP11`: first stable writer of a shared delivery-object axis

We build the same four prompts as `delivery_object_2x2_audit`, then:
1. define a shared file-vs-answer direction at each node from the write/develop pairs;
2. inject only that shared direction from file -> answer at `L2H14` or `MLP11`;
3. erase that shared direction from file prompts at `L2H14` or `MLP11`;
4. compare downstream node motion and final token/logit effects.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from tqdm.auto import tqdm

if __package__ in {None, ""}:
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from toolcall_circuit.bidirectional_causal_eval import load_sample_paths
from toolcall_circuit.delivery_object_2x2_audit import FRAMES, build_prompts
from toolcall_circuit.objective import build_bidirectional_endpoint_objectives
from toolcall_circuit.single_sample import load_hooked_qwen3, objective_from_logits


NODE_SPECS = {
    "L2H14": ("head", 2, 14),
    "MLP11": ("mlp", 11, None),
    "MLP16": ("mlp", 16, None),
    "MLP19": ("mlp", 19, None),
    "L20H5": ("head", 20, 5),
    "L21H1": ("head", 21, 1),
    "L21H12": ("head", 21, 12),
    "L24H6": ("head", 24, 6),
    "MLP27": ("mlp", 27, None),
}
INTERVENTION_NODES = ("L2H14", "MLP11")
SHARED_AXIS_NODES = tuple(NODE_SPECS.keys())


def finite(values: Iterable[float]) -> List[float]:
    return [float(v) for v in values if isinstance(v, (int, float)) and math.isfinite(float(v))]


def median(values: Iterable[float]) -> float:
    vals = finite(values)
    return float(np.median(vals)) if vals else float("nan")


def safe_rate(values: Iterable[bool]) -> float:
    vals = [1.0 if bool(v) else 0.0 for v in values]
    return float(np.mean(vals)) if vals else float("nan")


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(rows: Sequence[Dict[str, object]], out_path: Path) -> None:
    if not rows:
        return
    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def unit(vec: torch.Tensor) -> torch.Tensor:
    denom = float(vec.norm().item())
    if denom < 1e-8:
        return torch.zeros_like(vec)
    return vec / denom


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    ua = unit(a)
    ub = unit(b)
    if float(ua.norm().item()) < 1e-8 or float(ub.norm().item()) < 1e-8:
        return float("nan")
    return float(torch.dot(ua, ub).item())


def extract_node(cache: Dict[str, torch.Tensor], node: str) -> torch.Tensor:
    kind, layer, head = NODE_SPECS[node]
    if kind == "mlp":
        return cache[f"blocks.{layer}.hook_mlp_out"][0, -1, :].float()
    return cache[f"blocks.{layer}.attn.hook_z"][0, -1, int(head), :].float()


def collect_cache(model, tokens: torch.Tensor, names: Sequence[str]) -> Dict[str, torch.Tensor]:
    wanted = set(names)
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens, names_filter=lambda n: n in wanted)
    return {k: v.detach().cpu() for k, v in cache.items()}


def collect_names() -> List[str]:
    names: List[str] = []
    for kind, layer, _head in NODE_SPECS.values():
        if kind == "mlp":
            names.append(f"blocks.{layer}.hook_mlp_out")
        else:
            names.append(f"blocks.{layer}.attn.hook_z")
    return sorted(set(names))


def build_global_direction(file_write: torch.Tensor, answer_write: torch.Tensor, file_develop: torch.Tensor, answer_develop: torch.Tensor) -> Tuple[torch.Tensor, float]:
    write_dir = file_write - answer_write
    develop_dir = file_develop - answer_develop
    global_dir = unit(unit(write_dir) + unit(develop_dir))
    return global_dir, cosine(write_dir, develop_dir)


def projection_delta(base_vec: torch.Tensor, source_vec: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
    if float(direction.norm().item()) < 1e-8:
        return torch.zeros_like(base_vec)
    d = unit(direction)
    return (torch.dot(source_vec, d) - torch.dot(base_vec, d)) * d


def erase_to_pair_center(base_vec: torch.Tensor, center_vec: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
    if float(direction.norm().item()) < 1e-8:
        return torch.zeros_like(base_vec)
    d = unit(direction)
    return (torch.dot(center_vec, d) - torch.dot(base_vec, d)) * d


def run_with_edit_and_collect(
    model,
    base_tokens: torch.Tensor,
    *,
    edit_node: str,
    edit_delta: torch.Tensor,
    record_names: Sequence[str],
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    kind, layer, head = NODE_SPECS[edit_node]
    hooks = []

    if kind == "mlp":
        cache_name = f"blocks.{layer}.hook_mlp_out"
        delta = edit_delta.to(base_tokens.device)

        def edit_hook(mlp_out: torch.Tensor, hook, delta_vec=delta):  # noqa: ANN001
            out = mlp_out.clone()
            out[:, -1, :] = (out[:, -1, :].float() + delta_vec.unsqueeze(0)).to(dtype=out.dtype)
            return out

        hooks.append((cache_name, edit_hook))
    else:
        cache_name = f"blocks.{layer}.attn.hook_z"
        delta = edit_delta.to(base_tokens.device)

        def edit_hook(z: torch.Tensor, hook, delta_vec=delta):  # noqa: ANN001
            out = z.clone()
            out[:, -1, int(head), :] = (out[:, -1, int(head), :].float() + delta_vec.unsqueeze(0)).to(dtype=out.dtype)
            return out

        hooks.append((cache_name, edit_hook))

    recorded: Dict[str, torch.Tensor] = {}
    for name in record_names:
        def make_record(cache_name: str):
            def record_hook(act: torch.Tensor, hook):  # noqa: ANN001
                recorded[cache_name] = act.detach().cpu()
                return act
            return record_hook

        hooks.append((name, make_record(name)))

    with torch.no_grad():
        logits = model.run_with_hooks(base_tokens, fwd_hooks=hooks)
    return logits, recorded


def main() -> None:
    parser = argparse.ArgumentParser(description="Direction-level delivery-object intervention audit.")
    parser.add_argument("--run-root", type=str, required=True)
    parser.add_argument("--model-path", type=str, default="./external/models/Qwen3-1.7B")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--output-root", type=str, required=True)
    parser.add_argument("--save-every", type=int, default=25)
    args = parser.parse_args()

    run_root = Path(args.run_root).resolve()
    out_root = Path(args.output_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    samples = load_sample_paths(run_root / "forward_batch", run_root / "reverse_batch", max_samples=args.max_samples)
    model, tokenizer = load_hooked_qwen3(args.model_path, device=args.device, dtype=torch.bfloat16)
    hook_names = collect_names()

    direction_path = out_root / "delivery_object_direction_alignment_per_sample.csv"
    intervention_path = out_root / "delivery_object_direction_intervention_per_sample.csv"
    direction_rows: List[Dict[str, object]] = list(read_csv_rows(direction_path))
    intervention_rows: List[Dict[str, object]] = list(read_csv_rows(intervention_path))
    processed_ids = {
        str(row["sample_id"])
        for row in intervention_rows
        if sum(1 for r in intervention_rows if str(r["sample_id"]) == str(row["sample_id"])) == len(INTERVENTION_NODES) * len(FRAMES) * 2
    }

    def checkpoint() -> None:
        write_csv(direction_rows, direction_path)
        write_csv(intervention_rows, intervention_path)

    pbar = tqdm(samples, desc="Delivery object direction", dynamic_ncols=True)
    completed_new = 0
    for sp in pbar:
        if sp.sample_id in processed_ids:
            pbar.set_postfix(sample=sp.sample_id, resumed="skip")
            continue
        try:
            clean_text = sp.tool_call_prompt.read_text(encoding="utf-8")
            prompts = build_prompts(clean_text)
        except Exception:
            continue

        token_map: Dict[str, torch.Tensor] = {}
        logits_map: Dict[str, torch.Tensor] = {}
        cache_map: Dict[str, Dict[str, torch.Tensor]] = {}
        node_vecs: Dict[str, Dict[str, torch.Tensor]] = defaultdict(dict)

        for frame in FRAMES:
            for delivery_object in ("file", "answer"):
                name = f"{frame['frame']}_{delivery_object}"
                text = prompts[name]
                tokens = model.to_tokens(text, prepend_bos=False)
                token_map[name] = tokens
                with torch.no_grad():
                    logits_map[name] = model(tokens)
                cache_map[name] = collect_cache(model, tokens, hook_names)
                for node in SHARED_AXIS_NODES:
                    node_vecs[node][name] = extract_node(cache_map[name], node)

        frame_objectives: Dict[str, Tuple[object, object]] = {}
        for frame in FRAMES:
            frame_name = frame["frame"]
            frame_objectives[frame_name] = build_bidirectional_endpoint_objectives(
                logits_map[f"{frame_name}_file"],
                logits_map[f"{frame_name}_answer"],
                tokenizer=tokenizer,
            )

        global_dirs: Dict[str, torch.Tensor] = {}
        for node in SHARED_AXIS_NODES:
            gdir, frame_cos = build_global_direction(
                node_vecs[node]["write_file"],
                node_vecs[node]["write_answer"],
                node_vecs[node]["develop_file"],
                node_vecs[node]["develop_answer"],
            )
            global_dirs[node] = gdir
            direction_rows.append(
                {
                    "sample_id": sp.sample_id,
                    "node": node,
                    "write_vs_develop_direction_cosine": frame_cos,
                    "global_direction_norm": float(gdir.norm().item()),
                    "write_delta_norm": float((node_vecs[node]["write_file"] - node_vecs[node]["write_answer"]).norm().item()),
                    "develop_delta_norm": float((node_vecs[node]["develop_file"] - node_vecs[node]["develop_answer"]).norm().item()),
                }
            )

        for frame in FRAMES:
            frame_name = frame["frame"]
            file_name = f"{frame_name}_file"
            answer_name = f"{frame_name}_answer"
            file_objective, answer_objective = frame_objectives[frame_name]

            base_answer_tokens = token_map[answer_name]
            base_file_tokens = token_map[file_name]
            base_answer_logits = logits_map[answer_name]
            base_file_logits = logits_map[file_name]
            base_answer_file_score = float(objective_from_logits(base_answer_logits, file_objective).item())
            base_answer_answer_score = float(objective_from_logits(base_answer_logits, answer_objective).item())
            base_file_file_score = float(objective_from_logits(base_file_logits, file_objective).item())
            base_file_answer_score = float(objective_from_logits(base_file_logits, answer_objective).item())

            for node in INTERVENTION_NODES:
                direction = global_dirs[node]
                file_vec = node_vecs[node][file_name]
                answer_vec = node_vecs[node][answer_name]
                center_vec = 0.5 * (file_vec + answer_vec)

                inject_delta = projection_delta(answer_vec, file_vec, direction)
                erase_delta = erase_to_pair_center(file_vec, center_vec, direction)

                for mode, base_tokens, base_name, edit_delta in [
                    ("inject_file_into_answer", base_answer_tokens, answer_name, inject_delta),
                    ("erase_file_component", base_file_tokens, file_name, erase_delta),
                ]:
                    logits, edited_cache = run_with_edit_and_collect(
                        model,
                        base_tokens,
                        edit_node=node,
                        edit_delta=edit_delta,
                        record_names=hook_names,
                    )
                    file_score = float(objective_from_logits(logits, file_objective).item())
                    answer_score = float(objective_from_logits(logits, answer_objective).item())
                    if mode == "inject_file_into_answer":
                        base_file_score = base_answer_file_score
                        base_answer_score = base_answer_answer_score
                    else:
                        base_file_score = base_file_file_score
                        base_answer_score = base_file_answer_score

                    row: Dict[str, object] = {
                        "sample_id": sp.sample_id,
                        "frame": frame_name,
                        "node": node,
                        "mode": mode,
                        "object_score_delta": (file_score - answer_score) - (base_file_score - base_answer_score),
                        "file_score_delta": file_score - base_file_score,
                        "answer_score_delta": answer_score - base_answer_score,
                        "object_boundary_flip": file_score > answer_score,
                        "tool_token_delta": float(logits[0, -1, sp.target_tool_call].item())
                        - float((base_answer_logits if mode == "inject_file_into_answer" else base_file_logits)[0, -1, sp.target_tool_call].item()),
                        "distractor_token_delta": float(logits[0, -1, sp.distractor].item())
                        - float((base_answer_logits if mode == "inject_file_into_answer" else base_file_logits)[0, -1, sp.distractor].item()),
                    }

                    base_cache_name = answer_name if mode == "inject_file_into_answer" else file_name
                    for downstream in SHARED_AXIS_NODES:
                        base_down = node_vecs[downstream][base_cache_name]
                        edited_down = extract_node(edited_cache, downstream)
                        down_dir = global_dirs[downstream]
                        if float(down_dir.norm().item()) < 1e-8:
                            row[f"{downstream}_projection_delta"] = float("nan")
                        else:
                            row[f"{downstream}_projection_delta"] = float(torch.dot(edited_down - base_down, unit(down_dir)).item())
                    intervention_rows.append(row)

        processed_ids.add(sp.sample_id)
        completed_new += 1
        pbar.set_postfix(sample=sp.sample_id)
        if args.save_every > 0 and completed_new % args.save_every == 0:
            checkpoint()

    checkpoint()

    direction_summary: List[Dict[str, object]] = []
    by_node: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in direction_rows:
        by_node[str(row["node"])].append(row)
    for node in SHARED_AXIS_NODES:
        rows = by_node.get(node, [])
        if not rows:
            continue
        direction_summary.append(
            {
                "node": node,
                "write_vs_develop_direction_cosine_median": median(float(r["write_vs_develop_direction_cosine"]) for r in rows),
                "global_direction_norm_median": median(float(r["global_direction_norm"]) for r in rows),
                "write_delta_norm_median": median(float(r["write_delta_norm"]) for r in rows),
                "develop_delta_norm_median": median(float(r["develop_delta_norm"]) for r in rows),
            }
        )

    intervention_summary: List[Dict[str, object]] = []
    grouped: Dict[Tuple[str, str, str], List[Dict[str, object]]] = defaultdict(list)
    downstream_cols = [f"{node}_projection_delta" for node in SHARED_AXIS_NODES]
    for row in intervention_rows:
        grouped[(str(row["frame"]), str(row["node"]), str(row["mode"]))].append(row)
    for (frame_name, node, mode), rows in sorted(grouped.items()):
        out_row: Dict[str, object] = {
            "frame": frame_name,
            "node": node,
            "mode": mode,
            "n_samples": len(rows),
            "object_score_delta_median": median(float(r["object_score_delta"]) for r in rows),
            "file_score_delta_median": median(float(r["file_score_delta"]) for r in rows),
            "answer_score_delta_median": median(float(r["answer_score_delta"]) for r in rows),
            "object_boundary_flip_rate": safe_rate(bool(r["object_boundary_flip"]) for r in rows),
            "tool_token_delta_median": median(float(r["tool_token_delta"]) for r in rows),
            "distractor_token_delta_median": median(float(r["distractor_token_delta"]) for r in rows),
        }
        for col in downstream_cols:
            out_row[f"{col}_median"] = median(float(r[col]) for r in rows)
        intervention_summary.append(out_row)

    write_csv(direction_summary, out_root / "delivery_object_direction_alignment_summary.csv")
    write_csv(intervention_summary, out_root / "delivery_object_direction_intervention_summary.csv")

    lines = ["# Delivery Object Direction Intervention", ""]
    lines.append("## Shared Direction Alignment")
    lines.append("")
    for row in direction_summary:
        lines.append(
            f"- `{row['node']}`: write-vs-develop cosine `{row['write_vs_develop_direction_cosine_median']:.3f}`, global-dir norm `{row['global_direction_norm_median']:.3f}`."
        )
    lines.append("")
    lines.append("## Direction-Level Intervention")
    lines.append("")
    for row in intervention_summary:
        lines.append(
            f"- frame `{row['frame']}` / node `{row['node']}` / mode `{row['mode']}`: object-delta `{row['object_score_delta_median']:.3f}`, boundary `{row['object_boundary_flip_rate']:.3f}`, tool-logit `{row['tool_token_delta_median']:.3f}`, distractor-logit `{row['distractor_token_delta_median']:.3f}`, MLP16 `{row['MLP16_projection_delta_median']:.3f}`, MLP19 `{row['MLP19_projection_delta_median']:.3f}`, L20H5 `{row['L20H5_projection_delta_median']:.3f}`, L21H12 `{row['L21H12_projection_delta_median']:.3f}`, L24H6 `{row['L24H6_projection_delta_median']:.3f}`, MLP27 `{row['MLP27_projection_delta_median']:.3f}`."
        )
    lines.append("")
    lines.append("## Bottom Line")
    lines.append("")
    lines.append("If `MLP11` carries a cross-frame file-vs-answer direction and editing only that direction moves `MLP16 -> MLP19 -> late tool route`, while `L2H14` does not, then `MLP11` is the first stable delivery-object writer and `L2H14` is not.")
    (out_root / "delivery_object_direction_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    summary = {
        "direction_summary_rows": direction_summary,
        "intervention_summary_rows": intervention_summary,
        "artifacts": {
            "alignment_per_sample_csv": str(direction_path),
            "intervention_per_sample_csv": str(intervention_path),
            "report_md": str(out_root / "delivery_object_direction_report.md"),
        },
    }
    (out_root / "delivery_object_direction_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
