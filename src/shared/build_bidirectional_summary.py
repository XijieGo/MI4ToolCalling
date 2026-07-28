#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


def fmt_nodes(nodes: Sequence[str]) -> str:
    return ", ".join(nodes) if nodes else "(none)"


def fmt_edges(edges: Sequence[Sequence[str] | Tuple[str, str]]) -> str:
    normalized = []
    for edge in edges:
        if isinstance(edge, tuple):
            src, dst = edge
        else:
            src, dst = str(edge[0]), str(edge[1])
        normalized.append(f"{src} -> {dst}")
    return "; ".join(normalized) if normalized else "(none)"


def load_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def edge_set(edges: Iterable[Sequence[str]]) -> set[Tuple[str, str]]:
    return {(str(edge[0]), str(edge[1])) for edge in edges}


def intersection_many(items: Sequence[set]) -> set:
    if not items:
        return set()
    acc = set(items[0])
    for item in items[1:]:
        acc &= set(item)
    return acc


def difference_against_others(named_sets: Dict[str, set]) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for name, values in named_sets.items():
        others = set()
        for other_name, other_values in named_sets.items():
            if other_name == name:
                continue
            others |= set(other_values)
        only_here = sorted(values - others)
        if only_here:
            out[name] = only_here
    return out


def difference_against_others_edges(named_sets: Dict[str, set[Tuple[str, str]]]) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for name, values in named_sets.items():
        others: set[Tuple[str, str]] = set()
        for other_name, other_values in named_sets.items():
            if other_name == name:
                continue
            others |= set(other_values)
        only_here = sorted(values - others)
        if only_here:
            out[name] = [f"{src} -> {dst}" for src, dst in only_here]
    return out


def node_layer(node: str) -> int:
    if node.startswith("MLP"):
        return int(node[3:])
    match = re.fullmatch(r"L(\d+)H(\d+)", node)
    if match:
        return int(match.group(1))
    if node == "Input Embed":
        return -1
    raise ValueError(f"Unknown node format: {node}")


def layer_span(nodes: Sequence[str]) -> str:
    if not nodes:
        return "(none)"
    layers = sorted(node_layer(node) for node in nodes)
    return f"L{layers[0]}..L{layers[-1]}"


def tail_nodes(nodes: Sequence[str], *, count: int = 4) -> List[str]:
    return sorted(nodes, key=lambda node: (node_layer(node), node))[-count:]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a concise three-model bidirectional summary page.")
    parser.add_argument("--results-root", type=Path, default=Path("./results"))
    parser.add_argument("--output-path", type=Path, default=Path("./results/BIDIRECTIONAL_MODEL_SUMMARY.md"))
    parser.add_argument("--dataset-root", type=Path, default=Path("./datasets/train"))
    parser.add_argument("--model-label", action="append", dest="model_labels", default=[])
    args = parser.parse_args()

    model_labels = args.model_labels or ["1.7B", "4B", "8B"]

    per_model: Dict[str, Dict[str, object]] = {}
    for label in model_labels:
        model_root = (args.results_root / label).resolve()
        bidirectional = load_json(model_root / "bidirectional" / "bidirectional_summary.json")
        forward = load_json(model_root / "forward_aggregate" / "global_core_summary.json")
        reverse = load_json(model_root / "reverse_aggregate" / "global_core_summary.json")
        per_model[label] = {
            "root": model_root,
            "bidirectional": bidirectional,
            "forward": forward,
            "reverse": reverse,
        }

    shared_nodes_by_model: Dict[str, set[str]] = {}
    forward_nodes_by_model: Dict[str, set[str]] = {}
    reverse_nodes_by_model: Dict[str, set[str]] = {}
    shared_edges_by_model: Dict[str, set[Tuple[str, str]]] = {}
    forward_edges_by_model: Dict[str, set[Tuple[str, str]]] = {}
    reverse_edges_by_model: Dict[str, set[Tuple[str, str]]] = {}

    dataset_root = args.dataset_root.resolve()
    merge_summary_path = dataset_root / "merge_summary.json"
    merge_summary = load_json(merge_summary_path) if merge_summary_path.exists() else {}
    clean_rows = merge_summary.get("clean_rows")
    corrupt_rows = merge_summary.get("corrupt_rows")

    lines: List[str] = ["# Bidirectional Circuit Summary", ""]
    lines.extend(
        [
            f"- dataset: `{dataset_root}`",
            f"- dataset rows: clean = {clean_rows}, corrupt = {corrupt_rows}",
            "- note: this summary is rebuilt from the current full rerun outputs under `./results` using the legacy five-stage bidirectional pipeline.",
            "",
        ]
    )

    for label in model_labels:
        payload = per_model[label]
        bidirectional = payload["bidirectional"]
        support = bidirectional["support_analysis"]
        forward = payload["forward"]
        reverse = payload["reverse"]

        shared_nodes = list(support.get("shared_backbone_nodes", []))
        shared_edges = list(support.get("shared_backbone_edges", []))
        forward_nodes = list(support.get("forward_selective_nodes", []))
        forward_edges = list(support.get("forward_selective_edges", []))
        reverse_nodes = list(support.get("reverse_selective_nodes", []))
        reverse_edges = list(support.get("reverse_selective_edges", []))

        shared_nodes_by_model[label] = set(shared_nodes)
        forward_nodes_by_model[label] = set(forward_nodes)
        reverse_nodes_by_model[label] = set(reverse_nodes)
        shared_edges_by_model[label] = edge_set(shared_edges)
        forward_edges_by_model[label] = edge_set(forward_edges)
        reverse_edges_by_model[label] = edge_set(reverse_edges)

        lines.extend(
            [
                f"## {label}",
                f"- shared backbone: nodes = {fmt_nodes(shared_nodes)}; edges = {fmt_edges(shared_edges)}",
                f"- forward-selective 节点 / 边: nodes = {fmt_nodes(forward_nodes)}; edges = {fmt_edges(forward_edges)}",
                f"- reverse-selective 节点 / 边: nodes = {fmt_nodes(reverse_nodes)}; edges = {fmt_edges(reverse_edges)}",
                f"- discovery 样本数: forward = {forward.get('n_samples')}, reverse = {reverse.get('n_samples')}",
                "",
            ]
        )

    lines.append("## 三模型稳定 / 差异")
    lines.append(
        "- 模式级稳定结构: 三个模型都分成 `shared backbone`、`forward-selective`、`reverse-selective` 三块，而且 forward / reverse 两支都会在靠输出的晚层节点收束。"
    )
    lines.append(
        "- 模式级差异: 1.7B 和 4B 都保留较多 attention-head + MLP 混合路径；8B 更明显地转成 MLP 主导，只在 shared backbone 里保留一组最早层的 shared heads。"
    )
    lines.append(
        "- shared backbone 层跨度: "
        + "; ".join(
            f"{label} = {layer_span(sorted(shared_nodes_by_model[label]))}" for label in model_labels
        )
    )
    lines.append(
        "- forward-selective 晚层尾部: "
        + "; ".join(
            f"{label} = {fmt_nodes(tail_nodes(sorted(forward_nodes_by_model[label])))}" for label in model_labels
        )
    )
    lines.append(
        "- reverse-selective 晚层尾部: "
        + "; ".join(
            f"{label} = {fmt_nodes(tail_nodes(sorted(reverse_nodes_by_model[label])))}" for label in model_labels
        )
    )
    lines.append(
        f"- shared backbone 稳定节点: {fmt_nodes(sorted(intersection_many(list(shared_nodes_by_model.values()))))}"
    )
    lines.append(
        f"- shared backbone 稳定边: {fmt_edges(sorted(intersection_many(list(shared_edges_by_model.values()))))}"
    )
    lines.append(
        f"- forward-selective 稳定节点: {fmt_nodes(sorted(intersection_many(list(forward_nodes_by_model.values()))))}"
    )
    lines.append(
        f"- forward-selective 稳定边: {fmt_edges(sorted(intersection_many(list(forward_edges_by_model.values()))))}"
    )
    lines.append(
        f"- reverse-selective 稳定节点: {fmt_nodes(sorted(intersection_many(list(reverse_nodes_by_model.values()))))}"
    )
    lines.append(
        f"- reverse-selective 稳定边: {fmt_edges(sorted(intersection_many(list(reverse_edges_by_model.values()))))}"
    )

    shared_node_only = difference_against_others(shared_nodes_by_model)
    shared_edge_only = difference_against_others_edges(shared_edges_by_model)
    forward_node_only = difference_against_others(forward_nodes_by_model)
    forward_edge_only = difference_against_others_edges(forward_edges_by_model)
    reverse_node_only = difference_against_others(reverse_nodes_by_model)
    reverse_edge_only = difference_against_others_edges(reverse_edges_by_model)

    for label in model_labels:
        chunks: List[str] = []
        if label in shared_node_only:
            chunks.append(f"shared-only 节点 = {fmt_nodes(shared_node_only[label])}")
        if label in shared_edge_only:
            chunks.append(f"shared-only 边 = {'; '.join(shared_edge_only[label])}")
        if label in forward_node_only:
            chunks.append(f"forward-only 节点 = {fmt_nodes(forward_node_only[label])}")
        if label in forward_edge_only:
            chunks.append(f"forward-only 边 = {'; '.join(forward_edge_only[label])}")
        if label in reverse_node_only:
            chunks.append(f"reverse-only 节点 = {fmt_nodes(reverse_node_only[label])}")
        if label in reverse_edge_only:
            chunks.append(f"reverse-only 边 = {'; '.join(reverse_edge_only[label])}")
        if not chunks:
            chunks.append("没有只出现在该模型里的显著 overlay 结构。")
        lines.append(f"- {label}: {'; '.join(chunks)}")

    output_path = args.output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(output_path)


if __name__ == "__main__":
    main()
