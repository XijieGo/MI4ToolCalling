from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch

INPUT_NODE = "Input Embed"
DEFAULT_OUTPUT_NODE = "Residual Output: <tool_call>"


def node_layer(node_name: str) -> int:
    if node_name.startswith("MLP"):
        return int(node_name[3:])
    m = re.fullmatch(r"L(\d+)H(\d+)", node_name)
    if m:
        return int(m.group(1))
    raise ValueError(f"Unknown node name format: {node_name}")


def apply_plot_style() -> None:
    plt.style.use("default")
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 16,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        }
    )


def remap_output_node(
    edges: Sequence[Tuple[str, str]],
    *,
    source_output: str = DEFAULT_OUTPUT_NODE,
    target_output: str,
) -> list[Tuple[str, str]]:
    out: list[Tuple[str, str]] = []
    for src, dst in edges:
        out.append((src, target_output if dst == source_output else dst))
    return out


def draw_circuit_with_output(
    nodes: Sequence[str],
    edges: Sequence[Tuple[str, str]],
    out_path: Path,
    title: str,
    *,
    input_node: str = INPUT_NODE,
    output_node: str = DEFAULT_OUTPUT_NODE,
) -> None:
    all_nodes = [input_node] + list(nodes) + [output_node]

    if nodes:
        min_l = min(node_layer(n) for n in nodes)
        max_l = max(node_layer(n) for n in nodes)
    else:
        min_l = 0
        max_l = 1

    layer_map = {input_node: min_l - 2, output_node: max_l + 2}
    for n in nodes:
        layer_map[n] = node_layer(n)

    by_layer: Dict[int, list[str]] = {}
    for n in all_nodes:
        by_layer.setdefault(layer_map[n], []).append(n)

    pos: Dict[str, Tuple[float, float]] = {}
    for layer in sorted(by_layer):
        group = sorted(by_layer[layer], key=lambda n: (0 if n.startswith("MLP") else 1, n))
        k = len(group)
        span = 2.2
        xs = [0.0] if k == 1 else np.linspace(-span * (k - 1) / 2, span * (k - 1) / 2, k)
        for x, n in zip(xs, group):
            pos[n] = (float(x), float(layer))

    apply_plot_style()
    fig, ax = plt.subplots(figsize=(8.6, 11.0), constrained_layout=True)
    ax.set_title(title)
    ax.axis("off")

    edge_color = "#8a4f28"
    for src, dst in edges:
        x1, y1 = pos[src]
        x2, y2 = pos[dst]
        rad = 0.09 if abs(x2 - x1) > 0.2 else 0.0
        arrow = FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle="->",
            mutation_scale=14,
            linewidth=1.7,
            color=edge_color,
            connectionstyle=f"arc3,rad={rad}",
            zorder=1,
        )
        ax.add_patch(arrow)

    for n in all_nodes:
        x, y = pos[n]
        if n in {input_node, output_node}:
            fc = "#bcd3ea"
            ec = "#bcd3ea"
            size = 240
        else:
            fc = "#f5ede7"
            ec = "#2e2e2e"
            size = 200
        ax.scatter([x], [y], s=size, c=fc, edgecolors=ec, linewidths=1.6, zorder=3)
        label = n
        ax.text(
            x + 0.34,
            y,
            label,
            va="center",
            ha="left",
            fontsize=11 if n not in {input_node, output_node} else 13,
            zorder=4,
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.2, "alpha": 0.85},
        )

    ys = [p[1] for p in pos.values()]
    ax.set_ylim(min(ys) - 1.0, max(ys) + 1.3)
    ax.set_xlim(-6.8, 7.8)
    fig.savefig(out_path, dpi=280, bbox_inches="tight")
    plt.close(fig)
