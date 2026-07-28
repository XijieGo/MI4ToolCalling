#!/usr/bin/env python3
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

from torch.utils.data import DataLoader, Dataset

from toolcall_circuit.dataset import (
    ToolCallSample,
    load_toolcall_samples,
    parse_legacy_index_list,
    parse_sample_id_list,
    save_sample_catalog,
    select_samples,
)


@dataclass(frozen=True)
class DirectionSpec:
    direction: str
    clean_text: str
    corrupt_text: str
    clean_prompt_path: Path
    corrupt_prompt_path: Path
    clean_role: str
    corrupt_role: str
    objective_endpoint: str
    output_node_label: str
    decision_label: str


class SinglePairDataset(Dataset):
    def __init__(self, clean_text: str, corrupt_text: str, label: Dict[str, object]) -> None:
        self.clean_text = clean_text
        self.corrupt_text = corrupt_text
        self.label = label

    def __len__(self) -> int:
        return 1

    def __getitem__(self, idx: int):
        if idx != 0:
            raise IndexError(idx)
        return self.clean_text, self.corrupt_text, self.label


def single_pair_collate(rows):
    clean, corrupt, labels = zip(*rows)
    return list(clean), list(corrupt), list(labels)


def build_single_pair_dataloader(clean_text: str, corrupt_text: str, label: Dict[str, object]) -> DataLoader:
    dataset = SinglePairDataset(clean_text=clean_text, corrupt_text=corrupt_text, label=label)
    return DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=single_pair_collate)


def resolve_direction_spec(sample: ToolCallSample, direction: str) -> DirectionSpec:
    tool_text = sample.clean_path.read_text(encoding="utf-8")
    no_tool_text = sample.corrupt_path.read_text(encoding="utf-8")
    if direction == "forward":
        return DirectionSpec(
            direction="forward",
            clean_text=tool_text,
            corrupt_text=no_tool_text,
            clean_prompt_path=sample.clean_path,
            corrupt_prompt_path=sample.corrupt_path,
            clean_role="tool_call",
            corrupt_role="no_tool",
            objective_endpoint="tool_call",
            output_node_label="Residual Output: <tool_call>",
            decision_label="tool_call_vs_no_tool",
        )
    if direction == "reverse":
        return DirectionSpec(
            direction="reverse",
            clean_text=no_tool_text,
            corrupt_text=tool_text,
            clean_prompt_path=sample.corrupt_path,
            corrupt_prompt_path=sample.clean_path,
            clean_role="no_tool",
            corrupt_role="tool_call",
            objective_endpoint="no_tool",
            output_node_label="Residual Output: no_tool",
            decision_label="no_tool_vs_tool_call",
        )
    raise ValueError(f"Unknown direction: {direction}")


def sample_label(sample: ToolCallSample, direction: str) -> Dict[str, object]:
    return {
        "sample_id": sample.sample_id,
        "sample_rank": sample.sample_rank,
        "filename": sample.filename,
        "direction": direction,
        "legacy_index": sample.legacy_index,
    }


def select_directional_samples(
    *,
    source: str,
    dataset_root: Path,
    pair_dir: Path,
    sample_ids_raw: str,
    sample_rank_min: int,
    sample_rank_max: int,
    q_list_raw: str,
    q_min: int,
    q_max: int,
    max_samples: int,
) -> List[ToolCallSample]:
    if source == "dataset":
        samples = load_toolcall_samples(dataset_root=dataset_root)
        return select_samples(
            samples,
            sample_ids=parse_sample_id_list(sample_ids_raw),
            sample_rank_min=sample_rank_min if sample_rank_min > 0 else 1,
            sample_rank_max=sample_rank_max,
            max_samples=max_samples,
        )

    samples = load_toolcall_samples(pair_dir=pair_dir)
    if q_list_raw.strip():
        legacy_indices = parse_legacy_index_list(q_list_raw)
    elif q_min > 0 or q_max > 0:
        lo = q_min if q_min > 0 else 1
        hi = q_max if q_max > 0 else 10**9
        legacy_indices = [
            s.legacy_index
            for s in samples
            if s.legacy_index is not None and lo <= s.legacy_index <= hi
        ]
    else:
        legacy_indices = [s.legacy_index for s in samples if s.legacy_index is not None]
    return select_samples(
        samples,
        legacy_indices=[int(x) for x in legacy_indices if x is not None],
        max_samples=max_samples,
    )


def write_edge_count_manifest(edge_counts: Sequence[int], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {
                "edge_counts": [int(x) for x in edge_counts],
                "selected_policy": "largest_if_not_overridden",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


__all__ = [
    "DirectionSpec",
    "build_single_pair_dataloader",
    "resolve_direction_spec",
    "sample_label",
    "save_sample_catalog",
    "select_directional_samples",
    "write_edge_count_manifest",
]
