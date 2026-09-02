#!/usr/bin/env python3
"""Portable, shared defaults for the Qwen3.5/Transcoder rebuttal runners.

Weights are intentionally external to the release.  Each value below resolves
under ``external/`` by default and can be overridden with the corresponding
environment variable documented in :mod:`artifact_paths`.
"""

from __future__ import annotations

import sys
from pathlib import Path


SRC_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SRC_ROOT.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from artifact_paths import (  # noqa: E402
    GRANITE_3P3_8B_PATH,
    GRANITE_3P3_8B_TRANSCODER_PATH,
    QWEN3_14B_PATH,
    QWEN3_14B_TRANSCODER_PATH,
    QWEN3_4B_PATH,
    QWEN3_4B_TRANSCODER_PATH,
    QWEN3_8B_PATH,
    QWEN3_8B_TRANSCODER_PATH,
    QWEN35_4B_PATH,
    QWEN35_4B_TRANSCODER_PATH,
    QWEN35_9B_PATH,
    QWEN35_9B_TRANSCODER_PATH,
)


V5_DATASET_ROOT = REPO_ROOT / "datasets" / "v5_model_specific_balanced"
REBUTTAL_ROOT = REPO_ROOT / "rebuttal"

# The three models with the released cross-model K audit.  Keep the dataset,
# reference-layer, and external-weight settings in one place so that a server
# migration changes only environment variables, rather than source files.
CROSS_MODEL_SPECS: dict[str, dict[str, object]] = {
    "granite_3p3_8b": {
        "label": "Granite-3.3-8B-Instruct",
        "model_path": GRANITE_3P3_8B_PATH,
        "dataset_root": V5_DATASET_ROOT / "granite_3p3_8b",
        "transcoder_root": GRANITE_3P3_8B_TRANSCODER_PATH,
        "reference_layer": 36,
    },
    "qwen35_4b": {
        "label": "Qwen3.5-4B",
        "model_path": QWEN35_4B_PATH,
        "dataset_root": V5_DATASET_ROOT / "qwen35_4b",
        "transcoder_root": QWEN35_4B_TRANSCODER_PATH,
        "reference_layer": 29,
    },
    "qwen35_9b": {
        "label": "Qwen3.5-9B",
        "model_path": QWEN35_9B_PATH,
        "dataset_root": V5_DATASET_ROOT / "qwen35_9b",
        "transcoder_root": QWEN35_9B_TRANSCODER_PATH,
        "reference_layer": 28,
    },
}

# The bidirectional S/E audit includes the Qwen3 formation windows as well.
SE_DOMINANCE_SPECS: dict[str, dict[str, object]] = {
    "qwen3_4b": {
        "label": "Qwen3-4B",
        "model_path": QWEN3_4B_PATH,
        "dataset_root": V5_DATASET_ROOT / "qwen3_4b",
        "transcoder_root": QWEN3_4B_TRANSCODER_PATH,
        "decision_layer": 26,
        "layers": [22, 23, 24, 25],
    },
    "qwen3_8b": {
        "label": "Qwen3-8B",
        "model_path": QWEN3_8B_PATH,
        "dataset_root": V5_DATASET_ROOT / "qwen3_8b",
        "transcoder_root": QWEN3_8B_TRANSCODER_PATH,
        "decision_layer": 24,
        "layers": [20, 21, 22, 23],
    },
    "qwen3_14b": {
        "label": "Qwen3-14B",
        "model_path": QWEN3_14B_PATH,
        "dataset_root": V5_DATASET_ROOT / "qwen3_14b",
        "transcoder_root": QWEN3_14B_TRANSCODER_PATH,
        "decision_layer": 33,
        "layers": [29, 30, 31, 32],
    },
    "qwen35_4b": {
        "label": "Qwen3.5-4B",
        "model_path": QWEN35_4B_PATH,
        "dataset_root": V5_DATASET_ROOT / "qwen35_4b",
        "transcoder_root": QWEN35_4B_TRANSCODER_PATH,
        "decision_layer": 29,
        "layers": [26, 27, 28, 29],
    },
    "qwen35_9b": {
        "label": "Qwen3.5-9B",
        "model_path": QWEN35_9B_PATH,
        "dataset_root": V5_DATASET_ROOT / "qwen35_9b",
        "transcoder_root": QWEN35_9B_TRANSCODER_PATH,
        "decision_layer": 28,
        "layers": [26, 27, 28, 29],
    },
    "granite_3p3_8b": {
        "label": "Granite-3.3-8B-Instruct",
        "model_path": GRANITE_3P3_8B_PATH,
        "dataset_root": V5_DATASET_ROOT / "granite_3p3_8b",
        "transcoder_root": GRANITE_3P3_8B_TRANSCODER_PATH,
        "decision_layer": 36,
        "layers": [29, 30, 31, 32],
    },
}


def activation_cache_root(model_key: str) -> Path:
    """Return the released cache directory for a cross-model K audit."""
    cache_key = "granite_3p3_8b_strict" if model_key == "granite_3p3_8b" else model_key
    return REBUTTAL_ROOT / "10_cross_model_transcoder_k" / cache_key
