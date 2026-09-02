#!/usr/bin/env python3
"""Portable paths for the frozen implicit-intent rebuttal artifact bundle."""

from __future__ import annotations

import sys
from pathlib import Path


ARTIFACT_DIR = Path(__file__).resolve().parent
REPO_ROOT = ARTIFACT_DIR.parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from artifact_paths import (  # noqa: E402
    GRANITE_3P3_8B_PATH,
    MISTRAL_3P2_24B_PATH,
    QWEN3_14B_PATH,
    QWEN3_4B_PATH,
    QWEN3_8B_PATH,
    QWEN35_4B_PATH,
    QWEN35_9B_PATH,
)


# Frozen, release-contained data.  The 600-item source collection is the
# standalone rerun entrypoint; it does not need the historical source pools.
FROZEN_CANDIDATES = ARTIFACT_DIR / "implicit_intent_oversampled_600.jsonl"
FROZEN_SCREENED_CANDIDATES = ARTIFACT_DIR / "implicit_intent_oversampled_600_screened.jsonl"
FROZEN_QWEN3_8B_REMOVAL_ARM = ARTIFACT_DIR / "removal_arm_161.jsonl"
RERUN_ROOT = ARTIFACT_DIR / "reruns"
DEFAULT_CROSS_MODEL_OUTPUT_ROOT = RERUN_ROOT / "cross_model_removal"
DEFAULT_QWEN3_8B_SCREEN_ROOT = RERUN_ROOT / "qwen3_8b_screen"
DEFAULT_QWEN3_8B_REMOVAL_ROOT = RERUN_ROOT / "qwen3_8b_removal"

# Native-rendering inputs and frozen directions that replaced paths from the
# retired synchronization workspace.  Their hashes are checked by the release
# validator where applicable.
D1_REFERENCE = REPO_ROOT / "datasets" / "train" / "clean" / "apps_python_1.txt"
QWEN35_REFERENCE = REPO_ROOT / "datasets" / "provenance" / "external_historical" / "qwen35_9b" / "selected_native_pairs" / "clean_1.txt"
MISTRAL_SYSTEM_PATH = REPO_ROOT / "datasets" / "provenance" / "external_historical" / "mistral_3p2_24b" / "tokenizer_verification.json"

QWEN3_VECTOR_ROOT = REPO_ROOT / "rebuttal" / "02_schema_tool_identity" / "upstream_vector_inputs"
CODING_VECTOR_ROOT = REPO_ROOT / "rebuttal" / "05_tau2_bench_200" / "coding_direction_inputs" / "cross_family_tau2_20260726"

MODEL_PATHS = {
    "qwen3_4b": QWEN3_4B_PATH,
    "qwen3_8b": QWEN3_8B_PATH,
    "qwen3_14b": QWEN3_14B_PATH,
    "qwen35_4b": QWEN35_4B_PATH,
    "qwen35_9b": QWEN35_9B_PATH,
    "mistral": MISTRAL_3P2_24B_PATH,
    "granite": GRANITE_3P3_8B_PATH,
}

VECTOR_PATHS = {
    "qwen3_4b": QWEN3_VECTOR_ROOT / "v4_full_matrix_qwen3_20260724T183837Z" / "Qwen3-4B" / "native_vectors" / "D1_L24_pre.pt",
    "qwen3_8b": QWEN3_VECTOR_ROOT / "rebuttal_scaffold_components_d1_v4_full_20260725" / "native_vectors" / "RTF_L24_pre.pt",
    "qwen3_14b": QWEN3_VECTOR_ROOT / "v4_full_matrix_qwen3_20260724T183837Z" / "Qwen3-14B" / "native_vectors" / "D1_L24_pre.pt",
    "qwen35_4b": CODING_VECTOR_ROOT / "qwen35_4b" / "coding_vector_bundle.pt",
    "qwen35_9b": CODING_VECTOR_ROOT / "qwen35_9b" / "coding_vector_bundle.pt",
    "mistral": CODING_VECTOR_ROOT / "mistral" / "coding_vector_bundle.pt",
    "granite": CODING_VECTOR_ROOT / "granite" / "coding_vector_bundle.pt",
}
