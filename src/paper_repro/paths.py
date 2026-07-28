"""Path helpers for the reconstructed paper artifact.

No experiment should infer a path from the old ``project-new`` tree.  All
defaults are relative to the checked-out MI4ToolCalling repository, while
models and Transcoders remain configurable through environment variables.
"""
from __future__ import annotations

import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
DATASETS_ROOT = REPO_ROOT / "datasets"
RESULTS_ROOT = REPO_ROOT / "results"
PROVENANCE_ROOT = DATASETS_ROOT / "provenance"


def environment_path(name: str, default: Path) -> Path:
    """Resolve an optional absolute path supplied through an environment variable."""

    return Path(os.environ.get(name, str(default))).expanduser()


MODEL_ROOT = environment_path("MODEL_ROOT", REPO_ROOT / "external" / "models")
TRANSCODER_ROOT = environment_path("TRANSCODER_ROOT", REPO_ROOT / "external" / "transcoders")

MODEL_PATHS = {
    "qwen3_1p7b": environment_path("QWEN3_1P7B_PATH", MODEL_ROOT / "Qwen3-1.7B"),
    "qwen3_4b": environment_path("QWEN3_4B_PATH", MODEL_ROOT / "Qwen3-4B"),
    "qwen3_8b": environment_path("QWEN3_8B_PATH", MODEL_ROOT / "Qwen3-8B"),
    "qwen3_14b": environment_path("QWEN3_14B_PATH", MODEL_ROOT / "Qwen3-14B"),
    "qwen35_9b": environment_path("QWEN35_9B_PATH", MODEL_ROOT / "Qwen3.5-9B"),
    "mistral_3p2_24b": environment_path(
        "MISTRAL_3P2_24B_PATH", MODEL_ROOT / "Mistral-Small-3.2-24B-Instruct-2506"
    ),
    "devstral_2_24b": environment_path(
        "DEVSTRAL_2_24B_PATH", MODEL_ROOT / "Devstral-Small-2-24B-Instruct-2512"
    ),
    "granite_3p3_8b": environment_path("GRANITE_3P3_8B_PATH", MODEL_ROOT / "granite-3.3-8b-instruct"),
}

TRANSCODER_PATHS = {
    "qwen3_1p7b": environment_path("QWEN3_1P7B_TRANSCODER_PATH", TRANSCODER_ROOT / "Qwen3-1.7B"),
    "qwen3_4b": environment_path("QWEN3_4B_TRANSCODER_PATH", TRANSCODER_ROOT / "Qwen3-4B"),
    "qwen3_8b": environment_path("QWEN3_8B_TRANSCODER_PATH", TRANSCODER_ROOT / "Qwen3-8B"),
    "qwen3_14b": environment_path("QWEN3_14B_TRANSCODER_PATH", TRANSCODER_ROOT / "Qwen3-14B"),
}
