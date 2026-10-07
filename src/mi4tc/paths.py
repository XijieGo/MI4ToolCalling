"""Repository paths and a few explicit external-path hooks.

Callers pass dataset and output paths directly. The environment variables
below are only for model weights and Transcoder checkpoints, which stay
outside this repository.
"""

from __future__ import annotations

import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
DATASETS_ROOT = REPO_ROOT / "datasets"
EXPERIMENTS_ROOT = REPO_ROOT / "experiments"


def external_path(name: str, default: str | Path) -> Path:
    """Resolve one explicit external path without introducing a config layer."""

    return Path(os.environ.get(name, str(default))).expanduser()


def model_path(name: str = "qwen3_8b") -> Path:
    name = {"granite_3p3_8b": "granite", "mistral_3p2_24b": "mistral"}.get(name, name)
    defaults = {
        "qwen3_8b": ("MI4TC_QWEN3_8B_PATH", "external/models/Qwen3-8B"),
        "qwen3_4b": ("MI4TC_QWEN3_4B_PATH", "external/models/Qwen3-4B"),
        "qwen3_14b": ("MI4TC_QWEN3_14B_PATH", "external/models/Qwen3-14B"),
        "qwen35_4b": ("MI4TC_QWEN35_4B_PATH", "external/models/Qwen3.5-4B"),
        "qwen35_9b": ("MI4TC_QWEN35_9B_PATH", "external/models/Qwen3.5-9B"),
        "mistral": ("MI4TC_MISTRAL_PATH", "external/models/Mistral-Small-3.2-24B-Instruct-2506"),
        "granite": ("MI4TC_GRANITE_PATH", "external/models/granite-3.3-8b-instruct"),
        "devstral": ("MI4TC_DEVSTRAL_PATH", "external/models/Devstral-Small-2-24B-Instruct-2512"),
    }
    try:
        env_name, default = defaults[name]
    except KeyError as exc:
        raise ValueError(f"Unknown model key: {name!r}; choose from {sorted(defaults)}") from exc
    root = external_path("MI4TC_MODEL_ROOT", REPO_ROOT / "external/models")
    default_path = root / Path(default).name
    return external_path(env_name, default_path)


def transcoder_root() -> Path:
    return external_path("MI4TC_TRANSCODER_ROOT", REPO_ROOT / "external/transcoders")


def released_transcoder_root() -> Path:
    return external_path("MI4TC_RELEASE_ROOT", transcoder_root() / "release")


def require_inside_repo(path: Path) -> Path:
    """Return a resolved path and reject accidental predecessor-workspace use."""

    resolved = path.expanduser().resolve()
    # Detect the old checkout by its directory name without embedding a
    # machine-specific absolute path in the new repository.
    if any(parts == ("ToolCalling", "MI4ToolCalling") for parts in zip(resolved.parts, resolved.parts[1:])):
        raise ValueError(f"Path points into a predecessor repository: {resolved}")
    return resolved
