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
    local_fallbacks = {
        "qwen3_8b": Path("/home/xijie/models/Qwen3-8B"),
        "qwen3_4b": Path("/home/xijie/models/Qwen3-4B"),
        "qwen3_14b": Path("/home/xijie/models/Qwen3-14B"),
        "qwen35_4b": Path("/home/xijie/models/Qwen3.5-4B"),
        "qwen35_9b": Path("/home/xijie/models/Qwen3.5-9B"),
        "mistral": Path("/home/xijie/models/Mistral-Small-3.2-24B-Instruct-2506"),
        "granite": Path("/home/xijie/models/granite-3.3-8b-instruct"),
        "devstral": Path("/home/xijie/models/Devstral-Small-2-24B-Instruct-2512"),
    }
    autodl_fallbacks = {
        "qwen3_8b": Path("/root/autodl-tmp/Qwen/Qwen3-8B"),
        "qwen3_4b": Path("/root/autodl-tmp/Qwen/Qwen3-4B"),
        "qwen3_14b": Path("/root/autodl-tmp/Qwen/Qwen3-14B"),
        "qwen35_4b": Path("/root/autodl-tmp/Qwen/Qwen3.5-4B"),
        "qwen35_9b": Path("/root/autodl-tmp/Qwen/Qwen3.5-9B"),
        "mistral": Path("/root/autodl-tmp/Mistral-Small-3.2-24B-Instruct-2506"),
        "granite": Path("/root/autodl-tmp/Granite/granite-3.3-8b-instruct"),
        "devstral": Path("/root/autodl-tmp/Hermes/Devstral-Small-2-24B-Instruct-2512"),
    }
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
    default_path = local_fallbacks[name] if local_fallbacks[name].exists() else REPO_ROOT / default
    return external_path(env_name, default_path)


def transcoder_root() -> Path:
    local_tc = Path("/home/xijie/transcoders")
    default = local_tc if local_tc.exists() else REPO_ROOT / "external" / "transcoders"
    return external_path("MI4TC_TRANSCODER_ROOT", default)


def require_inside_repo(path: Path) -> Path:
    """Return a resolved path and reject accidental predecessor-workspace use."""

    resolved = path.expanduser().resolve()
    # Detect the old checkout by its directory name without embedding a
    # machine-specific absolute path in the new repository.
    if any(part == "MI4ToolCalling" for part in resolved.parts):
        raise ValueError(f"Path points into a predecessor repository: {resolved}")
    return resolved
