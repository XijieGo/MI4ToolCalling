#!/usr/bin/env python3
from __future__ import annotations

import os
from pathlib import Path


ARTIFACT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ARTIFACT_ROOT / "src"


def env_path(name: str, default: str | Path) -> Path:
    return Path(os.environ.get(name, str(default))).expanduser()


MODEL_ROOT = env_path("MODEL_ROOT", ARTIFACT_ROOT / "external" / "models")
TRANSCODER_ROOT = env_path("TRANSCODER_ROOT", ARTIFACT_ROOT / "external" / "transcoders")

QWEN3_1P7B_PATH = env_path("QWEN3_1P7B_PATH", MODEL_ROOT / "Qwen3-1.7B")
QWEN3_4B_PATH = env_path("QWEN3_4B_PATH", MODEL_ROOT / "Qwen3-4B")
QWEN3_8B_PATH = env_path("QWEN3_8B_PATH", MODEL_ROOT / "Qwen3-8B")
QWEN3_14B_PATH = env_path("QWEN3_14B_PATH", MODEL_ROOT / "Qwen3-14B")
QWEN35_4B_PATH = env_path("QWEN35_4B_PATH", MODEL_ROOT / "Qwen3.5-4B")
QWEN35_9B_PATH = env_path("QWEN35_9B_PATH", MODEL_ROOT / "Qwen3.5-9B")
MISTRAL_3P2_24B_PATH = env_path("MISTRAL_3P2_24B_PATH", MODEL_ROOT / "Mistral-Small-3.2-24B-Instruct-2506")
DEVSTRAL_2_24B_PATH = env_path("DEVSTRAL_2_24B_PATH", MODEL_ROOT / "Devstral-Small-2-24B-Instruct-2512")
GRANITE_3P3_8B_PATH = env_path("GRANITE_3P3_8B_PATH", MODEL_ROOT / "granite-3.3-8b-instruct")

QWEN3_1P7B_TRANSCODER_PATH = env_path("QWEN3_1P7B_TRANSCODER_PATH", TRANSCODER_ROOT / "Qwen3-1.7B")
QWEN3_4B_TRANSCODER_PATH = env_path("QWEN3_4B_TRANSCODER_PATH", TRANSCODER_ROOT / "Qwen3-4B")
QWEN3_8B_TRANSCODER_PATH = env_path("QWEN3_8B_TRANSCODER_PATH", TRANSCODER_ROOT / "Qwen3-8B")
QWEN3_14B_TRANSCODER_PATH = env_path("QWEN3_14B_TRANSCODER_PATH", TRANSCODER_ROOT / "Qwen3-14B")
