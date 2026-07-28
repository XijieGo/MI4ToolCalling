from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

ARTIFACT_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = ARTIFACT_ROOT
EXPERIMENT_ROOT = ARTIFACT_ROOT
DATASETS_ROOT = ARTIFACT_ROOT / "datasets"
RESULTS_ROOT = ARTIFACT_ROOT / "results"
MODEL_PATH_DEFAULT = Path(
    os.environ.get("QWEN3_1P7B_PATH", str(ARTIFACT_ROOT / "external" / "models" / "Qwen3-1.7B"))
)


def timestamp_tag(now: datetime | None = None) -> str:
    return (now or datetime.now()).strftime("%d-%H-%M")


def manual_run_root(name: str = "manual_run") -> Path:
    return RESULTS_ROOT / name
