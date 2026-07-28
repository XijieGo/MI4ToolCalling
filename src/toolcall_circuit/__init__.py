from pathlib import Path

from .paths import DATASETS_ROOT, MODEL_PATH_DEFAULT, PROJECT_ROOT, RESULTS_ROOT, timestamp_tag

__all__ = [
    "PROJECT_ROOT",
    "DATASETS_ROOT",
    "RESULTS_ROOT",
    "MODEL_PATH_DEFAULT",
    "timestamp_tag",
    "package_root",
]


def package_root() -> Path:
    return PROJECT_ROOT
