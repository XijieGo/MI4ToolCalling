#!/usr/bin/env python3
"""Export the code, scripts and required rendering configuration as a source archive."""

from __future__ import annotations

import argparse
import tarfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ROOT_FILES = ("README.md", "LICENSE", "pyproject.toml", ".gitignore", ".gitattributes", ".env.example")
LOCAL_ONLY_FILES = frozenset({
    "experiments/cross_model/audit_suspect_models.py",
    "scripts/recompute_all.py",
    "scripts/extend_recompute_queue.py",
    "scripts/recompute_pre_layers.py",
    "scripts/recompute_retry.py",
    "scripts/swap_pair_splits.py",
})


def release_files() -> list[Path]:
    files = {ROOT / name for name in ROOT_FILES}
    for folder in ("src", "scripts", "tests", "experiments"):
        for path in (ROOT / folder).rglob("*"):
            if not path.is_file() or any(part in {"__pycache__", "native_release"} for part in path.parts):
                continue
            if path.relative_to(ROOT).as_posix() in LOCAL_ONLY_FILES:
                continue
            if path.suffix == ".py" or (folder == "scripts" and path.suffix == ".sh") or path.name == "README.md":
                files.add(path)
    templates = ROOT / "experiments/cross_model/transfer/templates"
    files.add(templates / "LICENSE")
    for domain in ("telecom", "retail"):
        files.update(templates / domain / name for name in ("tau2_system_prompt.txt", "tau2_tool_schemas.json"))
    for path in files:
        if not path.is_file():
            raise FileNotFoundError(path)
        path.resolve().relative_to(ROOT.resolve())
    return sorted(files)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "dist/MI4ToolCalling-code.tar.gz")
    args = parser.parse_args()
    files = release_files()
    output = args.output.expanduser().resolve()
    if output in {path.resolve() for path in files}:
        raise ValueError("The output archive must be separate from its source files")
    if output.exists():
        raise FileExistsError(f"Choose a new archive path: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(output, "w:gz") as archive:
        for path in files:
            archive.add(path, arcname=str(Path("MI4ToolCalling") / path.relative_to(ROOT)), recursive=False)
    print(f"Exported {len(files)} files to {output}")


if __name__ == "__main__":
    main()
