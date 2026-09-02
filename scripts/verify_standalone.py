#!/usr/bin/env python3
"""Validate that the canonical MI4 experiments survive a directory migration.

This validator never imports model code or loads model/Transcoder weights. It
checks the registry, frozen datasets/evidence, Python syntax, external-path
leaks in canonical sources, and that LFS-tracked payloads are real files rather
than Git-LFS pointer text.
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = ROOT / "experiments" / "registry.json"
V5_MODELS = (
    "qwen3_4b",
    "qwen3_8b",
    "qwen3_14b",
    "qwen35_4b",
    "qwen35_9b",
    "mistral_3p2_24b",
    "granite_3p3_8b",
)
RETIRED_WORKSPACE_FRAGMENT = "MI4ToolCalling" + chr(45) + "sync"
OLD_AUTODL_ROOT = "/root/autodl-tmp"
CANONICAL_FORBIDDEN_PATHS = (
    RETIRED_WORKSPACE_FRAGMENT,
    OLD_AUTODL_ROOT + "/MI4ToolCalling",
    OLD_AUTODL_ROOT + "/Qwen",
    OLD_AUTODL_ROOT + "/Transcoder",
    OLD_AUTODL_ROOT + "/Mistral",
    OLD_AUTODL_ROOT + "/Granite",
)
PUBLIC_TREES = ("README.md", "datasets", "experiments", "paper", "rebuttal", "scripts", "src")
PAPER_SNAPSHOT_FILES = (
    "paper/neurips_2026.tex",
    "paper/main.tex",
    "paper/appendix.tex",
    "paper/references.bib",
    "paper/figures/feature_circuit.png",
)
LFS_POINTER_PREFIX = b"version https://git-lfs.github.com/spec/v1"
EXPERIMENT_TIERS = {"paper_source_map", "standalone_rebuttal"}


def add(results: list[dict[str, str]], check: str, ok: bool, detail: str) -> None:
    results.append({"check": check, "status": "ok" if ok else "error", "detail": detail})


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def jsonl_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def files_under(path: Path) -> Iterable[Path]:
    if path.is_file():
        yield path
    elif path.is_dir():
        yield from (item for item in path.rglob("*") if item.is_file())


def contains(path: Path, needle: bytes) -> bool:
    overlap = max(len(needle) - 1, 0)
    previous = b""
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            if needle in previous + chunk:
                return True
            previous = chunk[-overlap:] if overlap else b""
    return False


def collect_registry_paths(registry: dict[str, Any], results: list[dict[str, str]]) -> list[Path]:
    experiments = registry.get("experiments")
    if not isinstance(experiments, list) or not experiments:
        add(results, "registry experiments", False, "missing or empty")
        return []
    ids: list[str] = []
    source_paths: list[Path] = []
    for experiment in experiments:
        if not isinstance(experiment, dict):
            add(results, "registry experiment", False, "entry is not an object")
            continue
        identifier = experiment.get("id")
        if not isinstance(identifier, str) or not identifier:
            add(results, "registry experiment id", False, f"invalid id: {identifier!r}")
            continue
        ids.append(identifier)
        tier = experiment.get("tier")
        add(
            results,
            f"registry {identifier}.tier",
            tier in EXPERIMENT_TIERS,
            str(tier) if tier in EXPERIMENT_TIERS else f"expected one of {sorted(EXPERIMENT_TIERS)}, got {tier!r}",
        )
        for field in ("data", "code", "evidence"):
            values = experiment.get(field)
            if not isinstance(values, list):
                add(results, f"registry {identifier}.{field}", False, "not a list")
                continue
            for relative in values:
                if not isinstance(relative, str):
                    add(results, f"registry {identifier}.{field}", False, f"non-string path: {relative!r}")
                    continue
                path = ROOT / relative
                add(results, f"{identifier}:{relative}", path.exists(), "present" if path.exists() else "missing")
                if field == "code" and path.is_file():
                    source_paths.append(path)
    unique = len(ids) == len(set(ids))
    add(results, "registry unique experiment ids", unique, f"{len(ids)} ids" if unique else "duplicate id")

    support = registry.get("runtime_support", [])
    if not isinstance(support, list):
        add(results, "registry runtime_support", False, "not a list")
    else:
        for relative in support:
            if not isinstance(relative, str):
                add(results, "registry runtime_support", False, f"non-string path: {relative!r}")
                continue
            path = ROOT / relative
            add(results, f"runtime:{relative}", path.is_file(), "present" if path.is_file() else "missing")
            if path.is_file() and path.suffix == ".py":
                source_paths.append(path)
    return list(dict.fromkeys(source_paths))


def check_python_sources(paths: list[Path], results: list[dict[str, str]]) -> None:
    for path in paths:
        relative = str(path.relative_to(ROOT))
        try:
            text = path.read_text(encoding="utf-8")
            ast.parse(text, filename=str(path))
            add(results, f"syntax:{relative}", True, "AST parsed")
        except Exception as exc:
            add(results, f"syntax:{relative}", False, str(exc))
            continue
        violations = [fragment for fragment in CANONICAL_FORBIDDEN_PATHS if fragment in text]
        add(
            results,
            f"portable-source:{relative}",
            not violations,
            "no retired server path" if not violations else f"contains: {', '.join(violations)}",
        )


def check_v2(results: list[dict[str, str]]) -> None:
    expected = {"train": 1200, "test": 300}
    for split, expected_count in expected.items():
        path = ROOT / "datasets" / split / "clean" / "manifest.jsonl"
        try:
            rows = jsonl_rows(path)
            add(
                results,
                f"v2:{split}",
                len(rows) == expected_count,
                f"{len(rows)} rows" if len(rows) == expected_count else f"expected {expected_count}, got {len(rows)}",
            )
        except Exception as exc:
            add(results, f"v2:{split}", False, str(exc))


def check_paper_snapshot(results: list[dict[str, str]]) -> None:
    for relative in PAPER_SNAPSHOT_FILES:
        path = ROOT / relative
        add(results, f"paper snapshot:{relative}", path.is_file(), "present" if path.is_file() else "missing")


def check_v5(results: list[dict[str, str]]) -> None:
    for model in V5_MODELS:
        path = ROOT / "datasets" / "v5_model_specific_balanced" / model / "manifest.jsonl"
        try:
            rows = jsonl_rows(path)
            splits = Counter(str(row.get("split")) for row in rows)
            unique_ids = {str(row.get("sample_id")) for row in rows}
            ok = len(rows) == 500 and len(unique_ids) == 500 and splits == Counter({"train": 200, "heldout": 300})
            detail = "500 unique rows, 200 train / 300 heldout" if ok else f"rows={len(rows)}, unique={len(unique_ids)}, splits={dict(splits)}"
            add(results, f"v5:{model}", ok, detail)
        except Exception as exc:
            add(results, f"v5:{model}", False, str(exc))


def check_implicit_intent(results: list[dict[str, str]]) -> None:
    expected = {
        "implicit candidates": ("implicit_intent_oversampled_600.jsonl", 600),
        "implicit screen": ("implicit_intent_oversampled_600_screened.jsonl", 600),
        "implicit Qwen3-8B arm": ("removal_arm_161.jsonl", 161),
    }
    root = ROOT / "rebuttal" / "06_implicit_intent_tool_calls" / "artifacts"
    for label, (filename, expected_count) in expected.items():
        path = root / filename
        try:
            rows = jsonl_rows(path)
            add(results, label, len(rows) == expected_count, f"{len(rows)} rows" if len(rows) == expected_count else f"expected {expected_count}, got {len(rows)}")
        except Exception as exc:
            add(results, label, False, str(exc))


def check_json_evidence(registry: dict[str, Any], results: list[dict[str, str]]) -> None:
    for experiment in registry.get("experiments", []):
        if not isinstance(experiment, dict):
            continue
        for relative in experiment.get("evidence", []):
            if isinstance(relative, str) and relative.endswith(".json"):
                path = ROOT / relative
                try:
                    read_json(path)
                    add(results, f"json:{relative}", True, "valid JSON")
                except Exception as exc:
                    add(results, f"json:{relative}", False, str(exc))


def check_lfs_payloads(results: list[dict[str, str]]) -> None:
    candidates = [
        *(
            path
            for path in files_under(ROOT / "rebuttal")
            if path.suffix in {".pt", ".pth", ".safetensors"}
        ),
        *(path for path in files_under(ROOT / "datasets" / "external") if path.suffix == ".jsonl"),
    ]
    pointers: list[str] = []
    unreadable: list[str] = []
    for path in candidates:
        try:
            with path.open("rb") as handle:
                if handle.read(len(LFS_POINTER_PREFIX) + 32).startswith(LFS_POINTER_PREFIX):
                    pointers.append(str(path.relative_to(ROOT)))
        except OSError:
            unreadable.append(str(path.relative_to(ROOT)))
    ok = not pointers and not unreadable
    if ok:
        detail = f"{len(candidates)} checked payloads are materialized"
    else:
        detail = f"LFS pointers={pointers[:3]}, unreadable={unreadable[:3]}"
    add(results, "LFS payload materialization", ok, detail)


def check_retired_workspace(results: list[dict[str, str]]) -> None:
    files: list[Path] = []
    for relative in PUBLIC_TREES:
        files.extend(files_under(ROOT / relative))
    needle = RETIRED_WORKSPACE_FRAGMENT.encode("utf-8")
    violations: list[str] = []
    for path in files:
        try:
            if contains(path, needle):
                violations.append(str(path.relative_to(ROOT)))
        except OSError:
            violations.append(f"unreadable:{path.relative_to(ROOT)}")
    add(
        results,
        "retired-workspace references",
        not violations,
        f"no reference in {len(files)} public files" if not violations else f"found in: {', '.join(violations[:5])}",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Emit a machine-readable report.")
    args = parser.parse_args()
    results: list[dict[str, str]] = []

    try:
        registry = read_json(REGISTRY_PATH)
        add(results, "registry JSON", isinstance(registry, dict) and registry.get("schema_version") == 1, "schema version 1")
    except Exception as exc:
        add(results, "registry JSON", False, str(exc))
        registry = {}

    source_paths = collect_registry_paths(registry, results)
    check_python_sources(source_paths, results)
    check_paper_snapshot(results)
    check_v2(results)
    check_v5(results)
    check_implicit_intent(results)
    check_json_evidence(registry, results)
    check_lfs_payloads(results)
    check_retired_workspace(results)

    ok = all(item["status"] == "ok" for item in results)
    report = {"repository_root": str(ROOT), "ok": ok, "checks": results}
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        for item in results:
            print(f"[{item['status'].upper()}] {item['check']}: {item['detail']}")
        print(f"\nStandalone migration validation {'passed' if ok else 'failed'}: {sum(item['status'] == 'ok' for item in results)}/{len(results)} checks passed.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
