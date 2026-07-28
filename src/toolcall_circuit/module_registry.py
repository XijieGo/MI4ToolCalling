#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Sequence


def _candidate_registry_paths() -> List[Path]:
    paths: List[Path] = []
    env_path = os.environ.get("TOOLCALL_MODULE_REGISTRY", "").strip()
    if env_path:
        paths.append(Path(env_path).expanduser().resolve())
    project_root = Path(__file__).resolve().parents[4]
    paths.append(
        project_root
        / "experiment"
        / "results"
        / "split"
        / "discovery_method_branches"
        / "eap_ig"
        / "pipeline"
        / "module_registry.json"
    )
    return paths


def load_module_registry(path: str | Path | None = None) -> Dict[str, Any]:
    candidates = [Path(path).expanduser().resolve()] if path else _candidate_registry_paths()
    for candidate in candidates:
        if candidate.exists():
            return json.loads(candidate.read_text(encoding="utf-8"))
    return {}


def registry_section(name: str, *, registry: Dict[str, Any] | None = None) -> Dict[str, Any]:
    reg = registry or load_module_registry()
    section = reg.get(name, {})
    return section if isinstance(section, dict) else {}


def registry_list(
    section_name: str,
    key: str,
    default: Sequence[str],
    *,
    registry: Dict[str, Any] | None = None,
) -> List[str]:
    section = registry_section(section_name, registry=registry)
    value = section.get(key, default)
    if not isinstance(value, list):
        return list(default)
    return [str(x) for x in value]


def registry_value(
    section_name: str,
    key: str,
    default: str,
    *,
    registry: Dict[str, Any] | None = None,
) -> str:
    section = registry_section(section_name, registry=registry)
    value = section.get(key, default)
    return str(value) if value else str(default)


def registry_edge_specs(
    section_name: str,
    key: str,
    default: Sequence[Dict[str, Any]],
    *,
    registry: Dict[str, Any] | None = None,
) -> List[Dict[str, Any]]:
    section = registry_section(section_name, registry=registry)
    value = section.get(key, default)
    if not isinstance(value, list):
        return [dict(x) for x in default]
    out: List[Dict[str, Any]] = []
    for row in value:
        if isinstance(row, dict):
            out.append(dict(row))
    return out or [dict(x) for x in default]


__all__ = [
    "load_module_registry",
    "registry_edge_specs",
    "registry_list",
    "registry_section",
    "registry_value",
]
