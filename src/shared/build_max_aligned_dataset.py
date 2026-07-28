#!/usr/bin/env python3
"""Compatibility alias for the canonical fresh-dataset builder.

The pre-refactor implementation at this path reconstructed the instruction
line using a longest-common-suffix heuristic and contained hard-coded paths to
the old project.  Use this alias or, preferably,
``python -m paper_repro.build_dataset``.  The canonical implementation renders
only the leading instruction verb and requires an explicit output directory.
"""
from __future__ import annotations

import sys
from pathlib import Path


SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from paper_repro.build_dataset import main  # noqa: E402


if __name__ == "__main__":
    main()
