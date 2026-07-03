"""Compatibility exports for geometry probing config utilities."""

from __future__ import annotations

import sys
from pathlib import Path

try:
    from pba.geometry.config import Config, load_config  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover - direct legacy script fallback
    repo_root = Path(__file__).resolve().parents[4]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from pba.geometry.config import Config, load_config  # noqa: F401

__all__ = ["Config", "load_config"]
