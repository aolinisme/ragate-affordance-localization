"""Compatibility exports for AGD20K sample iteration utilities."""

from __future__ import annotations

import sys
from pathlib import Path

try:
    from pba.data.affordance import SampleEntry, iter_agd20k_samples  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover - direct legacy script fallback
    repo_root = Path(__file__).resolve().parents[4]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from pba.data.affordance import SampleEntry, iter_agd20k_samples  # noqa: F401

__all__ = ["SampleEntry", "iter_agd20k_samples"]
