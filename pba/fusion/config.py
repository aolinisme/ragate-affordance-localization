"""Fusion zero-shot configuration loading."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml

__all__ = ["load_config"]


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise TypeError(f"Configuration file {path} must contain a mapping at the top level")
    return config
