"""Compatibility exports for UMD split utilities."""

from __future__ import annotations

import sys
from pathlib import Path

try:
    from pba.data.geometry import (  # noqa: F401
        CategorySplitEntry,
        build_instance_index,
        build_split_record,
        parse_category_split,
        save_split_mapping,
        train_val_test_split,
    )
except ModuleNotFoundError:  # pragma: no cover - direct legacy script fallback
    repo_root = Path(__file__).resolve().parents[5]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from pba.data.geometry import (  # noqa: F401
        CategorySplitEntry,
        build_instance_index,
        build_split_record,
        parse_category_split,
        save_split_mapping,
        train_val_test_split,
    )

__all__ = [
    "CategorySplitEntry",
    "build_instance_index",
    "build_split_record",
    "parse_category_split",
    "save_split_mapping",
    "train_val_test_split",
]
