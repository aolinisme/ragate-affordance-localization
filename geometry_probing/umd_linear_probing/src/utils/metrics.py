"""Compatibility wrapper for shared segmentation metrics."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pba.metrics.segmentation import ConfusionMatrix, compute_iou, update_confusion_matrix

__all__ = ["ConfusionMatrix", "update_confusion_matrix", "compute_iou"]
