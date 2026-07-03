"""FLUX Kontext resolution helpers."""

from __future__ import annotations

from typing import List, Tuple

PREFERRED_KONTEXT_RESOLUTIONS = [
    (672, 1568),
    (688, 1504),
    (720, 1456),
    (752, 1392),
    (800, 1328),
    (832, 1248),
    (880, 1184),
    (944, 1104),
    (1024, 1024),
    (1104, 944),
    (1184, 880),
    (1248, 832),
    (1328, 800),
    (1392, 752),
    (1456, 720),
    (1504, 688),
    (1568, 672),
]

__all__ = ["PREFERRED_KONTEXT_RESOLUTIONS", "pick_preferred_resolution"]


def pick_preferred_resolution(
    target_hw: Tuple[int, int],
    candidates: List[Tuple[int, int]] = PREFERRED_KONTEXT_RESOLUTIONS,
) -> Tuple[int, int]:
    """Select closest Kontext recommended resolution by aspect ratio then area."""

    target_h, target_w = target_hw
    if target_h <= 0 or target_w <= 0:
        raise ValueError(f"Invalid target size {target_hw}")
    target_ratio = target_w / target_h
    target_area = target_h * target_w

    def metric(hw: Tuple[int, int]) -> Tuple[float, float]:
        height, width = hw
        ratio = width / height
        area = height * width
        return (abs(ratio - target_ratio), abs(area - target_area))

    return min(candidates, key=metric)
