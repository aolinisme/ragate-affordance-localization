"""DINO token cache contract for fusion evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

DINO_CACHE_KEYS = ("tokens", "Hp", "Wp", "meta")

__all__ = [
    "DINO_CACHE_KEYS",
    "DINOCacheReport",
    "build_cache_path",
    "cache_filename",
    "validate_cache_file",
]


@dataclass(frozen=True)
class DINOCacheReport:
    path: Path
    keys: tuple[str, ...]
    tokens_shape: tuple[int, ...]
    hp: int
    wp: int


def cache_filename(image_path: Path, target_wh: tuple[int, int], patch_size: int) -> str:
    return f"{image_path.stem}_{target_wh[0]}x{target_wh[1]}_p{patch_size}.npz"


def build_cache_path(cache_root: Path, image_path: Path, target_wh: tuple[int, int], patch_size: int) -> Path:
    return cache_root / cache_filename(image_path, target_wh, patch_size)


def validate_cache_file(path: Path) -> DINOCacheReport:
    if not path.exists():
        raise FileNotFoundError(f"DINO cache file not found: {path}")

    with np.load(path, allow_pickle=True) as data:
        keys = tuple(data.files)
        missing = [key for key in DINO_CACHE_KEYS if key not in data.files]
        if missing:
            raise ValueError(f"missing DINO cache keys: {', '.join(missing)}")
        tokens = data["tokens"]
        if tokens.ndim != 2:
            raise ValueError(f"DINO cache tokens must be 2D, got shape {tokens.shape}")
        hp = int(data["Hp"])
        wp = int(data["Wp"])

    return DINOCacheReport(path=path, keys=DINO_CACHE_KEYS, tokens_shape=tuple(tokens.shape), hp=hp, wp=wp)
