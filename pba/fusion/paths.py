"""Path helpers for fusion zero-shot evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from pba.fusion.prompts import sanitize_token_name

__all__ = ["ensure_dir", "get_heatmap_path"]


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_heatmap_path(per_token_dir: Path, index: int, token: Optional[str] = None) -> Path:
    if token:
        sanitized = sanitize_token_name(token)
        token_matches = sorted(per_token_dir.glob(f"heat_tok*_{sanitized}.png"))
        if token_matches:
            return token_matches[0]
    alias = per_token_dir / f"heat_tok{index:02d}.png"
    if alias.exists():
        return alias
    pattern = f"heat_tok{index:02d}_*.png"
    matches = list(per_token_dir.glob(pattern))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"No heatmap found for index {index} (token={token}) in {per_token_dir}")
