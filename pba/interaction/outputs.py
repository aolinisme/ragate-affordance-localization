"""Output path and metadata helpers for interaction probing."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

__all__ = ["build_probe_metadata", "prepare_output_root"]


def prepare_output_root(output_root: Path) -> Path:
    out_dir = output_root.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def build_probe_metadata(
    *,
    model_id: str,
    prompt: str,
    affordance: str,
    token_map: Dict[str, int],
    steps: int,
    guidance: float,
    seed: int,
) -> dict:
    return {
        "model_id": model_id,
        "prompt": prompt,
        "affordance": affordance,
        "tokens_tracked": token_map,
        "steps": steps,
        "guidance": guidance,
        "seed": seed,
    }
