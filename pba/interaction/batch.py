"""Batch interaction evaluation contract for AGD20K.

This module describes paths and result columns for the missing batch runner.
It does not execute Flux Kontext.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from pba.data.affordance import SampleEntry

INTERACTION_BATCH_OUTPUT_FILES = frozenset(
    {
        "verb_heatmap",
        "verb_heatmap_npy",
        "metadata",
    }
)

BATCH_METRIC_COLUMNS = (
    "sample_id",
    "affordance",
    "object_name",
    "image_path",
    "gt_path",
    "verb_heatmap",
    "kld",
    "sim",
    "nss",
)

__all__ = [
    "BATCH_METRIC_COLUMNS",
    "INTERACTION_BATCH_OUTPUT_FILES",
    "build_batch_sample_id",
    "build_batch_sample_paths",
    "validate_metric_columns",
]


def build_batch_sample_id(sample: SampleEntry) -> str:
    return f"{sample.affordance}/{sample.object_name}/{sample.image_path.stem}"


def build_batch_sample_paths(output_root: Path, sample: SampleEntry) -> dict[str, Path]:
    sample_dir = output_root / sample.affordance / sample.object_name / sample.image_path.stem
    return {
        "verb_heatmap": sample_dir / "verb_heat.png",
        "verb_heatmap_npy": sample_dir / "verb_heat.npy",
        "metadata": sample_dir / "meta.json",
    }


def validate_metric_columns(columns: Iterable[str]) -> tuple[str, ...]:
    values = tuple(columns)
    missing = [column for column in BATCH_METRIC_COLUMNS if column not in values]
    if missing:
        raise ValueError(f"missing metric columns: {', '.join(missing)}")
    return values
