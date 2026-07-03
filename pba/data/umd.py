"""Lightweight UMD dataset contract helpers.

This module documents and validates the public migration boundary for the UMD
PyTorch dataset without importing PyTorch, SciPy, OpenCV, or PIL.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping

__all__ = [
    "UMD_GEOMETRY_MANIFEST_KEYS",
    "UMD_OPTIONAL_GEOMETRY_KEYS",
    "UMD_SAMPLE_OUTPUT_KEYS",
    "UMD_SPLIT_RECORD_KEYS",
    "build_geometry_manifest_index",
    "build_umd_sample_paths",
    "resolve_umd_asset_path",
    "validate_umd_sample",
    "validate_umd_split_record",
]


UMD_SPLIT_RECORD_KEYS = ("tool", "frame_id", "rgb", "depth", "label_mat")
UMD_SAMPLE_OUTPUT_KEYS = ("image", "pixel_mask", "patch_mask", "meta")
UMD_OPTIONAL_GEOMETRY_KEYS = ("geom_depth", "geom_normal")
UMD_GEOMETRY_MANIFEST_KEYS = ("pred_depth_npy", "pred_normal_npy")
_GEOMETRY_SPLIT_KEYS = ("train", "val", "test", "data")


def validate_umd_split_record(record: Mapping[str, str]) -> Mapping[str, str]:
    """Validate one split record produced by `pba.data.geometry`."""

    missing = [key for key in UMD_SPLIT_RECORD_KEYS if not record.get(key)]
    if missing:
        raise ValueError(f"UMD split record missing required keys: {', '.join(missing)}")
    return record


def resolve_umd_asset_path(dataset_root: Path, value: str) -> Path:
    """Resolve absolute or UMD-root-relative asset paths."""

    path = Path(value)
    if path.is_absolute():
        return path
    return Path(dataset_root) / path


def build_umd_sample_paths(dataset_root: Path, record: Mapping[str, str]) -> dict[str, Path]:
    """Resolve RGB, depth, and label paths from one UMD split record."""

    validate_umd_split_record(record)
    return {
        "rgb": resolve_umd_asset_path(dataset_root, record["rgb"]),
        "depth": resolve_umd_asset_path(dataset_root, record["depth"]),
        "label_mat": resolve_umd_asset_path(dataset_root, record["label_mat"]),
    }


def build_geometry_manifest_index(manifest: Mapping[str, Any] | list[Mapping[str, Any]]) -> dict[str, dict[str, str]]:
    """Index optional geometry assets by frame id.

    Accepted manifests match the current legacy dataset behavior: either a flat
    list of records, or a mapping containing split keys such as `train`, `val`,
    `test`, or `data`.
    """

    index: dict[str, dict[str, str]] = {}
    for item in _iter_geometry_manifest_items(manifest):
        frame_id = item.get("frame_id")
        if not isinstance(frame_id, str) or not frame_id:
            continue
        assets = {
            key: value
            for key in UMD_GEOMETRY_MANIFEST_KEYS
            if isinstance((value := item.get(key)), str) and value
        }
        if assets:
            index[frame_id] = assets
    return index


def validate_umd_sample(
    sample: Mapping[str, Any],
    *,
    require_geometry: Iterable[str] = (),
) -> Mapping[str, Any]:
    """Validate runtime sample keys expected from the migrated PyTorch dataset."""

    missing = [key for key in UMD_SAMPLE_OUTPUT_KEYS if key not in sample]
    missing.extend(key for key in require_geometry if key not in sample)
    if missing:
        raise ValueError(f"UMD sample missing required keys: {', '.join(missing)}")

    meta = sample["meta"]
    if not isinstance(meta, Mapping):
        raise ValueError("UMD sample meta must be a mapping.")
    for key in ("tool", "frame_id"):
        if not meta.get(key):
            raise ValueError(f"UMD sample meta missing required key: {key}")
    return sample


def _iter_geometry_manifest_items(
    manifest: Mapping[str, Any] | list[Mapping[str, Any]],
) -> Iterable[Mapping[str, Any]]:
    if isinstance(manifest, list):
        yield from (item for item in manifest if isinstance(item, Mapping))
        return

    for split in _GEOMETRY_SPLIT_KEYS:
        value = manifest.get(split)
        if isinstance(value, list):
            yield from (item for item in value if isinstance(item, Mapping))
