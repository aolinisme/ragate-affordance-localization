#!/usr/bin/env python
"""Build lightweight geometry side-data for the smoke UMD split."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[3]
DATASET_ROOT = REPO_ROOT / "datasets" / "UMD" / "part-affordance-dataset"
SMOKE_SPLIT = REPO_ROOT / "geometry_probing" / "umd_linear_probing" / "metadata" / "splits" / "category_split_seed42_v20_smoke.json"
OUT_ROOT = DATASET_ROOT / "smoke_geometry"
MANIFEST_PATH = REPO_ROOT / "geometry_probing" / "umd_linear_probing" / "metadata" / "splits" / "smoke_metric3d_predictions.json"


def load_depth(path: Path) -> np.ndarray:
    depth = np.array(Image.open(path)).astype(np.float32)
    return depth


def normalize_depth(depth: np.ndarray) -> np.ndarray:
    finite = np.isfinite(depth)
    if not np.any(finite):
        return np.zeros_like(depth, dtype=np.float32)
    values = depth[finite]
    low = np.percentile(values, 2.0)
    high = np.percentile(values, 98.0)
    if high <= low + 1e-6:
        return np.zeros_like(depth, dtype=np.float32)
    depth = np.clip((depth - low) / (high - low), 0.0, 1.0)
    return depth.astype(np.float32)


def depth_to_normal(depth: np.ndarray) -> np.ndarray:
    dzdy, dzdx = np.gradient(depth)
    nx = -dzdx
    ny = -dzdy
    nz = np.ones_like(depth, dtype=np.float32)
    normal = np.stack([nx, ny, nz], axis=-1)
    denom = np.linalg.norm(normal, axis=-1, keepdims=True)
    denom = np.maximum(denom, 1e-6)
    normal = normal / denom
    return normal.astype(np.float32)


def ensure_dirs() -> dict[str, Path]:
    dirs = {
        "train_depth": OUT_ROOT / "train_depth",
        "train_normal": OUT_ROOT / "train_normal",
        "val_depth": OUT_ROOT / "val_depth",
        "val_normal": OUT_ROOT / "val_normal",
        "test_depth": OUT_ROOT / "test_depth",
        "test_normal": OUT_ROOT / "test_normal",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def main() -> None:
    with SMOKE_SPLIT.open("r", encoding="utf-8") as handle:
        split_data = json.load(handle)

    out_dirs = ensure_dirs()
    manifest: dict[str, list[dict[str, object]]] = {"train": [], "val": [], "test": []}

    for split_name in ("train", "val", "test"):
        depth_dir = out_dirs[f"{split_name}_depth"]
        normal_dir = out_dirs[f"{split_name}_normal"]
        for item in split_data[split_name]:
            frame_id = item["frame_id"]
            depth_png = DATASET_ROOT / item["depth"]
            depth = load_depth(depth_png)
            depth_norm = normalize_depth(depth)
            normal = depth_to_normal(depth_norm)

            depth_npy_name = f"{frame_id}.npy"
            normal_npy_name = f"{frame_id}.npy"
            np.save(depth_dir / depth_npy_name, depth_norm)
            np.save(normal_dir / normal_npy_name, normal)

            manifest[split_name].append(
                {
                    "frame_id": frame_id,
                    "rgb": item["rgb"],
                    "pred_depth_npy": f"smoke_geometry/{split_name}_depth/{depth_npy_name}",
                    "pred_normal_npy": f"smoke_geometry/{split_name}_normal/{normal_npy_name}",
                    "shape": list(depth.shape),
                }
            )

    with MANIFEST_PATH.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    print(f"Wrote geometry assets to {OUT_ROOT}")
    print(f"Wrote manifest to {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
