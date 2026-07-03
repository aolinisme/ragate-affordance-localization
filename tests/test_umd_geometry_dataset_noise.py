import json
import sys
from pathlib import Path

import numpy as np
import scipy.io as sio
from PIL import Image
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = REPO_ROOT / "geometry_probing" / "umd_linear_probing"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import UMDAffordanceDataset


def _write_fixture(tmp_path: Path) -> tuple[Path, dict, Path]:
    root = tmp_path / "umd"
    tool_dir = root / "tools" / "knife_01"
    geom_dir = root / "geom"
    tool_dir.mkdir(parents=True)
    geom_dir.mkdir(parents=True)

    image = np.full((16, 16, 3), 128, dtype=np.uint8)
    Image.fromarray(image).save(tool_dir / "knife_01_00000001_rgb.jpg")
    sio.savemat(tool_dir / "knife_01_00000001_label.mat", {"gt_label": np.ones((16, 16), dtype=np.int64)})

    depth = np.linspace(0.0, 1.0, 16 * 16, dtype=np.float32).reshape(16, 16)
    np.save(geom_dir / "knife_01_00000001_depth.npy", depth)
    manifest = {
        "data": [
            {
                "frame_id": "knife_01_00000001",
                "pred_depth_npy": "geom/knife_01_00000001_depth.npy",
            }
        ]
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    record = {
        "tool": "knife_01",
        "frame_id": "knife_01_00000001",
        "rgb": "tools/knife_01/knife_01_00000001_rgb.jpg",
        "label_mat": "tools/knife_01/knife_01_00000001_label.mat",
    }
    return root, record, manifest_path


def test_depth_noise_is_disabled_by_default(tmp_path: Path) -> None:
    root, record, manifest_path = _write_fixture(tmp_path)
    dataset = UMDAffordanceDataset(
        dataset_root=root,
        split_records=[record],
        patch_size=8,
        geometry={
            "manifest_path": str(manifest_path),
            "use_depth": True,
            "pool": {"kernel": 8, "stride": 8},
        },
    )

    sample_a = dataset[0]["geom_depth"]
    sample_b = dataset[0]["geom_depth"]

    assert torch.equal(sample_a, sample_b)
    assert sample_a.shape == (1, 2, 2)


def test_depth_dropout_noise_can_zero_geometry(tmp_path: Path) -> None:
    root, record, manifest_path = _write_fixture(tmp_path)
    dataset = UMDAffordanceDataset(
        dataset_root=root,
        split_records=[record],
        patch_size=8,
        geometry={
            "manifest_path": str(manifest_path),
            "use_depth": True,
            "pool": {"kernel": 8, "stride": 8},
            "depth_noise": {"dropout_prob": 1.0},
        },
    )

    sample = dataset[0]["geom_depth"]

    assert torch.count_nonzero(sample) == 0
