from pathlib import Path

import pytest

from pba.data.umd import (
    UMD_GEOMETRY_MANIFEST_KEYS,
    UMD_SAMPLE_OUTPUT_KEYS,
    build_geometry_manifest_index,
    build_umd_sample_paths,
    validate_umd_sample,
    validate_umd_split_record,
)


def test_validate_umd_split_record_accepts_existing_split_schema() -> None:
    record = {
        "tool": "hammer",
        "frame_id": "hammer_01_000001",
        "rgb": "tools/hammer/hammer_01_000001_rgb.jpg",
        "depth": "tools/hammer/hammer_01_000001_depth.png",
        "label_mat": "tools/hammer/hammer_01_000001_label.mat",
    }

    assert validate_umd_split_record(record) == record


def test_validate_umd_split_record_rejects_missing_keys() -> None:
    with pytest.raises(ValueError, match="label_mat"):
        validate_umd_split_record({"tool": "hammer", "frame_id": "hammer_01_000001", "rgb": "x.jpg"})


def test_build_umd_sample_paths_resolves_legacy_relative_assets(tmp_path: Path) -> None:
    paths = build_umd_sample_paths(
        tmp_path,
        {
            "tool": "hammer",
            "frame_id": "hammer_01_000001",
            "rgb": "tools/hammer/hammer_01_000001_rgb.jpg",
            "depth": "tools/hammer/hammer_01_000001_depth.png",
            "label_mat": "tools/hammer/hammer_01_000001_label.mat",
        },
    )

    assert paths["rgb"] == tmp_path / "tools/hammer/hammer_01_000001_rgb.jpg"
    assert paths["depth"] == tmp_path / "tools/hammer/hammer_01_000001_depth.png"
    assert paths["label_mat"] == tmp_path / "tools/hammer/hammer_01_000001_label.mat"


def test_build_geometry_manifest_index_accepts_split_or_flat_manifest() -> None:
    manifest = {
        "train": [{"frame_id": "a", "pred_depth_npy": "depth/a.npy"}],
        "val": [{"frame_id": "b", "pred_normal_npy": "normal/b.npy"}],
        "metadata": {"ignored": True},
    }

    index = build_geometry_manifest_index(manifest)

    assert index["a"] == {"pred_depth_npy": "depth/a.npy"}
    assert index["b"] == {"pred_normal_npy": "normal/b.npy"}
    assert UMD_GEOMETRY_MANIFEST_KEYS == ("pred_depth_npy", "pred_normal_npy")


def test_validate_umd_sample_documents_runtime_output_keys() -> None:
    sample = {
        "image": object(),
        "pixel_mask": object(),
        "patch_mask": object(),
        "meta": {"tool": "hammer", "frame_id": "hammer_01_000001"},
        "geom_depth": object(),
    }

    assert validate_umd_sample(sample, require_geometry=("geom_depth",)) == sample
    assert UMD_SAMPLE_OUTPUT_KEYS == ("image", "pixel_mask", "patch_mask", "meta")


def test_validate_umd_sample_rejects_missing_geometry_features() -> None:
    sample = {
        "image": object(),
        "pixel_mask": object(),
        "patch_mask": object(),
        "meta": {"tool": "hammer", "frame_id": "hammer_01_000001"},
    }

    with pytest.raises(ValueError, match="geom_normal"):
        validate_umd_sample(sample, require_geometry=("geom_normal",))
