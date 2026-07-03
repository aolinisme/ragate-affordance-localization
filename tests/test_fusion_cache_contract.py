from pathlib import Path

import numpy as np
import pytest

from pba.fusion.cache import DINO_CACHE_KEYS
from pba.fusion.cache import build_cache_path
from pba.fusion.cache import cache_filename
from pba.fusion.cache import validate_cache_file


def test_cache_filename_matches_fusion_runtime_naming() -> None:
    assert cache_filename(Path("images/knife.jpg"), (1280, 960), 16) == "knife_1280x960_p16.npz"


def test_build_cache_path_returns_expected_path(tmp_path: Path) -> None:
    cache_root = tmp_path / "fusion_cache"

    path = build_cache_path(cache_root, Path("cup.png"), (1260, 980), 14)

    assert path == cache_root / "cup_1260x980_p14.npz"


def test_validate_cache_file_accepts_required_keys(tmp_path: Path) -> None:
    cache_file = tmp_path / "sample_1280x960_p16.npz"
    np.savez_compressed(
        cache_file,
        tokens=np.zeros((4, 3), dtype=np.float32),
        Hp=2,
        Wp=2,
        meta={"orig_h": 10, "orig_w": 20},
    )

    report = validate_cache_file(cache_file)

    assert report.path == cache_file
    assert report.keys == DINO_CACHE_KEYS
    assert report.tokens_shape == (4, 3)


def test_validate_cache_file_rejects_missing_required_key(tmp_path: Path) -> None:
    cache_file = tmp_path / "sample_1280x960_p16.npz"
    np.savez_compressed(cache_file, tokens=np.zeros((4, 3), dtype=np.float32), Hp=2, meta={})

    with pytest.raises(ValueError, match="missing DINO cache keys: Wp"):
        validate_cache_file(cache_file)


def test_legacy_pca_stage_uses_shared_cache_contract() -> None:
    source = Path("fusion_zero_shot/src/pipeline/pca_stage.py").read_text(encoding="utf-8")

    assert "from pba.fusion.cache import build_cache_path" in source
    assert "cache_only: bool = False" in source
    assert "if cache_only:" in source
