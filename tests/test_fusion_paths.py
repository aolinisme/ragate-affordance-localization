from pathlib import Path

import pytest

from pba.fusion.paths import ensure_dir, get_heatmap_path


def test_ensure_dir_creates_directory_and_returns_path(tmp_path: Path) -> None:
    path = tmp_path / "outputs" / "sample"

    assert ensure_dir(path) == path
    assert path.is_dir()


def test_get_heatmap_path_prefers_sanitized_token_match(tmp_path: Path) -> None:
    fallback = tmp_path / "heat_tok02_other.png"
    preferred = tmp_path / "heat_tok99_cup_with_handle.png"
    fallback.write_bytes(b"fallback")
    preferred.write_bytes(b"preferred")

    assert get_heatmap_path(tmp_path, 2, "cup/with\\handle") == preferred


def test_get_heatmap_path_falls_back_to_token_index(tmp_path: Path) -> None:
    path = tmp_path / "heat_tok03_object.png"
    path.write_bytes(b"heat")

    assert get_heatmap_path(tmp_path, 3, "missing") == path


def test_get_heatmap_path_raises_when_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="No heatmap found"):
        get_heatmap_path(tmp_path, 7, "missing")
