from pathlib import Path

import pytest

from pba.geometry.sd21 import SD21_DEFAULT_MODEL_ID
from pba.geometry.sd21 import SD21_LOCAL_MODEL_DIR
from pba.geometry.sd21 import resolve_sd21_model_reference
from pba.geometry.sd21 import validate_sd21_local_model_dir


def test_resolve_sd21_model_reference_prefers_explicit_local_dir(tmp_path: Path) -> None:
    model_dir = tmp_path / "stable-diffusion-2-1"
    (model_dir / "unet").mkdir(parents=True)
    (model_dir / "scheduler").mkdir()
    (model_dir / "tokenizer").mkdir()
    (model_dir / "text_encoder").mkdir()
    (model_dir / "vae").mkdir()

    assert resolve_sd21_model_reference(local_model_dir=model_dir, offline=True) == model_dir


def test_resolve_sd21_model_reference_allows_online_hf_id_by_default() -> None:
    assert resolve_sd21_model_reference() == SD21_DEFAULT_MODEL_ID


def test_resolve_sd21_model_reference_requires_local_dir_when_offline(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="offline SD2.1 mode requires local model directory"):
        resolve_sd21_model_reference(local_model_dir=tmp_path / "missing", offline=True)


def test_validate_sd21_local_model_dir_reports_missing_subfolders(tmp_path: Path) -> None:
    model_dir = tmp_path / "sd21"
    model_dir.mkdir()

    with pytest.raises(ValueError, match="missing SD2.1 local model subdirectories"):
        validate_sd21_local_model_dir(model_dir)


def test_default_local_model_dir_documents_public_convention() -> None:
    assert SD21_LOCAL_MODEL_DIR == Path("models/stable-diffusion-2-1")
