"""Stable Diffusion 2.1 local/offline model contract."""

from __future__ import annotations

from pathlib import Path

SD21_DEFAULT_MODEL_ID = "stabilityai/stable-diffusion-2-1"
SD21_LOCAL_MODEL_DIR = Path("models/stable-diffusion-2-1")
SD21_REQUIRED_SUBDIRS = ("unet", "scheduler", "tokenizer", "text_encoder", "vae")

__all__ = [
    "SD21_DEFAULT_MODEL_ID",
    "SD21_LOCAL_MODEL_DIR",
    "SD21_REQUIRED_SUBDIRS",
    "resolve_sd21_model_reference",
    "validate_sd21_local_model_dir",
]


def validate_sd21_local_model_dir(model_dir: Path) -> Path:
    if not model_dir.exists():
        raise FileNotFoundError(f"SD2.1 local model directory not found: {model_dir}")
    missing = [name for name in SD21_REQUIRED_SUBDIRS if not (model_dir / name).is_dir()]
    if missing:
        raise ValueError(f"missing SD2.1 local model subdirectories: {', '.join(missing)}")
    return model_dir


def resolve_sd21_model_reference(
    *,
    model_id: str = SD21_DEFAULT_MODEL_ID,
    local_model_dir: Path | None = None,
    offline: bool = False,
) -> str | Path:
    if local_model_dir is not None:
        if offline:
            if not local_model_dir.exists():
                raise FileNotFoundError(
                    f"offline SD2.1 mode requires local model directory: {local_model_dir}"
                )
            return validate_sd21_local_model_dir(local_model_dir)
        return local_model_dir if local_model_dir.exists() else model_id

    if offline:
        if not SD21_LOCAL_MODEL_DIR.exists():
            raise FileNotFoundError(
                f"offline SD2.1 mode requires local model directory: {SD21_LOCAL_MODEL_DIR}"
            )
        return validate_sd21_local_model_dir(SD21_LOCAL_MODEL_DIR)

    return model_id
