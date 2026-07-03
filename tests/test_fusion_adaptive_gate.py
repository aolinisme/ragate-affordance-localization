from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
GEOMETRY_STAGE = REPO_ROOT / "fusion_zero_shot/src/pipeline/geometry_stage.py"


def load_geometry_stage():
    spec = spec_from_file_location("fusion_geometry_stage_for_tests", GEOMETRY_STAGE)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {GEOMETRY_STAGE}")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def peak_map(size: int = 16, peak: tuple[int, int] = (7, 7), value: float = 1.0) -> np.ndarray:
    arr = np.zeros((size, size), dtype=np.float32)
    row, col = peak
    arr[row, col] = value
    arr[max(0, row - 1) : min(size, row + 2), max(0, col - 1) : min(size, col + 2)] += value * 0.25
    arr[row, col] = value
    return arr


def test_adaptive_gate_disabled_preserves_fixed_lambda_params() -> None:
    stage = load_geometry_stage()
    pcs = np.dstack([peak_map(), np.zeros((16, 16), dtype=np.float32)])
    verb = peak_map(peak=(7, 7))

    result = stage.generate_geometry_mask(
        pcs,
        verb_map=verb,
        enable_soft_fusion=True,
        soft_lambda=0.65,
        smooth_sigma=0.0,
    )

    assert "adaptive_fusion" not in result
    assert result["soft_fusion"]["params"]["lambda"] == 0.65
    assert result["soft_fusion"]["params"]["adaptive_fusion"] is None


def test_adaptive_gate_enabled_populates_metadata_and_clamps_lambda() -> None:
    stage = load_geometry_stage()
    pcs = np.dstack([peak_map(), np.zeros((16, 16), dtype=np.float32)])
    verb = peak_map(peak=(7, 7))

    result = stage.generate_geometry_mask(
        pcs,
        verb_map=verb,
        enable_soft_fusion=True,
        adaptive_fusion=True,
        smooth_sigma=0.0,
        gate_min_lambda=0.35,
        gate_max_lambda=0.85,
    )

    meta = result["adaptive_fusion"]
    assert meta["adaptive_fusion_enabled"] is True
    assert 0.35 <= meta["adaptive_lambda"] <= 0.85
    assert 0.0 <= meta["interaction_confidence"] <= 1.0
    assert 0.0 <= meta["geometry_confidence"] <= 1.0
    assert 0.0 <= meta["alignment_confidence"] <= 1.0
    assert result["soft_fusion"]["params"]["lambda"] == meta["adaptive_lambda"]


def test_adaptive_gate_trusts_interaction_more_when_geometry_is_weak() -> None:
    stage = load_geometry_stage()
    weak_geometry = np.full((16, 16), 0.5, dtype=np.float32)
    pcs = np.dstack([weak_geometry, np.zeros((16, 16), dtype=np.float32)])
    verb = peak_map(peak=(7, 7))

    result = stage.generate_geometry_mask(
        pcs,
        verb_map=verb,
        enable_soft_fusion=True,
        adaptive_fusion=True,
        smooth_sigma=0.0,
        forced_pc_index=1,
        soft_lambda=0.65,
        gate_min_lambda=0.35,
        gate_max_lambda=0.85,
    )

    assert result["adaptive_fusion"]["adaptive_lambda"] > 0.65


def test_adaptive_gate_trusts_geometry_more_when_geometry_aligns() -> None:
    stage = load_geometry_stage()
    aligned_geometry = peak_map(peak=(7, 7))
    pcs = np.dstack([aligned_geometry, np.zeros((16, 16), dtype=np.float32)])
    verb = peak_map(peak=(7, 7))

    result = stage.generate_geometry_mask(
        pcs,
        verb_map=verb,
        enable_soft_fusion=True,
        adaptive_fusion=True,
        smooth_sigma=0.0,
        forced_pc_index=1,
        soft_lambda=0.65,
        gate_min_lambda=0.35,
        gate_max_lambda=0.85,
    )

    assert result["adaptive_fusion"]["adaptive_lambda"] < 0.65
    assert result["adaptive_fusion"]["alignment_confidence"] > 0.9


def test_adaptive_gate_similarity_floor_falls_back_to_base_lambda() -> None:
    stage = load_geometry_stage()
    misaligned_geometry = peak_map(peak=(2, 2))
    pcs = np.dstack([misaligned_geometry, np.zeros((16, 16), dtype=np.float32)])
    verb = peak_map(peak=(12, 12))

    result = stage.generate_geometry_mask(
        pcs,
        verb_map=verb,
        enable_soft_fusion=True,
        adaptive_fusion=True,
        smooth_sigma=0.0,
        forced_pc_index=1,
        soft_lambda=0.65,
        gate_min_lambda=0.20,
        gate_max_lambda=0.90,
        gate_similarity_floor=0.9,
        gate_fallback_lambda=0.65,
    )

    meta = result["adaptive_fusion"]
    assert meta["fallback_used"] is True
    assert meta["adaptive_lambda"] == 0.65
    assert result["soft_fusion"]["params"]["lambda"] == 0.65
