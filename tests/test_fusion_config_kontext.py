from pathlib import Path

import pytest
import yaml

from pba.fusion.config import load_config
from pba.fusion.kontext import PREFERRED_KONTEXT_RESOLUTIONS, pick_preferred_resolution


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_load_config_reads_yaml_mapping(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump({"dataset_root": "datasets/AGD20K", "geom_pipeline": {"enable": True}}),
        encoding="utf-8",
    )

    assert load_config(config_path) == {
        "dataset_root": "datasets/AGD20K",
        "geom_pipeline": {"enable": True},
    }


def test_load_config_rejects_non_mapping_yaml(tmp_path: Path) -> None:
    config_path = tmp_path / "bad.yaml"
    config_path.write_text("- item\n", encoding="utf-8")

    with pytest.raises(TypeError, match="must contain a mapping"):
        load_config(config_path)


def test_pick_preferred_resolution_matches_existing_aspect_ratio_rule() -> None:
    assert pick_preferred_resolution((1000, 1000)) == (1024, 1024)
    assert pick_preferred_resolution((960, 1280)) == (880, 1184)
    assert (1024, 1024) in PREFERRED_KONTEXT_RESOLUTIONS


def test_pick_preferred_resolution_rejects_invalid_target() -> None:
    with pytest.raises(ValueError, match="Invalid target size"):
        pick_preferred_resolution((0, 1024))


def test_default_fusion_config_keeps_adaptive_gate_disabled() -> None:
    cfg = load_config(REPO_ROOT / "fusion_zero_shot/src/agd20k_eval/config.yaml")
    geom_cfg = cfg["geom_pipeline"]

    assert cfg["enable_model_cpu_offload"] is False
    assert cfg["enable_sequential_cpu_offload"] is False
    assert cfg["max_area"] is None
    assert geom_cfg["geom_adaptive_fusion"] is False
    assert geom_cfg["geom_gate_min_lambda"] == 0.35
    assert geom_cfg["geom_gate_max_lambda"] == 0.85
    assert geom_cfg["geom_gate_verb_weight"] == 0.45
    assert geom_cfg["geom_gate_geometry_weight"] == 0.35
    assert geom_cfg["geom_gate_alignment_weight"] == 0.20


def test_local_smoke_configs_enable_low_vram_loading() -> None:
    fixed = load_config(REPO_ROOT / "fusion_zero_shot/src/agd20k_eval/config.local.fixed_smoke.yaml")
    adaptive = load_config(REPO_ROOT / "fusion_zero_shot/src/agd20k_eval/config.local.adaptive_smoke.yaml")

    assert fixed["max_area"] == 672 * 672
    assert fixed["enable_model_cpu_offload"] is False
    assert fixed["enable_sequential_cpu_offload"] is True
    assert adaptive["max_area"] == 672 * 672
    assert adaptive["enable_model_cpu_offload"] is False
    assert adaptive["enable_sequential_cpu_offload"] is True
