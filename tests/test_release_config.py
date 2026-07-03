from pathlib import Path

import pytest
import yaml

from pba.config import ReproduceConfig, load_reproduce_config


def write_yaml(path: Path, data: dict) -> None:
    path.write_text(yaml.safe_dump(data), encoding="utf-8")


def test_load_reproduce_config_resolves_paths_relative_to_repo(tmp_path: Path) -> None:
    cfg_path = tmp_path / "main.yaml"
    write_yaml(
        cfg_path,
        {
            "assets": {
                "umd_root": "datasets/UMD/part-affordance-dataset",
                "agd20k_root": "datasets/AGD20K/AGD20K/Unseen/testset",
                "flux_model": "models/FLUX.1-Kontext-dev",
                "dinov2_source": "models/dinov2",
                "dinov3_source": "models/dinov3",
                "dino_source": "models/dino",
                "openclip_source": "models/open_clip/src",
                "dinov2_checkpoint": "models/dinov2_vitb14_pretrain.pth",
                "dinov3_checkpoint": "models/dinov3_vit7b16_pretrain_lvd1689m.pth",
                "dino_checkpoint": "models/dino_vitbase16_pretrain.pth",
                "sam_checkpoint": "models/sam_vit_b_01ec64.pth",
            },
            "outputs": {"root": "outputs/reproduce_main"},
            "jobs": {
                "geometry": [
                    {
                        "name": "dinov2",
                        "command": "geometry-train",
                        "config": "geometry_probing/umd_linear_probing/configs/dinov2.yaml",
                    }
                ],
                "interaction": [
                    {
                        "name": "flux",
                        "command": "interaction-probe",
                        "config": "configs/interaction/flux_kontext.yaml",
                    }
                ],
                "fusion": [
                    {
                        "name": "dinov3_flux",
                        "command": "fusion-eval",
                        "config": "fusion_zero_shot/src/agd20k_eval/config.yaml",
                    }
                ],
            },
        },
    )

    config = load_reproduce_config(cfg_path, repo_root=Path("/repo"))

    assert isinstance(config, ReproduceConfig)
    assert config.assets["umd_root"] == Path("/repo/datasets/UMD/part-affordance-dataset")
    assert config.outputs_root == Path("/repo/outputs/reproduce_main")
    assert config.jobs["geometry"][0]["name"] == "dinov2"


def test_load_reproduce_config_rejects_missing_top_level_keys(tmp_path: Path) -> None:
    cfg_path = tmp_path / "bad.yaml"
    write_yaml(cfg_path, {"assets": {}, "outputs": {}})

    with pytest.raises(ValueError, match="Missing required config keys: jobs"):
        load_reproduce_config(cfg_path, repo_root=tmp_path)
