from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest
import yaml

from pba.geometry.config import Config, load_config
from pba.geometry.linear_probe import run_train


REPO_ROOT = Path(__file__).resolve().parents[1]


def write_yaml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data), encoding="utf-8")


def load_module(path: Path, name: str):
    spec = spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {path}")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_load_config_merges_model_template_and_resolves_paths(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    project_root = repo_root / "geometry_probing" / "umd_linear_probing"
    config_path = project_root / "configs" / "dinov2.yaml"
    template_path = project_root / "configs" / "models" / "dinov2_template.yaml"
    write_yaml(
        template_path,
        {
            "name": "dinov2",
            "repo_path": "../../models/dinov2",
            "checkpoint_path": "../../models/dinov2.pth",
        },
    )
    write_yaml(
        config_path,
        {
            "dataset": {
                "root": "../../datasets/UMD",
                "split_path": "metadata/splits/split.json",
                "geometry": {"manifest_path": "metadata/geometry.json"},
            },
            "model": {
                "config_path": "configs/models/dinov2_template.yaml",
                "params": {"patch_size": 14},
            },
            "training": {"output_dir": "outputs/dinov2"},
        },
    )

    config = load_config(config_path)

    assert isinstance(config, Config)
    assert config["dataset"]["root"] == (repo_root / "datasets/UMD").resolve()
    assert config["dataset"]["split_path"] == (project_root / "metadata/splits/split.json").resolve()
    assert config["dataset"]["geometry"]["manifest_path"] == (project_root / "metadata/geometry.json").resolve()
    assert config["model"]["target"] == "dinov2"
    assert config["model"]["params"]["patch_size"] == 14
    assert config["model"]["params"]["repo_path"] == (repo_root / "models/dinov2").resolve()
    assert config["model"]["params"]["checkpoint_path"] == (repo_root / "models/dinov2.pth").resolve()
    assert config["training"]["output_dir"] == (project_root / "outputs/dinov2").resolve()


def test_load_config_rejects_non_mapping_yaml(tmp_path: Path) -> None:
    config_path = tmp_path / "configs" / "bad.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text("- item\n", encoding="utf-8")

    with pytest.raises(TypeError, match="must contain a mapping"):
        load_config(config_path)


def test_legacy_geometry_config_reexports_pba_config() -> None:
    legacy = load_module(
        REPO_ROOT / "geometry_probing/umd_linear_probing/src/utils/config.py",
        "legacy_geometry_config",
    )

    assert legacy.Config is Config
    assert legacy.load_config is load_config


def test_run_train_applies_local_override(tmp_path: Path) -> None:
    config_path = tmp_path / "base.yaml"
    local_path = tmp_path / "local.yaml"
    write_yaml(
        config_path,
        {
            "dataset": {"root": ".", "split_path": "split.json"},
            "model": {"target": "open_clip", "params": {"model_id": "dummy"}},
            "training": {"seed": 1337},
        },
    )
    write_yaml(local_path, {"training": {"seed": 2026}})

    seen = {}

    class DummyExperiment:
        def __init__(self, config: Config) -> None:
            seen["seed"] = config["training"]["seed"]

        def train(self) -> None:
            return None

    run_train(
        ["--config", str(config_path), "--local", str(local_path)],
        experiment_cls=DummyExperiment,
    )

    assert seen["seed"] == 2026
