from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest
import yaml

from pba.geometry.linear_probe import parse_train_args, run_train


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


class FakeExperiment:
    instances = []

    def __init__(self, config):
        self.config = config
        self.trained = False
        FakeExperiment.instances.append(self)

    def train(self) -> None:
        self.trained = True


def test_parse_train_args_accepts_config_path() -> None:
    args = parse_train_args(["--config", "configs/dinov2.yaml"])

    assert args.config == Path("configs/dinov2.yaml")


def test_run_train_loads_config_and_runs_injected_experiment(tmp_path: Path) -> None:
    FakeExperiment.instances.clear()
    config_path = tmp_path / "umd_linear_probing" / "configs" / "dinov2.yaml"
    write_yaml(
        config_path,
        {
            "dataset": {"root": "../../datasets/UMD"},
            "model": {"target": "dinov2", "params": {"name": "dinov2"}},
            "training": {"output_root": "outputs"},
        },
    )

    experiment = run_train(["--config", str(config_path)], experiment_cls=FakeExperiment)

    assert experiment is FakeExperiment.instances[0]
    assert experiment.trained is True
    assert experiment.config["model"]["target"] == "dinov2"


def test_run_train_rejects_missing_config(tmp_path: Path) -> None:
    missing = tmp_path / "missing.yaml"

    with pytest.raises(FileNotFoundError, match="Please provide --config"):
        run_train(["--config", str(missing)], experiment_cls=FakeExperiment)


def test_legacy_train_script_reexports_package_entrypoint() -> None:
    legacy = load_module(
        REPO_ROOT / "geometry_probing/umd_linear_probing/scripts/train.py",
        "legacy_geometry_train_script",
    )

    assert legacy.parse_train_args is parse_train_args
    assert legacy.run_train is run_train
