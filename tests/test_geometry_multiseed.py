from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import json
import sys
from typing import Optional

import pytest
import yaml

from pba.geometry.multiseed import parse_multiseed_args, run_multiseed


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


def test_parse_multiseed_args_accepts_required_fields() -> None:
    args = parse_multiseed_args(
        ["--config", "configs/clip.yaml", "--seeds", "1337", "1338", "--label", "demo", "--hf-offline"]
    )

    assert args.config == Path("configs/clip.yaml")
    assert args.seeds == [1337, 1338]
    assert args.label == "demo"
    assert args.hf_offline is True


def test_run_multiseed_writes_seed_specific_outputs_and_aggregate(tmp_path: Path) -> None:
    project_root = tmp_path / "geometry_probing" / "umd_linear_probing"
    config_path = project_root / "configs" / "clip.yaml"
    output_root = project_root / "outputs"
    write_yaml(
        config_path,
        {
            "dataset": {"root": "../../datasets/UMD", "split_path": "metadata/splits/split.json"},
            "model": {
                "target": "open_clip",
                "params": {"model_id": "dummy", "output_dir_name": "CLIP_capped_geom_3070"},
            },
            "training": {"output_root": "outputs", "seed": 1337},
        },
    )

    seed_metrics = {
        1337: {"val": 0.20, "test": 0.24},
        1338: {"val": 0.22, "test": 0.30},
    }

    def fake_run_command(
        command: list[str], *, cwd: Path, env: Optional[dict[str, str]] = None
    ) -> None:
        del cwd
        del env
        seed = int(Path(command[-1]).stem.split("_")[-1])
        override_path = Path(command[-1])
        with override_path.open("r", encoding="utf-8") as handle:
            override = yaml.safe_load(handle)
        output_name = override["model"]["params"]["output_dir_name"]
        run_dir = output_root / f"{output_name}_20260616-220000"
        sweep_dir = run_dir / "lr1e-03_wd1e-02"
        sweep_dir.mkdir(parents=True, exist_ok=True)
        (sweep_dir / "linear_probe.pth").write_text("stub", encoding="utf-8")
        summary = {
            "best_val": {"miou": seed_metrics[seed]["val"]},
            "test_metrics": {"miou": seed_metrics[seed]["test"]},
            "checkpoint_path": str(sweep_dir / "linear_probe.pth"),
            "run_timestamp": "20260616-220000",
        }
        (run_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")

    result = run_multiseed(
        [
            "--config",
            str(config_path),
            "--seeds",
            "1337",
            "1338",
            "--python",
            sys.executable,
        ],
        run_command=fake_run_command,
    )

    assert result["base_output_name"] == "CLIP_capped_geom_3070"
    assert result["hf_offline"] is False
    assert [run["seed"] for run in result["runs"]] == [1337, 1338]
    assert result["runs"][0]["output_dir"].endswith("CLIP_capped_geom_3070_seed1337_20260616-220000")
    assert result["aggregate"]["best_val_miou"]["mean"] == pytest.approx(0.21)
    assert result["aggregate"]["test_miou"]["mean"] == pytest.approx(0.27)


def test_legacy_multiseed_script_reexports_package_entrypoint() -> None:
    legacy = load_module(
        REPO_ROOT / "geometry_probing/umd_linear_probing/scripts/multiseed.py",
        "legacy_geometry_multiseed_script",
    )

    assert legacy.parse_multiseed_args is parse_multiseed_args
    assert legacy.run_multiseed is run_multiseed
