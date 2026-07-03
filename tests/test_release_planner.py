from pathlib import Path
import subprocess
import sys

import yaml

from pba.config import ReproduceConfig
from pba.planner import PlannedCommand, build_reproduction_plan


def test_build_reproduction_plan_uses_existing_run_py_commands() -> None:
    cfg = ReproduceConfig(
        path=Path("/repo/configs/reproduce/main.yaml"),
        repo_root=Path("/repo"),
        assets={},
        outputs_root=Path("/repo/outputs/reproduce_main"),
        jobs={
            "geometry": [
                {
                    "name": "dinov2",
                    "command": "geometry-train",
                    "args": ["--config", "geometry_probing/umd_linear_probing/configs/dinov2.yaml"],
                }
            ],
            "fusion": [
                {
                    "name": "dinov3_flux",
                    "command": "fusion-eval",
                    "args": ["--config", "fusion_zero_shot/src/agd20k_eval/config.yaml"],
                }
            ],
        },
    )

    plan = build_reproduction_plan(cfg)

    assert plan == [
        PlannedCommand(
            group="geometry",
            name="dinov2",
            argv=[
                "python",
                "-m",
                "pba.run",
                "geometry-train",
                "--",
                "--config",
                "geometry_probing/umd_linear_probing/configs/dinov2.yaml",
            ],
        ),
        PlannedCommand(
            group="fusion",
            name="dinov3_flux",
            argv=[
                "python",
                "-m",
                "pba.run",
                "fusion-eval",
                "--",
                "--config",
                "fusion_zero_shot/src/agd20k_eval/config.yaml",
            ],
        ),
    ]


def test_reproduce_cli_dry_run_prints_planned_commands(tmp_path: Path) -> None:
    job_config = tmp_path / "geometry_probing/umd_linear_probing/configs/dinov2.yaml"
    job_config.parent.mkdir(parents=True)
    job_config.write_text("training: {}\n", encoding="utf-8")
    cfg_path = tmp_path / "main.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "assets": {},
                "outputs": {"root": "outputs/reproduce_main"},
                "jobs": {
                    "geometry": [
                        {
                            "name": "dinov2",
                            "command": "geometry-train",
                            "args": [
                                "--config",
                                "geometry_probing/umd_linear_probing/configs/dinov2.yaml",
                            ],
                        }
                    ]
                },
            }
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pba.reproduce",
            "main",
            "--config",
            str(cfg_path),
            "--repo-root",
            str(tmp_path),
            "--dry-run",
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    assert (
        "geometry/dinov2: python -m pba.run geometry-train -- --config "
        "geometry_probing/umd_linear_probing/configs/dinov2.yaml"
    ) in result.stdout


def test_reproduce_cli_reports_missing_job_config(tmp_path: Path) -> None:
    cfg_path = tmp_path / "main.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "assets": {},
                "outputs": {"root": "outputs/reproduce_main"},
                "jobs": {
                    "geometry": [
                        {
                            "name": "dinov2",
                            "command": "geometry-train",
                            "args": ["--config", "missing/dinov2.yaml"],
                        }
                    ]
                },
            }
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pba.reproduce",
            "main",
            "--config",
            str(cfg_path),
            "--repo-root",
            str(tmp_path),
            "--dry-run",
        ],
        text=True,
        capture_output=True,
    )

    assert result.returncode == 2
    assert "missing job config geometry/dinov2:" in result.stderr
