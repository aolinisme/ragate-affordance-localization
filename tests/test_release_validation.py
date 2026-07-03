from pathlib import Path

from pba.config import ReproduceConfig
from pba.validation import AssetProblem, validate_assets
from pba.validation import JobConfigProblem, validate_job_configs


def test_validate_assets_reports_missing_paths(tmp_path: Path) -> None:
    problems = validate_assets({"umd_root": tmp_path / "missing"})

    assert problems == [
        AssetProblem(name="umd_root", path=tmp_path / "missing", message="missing path")
    ]


def test_validate_assets_accepts_existing_file_and_directory(tmp_path: Path) -> None:
    data_dir = tmp_path / "dataset"
    data_dir.mkdir()
    ckpt = tmp_path / "model.pth"
    ckpt.write_bytes(b"checkpoint")

    problems = validate_assets({"dataset": data_dir, "checkpoint": ckpt})

    assert problems == []


def test_validate_job_configs_reports_missing_config_path(tmp_path: Path) -> None:
    config = ReproduceConfig(
        path=tmp_path / "main.yaml",
        repo_root=tmp_path,
        assets={},
        outputs_root=tmp_path / "outputs",
        jobs={
            "geometry": [
                {
                    "name": "dinov2",
                    "command": "geometry-train",
                    "args": ["--config", "missing/dinov2.yaml"],
                }
            ]
        },
    )

    problems = validate_job_configs(config)

    assert problems == [
        JobConfigProblem(
            group="geometry",
            name="dinov2",
            path=tmp_path / "missing/dinov2.yaml",
            message="missing job config",
        )
    ]


def test_validate_job_configs_accepts_existing_config_path(tmp_path: Path) -> None:
    config_path = tmp_path / "geometry.yaml"
    config_path.write_text("training: {}\n", encoding="utf-8")
    config = ReproduceConfig(
        path=tmp_path / "main.yaml",
        repo_root=tmp_path,
        assets={},
        outputs_root=tmp_path / "outputs",
        jobs={
            "geometry": [
                {
                    "name": "dinov2",
                    "command": "geometry-train",
                    "args": ["--config", "geometry.yaml"],
                }
            ]
        },
    )

    problems = validate_job_configs(config)

    assert problems == []
