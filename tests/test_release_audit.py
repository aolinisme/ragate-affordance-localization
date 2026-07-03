from pathlib import Path
import subprocess
import sys

import yaml

from pba.audit import AuditFinding, audit_release_readiness
from pba.config import ReproduceConfig
from pba.release_gaps import get_release_gaps


def test_audit_release_readiness_reports_empty_interaction_jobs(tmp_path: Path) -> None:
    config = ReproduceConfig(
        path=tmp_path / "main.yaml",
        repo_root=tmp_path,
        assets={},
        outputs_root=tmp_path / "outputs",
        jobs={"geometry": [], "interaction": [], "fusion": []},
    )

    report = audit_release_readiness(config, check_assets=False)

    assert AuditFinding(
        level="WARN",
        code="interaction-jobs-empty",
        message="No interaction reproduction jobs are configured in the current GitHub source.",
    ) in report.findings


def test_audit_release_readiness_reports_fusion_cache_only_config(tmp_path: Path) -> None:
    fusion_config = tmp_path / "fusion.yaml"
    fusion_config.write_text(
        yaml.safe_dump(
            {
                "geom_pipeline": {
                    "enable": True,
                    "dino_cache_only": True,
                    "cache_root": "outputs/fusion_cache",
                }
            }
        ),
        encoding="utf-8",
    )
    config = ReproduceConfig(
        path=tmp_path / "main.yaml",
        repo_root=tmp_path,
        assets={},
        outputs_root=tmp_path / "outputs",
        jobs={
            "interaction": [{"name": "placeholder", "command": "interaction-probe", "args": []}],
            "fusion": [
                {
                    "name": "agd20k_dinov3_flux",
                    "command": "fusion-eval",
                    "args": ["--config", "fusion.yaml"],
                }
            ],
        },
    )

    report = audit_release_readiness(config, check_assets=False)

    assert AuditFinding(
        level="WARN",
        code="fusion-cache-only",
        message=(
            "Fusion job agd20k_dinov3_flux requires existing DINO cache at "
            f"{tmp_path / 'outputs/fusion_cache'} because dino_cache_only is true. "
            "See docs/fusion_cache_contract.md."
        ),
    ) in report.findings


def test_audit_cli_prints_warn_findings(tmp_path: Path) -> None:
    fusion_config = tmp_path / "fusion.yaml"
    fusion_config.write_text(
        yaml.safe_dump(
            {
                "geom_pipeline": {
                    "enable": True,
                    "dino_cache_only": True,
                    "cache_root": "outputs/fusion_cache",
                }
            }
        ),
        encoding="utf-8",
    )
    reproduce_config = tmp_path / "main.yaml"
    reproduce_config.write_text(
        yaml.safe_dump(
            {
                "assets": {},
                "outputs": {"root": "outputs/reproduce_main"},
                "jobs": {
                    "interaction": [],
                    "fusion": [
                        {
                            "name": "agd20k_dinov3_flux",
                            "command": "fusion-eval",
                            "args": ["--config", "fusion.yaml"],
                        }
                    ],
                },
            }
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pba.audit",
            "--config",
            str(reproduce_config),
            "--repo-root",
            str(tmp_path),
            "--skip-asset-check",
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    assert "WARN interaction-jobs-empty" in result.stdout
    assert "WARN fusion-cache-only" in result.stdout


def test_release_gap_registry_lists_known_extension_points() -> None:
    codes = {gap.code for gap in get_release_gaps()}

    assert codes == {
        "interaction-batch-eval",
        "fusion-cache-build",
        "dinov3-external-source",
        "sd21-local-model",
        "umd-pytorch-dataset-migration",
        "geometry-heavy-runtime-migration",
        "fusion-heavy-runtime-migration",
        "interaction-heavy-runtime-migration",
    }


def test_audit_cli_show_gaps_prints_gap_registry(tmp_path: Path) -> None:
    reproduce_config = tmp_path / "main.yaml"
    reproduce_config.write_text(
        yaml.safe_dump(
            {
                "assets": {},
                "outputs": {"root": "outputs/reproduce_main"},
                "jobs": {"interaction": []},
            }
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pba.audit",
            "--config",
            str(reproduce_config),
            "--repo-root",
            str(tmp_path),
            "--skip-asset-check",
            "--show-gaps",
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    assert "GAP interaction-batch-eval: AGD20K batch interaction-only evaluation" in result.stdout
    assert "GAP fusion-cache-build: DINO/DINOv3 fusion cache generation" in result.stdout
    assert "GAP umd-pytorch-dataset-migration: UMD PyTorch dataset migration" in result.stdout
    assert "GAP geometry-heavy-runtime-migration: Geometry heavy runtime migration" in result.stdout
    assert "GAP fusion-heavy-runtime-migration: Fusion heavy runtime migration" in result.stdout
    assert "GAP interaction-heavy-runtime-migration: Interaction heavy runtime migration" in result.stdout


def test_audit_cli_public_release_reports_private_paths(tmp_path: Path) -> None:
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    private_config = tmp_path / "private.yaml"
    private_config.write_text("dataset_root: /Users/qing/private/data\n", encoding="utf-8")
    reproduce_config = tmp_path / "main.yaml"
    reproduce_config.write_text(
        yaml.safe_dump(
            {
                "assets": {},
                "outputs": {"root": "outputs/reproduce_main"},
                "jobs": {"interaction": [{"name": "placeholder", "command": "interaction-probe", "args": []}]},
            }
        ),
        encoding="utf-8",
    )
    subprocess.run(["git", "add", "private.yaml", "main.yaml"], cwd=tmp_path, check=True, capture_output=True)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pba.audit",
            "--config",
            str(reproduce_config),
            "--repo-root",
            str(tmp_path),
            "--skip-asset-check",
            "--public-release",
        ],
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 2
    assert "BLOCKED private-path: private.yaml: private local path pattern '/Users/' found" in result.stdout
