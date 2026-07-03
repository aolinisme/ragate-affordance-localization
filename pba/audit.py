from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from pba.config import ReproduceConfig
from pba.config import load_reproduce_config
from pba.public_release import check_public_release
from pba.release_gaps import get_release_gaps
from pba.validation import validate_assets, validate_job_configs


@dataclass(frozen=True)
class AuditFinding:
    level: str
    code: str
    message: str


@dataclass(frozen=True)
class AuditReport:
    findings: list[AuditFinding]


def _job_config_paths(config: ReproduceConfig, group: str) -> list[tuple[str, Path]]:
    paths: list[tuple[str, Path]] = []
    for index, job in enumerate(config.jobs.get(group, [])):
        args = job.get("args", [])
        if not isinstance(args, list):
            continue
        name = str(job.get("name", job.get("command", f"job-{index}")))
        for arg_index, arg in enumerate(args[:-1]):
            if str(arg) != "--config":
                continue
            path = Path(str(args[arg_index + 1])).expanduser()
            if not path.is_absolute():
                path = (config.repo_root / path).resolve()
            paths.append((name, path))
    return paths


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle) or {}
    if not isinstance(value, dict):
        return {}
    return value


def audit_release_readiness(
    config: ReproduceConfig,
    *,
    check_assets: bool = True,
    check_public: bool = False,
) -> AuditReport:
    findings: list[AuditFinding] = []

    if check_assets:
        for problem in validate_assets(config.assets):
            findings.append(
                AuditFinding(
                    level="BLOCKED",
                    code="missing-asset",
                    message=f"{problem.name}: {problem.path}",
                )
            )

    for problem in validate_job_configs(config):
        findings.append(
            AuditFinding(
                level="BLOCKED",
                code="missing-job-config",
                message=f"{problem.group}/{problem.name}: {problem.path}",
            )
        )

    if check_public:
        for finding in check_public_release(config.repo_root):
            findings.append(
                AuditFinding(
                    level=finding.level,
                    code=finding.code,
                    message=f"{finding.path}: {finding.message}",
                )
            )

    if not config.jobs.get("interaction"):
        findings.append(
            AuditFinding(
                level="WARN",
                code="interaction-jobs-empty",
                message="No interaction reproduction jobs are configured in the current GitHub source.",
            )
        )

    for name, path in _job_config_paths(config, "fusion"):
        if not path.exists():
            continue
        fusion_config = _read_yaml(path)
        geom_pipeline = fusion_config.get("geom_pipeline", {})
        if not isinstance(geom_pipeline, dict):
            continue
        if geom_pipeline.get("enable") is True and geom_pipeline.get("dino_cache_only") is True:
            cache_root = Path(str(geom_pipeline.get("cache_root", "outputs/fusion_cache"))).expanduser()
            if not cache_root.is_absolute():
                cache_root = (path.parent / cache_root).resolve()
            findings.append(
                AuditFinding(
                    level="WARN",
                    code="fusion-cache-only",
                    message=(
                        f"Fusion job {name} requires existing DINO cache at {cache_root} "
                        "because dino_cache_only is true. See docs/fusion_cache_contract.md."
                    ),
                )
            )

    return AuditReport(findings=findings)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit paper release readiness.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--skip-asset-check", action="store_true")
    parser.add_argument("--show-gaps", action="store_true")
    parser.add_argument("--public-release", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_reproduce_config(args.config, repo_root=args.repo_root)
    report = audit_release_readiness(
        config,
        check_assets=not args.skip_asset_check,
        check_public=args.public_release,
    )

    if not report.findings and not args.show_gaps:
        print("OK release-readiness: no findings")
        return 0

    blocked = False
    for finding in report.findings:
        print(f"{finding.level} {finding.code}: {finding.message}")
        if finding.level == "BLOCKED":
            blocked = True

    if args.show_gaps:
        for gap in get_release_gaps():
            print(f"GAP {gap.code}: {gap.title}")
            print(f"  current: {gap.current_status}")
            print(f"  expected: {gap.expected_extension}")
            print(f"  integration: {gap.integration_point}")
    return 2 if blocked else 0


if __name__ == "__main__":
    raise SystemExit(main())
