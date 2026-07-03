from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pba.config import ReproduceConfig


@dataclass(frozen=True)
class AssetProblem:
    name: str
    path: Path
    message: str


@dataclass(frozen=True)
class JobConfigProblem:
    group: str
    name: str
    path: Path
    message: str


def validate_assets(assets: dict[str, Path]) -> list[AssetProblem]:
    problems: list[AssetProblem] = []
    for name, path in assets.items():
        if not path.exists():
            problems.append(AssetProblem(name=name, path=path, message="missing path"))
    return problems


def _resolve_job_config_path(repo_root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return (repo_root / path).resolve()


def validate_job_configs(config: ReproduceConfig) -> list[JobConfigProblem]:
    problems: list[JobConfigProblem] = []
    for group, jobs in config.jobs.items():
        for index, job in enumerate(jobs):
            args = job.get("args", [])
            if not isinstance(args, list):
                continue
            name = str(job.get("name", job.get("command", f"job-{index}")))
            for arg_index, arg in enumerate(args[:-1]):
                if str(arg) != "--config":
                    continue
                config_path = _resolve_job_config_path(config.repo_root, str(args[arg_index + 1]))
                if not config_path.exists():
                    problems.append(
                        JobConfigProblem(
                            group=group,
                            name=name,
                            path=config_path,
                            message="missing job config",
                        )
                    )
    return problems
