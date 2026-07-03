from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


REQUIRED_TOP_LEVEL_KEYS = ("assets", "outputs", "jobs")


@dataclass(frozen=True)
class ReproduceConfig:
    path: Path
    repo_root: Path
    assets: dict[str, Path]
    outputs_root: Path
    jobs: dict[str, list[dict[str, Any]]]


def _resolve_repo_path(repo_root: Path, value: Any) -> Path:
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return path
    return (repo_root / path).resolve()


def _expect_mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    return value


def load_reproduce_config(path: Path, repo_root: Path) -> ReproduceConfig:
    config_path = path.expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    raw = _expect_mapping(raw, "config")

    missing = [key for key in REQUIRED_TOP_LEVEL_KEYS if key not in raw]
    if missing:
        raise ValueError(f"Missing required config keys: {', '.join(missing)}")

    assets_raw = _expect_mapping(raw["assets"], "assets")
    outputs_raw = _expect_mapping(raw["outputs"], "outputs")
    jobs_raw = _expect_mapping(raw["jobs"], "jobs")

    if "root" not in outputs_raw:
        raise ValueError("outputs.root is required")

    assets = {key: _resolve_repo_path(repo_root, value) for key, value in assets_raw.items()}
    outputs_root = _resolve_repo_path(repo_root, outputs_raw["root"])

    jobs: dict[str, list[dict[str, Any]]] = {}
    for group, entries in jobs_raw.items():
        if not isinstance(entries, list):
            raise ValueError(f"jobs.{group} must be a list")
        jobs[group] = []
        for index, entry in enumerate(entries):
            if not isinstance(entry, dict):
                raise ValueError(f"jobs.{group}[{index}] must be a mapping")
            jobs[group].append(dict(entry))

    return ReproduceConfig(
        path=config_path,
        repo_root=repo_root.resolve(),
        assets=assets,
        outputs_root=outputs_root,
        jobs=jobs,
    )
