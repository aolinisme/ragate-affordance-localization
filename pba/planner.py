from __future__ import annotations

from dataclasses import dataclass

from pba.config import ReproduceConfig


@dataclass(frozen=True)
class PlannedCommand:
    group: str
    name: str
    argv: list[str]


def build_reproduction_plan(config: ReproduceConfig) -> list[PlannedCommand]:
    commands: list[PlannedCommand] = []
    for group, jobs in config.jobs.items():
        for index, job in enumerate(jobs):
            command = str(job.get("command", "")).strip()
            if not command:
                raise ValueError(f"jobs.{group}[{index}].command is required")
            name = str(job.get("name", command)).strip()
            args = job.get("args", [])
            if not isinstance(args, list):
                raise ValueError(f"jobs.{group}[{index}].args must be a list")
            argv = ["python", "-m", "pba.run", command, "--", *[str(arg) for arg in args]]
            commands.append(PlannedCommand(group=group, name=name, argv=argv))
    return commands
