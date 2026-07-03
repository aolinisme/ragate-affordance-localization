from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType


@dataclass(frozen=True)
class CommandRegistry:
    repo_root: Path
    commands: dict[str, Path]


def _load_run_module(run_py: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("_pba_run_registry", run_py)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load command registry from {run_py}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_command_registry(repo_root: Path) -> CommandRegistry:
    root = repo_root.expanduser().resolve()
    run_py = root / "run.py"
    if not run_py.exists():
        raise FileNotFoundError(f"run.py not found at {run_py}")

    module = _load_run_module(run_py)
    raw_commands = getattr(module, "COMMANDS", None)
    if not isinstance(raw_commands, dict):
        raise ValueError(f"{run_py} must define COMMANDS as a mapping")

    commands: dict[str, Path] = {}
    for name, script_path in raw_commands.items():
        commands[str(name)] = Path(script_path).expanduser().resolve()

    return CommandRegistry(repo_root=root, commands=commands)
