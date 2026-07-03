from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from pba.runner import load_command_registry


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run registered Probing & Bridging commands.")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--list", action="store_true", help="List commands registered by top-level run.py.")
    parser.add_argument("command", nargs="?")
    parser.add_argument(
        "args",
        nargs=argparse.REMAINDER,
        help="Arguments passed to the target command. Prefix with '--' to avoid argparse capture.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    parsed = parse_args(argv)
    registry = load_command_registry(parsed.repo_root)

    if parsed.list:
        for command in sorted(registry.commands):
            print(command)
        return 0

    if not parsed.command:
        print("missing command", file=sys.stderr)
        return 2

    script = registry.commands.get(parsed.command)
    if script is None:
        print(f"unknown command: {parsed.command}", file=sys.stderr)
        return 2

    forward = parsed.args
    if forward and forward[0] == "--":
        forward = forward[1:]

    subprocess.run([sys.executable, str(script), *forward], cwd=registry.repo_root, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
