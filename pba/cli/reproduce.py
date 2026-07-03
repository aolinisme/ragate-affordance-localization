from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from pba.config import load_reproduce_config
from pba.planner import build_reproduction_plan
from pba.validation import validate_assets, validate_job_configs


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run paper reproduction jobs.")
    parser.add_argument("target", choices=["main"])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-asset-check", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_reproduce_config(args.config, repo_root=args.repo_root)
    plan = build_reproduction_plan(config)

    if not args.skip_asset_check:
        problems = validate_assets(config.assets)
        if problems:
            for problem in problems:
                print(f"missing asset {problem.name}: {problem.path}", file=sys.stderr)
            return 2

    job_config_problems = validate_job_configs(config)
    if job_config_problems:
        for problem in job_config_problems:
            print(
                f"missing job config {problem.group}/{problem.name}: {problem.path}",
                file=sys.stderr,
            )
        return 2

    for command in plan:
        printable = " ".join(command.argv)
        print(f"{command.group}/{command.name}: {printable}")
        if not args.dry_run:
            subprocess.run(command.argv, cwd=config.repo_root, check=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
