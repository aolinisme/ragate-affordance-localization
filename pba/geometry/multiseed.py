"""Sequential multi-seed runner for geometry probing experiments."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Sequence

import yaml

from pba.geometry.config import load_config

REPO_ROOT = Path(__file__).resolve().parents[2]
LEGACY_TRAIN_SCRIPT = (
    REPO_ROOT / "geometry_probing" / "umd_linear_probing" / "scripts" / "train.py"
)

RunCommand = Callable[..., Any]

__all__ = [
    "build_multiseed_parser",
    "parse_multiseed_args",
    "run_multiseed",
]


def build_multiseed_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a geometry probing config sequentially across multiple seeds."
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Base YAML config for the geometry probing experiment.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        required=True,
        help="Seeds to run sequentially.",
    )
    parser.add_argument(
        "--label",
        type=str,
        default=None,
        help="Optional base output_dir_name override used before appending _seed{seed}.",
    )
    parser.add_argument(
        "--python",
        type=Path,
        default=Path(sys.executable),
        help="Python executable used to launch each training run.",
    )
    parser.add_argument(
        "--summary-out",
        type=Path,
        default=None,
        help="Optional path for the aggregated multi-seed JSON summary.",
    )
    parser.add_argument(
        "--hf-offline",
        action="store_true",
        help="Set HF_HUB_OFFLINE=1 and TRANSFORMERS_OFFLINE=1 for child training runs.",
    )
    return parser


def parse_multiseed_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_multiseed_parser().parse_args(argv)


def _default_run_command(command: list[str], *, cwd: Path, env: dict[str, str] | None = None) -> Any:
    return subprocess.run(command, cwd=cwd, env=env, check=True)


def _derive_output_root(config: dict[str, Any]) -> Path:
    training_cfg = config["training"]
    output_root = training_cfg.get("output_root")
    if output_root is not None:
        return Path(output_root)

    output_dir = training_cfg.get("output_dir")
    if output_dir is None:
        raise KeyError("Training config must define output_root or output_dir.")
    return Path(output_dir).parent


def _derive_base_output_name(config: dict[str, Any], label: str | None) -> str:
    if label:
        return label
    model_cfg = config.get("model") or {}
    params = model_cfg.get("params") if isinstance(model_cfg, dict) else {}
    if isinstance(params, dict):
        output_dir_name = params.get("output_dir_name")
        if isinstance(output_dir_name, str) and output_dir_name.strip():
            return output_dir_name.strip()
    target = model_cfg.get("target")
    if isinstance(target, str) and target.strip():
        return target.strip()
    return "geometry_probe"


def _list_matching_run_dirs(output_root: Path, prefix: str) -> set[Path]:
    if not output_root.exists():
        return set()
    return {
        path.resolve()
        for path in output_root.glob(f"{prefix}_*")
        if path.is_dir() and (path / "summary.json").exists()
    }


def _sample_std(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    return float(statistics.stdev(values))


def _write_override(path: Path, *, seed: int, output_dir_name: str) -> None:
    payload = {
        "training": {"seed": seed},
        "model": {"params": {"output_dir_name": output_dir_name}},
    }
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def _resolve_summary_out(
    output_root: Path,
    *,
    base_output_name: str,
    seeds: list[int],
    summary_out: Path | None,
) -> Path:
    if summary_out is not None:
        return summary_out.resolve()
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    seed_token = "-".join(str(seed) for seed in seeds)
    summary_dir = output_root / "multiseed_summaries"
    summary_dir.mkdir(parents=True, exist_ok=True)
    return summary_dir / f"{base_output_name}_seeds_{seed_token}_{timestamp}.json"


def run_multiseed(
    argv: Sequence[str] | None = None,
    *,
    run_command: RunCommand | None = None,
) -> dict[str, Any]:
    args = parse_multiseed_args(argv)
    if not args.config.exists():
        raise FileNotFoundError("Please provide --config pointing to a valid YAML file.")
    if not args.python.exists():
        raise FileNotFoundError(f"Python executable not found: {args.python}")

    if run_command is None:
        run_command = _default_run_command

    config = load_config(args.config, None).data
    output_root = _derive_output_root(config).resolve()
    base_output_name = _derive_base_output_name(config, args.label)
    summary_out = _resolve_summary_out(
        output_root,
        base_output_name=base_output_name,
        seeds=list(args.seeds),
        summary_out=args.summary_out,
    )
    summary_out.parent.mkdir(parents=True, exist_ok=True)
    child_env = os.environ.copy()
    if args.hf_offline:
        child_env["HF_HUB_OFFLINE"] = "1"
        child_env["TRANSFORMERS_OFFLINE"] = "1"

    runs: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="pba_multiseed_") as temp_dir:
        temp_root = Path(temp_dir)
        for seed in args.seeds:
            seed_output_name = f"{base_output_name}_seed{seed}"
            override_path = temp_root / f"seed_{seed}.yaml"
            _write_override(override_path, seed=seed, output_dir_name=seed_output_name)

            before_dirs = _list_matching_run_dirs(output_root, seed_output_name)
            command = [
                str(args.python),
                str(LEGACY_TRAIN_SCRIPT),
                "--config",
                str(args.config.resolve()),
                "--local",
                str(override_path),
            ]

            start = time.perf_counter()
            run_command(command, cwd=REPO_ROOT, env=child_env)
            elapsed_seconds = time.perf_counter() - start

            after_dirs = _list_matching_run_dirs(output_root, seed_output_name)
            new_dirs = sorted(after_dirs - before_dirs, key=lambda path: path.stat().st_mtime)
            if not new_dirs:
                raise RuntimeError(
                    f"Could not identify the output directory for seed {seed} with prefix "
                    f"{seed_output_name!r} under {output_root}"
                )

            output_dir = new_dirs[-1]
            summary_path = output_dir / "summary.json"
            with summary_path.open("r", encoding="utf-8") as handle:
                summary = json.load(handle)

            runs.append(
                {
                    "seed": seed,
                    "command": command,
                    "override": {
                        "training": {"seed": seed},
                        "model": {"params": {"output_dir_name": seed_output_name}},
                    },
                    "output_dir": str(output_dir),
                    "summary_path": str(summary_path),
                    "checkpoint_path": summary.get("checkpoint_path"),
                    "best_val_miou": summary.get("best_val", {}).get("miou"),
                    "test_miou": summary.get("test_metrics", {}).get("miou"),
                    "elapsed_seconds": elapsed_seconds,
                    "run_timestamp": summary.get("run_timestamp"),
                }
            )

    val_scores = [float(run["best_val_miou"]) for run in runs]
    test_scores = [float(run["test_miou"]) for run in runs]
    elapsed_scores = [float(run["elapsed_seconds"]) for run in runs]
    aggregate = {
        "best_val_miou": {
            "mean": float(statistics.fmean(val_scores)),
            "std": _sample_std(val_scores),
        },
        "test_miou": {
            "mean": float(statistics.fmean(test_scores)),
            "std": _sample_std(test_scores),
        },
        "elapsed_seconds": {
            "mean": float(statistics.fmean(elapsed_scores)),
            "std": _sample_std(elapsed_scores),
        },
    }

    result = {
        "config": str(args.config.resolve()),
        "python_executable": str(args.python.resolve()),
        "base_output_name": base_output_name,
        "seeds": list(args.seeds),
        "hf_offline": bool(args.hf_offline),
        "runs": runs,
        "aggregate": aggregate,
        "summary_generated_at": datetime.now().isoformat(timespec="seconds"),
    }

    with summary_out.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)

    print(json.dumps(result, indent=2))
    return result
