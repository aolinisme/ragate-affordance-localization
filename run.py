#!/usr/bin/env python3
"""Unified launcher for Probing & Bridging affordance experiments.

Examples:
  python run.py geometry-train -- --config geometry_probing/umd_linear_probing/configs/dinov2.yaml
  python run.py geometry-eval -- --config geometry_probing/umd_linear_probing/configs/dinov2.yaml
  python run.py interaction-probe -- --model-id /path/to/FLUX.1-Kontext-dev --image /path/to/img.png --prompt "hold toothbrush" --affordance hold
  python run.py fusion-eval -- --config fusion_zero_shot/src/agd20k_eval/config.yaml
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

COMMANDS = {
    "geometry-train": ROOT / "geometry_probing" / "umd_linear_probing" / "scripts" / "train.py",
    "geometry-eval": ROOT / "geometry_probing" / "umd_linear_probing" / "scripts" / "eval.py",
    "geometry-multiseed": ROOT / "geometry_probing" / "umd_linear_probing" / "scripts" / "multiseed.py",
    "geometry-head-stats": ROOT / "pba" / "geometry" / "head_stats.py",
    "geometry-export-teacher-logits": ROOT / "pba" / "geometry" / "export_teacher_logits.py",
    "lightweight-student": ROOT / "pba" / "geometry" / "lightweight_student.py",
    "interaction-probe": ROOT / "interaction_probing" / "cross_attention_probe" / "cross_attention_probe.py",
    "fusion-eval": ROOT / "fusion_zero_shot" / "run_agd20k_eval.py",
    "sd-turbo-attention": ROOT / "fusion_zero_shot" / "scripts" / "run_sd_turbo_attention_pilot.py",
    "sd-dino-fusion": ROOT / "fusion_zero_shot" / "scripts" / "run_sd_dino_fusion_from_attention.py",
    "tiny-router": ROOT / "fusion_zero_shot" / "scripts" / "run_tiny_reliability_router.py",
    # Backward-compatible aliases for the main paper paths
    "exp1-train": ROOT / "geometry_probing" / "umd_linear_probing" / "scripts" / "train.py",
    "exp1-eval": ROOT / "geometry_probing" / "umd_linear_probing" / "scripts" / "eval.py",
    "exp3-probe": ROOT / "interaction_probing" / "cross_attention_probe" / "cross_attention_probe.py",
    "exp4-eval": ROOT / "fusion_zero_shot" / "run_agd20k_eval.py",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=sorted(COMMANDS.keys()))
    parser.add_argument(
        "args",
        nargs=argparse.REMAINDER,
        help="Arguments passed to the target script. Prefix with '--' to avoid argparse capture.",
    )
    return parser.parse_args()


def main() -> None:
    parsed = parse_args()
    script = COMMANDS[parsed.command]
    if not script.exists():
        raise FileNotFoundError(f"Script not found: {script}")

    forward = parsed.args
    if forward and forward[0] == "--":
        forward = forward[1:]

    cmd = [sys.executable, str(script), *forward]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
