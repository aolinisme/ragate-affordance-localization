"""CLI configuration for single-image interaction probing."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

__all__ = ["build_probe_parser", "parse_probe_args"]


def build_probe_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Flux Kontext cross-attention extractor.")
    parser.add_argument("--model-id", type=str, required=True, help="HF repo id or local path to Flux Kontext weights.")
    parser.add_argument("--image", type=Path, required=True, help="Path to the reference RGB image.")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt containing the affordance label.")
    parser.add_argument("--affordance", type=str, required=True, help="Affordance label to extract cross-attention for.")
    parser.add_argument("--output-root", type=Path, default=Path("probe_outputs"), help="Root directory for results.")
    parser.add_argument("--steps", type=int, default=20, help="Number of denoising steps.")
    parser.add_argument("--guidance", type=float, default=3.0, help="Classifier-free guidance scale.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--device", type=str, default="cuda", help="Device for inference.")
    return parser


def parse_probe_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_probe_parser().parse_args(argv)
