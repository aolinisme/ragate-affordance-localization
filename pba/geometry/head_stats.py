from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from pba.geometry.config import load_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Report lightweight geometry-head parameter statistics.")
    parser.add_argument("--config", type=Path, required=True, help="Experiment config YAML.")
    parser.add_argument(
        "--local",
        type=Path,
        default=None,
        help="[Optional] Local override config merged into --config.",
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def _resolve_feature_channels(config: dict[str, Any]) -> dict[Any, int]:
    params = config["model"]["params"]
    feature_keys = params["head"]["feature_keys"]
    model_id = str(params.get("model_id", ""))
    # Current local experiments use OpenCLIP ViT-B/16 with 4 fused layers of 769 channels each.
    if "CLIP-ViT-B-16" not in model_id:
        raise ValueError(f"Unsupported model_id for head stats: {model_id}")
    return {key: 769 for key in feature_keys}


def _count_conv_params(in_channels: int, out_channels: int, bias: bool = True) -> int:
    total = in_channels * out_channels
    if bias:
        total += out_channels
    return total


def compute_head_stats(config: dict[str, Any]) -> dict[str, Any]:
    dataset_cfg = config["dataset"]
    num_classes = int(dataset_cfg.get("num_classes", 8))
    head_cfg = config["model"]["params"]["head"]
    feature_keys = list(head_cfg["feature_keys"])
    channels = _resolve_feature_channels(config)
    fused_channels = sum(channels[key] for key in feature_keys)
    use_batchnorm = bool(head_cfg.get("use_batchnorm", True))
    use_geometry_gate = bool(head_cfg.get("use_geometry_gate", False))
    geometry_key = str(head_cfg.get("geometry_key", "geom_depth"))
    geometry_channels = 1 if geometry_key == "geom_depth" else 3

    proj_params = _count_conv_params(fused_channels, num_classes, bias=True)
    bn_params = fused_channels * 2 if use_batchnorm else 0
    gate_params = (
        _count_conv_params(geometry_channels, fused_channels, bias=True)
        if use_geometry_gate
        else 0
    )

    total = proj_params + bn_params + gate_params
    baseline_total = proj_params + bn_params
    return {
        "model_id": config["model"]["params"]["model_id"],
        "feature_keys": feature_keys,
        "fused_channels": fused_channels,
        "num_classes": num_classes,
        "use_batchnorm": use_batchnorm,
        "use_geometry_gate": use_geometry_gate,
        "geometry_key": geometry_key if use_geometry_gate else None,
        "parameter_breakdown": {
            "projection": proj_params,
            "batchnorm": bn_params,
            "geometry_gate": gate_params,
        },
        "total_parameters": total,
        "baseline_head_parameters": baseline_total,
        "extra_gate_parameters": gate_params,
        "gate_parameter_ratio_vs_head": (gate_params / baseline_total) if baseline_total else 0.0,
    }


def run_head_stats(argv: list[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    config = load_config(args.config, args.local)
    stats = compute_head_stats(config)
    print(json.dumps(stats, indent=2))
    return stats


if __name__ == "__main__":
    run_head_stats()
