#!/usr/bin/env python
"""Export success/failure qualitative cases for a saved geometry probe."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[3]
PROJECT_ROOT = REPO_ROOT / "geometry_probing" / "umd_linear_probing"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pba.geometry.config import load_config  # noqa: E402
from src.engine.trainer import LinearProbeExperiment  # noqa: E402
from src.visualization.plots import save_prediction_gallery  # noqa: E402


DEFAULT_CLASS_NAMES = [
    "background",
    "grasp",
    "cut",
    "scoop",
    "contain",
    "pound",
    "support",
    "wrap-grasp",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="Path to linear_probe.pth.")
    parser.add_argument("--config", type=Path, required=True, help="Experiment config used by the checkpoint.")
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--num-best", type=int, default=4)
    parser.add_argument("--num-worst", type=int, default=4)
    return parser.parse_args()


def _attach_geometry(features: dict[Any, torch.Tensor], batch: dict[str, Any], device: torch.device) -> None:
    for key in ("geom_depth", "geom_normal"):
        value = batch.get(key)
        if isinstance(value, torch.Tensor):
            features[key] = value.to(device, non_blocking=True)


def _ensure_geometry_placeholders(
    features: dict[Any, torch.Tensor],
    head: torch.nn.Module,
    device: torch.device,
) -> None:
    if not hasattr(head, "feature_keys") or not hasattr(head, "primary_key"):
        return
    primary = getattr(head, "primary_key")
    ref = features.get(primary, next(iter(features.values())))
    batch_size = ref.shape[0] if ref.dim() == 4 else 1
    grid = ref.shape[-2:]
    for key, channels in (("geom_depth", 1), ("geom_normal", 3)):
        if key in getattr(head, "feature_keys", []) and key not in features:
            features[key] = torch.zeros((batch_size, channels, *grid), dtype=torch.float32, device=device)
    if hasattr(head, "use_geometry_gate") and getattr(head, "use_geometry_gate"):
        gate_key = getattr(head, "geometry_key", "geom_depth")
        if gate_key not in features:
            channels = 1 if gate_key == "geom_depth" else 3
            features[gate_key] = torch.zeros((batch_size, channels, *grid), dtype=torch.float32, device=device)


def _sample_miou(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    num_classes: int,
    ignore_index: int,
    ignore_classes: set[int],
) -> float:
    valid = target != ignore_index
    values: list[float] = []
    for cls_idx in range(num_classes):
        if cls_idx in ignore_classes:
            continue
        pred_cls = (prediction == cls_idx) & valid
        target_cls = (target == cls_idx) & valid
        union = pred_cls | target_cls
        if not bool(union.any()):
            continue
        intersection = pred_cls & target_cls
        values.append(float(intersection.sum().item() / max(union.sum().item(), 1)))
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _json_safe_meta(meta: dict[str, Any]) -> dict[str, Any]:
    safe: dict[str, Any] = {}
    for key, value in meta.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            safe[key] = value
        else:
            safe[key] = str(value)
    return safe


def main() -> None:
    args = parse_args()
    config = load_config(args.config, None)
    experiment = LinearProbeExperiment(config)

    head = experiment._build_head().to(experiment.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    head.load_state_dict(checkpoint.get("state_dict", checkpoint))
    head.eval()
    experiment.backbone.eval()

    loader = experiment.val_loader if args.split == "val" else experiment.test_loader
    precision = experiment.training_cfg.get("precision", "bf16")
    cases: list[dict[str, Any]] = []

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(experiment.device, non_blocking=True)
            targets = batch["pixel_mask"].cpu()
            features = experiment.backbone(images, autocast_precision=precision)
            _attach_geometry(features, batch, experiment.device)
            if experiment.use_multi_head:
                _ensure_geometry_placeholders(features, head, experiment.device)
                logits = head(features)
            else:
                logits = head(features[experiment.target_layer])
            upsampled = F.interpolate(
                logits,
                size=targets.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            preds = upsampled.argmax(dim=1).cpu()
            for idx in range(images.size(0)):
                score = _sample_miou(
                    preds[idx],
                    targets[idx],
                    num_classes=experiment.num_classes,
                    ignore_index=experiment.ignore_index,
                    ignore_classes=experiment.metric_ignore_indices,
                )
                cases.append(
                    {
                        "score": score,
                        "example": {
                            "image": batch["image"][idx].cpu(),
                            "prediction": preds[idx],
                            "target": targets[idx],
                            "patch_logits": logits[idx].cpu(),
                            "meta": batch["meta"][idx],
                        },
                    }
                )

    cases.sort(key=lambda item: item["score"])
    worst = cases[: args.num_worst]
    best = list(reversed(cases[-args.num_best :]))

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = args.checkpoint.parent / f"qualitative_{args.split}_ranked"
    output_dir.mkdir(parents=True, exist_ok=True)

    save_prediction_gallery(
        [item["example"] for item in best],
        output_dir / "success_cases",
        DEFAULT_CLASS_NAMES,
    )
    save_prediction_gallery(
        [item["example"] for item in worst],
        output_dir / "failure_cases",
        DEFAULT_CLASS_NAMES,
    )

    payload = {
        "checkpoint": str(args.checkpoint.resolve()),
        "config": str(args.config.resolve()),
        "split": args.split,
        "num_cases_scored": len(cases),
        "best": [
            {"score": item["score"], "meta": _json_safe_meta(item["example"]["meta"])}
            for item in best
        ],
        "worst": [
            {"score": item["score"], "meta": _json_safe_meta(item["example"]["meta"])}
            for item in worst
        ],
    }
    with (output_dir / "case_scores.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
