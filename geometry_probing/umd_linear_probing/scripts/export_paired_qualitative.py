#!/usr/bin/env python
"""Export paired no-gate vs depth-gate qualitative comparisons."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
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
from src.visualization.plots import DEFAULT_PALETTE, DINOV3_MEAN, DINOV3_STD  # noqa: E402


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
    parser.add_argument("--baseline-config", type=Path, required=True)
    parser.add_argument("--baseline-checkpoint", type=Path, required=True)
    parser.add_argument("--gate-config", type=Path, required=True)
    parser.add_argument("--gate-checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-cases", type=int, default=4)
    return parser.parse_args()


def _to_numpy_image(tensor: torch.Tensor) -> np.ndarray:
    array = tensor.detach().cpu().numpy().transpose(1, 2, 0)
    array = (array * DINOV3_STD) + DINOV3_MEAN
    return np.clip(array, 0.0, 1.0)


def _mask_to_color(mask: torch.Tensor) -> np.ndarray:
    mask_np = mask.detach().cpu().numpy().astype(np.int64)
    mask_np = np.clip(mask_np, 0, DEFAULT_PALETTE.shape[0] - 1)
    return DEFAULT_PALETTE[mask_np]


def _overlay(image: np.ndarray, mask: torch.Tensor, alpha: float = 0.5) -> np.ndarray:
    return (1.0 - alpha) * image + alpha * _mask_to_color(mask)


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
    scores: list[float] = []
    for cls_idx in range(num_classes):
        if cls_idx in ignore_classes:
            continue
        pred_cls = (prediction == cls_idx) & valid
        target_cls = (target == cls_idx) & valid
        union = pred_cls | target_cls
        if not bool(union.any()):
            continue
        intersection = pred_cls & target_cls
        scores.append(float(intersection.sum().item() / max(union.sum().item(), 1)))
    if not scores:
        return 0.0
    return float(sum(scores) / len(scores))


def _json_safe_meta(meta: dict[str, Any]) -> dict[str, Any]:
    safe: dict[str, Any] = {}
    for key, value in meta.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            safe[key] = value
        else:
            safe[key] = str(value)
    return safe


def _build_experiment(config_path: Path, checkpoint_path: Path) -> tuple[LinearProbeExperiment, torch.nn.Module]:
    config = load_config(config_path, None)
    experiment = LinearProbeExperiment(config)
    head = experiment._build_head().to(experiment.device)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    head.load_state_dict(checkpoint.get("state_dict", checkpoint))
    head.eval()
    experiment.backbone.eval()
    return experiment, head


def _predict_batch(
    experiment: LinearProbeExperiment,
    head: torch.nn.Module,
    batch: dict[str, Any],
) -> torch.Tensor:
    images = batch["image"].to(experiment.device, non_blocking=True)
    features = experiment.backbone(
        images,
        autocast_precision=experiment.training_cfg.get("precision", "bf16"),
    )
    _attach_geometry(features, batch, experiment.device)
    if experiment.use_multi_head:
        _ensure_geometry_placeholders(features, head, experiment.device)
        logits = head(features)
    else:
        logits = head(features[experiment.target_layer])
    upsampled = F.interpolate(
        logits,
        size=batch["pixel_mask"].shape[-2:],
        mode="bilinear",
        align_corners=False,
    )
    return upsampled.argmax(dim=1).cpu()


def _save_case_gallery(cases: list[dict[str, Any]], output_dir: Path, title: str) -> None:
    if not cases:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(len(cases), 4, figsize=(13, 3.2 * len(cases)), squeeze=False)
    for row, case in enumerate(cases):
        image = _to_numpy_image(case["image"])
        views = [
            (image, f"RGB | {case['meta']['tool']}"),
            (_overlay(image, case["baseline_prediction"]), f"No gate {case['baseline_miou']:.3f}"),
            (_overlay(image, case["gate_prediction"]), f"Depth gate {case['gate_miou']:.3f}"),
            (_overlay(image, case["target"]), "Ground truth"),
        ]
        for col, (view, label) in enumerate(views):
            axes[row, col].imshow(view)
            axes[row, col].set_title(label)
            axes[row, col].axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_dir / "paired_examples.png", dpi=150)
    plt.close(fig)

    legend_fig = plt.figure(figsize=(4, 2))
    legend_ax = legend_fig.add_subplot(111)
    legend_ax.axis("off")
    handles = []
    for idx, name in enumerate(DEFAULT_CLASS_NAMES):
        handles.append(
            plt.Line2D(
                [0],
                [0],
                marker="s",
                color=DEFAULT_PALETTE[idx],
                label=name,
                linestyle="",
                markersize=10,
            )
        )
    legend_ax.legend(handles=handles, title="Affordance classes", loc="center", ncol=2)
    legend_fig.savefig(output_dir / "palette_legend.png", dpi=150)
    plt.close(legend_fig)


def main() -> None:
    args = parse_args()
    baseline_exp, baseline_head = _build_experiment(args.baseline_config, args.baseline_checkpoint)
    gate_exp, gate_head = _build_experiment(args.gate_config, args.gate_checkpoint)

    baseline_loader = baseline_exp.val_loader if args.split == "val" else baseline_exp.test_loader
    gate_loader = gate_exp.val_loader if args.split == "val" else gate_exp.test_loader

    cases: list[dict[str, Any]] = []
    with torch.no_grad():
        for baseline_batch, gate_batch in zip(baseline_loader, gate_loader):
            baseline_preds = _predict_batch(baseline_exp, baseline_head, baseline_batch)
            gate_preds = _predict_batch(gate_exp, gate_head, gate_batch)
            targets = baseline_batch["pixel_mask"].cpu()
            for idx in range(targets.shape[0]):
                baseline_meta = baseline_batch["meta"][idx]
                gate_meta = gate_batch["meta"][idx]
                if baseline_meta["frame_id"] != gate_meta["frame_id"]:
                    raise ValueError(
                        f"Split mismatch: {baseline_meta['frame_id']} != {gate_meta['frame_id']}"
                    )
                baseline_score = _sample_miou(
                    baseline_preds[idx],
                    targets[idx],
                    num_classes=baseline_exp.num_classes,
                    ignore_index=baseline_exp.ignore_index,
                    ignore_classes=baseline_exp.metric_ignore_indices,
                )
                gate_score = _sample_miou(
                    gate_preds[idx],
                    targets[idx],
                    num_classes=gate_exp.num_classes,
                    ignore_index=gate_exp.ignore_index,
                    ignore_classes=gate_exp.metric_ignore_indices,
                )
                cases.append(
                    {
                        "meta": baseline_meta,
                        "image": baseline_batch["image"][idx].cpu(),
                        "target": targets[idx],
                        "baseline_prediction": baseline_preds[idx],
                        "gate_prediction": gate_preds[idx],
                        "baseline_miou": baseline_score,
                        "gate_miou": gate_score,
                        "delta_miou": gate_score - baseline_score,
                    }
                )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    depth_wins = sorted(cases, key=lambda item: item["delta_miou"], reverse=True)[: args.num_cases]
    baseline_wins = sorted(cases, key=lambda item: item["delta_miou"])[: args.num_cases]
    both_fail = sorted(cases, key=lambda item: max(item["baseline_miou"], item["gate_miou"]))[: args.num_cases]

    _save_case_gallery(depth_wins, args.output_dir / "depth_gate_wins", "Depth Gate Wins")
    _save_case_gallery(baseline_wins, args.output_dir / "baseline_wins", "No Gate Wins")
    _save_case_gallery(both_fail, args.output_dir / "both_fail", "Both Fail / Low IoU")

    def summarize(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {
                "meta": _json_safe_meta(item["meta"]),
                "baseline_miou": item["baseline_miou"],
                "gate_miou": item["gate_miou"],
                "delta_miou": item["delta_miou"],
            }
            for item in items
        ]

    payload = {
        "baseline_config": str(args.baseline_config.resolve()),
        "baseline_checkpoint": str(args.baseline_checkpoint.resolve()),
        "gate_config": str(args.gate_config.resolve()),
        "gate_checkpoint": str(args.gate_checkpoint.resolve()),
        "split": args.split,
        "num_cases_scored": len(cases),
        "depth_gate_wins": summarize(depth_wins),
        "baseline_wins": summarize(baseline_wins),
        "both_fail": summarize(both_fail),
    }
    with (args.output_dir / "paired_case_scores.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
