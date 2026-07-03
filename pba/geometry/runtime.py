"""Lightweight geometry runtime migration contract."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

__all__ = [
    "GEOMETRY_BACKBONE_TARGETS",
    "GEOMETRY_RUNTIME_MODULES",
    "GEOMETRY_TRAINING_OUTPUT_FILES",
    "GeometryRuntimeBoundary",
    "build_runtime_boundary",
    "validate_geometry_metrics",
]


GEOMETRY_RUNTIME_MODULES = (
    "src.engine.trainer.LinearProbeExperiment",
    "src.engine.eval.evaluate_linear_probe",
    "src.data.dataset.UMDAffordanceDataset",
    "src.data.collate.collate_with_meta",
    "src.data.transforms.get_default_image_transform",
    "src.data.transforms.get_sam_image_transform",
    "src.models.DINOBackbone",
    "src.models.DINOv2Backbone",
    "src.models.DINOv3Backbone",
    "src.models.FluxBackbone",
    "src.models.SigLIP2Backbone",
    "src.models.SAMBackbone",
    "src.models.StableDiffusionBackbone",
    "src.models.head.LinearProbeHead",
    "src.models.linear_head.MultiLayerLinearHead",
    "src.utils.logging.create_logger",
    "src.utils.random.set_seed",
    "src.visualization.plots",
)

GEOMETRY_BACKBONE_TARGETS = (
    "dino",
    "dinov2",
    "dinov3",
    "open_clip",
    "flux",
    "stable_diffusion",
    "siglip2",
    "sam",
)

GEOMETRY_TRAINING_OUTPUT_FILES = (
    "master.log",
    "summary.json",
    "training_history.json",
    "val_examples.pt",
    "test_examples.pt",
    "{sweep}/train.log",
    "{sweep}/config_snapshot.yaml",
    "{sweep}/linear_probe.pth",
)

_GEOMETRY_METRIC_KEYS = ("loss", "miou", "per_class_iou")


@dataclass(frozen=True)
class GeometryRuntimeBoundary:
    """Public migration boundary for the legacy UMD linear probing runtime."""

    package_module: str
    legacy_root: str
    train_entrypoint: str
    eval_entrypoint: str
    modules: tuple[str, ...]
    backbone_targets: tuple[str, ...]


def build_runtime_boundary() -> GeometryRuntimeBoundary:
    """Return the accepted geometry heavy-runtime migration boundary."""

    return GeometryRuntimeBoundary(
        package_module="pba.geometry",
        legacy_root="geometry_probing/umd_linear_probing/src",
        train_entrypoint="pba.geometry.linear_probe.run_train",
        eval_entrypoint="pba.geometry.linear_probe.run_eval",
        modules=GEOMETRY_RUNTIME_MODULES,
        backbone_targets=GEOMETRY_BACKBONE_TARGETS,
    )


def validate_geometry_metrics(metrics: Mapping[str, Any]) -> Mapping[str, Any]:
    """Validate metric shape returned by geometry evaluation."""

    missing = [key for key in _GEOMETRY_METRIC_KEYS if key not in metrics]
    if missing:
        raise ValueError(f"Geometry metrics missing required keys: {', '.join(missing)}")
    per_class = metrics["per_class_iou"]
    if not isinstance(per_class, list):
        raise ValueError("Geometry metric per_class_iou must be a list.")
    return metrics
