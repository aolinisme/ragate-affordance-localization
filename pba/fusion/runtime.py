"""Lightweight fusion runtime migration contract."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

__all__ = [
    "FUSION_BASE_METRIC_KEYS",
    "FUSION_GLOBAL_METRIC_KEYS",
    "FUSION_RUNTIME_MODULES",
    "FUSION_SAMPLE_ARTIFACTS",
    "FUSION_SUMMARY_COLUMNS",
    "FusionRuntimeBoundary",
    "build_runtime_boundary",
    "validate_fusion_detail",
    "validate_global_metrics",
]


FUSION_RUNTIME_MODULES = (
    "run_flux_kontext_eval.process_sample",
    "run_flux_kontext_eval.run_geom_pipeline",
    "run_flux_kontext_eval.compute_metrics",
    "kontext_runner.run_kontext_generation",
    "heatmap_warper.warp_heatmap_cli",
    "utils.data_iter.iter_agd20k_samples",
    "utils.logging_utils.append_csv",
    "pipeline.roi_stage.build_roi_mask",
    "pipeline.roi_stage.compute_roi_tokens",
    "pipeline.roi_stage.restore_to_original",
    "pipeline.pca_stage.extract_dino_tokens",
    "pipeline.pca_stage.run_pca",
    "pipeline.geometry_stage.generate_geometry_mask",
    "pipeline.geometry_stage.largest_component",
    "pipeline.utils.save_overlay",
    "pipeline.utils.save_colormap",
)

FUSION_BASE_METRIC_KEYS = ("mKLD", "mSIM", "mNSS")

FUSION_SUMMARY_COLUMNS = (
    "affordance",
    "object",
    "image",
    "prompt",
    "token_index",
    "token_str",
    "warped_heatmap",
    "gt",
    "mKLD",
    "mSIM",
    "mNSS",
    "object_token_index",
    "object_token_str",
    "object_heatmap",
    "object_warped_heatmap",
    "geom_mask",
    "geom_selected_pc",
    "geom_similarity_best",
    "geom_mKLD",
    "geom_mSIM",
    "geom_mNSS",
    "soft_heatmap",
    "soft_best_temperature",
    "soft_mKLD",
    "soft_mSIM",
    "soft_mNSS",
    "soft_variants_detail",
    "final_mask",
    "final_threshold",
    "final_mKLD",
    "final_mSIM",
    "final_mNSS",
    "adaptive_fusion_enabled",
    "adaptive_lambda",
    "interaction_confidence",
    "geometry_confidence",
    "alignment_confidence",
    "base_lambda",
    "lambda_range",
    "adaptive_fusion_detail",
    "geom_similarity_scores",
    "final_masks_detail",
)

FUSION_GLOBAL_METRIC_KEYS = (
    "total_samples",
    "processed_samples",
    "resume_reused",
    "skipped_samples",
    "failed_samples",
    "overall_metrics",
    "per_affordance_metrics",
    "skipped_detail",
    "failed_detail",
)

FUSION_SAMPLE_ARTIFACTS = (
    "kontext/",
    "mapped/",
    "metrics.json",
    "geom_pipeline/stage_roi/",
    "geom_pipeline/stage_dino/",
    "geom_pipeline/stage_geom/",
    "geom_pipeline/stage_final/",
)


@dataclass(frozen=True)
class FusionRuntimeBoundary:
    """Public migration boundary for the AGD20K fusion heavy runtime."""

    package_module: str
    legacy_entrypoint: str
    config_path: str
    modules: tuple[str, ...]
    summary_columns: tuple[str, ...]


def build_runtime_boundary() -> FusionRuntimeBoundary:
    """Return the accepted fusion heavy-runtime migration boundary."""

    return FusionRuntimeBoundary(
        package_module="pba.fusion",
        legacy_entrypoint="fusion_zero_shot/src/agd20k_eval/run_flux_kontext_eval.py",
        config_path="fusion_zero_shot/src/agd20k_eval/config.yaml",
        modules=FUSION_RUNTIME_MODULES,
        summary_columns=FUSION_SUMMARY_COLUMNS,
    )


def validate_fusion_detail(detail: Mapping[str, Any]) -> Mapping[str, Any]:
    """Validate per-sample fusion summary row shape."""

    required = ("affordance", "object", "image", *FUSION_BASE_METRIC_KEYS)
    missing = [key for key in required if key not in detail]
    if missing:
        raise ValueError(f"Fusion detail missing required keys: {', '.join(missing)}")
    return detail


def validate_global_metrics(metrics: Mapping[str, Any]) -> Mapping[str, Any]:
    """Validate global metrics JSON shape written by fusion evaluation."""

    missing = [key for key in FUSION_GLOBAL_METRIC_KEYS if key not in metrics]
    if missing:
        raise ValueError(f"Fusion global metrics missing required keys: {', '.join(missing)}")
    return metrics
