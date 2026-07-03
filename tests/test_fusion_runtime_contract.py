from pba.fusion.runtime import (
    FUSION_BASE_METRIC_KEYS,
    FUSION_GLOBAL_METRIC_KEYS,
    FUSION_RUNTIME_MODULES,
    FUSION_SAMPLE_ARTIFACTS,
    FUSION_SUMMARY_COLUMNS,
    FusionRuntimeBoundary,
    build_runtime_boundary,
    validate_fusion_detail,
    validate_global_metrics,
)


def test_build_runtime_boundary_documents_legacy_modules() -> None:
    boundary = build_runtime_boundary()

    assert boundary == FusionRuntimeBoundary(
        package_module="pba.fusion",
        legacy_entrypoint="fusion_zero_shot/src/agd20k_eval/run_flux_kontext_eval.py",
        config_path="fusion_zero_shot/src/agd20k_eval/config.yaml",
        modules=FUSION_RUNTIME_MODULES,
        summary_columns=FUSION_SUMMARY_COLUMNS,
    )
    assert "kontext_runner.run_kontext_generation" in boundary.modules
    assert "heatmap_warper.warp_heatmap_cli" in boundary.modules
    assert "pipeline.pca_stage.run_pca" in boundary.modules
    assert "pipeline.geometry_stage.generate_geometry_mask" in boundary.modules


def test_fusion_summary_columns_preserve_base_and_geometry_metrics() -> None:
    assert FUSION_BASE_METRIC_KEYS == ("mKLD", "mSIM", "mNSS")
    for column in (
        "affordance",
        "object",
        "image",
        "warped_heatmap",
        "geom_mask",
        "soft_heatmap",
        "final_mask",
        "final_mKLD",
        "final_mSIM",
        "final_mNSS",
        "adaptive_fusion_enabled",
        "adaptive_lambda",
        "interaction_confidence",
        "geometry_confidence",
        "alignment_confidence",
        "adaptive_fusion_detail",
    ):
        assert column in FUSION_SUMMARY_COLUMNS


def test_validate_fusion_detail_accepts_summary_row_shape() -> None:
    detail = {
        "affordance": "grasp",
        "object": "mug",
        "image": "mug_001.jpg",
        "prompt": "grasp mug",
        "token_index": 2,
        "token_str": "grasp",
        "warped_heatmap": "mapped/heat.png",
        "gt": "gt/mask.png",
        "mKLD": 1.0,
        "mSIM": 0.5,
        "mNSS": 0.25,
    }

    assert validate_fusion_detail(detail) == detail


def test_validate_fusion_detail_rejects_missing_base_metric() -> None:
    detail = {"affordance": "grasp", "object": "mug", "image": "mug_001.jpg"}

    try:
        validate_fusion_detail(detail)
    except ValueError as exc:
        assert "mKLD" in str(exc)
    else:
        raise AssertionError("validate_fusion_detail should reject incomplete rows")


def test_validate_global_metrics_accepts_release_shape() -> None:
    metrics = {
        "total_samples": 3,
        "processed_samples": 2,
        "resume_reused": 0,
        "skipped_samples": 1,
        "failed_samples": 0,
        "overall_metrics": {"mKLD_mean": 1.0, "mSIM_mean": 0.5, "mNSS_mean": 0.25},
        "per_affordance_metrics": {},
        "skipped_detail": [],
        "failed_detail": [],
    }

    assert validate_global_metrics(metrics) == metrics
    assert FUSION_GLOBAL_METRIC_KEYS == tuple(metrics.keys())


def test_sample_artifact_contract_names_expected_directories() -> None:
    assert FUSION_SAMPLE_ARTIFACTS == (
        "kontext/",
        "mapped/",
        "metrics.json",
        "geom_pipeline/stage_roi/",
        "geom_pipeline/stage_dino/",
        "geom_pipeline/stage_geom/",
        "geom_pipeline/stage_final/",
    )
