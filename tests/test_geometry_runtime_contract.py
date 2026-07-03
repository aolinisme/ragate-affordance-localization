from pba.geometry.runtime import (
    GEOMETRY_BACKBONE_TARGETS,
    GEOMETRY_RUNTIME_MODULES,
    GEOMETRY_TRAINING_OUTPUT_FILES,
    GeometryRuntimeBoundary,
    build_runtime_boundary,
    validate_geometry_metrics,
)


def test_build_runtime_boundary_documents_legacy_modules() -> None:
    boundary = build_runtime_boundary()

    assert boundary == GeometryRuntimeBoundary(
        package_module="pba.geometry",
        legacy_root="geometry_probing/umd_linear_probing/src",
        train_entrypoint="pba.geometry.linear_probe.run_train",
        eval_entrypoint="pba.geometry.linear_probe.run_eval",
        modules=GEOMETRY_RUNTIME_MODULES,
        backbone_targets=GEOMETRY_BACKBONE_TARGETS,
    )
    assert "src.engine.trainer.LinearProbeExperiment" in boundary.modules
    assert "src.engine.eval.evaluate_linear_probe" in boundary.modules
    assert "src.data.dataset.UMDAffordanceDataset" in boundary.modules
    assert "dinov3" in boundary.backbone_targets
    assert "stable_diffusion" in boundary.backbone_targets


def test_validate_geometry_metrics_accepts_eval_result_shape() -> None:
    metrics = {
        "loss": 0.25,
        "miou": 0.42,
        "per_class_iou": [0.1, 0.2, 0.3],
    }

    assert validate_geometry_metrics(metrics) == metrics


def test_validate_geometry_metrics_rejects_missing_per_class_iou() -> None:
    metrics = {"loss": 0.25, "miou": 0.42}

    try:
        validate_geometry_metrics(metrics)
    except ValueError as exc:
        assert "per_class_iou" in str(exc)
    else:
        raise AssertionError("validate_geometry_metrics should reject incomplete metrics")


def test_geometry_training_output_contract_names_checkpoint_and_logs() -> None:
    assert GEOMETRY_TRAINING_OUTPUT_FILES == (
        "master.log",
        "summary.json",
        "training_history.json",
        "val_examples.pt",
        "test_examples.pt",
        "{sweep}/train.log",
        "{sweep}/config_snapshot.yaml",
        "{sweep}/linear_probe.pth",
    )
