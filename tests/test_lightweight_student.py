from __future__ import annotations

from pathlib import Path

import torch

from pba.geometry.lightweight_student import ReliabilityGatedStudent, _distillation_loss, _load_yaml


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_reliability_gated_student_output_shape() -> None:
    model = ReliabilityGatedStudent(num_classes=8, width=8)
    rgb = torch.randn(2, 3, 96, 128)
    depth = torch.randn(2, 1, 96, 128)

    logits = model(rgb, depth)

    assert logits.shape == (2, 8, 96, 128)
    assert sum(p.numel() for p in model.parameters()) < 200_000


def test_reliability_gated_student_fusion_modes() -> None:
    rgb = torch.randn(2, 3, 96, 128)
    depth = torch.randn(2, 1, 96, 128)
    param_counts = {}

    for mode in ("rgb", "depth", "concat", "gate", "depth_modulated_rgb"):
        model = ReliabilityGatedStudent(num_classes=8, width=8, fusion_mode=mode)
        logits = model(rgb, depth)
        param_counts[mode] = sum(p.numel() for p in model.parameters())

        assert logits.shape == (2, 8, 96, 128)

    assert param_counts["rgb"] < param_counts["concat"]
    assert param_counts["depth"] < param_counts["concat"]
    assert param_counts["concat"] != param_counts["gate"]
    assert param_counts["rgb"] < param_counts["depth_modulated_rgb"]


def test_reliability_gated_student_rejects_unknown_mode() -> None:
    try:
        ReliabilityGatedStudent(num_classes=8, width=8, fusion_mode="late_magic")
    except ValueError as exc:
        assert "Unsupported fusion_mode" in str(exc)
    else:
        raise AssertionError("Expected unsupported fusion mode to raise ValueError")


def test_lightweight_student_smoke_config_loads() -> None:
    cfg = _load_yaml(REPO_ROOT / "geometry_probing/umd_linear_probing/configs/lightweight_student_smoke.yaml")

    assert cfg["dataset"]["image_size"] == [240, 320]
    assert cfg["model"]["width"] == 16
    assert cfg["model"]["fusion_mode"] == "gate"
    assert cfg["training"]["smoke"] is True
    assert cfg["training"]["loss"]["class_balance"] is True


def test_distillation_loss_uses_teacher_availability() -> None:
    student = torch.randn(2, 8, 32, 32)
    teacher = torch.randn(2, 8, 2, 2)
    target = torch.ones(2, 32, 32, dtype=torch.long)

    active = _distillation_loss(
        student,
        teacher,
        torch.tensor([True, False]),
        target,
        temperature=2.0,
        foreground_only=True,
        ignore_index=255,
    )
    inactive = _distillation_loss(
        student,
        teacher,
        torch.tensor([False, False]),
        target,
        temperature=2.0,
        foreground_only=True,
        ignore_index=255,
    )

    assert active.item() >= 0.0
    assert inactive.item() == 0.0
