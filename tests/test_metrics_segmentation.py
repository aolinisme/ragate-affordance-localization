import importlib.util

import pytest


def test_segmentation_metrics_module_exists() -> None:
    assert importlib.util.find_spec("pba.metrics.segmentation") is not None


def test_segmentation_metrics_match_existing_confusion_matrix_behavior() -> None:
    torch = pytest.importorskip("torch")
    from pba.metrics.segmentation import compute_iou, update_confusion_matrix

    confmat = torch.zeros((3, 3), dtype=torch.int64)
    preds = torch.tensor([[0, 1, 2], [1, 2, 0]])
    targets = torch.tensor([[0, 1, 255], [2, 2, 0]])

    confmat = update_confusion_matrix(
        confmat,
        preds=preds,
        targets=targets,
        num_classes=3,
        ignore_index=255,
    )

    expected = torch.tensor(
        [
            [2, 0, 0],
            [0, 1, 0],
            [0, 1, 1],
        ],
        dtype=torch.int64,
    )
    assert torch.equal(confmat, expected)

    metrics = compute_iou(confmat)
    assert torch.allclose(metrics["per_class"], torch.tensor([1.0, 0.5, 0.5]))
    assert torch.isclose(metrics["miou"], torch.tensor(2.0 / 3.0))
