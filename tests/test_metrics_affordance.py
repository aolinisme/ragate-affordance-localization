import numpy as np

from pba.metrics.affordance import cal_kl, cal_nss, cal_sim


def test_cal_kl_matches_existing_formula() -> None:
    pred = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    gt = np.array([[4.0, 3.0], [2.0, 1.0]], dtype=np.float32)
    eps = 1e-12
    map1 = pred / (pred.sum() + eps)
    map2 = gt / (gt.sum() + eps)
    expected = float(np.sum(map2 * np.log(map2 / (map1 + eps) + eps)))

    assert cal_kl(pred, gt) == expected


def test_cal_sim_matches_existing_formula() -> None:
    pred = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    gt = np.array([[4.0, 3.0], [2.0, 1.0]], dtype=np.float32)
    eps = 1e-12
    map1 = pred / (pred.sum() + eps)
    map2 = gt / (gt.sum() + eps)
    expected = float(np.sum(np.minimum(map1, map2)))

    assert cal_sim(pred, gt) == expected


def test_cal_nss_matches_existing_formula() -> None:
    pred = np.array([[0.0, 64.0], [128.0, 255.0]], dtype=np.float32)
    gt = np.array([[0.0, 0.0], [255.0, 255.0]], dtype=np.float32)
    eps = 1e-12
    pred_norm = pred / 255.0
    gt_norm = gt / 255.0
    smap = (pred_norm - np.mean(pred_norm)) / np.std(pred_norm)
    fixation_map = (gt_norm - np.min(gt_norm)) / (np.max(gt_norm) - np.min(gt_norm) + eps)
    fixation_map = (fixation_map > 0.1).astype(np.float32)
    expected = float(np.sum(smap * fixation_map) / (fixation_map.sum() + eps))

    assert cal_nss(pred, gt) == expected


def test_cal_nss_returns_zero_for_constant_prediction() -> None:
    pred = np.ones((2, 2), dtype=np.float32)
    gt = np.array([[0.0, 255.0], [0.0, 255.0]], dtype=np.float32)

    assert cal_nss(pred, gt) == 0.0
