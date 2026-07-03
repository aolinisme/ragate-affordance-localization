from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np

from pba.metrics.affordance import cal_kl, cal_nss, cal_sim


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_module(path: Path, name: str):
    spec = spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {path}")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_legacy_agd20k_metrics_match_pba_metrics() -> None:
    legacy = load_module(
        REPO_ROOT / "fusion_zero_shot/src/agd20k_eval/metrics.py",
        "legacy_agd20k_metrics",
    )
    pred = np.array([[0.0, 64.0], [128.0, 255.0]], dtype=np.float32)
    gt = np.array([[255.0, 128.0], [64.0, 0.0]], dtype=np.float32)

    assert legacy.cal_kl(pred, gt) == cal_kl(pred, gt)
    assert legacy.cal_sim(pred, gt) == cal_sim(pred, gt)
    assert legacy.cal_nss(pred, gt) == cal_nss(pred, gt)
