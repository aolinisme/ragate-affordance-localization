from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "fusion_zero_shot/scripts/run_tiny_reliability_router.py"


def load_router_module():
    spec = spec_from_file_location("tiny_reliability_router_for_tests", SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {SCRIPT}")
    module = module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_tiny_router_lambda_is_clamped_to_configured_range() -> None:
    mod = load_router_module()
    model = mod.TinyReliabilityRouter(input_dim=len(mod.FEATURE_KEYS), hidden_dim=16)

    x = torch.zeros(4, len(mod.FEATURE_KEYS))
    p_flux, lam = model(x, 0.20, 0.90)

    assert torch.all((0.0 <= p_flux) & (p_flux <= 1.0))
    assert torch.all((0.20 <= lam) & (lam <= 0.90))


def test_router_feature_keys_exclude_metric_and_gt_leakage() -> None:
    mod = load_router_module()

    assert not (set(mod.FEATURE_KEYS) & mod.LEAKY_FEATURE_NAMES)
    for key in mod.FEATURE_KEYS:
        assert "KLD" not in key
        assert "SIM" not in key
        assert "NSS" not in key
        assert key != "gt"


def test_fold_assignment_is_deterministic_and_in_range() -> None:
    mod = load_router_module()

    key = "hold/axe/axe_000108.jpg"
    first = mod._fold_for(key, 5)
    second = mod._fold_for(key, 5)

    assert first == second
    assert 0 <= first < 5
