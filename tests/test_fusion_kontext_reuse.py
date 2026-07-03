import importlib.util
import json
import sys
from pathlib import Path

from pba.fusion.paths import get_heatmap_path


REPO_ROOT = Path(__file__).resolve().parents[1]
AGD_EVAL_DIR = REPO_ROOT / "fusion_zero_shot/src/agd20k_eval"


def _load_eval_module():
    sys.path.insert(0, str(AGD_EVAL_DIR))
    spec = importlib.util.spec_from_file_location(
        "run_flux_kontext_eval_for_test",
        AGD_EVAL_DIR / "run_flux_kontext_eval.py",
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_get_heatmap_path_accepts_ascii_alias(tmp_path: Path) -> None:
    heatmap = tmp_path / "heat_tok00.png"
    heatmap.write_bytes(b"placeholder")

    assert get_heatmap_path(tmp_path, 0, "hold") == heatmap


def test_load_reused_kontext_result_reads_existing_artifacts(tmp_path: Path) -> None:
    module = _load_eval_module()
    sample = module.SampleEntry(
        affordance="hold",
        object_name="axe",
        image_path=tmp_path / "axe_000108.jpg",
        gt_path=tmp_path / "axe_000108.png",
    )
    exp_dir = tmp_path / "old_run" / "hold" / "axe" / "axe_000108" / "kontext" / "hold_axe_001"
    per_token_dir = exp_dir / "per_token"
    per_token_dir.mkdir(parents=True)
    (exp_dir / "gen.png").write_bytes(b"placeholder")
    (per_token_dir / "heat_tok00.png").write_bytes(b"placeholder")
    tokens = [{"id": 0, "tid": 1, "tok": "hold"}]
    (exp_dir / "tokens_t5.json").write_text(json.dumps(tokens), encoding="utf-8")

    result = module.load_reused_kontext_result(sample, tmp_path / "old_run")

    assert result["tokens"] == tokens
    assert result["per_token_dir"] == per_token_dir
    assert result["generated_image"] == exp_dir / "gen.png"


def test_build_sample_allowlist_accepts_slash_and_backslash_keys() -> None:
    module = _load_eval_module()

    allowlist = module.build_sample_allowlist(
        ["hold/axe/axe_000108.jpg", r"cut\banana\banana_000253.jpg"]
    )

    assert allowlist == {
        ("hold", "axe", "axe_000108.jpg"),
        ("cut", "banana", "banana_000253.jpg"),
    }


def test_build_sample_allowlist_rejects_invalid_key() -> None:
    module = _load_eval_module()

    try:
        module.build_sample_allowlist(["hold/axe"])
    except ValueError as exc:
        assert "affordance/object/image" in str(exc)
    else:
        raise AssertionError("Expected ValueError for invalid sample key")
