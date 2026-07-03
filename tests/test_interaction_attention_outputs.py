from pathlib import Path

import numpy as np

from pba.interaction.attention import AttentionAccumulator
from pba.interaction.outputs import build_probe_metadata, prepare_output_root


def test_attention_accumulator_infers_square_grid_and_averages_maps() -> None:
    acc = AttentionAccumulator({"hold": 1})
    acc.storage["hold"].append(np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32))
    acc.storage["hold"].append(np.array([[2.0, 3.0], [4.0, 5.0]], dtype=np.float32))

    assert acc._infer_hw(4) == 2
    assert acc._infer_hw(5) is None
    np.testing.assert_allclose(acc.summary()["hold"], np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))


def test_attention_accumulator_export_writes_expected_files(tmp_path: Path) -> None:
    from PIL import Image

    acc = AttentionAccumulator({"hold": 0})
    maps = {"hold": np.array([[0.0, 1.0], [0.5, 0.25]], dtype=np.float32)}
    base = Image.new("RGB", (4, 4), color=(120, 120, 120))

    acc.export(maps, base, tmp_path)

    assert (tmp_path / "hold_heat.png").exists()
    assert (tmp_path / "hold_overlay.png").exists()
    assert (tmp_path / "hold_heat.npy").exists()
    assert np.load(tmp_path / "hold_heat.npy").shape == (4, 4)


def test_prepare_output_root_creates_resolved_directory(tmp_path: Path) -> None:
    output = prepare_output_root(tmp_path / "probe")

    assert output == (tmp_path / "probe").resolve()
    assert output.is_dir()


def test_build_probe_metadata_matches_legacy_schema() -> None:
    metadata = build_probe_metadata(
        model_id="models/FLUX.1-Kontext-dev",
        prompt="hold the mug",
        affordance="hold",
        token_map={"hold": 2},
        steps=20,
        guidance=3.0,
        seed=7,
    )

    assert metadata == {
        "model_id": "models/FLUX.1-Kontext-dev",
        "prompt": "hold the mug",
        "affordance": "hold",
        "tokens_tracked": {"hold": 2},
        "steps": 20,
        "guidance": 3.0,
        "seed": 7,
    }
