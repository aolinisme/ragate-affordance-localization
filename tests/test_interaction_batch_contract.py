from pathlib import Path

from pba.data.affordance import SampleEntry
from pba.interaction.batch import BATCH_METRIC_COLUMNS
from pba.interaction.batch import INTERACTION_BATCH_OUTPUT_FILES
from pba.interaction.batch import build_batch_sample_id
from pba.interaction.batch import build_batch_sample_paths
from pba.interaction.batch import validate_metric_columns


def test_build_batch_sample_id_is_stable_for_agd20k_sample() -> None:
    sample = SampleEntry(
        affordance="hold",
        object_name="mug",
        image_path=Path("egocentric/hold/mug/0001.jpg"),
        gt_path=Path("GT/hold/mug/0001.png"),
    )

    assert build_batch_sample_id(sample) == "hold/mug/0001"


def test_build_batch_sample_paths_declares_required_outputs(tmp_path: Path) -> None:
    sample = SampleEntry(
        affordance="cut",
        object_name="knife",
        image_path=Path("egocentric/cut/knife/img_01.jpg"),
        gt_path=Path("GT/cut/knife/img_01.png"),
    )

    paths = build_batch_sample_paths(tmp_path, sample)

    assert set(paths) == INTERACTION_BATCH_OUTPUT_FILES
    assert paths["verb_heatmap"].as_posix().endswith("cut/knife/img_01/verb_heat.png")
    assert paths["verb_heatmap_npy"].as_posix().endswith("cut/knife/img_01/verb_heat.npy")
    assert paths["metadata"].as_posix().endswith("cut/knife/img_01/meta.json")


def test_validate_metric_columns_accepts_contract_columns() -> None:
    assert validate_metric_columns(BATCH_METRIC_COLUMNS) == tuple(BATCH_METRIC_COLUMNS)


def test_validate_metric_columns_rejects_missing_column() -> None:
    bad_columns = [column for column in BATCH_METRIC_COLUMNS if column != "nss"]

    try:
        validate_metric_columns(bad_columns)
    except ValueError as exc:
        assert "missing metric columns: nss" in str(exc)
    else:
        raise AssertionError("missing nss column should fail")
