from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

from pba.data.affordance import SampleEntry, iter_agd20k_samples


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_module(path: Path, name: str):
    spec = spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {path}")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_iter_agd20k_samples_matches_unseen_testset_layout(tmp_path: Path) -> None:
    img_dir = tmp_path / "egocentric" / "grasp" / "mug"
    gt_dir = tmp_path / "GT" / "grasp" / "mug"
    img_dir.mkdir(parents=True)
    gt_dir.mkdir(parents=True)
    (img_dir / "0001.jpg").write_bytes(b"image")
    (gt_dir / "0001.png").write_bytes(b"mask")

    samples = list(iter_agd20k_samples(tmp_path))

    assert samples == [
        SampleEntry(
            affordance="grasp",
            object_name="mug",
            image_path=img_dir / "0001.jpg",
            gt_path=gt_dir / "0001.png",
        )
    ]


def test_iter_agd20k_samples_filters_affordances_and_limits_per_object(tmp_path: Path) -> None:
    for affordance in ("grasp", "cut"):
        img_dir = tmp_path / "egocentric" / affordance / "tool"
        gt_dir = tmp_path / "GT" / affordance / "tool"
        img_dir.mkdir(parents=True)
        gt_dir.mkdir(parents=True)
        for idx in range(2):
            (img_dir / f"{idx}.jpg").write_bytes(b"image")
            (gt_dir / f"{idx}.jpg").write_bytes(b"mask")

    samples = list(iter_agd20k_samples(tmp_path, affordances=["cut"], max_per_object=1))

    assert len(samples) == 1
    assert samples[0].affordance == "cut"


def test_legacy_agd20k_iterator_reexports_pba_data() -> None:
    legacy = load_module(
        REPO_ROOT / "fusion_zero_shot/src/agd20k_eval/utils/data_iter.py",
        "legacy_agd20k_data_iter",
    )

    assert legacy.SampleEntry is SampleEntry
    assert legacy.iter_agd20k_samples is iter_agd20k_samples
