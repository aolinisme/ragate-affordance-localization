"""AGD20K affordance sample iteration utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Optional

__all__ = ["SampleEntry", "iter_agd20k_samples"]


@dataclass
class SampleEntry:
    affordance: str
    object_name: str
    image_path: Path
    gt_path: Path


def iter_agd20k_samples(
    dataset_root: Path,
    affordances: Optional[Iterable[str]] = None,
    max_per_object: Optional[int] = None,
) -> Iterator[SampleEntry]:
    """Iterate through AGD20K Unseen/testset samples."""

    egocentric_root = dataset_root / "egocentric"
    gt_root = dataset_root / "GT"

    if not egocentric_root.exists():
        raise FileNotFoundError(f"Egocentric root not found: {egocentric_root}")
    if not gt_root.exists():
        raise FileNotFoundError(f"GT root not found: {gt_root}")

    affordance_list = sorted(path.name for path in egocentric_root.iterdir() if path.is_dir())
    if affordances is not None:
        allow = set(affordances)
        affordance_list = [affordance for affordance in affordance_list if affordance in allow]

    for affordance in affordance_list:
        affordance_dir = egocentric_root / affordance
        for object_dir in sorted(path for path in affordance_dir.iterdir() if path.is_dir()):
            object_name = object_dir.name
            gt_object_dir = gt_root / affordance / object_name
            if not gt_object_dir.exists():
                continue

            images = sorted(
                path for ext in ("*.png", "*.jpg", "*.jpeg") for path in object_dir.glob(ext)
            )
            if max_per_object is not None:
                images = images[:max_per_object]

            for image_path in images:
                gt_candidates = [
                    gt_object_dir / (image_path.stem + ext)
                    for ext in (".png", ".jpg", ".jpeg")
                ]
                gt_path = next((path for path in gt_candidates if path.exists()), None)
                if gt_path is None:
                    continue
                yield SampleEntry(
                    affordance=affordance,
                    object_name=object_name,
                    image_path=image_path,
                    gt_path=gt_path,
                )
