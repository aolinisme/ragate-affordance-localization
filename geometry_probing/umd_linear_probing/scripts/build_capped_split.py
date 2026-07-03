#!/usr/bin/env python
"""Build a small but category-diverse capped split from the public UMD split."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_SPLIT = REPO_ROOT / "geometry_probing" / "umd_linear_probing" / "metadata" / "splits" / "category_split_seed42_v20.json"
OUT_SPLIT = REPO_ROOT / "geometry_probing" / "umd_linear_probing" / "metadata" / "splits" / "category_split_seed42_v20_capped.json"


TARGET_TOOLS = {
    "train": ["bowl_01", "hammer_04", "knife_02", "saw_01", "shovel_02", "spoon_02", "trowel_02"],
    "val": ["bowl_02", "hammer_02", "knife_08", "scissors_01", "shovel_01", "spoon_04", "trowel_01"],
    "test": ["bowl_03", "hammer_01", "knife_01", "saw_03", "shovel_02", "spoon_01", "trowel_03"],
}

FRAMES_PER_TOOL = {
    "train": 12,
    "val": 8,
    "test": 8,
}


def main() -> None:
    with SRC_SPLIT.open("r", encoding="utf-8") as handle:
        src = json.load(handle)

    out: dict[str, list[dict[str, str]]] = {}
    for split in ("train", "val", "test"):
        grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
        for item in src[split]:
            grouped[item["tool"]].append(item)

        selected: list[dict[str, str]] = []
        for tool in TARGET_TOOLS[split]:
            selected.extend(grouped[tool][: FRAMES_PER_TOOL[split]])
        out[split] = selected

    with OUT_SPLIT.open("w", encoding="utf-8") as handle:
        json.dump(out, handle, indent=2)

    for split in ("train", "val", "test"):
        print(split, len(out[split]))


if __name__ == "__main__":
    main()
