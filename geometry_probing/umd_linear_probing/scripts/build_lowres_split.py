#!/usr/bin/env python
"""Build a lower-resource train split from the capped UMD split."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_SPLIT = REPO_ROOT / "geometry_probing" / "umd_linear_probing" / "metadata" / "splits" / "category_split_seed42_v20_capped.json"
OUT_SPLIT = REPO_ROOT / "geometry_probing" / "umd_linear_probing" / "metadata" / "splits" / "category_split_seed42_v20_lowres_train4.json"

TRAIN_FRAMES_PER_TOOL = 4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a lower-resource train split from the capped UMD split."
    )
    parser.add_argument(
        "--src-split",
        type=Path,
        default=SRC_SPLIT,
        help="Source capped split JSON.",
    )
    parser.add_argument(
        "--out-split",
        type=Path,
        default=OUT_SPLIT,
        help="Output low-resource split JSON.",
    )
    parser.add_argument(
        "--train-frames-per-tool",
        type=int,
        default=TRAIN_FRAMES_PER_TOOL,
        help="Number of training frames retained for each tool.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.train_frames_per_tool <= 0:
        raise ValueError("--train-frames-per-tool must be positive.")

    with args.src_split.open("r", encoding="utf-8") as handle:
        src = json.load(handle)

    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for item in src["train"]:
        grouped[item["tool"]].append(item)

    train_subset: list[dict[str, str]] = []
    for tool in sorted(grouped):
        train_subset.extend(grouped[tool][: args.train_frames_per_tool])

    out = {
        "train": train_subset,
        "val": src["val"],
        "test": src["test"],
    }

    args.out_split.parent.mkdir(parents=True, exist_ok=True)
    with args.out_split.open("w", encoding="utf-8") as handle:
        json.dump(out, handle, indent=2)

    print("train_frames_per_tool", args.train_frames_per_tool)
    print("train", len(out["train"]))
    print("val", len(out["val"]))
    print("test", len(out["test"]))


if __name__ == "__main__":
    main()
