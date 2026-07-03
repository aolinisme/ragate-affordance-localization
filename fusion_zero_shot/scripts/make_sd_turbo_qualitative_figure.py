#!/usr/bin/env python3
"""Create a qualitative SD-Turbo attention figure for the manuscript."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def _read_rows(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return {"/".join([r["affordance"], r["object"], r["image"]]): r for r in csv.DictReader(f)}


def _find_image(dataset_root: Path, row: dict[str, str]) -> Path:
    path = dataset_root / "egocentric" / row["affordance"] / row["object"] / row["image"]
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def _find_gt(dataset_root: Path, row: dict[str, str]) -> Path:
    stem = Path(row["image"]).stem
    base = dataset_root / "GT" / row["affordance"] / row["object"]
    for ext in (".png", ".jpg", ".jpeg"):
        path = base / f"{stem}{ext}"
        if path.exists():
            return path
    raise FileNotFoundError(base / f"{stem}.png")


def _load_rgb(path: Path, size: tuple[int, int]) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB").resize(size, Image.Resampling.LANCZOS))


def _load_gray(path: Path, size: tuple[int, int]) -> np.ndarray:
    arr = np.asarray(Image.open(path).convert("L").resize(size, Image.Resampling.BILINEAR)).astype(np.float32)
    if arr.max() > 0:
        arr /= arr.max()
    return arr


def _overlay(image: np.ndarray, heat: np.ndarray) -> np.ndarray:
    cmap = plt.get_cmap("magma")
    heat_rgb = (cmap(np.clip(heat, 0, 1))[..., :3] * 255).astype(np.uint8)
    return (0.58 * image + 0.42 * heat_rgb).astype(np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--keys",
        nargs="+",
        default=[
            "eat/broccoli/broccoli_000476.jpg",
            "type_on/laptop/laptop_000585.jpg",
            "open/refrigerator/refrigerator_000884.jpg",
        ],
    )
    parser.add_argument("--cell-size", type=int, default=224)
    args = parser.parse_args()

    rows = _read_rows(args.summary)
    selected = [rows[key] for key in args.keys]
    size = (args.cell_size, args.cell_size)

    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
    })
    fig, axes = plt.subplots(len(selected), 4, figsize=(9.2, 2.65 * len(selected)), dpi=220)
    if len(selected) == 1:
        axes = np.expand_dims(axes, axis=0)
    row_tags = ["aligned prior", "contact-region prior", "failure boundary"]
    headers = ["Input", "GT affordance", "SD-Turbo prior", "Prior overlay"]
    for col, header in enumerate(headers):
        axes[0, col].set_title(header, pad=7)

    for row_idx, row in enumerate(selected):
        image = _load_rgb(_find_image(args.dataset_root, row), size)
        gt = _load_gray(_find_gt(args.dataset_root, row), size)
        heat = _load_gray(Path(row["heatmap"]), size)
        overlay = _overlay(image, heat)
        panels = [image, gt, heat, overlay]
        cmaps = [None, "gray", "magma", None]
        label = (
            f"{row_tags[row_idx]}\n"
            f"{row['affordance'].replace('_', ' ')} {row['object'].replace('_', ' ')}"
        )
        score = f"KLD {float(row['mKLD']):.2f} | SIM {float(row['mSIM']):.2f} | NSS {float(row['mNSS']):.2f}"
        for col, (panel, cmap) in enumerate(zip(panels, cmaps)):
            ax = axes[row_idx, col]
            ax.imshow(panel, cmap=cmap, vmin=0, vmax=1 if panel.ndim == 2 else None)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(0.6)
                spine.set_edgecolor("#d0d0d0")
            if col == 0:
                ax.set_ylabel(label, fontsize=9.5, fontweight="bold", rotation=0, ha="right", va="center", labelpad=58)
            if col == 2:
                ax.text(
                    0.5,
                    -0.08,
                    score,
                    ha="center",
                    va="top",
                    transform=ax.transAxes,
                    fontsize=8.5,
                    color="#333333",
                )

    fig.tight_layout(pad=0.7, w_pad=0.55, h_pad=1.1)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    if args.out.suffix.lower() == ".pdf":
        fig.savefig(args.out.with_suffix(".png"), bbox_inches="tight")
    plt.close(fig)
    print(args.out)


if __name__ == "__main__":
    main()
