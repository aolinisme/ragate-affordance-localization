#!/usr/bin/env python3
"""Create qualitative TinyRouter figure from cached balanced44 artifacts."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def _read(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return {"/".join([r["affordance"], r["object"], r["image"]]): r for r in csv.DictReader(f)}


def _load_rgb(path: Path, size: tuple[int, int]) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB").resize(size, Image.Resampling.LANCZOS))


def _load_gray(path: Path, size: tuple[int, int]) -> np.ndarray:
    arr = np.asarray(Image.open(path).convert("L").resize(size, Image.Resampling.BILINEAR)).astype(np.float32)
    if float(arr.max()) > 0:
        arr = arr / float(arr.max())
    return arr


def _load_npy(path: Path, size: tuple[int, int]) -> np.ndarray:
    arr = np.load(path).astype(np.float32)
    if arr.ndim == 3:
        arr = arr.squeeze()
    img = Image.fromarray((np.clip(arr, 0.0, 1.0) * 255).astype(np.uint8), mode="L").resize(size, Image.Resampling.BILINEAR)
    out = np.asarray(img).astype(np.float32) / 255.0
    if float(out.max()) > 0:
        out = out / float(out.max())
    return out


def _overlay(image: np.ndarray, heat: np.ndarray) -> np.ndarray:
    cmap = plt.get_cmap("magma")
    heat_rgb = (cmap(np.clip(heat, 0, 1))[..., :3] * 255).astype(np.uint8)
    return (0.58 * image + 0.42 * heat_rgb).astype(np.uint8)


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


def _pick_keys(rows: list[dict[str, str]]) -> list[str]:
    preferred = [
        "throw/basketball/basketball_000647.jpg",
        "hold/knife/knife_000029.jpg",
        "open/refrigerator/refrigerator_000884.jpg",
    ]
    available = {r["key"] for r in rows}
    keys = [key for key in preferred if key in available]
    if len(keys) == 3:
        return keys

    sd_success = max(rows, key=lambda r: float(r["sd_interaction_mSIM"]))
    fallback = max([r for r in rows if int(r["use_flux"]) == 1] or rows, key=lambda r: float(r["router_route_mNSS"]))
    hard = min(rows, key=lambda r: max(float(r["sd_interaction_mSIM"]), float(r["flux_ragate_mSIM"])))
    for row in (sd_success, fallback, hard):
        key = row["key"]
        if key not in keys:
            keys.append(key)
    return keys[:3]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--router-csv", type=Path, required=True)
    parser.add_argument("--sd-summary", type=Path, required=True)
    parser.add_argument("--sd-dino-summary", type=Path, required=True)
    parser.add_argument("--flux-summary", type=Path, required=True)
    parser.add_argument("--adaptive-summary", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cell-size", type=int, default=208)
    args = parser.parse_args()

    with args.router_csv.open("r", encoding="utf-8-sig", newline="") as f:
        router_rows = list(csv.DictReader(f))
    router = {r["key"]: r for r in router_rows}
    sd = _read(args.sd_summary)
    sd_dino = _read(args.sd_dino_summary)
    adaptive = _read(args.adaptive_summary)
    keys = _pick_keys(router_rows)
    size = (args.cell_size, args.cell_size)

    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 10.5,
        "axes.titleweight": "bold",
    })
    row_tags = ["SD default", "FLUX fallback", "failure boundary"]
    headers = ["Input", "GT affordance", "SD prior", "SD-DINO prior", "FLUX RA-Gate", "Router output"]
    fig, axes = plt.subplots(len(keys), len(headers), figsize=(12.2, 2.6 * len(keys)), dpi=220)
    if len(keys) == 1:
        axes = np.expand_dims(axes, axis=0)
    for col, header in enumerate(headers):
        axes[0, col].set_title(header, pad=6)

    for row_idx, key in enumerate(keys):
        r = router[key]
        sd_row = sd[key]
        sd_dino_row = sd_dino[key]
        adaptive_row = adaptive[key]
        image = _load_rgb(_find_image(args.dataset_root, sd_row), size)
        gt = _load_gray(_find_gt(args.dataset_root, sd_row), size)
        sd_heat = _load_gray(Path(sd_row["heatmap"]), size)
        sd_dino_heat = _load_npy(Path(sd_dino_row["soft_heatmap"]), size)
        flux_heat = _load_gray(Path(adaptive_row["soft_heatmap"]), size)
        router_heat = flux_heat if int(r["use_flux"]) == 1 else sd_dino_heat
        panels = [image, gt, sd_heat, _overlay(image, sd_dino_heat), _overlay(image, flux_heat), _overlay(image, router_heat)]
        cmaps = [None, "gray", "magma", None, None, None]
        action_label = (
            f"{row_tags[row_idx]}\n"
            f"{r['affordance'].replace('_', ' ')} {r['object'].replace('_', ' ')}"
        )
        route_label = f"p(FLUX)={float(r['p_flux']):.2f}, call={r['use_flux']}"
        for col, (panel, cmap) in enumerate(zip(panels, cmaps)):
            ax = axes[row_idx, col]
            ax.imshow(panel, cmap=cmap, vmin=0, vmax=1 if getattr(panel, "ndim", 3) == 2 else None)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(0.6)
                spine.set_edgecolor("#d0d0d0")
            if col == 0:
                ax.set_ylabel(action_label, fontsize=9, fontweight="bold", rotation=0, ha="right", va="center", labelpad=55)
            if col == len(headers) - 1:
                ax.text(
                    0.5,
                    -0.08,
                    route_label,
                    ha="center",
                    va="top",
                    transform=ax.transAxes,
                    fontsize=8.4,
                    color="#333333",
                )
    fig.tight_layout(pad=0.65, w_pad=0.45, h_pad=1.05)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    if args.out.suffix.lower() == ".pdf":
        fig.savefig(args.out.with_suffix(".png"), bbox_inches="tight")
    plt.close(fig)
    print(args.out)


if __name__ == "__main__":
    main()
