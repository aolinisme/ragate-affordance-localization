#!/usr/bin/env python3
"""Fuse SD-Turbo attention with DINO/PCA geometry using SD object ROI."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
FUSION_SRC = REPO_ROOT / "fusion_zero_shot/src"
DINO_SRC = FUSION_SRC / "dino"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(FUSION_SRC) not in sys.path:
    sys.path.insert(0, str(FUSION_SRC))
if str(DINO_SRC) not in sys.path:
    sys.path.insert(0, str(DINO_SRC))

from pba.data.affordance import SampleEntry, iter_agd20k_samples  # noqa: E402
from pba.metrics.affordance import cal_kl, cal_nss, cal_sim  # noqa: E402
from pba.fusion.config import load_config  # noqa: E402
from pipeline.roi_stage import build_roi_mask, compute_roi_tokens, restore_to_original  # noqa: E402
from pipeline.pca_stage import extract_dino_tokens, run_pca  # noqa: E402
from pipeline.geometry_stage import generate_geometry_mask  # noqa: E402
from pipeline.utils import save_colormap, save_overlay  # noqa: E402


def _read_summary(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return {
            "/".join([row["affordance"], row["object"], row["image"]]): row
            for row in csv.DictReader(f)
            if row.get("affordance") and row.get("object") and row.get("image")
        }


def _sample_key(sample: SampleEntry) -> str:
    return "/".join([sample.affordance, sample.object_name, sample.image_path.name])


def _load_npy(path_value: str | Path) -> np.ndarray:
    arr = np.load(Path(path_value)).astype(np.float32)
    if arr.ndim == 3:
        arr = arr.squeeze()
    arr = np.clip(arr, 0.0, None)
    vmax = float(arr.max())
    if vmax > 0:
        arr = arr / vmax
    return arr.astype(np.float32)


def _load_gt(path: Path, shape: tuple[int, int]) -> np.ndarray:
    arr = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if arr is None:
        raise FileNotFoundError(path)
    arr = cv2.resize(arr.astype(np.float32) / 255.0, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)
    if float(arr.max()) > 0:
        arr = arr / float(arr.max())
    return arr.astype(np.float32)


def _metrics(pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    return {
        "mKLD": float(cal_kl(pred, gt)),
        "mSIM": float(cal_sim(pred, gt)),
        "mNSS": float(cal_nss(pred, gt)),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _save_gray(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((np.clip(arr, 0.0, 1.0) * 255).astype(np.uint8), mode="L").save(path)


def _resolve_config_path(value: str | None, base: Path) -> Path | None:
    if not value:
        return None
    path = Path(value)
    if not path.is_absolute():
        path = (base / path).resolve()
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SD object ROI + DINO/PCA + SD verb fusion from cached attention.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--sd-summary", type=Path, required=True)
    parser.add_argument("--fusion-config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--affordances", nargs="+", default=["hold", "cut", "open", "type_on", "drink_with", "eat", "ride", "throw", "wash"])
    parser.add_argument("--max-images-per-object", type=int, default=3)
    parser.add_argument("--max-samples-total", type=int, default=60)
    parser.add_argument("--base-lambda", type=float, default=0.65)
    args = parser.parse_args()

    cfg = load_config(args.fusion_config)
    geom_cfg = cfg.get("geom_pipeline", {})
    config_dir = args.fusion_config.parent
    sd_rows = _read_summary(args.sd_summary)
    out_dir = args.output_root / time.strftime("%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    dino_target_wh = geom_cfg.get("dino_target_wh") or (1280, 960)
    dino_patch_size = int(geom_cfg.get("dino_patch_size") or 16)
    dino_model_name = str(geom_cfg.get("dino_model_name", "dinov3_vit7b16"))
    dino_cache_only = bool(geom_cfg.get("dino_cache_only", True))
    cache_root = _resolve_config_path(str(geom_cfg.get("cache_root", "./cache")), config_dir)
    if cache_root is None:
        cache_root = config_dir / "cache"

    samples = list(iter_agd20k_samples(args.dataset_root, args.affordances, args.max_images_per_object))
    samples = [sample for sample in samples if _sample_key(sample) in sd_rows][: args.max_samples_total]

    rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for sample in samples:
        key = _sample_key(sample)
        sd_row = sd_rows[key]
        sample_dir = out_dir / sample.affordance / sample.object_name / sample.image_path.stem
        try:
            sd_heat = _load_npy(Path(sd_row["heatmap"]).with_name("verb_attention.npy"))
            object_heat_path = sd_row.get("object_heatmap", "")
            object_heat = _load_npy(Path(object_heat_path).with_name("object_attention.npy")) if object_heat_path else sd_heat
            artifacts = extract_dino_tokens(
                sample.image_path,
                target_wh=tuple(dino_target_wh),
                patch_size=dino_patch_size,
                cache_root=cache_root,
                cache_only=dino_cache_only,
                model_name=dino_model_name,
            )
            roi_dir = sample_dir / "stage_roi"
            geom_dir = sample_dir / "stage_geom"
            roi_dir.mkdir(parents=True, exist_ok=True)
            geom_dir.mkdir(parents=True, exist_ok=True)

            roi_orig, roi_letterbox, roi_info = build_roi_mask(
                object_heat,
                artifacts.meta,
                percentile=float(geom_cfg.get("roi_percentile", 85.0)),
            )
            roi_indices, token_mask = compute_roi_tokens(
                roi_letterbox,
                artifacts.Hp,
                artifacts.Wp,
                artifacts.patch,
                threshold=float(geom_cfg.get("token_threshold", 0.1)),
            )
            roi_info["token_count"] = int(roi_indices.size)
            roi_info["token_fraction"] = float(roi_indices.size) / max(1, artifacts.tokens.shape[0])
            _save_gray(roi_dir / "sd_object_roi_mask.png", roi_orig)
            np.save(roi_dir / "sd_object_roi_letterbox.npy", roi_letterbox.astype(np.float32))
            np.save(roi_dir / "token_mask.npy", token_mask.astype(np.float32))
            save_overlay(sample.image_path, roi_orig, roi_dir / "sd_object_roi_overlay.png", alpha=0.45)
            (roi_dir / "roi_info.json").write_text(json.dumps(roi_info, indent=2) + "\n", encoding="utf-8")

            pca_outputs = run_pca(artifacts, roi_indices, num_components=int(geom_cfg.get("pca_components", 3)))
            orig_full = np.clip(pca_outputs["orig_full"], 0.0, 1.0)
            if sd_heat.shape != orig_full.shape[:2]:
                sd_heat_for_geom = cv2.resize(sd_heat, (orig_full.shape[1], orig_full.shape[0]), interpolation=cv2.INTER_LINEAR)
            else:
                sd_heat_for_geom = sd_heat
            geom_outputs = generate_geometry_mask(
                orig_full,
                smooth_sigma=float(geom_cfg.get("geom_sigma", 1.2)),
                binary_threshold=float(geom_cfg.get("geom_threshold", 0.55)),
                verb_map=sd_heat_for_geom,
                enable_soft_fusion=True,
                soft_lambda=args.base_lambda,
                soft_gamma=float(geom_cfg.get("geom_soft_gamma", 0.7)),
                soft_temperature=float(geom_cfg.get("geom_soft_temperature", 1.15) if not isinstance(geom_cfg.get("geom_soft_temperature", 1.15), list) else geom_cfg.get("geom_soft_temperature", [1.15])[0]),
                soft_dirichlet=float(geom_cfg.get("geom_soft_base", 0.008)),
                soft_use_log1p=bool(geom_cfg.get("geom_soft_log1p", False)),
                max_channels=int(geom_cfg.get("pca_components", 3)),
                use_attention_fallback=bool(geom_cfg.get("geom_sim_use_nss", False)),
                attention_topk_percent=float(geom_cfg.get("geom_sim_topk_percent", 10.0)),
                attention_nss_weight=float(geom_cfg.get("geom_sim_nss_weight", 1.0)),
                attention_topk_weight=float(geom_cfg.get("geom_sim_topk_weight", 1.0)),
            )
            geom_energy = np.asarray(geom_outputs["energy"], dtype=np.float32)
            soft = geom_outputs["soft_fusion"]["map"]
            _save_gray(geom_dir / "geom_energy.png", geom_energy)
            save_colormap(geom_energy, geom_dir / "geom_energy_colormap.png")
            np.save(geom_dir / "geom_energy.npy", geom_energy.astype(np.float32))
            _save_gray(geom_dir / "soft_fusion_heat.png", soft)
            save_colormap(soft, geom_dir / "soft_fusion_colormap.png")
            np.save(geom_dir / "soft_fusion_heat.npy", np.asarray(soft, dtype=np.float32))

            gt = _load_gt(sample.gt_path, soft.shape)
            sd_metrics = _metrics(sd_heat_for_geom, gt)
            geom_metrics = _metrics(geom_energy, gt)
            soft_metrics = _metrics(soft, gt)
            rows.append(
                {
                    "affordance": sample.affordance,
                    "object": sample.object_name,
                    "image": sample.image_path.name,
                    "sd_heatmap": sd_row["heatmap"],
                    "sd_object_heatmap": sd_row.get("object_heatmap", ""),
                    "sd_roi_mask": str(roi_dir / "sd_object_roi_mask.png"),
                    "geom_energy": str(geom_dir / "geom_energy.npy"),
                    "soft_heatmap": str(geom_dir / "soft_fusion_heat.npy"),
                    "roi_token_count": roi_info["token_count"],
                    "roi_token_fraction": roi_info["token_fraction"],
                    "sd_mKLD": sd_metrics["mKLD"],
                    "sd_mSIM": sd_metrics["mSIM"],
                    "sd_mNSS": sd_metrics["mNSS"],
                    "geom_mKLD": geom_metrics["mKLD"],
                    "geom_mSIM": geom_metrics["mSIM"],
                    "geom_mNSS": geom_metrics["mNSS"],
                    "soft_mKLD": soft_metrics["mKLD"],
                    "soft_mSIM": soft_metrics["mSIM"],
                    "soft_mNSS": soft_metrics["mNSS"],
                    "sd_infer_seconds": sd_row.get("infer_seconds", ""),
                }
            )
        except Exception as exc:  # noqa: BLE001
            failures.append({"key": key, "error": repr(exc)})

    _write_csv(out_dir / "summary.csv", rows)
    metrics = {
        "samples": len(rows),
        "failed": len(failures),
        "mean": {
            name: float(np.mean([float(row[name]) for row in rows])) if rows else 0.0
            for name in ("sd_mKLD", "sd_mSIM", "sd_mNSS", "geom_mKLD", "geom_mSIM", "geom_mNSS", "soft_mKLD", "soft_mSIM", "soft_mNSS")
        },
        "mean_sd_infer_seconds": float(np.mean([float(row["sd_infer_seconds"]) for row in rows if str(row["sd_infer_seconds"]).strip()])) if rows else 0.0,
        "summary_csv": str(out_dir / "summary.csv"),
        "failures": failures,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
