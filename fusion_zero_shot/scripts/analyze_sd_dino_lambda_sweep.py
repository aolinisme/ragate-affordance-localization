#!/usr/bin/env python3
"""Cached SD-Turbo + DINO/PCA lambda sweep for low-compute analysis."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
FUSION_SRC = REPO_ROOT / "fusion_zero_shot/src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(FUSION_SRC) not in sys.path:
    sys.path.insert(0, str(FUSION_SRC))

from pba.data.affordance import iter_agd20k_samples  # noqa: E402
from pba.metrics.affordance import cal_kl, cal_nss, cal_sim  # noqa: E402
from pipeline.geometry_stage import _soft_fuse_heatmaps  # noqa: E402


METRICS = ("mKLD", "mSIM", "mNSS")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _sample_key(affordance: str, object_name: str, image: str) -> str:
    return "/".join([affordance, object_name, image])


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


def _prob_map(arr: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    values = np.clip(np.asarray(arr, dtype=np.float32), 0.0, None)
    total = float(values.sum())
    if total <= eps:
        return np.full(values.shape, 1.0 / max(1, values.size), dtype=np.float32)
    return (values / total).astype(np.float32)


def _linear_prob_fuse(verb_map: np.ndarray, geom_map: np.ndarray, lam: float) -> np.ndarray:
    lam = float(np.clip(lam, 0.0, 1.0))
    fused = lam * _prob_map(verb_map) + (1.0 - lam) * _prob_map(geom_map)
    return _prob_map(fused)


def _quality(metrics: dict[str, float], ranges: dict[str, tuple[float, float]]) -> float:
    values = []
    for metric in METRICS:
        lo, hi = ranges[metric]
        denom = max(hi - lo, 1e-8)
        value = float(metrics[metric])
        if metric == "mKLD":
            score = (hi - value) / denom
        else:
            score = (value - lo) / denom
        values.append(float(np.clip(score, 0.0, 1.0)))
    return float(np.mean(values))


def _mean(rows: list[dict[str, float]], key: str) -> float:
    return float(np.mean([float(row[key]) for row in rows])) if rows else 0.0


def _summarize(name: str, rows: list[dict[str, float]], seconds: float = 0.0) -> dict[str, Any]:
    return {
        "method": name,
        "samples": len(rows),
        "mKLD": _mean(rows, "mKLD"),
        "mSIM": _mean(rows, "mSIM"),
        "mNSS": _mean(rows, "mNSS"),
        "mean_seconds": seconds,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--prefix", default="sd_dino_lambda_sweep")
    parser.add_argument("--affordances", nargs="+", default=["hold", "cut", "open", "type_on", "drink_with", "eat", "ride", "throw", "wash"])
    parser.add_argument("--max-images-per-object", type=int, default=8)
    parser.add_argument("--gamma", type=float, default=0.7)
    parser.add_argument("--temperature", type=float, default=0.30)
    parser.add_argument("--dirichlet", type=float, default=0.0)
    parser.add_argument("--step", type=float, default=0.05)
    parser.add_argument("--fusion-mode", choices=["linear", "log"], default="linear")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    gt_by_key = {
        _sample_key(sample.affordance, sample.object_name, sample.image_path.name): sample.gt_path
        for sample in iter_agd20k_samples(args.dataset_root, args.affordances, args.max_images_per_object)
    }

    lambdas = [round(float(x), 4) for x in np.arange(0.0, 1.0 + args.step / 2.0, args.step)]
    rows = _read_csv(args.summary)
    per_sample_rows: list[dict[str, Any]] = []
    lambda_rows: list[dict[str, Any]] = []
    all_method_metrics: list[dict[str, float]] = []
    failures: list[dict[str, str]] = []

    loaded: list[dict[str, Any]] = []
    for row in rows:
        key = _sample_key(row["affordance"], row["object"], row["image"])
        try:
            sd_path = Path(row["sd_heatmap"]).with_name("verb_attention.npy")
            sd = _load_npy(sd_path)
            geom = _load_npy(row["geom_energy"])
            if sd.shape != geom.shape:
                sd = cv2.resize(sd, (geom.shape[1], geom.shape[0]), interpolation=cv2.INTER_LINEAR)
            gt = _load_gt(gt_by_key[key], geom.shape)
            sd_metrics = _metrics(sd, gt)
            geom_metrics = _metrics(geom, gt)
            loaded.append(
                {
                    "key": key,
                    "affordance": row["affordance"],
                    "object": row["object"],
                    "image": row["image"],
                    "sd": sd,
                    "geom": geom,
                    "gt": gt,
                    "sd_metrics": sd_metrics,
                    "geom_metrics": geom_metrics,
                    "sd_seconds": float(row.get("sd_infer_seconds") or 0.0),
                }
            )
            all_method_metrics.extend([sd_metrics, geom_metrics])
        except Exception as exc:  # noqa: BLE001
            failures.append({"key": key, "error": repr(exc)})

    sweep_by_lambda: dict[float, list[dict[str, float]]] = {lam: [] for lam in lambdas}
    for item in loaded:
        for lam in lambdas:
            if args.fusion_mode == "linear":
                fused = _linear_prob_fuse(item["sd"], item["geom"], lam=lam)
            else:
                fused = _soft_fuse_heatmaps(
                    item["sd"],
                    item["geom"],
                    lam=lam,
                    gamma=args.gamma,
                    temperature=args.temperature,
                    dirichlet=args.dirichlet,
                    use_log1p=False,
                )
            metrics = _metrics(fused, item["gt"])
            sweep_by_lambda[lam].append(metrics)
            all_method_metrics.append(metrics)

    metric_ranges: dict[str, tuple[float, float]] = {}
    for metric in METRICS:
        values = [float(m[metric]) for m in all_method_metrics]
        metric_ranges[metric] = (float(min(values)), float(max(values)))

    sd_rows = [item["sd_metrics"] for item in loaded]
    geom_rows = [item["geom_metrics"] for item in loaded]
    mean_sd_seconds = float(np.mean([item["sd_seconds"] for item in loaded])) if loaded else 0.0
    method_rows: list[dict[str, Any]] = [
        _summarize("SD-Turbo interaction", sd_rows, mean_sd_seconds),
        _summarize("DINO/PCA geometry", geom_rows, 0.0),
    ]
    for lam in lambdas:
        metrics_rows = sweep_by_lambda[lam]
        row = _summarize(f"SD+DINO fixed lambda={lam:.2f}", metrics_rows, mean_sd_seconds)
        row["lambda"] = lam
        lambda_rows.append(row)
        method_rows.append(row)

    best_by_kld = min(lambda_rows, key=lambda row: float(row["mKLD"])) if lambda_rows else None
    best_by_sim = max(lambda_rows, key=lambda row: float(row["mSIM"])) if lambda_rows else None
    best_by_nss = max(lambda_rows, key=lambda row: float(row["mNSS"])) if lambda_rows else None
    best_by_quality = max(
        lambda_rows,
        key=lambda row: _quality({"mKLD": row["mKLD"], "mSIM": row["mSIM"], "mNSS": row["mNSS"]}, metric_ranges),
    ) if lambda_rows else None

    oracle_rows: list[dict[str, float]] = []
    oracle_choice_counts: dict[str, int] = {}
    for item in loaded:
        candidates: list[tuple[str, float, dict[str, float]]] = [
            ("sd", _quality(item["sd_metrics"], metric_ranges), item["sd_metrics"]),
            ("geometry", _quality(item["geom_metrics"], metric_ranges), item["geom_metrics"]),
        ]
        for lam in lambdas:
            metrics = sweep_by_lambda[lam][len(oracle_rows)]
            candidates.append((f"lambda={lam:.2f}", _quality(metrics, metric_ranges), metrics))
        choice, quality, metrics = max(candidates, key=lambda entry: entry[1])
        oracle_choice_counts[choice] = oracle_choice_counts.get(choice, 0) + 1
        oracle_rows.append(metrics)
        per_sample = {
            "key": item["key"],
            "affordance": item["affordance"],
            "object": item["object"],
            "image": item["image"],
            "oracle_choice": choice,
            "oracle_quality": quality,
        }
        for metric in METRICS:
            per_sample[f"sd_{metric}"] = item["sd_metrics"][metric]
            per_sample[f"geom_{metric}"] = item["geom_metrics"][metric]
            per_sample[f"oracle_{metric}"] = metrics[metric]
        per_sample_rows.append(per_sample)
    method_rows.append(_summarize("Oracle per-sample selector", oracle_rows, mean_sd_seconds))

    paths = {
        "methods_csv": str(args.out_dir / f"{args.prefix}_methods.csv"),
        "lambda_csv": str(args.out_dir / f"{args.prefix}_lambda_sweep.csv"),
        "per_sample_csv": str(args.out_dir / f"{args.prefix}_per_sample.csv"),
        "json": str(args.out_dir / f"{args.prefix}.json"),
        "markdown": str(args.out_dir / f"{args.prefix}.md"),
    }
    summary = {
        "samples": len(loaded),
        "failed": len(failures),
        "failures": failures,
        "metric_ranges": metric_ranges,
        "fusion_mode": args.fusion_mode,
        "best_by_kld": best_by_kld,
        "best_by_sim": best_by_sim,
        "best_by_nss": best_by_nss,
        "best_by_quality": best_by_quality,
        "oracle_choice_counts": oracle_choice_counts,
        "paths": paths,
    }

    _write_csv(Path(paths["methods_csv"]), method_rows)
    _write_csv(Path(paths["lambda_csv"]), lambda_rows)
    _write_csv(Path(paths["per_sample_csv"]), per_sample_rows)
    Path(paths["json"]).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# SD-Turbo + DINO/PCA cached lambda sweep",
        "",
        f"- Samples: `{len(loaded)}`",
        f"- Failed: `{len(failures)}`",
        f"- Fusion mode: `{args.fusion_mode}`",
        f"- Best fixed lambda by KLD: `{best_by_kld['lambda'] if best_by_kld else 'n/a'}`",
        f"- Best fixed lambda by SIM: `{best_by_sim['lambda'] if best_by_sim else 'n/a'}`",
        f"- Best fixed lambda by NSS: `{best_by_nss['lambda'] if best_by_nss else 'n/a'}`",
        f"- Best fixed lambda by normalized mean quality: `{best_by_quality['lambda'] if best_by_quality else 'n/a'}`",
        f"- Oracle choices: `{oracle_choice_counts}`",
        "",
        "| Method | KLD | SIM | NSS |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in method_rows:
        if row["method"] in {
            "SD-Turbo interaction",
            "DINO/PCA geometry",
            "Oracle per-sample selector",
        } or row.get("lambda") in {0.0, 0.5, 0.65, 0.9, 1.0}:
            lines.append(f"| {row['method']} | {row['mKLD']:.4f} | {row['mSIM']:.4f} | {row['mNSS']:.4f} |")
    Path(paths["markdown"]).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
