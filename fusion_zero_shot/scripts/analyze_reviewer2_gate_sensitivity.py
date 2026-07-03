#!/usr/bin/env python3
"""Reviewer-facing cached sensitivity and uncertainty-gate baselines.

This script reuses cached balanced AGD20K artifacts. It does not call FLUX,
DINOv3, or SD-Turbo. It reloads mapped interaction heatmaps, DINO/PCA geometry
energy maps, and GT masks, then recomputes soft fusion under alternative gate
settings and simple uncertainty baselines.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from statistics import mean
from typing import Any

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fusion_zero_shot.src.pipeline.geometry_stage import (  # noqa: E402
    _compute_adaptive_fusion_meta,
    _soft_fuse_heatmaps,
)
from pba.metrics.affordance import cal_kl, cal_nss, cal_sim  # noqa: E402

METRICS = ("soft_mKLD", "soft_mSIM", "soft_mNSS")
HIGHER_BETTER = {"soft_mKLD": False, "soft_mSIM": True, "soft_mNSS": True}
KEY_FIELDS = ("affordance", "object", "image")


def _read_rows(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = csv.DictReader(f)
        return {
            "/".join(row.get(field, "") for field in KEY_FIELDS): row
            for row in rows
            if row.get("affordance") and row.get("object") and row.get("image")
        }


def _float(value: Any, default: float | None = None) -> float | None:
    if value is None:
        return default
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return default
    try:
        out = float(text)
    except ValueError:
        return default
    if not math.isfinite(out):
        return default
    return out


def _load_gray(path: str | Path, *, normalize: bool = False) -> np.ndarray | None:
    arr = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if arr is None:
        return None
    arr = arr.astype(np.float32)
    if normalize:
        arr = arr / 255.0
    return arr


def _load_npy(path: str | Path) -> np.ndarray | None:
    p = Path(path)
    if not p.exists():
        return None
    arr = np.load(p).astype(np.float32)
    if arr.ndim == 3:
        arr = arr.squeeze()
    return arr


def _normalize01(arr: np.ndarray) -> np.ndarray:
    values = np.clip(np.asarray(arr, dtype=np.float32), 0.0, None)
    vmin = float(values.min())
    vmax = float(values.max())
    if vmax - vmin < 1e-8:
        return np.zeros_like(values, dtype=np.float32)
    return ((values - vmin) / (vmax - vmin)).astype(np.float32)


def _load_gt(path: str | Path, shape: tuple[int, int]) -> np.ndarray | None:
    gt = _load_gray(path)
    if gt is None:
        return None
    gt = cv2.resize(gt, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)
    return gt.astype(np.float32)


def _metrics(pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    pred_eval = np.clip(pred.astype(np.float32), 0.0, None)
    if float(pred_eval.max()) <= 1.5:
        pred_eval = pred_eval * 255.0
    gt_eval = gt.astype(np.float32)
    return {
        "soft_mKLD": float(cal_kl(pred_eval, gt_eval)),
        "soft_mSIM": float(cal_sim(pred_eval, gt_eval)),
        "soft_mNSS": float(cal_nss(pred_eval, gt_eval)),
    }


def _quality(metrics: dict[str, float], ranges: dict[str, tuple[float, float]]) -> float:
    scores = []
    for metric in METRICS:
        lo, hi = ranges[metric]
        denom = max(hi - lo, 1e-8)
        value = float(metrics[metric])
        if HIGHER_BETTER[metric]:
            score = (value - lo) / denom
        else:
            score = (hi - value) / denom
        scores.append(float(np.clip(score, 0.0, 1.0)))
    return float(mean(scores))


def _ranges(metric_sets: list[dict[str, float]]) -> dict[str, tuple[float, float]]:
    out = {}
    for metric in METRICS:
        vals = [float(item[metric]) for item in metric_sets]
        out[metric] = (min(vals), max(vals)) if vals else (0.0, 1.0)
    return out


def _metric_delta(candidate: np.ndarray, fixed: np.ndarray, metric: str) -> np.ndarray:
    if HIGHER_BETTER[metric]:
        return candidate - fixed
    return fixed - candidate


def _wtl(values: list[dict[str, Any]], prefix: str, metric: str, eps: float = 1e-9) -> str:
    deltas = []
    for row in values:
        fixed = float(row[f"fixed_{metric}"])
        candidate = float(row[f"{prefix}_{metric}"])
        deltas.append(float(_metric_delta(np.asarray([candidate]), np.asarray([fixed]), metric)[0]))
    arr = np.asarray(deltas, dtype=np.float64)
    return f"{int((arr > eps).sum())}/{int((np.abs(arr) <= eps).sum())}/{int((arr < -eps).sum())}"


def _fold_id(key: str, folds: int) -> int:
    acc = 0
    for ch in key.encode("utf-8"):
        acc = (acc * 131 + ch) % 1000003
    return acc % folds


def _heat_stats(arr: np.ndarray) -> dict[str, float]:
    values = _normalize01(arr)
    flat = values.reshape(-1)
    if flat.size == 0 or float(flat.sum()) <= 1e-8:
        return {"entropy": 1.0, "variance": 0.0, "top5_mass": 0.0, "peak": 0.0}
    prob = flat / float(flat.sum())
    entropy = -float(np.sum(prob * np.log(prob + 1e-8))) / math.log(float(flat.size))
    k = max(1, int(math.ceil(flat.size * 0.05)))
    top = np.partition(prob, -k)[-k:]
    return {
        "entropy": entropy,
        "variance": float(np.var(values)),
        "top5_mass": float(top.sum()),
        "peak": float(flat.max()),
    }


def _candidate_thresholds(values: list[float]) -> list[float]:
    clean = sorted({float(v) for v in values if math.isfinite(float(v))})
    if not clean:
        return [0.0]
    out = [min(clean) - 1e-6, max(clean) + 1e-6]
    out.extend(clean)
    out.extend((a + b) / 2.0 for a, b in zip(clean[:-1], clean[1:]))
    return sorted(set(out))


def _aggregate(rows: list[dict[str, Any]], prefix: str) -> dict[str, float]:
    return {metric: float(mean(float(row[f"{prefix}_{metric}"]) for row in rows)) for metric in METRICS}


def _evaluate_threshold(rows: list[dict[str, Any]], score_field: str, threshold: float, ranges: dict[str, tuple[float, float]]) -> tuple[dict[str, float], int, float]:
    chosen = []
    adapted = 0
    for row in rows:
        use_adaptive = float(row[score_field]) >= threshold
        prefix = "adaptive_full" if use_adaptive else "fixed"
        adapted += int(use_adaptive)
        chosen.append({metric: float(row[f"{prefix}_{metric}"]) for metric in METRICS})
    metrics = {metric: float(mean(item[metric] for item in chosen)) for metric in METRICS}
    return metrics, adapted, _quality(metrics, ranges)


def _cv_selector(rows: list[dict[str, Any]], score_field: str, folds: int, ranges: dict[str, tuple[float, float]]) -> dict[str, Any]:
    thresholds = _candidate_thresholds([float(row[score_field]) for row in rows])
    predictions = {}
    selected_thresholds = []
    for fold in range(folds):
        train = [row for row in rows if int(row["fold"]) != fold]
        test = [row for row in rows if int(row["fold"]) == fold]
        if not train or not test:
            continue
        best_threshold = thresholds[0]
        best_quality = -1.0
        for threshold in thresholds:
            _, _, quality = _evaluate_threshold(train, score_field, threshold, ranges)
            if quality > best_quality:
                best_quality = quality
                best_threshold = threshold
        selected_thresholds.append(float(best_threshold))
        for row in test:
            use_adaptive = float(row[score_field]) >= best_threshold
            prefix = "adaptive_full" if use_adaptive else "fixed"
            predictions[row["key"]] = {
                "adapted": int(use_adaptive),
                "threshold": float(best_threshold),
                **{metric: float(row[f"{prefix}_{metric}"]) for metric in METRICS},
            }
    ordered = [predictions[row["key"]] for row in rows if row["key"] in predictions]
    metrics = {metric: float(mean(row[metric] for row in ordered)) for metric in METRICS}
    adapted = int(sum(int(row["adapted"]) for row in ordered))
    return {
        "metrics": metrics,
        "adapted": adapted,
        "fallback": len(ordered) - adapted,
        "thresholds": selected_thresholds,
        "threshold_mean": float(mean(selected_thresholds)) if selected_thresholds else float("nan"),
        "predictions": predictions,
    }


def _compute_lambda(
    row: dict[str, Any],
    *,
    min_lambda: float,
    max_lambda: float,
    verb_weight: float,
    geometry_weight: float,
    alignment_weight: float,
    base_lambda: float,
) -> float:
    interaction = max(verb_weight, 0.0) * float(row["interaction_confidence"])
    geometry = max(geometry_weight, 0.0) * float(row["geometry_confidence"]) + max(alignment_weight, 0.0) * float(row["alignment_confidence"])
    total = interaction + geometry
    lo, hi = sorted((float(min_lambda), float(max_lambda)))
    if total <= 1e-8:
        return float(np.clip(base_lambda, lo, hi))
    return float(np.clip(lo + (hi - lo) * interaction / total, lo, hi))


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path: Path, summary: dict[str, Any], method_rows: list[dict[str, Any]], sensitivity_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Reviewer-2 Cached Gate Sensitivity",
        "",
        "Evidence tier: cached balanced44 recomputation from mapped heatmaps, DINO/PCA geometry energy, and GT masks.",
        "",
        f"- Valid recomputed samples: `{summary['valid_samples']}`",
        f"- Folds: `{summary['folds']}`",
        f"- FLUX/DINO/SD inference rerun: `false`",
        "",
        "## Adaptive Baselines",
        "",
        "| Method | KLD lower | SIM higher | NSS higher | Adapted | Fallback |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in method_rows:
        lines.append(
            f"| {row['method']} | {row['soft_mKLD']:.4f} | {row['soft_mSIM']:.4f} | {row['soft_mNSS']:.4f} | {row['adapted']} | {row['fallback']} |"
        )
    lines.extend(
        [
            "",
            "## Parameter Sensitivity",
            "",
            "| Setting | KLD lower | SIM higher | NSS higher | Adapted | Note |",
            "|---|---:|---:|---:|---:|---|",
        ]
    )
    for row in sensitivity_rows:
        lines.append(
            f"| {row['setting']} | {row['soft_mKLD']:.4f} | {row['soft_mSIM']:.4f} | {row['soft_mNSS']:.4f} | {row['adapted']} | {row['note']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixed-summary", type=Path, required=True)
    parser.add_argument("--wide-summary", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--prefix", default="reviewer2_gate_sensitivity")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--base-lambda", type=float, default=0.65)
    parser.add_argument("--soft-gamma", type=float, default=0.7)
    parser.add_argument("--soft-temperature", type=float, default=0.5)
    parser.add_argument("--soft-dirichlet", type=float, default=0.008)
    args = parser.parse_args()

    fixed_rows = _read_rows(args.fixed_summary)
    wide_rows = _read_rows(args.wide_summary)
    keys = sorted(set(fixed_rows) & set(wide_rows))

    rows: list[dict[str, Any]] = []
    metric_sets: list[dict[str, float]] = []
    for key in keys:
        fixed = fixed_rows[key]
        wide = wide_rows[key]
        verb = _load_gray(wide.get("warped_heatmap", ""), normalize=True)
        geom_path = Path(wide.get("geom_mask", "")).parent / "geom_energy.npy"
        geom = _load_npy(geom_path)
        if verb is None or geom is None:
            continue
        if geom.shape != verb.shape:
            geom = cv2.resize(geom, (verb.shape[1], verb.shape[0]), interpolation=cv2.INTER_LINEAR)
        gt = _load_gt(wide.get("gt", ""), verb.shape)
        if gt is None:
            continue
        meta = _compute_adaptive_fusion_meta(
            verb,
            geom,
            base_lambda=args.base_lambda,
            min_lambda=0.20,
            max_lambda=0.90,
            verb_weight=0.45,
            geometry_weight=0.35,
            alignment_weight=0.20,
        )
        fixed_heat = _soft_fuse_heatmaps(
            verb,
            geom,
            lam=args.base_lambda,
            gamma=args.soft_gamma,
            temperature=args.soft_temperature,
            dirichlet=args.soft_dirichlet,
            use_log1p=False,
        )
        full_heat = _soft_fuse_heatmaps(
            verb,
            geom,
            lam=float(meta["adaptive_lambda"]),
            gamma=args.soft_gamma,
            temperature=args.soft_temperature,
            dirichlet=args.soft_dirichlet,
            use_log1p=False,
        )
        fixed_metrics = _metrics(fixed_heat, gt)
        full_metrics = _metrics(full_heat, gt)
        verb_stats = _heat_stats(verb)
        geom_stats = _heat_stats(geom)
        row = {
            "key": key,
            "affordance": wide.get("affordance", ""),
            "object": wide.get("object", ""),
            "image": wide.get("image", ""),
            "fold": _fold_id(key, args.folds),
            "interaction_confidence": float(meta["interaction_confidence"]),
            "geometry_confidence": float(meta["geometry_confidence"]),
            "alignment_confidence": float(meta["alignment_confidence"]),
            "selected_similarity": float(_float(wide.get("geom_similarity_best"), 0.0) or 0.0),
            "verb_entropy_score": 1.0 - verb_stats["entropy"],
            "verb_variance_score": verb_stats["variance"],
            "verb_top5_score": verb_stats["top5_mass"],
            "geom_entropy_score": 1.0 - geom_stats["entropy"],
            "geom_variance_score": geom_stats["variance"],
            "geom_top5_score": geom_stats["top5_mass"],
            "fixed_lambda": args.base_lambda,
            "adaptive_lambda": float(meta["adaptive_lambda"]),
        }
        for metric in METRICS:
            row[f"fixed_{metric}"] = fixed_metrics[metric]
            row[f"adaptive_full_{metric}"] = full_metrics[metric]
        rows.append(row)
        metric_sets.extend([fixed_metrics, full_metrics])

    ranges = _ranges(metric_sets)
    method_rows: list[dict[str, Any]] = [
        {"method": "Fixed fusion recomputed", **_aggregate(rows, "fixed"), "adapted": 0, "fallback": len(rows)},
        {"method": "Always-on RA weight recomputed", **_aggregate(rows, "adaptive_full"), "adapted": len(rows), "fallback": 0},
    ]
    per_sample_rows = [dict(row) for row in rows]

    selector_specs = [
        ("Entropy selector (verb)", "verb_entropy_score"),
        ("Variance selector (verb)", "verb_variance_score"),
        ("Top-k mass selector (verb)", "verb_top5_score"),
        ("Entropy selector (geometry)", "geom_entropy_score"),
        ("Variance selector (geometry)", "geom_variance_score"),
        ("Top-k mass selector (geometry)", "geom_top5_score"),
    ]
    selector_summary = {}
    for label, field in selector_specs:
        cv = _cv_selector(rows, field, args.folds, ranges)
        selector_summary[label] = {
            "score_field": field,
            "threshold_mean": cv["threshold_mean"],
            "thresholds": cv["thresholds"],
            "adapted": cv["adapted"],
            "fallback": cv["fallback"],
            "metrics": cv["metrics"],
        }
        method_rows.append({"method": label, **cv["metrics"], "adapted": cv["adapted"], "fallback": cv["fallback"]})
        safe = label.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_")
        for row in per_sample_rows:
            pred = cv["predictions"].get(row["key"])
            if pred:
                row[f"{safe}_adapted"] = pred["adapted"]
                row[f"{safe}_threshold"] = pred["threshold"]
                for metric in METRICS:
                    row[f"{safe}_{metric}"] = pred[metric]

    sensitivity_specs = [
        ("base", 0.20, 0.90, 0.45, 0.35, 0.20, 0.90, "paper setting"),
        ("lambda-range narrow", 0.35, 0.85, 0.45, 0.35, 0.20, 0.90, "narrower lambda range"),
        ("lambda-range wide", 0.10, 0.95, 0.45, 0.35, 0.20, 0.90, "wider lambda range"),
        ("interaction-heavy", 0.20, 0.90, 0.60, 0.25, 0.15, 0.90, "more verb evidence"),
        ("geometry-heavy", 0.20, 0.90, 0.30, 0.50, 0.20, 0.90, "more geometry evidence"),
        ("alignment-heavy", 0.20, 0.90, 0.40, 0.25, 0.35, 0.90, "more alignment evidence"),
        ("floor 0.70", 0.20, 0.90, 0.45, 0.35, 0.20, 0.70, "more aggressive floor"),
        ("floor 1.50", 0.20, 0.90, 0.45, 0.35, 0.20, 1.50, "more conservative floor"),
    ]
    sensitivity_rows: list[dict[str, Any]] = []
    for name, lo, hi, wi, wg, wa, floor, note in sensitivity_specs:
        chosen = []
        adapted = 0
        for row in rows:
            lam = _compute_lambda(
                row,
                min_lambda=lo,
                max_lambda=hi,
                verb_weight=wi,
                geometry_weight=wg,
                alignment_weight=wa,
                base_lambda=args.base_lambda,
            )
            if float(row["selected_similarity"]) < floor:
                prefix = "fixed"
                lam = args.base_lambda
            else:
                prefix = "sensitivity"
                adapted += 1
            if prefix == "fixed":
                metrics = {metric: float(row[f"fixed_{metric}"]) for metric in METRICS}
            else:
                wide = wide_rows[row["key"]]
                verb = _load_gray(wide.get("warped_heatmap", ""), normalize=True)
                geom = _load_npy(Path(wide.get("geom_mask", "")).parent / "geom_energy.npy")
                gt = _load_gt(wide.get("gt", ""), verb.shape) if verb is not None else None
                if verb is None or geom is None or gt is None:
                    metrics = {metric: float(row[f"fixed_{metric}"]) for metric in METRICS}
                else:
                    if geom.shape != verb.shape:
                        geom = cv2.resize(geom, (verb.shape[1], verb.shape[0]), interpolation=cv2.INTER_LINEAR)
                    heat = _soft_fuse_heatmaps(
                        verb,
                        geom,
                        lam=lam,
                        gamma=args.soft_gamma,
                        temperature=args.soft_temperature,
                        dirichlet=args.soft_dirichlet,
                        use_log1p=False,
                    )
                    metrics = _metrics(heat, gt)
            chosen.append(metrics)
        item = {"setting": name, "adapted": adapted, "note": note}
        for metric in METRICS:
            item[metric] = float(mean(m[metric] for m in chosen))
        sensitivity_rows.append(item)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "valid_samples": len(rows),
        "folds": args.folds,
        "method_table": method_rows,
        "selector_summary": selector_summary,
        "sensitivity_table": sensitivity_rows,
        "notes": {
            "inference_rerun": False,
            "uses_gt_for_evaluation_only": True,
            "selector_thresholds_selected_inside_folds": True,
        },
    }
    json_path = args.out_dir / f"{args.prefix}.json"
    methods_path = args.out_dir / f"{args.prefix}_methods.csv"
    sensitivity_path = args.out_dir / f"{args.prefix}_sensitivity.csv"
    per_sample_path = args.out_dir / f"{args.prefix}_per_sample.csv"
    md_path = args.out_dir / f"{args.prefix}.md"
    json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    _write_csv(methods_path, method_rows)
    _write_csv(sensitivity_path, sensitivity_rows)
    _write_csv(per_sample_path, per_sample_rows)
    _write_markdown(md_path, summary, method_rows, sensitivity_rows)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
