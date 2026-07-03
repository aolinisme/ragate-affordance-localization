#!/usr/bin/env python3
"""Offline TinyReliabilityRouter pilot for low-compute affordance fusion.

This script intentionally trains only on cached balanced-pilot artifacts.  It
does not call FLUX, SD-Turbo, or DINO.  The deployable features are computed
from SD-Turbo attention maps and cached DINO/PCA geometry maps; GT and metric
columns are used only to build offline oracle labels and evaluate held-out
folds.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pba.metrics.affordance import cal_kl, cal_nss, cal_sim  # noqa: E402
from fusion_zero_shot.src.pipeline.geometry_stage import _soft_fuse_heatmaps  # noqa: E402


METRICS = ("mKLD", "mSIM", "mNSS")
FEATURE_KEYS = (
    "sd_verb_entropy",
    "sd_verb_top5_mass",
    "sd_verb_peak",
    "sd_object_entropy",
    "sd_object_top5_mass",
    "sd_object_area",
    "geometry_confidence",
    "alignment_confidence",
    "sd_geometry_alignment",
    "geom_similarity_best",
    "roi_area_proxy",
)
LEAKY_FEATURE_NAMES = {
    "gt",
    "mKLD",
    "mSIM",
    "mNSS",
    "soft_mKLD",
    "soft_mSIM",
    "soft_mNSS",
    "final_mKLD",
    "final_mSIM",
    "final_mNSS",
    "flux_mKLD",
    "flux_mSIM",
    "flux_mNSS",
}


@dataclass
class Sample:
    key: str
    affordance: str
    object_name: str
    image: str
    features: np.ndarray
    sd_metrics: dict[str, float]
    flux_metrics: dict[str, float]
    flux_fixed_metrics: dict[str, float]
    flux_ragate_metrics: dict[str, float]
    sd_fixed_metrics: dict[str, float]
    lambda_metrics: dict[float, dict[str, float]]
    oracle_lambda: float
    route_label: float
    sd_seconds: float
    used_sd_object: bool


class TinyReliabilityRouter(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 16) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.route_head = nn.Linear(hidden_dim, 1)
        self.lambda_head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor, min_lambda: float, max_lambda: float) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.net(x)
        p_flux = torch.sigmoid(self.route_head(h)).squeeze(-1)
        lam_unit = torch.sigmoid(self.lambda_head(h)).squeeze(-1)
        lam = min_lambda + (max_lambda - min_lambda) * lam_unit
        return p_flux, lam


def _read_csv(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = csv.DictReader(f)
        return {
            "/".join([row["affordance"], row["object"], row["image"]]): row
            for row in rows
            if row.get("affordance") and row.get("object") and row.get("image")
        }


def _float(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = row.get(key, "")
    try:
        return float(value) if str(value).strip() else default
    except ValueError:
        return default


def _load_npy(path_value: str | Path) -> np.ndarray | None:
    path = Path(path_value)
    if not path.exists():
        return None
    arr = np.load(path).astype(np.float32)
    if arr.ndim == 3:
        arr = arr.squeeze()
    return arr


def _image_to_npy_path(path_value: str, name: str) -> Path:
    path = Path(path_value)
    return path.with_name(name)


def _load_gt(path_value: str, shape: tuple[int, int]) -> np.ndarray:
    arr = cv2.imread(path_value, cv2.IMREAD_GRAYSCALE)
    if arr is None:
        raise FileNotFoundError(path_value)
    arr = cv2.resize(arr.astype(np.float32) / 255.0, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)
    if float(arr.max()) > 0:
        arr = arr / float(arr.max())
    return arr.astype(np.float32)


def _normalize01(arr: np.ndarray) -> np.ndarray:
    values = np.clip(np.asarray(arr, dtype=np.float32), 0.0, None)
    vmin = float(values.min())
    vmax = float(values.max())
    if vmax - vmin < 1e-8:
        return np.zeros_like(values, dtype=np.float32)
    return ((values - vmin) / (vmax - vmin)).astype(np.float32)


def _heat_stats(arr: np.ndarray, top_percent: float = 5.0) -> dict[str, float]:
    values = _normalize01(arr)
    flat = values.reshape(-1)
    if flat.size == 0:
        return {"entropy": 0.0, "top5_mass": 0.0, "peak": 0.0, "area": 0.0}
    total = float(flat.sum())
    if total <= 1e-8:
        return {"entropy": 0.0, "top5_mass": 0.0, "peak": float(flat.max()), "area": 0.0}
    prob = flat / total
    entropy = -float(np.sum(prob * np.log(prob + 1e-8))) / math.log(float(flat.size))
    k = max(1, int(math.ceil(flat.size * top_percent / 100.0)))
    top = np.partition(prob, -k)[-k:]
    positive = flat[flat > 0.02]
    threshold = max(0.2, float(np.percentile(positive, 85.0))) if positive.size else 0.2
    area = float((flat >= threshold).mean())
    return {
        "entropy": entropy,
        "top5_mass": float(top.sum()),
        "peak": float(flat.max()),
        "area": area,
    }


def _alignment(first: np.ndarray, second: np.ndarray) -> float:
    a = _normalize01(first).reshape(-1)
    b = _normalize01(second).reshape(-1)
    if a.size != b.size:
        second = cv2.resize(second, (first.shape[1], first.shape[0]), interpolation=cv2.INTER_LINEAR)
        b = _normalize01(second).reshape(-1)
    an = float(np.linalg.norm(a))
    bn = float(np.linalg.norm(b))
    if an < 1e-8 or bn < 1e-8:
        return 0.0
    return float(np.clip(np.dot(a / an, b / bn), 0.0, 1.0))


def _metrics(pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    return {
        "mKLD": float(cal_kl(pred, gt)),
        "mSIM": float(cal_sim(pred, gt)),
        "mNSS": float(cal_nss(pred, gt)),
    }


def _fold_for(key: str, folds: int) -> int:
    digest = hashlib.md5(key.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % folds


def _quality(metrics: dict[str, float], ranges: dict[str, tuple[float, float]]) -> float:
    vals = []
    for metric in METRICS:
        lo, hi = ranges[metric]
        denom = max(hi - lo, 1e-8)
        raw = float(metrics[metric])
        if metric == "mKLD":
            vals.append((hi - raw) / denom)
        else:
            vals.append((raw - lo) / denom)
    return float(np.mean(np.clip(vals, 0.0, 1.0)))


def _metric_ranges(samples: list[dict[float, dict[str, float]]], extra: list[dict[str, float]]) -> dict[str, tuple[float, float]]:
    values: dict[str, list[float]] = {metric: [] for metric in METRICS}
    for sweep in samples:
        for metrics in sweep.values():
            for metric in METRICS:
                values[metric].append(float(metrics[metric]))
    for metrics in extra:
        for metric in METRICS:
            values[metric].append(float(metrics[metric]))
    return {
        metric: (float(min(vals)), float(max(vals))) if vals else (0.0, 1.0)
        for metric, vals in values.items()
    }


def _best_lambda(lambda_metrics: dict[float, dict[str, float]], ranges: dict[str, tuple[float, float]]) -> float:
    return max(lambda_metrics, key=lambda lam: _quality(lambda_metrics[lam], ranges))


def _train_model(
    train_samples: list[Sample],
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    *,
    min_lambda: float,
    max_lambda: float,
    epochs: int,
    lr: float,
    seed: int,
) -> TinyReliabilityRouter:
    torch.manual_seed(seed)
    model = TinyReliabilityRouter(len(FEATURE_KEYS))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    x_np = np.stack([(s.features - feature_mean) / feature_std for s in train_samples], axis=0).astype(np.float32)
    route_np = np.asarray([s.route_label for s in train_samples], dtype=np.float32)
    lambda_np = np.asarray([s.oracle_lambda for s in train_samples], dtype=np.float32)
    x = torch.from_numpy(x_np)
    route = torch.from_numpy(route_np)
    target_lambda = torch.from_numpy(lambda_np)
    for _ in range(epochs):
        p_flux, pred_lambda = model(x, min_lambda, max_lambda)
        route_loss = F.binary_cross_entropy(p_flux, route)
        lambda_loss = F.smooth_l1_loss(pred_lambda, target_lambda)
        loss = route_loss + lambda_loss
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model


def _aggregate(rows: list[dict[str, Any]], prefix: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for metric in METRICS:
        vals = [float(row[f"{prefix}_{metric}"]) for row in rows]
        out[metric] = float(np.mean(vals)) if vals else 0.0
    return out


def _wtl(rows: list[dict[str, Any]], cand_prefix: str, base_prefix: str, metric: str, eps: float = 1e-9) -> str:
    wins = ties = losses = 0
    for row in rows:
        cand = float(row[f"{cand_prefix}_{metric}"])
        base = float(row[f"{base_prefix}_{metric}"])
        diff = cand - base
        better = diff < -eps if metric == "mKLD" else diff > eps
        worse = diff > eps if metric == "mKLD" else diff < -eps
        if better:
            wins += 1
        elif worse:
            losses += 1
        else:
            ties += 1
    return f"{wins}/{ties}/{losses}"


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# TinyReliabilityRouter Balanced44 Pilot",
        "",
        "Evidence tier: balanced44 capped pilot with cached SD-Turbo, DINO/PCA, and FLUX artifacts.",
        "",
        "## Main Table",
        "",
        "| Method | KLD lower | SIM higher | NSS higher | Sec./img | FLUX calls |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, row in summary["methods"].items():
        lines.append(
            "| {name} | {mKLD:.4f} | {mSIM:.4f} | {mNSS:.4f} | {seconds:.4f} | {flux_calls}/{samples} |".format(
                name=name,
                samples=summary["samples"],
                **row,
            )
        )
    lines += [
        "",
        "## Notes",
        "",
        f"- Feature columns: `{', '.join(FEATURE_KEYS)}`.",
        f"- Lambda range: `{summary['lambda_range'][0]:.2f}` to `{summary['lambda_range'][1]:.2f}`.",
        f"- Router speedup vs always-FLUX: `{summary['router_speedup_vs_flux']:.2f}x`.",
        "- This is not a full benchmark claim; it tests whether a small pre-call router can support a low-compute deployment story.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_samples(args: argparse.Namespace) -> list[Sample]:
    sd_rows = _read_csv(args.sd_summary)
    flux_rows = _read_csv(args.flux_summary)
    adaptive_rows = _read_csv(args.adaptive_summary)
    sd_dino_rows = _read_csv(args.sd_dino_summary) if args.sd_dino_summary else {}
    keys = sorted(set(sd_rows) & set(flux_rows) & set(adaptive_rows))
    if sd_dino_rows:
        keys = sorted(set(keys) & set(sd_dino_rows))
    lambdas = [round(float(x), 2) for x in np.arange(args.min_lambda, args.max_lambda + 1e-6, args.lambda_step)]

    raw_records: list[dict[str, Any]] = []
    ranges_input: list[dict[float, dict[str, float]]] = []
    extras: list[dict[str, float]] = []

    for key in keys:
        sd = sd_rows[key]
        fx = flux_rows[key]
        ad = adaptive_rows[key]
        sd_dino = sd_dino_rows.get(key, {})
        sd_heat = _load_npy(_image_to_npy_path(sd["heatmap"], "verb_attention.npy"))
        if sd_dino.get("geom_energy"):
            geom = _load_npy(sd_dino["geom_energy"])
        else:
            geom = _load_npy(Path(ad["geom_mask"]).parent / "geom_energy.npy")
        if sd_heat is None or geom is None:
            continue
        geom = cv2.resize(geom, (sd_heat.shape[1], sd_heat.shape[0]), interpolation=cv2.INTER_LINEAR)
        gt = _load_gt(ad["gt"], sd_heat.shape)
        object_npy = _image_to_npy_path(sd.get("object_heatmap", ""), "object_attention.npy") if sd.get("object_heatmap") else None
        sd_object = _load_npy(object_npy) if object_npy is not None else None
        used_sd_object = sd_object is not None
        if sd_object is None:
            sd_object = sd_heat

        lambda_metrics: dict[float, dict[str, float]] = {}
        for lam in lambdas:
            fused = _soft_fuse_heatmaps(
                sd_heat,
                geom,
                lam=lam,
                gamma=args.soft_gamma,
                temperature=args.soft_temperature,
                dirichlet=args.soft_dirichlet,
                use_log1p=False,
            )
            lambda_metrics[lam] = _metrics(fused, gt)

        base_lambda_key = min(lambda_metrics, key=lambda lam: abs(lam - float(args.base_lambda)))
        sd_fixed = (
            {metric: _float(sd_dino, f"soft_{metric}") for metric in METRICS}
            if sd_dino.get("soft_mKLD")
            else lambda_metrics[base_lambda_key]
        )
        flux_fixed = {metric: _float(fx, f"soft_{metric}") for metric in METRICS}
        flux_ragate = {metric: _float(ad, f"soft_{metric}") for metric in METRICS}
        sd_metrics = {metric: _float(sd, metric) for metric in METRICS}
        flux_metrics = {metric: _float(fx, metric) for metric in METRICS}
        ranges_input.append(lambda_metrics)
        extras += [flux_fixed, flux_ragate, sd_metrics, flux_metrics]

        verb_stats = _heat_stats(sd_heat)
        object_stats = _heat_stats(sd_object)
        features = {
            "sd_verb_entropy": verb_stats["entropy"],
            "sd_verb_top5_mass": verb_stats["top5_mass"],
            "sd_verb_peak": verb_stats["peak"],
            "sd_object_entropy": object_stats["entropy"],
            "sd_object_top5_mass": object_stats["top5_mass"],
            "sd_object_area": object_stats["area"],
            "geometry_confidence": _float(ad, "geometry_confidence"),
            "alignment_confidence": _float(ad, "alignment_confidence"),
            "sd_geometry_alignment": _alignment(sd_heat, geom),
            "geom_similarity_best": _float(ad, "geom_similarity_best"),
            "roi_area_proxy": object_stats["area"],
        }
        if LEAKY_FEATURE_NAMES & set(features):
            raise RuntimeError("Feature set contains metric/GT leakage.")
        raw_records.append(
            {
                "key": key,
                "sd": sd,
                "fx": fx,
                "ad": ad,
                "features": features,
                "lambda_metrics": lambda_metrics,
                "sd_fixed": sd_fixed,
                "flux_fixed": flux_fixed,
                "flux_ragate": flux_ragate,
                "sd_metrics": sd_metrics,
                "flux_metrics": flux_metrics,
                "sd_seconds": _float(sd, "infer_seconds"),
                "used_sd_object": used_sd_object,
            }
        )

    ranges = _metric_ranges(ranges_input, extras)
    samples: list[Sample] = []
    for record in raw_records:
        oracle_lambda = _best_lambda(record["lambda_metrics"], ranges)
        sd_quality = _quality(record["sd_fixed"], ranges)
        flux_quality = _quality(record["flux_ragate"], ranges)
        route_label = 1.0 if flux_quality - sd_quality > args.route_margin else 0.0
        samples.append(
            Sample(
                key=record["key"],
                affordance=record["sd"]["affordance"],
                object_name=record["sd"]["object"],
                image=record["sd"]["image"],
                features=np.asarray([record["features"][name] for name in FEATURE_KEYS], dtype=np.float32),
                sd_metrics=record["sd_metrics"],
                flux_metrics=record["flux_metrics"],
                flux_fixed_metrics=record["flux_fixed"],
                flux_ragate_metrics=record["flux_ragate"],
                sd_fixed_metrics=record["sd_fixed"],
                lambda_metrics=record["lambda_metrics"],
                oracle_lambda=oracle_lambda,
                route_label=route_label,
                sd_seconds=record["sd_seconds"],
                used_sd_object=record["used_sd_object"],
            )
        )
    return samples


def run_cross_validation(args: argparse.Namespace, samples: list[Sample]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for fold in range(args.folds):
        train = [s for s in samples if _fold_for(s.key, args.folds) != fold]
        test = [s for s in samples if _fold_for(s.key, args.folds) == fold]
        if not train or not test:
            continue
        feature_mean = np.mean(np.stack([s.features for s in train], axis=0), axis=0)
        feature_std = np.std(np.stack([s.features for s in train], axis=0), axis=0)
        feature_std = np.where(feature_std < 1e-6, 1.0, feature_std)
        model = _train_model(
            train,
            feature_mean,
            feature_std,
            min_lambda=args.min_lambda,
            max_lambda=args.max_lambda,
            epochs=args.epochs,
            lr=args.lr,
            seed=args.seed + fold,
        )
        model.eval()
        for sample in test:
            x = torch.from_numpy(((sample.features - feature_mean) / feature_std).astype(np.float32))[None, :]
            with torch.no_grad():
                p_flux, pred_lambda = model(x, args.min_lambda, args.max_lambda)
            pred_lambda_value = float(pred_lambda.item())
            nearest_lambda = min(sample.lambda_metrics, key=lambda lam: abs(lam - pred_lambda_value))
            lambda_metrics = sample.lambda_metrics[nearest_lambda]
            use_flux = float(p_flux.item()) >= args.route_threshold
            route_metrics = sample.flux_ragate_metrics if use_flux else lambda_metrics
            row: dict[str, Any] = {
                "fold": fold,
                "key": sample.key,
                "affordance": sample.affordance,
                "object": sample.object_name,
                "image": sample.image,
                "p_flux": float(p_flux.item()),
                "use_flux": int(use_flux),
                "pred_lambda": pred_lambda_value,
                "nearest_lambda": nearest_lambda,
                "oracle_lambda": sample.oracle_lambda,
                "route_label": sample.route_label,
                "sd_seconds": sample.sd_seconds,
                "used_sd_object": int(sample.used_sd_object),
            }
            for metric in METRICS:
                row[f"flux_interaction_{metric}"] = sample.flux_metrics[metric]
                row[f"sd_interaction_{metric}"] = sample.sd_metrics[metric]
                row[f"flux_fixed_{metric}"] = sample.flux_fixed_metrics[metric]
                row[f"flux_ragate_{metric}"] = sample.flux_ragate_metrics[metric]
                row[f"sd_fixed_{metric}"] = sample.sd_fixed_metrics[metric]
                row[f"router_lambda_{metric}"] = lambda_metrics[metric]
                row[f"router_route_{metric}"] = route_metrics[metric]
            rows.append(row)

    n = len(rows)
    sd_seconds = float(np.mean([row["sd_seconds"] for row in rows])) if rows else 0.0
    flux_calls = int(sum(row["use_flux"] for row in rows))
    router_seconds = sd_seconds + args.dino_seconds + (flux_calls / max(1, n)) * args.flux_seconds
    methods = {
        "FLUX interaction": {**_aggregate(rows, "flux_interaction"), "seconds": args.flux_seconds, "flux_calls": n},
        "SD-Turbo interaction": {**_aggregate(rows, "sd_interaction"), "seconds": sd_seconds, "flux_calls": 0},
        "FLUX+DINO fixed": {**_aggregate(rows, "flux_fixed"), "seconds": args.flux_seconds + args.dino_seconds, "flux_calls": n},
        "FLUX+DINO RA-Gate": {**_aggregate(rows, "flux_ragate"), "seconds": args.flux_seconds + args.dino_seconds, "flux_calls": n},
        "SD+DINO fixed": {**_aggregate(rows, "sd_fixed"), "seconds": sd_seconds + args.dino_seconds, "flux_calls": 0},
        "SD+DINO TinyRouter(lambda)": {
            **_aggregate(rows, "router_lambda"),
            "seconds": sd_seconds + args.dino_seconds,
            "flux_calls": 0,
        },
        "SD+DINO TinyRouter(route+lambda)": {
            **_aggregate(rows, "router_route"),
            "seconds": router_seconds,
            "flux_calls": flux_calls,
        },
    }
    summary = {
        "samples": n,
        "folds": args.folds,
        "feature_keys": FEATURE_KEYS,
        "lambda_range": [args.min_lambda, args.max_lambda],
        "route_threshold": args.route_threshold,
        "route_margin": args.route_margin,
        "sd_object_attention_available": int(sum(row["used_sd_object"] for row in rows)),
        "methods": methods,
        "router_flux_call_rate": flux_calls / max(1, n),
        "router_speedup_vs_flux": (args.flux_seconds + args.dino_seconds) / max(router_seconds, 1e-8),
        "wtl_vs_flux_ragate": {
            metric: _wtl(rows, "router_route", "flux_ragate", metric)
            for metric in METRICS
        },
    }
    return rows, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train/evaluate a tiny low-compute reliability router.")
    parser.add_argument("--sd-summary", type=Path, required=True)
    parser.add_argument("--sd-dino-summary", type=Path, default=None)
    parser.add_argument("--flux-summary", type=Path, required=True)
    parser.add_argument("--adaptive-summary", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--prefix", default="tiny_router_balanced44")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--min-lambda", type=float, default=0.20)
    parser.add_argument("--max-lambda", type=float, default=0.90)
    parser.add_argument("--lambda-step", type=float, default=0.10)
    parser.add_argument("--base-lambda", type=float, default=0.65)
    parser.add_argument("--route-margin", type=float, default=0.05)
    parser.add_argument("--route-threshold", type=float, default=0.50)
    parser.add_argument("--soft-gamma", type=float, default=0.7)
    parser.add_argument("--soft-temperature", type=float, default=1.15)
    parser.add_argument("--soft-dirichlet", type=float, default=0.008)
    parser.add_argument("--flux-seconds", type=float, default=192.0)
    parser.add_argument("--dino-seconds", type=float, default=7.0)
    args = parser.parse_args()

    samples = build_samples(args)
    rows, summary = run_cross_validation(args, samples)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    per_sample_csv = args.out_dir / f"{args.prefix}_per_sample.csv"
    summary_json = args.out_dir / f"{args.prefix}.json"
    summary_md = args.out_dir / f"{args.prefix}.md"
    _write_csv(per_sample_csv, rows)
    summary["artifacts"] = {
        "per_sample_csv": str(per_sample_csv),
        "summary_json": str(summary_json),
        "summary_markdown": str(summary_md),
    }
    summary_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    _write_markdown(summary_md, summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
