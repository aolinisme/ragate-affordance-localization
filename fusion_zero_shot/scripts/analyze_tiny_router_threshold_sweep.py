#!/usr/bin/env python3
"""Analyze FLUX-call threshold trade-offs from TinyRouter per-sample output."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


METRICS = ("mKLD", "mSIM", "mNSS")


def _read(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _float(row: dict[str, Any], key: str) -> float:
    value = row.get(key, "")
    return float(value) if str(value).strip() else 0.0


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# TinyRouter Threshold Sweep",
        "",
        "| Threshold | FLUX calls | Sec./img | Speedup | KLD lower | SIM higher | NSS higher |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {threshold:.2f} | {flux_calls}/{samples} | {seconds_per_image:.4f} | {speedup_vs_flux:.2f}x | {mKLD:.4f} | {mSIM:.4f} | {mNSS:.4f} |".format(**row)
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-sample", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--prefix", default="tiny_router_threshold_sweep")
    parser.add_argument("--thresholds", nargs="+", type=float, default=[0.1, 0.3, 0.5, 0.7, 0.9])
    parser.add_argument("--flux-seconds", type=float, default=192.0)
    parser.add_argument("--dino-seconds", type=float, default=7.0)
    args = parser.parse_args()

    rows = _read(args.per_sample)
    n = len(rows)
    sd_seconds = float(np.mean([_float(row, "sd_seconds") for row in rows])) if rows else 0.0
    sweep: list[dict[str, Any]] = []
    for threshold in args.thresholds:
        metric_values = {metric: [] for metric in METRICS}
        flux_calls = 0
        for row in rows:
            use_flux = _float(row, "p_flux") >= threshold
            if use_flux:
                flux_calls += 1
            prefix = "flux_ragate" if use_flux else "router_lambda"
            for metric in METRICS:
                metric_values[metric].append(_float(row, f"{prefix}_{metric}"))
        seconds = sd_seconds + args.dino_seconds + (flux_calls / max(1, n)) * args.flux_seconds
        baseline_seconds = args.flux_seconds + args.dino_seconds
        sweep.append(
            {
                "threshold": float(threshold),
                "samples": n,
                "flux_calls": flux_calls,
                "flux_call_rate": flux_calls / max(1, n),
                "seconds_per_image": seconds,
                "speedup_vs_flux": baseline_seconds / max(seconds, 1e-8),
                **{metric: float(np.mean(metric_values[metric])) if rows else 0.0 for metric in METRICS},
            }
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / f"{args.prefix}.csv"
    md_path = args.out_dir / f"{args.prefix}.md"
    json_path = args.out_dir / f"{args.prefix}.json"
    _write_csv(csv_path, sweep)
    _write_markdown(md_path, sweep)
    json_path.write_text(json.dumps({"rows": sweep}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"rows": sweep, "artifacts": {"csv": str(csv_path), "markdown": str(md_path)}}, indent=2))


if __name__ == "__main__":
    main()
