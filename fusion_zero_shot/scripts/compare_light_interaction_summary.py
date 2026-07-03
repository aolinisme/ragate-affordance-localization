#!/usr/bin/env python3
"""Compare lightweight interaction summaries against FLUX interaction rows."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


METRICS = ("mKLD", "mSIM", "mNSS")


def _read(path: Path) -> dict[str, dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = csv.DictReader(f)
        return {
            "/".join([row["affordance"], row["object"], row["image"]]): row
            for row in rows
            if row.get("affordance") and row.get("object") and row.get("image")
        }


def _float(row: dict[str, Any], key: str) -> float:
    value = row.get(key, "")
    return float(value) if str(value).strip() else 0.0


def _mean(values: list[float]) -> float:
    return sum(values) / max(1, len(values))


def _better(metric: str, candidate: float, base: float, eps: float) -> int:
    if abs(candidate - base) <= eps:
        return 0
    if metric.endswith("KLD"):
        return 1 if candidate < base else -1
    return 1 if candidate > base else -1


def _wtl(metric: str, candidate: list[float], base: list[float], eps: float) -> tuple[int, int, int]:
    wins = ties = losses = 0
    for c, b in zip(candidate, base):
        flag = _better(metric, c, b, eps)
        if flag > 0:
            wins += 1
        elif flag < 0:
            losses += 1
        else:
            ties += 1
    return wins, ties, losses


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
        "# Lightweight Interaction Comparison",
        "",
        "Evidence tier: capped local paired comparison. The lightweight branch is SD-Turbo verb attention; the reference branch is FLUX interaction attention from the completed balanced44 run.",
        "",
        f"Valid paired samples: `{summary['paired_samples']}`.",
        "",
        "| Branch | KLD lower | SIM higher | NSS higher | Mean seconds/image |",
        "|---|---:|---:|---:|---:|",
        "| SD-Turbo attention | {sd_mKLD:.4f} | {sd_mSIM:.4f} | {sd_mNSS:.4f} | {sd_seconds:.4f} |".format(**summary),
        "| FLUX interaction | {flux_mKLD:.4f} | {flux_mSIM:.4f} | {flux_mNSS:.4f} | {flux_seconds:.4f} |".format(**summary),
        "",
        "| Metric | SD-Turbo W/T/L vs FLUX | Relative change |",
        "|---|---:|---:|",
        "| KLD | {mKLD_wtl} | {mKLD_relative_change_percent:+.2f}% |".format(**summary),
        "| SIM | {mSIM_wtl} | {mSIM_relative_change_percent:+.2f}% |".format(**summary),
        "| NSS | {mNSS_wtl} | {mNSS_relative_change_percent:+.2f}% |".format(**summary),
        "",
        "Interpretation: SD-Turbo is not a drop-in replacement for FLUX Kontext because this pilot uses prompt-generation attention rather than input-image editing attention. Its value is speed: it provides a seconds/sub-second interaction prior that can be used as a student or fast fallback branch, with a reliability gate deciding when full FLUX is still needed.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--light-summary", type=Path, required=True)
    parser.add_argument("--flux-summary", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--prefix", default="light_vs_flux_interaction")
    parser.add_argument("--flux-seconds", type=float, default=192.0)
    parser.add_argument("--eps", type=float, default=1e-9)
    args = parser.parse_args()

    light = _read(args.light_summary)
    flux = _read(args.flux_summary)
    keys = sorted(set(light) & set(flux))

    per_sample: list[dict[str, Any]] = []
    light_values: dict[str, list[float]] = {metric: [] for metric in METRICS}
    flux_values: dict[str, list[float]] = {metric: [] for metric in METRICS}
    light_seconds: list[float] = []

    for key in keys:
        lrow = light[key]
        frow = flux[key]
        row: dict[str, Any] = {"key": key}
        for metric in METRICS:
            lv = _float(lrow, metric)
            fv = _float(frow, metric)
            light_values[metric].append(lv)
            flux_values[metric].append(fv)
            row[f"light_{metric}"] = lv
            row[f"flux_{metric}"] = fv
            row[f"delta_{metric}"] = lv - fv
        sec = _float(lrow, "infer_seconds")
        light_seconds.append(sec)
        row["light_infer_seconds"] = sec
        per_sample.append(row)

    summary: dict[str, Any] = {
        "paired_samples": len(keys),
        "sd_seconds": _mean(light_seconds),
        "flux_seconds": args.flux_seconds,
    }
    if summary["sd_seconds"] > 0:
        summary["estimated_speedup_vs_flux"] = args.flux_seconds / summary["sd_seconds"]
    else:
        summary["estimated_speedup_vs_flux"] = 0.0

    for metric in METRICS:
        light_mean = _mean(light_values[metric])
        flux_mean = _mean(flux_values[metric])
        summary[f"sd_{metric}"] = light_mean
        summary[f"flux_{metric}"] = flux_mean
        if metric.endswith("KLD"):
            change = (flux_mean - light_mean) / flux_mean * 100.0 if flux_mean else 0.0
        else:
            change = (light_mean - flux_mean) / abs(flux_mean) * 100.0 if flux_mean else 0.0
        summary[f"{metric}_relative_change_percent"] = change
        summary[f"{metric}_wtl"] = "/".join(
            str(x) for x in _wtl(metric, light_values[metric], flux_values[metric], args.eps)
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / f"{args.prefix}_per_sample.csv"
    json_path = args.out_dir / f"{args.prefix}.json"
    md_path = args.out_dir / f"{args.prefix}.md"
    _write_csv(csv_path, per_sample)
    _write_markdown(md_path, summary)
    json_path.write_text(
        json.dumps({"summary": summary, "artifacts": {"per_sample_csv": str(csv_path), "markdown": str(md_path)}}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
