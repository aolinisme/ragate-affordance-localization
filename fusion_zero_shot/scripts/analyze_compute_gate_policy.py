#!/usr/bin/env python3
"""Analyze FLUX-call saving policies from completed fusion summaries.

This is an offline pilot utility. It does not train a deployable gate; instead it
asks how much room there is for a future compute-aware gate by replacing the
full FLUX interaction branch with the currently available geometry-only branch
on selected samples.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Callable


FULL_METRICS = ("soft_mKLD", "soft_mSIM", "soft_mNSS")
GEOM_METRICS = ("geom_mKLD", "geom_mSIM", "geom_mNSS")


def _read_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _valid(row: dict[str, Any]) -> bool:
    return all(_float_or_none(row.get(k)) is not None for k in FULL_METRICS + GEOM_METRICS)


def _mean(values: list[float]) -> float:
    return sum(values) / max(1, len(values))


def _metric_gain(metric: str, candidate: float, base: float) -> float:
    if base == 0:
        return 0.0
    if metric.endswith("KLD"):
        return (base - candidate) / base * 100.0
    return (candidate - base) / abs(base) * 100.0


def _better_or_equal(metric: str, candidate: float, base: float, tolerance: float = 0.0) -> bool:
    if metric.endswith("KLD"):
        return candidate <= base * (1.0 + tolerance)
    return candidate >= base * (1.0 - tolerance)


def _policy_summary(
    rows: list[dict[str, Any]],
    name: str,
    should_skip: Callable[[dict[str, Any]], bool],
    full_flux_seconds: float,
    cheap_seconds: float,
) -> dict[str, Any]:
    full_values: dict[str, list[float]] = {metric: [] for metric in FULL_METRICS}
    chosen_values: dict[str, list[float]] = {metric: [] for metric in FULL_METRICS}
    skip_count = 0

    for row in rows:
        skip = should_skip(row)
        if skip:
            skip_count += 1
        for full_key, geom_key in zip(FULL_METRICS, GEOM_METRICS):
            full = float(_float_or_none(row[full_key]) or 0.0)
            geom = float(_float_or_none(row[geom_key]) or 0.0)
            full_values[full_key].append(full)
            chosen_values[full_key].append(geom if skip else full)

    n = len(rows)
    baseline_seconds = n * full_flux_seconds
    policy_seconds = (n - skip_count) * full_flux_seconds + n * cheap_seconds
    if baseline_seconds > 0:
        speedup = baseline_seconds / policy_seconds
        saved_percent = (baseline_seconds - policy_seconds) / baseline_seconds * 100.0
    else:
        speedup = 1.0
        saved_percent = 0.0

    out: dict[str, Any] = {
        "policy": name,
        "samples": n,
        "skip_full_flux": skip_count,
        "run_full_flux": n - skip_count,
        "skip_percent": skip_count / max(1, n) * 100.0,
        "estimated_seconds": policy_seconds,
        "estimated_speedup_vs_full_flux": speedup,
        "estimated_time_saved_percent": saved_percent,
    }
    for metric in FULL_METRICS:
        base = _mean(full_values[metric])
        chosen = _mean(chosen_values[metric])
        out[metric] = chosen
        out[f"baseline_{metric}"] = base
        out[f"{metric}_gain_percent_vs_full"] = _metric_gain(metric, chosen, base)
    return out


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Compute-Aware Gate Offline Pilot",
        "",
        "Evidence tier: capped local offline analysis of completed summaries.",
        "",
        "The analysis asks whether full FLUX can be skipped on some samples and replaced by the currently available geometry-only branch.",
        "It is not a deployable trained gate yet, because the present summaries do not contain enough FLUX-free pre-call features to train a reliable predictor.",
        "",
        "| Policy | Skip full FLUX | Time saved | Speedup | KLD lower | SIM higher | NSS higher |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {policy} | {skip_full_flux}/{samples} ({skip_percent:.1f}%) | "
            "{estimated_time_saved_percent:.1f}% | {estimated_speedup_vs_full_flux:.2f}x | "
            "{soft_mKLD:.4f} ({soft_mKLD_gain_percent_vs_full:+.1f}%) | "
            "{soft_mSIM:.4f} ({soft_mSIM_gain_percent_vs_full:+.1f}%) | "
            "{soft_mNSS:.4f} ({soft_mNSS_gain_percent_vs_full:+.1f}%) |".format(**row)
        )
    lines.extend(
        [
            "",
            "Interpretation: a safe skip policy has little room when geometry-only maps are used as the replacement.",
            "A meaningful deployment route should therefore use a lightweight interaction student as the replacement branch, then train a small pre-call gate to decide between the student and full FLUX.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--prefix", default="compute_gate_policy")
    parser.add_argument("--full-flux-seconds", type=float, default=192.0)
    parser.add_argument("--cheap-seconds", type=float, default=7.0)
    parser.add_argument("--tolerance", type=float, default=0.0)
    args = parser.parse_args()

    rows = [row for row in _read_rows(args.summary) if _valid(row)]

    def safe_skip(row: dict[str, Any]) -> bool:
        return all(
            _better_or_equal(full_key, float(row[geom_key]), float(row[full_key]), args.tolerance)
            for full_key, geom_key in zip(FULL_METRICS, GEOM_METRICS)
        )

    def nss_skip(row: dict[str, Any]) -> bool:
        return _better_or_equal("soft_mNSS", float(row["geom_mNSS"]), float(row["soft_mNSS"]), args.tolerance)

    def sim_nss_skip(row: dict[str, Any]) -> bool:
        return _better_or_equal("soft_mSIM", float(row["geom_mSIM"]), float(row["soft_mSIM"]), args.tolerance) and _better_or_equal(
            "soft_mNSS", float(row["geom_mNSS"]), float(row["soft_mNSS"]), args.tolerance
        )

    policies = [
        ("full_flux_for_all", lambda row: False),
        ("oracle_safe_skip_all_metrics", safe_skip),
        ("oracle_skip_if_sim_and_nss_nonworse", sim_nss_skip),
        ("oracle_skip_if_nss_nonworse", nss_skip),
    ]
    summaries = [
        _policy_summary(rows, name, policy, args.full_flux_seconds, args.cheap_seconds)
        for name, policy in policies
    ]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / f"{args.prefix}.csv"
    md_path = args.out_dir / f"{args.prefix}.md"
    json_path = args.out_dir / f"{args.prefix}.json"
    _write_csv(csv_path, summaries)
    _write_markdown(md_path, summaries)
    json_path.write_text(
        json.dumps(
            {
                "summary": str(args.summary),
                "valid_samples": len(rows),
                "full_flux_seconds_per_image": args.full_flux_seconds,
                "cheap_seconds_per_image": args.cheap_seconds,
                "tolerance": args.tolerance,
                "rows": summaries,
                "artifacts": {
                    "csv": str(csv_path),
                    "markdown": str(md_path),
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"valid": len(rows), "csv": str(csv_path), "markdown": str(md_path)}, indent=2))


if __name__ == "__main__":
    main()
