#!/usr/bin/env python3
"""Analyze a similarity-floor sweep from fixed and always-on adaptive summaries."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


METRICS = ("soft_mKLD", "soft_mSIM", "soft_mNSS")


def _read_rows(path: Path) -> dict[str, dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = "/".join([row.get("affordance", ""), row.get("object", ""), row.get("image", "")])
        out[key] = row
    return out


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


def _is_valid_metric_row(row: dict[str, Any]) -> bool:
    return all(_float_or_none(row.get(metric)) is not None for metric in METRICS)


def _better(metric: str, candidate: float, fixed: float, eps: float) -> int:
    if abs(candidate - fixed) <= eps:
        return 0
    if metric == "soft_mKLD":
        return 1 if candidate < fixed else -1
    return 1 if candidate > fixed else -1


def _wtl(metric: str, candidate_values: list[float], fixed_values: list[float], eps: float) -> tuple[int, int, int]:
    wins = ties = losses = 0
    for candidate, fixed in zip(candidate_values, fixed_values):
        flag = _better(metric, candidate, fixed, eps)
        if flag > 0:
            wins += 1
        elif flag < 0:
            losses += 1
        else:
            ties += 1
    return wins, ties, losses


def _mean(values: list[float]) -> float:
    return sum(values) / max(1, len(values))


def _format_wtl(wtl: tuple[int, int, int]) -> str:
    return f"{wtl[0]}/{wtl[1]}/{wtl[2]}"


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path: Path, title: str, rows: list[dict[str, Any]], valid_n: int) -> None:
    lines = [
        f"# {title}",
        "",
        "Evidence tier: capped local analytical reuse of completed fixed and always-on adaptive runs.",
        "The rule is identical to the conservative gate: use always-on adaptive fusion when "
        "`selected_similarity >= floor`; otherwise use fixed fusion.",
        "",
        f"Valid paired samples: `{valid_n}`.",
        "",
        "| Floor | Adapted | Fallback | KLD lower | SIM higher | NSS higher | KLD W/T/L | SIM W/T/L | NSS W/T/L |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {threshold:.2f} | {adapted} | {fallback} | {soft_mKLD:.4f} | "
            "{soft_mSIM:.4f} | {soft_mNSS:.4f} | {soft_mKLD_win_tie_loss} | "
            "{soft_mSIM_win_tie_loss} | {soft_mNSS_win_tie_loss} |".format(**row)
        )
    lines.extend(
        [
            "",
            "Interpretation: lower floors behave like always-on adaptive fusion and adapt more images. "
            "Higher floors fallback more often, reducing the number of changed predictions. "
            "This sweep should be read as a reliability/safety trade-off, not as a full benchmark result.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_plot(path: Path, rows: list[dict[str, Any]], fixed_metrics: dict[str, float]) -> None:
    import matplotlib.pyplot as plt

    labels = ["on" if float(row["threshold"]) < 0 else f"{float(row['threshold']):g}" for row in rows]
    x = list(range(len(rows)))
    adapted = [int(row["adapted"]) for row in rows]
    kld_gain = [
        (fixed_metrics["soft_mKLD"] - float(row["soft_mKLD"])) / fixed_metrics["soft_mKLD"] * 100.0
        for row in rows
    ]
    sim_gain = [
        (float(row["soft_mSIM"]) - fixed_metrics["soft_mSIM"]) / fixed_metrics["soft_mSIM"] * 100.0
        for row in rows
    ]
    nss_gain = [
        (float(row["soft_mNSS"]) - fixed_metrics["soft_mNSS"]) / fixed_metrics["soft_mNSS"] * 100.0
        for row in rows
    ]

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.1), dpi=180)
    ax = axes[0]
    ax.plot(x, kld_gain, marker="o", label="KLD gain")
    ax.plot(x, sim_gain, marker="s", label="SIM gain")
    ax.plot(x, nss_gain, marker="^", label="NSS gain")
    ax.axhline(0.0, color="#5b5b5b", linewidth=0.9, linestyle="--", alpha=0.55)
    if "0.9" in labels:
        ax.axvline(labels.index("0.9"), color="#d08b00", linewidth=0.9, linestyle=":", alpha=0.8)
    ax.set_xlabel("Similarity floor")
    ax.set_ylabel("Gain vs fixed (%)")
    ax.set_title("Metric gain trade-off")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1]
    ax.bar(x, adapted, color="#4c78a8")
    if "0.9" in labels:
        ax.axvline(labels.index("0.9"), color="#d08b00", linewidth=0.9, linestyle=":", alpha=0.8)
    ax.set_xlabel("Similarity floor")
    ax.set_ylabel("# adapted samples")
    ax.set_title("Gate selectivity")
    ax.grid(axis="y", alpha=0.25)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45)

    fig.tight_layout()
    fig.savefig(path)
    if path.suffix.lower() == ".pdf":
        fig.savefig(path.with_suffix(".png"))
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixed-summary", type=Path, required=True)
    parser.add_argument("--adaptive-summary", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--prefix", default="similarity_floor_sweep")
    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        default=[0.0, 0.5, 0.7, 0.8, 0.85, 0.9, 1.0, 1.1, 1.5, 2.0],
    )
    parser.add_argument("--eps", type=float, default=1e-9)
    args = parser.parse_args()

    fixed_rows = _read_rows(args.fixed_summary)
    adaptive_rows = _read_rows(args.adaptive_summary)

    paired_keys = sorted(set(fixed_rows) & set(adaptive_rows))
    valid_keys = [
        key
        for key in paired_keys
        if _is_valid_metric_row(fixed_rows[key])
        and _is_valid_metric_row(adaptive_rows[key])
        and _float_or_none(adaptive_rows[key].get("geom_similarity_best")) is not None
    ]
    excluded = [key for key in paired_keys if key not in valid_keys]

    sweep_rows: list[dict[str, Any]] = []
    per_sample_rows: list[dict[str, Any]] = []
    for threshold in args.thresholds:
        metric_values: dict[str, list[float]] = {metric: [] for metric in METRICS}
        fixed_values: dict[str, list[float]] = {metric: [] for metric in METRICS}
        adapted = 0
        for key in valid_keys:
            fixed = fixed_rows[key]
            adaptive = adaptive_rows[key]
            selected_similarity = float(_float_or_none(adaptive.get("geom_similarity_best")) or 0.0)
            use_adaptive = selected_similarity >= threshold
            chosen = adaptive if use_adaptive else fixed
            if use_adaptive:
                adapted += 1
            per_sample = {
                "threshold": threshold,
                "key": key,
                "selected_similarity": selected_similarity,
                "use_adaptive": use_adaptive,
                "adaptive_lambda": _float_or_none(adaptive.get("adaptive_lambda")),
            }
            for metric in METRICS:
                value = float(_float_or_none(chosen.get(metric)) or 0.0)
                fixed_value = float(_float_or_none(fixed.get(metric)) or 0.0)
                metric_values[metric].append(value)
                fixed_values[metric].append(fixed_value)
                per_sample[metric] = value
                per_sample[f"fixed_{metric}"] = fixed_value
                per_sample[f"delta_{metric}"] = value - fixed_value
            per_sample_rows.append(per_sample)

        summary: dict[str, Any] = {
            "threshold": threshold,
            "adapted": adapted,
            "fallback": len(valid_keys) - adapted,
        }
        for metric in METRICS:
            summary[metric] = _mean(metric_values[metric])
            summary[f"{metric}_win_tie_loss"] = _format_wtl(
                _wtl(metric, metric_values[metric], fixed_values[metric], args.eps)
            )
        sweep_rows.append(summary)

    fixed_metric_means = {
        metric: _mean([float(_float_or_none(fixed_rows[key].get(metric)) or 0.0) for key in valid_keys])
        for metric in METRICS
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / f"{args.prefix}.csv"
    per_sample_path = args.out_dir / f"{args.prefix}_per_sample.csv"
    md_path = args.out_dir / f"{args.prefix}.md"
    json_path = args.out_dir / f"{args.prefix}.json"
    plot_path = args.out_dir / f"{args.prefix}.pdf"

    _write_csv(
        csv_path,
        sweep_rows,
        [
            "threshold",
            "adapted",
            "fallback",
            "soft_mKLD",
            "soft_mSIM",
            "soft_mNSS",
            "soft_mKLD_win_tie_loss",
            "soft_mSIM_win_tie_loss",
            "soft_mNSS_win_tie_loss",
        ],
    )
    _write_csv(
        per_sample_path,
        per_sample_rows,
        [
            "threshold",
            "key",
            "selected_similarity",
            "use_adaptive",
            "adaptive_lambda",
            "soft_mKLD",
            "fixed_soft_mKLD",
            "delta_soft_mKLD",
            "soft_mSIM",
            "fixed_soft_mSIM",
            "delta_soft_mSIM",
            "soft_mNSS",
            "fixed_soft_mNSS",
            "delta_soft_mNSS",
        ],
    )
    _write_markdown(md_path, args.prefix.replace("_", " ").title(), sweep_rows, len(valid_keys))
    _write_plot(plot_path, sweep_rows, fixed_metric_means)
    json_path.write_text(
        json.dumps(
            {
                "valid_paired_samples": len(valid_keys),
                "excluded": excluded,
                "fixed_metric_means": fixed_metric_means,
                "rows": sweep_rows,
                "artifacts": {
                    "csv": str(csv_path),
                    "per_sample_csv": str(per_sample_path),
                    "markdown": str(md_path),
                    "plot_pdf": str(plot_path),
                    "plot_png": str(plot_path.with_suffix(".png")),
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(json.dumps({"valid": len(valid_keys), "excluded": excluded, "csv": str(csv_path)}, indent=2))


if __name__ == "__main__":
    main()
