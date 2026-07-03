#!/usr/bin/env python3
"""Revision-facing statistics and simple gate baselines for balanced RA-Gate runs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np

try:
    from scipy.stats import wilcoxon
except Exception:  # pragma: no cover - fallback for minimal environments
    wilcoxon = None


METRICS = ("soft_mKLD", "soft_mSIM", "soft_mNSS")
METRIC_LABELS = {
    "soft_mKLD": "KLD",
    "soft_mSIM": "SIM",
    "soft_mNSS": "NSS",
}
HIGHER_BETTER = {
    "soft_mKLD": False,
    "soft_mSIM": True,
    "soft_mNSS": True,
}
KEY_FIELDS = ("affordance", "object", "image")


def _read_rows(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    out = {}
    for row in rows:
        key = "/".join(row.get(field, "") for field in KEY_FIELDS)
        out[key] = row
    return out


def _float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    try:
        value_f = float(text)
    except ValueError:
        return None
    if math.isnan(value_f) or math.isinf(value_f):
        return None
    return value_f


def _valid(row: dict[str, str], fields: tuple[str, ...] = METRICS) -> bool:
    return all(_float(row.get(field)) is not None for field in fields)


def _values(rows: list[dict[str, Any]], prefix: str, metric: str) -> np.ndarray:
    return np.asarray([float(row[f"{prefix}_{metric}"]) for row in rows], dtype=np.float64)


def _mean(values: np.ndarray) -> float:
    return float(values.mean()) if values.size else float("nan")


def _metric_delta(candidate: np.ndarray, fixed: np.ndarray, metric: str) -> np.ndarray:
    if HIGHER_BETTER[metric]:
        return candidate - fixed
    return fixed - candidate


def _bootstrap_ci(
    delta: np.ndarray,
    *,
    rng: np.random.Generator,
    iterations: int = 10000,
) -> tuple[float, float]:
    if delta.size == 0:
        return float("nan"), float("nan")
    indices = rng.integers(0, delta.size, size=(iterations, delta.size))
    samples = delta[indices].mean(axis=1)
    lo, hi = np.percentile(samples, [2.5, 97.5])
    return float(lo), float(hi)


def _sign_test_p(delta: np.ndarray, eps: float = 1e-9) -> float:
    nonzero = delta[np.abs(delta) > eps]
    n = int(nonzero.size)
    if n == 0:
        return 1.0
    wins = int((nonzero > 0).sum())
    k = min(wins, n - wins)
    # two-sided exact binomial with p=0.5
    prob = 0.0
    for i in range(k + 1):
        prob += math.comb(n, i) * (0.5**n)
    return float(min(1.0, 2.0 * prob))


def _wilcoxon_p(delta: np.ndarray, eps: float = 1e-9, alternative: str = "greater") -> float:
    nonzero = delta[np.abs(delta) > eps]
    if nonzero.size == 0:
        return 1.0
    if wilcoxon is None:
        return _sign_test_p(delta, eps=eps)
    try:
        return float(wilcoxon(nonzero, alternative=alternative, zero_method="wilcox").pvalue)
    except Exception:
        return _sign_test_p(delta, eps=eps)


def _wtl(delta: np.ndarray, eps: float = 1e-9) -> str:
    wins = int((delta > eps).sum())
    ties = int((np.abs(delta) <= eps).sum())
    losses = int((delta < -eps).sum())
    return f"{wins}/{ties}/{losses}"


def _quality(metrics: dict[str, float], ranges: dict[str, tuple[float, float]]) -> float:
    scores = []
    for metric in METRICS:
        lo, hi = ranges[metric]
        value = metrics[metric]
        denom = max(hi - lo, 1e-9)
        if HIGHER_BETTER[metric]:
            score = (value - lo) / denom
        else:
            score = (hi - value) / denom
        scores.append(float(np.clip(score, 0.0, 1.0)))
    return float(mean(scores))


def _aggregate_method(rows: list[dict[str, Any]], prefix: str) -> dict[str, float]:
    return {metric: _mean(_values(rows, prefix, metric)) for metric in METRICS}


def _fold_id(key: str, folds: int) -> int:
    # Stable deterministic fold without depending on Python's randomized hash.
    acc = 0
    for ch in key.encode("utf-8"):
        acc = (acc * 131 + ch) % 1000003
    return acc % folds


def _candidate_thresholds(values: list[float]) -> list[float]:
    clean = sorted({float(v) for v in values if v is not None and not math.isnan(float(v))})
    if not clean:
        return [0.0]
    candidates = [min(clean) - 1e-6, max(clean) + 1e-6]
    candidates.extend(clean)
    for a, b in zip(clean[:-1], clean[1:]):
        candidates.append((a + b) / 2.0)
    return sorted(set(candidates))


def _evaluate_selector(
    rows: list[dict[str, Any]],
    *,
    score_field: str,
    threshold: float,
) -> tuple[dict[str, float], int]:
    chosen = []
    adapted = 0
    for row in rows:
        use_wide = float(row[score_field]) >= threshold
        prefix = "wide" if use_wide else "fixed"
        adapted += int(use_wide)
        chosen.append({metric: float(row[f"{prefix}_{metric}"]) for metric in METRICS})
    metrics = {metric: float(mean(row[metric] for row in chosen)) for metric in METRICS}
    return metrics, adapted


def _cross_validated_selector(
    rows: list[dict[str, Any]],
    *,
    score_field: str,
    folds: int,
) -> dict[str, Any]:
    ranges = {}
    for metric in METRICS:
        vals = []
        for row in rows:
            vals.extend([float(row[f"fixed_{metric}"]), float(row[f"wide_{metric}"])])
        ranges[metric] = (min(vals), max(vals))

    thresholds = _candidate_thresholds([float(row[score_field]) for row in rows])
    predictions: dict[str, dict[str, float]] = {}
    selected_thresholds: list[float] = []

    for fold in range(folds):
        train = [row for row in rows if int(row["fold"]) != fold]
        test = [row for row in rows if int(row["fold"]) == fold]
        if not test:
            continue
        best_threshold = thresholds[0]
        best_quality = -1.0
        for threshold in thresholds:
            metrics, _ = _evaluate_selector(train, score_field=score_field, threshold=threshold)
            quality = _quality(metrics, ranges)
            if quality > best_quality:
                best_quality = quality
                best_threshold = threshold
        selected_thresholds.append(float(best_threshold))
        for row in test:
            use_wide = float(row[score_field]) >= best_threshold
            prefix = "wide" if use_wide else "fixed"
            predictions[row["key"]] = {
                "adapted": float(use_wide),
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
        "threshold_mean": float(mean(selected_thresholds)) if selected_thresholds else float("nan"),
        "thresholds": selected_thresholds,
        "predictions": predictions,
    }


def _method_stats(
    rows: list[dict[str, Any]],
    *,
    candidate_prefix: str,
    rng: np.random.Generator,
    bootstrap_iterations: int,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for metric in METRICS:
        fixed = _values(rows, "fixed", metric)
        candidate = _values(rows, candidate_prefix, metric)
        delta = _metric_delta(candidate, fixed, metric)
        ci_lo, ci_hi = _bootstrap_ci(delta, rng=rng, iterations=bootstrap_iterations)
        out[metric] = {
            "fixed_mean": _mean(fixed),
            "candidate_mean": _mean(candidate),
            "delta_mean_positive_is_better": _mean(delta),
            "bootstrap_ci95": [ci_lo, ci_hi],
            "wilcoxon_greater_p": _wilcoxon_p(delta, alternative="greater"),
            "wilcoxon_two_sided_p": _wilcoxon_p(delta, alternative="two-sided"),
            "sign_test_p": _sign_test_p(delta),
            "win_tie_loss": _wtl(delta),
        }
    return out


def _group_summary(rows: list[dict[str, Any]], group_field: str, candidate_prefix: str) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row.get(group_field, "")), []).append(row)

    out: list[dict[str, Any]] = []
    for group, group_rows in sorted(groups.items()):
        item: dict[str, Any] = {
            group_field: group,
            "count": len(group_rows),
            "adapted": sum(1 for row in group_rows if not bool(row.get("fallback_used", False))),
        }
        for metric in METRICS:
            fixed = _values(group_rows, "fixed", metric)
            candidate = _values(group_rows, candidate_prefix, metric)
            delta = _metric_delta(candidate, fixed, metric)
            item[f"fixed_{metric}"] = _mean(fixed)
            item[f"{candidate_prefix}_{metric}"] = _mean(candidate)
            item[f"delta_{metric}"] = _mean(delta)
            item[f"wtl_{metric}"] = _wtl(delta)
        out.append(item)
    return out


def _classify_failure(row: dict[str, Any], worst_metric: str) -> str:
    if float(row.get("interaction_confidence", 0.0)) < 0.45:
        return "diffuse_or_misplaced_interaction"
    if float(row.get("geometry_confidence", 0.0)) < 0.55:
        return "weak_geometry_primitive"
    if float(row.get("alignment_confidence", 0.0)) < 0.65 or float(row.get("similarity_best", 0.0)) < 0.90:
        return "cross_cue_alignment_boundary"
    if worst_metric == "soft_mNSS":
        return "metric_sensitive_sharpening"
    return "adaptive_weight_overfit_boundary"


def _failure_rows(rows: list[dict[str, Any]], candidate_prefix: str) -> list[dict[str, Any]]:
    failures = []
    for row in rows:
        deltas = {}
        for metric in METRICS:
            fixed = np.asarray([float(row[f"fixed_{metric}"])])
            candidate = np.asarray([float(row[f"{candidate_prefix}_{metric}"])])
            deltas[metric] = float(_metric_delta(candidate, fixed, metric)[0])
        if not any(value < -1e-9 for value in deltas.values()):
            continue
        worst_metric = min(deltas, key=deltas.get)
        failures.append(
            {
                "key": row["key"],
                "affordance": row["affordance"],
                "object": row["object"],
                "image": row["image"],
                "worst_metric": METRIC_LABELS[worst_metric],
                "failure_type": _classify_failure(row, worst_metric),
                "delta_kld_positive_better": deltas["soft_mKLD"],
                "delta_sim": deltas["soft_mSIM"],
                "delta_nss": deltas["soft_mNSS"],
                "adaptive_lambda": row["conservative_lambda"],
                "base_lambda": row["base_lambda"],
                "interaction_confidence": row["interaction_confidence"],
                "geometry_confidence": row["geometry_confidence"],
                "alignment_confidence": row["alignment_confidence"],
                "similarity_best": row["similarity_best"],
            }
        )
    return sorted(failures, key=lambda x: min(x["delta_kld_positive_better"], x["delta_sim"], x["delta_nss"]))


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(
    path: Path,
    *,
    summary: dict[str, Any],
    method_table: list[dict[str, Any]],
    stats_table: list[dict[str, Any]],
) -> None:
    lines = [
        "# RA-Gate Revision Statistics and Simple Baselines",
        "",
        "Evidence tier: balanced44 paired revision analysis using cached fixed, always-on adaptive, and conservative RA-Gate summaries.",
        "",
        f"- Valid paired samples: `{summary['valid_paired_samples']}`",
        f"- Folds for simple selector baselines: `{summary['folds']}`",
        f"- Bootstrap iterations: `{summary['bootstrap_iterations']}`",
        "",
        "## Method Means",
        "",
        "| Method | KLD lower | SIM higher | NSS higher | Adapted | Fallback |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in method_table:
        lines.append(
            f"| {row['method']} | {row['soft_mKLD']:.4f} | {row['soft_mSIM']:.4f} | "
            f"{row['soft_mNSS']:.4f} | {row['adapted']} | {row['fallback']} |"
        )

    lines.extend(
        [
            "",
            "## Paired Statistics Versus Fixed Fusion",
            "",
            "| Method | Metric | Delta mean | 95% bootstrap CI | Wilcoxon one-sided p | Wilcoxon two-sided p | W/T/L |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in stats_table:
        lines.append(
            f"| {row['method']} | {row['metric']} | {row['delta_mean']:.5f} | "
            f"[{row['ci_low']:.5f}, {row['ci_high']:.5f}] | {row['wilcoxon_p']:.4f} | "
            f"{row['wilcoxon_two_sided_p']:.4f} | "
            f"{row['win_tie_loss']} |"
        )

    lines.extend(
        [
            "",
            "Interpretation: positive delta means improvement over fixed fusion. "
            "KLD is internally reversed before computing deltas, so larger is better for every delta column. "
            "The one-signal selectors choose between fixed fusion and always-on adaptive fusion using only a single reliability score, "
            "with thresholds selected inside deterministic cross-validation folds.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixed-summary", type=Path, required=True)
    parser.add_argument("--conservative-summary", type=Path, required=True)
    parser.add_argument("--wide-summary", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--prefix", default="balanced44_revision_stats")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--bootstrap-iterations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260624)
    args = parser.parse_args()

    fixed_rows = _read_rows(args.fixed_summary)
    conservative_rows = _read_rows(args.conservative_summary)
    wide_rows = _read_rows(args.wide_summary)
    keys = sorted(set(fixed_rows) & set(conservative_rows) & set(wide_rows))

    rows: list[dict[str, Any]] = []
    for key in keys:
        fixed = fixed_rows[key]
        conservative = conservative_rows[key]
        wide = wide_rows[key]
        required_meta = ("interaction_confidence", "geometry_confidence", "alignment_confidence", "adaptive_lambda", "base_lambda")
        if not (_valid(fixed) and _valid(conservative) and _valid(wide) and _valid(wide, required_meta)):
            continue
        row: dict[str, Any] = {
            "key": key,
            "affordance": conservative.get("affordance", ""),
            "object": conservative.get("object", ""),
            "image": conservative.get("image", ""),
            "fold": _fold_id(key, args.folds),
            "interaction_confidence": float(_float(wide.get("interaction_confidence")) or 0.0),
            "geometry_confidence": float(_float(wide.get("geometry_confidence")) or 0.0),
            "alignment_confidence": float(_float(wide.get("alignment_confidence")) or 0.0),
            "similarity_best": float(_float(wide.get("geom_similarity_best")) or 0.0),
            "conservative_lambda": float(_float(conservative.get("adaptive_lambda")) or 0.0),
            "base_lambda": float(_float(conservative.get("base_lambda")) or 0.0),
        }
        row["fallback_used"] = abs(row["conservative_lambda"] - row["base_lambda"]) <= 1e-9
        row["max_confidence"] = max(row["interaction_confidence"], row["geometry_confidence"], row["alignment_confidence"])
        row["mean_confidence"] = mean(
            [row["interaction_confidence"], row["geometry_confidence"], row["alignment_confidence"]]
        )
        for prefix, source in (("fixed", fixed), ("conservative", conservative), ("wide", wide)):
            for metric in METRICS:
                row[f"{prefix}_{metric}"] = float(_float(source.get(metric)) or 0.0)
        rows.append(row)

    rng = np.random.default_rng(args.seed)
    method_rows: list[dict[str, Any]] = []
    stats_rows: list[dict[str, Any]] = []
    per_sample_rows: list[dict[str, Any]] = []
    method_predictions: dict[str, dict[str, dict[str, float]]] = {}

    fixed_metrics = _aggregate_method(rows, "fixed")
    method_rows.append({"method": "Fixed fusion", **fixed_metrics, "adapted": 0, "fallback": len(rows)})

    for label, prefix in (("Always-on adaptive", "wide"), ("RA-Gate conservative", "conservative")):
        metrics = _aggregate_method(rows, prefix)
        adapted = len(rows) if prefix == "wide" else sum(1 for row in rows if not row["fallback_used"])
        method_rows.append({"method": label, **metrics, "adapted": adapted, "fallback": len(rows) - adapted})
        stats = _method_stats(rows, candidate_prefix=prefix, rng=rng, bootstrap_iterations=args.bootstrap_iterations)
        for metric, item in stats.items():
            stats_rows.append(
                {
                    "method": label,
                    "metric": METRIC_LABELS[metric],
                    "delta_mean": item["delta_mean_positive_is_better"],
                    "ci_low": item["bootstrap_ci95"][0],
                    "ci_high": item["bootstrap_ci95"][1],
                    "wilcoxon_p": item["wilcoxon_greater_p"],
                    "wilcoxon_two_sided_p": item["wilcoxon_two_sided_p"],
                    "sign_test_p": item["sign_test_p"],
                    "win_tie_loss": item["win_tie_loss"],
                }
            )

    selector_specs = [
        ("Interaction-only selector", "interaction_confidence"),
        ("Geometry-only selector", "geometry_confidence"),
        ("Alignment-only selector", "alignment_confidence"),
        ("Similarity-only selector", "similarity_best"),
        ("Mean-confidence selector", "mean_confidence"),
        ("Max-confidence selector", "max_confidence"),
    ]
    selector_summaries = {}
    for label, score_field in selector_specs:
        cv = _cross_validated_selector(rows, score_field=score_field, folds=args.folds)
        selector_summaries[label] = {
            "score_field": score_field,
            "adapted": cv["adapted"],
            "fallback": cv["fallback"],
            "threshold_mean": cv["threshold_mean"],
            "thresholds": cv["thresholds"],
            "metrics": cv["metrics"],
        }
        method_rows.append(
            {
                "method": label,
                **cv["metrics"],
                "adapted": cv["adapted"],
                "fallback": cv["fallback"],
            }
        )
        method_predictions[label] = cv["predictions"]
        # Convert selector predictions to temporary rows for the common paired-stat function.
        tmp_rows = []
        for row in rows:
            pred = cv["predictions"][row["key"]]
            tmp = dict(row)
            for metric in METRICS:
                tmp[f"selector_{metric}"] = pred[metric]
            tmp_rows.append(tmp)
        stats = _method_stats(tmp_rows, candidate_prefix="selector", rng=rng, bootstrap_iterations=args.bootstrap_iterations)
        for metric, item in stats.items():
            stats_rows.append(
                {
                    "method": label,
                    "metric": METRIC_LABELS[metric],
                    "delta_mean": item["delta_mean_positive_is_better"],
                    "ci_low": item["bootstrap_ci95"][0],
                    "ci_high": item["bootstrap_ci95"][1],
                    "wilcoxon_p": item["wilcoxon_greater_p"],
                    "wilcoxon_two_sided_p": item["wilcoxon_two_sided_p"],
                    "sign_test_p": item["sign_test_p"],
                    "win_tie_loss": item["win_tie_loss"],
                }
            )

    for row in rows:
        out = dict(row)
        for label, preds in method_predictions.items():
            pred = preds.get(row["key"])
            if pred is None:
                continue
            safe = label.lower().replace("-", "_").replace(" ", "_")
            out[f"{safe}_adapted"] = int(pred["adapted"])
            out[f"{safe}_threshold"] = pred["threshold"]
            for metric in METRICS:
                out[f"{safe}_{metric}"] = pred[metric]
        per_sample_rows.append(out)

    summary = {
        "valid_paired_samples": len(rows),
        "folds": args.folds,
        "bootstrap_iterations": args.bootstrap_iterations,
        "seed": args.seed,
        "method_table": method_rows,
        "stats_table": stats_rows,
        "selector_summaries": selector_summaries,
        "per_affordance_conservative": _group_summary(rows, "affordance", "conservative"),
        "per_object_conservative": _group_summary(rows, "object", "conservative"),
        "failure_analysis": _failure_rows(rows, "conservative"),
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / f"{args.prefix}.json"
    method_csv = args.out_dir / f"{args.prefix}_methods.csv"
    stats_csv = args.out_dir / f"{args.prefix}_paired_stats.csv"
    per_sample_csv = args.out_dir / f"{args.prefix}_per_sample.csv"
    per_affordance_csv = args.out_dir / f"{args.prefix}_per_affordance.csv"
    per_object_csv = args.out_dir / f"{args.prefix}_per_object.csv"
    failure_csv = args.out_dir / f"{args.prefix}_failure_cases.csv"
    md_path = args.out_dir / f"{args.prefix}.md"

    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_csv(
        method_csv,
        method_rows,
        ["method", "soft_mKLD", "soft_mSIM", "soft_mNSS", "adapted", "fallback"],
    )
    _write_csv(
        stats_csv,
        stats_rows,
        [
            "method",
            "metric",
            "delta_mean",
            "ci_low",
            "ci_high",
            "wilcoxon_p",
            "wilcoxon_two_sided_p",
            "sign_test_p",
            "win_tie_loss",
        ],
    )
    _write_csv(per_sample_csv, per_sample_rows, list(per_sample_rows[0].keys()) if per_sample_rows else [])
    group_fields = [
        "count",
        "adapted",
        "fixed_soft_mKLD",
        "conservative_soft_mKLD",
        "delta_soft_mKLD",
        "wtl_soft_mKLD",
        "fixed_soft_mSIM",
        "conservative_soft_mSIM",
        "delta_soft_mSIM",
        "wtl_soft_mSIM",
        "fixed_soft_mNSS",
        "conservative_soft_mNSS",
        "delta_soft_mNSS",
        "wtl_soft_mNSS",
    ]
    _write_csv(per_affordance_csv, summary["per_affordance_conservative"], ["affordance", *group_fields])
    _write_csv(per_object_csv, summary["per_object_conservative"], ["object", *group_fields])
    _write_csv(
        failure_csv,
        summary["failure_analysis"],
        [
            "key",
            "affordance",
            "object",
            "image",
            "worst_metric",
            "failure_type",
            "delta_kld_positive_better",
            "delta_sim",
            "delta_nss",
            "adaptive_lambda",
            "base_lambda",
            "interaction_confidence",
            "geometry_confidence",
            "alignment_confidence",
            "similarity_best",
        ],
    )
    _write_markdown(md_path, summary=summary, method_table=method_rows, stats_table=stats_rows)

    print(json.dumps({
        "valid_paired_samples": len(rows),
        "json": str(json_path),
        "methods": str(method_csv),
        "stats": str(stats_csv),
        "per_affordance": str(per_affordance_csv),
        "failures": str(failure_csv),
        "markdown": str(md_path),
    }, indent=2))


if __name__ == "__main__":
    main()
