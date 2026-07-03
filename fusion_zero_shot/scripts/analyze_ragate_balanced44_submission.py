#!/usr/bin/env python3
"""Prepare reviewer-facing balanced44 RA-Gate summary from cached CSV files."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def key_of(row: dict[str, str]) -> str:
    return "/".join([row["affordance"], row["object"], row["image"]])


def to_float(row: dict[str, str], key: str) -> float:
    return float(row[key])


def has_value(row: dict[str, str], key: str) -> bool:
    return key in row and row[key] not in ("", None)


def summarize_metric(values: list[float]) -> dict[str, float]:
    return {
        "mean": mean(values) if values else float("nan"),
        "min": min(values) if values else float("nan"),
        "max": max(values) if values else float("nan"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixed-summary", type=Path, required=True)
    parser.add_argument("--adaptive-summary", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--prefix", type=str, default="balanced44_ragate_submission")
    args = parser.parse_args()

    fixed_rows = read_csv(args.fixed_summary)
    adaptive_rows = read_csv(args.adaptive_summary)

    fixed = {key_of(r): r for r in fixed_rows}
    adaptive = {key_of(r): r for r in adaptive_rows}
    keys = sorted(set(fixed) & set(adaptive))

    paired = []
    per_aff = defaultdict(list)
    changed = 0
    fallback = 0

    for key in keys:
        frow = fixed[key]
        arow = adaptive[key]

        needed = [
            "soft_mKLD",
            "soft_mSIM",
            "soft_mNSS",
            "adaptive_lambda",
            "base_lambda",
            "interaction_confidence",
            "geometry_confidence",
            "alignment_confidence",
        ]
        if not all(has_value(frow, k) for k in ["soft_mKLD", "soft_mSIM", "soft_mNSS"]):
            continue
        if not all(has_value(arow, k) for k in needed):
            continue

        fixed_kld = to_float(frow, "soft_mKLD")
        fixed_sim = to_float(frow, "soft_mSIM")
        fixed_nss = to_float(frow, "soft_mNSS")
        adaptive_kld = to_float(arow, "soft_mKLD")
        adaptive_sim = to_float(arow, "soft_mSIM")
        adaptive_nss = to_float(arow, "soft_mNSS")
        lam = to_float(arow, "adaptive_lambda")
        base = to_float(arow, "base_lambda")

        if abs(lam - base) > 1e-9:
            changed += 1
        else:
            fallback += 1

        record = {
            "key": key,
            "affordance": arow["affordance"],
            "object": arow["object"],
            "fixed_kld": fixed_kld,
            "adaptive_kld": adaptive_kld,
            "delta_kld": fixed_kld - adaptive_kld,
            "fixed_sim": fixed_sim,
            "adaptive_sim": adaptive_sim,
            "delta_sim": adaptive_sim - fixed_sim,
            "fixed_nss": fixed_nss,
            "adaptive_nss": adaptive_nss,
            "delta_nss": adaptive_nss - fixed_nss,
            "adaptive_lambda": lam,
            "base_lambda": base,
            "interaction_confidence": to_float(arow, "interaction_confidence"),
            "geometry_confidence": to_float(arow, "geometry_confidence"),
            "alignment_confidence": to_float(arow, "alignment_confidence"),
        }
        paired.append(record)
        per_aff[arow["affordance"]].append(record)

    summary = {
        "num_paired": len(paired),
        "changed_samples": changed,
        "fallback_samples": fallback,
        "metric_summary": {
            "delta_kld": summarize_metric([r["delta_kld"] for r in paired]),
            "delta_sim": summarize_metric([r["delta_sim"] for r in paired]),
            "delta_nss": summarize_metric([r["delta_nss"] for r in paired]),
            "adaptive_lambda": summarize_metric([r["adaptive_lambda"] for r in paired]),
            "interaction_confidence": summarize_metric([r["interaction_confidence"] for r in paired]),
            "geometry_confidence": summarize_metric([r["geometry_confidence"] for r in paired]),
            "alignment_confidence": summarize_metric([r["alignment_confidence"] for r in paired]),
        },
        "per_affordance": {},
    }

    for affordance, rows in sorted(per_aff.items()):
        summary["per_affordance"][affordance] = {
            "count": len(rows),
            "delta_kld_mean": mean(r["delta_kld"] for r in rows),
            "delta_sim_mean": mean(r["delta_sim"] for r in rows),
            "delta_nss_mean": mean(r["delta_nss"] for r in rows),
            "changed": sum(1 for r in rows if abs(r["adaptive_lambda"] - r["base_lambda"]) > 1e-9),
        }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / f"{args.prefix}.json"
    csv_path = args.out_dir / f"{args.prefix}_per_sample.csv"
    md_path = args.out_dir / f"{args.prefix}.md"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    if paired:
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(paired[0].keys()))
            writer.writeheader()
            writer.writerows(paired)
    else:
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            f.write("")

    lines = [
        "# Balanced44 RA-Gate Submission Summary",
        "",
        f"- Paired samples: `{summary['num_paired']}`",
        f"- Changed samples: `{summary['changed_samples']}`",
        f"- Fallback samples: `{summary['fallback_samples']}`",
        "",
        "## Per-affordance mean deltas",
        "",
        "| Affordance | Count | Delta KLD | Delta SIM | Delta NSS | Changed |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for affordance, item in summary["per_affordance"].items():
        lines.append(
            f"| {affordance} | {item['count']} | {item['delta_kld_mean']:.4f} | "
            f"{item['delta_sim_mean']:.4f} | {item['delta_nss_mean']:.4f} | {item['changed']} |"
        )

    with md_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(json_path)
    print(csv_path)
    print(md_path)


if __name__ == "__main__":
    main()
