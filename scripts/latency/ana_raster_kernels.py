#!/usr/bin/env python3
"""Aggregate NCU per-kernel CSV into preprocess / sort / render ratios.

Compatible with OpenGaussian scripts/latency/ana_opengaussian.sh output
(analysis_per_kernel.csv), same categorization as 4DGaussians ana_latency.py.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from typing import Dict, Tuple


GPU_TIME_METRIC = "gpu__time_duration.sum"


def _unit_to_seconds_multiplier(unit: str) -> float:
    u = unit.strip().lower()
    if u == "s":
        return 1.0
    if u == "ms":
        return 1e-3
    if u == "us":
        return 1e-6
    if u == "ns":
        return 1e-9
    raise ValueError(f"Unsupported Metric Unit: {unit!r}")


def _kernel_category(kernel_name: str) -> str:
    if "renderCUDA" in kernel_name:
        return "render"
    if "preprocessCUDA" in kernel_name:
        return "preprocess"
    return "sort"


def parse_csv(csv_path: str) -> Dict[str, float]:
    stage_sums = {"preprocess": 0.0, "render": 0.0, "sort": 0.0}
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if (row.get("Metric Name") or "").strip() != GPU_TIME_METRIC:
                continue
            kernel = (row.get("Kernel Name") or "").strip()
            unit = (row.get("Metric Unit") or "").strip()
            avg_str = (row.get("Average") or "").strip()
            if not kernel or not unit or not avg_str:
                continue
            try:
                avg = float(avg_str)
            except ValueError:
                continue
            stage_sums[_kernel_category(kernel)] += avg * _unit_to_seconds_multiplier(unit)
    return stage_sums


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True, help="analysis_per_kernel.csv from ncu")
    p.add_argument("--out_txt", required=True)
    p.add_argument("--out_json", required=True)
    args = p.parse_args()

    stages = parse_csv(args.csv)
    total = sum(stages.values()) or 1e-12
    ratios = {k: v / total for k, v in stages.items()}

    lines = [
        "Raster kernel GPU-time breakdown (from NCU gpu__time_duration.sum)",
        f"source: {os.path.abspath(args.csv)}",
        "-" * 64,
        f"{'stage':<16} {'gpu_s':>12} {'ratio':>10}",
        "-" * 64,
    ]
    for k in ("preprocess", "sort", "render"):
        lines.append(f"{k:<16} {stages[k]:12.6f} {ratios[k] * 100:9.2f}%")
    lines.append("-" * 64)
    lines.append(f"{'TOTAL':<16} {total:12.6f} {100.0:9.2f}%")
    lines.append("")
    lines.append("Notes:")
    lines.append("  - preprocess: preprocessCUDA")
    lines.append("  - render:     renderCUDA")
    lines.append("  - sort:       CUB scan/radix + duplicateWithKeys/identifyTileRanges")
    lines.append("  - Apply ratios to wall-clock render.raster")
    lines.append("    from latency_breakdown.json for absolute stage seconds.")

    text = "\n".join(lines) + "\n"
    os.makedirs(os.path.dirname(args.out_txt) or ".", exist_ok=True)
    with open(args.out_txt, "w", encoding="utf-8") as f:
        f.write(text)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump({"stages_s": stages, "ratios": ratios, "total_s": total}, f, indent=2)
    print(text)
    print(f"Wrote {args.out_txt}")
    print(f"Wrote {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
