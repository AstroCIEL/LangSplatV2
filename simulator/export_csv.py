"""Flatten a DRAM-traffic summary JSON into a CSV.

    python -m simulator.export_csv simulator/results/dram_traffic_summary.json

The default output path replaces the ``.json`` suffix with ``.csv``.
One row per scene. Nested ``*_bytes_per_frame`` objects become columns
prefixed by the parent key. A final ``average`` row carries the summary
fields from the top of the JSON.
"""

import argparse
import csv
import json
import os
from typing import List


_SUMMARY_TO_COLUMN = {
    "asic_avg_total_GB": "asic_total_GB",
    "pim_avg_total_GB": "pim_total_GB",
    "asic_avg_semantic_pct": "asic_semantic_pct",
    "avg_speedup": "speedup",
    "semantic_reduction_pct": "semantic_reduction_pct",
    "total_reduction_pct": "total_reduction_pct",
    "speedup_min": "speedup_min",
    "speedup_max": "speedup_max",
}


def flatten_summary(payload: dict) -> List[dict]:
    rows = []
    for scene in payload.get("scenes", []):
        row = {}
        for key, value in scene.items():
            if isinstance(value, dict):
                for sub, amount in value.items():
                    row[f"{key}.{sub}"] = amount
            else:
                row[key] = value
        rows.append(row)
    if not rows:
        return rows
    average = {key: "" for key in rows[0]}
    average["dataset"] = ""
    average["scene"] = "average"
    for src, dst in _SUMMARY_TO_COLUMN.items():
        if src in payload:
            average[dst] = payload[src]
    rows.append(average)
    return rows


def write_csv(rows: List[dict], path: str) -> None:
    columns: List[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                columns.append(key)
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in columns})


def json_to_csv(src: str, dst: str) -> str:
    with open(src, "r", encoding="utf-8") as f:
        payload = json.load(f)
    rows = flatten_summary(payload)
    write_csv(rows, dst)
    return dst


def main():
    parser = argparse.ArgumentParser(description="Convert a DRAM-traffic summary JSON to CSV")
    parser.add_argument("json_path", help="summary JSON, usually dram_traffic_summary.json")
    parser.add_argument(
        "-o",
        "--output",
        default="",
        help="CSV path. Default: same name as the JSON, with a .csv suffix",
    )
    args = parser.parse_args()
    dst = args.output
    if not dst:
        root, _ext = os.path.splitext(args.json_path)
        dst = root + ".csv"
    written = json_to_csv(args.json_path, dst)
    print(f"[export_csv] wrote {written}")


if __name__ == "__main__":
    main()
