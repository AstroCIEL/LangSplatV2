"""End-to-end ablation on real scenes: placement x {full, HLS, HLS+SCAR}.

Each scene uses its own checkpoint and every test camera. Speedup is against
ASIC with HLS and SCAR off. PIM and ASIC-only share one projection per
algorithm setting.

    python -m simulator.run_ablation

``scripts/run_ablation.sh`` activates the langsplat_v2 environment first.
"""

import argparse
import json
import os
import time

import yaml

from simulator.export_csv import write_csv
from simulator.scenes import DATASETS
from simulator.simulator import Simulator
from simulator.workload import _PLAN_CACHE


MODES = (
    ("full", False, False),
    ("hls", True, False),
    ("hls_scar", True, True),
)
PLACEMENTS = ("asic", "pim")
BASELINE = "asic_full"


def _order():
    return [("lerf", name) for name in DATASETS["lerf"]] + [("3d_ovs", name) for name in DATASETS["3d_ovs"]]


def _load_cfg(template, dataset, scene, placement, hls, scar):
    with open(template, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["simulation"]["dataset"] = dataset
    cfg["simulation"]["scene"] = scene
    cfg["simulation"].pop("resolution", None)
    cfg["simulation"].pop("num_queries", None)
    cfg["simulation"]["camera_ids"] = "all"
    cfg["simulation"]["camera_split"] = "test"
    cfg["hardware"]["semantic"]["placement"] = placement
    cfg["algorithm"]["hls_enabled"] = hls
    cfg["algorithm"]["scar_enabled"] = scar
    cfg["workload"]["use_synthetic"] = False
    cfg["output"]["verbose"] = False
    cfg["output"]["stats_file"] = os.devnull
    for key in ("roi_fraction", "fallback_rate", "scar_skip_ratio"):
        cfg.get("algorithm", {}).pop(key, None)
    return cfg


def _per_frame_bytes(stats_row):
    n_frames = stats_row["notes"]["num_frames"]
    return {k: v / n_frames for k, v in stats_row["dram_bytes"].items()}


def _semantic_bytes(parts):
    return parts.get("feature_write", 0.0) + parts.get("feature_readback", 0.0) + parts.get("result_return", 0.0)


def run(template, summary_path):
    import torch

    order = _order()
    raw = {}
    for dataset, name in order:
        raw[name] = {"dataset": dataset}
        for mode, hls, scar in MODES:
            for placement in PLACEMENTS:
                key = f"{placement}_{mode}"
                cfg = _load_cfg(template, dataset, name, placement, hls, scar)
                print(f"=== {key} {name} ===", flush=True)
                t0 = time.time()
                sim = Simulator(template, config=cfg, dump_enabled=False)
                stats = sim.run()
                row = stats.to_dict()
                row.pop("config", None)
                raw[name][key] = row
                print(
                    f"done {name} {key} frames={row['notes']['num_frames']} "
                    f"res={row['notes']['resolution']} "
                    f"fps={row['fps']:.2f} load+sim={time.time() - t0:.1f}s",
                    flush=True,
                )
        _PLAN_CACHE.clear()
        torch.cuda.empty_cache()

    rows = []
    for _dataset, name in order:
        block = raw[name]
        base = block[BASELINE]
        base_fps = base["fps"]
        row = {
            "dataset": block["dataset"],
            "scene": name,
            "frames": base["notes"]["num_frames"],
            "resolution": base["notes"]["resolution"],
            "num_queries": base["notes"].get("num_queries", 1),
            "num_scene": base["notes"]["num_scene"],
            "num_visible_mean": base["notes"]["num_visible_mean"],
        }
        for mode, _hls, _scar in MODES:
            for placement in PLACEMENTS:
                key = f"{placement}_{mode}"
                stats = block[key]
                parts = _per_frame_bytes(stats)
                row[f"{key}_fps"] = stats["fps"]
                row[f"{key}_GB"] = sum(parts.values()) / 1e9
                row[f"{key}_semantic_GB"] = _semantic_bytes(parts) / 1e9
                row[f"{key}_speedup"] = (stats["fps"] / base_fps) if base_fps else 0.0
        rows.append(row)

    keys = [f"{placement}_{mode}" for mode, _h, _s in MODES for placement in PLACEMENTS]
    average = {k: "" for k in rows[0]}
    average["scene"] = "average"
    n = len(rows)
    for key in keys:
        average[f"{key}_fps"] = sum(r[f"{key}_fps"] for r in rows) / n
        average[f"{key}_GB"] = sum(r[f"{key}_GB"] for r in rows) / n
        average[f"{key}_semantic_GB"] = sum(r[f"{key}_semantic_GB"] for r in rows) / n
        average[f"{key}_speedup"] = sum(r[f"{key}_speedup"] for r in rows) / n
    rows.append(average)

    summary = {
        "baseline": BASELINE,
        "note": (
            "Real checkpoints, test split (every 8th view). "
            "Each scene uses its labeled-test-frame query count. "
            "Speedup is FPS relative to ASIC with HLS and SCAR off. "
            "Bytes are per-frame averages. bench uses RGB 3DGS."
        ),
        "scenes": rows,
    }
    folder = os.path.dirname(summary_path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    csv_path = os.path.splitext(summary_path)[0] + ".csv"
    write_csv(rows, csv_path)

    print("\nABLATION  fps and speedup vs ASIC full-frame")
    header = f"{'scene':12}" + "".join(f"{key:>16}" for key in keys)
    print(header)
    for row in rows:
        cells = []
        for key in keys:
            cells.append(f"{row[f'{key}_fps']:.2f}/{row[f'{key}_speedup']:.2f}x")
        print(f"{row['scene']:12}" + "".join(f"{c:>16}" for c in cells))
    print(f"wrote {summary_path}")
    print(f"wrote {csv_path}")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Real-scene ablation: PIM/ASIC x full/HLS/HLS+SCAR")
    parser.add_argument("--template", default="simulator/configs/fullframe_pim.yaml")
    parser.add_argument("--output", default="simulator/results/ablation_summary.json")
    args = parser.parse_args()
    run(args.template, args.output)


if __name__ == "__main__":
    main()
