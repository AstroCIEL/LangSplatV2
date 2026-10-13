"""Reproduce the DRAM-traffic comparison in the paper.

HLS and SCAR stay off. Each scene uses its own checkpoint and every test
camera (every 8th view). PIM and ASIC-only share one projection per scene.
Bytes in the summary are per-frame averages.

From the LangSplatV2 root, with the langsplat_v2 environment active:

    python -m simulator.run_dram_traffic

``scripts/run_dram_traffic.sh`` activates that environment and calls this module.
"""

import argparse
import json
import os
import time

import yaml

from simulator.export_csv import json_to_csv
from simulator.scenes import DATASETS
from simulator.simulator import Simulator
from simulator.workload import _PLAN_CACHE


def _order():
    return [("lerf", name) for name in DATASETS["lerf"]] + [("3d_ovs", name) for name in DATASETS["3d_ovs"]]


def _load_cfg(path, dataset, scene):
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["simulation"]["dataset"] = dataset
    cfg["simulation"]["scene"] = scene
    cfg["simulation"].pop("resolution", None)
    cfg["simulation"].pop("num_queries", None)
    cfg["simulation"]["camera_ids"] = "all"
    cfg["simulation"]["camera_split"] = "test"
    cfg["workload"]["use_synthetic"] = False
    cfg["output"]["verbose"] = False
    cfg["output"]["stats_file"] = os.devnull
    for key in ("roi_fraction", "fallback_rate", "scar_skip_ratio"):
        cfg.get("algorithm", {}).pop(key, None)
    return cfg


def _semantic_bytes(parts):
    return parts.get("feature_write", 0.0) + parts.get("feature_readback", 0.0) + parts.get("result_return", 0.0)


def run(pim_config, asic_config, summary_path):
    import torch

    base_paths = {"pim": pim_config, "asic": asic_config}
    order = _order()
    raw = {placement: {} for placement in base_paths}

    for dataset, name in order:
        for placement, path in base_paths.items():
            cfg = _load_cfg(path, dataset, name)
            print(f"=== {placement} {name} ===", flush=True)
            t0 = time.time()
            sim = Simulator(path, config=cfg, dump_enabled=False)
            stats = sim.run()
            row = stats.to_dict()
            row.pop("config", None)
            raw[placement][name] = row
            print(
                f"done {name} {placement} frames={row['notes']['num_frames']} "
                f"res={row['notes']['resolution']} "
                f"N={row['notes']['num_scene']:.0f} "
                f"visible={row['notes']['num_visible_mean']:.0f} "
                f"fps={row['fps']:.2f} load+sim={time.time() - t0:.1f}s",
                flush=True,
            )
        _PLAN_CACHE.clear()
        torch.cuda.empty_cache()

    rows = []
    for dataset, name in order:
        pim, asic = raw["pim"][name], raw["asic"][name]
        n_frames = pim["notes"]["num_frames"]
        pim_bytes = {k: v / n_frames for k, v in pim["dram_bytes"].items()}
        asic_bytes = {k: v / n_frames for k, v in asic["dram_bytes"].items()}
        pim_sem, asic_sem = _semantic_bytes(pim_bytes), _semantic_bytes(asic_bytes)
        pim_total, asic_total = sum(pim_bytes.values()), sum(asic_bytes.values())
        rows.append({
            "dataset": dataset,
            "scene": name,
            "frames": n_frames,
            "resolution": pim["notes"]["resolution"],
            "num_scene": pim["notes"]["num_scene"],
            "num_queries": pim["notes"].get("num_queries", 1),
            "num_visible_mean": pim["notes"]["num_visible_mean"],
            "pixels": pim["pixels"]["full"],
            "pim_fps": pim["fps"],
            "asic_fps": asic["fps"],
            "speedup": (pim["fps"] / asic["fps"]) if asic["fps"] else 0.0,
            "pim_total_GB": pim_total / 1e9,
            "asic_total_GB": asic_total / 1e9,
            "pim_semantic_GB": pim_sem / 1e9,
            "asic_semantic_GB": asic_sem / 1e9,
            "asic_semantic_pct": 100.0 * asic_sem / asic_total if asic_total else 0.0,
            "semantic_reduction_pct": 100.0 * (asic_sem - pim_sem) / asic_sem if asic_sem else 0.0,
            "total_reduction_pct": 100.0 * (asic_total - pim_total) / asic_total if asic_total else 0.0,
            "pim_bytes_per_frame": pim_bytes,
            "asic_bytes_per_frame": asic_bytes,
        })

    n = len(rows)
    def avg(key):
        return sum(row[key] for row in rows) / n

    speedups = [row["speedup"] for row in rows]
    summary = {
        "n_scenes": n,
        "note": (
            "HLS and SCAR off. Real checkpoints, test split (every 8th view). "
            "num_queries is the mean prompt count of the original labeled test frames. "
            "ASIC reloads the feature map once per query; PIM does not. "
            "dram bytes are per-frame averages. bench uses RGB 3DGS because the "
            "language point cloud is missing."
        ),
        "asic_avg_total_GB": avg("asic_total_GB"),
        "asic_avg_semantic_pct": avg("asic_semantic_pct"),
        "semantic_reduction_pct": avg("semantic_reduction_pct"),
        "total_reduction_pct": avg("total_reduction_pct"),
        "pim_avg_total_GB": avg("pim_total_GB"),
        "avg_speedup": avg("speedup"),
        "speedup_min": min(speedups),
        "speedup_max": max(speedups),
        "scenes": rows,
    }
    folder = os.path.dirname(summary_path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    csv_path = json_to_csv(summary_path, os.path.splitext(summary_path)[0] + ".csv")

    print("\nSCENE TABLE")
    print(f"{'scene':12} {'q':>6} {'frames':>6} {'res':>10} {'asicGB':>8} {'sem%':>7} {'semRed':>7} {'pimGB':>8} {'spd':>6}")
    for row in rows:
        q = row["num_queries"]
        qtxt = f"{q:.3f}" if isinstance(q, float) else ""
        print(
            f"{row['scene']:12} {qtxt:>6} {row['frames']:6.0f} {str(row['resolution']):>10} "
            f"{row['asic_total_GB']:8.3f} {row['asic_semantic_pct']:7.1f} "
            f"{row['semantic_reduction_pct']:7.1f} "
            f"{row['pim_total_GB']:8.3f} {row['speedup']:6.3f}"
        )
    print("--- averages ---")
    for key in (
        "asic_avg_total_GB",
        "asic_avg_semantic_pct",
        "semantic_reduction_pct",
        "total_reduction_pct",
        "pim_avg_total_GB",
        "avg_speedup",
        "speedup_min",
        "speedup_max",
    ):
        print(f"{key}: {summary[key]:.4f}")
    print(f"wrote {summary_path}")
    print(f"wrote {csv_path}")
    return summary


def main():
    parser = argparse.ArgumentParser(description="DRAM traffic: real scenes, HLS and SCAR off")
    parser.add_argument("--pim-config", default="simulator/configs/fullframe_pim.yaml")
    parser.add_argument("--asic-config", default="simulator/configs/fullframe_asic.yaml")
    parser.add_argument("--output", default="simulator/results/dram_traffic_summary.json")
    args = parser.parse_args()
    run(args.pim_config, args.asic_config, args.output)


if __name__ == "__main__":
    main()
