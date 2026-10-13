"""Wire a yaml config into one simpy run.

FPS is single-view latency: frames run back to back and do not overlap.
Inside a view, tiles overlap across sort, raster, decode, and the feature
write. Norm and matching wait until that feature map is in DRAM. An
ASIC-side SPU then reloads every 512-d vector; a PIM SPU does not.
Stage busy times are unions, so their sum can exceed ``total_cycles``.
"""

import copy
import json
import math
import os
from datetime import datetime

import simpy
import yaml

from simulator.analyzer import Analyzer
from simulator.engines import (
    FTGConfig,
    FineTaskGenerator,
    GFPConfig,
    GaussianFrontEnd,
    HSEConfig,
    HierarchicalSort,
    SDEConfig,
    SemanticArray,
    SemanticDecoder,
    UREConfig,
    UnifiedRaster,
)
from simulator.memory import DramConfig, DramSystem, PimConfig, language_payload_bytes
from simulator.scenes import DATASETS, SCENES, canonical_dataset
from simulator.structures import SimStats
from simulator.pipeline import Pipeline
from simulator.workload import build_plans, resolve_scene


class Simulator:
    def __init__(self, config_path: str, config: dict = None, dump_enabled: bool = True):
        self.config_path = config_path
        self.config = resolve_scene(config if config is not None else _load(config_path))
        out = self.config.get("output", {})
        self.stats = SimStats(scene=self.config.get("simulation", {}).get("scene") or "")
        self.analyzer = Analyzer(
            self.stats,
            output_path=out.get("stats_file", "simulator/results/silf.json"),
            verbose=out.get("verbose", True),
            dump_enabled=dump_enabled,
        )
        self.plans = build_plans(self.config, verbose=out.get("verbose", True))

    def run(self) -> SimStats:
        self.analyzer.start(self.config)
        if not self.plans:
            self.analyzer.finalize(0.0, 1.0)
            return self.stats

        plan0 = self.plans[0]
        self.stats.pixels = {
            "full": plan0.full_pixels,
            "coarse": plan0.coarse_pixels,
            "fine": plan0.fine_pixels,
            "roi_fraction": plan0.roi_fraction,
            "fallback_rate": plan0.fallback_rate,
            "scar_skip_ratio": plan0.scar_skip_ratio,
        }
        self.stats.notes = per_pixel_cycles(self.config)
        visible = [p.num_visible for p in self.plans]
        self.stats.notes["num_scene"] = self.plans[0].num_scene
        self.stats.notes["num_visible_mean"] = sum(visible) / len(visible)
        self.stats.notes["num_frames"] = len(self.plans)
        self.stats.notes["resolution"] = f"{self.plans[0].width}x{self.plans[0].height}"
        self.stats.notes["num_queries"] = float(self.config.get("simulation", {}).get("num_queries", 1) or 1)

        env = simpy.Environment()
        gfp, hse, ure, sde, ftg, semantic, dram = self._build(env)
        pipe = Pipeline(env, gfp, hse, ure, sde, ftg, semantic)
        env.process(pipe.run_frames(self.plans))
        env.run()

        hse.finalize(self.analyzer)
        ure.finalize(self.analyzer)
        sde.finalize(self.analyzer)
        semantic.finalize(self.analyzer)
        dram_busy = dram.busy_cycles()
        if dram_busy > 0:
            self.analyzer.record_busy("dram", dram_busy)
        self.stats.dram_bytes = dict(dram.bytes)
        self.stats.frame_cycles = list(pipe.frame_cycles)

        clock = float(self.config.get("hardware", {}).get("clock_frequency", 1.0))
        period_ns = 1.0 / clock if clock > 0 else 1.0
        self.analyzer.finalize(env.now, period_ns)
        return self.stats

    def _build(self, env):
        hw = self.config.get("hardware", {})
        algo = self.config.get("algorithm", {})
        clock = float(hw.get("clock_frequency", 1.0))
        mem = hw.get("memory", {})
        sem = hw.get("semantic", {})
        sde_cfg = hw.get("sde", {})
        lang_bytes = language_payload_bytes(sde_cfg)
        geom_bytes = int(mem.get("geom_bytes", 240))

        dram = DramSystem(env, DramConfig(
            memory_bandwidth_gbps=float(mem.get("memory_bandwidth", 51.2)),
            clock_frequency_ghz=clock,
            read_latency_hiding_rate=float(mem.get("read_latency_hiding_rate", 0.3)),
            row_buffer_bytes=int(mem.get("row_buffer_bytes", 2048)),
            num_channels=int(sem.get("num_channels", 4)),
            tRCD_ns=float(mem.get("tRCD_ns", 18)),
            tRP_ns=float(mem.get("tRP_ns", 18)),
            tRAS_ns=float(mem.get("tRAS_ns", 42)),
            tWR_ns=float(mem.get("tWR_ns", 34)),
            tRRD_ns=float(mem.get("tRRD_ns", 5)),
            tREFI_ns=float(mem.get("tREFI_ns", 3906)),
            tRFC_ns=float(mem.get("tRFC_ns", 180)),
        ))
        gfp_cfg = hw.get("gfp", {})
        gfp = GaussianFrontEnd(env, GFPConfig(
            cull_cycles=float(gfp_cfg.get("cull_cycles", 2)),
            project_cycles=float(gfp_cfg.get("project_cycles", 2)),
            bin_cycles=float(gfp_cfg.get("bin_cycles", 2)),
            geom_bytes=geom_bytes,
        ), dram, self.analyzer)
        hse_cfg = hw.get("hse", {})
        hse = HierarchicalSort(env, HSEConfig(
            num_cores=int(hse_cfg.get("num_cores", 4)),
            coarse_cycles_per_chunk=float(hse_cfg.get("coarse_sort_cycles", 8)),
            fine_cycles_per_chunk=float(hse_cfg.get("fine_sort_cycles", 10)),
            bytes_per_gaussian=int(hse_cfg.get("bytes_per_gaussian", 8)),
        ), dram)
        ure_cfg = hw.get("ure", {})
        ure_bytes = int(ure_cfg.get("bytes_per_gaussian", 0))
        if ure_bytes <= 0:
            ure_bytes = geom_bytes + lang_bytes
        ure = UnifiedRaster(env, UREConfig(
            num_abus=int(ure_cfg.get("num_abus", 4)),
            cores_per_abu=int(ure_cfg.get("cores_per_abu", 64)),
            cycles_per_pass=float(ure_cfg.get("cycles_per_pass", 1)),
            early_stop_ratio=float(algo.get("early_stop_ratio", ure_cfg.get("early_stop_ratio", 0.7))),
            bytes_per_gaussian=ure_bytes,
        ), dram)
        sde = SemanticDecoder(env, SDEConfig(
            array_dim=int(sde_cfg.get("array_dim", 8)),
            macs_per_cycle=int(sde_cfg.get("macs_per_cycle", 0)),
            decode_k=int(sde_cfg.get("decode_k", 8)),
            codebook_size=int(sde_cfg.get("codebook_size", 64)),
            num_levels=int(sde_cfg.get("num_levels", 3)),
            clip_dim=int(sde_cfg.get("clip_dim", 512)),
            sparse_decode=bool(sde_cfg.get("sparse_decode", True)),
            output_buffer_tiles=int(sde_cfg.get("output_buffer_tiles", 2)),
        ))
        ftg_cfg = hw.get("ftg", {})
        ftg = FineTaskGenerator(env, FTGConfig(
            num_vrus=int(ftg_cfg.get("num_vrus", 4)),
            cycles_per_coarse_pixel=float(ftg_cfg.get("cycles_per_coarse_pixel", 8)),
            dilation_cycles_per_tile=float(ftg_cfg.get("dilation_cycles_per_tile", 4)),
        ), self.analyzer)
        pim = PimConfig(
            placement=str(sem.get("placement", "pim")),
            pim_frequency_ghz=float(hw.get("pim_frequency", 0.2)),
            asic_frequency_ghz=clock,
            num_channels=int(sem.get("num_channels", 4)),
            banks_per_channel=int(sem.get("banks_per_channel", 16)),
            spus_per_channel=int(sem.get("spus_per_channel", 8)),
            lanes=int(sem.get("lanes", 8)),
            clip_dim=int(sde_cfg.get("clip_dim", 512)),
            feature_bytes=int(sem.get("feature_bytes", 1024)),
            result_bytes_coarse=float(sem.get("result_bytes_coarse", 4)),
            result_bytes_fine=float(sem.get("result_bytes_fine", 0.125)),
            row_buffer_bytes=int(mem.get("row_buffer_bytes", 2048)),
            pdu_cycles_per_pixel=float(sem.get("pdu_cycles_per_pixel", 1)),
            tRCD_ns=float(mem.get("tRCD_ns", 18)),
            tRP_ns=float(mem.get("tRP_ns", 18)),
            tRAS_ns=float(mem.get("tRAS_ns", 42)),
            tRRD_ns=float(mem.get("tRRD_ns", 5)),
            tREFI_ns=float(mem.get("tREFI_ns", 3906)),
            tRFC_ns=float(mem.get("tRFC_ns", 180)),
            num_queries=float(self.config.get("simulation", {}).get("num_queries", 1) or 1),
        )
        semantic = SemanticArray(env, pim, dram)
        return gfp, hse, ure, sde, ftg, semantic, dram


def per_pixel_cycles(config: dict) -> dict:
    """Isolated stage cost of one dense pixel, before tile overlap."""
    hw = config.get("hardware", {})
    mem = hw.get("memory", {})
    sem = hw.get("semantic", {})
    sde = hw.get("sde", {})
    clock = float(hw.get("clock_frequency", 1.0))
    bw = float(mem.get("memory_bandwidth", 51.2)) * 1e9 / (clock * 1e9)
    feature_bytes = float(sem.get("feature_bytes", 1024))
    terms = int(sde.get("decode_k", 8) if sde.get("sparse_decode", True) else sde.get("codebook_size", 64))
    macs = int(sde.get("num_levels", 3)) * terms * int(sde.get("clip_dim", 512))
    macs_per = int(sde.get("macs_per_cycle", 0))
    if macs_per <= 0:
        k = max(1, int(sde.get("array_dim", 8)))
        macs_per = k * k
    pim_ghz = float(hw.get("pim_frequency", 0.2))
    lanes = max(1, int(sem.get("lanes", 8)))
    nspu = int(sem.get("num_channels", 4)) * int(sem.get("spus_per_channel", 8))
    queries = float(config.get("simulation", {}).get("num_queries", 1) or 1)
    stream = math.ceil(int(sde.get("clip_dim", 512)) / lanes) * (clock / pim_ghz if pim_ghz else 1.0)
    write = feature_bytes / bw if bw else 0.0
    placement = str(sem.get("placement", "pim"))
    return {
        "num_queries": queries,
        "sde": macs / macs_per,
        "feature_write": write,
        "feature_readback": write * queries if placement == "asic" else 0.0,
        "spu_stream": stream * queries / max(1, nspu),
        "gaussian_bytes": float(language_payload_bytes(sde) + int(mem.get("geom_bytes", 240))),
    }


def run_simulator(config_path: str):
    with open(config_path, "r", encoding="utf-8") as f:
        base = yaml.safe_load(f)
    sim_cfg = (base or {}).get("simulation", {})
    dataset = canonical_dataset(sim_cfg.get("dataset") or "")
    scene = sim_cfg.get("scene")
    scene_missing = scene is None or (isinstance(scene, str) and scene.strip() == "")
    if not dataset or not scene_missing:
        sim = Simulator(config_path, config=base, dump_enabled=True)
        return sim.run()

    names = DATASETS.get(dataset, [])
    if not names:
        sim = Simulator(config_path, config=base, dump_enabled=True)
        return sim.run()

    output_path = (base.get("output") or {}).get("stats_file", "simulator/results/silf.json")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    results = {
        "mode": "multi_scene",
        "dataset": dataset,
        "scenes": [],
        "generated_at": datetime.now().isoformat(),
        "config_path": os.path.abspath(config_path),
    }
    print(f"[simulator] dataset={dataset} scenes={len(names)}")
    for i, name in enumerate(names):
        cfg = copy.deepcopy(base)
        cfg.setdefault("simulation", {})
        cfg["simulation"]["dataset"] = dataset
        cfg["simulation"]["scene"] = name
        # Scene table supplies resolution and sparsity for this name.
        cfg["simulation"].pop("resolution", None)
        cfg["simulation"].pop("num_queries", None)
        for key in ("roi_fraction", "fallback_rate", "scar_skip_ratio"):
            cfg.get("algorithm", {}).pop(key, None)
        print(f"[simulator] ({i + 1}/{len(names)}) {name}")
        sim = Simulator(config_path, config=cfg, dump_enabled=False)
        stats = sim.run()
        row = stats.to_dict()
        row.pop("config", None)
        results["scenes"].append(row)
        print(f"[simulator] {name} cycles={stats.total_cycles:.0f} fps={stats.fps:.2f}")

    fps_list = [s["fps"] for s in results["scenes"] if s.get("fps")]
    if fps_list:
        results["average_fps"] = sum(fps_list) / len(fps_list)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"[simulator] wrote {output_path}")
    return results


def _load(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
