"""Collect stage occupancy, DRAM bytes, and FPS."""

import json
import os
import time
from datetime import datetime

from simulator.structures import SimStats


class Analyzer:
    def __init__(self, stats: SimStats, output_path: str, verbose: bool = True, dump_enabled: bool = True):
        self.stats = stats
        self.output_path = output_path
        self.verbose = verbose
        self.dump_enabled = dump_enabled
        self._stage_map = {
            "gfp": "gfp_cycles",
            "hse": "sort_cycles",
            "ure": "raster_cycles",
            "sde": "decode_cycles",
            "ftg": "ftg_cycles",
            "semantic": "semantic_cycles",
            "dram": "dram_cycles",
        }
        self._start = None

    def start(self, config: dict) -> None:
        self._start = time.time()
        self.stats.start_time = datetime.now().isoformat()
        self.stats.config = config

    def record_busy(self, module: str, cycles: float) -> None:
        self.stats.record_busy(module, cycles)
        field = self._stage_map.get(module)
        if field and hasattr(self.stats, field):
            setattr(self.stats, field, getattr(self.stats, field) + cycles)

    def finalize(self, total_cycles: float, clock_period_ns: float) -> None:
        self.stats.total_cycles = total_cycles
        if self._start is not None:
            self.stats.elapsed_time = time.time() - self._start
        period_s = clock_period_ns * 1e-9
        self.stats.frame_times = [c * period_s for c in self.stats.frame_cycles]
        if self.stats.frame_times:
            avg = sum(self.stats.frame_times) / len(self.stats.frame_times)
            self.stats.fps = (1.0 / avg) if avg > 0 else 0.0
        if self.dump_enabled:
            self._dump()
        if self.verbose:
            self._print()

    def _dump(self) -> None:
        folder = os.path.dirname(self.output_path)
        if folder:
            os.makedirs(folder, exist_ok=True)
        with open(self.output_path, "w", encoding="utf-8") as f:
            json.dump(self.stats.to_dict(), f, indent=2, ensure_ascii=False)

    def _print(self) -> None:
        st = self.stats
        print("[analyzer] === SILF simulation ===")
        if st.scene:
            print(f"scene             : {st.scene}")
        print(f"total_cycles      : {st.total_cycles:.1f}")
        print(f"gfp_cycles        : {st.gfp_cycles:.1f}")
        print(f"sort_cycles       : {st.sort_cycles:.1f}")
        print(f"raster_cycles     : {st.raster_cycles:.1f}")
        print(f"decode_cycles     : {st.decode_cycles:.1f}")
        print(f"ftg_cycles        : {st.ftg_cycles:.1f}")
        print(f"semantic_cycles   : {st.semantic_cycles:.1f}")
        print(f"dram_cycles       : {st.dram_cycles:.1f}")
        if st.frame_cycles:
            avg = sum(st.frame_cycles) / len(st.frame_cycles)
            print(f"frames            : {len(st.frame_cycles)}, avg {avg:.1f} cycles")
        if st.frame_times:
            avg_ms = 1000.0 * sum(st.frame_times) / len(st.frame_times)
            print(f"frame_time        : {avg_ms:.3f} ms")
            print(f"fps               : {st.fps:.2f}")
        if st.pixels:
            print(f"pixels            : {st.pixels}")
        if st.dram_bytes:
            total_b = sum(st.dram_bytes.values())
            pretty = {k: _fmt_bytes(v) for k, v in st.dram_bytes.items()}
            print(f"dram_bytes        : {pretty}  total {_fmt_bytes(total_b)}")
        if st.notes:
            balance = {k: round(v, 2) for k, v in st.notes.items()}
            print(f"per_pixel_cycles  : {balance}")
        if st.elapsed_time > 0:
            print(f"elapsed_time      : {st.elapsed_time:.3f} s")


def _fmt_bytes(n: float) -> str:
    if n >= 1e9:
        return f"{n / 1e9:.3f} GB"
    if n >= 1e6:
        return f"{n / 1e6:.2f} MB"
    if n >= 1e3:
        return f"{n / 1e3:.1f} KB"
    return f"{n:.0f} B"
