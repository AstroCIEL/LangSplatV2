"""Workload objects shared by the planner and the cycle engines."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import math


def parse_resolution(res_str: str) -> Tuple[int, int]:
    w, h = res_str.strip().lower().split("x")
    return int(w), int(h)


def tile_pixels(tx: int, ty: int, width: int, height: int, tile_size: int) -> int:
    """Pixel count of one tile, including a partial tile on the right or bottom edge."""
    w = min(tile_size, width - tx * tile_size)
    h = min(tile_size, height - ty * tile_size)
    if w <= 0 or h <= 0:
        return 0
    return w * h


def grid_size(width: int, height: int, tile_size: int) -> Tuple[int, int]:
    return math.ceil(width / tile_size), math.ceil(height / tile_size)


@dataclass
class TileTask:
    """One tile handed to sort / raster / decode / semantic.

    ``active_pixels`` may be fractional. That is the expected pixel count
    after averaging fallback and SCAR over queries, not a stochastic sample.
    """

    frame_id: int
    tile_id: int
    stage: str  # "coarse" | "fine"
    num_gaussians: int
    num_chunks: int
    active_pixels: float
    gaussian_ids: List[int] = field(default_factory=list)
    # Gaussians not shared with the previous tile in scan order.
    # -1 means "no reuse info", and the engines fall back to num_gaussians.
    uncached_gaussians: int = -1
    sort: bool = True


@dataclass
class FramePlan:
    """Everything the pipeline needs for one view. Built offline, timed online."""

    frame_id: int
    width: int
    height: int
    coarse_width: int
    coarse_height: int
    tile_size: int
    num_visible: int
    num_scene: int
    coarse_tasks: List[TileTask]
    fine_tasks: List[TileTask]
    run_coarse: bool
    fine_sort: bool
    coarse_pixels: int
    fine_pixels: float
    full_pixels: int
    roi_fraction: float
    fallback_rate: float
    scar_skip_ratio: float
    scene: str = ""


@dataclass
class SimStats:
    total_cycles: float = 0.0
    gfp_cycles: float = 0.0
    sort_cycles: float = 0.0
    raster_cycles: float = 0.0
    decode_cycles: float = 0.0
    ftg_cycles: float = 0.0
    semantic_cycles: float = 0.0
    dram_cycles: float = 0.0
    frame_cycles: List[float] = field(default_factory=list)
    frame_times: List[float] = field(default_factory=list)
    module_busy: Dict[str, float] = field(default_factory=dict)
    dram_bytes: Dict[str, float] = field(default_factory=dict)
    pixels: Dict[str, float] = field(default_factory=dict)
    fps: float = 0.0
    start_time: str = ""
    elapsed_time: float = 0.0
    config: dict = field(default_factory=dict)
    scene: str = ""
    notes: Dict[str, float] = field(default_factory=dict)

    def record_busy(self, module: str, cycles: float) -> None:
        self.module_busy[module] = self.module_busy.get(module, 0.0) + cycles

    def to_dict(self) -> dict:
        return {
            "scene": self.scene,
            "total_cycles": self.total_cycles,
            "fps": self.fps,
            "gfp_cycles": self.gfp_cycles,
            "sort_cycles": self.sort_cycles,
            "raster_cycles": self.raster_cycles,
            "decode_cycles": self.decode_cycles,
            "ftg_cycles": self.ftg_cycles,
            "semantic_cycles": self.semantic_cycles,
            "dram_cycles": self.dram_cycles,
            "frame_cycles": self.frame_cycles,
            "frame_times": self.frame_times,
            "module_busy": self.module_busy,
            "dram_bytes": self.dram_bytes,
            "pixels": self.pixels,
            "notes": self.notes,
            "start_time": self.start_time,
            "elapsed_time": self.elapsed_time,
            "config": self.config,
        }


def merge_intervals(intervals: List[Tuple[float, float]]) -> float:
    """Length of the union of [start, end) busy intervals."""
    if not intervals:
        return 0.0
    ordered = sorted(intervals, key=lambda x: x[0])
    total = 0.0
    cur_s, cur_e = ordered[0]
    for s, e in ordered[1:]:
        if s <= cur_e:
            cur_e = max(cur_e, e)
        else:
            total += cur_e - cur_s
            cur_s, cur_e = s, e
    total += cur_e - cur_s
    return total


def centered_roi(ntx: int, nty: int, fraction: float) -> set:
    """Tile ids of a centered rectangle whose area is about ``fraction`` of the grid."""
    n = ntx * nty
    target = int(round(fraction * n))
    if target <= 0 or n == 0:
        return set()
    target = min(n, target)
    # Match the image aspect so the window is not a thin strip.
    aspect = ntx / max(nty, 1)
    rh = max(1, int(round(math.sqrt(target / aspect))))
    rh = min(nty, rh)
    rw = max(1, int(round(target / rh)))
    rw = min(ntx, rw)
    # Grow until the rectangle covers at least the target, then shrink a row/col if overshot.
    while rw < ntx and (rw * rh) < target:
        rw += 1
    while rh < nty and (rw * rh) < target:
        rh += 1
    ids = []
    x0 = (ntx - rw) // 2
    y0 = (nty - rh) // 2
    for ty in range(y0, y0 + rh):
        for tx in range(x0, x0 + rw):
            ids.append(ty * ntx + tx)
    if len(ids) > target:
        ids = ids[:target]
    return set(ids)


def spread_ids(candidates: List[int], count: int) -> List[int]:
    """Deterministic evenly spaced subset. Used for the fallback query average."""
    if count <= 0 or not candidates:
        return []
    count = min(count, len(candidates))
    if count == len(candidates):
        return list(candidates)
    step = len(candidates) / count
    return [candidates[int(i * step)] for i in range(count)]
