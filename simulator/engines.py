"""Cycle models for the six SILF blocks.

GFP, HSE, URE, SDE, and FTG run at the ASIC clock.
SPU / PDU run at the PIM clock whether they sit on the DRAM die or,
in the ASIC-only baseline, next to the GRE. The paper keeps their
count, datapath, and frequency fixed so the two designs have the same
arithmetic throughput.
"""

from dataclasses import dataclass
from typing import List, Tuple
import math
import simpy

from simulator.memory import DramSystem, PimConfig, pim_compute_cycles
from simulator.structures import TileTask, merge_intervals


@dataclass
class GFPConfig:
    cull_cycles: float = 2.0
    project_cycles: float = 2.0
    bin_cycles: float = 2.0
    geom_bytes: int = 240


class GaussianFrontEnd:
    """Project, cull, and bin. One pass per view; the fine stage reuses it."""

    def __init__(self, env, cfg: GFPConfig, dram: DramSystem, analyzer):
        self.env = env
        self.cfg = cfg
        self.dram = dram
        self.analyzer = analyzer

    def run(self, num_scene: int, num_visible: int):
        return self.env.process(self._run(num_scene, num_visible))

    def _run(self, num_scene: int, num_visible: int):
        start = self.env.now
        nbytes = num_scene * self.cfg.geom_bytes
        yield self.dram.transfer(nbytes, "gaussian_read", hide=True)
        cycles = (
            num_scene * self.cfg.cull_cycles
            + num_visible * (self.cfg.project_cycles + self.cfg.bin_cycles)
        )
        if cycles > 0:
            yield self.env.timeout(cycles)
        self.analyzer.record_busy("gfp", self.env.now - start)


@dataclass
class HSEConfig:
    num_cores: int = 4
    coarse_cycles_per_chunk: float = 8.0
    fine_cycles_per_chunk: float = 10.0
    bytes_per_gaussian: int = 8


class HierarchicalSort:
    """GSCore-style hierarchical sort. One core owns one tile for all of its chunks.

    Per chunk the latency is coarse_cycles + fine_cycles, independent of how
    full the chunk is. More chunks therefore cost more, matching the previous
    simulator and the GSCore startup model.
    """

    def __init__(self, env, cfg: HSEConfig, dram: DramSystem):
        self.env = env
        self.cfg = cfg
        self.dram = dram
        self.resource = simpy.Resource(env, capacity=max(1, cfg.num_cores))
        self._intervals: List[Tuple[float, float]] = []

    def run(self, task: TileTask):
        return self.env.process(self._run(task))

    def _run(self, task: TileTask):
        if task.num_gaussians <= 0 or task.num_chunks <= 0:
            return
        with self.resource.request() as req:
            yield req
            start = self.env.now
            miss_n = task.num_gaussians if task.uncached_gaussians < 0 else task.uncached_gaussians
            nbytes = miss_n * self.cfg.bytes_per_gaussian
            yield self.dram.transfer(nbytes, "sort_read", hide=True)
            cycles = task.num_chunks * (
                self.cfg.coarse_cycles_per_chunk + self.cfg.fine_cycles_per_chunk
            )
            if cycles > 0:
                yield self.env.timeout(cycles)
            self._intervals.append((start, self.env.now))

    def finalize(self, analyzer) -> None:
        busy = merge_intervals(self._intervals)
        if busy > 0:
            analyzer.record_busy("hse", busy)


@dataclass
class UREConfig:
    num_abus: int = 4
    cores_per_abu: int = 64          # 4 x 64 = 256 raster cores
    cycles_per_pass: float = 1.0
    early_stop_ratio: float = 0.7
    bytes_per_gaussian: int = 0      # 0 → geometry + language payload


class UnifiedRaster:
    """Alpha blending of RGB and the latent code weights.

    A tile occupies one ABU. The ABU's cores cover the active pixels in
    ``ceil`` passes; SCAR and the coarse grid show up only as ``active_pixels``.
    """

    def __init__(self, env, cfg: UREConfig, dram: DramSystem):
        self.env = env
        self.cfg = cfg
        self.dram = dram
        self.resource = simpy.Resource(env, capacity=max(1, cfg.num_abus))
        self._intervals: List[Tuple[float, float]] = []

    def run(self, task: TileTask):
        return self.env.process(self._run(task))

    def cycles(self, task: TileTask) -> float:
        if task.num_gaussians <= 0 or task.active_pixels <= 0:
            return 0.0
        passes = task.active_pixels / max(1, self.cfg.cores_per_abu)
        return (
            task.num_gaussians
            * self.cfg.early_stop_ratio
            * passes
            * self.cfg.cycles_per_pass
        )

    def _run(self, task: TileTask):
        cyc = self.cycles(task)
        if cyc <= 0 and task.num_gaussians <= 0:
            return
        with self.resource.request() as req:
            yield req
            start = self.env.now
            miss_n = task.num_gaussians if task.uncached_gaussians < 0 else task.uncached_gaussians
            nbytes = miss_n * max(0, self.cfg.bytes_per_gaussian)
            yield self.dram.transfer(nbytes, "raster_read", hide=True)
            if cyc > 0:
                yield self.env.timeout(cyc)
            self._intervals.append((start, self.env.now))

    def finalize(self, analyzer) -> None:
        busy = merge_intervals(self._intervals)
        if busy > 0:
            analyzer.record_busy("ure", busy)


@dataclass
class SDEConfig:
    array_dim: int = 8
    macs_per_cycle: int = 0          # 0 → array_dim^2; set explicitly to match the feature stream
    decode_k: int = 8
    codebook_size: int = 64
    num_levels: int = 3
    clip_dim: int = 512
    sparse_decode: bool = True
    output_buffer_tiles: int = 2


class SemanticDecoder:
    """Codebook decode on a K x K systolic array.

    Sparse mode accumulates ``decode_k`` codewords per level (the paper's K).
    Dense mode multiplies the full codebook. The output buffer is the small
    on-chip tile buffer: decode of the next tile waits until this tile's
    feature vector has been written out.
    """

    def __init__(self, env, cfg: SDEConfig):
        self.env = env
        self.cfg = cfg
        self.resource = simpy.Resource(env, capacity=1)
        self.buffer = simpy.Resource(env, capacity=max(1, cfg.output_buffer_tiles))
        self._intervals: List[Tuple[float, float]] = []

    def macs_per_cycle(self) -> int:
        if self.cfg.macs_per_cycle > 0:
            return self.cfg.macs_per_cycle
        k = max(1, self.cfg.array_dim)
        return k * k

    def cycles_per_pixel(self) -> float:
        terms = self.cfg.decode_k if self.cfg.sparse_decode else self.cfg.codebook_size
        macs = self.cfg.num_levels * terms * self.cfg.clip_dim
        return macs / self.macs_per_cycle()

    def hold_buffer(self):
        return self.buffer.request()

    def run(self, task: TileTask):
        return self.env.process(self._run(task))

    def _run(self, task: TileTask):
        if task.active_pixels <= 0:
            return
        with self.resource.request() as req:
            yield req
            start = self.env.now
            yield self.env.timeout(task.active_pixels * self.cycles_per_pixel())
            self._intervals.append((start, self.env.now))

    def finalize(self, analyzer) -> None:
        busy = merge_intervals(self._intervals)
        if busy > 0:
            analyzer.record_busy("sde", busy)


@dataclass
class FTGConfig:
    num_vrus: int = 4
    cycles_per_coarse_pixel: float = 8.0
    dilation_cycles_per_tile: float = 4.0


class FineTaskGenerator:
    """Relevance from (s, n), SCAR tests, and the fine-task queue.

    The ROI bitmap itself is applied when the frame is planned. This block
    only spends the cycles of walking the coarse map with ``num_vrus`` units.
    """

    def __init__(self, env, cfg: FTGConfig, analyzer):
        self.env = env
        self.cfg = cfg
        self.analyzer = analyzer

    def run(self, coarse_pixels: int, coarse_tiles: int):
        return self.env.process(self._run(coarse_pixels, coarse_tiles))

    def _run(self, coarse_pixels: int, coarse_tiles: int):
        vrus = max(1, self.cfg.num_vrus)
        cycles = (
            math.ceil(coarse_pixels / vrus) * self.cfg.cycles_per_coarse_pixel
            + math.ceil(max(coarse_tiles, 1) / vrus) * self.cfg.dilation_cycles_per_tile
        )
        start = self.env.now
        if cycles > 0:
            yield self.env.timeout(cycles)
        self.analyzer.record_busy("ftg", self.env.now - start)


class SemanticArray:
    """Norm and query matching over one finished feature map.

    The feature map is written once, upstream of this stage. Each query is
    another dot product against that same map. ASIC-only reloads every
    512-d vector once per query before the SPU can use it. PIM leaves the
    vector in the bank row buffer for every query and only returns (s, n)
    or a predicate bit per query. The busy interval covers the reload, the
    SPU, and the compact write-back, because the frame waits for all three.
    """

    def __init__(self, env, cfg: PimConfig, dram: DramSystem):
        self.env = env
        self.cfg = cfg
        self.dram = dram
        self.resource = simpy.Resource(env, capacity=1)
        self._intervals: List[Tuple[float, float]] = []

    def write_features(self, n_pixels: float):
        nbytes = n_pixels * self.cfg.feature_bytes
        return self.dram.transfer(nbytes, "feature_write", hide=False)

    def run(self, n_pixels: float, stage: str):
        return self.env.process(self._run(n_pixels, stage))

    def _run(self, n_pixels: float, stage: str):
        queries = float(self.cfg.num_queries)
        if n_pixels <= 0 or queries <= 0:
            return
        start = self.env.now
        if self.cfg.placement == "asic":
            # One full feature-map reload per query. PIM never does this.
            nbytes = n_pixels * self.cfg.feature_bytes * queries
            yield self.dram.transfer(nbytes, "feature_readback", hide=False)
        with self.resource.request() as req:
            yield req
            cycles = pim_compute_cycles(n_pixels, self.cfg, stage)
            if cycles > 0:
                yield self.env.timeout(cycles)
        if self.cfg.placement == "pim":
            result = self.cfg.result_bytes_coarse if stage == "coarse" else self.cfg.result_bytes_fine
            yield self.dram.transfer(n_pixels * result * queries, "result_return", hide=False)
        self._intervals.append((start, self.env.now))

    def finalize(self, analyzer) -> None:
        busy = merge_intervals(self._intervals)
        if busy > 0:
            analyzer.record_busy("semantic", busy)
