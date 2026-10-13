"""One view through GRE, then the semantic phase, then the fine pass.

Coarse and fine share sort, raster, and decode. The fine pass does not
project or sort again. Norm and query matching do not start until that
stage's feature map has been written to DRAM: the on-chip buffer cannot
hold the 512-d map, so the semantic unit consumes it from memory. PIM
reads the vectors from the local row buffer. The ASIC-only SPU has to
reload every vector across the external bus before it can compute, and
that reload is on the frame's critical path.
"""

from typing import List
import simpy

from simulator.engines import (
    FineTaskGenerator,
    GaussianFrontEnd,
    HierarchicalSort,
    SemanticArray,
    SemanticDecoder,
    UnifiedRaster,
)
from simulator.structures import FramePlan, TileTask


class Pipeline:
    def __init__(
        self,
        env: simpy.Environment,
        gfp: GaussianFrontEnd,
        hse: HierarchicalSort,
        ure: UnifiedRaster,
        sde: SemanticDecoder,
        ftg: FineTaskGenerator,
        semantic: SemanticArray,
    ):
        self.env = env
        self.gfp = gfp
        self.hse = hse
        self.ure = ure
        self.sde = sde
        self.ftg = ftg
        self.semantic = semantic
        self.frame_cycles: List[float] = []

    def run_frames(self, plans: List[FramePlan]):
        for plan in plans:
            t0 = self.env.now
            yield self.gfp.run(plan.num_scene, plan.num_visible)
            if plan.run_coarse:
                yield from self._tiles(plan.coarse_tasks)
                yield self.semantic.run(self._pixels(plan.coarse_tasks), "coarse")
                yield self.ftg.run(plan.coarse_pixels, len(plan.coarse_tasks))
            yield from self._tiles(plan.fine_tasks)
            yield self.semantic.run(self._pixels(plan.fine_tasks), "fine")
            self.frame_cycles.append(self.env.now - t0)

    def _tiles(self, tasks: List[TileTask]):
        if not tasks:
            return
        procs = [self.env.process(self._tile(task)) for task in tasks]
        yield simpy.AllOf(self.env, procs)

    @staticmethod
    def _pixels(tasks: List[TileTask]) -> float:
        return sum(task.active_pixels for task in tasks)

    def _tile(self, task: TileTask):
        if task.sort and task.num_gaussians > 0:
            yield self.hse.run(task)
        if task.num_gaussians > 0:
            yield self.ure.run(task)
        if task.active_pixels <= 0:
            return
        # The on-chip buffer only holds the tile being decoded. Once the
        # 512-d vectors are in DRAM the buffer can be reused; matching them
        # is a separate phase and must not run under this decode.
        buf = self.sde.buffer.request()
        yield buf
        try:
            yield self.sde.run(task)
            yield self.semantic.write_features(task.active_pixels)
        finally:
            self.sde.buffer.release(buf)
