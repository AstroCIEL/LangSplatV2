"""Build a FramePlan from a synthetic Gaussian field or a LangSplatV2 checkpoint.

The cycle engines never call the rasterizer. They only see tile lists.
Real checkpoints are optional: if CUDA or the model is missing, the caller
falls back to the synthetic field, which is what the default configs use.
"""

from collections import defaultdict
import copy
import math
import os
import sys
from typing import Dict, List, Optional, Tuple

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from simulator.scenes import SCENES, canonical_dataset, scene_checkpoint
from simulator.structures import (
    FramePlan,
    TileTask,
    centered_roi,
    grid_size,
    parse_resolution,
    spread_ids,
    tile_pixels,
)


def resolve_scene(config: dict) -> dict:
    """Fill resolution and sparsity from the paper table when the yaml omits them."""
    sim = config.setdefault("simulation", {})
    algo = config.setdefault("algorithm", {})
    name = (sim.get("scene") or "").strip()
    meta = SCENES.get(name, {})
    dataset = canonical_dataset(sim.get("dataset") or meta.get("dataset", ""))
    if dataset:
        sim["dataset"] = dataset
    if meta and not sim.get("resolution"):
        sim["resolution"] = f"{meta['width']}x{meta['height']}"
    for key in ("roi_fraction", "fallback_rate", "scar_skip_ratio"):
        if algo.get(key) is None and meta:
            algo[key] = meta[key]
    if sim.get("num_queries") in (None, "") and meta.get("num_queries") is not None:
        sim["num_queries"] = float(meta["num_queries"])
    sim.setdefault("num_queries", 1.0)
    algo.setdefault("roi_fraction", 0.5)
    algo.setdefault("fallback_rate", 0.0)
    algo.setdefault("scar_skip_ratio", 0.0)
    if name and not config.get("workload", {}).get("use_synthetic", False):
        ckpt = scene_checkpoint(name)
        if ckpt:
            sim.setdefault("model_path", ckpt["model_path"])
            sim.setdefault("source_path", ckpt["source_path"])
            if ckpt.get("note"):
                sim["checkpoint_note"] = ckpt["note"]
    return config


def build_plans(config: dict, verbose: bool = True) -> List[FramePlan]:
    resolve_scene(config)
    algo = config.get("algorithm", {})
    tile_size = int(algo.get("tile_size", 16))
    chunk_size = int(algo.get("chunk_size", 64))
    plans = None
    if not config.get("workload", {}).get("use_synthetic", False):
        plans = _cached_real_plans(config, tile_size, chunk_size, verbose)
        if not plans:
            scene = config.get("simulation", {}).get("scene") or "-"
            model = config.get("simulation", {}).get("model_path") or ""
            raise RuntimeError(
                f"real workload failed for scene={scene} model_path={model}; "
                "refusing to fall back to the synthetic field"
            )
    if not plans:
        plans = [build_synthetic_plan(config, frame_id=0, tile_size=tile_size, chunk_size=chunk_size)]
        if verbose:
            p = plans[0]
            print(
                f"[workload] synthetic scene={p.scene or '-'} "
                f"{p.width}x{p.height} visible={p.num_visible} "
                f"coarse_tiles={len(p.coarse_tasks)} fine_tiles={len(p.fine_tasks)} "
                f"fine_pixels={p.fine_pixels:.0f}/{p.full_pixels}"
            )
    num_frames = int(config.get("simulation", {}).get("num_frames", 1))
    if len(plans) == 1 and num_frames > 1:
        base = plans[0]
        plans = [_retag(base, i) for i in range(num_frames)]
    return plans


_PLAN_CACHE: Dict[tuple, List[FramePlan]] = {}


def _plan_cache_key(config: dict, tile_size: int, chunk_size: int) -> tuple:
    sim = config.get("simulation", {})
    algo = config.get("algorithm", {})
    return (
        sim.get("model_path"),
        sim.get("source_path"),
        sim.get("scene"),
        str(sim.get("camera_ids", sim.get("frames", 0))),
        sim.get("camera_split", "test"),
        sim.get("iteration", -1),
        bool(algo.get("hls_enabled", True)),
        bool(algo.get("scar_enabled", True)),
        float(algo.get("roi_fraction", 0.5)),
        float(algo.get("fallback_rate", 0.0)),
        float(algo.get("scar_skip_ratio", 0.0)),
        tile_size,
        chunk_size,
        int(algo.get("coarse_downscale", 4)),
    )


def _cached_real_plans(config: dict, tile_size: int, chunk_size: int, verbose: bool) -> Optional[List[FramePlan]]:
    """Load each checkpoint once. PIM and ASIC-only share the same tile lists."""
    key = _plan_cache_key(config, tile_size, chunk_size)
    cached = _PLAN_CACHE.get(key)
    if cached is not None:
        if verbose:
            scene = config.get("simulation", {}).get("scene") or "-"
            print(f"[workload] reuse {len(cached)} cached frame(s) for {scene}")
        return cached
    plans = load_workload_from_scene(config, tile_size=tile_size, chunk_size=chunk_size, verbose=verbose)
    if plans:
        _PLAN_CACHE[key] = plans
    return plans


def _retag(plan: FramePlan, frame_id: int) -> FramePlan:
    # Tasks are read-only after planning, so frames share the tile lists.
    cloned = copy.copy(plan)
    cloned.frame_id = frame_id
    return cloned


def build_synthetic_plan(config: dict, frame_id: int, tile_size: int, chunk_size: int) -> FramePlan:
    sim = config.get("simulation", {})
    wl = config.get("workload", {})
    width, height = parse_resolution(sim.get("resolution", "986x728"))
    num_gaussians = int(wl.get("num_gaussians", 200_000))
    cover = max(1, int(wl.get("gaussian_tile_cover", 2)))
    ntx, nty = grid_size(width, height, tile_size)
    tiles: Dict[int, List[int]] = defaultdict(list)
    for gid in range(num_gaussians):
        tx = (gid * 17) % ntx
        ty = (gid * 13) % nty
        for dy in range(cover):
            yy = ty + dy
            if yy >= nty:
                break
            for dx in range(cover):
                xx = tx + dx
                if xx >= ntx:
                    break
                tiles[yy * ntx + xx].append(gid)
    survival = float(config.get("hardware", {}).get("gfp", {}).get("culling_survival_rate", 1.0))
    survival = min(1.0, max(1e-6, survival))
    num_scene = int(math.ceil(num_gaussians / survival))
    return _plan_from_fine_tiles(
        config,
        frame_id=frame_id,
        width=width,
        height=height,
        tile_size=tile_size,
        chunk_size=chunk_size,
        fine_ids=tiles,
        num_visible=num_gaussians,
        num_scene=num_scene,
    )


def _plan_from_fine_tiles(
    config: dict,
    frame_id: int,
    width: int,
    height: int,
    tile_size: int,
    chunk_size: int,
    fine_ids: Dict[int, List[int]],
    num_visible: int,
    num_scene: int,
) -> FramePlan:
    algo = config.get("algorithm", {})
    down = max(1, int(algo.get("coarse_downscale", 4)))
    hls = bool(algo.get("hls_enabled", True))
    scar = bool(algo.get("scar_enabled", True)) and hls
    ntx, nty = grid_size(width, height, tile_size)
    full_pixels = width * height

    coarse_w = math.ceil(width / down)
    coarse_h = math.ceil(height / down)
    ctx, cty = grid_size(coarse_w, coarse_h, tile_size)

    def _chunks(n: int) -> int:
        if n <= 0:
            return 0
        return math.ceil(n / chunk_size)

    fine_pixel_of = {
        tid: tile_pixels(tid % ntx, tid // ntx, width, height, tile_size)
        for tid in range(ntx * nty)
    }

    coarse_tasks: List[TileTask] = []
    if hls:
        for cy in range(cty):
            for cx in range(ctx):
                cid = cy * ctx + cx
                ids = set()
                tx0, ty0 = cx * down, cy * down
                tx1, ty1 = min(ntx, tx0 + down), min(nty, ty0 + down)
                for ty in range(ty0, ty1):
                    for tx in range(tx0, tx1):
                        ids.update(fine_ids.get(ty * ntx + tx, []))
                pix = tile_pixels(cx, cy, coarse_w, coarse_h, tile_size)
                if pix <= 0:
                    continue
                coarse_tasks.append(TileTask(
                    frame_id=frame_id,
                    tile_id=cid,
                    stage="coarse",
                    num_gaussians=len(ids),
                    num_chunks=_chunks(len(ids)),
                    active_pixels=pix,
                    gaussian_ids=sorted(ids),
                    sort=True,
                ))

    roi_fraction = float(algo.get("roi_fraction", 0.5))
    fallback_rate = float(algo.get("fallback_rate", 0.0))
    scar_skip = float(algo.get("scar_skip_ratio", 0.0)) if scar else 0.0
    roi_ids = centered_roi(ntx, nty, roi_fraction) if hls else set(range(ntx * nty))
    effective_roi = (len(roi_ids) / (ntx * nty)) if ntx * nty else 0.0

    # Expected fine workload over queries.
    # ROI tiles are always visited; SCAR drops stable pixels, fallback renders them fully.
    # Non-ROI tiles are visited only on fallback, so only that fraction is enqueued.
    uncertain = 1.0 - scar_skip
    roi_pixel_frac = (1.0 - fallback_rate) * uncertain + fallback_rate
    non_roi = [tid for tid in range(ntx * nty) if tid not in roi_ids]
    n_fallback = int(round(fallback_rate * len(non_roi))) if hls else 0
    fallback_ids = set(spread_ids(non_roi, n_fallback))

    fine_tasks: List[TileTask] = []
    fine_pixels = 0.0
    if hls:
        selected = [(tid, roi_pixel_frac) for tid in sorted(roi_ids)]
        selected += [(tid, 1.0) for tid in sorted(fallback_ids)]
    else:
        selected = [(tid, 1.0) for tid in range(ntx * nty)]
    for tid, frac in selected:
        if frac <= 0:
            continue
        pix = fine_pixel_of[tid] * frac
        if pix <= 0:
            continue
        ids = fine_ids.get(tid, [])
        fine_pixels += pix
        fine_tasks.append(TileTask(
            frame_id=frame_id,
            tile_id=tid,
            stage="fine",
            num_gaussians=len(ids),
            num_chunks=_chunks(len(ids)),
            active_pixels=pix,
            gaussian_ids=ids,
            sort=not hls,
        ))

    _annotate_reuse(coarse_tasks)
    _annotate_reuse(fine_tasks)

    scene_name = (config.get("simulation", {}).get("scene") or "")
    return FramePlan(
        frame_id=frame_id,
        width=width,
        height=height,
        coarse_width=coarse_w,
        coarse_height=coarse_h,
        tile_size=tile_size,
        num_visible=num_visible,
        num_scene=num_scene,
        coarse_tasks=coarse_tasks,
        fine_tasks=fine_tasks,
        run_coarse=hls,
        fine_sort=not hls,
        coarse_pixels=coarse_w * coarse_h if hls else 0,
        fine_pixels=fine_pixels,
        full_pixels=full_pixels,
        roi_fraction=effective_roi if hls else 1.0,
        fallback_rate=fallback_rate if hls else 0.0,
        scar_skip_ratio=scar_skip,
        scene=scene_name,
    )


def _plan_from_coverage(
    config: dict,
    frame_id: int,
    width: int,
    height: int,
    tile_size: int,
    chunk_size: int,
    counts,
    uncached,
    num_visible: int,
    num_scene: int,
    coarse_grids=None,
    rects=None,
) -> FramePlan:
    """Same task list as ``_plan_from_fine_tiles``, but counts come from rectangles.

    ``uncached`` is the miss count when every fine tile is visited in scan order.
    HLS selects a subset, so those fine misses are recomputed against the
    previous selected tile. Coarse tiles are a complete scan, so their grid is used as-is.
    """
    import numpy as np
    algo = config.get("algorithm", {})
    down = max(1, int(algo.get("coarse_downscale", 4)))
    hls = bool(algo.get("hls_enabled", True))
    scar = bool(algo.get("scar_enabled", True)) and hls
    nty, ntx = counts.shape
    full_pixels = width * height

    def _chunks(n: int) -> int:
        if n <= 0:
            return 0
        return math.ceil(n / chunk_size)

    coarse_w = math.ceil(width / down)
    coarse_h = math.ceil(height / down)
    coarse_tasks: List[TileTask] = []
    if hls and coarse_grids is not None:
        c_counts, c_uncached = coarse_grids
        cty, ctx = c_counts.shape
        for cy in range(cty):
            for cx in range(ctx):
                pix = tile_pixels(cx, cy, coarse_w, coarse_h, tile_size)
                if pix <= 0:
                    continue
                n = int(c_counts[cy, cx])
                coarse_tasks.append(TileTask(
                    frame_id=frame_id,
                    tile_id=cy * ctx + cx,
                    stage="coarse",
                    num_gaussians=n,
                    num_chunks=_chunks(n),
                    active_pixels=pix,
                    gaussian_ids=[],
                    uncached_gaussians=int(c_uncached[cy, cx]),
                    sort=True,
                ))

    roi_fraction = float(algo.get("roi_fraction", 0.5))
    fallback_rate = float(algo.get("fallback_rate", 0.0))
    scar_skip = float(algo.get("scar_skip_ratio", 0.0)) if scar else 0.0
    roi_ids = centered_roi(ntx, nty, roi_fraction) if hls else set(range(ntx * nty))
    effective_roi = (len(roi_ids) / (ntx * nty)) if ntx * nty else 0.0
    uncertain = 1.0 - scar_skip
    roi_pixel_frac = (1.0 - fallback_rate) * uncertain + fallback_rate
    non_roi = [tid for tid in range(ntx * nty) if tid not in roi_ids]
    n_fallback = int(round(fallback_rate * len(non_roi))) if hls else 0
    fallback_ids = set(spread_ids(non_roi, n_fallback))
    if hls:
        selected = [(tid, roi_pixel_frac) for tid in sorted(roi_ids)]
        selected += [(tid, 1.0) for tid in sorted(fallback_ids)]
    else:
        selected = [(tid, 1.0) for tid in range(ntx * nty)]

    fine_uncached = uncached
    if hls:
        fine_uncached = _uncached_subset([tid for tid, _frac in selected], counts, rects)

    fine_tasks: List[TileTask] = []
    fine_pixels = 0.0
    for tid, frac in selected:
        if frac <= 0:
            continue
        tx, ty = tid % ntx, tid // ntx
        pix = tile_pixels(tx, ty, width, height, tile_size) * frac
        if pix <= 0:
            continue
        n = int(counts[ty, tx])
        fine_pixels += pix
        miss = int(fine_uncached[ty, tx]) if not hls else int(fine_uncached[tid])
        fine_tasks.append(TileTask(
            frame_id=frame_id,
            tile_id=tid,
            stage="fine",
            num_gaussians=n,
            num_chunks=_chunks(n),
            active_pixels=pix,
            gaussian_ids=[],
            uncached_gaussians=miss,
            sort=not hls,
        ))

    scene_name = (config.get("simulation", {}).get("scene") or "")
    return FramePlan(
        frame_id=frame_id,
        width=width,
        height=height,
        coarse_width=coarse_w,
        coarse_height=coarse_h,
        tile_size=tile_size,
        num_visible=num_visible,
        num_scene=num_scene,
        coarse_tasks=coarse_tasks,
        fine_tasks=fine_tasks,
        run_coarse=hls,
        fine_sort=not hls,
        coarse_pixels=coarse_w * coarse_h if hls else 0,
        fine_pixels=fine_pixels,
        full_pixels=full_pixels,
        roi_fraction=effective_roi if hls else 1.0,
        fallback_rate=fallback_rate if hls else 0.0,
        scar_skip_ratio=scar_skip,
        scene=scene_name,
    )


def _uncached_subset(tile_ids: List[int], counts, rects) -> Dict[int, int]:
    """Miss count versus the previous selected tile, not the geometric neighbor."""
    import numpy as np
    nty, ntx = counts.shape
    tx0, tx1, ty0, ty1 = rects
    misses: Dict[int, int] = {}
    prev = None
    pairs = []
    for tid in tile_ids:
        tx, ty = tid % ntx, tid // ntx
        n = int(counts[ty, tx])
        if n == 0 or prev is None or tx0.size == 0:
            misses[tid] = n
            prev = None if n == 0 else (tx, ty)
            continue
        pairs.append((tid, prev[0], prev[1], tx, ty, n))
        prev = (tx, ty)
    if not pairs:
        return misses
    ptx = np.array([p[1] for p in pairs], dtype=np.int32)
    pty = np.array([p[2] for p in pairs], dtype=np.int32)
    ctx = np.array([p[3] for p in pairs], dtype=np.int32)
    cty = np.array([p[4] for p in pairs], dtype=np.int32)
    both = np.empty(len(pairs), dtype=np.int64)
    # A few dozen tile-pairs at a time. Each pair is one pass over the rectangles.
    chunk = 32
    for start in range(0, len(pairs), chunk):
        stop = min(len(pairs), start + chunk)
        lo_x = np.minimum(ptx[start:stop, None], ctx[start:stop, None])
        hi_x = np.maximum(ptx[start:stop, None], ctx[start:stop, None])
        lo_y = np.minimum(pty[start:stop, None], cty[start:stop, None])
        hi_y = np.maximum(pty[start:stop, None], cty[start:stop, None])
        hit = (
            (tx0[None, :] <= lo_x) & (tx1[None, :] >= hi_x)
            & (ty0[None, :] <= lo_y) & (ty1[None, :] >= hi_y)
        )
        both[start:stop] = hit.sum(axis=1)
    for i, (tid, _ptx, _pty, _tx, _ty, n) in enumerate(pairs):
        misses[tid] = n - int(both[i])
    return misses


def _annotate_reuse(tasks: List[TileTask]) -> None:
    """Miss count against the previous tile in scan order.

    Computed once so PIM and ASIC-only runs report the same geometry traffic.
    A runtime cache would follow whichever tile won the DRAM bus, and that
    order changes when feature readback is added.
    """
    prev = set()
    for task in sorted(tasks, key=lambda t: t.tile_id):
        if not task.gaussian_ids:
            task.uncached_gaussians = task.num_gaussians
            prev = set()
            continue
        ids = set(task.gaussian_ids)
        task.uncached_gaussians = len(ids - prev)
        prev = ids


def _project_3d_to_2d(means3d, projmatrix, width: int, height: int):
    import torch
    ones = torch.ones((means3d.shape[0], 1), device=means3d.device, dtype=means3d.dtype)
    p_hom = torch.cat([means3d, ones], dim=-1) @ projmatrix.T
    p_w = 1.0 / (p_hom[:, 3:4] + 1e-7)
    p_proj = p_hom[:, :3] * p_w
    x_pix = ((p_proj[:, 0] + 1.0) * width - 1.0) * 0.5
    y_pix = ((p_proj[:, 1] + 1.0) * height - 1.0) * 0.5
    return torch.stack([x_pix, y_pix], dim=1)


def _rect_bounds(means2d, radii, width: int, height: int, tile_size: int):
    """Inclusive tile ranges of every visible Gaussian. Does not expand pairs."""
    import numpy as np
    if hasattr(means2d, "detach"):
        means2d = means2d.detach().cpu().numpy()
    if hasattr(radii, "detach"):
        radii = radii.detach().cpu().numpy()
    ntx, nty = grid_size(width, height, tile_size)
    visible = np.asarray(radii) > 0
    means = np.asarray(means2d)[visible]
    rads = np.asarray(radii)[visible].astype(np.float64, copy=False)
    if means.shape[0] == 0:
        empty = np.zeros(0, dtype=np.int32)
        return ntx, nty, empty, empty, empty, empty
    x = np.nan_to_num(means[:, 0].astype(np.float64, copy=False), nan=-1e9, posinf=1e9, neginf=-1e9)
    y = np.nan_to_num(means[:, 1].astype(np.float64, copy=False), nan=-1e9, posinf=1e9, neginf=-1e9)
    tx0 = np.clip(np.floor(np.maximum(0.0, x - rads) / tile_size), 0, ntx - 1).astype(np.int32)
    tx1 = np.clip(np.floor(np.minimum(width - 1, x + rads) / tile_size), 0, ntx - 1).astype(np.int32)
    ty0 = np.clip(np.floor(np.maximum(0.0, y - rads) / tile_size), 0, nty - 1).astype(np.int32)
    ty1 = np.clip(np.floor(np.minimum(height - 1, y + rads) / tile_size), 0, nty - 1).astype(np.int32)
    return ntx, nty, tx0, tx1, ty0, ty1


def _rect_counts(tx0, tx1, ty0, ty1, ntx: int, nty: int):
    import numpy as np
    diff = np.zeros((nty + 1, ntx + 1), dtype=np.int32)
    if tx0.size == 0:
        return diff[:nty, :ntx]
    np.add.at(diff, (ty0, tx0), 1)
    np.add.at(diff, (ty0, tx1 + 1), -1)
    np.add.at(diff, (ty1 + 1, tx0), -1)
    np.add.at(diff, (ty1 + 1, tx1 + 1), 1)
    return np.cumsum(np.cumsum(diff, axis=0), axis=1)[:nty, :ntx]


def _uncached_scan(tx0, tx1, ty0, ty1, counts):
    """Misses against the previous tile in scan order, including empty-tile resets.

    A Gaussian stays cached only when the previous tile is inside its rectangle.
    Along a row that is the left edge of the rectangle. At the start of a row it
    is the Gaussians that also cover the last tile of the previous row.
    """
    import numpy as np
    nty, ntx = counts.shape
    edge = np.zeros((nty + 1, ntx), dtype=np.int32)
    if tx0.size:
        np.add.at(edge, (ty0, tx0), 1)
        np.add.at(edge, (ty1 + 1, tx0), -1)
    left = np.cumsum(edge, axis=0)[:nty]
    uncached = left.copy()
    uncached[0, 0] = counts[0, 0]
    full = (tx0 == 0) & (tx1 == ntx - 1) if tx0.size else np.zeros(0, dtype=bool)
    row = np.zeros(nty + 1, dtype=np.int32)
    if np.any(full):
        np.add.at(row, ty0[full] + 1, 1)
        np.add.at(row, ty1[full] + 1, -1)
    both = np.cumsum(row)[:nty]
    if nty > 1:
        uncached[1:, 0] = counts[1:, 0] - both[1:]
    flat_c = counts.ravel()
    flat_u = uncached.ravel()
    reset = np.empty(flat_c.size, dtype=bool)
    reset[0] = True
    if flat_c.size > 1:
        reset[1:] = flat_c[:-1] == 0
    flat_u[reset] = flat_c[reset]
    return uncached


def _coverage_grids(means2d, radii, width: int, height: int, tile_size: int, down: int, want_coarse: bool):
    """Per-tile Gaussian counts and scan-order miss counts. No per-tile id lists."""
    import numpy as np
    ntx, nty, tx0, tx1, ty0, ty1 = _rect_bounds(means2d, radii, width, height, tile_size)
    counts = _rect_counts(tx0, tx1, ty0, ty1, ntx, nty)
    uncached = _uncached_scan(tx0, tx1, ty0, ty1, counts)
    coarse = None
    if want_coarse:
        coarse_w = math.ceil(width / down)
        coarse_h = math.ceil(height / down)
        ctx, cty = grid_size(coarse_w, coarse_h, tile_size)
        if tx0.size:
            cx0 = np.clip(tx0 // down, 0, max(0, ctx - 1))
            cx1 = np.clip(tx1 // down, 0, max(0, ctx - 1))
            cy0 = np.clip(ty0 // down, 0, max(0, cty - 1))
            cy1 = np.clip(ty1 // down, 0, max(0, cty - 1))
            c_counts = _rect_counts(cx0, cx1, cy0, cy1, ctx, cty)
            c_uncached = _uncached_scan(cx0, cx1, cy0, cy1, c_counts)
        else:
            c_counts = np.zeros((cty, ctx), dtype=np.int32)
            c_uncached = np.zeros((cty, ctx), dtype=np.int32)
        coarse = (c_counts, c_uncached)
    return counts, uncached, coarse, (tx0, tx1, ty0, ty1), int(tx0.size)


def load_workload_from_scene(
    config: dict,
    tile_size: int,
    chunk_size: int,
    verbose: bool = True,
) -> Optional[List[FramePlan]]:
    """Project one LangSplatV2 checkpoint. Returns None on any failure."""
    sim = config.get("simulation", {})
    model_path = sim.get("model_path") or ""
    source_path = sim.get("source_path") or ""
    if not model_path or not os.path.isdir(model_path):
        if verbose:
            print("[workload] no model_path, using synthetic field")
        return None
    if not source_path or not os.path.isdir(source_path):
        if verbose:
            print(f"[workload] source_path missing: {source_path}")
        return None
    try:
        import torch
        from argparse import ArgumentParser, Namespace
        from arguments import ModelParams, PipelineParams, OptimizationParams
        from scene import Scene
        from scene.gaussian_model import GaussianModel
        from gaussian_renderer import render
    except Exception as exc:
        if verbose:
            print(f"[workload] import failed ({exc}), using synthetic field")
        return None

    parser = ArgumentParser(add_help=False)
    model_group = ModelParams(parser, sentinel=True)
    pipe_group = PipelineParams(parser)
    opt_group = OptimizationParams(parser)
    args = parser.parse_args([
        "-s", source_path,
        "-m", model_path,
        "--resolution", "-1",
        "--eval",
        "--feature_level", str(sim.get("feature_level", 1)),
    ])
    cfg_file = os.path.join(model_path, "cfg_args")
    if os.path.isfile(cfg_file):
        try:
            with open(cfg_file, "r", encoding="utf-8") as f:
                cfg = eval(f.read())
            for k, v in vars(cfg).items():
                if k not in ("model_path", "source_path") and hasattr(args, k):
                    setattr(args, k, v)
        except Exception:
            pass
    # Training cfg_args may set eval=False or a downsample resolution.
    # Inference follows resolution=-1 and the held-out test split.
    args.resolution = -1
    args.eval = True
    args.model_path = model_path
    args.source_path = source_path
    try:
        dataset = model_group.extract(args)
        pipe = pipe_group.extract(args)
        opt = opt_group.extract(args)
        opt.include_feature = False
        opt.quick_render = False
        opt.topk = 1
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=sim.get("iteration", -1), shuffle=False)
    except Exception as exc:
        if verbose:
            print(f"[workload] failed to load scene ({exc})")
        return None

    split = sim.get("camera_split", "test")
    cameras = scene.getTestCameras() if split == "test" else scene.getTrainCameras()
    if not cameras:
        cameras = scene.getTrainCameras() or scene.getTestCameras()
    if not cameras:
        if verbose:
            print("[workload] no cameras")
        return None
    frames_cfg = sim.get("camera_ids", sim.get("frames", 0))
    if isinstance(frames_cfg, str) and frames_cfg.strip().lower() in ("all", "*"):
        frame_ids = list(range(len(cameras)))
    elif isinstance(frames_cfg, int):
        frame_ids = [frames_cfg]
    else:
        frame_ids = [int(x) for x in str(frames_cfg).replace(" ", "").split(",") if x != ""]
    num_scene = int(gaussians.get_xyz.shape[0])
    note = sim.get("checkpoint_note")
    if verbose and note:
        print(f"[workload] {note}: {model_path}")
    if verbose:
        print(f"[workload] {num_scene} gaussians, {len(frame_ids)} {split} camera(s)")

    bg = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg, dtype=torch.float32, device="cuda")
    plans: List[FramePlan] = []
    for fid in frame_ids:
        if fid < 0 or fid >= len(cameras):
            continue
        view = cameras[fid]
        try:
            with torch.no_grad():
                out = render(view, gaussians, pipe, background, opt)
        except Exception as exc:
            if verbose:
                print(f"[workload] render camera {fid} failed ({exc})")
            continue
        radii = out.get("radii")
        if radii is None:
            continue
        width, height = int(view.image_width), int(view.image_height)
        means3d = gaussians.get_xyz
        proj = view.full_proj_transform
        if proj.device.type != "cuda":
            proj = proj.cuda()
        means2d = _project_3d_to_2d(means3d, proj, width, height)
        hls = bool(config.get("algorithm", {}).get("hls_enabled", True))
        down = max(1, int(config.get("algorithm", {}).get("coarse_downscale", 4)))
        counts, uncached, coarse, rects, visible = _coverage_grids(
            means2d, radii, width, height, tile_size, down, hls,
        )
        if verbose:
            print(
                f"[workload] camera {fid}: visible={visible}/{num_scene} "
                f"{width}x{height}",
                flush=True,
            )
        saved = config.get("simulation", {}).get("resolution")
        config.setdefault("simulation", {})["resolution"] = f"{width}x{height}"
        plan = _plan_from_coverage(
            config, fid, width, height, tile_size, chunk_size,
            counts, uncached, visible, num_scene, coarse, rects,
        )
        if saved:
            config["simulation"]["resolution"] = saved
        plans.append(plan)
    return plans or None
