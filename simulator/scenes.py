"""Sparsity statistics reported for LERF and 3D-OVS in the SILF paper.

ROI fraction, fallback rate, and SCAR skip ratio are query averages.
The published ROI fraction already includes tile dilation, so the planner
does not dilate again.
"""

import os
from typing import Dict, List, Optional


# num_queries is the mean number of text prompts matched per labeled test frame.
# LERF counts unique categories in each frame_*.json under label/<scene>/.
# 3D-OVS counts mask files in segmentations/<frame>/, including room's "wood wall".
SCENES: Dict[str, dict] = {
    # LERF inference resolution from the training images (resolution=-1, height already < 1080).
    # figurines frames: 17, 13, 11, 15
    "figurines": dict(dataset="lerf", width=986, height=728, roi_fraction=0.347, fallback_rate=0.11, scar_skip_ratio=0.68, num_queries=14.0),
    # teatime frames: 11, 11, 10, 11, 7, 9
    "teatime": dict(dataset="lerf", width=986, height=728, roi_fraction=0.400, fallback_rate=0.08, scar_skip_ratio=0.59, num_queries=59 / 6),
    # ramen frames: 10, 8, 8, 11, 12, 8, 14
    "ramen": dict(dataset="lerf", width=986, height=728, roi_fraction=0.582, fallback_rate=0.07, scar_skip_ratio=0.55, num_queries=71 / 7),
    # kitchen frames: 6, 4, 3, 5, 4
    "kitchen": dict(dataset="lerf", width=986, height=728, roi_fraction=0.630, fallback_rate=0.13, scar_skip_ratio=0.61, num_queries=22 / 5),
    # 3D-OVS: originals are ~4032x3024; resolution=-1 scales height to 1080 → 1440x1080.
    "bed": dict(dataset="3d_ovs", width=1440, height=1080, roi_fraction=0.642, fallback_rate=0.15, scar_skip_ratio=0.65, num_queries=6),
    "lawn": dict(dataset="3d_ovs", width=1440, height=1080, roi_fraction=0.552, fallback_rate=0.16, scar_skip_ratio=0.52, num_queries=6),
    "room": dict(dataset="3d_ovs", width=1440, height=1080, roi_fraction=0.414, fallback_rate=0.06, scar_skip_ratio=0.68, num_queries=6),
    "sofa": dict(dataset="3d_ovs", width=1440, height=1080, roi_fraction=0.523, fallback_rate=0.10, scar_skip_ratio=0.61, num_queries=6),
    "bench": dict(dataset="3d_ovs", width=1440, height=1080, roi_fraction=0.562, fallback_rate=0.12, scar_skip_ratio=0.72, num_queries=7),
}

DATASETS: Dict[str, List[str]] = {
    "lerf": ["figurines", "teatime", "ramen", "kitchen"],
    "3d_ovs": ["bed", "lawn", "room", "sofa", "bench"],
}


def canonical_dataset(name: str) -> str:
    key = (name or "").strip().lower().replace("-", "_")
    if key in ("3dovs", "3d_ovs"):
        return "3d_ovs"
    return key


_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_LERF_ROOT = "/data/home/rh_xu30/data/lerf_ovs"
_OVS_ROOT = "/data/home/rh_xu30/data/3D_OVS"
# Paper name "kitchen" is the LERF scene waldo_kitchen.
_DISK_NAME = {"kitchen": "waldo_kitchen"}


def scene_checkpoint(name: str) -> Optional[dict]:
    """Language-model directory and COLMAP source for one paper scene.

    Bench has no language point cloud in output/bench_0_1 (training did not
    finish). That scene falls back to the RGB 3DGS at output/bench_rgb.
    """
    meta = SCENES.get(name)
    if not meta:
        return None
    disk = _DISK_NAME.get(name, name)
    if meta["dataset"] == "lerf":
        source = os.path.join(_LERF_ROOT, disk)
    else:
        source = os.path.join(_OVS_ROOT, disk)
    model = os.path.join(_REPO, "output", f"{disk}_0_1")
    note = ""
    if name == "bench" and not os.path.isdir(os.path.join(model, "point_cloud")):
        model = os.path.join(_REPO, "output", "bench_rgb")
        note = "language checkpoint missing; using RGB 3DGS"
    return {"model_path": model, "source_path": source, "note": note}
