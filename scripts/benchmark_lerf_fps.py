#!/usr/bin/env python3
"""Benchmark LERF labeled test frames: quick_render + CLIP query (end-to-end FPS)."""
import argparse
import glob
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Union

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from arguments import ModelParams, PipelineParams, get_combined_args
from eval.openclip_encoder import OpenCLIPNetwork
from gaussian_renderer import GaussianModel, render
from scene import Scene
from utils.general_utils import safe_state
from utils.vq_utils import get_weights_and_indices


def polygon_to_mask(img_shape, points_list):
    points = np.asarray(points_list, dtype=np.int32)
    mask = np.zeros(img_shape, dtype=np.uint8)
    cv2.fillPoly(mask, [points], 1)
    return mask


def stack_mask(mask_base, mask_add):
    mask = mask_base.copy()
    mask[mask_add != 0] = 1
    return mask


def eval_gt_lerfdata(json_folder: Union[str, Path] = None):
    gt_json_paths = sorted(glob.glob(os.path.join(str(json_folder), "frame_*.json")))
    img_paths = sorted(glob.glob(os.path.join(str(json_folder), "frame_*.jpg")))
    gt_ann = {}
    h, w = None, None
    for js_path in gt_json_paths:
        img_ann = defaultdict(dict)
        with open(js_path, "r") as f:
            gt_data = json.load(f)
        h, w = gt_data["info"]["height"], gt_data["info"]["width"]
        idx = int(gt_data["info"]["name"].split("_")[-1].split(".jpg")[0]) - 1
        for prompt_data in gt_data["objects"]:
            label = prompt_data["category"]
            box = np.asarray(prompt_data["bbox"]).reshape(-1)
            mask = polygon_to_mask((h, w), prompt_data["segmentation"])
            if img_ann[label].get("mask", None) is not None:
                mask = stack_mask(img_ann[label]["mask"], mask)
                img_ann[label]["bboxes"] = np.concatenate(
                    [img_ann[label]["bboxes"].reshape(-1, 4), box.reshape(-1, 4)], axis=0
                )
            else:
                img_ann[label]["bboxes"] = box
            img_ann[label]["mask"] = mask
        gt_ann[f"{idx}"] = img_ann
    return gt_ann, (h, w), img_paths


def render_language_feature_map_quick(gaussians, view, pipeline, background, args):
    with torch.no_grad():
        output = render(view, gaussians, pipeline, background, args)
        language_feature_weight_map = output["language_feature_weight_map"]
        d, height, width = language_feature_weight_map.shape
        language_feature_weight_map = language_feature_weight_map.view(3, 64, height, width).view(
            3, 64, height * width
        )
        language_codebooks = gaussians._language_feature_codebooks.permute(0, 2, 1)
        language_feature_map = torch.einsum(
            "ldk,lkn->ldn", language_codebooks, language_feature_weight_map
        ).view(3, 512, height, width)
        language_feature_map = language_feature_map / (
            language_feature_map.norm(dim=1, keepdim=True) + 1e-10
        )
    return language_feature_map


def prepare_frame(dataset, args, idx, gt_ann):
    """Mirror eval_lerf.evaluate_quick per-frame model setup."""
    combined_gaussians = GaussianModel(dataset.sh_degree)
    dataset.model_path = args.ckpt_paths[0]
    scene = Scene(dataset, combined_gaussians, shuffle=False)
    views = scene.getTrainCameras()
    view = views[idx]
    checkpoint = os.path.join(args.ckpt_paths[0], f"chkpnt{args.checkpoint}.pth")
    model_params, _ = torch.load(checkpoint, weights_only=False)
    combined_gaussians.restore(model_params, args, mode="test")

    language_feature_codebooks = []
    language_feature_weights = []
    language_feature_indices = []
    for level_idx in range(3):
        gaussians = GaussianModel(dataset.sh_degree)
        ckpt = os.path.join(args.ckpt_paths[level_idx], f"chkpnt{args.checkpoint}.pth")
        params, _ = torch.load(ckpt, weights_only=False)
        gaussians.restore(params, args, mode="test")
        language_feature_codebooks.append(gaussians._language_feature_codebooks.view(-1, 512))
        weights, indices = get_weights_and_indices(gaussians._language_feature_logits, 4)
        language_feature_weights.append(weights)
        language_feature_indices.append(
            indices + int(level_idx * gaussians._language_feature_codebooks.shape[1])
        )

    combined_gaussians._language_feature_codebooks = torch.stack(language_feature_codebooks, dim=0)
    combined_gaussians._language_feature_weights = torch.cat(language_feature_weights, dim=1)
    language_feature_indices = torch.cat(language_feature_indices, dim=1)
    combined_gaussians._language_feature_indices = torch.from_numpy(
        language_feature_indices.detach().cpu().numpy()
    ).to(combined_gaussians._language_feature_weights.device)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    prompts = list(gt_ann[f"{idx}"].keys())
    return combined_gaussians, view, background, prompts


@torch.no_grad()
def run_e2e(combined, view, pipeline, background, args, clip_model, prompts):
    language_feature_image = render_language_feature_map_quick(
        combined, view, pipeline, background, args
    )
    restored_feat = language_feature_image.permute(0, 2, 3, 1)
    clip_model.set_positives(prompts)
    clip_model.get_max_across_quick(restored_feat)


def benchmark_scene(dataset, pipeline, args, warmup: int, repeats: int):
    gt_ann, _, _ = eval_gt_lerfdata(Path(args.json_folder))
    eval_index_list = [int(idx) for idx in gt_ann.keys()]
    clip_model = OpenCLIPNetwork(torch.device("cuda"))

    ms_per_frame_all = []
    sample_view = None
    n_gaussians = None
    for idx in eval_index_list:
        combined, view, background, prompts = prepare_frame(dataset, args, idx, gt_ann)
        if sample_view is None:
            sample_view = view
            n_gaussians = int(combined.get_xyz.shape[0])

        for _ in range(warmup):
            run_e2e(combined, view, pipeline, background, args, clip_model, prompts)
        torch.cuda.synchronize()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(repeats):
            run_e2e(combined, view, pipeline, background, args, clip_model, prompts)
        end.record()
        torch.cuda.synchronize()
        ms_per_frame_all.append(start.elapsed_time(end) / repeats)

        del combined
        torch.cuda.empty_cache()

    mean_ms = sum(ms_per_frame_all) / len(ms_per_frame_all)
    return {
        "scene": args.dataset_name,
        "test_frames": len(eval_index_list),
        "resolution": f"{sample_view.image_width}x{sample_view.image_height}",
        "n_gaussians": n_gaussians,
        "warmup": warmup,
        "repeats_per_frame": repeats,
        "ms_per_frame": mean_ms,
        "fps": 1000.0 / mean_ms,
        "per_frame_ms": ms_per_frame_all,
    }


def seed_everything(seed_value: int):
    random.seed(seed_value)
    os.environ["PYTHONHASHSEED"] = str(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True


def main():
    parser = argparse.ArgumentParser(description="LERF test-set end-to-end FPS benchmark")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--ckpt_root_path", default="output", type=str)
    parser.add_argument("--include_feature", action="store_true")
    parser.add_argument("--quick_render", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--index", type=str, default="0")
    parser.add_argument("--json_folder", type=str, default="../../data/lerf_ovs/label")
    parser.add_argument("--checkpoint", type=int, default=10000)
    parser.add_argument("--topk", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=30)
    args = get_combined_args(parser)

    args.include_feature = True
    args.quick_render = True
    args.ckpt_paths = [
        os.path.join(args.ckpt_root_path, f"{args.dataset_name}_{args.index}_{level}")
        for level in [1, 2, 3]
    ]
    args.json_folder = os.path.join(args.json_folder, args.dataset_name)

    seed_everything(42)
    safe_state(args.quiet)

    dataset = model.extract(args)
    pipe = pipeline.extract(args)
    result = benchmark_scene(dataset, pipe, args, warmup=args.warmup, repeats=args.repeats)
    print(
        f"{result['scene']}: {result['fps']:.2f} FPS "
        f"({result['ms_per_frame']:.2f} ms/frame) | "
        f"test_frames={result['test_frames']} res={result['resolution']} "
        f"N={result['n_gaussians']:,} warmup={result['warmup']} "
        f"repeats/frame={result['repeats_per_frame']}"
    )


if __name__ == "__main__":
    main()
