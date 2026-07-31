#!/usr/bin/env python3
"""Per-forward latency breakdown for LangSplatV2 LERF quick_render path.

Loads scene / VQ sparse weights once, then times only the recurring forward
stages for each annotated eval view:
  text.encode -> render.setup -> render.raster -> decode.* -> relevancy.*

Usage (from repo root, gs env active):
  python scripts/latency/profile_lerf_forward.py \\
      -s data/lerf_ovs/teatime -m output/teatime_0_1 \\
      --dataset_name teatime --index 0 --checkpoint 10000 \\
      --ckpt_root_path output --json_folder data/lerf_ovs/label \\
      --include_feature --quick_render --topk 4
"""

from __future__ import annotations

import os
import sys
from argparse import ArgumentParser
from pathlib import Path

import torch
from tqdm import tqdm

# repo root on path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from arguments import ModelParams, PipelineParams, get_combined_args
from eval.openclip_encoder import OpenCLIPNetwork
from eval_lerf import eval_gt_lerfdata
from gaussian_renderer import GaussianModel, render
from scene import Scene
from utils.general_utils import safe_state
from utils.latency_utils import LatencyProfiler
from utils.vq_utils import get_weights_and_indices


def _load_combined_gaussians(dataset, args, pipeline, profiler: LatencyProfiler):
    """Load level-0 scene once; attach 3-level sparse VQ weights/indices/codebooks."""
    combined = GaussianModel(dataset.sh_degree)
    dataset.model_path = args.ckpt_paths[0]
    with profiler.scope("load_scene"):
        scene = Scene(dataset, combined, shuffle=False)
        views = scene.getTrainCameras()
        ckpt0 = os.path.join(args.ckpt_paths[0], f"chkpnt{args.checkpoint}.pth")
        model_params, _ = torch.load(ckpt0, weights_only=False)
        combined.restore(model_params, args, mode="test")

    language_feature_codebooks = []
    language_feature_weights = []
    language_feature_indices = []
    with profiler.scope("prep.vq_sparse"):
        for level_idx in range(3):
            gaussians = GaussianModel(dataset.sh_degree)
            ckpt = os.path.join(args.ckpt_paths[level_idx], f"chkpnt{args.checkpoint}.pth")
            model_params, _ = torch.load(ckpt, weights_only=False)
            gaussians.restore(model_params, args, mode="test")
            language_feature_codebooks.append(
                gaussians._language_feature_codebooks.view(-1, 512)
            )
            weights, indices = get_weights_and_indices(
                gaussians._language_feature_logits, 4
            )
            language_feature_weights.append(weights)
            language_feature_indices.append(
                indices
                + int(level_idx * gaussians._language_feature_codebooks.shape[1])
            )
            del gaussians
        combined._language_feature_codebooks = torch.stack(
            language_feature_codebooks, dim=0
        )
        combined._language_feature_weights = torch.cat(
            language_feature_weights, dim=1
        )
        idxs = torch.cat(language_feature_indices, dim=1)
        combined._language_feature_indices = torch.from_numpy(
            idxs.detach().cpu().numpy()
        ).to(combined._language_feature_weights.device)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    return combined, views, background


def _forward_once(
    gaussians,
    view,
    pipeline,
    background,
    args,
    clip_model,
    prompts,
    profiler: LatencyProfiler,
):
    """One view forward with stage timing. Returns feature map [3,H,W,512]."""
    with profiler.scope("text.encode"):
        clip_model.set_positives(prompts)

    with torch.no_grad():
        output = render(
            view, gaussians, pipeline, background, args, profiler=profiler
        )
        language_feature_weight_map = output["language_feature_weight_map"]

        with profiler.scope("decode.einsum"):
            D, H, W = language_feature_weight_map.shape
            weight_map = language_feature_weight_map.view(3, 64, H, W).view(
                3, 64, H * W
            )
            codebooks = gaussians._language_feature_codebooks.permute(0, 2, 1)
            language_feature_map = torch.einsum(
                "ldk,lkn->ldn", codebooks, weight_map
            ).view(3, 512, H, W)

        with profiler.scope("decode.normalize"):
            language_feature_map = language_feature_map / (
                language_feature_map.norm(dim=1, keepdim=True) + 1e-10
            )

        restored = language_feature_map.permute(0, 2, 3, 1)

        with profiler.scope("relevancy.clip_match"):
            _ = clip_model.get_max_across_quick(restored)

    return restored


def main():
    parser = ArgumentParser(description="LangSplatV2 LERF forward latency")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--ckpt_root_path", default="output", type=str)
    parser.add_argument("--include_feature", action="store_true")
    parser.add_argument("--quick_render", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--index", type=str, default="0")
    parser.add_argument("--json_folder", type=str, default="data/lerf_ovs/label")
    parser.add_argument("--checkpoint", type=int, default=10000)
    parser.add_argument("--topk", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument(
        "--max_views",
        type=int,
        default=-1,
        help="Limit timed eval views (-1 = all annotated frames)",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Where to write latency_breakdown.{txt,json}",
    )
    parser.add_argument(
        "--ncu_mode",
        action="store_true",
        help="Minimal forwards for Nsight Compute (warmup=0, few views)",
    )
    parser.add_argument(
        "--queries_per_view",
        type=int,
        default=-1,
        help="Use only the first N GT phrases per view (-1 = all). "
        "Use 1 for paper-style single-query latency.",
    )

    args = get_combined_args(parser)
    args.include_feature = True
    args.quick_render = True
    args.ckpt_paths = [
        os.path.join(
            args.ckpt_root_path, args.dataset_name + f"_{args.index}_{level}"
        )
        for level in [1, 2, 3]
    ]
    args.json_folder = os.path.join(args.json_folder, args.dataset_name)

    out_dir = args.out_dir or os.path.join(
        "eval_result", f"{args.dataset_name}_{args.index}", "latency"
    )
    os.makedirs(out_dir, exist_ok=True)

    safe_state(args.quiet)
    profiler = LatencyProfiler(enabled=True)
    device = torch.device("cuda")

    dataset = model.extract(args)
    pipe = pipeline.extract(args)

    with profiler.scope("load_gt_ann"):
        gt_ann, _, _ = eval_gt_lerfdata(Path(args.json_folder), Path(out_dir) / "gt_tmp")

    eval_index_list = [int(idx) for idx in list(gt_ann.keys())]
    if args.ncu_mode:
        args.warmup = 0
        if args.max_views < 0:
            args.max_views = 2
    if args.max_views > 0:
        eval_index_list = eval_index_list[: args.max_views]

    with profiler.scope("load_clip"):
        clip_model = OpenCLIPNetwork(device)

    gaussians, views, background = _load_combined_gaussians(
        dataset, args, pipe, profiler
    )

    def _prompts_for(idx: int):
        prompts = list(gt_ann[f"{idx}"].keys())
        n = int(getattr(args, "queries_per_view", -1) or -1)
        if n > 0:
            prompts = prompts[:n]
        if not prompts:
            raise RuntimeError(f"No prompts for view idx={idx}")
        return prompts

    # Warmup (not recorded in focus averages via separate throwaway profiler)
    if args.warmup > 0:
        warm_idx = eval_index_list[0]
        warm_prompts = _prompts_for(warm_idx)
        for _ in range(args.warmup):
            _forward_once(
                gaussians,
                views[warm_idx],
                pipe,
                background,
                args,
                clip_model,
                warm_prompts,
                LatencyProfiler(enabled=False),
            )
        torch.cuda.synchronize()

    n_queries = 0
    prompts_per_fwd = []
    for idx in tqdm(eval_index_list, desc=f"latency/{args.dataset_name}"):
        prompts = _prompts_for(idx)
        prompts_per_fwd.append(len(prompts))
        n_queries += len(prompts)
        _forward_once(
            gaussians,
            views[idx],
            pipe,
            background,
            args,
            clip_model,
            prompts,
            profiler,
        )

    avg_p = (sum(prompts_per_fwd) / len(prompts_per_fwd)) if prompts_per_fwd else 0.0
    profiler.add_meta(
        dataset_name=args.dataset_name,
        checkpoint=args.checkpoint,
        quick_render=True,
        num_forwards=len(eval_index_list),
        num_queries=n_queries,
        queries_per_view=int(getattr(args, "queries_per_view", -1) or -1),
        avg_queries_per_forward=round(avg_p, 3),
        warmup=args.warmup,
        ncu_mode=bool(args.ncu_mode),
        image_hw=f"{views[eval_index_list[0]].image_height}x{views[eval_index_list[0]].image_width}",
        num_gaussians=int(gaussians.get_xyz.shape[0]),
        scales_with_P="text.encode, relevancy.clip_match",
        independent_of_P="render.setup, render.raster, decode.einsum, decode.normalize",
    )
    report = os.path.join(out_dir, "latency_breakdown.txt")
    profiler.print_summary()
    profiler.save(report)


if __name__ == "__main__":
    main()
