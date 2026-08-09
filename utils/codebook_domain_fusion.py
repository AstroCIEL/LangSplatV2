"""Exact Codebook-Domain Relevance Fusion (query-first semantic path).

Baseline feature-first:
  f = c @ S          # decode to D=512 feature map
  f = f / ||f||      # L2 normalize
  sim = f · q        # match

Fused codebook-domain (exact cosine):
  G = S @ S^T                    # LxL, scene-level
  z_q = S @ q                    # L, per query
  ||f|| = sqrt(c @ G @ c^T)      # per pixel in L-dim
  cos(f,q) = (c · z_q) / ||f||

Skips materializing the HxWx512 feature map for matching.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch


def precompute_gram(codebooks: torch.Tensor) -> torch.Tensor:
    """codebooks: [n_levels, L, D] -> G: [n_levels, L, L] = S S^T."""
    return torch.bmm(codebooks, codebooks.transpose(1, 2))


@torch.no_grad()
def relevancy_codebook_domain(
    weight_map: torch.Tensor,
    codebooks: torch.Tensor,
    gram: torch.Tensor,
    pos_embeds: torch.Tensor,
    neg_embeds: torch.Tensor,
    eps: float = 1e-10,
    profiler=None,
) -> torch.Tensor:
    """Soft relevancy maps without constructing D-dim features.

    Args:
        weight_map: [n_levels, L, H, W] or [n_levels*L, H, W] rendered coeffs
        codebooks: [n_levels, L, D]
        gram: [n_levels, L, L] precomputed S S^T
        pos_embeds: [P, D] L2-normalized
        neg_embeds: [N, D] L2-normalized
        profiler: optional LatencyProfiler for sub-stage timing

    Returns:
        relev_map: [n_levels, P, H, W]
    """
    n_levels, L, D = codebooks.shape
    if weight_map.dim() == 3:
        _, H, W = weight_map.shape
        c = weight_map.view(n_levels, L, H * W)
    else:
        n_levels2, L2, H, W = weight_map.shape
        assert n_levels2 == n_levels and L2 == L
        c = weight_map.view(n_levels, L, H * W)

    device = c.device
    dtype = c.dtype
    phrases = torch.cat([pos_embeds, neg_embeds], dim=0).to(device=device, dtype=dtype)
    n_phrases = pos_embeds.shape[0]

    def _scope(name):
        if profiler is None:
            from contextlib import nullcontext

            return nullcontext()
        return profiler.scope(name)

    with _scope("fuse.proj_text"):
        # z = S @ q  ->  [n_levels, L, P+N]
        z = torch.einsum("nld,pd->nlp", codebooks, phrases)

    with _scope("fuse.pixel_norm"):
        # ||f||^2 = c^T G c  per pixel
        Gc = torch.bmm(gram.to(dtype=dtype), c)
        norm = torch.sqrt((c * Gc).sum(dim=1).clamp_min(eps))  # [n_levels, HW]

    with _scope("fuse.match"):
        # dots: [n_levels, P+N, HW] = z^T @ c
        dots = torch.einsum("nlp,nlq->npq", z, c)
        sim = dots / norm.unsqueeze(1)

        # Align with get_max_across_quick layout: [n_levels, HW, P/N]
        pos_vals = sim[:, :n_phrases, :].permute(0, 2, 1)  # [n_levels, HW, P]
        neg_vals = sim[:, n_phrases:, :].permute(0, 2, 1)  # [n_levels, HW, N]
        prob_pos = torch.sigmoid(
            10.0 * (pos_vals.unsqueeze(-1) - neg_vals.unsqueeze(2))
        )  # [n_levels, HW, P, N]
        min_pos_prob = prob_pos.min(dim=-1).values  # [n_levels, HW, P]
        min_pos_prob = min_pos_prob.permute(0, 2, 1)  # [n_levels, P, HW]

    return min_pos_prob.view(n_levels, n_phrases, H, W)


@torch.no_grad()
def feature_first_relevancy(
    weight_map: torch.Tensor,
    codebooks: torch.Tensor,
    clip_model,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Baseline decode -> normalize -> match. Returns (relev, feature_nhwc)."""
    n_levels, L, D = codebooks.shape
    if weight_map.dim() == 3:
        _, H, W = weight_map.shape
        w = weight_map.view(n_levels, L, H * W)
    else:
        _, _, H, W = weight_map.shape
        w = weight_map.view(n_levels, L, H * W)

    cb = codebooks.permute(0, 2, 1)
    feat = torch.einsum("ldk,lkn->ldn", cb, w).view(n_levels, D, H, W)
    feat = feat / (feat.norm(dim=1, keepdim=True) + 1e-10)
    nhwc = feat.permute(0, 2, 3, 1)
    relev = clip_model.get_max_across_quick(nhwc)
    return relev, nhwc
