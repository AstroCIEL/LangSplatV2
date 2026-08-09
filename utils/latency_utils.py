"""CUDA-aware latency profiling helpers for LangSplatV2 inference."""

from __future__ import annotations

import json
import time
from collections import OrderedDict
from datetime import datetime
from typing import Dict, List, Optional

import torch


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


class LatencyProfiler:
    """Accumulate wall time (with CUDA sync) per named stage."""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.totals: Dict[str, float] = OrderedDict()
        self.counts: Dict[str, int] = OrderedDict()
        self._stack: List[tuple] = []
        self.meta: Dict[str, object] = OrderedDict()

    def add_meta(self, **kwargs):
        self.meta.update(kwargs)

    def start(self, name: str):
        if not self.enabled:
            return
        _sync()
        self._stack.append((name, time.perf_counter()))

    def stop(self, name: Optional[str] = None):
        if not self.enabled:
            return 0.0
        _sync()
        t1 = time.perf_counter()
        if not self._stack:
            return 0.0
        started_name, t0 = self._stack.pop()
        if name is not None and name != started_name:
            raise RuntimeError(
                f"LatencyProfiler stop mismatch: expected {started_name}, got {name}"
            )
        dt = t1 - t0
        self.totals[started_name] = self.totals.get(started_name, 0.0) + dt
        self.counts[started_name] = self.counts.get(started_name, 0) + 1
        return dt

    def record(self, name: str, dt: float):
        if not self.enabled:
            return
        self.totals[name] = self.totals.get(name, 0.0) + dt
        self.counts[name] = self.counts.get(name, 0) + 1

    class _Scope:
        def __init__(self, profiler: "LatencyProfiler", name: str):
            self.profiler = profiler
            self.name = name
            self.dt = 0.0

        def __enter__(self):
            self.profiler.start(self.name)
            return self

        def __exit__(self, exc_type, exc, tb):
            self.dt = self.profiler.stop(self.name)
            return False

    def scope(self, name: str) -> "_Scope":
        return self._Scope(self, name)

    def _group_totals(self, prefixes) -> float:
        if isinstance(prefixes, str):
            prefixes = (prefixes,)
        return sum(
            v
            for k, v in self.totals.items()
            if any(k.startswith(p) for p in prefixes)
        )

    def _format_rows(self, names: List[str], denom: float) -> List[str]:
        lines = []
        for name in names:
            tot = self.totals[name]
            cnt = self.counts.get(name, 0)
            avg_ms = (tot / cnt * 1000.0) if cnt else 0.0
            pct = tot / denom * 100.0 if denom > 0 else 0.0
            lines.append(
                f"{name:<40} {tot:10.4f} {cnt:8d} {avg_ms:10.2f} {pct:7.1f}%"
            )
        return lines

    def summary_lines(self) -> List[str]:
        total = sum(self.totals.values()) or 1e-12
        lines = []
        lines.append("=" * 80)
        lines.append("Latency breakdown (CUDA-synchronized wall time)")
        lines.append(f"Time: {datetime.now().isoformat(timespec='seconds')}")
        for k, v in self.meta.items():
            lines.append(f"  {k}: {v}")

        # Per-forward focus stages (excludes load_* / prep.vq / I/O)
        focus_prefixes = ("text.", "render.", "decode.", "relevancy.", "fuse.")
        focus_names = [k for k in self.totals if any(k.startswith(p) for p in focus_prefixes)]
        focus_tot = self._group_totals(focus_prefixes) or 1e-12

        header = f"{'stage':<40} {'total_s':>10} {'count':>8} {'avg_ms':>10} {'pct':>8}"

        lines.append("-" * 80)
        lines.append("FOCUS: per-forward stages (excludes load_* / prep.vq_* / I/O)")
        lines.append(f"  focus_total: {focus_tot:.4f} s")
        lines.append("-" * 80)
        lines.append(header)
        lines.append("-" * 80)
        lines.extend(self._format_rows(focus_names, focus_tot))

        n_fwd = int(self.meta.get("num_forwards", 0) or 0)
        lines.append("-" * 80)
        if n_fwd and focus_tot > 0:
            lines.append(
                f"  forward avg:  {focus_tot / n_fwd * 1000:.3f} ms  (n={n_fwd})"
            )
            for name in focus_names:
                lines.append(
                    f"    {name:<36} {self.totals[name] / n_fwd * 1000:8.3f} ms/fwd"
                )

        lines.append("")
        lines.append("=" * 80)
        lines.append("ALL stages (includes one-time load_* / prep.vq_*)")
        lines.append("-" * 80)
        lines.append(header)
        lines.append("-" * 80)
        for name, tot in self.totals.items():
            cnt = self.counts.get(name, 0)
            avg_ms = (tot / cnt * 1000.0) if cnt else 0.0
            pct = tot / total * 100.0
            lines.append(
                f"{name:<40} {tot:10.4f} {cnt:8d} {avg_ms:10.2f} {pct:7.1f}%"
            )
        lines.append("-" * 80)
        lines.append(f"{'SUM(stages)':<40} {total:10.4f}")
        lines.append("=" * 80)

        lines.append("")
        lines.append("Notes")
        lines.append("  - FOCUS excludes load_* and prep.vq_* (scene-level, not per-view).")
        mode = str(self.meta.get("semantic_mode", "feature"))
        lines.append("  - Scaling vs number of text queries P (per view):")
        if mode == "codebook":
            lines.append(
                "      DEPENDS ON P:  text.encode, fuse.proj_text, fuse.match"
            )
            lines.append(
                "      INDEPENDENT:  render.*, fuse.pixel_norm (no HxWx512 decode)"
            )
            lines.append(
                "  - Exact Codebook-Domain Relevance Fusion: cos = (c·z_q)/sqrt(cGc^T)"
            )
            lines.append("  - fuse.proj_text: z_q = S q  in L-dim (L=64).")
            lines.append("  - fuse.pixel_norm: ||f|| via Gram G=SS^T in coefficient domain.")
            lines.append("  - fuse.match: soft-relevancy from codebook-domain cosines.")
        else:
            lines.append("      DEPENDS ON P:  text.encode, relevancy.clip_match")
            lines.append(
                "      INDEPENDENT:  render.setup, render.raster, decode.einsum, decode.normalize"
            )
            lines.append(
                "    (feature map is rendered once; only text encode + relevancy grow with P)"
            )
            lines.append("  - decode.einsum: codebook x weight-map -> 3x512 feature maps.")
            lines.append("  - decode.normalize: L2 normalize language features.")
            lines.append(
                "  - relevancy.clip_match: pixel features vs P text embeds + soft relevancy"
            )
        lines.append("  - text.encode: OpenCLIP text encode for the P query phrases.")
        lines.append("  - render.setup: gather Gaussian attrs + build rasterizer settings.")
        lines.append("  - render.raster: CUDA efficient-langsplat raster (sparse top-k weights).")
        lines.append("  - For GPU-kernel split of render.raster, see scripts/latency/ana_langsplatv2.sh")
        return lines

    def print_summary(self):
        if not self.enabled:
            return
        for line in self.summary_lines():
            print(line)

    def save(self, path: str):
        if not self.enabled:
            return
        lines = self.summary_lines()
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

        payload = {
            "meta": dict(self.meta),
            "totals_s": dict(self.totals),
            "counts": dict(self.counts),
            "avg_ms": {
                k: (self.totals[k] / self.counts[k] * 1000.0)
                if self.counts.get(k)
                else 0.0
                for k in self.totals
            },
        }
        json_path = path.rsplit(".", 1)[0] + ".json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"Latency report saved to: {path}")
        print(f"Latency JSON saved to:   {json_path}")
