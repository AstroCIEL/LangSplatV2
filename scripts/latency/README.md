# LangSplatV2 Latency Breakdown

How we measure **per-forward** inference latency on Jetson Orin (and how to read the reports).

## What is measured

We time the **recurring online path** for one annotated camera view:

```
text.encode → render.setup → render.raster → [semantic path]
```

| Semantic mode | Stages after raster |
|---|---|
| `feature` (default / paper path) | `decode.einsum` → `decode.normalize` → `relevancy.clip_match` |
| `codebook` (Exact Codebook-Domain Relevance Fusion) | `fuse.proj_text` → `fuse.pixel_norm` → `fuse.match` |

**Not** counted in the FOCUS “forward avg” (one-time / scene-level):

- `load_gt_ann`, `load_clip`, `load_scene`, `prep.vq_sparse`, `prep.gram`

Those appear only in the ALL-stages section of the report.

### Timing method

- Implementation: `utils/latency_utils.py` (`LatencyProfiler`)
- Each stage wraps `torch.cuda.synchronize()` before start and after stop, then records wall time with `time.perf_counter()`
- Units in reports: **milliseconds per forward** (`ms/fwd`) = stage_total / `num_forwards`
- Warmup: default **3** untimed forwards on the first eval view (CUDA / allocator settle)

This is **CUDA-synced host wall time**, not Nsight kernel timers. For raster kernel split see [Optional NCU](#optional-ncu-raster-kernels) below.

### Views and queries (P)

- Eval views = frames that have GT annotations under `data/lerf_ovs/label/<scene>/`
- **P** = number of text phrases for that view
  - `--queries_per_view 1` → paper-style **single-query** latency (1q)
  - `--queries_per_view -1` → **all** GT phrases on that view (mq / multi-query)

Stages that grow with P:

| Mode | Depends on P | Independent of P |
|---|---|---|
| feature | `text.encode`, `relevancy.clip_match` | `render.*`, `decode.*` |
| codebook | `text.encode`, `fuse.proj_text`, `fuse.match` | `render.*`, `fuse.pixel_norm` |

---

## Stage glossary

### Shared

| Stage | Meaning |
|---|---|
| `text.encode` | OpenCLIP tokenize + encode positives; L2-normalize embeddings |
| `render.setup` | Gather Gaussian attrs / build raster settings |
| `render.raster` | Efficient LangSplat CUDA raster → sparse top-k **coefficient weight map** `[3×64, H, W]` |

### Feature-first (`--semantic_mode feature`)

| Stage | Meaning |
|---|---|
| `decode.einsum` | Codebook \(S\) × coeffs → language feature map **`[3, 512, H, W]`** |
| `decode.normalize` | Per-pixel L2 normalize along channel dim |
| `relevancy.clip_match` | Feature vs text/negatives → soft relevancy heatmap (`get_max_across_quick`) |

`relevancy.clip_match` uses NCHW `matmul` (not NHWC strided gather). See repo README “Local notes”.

### Codebook-domain fusion (`--semantic_mode codebook`)

Exact cosine without materializing the 512-d map:

\[
\cos(f_p,q)=\frac{c_p\,z_q}{\sqrt{c_p G c_p^\top}},\quad z_q=Sq,\quad G=SS^\top
\]

| Stage | Meaning |
|---|---|
| `prep.gram` | Once per scene: \(G=SS^\top\) (`[3,64,64]`) — **not** in FOCUS forward avg |
| `fuse.proj_text` | \(z_q = S q\) in \(L=64\) |
| `fuse.pixel_norm` | \(\|f\|=\sqrt{cGc^\top}\) per pixel |
| `fuse.match` | Codebook-domain dots + soft-relevancy (same sigmoid/min-neg form as feature path) |

Code: `utils/codebook_domain_fusion.py`. Synthetic check: max abs diff vs feature path \(\lt 2\times10^{-7}\).

---

## Prerequisites

```bash
cd ~/Work/LangSplatV2
source scripts/activate_env.sh

# Need:
#   output/<scene>_0_{1,2,3}/chkpnt10000.pth
#   data/lerf_ovs/<scene>/ + data/lerf_ovs/label/<scene>/
```

Scenes used here: `teatime`, `figurines`, `ramen`, `waldo_kitchen`.

---

## How to run

### A. Batch helper (feature path)

```bash
# Single-query (default): writes eval_result/<scene>_0/latency_1q/
bash scripts/latency/run_lerf_latency.sh
bash scripts/latency/run_lerf_latency.sh teatime   # one scene

# Multi-query (all GT phrases): writes eval_result/<scene>_0/latency/
MODE=mq bash scripts/latency/run_lerf_latency.sh
```

Summaries:

- `eval_result/lerf_latency_summary_1q.txt`
- `eval_result/lerf_latency_summary.txt` (mq)

### B. Direct profiler (full control)

```bash
python scripts/latency/profile_lerf_forward.py \
  -s data/lerf_ovs/teatime \
  -m output/teatime_0_1 \
  --dataset_name teatime --index 0 --checkpoint 10000 \
  --ckpt_root_path output \
  --json_folder data/lerf_ovs/label \
  --include_feature --quick_render --topk 4 \
  --warmup 3 \
  --queries_per_view 1 \
  --semantic_mode feature \
  --out_dir eval_result/teatime_0/latency_1q_feature \
  --quiet
```

Useful flags:

| Flag | Default | Notes |
|---|---|---|
| `--queries_per_view` | `-1` | `1` = 1q; `-1` = all phrases |
| `--semantic_mode` | `feature` | `feature` \| `codebook` |
| `--warmup` | `3` | Untimed forwards before measurement |
| `--max_views` | `-1` | Cap annotated views (`-1` = all) |
| `--out_dir` | `eval_result/<scene>_<index>/latency` | Report directory |

### C. Feature vs codebook comparison

```bash
for mode in feature codebook; do
  for qpv in 1 -1; do
    tag=$([[ $qpv -eq 1 ]] && echo 1q || echo mq)
    python scripts/latency/profile_lerf_forward.py \
      -s data/lerf_ovs/teatime -m output/teatime_0_1 \
      --dataset_name teatime --index 0 --checkpoint 10000 \
      --ckpt_root_path output --json_folder data/lerf_ovs/label \
      --include_feature --quick_render --topk 4 --warmup 3 \
      --queries_per_view "$qpv" --semantic_mode "$mode" \
      --out_dir "eval_result/teatime_0/latency_${tag}_${mode}" --quiet
  done
done
```

Reference compare dump from a full LERF sweep:

- `eval_result/codebook_fusion_latency_compare.txt`

Typical Orin result (order of magnitude): at **P=1**, codebook fusion cuts **semantic** ~3.8× and **forward** ~1.7–1.9× (render stays ~250–340 ms and becomes dominant after fusion).

---

## How to read a report

Files under each `--out_dir`:

- `latency_breakdown.txt` — human-readable
- `latency_breakdown.json` — same totals/meta for scripting

### FOCUS block (what to cite)

```
FOCUS: per-forward stages ...
  forward avg:  XXX.XXX ms  (n=<num_forwards>)
    text.encode              ...
    render.setup             ...
    render.raster            ...
    decode.* / fuse.*        ...
```

- **`forward avg`** = sum of FOCUS stages / number of timed views  
- **`pct`** = share of FOCUS total (not of wall-clock including CLIP load)
- Header meta includes `image_hw`, `num_gaussians`, `avg_queries_per_forward`, `semantic_mode`

### ALL block

Includes one-shot loads. Useful for setup cost; **do not** treat `load_clip` (~20s) as per-frame latency.

### Suggested “semantic” aggregate

For apples-to-apples post-raster cost:

- feature: `decode.einsum + decode.normalize + relevancy.clip_match`
- codebook: `fuse.proj_text + fuse.pixel_norm + fuse.match`

---

## Optional NCU (raster kernels)

Wall-clock `render.raster` is one blob. To split CUDA kernels:

```bash
# Needs root / Nsight Compute
sudo -E bash scripts/latency/ana_langsplatv2.sh teatime
```

See comments in `ana_langsplatv2.sh` / `ana_raster_kernels.py`. NCU runs are heavy and use a reduced forward count (`--ncu_mode` path).

---

## File map

| Path | Role |
|---|---|
| `profile_lerf_forward.py` | Main timed forward loop |
| `run_lerf_latency.sh` | Batch 1q / mq over LERF scenes |
| `utils/latency_utils.py` | Sync + stage accumulator + report writer |
| `utils/codebook_domain_fusion.py` | Codebook-domain fusion math |
| `ana_langsplatv2.sh` | NCU wrapper for raster |
| `eval_result/<scene>_0/latency*/` | Per-run reports |

---

## Caveats

1. **Device / resolution**: Numbers are Orin + default rescaling (&gt;1080P → 1080P). Different GPU/resolution changes absolute ms.
2. **P distribution**: mq “forward avg” mixes views with different P; check `avg_queries_per_forward` in the header.
3. **Power / clocks**: Jetson clocks can vary; use warmup and avoid thermal throttle mid-run for fair A/B.
4. **Fusion is software-only here**: same math intended for query-first / PIM dataflow in the paper outline; this folder only documents the **measured SW baseline**.
5. **Eval accuracy**: latency scripts force relevancy compute but discard masks; for mIoU use `eval_lerf.sh` / `eval_3d_ovs.sh`.
