# LangSplatV2 Environment Setup (local / Jetson Orin)

Reuse the existing OpenGaussian conda env `gs` (Python 3.10 + Torch 2.8.0 cu126 aarch64).
Do **not** use the upstream `environment.yml` (py3.7 / torch1.12 / CUDA11.6 — not available on Jetson).

## Activate
```bash
source ~/Work/LangSplatV2/scripts/activate_env.sh
# or manually:
source ~/miniforge3/etc/profile.d/conda.sh
conda activate gs
export PATH="$CONDA_PREFIX/bin:/usr/local/cuda-12.6/bin:$PATH"
export CUDA_HOME=/usr/local/cuda-12.6
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib/python3.10/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"
export TORCH_CUDA_ARCH_LIST=8.7
# If need network (Clash HTTP on this machine is :7890; SOCKS may be elsewhere):
export http_proxy="http://127.0.0.1:7890"
export https_proxy="http://127.0.0.1:7890"
```

## Installed into `gs` for this project
- Python deps: `open-clip-torch`, `tensorboard`, `jaxtyping`, `matplotlib`, `scikit-learn`, …
- Local wheels reused: `~/Downloads/.wheels/torch*-linux_aarch64.whl` (already in `gs`)
- CUDA extensions (built with `TORCH_CUDA_ARCH_LIST=8.7`, `CUDA_HOME=/usr/local/cuda-12.6`):
  - `submodules/efficient-langsplat-rasterization` → `diff_gaussian_rasterization`
  - `submodules/simple-knn` (patched `#include <float.h>`)
  - `submodules/segment-anything-langsplat`

## Layout
- Dataset (LERF): `~/data/lerf_ovs` → symlink to `~/Work/LangSplat/data/lerf_ovs`
  - scripts use `../../data/lerf_ovs` relative to this repo
- SAM ckpt: `ckpts/sam_vit_h_4b8939.pth` → LangSplat's copy
- RGB 3DGS ckpts: each scene `.../<scene>/output/{chkpnt30000.pth,cameras.json,...}` → LangSplat `output/<scene>_rgb/`
- Language features: reuse existing `language_features/` from LangSplat preprocess (same format)
- V2 pretrained weights: extract into `./output/` (e.g. `teatime_0_1`, ...)

## Latency (per-forward breakdown)
```bash
source scripts/activate_env.sh
bash scripts/latency/run_lerf_latency.sh            # all LERF scenes
# Optional NCU kernel split of render.raster (needs root):
sudo -E bash scripts/latency/ana_langsplatv2.sh teatime
```
Reports: `eval_result/<scene>_0/latency/latency_breakdown.{txt,json}`

## Pretrained weights download
```bash
export http_proxy="http://127.0.0.1:7890" https_proxy="http://127.0.0.1:7890"
source scripts/activate_env.sh
mkdir -p /tmp/langsplatv2_weights/LERF && cd /tmp/langsplatv2_weights/LERF
# LERF teatime (~6.5GB)
gdown "159jEgMG5SGeFbFcLoqFr7OpqW5yzctt3" -O teatime.tar.gz
# Other LERF scenes (optional):
# figurines: 1qKtjgTgt9xIkBsufiRZH6Mjo6DZSPqc_
# ramen:     12Zgqx_bXt7Hxi7vh3jko1AGvLedd96AX
# waldo:     18BDS9kp0ioCXrdBQrB04TO0fIaE8RowX
bash ~/Work/LangSplatV2/scripts/finish_weights.sh /tmp/langsplatv2_weights/LERF/teatime.tar.gz
```
