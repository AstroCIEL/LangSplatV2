# LangSplatV2 Environment Setup (local)

## Conda env
```bash
source ~/miniforge3/etc/profile.d/conda.sh
conda activate langsplat_v2
# Prefer env binaries over /usr/local/miniconda
export PATH="$CONDA_PREFIX/bin:/usr/local/cuda-12.8/bin:$PATH"
export CUDA_HOME=/usr/local/cuda-12.8
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib/python3.10/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"
export TORCH_CUDA_ARCH_LIST=12.0
# If need network:
export http_proxy="socks5h://127.0.0.1:2345"
export https_proxy="socks5h://127.0.0.1:2345"
export all_proxy="socks5h://127.0.0.1:2345"
```

## Layout
- Dataset (LERF): `~/data/lerf_ovs` → symlink to `~/Work/LangSplat/data/lerf_ovs`
  - scripts use `../../data/lerf_ovs` relative to this repo
- SAM ckpt: `ckpts/sam_vit_h_4b8939.pth` → LangSplat's copy
- RGB 3DGS ckpts: each scene `.../<scene>/output/{chkpnt30000.pth,cameras.json,...}` → LangSplat `output/<scene>_rgb/`
- Language features: reuse existing `language_features/` from LangSplat preprocess (same format)
- V2 pretrained weights: extract into `./output/` (e.g. `teatime_0_1`, ...)

## Quick eval (after weights downloaded)
```bash
bash eval_lerf.sh teatime 0 10000
```

## Pretrained weights download (in progress / resume)
```bash
export http_proxy="socks5h://127.0.0.1:2345" https_proxy="socks5h://127.0.0.1:2345" all_proxy="socks5h://127.0.0.1:2345"
source scripts/activate_env.sh
mkdir -p /tmp/langsplatv2_weights/LERF && cd /tmp/langsplatv2_weights/LERF
# LERF teatime (~6.5GB)
gdown "159jEgMG5SGeFbFcLoqFr7OpqW5yzctt3" -O teatime.tar.gz --proxy "socks5h://127.0.0.1:2345"
# Other LERF scenes (optional):
# figurines: 1qKtjgTgt9xIkBsufiRZH6Mjo6DZSPqc_
# ramen:     12Zgqx_bXt7Hxi7vh3jko1AGvLedd96AX
# waldo:     18BDS9kp0ioCXrdBQrB04TO0fIaE8RowX
bash ~/Work/LangSplatV2/scripts/finish_weights.sh /tmp/langsplatv2_weights/LERF/teatime.tar.gz
```
