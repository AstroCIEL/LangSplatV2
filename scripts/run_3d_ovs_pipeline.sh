#!/bin/bash
# LangSplatV2 full pipeline for 3D-OVS scenes.
# Stages: RGB 3DGS (30k) -> preprocess language features -> train.sh (levels 1-3)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "$ROOT/scripts/activate_env.sh"

DATA_ROOT="${DATA_ROOT:-/data/home/rh_xu30/data/3D_OVS}"
SCENES=("${@:-bed bench lawn room sofa}")
# If called with no args, default list:
if [[ $# -eq 0 ]]; then
  SCENES=(bed bench lawn room sofa)
fi

LOG_DIR="$ROOT/logs/3d_ovs"
mkdir -p "$LOG_DIR" "$ROOT/output"

link_rgb_ckpt() {
  local scene="$1"
  local rgb_dir="$ROOT/output/${scene}_rgb"
  local out_link="$DATA_ROOT/$scene/output"
  mkdir -p "$out_link"
  for f in chkpnt30000.pth cameras.json cfg_args input.ply point_cloud; do
    if [[ -e "$rgb_dir/$f" ]]; then
      ln -sfn "$rgb_dir/$f" "$out_link/$f"
    fi
  done
}

run_rgb() {
  local scene="$1" gpu="$2" port="$3"
  local rgb_dir="$ROOT/output/${scene}_rgb"
  local log="$LOG_DIR/${scene}_rgb.log"
  if [[ -f "$rgb_dir/chkpnt30000.pth" ]]; then
    echo "[SKIP] RGB exists: $rgb_dir/chkpnt30000.pth"
    link_rgb_ckpt "$scene"
    return 0
  fi
  echo "[RGB] $scene on GPU$gpu -> $rgb_dir"
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 python train.py \
    -s "$DATA_ROOT/$scene" \
    -m "$rgb_dir" \
    --no_include_feature \
    --iterations 30000 \
    --port "$port" \
    --quiet \
    >"$log" 2>&1
  echo "RGB_EXIT=$?" >>"$log"
  link_rgb_ckpt "$scene"
}

run_preprocess() {
  local scene="$1" gpu="$2"
  local feat="$DATA_ROOT/$scene/language_features"
  local log="$LOG_DIR/${scene}_preprocess.log"
  if [[ -d "$feat" ]] && [[ "$(ls -1 "$feat" 2>/dev/null | wc -l)" -gt 10 ]]; then
    echo "[SKIP] language_features exist: $feat"
    return 0
  fi
  echo "[PREPROCESS] $scene on GPU$gpu"
  export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 python preprocess.py \
    --dataset_path "$DATA_ROOT/$scene" \
    >"$log" 2>&1
  echo "PREPROCESS_EXIT=$?" >>"$log"
}

run_lang() {
  local scene="$1" gpu="$2" index="${3:-0}"
  local ckpt="$DATA_ROOT/$scene/output/chkpnt30000.pth"
  local feat="$DATA_ROOT/$scene/language_features"
  local log="$LOG_DIR/${scene}_lang.log"
  if [[ ! -f "$ckpt" ]]; then
    echo "[ERR] missing RGB ckpt: $ckpt" >&2
    return 1
  fi
  if [[ ! -d "$feat" ]]; then
    echo "[ERR] missing language_features: $feat" >&2
    return 1
  fi
  if [[ -f "$ROOT/output/${scene}_${index}_1/chkpnt10000.pth" ]]; then
    echo "[SKIP] language model exists: output/${scene}_${index}_1"
    return 0
  fi
  echo "[LANG] $scene index=$index on GPU$gpu"
  # train.sh loops levels 1-3; pin one GPU and unique ports via train.py defaults
  (
    cd "$ROOT"
    CUDA_VISIBLE_DEVICES="$gpu" bash train.sh "$DATA_ROOT" "$scene" "$index"
  ) >"$log" 2>&1
  echo "LANG_EXIT=$?" >>"$log"
}

echo "DATA_ROOT=$DATA_ROOT"
echo "SCENES=${SCENES[*]}"
echo "Logs: $LOG_DIR"
