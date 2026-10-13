#!/bin/bash
# Retrain 3D-OVS after fixing black COLMAP point colors.
# RGB: LangSplat V1 on GPU 0/1 → Language + eval: LangSplatV2
set -euo pipefail

V1=/data/home/rh_xu30/Work/LangSplat
V2=/data/home/rh_xu30/Work/LangSplatV2
DATA=/data/home/rh_xu30/data/3D_OVS
LOG=$V2/logs/3d_ovs_retrain
mkdir -p "$LOG" "$V2/output"

activate_v1() {
  # shellcheck disable=SC1091
  source /data/home/rh_xu30/miniforge3/etc/profile.d/conda.sh
  conda activate langsplat
  export PATH="$CONDA_PREFIX/bin:/usr/local/cuda-12.8/bin:$PATH"
  export CUDA_HOME=/usr/local/cuda-12.8
  export LD_LIBRARY_PATH="$CONDA_PREFIX/lib/python3.10/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"
  export TORCH_CUDA_ARCH_LIST=12.0
}

activate_v2() {
  # shellcheck disable=SC1091
  source "$V2/scripts/activate_env.sh"
}

link_rgb() {
  local scene=$1
  local rgb="$V2/output/${scene}_rgb"
  local out="$DATA/$scene/output"
  mkdir -p "$out"
  for f in chkpnt30000.pth cameras.json cfg_args input.ply point_cloud; do
    [[ -e "$rgb/$f" ]] && ln -sfn "$rgb/$f" "$out/$f"
  done
}

run_rgb_v1() {
  local scene=$1
  local gpu=$2
  local port=$3
  local rgb="$V2/output/${scene}_rgb"
  if [[ -f "$rgb/chkpnt30000.pth" ]]; then
    echo "[SKIP] RGB $scene exists"
    link_rgb "$scene"
    return 0
  fi
  echo "[$(date +%T)] RGB $scene GPU$gpu"
  activate_v1
  cd "$V1"
  CUDA_VISIBLE_DEVICES=$gpu PYTHONUNBUFFERED=1 python train.py \
    -s "$DATA/$scene" \
    -m "$rgb" \
    --no_include_feature \
    --iterations 30000 \
    --port "$port" \
    --quiet \
    >"$LOG/${scene}_rgb.log" 2>&1
  local ec=$?
  echo "RGB_EXIT=$ec" >>"$LOG/${scene}_rgb.log"
  if [[ $ec -ne 0 ]]; then
    echo "RGB failed for $scene" >&2
    return $ec
  fi
  link_rgb "$scene"
  python -c "
import torch
p,it=torch.load('${rgb}/chkpnt30000.pth', weights_only=False, map_location='cpu')
xyz=next(t for t in p if hasattr(t,'shape') and len(t.shape)==2 and t.shape[-1]==3)
print(f'FINAL_N={xyz.shape[0]} iter={it}')
" >>"$LOG/${scene}_rgb.log"
  echo "[$(date +%T)] RGB $scene done"
}

run_lang_v2() {
  local scene=$1
  local gpu=$2
  local port=$3
  if [[ -f "$V2/output/${scene}_0_1/chkpnt10000.pth" && -f "$V2/output/${scene}_0_3/chkpnt10000.pth" ]]; then
    echo "[SKIP] lang $scene"
    return 0
  fi
  echo "[$(date +%T)] LANG $scene GPU$gpu"
  activate_v2
  cd "$V2"
  : >"$LOG/${scene}_lang.log"
  for level in 1 2 3; do
    if [[ -f "output/${scene}_0_${level}/chkpnt10000.pth" ]]; then
      echo "[SKIP] ${scene}_0_${level}" | tee -a "$LOG/${scene}_lang.log"
      continue
    fi
    CUDA_VISIBLE_DEVICES=$gpu PYTHONUNBUFFERED=1 python train.py \
      -s "$DATA/$scene" \
      -m "output/${scene}_0" \
      --start_checkpoint "$DATA/$scene/output/chkpnt30000.pth" \
      --feature_level "$level" \
      --vq_layer_num 1 \
      --codebook_size 64 \
      --cos_loss \
      --topk 4 \
      --port "$port" \
      >>"$LOG/${scene}_lang.log" 2>&1
    echo "LEVEL_${level}_EXIT=$?" >>"$LOG/${scene}_lang.log"
  done
  echo "[$(date +%T)] LANG $scene done"
}

run_eval_v2() {
  local scene=$1
  local gpu=$2
  echo "[$(date +%T)] EVAL $scene GPU$gpu"
  activate_v2
  cd "$V2"
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  CUDA_VISIBLE_DEVICES=$gpu PYTHONUNBUFFERED=1 bash eval_3d_ovs.sh "$scene" 0 10000 \
    >"$LOG/eval_${scene}.log" 2>&1
  echo "EVAL_EXIT=$?" >>"$LOG/eval_${scene}.log"
  rg "iou chosen|EVAL_EXIT|OutOfMemory" "$LOG/eval_${scene}.log" | tail -5 || true
}

echo "========== Phase 1: RGB (V1) GPU0/1 =========="
(
  run_rgb_v1 bed 0 7100
  run_rgb_v1 lawn 0 7101
  run_rgb_v1 room 0 7102
) &
pid0=$!
(
  run_rgb_v1 bench 1 7200
  run_rgb_v1 sofa 1 7201
) &
pid1=$!
wait $pid0 $pid1
echo "RGB phase done at $(date +%T)"

echo "========== Phase 2: Language (V2) =========="
run_lang_v2 bed 0 7300 &
p0=$!
run_lang_v2 bench 1 7301 &
p1=$!
wait $p0 $p1
run_lang_v2 lawn 0 7310 &
p0=$!
run_lang_v2 room 1 7311 &
p1=$!
wait $p0 $p1
run_lang_v2 sofa 0 7320
echo "LANG phase done at $(date +%T)"

echo "========== Phase 3: Eval =========="
run_eval_v2 bed 0 &
run_eval_v2 bench 1 &
wait
run_eval_v2 lawn 0 &
run_eval_v2 room 1 &
wait
run_eval_v2 sofa 0

echo "========== SUMMARY =========="
for s in bed bench lawn room sofa; do
  iou=$(rg -o "iou chosen: [0-9.]+" "$LOG/eval_${s}.log" 2>/dev/null | tail -1 || echo "iou chosen: N/A")
  echo "$s: $iou"
done | tee "$LOG/eval_summary.txt"
echo "[$(date +%F\ %T)] ALL DONE"
