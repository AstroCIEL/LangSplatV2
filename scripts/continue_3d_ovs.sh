#!/bin/bash
# Continue 3D-OVS pipeline after RGB round-1 (bed/bench/lawn).
# Usage: bash scripts/continue_3d_ovs.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT/scripts/activate_env.sh"
cd "$ROOT"
DATA=/data/home/rh_xu30/data/3D_OVS
LOG="$ROOT/logs/3d_ovs"
mkdir -p "$LOG"

link_rgb() {
  local scene=$1
  local rgb="output/${scene}_rgb"
  local out="$DATA/$scene/output"
  mkdir -p "$out"
  for f in chkpnt30000.pth cameras.json cfg_args input.ply point_cloud; do
    [[ -e "$rgb/$f" ]] && ln -sfn "$ROOT/$rgb/$f" "$out/$f"
  done
}

run_rgb() {
  local scene=$1 gpu=$2 port=$3
  if [[ -f "output/${scene}_rgb/chkpnt30000.pth" ]]; then
    echo "[SKIP] RGB $scene"; link_rgb "$scene"; return 0
  fi
  echo "[RGB] $scene GPU$gpu"
  CUDA_VISIBLE_DEVICES=$gpu PYTHONUNBUFFERED=1 python train.py \
    -s "$DATA/$scene" -m "output/${scene}_rgb" \
    --no_include_feature --iterations 30000 --port "$port" --quiet \
    >"$LOG/${scene}_rgb.log" 2>&1
  echo "RGB_EXIT=$?" >>"$LOG/${scene}_rgb.log"
  link_rgb "$scene"
}

run_preprocess() {
  local scene=$1 gpu=$2
  local n
  n=$(ls -1 "$DATA/$scene/language_features" 2>/dev/null | wc -l || true)
  if [[ "$n" -gt 10 ]]; then
    echo "[SKIP] preprocess $scene ($n files)"; return 0
  fi
  echo "[PREPROCESS] $scene GPU$gpu"
  HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}" \
  CUDA_VISIBLE_DEVICES=$gpu PYTHONUNBUFFERED=1 python preprocess.py \
    --dataset_path "$DATA/$scene" \
    >"$LOG/${scene}_preprocess.log" 2>&1
  echo "PREPROCESS_EXIT=$?" >>"$LOG/${scene}_preprocess.log"
}

run_lang() {
  local scene=$1 gpu=$2 index=${3:-0} port=${4:-55557}
  # Skip only if all 3 feature levels finished (avoid false skip on partial/pretrained dirs)
  if [[ -f "output/${scene}_${index}_1/chkpnt10000.pth" \
     && -f "output/${scene}_${index}_2/chkpnt10000.pth" \
     && -f "output/${scene}_${index}_3/chkpnt10000.pth" ]]; then
    echo "[SKIP] lang $scene (all levels present)"; return 0
  fi
  echo "[LANG] $scene GPU$gpu port=$port"
  (
    cd "$ROOT"
    for level in 1 2 3; do
      if [[ -f "output/${scene}_${index}_${level}/chkpnt10000.pth" ]]; then
        echo "[SKIP] ${scene}_${index}_${level}"
        continue
      fi
      CUDA_VISIBLE_DEVICES=$gpu PYTHONUNBUFFERED=1 python train.py \
        -s "$DATA/$scene" \
        -m "output/${scene}_${index}" \
        --start_checkpoint "$DATA/$scene/output/chkpnt30000.pth" \
        --feature_level "$level" \
        --vq_layer_num 1 \
        --codebook_size 64 \
        --cos_loss \
        --topk 4 \
        --port "$port"
    done
  ) >"$LOG/${scene}_lang.log" 2>&1
  echo "LANG_EXIT=$?" >>"$LOG/${scene}_lang.log"
}

echo "=== Round 2: RGB room/sofa + preprocess bed ==="
run_rgb room 1 6221 &
pid_room=$!
run_rgb sofa 2 6222 &
pid_sofa=$!
run_preprocess bed 3 &
pid_pp_bed=$!
wait $pid_room $pid_sofa $pid_pp_bed

echo "=== Round 3: preprocess bench/lawn/room ==="
run_preprocess bench 1 &
run_preprocess lawn 2 &
run_preprocess room 3 &
wait

echo "=== Round 4: preprocess sofa + lang bed/bench ==="
run_preprocess sofa 1 &
run_lang bed 2 0 6320 &
run_lang bench 3 0 6321 &
wait

echo "=== Round 5: lang lawn/room/sofa ==="
run_lang lawn 1 0 6331 &
run_lang room 2 0 6332 &
run_lang sofa 3 0 6333 &
wait

echo "=== ALL 3D-OVS PIPELINE DONE ==="
