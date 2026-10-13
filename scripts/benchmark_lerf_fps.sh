#!/bin/bash
# Benchmark LERF test-set end-to-end FPS for all scenes with official weights.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT/scripts/activate_env.sh"
cd "$ROOT"

DATA=../../data/lerf_ovs
LOG="$ROOT/logs/lerf_fps"
mkdir -p "$LOG"
SUMMARY="$LOG/summary.txt"
: > "$SUMMARY"

if [ $# -eq 0 ]; then
  SCENES=(teatime figurines ramen waldo_kitchen)
else
  SCENES=("$@")
fi
GPU=${GPU:-0}
WARMUP=${WARMUP:-20}
REPEATS=${REPEATS:-50}

echo "LERF E2E FPS benchmark (quick_render + CLIP query)"
echo "GPU=$GPU warmup=$WARMUP repeats=$REPEATS"
echo "========================================" | tee -a "$SUMMARY"

for scene in "${SCENES[@]}"; do
  if [[ ! -d "output/${scene}_0_1" ]]; then
    echo "[$scene] weights missing, skip" | tee -a "$SUMMARY"
    continue
  fi
  echo "=== [$scene] ==="
  CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 python scripts/benchmark_lerf_fps.py \
    -s "$DATA/$scene" \
    -m "output/${scene}_0_1" \
    --dataset_name "$scene" \
    --index 0 \
    --ckpt_root_path output \
    --json_folder "$DATA/label" \
    --checkpoint 10000 \
    --topk 4 \
    --include_feature \
    --quick_render \
    --warmup "$WARMUP" \
    --repeats "$REPEATS" \
    2>&1 | tee "$LOG/${scene}.log" | tee -a "$SUMMARY"
done

echo "========================================" | tee -a "$SUMMARY"
echo "Saved logs: $LOG/"
