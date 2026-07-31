#!/bin/bash
# Evaluate all installed LERF scenes and collect the summary metrics.
export PATH="/data/home/rh_xu30/miniforge3/envs/langsplat_v2/bin:/usr/local/cuda-12.8/bin:$PATH"
export CUDA_HOME=/usr/local/cuda-12.8
export LD_LIBRARY_PATH="/data/home/rh_xu30/miniforge3/envs/langsplat_v2/lib/python3.10/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
SUMMARY=/tmp/lerf_eval_summary.txt
: > "$SUMMARY"

for scene in "${@:-teatime figurines ramen waldo_kitchen}"; do
  if [ ! -d "output/${scene}_0_1" ]; then
    echo "[$scene] weights missing, skipping" | tee -a "$SUMMARY"
    continue
  fi
  echo "=== [$scene] evaluating ==="
  log="/tmp/eval_${scene}.log"
  bash eval_lerf.sh "$scene" 0 10000 > "$log" 2>&1
  rc=$?
  if [ $rc -ne 0 ]; then
    echo "[$scene] FAILED (rc=$rc)" | tee -a "$SUMMARY"
    grep -a -A5 "Traceback" "$log" | tail -20
    continue
  fi
  iou=$(grep -aoE "iou chosen: [0-9.]+" "$log" | tail -1 | awk '{print $3}')
  acc=$(grep -aoE "Localization accuracy: [0-9.]+" "$log" | tail -1 | awk '{print $3}')
  echo "[$scene] mIoU=${iou:-NA}  loc_acc=${acc:-NA}" | tee -a "$SUMMARY"
done

echo "=== EVAL SUMMARY ==="
cat "$SUMMARY"
echo "ALL_EVAL_DONE"
