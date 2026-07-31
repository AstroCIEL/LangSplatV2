#!/bin/bash
# Evaluate all installed LERF scenes and collect the summary metrics.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
source "$ROOT/scripts/activate_env.sh"

# CLIP / HF weight downloads need proxy on this machine; disable broken IPv6 path.
export http_proxy="${http_proxy:-http://127.0.0.1:7890}"
export https_proxy="${https_proxy:-http://127.0.0.1:7890}"
export all_proxy="${all_proxy:-http://127.0.0.1:7890}"
export no_proxy="${no_proxy:-localhost,127.0.0.1}"
export PYTHONUNBUFFERED=1

SUMMARY=/tmp/lerf_eval_summary.txt
: > "$SUMMARY"

for scene in "${@:-teatime figurines ramen waldo_kitchen}"; do
  if [ ! -d "output/${scene}_0_1" ]; then
    echo "[$scene] weights missing, skipping" | tee -a "$SUMMARY"
    continue
  fi
  echo "=== [$scene] evaluating ==="
  log="/tmp/eval_${scene}.log"
  set +e
  bash eval_lerf.sh "$scene" 0 10000 > "$log" 2>&1
  rc=$?
  set -e
  if [ $rc -ne 0 ]; then
    echo "[$scene] FAILED (rc=$rc)" | tee -a "$SUMMARY"
    grep -a -A8 "Traceback\|Error\|error:" "$log" | tail -40
    continue
  fi
  iou=$(grep -aoE "iou chosen: [0-9.]+" "$log" | tail -1 | awk '{print $3}')
  acc=$(grep -aoE "Localization accuracy: [0-9.]+" "$log" | tail -1 | awk '{print $3}')
  echo "[$scene] mIoU=${iou:-NA}  loc_acc=${acc:-NA}" | tee -a "$SUMMARY"
done

echo "=== EVAL SUMMARY ==="
cat "$SUMMARY"
echo "ALL_EVAL_DONE"
