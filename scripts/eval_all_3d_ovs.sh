#!/bin/bash
# Wait for sofa lang training, then eval all 3D-OVS scenes.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT/scripts/activate_env.sh"
cd "$ROOT"

LOG="$ROOT/logs/3d_ovs"
mkdir -p "$LOG" "$ROOT/eval_result"
CKPT=10000
INDEX=0
SCENES=(bed bench lawn room sofa)

echo "[$(date +%F\ %T)] waiting for sofa_0_3/chkpnt${CKPT}.pth ..."
while [[ ! -f "$ROOT/output/sofa_0_3/chkpnt${CKPT}.pth" ]]; do
  if ! pgrep -f "train.py.*sofa_0" >/dev/null 2>&1; then
    if [[ ! -f "$ROOT/output/sofa_0_3/chkpnt${CKPT}.pth" ]]; then
      echo "ERROR: sofa training stopped without checkpoint" >&2
      tr '\r' '\n' <"$LOG/sofa_lang.log" | tail -40 >&2
      exit 1
    fi
  fi
  sleep 60
done
echo "[$(date +%F\ %T)] sofa training complete"

# Ensure all scenes have all 3 levels
missing=0
for s in "${SCENES[@]}"; do
  for lvl in 1 2 3; do
    if [[ ! -f "$ROOT/output/${s}_${INDEX}_${lvl}/chkpnt${CKPT}.pth" ]]; then
      echo "MISSING: output/${s}_${INDEX}_${lvl}/chkpnt${CKPT}.pth"
      missing=1
    fi
  done
done
if [[ $missing -ne 0 ]]; then
  echo "ERROR: not all lang checkpoints ready" >&2
  exit 1
fi

run_eval() {
  local scene=$1 gpu=$2
  local out="$LOG/eval_${scene}.log"
  echo "[$(date +%T)] eval $scene on GPU$gpu"
  CUDA_VISIBLE_DEVICES=$gpu PYTHONUNBUFFERED=1 bash eval_3d_ovs.sh "$scene" "$INDEX" "$CKPT" \
    >"$out" 2>&1
  echo "EVAL_EXIT=$?" >>"$out"
  rg "iou chosen|mean_iou|ERROR|Error|Traceback|prompt_mean" "$out" | tail -20 || true
}

echo "[$(date +%F\ %T)] starting evaluations"
# Round 1: 3 scenes on GPUs 1/2/3
run_eval bed 1 &
pid1=$!
run_eval bench 2 &
pid2=$!
run_eval lawn 3 &
pid3=$!
wait $pid1 $pid2 $pid3

# Round 2
run_eval room 1 &
pid1=$!
run_eval sofa 2 &
pid2=$!
wait $pid1 $pid2

echo ""
echo "========== 3D-OVS EVAL SUMMARY =========="
for s in "${SCENES[@]}"; do
  iou=$(rg -o "iou chosen: [0-9.]+" "$LOG/eval_${s}.log" 2>/dev/null | tail -1 || echo "iou chosen: N/A")
  echo "$s: $iou"
done | tee "$LOG/eval_summary.txt"
echo "[$(date +%F\ %T)] all evals done"
