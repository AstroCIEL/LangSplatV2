#!/usr/bin/env bash
# Wall-clock per-forward latency for LangSplatV2 LERF (quick_render).
# Usage (from repo root):
#   source scripts/activate_env.sh
#   bash scripts/latency/run_lerf_latency.sh              # all scenes
#   bash scripts/latency/run_lerf_latency.sh teatime      # one scene

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source "$ROOT/scripts/activate_env.sh"

export PYTHONUNBUFFERED=1
SCENES=("${@:-teatime figurines ramen waldo_kitchen}")
SUMMARY="$ROOT/eval_result/lerf_latency_summary_1q.txt"
: > "$SUMMARY"

for scene in "${SCENES[@]}"; do
  if [[ ! -d "output/${scene}_0_1" ]]; then
    echo "[$scene] weights missing, skip" | tee -a "$SUMMARY"
    continue
  fi
  echo "=== [$scene] latency ==="
  out="eval_result/${scene}_0/latency_1q"
  mkdir -p "$out"
  # Paper-comparable: 1 view forward with exactly 1 text query
  python scripts/latency/profile_lerf_forward.py \
    -s "data/lerf_ovs/${scene}" \
    -m "output/${scene}_0_1" \
    --dataset_name "$scene" \
    --index 0 \
    --checkpoint 10000 \
    --ckpt_root_path output \
    --json_folder data/lerf_ovs/label \
    --include_feature \
    --quick_render \
    --topk 4 \
    --warmup 3 \
    --queries_per_view 1 \
    --out_dir "$out" \
    --quiet
  avg=$(grep -E '^\s+forward avg:' "$out/latency_breakdown.txt" | awk '{print $3,$4}')
  echo "[$scene] single-query forward_avg=${avg:-NA}" | tee -a "$SUMMARY"
  echo "  scales_with_P: text.encode, relevancy.clip_match" | tee -a "$SUMMARY"
  echo "  independent_of_P: render.setup, render.raster, decode.einsum, decode.normalize" | tee -a "$SUMMARY"
  awk '/FOCUS:/{p=1} p && /ms\/fwd/{print}' "$out/latency_breakdown.txt" | tee -a "$SUMMARY"
  echo "" | tee -a "$SUMMARY"
done

echo "=== LATENCY SUMMARY: 1 view x 1 query ==="
cat "$SUMMARY"
echo "Reports under eval_result/<scene>_0/latency_1q/"
