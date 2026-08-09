#!/usr/bin/env bash
# Wall-clock per-forward latency for LangSplatV2 LERF (quick_render).
# Usage (from repo root):
#   source scripts/activate_env.sh
#   bash scripts/latency/run_lerf_latency.sh              # 1q, all scenes
#   bash scripts/latency/run_lerf_latency.sh teatime      # 1q, one scene
#   MODE=mq bash scripts/latency/run_lerf_latency.sh     # multi-query (all GT phrases)

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source "$ROOT/scripts/activate_env.sh"

export PYTHONUNBUFFERED=1
MODE="${MODE:-1q}"   # 1q | mq
SCENES=("${@:-teatime figurines ramen waldo_kitchen}")

if [[ "$MODE" == "mq" ]]; then
  SUMMARY="$ROOT/eval_result/lerf_latency_summary.txt"
  OUT_SUFFIX="latency"
  QPV=-1
  LABEL="multi-query (all GT phrases per view)"
else
  SUMMARY="$ROOT/eval_result/lerf_latency_summary_1q.txt"
  OUT_SUFFIX="latency_1q"
  QPV=1
  LABEL="1 view x 1 query"
fi

{
  echo "LangSplatV2 LERF latency — ${LABEL}"
  echo "After relevancy NHWC→NCHW matmul fix | Device: Jetson Orin | quick_render"
  echo ""
} > "$SUMMARY"

for scene in "${SCENES[@]}"; do
  if [[ ! -d "output/${scene}_0_1" ]]; then
    echo "[$scene] weights missing, skip" | tee -a "$SUMMARY"
    continue
  fi
  echo "=== [$scene] latency (${MODE}) ==="
  out="eval_result/${scene}_0/${OUT_SUFFIX}"
  mkdir -p "$out"
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
    --queries_per_view "$QPV" \
    --out_dir "$out" \
    --quiet
  {
    echo "[$scene]"
    grep -E 'num_queries:|avg_queries_per_forward:|image_hw:|num_gaussians:' "$out/latency_breakdown.txt" | sed 's/^/  /' || true
    grep -E '^\s+forward avg:' "$out/latency_breakdown.txt"
    echo "  scales_with_P: text.encode, relevancy.clip_match"
    echo "  independent_of_P: render.setup, render.raster, decode.einsum, decode.normalize"
    awk '/FOCUS:/{p=1} p && /ms\/fwd/{print}' "$out/latency_breakdown.txt"
    echo ""
  } | tee -a "$SUMMARY"
done

echo "=== LATENCY SUMMARY: ${LABEL} ==="
cat "$SUMMARY"
echo "Reports under eval_result/<scene>_0/${OUT_SUFFIX}/"