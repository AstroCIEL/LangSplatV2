#!/bin/bash
# End-to-end ablation: ASIC vs PIM, each with full-frame, HLS, and HLS+SCAR.
# Real checkpoint per scene, all test cameras.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT/scripts/activate_env.sh"
cd "$ROOT"

export CUDA_VISIBLE_DEVICES="${GPU:-0}"
python -m simulator.run_ablation "$@"
