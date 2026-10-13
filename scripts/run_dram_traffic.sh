#!/bin/bash
# DRAM-traffic comparison for the paper: full-frame PIM vs ASIC-only,
# HLS and SCAR off, one real checkpoint per scene, all test cameras.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT/scripts/activate_env.sh"
cd "$ROOT"

export CUDA_VISIBLE_DEVICES="${GPU:-0}"
python -m simulator.run_dram_traffic "$@"
