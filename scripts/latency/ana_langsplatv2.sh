#!/usr/bin/env bash
# Nsight Compute kernel-level latency for LangSplatV2 quick_render raster.
# Mirrors OpenGaussian/scripts/latency/ana_opengaussian.sh
#
# Usage (from repo root, gs env active):
#   bash scripts/latency/ana_langsplatv2.sh teatime
# Prefer root for NCU privileges:
#   sudo -E bash scripts/latency/ana_langsplatv2.sh teatime

set -eo pipefail

SCENE="${1:-teatime}"
NCU_PATH=${NCU_PATH:-/opt/nvidia/nsight-compute/2024.3.1}
REPOSITORY_PATH="$(cd "$(dirname "$0")/../.." && pwd)"
OUTPUT_PATH="$REPOSITORY_PATH/eval_result/${SCENE}_0/latency"
NCU_BIN=${NCU_BIN:-$NCU_PATH/ncu}
if [[ ! -x "$NCU_BIN" ]]; then
    NCU_BIN="$(command -v ncu || true)"
fi
if [[ -z "$NCU_BIN" ]]; then
    echo "ERROR: ncu not found" >&2
    exit 1
fi

if [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
    PYTHON_BIN="${CONDA_PREFIX}/bin/python"
    export PATH="${CONDA_PREFIX}/bin:${PATH}"
else
    PYTHON_BIN="$(command -v python || true)"
fi
if [[ -z "${PYTHON_BIN}" ]]; then
    echo "ERROR: Cannot find python. Activate conda env gs first." >&2
    exit 1
fi

if [[ "${EUID:-$(id -u)}" -ne 0 ]]; then
    echo "==WARNING== Nsight Compute usually needs root. Re-run with:"
    echo "sudo -E bash $0 $*"
    exec sudo -E bash "$0" "$@"
fi

mkdir -p "$OUTPUT_PATH"
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-12.6}
export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-8.7}

kernel_name='regex:"preprocessCUDA'
kernel_name+="|DeviceScanInitKernel"
kernel_name+="|DeviceScanKernel"
kernel_name+="|duplicateWithKeys"
kernel_name+="|DeviceRadixSortHistogramKernel"
kernel_name+="|DeviceRadixSortExclusiveSumKernel"
kernel_name+="|DeviceRadixSortOnesweepKernel"
kernel_name+="|identifyTileRanges"
kernel_name+='|renderCUDA"'

metrics=(
    "gpu__time_duration.sum"
    "sm__throughput.avg.pct_of_peak_sustained_elapsed"
    "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed"
)
metrics_csv="$(IFS=','; echo "${metrics[*]}")"

target="$PYTHON_BIN $REPOSITORY_PATH/scripts/latency/profile_lerf_forward.py \
    -s $REPOSITORY_PATH/data/lerf_ovs/$SCENE \
    -m $REPOSITORY_PATH/output/${SCENE}_0_1 \
    --dataset_name $SCENE \
    --index 0 \
    --checkpoint 10000 \
    --ckpt_root_path $REPOSITORY_PATH/output \
    --json_folder $REPOSITORY_PATH/data/lerf_ovs/label \
    --include_feature --quick_render --topk 4 \
    --ncu_mode --out_dir $OUTPUT_PATH --quiet"

echo "NCU target: $target"
echo "Output dir: $OUTPUT_PATH"

$NCU_BIN \
    --set full \
    --target-processes all \
    -k "$kernel_name" \
    -f \
    -o "$OUTPUT_PATH/analysis" \
    $target

$NCU_BIN \
    -i "$OUTPUT_PATH/analysis.ncu-rep" \
    --page raw \
    --csv \
    --metrics "$metrics_csv" \
    --log-file "$OUTPUT_PATH/analysis.csv"

$NCU_BIN \
    -i "$OUTPUT_PATH/analysis.ncu-rep" \
    --page raw \
    --csv \
    --metrics "$metrics_csv" \
    --print-summary per-kernel \
    --log-file "$OUTPUT_PATH/analysis_per_kernel.csv"

$PYTHON_BIN "$REPOSITORY_PATH/scripts/latency/ana_raster_kernels.py" \
    --csv "$OUTPUT_PATH/analysis_per_kernel.csv" \
    --out_txt "$OUTPUT_PATH/raster_kernel_breakdown.txt" \
    --out_json "$OUTPUT_PATH/raster_kernel_breakdown.json"

echo "Done. See $OUTPUT_PATH/raster_kernel_breakdown.txt"
