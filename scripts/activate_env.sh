#!/bin/bash
source ~/miniforge3/etc/profile.d/conda.sh
conda activate gs
export PATH="$CONDA_PREFIX/bin:/usr/local/cuda-12.6/bin:$PATH"
export CUDA_HOME=/usr/local/cuda-12.6
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib/python3.10/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"
export TORCH_CUDA_ARCH_LIST=8.7
echo "gs ready for LangSplatV2: $(python -c 'import torch; print(torch.__version__, torch.cuda.get_device_name(0))')"
