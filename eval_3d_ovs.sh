#!/bin/bash
DATASET_NAME=$1
INDEX=$2
CHECKPOINT=$3

# path to 3D_OVS dataset
DATASET_ROOT_PATH=./data/3D_OVS
gt_folder=./data/3D_OVS/${DATASET_NAME}/segmentations

ROOT_PATH="."

python eval_3d_ovs.py \
    -s ${DATASET_ROOT_PATH}/${DATASET_NAME} \
    -m "${ROOT_PATH}/output/${DATASET_NAME}_${INDEX}_1" \
    --dataset_name ${DATASET_NAME} \
    --index ${INDEX} \
    --ckpt_root_path ${ROOT_PATH}/output \
    --output_dir ${ROOT_PATH}/eval_result \
    --mask_thresh 0.25 \
    --gt_mask_dir ${gt_folder} \
    --checkpoint ${CHECKPOINT} \
    --include_feature \
    --topk 4 \
    --quick_render