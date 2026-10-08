#!/bin/bash

# Số lượng GPU trên mỗi node (máy)
NUM_GPUS_PER_NODE=1

# Đường dẫn tới file script training của bạn
TRAIN_SCRIPT="teacher_cache.py"

# SUBSETS=(
#   "VOC2007"
#   "OK-VQA"
# )

SUBSETS=(
  "ImageNet_1K" "N24News" "HatefulMemes" "VOC2007" "SUN397"
  # "OK-VQA" "A-OKVQA" "DocVQA" "InfographicsVQA" "ChartQA"
)

# =========================================================================
# Dùng torchrun để khởi chạy
# =========================================================================

# MODEL_NAME=training/FastVLM-0.5B_cls_struct_sigreg_SW_gmm_d1_sig1_kd10_sw0.1_numt17_tmax5_l24/checkpoint-epoch-0
# OUTPUT_DIR=caching/FastVLM-0.5B_cls_struct_sigreg_SW_gmm_d1_sig1_kd10_sw0.1_numt17_tmax5_l24

MODEL_NAME=training1/FastVLM-0.5B_base_16_eos_cls/checkpoint-epoch-0
OUTPUT_DIR=caching/FastVLM-0.5B_base_16_eos_cls

echo "MODEL_NAME: $MODEL_NAME"
echo "OUTPUT_DIR: $OUTPUT_DIR"

PORT=12345

torchrun --master_port=$PORT \
    --nproc_per_node=$NUM_GPUS_PER_NODE \
    $TRAIN_SCRIPT \
    --model_name $MODEL_NAME \
    --lora True \
    --lora_r 8 \
    --lora_alpha 64 \
    --model_backbone "llava_qwen2" \
    --pooling "eos" \
    --dataset_name "TIGER-Lab/MMEB-train" \
    --subset_name "${SUBSETS[@]}" \
    --dataset_split "original" \
    --image_dir "vlm2vec_train/MMEB-train" \
    --output_dir $OUTPUT_DIR \
    --per_device_train_batch_size 4 \
    --seed 42 \
    --normalize False \
    --report_to "none" 