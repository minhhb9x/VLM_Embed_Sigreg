#!/bin/bash
set -euo pipefail

SUBSETS=(
    "ImageNet-1K" "N24News" "HatefulMemes" "VOC2007" "SUN397"
    "Place365" "ImageNet-A" "ImageNet-R" "ObjectNet" "Country211"
)

# Must match the teacher checkpoint used to generate GMM training embeddings.
MODEL="${MODEL:-raghavlite/B3_Qwen2_2B}"
GMM_PATH="${GMM_PATH:-gmm_training/B3_Qwen2_2B_cls/gmm.joblib}"
OUTPUT_DIR="${OUTPUT_DIR:-./MMEB-gmm-eval_outputs/B3_Qwen2_2B_cls_raw}"
MODEL="${MODEL:-raghavlite/B3_Qwen2_2B}"
GMM_PATH="${GMM_PATH:-gmm_training/B3_Qwen2_2B_cls/gmm.joblib}"
OUTPUT_DIR="${OUTPUT_DIR:-./MMEB-gmm-eval_outputs/B3_Qwen2_2B_cls_raw}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python infer_eval_gmm.py \
    --model_name "$MODEL" \
    --encode_output_path "$OUTPUT_DIR" \
    --lora True --lora_r 8 --lora_alpha 8 \
    --pooling eos \
    --model_backbone qwen2_vl \
    --normalize False \
    --bf16 \
    --dataset_name TIGER-Lab/MMEB-eval \
    --subset_name "${SUBSETS[@]}" \
    --dataset_split test \
    --per_device_eval_batch_size 16 \
    --image_dir eval_images/ \
    --tgt_prefix_mod \
    --load_pretrained_lora True \
    --report_to none \
    --gmm_path "$GMM_PATH" \
    --gmm_eval_sides both \
    --gmm_score_batch_size 1024
