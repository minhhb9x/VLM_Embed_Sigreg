#!/bin/bash

SUBSETS=(
  "ImageNet_1K" "N24News" "HatefulMemes" "VOC2007" "SUN397"
  # "OK-VQA" "A-OKVQA" "DocVQA" "InfographicsVQA" "ChartQA"
)

# =========================================================================
# Dùng torchrun để khởi chạy
# =========================================================================
python train_teacher_gmm.py \
    --caching_dir "caching/B3_Qwen2_2B_cls" \
    --subset_name "${SUBSETS[@]}" \
    --seed 42 \
    --n_components 32 \
    --cov_type "diag" \
    --reg_covar 1e-4 \
    --max_iter 100 \
    --tol 1e-3 \
    --output_dir "gmm_training/B3_Qwen2_2B_cls"
