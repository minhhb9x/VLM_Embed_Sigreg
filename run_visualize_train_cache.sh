#!/bin/bash
set -euo pipefail

SUBSETS=("ImageNet_1K" "N24News" "HatefulMemes" "VOC2007" "SUN397")
CACHE_DIR="${CACHE_DIR:-caching/B3_Qwen2_2B_cls}"
PLOT_DIR="${PLOT_DIR:-projection_plots/B3_Qwen2_2B_cls_train_raw_gmm}"
GMM_PATH="${GMM_PATH:-gmm_training/B3_Qwen2_2B_cls/gmm.joblib}"

# 0: use all sample directories; a positive number selects a random subset.
NUM_SAMPLES="${NUM_SAMPLES:-0}"
DEVICE="${DEVICE:-cpu}"

python visualize_train_cache.py \
    --caching_dir "$CACHE_DIR" \
    --gmm_path "$GMM_PATH" \
    --subset_name "${SUBSETS[@]}" \
    --side both \
    --num_samples "$NUM_SAMPLES" \
    --num_projections 16 \
    --plot_seed 42 \
    --sample_seed 42 \
    --plot_bins 100 \
    --projection_batch_size 256 \
    --device "$DEVICE" \
    --plot_dir "$PLOT_DIR" \
    --include_combined_all
