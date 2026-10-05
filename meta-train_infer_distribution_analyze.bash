#!/bin/bash

INFER_SCRIPT="infer_eval_hidden_attention.py"   # sửa thành script infer của bạn

INFER_SUBSETS=(
    "ImageNet-1K"
    # "N24News" "HatefulMemes" "VOC2007" "SUN397"
    # "Place365" "ImageNet-A" "ImageNet-R" "ObjectNet" "Country211"
)

MODELS=(
    # "raghavlite/B3_Qwen2_2B"
    # training/FastVLM-0.5B_cls_struct_sigreg_gmm_d0_sig1_kd1_sw0.05_l1/checkpoint-epoch-0
    # training_old/FastVLM-0.5B_cls_rkd_kdjepa_d0_sig1_kd1_sw0.05_l1/checkpoint-epoch-0
    # training_old/FastVLM-0.5B_cls_rkd_kdjepa_d0_sig1_kd1_sw0.05_l1/checkpoint-epoch-0
    # training_old/FastVLM-0.5B_cls_teacherinfo_jepa_d1_sig1_kd1_sw0.05_l15/checkpoint-epoch-0
    # training_old/FastVLM-0.5B_cls_teacherinfo_jepa_d0_sig1_kd1_sw0.05_l15/checkpoint-epoch-0
    # "apple/FastVLM-0.5B"
    # "training1/FastVLM-0.5B_base_16_eos_cls/checkpoint-epoch-0"
    # "training_old/FastVLM-0.5B_1st_jepa0.05_cls/checkpoint-epoch-0"
    # "training_old/FastVLM-0.5B_last_jepa0.05_cls/checkpoint-epoch-0"
    "training/FastVLM-0.5B_cls_struct_sigreg_SW_gmm_d0_sig1_kd1_sw0.05_l24/checkpoint-epoch-0"
)

BACKBONES=(
    # "qwen2_vl"
    # "llava_qwen2"
    # "llava_qwen2"
    # "llava_qwen2"
    # "llava_qwen2"
    # "llava_qwen2_old"
    # "llava_qwen2"
    # "llava_qwen2"
    # "llava_qwen2"
    "llava_qwen2"
)

# Kiểm tra MODELS và BACKBONES có cùng số phần tử không
if [ "${#MODELS[@]}" -ne "${#BACKBONES[@]}" ]; then
    echo "Error: MODELS and BACKBONES must have the same length."
    exit 1
fi

mkdir -p infer
mkdir -p projection_plots

for i in "${!MODELS[@]}"; do

    MODEL="${MODELS[$i]}"
    BACKBONE="${BACKBONES[$i]}"

    EXP_NAME=$(basename "$(dirname "$MODEL")")

    EXTRA_ARGS=()
    OUTPUT_SUFFIX=""

    if [ "$BACKBONE" = "llava_onevision" ] || [ "$BACKBONE" = "llava_onevision_old" ]; then
        IMAGE_RESOLUTION="tiny"

        EXTRA_ARGS+=(--image_resolution "$IMAGE_RESOLUTION")
        OUTPUT_SUFFIX="_${IMAGE_RESOLUTION}"
    fi

    INFER_OUTPUT="infer/${EXP_NAME}${OUTPUT_SUFFIX}"
    PROJECTION_OUTPUT="projection_plots/${EXP_NAME}${OUTPUT_SUFFIX}"


    echo "========================================================"
    echo "Model     : $MODEL"
    echo "Backbone  : $BACKBONE"
    echo "Experiment: $EXP_NAME"
    echo "Infer out : $INFER_OUTPUT"
    echo "Projection: $PROJECTION_OUTPUT"
    echo "Extra args: ${EXTRA_ARGS[*]}"
    echo "========================================================"

    CUDA_VISIBLE_DEVICES=0 python "$INFER_SCRIPT" \
        --model_name "$MODEL" \
        --lora True \
        --lora_r 64 \
        --lora_alpha 64 \
        --pooling eos \
        --model_backbone "$BACKBONE" \
        --normalize False \
        --bf16 \
        --dataset_name "TIGER-Lab/MMEB-eval" \
        --subset_name "${INFER_SUBSETS[@]}" \
        --dataset_split "test" \
        --image_dir "eval_images/" \
        --tgt_prefix_mod \
        --encode_output_path "$INFER_OUTPUT" \
        --per_device_eval_batch_size 8 \
        --load_pretrained_lora True \
        --report_to None \
        "${EXTRA_ARGS[@]}"

    if [ $? -ne 0 ]; then
        echo "ERROR: Infer failed for $EXP_NAME"
        continue
    fi

    CUDA_VISIBLE_DEVICES=0 python distribution_visualize.py \
        --pt_dir "${INFER_OUTPUT}/${INFER_SUBSETS[0]}/query" \
        --num_samples 0 \
        --plot_dir "${PROJECTION_OUTPUT}"
    
    # rm -rf "${INFER_OUTPUT}"

    echo "Finished: ${EXP_NAME}${OUTPUT_SUFFIX}"
    echo

done