#!/bin/bash

# Số lượng GPU trên mỗi node (máy)
NUM_GPUS_PER_NODE=1

# Đường dẫn tới file script training của bạn
TRAIN_SCRIPT="train_ddp.py"

# =========================================================================
# Dùng torchrun để khởi chạy
# =========================================================================

KD_WEIGHT=${1:-0.1}
NUM_LAYER=${2:-23}
PORT=${3:-29521}

bool_to_python() {
    case "$1" in
        1|true|True|TRUE)
            echo "True"
            ;;
        0|false|False|FALSE)
            echo "False"
            ;;
        *)
            echo "ERROR: Boolean argument must be 0/1 or True/False, got '$1'" >&2
            exit 1
            ;;
    esac
}

bool_to_int() {
    case "$1" in
        1|true|True|TRUE)
            echo "1"
            ;;
        0|false|False|FALSE)
            echo "0"
            ;;
        *)
            echo "ERROR: Boolean argument must be 0/1 or True/False, got '$1'" >&2
            exit 1
            ;;
    esac
}

echo "============================================================"
echo "Experiment:"
echo "  KD_WEIGHT: $KD_WEIGHT"
echo "  NUM_LAYER: $NUM_LAYER"
echo "============================================================"

EXP_NAME="vis_cosine_reg_kd${KD_WEIGHT}_l${NUM_LAYER}"
OUTPUT_MODEL="training/FastVLM-0.5B_cls_${EXP_NAME}"

torchrun --master_addr=127.0.0.1 --master_port=$PORT \
    --nproc_per_node=$NUM_GPUS_PER_NODE $TRAIN_SCRIPT \
    --model_name apple/FastVLM-0.5B \
    --lora True \
    --teacher_lora True \
    --lora_r 64 \
    --lora_alpha 64 \
    --model_backbone "llava_qwen2" \
    --pooling "eos" \
    --dataset_name "TIGER-Lab/MMEB-train" \
    --subset_name "ImageNet_1K" "N24News" "HatefulMemes" "VOC2007" "SUN397" \
    --dataset_split "original" \
    --image_dir "vlm2vec_train/MMEB-train" \
    --percent_data 1.0 \
    --output_dir $OUTPUT_MODEL \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-4 \
    --num_train_epochs 1 \
    --bf16 \
    --save_total_limit 5 \
    --logging_steps 1 \
    --save_strategy "epoch" \
    --seed 42 \
    --weight_decay 0.01 \
    --normalize True \
    --teacher_normalize True \
    --lr_scheduler_type "constant" \
    --warmup_ratio 0.05 \
    --kd_weight $KD_WEIGHT \
    --caching_dir "caching/B3_Qwen2_2B_cls" \
    --kd_loss_type "vis_cosine_reg" \
    --image_resolution "low" \
    --num_layers $NUM_LAYER \
    --projector_lr 5e-5 \
    --report_to None


# ============================================================
# 2. CHECKPOINT
# ============================================================

MODEL="$OUTPUT_DIR/checkpoint-epoch-0"

if [ ! -d "$MODEL" ]; then
    echo "ERROR: Checkpoint not found:"
    echo "$MODEL"
    exit 1
fi

echo ""
echo "============================================================"
echo "Training finished."
echo "Checkpoint:"
echo "  $MODEL"
echo "============================================================"


SUBSETS=(
    "ImageNet-1K"
    "N24News"
    "HatefulMemes"
    "VOC2007"
    "SUN397"
    "Place365"
    "ImageNet-A"
    "ImageNet-R"
    "ObjectNet"
    "Country211"
)

EVAL_OUTPUT="./MMEB-eval_outputs/FastVLM-0.5B_cls_${EXP_NAME}/"

python eval_mmeb.py \
    --model_name "$MODEL" \
    --encode_output_path "$EVAL_OUTPUT" \
    --lora True --lora_r 64 --lora_alpha 64 \
    --pooling eos \
    --model_backbone llava_qwen2 \
    --normalize True \
    --bf16 \
    --dataset_name TIGER-Lab/MMEB-eval \
    --subset_name "${SUBSETS[@]}" \
    --dataset_split test \
    --per_device_eval_batch_size 32 \
    --image_dir eval_images/ \
    --tgt_prefix_mod \
    --load_pretrained_lora True \
    --report_to none


echo ""
echo "============================================================"
echo "DONE"
echo "Experiment:"
echo "  $EXP_NAME"
echo ""
echo "Train:"
echo "  $OUTPUT_DIR"
echo ""
echo "Eval:"
echo "  $EVAL_OUTPUT"
echo "============================================================"