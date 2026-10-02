SUBSETS=(
    "ImageNet-1K"
#   "ImageNet-1K" "N24News" "HatefulMemes" "VOC2007" "SUN397" 
#   "Place365" "ImageNet-A" "ImageNet-R" "ObjectNet" "Country211"
  # "OK-VQA" "A-OKVQA" "DocVQA" "InfographicsVQA" "ChartQA" "Visual7W"
  # "ScienceQA" "VizWiz" "GQA" "TextVQA"
)


INFER_SCRIPT="infer_eval_hidden_attention.py"

# MODEL="raghavlite/B3_Qwen2_2B"
# MODEL="training/FastVLM-0.5B_cls_1.0_talas/checkpoint-epoch-0"
# MODEL="training/FastVLM-0.5B_base_16_eos_cls/checkpoint-epoch-0"
# MODEL="apple/FastVLM-0.5B"
MODEL="training/FastVLM-0.5B_1st_jepa0.05_cls/checkpoint-epoch-0"
export CUDA_VISIBLE_DEVICES=0
python $INFER_SCRIPT \
    --model_name $MODEL \
    --lora False \
    --lora_r 64 \
    --lora_alpha 64 \
    --pooling eos \
    --model_backbone llava_qwen2 \
    --normalize True \
    --bf16 \
    --dataset_name "TIGER-Lab/MMEB-eval" \
    --subset_name "${SUBSETS[@]}" \
    --dataset_split "test" \
    --image_dir "eval_images/" \
    --tgt_prefix_mod \
    --encode_output_path "infer/FastVLM-0.5B_1st_jepa0.05_cls" \
    --per_device_eval_batch_size 4 \
    --load_pretrained_lora True \
    --report_to None
