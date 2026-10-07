SUBSETS=(
#   "ImageNet-1K" "N24News" "HatefulMemes" "VOC2007" "SUN397" 
#   "Place365" "ImageNet-A" "ImageNet-R" "ObjectNet" "Country211"
#   "OK-VQA" "A-OKVQA" "DocVQA" "InfographicsVQA" "ChartQA" "Visual7W"
#   "ScienceQA" "VizWiz" "GQA" "TextVQA"
  "MSCOCO" "RefCOCO" "RefCOCO-Matching" "Visual7W-Pointing"
)

# MODEL=training/FastVLM-0.5B_base_16_eos_cls/checkpoint-final
# MODEL=training/FastVLM-0.5B_vqa_struct_sigreg_SW_gmm_d1_sig1_kd10_sw0.1_numt17_tmax5_l24/checkpoint-final
MODEL=training/FastVLM-0.5B_grounding_struct_sigreg_SW_gmm_d1_sig1_kd10_sw0.1_numt17_tmax5_l24/checkpoint-final

BASE_BATCH=16
OUT_ROOT=./MMEB-eval_outputs/FastVLM-0.5B_grounding_struct_sigreg_SW_gmm_d1_sig1_kd10_sw0.1_numt17_tmax5_l24

for i in 0 1 2; do
  BATCH=$((BASE_BATCH - i))
  echo "=== Run $((i+1))/3 | batch size = $BATCH ==="

  CUDA_VISIBLE_DEVICES=0 python eval_mmeb.py \
      --model_name $MODEL \
      --encode_output_path "${OUT_ROOT}_batch${BATCH}/" \
      --lora True --lora_r 64 --lora_alpha 64 \
      --pooling eos \
      --model_backbone llava_qwen2 \
      --normalize True \
      --bf16 \
      --dataset_name TIGER-Lab/MMEB-eval \
      --subset_name "${SUBSETS[@]}" \
      --dataset_split test \
      --per_device_eval_batch_size $BATCH \
      --image_dir eval_images/ \
      --tgt_prefix_mod \
      --load_pretrained_lora True \
      --report_to none
done