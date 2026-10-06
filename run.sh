######

bash scripts/train_distill_talas_jepa_cls.sh 0 1 1 0.1 17 5 24 29514
bash scripts/train_distill_talas_jepa_cls.sh 1 0 10 0.1 17 5 24 29515
bash scripts/train_distill_talas_jepa_cls.sh 1 1 10 0.1 17 5 24 29515


bash scripts/train_distill_talas_jepa_vqa.sh 1 1 10 0.1 17 5 24 29512

bash scripts/train_distill_talas_jepa_grounding.sh 1 1 10 0.1 17 5 24 29512

bash scripts/train_distill_talas_jepa_tea_qwen7b_cls.sh 1 1 10 0.1 17 5 24 29515

bash scripts/train_distill_talas_jepa_llava-ov_cls.sh 1 1 10 0.1 17 5 24 29512
bash scripts/train_distill_talas_jepa_llava-ov_vqa.sh 1 1 10 0.1 17 5 24 29519


bash scripts/train_distill_vis_cosine_reg.sh 0.1 23 29525
