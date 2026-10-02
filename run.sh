######

bash scripts/train_distill_talas_jepa_cls.sh 0 1 1 0.05 8 1 29514
bash scripts/train_distill_talas_jepa_cls.sh 1 0 1 0.05 8 1 29515
bash scripts/train_distill_talas_jepa_cls.sh 1 1 10 0.1 8 1 29515

bash scripts/train_distill_vis_cosine_reg.sh 0.1 23 29525
