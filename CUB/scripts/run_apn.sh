rm -r outputs/test2/tensorboard

python experiments.py cub APN \
    --seed 1 \
    -ckpt "" \
    -log_dir outputs/test2 \
    -e 1000 \
    -optimizer sgd \
    -pretrained \
    -use_aux \
    -use_attr \
    -weighted_loss multiple \
    -data_dir data/CUB_processed/class_attr_data_10 \
    -n_attributes 112 \
    -attr_loss_weight 1 \
    -normalize_loss -b 64 \
    -weight_decay 0.00004 \
    -lr 0.01 \
    -scheduler_step 1000 \
    -end2end