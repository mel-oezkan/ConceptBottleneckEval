python3 experiments.py cub Concept_XtoC \
    --seed 1 -ckpt 1 -log_dir ConceptModel__Seed1/outputs/ \
    -e 1000 -optimizer sgd -pretrained -use_aux -use_attr \
    -weighted_loss multiple -data_dir ./class_attr_data_10 \
    -n_attributes 112 -normalize_loss -b 64 -weight_decay 0.00004 \
    -lr 0.01 -scheduler_step 1000 -bottleneck 

# python3 src/CUB/generate_new_data.py ExtractConcepts \
#     --model_path ConceptModel__Seed1/outputs/best_model_1.pth \
#     --data_dir CUB_processed/class_attr_data_10 \
#     --out_dir ConceptModel1__PredConcepts