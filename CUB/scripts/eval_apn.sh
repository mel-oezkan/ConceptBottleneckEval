python3 CUB/eval_apn_maps.py  \
    -model_dirs outputs/CBMAPN_1e7CPT_CUB_1/best_model_1.pth \
    -n_attributes 112 \
    -use_attr \
    -data_dir data/CUB_processed/class_attr_data_10 \
    -image_dir data/CUB_200_2011/images \
    -part_seg_dir data/CUB_200_2011/part_segmentations \
    -log_dir outputs/CBMAPN_1e7CPT_CUB_1