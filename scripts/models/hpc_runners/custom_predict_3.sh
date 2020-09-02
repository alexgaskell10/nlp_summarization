export HOME=/rds/general/user/aeg19/home/
DATA_DIR=/rds/general/user/aeg19/home/datasets/bart-pubmed-custom
OUT_DIR=/rds/general/user/aeg19/home/datasets/bart-pubmed-custom/outputwandb_04_custom
# rm -rf $OUT_DIR

python finetune.py \
    --model_name_or_path $OUT_DIR/best_tfmr \
    --data_dir $DATA_DIR \
    --output_dir $OUT_DIR \
    $@
