export HOME=/rds/general/user/aeg19/home/
DATA_DIR=/rds/general/user/aeg19/home/datasets/bart-pubmed-custom
OUT_DIR=/rds/general/user/aeg19/home/datasets/bart-pubmed/outputwandb_132

python finetune.py \
    --model_name_or_path $OUT_DIR/best_tfmr \
    --data_dir $DATA_DIR \
    --output_dir $OUT_DIR \
    $@

OUT_DIR=/rds/general/user/aeg19/home/datasets/bart-pubmed/outputwandb_138

python finetune.py \
    --model_name_or_path $OUT_DIR/best_tfmr \
    --data_dir $DATA_DIR \
    --output_dir $OUT_DIR \
    $@
