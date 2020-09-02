DATA_DIR=$1/data
OUT_DIR=$1/example_outdir

python finetune.py \
    --model_name_or_path $OUT_DIR/best_tfmr \
    --data_dir $DATA_DIR \
    --output_dir $OUT_DIR