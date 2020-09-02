export HOME=/vol/bitbucket/aeg19
export TOKENIZERS_PARALLELISM=True
DATA_DIR=/vol/bitbucket/aeg19/datasets/cnn_dm/longbart
OUT_DIR=/vol/bitbucket/aeg19/datasets/cnn_dm/longbart/outputwandb_54_cnn
# rm -rf $OUT_DIR

python finetune.py \
    --model_name_or_path $OUT_DIR/best_tfmr \
    --data_dir $DATA_DIR \
    --output_dir $OUT_DIR \
    $@
