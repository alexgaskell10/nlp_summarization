export HOME=/vol/bitbucket/aeg19
export TOKENIZERS_PARALLELISM=True
DATA_DIR=/vol/bitbucket/aeg19/datasets/bart-arxiv-new-hpc/
OUT_DIR=/vol/bitbucket/aeg19/datasets/bart-arxiv-new-hpc/outputwandb_01_custom_repeat

python finetune.py \
    --model_name_or_path $OUT_DIR/best_tfmr \
    --data_dir $DATA_DIR \
    --output_dir $OUT_DIR \
    $@
