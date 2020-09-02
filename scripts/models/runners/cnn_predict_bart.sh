export HOME=/vol/bitbucket/aeg19
export TOKENIZERS_PARALLELISM=True
DATA_DIR=/vol/bitbucket/aeg19/datasets/cnn_dm/longbart
OUT_DIR=/vol/bitbucket/aeg19/datasets/cnn_dm/longbart/outputwandb_bartcnnpredict_01
# rm -rf $OUT_DIR

python finetune.py \
    --model_name_or_path facebook/bart-large-cnn \
    --data_dir $DATA_DIR \
    --output_dir $OUT_DIR \
    --sortish_sampler 56 \
    --val_max_target_length 154 \
    --test_max_target_length 154 \
    --max_source_length 1024 \
    --train_batch_size 1 \
    --eval_batch_size 1 \
    --warmup_steps 0 \
    --do_predict \
    $@

