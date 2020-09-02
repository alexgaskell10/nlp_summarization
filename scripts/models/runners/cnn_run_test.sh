export HOME=/vol/bitbucket/aeg19
export TOKENIZERS_PARALLELISM=true

DATA_DIR=/vol/bitbucket/aeg19/datasets/cnn_dm/longbart
OUT_DIR=/vol/bitbucket/aeg19/datasets/bart-pubmed/outputwandb_163_test
#rm -rf $OUT_DIR

# the proper usage is documented in the README
python finetune.py \
    --model_name_or_path longbart \
    --longbart_base_model facebook/bart-large-cnn \
    --gpus 1 \
    --do_predict \
    --sortish_sampler \
    --max_target_length 56 \
    --val_max_target_length 154 \
    --test_max_target_length 154 \
    --max_source_length 1024 \
    --eval_batch_size 8 \
    --data_dir $DATA_DIR \
    --output_dir $OUT_DIR \
    --logger wandb \
    --attn_window 512 \
    --warmup_steps 0 \
    $@


    # --learning_rate 1e-5 \
    # --adam_epsilon 1e-08 \
    # --train_batch_size 1 \
    # --n_val 1000 \
    # --val_check_interval 0.1 \
