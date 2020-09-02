export HOME=/rds/general/user/aeg19/home/
DATA_DIR=/rds/general/user/aeg19/home/datasets/bart-pubmed
OUT_DIR=/rds/general/user/aeg19/home/datasets/bart-pubmed/outputwandb_138
#rm -rf $OUT_DIR

# the proper usage is documented in the README
python finetune.py \
    --model_name_or_path longbart \
    --learning_rate 3e-5 \
    --gpus 1 \
    --do_predict \
    --do_train \
    --n_val 500 \
    --val_check_interval 0.1 \
    --sortish_sampler \
    --max_target_length 160 \
    --val_max_target_length 200 \
    --test_max_target_length 200 \
    --max_source_length 4096 \
    --train_batch_size 1 \
    --eval_batch_size 1 \
    --data_dir $DATA_DIR \
    --output_dir $OUT_DIR \
    --warmup_steps 0 \
    --logger wandb \
    --attn_window 512 \
    --adam_epsilon 1e-08 \
    --longbart_base_model facebook/bart-large-cnn \
    $@

    # --longbart_base_model facebook/bart-large \    
    # --freeze_encoder \
    # --freeze_embeds \
