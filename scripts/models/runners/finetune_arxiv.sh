export HOME=/vol/bitbucket/aeg19
DATA_DIR=/vol/bitbucket/aeg19/datasets/bart-arxiv-new
OUT_DIR=/vol/bitbucket/aeg19/datasets/bart-arxiv-new/outputwandb_arxiv_02_repeat
# rm -rf $OUT_DIR

# the proper usage is documented in the README
python finetune.py \
    --model_name_or_path facebook/bart-large-cnn \
    --learning_rate 1e-5 \
    --gpus 1 \
    --do_predict \
    --do_train \
    --n_val 500 \
    --val_check_interval 0.05 \
    --sortish_sampler \
    --max_target_length 160 \
    --val_max_target_length 200 \
    --test_max_target_length 200 \
    --max_source_length 1536 \
    --train_batch_size 1 \
    --eval_batch_size 1 \
    --data_dir $DATA_DIR \
    --output_dir $OUT_DIR \
    --warmup_steps 0 \
    --attn_window 512 \
    --adam_epsilon 1e-08 \
    --model_variant longbart \
    --logger wandb \
    $@

    # --longbart_base_model facebook/bart-large \    
    # --freeze_encoder \
    # --freeze_embeds \
    # --grad_checkpointing \
    # --led \


