export HOME=/vol/bitbucket/aeg19
export TOKENIZERS_PARALLELISM=true
DATA_DIR=/vol/bitbucket/aeg19/datasets/bart-pubmed
# MODEL_PATH=/vol/bitbucket/aeg19/datasets/bart-pubmed/outputwandb_00/best_tfmr/
OUT_DIR=/vol/bitbucket/aeg19/datasets/bart-pubmed/outputwandb_red_04
# rm -rf $OUT_DIR

# the proper usage is documented in the README
python finetune.py \
    --model_name_or_path reformer_encoder_decoder \
    --learning_rate 1e-5 \
    --gpus 1 \
    --do_predict \
    --do_train \
    --n_val 500 \
    --val_check_interval 0.1 \
    --sortish_sampler \
    --max_target_length 160 \
    --val_max_target_length 200 \
    --test_max_target_length 200 \
    --max_source_length 1024 \
    --train_batch_size 1 \
    --eval_batch_size 1 \
    --data_dir $DATA_DIR \
    --output_dir $OUT_DIR \
    --warmup_steps 0 \
    --logger wandb \
    --adam_epsilon 1e-08 \
    --longbart_base_model facebook/bart-large-cnn \
    --reformerencoderdecoder_attn_type 'local' \
    $@

    # --longbart_base_model facebook/bart-large \    
