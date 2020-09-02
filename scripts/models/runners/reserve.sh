export HOME=/vol/bitbucket/aeg19
export TOKENIZERS_PARALLELISM=True
DATA_DIR=/vol/bitbucket/aeg19/datasets/cnn_dm/bart/predict
OUT_DIR=/vol/bitbucket/aeg19/datasets/cnn_dm/bart/predict/temp01
# MODEL_PATH=/vol/bitbucket/aeg19/datasets/bart-pubmed/outputwandb_125/best_tfmr
rm -rf $OUT_DIR

# the proper usage is documented in the README
python finetune.py \
    --model_name_or_path facebook/bart-large-cnn \
    --learning_rate 3e-5 \
    --gpus 1 \
    --do_predict \
    --do_train \
    --n_val 10 \
    --val_check_interval 0.00001 \
    --sortish_sampler \
    --max_target_length 160 \
    --val_max_target_length 200 \
    --test_max_target_length 200 \
    --max_source_length 800 \
    --train_batch_size 1 \
    --eval_batch_size 1 \
    --data_dir $DATA_DIR \
    --output_dir $OUT_DIR \
    --warmup_steps 0 \
    --seed 123 \
    $@

    # --grad_checkpointing \
    # --fp16 \
    # --gradient_accumulation_steps 4 \
    # --model_name_or_path facebook/bart-large-cnn \
        # --fp16 \
