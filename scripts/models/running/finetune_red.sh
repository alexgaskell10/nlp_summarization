DATA_DIR= # set this
OUT_DIR= # set this

python finetune.py \
    --model_name_or_path facebook/bart-large-cnn \
    --learning_rate 1e-5 \
    --gpus 1 \
    --do_predict \
    --do_train \
    --n_val 1000 \
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
    --attn_window 512 \
    --adam_epsilon 1e-08 \
    --num_train_epochs 1 \
    --model_variant reformer_encoder_decoder \
    $@
