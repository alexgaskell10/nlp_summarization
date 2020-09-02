DATA_DIR=$1/data
OUT_DIR=$1/example_outdir
rm -rf $OUT_DIR

python finetune.py \
    --model_name_or_path facebook/bart-large-cnn \
    --learning_rate 1e-5 \
    --gpus 1 \
    --do_train \
    --do_predict \
    --n_val 1000 \
    --val_check_interval 0.5 \
    --sortish_sampler \
    --max_source_length 1024 \
    --train_batch_size 1 \
    --eval_batch_size 1 \
    --data_dir $DATA_DIR \
    --output_dir $OUT_DIR \
    --warmup_steps 0 \
    --attn_window 512 \
    --adam_epsilon 1e-08 \
    --num_train_epochs 1 \
    --model_variant longbart
