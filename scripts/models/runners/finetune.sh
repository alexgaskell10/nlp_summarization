export HOME=/vol/bitbucket/aeg19
# export PYTHONPATH=$PYTHONPATH:$NH/Covid01/apex/
export TOKENIZERS_PARALLELISM=true
DATA_DIR=/vol/bitbucket/aeg19/datasets/debug
OUT_DIR=/vol/bitbucket/aeg19/datasets/debug/tmp
rm -rf /vol/bitbucket/aeg19/datasets/debug/tmp

# the proper usage is documented in the README
python finetune.py \
    --model_name_or_path facebook/bart-large-cnn \
    --learning_rate 3e-6 \
    --gpus 1 \
    --do_predict \
    --n_val 5 \
    --val_check_interval 0.001 \
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
    --model_variant longbart \
    --do_train \
    --custom_tokenizer \
    $@

    # 


