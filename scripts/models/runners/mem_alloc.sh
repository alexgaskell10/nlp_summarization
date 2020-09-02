export HOME=/vol/bitbucket/aeg19
# export PYTHONPATH=$PYTHONPATH:$NH/Covid01/apex/
export TOKENIZERS_PARALLELISM=true
DATA_DIR=/vol/bitbucket/aeg19/datasets/bart-pubmed
OUT_DIR=/vol/bitbucket/aeg19/datasets/bart-pubmed/mem_alloc
# rm -rf $OUT_DIR

for ATTN in 512 256 128 64
do
    python finetune.py \
        --model_name_or_path facebook/bart-large-cnn \
        --learning_rate 3e-5 \
        --gpus 1 \
        --do_predict \
        --do_train \
        --n_val 5 \
        --val_check_interval 2e-5 \
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
        --attn_window $ATTN \
        --adam_epsilon 1e-08 \
        --model_variant longbart \
        --memory_alloc \
        $@
done

# python finetune.py \
#     --model_name_or_path facebook/bart-large-cnn \
#     --learning_rate 3e-5 \
#     --gpus 1 \
#     --do_predict \
#     --do_train \
#     --n_val 5 \
#     --val_check_interval 1e-5 \
#     --sortish_sampler \
#     --max_target_length 160 \
#     --val_max_target_length 200 \
#     --test_max_target_length 200 \
#     --max_source_length 1024 \
#     --train_batch_size 1 \
#     --eval_batch_size 1 \
#     --data_dir $DATA_DIR \
#     --output_dir $OUT_DIR \
#     --warmup_steps 0 \
#     --attn_window 64 \
#     --adam_epsilon 1e-08 \
#     --longbart_base_model longbart \
#     --memory_alloc \
#     $@
