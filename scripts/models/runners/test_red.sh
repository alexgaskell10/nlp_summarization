export HOME=/vol/bitbucket/aeg19
export TOKENIZERS_PARALLELISM=True
# export PYTHONPATH=$PYTHONPATH:$NH/Covid01/apex/
DATA_DIR=/vol/bitbucket/aeg19/datasets/bart-pubmed
OUT_DIR=/vol/bitbucket/aeg19/datasets/bart-pubmed/out_temp00
rm -rf $OUT_DIR

# the proper usage is documented in the README
python finetune.py \
    --model_name_or_path reformer_encoder_decoder \
    --learning_rate 3e-5 \
    --gpus 1 \
    --do_predict \
    --do_train \
    --n_val 10 \
    --val_check_interval 0.001 \
    --sortish_sampler \
    --max_target_length 160 \
    --val_max_target_length 200 \
    --test_max_target_length 200 \
    --max_source_length 1024 \
    --train_batch_size 3 \
    --eval_batch_size 3 \
    --data_dir $DATA_DIR \
    --output_dir $OUT_DIR \
    --warmup_steps 0 \
    --longbart_base_model facebook/bart-large-cnn \
    --seed 123 \
    $@

    # --grad_checkpointing \
    # --fp16 \
    # --gradient_accumulation_steps 4 \
    # --model_name_or_path facebook/bart-large-cnn \
        # --fp16 \
