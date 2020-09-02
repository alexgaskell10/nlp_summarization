
# Add parent directory to python path to access lightning_base.py
export PYTHONPATH="../":"${PYTHONPATH}"

export HOME=/vol/bitbucket/aeg19
DATA_DIR=/vol/bitbucket/aeg19/datasets/bart-pubmed
OUT_DIR=/vol/bitbucket/aeg19/datasets/bart-pubmed/out_temp01
rm -rf $OUT_DIR

# the proper usage is documented in the README
python finetune.py \
    --model_name_or_path sshleifer/bart-tiny-random \
    --learning_rate 3e-5 \
    --gpus 1 \
    --do_predict \
    --do_train \
    --n_val 500 \
    --val_check_interval 0.0001 \
    --sortish_sampler \
    --max_target_length 80 \
    --val_max_target_length 200 \
    --test_max_target_length 200 \
    --max_source_length 1024 \
    --train_batch_size 2 \
    --eval_batch_size 1 \
    --data_dir $DATA_DIR \
    --output_dir $OUT_DIR \
    --warmup_steps 0 \
    --freeze_embeds \
    --freeze_encoder \
    $@

    # --gradient_accumulation_steps 4 \
    # --model_name_or_path facebook/bart-large-cnn \
    # --model_name_or_path sshleifer/bart-tiny-random \
