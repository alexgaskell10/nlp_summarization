export HOME=/vol/bitbucket/aeg19
export TOKENIZERS_PARALLELISM=True
# export PYTHONPATH=$PYTHONPATH:$NH/Covid01/apex/
DATA_DIR=/vol/bitbucket/aeg19/datasets/cnn_dm/longbart
OUT_DIR=/vol/bitbucket/aeg19/datasets/bart-pubmed/out_temp05
rm -rf $OUT_DIR

# the proper usage is documented in the README
python finetune.py \
    --model_name_or_path sshleifer/bart-tiny-random \
    --learning_rate 1e-5 \
    --gpus 1 \
    --do_predict \
    --do_train \
    --n_val 5000 \
    --val_check_interval 0.00001 \
    --sortish_sampler \
    --max_target_length 46 \
    --val_max_target_length 154 \
    --test_max_target_length 154 \
    --max_source_length 1024 \
    --train_batch_size 64 \
    --eval_batch_size 64 \
    --data_dir $DATA_DIR \
    --output_dir $OUT_DIR \
    --warmup_steps 0 \
    --attn_window 512 \
    --num_train_epochs 10 \
    $@

