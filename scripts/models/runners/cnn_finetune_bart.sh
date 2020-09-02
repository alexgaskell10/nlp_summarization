export HOME=/vol/bitbucket/aeg19
export TOKENIZERS_PARALLELISM=true
DATA_DIR=/vol/bitbucket/aeg19/datasets/cnn_dm/longbart
# MODEL_PATH=/vol/bitbucket/aeg19/datasets/bart-pubmed/outputwandb_00/best_tfmr/
OUT_DIR=/vol/bitbucket/aeg19/datasets/cnn_dm/longbart/outputwandb_labelsmoothing_3
#rm -rf $OUT_DIR

# the proper usage is documented in the README
python finetune.py \
    --model_name_or_path facebook/bart-large-cnn \
    --learning_rate 3e-5 \
    --gpus 1 \
    --do_predict \
    --do_train \
    --n_val 1000 \
    --val_check_interval 0.1 \
    --sortish_sampler \
    --max_target_length 56 \
    --val_max_target_length 154 \
    --test_max_target_length 154 \
    --max_source_length 1024 \
    --train_batch_size 1 \
    --eval_batch_size 1 \
    --data_dir $DATA_DIR \
    --output_dir $OUT_DIR \
    --warmup_steps 0 \
    --logger wandb \
    --model_variant bart \
    --label_smoothing 0.1 \
    --adam_epsilon 1e-08 \
    $@


    # --longbart_base_model facebook/bart-large \    
    # --freeze_encoder \
    # --freeze_embeds \
    # --do_train \
