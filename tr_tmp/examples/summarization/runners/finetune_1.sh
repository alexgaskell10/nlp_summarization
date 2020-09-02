
# Add parent directory to python path to access lightning_base.py
export PYTHONPATH="../":"${PYTHONPATH}"

export HOME=/vol/bitbucket/aeg19
DATA_DIR=/vol/bitbucket/aeg19/datasets/bart-pubmed
OUT_DIR=/vol/bitbucket/aeg19/datasets/bart-pubmed/outputwandb_00
# rm -rf $OUT_DIR
# --model_name_or_path=t5-base for t5

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
    --max_target_length 80 \
    --val_max_target_length 200 \
    --test_max_target_length 200 \
    --max_source_length 850 \
    --train_batch_size 2 \
    --eval_batch_size 1 \
    --data_dir $DATA_DIR \
    --output_dir $OUT_DIR \
    --logger wandb \
    --warmup_steps 0 \
    $@


# python finetune.py \
#     --model_name_or_path=facebook/bart-large-cnn \
#     --learning_rate=3e-5 \
#     --gpus 1 \
#     --do_predict \
#     --do_train \
#     --n_val 1000 \
#     --val_check_interval 0.1 \
#     --sortish_sampler \
#     --max_target_length=80 \
#     --val_max_target_length=200 \
#     --test_max_target_length=200 \
#     --max_source_length=851 \
#     --train_batch_size=2 \
#     --eval_batch_size=1 \
#     --data_dir=$DATA_DIR \
#     --output_dir=$OUT_DIR \
#     --gradient_accumulation_steps=8 \
#     --logger=wandb \
#     $@
