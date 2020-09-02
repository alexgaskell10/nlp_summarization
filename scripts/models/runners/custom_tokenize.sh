export HOME=/vol/bitbucket/aeg19
DATA_DIR=/vol/bitbucket/aeg19/datasets/bart-arxiv-new
OUT_DIR=/vol/bitbucket/aeg19/datasets/bart-arxiv-new/tmp

for LEN in 1024 1536 2048 2560 3072 3584 4096
do
    python finetune_1.py \
        --model_name_or_path sshleifer/bart-tiny-random \
        --data_dir $DATA_DIR \
        --output_dir $OUT_DIR \
        --max_source_length $LEN \
        --custom_tokenizer \
        --max_target_length 160 \
        --val_max_target_length 200 \
        --test_max_target_length 200 \
        --do_predict \
        --gpus 0 \
        --logger default \
        $@
done
