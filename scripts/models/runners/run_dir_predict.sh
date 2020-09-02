DIR=/vol/bitbucket/aeg19/datasets/bart-pubmed-custom
DATA_DIR=$DIR
ALL_DIRS=$(ls $DIR | grep outputwandb | grep _custom)


for SUB_DIR in $ALL_DIRS
do
    echo $DIR/$SUB_DIR
    OUT_DIR=$DIR/$SUB_DIR
    python finetune.py \
        --model_name_or_path $OUT_DIR/best_tfmr \
        --data_dir $DATA_DIR \
        --output_dir $OUT_DIR \
        $@
done