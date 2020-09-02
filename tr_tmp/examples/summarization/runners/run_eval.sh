DIR=/vol/bitbucket/aeg19/datasets/cnn_dm/bart_2

python run_eval.py \
    $DIR/targets.source \
    $DIR/targets.hypo \
    facebook/bart-large-cnn \
    --score_path $DIR/scores.txt \
    --reference_path $DIR/targets \
    $@

# python run_eval.py \
#     /vol/bitbucket/aeg19/datasets/bart-pubmed/val.source \
#     /vol/bitbucket/aeg19/datasets/bart-pubmed/save_dir/02 \
#     /vol/bitbucket/aeg19/datasets/bart-pubmed/output/best_tfmr/ \
#     $@