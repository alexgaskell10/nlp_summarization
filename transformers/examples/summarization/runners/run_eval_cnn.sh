DIR=/vol/bitbucket/aeg19/datasets/cnn_dm/bart

python run_eval.py \
    $DIR/test.source \
    $DIR/test.hypo2 \
    facebook/bart-large-cnn \
    --score_path $DIR/scores.txt \
    --reference_path $DIR/test.target \
    $@

# python run_eval.py \
#     /vol/bitbucket/aeg19/datasets/bart-pubmed/val.source \
#     /vol/bitbucket/aeg19/datasets/bart-pubmed/save_dir/02 \
#     /vol/bitbucket/aeg19/datasets/bart-pubmed/output/best_tfmr/ \
#     $@