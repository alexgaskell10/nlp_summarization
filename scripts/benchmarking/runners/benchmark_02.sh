export HOME=/vol/bitbucket/aeg19
OUTDIR=/vol/bitbucket/aeg19/datasets/bart-pubmed/analysis

python benchmark.py \
    --metric all \
    --infiles manual \
    --outdir $OUTDIR \
    --save_scores_as_dct \
    $@

    # --analysis_out_dir $OUTDIR \

