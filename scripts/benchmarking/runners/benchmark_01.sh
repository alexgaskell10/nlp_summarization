export HOME=/vol/bitbucket/aeg19
OUTDIR=/vol/bitbucket/aeg19/datasets/bart-pubmed/analysis

python benchmark.py \
    --metric all \
    --infiles manual \
    --outdir $OUTDIR \
    --run_eval \
    $@

    # --analysis_out_dir $OUTDIR \

