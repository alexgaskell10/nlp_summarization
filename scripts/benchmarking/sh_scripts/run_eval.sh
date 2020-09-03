SUMS_DIR= # enter here
OUTDIR= # enter here

python benchmark.py \
    --metric all \
    --infiles $SUMS_DIR \
    --outdir $OUTDIR \
    --save_scores_as_dct \
    $@