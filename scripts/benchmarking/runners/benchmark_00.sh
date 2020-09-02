export HOME=/vol/bitbucket/aeg19
OUTDIR=/vol/bitbucket/aeg19/datasets/bart-pubmed/analysis/tmp

# for metric in mover-1 mover-2 bleurt bertscore bartscore rouge
for metric in bleurt bertscore
do
    python benchmark.py \
        --metric $metric \
        --infiles manual \
        --outdir $OUTDIR \
        --run_eval \
        $@
done