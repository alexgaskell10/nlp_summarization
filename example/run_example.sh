# Script to demonstrate the summarization pipeline. Process is as follows:
# 1. Finetune the model
# 2. Produce test summaries using best model
# 3. Run eval on test summaries

CWD=$(pwd)

### 1. Finetune the model ###
cd ../scripts/models
sh $CWD/sh_scripts/finetune_led.sh $CWD
cd $CWD

### 2. Produce test sumamries using best model ###
cd ../scripts/models
sh $CWD/sh_scripts/predict.sh $CWD
cd $CWD

### 3. Run eval on test summaries ###
SUMS_DIR=$CWD/example_outdir
OUT_DIR=$CWD/eval_output
mkdir $OUT_DIR
cd ../scripts/benchmarking
python benchmark.py \
    --metric all \
    --infiles $SUMS_DIR \
    --outdir $OUT_DIR \
    --run_eval
cd $CWD

# This should produce a dict in ./eval_output/eval_output.txt and the metric scores should match below:
# "mover-1": [0.1659062701482688, 0.14772925661712547], 
# "mover-2": [0.24036794165750885, 0.1374096760833757], 
# "bleurt": [-0.43346463292837145, 0.17563896384337893], 
# "bertscore": [0.2700189657509327, 0.14320327037774325], 
# "bartscore": [0.5782066583633423, 0.06002901781844203], 
# "rouge1": [0.4039302768190128, 0.11840753987651191], 
# "rouge2": [0.19591877034876182, 0.11633365549493609], 
# "rougeL": [0.27764645663741094, 0.10869274367383483], 
# "rougeLsum": [0.38035611413958686, 0.11537458490944609]