# On the Summarization and Evaluation of Long Documents #

This code accompanies the MSc thesis: On the Summarization and Evaluation of Long Documents. This file documents the steps required to finetune a model, generate summaries and then run evaluation on these predictions.

## Prerequisites ##
### Environment ###
```python >= 3.6, GPU >= 12Gb```, tested on Linux only
1. Create and activate a new virtual environment
2. cd to the root of this directory
3. Run the following command to install packages: ```sh install_packages.sh```
### Data ###
1. CNN/DailyMail: [Download instructions here](https://github.com/JafferWilson/Process-Data-of-CNN-DailyMail)
2. PubMed & arXiv: [Download instructions here](https://github.com/armancohan/long-summarization)
3. Quora Question Pairs: [Download instructions here](https://www.kaggle.com/c/quora-question-pairs)
4. Annotated CNN/DailyMail dataset: [Download instructions here](https://www.kaggle.com/c/quora-question-pairs)

### Check install has worked correctly ###
A demo script is provided in ```example```. ```cd``` here and run ```sh run_example.sh``` to run this pipeline. Instructions included within this file.

## Evaluation Analysis ##
### Eval metrics ###
In project we test the relative merits of a set of eval metrics. These metrics are as follows and will here on in be referred to as the metrics.
- [BLEURT](https://github.com/google-research/bleurt)
- [MoverScore](https://github.com/AIPHES/emnlp19-moverscore)
- [BERTScore](https://github.com/Tiiiger/bert_score)
- BARTScore
- [ROUGE](https://github.com/google-research/google-research/tree/master/rouge)

### Human-Metric Correlations ###
The file here is: ```scripts/benchmarking/get_corrs.py```. This is based heavily on [this file](https://github.com/yg211/summary-reward-no-reference/blob/master/compare_reward.py). The objective here is to compute the correlation between human judgement and the metric eval scores using 2.5K human scores across 500 CNN/DailyMail summaries. The data for this task is in: ```scripts/benchmarking/learned_eval/data```.
Example usage:
- ``` cd scripts/benchmarking/ ```
- ``` python get_corrs.py -m ROUGE-1-F ```

The results in section 6.1.2 can be replicated by performing this for each evaluation metric.

### Quora Question Pairs Analysis ###
The file here is: ```scripts/benchmarking/qqp_corrs.py```. This is also based heavily on [this file](https://github.com/yg211/summary-reward-no-reference/blob/master/compare_reward.py). The objective here is to distingush whether two sentences are semantically equivalent or not. The path to the data can be set in ```scripts/benchmarking/resources.py```. Example usage:
- ``` cd scripts/benchmarking/ ```
- ``` python qqp_corrs.py ```

### Adversarial Analysis ###
The file here is: ``` scripts/benchmarking/adversarial.py ```. The objective here is to test which of the eval metrics are most sensitive to artificial corruption when applied to summaries. We performed this analysis using [PEGASUS](https://github.com/google-research/pegasus) as these have been shown to be of human quality. This process involves 2 steps:
1. Corrupt the summaries using 3 different sources of noise:
  - Random word dropping
  - Random word permutation
  - BERT mask-filling (mask random words then use BERT to in-fill these)
 2. Perform evaluation using PEGASUS's original and corrupted summaries using the (human-produced) target summaries as the ground truth.
 
**To perform step 1):**
 - ``` cd scripts/benchmarking/ ```
 - Open ```adversarial.py``` and set the SUMMS_PATH (the path of the summaries to corrupt) and OUT_DIR (where the corrupted summaries should be saved)
 - Run ``` python adversarial.py ```

**To perform step 2):**
 - Follow the instructions in **Step 3. Running evaluation** to score each summary.
 - Run the ```AdversarialAnalyzer()``` class in ```adversarial.py``` to compute the accuracy of each metric. You will have to manually add the paths to the original and corrupted summaries in this class (lines 145-158)
 
## Architecures Analysis ##

**Replicating LED Results**
This is a multi-stage pipeline as each step is GPU intensive. The steps in the pipeline are as follows:
1. Finetune a model
2. Generate the test set predictions (summaries)
3. Evaluate the test summaries

__A demo script is provided in ```example```. ```cd``` here and run ```sh run_example.sh``` to run this pipeline__

### Step 1. Finetune the LED ###
The main file is ```scrips/models/finetune.py```. This is based heavily on the [Huggingface script](https://github.com/huggingface/transformers/blob/master/examples/seq2seq/finetune.py) and this is a useful resource if you have any difficulties here. There are a number of shell scripts in ```sh_scripts/``` which are configured for many of the tasks you will be interested in with this repo. To finetine the LED to replicate results from the thesis use the following steps:
1. cd to scripts/models _# TODO: check path_
2. Open ```sh_scripts/finetune_led.sh``` 
3. Change DATA_DIR and OUT_DIR to point to where the data is stored and where you want results to be saved to
4. Run ```sh_scripts/finetune_led.sh```

This script takes approx 40hrs to run per epoch. This is because we used batch size=1 to fit onto a 12Gb GPU. If you have larger hardware this can be sped-up considerably. The best version of the model will be saved in ```$OUT_DIR/best_tfmr``` and this will be used to generate the test set summaries in the next step.

### Step 2. Generate test set summaries ###
Here you also use ```scrips/models/finetune.py```. This step should be run after finetuning as it requires a saved model in ```$OUT_DIR/best_tfmr```.
1. cd to ```scripts/models``` _# TODO: check path_
2. Open ```sh_scripts/predict.sh``` 
3. Change DATA_DIR and OUT_DIR to point to where the data is stored and where you want results to be saved to. $OUT_DIR should point to the location of the saved model you want to generate predictions using
4. Run ```sh_scripts/predict.sh```

This will generate 2 files within $OUT_DIR: 1) ```test_generations.txt``` and ```test_targets.txt```. These contain the generated summaries from the test set and the targets respectively. These files will be used for the evaluation stage.

### Step 3. Running evaluation ###
Here you use ```scripts/benchmarking/benchmark.py```. The purpose of this step is to generate evaluation scores using each of eval metrics.

Process:
1. cd to ```scripts/benchmarking```
2. Open ```sh_scripts/run_eval.sh```
3. Change SUMS_DIR and OUTDIR appropriately. OUTDIR is where the eval output will be saved to. SUMS_DIR is where the dir containing the generated summaries along with the targets. These should be named according to Step 2. above.
4. Run ```sh_scripts/run_eval.sh```

This will save the output to $OUTDIR/eval_output_raw.txt and $OUTDIR/eval_output.txt. The first contains the scores for each metric per summary; the second contains the means and standard deviations only. These are saved as a single line per directory- if you want to perform eval for a different model it will append these scores to a new line.


