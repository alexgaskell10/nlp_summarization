import os
import sys
from tqdm import tqdm
from multiprocessing import Process, Value, Array, Manager
import numpy as np
import json
from glob import glob
import re
import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
from datetime import datetime
import time

from resources import BLEURT_DIR

from bert_score import BERTScorer
from learned_eval.helpers.data_helpers import text_normalization
from bleurt import score
from rouge_score import rouge_scorer
from rouge_score.scoring import BootstrapAggregator
# from moverscore import get_idf_dict, word_mover_score

ROUGE_METRICS = ['rouge1', 'rouge2', 'rougeL', 'rougeLsum']
MODEL_METRICS = ['bleurt', 'mover-1', 'mover-2', 'bertscore', 'bartscore']
ALL_METRICS = MODEL_METRICS + ROUGE_METRICS

# region
HYPS_PATHS = [
    # ### cnn_dm
    # '../../../datasets/cnn_dm/pegasus/test.hypo', # pegasus
    # '../../../datasets/cnn_dm/bart/test.hypo', # bart
    # '../../../datasets/cnn_dm/pgn/test.hypo', # pgn
    # '../../../datasets/cnn_dm/prophetnet/output/test.hypo', # pronet
    # ### pubmed
    # '../../../datasets/pubmed/pegasus/test.hypo', # pegasus
    # ### Adversarial
    # '/vol/bitbucket/aeg19/datasets/adversarial/pubmed/dropped.txt', 
    # '/vol/bitbucket/aeg19/datasets/adversarial/pubmed/masked.txt',
    # '/vol/bitbucket/aeg19/datasets/adversarial/pubmed/permuted.txt',
    # ### Attnwin
    # '/vol/bitbucket/aeg19/datasets/bart-pubmed/outputwandb_01_attnwin/test_generations.txt',
    # '/vol/bitbucket/aeg19/datasets/bart-pubmed/outputwandb_02_attnwin/test_generations.txt',
    # '/vol/bitbucket/aeg19/datasets/bart-pubmed/outputwandb_03_attnwin/test_generations.txt',
    # '/vol/bitbucket/aeg19/datasets/bart-pubmed/outputwandb_04_attnwin/test_generations.txt',
    # ### Ad hoc
    # '/vol/bitbucket/aeg19/datasets/bart-pubmed/outputwandb_16/test_generations.txt',
    # '/vol/bitbucket/aeg19/datasets/bart-pubmed/outputwandb_21/test_generations.txt',
    # '/vol/bitbucket/aeg19/datasets/cnn_dm/longbart/outputwandb_54_cnn/test_generations.txt',
    # '/vol/bitbucket/aeg19/datasets/cnn_dm/longbart/outputwandb_08_cnn/test_generations.txt',
    # ### Custom-finetune
    # '/vol/bitbucket/aeg19/datasets/bart-pubmed-new-ic/outputwandb_01_custom_new_repeat/test_generations.txt',
    # '/vol/bitbucket/aeg19/datasets/bart-pubmed-new-ic/outputwandb_02_custom_new_repeat/test_generations.txt',
    # '/vol/bitbucket/aeg19/datasets/bart-pubmed-new-ic/outputwandb_03_custom_new/test_generations.txt',
    # '/vol/bitbucket/aeg19/datasets/bart-pubmed-new-ic/outputwandb_04_custom_new/test_generations.txt',
    # '/vol/bitbucket/aeg19/datasets/bart-pubmed-new-ic/outputwandb_05_custom_new/test_generations.txt',
    # '/vol/bitbucket/aeg19/datasets/bart-pubmed-new-ic/outputwandb_06_custom_new/test_generations.txt',
    # '/vol/bitbucket/aeg19/datasets/bart-pubmed-new-ic/outputwandb_07_custom_new/test_generations.txt',
    # ### LED Long
    # '/vol/bitbucket/aeg19/datasets/bart-pubmed-new-ic/outputwandb_143/test_generations.txt',
    # '/vol/bitbucket/aeg19/datasets/bart-pubmed-new-ic/outputwandb_new_132/test_generations.txt',
    # '/vol/bitbucket/aeg19/datasets/bart-pubmed-new-ic/outputwandb_new_138/test_generations.txt',
    # '/vol/bitbucket/aeg19/datasets/bart-pubmed-new-ic/outputwandb_new_140/test_generations.txt',
    # '/vol/bitbucket/aeg19/datasets/bart-pubmed-new-ic/outputwandb_new_144/test_generations.txt',
    # '/vol/bitbucket/aeg19/datasets/bart-pubmed-new-ic/outputwandb_new_145/test_generations.txt',
    # '/vol/bitbucket/aeg19/datasets/bart-pubmed-new-ic/outputwandb_new_146/test_generations.txt',
    # ### arXiv custom
    # '/vol/bitbucket/aeg19/datasets/bart-arxiv-new-hpc/outputwandb_01_custom_repeat/test_generations.txt',
    # '/vol/bitbucket/aeg19/datasets/bart-arxiv-new-hpc/outputwandb_02_custom/test_generations.txt',
    # '/vol/bitbucket/aeg19/datasets/bart-arxiv-new-hpc/outputwandb_03_custom/test_generations.txt',
    # '/vol/bitbucket/aeg19/datasets/bart-arxiv-new-hpc/outputwandb_04_custom/test_generations.txt',
    # '/vol/bitbucket/aeg19/datasets/bart-arxiv-new-hpc/outputwandb_05_custom/test_generations.txt',
    # '/vol/bitbucket/aeg19/datasets/bart-arxiv-new-hpc/outputwandb_06_custom/test_generations.txt',
    # '/vol/bitbucket/aeg19/datasets/bart-arxiv-new-hpc/outputwandb_07_custom/test_generations.txt',
    # ### pubmed custom predict repeats
    # '/vol/bitbucket/aeg19/datasets/bart-pubmed-custom-predicts/outputwandb_01_custom_new_repeat/set_1/test_generations.txt',
    # '/vol/bitbucket/aeg19/datasets/bart-pubmed-custom-predicts/outputwandb_01_custom_new_repeat/set_2/test_generations.txt',
    # '/vol/bitbucket/aeg19/datasets/bart-pubmed-custom-predicts/outputwandb_01_custom_new_repeat/set_3/test_generations.txt',
    # '/vol/bitbucket/aeg19/datasets/bart-pubmed-custom-predicts/outputwandb_02_custom_new_repeat/set_1/test_generations.txt',
    # '/vol/bitbucket/aeg19/datasets/bart-pubmed-custom-predicts/outputwandb_02_custom_new_repeat/set_2/test_generations.txt',
    # '/vol/bitbucket/aeg19/datasets/bart-pubmed-custom-predicts/outputwandb_02_custom_new_repeat/set_3/test_generations.txt',
    # '/vol/bitbucket/aeg19/datasets/bart-pubmed-custom-predicts/outputwandb_03_custom_new/set_1/test_generations.txt',
    # '/vol/bitbucket/aeg19/datasets/bart-pubmed-custom-predicts/outputwandb_03_custom_new/set_2/test_generations.txt',
    # '/vol/bitbucket/aeg19/datasets/bart-pubmed-custom-predicts/outputwandb_03_custom_new/set_3/test_generations.txt',
    # '/vol/bitbucket/aeg19/datasets/bart-pubmed-custom-predicts/outputwandb_04_custom_new/set_1/test_generations.txt',
    # '/vol/bitbucket/aeg19/datasets/bart-pubmed-custom-predicts/outputwandb_04_custom_new/set_2/test_generations.txt',
    # '/vol/bitbucket/aeg19/datasets/bart-pubmed-custom-predicts/outputwandb_04_custom_new/set_3/test_generations.txt',
    # '/vol/bitbucket/aeg19/datasets/bart-pubmed-custom-predicts/outputwandb_05_custom_new/set_1/test_generations.txt',
    # '/vol/bitbucket/aeg19/datasets/bart-pubmed-custom-predicts/outputwandb_05_custom_new/set_2/test_generations.txt',
    # '/vol/bitbucket/aeg19/datasets/bart-pubmed-custom-predicts/outputwandb_05_custom_new/set_3/test_generations.txt',
    # '/vol/bitbucket/aeg19/datasets/bart-pubmed-custom-predicts/outputwandb_06_custom_new/set_1/test_generations.txt',
    # '/vol/bitbucket/aeg19/datasets/bart-pubmed-custom-predicts/outputwandb_06_custom_new/set_2/test_generations.txt',
    # '/vol/bitbucket/aeg19/datasets/bart-pubmed-custom-predicts/outputwandb_06_custom_new/set_3/test_generations.txt',
    # '/vol/bitbucket/aeg19/datasets/bart-pubmed-custom-predicts/outputwandb_07_custom_new/set_1/test_generations.txt',
    # '/vol/bitbucket/aeg19/datasets/bart-pubmed-custom-predicts/outputwandb_07_custom_new/set_2/test_generations.txt',
    # '/vol/bitbucket/aeg19/datasets/bart-pubmed-custom-predicts/outputwandb_07_custom_new/set_3/test_generations.txt',
    # ### No pre-trained LongBart vs RED
    # '/vol/bitbucket/aeg19/datasets/bart-pubmed/outputwandb_red_02/test_generations.txt',
    # '/vol/bitbucket/aeg19/datasets/bart-pubmed/outputwandb_randomweights_02/test_generations.txt',
    ### arXiv LED vs bart
    # '/vol/bitbucket/aeg19/datasets/bart-arxiv-new/outputwandb_arxiv_100/test_generations.txt',
    '/vol/bitbucket/aeg19/datasets/bart-arxiv-new/outputwandb_arxiv_01/test_generations.txt',
    ### arXiv LED vs input len
    # '/vol/bitbucket/aeg19/datasets/bart-arxiv-new/outputwandb_arxiv_02_repeat/test_generations.txt',
    # '/vol/bitbucket/aeg19/datasets/bart-arxiv-new-hpc/outputwandb_new_03/test_generations.txt',
    # '/vol/bitbucket/aeg19/datasets/bart-arxiv-new-hpc/outputwandb_new_04/test_generations.txt',
    # '/vol/bitbucket/aeg19/datasets/bart-arxiv-new-hpc/outputwandb_new_05/test_generations.txt',
    # '/vol/bitbucket/aeg19/datasets/bart-arxiv-new-hpc/outputwandb_new_06/test_generations.txt',
    # '/vol/bitbucket/aeg19/datasets/bart-arxiv-new-hpc/send/outputwandb_new_07/test_generations.txt'
]
REFS_PATHS = [
    # ### cnn_dm
    # '../../../datasets/cnn_dm/pegasus/test.target', # pegasus
    # '../../../datasets/cnn_dm/bart/test.target', # bart
    # '../../../datasets/cnn_dm/pgn/test.target', # pgn
    # '../../../datasets/cnn_dm/prophetnet/output/test.target', # pronet
    # ### pubmed
    # '../../../datasets/pubmed/pegasus/test.target', # pegasus
    # ### Adversarial
    # '/vol/bitbucket/aeg19/datasets/adversarial/pubmed/test.target', 
    # '/vol/bitbucket/aeg19/datasets/adversarial/pubmed/test.target',
    # '/vol/bitbucket/aeg19/datasets/adversarial/pubmed/test.target',
    # ### Attnwin
    # '/vol/bitbucket/aeg19/datasets/bart-pubmed/outputwandb_01_attnwin/test_targets.txt',
    # '/vol/bitbucket/aeg19/datasets/bart-pubmed/outputwandb_02_attnwin/test_targets.txt',
    # '/vol/bitbucket/aeg19/datasets/bart-pubmed/outputwandb_03_attnwin/test_targets.txt',
    # '/vol/bitbucket/aeg19/datasets/bart-pubmed/outputwandb_04_attnwin/test_targets.txt',
    # ### Custom-finetune
    # '/vol/bitbucket/aeg19/datasets/bart-pubmed-new-ic/outputwandb_01_custom_new_repeat/test_targets.txt',
    # '/vol/bitbucket/aeg19/datasets/bart-pubmed-new-ic/outputwandb_02_custom_new_repeat/test_targets.txt',
    # '/vol/bitbucket/aeg19/datasets/bart-pubmed-new-ic/outputwandb_03_custom_new/test_targets.txt',
    # '/vol/bitbucket/aeg19/datasets/bart-pubmed-new-ic/outputwandb_04_custom_new/test_targets.txt',
    # '/vol/bitbucket/aeg19/datasets/bart-pubmed-new-ic/outputwandb_05_custom_new/test_targets.txt',
    # '/vol/bitbucket/aeg19/datasets/bart-pubmed-new-ic/outputwandb_06_custom_new/test_targets.txt',
    # '/vol/bitbucket/aeg19/datasets/bart-pubmed-new-ic/outputwandb_07_custom_new/test_targets.txt',
    # ### LED Long
    # '/vol/bitbucket/aeg19/datasets/bart-pubmed-new-ic/outputwandb_143/test_targets.txt',
    # '/vol/bitbucket/aeg19/datasets/bart-pubmed-new-ic/outputwandb_new_132/test_targets.txt',
    # '/vol/bitbucket/aeg19/datasets/bart-pubmed-new-ic/outputwandb_new_138/test_targets.txt',
    # '/vol/bitbucket/aeg19/datasets/bart-pubmed-new-ic/outputwandb_new_140/test_targets.txt',
    # '/vol/bitbucket/aeg19/datasets/bart-pubmed-new-ic/outputwandb_new_144/test_targets.txt',
    # '/vol/bitbucket/aeg19/datasets/bart-pubmed-new-ic/outputwandb_new_145/test_targets.txt',
    # '/vol/bitbucket/aeg19/datasets/bart-pubmed-new-ic/outputwandb_new_146/test_targets.txt',
    # ### arXiv custom
    # '/vol/bitbucket/aeg19/datasets/bart-arxiv-new-hpc/outputwandb_01_custom_repeat/test_targets.txt',
    # '/vol/bitbucket/aeg19/datasets/bart-arxiv-new-hpc/outputwandb_02_custom/test_targets.txt',
    # '/vol/bitbucket/aeg19/datasets/bart-arxiv-new-hpc/outputwandb_03_custom/test_targets.txt',
    # '/vol/bitbucket/aeg19/datasets/bart-arxiv-new-hpc/outputwandb_04_custom/test_targets.txt',
    # '/vol/bitbucket/aeg19/datasets/bart-arxiv-new-hpc/outputwandb_05_custom/test_targets.txt',
    # '/vol/bitbucket/aeg19/datasets/bart-arxiv-new-hpc/outputwandb_06_custom/test_targets.txt',
    # '/vol/bitbucket/aeg19/datasets/bart-arxiv-new-hpc/outputwandb_07_custom/test_targets.txt',
    # ### pubmed custom predict repeats
    # '/vol/bitbucket/aeg19/datasets/bart-pubmed-custom-predicts/outputwandb_01_custom_new_repeat/set_1/test_targets.txt',
    # '/vol/bitbucket/aeg19/datasets/bart-pubmed-custom-predicts/outputwandb_01_custom_new_repeat/set_2/test_targets.txt',
    # '/vol/bitbucket/aeg19/datasets/bart-pubmed-custom-predicts/outputwandb_01_custom_new_repeat/set_3/test_targets.txt',
    # '/vol/bitbucket/aeg19/datasets/bart-pubmed-custom-predicts/outputwandb_02_custom_new_repeat/set_1/test_targets.txt',
    # '/vol/bitbucket/aeg19/datasets/bart-pubmed-custom-predicts/outputwandb_02_custom_new_repeat/set_2/test_targets.txt',
    # '/vol/bitbucket/aeg19/datasets/bart-pubmed-custom-predicts/outputwandb_02_custom_new_repeat/set_3/test_targets.txt',
    # '/vol/bitbucket/aeg19/datasets/bart-pubmed-custom-predicts/outputwandb_03_custom_new/set_1/test_targets.txt',
    # '/vol/bitbucket/aeg19/datasets/bart-pubmed-custom-predicts/outputwandb_03_custom_new/set_2/test_targets.txt',
    # '/vol/bitbucket/aeg19/datasets/bart-pubmed-custom-predicts/outputwandb_03_custom_new/set_3/test_targets.txt',
    # '/vol/bitbucket/aeg19/datasets/bart-pubmed-custom-predicts/outputwandb_04_custom_new/set_1/test_targets.txt',
    # '/vol/bitbucket/aeg19/datasets/bart-pubmed-custom-predicts/outputwandb_04_custom_new/set_2/test_targets.txt',
    # '/vol/bitbucket/aeg19/datasets/bart-pubmed-custom-predicts/outputwandb_04_custom_new/set_3/test_targets.txt',
    # '/vol/bitbucket/aeg19/datasets/bart-pubmed-custom-predicts/outputwandb_05_custom_new/set_1/test_targets.txt',
    # '/vol/bitbucket/aeg19/datasets/bart-pubmed-custom-predicts/outputwandb_05_custom_new/set_2/test_targets.txt',
    # '/vol/bitbucket/aeg19/datasets/bart-pubmed-custom-predicts/outputwandb_05_custom_new/set_3/test_targets.txt',
    # '/vol/bitbucket/aeg19/datasets/bart-pubmed-custom-predicts/outputwandb_06_custom_new/set_1/test_targets.txt',
    # '/vol/bitbucket/aeg19/datasets/bart-pubmed-custom-predicts/outputwandb_06_custom_new/set_2/test_targets.txt',
    # '/vol/bitbucket/aeg19/datasets/bart-pubmed-custom-predicts/outputwandb_06_custom_new/set_3/test_targets.txt',
    # '/vol/bitbucket/aeg19/datasets/bart-pubmed-custom-predicts/outputwandb_07_custom_new/set_1/test_targets.txt',
    # '/vol/bitbucket/aeg19/datasets/bart-pubmed-custom-predicts/outputwandb_07_custom_new/set_2/test_targets.txt',
    # '/vol/bitbucket/aeg19/datasets/bart-pubmed-custom-predicts/outputwandb_07_custom_new/set_3/test_targets.txt',
    # ### No pre-trained LongBart vs RED
    # '/vol/bitbucket/aeg19/datasets/bart-pubmed/outputwandb_red_02/test_targets.txt',
    # '/vol/bitbucket/aeg19/datasets/bart-pubmed/outputwandb_randomweights_02/test_targets.txt',
    ### arXiv LED vs bart
    # '/vol/bitbucket/aeg19/datasets/bart-arxiv-new/outputwandb_arxiv_100/test_targets.txt',
    '/vol/bitbucket/aeg19/datasets/bart-arxiv-new/outputwandb_arxiv_01/test_targets.txt',
    ### arXiv LED vs input len
    # '/vol/bitbucket/aeg19/datasets/bart-arxiv-new/outputwandb_arxiv_02_repeat/test_targets.txt',
    # '/vol/bitbucket/aeg19/datasets/bart-arxiv-new-h/pc/outputwandb_new_03/test_targets.txt',
    # '/vol/bitbucket/aeg19/datasets/bart-arxiv-new-hpc/outputwandb_new_04/test_targets.txt',
    # '/vol/bitbucket/aeg19/datasets/bart-arxiv-new-hpc/outputwandb_new_05/test_targets.txt',
    # '/vol/bitbucket/aeg19/datasets/bart-arxiv-new-hpc/outputwandb_new_06/test_targets.txt',
    # '/vol/bitbucket/aeg19/datasets/bart-arxiv-new-hpc/send/outputwandb_new_07/test_targets.txt'
]
# endregion

class Benchmarker:
    def __init__(self, args):
        self.args = args
        self.refs_paths = args.refs_paths
        self.hyps_paths = args.hyps_paths
        self.outdir = args.outdir

        self.outfile = os.path.join(self.outdir, 'eval_output.txt')
        self.raw_outfile = os.path.join(self.outdir, 'eval_output_raw.txt')
        self.outfile_tmp = os.path.join(self.outdir, f'{datetime.now()}eval_output_raw.csv')
        self.save = True if args.outdir else False

        self.all_scores = []
        self.rouge_metrics = ROUGE_METRICS
        self.model_metrics = MODEL_METRICS
        self.bleurt_model = "bleurt/bleurt-large-512-ckpt"
        self.bertscore_model = 'roberta-large-mnli'
        self.bartscore_model = 'facebook/bart-large-mnli'
        
        if args.metric == 'all':
            self.metrics = self.model_metrics + self.rouge_metrics 
        elif args.metric == 'rouge':
            self.metrics = self.rouge_metrics 
        elif args.metric == 'model':
            self.metrics = self.model_metrics
        else:
            self.metrics = [args.metric]

    def load_summs(self, HYPS_PATH, REFS_PATH, trim=False):
        if 'prophetnet' in HYPS_PATH:
            self.refs = [line.strip().lower() for line in open(REFS_PATH, encoding='utf-8')]
            self.hyps = [line.strip().lower() for line in open(HYPS_PATH, encoding='utf-8')]
        else:
            self.refs = [line.strip() for line in open(REFS_PATH, encoding='utf-8')]
            self.hyps = [line.strip() for line in open(HYPS_PATH, encoding='utf-8')]

        if trim:
            self.refs = self.refs[:len(self.hyps)]
            self.hyps = self.hyps[:len(self.refs)]

        assert len(self.hyps) == len(self.refs) and len(self.hyps) >= 0, \
            f"For file {HYPS_PATH} you have {len(self.hyps)} hyps and {len(self.refs)} refs. Must be equal."

        if HYPS_PATH not in self.df_scores.hyps_path.unique():
            tmp = {**{'hyps_path': [HYPS_PATH] * len(self.hyps)}, 
                **{col:[None]*len(self.hyps) for col in self.metrics}}
            self.df_scores = pd.concat([self.df_scores, pd.DataFrame(tmp)])

    def save_as_dict(self):
        if not hasattr(self, 'df_scores'):
            self.df_scores = pd.read_csv('/vol/bitbucket/aeg19/datasets/bart-pubmed/analysis/2020-08-14 14:14:32.576500_eval_output_tmp.csv')
        df = self.df_scores

        column_ordering = ["mover-1", "mover-2", "bleurt", "bertscore", "bartscore", "rouge1", "rouge2", "rougeL", "rougeLsum"]

        for hyps_path in df.hyps_path.unique():
            rows = df.loc[df['hyps_path'] == hyps_path, column_ordering]
            self.raw_scores = {**{'hyps_path': hyps_path}, **rows.to_dict('list')}
            self.model_scores = {**{'hyps_path': hyps_path}, 
                **{col:[rows[col].mean(), rows[col].std()] for col in rows.columns}}

            self.write_to_file()

    def run_bleurt(self):
        print('\n===== BLEURT =====\n')
        sys.argv = [sys.argv[0]]
        checkpoint = os.path.join(BLEURT_DIR, self.bleurt_model)
        bleurt = score.BleurtScorer(checkpoint)

        for hyps_path, refs_path in zip(self.hyps_paths, self.refs_paths):
            self.load_summs(hyps_path, refs_path)
            start_time = time.time()    # TODO
            scores = bleurt.score(self.hyps, self.refs, batch_size=64)
            print(time.time() - start_time, torch.cuda.max_memory_allocated() / 1e9)     # TODO
            self.df_scores.loc[self.df_scores['hyps_path'] == hyps_path, 'bleurt'] = scores
            self.save_temp_csv()
            print(np.mean(scores))

        del bleurt, scores, checkpoint
        torch.cuda.empty_cache()

    def run_bertscore(self):
        print('\n===== BERTScore =====\n')
        bertscore = BERTScorer(lang="en", rescale_with_baseline=True, model_type=self.bertscore_model)

        for hyps_path, refs_path in zip(self.hyps_paths, self.refs_paths):
            self.load_summs(hyps_path, refs_path)
            start_time = time.time()    # TODO
            P, R, F1 = bertscore.score(self.hyps, self.refs, batch_size=64)
            print(time.time() - start_time, torch.cuda.max_memory_allocated() / 1e9)     # TODO
            self.df_scores.loc[self.df_scores['hyps_path'] == hyps_path, 'bertscore'] = F1.tolist()
            self.save_temp_csv()
            print(F1.mean())
        
        del P, R, F1, bertscore
        torch.cuda.empty_cache()

    def run_bartscore(self):
        print('\n===== BARTScore =====\n')
        bartscore = BERTScorer(lang="en", model_type=self.bartscore_model, num_layers=12)

        for hyps_path, refs_path in zip(self.hyps_paths, self.refs_paths):
            self.load_summs(hyps_path, refs_path)
            start_time = time.time()    # TODO
            P, R, F1 = bartscore.score(self.hyps, self.refs, batch_size=64)
            print(time.time() - start_time, torch.cuda.max_memory_allocated() / 1e9)     # TODO
            self.df_scores.loc[self.df_scores['hyps_path'] == hyps_path, 'bartscore'] = F1.tolist()
            self.save_temp_csv()
            print(F1.mean())
        
        del P, R, F1, bartscore
        torch.cuda.empty_cache()

    def run_moverscore(self):
        print('\n===== Moverscore =====\n')
        from moverscore import get_idf_dict, word_mover_score

        for hyps_path, refs_path in zip(self.hyps_paths, self.refs_paths):
            self.load_summs(hyps_path, refs_path)

            # Truncate hyps and refs if too long (bert positional embeddings max=512)
            hyps = [' '.join(h.split()[:300]) for h in self.hyps]
            refs = [' '.join(r.split()[:300]) for r in self.refs]

            idf_dict_hyp = get_idf_dict(hyps)
            idf_dict_ref = get_idf_dict(refs)

            n_grams = []
            if 'mover-1' in self.metrics:
                n_grams.append(1)
            if 'mover-2' in self.metrics:
                n_grams.append(2)

            for n in n_grams:
                start_time = time.time()    # TODO
                scores = word_mover_score(refs, hyps, idf_dict_ref, idf_dict_hyp,
                                stop_words=[], n_gram=n, remove_subwords=True, batch_size=64)
                print(time.time() - start_time, torch.cuda.max_memory_allocated() / 1e9)     # TODO
                self.df_scores.loc[self.df_scores['hyps_path'] == hyps_path, f'mover-{n}'] = scores
                self.save_temp_csv()
                # print(np.mean(scores))

        del get_idf_dict, word_mover_score, scores
        torch.cuda.empty_cache()

    def run_rouge(self):
        print('\n===== ROUGE =====\n')
        rouge = rouge_scorer.RougeScorer(self.rouge_metrics, use_stemmer=True)

        for hyps_path, refs_path in zip(self.hyps_paths, self.refs_paths):
            self.load_summs(hyps_path, refs_path)
            hyps, refs = self.hyps, self.refs

            start_time = time.time()
            scores = []
            for i, (c, r) in tqdm(enumerate(zip(hyps, refs))):
                c = c.replace('. ', '\n')
                r = r.replace('. ', '\n')
                ref = text_normalization(c)
                hyp = text_normalization(r)
                rouge_scores = rouge.score(r, c)
                scores.append([rouge_scores[m].fmeasure for m in self.rouge_metrics])

            print(time.time() - start_time)     # TODO
            self.df_scores.loc[self.df_scores['hyps_path'] == hyps_path, ROUGE_METRICS] = scores
            self.save_temp_csv()

    def write_to_file(self):
        pass    # TODO
        # with open(self.outfile, 'a+') as f:
        #     json.dump(self.model_scores, f)
        #     f.write('\n')

        # with open(self.raw_outfile, 'a+') as f:
        #     json.dump(self.raw_scores, f)
        #     f.write('\n')

    def run_eval(self):
        self.df_scores = pd.DataFrame(columns=['hyps_path'] + self.metrics)

        if any('mover' in m for m in self.metrics):
            self.run_moverscore()
        if 'bertscore' in self.metrics:
            self.run_bertscore()
        if 'bartscore' in self.metrics:
            self.run_bartscore()
        if 'bleurt' in self.metrics:
            self.run_bleurt()
        if any('rouge' in m for m in self.metrics):
            self.run_rouge()

        self.save_as_dict()

    def save_temp_csv(self):
        # self.df_scores.to_csv(self.outfile_tmp)   # TODO
        pass


class DirBenchmarker(Benchmarker):
    def __init__(self, dir_path: str, save_to_file=True):
        super().__init__(save_to_file=save_to_file)
        self.dir_path = dir_path

        self.outfile = os.path.join(dir_path, 'analysis/dir_benchmark.txt')
        if save_to_file:
            if not os.path.exists(os.path.join(dir_path, 'analysis')):
                os.makedirs(os.path.join(dir_path, 'analysis'))
            # assert not glob(self.outfile), "Outfile already exists"

    def get_summary_paths(self):
        patterns = ['outputwandb_[0-9]*', 'outputwandb_red_*']
        sub_dirs = []
        for pattern in patterns:
            sub_dirs.extend(glob(os.path.join(self.dir_path, pattern)))
        sub_dirs.sort()

        self.val_results = {}
        for sub_dir in sub_dirs:
            summaries_path, has_model_ckpt, rouge_2_score, has_test_results = self.get_best_summaries_path(sub_dir)
            self.val_results[sub_dir] = {
                'summaries_path': summaries_path,
                'has_model_ckpt': has_model_ckpt,
                'rouge_2_score': rouge_2_score,
                'has_test_results': has_test_results,
            }

    def run_eval(self):
        print('===== Getting paths of summary dirs =====\n\n')
        self.get_summary_paths()

        self.hyps_paths, self.refs_paths = [], []
        i = 0
        for dir, data in self.val_results.items():
            if data['rouge_2_score'] and data['rouge_2_score'] > 0.15:
                self.hyps_paths.append(data['summaries_path'])
                self.refs_paths.append(os.path.join(dir, 'targets.txt'))

            # i += 1
            # if i > 10: break

        print(f'===== Running eval on {len(self.hyps_paths)} sets of summaries =====\n\n')
        super().run_eval()

    def get_best_summaries_path(self, sub_dir):
        val_generation_pattern = 'val_generations_*.txt'
        val_generation_paths = glob(os.path.join(sub_dir, val_generation_pattern))

        test_results_pattern = 'test_results.txt'
        has_test_results = True if glob(os.path.join(sub_dir, test_results_pattern)) else False

        model_ckpt_pattern = 'val_avg_rouge2=*.ckpt'
        model_ckpt_path = glob(os.path.join(sub_dir, model_ckpt_pattern))
        if model_ckpt_path:
            step_count = int(re.search(r'step_count=(.*?)\.ckpt', model_ckpt_path[0]).group(1))
            rouge_2_score = float(re.search(r'avg_rouge2=(.*?)\-step_count', model_ckpt_path[0]).group(1))
            summaries_path = val_generation_paths[step_count - 2]
            return summaries_path, True, rouge_2_score, has_test_results
        
        val_results_pattern = 'val_results_*.txt'
        val_results_paths = sorted(glob(os.path.join(sub_dir, val_results_pattern)))
        if val_results_paths:
            rouge_2_scores = []
            for val_results_path in val_results_paths:
                for line in open(val_results_path, 'r'):
                    rouge_2_score = 0
                    if re.search('val_avg_rouge2:', line):
                        rouge_2_score = float(re.search('val_avg_rouge2:\s(.*)', line).group(1))                        
                        break
                rouge_2_scores.append(rouge_2_score)

            summaries_path, rouge_2_score = sorted(zip(val_generation_paths, rouge_2_scores), key=lambda x: x[1])[-1]
            return summaries_path, False, rouge_2_score, has_test_results

        return None, False, None, has_test_results


def parse_args():
    ap = argparse.ArgumentParser("arguments for summary sampler")
    ap.add_argument('-m', '--metric', type=str, default='all', required=False,
        choices=ROUGE_METRICS+MODEL_METRICS+['all', 'rouge', 'model'], help='Metrics to use for running eval')
    ap.add_argument('-i', '--infiles', type=str, nargs='+', required=True, 
        help='Paths to dirs containing hyps to be evaluated. Hyps should have name <test_generations.txt> and refs have name <test_targets.txt>. ')
    ap.add_argument('-o', '--outdir', type=str, required=True, help='Dir to write results into.')
    ap.add_argument('--run_eval', action='store_true', default=False, required=False, help='Should eval be run?.')
    ap.add_argument('--run_dir_benchmarker', action='store_true', default=False, required=False, help='Path to dir to run benchmarking on the whole dir')
    ap.add_argument('--analysis_out_dir', type=str, default=None, required=False, help='Path to saved file for metrics analyzer')
    args = ap.parse_args()

    if args.infiles[0] == 'manual':
        args.hyps_paths = HYPS_PATHS
        args.refs_paths = REFS_PATHS
    else:
        args.hyps_paths = [os.path.join(path, 'test_generations.txt') for path in args.infiles]
        args.refs_paths = [os.path.join(path, 'test_targets.txt') for path in args.infiles]

    sys.args = [sys.argv[0]]    # clear sys argv else they cause errors for bleurt
    return args

def main():
    args = parse_args()

    benchmarker = Benchmarker(args)
    if args.run_eval:
        benchmarker.run_eval()

    # if args.analysis_out_dir:
    #     if args.analysis_out_dir == "outfile":
    #         args.analysis_out_dir = benchmarker.outdir
    #     ma = MetricsAnalyzer(args)

    if args.run_dir_benchmarker:
        dir_benchmarker = DirBenchmarker(args)
        dir_benchmarker.run_eval()

class Args:
    def __init__(self):
        self.outdir = '/vol/bitbucket/aeg19/datasets/bart-pubmed/analysis'
        self.analysis_out_dir = self.outdir

if __name__ == '__main__':
    main()

    # args = Args()
    # MetricsPlotter(args)