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

ROUGE_METRICS = ['rouge1', 'rouge2', 'rougeL', 'rougeLsum']
MODEL_METRICS = ['bleurt', 'mover-1', 'mover-2', 'bertscore', 'bartscore']
ALL_METRICS = MODEL_METRICS + ROUGE_METRICS


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
        self.bleurt_model = "bleurt-large-512"
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
            self.refs = [line.strip() for line in open(REFS_PATH, encoding='utf-8')][:10]
            self.hyps = [line.strip() for line in open(HYPS_PATH, encoding='utf-8')][:10]

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
        checkpoint = self.bleurt_model
        bleurt = score.BleurtScorer(checkpoint)

        for hyps_path, refs_path in zip(self.hyps_paths, self.refs_paths):
            self.load_summs(hyps_path, refs_path)
            scores = bleurt.score(self.hyps, self.refs, batch_size=64)
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
            P, R, F1 = bertscore.score(self.hyps, self.refs, batch_size=64)
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
            P, R, F1 = bartscore.score(self.hyps, self.refs, batch_size=64)
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
                scores = word_mover_score(refs, hyps, idf_dict_ref, idf_dict_hyp,
                                stop_words=[], n_gram=n, remove_subwords=True, batch_size=64)
                self.df_scores.loc[self.df_scores['hyps_path'] == hyps_path, f'mover-{n}'] = scores
                self.save_temp_csv()
                print(np.mean(scores))

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

            self.df_scores.loc[self.df_scores['hyps_path'] == hyps_path, ROUGE_METRICS] = scores
            self.save_temp_csv()

    def write_to_file(self):
        with open(self.outfile, 'a+') as f:
            json.dump(self.model_scores, f)
            f.write('\n')

        with open(self.raw_outfile, 'a+') as f:
            json.dump(self.raw_scores, f)
            f.write('\n')

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
        self.df_scores.to_csv(self.outfile_tmp)


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

    args.hyps_paths = [os.path.join(path, 'test_generations.txt') for path in args.infiles]
    args.refs_paths = [os.path.join(path, 'test_targets.txt') for path in args.infiles]

    sys.args = [sys.argv[0]]    # clear sys argv else they cause errors for bleurt
    return args


def main():
    args = parse_args()

    benchmarker = Benchmarker(args)
    if args.run_eval:
        benchmarker.run_eval()

    if args.run_dir_benchmarker:
        dir_benchmarker = DirBenchmarker(args)
        dir_benchmarker.run_eval()


if __name__ == '__main__':
    main()
