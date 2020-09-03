### This code is based on the code from summary-reward-no-reference/compare_reward.py file ###
### Link: https://github.com/yg211/summary-reward-no-reference/blob/master/compare_reward.py ###

import numpy as np
import pandas as pd

import os
import sys
from tqdm import tqdm
from scipy.stats import spearmanr, pearsonr, kendalltau
import argparse
import csv

from learned_eval.scorer.auto_metrics.rouge.rouge import RougeScorer
from resources import ROUGE_DIR, BASE_DIR, MODEL_WEIGHT_DIR, QQP_DATA_PATH, QQP_OUT_PATH

from bert_score import BERTScorer
from bleurt import score
from rouge_score import rouge_scorer
from moverscore import get_idf_dict, word_mover_score

from time import time


def get_scores(nrows, metrics=None):
    ''' Get correlations between metric similarity and label similarity '''
    df = pd.read_csv(QQP_DATA_PATH, nrows=nrows)
    start_time = time()
    if not metrics:
        metrics = ['mover-1', 'mover-2', 'bleurt', 'bertscore', 'bartscore', 'rouge1', 'rouge2', 'rougeLsum',]
    for m in tqdm(metrics):
        if m.startswith('rouge'):
            scorer = rouge_scorer.RougeScorer([met for met in metrics if met.startswith('rouge')], use_stemmer=True)
            scores = [scorer.score(r, c)[m].fmeasure for c,r in zip(df.question1, df.question2)]
        elif m == 'bertscore':
            scorer = BERTScorer(lang="en", rescale_with_baseline=True, model_type='roberta-large-mnli')
            _, _, scores = scorer.score(df.question1.tolist(), df.question2.tolist())
        elif m == 'bartscore':
            scorer = BERTScorer(lang="en", model_type="facebook/bart-large-mnli", num_layers=12)
            _, _, scores = scorer.score(df.question1.tolist(), df.question2.tolist())
        elif m == 'bleurt':
            checkpoint = "bleurt-large-512"
            scorer = score.BleurtScorer(checkpoint)
            scores = scorer.score(df.question1, df.question2, batch_size=50)
        elif m.startswith('mover'):
            # Truncate long questions else moverscore gets OOM
            q1 = df['question1'].apply(lambda s: s[:300]).tolist()
            q2 = df['question2'].apply(lambda s: s[:300]).tolist()
            idf_dict_hyp = get_idf_dict(q1)
            idf_dict_ref = get_idf_dict(q2)
            if '1' in m: 
                n_gram = 1
            else:
                n_gram = 2
            scores = word_mover_score(q2, q1, idf_dict_ref, idf_dict_hyp,
                        stop_words=[], n_gram=n_gram, remove_subwords=True, batch_size=64)

        df[m] = scores
        print('\n'*10, m, '\n'*10)
        df.to_csv(QQP_OUT_PATH)


def get_corrs():
    df = pd.read_csv(QQP_OUT_PATH)
    labels = df.is_duplicate
    for metric in df.columns[7:]:
        scores = df[metric]
        spearmanr_result = spearmanr(scores, labels)
        pearsonr_result = pearsonr(scores, labels)
        kendalltau_result = kendalltau(scores, labels)
        print(f'===== {metric} =====')
        print(spearmanr_result, pearsonr_result, kendalltau_result)


def logreg():
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score
    df = pd.read_csv(QQP_OUT_PATH)
    df = df.iloc[np.random.permutation(len(df))]
    df_train = df.iloc[:len(df)//2]
    df_test = df.iloc[len(df)//2:]
    y_train = df_train['is_duplicate']
    y_test = df_test['is_duplicate']

    for metric in df.columns[7:]:
        model = LogisticRegression().fit(df_train[metric].to_numpy().reshape(-1,1), y_train)
        preds = model.predict(df_test[metric].to_numpy().reshape(-1,1))
        print(f'\n===== {metric} =====')
        print(f'Acc: {accuracy_score(y_test, preds)}, F1: {f1_score(y_test, preds)}\n')


def main():
    get_scores(nrows=50000)
    get_corrs()
    logreg()


if __name__ == '__main__':
    main()
