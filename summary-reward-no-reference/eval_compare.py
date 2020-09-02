import numpy as np
import os
from nltk import PorterStemmer, word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from tqdm import tqdm
from scipy.stats import spearmanr, pearsonr, kendalltau
from pytorch_transformers import *
from sentence_transformers import SentenceTransformer
import argparse

from scorer.data_helper.json_reader import read_sorted_scores, read_articles, read_processed_scores, read_scores
from helpers.data_helpers import sent2stokens_wostop, sent2tokens_wostop, sent2stokens, text_normalization
from resources import ROUGE_DIR, BASE_DIR, MODEL_WEIGHT_DIR
from scorer.auto_metrics.rouge.rouge import RougeScorer
from rewarder import Rewarder
from bert_score import BERTScorer

import sys
sys.path.append("../bleurt")
from bleurt import score


def evaluate_metric(metric, stem, remove_stop, prompt='overall'):
    ''' metrics that use reference summaries '''
    assert metric in ['ROUGE-1-F', 'ROUGE-2-F', 'ROUGE-L-F', 'bert-human', 'bert-score', 'bleurt', 'mover-1', 'mover-2', 'mover-smd']
    stemmed_str = "_stem" if stem else ""
    stop_str = "_removestop" if remove_stop else ""
    ranks_file_path = os.path.join('outputs', 'wref_{}{}{}_{}_rank_correlation.csv'.format(metric, stemmed_str, stop_str, prompt))
    print('\n====={}=====\n'.format(ranks_file_path))

    # if os.path.isfile(ranks_file_path):
    #     return ranks_file_path

    ranks_file = open(ranks_file_path, 'w')
    ranks_file.write('article,summ_id,human_score,metric_score\n')

    sorted_scores = read_sorted_scores()
    input_articles, _ = read_articles()
    corr_data = np.zeros((len(sorted_scores), 3))

    stopwords_list = set(stopwords.words("english"))
    stemmer = PorterStemmer()

    if metric.startswith('bert'):
        if 'human' in metric:
            rewarder = Rewarder(os.path.join(MODEL_WEIGHT_DIR,'sample.model'))
        elif 'score' in metric:
            rewarder = BERTScorer(lang="en", rescale_with_baseline=True)
    elif metric == 'bleurt':
        # checkpoint = "../bleurt/bleurt/bleurt-base-512-ckpt"
        checkpoint = "../bleurt/bleurt/bleurt-large-512-ckpt"
        rewarder = score.BleurtScorer(checkpoint)
    elif metric.startswith('mover'):
        print('Make sure that your have started the mover server. Find details at https://github.com/AIPHES/emnlp19-moverscore.')
        from summ_eval.client import EvalClient
        mover_scorer = EvalClient()

    for i, (article_id, scores) in tqdm(enumerate(sorted_scores.items())):
        scores_list = [s for s in scores if s['sys_name'] != 'reference']
        human_ranks = [s['scores'][prompt] for s in scores_list]
        if len(human_ranks) < 2: continue
        ref_summ = scores_list[0]['ref']
        article = [entry['article'] for entry in input_articles if entry['id']==article_id][0]

        #region
        if stem and remove_stop:
            sys_summs = [" ".join(sent2stokens_wostop(s['sys_summ'], stemmer, stopwords_list, 'english', True)) for s in scores_list]
            ref_summ = " ".join(sent2stokens_wostop(ref_summ, stemmer, stopwords_list, 'english', True))
            article = " ".join(sent2stokens_wostop(article, stemmer, stopwords_list, 'english', True))
        elif not stem and remove_stop:
            sys_summs = [" ".join(sent2tokens_wostop(s['sys_summ'], stopwords_list, 'english', True)) for s in scores_list]
            ref_summ = " ".join(sent2tokens_wostop(ref_summ, stopwords_list, 'english', True))
            article = " ".join(sent2tokens_wostop(article, stopwords_list, 'english', True))
        elif not remove_stop and stem:
            sys_summs = [" ".join(sent2stokens(s['sys_summ'], stemmer, 'english', True)) for s in scores_list]
            ref_summ = " ".join(sent2stokens(ref_summ, stemmer, 'english', True))
            article = " ".join(sent2stokens(article, stemmer, 'english', True))
        else:
            sys_summs = [s['sys_summ'] for s in scores_list]
        #endregion

        summ_ids = [s['summ_id'] for s in scores_list]
        sys_summs = [text_normalization(s) for s in sys_summs]
        ref_summ = text_normalization(ref_summ)
        article = text_normalization(article)

        if 'rouge' in metric.lower():
            auto_metric_ranks = []
            for ss in sys_summs:
                rouge_scorer = RougeScorer(ROUGE_DIR, BASE_DIR)
                auto_metric_ranks.append(rouge_scorer(ss, ref_summ)[metric])
        elif metric.startswith('bert'):
            if 'human' in metric: 
                auto_metric_ranks = [rewarder(ref_summ,ss) for ss in sys_summs]
            elif 'score' in metric:
                auto_metric_ranks = [rewarder.score([ref_summ], [ss])[-1].item() for ss in sys_summs]
        elif metric == 'bleurt':
            auto_metric_ranks = [rewarder.score([ref_summ], [ss])[0] for ss in sys_summs]
        elif metric.startswith('mover'):
            if '1' in metric: 
                mm = 'wmd_1'
            elif '2' in metric: 
                mm = 'wmd_2'
            else: 
                mm = 'smd'
            cases = [ [[ss], sent_tokenize(article), mm] for ss in sys_summs ]
            auto_metric_ranks = mover_scorer.eval(cases)['0']

        for sid, amr, hr in zip(summ_ids, auto_metric_ranks, human_ranks):
            ranks_file.write('{},{},{:.2f},{:.4f}\n'.format(article_id, sid, hr, amr))

        spearmanr_result = spearmanr(human_ranks, auto_metric_ranks)
        pearsonr_result = pearsonr(human_ranks, auto_metric_ranks)
        kendalltau_result = kendalltau(human_ranks, auto_metric_ranks)
        corr_data[i, :] = [spearmanr_result[0], pearsonr_result[0], kendalltau_result[0]]
        print(spearmanr_result[0], np.nanmean(corr_data[:i+1, 0], axis=0))

    corr_mean_all = np.nanmean(corr_data, axis=0)
    corr_std_all = np.nanstd(corr_data, axis=0)
    print('\n====={}=====\n'.format(ranks_file_path))
    print("Correlation mean on all data spearman/pearsonr/kendall: {}".format(corr_mean_all))
    print("Correlation std dev on all data spearman/pearsonr/kendall: {}".format(corr_std_all))

    ranks_file.flush()
    ranks_file.close()

    return ranks_file_path

def parse_args():
    ap = argparse.ArgumentParser("arguments for summary sampler")
    ap.add_argument('-m','--metric',type=str,default='ROUGE-1-F',choices=['ROUGE-1-F', 'ROUGE-2-F', 'ROUGE-L-F',
        'bert-human', 'bert-score', 'mover-1', 'mover-2', 'mover-smd', 'bleurt'],help='compare which metric against the human judgements')
    ap.add_argument('-p','--prompt',type=str,default='overall',help='which human ratings you want to use as ground truth',choices=['overall','grammar'])
    ap.add_argument('-s','--stem',type=int,help='whether stem the texts before computing the metrics; 1 yes, 0 no')
    ap.add_argument('-rs','--remove_stop',type=int,help='whether remove stop words in texts before computing the metrics; 1 yes, 0 no')
    args = ap.parse_args()
    return args.metric, args.prompt, args.stem, args.remove_stop


if __name__ == '__main__':
    metric, prompt, stem, remove_stop = parse_args()
    stem = bool(stem)
    remove_stop = bool(remove_stop)

    print('\n=====Arguments====')
    print('metric: '+metric)
    print('prompt: '+prompt)
    print('stem: '+repr(stem))
    print('remove stopwords: '+repr(remove_stop))
    print('=====Arguments====\n')

    sys.argv = [sys.argv[0]]
    metric_scores_file = evaluate_metric(metric,stem,remove_stop,prompt)

'''
. ~/.envs/rl_sum/bin/activate
python eval_compare.py -m ROUGE-L-F
python eval_compare.py -m bert-score -s 0 -rs 0
'''