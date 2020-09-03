### This code is based on the code from summary-reward-no-reference/compare_reward.py file ###
### Link: https://github.com/yg211/summary-reward-no-reference/blob/master/compare_reward.py ###

import numpy as np
import os
import sys
from nltk import PorterStemmer, word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from tqdm import tqdm
from scipy.stats import spearmanr, pearsonr, kendalltau
import argparse

from learned_eval.scorer.data_helper.json_reader import read_sorted_scores, read_articles, read_processed_scores, read_scores
from learned_eval.helpers.data_helpers import sent2stokens_wostop, sent2tokens_wostop, sent2stokens, text_normalization
from learned_eval.scorer.auto_metrics.rouge.rouge import RougeScorer
from learned_eval.rewarder import Rewarder
from resources import ROUGE_DIR, BASE_DIR, MODEL_WEIGHT_DIR


def evaluate_metric(metric, stem, remove_stop, prompt='overall'):
    ''' Compute the correlation between the human eval scores and the scores awarded by the
        eval metric.
    '''
    assert metric in ['ROUGE-1-F', 'ROUGE-2-F', 'ROUGE-L-F', 'bert-human', 'bert-score', 'bart-score', 
        'bleurt-base', 'bleurt-lg', 'mover-1', 'mover-2', 'mover-smd', 'bert-avg-score']
    stemmed_str = "_stem" if stem else ""
    stop_str = "_removestop" if remove_stop else ""
    ranks_file_path = os.path.join('learned_eval/outputs', 'wref_{}{}{}_{}_rank_correlation.csv'.format(metric, stemmed_str, stop_str, prompt))
    print('\n====={}=====\n'.format(ranks_file_path))

    ranks_file = open(ranks_file_path, 'w')
    ranks_file.write('article,summ_id, human_score, metric_score\n')

    sorted_scores = read_sorted_scores()
    input_articles, _ = read_articles()
    corr_data = np.zeros((len(sorted_scores), 3))

    stopwords_list = set(stopwords.words("english"))
    stemmer = PorterStemmer()

    # Init the metric
    if metric == 'bert-human':
        rewarder = Rewarder(os.path.join(MODEL_WEIGHT_DIR, 'sample.model'))
    elif metric.endswith('score'):   
        from bert_score import BERTScorer
        if 'bert-score' == metric:
            rewarder = BERTScorer(lang="en", rescale_with_baseline=True, model_type='roberta-large-mnli')
        elif 'bart-score' == metric:
            rewarder = BERTScorer(lang="en", model_type="facebook/bart-large-mnli", num_layers=12)
        elif 'bert-avg' in metric:
            r1 = BERTScorer(lang="en", rescale_with_baseline=False, model_type='roberta-large')
            r2 = BERTScorer(lang="en", rescale_with_baseline=False, model_type='albert-xxlarge-v2')
            r3 = BERTScorer(lang="en", rescale_with_baseline=False, model_type='bart-large-mnli', num_layers=12)
    elif metric.startswith('bleurt'):
        from bleurt import score
        if 'base' in metric: 
            checkpoint = "bleurt-base-512"
        elif 'lg' in metric: 
            checkpoint = "bleurt-large-512"
        rewarder = score.BleurtScorer(checkpoint)
    elif metric.startswith('mover'):
        from moverscore import get_idf_dict, word_mover_score
        hyps = [s['sys_summ'] for score in sorted_scores.values() for s in score if s['sys_name'] != 'reference']
        refs = [s['sys_summ'] for score in sorted_scores.values() for s in score if s['sys_name'] == 'reference']
        idf_dict_hyp = get_idf_dict(hyps)
        idf_dict_ref = get_idf_dict(refs)
    elif 'rouge' in metric.lower():
        from rouge_score import rouge_scorer
        from rouge_score.scoring import BootstrapAggregator

    # Loop over each article and compute the correlation between human judgement
    # and the metric scores. 
    for i, (article_id, scores) in tqdm(enumerate(sorted_scores.items())):
        scores_list = [s for s in scores if s['sys_name'] != 'reference']
        human_ranks = [s['scores'][prompt] for s in scores_list]
        if len(human_ranks) < 2: 
            continue    # Must be at least 2 scores to compute the correlation
        ref_summ = scores_list[0]['ref']
        article = [entry['article'] for entry in input_articles if entry['id']==article_id][0]

        # Pre-processing (if necessary)
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

        # Clean summaries
        summ_ids = [s['summ_id'] for s in scores_list]
        sys_summs = [text_normalization(s) for s in sys_summs]
        ref_summ = text_normalization(ref_summ)
        article = text_normalization(article)

        # Compute metric scores
        if 'rouge' in metric.lower():
            auto_metric_ranks = []
            if '1' in metric:
                rouge_metric = 'rouge1'
            elif '2' in metric:
                rouge_metric = 'rouge2'
            elif 'L' in metric:
                rouge_metric = 'rougeL'
            rew_rouge = rouge_scorer.RougeScorer([rouge_metric], use_stemmer=True)
            for ss in sys_summs:
                ss = ss.replace('. ', '\n')
                ref_summ = ref_summ.replace('. ', '\n')
                score = rew_rouge.score(ref_summ, ss)
                auto_metric_ranks.append(score[rouge_metric].fmeasure)
        if metric == 'bert-human':
            auto_metric_ranks = [rewarder(ref_summ,ss) for ss in sys_summs]
        elif metric.endswith('score'):   
            if 'bert-score' == metric:
                auto_metric_ranks = [rewarder.score([ref_summ], [ss])[-1].item() for ss in sys_summs]
            elif 'bart-score' == metric:
                auto_metric_ranks = [rewarder.score([ref_summ], [ss])[-1].item() for ss in sys_summs]
            elif 'bert-avg' in metric:
                rewarder_scores = []
                for rewarder in [r1, r2, r3]:
                    r_scores = np.array([rewarder.score([ref_summ], [ss])[-1].item() for ss in sys_summs])
                    r_scores = (r_scores - np.min(r_scores)) / (np.max(r_scores) - np.min(r_scores))
                    rewarder_scores.append(r_scores)
                auto_metric_ranks = list(np.mean(rewarder_scores, axis=0))
        elif metric.startswith('bleurt'):
            auto_metric_ranks = [rewarder.score([ref_summ], [ss])[0] for ss in sys_summs]
        elif metric.startswith('mover'):
            if '1' in metric: 
                n_gram = 1
            elif '2' in metric: 
                n_gram = 2
            else: 
                raise ValueError("smd not implemented currently")
            auto_metric_ranks = [word_mover_score([ref_summ], [ss], idf_dict_ref, idf_dict_hyp,
                                stop_words=[], n_gram=n_gram, remove_subwords=True)[0] for ss in sys_summs]
   
        for sid, amr, hr in zip(summ_ids, auto_metric_ranks, human_ranks):
            ranks_file.write('{},{},{:.2f},{:.4f}\n'.format(article_id, sid, hr, amr))

        # Compute correlations
        spearmanr_result = spearmanr(human_ranks, auto_metric_ranks)
        pearsonr_result = pearsonr(human_ranks, auto_metric_ranks)
        kendalltau_result = kendalltau(human_ranks, auto_metric_ranks)
        corr_data[i, :] = [spearmanr_result[0], pearsonr_result[0], kendalltau_result[0]]

    corr_mean_all = np.nanmean(corr_data, axis=0)
    corr_std_all = np.nanstd(corr_data, axis=0)
    print('\n====={}=====\n'.format(ranks_file_path))
    print("Correlation mean on all data spearman/pearsonr/kendall: {}".format(corr_mean_all))
    print("Correlation std on all data spearman/pearsonr/kendall: {}".format(corr_std_all))

    ranks_file.flush()
    ranks_file.close()

    return ranks_file_path


def parse_args():
    ap = argparse.ArgumentParser("arguments for summary sampler")
    ap.add_argument('-m','--metric',type=str,default='ROUGE-1-F',choices=['ROUGE-1-F', 'ROUGE-2-F', 'ROUGE-L-F', 'bert-human', 'bert-score', 'bart-score', 
        'mover-1', 'mover-2', 'mover-smd', 'bleurt-base', 'bleurt-lg', 'bert-avg-score'],help='compare which metric against the human judgements')
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
    metric_scores_file = evaluate_metric(metric, stem, remove_stop, prompt)
