import json
# import torch
import os
import sys
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import pandas as pd
from glob import glob
import re
import scipy.stats as stats
from matplotlib.ticker import FormatStrFormatter

ROUGE_METRICS = ['rouge1', 'rouge2', 'rougeL', 'rougeLsum']
MODEL_METRICS = ['bleurt', 'mover-1', 'mover-2', 'bertscore', 'bartscore']
ALL_METRICS = MODEL_METRICS + ROUGE_METRICS

sns.set_palette("pastel")
PALETTE = sns.color_palette()

class Analyzer:
    def __init__(self):
        self.dir = '/vol/bitbucket/aeg19/datasets/bart-arxiv-new-hpc'
        self.out_dir = '/vol/bitbucket/aeg19/datasets/bart-pubmed/analysis/transfer/analysis'
        self.out_dir_orig = '/vol/bitbucket/aeg19/datasets/bart-pubmed/analysis'
        self.token_counts_outfile = os.path.join(self.out_dir, 'token_counts.txt')
        self.raw_scores_path = os.path.join(self.out_dir_orig, 'eval_output_raw.txt')
        self.metrics = ['bleurt', 'mover-1', 'mover-2', 'bertscore', 'bartscore', 
                            'rouge1', 'rouge2', 'rougeL', 'rougeLsum']
        self.exts = [
            ('test.source', 'test.target'), 
            ('val.source', 'val.target'), 
            ('train.source', 'train.target'),
        ]

        # self.run_counts()
        # self.load_counts()

    def run_counts(self):
        from transformers.tokenization_bart import BartTokenizer

        self.counts_dict = {}
        self.tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn', model_max_length=10000)
        for ext in self.exts:
            src, tgt = ext
            src_path = os.path.join(self.dir, src)
            tgt_path = os.path.join(self.dir, tgt)

            print(ext)
            src_data = self.load_data(src_path)
            tgt_data = self.load_data(tgt_path)

            src_counts = self.count_tokens(src_data)
            tgt_counts = self.count_tokens(tgt_data)

            # filtered_src_counts = []
            # filtered_tgt_counts = []
            # for s,t in zip(src_counts, tgt_counts):
            #     if s >= 200 and t >= 4:
            #         filtered_src_counts.append(s)
            #         filtered_src_counts.append(t)

            self.counts_dict[src] = src_counts
            self.counts_dict[tgt] = tgt_counts

            with open(self.token_counts_outfile, 'w') as f:
                json.dump(self.counts_dict, f)

    def load_data(self, path):
        return open(path, 'r').readlines()

    def count_tokens(self, data):
        tokenized = self.tokenizer.batch_encode_plus(data, truncation=False, max_length=10000)
        counts = [len(line) for line in tokenized.input_ids]
        print(np.mean(counts))
        return counts

    def load_counts(self, ext='test'):
        with open(self.token_counts_outfile, 'r') as f:
            data_dct = json.load(f)
        
        ext_data_dict = {k:v for k,v in data_dct.items() if ext in k}
        self.df_counts = pd.DataFrame(ext_data_dict)
        # print(len(self.df_counts))

    def load_raw_scores_from_file(self):
        self.raw_scores = pd.DataFrame(columns=['hyps_path'] + self.metrics)
        scores = [eval(line) for line in open(self.raw_scores_path, 'r')]
        for score in scores:
            if score['hyps_path'] not in self.raw_scores['hyps_path'].tolist():
                score['hyps_path'] = [score['hyps_path']] * len(score[self.metrics[0]])
                temp_df = pd.DataFrame(score)
                self.raw_scores = pd.concat([self.raw_scores, temp_df], ignore_index=True)
        # print(len(self.raw_scores))

    def load_summary_scores_from_file(self):
        self.stats = pd.DataFrame(columns=['hyps_path']+self.metrics)
        for path in self.scores_path:
            scores = [eval(line) for line in open(path, 'r')]
            for score in scores:
                if pattern and pattern not in score['hyps_path']:
                    continue
                self.stats = self.stats.append(score, ignore_index=True)


class SummaryMetricAnalyzer(Analyzer):
    def __init__(self):
        super().__init__()

        self.load_summs()

    def load_summs(self):
        args = {
            'paths': {
                '/vol/bitbucket/aeg19/datasets/cnn_dm/longbart/outputwandb_08_cnn/test_generations.txt': 'led'
            },
            'metrics': list(filter(lambda x: x not in ['rougeL'], self.metrics)),
        }
        self.load_raw_scores_from_file()
        df = clean_and_filter(self.raw_scores, args).reset_index(drop=True)
        
        quants = df.quantile([0.01, 0.33, 0.66, 0.99])
        # print(quants.bertscore.iloc[0])
        worst_summs = df.loc[
            (df.bartscore < quants.bartscore.iloc[0]) & \
            (df['mover-2'] < quants['mover-2'].iloc[0]) & \
            (df.rouge1 < quants.rouge1.iloc[0]) & \
            (df.rougeLsum < quants.rougeLsum.iloc[0])]

        best_summs = df.loc[
            (df.bartscore > quants.bartscore.iloc[-1]) & \
            (df['mover-2'] > quants['mover-2'].iloc[-1]) & \
            (df.rouge1 > quants.rouge1.iloc[-1]) & \
            (df.rougeLsum > quants.rougeLsum.iloc[-1])]

        good_model_bad_rouge = df.loc[
            (df.bartscore > quants.bartscore.iloc[2]) & \
            (df['mover-2'] > quants['mover-2'].iloc[2]) & \
            (df.rouge1 < quants.rouge1.iloc[1]) & \
            (df.rougeLsum < quants.rougeLsum.iloc[1])]

        bad_model_good_rouge = df.loc[
            (df.bartscore < quants.bartscore.iloc[1]) & \
            (df['mover-2'] < quants['mover-2'].iloc[1]) & \
            (df.rouge1 > quants.rouge1.iloc[2]) & \
            (df.rougeLsum > quants.rougeLsum.iloc[2])]

        # print(len(worst_summs), len(best_summs), len(good_model_bad_rouge), len(bad_model_good_rouge))

        # print(worst_summs)

        # good_model_bad_rouge: 2056, 9633, 3515
        # bad_model_good_rouge: 11024, 2330, 9824
        # best_summs: 4126, 1040, 10792
        # worst_summs: 993, 25, 6968

        # 2056, 9633, 3515, 11024, 2330, 9824, 4126, 1040, 10792, 993, 25, 6968
        # for line in 2056 9633 3515 11024 2330 9824 4126 1040 10792 993 25 6968; do echo $line; echo $(head -$line /vol/bitbucket/aeg19/datasets/cnn_dm/longbart/outputwandb_08_cnn/test_targets.txt | tail -1); echo $(head -$line /vol/bitbucket/aeg19/datasets/cnn_dm/longbart/outputwandb_08_cnn/test_generations.txt | tail -1); echo ""; done

        for line in [2056, 9633, 3515, 11024, 2330, 9824, 4126, 1040, 10792, 993, 25, 6968]:
            nums = df.iloc[line].tolist()[1:]
            out_str = ' \quad '.join([f'\ {i:.3f}' if i > 0 else f'{i:.3f}' for i in nums])
            print(out_str)


class AdversarialAnalyzer(Analyzer):
    def __init__(self):
        super().__init__()
        self.adversarial_test()

    def adversarial_test(self):
        args = {
            'pubmed': {
                'paths': {
                    'Dropped': '/vol/bitbucket/aeg19/datasets/adversarial/pubmed/dropped.txt', 
                    'Masked': '/vol/bitbucket/aeg19/datasets/adversarial/pubmed/masked.txt',
                    'Permuted': '/vol/bitbucket/aeg19/datasets/adversarial/pubmed/permuted.txt',
                },
                'orig': '../../../datasets/pubmed/pegasus/test.hypo',
            },
            'cnn_dm': {
                'paths': {
                    'Dropped': '/vol/bitbucket/aeg19/datasets/cnn_dm/adversarial/dropped.txt', 
                    'Masked': '/vol/bitbucket/aeg19/datasets/cnn_dm/adversarial/masked.txt',
                    'Permuted': '/vol/bitbucket/aeg19/datasets/cnn_dm/adversarial/permuted.txt',
                },
                'orig': '../../../datasets/cnn_dm/pegasus/test.hypo',
            },
        }
        outfile = os.path.join(self.out_dir_orig, 'adversarial_results.txt')
        self.load_raw_scores_from_file()
        test_output = {}
        for name, path in args['cnn_dm']['paths'].items():
            task_output = {}
            for metric in self.metrics:
                orig_scores = self.raw_scores[metric].loc[self.raw_scores.hyps_path == args['cnn_dm']['orig']].tolist()
                corrupted_scores = self.raw_scores[metric].loc[self.raw_scores.hyps_path == path].tolist()
                assert len(orig_scores) == len(corrupted_scores)
                preds = [old > new for old, new in zip(orig_scores, corrupted_scores)]
                p_hat = sum(preds) / len(preds)
                std = (p_hat * (1-p_hat)) / len(orig_scores)**0.5
                task_output[metric] = [p_hat, std]
            
            test_output[name] = task_output

        # save to file
        with open(outfile, 'w') as f:
            json.dump(test_output, f, indent=2)

        # print as output table
        cols = ['Metrics'] + list(args['cnn_dm']['paths'].keys())
        rows = test_output[cols[1]].keys()
        results = []
        for row in rows:
            p_data = [row] + [round(test_output[col][row][0], 3) for col in cols[1:]]
            results.append(p_data)

        df_results = pd.DataFrame(results, columns=cols).sort_values('Masked', ascending=False)
        print(df_results.to_string(index=False))


class MetricsPlotter(Analyzer):
    def __init__(self):
        super().__init__()
        # self.make_eval_plots()
        self.plot_distrs()
        # self.attn_win_plots()

    def attn_win_plots(self):
        args = {
            'paths': {
                '/vol/bitbucket/aeg19/datasets/bart-pubmed/outputwandb_01_attnwin/test_generations.txt': '512',
                '/vol/bitbucket/aeg19/datasets/bart-pubmed/outputwandb_02_attnwin/test_generations.txt': '256',
                '/vol/bitbucket/aeg19/datasets/bart-pubmed/outputwandb_03_attnwin/test_generations.txt': '128',
                '/vol/bitbucket/aeg19/datasets/bart-pubmed/outputwandb_04_attnwin/test_generations.txt': '64',
            },
            'metrics': ['bartscore', 'bertscore', 'mover-1', 'mover-2', 'rouge1', 'rouge2', 'rougeLsum',],
            'outfile': os.path.join(self.out_dir, 'figs', f'attn_win_plots.png'),
            'title': 'LED 1024: Performance by Attention Window Size \n (PubMed Test Set)'
        }
        self.load_raw_scores_from_file()
        df = clean_and_filter(self.raw_scores, args)

        # Normalize and index
        for col in df.columns[1:]:
            # Normalize
            df[col] /= df[col].std()
            # Index
            df[col] /= df.loc[df.hyps_path == '512', col].mean()

        # Preprocessing
        df = df.melt('hyps_path')
        df.columns = ['Attn window', 'Metric', 'Score']
        df['Attn window'] = df['Attn window'].astype(int)

        # Make plot
        f, ax = plt.subplots(figsize=(15, 15))
        g = sns.relplot(x="Attn window", y='Score', data=df, hue='Metric', marker='o', kind='line',
            palette='pastel', ci=None,)
        plt.xticks(df['Attn window'].unique()[::-1])
        plt.xlim(512, 64)
        # plt.subplots_adjust(top=0.8)
        g.fig.suptitle(args['title'])
        plt.savefig(args['outfile'])
        plt.clf()

    def plot_distrs(self):
        ''' Plot the distributions of the arXiv, PubMed and CNN/DM test sets '''
        paths = {
            'pubmed': [
                '/vol/bitbucket/aeg19/datasets/bart-pubmed-new/test.source',
                '/vol/bitbucket/aeg19/datasets/bart-pubmed-new/test.target',
            ],
            'arxiv': [
                '/vol/bitbucket/aeg19/datasets/bart-arxiv-new/test.source', 
                '/vol/bitbucket/aeg19/datasets/bart-arxiv-new/test.target',
            ],
            'cnn_dm': [
                '/vol/bitbucket/aeg19/datasets/cnn_dm/bart/test.source',
                '/vol/bitbucket/aeg19/datasets/cnn_dm/bart/test.target',
            ],
        }
        df = self.load_distr_counts(paths)

        args = {
            'title': f'Distribution of Document Lengths by Dataset (Test Sets Only)', 
            'row': 'Doc Type', 'col': 'Dataset',
            'outfile': os.path.join(self.out_dir, 'figs','batch_dset_dist_plots.png'),
            'xlims': [10000, 10000, 1000, 700, 700, 300],
        }

        # Preprocessing
        df = df.melt('dataset')
        df.columns = [args['col'], args['row'], 'Length']

        batch_hists(df, args)

    def load_distr_counts(self, paths):
        ''' Helper to count the lengths of documents in each of the provided datasets '''
        df = pd.DataFrame(columns=['source', 'target', 'dataset'])
        for name, path in paths.items():
            src = open(path[0], 'r').readlines()
            tgt = open(path[1], 'r').readlines()

            src_cnt = [len(l.split()) for l in src]
            tgt_cnt = [len(l.split()) for l in tgt]

            assert len(src_cnt) == len(tgt_cnt)

            tmp_dct = {'source': src_cnt, 'target': tgt_cnt, 'dataset': [name] * len(tgt_cnt)}
            df = pd.concat([df, pd.DataFrame(tmp_dct)])

        return df

    def make_eval_plots(self):
        paths = {
            'boxplots': {
                '../../../datasets/cnn_dm/bart/test.hypo': ['BART', 2],
                '../../../datasets/cnn_dm/pegasus/test.hypo': ['PEGASUS', 1],
                '../../../datasets/cnn_dm/prophetnet/output/test.hypo': ['ProphetNet', 0],
                '../../../datasets/cnn_dm/pgn/test.hypo': ['PGN', 3],
            },
            'distributions': {
                '../../../datasets/cnn_dm/prophetnet/output/test.hypo': ['ProphetNet', 0],
            },
            'corrplot': {
                '../../../datasets/cnn_dm/bart/test.hypo': ['BART', None],
                '../../../datasets/cnn_dm/pegasus/test.hypo': ['PEGASUS', None],
                '../../../datasets/cnn_dm/prophetnet/output/test.hypo': ['ProphetNet', None],
            },
        }
        self.load_raw_scores_from_file()

        # Boxplots
        args = {'title': 'Boxplots of Evaluation Metric Scores by Model \n (CNN/DailyMail Test Set)', 
                'ext': 'EVAL_', 'paths': paths['boxplots'],
                'metrics': ['bertscore', 'mover-2', 'rouge1']}
        # self.make_boxplots(args)

        # Distplot
        args = {'title': 'Distribution of Evaluation Metric Scores For ProphetNet \n (CNN/DailyMail Test Set)',
                'ext': 'EVAL_', 'paths': paths['distributions'], 
                'metrics': ['bertscore', 'bartscore', 'mover-1', 'mover-2', 'bleurt', 'rouge1', 'rouge2', 'rougeLsum',]}
        self.make_distrplot(args)

        # Corrpot
        args = {'title': 'Correlation Matrix of Scores Produced using BART, PEGASUS and ProphetNet \n (CNN/DailyMail Test Set)',
                'ext': 'EVAL_', 'paths': paths['corrplot'], 
                'metrics': ['bertscore', 'bartscore', 'mover-1', 'mover-2', 'bleurt', 'rouge1', 'rouge2', 'rougeLsum',]}
        # self.make_corrplot(args)

    def make_corrplot(self, args):
        '''Plot covariance matrix of the metrics for ProphetNet, BART and PEGASUS summaries'''
        # assert all(map(lambda x: x in args, ['paths', 'title']))

        # Add run name and select desired metrics only
        df = clean_and_filter(self.raw_scores, args).drop(['hyps_path'], axis=1)

        # Preprocess and plot
        args['kind'] = 'corrplot'
        args['outfile'] = os.path.join(self.out_dir_orig, 'plots', f'{args["ext"]}{args["kind"]}_plot.png')
        self.corrplot(df, args)

    def make_distrplot(self, args):
        assert all(map(lambda x: x in args, ['paths', 'title']))

        # Add run name and select desired metrics only
        df = clean_and_filter(self.raw_scores, args)

        # Pre-process and plot
        args['kind'] = 'batch-hist'
        args['outfile'] = os.path.join(self.out_dir_orig, 'plots', f'{args["ext"]}{args["kind"]}_plot.png')
        args['rows'], args['cols'] = 2, 4
        self.batch_histplot(df, args)

    def make_boxplots(self, args):
        assert all(map(lambda x: x in args, ['paths', 'title', 'metrics']))

        # Add run name and select desired metrics only
        df = clean_and_filter(self.raw_scores, args)

        # Add column for dataset
        df['dset'] = 'CNN_DailyMail' 
        df['dset'].loc[df.hyps_path.str.contains('pubmed')] = 'PubMed'

        # Get order to display hue
        args['hue_order'] = map(lambda x: x[0], sorted(args['paths'].values(), key=lambda x: x[1]))

        # Generate plots
        args['kind'] = 'box'
        if not 'ext' in args:
            args['ext'] = None
        args['outfile'] = os.path.join(self.out_dir, 'plots', f'{args["ext"]}{args["kind"]}_plot.png')
        self.barplot(df, args)

    def make_bart_led_plots(self):
        paths = {
            '/vol/bitbucket/aeg19/datasets/cnn_dm/longbart/outputwandb_08_cnn/test_generations.txt': 'cnndm_LED_1024',
            '/vol/bitbucket/aeg19/datasets/cnn_dm/longbart/outputwandb_54_cnn/test_generations.txt': 'cnndm_BART_1024',
            '/vol/bitbucket/aeg19/datasets/bart-pubmed/outputwandb_16/test_generations.txt': 'pubmed_BART_1024',
            '/vol/bitbucket/aeg19/datasets/bart-pubmed/outputwandb_127/test_generations.txt': 'pubmed_LED_1024',
        }

    def get_stats(self, stats):
        assert all(stat in ['mean','std'] for stat in stats)

        self.means = pd.DataFrame(columns=['hyps_path']+self.metrics)
        self.stds = pd.DataFrame(columns=['hyps_path']+self.metrics)
        
        for run_name, df in self.raw_scores.items():
            if 'mean' in stats:
                data = df.mean().to_dict()
                data['hyps_path'] = run_name
                self.means = self.means.append(data, ignore_index=True)

            if 'std' in stats:
                data = df.std().to_dict()
                data['hyps_path'] = run_name
                self.stds = self.stds.append(data, ignore_index=True)

        # print(self.means, '\n', self.stds)

    def batch_histplot(self, data, args):
        assert all(map(lambda x: x in args, ['rows', 'cols', 'paths', 'title', 'metrics', 'outfile']))
        f, axes = plt.subplots(args['rows'], args['cols'], figsize=(12, 7), sharex=False, sharey=True,)
        for r in range(args['rows']):
            for c in range(args['cols']):
                # Make dist plot
                col = args['metrics'][r * args['cols'] + c]
                d = data[col]
                ax = axes[r,c]
                sns.distplot(d, kde=False, bins=50, ax=ax,
                    hist_kws={'weights': np.full(len(data), 1/len(data))})

                # Annotate with statistics
                upper_q, lower_q = d.quantile([0.25,0.75]).to_list()
                mean = d.mean()
                median = d.median()
                ax.axvline(upper_q, color=PALETTE[6], linestyle='--')
                ax.axvline(lower_q, color=PALETTE[5], linestyle='--')
                ax.axvline(mean, color=PALETTE[1], linestyle='--')
                ax.axvline(median, color=PALETTE[4], linestyle='--')
        
        # Annotate
        f.text(0.08, 0.5, 'Share of documents', va='center', rotation='vertical')
        f.text(0.5, 0.04, 'Score', ha='center')
        f.legend({'Upper_quart': upper_q, 'Lower_quart': lower_q, 'Mean': mean, 'Median': median},
            loc=(0.08,0.86), framealpha=0.4)     
        f.suptitle('Distribution of ProphetNet Output Scores by Metric \n (CNN/DailyMail Test Set)')

        plt.savefig(args['outfile'])
        plt.clf()

    def corrplot(self, df, args):
        df = df.dropna().reset_index(drop=True)
        corr = df.corr(method='pearson')
        mask = np.triu(np.ones_like(corr, dtype=np.bool))
        f, ax = plt.subplots(figsize=(15, 13))
        cmap = sns.diverging_palette(220, 10, as_cmap=True)
        sns.heatmap(corr, mask=mask, cmap=cmap, #vmax=.3, center=0,
                    square=True, linewidths=.9, cbar_kws={"shrink": .5},
                    annot=True)
        plt.xticks(rotation=60, size=15)
        plt.yticks(rotation=60, size=15)
        plt.title(args['title'], size=20)
        plt.savefig(args['outfile'])
        plt.clf()

    def barplot(self, data, args):
        assert all(map(lambda x: x in args, ['outfile', 'title']))

        # Reshape data
        df = data.dropna().reset_index(drop=True)
        df = df.melt(id_vars=['hyps_path', 'dset'])
        df.columns = ['Run', 'Dset', 'Metric', 'Score',]

        # Make plot
        f, ax = plt.subplots(figsize=(15, 15))
        ax = sns.catplot(x='Metric', y='Score', hue='Run', data=df, 
            kind=args['kind'], palette="Pastel1", showfliers=False, hue_order=args['hue_order'])
        plt.xticks(rotation=60)
        plt.title(args['title'])
        plt.ylabel('Score')
        plt.subplots_adjust(bottom=0.25, right=0.7, top=0.9)
        plt.savefig(args['outfile'])
        plt.clf()

    def scatter(self):
        df = self.raw_scores.dropna().reset_index(drop=True)
        x = df.loc[df['hyps_path'] == 'pegasus', 'mover-1']
        y = df.loc[df['hyps_path'] == 'pegasus', 'mover-2']
        plt.scatter(x, y)
        plt.savefig(os.path.join(self.out_dir, 'plots', 'scatterplot.png'))
        plt.clf()

    def plot_dist(self):
        ''' Make pdf / cdf of token counts '''
        args = {
            'test.source': {'xlim': 25000, 'bins': 200},
            'val.source': {'xlim': 25000, 'bins': 200},
            'train.source': {'xlim': 25000, 'bins': 200},
            'test.target': {'xlim': 1000, 'bins': 50},
            'val.target': {'xlim': 1000, 'bins': 50},
            'train.target': {'xlim': 1000, 'bins': 50},
        }
        for col in self.df_counts.columns:
            token_counts = self.df_counts[col]
            out_path = os.path.join(self.out_dir, 'figs', f'dist_plot.{col}.png')
            plot_data = self.args[col]

            # Plot pdf plot
            sns.distplot(token_counts, kde=False, bins=plot_data['bins'])
            plt.xlim(0, plot_data['xlim'])
            plt.xlabel('Document length (tokens)')
            plt.xlabel('Document no. of tokens')
            plt.title(f'Histogram of document lengths for {col} on the PubMed dataset')
            plt.savefig(out_path)
            plt.clf()
        
    def plot_memory_usage(self):
        ''' Plots memory requirement to finetune LED by input length and attention window size '''
        with open(os.path.join(self.out_dir, 'mem_analysis.txt')) as f:
            data = [json.loads(line) for line in f if len(line) > 1]

        # Reformat as df
        reformatted = {k:[] for k in data[0].keys()}
        for line in data:
            for k,v in line.items():
                reformatted[k].append(v)
        df = pd.DataFrame(reformatted)
        
        # Preprocessing
        df.columns = [c.replace('attn', 'attention').replace('_',' ').replace('max ','') for c in df.columns]        
        outfile = os.path.join(self.out_dir, 'figs', f'pubmed_mem_alloc_lineplot.png')
        y = df['mem alloc gb']
        x1 = df['source length']
        x2 = df['attention window']

        # Make plots
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 7), sharey=True)
        sns.lineplot(x=x1.name, y=y.name, data=df, hue='attention window', marker='o', 
            palette='pastel', ax=ax1)
        ax1.set_xlabel(x1.name.capitalize())
        ax1.set_ylabel('Memory consumption (Gb)')
        ax1.set_xticks(df[x1.name].unique())

        sns.lineplot(x=x2.name, y=y.name, data=df, hue='source length', marker='o', 
            palette='pastel', ax=ax2)
        ax2.set_xlabel(x2.name.capitalize())
        ax2.set_xticks(df[x2.name].unique())

        f.suptitle('Maximum Memory Consumption to Finetune the LED \n (by Attention Window and Input Document Length, batch size = 1)')
        f.savefig(outfile)
        plt.clf()


class RandomStartLoader(Analyzer):
    def __init__(self):
        super().__init__()
        self.paths = {
            'led_random_starts': {
                'title' : 'LED vs Input Length Using Random Uniform Starting Point',
                'show_legend': False,
                'start_type': 'Random',
                'dset': 'PubMed',
                'paths' : {
                    '/vol/bitbucket/aeg19/datasets/bart-pubmed-new-ic/outputwandb_01_custom_new_repeat/test_generations.txt': 'led_1024',
                    '/vol/bitbucket/aeg19/datasets/bart-pubmed-new-ic/outputwandb_02_custom_new_repeat/test_generations.txt': 'led_1536',
                    '/vol/bitbucket/aeg19/datasets/bart-pubmed-new-ic/outputwandb_03_custom_new/test_generations.txt': 'led_3584',
                    '/vol/bitbucket/aeg19/datasets/bart-pubmed-new-ic/outputwandb_04_custom_new/test_generations.txt': 'led_2560',
                    '/vol/bitbucket/aeg19/datasets/bart-pubmed-new-ic/outputwandb_05_custom_new/test_generations.txt': 'led_2048',
                    '/vol/bitbucket/aeg19/datasets/bart-pubmed-new-ic/outputwandb_06_custom_new/test_generations.txt': 'led_3072',
                    '/vol/bitbucket/aeg19/datasets/bart-pubmed-new-ic/outputwandb_07_custom_new/test_generations.txt': 'led_4096',
                }
            },
            'led_long': {
                'title' : 'LED vs Input Length',
                'show_legend': True,
                'start_type': 'Beginning',
                'dset': 'PubMed',
                'paths': {
                    "/vol/bitbucket/aeg19/datasets/bart-pubmed-new-ic/outputwandb_143/test_generations.txt": 'led_1024',
                    "/vol/bitbucket/aeg19/datasets/bart-pubmed-new-ic/outputwandb_new_132/test_generations.txt": 'led_2048',
                    "/vol/bitbucket/aeg19/datasets/bart-pubmed-new-ic/outputwandb_new_138/test_generations.txt": 'led_1536',
                    "/vol/bitbucket/aeg19/datasets/bart-pubmed-new-ic/outputwandb_new_140/test_generations.txt": 'led_4096',
                    "/vol/bitbucket/aeg19/datasets/bart-pubmed-new-ic/outputwandb_new_144/test_generations.txt": 'led_3072',
                    "/vol/bitbucket/aeg19/datasets/bart-pubmed-new-ic/outputwandb_new_145/test_generations.txt": 'led_2560',
                    "/vol/bitbucket/aeg19/datasets/bart-pubmed-new-ic/outputwandb_new_146/test_generations.txt": 'led_3584',
                },
            },
            'arxiv_random_starts': {
                'title' : 'LED vs Input Length Using Random Uniform Starting Point',
                'show_legend': False,
                'start_type': 'Random',
                'dset': 'arXiv',
                'paths' : {
                    "/vol/bitbucket/aeg19/datasets/bart-arxiv-new-hpc/outputwandb_01_custom_repeat/test_generations.txt": 'led_1024', 
                    "/vol/bitbucket/aeg19/datasets/bart-arxiv-new-hpc/outputwandb_02_custom/test_generations.txt": 'led_1536', 
                    "/vol/bitbucket/aeg19/datasets/bart-arxiv-new-hpc/outputwandb_04_custom/test_generations.txt": 'led_2048', 
                    "/vol/bitbucket/aeg19/datasets/bart-arxiv-new-hpc/outputwandb_05_custom/test_generations.txt": 'led_2560', 
                    "/vol/bitbucket/aeg19/datasets/bart-arxiv-new-hpc/outputwandb_06_custom/test_generations.txt": 'led_3072', 
                    "/vol/bitbucket/aeg19/datasets/bart-arxiv-new-hpc/outputwandb_03_custom/test_generations.txt": 'led_3584', 
                    "/vol/bitbucket/aeg19/datasets/bart-arxiv-new-hpc/outputwandb_07_custom/test_generations.txt": 'led_4096', 
                }
            },
            'arxiv_long': {
                'title' : 'LED vs Input Length',
                'show_legend': False,
                'start_type': 'Beginning',
                'dset': 'arXiv',
                'paths' : {
                    "/vol/bitbucket/aeg19/datasets/bart-arxiv-new/outputwandb_arxiv_01/test_generations.txt": 'led_1024',
                    "/vol/bitbucket/aeg19/datasets/bart-arxiv-new/outputwandb_arxiv_02_repeat/test_generations.txt": 'led_1536',
                    "/vol/bitbucket/aeg19/datasets/bart-arxiv-new-hpc/outputwandb_new_03/test_generations.txt": 'led_2048',
                    "/vol/bitbucket/aeg19/datasets/bart-arxiv-new-hpc/outputwandb_new_04/test_generations.txt": 'led_2560',
                    "/vol/bitbucket/aeg19/datasets/bart-arxiv-new-hpc/outputwandb_new_05/test_generations.txt": 'led_3072',
                    "/vol/bitbucket/aeg19/datasets/bart-arxiv-new-hpc/outputwandb_new_06/test_generations.txt": 'led_3584',
                    "/vol/bitbucket/aeg19/datasets/bart-arxiv-new-hpc/send/outputwandb_new_07/test_generations.txt": 'led_4096',
                }
            },
        }
        
        # self.load_random_start_info_from_files_arxiv()

    def load_files(self):
        self.files = glob(os.path.join(self.dir, 'test*_randomstartinfo.txt'))

    def load_random_starts_csv(self, ext="pubmed"):
        self.random_starts = pd.read_csv(os.path.join(self.out_dir, f'{ext.lower()}_random_start.csv'))

    def load_random_start_info_from_files(self):
        self.load_counts('test')
        self.df = self.df_counts
        for file in self.files:
            print(file)
            file_ext = file[54:79]
            
            random_start_data = [json.loads(f) for f in open(file)][0]
            rsd = {f'{file_ext}_{k}':v for k,v in random_start_data.items()}
            rsd_df = pd.DataFrame(rsd)
            
            assert len(rsd_df) == len(self.df)

            self.df = pd.concat([self.df, rsd_df], axis=1)

        self.df.to_csv(os.path.join(self.out_dir, 'pubmed_random_start.csv'))

    def load_random_start_info_from_files_arxiv(self):
        self.load_files()
        for file in self.files:
            file_ext = file[49:73]
            
            random_start_data = [json.loads(f) for f in open(file)][0]
            rsd = {f'{file_ext}__{k}':v for k,v in random_start_data.items()}
            rsd_df = pd.DataFrame(rsd)
            
            if hasattr(self, 'df'):
                assert len(rsd_df) == len(self.df)
                self.df = pd.concat([self.df, rsd_df], axis=1)
            else:
                self.df = rsd_df

        self.df['test.source'] = self.df['test.source_customba1024__tokenized_seq_len'] + self.df['test.source_customba1024__num_tokens_to_remove']
        self.df.to_csv(os.path.join(self.out_dir, 'arxiv_random_start.csv'))
        print(os.path.join(self.out_dir, 'arxiv_random_start.csv'))

    def merge_dfs(self, path):

        # Load both dataframes
        dset = 'arxiv' if 'arxiv' in list(path.keys())[0] else 'pubmed'
        if not hasattr(self, 'random_starts'):
            self.load_random_starts_csv(dset)
        if not hasattr(self, 'raw_scores'):
            self.load_raw_scores_from_file()

        # Clean path
        df = self.raw_scores.loc[self.raw_scores.hyps_path.isin(path.keys())]
        for k,v in path.items():
            df = df.replace(k, v)

        # Select relevant columns only
        doc_len = parse_seqlen(path)
        df_random_starts = self.random_starts[ ['test.source'] + \
            [c for c in self.random_starts.columns if doc_len in c] ]
        
        # Confirm the two dataframes are equivalently ordered
        # self.check_scores_aligned(path)

        # Merge both dfs
        df_1 = df.reset_index(drop=True)
        df_2 = df_random_starts.iloc[:len(df)].reset_index(drop=True)
        self.df_starts_and_scores = pd.concat([df_1, df_2], axis=1)

    def check_scores_aligned(self, path):
        import subprocess
        from rouge_score import rouge_scorer
        if not hasattr(self, 'tokenizer'):
            from transformers.tokenization_bart import BartTokenizer
            self.tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn', model_max_length=10000)

        path_trunc = '/vol/bitbucket/aeg19/datasets/bart-pubmed'
        path_orig = '/vol/bitbucket/aeg19/datasets/bart-pubmed-new'

        for ext in ['target', 'source']:
            # Step 1: check short and long versions of test sums are the same
            trunc_path = os.path.join(path_trunc, f'test.{ext}')
            trunc_first_line = subprocess.check_output(['head', '-1', trunc_path]).decode("utf-8") 
            trunc_last_line = subprocess.check_output(['tail', '-1', trunc_path]).decode("utf-8")

            orig_path = os.path.join(path_orig, f'test.{ext}')
            orig_first_line = subprocess.check_output(['head', '-1', orig_path]).decode("utf-8")
            intermed = subprocess.Popen(['head', '-5975', orig_path], stdout=subprocess.PIPE)   # use 5975th here as final few hundred are tuncated in short version for some reason...
            orig_last_line = subprocess.check_output(['tail', '-1'], stdin=intermed.stdout).decode("utf-8") 

            assert trunc_first_line[:500].strip() in orig_first_line.strip()
            assert trunc_last_line[:500].strip() in orig_last_line.strip()

            # Step 2: compute number of tokens in first and last lines. Check these match previously recorded counts
            data = [orig_first_line, orig_last_line]
            counts = self.count_tokens(data)

            assert counts[0] == self.random_starts[f'test.{ext}'].iloc[0]
            assert counts[1] == self.random_starts[f'test.{ext}'].iloc[5974]

            # Step 3: confirm rouge scores for first and last lines match recorded rouge scores
            if ext == 'target':
                # First load hyp and target summaries' first and last lines
                hyp_path = list(path.keys())[0]
                hyp_first_line = subprocess.check_output(['head', '-1', hyp_path]).decode("utf-8") 
                hyp_last_line = subprocess.check_output(['tail', '-1', hyp_path]).decode("utf-8")
                hyps = [hyp_first_line, hyp_last_line]

                ref_path = hyp_path.replace('test_generations.txt', 'test_targets.txt')
                ref_first_line = subprocess.check_output(['head', '-1', ref_path]).decode("utf-8") 
                intermed = subprocess.Popen(['head', '-1', ref_path], stdout=subprocess.PIPE)   # use 5975th here as final few hundred are tuncated in short version for some reason...
                ref_last_line = subprocess.check_output(['tail', '-1'], stdin=intermed.stdout).decode("utf-8")
                refs = [ref_first_line, ref_last_line]

                # Check the refs are ordered the same as the original summaries (remove spaces to account for different tokenization)
                # assert ref_first_line[:500].replace(' ','') in trunc_first_line.replace(' ','')
                # assert ref_last_line[:500].replace(' ','') in trunc_last_line.replace(' ','')

                # Compute Rouge scores and check these match existing rouge scores
                rouge_metrics = ['rouge1', 'rouge2', 'rougeL']
                rouge = rouge_scorer.RougeScorer(rouge_metrics, use_stemmer=True)
                rouge_scores_raw = [rouge.score(h, r) for h,r in zip(hyps, refs)]
                rouge_scores = [[d[m].fmeasure for m in rouge_metrics] for d in rouge_scores_raw]
                
                # Check they match recorded scores
                df = self.raw_scores.loc[self.raw_scores['hyps_path'] == hyp_path, rouge_metrics]
                # # print(df)
                # print(df.iloc[-1].to_numpy(), np.array(rouge_scores[1]))

                # df['tmp'] = df.rouge1 - 0.23300970873786406
                # df = df.reset_index(drop=True)
                # print(df.reset_index(drop=True).loc[df.tmp.abs() < 1e-5])
                print((df.iloc[-1].to_numpy(), np.array(rouge_scores[1])))

                # assert all((df.iloc[0].to_numpy() - np.array(rouge_scores[0])) < 1e-4)
                # assert all((df.iloc[-1].to_numpy() - np.array(rouge_scores[1])) < 1e-4 )

        # hyps_path = '/vol/bitbucket/aeg19/datasets/bart-pubmed-new-ic/outputwandb_01_custom_new_repeat/test_generations.txt'
        # refs_path = '/vol/bitbucket/aeg19/datasets/bart-pubmed-new-ic/outputwandb_01_custom_new_repeat/test_targets.txt'

        # hyps = open(hyps_path, 'r').readlines()
        # refs = open(refs_path, 'r').readlines()
        # rouge_metrics = ['rouge1', 'rouge2', 'rougeL']
        # rouge = rouge_scorer.RougeScorer(rouge_metrics, use_stemmer=True)
        # rouge_scores_raw = [rouge.score(h, r) for h,r in zip(hyps[-100:], refs[-100:])]
        # rouge_scores = [[d[m].fmeasure for m in rouge_metrics] for d in rouge_scores_raw]

        # path = '/vol/bitbucket/aeg19/datasets/bart-pubmed-new-ic/outputwandb_01_custom_new_repeat/test_generations.txt'
        # df = self.raw_scores.loc[self.raw_scores['hyps_path'] == path, rouge_metrics]

        # [print(rouge_scores[-r]) for r in range(10)]
        # print(df.tail())

        # print(len(df), len(hyps))
        # sys.exit()

    def load_random_starts(self, paths):
        df = pd.DataFrame(columns=['tokenized_seq_len', 'num_tokens_to_remove', 'starting_position', 'length', 'set'])
        for k,v in paths.items():
            model_len, set_num = v.split('_')
            file_name = k.replace('test_generations.txt', f'test.source_customba{model_len}_randomstartinfo.txt')
            
            # Load data
            raw_dict = [json.loads(f) for f in open(file_name)][0]
            if set_num == '1':
                raw_dict = {k:v[:1000] for k,v in raw_dict.items()}
            length = len(raw_dict['tokenized_seq_len'])
            assert length == 1000
            tmp_df = pd.DataFrame({**raw_dict, **{'length': [model_len]*length, 'set': [set_num]*length}})
            df = pd.concat([df, tmp_df])

        return df


class RandomStartPlotter(RandomStartLoader):
    def __init__(self):
        super().__init__()
        self.output_metrics = ['bartscore', 'bertscore', 'mover-1', 'mover-2', 'rouge1', 'rouge2', 'rougeLsum', 'bleurt',]
        # self.plot_start_scores(self.paths['arxiv_random_starts'])
        # self.plot_start_scores(self.paths['led_long'])
        # self.plot_random_starts()
        # self.plot_output_table(self.paths, 'arxiv')
        self.make_aggr_corr_lineplot({k:v for k,v in self.paths.items() if 'arxiv' in k.lower()})
        # self.make_aggr_corr_norm_startpos_lineplot({'arxiv_random_starts': self.paths['arxiv_random_starts']})
        # self.make_aggr_corr_norm_startpos_lineplot({'led_random_starts': self.paths['led_random_starts']})
        # self.make_aggr_corr_norm_startpos_lineplot({'led_long': self.paths['led_long']})
        # self.plot_distrs()
        # self.random_starts_repeats()
        # self.plot_norm_start_by_model(self.paths['led_random_starts'], 'normalized_starting_position')
        # self.plot_norm_start_by_model(self.paths['led_long'], 'source_len')
        # self.plot_norm_start_by_model(self.paths['arxiv_random_starts'], 'normalized_starting_position')
        # self.make_output_table(self.paths['arxiv_long'])

    def make_aggr_corr_lineplot(self, args):
        self.load_raw_scores_from_file()
        summary_corrs = {m: [] for m in ['hyps_path', 'expr_name'] + self.output_metrics}
        for expt_name, info in args.items():

            for k,v in info['paths'].items():
                # Load data
                self.merge_dfs( {k: v} )
                df_tmp = self.df_starts_and_scores

                # Save correlations between scores and doc length for each model
                summary_corrs['expr_name'].append(info['start_type'])
                summary_corrs['hyps_path'].append(v)
                for col in self.output_metrics:
                    corr = df_tmp[col].corr(df_tmp['test.source'])
                    summary_corrs[col].append(corr)
            
        # Make doc length plots
        df = pd.DataFrame(summary_corrs)

        # Lineplots
        args['rebase'] = True
        self.aggr_corr_lineplot(df.copy(), args)
        args['rebase'] = False
        self.aggr_corr_lineplot(df.copy(), args)

    def make_aggr_corr_norm_startpos_lineplot(self, args):
        self.output_metrics = ['bartscore', 'bertscore', 'mover-1', 'mover-2', 'rouge1', 'rouge2', 'rougeLsum', 'bleurt',]
        assert len(args) == 1

        self.load_raw_scores_from_file()
        summary_corrs = {m: [] for m in ['hyps_path', 'expr_name'] + self.output_metrics}
        stds = {m: [] for m in ['hyps_path', 'expr_name'] + self.output_metrics}
        for expt_name, info in args.items():
            for k,v in info['paths'].items():
                # Load data
                dset = 'arxiv' if 'arxiv' in k else 'pubmed'
                self.merge_dfs( {k: v} )
                df_tmp = self.df_starts_and_scores

                # Save correlations between scores and doc length for each model
                summary_corrs['expr_name'].append(info['start_type'])
                summary_corrs['hyps_path'].append(v)

                starting_pos_label = f'test.source_customba{v.split("_")[1]}__starting_position'
                ref_col = df_tmp[starting_pos_label] / df_tmp['test.source']
                
                for col in self.output_metrics:
                    corr = df_tmp[col].corr(ref_col)
                    summary_corrs[col].append(corr)
            
        # Make doc length plots
        df = pd.DataFrame(summary_corrs).sort_values('hyps_path')

        # Index
        for expr in df.expr_name.unique():
            # index
            for col in df.columns[2:]:
                df.loc[:, col] /= df.loc[df.hyps_path == 'led_1024', col].mean()

        df = df.drop(['bleurt', 'expr_name'], axis=1)
        args['dset'] = list(args.values())[0]["dset"]
        args['outfile'] = os.path.join(self.out_dir, 'figs', 'norm_starting_pos', f'{args["dset"]}_aggr_corr_normstart_lineplot.png')
        args['ylabel'] = 'Correlation of Eval Scores and Normalized Starting Position \n (Pearson r, rebased)'

        # Preprocessing
        args['plot_title'] = f'Evolution of the Correlation Between Evaluation Scores and Normalized \n' + \
            f'Starting Position by Model Length ({args["dset"]} test set)'
        df = df.melt(id_vars=['hyps_path'])
        df.columns = ['Model length (tokens)', 'Metric', args['ylabel'],]
        df['Model length (tokens)'] = df['Model length (tokens)'].str[-4:]

        # Make plot
        g = sns.relplot(x="Model length (tokens)", y=args['ylabel'], data=df, hue='Metric', marker='o', kind='line',
            palette='pastel', ci=None)

        plt.subplots_adjust(top=0.8)
        g.fig.suptitle(args['plot_title'])
        plt.savefig(args['outfile'])
        plt.clf()
        print(args['outfile'])

    def aggr_corr_barplot(self, df, args):
        df = df.copy()
        args['outfile'] = os.path.join(self.out_dir, 'figs', f'pubmed_aggr_corr_barplot.png')
        args['ylabel'] = 'Correlation of Eval Scores and Doc. Length \n (Pearson r)'
        
        # Preprocessing
        args['plot_title'] = f'Comparison of Correlations between Output Scores and Document Length by Model Length \n (PubMed test set)'
        df = df.melt(id_vars=['hyps_path', 'expr_name'])
        df.columns = ['Model length', "Start type", 'Metric', args['ylabel'],]
        df['Model length'] = df['Model length'].str[-4:]

        # Make plot
        data = df.loc[df['Model length'].isin(['1024', '2048', '3072', '4096'])]
        g = sns.catplot(x="Metric", y=args['ylabel'], data=data, hue='Start type', kind='bar',
            palette='pastel', col="Model length", ci=None, col_wrap=2)

        plt.subplots_adjust(top=0.8)
        g.fig.suptitle(args['plot_title'])
        plt.savefig(args['outfile'])
        plt.clf()

    def aggr_corr_lineplot(self, df, args):
        # Index so all lines begin at one
        args['dset'] = 'arXiv' if 'arxiv' in list(args.keys())[0] else 'PubMed'
        if args['rebase']:
            for expr in df.expr_name.unique():
                # index
                for col in df.columns[2:]:
                    df.loc[df.expr_name == expr, col] = df.loc[df.expr_name == expr, col] + 1 / (df.loc[(df.expr_name == expr) & (df.hyps_path == 'led_1024'), col] + 2).mean()

            df = df.drop(['bleurt'], axis=1)
            args['outfile'] = os.path.join(self.out_dir, 'figs', f'{args["dset"]}_aggr_corr_lineplot.png')
            args['ylabel'] = 'Correlation of Eval Scores and Doc. Length \n (Pearson r, rebased)'
        else:
            args['outfile'] = os.path.join(self.out_dir, 'figs', f'{args["dset"]}_aggr_corr_lineplot_raw.png')
            args['ylabel'] = 'Correlation of Eval Scores and Doc. Length \n (Pearson r)'
        
        # Preprocessing
        args['plot_title'] = f'Evolution of the Correlation Between Evaluation Scores and Document Length by Model Length \n ({args["dset"]} test set)'
        df = df.melt(id_vars=['hyps_path', 'expr_name'])
        df.columns = ['Model length (tokens)', "Start type", 'Metric', args['ylabel'],]
        df['Model length (tokens)'] = df['Model length (tokens)'].str[-4:]

        # Make plot
        g = sns.relplot(x="Model length (tokens)", y=args['ylabel'], data=df, hue='Metric', marker='o', kind='line',
            palette='pastel', col="Start type", ci=None, col_order=["Beginning", "Random"])

        plt.subplots_adjust(top=0.8)
        g.fig.suptitle(args['plot_title'])
        plt.savefig(args['outfile'])
        plt.clf()

    def make_output_table(self, args):
        args['metrics'] = ['bartscore', 'bertscore', 'mover-1', 'mover-2', 'bleurt', 'rouge1', 'rouge2', 'rougeLsum',]
        paths = args['paths']
        self.load_raw_scores_from_file()
        df = clean_and_filter(self.raw_scores, args)

        means = df.groupby('hyps_path').mean()
        stds = df.groupby('hyps_path').std()

        # Print output table
        print(means, '\n')

        # Print output table in latex format
        to_latex(means, stds)

    def plot_norm_start_by_model(self, args, feature='normalized_starting_position'):
        ''' Make plots showing correlation between scores and [doc_len, starting position]
        '''
        assert feature in ['normalized_starting_position', 'source_len']
        df = None
        for k,v in args['paths'].items():
            path = {k:v}
            # Load merged df
            self.merge_dfs(path)
            tmp_df = self.df_starts_and_scores
            doc_len = parse_seqlen(path)

            # Clean col names:
            cols = tmp_df.columns
            cols = [re.search('__(.*)', c).group(1) if '__' in c else c for c in cols]
            cols = [c.replace('test.','') + '_len' if 'test.' in c else c for c in cols]
            tmp_df.columns = cols

            # Correct columns
            if feature == 'normalized_starting_position':
                tmp_df['normalized_starting_position'] = tmp_df['starting_position'] / tmp_df['source_len']
            tmp_df = tmp_df[ ['hyps_path', 'mover-1', 'bertscore', 'rouge1', feature] ]

            if df is None:
                df = tmp_df
            else:
                df = pd.concat([df, tmp_df])

        # Batch scatter: normed starting pos vs score
        args= {'feature': 'norm_starting_pos' if feature == 'normalized_starting_position' else 'source_len', 
            'xlim': 1, 'dset':args["dset"]}
        args['outfile'] = os.path.join(self.out_dir, 'figs', args['feature'], f'{args["dset"]}_norm_start_pos_scatter.png')
        args['rows'], args['cols'] = 2, 4
        # args['metrics'] = ['bertscore', 'bartscore', 'mover-1', 'mover-2', 'bleurt', 'rouge1', 'rouge2', 'rougeLsum']
        args['metrics'] = ['bertscore', 'mover-1', 'rouge1']
        args['xlabel'] = 'Starting position (normalized)' if feature == 'normalized_starting_position' else 'Document Length'
        args['title'] = f'Output Scores Against {args["xlabel"]} by Metric using the LED \n ({args["dset"]} Test Set)'
        
        df = df.melt(id_vars=['hyps_path', feature])
        df.columns = ['Model length', args['xlabel'], 'Metric', 'Score']
        batch_scatter_2(df, args)

    def plot_start_scores(self, args):
        ''' Make plots showing correlation between scores and [doc_len, starting position]
        '''
        for k,v in args['paths'].items():
            path = {k:v}
            # Load merged df
            self.merge_dfs(path)
            df = self.df_starts_and_scores
            doc_len = parse_seqlen(path)

            # Clean col names:
            cols = df.columns
            cols = [re.search('__(.*)', c).group(1) if '__' in c else c for c in cols]
            cols = [c.replace('test.','') + '_len' if 'test.' in c else c for c in cols]
            df.columns = cols

            ## Make corr table
            # outfile = os.path.join(self.out_dir, 'figs', f'random_starts_{doc_len}_corr_table.png')
            # corr_table(df, outfile)
        
            # Batch scatter: input len vs score
            args = {**{'feature': 'source_len', 'xlim': 20000,}, **args}
            args['outfile'] = os.path.join(self.out_dir, 'figs', args['feature'], f'pubmed_{args["start_type"]}_{doc_len}.png')
            args['rows'], args['cols'] = 2, 4
            args['metrics'] = ['bertscore', 'bartscore', 'mover-1', 'mover-2', 'bleurt', 'rouge1', 'rouge2', 'rougeLsum']
            args['title'] = f'LED {doc_len}: Output Scores Against Input Doc. Length by Metric \n (PubMed Test Set)'
            args['xlabel'] = 'Input doc length'
            batch_scatter(df, args)

            # Batch scatter: normed starting pos vs score
            if args['start_type'] == 'random-start':
                args= {'feature': 'norm_starting_pos', 'xlim': 1,}
                args['outfile'] = os.path.join(self.out_dir, 'figs', args['feature'], f'pubmed_{args["start_type"]}_{doc_len}.png')
                args['rows'], args['cols'] = 2, 4
                args['metrics'] = ['bertscore', 'bartscore', 'mover-1', 'mover-2', 'bleurt', 'rouge1', 'rouge2', 'rougeLsum']
                args['title'] = f'LED {doc_len}: Output Scores Against Normalized Starting Position by Metric \n (PubMed Test Set)'
                df[args['feature']] = df['starting_position'] / df['source_len']
                args['xlabel'] = 'Normalized starting position'
                batch_scatter(df, args)

    def plot_random_starts(self):
        ''' Plot the distibutions of the random starts data '''
        self.load_random_starts_csv()
        df = self.random_starts
        source_lens = df['test.source']
        normalized_starting_pos_dct, starting_pos_dct = {}, {}
        for i in range(1024, 4097, 1024):
            starting_pos_col = [c for c in df.columns if f'source_customba{i}' in c and 'starting_position' in c][0]
            starting_pos = df[starting_pos_col]

            assert len(starting_pos) == len(source_lens)

            normalized_starting_pos = starting_pos / source_lens

            normalized_starting_pos_dct[i] = normalized_starting_pos.tolist()
            starting_pos_dct[i] = starting_pos.tolist()

        # region
        plot_hists(
            normalized_starting_pos_dct, 
            {'xlim': 1, 'bins': 25, 'title': f'Distribution of Normalized Random Starting Positions \n (PubMed Test Set, total = {len(source_lens)})', 'xlabel': 'Starting position (normalized by document length)'},
            os.path.join(self.out_dir, 'figs','pubmed_random_start_normalized.png'),
            max_i=2048,
        )
        plot_hists(
            starting_pos_dct,
            {'xlim': 10000, 'bins': 500, 'title': f'Distribution of Random Starting Positions \n (PubMed Test Set, total = {len(source_lens)})', 'xlabel': 'Starting position (tokens)'},
            os.path.join(self.out_dir, 'figs','pubmed_random_start_raw.png'),
            max_i=2048,
        )
        plot_dist(
            source_lens, 
            {'xlim': 10000, 'bins': 500, 'title': f'Distribution of Document Lengths \n (PubMed Test Set, total = {len(source_lens)})', 'xlabel': 'Document length (tokens)'},
            os.path.join(self.out_dir, 'figs','pubmed_doc_lens.png')
        )
        # endregion

        # Process data
        df = pd.DataFrame(normalized_starting_pos_dct).melt()
        df.columns = ['Input length', 'Start position (normalized)']
        args = {'rows': 2, 'cols': 2,
            'outfile': os.path.join(self.out_dir, 'figs', f'norm_starting_pos_distr.png')
        }
        input_lens = np.sort(df['Input length'].unique())

        assert all(map(lambda x: x in args, ['rows', 'cols', 'outfile']))
        f, axes = plt.subplots(args['rows'], args['cols'], figsize=(12, 7), sharex=True, sharey=False,)
        for r in range(args['rows']):
            for c in range(args['cols']):
                # Make dist plot
                d = df.loc[df['Input length'] == input_lens[r + args['rows']*c], 'Start position (normalized)']
                print(d.loc[d == 0].count() / d.count())
                ax = axes[c,r]
                sns.distplot(d, kde=False, bins=50, ax=ax, 
                    hist_kws={'weights': np.full(len(d), 1/len(d))})
                ax.set_xlabel('')
                ax.title.set_text(f'Model length: {input_lens[r + args["rows"]*c]}')
        
        # Annotate
        f.text(0.06, 0.5, 'Share of documents', va='center', rotation='vertical', size=14)
        f.text(0.5, 0.04, 'Start position (normalized)', ha='center', size=14)
        f.suptitle('Distribution of Normalized Starting Positions by LED Length \n (PubMed Test Set)', size=18)

        print(args['outfile'])
        plt.savefig(args['outfile'])
        plt.clf()

    def plot_output_table(self, info_dict, dset):
        ''' ... '''
        self.load_raw_scores_from_file()

        metrics = ['bartscore', 'bertscore', 'mover-1', 'mover-2', 'bleurt', 'rouge1', 'rouge2', 'rougeLsum']
        df = pd.DataFrame(columns=metrics)
        for args in info_dict.values():
            if args['dset'].lower() != dset.lower():
                continue

            args['metrics'] = metrics
            args['outfile'] = os.path.join(self.out_dir, 'figs', f'{args["dset"]}_{args["title"]}.png')
            args['plot_title'] = f'LED Input Length vs Evaluation Scores \n (Normalized and Rebased, {args["dset"]} test set)'

            df_tmp = clean_and_filter(self.raw_scores, args)
            df_tmp['start_type'] = args['start_type']
            df = pd.concat([df, df_tmp], axis=0)
        
        self.plot_inp_len_scores(df, args)

    def plot_inp_len_scores(self, df, args):
        ''' Plot showing scores vs input length for LED plots'''
        assert all(map(lambda x: x in args, ['plot_title', 'outfile']))

        # Normalize and index
        for start_type in df.start_type.unique():
            for col in filter(lambda x: x not in ['hyps_path', 'start_type'], df.columns):
                # Normalize
                df.loc[df.start_type == start_type, col] /= df.loc[df.start_type == start_type, col].std()
                # Index
                df.loc[df.start_type == start_type, col] -= df.loc[(df.start_type == start_type) & (df.hyps_path == 'led_1024'), col].mean() - 1

        # Preprocessing
        df = df.drop(['bleurt'], axis=1)
        df = df.melt(id_vars=['hyps_path', 'start_type'])
        df.columns = ['hyps_path', 'start_type', 'Metric', 'Score',]
        df['Input length'] = df['hyps_path'].str[-4:]

        # Make plot
        g = sns.relplot(x="Input length", y="Score", data=df, hue='Metric', marker='o', col='start_type',
            palette='pastel', ci=None, kind='line', col_order=['Beginning', 'Random'],)
        g.fig.subplots_adjust(top=0.8)
        g.fig.suptitle(args['plot_title'])
        plt.savefig(args['outfile'])
        plt.clf()
        print(args['outfile'])

    def random_starts_repeats(self):
        dirs = {
            '/vol/bitbucket/aeg19/datasets/bart-pubmed-custom-predicts/outputwandb_01_custom_new_repeat': '1024',
            '/vol/bitbucket/aeg19/datasets/bart-pubmed-custom-predicts/outputwandb_02_custom_new_repeat': '1536',
            '/vol/bitbucket/aeg19/datasets/bart-pubmed-custom-predicts/outputwandb_03_custom_new': '3584',
            '/vol/bitbucket/aeg19/datasets/bart-pubmed-custom-predicts/outputwandb_04_custom_new': '2560',
            '/vol/bitbucket/aeg19/datasets/bart-pubmed-custom-predicts/outputwandb_05_custom_new': '2048',
            '/vol/bitbucket/aeg19/datasets/bart-pubmed-custom-predicts/outputwandb_06_custom_new': '3072',
            '/vol/bitbucket/aeg19/datasets/bart-pubmed-custom-predicts/outputwandb_07_custom_new': '4096',
        }
        paths = {}
        for k,v in dirs.items():
            for i in range(3):
                paths[os.path.join(k, f'set_{i+1}', 'test_generations.txt')] = f'{v}_{i+1}'

        args = {'paths': paths, 'metrics': list(filter(lambda x: x not in ['rougeL'], self.metrics))}

        # Load data
        self.load_raw_scores_from_file()
        df = clean_and_filter(self.raw_scores, args)

        # Preprocess
        df[['length', 'set']] = df.hyps_path.str.split("_", n=1, expand=True)
        df = df.drop(['hyps_path'], axis=1)

        # Load random starts data
        df_random_starts = self.load_random_starts(paths)
        assert len(df) == len(df_random_starts)

        # Merge dfs
        df = pd.concat([df.reset_index(drop=True), df_random_starts.reset_index(drop=True)], axis=1)
        assert len(df) == len(df_random_starts)

        # Compute corrs
        df = df.iloc[:,:-2].drop(['num_tokens_to_remove', 'tokenized_seq_len'], axis=1)
        n_rows = len(df.loc[df.set == '1'])
        corrs = []
        for row in range(n_rows):
            # for col in filter(lambda x: x not in ['rougeL'], self.metrics):
            for col in ['bartscore']:
                row_inds = [row, row + n_rows, row + 2*n_rows]
                x = df.loc[row_inds, col]
                y = df.loc[row_inds, 'starting_position']
                corr = stats.pearsonr(x, y)[0]
                if np.isnan(corr):
                    corrs.append(0)
                else:
                    corrs.append(corr)

        print(np.mean(corrs))


def batch_scatter_2(data, args):
    assert all(map(lambda x: x in args, ['outfile', 'rows', 'cols', 'metrics', 'title', 'xlabel']))

    f = sns.lmplot(x=args['xlabel'], y="Score", data=data,
        col='Metric', row='Model length', col_order=args['metrics'], row_order=np.sort(data['Model length'].unique()),
        scatter_kws={'alpha': 0.2}, line_kws={'color':PALETTE[3], 'linestyle':'--'}, ci=None)

    for r, row in enumerate(np.sort(data['Model length'].unique())):
        for c, col in enumerate(args['metrics']):
            ax = f.axes[r,c]
            y = data.loc[(data['Model length'] == row) & (data['Metric'] == col), 'Score']
            x = data.loc[(data['Model length'] == row) & (data['Metric'] == col), args['xlabel']]
            annot = (lambda m: f' Pearson r: {m[0]:.3f} \n p = {m[1]:.2E}')(stats.pearsonr(x, y))
            ax.text(0.5, 0.85, annot, transform=ax.transAxes, size=12)
            ax.set_xlabel('')
            ax.set_ylabel('')
            if args['xlabel'] == 'Document Length':
                ax.set_xlim(0,10000)

    plt.subplots_adjust(top=0.9, left=0.1, bottom=0.1)
    f.fig.suptitle(args['title'], size=24)
    f.fig.text(0.04, 0.5, 'Metric Score', va='center', rotation='vertical', size=18)
    f.fig.text(0.5, 0.04, args['xlabel'], ha='center', size=18)
    plt.savefig(args['outfile'])
    plt.clf()

def plot_dist(data, args, out_path):
    sns.distplot(data, kde=False, bins=args['bins'], hist_kws={'weights': np.full(len(data), 1/len(data))})
    plt.xlim(0, args['xlim'])
    plt.ylabel('Share of documents')
    plt.xlabel('Document length (tokens)')
    plt.title(args['title'])

    upper_q, lower_q = data.quantile([0.25,0.75]).to_list()
    mean = data.mean()
    median = data.median()
    mode = data.mode().to_list()[0]
    plt.axvline(mean, color=PALETTE[1], linestyle='--')
    plt.axvline(median, color=PALETTE[4], linestyle='--')
    plt.axvline(mode, color=PALETTE[3], linestyle='--')
    plt.axvline(lower_q, color=PALETTE[5], linestyle='--')
    plt.axvline(upper_q, color=PALETTE[6], linestyle='--')
    plt.legend({'Mean': mean, 'Median': median, 'Mode': mode, 'Upper_quart': upper_q, 'Lower_quart': lower_q})

    plt.savefig(out_path)
    plt.clf()

def scatter(df, outfile):
    x = df.iloc[:,0]
    y = df.iloc[:,1]
    sns.jointplot(x=x.name, y=y.name, data=df, kind='reg', marginal_kws={'bins': 200, 'rug': True},
        scatter_kws={'alpha': 0.5}, joint_kws={ 'line_kws':{'color':PALETTE[3], 'linestyle':'--'} },
        stat_func=stats.pearsonr)
    plt.xlim(0, 20000)
    # plt.xscale('log')
    plt.savefig(outfile)
    plt.clf()

def plot_hists(data, plot_data, out_path, max_i=10000):
    for i, data in data.items():
        if i <= max_i:
            sns.distplot(data, kde=False, bins=plot_data['bins'], label=f'len = {i}', hist_kws={'alpha':0.4, 'weights': np.full(len(data), 1/len(data))})
    plt.xlim(0, plot_data['xlim'])
    plt.xlabel(plot_data['xlabel'])
    plt.ylabel('Share of documents')
    plt.title(plot_data['title'])
    plt.legend()
    plt.savefig(out_path)
    plt.clf()

def corr_table(df, outfile, to_latex=True):
    cols = ['source_len', 'target_len', 'num_tokens_to_remove', 'starting_position']
    rows = ['bleurt', 'mover-1', 'mover-2', 'bertscore', 'bartscore', 'rouge1', 'rouge2', 'rougeLsum']

    # Compute correlations for all rows and columns
    all_corrs = []
    for r in rows:
        corrs = []
        for c in cols:
            corr = df[ [r, c] ].corr(method='pearson')
            corrs.append(corr[c].iloc[0])
        all_corrs.append(corrs)

    corr_df = pd.DataFrame(all_corrs, columns=cols).set_index([pd.Index(rows)])
    
    if to_latex:
        for i in range(len(corr_df)):
            index = corr_df.index[i]
            values = corr_df.iloc[i].tolist()
            output = index + '} & ' + ' & '.join([f'{v:.3f}' for v in values])
            print(output)

def corrplot(df, outfile):
    df = df.dropna().reset_index(drop=True)
    corr = df.corr(method='pearson')
    mask = np.triu(np.ones_like(corr, dtype=np.bool))
    f, ax = plt.subplots(figsize=(15, 13))
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    sns.heatmap(corr, mask=mask, cmap=cmap, #vmax=.3, center=0,
                square=True, linewidths=.9, cbar_kws={"shrink": .5},
                annot=True)
    plt.xticks(rotation=60, size=15)
    plt.yticks(rotation=60, size=15)
    plt.title('Correlation plot using Pearson r...', size=20)
    plt.savefig(outfile)
    plt.clf()

def parse_seqlen(paths):
    if isinstance(paths, dict):
        return list(paths.values())[0].split('_')[1]
    elif isinstance(paths, str):
        return paths.split('_')[1]

def to_latex(means, stds):
    for i in range(len(means)):
        index = means.index[i]
        mns = means.iloc[i].tolist()
        sdevs = stds.iloc[i].tolist()
        output1 = "\multicolumn{1}{l||}{\multirow{2}{*}{" + index.replace('led_', '') + '}} & ' + ' & '.join([f'{v:.3f}' for v in mns]) + r' \\'
        output2 = "\\rowfont{\small} \multicolumn{1}{l||}{} & " + ' & '.join([f'{v:.2f}' for v in sdevs]) + r' \\'
        print(output1)
        print(output2)

def clean_and_filter(df, args):
    df = df.loc[df['hyps_path'].isin(args['paths'].keys()), ['hyps_path'] + args['metrics']]
    for k,v in args['paths'].items():
        if isinstance(v, list):
            df = df.replace(k, v[0])
        else:
            df = df.replace(k, v)
    return df

def batch_scatter(data, args):
    assert all(map(lambda x: x in args, ['outfile', 'rows', 'cols', 'metrics', 'feature', 'title']))

    f, axes = plt.subplots(args['rows'], args['cols'], figsize=(16, 7), sharex=True, sharey=False)
    for r in range(args['rows']):
        for c in range(args['cols']):
            # Make dist plot
            cols = [args['feature'], args['metrics'][r * args['cols'] + c]]
            df = data[cols]
            ax = axes[r,c]
            x = df.iloc[:,0]
            y = df.iloc[:,1]
            j = sns.jointplot(x=x.name, y=y.name, data=df, kind='reg', marginal_kws={'bins': 200, 'rug': True},
                scatter_kws={'alpha': 0.2}, joint_kws={ 'line_kws':{'color':PALETTE[3], 'linestyle':'--'} },
                ax=ax)
            ax.set_xlim(0, args['xlim'])
            ax.set_title(y.name)
            annot = (lambda m: f' Pearson r: {m[0]:.3f} \n p = {m[1]:.2E}')(stats.pearsonr(x, y))
            ax.text(0.5, 0.85, annot, transform=ax.transAxes) 
    
    # Annotate
    f.text(0.08, 0.5, 'Metric Score', va='center', rotation='vertical')
    f.text(0.5, 0.04, args['xlabel'], ha='center')
    f.suptitle(args['title'])

    f.savefig(args['outfile'])
    f.clf()

def batch_hists(data, args):
    assert all(map(lambda x: x in args, ['outfile', 'row', 'col', 'title']))

    rows = ['source', 'target']
    cols = ['pubmed', 'arxiv', 'cnn_dm']
    f, axes = plt.subplots(len(rows), len(cols), figsize=(16, 7), sharex=False, sharey=False)
    for n, r in enumerate(rows):
        for m, c in enumerate(cols):
            # Make dist plot
            d = data.loc[(data['Doc Type'] == r) & (data['Dataset'] == c), 'Length'].copy()
            d = d.loc[d < d.quantile(0.995)]
            ax = axes[n,m]
            sns.distplot(d, kde=False, bins=50, ax=ax,
                hist_kws={'weights': np.full(len(d), 1/len(d))})
            ax.set_xlim(0, d.max())
            ax.set_title(c + '-' + r)
            ax.set_xlabel('')
            # ax.ticklabel_format(axis='y', style='sci', scilimits=(-4,-3))

            # Annotate with statistics
            lower_q, upper_q = d.quantile([0.25,0.75]).to_list()
            mean = d.mean()
            median = d.median()
            ax.axvline(upper_q, color=PALETTE[6], linestyle='--')
            ax.axvline(lower_q, color=PALETTE[5], linestyle='--')
            ax.axvline(median, color=PALETTE[4], linestyle='--')
            ax.axvline(mean, color=PALETTE[1], linestyle='--')


    # Annotate
    f.text(0.08, 0.5, 'Share of documents', va='center', rotation='vertical', size=15)
    f.text(0.5, 0.04, 'Document Length', ha='center', size=15)
    f.legend({'Upper_quart': upper_q, 'Lower_quart': lower_q, 'Median': median, 'Mean': mean,},
        loc=(0.07,0.87), framealpha=0.4)     
    f.suptitle(args['title'], size=18)

    plt.savefig(args['outfile'])
    plt.clf()


if __name__ == '__main__':
    # Analyzer()
    # Plotter()
    # RandomStartAnalyzer()
    RandomStartPlotter()
    # RandomStartLoader()
    # MetricsPlotter()
    # AdversarialAnalyzer()
    # SummaryMetricAnalyzer()

''' env: led '''
