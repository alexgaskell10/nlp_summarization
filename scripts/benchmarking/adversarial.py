import os
from tqdm import tqdm
from math import ceil

import torch
from transformers import pipeline

class SummaryCorruptor:
    ''' Abstract corruptor class. This class defines generic methods and the
        child classes define the specific types of corruptions.
    '''
    def __init__(self, summs_path, out_dir, chunk_len):
        self.chunk_len = chunk_len
        self.SUMMS_PATH = summs_path
        self.OUT_DIR = out_dir

        self.run()

    def run(self):
        ''' Main method '''
        self.load_lines()
        self.preprocess()
        self.corrupt()
        self.postprocess()
        self.save_lines()

    def load_lines(self):
        self.txt = [line.strip() for line in open(self.SUMMS_PATH, encoding='utf-8')]

    def save_lines(self):
        with open(self.outfile, 'w') as f:
            for line in [l.strip() for l in self.outputs]:
                f.write(line + '\n')

    def preprocess(self):
        ''' Tokenizes based on spaces and splits the sequence into chunks of
            length self.chunk_len. This makes it easier to perform the 
            subsequent corruption and ensures it is distributed throughout
            the summary
        '''
        self.masked, self.txt_lens = [], []
        for article in self.txt:
            split_article = article.split(' ')
            chunked_article = [split_article[i:i+self.chunk_len] for i in range(0, len(split_article), self.chunk_len)]

            processed_article = self.preprocessing_step(chunked_article)

            self.masked.extend([a for a in processed_article if a])
            self.txt_lens.append(ceil(len(split_article) / self.chunk_len))
        
        self.inputs = [' '.join(m) for m in self.masked]

    def postprocess(self):
        ''' Joins sequence into string so it can be written to file '''
        self.outputs = []
        index = 0
        for txt_len in self.txt_lens:
            self.outputs.append(' '.join(self.corrupted_outputs[index:index+txt_len]) + '\n')
            index += txt_len


class BERTMaskFiller(SummaryCorruptor):
    ''' Class performing BERT Mask-filling.
        This masks random tokens in the summary and uses a 
        pre-trained BERT to in-fill these.
    '''
    def __init__(self, summs_path, out_dir, chunk_len=10):
        self.outfile = os.path.join(out_dir, 'masked.txt')
        self.nlp = pipeline("fill-mask", device=0, model='distilroberta-base')
        self.MASK = self.nlp.tokenizer.mask_token
        self.bert_chunksize = 128
        super().__init__(summs_path, out_dir, chunk_len)

    def preprocessing_step(self, chunks):
        ''' Mask random word in chunk '''
        for chunk in chunks:
            mask_index = torch.randint(len(chunk), (1,)).item()
            chunk[mask_index] = self.MASK
        return chunks

    def corrupt(self):
        ''' Feed through pipline to predict masked words '''
        self.corrupted_outputs = []
        for chunk in tqdm(range(0, len(self.inputs), self.bert_chunksize)):
            chunk_outputs = self.nlp(self.inputs[chunk:chunk+self.bert_chunksize])
            clean_outputs = [s[0]['sequence'].replace('<s>', '').replace('</s>', '').replace('\n', ' ').strip() for s in chunk_outputs]
            self.corrupted_outputs.extend(clean_outputs)


class WordDropper(SummaryCorruptor):
    ''' Class performing random word-dropping. This drops a token from
        each chunk
    '''
    def __init__(self, summs_path, out_dir, chunk_len=10):
        self.MASK = '<MASK>'
        self.outfile = os.path.join(out_dir, 'dropped.txt')
        super().__init__(summs_path, out_dir, chunk_len)

    def preprocessing_step(self, chunks):
        ''' Mask random word in chunk '''
        for chunk in chunks:
            mask_index = torch.randint(len(chunk), (1,)).item()
            chunk[mask_index] = self.MASK
        return chunks

    def corrupt(self):
        ''' Drop masked words '''
        self.corrupted_outputs = [s.replace(self.MASK, '').replace('  ', ' ') for s in self.inputs]


class WordPermuter(SummaryCorruptor):
    ''' Class performing random word permutation.
        This switched the ordering of two adjacent tokens for each chunk
    '''
    def __init__(self, summs_path, out_dir, chunk_len=10):
        self.MASK = '<MASK>'
        self.outfile = os.path.join(out_dir, 'permuted.txt')
        super().__init__(summs_path, out_dir, chunk_len)

    def preprocessing_step(self, chunks):
        ''' Insert mask token before tokens to be permuted '''
        for chunk in chunks:
            if len(chunk) >= 2:
                mask_index = torch.randint(len(chunk)-1, (1,)).item()
                chunk.insert(mask_index, self.MASK)
        return chunks

    def corrupt(self):
        ''' Permute the two following the mask token '''
        self.corrupted_outputs = []
        for line in self.inputs:
            tokens = line.split(' ')
            if self.MASK in tokens:
                mask_index = tokens.index(self.MASK)
                tokens.pop(mask_index)
                tokens.insert(mask_index, tokens.pop(mask_index+1))
            self.corrupted_outputs.append(' '.join(tokens))


class AdversarialAnalyzer:
    ''' Class to determine the accuracy of each metric on the corruption tasks '''
    def __init__(self):
        self.out_dir_orig = OUT_DIR
        self.out_dir = OUT_DIR
        self.raw_scores_path = os.path.join(self.out_dir_orig, 'eval_output_raw.txt')
        self.metrics = ['bleurt', 'mover-1', 'mover-2', 'bertscore', 'bartscore', 
                            'rouge1', 'rouge2', 'rougeL', 'rougeLsum']
        self.adversarial_test()

    def load_raw_scores_from_file(self):
        ''' Loads the saved scores into a pd dataframe '''
        self.raw_scores = pd.DataFrame(columns=['hyps_path'] + self.metrics)
        scores = [eval(line) for line in open(self.raw_scores_path, 'r')]
        for score in scores:
            if score['hyps_path'] not in self.raw_scores['hyps_path'].tolist():
                score['hyps_path'] = [score['hyps_path']] * len(score[self.metrics[0]])
                temp_df = pd.DataFrame(score)
                self.raw_scores = pd.concat([self.raw_scores, temp_df], ignore_index=True)

    def adversarial_test(self):
        ''' Computes the score per metric on the corruption tasks.
            These are saved to file and printed as output table
        '''
        # args = {
        #     'pubmed': {
        #         'paths': {
        #             'Dropped': # Add path to corrupted summaries here,
        #             'Masked': # Add path to corrupted summaries here,
        #             'Permuted': # Add path to corrupted summaries here,
        #         },
        #         'orig': # Add path to original summaries here,
        #     },
        #     'cnn_dm': {
        #         'paths': {
        #             'Dropped': # Add path to corrupted summaries here, 
        #             'Masked': # Add path to corrupted summaries here,
        #             'Permuted': # Add path to corrupted summaries here,
        #         },
        #         'orig': # Add path to original summaries here,
        #     },
        # }
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


def main():
    paths = {
        'pubmed': {
            'SUMMS_PATH': '/vol/bitbucket/aeg19/datasets/adversarial/pubmed/variable_corruption/test.hypo',
            'OUT_DIR': '/vol/bitbucket/aeg19/datasets/adversarial/pubmed/variable_corruption/',
        },
        'cnn_dm': {
            'SUMMS_PATH': '/vol/bitbucket/aeg19/datasets/adversarial/cnn_dm/variable_corruption/test.hypo',
            'OUT_DIR': '/vol/bitbucket/aeg19/datasets/adversarial/cnn_dm/variable_corruption/',
        },
        'exts': ['_8', '_6', '_4']
    }

    for dset in ['pubmed', 'cnn_dm']:
        SUMMS_PATH = paths[dset]['SUMMS_PATH']
        for ext in paths['exts']:
            OUT_DIR = os.path.join(paths[dset]['OUT_DIR'], ext)
            chunk_len = int(ext.replace('_', ''))
            
            # Run corrpution
            BERTMaskFiller(SUMMS_PATH, OUT_DIR, chunk_len)
            WordDropper(SUMMS_PATH, OUT_DIR, chunk_len)
            WordPermuter(SUMMS_PATH, OUT_DIR, chunk_len)


if __name__ == '__main__':
    main()
