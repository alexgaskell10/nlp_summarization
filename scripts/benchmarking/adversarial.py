## . /vol/bitbucket/aeg19/.envs/benchmark/bin/activate

import os
from tqdm import tqdm
from math import ceil

import torch
from transformers import pipeline

# SUMMS_PATH = '/vol/bitbucket/aeg19/datasets/adversarial/cnn_dm/test.hypo'
# OUT_DIR = '/vol/bitbucket/aeg19/datasets/adversarial/cnn_dm/'
SUMMS_PATH = '/vol/bitbucket/aeg19/datasets/adversarial/pubmed/test.hypo'
OUT_DIR = '/vol/bitbucket/aeg19/datasets/adversarial/pubmed/'

class SummaryCorruptor:
    ''' Abstract class defining generic methods '''
    def __init__(self):
        self.chunk_len = 10

        self.run()

    def run(self):
        self.load_lines()
        self.preprocess()
        self.corrupt()
        self.postprocess()
        self.save_lines()

    def load_lines(self):
        self.txt = [line.strip() for line in open(SUMMS_PATH, encoding='utf-8')]

    def save_lines(self):
        open(self.outfile, 'w').writelines([l.strip() for l in self.outputs])

    def preprocess(self):
        self.masked, self.txt_lens = [], []
        for article in self.txt:
            split_article = article.split(' ')
            chunked_article = [split_article[i:i+self.chunk_len] for i in range(0, len(split_article), self.chunk_len)]

            processed_article = self.preprocessing_step(chunked_article)

            self.masked.extend([a for a in processed_article if a])
            self.txt_lens.append(ceil(len(split_article) / self.chunk_len))
        
        self.inputs = [' '.join(m) for m in self.masked]

    def postprocess(self):
        self.outputs = []
        index = 0
        for txt_len in self.txt_lens:
            self.outputs.append(' '.join(self.corrupted_outputs[index:index+txt_len]) + '\n')
            index += txt_len


class BERTMaskFiller(SummaryCorruptor):
    ''' Masks tokens and uses pre-trained BERT to in-fill '''
    def __init__(self):
        self.outfile = os.path.join(OUT_DIR, 'masked.txt')
        self.nlp = pipeline("fill-mask", device=0, model='distilroberta-base')
        self.MASK = self.nlp.tokenizer.mask_token
        self.bert_chunksize = 128
        super().__init__()

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
    def __init__(self):
        self.MASK = '<MASK>'
        self.outfile = os.path.join(OUT_DIR, 'dropped.txt')
        super().__init__()

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
    ''' Switch the ordering of two adjacent tokens for each chunk '''
    def __init__(self):
        self.outfile = os.path.join(OUT_DIR, 'permuted.txt')
        self.MASK = '<MASK>'
        super().__init__()

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


if __name__ == '__main__':
    BERTMaskFiller()
    WordDropper()
    WordPermuter()