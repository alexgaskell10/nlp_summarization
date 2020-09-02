import json
# import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import os
# sns.set(color_codes=True)

class DatasetExplorer:
    def __init__(self, dset, interface, min_len):
        # Set paths
        # if interface == 'local':
        #     self.data_dir = '/Users/alexgaskell/GoogleDrive/Documents/Imperial' + \
        #         '/Individual_project/datasets/'
        # else:
        #     self.data_dir = '/content/drive/My Drive/Documents/Imperial' + \
        #         '/Individual_project/datasets/'

        # self.dir = self.data_dir + dset + '-dataset/'
        self.dir = '/rds/general/user/aeg19/home/datasets/custom-bart-test/'

        # self.exts = ['train.txt', 'test.txt', 'val.txt']
        self.exts = ['test.source']
        # self.out_dir = self.data_dir + 'analysis/' + dset + '/'
        self.out_dir = '/rds/general/user/aeg19/home/datasets/bart-pubmed/analysis/'
        self.dset = dset

        # Minimum length of document
        self.min_len = 0

    def get_token_counts(self):
        ''' Obtain token counts from each dataset '''
        self.token_counts = {}
        for ext in self.exts:
            # Load json data into memory
            with open(self.dir + ext, 'r') as f:
                d_set_data = [json.loads(line) for line in f]

            # Obtain token counts by sample
            d_counts = []
            for d in d_set_data:
                sentence_counts = [len(sen.split()) for sen in d['article_text']]
                d_counts.append(sum(sentence_counts))

            self.token_counts[d_set] = d_counts

    def save_counts(self):
        ''' Save token counts using pickle '''
        assert self.token_counts, \
            'Token counts are empty; run self.get_token_counts() first'
        with open(self.out_dir + 'token_counts.pkl', 'wb') as f:
            pickle.dump(self.token_counts, f)

    def load_counts(self):
        ''' Save token counts using pickle '''
        with open(self.out_dir + 'token_counts.pkl', 'rb') as f:
            self.token_counts = pickle.load(f)

    def dset_stats(self):
        ''' Returns the share of documents which are shorter than "length" tokens 
        '''
        token_counts = [item for sublist in self.token_counts.values() 
                        for item in sublist if item > self.min_len]

        print('Dataset: ', self.dset)
        print('Docs: ', len(token_counts))
        print('Avg len: ', sum(token_counts)/len(token_counts))

    def count_valid_docs(self, length):
        ''' Returns the share of documents which are shorter than "length" tokens 
        '''
        token_counts = [item for sublist in self.token_counts.values() 
                        for item in sublist]# if item > self.min_len]

        valid_token_counts = [i for i in token_counts if i <= length]

        return len(valid_token_counts), len(token_counts)

    def plot_dist(self):
        ''' Make pdf / cdf of token counts '''
        # Flatten list
        token_counts = [item for sublist in self.token_counts.values() 
                            for item in sublist if item > self.min_len]
        
        # Plot cumulative plot
        sns.distplot(token_counts, kde=False, bins=200)
        plt.xlim(0, 25000)
        plt.xlabel('Document length (tokens)')
        plt.xlabel('Document no. of tokens')
        plt.title(f'Histogram of document lengths for {self.dset} dataset')
        plt.savefig(self.out_dir + 'dist_plot.png')
        # plt.show()
        plt.clf()
        
        # Plot cumulative plot
        kwargs = {'cumulative': True}
        sns.distplot(token_counts, hist_kws=kwargs, kde_kws=kwargs, bins=200)
        plt.xlim(0, 25000)
        plt.xlabel('Document no. of tokens')
        plt.ylabel('Share of total documents')
        plt.title(f'Histogram of document lengths for {self.dset} dataset')        
        plt.savefig(self.out_dir + 'dist_plot_cuml.png')
        # plt.show()

    def fetch_samples(self):
        ''' Return samples from a specified .txt data file '''
        with open(self.out_dir + 'dummy.txt', 'r') as f:
            return [json.loads(line) for line in f]


class DatasetSplitter:
    def __init__(self):
        # self.dir = '/vol/bitbucket/aeg19/datasets/bart-pubmed-new/'
        self.dir = '/vol/bitbucket/aeg19/datasets/bart-arxiv-new/'
        self.exts = ['test.txt']

        # self.clean_pubmed()
        self.clean_cnn()

    def get_token_counts(self):
        ''' Obtain token counts from each dataset '''
        self.token_counts = {}
        for ext in self.exts:
            # Load json data into memory
            with open(self.dir + ext, 'r') as f:
                d_set_data = [json.loads(line) for line in f]

            # Obtain token counts by sample
            d_counts = []
            for d in d_set_data:
                sentence_counts = [len(sen.split()) for sen in d['article_text']]
                d_counts.append(sum(sentence_counts))
                break

            # self.token_counts[d_set] = d_counts

    def write_data(self):
        exts = [
            ('unprocessed/test.txt', 'test.source', 'test.target'),
            ('unprocessed/train.txt', 'train.source', 'train.target'),
            ('unprocessed/val.txt', 'val.source', 'val.target'),
        ]
        for ext in exts:
            orig, src, tgt = ext

            with open(os.path.join(self.dir, orig), 'r') as f:
                d_set_data = [json.loads(line) for line in f]

            src_data = [' '.join(d['article_text']) for d in d_set_data]
            tgt_data = [' '.join(d['abstract_text']) for d in d_set_data]

            print(len(src_data), len(tgt_data))

            with open(os.path.join(self.dir, src), 'w') as f, open(os.path.join(self.dir, tgt), 'w') as g:
                omitted = 0
                for line in range(len(src_data)):
                    src_line = src_data[line].strip().replace('\n','')
                    tgt_line = tgt_data[line].strip().replace('\n','')

                    if len(src_line.split()) < 200 or len(tgt_line) < 4:
                        omitted += 1
                        continue

                    f.write(src_line)
                    f.write('\n')

                    g.write(tgt_line)
                    g.write('\n')

            assert len(src_data) - omitted == len(open(os.path.join(self.dir, src)).readlines())
            assert len(src_data) - omitted == len(open(os.path.join(self.dir, tgt)).readlines())

    def clean_pubmed(self):
        ''' Remove the </S> and </S> tokens from target files '''
        dir = '/vol/bitbucket/aeg19/datasets/pubmed-newdata-test'
        exts = [
            ('orig/test.target', 'test.target'),
            ('orig/train.target', 'train.target'),
            ('orig/val.target', 'val.target'),
        ]
        for ext in exts:
            with open(os.path.join(dir, ext[0]), 'r') as infile, open(os.path.join(dir, ext[1]), 'w') as outfile:
                data = infile.readlines()

                clean = [d.replace('</S>', '').replace('<S>', '').strip() for d in data]
                
                for line in clean:
                    outfile.write(line)
                    outfile.write('\n')

    def clean_cnn(self):
        ''' Remove the (CNN) token at the beginning of source files '''
        dir = '/vol/bitbucket/aeg19/datasets/cnn_dm/cleaned'
        exts = [
            ('orig/test.source', 'test.source'),
            ('orig/train.source', 'train.source'),
            ('orig/val.source', 'val.source'),
        ]
        for ext in exts:
            with open(os.path.join(dir, ext[0]), 'r') as infile, open(os.path.join(dir, ext[1]), 'w') as outfile:
                data = infile.readlines()

                clean = [d.replace('(CNN)', '').strip() for d in data]
                
                for line in clean:
                    outfile.write(line)
                    outfile.write('\n')


def token_counts():
    dir = '/vol/bitbucket/aeg19/datasets/bart-pubmed'
    exts = [
        ('unprocessed/test.txt', 'test.source', 'test.target'),
        # ('unprocessed/train.txt', 'train.source', 'train.target'),
        # ('unprocessed/val.txt', 'val.source', 'val.target'),
    ]
    for ext in exts:
        orig, src, tgt = ext

        src_data = open(os.path.join(dir, src)).readlines()
        tgt_data = open(os.path.join(dir, tgt)).readlines()
        
        src_lens = [len(l.split(' ')) for l in src_data]
        tgt_lens = [len(l.split(' ')) for l in tgt_data]

        print(min(src_lens))
        print(min(tgt_lens))


if __name__ == '__main__':
    ds = DatasetSplitter()
    # ds.get_token_counts()
    # ds.write_data()
    # token_counts()

    # from argparse import ArgumentParser
    # ap = ArgumentParser()
    # ap.add_argument('dataset', help='Dataset to explore [arxiv, pubmed]', 
    #     choices={'pubmed', 'arxiv'})
    # # ap.add_argument('data_root_dir',
    # #     help='path to the root dir containing arxiv and pubmed data dirs')
    # ap.add_argument('interface', choices={'local', 'colab'},
    #     help='where is this script being run from? [local, colab]')

    # args = ap.parse_args()

    # # Parse args
    # # data_dir = args.data_root_dir
    # interface = args.interface
    # dset = args.dataset 

    # # Run script
    # da = DatasetExplorer(dset, interface, 250)
    # # da.show_sample()
    # # da.get_token_counts()
    # # da.save_counts()
    # da.load_counts()
    # # da.plot_dist()
    # da.dset_stats()
    # short_docs, docs = da.count_valid_docs(1024)
    # # short_docs, docs = da.count_valid_docs(4096)
    # print(short_docs, docs)
    # print(1-short_docs/docs)
    
