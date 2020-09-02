# import seaborn as sns
# import matplotlib.pyplot as plt
import pickle
from glob import glob
from os import path
# sns.set(color_codes=True)

class DatasetExplorer:
    def __init__(self, interface, min_len):
        # Set paths
        if interface == 'local':
            self.data_dir = '/Users/alexgaskell/GoogleDrive/Documents/Imperial' + \
                '/Individual_project/datasets/'
        else:
            self.data_dir = '/content/drive/My Drive/Documents/Imperial' + \
                '/Individual_project/datasets/'

        self.dir = self.data_dir + 'cnn_dm/'

        self.exts = glob(self.dir + '*.target') + glob(self.dir + '*.source')
        print(self.exts)
        # self.exts = ['sources.txt']

        self.out_dir = self.data_dir + 'analysis/cnn_dm/'

        # Minimum length of document
        self.min_len = min_len

    def get_token_counts(self):
        ''' Obtain token counts from each dataset '''
        self.token_counts = {}
        # Load data and count tokens for each specified file extension
        for ext in self.exts:
            print(path.basename(ext))
            with open(ext, 'r') as f:
                d_set_data = f.read().split('\n')

            # Obtain token counts by sample
            d_counts = []
            for d in d_set_data:
                d_counts.append(len(d.split()))

            self.token_counts[path.basename(ext)] = d_counts

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
        token_counts = [item for k, sublist in self.token_counts.items() 
                            for item in sublist if k.find('.source') > 0]

        print('Dataset: cnn_dm')
        print('Docs: ', len(token_counts))
        print('Avg len: ', sum(token_counts)/len(token_counts))

    def count_valid_docs(self, length):
        ''' Returns the share of documents which are shorter than "length" tokens 
        '''
        token_counts = [item for sublist in self.token_counts.values() 
                        for item in sublist]# if item > self.min_len]

        valid_token_counts = [i for i in token_counts if i <= length]

        return len(valid_token_counts), len(token_counts)


if __name__ == '__main__':
    from argparse import ArgumentParser
    ap = ArgumentParser()
    ap.add_argument('interface', choices={'local', 'colab'},
        help='where is this script being run from? [local, colab]')

    args = ap.parse_args()

    # Run script
    da = DatasetExplorer(args.interface, 0)
    # da.get_token_counts()
    # da.save_counts()
    da.load_counts()
    # da.dset_stats()
    # short_docs, docs = da.count_valid_docs(1024)
    short_docs, docs = da.count_valid_docs(4096)
    print(short_docs, docs)
    print(1-short_docs/docs)

