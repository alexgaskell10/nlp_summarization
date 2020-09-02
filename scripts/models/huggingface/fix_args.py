import argparse

from hf.utils import pickle_save, pickle_load

args_file = '/vol/bitbucket/aeg19/datasets/bart-pubmed/outputwandb_114/hparams.pkl'
args = argparse.Namespace(**pickle_load(args_file))
print('')
args.model_name_or_path = 'longbart'
# args.