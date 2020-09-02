import os

LANGUAGE = 'english'

ROOT_DIR = os.path.dirname(os.path.abspath(__file__)) # os.path.abspath('.')
BASE_DIR = os.path.join(ROOT_DIR, 'learned_eval')
DATA_DIR = os.path.join(BASE_DIR, 'data', 'cdm')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw_json')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed_json')
FULL_CDM_RAW_DIR = os.path.join(DATA_DIR, 'cdm_full', 'raw')
FULL_CDM_SAMPLES_DIR = os.path.join(DATA_DIR, 'cdm_full', 'samples')
SWAP_FEATURES_DIR = os.path.join(DATA_DIR, 'swap_features')
TEST_FEATURES_DIR = os.path.join(DATA_DIR, 'test_features')
ROUGE_DIR = os.path.join(BASE_DIR,'scorer','auto_metrics','rouge','ROUGE-RELEASE-1.5.5/')

METEOR_DIR = os.path.join(BASE_DIR, 'scorer', 'auto_metrics')
SENT2VEC_DIR = os.path.join('/home/gao/Library/NLP-Related/sent2vec')

EMBEDDING_PATH = os.path.join(BASE_DIR,'.vector_cache')
INFERSENT_PATH = os.path.join(EMBEDDING_PATH, 'infersent1.pkl')
W2V_PATH = os.path.join(EMBEDDING_PATH, 'glove.840B.300d.txt')
VEC_DIM = 2048
ABS_MODEL_DIR = os.path.join(BASE_DIR, 'fast_abs_rl', 'pretrained', 'abstractor')
FAST_ABS_RL = os.path.join(BASE_DIR, 'fast_abs_rl')
RUNS_DIR = os.path.join(BASE_DIR, 'runs')
MODEL_WEIGHT_DIR = os.path.join(BASE_DIR,'trained_models')

import sys
BLEURT_DIR = os.path.join(ROOT_DIR, '../../bleurt')     # TODO: FIX BLEURT INSTALL
sys.path.append(BLEURT_DIR)

QQP_DATA_PATH = '/vol/bitbucket/aeg19/datasets/quora_question_pairs/train.csv'
QQP_OUT_PATH = '/vol/bitbucket/aeg19/datasets/quora_question_pairs/out.csv'

# # Output data
# OUT_DATA_DIR


