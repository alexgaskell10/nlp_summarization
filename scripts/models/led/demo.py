import os
import sys
import logging
# import math
from dataclasses import dataclass, field
from transformers import RobertaForMaskedLM, RobertaTokenizerFast, TextDataset, DataCollatorForLanguageModeling, Trainer
from transformers import BartTokenizer
from transformers import TrainingArguments, HfArgumentParser
from transformers.modeling_longformer import LongformerSelfAttention

from longbart.modeling_bart import BartForConditionalGeneration
from longbart.modeling_longbart import LongBartForConditionalGeneration, LongformerSelfAttentionForBart
from longbart.convert_bart_to_longbart import create_long_model
import torch

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.WARNING)
# logger.disabled = True

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def short():
    # lets use a tiny version of bart for initial experiment 
    tokenizer = BartTokenizer.from_pretrained('sshleifer/bart-tiny-random')
    bart = BartForConditionalGeneration.from_pretrained('sshleifer/bart-tiny-random')

    # load ROBERta model to see the difference between bart encoder layer and roberta encoder layer 
    roberta = RobertaForMaskedLM.from_pretrained('roberta-base')

    # print(roberta.config, '\n')
    # print(bart.config, '\n')

    bart_layer = bart.model.encoder.layers[0]
    roberta_layer = roberta.roberta.encoder.layer[0]

    # print(roberta_layer, '\n')
    # print(bart_layer, '\n')

    # model_path = f'{training_args.output_dir}/roberta-base-{model_args.max_pos}'
    base_model = "sshleifer/bart-tiny-random"
    model_path = "bart-tiny-random-4096"
    attention_window = 512
    max_pos = 4096

    if not os.path.exists(model_path):
        os.makedirs(model_path)

    # logger.info(f'Converting roberta-base into roberta-base-{model_args.max_pos}')
    model, tokenizer = create_long_model(
        save_model_to=model_path,
        base_model=base_model,
        attention_window=attention_window,
        max_pos=max_pos
    )

    long_model_tiny = LongBartForConditionalGeneration.from_pretrained('bart-tiny-random-4096')

    TXT = "My friends are <mask> but they eat too many carbs."

    input_ids = tokenizer.batch_encode_plus([TXT], return_tensors='pt', max_length=4096, pad_to_max_length=True)['input_ids']

    input_ids.to(DEVICE)
    long_model_tiny.to(DEVICE)

    logits = long_model_tiny(input_ids.to(DEVICE))[0]

    masked_index = (input_ids[0] == tokenizer.mask_token_id).nonzero().item()
    probs = logits[0, masked_index].softmax(dim=0)
    values, predictions = probs.topk(5)
    preds = tokenizer.decode(predictions).split()
    print('Masked LM:')
    print(preds)

    # ARTICLES_TO_SUMMARIZE = [["My friends are cool but they eat too many carbs."], ["The quick brown fox jumped over the lazy dog"]]
    ARTICLES_TO_SUMMARIZE = [["My friends are cool but they eat too many carbs."]]
    inputs = tokenizer.batch_encode_plus(ARTICLES_TO_SUMMARIZE, max_length=1024, pad_to_max_length=True, return_tensors='pt')
    # Generate Summary
    summary_ids = long_model_tiny.generate(inputs['input_ids'].to(DEVICE), num_beams=4, max_length=5, early_stopping=True)
    print('Decoded:')
    print([tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_ids])


def long():
  # model_path = f'{training_args.output_dir}/roberta-base-{model_args.max_pos}'
  base_model = "facebook/bart-large"
  model_path = "bart-large-4096"
  attention_window = 512
  max_pos = 4096

  if not os.path.exists(model_path):
      os.makedirs(model_path)

  # logger.info(f'Converting roberta-base into roberta-base-{model_args.max_pos}')
  model, tokenizer = create_long_model(
      save_model_to=model_path,
      base_model=base_model,
      attention_window=attention_window,
      max_pos=max_pos
  )

  long_model = LongBartForConditionalGeneration.from_pretrained('bart-large-4096').to(DEVICE)
  tokenizer = BartTokenizer.from_pretrained('bart-large-4096')

  TXT = "My friends are <mask> but they eat too many carbs."

  # 4096 seq len crashes even with 35 GB memory
  # so we also probably need sliding-window attention in decoder as well
  input_ids = tokenizer.batch_encode_plus([TXT], return_tensors='pt', max_length=512, pad_to_max_length=True)['input_ids'].to(DEVICE)

  logits = long_model(input_ids)[0]

  masked_index = (input_ids[0] == tokenizer.mask_token_id).nonzero().item()
  probs = logits[0, masked_index].softmax(dim=0)
  values, predictions = probs.topk(5)
  print(tokenizer.decode(predictions).split())

  # ARTICLES_TO_SUMMARIZE = [["My friends are cool but they eat too many carbs."], ["The quick brown fox jumped over the lazy dog"]]
  ARTICLES_TO_SUMMARIZE = [["My friends are cool but they eat too many carbs."]]
  inputs = tokenizer.batch_encode_plus(ARTICLES_TO_SUMMARIZE, max_length=1024, pad_to_max_length=True, return_tensors='pt')
  # Generate Summary
  print(inputs['input_ids'].shape)
  summary_ids = long_model.generate(inputs['input_ids'].to(DEVICE), num_beams=4, max_length=5, early_stopping=True)
  print(inputs['input_ids'][:, :20])
  print('Decoded:')
  print([tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_ids])

# short()
long()