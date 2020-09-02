import os
import sys
import torch
from torch import Tensor, nn
from typing import Dict, List, Optional, Tuple

from transformers import BartTokenizer, ReformerTokenizer

from reformer.modeling_bart import BartForConditionalGeneration, EncoderLayer, SelfAttention
from reformer.modeling_reformer import ReformerModel
from reformer.reformer_encoder_decoder import (
    ReformerAttentionForBart, 
    ReformerEncoderDecoderConfig,
    ReformerEncoderDecoderForConditionalGeneration,
)

def create_long_model(
    save_model_to,
    base_model='facebook/bart-large-cnn',
    tokenizer_name_or_path='facebook/bart-large-cnn',
    attn_type='mix',
    max_pos=1024,
):
    model = BartForConditionalGeneration.from_pretrained(base_model)
    tokenizer = BartTokenizer.from_pretrained(tokenizer_name_or_path, model_max_length=max_pos)
    config = ReformerEncoderDecoderConfig(base_model).reformer_config

    ref_model = ReformerModel.from_pretrained('google/reformer-crime-and-punishment')
    ref_tokenizer = ReformerTokenizer.from_pretrained('google/reformer-crime-and-punishment')

    # Default attn pattern is to alternate between lsh and loc layers
    attn_type = attn_type.lower()
    assert attn_type in ['mix', 'local', 'lsh']
    if attn_type == 'mix':
        attn_layers = ['lsh', 'local'] * (len(model.model.encoder.layers) // 2)
    else:
        attn_layers = [attn_type] * len(model.model.encoder.layers)

    # Replace encoder self-attn with reformer self-attn
    for attn_layer, layer in zip(attn_layers, model.model.encoder.layers):
        layer.self_attn = ReformerAttentionForBart(config, attn_layer)

    return model, tokenizer

if __name__ == '__main__':
    model, tokenizer = create_long_model(None, base_model='facebook/bart-large-cnn')

    ARTICLE_TO_SUMMARIZE = "My friends are cool but they eat too many carbs."
    inputs = tokenizer.batch_encode_plus(
        [ARTICLE_TO_SUMMARIZE], max_length=64, return_tensors='pt', pad_to_max_length=True, truncation=True)
    DEVICE = torch.device("cuda")
    inputs = inputs['input_ids'].to(DEVICE)
    model = model.train().to(DEVICE)
    outs = model(inputs)

    print('------')
