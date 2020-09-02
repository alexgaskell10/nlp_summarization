from typing import List, Optional, Tuple, Dict
from torch import nn, Tensor

from .modeling_bart import BartConfig, BartForConditionalGeneration

from .configuration_reformer import (
    ReformerConfig,
    CONFIG_MATCH_BART_TINY_RANDOM,
    CONFIG_MATCH_BART_LARGE_CNN,
)

from .modeling_reformer import (
    LSHSelfAttention, 
    LocalSelfAttention,
    ReformerAttention,
)


class ReformerEncoderDecoderForConditionalGeneration(BartForConditionalGeneration):
    pass


class ReformerEncoderDecoderConfig(BartConfig):
    def __init__(self, base_model):
        config = CONFIG_MATCH_BART_TINY_RANDOM if 'tiny' in base_model else CONFIG_MATCH_BART_LARGE_CNN
        self.reformer_config = ReformerConfig(**config)


class ReformerAttentionForBart(nn.Module):
    def __init__(self, config, attn_type):
        super().__init__()
        attn_types = ['local', 'lsh']
        assert attn_type in attn_types, f'Attention type should be one of {attn_types}'
        config.attn_layers = [attn_type]
        self.self_attention = ReformerAttention(config, layer_id=0)

    def forward(
        self,
        query,
        key: Optional[Tensor],
        key_padding_mask: Optional[Tensor] = None,
        layer_state: Optional[Dict[str, Optional[Tensor]]] = None,
        attn_mask: Optional[Tensor] = None,
        need_weights=False,
    ) -> Tuple[Tensor, Optional[Tensor]]:

        tgt_len, bsz, embed_dim = query.size()
        query = query.transpose(0,1)
        self_attention_outputs = self.self_attention(
            hidden_states=query,
            head_mask=None,     # Doesn't seem to exist for BART
            attention_mask=attn_mask,
            num_hashes=None,    # Set by config
            do_output_attentions=need_weights,
            buckets=None,       # Set by config
        )
        hidden_states_output = self_attention_outputs.hidden_states.transpose(0,1)
        
        return hidden_states_output, self_attention_outputs.attention_probs
