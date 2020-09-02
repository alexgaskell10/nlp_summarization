from typing import Dict, List, Optional, Tuple

from torch import Tensor, nn

from transformers.modeling_longformer import LongformerSelfAttention

from .modeling_bart import BartForConditionalGeneration

# BART calculates the output projection in the attention layer itself, also the `forward` paramter names of `SelfAttention` layer used in BART are different than that of `BertSelfAttention`. So we'll need to wrap `LongformerSelfAttention` to use it for BART

class LongBartForConditionalGeneration(BartForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        for i, layer in enumerate(self.model.encoder.layers):
            # replace the `modeling_bart.SelfAttention` object with `LongformerSelfAttention`
            layer.self_attn = LongformerSelfAttentionForBart(config, layer_id=i)


class LongformerSelfAttentionForBart(nn.Module):
    def __init__(self, config, layer_id):
        super().__init__()
        self.embed_dim = config.d_model
        self.longformer_self_attn = LongformerSelfAttention(config, layer_id=layer_id)
        self.output = nn.Linear(self.embed_dim, self.embed_dim)
    
    def forward(
        self,
        query,
        key: Optional[Tensor],
        key_padding_mask: Optional[Tensor] = None,
        layer_state: Optional[Dict[str, Optional[Tensor]]] = None,
        attn_mask: Optional[Tensor] = None,
        need_weights=False,
        output_attentions=False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        
        tgt_len, bsz, embed_dim = query.size()
        assert embed_dim == self.embed_dim
        assert list(query.size()) == [tgt_len, bsz, embed_dim]

        # LongformerSelfAttention expects this shape
        query = query.transpose(0,1)

        outputs = self.longformer_self_attn(
            query, 
            attention_mask=key_padding_mask.unsqueeze(dim=1).unsqueeze(dim=1) * -1,
            output_attentions=output_attentions,
        )

        attn_output = outputs[0]
        attn_output = attn_output.transpose(0,1)
        attn_output = self.output(attn_output)

        return (attn_output,) + outputs[1:] if len(outputs) == 2 else (attn_output, None)
