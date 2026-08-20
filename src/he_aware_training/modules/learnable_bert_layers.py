import torch
import torch.nn as nn

from he_aware_training.modules.learnable_thor_layers import THORLearnableSoftmax


class THORLearnableBertAttention(nn.Module):
    def __init__(self, approx_config, config, layer_idx=None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = config.hidden_size // config.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        self.dropout = nn.Dropout(getattr(config, "attention_probs_dropout_prob", 0.0))
        self.scaling = self.attention_head_size ** -0.5

        # bidirectional: THORLearnableSoftmax handles causal_mask=None (k_eff = T)
        self.softmax = THORLearnableSoftmax(approx_config, config)

    def from_layer(self, bert_self_attention):
        """Copy q/k/v from a transformers `BertSelfAttention` (separate Linears)."""
        with torch.no_grad():
            for mine, theirs in ((self.query, bert_self_attention.query),
                                 (self.key, bert_self_attention.key),
                                 (self.value, bert_self_attention.value)):
                mine.weight.copy_(theirs.weight)
                mine.bias.copy_(theirs.bias)
        return self

    def _shape(self, x, B, T):
        return x.view(B, T, self.num_attention_heads,
                      self.attention_head_size).transpose(1, 2)

    def forward(self, hidden_states, attention_mask=None, head_mask=None,
                encoder_hidden_states=None, encoder_attention_mask=None,
                past_key_value=None, output_attentions=False, **kw):
        B, T, _ = hidden_states.size()
        q = self._shape(self.query(hidden_states), B, T)
        k = self._shape(self.key(hidden_states), B, T)
        v = self._shape(self.value(hidden_states), B, T)

        att = torch.matmul(q, k.transpose(-2, -1)) * self.scaling
        if attention_mask is not None:
            att = att + attention_mask.clamp(min=-30.0)
        att = self.softmax(att, causal_mask=None)     # bidirectional (k_eff = T)
        att = self.dropout(att.type(v.dtype))
        if head_mask is not None:
            att = att * head_mask

        ctx = torch.matmul(att, v).transpose(1, 2).contiguous().view(B, T, self.all_head_size)
        return ctx, (att if output_attentions else None)
