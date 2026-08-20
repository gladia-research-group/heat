import torch
import torch.nn as nn

from he_aware_training.modules.learnable_thor_layers import THORLearnableSoftmax


class THORLearnableViTAttention(nn.Module):
    def __init__(self, approx_config, config, layer_idx=None):
        super().__init__()
        self.config = config
        self.n_embd = config.hidden_size
        self.n_head = config.num_attention_heads
        assert self.n_embd % self.n_head == 0, (
            f"hidden_size {self.n_embd} not divisible by n_head {self.n_head}"
        )
        self.head_dim = self.n_embd // self.n_head
        self.scaling = self.head_dim ** -0.5          # ViTSelfAttention.scaling
        self.layer_idx = layer_idx

        qkv_bias = getattr(config, "qkv_bias", True)
        self.query = nn.Linear(self.n_embd, self.n_embd, bias=qkv_bias)
        self.key = nn.Linear(self.n_embd, self.n_embd, bias=qkv_bias)
        self.value = nn.Linear(self.n_embd, self.n_embd, bias=qkv_bias)
        self.output_dense = nn.Linear(self.n_embd, self.n_embd)

        self.attn_dropout = nn.Dropout(getattr(config, "attention_probs_dropout_prob", 0.0))
        self.output_dropout = nn.Dropout(getattr(config, "hidden_dropout_prob", 0.0))

        # bidirectional: THORLearnableSoftmax handles causal_mask=None (k_eff = T)
        self.softmax = THORLearnableSoftmax(approx_config, config)

    def from_layer(self, vit_attention):
        with torch.no_grad():
            if hasattr(vit_attention, "q_proj"):       # transformers v5 (flat)
                pairs = ((self.query, vit_attention.q_proj),
                         (self.key, vit_attention.k_proj),
                         (self.value, vit_attention.v_proj),
                         (self.output_dense, vit_attention.o_proj))
            else:                                       # v4 (nested)
                sa = vit_attention.attention
                pairs = ((self.query, sa.query), (self.key, sa.key),
                         (self.value, sa.value),
                         (self.output_dense, vit_attention.output.dense))
            for mine, theirs in pairs:
                mine.weight.copy_(theirs.weight)
                if theirs.bias is not None and mine.bias is not None:
                    mine.bias.copy_(theirs.bias)
        return self

    def forward(self, hidden_states, attention_mask=None, **kwargs):
        B, T, C = hidden_states.size()

        q = self.query(hidden_states).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.key(hidden_states).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = self.value(hidden_states).view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        att = torch.matmul(q, k.transpose(-2, -1)) * self.scaling
        att = self.softmax(att, causal_mask=None)     # bidirectional (k_eff = T)
        att = att.type(v.dtype)
        att = self.attn_dropout(att)

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.output_dropout(self.output_dense(y))
        # transformers v5 ViTLayer unpacks (attn_output, attn_weights)
        return y, None
