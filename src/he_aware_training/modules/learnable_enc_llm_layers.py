import math
from typing import Optional

import torch
import torch.nn as nn

from he_aware_training.modules.learnable_components import (
    LearnedGoldschmidt,
)
from he_aware_training.modules.learnable_layers import InvSqrtApproximation
from he_aware_training.approximation.classic import polyval_torch, compute_optimal_linear_init
from he_aware_training.utils.utils import squeeze

# TODO: this approximation is too scusceptible to the max estimation problem with vanishing denominator....
class EncLLMLearnedHESoftmax(nn.Module):
    """Softmax approximation using exp-by-definition and Goldschmidt division."""

    class ExpApproximationByDefinition(nn.Module):
        def __init__(self, n_iters=10):
            super().__init__()
            self.n_iters = n_iters

        def forward(self, x):
            orig_dtype = x.dtype
            x = x.float()
            x = 1 + x / (2 ** self.n_iters)
            for _ in range(self.n_iters):
                x = x * x
            return x.to(orig_dtype)

    def __init__(self, approx_config, config):
        super().__init__()
        self.config = config
        self.exp_iters = approx_config.exp_iters

        squeeze_bound = getattr(approx_config, 'squeeze_bound', None)
        if squeeze_bound is None:
            squeeze_bound = (2 ** (self.exp_iters - 2)) - 8
            
        self.squeeze_bound = float(squeeze_bound)
        self.squeeze_strength = float(getattr(approx_config, 'squeeze_strength', 1.0))
        self.g_squeeze_d_min = float(getattr(approx_config, 'g_squeeze_d_min', 1.0))
        self.g_squeeze_d_max = float(approx_config.g_squeeze_d_max)
        self.g_squeeze_strength = float(approx_config.g_squeeze_strength)

        self.act = self.ExpApproximationByDefinition(approx_config.exp_iters)

        self.ponder_goldschmidt = LearnedGoldschmidt(
            max_iters=approx_config.g_iters,
            min_iter=approx_config.g_min_iter,
            soft_training=approx_config.soft_training,
            halt_init_low=approx_config.halt_init_low,
            halt_init_high=approx_config.halt_init_high,
            halt_init_at=approx_config.g_halt_at,
            gradient_checkpointing=approx_config.gradient_checkpointing,
            analytic=approx_config.g_analytic,
            wall_iter=approx_config.g_wall_iter,
            inference_mode=approx_config.inference_mode,
        )

        self.init_mode = getattr(approx_config, 'g_init_mode', 'heuristic')
        d_min = float(approx_config.g_init_d_min)
        d_max = float(approx_config.g_init_d_max)

        if self.init_mode == 'constant':
            c = 2.0 / (d_min + d_max)
            self.register_buffer('g_c', torch.tensor(c, dtype=torch.float32))
        elif self.init_mode == 'linear':
            alpha_init, beta_init = compute_optimal_linear_init(d_min, d_max)
            self.register_buffer('g_alpha', torch.tensor(alpha_init, dtype=torch.float32))
            self.register_buffer('g_beta', torch.tensor(beta_init, dtype=torch.float32))
        elif self.init_mode != 'heuristic':
            raise ValueError(f"Unknown g_init_mode: {self.init_mode}")

        self.register_buffer('m_star', None)

    def forward(self, x, causal_mask=None, q_offset=0):
        x = squeeze(x, min_val=-self.squeeze_bound, max_val=self.squeeze_bound,
                    strength=self.squeeze_strength)

        x_penalized = x - (2**(self.exp_iters-1)) * causal_mask

        T_q = x_penalized.shape[-2]

        if self.m_star is not None:
            m = self.m_star[q_offset:q_offset + T_q].view(1, 1, -1, 1)
            diff = x_penalized - m
            x_shift = squeeze(diff, max_val=0.0, strength=self.squeeze_strength)
            row_max = m
        else:
            row_max = x_penalized.amax(dim=-1, keepdim=True)
            diff = x_penalized - row_max
            x_shift = diff

        exp_x = self.act(x_shift)

        sum_exp_raw = exp_x.sum(dim=-1, keepdim=True)

        if self.g_squeeze_d_max > 0:
            sum_exp = squeeze(sum_exp_raw, min_val=self.g_squeeze_d_min, max_val=self.g_squeeze_d_max, strength=self.g_squeeze_strength)
        else:
            sum_exp = sum_exp_raw

        if self.init_mode == 'heuristic':
            base_avg = (0.5 + 0.5 * torch.exp(-2.0 * row_max.clamp(min=0.0))).clamp(min=1e-6)
            num_valid = (~causal_mask).sum(dim=-1, keepdim=True).to(x.dtype).clamp(min=1.0)
            f_init = 1.0 / (1.0 + (num_valid - 1) * base_avg)
            f_init = f_init.expand_as(sum_exp)
        elif self.init_mode == 'constant':
            f_init = self.get_buffer('g_c').expand_as(sum_exp)
        elif self.init_mode == 'linear':
            f_init = self.get_buffer('g_alpha') - self.get_buffer('g_beta') * sum_exp
        else:
            raise ValueError(f"Unknown init_mode: {self.init_mode}")

        inv_sum_exp = self.ponder_goldschmidt(
            torch.ones_like(sum_exp),
            sum_exp,
            (None, None),
            F=f_init,
        )

        return exp_x * inv_sum_exp


class EncLLMLearnedHEAttention(nn.Module):
    """Multi-head self-attention with EncLLM-approximated softmax."""

    def __init__(self, approx_config, config, layer_idx=None):
        super().__init__()
        self.config = config
        self.n_embd = config.hidden_size
        self.n_head = config.num_attention_heads

        assert (
            self.n_embd % self.n_head == 0
        ), f"n_embd {self.n_embd} not divisible by n_head {self.n_head}"
        self.head_dim = self.n_embd // self.n_head

        self.layer_idx = layer_idx

        self.c_attn = nn.Linear(self.n_embd, 3 * self.n_embd)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd)

        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)

        self.softmax = EncLLMLearnedHESoftmax(approx_config, config)

    def from_layer(self, gpt2_attn):
        """Copy weights from a GPT2Attention layer (handles Conv1D -> Linear transpose)."""
        with torch.no_grad():
            self.c_attn.weight.copy_(gpt2_attn.c_attn.weight.T)
            self.c_attn.bias.copy_(gpt2_attn.c_attn.bias)
            self.c_proj.weight.copy_(gpt2_attn.c_proj.weight.T)
            self.c_proj.bias.copy_(gpt2_attn.c_proj.bias)
            self.attn_dropout.p = gpt2_attn.attn_dropout.p
            self.resid_dropout.p = gpt2_attn.resid_dropout.p
        return self

    def forward(
        self,
        hidden_states=None,
        past_key_values=None,
        cache_position=None,
        attention_mask=None,
        use_cache=False,
        output_attentions=False,
        **kwargs,
    ):
        if hidden_states is None:
            raise ValueError(
                "EncLLMLearnedHEAttention received None for 'hidden_states'"
            )

        B, T, C = hidden_states.size()

        qkv = self.c_attn(hidden_states)
        q, k, v = qkv.split(self.n_embd, dim=2)

        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        if past_key_values is not None:
            k, v = past_key_values.update(
                k, v, self.layer_idx, {"cache_position": cache_position}
            )

        T_q = q.shape[-2]
        T_kv = k.shape[-2]
        offset = T_kv - T_q

        att = torch.matmul(q, k.transpose(-2, -1))
        att = att / math.sqrt(self.head_dim)

        if attention_mask is not None:
            mask_slice = attention_mask[..., :T_kv]
            causal_mask = mask_slice < -1.0
        else:
            causal_mask = torch.triu(
                torch.ones((T_q, T_kv), dtype=torch.bool, device=hidden_states.device),
                diagonal=offset + 1,
            )

        att = self.softmax(att, causal_mask=causal_mask, q_offset=offset)
        att = att.type(v.dtype)
        att = self.attn_dropout(att)

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T_q, C)
        y = self.resid_dropout(self.c_proj(y))

        return y, None


class EncLLMLearnedGeLU(nn.Module):

    def __init__(self):
        super().__init__()
        self.max_val: Optional[float] = None

        self.register_buffer("t1", torch.tensor(-4.0))
        self.register_buffer("t2", torch.tensor(-1.95))
        self.register_buffer("t3", torch.tensor(3.0))

        _polys = {
            "f0": torch.tensor(
                [-0.5054031199708174, -0.42226581151983866, -0.11807612951181953, -0.011034134030615728],
                dtype=torch.float64,
            ),
            "f1": torch.tensor(
                [0.008526321541038084, 0.5, 0.3603292692789629, 0.0, -0.037688200365904236, 0.0, 0.0018067462606141187],
                dtype=torch.float64,
            ),
            "f4": torch.tensor(
                [0.0, 315/128, 0.0, -420/128, 0.0, 378/128, 0.0, -180/128, 0.0, 35/128],
                dtype=torch.float64,
            ),
            "g4": torch.tensor(
                [0.0, 5850/1024, 0.0, -34974/1024, 0.0, 97015/1024, 0.0, -113492/1024, 0.0, 46623/1024],
                dtype=torch.float64,
            ),
        }

        for name, power_coeffs in _polys.items():
            self.register_buffer(f"{name}_coeffs", power_coeffs)

    def _sign_approx(self, z):
        """h(z) = f_4(f_4(g_4(g_4(z)))), approximates sign(z) for z in [-1, 1]."""
        z64 = z.to(torch.float64)
        h = polyval_torch(z64, self.g4_coeffs)
        h = polyval_torch(h, self.g4_coeffs)
        h = polyval_torch(h, self.f4_coeffs)
        h = polyval_torch(h, self.f4_coeffs)
        return h.to(z.dtype)

    def _indicator_lt(self, x, threshold, scale):
        """Smooth approximation of 1_{x < threshold} in [0, 1]."""
        z = (threshold - x) / scale
        return (1.0 + self._sign_approx(z)) * 0.5

    def from_layer(self, gelu_layer):
        return self

    def forward(self, x):
        if self.max_val is not None and self.training:
            x = squeeze(x, min_val=-self.max_val, max_val=self.max_val)

        if self.training:
            x_min = x.amin(dim=-1, keepdim=True)
            x_max = x.amax(dim=-1, keepdim=True)
            scale = torch.maximum(self.t3 - x_min, x_max - self.t1).clamp(min=1e-6)
        else:
            m = float(self.max_val) if self.max_val is not None else 16.0
            scale_val = m + max(abs(float(self.t1)), abs(float(self.t3)))
            scale = torch.tensor(scale_val, dtype=x.dtype, device=x.device)

        ind1 = self._indicator_lt(x, self.t1, scale)
        ind2 = self._indicator_lt(x, self.t2, scale)
        ind3 = self._indicator_lt(x, self.t3, scale)

        x64 = x.to(torch.float64)
        f0 = polyval_torch(x64, self.f0_coeffs).to(x.dtype)
        f1 = polyval_torch(x64, self.f1_coeffs).to(x.dtype)

        return f0 * (ind2 - ind1) + f1 * (ind3 - ind2) + x * (1.0 - ind3)
