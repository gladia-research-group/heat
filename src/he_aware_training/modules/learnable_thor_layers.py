import math
import torch
import torch.nn as nn

from he_aware_training.approximation.classic import chebyshev_series_torch, polyval_torch
from he_aware_training.modules.learnable_components import LearnedGoldschmidt
from he_aware_training.utils.utils import squeeze, soft_squeeze, observe_range, record_squeeze_stat


class THORLearnableSoftmax(nn.Module):
    def __init__(self, approx_config, config):
        super().__init__()
        self.config = config

        thor = approx_config.thor

        self.squeeze_bound = thor.squeeze_bound
        self.squeeze_strength = thor.squeeze_strength
        self.eval_squeeze = approx_config.eval_squeeze
        self.train_squeeze = approx_config.train_squeeze
        self._squeeze = soft_squeeze if thor.soft_squeeze else squeeze
        self.iterative = thor.calib_path is not None
        if not self.iterative:
            self.mid_x = 0.0
            return

        self.mid_x = thor.mid_x
        self.input_scale = thor.input_scale
        self.n_squarings = thor.n_squarings
        self.refinement_iters = thor.refinement_iters
        self.exp_mode = getattr(thor, "exp_mode", "cheb_series")
        self.refine_mode = getattr(thor, "refine_mode", "cuda_kc")

        self.ponder_goldschmidt_init = LearnedGoldschmidt(
            max_iters=thor.gs_iters_scaled + 1,
            min_iter=thor.g_min_iter_init,
            soft_training=approx_config.soft_training,
            halt_init_low=approx_config.halt_init_low,
            halt_init_high=approx_config.halt_init_high,
            halt_init_at=thor.g_halt_at_init,
            gradient_checkpointing=approx_config.gradient_checkpointing,
            analytic=thor.g_analytic_init,
            wall_iter=thor.g_wall_iter_init,
            inference_mode=approx_config.inference_mode,
        )
        self.ponder_goldschmidt_refine = LearnedGoldschmidt(
            max_iters=thor.gs_iters_refine_scaled + 1,
            min_iter=thor.g_min_iter_refine,
            soft_training=approx_config.soft_training,
            halt_init_low=approx_config.halt_init_low,
            halt_init_high=approx_config.halt_init_high,
            halt_init_at=thor.g_halt_at_refine,
            gradient_checkpointing=approx_config.gradient_checkpointing,
            analytic=thor.g_analytic_refine,
            wall_iter=thor.g_wall_iter_refine,
            inference_mode=approx_config.inference_mode,
        )

        def _reg(name, tensor):
            # operator learning (C arm): GS seeds become task-trainable from the calib point
            if approx_config.learn_operator:
                setattr(self, name, nn.Parameter(tensor))
            else:
                self.register_buffer(name, tensor)
        _reg("init_alpha", torch.tensor(float(thor.init_alpha), dtype=torch.float32))
        _reg("init_beta", torch.tensor(float(thor.init_beta), dtype=torch.float32))
        _reg("refine_alpha", torch.tensor(list(thor.refine_alpha), dtype=torch.float32))
        _reg("refine_beta", torch.tensor(list(thor.refine_beta), dtype=torch.float32))
        self.register_buffer("sm_kc_r",
            torch.tensor(list(thor.sm_kc_r), dtype=torch.float32))

        if self.exp_mode == "legacy_poly":
            self.register_buffer("exp_poly_coeffs",
                torch.tensor(list(thor.poly_coeffs), dtype=torch.float64))
        else:
            self.cheb_a = thor.cheb_a
            self.cheb_b = thor.cheb_b
            self.register_buffer("cheb_coeffs",
                torch.tensor(list(thor.cheb_coeffs), dtype=torch.float64))

    def _exp_approx(self, x):
        orig_dtype = x.dtype
        if self.exp_mode == "legacy_poly":
            y = ((x - self.mid_x) / self.input_scale).to(torch.float64)
            val = polyval_torch(y, self.exp_poly_coeffs)
            for _ in range(self.n_squarings):
                val = val * val
            return val.to(orig_dtype)

        xc = (x - self.mid_x).to(torch.float64)
        if (self.training and self.train_squeeze) or self.eval_squeeze:
            xc = squeeze(
                xc,
                self.cheb_a * self.input_scale,
                self.cheb_b * self.input_scale,
                strength=self.squeeze_strength,
                tag="softmax.exp_x",
            )
        elif self.training:
            observe_range(
                xc,
                self.cheb_a * self.input_scale,
                self.cheb_b * self.input_scale,
                tag="softmax.exp_x",
            )
        val = chebyshev_series_torch(
            xc, self.cheb_coeffs,
            self.cheb_a * self.input_scale,
            self.cheb_b * self.input_scale,
        )
        for _ in range(self.n_squarings):
            val = val * val
        return val.to(orig_dtype)

    def _ponder_inv(self, denom, alpha, beta, ponder, tag="softmax.df"):
        """Goldschmidt inv on the UNNORMALISED denominator. α/β are calibrated
        for [d_min, d_max] (the per-step `1/d_min` factor is already folded in
        at calibration time), so F_init ≈ 1/denom directly. Mirrors the CUDA
        kernel's `cfg.refine_alpha[i] / cfg.refine_beta[i]` usage."""
        f_init = alpha - beta * denom
        raw_df = denom * f_init
        df = raw_df.clamp(min=1e-3, max=2.0 - 1e-3)
        if self.training:
            record_squeeze_stat(tag, raw_df - df.detach())
        f_init = df / denom.clamp(min=df.detach() / 100.0)
        return ponder(
            torch.ones_like(denom),
            denom,
            (None, None),
            F=f_init,
        )

    def forward(self, x, causal_mask=None, q_offset=0):

        if not self.iterative:
            # squeeze-warmup: exact softmax over the valid (causal) keys
            if causal_mask is not None:
                x = x.masked_fill(causal_mask, float('-inf'))
            return torch.softmax(x, dim=-1).to(x.dtype)

        exp_x = self._exp_approx(x)

        if causal_mask is not None:
            keep = (~causal_mask).to(exp_x.dtype)
            exp_x = exp_x * keep
            # `inf.k_count` fold from attention.cu — makes init α/β k-invariant.
            k_eff = keep.sum(dim=-1, keepdim=True).clamp(min=1.0)
        else:
            k_eff = torch.full_like(exp_x.sum(dim=-1, keepdim=True),
                                    float(exp_x.shape[-1]))

        sum_exp = exp_x.sum(dim=-1, keepdim=True)
        inv_sum = self._ponder_inv(
            sum_exp,
            self.init_alpha / k_eff,
            self.init_beta  / (k_eff * k_eff),
            self.ponder_goldschmidt_init,
            tag="softmax.df_init",
        )

        if self.refine_mode == "legacy_global":
            for i in range(self.refinement_iters):
                exp_x = (exp_x * inv_sum) ** 2
                sum_exp = exp_x.sum(dim=-1, keepdim=True)
                inv_sum = self._ponder_inv(
                    sum_exp,
                    self.refine_alpha[i],
                    self.refine_beta[i],
                    self.ponder_goldschmidt_refine,
                )
            return exp_x * inv_sum

        sqrt_k = k_eff.sqrt()
        C = self.sm_kc_r.numel() // self.refinement_iters if self.sm_kc_r.numel() else 0
        y = exp_x * inv_sum                      # softmax after init
        for i in range(self.refinement_iters):
            z = (y * y) * (0.5 * sqrt_k)
            s = z.sum(dim=-1, keepdim=True)
            if C:
                kpos = (k_eff.long() - 1).clamp(0, C - 1)
                r = self.sm_kc_r[i * C + kpos]
            else:
                r = (4.0 / k_eff).clamp(max=1.0)          # fallback kc_ref=4
            inv = self._ponder_inv(
                s,
                self.refine_alpha[i] * r.sqrt(),
                self.refine_beta[i] * r,
                self.ponder_goldschmidt_refine,
                tag="softmax.df_refine",
            )
            y = z * inv

        return y


class THORLearnableHEAttention(nn.Module):
    """Multi-head self-attention with THOR-approximated softmax.

    Mirrors EncLLMLearnedHEAttention's surgery contract (same __init__ signature,
    from_layer, and forward kwargs) so it can be swapped in by the existing surgery
    pass via the attention_kind config field.
    """

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

        self.softmax = THORLearnableSoftmax(approx_config, config)

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
                "THORLearnableHEAttention received None for 'hidden_states'"
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
