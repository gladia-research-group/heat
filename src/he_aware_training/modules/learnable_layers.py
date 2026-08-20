import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from he_aware_training.approximation.classic import (
    chebyshev_series_torch,
    compute_optimal_linear_init,
    compute_optimal_linear_init_minimax,
    polyval_torch,
    rational_remez,
    taylor_inv_sqrt,
    taylor_inv_sqrt_coeffs,
)
from he_aware_training.modules.learnable_components import (
    LearnedGoldschmidt,
    LearnedNewtonInverseSqrt
)
from he_aware_training.utils.utils import squeeze, observe_range

GELU_RUNTIME_H = 0.5


def _inverse_sqrt(x):
    return 1.0 / np.sqrt(x)

class InvSqrtApproximation(nn.Module):
    def __init__(self,
                 min_val,
                 max_val,
                 d_num=3,
                 d_den=1,
                 coeffs_method="remez",
                 lin_alpha=None,
                 lin_beta=None,
                 Ncoeffs=None,
                 Dcoeffs=None,
                 inv_out_scale=None,
                 taylor_z0=None,
                 halt_init_low=0.05,
                 halt_init_high=0.95,
                 g_optimal_iters=None,
                 n_optimal_iters=None,
                 soft_training=True,
                 gradient_checkpointing=False,
                 g_iters=10,
                 g_min_iter=0,
                 n_iters=10,
                 n_min_iter=0,
                 g_analytic=False,
                 n_analytic=False,
                 g_wall_iter=None,
                 n_wall_iter=None,
                 inference_mode="threshold",
                 eval_squeeze=False,
                 train_squeeze=True,
                 learn_operator=False,
                 tag="invsqrt",
                ):
        super().__init__()
        self.z_min = min_val
        self.z_max = max_val
        self.use_fixed_z_max = False
        self.eval_squeeze = eval_squeeze
        self.train_squeeze = train_squeeze
        self.tag = tag

        def _reg(name, tensor):
            # operator learning (C arm): calib coeffs/seeds become task-trainable,
            # init at the calibrated (validated) point; export path unchanged
            if learn_operator:
                setattr(self, name, nn.Parameter(tensor))
            else:
                self.register_buffer(name, tensor)
        self._reg_op = _reg

        self.use_taylor = (coeffs_method == "taylor")
        self.inv_out_scale = inv_out_scale

        if coeffs_method == "remez":
            if Ncoeffs is not None and Dcoeffs is not None:
                p_np = np.asarray(list(Ncoeffs), dtype=np.float64)
                q_np = np.asarray(list(Dcoeffs), dtype=np.float64)
            else:
                p_np, q_np, q_min, q_max = rational_remez(
                    _inverse_sqrt,
                    min_val,
                    max_val,
                    d_num,
                    d_den,
                )
                if lin_alpha is None or lin_beta is None:
                    lin_alpha, lin_beta = compute_optimal_linear_init(q_min, q_max)
            self._reg_op("q", torch.tensor(q_np, dtype=torch.float32))
            self._reg_op("lin_alpha", torch.tensor(float(lin_alpha), dtype=torch.float32))
            self._reg_op("lin_beta", torch.tensor(float(lin_beta), dtype=torch.float32))
        elif coeffs_method == "taylor":
            if taylor_z0 is not None:
                z0 = float(taylor_z0)
                p_np = taylor_inv_sqrt_coeffs(z0)
            else:
                p_np, z0 = taylor_inv_sqrt()
            self.register_buffer("taylor_z0", torch.tensor(z0, dtype=torch.float32))
        else:
            raise ValueError(f"Unknown coeffs method: {coeffs_method}")

        self._reg_op("p", torch.tensor(p_np, dtype=torch.float32))

        n_optimal = n_optimal_iters
        n_halt_at = (n_optimal - 1) if n_optimal is not None else None

        if not self.use_taylor:
            g_optimal = g_optimal_iters
            g_halt_at = (g_optimal - 1) if g_optimal is not None else None
            self.ponder_goldschmidt = LearnedGoldschmidt(
                max_iters=g_iters,
                min_iter=g_min_iter,
                soft_training=soft_training,
                halt_init_low=halt_init_low,
                halt_init_high=halt_init_high,
                halt_init_at=g_halt_at,
                gradient_checkpointing=gradient_checkpointing,
                analytic=g_analytic,
                wall_iter=g_wall_iter,
                inference_mode=inference_mode,
            )

        self.ponder_newton = LearnedNewtonInverseSqrt(
            max_iters=n_iters,
            min_iter=n_min_iter,
            soft_training=soft_training,
            halt_init_low=halt_init_low,
            halt_init_high=halt_init_high,
            halt_init_at=n_halt_at,
            gradient_checkpointing=gradient_checkpointing,
            analytic=n_analytic,
            wall_iter=n_wall_iter,
            inference_mode=inference_mode,
        )

    def forward(self, z):
        # stability-class: z out of the Remez/GS domain can't renormalize (inv_sqrt goes
        # flat → block output ∝ input → 1e15 in one block); keep guarded even train_squeeze=off
        if self.training or self.eval_squeeze:
            z = squeeze(z, self.z_min, self.z_max, tag=f"{self.tag}.z")

        if self.use_taylor:
            if self.use_fixed_z_max:
                s = torch.as_tensor(1.0 / self.z_max, dtype=z.dtype, device=z.device)
            else:
                z_max = z.amax(dim=1, keepdim=True)  # (B, 1, 1)
                s = 1.0 / z_max
            u = z * s - self.taylor_z0
            y_approx = polyval_torch(u, self.p) * torch.sqrt(s)
        else:
            num = polyval_torch(z, self.p)
            den = polyval_torch(z, self.q)
            y_approx = self.ponder_goldschmidt(num, den, (self.lin_alpha, self.lin_beta))

        nx = z if self.inv_out_scale is None else z / (self.inv_out_scale ** 2)
        if self.training or self.eval_squeeze:
            y_approx = squeeze(y_approx, None, (3.0 / nx).sqrt() * 0.95, tag=f"{self.tag}.newton_seed")
            y_approx = squeeze(y_approx, 0.0, None)
        inv_sqrt = self.ponder_newton(nx, y_approx)
        return inv_sqrt
    
def _exp(x):
    return np.exp(x)

class ExpApproximation(nn.Module):
    def __init__(self,
                 min_val,          # e.g., -16.0
                 max_val=0.0,      # strictly <= 0
                 d_num=4,          # P_3
                 d_den=4,          # Q_4
                 k_squarings=4,
                 coeffs_method="remez",
                 halt_init_low=0.05,
                 halt_init_high=0.95,
                 g_optimal_iters=None,
                 soft_training=True,
                 gradient_checkpointing=False,
                 g_iters=6,
                 g_min_iter=0,
                 g_analytic=False,
                 g_wall_iter=None,
                 inference_mode="threshold",
                ):
        super().__init__()
        self.x_min = min_val
        self.x_max = max_val 
        self.k_squarings = k_squarings

        u_min = min_val * (2.0 ** -k_squarings)
        u_max = max_val * (2.0 ** -k_squarings)

        if coeffs_method == "remez":
            p_np, q_np, q_min, q_max = rational_remez(
                _exp,
                u_min,
                u_max,
                d_num,
                d_den,
            )
        else:
            raise ValueError(f"Unknown coeffs method: {coeffs_method}")

        self.register_buffer("p", torch.tensor(p_np, dtype=torch.float32))
        self.register_buffer("q", torch.tensor(q_np, dtype=torch.float32))

        alpha_val, beta_val = compute_optimal_linear_init_minimax(q_min, q_max)
        self.register_buffer("lin_alpha", torch.tensor(alpha_val, dtype=torch.float32))
        self.register_buffer("lin_beta", torch.tensor(beta_val, dtype=torch.float32))

        g_optimal = g_optimal_iters
        g_halt_at = (g_optimal - 1) if g_optimal is not None else None

        self.ponder_goldschmidt = LearnedGoldschmidt(
            max_iters=g_iters,
            min_iter=g_min_iter,
            soft_training=soft_training,
            halt_init_low=halt_init_low,
            halt_init_high=halt_init_high,
            halt_init_at=g_halt_at,
            gradient_checkpointing=gradient_checkpointing,
            analytic=g_analytic,
            wall_iter=g_wall_iter,
            inference_mode=inference_mode,
        )

    def forward(self, x):
        if self.training:
            x = squeeze(x, self.x_min)

        u = x * (2.0 ** -self.k_squarings)

        num = polyval_torch(u, self.p)
        den = polyval_torch(u, self.q)

        y_approx = self.ponder_goldschmidt(num, den, (self.lin_alpha, self.lin_beta))

        return y_approx ** (2.0 ** self.k_squarings)

class LearnedLayerNormHE(nn.Module):
    def __init__(self, config, n_embd):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(n_embd))
        self.bias = nn.Parameter(torch.zeros(n_embd))
        ln = config.ln
        self.eps = 1e-5   # placeholder; from_layer copies the real GPT-2 LN eps after construction

        self.iterative = ln.calib_path is not None
        if not self.iterative:
            return

        self.center_scale = ln.center_scale          # c  — input scale (LN-2b)
        self.inv_out_scale = ln.inv_out_scale        # s  — output value-scale (LN-2c, folded into weight)
        self.register_buffer("center_scale_sq",      # per-position c_eff²  (LN-2d)
            torch.tensor(list(ln.center_scale_sq), dtype=torch.float32))

        self.inv_sqrt_approx = InvSqrtApproximation(
            min_val=ln.fit_lo,                        # squeeze on the c²·var (Remez fit) domain
            max_val=ln.fit_hi,
            coeffs_method=ln.method,                  # "remez"
            Ncoeffs=ln.Ncoeffs,
            Dcoeffs=ln.Dcoeffs,
            lin_alpha=ln.lin_alpha,
            lin_beta=ln.lin_beta,
            inv_out_scale=ln.inv_out_scale,           # Newton value-scale (used in LN-2c)
            halt_init_low=config.halt_init_low,
            halt_init_high=config.halt_init_high,
            g_optimal_iters=ln.g_optimal_iters,
            n_optimal_iters=ln.optimal_iters,
            soft_training=config.soft_training,
            gradient_checkpointing=config.gradient_checkpointing,
            g_iters=ln.gs_iters + 1,
            g_min_iter=ln.g_min_iter,
            n_iters=ln.nr_iters + ln.n_min_iter + 1,
            n_min_iter=ln.n_min_iter,
            g_analytic=ln.g_analytic,
            n_analytic=ln.n_analytic,
            g_wall_iter=ln.g_wall_iter,
            n_wall_iter=ln.n_wall_iter,
            inference_mode=config.inference_mode,
            eval_squeeze=config.eval_squeeze,
            train_squeeze=config.train_squeeze,
            learn_operator=config.learn_operator,
            tag="ln",
        )

    def from_layer(self, layer_norm):
        with torch.no_grad():
            self.weight.copy_(layer_norm.weight)
            self.bias.copy_(layer_norm.bias)
        self.eps = layer_norm.eps

        return self

    def forward(self, x):
        if not self.iterative:
            return F.layer_norm(x, (x.shape[-1],), self.weight, self.bias, self.eps)

        if self.center_scale_sq.numel():
            T = x.shape[-2]
            pos = torch.arange(T, device=x.device).clamp(max=self.center_scale_sq.numel() - 1)
            c = self.center_scale_sq[pos].sqrt().view(1, T, 1)       # [1,T,1] broadcasts over [B,T,D]
        else:
            c = torch.as_tensor(self.center_scale, dtype=x.dtype, device=x.device)
        xs = c * x
        mean = xs.mean(dim=-1, keepdim=True)                          # = c·mean(x)
        cx = xs - mean                                               # = c·(x − mean)
        var = cx.var(dim=-1, keepdim=True, unbiased=False) + (c * c) * self.eps  # c²·(var+eps)

        inv_sqrt = self.inv_sqrt_approx(var)                         # → s·var^(−1/2)  (LN-2c)

        out = cx * inv_sqrt                                          # = s·LN
        return out * (self.weight / self.inv_out_scale) + self.bias  # γ/s · s·LN + β = γ·LN + β

class LearnedHEGeLU(nn.Module):
    def __init__(self, approx_config):
        super().__init__()

        sg = approx_config.softgelu
        self.eval_squeeze = approx_config.eval_squeeze
        self.train_squeeze = approx_config.train_squeeze

        self.iterative = sg.calib_path is not None
        if not self.iterative:
            return

        self.method = sg.method
        self.max_val: float = float(sg.xmax)
        self.squeeze_strength = sg.squeeze_strength

        if self.method == "chebyshev":
            self.register_buffer(
                "cheb_coeffs",
                torch.tensor(list(sg.cheb_coeffs), dtype=torch.float32),
            )
            self.cheb_a = float(sg.cheb_a)
            self.cheb_b = float(sg.cheb_b)
            return

        # erf(x) ≈ (a·x/√(1+a²x²)) · (1 − b·exp(−c·x²)); GeLU(x) = 0.5·x·(1+erf(x/√2))
        self.register_buffer("a", torch.tensor(sg.a, dtype=torch.float32))
        self.register_buffer("b", torch.tensor(sg.b, dtype=torch.float32))
        self.register_buffer("c", torch.tensor(sg.c, dtype=torch.float32))
        self.exp_iters = sg.exp_iters
        # s: inv_sqrt outputs s·z^{-1/2}; softsign folds 1/s. null/absent ⇒ 1.0 (legacy configs,
        # mirrors the CUDA config_loader opt-binder default so Python == CUDA on old configs.json)
        self.inv_out_scale = 1.0 if sg.inv_out_scale is None else float(sg.inv_out_scale)

        gate_cheb = list(sg.gate_cheb_coeffs) if sg.gate_cheb_coeffs else None
        if gate_cheb is not None:
            self.register_buffer("gate_cheb_coeffs",
                                 torch.tensor(gate_cheb, dtype=torch.float64))
            self.gate_cheb_a = float(sg.gate_cheb_a)
            self.gate_cheb_b = float(sg.gate_cheb_b)
        else:
            self.gate_cheb_coeffs = None

        self.inv_sqrt_approx = InvSqrtApproximation(
            min_val=sg.z_min,
            max_val=sg.z_max,
            coeffs_method="remez",
            lin_alpha=sg.lin_alpha,
            lin_beta=sg.lin_beta,
            Ncoeffs=sg.Ncoeffs,
            Dcoeffs=sg.Dcoeffs,
            inv_out_scale=sg.inv_out_scale,   # Newton runs on z/s²; output = s·z^{-1/2} (matches FHE)
            halt_init_low=approx_config.halt_init_low,
            halt_init_high=approx_config.halt_init_high,
            g_optimal_iters=sg.g_optimal_iters,
            n_optimal_iters=sg.optimal_iters,
            soft_training=approx_config.soft_training,
            gradient_checkpointing=approx_config.gradient_checkpointing,
            g_iters=sg.gs_iters + 1,                        # config gs_iters (kernel count); mirror LN
            g_min_iter=sg.g_min_iter,
            n_iters=sg.newton_iters + sg.n_min_iter + 1,    # config newton_iters (kernel count); mirror LN
            n_min_iter=sg.n_min_iter,
            g_analytic=sg.g_analytic,
            n_analytic=sg.n_analytic,
            g_wall_iter=sg.g_wall_iter,
            n_wall_iter=sg.n_wall_iter,
            inference_mode=approx_config.inference_mode,
            eval_squeeze=approx_config.eval_squeeze,
            train_squeeze=approx_config.train_squeeze,
            learn_operator=approx_config.learn_operator,
            tag="gelu",
        )
        self.inv_sqrt_approx.use_fixed_z_max = True

        self.max_val: float = float(sg.xmax)
        self.squeeze_strength = sg.squeeze_strength
        self.eval_squeeze = approx_config.eval_squeeze

    def from_layer(self, gelu_layer):
        return self

    def forward(self, x):
        if not self.iterative:
            return F.gelu(x)          # squeeze-warmup: exact (erf) GeLU; range via ActivationRegularizer

        if ((self.training and self.train_squeeze) or self.eval_squeeze) and self.max_val is not None:
            x = squeeze(
                x,
                min_val=-self.max_val,
                max_val=+self.max_val,
                strength=self.squeeze_strength,
                tag="gelu.x",
            )
        elif self.training and self.max_val is not None:
            observe_range(x, -self.max_val, +self.max_val, tag="gelu.x")

        if self.method == "chebyshev":
            return chebyshev_series_torch(x, self.cheb_coeffs, self.cheb_a, self.cheb_b)

        ax = GELU_RUNTIME_H * self.a * x
        z = 1.0 + ax * ax
        inv = self.inv_sqrt_approx(z)                 # = s·z^{-1/2} (value-scaled, above the bts floor)
        saturation = ax * inv / self.inv_out_scale    # fold 1/s back: (ax/s)·(s·z^{-1/2}) = softsign

        if self.gate_cheb_coeffs is not None:
            axsq = (ax * ax).to(torch.float64)            # (h·a·x)², fed to the cheb exactly like FHE
            y = chebyshev_series_torch(axsq, self.gate_cheb_coeffs,
                                       self.gate_cheb_a, self.gate_cheb_b).to(x.dtype)
            for _ in range(self.exp_iters):
                y = y * y                                 # cheb(axsq)^(2^K) = exp(−c·x²) ∈ [0,1]
            gate = 1.0 - self.b * y
        else:
            gate = torch.ones_like(x)                     # gate-off block (no exp branch)

        return 0.5 * x * (1.0 + saturation * gate)


class THORCompositeGeLU(nn.Module):
    """THOR composite GeLU (Moon et al. 2024): GeLU(x) ≈ x·(P2(P1(x/S)) + 1/2),"""

    def __init__(self, approx_config):
        super().__init__()
        g = approx_config.gelu
        self.eval_squeeze = approx_config.eval_squeeze
        self.train_squeeze = approx_config.train_squeeze

        self.iterative = g.calib_path is not None
        if not self.iterative:
            return

        self.max_val: float = float(g.xmax)                 # S = calibrated range
        self.squeeze_strength = g.squeeze_strength
        self.register_buffer("thor_p1", torch.tensor(list(g.thor_p1), dtype=torch.float64))
        self.register_buffer("thor_p2", torch.tensor(list(g.thor_p2), dtype=torch.float64))

    def from_layer(self, gelu_layer):
        return self

    def forward(self, x):
        if not self.iterative:
            return F.gelu(x)          # squeeze-warmup: exact GeLU; range via ActivationRegularizer

        if (self.training and self.train_squeeze) or self.eval_squeeze:
            x = squeeze(x, min_val=-self.max_val, max_val=+self.max_val,
                        strength=self.squeeze_strength, tag="gelu.x")
        elif self.training:
            observe_range(x, -self.max_val, +self.max_val, tag="gelu.x")

        t = (x / self.max_val).to(torch.float64)            # t = x/S ∈ [-1,1] (x squeezed above)
        comp = polyval_torch(polyval_torch(t, self.thor_p1), self.thor_p2)
        return (x.to(torch.float64) * (comp + 0.5)).to(x.dtype) 


