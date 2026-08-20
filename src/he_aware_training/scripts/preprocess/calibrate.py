from pathlib import Path
import json
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention

try:
    from transformers.activations import GELUActivation, NewGELUActivation
    GELU_TYPES: tuple = (nn.GELU, GELUActivation, NewGELUActivation)
except ImportError:
    GELU_TYPES = (nn.GELU,)

from he_aware_training import PROJECT_ROOT
from he_aware_training.utils.surgery import perform_he_surgery
from he_aware_training.utils.checkpoint import load_init_checkpoint
from he_aware_training.modules.learnable_thor_layers import THORLearnableHEAttention
from he_aware_training.modules.learnable_vit_layers import THORLearnableViTAttention
from he_aware_training.modules.learnable_bert_layers import THORLearnableBertAttention
from transformers.models.vit.modeling_vit import ViTAttention
from transformers.models.bert.modeling_bert import BertSelfAttention

ATTN_TYPES = (GPT2Attention, THORLearnableHEAttention,
              THORLearnableViTAttention, THORLearnableBertAttention,
              ViTAttention, BertSelfAttention)


def _qk(mod, hs):
    """Query and key for one attention module, whatever its projection layout."""
    if hasattr(mod, "c_attn"):                       # GPT-2: fused qkv
        C = hs.shape[-1]
        q, k, _ = mod.c_attn(hs).split(C, dim=2)
        return q, k
    if hasattr(mod, "query") and hasattr(mod, "key"):   # BERT, surgery'd ViT/BERT
        return mod.query(hs), mod.key(hs)
    if hasattr(mod, "q_proj") and hasattr(mod, "k_proj"):   # stock ViT
        return mod.q_proj(hs), mod.k_proj(hs)
    inner = getattr(mod, "attention", None)             # stock wrappers
    if inner is not None and hasattr(inner, "query"):
        return inner.query(hs), inner.key(hs)
    return None, None


def _n_heads(mod):
    for attr in ("num_heads", "num_attention_heads"):
        v = getattr(mod, attr, None)
        if isinstance(v, int):
            return v
    return mod.config.num_attention_heads
from he_aware_training.approximation.classic import (
    polyval_torch,
    rational_remez,
)
from he_aware_training.scripts.preprocess._calib_common import _safe_quantile


def _fit_cutmax(logits_rows, cfg):
    from he_aware_training.scripts.preprocess._cutmax_engine import (
        fit_cutmax_section,
    )
    knobs = dict(
        gap_floor=cfg.cutmax_gap_floor,
        entry_scale=cfg.cutmax_entry_scale,
        noise=cfg.cutmax_noise,
        margin=cfg.cutmax_margin,
        c0=cfg.cutmax_c0,
        c_late=cfg.cutmax_c_late,
        shift_max=cfg.cutmax_shift_max,
        spread_cap=cfg.cutmax_spread_cap,
        t_max=cfg.cutmax_t_max,
        mass_target=cfg.cutmax_mass_target,
        sum_margin=cfg.cutmax_sum_margin,
        newton_per_pass=cfg.cutmax_newton_per_pass,
        gs_sum_iters=cfg.cutmax_gs_sum_iters,
        band_margin=cfg.cutmax_band_margin,
        floor_safety=cfg.cutmax_floor_safety,
        floor1=cfg.cutmax_floor1,
        floor2=cfg.cutmax_floor2,
        wall_x1=cfg.cutmax_wall_x1,
        wall_x2=cfg.cutmax_wall_x2,
        wall_y1=cfg.cutmax_wall_y1,
        wall_y2=cfg.cutmax_wall_y2,
        conv=cfg.cutmax_conv,
    )
    return fit_cutmax_section(logits_rows.numpy(), knobs)


@torch.no_grad()
def _pick_attn_bucket(name, by_T):
    """One sequence length per site: take the widest bucket, and say what that drops."""
    T = max(by_T)
    if len(by_T) > 1:
        kept = sum(r.size(0) for r in by_T[T]["rows"])
        dropped = sum(r.size(0) for t, e in by_T.items() if t != T for r in e["rows"])
        print(f"[attn] {name}: {len(by_T)} sequence lengths seen "
              f"{sorted(by_T)}; fitting T={T} ({kept} rows, {dropped} dropped)")
    e = by_T[T]
    return (torch.cat(e["rows"], dim=0).float().cpu(),
            torch.cat(e["row_q"], dim=0).cpu(),
            T)


def collect_samples(model, get_batch, n_batches: int, rows_per_batch: int,
                    logit_rows: int = 0):
    gelu_bufs: dict[str, list[torch.Tensor]] = {}
    logit_bufs: list[torch.Tensor] = []
    ln_bufs: dict[str, list[torch.Tensor]] = {}
    ln_pos_sum: dict[str, torch.Tensor] = {}
    ln_pos_cnt: dict[str, torch.Tensor] = {}   # per-position sample count           
    ln_pos_max: dict[str, torch.Tensor] = {}
    ln_cx_max: dict[str, list[torch.Tensor]] = {}
    ln_eps: dict[str, float] = {}
    attn_bufs: dict[str, dict] = {}

    def gelu_hook(name):
        def hook(module, inp, output):
            del module, output
            x = inp[0] if isinstance(inp, tuple) else inp
            if torch.is_tensor(x):
                gelu_bufs.setdefault(name, []).append(x.float().flatten().detach().cpu())
        return hook

    def ln_hook(name, eps):
        def hook(module, inp, output):
            del module, output
            x = inp[0] if isinstance(inp, tuple) else inp
            if torch.is_tensor(x):
                xf = x.float()
                z = xf.var(dim=-1, unbiased=False) + eps
                ln_bufs.setdefault(name, []).append(z.flatten().detach().cpu())
                if z.dim() == 2 and z.size(1) > 1:
                    zc = z.detach().cpu()
                    w = zc.size(1)
                    def _grow(t):
                        if t.numel() >= w:
                            return t
                        g = torch.zeros(w)
                        g[:t.numel()] = t
                        return g
                    if name not in ln_pos_sum:
                        ln_pos_sum[name] = torch.zeros(w)
                        ln_pos_max[name] = torch.zeros(w)
                        ln_pos_cnt[name] = torch.zeros(w)
                    else:
                        ln_pos_sum[name] = _grow(ln_pos_sum[name])
                        ln_pos_max[name] = _grow(ln_pos_max[name])
                        ln_pos_cnt[name] = _grow(ln_pos_cnt[name])
                    ln_pos_sum[name][:w] += zc.sum(dim=0)
                    ln_pos_max[name][:w] = torch.maximum(ln_pos_max[name][:w], zc.amax(dim=0))
                    ln_pos_cnt[name][:w] += zc.size(0)
                cx = xf - xf.mean(dim=-1, keepdim=True)
                ln_cx_max.setdefault(name, []).append(cx.abs().amax().reshape(1).detach().cpu())
        return hook

    def attn_hook(name):
        def hook(mod, args, kwargs):
            hs = kwargs.get("hidden_states", args[0] if args else None)
            if not torch.is_tensor(hs):
                return
            B, T, C = hs.shape
            H = _n_heads(mod)
            d = C // H
            q, k = _qk(mod, hs)
            if q is None:
                return
            q = q.view(B, T, H, d).transpose(1, 2)
            k = k.view(B, T, H, d).transpose(1, 2)
            scores2d = ((q @ k.transpose(-2, -1)) / math.sqrt(d)).reshape(-1, T)
            n = scores2d.size(0)
            keep = min(rows_per_batch, n)
            idx = torch.randperm(n, device=scores2d.device)[:keep]
            e = attn_bufs.setdefault(name, {}).setdefault(T, {"rows": [], "row_q": []})
            e["rows"].append(scores2d[idx].detach())
            e["row_q"].append((idx % T).detach())
        return hook

    handles = []
    for name, m in model.named_modules():
        if isinstance(m, GELU_TYPES):
            handles.append(m.register_forward_hook(gelu_hook(name)))
        elif isinstance(m, nn.LayerNorm):
            eps = float(getattr(m, "eps", 1e-5))
            ln_eps[name] = eps
            handles.append(m.register_forward_hook(ln_hook(name, eps)))
        elif isinstance(m, ATTN_TYPES):
            handles.append(m.register_forward_pre_hook(attn_hook(name), with_kwargs=True))

    collected = 0
    for _ in tqdm(range(n_batches), desc="collect", unit="batch"):
        X, _ = get_batch("train")
        if X is None:
            break
        out = model(X)
        if logit_rows and collected < logit_rows:
            lg = out.logits if hasattr(out, "logits") else out[0]
            lg2d = lg.float().reshape(-1, lg.size(-1))
            good = torch.isfinite(lg2d).all(dim=-1) & (lg2d.abs().amax(dim=-1) < 1e4)
            lg2d = lg2d[good]
            keep = min(logit_rows - collected, lg2d.size(0))
            idx = torch.randperm(lg2d.size(0), device=lg2d.device)[:keep]
            logit_bufs.append(lg2d[idx].detach().cpu())
            collected += keep
    for h in handles:
        h.remove()

    gelu = {n: torch.cat(parts) for n, parts in gelu_bufs.items()}
    ln = {n: torch.cat(parts) for n, parts in ln_bufs.items()}
    ln_cx = {n: float(torch.cat(parts).max()) for n, parts in ln_cx_max.items()}
    ln_pos: dict[str, dict] = {}
    for n in ln_pos_sum:
        ln_pos[n] = {
            "per_pos_mean": (ln_pos_sum[n] / ln_pos_cnt[n].clamp_min(1)).tolist(),
            "per_pos_max": ln_pos_max[n].tolist(),                         
        }
    attn = {
        n: _pick_attn_bucket(n, by_T)
        for n, by_T in attn_bufs.items()
    }
    logits_rows = torch.cat(logit_bufs) if logit_bufs else None
    return gelu, ln, ln_cx, ln_eps, attn, ln_pos, logits_rows

def _build_fit_grid(xmax, fit_grid, grid_points, sample_points, samples):
    if fit_grid == "uniform":
        return torch.linspace(-xmax, +xmax, grid_points, dtype=torch.float64)
    if fit_grid == "chebyshev":
        k = torch.arange(1, grid_points + 1, dtype=torch.float64)
        return xmax * torch.cos((2.0 * k - 1.0) / (2.0 * grid_points) * math.pi)
    if fit_grid == "samples":
        if samples is None:
            raise ValueError("fit_grid='samples' requires the activation samples")
        x = samples.detach().to(torch.float64).flatten()
        if x.numel() > sample_points:
            gen = torch.Generator(device="cpu").manual_seed(0)
            idx = torch.randint(0, x.numel(), (sample_points,), generator=gen)
            x = x[idx]
        return x
    raise ValueError(f"unknown fit_grid: {fit_grid}")


def _gelu_exact(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

GELU_RUNTIME_H = 0.5
GELU_ZMAX_CAP = 1.5e4


def _fit_gate_cheb_neghalf(c, a, xmax, K, target_err=5.0e-4, max_degree=40):
    h = GELU_RUNTIME_H
    lo = -c * xmax * xmax / (2.0 ** K)
    cheb_a = (h * a * xmax) ** 2
    xs = np.linspace(0.0, xmax, 4000)
    t = 1.0 - 2.0 * (h * a * xs) ** 2 / cheb_a
    exact = np.exp(-c * xs * xs)
    coef, degree = None, max_degree
    for degree in range(6, max_degree + 1, 2):
        n = max(8 * (degree + 1), 600)
        k = np.arange(1, n + 1)
        yg = 0.5 * lo * (1.0 - np.cos((2 * k - 1) * np.pi / (2 * n)))
        series = np.polynomial.chebyshev.Chebyshev.fit(yg, np.exp(yg), degree, domain=[lo, 0.0])
        coef = (series.coef / float(series(0.0))).astype(np.float64)
        b1 = np.zeros_like(t); b2 = np.zeros_like(t)                   
        for cc in coef[::-1]:
            b1, b2 = 2.0 * t * b1 - b2 + cc, b1
        y = b1 - t * b2                                                
        gate_exp = y.copy()
        for _ in range(K):
            gate_exp = gate_exp * gate_exp
        err = float(np.max(np.abs(gate_exp - exact)))
        if err < target_err and float(np.max(np.abs(y))) < 1.001:
            break
    return coef.tolist(), int(degree)


class _GELUApprox(nn.Module):
    def __init__(self, use_gate=True):
        super().__init__()
        self.use_gate = use_gate
        self._log_a = nn.Parameter(torch.log(torch.expm1(torch.tensor(6.0, dtype=torch.float64))))
        self._log_b = nn.Parameter(torch.tensor(0.1, dtype=torch.float64))
        self._log_c = nn.Parameter(torch.log(torch.expm1(torch.tensor(0.8, dtype=torch.float64))))

    @property
    def a(self): return F.softplus(self._log_a)
    @property
    def b(self): return torch.sigmoid(self._log_b) if self.use_gate else torch.zeros((), dtype=torch.float64)
    @property
    def c(self): return F.softplus(self._log_c)

    def forward(self, x):
        ax = GELU_RUNTIME_H * self.a * x
        sat = ax / torch.sqrt(1.0 + ax * ax)
        if not self.use_gate:
            return 0.5 * x * (1.0 + sat)        
        gate = 1.0 - self.b * torch.exp(-self.c * x * x)
        return 0.5 * x * (1.0 + sat * gate)

    @classmethod
    def fit(cls, samples, range_q_hi, range_margin, raw_max_safety, fit_grid,
            grid_points, sample_points, max_iter, d_num, d_den,
            exp_iters, newton_iters, gs_iters, gs_init_method,
            use_gate, denom_floor_frac, gate_fhe_k_ceiling):
        abs_s = samples.abs()
        q_hi = float(_safe_quantile(abs_s, range_q_hi))
        raw_max = float(abs_s.max())
        xmax = float(max((1.0 + range_margin) * q_hi, raw_max * raw_max_safety))

        x = _build_fit_grid(xmax, fit_grid, grid_points, sample_points, samples)
        target = _gelu_exact(x)

        model = cls(use_gate=use_gate)
        opt = torch.optim.LBFGS(
            model.parameters(), lr=1e-3, max_iter=max_iter,
            tolerance_grad=1e-12, tolerance_change=1e-14,
            line_search_fn="strong_wolfe",
        )
        best = [None, float("inf")]

        def closure():
            opt.zero_grad()
            residual = model(x) - target
            loss = residual.abs().mean() + 0.3 * residual.abs().max()
            loss.backward()
            v = loss.item()
            if v < best[1]:
                best[0] = {k: t.detach().clone() for k, t in model.state_dict().items()}
                best[1] = v
            return loss

        opt.step(closure)
        if best[0] is not None:
            model.load_state_dict(best[0])

        a = float(model.a)
        b = float(model.b)
        c = float(model.c)
        z_min = 1.0
        z_max = 1.0 + (GELU_RUNTIME_H * a * xmax) ** 2

        if z_max > GELU_ZMAX_CAP:
            a = math.sqrt(GELU_ZMAX_CAP - 1.0) / (GELU_RUNTIME_H * xmax)  
            if use_gate: 
                m2 = cls(use_gate=use_gate)
                with torch.no_grad():
                    m2._log_a.copy_(torch.log(torch.expm1(torch.tensor(a, dtype=torch.float64))))
                m2._log_a.requires_grad_(False)
                opt2 = torch.optim.LBFGS(
                    [m2._log_b, m2._log_c], lr=1e-3, max_iter=max_iter,
                    tolerance_grad=1e-12, tolerance_change=1e-14, line_search_fn="strong_wolfe")
                best2 = [None, float("inf")]

                def closure2():
                    opt2.zero_grad()
                    residual = m2(x) - target
                    loss = residual.abs().mean() + 0.3 * residual.abs().max()
                    loss.backward()
                    v = loss.item()
                    if v < best2[1]:
                        best2[0] = {k: t.detach().clone() for k, t in m2.state_dict().items()}
                        best2[1] = v
                    return loss

                opt2.step(closure2)
                if best2[0] is not None:
                    m2.load_state_dict(best2[0])
                b = float(m2.b)
                c = float(m2.c)
            z_max = 1.0 + (GELU_RUNTIME_H * a * xmax) ** 2  

        _bts_floor, _bts_wall = 0.01, 8.0                   
        inv_out_scale = min(_bts_wall, math.sqrt(_bts_floor * _bts_wall) * (z_max ** 0.25))
        p, q, gs_lo, gs_hi = rational_remez(
            lambda t: inv_out_scale * t ** -0.5, z_min, z_max, d_num, d_den)

        lin_alpha, lin_beta = _GS_INITS[gs_init_method](float(gs_lo), float(gs_hi))


        denom_floor = float(denom_floor_frac) * float(gs_lo)

        gate_delta0 = 2.6                                 
        gate_on = bool(use_gate)
        gate_K = 0
        gate_cheb, gate_cheb_a, gate_cheb_b = [], 0.0, 0.0
        if gate_on:
            natural_K = max(2, int(math.ceil(math.log2(c * raw_max * raw_max / (0.5 * gate_delta0)))))
            gate_K = min(int(gate_fhe_k_ceiling), natural_K)
            gate_cheb, _gate_deg = _fit_gate_cheb_neghalf(c, a, xmax, gate_K)
            gate_cheb_a = (GELU_RUNTIME_H * a * xmax) ** 2 
            gate_cheb_b = 0.0                              

        return {
            "method": "softsign_inv_sqrt",
            "gate": gate_on,                          
            "a": a, "b": b, "c": c,
            "xmax": xmax,
            "z_min": z_min, "z_max": z_max,
            "Ncoeffs": p.tolist(),
            "Dcoeffs": q.tolist(),
            "gs_lo": float(gs_lo),
            "gs_hi": float(gs_hi),
            "lin_alpha": lin_alpha,
            "lin_beta": lin_beta,
            "inv_out_scale": float(inv_out_scale),      
            "denom_floor": denom_floor,
            "exp_iters":    gate_K,                     
            "gate_cheb_coeffs": gate_cheb,              
            "gate_cheb_a":  gate_cheb_a,                
            "gate_cheb_b":  gate_cheb_b,                
            "newton_iters": int(newton_iters),
            "gs_iters":     int(gs_iters),
        }

def _thor_g(x):
    return math.sqrt(2.0 / math.pi) * (x + 0.044715 * x ** 3)


def _thor_wlsq(x, y, deg, w):
    V = np.vander(x, deg + 1, increasing=True)
    coef, *_ = np.linalg.lstsq(V * w[:, None], y * w, rcond=None)
    return coef


def _fit_thor_composite(S, d1, d2):
    from scipy.special import erf
    k = np.arange(1, 4001)
    cheb = np.cos((2 * k - 1) * np.pi / (2 * 4000))
    t = np.sort(np.concatenate([cheb, np.linspace(-1.0, 1.0, 4000)]))
    x = S * t
    gt = _thor_g(x)
    T_tgt = np.tanh(gt) / 2.0
    ref = 0.5 * x * (1.0 + erf(x / math.sqrt(2.0)))
    wgt = 1.0 + 6.0 * np.exp(-0.5 * (gt / 2.5) ** 2)          
    smax_g = np.abs(gt).max() ** (1.0 / 3.0)
    squashes = [
        np.sign(gt) * np.abs(gt) ** (1.0 / 3.0) / smax_g,    
        gt / np.abs(gt).max(),                               
        t,                                                   
    ]
    best = None
    for s_mono in squashes:
        for gamma in np.linspace(1.0, 14.0, 27):
            p1 = _thor_wlsq(t, np.tanh(gamma * s_mono), d1, wgt)
            y1 = np.polynomial.polynomial.polyval(t, p1)
            p2 = _thor_wlsq(y1, T_tgt, d2, wgt)
            comp = np.polynomial.polynomial.polyval(y1, p2)
            err = np.abs(x * (comp + 0.5) - ref)
            e = float(err.max())
            if best is None or e < best[0]:
                best = (e, float(np.median(err)), p1, p2)
    e_max, e_med, p1, p2 = best
    return p1, p2, e_max, e_med


def _fit_thor_gelu(samples, range_q_hi, range_margin, raw_max_safety, d1, d2):
    abs_s = samples.abs()
    q_hi = float(_safe_quantile(abs_s, range_q_hi))
    raw_max = float(abs_s.max())
    S = float(max((1.0 + range_margin) * q_hi, raw_max * raw_max_safety))
    p1, p2, e_max, e_med = _fit_thor_composite(S, int(d1), int(d2))
    print(f"[thor-gelu] S={S:8.3f}  d1={d1} d2={d2}  max|err|={e_max:.3e}  med={e_med:.3e}")
    return {
        "method": "thor_composite",
        "xmax": S,
        "thor_p1": p1.tolist(),      
        "thor_p2": p2.tolist(),      
    }


LN_CENTER_TARGET = 8.0
LN_INV_Y_MAX = 3.0  # vit: 2.5, bert: 2.5 (hardcoded; no config knob)


@torch.no_grad()
def _newton_inv_sqrt(x, y, iters):
    for _ in range(iters):
        y = 0.5 * y * (3.0 - x * y * y)
    return y


@torch.no_grad()
def _goldschmidt_inv(N, D, alpha, beta, iters):
    F = alpha - beta * D
    N_cur = N * F
    D_cur = -D * F
    F = 2.0 + D_cur
    for _ in range(iters - 1):
        N_cur = N_cur * F
        D_cur = D_cur * F
        F = 2.0 + D_cur
    return N_cur


@torch.no_grad()
def _eval_remez_inv_sqrt(z, Ncoeffs, Dcoeffs, lin_alpha, lin_beta, gs_iters):
    p = torch.tensor(list(Ncoeffs), dtype=torch.float64)
    q = torch.tensor(list(Dcoeffs), dtype=torch.float64)
    N = polyval_torch(z, p)
    D = polyval_torch(z, q)
    return _goldschmidt_inv(N, D, lin_alpha, lin_beta, gs_iters)


@torch.no_grad()
def _estimate_gs_iters_remez(z_grid, Dcoeffs, lin_alpha, lin_beta, target, max_iters):
    q = torch.tensor(list(Dcoeffs), dtype=torch.float64)
    D = polyval_torch(z_grid, q)
    inv_D_true = 1.0 / D
    F = lin_alpha - lin_beta * D
    N_cur = F
    D_cur = -D * F
    F = 2.0 + D_cur
    err = (N_cur - inv_D_true).abs() / inv_D_true.abs()
    if float(err.max()) < target:
        return 1
    for it in range(2, max_iters + 1):
        N_cur = N_cur * F
        D_cur = D_cur * F
        F = 2.0 + D_cur
        err = (N_cur - inv_D_true).abs() / inv_D_true.abs()
        if float(err.max()) < target:
            return it
    return max_iters


class _LayerNormRemezApprox:
    @classmethod
    def fit(cls, samples, margin, q_lo, q_hi, eps, d_num, d_den,
            gs_target_err, gs_max_iters, nr_iters,
            z_max_safety, gs_init_method, cx_abs_max, center_target, denom_floor_frac,
            gs_drift_margin=1.0, z_min_safety=1.0,
            pos_stats=None, rescale_target_df=0.35, rescale_basin_safety=0.85,
            gs_iters_fixed=None):
        s = samples.detach().float().flatten().cpu()
        s = s[s > 0]
        z_lo = _safe_quantile(s, q_lo)
        z_hi = _safe_quantile(s, q_hi)
        z_lo = max(z_lo, eps)

        z_min = max(z_lo / z_min_safety, eps)
        z_max = z_hi * (1.0 + margin) * z_max_safety
        if not (z_min < z_max):
            raise RuntimeError(f"z_min={z_min:.3g} >= z_max={z_max:.3g}")

        center_scale = center_target / float(cx_abs_max)
        c2 = center_scale * center_scale
        fit_lo = z_min * c2
        fit_hi = z_max * c2

        inv_out_scale = LN_INV_Y_MAX * math.sqrt(fit_lo)
        p_np, q_np, q_min, q_max = rational_remez(
            lambda t: inv_out_scale * t ** -0.5, fit_lo, fit_hi, d_num, d_den
        )
        p_np = p_np / q_max
        q_np = q_np / q_max

        gs_lo = float(q_min / q_max)
        gs_hi = 1.0

        gs_lo_fit = gs_lo / gs_drift_margin
        lin_alpha, lin_beta = _GS_INITS[gs_init_method](gs_lo_fit, gs_hi)
        budget_alpha, budget_beta = _linear_gs_init(gs_lo, gs_hi)

        fit_grid = torch.linspace(fit_lo, fit_hi, 4096, dtype=torch.float64)
        if gs_iters_fixed is not None:
            gs_iters = int(gs_iters_fixed)
        else:
            gs_iters = _estimate_gs_iters_remez(
                fit_grid, q_np, budget_alpha, budget_beta, gs_target_err, gs_max_iters,
            )

        approx_scaled = _eval_remez_inv_sqrt(
            fit_grid, p_np, q_np, lin_alpha, lin_beta, gs_iters,
        )
        target_scaled = inv_out_scale * fit_grid ** -0.5
        rel_err = ((approx_scaled - target_scaled) / target_scaled).abs()

        post_newton_scaled = _newton_inv_sqrt(
            fit_grid / (inv_out_scale * inv_out_scale), approx_scaled, nr_iters)
        rel_err_nr = ((post_newton_scaled - target_scaled) / target_scaled).abs()

        finite = float(torch.quantile(s, 0.5))
        z0_floor = float(min(max(finite, z_min), z_max)) * c2

        denom_floor = float(denom_floor_frac) * float(gs_lo)

        def _center_sq(var_med, var_max):
            a, b = float(lin_alpha), float(lin_beta)
            disc = a * a - 4.0 * b * float(rescale_target_df)
            if disc <= 0.0 or var_med <= 0.0 or q_np[1] <= 0.0:
                return c2                                
            d_t = (a - math.sqrt(disc)) / (2.0 * b)      
            x_t = (d_t - float(q_np[0])) / float(q_np[1])
            x_t = min(max(x_t, fit_lo), float(rescale_basin_safety) * fit_hi)
            cs2 = x_t / var_med                         
            cs2_cap = float(rescale_basin_safety) * fit_hi / max(var_max, var_med)
            return float(min(cs2, cs2_cap))

        if pos_stats is not None:
            mean = pos_stats["per_pos_mean"]; pmax = pos_stats["per_pos_max"]
            center_scale_sq = [_center_sq(v, pmax[i]) for i, v in enumerate(mean)]
        else:
            center_scale_sq = [c2]

        return {
            "method": "remez",
            "eps": eps,
            "z0": z0_floor,
            "center_scale": float(center_scale),
            "inv_out_scale": float(inv_out_scale),
            "z_min": z_min,
            "z_max": z_max,
            "fit_lo": float(fit_lo),
            "fit_hi": float(fit_hi),
            "Ncoeffs": p_np.tolist(),
            "Dcoeffs": q_np.tolist(),
            "gs_lo": gs_lo,
            "gs_hi": gs_hi,
            "lin_alpha": float(lin_alpha),
            "lin_beta":  float(lin_beta),
            "denom_floor": denom_floor,
            "gs_iters":  int(gs_iters),
            "nr_iters":  int(nr_iters),
            "center_scale_sq": center_scale_sq,
        }

def _fit_exp_poly(delta0, degree):
    half = 0.5 * delta0
    n = max(8 * (degree + 1), 200)
    k = np.arange(1, n + 1)
    x = half * np.cos((2 * k - 1) * np.pi / (2 * n))
    return np.polyfit(x, np.exp(x), degree)[::-1].astype(np.float64)


def _fit_exp_cheb(delta0, degree):
    half = 0.5 * delta0
    n = max(8 * (degree + 1), 200)
    k = np.arange(1, n + 1)
    x = half * np.cos((2 * k - 1) * np.pi / (2 * n))
    series = np.polynomial.chebyshev.Chebyshev.fit(
        x, np.exp(x), degree, domain=[-half, half])
    return (series.coef / float(series(0.0))).astype(np.float64)

@torch.no_grad()
def _phase1_val(rows, valid, mid, scale, coeffs, n_sq):
    y = ((rows - mid) / scale).double()
    val = polyval_torch(y, coeffs)
    for _ in range(n_sq):
        val = val * val
    return val * valid.double()


@torch.no_grad()
def _refine_S_per_step(val, ref_iters, k_eff):
    out = []
    sum_exp = val.sum(dim=-1, keepdim=True)
    inv_sum = sum_exp.clamp(min=1e-30).reciprocal()
    sqrt_k = k_eff.sqrt()
    for _ in range(ref_iters):
        val = (val * inv_sum) ** 2
        sum_exp = val.sum(dim=-1, keepdim=True)
        s = (sum_exp.flatten() * sqrt_k)
        out.append(s[sum_exp.flatten() > 0])
        inv_sum = sum_exp.clamp(min=1e-30).reciprocal()
    return out


@torch.no_grad()
def _refine_S_tail_per_kc(val, ref_iters, k_eff, row_q, C, q):
    kc = (row_q + 1).long()
    out = []
    sum_exp = val.sum(dim=-1, keepdim=True)
    inv_sum = sum_exp.clamp(min=1e-30).reciprocal()
    sqrt_k = k_eff.sqrt()
    for _ in range(ref_iters):
        val = (val * inv_sum) ** 2
        sum_exp = val.sum(dim=-1, keepdim=True)
        s = sum_exp.flatten() * sqrt_k          # [n_rows]
        prof = []
        for k in range(1, C + 1):
            m = kc == k
            prof.append(float(torch.quantile(s[m].double(), q)) if m.any() else float("nan"))
        out.append(prof)
        inv_sum = sum_exp.clamp(min=1e-30).reciprocal()
    return out


def _linear_gs_init(d_min, d_max):
    alpha = 4.0 / (d_min + d_max)
    return alpha, alpha * alpha / 4.0


def _chebyshev_gs_init(d_min, d_max):
    s = d_min + d_max
    beta = 8.0 / (s * s + 4.0 * d_min * d_max)
    return beta * s, beta


_GS_INITS = {"linear": _linear_gs_init, "chebyshev": _chebyshev_gs_init}


@torch.no_grad()
def _estimate_gs_iters(S, d_min, d_max, target, max_iters, method):
    alpha, beta = _GS_INITS[method](d_min, d_max)

    in_band = S[(S >= d_min) & (S <= d_max)]
    S_test = in_band if in_band.numel() > 0 else S
    y = alpha - beta * S_test
    if (1.0 - y * S_test).abs().max().item() < target:
        return 0
    for it in range(1, max_iters + 1):
        y = y * (2.0 - y * S_test)
        if (1.0 - y * S_test).abs().max().item() < target:
            return it

    print(f"[gs] WARN: did not converge to {target:g} in {max_iters} iters "
          f"(range=[{float(d_min):.3g}, {float(d_max):.3g}], "
          f"|S|={int(S.numel())}, in_band={int(in_band.numel())}); "
          f"saturating at {max_iters}")
    return max_iters


def _next_pow2(x):
    """Smallest power of two >= x (>=1). x is a positive float."""
    if x <= 1.0:
        return 1
    return 1 << math.ceil(math.log2(x))


def _derive_per_layer_delta(M, magnitude_cap, target_delta0, delta1_ub, delta2_ub,
                            delta1_min=1, delta1_max=16,
                            delta2_min=2, delta2_max=32):
    # delta2: exp(M/(2*delta2)) <= B  <=>  delta2 >= M / (2*ln B).
    delta2 = _next_pow2(M / (2.0 * math.log(magnitude_cap)))
    delta2 = min(max(delta2, int(delta2_min)), int(delta2_max))
    # delta1: M/(delta1*delta2) <= target_delta0  <=>  delta1 >= M / (target_delta0*delta2).
    delta1 = _next_pow2(M / (target_delta0 * delta2))
    delta1 = min(max(delta1, int(delta1_min)), int(delta1_max))
    return delta1, delta2


class _SoftmaxApprox:
    @classmethod
    def fit(cls, rows, row_q, T, delta1, delta2, degree, delta0_override,
            margin, range_q_lo, range_q_hi, init_q_lo, init_q_hi, gs_init_method,
            d_safety, gs_target_err, gs_max_iters, denom_floor_frac,
            decode_window=None, refine_drift_margin=1.0, sm_kc_anchor=4,
            per_layer_delta=False, target_delta0=None, magnitude_cap=None,
            delta1_min=1, delta1_max=16, delta2_min=2, delta2_max=32,
            gs_iters_fixed=None, sm_kc_tail_margin=0.0):
        win = int(decode_window) if decode_window else 0
        if 0 < win < T:
            keep = row_q < win
            rows = rows[keep]
            row_q = row_q[keep]

        valid = torch.arange(T).unsqueeze(0) <= row_q.unsqueeze(1)  # causal; vit: bidirectional, bert: bidirectional
        flat = rows[valid]
        a_obs = _safe_quantile(flat, range_q_lo)
        b_obs = _safe_quantile(flat, range_q_hi)
        pad = margin * (b_obs - a_obs)
        a, b = a_obs - pad, b_obs + pad
        M, mid = b - a, 0.5 * (a + b)

        if per_layer_delta:
            delta1, delta2 = _derive_per_layer_delta(
                float(M), magnitude_cap, target_delta0, delta1, delta2,
                delta1_min=delta1_min, delta1_max=delta1_max,
                delta2_min=delta2_min, delta2_max=delta2_max)

        n_sq = int(round(math.log2(delta1)))
        ref_iters = int(round(math.log2(delta2)))
        scale = float(delta1 * delta2)
        delta0 = float(delta0_override) if delta0_override is not None else M / scale

        coeffs = torch.tensor(_fit_exp_poly(delta0, degree), dtype=torch.float64)
        cheb_coeffs = _fit_exp_cheb(delta0, degree)
        val1 = _phase1_val(rows, valid, mid, scale, coeffs, n_sq)
        k_eff = (row_q + 1).double()
        S_mean = val1.sum(-1) / k_eff
        S_mean = S_mean[S_mean > 0]

        init_d_min = _safe_quantile(S_mean, init_q_lo) / d_safety
        init_d_max = _safe_quantile(S_mean, init_q_hi) * d_safety

        gs_init_fn = _GS_INITS[gs_init_method]
        init_alpha, init_beta = gs_init_fn(init_d_min, init_d_max)
        if gs_iters_fixed is not None:
            gs_iters_scaled = int(gs_iters_fixed)   # fixed uniform GS init (learnable refine starts here)
        else:
            gs_iters_scaled = _estimate_gs_iters(
                S_mean, init_d_min, init_d_max,
                gs_target_err, gs_max_iters, gs_init_method,
            )

        refine_S = _refine_S_per_step(val1, ref_iters, k_eff)
        refine_alpha, refine_beta = [], []
        refine_d_min_per, refine_d_max_per, per_step_iters = [], [], []
        for s in refine_S:
            d_min_i = _safe_quantile(s, init_q_lo) / d_safety
            d_max_i = _safe_quantile(s, init_q_hi) * d_safety
            d_max_fit = d_max_i * refine_drift_margin
            iters_i = _estimate_gs_iters(
                s, d_min_i, d_max_fit, gs_target_err, gs_max_iters, gs_init_method,
            )
            a_i, b_i = gs_init_fn(d_min_i, d_max_fit)
            refine_alpha.append(a_i)
            refine_beta.append(b_i)
            refine_d_min_per.append(d_min_i)
            refine_d_max_per.append(d_max_fit)
            per_step_iters.append(iters_i)
        finite = [it for it in per_step_iters if it is not None]
        if not per_step_iters:
            gs_iters_refine_scaled = 0
        elif len(finite) < len(per_step_iters):
            gs_iters_refine_scaled = None
        else:
            gs_iters_refine_scaled = max(finite)

        denom_floor = float(denom_floor_frac) * float(init_d_min)

        C = win if win else int(T)
        anchor = max(1, min(int(sm_kc_anchor), C))
        S_kc = _refine_S_tail_per_kc(val1, ref_iters, k_eff, row_q, C, init_q_hi)
        sm_kc_r = []
        for i in range(ref_iters):
            lows = [S_kc[i][k] for k in range(anchor) if S_kc[i][k] == S_kc[i][k] and S_kc[i][k] > 0]
            s_ref = max(lows) if lows else 1.0          
            last = 1.0
            for k in range(C):
                kc = k + 1
                st = S_kc[i][k]
                if kc <= anchor:
                    last = 1.0
                elif st == st and st > 0.0:             
                    st_eff = st * (1.0 + sm_kc_tail_margin)
                    last = min(1.0, (s_ref / st_eff) ** 2)
                sm_kc_r.append(last)

        return {
            "mid_x": mid,
            "input_scale": scale,
            "squeeze_bound": 0.5 * M,
            "clip_lo": a,
            "clip_hi": b,
            "n_squarings": n_sq,
            "refinement_iters": ref_iters,
            "poly_coeffs": coeffs.tolist(),
            "cheb_coeffs": cheb_coeffs.tolist(),  
            "cheb_a": -0.5 * delta0,              
            "cheb_b":  0.5 * delta0,
            "init_alpha": init_alpha,
            "init_beta":  init_beta,
            "refine_alpha": refine_alpha,
            "refine_beta":  refine_beta,
            "denom_floor": denom_floor,
            "gs_iters_scaled": gs_iters_scaled,
            "gs_iters_refine_scaled": gs_iters_refine_scaled,
            "per_step_refine_iters": per_step_iters,
            "sm_kc_r": sm_kc_r,
            "delta0": delta0,           
            "delta1": int(delta1),
            "delta2": int(delta2),
            "decode_window": win,       
            "calib_T": int(T),          
        }

@hydra.main(version_base=None,
            config_path=f"{PROJECT_ROOT}/configs",
            config_name="calibrate")
def main(cfg: DictConfig):
    torch.manual_seed(42)

    model = instantiate(cfg.model.instance)
    if model.config.vocab_size != cfg.model.vocab_size:
        model.resize_token_embeddings(cfg.model.vocab_size)
    if OmegaConf.select(cfg, "trainer.init_from"):     
        model = perform_he_surgery(model, cfg)         
        model.to(cfg.device).eval()
        load_init_checkpoint(model, cfg, cfg.device)   
    model.to(device=cfg.device, dtype=torch.float32).eval()
    get_batch, _ = instantiate(cfg.dataset.get_batch)

    gelu_samples, ln_samples, ln_cx_samples, ln_eps, attn_samples, ln_pos, logits_rows =         collect_samples(
            model, get_batch, cfg.n_calib_batches, cfg.thor_rows_per_batch,
            logit_rows=cfg.cutmax_rows if cfg.cutmax_calibrate else 0,
        )
    if not (gelu_samples or ln_samples or attn_samples):
        raise RuntimeError("no calibration samples collected — check hook coverage")

    sg_kwargs = dict(
        range_q_hi=cfg.softgelu_range_q_hi,
        range_margin=cfg.softgelu_range_margin,
        raw_max_safety=cfg.softgelu_raw_max_safety,
        fit_grid=cfg.softgelu_fit_grid,
        grid_points=cfg.softgelu_fit_grid_points,
        sample_points=cfg.softgelu_fit_sample_points,
        max_iter=cfg.softgelu_fit_max_iter,
        d_num=cfg.softgelu_remez_d_num,
        d_den=cfg.softgelu_remez_d_den,
        exp_iters=cfg.softgelu_exp_iters,
        newton_iters=cfg.softgelu_newton_iters,
        gs_iters=cfg.softgelu_gs_iters,
        gs_init_method=cfg.softgelu_gs_init_method,
        use_gate=cfg.softgelu_use_gate,
        denom_floor_frac=cfg.softgelu_denom_floor_frac,
        gate_fhe_k_ceiling=cfg.softgelu_gate_fhe_k_ceiling,
    )
    if cfg.softgelu_method == "thor_composite":
        def _gelu_fit(s):
            return _fit_thor_gelu(
                s, range_q_hi=cfg.softgelu_range_q_hi,
                range_margin=cfg.softgelu_range_margin,
                raw_max_safety=cfg.softgelu_raw_max_safety,
                d1=cfg.thor_gelu_d1, d2=cfg.thor_gelu_d2)
    elif cfg.softgelu_method == "softsign_inv_sqrt":
        def _gelu_fit(s):
            return _GELUApprox.fit(s, **sg_kwargs)
    else:
        raise ValueError(f"unknown softgelu_method: {cfg.softgelu_method!r}")

    if cfg.softgelu_per_layer:
        softgelu_section = {n: _gelu_fit(s) for n, s in sorted(gelu_samples.items())}
    else:
        all_g = torch.cat([gelu_samples[n] for n in sorted(gelu_samples)])
        softgelu_section = {"global": _gelu_fit(all_g)}

    ln_kwargs = dict(
        margin=cfg.ln_range_margin,
        q_lo=cfg.ln_range_q_lo,
        q_hi=cfg.ln_range_q_hi,
        z_max_safety=cfg.ln_z_max_safety,
        z_min_safety=cfg.ln_z_min_safety,
        d_num=cfg.ln_remez_d_num,
        d_den=cfg.ln_remez_d_den,
        gs_init_method=cfg.ln_gs_init_method,
        gs_target_err=cfg.ln_gs_target_err,
        gs_max_iters=cfg.ln_gs_max_iters,
        gs_iters_fixed=cfg.ln_gs_iters,
        gs_drift_margin=cfg.ln_gs_drift_margin,
        nr_iters=cfg.ln_nr_iters,
        center_target=cfg.ln_center_target,
        denom_floor_frac=cfg.ln_denom_floor_frac,
        rescale_target_df=cfg.ln_rescale_target_df,
        rescale_basin_safety=cfg.ln_rescale_basin_safety,
    )
    ln_cls = _LayerNormRemezApprox

    if cfg.ln_calib_per_layer:
        ln_section = {
            n: ln_cls.fit(s, eps=ln_eps[n], cx_abs_max=ln_cx_samples[n],
                          pos_stats=ln_pos.get(n), **ln_kwargs)
            for n, s in sorted(ln_samples.items())
        }
    else:
        all_ln = torch.cat([ln_samples[n] for n in sorted(ln_samples)])
        eps_used = max(ln_eps.values()) if ln_eps else 1e-5
        cx_used = max(ln_cx_samples.values()) if ln_cx_samples else 1.0
        ln_section = {"global": ln_cls.fit(all_ln, eps=eps_used, cx_abs_max=cx_used, **ln_kwargs)}

    fixed_delta_per_layer = bool(OmegaConf.select(
        cfg, "thor_fixed_delta_per_layer", default=False))
    thor_kwargs = dict(
        delta1=cfg.thor_delta1,
        delta2=cfg.thor_delta2,
        degree=cfg.thor_poly_degree,
        delta0_override=cfg.thor_delta0,
        margin=cfg.thor_range_margin,
        range_q_lo=cfg.thor_range_q_lo,
        range_q_hi=cfg.thor_range_q_hi,
        init_q_lo=cfg.thor_init_q_lo,
        init_q_hi=cfg.thor_init_q_hi,
        gs_init_method=cfg.thor_gs_init_method,
        d_safety=cfg.thor_d_safety,
        gs_target_err=cfg.thor_gs_target_err,
        gs_max_iters=cfg.thor_gs_max_iters,
        gs_iters_fixed=cfg.thor_gs_iters,
        denom_floor_frac=cfg.thor_denom_floor_frac,
        refine_drift_margin=cfg.thor_refine_drift_margin,
        decode_window=cfg.thor_decode_window,
        sm_kc_anchor=cfg.thor_sm_kc_anchor,
        sm_kc_tail_margin=cfg.thor_sm_kc_tail_margin,
        per_layer_delta=cfg.thor_calib_per_layer and not fixed_delta_per_layer,
        target_delta0=cfg.thor_target_delta0,
        magnitude_cap=cfg.thor_magnitude_cap,
        delta1_min=OmegaConf.select(cfg, "thor_delta1_min", default=1),
        delta1_max=OmegaConf.select(cfg, "thor_delta1_max", default=16),
        delta2_min=OmegaConf.select(cfg, "thor_delta2_min", default=2),
        delta2_max=OmegaConf.select(cfg, "thor_delta2_max", default=32),
    )
    if cfg.thor_calib_per_layer:
        softmax_section = {
            n: _SoftmaxApprox.fit(r, rq, T, **thor_kwargs)
            for n, (r, rq, T) in sorted(attn_samples.items())
        }
    else:
        Ts = {T for _, _, T in attn_samples.values()}
        if len(Ts) != 1:
            raise RuntimeError(f"mixed T across attention layers: {Ts}")
        T = Ts.pop()
        rows = torch.cat([r for r, _, _ in attn_samples.values()], dim=0)
        row_q = torch.cat([rq for _, rq, _ in attn_samples.values()], dim=0)
        softmax_section = {"global": _SoftmaxApprox.fit(rows, row_q, T, **thor_kwargs)}

    out_path = Path(
        cfg.calib_out_path
        or f"{PROJECT_ROOT}/configs/model/approximation/"
           f"{cfg.model.approximation.name}/configs.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    calib = json.load(open(out_path)) if out_path.exists() else {}
    mc = model.config
    calib["model"] = {
        "n_layers": mc.num_hidden_layers,
        "n_embd":   mc.hidden_size,
        "n_head":   mc.num_attention_heads,
        "n_inner":  (getattr(mc, "n_inner", None)
                     or getattr(mc, "intermediate_size", None)
                     or 4 * mc.hidden_size),
    }
    calib["softgelu"] = softgelu_section
    calib["norm"] = {k: ({**v, "kind": "layernorm", "precise_var_bts": False}
                         if isinstance(v, dict) else v)
                     for k, v in ln_section.items()}
    calib["softmax"] = softmax_section
    if cfg.cutmax_calibrate:
        np.save(str(out_path) + ".cutmax_rows.npy",
                logits_rows.numpy())   
        calib["cutmax"] = _fit_cutmax(logits_rows, cfg)
    with open(out_path, "w") as f:
        json.dump(calib, f, indent=2)
    print(f"[save] wrote calibration to {out_path}")



if __name__ == "__main__":
    main()
