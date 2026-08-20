import re
from pathlib import Path
import json

import numpy as np
import torch
import torch.nn as nn
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
from tqdm import tqdm
from transformers import AutoTokenizer

from he_aware_training import PROJECT_ROOT

_ACT_KEYS = ("gelu", "silu", "swish")
_LAYER_RE = re.compile(r"\.(\d+)\.")


def _is_norm(m):
    return isinstance(m, nn.LayerNorm) or m.__class__.__name__.endswith("RMSNorm")


def _is_act(m):
    if isinstance(m, (nn.GELU, nn.SiLU)):
        return True
    return any(k in m.__class__.__name__.lower() for k in _ACT_KEYS)


def _layer_idx(name):
    m = _LAYER_RE.search(name)
    return int(m.group(1)) if m else None


def _norm_arg(m, x):
    xf = x.float()
    if isinstance(m, nn.LayerNorm):
        return xf.var(dim=-1, unbiased=False) + float(m.eps)
    return xf.pow(2).mean(dim=-1) + float(getattr(m, "variance_epsilon", 1e-6))


def _add(store, quantity, idx, t, cap, gen):
    if idx is None:
        return
    t = t.reshape(-1).double().cpu()
    if t.numel() > cap:
        t = t[torch.randint(0, t.numel(), (cap,), generator=gen)]
    store.setdefault(quantity, {}).setdefault(idx, []).append(t)


@torch.no_grad()
def collect(model, batches, n_layers, delta2, q_lo, q_hi, margin, cap, gen):
    store = {}

    def norm_hook(name):
        idx = _layer_idx(name)
        def hook(mod, inp):
            x = inp[0]
            if torch.is_tensor(x):
                _add(store, "norm_input", idx, x, cap, gen)
                _add(store, "norm_var", idx, _norm_arg(mod, x), cap, gen)
        return hook

    def act_hook(name):
        idx = _layer_idx(name)
        def hook(_mod, inp):
            if torch.is_tensor(inp[0]):
                _add(store, "activation_in", idx, inp[0], cap, gen)
        return hook

    handles = []
    for name, m in model.named_modules():
        if _is_norm(m):
            handles.append(m.register_forward_pre_hook(norm_hook(name)))
        elif _is_act(m):
            handles.append(m.register_forward_pre_hook(act_hook(name)))

    orig_softmax = torch.nn.functional.softmax
    call = [0]

    def patched_softmax(x, *a, **k):
        if torch.is_tensor(x) and x.dim() == 4:
            idx = call[0] % n_layers
            call[0] += 1
            valid = x > -1e30
            v = x[valid]
            if v.numel() > 0:
                _add(store, "softmax_input", idx, v, cap, gen)
                vq = v if v.numel() <= (1 << 20) else v[torch.randint(0, v.numel(), (1 << 20,), generator=gen)]
                lo = torch.quantile(vq.float(), q_lo)
                hi = torch.quantile(vq.float(), q_hi)
                pad = margin * (hi - lo)
                clip_lo, clip_hi = float(lo - pad), float(hi + pad)
                mid = 0.5 * (clip_lo + clip_hi)
                xc = x.double().clamp(clip_lo, clip_hi)
                z = torch.exp((xc - mid) / delta2) * valid
                kc = valid.sum(dim=-1).clamp(min=1)
                _add(store, "softmax_denom", idx, z.sum(dim=-1) / kc, cap, gen)
        return orig_softmax(x, *a, **k)

    torch.nn.functional.softmax = patched_softmax
    try:
        for xb in tqdm(batches, desc="forward", unit="batch"):
            model(xb)
    finally:
        torch.nn.functional.softmax = orig_softmax
        for h in handles:
            h.remove()
    return store


def _stats(parts):
    t = torch.cat(parts)
    t = t[torch.isfinite(t)]
    if t.numel() == 0:
        return None
    return {
        "count": int(t.numel()),
        "mean": float(t.mean()),
        "std": float(t.std(unbiased=False)),
        "min": float(t.min()),
        "max": float(t.max()),
        "median": float(t.median()),
    }


def build_batches(cfg, tok, device):
    data = np.memmap(cfg.text_bin, dtype=np.uint16, mode="r")
    n_src = min(cfg.n_src_tokens, len(data))
    text = AutoTokenizer.from_pretrained(cfg.gpt2_tokenizer).decode(
        data[:n_src].astype(np.int64).tolist()
    )
    ids = tok(text, return_tensors="pt").input_ids[0]
    total = cfg.block_size * cfg.batch_size * cfg.n_batches
    if ids.numel() < total:
        raise RuntimeError(f"only {ids.numel()} tokens, need {total}; raise n_src_tokens")
    ids = ids[:total].view(cfg.n_batches * cfg.batch_size, cfg.block_size)
    return [ids[i : i + cfg.batch_size].to(device) for i in range(0, ids.size(0), cfg.batch_size)]


@hydra.main(version_base=None, config_path=f"{PROJECT_ROOT}/configs", config_name="get_ranges")
def main(cfg: DictConfig):
    torch.manual_seed(0)
    gen = torch.Generator().manual_seed(0)

    model = instantiate(cfg.model.instance).to(cfg.device).eval()
    n_layers = model.config.num_hidden_layers
    tok = AutoTokenizer.from_pretrained(cfg.model.name)
    batches = build_batches(cfg, tok, cfg.device)

    store = collect(model, batches, n_layers, cfg.thor_delta2,
                    cfg.thor_range_q_lo, cfg.thor_range_q_hi, cfg.thor_range_margin,
                    cfg.per_call_cap, gen)

    out = {
        "model": cfg.model.name,
        "n_layers": n_layers,
        "norm_type": next((m.__class__.__name__ for _, m in model.named_modules() if _is_norm(m)), None),
        "act_type": next((m.__class__.__name__ for _, m in model.named_modules() if _is_act(m)), None),
    }
    for quantity, per_layer in store.items():
        out[quantity] = {str(idx): _stats(parts) for idx, parts in sorted(per_layer.items())}

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{cfg.model.name.split('/')[-1]}.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    for q in ("softmax_input", "norm_input", "activation_in"):
        s = out.get(q, {})
        for idx in ("0", str(n_layers // 2), str(n_layers - 1)):
            v = s.get(idx)
            if v:
                print(f"[{cfg.model.name}] {q:13s} L{idx:>2s} "
                      f"mean={v['mean']:.4g} std={v['std']:.4g} "
                      f"min={v['min']:.4g} max={v['max']:.4g}")
    print(f"[save] wrote per-layer ranges to {out_path}")


if __name__ == "__main__":
    main()
