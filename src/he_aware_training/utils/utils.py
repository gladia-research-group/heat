import torch
import torch.nn as nn
import torch.nn.functional as F

_SQUEEZE_STATS = {}


def record_squeeze_stat(tag, excess):
    with torch.no_grad():
        s = _SQUEEZE_STATS.get(tag)
        v = (excess != 0).sum()
        m = excess.abs().max()
        if s is None:
            _SQUEEZE_STATS[tag] = [v, excess.numel(), m]
        else:
            s[0] += v
            s[1] += excess.numel()
            s[2] = torch.maximum(s[2], m)


def pop_squeeze_stats():
    out = {tag: (float(v) / max(n, 1), float(m)) for tag, (v, n, m) in _SQUEEZE_STATS.items()}
    _SQUEEZE_STATS.clear()
    return out


def soft_clamp(x, min_val=None, max_val=None, grad_scale=1.0):
    clamped = torch.clamp(x, min=min_val, max=max_val)
    penalty = 0.0
    if min_val is not None:
        penalty = penalty + F.relu(min_val - x)
    if max_val is not None:
        penalty = penalty + F.relu(x - max_val)
    soft_penalty = grad_scale * penalty
    return clamped - soft_penalty + soft_penalty.detach()


def squeeze(x, min_val=None, max_val=None, strength=1.0, tag=None):
    clamped = torch.clamp(x, min=min_val, max=max_val)
    excess = (x - clamped.detach()).clamp(min=-1e4, max=1e4)
    if tag is not None:
        record_squeeze_stat(tag, excess)
    penalty = 0.5 * strength * excess * excess
    return clamped - penalty + penalty.detach()


def observe_range(x, min_val=None, max_val=None, tag=None):
    if tag is not None:
        clamped = torch.clamp(x, min=min_val, max=max_val)
        record_squeeze_stat(tag, (x - clamped).clamp(min=-1e4, max=1e4))
    return x


def soft_squeeze(x, min_val=None, max_val=None, strength=1.0):
    clamped = torch.clamp(x, min=min_val, max=max_val)
    excess = (x - clamped.detach()).clamp(min=-1e4, max=1e4)  # bounded (see squeeze)
    penalty = 0.5 * strength * excess * excess
    return x - penalty + penalty.detach()
