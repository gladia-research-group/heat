import torch


_QUANTILE_MAX = 1 << 23


def _quantile_input(t: torch.Tensor) -> torch.Tensor:
    flat = t.reshape(-1)
    if flat.numel() > _QUANTILE_MAX:
        gen = torch.Generator(device="cpu").manual_seed(0)
        idx = torch.randint(0, flat.numel(), (_QUANTILE_MAX,), generator=gen)
        flat = flat[idx.to(flat.device)]
    return flat


def _safe_quantile(t: torch.Tensor, q: float) -> float:
    flat = _quantile_input(t)
    return float(torch.quantile(flat, q))


def _safe_quantiles(t: torch.Tensor, qs) -> list:
    flat = _quantile_input(t)
    qs_t = torch.tensor(list(qs), dtype=flat.dtype, device=flat.device)
    return torch.quantile(flat, qs_t).tolist()


def _qs(t: torch.Tensor, qs):
    return _safe_quantiles(t, qs)
