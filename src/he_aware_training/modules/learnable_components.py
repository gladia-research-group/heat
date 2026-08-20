import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from he_aware_training.modules.mem_eff_ponder_functions import ExactMemEffPonderGoldschmidt, AnalyticMemEffPonderGoldschmidt, ExactMemEffPonderNewton, AnalyticMemEffPonderNewton, AnalyticMemEffPonderNewtonDiv


def logit_sequential_distribution(logits, temp=1.0):
    if temp != 1.0:
        logits = logits / temp
    log_halt = F.logsigmoid(logits)
    log_continue = F.logsigmoid(-logits)
    log_probs = log_halt + torch.cumsum(log_continue, dim=0) - log_continue

    p_vals = torch.exp(log_probs)
    p_last = (1.0 - p_vals.sum()).clamp_min(0.0)

    lambdas = torch.sigmoid(logits)

    return torch.cat([p_vals, p_last.unsqueeze(0)]), lambdas

# PonderNet Section 2.3
class PonderApproximation(nn.Module):
    def __init__(
        self,
        max_iters,
        soft_training=True,
        halt_init_low=0.05,
        halt_init_high=0.95,
        halt_init_at=None,
        gradient_checkpointing=False,
        wall_iter=None,
        inference_mode="threshold",
    ):
        super().__init__()
        self.max_iters = max_iters
        self.min_iter = 0
        self.gradient_checkpointing = gradient_checkpointing
        self.inference_mode = inference_mode

        halt_init_low = math.log(halt_init_low / (1 - halt_init_low))
        halt_init_high = math.log(halt_init_high / (1 - halt_init_high))

        init_logits = torch.ones(max_iters - 1) * halt_init_low
        if init_logits.numel():   # max_iters=1 (count-0 clone): no logits, p_dist degenerates to [1.0]
            if halt_init_at is not None:
                init_logits[halt_init_at] = halt_init_high
            else:
                init_logits[-1] = halt_init_high
        self.halt_logits = nn.Parameter(
            init_logits
        )  # lambdas for the Benoulli distribution.
        self.optimal_idx = halt_init_at if halt_init_at is not None else max_iters - 2

        self.threshold = 0.95

        self.p_dist = None
        self.soft_training = soft_training

        wall_val = -1 if wall_iter is None else int(wall_iter)
        self.register_buffer('_wall_iter_buf', torch.tensor(wall_val, dtype=torch.long))
        self.register_buffer('_decr_patience_count', torch.tensor(0, dtype=torch.long))
        self.register_buffer('_last_decr_step', torch.tensor(-10**9, dtype=torch.long))
        self.register_buffer('_halt_temp', torch.tensor(1.0, dtype=torch.float32))

    @property
    def wall_iter(self):
        v = int(self._wall_iter_buf.item())
        return None if v < 0 else v

    @wall_iter.setter
    def wall_iter(self, value):
        self._wall_iter_buf.fill_(-1 if value is None else int(value))

    @property
    def decr_patience_count(self):
        return int(self._decr_patience_count.item())

    @decr_patience_count.setter
    def decr_patience_count(self, value):
        self._decr_patience_count.fill_(int(value))

    @property
    def last_decr_step(self):
        return int(self._last_decr_step.item())

    @last_decr_step.setter
    def last_decr_step(self, value):
        self._last_decr_step.fill_(int(value))

    @property
    def halt_temp(self):
        return float(self._halt_temp.item())

    @halt_temp.setter
    def halt_temp(self, value):
        self._halt_temp.fill_(float(value))

    def get_eval_iters(self):
        p_dist, _ = self._compute_probabilities()
        if self.inference_mode == "mode":
            idx = int(p_dist.argmax().item())
        elif self.inference_mode == "fixed":
            idx = len(p_dist) - 1
        else:
            idx = (p_dist.cumsum(dim=0) >= self.threshold).long().argmax().item()
        return self.min_iter + idx

    def forward(self, *args, **kwargs):
        if self.gradient_checkpointing and self.training:
            return checkpoint(self._forward, *args, use_reentrant=False, **kwargs)
        return self._forward(*args, **kwargs)

    def _compute_probabilities(self):
        return logit_sequential_distribution(self.halt_logits[self.min_iter:], self.halt_temp)

    def get_sampling_mask(self, batch_size, input_dim_count):
        p_dist, lambdas = self._compute_probabilities()
        self.p_dist = p_dist
        self.lambdas = lambdas

        halt_probs = lambdas.unsqueeze(0).expand(batch_size, -1)
        # TODO: this should be called at train
        if self.training and self.soft_training:
            noise = torch.rand_like(halt_probs)
            stops_hard = (halt_probs > noise).float()

            stops_hat = stops_hard - halt_probs.detach() + halt_probs

            continue_decisions = 1.0 - stops_hat

            ones = torch.ones(batch_size, 1, device=lambdas.device)
            run_probs = torch.cat([ones, continue_decisions], dim=1)

            cum_run_mask = torch.cumprod(run_probs, dim=1)

            stop_last = torch.ones(batch_size, 1, device=lambdas.device)
            stops_hat_full = torch.cat([stops_hat, stop_last], dim=1)

            z = stops_hat_full * cum_run_mask

        else:
            if self.inference_mode == "mode":
                stop_idx = int(p_dist.argmax().item())
            elif self.inference_mode == "fixed":
                stop_idx = len(p_dist) - 1
            else:
                stop_idx = (p_dist.cumsum(dim=0) >= self.threshold).long().argmax().item()
            z = F.one_hot(
                torch.tensor([stop_idx], device=lambdas.device).expand(batch_size),
                num_classes=len(p_dist),  # dynamic: respects active slice in subclasses
            ).float()

        extra_dims = (input_dim_count + 1) - 2
        shape = list(z.shape) + [1] * extra_dims

        return z.view(*shape)

    def _get_mask(self, batch_size, input_ndim):
        """
        - "weighted": use p_dist as soft weights (clean gradient to halt_logits)
        - "ste": stochastic mask with STE gradient
        - "detached": stochastic mask, no gradient through mask
        """
        if self.training and self.soft_training:
            p_dist, _ = self._compute_probabilities()
            self.p_dist = p_dist
            weights = p_dist.unsqueeze(0).expand(batch_size, -1)
            extra_dims = (input_ndim + 1) - 2
            shape = list(weights.shape) + [1] * extra_dims
            return weights.view(*shape)
        else:
            mask = self.get_sampling_mask(batch_size, input_ndim)
            return mask.detach()

class LearnedGoldschmidt(PonderApproximation):
    def __init__(
        self,
        max_iters,
        min_iter=0,
        soft_training=True,
        halt_init_low=0.05,
        halt_init_high=0.95,
        halt_init_at=None,
        gradient_checkpointing=False,
        analytic=False,
        wall_iter=None,
        inference_mode="threshold",
    ):
        super().__init__(
            max_iters,
            soft_training=soft_training,
            halt_init_low=halt_init_low,
            halt_init_high=halt_init_high,
            halt_init_at=halt_init_at,
            gradient_checkpointing=gradient_checkpointing,
            wall_iter=wall_iter,
            inference_mode=inference_mode,
        )
        self.min_iter = min_iter
        self.analytic = analytic

        # these positions won't receive gradients as they don't participate in output weighting
        if min_iter > 0:
            with torch.no_grad():
                self.halt_logits[:min_iter] = -10.0

    def _forward(self, num, den, lin_init, F=None):
        batch_size = num.shape[0]
        alpha, beta = lin_init[0], lin_init[1]
        mask = self._get_mask(batch_size, num.ndim)

        fixed_iters = self.min_iter
        learnable_iters = self.max_iters - self.min_iter - 1

        if self.analytic:
            return AnalyticMemEffPonderGoldschmidt.apply(num, den, alpha, beta, mask, fixed_iters, learnable_iters, F)
        else:
            return ExactMemEffPonderGoldschmidt.apply(num, den, alpha, beta, mask, fixed_iters, learnable_iters, F)

class LearnedNewtonInverseSqrt(PonderApproximation):
    def __init__(
        self,
        max_iters,
        min_iter=0,
        soft_training=True,
        halt_init_low=0.05,
        halt_init_high=0.95,
        halt_init_at=None,
        gradient_checkpointing=False,
        analytic=False,
        wall_iter=None,
        inference_mode="threshold",
    ):
        super().__init__(
            max_iters,
            soft_training=soft_training,
            halt_init_low=halt_init_low,
            halt_init_high=halt_init_high,
            halt_init_at=halt_init_at,
            gradient_checkpointing=gradient_checkpointing,
            wall_iter=wall_iter,
            inference_mode=inference_mode,
        )

        self.min_iter = min_iter
        self.analytic = analytic

        if min_iter > 0:
            with torch.no_grad():
                self.halt_logits[:min_iter] = -10.0

    def _forward(self, x, y_init):
        mask = self._get_mask(x.shape[0], x.ndim)
        learnable_iters = self.max_iters - self.min_iter - 1

        if self.analytic:
            return AnalyticMemEffPonderNewton.apply(x, y_init, mask, learnable_iters)
        else:
            return ExactMemEffPonderNewton.apply(x, y_init, mask, learnable_iters)