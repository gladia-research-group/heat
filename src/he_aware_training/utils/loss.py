import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from he_aware_training.modules.learnable_components import PonderApproximation


def get_geometric_prior(max_iters, p, device="cuda"):
    """
    Compute a geometric prior distribution over iterations.
    """
    k = torch.arange(0, max_iters, dtype=torch.float32, device=device)

    p = max(p, 1e-6)
    probs = p * ((1 - p) ** k)

    probs = probs / (probs.sum() + 1e-9)

    return probs


def get_negative_binomial_target(max_iters, p, r=2, device="cuda"):
    p = max(p, 1e-6)
    p = min(p, 1.0 - 1e-6)
    r_t = torch.tensor(float(r), device=device)
    k = torch.arange(0, max_iters, dtype=torch.float32, device=device)

    log_binom = torch.lgamma(k + r_t) - torch.lgamma(k + 1.0) - torch.lgamma(r_t)
    log_probs = log_binom + r_t * torch.log(torch.tensor(p, device=device)) + k * torch.log(
        torch.tensor(1.0 - p, device=device)
    )
    probs = torch.exp(log_probs)
    probs = probs / (probs.sum() + 1e-9)

    return probs

class ActivationRegularizer(nn.Module):
    def __init__(
        self,
        model,
        components,
        lambda_reg=0.01,
        start_step=0,
        log_interval=100,
    ):

        super().__init__()
        self.lambda_reg = lambda_reg
        self.start_step = start_step

        self.step_counter: int = 0

        self.penalties = []
        self.hooks = []

        self.log_interval = log_interval
        self.stats: dict[str, float] = {"general_violations": 0.0, "general_total": 0.0}

        print(f"\n[Regularizer] Sleeping until step {start_step}...")
        for name, module in model.named_modules():
            for comp in components:
                if name.endswith(comp.suffix):
                    threshold = comp.threshold
                    if comp.kind == "max_calib":
                        per_layer = self._load_calib_map(comp.threshold, comp.get("field", "xmax"))
                        if per_layer is None or name not in per_layer:
                            continue
                        threshold = comp.get("factor", 0.95) * per_layer[name]
                    print(f"[Regularizer] Hooking {name} as {comp.kind} @ threshold={threshold}")
                    kind = "max" if comp.kind == "max_calib" else comp.kind
                    self.hooks.append(
                        module.register_forward_hook(self._make_hook(kind, threshold))
                    )

    def _load_calib_map(self, path, field):
        import json
        if not isinstance(path, str):
            return None
        cache = getattr(self, "_calib_maps", {})
        if path not in cache:
            try:
                calib = json.load(open(path))
                sections = ([calib["per_layer"]] if "per_layer" in calib
                            else [v for v in calib.values() if isinstance(v, dict)])
                cache[path] = {k: float(v[field]) for sec in sections
                               for k, v in sec.items() if isinstance(v, dict) and field in v}
                if not cache[path]:
                    print(f"[Regularizer] max_calib map has no '{field}' entries ({path}) — component skipped")
            except (OSError, TypeError, KeyError, ValueError) as e:
                print(f"[Regularizer] max_calib map unavailable ({path}: {e}) — component skipped")
                cache[path] = None
            self._calib_maps = cache
        return cache[path]

    def _make_hook(self, kind, threshold):
        def _hook(module, inputs, _):
            if not module.training or self.step_counter < self.start_step:
                return
            x = inputs[0] if isinstance(inputs[0], torch.Tensor) else inputs[0][0]
            if not (isinstance(x, torch.Tensor) and x.requires_grad):
                return
            if kind == "domain":
                stat = (x.float() / (float(module.z_max) * float(threshold))).amax(dim=-1)
                ref = 1.0
            elif kind == "variance":
                stat = x.float().var(dim=-1).clamp_min(0).sqrt()
                ref = float(threshold)
            elif kind == "max":
                stat = x.float().abs().amax(dim=-1)
                ref = threshold
            elif kind == "denom":
                xf = x.float()
                Tq, Tk = xf.shape[-2], xf.shape[-1]
                qi = torch.arange(Tq, device=xf.device).view(-1, 1)
                ki = torch.arange(Tk, device=xf.device).view(1, -1)
                valid = ki <= qi + (Tk - Tq)                       # causal lower-triangle
                lse = torch.logsumexp(xf.masked_fill(~valid, float("-inf")), dim=-1)
                logk = valid.sum(-1).clamp_min(1).float().log()
                stat = lse - logk                                  # log(avg exp per valid key)
                ref = math.log(float(threshold))
            else:
                stat = x.abs()
                ref = threshold
            excess = torch.relu(stat - ref)
            if excess.any():
                self.penalties.append(torch.mean(excess**2))
                if kind == "domain":
                    self.penalties.append(excess.amax() ** 2)
                if self.step_counter % self.log_interval == 0:
                    with torch.no_grad():
                        self.stats["general_violations"] += (stat > ref).sum().item()
                        self.stats["general_total"] += stat.numel()
        return _hook

    def pop_loss(self):
        self.step_counter += 1

        if not self.penalties:
            return torch.tensor(
                0.0, device="cuda" if torch.cuda.is_available() else "cpu"
            )

        # we use a single sum for both regularization terms
        total_reg = torch.stack(self.penalties).sum()
        self.penalties = []
        return self.lambda_reg * total_reg

    def get_stats_and_reset(self):
        """Get accumulated stats and reset counters. Call periodically for logging."""
        stats = {}

        if self.stats["general_total"] > 0:
            stats["squeeze/general_violation_rate"] = (
                self.stats["general_violations"] / self.stats["general_total"]
            )

        for k in self.stats:
            self.stats[k] = 0.0

        return stats

    def close(self):
        for hook in self.hooks:
            hook.remove()


class DepthRegularizer(nn.Module):
    def __init__(
        self,
        model,
        lambda_depth,
        scheduler,
        log_interval=100,
        decr_trigger="hard",
        decr_patience=1,
        decr_min_gap=0,
        decr_wall_floor=0,
        settle_mass=0.0,
        tau=1.0,
    ):
        super().__init__()
        self.lambda_depth = lambda_depth
        self.scheduler = scheduler
        self.log_interval = log_interval
        self.decr_trigger = decr_trigger
        self.decr_patience = int(decr_patience)
        self.decr_min_gap = int(decr_min_gap)
        self.decr_wall_floor = int(decr_wall_floor)
        self.settle_mass = float(settle_mass)
        self.tau = float(tau)

        self.step_counter: int = 0
        self.collected_dists: list = []
        self.hooks = []
        self.ponder_modules: list = []
        self.stats: dict[str, float] = {}

        print("[DepthRegularizer] Registering PonderApproximation modules...")
        for name, module in model.named_modules():
            if isinstance(module, PonderApproximation):
                optimal = module.optimal_idx
                print(f"  Registered: {name} (optimal_idx={optimal})")
                self.ponder_modules.append((name, module))

    def _hook_fn(self, module, inputs, outputs, name=None):
        if not module.training:
            return
        if module.p_dist is not None:
            self.collected_dists.append(
                (module.p_dist, module.min_iter, name, module.threshold)
            )

    def _maybe_decrement_wall(self, module, scheduler_step):
        if module.wall_iter is None or module.wall_iter <= max(module.min_iter, self.decr_wall_floor):
            return

        if self.settle_mass > 0.0 and float(module.p_dist.max()) < self.settle_mass:
            module.decr_patience_count = 0
            return

        if self.decr_trigger == "hard":
            hard_relative = (
                ((module.p_dist.cumsum(dim=0) < module.threshold)
                 * torch.arange(module.p_dist.shape[0], device=module.p_dist.device))
                .long().argmax().item()
            ) + 1
            triggered = hard_relative < module.wall_iter
        elif self.decr_trigger == "mode":
            learned_mode = int(module.p_dist.argmax().item())
            wall_relative = module.wall_iter - module.min_iter
            triggered = learned_mode < wall_relative
        elif self.decr_trigger == "mode_sat":
            learned_mode = int(module.p_dist.argmax().item())
            wall_relative = module.wall_iter - module.min_iter
            triggered = learned_mode <= wall_relative
        else:
            raise ValueError(f"Unknown decr_trigger: {self.decr_trigger}")

        if triggered:
            module.decr_patience_count = module.decr_patience_count + 1
            if (module.decr_patience_count >= self.decr_patience
                    and (scheduler_step - module.last_decr_step) >= self.decr_min_gap):
                module.wall_iter = module.wall_iter - 1
                module.decr_patience_count = 0
                module.last_decr_step = scheduler_step
        else:
            module.decr_patience_count = 0

    def _sharpen(self, probs):
        if self.tau == 1.0:
            return probs
        probs = probs ** self.tau
        return probs / (probs.sum() + 1e-9)

    def _get_target_dist(self, module, scheduler_state, scheduler_step, device):
        self._maybe_decrement_wall(module, scheduler_step)

        wall_iter = module.wall_iter if module.wall_iter is not None else module.min_iter
        active_len = int(module.p_dist.shape[0])

        mode = self.scheduler.mode
        if mode == "negative_binomial":
            p = scheduler_state
            w = max(0, wall_iter - module.min_iter)
            if w == 0:
                return self._sharpen(get_geometric_prior(active_len, p=p, device=device))
            r = 1.0 + (w + 0.5) * p / (1.0 - p)
            return self._sharpen(get_negative_binomial_target(active_len, p=p, r=r, device=device))

        # geometric: forced prefix + decaying tail.
        forced_prefix = max(0, wall_iter - module.min_iter)
        forced_prefix = min(forced_prefix, max(active_len - 1, 0))
        tail_len = active_len - forced_prefix

        if mode == "geometric":
            distrib = get_geometric_prior(tail_len, p=scheduler_state, device=device)
        else:
            raise ValueError(f"Unknown scheduler mode: {mode}")

        if forced_prefix > 0:
            distrib = torch.cat(
                [torch.ones(forced_prefix, device=device) * distrib[0], distrib], dim=0
            )

        return self._sharpen(distrib)

    def pop_loss(self, iter_num=None):
        self.step_counter += 1

        # Collect p_dist directly from ponder modules (hooks are disabled)
        active = [(name, m) for name, m in self.ponder_modules if m.p_dist is not None]

        if not active:
            return torch.tensor(
                0.0, device="cuda" if torch.cuda.is_available() else "cpu"
            )

        scheduler_step = iter_num if iter_num is not None else self.step_counter
        scheduler_state = self.scheduler.get_state(scheduler_step)
        device = active[0][1].p_dist.device
        loss_reg = torch.tensor(0.0, device=device)

        for name, module in active:
            p_dist = module.p_dist
            p_target_dist = self._get_target_dist(module, scheduler_state, scheduler_step, device)
            p = p_dist.clamp_min(1e-9)
            loss_reg += (p * (p.log() - p_target_dist.clamp_min(1e-9).log())).sum()

        loss_reg = loss_reg / len(active)

        if self.step_counter % self.log_interval == 0:
            with torch.no_grad():
                for name, module in active:
                    p_dist = module.p_dist
                    min_iter = module.min_iter
                    iter_indices = torch.arange(
                        p_dist.shape[0], device=device, dtype=torch.float32
                    )
                    soft = (p_dist * iter_indices).sum().item() + min_iter
                    hard = (
                        p_dist.cumsum(dim=0) >= module.threshold
                    ).long().argmax().item() + min_iter
                    mode = int(p_dist.argmax().item()) + min_iter

                    self.stats[f"depth_soft/{name}_soft"] = soft
                    self.stats[f"depth_hard/{name}_hard"] = hard
                    self.stats[f"depth_mode/{name}_mode"] = mode
                    self.stats[f"depth_mass/{name}_mass"] = float(p_dist.max())
                    if module.wall_iter is not None:
                        self.stats[f"depth_wall/{name}_wall_iter"] = module.wall_iter

                self.stats["depth/target_p"] = scheduler_state

        return self.lambda_depth * loss_reg

    def get_stats_and_reset(self):
        stats = dict(self.stats)
        self.stats = {}
        return stats

    def get_eval_stats(self):
        stats = {}
        total_soft = 0.0
        total_hard = 0.0
        total_mode = 0.0
        total_mass = 0.0
        num = 0
        for name, module in self.ponder_modules:
            if module.p_dist is not None:
                p_dist = module.p_dist
                min_iter = module.min_iter
                device = p_dist.device
                iter_indices = torch.arange(
                    p_dist.shape[0], device=device, dtype=torch.float32
                )
                soft = (p_dist * iter_indices).sum().item() + min_iter
                hard = (
                    p_dist.cumsum(dim=0) >= module.threshold
                ).long().argmax().item() + min_iter
                mode = int(p_dist.argmax().item()) + min_iter
                mass = float(p_dist.max())
                stats[f"depth_soft/{name}_soft"] = soft
                stats[f"depth_hard/{name}_hard"] = hard
                stats[f"depth_mode/{name}_mode"] = mode
                stats[f"depth_mass/{name}_mass"] = mass
                if module.wall_iter is not None:
                    stats[f"depth_wall/{name}_wall_iter"] = module.wall_iter
                total_soft += soft
                total_hard += hard
                total_mode += mode
                total_mass += mass
                num += 1
        if num > 0:
            stats["depth/global_avg_soft"] = total_soft / num
            stats["depth/global_avg_hard"] = total_hard / num
            stats["depth/global_avg_mode"] = total_mode / num
            stats["depth/global_avg_mass"] = total_mass / num
        return stats

    def close(self):
        for hook in self.hooks:
            hook.remove()


class CurriculumScheduler:
    def __init__(
        self,
        max_iters,
        ramp_fraction=0.5,
        mode="geometric",
        p_start=0.05,
        p_end=0.5,
        start_step=0,
    ):
        self.max_iters = max_iters
        self.ramp_fraction = ramp_fraction
        self.mode = mode
        self.p_start = p_start
        self.p_end = p_end
        self.start_step = start_step
        print(f"[CurriculumScheduler] {mode} mode: p {p_start} -> {p_end}, start_step={start_step}")

    def get_state(self, step):
        if step < self.start_step:
            return self.p_start
        effective_step = step - self.start_step
        effective_max = self.max_iters - self.start_step
        ramp_end = effective_max * self.ramp_fraction
        progress = min(effective_step / max(ramp_end, 1), 1.0)

        return self.p_start + progress * (self.p_end - self.p_start)
