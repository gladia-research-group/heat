"""Scheduled count shrink for FIXED-count arms (2026-09-07, arm `fixed` + ANNEAL).

The iteration counts of every ponder site walk linearly from the SEED calibration's counts (the config the
model was built from) to a TARGET config's counts over `iters` optimizer steps, then stay pinned. Nothing is
learned about the allocation: the halting logits are overwritten every time a site's count changes
(peaked profile: `halt_init_low` everywhere, `halt_init_high` at the current index), so the executed count
under hard halting / `inference_mode=mode` is the scheduled one, and the checkpoint's modes export to the
target counts. Requires ponder lr 0 (the logits are state, not parameters, while this runs).

Site keys are the config's (`transformer.h.N.ln_{1,2}`, `transformer.ln_f`, `transformer.h.N.attn`); the
solver is the ponder module's own name. Counts move by shifting each site's `optimal_idx` by the difference
(executed count shifts 1:1 with the index in every solver), so no per-solver count convention is needed.
"""
import json
import math
import re

import torch

from he_aware_training.modules.learnable_components import PonderApproximation

_NAME = re.compile(r"^(?P<site>.+)\.(?:inv_sqrt_approx|softmax)\.(?P<solver>ponder_goldschmidt|ponder_newton|"
                   r"ponder_goldschmidt_init|ponder_goldschmidt_refine)$")
_FIELD = {"ponder_goldschmidt": ("norm", "gs_iters"), "ponder_newton": ("norm", "nr_iters"),
          "ponder_goldschmidt_init": ("softmax", "gs_iters_scaled"),
          "ponder_goldschmidt_refine": ("softmax", "gs_iters_refine_scaled")}


def _count(cfg, site, solver):
    section, field = _FIELD[solver]
    return int(cfg[section][site][field])


class CountAnneal:
    def __init__(self, model, seed_calib, target_calib, iters, halt_init_low=0.02, halt_init_high=0.98):
        seed, target = json.load(open(seed_calib)), json.load(open(target_calib))
        self.iters = int(iters)
        self.lo = math.log(halt_init_low / (1 - halt_init_low))
        self.hi = math.log(halt_init_high / (1 - halt_init_high))
        self.sites = []   # (module, name, idx0, delta)
        for name, m in model.named_modules():
            if not isinstance(m, PonderApproximation) or m.halt_logits.numel() == 0:
                continue
            g = _NAME.match(name)
            if g is None:
                raise ValueError(f"[count-anneal] cannot parse ponder site {name}")
            s, t = _count(seed, g["site"], g["solver"]), _count(target, g["site"], g["solver"])
            if t > s:
                raise ValueError(f"[count-anneal] target {t} > seed {s} at {name}: the seed vector cannot represent it")
            if m.halt_logits.numel() != s:
                raise ValueError(f"[count-anneal] {name}: halting vector has {m.halt_logits.numel()} logits, seed count is {s}")
            self.sites.append((m, name, s, s - t))
        self.total_delta = sum(d for *_, d in self.sites)
        self._last = None
        print(f"[count-anneal] {len(self.sites)} sites, total shrink {self.total_delta} iterations over {self.iters} steps "
              f"(seed {seed_calib} -> target {target_calib})")

    def _write(self, m, count, seed_count):
        """Executed/exported count == index of the peak (verified on CPU 2026-09-07: exporter mode_of and the
        hard-halting forward agree); the seed (maximum) count is the profile with no peak (p_last dominates)."""
        with torch.no_grad():
            m.halt_logits.fill_(self.lo)
            if count < seed_count:
                m.halt_logits[count] = self.hi
            if m.min_iter > 0:
                m.halt_logits[:m.min_iter] = -10.0
        m.optimal_idx = count

    def step(self, it):
        frac = min(1.0, max(0.0, it / self.iters)) if self.iters > 0 else 1.0
        shrunk = 0
        for m, name, seed_count, delta in self.sites:
            d = int(round(delta * frac))
            count = seed_count - d
            if m.optimal_idx != count:
                self._write(m, count, seed_count)
            shrunk += d
        if self._last is None or (it % 25 == 0 and shrunk != self._last) or (frac >= 1.0 and self._last != shrunk):
            print(f"[count-anneal] iter {it}: shrunk {shrunk}/{self.total_delta} iterations ({frac*100:.0f}% of schedule)")
        self._last = shrunk
        return shrunk

    def init(self):
        """Write the peaked profile at the seed index on every site (replaces the ramp init)."""
        for m, name, seed_count, _ in self.sites:
            self._write(m, seed_count, seed_count)
        self._last = None
