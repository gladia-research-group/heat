import numpy as np

np.seterr(over="ignore", invalid="ignore")  
                                            

P_CANDIDATES = [9, 7, 5, 3]
FLOOR = {1: 1.5e-3, 2: 4e-7}     
WALL_X = {1: 30.0, 2: 80.0}      
WALL_Y = {1: 30.0, 2: 80.0}      
CONV = 0.05       
WALL_VEC = 10.0   


def set_envelopes(knobs):
    global FLOOR, WALL_X, WALL_Y, CONV
    FLOOR = {1: knobs["floor1"], 2: knobs["floor2"]}
    WALL_X = {1: knobs["wall_x1"], 2: knobs["wall_x2"]}
    WALL_Y = {1: knobs["wall_y1"], 2: knobs["wall_y2"]}
    CONV = knobs["conv"]


def pick_prescale(lo, hi, floor_safety):
    for iters in (1, 2):
        g_min = hi / WALL_X[iters]                   # ceiling
        g_max = lo / (floor_safety * FLOOR[iters])   # floor
        if g_min <= g_max:
            return float(np.sqrt(g_min * g_max)), iters
    return float(hi / WALL_X[2]), 2


def _chord(lo, hi):
    va, vb = lo ** -0.5, hi ** -0.5
    beta = (va - vb) / (hi - lo)
    alpha = va + beta * lo
    grid = np.geomspace(lo, hi, 2048)
    gamma = (grid ** -0.5 / (alpha - beta * grid)).min()
    return gamma * alpha, gamma * beta


def _cascade(s2, lo, hi, k, passes, g=None, chord=True, rng=None, rel=0.0,
             absf=0.0, wall_x=None, wall_y=None):
    if g is None:
        g = np.sqrt(lo * hi)
    x = s2 / g
    xlo, xhi = lo / g, hi / g
    if isinstance(chord, tuple):
        ca, cb = chord
        y = np.full_like(x, 1.0) if ca == 0.0 else ca - cb * x
    elif chord:
        ca, cb = _chord(xlo, xhi)
        y = ca - cb * x
    else:
        y = np.full_like(x, 1.0 / np.sqrt(xhi))
    vmax = np.abs(y).max()
    ymax_r, xmax_r = np.abs(y), np.abs(x)
    u_prod = np.ones_like(x)
    for j in range(passes):
        for _ in range(k):
            y = y * (3.0 - x * y * y) / 2.0
            vmax = max(vmax, np.abs(y).max())
            ymax_r = np.maximum(ymax_r, np.abs(y))
        if rng is not None:                       # u refresh bts
            y = y + rng.normal(0, rel, y.shape) * np.abs(y) \
                  + rng.normal(0, absf, y.shape)
        u_prod = u_prod * y
        if j + 1 < passes:
            x = x * y * y
            if rng is not None:                   # chain autos
                x = x + rng.normal(0, rel, x.shape) * np.abs(x) \
                      + rng.normal(0, absf, x.shape)
            xmax_r = np.maximum(xmax_r, np.abs(x))
            y = np.ones_like(x)                   # const from-below (top ~1)
    crossed = None
    if wall_x is not None:
        crossed = (ymax_r > wall_y) | (xmax_r > wall_x)
    return u_prod / np.sqrt(g), vmax, crossed


def _calib_passes(s2, lo, hi, k, g, chord, iters):
    truth = 1.0 / np.sqrt(s2)
    for passes in range(1, 8):
        est, vmax, _ = _cascade(s2, lo, hi, k, passes, g, chord)
        if vmax > WALL_Y[iters]:
            return None, vmax
        if np.abs(est / truth - 1.0).max() <= CONV:
            return passes, vmax
    return None, vmax


class _Sim:
    def __init__(self, rows):
        self.vocab = rows.shape[1]
        self.ys = [r.astype(np.float64).copy() for r in rows]
        self.crossed = np.zeros(rows.shape[0], dtype=bool)   # wall-replay flags

    def s2_band(self):
        self.pre, s2s = [], []
        for y in self.ys:
            mu = y.sum() / self.vocab
            d = y - mu
            s2 = (d * d).sum() / self.vocab
            self.pre.append((d, s2))
            s2s.append(s2)
        return min(s2s), max(s2s)

    def step(self, p, c, m, band, k, passes, g=None, chord=True, rng=None,
             rel=0.0, absf=0.0, casc_rel=None, casc_absf=None,
             wall_x=None, wall_y=None):
        lo, hi = band
        s2 = np.array([s2_ for _, s2_ in self.pre])
        inv_sigma, vmax, crossed = _cascade(
            s2, lo, hi, k, passes, g, chord, rng,
            rel if casc_rel is None else casc_rel,
            absf if casc_absf is None else casc_absf,
            wall_x=wall_x, wall_y=wall_y)
        if crossed is not None:
            self.crossed |= crossed
        for i in range(len(self.ys)):
            d, _ = self.pre[i]
            ynew = d * inv_sigma[i] / (c * m) + 1.0 / m
            if wall_x is not None and np.abs(ynew).max() > WALL_VEC:
                self.crossed[i] = True            # shift-point bts input breach
            if rng is not None:                   # shift-point vector bts
                ynew = ynew + rng.normal(0, rel, ynew.shape) * np.abs(ynew) \
                            + rng.normal(0, absf, ynew.shape)
            self.ys[i] = ynew ** p
        return vmax

    def masses(self):
        return np.array([np.sort(y)[-1] / y.sum() for y in self.ys])

    def finalize(self, rng=None, rel=0.0, absf=0.0):
        zs, sums = [], []
        for y in self.ys:
            tot = y.sum()
            if rng is not None:               # sum-lane bts abs noise (the GS
                tot += rng.normal(0.0, absf)  # chain bootstraps S itself)
            sums.append(tot)
            inv = 1.0 / tot
            if rng is not None:
                inv *= 1.0 + rng.normal(0.0, rel)
            z = y * inv
            if rng is not None:
                z = z + rng.normal(0, absf, z.shape)
            zs.append(z)
        return zs, (float(min(sums)), float(max(sums))), sums


def derive(rows, knobs, mass_mask=None):
    sim = _Sim(rows)
    k = knobs["newton_per_pass"]
    bm = knobs["band_margin"]
    sched = []
    for it in range(knobs["t_max"]):
        c = knobs["c0"] if it == 0 else knobs["c_late"]
        lo, hi = sim.s2_band()
        lo, hi = lo / bm, hi * bm
        s2 = np.array([s2_ for _, s2_ in sim.pre])
        g_pre, casc_iters = pick_prescale(lo, hi, knobs["floor_safety"])
        wy = WALL_Y[casc_iters]
        chord_ok = (lo / g_pre) >= 1.0 / (wy * wy)
        passes, vmax = _calib_passes(s2, lo, hi, k, g_pre, chord_ok, casc_iters)
        if passes is None:
            raise RuntimeError(f"cutmax calib iter {it}: cascade won't "
                               f"converge under the wall (vmax={vmax:.2f})")
        probe = [(d / (np.sqrt(s2_) * c)).max() for d, s2_ in sim.pre]
        w_lo, w_hi = min(probe), max(probe)
        masses = sim.masses()
        gate = masses[mass_mask] if mass_mask is not None else masses
        final = gate.min() >= knobs["mass_target"] or it + 1 == knobs["t_max"]
        m_final = round(knobs["margin"] * (1.0 + w_hi), 3)
        m = m_final if final else round(m_final / knobs["shift_max"], 3)
        ratio = (1.0 + w_hi) / (1.0 + w_lo)
        p = P_CANDIDATES[-1]
        for cand in P_CANDIDATES:
            if ratio ** (2 * cand) <= knobs["spread_cap"]:
                p = cand
                break
        ca, cb = (_chord(lo / g_pre, hi / g_pre) if chord_ok
                  else (1.0 / np.sqrt(hi / g_pre), 0.0))
        sim.step(p, c, m, (lo, hi), k, passes, g_pre, chord_ok)
        sched.append({"p": p, "c": c, "m": m, "lo": lo, "hi": hi, "g": g_pre,
                      "passes": passes, "chord_a": float(ca),
                      "chord_b": float(cb), "cascade_iters": casc_iters})
        if final:
            break
    return sched


def validate(rows, sched, k, rel, absf, seed=0, sum_band=None):
    rng = np.random.default_rng(seed)
    sim = _Sim(rows)
    for e in sched:
        sim.s2_band()
        ci = e.get("cascade_iters", 2)
        sim.step(e["p"], e["c"], e["m"], (e["lo"], e["hi"]), k, e["passes"],
                 e.get("g"), (e.get("chord_a", 0.0), e.get("chord_b", 0.0)),
                 rng, rel, absf,
                 casc_rel=FLOOR[ci] / 3.0, casc_absf=FLOOR[ci],
                 wall_x=WALL_X[ci], wall_y=WALL_Y[ci])
    zs, sum_band_obs, sums = sim.finalize(rng, rel, absf)
    fails, top_mass = 0, []
    for i, (r, z) in enumerate(zip(rows, zs)):
        order = np.argsort(r)
        bad = int(np.argmax(z)) != int(order[-1]) or bool(sim.crossed[i])
        if sum_band is not None and not (sum_band[0] <= sums[i] <= sum_band[1]):
            bad = True
        fails += bad
        top_mass.append(float(z[order[-1]]))
    return fails, top_mass, sum_band_obs


def fit_cutmax_section(rows, knobs):
    set_envelopes(knobs)
    gaps = np.sort(rows, axis=1)
    gaps = gaps[:, -1] - gaps[:, -2]
    mass_mask = gaps >= knobs["gap_floor"]
    if not mass_mask.any():
        mass_mask = np.ones(len(rows), dtype=bool)
    sched = derive(rows, knobs, mass_mask=mass_mask)
    k = knobs["newton_per_pass"]

    sim = _Sim(rows[mass_mask])
    for e in sched:
        sim.s2_band()
        sim.step(e["p"], e["c"], e["m"], (e["lo"], e["hi"]), k, e["passes"],
                 e.get("g"), (e.get("chord_a", 0.0), e.get("chord_b", 0.0)))
    _, sum_band, _ = sim.finalize()
    if not (sum_band[0] > 0.0):
        raise RuntimeError(f"cutmax calib: degenerate noise-free sum band "
                           f"{sum_band}")
    sm = knobs["sum_margin"]
    emit_band = (sum_band[0] * (1.0 - sm), sum_band[1] * (1.0 + sm))
    if emit_band[1] / emit_band[0] > 4.0:
        raise RuntimeError(f"cutmax calib: sum band kappa "
                           f"{emit_band[1]/emit_band[0]:.2f} > 4 breaks the "
                           f"geo-mid GS init (needs den*w0 in (0,2))")

    lines = []
    for name, ci in (("ambient1", 1), ("ambient2", 2)):
        fails, mass, _ = validate(rows, sched, k, FLOOR[ci] / 3.0, FLOOR[ci],
                                  sum_band=emit_band)
        lines.append(f"{name}: {len(rows)-fails}/{len(rows)} "
                     f"min_mass={min(mass):.3f}")
    inv1 = sum(e["passes"] * 7.0 * (k / 4.0) * e["cascade_iters"] + 2
               for e in sched) + 3
    print(f"[cutmax] T={len(sched)} passes={sum(e['passes'] for e in sched)} "
          f"casc_iters={[e['cascade_iters'] for e in sched]} "
          f"est_inv@ambient1~{inv1:.0f} (~{inv1*0.093:.1f}s) "
          f"(mass gate on {int(mass_mask.sum())}/{rows.shape[0]} rows, "
          f"gap_floor={knobs['gap_floor']}) | " + " | ".join(lines))

    es = knobs["entry_scale"]
    return {
        "entry_scale": es,
        "newton_per_pass": k,
        "newton_polish": 0,
        "gs_sum_iters": knobs["gs_sum_iters"],
        "sum_lo": emit_band[0],
        "sum_hi": emit_band[1],
        "p":       [e["p"] for e in sched],
        "c":       [e["c"] for e in sched],
        "m":       [e["m"] for e in sched],
        "s2_hi":   [e["g"] * (es * es if i == 0 else 1.0)
                    for i, e in enumerate(sched)],
        "passes":  [e["passes"] for e in sched],
        "ex2":     [0] * len(sched),
        "chord_a": [e["chord_a"] for e in sched],
        "chord_b": [e["chord_b"] for e in sched],
        "cascade_iters": [e["cascade_iters"] for e in sched],
        "oracle_rows": int(rows.shape[0]),
        "oracle_noise": knobs["noise"],
    }
