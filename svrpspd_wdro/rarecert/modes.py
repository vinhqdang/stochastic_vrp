"""RARECERT R2: mode-decomposed importance sampling (MDIS) for a route failure event
that is a union of generic modes g_m(x) > 0 (sim2.RouteDay), plus the generic
cross-entropy baseline, the single-shift baseline and the threshold choice.

Spec: scratchpad r2_spec.md, sections 1 and 2. The mixture likelihood ratio and the
CE level machinery are reused from quantum.rare (_draws, _upper_quantile,
_ce_update); the block directions of the load modes come from rarecert.smis.

Modes are normalised by the positive per-mode scale of RouteDay, so every dominating
point is computed on O(1) constraint values; signs (and so the failure sets) are
unchanged.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from quantum.rare import Mixture, RouteSim, _draws, _ce_update, _upper_quantile, threshold_for
from rarecert.sim2 import RouteDay
from rarecert.smis import block_direction, _block_index_theta

FD_STEP = 1e-4
RAY_HI = 50.0
RAY_STEPS = 60
RAY_MAX = 1e6
NEWTON_POLISH = 10
NEWTON_TOL = 1e-10
FEAS_TOL = 1e-6
DEDUP_DIST = 0.25
MAX_COMP = 40
DEFENSIVE = 0.05
N_BIS_DRAWS = 200_000
BIS_TOL = 0.05
BIS_STEPS = 25
SLACK_HI0 = 600.0
SLACK_HI_MAX = 6000.0
CHUNK = 50_000


# ============================================================
# Dominating points (generic)
# ============================================================


def dominating_point(fun, x0, iters: int = 50):
    """Minimum-norm boundary point of the failure set {fun > 0}, from the ray of x0.

    fun maps an array (n, dim) to (n,) (normalised modes, so O(1) values). x0 is
    scaled to unit norm, the ray is bisected to fun(t u) = 0 (t in [0, 50], doubled
    up to RAY_MAX), then SLSQP minimises |x|^2 / 2 subject to fun(x) = 0 with a
    central-difference gradient, followed by Newton projections onto the level set.
    The ray point is always a candidate, so the returned E never exceeds the ray
    point's energy. Returns (x*, E) with E = |x*|^2 / 2, or None if the ray never
    reaches the boundary or no boundary point within FEAS_TOL is found.
    """
    u = np.asarray(x0, dtype=np.float64).ravel()
    nrm = float(np.linalg.norm(u))
    if nrm == 0.0 or not np.isfinite(nrm):
        raise ValueError("x0 must be finite and non-zero")
    u = u / nrm
    dim = u.size

    def f1(x):
        return float(np.asarray(fun(x[None, :]))[0])

    def grad(x):
        E = np.eye(dim) * FD_STEP
        vals = np.asarray(fun(np.vstack([x + E, x - E])))
        return (vals[:dim] - vals[dim:]) / (2.0 * FD_STEP)

    if f1(np.zeros(dim)) >= 0.0:
        t0 = 0.0
    else:
        lo, hi = 0.0, RAY_HI
        while f1(hi * u) < 0.0:
            hi *= 2.0
            if hi > RAY_MAX:
                return None
        for _ in range(RAY_STEPS):
            mid = 0.5 * (lo + hi)
            if f1(mid * u) < 0.0:
                lo = mid
            else:
                hi = mid
        t0 = 0.5 * (lo + hi)
    ray = t0 * u
    candidates = [ray]

    try:
        res = minimize(
            lambda x: 0.5 * float(x @ x), ray, jac=lambda x: x, method="SLSQP",
            constraints=[{"type": "eq", "fun": f1, "jac": grad}],
            options={"maxiter": int(iters), "ftol": 1e-14},
        )
        candidates.append(np.asarray(res.x, dtype=np.float64))
    except (ValueError, FloatingPointError):
        pass

    best = None
    for x in candidates:
        x = np.array(x, dtype=np.float64)
        for _ in range(NEWTON_POLISH):
            r = f1(x)
            if abs(r) < NEWTON_TOL:
                break
            g = grad(x)
            den = float(g @ g)
            if den <= 0.0 or not np.isfinite(den):
                break
            x = x - r * g / den
        if not np.all(np.isfinite(x)) or abs(f1(x)) > FEAS_TOL:
            continue
        E = 0.5 * float(x @ x)
        if best is None or E < best[1]:
            best = (x, E)
    return best


def _embed_demand(sim: RouteDay, u_demand) -> np.ndarray:
    u = np.zeros(sim.dim)
    u[:sim.n_dem] = u_demand
    return u


def _unit(v):
    v = np.asarray(v, dtype=np.float64)
    n = float(np.linalg.norm(v))
    return v / n if n > 0 and np.isfinite(n) else None


def mode_starts(sim: RouteDay, m: int):
    """(label, direction) starts for mode m, as in the R2 spec.

    Load mode k: the SMIS block directions of stage k (deliveries and pickups, where
    the block is non-empty), embedded in the full input.
    Time modes (late_i and return): the gradient of g_m at 0, the factor V alone, the
    leg noise H_l alone for the legs that enter the mode (l < i for late_i, all legs
    for return), and the demand factors W_d and W_p alone.
    """
    L = sim.L
    out = []
    if sim.is_load(m):
        k = m
        for block in ("d", "p"):
            _, th = _block_index_theta(sim.demand, k, block)
            if th.size:
                out.append((f"smis_{block}", _embed_demand(sim, block_direction(sim.demand, k, block))))
        return out
    g0 = _normalised_gradient_at_zero(sim, m)
    if g0 is not None:
        out.append(("grad0", g0))
    e = np.eye(sim.dim)
    out.append(("V", e[sim.i_V]))
    if m == 2 * L + 1:
        legs = range(0, L + 1)
    else:
        i = m - L
        legs = range(0, i)
    for l in legs:
        out.append((f"H{l}", e[sim.i_H + l]))
    out.append(("Wd", e[0]))
    out.append(("Wp", e[1]))
    return out


def _normalised_gradient_at_zero(sim: RouteDay, m: int):
    dim = sim.dim
    E = np.eye(dim) * FD_STEP
    vals = sim.normalised_modes(np.vstack([E, -E]))[:, m]
    return _unit((vals[:dim] - vals[dim:]) / (2.0 * FD_STEP))


# ============================================================
# Mixture build (no simulation of the failure event)
# ============================================================


def mdis_mixture(sim: RouteDay, idx=None, eta: float = 1e-3, max_comp: int = MAX_COMP,
                 defensive: float = DEFENSIVE):
    """Mixture of dominating points over the modes idx (default all modes).

    Every (mode, start) pair gives one boundary point; points are sorted by energy,
    duplicates within DEDUP_DIST are dropped, points with E <= E_min + log(1/eta)
    are kept (at most max_comp, lowest energy first). Weights are proportional to
    exp(-(E - E_min)) scaled to 1 - defensive, plus N(0, I) with weight `defensive`.
    Returns (Mixture, diagnostics); the Mixture's stages hold the mode index of each
    component, its q array the unnormalised masses.
    """
    t_start = time.perf_counter()
    if not 0.0 <= defensive < 1.0:
        raise ValueError(f"defensive weight must lie in [0, 1), got {defensive}")
    if not 0.0 < eta < 1.0:
        raise ValueError(f"eta must lie in (0, 1), got {eta}")
    modes = list(range(sim.n_modes)) if idx is None else [int(m) for m in idx]
    cands = []
    n_unreach = 0
    for m in modes:
        fun = (lambda X, m=m: sim.normalised_modes(X)[:, m])
        for label, u in mode_starts(sim, m):
            res = dominating_point(fun, u)
            if res is None:
                n_unreach += 1
                continue
            x, E = res
            cands.append((E, x, m, label))
    if not cands:
        raise RuntimeError("no reachable failure point for the requested modes")
    cands.sort(key=lambda t: t[0])
    kept = []
    for E, x, m, label in cands:
        if all(np.linalg.norm(x - y[1]) > DEDUP_DIST for y in kept):
            kept.append((E, x, m, label))
    E_min = kept[0][0]
    cut = E_min + math.log(1.0 / eta)
    comp = [c for c in kept if c[0] <= cut][:max_comp]
    E = np.array([c[0] for c in comp])
    mass = np.exp(-(E - E_min))
    a = (1.0 - defensive) * mass / mass.sum()
    thetas = np.vstack([np.zeros(sim.dim)] + [c[1][None, :] for c in comp])
    weights = np.concatenate([[defensive], a])
    mix = Mixture(thetas=thetas, weights=weights, stages=[None] + [c[2] for c in comp],
                  q=mass, draws=0, skipped=[])
    kinds = [("load" if sim.is_load(c[2]) else "time") for c in comp]
    load_share = float(mass[[k == "load" for k in kinds]].sum() / mass.sum())
    diag = {
        "n_modes": len(comp),
        "n_candidates": len(cands),
        "n_unreachable": n_unreach,
        "E": E.tolist(),
        "labels": [(c[2], c[3]) for c in comp],
        "E_min": float(E_min),
        "load_mass_share": load_share,
        "build_seconds": time.perf_counter() - t_start,
    }
    return mix, diag


# ============================================================
# Estimators
# ============================================================


def is_generic(sim: RouteDay, thetas, log_a, N: int, rng: np.random.Generator, idx=None,
               chunk: int = CHUNK) -> dict:
    """Importance sampling of P(union over idx of {g_m > 0}) with a Gaussian mixture
    proposal (rare._draws). Returns p, RV, rse, ESS and share_load, the fraction of
    failure weight on draws whose largest normalised mode is a load mode."""
    N = int(N)
    if N < 2:
        raise ValueError("N must be >= 2")
    thetas = np.asarray(thetas, dtype=np.float64)
    log_a = np.asarray(log_a, dtype=np.float64)
    cols = np.arange(sim.n_modes) if idx is None else np.asarray(idx, dtype=int)
    load_mask = np.array([sim.is_load(m) for m in cols])
    s1 = s2 = sw = sw2 = wl = 0.0
    done = 0
    while done < N:
        n = min(chunk, N - done)
        x, logw = _draws(thetas, log_a, n, rng)
        w = np.exp(logw)
        g = sim.normalised_modes(x)[:, cols]
        ind = g.max(axis=1) > 0.0
        y = w * ind
        s1 += float(y.sum())
        s2 += float(y @ y)
        sw += float(w.sum())
        sw2 += float(w @ w)
        am = g.argmax(axis=1)
        wl += float((y * load_mask[am]).sum())
        done += n
    p = s1 / N
    if p <= 0.0:
        return {"p": 0.0, "RV": math.nan, "rse": math.nan, "ESS": sw * sw / sw2, "share_load": math.nan}
    var = max((s2 - s1 * s1 / N) / (N - 1), 0.0)
    RV = var / (p * p)
    return {"p": p, "RV": RV, "rse": math.sqrt(RV / N), "ESS": sw * sw / sw2,
            "share_load": wl / s1}


def crude_generic(sim: RouteDay, N: int, rng: np.random.Generator, chunk: int = CHUNK) -> dict:
    """Plain Monte Carlo of the route failure probability on N draws."""
    N = int(N)
    hits = 0
    done = 0
    while done < N:
        n = min(chunk, N - done)
        x = rng.standard_normal((n, sim.dim))
        hits += int(np.count_nonzero(sim.fail(x)))
        done += n
    if hits == 0:
        return {"p": 0.0, "RV": math.nan, "rse": math.nan}
    p = hits / N
    RV = (1.0 - p) / p
    return {"p": p, "RV": RV, "rse": math.sqrt(RV / N)}


def mdis_estimate(sim: RouteDay, N: int, rng: np.random.Generator, idx=None) -> dict:
    """MDIS estimate of the failure probability (idx restricts the failure event and
    the mixture to a subset of modes). Adds the mixture diagnostics."""
    mix, diag = mdis_mixture(sim, idx=idx)
    out = is_generic(sim, mix.thetas, mix.log_a, N, rng, idx=idx)
    out.update({"n_modes": diag["n_modes"], "build_seconds": diag["build_seconds"],
                "load_mass_share": diag["load_mass_share"], "n_unreachable": diag["n_unreachable"]})
    return out


# ============================================================
# Cross-entropy baselines
# ============================================================


def ce_mode(sim: RouteDay, m: int, rng: np.random.Generator, n_pilot: int = 20_000,
            rho_q: float = 0.1, max_iter: int = 60, info: dict | None = None) -> np.ndarray:
    """Multilevel cross-entropy mean shift for {g_m > 0} (the generic analogue of
    quantum.rare.ce_shift, target level 0). Returns the shift; info["draws"] counts
    the pilot draws."""
    dim = sim.dim
    theta = np.zeros(dim)
    g = float(rho_q)
    used = 0
    try:
        for _ in range(max_iter):
            x = rng.standard_normal((n_pilot, dim)) + theta
            w = np.exp(-(x @ theta) + 0.5 * float(theta @ theta))
            S = sim.normalised_modes(x)[:, m]
            used += n_pilot
            level = min(0.0, _upper_quantile(S, w, g))
            theta = _ce_update(x, w, S, level)
            if level >= 0.0:
                return theta
            g *= rho_q
        raise RuntimeError(f"cross-entropy for mode {m} did not reach level 0 in {max_iter} iterations")
    finally:
        if info is not None:
            info["draws"] = used


def ce_generic_mixture(sim: RouteDay, rng: np.random.Generator, n_pilot: int = 20_000,
                       n_q: int = 20_000, keep_frac: float = 1e-3, max_stages: int = 12,
                       defensive: float = DEFENSIVE):
    """Mixture over the modes: one CE shift per mode, mode probabilities by IS with n_q
    draws, components proportional to the mode probabilities (as quantum.rare.ce_mixture).
    Returns (Mixture, info) with info["draws"] the total pilot draws (CE and q estimates)
    and info["skipped"] the modes without a usable shift."""
    dim = sim.dim
    draws = 0
    found = []
    skipped = []
    for m in range(sim.n_modes):
        cinfo: dict = {}
        try:
            th = ce_mode(sim, m, rng, n_pilot=n_pilot, info=cinfo)
        except RuntimeError:
            skipped.append(m)
            draws += cinfo.get("draws", 0)
            continue
        draws += cinfo["draws"]
        qm = is_generic(sim, th[None, :], np.zeros(1), n_q, rng, idx=[m])["p"]
        draws += int(n_q)
        if qm > 0.0:
            found.append((m, th, qm))
        else:
            skipped.append(m)
    if not found:
        raise RuntimeError("no mode of the route reaches its failure set")
    qmax = max(f[2] for f in found)
    kept = sorted((f for f in found if f[2] >= keep_frac * qmax), key=lambda f: -f[2])[:max_stages]
    qs = np.array([f[2] for f in kept])
    a = (1.0 - defensive) * qs / qs.sum()
    thetas = np.vstack([np.zeros(dim)] + [f[1] for f in kept])
    weights = np.concatenate([[defensive], a])
    mix = Mixture(thetas=thetas, weights=weights, stages=[None] + [f[0] for f in kept],
                  q=qs, draws=int(draws), skipped=skipped)
    return mix, {"draws": int(draws), "skipped": skipped, "n_modes": len(kept),
                 "found": len(found)}


# ============================================================
# Thresholds
# ============================================================


def choose_thresholds(dbar, pbar, legs_km, gamma: float, rng: np.random.Generator,
                      cv: float = 0.3, rho: float = 0.6) -> dict:
    """Capacity Qc and slack with P(load failure) = gamma/2 and P(time failure) = gamma/2.

    Qc is rare.threshold_for on the demand-only RouteSim at gamma/2. The slack is found by
    bisection on [0, SLACK_HI0] (doubled up to SLACK_HI_MAX if the upper end is still
    above the target) so that P(any late or return mode) is gamma/2, estimated by MDIS
    restricted to the time modes with N_BIS_DRAWS draws per step, stopping once the
    estimate is within BIS_TOL of the target or after BIS_STEPS steps.
    """
    t0 = time.perf_counter()
    target = 0.5 * float(gamma)
    Qc = threshold_for(RouteSim(dbar, pbar, cv=cv, rho=rho), target, rng)
    base = RouteDay(dbar, pbar, legs_km, cv=cv, rho=rho, Qc=Qc, slack=0.0)
    tidx = base.time_index()
    steps = []

    def p_time(s):
        sim_s = base.with_thresholds(Qc, s)
        mix, _ = mdis_mixture(sim_s, idx=tidx)
        out = is_generic(sim_s, mix.thetas, mix.log_a, N_BIS_DRAWS, rng, idx=tidx)
        steps.append((float(s), out["p"]))
        return out["p"]

    result = {"Qc": float(Qc), "gamma": float(gamma), "target_time": target,
              "hi_expanded": False, "converged": False}
    p0 = p_time(0.0)
    if p0 <= target * (1.0 + BIS_TOL):
        result.update({"slack": 0.0, "p_time": p0, "converged": True})
    else:
        lo, hi = 0.0, SLACK_HI0
        p_hi = p_time(hi)
        while p_hi > target and hi < SLACK_HI_MAX:
            lo = hi
            hi = min(2.0 * hi, SLACK_HI_MAX)
            result["hi_expanded"] = True
            p_hi = p_time(hi)
        mid, p_mid = hi, p_hi
        for _ in range(BIS_STEPS):
            mid = 0.5 * (lo + hi)
            p_mid = p_time(mid)
            if abs(p_mid / target - 1.0) < BIS_TOL:
                result["converged"] = True
                break
            if p_mid > target:
                lo = mid
            else:
                hi = mid
        result.update({"slack": float(mid), "p_time": float(p_mid)})
    result["bisection"] = steps
    result["seconds"] = time.perf_counter() - t0
    return result
