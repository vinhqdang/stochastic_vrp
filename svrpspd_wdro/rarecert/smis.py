"""Closed-form stage-mixture importance sampling (SMIS) for route overflow.

Spec: papers/rarecert/THEORY.md (sections 1, 3, 4). The route simulator, the
mixture proposal and the IS estimator are reused from quantum.rare; this module
only builds the proposal from closed-form quantities. No simulation is run in
smis_mixture: the component means are minimum-norm points of the stage-load
level sets, found with the simulator's exact quantile transform on a few rows.

Input ordering is RouteSim's: x = (W_d, W_p, E_d[0..L-1], E_p[0..L-1]).
Stage k (0..L) has the delivery block {customers j (0-based) >= k} and the
pickup block {j < k}, matching the stage loads of RouteSim.peak(stage=k).
"""

from __future__ import annotations

import math
import time

import numpy as np
from scipy.optimize import minimize

from quantum.rare import Mixture, RouteSim, is_mixture

FD_STEP = 1e-4          # central-difference step for the constraint gradient (x-space)
RAY_HI = 50.0           # initial upper end of the ray bisection (doubled if not bracketed)
RAY_STEPS = 60          # bisection halvings on the ray
ARC_MIN_RATIO = 1.0 / 3.0
DEDUP_DIST = 0.25
NEWTON_POLISH = 10


# ============================================================
# Block scales and directions
# ============================================================


def block_scale(theta, rho: float):
    """Largest and second eigenvalue of diag(sqrt th) (rho 11' + (1-rho) I) diag(sqrt th)
    and the positive Perron vector of the largest one.

    Returns (lam1, lam2, v). An empty block returns (0, 0, empty). lam2 is 0 for a
    single customer.
    """
    theta = np.asarray(theta, dtype=np.float64).ravel()
    n = theta.size
    if n == 0:
        return 0.0, 0.0, np.zeros(0)
    if np.any(theta < 0) or not np.all(np.isfinite(theta)):
        raise ValueError("block thetas must be finite and non-negative")
    s = np.sqrt(theta)
    M = rho * np.outer(s, s) + (1.0 - rho) * np.diag(theta)
    w, V = np.linalg.eigh(M)
    lam1 = float(w[-1])
    lam2 = float(w[-2]) if n > 1 else 0.0
    v = V[:, -1].copy()
    if v.sum() < 0:
        v = -v
    return lam1, lam2, v


def _block_index_theta(sim: RouteSim, k: int, block: str):
    """Customer indices (0-based, active only) and thetas of the block of stage k."""
    if block == "d":
        idx = np.array([j for j in range(k, sim.L) if sim.act_d[j]], dtype=int)
        th = sim.dbar[idx] * sim.cv * sim.cv
    elif block == "p":
        idx = np.array([j for j in range(0, k) if sim.act_p[j]], dtype=int)
        th = sim.pbar[idx] * sim.cv * sim.cv
    else:
        raise ValueError(f"block must be 'd' or 'p', got {block!r}")
    return idx, th


def stage_scales(sim: RouteSim) -> dict:
    """Block scales of every stage. lam_d[k] (deliveries j >= k) and lam_p[k]
    (pickups j < k) for k = 0..L, empty blocks 0; lam_star = max over all blocks;
    ratio = min(lam_d[0], lam_p[L]) / max(lam_d[0], lam_p[L]) (nan if both are 0)."""
    L = sim.L
    lam_d = np.zeros(L + 1)
    lam_p = np.zeros(L + 1)
    for k in range(L + 1):
        _, th = _block_index_theta(sim, k, "d")
        lam_d[k] = block_scale(th, sim.rho)[0] if th.size else 0.0
        _, th = _block_index_theta(sim, k, "p")
        lam_p[k] = block_scale(th, sim.rho)[0] if th.size else 0.0
    lam_star = float(max(lam_d.max(), lam_p.max()))
    lo = min(lam_d[0], lam_p[L])
    hi = max(lam_d[0], lam_p[L])
    ratio = float(lo / hi) if hi > 0 else math.nan
    return {"lam_d": lam_d, "lam_p": lam_p, "lam_star": lam_star, "ratio": ratio}


def block_direction(sim: RouteSim, k: int, block: str) -> np.ndarray:
    """Unit input-space direction of the Perron vector of the block of stage k.

    u = A' Theta^{1/2} v / sqrt(lam1) with A the map (W, E) -> latent normal scores:
    W_d (or W_p) component sqrt(rho) sum_i sqrt(th_i) v_i, E_i component
    sqrt(1-rho) sqrt(th_i) v_i, zero elsewhere.
    """
    idx, th = _block_index_theta(sim, k, block)
    if th.size == 0:
        raise ValueError(f"block {block} of stage {k} is empty")
    lam1, _, v = block_scale(th, sim.rho)
    s = np.sqrt(th)
    u = np.zeros(sim.dim)
    sr = math.sqrt(sim.rho)
    s1 = math.sqrt(1.0 - sim.rho)
    if block == "d":
        u[0] = sr * float(s @ v)
        u[2 + idx] = s1 * s * v
    else:
        u[1] = sr * float(s @ v)
        u[2 + sim.L + idx] = s1 * s * v
    return u / math.sqrt(lam1)


# ============================================================
# Dominating points
# ============================================================


class Unreachable(RuntimeError):
    """The stage load never reaches c along the ray (the simulator's quantile transform
    is clipped at Phi^{-1}(1 - 1e-15), so a block alone can be bounded below c)."""


def dominating_point(sim: RouteSim, c: float, k: int, u0, iters: int = 50):
    """Minimum-norm point of {L_k(x) >= c} (stage load k of the exact simulator).

    The ray t u0 is bisected to L_k(t u0) = c (t in [0, 50]; the bracket is doubled
    if the root is not inside it). From there SLSQP minimises |x|^2 / 2 subject to
    the scaled constraint L_k(x)/c - 1 = 0, with a central-difference gradient taken
    from one batched simulator call; a few Newton projections along the gradient
    then remove the residual. Returns (x*, |x*|^2 / 2).
    """
    c = float(c)
    if not c > 0:
        raise ValueError(f"c must be positive, got {c}")
    u0 = np.asarray(u0, dtype=np.float64).ravel()
    if u0.shape != (sim.dim,):
        raise ValueError(f"u0 must have length {sim.dim}, got {u0.shape}")
    nrm = float(np.linalg.norm(u0))
    if nrm == 0.0:
        raise ValueError("u0 must be non-zero")
    u0 = u0 / nrm
    dim = sim.dim

    def resid_rows(X):
        return sim.peak(X, stage=k) / c - 1.0

    def resid(x):
        return float(resid_rows(x[None, :])[0])

    def grad(x):
        E = np.eye(dim) * FD_STEP
        vals = resid_rows(np.vstack([x + E, x - E]))
        return (vals[:dim] - vals[dim:]) / (2.0 * FD_STEP)

    # ray root
    if resid(0.0 * u0) >= 0.0:
        t0 = 0.0
    else:
        lo, hi = 0.0, RAY_HI
        while resid(hi * u0) < 0.0:
            hi *= 2.0
            if hi > 1e9:
                raise Unreachable(f"stage {k} load never reaches c={c} along the ray")
        for _ in range(RAY_STEPS):
            mid = 0.5 * (lo + hi)
            if resid(mid * u0) < 0.0:
                lo = mid
            else:
                hi = mid
        t0 = 0.5 * (lo + hi)
    x0 = t0 * u0

    res = minimize(
        lambda x: 0.5 * float(x @ x),
        x0,
        jac=lambda x: x,
        method="SLSQP",
        constraints=[{"type": "eq", "fun": resid, "jac": grad}],
        options={"maxiter": int(iters), "ftol": 1e-14},
    )
    x = np.asarray(res.x, dtype=np.float64)
    for _ in range(NEWTON_POLISH):
        r = resid(x)
        if abs(r) < 1e-12:
            break
        g = grad(x)
        den = float(g @ g)
        if den <= 0.0:
            break
        x = x - r * g / den
    return x, 0.5 * float(x @ x)


# ============================================================
# Mixture and estimator
# ============================================================


def smis_mixture(sim: RouteSim, c: float, eta: float = 1e-3, n_arc: int = 0,
                 defensive: float = 0.05):
    """Stage-mixture proposal at threshold c, built without simulation.

    Candidates: for every stage k and non-empty block, the dominating point along
    the block's Perron direction; if n_arc > 0 and both blocks of stage k are
    non-empty with min/max block scale >= 1/3, also the dominating points along
    cos(phi) u_d + sin(phi) u_p for n_arc interior angles phi in (0, pi/2).
    Candidates are sorted by energy E = |x|^2/2, duplicates (within 0.25 of a kept
    point) are dropped, and modes with E <= E_min + log(1/eta) are kept with weights
    proportional to exp(-(E - E_min)) scaled to 1 - defensive, plus N(0, I) with
    weight `defensive`. A candidate whose ray never reaches c (Unreachable) is skipped
    and listed in diagnostics["unreachable"].

    Returns (Mixture, diagnostics). The Mixture's stages list holds the stage index
    of each mode, and its q array holds the unnormalised mode masses exp(-(E - E_min)).
    """
    t_start = time.perf_counter()
    if not 0.0 <= defensive < 1.0:
        raise ValueError(f"defensive weight must lie in [0, 1), got {defensive}")
    if not 0.0 < eta < 1.0:
        raise ValueError(f"eta must lie in (0, 1), got {eta}")
    if n_arc < 0:
        raise ValueError("n_arc must be non-negative")
    st = stage_scales(sim)
    cands = []   # (E, x, stage, label)
    unreachable = []
    for k in range(sim.L + 1):
        has_d = _block_index_theta(sim, k, "d")[1].size > 0
        has_p = _block_index_theta(sim, k, "p")[1].size > 0
        for block, present in (("d", has_d), ("p", has_p)):
            if not present:
                continue
            try:
                x, E = dominating_point(sim, c, k, block_direction(sim, k, block))
            except Unreachable:
                unreachable.append((k, block))
                continue
            cands.append((E, x, k, block))
        if n_arc > 0 and has_d and has_p:
            ld, lp = st["lam_d"][k], st["lam_p"][k]
            if min(ld, lp) / max(ld, lp) >= ARC_MIN_RATIO:
                ud = block_direction(sim, k, "d")
                up = block_direction(sim, k, "p")
                for j in range(n_arc):
                    phi = (j + 0.5) * (math.pi / 2.0) / n_arc
                    u = math.cos(phi) * ud + math.sin(phi) * up
                    try:
                        x, E = dominating_point(sim, c, k, u)
                    except Unreachable:
                        unreachable.append((k, f"arc{j}"))
                        continue
                    cands.append((E, x, k, f"arc{j}"))
    cands.sort(key=lambda t: t[0])
    kept = []
    for E, x, k, label in cands:
        if all(np.linalg.norm(x - y[1]) > DEDUP_DIST for y in kept):
            kept.append((E, x, k, label))
    E_min = kept[0][0]
    cut = E_min + math.log(1.0 / eta)
    modes = [m for m in kept if m[0] <= cut]
    E = np.array([m[0] for m in modes])
    mass = np.exp(-(E - E_min))
    a = (1.0 - defensive) * mass / mass.sum()
    thetas = np.vstack([np.zeros(sim.dim)] + [m[1][None, :] for m in modes])
    weights = np.concatenate([[defensive], a])
    mix = Mixture(
        thetas=thetas,
        weights=weights,
        stages=[None] + [m[2] for m in modes],
        q=mass,
        draws=0,
        skipped=[],
    )
    diag = {
        "n_modes": len(modes),
        "n_candidates": len(cands),
        "unreachable": unreachable,
        "E": E.tolist(),
        "labels": [(m[2], m[3]) for m in modes],
        "E_min": float(E_min),
        "lam_star": st["lam_star"],
        "ratio": st["ratio"],
        "build_seconds": time.perf_counter() - t_start,
    }
    return mix, diag


def smis_estimate(sim: RouteSim, c: float, N: int, rng: np.random.Generator, **kw) -> dict:
    """SMIS estimate of P(F > c): build the mixture, then importance sample with N draws.

    kw are passed to smis_mixture. Returns p_hat, RV, rse, ESS, n_modes, build_seconds
    and the diagnostics of the mixture (E, labels, n_candidates, lam_star, ratio).
    """
    mix, diag = smis_mixture(sim, c, **kw)
    p, RV, rse, ess = is_mixture(sim, c, mix, N, rng)
    out = {"p_hat": p, "RV": RV, "rse": rse, "ESS": ess,
           "n_modes": diag["n_modes"], "build_seconds": diag["build_seconds"]}
    out.update({k: v for k, v in diag.items() if k not in out})
    return out
