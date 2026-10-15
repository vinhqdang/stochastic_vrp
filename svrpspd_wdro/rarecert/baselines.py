"""Exact conditional reference and classical rivals for the rare route and hub overflow
probabilities (RARECERT revision round 1, task A; referee items 3.1-3.4).

Conditional on the shared factors z = (W_d, W_p) the customers of a route are independent,
so for a route with stage-load peak F and threshold c

    p(c) = E[ P(F > c | W) ],      W = (W_d, W_p) ~ N(0, I_2),

and for a hub of routes (one common c, factors shared by all routes of the hub)

    P_hub(c) = E[ 1 - prod_r (1 - p_r(c | W)) ].

P(F > c | W) is the discretised DP of quantum.rare (exceed_prob on the conditional
level pmfs of _conditional_pmfs). The factor expectation is a two-dimensional quadrature:

* rule "trapz": product trapezoid on [-6, 9]^2 with n points per axis (weights h phi(z_i),
  half weight at the ends). The integrand is smooth and the tails beyond the box carry
  no weight that matters at the 1e-6 level, because P(F > c | W) is essentially zero for
  W below -6. Nested sub-grids (every second node) give a quadrature error estimate.
* rule "gh": probabilists' Gauss-Hermite with n nodes (dp_check uses n = 40).

Discretisation in the demand levels: exceed_prob works on V levels, unit u = c / (K + 1/2),
K = floor((V - 1) / 1.05 - 1/2). The discrete value p_V is therefore a function of V, and
exact_reference combines the two V values by Richardson extrapolation of order 2. On a six-stop
route the successive differences p(128) - p(256) and p(256) - p(512) have ratio 4.1 at gamma 1e-3
and at gamma 1e-6, which is the order-2 behaviour assumed here. Where V = 128 is not yet in the
asymptotic regime (route of eleven stops, gap of 2 per cent) the extrapolation is not reliable, so
the reference is only as good as its V pair.

Rivals (route level, same interface conventions):
* gl_two_step: Glasserman-Li two-step IS. Factor shift to the factor part of the dominant
  dominating point (minimum energy over all stage candidates, rarecert.smis), then inner
  conditional MC over the idiosyncratic inputs, either crude ("crude") or with the
  idiosyncratic block shifted to the dominant point's idiosyncratic part ("shift").
* cond_mc: conditional Monte Carlo over the factors, each factor draw evaluated by the
  conditional DP; the factor proposal is N(mu, I_2) with mu the dominant factor point
  (shift=True) or N(0, I_2) (shift=False).
* subset_sim: subset simulation (Au and Beck 2001) on the standard normal input space with
  p0 = 0.1, modified Metropolis (component-wise proposals xi_i ~ N(x_i, width^2), accepted
  with the standard normal ratio; the candidate vector is accepted if it stays in the
  subset).

RV conventions: every estimator reports RV = relative variance per unit of its cost
(per model evaluation for the simulator-based estimators, per conditional DP call for
cond_mc), so that the sample count for relative error eps at 95 per cent is
1.96^2 RV / eps^2. rse of an estimate with total cost n is sqrt(RV / n).
"""

from __future__ import annotations

import math
import time

import numpy as np

from quantum.rare import RouteSim, _conditional_pmfs, exceed_prob, gauss_hermite
from rarecert.smis import smis_mixture

TRAPZ_LO = -6.0
TRAPZ_HI = 9.0
DEFAULT_RULE = "trapz"
DEFAULT_N = 161
DEFAULT_V = (256, 512)
RICHARDSON_ORDER = 2.0
GL_INNER = 8          # inner draws per outer factor draw in gl_two_step
SUBSET_P0 = 0.1
SUBSET_WIDTH = 1.0
_TINY = 1e-300


# ============================================================
# Discretisation and factor quadrature
# ============================================================


def dp_levels(c: float, V: int) -> tuple[int, float]:
    """(K, u): threshold level K and unit u = c / (K + 1/2) of the discrete DP at V levels
    (the same convention as quantum.rare.dp_check: F > c iff the level exceeds K)."""
    if not c > 0:
        raise ValueError(f"c must be positive, got {c}")
    V = int(V)
    K = int(math.floor((V - 1) / 1.05 - 0.5))
    if K < 1:
        raise ValueError(f"V={V} too small")
    return K, float(c) / (K + 0.5)


def factor_rule(rule: str, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Nodes z and weights w of the factor quadrature: sum_i w_i f(z_i) approximates
    E[f(W)] for W ~ N(0, 1)."""
    n = int(n)
    if rule == "trapz":
        if n < 5:
            raise ValueError("trapz needs at least 5 nodes")
        z = np.linspace(TRAPZ_LO, TRAPZ_HI, n)
        h = z[1] - z[0]
        w = h * np.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
        w[0] *= 0.5
        w[-1] *= 0.5
        return z, w
    if rule == "gh":
        z, w = gauss_hermite(n)
        return np.asarray(z, dtype=np.float64), np.asarray(w, dtype=np.float64)
    raise ValueError(f"rule must be 'trapz' or 'gh', got {rule!r}")


def _route_pair_matrix(sim: RouteSim, z: np.ndarray, c: float, V: int) -> np.ndarray:
    """P[a, b] = P(F > c | W_d = z_a, W_p = z_b) for one route (discrete DP at V levels)."""
    K, u = dp_levels(c, V)
    PD = _conditional_pmfs(sim, None, z, "d", V, u)
    PP = _conditional_pmfs(sim, None, z, "p", V, u)
    n = z.size
    P = np.empty((n, n), dtype=np.float64)
    for a in range(n):
        da = PD[a]
        for b in range(n):
            P[a, b] = exceed_prob(da, PP[b], K)
    return P


def _model_pair_matrix(sims, z: np.ndarray, c: float, V: int) -> np.ndarray:
    """G[a, b] = P(some route fails | W = (z_a, z_b)) for the routes in sims (independent
    given the factors): 1 - prod_r (1 - P_r). For one route this is the route matrix."""
    if len(sims) == 1:
        return _route_pair_matrix(sims[0], z, c, V)
    logsurv = np.zeros((z.size, z.size))
    for sim in sims:
        P = _route_pair_matrix(sim, z, c, V)
        logsurv += np.log1p(-np.minimum(P, 1.0 - 1e-16))
    return -np.expm1(logsurv)


def _quadrature(G: np.ndarray, z: np.ndarray, w: np.ndarray, rule: str):
    """Fine value and nested coarse value (every second node, trapz only)."""
    p = float(w @ G @ w)
    if rule != "trapz" or z.size % 2 == 0:
        return p, math.nan
    zc, wc = factor_rule("trapz", (z.size + 1) // 2)
    pc = float(wc @ G[::2, ::2] @ wc)
    return p, pc


def _probability(sims, c: float, V: int, rule: str, n: int) -> dict:
    t0 = time.perf_counter()
    z, w = factor_rule(rule, n)
    G = _model_pair_matrix(sims, z, c, V)
    p, p_coarse = _quadrature(G, z, w, rule)
    quad = abs(p - p_coarse) if math.isfinite(p_coarse) else math.nan
    return {"p": p, "p_coarse": p_coarse, "quad_err": quad, "V": int(V), "rule": rule,
            "n": int(n), "dp_calls": len(sims) * z.size ** 2, "wall_s": time.perf_counter() - t0}


# ============================================================
# Exact conditional probabilities
# ============================================================


def exact_route_prob(sim: RouteSim, c: float, V: int = 256, rule: str = DEFAULT_RULE,
                     n: int = DEFAULT_N) -> dict:
    """Exact conditional-DP probability P(F > c) of one route at V levels.

    Returns p (the quadrature value), p_coarse (the nested coarse rule), quad_err =
    |p - p_coarse| (an estimate of the quadrature error; the coarse error is the larger),
    V, rule, n, dp_calls (number of exceed_prob evaluations) and wall_s.
    """
    return _probability([sim], c, V, rule, n)


def exact_hub_prob(hub, c: float, V: int = 256, rule: str = DEFAULT_RULE,
                   n: int = DEFAULT_N) -> dict:
    """Exact conditional-DP probability P_hub(c) = P(some route of the hub overflows) at V
    levels, from the per-route conditional DP on the shared factor grid. Routes whose
    largest reachable peak is at most c (HubSim.reachable) contribute exactly zero and are
    skipped. Same return fields as exact_route_prob; R is the number of routes evaluated."""
    sims = [hub.sims[r] for r in range(hub.R) if hub.reachable(r, c)]
    if not sims:
        return {"p": 0.0, "p_coarse": 0.0, "quad_err": 0.0, "V": int(V), "rule": rule,
                "n": int(n), "dp_calls": 0, "wall_s": 0.0, "R": 0}
    out = _probability(sims, c, V, rule, n)
    out["R"] = len(sims)
    return out


def exact_reference(model, c: float, n: int = 41, rule: str = DEFAULT_RULE,
                    order: float = RICHARDSON_ORDER, V_pair: tuple = DEFAULT_V) -> dict:
    """Reference probability of a route (RouteSim) or a hub (HubSim): the discrete DP at
    V = 256 and V = 512 with quadrature rule `rule` on n nodes per factor axis, combined by
    Richardson extrapolation of order `order` in the level unit (the unit halves when V
    doubles, so the ratio is 2).

    Returns p_ext (the reference), p_lo, p_hi (the two V values), gap = |p_hi - p_lo|,
    quad_err (the fine-grid quadrature estimate at V_hi), err_est = |p_hi - p_ext| +
    quad_err (estimated absolute error of p_ext under the assumed order), and wall_s
    (total), wall_lo, wall_hi, dp_calls.
    """
    if isinstance(model, RouteSim):
        def run(V):
            return exact_route_prob(model, c, V=V, rule=rule, n=n)
    else:
        def run(V):
            return exact_hub_prob(model, c, V=V, rule=rule, n=n)
    lo = run(V_pair[0])
    hi = run(V_pair[1])
    f = 2.0 ** order
    p_ext = (f * hi["p"] - lo["p"]) / (f - 1.0)
    return {
        "p_ext": p_ext, "p_lo": lo["p"], "p_hi": hi["p"],
        "gap": abs(hi["p"] - lo["p"]), "quad_err": hi["quad_err"],
        "err_est": abs(hi["p"] - p_ext) + (hi["quad_err"] if math.isfinite(hi["quad_err"]) else 0.0),
        "wall_s": lo["wall_s"] + hi["wall_s"], "wall_lo": lo["wall_s"], "wall_hi": hi["wall_s"],
        "dp_calls": lo["dp_calls"] + hi["dp_calls"], "n": int(n), "rule": rule,
        "order": float(order), "R": hi.get("R", 1),
    }


# ============================================================
# Dominant factor point
# ============================================================


def dominant_point(sim: RouteSim, c: float):
    """Minimum-energy dominating point of {F > c} over all stage candidates (the first
    kept mode of rarecert.smis.smis_mixture). Returns (x*, E*) with x* of length dim."""
    mix, diag = smis_mixture(sim, c, n_arc=0)
    return np.asarray(mix.thetas[1], dtype=np.float64).copy(), float(diag["E_min"])


# ============================================================
# Classical rivals
# ============================================================


def gl_two_step(sim: RouteSim, c: float, N: int, rng: np.random.Generator, variant: str = "crude",
                x_star=None, mu=None, inner: int = GL_INNER):
    """Glasserman-Li two-step IS for P(F > c).

    Step 1: the factors W are drawn from N(mu, I_2) with mu = factor part of x* (the
    dominant dominating point, the Glasserman-Li choice) unless mu is given explicitly, with
    likelihood ratio exp(-W.mu + |mu|^2 / 2).
    Step 2: for each factor draw, `inner` idiosyncratic draws E. variant "crude": E ~ N(0, I);
    variant "shift": E ~ N(e*, I) with e* the idiosyncratic part of x* and the likelihood
    ratio exp(-E.e* + |e*|^2 / 2). The outer estimate is the mean over the inner draws
    of the indicator (times the inner ratio), times the factor ratio.

    N is the total number of simulator evaluations (N // inner outer draws).
    Returns (p, RV, rse): RV is the relative variance per model evaluation
    (RV_outer * inner) and rse = sqrt(RV / n_evals) with n_evals = (N // inner) * inner.
    """
    if variant not in ("crude", "shift"):
        raise ValueError(f"variant must be 'crude' or 'shift', got {variant!r}")
    inner = int(inner)
    n_out = int(N) // inner
    if n_out < 2:
        raise ValueError("N must allow at least two outer draws")
    if x_star is None:
        x_star, _ = dominant_point(sim, c)
    x_star = np.asarray(x_star, dtype=np.float64)
    mu = x_star[:2] if mu is None else np.asarray(mu, dtype=np.float64).ravel()
    if mu.shape != (2,):
        raise ValueError("mu must have length 2 (the factor shift)")
    L2 = 2 * sim.L
    if variant == "shift":
        e_star = x_star[2:2 + L2]
    else:
        e_star = np.zeros(L2)
    W = rng.standard_normal((n_out, 2)) + mu
    logw = -(W @ mu) + 0.5 * float(mu @ mu)
    E = rng.standard_normal((n_out, inner, L2)) + e_star[None, None, :]
    x = np.empty((n_out, inner, sim.dim))
    x[:, :, :2] = W[:, None, :]
    x[:, :, 2:] = E
    hit = (sim.peak(x.reshape(-1, sim.dim)) > c).reshape(n_out, inner).astype(np.float64)
    if variant == "shift":
        logv = -(E @ e_star) + 0.5 * float(e_star @ e_star)
        hit = hit * np.exp(logv)
    h = hit.mean(axis=1)
    Y = np.exp(logw) * h
    p = float(Y.mean())
    if p <= 0.0:
        return 0.0, math.nan, math.nan
    RV_out = float(Y.var(ddof=1)) / (p * p)
    RV = RV_out * inner
    rse = math.sqrt(RV_out / n_out)
    return p, RV, rse


COND_PILOT = (-2.0, 5.0, 21)   # pilot grid of cond_shift: lo, hi, nodes per factor axis


def cond_shift(sim: RouteSim, c: float, V: int = 256, pilot=COND_PILOT, mu_dom=None):
    """Factor shift for conditional MC chosen by the relative variance it implies.

    The conditional probability h(W) = P(F > c | W) is evaluated by the discrete DP on a
    pilot grid of pilot[2] nodes per factor axis over [pilot[0], pilot[1]]^2 (one DP call
    per node). For a candidate shift mu the proposal N(mu, I_2) gives the per-draw relative
    variance (sum phi r h^2 - p^2) / p^2 with r = phi / phi(. - mu), evaluated on the grid. The
    candidates are the grid {0, 0.5, ..., 3}^2 and multiples of the factor part of the
    dominant point (mu_dom). Returns (mu, RV_pilot, p_pilot). A zero pilot probability
    returns mu_dom (or zero).

    The dominant point itself is a poor factor shift for the conditional problem (its
    factor part can be far beyond the bulk of phi(W) h(W)); see the summary of the script.
    """
    lo, hi, n = pilot[0], pilot[1], int(pilot[2])
    z = np.linspace(lo, hi, n)
    h = z[1] - z[0]
    if mu_dom is None:
        mu_dom, _ = dominant_point(sim, c)
    mu_dom = np.asarray(mu_dom, dtype=np.float64)[:2]
    G = _route_pair_matrix(sim, z, c, V)
    Z1, Z2 = np.meshgrid(z, z, indexing="ij")
    phi = np.exp(-0.5 * (Z1 * Z1 + Z2 * Z2)) / (2.0 * math.pi)
    p_pilot = float((phi * G).sum() * h * h)
    if not p_pilot > 0.0:
        return mu_dom.copy(), math.nan, 0.0
    cands = [np.array([a, b]) for a in np.arange(0.0, 3.01, 0.5) for b in np.arange(0.0, 3.01, 0.5)]
    cands += [s * mu_dom for s in (0.25, 0.5, 0.75, 1.0)]
    G2 = G * G
    best = (math.inf, None)
    for mu in cands:
        # E_q[(r h)^2] = int phi^2 / q h^2 with q = phi(. - mu) and phi = phi(.)
        expo = -(Z1 * Z1 + Z2 * Z2) + 0.5 * ((Z1 - mu[0]) ** 2 + (Z2 - mu[1]) ** 2)
        m2 = float((np.exp(expo) / (2.0 * math.pi) * G2).sum() * h * h)
        rv = (m2 - p_pilot * p_pilot) / (p_pilot * p_pilot)
        if rv < best[0]:
            best = (rv, mu)
    return np.asarray(best[1], dtype=np.float64), float(best[0]), p_pilot


def cond_mc(sim: RouteSim, c: float, N_factor: int, rng: np.random.Generator, shift: bool = True,
            V: int = 256, mu=None, chunk: int = 500):
    """Conditional Monte Carlo over the factors: W ~ N(mu, I_2), h(W) = P(F > c | W) by the
    conditional DP at V levels, p = mean of exp(-W.mu + |mu|^2/2) h(W).

    mu is the factor shift: if None it is chosen by cond_shift when shift=True and is zero
    when shift=False. The caller may pass mu (and time cond_shift separately).

    Returns (p, RV, rse) with RV the relative variance per conditional DP call (one call per
    factor draw) and rse = sqrt(RV / N_factor).
    """
    N = int(N_factor)
    if N < 2:
        raise ValueError("N_factor must be >= 2")
    if mu is None:
        mu = cond_shift(sim, c, V=V)[0] if shift else np.zeros(2)
    mu = np.asarray(mu, dtype=np.float64).ravel()
    K, u = dp_levels(c, V)
    W = rng.standard_normal((N, 2)) + mu
    logw = -(W @ mu) + 0.5 * float(mu @ mu)
    h = np.empty(N)
    for s in range(0, N, chunk):
        Wc = W[s:s + chunk]
        PD = _conditional_pmfs(sim, None, Wc[:, 0], "d", V, u)
        PP = _conditional_pmfs(sim, None, Wc[:, 1], "p", V, u)
        for i in range(Wc.shape[0]):
            h[s + i] = exceed_prob(PD[i], PP[i], K)
    Y = np.exp(logw) * h
    p = float(Y.mean())
    if p <= 0.0:
        return 0.0, math.nan, math.nan
    RV = float(Y.var(ddof=1)) / (p * p)
    return p, RV, math.sqrt(RV / N)


def _modified_candidate(cur: np.ndarray, rng: np.random.Generator, width: float) -> np.ndarray:
    """Au-Beck modified Metropolis candidate: each component proposes xi_i ~ N(x_i, width^2)
    and is accepted with probability min(1, phi(xi_i) / phi(x_i)); rejected components keep
    their value. Returns the candidate vectors (same shape as cur)."""
    xi = cur + width * rng.standard_normal(cur.shape)
    logr = -0.5 * (xi * xi - cur * cur)
    acc = np.log(rng.random(cur.shape) + _TINY) < np.minimum(logr, 0.0)
    return np.where(acc, xi, cur)


def subset_sim(sim: RouteSim, c: float, N_level: int, rng: np.random.Generator,
               p0: float = SUBSET_P0, width: float = SUBSET_WIDTH, max_levels: int = 60):
    """Subset simulation (Au and Beck 2001) of P(F > c) on the standard normal input space.

    Level 0: N_level standard normal inputs. Each level: the threshold is the (1 - p0)
    quantile of F among the population (or c, if that is reached first); the n_seed = p0 N
    samples above it seed Markov chains of N / n_seed states each (the seeds are the first
    state of their chain) with the modified Metropolis kernel restricted to {F > threshold}.
    When the threshold reaches c, p = p0^k times the fraction of the population above c.

    Returns (p, n_evals): n_evals is the number of simulator evaluations, including the
    level-0 sample and one per candidate state.
    """
    N = int(N_level)
    n_seed = int(round(p0 * N))
    if n_seed < 1 or N % n_seed != 0:
        raise ValueError("N_level must be a multiple of the seed count round(p0 N)")
    steps = N // n_seed
    dim = sim.dim
    X = rng.standard_normal((N, dim))
    F = sim.peak(X)
    evals = N
    prob = 1.0
    for _ in range(max_levels):
        order = np.argsort(-F, kind="stable")
        ell = float(F[order[n_seed - 1]])
        if ell >= c:
            frac = float(np.count_nonzero(F > c)) / N
            return prob * frac, evals
        seeds = order[:n_seed]
        Xn = np.empty_like(X)
        Fn = np.empty_like(F)
        cur = X[seeds].copy()
        curF = F[seeds].copy()
        Xn[:n_seed] = cur
        Fn[:n_seed] = curF
        for s in range(1, steps):
            cand = _modified_candidate(cur, rng, width)
            Fc = sim.peak(cand)
            evals += n_seed
            acc = Fc > ell
            cur[acc] = cand[acc]
            curF[acc] = Fc[acc]
            Xn[s * n_seed:(s + 1) * n_seed] = cur
            Fn[s * n_seed:(s + 1) * n_seed] = curF
        X, F = Xn, Fn
        prob *= p0
    raise RuntimeError(f"subset simulation did not reach c={c} in {max_levels} levels")
