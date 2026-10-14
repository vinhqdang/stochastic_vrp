"""Rare-event route overflow probabilities for QUASAR direction A (design:
papers/quasar/DESIGN_RARE.md).

Route model: L customers with mean deliveries dbar and pickups pbar, gamma
marginals with coefficient of variation cv, one-factor Gaussian copula per
demand type (rho), deliveries and pickups driven by independent factors. The
simulator is written in standard normal inputs

    x = (W_d, W_p, E_d[0..L-1], E_p[0..L-1])  in R^(2L+2),

    d_i = G_i^{-1}(Phi(sqrt(rho) W_d + sqrt(1-rho) E_d,i)),   likewise p_i,

and the failure event is {F > c} with F the peak load max_k L_k,
L_0 = sum d, L_k = L_{k-1} - d_k + p_k.

Estimators
----------
crude         plain Monte Carlo on x ~ N(0, I).
is_shift      importance sampling by the mean shift x ~ N(theta, I) with
              likelihood ratio w = exp(-x.theta + |theta|^2/2).
ce_shift      multilevel cross-entropy for theta (self-normalised weights); can
              target a single stage load L_k instead of the peak.
ce_mixture    mixture importance sampling over failure modes: one CE shift per
              stage load L_k (k = 0..L), stage probabilities q_k from IS, and the
              proposal sum_k a_k N(theta_k, I) + a_0 N(0, I) (defensive a_0).
is_mixture    importance sampling with a Mixture proposal (likelihood ratio by
              log-sum-exp).
threshold_for capacity c with P(F > c) = gamma, via a CE ladder, then a mixture
              survival estimate in the inversion.
dp_check      exact probability for small routes (L <= 8): Gauss-Hermite over
              the two factors, conditional independent discretised demands
              on a level grid, and a truncated backward (A, G) recursion.

Cost model (95 per cent confidence, relative error eps_rel)
-----------------------------------------------------------
c_crude, c_is  classical sample counts, z^2 (1-p)/(p eps^2) and z^2 RV/eps^2.
aqe_cost       explicit canonical amplitude estimation (cert.h, cert.r_needed),
               ideal (M known) and with the doubling search for M, for a fixed k.
aqe_is_cost    idealised quantum cost on the normalised IS variable Y/p (mean 1,
               variance RV); see its docstring.

Only NumPy, SciPy and quantum.cert are used here; no Qiskit.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy import stats
from scipy.special import ndtr, ndtri

from quantum.cert import h as cert_h, r_needed

Z95 = 1.96
PEAK_CHUNK = 100_000
_U_CLIP = 1e-15
_PMF_TOL = 1e-16


# ============================================================
# Route simulator
# ============================================================


class RouteSim:
    """Gaussian-input simulator of one route's peak load.

    Parameters
    ----------
    dbar, pbar : array_like, shape (L,)
        Mean delivery and pickup per stop, in route order. A zero mean gives a
        deterministic zero demand (no copula dependence, no gamma draw).
    cv : float
        Coefficient of variation of every demand component.
    rho : float
        One-factor copula loading in [0, 1): correlation of the normal scores.
    """

    def __init__(self, dbar, pbar, cv: float = 0.3, rho: float = 0.6):
        dbar = np.asarray(dbar, dtype=np.float64).ravel()
        pbar = np.asarray(pbar, dtype=np.float64).ravel()
        if dbar.size == 0 or dbar.shape != pbar.shape:
            raise ValueError(f"dbar and pbar must be non-empty and equal length, got {dbar.shape}, {pbar.shape}")
        if not (np.all(np.isfinite(dbar)) and np.all(np.isfinite(pbar))):
            raise ValueError("means must be finite")
        if np.any(dbar < 0) or np.any(pbar < 0):
            raise ValueError("means must be non-negative")
        if not cv > 0:
            raise ValueError(f"cv must be positive, got {cv}")
        if not 0.0 <= rho < 1.0:
            raise ValueError(f"rho must lie in [0, 1), got {rho}")
        self.dbar = dbar
        self.pbar = pbar
        self.L = int(dbar.size)
        self.dim = 2 * self.L + 2
        self.cv = float(cv)
        self.rho = float(rho)
        self.shape = 1.0 / (self.cv * self.cv)           # gamma shape k
        self.act_d = dbar > 0
        self.act_p = pbar > 0
        self.scale_d = np.where(self.act_d, dbar * self.cv * self.cv, 1.0)
        self.scale_p = np.where(self.act_p, pbar * self.cv * self.cv, 1.0)
        self._sr = math.sqrt(self.rho)
        self._s1 = math.sqrt(1.0 - self.rho)

    def demands(self, x):
        """Delivery and pickup realisations for inputs x of shape (N, dim).

        Returns (d, p), each of shape (N, L), in the units of dbar and pbar.
        """
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            x = x[None, :]
        if x.ndim != 2 or x.shape[1] != self.dim:
            raise ValueError(f"x must have shape (N, {self.dim}), got {x.shape}")
        L = self.L
        Wd = x[:, 0:1]
        Wp = x[:, 1:2]
        Ed = x[:, 2:2 + L]
        Ep = x[:, 2 + L:2 + 2 * L]
        ud = np.clip(ndtr(self._sr * Wd + self._s1 * Ed), _U_CLIP, 1.0 - _U_CLIP)
        up = np.clip(ndtr(self._sr * Wp + self._s1 * Ep), _U_CLIP, 1.0 - _U_CLIP)
        d = stats.gamma.ppf(ud, self.shape, scale=self.scale_d[None, :])
        p = stats.gamma.ppf(up, self.shape, scale=self.scale_p[None, :])
        d = np.where(self.act_d[None, :], d, 0.0)
        p = np.where(self.act_p[None, :], p, 0.0)
        return d, p

    def peak(self, x, chunk: int = PEAK_CHUNK, stage=None):
        """Peak load F = max_k L_k per row of x, shape (N,). With stage = k the
        stage load L_k (k = 0..L; L_0 = sum of deliveries) is returned instead."""
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            x = x[None, :]
        if stage is not None and not 0 <= int(stage) <= self.L:
            raise ValueError(f"stage must lie in 0..{self.L}, got {stage}")
        out = np.empty(x.shape[0], dtype=np.float64)
        for s in range(0, x.shape[0], chunk):
            d, p = self.demands(x[s:s + chunk])
            loads = _loads_rows(d, p)
            out[s:s + chunk] = loads.max(axis=1) if stage is None else loads[:, int(stage)]
        return out


def _loads_rows(d: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Stage loads per row for demands d, pickups p of shape (N, L): column 0 is
    L_0 = sum d, column k is L_k = L_{k-1} - d_k + p_k (route 0, ..., L-1, the same
    recursion as experiments.route_peak). The peak is the row maximum."""
    N, L = d.shape
    loads = np.empty((N, L + 1), dtype=np.float64)
    run = d.sum(axis=1)
    loads[:, 0] = run
    for k in range(L):
        run = run - d[:, k] + p[:, k]
        loads[:, k + 1] = run
    return loads


# ============================================================
# Crude and importance sampling
# ============================================================


@dataclass
class Mixture:
    """Importance proposal sum_k weights[k] N(thetas[k], I).

    Row 0 is the defensive component N(0, I); rows 1.. are the stage shifts
    (stages[k] is the stage index of row k, None for the defensive row).
    draws counts every pilot draw spent building the proposal (CE ladders and the
    stage probability estimates); skipped lists stages whose CE did not converge.
    """

    thetas: np.ndarray
    weights: np.ndarray
    stages: list
    q: np.ndarray
    draws: int
    skipped: list = field(default_factory=list)

    @property
    def log_a(self) -> np.ndarray:
        return np.log(self.weights)


def crude(sim: RouteSim, c: float, N: int, rng: np.random.Generator,
          chunk: int = PEAK_CHUNK) -> tuple[float, float]:
    """Crude Monte Carlo estimate of P(F > c) and its relative standard error."""
    N = int(N)
    if N < 1:
        raise ValueError("N must be >= 1")
    hits = 0
    done = 0
    while done < N:
        n = min(chunk, N - done)
        x = rng.standard_normal((n, sim.dim))
        hits += int(np.count_nonzero(sim.peak(x) > c))
        done += n
    p = hits / N
    if hits == 0:
        return 0.0, math.inf
    return p, math.sqrt((1.0 - p) / (p * N))


def _draws(thetas: np.ndarray, log_a: np.ndarray, n: int, rng: np.random.Generator):
    """n draws from the mixture sum_k exp(log_a[k]) N(thetas[k], I) and their
    likelihood ratios w = phi(x) / q(x), returned as (x, log w). A single component
    with weight one is the mean shift and gives log w = -x.theta + |theta|^2 / 2."""
    K = thetas.shape[0]
    if K == 1:
        theta = thetas[0]
        x = rng.standard_normal((n, thetas.shape[1])) + theta
        return x, -(x @ theta) + 0.5 * float(theta @ theta)
    weights = np.exp(log_a)
    comp = rng.choice(K, size=n, p=weights / weights.sum())
    x = rng.standard_normal((n, thetas.shape[1])) + thetas[comp]
    sq = (x * x).sum(axis=1)[:, None] - 2.0 * (x @ thetas.T) + (thetas * thetas).sum(axis=1)[None, :]
    A = log_a[None, :] - 0.5 * sq
    m = A.max(axis=1, keepdims=True)
    lse = m[:, 0] + np.log(np.exp(A - m).sum(axis=1))
    return x, -0.5 * (x * x).sum(axis=1) - lse


def _is_core(sim: RouteSim, c: float, thetas: np.ndarray, log_a: np.ndarray, N: int,
             rng: np.random.Generator, chunk: int, stage=None):
    N = int(N)
    if N < 2:
        raise ValueError("N must be >= 2")
    s1 = s2 = sw = sw2 = 0.0
    done = 0
    while done < N:
        n = min(chunk, N - done)
        x, logw = _draws(thetas, log_a, n, rng)
        w = np.exp(logw)
        y = w * (sim.peak(x, stage=stage) > c)
        s1 += float(y.sum())
        s2 += float(y @ y)
        sw += float(w.sum())
        sw2 += float(w @ w)
        done += n
    p = s1 / N
    var = max((s2 - s1 * s1 / N) / (N - 1), 0.0)
    if p <= 0.0:
        return 0.0, math.nan, math.nan, sw * sw / sw2
    RV = var / (p * p)
    return p, RV, math.sqrt(RV / N), sw * sw / sw2


def is_shift(sim: RouteSim, c: float, theta, N: int, rng: np.random.Generator,
             chunk: int = 50_000, stage=None):
    """Importance sampling of P(F > c) with the proposal N(theta, I).

    Returns (p, RV, rse, ESS): p = mean of Y = w 1[F > c]; RV = sample variance
    of Y divided by p^2 (relative variance per draw); rse = sqrt(RV / N) the
    relative standard error of p; ESS = (sum w)^2 / sum w^2 over all N draws.
    With stage = k the event is {L_k > c} instead of {F > c}.
    """
    theta = np.asarray(theta, dtype=np.float64).ravel()
    if theta.shape != (sim.dim,):
        raise ValueError(f"theta must have length {sim.dim}, got {theta.shape}")
    return _is_core(sim, c, theta[None, :], np.zeros(1), N, rng, chunk, stage)


def is_mixture(sim: RouteSim, c: float, mix: Mixture, N: int, rng: np.random.Generator,
               chunk: int = 50_000):
    """Importance sampling of P(F > c) with a Mixture proposal. Same outputs as
    is_shift: (p, RV, rse, ESS)."""
    if mix.thetas.shape[1] != sim.dim:
        raise ValueError(f"mixture dimension {mix.thetas.shape[1]} != {sim.dim}")
    return _is_core(sim, c, mix.thetas, mix.log_a, N, rng, chunk)


def _upper_quantile(F: np.ndarray, w: np.ndarray, g: float) -> float:
    """Smallest value f among F with weighted P(F > f) <= g (self-normalised)."""
    order = np.argsort(F, kind="stable")
    Fs = F[order]
    ws = w[order]
    W = float(ws.sum())
    tail = np.cumsum(ws[::-1])[::-1] - ws          # sum of weights strictly above each index
    k = int(np.nonzero(tail <= g * W)[0][0])
    return float(Fs[k])


def _ce_update(x: np.ndarray, w: np.ndarray, F: np.ndarray, level: float) -> np.ndarray:
    sel = F >= level
    ws = w[sel]
    if ws.size == 0 or ws.sum() <= 0:
        raise RuntimeError("cross-entropy update has no weighted hits")
    return (ws[:, None] * x[sel]).sum(axis=0) / ws.sum()


def ce_shift(sim: RouteSim, c: float, rng: np.random.Generator, n_pilot: int = 20_000,
             rho_q: float = 0.1, max_iter: int = 60, stage=None, info: dict | None = None) -> np.ndarray:
    """Multilevel cross-entropy for the mean shift theta of the Gaussian inputs.

    Tail levels g_j = rho_q^(j+1) (0.1, 0.01, ...) are visited in turn: each
    iteration draws n_pilot points from N(theta, I), sets the level
    c_t = min(c, weighted upper quantile of F at g_j) and moves theta to the
    likelihood-ratio weighted mean of x over {F >= c_t}. The tail level must
    keep tightening: at a fixed level the mean-shift update reproduces the same
    quantile and stalls far below rare thresholds. The last update is made at
    level c and the function returns theta. With stage = k the statistic is the
    stage load L_k. If info is a dict, info["draws"] receives the pilot draw count.
    """
    dim = sim.dim
    theta = np.zeros(dim)
    g = float(rho_q)
    used = 0
    try:
        for _ in range(max_iter):
            x = rng.standard_normal((n_pilot, dim)) + theta
            w = np.exp(-(x @ theta) + 0.5 * float(theta @ theta))
            F = sim.peak(x, stage=stage)
            used += n_pilot
            level = min(c, _upper_quantile(F, w, g))
            theta = _ce_update(x, w, F, level)
            if level >= c:
                return theta
            g *= rho_q
        raise RuntimeError(f"cross-entropy did not reach level c={c} in {max_iter} iterations")
    finally:
        if info is not None:
            info["draws"] = used


def ce_mixture(sim: RouteSim, c: float, rng: np.random.Generator, n_pilot: int = 20_000,
               n_q: int = 20_000, keep_frac: float = 1e-3, max_stages: int = 12,
               defensive: float = 0.05) -> Mixture:
    """Mixture proposal over the failure modes of the route at threshold c.

    For each stage k = 0..L the event {L_k > c} gets its own CE shift theta_k
    (ce_shift with stage=k) and its probability q_k is estimated by IS with n_q
    draws under N(theta_k, I). Stages with q_k >= keep_frac max_k q_k are kept
    (at most max_stages, the largest). The proposal is
        sum_k a_k N(theta_k, I) + a_0 N(0, I),  a_k proportional to q_k,
        a_0 = defensive (the kept stage weights share 1 - a_0).
    Every event {L_k > c} is covered by its own component, so the union {F > c}
    is covered and the likelihood ratio is bounded by 1 / a_0 on the sample.
    """
    L = sim.L
    dim = sim.dim
    if not 0.0 <= defensive < 1.0:
        raise ValueError(f"defensive weight must lie in [0, 1), got {defensive}")
    draws = 0
    found = []
    skipped = []
    for k in range(L + 1):
        info: dict = {}
        try:
            th = ce_shift(sim, c, rng, n_pilot=n_pilot, stage=k, info=info)
        except RuntimeError:
            skipped.append(k)
            draws += info.get("draws", 0)
            continue
        draws += info["draws"]
        qk, _, _, _ = is_shift(sim, c, th, n_q, rng, stage=k)
        draws += int(n_q)
        if qk > 0.0:
            found.append((k, th, qk))
        else:
            skipped.append(k)
    if not found:
        raise RuntimeError(f"no stage of the route reaches c={c}")
    qmax = max(f[2] for f in found)
    kept = sorted((f for f in found if f[2] >= keep_frac * qmax), key=lambda f: -f[2])[:max_stages]
    qs = np.array([f[2] for f in kept])
    a = (1.0 - defensive) * qs / qs.sum()
    thetas = np.vstack([np.zeros(dim)] + [f[1] for f in kept])
    weights = np.concatenate([[defensive], a])
    return Mixture(thetas=thetas, weights=weights, stages=[None] + [f[0] for f in kept],
                   q=qs, draws=int(draws), skipped=skipped)


def _upper_quantile_interp(F: np.ndarray, w: np.ndarray, g: float) -> float:
    """Upper quantile at level g of the plain IS survival (1/N) sum w 1[F > f].

    Self-normalised weights are avoided here: with |theta| of a few units the
    likelihood ratios have enormous variance and dividing by their sum biases
    the tail low. Between consecutive order statistics the survival is interpolated
    log-linearly, so a single heavy weight does not produce a jump of the
    estimated threshold; for dense tails the result agrees with _upper_quantile.
    """
    order = np.argsort(F, kind="stable")
    Fs = F[order]
    ws = w[order]
    N = float(ws.size)
    tail = (np.cumsum(ws[::-1])[::-1] - ws) / N   # unbiased IS survival, not self-normalised
    k = int(np.nonzero(tail <= g)[0][0])
    if k == 0 or tail[k] <= 0.0:
        return float(Fs[k])
    t0 = tail[k - 1]
    t1 = tail[k]
    frac = (math.log(t0) - math.log(g)) / (math.log(t0) - math.log(t1))
    return float(Fs[k - 1] + frac * (Fs[k] - Fs[k - 1]))


def _survival_quantile(sim: RouteSim, thetas: np.ndarray, log_a: np.ndarray, gamma: float,
                       n_final: int, rng: np.random.Generator) -> float:
    """Capacity at level gamma from n_final draws of the proposal (thetas, log_a)."""
    Fs, ws = [], []
    done = 0
    while done < n_final:
        n = min(50_000, n_final - done)
        x, logw = _draws(thetas, log_a, n, rng)
        Fs.append(sim.peak(x))
        ws.append(np.exp(logw))
        done += n
    return _upper_quantile_interp(np.concatenate(Fs), np.concatenate(ws), gamma)


def threshold_for(sim: RouteSim, gamma: float, rng: np.random.Generator,
                  n_step: int = 20_000, n_final: int = 200_000, max_refine: int = 3,
                  refine_tol: float = 1e-3) -> float:
    """Capacity c with P(F > c) = gamma (estimated).

    Ladder of levels 1e-1, 1e-2, ... down to gamma: at each level the weighted
    upper quantile under the current proposal is computed and the proposal is
    updated by cross-entropy toward that level. The ladder gives a starting c
    from the single-shift survival. Then the threshold is refined: at the current
    c a mixture proposal (ce_mixture) is built and its survival estimate is
    inverted at gamma, until c is stable to refine_tol (at most max_refine rounds). The survival
    estimate from n_final draws carries a few tenths of a per cent of quantile
    noise, so refine_tol sits above that noise. The reported p is always the IS
    estimate at the final c, so the threshold error does not enter the ground truth.
    """
    gamma = float(gamma)
    if not 0.0 < gamma < 1.0:
        raise ValueError(f"gamma must lie in (0, 1), got {gamma}")
    ladder = []
    g = 0.1
    while g > gamma * (1 + 1e-12):
        ladder.append(g)
        g *= 0.1
    ladder.append(gamma)
    dim = sim.dim
    theta = np.zeros(dim)
    for g in ladder:
        x = rng.standard_normal((n_step, dim)) + theta
        w = np.exp(-(x @ theta) + 0.5 * float(theta @ theta))
        F = sim.peak(x)
        cg = _upper_quantile(F, w, g)
        theta = _ce_update(x, w, F, cg)
    c = _survival_quantile(sim, theta[None, :], np.zeros(1), gamma, n_final, rng)
    for _ in range(max_refine):
        mix = ce_mixture(sim, c, rng)
        c_new = _survival_quantile(sim, mix.thetas, mix.log_a, gamma, n_final, rng)
        done_refine = abs(c_new / c - 1.0) < refine_tol
        c = c_new
        if done_refine:
            break
    return c


# ============================================================
# Exact small-route probability (cross-check only)
# ============================================================


def exceed_prob(pmf_d: np.ndarray, pmf_p: np.ndarray, T: int) -> float:
    """P(F > T) for independent discrete demands on levels 0..V-1.

    pmf_d, pmf_p : shape (L, V); row k is the pmf of stop k (route order).
    Backward recursion over (A, G) with A_k = sum_{l>k} d_l and
    G_{k-1} = max(A_{k-1}, p_k + G_k), A_{k-1} = A_k + d_k (classical.peak_distribution
    derivation). States with A > T or G > T are absorbed: A_k <= F and G_k <= F, and
    F is non-decreasing in both, so their mass is exactly the exceedance mass.
    """
    pmf_d = np.asarray(pmf_d, dtype=np.float64)
    pmf_p = np.asarray(pmf_p, dtype=np.float64)
    if pmf_d.ndim != 2 or pmf_d.shape != pmf_p.shape:
        raise ValueError("pmf_d and pmf_p must be 2-D with equal shape")
    L, V = pmf_d.shape
    T = int(T)
    if T < 0:
        return 1.0
    S = T + 1
    idx = np.arange(S)
    diffs = idx[:, None] - idx[None, :]          # A' - g
    upper = idx[None, :] > idx[:, None]          # G' > A' (strict)
    P = np.zeros((S, S))
    P[0, 0] = 1.0
    exc = 0.0
    for k in reversed(range(L)):
        wd = np.where(pmf_d[k] < _PMF_TOL, 0.0, pmf_d[k])
        wp = np.where(pmf_p[k] < _PMF_TOL, 0.0, pmf_p[k])
        # delivery sub-step: A' = a + d
        Q = np.zeros((S, S))
        total = P.sum()
        for dv in np.nonzero(wd)[0]:
            if dv <= T:
                Q[dv:, :] += wd[dv] * P[:S - dv, :]
                exc += wd[dv] * P[S - dv:, :].sum()
            else:
                exc += wd[dv] * total
        # pickup sub-step: G' = max(A', g + p)
        cdf_p = np.cumsum(wp)
        Mcdf = np.where(diffs >= 0, cdf_p[np.clip(diffs, 0, V - 1)], 0.0)
        R_diag = (Q * Mcdf).sum(axis=1)          # g + p <= A' : G' = A'
        R = np.zeros((S, S))
        qtot = Q.sum()
        for pv in np.nonzero(wp)[0]:
            if pv <= T:
                R[:, pv:] += wp[pv] * Q[:, :S - pv]
                exc += wp[pv] * Q[:, S - pv:].sum()
            else:
                exc += wp[pv] * qtot
        R = np.where(upper, R, 0.0)              # keep g + p > A' only
        R[idx, idx] = R_diag
        P = R
    return float(exc)


def gauss_hermite(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Probabilists' Gauss-Hermite nodes and weights normalised to sum to one."""
    x, w = np.polynomial.hermite_e.hermegauss(int(n))
    return x, w / math.sqrt(2.0 * math.pi)


def _conditional_pmfs(sim: RouteSim, edges_u: np.ndarray, z: np.ndarray,
                      which: str, V: int, u: float) -> np.ndarray:
    """Level pmfs (n_nodes, L, V) of the demands given the factor value z.

    Level v carries the nearest-level bin [(v - 1/2) u, (v + 1/2) u) of the
    continuous demand; level 0 carries [0, u/2); the top level carries the
    upper tail. Conditional cdf: P(d <= x | z) = Phi((Phi^{-1}(G(x)) - sqrt(rho) z) / sqrt(1 - rho)).
    """
    if which == "d":
        mean_active, scale = sim.act_d, sim.scale_d
    else:
        mean_active, scale = sim.act_p, sim.scale_p
    edges = (np.arange(1, V) - 0.5) * u                     # E_1 .. E_{V-1}
    G = stats.gamma.cdf(edges[None, :], sim.shape, scale=scale[:, None])
    Ginv = ndtri(np.clip(G, 0.0, 1.0))                      # (L, V-1), may be +-inf
    cdf = ndtr((Ginv[None, :, :] - sim._sr * z[:, None, None]) / sim._s1)
    n = z.size
    L = sim.L
    Fx = np.concatenate([np.zeros((n, L, 1)), cdf, np.ones((n, L, 1))], axis=2)
    pmf = np.clip(np.diff(Fx, axis=2), 0.0, None)
    pmf[:, ~mean_active, :] = 0.0
    pmf[:, ~mean_active, 0] = 1.0
    return pmf


def dp_check(sim: RouteSim, c: float, V: int = 256, n_gh: int = 40) -> float:
    """Exact P(F > c) for a route with L <= 8 under the discretised model.

    Demands are discretised on V levels with unit u = c / (K + 1/2), where
    K = floor((V - 1) / 1.05 - 1/2), so the top level is at least 1.05 c and the
    threshold sits on a bin boundary (F > c iff level > K). Given the factor
    pair (W_d, W_p) the customers are independent; the exact probability is the
    Gauss-Hermite average over the two factors of exceed_prob.
    """
    if sim.L > 8:
        raise ValueError(f"dp_check is for L <= 8, got L={sim.L}")
    if not c > 0:
        raise ValueError(f"c must be positive, got {c}")
    V = int(V)
    K = int(math.floor((V - 1) / 1.05 - 0.5))
    if K < 1:
        raise ValueError(f"V={V} too small")
    u = c / (K + 0.5)
    z, wz = gauss_hermite(n_gh)
    PD = _conditional_pmfs(sim, None, z, "d", V, u)   # (n, L, V)
    PP = _conditional_pmfs(sim, None, z, "p", V, u)
    total = 0.0
    for a in range(z.size):
        for b in range(z.size):
            total += wz[a] * wz[b] * exceed_prob(PD[a], PP[b], K)
    return float(total)


# ============================================================
# Cost model
# ============================================================


def c_crude(p: float, eps_rel: float) -> float:
    """Classical crude Monte Carlo sample count for relative error eps_rel at 95 per cent."""
    return Z95 * Z95 * (1.0 - p) / (p * eps_rel * eps_rel)


def c_is(RV: float, eps_rel: float) -> float:
    """Classical importance sampling sample count, z^2 RV / eps_rel^2."""
    return Z95 * Z95 * RV / (eps_rel * eps_rel)


def _doubling_cost(k: int, delta: float, M: int) -> float:
    """Sum over rounds t = 0..log2 M of r_t (2 2^t + 1), r_t = r_needed(k, 6 delta / (pi^2 (t+1)^2))."""
    total = 0.0
    T = int(round(math.log2(M)))
    for t in range(T + 1):
        r_t = r_needed(k, 6.0 * delta / (math.pi ** 2 * (t + 1) ** 2))
        total += r_t * (2 * 2 ** t + 1)
    return total


def _min_pow2(err, target: float) -> int:
    M = 1
    while err(M) > target:
        M *= 2
        if M > 2 ** 62:
            raise ValueError("amplitude estimation precision not reached")
    return M


def aqe_cost(p: float, eps_rel: float, delta: float = 0.05, k: int = 1) -> dict:
    """Explicit canonical amplitude estimation cost of crude MC (Fact 1 constants).

    M is the smallest power of 2 with h_{M,k}(p(1-p)) <= eps_rel p. Returns
    {"M", "ideal", "doubling"} with ideal = r (2M + 1), r = r_needed(k, delta), and
    doubling = sum over the search rounds (see _doubling_cost).
    """
    if not 0.0 < p < 1.0:
        raise ValueError(f"p must lie in (0, 1), got {p}")
    target = eps_rel * p
    M = _min_pow2(lambda m: cert_h(m, k, p * (1.0 - p)), target)
    r = r_needed(k, delta)
    return {"M": M, "ideal": float(r * (2 * M + 1)), "doubling": _doubling_cost(k, delta, M)}


def aqe_is_cost(RV: float, p: float, eps_rel: float, delta: float = 0.05, k: int = 1) -> dict:
    """Idealised quantum cost of estimating P(F > c) by amplitude (mean) estimation
    of the importance weight times the indicator, under the IS proposal.

    The estimated quantity is the normalised variable Y / p (mean 1, variance RV),
    so the target is an absolute error eps_rel on it and the amplitude-estimation
    error is 2 pi k sqrt(RV) / M + pi^2 k^2 / M^2 (no p factor). M is the smallest
    power of 2 meeting the target; cost r (2M + 1) with r = r_needed(k, delta), and
    the doubling variant as in aqe_cost.

    This is an idealisation: mean estimation costs O(sigma / eps) with constants
    taken from the canonical amplitude-estimation bound (Fact 1), and it ignores the
    reversible likelihood-ratio arithmetic and the heavier tail of the weights. It is
    optimistic for the quantum side. The cost depends on p only through the validity
    check: for fixed RV it is constant in p. For RV = (1 - p) / p it is at most the
    crude cost aqe_cost(p, eps_rel), because the crude floor pi^2 k^2 / (p M^2) is
    stricter than the IS floor pi^2 k^2 / M^2.
    """
    if not 0.0 < p < 1.0:
        raise ValueError(f"p must lie in (0, 1), got {p}")
    if not RV > 0:
        raise ValueError(f"RV must be positive, got {RV}")
    if not eps_rel > 0:
        raise ValueError(f"eps_rel must be positive, got {eps_rel}")

    def err(M):
        return 2 * math.pi * k * math.sqrt(RV) / M + math.pi ** 2 * k * k / (M * M)

    M = _min_pow2(err, eps_rel)
    r = r_needed(k, delta)
    return {"M": M, "ideal": float(r * (2 * M + 1)), "doubling": _doubling_cost(k, delta, M)}


KS_DEFAULT = (1,) + tuple(range(3, 21))


def best_aqe(cost_fn, ks=KS_DEFAULT, delta: float = 0.05) -> dict:
    """Minimum over k of ideal and of doubling cost; returns the per-variant best.

    cost_fn(k, delta) -> dict as aqe_cost. Result: {"ideal", "doubling", "k_ideal", "k_doubling"}.
    """
    best = {"ideal": (math.inf, None), "doubling": (math.inf, None)}
    for k in ks:
        out = cost_fn(int(k), delta)
        for key in ("ideal", "doubling"):
            if out[key] < best[key][0]:
                best[key] = (out[key], int(k))
    return {"ideal": best["ideal"][0], "doubling": best["doubling"][0],
            "k_ideal": best["ideal"][1], "k_doubling": best["doubling"][1]}
