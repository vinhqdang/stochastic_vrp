"""Plan-level (hub-level) rare-overflow certification (THEORY.md section 5).

A hub is one region instance with its SAA plan: routes r = 1..R, every customer in
exactly one route. The demand copula is one factor per demand type for the whole
instance (dethloff_runner.sample_demands), so the hub input is

    x = (W_d, W_p, E_d[all n customers], E_p[all n customers]),   dim = 2 + 2n,

and route r reads the sub-vector x[cols_r] with cols_r = (0, 1, 2 + j, 2 + n + j)
over its customers j in route order; this is exactly RouteSim's input ordering for
that route. A hub overflow is the union of the route events {F_r > c}, all with the
same capacity c. Hubs of a region have separate factors, so the region probability
is 1 - prod_h (1 - P_h) exactly in this model (a model property, not an estimate).

Proposal: for every route, the route-level SMIS components (rarecert.smis) are
embedded in the hub coordinates (factor coordinates plus that route's coordinates,
zeros elsewhere); components with energy E <= E_min + log(1/eta) are kept over all
routes, with weights proportional to exp(-(E - E_min)) scaled to 1 - defensive, and
a defensive N(0, I) component of weight `defensive`. Each embedded component is
sparse (2 + 2 L_r nonzeros), so the likelihood ratio uses index lists.

Entry points: HubSim (simulator, embedding, proposal, likelihood ratio),
hub_estimate, route_baselines, capacity_for.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np
from scipy.special import logsumexp

from quantum.rare import RouteSim, is_mixture
from rarecert.smis import smis_mixture

ETA_HUB = 1e-4          # hub-level mode cut (also used for the route-level builds)
DEFENSIVE = 0.05        # weight of the defensive N(0, I) component
KEEP = 1.0 - DEFENSIVE  # total weight of the kept modes
CHUNK_BYTES = 1e8       # rows of x are chunked so one (rows, dim) float64 block stays near this size


@dataclass
class HubMixture:
    """Sparse hub proposal. Row 0 is the defensive N(0, I) (empty index list).

    idx[m], val[m]: hub columns and values of the mean theta_m of component m.
    log_a[m]: log weight; sq[m] = |theta_m|^2; energy[m]: |theta_m|^2 / 2 (row 0: 0).
    route[m]: the route whose SMIS build produced component m (-1 for defensive).
    """

    idx: list
    val: list
    log_a: np.ndarray
    sq: np.ndarray
    energy: np.ndarray
    route: np.ndarray
    build_seconds: float = 0.0
    extra: dict = field(default_factory=dict)

    @property
    def K(self) -> int:
        return len(self.idx)

    @property
    def n_components(self) -> int:
        return self.K - 1

    @property
    def a(self) -> np.ndarray:
        return np.exp(self.log_a)


ARC_FALLBACK = (3, 8)   # route builds: n_arc = 0 first; arc candidates only if no ray reaches c


def route_mixture(sim: RouteSim, c: float, eta: float = ETA_HUB, defensive: float = DEFENSIVE):
    """Route-level SMIS build (smis.smis_mixture). The block rays of smis.py reach c only
    if one block alone can push the stage load past c; a route whose peak needs both blocks
    at the same stage has no block-ray candidate (IndexError inside smis_mixture). Then the
    arc candidates between the two blocks of a stage are used (n_arc = 3, then 8). Returns
    (mixture, diagnostics, n_arc used)."""
    last = None
    for n_arc in (0,) + ARC_FALLBACK:
        try:
            mix, diag = smis_mixture(sim, c, eta=eta, n_arc=n_arc, defensive=defensive)
        except IndexError as exc:   # empty candidate list: no ray or arc reaches c
            last = exc
            continue
        return mix, diag, n_arc
    raise RuntimeError(f"no SMIS candidate reaches c={c} (also with arcs): {last}")


class HubSim:
    """Hub-level simulator of a region plan.

    plan : dict with keys
        "dbar", "pbar"  per-node mean delivery / pickup (indexed by node id, depot 0),
        "routes"        list of routes, each a list of customer node ids in route order,
        "Q"             (optional) vehicle capacity, used by capacity_for.
    cv, rho : demand coefficient of variation and copula loading (as RouteSim).
    """

    def __init__(self, plan: dict, cv: float = 0.3, rho: float = 0.6):
        dbar_all = np.asarray(plan["dbar"], dtype=np.float64).ravel()
        pbar_all = np.asarray(plan["pbar"], dtype=np.float64).ravel()
        routes = [[int(v) for v in r] for r in plan["routes"]]
        if not routes or any(len(r) == 0 for r in routes):
            raise ValueError("plan must have at least one route and no empty route")
        custs = [c for r in routes for c in r]
        if len(set(custs)) != len(custs):
            raise ValueError("every customer must appear in exactly one route")
        self.routes = routes
        self.R = len(routes)
        self.n = len(custs)
        self.dim = 2 + 2 * self.n
        self.Q = float(plan["Q"]) if plan.get("Q") is not None else None
        self.cv = float(cv)
        self.rho = float(rho)
        pos = {c: j for j, c in enumerate(custs)}
        self.cols = []
        self.sims = []
        for r in routes:
            j = np.array([pos[c] for c in r], dtype=int)
            self.cols.append(np.concatenate([[0, 1], 2 + j, 2 + self.n + j]))
            self.sims.append(RouteSim(dbar_all[r], pbar_all[r], cv=self.cv, rho=self.rho))
        self.row_chunk = max(256, int(CHUNK_BYTES // (8 * self.dim)))
        self._max_peak = [None] * self.R

    def route_max_peak(self, r: int) -> float:
        """Largest peak route r can reach in the simulator: the peak is non-decreasing in
        every input and the inputs are clipped at Phi^{-1}(1 - 1e-15), so the value at
        x = +40 everywhere is the supremum. If it is <= c, P(F_r > c) = 0 exactly."""
        if self._max_peak[r] is None:
            self._max_peak[r] = float(self.sims[r].peak(np.full((1, self.sims[r].dim), 40.0))[0])
        return self._max_peak[r]

    def reachable(self, r: int, c: float) -> bool:
        return self.route_max_peak(r) > c

    # ---------------- simulator ----------------

    def route_sim(self, r: int) -> RouteSim:
        """RouteSim of route r (same means and copula as the hub)."""
        return self.sims[r]

    def route_cols(self, r: int) -> np.ndarray:
        """Hub columns of route r's RouteSim input, in RouteSim order."""
        return self.cols[r]

    def embed(self, r: int, theta_r) -> np.ndarray:
        """Embed a route-r input vector (RouteSim coordinates) in the hub input space."""
        theta_r = np.asarray(theta_r, dtype=np.float64).ravel()
        if theta_r.shape != (self.sims[r].dim,):
            raise ValueError(f"route {r} input must have length {self.sims[r].dim}")
        out = np.zeros(self.dim)
        out[self.cols[r]] = theta_r
        return out

    def _check_x(self, x) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            x = x[None, :]
        if x.ndim != 2 or x.shape[1] != self.dim:
            raise ValueError(f"x must have shape (N, {self.dim}), got {x.shape}")
        return x

    def peaks(self, x) -> np.ndarray:
        """Route peaks F_r for hub inputs x of shape (N, dim): array (N, R)."""
        x = self._check_x(x)
        out = np.empty((x.shape[0], self.R), dtype=np.float64)
        for s in range(0, x.shape[0], self.row_chunk):
            xs = x[s:s + self.row_chunk]
            for r in range(self.R):
                out[s:s + xs.shape[0], r] = self.sims[r].peak(xs[:, self.cols[r]])
        return out

    def fail(self, x, c: float) -> np.ndarray:
        """Boolean (N,): some route's peak exceeds c."""
        return (self.peaks(x) > c).any(axis=1)

    # ---------------- proposal ----------------

    def build_mixture(self, c: float, eta: float = ETA_HUB, defensive: float = DEFENSIVE,
                      keep: float = None) -> HubMixture:
        """Hub proposal at capacity c from the route-level SMIS builds (no simulation)."""
        if keep is None:
            keep = 1.0 - defensive
        t0 = time.perf_counter()
        cand = []   # (energy, route, route-coordinate mean)
        n_fallback = 0
        for r in range(self.R):
            if not self.reachable(r, c):
                continue   # P(F_r > c) = 0 exactly in the simulator
            mix_r, diag_r, n_arc = route_mixture(self.sims[r], c, eta=eta, defensive=defensive)
            n_fallback += int(n_arc > 0)
            E_r = diag_r["E"]
            if not E_r:
                raise RuntimeError(f"route {r} has no SMIS mode at c={c}")
            for m, E in enumerate(E_r, start=1):
                cand.append((float(E), r, mix_r.thetas[m]))
        if not cand:
            raise RuntimeError(f"no route of the hub reaches c={c}")
        E_all = np.array([t[0] for t in cand])
        E_min = float(E_all.min())
        keep_mask = E_all <= E_min + math.log(1.0 / eta)
        sel = [cand[i] for i in np.nonzero(keep_mask)[0]]
        E = np.array([t[0] for t in sel])
        mass = np.exp(-(E - E_min))
        a = keep * mass / mass.sum()
        idx = [np.zeros(0, dtype=int)]
        val = [np.zeros(0)]
        log_a = [math.log(defensive)] if defensive > 0 else [-math.inf]
        sq = [0.0]
        energy = [0.0]
        route = [-1]
        for (Em, r, th), am in zip(sel, a):
            ii = self.cols[r]
            vv = np.asarray(th, dtype=np.float64)
            idx.append(ii)
            val.append(vv)
            log_a.append(math.log(am))
            sq.append(float(vv @ vv))
            energy.append(Em)
            route.append(r)
        return HubMixture(idx=idx, val=val, log_a=np.array(log_a), sq=np.array(sq),
                          energy=np.array(energy), route=np.array(route, dtype=int),
                          build_seconds=time.perf_counter() - t0,
                          extra={"E_min": E_min, "n_candidates": len(cand), "n_arc_fallback": n_fallback})

    def log_weights(self, x, mix: HubMixture) -> np.ndarray:
        """log w(x) = log phi(x) - log q(x), with q = sum_m a_m N(theta_m, I).

        q(x)/phi(x) = sum_m a_m exp(<x, theta_m> - |theta_m|^2/2), so
        log w = -logsumexp_m (log a_m + <x, theta_m> - |theta_m|^2/2). The inner products
        use the index lists of the sparse components. A single shift gives the familiar
        -x.theta + |theta|^2/2 (quantum.rare._draws)."""
        x = self._check_x(x)
        A = np.empty((x.shape[0], mix.K))
        A[:, 0] = mix.log_a[0]
        for m in range(1, mix.K):
            A[:, m] = mix.log_a[m] + x[:, mix.idx[m]] @ mix.val[m] - 0.5 * mix.sq[m]
        return -logsumexp(A, axis=1)

    def sample(self, mix: HubMixture, n: int, rng: np.random.Generator) -> np.ndarray:
        """n draws x = theta_m + z from the hub proposal (component m chosen by its weight)."""
        comp = rng.choice(mix.K, size=n, p=mix.a / mix.a.sum())
        x = rng.standard_normal((n, self.dim))
        for m in range(1, mix.K):
            rows = np.nonzero(comp == m)[0]
            if rows.size:
                x[np.ix_(rows, mix.idx[m])] += mix.val[m]
        return x


# ============================================================
# Estimators
# ============================================================


def hub_estimate(hub: HubSim, c: float, N: int, rng: np.random.Generator,
                 mix: HubMixture = None, chunk: int = None) -> dict:
    """SMIS estimate of P_hub(c) = P(some route peak > c) with N draws of the hub proposal.

    Returns p_hub, RV (relative variance per draw), rse, ESS (sum w)^2 / sum w^2,
    n_components (kept modes, defensive excluded), build_seconds, sample_seconds, and the
    risk attribution from the same draws:
      route_p[r]     IS estimate of P(F_r > c);
      route_attr[r]  estimate of P(route r is the unique failing route | hub fails)
                     = mean(w 1{only r fails}) / mean(w 1{hub fails}); the attribution
                     sums to P(exactly one route fails | hub fails) <= 1.
      multi_frac     P(two or more routes fail | hub fails), the remainder of the sum.
    """
    N = int(N)
    if N < 2:
        raise ValueError("N must be >= 2")
    if mix is None:
        mix = hub.build_mixture(c)
    chunk = int(chunk) if chunk else hub.row_chunk
    t0 = time.perf_counter()
    s1 = s2 = sw = sw2 = 0.0
    route_w = np.zeros(hub.R)
    uniq_w = np.zeros(hub.R)
    multi_w = 0.0
    done = 0
    while done < N:
        n = min(chunk, N - done)
        x = hub.sample(mix, n, rng)
        w = np.exp(hub.log_weights(x, mix))
        F = hub.peaks(x) > c
        fail = F.any(axis=1)
        y = w * fail
        s1 += float(y.sum())
        s2 += float(y @ y)
        sw += float(w.sum())
        sw2 += float(w @ w)
        Ff = F.astype(np.float64)
        route_w += w @ Ff
        cnt = F.sum(axis=1)
        only = cnt == 1
        if only.any():
            r_idx = F[only].argmax(axis=1)
            np.add.at(uniq_w, r_idx, w[only])
        multi_w += float(w[cnt >= 2].sum())
        done += n
    p = s1 / N
    var = max((s2 - s1 * s1 / N) / (N - 1), 0.0)
    if p > 0.0:
        RV = var / (p * p)
        rse = math.sqrt(RV / N)
        attr = uniq_w / s1
        multi = multi_w / s1
    else:
        RV = rse = math.nan
        attr = np.full(hub.R, math.nan)
        multi = math.nan
    return {
        "p_hub": p, "RV": RV, "rse": rse, "ESS": sw * sw / sw2 if sw2 > 0 else math.nan,
        "n_components": mix.n_components, "build_seconds": mix.build_seconds,
        "n_arc_fallback": mix.extra["n_arc_fallback"],
        "sample_seconds": time.perf_counter() - t0, "N": N,
        "route_p": route_w / N, "route_attr": attr, "multi_frac": multi,
    }


def route_baselines(hub: HubSim, c: float, rng: np.random.Generator, N_r: int = 20_000) -> dict:
    """Route probabilities p_r = smis_estimate per route (N_r draws each) at capacity c,
    and the baselines derived from them: the bounds max_r p_r <= P_hub <= sum_r p_r and
    the independence approximation 1 - prod_r (1 - p_r), which ignores the shared factors."""
    p = np.empty(hub.R)
    rse = np.empty(hub.R)
    n_modes = np.empty(hub.R, dtype=int)
    for r in range(hub.R):
        if not hub.reachable(r, c):
            p[r], rse[r], n_modes[r] = 0.0, 0.0, 0   # exact zero, see HubSim.route_max_peak
            continue
        mix, diag, _ = route_mixture(hub.sims[r], c, eta=ETA_HUB)
        p_hat, _, rse_r, _ = is_mixture(hub.sims[r], c, mix, int(N_r), rng)
        p[r], rse[r], n_modes[r] = p_hat, rse_r, diag["n_modes"]
    return {
        "route_p_smis": p, "route_rse": rse, "route_modes": n_modes,
        "upper_sum": float(p.sum()), "lower_max": float(p.max()),
        "indep": float(-np.expm1(np.sum(np.log1p(-np.minimum(p, 1.0 - 1e-16))))),
    }


def route_min_energy(hub: HubSim, r: int, c: float):
    """Smallest dominating-point energy E_r(c) of route r at capacity c, from the route SMIS
    build (no draws). None if no route point reaches c (P_r = 0 exactly)."""
    if not hub.reachable(r, c):
        return None
    _, diag, _ = route_mixture(hub.sims[r], c, eta=ETA_HUB)
    return float(diag["E_min"])


def log_S(hub: HubSim, c: float) -> float:
    """Laplace-type proxy log S(c) = log sum_r exp(-E_r(c)). It is a proxy, NOT a bound: the
    route tail is approximated by exp(-E_r(c)) with no prefactor, and the sum ignores the
    shared factors. It is used only to locate c."""
    vals = [-e for e in (route_min_energy(hub, r, c) for r in range(hub.R)) if e is not None]
    if not vals:
        return -math.inf
    return float(logsumexp(np.asarray(vals)))


def capacity_for(hub: HubSim, gamma: float, max_evals: int = 6, tol: float = 0.02) -> dict:
    """Capacity c with log S(c) = log gamma, S the Laplace-type proxy of log_S.

    Evaluations are on [Q, 3Q] (S decreasing in c): f(c) = log S(c) - log gamma. Endpoints
    first, then safeguarded secant steps inside the bracket, at most max_evals evaluations
    in total. The returned c is the bracket end with f <= 0 (proxy at or below gamma), so
    the search is conservative with respect to the proxy. Returns {c, buffer (= c/Q - 1),
    logS_at_c, residual (log S - log gamma at c), evals, status}. status is 'secant',
    'capped_3Q' (f(3Q) > 0) or 'at_or_below_Q' (f(Q) <= 0, c = Q).
    """
    if hub.Q is None:
        raise ValueError("capacity_for needs plan['Q']")
    if not 0.0 < gamma < 1.0:
        raise ValueError(f"gamma must lie in (0, 1), got {gamma}")
    Q = hub.Q
    lg = math.log(gamma)
    evals = 0

    def f(c):
        nonlocal evals
        evals += 1
        return log_S(hub, c) - lg

    flo = f(Q)
    if flo <= 0.0:
        return {"c": Q, "buffer": 0.0, "logS_at_c": flo + lg, "residual": flo,
                "evals": evals, "status": "at_or_below_Q"}
    lo, hi = Q, 3.0 * Q
    fhi = f(hi)
    if fhi > 0.0:
        return {"c": hi, "buffer": 2.0, "logS_at_c": fhi + lg, "residual": fhi,
                "evals": evals, "status": "capped_3Q"}
    # Illinois regula falsi: the secant uses weighted endpoint values (halved when the same
    # endpoint is replaced twice in a row, which prevents stagnation); fhi, flo stay the true
    # residuals for reporting and the stopping rule.
    wlo, whi = flo, fhi
    last = 0
    while evals < max_evals and abs(fhi) > tol:
        w = hi - lo
        c_new = lo + (-wlo) * w / (whi - wlo)
        c_new = min(max(c_new, lo + 1e-6 * w), hi - 1e-6 * w)
        fn = f(c_new)
        if fn > 0.0:
            lo, flo, wlo = c_new, fn, fn
            if last == -1:
                whi *= 0.5
            last = -1
        else:
            hi, fhi, whi = c_new, fn, fn
            if last == 1:
                wlo *= 0.5
            last = 1
    return {"c": hi, "buffer": hi / Q - 1.0, "logS_at_c": fhi + lg, "residual": fhi,
            "evals": evals, "status": "secant"}
