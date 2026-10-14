"""Tests for rarecert/plan.py (plan-level hub certification, RARECERT THEORY.md section 5).

Covers the hub simulator against RouteSim (exact equality), the sparse likelihood ratio
against the dense and the direct density computation (1e-10), the unit mean of the
mixture likelihood ratio under the proposal, and on a small synthetic hub (3 routes,
L = 3, gamma about 1e-2): agreement with crude Monte Carlo, the route bounds, and the
direction of the independence approximation (truth <= independence, positive association
of the increasing route events through the shared factors)."""

from __future__ import annotations

import math

import numpy as np
from scipy.special import logsumexp

from quantum.rare import RouteSim
from rarecert.plan import HubSim, hub_estimate, route_baselines

CV = 0.3
RHO = 0.6


def _synthetic_plan(seed: int = 7) -> dict:
    """3 routes with L = 3 each, customers 1..9, depot 0 unused."""
    rng = np.random.default_rng(seed)
    dbar = np.concatenate([[0.0], rng.uniform(500.0, 1500.0, 9)])
    pbar = np.concatenate([[0.0], rng.uniform(500.0, 1500.0, 9)])
    routes = [[1, 4, 7], [2, 5, 8], [3, 6, 9]]
    return {"dbar": dbar.tolist(), "pbar": pbar.tolist(), "routes": routes, "Q": 6000.0}


def _hub_index(hub: HubSim, cust: int) -> int:
    """Position of a customer in the flattened hub order (route by route)."""
    flat = [c for r in hub.routes for c in r]
    return flat.index(cust)


def _calibrated_capacity(hub: HubSim, gamma: float, seed: int = 11) -> float:
    """Capacity at which the hub overflow probability is about gamma, from a fixed-seed
    crude sample of the hub's maximal route peak (only used to place the test threshold)."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((200_000, hub.dim))
    return float(np.quantile(hub.peaks(x).max(axis=1), 1.0 - gamma))


def _crude_hub(hub: HubSim, c: float, N: int, rng: np.random.Generator, chunk: int = 100_000):
    hits = 0
    done = 0
    while done < N:
        n = min(chunk, N - done)
        x = rng.standard_normal((n, hub.dim))
        hits += int(np.count_nonzero(hub.fail(x, c)))
        done += n
    p = hits / N
    rse = math.sqrt((1.0 - p) / (p * N))
    return p, rse


def test_hub_peaks_equal_route_sim_exactly():
    plan = _synthetic_plan()
    hub = HubSim(plan, cv=CV, rho=RHO)
    rng = np.random.default_rng(1)
    x = rng.standard_normal((50, hub.dim))
    peaks = hub.peaks(x)
    assert peaks.shape == (50, 3)
    for r, route in enumerate(plan["routes"]):
        j = [_hub_index(hub, c) for c in route]
        n = hub.n
        # route input in RouteSim order: (W_d, W_p, E_d[route order], E_p[route order])
        xr = np.column_stack([x[:, 0], x[:, 1], x[:, [2 + k for k in j]], x[:, [2 + n + k for k in j]]])
        sim = RouteSim([plan["dbar"][c] for c in route], [plan["pbar"][c] for c in route], cv=CV, rho=RHO)
        assert np.array_equal(peaks[:, r], sim.peak(xr))
    # a single-row input and the chunked path agree with the batch
    assert np.array_equal(hub.peaks(x[3]), peaks[3:4])


def test_sparse_log_weight_matches_dense_and_density():
    plan = _synthetic_plan()
    hub = HubSim(plan, cv=CV, rho=RHO)
    c = _calibrated_capacity(hub, 1e-2)
    mix = hub.build_mixture(c)
    assert mix.K > 2
    # dense matrix of the proposal means
    T = np.zeros((mix.K, hub.dim))
    for m in range(1, mix.K):
        T[m, mix.idx[m]] = mix.val[m]
    rng = np.random.default_rng(2)
    x = hub.sample(mix, 1000, rng)
    A = mix.log_a[None, :] + x @ T.T - 0.5 * (T * T).sum(axis=1)[None, :]
    lw_dense = -logsumexp(A, axis=1)
    lw = hub.log_weights(x, mix)
    assert np.max(np.abs(lw - lw_dense)) < 1e-10
    # direct density evaluation: log w = log phi(x) - log q(x), q = sum_m a_m N(theta_m, I)
    d = hub.dim
    log_phi = lambda y: -0.5 * np.sum(y * y, axis=1) - 0.5 * d * math.log(2 * math.pi)
    logq = logsumexp(np.stack([math.log(a) + log_phi(x - T[m]) for m, a in enumerate(mix.a)]), axis=0)
    assert np.max(np.abs(lw - (log_phi(x) - logq))) < 1e-10


def test_mixture_likelihood_ratio_has_unit_mean():
    plan = _synthetic_plan()
    hub = HubSim(plan, cv=CV, rho=RHO)
    c = _calibrated_capacity(hub, 1e-2)
    mix = hub.build_mixture(c)
    rng = np.random.default_rng(3)
    x = hub.sample(mix, 200_000, rng)
    w = np.exp(hub.log_weights(x, mix))
    assert w.max() <= 1.0 / mix.a[0] + 1e-9        # defensive component bounds the ratio
    assert abs(w.mean() - 1.0) < 0.02


def test_hub_estimate_agrees_with_crude_and_bounds_hold():
    plan = _synthetic_plan()
    hub = HubSim(plan, cv=CV, rho=RHO)
    gamma = 1e-2
    c = _calibrated_capacity(hub, gamma)
    rng = np.random.default_rng(4)
    est = hub_estimate(hub, c, 200_000, rng)
    crude_p, crude_rse = _crude_hub(hub, c, 1_000_000, np.random.default_rng(5))
    assert 0.3 * gamma < est["p_hub"] < 3.0 * gamma        # calibration sanity
    # SMIS vs crude MC within 4 combined rse
    comb = math.sqrt((est["rse"] * est["p_hub"]) ** 2 + (crude_rse * crude_p) ** 2)
    assert abs(est["p_hub"] - crude_p) <= 4.0 * comb

    base = route_baselines(hub, c, np.random.default_rng(6), N_r=200_000)
    p_r = base["route_p_smis"]
    rel_r = np.max(base["route_rse"][p_r > 0]) if np.any(p_r > 0) else 0.0
    slack = 4.0 * (est["rse"] + rel_r)
    # bounds: max_r p_r <= P_hub <= sum_r p_r (up to estimation noise)
    assert est["p_hub"] * (1.0 + slack) >= base["lower_max"]
    assert est["p_hub"] * (1.0 - slack) <= base["upper_sum"]
    # independence approximation: truth <= 1 - prod(1 - p_r) for positively dependent routes
    indep = base["indep"]
    assert indep >= base["lower_max"]
    assert est["p_hub"] <= indep * (1.0 + slack)
    assert crude_p <= indep * (1.0 + 4.0 * crude_rse)
    assert indep <= base["upper_sum"] + 1e-12
    # attribution: non-negative, sums to P(exactly one route fails | hub fails) <= 1
    assert np.all(est["route_attr"] >= 0.0)
    assert est["route_attr"].sum() <= 1.0 + 1e-12
    assert 0.0 <= est["multi_frac"] <= 1.0
