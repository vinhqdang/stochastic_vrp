"""Heterogeneous arc noise: H construction, Monte Carlo moments, solver consistency."""

import itertools
import math

import numpy as np
import pytest

from sortie.instances import DATA_PATH, effective_H, load_x641, make_instance, noise_matrix, with_congestion
from sortie.sim import simulate
from sortie.tsp import TourSolver, tour_stats

PLAN = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11], [12, 13, 14, 15]]


@pytest.fixture(scope="module")
def base15():
    coords, demand = load_x641(DATA_PATH)
    return make_instance(coords, demand, n=15, b=0, m=5)


def test_with_congestion_structure(base15):
    lo, hi = 1.0 / 3.0, 2.0
    inst = with_congestion(base15, v_lo_mult=lo, v_hi_mult=hi, radius_frac=0.3, seed=0)
    H = inst.H
    assert inst.name == base15.name + "-cong"
    assert H.shape == base15.C.shape
    assert np.array_equal(H, H.T)
    assert np.all(np.diag(H) == 1.0)
    off = H[~np.eye(H.shape[0], dtype=bool)]
    assert off.min() >= lo - 1e-12
    assert off.max() <= hi + 1e-12
    assert off.max() - off.min() > 0.5 * (hi - lo)
    assert np.array_equal(with_congestion(base15, seed=0).H, H)
    assert base15.H is None
    assert np.array_equal(noise_matrix(inst), base15.C * H)
    assert effective_H(base15) is None
    assert effective_H(inst) is not None


def test_with_congestion_is_centred_on_a_customer(base15):
    inst = with_congestion(base15, seed=0)
    rng = np.random.default_rng([0, 15, 0])
    centre = base15.coords[int(rng.integers(1, 16))]
    mid = 0.5 * (base15.coords[:, None, :] + base15.coords[None, :, :])
    d = np.sqrt(((mid - centre) ** 2).sum(axis=2))
    # the arc nearest the centre carries the largest multiplier
    off = ~np.eye(16, dtype=bool)
    i, j = np.unravel_index(np.argmin(np.where(off, d, np.inf)), d.shape)
    assert inst.H[i, j] == pytest.approx(inst.H[off].max(), rel=1e-6)


def test_moments_match_model_under_congestion(base15):
    inst = with_congestion(base15)
    v = 0.3
    S = 40000
    X, _ = simulate(inst, PLAN, v=v, sY=0.0, S=S, seed=21)
    C = inst.C
    D = noise_matrix(inst)
    homog_W = np.array([tour_stats(r, C)[1] for r in PLAN])
    het_W = np.array([tour_stats(r, C, D)[1] for r in PLAN])
    assert np.any(np.abs(het_W - homog_W) > 0.05 * homog_W)
    for k, r in enumerate(PLAN):
        L = tour_stats(r, C)[0]
        W = tour_stats(r, C, D)[1]
        x = X[:, k]
        mean_se = x.std(ddof=1) / math.sqrt(S)
        assert abs(x.mean() - L) < 4.0 * mean_se, (k, x.mean(), L)
        var_hat = x.var(ddof=1)
        var_model = v * v * W
        sq = (x - x.mean()) ** 2
        var_se = sq.std(ddof=1) / math.sqrt(S)
        assert abs(var_hat - var_model) < 4.0 * var_se, (k, var_hat, var_model)


def _euclid_het(rng, k):
    pts = rng.uniform(0.0, 100.0, size=(k + 1, 2))
    diff = pts[:, None, :] - pts[None, :, :]
    C = np.sqrt((diff ** 2).sum(axis=2))
    H = rng.uniform(0.3, 2.5, size=(k + 1, k + 1))
    H = (H + H.T) / 2.0
    np.fill_diagonal(H, 1.0)
    return C, C * H


def test_tour_solver_heterogeneous_kappa_equals_brute_force():
    rng = np.random.default_rng(606)
    v = 0.3
    kappas = (0.5, 1.0, 2.0)
    for case in range(20):
        k = 1 + case % 7
        kappa = kappas[case % 3]
        C, D = _euclid_het(rng, k)
        solver = TourSolver(C, kappa=kappa, v=v, D=D)
        tour, L, W = solver.solve(range(1, k + 1))
        got = L + kappa * v * math.sqrt(W)
        best = math.inf
        for p in itertools.permutations(range(1, k + 1)):
            Lp, Wp = tour_stats(list(p), C, D)
            best = min(best, Lp + kappa * v * math.sqrt(Wp))
        assert got == pytest.approx(best, rel=1e-9, abs=1e-9), (case, k, kappa)
        assert sorted(tour) == list(range(1, k + 1))
        assert (L, W) == pytest.approx(tour_stats(tour, C, D))


def test_tour_solver_kappa0_uses_L_with_tiebreak_on_D():
    rng = np.random.default_rng(707)
    for case in range(6):
        k = 2 + case % 5
        C, D = _euclid_het(rng, k)
        tour, L, W = TourSolver(C, kappa=0.0, v=0.3, D=D).solve(range(1, k + 1))
        best = min(tour_stats(list(p), C)[0] for p in itertools.permutations(range(1, k + 1)))
        assert L == pytest.approx(best, rel=1e-10, abs=1e-9)
        assert (L, W) == pytest.approx(tour_stats(tour, C, D))
