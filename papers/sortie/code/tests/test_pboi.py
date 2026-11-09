import math

import numpy as np
import pytest
from scipy.special import ndtr

from sortie.pboi import (
    Marginals,
    expost_var,
    gmd_normal_closed,
    lam_vector,
    objective,
    ordered,
    pboi,
    rank_prices,
    sigma_prices,
)

MEASURES = ("max", "range", "gmd", "mad")


def _folded_mean(d, s):
    z = d / s
    return s * math.sqrt(2.0 / math.pi) * math.exp(-0.5 * z * z) + d * (1.0 - 2.0 * ndtr(-z))


def _gamma_samples(L, sigma, n, seed):
    rng = np.random.default_rng(seed)
    L = np.asarray(L, float)
    sigma = np.asarray(sigma, float)
    shape = (L / sigma) ** 2
    scale = sigma ** 2 / L
    return rng.gamma(shape=shape[None, :], scale=scale[None, :], size=(n, L.shape[0]))


def test_lam_vectors():
    assert np.allclose(lam_vector("max", 4), [0, 0, 0, 1])
    assert np.allclose(lam_vector("range", 4), [-1, 0, 0, 1])
    g = lam_vector("gmd", 5)
    assert np.allclose(g, 2 * (2 * np.arange(1, 6) - 6) / 25.0)
    assert g.sum() == pytest.approx(0.0, abs=1e-15)
    assert np.all(np.diff(g) >= 0)
    assert np.allclose(lam_vector("mad", 5), [-0.2, -0.2, 0, 0.2, 0.2])
    assert np.allclose(lam_vector("mad", 3), [-1 / 3, 0, 1 / 3])
    with pytest.raises(ValueError):
        lam_vector("mad", 4)
    with pytest.raises(ValueError):
        lam_vector("var", 4)


def test_ordered_matrix_and_vector():
    a = np.array([[3.0, 1.0, 2.0], [0.0, 5.0, -1.0]])
    lam = lam_vector("range", 3)
    assert np.allclose(ordered(a, lam), [2.0, 6.0])
    assert ordered(a[0], lam) == pytest.approx(3.0 - 1.0)


def test_pboi_normal_vs_gmd_closed_form():
    rng = np.random.default_rng(11)
    for m in range(2, 7):
        L = 100.0 + rng.normal(0.0, 5.0, m)
        sigma = rng.uniform(3.0, 10.0, m)
        val = pboi(L, sigma, "gmd", family="normal")
        ref = gmd_normal_closed(L, sigma)
        assert abs(val - ref) / abs(ref) < 1e-4, (m, val, ref)


def test_pboi_normal_range_m2_folded_normal():
    L = np.array([100.0, 107.0])
    sigma = np.array([6.0, 9.0])
    d = L[0] - L[1]
    s = math.hypot(sigma[0], sigma[1])
    ref = _folded_mean(d, s)
    val = pboi(L, sigma, "range", family="normal")
    assert abs(val - ref) / ref < 1e-4


def test_pboi_max_iid_normals():
    mu, sd = 50.0, 4.0
    val = pboi([mu, mu], [sd, sd], "max", family="normal")
    assert val == pytest.approx(mu + sd / math.sqrt(math.pi), rel=1e-6)


def test_pboi_deterministic_limit_is_ordered_measure():
    L = np.array([80.0, 120.0, 95.0, 101.0, 60.0])
    for name in MEASURES:
        lam = lam_vector(name, 5)
        val = pboi(L, 3e-3 * L, lam, G=1025, family="gamma")
        assert val == pytest.approx(ordered(L, lam), rel=1e-7)


@pytest.mark.parametrize("name", MEASURES)
def test_pboi_gamma_matches_monte_carlo(name):
    L = np.array([120.0, 130.0, 140.0, 150.0, 160.0])
    sigma = 0.2 * L
    n = 400_000
    X = _gamma_samples(L, sigma, n, seed=1234)
    vals = ordered(X, lam_vector(name, 5))
    mc = float(vals.mean())
    se = float(vals.std(ddof=1) / math.sqrt(n))
    exact = pboi(L, sigma, name, family="gamma")
    assert abs(exact - mc) <= 4.0 * se, (name, exact, mc, se)


@pytest.mark.parametrize("family", ["gamma", "normal"])
def test_rank_prices_sum_to_sum_of_lambda(family):
    L = np.array([100.0, 110.0, 96.0, 130.0, 105.0])
    sigma = np.array([8.0, 12.0, 7.0, 15.0, 9.0])
    for name in MEASURES:
        lam = lam_vector(name, 5)
        rp = rank_prices(L, sigma, lam, family=family)
        assert abs(rp.sum() - lam.sum()) < 1e-6, (name, rp.sum(), lam.sum())


def test_rank_prices_match_finite_difference_in_means():
    L = np.array([100.0, 110.0, 96.0, 130.0, 105.0])
    sigma = np.array([8.0, 12.0, 7.0, 15.0, 9.0])
    for name in ("gmd", "range", "max"):
        lam = lam_vector(name, 5)
        rp = rank_prices(L, sigma, lam, family="normal")
        for k in range(5):
            h = 1e-3 * sigma[k]
            Lp = L.copy()
            Lm = L.copy()
            Lp[k] += h
            Lm[k] -= h
            fd = (pboi(Lp, sigma, lam, family="normal") - pboi(Lm, sigma, lam, family="normal")) / (2 * h)
            assert abs(rp[k] - fd) <= 1e-3 * abs(rp[k]), (name, k, rp[k], fd)


def test_sigma_prices_var_exact_and_fd_positive_for_max():
    L = np.array([100.0, 110.0, 96.0])
    sigma = np.array([8.0, 12.0, 7.0])
    psi_var = sigma_prices(L, sigma, "var")
    assert np.allclose(psi_var, 2 * 2 / 9.0 * sigma)
    psi = sigma_prices(L, sigma, "max", family="normal")
    assert np.all(psi > 0)


def test_expost_var_matches_monte_carlo():
    L = np.array([120.0, 150.0, 135.0, 140.0])
    sigma = np.array([20.0, 30.0, 25.0, 15.0])
    n = 400_000
    X = _gamma_samples(L, sigma, n, seed=99)
    vals = X.var(axis=1)
    mc = float(vals.mean())
    se = float(vals.std(ddof=1) / math.sqrt(n))
    exact = expost_var(L, sigma)
    assert abs(exact - mc) <= 4.0 * se, (exact, mc, se)


def test_objective_dispatch():
    L = np.array([100.0, 110.0, 96.0, 104.0, 99.0])
    W = np.array([500.0, 600.0, 450.0, 520.0, 480.0])
    v = 0.3
    assert objective(L, W, v, "var") == pytest.approx(expost_var(L, v * np.sqrt(W)))
    assert objective(L, W, v, "gmd") == pytest.approx(
        pboi(L, v * np.sqrt(W), lam_vector("gmd", 5)))


def test_marginals_gamma_moments():
    L = np.array([50.0, 70.0])
    sigma = np.array([5.0, 9.0])
    marg = Marginals(L, sigma, "gamma")
    assert np.allclose(marg.alpha * marg.theta, L)
    assert np.allclose(marg.alpha * marg.theta ** 2, sigma ** 2)
    q = marg.ppf(0.5)
    cdf = marg.cdf(q)
    assert np.allclose(np.diag(cdf), 0.5, atol=1e-8)


def test_narrow_route_resolved():
    """A near-trivial short route next to long noisy routes (review finding):
    quadrature must resolve it; its range price is -1 and prices sum to 0."""
    import numpy as np
    from sortie.pboi import pboi, rank_prices
    L = np.array([2.0, 900, 950, 880, 920])
    W = np.r_[2.0, L[1:] ** 2 / 10]
    s = 0.3 * np.sqrt(W)
    assert abs(pboi(L, s, "range") - 1006.6818547) < 1e-5
    rp = rank_prices(L, s, "range")
    assert abs(rp[0] + 1.0) < 1e-6 and abs(rp.sum()) < 1e-8
