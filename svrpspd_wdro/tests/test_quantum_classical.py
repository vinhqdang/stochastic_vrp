"""Tests for quantum/classical.py: exact peak-load pmf, risk measures,
insertion lemma, insertion window and discretisation."""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from core.instance import Instance
from core.route import Route
from core.scenarios import ScenarioConfig, generate_scenarios
from quantum.classical import (
    DiscreteDemand,
    _cvar_from_pmf,
    chance,
    cvar,
    discretize_instance,
    discretize_samples,
    insertion_window,
    peak_distribution,
    peak_load,
)


def _random_dem(rng, n, V):
    pd = rng.dirichlet(np.ones(V), size=n)
    pp = rng.dirichlet(np.ones(V), size=n)
    return DiscreteDemand(pmf_d=pd, pmf_p=pp, unit=1.0)


def _peak_distribution_reference(route, dem):
    """Slow reference: the backward (A, G) recursion exactly as specified,
    with explicit enumeration of each (d, p) level pair per stop."""
    V = dem.levels
    L = len(route)
    M = L * (V - 1) + 1
    P = np.zeros((M, M))
    P[0, 0] = 1.0
    for c in reversed(list(route)):
        new = np.zeros((M, M))
        for a in range(M):
            for g in range(M):
                w0 = P[a, g]
                if w0 == 0.0:
                    continue
                for dv in range(V):
                    for pv in range(V):
                        w = w0 * dem.pmf_d[c][dv] * dem.pmf_p[c][pv]
                        if w == 0.0:
                            continue
                        new[a + dv, max(dv + a, pv + g)] += w
        P = new
    return P.sum(axis=0)


def _brute_pmf(route, dem):
    """Independent enumeration over all (d, p) level combinations (python)."""
    V = dem.levels
    L = len(route)
    out = np.zeros(L * (V - 1) + 1)
    custs = list(route)
    for combo in itertools.product(range(V), repeat=2 * L):
        d = {}
        p = {}
        w = 1.0
        for t, c in enumerate(custs):
            d[c] = combo[2 * t]
            p[c] = combo[2 * t + 1]
            w *= dem.pmf_d[c][d[c]] * dem.pmf_p[c][p[c]]
        out[int(peak_load(route, d, p))] += w
    return out


def _brute_pmf_np(route, dem):
    """Enumeration over all level combinations, vectorised over scenarios."""
    V = dem.levels
    L = len(route)
    K = 2 * L
    grid = np.indices((V,) * K).reshape(K, -1).T
    w = np.ones(grid.shape[0])
    d_lv = grid[:, 0::2]
    p_lv = grid[:, 1::2]
    for t, c in enumerate(route):
        w *= dem.pmf_d[c][d_lv[:, t]] * dem.pmf_p[c][p_lv[:, t]]
    load0 = d_lv.sum(axis=1)
    Lk = load0[:, None] - np.cumsum(d_lv, axis=1) + np.cumsum(p_lv, axis=1)
    peak = np.maximum(load0, Lk.max(axis=1)) if L else load0
    out = np.zeros(L * (V - 1) + 1)
    np.add.at(out, peak, w)
    return out


# ---------------------------------------------------------------
# peak_load
# ---------------------------------------------------------------


class TestPeakLoad:
    def test_hand_example(self):
        # route 0 -> 1 -> 2; L0 = 3+2+1 = 6; L1 = 3; L2 = 6; L3 = 9
        d = np.array([3, 2, 1])
        p = np.array([0, 5, 4])
        assert peak_load([0, 1, 2], d, p) == 9

    def test_empty_route(self):
        assert peak_load([], np.array([1.0]), np.array([2.0])) == 0

    def test_float_inputs(self):
        d = np.array([0.5, 1.5])
        p = np.array([2.0, 0.25])
        # L0 = 2.0 ; L1 = 3.5 ; L2 = 2.25
        assert peak_load([0, 1], d, p) == pytest.approx(3.5)


# ---------------------------------------------------------------
# DiscreteDemand validation
# ---------------------------------------------------------------


class TestDiscreteDemandValidation:
    def test_rows_must_sum_to_one(self):
        with pytest.raises(ValueError):
            DiscreteDemand(np.array([[0.5, 0.4]]), np.array([[0.5, 0.5]]), 1.0)

    def test_negative_entries_rejected(self):
        with pytest.raises(ValueError):
            DiscreteDemand(np.array([[1.5, -0.5]]), np.array([[0.5, 0.5]]), 1.0)

    def test_shape_mismatch_rejected(self):
        with pytest.raises(ValueError):
            DiscreteDemand(np.array([[0.5, 0.5]]), np.array([[1 / 3] * 3]), 1.0)


# ---------------------------------------------------------------
# peak_distribution: exact DP vs brute force, reference, Monte Carlo
# ---------------------------------------------------------------


def _brute_cases():
    rng = np.random.default_rng(20261012)
    cases = []
    for case in range(36):          # L = 1..3, V = 2..3
        L = 1 + case % 3
        V = 2 + (case // 3) % 2
        route = list(rng.permutation(4)[:L])
        cases.append((route, _random_dem(rng, 4, V), "py"))
    cases.append(([], _random_dem(rng, 3, 3), "py"))
    for k in range(12):             # L = 4, V = 2..3
        V = 2 + k % 2
        route = list(rng.permutation(5)[:4])
        cases.append((route, _random_dem(rng, 5, V), "np"))
    for k in range(10):             # L = 5, V = 2..3
        V = 2 + k % 2
        route = list(rng.permutation(6)[:5])
        cases.append((route, _random_dem(rng, 6, V), "np"))
    return cases


_CASES = _brute_cases()


@pytest.mark.parametrize("idx", range(len(_CASES)))
def test_peak_distribution_matches_brute_force(idx):
    route, dem, kind = _CASES[idx]
    pmf = peak_distribution(route, dem)
    ref = _brute_pmf(route, dem) if kind == "py" else _brute_pmf_np(route, dem)
    assert pmf.shape == ref.shape
    assert np.max(np.abs(pmf - ref)) < 1e-12
    assert pmf.sum() == pytest.approx(1.0, abs=1e-12)


def test_fast_matches_reference_recursion():
    rng = np.random.default_rng(31)
    for case in range(12):
        L = 1 + case % 5
        V = 2 + case % 7
        route = list(rng.permutation(L + 2)[:L])
        dem = _random_dem(rng, L + 2, V)
        fast = peak_distribution(route, dem)
        ref = _peak_distribution_reference(route, dem)
        assert np.max(np.abs(fast - ref)) < 1e-12


def test_peak_distribution_large_runs():
    # Size and normalisation only; no timing assertion.
    rng = np.random.default_rng(12)
    for V, L in ((16, 30), (32, 10)):
        dem = _random_dem(rng, L, V)
        pmf = peak_distribution(list(range(L)), dem)
        assert pmf.shape == (L * (V - 1) + 1,)
        assert np.all(pmf >= 0)
        assert pmf.sum() == pytest.approx(1.0, abs=1e-9)


@pytest.mark.parametrize("V", [16, 32])
def test_deterministic_long_route_closed_form(V):
    # Point-mass demands: F is deterministic and equals peak_load on the levels.
    rng = np.random.default_rng(V)
    L = 30
    dv = rng.integers(0, V, size=L)
    pv = rng.integers(0, V, size=L)
    dem = DiscreteDemand(
        pmf_d=np.eye(V)[dv], pmf_p=np.eye(V)[pv], unit=1.0
    )
    expected = peak_load(list(range(L)), dv, pv)
    pmf = peak_distribution(list(range(L)), dem)
    assert pmf.shape == (L * (V - 1) + 1,)
    assert pmf[expected] == pytest.approx(1.0, abs=1e-12)
    assert pmf.sum() == pytest.approx(1.0, abs=1e-12)


def test_peak_distribution_monte_carlo():
    rng = np.random.default_rng(7)
    n, V = 6, 4
    dem = _random_dem(rng, n, V)
    route = [2, 0, 5, 1, 4, 3]
    pmf = peak_distribution(route, dem)

    N = 200_000
    mc = np.random.default_rng(11)
    cd = np.stack([mc.choice(V, size=N, p=dem.pmf_d[c]) for c in range(n)], axis=1)
    cp = np.stack([mc.choice(V, size=N, p=dem.pmf_p[c]) for c in range(n)], axis=1)
    L0 = sum(cd[:, c] for c in route)
    load = L0.copy()
    peak = L0.copy()
    for c in route:
        load = load - cd[:, c] + cp[:, c]
        peak = np.maximum(peak, load)
    emp = np.bincount(peak, minlength=pmf.size) / N
    assert emp.size == pmf.size
    tv = 0.5 * np.abs(emp - pmf).sum()
    assert tv < 0.01


# ---------------------------------------------------------------
# chance and cvar consistency
# ---------------------------------------------------------------


class TestRiskMeasures:
    @pytest.fixture
    def case(self):
        rng = np.random.default_rng(3)
        dem = _random_dem(rng, 5, 3)
        route = [4, 1, 0]
        return route, dem, _brute_pmf(route, dem)

    def test_chance_matches_pmf_tail(self, case):
        route, dem, ref = case
        for Q in range(-1, ref.size + 2):
            expected = ref[Q + 1:].sum() if Q + 1 >= 0 else ref.sum()
            assert chance(route, dem, Q) == pytest.approx(expected, abs=1e-12)

    def test_chance_nonincreasing_in_Q(self, case):
        route, dem, _ = case
        vals = [chance(route, dem, Q) for Q in range(0, 8)]
        assert all(a >= b - 1e-15 for a, b in zip(vals, vals[1:]))

    def test_cvar_matches_definition(self, case):
        route, dem, ref = case
        levels = np.arange(ref.size)
        for alpha in (0.1, 0.5, 0.8, 0.95):
            cdf = np.cumsum(ref)
            var = int(np.nonzero(cdf >= alpha - 1e-12)[0][0])
            expected = var + np.sum(ref * np.maximum(levels - var, 0)) / (1 - alpha)
            assert cvar(route, dem, alpha) == pytest.approx(expected, abs=1e-12)

    def test_cvar_at_least_var(self, case):
        route, dem, ref = case
        cdf = np.cumsum(ref)
        for alpha in (0.05, 0.3, 0.6, 0.9, 0.99):
            var = int(np.nonzero(cdf >= alpha - 1e-12)[0][0])
            assert cvar(route, dem, alpha) >= var - 1e-12

    def test_cvar_monotone_in_alpha(self, case):
        route, dem, _ = case
        alphas = np.linspace(0.01, 0.99, 50)
        vals = [cvar(route, dem, a) for a in alphas]
        assert all(b >= a - 1e-12 for a, b in zip(vals, vals[1:]))

    def test_cvar_rejects_bad_alpha(self, case):
        route, dem, _ = case
        with pytest.raises(ValueError):
            cvar(route, dem, 1.0)
        with pytest.raises(ValueError):
            cvar(route, dem, 0.0)


def _single_customer_dem(pmf_d, pmf_p):
    """Single-customer route [0]: L0 = d_0 and L1 = p_0, so F = max(d_0, p_0)."""
    return DiscreteDemand(
        pmf_d=np.asarray([pmf_d], dtype=float),
        pmf_p=np.asarray([pmf_p], dtype=float),
        unit=1.0,
    )


class TestCvarEdges:
    def test_one_hot_pmf(self):
        pmf = np.zeros(6)
        pmf[4] = 1.0
        for alpha in (1e-9, 0.3, 0.5, 0.999, 1 - 1e-9):
            assert _cvar_from_pmf(pmf, alpha) == pytest.approx(4.0, abs=1e-12)

    def test_one_hot_route(self):
        # d = delta_0, p = delta_3 gives F = 3 almost surely
        dem = _single_customer_dem([1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0])
        for alpha in (0.2, 0.9):
            assert cvar([0], dem, alpha) == pytest.approx(3.0, abs=1e-12)
            assert chance([0], dem, 2) == pytest.approx(1.0)
            assert chance([0], dem, 3) == pytest.approx(0.0)

    def test_atom_exactly_at_alpha(self):
        # pmf [0.25, 0.25, 0.5]: cdf hits 0.5 exactly at level 1
        pmf = np.array([0.25, 0.25, 0.5])
        # VaR_0.5 = 1 (cdf(1) = 0.5 >= 0.5); tail E[(F-1)^+] = 0.5; CVaR = 1 + 0.5/0.5
        assert _cvar_from_pmf(pmf, 0.5) == pytest.approx(2.0, abs=1e-12)
        # via a route: F = max(d, p) with d = delta_0, p ~ pmf
        dem = _single_customer_dem([1.0, 0.0, 0.0], pmf)
        assert cvar([0], dem, 0.5) == pytest.approx(2.0, abs=1e-12)

    def test_alpha_near_zero_gives_mean(self):
        pmf = np.array([0.0, 0.0, 0.3, 0.0, 0.7])
        mean = 2 * 0.3 + 4 * 0.7
        # smallest level with cdf >= 1e-9 is 2; CVaR -> mean as alpha -> 0
        assert _cvar_from_pmf(pmf, 1e-9) == pytest.approx(mean, abs=1e-6)

    def test_alpha_near_one_gives_max_level(self):
        pmf = np.array([0.5, 0.0, 0.5 - 1e-6, 0.0, 1e-6])
        # VaR at alpha = 1 - 1e-9 is the top support level 4
        assert _cvar_from_pmf(pmf, 1 - 1e-9) == pytest.approx(4.0, abs=1e-6)

    def test_no_index_qualifies_falls_back_to_last_level(self):
        # total mass short of 1 by more than the tolerance
        pmf = np.array([0.2, 0.2, 0.2])
        val = _cvar_from_pmf(pmf, 0.99)
        assert np.isfinite(val)
        assert val >= 2.0 - 1e-12


# ---------------------------------------------------------------
# Insertion lemma (scenario-wise)
# ---------------------------------------------------------------


def test_insertion_lemma_scenariowise():
    rng = np.random.default_rng(99)
    checked = 0
    while checked < 3000:
        L = int(rng.integers(0, 6))
        n = L + 1
        customers = list(rng.permutation(n))
        j = customers.pop()
        route = customers
        pos = int(rng.integers(1, L + 2))
        new_route = route[: pos - 1] + [j] + route[pos - 1:]
        d = rng.uniform(0.0, 10.0, size=n)
        p = rng.uniform(0.0, 10.0, size=n)
        if rng.random() < 0.3:  # integer-valued and zero demands too
            d = np.floor(d)
            p = np.floor(p) * (rng.random(n) < 0.7)
        F_r = peak_load(route, d, p)
        F_new = peak_load(new_route, d, p)
        m = max(d[j], p[j])
        assert F_r <= F_new + 1e-12
        assert F_new <= F_r + m + 1e-12
        checked += 1
    assert checked >= 2000


# ---------------------------------------------------------------
# insertion_window
# ---------------------------------------------------------------


def _window_cases():
    rng = np.random.default_rng(5)
    cases = []
    for k in range(20):
        L = k % 4              # 0..3
        V = 2 if L == 3 else 3  # 3**8 = 6561 enumerations at most
        n = 5
        perm = list(rng.permutation(n))
        route = perm[:L]
        j = perm[L]
        pos = int(rng.integers(1, L + 2))
        Q = int(rng.integers(0, max(L * (V - 1), 1) + 1))
        cases.append((route, j, pos, _random_dem(rng, n, V), Q))
    return cases


def _enumerate_window(route, new_route, j, dem, Q):
    """Enumeration of P(F_r <= Q < F_r') with the per-scenario insertion
    bound, plus the enumeration of the flip event directly."""
    custs = sorted(set(route) | {j})
    V = dem.levels
    total = 0.0
    bound = 0.0
    flip = 0.0
    for combo in itertools.product(range(V), repeat=2 * len(custs)):
        d = {}
        p = {}
        w = 1.0
        for t, c in enumerate(custs):
            d[c] = combo[2 * t]
            p[c] = combo[2 * t + 1]
            w *= dem.pmf_d[c][d[c]] * dem.pmf_p[c][p[c]]
        Fr = peak_load(route, d, p)
        Fn = peak_load(new_route, d, p)
        m = max(d[j], p[j])
        if Q - m < Fr <= Q and Fn > Q:
            total += w
        if Q - (V - 1) < Fr <= Q:
            bound += w
        if (Fn > Q) != (Fr > Q):
            flip += w
    return total, bound, flip


@pytest.mark.parametrize("idx", range(20))
def test_insertion_window_matches_enumeration(idx):
    route, j, pos, dem, Q = _window_cases()[idx]
    new_route = route[: pos - 1] + [j] + route[pos - 1:]
    total, bound, flip = _enumerate_window(route, new_route, j, dem, Q)
    got = insertion_window(route, j, pos, dem, Q)
    assert got == pytest.approx(total, abs=1e-12)
    assert got == pytest.approx(flip, abs=1e-12)
    assert got <= bound + 1e-12


def test_insertion_window_uses_route_insert():
    """Position convention checked against the real Route class."""
    rng = np.random.default_rng(77)
    for case in range(8):
        L = case % 4
        V = 3 if L <= 2 else 2
        n = 5
        perm = list(rng.permutation(n))
        route_list = perm[:L]
        j = perm[L]
        pos = int(rng.integers(1, L + 2))
        dem = _random_dem(rng, n, V)
        Q = int(rng.integers(0, max(L * (V - 1), 1) + 1))

        r = Route(route_list)
        r.insert(j, pos=pos)
        new_seq = r.customers
        assert new_seq[pos - 1] == j

        custs = sorted(set(route_list) | {j})
        expected = 0.0
        for combo in itertools.product(range(V), repeat=2 * len(custs)):
            d = {}
            p = {}
            w = 1.0
            for t, c in enumerate(custs):
                d[c] = combo[2 * t]
                p[c] = combo[2 * t + 1]
                w *= dem.pmf_d[c][d[c]] * dem.pmf_p[c][p[c]]
            Fr = peak_load(route_list, d, p)
            Fn = peak_load(new_seq, d, p)
            if (Fn > Q) != (Fr > Q):
                expected += w
        got = insertion_window(route_list, j, pos, dem, Q)
        assert got == pytest.approx(expected, abs=1e-12)


def test_insertion_window_rejects_large_case():
    rng = np.random.default_rng(1)
    dem = _random_dem(rng, 12, 8)
    with pytest.raises(ValueError):
        insertion_window(list(range(10)), 11, 1, dem, 3)


# ---------------------------------------------------------------
# Discretisation
# ---------------------------------------------------------------


class TestDiscretizeSamples:
    def test_default_unit_maps_max_to_top_level(self):
        s = np.array([0.0, 1.0, 2.5, 7.0])
        pmf, unit = discretize_samples(s, b=3)
        assert unit == pytest.approx(1.0)  # 7 / (8 - 1)
        assert pmf.shape == (8,)
        assert pmf.sum() == pytest.approx(1.0)
        # 0 -> 0, 1 -> 1, 2.5 -> rint(2.5)=2 (ties to even), 7 -> 7
        assert pmf[[0, 1, 2, 7]] == pytest.approx([0.25] * 4)

    def test_explicit_unit_and_clipping(self):
        s = np.array([0.4, 3.0, 100.0])
        pmf, unit = discretize_samples(s, b=2, unit=2.0)
        assert unit == 2.0
        assert pmf.shape == (4,)
        assert pmf.sum() == pytest.approx(1.0)
        # 0.2 -> 0, 1.5 -> 2, 50 -> clipped to 3
        assert pmf == pytest.approx([1 / 3, 0, 1 / 3, 1 / 3])

    def test_all_zero_samples_unit_one(self):
        pmf, unit = discretize_samples(np.zeros(10), b=2)
        assert unit == 1.0
        assert pmf == pytest.approx([1.0, 0.0, 0.0, 0.0])

    def test_gamma_samples_sum_to_one(self):
        rng = np.random.default_rng(0)
        s = rng.gamma(shape=25.0, scale=0.04, size=5000)
        pmf, unit = discretize_samples(s, b=4)
        assert pmf.shape == (16,)
        assert pmf.sum() == pytest.approx(1.0)
        assert unit == pytest.approx(s.max() / 15)


class TestDiscretizeInstance:
    @pytest.fixture
    def toy_instance(self, tmp_path):
        p = tmp_path / "toy.txt"
        p.write_text(
            "5\n40\n"
            "50 50 0 0\n"
            "20 30 10 5\n"
            "80 70 8 12\n"
            "30 80 15 3\n"
            "70 20 6 14\n"
            "60 60 9 7\n"
        )
        return Instance.from_file(p)

    def test_from_generated_scenarios(self, toy_instance):
        inst = toy_instance
        X = generate_scenarios(
            inst, n_scenarios=500, config=ScenarioConfig("gamma", cv=0.2, seed=1)
        )
        b = 3
        dem = discretize_instance(X, n=inst.n, b=b)
        V = 2 ** b
        assert dem.pmf_d.shape == (inst.n, V)
        assert dem.pmf_p.shape == (inst.n, V)
        assert dem.unit == pytest.approx(X.max() / (V - 1))
        assert np.allclose(dem.pmf_d.sum(axis=1), 1.0)
        assert np.allclose(dem.pmf_p.sum(axis=1), 1.0)

        # column j of X feeds row j, with the common unit
        pmf0, _ = discretize_samples(X[:, 0], b, unit=dem.unit)
        assert np.allclose(dem.pmf_d[0], pmf0)
        pmf1, _ = discretize_samples(X[:, 3], b, unit=dem.unit)
        assert np.allclose(dem.pmf_p[1], pmf1)

        pmf = peak_distribution([0, 2, 4], dem)
        assert pmf.sum() == pytest.approx(1.0)

    def test_synthetic_gamma_common_unit(self):
        rng = np.random.default_rng(4)
        n, b = 4, 2
        X = rng.gamma(shape=4.0, scale=2.0, size=(800, 2 * n))
        dem = discretize_instance(X, n=n, b=b)
        assert dem.unit == pytest.approx(X.max() / 3)
        assert dem.pmf_d.shape == (n, 4)
        assert np.allclose(dem.pmf_d.sum(axis=1), 1.0)
        assert np.allclose(dem.pmf_p.sum(axis=1), 1.0)
        dem2 = discretize_instance(X, n=n, b=b, unit=1.5)
        assert dem2.unit == 1.5
        assert np.allclose(dem2.pmf_d.sum(axis=1), 1.0)

    def test_wrong_width_rejected(self):
        with pytest.raises(ValueError):
            discretize_instance(np.zeros((5, 7)), n=3, b=2)
