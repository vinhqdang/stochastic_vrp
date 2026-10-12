"""Tests for quantum/experiments.py (QUASAR M3, design Revision 2): gamma level
pmfs and their binning, the copula conditional pmfs, the discretisation rule,
the copula discrete window probability, insertion and intra-route move
statistics, delta-method errors, query counts, and the M3a v2 worker. Fast
(well under 60 s)."""

from __future__ import annotations

import importlib.util
import json
import math
import re
from pathlib import Path

import numpy as np
import pytest
import scipy.stats as st

from quantum.classical import DiscreteDemand, chance, insertion_window, peak_load
from quantum.experiments import (
    DEFAULT_CV,
    DEFAULT_DIST,
    DEFAULT_RHO,
    canonical_ae_M,
    chance_copula,
    classical_queries,
    conditional_level_pmf,
    copula_demand,
    gamma_level_pmf,
    gh_nodes,
    insertion_candidates,
    insertion_envelopes,
    insertion_peak,
    instance_demand,
    intra_candidates,
    level_rule,
    load_profile,
    move_stats,
    q999_max,
    quantum_queries,
    signed_costs,
    window_stats,
)

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "results" / "plans" / "CMT1X.json"
SCRIPTS = ROOT / "scripts"


def _import_dethloff():
    import sys

    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    import dethloff_runner

    return dethloff_runner


def _load_driver():
    spec = importlib.util.spec_from_file_location("run_quasar_m3", SCRIPTS / "run_quasar_m3.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _synthetic_plan(name="SYN", Q=25.0):
    return {"name": name, "n": 4, "Q": Q, "dbar": [0.0, 10.0, 8.0, 12.0],
            "pbar": [0.0, 4.0, 6.0, 3.0], "res": {}}


def _enumerated_scenarios(n_cust: int, V: int, seed_levels: int = 0):
    """All demand combinations of customers 1..n_cust (uniform pmfs on V levels).

    Returns (dem, dsc, psc): dem is the uniform DiscreteDemand, dsc/psc are
    (n, N) customer-major level arrays with N = V ** (2 n_cust) equal-weight
    scenarios (the depot row is zero)."""
    n = n_cust + 1
    unif = np.full((n, V), 1.0 / V)
    unif[0] = 0.0
    unif[0, 0] = 1.0
    dem = DiscreteDemand(pmf_d=unif, pmf_p=unif.copy(), unit=1.0)
    grid = np.indices((V,) * (2 * n_cust), dtype=np.int64).reshape(2 * n_cust, -1)
    dsc = np.zeros((n, grid.shape[1]))
    psc = np.zeros((n, grid.shape[1]))
    for c in range(1, n):
        dsc[c] = grid[2 * (c - 1)]
        psc[c] = grid[2 * (c - 1) + 1]
    return dem, dsc, psc


# ------------------------------------------------------------ gamma binning

@pytest.mark.parametrize("mean,cv,unit,V", [(10.0, 0.3, 1.0, 16), (3.2, 0.3, 0.5, 8), (50.0, 0.3, 4.0, 32)])
def test_gamma_level_pmf_normalised(mean, cv, unit, V):
    pmf = gamma_level_pmf(mean, cv, unit, V)
    assert pmf.shape == (V,)
    assert np.all(pmf >= 0)
    assert pmf.sum() == pytest.approx(1.0, abs=1e-12)


def test_gamma_level_pmf_mean_and_sheppard_variance():
    """Away from the boundaries, nearest-level binning keeps the mean (bias < u/10)
    and inflates the variance by about u^2/12 (Sheppard)."""
    mean, cv, unit, V = 10.0, 0.3, 1.0, 64
    pmf = gamma_level_pmf(mean, cv, unit, V)
    lev = np.arange(V)
    m_lvl = float(pmf @ lev) * unit
    v_lvl = (float(pmf @ (lev**2)) - float(pmf @ lev) ** 2) * unit * unit
    assert abs(m_lvl - mean) < unit / 10
    gamma_var = (cv * mean) ** 2
    assert v_lvl - gamma_var == pytest.approx(unit * unit / 12.0, rel=0.05)


def test_gamma_level_pmf_bins_match_cdf():
    mean, cv, unit, V = 10.0, 0.3, 1.0, 16
    k, scale = 1 / cv**2, mean * cv**2
    pmf = gamma_level_pmf(mean, cv, unit, V)
    assert pmf[0] == pytest.approx(st.gamma.cdf(0.5, k, scale=scale), rel=1e-10)
    assert pmf[5] == pytest.approx(
        st.gamma.cdf(5.5, k, scale=scale) - st.gamma.cdf(4.5, k, scale=scale), rel=1e-10
    )
    assert pmf[V - 1] == pytest.approx(1 - st.gamma.cdf(V - 1.5, k, scale=scale), rel=1e-10)


def test_gamma_level_pmf_zero_mean_is_point_mass():
    pmf = gamma_level_pmf(0.0, 0.3, 2.0, 8)
    assert pmf[0] == 1.0 and pmf[1:].sum() == 0.0


def test_default_constants_match_runner():
    dr = _import_dethloff()
    txt = (SCRIPTS / "dethloff_runner.py").read_text()
    assert float(re.search(r"^CV\s*=\s*([0-9.]+)", txt, re.M).group(1)) == DEFAULT_CV
    assert re.search(r'^DIST\s*=\s*"(\w+)"', txt, re.M).group(1) == DEFAULT_DIST
    assert dr.RHO == pytest.approx(DEFAULT_RHO)


# ------------------------------------------------------------ copula conditional pmfs

@pytest.mark.parametrize("nodes", [10, 16])
def test_conditional_pmf_sums_to_one_and_mixes_to_marginal(nodes):
    mean, cv, unit, V, rho = 10.0, 0.3, 1.0, 32, 0.6
    z, w = gh_nodes(nodes)
    cond = conditional_level_pmf(mean, cv, unit, V, z, rho)
    assert cond.shape == (nodes, V)
    assert np.all(cond >= 0)
    np.testing.assert_allclose(cond.sum(axis=1), 1.0, atol=1e-12)
    marg = gamma_level_pmf(mean, cv, unit, V)
    mix = w @ cond
    tol = 1e-4 if nodes == 10 else 1e-6
    assert np.abs(mix - marg).max() < tol


def test_conditional_pmf_rho_zero_is_marginal():
    z, _ = gh_nodes(7)
    cond = conditional_level_pmf(6.0, 0.3, 0.5, 16, z, 0.0)
    marg = gamma_level_pmf(6.0, 0.3, 0.5, 16)
    np.testing.assert_allclose(cond, np.tile(marg, (z.size, 1)), atol=1e-12)


def test_conditional_pmf_point_mass_for_zero_mean():
    z, _ = gh_nodes(5)
    cond = conditional_level_pmf(0.0, 0.3, 1.0, 8, z, 0.6)
    assert np.all(cond[:, 0] == 1.0) and np.all(cond[:, 1:] == 0.0)


def test_gh_nodes_normalised_and_moments():
    z, w = gh_nodes(10)
    assert w.sum() == pytest.approx(1.0, abs=1e-14)
    assert float(w @ z) == pytest.approx(0.0, abs=1e-12)
    assert float(w @ (z * z)) == pytest.approx(1.0, abs=1e-10)


# ------------------------------------------------------------ discretisation rule

def test_level_rule_half_level_threshold():
    Q, V, q_max = 160.0, 16, 97.0
    K, unit = level_rule(Q, V, q_max)
    x = Q * (V - 1) / q_max
    assert K + 0.5 >= x                        # the smallest such integer ...
    assert K - 1 + 0.5 < x or K == 1          # ... so K - 1 would fail
    assert unit == pytest.approx(Q / (K + 0.5))
    assert Q / unit == pytest.approx(K + 0.5)  # threshold on a half-level boundary


def test_instance_demand_threshold_is_half_level():
    plan = json.loads(PLAN.read_text())
    for b in (3, 4, 5):
        dem, unit, Ql = instance_demand(plan, b)
        n, V = plan["n"], 2**b
        assert dem.pmf_d.shape == (n, V) and dem.pmf_p.shape == (n, V)
        assert dem.levels == V and dem.unit == pytest.approx(unit)
        assert isinstance(Ql, int)
        assert Ql + 0.5 == pytest.approx(plan["Q"] / unit, rel=1e-12)
        assert dem.pmf_d[0, 0] == 1.0 and dem.pmf_p[0, 0] == 1.0
        K, u2 = level_rule(plan["Q"], V, q999_max(np.array(plan["dbar"]), np.array(plan["pbar"])))
        assert (Ql, unit) == (K, pytest.approx(u2))


def test_discrete_copula_converges_to_monte_carlo():
    """The copula discrete window probability approaches the continuous copula
    Monte Carlo as b grows (synthetic 3-customer route, N = 4e5)."""
    dr = _import_dethloff()
    plan = _synthetic_plan()
    route = [1, 2, 3]
    N = 400_000
    rng = np.random.default_rng(5)
    d = dr.sample_demands(np.array(plan["dbar"]), 4, N, dr.CV, dr.DIST, rng, rho=0.6)
    p = dr.sample_demands(np.array(plan["pbar"]), 4, N, dr.CV, dr.DIST, rng, rho=0.6)
    from quantum.experiments import load_profile as lp

    flags = lp(route, np.ascontiguousarray(d.T), np.ascontiguousarray(p.T)).max(0) > plan["Q"]
    p_mc = float(flags.mean())
    se = math.sqrt(p_mc * (1 - p_mc) / N)
    errs = []
    for b in (3, 4, 5, 6):
        cd = copula_demand(plan, b, 0.6, 10)
        errs.append(abs(chance_copula(route, cd) - p_mc))
    assert errs[-1] < 4 * se + 0.002
    assert errs[-1] <= errs[0] + 1e-12


def test_copula_rho_zero_matches_independent_discrete_model():
    plan = _synthetic_plan()
    route = [1, 2, 3]
    cd = copula_demand(plan, 4, 0.0, 4)
    dem, unit, Ql = instance_demand(plan, 4)
    assert cd.Q_level == Ql and cd.unit == pytest.approx(unit)
    assert chance_copula(route, cd) == pytest.approx(chance(route, dem, Ql), abs=1e-12)


# ------------------------------------------------------------ insertion peaks

def test_insertion_peak_matches_brute_force_on_enumeration():
    dem, dsc, psc = _enumerated_scenarios(3, 3)
    route = [1, 2]
    prof = load_profile(route, dsc, psc)
    PM, SM = insertion_envelopes(prof)
    for j in (3,):
        for pos in (1, 2, 3):
            new = route[: pos - 1] + [j] + route[pos - 1:]
            got = insertion_peak(PM, SM, pos, dsc[j], psc[j])
            want = np.array([peak_load(new, dsc[:, s], psc[:, s]) for s in range(dsc.shape[1])])
            np.testing.assert_allclose(got, want, atol=1e-12)


def test_insertion_lemma_scenario_wise_on_random_draws():
    """Inserting a customer raises the peak by between 0 and max(d_j, p_j)."""
    dr = _import_dethloff()
    plan = _synthetic_plan()
    rng = np.random.default_rng(3)
    N = 5000
    d = dr.sample_demands(np.array(plan["dbar"]), 4, N, dr.CV, dr.DIST, rng, rho=0.6)
    p = dr.sample_demands(np.array(plan["pbar"]), 4, N, dr.CV, dr.DIST, rng, rho=0.6)
    dsc, psc = np.ascontiguousarray(d.T), np.ascontiguousarray(p.T)
    route = [2, 1]
    prof = load_profile(route, dsc, psc)
    base = prof.max(0)
    PM, SM = insertion_envelopes(prof)
    for pos in (1, 2, 3):
        new_peak = insertion_peak(PM, SM, pos, dsc[3], psc[3])
        diff = new_peak - base
        assert diff.min() >= -1e-9
        assert np.all(diff <= np.maximum(dsc[3], psc[3]) + 1e-9)


def test_window_stats_insertion_matches_enumeration():
    """Insertion statistics on a uniform enumeration against exact discrete values."""
    dem, dsc, psc = _enumerated_scenarios(3, 3)
    route, j, Q = [1, 2], 3, 3
    for pos in (1, 2, 3):
        prof = load_profile(route, dsc, psc)
        y = prof.max(0) > Q
        new = route[: pos - 1] + [j] + route[pos - 1:]
        x = load_profile(new, dsc, psc).max(0) > Q
        st_ = window_stats(y, x)
        assert st_["p"] == pytest.approx(chance(route, dem, Q), abs=1e-12)
        assert st_["p_new"] == pytest.approx(chance(new, dem, Q), abs=1e-12)
        assert st_["P_Dm1"] == 0.0
        assert st_["delta"] == pytest.approx(st_["p_new"] - st_["p"], abs=1e-12)
        iw = insertion_window(route, j, pos, dem, Q)
        assert st_["var_D"] == pytest.approx(iw - st_["delta"] ** 2, abs=1e-12)


def test_window_stats_intra_matches_brute_force():
    """Signed D (swap of two stops) against enumeration of peak_load."""
    dem, dsc, psc = _enumerated_scenarios(3, 3)
    route, Q = [1, 2, 3], 4
    swapped = [2, 1, 3]
    y = np.array([peak_load(route, dsc[:, s], psc[:, s]) > Q for s in range(dsc.shape[1])])
    x = np.array([peak_load(swapped, dsc[:, s], psc[:, s]) > Q for s in range(dsc.shape[1])])
    st_ = window_stats(y, x)
    D = x.astype(int) - y.astype(int)
    assert st_["P_D1"] == pytest.approx(np.mean(D == 1))
    assert st_["P_Dm1"] == pytest.approx(np.mean(D == -1))
    assert st_["P_Dm1"] > 0 or st_["P_D1"] > 0  # the case is not trivial
    assert st_["var_D"] == pytest.approx(np.mean(D * D) - np.mean(D) ** 2)
    assert st_["rho_q"] == pytest.approx(math.sqrt(st_["var_D"]) / math.sqrt(
        st_["p_new"] * (1 - st_["p_new"])))


def test_window_stats_delta_method_se_matches_bootstrap():
    rng = np.random.default_rng(11)
    N = 20000
    y = rng.random(N) < 0.3
    flip = rng.random(N) < 0.08
    x = y | (flip & ~y)                       # insertion-like: D in {0, 1}
    st_ = window_stats(y, x)
    boots = []
    for _ in range(300):
        idx = rng.integers(0, N, N)
        boots.append(window_stats(y[idx], x[idx])["rho_q"])
    boot_se = float(np.std(boots, ddof=1))
    assert st_["se_rho_q"] == pytest.approx(boot_se, rel=0.25)
    assert st_["se_rho_c"] > 0


def test_window_stats_nan_and_zero_variance():
    y = np.zeros(100, dtype=bool)
    y[:30] = True
    s = window_stats(y, y.copy())             # D = 0 everywhere, 0 < p' < 1
    assert s["zero"] and s["var_D"] == 0.0 and s["rho_q"] == 0.0
    assert math.isnan(s["se_rho_q"]) and s["rho_c"] == 0.0
    all_zero = window_stats(np.zeros(100, bool), np.zeros(100, bool))
    assert math.isnan(all_zero["rho_q"])      # p' = 0 gives NaN ratios
    s1 = window_stats(np.zeros(10, bool), np.ones(10, bool))
    assert math.isnan(s1["rho_q"]) and math.isnan(s1["rho_c"])


# ------------------------------------------------------------ move candidates

def test_insertion_candidates_count_and_exclusion():
    rng = np.random.default_rng(0)
    pool = {3, 4, 5, 6}
    route = [1, 2]
    cands = insertion_candidates(pool, route, cap=1000, rng=rng)
    assert len(cands) == 4 * 3
    assert len(set(cands)) == 12
    assert all(j in pool and 1 <= pos <= 3 for j, pos in cands)


def test_insertion_candidates_cap_reproducible():
    pool = set(range(1, 30))
    route = [30, 31, 32]
    a = insertion_candidates(pool, route, 7, np.random.default_rng(3))
    b = insertion_candidates(pool, route, 7, np.random.default_rng(3))
    assert len(a) == 7 and a == b


def test_intra_candidates_all_distinct_and_neighbours():
    rng = np.random.default_rng(0)
    route = (4, 7, 1, 9, 3)
    moves, total = intra_candidates(route, 200, rng, all_upto=12)
    assert total == len(moves)
    new_routes = [t for _, _, _, t in moves]
    assert len(set(new_routes)) == len(new_routes)
    assert route not in new_routes
    for t in new_routes:
        assert sorted(t) == sorted(route)
    kinds = {k for k, _, _, _ in moves}
    assert kinds == {"relocate", "swap", "2opt"}


def test_intra_candidates_subsample_long_routes():
    rng = np.random.default_rng(1)
    route = tuple(range(1, 16))
    moves, total = intra_candidates(route, 50, rng, all_upto=12)
    assert len(moves) == 50 and total > 50


# ------------------------------------------------------------ move_stats (discrete, part b)

def _tiny_demand(seed: int = 1, n: int = 5, V: int = 3) -> DiscreteDemand:
    rng = np.random.default_rng(seed)
    pd_ = rng.random((n, V)) + 0.05
    pp = rng.random((n, V)) + 0.05
    pd_ /= pd_.sum(axis=1, keepdims=True)
    pp /= pp.sum(axis=1, keepdims=True)
    return DiscreteDemand(pmf_d=pd_, pmf_p=pp, unit=1.0)


@pytest.mark.parametrize("pos", [1, 2, 3])
@pytest.mark.parametrize("Q", [1, 2, 3, 4])
def test_move_stats_matches_insertion_window(pos, Q):
    dem = _tiny_demand()
    s = move_stats([0, 1], 2, pos, dem, Q)
    assert s["delta"] >= -1e-12
    assert s["delta"] == pytest.approx(insertion_window([0, 1], 2, pos, dem, Q), abs=1e-12)


def test_move_stats_rho_formulas_and_nan():
    dem = _tiny_demand(seed=4)
    s = move_stats([0, 1], 3, 2, dem, 2)
    d, pn = s["delta"], s["p_new"]
    if 0 < pn < 1:
        assert s["rho_q"] == pytest.approx(math.sqrt(d * (1 - d)) / math.sqrt(pn * (1 - pn)))
        assert s["rho_c"] == pytest.approx(d * (1 - d) / (pn * (1 - pn)))
    big = move_stats([0, 1], 3, 2, dem, 10**6)
    assert big["p_new"] == 0.0 and math.isnan(big["rho_q"]) and math.isnan(big["rho_c"])


def test_move_stats_rejects_bad_input():
    dem = _tiny_demand()
    with pytest.raises(ValueError):
        move_stats([0, 1], 1, 1, dem, 2)
    with pytest.raises(IndexError):
        move_stats([0, 1], 2, 4, dem, 2)


# ------------------------------------------------------------ query counts

def test_canonical_ae_M_is_smallest_power_of_two():
    for var, eps in [(0.25, 1e-3), (0.01, 1e-2), (0.2, 1e-4), (0.0, 1e-3)]:
        M = canonical_ae_M(var, eps)
        assert M & (M - 1) == 0
        bound = lambda m: 2 * math.pi * math.sqrt(var) / m + math.pi**2 / m**2
        assert bound(M) <= eps
        if M > 1:
            assert bound(M // 2) > eps


def test_canonical_ae_M_zero_variance_floor():
    """With var = 0 only the pi^2/M^2 term remains: M ~ pi / sqrt(eps)."""
    eps = 1e-3
    M = canonical_ae_M(0.0, eps)
    assert M == 2 ** math.ceil(math.log2(math.pi / math.sqrt(eps)))


def test_quantum_and_classical_query_formulas():
    var, eps = 0.25, 1e-3
    M = canonical_ae_M(var, eps)
    assert quantum_queries(var, eps) == pytest.approx(5 * (2 * M + 1))
    assert quantum_queries(var, eps, reps=1) == pytest.approx(2 * M + 1)
    assert classical_queries(var, eps) == pytest.approx(1.96**2 * var / eps**2)
    assert classical_queries(0.0, eps) == 0.0
    assert quantum_queries(var, 1e-4) > quantum_queries(var, 1e-3)
    with pytest.raises(ValueError):
        classical_queries(0.2, 0.0)


def test_signed_costs_are_two_estimates_at_half_eps():
    P1, Pm1, eps = 0.03, 0.02, 1e-3
    q, c = signed_costs(P1, Pm1, eps)
    assert q == pytest.approx(quantum_queries(P1 * (1 - P1), eps / 2) + quantum_queries(Pm1 * (1 - Pm1), eps / 2))
    assert c == pytest.approx(classical_queries(P1 * (1 - P1), eps / 2) + classical_queries(Pm1 * (1 - Pm1), eps / 2))


# ------------------------------------------------------------ driver

def test_draw_scenarios_equals_make_scenarios():
    dr = _import_dethloff()
    drv = _load_driver()
    plan = _synthetic_plan()
    N = 3000
    dsc, psc = drv.draw_scenarios(plan, N, dr.RHO, dr.SEED)
    d_ref, p_ref = dr.make_scenarios(np.array(plan["dbar"]), np.array(plan["pbar"]), N, dr.CV, dr.DIST, dr.SEED)
    np.testing.assert_array_equal(dsc, d_ref.T)
    np.testing.assert_array_equal(psc, p_ref.T)


def test_worker_identities_on_cmt1x(tmp_path):
    if not PLAN.exists():
        pytest.skip("CMT1X plan not available")
    drv = _load_driver()
    out = drv._work_v2((str(PLAN), DEFAULT_RHO, 2000, 40, tmp_path))
    assert out["n_routes"] >= 1
    import pandas as pd

    d = pd.read_pickle(tmp_path / "CMT1X.pkl")
    mv, rt = d["moves"], d["routes"]
    assert not ((mv["cls"] == 1) & (mv["P_Dm1"] > 0)).any()
    ins = mv[mv["cls"] == 1]
    assert (ins["delta"] >= -1e-12).all() and (ins["P_D1"] >= 0).all()
    intra = mv[mv["cls"] == 2]
    assert len(intra) > 0 and (intra["P_Dm1"] > 0).any()
    # Route-level stand-alone totals are sums over classes 1 and 2 of the move-level counts.
    for r in rt.itertuples():
        rows = mv[(mv["route_id"] == r.route_id) & (mv["cls"].isin([1, 2]))]
        q = sum(quantum_queries(pn * (1 - pn), 1e-3) for pn in rows["p_new"])
        assert q == pytest.approx(r.sa_Nq_e3, rel=1e-9)
        assert len(rows) == r.n_c1 + r.n_c2
    # Incremental totals are base once plus increments.
    assert (rt["inc_Nq_e3"] > 0).all()


def test_level_rule_nearest_variant():
    """Review variant: K = round(x - 1/2), unit = Q / (K + 1/2), K at least 1."""
    V, q_max = 16, 1.0
    for Q in (2.2, 4.7, 7.3, 8.5, 12.9):
        x = Q * (V - 1) / q_max
        K, unit = level_rule(Q, V, q_max, "nearest")
        assert K == max(1, round(x - 0.5))
        assert unit == pytest.approx(Q / (K + 0.5))
        Kd, _ = level_rule(Q, V, q_max, "design")
        assert Kd + 0.5 >= x and Kd - 0.5 < x or Kd == 1
    with pytest.raises(ValueError):
        level_rule(1.0, V, q_max, "floor")
