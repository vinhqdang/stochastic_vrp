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
    CopulaDemand,
    bin_averaged_nodes,
    canonical_ae_M,
    chance_copula,
    crn_factor_grid,
    classical_queries,
    conditional_level_pmf,
    copula_demand,
    gamma_level_pmf,
    gh_nodes,
    equal_mass_grid,
    factor_grid,
    tail_over_pickups,
    classical_queries_certified,
    classical_queries_certified_array,
    length_label,
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
    ALT_DIST_LABELS,
    alt_distance_label,
    band_label,
    canonical_ae_M_array,
    class_costs,
    class_costs_array,
    classical_queries_array,
    end_load_bound,
    end_load_violations,
    gate_distance,
    paired_level_check,
    plug_in_variance,
    quantise_levels,
    quantum_queries_array,
    reconstruct_intra,
    route_bootstrap_quantile,
    route_peak,
    rho_q_upper,
    rule_of_three_bound,
    sum_loads,
    weighted_quantile_sorted,
    PPRIME_BANDS,
    GATE_BANDS,
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


# ============================================================
# M3a v3 interpretation layer
# ============================================================


def _toy_scenarios(n, levels=3):
    """All equal-weight scenarios of n customers with integer d_i, p_i uniform on 0..levels-1."""
    grid = np.array(np.meshgrid(*([np.arange(levels)] * (2 * n)), indexing="ij")).reshape(2 * n, -1)
    dsc = grid[:n].astype(np.float64)
    psc = grid[n:].astype(np.float64)
    return dsc, psc


def test_end_load_bound_dominates_observed_p_ne0_on_toy_route():
    """Exact toy: 3 customers, levels 0..2, all intra moves. D != 0 implies both end
    loads <= Q (zero violations), and the union end-load bound is >= P(D != 0) per move."""
    from quantum.experiments import intra_candidates

    dsc, psc = _toy_scenarios(3)
    Q = 3.0
    route = [0, 1, 2]
    y = route_peak(route, dsc, psc) > Q
    end_ok = (sum_loads(route, dsc) <= Q) & (sum_loads(route, psc) <= Q)
    moves, _ = intra_candidates(route, cap=200, rng=np.random.default_rng(0), all_upto=12)
    xs, p_ne0 = [], []
    for kind, i, k, t in moves:
        x = route_peak(list(t), dsc, psc) > Q
        D = x.astype(np.int8) - y.astype(np.int8)
        assert end_load_violations(D, end_ok) == 0
        xs.append(x)
        p_ne0.append(float(np.mean(D != 0)))
    bound = end_load_bound(end_ok, y, xs)
    assert max(p_ne0) > 0.0  # the toy is not trivially zero
    assert bound["B_union"] >= max(p_ne0) - 1e-12
    assert bound["B_union"] >= bound["B_spec"] - 1e-12
    assert 0.0 <= bound["B_spec"] <= 1.0


def test_end_load_violation_counter_flags_bad_scenarios():
    D = np.array([0, 1, -1, 0, 1])
    end_ok = np.array([True, True, False, True, False])
    assert end_load_violations(D, end_ok) == 2
    assert end_load_violations(np.zeros(5, dtype=int), np.zeros(5, dtype=bool)) == 0


def test_end_load_bound_shape_checks():
    with pytest.raises(ValueError):
        end_load_bound(np.array([True]), np.array([True, False]))
    with pytest.raises(ValueError):
        end_load_bound(np.array([True, True]), np.array([True, False]), [np.array([True])])


def test_reconstruct_intra_inverts_candidate_generation():
    route = [4, 1, 7, 3, 9]
    moves, _ = intra_candidates(route, cap=10_000, rng=np.random.default_rng(1), all_upto=12)
    assert len(moves) > 0
    for kind, i, k, t in moves:
        assert tuple(reconstruct_intra(route, kind, i, k)) == t
    with pytest.raises(ValueError):
        reconstruct_intra(route, "shuffle", 0, 1)


def test_rule_of_three_plug_in_and_bounds():
    N = 200_000
    assert rule_of_three_bound(N) == pytest.approx(1.5e-5)
    assert plug_in_variance(0.0, N) == pytest.approx(3.0 / N)
    assert plug_in_variance(0.5, N) == pytest.approx(0.5)
    assert rho_q_upper(0.0, 0.3, N) == pytest.approx(math.sqrt(3.0 / N) / math.sqrt(0.21))
    assert math.isnan(rho_q_upper(0.0, 0.0, N))
    with pytest.raises(ValueError):
        rule_of_three_bound(0)


def test_weighted_quantile_is_lower_quantile():
    v = np.array([1.0, 2.0, 3.0, 4.0])
    assert weighted_quantile_sorted(v, np.ones(4), 0.5) == 2.0
    v5 = np.arange(5.0)
    assert weighted_quantile_sorted(v5, np.ones(5), 0.5) == 2.0
    # weights as route multiplicities: value 10 carries most of the mass
    assert weighted_quantile_sorted(np.array([1.0, 10.0]), np.array([1.0, 5.0]), 0.5) == 10.0


def test_route_bootstrap_shape_and_degenerate_cases():
    rng = np.random.default_rng(3)
    # all zeros: point and interval are zero
    z = np.zeros(50)
    assert route_bootstrap_quantile(z, np.arange(50), reps=200) == (0.0, 0.0, 0.0)
    # one cluster: every replicate is the same sample, so the interval collapses
    v = rng.normal(size=40)
    pt, lo, hi = route_bootstrap_quantile(v, np.zeros(40, dtype=int), reps=200)
    assert lo == pytest.approx(pt) and hi == pytest.approx(pt)
    # many clusters: ordered interval around the point, and it narrows with more clusters
    vals = rng.normal(size=2000)
    few = route_bootstrap_quantile(vals, np.arange(2000) % 20, reps=500, seed=1)
    many = route_bootstrap_quantile(vals, np.arange(2000) % 1000, reps=500, seed=1)
    assert few[1] <= few[0] <= few[2] and many[1] <= many[0] <= many[2]
    assert (many[2] - many[1]) < (few[2] - few[1])
    # NaN values are dropped; the default replicate count is 1000 and the result is reproducible
    a = route_bootstrap_quantile(np.array([1.0, np.nan, 2.0, 3.0]), np.array([0, 0, 1, 2]), seed=5)
    b = route_bootstrap_quantile(np.array([1.0, np.nan, 2.0, 3.0]), np.array([0, 0, 1, 2]), seed=5)
    assert a == b and all(np.isfinite(a))
    assert route_bootstrap_quantile(np.array([np.nan]), np.array([0]))[0] != route_bootstrap_quantile(
        np.array([np.nan]), np.array([0]))[0]  # NaN point for an empty sample
    with pytest.raises(ValueError):
        route_bootstrap_quantile(np.zeros(3), np.zeros(2))


def test_class_costs_formulas_per_class():
    N = 200_000
    eps = 1e-3
    z2 = 1.96 ** 2
    # class 1, Var > floor: quantum CRN = one amplitude, classical = z^2 var / eps^2
    P1, pn = 0.01, 0.05
    c = class_costs(1, P1, 0.0, pn, eps, N)
    assert c["Nq"] == pytest.approx(quantum_queries(P1 * (1 - P1), eps))
    assert c["Nc"] == pytest.approx(z2 * P1 * (1 - P1) / eps ** 2)
    assert c["Nq_sa"] == pytest.approx(quantum_queries(pn * (1 - pn), eps))
    assert c["Nc_sa"] == pytest.approx(z2 * pn * (1 - pn) / eps ** 2)
    # class 1 with zero count: quantum on the 3/N plug-in; classical is the certified
    # count max(z^2 (3/N) / eps^2, 3/eps), which is the floor 3/eps here
    z = class_costs(1, 0.0, 0.0, pn, eps, N)
    assert z["Nq"] == pytest.approx(quantum_queries(3.0 / N, eps))
    assert z["Nc"] == pytest.approx(3.0 / eps)
    # classes 2-3: quantum two-estimate at eps/2 each; classical DIRECT signed mean
    P1s, Pm1s = 0.01, 0.02
    s = class_costs(2, P1s, Pm1s, pn, eps, N)
    assert s["Nq"] == pytest.approx(quantum_queries(P1s * (1 - P1s), eps / 2)
                                    + quantum_queries(Pm1s * (1 - Pm1s), eps / 2))
    var_dir = P1s + Pm1s - (P1s - Pm1s) ** 2
    assert s["Nc"] == pytest.approx(z2 * var_dir / eps ** 2)
    # classes 2-3 with both counts zero: both estimates on the 3/N floor, direct on 3/N
    s0 = class_costs(3, 0.0, 0.0, pn, eps, N)
    assert s0["Nq"] == pytest.approx(2 * quantum_queries(3.0 / N, eps / 2))
    assert s0["Nc"] == pytest.approx(3.0 / eps)
    # the certified floor binds only when the CLT term is below 3/eps
    big = class_costs(2, 0.2, 0.1, pn, eps, N)
    assert big["Nc"] == pytest.approx(z2 * (0.2 + 0.1 - 0.1 ** 2) / eps ** 2)
    with pytest.raises(ValueError):
        class_costs(1, 0.1, 0.1, pn, eps, N)
    with pytest.raises(ValueError):
        class_costs(4, 0.1, 0.0, pn, eps, N)


def test_class_costs_array_matches_scalar_and_query_helpers():
    rng = np.random.default_rng(7)
    n = 60
    cls = rng.integers(1, 4, size=n)
    P1 = np.where(cls == 1, rng.uniform(0.02, 0.5, n), rng.uniform(0, 0.2, n) * (rng.random(n) < 0.7))
    Pm1 = np.where(cls == 1, 0.0, rng.uniform(0, 0.2, n) * (rng.random(n) < 0.7))
    pn = rng.uniform(0.02, 0.5, n)
    eps = 0.1 * pn
    arr = class_costs_array(cls, P1, Pm1, pn, eps, 200_000)
    for i in range(n):
        sc = class_costs(int(cls[i]), P1[i], Pm1[i], pn[i], float(eps[i]), 200_000)
        for key in ("Nq", "Nc", "Nq_sa", "Nc_sa"):
            assert arr[key][i] == pytest.approx(sc[key], rel=1e-12)
    # vectorised query helpers equal the scalar ones
    v = np.array([0.0, 1e-6, 1e-4, 0.05, 0.2])
    for e in (1e-2, 1e-3, 1e-4):
        assert canonical_ae_M_array(v, e).tolist() == [canonical_ae_M(x, e) for x in v]
        assert quantum_queries_array(v, e).tolist() == [quantum_queries(x, e) for x in v]
    assert classical_queries_array(v, 1e-3)[1] == pytest.approx(classical_queries(1e-6, 1e-3))


def test_quantise_levels_nearest_bins_and_absorbing_top():
    unit, V = 1.0, 4
    x = np.array([0.0, 0.49, 0.5, 1.49, 1.5, 2.6, 100.0])
    assert quantise_levels(x, unit, V).tolist() == [0, 0, 1, 1, 2, 3, 3]
    with pytest.raises(ValueError):
        quantise_levels(x, 0.0, V)


def test_paired_level_check_is_zero_when_demands_sit_on_levels():
    """Deterministic integer demands equal their levels, so the level check has no bias."""
    N = 16
    dsc = np.tile(np.array([[1.0], [2.0], [1.0]]), (1, N))
    psc = np.tile(np.array([[0.0], [1.0], [2.0]]), (1, N))
    route = [0, 1, 2]
    for Q in (2.5, 3.5, 4.5):
        K = int(Q - 0.5)  # unit 1: Q = K + 1/2
        r = paired_level_check(route, dsc, psc, Q, 1.0, 8, K)
        assert r["bias"] == pytest.approx(0.0)
        assert r["se"] == 0.0


def test_band_and_distance_labels():
    assert band_label(0.0, PPRIME_BANDS) == "<0.02"
    assert band_label(0.02, PPRIME_BANDS) == "0.02-0.2"
    assert band_label(0.2, PPRIME_BANDS) == "0.2-0.5"
    assert band_label(0.5, PPRIME_BANDS) == ">0.5"
    assert band_label(float("nan"), PPRIME_BANDS) == "NaN"
    assert gate_distance(0.13) == pytest.approx(0.03)
    assert band_label(gate_distance(0.13), GATE_BANDS) == "0.02-0.1"
    assert [alt_distance_label(d) for d in (1, 2, 3, 4, 9)] == ["1", "2-3", "2-3", ">3", ">3"]
    assert ALT_DIST_LABELS == ("1", "2-3", ">3")
    with pytest.raises(ValueError):
        alt_distance_label(0)


# ------------------------------------------------------------ Revision 3: factor grids

@pytest.mark.parametrize("T", [1, 2, 8, 16, 32, 128])
def test_equal_mass_grid_bins_have_equal_probability(T):
    """Bin t is [Phi^-1 edges]: each bin has probability 1/T and its node is the midpoint."""
    z, w = equal_mass_grid(T)
    assert z.shape == (T,) and w.shape == (T,)
    assert np.all(np.diff(z) > 0) if T > 1 else True
    assert w.sum() == pytest.approx(1.0, abs=1e-15)
    assert np.allclose(w, 1.0 / T, rtol=0, atol=1e-15)
    with np.errstate(divide="ignore"):
        edges = st.norm.ppf(np.arange(T + 1) / T)     # bin edges in z: Phi^-1(t / T)
    edges[0], edges[-1] = -np.inf, np.inf
    assert np.allclose(np.diff(st.norm.cdf(edges)), 1.0 / T, rtol=0, atol=1e-12)   # mass 1/T each
    assert np.all((edges[:-1] < z) & (z < edges[1:]))                             # node inside its bin
    # node is the probability midpoint of its bin: Phi(z_t) = (t + 1/2) / T
    assert np.allclose(st.norm.cdf(z), (np.arange(T) + 0.5) / T, rtol=0, atol=1e-12)


def test_equal_mass_grid_symmetry_and_moments():
    for T in (8, 32, 128):
        z, w = equal_mass_grid(T)
        assert np.allclose(z, -z[::-1], atol=1e-12)          # symmetric about 0
        assert float(np.dot(w, z)) == pytest.approx(0.0, abs=1e-12)
    # Midpoint bins under-weight the Gaussian tails: E[Z^2] is 0.961 at 32 bins
    # (the mass beyond the last midpoint is not represented); the error shrinks with T.
    err = {}
    for T in (32, 128):
        zt, wt = equal_mass_grid(T)
        err[T] = abs(float(np.dot(wt, zt ** 2)) - 1.0)
    assert err[32] < 0.05 and err[128] < err[32]
    with pytest.raises(ValueError):
        equal_mass_grid(0)


def test_factor_grid_dispatch():
    z_gh, w_gh = factor_grid("gh", 16)
    z_g, w_g = gh_nodes(16)
    assert np.array_equal(z_gh, z_g) and np.array_equal(w_gh, w_g)
    z_em, w_em = factor_grid("em", 16)
    z_e, w_e = equal_mass_grid(16)
    assert np.array_equal(z_em, z_e) and np.array_equal(w_em, w_e)
    with pytest.raises(ValueError):
        factor_grid("sobol", 16)


def test_copula_equal_mass_rho_zero_matches_independent_model():
    plan = _synthetic_plan()
    route = [1, 2, 3]
    cd = copula_demand(plan, 4, 0.0, 8, grid="em")
    dem, unit, Ql = instance_demand(plan, 4)
    assert cd.z.size == 8 and np.allclose(cd.w, 1.0 / 8)
    assert chance_copula(route, cd) == pytest.approx(chance(route, dem, Ql), abs=1e-12)


def test_copula_equal_mass_converges_to_gauss_hermite_limit():
    """Two grids of different kind converge to the same copula window probability."""
    plan = _synthetic_plan()
    route = [1, 2, 3]
    ref = chance_copula(route, copula_demand(plan, 4, 0.6, 128, grid="gh"))
    em = chance_copula(route, copula_demand(plan, 4, 0.6, 64, grid="em"))
    assert em == pytest.approx(ref, abs=5e-3)


# ------------------------------------------------------------ Revision 3: exact window DP

def test_tail_over_pickups_matches_classical_chance_on_random_routes():
    """The batched recursion equals the exact peak pmf of classical.chance for every K."""
    rng = np.random.default_rng(11)
    worst = 0.0
    for _ in range(40):
        n = int(rng.integers(1, 7))
        V = int(rng.choice([2, 3, 4, 6]))
        T = int(rng.integers(1, 4))
        pd = rng.dirichlet(np.ones(V), size=n)
        pd[rng.random(n) < 0.3] = np.eye(V)[0]
        pp = rng.dirichlet(np.ones(V), size=(T, n))
        route = [int(c) for c in rng.permutation(n)]
        for K in sorted({0, 1, n * (V - 1) // 2, n * (V - 1), n * (V - 1) + 2}):
            got = tail_over_pickups(route, pd, pp, K)
            for s in range(T):
                dem = DiscreteDemand(pmf_d=pd, pmf_p=pp[s], unit=1.0)
                worst = max(worst, abs(got[s] - chance(route, dem, K)))
    assert worst < 1e-12


def test_tail_over_pickups_negative_threshold_and_zero_mass():
    pd = np.array([[0.0, 1.0], [0.0, 1.0]])
    pp = np.array([[[1.0, 0.0], [1.0, 0.0]]])
    assert tail_over_pickups([0, 1], pd, pp, -1).tolist() == [1.0]
    assert tail_over_pickups([0, 1], pd, pp, 2).tolist() == [0.0]
    assert tail_over_pickups([0, 1], pd, pp, 1).tolist() == [1.0]   # F = 2 > 1 (deliveries 1 + 1)


def test_copula_window_equals_double_sum_of_classical_chance():
    """chance_copula is exactly sum_t sum_s w_t w_s P(F > K | z_t, z_s) (classical.chance per pair)."""
    plan = _synthetic_plan()
    route = [1, 2, 3]
    cd = copula_demand(plan, 3, 0.6, 5, grid="em")
    direct = 0.0
    for t in range(cd.z.size):
        for s in range(cd.z.size):
            direct += cd.w[t] * cd.w[s] * chance(route, cd.demand(t, s), cd.Q_level)
    assert chance_copula(route, cd) == pytest.approx(direct, abs=1e-12)


# ------------------------------------------------------------ Revision 3: certified classical floor

def test_classical_certified_floor_and_clt_branch():
    eps = 1e-3
    assert classical_queries_certified(0.0, eps) == pytest.approx(3.0 / eps)
    assert classical_queries_certified(1e-9, eps) == pytest.approx(3.0 / eps)
    v = 0.2
    assert classical_queries_certified(v, eps) == pytest.approx(1.96 ** 2 * v / eps ** 2)
    # crossover: the CLT term equals the floor at var = 3 eps / z^2
    v_star = 3 * eps / 1.96 ** 2
    assert classical_queries_certified(v_star, eps) == pytest.approx(3.0 / eps)
    with pytest.raises(ValueError):
        classical_queries_certified(0.1, 0.0)


def test_classical_certified_array_matches_scalar():
    v = np.array([0.0, 1e-7, 1e-4, 0.05, 0.25])
    eps = np.array([1e-3, 1e-3, 1e-4, 1e-4, 1e-2])
    arr = classical_queries_certified_array(v, eps)
    assert arr.tolist() == pytest.approx([classical_queries_certified(x, e) for x, e in zip(v, eps)])
    with pytest.raises(ValueError):
        classical_queries_certified_array(v, np.array([0.0] * 5))


def test_length_label_bands():
    assert [length_label(L) for L in (1, 8, 9, 15, 16, 25, 26, 40)] == \
        ["<=8", "<=8", "9-15", "9-15", "16-25", "16-25", ">25", ">25"]
    with pytest.raises(ValueError):
        length_label(0)


# ------------------------------------------------------------ Revision 3: driver helpers

def test_gh_selection_and_grid_configs():
    import pandas as pd

    routes_csv = ROOT / "results" / "quasar" / "m3a_v2_routes.csv"
    if not routes_csv.exists():
        pytest.skip("M3a v2 routes not available")
    drv = _load_driver()
    routes = pd.read_csv(routes_csv)
    sel = drv._gh_selection(routes)
    keys = {(s["instance"], s["route_id"]) for s in sel}
    assert len(sel) == 9 and len(keys) == 9
    assert sum(s["long"] for s in sel) == 3
    for s in sel:
        row = routes[(routes["instance"] == s["instance"]) & (routes["route_id"] == s["route_id"])].iloc[0]
        assert row["L"] >= 3
        if not s["long"]:
            assert 0.05 < row["p_base"] < 0.95
    labels = [length_label(int(routes[(routes["instance"] == s["instance"]) &
                                      (routes["route_id"] == s["route_id"])]["L"].iloc[0]))
              for s in sel if not s["long"]]
    assert sorted(labels) == ["16-25", "16-25", "9-15", "9-15", "<=8", "<=8"]
    assert [(s["instance"], s["route_id"]) for s in drv._gh_selection(routes)] == \
        [(s["instance"], s["route_id"]) for s in sel]
    assert ("em", 128) not in drv._gh_configs(6) and ("gh", 128) not in drv._gh_configs(6)
    assert ("em", 128) in drv._gh_configs(5) and ("gh", 128) in drv._gh_configs(5)
    assert ("em", 64) in drv._gh_configs(6)


def test_work_gh_matches_chance_copula(tmp_path):
    drv = _load_driver()
    plan = _synthetic_plan()
    plan_path = tmp_path / "SYN.json"
    plan_path.write_text(json.dumps(plan))
    route = [1, 2, 3]
    out = tmp_path / "row.json"
    row = drv._work_gh((str(plan_path), "SYN", "CON", 0, route, 3, False, 4, "em", 8, 0.6, 0.5, 0.01,
                        str(out)))
    cd = copula_demand(plan, 4, 0.6, 8, DEFAULT_CV, "nearest", "em")
    assert row["p_disc"] == pytest.approx(chance_copula(route, cd), abs=1e-14)
    assert row["f"] == 3 and row["grid"] == "em" and row["K"] == cd.Q_level
    assert json.loads(out.read_text())["p_disc"] == pytest.approx(row["p_disc"])


def test_work_v3_toy_route_has_no_gh_output(tmp_path, monkeypatch):
    """The v3 worker still computes the end-load bound and the paired check on a toy route
    after the Gauss-Hermite block moved to the v3gh part."""
    import pickle

    drv = _load_driver()
    monkeypatch.setattr(drv, "V3_PARTS", tmp_path)
    plan = _synthetic_plan()
    plan_path = tmp_path / "SYN.json"
    plan_path.write_text(json.dumps(plan))
    rrows = [{"route_id": 0, "route": "1 2 3"}]
    out = drv._work_v3((str(plan_path), rrows, [], {("SYN", 0)}, 4000, 0.6))
    assert out["n_routes"] == 1 and out["viol"] == 0
    with open(tmp_path / "SYN.pkl", "rb") as fh:
        d = pickle.load(fh)
    assert "gh" not in d
    assert {r["b"] for r in d["paired"]} == {4, 5, 6}
    assert len(d["routes"]) == 1 and 0 <= d["routes"][0]["B_union"] <= 1


# ------------------------------------------------------------ Revision 3b: bin-averaged factor grid

def test_bin_averaged_nodes_normalised_inside_bins_and_exact_for_phi_moment():
    for T in (8, 64):
        z_mid, nodes, W = bin_averaged_nodes(T, 32)
        assert nodes.shape == (T, 32) and W.shape == (T, 32)
        assert np.allclose(W.sum(axis=1), 1.0, atol=1e-14) and (W >= 0).all()
        assert np.allclose(z_mid, st.norm.ppf((np.arange(T) + 0.5) / T))
        # interior bins: inner nodes lie between the bin edges
        edges = st.norm.ppf(np.arange(T + 1) / T)
        for t in range(1, T - 1):
            assert (nodes[t] > edges[t]).all() and (nodes[t] < edges[t + 1]).all()
        # E[Phi(Z)] = 1/2 is reproduced by the mixture over bins (1/T) sum_t avg_t Phi
        mix = float(np.sum(W * st.norm.cdf(nodes)) / T)
        assert mix == pytest.approx(0.5, abs=1e-13)


@pytest.mark.parametrize("T", [8, 128])
def test_em_avg_mixture_reproduces_marginal_level_pmf(T):
    """Mixture over equal-probability bins of the bin-averaged conditionals = marginal (1e-12)."""
    plan = _synthetic_plan()
    b = 4
    cd = copula_demand(plan, b, 0.6, T, DEFAULT_CV, "nearest", "em_avg")
    V = 2 ** b
    assert cd.pmf_d.shape[0] == T and np.allclose(cd.w, 1.0 / T)
    assert np.allclose(cd.pmf_d.sum(axis=2), 1.0, atol=1e-12)
    dbar = np.asarray(plan["dbar"], dtype=np.float64)
    pbar = np.asarray(plan["pbar"], dtype=np.float64)
    n = dbar.size
    for c in range(n):
        marg_d = gamma_level_pmf(float(dbar[c]), DEFAULT_CV, cd.unit, V)
        marg_p = gamma_level_pmf(float(pbar[c]), DEFAULT_CV, cd.unit, V)
        assert np.max(np.abs(cd.pmf_d[:, c, :].mean(axis=0) - marg_d)) < 1e-12
        assert np.max(np.abs(cd.pmf_p[:, c, :].mean(axis=0) - marg_p)) < 1e-12


def test_copula_em_avg_rho_zero_matches_independent_model():
    plan = _synthetic_plan()
    route = [1, 2, 3]
    cd = copula_demand(plan, 4, 0.0, 8, grid="em_avg")
    dem, unit, Ql = instance_demand(plan, 4)
    assert chance_copula(route, cd) == pytest.approx(chance(route, dem, Ql), abs=1e-12)


def test_copula_em_avg_converges_to_gauss_hermite_limit():
    plan = _synthetic_plan()
    route = [1, 2, 3]
    ref = chance_copula(route, copula_demand(plan, 4, 0.6, 128, grid="gh"))
    avg = chance_copula(route, copula_demand(plan, 4, 0.6, 64, grid="em_avg"))
    assert avg == pytest.approx(ref, abs=5e-3)


def test_factor_grid_em_avg_dispatch():
    z, w = factor_grid("em_avg", 8)
    z_e, w_e = equal_mass_grid(8)
    assert np.array_equal(z, z_e) and np.array_equal(w, w_e)
    with pytest.raises(ValueError):
        factor_grid("em_mid", 8)


def test_driver_configs_include_bin_averaged_grid():
    drv = _load_driver()
    assert ("em_avg", 128) in drv._gh_configs(5) and ("em_avg", 8) in drv._gh_configs(5)
    assert ("em_avg", 32) in drv._gh_configs(6) and ("em_avg", 64) not in drv._gh_configs(6)
    assert drv._cfg_label("em_avg", 8) == "EMa8" and drv._cfg_label("em", 8) == "EMm8"
    assert drv._cfg_label("gh", 16) == "GH16"


# ------------------------------------------------------------ Revision 3b: CRN on the factor grid

def _toy_copula(T=3, n=4, V=4, seed=1, K=5):
    rng = np.random.default_rng(seed)
    pmf_d = rng.dirichlet(np.ones(V), size=(T, n))
    pmf_p = rng.dirichlet(np.ones(V), size=(T, n))
    pmf_d[:, 0] = np.eye(V)[0]
    pmf_p[:, 0] = np.eye(V)[0]
    return CopulaDemand(pmf_d=pmf_d, pmf_p=pmf_p, z=np.arange(T, dtype=float),
                        w=np.full(T, 1.0 / T), unit=1.0, Q_level=K, rho=0.5, b=2)


def test_crn_identical_routes_have_no_difference():
    cd = _toy_copula()
    out = crn_factor_grid([([1, 2, 3], [1, 2, 3])], cd, K=2000, seed=3)
    assert out["P_D1"][0] == 0.0 and out["P_Dm1"][0] == 0.0
    assert out["p_a"][0] == out["p_b"][0]


def test_crn_point_mass_pairs_are_exact():
    """Deterministic demands: route [1,2] peaks at 4, route [2,1] at 5; with K = 4 only [2,1] overflows."""
    V = 4
    cd = _toy_copula(T=2, n=3, V=V, K=4)
    pmf_d = np.zeros_like(cd.pmf_d)
    pmf_p = np.zeros_like(cd.pmf_p)
    pmf_d[:, :, 0] = 1.0
    pmf_d[:, 1, :] = np.eye(V)[2]
    pmf_d[:, 2, :] = np.eye(V)[2]
    pmf_p[:, :, :] = np.eye(V)[0]
    pmf_p[:, 2, :] = np.eye(V)[3]
    cd = CopulaDemand(pmf_d=pmf_d, pmf_p=pmf_p, z=cd.z, w=cd.w, unit=1.0, Q_level=4, rho=0.5, b=2)
    out = crn_factor_grid([([1, 2], [2, 1])], cd, K=50, seed=0)
    assert out["P_D1"][0] == pytest.approx(1.0, abs=1e-15)
    assert out["P_Dm1"][0] == pytest.approx(0.0, abs=1e-15)
    assert out["p_a"][0] == pytest.approx(0.0, abs=1e-15) and out["p_b"][0] == pytest.approx(1.0, abs=1e-15)


def test_crn_per_draw_identity_and_exact_tail_agreement():
    cd = _toy_copula(T=3, n=4, V=4, seed=2, K=5)
    route = [1, 2, 3]
    alt = [3, 1, 2]
    other = [2, 3, 1]
    out = crn_factor_grid([(route, alt), (route, other)], cd, K=200_000, seed=7)
    for m in range(2):
        # per draw, D = 1[F_b > Q] - 1[F_a > Q], so the two estimates satisfy this exactly
        assert out["P_D1"][m] - out["P_Dm1"][m] == pytest.approx(out["p_b"][m] - out["p_a"][m], abs=1e-12)
    exact = chance_copula(route, cd)
    assert abs(out["p_a"][0] - exact) < 0.005
    assert abs(out["p_b"][0] - chance_copula(alt, cd)) < 0.005


def test_crn_rejects_bad_input():
    cd = _toy_copula()
    with pytest.raises(ValueError):
        crn_factor_grid([], cd, K=10, seed=0)
    with pytest.raises(ValueError):
        crn_factor_grid([([1], [1])], cd, K=0, seed=0)

