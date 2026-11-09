"""Extension 2 (SPEC_V2): generic instances, arc-noise families, detsig / pboi_normal, move diagnostics."""

import hashlib
import math
import os

import numpy as np
import pytest
from scipy.stats import spearmanr

from sortie import search
from sortie.instances import (
    DATA_PATH, effective_H, load_vrp, load_x641, make_head_instance, make_instance,
    noise_matrix, with_congestion,
)
from sortie.pboi import expost_var, lam_vector, ordered, pboi, rank_prices
from sortie.search import (
    construct_initial, make_objective, move_diagnostics, solve_fair,
)
from sortie.sim import _draw_chunk, evaluate_plan, region_labels, simulate
from sortie.tsp import TourSolver, tour_stats

DATA_DIR = os.path.dirname(DATA_PATH)
X101 = os.path.join(DATA_DIR, "X", "X-n101-k25.vrp")


@pytest.fixture(scope="module")
def inst20():
    coords, demand = load_x641()
    return make_instance(coords, demand, n=20, b=0)


@pytest.fixture(scope="module")
def inst15():
    coords, demand = load_x641()
    return make_instance(coords, demand, n=15, b=0, m=5)


def _sumL(routes, C):
    return sum(tour_stats(r, C)[0] for r in routes)


def _check_partition(routes, inst):
    assert len(routes) == inst.m
    assert all(len(r) > 0 for r in routes)
    assert sorted(x for r in routes for x in r) == list(range(1, inst.C.shape[0]))
    for r in routes:
        assert int(inst.demand[np.asarray(r)].sum()) <= inst.Q


def _random_plan(rng, n, m):
    """m nonempty routes partitioning customers 1..n (random order)."""
    perm = rng.permutation(np.arange(1, n + 1))
    cuts = np.sort(rng.choice(np.arange(1, n), size=m - 1, replace=False))
    return [[int(x) for x in part] for part in np.split(perm, cuts)]


# ---------------------------------------------------------------------------
# L1: generic instances
# ---------------------------------------------------------------------------

def test_load_vrp_x641_equals_previous_arrays():
    # fingerprints of the arrays returned by load_x641 before the extension
    name, coords, demand, cap = load_vrp(DATA_PATH)
    assert name == "X-n641-k35" and cap == 1381
    assert coords.shape == (641, 2) and demand.shape == (641,)
    assert hashlib.sha256(coords.tobytes()).hexdigest() == (
        "3254701d48276030b61888e5e97c9dc874970c98865e5b4a631bbd7f6cddb6fe")
    assert hashlib.sha256(demand.tobytes()).hexdigest() == (
        "79fe448169ae762bcf7f9badcffba4e0196a9fffe0f22caa494ae20c2a1fdc2f")
    c2, d2 = load_x641(DATA_PATH)
    assert np.array_equal(c2, coords) and np.array_equal(d2, demand)


def test_load_vrp_crlf_tabs_and_spaces_equal_original(tmp_path):
    with open(DATA_PATH, "rb") as fh:
        raw = fh.read().decode("utf-8")
    # rewrite as CRLF; coordinate lines get tabs, runs of blanks and trailing blanks
    lines = raw.replace("\r\n", "\n").split("\n")
    out = []
    k = 0
    for ln in lines:
        parts = ln.split()
        if len(parts) == 3 and parts[0].isdigit():
            sep = "\t" if k % 2 else "   "
            ln = "  " + sep.join(parts) + "   "
            k += 1
        out.append(ln)
    path = tmp_path / "crlf.vrp"
    path.write_bytes("\r\n".join(out).encode("utf-8"))
    name, coords, demand, cap = load_vrp(str(path))
    _, c0, d0, _ = load_vrp(DATA_PATH)
    assert name == "X-n641-k35" and cap == 1381
    assert np.array_equal(coords, c0) and np.array_equal(demand, d0)


def test_load_vrp_x101_depot_first_and_header():
    name, coords, demand, cap = load_vrp(X101)
    assert name == "X-n101-k25"
    assert cap == 206
    assert coords.shape == (101, 2) and demand.shape == (101,)
    assert demand.dtype == np.int64 and coords.dtype == np.float64
    assert np.array_equal(coords[0], [365.0, 689.0])  # file node 1 is the depot
    assert demand[0] == 0


def test_load_vrp_depot_section_moves_depot_first(tmp_path):
    text = "\n".join([
        "NAME : tiny", "DIMENSION : 4", "EDGE_WEIGHT_TYPE : EUC_2D", "CAPACITY : 10",
        "NODE_COORD_SECTION", "1 0 0", "2 1 0", "3 5 5", "4 0 2", "DEMAND_SECTION",
        "1 0", "2 1", "3 2", "4 3", "DEPOT_SECTION", "3", "-1", "EOF", "",
    ])
    path = tmp_path / "depot3.vrp"
    path.write_text(text)
    name, coords, demand, cap = load_vrp(str(path))
    assert name == "tiny" and cap == 10
    assert np.array_equal(coords, [[5, 5], [0, 0], [1, 0], [0, 2]])
    assert np.array_equal(demand, [2, 0, 1, 3])


def test_load_vrp_rejects_non_euclidean(tmp_path):
    path = tmp_path / "bad.vrp"
    path.write_text("NAME : x\nEDGE_WEIGHT_TYPE : EXPLICIT\nCAPACITY : 5\n"
                    "NODE_COORD_SECTION\n1 0 0\nDEMAND_SECTION\n1 0\nEOF\n")
    with pytest.raises(ValueError):
        load_vrp(str(path))


def test_make_instance_prefix_default_and_custom(inst20):
    coords, demand = load_x641()
    assert inst20.name == "X641-n20-b0"
    other = make_instance(coords, demand, n=20, b=0, prefix="ZZ")
    assert other.name == "ZZ-n20-b0"
    assert np.array_equal(other.C, inst20.C)


def test_make_head_instance_on_x101():
    _, coords, demand, _ = load_vrp(X101)
    inst = make_head_instance(coords, demand, "X-n101-k25", 50)
    assert inst.name == "X-n101-k25-h50"
    assert inst.coords.shape == (51, 2) and inst.demand.shape == (51,)
    assert np.array_equal(inst.coords[0], [365.0, 689.0])  # the file's own depot
    assert np.array_equal(inst.coords[1:], coords[1:51])  # first customers in file order
    assert inst.demand[0] == 0
    assert np.array_equal(inst.demand[1:], demand[1:51])
    assert inst.Q == int(math.floor(demand[1:51].sum() / 4.0))
    assert np.array_equal(inst.C, inst.C.T) and np.all(np.diag(inst.C) == 0)
    diff = coords[:51, None, :] - coords[None, :51, :]
    expected = np.ceil(np.sqrt((diff ** 2).sum(axis=2)))
    np.fill_diagonal(expected, 0.0)
    assert np.array_equal(inst.C, expected)


def test_make_head_instance_bounds(inst20):
    _, coords, demand, _ = load_vrp(X101)
    with pytest.raises(ValueError):
        make_head_instance(coords, demand, "x", 0)
    with pytest.raises(ValueError):
        make_head_instance(coords, demand, "x", 101)


# ---------------------------------------------------------------------------
# L2: arc-noise families
# ---------------------------------------------------------------------------

def _legacy_draw(N, v, sY, size, seed, chunk_index, H=None):
    """The _draw_chunk of the committed version (reference for bit-identity)."""
    rng = np.random.default_rng([int(seed), int(chunk_index)])
    Z = rng.standard_normal(size)
    Y = np.exp(sY * Z - 0.5 * sY * sY)
    if v == 0:
        return Y, None
    if H is None:
        eps = rng.gamma(shape=1.0 / (v * v), scale=v * v, size=(size, N, N))
    else:
        vh = v * np.asarray(H, dtype=np.float64)
        eps = rng.gamma(shape=1.0 / (vh * vh), scale=vh * vh, size=(size, N, N))
    return Y, eps


def test_gamma_draw_bit_identical_to_legacy(inst20):
    N = inst20.C.shape[0]
    for H in (None, with_congestion(inst20).H):
        for ci in (0, 3):
            Yn, en = _draw_chunk(N, 0.3, 0.15, 500, 77, ci, H)
            Yl, el = _legacy_draw(N, 0.3, 0.15, 500, 77, ci, H)
            assert np.array_equal(Yn, Yl)
            assert np.array_equal(en, el)
    assert np.array_equal(_draw_chunk(N, 0.0, 0.15, 10, 1, 0)[0],
                          _legacy_draw(N, 0.0, 0.15, 10, 1, 0)[0])


def test_simulate_default_is_gamma_and_bit_identical(inst20):
    plan = [[1, 2, 3, 4], [5, 6, 7], [8, 9, 10, 11], [12, 13, 14], [15, 16, 17, 18, 19, 20]]
    X0, Y0 = simulate(inst20, plan, 0.3, 0.15, 700, 9)
    X1, Y1 = simulate(inst20, plan, 0.3, 0.15, 700, 9, dist="gamma", dist_params=None)
    assert np.array_equal(X0, X1) and np.array_equal(Y0, Y1)
    ev0 = evaluate_plan(inst20, plan, 0.3, 0.15, S=400, seed=2)
    ev1 = evaluate_plan(inst20, plan, 0.3, 0.15, S=400, seed=2, dist="gamma")
    assert ev0 == ev1


def _pooled_moments(eps, N):
    off = ~np.eye(N, dtype=bool)
    x = eps[:, off]
    mean = float(x.mean())
    return mean, float(x.std() / mean)


@pytest.mark.parametrize("dist,params", [
    ("gamma", None),
    ("lognormal", None),
    ("incident", None),
    ("regional", None),
])
def test_noise_family_mean_and_cv(inst20, dist, params):
    N = inst20.C.shape[0]
    v = 0.3
    region = region_labels(inst20, dist, params)
    _, eps = _draw_chunk(N, v, 0.0, 20000, 2027, 0, None, dist, params, region)
    mean, cv = _pooled_moments(eps, N)
    assert abs(mean - 1.0) < 0.03
    assert abs(cv - v) / v < 0.03


@pytest.mark.parametrize("dist", ["gamma", "lognormal"])
def test_noise_family_mean_and_cv_per_arc_with_congestion(inst20, dist):
    cong = with_congestion(inst20)
    N = cong.C.shape[0]
    H = effective_H(cong)
    _, eps = _draw_chunk(N, 0.3, 0.0, 20000, 2028, 0, H, dist)
    off = ~np.eye(N, dtype=bool)
    target = 0.3 * np.asarray(H)
    means = eps.mean(axis=0)[off]
    sds = eps.std(axis=0)[off]
    assert np.max(np.abs(means - 1.0)) < 0.05
    assert np.max(np.abs(sds / means - target[off]) / target[off]) < 0.06


def test_incident_and_regional_raise_when_cv_too_small(inst20):
    N = inst20.C.shape[0]
    region = region_labels(inst20, "regional")
    with pytest.raises(ValueError):
        _draw_chunk(N, 0.05, 0.0, 10, 1, 0, None, "incident")
    with pytest.raises(ValueError):
        _draw_chunk(N, 0.05, 0.0, 10, 1, 0, None, "regional", None, region)
    with pytest.raises(ValueError):
        simulate(inst20, [[1, 2]], 0.05, 0.0, 10, 1, dist="incident")
    with pytest.raises(ValueError):
        simulate(inst20, [[1, 2]], 0.3, 0.0, 10, 1, dist="nonsense")
    with pytest.raises(ValueError):
        simulate(inst20, [[1, 2]], 0.3, 0.0, 10, 1, dist="regional",
                 dist_params={"n_regions": 3})


def test_regional_routes_in_same_region_are_positively_correlated(inst20):
    lab = region_labels(inst20, "regional")
    counts = np.bincount(lab[1:], minlength=4)
    g = int(np.argmax(counts))
    cust = [int(i) for i in range(1, inst20.C.shape[0]) if lab[i] == g]
    assert len(cust) >= 4
    half = len(cust) // 2
    r1, r2 = cust[:half], cust[half:]
    X, _ = simulate(inst20, [r1, r2], v=0.3, sY=0.0, S=20000, seed=3, dist="regional")
    corr = float(np.corrcoef(X[:, 0], X[:, 1])[0, 1])
    assert corr > 0.05
    # the gamma law has independent arcs: no such correlation beyond sampling noise
    Xg, _ = simulate(inst20, [r1, r2], v=0.3, sY=0.0, S=20000, seed=3)
    corr_g = float(np.corrcoef(Xg[:, 0], Xg[:, 1])[0, 1])
    assert abs(corr_g) < 0.03


def test_evaluate_plans_accepts_families(inst20):
    plan = [[1, 2, 3, 4], [5, 6, 7], [8, 9, 10, 11], [12, 13, 14], [15, 16, 17, 18, 19, 20]]
    for dist in ("lognormal", "incident", "regional"):
        res = evaluate_plan(inst20, plan, 0.3, 0.15, S=300, seed=4, dist=dist)
        assert math.isfinite(res["E_range"]) and res["E_range"] >= 0.0
        assert len(res["mean_driver_X"]) == 5


# ---------------------------------------------------------------------------
# L3: detsig and pboi_normal
# ---------------------------------------------------------------------------

def _plan_stats(seqs, inst, D):
    L = np.array([tour_stats(s, inst.C)[0] for s in seqs])
    W = np.array([tour_stats(s, inst.C, D)[1] for s in seqs])
    return L, W


def test_detsig_z0_equals_det_on_random_plans(inst20):
    D = noise_matrix(inst20)
    rng = np.random.default_rng(101)
    for case in range(6):
        seqs = _random_plan(rng, 20, 5)
        L, W = _plan_stats(seqs, inst20, D)
        for measure in ("max", "range", "gmd", "var"):
            det = make_objective("det", measure, 0.3)
            dsig = make_objective("detsig", measure, 0.3, z=0.0)
            assert dsig.value(L, W, seqs) == pytest.approx(det.value(L, W, seqs), rel=1e-12)
            vd, _ = det.prices(L, W, seqs)
            vs, ps = dsig.prices(L, W, seqs)
            assert np.allclose(vs, vd, rtol=1e-12, atol=0.0)
            assert np.all(ps == 0.0)


def test_detsig_value_and_price_definition(inst20):
    D = noise_matrix(inst20)
    rng = np.random.default_rng(103)
    seqs = _random_plan(rng, 20, 5)
    L, W = _plan_stats(seqs, inst20, D)
    v, z = 0.3, 1.5
    y = L + z * v * np.sqrt(W)
    ad = make_objective("detsig", "range", v, z=z)
    assert ad.value(L, W, seqs) == pytest.approx(ordered(y, lam_vector("range", 5)), rel=1e-12)
    varpi, psi = ad.prices(L, W, seqs)
    assert np.allclose(psi, z * varpi, rtol=0, atol=0)
    # varpi is the gradient of the ordered measure with respect to the risk-adjusted lengths
    eps = 1e-6
    grad = np.zeros(5)
    for k in range(5):
        dy = np.zeros(5)
        dy[k] = eps
        grad[k] = (ordered(y + dy, lam_vector("range", 5)) - ordered(y - dy, lam_vector("range", 5))) / (2 * eps)
    assert np.allclose(varpi, grad, atol=1e-6)
    ad_var = make_objective("detsig", "var", v, z=z)
    assert ad_var.value(L, W, seqs) == pytest.approx(float(np.mean((y - y.mean()) ** 2)), rel=1e-12)


def test_detsig_z_changes_value(inst20):
    D = noise_matrix(inst20)
    seqs = _random_plan(np.random.default_rng(107), 20, 5)
    L, W = _plan_stats(seqs, inst20, D)
    v0 = make_objective("detsig", "max", 0.3, z=0.0).value(L, W, seqs)
    v1 = make_objective("detsig", "max", 0.3, z=2.0).value(L, W, seqs)
    assert v1 > v0


def test_pboi_normal_matches_pboi_normal_family(inst20):
    D = noise_matrix(inst20)
    seqs = _random_plan(np.random.default_rng(109), 20, 5)
    L, W = _plan_stats(seqs, inst20, D)
    v = 0.3
    sigma = v * np.sqrt(W)
    ad = make_objective("pboi_normal", "range", v)
    assert ad.family == "normal"
    lam = lam_vector("range", 5)
    assert ad.value(L, W, seqs) == pytest.approx(pboi(L, sigma, lam, family="normal"), rel=1e-12)
    varpi, psi = ad.prices(L, W, seqs)
    assert np.allclose(varpi, rank_prices(L, sigma, lam, family="normal"), rtol=1e-12, atol=0.0)
    # gamma-marginal PBOI is a different objective
    assert make_objective("pboi", "range", v).value(L, W, seqs) != pytest.approx(
        ad.value(L, W, seqs), rel=1e-6)
    adv = make_objective("pboi_normal", "var", v)
    assert adv.value(L, W, seqs) == pytest.approx(expost_var(L, sigma), rel=1e-12)


def test_pboi_default_unchanged_by_family_argument(inst20):
    D = noise_matrix(inst20)
    seqs = _random_plan(np.random.default_rng(113), 20, 5)
    L, W = _plan_stats(seqs, inst20, D)
    ad = make_objective("pboi", "gmd", 0.3)
    assert ad.family == "gamma"
    assert ad.value(L, W, seqs) == pboi(L, 0.3 * np.sqrt(W), lam_vector("gmd", 5))


@pytest.mark.parametrize("mode", ["detsig", "pboi_normal"])
@pytest.mark.parametrize("kappa", [0.0, 1.0])
def test_solve_fair_new_modes_keep_lower_level_invariant(inst20, mode, kappa):
    inst = inst20
    C = inst.C
    D = noise_matrix(inst)
    start = construct_initial(inst, np.random.default_rng(0))
    solver = TourSolver(C, kappa, 0.3, D=D)
    budget = 1.10 * sum(TourSolver(C, 0.0, 0.3).solve(r)[1] for r in start)
    res = solve_fair(inst, start, mode, "range", 0.3, kappa, budget,
                     time_limit=60.0, seed=5, max_iters=20, solver=solver, z=1.0)
    routes = res["routes"]
    _check_partition(routes, inst)
    for r in routes:
        tour = solver.solve(r)[0]
        assert list(r) == list(tour) or list(r) == list(tour)[::-1]
        if len(r) <= 15:
            fresh = TourSolver(C, kappa, 0.3).solve(r)[0]
            assert list(r) == list(fresh) or list(r) == list(fresh)[::-1]
    assert res["sumL"] <= budget + 1e-6
    assert res["sumL"] == pytest.approx(_sumL(routes, C), abs=1e-6)
    assert res["n_confirm"] >= res["n_improve"]
    assert res["mode"] == mode


def test_solve_fair_saa_S_argument(inst20, monkeypatch):
    seen = []
    real = search.make_objective

    def spy(*args, **kwargs):
        if args[0] == "saa":
            seen.append(kwargs.get("S"))
        return real(*args, **kwargs)

    monkeypatch.setattr(search, "make_objective", spy)
    start = construct_initial(inst20, np.random.default_rng(1))
    budget = 1.10 * sum(TourSolver(inst20.C, 0.0, 0.3).solve(r)[1] for r in start)
    solve_fair(inst20, start, "saa", "range", 0.3, 0.0, budget, time_limit=60.0, seed=2,
               max_iters=3, saa_S=50)
    assert seen[-1] == 50
    solve_fair(inst20, start, "saa", "range", 0.3, 0.0, budget, time_limit=60.0, seed=2,
               max_iters=3)
    assert seen[-1] == search.SCEN_S
    adapter = real("saa", "range", 0.3, C=inst20.C, S=50)
    assert adapter.eps.shape[0] == 50


# ---------------------------------------------------------------------------
# L4: move diagnostics
# ---------------------------------------------------------------------------

PLAN15 = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11], [12, 13, 14, 15]]


@pytest.fixture(scope="module")
def diag15(inst15):
    budget = 1.10 * _sumL(PLAN15, inst15.C)
    df = move_diagnostics(inst15, PLAN15, "pboi", "range", 0.3, 0.0, budget, seed=0)
    return df, budget


def test_move_diagnostics_columns_and_ranks(inst15, diag15):
    df, budget = diag15
    assert list(df.columns) == ["kind", "a", "b", "i", "j", "predicted", "exact",
                                "pred_rank", "exact_rank"]
    assert len(df) > 50
    assert set(df["kind"]) == {"relocate", "swap"}
    assert np.all(np.isfinite(df["predicted"]))
    assert np.array_equal(np.sort(df["pred_rank"].to_numpy()), np.arange(len(df)))
    best = df.loc[df["pred_rank"] == 0, "predicted"].iloc[0]
    assert best == df["predicted"].min()
    exact_ok = df["exact"].notna()
    assert exact_ok.sum() > 0
    assert np.array_equal(df["exact_rank"].isna().to_numpy(), ~exact_ok.to_numpy())
    rel = df[df["kind"] == "relocate"]
    assert np.all(rel["j"] == -1)
    sw = df[df["kind"] == "swap"]
    assert np.all(sw["j"] >= 1)


def test_move_diagnostics_predicted_exact_spearman_positive(diag15):
    df, _ = diag15
    ok = df["exact"].notna()
    rho, _ = spearmanr(df.loc[ok, "predicted"], df.loc[ok, "exact"])
    assert rho > 0


def test_move_diagnostics_exact_matches_direct_resolve(inst15, diag15):
    df, budget = diag15
    C = inst15.C
    D = noise_matrix(inst15)
    solver = TourSolver(C, 0.0, 0.3, D=D)
    seqs0 = [list(solver.solve(r)[0]) for r in PLAN15]
    L0 = np.array([tour_stats(s, C)[0] for s in seqs0])
    W0 = np.array([tour_stats(s, C, D)[1] for s in seqs0])
    ad = make_objective("pboi", "range", 0.3)
    cur = ad.value(L0, W0, seqs0)
    for _, row in list(df.iloc[[0, len(df) // 2, len(df) - 1]].iterrows()):
        new = [list(s) for s in seqs0]
        a, b, i, j = int(row.a), int(row.b), int(row.i), int(row.j)
        if row.kind == "relocate":
            new[a] = [x for x in seqs0[a] if x != i]
            new[b] = seqs0[b] + [i]
        else:
            new[a] = [j if x == i else x for x in seqs0[a]]
            new[b] = [i if x == j else x for x in seqs0[b]]
        Ln = np.array([solver.solve(s)[1] for s in new])
        Wn = np.array([solver.solve(s)[2] for s in new])
        if Ln.sum() > budget + 1e-9:
            assert math.isnan(row.exact)
        else:
            assert row.exact == pytest.approx(ad.value(Ln, Wn, new) - cur, rel=1e-9, abs=1e-9)


def test_move_diagnostics_budget_break_gives_nan(inst15):
    solver = TourSolver(inst15.C, 0.0, 0.3)
    budget = sum(solver.solve(r)[1] for r in PLAN15)  # the solved plan's own total: no slack
    df = move_diagnostics(inst15, PLAN15, "pboi", "range", 0.3, 0.0, budget, seed=0)
    assert df["exact"].isna().any()


def test_move_diagnostics_subsample_is_reproducible_subset(inst15, diag15):
    full, budget = diag15
    sub = move_diagnostics(inst15, PLAN15, "pboi", "range", 0.3, 0.0, budget,
                           max_moves=25, seed=7)
    assert len(sub) == 25
    key = ["kind", "a", "b", "i", "j"]
    merged = sub.merge(full, on=key, suffixes=("_s", "_f"))
    assert len(merged) == 25
    assert np.allclose(merged["predicted_s"], merged["predicted_f"])
    again = move_diagnostics(inst15, PLAN15, "pboi", "range", 0.3, 0.0, budget,
                             max_moves=25, seed=7)
    assert sub.equals(again)


def test_move_diagnostics_rejects_bad_plan(inst15):
    with pytest.raises(ValueError):
        move_diagnostics(inst15, [[1, 2, 3]], "pboi", "range", 0.3, 0.0, np.inf)
