"""M3: generic backward driver (EXPERIMENT_A.md sections 2.2, 2.3)."""
import numpy as np
import pytest

from transvrp.baton.costs import fit_lsm_general
from transvrp.baton.otr2 import _overflow_step
from transvrp.data import route_problem
from transvrp.expa.backward import iso_fit_fn, ls_backward, split_80_20
from transvrp.expa.features import RouteCache, RouteStats
from transvrp.expa.paths import Paths, route_specs


def _con3_routes(n_train=500):
    specs = {s.route_id: s for s in route_specs("CON3-0")}
    out = []
    for p in route_problem("CON3-0", n_train=n_train, n_test=10):
        rid = p["route_id"]
        g = p["g_train"]
        paths = Paths(p=g.copy(), d=np.zeros_like(g))   # g = p - d exactly
        cache = RouteCache(specs[rid], paths, RouteStats.fit(paths))
        out.append((p, cache))
    return out


def test_ls_parity_with_fit_lsm_general_on_con3():
    for p, cache in _con3_routes():
        B, H, E = p["B"], p["H"], p["E"]
        m = len(H)
        models = ls_backward(cache, B, H, E, iso_fit_fn(), None, None)
        ref = fit_lsm_general(p["g_train"], B, H, E)
        assert set(models) == set(ref) == set(range(1, m))
        cum = np.cumsum(p["g_train"], axis=1)
        grid = np.linspace(cum.min() - 1.0, cum.max() + 1.0, 50)
        for k in range(1, m):
            ours = models[k][0].predict_w(grid)
            theirs = ref[k].predict(grid)
            assert np.allclose(ours, theirs, atol=1e-9, rtol=0)


class _Recorder:
    def __init__(self, fn):
        self.fn = fn
        self.calls = []

    def __call__(self, k, cache, fit_idx, y_fit, val_idx, y_val, warm):
        self.calls.append((k, np.asarray(fit_idx).copy(), np.asarray(y_fit).copy()))
        return self.fn(k, cache, fit_idx, y_fit, val_idx, y_val, warm)


def _folded(cache, N):
    folds, val = split_80_20(N)
    return folds, val


def test_leakage_log_is_disjoint_and_validation_is_unseen():
    p, cache = _con3_routes()[0]
    N = cache.W.shape[0]
    folds, val = _folded(cache, N)
    log: list = []
    models = ls_backward(cache, p["B"], p["H"], p["E"], iso_fit_fn(),
                         folds, val, log=log)
    m = len(p["H"])
    assert set(models) == set(range(1, m))
    assert all(len(v) == 2 for v in models.values())
    for entry in log:
        fit_rows, applied = entry["fit_rows"], entry["applied_rows"]
        assert np.intersect1d(fit_rows, applied).size == 0
        assert np.intersect1d(fit_rows, val).size == 0
        assert 0.0 <= entry["disagree_rate"] <= 1.0
    assert {e["fold"] for e in log} == {"A", "B", "val"}
    assert len(log) == 3 * (m - 1)


def test_targets_and_fitted_rows():
    p, cache = _con3_routes()[0]
    N = cache.W.shape[0]
    folds, val = _folded(cache, N)
    rec = _Recorder(iso_fit_fn())
    ls_backward(cache, p["B"], p["H"], p["E"], rec, folds, val)
    tau = _overflow_step(cache.W, p["B"])
    allowed = np.r_[0.0, p["H"], p["E"]]
    assert len(rec.calls) > 0
    for k, fit_idx, y in rec.calls:
        assert np.all(tau[fit_idx] > k)
        assert np.all(np.isin(y, allowed))


def _grid(cache, k):
    w = cache.W[:, k - 1]
    return np.linspace(w.min(), w.max(), 41)


def test_outcomes_of_dead_rows_after_k_do_not_matter():
    """Perturb stops after k for a fold-A row that has already overflowed by k.
    Such a row contributes nothing after k, so no predictor changes."""
    p, cache = _con3_routes()[0]
    B = p["B"]
    N = cache.W.shape[0]
    folds, val = _folded(cache, N)
    tau = _overflow_step(cache.W, B)
    m = len(p["H"])
    base = ls_backward(cache, B, p["H"], p["E"], iso_fit_fn(), folds, val)
    checked = 0
    for k in range(2, m - 1):
        cand = [r for r in folds[0] if tau[r] <= k]
        if not cand:
            continue
        row = cand[0]
        g2 = p["g_train"].copy()
        g2[row, k:] += 2.0 * B                   # stops k+1..m (0-based k..)
        paths2 = Paths(p=g2, d=np.zeros_like(g2))
        cache2 = RouteCache(cache.spec, paths2, RouteStats.fit(paths2))
        alt = ls_backward(cache2, B, p["H"], p["E"], iso_fit_fn(), folds, val)
        for kk in range(1, m):
            w = _grid(cache, kk)
            for a, b in zip(base[kk], alt[kk]):
                assert np.array_equal(a.predict_w(w), b.predict_w(w))
        checked += 1
    assert checked > 0


def test_literal_M3d_claim_fails_for_alive_rows():
    """Characterization, not a pass criterion of the spec. EXPERIMENT_A.md
    M3(d) asks that changing an outcome after stop k of a fold-A row changes no
    fold-A predictor at k. For an ALIVE row this cannot hold: its target at k
    is built from later decisions and outcomes, so its own fold's fit at k
    moves. The test checks that the effect is real (the spec's leakage
    guarantee is the disjointness of M3(b))."""
    p, cache = _con3_routes()[0]
    B = p["B"]
    N = cache.W.shape[0]
    folds, val = _folded(cache, N)
    m = len(p["H"])
    tau = _overflow_step(cache.W, B)
    k = m - 2
    base = ls_backward(cache, B, p["H"], p["E"], iso_fit_fn(), folds, val)
    w = _grid(cache, k)
    changed = 0
    for row in [r for r in folds[0] if tau[r] > k + 1][:20]:
        g2 = p["g_train"].copy()
        g2[row, k + 1:] += 2.0 * B
        paths2 = Paths(p=g2, d=np.zeros_like(g2))
        cache2 = RouteCache(cache.spec, paths2, RouteStats.fit(paths2))
        alt = ls_backward(cache2, B, p["H"], p["E"], iso_fit_fn(), folds, val)
        if not np.array_equal(base[k][0].predict_w(w), alt[k][0].predict_w(w)):
            changed += 1
    assert changed > 0
