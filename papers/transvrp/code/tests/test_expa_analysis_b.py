"""M10 additions (EXPERIMENT_A.md sections 1.2, 4.2, 8 item 2b): cluster bootstrap,
section 4.2 monotonicity diagnostic and per-day paired SE.

Every expected value below is either hand-computed in the comment or obtained
by an independent brute-force loop in the test itself.
"""
import numpy as np
import pandas as pd
import pytest

from transvrp.expa import analysis
from transvrp.expa.features import RouteStats
from transvrp.expa.paths import RouteSpec


# ---------------------------------------------------------------- cluster bootstrap (1.2)

def test_bootstrap_two_instances_hand_checked():
    # instance means 1 and 2: each draw is 1 (both picked twice? no: both 0),
    # 1.5 (mixed) or 2 (both 1), with probabilities 1/4, 1/2, 1/4.
    res = analysis.cluster_bootstrap([[1.0], [2.0]], n_draws=4000)
    assert set(np.unique(res["draws"])) <= {1.0, 1.5, 2.0}
    assert abs(np.mean(res["draws"] == 1.5) - 0.5) < 0.03
    # 2.5th and 97.5th percentiles of this distribution are 1 and 2
    assert res["ci95"] == pytest.approx((1.0, 2.0))
    assert res["ci90"] == pytest.approx((1.0, 2.0))
    assert res["mean"] == pytest.approx(1.5)


def test_bootstrap_matches_an_independent_per_draw_loop():
    groups = [np.array([0.3, 0.9]), np.array([-0.4]), np.array([1.2, 0.1, 0.5]),
              np.array([2.0, 1.0]), np.array([-1.0, 0.0])]
    res = analysis.cluster_bootstrap(groups, n_draws=300)
    # independent: one scalar-index draw per resampled instance, route means taken
    # explicitly, the statistic is the mean of the drawn instance means
    means = [float(np.mean(g)) for g in groups]
    rng = np.random.default_rng(analysis.BOOT_SEED)
    brute = []
    for _ in range(300):
        pick = [int(rng.integers(0, len(groups))) for _ in range(len(groups))]
        brute.append(sum(means[i] for i in pick) / len(pick))
    # same draws; the mean is summed in a different order, so compare to 1e-12
    assert np.allclose(res["draws"], np.array(brute), rtol=0, atol=1e-12)


def test_bootstrap_keeps_routes_together_and_is_not_route_level():
    # instance A has routes (0, 0) (mean 0), instance B has route (10). A draw is
    # 0, 5 or 10 with probabilities 1/4, 1/2, 1/4. Resampling routes instead would
    # give values such as 10/3 or 20/3.
    res = analysis.cluster_bootstrap([[0.0, 0.0], [10.0]], n_draws=10_000)
    assert set(np.unique(res["draws"])) == {0.0, 5.0, 10.0}
    assert abs(np.mean(res["draws"] == 5.0) - 0.5) < 0.02
    assert abs(np.mean(res["draws"] == 0.0) - 0.25) < 0.02


def test_bootstrap_default_seed_and_deterministic():
    g = [[1.0, 2.0], [3.0], [0.5]]
    a = analysis.cluster_bootstrap(g)
    b = analysis.cluster_bootstrap(g)
    assert a["n_draws"] == 10_000 and a["seed"] == 20261115 == analysis.BOOT_SEED
    assert np.array_equal(a["draws"], b["draws"])
    c = analysis.cluster_bootstrap(g, seed=1)
    assert not np.array_equal(a["draws"], c["draws"])


def test_instance_bootstrap_reads_the_route_table():
    rows = []
    for i, d in enumerate([1.0, 2.0]):
        for rid in range(2):
            for rep in range(2):
                rows.append(dict(inst=f"I{i}", route_id=rid, rep=rep, N=10,
                                 policy="a", mean_cost=10 * (1 - d / 100)))
                rows.append(dict(inst=f"I{i}", route_id=rid, rep=rep, N=10,
                                 policy="b", mean_cost=10.0))
    df = pd.DataFrame(rows)
    res = analysis.instance_bootstrap(df, "a", "b", 10)
    # costs 10 (1 - d/100) are not exact in binary, so round before comparing
    assert set(np.unique(np.round(res["draws"], 9))) <= {1.0, 1.5, 2.0}
    assert res["mean"] == pytest.approx(1.5)


# ---------------------------------------------------------------- section 4.2 diagnostic

class _IdentityW:
    """Predictor whose cost equals the running sum W_k (cost units)."""

    def predict(self, cache, k, idx):
        return cache.W[np.asarray(idx, dtype=np.int64), k - 1]


class _StepW:
    """E0 for W below 0.505 B, E0/2 above: one drop of E0/2 between grid points
    W = 0.5 B and 0.51 B (B = 10, grid step 0.1)."""

    def __init__(self, E0):
        self.E0 = E0

    def predict(self, cache, k, idx):
        W = cache.W[np.asarray(idx, dtype=np.int64), k - 1]
        return self.E0 * np.where(W < 5.05, 1.0, 0.5)


class _Rising:
    def __init__(self, E0):
        self.E0 = E0

    def predict(self, cache, k, idx):
        W = cache.W[np.asarray(idx, dtype=np.int64), k - 1]
        return self.E0 * W / 10.0


class _Falling:
    """Drops by 0.02 E0 per grid step (grid step 0.1 on B = 10)."""

    def __init__(self, E0):
        self.E0 = E0

    def predict(self, cache, k, idx):
        W = cache.W[np.asarray(idx, dtype=np.int64), k - 1]
        return self.E0 * (2.0 - W / 5.0)


@pytest.fixture
def toy():
    spec = RouteSpec(inst="T", route_id=0, route=(0, 1, 2), m=3, B=10.0,
                     H=np.array([1.0, 1.0, 1.0]), E=np.array([4.0, 4.0, 4.0]), E0=4.0)
    stats = RouteStats(mu_p=np.array([1.0, 1.0, 1.0]), mu_d=np.array([0.5, 0.5, 0.5]),
                       sd_p=np.ones(3), sd_d=np.ones(3))
    return spec, stats


def test_grid_puts_the_stated_W_into_the_features(toy):
    spec, stats = toy
    grid, vals = analysis.w_grid_curve([_IdentityW()], spec, stats, 2, lo=0.0)
    assert grid.size == analysis.MONO_GRID == 101
    assert grid[0] == 0.0 and grid[-1] == 10.0
    # the identity predictor returns the W_2 that the cache was built with
    assert np.max(np.abs(vals - grid)) < 1e-9


def test_step_function_has_exactly_one_violating_neighbour_pair(toy):
    spec, stats = toy
    _, vals = analysis.w_grid_curve([_StepW(4.0)], spec, stats, 2, lo=0.0)
    res = analysis.monotonicity_diagnostic([vals], E0=4.0)
    # one drop of 2.0 = 0.5 E0 > 0.01 E0, all other steps flat
    assert res["n_steps"] == 100
    assert res["n_steps_bad"] == 1
    assert res["frac_steps"] == pytest.approx(0.01)
    assert res["frac_curves"] == 1.0


def test_rising_and_falling_curves(toy):
    spec, stats = toy
    _, up = analysis.w_grid_curve([_Rising(4.0)], spec, stats, 2, lo=0.0)
    _, down = analysis.w_grid_curve([_Falling(4.0)], spec, stats, 2, lo=0.0)
    r_up = analysis.monotonicity_diagnostic([up], E0=4.0)
    r_down = analysis.monotonicity_diagnostic([down], E0=4.0)
    assert r_up["n_steps_bad"] == 0 and r_up["frac_curves"] == 0.0
    # every step drops by 0.02 E0 = 0.08 > 0.04 = 0.01 E0
    assert r_down["n_steps_bad"] == 100 and r_down["frac_curves"] == 1.0


def test_small_drop_below_threshold_is_not_a_violation():
    # a drop of 0.005 E0 per step is inside the tolerance
    flat_ish = np.linspace(4.0, 4.0 - 0.005 * 4.0 * 100, 101)
    res = analysis.monotonicity_diagnostic([flat_ish], E0=4.0)
    assert res["n_steps_bad"] == 0


def test_curves_over_a_route_cover_each_stage_with_W_below_B(toy):
    spec, stats = toy
    rng = np.random.default_rng(0)
    g = rng.normal(0.5, 0.3, size=(200, 3))
    W = np.cumsum(g, axis=1)
    curves = analysis.monotonicity_curves({1: [_IdentityW()], 2: [_IdentityW()]},
                                          spec, stats, g)
    assert len(curves) == 2
    # stage 1 curve runs from min W_1 to B: its last value is B
    assert curves[0][-1] == pytest.approx(10.0)
    assert curves[0][0] == pytest.approx(W[:, 0].min())


# ---------------------------------------------------------------- variance (b), section 8

def test_per_day_paired_se_hand_value():
    # diffs 1, 2, 3: mean 2, sd (ddof 1) 1, SE = 1 / sqrt(3)
    assert analysis.per_day_paired_se([1.0, 2.0, 3.0], [0.0, 0.0, 0.0]) == \
        pytest.approx(1 / np.sqrt(3))


def test_per_day_paired_se_needs_two_days():
    with pytest.raises(ValueError):
        analysis.per_day_paired_se([1.0], [0.0])


def test_variance_b_median_from_npz_files(tmp_path):
    N = 100000
    # three routes with nn - baton day differences +c, -c, +c, -c (c = 1, 2, 4):
    # SE = c * sqrt(4/3) / 2 = c * 0.57735, i.e. 0.57735, 1.1547, 2.3094.
    # The median is 1.1547 (the mean would be 1.3471, so this distinguishes them).
    routes = []
    for rid, c in enumerate((1.0, 2.0, 4.0)):
        inst = "I0"
        routes.append((inst, rid))
        e = c * np.array([1.0, -1.0, 1.0, -1.0])
        base = np.full(4, 10.0)
        for rep in range(3):
            np.savez(tmp_path / f"{analysis.job_stem('full', inst, rid, N, rep)}.dev.npz",
                     nn_full=base + e)
            np.savez(tmp_path / f"{analysis.job_stem('ladder', inst, rid, N, rep)}.dev.npz",
                     baton_eq=base)
    got = analysis.variance_b_median(tmp_path, routes, N)
    assert got == pytest.approx(2.0 * np.sqrt(4 / 3) / 2)
    assert got == pytest.approx(1.1547005, abs=1e-6)


def test_job_stem_matches_sweep_key():
    from transvrp.expa.sweep import Job
    assert analysis.job_stem("full", "CON3-0", 3, 100000, 0) == \
        Job("neural", "CON3-0", 3, 100000, 0, arm="full").key
    assert analysis.job_stem("ladder", "CON3-0", 3, 100000, 2) == \
        Job("ladder", "CON3-0", 3, 100000, 2).key


# ---------------------------------------------------------------- summary wiring

def _synthetic_rows(seed=0):
    """Same construction as test_expa_analysis._synthetic_rows (20 instances x 2 routes)."""
    rng = np.random.default_rng(seed)
    grid = (100, 500, 2000, 10000, 100000)
    factor = {("nn_full", N): 1.5 for N in grid}
    factor.update({("baton_eq", N): 0.0 for N in grid})
    factor.update({("nn_scalar", 2000): 1.0, ("nn_scalar", 100000): 1.0,
                   ("nn_mlp", 100000): 0.5, ("boost", 100000): 1.0,
                   ("baton_big", 1000000): 0.0, ("dp_big", 1000000): 0.3,
                   ("none", 1000000): -5.0, ("oracle", 1000000): 0.0})
    for N in grid:
        factor.update({("none", N): -5.0, ("oracle", N): 0.0})
    rows = []
    for ii in range(20):
        inst = f"I{ii:02d}"
        for rid in range(2):
            base = 10.0 + rng.uniform(0, 1)
            for rep in range(3):
                for (pol, N), f in factor.items():
                    noise = rng.normal(0, 0.05)
                    cost = base * (1 - (f + noise) / 100.0) if pol not in ("none", "oracle") \
                        else base * (1 - f / 100.0)
                    rows.append(dict(inst=inst, route_id=rid, m=8, stratum="short", N=N,
                                     rep=rep, policy=pol, mean_cost=cost, handoff_rate=0.1,
                                     fail_rate=0.0, gap_closed=np.nan, n_eval=5000,
                                     eval_set="dev", fit_seconds=1.0, fit_commit="x",
                                     eval_commit="x"))
    from transvrp.expa.sweep import add_gap_closed
    return add_gap_closed(pd.DataFrame(rows))


def test_summary_reports_bootstrap_and_variance_b(tmp_path):
    from transvrp.expa.analysis import write_summary

    df = _synthetic_rows()
    rows_dir = tmp_path / "rows"
    rows_dir.mkdir()
    rng = np.random.default_rng(1)
    for inst in sorted(df["inst"].unique()):
        for rid in range(2):
            for rep in range(3):
                base = rng.uniform(9, 11, size=6)
                np.savez(rows_dir / f"{analysis.job_stem('full', inst, rid, 100000, rep)}.dev.npz",
                         nn_full=base + 0.1)
                np.savez(rows_dir / f"{analysis.job_stem('ladder', inst, rid, 100000, rep)}.dev.npz",
                         baton_eq=base)
    out = tmp_path / "out"
    res = write_summary(df, out, n_routes=40, rows_dir=rows_dir)
    text = (out / "summary.md").read_text()
    assert "Cluster bootstrap (sensitivity only" in text
    assert "seed 20261115" in text
    # nn_full - baton_eq = 0.1 on every day: the paired SE is 0 on every route
    assert res["variance_b"] == pytest.approx(0.0, abs=1e-12)
    assert "(b) median over routes" in text
    assert res["bootstrap"]["n_draws"] == 10_000
    assert res["bootstrap"]["mean"] == pytest.approx(res["primary"]["mean"])


def test_summary_without_rows_dir_says_variance_b_needs_arrays(tmp_path):
    from transvrp.expa.analysis import write_summary

    res = write_summary(_synthetic_rows(), tmp_path, n_routes=40)
    assert res["variance_b"] is None
    assert "needs the per-day arrays" in (tmp_path / "summary.md").read_text()
