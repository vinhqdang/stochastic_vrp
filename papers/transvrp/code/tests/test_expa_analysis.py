"""M10: decision rules and summary tables (EXPERIMENT_A.md section 9, M10).

All decision tests use the deterministic constructed 20-vectors of M10 with
L = linspace(-1, 1, 20). No test data is read.
"""
import numpy as np
import pandas as pd
import pytest

from transvrp.expa import analysis
from transvrp.expa.sweep import add_gap_closed

L = np.linspace(-1, 1, 20)
SYN_ROUTES = 40          # the synthetic table has 20 instances x 2 routes


def _go(d):
    return analysis.go_no_go(np.asarray(d, dtype=float), delta=0.7)


# ---------------------------------------------------------------- go/no-go (1.4)

def test_clear_win_and_clear_loss():
    assert _go(2 + 0.1 * L)["branch"] == "WIN"
    assert _go(-2 + 0.1 * L)["branch"] == "LOSE"


def test_tie_strong_and_tie_weak():
    assert _go(0.1 * L)["branch"] == "TIE_STRONG"
    assert _go(5 * L)["branch"] == "TIE_WEAK"


def test_delta_boundary_is_included_in_win():
    res = _go(0.7 + 0.05 * L)
    assert res["mean"] == pytest.approx(0.7, abs=1e-9)
    assert res["branch"] == "WIN"


def test_just_below_delta_is_a_strong_tie_flagged_better():
    res = _go(0.6 + 0.05 * L)
    assert res["branch"] == "TIE_STRONG"
    assert res["significance"] == "better"


def test_small_significant_effect_is_strong_tie_flagged_better():
    res = _go(0.3 + 0.05 * L)
    assert res["branch"] == "TIE_STRONG"
    assert res["significance"] == "better"


def test_win_count_boundary_thirteen_is_win():
    d = np.r_[1.5 + 0.01 * np.arange(13), -0.1 - 0.01 * np.arange(7)]
    res = _go(d)
    assert res["n_pos"] == 13
    assert res["branch"] == "WIN"


def test_win_count_boundary_twelve_is_tie_weak_but_significant():
    d = np.r_[1.5 + 0.01 * np.arange(12), -0.1 - 0.01 * np.arange(8)]
    res = _go(d)
    # exact mean is 0.879; the spec quotes it rounded to 0.88
    assert res["mean"] == pytest.approx(0.879, abs=1e-9)
    assert res["n_pos"] == 12
    assert res["p_greater"] < analysis.ALPHA_DIRECTION
    assert res["branch"] == "TIE_WEAK"
    assert res["significance"] == "better"


# ---------------------------------------------------------------- four-way rule (1.5)

def test_classify_four_way_rule():
    assert analysis.classify(2 + 0.1 * L) == "better"
    assert analysis.classify(-2 + 0.1 * L) == "worse"
    assert analysis.classify(0.1 * L) == "equivalent"
    assert analysis.classify(5 * L) == "inconclusive"


def test_classify_below_delta_is_not_better():
    # significant and positive, but Delta-bar < delta and CI90 inside (-delta, delta)
    assert analysis.classify(0.6 + 0.05 * L) == "equivalent"


def test_classify_uses_supplied_holm_p():
    # a clearly significant WIN-like vector is not 'better' if the adjusted p is 0.2
    assert analysis.classify(2 + 0.1 * L, p_adj=0.2) == "inconclusive"


# ---------------------------------------------------------------- Holm

def test_holm_values():
    assert analysis.holm([0.01, 0.04, 0.03]) == [0.03, 0.06, 0.06]


def test_holm_is_monotone_and_capped():
    adj = analysis.holm([0.5, 0.6, 0.001])
    assert adj[2] == pytest.approx(0.003)
    assert max(adj) <= 1.0
    assert all(a >= p for a, p in zip(adj, [0.5, 0.6, 0.001]))


# ---------------------------------------------------------------- tables

def _synthetic_rows(seed=0):
    """Route-level costs for all policies used by the summary. Instance-level
    noise makes SDs non-zero. Route means are constructed so that
    nn_full vs baton_eq at every N is about +1.5%, and everything else is
    close to 0 except the designed gaps."""
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
                    rows.append(dict(inst=inst, route_id=rid, m=8, stratum="short",
                                     N=N, rep=rep, policy=pol, mean_cost=cost,
                                     handoff_rate=0.1, fail_rate=0.0,
                                     gap_closed=np.nan, n_eval=5000, eval_set="dev",
                                     fit_seconds=1.0, fit_commit="x", eval_commit="x"))
    df = pd.DataFrame(rows)
    return add_gap_closed(df)


def test_instance_deltas_constructed():
    rows = []
    for i, d in enumerate(L):
        for rep in range(2):
            rows.append(dict(inst=f"I{i:02d}", route_id=0, rep=rep, N=10,
                             policy="a", mean_cost=10 * (1 - d / 100.0)))
            rows.append(dict(inst=f"I{i:02d}", route_id=0, rep=rep, N=10,
                             policy="b", mean_cost=10.0))
    df = pd.DataFrame(rows)
    got = analysis.instance_deltas(df, "a", "b", 10)
    assert np.allclose(got.to_numpy(), L, atol=1e-9)


def test_instance_deltas_refuses_unmatched_rows():
    df = pd.DataFrame([dict(inst="I0", route_id=0, rep=0, N=10, policy="a", mean_cost=1.0),
                       dict(inst="I0", route_id=0, rep=0, N=10, policy="b", mean_cost=1.0),
                       dict(inst="I0", route_id=0, rep=1, N=10, policy="a", mean_cost=1.0)])
    with pytest.raises(ValueError):
        analysis.instance_deltas(df, "a", "b", 10)


def test_crossover_rule():
    c = pd.DataFrame(dict(N=[100, 500, 2000], ci95_lo=[-2.0, -0.5, -0.1]))
    assert analysis.crossover_n(c) == 500
    c2 = pd.DataFrame(dict(N=[100, 500, 2000], ci95_lo=[-2.0, -0.1, -2.0]))
    assert analysis.crossover_n(c2) is None


def test_summary_end_to_end_on_synthetic_table(tmp_path):
    df = _synthetic_rows()
    res = analysis.write_summary(df, tmp_path, n_routes=SYN_ROUTES)
    assert res["primary"]["branch"] == "WIN"
    assert res["primary"]["n_pos"] == 20
    assert (tmp_path / "summary.md").exists()
    text = (tmp_path / "summary.md").read_text()
    assert "Branch: **WIN**" in text
    assert "S1" in text and "S11" in text
    cur = pd.read_csv(tmp_path / "curve.csv")
    assert list(cur["N"]) == [100, 500, 2000, 10000, 100000]
    assert res["crossover_n"] == 100
    assert res["family"]["S1"]["classification"] == "better"


def test_missing_family_contrast_is_refused_not_dropped(tmp_path):
    """Without the 10^6 rows S7, S10 and S11 cannot be computed. The summary must
    refuse (Holm is over the 11 pre-registered tests), not run over 8 tests."""
    df = _synthetic_rows()
    df = df[df["N"] != 1000000]
    with pytest.raises(ValueError, match="S7"):
        analysis.write_summary(df, tmp_path, n_routes=SYN_ROUTES)
    assert not (tmp_path / "summary.md").exists()


def test_missing_primary_rows_are_refused(tmp_path):
    df = _synthetic_rows()
    df = df[~((df["policy"] == "baton_eq") & (df["N"] == 100000))]
    with pytest.raises(ValueError, match="PRIMARY"):
        analysis.write_summary(df, tmp_path, n_routes=SYN_ROUTES)


def test_contrast_does_not_swallow_incomplete_coverage():
    df = _synthetic_rows()
    partial = df[~((df["policy"] == "nn_scalar") & (df["N"] == 2000)
                   & (df["inst"] == "I00"))]
    with pytest.raises(ValueError):
        analysis.contrast(partial, *analysis.FAMILY[7])     # S8 needs nn_scalar @ 2000


def test_family_is_eleven_tests_and_defaults_are_the_spec_counts():
    assert len(analysis.FAMILY) == 11 == analysis.FAMILY_SIZE
    assert analysis.N_INSTANCES == 20
    assert analysis.N_ROUTES == 62
    assert analysis.N_REPS == 3


def test_coverage_counts_are_checked(tmp_path):
    df = _synthetic_rows()
    with pytest.raises(ValueError, match="routes"):
        analysis.write_summary(df, tmp_path)                 # 40 routes, default 62
    with pytest.raises(ValueError, match="instances"):
        analysis.write_summary(df, tmp_path, n_instances=19, n_routes=SYN_ROUTES)
    with pytest.raises(ValueError, match="replicates"):
        short = df[~((df["policy"] == "nn_full") & (df["rep"] == 2))]
        analysis.write_summary(short, tmp_path, n_routes=SYN_ROUTES)


def test_holm_runs_over_all_eleven_family_p_values(tmp_path):
    df = _synthetic_rows()
    res = analysis.write_summary(df, tmp_path, n_routes=SYN_ROUTES)
    ids = [spec[0] for spec in analysis.FAMILY]
    p_raw = [res["family"][i]["p_two_sided"] for i in ids]
    p_adj = [res["family"][i]["p_holm"] for i in ids]
    assert np.allclose(p_adj, analysis.holm(p_raw), rtol=0, atol=1e-12)
    assert len(ids) == 11


def test_cli_reads_csv_and_writes_outputs(tmp_path):
    df = _synthetic_rows()
    merged = tmp_path / "main.csv"
    df.to_csv(merged, index=False)
    out = tmp_path / "out"
    assert analysis.main(["--merged", str(merged), "--out", str(out),
                          "--routes", str(SYN_ROUTES)]) == 0
    assert (out / "summary.md").exists() and (out / "curve.csv").exists()


def test_big_stage_rows_shared_with_main_are_not_double_counted(tmp_path):
    df = _synthetic_rows()
    big = df[(df["N"] == 1000000) & (df["rep"] == 0)].copy()
    big = big[big["policy"].isin(["baton_big", "dp_big", "none", "oracle"])]
    res = analysis.write_summary(df, tmp_path, big_df=big, n_routes=SYN_ROUTES)
    assert res["family"]["S10"]["n"] == 20
    assert res["family"]["S11"]["n"] == 20


def test_big_stage_disagreeing_with_main_is_refused(tmp_path):
    df = _synthetic_rows()
    big = df[(df["N"] == 1000000) & (df["rep"] == 0)].copy()
    big["mean_cost"] = big["mean_cost"] * 1.01
    with pytest.raises(ValueError, match="disagree"):
        analysis.write_summary(df, tmp_path, big_df=big, n_routes=SYN_ROUTES)
