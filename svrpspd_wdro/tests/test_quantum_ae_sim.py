"""Tests for quantum/ae_sim.py and scripts/run_quasar_m3b.py (QUASAR M3b v2/v3):
QAE outcome distribution and exact error metrics, MLAE against brute force,
classical RMS law, the Grover rotation validation, cost-to-target extraction,
prediction helpers, ratio table and move sampling. Fast."""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quantum.ae_sim import (
    cost_at,
    exact_qae_metrics,
    grover_validation,
    mlae_cost,
    mlae_counts,
    mlae_estimates,
    predicted_cost_ratio,
    qae_cost,
    qae_estimates,
    qae_outcome_pmf,
    rms_vs_cost,
    sample_metrics,
    target_scales,
    theta_of,
)
from quantum.classical import DiscreteDemand
from quantum.oracle import build_peak_oracle

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_quasar_m3b.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("run_quasar_m3b", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


R = _load_script()


def test_qae_outcome_distribution_normalised_and_mode_near_phase():
    a, m = 0.1, 8
    N = 1 << m
    pmf = qae_outcome_pmf(a, m)
    assert pmf.shape == (N,)
    assert np.all(pmf >= 0.0)
    assert abs(pmf.sum() - 1.0) < 1e-12
    phase_y = N * theta_of(a) / math.pi
    assert abs(int(np.argmax(pmf)) - phase_y) <= 1.0


@pytest.mark.parametrize("a", [0.0, 0.001, 0.3, 0.5, 1.0])
def test_qae_pmf_normalised_for_edge_amplitudes(a):
    for m in (1, 3, 7):
        pmf = qae_outcome_pmf(a, m)
        assert abs(pmf.sum() - 1.0) < 1e-12
        assert np.all(np.isfinite(pmf))


def test_qae_cost_convention():
    assert qae_cost(3) == 2 * 7 + 1
    assert qae_cost(12) == 2 * 4095 + 1


def test_mlae_converges_as_J_grows_on_fixed_amplitude():
    a = 0.1
    rows = rms_vs_cost(a, "mlae", [(2, 64), (8, 64)], n_seeds=300, rng=11)
    rms_small, rms_large = rows[0][2], rows[1][2]
    assert rms_large < rms_small / 5.0
    assert mlae_cost(8, 64) == 64 * sum(2 * k + 1 for k in [0, 1, 2, 4, 8, 16, 32, 64, 128])


def test_classical_rms_follows_sqrt_law():
    a, N, n_seeds = 0.1, 4096, 4000
    (_, cost, rms, bias), = rms_vs_cost(a, "classical", [N], n_seeds=n_seeds, rng=5)
    assert cost == N
    expected = math.sqrt(a * (1 - a) / N)
    assert 0.93 * expected < rms < 1.07 * expected
    assert abs(bias) < 4 * expected / math.sqrt(n_seeds)


def _tiny_oracle():
    V = 2
    n = 3
    pd = np.zeros((n, V))
    pp = np.zeros((n, V))
    pd[:, 0] = 0.6
    pd[:, 1] = 0.4
    pp[:, 0] = 0.5
    pp[:, 1] = 0.5
    pd[0] = [1.0, 0.0]
    pp[0] = [1.0, 0.0]
    dem = DiscreteDemand(pmf_d=pd, pmf_p=pp, unit=1.0)
    return build_peak_oracle([1, 2], dem, 1)


def test_grover_validation_matches_rotation_model_on_tiny_oracle():
    oracle = _tiny_oracle()
    rows = grover_validation(oracle, ks=range(4))
    assert [r[0] for r in rows] == [0, 1, 2, 3]
    for k, p_circ, p_model in rows:
        assert abs(p_circ - p_model) < 1e-9, (k, p_circ, p_model)


def test_qae_sampler_matches_exact_rms():
    a, m = 0.1, 6
    N = 1 << m
    pmf = qae_outcome_pmf(a, m)
    est_grid = np.sin(np.pi * np.arange(N) / N) ** 2
    exact_rms = math.sqrt(float(np.sum(pmf * (est_grid - a) ** 2)))
    (_, _, rms, _), = rms_vs_cost(a, "qae", [m], n_seeds=20000, rng=3)
    assert abs(rms - exact_rms) < 0.03 * exact_rms


@pytest.mark.parametrize("a", [0.001, 0.05, 0.25, 0.5, 0.95])
def test_qae_pmf_meets_brassard_guarantee(a):
    # P(|a_hat - a| <= 2 pi sqrt(a(1-a))/M + pi^2/M^2) >= 8/pi^2 with M = 2^m.
    for m in (3, 6, 9):
        M = 1 << m
        pmf = qae_outcome_pmf(a, m)
        est = np.sin(np.pi * np.arange(M) / M) ** 2
        bound = 2 * math.pi * math.sqrt(a * (1 - a)) / M + math.pi**2 / M**2
        mass = float(pmf[np.abs(est - a) <= bound].sum())
        assert mass >= 8 / math.pi**2 - 1e-9, (a, m, mass)


# ---------------- v3: exact metrics, quantiles, MLAE ----------------


def test_exact_qae_mean_abs_error_matches_large_sample():
    a, m, n = 0.07, 6, 400_000
    ex = exact_qae_metrics(a, m)
    est = qae_estimates(a, m, n, np.random.default_rng(99))
    ae = np.abs(est - a)
    se = float(np.std(ae, ddof=1) / math.sqrt(n))
    assert abs(float(np.mean(ae)) - ex["mean_abs_err"]) < 5 * se


def test_exact_quantiles_satisfy_cdf_definition_and_ordering():
    a, m = 0.2, 7
    N = 1 << m
    pmf = qae_outcome_pmf(a, m)
    err = np.sin(np.pi * np.arange(N) / N) ** 2 - a
    ae = np.abs(err)
    ex = exact_qae_metrics(a, m)
    assert ex["median_abs_err"] <= ex["q90_abs_err"] <= ex["q95_abs_err"]
    for q, key in ((0.9, "q90_abs_err"), (0.95, "q95_abs_err")):
        x = ex[key]
        assert float(pmf[ae <= x + 1e-15].sum()) >= q - 1e-12
        assert float(pmf[ae < x - 1e-15].sum()) < q
    assert ex["mean_abs_err"] == pytest.approx(float(np.sum(pmf * ae)), rel=1e-12)
    assert ex["rms"] == pytest.approx(math.sqrt(float(np.sum(pmf * err * err))), rel=1e-12)
    assert ex["rms"] >= ex["mean_abs_err"]  # Cauchy-Schwarz


def test_sample_metrics_definitions_on_known_values():
    a = 0.5
    est = np.array([0.4, 0.5, 0.6, 0.9])  # |err| = 0.1, 0, 0.1, 0.4
    met = sample_metrics(est, a)
    assert met["mean_abs_err"] == pytest.approx(0.15)
    assert met["median_abs_err"] == pytest.approx(0.1)
    assert met["q90_abs_err"] == pytest.approx(0.4)  # inverted-cdf: smallest x with ecdf >= 0.9
    assert met["bias"] == pytest.approx(0.1)  # errors -0.1, 0, 0.1, 0.4


def test_mlae_matches_brute_force_argmax():
    J, S, a = 3, 64, 0.2
    H = mlae_counts(a, J, S, 40, np.random.default_rng(2))
    est = mlae_estimates(H, J, S)
    ks = np.array([0] + [2**j for j in range(J)], float)
    th = np.linspace(0.0, math.pi / 2.0, 200_001)
    ang = (2 * ks[:, None] + 1) * th[None, :]
    lp = 2 * np.log(np.maximum(np.abs(np.sin(ang)), 1e-300))
    lq = 2 * np.log(np.maximum(np.abs(np.cos(ang)), 1e-300))
    brute = np.sin(th[np.argmax(H @ lp + (S - H) @ lq, axis=1)]) ** 2
    assert float(np.max(np.abs(est - brute))) < 1e-5


def test_mlae_at_zero_hits_returns_near_zero():
    H = np.zeros((5, 3))  # J = 2: schedule (0, 1, 2), no hits
    est = mlae_estimates(H, 2, 32)
    assert np.all(est < 1e-7)


# ---------------- v3: cost-to-target and prediction helpers ----------------


def test_cost_at_uses_suffix_maximum_and_interpolates_loglog():
    cost = np.array([1.0, 2.0, 4.0, 8.0, 16.0, 32.0])
    val = np.array([0.5, 0.2, 0.05, 0.3, 0.04, 0.01])  # dip at cost 4 is not credited
    c, floor = cost_at(cost, val, 0.1)
    t = (math.log(0.1) - math.log(0.3)) / (math.log(0.04) - math.log(0.3))
    assert not floor
    assert c == pytest.approx(8.0 * 2.0**t, rel=1e-12)


def test_cost_at_floor_fail_and_zero_cases():
    cost = np.array([1.0, 2.0])
    assert cost_at(cost, np.array([0.2, 0.05]), 1.0) == (1.0, True)
    c, floor = cost_at(cost, np.array([0.2, 0.05]), 0.001)
    assert math.isnan(c) and not floor
    assert cost_at(cost, np.array([0.2, 0.0]), 0.1) == (2.0, False)


def test_target_scales_absolute_and_relative():
    assert target_scales(0.2, 0.05, "abs", 1e-3) == (1e-3, 1e-3)
    T_sa, T_D = target_scales(0.2, 0.05, "rel", 0.1)
    assert T_sa == pytest.approx(0.02)
    assert T_D == pytest.approx(0.005)
    with pytest.raises(ValueError):
        target_scales(0.2, 0.05, "log", 0.1)


def test_predicted_cost_ratio_closed_forms():
    v_sa, vp, vm = 0.16, 0.04, 0.01
    # one-sided, absolute: sqrt(vp / v_sa) = rho_q
    assert predicted_cost_ratio("qae", v_sa, [vp], 1e-3, 1e-3) == pytest.approx(math.sqrt(vp / v_sa))
    assert predicted_cost_ratio("classical", v_sa, [vp], 1e-3, 1e-3) == pytest.approx(vp / v_sa)
    # signed, absolute: 2 (sqrt v+ + sqrt v-) / sqrt(v_sa) and 4 (v+ + v-) / v_sa
    assert predicted_cost_ratio("mlae", v_sa, [vp, vm], 1e-3, 1e-3) == pytest.approx(
        2 * (math.sqrt(vp) + math.sqrt(vm)) / math.sqrt(v_sa))
    assert predicted_cost_ratio("classical", v_sa, [vp, vm], 1e-3, 1e-3) == pytest.approx(
        4 * (vp + vm) / v_sa)
    with pytest.raises(ValueError):
        predicted_cost_ratio("qae", v_sa, [], 1e-3, 1e-3)


# ---------------- v3: ratio table and move sampling (script) ----------------


def _power_law_curves(amps: dict[str, float], pairs: list[dict]) -> dict:
    """Exact power-law error curves: quantum q90 = sqrt(v)/N (cost N), classical q90 = sqrt(v/N)."""
    qv = {}
    costs = np.array([2.0**e for e in range(2, 31)])
    labels = set()
    for p in pairs:
        labels.add(p["sa"])
        labels.update(lab for lab, _ in p["components"])
    for lab in labels:
        a = amps[lab]
        v = a * (1 - a)
        for fam in R.FAMILIES:
            if fam == "classical":
                val = np.sqrt(v / costs)
            else:
                val = np.sqrt(v) / costs
            qv[(lab, fam)] = (costs, val)
    return qv


def test_ratio_rows_recover_predictions_on_power_laws():
    pairs = [
        {"cell": "cls1|decision", "cls": 1, "band": "decision", "mode": "one-sided+", "base": 0.2,
         "a_plus": 0.05, "a_minus": 0.0, "sa": "s1", "components": [("p1", 0.05)], "rho_q": 0.5, "ids": "x"},
        {"cell": "cls2|high", "cls": 2, "band": "high", "mode": "signed", "base": 0.1,
         "a_plus": 0.03, "a_minus": 0.01, "sa": "s2", "components": [("p2", 0.03), ("m2", 0.01)],
         "rho_q": 0.4, "ids": "y"},
        {"cell": "cls3|low", "cls": 3, "band": "low", "mode": "one-sided-", "base": 0.3,
         "a_plus": 0.0, "a_minus": 0.02, "sa": "s3", "components": [("m3", 0.02)], "rho_q": 0.1, "ids": "z"},
    ]
    amps = {"s1": 0.2, "p1": 0.05, "s2": 0.1, "p2": 0.03, "m2": 0.01, "s3": 0.3, "m3": 0.02}
    rows = R.ratio_rows(_power_law_curves(amps, pairs), pairs)
    assert len(rows) == len(pairs) * len(R.FAMILIES) * len(R.TARGETS)  # no silent drops
    assert all(r["status"] == "ok" for r in rows)
    for r in rows:
        assert r["ratio"] == pytest.approx(r["pred"], rel=1e-9), r
        assert r["ratio_over_pred"] == pytest.approx(1.0, rel=1e-9)


def test_ratio_rows_report_floor_and_fail_explicitly():
    pair = {"cell": "c", "cls": 1, "band": "low", "mode": "one-sided+", "base": 0.2,
            "a_plus": 0.05, "a_minus": 0.0, "sa": "s", "components": [("p", 0.05)], "rho_q": 0.5, "ids": "x"}
    costs = np.array([4.0, 8.0])
    qv = {("s", "qae"): (costs, np.array([0.5, 0.4])), ("p", "qae"): (costs, np.array([0.5, 0.4])),
          ("s", "classical"): (costs, np.array([0.5, 0.4])), ("p", "classical"): (costs, np.array([0.5, 0.4]))}
    for fam in R.FAMILIES:
        qv.setdefault(("s", fam), (costs, np.array([0.5, 0.4])))
        qv.setdefault(("p", fam), (costs, np.array([0.5, 0.4])))
    rows = R.ratio_rows(qv, [pair], families=["qae"], targets=[("abs", 1e-3)])
    assert rows[0]["status"] == "fail" and math.isnan(rows[0]["ratio"])
    rows = R.ratio_rows(qv, [pair], families=["qae"], targets=[("abs", 1.0)])
    assert rows[0]["status"] == "floor" and math.isnan(rows[0]["ratio"])


def _move_df() -> pd.DataFrame:
    base = dict(instance="I", family="CON", route_id=0, kind="insertion", j=1, pos=2, pos2=-1, rho_q=0.5)
    recs = [
        dict(base, cls=1, p_new=0.05, P_D1=0.02, P_Dm1=0.0),        # one-sided+, decision
        dict(base, cls=2, p_new=0.01, P_D1=0.001, P_Dm1=0.0005),    # 200 and 100 hits: under threshold
        dict(base, cls=2, p_new=0.01, P_D1=0.001, P_Dm1=0.0006),    # 200 and 120 hits: under threshold
        dict(base, cls=2, p_new=0.01, P_D1=0.0011, P_Dm1=0.0010),   # 220 and 200 hits: signed, low
        dict(base, cls=3, p_new=0.5, P_D1=0.0, P_Dm1=0.0),          # zero difference
        dict(base, cls=3, p_new=0.2, P_D1=0.1, P_Dm1=0.0),          # boundary 0.2: decision
        dict(base, cls=3, p_new=0.2000001, P_D1=0.0, P_Dm1=0.05),   # just above 0.2: high, one-sided-
        dict(base, cls=3, p_new=0.0009, P_D1=0.0, P_Dm1=0.0004),    # 180 hits: under threshold
        dict(base, cls=3, p_new=0.02, P_D1=0.03, P_Dm1=0.0),        # boundary 0.02: decision
    ]
    recs += [dict(base, cls=1, p_new=0.05, P_D1=0.02, P_Dm1=0.0)] * 40  # pool of 41 in cls1|decision
    return pd.DataFrame(recs)


def test_build_pairs_eligibility_exclusions_and_bands():
    info, records, pairs = R.build_pairs(_move_df(), seed=1)
    assert info["n_moves"] == 49
    assert info["moves_zero_difference"] == 1
    assert info["moves_under_threshold"] == 3  # rows 2, 3 and 8 of the table
    assert info["moves_eligible"] == 45
    assert info["amp_P_D1_1_to_199_hits"] == 0
    assert info["amp_P_Dm1_1_to_199_hits"] == 3
    cells = {c["cell"]: c for c in info["cells"]}
    assert len(cells) == 9
    assert cells["cls1|decision"]["pool"] == 41 and cells["cls1|decision"]["sampled"] == R.MOVES_PER_CELL
    assert cells["cls3|decision"]["pool"] == 2  # p = 0.2 and p = 0.02
    assert cells["cls3|high"]["pool"] == 1 and cells["cls3|low"]["pool"] == 0
    assert cells["cls2|low"]["pool"] == 1
    by_mode = {p["mode"] for p in pairs}
    assert by_mode == {"one-sided+", "one-sided-", "signed"}
    for p in pairs:
        assert len(p["components"]) == (2 if p["mode"] == "signed" else 1)
        assert p["base"] > 0.0


def test_build_pairs_is_deterministic_and_dedupes_amplitudes():
    df = _move_df()
    _, rec1, p1 = R.build_pairs(df, seed=7)
    _, rec2, p2 = R.build_pairs(df, seed=7)
    assert [p["ids"] for p in p1] == [p["ids"] for p in p2]
    assert [r["a"] for r in rec1] == [r["a"] for r in rec2]
    values = [r["a"] for r in rec1]
    assert len(values) == len(set(values))
    sa_labels = [p["sa"] for p in p1]
    assert all(lab in {r["label"] for r in rec1} for lab in sa_labels)


def test_band_boundaries():
    got = R.band_of(np.array([0.001, 0.0199999, 0.02, 0.2, 0.2000001, 0.9]))
    assert list(got) == ["low", "low", "decision", "decision", "high", "high"]
