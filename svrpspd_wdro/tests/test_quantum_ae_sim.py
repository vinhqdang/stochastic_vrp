"""Tests for quantum/ae_sim.py and scripts/run_quasar_m3b.py (QUASAR M3b v2/v3, review-fix revision):
QAE outcome distribution and exact error metrics, MLAE against brute force,
classical RMS law, the direct signed-mean estimator, the Grover rotation validation, cost-to-target
extraction, prediction helpers, log-log slopes, ratio tables, exact-check, stratified sampling and the
amplitude cache. Fast."""

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
    signed_mean_estimates,
    signed_mean_variance,
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


# ---------------- cost-to-target and prediction helpers ----------------


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
    assert predicted_cost_ratio("qae", v_sa, [vp], 1e-3, 1e-3) == pytest.approx(math.sqrt(vp / v_sa))
    assert predicted_cost_ratio("classical", v_sa, [vp], 1e-3, 1e-3) == pytest.approx(vp / v_sa)
    assert predicted_cost_ratio("mlae", v_sa, [vp, vm], 1e-3, 1e-3) == pytest.approx(
        2 * (math.sqrt(vp) + math.sqrt(vm)) / math.sqrt(v_sa))
    assert predicted_cost_ratio("classical", v_sa, [vp, vm], 1e-3, 1e-3) == pytest.approx(
        4 * (vp + vm) / v_sa)
    with pytest.raises(ValueError):
        predicted_cost_ratio("qae", v_sa, [], 1e-3, 1e-3)


# ---------------- direct signed-mean estimator ----------------


def test_signed_mean_variance_formula_and_one_sided_case():
    assert signed_mean_variance(0.03, 0.01) == pytest.approx(0.04 - 0.0004)
    assert signed_mean_variance(0.05, 0.0) == pytest.approx(0.05 * 0.95)  # one-sided: a(1 - a)


def test_signed_mean_estimates_are_unbiased_with_variance_var_over_n():
    rng = np.random.default_rng(3)
    pp, pm, N = 0.03, 0.01, 2000
    est = signed_mean_estimates(pp, pm, N, 20000, rng)
    truth = pp - pm
    var = signed_mean_variance(pp, pm)
    se = math.sqrt(var / N / est.size)
    assert abs(est.mean() - truth) < 6 * se
    assert est.var(ddof=1) == pytest.approx(var / N, rel=0.05)
    assert set(np.unique(np.round(est * N))) <= set(range(-N, N + 1))


def test_direct_classical_cost_follows_var_over_t_squared():
    """Direct signed-mean estimator: the 90 % quantile at N scales as sqrt(Var / N) (CLT), so cost ~ Var / T^2."""
    rng = np.random.default_rng(11)
    pp, pm = 0.02, 0.01
    var = signed_mean_variance(pp, pm)
    for N in (4000, 16000):
        est = signed_mean_estimates(pp, pm, N, 20000, rng)
        q90 = float(np.quantile(np.abs(est - (pp - pm)), 0.9, method="inverted_cdf"))
        assert q90 == pytest.approx(1.645 * math.sqrt(var / N), rel=0.08)


# ---------------- family groups, sampling and cache (script) ----------------


def test_group_of_maps_the_four_families_and_rejects_others():
    assert R.group_of("CON") == R.group_of("SCA") == "CON/SCA"
    assert R.group_of("CMT") == "CMT"
    assert R.group_of("city") == "city"
    with pytest.raises(ValueError):
        R.group_of("HANOI-100-1")


def test_record_seed_matches_spawn_child_of_the_previous_run():
    """Seeds are fixed by slot: the slot-keyed seed equals the child of spawn(n) for any n > slot."""
    kids = np.random.SeedSequence(R.V3_SEED).spawn(900)
    for slot in (0, 7, 587, 899):
        a = R._record_seed(slot).generate_state(4)
        b = kids[slot].generate_state(4)
        assert np.array_equal(a, b)


def _move_df() -> pd.DataFrame:
    base = dict(instance="I", family="CON", route_id=0, kind="insertion", j=1, pos=2, pos2=-1, rho_q=0.5)
    recs = [
        dict(base, cls=1, p_new=0.05, P_D1=0.02, P_Dm1=0.0),        # one-sided+, CON, decision
        dict(base, cls=2, p_new=0.01, P_D1=0.001, P_Dm1=0.0005),    # 200 and 100 hits: under threshold
        dict(base, cls=2, p_new=0.01, P_D1=0.001, P_Dm1=0.0006),    # 200 and 120 hits: under threshold
        dict(base, cls=2, p_new=0.01, P_D1=0.0011, P_Dm1=0.0010),   # 220 and 200 hits: signed, low
        dict(base, cls=3, p_new=0.5, P_D1=0.0, P_Dm1=0.0),          # zero difference
        dict(base, cls=3, p_new=0.2, P_D1=0.1, P_Dm1=0.0),          # boundary 0.2: decision
        dict(base, cls=3, p_new=0.2000001, P_D1=0.0, P_Dm1=0.05),   # just above 0.2: high, one-sided-
        dict(base, cls=3, p_new=0.0009, P_D1=0.0, P_Dm1=0.0004),    # 180 hits: under threshold
        dict(base, cls=3, p_new=0.02, P_D1=0.03, P_Dm1=0.0),        # boundary 0.02: decision
        dict(base, family="CMT", cls=1, p_new=0.05, P_D1=0.02, P_Dm1=0.0),   # CMT, cls1 decision
        dict(base, family="city", cls=3, p_new=0.3, P_D1=0.0, P_Dm1=0.05),   # city, cls3 high
    ]
    recs += [dict(base, cls=1, p_new=0.05, P_D1=0.02, P_Dm1=0.0)] * 40  # CON cls1 decision pool 41
    return pd.DataFrame(recs)


def test_population_and_build_pairs_eligibility_exclusions_and_bands():
    _, d = R.population_table(_move_df())
    info, records, pairs = R.build_pairs(d, seed=1)
    assert info["n_moves"] == 51
    assert info["moves_zero_difference"] == 1
    assert info["moves_under_threshold"] == 3
    assert info["moves_eligible"] == 47
    assert info["amp_P_D1_1_to_199_hits"] == 0
    assert info["amp_P_Dm1_1_to_199_hits"] == 3
    cells = {c["cell"]: c for c in info["cells"]}
    assert len(cells) == 27
    assert cells["cls1|decision|CON/SCA"]["pool"] == 41
    assert cells["cls1|decision|CON/SCA"]["sampled"] == R.MOVES_PER_CELL == 20
    assert cells["cls1|decision|CON/SCA"]["shortfall"] is False
    assert cells["cls1|decision|CMT"]["pool"] == 1 and cells["cls1|decision|CMT"]["sampled"] == 1
    assert cells["cls1|decision|CMT"]["shortfall"] is True
    assert cells["cls3|high|city"]["pool"] == 1
    assert cells["cls3|decision|CON/SCA"]["pool"] == 2  # p = 0.2 and p = 0.02
    assert cells["cls3|low|CON/SCA"]["pool"] == 0 and cells["cls3|low|CON/SCA"]["sampled"] == 0
    assert cells["cls2|low|CON/SCA"]["pool"] == 1
    assert {p["mode"] for p in pairs} == {"one-sided+", "one-sided-", "signed"}
    for p in pairs:
        assert len(p["components"]) == (2 if p["mode"] == "signed" else 1)
        assert p["base"] > 0.0
        assert p["group"] == R.group_of(p["ids"].split()[1])


def test_population_table_counts_zero_and_below_resolution_moves():
    pop, _ = R.population_table(_move_df())
    row = {(r["group"], r["cls"]): r for r in pop}
    con2 = row[("CON/SCA", 2)]
    assert con2["n_moves"] == 3 and con2["n_under"] == 2 and con2["n_eligible"] == 1
    assert row[("CON/SCA", 3)]["n_zero"] == 1
    assert row[("CMT", 1)]["n_eligible"] == 1
    assert row[("CON/SCA", 2)]["share_pnz_below_1e-3"] == 0.0  # P(D != 0) of the cls2 rows is 1.5e-3 to 2.1e-3
    # cls3 CON/SCA rows: zero-difference move (P(D != 0) = 0) and the 0.0009 move (P(D != 0) = 4e-4) out of five
    assert row[("CON/SCA", 3)]["share_pnz_below_1e-3"] == pytest.approx(0.4)


def test_build_pairs_is_deterministic_and_dedupes_amplitudes():
    _, d = R.population_table(_move_df())
    _, rec1, p1 = R.build_pairs(d, seed=7)
    _, rec2, p2 = R.build_pairs(d, seed=7)
    assert [p["ids"] for p in p1] == [p["ids"] for p in p2]
    assert [r["a"] for r in rec1] == [r["a"] for r in rec2]
    values = [r["a"] for r in rec1]
    assert len(values) == len(set(values))
    assert all(p["sa"] in {r["label"] for r in rec1} for p in p1)


def test_build_pairs_reuses_known_slots_and_numbers_new_ones_above_them():
    _, d = R.population_table(_move_df())
    _, base_records, _ = R.build_pairs(d, seed=7)
    target = base_records[0]
    known = {target["key"]: 42}  # pretend this amplitude was simulated in slot 42 earlier
    _, records, _ = R.build_pairs(d, seed=7, known_slots=known)
    by_key = {r["key"]: r for r in records}
    assert by_key[target["key"]]["slot"] == 42
    assert by_key[target["key"]]["label"] == "amp0042"
    new_slots = sorted(r["slot"] for r in records if r["key"] not in known)
    assert new_slots and min(new_slots) > 42
    assert len({r["slot"] for r in records}) == len(records)


def test_load_cache_reads_slots_and_rows(tmp_path):
    path = tmp_path / "metrics.csv"
    path.write_text(
        "amplitude,kind,a,family,setting,cost_A_apps_or_samples,evaluation,n_seeds,mean_abs_err,"
        "median_abs_err,q90_abs_err,q95_abs_err,rms,bias\n"
        "amp0007,standalone,0.25,qae,m=3,15,exact pmf,0,0.1,0.1,0.2,0.3,0.15,0.0\n"
        "amp0007,standalone,0.25,classical,N=16,16,sampled 4000,4000,0.1,0.1,0.2,0.3,0.15,0.0\n")
    known, rows = R.load_cache(path)
    assert known == {R._akey(0.25): 7}
    assert len(rows["amp0007"]) == 2
    assert rows["amp0007"][0]["cost_A_apps_or_samples"] == 15 and isinstance(rows["amp0007"][0]["a"], float)
    assert R.load_cache(tmp_path / "missing.csv") == ({}, {})


def test_simulate_skips_cached_records_and_reuse_check_detects_mismatch(monkeypatch):
    rec = {"label": "amp0000", "slot": 0, "a": 0.2, "kind": "standalone", "key": R._akey(0.2)}
    fake = [{"amplitude": "amp0000", "kind": "old", "a": 0.2, "family": "qae", "setting": "m=3",
             "cost_A_apps_or_samples": 15, "evaluation": "exact pmf", "n_seeds": 0,
             "mean_abs_err": 0.1, "median_abs_err": 0.1, "q90_abs_err": 0.2, "q95_abs_err": 0.3,
             "rms": 0.15, "bias": 0.0}]
    cache = {"amp0000": fake}
    rows = R._cached_rows(rec, cache)
    assert rows[0]["kind"] == "standalone" and rows[0]["amplitude"] == "amp0000"
    calls = []
    monkeypatch.setattr(R, "_simulate_record", lambda job: calls.append(job) or ("x", [], 0.0))
    monkeypatch.setattr(R, "_direct_job", lambda pr: (pr["pid"], [], 0.0))
    monkeypatch.setattr(R, "N_WORKERS", 1)
    out_rows, direct, n_new, _ = R.simulate([rec], [], cache)
    assert n_new == 0 and calls == [] and len(out_rows) == 1
    # the check compares a re-simulated amplitude with the cache; a mismatch must raise
    monkeypatch.setattr(R, "_simulate_record", lambda job: ("amp0000", [dict(
        fake[0], q90_abs_err=0.5)], 0.0))
    with pytest.raises(RuntimeError):
        R.reuse_check([rec], [], cache)


# ---------------- family-group ratio tables (script) ----------------


def _power_law_curves(amps: dict[str, float], pairs: list[dict]) -> dict:
    """Exact power-law error curves: quantum q90 = sqrt(v)/N, classical (binomial and direct) q90 = sqrt(v/N)."""
    qv = {}
    costs = np.array([2.0**e for e in range(2, 31)])
    labels = set()
    for p in pairs:
        labels.add(p["sa"])
        labels.update(lab for lab, _ in p["components"])
    for lab in labels:
        v = amps[lab] * (1 - amps[lab])
        for fam in R.RECORD_FAMILIES:
            val = np.sqrt(v / costs) if fam == "classical" else np.sqrt(v) / costs
            qv[(lab, fam)] = (costs, val)
    for p in pairs:
        var_D = signed_mean_variance(p["a_plus"], p["a_minus"])
        qv[(p["pid"], "classical_direct")] = (costs, np.sqrt(var_D / costs))
    return qv


def _pairs_for_powerlaw():
    base = dict(ids="x", group="CON/SCA", rho_q=0.5)
    return [
        dict(base, pid="p1", cell="cls1|decision|CON/SCA", cls=1, band="decision", mode="one-sided+", base=0.2,
             a_plus=0.05, a_minus=0.0, sa="s1", components=[("p1", 0.05)]),
        dict(base, pid="p2", cell="cls2|high|CMT", cls=2, band="high", mode="signed", base=0.1,
             a_plus=0.03, a_minus=0.01, sa="s2", components=[("p2", 0.03), ("m2", 0.01)], group="CMT"),
        dict(base, pid="p3", cell="cls3|low|city", cls=3, band="low", mode="one-sided-", base=0.3,
             a_plus=0.0, a_minus=0.02, sa="s3", components=[("m3", 0.02)], group="city"),
    ]


def test_ratio_rows_recover_predictions_on_power_laws_for_every_family():
    pairs = _pairs_for_powerlaw()
    amps = {"s1": 0.2, "p1": 0.05, "s2": 0.1, "p2": 0.03, "m2": 0.01, "s3": 0.3, "m3": 0.02}
    rows = R.ratio_rows(_power_law_curves(amps, pairs), pairs)
    assert len(rows) == len(pairs) * len(R.RATIO_FAMILIES) * len(R.TARGETS)  # no silent drops
    assert all(r["status"] == "ok" for r in rows)
    for r in rows:
        assert r["ratio"] == pytest.approx(r["pred"], rel=1e-9), r
        assert r["ratio_over_pred"] == pytest.approx(1.0, rel=1e-9)


def test_classical_column_is_the_direct_estimator_and_2est_is_the_protocol():
    pairs = _pairs_for_powerlaw()
    amps = {"s1": 0.2, "p1": 0.05, "s2": 0.1, "p2": 0.03, "m2": 0.01, "s3": 0.3, "m3": 0.02}
    rows = R.ratio_rows(_power_law_curves(amps, pairs), pairs, families=["classical", "classical_2est"],
                        targets=[("abs", 1e-3)])
    signed = {r["family"]: r for r in rows if r["cell"] == "cls2|high|CMT"}
    # direct: one estimate of D at T_D; prediction Var(D) / v_sa
    assert signed["classical"]["n_estimates"] == 1
    assert signed["classical"]["pred"] == pytest.approx(signed["classical"]["var_D"] / (0.1 * 0.9))
    # two-estimate protocol: two estimates at T_D / 2 each; costs about 4 times the direct estimator
    assert signed["classical_2est"]["n_estimates"] == 2
    vp, vm = 0.03 * 0.97, 0.01 * 0.99
    var_D = signed["classical"]["var_D"]
    assert signed["classical_2est"]["ratio"] / signed["classical"]["ratio"] == pytest.approx(
        4 * (vp + vm) / var_D, rel=1e-9)
    assert 3.9 < 4 * (vp + vm) / var_D < 4.0  # close to 4 because Var(D) = v+ + v- - (a+ - a-)^2


def test_ratio_rows_report_floor_and_fail_explicitly():
    pair = {"cell": "c|low|CON/SCA", "cls": 1, "band": "low", "group": "CON/SCA", "mode": "one-sided+", "base": 0.2,
            "a_plus": 0.05, "a_minus": 0.0, "sa": "s", "components": [("p", 0.05)], "pid": "q", "rho_q": 0.5, "ids": "x"}
    costs = np.array([4.0, 8.0])
    curve = (costs, np.array([0.5, 0.4]))
    qv = {("s", "qae"): curve, ("p", "qae"): curve, ("s", "classical"): curve, ("p", "classical"): curve,
          ("q", "classical_direct"): curve}
    rows = R.ratio_rows(qv, [pair], families=["qae"], targets=[("abs", 1e-3)])
    assert rows[0]["status"] == "fail" and math.isnan(rows[0]["ratio"])
    rows = R.ratio_rows(qv, [pair], families=["qae"], targets=[("abs", 1.0)])
    assert rows[0]["status"] == "floor" and math.isnan(rows[0]["ratio"])


def test_fitted_slope_recovers_power_laws_and_uses_last_settings():
    cost = 2.0 ** np.arange(1, 25)
    assert R.fitted_slope(cost, 3.0 / cost) == pytest.approx(-1.0, abs=1e-12)
    assert R.fitted_slope(cost, 0.5 / np.sqrt(cost)) == pytest.approx(-0.5, abs=1e-12)
    val = 1.0 / cost
    val[:-R.ASYMPTOTIC_LAST] = 5.0  # early settings are not used
    assert R.fitted_slope(cost, val) == pytest.approx(-1.0, abs=1e-12)
    zero_last = np.ones(cost.size)
    zero_last[-1] = 0.0
    assert math.isnan(R.fitted_slope(cost, zero_last))


def test_ratio_summary_counts_and_medians_use_ok_rows_only():
    def row(status, ratio, pred, mode="one-sided+"):
        return {"cell": "cls1|decision|CON/SCA", "family": "qae", "target_type": "abs", "target_value": 1e-3,
                "status": status, "ratio": ratio, "pred": pred,
                "ratio_over_pred": ratio / pred if status == "ok" else float("nan"), "mode": mode}
    rr = [row("ok", 0.2, 0.25), row("ok", 0.4, 0.5), row("ok", 0.9, 1.0), row("floor", float("nan"), 0.3),
          row("fail", float("nan"), 0.3, mode="signed")]
    (s,) = R.ratio_summary(rr)
    assert (s["n_total"], s["n_ok"], s["n_floor"], s["n_fail"]) == (5, 3, 1, 1)
    assert s["median_ratio"] == pytest.approx(0.4)
    assert s["median_pred"] == pytest.approx(0.5)
    assert s["median_ratio_over_pred"] == pytest.approx(0.8)
    assert s["cls"] == 1 and s["band"] == "decision" and s["group"] == "CON/SCA"
    assert s["mode_mix"] == "1 signed / 4 one-sided"


def test_group_summary_separates_families_and_counts_fail_rows():
    pairs = _pairs_for_powerlaw()
    amps = {"s1": 0.2, "p1": 0.05, "s2": 0.1, "p2": 0.03, "m2": 0.01, "s3": 0.3, "m3": 0.02}
    rr = R.ratio_rows(_power_law_curves(amps, pairs), pairs, families=["qae"], targets=[("abs", 1e-3)])
    gs = {(g["group"], g["family"], g["target_type"], g["target_value"]): g for g in R.group_summary(rr)}
    gs = {(k[0], k[1]): v for k, v in gs.items() if (k[2], k[3]) == ("abs", 1e-3)}
    assert gs[("CMT", "qae")]["n_ok"] == 1 and gs[("city", "qae")]["n_ok"] == 1
    assert gs[("CON/SCA", "qae")]["n_ok"] == 1
    assert gs[("CON/SCA", "mlae_S32")]["n_total"] == 0


def test_exact_check_runs_and_agrees_with_sampling(monkeypatch):
    monkeypatch.setattr(R, "EXACT_CHECK_N", 4000)
    out = R.exact_check()
    assert len(out) == 6
    for r in out:
        assert r["exact_mean_abs"] > 0 and r["exact_q90"] >= 0 and r["exact_rms"] > 0
        assert abs(r["z_mean_abs"]) < 6.0
        assert r["sampled_q90"] == pytest.approx(r["exact_q90"], abs=max(r["exact_q90"], 1e-3) * 1.5)


def test_band_boundaries():
    got = R.band_of(np.array([0.001, 0.0199999, 0.02, 0.2, 0.2000001, 0.9]))
    assert list(got) == ["low", "low", "decision", "decision", "high", "high"]
