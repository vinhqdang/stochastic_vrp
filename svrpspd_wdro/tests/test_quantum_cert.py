"""Tests for quantum/cert.py and scripts/run_quasar_cert.py (QUASAR cert).

Covers the amplitude-estimation bound h, the median success constants s_k and
beta(r, k), the minimal r_needed, the variance bound vbar, the Lemma 2 consistent
interval (numerically), the Q-CERT and C-CERT worst-case schedules, and the
summary script on a tiny synthetic risk/plan pair. Fast."""

from __future__ import annotations

import csv
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pytest

from quantum.cert import (
    beta,
    best_k,
    c_cert_route,
    eta_round,
    h,
    q_cert_route,
    r_needed,
    s_k,
    vbar,
)
from quantum.cert import VAR_MAX

PI = math.pi
ALPHA = 0.10
DELTA = 0.05
_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_quasar_cert.py"


def _h_vec(M, k, q):
    q = np.asarray(q, dtype=np.float64)
    return 2 * PI * k * np.sqrt(q * (1 - q)) / M + PI ** 2 * k ** 2 / M ** 2


def test_h_hand_values():
    assert h(1, 1, 0.25) == pytest.approx(PI + PI ** 2)
    assert h(2, 3, 0.25) == pytest.approx(2 * PI * 3 * 0.5 / 2 + PI ** 2 * 9 / 4)
    assert h(4, 1, 0.0) == pytest.approx(PI ** 2 / 16)


def test_h_rejects_bad_inputs():
    with pytest.raises(ValueError):
        h(0, 1, 0.1)
    with pytest.raises(ValueError):
        h(1, 0, 0.1)
    with pytest.raises(ValueError):
        h(1, 1, 0.3)  # variance above 1/4


def test_s_k_values():
    assert s_k(1) == pytest.approx(8 / PI ** 2)
    assert s_k(3) == pytest.approx(0.75)
    assert s_k(4) == pytest.approx(1 - 1 / 6)
    assert all(s_k(k) > 0.5 for k in (1, 3, 4, 20))
    with pytest.raises(ValueError):
        s_k(2)  # s_2 = 1/2 is not allowed


def test_beta_hand_values():
    assert beta(1, 1) == pytest.approx(1 - 8 / PI ** 2)
    assert beta(2, 3) == pytest.approx(1 - 0.75 ** 2)            # P(Bin(2, 1/4) >= 1)
    assert beta(3, 3) == pytest.approx(3 * 0.25 ** 2 * 0.75 + 0.25 ** 3)  # P(Bin(3, 1/4) >= 2)


@pytest.mark.parametrize("k", [1, 3, 8])
def test_beta_decreasing_over_odd_r(k):
    # Beta is not monotone in even r (an even run count never beats the odd one below it),
    # so the monotonicity check is over odd r.
    vals = [beta(r, k) for r in range(1, 40, 2)]
    assert all(b2 < b1 for b1, b2 in zip(vals, vals[1:]))


@pytest.mark.parametrize("k", [1, 3, 8])
@pytest.mark.parametrize("eta", [0.05, 1e-3, 1e-6])
def test_r_needed_is_minimal(k, eta):
    r = r_needed(k, eta)
    assert beta(r, k) <= eta
    assert all(beta(rp, k) > eta for rp in range(1, r))


def test_vbar_hand_cases():
    assert vbar(0.05, 0.01) == pytest.approx(0.06 * 0.94)
    assert vbar(0.4, 0.2) == pytest.approx(0.25)
    assert vbar(0.9, 0.05) == pytest.approx(0.85 * 0.15)
    assert vbar(0.0, 0.01) == pytest.approx(0.01 * 0.99)


def test_eta_round_sums_to_delta_over_d():
    d, delta = 7, 0.05
    total = sum(eta_round(t, delta, d) for t in range(200000))
    assert total == pytest.approx(delta / d, rel=1e-4)


def test_lemma2_consistent_interval_numerically():
    rng = np.random.default_rng(0)
    grid = np.linspace(0.0, 1.0, 4001)
    checked_side = 0
    for _ in range(300):
        p = rng.uniform(0.001, 0.5)
        M = int(rng.integers(1, 2000))
        k = int(rng.choice([1, 3, 4, 5]))
        hp = h(M, k, p * (1 - p))
        p_hat = p + rng.uniform(-1, 1) * hp
        q_grid = np.union1d(grid, [p])
        mask = np.abs(p_hat - q_grid) <= _h_vec(M, k, q_grid)
        idx = np.flatnonzero(mask)
        assert idx.size > 0
        assert np.all(np.diff(idx) == 1), "consistent set is not an interval"   # (i)
        assert mask[np.searchsorted(q_grid, p)], "p not in consistent set"      # (ii)
        gap = rng.uniform(0.001, 0.3)
        for alpha in (p - gap, p + gap):
            if not (0.0 < alpha < 1.0):
                continue
            g = abs(p - alpha)
            if 2 * h(M, k, vbar(p, g)) >= g:
                continue
            checked_side += 1
            other_side = (q_grid - alpha) * (p - alpha) < 0
            assert not np.any(mask & other_side), "interval crosses alpha although 2h < gap"  # (iii)
    assert checked_side >= 50


def test_q_cert_decision_away_from_band():
    for p in (0.01, 0.05, 0.08, 0.2, 0.3, 0.46):
        for k in (1, 3):
            res = q_cert_route(p, ALPHA, 1e-2, DELTA, 40, k)
            assert res["certified"] == (p <= ALPHA)
            assert res["rounds"] >= 1 and res["cost"] > 0


def test_q_cert_floor_at_zero_gap():
    res = q_cert_route(ALPHA, ALPHA, 1e-2, DELTA, 10, 1)
    assert res["floor"] is True
    assert res["certified"] is True


def test_q_cert_cost_grows_as_gap_shrinks():
    gaps = [0.3, 0.15, 0.075, 0.0375, 0.02]
    costs = [q_cert_route(ALPHA + g, ALPHA, 1e-4, DELTA, 50, 1)["cost"] for g in gaps]
    assert all(c2 >= c1 for c1, c2 in zip(costs, costs[1:]))


def test_q_cert_cost_scales_like_inverse_gap():
    c = {g: q_cert_route(ALPHA + g, ALPHA, 1e-4, DELTA, 50, 1)["cost"] for g in (0.04, 0.02, 0.01)}
    for big, small in ((0.04, 0.02), (0.02, 0.01)):
        ratio = c[small] / c[big]
        assert 2 / 2.5 <= ratio <= 2 * 2.5, ratio


def test_c_cert_inequalities_are_tight():
    for p, eps in ((0.3, 1e-4), (0.12, 1e-2), (0.07, 1e-3), (ALPHA, 1e-3)):
        d = 60
        res = c_cert_route(p, ALPHA, eps, DELTA, d)
        L = math.log(3 * d / DELTA)
        gap = abs(p - ALPHA)

        def side_ok(m):
            v = vbar(p, gap)
            return gap > 0 and 2 * (math.sqrt(2 * v * L / m) + 3 * L / m) < gap

        def floor_ok(m):
            # adaptive floor at variance vbar(p, eps/2), capped by the 0.25-variance floor
            v = vbar(p, eps / 2)
            adaptive = 2 * (math.sqrt(2 * v * L / m) + 3 * L / m) < eps / 2
            cap = math.sqrt(0.5 * L / m) + 3 * L / m <= eps / 2
            return adaptive or cap

        n = res["n"]
        if res["floor"]:
            assert floor_ok(n) and not floor_ok(n - 1)
        else:
            assert side_ok(n) and not side_ok(n - 1)


def test_q_cert_ideal_cost_is_last_round_only():
    res = q_cert_route(0.3, ALPHA, 1e-3, DELTA, 40, 1)
    t = res["rounds"] - 1
    assert res["ideal_cost"] == pytest.approx(r_needed(1, eta_round(t, DELTA, 40)) * (2 * 2 ** t + 1))
    assert res["ideal_cost"] <= res["cost"]


def test_q_cert_adaptive_floor_stops_no_later_than_fixed_floor():
    # the adaptive floor uses vbar(p, eps/2) <= 1/4, so it can only stop earlier than the fixed floor
    for p in (0.1, 0.11, 0.15):
        res = q_cert_route(p, ALPHA, 1e-3, DELTA, 40, 1)
        fixed_rounds = 0
        while h(2 ** fixed_rounds, 1, VAR_MAX) > 1e-3 / 2:
            fixed_rounds += 1
        assert res["rounds"] <= fixed_rounds + 1


def test_c_cert_scales_like_inverse_gap_squared():
    n = {g: c_cert_route(ALPHA + g, ALPHA, 1e-4, DELTA, 50)["n"] for g in (0.04, 0.02)}
    ratio = n[0.02] / n[0.04]
    assert 4 / 2.5 <= ratio <= 4 * 2.5, ratio


def test_c_cert_zero_gap_uses_floor():
    res = c_cert_route(ALPHA, ALPHA, 1e-2, DELTA, 10)
    assert res["floor"] is True and res["certified"] is True


def test_best_k_returns_minimiser():
    ps = [0.02, 0.09, 0.3, 0.45]
    k, total = best_k(ps, ALPHA, 1e-2, DELTA, 4, ks=(1, 3, 4))
    totals = {kk: sum(q_cert_route(p, ALPHA, 1e-2, DELTA, 4, kk)["cost"] for p in ps)
              for kk in (1, 3, 4)}
    assert k == min(totals, key=totals.get)
    assert total == pytest.approx(totals[k])


def _load_script():
    spec = importlib.util.spec_from_file_location("run_quasar_cert", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_summary_script_on_tiny_pair(tmp_path):
    risks = {
        "VNN-A-1": {"Det": [0.05, 0.2], "SAA": [0.08]},
        "GD-B-1": {"Det": [0.3], "SAA": [0.1, 0.02]},
    }
    lengths = {"VNN-A-1": {"Det": [3, 2], "SAA": [4]},
               "GD-B-1": {"Det": [5], "SAA": [2, 6]}}
    plans = tmp_path / "plans"
    plans.mkdir()
    for inst in risks:
        res = {g: {"plan": [list(range(n)) for n in lengths[inst][g]]} for g in ("Det", "SAA")}
        (plans / f"{inst}.json").write_text(json.dumps({"name": inst, "res": res}))
    risks_path = tmp_path / "risks.json"
    risks_path.write_text(json.dumps(risks))
    out = tmp_path / "out"

    mod = _load_script()
    mod.main(["--risks", str(risks_path), "--plans", str(plans), "--out-dir", str(out)])

    with open(out / "cert_region_routes.csv") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 6 * len(mod.EPS)   # 6 routes (2+1 VNN, 1+2 GD), one row per eps
    assert {r["region"] for r in rows} == {"VNN", "GD"}
    assert {r["gate"] for r in rows} == {"Det", "SAA"}
    first = next(r for r in rows if r["instance"] == "VNN-A-1" and r["gate"] == "Det" and r["index"] == "0")
    assert first["L"] == "3" and float(first["p"]) == pytest.approx(0.05)
    assert first["certified"] == "True"

    md = (out / "cert_region_summary.md").read_text()
    assert "| VNN | Det |" in md and "| GD | SAA |" in md
    assert md.count("| VNN | Det |") == len(mod.EPS)
    assert "Q_ideal" in md
    assert all(r["q_ideal"] for r in rows)


def test_summary_script_rejects_length_mismatch(tmp_path):
    plans = tmp_path / "plans"
    plans.mkdir()
    (plans / "VNN-X-1.json").write_text(json.dumps({"res": {"Det": {"plan": [[1, 2]]}, "SAA": {"plan": [[1]]}}}))
    risks_path = tmp_path / "risks.json"
    risks_path.write_text(json.dumps({"VNN-X-1": {"Det": [0.1, 0.2], "SAA": [0.1]}}))
    mod = _load_script()
    with pytest.raises(ValueError):
        mod.main(["--risks", str(risks_path), "--plans", str(plans), "--out-dir", str(tmp_path / "o")])
