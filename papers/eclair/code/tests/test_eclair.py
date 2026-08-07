"""Fast self-contained tests for the ECLAIR implementation.
Run from papers/eclair/code:  ../../../.venv/bin/python -m pytest tests -q
(deliberately NOT under svrpspd_wdro/tests — see ../README.md on
self-containment)."""

import math
import random
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from eclair.calibrate import cp_upper, _binom_cdf
from eclair.certify import Bets, e_bh, run_certification
from eclair.mutations import make_faithful, make_mutant
from eclair.probes import brute_optimum, solve_value, tier_a, tier_b
from eclair.problems import ALL_SPECS


def test_faithful_matches_brute_force():
    rng = random.Random(11)
    for spec in ALL_SPECS:
        cand = make_faithful(spec)
        for _ in range(5):
            p = spec.gen(rng, "micro")
            assert solve_value(cand.build(p)) == brute_optimum(spec, p), \
                spec.name


def test_mrs_never_alarm_on_faithful():
    rng = random.Random(12)
    for spec in ALL_SPECS:
        cand = make_faithful(spec)
        for _ in range(12):
            alarm, _ = tier_b(cand, rng)
            assert not alarm, f"MR false alarm on faithful {spec.name}"


def test_mutants_are_semantically_different():
    rng = random.Random(13)
    for spec in ALL_SPECS:
        mut = make_mutant(spec, rng)
        assert mut is not None and not mut.faithful
        diff = any(
            solve_value(mut.build(p)) != brute_optimum(spec, p)
            for p in (spec.gen(rng, "micro") for _ in range(12)))
        assert diff, spec.name


def test_conservative_bet_is_supermartingale():
    """E_H0[e] <= 1 whenever p0_true <= p0_bar < p1_bet."""
    p0_bar, p1_bet = 0.05, 0.4
    e_alarm = p1_bet / p0_bar
    e_pass = (1 - p1_bet) / (1 - p0_bar)
    for p0_true in (0.0, 0.01, 0.03, 0.05):
        mean = p0_true * e_alarm + (1 - p0_true) * e_pass
        assert mean <= 1.0 + 1e-12


def test_cp_upper_bound():
    # zero alarms in n trials: rule-of-three-ish upper bound
    assert 0.0 < cp_upper(0, 100) < 0.05
    assert cp_upper(0, 100) > 0 / 100
    # bound must be conservative: CDF at the bound equals ~2.5%
    n, k = 200, 3
    ub = cp_upper(k, n)
    assert abs(_binom_cdf(k, n, ub) - 0.025) < 1e-3
    assert ub > k / n


def test_e_bh():
    # K=4, alpha=0.05: reject k with e_(k) >= K/(alpha k) = 80/k
    assert e_bh([100, 100, 1, 1], 0.05) == {0, 1}      # 100 >= 40
    assert e_bh([100, 30, 1, 1], 0.05) == {0}          # 30 < 40
    assert e_bh([1, 1, 1, 1], 0.05) == set()


def _toy_bets():
    return Bets({
        "A": {"p0_bar": 0.02, "p1_bet": 0.4, "mean_cost": 0.001},
        "B": {"delta": 0.05, "mean_cost": 0.001},
        "C": {"p0_bar": 0.15, "p1_bet": 0.45, "mean_cost": 0.001},
    })


def test_certification_separates():
    rng = random.Random(14)
    spec = ALL_SPECS[0]
    bets = _toy_bets()
    pool = [make_faithful(spec), make_faithful(spec)]
    labels = [True, True]
    for _ in range(2):
        m = make_mutant(spec, rng)
        assert m is not None
        pool.append(m)
        labels.append(False)
    res = run_certification(pool, bets, "kelly", budget=1.5, alpha=0.05,
                            rng=rng)
    rejected = [st.rejected for st in res["states"]]
    # no faithful candidate rejected; at least one mutant caught
    assert not rejected[0] and not rejected[1]
    assert any(rejected[2:])
    if res["pick"] is not None:
        assert labels[res["states"].index(res["pick"])]
