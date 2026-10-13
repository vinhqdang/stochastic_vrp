"""Tests for quantum/search.py (QUASAR M4b, cost model of FIRST-FEASIBLE).

Covers the first-index helpers, the ideal TEST schedule of FIRST-FEASIBLE
(exhaustive for M <= 70), the median and prefix-test costs (Lemma 2 condition),
the Bernstein sample bound, and the core cost formulas on hand-computed cases.
Fast."""

from __future__ import annotations

import math

import numpy as np
import pytest

from quantum.experiments import canonical_ae_M
from quantum.search import (
    P_AE,
    bernstein_n,
    classical_core_cost,
    classical_scan_cost,
    crn_classical_core_cost,
    crn_quantum_core_cost,
    first_index,
    ideal_schedule,
    kstar,
    median_reps,
    prefix_test_cost,
    quantum_core_cost,
    quantum_search_cost,
    stop_index,
    threshold_test_cost,
)


def _direct_M(var: float, eps: float) -> int:
    """Independent computation of the smallest power of two meeting the AE bound."""
    s = math.sqrt(max(var, 0.0))
    M = 1
    while 2 * math.pi * s / M + math.pi ** 2 / M ** 2 > eps:
        M *= 2
    return M


# ---------------------------------------------------------------- first_index


def test_first_index_basic_and_none():
    p = np.array([0.9, 0.6, 0.4, 0.5, 0.1])
    assert first_index(p, 0.5) == 3          # p[2] = 0.4 is the first <= 0.5
    assert first_index(p, 0.95) == 1         # first entry qualifies
    assert first_index(p, 0.05) == 0         # none qualifies -> 0
    assert first_index(p, 0.1) == 5          # equality counts
    assert first_index([], 0.5) == 0         # empty array -> 0


def test_first_index_rejects_bad_input():
    with pytest.raises(ValueError):
        first_index(np.zeros((2, 2)), 0.5)
    with pytest.raises(ValueError):
        first_index([0.1, float("nan")], 0.5)
    with pytest.raises(ValueError):
        first_index([0.1], float("inf"))


def test_kstar_and_stop_index_on_hand_array():
    p = np.array([0.9, 0.7, 0.45, 0.35, 0.2])
    alpha, eps = 0.4, 0.1
    assert stop_index(p, alpha) == 4        # first p <= 0.4 is 0.35 at index 4
    assert kstar(p, alpha, eps) == 5        # first p <= 0.3 is 0.2 at index 5
    assert kstar(p, alpha, eps) > stop_index(p, alpha)   # band case


def test_kstar_none_and_eps_zero():
    p = np.array([0.9, 0.8, 0.7])
    assert kstar(p, 0.5, 0.1) == 0          # nothing certified -> 0
    assert stop_index(p, 0.5) == 0          # nothing at the alpha level -> 0
    q = np.array([0.9, 0.3, 0.1])
    assert kstar(q, 0.4, 0.0) == stop_index(q, 0.4) == 2   # no band when eps = 0


def test_alpha_and_eps_validation():
    with pytest.raises(ValueError):
        stop_index([0.1], 0.0)
    with pytest.raises(ValueError):
        stop_index([0.1], 1.0)
    with pytest.raises(ValueError):
        kstar([0.1], 0.5, -0.01)


# ------------------------------------------------------------- ideal_schedule


def _replay_hi(tests: list[int], s: int, M: int) -> int:
    """Replay the YES rule on a schedule and return the final hi (0 if no YES)."""
    hi = 0
    lo = 0
    seen_yes = False
    for m in tests:
        if not seen_yes:
            if s >= 1 and m >= s:
                seen_yes = True
                hi = m
            else:
                lo = m
        else:
            if m >= s:
                hi = m
            else:
                lo = m
    return hi


@pytest.mark.parametrize("M", list(range(1, 71)))
def test_ideal_schedule_final_hi_equals_stop_index(M):
    for s in range(0, M + 1):
        tests = ideal_schedule(s, M)
        if s >= 1:
            assert _replay_hi(tests, s, M) == s
        assert all(1 <= m <= M for m in tests)
        assert len(tests) <= 2 * math.ceil(math.log2(M)) + 2


@pytest.mark.parametrize("M", list(range(1, 71)))
def test_ideal_schedule_exponential_phase(M):
    expo = []
    m = 1
    while m < M:
        expo.append(m)
        m *= 2
    expo.append(M)
    # s = 0: no YES, exactly the exponential list
    assert ideal_schedule(0, M) == expo
    # every s >= 1 starts with the exponential prefix up to the first YES
    for s in range(1, M + 1):
        tests = ideal_schedule(s, M)
        first_yes = next(i for i, m in enumerate(expo) if m >= s)
        assert tests[: first_yes + 1] == expo[: first_yes + 1]


def test_ideal_schedule_hand_cases():
    assert ideal_schedule(0, 5) == [1, 2, 4, 5]
    assert ideal_schedule(1, 1) == [1]
    assert ideal_schedule(3, 4) == [1, 2, 4, 3]     # YES at 4, bisect (2,4)
    assert ideal_schedule(5, 8) == [1, 2, 4, 8, 6, 5]
    assert ideal_schedule(8, 8) == [1, 2, 4, 8, 6, 7]
    assert ideal_schedule(2, 8) == [1, 2]           # YES at 2, nothing to bisect


def test_ideal_schedule_sizes_never_exceed_M_and_no_duplicate_last():
    for M in (1, 2, 4, 8, 16, 33, 64):
        for s in range(0, M + 1):
            tests = ideal_schedule(s, M)
            assert max(tests) <= M
    assert ideal_schedule(0, 8) == [1, 2, 4, 8]     # power of two: no duplicate M


def test_ideal_schedule_rejects_bad_input():
    with pytest.raises(ValueError):
        ideal_schedule(6, 5)
    with pytest.raises(ValueError):
        ideal_schedule(-1, 5)
    with pytest.raises(ValueError):
        ideal_schedule(1, 0)


# ------------------------------------------------------------- median / threshold


def test_median_reps_monotone_and_at_least_one():
    etas = [0.9, 0.5, 0.1, 1e-2, 1e-4, 1e-8, 1e-12]
    reps = [median_reps(e) for e in etas]
    assert all(r >= 1 for r in reps)
    assert all(a <= b for a, b in zip(reps, reps[1:]))   # decreasing in eta
    assert median_reps(0.999999) == 1


def test_median_reps_formula():
    eta = 0.01
    expected = math.ceil(math.log(1 / eta) / (2 * (P_AE - 0.5) ** 2))
    assert median_reps(eta) == expected


def test_threshold_test_cost_uses_canonical_ae():
    for v, eps, eta in [(0.0, 0.5, 0.1), (0.04, 0.05, 0.01), (1.0, 0.1, 1e-3)]:
        M = _direct_M(v, eps)
        assert canonical_ae_M(v, eps) == M
        expected = median_reps(eta) * (2 * M + 1)
        assert threshold_test_cost(v, eps, eta) == pytest.approx(expected, rel=0, abs=0)


# ------------------------------------------------------------- prefix test


@pytest.mark.parametrize("m", [1, 2, 3, 7, 16, 50])
@pytest.mark.parametrize("delta_p", [0.5, 0.05, 1e-3])
def test_prefix_test_cost_lemma2_condition(m, delta_p):
    out = prefix_test_cost(m, 0.04, 0.05, delta_p)
    K, R, eta = out["K"], out["R"], out["eta"]
    assert K == math.ceil(math.sqrt(2 * m))
    assert R * (1 + (2 * K - 1) ** 2) * eta == pytest.approx(delta_p, abs=1e-12)
    assert (1 - 0.25) ** R <= delta_p + 1e-15
    assert out["reps"] == median_reps(eta)
    assert out["M_ae"] == _direct_M(0.04, 0.05)
    expected = R * 2 * K * threshold_test_cost(0.04, 0.05, eta)
    assert out["cost"] == pytest.approx(expected, rel=1e-12)


def test_prefix_test_cost_rejects_bad_input():
    with pytest.raises(ValueError):
        prefix_test_cost(0, 0.04, 0.05, 0.1)
    with pytest.raises(ValueError):
        prefix_test_cost(3, 0.04, 0.05, 1.0)


def test_quantum_search_cost_sums_prefix_costs():
    M, v, eps, delta = 10, 0.04, 0.05, 0.1
    delta_p = delta / (4 * math.ceil(math.log2(M)) + 4)
    sizes = ideal_schedule(6, M)
    expected = sum(prefix_test_cost(m, v, eps, delta_p)["cost"] for m in sizes)
    assert quantum_search_cost(sizes, v, eps, delta, M) == pytest.approx(expected, rel=1e-12)
    assert quantum_search_cost([], v, eps, delta, M) == 0.0


# ------------------------------------------------------------- Bernstein


@pytest.mark.parametrize(
    "v, eps, delta",
    [(0.0, 0.1, 0.05), (0.04, 0.05, 0.01), (0.5, 0.02, 1e-3), (2.0, 0.3, 0.2)],
)
def test_bernstein_n_is_minimal(v, eps, delta):
    n = bernstein_n(v, eps, delta)

    def bound(k: int) -> float:
        return 2 * math.exp(-k * eps ** 2 / (2 * v + 2 * eps / 3))

    assert bound(n) <= delta
    if n > 1:
        assert bound(n - 1) > delta


def test_classical_scan_cost_counts_tests():
    v, eps, delta, M = 0.04, 0.05, 0.1, 12
    n_s = bernstein_n(v, eps, delta / 4)
    assert classical_scan_cost(4, v, eps, delta, M) == 4 * n_s
    n_all = bernstein_n(v, eps, delta / M)
    assert classical_scan_cost(0, v, eps, delta, M) == M * n_all
    with pytest.raises(ValueError):
        classical_scan_cost(13, v, eps, delta, M)


# ------------------------------------------------------------- core costs


def test_quantum_core_cost_hand_case_and_sqrt_scaling():
    # v = 0, eps = 1: pi^2 / M^2 <= 1 first at M = 4, cost 2*4 + 1 = 9
    assert quantum_core_cost(1, 0.0, 1.0) == pytest.approx(9.0)
    # k = 4 gives sqrt(4) * 9 = 18
    assert quantum_core_cost(4, 0.0, 1.0) == pytest.approx(18.0)
    base = quantum_core_cost(3, 0.04, 0.05)
    assert quantum_core_cost(12, 0.04, 0.05) == pytest.approx(2 * base)
    with pytest.raises(ValueError):
        quantum_core_cost(0, 0.0, 1.0)


def test_classical_core_cost_hand_cases():
    # max(0.01/0.01, 3/0.1) = 30, times k = 10
    assert classical_core_cost(10, 0.01, 0.1) == pytest.approx(300.0)
    # max(1/0.01, 30) = 100, times k = 2
    assert classical_core_cost(2, 1.0, 0.1) == pytest.approx(200.0)


def test_crn_quantum_core_cost_hand_case():
    # reference: v_ref = 0.0625 at eps/2 = 0.25 -> M = 16 -> 33
    # diff: vD = 0.0625 at eps/4 = 0.125 -> M = 32 -> 65, two signed estimates
    # k = 1: 33 + 1 * 2 * 65 = 163
    assert canonical_ae_M(0.25, 0.25) == 16
    assert canonical_ae_M(0.0625, 0.125) == 32
    assert crn_quantum_core_cost(1, 0.0625, 0.25, 0.5) == pytest.approx(163.0)


def test_crn_classical_core_cost_hand_cases():
    # eps = 0.1: ref max(4*0.01/0.01, 60) = 60; diff max(4*0.04/0.01, 60) = 60, k = 2
    assert crn_classical_core_cost(2, 0.04, 0.01, 0.1) == pytest.approx(180.0)
    # ref max(4*1/0.01, 60) = 400; diff max(4*0.5/0.01, 60) = 200, k = 3 -> 1000
    assert crn_classical_core_cost(3, 0.5, 1.0, 0.1) == pytest.approx(1000.0)
