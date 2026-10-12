"""Tests for quantum/ae_sim.py (QUASAR M3b v2): QAE outcome distribution, MLAE
convergence, classical RMS law and the Grover rotation validation. Fast."""

from __future__ import annotations

import math

import numpy as np
import pytest

from quantum.ae_sim import (
    grover_validation,
    mlae_cost,
    qae_cost,
    qae_outcome_pmf,
    rms_vs_cost,
    theta_of,
)
from quantum.classical import DiscreteDemand
from quantum.oracle import build_peak_oracle


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
