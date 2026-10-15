"""Tests for the RARECERT R1 task B code: the lognormal marginal family, the pickup
scaling of the end-mode ratio varrho, and the coverage routine (checked on a crude-MC
toy whose truth is known). Runs in well under three minutes."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from quantum.rare import RouteSim  # noqa: E402
from rarecert.family import FamilyRouteSim  # noqa: E402
from rarecert.smis import stage_scales  # noqa: E402
from run_rarecert_sensitivity import coverage_rate  # noqa: E402


def test_lognormal_family_mean_and_cv():
    dbar = np.array([2.0, 1.0, 3.0])
    pbar = np.array([1.5, 0.5, 2.5])
    sim = FamilyRouteSim(dbar, pbar, cv=0.3, rho=0.6, family="lognormal")
    rng = np.random.default_rng(1)
    x = rng.standard_normal((400_000, sim.dim))
    d, p = sim.demands(x)
    for j in range(3):
        m, sd = d[:, j].mean(), d[:, j].std()
        assert abs(m / dbar[j] - 1.0) < 0.01
        assert abs(sd / m - 0.3) < 0.01
        mp_, sp = p[:, j].mean(), p[:, j].std()
        assert abs(mp_ / pbar[j] - 1.0) < 0.01
        assert abs(sp / mp_ - 0.3) < 0.01


def test_gamma_family_is_routesim():
    dbar = np.array([2.0, 1.0, 3.0])
    pbar = np.array([1.5, 0.5, 2.5])
    x = np.random.default_rng(2).standard_normal((1000, 8))
    a = FamilyRouteSim(dbar, pbar, cv=0.3, rho=0.6, family="gamma").peak(x)
    b = RouteSim(dbar, pbar, cv=0.3, rho=0.6).peak(x)
    assert np.array_equal(a, b)


def test_pickup_scaling_moves_varrho_in_expected_direction():
    dbar = np.full(6, 3.0)
    pbar = np.full(6, 0.6)           # pickups are the smaller end: varrho = lam_p[L] / lam_d[0] < 1
    r = {}
    for s in (0.5, 1.0, 2.0):
        sim = RouteSim(dbar, pbar * s, cv=0.3, rho=0.6)
        r[s] = stage_scales(sim)["ratio"]
    assert 0.0 < r[0.5] < r[1.0] < r[2.0] < 1.0
    # block scales are homogeneous of degree one in the thetas, so the ratio is linear in s
    assert math.isclose(r[2.0], 2.0 * r[1.0], rel_tol=1e-9)
    assert math.isclose(r[0.5], 0.5 * r[1.0], rel_tol=1e-9)


def test_coverage_routine_on_crude_mc_toy():
    c = 1.5
    truth = 0.5 * math.erfc(c / math.sqrt(2.0))   # P(Z > 1.5)
    N = 2_000

    def toy(rng):
        x = rng.standard_normal(N)
        p = float(np.mean(x > c))
        rse = math.sqrt((1.0 - p) / (p * N)) if p > 0 else math.inf
        return p, rse

    cov, med_rse, zeros = coverage_rate(toy, truth, 800, np.random.default_rng(7))
    assert zeros == 0
    assert 0.92 <= cov <= 0.98, cov
    assert med_rse > 0
