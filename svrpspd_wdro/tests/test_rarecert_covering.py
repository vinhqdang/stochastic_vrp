"""Covering constant of the stage-mixture theorem (rarecert.covering.psi)."""
import math

from rarecert.covering import psi


def test_dominant_block_dominates_gives_four():
    # Lemma (b): varrho, nu <= 1/3 -> psi_H = psi = 4
    for vr, nu in ((0.0, 0.1), (0.2, 0.2), (1 / 3, 1 / 3)):
        assert abs(psi(vr, nu, use_lo=False, n=301) - 4.0) < 1e-6
        assert abs(psi(vr, nu, n=301) - 4.0) < 1e-6


def test_tie_gives_two_sqrt_two():
    # Lemma (c): equality at varrho = 1 for nu <= 1/2
    for nu in (0.05, 0.3, 0.5):
        assert abs(psi(1.0, nu, n=301) - 2 * math.sqrt(2)) < 1e-6


def test_bounds():
    for vr in (0.0, 0.3, 0.6, 0.9, 1.0):
        for nu in (0.1, 0.5, 1.0):
            v = psi(vr, nu, n=201)
            assert 2.0 - 1e-9 <= v <= 4.0 + 1e-9
            if nu <= 0.5:
                assert v >= 2 * math.sqrt(2) - 1e-6


def test_penalty_lowers_and_is_monotone():
    a = psi(0.5, 0.2, n=201)
    b = psi(0.5, 0.2, n=201, pen_h=0.5, pen_lo=0.5)
    assert b <= a + 1e-12
