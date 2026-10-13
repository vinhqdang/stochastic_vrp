"""Simultaneous risk certification of routes (QUASAR cert; reference:
papers/quasar/THEORY_CERT.md).

Route j of a plan has risk p_j; a certification must decide, simultaneously
for all d routes, whether p_j <= alpha (certify) with soundness at alpha + eps/2
and completeness at alpha - eps. This module holds the explicit-constant
quantum schedule Q-CERT (canonical amplitude estimation, median of r runs,
consistent interval of Lemma 2) and the classical counterpart C-CERT
(Bernstein sample size), both as deterministic worst-case schedules that
assume every estimate is good.

Only NumPy and SciPy are used here; no Qiskit.
"""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Sequence

import numpy as np
from scipy import stats

PI2 = math.pi ** 2
VAR_MAX = 0.25                       # largest variance p(1-p)
KS_DEFAULT = (1,) + tuple(range(3, 21))   # k = 2 is excluded: s_2 = 1/2 is not > 1/2


def _real(name: str, x: float, lo: float = 0.0, open_lo: bool = False, hi: float = math.inf) -> float:
    """Validate a finite real x with lo <= x <= hi (strict lower bound if open_lo)."""
    x = float(x)
    if not math.isfinite(x) or x > hi or x < lo or (open_lo and x == lo):
        raise ValueError(f"{name} out of range, got {x}")
    return x


def _int(name: str, x: int, lo: int) -> int:
    """Validate an integer x >= lo."""
    if isinstance(x, bool) or int(x) != x or int(x) < lo:
        raise ValueError(f"{name} must be an integer >= {lo}, got {x!r}")
    return int(x)


def _unit(name: str, x: float) -> float:
    """Validate x in (0, 1)."""
    x = _real(name, x, open_lo=True, hi=1.0)
    if x == 1.0:
        raise ValueError(f"{name} must lie in (0, 1), got {x}")
    return x


def h(M: int, k: int, v: float) -> float:
    """Amplitude-estimation error bound h_{M,k} as a function of the variance v = p(1-p)."""
    M = _int("M", M, 1)
    k = _int("k", k, 1)
    v = _real("v", v, hi=VAR_MAX)
    return 2 * math.pi * k * math.sqrt(v) / M + PI2 * k ** 2 / M ** 2


def s_k(k: int) -> float:
    """Success probability of Fact 1 (canonical AE): 8/pi^2 for k = 1, 1 - 1/(2(k-1)) otherwise."""
    k = _int("k", k, 1)
    if k == 1:
        return 8.0 / PI2
    if k == 2:
        raise ValueError("k = 2 gives s_k = 1/2, which the median argument does not allow")
    return 1.0 - 1.0 / (2.0 * (k - 1))


def beta(r: int, k: int) -> float:
    """Median failure probability P(Bin(r, 1 - s_k) >= ceil(r/2)) of r runs (Lemma 1)."""
    r = _int("r", r, 1)
    return float(stats.binom.sf(math.ceil(r / 2) - 1, r, 1.0 - s_k(k)))


@lru_cache(maxsize=None)
def _r_needed(k: int, eta: float) -> int:
    r = 1
    while beta(r, k) > eta:
        r += 1
    return r


def r_needed(k: int, eta: float) -> int:
    """Smallest r >= 1 with beta(r, k) <= eta (linear scan: beta is not monotone in even r)."""
    k = _int("k", k, 1)
    eta = _unit("eta", eta)
    s_k(k)  # validates k
    return _r_needed(k, eta)


def vbar(p: float, g: float) -> float:
    """Largest variance q(1-q) over q with |q - p| <= g, q in [0, 1]."""
    p = _real("p", p, hi=1.0)
    g = _real("g", g)
    lo, hi = max(0.0, p - g), min(1.0, p + g)
    if lo <= 0.5 <= hi:
        return 0.25
    q = hi if hi < 0.5 else lo
    return q * (1.0 - q)


def eta_round(t: int, delta: float, d: int) -> float:
    """Failure probability of round t: 6 delta / (pi^2 d (t+1)^2); sums to delta over t, d routes."""
    t = _int("t", t, 0)
    delta = _unit("delta", delta)
    d = _int("d", d, 1)
    return 6.0 * delta / (PI2 * d * (t + 1) ** 2)


def _check_route(p: float, alpha: float, eps: float, delta: float, d: int) -> None:
    _real("p", p, hi=1.0)
    _unit("alpha", alpha)
    if not 0.0 < eps < alpha:
        raise ValueError(f"eps must satisfy 0 < eps < alpha, got eps={eps}, alpha={alpha}")
    _unit("delta", delta)
    _int("d", d, 1)


def q_cert_route(p: float, alpha: float, eps: float, delta: float, d: int, k: int,
                 t_max: int = 60) -> dict:
    """Deterministic worst-case schedule of Q-CERT for one route (all estimates good).

    Stops at the first round t where one of three conditions holds: the gap
    condition 2 h(vbar(p, gap)) < gap (gap = |p - alpha| > 0); the adaptive floor
    2 h(vbar(p, eps/2)) < eps/2; or the fixed floor h(1/4) <= eps/2. The cost is
    the number of applications of A, A^dagger: sum_{s<=t} r_s (2 M_s + 1).
    ``floor`` is True when the route stops by one of the two floor conditions and
    the gap condition does not hold in the same round. ``ideal_cost`` is the cost of
    the stopping round alone, r_t (2 M_t + 1): no doubling and no earlier rounds.
    """
    _check_route(p, alpha, eps, delta, d)
    _int("t_max", t_max, 0)
    gap = abs(p - alpha)
    half = eps / 2
    certified = bool(p <= alpha)
    cost = 0.0
    for t in range(t_max + 1):
        M = 2 ** t
        round_cost = r_needed(k, eta_round(t, delta, d)) * (2 * M + 1)
        cost += round_cost
        out = {"rounds": t + 1, "M_last": M, "cost": cost, "ideal_cost": float(round_cost),
               "certified": certified}
        if gap > 0 and 2 * h(M, k, vbar(p, gap)) < gap:
            return {**out, "floor": False}
        if 2 * h(M, k, vbar(p, half)) < half or h(M, k, VAR_MAX) <= half:
            return {**out, "floor": True}
    raise RuntimeError(f"Q-CERT did not stop within t_max={t_max} rounds (p={p}, eps={eps})")


def _n_bern(L: float, v: float, T: float) -> int:
    """Smallest n >= 1 with 2 (sqrt(2 v L / n) + 3 L / n) < T (empirical-Bernstein form)."""
    def ok(n: int) -> bool:
        return 2.0 * (math.sqrt(2.0 * v * L / n) + 3.0 * L / n) < T

    a, b = 6.0 * L, 2.0 * math.sqrt(2.0 * v * L)
    x = (-b + math.sqrt(b * b + 4.0 * a * T)) / (2.0 * a)  # strict bound: n > 1/x^2
    n = max(1, math.floor(1.0 / (x * x)) + 1)
    while not ok(n):
        n += 1
    while n > 1 and ok(n - 1):
        n -= 1
    return n


def _n_cap(L: float, eps: float) -> int:
    """Smallest n with sqrt(2 * 0.25 * L / n) + 3 L / n <= eps/2 (fixed 0.25-variance cap)."""
    def ok(n: int) -> bool:
        return math.sqrt(0.5 * L / n) + 3.0 * L / n <= eps / 2

    a, b, c = 3.0 * L, math.sqrt(0.5 * L), eps / 2
    x = (-b + math.sqrt(b * b + 4.0 * a * c)) / (2.0 * a)   # x = 1/sqrt(n) at the root
    n = max(1, math.ceil(1.0 / (x * x)))
    while not ok(n):
        n += 1
    while n > 1 and ok(n - 1):
        n -= 1
    return n


def c_cert_route(p: float, alpha: float, eps: float, delta: float, d: int) -> dict:
    """Classical per-route sample count with no doubling loss (favours the classical side).

    Uses the empirical-Bernstein bound of Audibert, Munos and Szepesvari (2009) with
    eta = delta/d and L = ln(3/eta). n_side is the smallest n whose half-width puts the
    interval strictly on one side of alpha (variance vbar(p, gap)). n_floor is the
    smallest n with the adaptive half-width condition at variance vbar(p, eps/2), capped
    by the fixed 0.25-variance floor. n = min(n_side, n_floor); ``floor`` is True when
    n_floor <= n_side.
    """
    _check_route(p, alpha, eps, delta, d)
    L = math.log(3.0 * d / delta)
    gap = abs(p - alpha)
    n_floor = min(_n_bern(L, vbar(p, eps / 2), eps / 2), _n_cap(L, eps))
    n_side = _n_bern(L, vbar(p, gap), gap) if gap > 0 else math.inf
    return {"n": min(n_floor, n_side), "floor": n_floor <= n_side, "certified": bool(p <= alpha)}


def best_k(ps: Sequence[float], alpha: float, eps: float, delta: float, d: int,
           ks: Sequence[int] = KS_DEFAULT, weights: Sequence[float] | None = None) -> tuple[int, float]:
    """Fixed k in ks minimising sum_j weights_j * Q-CERT cost of route j; returns (k, total).

    The choice uses the true risks, so it is an oracle choice favouring the quantum side.
    Default weights are 1 per route.
    """
    arr = np.asarray(ps, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"ps must be 1-D, got shape {arr.shape}")
    w = np.ones(arr.size) if weights is None else np.asarray(weights, dtype=np.float64)
    if w.shape != arr.shape or not np.all(np.isfinite(w)) or np.any(w < 0):
        raise ValueError("weights must be finite, non-negative and match ps")
    best = None
    for k in ks:
        total = sum(wj * q_cert_route(float(p), alpha, eps, delta, d, k)["cost"]
                    for p, wj in zip(arr, w))
        if best is None or total < best[1]:
            best = (int(k), float(total))
    if best is None:
        raise ValueError("ks must be non-empty")
    return best
