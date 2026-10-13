"""Cost model of FIRST-FEASIBLE for QUASAR M4b (reference: papers/quasar/THEORY_M4.md).

Schedule of TEST calls under an ideal test (exponential phase, then bisection),
query costs of the quantum prefix test and the classical scan, and the core
costs of the quantum and classical estimators, including CRN variants.
AE costs reuse :func:`quantum.experiments.canonical_ae_M`.
"""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np

from quantum.experiments import canonical_ae_M

PI2 = math.pi ** 2
P_AE = 8.0 / PI2     # success probability of canonical amplitude estimation
P_RUN = 0.25         # success probability per capped run (two attempts)


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


def first_index(p, thr) -> int:
    """1-based index of the first p[j] <= thr in the 1-D array p (0 if none)."""
    arr = np.asarray(p, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"p must be 1-D, got shape {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError("p must contain only finite values")
    hits = np.flatnonzero(arr <= _real("thr", thr, lo=-math.inf))
    return int(hits[0]) + 1 if hits.size else 0


def kstar(p, alpha: float, eps: float) -> int:
    """First index with p <= alpha - eps (0 = none): the certified-feasible index."""
    alpha = _unit("alpha", alpha)
    return first_index(p, alpha - _real("eps", eps))


def stop_index(p, alpha: float) -> int:
    """First index with p <= alpha (0 = none): where the ideal test resolves the band."""
    return first_index(p, _unit("alpha", alpha))


def ideal_schedule(s: int, M: int) -> list[int]:
    """Sizes of every TEST call of FIRST-FEASIBLE, ideal test YES iff s >= 1 and m >= s.

    Exponential phase 1, 2, 4, ... < M, then M; then bisection between the last
    NO and the first YES. With s >= 1 the final hi equals s.
    """
    M = _int("M", M, 1)
    s = _int("s", s, 0)
    if s > M:
        raise ValueError(f"s must be in [0, M={M}], got {s}")
    seq = []
    m = 1
    while m < M:
        seq.append(m)
        m *= 2
    seq.append(M)

    tests: list[int] = []
    lo, hi = 0, None
    for m in seq:
        tests.append(m)
        if s >= 1 and m >= s:
            hi = m
            break
        lo = m
    if hi is None:
        return tests
    while hi > lo + 1:
        mid = (lo + hi) // 2
        tests.append(mid)
        if mid >= s:
            hi = mid
        else:
            lo = mid
    return tests


def median_reps(eta: float) -> int:
    """Runs of canonical AE whose median fails with probability at most eta (Hoeffding)."""
    eta = _unit("eta", eta)
    return max(1, math.ceil(math.log(1.0 / eta) / (2.0 * (P_AE - 0.5) ** 2)))


def threshold_test_cost(v: float, eps: float, eta: float) -> float:
    """Applications of A per call of the threshold test B_j (median of AE runs)."""
    return float(median_reps(eta) * (2 * canonical_ae_M(v, eps) + 1))


def prefix_test_cost(m: int, v: float, eps: float, delta_p: float) -> dict:
    """Cost of one prefix test of size m at failure probability delta_p.

    K = ceil(sqrt(2m)); each run is two attempts (k = 0, then k uniform in
    {0..K-1}), success at least P_RUN per run; R = capped runs; eta =
    delta_p / (R (1 + (2K-1)^2)) (Lemma 2 condition). Each run uses at most
    2K threshold tests.
    """
    m = _int("m", m, 1)
    delta_p = _unit("delta_p", delta_p)
    K = math.ceil(math.sqrt(2 * m))
    R = math.ceil(math.log(1.0 / delta_p) / math.log(1.0 / (1.0 - P_RUN)))
    eta = delta_p / (R * (1 + (2 * K - 1) ** 2))
    return {
        "K": K,
        "R": R,
        "eta": eta,
        "reps": median_reps(eta),
        "M_ae": canonical_ae_M(v, eps),
        "cost": float(R * 2 * K * threshold_test_cost(v, eps, eta)),
    }


def quantum_search_cost(sizes: Iterable[int], v: float, eps: float, delta: float, M: int) -> float:
    """Quantum cost of the TEST calls in sizes, union bound over 4 ceil(log2 M) + 4 attempts."""
    delta = _unit("delta", delta)
    M = _int("M", M, 1)
    delta_p = delta / (4 * math.ceil(math.log2(max(M, 2))) + 4)
    return float(sum(prefix_test_cost(m, v, eps, delta_p)["cost"] for m in sizes))


def quantum_core_cost(k: int, v: float, eps: float) -> float:
    """Quantum core over k items: sqrt(k) (2 M + 1) applications of A."""
    k = _int("k", k, 1)
    return float(math.sqrt(k) * (2 * canonical_ae_M(v, eps) + 1))


def bernstein_n(v: float, eps: float, delta: float) -> int:
    """Smallest n with the two-sided Bernstein bound 2 exp(-n eps^2 / (2v + 2eps/3)) <= delta."""
    v = _real("v", v)
    eps = _real("eps", eps, open_lo=True)
    delta = _unit("delta", delta)
    return math.ceil((2.0 * v + 2.0 * eps / 3.0) * math.log(2.0 / delta) / eps ** 2)


def classical_scan_cost(s: int, v: float, eps: float, delta: float, M: int) -> float:
    """Classical scan: s_eff tests (s_eff = s, or M if s = 0), each bernstein_n(delta/s_eff)."""
    M = _int("M", M, 1)
    s = _int("s", s, 0)
    if s > M:
        raise ValueError(f"s must be in [0, M={M}], got {s}")
    s_eff = s if s >= 1 else M
    return float(s_eff * bernstein_n(v, eps, delta / s_eff))


def classical_core_cost(k: int, v: float, eps: float) -> float:
    """Classical Monte Carlo core over k items: k max(v/eps^2, 3/eps)."""
    k = _int("k", k, 1)
    v, eps = _real("v", v), _real("eps", eps, open_lo=True)
    return float(k * max(v / eps ** 2, 3.0 / eps))


def crn_quantum_core_cost(k: int, vD: float, v_ref: float, eps: float) -> float:
    """CRN quantum core: one reference AE to eps/2, then sqrt(k) * 2 signed AEs to eps/4."""
    k = _int("k", k, 1)
    vD, v_ref = _real("vD", vD), _real("v_ref", v_ref)
    ref = 2 * canonical_ae_M(v_ref, eps / 2.0) + 1
    diff = 2 * canonical_ae_M(vD, eps / 4.0) + 1
    return float(ref + math.sqrt(k) * 2 * diff)


def crn_classical_core_cost(k: int, varD: float, v_ref: float, eps: float) -> float:
    """CRN classical core: reference mean to eps/2 plus k difference samples to eps/2."""
    k = _int("k", k, 1)
    varD, v_ref = _real("varD", varD), _real("v_ref", v_ref)
    eps = _real("eps", eps, open_lo=True)
    ref = max(4.0 * v_ref / eps ** 2, 6.0 / eps)
    diff = max(4.0 * varD / eps ** 2, 6.0 / eps)
    return float(ref + k * diff)
