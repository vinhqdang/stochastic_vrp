"""Rotation-model simulators for amplitude estimation (QUASAR M3b v2).

Design: papers/quasar/DESIGN_M3.md, Revision 2, "M3b v2 - estimator behaviour".

Conventions
-----------
* a = P(objective = 1) for the state A|0>, theta = arcsin(sqrt(a)).
* The Grover iterate Q = -A S_0 A^dag S_f has eigenphases +-2 theta. After
  k iterates the objective probability is sin^2((2k+1) theta) (the rotation
  model). :func:`grover_validation` checks this on the real circuits.
* A-application count. The initial A counts 1 and each Grover iterate counts
  2 (one A and one A^dag). Canonical QAE with m evaluation qubits applies
  controlled-Q^(2^j) for j < m, hence costs 2 (2^m - 1) + 1 applications of
  A per run. MLAE with schedule k_j and S shots per power costs
  S * sum_j (2 k_j + 1). Classical Monte Carlo with N samples costs N.
* MLAE schedule: k in {0} U {2^j : j < J}.
"""

from __future__ import annotations

import math
from functools import lru_cache

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import grover_operator
from qiskit.quantum_info import Statevector

from quantum.oracle import exact_probability

GRID_POINTS = 20001
_REFINE_ITERS = 80
_TINY = 1e-300
_GOLD = (math.sqrt(5.0) - 1.0) / 2.0


def theta_of(a: float) -> float:
    """Angle theta in [0, pi/2] with sin^2(theta) = a."""
    return math.asin(math.sqrt(min(max(float(a), 0.0), 1.0)))


def rotation_probability(a: float, k) -> np.ndarray:
    """Model probability sin^2((2k+1) theta) after k Grover iterates."""
    th = theta_of(a)
    return np.sin((2.0 * np.asarray(k, dtype=float) + 1.0) * th) ** 2


# ============================================================
# Circuit validation of the rotation model
# ============================================================


_STD_LEAF = {
    "x", "y", "z", "h", "s", "sdg", "t", "tdg", "sx", "sxdg", "id",
    "rx", "ry", "rz", "p", "u", "u1", "u2", "u3", "cx", "cy", "cz", "cp",
    "swap", "ccx",
}


def _flatten(circuit, qmap=None) -> list:
    """Leaf operations of a circuit as (kind, matrix-or-None, qubit indices).

    Standard gates (and multi-controlled X gates named mcx*) are leaves; any
    other instruction is expanded through its definition. Qubit indices are
    positions in the outermost circuit.
    """
    out: list = []
    if qmap is None:
        qmap = list(range(circuit.num_qubits))
    for inst in circuit.data:
        op = inst.operation
        if op.name == "barrier":
            continue
        qidx = [qmap[circuit.find_bit(q).index] for q in inst.qubits]
        if op.name.startswith("mcx") or op.name in ("x", "cx", "ccx"):
            out.append(("mcx", None, qidx))
        elif op.name in _STD_LEAF:
            out.append(("gate", np.asarray(op.to_matrix(), dtype=complex), qidx))
        else:
            if op.definition is None:
                raise ValueError(f"cannot simulate instruction {op.name!r} without a definition")
            out.extend(_flatten(op.definition, qidx))
    return out


def _apply_gate(psi: np.ndarray, U: np.ndarray, qs, n: int) -> np.ndarray:
    """Apply the k-qubit unitary U (qiskit little-endian on qs) to psi of shape (2,)*n.

    One-qubit gates update the two slices in place; larger gates use tensordot.
    """
    k = len(qs)
    if k == 1:
        ax = n - 1 - qs[0]
        i0 = [slice(None)] * n
        i1 = [slice(None)] * n
        i0[ax] = 0
        i1[ax] = 1
        i0 = tuple(i0)
        i1 = tuple(i1)
        v0 = psi[i0]
        v1 = psi[i1]
        new0 = U[0, 0] * v0 + U[0, 1] * v1
        new1 = U[1, 0] * v0 + U[1, 1] * v1
        psi[i0] = new0
        psi[i1] = new1
        return psi
    Ut = U.reshape([2] * (2 * k))
    axes = [n - 1 - q for q in reversed(qs)]  # big-endian order of the gate qubits
    out = np.tensordot(Ut, psi, axes=(list(range(k, 2 * k)), axes))
    return np.moveaxis(out, list(range(k)), axes)


def _apply_mcx(psi: np.ndarray, controls, target: int, n: int) -> None:
    """In-place multi-controlled X on psi of shape (2,)*n."""
    sl = [slice(None)] * n
    for c in controls:
        sl[n - 1 - c] = 1
    sl0 = list(sl)
    sl1 = list(sl)
    sl0[n - 1 - target] = 0
    sl1[n - 1 - target] = 1
    t0 = tuple(sl0)
    t1 = tuple(sl1)
    tmp = psi[t0].copy()
    psi[t0] = psi[t1]
    psi[t1] = tmp


def _run_leaves(psi: np.ndarray, leaves: list, n: int) -> np.ndarray:
    for kind, U, qs in leaves:
        if kind == "mcx":
            _apply_mcx(psi, qs[:-1], qs[-1], n)
        else:
            psi = _apply_gate(psi, U, qs, n)
    return psi


def grover_validation(oracle, ks=range(7)) -> list[tuple[int, float, float]]:
    """Compare P(objective) after Q^k A|0> with sin^2((2k+1) theta).

    Q is built as in ``quantum.oracle.estimate_probability``: a phase circuit
    with Z on the objective qubit, passed to ``grover_operator`` with the
    oracle circuit as state preparation. The circuits are flattened to leaf
    gates (standard gates, and the multi-controlled X of the reflection kept
    as a native operation) and applied to the dense statevector, one leaf at a
    time. This is an exact statevector simulation without shots; Qiskit's
    Statevector class needs about 0.3 s per gate at 21 qubits and the Aer
    statevector backend segfaults on the reflection, so neither is used.

    sin^2 theta is exact_probability(oracle, method="enumeration"), the
    classical sum over demand assignments, which agrees with the statevector.

    Returns
    -------
    list of (k, p_circuit, p_model)
        p_circuit is the simulated probability of the objective qubit being 1;
        p_model = sin^2((2k+1) theta).
    """
    ks = sorted({int(k) for k in ks})
    if not ks or ks[0] < 0:
        raise ValueError("ks must be non-empty non-negative integers")
    n = oracle.num_qubits
    phase = QuantumCircuit(n)
    phase.z(oracle.objective)
    grover_q = grover_operator(phase, state_preparation=oracle.circuit)
    a = exact_probability(oracle, method="enumeration")
    theta = theta_of(a)
    A_leaves = _flatten(oracle.circuit)
    Q_leaves = _flatten(grover_q)
    psi = np.zeros((2,) * n, dtype=complex)
    psi[(0,) * n] = 1.0
    psi = _run_leaves(psi, A_leaves, n)
    out: list[tuple[int, float, float]] = []
    k_now = 0
    obj_axis = n - 1 - oracle.objective
    for k in ks:
        while k_now < k:
            psi = _run_leaves(psi, Q_leaves, n)
            k_now += 1
        sl = [slice(None)] * n
        sl[obj_axis] = 1
        p_circ = float(np.sum(np.abs(psi[tuple(sl)]) ** 2))
        p_model = math.sin((2 * k + 1) * theta) ** 2
        out.append((k, p_circ, p_model))
    return out


# ============================================================
# Canonical quantum amplitude estimation
# ============================================================


def _dirichlet_sq(x: np.ndarray, N: int) -> np.ndarray:
    """|(1/N) sum_j exp(2 pi i j x)|^2 = sin^2(pi N x) / (N^2 sin^2(pi x))."""
    s = np.sin(np.pi * x)
    small = np.abs(s) < 1e-12
    safe = np.where(small, 1.0, s)
    val = (np.sin(np.pi * N * x) / (N * safe)) ** 2
    return np.where(small, 1.0, val)


def qae_outcome_pmf(a: float, m: int) -> np.ndarray:
    """Exact output distribution of canonical QAE with m evaluation qubits.

    The state A|0> has weight 1/2 on each eigenvector of Q, with eigenphases
    +-theta/pi (in units of 2 pi). Phase estimation on an eigenphase phi gives
    outcome y with probability |D_N(phi - y/N)|^2, N = 2^m. Hence
    P(y) = (1/2) [ |D_N(phi - y/N)|^2 + |D_N(-phi - y/N)|^2 ], phi = theta/pi,
    and the estimate is a_hat = sin^2(pi y / N).
    """
    if m < 1:
        raise ValueError("m must be >= 1")
    N = 1 << m
    phi = theta_of(a) / math.pi
    y = np.arange(N, dtype=float) / N
    p = 0.5 * (_dirichlet_sq(phi - y, N) + _dirichlet_sq(-phi - y, N))
    p = np.clip(p, 0.0, None)
    return p / p.sum()


def qae_cost(m: int) -> int:
    """A-applications per run: 2 (2^m - 1) + 1."""
    return 2 * ((1 << m) - 1) + 1


def qae_estimates(a: float, m: int, n_seeds: int, rng: np.random.Generator) -> np.ndarray:
    """Sample n_seeds independent QAE estimates of a (inverse-CDF sampling)."""
    N = 1 << m
    cdf = np.cumsum(qae_outcome_pmf(a, m))
    cdf[-1] = 1.0
    y = np.searchsorted(cdf, rng.random(n_seeds), side="right")
    y = np.minimum(y, N - 1)
    return np.sin(math.pi * y / N) ** 2


# ============================================================
# Maximum-likelihood amplitude estimation
# ============================================================


def mlae_schedule(J: int) -> list[int]:
    """Evaluation schedule {0} U {2^j : j < J}."""
    if J < 1:
        raise ValueError("J must be >= 1")
    return [0] + [1 << j for j in range(J)]


def mlae_cost(J: int, S: int) -> int:
    """A-applications per run: S * sum_k (2k + 1) over the schedule."""
    return S * sum(2 * k + 1 for k in mlae_schedule(J))


def mlae_counts(a: float, J: int, S: int, n_seeds: int, rng: np.random.Generator) -> np.ndarray:
    """Binomial hit counts, shape (n_seeds, K), for schedule k_j with S shots each."""
    ks = np.array(mlae_schedule(J), dtype=float)
    p = rotation_probability(a, ks)
    return rng.binomial(S, p[None, :], size=(n_seeds, ks.size)).astype(float)


@lru_cache(maxsize=None)
def _grid_logs(J: int):
    """Grid theta, log p_k(theta) and log(1 - p_k(theta)) for schedule J."""
    ks = np.array(mlae_schedule(J), dtype=float)
    th = np.linspace(0.0, math.pi / 2.0, GRID_POINTS)
    ang = (2.0 * ks[:, None] + 1.0) * th[None, :]
    lp = 2.0 * np.log(np.maximum(np.abs(np.sin(ang)), _TINY))
    lq = 2.0 * np.log(np.maximum(np.abs(np.cos(ang)), _TINY))
    return th, lp, lq


def _loglik(theta: np.ndarray, H: np.ndarray, S: int, ks: np.ndarray) -> np.ndarray:
    """Log-likelihood at per-seed angles theta (n,), counts H (n, K)."""
    ang = (2.0 * ks[None, :] + 1.0) * theta[:, None]
    lp = 2.0 * np.log(np.maximum(np.abs(np.sin(ang)), _TINY))
    lq = 2.0 * np.log(np.maximum(np.abs(np.cos(ang)), _TINY))
    return np.sum(H * lp + (S - H) * lq, axis=1)


def mlae_estimates(H: np.ndarray, J: int, S: int, chunk: int = 256) -> np.ndarray:
    """Maximum-likelihood estimates of a from hit counts H (n_seeds, K).

    The likelihood is maximised over theta in [0, pi/2] by a grid of
    GRID_POINTS points (matrix product over seeds), then refined per seed by
    golden-section search on the bracket of the two neighbouring grid points.
    """
    H = np.asarray(H, dtype=float)
    ks = np.array(mlae_schedule(J), dtype=float)
    th, lp, lq = _grid_logs(J)
    G = th.size
    out = np.empty(H.shape[0])
    for s in range(0, H.shape[0], chunk):
        Hc = H[s : s + chunk]
        ll = Hc @ lp + (S - Hc) @ lq
        i = np.argmax(ll, axis=1)
        lo = th[np.maximum(i - 1, 0)]
        hi = th[np.minimum(i + 1, G - 1)]
        for _ in range(_REFINE_ITERS):
            w = hi - lo
            c = hi - _GOLD * w
            d = lo + _GOLD * w
            left = _loglik(c, Hc, S, ks) > _loglik(d, Hc, S, ks)
            hi = np.where(left, d, hi)
            lo = np.where(left, lo, c)
        out[s : s + chunk] = np.sin(0.5 * (lo + hi)) ** 2
    return out


# ============================================================
# Classical Monte Carlo
# ============================================================


def classical_cost(N: int) -> int:
    return int(N)


def classical_estimates(a: float, N: int, n_seeds: int, rng: np.random.Generator) -> np.ndarray:
    """Plain Monte Carlo: n_seeds independent means of N Bernoulli(a) draws."""
    return rng.binomial(N, min(max(float(a), 0.0), 1.0), size=n_seeds) / N


# ============================================================
# RMS error versus cost
# ============================================================


def estimates(a: float, method: str, setting, n_seeds: int, rng: np.random.Generator):
    """Independent estimates of a for one setting. Returns (estimates, cost)."""
    if method == "qae":
        m = int(setting)
        return qae_estimates(a, m, n_seeds, rng), qae_cost(m)
    if method == "mlae":
        J, S = int(setting[0]), int(setting[1])
        H = mlae_counts(a, J, S, n_seeds, rng)
        return mlae_estimates(H, J, S), mlae_cost(J, S)
    if method == "classical":
        N = int(setting)
        return classical_estimates(a, N, n_seeds, rng), classical_cost(N)
    raise ValueError(f"unknown method {method!r}")


def rms_vs_cost(a: float, method: str, settings, n_seeds: int, rng) -> list[tuple]:
    """Error table of one estimator on the rotation model.

    Parameters
    ----------
    a : float
        True amplitude.
    method : {"qae", "mlae", "classical"}
        qae: settings are the evaluation-qubit counts m.
        mlae: settings are (J, S) pairs (schedule J, shots S per power).
        classical: settings are sample sizes N.
    settings : iterable
        Settings for the method.
    n_seeds : int
        Independent replicates per setting.
    rng : numpy.random.SeedSequence or int
        Root seed stream. Each setting receives its own spawned child stream.

    Returns
    -------
    list of (setting, cost, rms, bias)
        cost counts A-applications (qae, mlae) or samples (classical);
        rms = sqrt(mean (a_hat - a)^2); bias = mean (a_hat - a).
    """
    if method not in ("qae", "mlae", "classical"):
        raise ValueError(f"unknown method {method!r}")
    settings = list(settings)
    ss = rng if isinstance(rng, np.random.SeedSequence) else np.random.SeedSequence(rng)
    children = ss.spawn(len(settings))
    rows: list[tuple] = []
    for setting, child in zip(settings, children):
        est, cost = estimates(a, method, setting, n_seeds, np.random.default_rng(child))
        err = est - a
        rows.append((setting, int(cost), float(np.sqrt(np.mean(err * err))), float(np.mean(err))))
    return rows
