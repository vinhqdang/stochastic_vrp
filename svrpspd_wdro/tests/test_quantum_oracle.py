"""Tests for quantum/oracle.py: adder and comparator layouts, the dirty-ancilla
multi-controlled X, exhaustive and randomised basis checks of the peak and
difference oracles (objective, load, counter, restored ancillas), superposition
agreement with the exact classical pmfs, the statevector/enumeration exact
probability, degenerate and infinite thresholds, amplitude estimation query
accounting, the Toffoli-count formula and resource counts."""

from __future__ import annotations

import itertools

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.synthesis import adder_ripple_c04, synth_integer_comparator_2s

from quantum.classical import DiscreteDemand, chance, insertion_window, peak_load
from quantum.oracle import (
    _basis_input,
    _mcx,
    _simulate_basis,
    build_difference_oracle,
    build_peak_oracle,
    estimate_probability,
    evaluate_basis,
    exact_probability,
    resource_counts,
)


def _random_dem(rng, n, V):
    pd = rng.dirichlet(np.ones(V), size=n)
    pp = rng.dirichlet(np.ones(V), size=n)
    return DiscreteDemand(pmf_d=pd, pmf_p=pp, unit=1.0)


def _basis_output(circ: QuantumCircuit, x: int) -> int:
    """Index of the basis state reached from |x> (circuit is a permutation)."""
    sv = Statevector.from_int(x, 2**circ.num_qubits).evolve(circ)
    probs = sv.probabilities()
    j = int(np.argmax(probs))
    assert probs[j] > 1 - 1e-9
    return j


def _bit(v: int, i: int) -> int:
    return (v >> i) & 1


def _reg_value(x: int, oracle, name: str) -> int:
    return sum(_bit(x, q) << i for i, q in enumerate(oracle.registers[name]))


def _route_loads(route, d, p):
    """L_0, ..., L_L of one route for level dictionaries d and p."""
    load = sum(d[c] for c in route)
    loads = [load]
    for c in route:
        load = load - d[c] + p[c]
        loads.append(load)
    return loads


def _dv_to_levels(customers, dv):
    return {c: dv[c][0] for c in customers}, {c: dv[c][1] for c in customers}


# ============================================================
# Building blocks
# ============================================================


def test_adder_layout_basis_states():
    n = 3
    add = adder_ripple_c04(n, kind="fixed")
    assert add.num_qubits == 2 * n + 1
    # Layout: a = qubits [0, n) addend (preserved), b = [n, 2n) receives
    # (a + b) mod 2^n, help = 2n (restored to |0>).
    for a, b in [(0, 0), (1, 0), (3, 5), (7, 7), (5, 6)]:
        x = a | (b << n)
        out = _basis_output(add, x)
        A = sum(_bit(out, i) << i for i in range(n))
        B = sum(_bit(out, n + i) << i for i in range(n))
        assert A == a
        assert B == (a + b) % 2**n
        assert _bit(out, 2 * n) == 0


def test_comparator_layout_basis_states():
    c = 3
    for value in (1, 5, 7):
        comp = synth_integer_comparator_2s(c, value, geq=True)
        assert comp.num_qubits == 2 * c
        # Layout: state qubits [0, c), result q_c, ancillas [c+1, 2c).
        for x in (0, 4, 5, 7):
            out = _basis_output(comp, x)
            state = sum(_bit(out, i) << i for i in range(c))
            assert state == x
            assert _bit(out, c) == int(x >= value)
            assert all(_bit(out, c + 1 + i) == 0 for i in range(c - 1))


@pytest.mark.parametrize("k", range(0, 8))
@pytest.mark.parametrize("pool_kind", ["k-2", "k"])
def test_mcx_ideal_action_and_pool_restored(k, pool_kind):
    """Every basis input (including arbitrary pool contents) is mapped to the
    ideal MCX action, and every pool qubit is restored."""
    if pool_kind == "k-2":
        psize = max(k - 2, 0)
    else:
        psize = k
    n = k + 1 + psize
    controls = list(range(k))
    target = k
    pool = list(range(k + 1, k + 1 + psize))
    circ = QuantumCircuit(n)
    _mcx(circ, controls, target, pool)
    idx = np.arange(2**n, dtype=np.int64)
    wire = [((idx >> q) & 1).astype(bool) for q in range(n)]
    for inst in circ.data:
        name = inst.operation.name
        qs = [circ.find_bit(q).index for q in inst.qubits]
        assert name in ("x", "cx", "ccx")
        if name == "x":
            wire[qs[0]] = ~wire[qs[0]]
        elif name == "cx":
            wire[qs[1]] ^= wire[qs[0]]
        else:
            wire[qs[2]] ^= wire[qs[0]] & wire[qs[1]]
    ctrl_on = np.ones(idx.size, dtype=bool)
    for q in controls:
        ctrl_on &= ((idx >> q) & 1).astype(bool)
    expected_target = ((idx >> target) & 1).astype(bool) ^ ctrl_on
    assert np.array_equal(wire[target], expected_target)
    for q in controls + pool:
        assert np.array_equal(wire[q], ((idx >> q) & 1).astype(bool))


def test_mcx_rejects_overlapping_pool():
    circ = QuantumCircuit(6)
    with pytest.raises(ValueError):
        _mcx(circ, [0, 1, 2], 3, [2, 4, 5])  # pool shares control 2
    with pytest.raises(ValueError):
        _mcx(circ, [0, 1, 2], 3, [3, 4, 5])  # pool shares target 3
    with pytest.raises(ValueError):
        _mcx(circ, [0, 1, 2], 3, [4, 4])  # repeated pool qubit


# ============================================================
# Exhaustive basis checks
# ============================================================


@pytest.mark.parametrize(
    "L,b,Qs",
    [
        (1, 1, [-1, 0, 1]),
        (2, 1, [0, 1, 2]),
        (2, 2, [0, 2, 3, 5, 6]),
    ],
)
def test_peak_oracle_exhaustive_basis(L, b, Qs):
    V = 2**b
    rng = np.random.default_rng(11 + L + b)
    dem = _random_dem(rng, L, V)
    route = list(range(L))
    for Q in Qs:
        o = build_peak_oracle(route, dem, Q)
        for levels in itertools.product(range(V), repeat=2 * L):
            d = {i: levels[2 * i] for i in route}
            p = {i: levels[2 * i + 1] for i in route}
            dv = {i: (d[i], p[i]) for i in route}
            expected = int(peak_load(route, d, p) > Q)
            assert evaluate_basis(o, dv) == expected, (L, b, Q, levels)


def test_peak_oracle_basis_matches_statevector_on_samples():
    rng = np.random.default_rng(3)
    dem = _random_dem(rng, 3, 2)
    o = build_peak_oracle([0, 1, 2], dem, 1)
    for _ in range(4):
        levels = rng.integers(0, 2, size=6)
        dv = {0: (int(levels[0]), int(levels[1])), 1: (int(levels[2]), int(levels[3])),
              2: (int(levels[4]), int(levels[5]))}
        x = _basis_input(o, dv)
        sv = Statevector.from_int(x, 2**o.num_qubits).evolve(o.eval_circuit)
        p1 = sv.probabilities([o.objective])[1]
        assert p1 == pytest.approx(float(evaluate_basis(o, dv)), abs=1e-12)


def test_difference_oracle_exhaustive_basis():
    V = 2
    rng = np.random.default_rng(21)
    dem = _random_dem(rng, 3, V)
    route, j = [0], 1
    for pos, new_route in ((1, [1, 0]), (2, [0, 1])):
        for Q in (0, 1):
            o = build_difference_oracle(route, j, pos, dem, Q)
            for levels in itertools.product(range(V), repeat=4):
                d = {0: levels[0], 1: levels[2]}
                p = {0: levels[1], 1: levels[3]}
                dv = {0: (d[0], p[0]), 1: (d[1], p[1])}
                F_r = peak_load(route, d, p)
                F_new = peak_load(new_route, d, p)
                expected = int(F_new > Q) ^ int(F_r > Q)
                assert evaluate_basis(o, dv) == expected
                assert not (int(F_r > Q) == 1 and int(F_new > Q) == 0)


# ============================================================
# Randomised basis checks: objective, load, counter, restored ancillas
# ============================================================


@pytest.mark.parametrize("L,V", [(7, 2), (7, 4), (12, 2)])
def test_peak_oracle_random_basis_registers(L, V):
    rng = np.random.default_rng(100 + L + V)
    b = V.bit_length() - 1
    dem = _random_dem(rng, L, V)
    route = list(range(L))
    Q = int(rng.integers(0, L * (V - 1)))  # non-degenerate: Q + 1 <= L (V - 1)
    o = build_peak_oracle(route, dem, Q)
    for _ in range(200):
        dv = {i: (int(rng.integers(V)), int(rng.integers(V))) for i in route}
        d, p = _dv_to_levels(route, dv)
        x = _simulate_basis(o, dv)
        loads = _route_loads(route, d, p)
        assert _bit(x, o.objective) == int(max(loads) > Q)
        assert _reg_value(x, o, "load") == loads[-1] == sum(p.values())
        assert _reg_value(x, o, "count") == sum(1 for v in loads if v > Q)
        for name in ("pad", "carry", "cmp"):
            assert _reg_value(x, o, name) == 0, name
        assert _reg_value(x, o, "cmp_anc") == 0
        for i in route:
            assert _reg_value(x, o, f"d_{i}") == dv[i][0]
            assert _reg_value(x, o, f"p_{i}") == dv[i][1]


@pytest.mark.parametrize(
    "V,route,j,pos",
    [
        (2, [0, 1], 2, 1),
        (2, [0, 1], 2, 3),
        (4, [0, 1], 2, 2),
        (2, [0, 1, 2], 3, 2),
        (2, [0, 1, 2], 3, 4),
    ],
)
def test_difference_oracle_random_basis_registers(V, route, j, pos):
    rng = np.random.default_rng(200 + 10 * len(route) + pos + V)
    dem = _random_dem(rng, len(route) + 1, V)
    new_route = route[: pos - 1] + [j] + route[pos - 1 :]
    custs = sorted(set(route) | {j})
    Q = int(rng.integers(0, len(route) * (V - 1)))
    o = build_difference_oracle(route, j, pos, dem, Q)
    for _ in range(200):
        dv = {c: (int(rng.integers(V)), int(rng.integers(V))) for c in custs}
        d, p = _dv_to_levels(custs, dv)
        x = _simulate_basis(o, dv)
        LA = _route_loads(route, d, p)
        LB = _route_loads(new_route, d, p)
        a = int(max(LA) > Q)
        bq = int(max(LB) > Q)
        assert _bit(x, o.registers["obj_A"][0]) == a
        assert _bit(x, o.registers["obj_B"][0]) == bq
        assert _bit(x, o.objective) == a ^ bq
        assert not (a == 1 and bq == 0)
        assert _reg_value(x, o, "load_A") == LA[-1] == sum(p[c] for c in route)
        assert _reg_value(x, o, "load_B") == LB[-1] == sum(p[c] for c in new_route)
        assert _reg_value(x, o, "count_A") == sum(1 for v in LA if v > Q)
        assert _reg_value(x, o, "count_B") == sum(1 for v in LB if v > Q)
        for name in ("pad", "carry", "cmp", "cmp_anc"):
            assert _reg_value(x, o, name) == 0, name
        for c in custs:
            assert _reg_value(x, o, f"d_{c}") == dv[c][0]
            assert _reg_value(x, o, f"p_{c}") == dv[c][1]


# ============================================================
# Superposition agreement and statevector/enumeration equivalence
# ============================================================


@pytest.mark.parametrize(
    "seed,L,b,Q",
    [
        (101, 1, 1, 0),
        (102, 1, 2, 1),
        (103, 2, 1, 1),
        (104, 2, 2, 3),
        (105, 3, 1, 2),
        (106, 3, 1, 4),
        (107, 2, 2, 0),
    ],
)
def test_peak_oracle_probability_matches_chance(seed, L, b, Q):
    rng = np.random.default_rng(seed)
    V = 2**b
    dem = _random_dem(rng, L + 2, V)
    route = [int(c) for c in rng.permutation(L + 2)[:L]]
    o = build_peak_oracle(route, dem, Q)
    assert o.num_qubits <= 24
    assert exact_probability(o) == pytest.approx(chance(route, dem, Q), abs=1e-9)


@pytest.mark.parametrize(
    "seed,route,j,pos",
    [
        (201, [0], 1, 1),
        (202, [0], 1, 2),
        (203, [0, 1], 2, 2),
    ],
)
def test_difference_oracle_probability_matches_insertion_window(seed, route, j, pos):
    rng = np.random.default_rng(seed)
    dem = _random_dem(rng, 3, 2)
    Q = 1
    o = build_difference_oracle(route, j, pos, dem, Q)
    assert exact_probability(o, method="enumeration") == pytest.approx(
        insertion_window(route, j, pos, dem, Q), abs=1e-9
    )


@pytest.mark.parametrize(
    "seed,kind,route,j,pos,Q",
    [
        (301, "peak", [0, 1], None, None, 1),
        (302, "peak", [0, 1, 2], None, None, 2),
        (303, "diff", [0], 1, 2, 1),
        (304, "diff", [0, 1], 2, 3, 1),
    ],
)
def test_enumeration_equals_statevector_small(seed, kind, route, j, pos, Q):
    rng = np.random.default_rng(seed)
    dem = _random_dem(rng, 4, 2)
    if kind == "peak":
        o = build_peak_oracle(route, dem, Q)
    else:
        o = build_difference_oracle(route, j, pos, dem, Q)
    sv = exact_probability(o, method="statevector")
    en = exact_probability(o, method="enumeration")
    assert en == pytest.approx(sv, abs=1e-12)
    assert exact_probability(o) == pytest.approx(sv, abs=1e-12)  # auto -> statevector


def test_enumeration_matches_classical_29_to_40_qubits():
    rng = np.random.default_rng(401)
    # Single route, 31 qubits (V = 4, L = 4).
    dem4 = _random_dem(rng, 5, 4)
    o = build_peak_oracle([0, 1, 2, 3], dem4, 6)
    assert 29 <= o.num_qubits <= 40
    assert exact_probability(o) == pytest.approx(chance([0, 1, 2, 3], dem4, 6), abs=1e-9)

    # Difference oracle, 29 qubits (V = 2, base length 3, insertion at 2).
    dem2 = _random_dem(rng, 4, 2)
    o = build_difference_oracle([0, 1, 2], 3, 2, dem2, 2)
    assert o.num_qubits == 29
    assert exact_probability(o) == pytest.approx(
        insertion_window([0, 1, 2], 3, 2, dem2, 2), abs=1e-9
    )

    # Difference oracle, 40 qubits (V = 4, base length 3).
    dem4b = _random_dem(rng, 4, 4)
    o = build_difference_oracle([0, 1, 2], 3, 2, dem4b, 6)
    assert o.num_qubits == 40
    assert exact_probability(o, method="enumeration") == pytest.approx(
        insertion_window([0, 1, 2], 3, 2, dem4b, 6), abs=1e-9
    )


def test_enumeration_refuses_too_many_assignments():
    rng = np.random.default_rng(402)
    dem = _random_dem(rng, 8, 4)  # V = 4, 8 customers: 4**16 = 2**32 assignments
    o = build_peak_oracle(list(range(8)), dem, 5)
    with pytest.raises(ValueError, match="assignments"):
        exact_probability(o, method="enumeration")


def test_statevector_refuses_above_limit():
    rng = np.random.default_rng(403)
    dem = _random_dem(rng, 8, 4)
    o = build_peak_oracle(list(range(8)), dem, 5)  # 36 + ... qubits > 30
    assert o.num_qubits > 30
    with pytest.raises(ValueError, match="limit"):
        exact_probability(o, method="statevector")


# ============================================================
# Degenerate and infinite thresholds
# ============================================================


def test_degenerate_thresholds():
    rng = np.random.default_rng(31)
    dem = _random_dem(rng, 2, 2)
    L, V = 2, 2
    for Q in (L * (V - 1), L * (V - 1) + 3):
        assert exact_probability(build_peak_oracle([0, 1], dem, Q)) == 0.0
    for Q in (-1, -0.5):
        assert exact_probability(build_peak_oracle([0, 1], dem, Q)) == 1.0


def test_infinite_thresholds_and_nan():
    rng = np.random.default_rng(32)
    dem = _random_dem(rng, 2, 2)
    assert exact_probability(build_peak_oracle([0, 1], dem, float("inf"))) == 0.0
    assert exact_probability(build_peak_oracle([0, 1], dem, float("-inf"))) == 1.0
    o = build_difference_oracle([0], 1, 1, dem, float("inf"))
    assert exact_probability(o) == 0.0
    with pytest.raises(ValueError):
        build_peak_oracle([0, 1], dem, float("nan"))


# ============================================================
# Amplitude estimation
# ============================================================


@pytest.mark.parametrize("method", ["iae", "mlae"])
def test_amplitude_estimation_close_to_exact(method):
    rng = np.random.default_rng(41)
    dem = _random_dem(rng, 3, 2)
    o = build_peak_oracle([0, 1], dem, 1)
    exact = exact_probability(o)
    res = estimate_probability(o, epsilon=0.01, alpha=0.05, method=method, seed=7)
    assert res["method"] == method
    assert abs(res["estimate"] - exact) < 0.05
    assert res["num_oracle_queries"] > res["num_oracle_queries_library"] > 0
    if method == "iae":
        lo, hi = res["ci"]
        assert lo <= exact <= hi


def test_query_accounting_counts_a_applications():
    rng = np.random.default_rng(42)
    dem = _random_dem(rng, 3, 2)
    o = build_peak_oracle([0, 1], dem, 1)
    shots = 1024
    res = estimate_probability(o, method="mlae", seed=7)
    # MLAE schedule [0, 1, 2, 4, 8]: shots * sum(2k + 1) A-applications.
    assert res["num_oracle_queries"] == shots * (1 + 3 + 5 + 9 + 17) == 35840
    assert res["num_oracle_queries_library"] == shots * (0 + 1 + 2 + 4 + 8) == 15360
    res = estimate_probability(o, method="iae", seed=7)
    extra = res["num_oracle_queries"] - 2 * res["num_oracle_queries_library"]
    assert extra > 0 and extra % shots == 0  # one initial A per IAE round


def test_estimate_guard_above_30_qubits():
    rng = np.random.default_rng(43)
    dem = _random_dem(rng, 8, 4)
    o = build_peak_oracle(list(range(8)), dem, 5)
    with pytest.raises(ValueError, match="force"):
        estimate_probability(o, method="mlae")


# ============================================================
# Resource counts and the Toffoli-count formula
# ============================================================


def _expected_ccx(L, c, m):
    """Toffoli count of the single-route U_eval (m >= 3, c >= 2)."""
    return 6 * c * L + 2 * (L + 1) * (2 * c - 3) + (L + 1) * (1 + 2 * (m - 1) * (m - 2)) + 4 * (m - 2)


@pytest.mark.parametrize("L,b", [(5, 1), (10, 3), (20, 3)])
def test_toffoli_count_formula(L, b):
    rng = np.random.default_rng(51 + L)
    V = 2**b
    dem = _random_dem(rng, L, V)
    Q = int(0.6 * L * (V - 1))
    o = build_peak_oracle(list(range(L)), dem, Q)
    rc = resource_counts(o)
    c = (L * (V - 1)).bit_length()
    m = (L + 1).bit_length()
    assert m >= 3
    assert rc["ccx"] == _expected_ccx(L, c, m)


def test_toffoli_count_small_m_terms():
    # m = 2 (L = 2, b = 1): the objective MCX is one Toffoli.
    rng = np.random.default_rng(52)
    dem = _random_dem(rng, 2, 2)
    o = build_peak_oracle([0, 1], dem, 1)
    rc = resource_counts(o)
    c, L = 2, 2
    assert rc["ccx"] == 6 * c * L + 2 * (L + 1) * (2 * c - 3) + (L + 1) + 1 == 34
    # c = 1 (L = 1, b = 1): comparators carry no Toffoli.
    dem1 = _random_dem(rng, 1, 2)
    o1 = build_peak_oracle([0], dem1, 0)
    assert resource_counts(o1)["ccx"] == 6 * 1 * 1 + 0 + 2 + 1 == 9


@pytest.mark.parametrize("L,b", [(2, 1), (3, 2), (5, 2), (10, 3), (20, 3)])
def test_resource_counts_single_route(L, b):
    rng = np.random.default_rng(61 + L)
    V = 2**b
    dem = _random_dem(rng, L, V)
    Q = int(0.6 * L * (V - 1))
    o = build_peak_oracle(list(range(L)), dem, Q)
    rc = resource_counts(o)
    for key in ("qubits", "ccx", "cx", "x", "depth"):
        assert isinstance(rc[key], int)
    assert rc["qubits"] > 0 and rc["ccx"] > 0 and rc["cx"] > 0 and rc["x"] > 0 and rc["depth"] > 0
    c = (L * (V - 1)).bit_length()
    m = (L + 1).bit_length()
    formula = 2 * b * L + c + (c - b) + 1 + 1 + (c - 1) + m + 1
    assert rc["qubits"] == formula
    assert o.num_qubits == formula


def test_resource_counts_prep_adds_rotations():
    rng = np.random.default_rng(71)
    dem = _random_dem(rng, 2, 2)
    o = build_peak_oracle([0, 1], dem, 1)
    eval_rc = resource_counts(o)
    full_rc = resource_counts(o, include_prep=True)
    assert full_rc["u"] > 0
    assert eval_rc["u"] == 0
    assert full_rc["qubits"] == eval_rc["qubits"]
