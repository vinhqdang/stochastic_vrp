"""Coherent peak-load oracles for single routes and single insertions.

The circuits implement the design in ``papers/quasar/DESIGN_M2.md``. A
route r = (i_1, ..., i_L) with demand registers d_i and pickup registers p_i
(levels 0..V-1, V = 2**b) is loaded into a load register of c qubits, where
c = bit_length(L (V - 1)). The running load L_k is tested against the
threshold Q + 1 at every stage (L + 1 stages); the number of stages above the
threshold is accumulated in a counter and the objective qubit is set to
[count != 0] = [F > Q].

Conventions
-----------
* Qiskit little-endian: bit i of a register's integer is stored on the i-th
  qubit of its list.
* Oracle circuits act on a single quantum register; every named register is
  a list of qubit indices into that register.
* The exact pmf of the peak load is preserved by the state preparation
  ``StatePreparation(sqrt(pmf))`` on each demand register.
* Multi-controlled X gates with k >= 3 controls are decomposed into Toffoli
  gates with the dirty-ancilla V-chain (Barenco et al., Lemma 7.2), borrowing
  qubits from the load and comparator-ancilla registers. The borrowed qubits
  are restored for any input state, so the decomposition is exact and uses no
  extra qubits. Only ``x``, ``cx`` and ``ccx`` gates are emitted by the oracle
  itself; the adder and comparator come from ``qiskit.synthesis``.

Insertion convention
--------------------
``build_difference_oracle`` uses the position convention of
``core.route.Route.insert``: pos is the 1-indexed position of the inserted
customer in the new route.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import StatePreparation, grover_operator
from qiskit.quantum_info import Statevector
from qiskit.synthesis import adder_ripple_c04, synth_integer_comparator_2s

from quantum.classical import DiscreteDemand

_CLASSICAL_GATES = ("x", "cx", "ccx")
_INF_LEVEL = 1 << 62  # stands in for +/- infinity thresholds
_MAX_STATEVECTOR_QUBITS = 30
_AUTO_STATEVECTOR_QUBITS = 22
_MAX_ENUM_ASSIGNMENTS = 2**26
_ENUM_CHUNK = 2**18
_SHOTS = 1024


@dataclass
class PeakOracle:
    """A state-preparation circuit A = U_eval . P with a marked objective qubit.

    Attributes
    ----------
    circuit : QuantumCircuit
        A = U_eval . P acting on the full register set.
    objective : int
        Index of the objective qubit; P(objective = 1) is the target
        probability.
    num_qubits : int
        Total number of qubits.
    registers : dict[str, list[int]]
        Named registers as lists of qubit indices (little-endian).
    eval_circuit : QuantumCircuit
        U_eval alone, on the same qubits.
    prep_circuit : QuantumCircuit
        P alone (state preparation of the demand registers).
    customers : tuple[int, ...]
        Customers whose demand registers are part of the oracle, in the order
        used for the register layout.
    levels : int
        Number of demand levels V = 2**b.
    demand : DiscreteDemand or None
        Demand model the oracle was built from; used by the enumeration method
        of :func:`exact_probability`.
    """

    circuit: QuantumCircuit
    objective: int
    num_qubits: int
    registers: dict[str, list[int]]
    eval_circuit: QuantumCircuit
    prep_circuit: QuantumCircuit
    customers: tuple[int, ...] = ()
    levels: int = 0
    demand: DiscreteDemand | None = None
    _classical_ops: list | None = field(default=None, repr=False, compare=False)


# ============================================================
# Helpers
# ============================================================


def _demand_bits(dem: DiscreteDemand) -> int:
    V = dem.levels
    if V < 2 or V & (V - 1):
        raise ValueError(f"Number of levels must be a power of two >= 2, got {V}")
    return V.bit_length() - 1


def _check_route(route: Sequence[int], dem: DiscreteDemand) -> list[int]:
    seq = [int(c) for c in route]
    if not seq:
        raise ValueError("Route must be non-empty")
    if len(set(seq)) != len(seq):
        raise ValueError(f"Route contains repeated customers: {seq}")
    for c in seq:
        if not 0 <= c < dem.n_customers:
            raise IndexError(f"Customer {c} out of range [0, {dem.n_customers})")
    return seq


def _floor_level(Q_level: float) -> int:
    """floor(Q_level) as an int; +/- infinity map to large sentinels."""
    q = float(Q_level)
    if math.isnan(q):
        raise ValueError("Q_level must not be NaN")
    if math.isinf(q):
        return _INF_LEVEL if q > 0 else -_INF_LEVEL
    return math.floor(q)


def _layout(spec: Sequence[tuple[str, int]]) -> tuple[dict[str, list[int]], int]:
    """Assign consecutive qubit indices to named registers of given sizes."""
    registers: dict[str, list[int]] = {}
    nxt = 0
    for name, size in spec:
        if name in registers:
            raise ValueError(f"Duplicate register name {name}")
        registers[name] = list(range(nxt, nxt + size))
        nxt += size
    return registers, nxt


def _mcx(circ: QuantumCircuit, controls: Sequence[int], target: int, pool: Sequence[int]) -> None:
    """Append a multi-controlled X using only x, cx and ccx gates.

    With k >= 3 controls the dirty-ancilla V-chain is used, borrowing the
    first k - 2 qubits of ``pool``; the borrowed qubits are restored for any
    input state. The pool must be duplicate-free and disjoint from the
    controls and the target.
    """
    controls = list(controls)
    pool = list(pool)
    used = controls + [target]
    if len(set(used)) != len(used):
        raise ValueError(f"Controls and target must be distinct, got {used}")
    if len(set(pool)) != len(pool):
        raise ValueError(f"Dirty-ancilla pool has repeated qubits: {pool}")
    if set(pool) & set(used):
        raise ValueError("Dirty-ancilla pool must be disjoint from controls and target")
    k = len(controls)
    if k == 0:
        circ.x(target)
    elif k == 1:
        circ.cx(controls[0], target)
    elif k == 2:
        circ.ccx(controls[0], controls[1], target)
    else:
        if len(pool) < k - 2:
            raise ValueError(f"Need {k - 2} dirty ancillas, pool has {len(pool)}")
        a = list(pool[: k - 2])
        c = list(controls)
        top = [(c[k - 1], a[k - 3], target)]
        down = [(c[i + 1], a[i - 1], a[i]) for i in range(k - 3, 0, -1)]
        base = [(c[0], c[1], a[0])]
        up = [(c[i + 1], a[i - 1], a[i]) for i in range(1, k - 2)]
        for x, y, z in top + down + base + up + top + down + base + up:
            circ.ccx(x, y, z)


def _append_pipeline(
    circ: QuantumCircuit,
    route: Sequence[int],
    dreg: Mapping[int, list[int]],
    preg: Mapping[int, list[int]],
    V: int,
    Q_floor: int,
    pad: list[int],
    load: list[int],
    carry: int,
    cmp_q: int,
    cmp_anc: list[int],
    count: list[int],
    obj: int,
    c: int,
) -> None:
    """Append U_eval for one route: steps 2-5 of DESIGN_M2.md.

    Q_floor is floor(Q). A constant objective is handled without comparators:
    Q_floor < 0 gives 1 and Q_floor + 1 > L (V - 1) gives 0.
    """
    L = len(route)
    if Q_floor < 0:
        circ.x(obj)
        return
    if Q_floor + 1 > L * (V - 1):
        return
    value = Q_floor + 1
    m = len(count)

    add = adder_ripple_c04(c, kind="fixed")
    sub = add.inverse()
    comp = synth_integer_comparator_2s(c, value, geq=True)
    comp_inv = comp.inverse()
    comp_qubits = list(load) + [cmp_q] + list(cmp_anc)
    pool = list(cmp_anc) + list(load)

    def add_reg(reg: list[int], op: QuantumCircuit) -> None:
        circ.compose(op, qubits=list(reg) + list(load) + [carry], inplace=True)

    def stage() -> None:
        circ.compose(comp, qubits=comp_qubits, inplace=True)
        for i in reversed(range(m)):
            _mcx(circ, [cmp_q] + list(count[:i]), count[i], pool)
        circ.compose(comp_inv, qubits=comp_qubits, inplace=True)

    # Step 2: load = sum of deliveries (L_0).
    for cust in route:
        add_reg(list(dreg[cust]) + list(pad), add)
    # Step 3: stage 0.
    stage()
    # Step 4: stages 1..L.
    for cust in route:
        add_reg(list(dreg[cust]) + list(pad), sub)
        add_reg(list(preg[cust]) + list(pad), add)
        stage()
    # Step 5: obj = [count != 0] via X-conjugated multi-controlled X.
    for q in count:
        circ.x(q)
    _mcx(circ, list(count), obj, pool)
    circ.x(obj)
    for q in count:
        circ.x(q)


def _demand_prep(prep: QuantumCircuit, dem: DiscreteDemand, cust: int, dreg: list[int], preg: list[int]) -> None:
    for pmf, reg in ((dem.pmf_d[cust], dreg), (dem.pmf_p[cust], preg)):
        amp = np.sqrt(np.clip(np.asarray(pmf, dtype=np.float64), 0.0, None))
        amp = amp / np.linalg.norm(amp)
        prep.compose(StatePreparation(amp), qubits=list(reg), inplace=True)


def _classical_program(oracle: PeakOracle) -> list[tuple[str, tuple[int, ...]]]:
    if oracle._classical_ops is None:
        t = transpile(oracle.eval_circuit, basis_gates=list(_CLASSICAL_GATES), optimization_level=0)
        ops: list[tuple[str, tuple[int, ...]]] = []
        for inst in t.data:
            name = inst.operation.name
            if name not in _CLASSICAL_GATES:
                raise ValueError(f"Non-classical gate {name} in eval circuit")
            qs = tuple(t.find_bit(q).index for q in inst.qubits)
            ops.append((name, qs))
        oracle._classical_ops = ops
    return oracle._classical_ops


# ============================================================
# Builders
# ============================================================


def build_peak_oracle(route: Sequence[int], dem: DiscreteDemand, Q_level: float) -> PeakOracle:
    """Coherent oracle for the objective [F_r > Q] of a single route.

    Parameters
    ----------
    route : sequence of int
        Customer indices in visiting order.
    dem : DiscreteDemand
        Discrete demand model with V = 2**b levels.
    Q_level : float
        Threshold in level units; the objective is 1 iff F > Q_level. A
        non-integer value is handled as floor(Q_level), as in
        ``classical.chance``.

    Returns
    -------
    PeakOracle
        Circuit A = U_eval . P with P(objective = 1) = classical.chance(route,
        dem, Q_level). Registers: ``d_<i>``, ``p_<i>`` (demand registers of
        customer i), ``pad``, ``load``, ``carry``, ``cmp``, ``cmp_anc``,
        ``count``, ``obj``.
    """
    seq = _check_route(route, dem)
    b = _demand_bits(dem)
    V = dem.levels
    L = len(seq)
    c = (L * (V - 1)).bit_length()
    m = (L + 1).bit_length()
    q = _floor_level(Q_level)

    spec: list[tuple[str, int]] = []
    for cust in seq:
        spec.append((f"d_{cust}", b))
        spec.append((f"p_{cust}", b))
    spec += [("pad", c - b), ("load", c), ("carry", 1), ("cmp", 1), ("cmp_anc", c - 1), ("count", m), ("obj", 1)]
    regs, n = _layout(spec)

    prep = QuantumCircuit(n)
    for cust in seq:
        _demand_prep(prep, dem, cust, regs[f"d_{cust}"], regs[f"p_{cust}"])

    eval_c = QuantumCircuit(n)
    dreg = {cust: regs[f"d_{cust}"] for cust in seq}
    preg = {cust: regs[f"p_{cust}"] for cust in seq}
    _append_pipeline(
        eval_c, seq, dreg, preg, V, q, regs["pad"], regs["load"], regs["carry"][0],
        regs["cmp"][0], regs["cmp_anc"], regs["count"], regs["obj"][0], c,
    )
    circuit = prep.compose(eval_c)
    return PeakOracle(
        circuit=circuit,
        objective=regs["obj"][0],
        num_qubits=n,
        registers=regs,
        eval_circuit=eval_c,
        prep_circuit=prep,
        customers=tuple(seq),
        levels=V,
        demand=dem,
    )


def build_difference_oracle(
    route: Sequence[int], j: int, pos: int, dem: DiscreteDemand, Q_level: float
) -> PeakOracle:
    """Coherent oracle for D = 1[F_{r'} > Q] XOR 1[F_r > Q].

    r is ``route`` and r' is ``route`` with customer j inserted at 1-indexed
    position ``pos`` (``core.route.Route.insert`` convention). Both pipelines
    share the demand registers of all customers in r and r', and reuse the
    pad, carry, comparator and comparator-ancilla registers. The objective is
    the ``D`` qubit. By the insertion lemma, P(D = 1) equals
    ``classical.insertion_window`` for the same arguments.

    Registers: ``d_<i>``, ``p_<i>`` for i in r and r', ``pad``, ``load_A``,
    ``load_B``, ``carry``, ``cmp``, ``cmp_anc``, ``count_A``, ``count_B``,
    ``obj_A``, ``obj_B``, ``D``.
    """
    seq = _check_route(route, dem)
    j = int(j)
    if j in seq:
        raise ValueError(f"Customer {j} already in route")
    if not 0 <= j < dem.n_customers:
        raise IndexError(f"Customer {j} out of range [0, {dem.n_customers})")
    if not 1 <= pos <= len(seq) + 1:
        raise IndexError(f"Insertion position {pos} out of range [1, {len(seq) + 1}]")
    new_seq = seq[: pos - 1] + [j] + seq[pos - 1 :]
    b = _demand_bits(dem)
    V = dem.levels
    L = len(seq)
    c = ((L + 1) * (V - 1)).bit_length()
    m_A = (L + 1).bit_length()
    m_B = (L + 2).bit_length()
    custs = sorted(set(seq) | {j})

    spec: list[tuple[str, int]] = []
    for cust in custs:
        spec.append((f"d_{cust}", b))
        spec.append((f"p_{cust}", b))
    spec += [
        ("pad", c - b), ("load_A", c), ("load_B", c), ("carry", 1), ("cmp", 1),
        ("cmp_anc", c - 1), ("count_A", m_A), ("count_B", m_B),
        ("obj_A", 1), ("obj_B", 1), ("D", 1),
    ]
    regs, n = _layout(spec)

    prep = QuantumCircuit(n)
    for cust in custs:
        _demand_prep(prep, dem, cust, regs[f"d_{cust}"], regs[f"p_{cust}"])

    eval_c = QuantumCircuit(n)
    dreg = {cust: regs[f"d_{cust}"] for cust in custs}
    preg = {cust: regs[f"p_{cust}"] for cust in custs}
    q = _floor_level(Q_level)
    _append_pipeline(
        eval_c, seq, dreg, preg, V, q, regs["pad"],
        regs["load_A"], regs["carry"][0], regs["cmp"][0], regs["cmp_anc"],
        regs["count_A"], regs["obj_A"][0], c,
    )
    _append_pipeline(
        eval_c, new_seq, dreg, preg, V, q, regs["pad"],
        regs["load_B"], regs["carry"][0], regs["cmp"][0], regs["cmp_anc"],
        regs["count_B"], regs["obj_B"][0], c,
    )
    d_q = regs["D"][0]
    eval_c.cx(regs["obj_A"][0], d_q)
    eval_c.cx(regs["obj_B"][0], d_q)

    circuit = prep.compose(eval_c)
    return PeakOracle(
        circuit=circuit,
        objective=d_q,
        num_qubits=n,
        registers=regs,
        eval_circuit=eval_c,
        prep_circuit=prep,
        customers=tuple(custs),
        levels=V,
        demand=dem,
    )


# ============================================================
# Evaluation
# ============================================================


def _basis_input(oracle: PeakOracle, demand_values: Mapping[int, tuple[int, int]]) -> int:
    """Integer basis label with the given demand levels and all else |0>."""
    V = oracle.levels
    x = 0
    for cust in oracle.customers:
        if cust not in demand_values:
            raise KeyError(f"No demand value given for customer {cust}")
        d, p = demand_values[cust]
        for name, lv in ((f"d_{cust}", d), (f"p_{cust}", p)):
            if not 0 <= int(lv) < V:
                raise ValueError(f"Level {lv} for {name} outside [0, {V - 1}]")
            for i, q in enumerate(oracle.registers[name]):
                if (int(lv) >> i) & 1:
                    x |= 1 << q
    return x


def _simulate_basis(oracle: PeakOracle, demand_values: Mapping[int, tuple[int, int]]) -> int:
    """Final basis label (all qubits) of U_eval on the given demand input."""
    x = _basis_input(oracle, demand_values)
    for gate, qs in _classical_program(oracle):
        if gate == "x":
            x ^= 1 << qs[0]
        elif gate == "cx":
            if (x >> qs[0]) & 1:
                x ^= 1 << qs[1]
        elif (x >> qs[0]) & 1 and (x >> qs[1]) & 1:
            x ^= 1 << qs[2]
    return x


def evaluate_basis(oracle: PeakOracle, demand_values: Mapping[int, tuple[int, int]]) -> int:
    """Objective bit of U_eval on a computational-basis demand assignment.

    Parameters
    ----------
    oracle : PeakOracle
        Oracle built by one of the builders.
    demand_values : mapping customer -> (d_level, p_level)
        Integer levels in [0, V-1] for every customer in ``oracle.customers``.
        All non-demand qubits start in |0>.

    Returns
    -------
    int
        0 or 1, the value of the objective qubit after U_eval (U_eval maps
        basis states to basis states, so the result is deterministic).

    Notes
    -----
    U_eval consists of x, cx and ccx gates only, so it is simulated exactly on
    the integer basis label. The circuit is transpiled once and cached on the
    oracle.
    """
    return (_simulate_basis(oracle, demand_values) >> oracle.objective) & 1


def _enumerate_probability(oracle: PeakOracle, chunk: int = _ENUM_CHUNK) -> float:
    """Exact P(objective = 1) by enumerating all demand assignments.

    A is U_eval after the product state preparation P. Since U_eval is a
    classical permutation and every non-demand qubit starts in |0>,
    P(objective = 1) = sum_x w(x) * obj(U_eval x), where w(x) is the product of
    the normalised per-register pmfs. The assignments are processed in chunks
    of bit-sliced boolean arrays, one array per qubit.
    """
    dem = oracle.demand
    if dem is None:
        raise ValueError("Enumeration needs the demand model; build the oracle with a builder")
    V = oracle.levels
    custs = oracle.customers
    K = 2 * len(custs)
    N = V**K
    if N > _MAX_ENUM_ASSIGNMENTS:
        raise ValueError(
            f"Enumeration needs {N} demand assignments, above the limit "
            f"{_MAX_ENUM_ASSIGNMENTS} (2**26); use a smaller instance"
        )
    pmfs: list[np.ndarray] = []
    for cu in custs:
        for raw in (dem.pmf_d[cu], dem.pmf_p[cu]):
            raw = np.asarray(raw, dtype=np.float64)
            pmfs.append(raw / raw.sum())
    program = _classical_program(oracle)
    n = oracle.num_qubits
    total = 0.0
    for start in range(0, N, chunk):
        idx = np.arange(start, min(N, start + chunk), dtype=np.int64)
        w = np.ones(idx.size, dtype=np.float64)
        wire = [np.zeros(idx.size, dtype=bool) for _ in range(n)]
        for r in range(K):
            t, kind = divmod(r, 2)
            name = f"{'d' if kind == 0 else 'p'}_{custs[t]}"
            lv = (idx // V ** (K - 1 - r)) % V
            w *= pmfs[r][lv]
            for i, q in enumerate(oracle.registers[name]):
                wire[q] = ((lv >> i) & 1).astype(bool)
        for gate, qs in program:
            if gate == "x":
                wire[qs[0]] = ~wire[qs[0]]
            elif gate == "cx":
                wire[qs[1]] ^= wire[qs[0]]
            else:
                wire[qs[2]] ^= wire[qs[0]] & wire[qs[1]]
        total += float(w[wire[oracle.objective]].sum())
    return total


def exact_probability(oracle: PeakOracle, method: str = "auto") -> float:
    """P(objective = 1) for the state A|0...0>.

    Parameters
    ----------
    oracle : PeakOracle
        Oracle built by one of the builders.
    method : {"auto", "statevector", "enumeration"}
        ``"statevector"`` evolves the full state (limited to 30 qubits).
        ``"enumeration"`` sums w(x) * obj(U_eval x) over all demand assignments
        (limited to 2**26 assignments; needs the demand model stored by the
        builders). ``"auto"`` uses the statevector for at most 22 qubits and
        the enumeration otherwise.

    Returns
    -------
    float
    """
    if method not in ("auto", "statevector", "enumeration"):
        raise ValueError(f"Unknown method {method!r}")
    if method == "auto":
        method = "statevector" if oracle.num_qubits <= _AUTO_STATEVECTOR_QUBITS else "enumeration"
    if method == "enumeration":
        return _enumerate_probability(oracle)
    if oracle.num_qubits > _MAX_STATEVECTOR_QUBITS:
        raise ValueError(
            f"Statevector needs {oracle.num_qubits} qubits, above the limit "
            f"{_MAX_STATEVECTOR_QUBITS}; use method='enumeration'"
        )
    sv = Statevector(oracle.circuit)
    probs = sv.probabilities([oracle.objective])
    return float(probs[1])


def resource_counts(oracle: PeakOracle, include_prep: bool = False) -> dict:
    """Gate and depth counts after transpiling to {x, cx, ccx, u}.

    Parameters
    ----------
    oracle : PeakOracle
        Oracle to measure.
    include_prep : bool
        If True count the full circuit A = U_eval . P (state preparation
        rotations appear as u gates); otherwise count U_eval only.

    Returns
    -------
    dict
        Keys ``qubits``, ``ccx``, ``cx``, ``x``, ``u``, ``depth``. Transpiled
        with ``optimization_level=0``. Multi-controlled X gates are emitted as
        Toffoli V-chains, so no mcx gate remains after transpilation.

    Notes
    -----
    For a single-route U_eval with L stops, c = bit_length(L (V-1)) load
    qubits and m = bit_length(L+1) counter qubits, the Toffoli count is

        ccx = 6 c L + 2 (L+1) (2c - 3) + (L+1) (1 + 2 (m-1)(m-2)) + 4 (m-2),

    valid for m >= 3 and c >= 2. The terms are: 3L adder applications at 2c
    Toffolis each; L+1 comparators, each computed and uncomputed, at 2c-3
    each; per stage a controlled increment of cost 1 (the cx at bit 0 is not
    a Toffoli) plus 4(i-1) for each bit i >= 2, summing to 1 + 2(m-1)(m-2);
    and the objective MCX with m controls at 4(m-2).

    Small cases observed on the circuit: for m = 2 (L = 1 or 2) the objective
    MCX is a single Toffoli, so the formula is short by one, i.e.
    ccx = 6cL + 2(L+1)(2c-3) + (L+1) + 1 (checked at (L,b) = (2,1), 34 gates).
    For c = 1 (L = 1, V = 2) the comparators contain no Toffoli, so their
    term is 0 instead of 2c-3 = -1; ccx = 6 + 0 + 2 + 1 = 9 (checked).
    """
    circ = oracle.circuit if include_prep else oracle.eval_circuit
    t = transpile(circ, basis_gates=["x", "cx", "ccx", "u"], optimization_level=0)
    ops = t.count_ops()
    return {
        "qubits": int(circ.num_qubits),
        "ccx": int(ops.get("ccx", 0)),
        "cx": int(ops.get("cx", 0)),
        "x": int(ops.get("x", 0)),
        "u": int(ops.get("u", 0)),
        "depth": int(t.depth()),
    }


def estimate_probability(
    oracle: PeakOracle,
    epsilon: float = 0.01,
    alpha: float = 0.05,
    method: str = "iae",
    seed: int = 7,
    force: bool = False,
) -> dict:
    """Amplitude estimation of P(objective = 1).

    Parameters
    ----------
    oracle : PeakOracle
        Oracle whose circuit is the state-preparation operator A.
    epsilon : float
        Target accuracy (additive, on the probability) for IAE.
    alpha : float
        Confidence level 1 - alpha for IAE's interval.
    method : {"iae", "mlae"}
        Iterative amplitude estimation, or maximum-likelihood amplitude
        estimation with evaluation schedule 4 (powers 0, 1, 2, 4, 8).
    seed : int
        Seed for the sampler.
    force : bool
        Allow more than 30 qubits. The statevector sampler needs 2**n
        amplitudes, so larger oracles are refused by default.

    Returns
    -------
    dict
        ``estimate`` (float), ``ci`` (lower, upper), ``num_oracle_queries``
        (int), ``num_oracle_queries_library`` (int) and ``method``.

    Notes
    -----
    ``num_oracle_queries`` counts applications of A: the initial A plus, for
    each shot of a round with Grover power k, the k iterates Q = -A S0 A^dag S_f
    (two applications of A each). Per round this is shots * (2k + 1). The
    rounds are the Grover powers of the run: ``powers[1:]`` for IAE and
    ``evaluation_schedule`` for MLAE. ``num_oracle_queries_library`` is the
    raw count reported by qiskit_algorithms, which counts shots * k only (it
    omits the initial A and the reflections).
    """
    from qiskit_algorithms import (
        EstimationProblem,
        IterativeAmplitudeEstimation,
        MaximumLikelihoodAmplitudeEstimation,
    )

    if oracle.num_qubits > _MAX_STATEVECTOR_QUBITS and not force:
        raise ValueError(
            f"estimate_probability simulates {oracle.num_qubits} qubits with a statevector "
            f"sampler (limit {_MAX_STATEVECTOR_QUBITS}); pass force=True to override"
        )
    sampler = _make_sampler(seed)
    # Grover operator built with the non-deprecated constructor. The phase
    # oracle Z on the objective qubit marks the good states (objective = 1).
    phase = QuantumCircuit(oracle.num_qubits)
    phase.z(oracle.objective)
    grover_q = grover_operator(phase, state_preparation=oracle.circuit)
    problem = EstimationProblem(
        state_preparation=oracle.circuit,
        objective_qubits=[oracle.objective],
        grover_operator=grover_q,
    )
    if method == "iae":
        estimator = IterativeAmplitudeEstimation(epsilon_target=epsilon, alpha=alpha, sampler=sampler)
    elif method == "mlae":
        estimator = MaximumLikelihoodAmplitudeEstimation(evaluation_schedule=4, sampler=sampler)
    else:
        raise ValueError(f"Unknown method {method!r}; use 'iae' or 'mlae'")
    res = estimator.estimate(problem)
    if method == "iae":
        rounds = [int(k) for k in list(res.powers)[1:]]
    else:
        rounds = [int(k) for k in res.evaluation_schedule]
    shots = int(sampler.default_shots)
    lo, hi = res.confidence_interval
    return {
        "estimate": float(res.estimation),
        "ci": (float(lo), float(hi)),
        "num_oracle_queries": int(shots * sum(2 * k + 1 for k in rounds)),
        "num_oracle_queries_library": int(res.num_oracle_queries),
        "method": method,
    }


def _make_sampler(seed: int):
    """Seeded statevector sampler (exact amplitudes, reproducible shots).

    Aer's SamplerV2 does not execute the ``StatePreparation`` instruction in
    the oracle circuit without a prior transpilation, so the exact statevector
    sampler from ``qiskit.primitives`` is used.
    """
    from qiskit.primitives import StatevectorSampler

    return StatevectorSampler(default_shots=_SHOTS, seed=seed)
