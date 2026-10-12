# QUASAR M2 — oracle design

Design record for the coherent peak-load oracle (PROJECT.md §3.1) and the
shared-register difference oracle (§3.2). Implementation:
`svrpspd_wdro/quantum/oracle.py`; tests: `tests/test_quantum_oracle.py`.

## Conventions
- Qiskit little-endian: a register's integer is Σ_i bit_i 2^i.
- Demands in level units 0..V−1, V = 2^b (from `classical.DiscreteDemand`).
- Route r = (i_1, …, i_L); load L_0 = Σ_k d_{i_k}, L_k = L_{k−1} − d_{i_k} +
  p_{i_k}; peak F = max_k L_k. Every intermediate value lies in
  [0, L(V−1)] (L_{k−1} − d_{i_k} = Σ_{l>k} d + Σ_{l<k} p ≥ 0), so a load
  register of c = ⌈log₂(L(V−1)+1)⌉ qubits never overflows.

## Registers (single route)
| register | qubits | role |
|---|---|---|
| d_k, p_k (k = 1..L) | b each | demand registers, prepared in superposition |
| pad | c − b | always |0⟩; extends a b-qubit demand to c bits for the adder |
| load | c | running load |
| carry | 1 | helper of the ripple-carry adder, restored to |0⟩ |
| cmp | 1 | comparator output, restored after use |
| cmp_anc | c − 1 | comparator ancillas, restored |
| count | m = ⌈log₂(L+2)⌉ | number of stages with load > Q (garbage) |
| obj | 1 | objective qubit = [F > Q] |

## Circuit A = U_eval · P
1. P: on every demand register, a state preparation of √pmf (Qiskit
   `StatePreparation`; resource formulas use the analytic 2^b − 1 rotations of
   a uniformly controlled Ry tree).
2. load += d_k for k = 1..L (Cuccaro adder `adder_ripple_c04(c, kind="fixed")`
   on (d_k ‖ pad, load, carry)).
3. Stage 0: cmp ← [load ≥ Q+1] (`synth_integer_comparator_2s(c, Q+1,
   geq=True)`); count += cmp (controlled increment, MCX ladder from the top
   bit down); un-compare.
4. For k = 1..L: load −= d_k (inverse adder); load += p_k; compare,
   controlled increment, un-compare.
5. obj ← [count ≠ 0] (X on count, MCX into obj, X on obj, X on count).
Load and count are left as garbage: amplitude estimation needs only a
unitary A with the objective marked, and the Grover iterate uses A and A†.
Degenerate thresholds: if Q+1 > L(V−1) the objective is identically 0; if
Q < 0 it is identically 1 (no comparators are emitted).

Expected size: 2bL + c + (c−b) + 1 + 1 + (c−1) + m + 1 qubits;
Toffoli count ≈ 3L · 2c (adders) + 2(L+1)·T_cmp(c) + (L+1)·T_inc(m), with
T_cmp and T_inc read off the synthesised circuits. Exact formulas are
derived from the counts in M2 and stated in the paper.

## Difference oracle (coherent common random numbers)
Base route r and moved route r' (customer j inserted at position pos) share
the demand registers of all customers in r ∪ {j}. Run steps 2–5 twice, into
(load_A, count_A, obj_A) for r and (load_B, count_B, obj_B) for r', reusing
pad, carry, cmp and cmp_anc (they are restored). Then D ← obj_A ⊕ obj_B.
By the insertion lemma D = 1[F_{r'} > Q] − 1[F_r > Q] ∈ {0,1}, so P(D=1) is
exactly `classical.insertion_window`.

## Correctness checks (tests)
- Basis inputs: for every assignment of the demand registers (small L, b),
  U_eval sets obj = [F > Q] (and D as above) deterministically.
- Superposition: P(obj = 1) of A equals `classical.chance` to 1e−9
  (statevector); P(D = 1) equals `classical.insertion_window`.
- Amplitude estimation (qiskit-algorithms IAE / MLAE) on A returns an
  estimate within its confidence interval of the exact value.
