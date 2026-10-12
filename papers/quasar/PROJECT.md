# QUASAR — project specification

*Quantum Amplitude estimation for Stochastic vehicle Routing: coherent
feasibility oracles, common random numbers in superposition, and nested
search for chance-constrained SVRPSPD.*

Paper 7 of the repository. Shares instances, scenario generators and the
route/load-profile conventions of the shared engine (`svrpspd_wdro/core`);
all quantum code lives in its own subpackage `svrpspd_wdro/quantum/` with
tests in `svrpspd_wdro/tests/test_quantum_*.py`. Implementation in Qiskit
(simulation on Aer; GPU through Colab only if a statevector run needs it).

Status of every claim below: **PLAN** (to be proved / measured). Nothing in
this file is a result yet.

---

## 1. Problem

SVRPSPD as in BATON: a route r = (i_1, ..., i_L) leaves the depot carrying all
deliveries, drops d_{i_k} and picks up p_{i_k} at the k-th stop. With random,
independent, discrete demands ξ = (d_i, p_i)_i the load profile is

    L_0 = Σ_k d_{i_k},   L_k = L_{k-1} − d_{i_k} + p_{i_k},
    F_r(ξ) = max_{0≤k≤L} L_k        (peak load, core/route.py)

Planning gates: the chance gate P(F_r > Q) ≤ α and the CVaR gate
CVaR_α(F_r) ≤ Q. ALNS evaluates thousands of candidate moves per iteration,
each needing an estimate of a gate quantity for a modified route. Classical
Monte Carlo pays O(σ²/ε²) samples per estimate; this is the cost the project
attacks.

## 2. Research questions

- RQ1 (oracle). What does a coherent quantum oracle for the route peak-load
  predicate cost (qubits, Toffoli/T-count, depth) as a function of the route
  length L, the demand resolution b and the capacity bit-width?
- RQ2 (coherent common random numbers). If two candidate routes are
  evaluated on one shared, superposed scenario register, how much does the
  quantum estimation cost of their *difference* drop, and how does this
  depend on the move type?
- RQ3 (search). What is the query complexity of finding the cheapest
  gate-feasible move among M candidates when feasibility is itself only
  estimable, and is it optimal?
- RQ4 (realism). With fault-tolerant resource estimates, at what precision
  and instance size (if any) would the quadratic speedup beat classical
  Monte Carlo on the repository's real instances?

## 3. Algorithmic design (the novel part)

### 3.1 Peak-load predicate oracle (RQ1)
- State preparation: product of per-customer demand registers,
  |ψ⟩ = ⊗_k Σ_v √P(d_{i_k}=v) |v⟩ ⊗ Σ_u √P(p_{i_k}=u) |u⟩, each b qubits,
  built with uniformly controlled Ry trees (2^b − 1 rotations per register).
  Demands are discretised from the repository's scenario models
  (gamma / lognormal / bimodal) to 2^b levels; the discretisation error on
  P(F_r > Q) is measured, not assumed.
- Streaming reversible evaluation: a load register (c qubits) initialised to
  Σ_k d_{i_k}; for each stop L ← L − d + p (in-place adders), compare L > Q
  against the classical constant Q, and increment an exceedance counter
  (⌈log₂(L+2)⌉ qubits) controlled on the comparator; the objective qubit is
  [counter > 0]; everything except the objective is uncomputed.
- Deliverables: exact qubit and Toffoli formulas; gate-level correctness
  checked by classical simulation of the reversible circuit on every basis
  input for small sizes; end-to-end amplitude checked against an exact
  dynamic program for P(F_r > Q).
- Exact DP (classical reference): state (current load, running peak) over
  stops gives the exact distribution of F_r for independent discrete
  demands; used to obtain true amplitudes on real instances at scale.

### 3.2 Coherent common random numbers (RQ2) — central contribution
- Positioning (LITERATURE.md §5, §7): a difference estimator on shared
  randomness is a corollary of quantum mean estimation, and quantum MLMC
  already estimates coupled differences. What is specific here is the
  route-comparison construction, the insertion lemma and the measured
  window masses; quantum selection papers (2026) use independent noise per
  candidate and leave correlated sampling open.
- Evaluate base route r and moved route r' on the same register ξ and encode
  D(ξ) = 1[F_{r'}(ξ) > Q] − 1[F_r(ξ) > Q].
- **Insertion lemma (to prove).** Inserting customer j anywhere in r gives
  F_r ≤ F_{r'} ≤ F_r + max(d_j, p_j) scenario-wise. Hence D ∈ {0, 1} is an
  indicator and P(D = 1) = P(Q − max(d_j,p_j) < F_r ≤ Q) (up to the exact
  event), the mass of F_r in a window of one demand width below Q.
  (Sketch: stages before j carry d_j extra, the new stage and later carry
  p_j extra.)
- Consequence: amplitude estimation on D needs O(√Δ_j / ε + 1/√ε) oracle
  calls, Δ_j = P(D = 1), against O(√(p'(1−p'))/ε) for estimating the moved
  route alone and O(Δ_j/ε²) for classical CRN. Removal moves give D ∈ {−1,0}
  by symmetry; intra-route moves (relocate, 2-opt) give a signed D, handled
  by two amplitude estimates.
- Incremental feasibility: estimate the base route once to ε/2, then each
  move's increment to ε/2. Total cost Õ(σ_r/ε + Σ_j √Δ_j/ε) instead of
  Õ(Σ_j σ_{r'_j}/ε).
- Measure Δ_j against σ_{r'_j} on real ALNS moves of the Dethloff,
  Salhi–Nagy and city instances (exact DP, no quantum simulation needed for
  the statistics), to show when the window mass is small (anti-concentration
  of the peak for long routes).

- **M3 finding (2026-10-12).** For inter-route insertions the gain is
  negligible (median ρ_q = 0.98 on 648k real moves): Δ = p' − p and an
  insertion multiplies the risk, so the base carries little of it. For
  intra-route moves (relocate, 2-opt) a pilot gives median ρ_q = 0.27 and
  0.44. RQ2 is therefore re-targeted at intra-route moves and at comparisons
  between candidate positions; the insertion result is kept as a negative
  baseline (DESIGN_M3.md).

### 3.3 Cheapest-feasible move search (RQ3)
- Classical ALNS repair sorts candidate insertions by deterministic cost and
  takes the first that passes the gate. Quantum version: exponential search
  over prefixes of the sorted list with Grover search on a bounded-error
  feasibility predicate (amplitude estimation + threshold, gap promise
  α ± ε), giving Õ(√k* · T) predicate calls where k* is the rank of the
  cheapest feasible move and T the per-move estimation cost; with
  variable-time search Õ(√(Σ_{j≤k*} T_j²)) with T_j ∝ √Δ_j/ε from 3.2.
- Lower bounds to establish or cite: classical Ω(k* Δ/ε²) under the gap
  promise; quantum Ω(√k* · √Δ/ε) by composing search and counting lower
  bounds. Generic quantum best-arm identification results must be cited and
  the routing-specific part (CRN variance, sorted-prefix structure) isolated.

### 3.4 CVaR gate (secondary)
- VaR by bisection on P(F_r ≤ x) and the tail expectation by amplitude
  encoding of a bounded linear function (the known quantum risk-analysis
  construction), applied to the peak load; insertion lemma analogue for the
  tail expectation.

### 3.5 Execution layer (later, optional)
- BATON's Longstaff–Schwartz recourse pricing with quantum Monte Carlo
  regression; only if 3.1–3.3 are finished and the novelty check allows.

## 4. Evaluation plan

1. Correctness: exhaustive classical simulation of the reversible circuits;
   Qiskit statevector amplitude vs exact DP on small routes (L ≤ 4, b ≤ 2).
2. Amplitude-estimation behaviour: Qiskit IterativeAmplitudeEstimation and
   MaximumLikelihoodAmplitudeEstimation on the real oracles (small), and the
   exact rotation model of the Grover iterate for larger amplitudes taken
   from the DP (query count vs error).
3. CRN statistics on real instances (3.2), predicted query counts quantum vs
   classical, with and without CRN.
4. Resource estimates (logical qubits, Toffoli counts, surface-code time)
   and the crossover analysis of RQ4, with stated hardware assumptions from
   the literature. A negative crossover result is an acceptable outcome and
   will be reported as such.

## 5. Positioning

See `LITERATURE.md` §7. Closest prior work: amplitude estimation for travel
times in a stochastic VRP (August 2026), amplitude estimation over discrete
decisions with VaR/CVaR (2019–2025), quantum selection with independent
noise (2021, 2026), quantum multilevel Monte Carlo (2021). Not to be
claimed: a first quantum algorithm for stochastic routing, a new variance
reduction technique, or a practical advantage without RQ4.

## 6. Rules

- Every literature claim verified (VERIFY_CITATIONS.md); 2025–2026 work
  covered.
- Every theorem: statement, full proof, numerical check, labelled PROVED /
  PROVED+TESTED / CONJECTURE.
- No claim of practical quantum advantage without the RQ4 analysis.
- Roles: the master plans and designs algorithms; execution is delegated in
  small, specified modules with tests; review is a separate pass.

## 7. Milestones

- M0 project scaffold and literature review (2025–2026 included).
- M1 classical references: exact DP for the peak distribution, demand
  discretisation, insertion lemma checked numerically.
- M2 Qiskit oracle (state preparation + reversible peak predicate) with
  resource counts; correctness tests.
- M3 amplitude estimation experiments and coherent CRN.
- M4 cheapest-feasible search analysis and lower bounds.
- M5 resource estimation and crossover; manuscript.
