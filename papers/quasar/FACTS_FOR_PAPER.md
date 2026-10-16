# QUASAR: facts for the manuscript (IJOC, Quantum Computing area)

Compiled 2026-10-16 from the files listed per fact. Numbers are copied from the source; anything
marked COMPUTED-HERE was evaluated by the compiler from a stated formula or table and is not in
any file. MISSING = not in the files. Paths are relative to /home/user/stochastic_vrp.
Abbreviations: PQ = papers/quasar, RQ = svrpspd_wdro/results/quasar, QD = svrpspd_wdro/quantum,
"line N" = line in the file as numbered by `cat -n`.

Global setting (all tests)
- Model: SVRPSPD route r=(i_1..i_L); L_0 = sum_k d_{i_k}, L_k = L_{k-1} - d_{i_k} + p_{i_k}; peak F = max_{0<=k<=L} L_k; failure X = 1[F > Q]. (PQ/PROJECT.md l20-26; QD/classical.py docstring.)
- Demand model of all measurements: BATON's: gamma marginals cv 0.3, one-factor Gaussian copula rho = 0.6, separately for deliveries and pickups (independent factors). (PQ/DESIGN_M3.md l100-106; PQ/DESIGN_RARE.md l12-14.)
- Instances: 73 cached plans (Dethloff CON, SCA; Salhi-Nagy CMT; city) (PQ/DESIGN_M3.md l9-10) and 148 region instances, Northern Vietnam (VNN) 35.3 M and Guangdong (GD) 115.5 M people, 10 518 routes in the plans (PQ/STATUS.md l24; RQ/cert_region_summary.md l3). Instances per region: MISSING.
- Gate level alpha = 0.10 (chance gate: certified route has p <= alpha), delta = 0.05 in M3/M4/cert.
- In the BATON model P(F>Q) has an exact DP (with quadrature over factors). So every test below is a testbed measuring query counts, not a use case; the paper must say so (PQ/REFRAME_M4.md l21-26, PQ/THEORY_CERT.md l32-37).
- Plans: "Det" = deterministic-gate plans, "SAA" = SAA-CVaR/chance-gate plans.

---------------------------------------------------------------------------------------------------
## 1. The model and the oracle (M1, M2)

### 1.1 Oracle design (PQ/DESIGN_M2.md; code QD/oracle.py)
- Demand registers d_k, p_k for k=1..L, b qubits each (levels 0..V-1, V=2^b), prepared by StatePreparation(sqrt pmf) on each register; resource formulas use the analytic 2^b - 1 rotations of a uniformly controlled Ry tree (DESIGN_M2 l28-30).
- Load register c = ceil(log2(L(V-1)+1)) = bit_length(L(V-1)) qubits. Every intermediate load lies in [0, L(V-1)] since L_{k-1} - d_{i_k} = sum_{l>k} d + sum_{l<k} p >= 0, so no overflow (l12-14).
- Counter m = ceil(log2(L+2)) = bit_length(L+1) qubits (number of stages with load > Q; garbage). Objective qubit obj = [count != 0] = [F > Q].
- Other registers: pad (c-b, always 0), carry 1, cmp 1, cmp_anc (c-1). All restored except load, count (left as garbage: AE needs only a unitary A with marked objective; Grover iterate uses A, A^dagger).
- Circuit A = U_eval . P: (1) P on all demand registers; (2) load += d_k for k=1..L (Cuccaro adder adder_ripple_c04, "fixed"); (3) stage 0: cmp <- [load >= Q+1] (synth_integer_comparator_2s), count += cmp (controlled increment, MCX ladder), un-compare; (4) for k=1..L: load -= d_k (inverse adder), load += p_k, compare, controlled increment, un-compare; (5) obj <- [count != 0].
- Degenerate thresholds: Q+1 > L(V-1) gives objective identically 0; Q<0 identically 1. Non-integer Q is used as floor(Q) (DESIGN_M2 l41-42, l73).
- Qubits (DESIGN_M2 l44; tested in tests/test_quantum_oracle.py::test_resource_counts_single_route): N_qubits = 2bL + c + (c-b) + 1 + 1 + (c-1) + m + 1.
- Toffoli count (DESIGN_M2 l46-55, exact, verified component by component against the transpiled circuit):
  - Cuccaro adder: 2c per application, 3L applications.
  - Comparator with a constant: 2c-3 per application, 2(L+1) applications.
  - Controlled increment of the m-bit counter: 1 + 2(m-1)(m-2) per stage, L+1 stages.
  - Objective [count != 0]: one m-control MCX, 4(m-2).
  - **T(L,c,m) = 6cL + 2(L+1)(2c-3) + (L+1)(1 + 2(m-1)(m-2)) + 4(m-2), valid for m >= 3 (and c >= 2).**
  - MCX with k >= 3 controls is a Toffoli-only dirty-ancilla V-chain (Barenco et al. 1995, Lemma 7.2), 4(k-2) Toffolis, borrowing k-2 qubits from cmp_anc and load; uses no extra qubits.
  - Small cases (QD/oracle.py docstring of resource_counts; tests): m=2 (L=1 or 2): formula short by 1, ccx = 6cL + 2(L+1)(2c-3) + (L+1) + 1 (checked at (L,b)=(2,1): 34). c=1 (L=1,V=2): ccx = 9 (checked).
- Asymptotics: PQ/THEORY_CERT.md l25 states O(L_j log^2 L_j) Toffolis (independent demands). It is a reading of the formula (counter-increment term), not a separate proof.
- Resource numbers given in the files (verified against transpiled circuit):
  | (L, V) | b | Toffolis | source |
  |---|---|---|---|
  | (5, 2) | 1 | 160 | DESIGN_M2 l46-47; test (5,1) |
  | (10, 8) | 3 | 813 | same; test (10,3) |
  | (20, 8) | 3 | 2043 | same; test (20,3) |
  | (2, 2) | 1 | 34 | oracle.py docstring (m=2 case) |
  | (1, 2) | 1 | 9 | oracle.py docstring (c=1 case) |
  | (8, 256) | 8 | 995 per call | papers/rarecert/submission_ijoc/rarecert_ijoc.tex l947 ("compiled circuit"); outside PQ |
  COMPUTED-HERE (formula arithmetic, not in the files): the formula reproduces 160, 813, 2043 and 995 (c=11, m=4 for (8, b=8)). Qubit counts from the qubit formula: (5,b=1) 23; (10,b=3) 84; (20,b=3) 148; (8,b=8) 159. Depth, CX count, T-count and physical resources: MISSING (resource_counts reports cx, x, depth but no values are recorded in any summary file).
- Not built: an oracle with the copula factor register (design only: PQ/DESIGN_M3.md l133-143, l198-199 "about 63 rotations" for a 6-qubit factor register with arbitrary weights; Hadamard-only equal-probability bins in l173-186). QD/oracle.py is for independent demands. Resource count of the copula oracle: MISSING. Resource count of the difference oracle: MISSING (only qubit sizing rule in DESIGN_M2 l67-71).

### 1.2 Difference oracle (coherent CRN) (DESIGN_M2 l62-73)
- Base route r and moved route r' (customer j inserted at pos) share the demand registers of r U {j}; steps 2-5 are run twice into (load_A,count_A,obj_A) and (load_B,count_B,obj_B), reusing pad, carry, cmp, cmp_anc; D <- obj_A xor obj_B.
- Sizes: c = bit_length((L+1)(V-1)) for both; m_A = bit_length(L+1), m_B = bit_length(L+2). Both necessary: a smaller m_B fails on 44 of 64 inputs in a mutation test.
- Position convention: pos = 1-indexed position of the inserted customer in the new route (oracle.py).

### 1.3 Correctness checks done (PQ/DESIGN_M2 l75-90; tests/test_quantum_oracle.py)
- Exhaustive basis inputs (classical simulation of the reversible circuit): peak oracle for (L,b,Q) in {(1,1,[-1,0,1]), (2,1,[0,1,2]), (2,2,[0,2,3,5,6])}; difference oracle V=2, base route [0], j=1, pos in {1,2}, Q in {0,1}. Random basis registers for several (L,V) and routes.
- Superposition: P(obj=1) of A equals classical.chance to 1e-9 (statevector) for 7 parametrised cases (L<=3, b<=2); P(D=1) equals classical.insertion_window.
- Enumeration through the classical permutation U_eval agrees with the exact DP at 31 qubits (V=4, L=4), 29 qubits (difference, V=2, base length 3) and 40 qubits (difference, V=4); statevector only below 23 qubits (cross-check).
- Qiskit IAE / MLAE on A within the confidence interval of the exact value (test_amplitude_estimation_close_to_exact).
- Toffoli formula checked at (L,b) = (5,1),(10,3),(20,3); qubit formula at (2,1),(3,2),(5,2),(10,3),(20,3).
- Grover-rotation model P(good after k iterates) = sin^2((2k+1) theta) validated on four real oracles (three peak, one difference; 19 to 22 qubits), k=0..6: max |p_circuit - p_model| = 6.11e-15 (RQ/m3b_v3_summary.md Sec. 1; "6e-15" in PQ/DESIGN_M3.md l162-163).
- Query accounting: num_oracle_queries counts applications of A: a round of k Grover iterates costs 2k+1 applications per shot (DESIGN_M2 l88-90). Canonical AE with M iterates costs 2M+1.
- Exact classical reference (M1): backward recursion on (A_k, G_k) gives the exact pmf of F for independent discrete demands, O(V M^2) per stop (QD/classical.py docstring). M1 had 112 tests (PQ/STATUS.md l21). Whole repo ~220 tests (CLAUDE.md).

### 1.4 Discretisation (M3a v3, RQ/m3a_v3_summary.md Sec. 4a; PQ/DESIGN_M3.md l187-189, l208)
- Unit u = Q/(K+1/2) with K the smallest integer with K+1/2 >= Q(V-1)/q_{0.999}^{max}; Q_level=K; nearest-level binning. Paired check, 105 routes (same continuous copula scenarios, N=200000):
  | rule | b | median abs bias | max abs bias | median abs rel bias (p>=0.01) | max abs rel bias |
  |---|---|---|---|---|---|
  | nearest | 4 | 2.570e-03 | 3.728e-02 | 0.0293 | 0.48 |
  | nearest | 5 | 5.150e-04 | 1.184e-02 | 0.00587 | 0.0842 |
  | nearest | 6 | 1.150e-04 | 6.710e-03 | 0.00117 | 0.0348 |
  Rule (a choice): median |rel bias| <= 1% and max <= 5%. Met at b=6 (median 0.12%, max 3.48%). At b=5 median meets (0.59%) but max does not (8.42%; two routes above 5%, L=5 and L=8). Worst cases are short routes with few large demands. At b=6 the largest absolute bias is 6.71e-03 (not below 1e-3 absolute on every route). Cost of b=6: 6 qubits per customer per demand type (12 per customer) vs 5 (10).
- Factor-grid study (RQ/m3a_v3_summary.md Sec. 4b-4c, nine routes, b=5): Gauss-Hermite (GH) stable from 32 nodes on short low-probability routes, not converged at 64 on long routes (GH64-GH128 7.7e-4 on CMT11X, 3.9e-4 on HCMC-100-3). Bin-averaged equal-probability bins (EMa) converge on long routes (64 bins within 5.6e-5 of EMa128; f=6 recommended for L>=16) and not on short ones (SCA3-1: EMa64 is 9.8e-4 from EMa128, still moving). Largest gap GH128 vs EMa128 over the nine routes: 2.68e-04 (SCA3-1). The "factor-grid error budget" of 2.6e-4 (DESIGN_M3 l205-207) and "~3e-4" (STATUS l23) is this b=5 number. P(D=+-1) deviations across grids against the references: 3.5e-4 to 4.6e-4 on the short route, <= 3.6e-4 medium, <= 2.8e-4 long (Table 4c-1), Monte Carlo SE 5e-6 to 8e-5. CAUTION: see Section 7 item 16 (b=6 table shows larger disagreement).

---------------------------------------------------------------------------------------------------
## 2. Test 1: coherent common random numbers / move comparison (M3)

### 2.1 Insertion lemma and what is proved
- Statement (PQ/PROJECT.md l81-85; QD/classical.py docstring): inserting customer j anywhere in r gives, scenario by scenario, F_r <= F_{r'} <= F_r + max(d_j, p_j). Hence D = 1[F_{r'}>Q] - 1[F_r>Q] in {0,1} and {D=1} = {F_r <= Q < F_{r'}}, P(D=1) = p' - p = Delta (exact from two DP calls), the mass of F_r in a window of width max(d_j,p_j) below Q (up to the exact event). Removal gives D in {-1,0}; intra-route moves give signed D (two amplitude estimates).
- Status: PROJECT.md l79 labels it "(to prove)" with a one-line sketch (stages before j carry d_j extra, new stage and later carry p_j extra). No written proof in any file. Checked numerically: assertion Delta >= -1e-12 on 648 036 moves (first run, DESIGN_M3 l61-62: "no lemma violation"), and used as scenario-wise identity in M3a v2/v3 (D in {0,1} for class 1). Treat as: sketch + numerical check; the paper must supply the proof (it is elementary: all added loads are non-negative and at most max(d_j,p_j)).
- Consequence: rho_q = sqrt(Delta(1-Delta))/sqrt(p'(1-p')) ~ sqrt(1 - p/p') for insertions; an insertion multiplies the risk (median p/p' = 0.10 in the first run, DESIGN_M3 l66-68), so the base carries little of the risk: structural reason for no CRN gain on insertions.
- End-load lemma (RQ/m3a_v3_summary.md Sec. 2; PQ/DESIGN_M3.md l166-171, l194-195): for any reordering r' of the same customers L_0 = sum d and L_L = sum p are order-invariant; with E = {sum d <= Q and sum p <= Q}, outside E both indicators equal 1 so D = 0; on E, D != 0 forces max(F_r,F_r') > Q. So P(D != 0) <= P(E and (F_r > Q or F_r' > Q)). Covers intra-route moves and alternative-position moves. Status: proof given in two sentences in the summary (an identity); DESIGN_M3 l166 says "to be written up". Numerical consistency check: 211 814 intra moves over 3 012 routes (all 3 049 routes evaluated); scenarios with D != 0 and an end load above Q: 0; routes with B_union below their largest observed P(D != 0): 0. Pooled explained share of Var=0 moves: all 0.959 (CON 0.953, SCA 0.954, CMT 0.932, city 0.997). Observed Var=0 share of intra moves: CON 0.092, SCA 0.104, CMT 0.503, city 0.549, all 0.279. B_spec is a diagnostic, not a valid bound; B_union is the valid lemma bound.

### 2.2 Experiment definition (RQ/m3a_v3_summary.md header; PQ/DESIGN_M3.md l98-171)
- Scenarios N = 200 000 per instance, seed 7, rho = 0.6, continuous BATON copula (no discretisation for the variance ratios); identical routes under several gates evaluated once. 73 instances.
- Move classes: (1) inter-route insertion (negative baseline), (2) intra-route relocate/swap/2-opt (all moves if |r|<=12 else 200 random), (3) alternative positions of the same inserted customer.
- Definitions: D = 1[F'>Q] - 1[F>Q]; rho_q = sqrt(Var D)/sqrt(p'(1-p')); rho_c = Var D/(p'(1-p')). Zero in sample = upper bound: P(D != 0) <= 3/N = 1.5e-05 (95%).
- Query model: canonical AE with M Grover iterates, error <= 2 pi sqrt(a(1-a))/M + pi^2/M^2 w.p. >= 8/pi^2, cost 2M+1; M = smallest power of two meeting eps; confidence ~95% by median of R=5 repetitions (assumption; common to all variants). Classical N = max(z^2 v/eps^2, 3/eps), z=1.96 (3/eps = rule-of-three certificate floor). Signed D: quantum two estimates (P(D=1), P(D=-1)) at eps/2 each; classical direct signed mean (one estimate). eps absolute (1e-3, 1e-4) or relative 0.1 p'.
- Population: 1 369 458 moves in M3a v2 (RQ/m3b_v3_summary.md Sec. 2). First run (superseded): 648 036 insertion moves, independent demands, b=4.

### 2.3 Measured rho_q (median, in sample; p' in (0.02, 0.5)) (RQ/m3a_v3_summary.md Sec. 3a)
| class | family | moves | median rho_q | median rho_c | rho_q upper bound (zeros at 3/N) | Var=0 share |
|---|---|---|---|---|---|---|
| insertion | CON | 152301 | 0.917 | 0.841 | 0.917 | 0.000 |
| insertion | SCA | 149612 | 0.92 | 0.846 | 0.92 | 0.000 |
| insertion | CMT | 6249 | 0.394 | 0.155 | 0.394 | 0.008 |
| insertion | city | 51882 | 0.674 | 0.455 | 0.674 | 0.000 |
| intra | CON | 36355 | 0.133 | 0.0176 | 0.133 | 0.023 |
| intra | SCA | 36929 | 0.136 | 0.0186 | 0.136 | 0.021 |
| intra | CMT | 12603 | 0 | 0 | 0.00798 | 0.566 |
| intra | city | 65397 | 0 | 0 | 0.0205 | 0.526 |
| alt-position | CON | 152296 | 0.161 | 0.026 | 0.161 | 0.019 |
| alt-position | SCA | 149608 | 0.162 | 0.0262 | 0.162 | 0.018 |
| alt-position | CMT | 6263 | 0 | 0 | 0.0111 | 0.510 |
| alt-position | city | 51874 | 0 | 0 | 0.0158 | 0.634 |
- Alternative positions by distance |pos2-pos| (Sec. 5, decision band p' in [0.02,0.2]): CON median rho_q 0.111 (dist 1), 0.162 (2-3), 0.15 (>3); SCA 0.11, 0.166, 0.152; CMT 0, 0, 0.00615; city 0, 0, 0.
- Within-paradigm CRN gain, absolute eps=1e-3 (Sec. 3b): quantum Nq/Nq_sa: insertion CON 0.864, SCA 0.861, CMT 0.429, city 0.674; intra CON 0.441, SCA 0.451, CMT 0.261, city 0.199; alt CON 0.518, SCA 0.523, CMT 0.274, city 0.206. Classical Nc/Nc_sa: insertion 0.769, 0.766, 0.185, 0.406; intra 0.047, 0.047, 0.028, 0.007; alt 0.059, 0.060, 0.031, 0.008.

### 2.4 Certified query-count ratios quantum/classical (sums over moves; RQ/m3a_v3_summary.md Sec. 3b)
Nq = applications of A; Nc = classical scenario evaluations. Units differ (one difference-oracle
application costs more Toffolis than a stand-alone one; the tables do not convert to Toffolis). Ratio
< 1 means fewer quantum queries, not shorter runtime.
| class | family | absolute eps=1e-3 | absolute eps=1e-4 | relative eps=0.1 p' |
|---|---|---|---|---|
| insertion | CON | 0.0691 | 0.00635 | 0.815 |
| insertion | SCA | 0.069 | 0.00631 | 0.818 |
| insertion | CMT | 0.108 | 0.0109 | 2.14 |
| insertion | city | 0.111 | 0.0109 | 1.06 |
| intra | CON | 0.709 | 0.0643 | 4.24 |
| intra | SCA | 0.716 | 0.0644 | 4.2 |
| intra | CMT | 0.421 | 0.0346 | 7.23 |
| intra | city | 1.71 | 0.306 | 3.44 |
| alt-position | CON | 0.544 | 0.0501 | 5.09 |
| alt-position | SCA | 0.533 | 0.0492 | 5.06 |
| alt-position | CMT | 0.415 | 0.0345 | 7.27 |
| alt-position | city | 1.74 | 0.5 | 3.79 |
- Text of the summary: at 1e-3 insertions 0.069 to 0.11, intra 0.42 to 1.7, alt 0.42 to 1.7; at 1e-4 insertions 0.0063 to 0.011, intra 0.035 to 0.31, alt 0.034 to 0.5; relative block: signed-move cells 3.44 to 7.27, insertions 0.81 to 2.1: classical direct estimator is cheaper at relative precision. Relative block "answers a different question; quote only with that label"; absolute blocks are the main comparison.
- City cells at 1e-3 are floor-dominated: a move with no event costs 2R(2M+1) = 5130 applications (R=5, M=256, two estimates) vs the classical floor 3/eps = 3000; 5130/3000 = 1.71. At 1e-4 the same count is 20490 vs 30000 (0.68). "Comparable cost at 1e-3 (floor constants set the ratio), lower quantum count at 1e-4 in every class and family."
- Single-estimate convention for one-sided moves (Sec. 3c, 1e-3): intra CON 0.653, SCA 0.662, CMT 0.307, city 1.38; alt CON 0.511, SCA 0.5, CMT 0.257, city 1.43 (main-convention values in table above).
- Conservativeness of the quantum count (Sec. 3b reading): on the exact QPE distribution the 81% quantile of the actual error was below the bound by factors 1.6, 2.0 and 4.6 in three non-aligned cases checked; "no larger factor is claimed".

### 2.5 Estimator behaviour: MLAE / QAE slopes (M3b v3, RQ/m3b_v3_summary.md)
- Setting: exact Grover-rotation model P(good after k iterates)=sin^2((2k+1)theta), validated on 4 real circuits (6.11e-15). 4000 seeds per sampled setting (QAE from exact outcome pmf, m=3..22; MLAE J=1..10 with S in {32,64,128} shots; classical N = 2^4..2^26). Sample: 445 moves (374 one-sided, 71 signed), 901 unique amplitudes, 27 cells (class x p' band x family group), up to 20 moves per cell, 6 cells short. Eligibility: nonzero amplitudes need >=200 hits at N=2e5; excluded 149 236 zero-difference moves and 529 674 moves with a nonzero amplitude below 200 hits; eligible 690 548 of 1 369 458.
- Slopes of the 90% quantile of |a_hat - a| (log-log, last six settings), median over records: standalone qae -1.007, mlae_S32 -0.996, mlae_S64 -1.006, mlae_S128 -1.005, classical -0.500; difference+ qae -1.007, S64 -0.993, classical -0.500; difference- qae -0.913, S64 -0.975. Same in every family group (CON/SCA, CMT, city: quantum about -1, classical -0.50).
- RMS slope (secondary): standalone qae -0.492 (canonical QAE has heavy tails, P(|err|>e) ~ 1/(N e); RMS scales N^(-1/2)), mlae_S64 -0.980, S128 -1.000. Hence "slope -1" holds for the 90% quantile and mean absolute error (qae -0.928) and for MLAE with S>=64, not for the RMS of canonical QAE.
- MLAE S=1 does not converge (median 90% error 0.51 at J=10; only 17% of records reach 1e-3); S=8 too slow for 1e-4 (19% reach 1e-4; S=64: 98%) (Sec. 7, from the v2 run, 183 records).
- Cost ratio difference/stand-alone (same paradigm; NOT quantum vs classical), Sec. 6 "Supported": absolute 1e-3: QAE below 1 in 23 of 23 cells (range 0.0828 to 0.877); MLAE S=64 23 of 23 (0.0687 to 0.928). Absolute 1e-4: QAE below 1 in 22 of 23 cells (0.0898 to 1.24). Relative target r=0.1: QAE difference costs more than stand-alone in all 23 cells (1.05 to 22.3); classical direct in all 23 (1.17 to 450); MLAE S=64 in all 14 cells with n_ok>=10 (1.03 to 14.1; 9 cells n_ok<10).
- sqrt(v) law tested on class 1: ratio/rho_q for MLAE S=64 and 128 between 0.94 and 1.11 (abs targets); QAE alignment scatter 0.72 to 1.29. Classes 2 and 3 are NOT tested against rho_q. Classical ratio/rho_c 0.97 to 1.04 (consistency check only).
- Not supported by M3b: regime P(D != 0) < 1e-3 (excluded by eligibility; 0.38 to 0.95 of class 2/3 moves are in it), city classes 2 and 3 except the high band, any quantum-vs-classical cost comparison (not computed in M3b), cells with n_ok < 10.
- Median pooled group tables in Sec. 6 are "not results" (mix classes and bands).

### 2.6 Discretisation / level b error budget: see 1.4.

---------------------------------------------------------------------------------------------------
## 3. Test 2: search inside the optimiser (M4)

### 3.1 Problem and theorems (PQ/THEORY_M4.md, revision 2)
- eps-CF (eps-cheapest feasible): candidates sorted by exact deterministic cost; k* = min{j: p_j <= alpha - eps}; output j correct if p_j <= alpha + eps and j <= k* ("none" correct if k*=infinity); 0 < eps <= alpha/2. Access model: select oracle U: |j>|0> -> |j>A_j|0>, cost unit one application of U or U^dagger; classically one sample of one candidate. v-bar: known bound p_j(1-p_j) <= v-bar for all j with p_j <= 2 alpha.
- Lemma 1 (PROVED, standard): threshold test B_j with O(T log(1/eta)) applications, T(eps,v) = sqrt(v)/eps + 1/sqrt(eps) + 1/sqrt(alpha); median of canonical AE runs computed coherently (reversible arithmetic), deferred measurement. CRN version: estimate p_rho once to eps/2 (failure delta/4), difference to eps/4 each; total error eps/2 + eps/4 + eps/4.
- Lemma 2 (PROVED): prefix test TEST(m), K_m = ceil(sqrt(2m)), R = ceil(ln(1/delta')/ln(4/3)) runs of two attempts each (0 iterates, then k uniform in {0..K_m-1}); eta <= delta'/(R(1+(2K_m-1)^2)). (a) completeness: YES w.p. >= 1-delta' if prefix holds an index with p_j <= alpha-eps; (b) soundness: P(YES with witness p_w > alpha+eps) <= delta' via |sin nx| <= n |sin x|. Uses Boyer-Brassard-Hoyer-Tapp Lemma 2.
- Algorithm FIRST-FEASIBLE: exponential phase m=1,2,4,... then M, bisection between last NO and first YES.
- **Theorem 1 (PROVED)**: with delta' = delta/(4 ceil(log2 M)+4), L = log(log M/delta), T as above, FIRST-FEASIBLE is a correct eps-CF w.p. >= 1-delta using O( sqrt(k*) log k* . L . log(k* L/delta) . T ) applications of U if k* < infinity and O( sqrt(M) . L . log(M L/delta) . T ) if k* = infinity. Constants in the O are not explicit in the theorem (explicit constants only in the cost model QD/search.py). CRN: T becomes sqrt(v_D)/eps + 1/sqrt(eps) + 1/sqrt(alpha), plus one stand-alone estimate of p_rho. The log k* factor comes from the bisection and is not optimal (Remark 1).
- Prop. 2 (PROVED), classical scan: N = ceil((2v + 2eps/3) ln(2k/delta)/eps^2) samples per candidate; cost <= min(k*, M) . N = O(k*(v/eps^2 + 1/eps) log(M/delta)).
- Prop. 3 lower bounds (l178-191): (a) classical Omega(k alpha(1-alpha)/eps^2): **PROVED** (change of measure, Kaufmann-Cappe-Garivier Lemma 1 transportation lemma), for k >= 16, constant 1/1440 (N* >= k alpha(1-alpha)/(1440 eps^2)). (b) quantum Omega(sqrt(k) + sqrt(alpha)/eps): **PROVED** (BBBV unique-marked-element for sqrt(k); direct hybrid argument with T >= sqrt(alpha)/(9 eps) for the second term). (c) Omega(sqrt(k) sqrt(alpha(1-alpha))/eps) (product bound; would make Theorem 1 optimal up to logs): **CONJECTURE** (no composition theorem for OR_k with a promise inner function; Reichardt 2011 and Lee-Mittal-Reichardt-Spalek-Szegedy 2011 are stated for total Boolean functions).
- Remark 3: variable-time version O~(sqrt(sum_{j<=k*} T_j^2)) (Ambainis 2010) is a CONJECTURE for the ambiguous predicate. Remark 1: find-first in O(sqrt p) expected queries is only a remark in Kothari 2014 Sec. 2.2; bounded-error search without log loss (Hoyer-Mosca-de Wolf 2003) covers total predicates only.
- Corollary 4 (no status label): ratio upper/lower bound <= O~(eps/sqrt(k* alpha(1-alpha))) = O~(1/sqrt(N_cl)), log factors differ; "consistent with a quadratic advantage".

### 3.2 Repair-step measurements (RQ/m4b_rr_summary.md; RQ/m4b_rr_region_summary.md)
- Mode rr (ruin-and-recreate), alpha = 0.1, DELTA = 0.05, V_WC = 0.25, ratios = medians of per-step Nq/Nc. Code defaults (scripts/run_quasar_m4b.py): N_RR = 50 000 common scenarios, ROUNDS = 2 per (instance, gate), Q_FRAC = 0.2 of customers removed per round, AE_REPS = 5.
- Steps: 4 392 over 73 instances; 31 840 over 148 region instances (total 36 232).
- Columns: k0 = share of steps with k* = infinity; k1 = share with k* = 1 (cheapest position already feasible, no search needed); k_q50/k_q90/k_max = quantiles of k*; band_med = 0 in every group (the (alpha-eps, alpha+eps] band is empty at median). "full wc" = explicit constants of Lemma 2 with the worst-case variance proxy V_WC=0.25; "full opt" = same with the oracle variance bound v_opt (favours quantum); "core" = sqrt(k*) (2M+1) vs k* max(v/eps^2, 3/eps); "crn core" = CRN version. Test schedule: ideal test (YES iff m >= s) in QD/search.py ideal_schedule.
- Stop index s (eps-free; first p <= alpha) (m4b_rr_summary l7-22; region l6-17):
  | set | gate | steps | s=0 | s_q50 | s_q90 | s_max | M_med |
  |---|---|---|---|---|---|---|---|
  | 73-set | Det | 2512 | 0.0943 | 5 | 30 | 226 | 95 |
  | 73-set | SAA | 1880 | 0.00691 | 1 | 3 | 41 | 93 |
  | 73-set | ALL | 4392 | 0.0569 | 1 | 22 | 226 | 94 |
  | region | Det | 15920 | 0.0718 | 16 | 122 | 446 | 380 |
  | region | SAA | 15920 | 0 | 1 | 5 | 218 | 396 |
  | region | ALL | 31840 | 0.0359 | 2 | 72 | 446 | 388 |
  Region family rows: VNN Det s_q50 10, s_q90 50, s_max 315; GD Det 17, 125, 446; VNN/GD SAA 1/4, 1/5.
- Per eps (all steps of the set; ALL gates):
  | set | eps | gate | k1 | k_q50 | k_q90 | k_max | full wc | full opt | core | crn core | share Nq<Nc |
  |---|---|---|---|---|---|---|---|---|---|---|---|
  | 73-set | 0.01 | Det | 0.147 | 5 | 32 | 226 | 287 | 306 | 0.162 | 0.51 | 0 |
  | 73-set | 0.01 | SAA | 0.812 | 1 | 3 | 45 | 313 | 457 | 0.626 | 0.642 | 0 |
  | 73-set | 0.01 | ALL | 0.432 | 2 | 23 | 226 | 294 | 413 | 0.361 | 0.569 | 0 |
  | 73-set | 0.001 | Det | 0.17 | 5 | 30.9 | 226 | 23.2 | 35.4 | 0.0203 | 0.0546 | 0 |
  | 73-set | 0.001 | SAA | 0.832 | 1 | 3 | 41 | 25.3 | 35.4 | 0.0459 | 0.124 | 0 |
  | 73-set | 0.001 | ALL | 0.454 | 1 | 22 | 226 | 23.8 | 35.4 | 0.0365 | 0.073 | 0 |
  | region | 0.01 | Det | 0.0873 | 19 | 131 | 446 | 248 | 227 | 0.081 | 0.289 | 0 |
  | region | 0.01 | SAA | 0.734 | 1 | 6 | 266 | 337 | 501 | 0.626 | 0.594 | 0 |
  | region | 0.01 | ALL | 0.411 | 2 | 81 | 446 | 318 | 394 | 0.265 | 0.429 | 0 |
  | region | 0.001 | Det | 0.105 | 16 | 123 | 446 | 20 | 31.8 | 0.0113 | 0.0276 | 0 |
  | region | 0.001 | SAA | 0.755 | 1 | 5 | 218 | 27.3 | 38.1 | 0.0459 | 0.114 | 0 |
  | region | 0.001 | ALL | 0.43 | 2 | 73 | 446 | 25.7 | 38.1 | 0.0322 | 0.0617 | 0 |
  (eps=0.003 rows exist in both files: e.g. region ALL ALL core 0.0942, crn core 0.17, full wc 115, full opt 146.) k0 (k*=infinity share) for ALL: 73-set 0.0656 (eps .01), region 0.0405.
- Ranges over all groups/eps: core ratio 0.011 (region GD Det eps 0.001: 0.011) to 0.626; full (constants included) ratio medians range 19.7 (region GD Det, eps 0.001, wc) to 337 (wc) and 31.6 to 501 (opt); share of steps with Nq < Nc is 0 in every group, for wc and opt, at every eps. The summaries show no row where the constants-included quantum cost is below the classical.
- Rank structure: SAA plans: cheapest position is feasible in k1 = 0.734 to 0.832 of steps (region 0.734 to 0.755; 73-set 0.812 to 0.832); Det plans k_q90 grows with scale: 32 (73-set) / 51 (VNN) / 134 (GD) at eps 0.01 (VNN 51 and GD 134 from the family rows at eps 0.01). Chosen rank (median): 73-set Det 5, SAA 1; region Det 16, SAA 1; "new route share" Det 0.0943 / 0.0718, SAA 0.00691 / 0.
- Open (PQ/STATUS.md l24): a search with smaller constants (expected O(sqrt(k*)) find-first without per-bisection re-testing), then re-measure. Not done.

---------------------------------------------------------------------------------------------------
## 4. Test 3: simultaneous certification of all routes of a region plan (black-box estimation)

### 4.1 Cost models (PQ/THEORY_CERT.md l37-48)
- Model L (local): one application of A_j or A_j^dagger costs L_j units; one classical sample of route j costs L_j; a classical sample of the whole plan costs sum_j L_j = n and yields all d indicators.
- Model S (black box): only a whole-day simulator; one call (classical sample, or one application of a state preparation producing all d indicators coherently) costs 1.
- Model L is the relevant one for routing (locality: X_j depends only on route j's customers and the copula factors); S is the setting of Huggins et al. 2022 and Tang 2025.

### 4.2 Black-box (model S) measurements (RQ/cert_crossover_v0.txt; PQ/REFRAME_M4.md l64-111)
Route risks from 20 000 scenarios per instance, delta=0.05. Classical = shared samples, Hoeffding (H)
or Bernstein (B, v = max route variance); quantum = Huggins et al. (2022) Thm 4 form 2 sqrt(d)
log(d/delta)/eps (variance-agnostic, c=2).
| plan | d | p_med | p_max | v_max | eps | classical_H | classical_B | quantum | ratio_H | ratio_B |
|---|---|---|---|---|---|---|---|---|---|---|
| VNN Det | 257 | 0.568 | 0.707 | 0.250 | 0.01 | 4.62e+04 | 4.68e+04 | 2.74e+04 | 0.593 | 0.585 |
| VNN Det | 257 | | | | 0.001 | 4.62e+06 | 4.63e+06 | 2.74e+05 | 0.0593 | 0.0592 |
| VNN Det | 257 | | | | 0.0001 | 4.62e+08 | 4.62e+08 | 2.74e+06 | 0.00593 | 0.00593 |
| VNN SAA | 375 | 0.022 | 0.047 | 0.045 | 0.01 | 4.81e+04 | 9.33e+03 | 3.46e+04 | 0.719 | 3.7 |
| VNN SAA | 375 | | | | 0.001 | 4.81e+06 | 8.76e+05 | 3.46e+05 | 0.0719 | 0.395 |
| VNN SAA | 375 | | | | 0.0001 | 4.81e+08 | 8.7e+07 | 3.46e+06 | 0.00719 | 0.0397 |
| GD Det | 4019 | 0.412 | 0.476 | 0.249 | 0.01 | 5.99e+04 | 6.06e+04 | 1.43e+05 | 2.39 | 2.36 |
| GD Det | 4019 | | | | 0.001 | 5.99e+06 | 5.99e+06 | 1.43e+06 | 0.239 | 0.239 |
| GD Det | 4019 | | | | 0.0001 | 5.99e+08 | 5.98e+08 | 1.43e+07 | 0.0239 | 0.0239 |
| GD SAA | 5867 | 0.024 | 0.052 | 0.049 | 0.01 | 6.18e+04 | 1.29e+04 | 1.79e+05 | 2.89 | 13.8 |
| GD SAA | 5867 | | | | 0.001 | 6.18e+06 | 1.22e+06 | 1.79e+06 | 0.289 | 1.47 |
| GD SAA | 5867 | | | | 0.0001 | 6.18e+08 | 1.21e+08 | 1.79e+07 | 0.0289 | 0.148 |
- REFRAME l78-84: a larger network HURTS the simultaneous quantum estimator (sqrt d vs log d classically); crossover roughly sqrt(d) eps < v; for SAA plans (v ~ 0.05) quantum needs eps <~ 1e-3 on VNN and eps <~ 5e-4 on GD ("roughly", heuristic).

### 4.3 Variance-aware bound (L. Tang, arXiv 2504.06940v3; PQ/REFRAME_M4.md l92-111)
- Thm I.2: l_infty error sqrt(tr Sigma)/n with O(n log(d/delta)) calls of a state preparation returning the whole d-vector in one call; for indicators tr Sigma = sum_j p_j(1-p_j) exactly, whatever the correlations; cost O(sqrt(sum_j v_j) log(d/delta)/eps).
- Status of constants: NOT GIVEN (hidden O, an unspecified D). Needs a known bound on tr Sigma and on
  ||E X||_2. Preprint only; optimality for l_infty conjectured. The table below ASSUMES C = 1 (optimistic)
  and ratio scales linearly in C, so crossover eps* scales as 1/C.
| plan | d | tr Sigma | ratio quantum(C=1)/classical Bernstein at eps = 1e-2 / 1e-3 / 1e-4 |
|---|---|---|---|
| VNN SAA | 375 | 7.5 | 0.26 / 0.028 / 0.0028 |
| GD SAA | 5 867 | 124 | 1.0 / 0.11 / 0.011 |
| VNN Det | 257 | 51 | 0.13 / 0.013 / 0.0013 |
| GD Det | 4 019 | 826 | 0.54 / 0.054 / 0.0054 |
  No result file or script output holds this table (only REFRAME_M4.md l101-111); the generating script was not located (scripts/quasar_cert_crossover_v0.py only produces cert_crossover_v0.txt).
- Other multivariate bounds: Huggins et al. 2022 (read in full: Thm 1, Thm 2 lower bound for eps < 1/(3 sqrt M), Thm 4 with c = 2); van Apeldoorn TQC 2021 and Cornelissen-Hamoudi-Jerbi STOC 2022 (abstract only; l2 not l_infty); no citable Omega(log d/eps^2) classical shared-sample lower bound found (candidate Wainwright 2019) (PQ/VERIFY_CITATIONS.md l71-80).

### 4.4 Model L plan-level measurement, Q-CERT vs C-CERT (RQ/cert_region_summary.md)
alpha = 0.1, delta = 0.05, eps in {0.01, 0.003, 0.001, 0.0001}, 148 instances, 10 518 routes. Groups =
(region family, gate), d = routes in the group. Q = sum_j L_j q_cost_j (applications of A, A^dagger);
Q_ideal = sum_j L_j r_t (2M_t+1) for the stopping round alone (no doubling, no earlier rounds);
C = sum_j L_j n_j; C_shared = sum over instances of max_j n_j . n_total(instance) (one whole-plan
sample per instance, model L). k* = best fixed k per group and eps (oracle choice on the true risks,
favours quantum). Route risks from 20 000 common scenarios (SE up to ~0.0016 at p=0.05), so gaps below
about 3e-3 are not resolved. Run time 38.3 s.
| region | gate | eps | d | k* | Q | C | C_shared | Q/C | Q/C_shared | Q_ideal | Q_ideal/C | Q floor share | median q/c | 90% q/c | share q<c |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| VNN | Det | all four | 257 | 1 | 1.255e+07 | 1.699e+06 | 1.306e+07 | 7.382 | 0.961 | 6.477e+06 | 3.811 | 0.000 | 11.021 | 18.156 | 0.000 |
| VNN | SAA | all four | 375 | 1 | 3.284e+07 | 6.931e+06 | 9.251e+06 | 4.739 | 3.550 | 1.678e+07 | 2.421 | 0.000 | 4.833 | 6.610 | 0.000 |
| GD | Det | 0.01 | 4019 | 1 | 3.796e+08 | 1.045e+08 | 2.855e+09 | 3.634 | 0.133 | 1.938e+08 | 1.855 | 0.003 | 10.642 | 12.860 | 0.006 |
| GD | Det | 0.003 | 4019 | 1 | 4.072e+08 | 2.339e+08 | 1.081e+10 | 1.741 | 0.038 | 2.078e+08 | 0.889 | 0.000 | 10.642 | 12.860 | 0.006 |
| GD | Det | 0.001 | 4019 | 1 | 4.140e+08 | 4.013e+08 | 2.214e+10 | 1.032 | 0.019 | 2.112e+08 | 0.526 | 0.000 | 10.642 | 12.860 | 0.006 |
| GD | Det | 0.0001 | 4019 | 1 | 4.140e+08 | 4.013e+08 | 2.214e+10 | 1.032 | 0.019 | 2.112e+08 | 0.526 | 0.000 | 10.642 | 12.860 | 0.006 |
| GD | SAA | all four | 5867 | 1 | 6.319e+08 | 1.330e+08 | 1.892e+08 | 4.751 | 3.340 | 3.198e+08 | 2.404 | 0.000 | 4.877 | 6.445 | 0.000 |
("all four" = the four eps rows are identical in the file for that group.) Reading: Q/C is 4.7 to 7.4
on SAA and VNN Det; GD Det falls to 1.03 at eps <= 1e-3 (ideal-vs-ideal 0.526). Q/C_shared (quantum
against ONE whole-plan classical sample stream): 0.961 (VNN Det), 3.550 (VNN SAA), 0.133 / 0.038 /
0.019 / 0.019 (GD Det), 3.340 (GD SAA). The summary file does not discuss Q/C_shared; the quantum
side is below the shared-sample classical on Det plans because C_shared is set by the single hardest
route and multiplied by the whole-plan sample cost n_total.
- Algorithms (PQ/THEORY_CERT.md l82-144): Q-CERT per route, rounds t with M_t = 2^t, eta_t = 6 delta/ (pi^2 d (t+1)^2), median of r_t canonical AE runs, consistent interval I of Lemma 2, stop if sup I <= alpha (certify), inf I > alpha (reject) or h(v_I) <= eps/2 (precision floor). C-CERT: same doubling scheme with empirical Bernstein (Audibert-Munos-Szepesvari 2009; sqrt(2 v ln(3/eta)/n) + 3 ln(3/eta)/n); the experiments give the classical side the exact stopping sample size with no doubling loss (QD/cert.py c_cert_route: empirical Bernstein, L = ln(3d/delta)).
- Theorem 1 (THEORY_CERT l107-129, PROVED): Q-CERT is an eps-certification w.p. >= 1-delta (soundness p_j <= alpha + eps/2; completeness p_j <= alpha - eps => certified); route j stops no later than the first round with 2 h_{M_t,k_t}(v-bar_{g'_j}) < g'_j, g'_j = max(|p_j - alpha|, eps/2); cost O(log(d log(1/g_j)/delta) (sqrt(v-bar_j)/g_j + 1/sqrt(g_j))) per route; total in model L sum_j L_j C_j. Fact 1 (CITED, Brassard-Hoyer-Mosca-Tapp Thm 12): |p_hat - p| <= h_{M,k}(p) = 2 pi k sqrt(p(1-p))/M + pi^2 k^2/M^2 w.p. >= s_k, s_1 = 8/pi^2, s_k = 1 - 1/(2(k-1)) for k >= 2 (k=2 excluded: s_2 = 1/2). Lemma 1 (PROVED): median of r runs; Lemma 2 (PROVED): inversion interval is convex (concavity of sqrt(q(1-q)) minus |.|) and separates from alpha when 2h(v-bar_g) < g.
- Per-route lower bounds (Prop. 2, labelled PROVED, from THEORY_M4 Prop. 3, k = 1 case): Omega(alpha(1- alpha)/eps^2) classical, Omega(sqrt(alpha)/eps) quantum (alpha <= 1/4, eps <= alpha/2); per-route ratio Theta~(g_j/sqrt(v-bar_j)) for v-bar_j >= g_j. The direct-sum statement for d routes with separate oracles is a CONJECTURE (THEORY_CERT l155-159). See Section 7 on the k=1 caveat.
- Crossover statements available: (i) black-box: crossover roughly sqrt(d) eps < v (REFRAME l81); with Tang constant C=1 the table in 4.3; crossover eps* ~ 1/C. (ii) local model: quantum pays only for binding routes and for high-precision or rare-event targets (REFRAME l143-145). Measured d*(eps) crossover on the 148 instances: MISSING (the "crossover d*(eps)" promised in REFRAME l47-49 is not in any result file).

---------------------------------------------------------------------------------------------------
## 5. Test 4: per-route certification with threshold gap; Test 5: rare events vs mixture IS

### 5.1 Test 4 (same experiment as 4.4, per route; PQ/REFRAME_M4.md l123-145)
- 10 518 routes, alpha = 0.10, delta = 0.05, explicit constants (canonical AE + median + union bound) against the empirical-Bernstein classical size without doubling loss. Per-route ratio quantum/classical at eps = 1e-3 by gap |p - alpha| (table only in REFRAME_M4.md; source data RQ/cert_region_routes.csv; cert_region_summary.md does NOT reproduce it):
  | gap | routes | median ratio |
  |---|---:|---:|
  | <= 1e-3 | 1 | 0.09 |
  | 1e-3 - 3e-3 | 8 | 0.25 |
  | 3e-3 - 1e-2 | 21 | 0.84 |
  | 1e-2 - 3e-2 | 58 | 1.9 |
  | 3e-2 - 0.1 | 6 852 | 5.0 |
  | > 0.1 | 3 578 | 11.2 |
  Crossover gap "approximately 5e-3". 99% of the routes are far from the target (SAA plans at p ~ 0.02-0.05 against alpha = 0.10; Det plans far above): classical needs hundreds of samples and quantum constants (median under union bound, 1/sqrt(g) term) dominate. Plan-level totals: quantum/classical 4.7-7.4, except GD Det at eps <= 1e-3 (1.03; ideal-vs-ideal 0.53). Medians per group in the summary: 11.021 (VNN Det), 4.833 (VNN SAA), 10.642 (GD Det), 4.877 (GD SAA); share of routes with q < c: 0 except GD Det 0.006.
- Honest reading: 6 852 + 3 578 = 10 430 of 10 518 routes (99.2% by the table, COMPUTED-HERE) have gap > 0.03; the six rows below gap 0.03 total 88 routes; the 9 routes with gap < 3e-3 lie at or below the stated resolution of the p_j estimates (SE up to 0.0016).

### 5.2 Test 5: rare events (DESIGN_RARE.md; RQ/rare_summary.md; RQ/rare_fair_comparison.md)
- Question: certify route overflow p in {1e-3, 1e-4, 1e-5, 1e-6} to relative error eps_rel in {0.1, 0.03, 0.01}; 120 SAA-plan routes (L >= 4) from the region set. Simulator written in standard normal inputs x = (W_d, W_p, E_d,1..L, E_p,1..L) in R^(2L+2); threshold c set per route so P(F>c) = gamma. No classical method may use the exact DP (ground truth only).
- Costs (95% confidence; DESIGN_RARE l28-47): classical crude N = z^2 (1-p)/(p eps^2); classical IS N = z^2 RV/eps^2, RV = Var_q(wX)/p^2 measured; quantum crude: smallest M with 2 pi sqrt(p(1-p))/M + pi^2/M^2 <= eps p, cost r(2M+1), median of r runs (THEORY_CERT Fact 1), "ideal" (M known) and "doubling" (search for M); quantum with IS: crude cost with (1-p)/p replaced by RV (QD/rare.py aqe_is_cost; absolute error eps_rel on normalised Y/p, mean 1, variance RV).
- **"Idealised accounting"** (DESIGN_RARE l37-41; rare_summary l7): the quantum IS cost uses the crude canonical AE constants on the normalised weight variable with variance RV; it charges NO pilot (cross-entropy proposal design) cost to the quantum side, ignores reversible likelihood-ratio arithmetic and heavier-tailed weights, and (ideal variant) assumes M known. Optimistic for quantum. Classical IS costs are ideal too (true RV, no stopping rule) but "total" includes the cross-entropy pilot.
- Back-of-envelope: Q_IS/C_IS ~ 29 eps_rel/sqrt(RV) (M3 constants): wins only if RV > ~9 at eps_rel = 0.1.
- Ground truth: mixture IS (1 000 000 draws per (route, gamma)) vs exact DP (L <= 8, gamma >= 1e-4, Gauss-Hermite over the two factors): 178 cases, pass 178, max |rel diff| 0.0074, median relative difference +0.0013; tolerance max(4 rse_IS, 2%); rse_IS about 0.0019 to 0.0022. Single cross-entropy mean shift: off by up to 11-19% with understated errors (DESIGN_RARE l58-60); see Section 7 item 20.
- Classical mixture IS (cross-entropy shifts per stage load, defensive component): RV median 3.88 (gamma 1e-3), 4.61 (1e-4), 5.28 (1e-5), 5.87 (1e-6) [rare_fair table 3.88, 4.61, 5.28, 5.87]; pilot median 8e4, 9e4, 1e5, 1.2e5 draws (overall pilot median 120 000, range 60 000 to 3 900 000).
- Headline fair comparison (rare_fair_comparison.md): medians (q10-q90) over routes, ratios quantum/classical; quantum uses the doubling search.
  | gamma | eps_rel | Qcrude/Ccrude | Qcrude/CIS_total (incl. pilot) | Qcrude/CIS_sampling | QIS/CIS_sampling | share Qcrude<CIS_total |
  |---|---|---|---|---|---|---|
  | 1e-06 | 0.01 | 0.00327 | 380 (68-3.8e+02) | 598 | 0.962 (0.87-0.97) | 0.00 |
  | 1e-06 | 0.1 | 0.0194 | 61 (4.6-61) | 3.55e+03 | 10.9 | 0.00 |
  | 1e-05 | 0.01 | 0.00787 | 105 | 161 | 1.08 | 0.00 |
  | 1e-05 | 0.1 | 0.095 | 35.8 | 1.94e+03 | 12.2 | 0.00 |
  | 0.0001 | 0.01 | 0.0194 | 29.5 | 45.7 | 1.24 | 0.00 |
  | 0.0001 | 0.1 | 0.225 | 9.56 (0.63-11) | 531 | 14.1 | 0.28 |
  | 0.001 | 0.01 | 0.0951 | 17 | 27 | 1.5 | 0.00 |
  | 0.001 | 0.03 | 0.203 | 9.12 | 57.7 | 3.24 | 0.23 |
  | 0.001 | 0.1 | 0.527 | 2.49 (0.17-3.3) | 150 | 8.25 | 0.34 |
  (Full 12-row table in the file; 1e-06/0.03: 0.00709, 211, 1.3e+03, 2.08; 1e-05/0.03: 0.0353, 125, 722, 2.33; 1e-4/0.03: 0.0856, 34.1, 202, 2.68; 1e-4: share 0.00 at eps 0.01 and 0.03.) Ranges stated: quantum crude vs crude MC 0.003-0.5; quantum crude vs classical IS incl. pilot 2.5-380 times MORE (median); cheaper on 23-34% of routes (gamma 1e-3 eps 0.03, 0.1; gamma 1e-4 eps 0.1).
- Ideal quantum (M known, no search) (same file, second table): Qcrude_ideal/CIS_total 44.4 (1e-6,0.01), 25.6, 7.51; 12.7, 15.2, 4.5; 3.63, 4.28, 1.27; 2.14, 1.21, **0.353** (1e-3, 0.1) (rows gamma 1e-6, 1e-5, 1e-4, 1e-3; eps 0.01, 0.03, 0.1). QIS_ideal/CIS_sampling: 0.136, 0.308, 1.71 (1e-6); 0.153, 0.344, 1.91 (1e-5); 0.176, 0.396, 2.2 (1e-4); 0.212, 0.478, 1.33 (1e-3). So idealised quantum-with-IS wins on the sampling stage only at eps_rel <= 3% (0.14-0.48) and loses at 10% (1.3-2.2); with doubling search 0.962-14.1.
- Earlier (optimistic-denominator) table RQ/rare_summary.md "Per gamma and eps": QIS/CIS ideal medians 0.022 .. 0.134 and "share QIS<minC ideal = 1.00" in all 12 cells; QIS/minC doubling 0.137 .. 0.942 (share 0.59 at gamma 1e-3 eps 0.01, otherwise 1.00). In that file the classical IS is charged its pilot (median total 121 877 vs sampling-only 1 868 at eps_rel = 0.1) and quantum none. It is the idealised accounting; the fair comparison supersedes it (Section 7 item 21).
- Wall time of the rare experiment: 255.8 min (rare_summary l5).
- Conclusion on this model (DESIGN_RARE l73-79): rare overflow is a union of near-linear stage events in Gaussian inputs, captured by a mixture of mean shifts; with RV ~ 5 the quadratic gain cannot repay the constants. Quantum would need eps_rel <= 1% with an IS-equipped oracle and no doubling overhead, or a simulator whose failure region defeats IS.
- Parity time per Toffoli: NOT in any QUASAR file. A figure exists in the spin-off: scripts/ make_rarecert_figures2.py (panel "time per Toffoli for parity", tb = C_IS t_sim/(Q_IS gates), gates = ccx of the oracle at 8 stops, 8 bits = 995 Toffolis, t_sim = 1.2e-5 s measured numpy draw, alternative 1e-7 s assumed compiled draw) and papers/rarecert/submission_ijoc/rarecert_ijoc.tex l962-964: "At eps = 1% that time is about 4e-8 s, well below the 1 microsecond order ... reference value, not a hardware claim." Numbers beyond that sentence: MISSING in PQ and RQ. That sentence is with RV of SMIS and no pilot (idealised); it is a paper-8 result, not QUASAR's.
- Fault-tolerant crossover (M5, RQ4): NOT DONE (PQ/STATUS.md l25 unchecked; no file). No physical qubit or time estimate exists for any QUASAR algorithm. Literature figures only: omanakuttan2025threshold (179 variables, ~15 h, ~7.4e7 physical qubits at p=1e-3, 1 microsecond cycle) and "typical assumptions" (physical error 1e-3, 1 microsecond cycle, 10 microsecond reaction, code distance ~15) (PQ/LITERATURE.md l105-116).

---------------------------------------------------------------------------------------------------
## 6. One-sentence honest conclusion per test, with what favours each side

T1 (coherent CRN, M3). Conclusion: a shared register cuts the quantum cost of a difference by about
rho_q (about 0.13-0.16 on CON/SCA intra and alternative-position moves, about 0 in sample on CMT/city
because the end-load lemma makes most reorderings identical), but not for insertions (0.92 CON/SCA), and
at relative precision the classical direct difference estimator is cheaper (Nq/Nc 3.4-7.3 on signed
moves).
- Favours quantum: absolute precision 1e-3/1e-4 (ratios 0.0063-0.5 except city 1.7 at 1e-3 floor); sqrt(v)/eps vs v/eps^2; query count (not time) as cost; AE constants "conservative by 1.6-4.6x" in three checked cases; MLAE S>=64 on the rotation model (noise-free).
- Favours classical: queries are not equal-cost (difference oracle = two route circuits, copula factor register not built); relative precision; floor regime (3/eps vs 2R(2M+1)); exact DP exists for the model; classical CRN gain is larger in the same-paradigm ratio (Nc/Nc_sa 0.007-0.06 on signed moves); M3b done on an idealised, noise-free rotation model, not on circuits beyond 22 qubits.

T2 (search inside the optimiser, M4). Conclusion: FIRST-FEASIBLE has a favourable core ratio (0.011 to
0.626) but with the explicit constants of its bounded-error search the quantum cost is 19.7 to 501
times the classical in every one of 36 232 real repair steps (0 steps with Nq < Nc).
- Favours quantum: sqrt(k*) vs k*; k* up to 446 on Det plans; CRN core ratios 0.0276-0.642.
- Favours classical: SAA plans have k* = 1 in 0.73-0.83 of steps (no search needed); constants of coherent median + capped runs + bisection; the lower bound (c) matching Theorem 1 is only a conjecture; local search steps cannot be sped up (Aaronson 2006: no general speedup in number of steps).

T3 (simultaneous certification, black box and local). Conclusion: in the black-box model with a
variance-agnostic quantum bound, quantum loses for coarse eps and for larger d (ratio_B 3.7, 13.8 at
eps 1e-2 on SAA plans), wins only for eps <= 1e-3 (0.395, 1.47 ... 0.0397, 0.148 at 1e-4); the
variance-aware bound would win from eps ~ 1e-2 but its constant is unknown; in the local model Q/C is 4.7 to 7.4 (GD Det 1.03 at best).
- Favours quantum: eps 1e-4 (ratios 0.006-0.148); against a single whole-plan classical stream (Q/C_shared 0.019-0.961 on Det plans); ideal-vs-ideal 0.526 (GD Det); constant C=1 in Tang.
- Favours classical: shared samples give all d indicators in one call; Bernstein exploits small route risks; the quantum side gets the oracle choice of k (best fixed k on true risks) and the classical side gets a no-doubling exact stopping size, so both baselines are favoured; Tang's bound is a preprint with unstated constants and requires bounds on tr Sigma.

T4 (per-route certification with gap). Conclusion: quantum beats classical only for the 9 routes with
gap < 3e-3 (about 0.1% of 10 518); for the 99% of routes at gap > 0.03 the quantum median ratio is 5.0
to 11.2.
- Favours quantum: eps tight, binding routes (gap ~ eps). Favours classical: real plans keep routes far from the gate; empirical Bernstein with tiny samples; gap crossover (~5e-3) sits at the resolution of the risk estimates.

T5 (rare events vs mixture IS). Conclusion: against classical mixture IS (RV 3.9-5.9, verified against
the exact DP in 178/178 cases) the quantum amplitude estimate costs 2.5-380 times more (crude oracle,
pilot counted); idealised IS-equipped quantum is cheaper only at eps_rel <= 3% (0.14-0.48).
- Favours quantum: crude MC comparison (0.003-0.5); idealised accounting (no pilot, no likelihood-ratio arithmetic, M known); eps_rel = 1%. Favours classical: IS with mixture of mean shifts exploits the union structure; amortised pilot (Qcrude/CIS_amortised30 median still 26.6-1.22e3).

---------------------------------------------------------------------------------------------------
## 7. Flags: conjectures, unverified, superseded, inconsistent (must NOT be stated as results)

Conjectures / unproven
1. PQ/THEORY_M4.md l185-191 Prop. 3(c) product lower bound Omega(sqrt(k) sqrt(alpha(1-alpha))/eps): CONJECTURE.
2. PQ/THEORY_M4.md l153-155 Remark 3 variable-time bound O~(sqrt(sum T_j^2)): CONJECTURE.
3. PQ/THEORY_CERT.md l155-159 direct-sum lower bound for d routes: CONJECTURE.
4. PQ/REFRAME_M4.md l99 and PQ/THEORY_CERT.md l167: Tang (2025) optimality for l_infty is conjectured by its author; constants not given; preprint; assumed C = 1 in the table l101-111.
5. PQ/PROJECT.md l79-85 insertion lemma labelled "(to prove)"; no written proof anywhere (sketch + numerical check). PQ/DESIGN_M3.md l166 end-load lemma "to be written up" (short proof exists in RQ/m3a_v3_summary.md Sec. 2).
6. PQ/THEORY_M4.md l182: Prop. 3(a) PROVED only for k >= 16. PQ/THEORY_CERT.md l148-153 Prop. 2 is labelled PROVED "from the k = 1 case of Prop. 3", while l188-190 of the same file says the k = 1 case "needs its own two-point argument (Prop. 3(a) is proved for k >= 16)": the classical per-route bound is not proved in the files (the quantum Omega(sqrt(alpha)/eps) is, via Prop. 3(b)).
7. PQ/THEORY_M4.md l224-231 Corollary 4 "consistent with a quadratic advantage" and PQ/PROJECT.md l73-97 query formulas O(sqrt(Delta)/eps + 1/sqrt(eps)): heuristic, not theorems; optimality not established.
8. PQ/REFRAME_M4.md l81-84 crossover "roughly sqrt(d) eps < v" and "eps <~ 1e-3 on VNN, 5e-4 on GD": heuristic readings of one table (v0).
9. PQ/REFRAME_M4.md l139 crossover gap "approximately 5e-3": interpolation between the 3e-3-1e-2 band (0.84) and 1e-3-3e-3 band (0.25) built on 1+8+21 routes, near the 3e-3 resolution of p_j (RQ/cert_region_summary.md l5).
10. PQ/DESIGN_M3.md l184-185: heuristic conditional overflow step width sqrt((1-rho)/(rho L)) (RQ/ m3a_v3_summary.md Sec. 4b: "Heuristic, not established by the data").

Superseded
11. PQ/PROJECT.md l99-105 "M3 finding": median rho_q = 0.98 on 648k insertion moves and pilot intra 0.27 / 0.44: first run on independent discretised demands (b=4), superseded by M3a v3 (0.917 / 0.92 / 0.394 / 0.674 insertion; 0.133 / 0.136 / 0 / 0 intra). Same in PQ/DESIGN_M3.md l61-76 (marked there as first run). PQ/PROJECT.md l13-14 says "Nothing in this file is a result yet" (stale).
12. PQ/REFRAME_M4.md l31-35 (simultaneous quantum "Õ(sqrt d/ε), to be verified"; "quantum wins iff sqrt d ε below log d; region scale matters") is corrected by l78-84 of the same file (larger network hurts).
13. PQ/DESIGN_M3.md l133-136 "b = 5 by default" superseded by l187-188 and l208 (b = 6). PQ/DESIGN_M3.md l125-130 query constants superseded the "optimistic by 2-4x" ones (l105-106).
14. RQ/m4b: PQ/THEORY_M4.md l236 says 2e5 common scenarios; code default N_RR = 50 000 (summaries do not state N). Report the code value only after checking the run log (RQ/m4b_rr.log).

Inconsistent between files
15. PQ/STATUS.md l23 vs RQ/m3a_v3_summary.md Sec. 3b: STATUS says "0.42-0.72 (signed moves) at eps 1e-3": the table gives 0.415-0.716 for CON/SCA/CMT and 1.71 / 1.74 for city. STATUS "0.006-0.06 at 1e-4": the table gives 0.0063-0.011 (insertions) and 0.0345-0.306 / 0.0345-0.5 (signed). Use the tables.
16. PQ/STATUS.md l23 "factor-grid error budget ~3e-4" (and DESIGN_M3 l205-207 "up to 2.6e-4 in P(F>Q)") vs RQ/m3a_v3_summary.md Table 4b-2 (b=6): GH64 minus EMa32 is -4.58e-03 on SCA3-1, +1.00e-03 on HCMC-100-3 and +9.88e-04 on CMT11X (EMa increments on SCA3-1: -2.11e-3, +1.79e-2); EMa64 at b=6 was "not run for all routes". The ~3e-4 number is established at b=5 only (largest GH128-EMa128 gap 2.68e-04). The text also says no bin count up to 2^6 is shown to meet 1e-4 on short low-probability routes. b=6 is chosen for demand levels, not demonstrated for the factor grid.
17. PQ/STATUS.md l24 and PQ/REFRAME_M4.md l9-12 "22-500x costlier in every step": tables give medians of per-step ratios from 19.7 to 501 (wc 19.7-337, opt 31.6-501). "72-83%" cheapest feasible on SAA plans: tables give 0.734-0.755 (region) and 0.812-0.832 (73-set) = 73%-83%. "q90 30/50/130" for Det: 30.9-32 / 50-51 / 123-134 depending on eps.
18. PQ/REFRAME_M4.md l143-145 "plan-level totals 4.7-7.4, except GD Det at eps <= 1e-3 (1.03)": RQ/ cert_region_summary.md has GD Det 3.634 at eps = 0.01 and 1.741 at 0.003 as well (only SAA and VNN Det are 4.7-7.4 for all eps).
19. PQ/THEORY_CERT.md l137-139 says the classical side gets the exact Clopper-Pearson stopping sample size; QD/cert.py c_cert_route and PQ/REFRAME_M4.md l126 use the empirical-Bernstein size (Audibert et al. 2009). PQ/THEORY_CERT.md l104 "k in {1..20}" vs QD/cert.py KS_DEFAULT = {1,3..20} (k=2 invalid, s_2 = 1/2). PQ/THEORY_CERT.md l183-186 lists classical rivals (sequential tests, conditional Monte Carlo over the factor, importance sampling) that are NOT implemented in cert_region (only Bernstein).
20. PQ/DESIGN_RARE.md l59-60 "single mean shift was off by up to 11-19%": the DP table in RQ/ rare_summary.md (178 cases, column p_dp/p_single - 1) has max |.| = 0.4154 (VNN-HANOI-1#60, gamma 1e-4) and 7 of 178 cases above 0.10 (COMPUTED-HERE from the table; e.g. VNN-VINHPHUC#5 gamma 1e-4 +0.3093).
21. RQ/rare_summary.md vs RQ/rare_fair_comparison.md: the former charges classical IS its pilot and quantum none (QIS/CIS ideal 0.022-0.134 "share QIS<minC ideal = 1.00"); the latter's fair comparison gives QIS_ideal/CIS_sampling 0.136-2.2. Do not quote rare_summary ratios as results; they are the idealised accounting.
22. PQ/DESIGN_RARE.md l66-67 "cheaper on 23-34% of routes only at gamma = 1e-3...1e-4 with eps_rel >= 3%": the file gives 0.23 and 0.34 (gamma 1e-3, eps 0.03 and 0.1) and 0.28 (gamma 1e-4, eps 0.1; 0.00 at 1e-4, eps 0.03). PQ/DESIGN_RARE.md l70-71 "with the doubling search 1-14 (quantum loses everywhere)": median QIS_doubling/CIS_sampling is 0.962 (q10 0.87) at gamma 1e-6, eps 0.01, below 1.
23. PQ/DESIGN_RARE.md l50-51 ground-truth cross-check "for L <= 8 and gamma >= 1e-4": 178 cases (2 gammas x up to 120 routes; routes with L <= 8 not enumerated in the summary): MISSING count of routes.
24. PQ/DESIGN_M3.md l187-189 "b=5 median 0.5%": RQ/m3a_v3_summary.md Sec. 4a gives 0.587% (nearest), 0.549% (design rule), 0.59% in the text.
25. PQ/STATUS.md l9-10 state "PAUSED ... Open for edits"; l24-25 M4 and M5 unchecked; PQ/STATUS.md l6-8 venue "not fixed".
26. Not built / not measured, so not claimable: a copula-factor-register oracle; a select oracle over M candidate routes for M4 (Toffoli cost); difference-oracle Toffoli formula; any fault-tolerant resource estimate or crossover (M5); a d*(eps) crossover measurement; the sqrt(v) law for signed moves; hardware or noisy-circuit AE.
27. PQ/PROJECT.md l126-128 execution layer (quantum LSM recourse): not done. PQ/PROJECT.md l120-124 CVaR gate: not done (no result).
28. Region/instance details missing: number of instances per region (VNN vs GD), route counts per region beyond d in the tables, ALNS settings: MISSING in the QUASAR files.

---------------------------------------------------------------------------------------------------
## 8. References per test (bib keys in PQ/references.bib) and verification status

Status source: PQ/VERIFY_CITATIONS.md. "OPEN" = listed there with an open issue; "NOT IN BIB" = no
entry in references.bib (149 entries counted); "memory" = added from memory, metadata unchecked.

T0/Oracle (Section 1)
- Brassard2002 (AE, Fact 1 with Thm 12 bound; Thm 15 relative error read in full; Thm 12 not named as read in VERIFY l71-80: check), Cuccaro2004 (adder), Gidney2018 (arithmetic), GroverRudolph2002, Mottonen2004, Mottonen2005, zoufal2019 (state preparation), Barenco et al. 1995 (V-chain; NOT IN BIB), Gacon2020 (same load-vs-capacity primitive; "must be read in full" VERIFY l36-38).
T1 (CRN, Section 2)
- Montanaro2015, KothariODonnell2023, Hamoudi2021 (OPEN: confirmed by one review, not located by another; re-check DataCite DOI), HamoudiMagniez2019 (OPEN: page range), An2021mlmc (read in full pending), Blanchet2025, Suzuki2020 (MLAE), Grinko2021 (IAE), AaronsonRall2020, nakaji2020, HuHuZhou2026 and HuHu2026 (independent-noise selection; read-in-full pending), Wang2021bandits (Thm 5 only for biases in [p, 1-p]; read in full pending), WangFilipovska2026 (only stochastic-VRP QAE; read-in-full pending), WoernerEgger2019, Rotello2024, Skarlatos2025, Sharma2024.
T2 (search, Section 3)
- boyer1998tight, hoyer2003bounded, ambainis2006variable (TOCS 2010 version; Thm 3 for ideal subroutines), Kothari2014OracleID (remark only), DurrHeiligmanHoyerMhalla2006 (attribution of find-first not found in DHHM text), grover1996fast, DurrHoyer1996, Aaronson2006, NayakWu1998Median (venue STOC 1999 not verified; cite arXiv; specialisation is our derivation), ReichardtReflections2011 (DOI unchecked), LeeMittalReichardtSpalekSzegedy2011, MannorTsitsiklis2004, KanoHondaSakamakiMatsuuraNakamuraSugiyama2019, LocatelliGutzeitCarpentier2016 (pages unchecked), KaufmannCappeGarivier2016 (memory; metadata to check), Bennett1997BBBV (memory; metadata to check), Moosavi2026RLQALNS (read-in-full pending), Kim2026FeasibilityQS, MeghaziBourreau2026, Montanaro2020bb, chakrabarti2022bnb (OPEN: statement seen only in summary), regev2012faulty.
T3/T4 (certification, Sections 4-5)
- Brassard2002 (Thm 12), Huggins et al. PRL 129 240501 (2022) arXiv 2111.09283 (NOT IN BIB; read in full), Tang arXiv 2504.06940v3 (NOT IN BIB; preprint; read in full), Cornelissen2022 (abstract only; l2 not l_infty), van Apeldoorn TQC 2021 (NOT IN BIB; abstract only), Audibert-Munos- Szepesvari 2009 (empirical Bernstein; NOT IN BIB), Wainwright 2019 (classical lower bound candidate; NOT IN BIB; not verified), Hoeffding/Clopper-Pearson (classical, NOT IN BIB), Montanaro2015, KothariODonnell2023, Suzuki2020, Grinko2021, AaronsonRall2020.
T5 (rare events, Section 5)
- Brassard2002 (Thm 15, Theta((1/eps) sqrt(1/p)), verified), Montanaro2015, KothariODonnell2023, tabarraei2026 (rare events), Glasserman-Li 2005 and Owen-Maximov-Chertkov 2019 (VERIFIED in papers/rarecert/VERIFY_CITATIONS.md l13, l29 but NOT IN papers/quasar BIB), cross-entropy method (NOT IN BIB).
Resources / realism (Sections 1, 5; M5 not done)
- Babbush2021, sanders2020compilation (OPEN: author list "and others"), omanakuttan2025threshold (OPEN: "and others"), beverland2022requirements (OPEN), gidney2021rsa, gidney2025rsa, gidney2024cultivation, litinski2019distillation, fowler2018lattice, remaud2024adders, acharya2024below (OPEN), vandam2023azure, leblond2026surface, borras2026 and lal2026 (hardware MLAE error floor ~2-5e-2).
Other keys with open issues in VERIFY (avoid unless essential): Feld2019 (author list), Doriguello2022
(DOI does not resolve at Crossref), Borowski2020, koch2026qoblib, abbas2024challenges, erdmann2026strategic
("and others"); Mahroo2026benders, RibesMarza2026jcc, Zheng2026gas, AmaniBani2026ride, Towada2026kernel
(incomplete Crossref author strings). Records in VERIFY "not in references.bib" (snippet only, must not
be cited): Muralitharan et al. NeurIPS 2026 poster, Goto-Ohzeki arXiv 2601.20887 (read before claiming
no prior quantum work on random demand), Palanivel QIC 2026, Eder 2026, Fedouaki 2026, Weinberg 2026,
Ghosh 2026, arXiv 2309.17299, 2602.09847.
Literature caveat: arXiv API was rate-limited, so "nothing found" statements are "evidence, not proof;
re-checked before submission" (PQ/LITERATURE.md l5-7). Claims to avoid: "first quantum algorithm for
stochastic VRP" (false since WangFilipovska2026), "new quantum variance reduction technique", any
practical advantage without the crossover analysis (PQ/LITERATURE.md l129-132).
