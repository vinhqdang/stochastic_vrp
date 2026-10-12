# QUASAR M3 — experiment design

Two experiments. Code: `svrpspd_wdro/quantum/experiments.py` (library) and
`svrpspd_wdro/scripts/run_quasar_m3.py` (driver); outputs in
`svrpspd_wdro/results/quasar/`; tests in `tests/test_quantum_experiments.py`.

## M3a — how much do coherent common random numbers save on real moves?

Data: every cached plan in `results/plans/*.json` (73 instances: Dethloff
CON/SCA, Salhi–Nagy CMT, city), every gate present in `res`, every route.
Demand model: BATON's marginals (gamma with the runner's default
coefficient of variation, means `dbar`, `pbar`), taken **independent**
across customers (the exact DP and the oracle assume independence; BATON's
copula experiments are out of scope and this is stated as a limitation).

Discretisation: V = 2^b levels with one common unit u per instance,
u = max_i q_{0.999}(demand_i)/(V−1); pmf by binning the exact gamma cdf on
[(v−½)u, (v+½)u), the top level taking the upper tail. Q_level = ⌊Q/u⌋.
Report the discretisation error of P(F_r > Q) against 10⁵ continuous Monte
Carlo draws per route (b = 4 default; b = 3, 5 as sensitivity).

Moves: for each route r and each customer j of another route of the same
plan, each insertion position pos = 1..|r|+1 (cap: 200 random moves per
route, fixed seed). By the insertion lemma F_{r'} ≥ F_r scenario-wise, so the
window mass is exact from two DP calls:

    p = P(F_r > Q),  p' = P(F_{r'} > Q),  Δ = p' − p   (assert Δ ≥ −1e−12).

Per move record p, p', Δ and the variance ratios
- ρ_q = √(Δ(1−Δ)) / √(p'(1−p'))   quantum CRN vs quantum stand-alone,
- ρ_c = Δ(1−Δ) / (p'(1−p'))       classical CRN vs classical stand-alone,
and the query counts at precision ε ∈ {10⁻², 10⁻³, 10⁻⁴} with the standard
constants: classical N_c = z² v/ε², quantum N_q = π√v/ε (amplitude estimation
error bound 2π√(a(1−a))/M; the constant is the same for both quantum
variants, so ρ_q does not depend on it), v the variance of the estimated
indicator (p'(1−p') stand-alone, Δ(1−Δ) with CRN), z = 1.96.

Summaries (by family CON/SCA/CMT/city and overall): quantiles of ρ_q and ρ_c,
the fraction of moves with Δ = 0, conditioned on the decision-relevant band
p' ∈ [0.01, 0.30]; and incremental feasibility totals per route (base once +
increments vs every move stand-alone).

## M3b — amplitude estimation on the actual oracles

Small routes taken from the real instances (L ∈ {2, 3}, b ∈ {1, 2}, at most
22 qubits) and their difference oracles. For ε_target ∈ {0.05, 0.02, 0.01,
0.005}: IAE (α = 0.05) and MLAE (schedules 2–6), 10 seeds each, recording the
absolute error against `exact_probability`, the number of applications of A,
and the classical Monte Carlo sample count needed for the same error at 95%.
Purpose: confirm the error–query scaling on the real circuits and that the
difference oracle's cost follows √Δ rather than √p'.

## Outputs
- `results/quasar/m3a_moves.csv` (one row per move), `m3a_routes.csv`,
  `m3a_discretisation.csv`, `m3a_summary.md`;
- `results/quasar/m3b_ae.csv`, `m3b_summary.md`;
- logs `results/quasar/*.log`.

## Findings of the first run (2026-10-12) and the revised design

**Inter-route insertions: negative result.** On 648,036 real insertion moves
(73 instances, cap 200 per route, b = 4, no lemma violation) the CRN gain is
negligible: median ρ_q = 0.98 overall and 0.98 in the band p' ∈ [0.01, 0.30];
incremental feasibility costs 0.98 of the stand-alone quantum queries. The
reason is structural: by the insertion lemma Δ = p' − p, so
ρ_q = √(Δ(1−Δ))/√(p'(1−p')) ≈ √(1 − p/p'), and an insertion of a whole
customer multiplies the risk (median p/p' = 0.10). The base route carries
little of the moved route's risk, so the difference is almost as variable as
the moved route itself. This is reported as a negative result.

**Intra-route moves: where CRN pays.** A pilot (every 4th plan, routes with
0.005 < p < 0.4, 15 random moves each, 4·10⁴ common scenarios) gives median
ρ_q = 0.27 for relocate (25% of moves leave the overflow indicator unchanged
in every scenario) and 0.44 for 2-opt, i.e. 2.3–3.7 times fewer quantum
queries and 5–13 times fewer classical samples. The same customers are
reordered, so the two peaks are strongly coupled.

**Revised M3a.**
1. Keep the insertion experiment as the negative baseline.
2. Intra-route moves on every route of every plan with p ∈ (0, 1): relocate,
   swap, 2-opt (all moves for |r| ≤ 12, else 200 random), and alternative
   positions of the same inserted customer (r'_{pos1} vs r'_{pos2}). D ∈
   {−1, 0, 1}; report Var(D)/(p'(1−p')), P(D ≠ 0), ρ_q, ρ_c by family and base
   band, from 10⁵ common scenarios of the discretised model (Monte Carlo
   standard errors reported), with an exact enumeration check on small routes.
3. Discretisation: choose u so that Q is a level boundary (Q = K·u, K =
   ⌈Q/u₀⌉ with u₀ the old unit), and report both nearest-level and ceiling
   rounding (ceiling is conservative); recheck against continuous Monte Carlo.

**Revised M3b.** Qiskit IAE stops after the k = 0 round at ε ≥ 0.05 and
overshoots the target below, so its query counts do not reveal the scaling.
Use maximum-likelihood AE with explicit power schedules (k = 0, 1, 2, 4, …,
2^J, fixed shots) on the real small circuits, and the exact Grover-rotation
model (P(good after k iterates) = sin²((2k+1)θ), validated on the circuits)
for the same schedules at the real amplitudes p' and Var(D) of M3a, to show
error ∝ 1/(A-applications) and the √Var(D) cost of the difference oracle.

## Revision 2 (after review of the first run)

Review findings that change the design: (i) BATON's demand model is a
one-factor Gaussian copula with ρ = 0.6, applied separately to deliveries
and to pickups (independent factors), gamma marginals with cv = 0.3; under
independence the protective-gate routes look 5–30 times safer, so every M3
statistic must use the copula model. (ii) floor(Q/u) with nearest rounding
causes errors up to 0.15 in P(F > Q). (iii) The quantum query constant was
optimistic by 2–4×. (iv) M3b did not test its purpose.

**M3a v2 — statistics on the BATON model itself.** The variance ratios are
properties of the routing model, not of the circuit, so they are estimated
on the continuous copula model with common scenarios (no discretisation):
N = 2·10⁵ scenarios per instance, shared by all routes and moves of that
instance, drawn exactly as `dethloff_runner.make_scenarios` (same marginals,
same copula, ρ = 0.6; ρ = 0 as a sensitivity run). Move classes:
1. inter-route insertion (negative baseline; the lemma still holds
   scenario-wise, D ∈ {0, 1});
2. intra-route relocate, swap, 2-opt (all moves if |r| ≤ 12, else 200 random);
3. alternative positions of the same inserted customer (r'_{pos1} vs
   r'_{pos2}).
Per move: p, p', P(D = 1), P(D = −1), Var(D), ρ_q = √Var(D)/√(p'(1−p')),
ρ_c = Var(D)/(p'(1−p')), with Monte Carlo standard errors (delta method).
Identical routes appearing under several gates are evaluated once.
Summaries by move class × family × base-p stratum (p < 0.005, 0.005–0.05,
0.05–0.1, 0.1–0.3, > 0.3).

**Query constants.** Canonical amplitude estimation with M Grover iterates
has error ≤ 2π√(a(1−a))/M + π²/M² with probability ≥ 8/π² and costs 2M + 1
applications of A. For error ε: M(ε, v) = smallest power of two with
2π√v/M + π²/M² ≤ ε. Confidence 95% by the median of R = 5 repetitions
(stated as an assumption; the factor is common to all variants). Classical:
N = z²v/ε², z = 1.96. A signed D needs two estimates (P(D = 1), P(D = −1)),
each at ε/2. Incremental feasibility uses ε/√2 per estimate.

**Discretisation for the circuits.** Unit u = Q/(K + ½), K the smallest
integer with K + ½ ≥ Q·(V−1)/q_{0.999}^{max}, Q_level = K, nearest-level
rounding; b = 5 by default. The check routes are stratified by base p (two
per stratum per family) and report the bias with a Monte Carlo interval.
The copula enters the circuit as a factor register: the conditional pmf of
each demand given a quantised factor value z_t (Gauss–Hermite nodes) is
prepared under uniformly controlled rotations; the arithmetic part of the
oracle is unchanged. (Circuit implementation is a later step; M3 only needs
the discretised conditional pmfs to measure the discretisation error, using
the exact identity P(F > Q) = Σ_t w_t Σ_s w_s P(F > Q | z_t, z_s) with the
classical DP for each node pair.)

**M3b v2 — estimator behaviour.** (1) Validate P(objective after k Grover
iterates) = sin²((2k+1)θ) on the real peak and difference circuits for
k = 0..6 (statevector, ≤ 22 qubits). (2) On this validated rotation model,
simulate canonical QAE (m = 3..12 evaluation qubits) and maximum-likelihood
AE with exponential schedules (k = 0, 1, 2, 4, …, 2^J; shots 1, 8, 64) for
amplitudes taken from M3a v2 (p' of stand-alone estimates and P(D ≠ 0) of
difference estimates, 30 moves per class), 2000 independent seeds each.
Report RMS error vs A-applications, the fitted slopes (quantum ≈ −1,
classical ≈ −½), and the cost ratio difference/stand-alone at equal RMS,
which should track ρ_q. Independent seed streams per configuration.

## Status after review of revision 2 (2026-10-12)

Measurements verified (scenarios = BATON model; 15 moves recomputed
independently within 1.5 SE; Grover rotation model holds on the real
circuits to 6e−15; MLAE re-implementation matches). Interpretation layer to
be redone: QAE errors from the exact outcome pmf (sampled RMS misses the
tails), per-class cost comparison (classical estimates a signed mean
directly), discretisation without quadrature error (≥ 64 nodes per factor
beyond L ≈ 6), ratio table without silent drops, MLAE only with S ≥ 32,
p'-band tables and route-bootstrap intervals.

**End-load lemma (to be written up).** For any reordering r' of the same
customers, L_0 = Σd and L_L = Σp are order-invariant, so
1[F_r > Q] = 1[F_{r'} > Q] whenever max(Σd, Σp) > Q; hence
P(D ≠ 0) ≤ P(Σd ≤ Q, Σp ≤ Q, max(F_r, F_{r'}) > Q). This explains the large
share of intra-route moves with Var(D) = 0 (55% on city, 50% on CMT).

## Revision 3 decisions (2026-10-12, after review of v3)

- **Factor register for the copula: equal-probability bins.** Quantise each
  copula factor (delivery, pickup) into 2^f bins of equal probability,
  u_t = (t + ½)/2^f, prepared by Hadamard gates alone (no rotations); each
  demand register is then prepared conditionally on the bin. Gauss–Hermite
  nodes (spacing 0.39 at 64 nodes) are coarser than the width of the
  conditional overflow step (about 0.18) and do not converge on long routes;
  equal-probability bins converge at 2^f = 16–32 on the tested routes.
  Default f = 5, to be confirmed on more routes at b = 5 and 6.
- **Demand levels:** b = 6 meets the rule "median relative bias ≤ 1 %, max ≤
  5 %" on the paired check (105 routes); b = 5 has median 0.5 % and max 8 %.
  The rule is a choice, stated as such; the worst cases are short routes.
- **Classical certificate floor:** a classical estimate that sees no event
  needs about 3/ε samples to certify a probability below ε, so the classical
  cost is max(z²v/ε², 3/ε); this changes the city cells of the cost table.
- **End-load lemma scope:** it covers every reordering of the same customers,
  hence also alternative-position comparisons.
