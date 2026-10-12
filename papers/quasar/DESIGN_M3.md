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
