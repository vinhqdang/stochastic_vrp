# QUASAR — re-aim after M4 (2026-10-13)

## Decision

QUASAR stops competing with classical optimisation. Planning (route
construction, repair, local search) stays classical: MIP solvers and ALNS
are far ahead and no quadratic query speedup changes that. M4 measured it
directly: on 36 000 real repair steps (73 benchmark instances and the region
set of 148 instances) the cheapest-feasible search has a favourable core
ratio (quantum/classical 0.011–0.63) but loses by 22–500× once the constants
of the bounded-error search are counted, in every step.

The new aim is the **certification layer**: given a plan produced by any
classical solver, certify its risk. This is an estimation problem, where the
classical rival is Monte Carlo (with shared samples, importance sampling),
not a MIP solver, and where quantum amplitude estimation has a provable
quadratic edge in the number of calls to the source of randomness.

## Setting in which the claim is meaningful

1. **Black-box uncertainty.** The day is produced by a stochastic simulator
   (a regional digital twin: demand, traffic, weather, breakdowns) that can
   only be sampled. The cost unit is one simulator call. In the BATON demand
   model the peak-load risk has an exact dynamic program, so there the
   quantum estimator is a validated testbed, not a use case; the paper must
   say so.
2. **Network-wide certification.** A regional plan has d routes (Northern
   Vietnam about 500 under the SAA gate, Guangdong several thousand) and,
   during repair, d candidate moves. The task is to certify every route's
   risk to additive error ε at once. Classically one simulator call yields
   all d indicators, so shared samples cost O(log(d/δ)/ε²) calls; per-route
   quantum estimation costs d/ε, which is worse; simultaneous quantum
   estimation of d expectation values costs Õ(√d/ε) (to be verified, see
   below). Quantum then wins iff √d·ε is below log d up to constants: the
   region scale matters here, unlike in M4.
3. **Safety-critical tails.** Medical, cold-chain or hazardous-goods
   service levels need p ≤ 10⁻⁴…10⁻⁶ certified to relative error, where
   classical cost is 1/(ε_rel² p) against 1/(ε_rel √p) quantum; importance
   sampling is the classical competitor and must be included.

## Contributions to target

- A coherent region-level oracle: all routes of a hub (or of a region) on
  one scenario register with the shared copula factor, so one call of the
  oracle is one simulated day, matching the classical cost unit.
- Simultaneous certification of all routes (and of the candidate moves of
  a repair pass) with an explicit constant-carrying cost, against the
  shared-sample classical bound; the crossover d*(ε) measured on the 148
  region instances.
- The rare-event regime, with importance sampling on both sides.
- Fault-tolerant resource estimates (M5) in simulator-call terms, using the
  M2 Toffoli formula, and the honest negative result of M4 (search inside
  the optimiser does not pay).

## Checks before committing to this aim

- Exact statements of the multivariate bounds (Huggins et al. 2022; van
  Apeldoorn 2021; Cornelissen–Hamoudi–Jerbi 2022), their oracle model and
  their constants — under verification.
- Whether the oracle model of those results (a single state-preparation
  unitary producing all d indicators) matches our region oracle.
- Classical lower bound for shared-sample simultaneous estimation.

## First measurement (2026-10-13, `results/quasar/cert_crossover_v0.txt`)

Route risks of every route of the region plans (20 000 scenarios per
instance); classical = shared samples with Hoeffding or Bernstein (v = max
route variance), quantum = Huggins et al. (2022) Thm 4 form
2·√d·log(d/δ)/ε (variance-agnostic), δ = 0.05.

| plan | d | quantum/classical (Bernstein) at ε = 1e-2 / 1e-3 / 1e-4 |
|---|---:|---|
| VNN SAA | 375 | 3.7 / 0.40 / 0.040 |
| GD SAA | 5 867 | 13.8 / 1.47 / 0.148 |
| VNN Det | 257 | 0.59 / 0.059 / 0.006 |
| GD Det | 4 019 | 2.4 / 0.24 / 0.024 |

Correction to point 2 above: a larger network HURTS the simultaneous
quantum estimator (cost √d against log d classically, since one classical
sample gives all d indicators). The crossover is roughly √d·ε < v: for the
well-planned SAA plans (v ≈ 0.05) quantum needs ε ≲ 1e-3 on Northern Vietnam
and ε ≲ 5e-4 on Guangdong. The variance-agnostic quantum bound is the
problem: classical Bernstein exploits the small route risks, the quantum
bound does not.

Next: a variance-aware simultaneous estimator whose cost scales with
√(Σ_j p_j(1−p_j)) — the square root of the expected number of failing
routes per day — instead of √d (candidate: L. Tang, arXiv 2504.06940, to be
read in full); and the rare-event relative-error axis (Brassard–Høyer–Mosca–
Tapp Thm 15, Θ((1/ε)√(1/p)), verified).

## Variance-aware bound (L. Tang, arXiv 2504.06940v3, read in full)

Thm I.2: ℓ∞ error √(tr Σ)/n with O(n log(d/δ)) calls of a state preparation
that returns the whole d-vector in one call; for indicators tr Σ = Σ_j
p_j(1−p_j) exactly, whatever the correlations. So the cost is
O(√(Σ_j v_j)·log(d/δ)/ε) — the √(expected number of failing routes) form.
Constants are not given (hidden O, an unspecified D); it needs a known bound
on tr Σ and on ‖E X‖₂; preprint only, optimality for ℓ∞ conjectured.

With constant C = 1 (optimistic) against classical Bernstein:

| plan | d | tr Σ | ratio at ε = 1e-2 / 1e-3 / 1e-4 |
|---|---:|---:|---|
| VNN SAA | 375 | 7.5 | 0.26 / 0.028 / 0.0028 |
| GD SAA | 5 867 | 124 | 1.0 / 0.11 / 0.011 |
| VNN Det | 257 | 51 | 0.13 / 0.013 / 0.0013 |
| GD Det | 4 019 | 826 | 0.54 / 0.054 / 0.0054 |

The ratio scales linearly in C, so the crossover precision ε* scales as
1/C: the whole practical claim rests on a constant nobody has computed.

## Decision (proposed)

Algorithmic contribution of QUASAR = an explicit-constant, variance-aware
simultaneous certification algorithm for d correlated indicator events
(route overflow) with one-sided thresholds (certify p_j ≤ α for every j,
not estimate every p_j), proved and counted gate by gate, plus the coherent
region-level oracle; evaluated on the 148 region instances against the
best classical shared-sample certificate. M4 (search inside the optimiser)
is reported as a negative result.
