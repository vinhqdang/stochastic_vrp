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
