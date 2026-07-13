# TEMPO — the theory that makes it an OR paper

Motivation: paper 1 was desk-rejected at EJOR for "contribution to OR
not enough." The statistical machinery in TEMPO (Ville, e-processes,
mixture/adaptive betting) is imported and citable, NOT a contribution.
The contribution must live in the COUPLING of sequential testing to
routing economics. This file states the theorems that do that, with
proofs or proof routes. Notation follows PROJECT.md.

## Honest inventory

| ingredient | status |
|---|---|
| e-processes, Ville, anytime validity | imported (Ramdas et al.) |
| mixture bets, aGRAPA adaptive tilts | imported (Waudby-Smith–Ramdas) |
| predictable-tilt validity, alpha-split | routine lemmas (state, don't sell) |
| **cost-regret bound for a statistically triggered replan (Thm 1)** | **new** |
| **closed-form optimal evidence level alpha\* from fleet economics (Cor 1)** | **new** |
| **the value of waiting: evidence-timed > immediate replan (Thm 2)** | **new** |
| **decision-relevant betting is growth-optimal for cost-weighted risk (Thm 3)** | new, harder |

## Setup (for all results)

One operating day produces events s = 1, 2, ..., S (services/legs) with
observations Z_s ~ P0 (the planning law) up to an unknown change point
nu, and Z_s ~ P1 after it, where KL := KL(P1 || P0) > 0 per event.
TEMPO runs an e-process E_s with alarm time
tau = inf{s : E_s >= 1/alpha}, and on alarm invokes a fixed replanner R
(any of ev/replan.py; the result is optimizer-agnostic). Costs:

- Delta_c  >= 0: per-event staleness cost — the expected extra cost of
  executing one more event under the stale plan instead of the
  R-replanned plan, once the world is in the P1 regime. (Measured
  empirically per scenario; bounded under bounded detours/penalties.)
- C_fr > 0: expected cost of one unnecessary replan on an on-model day
  (the detour/decoupling cost R induces when nothing was wrong).
- Oracle policy: invoke the SAME R exactly at nu.

## Theorem 1 (regret of the TEMPO-triggered replan)

Assume (A1) events after nu are iid P1 given the filtration (or,
weaker, the per-event conditional KL is >= KL_min > 0); (A2) the
per-event staleness cost is bounded by Delta_c; (A3) the e-process bets
achieve log-growth rate g > 0 per post-change event (for the grid
mixture, g = max_j E_P1[log e(theta_j)] > 0; for the adaptive bet,
g -> KL by aGRAPA consistency). Then TEMPO's expected day cost exceeds
the oracle's by at most

    Regret(alpha) <= Delta_c * ( log(1/alpha) + kappa ) / g
                     + alpha * C_fr,

where kappa bounds the expected log-overshoot of the mixture bet
(kappa <= max_j E_P1[log e(theta_j)] over one event; finite by
bounded tilts).

*Proof route.* (i) On {no false alarm before nu}, the day-cost
difference between TEMPO and the oracle is at most Delta_c * (tau - nu)
by (A2) and the fact that both policies coincide after their (common)
replanner call modulo the delayed state — telescoping over the
tau - nu stale events. (ii) E[tau - nu | tau > nu] <=
(log(1/alpha) + kappa)/g is Wald's identity applied to the log
e-process increments after nu (standard sequential-analysis argument;
the mixture keeps increments' conditional mean >= g under P1).
(iii) On {false alarm before nu} (probability <= alpha by Ville), the
extra cost over the oracle is at most C_fr — the day is on-model until
nu, so the only difference is one unnecessary R-call — plus the
post-nu difference, which after a reset is bounded by the same delay
term (segment-wise validity). Union the two events. QED route: all
three steps are individually standard; the theorem is their
composition in a routing-cost metric, which is the new object.

## Corollary 1 (the optimal evidence level — "how sure should a
dispatcher be before replanning?")

Regret(alpha) is convex in alpha with a unique interior minimizer

    alpha* = min( 1, Delta_c / ( g * C_fr ) ).

Consequences, all OR-native:
- alpha is not a "significance level" chosen by convention: it is a
  PRICE, and its optimal value is a ratio of fleet economics —
  staleness cost rate over (evidence rate x false-replan cost).
- Cheap replans (small C_fr, e.g. software-only re-sequencing) push
  alpha* toward 1 — replan liberally; expensive replans (driver
  coordination, standby vehicles) push alpha* down — demand evidence.
- This gives practitioners a defensible dial where today they use
  fixed review periods; no analogous result exists in the
  rolling-horizon literature (their re-optimization epochs are
  exogenous).

## Theorem 2 (the value of waiting for evidence)

Let R be a state-dependent replanner whose output depends on realized
loads/slacks (e.g. rebalance_regret), and let cost_R(s) denote the
expected remaining-day cost when R is invoked at event s. Suppose
(B1) invoking R before nu on a drift day acts on unbiased but
higher-variance slack estimates, with expected penalty V(s) decreasing
in s (information accrues: V'(s) < 0); (B2) staleness accrues linearly
after nu at rate Delta_c. Then the expected day cost of replanning at
s has the form

    J(s) = J* + V(s) * 1{s < nu'}  + Delta_c * max(0, s - nu),

with nu' the time realized information saturates, so the minimizer is
INTERIOR whenever V(nu) > 0: replanning immediately is dominated by
waiting until enough uncertainty has realized, and TEMPO's alarm time —
which by construction fires only when the evidence (hence the realized
information) is large — lands near the interior optimum.

*Status.* The empirical version is already measured: TEMPO+rebalance
56.7% saving vs 50.8% for replan-at-first-stop (results_ev_replan.csv).
The theorem needs a clean stylized model (two vehicles, Gaussian
demands, one rebalancing decision) where V(s) is the posterior variance
of the slack difference — then J(s) is exact and the interior optimum
is a two-line calculus argument. This is the "why waiting pays" story
formalized; we have found no analogue in the DVRP literature, where
more re-optimization is always weakly better because information is
assumed free and replans costless.

## Theorem 3 (decision-relevant betting is growth-optimal for
cost-weighted risk) — the coupling theorem

Define the cost-relevant alternative class P_Delta = {tilted laws whose
expected one-event cost increase for the CURRENT plan is >= Delta}.
Claim: among predictable betting strategies with the same budget, the
sensitivity-tilted bet (theta proportional to the plan's cost
subgradient direction, i.e. our s_k(t) weights) maximizes the worst-
case log-growth over P_Delta; equivalently, TEMPO's detection delay
against the LEAST detectable cost-Delta drift is minimized.

*Proof route.* For exponential-family channels the growth of a tilt
theta against alternative eta is theta.eta - psi(theta); the worst-case
eta over the half-space {cost-gradient . eta >= Delta} aligns with the
cost gradient by linear-programming duality; the max-min then forces
theta parallel to the gradient (standard saddle-point argument, cf.
GRO). The OR content: the plan's economics define the alternative
class, and the optimal statistical design follows from the
optimization's sensitivity structure — this is the formal version of
"e-process betting informed by optimization state" and is, to our
knowledge, new.

## What we will claim, and what we will not

Claim: Theorems 1-3 + Corollary 1 as the OR-theory core; the routine
validity lemmas stated as lemmas with short proofs; the statistics
imports cited generously. Not claim: any novelty in e-process theory
per se.

## Proof-completion checklist (next theory milestone)

- [ ] Thm 1: write the telescoping step (i) carefully for the reset
      scheme; verify the overshoot constant for our grids.
- [ ] Cor 1: derivative + convexity; add the plot alpha -> empirical
      regret from the cost harness to confirm the interior optimum.
- [ ] Thm 2: build the two-vehicle stylized model; exact V(s).
- [ ] Thm 3: write the saddle-point argument for the Gaussian and
      Poisson channels we actually use.
- [ ] Empirical validation figure per theorem (regret vs alpha sweep;
      J(s) curve vs replan time; delay vs cost-Delta drift direction).
