# RARECERT — project specification

*Certifying rare overflow risk of regional pickup-and-delivery plans.*
Status of every claim: PLAN until proved or measured.

## 1. Problem

A plan for a vehicle routing problem with simultaneous pickup and delivery
(produced by any solver) must meet a service level: every route, and the
plan as a whole, overflows its vehicle with probability at most γ, with γ
between 10⁻³ and 10⁻⁶ (medical, cold-chain and dangerous-goods
distribution). The load of route r after stage k is
L_k = Σ_{i>k} d_i + Σ_{i≤k} p_i, the peak is F = max_k L_k, and the route
overflows iff F > Q. Demands: gamma marginals, one-factor Gaussian copula
per demand type (the BATON/QUASAR model), and in R2 a richer simulator.

Tasks: (a) estimate p_r = P(F_r > Q) to relative error ε_rel with
confidence 1 − δ; (b) certify p_r ≤ γ for every route and P(∪_r {F_r > Q})
≤ γ_plan for the plan; (c) do so at regional scale (thousands of routes).

## 2. What is known (R0, 2026-10-14; details in VERIFY_CITATIONS.md)

- Mixture IS for a union of rare events: Owen, Maximov & Chertkov (EJS
  2019, ALOE) with a uniform CV bound when each event can be sampled
  exactly conditionally (Gaussian half-spaces); Gaussian-mixture
  cross-entropy (Kurtz & Song 2013; Geyer, Papaioannou & Straub 2019;
  Papaioannou et al. 2019). NOT claimable.
- Gaussian one-factor copula: factor shift + conditional exponential
  twisting (Glasserman & Li, Management Science 2005), t-copula variants
  (Bassamboo, Juneja & Zeevi 2008; Chan & Kroese 2010). NOT claimable.
- Routing: chance-constrained and DRO VRP evaluate route risk exactly, by
  normal approximation, worst-case bounds or plain Monte Carlo (Dinh,
  Fukasawa & Luedtke 2018; Ghosal & Wiesemann 2020; Gounaris et al. 2013;
  Juan et al. 2011; Chen et al. 2018; Sluijk et al. 2023). No IS-based
  certification of a given routing plan and no treatment of the
  mid-route peak of the pickup-and-delivery load as a rare event found.
  Closest IS precedent in logistics: Glasserman & Liu (1996) for stockouts.

## 3. Contributions to target

C1. The certification problem for routing plans at service levels
    10⁻³…10⁻⁶ (route-wise and plan-wise), as a post-optimisation step that
    any planner can use.
C2. Structure and estimator: the overflow event is a union over stages of
    sums of factor-driven demands; a two-step estimator — IS on the copula
    factors, then a stage mixture of conditional exponential twists (the
    union handled ALOE-style) — with a theorem: logarithmic efficiency (or
    bounded relative error) as Q → ∞ for gamma marginals and the one-factor
    Gaussian copula. Extends Glasserman–Li from a single sum to a maximum
    over stages; the union costs at most a factor equal to the number of
    stages.
C3. Silent failure of single-mode IS: measured bias up to 11–19% with a
    reported relative standard error of 0.2% on real routes (QUASAR A), and
    an explanation from the stage structure (start-of-route delivery
    overflow versus end-of-route pickup overflow). Practical diagnostic.
C4. Plan-level certification: the union over routes (correlated through the
    shared factors) — the same two-step machinery at plan level.
C5. A richer simulator where the exact DP does not exist (stochastic travel
    and service times with time windows, vehicle breakdowns; the event is
    again a union over stages), showing the method beyond the DP-solvable
    model.
C6. Experiments on the 148 region instances (Northern Vietnam, Guangdong;
    151 million people, OSRM road distances, demand rates from official
    postal statistics), exact DP checks where available, against crude MC,
    single-shift CE, Gaussian-mixture CE, Glasserman–Li two-step, subset
    simulation; and a short honest section on quantum amplitude estimation
    (QUASAR results: it loses to well-designed IS).

## 4. Milestones

- R1 theory (C2, C4): statements and proofs; numerical check of the
  efficiency on growing Q.
- R2 simulator (C5): specification and implementation; no DP.
- R3 experiments (C3, C6).
- R4 manuscript; venue: INFORMS Journal on Computing (first choice),
  Transportation Science.

## 5. Rules

- Every literature claim verified; snippet-only items stay in
  VERIFY_CITATIONS.md.
- Theorems labelled PROVED / PROVED+TESTED / CONJECTURE.
- Separation from TEMPO (under review): no e-values or anytime-valid
  monitoring here; certification uses classical confidence intervals.
- Roles: master designs and proves; execution in small specified modules
  with tests; review as a separate pass.
