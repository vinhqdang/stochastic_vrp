# EXPERIMENT C — which information, how much dependence, a model-based rival, and the cut baselines

Status: pre-registration, **revision 2 (2026-11-25)**, after the independent review in
`REVIEW_C.md` (accept with required changes; the mapping is in section 11). Written after
experiments A and B were analysed and before any C code exists. It answers the internal
referee report on the TS draft: (1) which features carry the gain; (2) how the gain scales
with the dependence between stops; (3) whether a model-based policy that sees the rich
information through the right model class matches the learners; (4) the baselines B could
not run (deep optimal stopping) or ran at one budget (PPO). Conventions are A's and B's
(cost = recourse cost; unit = instance; t intervals; exact Wilcoxon; Holm) unless stated.
**[DC]** marks a design choice without evidence. After the pilot (section 8) sections 1–7
are frozen; later changes go to RESEARCH_LOG.md with date and reason and are reported as
deviations. A's and B's test streams are not read again; C uses a new gated stream
**test_c** (section 5).

## 1. Questions, estimands, decision rules

### 1.1 Questions

- **Q1 (which information).** With the learner fixed, which feature group carries the gain
  of the rich state over the load?
- **Q2 (dose–response).** How do the gain of the rich state over BATON and the loss of the
  independence-based DP (THR) change with the copula correlation, and with a single copula
  shared by pickups and deliveries?
- **Q3 (model-based rival).** Does a DP that uses the right model class (Gaussian copula,
  fitted on the training paths) and the rich information (posterior of the common factors)
  reach FULL's cost?
- **Q4 (baselines).** How do deep optimal stopping (DOS, cut in B) and PPO at 3.6 times its
  source's budget compare with FULL?

### 1.2 Learners

A and B showed that on the rich state a perceptron, boosted trees (BOOST; FULL vs BOOST
+0.25%), randomized least squares (RLSM; +0.16%) and DQN are equivalent to FULL within
0.7%, but BOOST and RLSM are consistently slightly dearer. **BOOST and RLSM are co-primary
learners for Q1** (both fit in under a minute per route); Q1 is answered per learner and
claimed only where both agree. FULL is not refitted: B's frozen FULL models are evaluated
on test_c (section 4.3) and are the reference of Q3 and Q4.

### 1.3 Feature sets (Q1) — nested column masks, fixed now

Global features g1–g12 as in A §3.2 (`features.stage_batch`); token summaries s1–s3 as in A
§3.4 (`mlp_batch` columns 13–15). Masks act on the same arrays; no new features. Models are
fitted per route and per stage.

| set | globals | token summaries | adds |
|---|---|---|---|
| F0 (load) | g1, g3–g7 | s1–s3 | functions of (k, W_k), prices, route data = A's SCALAR information |
| F0n (net history) | F0 + g2 (running max), g12 (last increment) | s1–s3 | net-load history, no split |
| F1 (split) | F0 + g8 (P_k/B), g9 (D_k/B) | s1–s3 | the pickup/delivery split |
| F2 (split + deviations) | F1 + g10, g11 | s1–s3 | standardised deviations (per stage, a re-weighting of P/D) |
| F3 (full) | F2 + g2, g12 (all 12) | s1–s3 | everything |

BOOST: `HistGradientBoostingRegressor` with A's M11 parameters on the masked columns (in C,
BOOST receives s1–s3 for every set; A's BOOST had no token summaries) [DC]. RLSM: B's chosen
configuration (K = 100, ridge 1e-6·n) on the 15 `mlp_batch` inputs with masked columns set
to zero, so the random layer is the same for every set (same seed per stage and fold).

### 1.4 Worlds (Q2) — fixed now

All worlds keep A's marginals (gamma, CV 0.3, same means) and change only the copula.

| world | definition |
|---|---|
| W0 | ρ = 0 for deliveries and pickups (independent stops) |
| W3 | ρ = 0.3, separate delivery and pickup copulas |
| W6 | ρ = 0.6, separate copulas: A's and B's generator |
| W9 | ρ = 0.9, separate copulas |
| WS | one equicorrelated Gaussian copula, ρ = 0.6, over the vector (deliveries of the active customers, then pickups of the active customers) of the instance: every pair, including the delivery and pickup of one customer, has correlation 0.6. Drawn by one `standard_normal((N, 2m'))` call per stream |

The 62 eligible routes of A are used in every world (eligibility is not re-applied). A
table reports, per world, the breach rate without recourse and the number of routes with
no breach on test_c.

### 1.5 Estimands

- **Family contrasts in W6** (all 62 routes breach ≥ 2% on dev): A's Δ_r, Δ_i, Δ̄ (route
  percentage of b's cost, averaged over routes, then instances).
- **Cross-world contrasts (Q2)**, because some W0 routes never breach: the instance ratio
  of summed costs, Δ_i^Σ = 100 (Σ_r C_b,r − Σ_r C_a,r) / Σ_r C_b,r over the instance's
  eligible routes, and Δ̄^Σ its mean over instances. An instance with Σ_r C_b,r <
  10⁻³·Σ_r E0_r is excluded from that contrast and counted in the table. W6 values are
  reported in both forms.
- **Share of gain (Q1, descriptive with CI):** for learner L, share(F) = Δ̄(L-F vs L-F0) /
  Δ̄(L-F3 vs L-F0), with an instance bootstrap 95% CI (10,000 draws, seed 20261125).

### 1.6 Margins

- **δ = 0.7%** (A's) for contrasts against BATON, CDP, DOS (Q3, Q4).
- **δ_Q1 = 0.25%** for contrasts between feature sets of one learner [DC]: about one third
  of the rich-state gain (0.82%), so that "equivalent" means "less than a third of the
  gain" and "better" means "at least a third". Classes as A §1.5 with δ_Q1.

### 1.7 CDP — copula-conditional DP (Q3), fixed now

- **Model fit (all N training paths):** per stop j, gamma marginals of d_j and p_j by moments
  (zero-mean stops stay zero). Normal scores z = Φ⁻¹(F̂(x)) per stop; ρ̂_d = mean of the
  off-diagonal sample correlations of the delivery scores over the route's active stops
  (ρ̂_p likewise), clipped to [0, 0.99]. The model has separate delivery and pickup factors
  (it is mis-specified in WS).
- **Posterior** (z_j = √ρ Z + √(1−ρ) ε_j): after k observed active stops,
  Z | z_1..z_k ~ N(μ_k, s_k²), μ_k = √ρ Σ z_j / (1 − ρ + kρ), s_k² = (1 − ρ)/(1 − ρ + kρ).
  Observed scores come from p_j, d_j recovered as differences of the cumulative P and D.
- **Tables:** factor nodes ζ on a uniform grid of 25 points in [−4, 4] per factor (625
  pairs). For each pair and each stop, the conditional law of g_j is represented by
  S = 256 scrambled-Sobol draws of ε (seed = job seed) mapped through the fitted marginals;
  the THR backward recursion on a 1,001-point load grid gives C_k(w | ζ_d, ζ_p).
- **Decision at stop k:** Ĉ_k = E over the posterior of C_k(W_k | ζ), computed by 9 × 9
  Gauss–Hermite points with bilinear interpolation of the tables in (ζ_d, ζ_p) and linear
  interpolation in w; hand off iff Ĉ_k ≥ H_k − 10⁻⁶E0 (B's tie rule).
- **Approximation (stated in the paper):** each table assumes the factor known from now
  on, so the average uses more information about the future than the driver will have;
  CDP under-estimates the continuation cost and is biased towards handing off too late. It
  is exact (up to marginal fit and grids) when ρ = 0, where it reduces to THR. It is a
  model-based baseline, not the Bayes-optimal policy.
- **Runs:** CDP in every world (W0, W3, W6, W9, WS); in W6 also **CDP-true** with the
  generator's ρ = 0.6 plugged in (sensitivity, descriptive).
- If the pilot measures more than 10 min per route-replicate, the factor grid becomes 17
  points per factor (logged).

### 1.8 Baselines (Q4), fixed now

- **DOS:** B's implementation (`expb/dos.py`, unchanged since its tuning commit `155cb19`),
  configuration chosen by B's rule 6.1 from B's committed dev tuning CSVs; before the fit,
  `choose` is run for DOS and the record (scores, hashes of the rows) is committed. The
  scores reproduced for this spec are c0 −0.60, c1 +1.55 relative to BATON (c1 driven by
  two routes), so c0 is expected. W6, 62 routes, reps 0–1, 400 steps per stage.
- **PPO-long:** B's PPO, chosen configuration (lr 1e-4, γ = 1), **9 passes** (7.2·10^5
  episodes per route, 3.6 times the source's 2·10^5) [DC], W6, the 12-route subset, rep 0.
  B's PPO models (3 passes) on the same routes are evaluated on test_c for the comparison.
  Descriptive.

### 1.9 Family C and the other contrasts

**Family C** — Holm over its 11 members, two-sided α = 0.05; classes as A §1.5:

| id | contrast (a vs b) | world | margin |
|---|---|---|---|
| C1-B, C1-R | L-F1 vs L-F0 (L = BOOST, RLSM) | W6 | δ_Q1 |
| C2-B, C2-R | L-F1 vs L-F0n | W6 | δ_Q1 |
| C3-B, C3-R | L-F2 vs L-F1 | W6 | δ_Q1 |
| C4-B, C4-R | L-F3 vs L-F2 | W6 | δ_Q1 |
| C5 | CDP vs BATON | W6 | δ |
| C6 | FULL vs CDP | W6 | δ |
| C7 | FULL vs DOS | W6 | δ |

A Q1 statement ("the split carries the gain") is made only if C1 is "better" for both
learners. No predictions are used for decisions.

**Descriptive (unadjusted):**
- Q1: F0n vs F0 per learner; share(F) for F0n, F1, F2 per learner with bootstrap CI.
- Q2: per world, Δ̄^Σ and CI95 of BOOST-F3 vs BATON, RLSM-F3 vs BATON, BOOST-F0 vs BATON,
  BATON vs THR (positive = BATON cheaper), CDP vs BATON; the breach table. Reported as a
  pattern across worlds, without a trend test.
- Q3: CDP vs THR (same DP engine) per world; CDP-true vs CDP in W6.
- Q4: PPO-long vs FULL and vs B's PPO on the subset (rep 0, 10 instances).
- The share of the BATON-to-oracle gap closed by FULL in W6:
  (C_BATON − C_FULL)/(C_BATON − C_oracle), summed over routes then averaged over instances.

## 2. Problem and interface

Unchanged (A §2, B §2). Every new arm deploys through `simulate.simulate_policy`; CDP
returns Ĉ_k + τ (τ = 1e-6·E0) like THR.

## 3. Data and worlds

- `data.gen_scenarios` / `sample_demands` gain a world argument. **W6 with default
  arguments reproduces the current streams bit for bit** (test: arrays equal for A's train
  chunk 0, dev and test_b seeds of CON3-0, without reading test values into any result).
- Seeds: W6 uses A's seeds for train (rep r, chunk 0) and dev. Other worlds: train
  `[base, 1, r, c, w]`, dev `[base, 3, w]`; test_c `[base, 7, w]` for every world including
  W6; w = 10 (W0), 13 (W3), 16 (W6), 19 (W9), 26 (WS) [DC]. A test compares the **draws**
  (first rows of every stream) for distinctness from every A/B stream and from each other,
  except W6 train/dev, which equal A's by design. Chunk cache keys include the world.
- Training size N = 10^5 (rep r, chunk 0 prefix), reps 0–1 (PPO-long rep 0).

## 4. Stages

| stage | world | arms | routes | reps |
|---|---|---|---|---|
| c_ladder_w{0,3,6,9,S} | each | none, baton_eq, oracle | 62 | 0–1 |
| c_thr_w{0,3,6,9,S} | each | THR | 62 | 0–1 |
| c_boost_{F0,F3}_w{0,3,9,S}, c_rlsm_{F0,F3}_w{0,3,9,S} | W0, W3, W9, WS | BOOST, RLSM | 62 | 0–1 |
| c_boost_{F0,F0n,F1,F2,F3}_w6, c_rlsm_{F0,F0n,F1,F2,F3}_w6 | W6 | BOOST, RLSM | 62 | 0–1 |
| c_cdp_w{0,3,6,9,S}, c_cdptrue_w6 | each / W6 | CDP | 62 | 0–1 |
| c_dos_w6 | W6 | DOS | 62 | 0–1 |
| c_ppolong_w6 | W6 | PPO, 9 passes | 12 subset | 0 |
| c_full_w6 (eval only) | W6 | B's frozen FULL models | 62 | 0–1 |
| c_ppob_w6 (eval only) | W6 | B's frozen PPO models | 12 subset | 0 |

4.3 **Reused models.** c_full_w6 and c_ppob_w6 have no fit; their manifests are B's
(`b_full`, `b_ppo` models.sha256) and their fit commit is B's F (`bf3edab`). The gate
accepts these two stages with fit commit bf3edab provided that (a) every model file hashes
to B's manifest, (b) a test shows that B's FULL and PPO predictors give bit-identical
predictions on W6 dev data at bf3edab and at C's fit commit (features and data code may
change only in ways that leave W6 unchanged), and (c) the library versions recorded in the
payloads equal the running ones.

## 5. Stream test_c and the gate

Every stage is evaluated on its world's test_c (20,000 days); all Q2 numbers come from
test_c. test_c is a new gated purpose with B's machinery generalised: `FROZEN COMMIT
(test_c)` lines and one `TEST_C FAMILY CLOSED: family=[...] descriptive=[...]
dropped=[...] references=[...] fidelity_sha256=... manifests_sha256=... code_sha256=...
analysis_sha256=...` line, printed by `gatecheck family-line --stream test_c --commit M`
and committed in a log-only commit L. The commit sequence is B §5.3 with test_b replaced
by test_c. CODE changes after C's fit commit F_C are forbidden except a logged bug repair
(one per arm).

## 6. Gates (dev only, before the family line)

- **G0 (all fitted stages):** every planned job fitted, finite, cost ≥ oracle (B0).
- **G-W6:** W6 train/dev arrays equal A's (hash of CON3-0 dev and train rep 0 chunk 0).
- **G-CDP0:** in W0, CDP vs THR per route on dev: |C̄_CDP − C̄_THR| ≤ 4 SE + 10⁻³E0, SE of
  the paired per-day difference (replicate-averaged), on ≥ 95% of routes [DC: the floor
  10⁻³E0 removes B's zero-width band; 95% because two discretisations are compared].
- **G-CDP-node:** for each route (rep 0), the node (ζ_d, ζ_p) = (0.5, −0.5): simulate
  20,000 paths from the fitted conditional model, run the node's own DP policy, and compare
  the mean cost with the table value C at k = 0, within 4 SE + 10⁻³E0 on ≥ 95% of routes.
- **Unit tests (not gates):** posterior vs brute-force Gaussian conditioning on 1,000
  random cases to 1e-9; tables nondecreasing in w per node; ρ = 0 CDP equals THR on a toy
  route; interpolation edges.
- **G-DOS:** B's B-DOS gate. **G-PPO:** B's B-PPO gate on c_ppolong_w6.
- **P1 (all learned and DP arms in W6):** B's P1 rule.
A gate failure allows one bug repair per arm (logged, with a test); otherwise the
contrasts involving that arm leave the family and are reported descriptively.

## 7. Statistics

As B §8 for family C (Δ̄, median, CI95, CI90, Holm-level interval, n better, exact Wilcoxon
two-sided p, Holm over 11). Cross-world contrasts with Δ̄^Σ (1.5). Unit instance (n = 20;
PPO-long and its comparison n = 10).

## 8. Pilot (dev only) and compute

Pilot: CON3-0's eligible routes, every C stage except c_ppolong_w6, at N = 10^5, rep 0,
to measure fit times and check that every stage runs. No decision depends on its costs.

Projection (review estimate from B's measured fit times; the pilot replaces it): about 15 h
wall for all stages except PPO-long with 12 workers, plus PPO-long about 22 h wall (9
passes at 12.8–21.5 h per route, 12 routes in parallel). PPO-long runs concurrently on the
remaining cores, so the total is about 22–25 h; sequentially about 37 h. Ceiling 72 h;
cut order if exceeded: (i) RLSM in W0/W3/W9/WS, (ii) WS, (iii) PPO-long to 6 passes.

## 9. Implementation modules (Haiku implements with mutation-checked tests; Sonnet reviews)

- **MC1 worlds:** world-aware generator and seeds (section 3), purpose test_c with gate
  token, cache key with world, W6 bit-identity and draw-distinctness tests.
- **MC2 feature sets:** masks; BOOST and RLSM fit functions take a set; tests that F3
  equals the unmasked arrays, masked columns do not affect predictions, RLSM's random
  layer is identical across sets.
- **MC3 CDP:** `expc/cdp.py` (fit, posterior, tables, predictor) with the unit tests of
  section 6 and the two gates.
- **MC4 stages/sweep:** `expc/stages.py`; world and set through StageSpec, jobs, payloads,
  merge; eval-only stages for reused models with the 4.3 checks; DOS choose record.
- **MC5 gates and analysis:** gatecheck/fidelity generalised to test_c; `expc/analysis_c.py`
  for family C and the descriptive tables; manuscript tables read its CSVs.

## 10. Threats

- Synthetic one-parameter copula family; fixed plans across worlds (plans were made for
  W6's demand means, which do not change); routes selected on W6 breach rates (W0 routes
  breach less). All stated in the paper.
- CDP is one model-based design among several; its known bias (1.7) is stated, and CDP-true
  separates estimation error of ρ from the approximation.

## 11. Changelog

- Revision 2 (2026-11-25), answering REVIEW_C.md: (1) δ_Q1 = 0.25% and share-of-gain for
  Q1 (finding 1); (2) summed-cost estimand across worlds with an exclusion floor and a
  breach table (2); (3) RLSM co-primary, C6 = FULL vs CDP (3); (4) CDP grids 25 × 25 nodes,
  Gauss–Hermite posterior expectation with interpolation, S = 256, 1,001-point load grid,
  bias stated, CDP-true sensitivity (4); (5) G-CDP0 with paired SE and floor, node gate at
  ρ > 0, p/d recovered from P and D (5); (6) test_c scope, B's PPO eval-only stage, gate
  exception for reused models with identity and library checks (6); (7) DOS choose record
  committed before the fit (7); (8) Q1 contrasts split into F1 vs F0, F1 vs F0n, F2 vs F1,
  F3 vs F2; RLSM zero-masking with a fixed random layer (8); (9) WS defined, distinctness
  tested on draws, cache key with world (9); (10) projection added, pilot excludes PPO-long
  (10); (11) family of 11, CDP vs THR descriptive, gap-closed denominator (11); suggested:
  CDP and RLSM in all worlds added; a factor-known ceiling and a second copula family are
  not added (the factor is not stored by the generator; scope).
- Revision 1 (2026-11-25): first version.
