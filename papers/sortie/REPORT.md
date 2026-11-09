# SORTIE — experiment report

*Stochastic Ordered Route fairness with Tour-consistency via Integral Evaluation.*
Experiments run 9–10 November 2026 on 4 cores (full rerun with the faster search). Every number below comes from
`results/*.csv` through `code/experiments/analyze.py`; full tables are in
`results/tables/ALL_TABLES.md`, and the paper tables are in `results/tables/paper/`.

## 1. Question

Fair vehicle routing (Ljubić, Puerto and Torrejón 2026) balances the drivers'
route lengths with an ordered measure (max, range, GMD, MAD, variance) and
requires every route to be a shortest tour of its customers. What changes
when travel times are random, and what do the drivers actually experience?

## 2. What is new

**Theory** (proofs in `sortie.tex` §4, working notes in `THEORY.md`)

1. **Three notions of fairness.** Ex-ante ℓ(EΦ), ex-post Eℓ(Φ) and the
   CVaR of unfairness. A deterministic fair model optimises the first;
   drivers experience the second.
2. **Common shocks (Prop. 1).** A fleet-wide multiplicative day factor Y
   with E Y = 1 leaves the expected value of every ordered measure unchanged.
   It inflates the variance measure by e^{s_Y²} and the tail of unfairness.
3. **PBOI (Thm. 2).** For independent route durations,
   E ℓ_λ(X) = ∫ Σ_r Λ_r π_r(t) dt. Here π is the Poisson-binomial pmf of
   #{k : X_k ≤ t} and Λ_r = Σ_{j>r} λ_j. The formula is exact for any
   weights, negative weights included, and takes 0.44 ms for m = 5.
4. **Rank prices (Prop. 3).** ϖ_k = Σ_j λ_j P(rank of k = j) is the
   derivative of expected unfairness under a shift of route k. A negative
   price means the objective wants that route longer.
5. **Fairness illusion (Prop. 4).**
   0 ≤ Eℓ(X) − ℓ(L) ≤ ‖λ‖₁ v (Σ_k W_k)^{1/2}, where W_k = Σ c² is the
   route's quadratic length.
6. **Monotonicity no longer protects drivers (Prop. 5).** Under uncertainty
   even E max can prefer a longer tour.
7. **Risk-consistent tours (Thm. 6).** The lower-level criterion is
   L + κ v √W. Its minimisers are supported Pareto tours, so dichotomic
   parametric Held–Karp with arc weights c + θc² finds them.
**Algorithm.** SORTIE is a bilevel local search.
- **Move screening.** It screens every relocate and swap move by first-order
  rank and σ prices.
- **Candidate evaluation.** The 8 best moves are evaluated with fast
  screening tours (one warm-started 2-opt/Or-opt descent) and PBOI.
- **Confirmation.** The improving candidates are then confirmed with the
  exact lower level before one is applied, so every accepted plan satisfies
  the bilevel constraint.
- **Diversification.** Otherwise it applies price-guided ruin (at most 12
  customers) and recreate, then cost repair. Changed routes are confirmed,
  and the result is accepted by simulated annealing.

Confirmation plus the ruin cap raise the iteration rate at n = 100 about 7×,
from about 250 to 1700 per minute for SORTIE-0. The old local search could
also loop forever on floating-point round-off at n = 100 with κ = 1; this is
now capped.

## 3. Set-up

- **Instances.** All instances are blocks of n + 1 consecutive nodes of
  CVRPLIB X-n641-k35 (the slicing rule of the deterministic literature), with
  m = 5, Q = ⌊Σq/4⌋ and ceiled Euclidean distances.

  | Grid | n | Blocks | Other |
  |---|---|---|---|
  | Main | 15, 20, 25, 30, 35 | 5 | |
  | Scale | 50, 75, 100 | 6 | |
  | Exact | 10 | 10 | m = 3 |

- **Uncertainty.** Gamma arc noise with coefficient of variation v = 0.3 and a
  lognormal day factor with s_Y = 0.15. In E8 the arc coefficient of
  variation is v·H_ij with H ∈ [1/3, 2], largest near a congestion centre.
- **Budget.** Σ L ≤ B·C* with B ∈ {1.05, 1.10}, where C* is the best of three
  cost-search restarts.
- **Methods.** All methods share the search, the start plan, the time limit
  and the seed.

  | Method | Objective |
  |---|---|
  | Det | Deterministic ℓ(L), the Ljubić et al. model |
  | SAA | 200 fixed scenarios |
  | SORTIE-0 | PBOI, κ = 0 |
  | SORTIE-1 | PBOI, κ = 1 |

  Time limits: 15 s for n ≤ 25, 25 s for n ≥ 30, 120 s for n ≥ 50, and 240 s
  in E7b.
- **Evaluation.** Arc-level Monte Carlo out of sample, with 20 000 scenarios
  (10 000 for n ≥ 50) and common random numbers. Paired comparisons use the
  Wilcoxon signed-rank test.

## 4. Results

### E1 — PBOI accuracy and cost (168 plans, 50 000-sample MC)

| Measure | Mean abs. err. % | Max abs. err. % | Share abs(z) < 3 |
|---|---|---|---|
| Max | 0.041 | 0.137 | 0.994 |
| Range | 0.062 | 0.249 | 1.000 |
| GMD | 0.057 | 0.241 | 1.000 |
| MAD | 0.064 | 0.272 | 0.988 |
| Var | 0.108 | 0.440 | 0.994 |

The bias is a few hundredths of a percent. It comes from the moment-matched
gamma marginals (mean z up to 0.34 for MAD).

| Evaluator | Time per plan |
|---|---|
| PBOI | 0.44 ms |
| SAA, 200 scenarios (noisy) | 0.06 ms |
| Monte Carlo, 20 000 scenarios | 0.28 s |

### E2 — exactness (n = 10, full enumeration)

The results are identical to the first run:
- The cost search found C* on 10 of 10 instances.
- SORTIE-0 found the plan with the smallest exact ex-post value on 9 of 10
  instances (mean gap 0.39%, max 3.9%).
- Det missed its own deterministic optimum on 1 of 10 instances (twice the
  optimal range).

### E3 — main grid (25 instances × 5 measures × 2 budgets)

**Fairness illusion.** Believed minus true unfairness, mean signed %.

| Measure | Det | SAA (in-sample) | SORTIE (PBOI) |
|---|---|---|---|
| Max | −8.0 | −0.7 | −0.01 |
| Range | −17.6 | −1.0 | +0.03 |
| GMD | −14.7 | −0.9 | +0.04 |
| MAD | −16.2 | −0.9 | +0.05 |
| Var | −27.2 | −1.3 | +0.00 |

**Out-of-sample ex-post value relative to Det** (pooled over 250 runs; full
table in `tab_e3`):

| Method | Mean | Median | Better | Tied | Worse | p |
|---|---|---|---|---|---|---|
| SORTIE-0 | +0.89% | +0.23% | 141 | 96 | 13 | 3e-19 |
| SORTIE-1 | +0.77% | | 154 | | 31 | |
| SAA | +0.14% | 0 | 115 | | 74 | |

- **Head to head.** SORTIE-0 beats SAA in 145 runs and loses in 15; SORTIE-1
  wins 147 and loses 27 (p < 1e-17).
- **Significance by cell.** SORTIE-0 and SORTIE-1 are each significant in 9 of
  10 cells, SAA in 3.
- **Largest cell.** Max at B = 1.10: +2.1% (24/25).
- **CVaR90 of the range** (range-optimised plans):

  | Det | SAA | SORTIE-0 | SORTIE-1 |
  |---|---|---|---|
  | 2054 | 2059 | 2037 | 2032 |

- **Fairness itself.** The cost plan has an ex-post range of 2087, against
  about 1340 for the fair plans, at +6.5% cost.

### E4 — route consistency (B = 1.10)

- **Detours appear without the lower level.** For the range, 23 of 25 Det
  plans and 24 of 25 SORTIE plans contain a route that is not a shortest tour,
  by 95 (Det) and 113 (SORTIE) distance units on average over runs with a detour.
- **Effect on the ex-post value.** Detours lower it by 1–3%, except SORTIE on
  the GMD, which ends 0.9% worse because of the larger search space.
- **Max.** Free sequencing barely changes anything.
- **Beliefs.** Det's belief error persists with free tours (−24% for the
  range), while PBOI's stays exact.

### E5 — rank-price screening vs random moves

| Measure | Mean | Median | Better / worse (of 25) | p |
|---|---|---|---|---|
| Range | 5.6% | 0% | 12 / 3 | 0.005 |
| GMD | 8.5% | 5.4% | 17 / 2 | 5e-4 |

### E6 — noise level v (n = 25, 5 instances)

| v | Det belief error, range | SORTIE-0 vs Det, range | SORTIE-0 vs Det, GMD |
|---|---|---|---|
| 0.1 | −8.7% | −0.5% (search noise) | +1.1% |
| 0.3 | −28.7% | +1.0% | +0.7% |
| 0.5 | −42.7% | +2.1% (5/5) | +1.8% |

### E7 — n = 50, 75, 100 (120 s, 6 instances × {range, GMD, max})

- **Iterations per run.** SORTIE-0 runs about 9000 at n = 50 and about 3500
  at n = 100. With the old code, at 60 s, it was about 1500 and 250.
- **Accuracy.** PBOI's mean absolute error is 0.14% (at most 0.9%). Det's
  belief error is −17%.
- **Pooled improvement over Det** (18 pairs per n):

  | n | SAA | SORTIE-0 | SORTIE-1 |
  |---|---|---|---|
  | 50 | +1.6% (12/18) | +2.0% (13/18, p = 0.035) | +1.2% (15/18, p = 0.021) |
  | 75 | +2.0% (11/18) | +2.0% (13/18, p = 0.17) | −0.3% |
  | 100 | +1.5% (10/18) | +2.2% (12/18, p = 0.17) | −0.0% |

- **Max** is consistent at every size (SORTIE-0 better on 5–6 of 6).
  **Range and GMD** single runs swing ±5–10%.
- **SORTIE-1** is too slow at n ≥ 75: about 1150 iterations at n = 75 and
  840 at n = 100; 39% and 56% of its runs still improved by more than 0.5% in
  the second half.

### E7b — n = 100, three solver seeds, 240 s (Det, SAA, SORTIE-0)

Each method is represented by its mean ex-post value over seeds.

| Measure | SAA vs Det | SORTIE-0 vs Det | Seed CV Det | Seed CV SAA | Seed CV SORTIE-0 |
|---|---|---|---|---|---|
| Range | +6.5% (4/6) | +8.7% (5/6) | 11.3% | 4.7% | 3.0% |
| GMD | +2.5% (3/6) | +3.2% (4/6) | 6.4% | 2.7% | 4.5% |
| Max | +0.8% (4/6) | +0.9% (4/6) | 1.4% | 0.8% | 1.4% |
| **All 18** | +3.2% (11/18, p = 0.10) | **+4.2% (13/18, p = 0.021)**, median 2.5% | | | |

- **SORTIE-0 vs SAA:** +1.1% (12/18), p = 0.25, not significant.
- **Det is the noisiest method for the range** (CV 11%). The deterministic
  range is flat over many plans whose realised fairness differs.
- **Convergence.** 19% of SORTIE-0 runs still improved by more than 0.5% in
  the second half (120–240 s).

**Verdict for n = 100.**
- SORTIE vs the deterministic fair plan is conclusive once search noise is
  averaged over seeds.
- SORTIE vs SAA is not separable at this size.
- No single measure is significant on six instances.

### E8 — congested (heterogeneous) noise, B = 1.10, 25 instances × {range, GMD, max}

- **The illusion grows somewhat.** Det's belief error is −25.7% for the range,
  −21.7% for the GMD and −10.1% for the max, against −23.1%, −19.3% and −8.9%
  on the same cells (B = 1.10) under homogeneous noise. PBOI's mean signed
  error stays below 0.2%.
- **The gain grows only slightly.**

  | Method vs Det | Mean | Median | Better / worse (of 75) | p |
  |---|---|---|---|---|
  | SORTIE-0 | +1.51% | +1.24% | 53 / 3 | 2e-9 |
  | SORTIE-1 | +1.66% | | 57 / 10 | |
  | SAA | +1.28% | +0.47% | | |

- **Same cells without congestion** (SORTIE-0): +1.36% overall.

  | Measure | Without congestion | With congestion |
  |---|---|---|
  | GMD | 0.94% | 1.40% |
  | Max | 2.08% | 2.15% |
  | Range | 1.07% | 0.96% |

- **Head to head**, SORTIE-0 beats SAA in 49 of 75 runs (worse in 6,
  p = 5e-7).
- **Correction.** The first run's claim that the gain doubles under
  congestion did not survive the rerun.

## 5. Summary

1. **The fairness illusion is large and certain.** Deterministic fair plans
   understate realised unfairness by 8–27% at v = 0.3, 43% at v = 0.5 and 26%
   for the range under congestion. PBOI removes the illusion at 0.4 ms per
   plan.
2. **Optimising ex-post fairness helps, but modestly.** The gain over Det is
   about 1% at n ≤ 35, about 1.5% under congestion and about 2% at
   n = 50–100.
   - It is significant in the pooled E3 and E8 comparisons and at n = 50.
   - At n = 75 and n = 100 single runs are not significant (p = 0.17).
   - At n = 100 it becomes significant once seeds are averaged (pooled
     p = 0.021), but no single measure is significant on six instances.
   - It clearly beats SAA at n ≤ 35 and under congestion, but not at
     n = 100.
3. **Rank prices help the search.** Plans are 6–9% better than with random
   screening.
4. **The bilevel constraint matters.** Without it, fairness is bought with
   detours.
5. **The deterministic objective makes the search erratic.** At n = 100 its
   flat range objective gives an 11% seed-to-seed spread in realised fairness.

## 6. Limitations

- The lower level is exact only for routes with at most 15 customers. Beyond
  that it uses multi-start local search warm-started from the screening tour,
  so the bilevel constraint holds approximately.
- PBOI assumes route durations that are independent given the day factor.
  Spatially correlated congestion would break this.
- At n = 100 single runs are noisy, and 19% are not converged after 240 s.
- SORTIE-1 (κ = 1) does not converge at n ≥ 75.
- The time limits are soft: the final iteration may overrun.
- All instances come from a single benchmark file with m = 5.

## 7. Reproduce

See `README.md`. In short: `cd code && bash experiments/run_all.sh && python experiments/analyze.py`.
