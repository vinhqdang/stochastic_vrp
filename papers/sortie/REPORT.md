# SORTIE — experiment report

*Stochastic Ordered Route fairness with Tour-consistency via Integral Evaluation.*
Experiments run 9 November 2026 on 4 cores. Every number below comes from
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
   weights, negative weights included, and takes 0.46 ms for m = 5.
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

**Algorithm.** SORTIE is a bilevel local search. It screens every relocate
and swap move by first-order rank and σ prices, and evaluates the 8 best
moves exactly (lower-level tours plus PBOI). It falls back to price-guided
ruin and recreate under simulated annealing.

## 3. Set-up

- **Instances.**
  - Main grid: blocks of n + 1 consecutive nodes of CVRPLIB X-n641-k35 (the
    slicing rule of the deterministic literature), with m = 5,
    Q = ⌊Σq/4⌋ and ceiled Euclidean distances. n ∈ {15, 20, 25, 30, 35} × 5
    blocks.
  - Scale grid: n ∈ {50, 75, 100} × 3 blocks.
  - Exact grid: n = 10 and m = 3 on 10 blocks.
- **Uncertainty.** Gamma arc noise with cv v = 0.3 and a lognormal day factor
  with s_Y = 0.15. In E8 the arc cv is v·H_ij with H ∈ [1/3, 2], largest near
  a congestion centre.
- **Budget.** Σ L ≤ B·C*, with B ∈ {1.05, 1.10}. C* is the best of three
  cost-search restarts.
- **Methods.** All methods share the search, the start plan, the time limit
  (15 s for n ≤ 25, 25 s for n ≥ 30, 60 s for n ≥ 50) and the seed.
  - Det: deterministic ℓ(L), the Ljubić et al. model.
  - SAA: 200 fixed scenarios.
  - SORTIE-0: PBOI with κ = 0.
  - SORTIE-1: PBOI with κ = 1.
- **Evaluation.** Arc-level Monte Carlo, out of sample, with 20 000 scenarios
  (10 000 for n ≥ 50) and common random numbers. Paired comparisons use the
  Wilcoxon signed-rank test.

## 4. Results

### E1 — PBOI accuracy and cost (168 plans, 50 000-sample MC)

| Measure | Mean abs. err. % | Max abs. err. % | Share abs(z) < 3 |
|---|---|---|---|
| Max | 0.041 | 0.137 | 0.994 |
| Range | 0.063 | 0.249 | 1.000 |
| GMD | 0.058 | 0.241 | 1.000 |
| MAD | 0.065 | 0.272 | 0.994 |
| Var | 0.109 | 0.440 | 1.000 |

PBOI has a small systematic bias of a few hundredths of a percent, which
comes from moment-matching each route by a gamma distribution. Timing per
plan:

| Evaluator | Time per plan | Notes |
|---|---|---|
| PBOI | 0.46 ms | exact, no sampling noise |
| SAA (200 scenarios) | 0.07 ms | noisy |
| Monte Carlo (20 000 scenarios) | 0.28 s | |

### E2 — exactness (n = 10, full enumeration)

- The cost search found C* on 10 of 10 instances.
- SORTIE-0 found the plan with the smallest exact ex-post value on 9 of 10
  instances. The mean gap is 0.39% and the maximum 3.9%. Range and GMD give
  identical results, because the GMD is (4/9)·range for m = 3.
- Det missed the deterministic optimum on one instance, ending at twice the
  optimal range. That instance has only 20 budget-feasible partitions.

### E3 — main grid (25 instances × 5 measures × 2 budgets)

**Fairness illusion.** Each cell is believed minus true unfairness, as a
mean signed %.

| Measure | Det | SAA (in-sample) | SORTIE (PBOI) |
|---|---|---|---|
| Max | −8.0 | −0.7 | −0.01 |
| Range | −17.0 | −0.8 | +0.02 |
| GMD | −14.6 | −0.8 | +0.04 |
| MAD | −16.3 | −1.0 | +0.05 |
| Var | −26.5 | −1.6 | +0.00 |

**Out-of-sample ex-post value relative to Det** (`tab_e3`), pooled over 250
runs:

| Method | Mean improvement | Better | Tied | p |
|---|---|---|---|---|
| SORTIE-1 | +0.92% | 155 | 69 | 1e-17 |
| SORTIE-0 | +0.77% | 133 | 102 | 2e-17 |
| SAA | +0.60% | — | — | median 0 |

- SORTIE-0 and SORTIE-1 each beat SAA head to head (138/250 and 150/250 runs,
  p < 1e-14).
- The largest cell is Max at B = 1.10: +2.1%, with SORTIE better on 24 of
  25 instances.
- For the variance measure only the SORTIE variants improve on Det.

**Interpretation.**
- Fairness optimisation itself matters: the cost-optimal plan has an ex-post
  range of 2075, against about 1340 for any fair plan, at +6.6% cost.
- The stochastic refinement adds about 1% on top. Under homogeneous noise a
  route's variance grows with its length, so balancing lengths roughly
  balances risk.
- The deterministic plan's *assessment* of its fairness is badly wrong
  (−8% to −27%), even though its *choice* is close to the best.
- CVaR90 of the range (range-optimised plans): Det 2054, SAA 2046,
  SORTIE-0 2047, SORTIE-1 2034. Only SORTIE-1 improves it at both budgets.

### E4 — route consistency (free sequencing vs lower-level tours, B = 1.10)

- Without the lower level, 23 of 25 Det plans and 25 of 25 SORTIE plans for
  the range contain a route that is not a shortest tour. The excess is 92–115
  distance units on average.
- These detours lower the measured ex-post range by about 2%. That is
  fairness bought by making drivers drive detours, which the bilevel
  constraint excludes.
- For the max, free sequencing barely changes anything.

### E5 — rank-price screening vs random moves

Price screening gives better plans than random screening, at the same number
of iterations:

| Measure | Mean | Median | Better | Worse | p |
|---|---|---|---|---|---|
| Range | 6.6% | 1.7% | 14/25 | 1 | < 1e-3 |
| GMD | 7.2% | 2.5% | 15/25 | 0 | < 1e-3 |

### E6 — noise level v (n = 25)

| v | Det belief error, range | SORTIE-0 vs Det, range |
|---|---|---|
| 0.1 | −8.7% | +0.25% (2/5) |
| 0.3 | −28.7% | +0.96% (3/5) |
| 0.5 | −42.5% | +1.75% (5/5) |

The illusion and the gain both grow with v.

### E7 — n = 50, 75, 100 (60 s)

- PBOI stays accurate: mean absolute error 0.2%, at most 1%. The belief error
  of Det is about −22%.
- The search does not scale yet. Routes have 10–20 customers, so the lower
  level falls back to local search, and only 100–400 iterations run at
  n ≥ 75.
- The method differences are dominated by search noise: −65% to +40% per
  run. SORTIE-0 beats Det on 5/6 runs at n = 50 and at n = 100, but on 1/6
  at n = 75.
- No conclusion about the objective can be drawn at this size.

### E8 — congested (heterogeneous) noise, B = 1.10, 25 instances × {range, GMD, max}

- **Illusion.** Det understates its unfairness by more than under
  homogeneous noise: range −24.7%, GMD −21.2%, max −10.1%. PBOI stays
  within 0.2% on average.
- **Gain of ex-post optimisation over Det** (pooled over 75 runs):

  | Method | Mean | Median | Better | Worse | p |
  |---|---|---|---|---|---|
  | SORTIE-0 | 1.88% | 1.14% | 52 | 3 | 3e-10 |
  | SORTIE-1 | 1.85% | 1.51% | 56 | 11 | 3e-9 |
  | SAA | 1.27% | — | — | — | — |

- **Comparison with the same cells without congestion (E3).** For SORTIE-0,
  pooled 1.4% → 1.9%:

  | Measure | E3 | E8 |
  |---|---|---|
  | GMD | 0.74% | 1.36% |
  | Range | 1.34% | 2.23% |
  | Max | 2.11% | 2.03% |

- SORTIE-0 beats SAA head to head in 50 of 75 runs (worse in 4).
- CVaR90 of the range: Det 1932, SAA 1912, SORTIE-0 1907, SORTIE-1 1904.

When a route's risk is not a function of its length, balancing lengths no
longer balances realised workloads. That is the regime in which the
stochastic model pays.

## 4b. Summary

- **The illusion is large and certain.** Deterministic fair plans understate
  realised unfairness by 8–27% (up to 43% at v = 0.5). PBOI removes the
  illusion at 0.46 ms per plan.
- **The gain is real but modest.** It is about 1% under homogeneous noise
  and about 2% under congestion, statistically significant in both cases.
- **Rank prices matter for the search.** They give 7% better plans than
  random screening (2% at the median).
- **The bilevel constraint matters.** Without it, fairness is bought with
  detours.
- **Scaling.** The search does not yet scale beyond n ≈ 50 within 60 s.
  The lower level is the bottleneck.

## 5. Limitations

- The lower level is exact only for routes with at most 15 customers.
  Beyond that it is local search, so the bilevel constraint holds only
  approximately.
- PBOI assumes route durations that are independent given the day factor.
  Shared congested arcs would induce dependence that it ignores.
- The time limits are soft: the final iteration may overrun.
- Instances are slices of a single CVRPLIB file with m = 5.
- PBOI's advantage over SAA is exactness, not speed. A 200-scenario SAA is
  faster per call but noisy, and optimistic in-sample by about 1%.

## 6. Reproduce

See `README.md`. In short: `cd code && bash experiments/run_all.sh && python experiments/analyze.py`.
