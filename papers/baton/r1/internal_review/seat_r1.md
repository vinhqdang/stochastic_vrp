# Seat R1 — Methodology (design and statistical analysis of computational experiments)

## Seat and angle
I specialise in how computational experiments in stochastic optimisation are designed and analysed. For this revision I checked R1.4, R1.5, R1.6, R1.2 (the capacitated pool), R2.M1 (the rho sweep and the shape test), R2.M2, R2.m3, R2.M6 and R3.5. I read `make_tables.py`, `scripts/run_baton_r1.py`, `scripts/run_realistic_eval.py` and `core/extra_policies.py`, and recomputed the key numbers from `results/*.csv` and `results/r1/*.csv`. To diagnose the position-dependent threshold I also ran one read-only check that wrote no files.

## Verdicts on round-1 comments

| item | verdict | evidence anchor | residual gap |
|---|---|---|---|
| R1.2 shared standby | PARTLY_RESOLVED | §3.1 "Standby versus emergency"; §4.6, Table 12 | The pool is capped only within one plan. S⋆=0 then follows almost by arithmetic: the holding cost F_sb=20 exceeds the whole plan's daily recourse bill on Dethloff (8.3). The "naive" baseline is a straw man (see N4). The regime the paper recommends, one pool shared across plans, is never simulated under a cap. |
| R1.4 flexible thresholds | PARTLY_RESOLVED | §4.1 Competitors; Table 2 columns thr.-k and roll.-θ | The benchmarks exist, but the position-dependent threshold is under-optimised, not "high-variance" (N1). The added value of backward induction over it is therefore not isolated yet. |
| R1.5 computational fairness | RESOLVED | §4.1 Protocol; Table 14; §4.7 | Every policy gets the same 10^3 training days. Offline and online times are reported separately. Only small wording slips remain (C3, C4). |
| R1.6 statistics | RESOLVED (headline) | §4.1 "Statistical analysis"; Table 3; Table 4 | The code is correct: percentile bootstrap with 10^4 resamples, Holm step-down over 18 comparisons, correct rank-biserial. The new experiments in Tables 6 and 9–12 still carry no interval or test (N5). |
| R2.M1 dependence / Prop. 3–4 | PARTLY_RESOLVED | §3.2 after Assumption 1; §3 opening (finite-tree formalisation); §4.5, Table 9 | The sweep over rho ∈ {0, .3, .6, .9} plus a day-factor law is done well. The shape test that justifies the isotonic step at rho=0.6 has no demonstrated power, and it was not run on the one law where monotonicity is most doubtful (N3). |
| R2.M2 fresh-start bias | RESOLVED | §3.5 biases (i)–(iii); Table 11 | The bias is quantified on Salhi–Nagy, as requested, and Baton-cf removes bias (iii). Caveats: the "near-exact F_k" is itself a policy-based value from a binned DP, and the bias is a signed mean over stops, so errors can cancel (N6). |
| R2.m3 zero-pickup | RESOLVED | §4.1 Instances; Table 6 rows for 25% and 50% deliver-only | At 50% deliver-only, thr.-k has the higher mean (3.3 vs 3.1). The paper calls this a "tie" without a test. My check gives Wilcoxon p=0.10, Baton better on 13 of 19 instances. |
| R2.M6 day types | PARTLY_RESOLVED | §4.5 "Non-exchangeable days"; Table 10; §5 Limitations | The discussion is adequate. The experiment gives the day-type-specific fit five times more promotion data than the pooled fit (N2), which inflates the claimed recovery. |
| R3.5 Baton below DP50k / oracle on city | RESOLVED | §4.3 paragraph "Two reference points lie above Baton"; Table 13 | The explanation is correct, and the budget experiment shows the gap closing. It also shows that the DP references are not near-exact (N7). |

## New issues introduced or exposed by the revision

**N1 — MAJOR. The position-dependent threshold fails as an optimiser, not through estimation variance.** Anchors: §4.2 third observation ("searching m−1 boundaries … has more variance"); `fit_threshold_k` in `core/extra_policies.py` lines 68–107.
- The search starts from the reactive policy (all thresholds infinite), tries 24 quantile candidates per stop, and stops after at most 2 backward sweeps.
- The per-stop class contains the global peak-label threshold: with a monotone p̂_k, the rule p̂_k(W) > τ is a per-stop cut in W. So in-sample, thr.-k should never do worse than thr.
- I re-ran both on 10 Dethloff instances (every fourth) with the paper's seeds and computed **training-set** savings:

  | gate | thr. (global) | thr.-k | thr.-k, 48 candidates and 6 sweeps |
  |---|---|---|---|
  | SAA | 23.2% | 16.3% | 17.1% |
  | WDRO | 20.2% | 0.0% (stuck at the reactive start) | — |

- Test-set values on the same subset reproduce Table 2 (15.3% and 0.0%). The −0.2% on WDRO and 15.4% on SAA are therefore failures to leave a plateau, not over-fitting.
- **Fix:** start the descent from the tuned global threshold's per-stop cuts (optionally also from Baton-ho's boundaries), report in-sample and out-of-sample cost, and rewrite the "variance" sentence. The headline is unlikely to change, since Baton leads thr. by a wide margin, but R1.4 asks exactly this question.

**N2 — MAJOR. The day-type experiment confounds day-type specificity with the amount of data.** Anchors: Table 10; `_daytype_job` and `_mix` in `run_baton_r1.py`.
- The pooled fit uses 1,000 mixed days, about 200 of them promotions (`P_PROMO=0.2`).
- The "day-type-specific" fit uses 1,000 normal days plus a separate 1,000 **promotion** days, i.e. 2,000 days in total.
- An operator whose promotions occur one day in five has about 200 promotion days per 1,000.
- The claim "a separate fit per known day type recovers the remainder" (30.9% vs 29.4%) is therefore partly a five-fold data advantage.
- **Fix:** split the pooled history by day type (about 800 and 200 days) and refit. Optionally add a pooled fit with a day-type indicator.

**N3 — MAJOR. The monotonicity ("shape") test has no demonstrated power and misses the critical law.** Anchors: Table 9, last column and caption; §4.5 "Third"; `_shape_job`.
- The test is an adjacent-bin one-sided z-test at 1.96 over 25 quantile bins, pooled across all stops that have at least 500 alive paths.
- The rejection rate under independence (0.22%, i.e. 94 of 42,684 pairs) is an order of magnitude below the nominal 2.5%. That shows only that the test is conservative when the truth is strictly increasing. It says nothing about power. A low rejection rate at rho=0.6 or 0.9 (0.08% and 0.008%) is therefore uninformative on its own.
- Most tested pairs lie where the cost curve is flat near zero or steep, far from the stopping boundary, so a local dip near the boundary would be diluted.
- The test was **not** run for rho=0.3 or for the day-factor law (dashes in Table 9). Under g = z(p−d), a very negative W_k also signals a large z and hence a larger future variance, so E[cost | W_k] can be non-monotone. On that law Baton's share of DP350k's saving drops to 60% on SAA (7.4 vs 12.4).
- **Fix:**
  - Add a power check: inject a known dip of size δ·H_k into the target and report the detection rate.
  - Report the largest estimated decrease near the boundary, in units of H_k, with a confidence interval.
  - Run the test for rho=0.3 and the day-factor law.

**N4 — MINOR. The capacitated-pool comparison is set up against a weak baseline.** Anchors: §4.6; Table 12; `_pool_job`, `_run_fallback`.
- "Naive" decides at the marginal price H_k − F_sb, which is cheaper than pay-per-use, and ignores the cap. At S=0 every request it makes is refused.
- The reported gains of 15.8% and 4.1% are measured against this policy. The λ grid already contains λ=F_sb (the pay-per-use policy facing the cap) and λ=10^6 (the fallback menu only). Those are the informative baselines.
- A refused route cannot act at the refusal stop itself: the fallback starts at k+1. This penalises every refused request.
- **Fix:** report shadow pricing against λ=F_sb and λ=∞, allow the fallback action at the refusal stop, and add a multi-plan pool with the cap binding.

**N5 — MINOR. The new experiments carry no uncertainty.** Anchors: Tables 6 and 9–12.
- Claims such as "tie within noise" (Table 9 SAA rho=0; Table 6, 50% deliver-only), "Baton-cf performs as well as Baton", 30.9 / 29.4 / 28.4, and 15.8% are bare point estimates.
- My Wilcoxon checks of the two "ties" give p=0.073 (Baton against thr.-k, SAA rho=0: Baton better on 20 instances, worse on 6) and p=0.10 (50% deliver-only). Both are consistent with the wording, but the paper does not report them.
- **Fix:** add paired bootstrap confidence intervals, as in Table 3, for the key contrast of each new table. Consider stratifying the bootstrap by Dethloff class (the 40 instances come from four structured classes).

**N6 — MINOR. The fresh-start bias is measured against a reference that is not the optimum.** Anchor: Table 11, bias columns.
- F_k⋆ comes from `fit_dp_actions`: a binned DP whose F_k is itself simulated through its own fitted downstream policy. Bias (i) is therefore measured against another suboptimal policy.
- The bias is a signed mean over stops, so errors of opposite sign can cancel.
- **Fix:** also report the mean absolute error, and state the nature of the reference.

**N7 — MAJOR (claim scope). The "near-exact" reference programs are not bounds.** Anchors: §3.6 reference points; abstract "86%–96%"; Table 13; Table 11.
- The DP programs use at most 256 quantile bins with no shape constraint (`n_bins = clip(N//30, 8, 256)`).
- Baton exceeds them on city at N=2×10^4 (12.9 vs 12.2 and 12.3), and the exact-reset variant exceeds DP350k.
- The 86%–96% ratios and the claim that "the shortfall is the finite-sample price" therefore rest on a denominator of unknown accuracy.
- The DP350k in the main tables is the **state-conditional** variant (`fit_dp_actions_cf`), whereas §3.6 and Table 1 describe an unconditional fresh-start program.
- **Fix:**
  - Add an information-relaxation (dual) upper bound on the saving for the handoff-only problem (Andersen–Broadie or Brown–Smith–Sun, with martingale penalties built from the fitted C_k), at least on a subset of instances.
  - Use Baton fitted on 5×10^4 paths as a second reference.
  - Document the cf construction of DP350k.

## Claim/number consistency problems
Numbers that match:
- **Text against tables/macros (all consistent):** 2.1–10.5 pp and "at least 31" wins (Table 3); 86%–96% (CSV ratios 0.862–0.956); 27.9 vs 24.7 (Table 2); 22.3%–54.3% (Table 4); 30.9 / 29.4 / 28.4 (macros `dtPromo*`); 734/735, 3.8 vs 7.7, 90% (macros `regret*`).
- **Against the raw CSVs (all consistent):**
  - `daytype.csv`: 30.86 / 29.40 / 28.42.
  - `shape.csv`: 0.220%, 0.077%, 0.008%.
  - `pool.csv`: 15.85%, 4.10%, 9.12%, 3.45%.
  - `budget.csv`: 25.8 / 41.7 / −2.4 / 56.3 / 12.9 (against 12.2 and 12.3).
  - `timing.csv`: 7.5 / 27.5 ms and 1.24 µs.
  - `results_grand_dethloff.csv`: Det Baton 27.85, thr.-k 25.78, WDRO ratio 0.862.
  - `results_city_eval.csv`: Baton 10.60, Baton-ho 10.55.

Problems:
- **C1.** Table 11 against Tables 2, 6 and 9, for what is the same policy on the same benchmark:

  | quantity | Table 11 | Tables 2 / 6 / 9 |
  |---|---|---|
  | DP350k, Dethloff SAA | 54.6 | 56.4 (Tables 2 and 9) |
  | Baton-ho, Dethloff SAA | 18.9 | 20.1 (Table 2) |
  | Baton-ho, Salhi–Nagy | 41.3 | 42.3 (Table 6) |
  | DP350k, Salhi–Nagy | 53.5 | 54.0 (Table 6) |

  Cause: Table 11 pools cost sums over routes with m ≥ 3, while the other tables average per-instance percentages. Readers will compare "exact reset 54.6" with 56.4. **Fix:** state the aggregation in every caption, or use one aggregation throughout.
- **C2.** "Averaged over all routes the myopic rule's regret is 3.8%" (§4.4). It is actually a ratio of cost sums dominated by Det plans: the all-routes value is 3.83%, Det alone 3.78%, SAA 6.83%, City 1.97%. The figure is not a per-route average. Reword.
- **C3.** §4.7 says "the grid-tuned competitors cost about the same or more". Table 14 shows π3 at 5.1 ms and restock at 5.1 ms, both below Baton's 27.5 ms.
- **C4.** Table 14 and §3.6 describe the online lookup as a "monotone step function". The code evaluates `np.interp` over `X_thresholds_`, and scikit-learn's `predict` also interpolates linearly, so it is a piecewise-linear lookup. The substance is unaffected; fix the wording.
- **C5.** Table 13's reference row gives DP50k on Dethloff SAA as 22.2, against 22.1 in Table 2. This is a trivial difference, but it indicates the two pipelines use different reactive baselines or slack calibrations; `_budget_job` calibrates B on `gbig[:1000]` at every N.

## Recommendation signal
**Minor revision.**
- The headline comparison is statistically sound: correct Holm correction, bootstrap and effect sizes, with the instance as the experimental unit. There is no train/test leakage: seeds are disjoint and all tuning and selection is on training days only. Every number I spot-checked reproduces.
- The robustness experiments added for R1.4 and R2.M6 have fixable design flaws: an under-optimised thr.-k (N1) and a data-size confound in the day-type fits (N2). The shape test lacks a power demonstration (N3).
- The "near-exact" yardsticks should be backed by a genuine upper bound or re-labelled (N7). None of these is likely to overturn the main ranking.
