# EXPERIMENT B: test_b analysis

Stream: `test_b` (20,000 days per instance). Evaluation commit: `fab25c5cef65d0e2bd361e4b7f5036220e30df50`.
FAMILY line (HEAD, pinned): `TEST_B FAMILY CLOSED: family=[alpz,rlsm,dqn,ppo] descriptive=[thr] dropped=[thr:b_thr,dos:cut] references=[full,ladder] independence=[b_alpz_ind,b_full_ind,b_ladder_ind,b_thr_ind] fidelity_sha256=10de648a66b15694a3d735f29a22a0a0bb8de0a10a2e29589be53394caa9267e manifests_sha256=c9a17214d138babdcc8c40b890b4226d0c0452768d9c5a959c6fa8eaaddb2483 code_sha256=62d89cd311ec66037067f266d49f2fb454e4eac66f6902707a854d1a63d6f64f analysis_sha256=179beae5100e74f950423e4457659253173200d516ac94dc1f0eae530a4d6e9f`
family_sha256: `71361a5ba0d945ebc8f3624b665b7fb31d1917f32b2239eeb6a12792614c7232`

Family F = ['alpz', 'rlsm', 'dqn', 'ppo']; descriptive = ['thr']; dropped = ['thr:b_thr', 'dos:cut']; references = ['full', 'ladder']; independence = ['b_alpz_ind', 'b_full_ind', 'b_ladder_ind', 'b_thr_ind'].

**All family arms are reported below regardless of outcome. No arm is omitted from any table because of its result.**

## 1. Primary family (FULL vs each arm at 10^5, Holm over |F| = 4)

| arm | n | mean | median | ci_95 | holm_ci | ci_90 | n_better | rank_biserial | pooled_ratio | gc_mean_full | p_raw | p_holm | branch |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| alpz | 20 | 11.33 | 11.1 | [+8.76, +13.91] | [+7.94, +14.72] | [+9.21, +13.46] | 20/20 | 1 | 0.9281 | nan | 1.907e-06 | 7.629e-06 | WIN |
| rlsm | 20 | 0.1611 | 0.1602 | [+0.09, +0.24] | [+0.06, +0.26] | [+0.10, +0.22] | 17/20 | 0.8476 | 0.9984 | nan | 0.0003223 | 0.0003223 | TIE_STRONG |
| dqn | 20 | 0.1567 | 0.1463 | [+0.11, +0.21] | [+0.09, +0.22] | [+0.11, +0.20] | 19/20 | 0.981 | 0.9987 | nan | 5.722e-06 | 1.144e-05 | TIE_STRONG |
| ppo | 20 | 1.679 | 1.126 | [+0.46, +2.90] | [+0.07, +3.29] | [+0.67, +2.69] | 20/20 | 1 | 0.9807 | nan | 1.907e-06 | 7.629e-06 | WIN |

Claims (section 1.3 templates):

- **alpz** [WIN]: At 10^5 paths, FULL lowers recourse cost by 11.33% [+8.76, +13.91] versus alpz trained on the same paths (20/20 instances; Holm-level CI [+7.94, +14.72]).
- **rlsm** [TIE_STRONG]: FULL and rlsm are equivalent within ±0.7% [+0.10, +0.22] (CI90; per arm, not simultaneous), although significantly different (p_Holm = 0.000322).
- **dqn** [TIE_STRONG]: FULL and dqn are equivalent within ±0.7% [+0.11, +0.20] (CI90; per arm, not simultaneous), although significantly different (p_Holm = 1.14e-05).
- **ppo** [WIN]: At 10^5 paths, FULL lowers recourse cost by 1.68% [+0.46, +2.90] versus ppo trained on the same paths (20/20 instances; Holm-level CI [+0.07, +3.29]).

Arms removed by a compute cut (9.2 (ii)), not fitted and not evaluated: ['dos'].

## 2. Descriptive and dropped arms (outside F; unadjusted; no branch)

| arm | reason | stage | n | mean | median | ci95_lo | ci95_hi | ci90_lo | ci90_hi | n_better | p_raw | ci90_inside_delta |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| thr | dropped=thr:b_thr | b_thr | 20 | 11.01 | 10.49 | 8.484 | 13.54 | 8.923 | 13.1 | 20 | 1.907e-06 | False |

## 3. Independence world (section 3.8; descriptive; no Holm; 12 routes, rep 0)

| comparison | stage | n | mean | ci95_lo | ci95_hi | ci90_lo | ci90_hi | n_better | p_raw |
|---|---|---|---|---|---|---|---|---|---|
| FULL vs none | b_ladder_ind | 10 | 12.25 | 7.817 | 16.67 | 8.657 | 15.83 | 10 | 0.001953 |
| FULL vs baton_eq | b_ladder_ind | 10 | 0.08901 | -0.3851 | 0.5631 | -0.2951 | 0.4732 | 4 | 0.375 |
| FULL vs oracle | b_ladder_ind | 10 | -64.99 | -84.65 | -45.33 | -80.92 | -49.06 | 0 | 0.001953 |
| FULL vs THR | b_thr_ind | 10 | -0.01377 | -0.3109 | 0.2834 | -0.2546 | 0.227 | 2 | 0.375 |
| FULL vs ALP-Z | b_alpz_ind | 10 | -0.03296 | -0.3475 | 0.2815 | -0.2878 | 0.2219 | 4 | 0.625 |

## 4. RE-EVAL (outside F; renamed from REP; re-samples test days only; not a replication)

FULL vs baton_eq at 10^5 on test_b: Δ̄ = +0.85% [+0.68, +1.02] (CI95), 20/20 instances better; A's unadjusted rule gives **WIN** (one-sided p: greater 9.54e-07, less 1).

## 5. D_a: each evaluated arm vs baton_eq at 10^5 (outside F; A's unadjusted rule)

| arm | stage | n | mean | median | ci95_lo | ci95_hi | ci90_lo | ci90_hi | n_better | p_greater | p_less | branch |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| thr | b_thr | 20 | -13.16 | -12.09 | -16.86 | -9.463 | -16.22 | -10.11 | 0 | 1 | 9.537e-07 | LOSE |
| alpz | b_alpz | 20 | -13.62 | -12.84 | -17.38 | -9.856 | -16.73 | -10.51 | 0 | 1 | 9.537e-07 | LOSE |
| rlsm | b_rlsm | 20 | 0.692 | 0.695 | 0.5373 | 0.8466 | 0.5642 | 0.8197 | 20 | 9.537e-07 | 1 | TIE_WEAK |
| dqn | b_dqn | 20 | 0.6961 | 0.7132 | 0.5153 | 0.8768 | 0.5467 | 0.8454 | 19 | 1.907e-06 | 1 | TIE_WEAK |
| ppo | b_ppo | 20 | -1.115 | -0.3081 | -2.98 | 0.7494 | -2.656 | 0.4252 | 5 | 0.994 | 0.006808 | TIE_WEAK |

## 6. ALP-Z vs THR on test_b (descriptive)

Paired difference (positive: ALP-Z cheaper) -0.385% [-0.55, -0.22], ALP-Z better on 2/20 instances; identical mean cost on 0.0% of 124 route-reps (rows hold costs, not decisions); mean |handoff-rate gap| 0.0051.

## 7. THR realised cost over its own-model DP value (section 3.8)

THR's realised test_b cost exceeds its own-model DP value by -8.30% (pooled over 124 route-reps).

## 8. Strata (descriptive; FULL vs arm, positive: FULL cheaper)

| arm | stratum | n | mean | ci95_lo | ci95_hi | n_better |
|---|---|---|---|---|---|---|
| thr | long | 19 | 7.317 | 4.328 | 10.31 | 19 |
| thr | medium | 18 | 13.67 | 8.515 | 18.83 | 18 |
| thr | short | 2 | 29.52 | 3.846 | 55.2 | 2 |
| alpz | long | 19 | 7.652 | 4.544 | 10.76 | 19 |
| alpz | medium | 18 | 13.98 | 8.785 | 19.18 | 18 |
| alpz | short | 2 | 29.79 | 5.91 | 53.66 | 2 |
| rlsm | long | 19 | 0.1821 | 0.1144 | 0.2499 | 18 |
| rlsm | medium | 18 | 0.1506 | 0.0601 | 0.241 | 12 |
| rlsm | short | 2 | -0.3549 | -2.519 | 1.809 | 0 |
| dqn | long | 19 | 0.1764 | 0.135 | 0.2178 | 19 |
| dqn | medium | 18 | 0.1686 | 0.04524 | 0.292 | 13 |
| dqn | short | 2 | -0.1221 | -1.778 | 1.534 | 1 |
| ppo | long | 19 | 1.248 | 0.9412 | 1.554 | 19 |
| ppo | medium | 18 | 1.996 | 0.06462 | 3.927 | 18 |
| ppo | short | 2 | 0.7652 | -3.646 | 5.177 | 2 |

## 9. Between-replicate SD of the route mean cost (percent)

| arm | routes | mean_sd_pct | max_sd_pct |
|---|---|---|---|
| alpz | 62 | 0.2512 | 1.032 |
| baton_eq | 62 | 0.1268 | 0.9079 |
| dqn | 62 | 0.1631 | 1.561 |
| nn_full | 62 | 0.1132 | 1.207 |
| none | 62 | 0 | 0 |
| oracle | 62 | 0 | 0 |
| ppo | 62 | 0.3728 | 3.255 |
| rlsm | 62 | 0.1132 | 0.5005 |
| thr | 62 | 0.08692 | 0.413 |

## 10. Wall-clock and budget

| arm | payloads | fit_median_s | fit_max_s | path_stop_reads_median | peak_rss_max_mb | n_params_median | devices |
|---|---|---|---|---|---|---|---|
| full | 124 | 1947 | 4554 | nan | nan | nan | - |

