# EXPERIMENT A summary

Eval set: **dev**.
delta = 0.70% of BATON recourse cost (section 1.3). Unit: instance (n = 20). CI95 uses t_{19,0.975}; CI90 uses t_{19,0.95}.

## Primary: FULL vs BATON-eq at N = 10^5 (section 1.4)

Branch: **WIN**. Mean Δ̄ = +0.856%, median +0.751%, CI95 [+0.66, +1.06], CI90 [+0.69, +1.02], s = 0.425.
One-sided exact Wilcoxon: p(FULL cheaper) = 9.537e-07, p(FULL dearer) = 1. Instances better: 20/20, worse: 0/20. Significance flag: better.

Claim (template filled from the data):

> At 10^5 paths, FULL lowers recourse cost by 0.86% [+0.66, +1.06] versus BATON at equal paths (n = 20 instances; 20/20 better).

Cluster bootstrap (sensitivity only, not used for decisions; 10000 draws, seed 20261115): CI95 [+0.68, +1.05], CI90 [+0.71, +1.01].

## Secondary family (section 8, Holm over S1 to S11)

| id | a @ N | b @ N | Δ̄ % | median | CI95 | CI90 | better | rank-biserial | pooled ratio a/b | mean GC(a) | p (two-sided) | p (Holm) | class |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| S1 | nn_full @ 100 | baton_eq @ 100 | -1.495 | -0.910 | [-2.58, -0.41] | [-2.39, -0.60] | 5/20 | -0.70 | 1.0069 | 0.619 | 0.00486 | 0.0243 | worse |
| S2 | nn_full @ 500 | baton_eq @ 500 | -0.045 | +0.318 | [-0.66, +0.57] | [-0.56, +0.47] | 14/20 | +0.18 | 0.9962 | 0.710 | 0.498 | 0.9048 | equivalent |
| S3 | nn_full @ 2000 | baton_eq @ 2000 | +0.268 | +0.381 | [+0.01, +0.52] | [+0.05, +0.48] | 15/20 | +0.58 | 0.9954 | 0.736 | 0.02148 | 0.08594 | equivalent |
| S4 | nn_full @ 10000 | baton_eq @ 10000 | +0.618 | +0.631 | [+0.46, +0.78] | [+0.48, +0.75] | 20/20 | +1.00 | 0.9928 | 0.750 | 1.907e-06 | 2.098e-05 | inconclusive |
| S5 | nn_full @ 100000 | nn_scalar @ 100000 | +0.847 | +0.782 | [+0.64, +1.05] | [+0.68, +1.02] | 20/20 | +1.00 | 0.9908 | 0.756 | 1.907e-06 | 2.098e-05 | better |
| S6 | nn_full @ 2000 | nn_scalar @ 2000 | +0.388 | +0.444 | [+0.19, +0.58] | [+0.23, +0.55] | 17/20 | +0.76 | 0.9946 | 0.736 | 0.00169 | 0.01014 | equivalent |
| S7 | nn_scalar @ 100000 | dp_big @ 1000000 | +0.072 | +0.005 | [-0.03, +0.17] | [-0.01, +0.16] | 10/20 | +0.20 | 0.9999 | 0.744 | 0.4524 | 0.9048 | equivalent |
| S8 | nn_scalar @ 2000 | baton_eq @ 2000 | -0.123 | -0.219 | [-0.33, +0.08] | [-0.29, +0.05] | 7/20 | -0.36 | 1.0009 | 0.730 | 0.165 | 0.4949 | equivalent |
| S9 | nn_full @ 100000 | nn_mlp @ 100000 | +0.415 | +0.288 | [+0.25, +0.58] | [+0.28, +0.55] | 20/20 | +1.00 | 0.9977 | 0.756 | 1.907e-06 | 2.098e-05 | equivalent |
| S10 | nn_full @ 100000 | baton_big @ 1000000 | +0.838 | +0.786 | [+0.63, +1.04] | [+0.67, +1.01] | 20/20 | +1.00 | 0.9909 | 0.756 | 1.907e-06 | 2.098e-05 | better |
| S11 | nn_full @ 100000 | dp_big @ 1000000 | +0.919 | +0.932 | [+0.73, +1.11] | [+0.76, +1.08] | 20/20 | +1.00 | 0.9906 | 0.756 | 1.907e-06 | 2.098e-05 | better |

## Attribution (section 1.5)

- H2 (FULL vs SCALAR @ 10^5): better; the gain is information.
- H3 (SCALAR vs DP-1e6 @ 10^5): equivalent; the pipeline reaches the (k, W) optimum.
- ARCH (FULL vs MLP @ 10^5): equivalent; the paper says only "neural regressor".
- M11 (FULL vs BOOST @ 10^5): not run.

## Data-size curve (section 8, item 5)

| N | Δ̄ % (FULL vs BATON-eq) | CI95 |
|---|---|---|
| 100 | -1.495 | [-2.58, -0.41] |
| 500 | -0.045 | [-0.66, +0.57] |
| 2000 | +0.268 | [+0.01, +0.52] |
| 10000 | +0.618 | [+0.46, +0.78] |
| 100000 | +0.856 | [+0.66, +1.06] |

Crossover N: 500.

## Variance reporting (section 8, item 2)

- median over routes of the between-replicate SD of nn_full @ 100000: 0.0204.
- median over routes of the between-replicate SD of baton_eq @ 100000: 0.0210.
- (b) median over routes of the per-day paired SE of nn_full - baton_eq @ 100000 (replicate-averaged days, dev stream): 0.0757 in route-cost units (same units as the costs).
