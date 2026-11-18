# EXPERIMENT A summary

Eval set: **test**.
delta = 0.70% of BATON recourse cost (section 1.3). Unit: instance (n = 20). CI95 uses t_{19,0.975}; CI90 uses t_{19,0.95}.

## Primary: FULL vs BATON-eq at N = 10^5 (section 1.4)

Branch: **WIN**. Mean Δ̄ = +0.823%, median +0.823%, CI95 [+0.65, +0.99], CI90 [+0.68, +0.96], s = 0.365.
One-sided exact Wilcoxon: p(FULL cheaper) = 1.907e-06, p(FULL dearer) = 1. Instances better: 19/20, worse: 1/20. Significance flag: better.

Claim (template filled from the data):

> At 10^5 paths, FULL lowers recourse cost by 0.82% [+0.65, +0.99] versus BATON at equal paths (n = 20 instances; 19/20 better).

Cluster bootstrap (sensitivity only, not used for decisions; 10000 draws, seed 20261115): CI95 [+0.66, +0.98], CI90 [+0.69, +0.95].

## Secondary family (section 8, Holm over S1 to S11)

| id | a @ N | b @ N | Δ̄ % | median | CI95 | CI90 | better | rank-biserial | pooled ratio a/b | mean GC(a) | p (two-sided) | p (Holm) | class |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| S1 | nn_full @ 100 | baton_eq @ 100 | -1.339 | -0.629 | [-2.34, -0.33] | [-2.17, -0.51] | 5/20 | -0.64 | 1.0059 | 0.622 | 0.01069 | 0.04276 | worse |
| S2 | nn_full @ 500 | baton_eq @ 500 | +0.113 | +0.304 | [-0.40, +0.63] | [-0.31, +0.54] | 15/20 | +0.31 | 0.9964 | 0.713 | 0.2305 | 0.6915 | equivalent |
| S3 | nn_full @ 2000 | baton_eq @ 2000 | +0.418 | +0.389 | [+0.25, +0.59] | [+0.28, +0.56] | 17/20 | +0.90 | 0.9953 | 0.740 | 8.202e-05 | 0.0004101 | equivalent |
| S4 | nn_full @ 10000 | baton_eq @ 10000 | +0.613 | +0.655 | [+0.45, +0.77] | [+0.48, +0.75] | 19/20 | +0.98 | 0.9934 | 0.752 | 5.722e-06 | 3.433e-05 | inconclusive |
| S5 | nn_full @ 100000 | nn_scalar @ 100000 | +0.826 | +0.853 | [+0.66, +0.99] | [+0.69, +0.96] | 20/20 | +1.00 | 0.9918 | 0.758 | 1.907e-06 | 2.098e-05 | better |
| S6 | nn_full @ 2000 | nn_scalar @ 2000 | +0.451 | +0.504 | [+0.32, +0.59] | [+0.34, +0.56] | 19/20 | +0.99 | 0.9949 | 0.740 | 3.815e-06 | 3.052e-05 | equivalent |
| S7 | nn_scalar @ 100000 | dp_big @ 1000000 | +0.071 | +0.003 | [-0.02, +0.16] | [-0.00, +0.14] | 10/20 | +0.30 | 0.9999 | 0.746 | 0.2455 | 0.6915 | equivalent |
| S8 | nn_scalar @ 2000 | baton_eq @ 2000 | -0.035 | -0.090 | [-0.19, +0.12] | [-0.16, +0.09] | 9/20 | -0.13 | 1.0005 | 0.733 | 0.6215 | 0.6915 | equivalent |
| S9 | nn_full @ 100000 | nn_mlp @ 100000 | +0.305 | +0.253 | [+0.21, +0.40] | [+0.23, +0.38] | 20/20 | +1.00 | 0.9982 | 0.758 | 1.907e-06 | 2.098e-05 | equivalent |
| S10 | nn_full @ 100000 | baton_big @ 1000000 | +0.757 | +0.800 | [+0.58, +0.93] | [+0.61, +0.90] | 19/20 | +0.99 | 0.9920 | 0.758 | 3.815e-06 | 3.052e-05 | better |
| S11 | nn_full @ 100000 | dp_big @ 1000000 | +0.896 | +0.849 | [+0.71, +1.08] | [+0.75, +1.05] | 20/20 | +1.00 | 0.9917 | 0.758 | 1.907e-06 | 2.098e-05 | better |

## Attribution (section 1.5)

- H2 (FULL vs SCALAR @ 10^5): better; the gain is information.
- H3 (SCALAR vs DP-1e6 @ 10^5): equivalent; the pipeline reaches the (k, W) optimum.
- ARCH (FULL vs MLP @ 10^5): equivalent; the paper says only "neural regressor".
- M11 (FULL vs BOOST @ 10^5): not run.

## Data-size curve (section 8, item 5)

| N | Δ̄ % (FULL vs BATON-eq) | CI95 |
|---|---|---|
| 100 | -1.339 | [-2.34, -0.33] |
| 500 | +0.113 | [-0.40, +0.63] |
| 2000 | +0.418 | [+0.25, +0.59] |
| 10000 | +0.613 | [+0.45, +0.77] |
| 100000 | +0.823 | [+0.65, +0.99] |

Crossover N: 500.

## Variance reporting (section 8, item 2)

- median over routes of the between-replicate SD of nn_full @ 100000: 0.0122.
- median over routes of the between-replicate SD of baton_eq @ 100000: 0.0128.
- (b) median over routes of the per-day paired SE of nn_full - baton_eq @ 100000 (replicate-averaged days, test stream): 0.0379 in route-cost units (same units as the costs).
