# RARECERT revision task A: exact conditional reference and classical rivals

30 routes from results/quasar/rare_routes.csv, gamma in 0.0001, 1e-06 (60 route-gamma rows, 0 failed). Wall time 270.7 min.

Reference: exact conditional DP (rarecert.baselines.exact_reference). P(F > c | W) is the discrete DP at V = 256 and 512 levels (the lower and upper of the pair); the factor integral is a product trapezoid on [-6, 9]^2 with 41 nodes per axis; the two V values are combined by Richardson extrapolation of order 2. The reference error estimate is |p_hi - p_ext| + quad_err, where quad_err = |p - p_coarse| is the difference to the nested coarse grid (every second node), a conservative bound on the quadrature error. Relative errors are |p/p_ref - 1|. The reference itself is the discrete model at V levels extrapolated, not the continuum; its estimated error is reported per row.

Estimators: SMIS n_arc = 0 (N = 1e6, from smis_routes.csv); CE mixture and single shift (from rare_routes.csv, the CE pilot counted in the sample cost, none for the single shift); GL two-step (N = 20000 simulator evaluations, inner draws 8): crude and shift with the factor shift at the dominant point (Glasserman-Li), and shift with the factor shift tuned as below; conditional MC over the factors (N_factor = 2000 conditional DP calls at V = 256) with the factor shift tuned by cond_shift (per-draw relative variance on a pilot grid of DP values); subset simulation (5 replications, N_level = 2000, p0 = 0.1, modified Metropolis width 1.0; relative error of the replicate mean, RV from the replicate spread).

RV is the relative variance per unit of cost (simulator evaluation, or conditional DP call for cond MC); n(eps) = 1.96^2 RV / eps^2 units (CE adds its pilot). t(eps) = build + n(eps) x seconds per unit, with seconds per unit measured on this machine (simulator for the simulator-based rivals, one conditional DP for cond MC).

## Accuracy: relative error vs the exact reference (median / max)

| group | n | SMIS med / max | CE mixture med / max | single shift med / max | GL crude med / max | GL shift med / max | GL tuned shift med / max | cond MC med / max | subset sim med / max | ref wall s med |
|---|---|---|---|---|---|---|---|---|---|---|
| gamma 0.0001, rho band [0,1/3] | 3 | 0.16% / 0.29% | 0.13% / 0.17% | 0.13% / 0.38% | 5.05% / 12.36% | 1.07% / 1.46% | 2.02% / 3.31% | 2.17% / 2.38% | 9.18% / 16.30% | 265 |
| gamma 0.0001, rho band (1/3,0.5] | 12 | 0.10% / 0.36% | 0.24% / 0.32% | 0.14% / 0.41% | 1.93% / 10.31% | 1.82% / 7.31% | 1.33% / 4.71% | 1.17% / 2.79% | 6.73% / 20.39% | 312 |
| gamma 0.0001, rho band (0.75,1] | 15 | 0.15% / 0.51% | 0.16% / 0.38% | 6.19% / 23.49% | 13.39% / 47.78% | 15.47% / 47.05% | 10.43% / 45.31% | 13.73% / 93.69% | 12.84% / 38.12% | 296 |
| gamma 0.0001, all | 30 | 0.11% / 0.51% | 0.16% / 0.38% | 0.41% / 23.49% | 4.99% / 47.78% | 3.10% / 47.05% | 2.89% / 45.31% | 2.62% / 93.69% | 8.06% / 38.12% | 297 |
| gamma 1e-06, rho band [0,1/3] | 3 | 0.20% / 0.52% | 0.29% / 0.32% | 0.22% / 0.45% | 4.54% / 7.93% | 1.43% / 3.69% | 1.75% / 3.23% | 2.06% / 4.61% | 9.80% / 30.93% | 267 |
| gamma 1e-06, rho band (1/3,0.5] | 12 | 0.14% / 0.33% | 0.33% / 0.85% | 0.14% / 0.53% | 5.13% / 12.72% | 1.97% / 3.93% | 2.62% / 5.03% | 0.82% / 6.12% | 26.98% / 72.72% | 314 |
| gamma 1e-06, rho band (0.75,1] | 15 | 0.15% / 0.58% | 0.27% / 0.57% | 9.65% / 94.68% | 10.56% / 48.95% | 7.36% / 47.59% | 16.53% / 82.72% | 13.16% / 63.01% | 16.71% / 108.48% | 299 |
| gamma 1e-06, all | 30 | 0.16% / 0.58% | 0.29% / 0.85% | 0.36% / 94.68% | 7.13% / 48.95% | 2.86% / 47.59% | 4.12% / 82.72% | 3.14% / 63.01% | 21.64% / 108.48% | 303 |

## Cost: RV per unit / samples for eps 0.1 / 0.01

| group | SMIS RV / n(0.1) / n(0.01) | CE mixture RV / n(0.1) / n(0.01) | single shift RV / n(0.1) / n(0.01) | GL crude RV / n(0.1) / n(0.01) | GL shift RV / n(0.1) / n(0.01) | GL tuned shift RV / n(0.1) / n(0.01) | cond MC RV / n(0.1) / n(0.01) | subset sim RV / n(0.1) / n(0.01) |
|---|---|---|---|---|---|---|---|---|
| gamma 0.0001, rho band [0,1/3] | 4.49 / 1.73e+03 / 1.73e+05 | 4.56 / 8.18e+04 / 2.58e+05 | 4.23 / 1.63e+03 / 1.63e+05 | 32.4 / 1.24e+04 / 1.24e+06 | 10.4 / 3.98e+03 / 3.98e+05 | 10.4 / 3.99e+03 / 3.99e+05 | 0.883 / 339 / 3.39e+04 | 1.96e+03 / 7.54e+05 / 7.54e+07 |
| gamma 0.0001, rho band (1/3,0.5] | 4.51 / 1.73e+03 / 1.73e+05 | 4.59 / 8.18e+04 / 2.58e+05 | 4.23 / 1.63e+03 / 1.63e+05 | 28.9 / 1.11e+04 / 1.11e+06 | 11 / 4.22e+03 / 4.22e+05 | 10.8 / 4.17e+03 / 4.17e+05 | 0.909 / 349 / 3.49e+04 | 597 / 2.29e+05 / 2.29e+07 |
| gamma 0.0001, rho band (0.75,1] | 4.72 / 1.81e+03 / 1.81e+05 | 4.8 / 1.04e+06 / 1.22e+06 | 212 / 8.14e+04 / 8.14e+06 | 27.2 / 1.05e+04 / 1.05e+06 | 11.6 / 4.45e+03 / 4.45e+05 | 817 / 3.14e+05 / 3.14e+07 | 57.1 / 2.19e+04 / 2.19e+06 | 856 / 3.29e+05 / 3.29e+07 |
| gamma 0.0001, all | 4.55 / 1.75e+03 / 1.75e+05 | 4.64 / 1.02e+05 / 2.76e+05 | 4.26 / 1.64e+03 / 1.64e+05 | 28.8 / 1.1e+04 / 1.1e+06 | 11.2 / 4.32e+03 / 4.32e+05 | 83.1 / 3.19e+04 / 3.19e+06 | 6.46 / 2.48e+03 / 2.48e+05 | 716 / 2.75e+05 / 2.75e+07 |
| gamma 1e-06, rho band [0,1/3] | 5.76 / 2.21e+03 / 2.21e+05 | 5.83 / 1.22e+05 / 3.44e+05 | 5.46 / 2.1e+03 / 2.1e+05 | 142 / 5.44e+04 / 5.44e+06 | 12 / 4.6e+03 / 4.6e+05 | 11.9 / 4.59e+03 / 4.59e+05 | 0.896 / 344 / 3.44e+04 | 2.1e+03 / 8.05e+05 / 8.05e+07 |
| gamma 1e-06, rho band (1/3,0.5] | 5.78 / 2.22e+03 / 2.22e+05 | 5.86 / 1.22e+05 / 3.45e+05 | 5.46 / 2.1e+03 / 2.1e+05 | 71 / 2.73e+04 / 2.73e+06 | 12.3 / 4.72e+03 / 4.72e+05 | 12.3 / 4.74e+03 / 4.74e+05 | 1.01 / 389 / 3.89e+04 | 3.16e+03 / 1.22e+06 / 1.22e+08 |
| gamma 1e-06, rho band (0.75,1] | 5.92 / 2.27e+03 / 2.27e+05 | 6 / 1.28e+06 / 1.51e+06 | 5.51 / 2.12e+03 / 2.12e+05 | 52.4 / 2.01e+04 / 2.01e+06 | 13.3 / 5.1e+03 / 5.1e+05 | 1.8e+03 / 6.91e+05 / 6.91e+07 | 153 / 5.88e+04 / 5.88e+06 | 4.99e+03 / 1.92e+06 / 1.92e+08 |
| gamma 1e-06, all | 5.81 / 2.23e+03 / 2.23e+05 | 5.89 / 1.22e+05 / 3.47e+05 | 5.47 / 2.1e+03 / 2.1e+05 | 57.2 / 2.2e+04 / 2.2e+06 | 12.7 / 4.87e+03 / 4.87e+05 | 15.2 / 5.84e+03 / 5.84e+05 | 1.35 / 519 / 5.19e+04 | 4.43e+03 / 1.7e+06 / 1.7e+08 |

## Wall time: build + samples x time per unit (s), medians

| group | SMIS t(0.1) / t(0.01) s | CE mixture t(0.1) / t(0.01) s | single shift t(0.1) / t(0.01) s | GL crude t(0.1) / t(0.01) s | GL shift t(0.1) / t(0.01) s | GL tuned shift t(0.1) / t(0.01) s | cond MC t(0.1) / t(0.01) s | subset sim t(0.1) / t(0.01) s |
|---|---|---|---|---|---|---|---|---|
| gamma 0.0001, rho band [0,1/3] | 0.478 / 2.34 | 1.25 / 3.37 | 0.0199 / 1.99 | 0.843 / 17.8 | 0.706 / 5.39 | 0.717 / 5.21 | 18.7 / 773 | 10.3 / 1.03e+03 |
| gamma 0.0001, rho band (1/3,0.5] | 0.583 / 2.71 | 1.08 / 3.15 | 0.0198 / 1.98 | 0.746 / 15.2 | 0.555 / 5.14 | 0.564 / 5.32 | 21.6 / 967 | 3.28 / 328 |
| gamma 0.0001, rho band (0.75,1] | 0.405 / 2.67 | 13.9 / 16.2 | 0.978 / 97.8 | 0.486 / 13.2 | 0.426 / 6.2 | 3.49 / 320 | 562 / 5.52e+04 | 4.57 / 457 |
| gamma 0.0001, all | 0.566 / 2.59 | 1.54 / 4.6 | 0.0248 / 2.48 | 0.593 / 14.6 | 0.503 / 5.4 | 1.3 / 40.8 | 113 / 9.31e+03 | 4.31 / 431 |
| gamma 1e-06, rho band [0,1/3] | 0.422 / 2.27 | 1.09 / 3.06 | 0.0187 / 1.87 | 0.877 / 41 | 0.502 / 4.94 | 0.499 / 6.6 | 19.2 / 850 | 13.2 / 1.32e+03 |
| gamma 1e-06, rho band (1/3,0.5] | 0.582 / 3.23 | 1.76 / 4.93 | 0.0256 / 2.56 | 0.945 / 37.9 | 0.693 / 6.41 | 0.692 / 5.36 | 22 / 1.02e+03 | 16.9 / 1.69e+03 |
| gamma 1e-06, rho band (0.75,1] | 0.814 / 3.57 | 13.2 / 15.4 | 0.0249 / 2.49 | 1.26 / 21.3 | 1.09 / 7.1 | 8.97 / 757 | 1.01e+03 / 9.89e+04 | 23.1 / 2.31e+03 |
| gamma 1e-06, all | 0.678 / 3.32 | 2.26 / 6.4 | 0.0245 / 2.45 | 1.16 / 31.8 | 0.854 / 6.64 | 1.12 / 16.2 | 42.6 / 2.19e+03 | 18 / 1.8e+03 |

## By route length tercile (cut points L = 6.66667, 8.33333)

| group | n | SMIS med / max | CE mixture med / max | single shift med / max | GL crude med / max | GL shift med / max | GL tuned shift med / max | cond MC med / max | subset sim med / max | ref wall s med |
|---|---|---|---|---|---|---|---|---|---|---|
| gamma 0.0001, short | 10 | 0.12% / 0.51% | 0.17% / 0.31% | 0.81% / 23.49% | 9.92% / 45.50% | 11.66% / 42.01% | 2.70% / 42.88% | 2.48% / 93.69% | 11.46% / 36.54% | 225 |
| gamma 0.0001, medium | 10 | 0.10% / 0.36% | 0.14% / 0.38% | 0.55% / 9.56% | 4.38% / 47.78% | 2.92% / 47.05% | 2.41% / 45.31% | 2.72% / 20.12% | 4.27% / 38.12% | 297 |
| gamma 0.0001, long | 10 | 0.13% / 0.30% | 0.17% / 0.32% | 0.37% / 17.83% | 3.70% / 20.04% | 2.01% / 22.78% | 3.85% / 13.70% | 2.68% / 17.81% | 9.76% / 21.33% | 426 |
| gamma 1e-06, short | 10 | 0.14% / 0.37% | 0.22% / 0.55% | 4.91% / 58.58% | 11.15% / 38.23% | 5.93% / 40.10% | 3.92% / 68.04% | 8.58% / 60.61% | 19.62% / 72.72% | 228 |
| gamma 1e-06, medium | 10 | 0.16% / 0.52% | 0.31% / 0.85% | 0.40% / 51.25% | 4.84% / 48.95% | 1.38% / 47.59% | 6.39% / 82.72% | 2.18% / 23.11% | 29.07% / 108.48% | 303 |
| gamma 1e-06, long | 10 | 0.20% / 0.58% | 0.30% / 0.57% | 0.34% / 94.68% | 6.08% / 15.18% | 3.47% / 13.30% | 4.13% / 43.31% | 3.14% / 63.01% | 14.38% / 57.62% | 422 |

| group | SMIS RV / n(0.1) / n(0.01) | CE mixture RV / n(0.1) / n(0.01) | single shift RV / n(0.1) / n(0.01) | GL crude RV / n(0.1) / n(0.01) | GL shift RV / n(0.1) / n(0.01) | GL tuned shift RV / n(0.1) / n(0.01) | cond MC RV / n(0.1) / n(0.01) | subset sim RV / n(0.1) / n(0.01) |
|---|---|---|---|---|---|---|---|---|
| gamma 0.0001, short | 4.59 / 1.77e+03 / 1.77e+05 | 4.67 / 4.62e+05 / 6.41e+05 | 28.1 / 1.08e+04 / 1.08e+06 | 32.8 / 1.26e+04 / 1.26e+06 | 10.7 / 4.11e+03 / 4.11e+05 | 156 / 5.98e+04 / 5.98e+06 | 22.7 / 8.71e+03 / 8.71e+05 | 404 / 1.55e+05 / 1.55e+07 |
| gamma 0.0001, medium | 4.53 / 1.74e+03 / 1.74e+05 | 4.58 / 1.02e+05 / 2.76e+05 | 4.24 / 1.63e+03 / 1.63e+05 | 27.5 / 1.06e+04 / 1.06e+06 | 11.5 / 4.41e+03 / 4.41e+05 | 96.5 / 3.71e+04 / 3.71e+06 | 12.7 / 4.86e+03 / 4.86e+05 | 722 / 2.77e+05 / 2.77e+07 |
| gamma 0.0001, long | 4.58 / 1.76e+03 / 1.76e+05 | 4.69 / 9.18e+04 / 2.7e+05 | 4.27 / 1.64e+03 / 1.64e+05 | 23.1 / 8.87e+03 / 8.87e+05 | 12.6 / 4.86e+03 / 4.86e+05 | 83.1 / 3.19e+04 / 3.19e+06 | 6.46 / 2.48e+03 / 2.48e+05 | 977 / 3.75e+05 / 3.75e+07 |
| gamma 1e-06, short | 5.83 / 2.24e+03 / 2.24e+05 | 5.9 / 6.02e+05 / 8.3e+05 | 5.49 / 2.11e+03 / 2.11e+05 | 84 / 3.23e+04 / 3.23e+06 | 11.9 / 4.59e+03 / 4.59e+05 | 543 / 2.09e+05 / 2.09e+07 | 57.3 / 2.2e+04 / 2.2e+06 | 4.69e+03 / 1.8e+06 / 1.8e+08 |
| gamma 1e-06, medium | 5.78 / 2.22e+03 / 2.22e+05 | 5.83 / 1.22e+05 / 3.44e+05 | 5.46 / 2.1e+03 / 2.1e+05 | 55 / 2.11e+04 / 2.11e+06 | 12.6 / 4.86e+03 / 4.86e+05 | 45.8 / 1.76e+04 / 1.76e+06 | 4.87 / 1.87e+03 / 1.87e+05 | 2.65e+03 / 1.02e+06 / 1.02e+08 |
| gamma 1e-06, long | 5.81 / 2.23e+03 / 2.23e+05 | 5.91 / 1.22e+05 / 3.47e+05 | 5.48 / 2.11e+03 / 2.11e+05 | 39.7 / 1.53e+04 / 1.53e+06 | 13.8 / 5.31e+03 / 5.31e+05 | 15.2 / 5.84e+03 / 5.84e+05 | 1.35 / 519 / 5.19e+04 | 4.48e+03 / 1.72e+06 / 1.72e+08 |

## Reference cost

Total reference wall time over all routes: 315.7 min. Median per route-gamma: 298 s; median estimated quadrature error relative to p_ref: 1.18e-02; median |p_hi - p_lo| / p_ref: 2.16e-03.

## Hubs: exact P_hub (reference) vs the hub SMIS estimate

| hub | region | customers | routes (reachable) | gamma | c | p_ref | p_hi - p_lo | p_SMIS | rse SMIS | SMIS/ref - 1 | z | ref wall s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| GD-RUYUANYAOZUZIZHIXIAN | GD | 54 | 7 (6) | 0.001 | 42312.6 | 0.001016 | -2.45e-06 | 0.001008 | 0.67% | -0.76% | -0.26 | 1.88e+03 |
| VNN-BACKAN | VNN | 25 | 5 (3) | 0.001 | 18880.6 | 0.0009847 | -1.49e-06 | 0.000991 | 0.67% | 0.64% | 0.12 | 779 |
| VNN-CAOBANG | VNN | 50 | 7 (6) | 0.001 | 20038.3 | 0.0008224 | -2.75e-06 | 0.0008255 | 0.68% | 0.37% | 0.04 | 1.75e+03 |
| VNN-DIENBIEN | VNN | 55 | 8 (7) | 0.001 | 20311.2 | 0.0009377 | -1.56e-06 | 0.0009563 | 0.67% | 1.98% | 0.46 | 1.83e+03 |
| VNN-LAICHAU | VNN | 29 | 5 (4) | 0.001 | 19050.8 | 0.000853 | -1.24e-06 | 0.0008621 | 0.66% | 1.07% | 0.26 | 970 |
| VNN-LAOCAI | VNN | 54 | 9 (7) | 0.001 | 20120.8 | 0.0008783 | -1.34e-06 | 0.0008835 | 0.66% | 0.60% | 0.16 | 1.79e+03 |
