# RARECERT round 2: extra experiments

## Plan-level result: overflow probability at the planned capacity Q

143 of 148 hubs done. Hub SMIS N = 1e5 and route SMIS N = 2e4 per route at c = Q (results/plans_Region, SAA plans).

| region | hubs | median P_hub(Q) | q10 | q90 | P_region(Q) = 1 - prod(1 - P_h) |
|---|---|---|---|---|---|
| VNN | 22 | 0.0792 | 0.0616 | 0.103 | 0.843 |
| GD | 121 | 0.108 | 0.0806 | 0.136 | 1 |
| all | 143 | 0.103 | 0.072 | 0.135 | 1 |

Share of hubs with P_hub(Q) <= gamma, and of routes with P_route(Q) <= gamma (route SMIS, tolerance: estimate <= gamma):

| gamma | VNN hubs | GD hubs | all hubs | VNN routes | GD routes | all routes |
|---|---|---|---|---|---|---|
| 0.001 | 0.0% (0/22) | 0.0% (0/121) | 0.0% (0/143) | 11.2% (36/322) | 12.3% (712/5787) | 12.2% (748/6109) |
| 0.0001 | 0.0% (0/22) | 0.0% (0/121) | 0.0% (0/143) | 9.3% (30/322) | 8.7% (506/5787) | 8.8% (536/6109) |
| 1e-05 | 0.0% (0/22) | 0.0% (0/121) | 0.0% (0/143) | 9.0% (29/322) | 6.7% (390/5787) | 6.9% (419/6109) |
| 1e-06 | 0.0% (0/22) | 0.0% (0/121) | 0.0% (0/143) | 8.7% (28/322) | 5.9% (342/5787) | 6.1% (370/6109) |

Routes: 6109; median P_route(Q) 0.0239, q90 0.0383, share with P_route(Q) = 0: 4.7%.

## Missed-mode mass on all 480 route-gamma rows

358 rows. Single shift error e = p_single / p_ref - 1.

| m_hat bin | rows | region VNN / GD | median e | median abs e | max abs e | share e < 0 | median abs e / m_hat | median 2 rse |
|---|---|---|---|---|---|---|---|---|
| [0, 1e-06) | 249 | 0 / 249 | -0.05% | 0.21% | 1.03% | 0.57 | 3.22e+12 | 0.44% |
| [1e-06, 0.001) | 2 | 2 / 0 | -0.08% | 0.35% | 0.43% | 0.50 | 10.7 | 0.46% |
| [0.001, 0.01) | 11 | 11 / 0 | -0.45% | 0.45% | 0.95% | 0.91 | 1.1 | 0.47% |
| [0.01, 0.1) | 39 | 39 / 0 | -2.27% | 2.28% | 94.70% | 0.95 | 0.784 | 0.47% |
| [0.1, 2) | 57 | 57 / 0 | -10.82% | 10.82% | 58.52% | 0.89 | 0.637 | 2.71% |

By level (gamma) for rows with m_hat >= 1e-3:

| gamma | rows | median e | median abs e | median abs e / m_hat |
|---|---|---|---|---|
| 0.001 | 27 | -2.25% | 2.65% | 0.292 |
| 0.0001 | 28 | -4.73% | 5.10% | 0.738 |
| 1e-05 | 26 | -8.34% | 8.34% | 0.779 |
| 1e-06 | 26 | -9.61% | 9.61% | 0.841 |

## Pickup rescaling and correlated factors

60 rows. Reference = SMIS with 1e6 draws; e = p_method / p_ref - 1.

| region | pickup scale s | factor corr r | gamma | rows | varrho median | m_hat median | single e median | min | max | share e < 0 | share abs e > 2 rse | CE mixture abs e median |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| GD | 0.5 | 0 | 1e-05 | 10 | 0.717 | 3.6e-05 | 0.09% | -42.61% | 0.62% | 0.40 | 0.10 | 0.27% |
| GD | 0.5 | 0 | 1e-06 | 11 | 0.767 | 9.05e-05 | -0.07% | -40.75% | 0.74% | 0.55 | 0.09 | 0.56% |
| GD | 1 | 0.5 | 0.0001 | 3 | 0.307 | 0 | -0.17% | -0.62% | -0.14% | 1.00 | 0.00 | 0.59% |
| VNN | 0.5 | 0 | 0.0001 | 9 | 0.481 | 1.62e-12 | -0.30% | -0.63% | 0.61% | 0.89 | 0.00 | 0.20% |
| VNN | 0.5 | 0 | 1e-06 | 9 | 0.481 | 0 | 0.29% | -0.33% | 1.43% | 0.33 | 0.33 | 0.51% |
| VNN | 2 | 0 | 0.0001 | 8 | 0.507 | 9.65e-11 | 0.03% | -1.27% | 0.65% | 0.50 | 0.12 | 0.39% |
| VNN | 2 | 0 | 1e-06 | 7 | 0.495 | 1.67e-15 | -0.01% | -0.54% | 1.13% | 0.57 | 0.14 | 0.37% |
| VNN | 1 | 0.5 | 0.0001 | 3 | 0.942 | 0.262 | 1.54% | -3.76% | 47.16% | 0.33 | 0.33 | 0.18% |

## Budget-fair rivals and quadrature grids

21 route-gamma rows (N = 1e6 simulator evaluations for every sampling method). Times are wall times on a loaded shared machine (indicative).

| gamma | group | rows | SMIS e med / max | two-end e med / max | GL shifted e med / max | CE small-pilot e med / max | SMIS RV | two-end RV | GL RV | CE small RV | CE small pilot | GL signed e median | SMIS signed e median |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.0001 | balanced (VNN) | 5 | 0.13% / 0.57% | 0.20% / 0.39% | 15.07% / 27.34% | 0.08% / 0.14% | 4.74 | 4.68 | 84.2 | 5.08 | 9.8e+04 | -15.07% | 0.07% |
| 0.0001 | other (GD) | 6 | 0.18% / 0.32% | 0.07% / 0.14% | 0.13% / 0.50% | 0.32% / 0.41% | 4.53 | 4.48 | 10.3 | 4.76 | 2.76e+05 | 0.02% | -0.03% |
| 0.0001 | all | 11 | 0.17% / 0.57% | 0.09% / 0.39% | 0.50% / 27.34% | 0.09% / 0.41% | 4.54 | 4.48 | 11.5 | 4.82 | 1.26e+05 | -0.21% | 0.07% |
| 1e-06 | balanced (VNN) | 4 | 0.16% / 0.57% | 0.19% / 0.36% | 2.02% / 18.72% | 0.16% / 0.51% | 5.85 | 5.82 | 14.1 | 6.35 | 1.41e+05 | -2.02% | 0.03% |
| 1e-06 | other (GD) | 6 | 0.19% / 0.45% | 0.17% / 0.58% | 0.34% / 0.70% | 0.18% / 0.39% | 5.78 | 5.76 | 11.8 | 6.16 | 4.72e+05 | 0.03% | 0.03% |
| 1e-06 | all | 10 | 0.19% / 0.57% | 0.17% / 0.58% | 0.54% / 18.72% | 0.17% / 0.51% | 5.79 | 5.77 | 12.2 | 6.26 | 2.46e+05 | -0.28% | 0.03% |

Quadrature reference on coarser grids (error against the fine reference p_ref, wall time per evaluation):

| grid (nodes, V pair) | rows | abs e median | q90 | max | wall s median | q90 |
|---|---|---|---|---|---|---|
| 41, (128,256) | 21 | 0.01% | 0.42% | 0.72% | 31.2 | 48.6 |
| 21, (256,512) | 21 | 1.02% | 3.05% | 4.09% | 44.1 | 68.5 |
| 21, (128,256) | 21 | 0.97% | 3.05% | 4.17% | 8.2 | 12.7 |
| 11, (256,512) | 21 | 48.16% | 56.77% | 64.16% | 12.4 | 18.3 |

Reference grid for comparison: 41 nodes, V = (256, 512); median wall time in baselines_summary.md.

Wall time to reach relative precision 0.01 at 95% confidence (build + 1.96^2 RV / 0.01^2 draws at the measured per-draw time; medians over rows):

- SMIS: median 2.32 s (q90 3.67 s)
- CE small pilot: median 3.73 s (q90 9.26 s)

## Coverage at rare levels on balanced routes

300 replications of N = 1e4 draws per route; SMIS proposal built once, single shift rebuilt in every replication. Reference: quadrature (41 nodes, V = 256/512).

| gamma | method | routes | coverage pooled | min | max | median mean error | median sd |
|---|---|---|---|---|---|---|---|
| 1e-05 | smis | 4 | 0.947 | 0.930 | 0.963 | 0.08% | 2.34% |
| 1e-05 | single_rebuilt | 4 | 0.270 | 0.020 | 0.617 | -5.17% | 24.42% |
| 1e-06 | smis | 4 | 0.943 | 0.930 | 0.960 | -0.03% | 2.42% |
| 1e-06 | single_rebuilt | 4 | 0.365 | 0.013 | 0.860 | -8.22% | 10.82% |

## Hub SMIS against crude Monte Carlo (gamma = 1e-3 calibrated capacity)

| hub | customers | routes | SMIS (N=1e6) | rse | crude | rse | z | SMIS/crude - 1 | exact quadrature | err est | SMIS/exact - 1 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| VNN-BACKAN | 25 | 5 | 0.000986 | 0.21% | 0.000988 | 0.71% | -0.37 | -0.27% | 0.000985 | 5.13% | 0.09% |
## Routes with 15 or more stops (crude Monte Carlo reference)

No rows yet.

## Route-day simulator: crude Monte Carlo at gamma = 1e-4 and 1e-5

| route | L | gamma | class | crude draws | crude p | rse | MDIS e | CE mixture e | single e |
|---|---|---|---|---|---|---|---|---|---|
| GD-BOLUOXIAN#50 | 11 | 0.0001 | mixed | 100,000,000 | 9.82e-05 | 1.01% | -0.15% | 0.06% | -16.38% |
| VNN-BACNINH#6 | 9 | 0.0001 | mixed | 100,000,000 | 0.000101 | 0.99% | -2.06% | -2.47% | -27.77% |
| VNN-HAIDUONG#14 | 9 | 0.0001 | mixed | 100,000,000 | 9.93e-05 | 1.00% | -1.21% | -1.56% | -37.84% |
| VNN-HANAM#4 | 7 | 0.0001 | mixed | 100,000,000 | 9.8e-05 | 1.01% | -0.33% | 0.34% | -14.33% |
| VNN-BACNINH#6 | 9 | 1e-05 | mixed | 200,000,000 | 9.97e-06 | 2.24% | 1.84% | 1.79% | -39.77% |
| VNN-HAIDUONG#14 | 9 | 1e-05 | mixed | 200,000,000 | 1.03e-05 | 2.20% | -4.85% | -4.66% | -45.02% |

gamma 0.0001: 4 routes, MDIS abs e median 0.77%, max 2.06%; median crude rse 1.01%.

gamma 1e-05: 2 routes, MDIS abs e median 3.34%, max 4.85%; median crude rse 2.22%.
