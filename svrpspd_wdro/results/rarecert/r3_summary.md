# RARECERT round 2: extra experiments

## Plan-level result: overflow probability at the planned capacity Q

148 of 148 hubs done. Hub SMIS N = 1e5 and route SMIS N = 2e4 per route at c = Q (results/plans_Region, SAA plans).

| region | hubs | median P_hub(Q) | q10 | q90 | P_region(Q) = 1 - prod(1 - P_h) |
|---|---|---|---|---|---|
| VNN | 26 | 0.0792 | 0.0619 | 0.0995 | 0.886 |
| GD | 122 | 0.108 | 0.0807 | 0.138 | 1 |
| all | 148 | 0.102 | 0.07 | 0.135 | 1 |

Share of hubs with P_hub(Q) <= gamma, and of routes with P_route(Q) <= gamma (route SMIS, tolerance: estimate <= gamma):

| gamma | VNN hubs | GD hubs | all hubs | VNN routes | GD routes | all routes |
|---|---|---|---|---|---|---|
| 0.001 | 0.0% (0/26) | 0.0% (0/122) | 0.0% (0/148) | 11.5% (43/375) | 12.2% (716/5867) | 12.2% (759/6242) |
| 0.0001 | 0.0% (0/26) | 0.0% (0/122) | 0.0% (0/148) | 9.9% (37/375) | 8.6% (507/5867) | 8.7% (544/6242) |
| 1e-05 | 0.0% (0/26) | 0.0% (0/122) | 0.0% (0/148) | 9.6% (36/375) | 6.7% (391/5867) | 6.8% (427/6242) |
| 1e-06 | 0.0% (0/26) | 0.0% (0/122) | 0.0% (0/148) | 9.1% (34/375) | 5.8% (343/5867) | 6.0% (377/6242) |

Routes: 6242; median P_route(Q) 0.0239, q90 0.0383, share with P_route(Q) = 0: 4.7%.

## Missed-mode mass on all 480 route-gamma rows

480 rows. Single shift error e = p_single / p_ref - 1.

| m_hat bin | rows | region VNN / GD | median e | median abs e | max abs e | share e < 0 | median abs e / m_hat | median 2 rse |
|---|---|---|---|---|---|---|---|---|
| [0, 1e-06) | 320 | 0 / 320 | -0.05% | 0.19% | 1.03% | 0.59 | 6.04e+12 | 0.43% |
| [1e-06, 0.001) | 2 | 2 / 0 | -0.08% | 0.35% | 0.43% | 0.50 | 10.7 | 0.46% |
| [0.001, 0.01) | 16 | 16 / 0 | -0.43% | 0.48% | 1.00% | 0.88 | 0.976 | 0.47% |
| [0.01, 0.1) | 58 | 58 / 0 | -2.24% | 2.28% | 94.70% | 0.93 | 0.81 | 0.47% |
| [0.1, 2) | 84 | 84 / 0 | -9.86% | 10.51% | 82.92% | 0.81 | 0.576 | 3.17% |

By level (gamma) for rows with m_hat >= 1e-3:

| gamma | rows | median e | median abs e | median abs e / m_hat |
|---|---|---|---|---|
| 0.001 | 40 | -1.55% | 2.86% | 0.217 |
| 0.0001 | 40 | -4.34% | 5.10% | 0.744 |
| 1e-05 | 39 | -5.58% | 6.43% | 0.784 |
| 1e-06 | 39 | -5.94% | 6.16% | 0.794 |

## Pickup rescaling and correlated factors

180 rows. Reference = SMIS with 1e6 draws; e = p_method / p_ref - 1.

| region | pickup scale s | factor corr r | gamma | rows | varrho median | m_hat median | single e median | min | max | share e < 0 | share abs e > 2 rse | CE mixture abs e median |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| GD | 1 | -0.5 | 0.0001 | 10 | 0.353 | 0 | -0.21% | -0.71% | 0.92% | 0.60 | 0.10 | 0.39% |
| GD | 0.5 | 0 | 1e-05 | 30 | 0.73 | 7.6e-05 | 0.08% | -42.61% | 1.29% | 0.43 | 0.10 | 0.36% |
| GD | 0.5 | 0 | 1e-06 | 30 | 0.73 | 2.02e-05 | -0.08% | -40.75% | 0.74% | 0.63 | 0.13 | 0.49% |
| GD | 1 | 0.5 | 0.0001 | 10 | 0.353 | 0 | -0.16% | -0.84% | 0.88% | 0.60 | 0.00 | 0.59% |
| VNN | 1 | -0.5 | 0.0001 | 10 | 0.947 | 0.287 | -24.55% | -80.46% | -1.86% | 1.00 | 1.00 | 0.35% |
| VNN | 0.5 | 0 | 0.0001 | 20 | 0.51 | 2.55e-11 | -0.23% | -0.67% | 0.61% | 0.65 | 0.00 | 0.26% |
| VNN | 0.5 | 0 | 1e-06 | 20 | 0.51 | 9.95e-15 | 0.11% | -1.72% | 1.43% | 0.45 | 0.20 | 0.61% |
| VNN | 2 | 0 | 0.0001 | 20 | 0.49 | 4.91e-12 | 0.13% | -1.27% | 1.17% | 0.45 | 0.25 | 0.46% |
| VNN | 2 | 0 | 1e-06 | 20 | 0.49 | 1.01e-15 | -0.21% | -0.92% | 1.13% | 0.60 | 0.05 | 0.38% |
| VNN | 1 | 0.5 | 0.0001 | 10 | 0.947 | 0.284 | 0.91% | -5.79% | 47.16% | 0.40 | 0.20 | 0.40% |

## Budget-fair rivals and quadrature grids

60 route-gamma rows (N = 1e6 simulator evaluations for every sampling method). Times are wall times on a loaded shared machine (indicative).

| gamma | group | rows | SMIS e med / max | two-end e med / max | GL shifted e med / max | CE small-pilot e med / max | SMIS RV | two-end RV | GL RV | CE small RV | CE small pilot | GL signed e median | SMIS signed e median |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.0001 | balanced (VNN) | 15 | 0.15% / 0.57% | 0.13% / 0.59% | 14.99% / 31.20% | 0.10% / 0.30% | 4.74 | 4.64 | 30 | 5.02 | 1.2e+05 | -14.99% | -0.02% |
| 0.0001 | other (GD) | 15 | 0.17% / 0.38% | 0.08% / 0.45% | 0.27% / 0.92% | 0.12% / 0.56% | 4.52 | 4.47 | 10.6 | 4.75 | 2.76e+05 | -0.02% | -0.05% |
| 0.0001 | all | 30 | 0.16% / 0.57% | 0.09% / 0.59% | 0.66% / 31.20% | 0.11% / 0.56% | 4.55 | 4.48 | 11.9 | 4.86 | 1.52e+05 | -0.58% | -0.03% |
| 1e-06 | balanced (VNN) | 15 | 0.12% / 0.57% | 0.13% / 0.39% | 6.63% / 48.54% | 0.13% / 0.51% | 5.95 | 5.86 | 13.9 | 6.32 | 1.54e+05 | -6.63% | -0.08% |
| 1e-06 | other (GD) | 15 | 0.24% / 0.58% | 0.19% / 0.60% | 0.24% / 0.70% | 0.23% / 0.49% | 5.78 | 5.76 | 12.2 | 6.12 | 4.02e+05 | -0.10% | 0.03% |
| 1e-06 | all | 30 | 0.18% / 0.58% | 0.14% / 0.60% | 0.62% / 48.54% | 0.15% / 0.51% | 5.81 | 5.77 | 12.9 | 6.24 | 2.41e+05 | -0.44% | -0.02% |

Quadrature reference on coarser grids (error against the fine reference p_ref, wall time per evaluation):

| grid (nodes, V pair) | rows | abs e median | q90 | max | wall s median | q90 |
|---|---|---|---|---|---|---|
| 41, (128,256) | 60 | 0.01% | 0.44% | 0.97% | 94.9 | 157 |
| 21, (256,512) | 60 | 1.18% | 4.65% | 8.45% | 140 | 231 |
| 21, (128,256) | 60 | 1.20% | 4.64% | 8.45% | 24.1 | 38.9 |
| 11, (256,512) | 60 | 48.50% | 59.11% | 70.86% | 38.2 | 62.6 |

Reference grid for comparison: 41 nodes, V = (256, 512); median wall time in baselines_summary.md.

Wall time to reach relative precision 0.01 at 95% confidence (build + 1.96^2 RV / 0.01^2 draws at the measured per-draw time; medians over rows):

- SMIS: median 5.06 s (q90 9.37 s)
- CE small pilot: median 9.09 s (q90 16.2 s)

## Coverage at rare levels on balanced routes

300 replications of N = 1e4 draws per route; SMIS proposal built once, single shift rebuilt in every replication. Reference: quadrature (41 nodes, V = 256/512).

| gamma | method | routes | coverage pooled | min | max | median mean error | median sd |
|---|---|---|---|---|---|---|---|
| 1e-05 | smis | 10 | 0.949 | 0.930 | 0.970 | 0.10% | 2.27% |
| 1e-05 | single_rebuilt | 10 | 0.338 | 0.020 | 0.953 | -5.17% | 13.59% |
| 1e-06 | smis | 10 | 0.953 | 0.930 | 0.987 | -0.01% | 2.40% |
| 1e-06 | single_rebuilt | 10 | 0.364 | 0.007 | 0.963 | -8.22% | 13.70% |

## Hub SMIS against crude Monte Carlo (gamma = 1e-3 calibrated capacity)

| hub | customers | routes | SMIS (N=1e6) | rse | crude | rse | z | SMIS/crude - 1 | exact quadrature | err est | SMIS/exact - 1 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| VNN-BACKAN | 25 | 5 | 0.000986 | 0.21% | 0.000988 | 0.71% | -0.37 | -0.27% | 0.000985 | 5.13% | 0.09% |
| VNN-LAICHAU | 29 | 5 | 0.000852 | 0.21% | 0.00086 | 0.76% | -1.16 | -0.91% | 0.000853 | 4.01% | -0.07% |
| VNN-CAOBANG | 50 | 7 | 0.000823 | 0.21% | 0.000821 | 0.78% | 0.35 | 0.28% | 0.000822 | 9.06% | 0.10% |
| VNN-THAINGUYEN | 102 | 14 | 0.000939 | 0.22% | 0.000943 | 0.73% | -0.66 | -0.50% | n/a | n/a | n/a |

Mean z -0.46 over 4 hubs (t = -1.46); mean relative difference -0.35%, median -0.39%.

## Routes with 15 or more stops (crude Monte Carlo reference)

| gamma | rows | L range | crude draws | ref rse median | SMIS e median / max (abs) | single e median / max | CE e median / max | SMIS RV | single RV | CE RV |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.001 | 5 | 15-17 | 50,000,000 | 0.45% | 0.36% / 0.93% | 0.30% / 0.32% | 0.31% / 0.52% | 3.81 | 3.52 | 3.94 |
| 0.0001 | 5 | 15-33 | 100,000,000 | 1.00% | 0.88% / 1.48% | 0.53% / 1.53% | 1.02% / 1.77% | 4.56 | 4.25 | 4.68 |

## Route-day simulator: crude Monte Carlo at gamma = 1e-4 and 1e-5

| route | L | gamma | class | crude draws | crude p | rse | MDIS e | CE mixture e | single e |
|---|---|---|---|---|---|---|---|---|---|
| GD-BOLUOXIAN#50 | 11 | 0.0001 | mixed | 100,000,000 | 9.82e-05 | 1.01% | -0.15% | 0.06% | -16.38% |
| GD-DONGGUANSHI-4#48 | 6 | 0.0001 | mixed | 100,000,000 | 7.57e-05 | 1.15% | -0.00% | -0.11% | 0.38% |
| GD-HAIKANGXIAN#22 | 8 | 0.0001 | mixed | 100,000,000 | 7.23e-05 | 1.18% | 0.83% | 0.94% | 0.65% |
| GD-SHENZHENSHI-1#5 | 6 | 0.0001 | mixed | 100,000,000 | 7.02e-05 | 1.19% | 2.03% | 1.51% | 1.70% |
| VNN-BACNINH#6 | 9 | 0.0001 | mixed | 100,000,000 | 0.000101 | 0.99% | -2.06% | -2.47% | -27.77% |
| VNN-HAIDUONG#14 | 9 | 0.0001 | mixed | 100,000,000 | 9.93e-05 | 1.00% | -1.21% | -1.56% | -37.84% |
| VNN-HANAM#4 | 7 | 0.0001 | mixed | 100,000,000 | 9.8e-05 | 1.01% | -0.33% | 0.34% | -14.33% |
| VNN-HANOI-1#57 | 4 | 0.0001 | mixed | 100,000,000 | 8.94e-05 | 1.06% | -0.43% | -0.25% | -7.83% |
| GD-BOLUOXIAN#50 | 11 | 1e-05 | mixed | 200,000,000 | 9.99e-06 | 2.24% | -2.60% | -2.96% | -12.49% |
| VNN-BACNINH#6 | 9 | 1e-05 | mixed | 200,000,000 | 9.97e-06 | 2.24% | 1.84% | 1.79% | -39.77% |
| VNN-HAIDUONG#14 | 9 | 1e-05 | mixed | 200,000,000 | 1.03e-05 | 2.20% | -4.85% | -4.66% | -45.02% |

gamma 0.0001: 8 routes, MDIS abs e median 0.63%, max 2.06%; median crude rse 1.03%.

gamma 1e-05: 3 routes, MDIS abs e median 2.60%, max 4.85%; median crude rse 2.24%.
