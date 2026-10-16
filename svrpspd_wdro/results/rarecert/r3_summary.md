# RARECERT round 2: extra experiments

## Plan-level result: overflow probability at the planned capacity Q

51 of 148 hubs done. Hub SMIS N = 1e5 and route SMIS N = 2e4 per route at c = Q (results/plans_Region, SAA plans).

| region | hubs | median P_hub(Q) | q10 | q90 | P_region(Q) = 1 - prod(1 - P_h) |
|---|---|---|---|---|---|
| VNN | 9 | 0.0741 | 0.0565 | 0.0913 | 0.495 |
| GD | 42 | 0.111 | 0.0845 | 0.148 | 0.993 |
| all | 51 | 0.106 | 0.0761 | 0.141 | 0.997 |

Share of hubs with P_hub(Q) <= gamma, and of routes with P_route(Q) <= gamma (route SMIS, tolerance: estimate <= gamma):

| gamma | VNN hubs | GD hubs | all hubs | VNN routes | GD routes | all routes |
|---|---|---|---|---|---|---|
| 0.001 | 0.0% (0/9) | 0.0% (0/42) | 0.0% (0/51) | 16.0% (16/100) | 10.8% (239/2207) | 11.1% (255/2307) |
| 0.0001 | 0.0% (0/9) | 0.0% (0/42) | 0.0% (0/51) | 12.0% (12/100) | 7.7% (169/2207) | 7.8% (181/2307) |
| 1e-05 | 0.0% (0/9) | 0.0% (0/42) | 0.0% (0/51) | 12.0% (12/100) | 5.8% (129/2207) | 6.1% (141/2307) |
| 1e-06 | 0.0% (0/9) | 0.0% (0/42) | 0.0% (0/51) | 12.0% (12/100) | 5.3% (116/2207) | 5.5% (128/2307) |

Routes: 2307; median P_route(Q) 0.0249, q90 0.0378, share with P_route(Q) = 0: 4.6%.

## Missed-mode mass on all 480 route-gamma rows

162 rows. Single shift error e = p_single / p_ref - 1.

| m_hat bin | rows | region VNN / GD | median e | median abs e | max abs e | share e < 0 | median abs e / m_hat | median 2 rse |
|---|---|---|---|---|---|---|---|---|
| [0, 1e-06) | 108 | 0 / 108 | -0.02% | 0.20% | 0.85% | 0.53 | 6.04e+12 | 0.43% |
| [1e-06, 0.001) | 2 | 2 / 0 | -0.08% | 0.35% | 0.43% | 0.50 | 10.7 | 0.46% |
| [0.001, 0.01) | 7 | 7 / 0 | -0.45% | 0.45% | 0.95% | 0.86 | 1.1 | 0.44% |
| [0.01, 0.1) | 19 | 19 / 0 | -2.35% | 2.35% | 94.70% | 0.95 | 0.786 | 0.47% |
| [0.1, 2) | 26 | 26 / 0 | -9.57% | 9.57% | 29.35% | 0.92 | 0.619 | 2.88% |

By level (gamma) for rows with m_hat >= 1e-3:

| gamma | rows | median e | median abs e | median abs e / m_hat |
|---|---|---|---|---|
| 0.001 | 13 | -2.35% | 3.10% | 0.297 |
| 0.0001 | 14 | -5.10% | 5.10% | 0.678 |
| 1e-05 | 13 | -6.43% | 6.43% | 0.805 |
| 1e-06 | 12 | -6.24% | 6.24% | 0.841 |

## Pickup rescaling and correlated factors

8 rows. Reference = SMIS with 1e6 draws; e = p_method / p_ref - 1.

| region | pickup scale s | factor corr r | gamma | rows | varrho median | m_hat median | single e median | min | max | share e < 0 | share abs e > 2 rse | CE mixture abs e median |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| VNN | 0.5 | 0 | 0.0001 | 2 | 0.498 | 1.79e-11 | -0.50% | -0.57% | -0.43% | 1.00 | 0.00 | 0.53% |
| VNN | 0.5 | 0 | 1e-06 | 2 | 0.503 | 1.09e-13 | 0.03% | -0.14% | 0.21% | 0.50 | 0.00 | 0.72% |
| VNN | 2 | 0 | 0.0001 | 2 | 0.49 | 4.91e-12 | 0.15% | -0.29% | 0.58% | 0.50 | 0.00 | 0.23% |
| VNN | 2 | 0 | 1e-06 | 2 | 0.532 | 2.16e-10 | 0.20% | 0.12% | 0.27% | 0.00 | 0.00 | 0.28% |

## Budget-fair rivals and quadrature grids

No rows yet.

## Coverage at rare levels on balanced routes

No rows yet.

## Hub SMIS against crude Monte Carlo (gamma = 1e-3 calibrated capacity)

No rows yet.

## Routes with 15 or more stops (crude Monte Carlo reference)

No rows yet.

## Route-day simulator: crude Monte Carlo at gamma = 1e-4 and 1e-5

No rows yet.
