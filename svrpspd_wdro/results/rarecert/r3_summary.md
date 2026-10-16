# RARECERT round 2: extra experiments

## Plan-level result: overflow probability at the planned capacity Q

21 of 148 hubs done. Hub SMIS N = 1e5 and route SMIS N = 2e4 per route at c = Q (results/plans_Region, SAA plans).

| region | hubs | median P_hub(Q) | q10 | q90 | P_region(Q) = 1 - prod(1 - P_h) |
|---|---|---|---|---|---|
| VNN | 0 | n/a | n/a | n/a | 0 |
| GD | 21 | 0.108 | 0.0884 | 0.151 | 0.918 |
| all | 21 | 0.108 | 0.0884 | 0.151 | 0.918 |

Share of hubs with P_hub(Q) <= gamma, and of routes with P_route(Q) <= gamma (route SMIS, tolerance: estimate <= gamma):

| gamma | VNN hubs | GD hubs | all hubs | VNN routes | GD routes | all routes |
|---|---|---|---|---|---|---|
| 0.001 | n/a | 0.0% (0/21) | 0.0% (0/21) | n/a | 9.9% (111/1118) | 9.9% (111/1118) |
| 0.0001 | n/a | 0.0% (0/21) | 0.0% (0/21) | n/a | 7.1% (79/1118) | 7.1% (79/1118) |
| 1e-05 | n/a | 0.0% (0/21) | 0.0% (0/21) | n/a | 5.2% (58/1118) | 5.2% (58/1118) |
| 1e-06 | n/a | 0.0% (0/21) | 0.0% (0/21) | n/a | 4.7% (53/1118) | 4.7% (53/1118) |

Routes: 1118; median P_route(Q) 0.0255, q90 0.0377, share with P_route(Q) = 0: 3.6%.

## Missed-mode mass on all 480 route-gamma rows

No rows yet.

## Pickup rescaling and correlated factors

No rows yet.

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
