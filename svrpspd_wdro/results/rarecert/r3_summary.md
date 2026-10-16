# RARECERT round 2: extra experiments

## Plan-level result: overflow probability at the planned capacity Q

3 of 148 hubs done. Hub SMIS N = 1e5 and route SMIS N = 2e4 per route at c = Q (results/plans_Region, SAA plans).

| region | hubs | median P_hub(Q) | q10 | q90 | P_region(Q) = 1 - prod(1 - P_h) |
|---|---|---|---|---|---|
| VNN | 0 | n/a | n/a | n/a | 0 |
| GD | 3 | 0.153 | 0.124 | 0.155 | 0.368 |
| all | 3 | 0.153 | 0.124 | 0.155 | 0.368 |

Share of hubs with P_hub(Q) <= gamma, and of routes with P_route(Q) <= gamma (route SMIS, tolerance: estimate <= gamma):

| gamma | VNN hubs | GD hubs | all hubs | VNN routes | GD routes | all routes |
|---|---|---|---|---|---|---|
| 0.001 | n/a | 0.0% (0/3) | 0.0% (0/3) | n/a | 4.8% (11/230) | 4.8% (11/230) |
| 0.0001 | n/a | 0.0% (0/3) | 0.0% (0/3) | n/a | 4.3% (10/230) | 4.3% (10/230) |
| 1e-05 | n/a | 0.0% (0/3) | 0.0% (0/3) | n/a | 3.0% (7/230) | 3.0% (7/230) |
| 1e-06 | n/a | 0.0% (0/3) | 0.0% (0/3) | n/a | 2.6% (6/230) | 2.6% (6/230) |

Routes: 230; median P_route(Q) 0.0306, q90 0.0398, share with P_route(Q) = 0: 2.2%.

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
