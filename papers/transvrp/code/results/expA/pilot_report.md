# EXPERIMENT A pilot report (section 7.3)

| criterion | status | detail |
|---|---|---|
| P1 | PASS | tests green: True |
| P2 | PASS | 1 rows compared on mean_cost, handoff_rate, fail_rate; max |diff| = 0.000e+00 |
| P3 | PASS | SCALAR / BATON-eq = 1.00017, threshold 1.02000 (tolerance max(2%, 3 SD) = 2.00%) |
| P4 | PASS | 72 (route, cell, policy) triples; 0 below oracle, 0 above 1.005 x none |
| P5 | PASS | fit_seconds recorded on all rows: True; cells measured: 2, extrapolated: 8; projection 2.62 days -> CPU |
| P6 | PASS | warm-start checkpoint on 7.6% of 552 trained stages (limit < 50%); mean fold disagreement 0.47%; 0 stages selected on training MSE (no validation row alive) |
| P3' (flag) | PASS | SCALAR is -0.02% cheaper than BATON-eq at N = 10^4 (flag above +1%: audit the M3 index logs) |

## P5 projection (section 7.2 plan, minutes per job)

Rule: see `plan_cells` in pilot.py. Scaled to the mean eligible route (work 117). Hours = job minutes / 60 / 4 workers + 3 h data generation.

| cell | minutes per job | source |
|---|---|---|
| full@100 | 2.86 | model: a + b N |
| full@500 | 2.92 | model: a + b N |
| full@2000 | 3.14 | model: a + b N |
| full@10000 | 5.42 | measured |
| full@100000 | 17.61 | measured |
| scalar@2000 | 3.09 | ratio x FULL |
| scalar@100000 | 17.31 | ratio x FULL |
| mlp@100000 | 3.52 | 0.2 x FULL (prior) |
| full@1000000 | 166.34 | model: a + b N + stop updates (timing) |
| scalar@1000000 | 163.78 | ratio x FULL + stop updates (timing) |

Measured inputs (scaled as above, including cells outside the plan): full@1000 = 2.01, full@10000 = 5.42, full@100000 = 17.61, scalar@1000 = 1.98, scalar@10000 = 5.31.

Projection: 62.8 h = 2.62 days -> CPU.

## Timing (10^6 pass)

- Peak memory, headline (process RSS high-water mark, whole run): **1789 MB**.
- Secondary (tracemalloc, NumPy buffers only): cache plus stop updates peak 1167 MB; generation peak 550 MB.
- Seconds: generation 62.1, cache 1.1, stop updates 1168.3, total 1231.5.

## Monotonicity diagnostic (section 4.2, reported)

SCALAR at N = 10,000 over 12 jobs (138 W-grid curves): 0.0% of curves have a neighbour decrease above 0.01 E0; 0.00% of neighbour steps do. (Section 4.2 names N*; the pilot's largest SCALAR cell is used.)


## Stop rules (section 10)

| rule | triggered | detail |
|---|---|---|
| 2 | no | 0 (route, cell, rep, policy) below the oracle |
| 3 | not applicable | not applicable: N* rows of SCALAR and DP-1e6 are absent |
| 6 | no | projection 2.62 days (ceiling 7 after cuts) |
| 7 | no | 0.0% of 25 jobs at N >= 10000 have warm-start on > 50% of stages (rule: > 25%) |
| 5 | no | 62 eligible routes over 20 instances |
