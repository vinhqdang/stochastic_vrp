# RARECERT plan level: certified overflow of the region hubs

66 hubs (results/plans_Region, gate SAA), gamma in 0.001, 1e-05 (66 hub-gamma rows ok, 0 failed). Proxy c0: Laplace-type log S(c) = log sum_r exp(-E_r(c)) (a proxy, not a bound), secant on [Q, 3Q] within 6 evaluations. Calibration: exact hub SMIS with N = 20,000 draws per evaluation, secant on log P_hub - log gamma from (c0, 0.97 c0), at most 3 steps, stop at |f| < log 1.25; final c_final by hub SMIS with N = 100,000 draws, eta_hub = 1e-4. Route baselines N_r = 20,000 at c_final. Wall time 88.8 min.

P_region = 1 - prod over the hubs of the region of (1 - P_h), with P_h the hub SMIS estimate at the calibrated c_final (exact in the model: hubs have separate factors). b_final = c_final/Q - 1 is the calibrated buffer; b0 = c0/Q - 1 is the proxy buffer, kept for comparison.

## Region totals and buffers

| gamma | region | hubs | P_region (SMIS) | median b_final | q90 b_final | median b0 (proxy) | median P_h | calibration converged / max_steps / stalled | median steps |
|---|---|---|---|---|---|---|---|---|---|
| 1e-05 | VNN | 26 | 0.000246 | 0.6446 | 0.6801 | 0.8150 | 9.43e-06 | 26 / 0 / 0 | 1.0 |
| 1e-05 | GD | 40 | 0.000375 | 0.7581 | 0.7995 | 0.9675 | 9.31e-06 | 40 / 0 / 0 | 1.0 |
| 1e-05 | all | 66 | 0.000622 | 0.7303 | 0.7925 | 0.9249 | 9.39e-06 | 66 / 0 / 0 | 1.0 |

## Baselines and bounds

| gamma | bounds hold (hubs) | P_h / max p_r (median) | P_h / sum p_r (median) | indep / P_h (median) | indep / P_h (q90) | multi-route failure share (median) |
|---|---|---|---|---|---|---|
| 1e-05 | 66 / 66 | 3.67 | 0.51 | 1.96 | 2.91 | 0.331 |

Bounds are checked with slack 4 (rse_hub + max rse_r) on both sides. Calibration stops at |log(P_h/gamma)| < log 1.25 (or after 3 secant steps, N = 20,000 per evaluation); the final P_h (N = 1e5 at c_final) is then compared with gamma below.

## P_h / gamma distribution (final estimate at c_final)

| gamma | region | hubs | median P_h / gamma | q10 | q90 | min | max | outside log 1.25 band |
|---|---|---|---|---|---|---|---|---|
| 1e-05 | VNN | 26 | 0.943 | 0.847 | 1.05 | 0.827 | 1.08 | 0 / 26 |
| 1e-05 | GD | 40 | 0.931 | 0.839 | 1.04 | 0.801 | 1.13 | 0 / 40 |
| 1e-05 | all | 66 | 0.939 | 0.839 | 1.05 | 0.801 | 1.13 | 0 / 66 |

## Routes without an SMIS candidate (P_r = 0 at c_final)

None.

## Crude Monte Carlo check (gamma = 0.001, 2,000,000 draws, 20 smallest hubs)

| hub | customers | routes | c | P_hub SMIS | P_crude | rse crude | z (combined) |
|---|---|---|---|---|---|---|---|
