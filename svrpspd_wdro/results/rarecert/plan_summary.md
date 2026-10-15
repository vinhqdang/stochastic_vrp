# RARECERT plan level: certified overflow of the region hubs

66 hubs (results/plans_Region, gate SAA), gamma in 0.001, 1e-05 (132 hub-gamma rows ok, 0 failed). Proxy c0: Laplace-type log S(c) = log sum_r exp(-E_r(c)) (a proxy, not a bound), secant on [Q, 3Q] within 6 evaluations. Calibration: exact hub SMIS with N = 20,000 draws per evaluation, secant on log P_hub - log gamma from (c0, 0.97 c0), at most 3 steps, stop at |f| < log 1.25; final c_final by hub SMIS with N = 100,000 draws, eta_hub = 1e-4. Route baselines N_r = 20,000 at c_final. Wall time 0.0 min.

P_region = 1 - prod over the hubs of the region of (1 - P_h), with P_h the hub SMIS estimate at the calibrated c_final (exact in the model: hubs have separate factors). b_final = c_final/Q - 1 is the calibrated buffer; b0 = c0/Q - 1 is the proxy buffer, kept for comparison.

## Region totals and buffers

| gamma | region | hubs | P_region (SMIS) | median b_final | q90 b_final | median b0 (proxy) | median P_h | calibration converged / max_steps / stalled | median steps |
|---|---|---|---|---|---|---|---|---|---|
| 0.001 | VNN | 26 | 0.0238 | 0.3507 | 0.3765 | 0.5301 | 0.000928 | 26 / 0 / 0 | 1.0 |
| 0.001 | GD | 40 | 0.0361 | 0.4337 | 0.4725 | 0.6846 | 0.000921 | 40 / 0 / 0 | 1.0 |
| 0.001 | all | 66 | 0.059 | 0.4104 | 0.4716 | 0.6418 | 0.000921 | 66 / 0 / 0 | 1.0 |
| 1e-05 | VNN | 26 | 0.000246 | 0.6446 | 0.6801 | 0.8150 | 9.43e-06 | 26 / 0 / 0 | 1.0 |
| 1e-05 | GD | 40 | 0.000375 | 0.7581 | 0.7995 | 0.9675 | 9.31e-06 | 40 / 0 / 0 | 1.0 |
| 1e-05 | all | 66 | 0.000622 | 0.7303 | 0.7925 | 0.9249 | 9.39e-06 | 66 / 0 / 0 | 1.0 |

## Baselines and bounds

| gamma | bounds hold (hubs) | P_h / max p_r (median) | P_h / sum p_r (median) | indep / P_h (median) | indep / P_h (q90) | multi-route failure share (median) |
|---|---|---|---|---|---|---|
| 0.001 | 66 / 66 | 3.44 | 0.365 | 2.74 | 4.64 | 0.469 |
| 1e-05 | 66 / 66 | 3.67 | 0.51 | 1.96 | 2.91 | 0.331 |

Bounds are checked with slack 4 (rse_hub + max rse_r) on both sides. Calibration stops at |log(P_h/gamma)| < log 1.25 (or after 3 secant steps, N = 20,000 per evaluation); the final P_h (N = 1e5 at c_final) is then compared with gamma below.

## P_h / gamma distribution (final estimate at c_final)

| gamma | region | hubs | median P_h / gamma | q10 | q90 | min | max | outside log 1.25 band |
|---|---|---|---|---|---|---|---|---|
| 0.001 | VNN | 26 | 0.928 | 0.847 | 1 | 0.825 | 1.06 | 0 / 26 |
| 0.001 | GD | 40 | 0.921 | 0.817 | 1 | 0.805 | 1.03 | 0 / 40 |
| 0.001 | all | 66 | 0.921 | 0.834 | 1 | 0.805 | 1.06 | 0 / 66 |
| 1e-05 | VNN | 26 | 0.943 | 0.847 | 1.05 | 0.827 | 1.08 | 0 / 26 |
| 1e-05 | GD | 40 | 0.931 | 0.839 | 1.04 | 0.801 | 1.13 | 0 / 40 |
| 1e-05 | all | 66 | 0.939 | 0.839 | 1.05 | 0.801 | 1.13 | 0 / 66 |

## Routes without an SMIS candidate (P_r = 0 at c_final)

None.

## Crude Monte Carlo check (gamma = 0.001, 2,000,000 draws, 20 smallest hubs)

| hub | customers | routes | c | P_hub SMIS | P_crude | rse crude | z (combined) |
|---|---|---|---|---|---|---|---|
| VNN-BACKAN | 25 | 5 | 18880.6 | 0.000991 | 0.000927 | 2.32% | 2.84 |
| VNN-LAICHAU | 29 | 5 | 19050.8 | 0.000862 | 0.000825 | 2.46% | 1.76 |
| VNN-CAOBANG | 50 | 7 | 20038.3 | 0.000825 | 0.000793 | 2.51% | 1.57 |
| GD-RUYUANYAOZUZIZHIXIAN | 54 | 7 | 42312.6 | 0.00101 | 0.000972 | 2.27% | 1.54 |
| VNN-LAOCAI | 54 | 9 | 20120.8 | 0.000884 | 0.000869 | 2.40% | 0.67 |
| VNN-DIENBIEN | 55 | 8 | 20311.2 | 0.000956 | 0.000935 | 2.31% | 0.94 |
| VNN-LANGSON | 59 | 9 | 19611.8 | 0.00106 | 0.00103 | 2.21% | 1.28 |
| VNN-TUYENQUANG | 60 | 9 | 19831 | 0.000846 | 0.000831 | 2.45% | 0.68 |
| VNN-HAGIANG | 68 | 10 | 20566.6 | 0.000849 | 0.000816 | 2.47% | 1.53 |
| VNN-HANAM | 69 | 11 | 20013.6 | 0.000886 | 0.00087 | 2.40% | 0.71 |
| VNN-NINHBINH | 71 | 11 | 20565.3 | 0.000902 | 0.000914 | 2.34% | -0.54 |
| VNN-YENBAI | 75 | 10 | 19918.4 | 0.00102 | 0.001 | 2.23% | 0.86 |
| VNN-HUNGYEN | 76 | 12 | 20088.7 | 0.000985 | 0.00102 | 2.21% | -1.60 |
| VNN-QUANGNINH | 76 | 10 | 20654.1 | 0.000967 | 0.000949 | 2.29% | 0.80 |
| VNN-VINHPHUC | 83 | 13 | 20229.3 | 0.000842 | 0.000857 | 2.41% | -0.69 |
| VNN-HOABINH | 91 | 13 | 19815.2 | 0.00094 | 0.000986 | 2.25% | -1.97 |
| GD-XINFENGXIAN | 93 | 16 | 41157.6 | 0.000814 | 0.000808 | 2.49% | 0.26 |
| VNN-BACNINH | 93 | 13 | 19893.3 | 0.000896 | 0.000868 | 2.40% | 1.29 |
| VNN-BACGIANG | 95 | 14 | 20083.1 | 0.00102 | 0.00102 | 2.21% | -0.08 |
| VNN-THAINGUYEN | 102 | 14 | 20375.5 | 0.000934 | 0.000935 | 2.31% | -0.05 |
