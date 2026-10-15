# RARECERT plan level: certified overflow of the region hubs

66 hubs (results/plans_Region, gate SAA), gamma in 0.001, 1e-05 (128 hub-gamma rows ok, 4 failed). Capacity: Laplace-type proxy log S(c) = log sum_r exp(-E_r(c)) (a proxy, not a bound), secant on [Q, 3Q] within 6 evaluations. Hub SMIS: N = 100,000 draws at that c, eta_hub = 1e-4. Route baselines N_r = 20,000. Wall time 139.7 min.

P_region = 1 - prod over the hubs of the region of (1 - P_h), with P_h the hub SMIS estimate at its own c (exact in the model: hubs have separate factors). Buffer b = c/Q - 1.

## Region totals and buffers

| gamma | region | hubs | P_region (SMIS) | median buffer | q90 buffer | median P_h | capacity status (bisect / capped / below Q) |
|---|---|---|---|---|---|---|---|
| 0.001 | VNN | 24 | 0.00145 | 0.5301 | 0.5679 | 5.7e-05 | 0 / 0 / 0 |
| 0.001 | GD | 40 | 0.00126 | 0.6846 | 0.7372 | 2.97e-05 | 0 / 0 / 0 |
| 0.001 | all | 64 | 0.0027 | 0.6433 | 0.7205 | 3.58e-05 | 0 / 0 / 0 |
| 1e-05 | VNN | 24 | 1.38e-05 | 0.8150 | 0.8530 | 5.58e-07 | 0 / 0 / 0 |
| 1e-05 | GD | 40 | 1.48e-05 | 0.9675 | 1.0204 | 3.59e-07 | 0 / 0 / 0 |
| 1e-05 | all | 64 | 2.86e-05 | 0.9289 | 1.0079 | 4.1e-07 | 0 / 0 / 0 |

## Baselines and bounds

| gamma | bounds hold (hubs) | P_h / max p_r (median) | P_h / sum p_r (median) | indep / P_h (median) | indep / P_h (q90) | multi-route failure share (median) |
|---|---|---|---|---|---|---|
| 0.001 | 64 / 64 | 3.66 | 0.455 | 2.2 | 3.17 | 0.371 |
| 1e-05 | 64 / 64 | 3.69 | 0.573 | 1.75 | 2.27 | 0.273 |

Bounds are checked with slack 4 (rse_hub + max rse_r) on both sides. The capacity proxy is not a bound, so P_h at the chosen c can sit above or below gamma; the ratio is reported below.

| gamma | median P_h / gamma | q90 P_h / gamma | min P_h / gamma |
|---|---|---|---|
| 0.001 | 0.0358 | 0.0646 | 0.0222 |
| 1e-05 | 0.041 | 0.0612 | 0.0291 |

## Crude Monte Carlo check (gamma = 0.001, 2,000,000 draws, 20 smallest hubs)

| hub | customers | routes | c | P_hub SMIS | P_crude | rse crude | z (combined) |
|---|---|---|---|---|---|---|---|
| VNN-BACKAN | 25 | 5 | 21142 | 9.31e-05 | 8.35e-05 | 7.74% | 1.48 |
| VNN-LAICHAU | 29 | 5 | 21253.4 | 8.41e-05 | 7.65e-05 | 8.08% | 1.22 |
| VNN-CAOBANG | 50 | 7 | 22419.2 | 7.38e-05 | 6.6e-05 | 8.70% | 1.35 |
| GD-RUYUANYAOZUZIZHIXIAN | 54 | 7 | 48945.7 | 5.04e-05 | 5.95e-05 | 9.17% | -1.67 |
| VNN-LAOCAI | 54 | 9 | 22675.9 | 6.96e-05 | 7.75e-05 | 8.03% | -1.26 |
| VNN-DIENBIEN | 55 | 8 | 22960.4 | 7.21e-05 | 6.75e-05 | 8.61% | 0.80 |
| VNN-TUYENQUANG | 60 | 9 | 22127.6 | 7.81e-05 | 7e-05 | 8.45% | 1.37 |
| VNN-HAGIANG | 68 | 10 | 23149 | 6.48e-05 | 6.3e-05 | 8.91% | 0.33 |
| VNN-HANAM | 69 | 11 | 22596.4 | 6.35e-05 | 6.1e-05 | 9.05% | 0.44 |
| VNN-NINHBINH | 71 | 11 | 23329.3 | 5.92e-05 | 6.35e-05 | 8.87% | -0.77 |
| VNN-YENBAI | 75 | 10 | 22884 | 5.35e-05 | 4.95e-05 | 10.05% | 0.81 |
| VNN-HUNGYEN | 76 | 12 | 22760.6 | 6.42e-05 | 5.9e-05 | 9.21% | 0.95 |
| VNN-QUANGNINH | 76 | 10 | 23475.9 | 6.26e-05 | 6.75e-05 | 8.61% | -0.85 |
| VNN-VINHPHUC | 83 | 13 | 22834.2 | 5.81e-05 | 6.1e-05 | 9.05% | -0.53 |
| VNN-HOABINH | 91 | 13 | 22600.6 | 5.42e-05 | 5.25e-05 | 9.76% | 0.34 |
| GD-XINFENGXIAN | 93 | 16 | 47087.2 | 4.9e-05 | 5.4e-05 | 9.62% | -0.96 |
| VNN-BACNINH | 93 | 13 | 22553.5 | 5.6e-05 | 5.35e-05 | 9.67% | 0.47 |
| VNN-BACGIANG | 95 | 14 | 22941.6 | 5.38e-05 | 4.95e-05 | 10.05% | 0.86 |
| VNN-THAINGUYEN | 102 | 14 | 23188.4 | 5.46e-05 | 4.95e-05 | 10.05% | 1.02 |

## Failures

- VNN-HANOI-2 gamma=0.001: RuntimeError: no SMIS candidate reaches c=45000.0 (also with arcs): list index out of range
- VNN-HANOI-2 gamma=1e-05: RuntimeError: no SMIS candidate reaches c=45000.0 (also with arcs): list index out of range
- VNN-LANGSON gamma=0.001: RuntimeError: no SMIS candidate reaches c=45000.0 (also with arcs): list index out of range
- VNN-LANGSON gamma=1e-05: RuntimeError: no SMIS candidate reaches c=45000.0 (also with arcs): list index out of range
