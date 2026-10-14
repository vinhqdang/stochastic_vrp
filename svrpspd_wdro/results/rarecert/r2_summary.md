# RARECERT R2: mode-decomposed IS on the route-day simulator

60 routes (QUASAR sample, results/quasar/rare_routes.csv), gamma in 0.001, 0.0001, 1e-05: 180 route-gamma rows, 0 failed. MDIS, generic CE mixture and single-shift CE with N = 1,000,000 draws; crude MC with N = 10,000,000 at gamma = 1e-3 (reference). Wall time 161.5 min.

Class = share of the MDIS mixture mass on load modes: load only (>= 0.9), time only (<= 0.1), mixed. Errors are |estimate / reference - 1|; 'vs crude' uses crude MC (gamma = 1e-3 only); 'MDIS vs CE' and 'MDIS vs single' are disagreements (all gammas). Costs are sample counts at 95 per cent: MDIS = 1.96^2 RV / eps^2; CE = pilot draws + 1.96^2 RV / eps^2. Threshold bisection cost is common to all methods and not counted.

## Per gamma and class (median / q90)

| gamma | class | n | err MDIS vs crude | err CE vs crude | err single vs crude | MDIS vs CE | MDIS vs single | RV MDIS med | RV CE med | RV single med | modes MDIS med | modes CE med | build s MDIS med | build s CE med | share fail load med |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.001 | all | 60 | 0.76% / 1.77% | 0.89% / 1.75% | 2.18% / 21.85% | 0.20% / 0.47% | 1.98% / 28.37% | 4.33 | 4.41 | 742 | 8 | 9 | 3.13 | 43.05 | 0.509 |
| 0.001 | load only | 0 | | | | | | | | | | | | | | |
| 0.001 | time only | 0 | | | | | | | | | | | | | | |
| 0.001 | mixed | 60 | 0.76% / 1.77% | 0.89% / 1.75% | 2.18% / 21.85% | 0.20% / 0.47% | 1.98% / 28.37% | 4.33 | 4.41 | 742 | 8 | 9 | 3.13 | 43.05 | 0.509 |
| 0.0001 | all | 60 | n/a / n/a | n/a / n/a | n/a / n/a | 0.24% / 0.53% | 3.42% / 63.27% | 5.25 | 5.24 | 1.43e+03 | 7 | 8 | 3.52 | 49.44 | 0.502 |
| 0.0001 | load only | 0 | | | | | | | | | | | | | | |
| 0.0001 | time only | 0 | | | | | | | | | | | | | | |
| 0.0001 | mixed | 60 | n/a / n/a | n/a / n/a | n/a / n/a | 0.24% / 0.53% | 3.42% / 63.27% | 5.25 | 5.24 | 1.43e+03 | 7 | 8 | 3.52 | 49.44 | 0.502 |
| 1e-05 | all | 60 | n/a / n/a | n/a / n/a | n/a / n/a | 0.23% / 0.59% | 9.25% / 97.59% | 6.12 | 6.04 | 357 | 6.5 | 7 | 3.81 | 55.12 | 0.501 |
| 1e-05 | load only | 0 | | | | | | | | | | | | | | |
| 1e-05 | time only | 0 | | | | | | | | | | | | | | |
| 1e-05 | mixed | 60 | n/a / n/a | n/a / n/a | n/a / n/a | 0.23% / 0.59% | 9.25% / 97.59% | 6.12 | 6.04 | 357 | 6.5 | 7 | 3.81 | 55.12 | 0.501 |

## Costs and thresholds per gamma

Totals are sums over the routes of the gamma (sample counts). CE / MDIS is the ratio of the totals.

| gamma | eps_rel | rows | MDIS total | CE total | CE / MDIS | share of failure on load modes (med) | p_MDIS / gamma med | threshold s med | unreachable modes med | not converged | slack expanded |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.001 | 0.1 | 60 | 1.021e+05 | 1.853e+08 | 1.82e+03 | 0.509 | 0.906 | 75.5 | 16.5 | 0 | 1 |
| 0.001 | 0.01 | 60 | 1.021e+07 | 1.956e+08 | 19.2 | 0.509 | 0.906 | 75.5 | 16.5 | 0 | 1 |
| 0.0001 | 0.1 | 60 | 1.264e+05 | 2.314e+08 | 1.83e+03 | 0.502 | 0.947 | 82.5 | 19 | 0 | 1 |
| 0.0001 | 0.01 | 60 | 1.264e+07 | 2.436e+08 | 19.3 | 0.502 | 0.947 | 82.5 | 19 | 0 | 1 |
| 1e-05 | 0.1 | 60 | 1.539e+05 | 2.695e+08 | 1.75e+03 | 0.501 | 0.958 | 92.3 | 21 | 0 | 3 |
| 1e-05 | 0.01 | 60 | 1.539e+07 | 2.837e+08 | 18.4 | 0.501 | 0.958 | 92.3 | 21 | 0 | 3 |

Single-shift mode chosen (count over rows): 0.001: late_7 8; load_7 7; late_6 7; late_8 7; late_9 6; load_6 5; late_10 4; load_5 3; late_4 3; load_11 2; load_9 1; load_13 1; load_8 1; load_4 1; late_14 1; late_5 1; late_12 1; late_11 1, 0.0001: late_7 11; late_9 7; late_6 7; load_6 5; load_7 4; late_4 4; late_10 4; late_11 3; load_8 3; late_8 3; late_5 2; load_5 2; load_0 2; late_13 1; load_14 1; late_12 1, 1e-05: late_7 10; late_6 9; late_8 6; load_7 4; late_4 4; late_9 4; late_10 4; load_5 3; load_6 3; load_0 3; load_9 2; late_11 2; load_11 1; load_13 1; load_14 1; load_8 1; late_5 1; late_12 1.
