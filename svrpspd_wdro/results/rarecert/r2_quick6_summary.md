# RARECERT R2: mode-decomposed IS on the route-day simulator

6 routes (QUASAR sample, results/quasar/rare_routes.csv), gamma in 0.001, 0.0001, 1e-05: 18 route-gamma rows, 0 failed. MDIS, generic CE mixture and single-shift CE with N = 1,000,000 draws; crude MC with N = 10,000,000 at gamma = 1e-3 (reference). Wall time 31.5 min.

Class = share of the MDIS mixture mass on load modes: load only (>= 0.9), time only (<= 0.1), mixed. Errors are |estimate / reference - 1|; 'vs crude' uses crude MC (gamma = 1e-3 only); 'MDIS vs CE' and 'MDIS vs single' are disagreements (all gammas). Costs are sample counts at 95 per cent: MDIS = 1.96^2 RV / eps^2; CE = pilot draws + 1.96^2 RV / eps^2. Threshold bisection cost is common to all methods and not counted.

## Per gamma and class (median / q90)

| gamma | class | n | err MDIS vs crude | err CE vs crude | err single vs crude | MDIS vs CE | MDIS vs single | RV MDIS med | RV CE med | RV single med | modes MDIS med | modes CE med | build s MDIS med | build s CE med | share fail load med |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.001 | all | 6 | 0.59% / 0.97% | 0.61% / 1.30% | 3.66% / 20.14% | 0.18% / 0.52% | 4.09% / 27.83% | 4.37 | 4.43 | 1.12e+03 | 9 | 9.5 | 6.03 | 57.25 | 0.515 |
| 0.001 | load only | 0 | | | | | | | | | | | | | | |
| 0.001 | time only | 0 | | | | | | | | | | | | | | |
| 0.001 | mixed | 6 | 0.59% / 0.97% | 0.61% / 1.30% | 3.66% / 20.14% | 0.18% / 0.52% | 4.09% / 27.83% | 4.37 | 4.43 | 1.12e+03 | 9 | 9.5 | 6.03 | 57.25 | 0.515 |
| 0.0001 | all | 6 | n/a / n/a | n/a / n/a | n/a / n/a | 0.22% / 0.68% | 10.44% / 58.76% | 5.45 | 5.26 | 360 | 7 | 7.5 | 7.31 | 69.50 | 0.503 |
| 0.0001 | load only | 0 | | | | | | | | | | | | | | |
| 0.0001 | time only | 0 | | | | | | | | | | | | | | |
| 0.0001 | mixed | 6 | n/a / n/a | n/a / n/a | n/a / n/a | 0.22% / 0.68% | 10.44% / 58.76% | 5.45 | 5.26 | 360 | 7 | 7.5 | 7.31 | 69.50 | 0.503 |
| 1e-05 | all | 6 | n/a / n/a | n/a / n/a | n/a / n/a | 0.14% / 0.40% | 20.52% / 72.53% | 6.44 | 6.15 | 1.9e+03 | 6.5 | 7 | 7.78 | 63.31 | 0.492 |
| 1e-05 | load only | 0 | | | | | | | | | | | | | | |
| 1e-05 | time only | 0 | | | | | | | | | | | | | | |
| 1e-05 | mixed | 6 | n/a / n/a | n/a / n/a | n/a / n/a | 0.14% / 0.40% | 20.52% / 72.53% | 6.44 | 6.15 | 1.9e+03 | 6.5 | 7 | 7.78 | 63.31 | 0.492 |

## Costs and thresholds per gamma

Totals are sums over the routes of the gamma (sample counts). CE / MDIS is the ratio of the totals.

| gamma | eps_rel | rows | MDIS total | CE total | CE / MDIS | share of failure on load modes (med) | p_MDIS / gamma med | threshold s med | unreachable modes med | not converged | slack expanded |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.001 | 0.1 | 6 | 1.042e+04 | 1.719e+07 | 1.65e+03 | 0.515 | 0.925 | 99.3 | 17 | 0 | 1 |
| 0.001 | 0.01 | 6 | 1.042e+06 | 1.823e+07 | 17.5 | 0.515 | 0.925 | 99.3 | 17 | 0 | 1 |
| 0.0001 | 0.1 | 6 | 1.323e+04 | 2.081e+07 | 1.57e+03 | 0.503 | 0.937 | 134 | 20.5 | 0 | 1 |
| 0.0001 | 0.01 | 6 | 1.323e+06 | 2.204e+07 | 16.7 | 0.503 | 0.937 | 134 | 20.5 | 0 | 1 |
| 1e-05 | 0.1 | 6 | 1.545e+04 | 2.648e+07 | 1.71e+03 | 0.492 | 0.961 | 171 | 21 | 0 | 1 |
| 1e-05 | 0.01 | 6 | 1.545e+06 | 2.8e+07 | 18.1 | 0.492 | 0.961 | 171 | 21 | 0 | 1 |

Single-shift mode chosen (count over rows): 0.001: late_7 2; load_11 1; load_6 1; late_10 1; late_6 1, 0.0001: late_7 2; late_11 1; load_6 1; late_10 1; late_6 1, 1e-05: late_7 2; late_11 1; load_6 1; late_10 1; late_6 1.
